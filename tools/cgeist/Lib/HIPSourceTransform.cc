//===- HIPSourceTransform.cc - HIP source-to-source transform -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HIPSourceTransform.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include <sstream>
#include <algorithm>
#include <cctype>

using namespace clang;

namespace vortex {

HIPSourceTransformer::HIPSourceTransformer(ASTContext &ctx,
                                           SourceManager &SM,
                                           const LangOptions &LangOpts)
    : ctx(ctx), SM(SM), rewriter(SM, LangOpts) {}

bool HIPSourceTransformer::transform(TranslationUnitDecl *TU) {
  // First pass: collect kernels and launch sites
  TraverseDecl(TU);

  if (kernels.empty()) {
    llvm::errs() << "[HIPSourceTransform] No kernels found\n";
    return false;
  }

  if (launchSites.empty()) {
    llvm::errs() << "[HIPSourceTransform] No kernel launch sites found\n";
    return false;
  }

  llvm::outs() << "[HIPSourceTransform] Found " << kernels.size()
               << " kernel(s) and " << launchSites.size() << " launch site(s)\n";

  // First: For kernels that use blockIdx without threadIdx, replace blockIdx→threadIdx
  // and swap grid/block dimensions in launch calls
  for (const auto &kernel : kernels) {
    if (kernel.needsBlockIdxConversion()) {
      replaceBlockIdxWithThreadIdx(kernel);
      kernelsNeedingConversion[kernel.demangledName] = true;
    }
  }

  // Second pass: insert wrappers, stub includes, and replace calls
  // Insert wrappers for each unique kernel that has launch sites
  for (const auto &site : launchSites) {
    if (wrappersGenerated.find(site.kernelName) == wrappersGenerated.end()) {
      // Find the kernel info
      const KernelInfo *kernelInfo = nullptr;
      for (const auto &k : kernels) {
        if (k.demangledName == site.kernelName) {
          kernelInfo = &k;
          break;
        }
      }

      if (kernelInfo) {
        insertWrapper(*kernelInfo);
        insertStubInclude(*kernelInfo);  // Add include for generated stub
        wrappersGenerated[site.kernelName] = true;
      }
    }
  }

  // Replace all launch calls with wrapper calls
  for (const auto &site : launchSites) {
    replaceLaunchWithWrapper(site);
  }

  return true;
}

bool HIPSourceTransformer::VisitFunctionDecl(FunctionDecl *FD) {
  if (!FD->hasBody())
    return true;

  if (isKernelFunction(FD)) {
    KernelInfo info = analyzeKernel(FD);
    kernels.push_back(std::move(info));
    llvm::outs() << "[HIPSourceTransform] Found kernel: " << FD->getNameAsString()
                 << " with " << FD->getNumParams() << " params\n";
  }

  return true;
}

bool HIPSourceTransformer::VisitCUDAKernelCallExpr(CUDAKernelCallExpr *CE) {
  const FunctionDecl *callee = CE->getDirectCallee();
  if (!callee)
    return true;

  if (!isKernelFunction(callee))
    return true;

  LaunchSiteInfo site;
  site.callExpr = CE;
  site.kernel = callee;
  site.kernelName = callee->getNameAsString();
  site.sourceRange = CE->getSourceRange();

  // Get grid and block expressions from the config
  if (auto *config = CE->getConfig()) {
    if (config->getNumArgs() >= 2) {
      site.gridExpr = config->getArg(0);
      site.blockExpr = config->getArg(1);
    }
  }

  launchSites.push_back(site);
  llvm::outs() << "[HIPSourceTransform] Found launch site for: "
               << site.kernelName << "\n";

  return true;
}

bool HIPSourceTransformer::isKernelFunction(const FunctionDecl *FD) const {
  return FD->hasAttr<CUDAGlobalAttr>();
}

KernelInfo HIPSourceTransformer::analyzeKernel(const FunctionDecl *FD) {
  KernelInfo info;
  info.decl = FD;
  info.demangledName = FD->getNameAsString();

  for (const auto *param : FD->parameters()) {
    KernelArgInfo argInfo;
    argInfo.name = param->getNameAsString();
    argInfo.type = param->getType();
    argInfo.isPointer = param->getType()->isPointerType() ||
                        param->getType()->isReferenceType() ||
                        param->getType()->isArrayType();

    CharUnits size = ctx.getTypeSizeInChars(param->getType());
    CharUnits align = ctx.getTypeAlignInChars(param->getType());
    argInfo.sizeBytes = static_cast<unsigned>(size.getQuantity());
    argInfo.alignBytes = static_cast<unsigned>(align.getQuantity());

    info.arguments.push_back(argInfo);
  }

  // Detect blockIdx/threadIdx usage in kernel body
  if (FD->hasBody()) {
    BlockThreadIdxVisitor visitor;
    visitor.TraverseStmt(const_cast<Stmt*>(FD->getBody()));
    info.usesBlockIdx = visitor.usesBlockIdx;
    info.usesThreadIdx = visitor.usesThreadIdx;
    info.usesBlockDim = visitor.usesBlockDim;
    info.usesGridDim = visitor.usesGridDim;

    if (info.needsBlockIdxConversion()) {
      llvm::outs() << "[HIPSourceTransform] Kernel " << info.demangledName
                   << " uses blockIdx but not threadIdx - will convert\n";
    }
  }

  return info;
}

std::string HIPSourceTransformer::getParamTypeString(const ParmVarDecl *PVD) const {
  // Get the type as a string, preserving qualifiers
  PrintingPolicy policy(ctx.getLangOpts());
  policy.SuppressTagKeyword = false;
  return PVD->getType().getAsString(policy);
}

std::string HIPSourceTransformer::generateWrapperFunction(const KernelInfo &kernel) {
  std::ostringstream os;

  std::string wrapperName = getWrapperName(kernel.demangledName);
  std::string stubName = "launch_" + kernel.demangledName;

  // Check if this kernel needs grid/block swap (blockIdx→threadIdx conversion)
  bool needsSwap = kernel.needsBlockIdxConversion();

  // Build parameter list string: kernel args first, then grid/block dims
  std::ostringstream paramList;
  std::ostringstream argNames;

  bool first = true;
  for (const auto &arg : kernel.arguments) {
    if (!first) {
      paramList << ", ";
      argNames << ", ";
    }
    first = false;

    // Get type string from the original declaration
    PrintingPolicy policy(ctx.getLangOpts());
    paramList << arg.type.getAsString(policy) << " " << arg.name;
    argNames << arg.name;
  }

  // Add grid and block parameters
  if (!kernel.arguments.empty())
    paramList << ", ";
  paramList << "dim3 __grid, dim3 __block";

  std::string params = paramList.str();
  std::string args = argNames.str();

  // Determine grid/block args for the launch (swap if needed)
  std::string gridArg = needsSwap ? "__block" : "__grid";
  std::string blockArg = needsSwap ? "__grid" : "__block";

  // Generate conditional wrapper for host vs device compilation
  os << "\n// Generated wrapper for kernel argument order preservation\n";
  if (needsSwap) {
    os << "// NOTE: blockIdx→threadIdx conversion - grid/block dimensions swapped\n";
  }
  os << "// On host: calls generated stub for proper argument marshaling\n";
  os << "// On device: uses <<<>>> syntax for Polygeist processing\n";
  os << "#ifdef HIP_HOST_COMPILATION\n";

  // HOST version: call the generated stub function which uses vortexLaunchKernel
  os << "__attribute__((noinline)) void " << wrapperName << "(" << params << ") {\n";
  os << "    " << stubName << "(" << gridArg << ", " << blockArg << ", " << args << ");\n";
  os << "}\n";

  os << "#else\n";

  // DEVICE version: use kernel launch syntax for Polygeist/MLIR processing
  // Use swapped grid/block if this kernel had blockIdx→threadIdx conversion
  os << "__attribute__((noinline)) void " << wrapperName << "(" << params << ") {\n";
  os << "    " << kernel.demangledName << "<<<" << gridArg << ", " << blockArg << ">>>(" << args << ");\n";
  os << "}\n";

  os << "#endif\n\n";

  return os.str();
}

/// Parse hipLaunchKernelGGL arguments from original source text
/// Format: hipLaunchKernelGGL(kernel, grid, block, sharedMem, stream, args...)
std::vector<std::string> HIPSourceTransformer::parseHipLaunchArgs(const std::string &text) const {
  std::vector<std::string> args;

  // Find the opening paren after hipLaunchKernelGGL
  size_t start = text.find('(');
  if (start == std::string::npos)
    return args;

  start++; // Skip the '('

  // Parse comma-separated arguments, handling nested parens
  int parenDepth = 0;
  size_t argStart = start;

  for (size_t i = start; i < text.size(); ++i) {
    char c = text[i];
    if (c == '(') {
      parenDepth++;
    } else if (c == ')') {
      if (parenDepth == 0) {
        // End of arguments
        std::string arg = text.substr(argStart, i - argStart);
        // Trim whitespace
        size_t first = arg.find_first_not_of(" \t\n\r");
        size_t last = arg.find_last_not_of(" \t\n\r");
        if (first != std::string::npos && last != std::string::npos)
          args.push_back(arg.substr(first, last - first + 1));
        break;
      }
      parenDepth--;
    } else if (c == ',' && parenDepth == 0) {
      std::string arg = text.substr(argStart, i - argStart);
      // Trim whitespace
      size_t first = arg.find_first_not_of(" \t\n\r");
      size_t last = arg.find_last_not_of(" \t\n\r");
      if (first != std::string::npos && last != std::string::npos)
        args.push_back(arg.substr(first, last - first + 1));
      argStart = i + 1;
    }
  }

  return args;
}

std::string HIPSourceTransformer::generateWrapperCall(const LaunchSiteInfo &site) {
  std::ostringstream os;

  std::string wrapperName = getWrapperName(site.kernelName);

  os << wrapperName << "(";

  // Get the original hipLaunchKernelGGL call text to extract grid/block args
  SourceRange range = site.callExpr->getSourceRange();
  SourceLocation beginLoc = range.getBegin();
  SourceLocation endLoc = range.getEnd();

  // Get expansion location for macro
  if (beginLoc.isMacroID()) {
    CharSourceRange expansionRange = SM.getExpansionRange(beginLoc);
    beginLoc = expansionRange.getBegin();
    CharSourceRange endExpansionRange = SM.getExpansionRange(endLoc);
    endLoc = endExpansionRange.getEnd();
  }
  endLoc = Lexer::getLocForEndOfToken(endLoc, 0, SM, ctx.getLangOpts());

  std::string originalText = getSourceText(SourceRange(beginLoc, endLoc));
  std::vector<std::string> launchArgs = parseHipLaunchArgs(originalText);

  // hipLaunchKernelGGL args: kernel, grid, block, sharedMem, stream, kernelArgs...
  std::string gridArg, blockArg;
  std::vector<std::string> kernelArgs;

  if (launchArgs.size() >= 6) {
    gridArg = launchArgs[1];   // grid
    blockArg = launchArgs[2];  // block
    // Kernel args start at index 5
    for (size_t i = 5; i < launchArgs.size(); ++i) {
      kernelArgs.push_back(launchArgs[i]);
    }
  }

  // Add kernel arguments
  bool first = true;
  for (const auto &arg : kernelArgs) {
    if (!first)
      os << ", ";
    first = false;
    os << arg;
  }

  // Add grid and block arguments
  if (!gridArg.empty() && !blockArg.empty()) {
    if (!first)
      os << ", ";
    os << gridArg << ", " << blockArg;
  }

  os << ")";

  return os.str();
}

std::string HIPSourceTransformer::getSourceText(const Expr *E) const {
  if (!E)
    return "";

  SourceRange range = E->getSourceRange();

  // For macro-expanded expressions, get the spelling location
  SourceLocation begin = range.getBegin();
  SourceLocation end = range.getEnd();

  if (begin.isMacroID()) {
    // Get spelling location (where the token was actually written)
    begin = SM.getSpellingLoc(begin);
  }
  if (end.isMacroID()) {
    end = SM.getSpellingLoc(end);
  }

  return getSourceText(SourceRange(begin, end));
}

std::string HIPSourceTransformer::getSourceText(SourceRange range) const {
  if (range.isInvalid())
    return "";

  // Get the end location that includes the last token
  SourceLocation endLoc = Lexer::getLocForEndOfToken(
      range.getEnd(), 0, SM, ctx.getLangOpts());

  bool invalid = false;
  const char *begin = SM.getCharacterData(range.getBegin(), &invalid);
  if (invalid)
    return "";

  const char *end = SM.getCharacterData(endLoc, &invalid);
  if (invalid)
    return "";

  return std::string(begin, end);
}

void HIPSourceTransformer::insertWrapper(const KernelInfo &kernel) {
  std::string wrapper = generateWrapperFunction(kernel);

  // Insert AFTER the kernel definition
  // The wrapper calls the kernel, so kernel must be declared first
  SourceLocation insertLoc = kernel.decl->getEndLoc();

  // Debug: check if location is in main file
  FileID mainFileID = SM.getMainFileID();
  FileID insertFileID = SM.getFileID(insertLoc);
  llvm::outs() << "[HIPSourceTransform] Insert location file ID: "
               << insertFileID.getHashValue() << ", main file ID: "
               << mainFileID.getHashValue() << "\n";

  // If the end location is a macro, get the expansion location
  if (insertLoc.isMacroID()) {
    CharSourceRange expansionRange = SM.getExpansionRange(insertLoc);
    insertLoc = expansionRange.getEnd();
    insertFileID = SM.getFileID(insertLoc);
    llvm::outs() << "[HIPSourceTransform] Expansion location file ID: "
                 << insertFileID.getHashValue() << "\n";
  }

  // Verify we're in the main file
  if (insertFileID != mainFileID) {
    llvm::errs() << "[HIPSourceTransform] Warning: Insert location not in main file, "
                 << "skipping wrapper for " << kernel.demangledName << "\n";
    return;
  }

  // Move to end of the closing brace token
  insertLoc = Lexer::getLocForEndOfToken(insertLoc, 0, SM, ctx.getLangOpts());

  // Insert after the kernel definition
  bool success = rewriter.InsertTextAfter(insertLoc, wrapper);
  llvm::outs() << "[HIPSourceTransform] InsertTextAfter returned: "
               << (success ? "success" : "failure") << "\n";
  llvm::outs() << "[HIPSourceTransform] Inserted wrapper for: "
               << kernel.demangledName << "\n";
}

void HIPSourceTransformer::replaceLaunchWithWrapper(const LaunchSiteInfo &site) {
  std::string wrapperCall = generateWrapperCall(site);

  // Get the full source range of the kernel call expression
  // For CUDAKernelCallExpr from hipLaunchKernelGGL macro, we need the expansion range
  SourceRange fullRange = site.callExpr->getSourceRange();

  // Debug: check file IDs
  FileID mainFileID = SM.getMainFileID();

  // Check if this is from a macro expansion (hipLaunchKernelGGL)
  SourceLocation beginLoc = fullRange.getBegin();
  SourceLocation endLoc = fullRange.getEnd();

  if (beginLoc.isMacroID()) {
    // Get the full expansion range - this gives us the hipLaunchKernelGGL(...) call
    CharSourceRange expansionRange = SM.getExpansionRange(beginLoc);
    beginLoc = expansionRange.getBegin();

    // For the end, we also need the expansion location
    CharSourceRange endExpansionRange = SM.getExpansionRange(endLoc);
    endLoc = endExpansionRange.getEnd();

    llvm::outs() << "[HIPSourceTransform] Macro expansion detected\n";
  }

  FileID beginFileID = SM.getFileID(beginLoc);
  llvm::outs() << "[HIPSourceTransform] Replace range begin file ID: "
               << beginFileID.getHashValue() << ", main file ID: "
               << mainFileID.getHashValue() << "\n";

  // Extend to include the closing paren
  endLoc = Lexer::getLocForEndOfToken(endLoc, 0, SM, ctx.getLangOpts());

  CharSourceRange charRange = CharSourceRange::getCharRange(beginLoc, endLoc);

  // Debug: print original text being replaced
  std::string originalText = getSourceText(SourceRange(beginLoc, endLoc));
  llvm::outs() << "[HIPSourceTransform] Original text: " << originalText << "\n";
  llvm::outs() << "[HIPSourceTransform] Replacement: " << wrapperCall << "\n";

  bool success = rewriter.ReplaceText(charRange, wrapperCall);
  llvm::outs() << "[HIPSourceTransform] ReplaceText returned: "
               << (success ? "failure" : "success") << "\n";  // Note: ReplaceText returns true on error
  llvm::outs() << "[HIPSourceTransform] Replaced launch with wrapper call for: "
               << site.kernelName << "\n";
}

void HIPSourceTransformer::replaceBlockIdxWithThreadIdx(const KernelInfo &kernel) {
  // Replace all occurrences of "blockIdx" with "threadIdx" in the kernel body
  // This is needed for kernels that only use blockIdx (unusual pattern)
  // The grid/block dimensions will be swapped in the wrapper call

  if (!kernel.decl->hasBody()) {
    llvm::errs() << "[HIPSourceTransform] Kernel " << kernel.demangledName
                 << " has no body - cannot replace blockIdx\n";
    return;
  }

  // Get the source range of the kernel body
  Stmt *body = const_cast<Stmt*>(kernel.decl->getBody());
  SourceRange bodyRange = body->getSourceRange();

  // Get the source text
  std::string bodyText = getSourceText(bodyRange);
  if (bodyText.empty()) {
    llvm::errs() << "[HIPSourceTransform] Failed to get kernel body text for "
                 << kernel.demangledName << "\n";
    return;
  }

  // Simple string replacement - replace "blockIdx" with "threadIdx"
  std::string newText = bodyText;
  size_t pos = 0;
  while ((pos = newText.find("blockIdx", pos)) != std::string::npos) {
    newText.replace(pos, 8, "threadIdx");
    pos += 9;  // Length of "threadIdx"
  }

  if (newText == bodyText) {
    llvm::outs() << "[HIPSourceTransform] No blockIdx found in kernel "
                 << kernel.demangledName << "\n";
    return;
  }

  // Replace the kernel body with the modified text
  SourceLocation begin = bodyRange.getBegin();
  SourceLocation end = Lexer::getLocForEndOfToken(bodyRange.getEnd(), 0, SM, ctx.getLangOpts());
  CharSourceRange charRange = CharSourceRange::getCharRange(begin, end);

  bool success = rewriter.ReplaceText(charRange, newText);
  if (success) {  // ReplaceText returns true on error
    llvm::errs() << "[HIPSourceTransform] Failed to replace blockIdx in kernel "
                 << kernel.demangledName << "\n";
  } else {
    llvm::outs() << "[HIPSourceTransform] Replaced blockIdx with threadIdx in kernel "
                 << kernel.demangledName << "\n";
  }
}

std::string HIPSourceTransformer::getTransformedSource() const {
  // Get the main file ID
  FileID mainFileID = SM.getMainFileID();
  llvm::outs() << "[HIPSourceTransform] getTransformedSource: main file ID = "
               << mainFileID.getHashValue() << "\n";

  // Get the rewrite buffer for the main file
  const RewriteBuffer *buffer = rewriter.getRewriteBufferFor(mainFileID);
  if (!buffer) {
    // No modifications were made, return original source
    llvm::outs() << "[HIPSourceTransform] No rewrite buffer for main file - returning original\n";
    bool invalid = false;
    llvm::StringRef source = SM.getBufferData(mainFileID, &invalid);
    if (invalid)
      return "";
    return source.str();
  }

  llvm::outs() << "[HIPSourceTransform] Got rewrite buffer, writing transformed source\n";
  std::string result;
  llvm::raw_string_ostream os(result);
  buffer->write(os);
  return result;
}

bool HIPSourceTransformer::writeToFile(llvm::StringRef filename) const {
  std::string transformed = getTransformedSource();
  if (transformed.empty())
    return false;

  std::error_code EC;
  llvm::raw_fd_ostream out(filename, EC, llvm::sys::fs::OF_Text);
  if (EC) {
    llvm::errs() << "[HIPSourceTransform] Error opening output file: "
                 << EC.message() << "\n";
    return false;
  }

  out << transformed;
  llvm::outs() << "[HIPSourceTransform] Written transformed source to: "
               << filename << "\n";
  return true;
}

bool HIPSourceTransformer::writeStubHeader(llvm::StringRef outputDir) const {
  // Generate stub headers for each kernel
  // This creates the _args.h with CORRECT argument order from AST
  for (const auto &kernel : kernels) {
    std::string stubName = kernel.demangledName + "_args.h";
    std::string stubPath;
    if (outputDir.empty()) {
      stubPath = stubName;
    } else {
      stubPath = (outputDir + "/" + stubName).str();
    }

    std::error_code EC;
    llvm::raw_fd_ostream out(stubPath, EC, llvm::sys::fs::OF_Text);
    if (EC) {
      llvm::errs() << "[HIPSourceTransform] Error creating stub file: "
                   << EC.message() << "\n";
      continue;
    }

    std::string guardName = kernel.demangledName + "_ARGS_H";
    std::transform(guardName.begin(), guardName.end(), guardName.begin(), ::toupper);

    out << "// Auto-generated host stub for " << kernel.demangledName << "\n";
    out << "// Generated by Polygeist HIPSourceTransform from HIP AST\n";
    out << "// Arguments in ORIGINAL order from HIP source (pointers first if applicable)\n";
    out << "#ifndef " << guardName << "\n";
    out << "#define " << guardName << "\n\n";
    out << "#include <stdint.h>\n\n";
    out << "#ifdef HIP_HOST_COMPILATION\n";
    out << "#ifndef HIP_VORTEX_RUNTIME_H\n";
    out << "#error \"" << stubName << " must be included after hip_vortex_runtime.h\"\n";
    out << "#endif\n\n";

    // Generate argument structure with ORIGINAL order from HIP source
    out << "// Argument structure for " << kernel.demangledName << "\n";
    out << "typedef struct __attribute__((packed)) {\n";

    unsigned offset = 0;
    for (size_t i = 0; i < kernel.arguments.size(); ++i) {
      const auto &arg = kernel.arguments[i];
      // Align offset
      unsigned alignedOffset = (offset + arg.alignBytes - 1) & ~(arg.alignBytes - 1);

      if (arg.isPointer) {
        out << "  void* " << arg.name << ";  // offset=" << alignedOffset
            << ", size=8, device pointer\n";
        offset = alignedOffset + 8;  // 64-bit pointers on host
      } else {
        PrintingPolicy policy(ctx.getLangOpts());
        std::string cType = arg.type.getAsString(policy);

        out << "  " << cType << " " << arg.name << ";  // offset=" << alignedOffset
            << ", size=" << arg.sizeBytes << "\n";
        offset = alignedOffset + arg.sizeBytes;
      }
    }

    out << "} " << kernel.demangledName << "_args_t;\n\n";

    // Generate metadata array
    out << "// Metadata array for vortexLaunchKernel\n";
    out << "static const VortexKernelArgMeta " << kernel.demangledName << "_metadata[] = {\n";

    offset = 0;
    for (size_t i = 0; i < kernel.arguments.size(); ++i) {
      const auto &arg = kernel.arguments[i];
      unsigned alignedOffset = (offset + arg.alignBytes - 1) & ~(arg.alignBytes - 1);
      unsigned size = arg.isPointer ? 8 : arg.sizeBytes;

      out << "  { .offset = " << alignedOffset
          << ", .size = " << size
          << ", .is_pointer = " << (arg.isPointer ? 1 : 0)
          << " },  // " << arg.name << "\n";

      offset = alignedOffset + size;
    }

    out << "};\n";
    out << "#define " << kernel.demangledName << "_NUM_ARGS " << kernel.arguments.size() << "\n\n";

    // Generate type-safe launcher
    out << "// Type-safe launcher (use this instead of hipLaunchKernelGGL)\n";
    out << "static inline hipError_t launch_" << kernel.demangledName << "(\n";
    out << "    dim3 gridDim, dim3 blockDim";

    for (const auto &arg : kernel.arguments) {
      out << ",\n    ";
      if (arg.isPointer) {
        out << "const void* " << arg.name;
      } else {
        PrintingPolicy policy(ctx.getLangOpts());
        out << arg.type.getAsString(policy) << " " << arg.name;
      }
    }

    out << ") {\n\n";
    out << "  " << kernel.demangledName << "_args_t args;\n";

    for (const auto &arg : kernel.arguments) {
      if (arg.isPointer) {
        out << "  args." << arg.name << " = (void*)" << arg.name << ";\n";
      } else {
        out << "  args." << arg.name << " = " << arg.name << ";\n";
      }
    }

    out << "\n  return vortexLaunchKernel(\n";
    out << "    \"" << kernel.demangledName << "\",\n";
    out << "    gridDim, blockDim,\n";
    out << "    &args, sizeof(args),\n";
    out << "    " << kernel.demangledName << "_metadata, " << kernel.demangledName << "_NUM_ARGS);\n";
    out << "}\n\n";

    out << "#endif // HIP_HOST_COMPILATION\n";
    out << "#endif // " << guardName << "\n";

    llvm::outs() << "[HIPSourceTransform] Written stub header: " << stubPath << "\n";
  }

  return !kernels.empty();
}

void HIPSourceTransformer::insertStubInclude(const KernelInfo &kernel) {
  // Insert #include for the stub header after the last #include
  // We need to find a good location - after the last #include in the file

  // Get the beginning of the main file
  FileID mainFileID = SM.getMainFileID();
  SourceLocation fileStart = SM.getLocForStartOfFile(mainFileID);

  // Find the last #include directive by scanning the file
  // For simplicity, we'll insert right after the first line that doesn't start with #
  bool invalid = false;
  llvm::StringRef fileContent = SM.getBufferData(mainFileID, &invalid);
  if (invalid)
    return;

  // Find a good insertion point - after the last #include
  size_t lastIncludeEnd = 0;
  size_t pos = 0;
  while (pos < fileContent.size()) {
    // Find start of line
    size_t lineStart = pos;
    // Find end of line
    size_t lineEnd = fileContent.find('\n', pos);
    if (lineEnd == llvm::StringRef::npos)
      lineEnd = fileContent.size();

    llvm::StringRef line = fileContent.substr(lineStart, lineEnd - lineStart).trim();
    if (line.starts_with("#include")) {
      lastIncludeEnd = lineEnd + 1;  // After the newline
    }

    pos = lineEnd + 1;
    if (pos > fileContent.size())
      break;
  }

  if (lastIncludeEnd > 0 && lastIncludeEnd < fileContent.size()) {
    SourceLocation insertLoc = fileStart.getLocWithOffset(lastIncludeEnd);

    std::string includeDirective = "\n// Auto-generated stub for host compilation\n"
                                   "#ifdef HIP_HOST_COMPILATION\n"
                                   "#include \"" + kernel.demangledName + "_args.h\"\n"
                                   "#endif\n";

    rewriter.InsertTextAfter(insertLoc, includeDirective);
    llvm::outs() << "[HIPSourceTransform] Inserted stub include for: "
                 << kernel.demangledName << "\n";
  }
}

} // namespace vortex
