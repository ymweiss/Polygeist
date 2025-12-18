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

  // Second pass: insert wrappers and replace calls
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

  // Build parameter list: kernel args first, then grid/block dims
  os << "\n// Generated wrapper for kernel argument order preservation\n";
  os << "inline void " << wrapperName << "(";

  bool first = true;
  for (const auto &arg : kernel.arguments) {
    if (!first)
      os << ", ";
    first = false;

    // Get type string from the original declaration
    PrintingPolicy policy(ctx.getLangOpts());
    os << arg.type.getAsString(policy) << " " << arg.name;
  }

  // Add grid and block parameters
  if (!kernel.arguments.empty())
    os << ", ";
  os << "dim3 __grid, dim3 __block";

  os << ") {\n";

  // Generate the launch call inside the wrapper
  os << "    hipLaunchKernelGGL(" << kernel.demangledName
     << ", __grid, __block, 0, 0";

  for (const auto &arg : kernel.arguments) {
    os << ", " << arg.name;
  }

  os << ");\n";
  os << "}\n\n";

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

} // namespace vortex
