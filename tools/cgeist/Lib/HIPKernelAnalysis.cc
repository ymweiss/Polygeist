//===- HIPKernelAnalysis.cc - AST-based HIP kernel analysis ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HIPKernelAnalysis.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Type.h"
#include "clang/Basic/LLVM.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace vortex;

HIPKernelCollector::HIPKernelCollector(ASTContext &ctx,
                                       CodeGen::CodeGenModule &CGM)
    : ctx(ctx), CGM(CGM) {}

bool HIPKernelCollector::VisitFunctionDecl(FunctionDecl *FD) {
  // Skip function declarations without bodies (we want definitions)
  if (!FD->hasBody())
    return true;

  // Skip template declarations (wait for instantiations)
  if (FD->getTemplatedKind() == FunctionDecl::TK_FunctionTemplate)
    return true;

  // Skip constructors and destructors - they need special GlobalDecl handling
  if (isa<CXXConstructorDecl>(FD) || isa<CXXDestructorDecl>(FD))
    return true;

  DeviceFunctionKind kind = classifyFunction(FD);

  // Track all device-visible functions
  if (kind != DeviceFunctionKind::HostOnly) {
    DeviceFunctionInfo devInfo;
    devInfo.decl = FD;
    devInfo.demangledName = FD->getNameAsString();
    devInfo.kind = kind;

    // Get mangled name - kernels have special mangling
    if (kind == DeviceFunctionKind::Kernel) {
      devInfo.mangledName = getMangledKernelName(FD);
    } else {
      devInfo.mangledName = CGM.getMangledName(FD).str();
    }

    deviceFunctions.push_back(std::move(devInfo));
  }

  // Additionally, if it's a kernel, extract full argument info
  if (kind == DeviceFunctionKind::Kernel) {
    KernelInfo info = analyzeKernel(FD);
    kernels.push_back(std::move(info));
  }

  return true; // Continue traversal
}

DeviceFunctionKind HIPKernelCollector::classifyFunction(const FunctionDecl *FD) const {
  bool hasGlobal = FD->hasAttr<CUDAGlobalAttr>();
  bool hasDevice = FD->hasAttr<CUDADeviceAttr>();
  bool hasHost = FD->hasAttr<CUDAHostAttr>();

  if (hasGlobal) {
    return DeviceFunctionKind::Kernel;
  } else if (hasDevice && hasHost) {
    return DeviceFunctionKind::HostDevice;
  } else if (hasDevice) {
    return DeviceFunctionKind::DeviceOnly;
  } else {
    return DeviceFunctionKind::HostOnly;
  }
}

bool HIPKernelCollector::isKernelFunction(const FunctionDecl *FD) const {
  return FD->hasAttr<CUDAGlobalAttr>();
}

bool HIPKernelCollector::isDeviceOnlyFunction(const FunctionDecl *FD) const {
  return FD->hasAttr<CUDADeviceAttr>() && !FD->hasAttr<CUDAHostAttr>();
}

bool HIPKernelCollector::isDeviceVisibleFunction(const FunctionDecl *FD) const {
  // Device-visible means: kernels, device-only, or host-device functions
  // These should all be included in device-side compilation
  DeviceFunctionKind kind = classifyFunction(FD);
  return kind != DeviceFunctionKind::HostOnly;
}

KernelInfo HIPKernelCollector::analyzeKernel(const FunctionDecl *FD) {
  KernelInfo info;
  info.decl = FD;
  info.demangledName = FD->getNameAsString();
  info.mangledName = getMangledKernelName(FD);

  // Analyze each parameter
  for (const ParmVarDecl *param : FD->parameters()) {
    info.arguments.push_back(analyzeArgument(param));
  }

  return info;
}

KernelArgInfo HIPKernelCollector::analyzeArgument(const ParmVarDecl *PVD) {
  KernelArgInfo arg;
  arg.name = PVD->getNameAsString();
  arg.type = PVD->getType();
  arg.isPointer = isPointerType(arg.type);
  arg.sizeBytes = getTypeSizeBytes(arg.type);
  arg.alignBytes = getTypeAlignBytes(arg.type);
  return arg;
}

std::string HIPKernelCollector::getMangledKernelName(const FunctionDecl *FD) {
  // Use clang's CodeGenModule to get the mangled name
  // This matches how cgeist handles kernel names in clang-mlir.cc
  return CGM.getMangledName(GlobalDecl(FD, KernelReferenceKind::Kernel)).str();
}

bool HIPKernelCollector::isPointerType(QualType type) const {
  // Get canonical type to see through typedefs
  QualType canonType = type.getCanonicalType();

  // Check for pointer types
  if (canonType->isPointerType())
    return true;

  // Check for reference types (treated as pointers)
  if (canonType->isReferenceType())
    return true;

  // Check for array types (decay to pointers)
  if (canonType->isArrayType())
    return true;

  return false;
}

unsigned HIPKernelCollector::getTypeSizeBytes(QualType type) const {
  // Use ASTContext to get size for target architecture
  // This accounts for pointer sizes, struct padding, etc.
  CharUnits size = ctx.getTypeSizeInChars(type);
  return static_cast<unsigned>(size.getQuantity());
}

unsigned HIPKernelCollector::getTypeAlignBytes(QualType type) const {
  // Use ASTContext to get alignment for target architecture
  CharUnits align = ctx.getTypeAlignInChars(type);
  return static_cast<unsigned>(align.getQuantity());
}

const KernelInfo *HIPKernelCollector::findKernel(llvm::StringRef name) const {
  for (const auto &kernel : kernels) {
    if (kernel.demangledName == name || kernel.mangledName == name)
      return &kernel;
  }
  return nullptr;
}

std::string vortex::kernelInfoToMLIRAttr(const KernelInfo &info) {
  // Generate MLIR attribute string for embedding in operations
  // Format: {kernel_name = "...", args = [{name = "...", type = "...", ...}, ...]}
  std::string result;
  llvm::raw_string_ostream os(result);

  os << "{";
  os << "kernel_name = \"" << info.mangledName << "\", ";
  os << "demangled_name = \"" << info.demangledName << "\", ";
  os << "args = [";

  bool first = true;
  unsigned offset = 0;
  for (const auto &arg : info.arguments) {
    if (!first)
      os << ", ";
    first = false;

    // Compute aligned offset
    unsigned padding = 0;
    if (arg.alignBytes > 0) {
      unsigned misalign = offset % arg.alignBytes;
      if (misalign > 0)
        padding = arg.alignBytes - misalign;
    }
    offset += padding;

    os << "{";
    os << "name = \"" << arg.name << "\", ";
    os << "type = \"" << arg.type.getAsString() << "\", ";
    os << "is_pointer = " << (arg.isPointer ? "true" : "false") << ", ";
    os << "size = " << arg.sizeBytes << ", ";
    os << "align = " << arg.alignBytes << ", ";
    os << "offset = " << offset;
    os << "}";

    offset += arg.sizeBytes;
  }

  os << "], ";
  os << "total_size = " << offset;
  os << "}";

  return result;
}

void vortex::dumpKernelInfo(const KernelInfo &info, llvm::raw_ostream &os) {
  os << "Kernel: " << info.demangledName << "\n";
  os << "  Mangled: " << info.mangledName << "\n";
  os << "  Arguments (" << info.arguments.size() << "):\n";

  unsigned offset = 0;
  for (const auto &arg : info.arguments) {
    // Compute aligned offset
    unsigned padding = 0;
    if (arg.alignBytes > 0) {
      unsigned misalign = offset % arg.alignBytes;
      if (misalign > 0)
        padding = arg.alignBytes - misalign;
    }
    offset += padding;

    os << "    " << arg.name << ": " << arg.type.getAsString();
    os << " (size=" << arg.sizeBytes;
    os << ", align=" << arg.alignBytes;
    os << ", offset=" << offset;
    if (arg.isPointer)
      os << ", ptr";
    os << ")\n";

    offset += arg.sizeBytes;
  }

  os << "  Total args size: " << offset << " bytes\n";
}
