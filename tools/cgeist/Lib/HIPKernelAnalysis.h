//===- HIPKernelAnalysis.h - AST-based HIP kernel analysis -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides AST-based analysis infrastructure for HIP kernels.
// It replaces regex-based Python scripts with robust clang AST traversal
// for extracting kernel information (arguments, types, mangled names).
//
//===----------------------------------------------------------------------===//

#ifndef POLYGEIST_HIP_KERNEL_ANALYSIS_H
#define POLYGEIST_HIP_KERNEL_ANALYSIS_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Mangle.h"
#include "clang/../../lib/CodeGen/CodeGenModule.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <vector>

namespace vortex {

/// Information about a single kernel argument extracted from AST
struct KernelArgInfo {
  std::string name;           // Parameter name
  clang::QualType type;       // Clang type
  bool isPointer;             // True if pointer/memref type
  unsigned sizeBytes;         // Size in bytes (for target architecture)
  unsigned alignBytes;        // Alignment in bytes

  KernelArgInfo() : isPointer(false), sizeBytes(0), alignBytes(0) {}
};

/// Function classification for HIP/CUDA code
enum class DeviceFunctionKind {
  Kernel,       // __global__ - entry point callable from host
  DeviceOnly,   // __device__ without __host__ - only callable from device
  HostDevice,   // Both __host__ and __device__ - callable from both
  HostOnly      // No attributes or __host__ only - host code
};

/// Information extracted from a HIP __global__ kernel function
struct KernelInfo {
  const clang::FunctionDecl *decl;  // Original AST declaration
  std::string mangledName;          // Mangled kernel name (for linking)
  std::string demangledName;        // Human-readable name
  llvm::SmallVector<KernelArgInfo, 8> arguments;
  bool usesBlockIdx = false;        // Kernel uses blockIdx.x/y/z
  bool usesThreadIdx = false;       // Kernel uses threadIdx.x/y/z
  bool usesBlockDim = false;        // Kernel uses blockDim.x/y/z
  bool usesGridDim = false;         // Kernel uses gridDim.x/y/z

  /// Returns true if kernel uses only blockIdx (no threadIdx)
  /// This pattern needs blockIdx→threadIdx conversion
  bool needsBlockIdxConversion() const {
    return usesBlockIdx && !usesThreadIdx;
  }

  /// Total size of all arguments (for struct packing)
  unsigned getTotalArgsSize() const {
    unsigned total = 0;
    for (const auto &arg : arguments) {
      // Align to argument's alignment requirement
      unsigned padding = (alignBytes(arg.alignBytes, total) - total);
      total += padding + arg.sizeBytes;
    }
    return total;
  }

private:
  static unsigned alignBytes(unsigned align, unsigned offset) {
    return (offset + align - 1) / align * align;
  }
};

/// Information about a device function (__device__ or __global__)
struct DeviceFunctionInfo {
  const clang::FunctionDecl *decl;
  std::string mangledName;
  std::string demangledName;
  DeviceFunctionKind kind;
};

/// Visitor to detect blockIdx/threadIdx usage in a kernel body
class BlockThreadIdxVisitor : public clang::RecursiveASTVisitor<BlockThreadIdxVisitor> {
public:
  bool usesBlockIdx = false;
  bool usesThreadIdx = false;
  bool usesBlockDim = false;
  bool usesGridDim = false;

  bool VisitDeclRefExpr(clang::DeclRefExpr *DRE) {
    if (auto *VD = clang::dyn_cast<clang::VarDecl>(DRE->getDecl())) {
      llvm::StringRef name = VD->getName();
      if (name == "blockIdx") usesBlockIdx = true;
      else if (name == "threadIdx") usesThreadIdx = true;
      else if (name == "blockDim") usesBlockDim = true;
      else if (name == "gridDim") usesGridDim = true;
    }
    return true;
  }
};

/// Collects kernel information from a HIP/CUDA translation unit using AST traversal.
/// This replaces the fragile regex-based parsing in Python scripts.
class HIPKernelCollector : public clang::RecursiveASTVisitor<HIPKernelCollector> {
public:
  HIPKernelCollector(clang::ASTContext &ctx, clang::CodeGen::CodeGenModule &CGM);

  /// Visit function declarations to find __global__ kernels and __device__ functions
  bool VisitFunctionDecl(clang::FunctionDecl *FD);

  /// Classify a function based on its CUDA/HIP attributes
  DeviceFunctionKind classifyFunction(const clang::FunctionDecl *FD) const;

  /// Check if a function is a HIP/CUDA kernel (__global__ attribute)
  bool isKernelFunction(const clang::FunctionDecl *FD) const;

  /// Check if a function is device-only (__device__ without __host__)
  bool isDeviceOnlyFunction(const clang::FunctionDecl *FD) const;

  /// Check if a function should be included in device compilation
  /// (kernels, device-only, or host-device functions)
  bool isDeviceVisibleFunction(const clang::FunctionDecl *FD) const;

  /// Analyze a kernel function and extract all metadata
  KernelInfo analyzeKernel(const clang::FunctionDecl *FD);

  /// Get all discovered kernels (__global__ functions)
  const std::vector<KernelInfo> &getKernels() const { return kernels; }

  /// Get all device functions (__device__ and __global__)
  const std::vector<DeviceFunctionInfo> &getDeviceFunctions() const {
    return deviceFunctions;
  }

  /// Find kernel by demangled name
  const KernelInfo *findKernel(llvm::StringRef name) const;

  /// Clear collected data (for reuse)
  void clear() {
    kernels.clear();
    deviceFunctions.clear();
  }

private:
  clang::ASTContext &ctx;
  clang::CodeGen::CodeGenModule &CGM;
  std::vector<KernelInfo> kernels;
  std::vector<DeviceFunctionInfo> deviceFunctions;

  /// Extract argument information from a parameter declaration
  KernelArgInfo analyzeArgument(const clang::ParmVarDecl *PVD);

  /// Get the mangled name for a kernel function
  std::string getMangledKernelName(const clang::FunctionDecl *FD);

  /// Determine if a type is a pointer type (including arrays)
  bool isPointerType(clang::QualType type) const;

  /// Get size of type in bytes for target architecture
  unsigned getTypeSizeBytes(clang::QualType type) const;

  /// Get alignment of type in bytes for target architecture
  unsigned getTypeAlignBytes(clang::QualType type) const;
};

/// Utility functions for working with kernel metadata

/// Convert KernelInfo to an MLIR attribute dictionary (for embedding in ops)
/// Returns a string representation suitable for setAttr()
std::string kernelInfoToMLIRAttr(const KernelInfo &info);

/// Print kernel info for debugging
void dumpKernelInfo(const KernelInfo &info, llvm::raw_ostream &os);

} // namespace vortex

#endif // POLYGEIST_HIP_KERNEL_ANALYSIS_H
