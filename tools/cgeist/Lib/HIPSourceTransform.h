//===- HIPSourceTransform.h - HIP source-to-source transform ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides source-to-source transformation for HIP code to insert
// kernel launch wrapper functions. This ensures kernel arguments maintain
// their correct order during MLIR codegen.
//
// Problem: When hipLaunchKernelGGL is called in main(), kernel args are local
// variables. During MLIR kernel outlining, traceToFunctionArg() fails because
// args aren't function parameters, resulting in incorrect argument reordering.
//
// Solution: Generate wrapper functions where kernel args ARE function params:
//   void __launch_<kernel>(<kernel_args>, dim3 grid, dim3 block) {
//       hipLaunchKernelGGL(<kernel>, grid, block, 0, 0, <kernel_args>);
//   }
// Then replace the original call with: __launch_<kernel>(args, grid, block);
//
//===----------------------------------------------------------------------===//

#ifndef POLYGEIST_HIP_SOURCE_TRANSFORM_H
#define POLYGEIST_HIP_SOURCE_TRANSFORM_H

#include "HIPKernelAnalysis.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include <optional>
#include <set>
#include <string>
#include <vector>
#include <map>

namespace vortex {

/// Analyzes kernel launch arguments to detect compile-time constants.
/// MLIR performs constant folding on kernel arguments, so we must do
/// the same on the host side to ensure argument counts match.
class ConstantArgumentAnalyzer {
public:
  /// Analyze a kernel launch site and determine which args are constants
  void analyzeLaunchSite(const clang::CUDAKernelCallExpr *launch,
                         clang::ASTContext &ctx);

  /// Check if argument at index is a compile-time constant
  bool isConstant(unsigned argIndex) const;

  /// Get the constant value (if known) for an argument
  std::optional<llvm::APSInt> getConstantValue(unsigned argIndex) const;

  /// Get set of constant argument indices
  std::set<unsigned> getConstantArgIndices() const;

  /// Clear the analysis
  void clear();

private:
  /// Map from arg index to constant value (if constant)
  llvm::DenseMap<unsigned, llvm::APSInt> constantArgs;

  /// Recursively evaluate an expression to find constant value
  std::optional<llvm::APSInt> evaluateConstant(const clang::Expr *expr,
                                                clang::ASTContext &ctx);
};

/// Information about a kernel launch site (hipLaunchKernelGGL call)
struct LaunchSiteInfo {
  const clang::CUDAKernelCallExpr *callExpr;  // The kernel launch expression
  const clang::FunctionDecl *kernel;           // Target kernel
  std::string kernelName;                       // Kernel name (demangled)
  clang::SourceRange sourceRange;               // Full range of the call
  const clang::Expr *gridExpr;                  // Grid dimension expression
  const clang::Expr *blockExpr;                 // Block dimension expression
  std::set<unsigned> constantArgIndices;        // Which args are compile-time constants
  std::map<unsigned, llvm::APSInt> constantArgValues;  // Constant values for folded args
};

/// Transforms HIP source code to insert kernel launch wrappers.
/// Uses Clang AST traversal and Rewriter for source modification.
class HIPSourceTransformer : public clang::RecursiveASTVisitor<HIPSourceTransformer> {
public:
  HIPSourceTransformer(clang::ASTContext &ctx,
                       clang::SourceManager &SM,
                       const clang::LangOptions &LangOpts);

  /// Run the transformation on the translation unit
  /// Returns true if any transformations were made
  bool transform(clang::TranslationUnitDecl *TU);

  /// Visit function declarations to find __global__ kernels
  bool VisitFunctionDecl(clang::FunctionDecl *FD);

  /// Visit call expressions to find kernel launches
  bool VisitCUDAKernelCallExpr(clang::CUDAKernelCallExpr *CE);

  /// Get the transformed source code
  std::string getTransformedSource() const;

  /// Write transformed source to a file
  bool writeToFile(llvm::StringRef filename) const;

  /// Generate and write host stub header file (_args.h)
  /// This generates the stub with correct argument order directly from AST
  bool writeStubHeader(llvm::StringRef outputDir) const;

  /// Get collected kernel information
  const std::vector<KernelInfo> &getKernels() const { return kernels; }

  /// Get collected launch sites
  const std::vector<LaunchSiteInfo> &getLaunchSites() const { return launchSites; }

private:
  clang::ASTContext &ctx;
  clang::SourceManager &SM;
  clang::Rewriter rewriter;

  std::vector<KernelInfo> kernels;
  std::vector<LaunchSiteInfo> launchSites;
  std::map<std::string, bool> wrappersGenerated;  // Track which wrappers we've generated
  std::map<std::string, bool> kernelsNeedingConversion;  // Track kernels that need blockIdx→threadIdx
  std::map<std::string, std::set<unsigned>> kernelConstantArgs;  // Track constant args per kernel
  std::map<std::string, std::map<unsigned, llvm::APSInt>> kernelConstantArgValues;  // Constant values

  /// Analyze a kernel function
  KernelInfo analyzeKernel(const clang::FunctionDecl *FD);

  /// Generate wrapper function source code for a kernel
  std::string generateWrapperFunction(const KernelInfo &kernel);

  /// Generate wrapper function name from kernel name
  std::string getWrapperName(llvm::StringRef kernelName) const {
    return ("__launch_" + kernelName).str();
  }

  /// Generate the replacement call to the wrapper
  std::string generateWrapperCall(const LaunchSiteInfo &site);

  /// Insert wrapper function before the kernel definition
  void insertWrapper(const KernelInfo &kernel);

  /// Replace a kernel launch with a wrapper call
  void replaceLaunchWithWrapper(const LaunchSiteInfo &site);

  /// Get source text for an expression
  std::string getSourceText(const clang::Expr *E) const;

  /// Get source text for a source range
  std::string getSourceText(clang::SourceRange range) const;

  /// Check if a function is a kernel (__global__)
  bool isKernelFunction(const clang::FunctionDecl *FD) const;

  /// Get type string for a parameter (for wrapper generation)
  std::string getParamTypeString(const clang::ParmVarDecl *PVD) const;

  /// Parse arguments from hipLaunchKernelGGL source text
  std::vector<std::string> parseHipLaunchArgs(const std::string &text) const;

  /// Insert #include directive for the stub header into the source
  void insertStubInclude(const KernelInfo &kernel);

  /// Replace blockIdx with threadIdx in a kernel body (for blockIdx-only kernels)
  void replaceBlockIdxWithThreadIdx(const KernelInfo &kernel);
};

} // namespace vortex

#endif // POLYGEIST_HIP_SOURCE_TRANSFORM_H
