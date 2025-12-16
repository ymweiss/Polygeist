/*===-- bits/c++config.h - Polygeist device stub for libstdc++ config -----===*\
|*                                                                            *|
|* Part of the Polygeist Project, under the Apache License v2.0               *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|*                                                                            *|
|* This stub header suppresses __float128 usage in GCC's libstdc++ during     *|
|* CUDA/HIP device compilation. Clang's CUDA frontend doesn't support         *|
|* __float128, but device code doesn't need it anyway.                        *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef __POLYGEIST_DEVICE_STUBS_CPPCONFIG
#define __POLYGEIST_DEVICE_STUBS_CPPCONFIG

#ifdef __CUDA_ARCH__
// Suppress __float128 usage in device mode
#define _GLIBCXX_USE_FLOAT128 0
#endif

// Chain to the real header
#include_next <bits/c++config.h>

#endif // __POLYGEIST_DEVICE_STUBS_CPPCONFIG
