// Polygeist device stub for <stdio.h>
// This stub intercepts the C stdio header during CUDA/HIP device compilation.

#ifndef __POLYGEIST_DEVICE_STUBS_STDIO_H
#define __POLYGEIST_DEVICE_STUBS_STDIO_H

#ifdef __CUDA__
// During device compilation (cgeist with --emit-cuda):
// Provide minimal stdio declarations. printf is handled by hip_runtime.h
// to ensure proper __device__ attribute for kernel code.

#include <stddef.h>  // For size_t, NULL
#include <stdarg.h>  // For va_list

#ifdef __cplusplus
extern "C" {
#endif

// NOTE: printf is NOT declared here - it comes from hip_runtime.h
// with proper __device__ / __host__ attributes

typedef void FILE;

extern FILE* stdin;
extern FILE* stdout;
extern FILE* stderr;

int fprintf(FILE* stream, const char* format, ...);
int sprintf(char* str, const char* format, ...);
int snprintf(char* str, size_t size, const char* format, ...);
int vprintf(const char* format, va_list ap);
int vfprintf(FILE* stream, const char* format, va_list ap);
int vsprintf(char* str, const char* format, va_list ap);
int vsnprintf(char* str, size_t size, const char* format, va_list ap);

int scanf(const char* format, ...);
int fscanf(FILE* stream, const char* format, ...);
int sscanf(const char* str, const char* format, ...);

int putchar(int c);
int puts(const char* s);
int fputc(int c, FILE* stream);
int fputs(const char* s, FILE* stream);
int getchar(void);
int fgetc(FILE* stream);
char* fgets(char* s, int n, FILE* stream);

FILE* fopen(const char* filename, const char* mode);
FILE* freopen(const char* filename, const char* mode, FILE* stream);
int fclose(FILE* stream);
size_t fread(void* ptr, size_t size, size_t nmemb, FILE* stream);
size_t fwrite(const void* ptr, size_t size, size_t nmemb, FILE* stream);
int fflush(FILE* stream);
int fseek(FILE* stream, long offset, int whence);
long ftell(FILE* stream);
void rewind(FILE* stream);
int feof(FILE* stream);
int ferror(FILE* stream);
void clearerr(FILE* stream);

#define EOF (-1)
#define SEEK_SET 0
#define SEEK_CUR 1
#define SEEK_END 2

#ifdef __cplusplus
}
#endif

#else
// Pure host compilation (not cgeist): include the real header
#include_next <stdio.h>
#endif

#endif // __POLYGEIST_DEVICE_STUBS_STDIO_H
