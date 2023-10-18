#include <iostream>
#include <stdint.h>
#include <hip/hip_runtime.h>
#include <zfp.h>
#include "compress.h"
#include "decompress.h"





#define HIP_ERRCHK(err) (hip_errchk(err, __FILE__, __LINE__ ))
static inline void hip_errchk(hipError_t err, const char *file, int line) {
  if (err != hipSuccess) {
    printf("\n\n%s in %s at line %d\n", hipGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

void print_state(const double* const X, const int size)
{
    std::cout << "X =" << ":" << std::endl;
    std::cout << "[ ";
    for (int i = 0; i < size; i++) {
        std::cout << X[i];
        if (i != size - 1) {
            std::cout << ", ";
        }
    }
    std::cout << " ]" << std::endl;
}

int main(int argc, char *argv[])
{
  int nx = 100;
  const double rate = 10.0;
  double *X_gpu;
  double *buffer_gpu;
  double X[nx];
  void* buffer;      /* storage for compressed stream */
  int ret;





  // Initialise the initial state
  for (int i = 0; i < nx; i++) {
      X[i] = double(i)/10;
  }
  print_state(X, nx);

  HIP_ERRCHK(hipMalloc(&X_gpu, sizeof(double) * nx));

  HIP_ERRCHK(hipMemcpy(X_gpu, X, sizeof(double) * nx, hipMemcpyHostToDevice));

  buffer = malloc(100);

  HIP_ERRCHK(hipMalloc(&buffer_gpu, sizeof(double) * nx));

  HIP_ERRCHK(hipMemcpy(buffer_gpu, buffer, sizeof(double) * nx, hipMemcpyHostToDevice));

  ret = compress(X_gpu,nx, buffer_gpu, rate, 2);

  if (!ret) {
    fprintf(stderr, "compression failed\n");
  }

  ret = decompress(X_gpu,nx, buffer_gpu, rate, 2);

  if (!ret) {
    fprintf(stderr, "decompression failed\n");
  }

  HIP_ERRCHK(hipMemcpy(X, X_gpu, sizeof(double) * nx, hipMemcpyDeviceToHost));
  HIP_ERRCHK(hipFree(X_gpu));
  print_state(X, nx);
  return 0;
}
