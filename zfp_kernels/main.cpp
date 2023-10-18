#include <iostream>
#include <stdint.h>
#include <hip/hip_runtime.h>
#include <zfp.h>
#include "compress.h"


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
  const double rate = 4.0;
  double X[nx];
  void* buffer;      /* storage for compressed stream */
  int ret;




  // Initialise the initial state
  for (int i = 0; i < nx; i++) {
      X[i] = double(i);
  }
  buffer = malloc(100);

  ret = compress(X,nx,buffer, rate,1);

  if (!ret) {
    fprintf(stderr, "compression failed\n");
  }
  else
    fwrite(buffer, 1, ret, stdout);
  return 0;
}
