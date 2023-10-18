#include <iostream>
#include <stdint.h>
#include <hip/hip_runtime.h>
#include <zfp.h>

/* HIP error handling macro */
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
  const double rate = 4.0;
  double *X_gpu;
  double X[nx];

  int status = 0;
  zfp_field* field;  /* array meta data */
  zfp_stream* zfp;   /* compressed stream */
  zfp_type type;     /* array scalar type */
  zfp_exec_policy exec_policy = zfp_exec_hip;
  void* buffer;      /* storage for compressed stream */
  size_t bufsize;    /* byte size of compressed buffer */
  bitstream* stream; /* bit stream to write to or read from */
  size_t zfpsize;    /* byte size of compressed stream */
  uint minbits;      /* min bits per block */
  uint maxbits;      /* max bits per block */
  uint maxprec;      /* max precision */
  int minexp;        /* min bit plane encoded */




  // Initialise the initial state
  for (int i = 0; i < nx; i++) {
      X[i] = double(i);
  }

  type = zfp_type_double;
  field = zfp_field_1d(X, type, nx);

  zfp = zfp_stream_open(NULL);
  zfp_stream_set_rate(zfp, rate, type, zfp_field_dimensionality(field), zfp_false);



   /* allocate buffer for compressed data */
  bufsize = zfp_stream_maximum_size(zfp, field);
  buffer = malloc(bufsize);

  zfp_stream_params(zfp, &minbits, &maxbits, &maxprec, &minexp);
  zfp_stream_set_params(zfp, minbits, bufsize, maxprec, minexp);

  /* associate bit stream with allocated buffer */
  stream = stream_open(buffer, bufsize);
  zfp_stream_set_bit_stream(zfp, stream);
  zfp_stream_rewind(zfp);
  //zfp_stream_set_execution(zfp, exec_policy);
  zfp_stream_set_execution(zfp, zfp_exec_hip);
  zfp->exec.policy = zfp_exec_hip;

  print_state(X, nx);

  //HIP_ERRCHK(hipMalloc(&X_gpu, sizeof(double) * nx));

  //HIP_ERRCHK(hipMemcpy(X_gpu, X, sizeof(double) * nx, hipMemcpyHostToDevice));

  /* compress array and output compressed stream */
  zfpsize = zfp_compress(zfp, field);
  if (!zfpsize) {
    fprintf(stderr, "compression failed\n");
    status = EXIT_FAILURE;
  }
  else
    fwrite(buffer, 1, zfpsize, stdout);

  //HIP_ERRCHK(hipMemcpy(X, X_gpu, sizeof(double) * nx, hipMemcpyDeviceToHost));
  //HIP_ERRCHK(hipFree(X_gpu));
  //print_state(X, nx);

    return 0;
}
