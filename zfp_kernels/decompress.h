#include <iostream>
#include <stdint.h>
#include <hip/hip_runtime.h>
#include <zfp.h>

// target = 1 // CPU
// target = 2 // GPU
int decompress ( double * data, int nx, void * compressed, double rate, int target) {

  int status = 0;
  zfp_field* field;  /* array meta data */
  zfp_stream* zfp;   /* compressed stream */
  zfp_type type;     /* array scalar type */
  zfp_exec_policy exec_policy;
  size_t bufsize;    /* byte size of compressed buffer */
  bitstream* stream; /* bit stream to write to or read from */
  size_t zfpsize;    /* byte size of compressed stream */
  uint minbits;      /* min bits per block */
  uint maxbits;      /* max bits per block */
  uint maxprec;      /* max precision */
  int minexp;        /* min bit plane encoded */

  if (target == 1 ) {
    exec_policy = zfp_exec_serial;
  }
  else if (target == 2 ) {
    exec_policy = zfp_exec_hip ;
  }
  else {
    return -1;
  }

  type = zfp_type_double;
  field = zfp_field_1d(data, type, nx);

  zfp = zfp_stream_open(NULL);
  zfp_stream_set_rate(zfp, rate, type, zfp_field_dimensionality(field), zfp_false);



   /* allocate buffer for compressed data */
  bufsize = zfp_stream_maximum_size(zfp, field);

  zfp_stream_params(zfp, &minbits, &maxbits, &maxprec, &minexp);
  //zfp_stream_set_params(zfp, minbits, bufsize, maxprec, minexp);

  /* associate bit stream with allocated buffer */
  stream = stream_open(compressed, bufsize);
  zfp_stream_set_bit_stream(zfp, stream);
  zfp_stream_rewind(zfp);
  //zfp_stream_set_execution(zfp, exec_policy);
  zfp_stream_set_execution(zfp, exec_policy);
  zfp->exec.policy = exec_policy;

  /* compress array and output compressed stream */
  zfpsize = zfp_decompress(zfp, field);
  if (!zfpsize) {
    fprintf(stderr, "decompression failed\n");
    status = EXIT_FAILURE;
    return -1;
  }

  return zfpsize;
}
