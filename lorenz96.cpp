#include <iostream>
#include <hip/hip_runtime.h>

/* HIP error handling macro */
#define HIP_ERRCHK(err) (hip_errchk(err, __FILE__, __LINE__ ))
static inline void hip_errchk(hipError_t err, const char *file, int line) {
  if (err != hipSuccess) {
    printf("\n\n%s in %s at line %d\n", hipGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

__global__ void lorenz96_tendency(const int k, const double forcing, const double* const X, double* const dXdt) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= k) return;

    dXdt[tid] = -X[(tid-2+k)%k]*X[(tid-1+k)%k] + X[(tid-1+k)%k]*X[(tid+1)%k] - X[tid] + forcing;
}

__global__ void lorenz96_update(const int k, const double dt, double* const X, const double* const dXdt) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= k) return;

    X[tid] += dXdt[tid] * dt;
}

void print_state(const double* const X, const int size, const double t)
{
    std::cout << "X at t=" << t << ":" << std::endl;
    std::cout << "[ ";
    for (int i = 0; i < size; i++) {
        std::cout << X[i];
        if (i != size - 1) {
            std::cout << ", ";
        }
    }
    std::cout << " ]" << std::endl;
}

int main(void)
{
    const int k = 36;
    const double forcing = 8.0;
    const double dt = 0.01;
    const int size = k;

    double X[size];
    double *X_gpu, *dXdt_gpu;

    // Initialise the initial state
    for (int i = 0; i < size; i++) {
        X[i] = 0.0;
    }
    X[0] = 1.0;

    print_state(X, size, 0.0);

    HIP_ERRCHK(hipMalloc(&X_gpu, sizeof(double) * size));
    HIP_ERRCHK(hipMalloc(&dXdt_gpu, sizeof(double) * size));

    HIP_ERRCHK(hipMemcpy(X_gpu, X, sizeof(double) * size, hipMemcpyHostToDevice));

    dim3 blocks(1);
    dim3 threads(64);

    lorenz96_tendency<<<blocks, threads>>>(k, forcing, X_gpu, dXdt_gpu);
    lorenz96_update<<<blocks, threads>>>(k, dt, X_gpu, dXdt_gpu);

    HIP_ERRCHK(hipMemcpy(X, X_gpu, sizeof(double) * size, hipMemcpyDeviceToHost));

    HIP_ERRCHK(hipFree(X_gpu));
    HIP_ERRCHK(hipFree(dXdt_gpu));

    print_state(X, size, dt);

    return 0;
}
