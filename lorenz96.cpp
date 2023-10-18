#include <iostream>
#include <hip/hip_runtime.h>

#include "cmdparser.hpp"

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

void print_state(const double* const X, const int k, const double t)
{
    std::cout << "X at t=" << t << ":" << std::endl;
    std::cout << "[ ";
    for (int i = 0; i < k; i++) {
        std::cout << X[i];
        if (i != k - 1) {
            std::cout << ", ";
        }
    }
    std::cout << " ]" << std::endl;
}

void configure_cli(cli::Parser& parser) {
    parser.set_required<double>("t", "max-time");
    parser.set_optional<double>("s", "dt", 0.01);
    parser.set_optional<double>("f", "forcing", 8.0);
    parser.set_optional<int>("k", "", 36);
}

int main(int argc, char *argv[])
{
    cli::Parser parser(argc, argv);
    configure_cli(parser);
    parser.run_and_exit_if_error();
    
    double max_time = parser.get<double>("t");
    double dt = parser.get<double>("s");
    double forcing = parser.get<double>("f");
    int k = parser.get<int>("k");

    std::cout << "Lorenz96(k=" << k << ", F=" << forcing << ", dt=" << dt << ", t_max=" << max_time << ")" << std::endl << std::endl;

    double X[k];
    double *X_gpu, *dXdt_gpu;

    // Initialise the initial state
    for (int i = 0; i < k; i++) {
        X[i] = 0.0;
    }
    X[0] = 1.0;

    std::cout << "Initial state:" << std::endl;
    print_state(X, k, 0.0);
    std::cout << std::endl;

    HIP_ERRCHK(hipMalloc(&X_gpu, sizeof(double) * k));
    HIP_ERRCHK(hipMalloc(&dXdt_gpu, sizeof(double) * k));

    HIP_ERRCHK(hipMemcpy(X_gpu, X, sizeof(double) * k, hipMemcpyHostToDevice));

    dim3 blocks(1);
    dim3 threads(64);

    double t = 0.0;

    while ((t += dt) <= max_time) {
        lorenz96_tendency<<<blocks, threads>>>(k, forcing, X_gpu, dXdt_gpu);
        lorenz96_update<<<blocks, threads>>>(k, dt, X_gpu, dXdt_gpu);

        HIP_ERRCHK(hipMemcpy(X, X_gpu, sizeof(double) * k, hipMemcpyDeviceToHost));

        if ((t + dt) > max_time) {
            std::cout << std::endl << "Final state:" << std::endl;
        }
        print_state(X, k, t);
    }

    HIP_ERRCHK(hipFree(X_gpu));
    HIP_ERRCHK(hipFree(dXdt_gpu));

    return 0;
}
