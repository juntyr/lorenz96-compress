#include <iostream>
#include <fstream>
#include <sstream> 
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

__global__ void lorenz96_tendency(const double forcing, const double* const X_ensemble, double* const dXdt_ensemble) {
    int k = threadIdx.x;
    int k_max = blockDim.x;

    int ensemble_id = blockIdx.x;
    int ensemble_size = gridDim.x;
    
    const double* const X = X_ensemble + ensemble_id * k_max;
    double* const dXdt = dXdt_ensemble + ensemble_id * k_max;

    int k_m2 = (k-2 + k_max) % k_max;
    int k_m1 = (k-1 + k_max) % k_max;
    int k_p1 = (k+1) % k_max;

    dXdt[k] = -X[k_m2]*X[k_m1] + X[k_m1]*X[k_p1] - X[k] + forcing;
}

__global__ void lorenz96_timestep_direct(const double dt, double* const X_ensemble, const double* const dXdt_ensemble) {
    int k = threadIdx.x;
    int k_max = blockDim.x;

    int ensemble_id = blockIdx.x;
    int ensemble_size = gridDim.x;

    double* const X = X_ensemble + ensemble_id * k_max;
    const double* const dXdt = dXdt_ensemble + ensemble_id * k_max;

    X[k] += dXdt[k] * dt;
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
    parser.set_optional<int>("e", "ensemble-size", 10);
    parser.set_optional<std::string>("o", "output", "state");
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
    int ensemble_size = parser.get<int>("e");
    std::string output = parser.get<std::string>("o");

    std::cout << "Lorenz96(k=" << k << ", F=" << forcing << ", dt=" << dt << ", t_max=" << max_time << ")" << std::endl;
    std::cout << " running ensemble of size " << ensemble_size << " and saving to '" << output << "_[i]'" << std::endl << std::endl;

    int size = k * ensemble_size;

    double X_ensemble[size];
    double *X_ensemble_gpu, *dXdt_ensemble_gpu;

    // Initialise the initial state
    for (int i = 0; i < size; i++) {
        X_ensemble[i] = 0.0;
    }
    for (int i = 0; i < ensemble_size; i++) {
        X_ensemble[k*i] = 1.0;
    }

    std::ofstream out_files[ensemble_size];
    for (int i = 0; i < ensemble_size; i++) {
        std::stringstream file_name;
        file_name << output << "_" << i;
        out_files[i].open(file_name.str(), std::ios::out | std::ios::trunc | std::ios::binary);
    }

    std::cout << "Initial state:" << std::endl;
    print_state(X_ensemble, k, 0.0);
    std::cout << std::endl;

    for (int i = 0; i < ensemble_size; i++) {
        for (int j = 0; j < k; j++) {
            out_files[i].write(reinterpret_cast<const char*>(&X_ensemble[i*k+j]), sizeof(X_ensemble[i*k+j]));
        }
    }

    HIP_ERRCHK(hipMalloc(&X_ensemble_gpu, sizeof(double) * size));
    HIP_ERRCHK(hipMalloc(&dXdt_ensemble_gpu, sizeof(double) * size));

    HIP_ERRCHK(hipMemcpy(X_ensemble_gpu, X_ensemble, sizeof(double) * size, hipMemcpyHostToDevice));

    dim3 blocks(ensemble_size);
    dim3 threads(k);

    double t = 0.0;

    while ((t += dt) <= max_time) {
        lorenz96_tendency<<<blocks, threads>>>(forcing, X_ensemble_gpu, dXdt_ensemble_gpu);
        lorenz96_timestep_direct<<<blocks, threads>>>(dt, X_ensemble_gpu, dXdt_ensemble_gpu);

        HIP_ERRCHK(hipMemcpy(X_ensemble, X_ensemble_gpu, sizeof(double) * size, hipMemcpyDeviceToHost));

        if ((t + dt) > max_time) {
            std::cout << std::endl << "Final state:" << std::endl;
        }
        print_state(X_ensemble, k, t);

        for (int i = 0; i < ensemble_size; i++) {
            for (int j = 0; j < k; j++) {
                out_files[i].write(reinterpret_cast<const char*>(&X_ensemble[i*k+j]), sizeof(X_ensemble[i*k+j]));
            }
        }
    }

    for (int i = 0; i < ensemble_size; i++) {
        out_files[i].close();
    }

    HIP_ERRCHK(hipFree(X_ensemble_gpu));
    HIP_ERRCHK(hipFree(dXdt_ensemble_gpu));

    return 0;
}
