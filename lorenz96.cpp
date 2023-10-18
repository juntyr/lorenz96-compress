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

__global__ void lorenz96_timestep_euler_smoothing(double* const Xp0_ensemble, const double* const Xp2_ensemble) {
    int k = threadIdx.x;
    int k_max = blockDim.x;

    int ensemble_id = blockIdx.x;
    int ensemble_size = gridDim.x;

    double* const Xp1 = Xp0_ensemble + ensemble_id * k_max;
    const double* const Xp0 = Xp0_ensemble + ensemble_id * k_max;
    const double* const Xp2 = Xp2_ensemble + ensemble_id * k_max;

    Xp1[k] = (Xp0[k] + Xp2[k]) * 0.5;
}

__global__ void lorenz96_timestep_runge_kutta(double* const dXdt_ensemble, const double* const k1_ensemble, const double* const k2_ensemble, const double* const k3_ensemble, const double* const k4_ensemble) {
    int k = threadIdx.x;
    int k_max = blockDim.x;

    int ensemble_id = blockIdx.x;
    int ensemble_size = gridDim.x;

    double* const dXdt = dXdt_ensemble + ensemble_id * k_max;
    const double* const k1 = k1_ensemble + ensemble_id * k_max;
    const double* const k2 = k2_ensemble + ensemble_id * k_max;
    const double* const k3 = k3_ensemble + ensemble_id * k_max;
    const double* const k4 = k4_ensemble + ensemble_id * k_max;

    dXdt[k] = (k1[k] + k2[k]*2.0 + k3[k]*2.0 + k4[k]) / 6.0;
}

struct TimeStep {
    TimeStep(const int k, const int ensemble_size): size(k * ensemble_size), blocks(ensemble_size), threads(k) {}
    virtual ~TimeStep() {}

    virtual void time_step(double* const X_ensemble_gpu, const double dt, const double forcing) = 0;

    const int size;
    const dim3 blocks;
    const dim3 threads;
};

struct Direct: TimeStep {
    Direct(const int k, const int ensemble_size): TimeStep(k, ensemble_size) {
        HIP_ERRCHK(hipMalloc(&this->dXdt_ensemble_gpu, sizeof(double) * this->size));
    }

    ~Direct() {
        HIP_ERRCHK(hipFree(this->dXdt_ensemble_gpu));
    }

    void time_step(double* const X_ensemble_gpu, const double dt, const double forcing) {
        lorenz96_tendency<<<this->blocks, this->threads>>>(forcing, X_ensemble_gpu, this->dXdt_ensemble_gpu);
        lorenz96_timestep_direct<<<this->blocks, this->threads>>>(dt, X_ensemble_gpu, this->dXdt_ensemble_gpu);
    }

    double* dXdt_ensemble_gpu;
};

struct EulerSmoothing: TimeStep {
    EulerSmoothing(const int k, const int ensemble_size): TimeStep(k, ensemble_size) {
        HIP_ERRCHK(hipMalloc(&this->dXdt_ensemble_gpu, sizeof(double) * this->size));
        HIP_ERRCHK(hipMalloc(&this->Xtemp_ensemble_gpu, sizeof(double) * this->size));
    }

    ~EulerSmoothing() {
        HIP_ERRCHK(hipFree(this->dXdt_ensemble_gpu));
        HIP_ERRCHK(hipFree(this->Xtemp_ensemble_gpu));
    }

    void time_step(double* const X_ensemble_gpu, const double dt, const double forcing) {
        // Xtemp = X_(n)
        HIP_ERRCHK(hipMemcpy(this->Xtemp_ensemble_gpu, X_ensemble_gpu, sizeof(double) * this->size, hipMemcpyDeviceToDevice));

        // Xtemp = X_(n+1) = X_(n) + X'_(n) * dt
        lorenz96_tendency<<<this->blocks, this->threads>>>(forcing, this->Xtemp_ensemble_gpu, this->dXdt_ensemble_gpu);
        lorenz96_timestep_direct<<<this->blocks, this->threads>>>(dt, this->Xtemp_ensemble_gpu, this->dXdt_ensemble_gpu);

        // Xtemp = X_(n+2) = X_(n+1) + X'_(n+1) * dt
        lorenz96_tendency<<<this->blocks, this->threads>>>(forcing, this->Xtemp_ensemble_gpu, this->dXdt_ensemble_gpu);
        lorenz96_timestep_direct<<<this->blocks, this->threads>>>(dt, this->Xtemp_ensemble_gpu, this->dXdt_ensemble_gpu);

        // X = X_(n_1) = ( X_(n) + X_(n+2) ) / 2
        lorenz96_timestep_euler_smoothing<<<this->blocks, this->threads>>>(X_ensemble_gpu, this->Xtemp_ensemble_gpu);
    }

    double* dXdt_ensemble_gpu;
    double* Xtemp_ensemble_gpu;
};

struct RungeKutta: TimeStep {
    RungeKutta(const int k, const int ensemble_size): TimeStep(k, ensemble_size) {
        HIP_ERRCHK(hipMalloc(&this->dXdt_ensemble_gpu, sizeof(double) * this->size));
        HIP_ERRCHK(hipMalloc(&this->Xtemp_ensemble_gpu, sizeof(double) * this->size));
        HIP_ERRCHK(hipMalloc(&this->k1_ensemble_gpu, sizeof(double) * this->size));
        HIP_ERRCHK(hipMalloc(&this->k2_ensemble_gpu, sizeof(double) * this->size));
        HIP_ERRCHK(hipMalloc(&this->k3_ensemble_gpu, sizeof(double) * this->size));
        HIP_ERRCHK(hipMalloc(&this->k4_ensemble_gpu, sizeof(double) * this->size));
    }

    ~RungeKutta() {
        HIP_ERRCHK(hipFree(this->dXdt_ensemble_gpu));
        HIP_ERRCHK(hipFree(this->Xtemp_ensemble_gpu));
        HIP_ERRCHK(hipFree(this->k1_ensemble_gpu));
        HIP_ERRCHK(hipFree(this->k2_ensemble_gpu));
        HIP_ERRCHK(hipFree(this->k3_ensemble_gpu));
        HIP_ERRCHK(hipFree(this->k4_ensemble_gpu));
    }

    void time_step(double* const X_ensemble_gpu, const double dt, const double forcing) {
        // k1 = X'(X_n)
        lorenz96_tendency<<<this->blocks, this->threads>>>(forcing, X_ensemble_gpu, this->k1_ensemble_gpu);

        // k2 = X'(X_n + k1 * dt/2)
        HIP_ERRCHK(hipMemcpy(this->Xtemp_ensemble_gpu, X_ensemble_gpu, sizeof(double) * this->size, hipMemcpyDeviceToDevice));
        lorenz96_timestep_direct<<<this->blocks, this->threads>>>(dt * 0.5, this->Xtemp_ensemble_gpu, this->k1_ensemble_gpu);
        lorenz96_tendency<<<this->blocks, this->threads>>>(forcing, this->Xtemp_ensemble_gpu, this->k2_ensemble_gpu);

        // k3 = X'(X_n + k2 * dt/2)
        HIP_ERRCHK(hipMemcpy(this->Xtemp_ensemble_gpu, X_ensemble_gpu, sizeof(double) * this->size, hipMemcpyDeviceToDevice));
        lorenz96_timestep_direct<<<this->blocks, this->threads>>>(dt * 0.5, this->Xtemp_ensemble_gpu, this->k2_ensemble_gpu);
        lorenz96_tendency<<<this->blocks, this->threads>>>(forcing, this->Xtemp_ensemble_gpu, this->k3_ensemble_gpu);

        // k4 = X'(X_n + k3 * dt)
        HIP_ERRCHK(hipMemcpy(this->Xtemp_ensemble_gpu, X_ensemble_gpu, sizeof(double) * this->size, hipMemcpyDeviceToDevice));
        lorenz96_timestep_direct<<<this->blocks, this->threads>>>(dt, this->Xtemp_ensemble_gpu, this->k3_ensemble_gpu);
        lorenz96_tendency<<<this->blocks, this->threads>>>(forcing, this->Xtemp_ensemble_gpu, this->k4_ensemble_gpu);

        // X = X_(n_1) = X_(n) + (k1 + k2*2 + k3*2 + k4) * dt/6
        lorenz96_timestep_runge_kutta<<<this->blocks, this->threads>>>(this->dXdt_ensemble_gpu, this->k1_ensemble_gpu, this->k2_ensemble_gpu, this->k3_ensemble_gpu, this->k4_ensemble_gpu);
        lorenz96_timestep_direct<<<this->blocks, this->threads>>>(dt, X_ensemble_gpu, this->dXdt_ensemble_gpu);
    }

    double* dXdt_ensemble_gpu;
    double* Xtemp_ensemble_gpu;
    double* k1_ensemble_gpu;
    double* k2_ensemble_gpu;
    double* k3_ensemble_gpu;
    double* k4_ensemble_gpu;
};

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
    double *X_ensemble_gpu;

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
    HIP_ERRCHK(hipMemcpy(X_ensemble_gpu, X_ensemble, sizeof(double) * size, hipMemcpyHostToDevice));

    dim3 blocks(ensemble_size);
    dim3 threads(k);

    auto time_step = RungeKutta(k, ensemble_size);

    double t = 0.0;

    while ((t += dt) <= max_time) {
        time_step.time_step(X_ensemble_gpu, dt, forcing);

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

    return 0;
}
