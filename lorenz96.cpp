#include <iostream>
#include <fstream>
#include <sstream> 
#include <random>
#include <cstdint>
#include <hip/hip_runtime.h>

#include "cmdparser.hpp"
#include "compress.hpp"
#include "decompress.hpp"

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
    
    const double* const X = &X_ensemble[ensemble_id * k_max];
    double* const dXdt = &dXdt_ensemble[ensemble_id * k_max];

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

    double* const X = &X_ensemble[ensemble_id * k_max];
    const double* const dXdt = &dXdt_ensemble[ensemble_id * k_max];

    X[k] += dXdt[k] * dt;
}

__global__ void lorenz96_timestep_euler_smoothing(double* const Xp0_ensemble, const double* const Xp2_ensemble) {
    int k = threadIdx.x;
    int k_max = blockDim.x;

    int ensemble_id = blockIdx.x;
    int ensemble_size = gridDim.x;

    double* const Xp1 = &Xp0_ensemble[ensemble_id * k_max];
    const double* const Xp0 = &Xp0_ensemble[ensemble_id * k_max];
    const double* const Xp2 = &Xp2_ensemble[ensemble_id * k_max];

    Xp1[k] = (Xp0[k] + Xp2[k]) * 0.5;
}

__global__ void lorenz96_timestep_runge_kutta(double* const dXdt_ensemble, const double* const k1_ensemble, const double* const k2_ensemble, const double* const k3_ensemble, const double* const k4_ensemble) {
    int k = threadIdx.x;
    int k_max = blockDim.x;

    int ensemble_id = blockIdx.x;
    int ensemble_size = gridDim.x;

    double* const dXdt = &dXdt_ensemble[ensemble_id * k_max];
    const double* const k1 = &k1_ensemble[ensemble_id * k_max];
    const double* const k2 = &k2_ensemble[ensemble_id * k_max];
    const double* const k3 = &k3_ensemble[ensemble_id * k_max];
    const double* const k4 = &k4_ensemble[ensemble_id * k_max];

    dXdt[k] = (k1[k] + k2[k]*2.0 + k3[k]*2.0 + k4[k]) / 6.0;
}

__global__ void compress_bitround(const uint64_t mask, const double* const X_ensemble, double* const X_compressed_ensemble) {
    union Binary {
        std::uint64_t u;
        double f;
    };

    int k = threadIdx.x;
    int k_max = blockDim.x;

    int ensemble_id = blockIdx.x;
    int ensemble_size = gridDim.x;

    const double* const X = &X_ensemble[ensemble_id * k_max];
    double* const X_compressed = &X_compressed_ensemble[ensemble_id * k_max];

    X_compressed[k] = Binary { .u = Binary { .f = X[k] }.u & mask }.f; 
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

struct Compressor {
    Compressor(const int k, const int ensemble_size): k(k), ensemble_size(ensemble_size), size(k * ensemble_size) {}
    virtual ~Compressor() {}

    virtual int compress_cpu(const double* const X_ensemble, char* const X_compressed_ensemble) = 0;
    virtual int compress_gpu(const double* const X_ensemble_gpu, char* const X_compressed_ensemble_gpu) = 0;

    virtual int decompress_cpu(double* const X_ensemble, const char* const X_compressed_ensemble) = 0;
    virtual int decompress_gpu(double* const X_ensemble_gpu, const char* const X_compressed_ensemble_gpu) = 0;

    const int k;
    const int ensemble_size;
    const int size;
};

struct Zfp: Compressor {
    Zfp(const int k, const int ensemble_size, const double fixed_rate): Compressor(k, ensemble_size), fixed_rate(fixed_rate) {

    }

    ~Zfp() {

    }

    int compress_cpu(const double* const X_ensemble, char* const X_compressed_ensemble) {
        int compressed_bytes_total = 0;        

        for (int i = 0; i < ensemble_size; i++) {
            int compressed_bytes = compress(&X_ensemble[i*this->k], this->k, &X_compressed_ensemble[compressed_bytes_total], this->fixed_rate, 1);
            if (compressed_bytes <= 0) {
                std::cout << "ZFP CPU compression failed" << std::endl;
                return 0;
            }
            compressed_bytes_total += compressed_bytes;
        }

        return compressed_bytes_total;
    }

    int compress_gpu(const double* const X_ensemble_gpu, char* const X_compressed_ensemble_gpu) {
        int compressed_bytes_total = 0;

        for (int i = 0; i < ensemble_size; i++) {
            int compressed_bytes = compress(&X_ensemble_gpu[i*this->k], this->k, &X_compressed_ensemble_gpu[compressed_bytes_total], this->fixed_rate, 2);
            if (compressed_bytes <= 0) {
                std::cout << "ZFP GPU compression failed" << std::endl;
                return 0;
            }
            compressed_bytes_total += compressed_bytes;
        }

        return compressed_bytes_total;
    }

    int decompress_cpu(double* const X_ensemble, const char* const X_compressed_ensemble) {
        int decompressed_bytes_total = 0;

        for (int i = 0; i < ensemble_size; i++) {
            //int decompressed_bytes = decompress(&X_ensemble[i*this->k], this->k, &X_compressed_ensemble[decompressed_bytes_total], this->fixed_rate, 1);
            int decompressed_bytes = decompress(&X_ensemble[i*this->k], this->k, &X_compressed_ensemble[i*this->k*sizeof(double)*2], this->fixed_rate, 1);
            if (decompressed_bytes <= 0) {
                std::cout << "ZFP CPU decompression failed" << std::endl;
                return 0;
            }
            decompressed_bytes_total += decompressed_bytes;
        }

        return decompressed_bytes_total;
    }

    int decompress_gpu(double* const X_ensemble_gpu, const char* const X_compressed_ensemble_gpu) {
        int decompressed_bytes_total = 0;

        for (int i = 0; i < ensemble_size; i++) {
            //int decompressed_bytes = decompress(&X_ensemble_gpu[i*this->k], this->k, &X_compressed_ensemble_gpu[decompressed_bytes_total], this->fixed_rate, 2);
            int decompressed_bytes = decompress(&X_ensemble_gpu[i*this->k], this->k, &X_compressed_ensemble_gpu[i*this->k*sizeof(double)*2], this->fixed_rate, 2);
            if (decompressed_bytes <= 0) {
                std::cout << "ZFP GPU decompression failed" << std::endl;
                return 0;
            }
            decompressed_bytes_total += decompressed_bytes;
        }

        return decompressed_bytes_total;
    }

    const double fixed_rate;
};

struct BitRound: Compressor {
    BitRound(const int k, const int ensemble_size, const int bits): Compressor(k, ensemble_size), mask(~((1ULL << (52 - bits)) - 1)) {

    }

    ~BitRound() {

    }

    int compress_cpu(const double* const X_ensemble, char* const X_compressed_ensemble) {
        union Binary {
            std::uint64_t u;
            double f;
        };

        double* const X_compressed_ensemble_f = reinterpret_cast<double*>(X_compressed_ensemble);

        for (int i = 0; i < size; ++i) {
            X_compressed_ensemble_f[i] = Binary { .u = Binary { .f = X_ensemble[i] }.u & this->mask }.f;
        }

        return this->size * sizeof(double);
    }

    int compress_gpu(const double* const X_ensemble_gpu, char* const X_compressed_ensemble_gpu) {
        dim3 blocks(this->ensemble_size);
        dim3 threads(this->k);

        double* const X_compressed_ensemble_gpu_f = reinterpret_cast<double*>(X_compressed_ensemble_gpu);

        compress_bitround<<<blocks, threads>>>(this->mask, X_ensemble_gpu, X_compressed_ensemble_gpu_f);

        return this->size * sizeof(double);
    }

    int decompress_cpu(double* const X_ensemble, const char* const X_compressed_ensemble) {
        HIP_ERRCHK(hipMemcpy(X_ensemble, X_compressed_ensemble, sizeof(double) * this->size, hipMemcpyHostToHost));

        return this->size * sizeof(double);
    }

    int decompress_gpu(double* const X_ensemble_gpu, const char* const X_compressed_ensemble_gpu) {
        HIP_ERRCHK(hipMemcpy(X_ensemble_gpu, X_compressed_ensemble_gpu, sizeof(double) * this->size, hipMemcpyDeviceToDevice));

        return this->size * sizeof(double);
    }

    const uint64_t mask;
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
    parser.set_optional<double>("d", "dt", 0.01);
    parser.set_optional<double>("f", "forcing", 8.0);
    parser.set_optional<int>("k", "", 36);
    parser.set_optional<int>("e", "ensemble-size", 11);
    parser.set_optional<std::string>("o", "output", "state");
    parser.set_optional<int>("s", "seed", 42);
    parser.set_optional<double>("p", "ensemble-perturbation", 0.001);
    parser.set_optional<double>("r", "zfp-fixed-rate", 64.25);
    parser.set_optional<int>("c", "compression-frequency", -1);
    parser.set_optional<double>("w", "warmup-time", 0.0);
    parser.set_optional<int>("b", "bitround-bits", 52);
    parser.set_optional<std::string>("a", "compression-algorithm", "zfp");
}

int main(int argc, char *argv[])
{
    cli::Parser parser(argc, argv);
    configure_cli(parser);
    parser.run_and_exit_if_error();
    
    double max_time = parser.get<double>("t");
    double dt = parser.get<double>("d");
    double forcing = parser.get<double>("f");
    int k = parser.get<int>("k");
    int ensemble_size = parser.get<int>("e");
    std::string output = parser.get<std::string>("o");
    int seed = parser.get<int>("s");
    double ensemble_perturbation_stdv = parser.get<double>("p");
    double zfp_fixed_rate = parser.get<double>("r");
    int compression_frequency = parser.get<int>("c");
    double warmup_time = parser.get<double>("w");
    int bitround_bits = parser.get<int>("b");
    std::string compression_algorithm = parser.get<std::string>("a");

    if (dt <= 0.0) {
        std::cout << "dt must be a positive number" << std::endl;
        return 1;
    }

    if (forcing < 0.0) {
        std::cout << "the forcing must be non-negative" << std::endl;
        return 1;
    }

    if (k < 4) {
        std::cout << "the Lorenz96 model requires k >= 4" << std::endl;
        return 1;
    }

    if (ensemble_size < 1) {
        std::cout << "the ensemble size must be positive" << std::endl;
        return 1;
    }

    if ((ensemble_size % 2) != 1) {
        std::cout << "the ensemble-size must be an odd integer" << std::endl;
        return 1;
    }

    if (seed < 0) {
        std::cout << "the seed must be non-negative" << std::endl;
        return 1;
    }

    if (ensemble_perturbation_stdv < 0.0) {
        std::cout << "the ensemble member perturbation must be non-negative" << std::endl;
        return 1;
    }

    if (zfp_fixed_rate <= 0.0) {
        std::cout << "ZFP's fixed-rate must be positive" << std::endl;
        return 1;
    }

    if (warmup_time < 0.0) {
        std::cout << "warmup time must be non-negative" << std::endl;
        return 1;
    }

    if (bitround_bits < 0) {
        std::cout << "BitRound bits must not be negative" << std::endl;
        return 1;
    }

    if (bitround_bits > 52) {
        std::cout << "BitRound bits must not exceed the mantissa size of 52" << std::endl;
        return 1;
    }

    if (compression_algorithm != "zfp" && compression_algorithm != "bitround") {
        std::cout << "Unknown compression algorithm '" << compression_algorithm << "'" << std::endl;
        return 1;
    }

    std::cout << "Lorenz96(k=" << k << ", F=" << forcing << ", dt=" << dt << ", t_max=" << max_time << ")" << std::endl;
    if (warmup_time > 0.0) {
        std::cout << " - warming up the simulation for " << warmup_time << std::endl;
    }
    std::cout << " - running ensemble of size " << ensemble_size << " with initial perturbation N(0.0, " << ensemble_perturbation_stdv << ")" << std::endl;

    if (compression_frequency < 0) {
        std::cout << " - without compression" << std::endl;
    } else if (compression_frequency == 0) {
        std::cout << " - compressing every output on the CPU with ";
    } else {
        std::cout << " - compressing every " << compression_frequency << "-th model state online on the GPU with ";
    }

    if (compression_frequency >= 0) {
        if (compression_algorithm == "zfp") {
            std::cout << "ZFP(fixed_rate=" << zfp_fixed_rate << ")";
        } else if (compression_algorithm == "bitround") {
            std::cout << "BitRound(bitround_bits=" << bitround_bits << ")";
        }

        std::cout << std::endl;
    }
    
    std::cout << " - saving output files to '" << output << "_[i]' for i in 0.." << ensemble_size << std::endl << std::endl;

    {
        std::ofstream config_file;
        std::stringstream file_name;
        file_name << output << "_config.json";
        config_file.open(file_name.str(), std::ios::out | std::ios::trunc);
        
        config_file << "{ ";
        config_file << "\"max_time\": " << max_time << ", ";
        config_file << "\"dt\": " << dt << ", ";
        config_file << "\"forcing\": " << forcing << ", ";
        config_file << "\"k\": " << k << ", ";
        config_file << "\"ensemble_size\": " << ensemble_size << ", ";
        config_file << "\"output\": \"" << output << "\", ";
        config_file << "\"seed\": " << seed << ", ";
        config_file << "\"ensemble_perturbation\": " << ensemble_perturbation_stdv << ", ";
        config_file << "\"zfp_fixed_rate\": " << zfp_fixed_rate << ", ";
        config_file << "\"compression_frequency\": " << compression_frequency << ", ";
        config_file << "\"warmup_time\": " << warmup_time << ", ";
        config_file << "\"bitround_bits\": " << bitround_bits << ", ";
        config_file << "\"compression_algorithm\": \"" << compression_algorithm << "\"";
        config_file << " }" << std::endl;
        
        config_file.close();
    }

    int size = k * ensemble_size;

    double X_ensemble[size];
    char X_compressed[sizeof(double) * size * 2];
    double *X_ensemble_gpu;
    char *X_compressed_gpu;

    HIP_ERRCHK(hipMalloc(&X_ensemble_gpu, sizeof(double) * size));
    HIP_ERRCHK(hipMalloc(&X_compressed_gpu, sizeof(double) * size * 2));

    Compressor *compressor;

    if (compression_algorithm == "zfp") {
        compressor = new Zfp(k, ensemble_size, zfp_fixed_rate);
    } else if (compression_algorithm == "bitround") {
        compressor = new BitRound(k, ensemble_size, bitround_bits);
    }

    // Initialise the initial state
    for (int i = 0; i < k; i++) {
        X_ensemble[i] = 0.0;
    }
    X_ensemble[0] = 1.0;

    auto time_step_warm = RungeKutta(k, 1);
    double t_warm = 0.0;

    if (warmup_time > 0.0) {
        HIP_ERRCHK(hipMemcpy(X_ensemble_gpu, X_ensemble, sizeof(double) * k, hipMemcpyHostToDevice));

        while ((t_warm += dt) <= warmup_time) {
            time_step_warm.time_step(X_ensemble_gpu, dt, forcing);
        }

        HIP_ERRCHK(hipMemcpy(X_ensemble, X_ensemble_gpu, sizeof(double) * k, hipMemcpyDeviceToHost));
    }

    // Copy the initial state to the other ensemble members
    for (int i = 1; i < ensemble_size; i++) {
        for (int j = 0; j < k; j++) {
            X_ensemble[k*i + j] = X_ensemble[j];
        }
    }

    std::mt19937 rng;
    rng.seed(seed);

    std::normal_distribution<double> ensemble_perturbation(0.0, ensemble_perturbation_stdv);

    // Initialise the perturbations, keep the first ensemble member perfectly centred
    for (int i = 0; i < (ensemble_size / 2); i++) {
        for (int j = 0; j < k; j++) {
            double p = ensemble_perturbation(rng);

            X_ensemble[(1 + i*2 + 0)*k + j] += p;
            X_ensemble[(1 + i*2 + 1)*k + j] -= p;
        }
    }

    if (compression_frequency > 0) {
        if (compressor->compress_cpu(X_ensemble, X_compressed) == 0) {
            std::cout << compression_algorithm << " compression failed" << std::endl;
            return 1;
        }
        if (compressor->decompress_cpu(X_ensemble, X_compressed) == 0) {
            std::cout << compression_algorithm << " decompression failed" << std::endl;
            return 1;
        }
    }

    HIP_ERRCHK(hipMemcpy(X_ensemble_gpu, X_ensemble, sizeof(double) * size, hipMemcpyHostToDevice));

    std::ofstream out_files[ensemble_size];
    for (int i = 0; i < ensemble_size; i++) {
        std::stringstream file_name;
        file_name << output << "_" << i;
        out_files[i].open(file_name.str(), std::ios::out | std::ios::trunc | std::ios::binary);
    }

    if (compression_frequency == 0) {
        if (compressor->compress_cpu(X_ensemble, X_compressed) == 0) {
            std::cout << compression_algorithm << " compression failed" << std::endl;
            return 1;
        }
        if (compressor->decompress_cpu(X_ensemble, X_compressed) == 0) {
            std::cout << compression_algorithm << " decompression failed" << std::endl;
            return 1;
        }
    }

    std::cout << "Initial state:" << std::endl;
    print_state(X_ensemble, k, 0.0);
    std::cout << std::endl;

    for (int i = 0; i < ensemble_size; i++) {
        for (int j = 0; j < k; j++) {
            out_files[i].write(reinterpret_cast<const char*>(&X_ensemble[i*k+j]), sizeof(X_ensemble[i*k+j]));
        }
    }

    dim3 blocks(ensemble_size);
    dim3 threads(k);

    auto time_step = RungeKutta(k, ensemble_size);

    double t = 0.0;
    int step = 0;

    while ((t += dt) <= max_time) {
        step += 1;

        time_step.time_step(X_ensemble_gpu, dt, forcing);

        if ((compression_frequency > 0) && ((step % compression_frequency) == 0)) {
            if (compressor->compress_gpu(X_ensemble_gpu, X_compressed_gpu) == 0) {
                std::cout << compression_algorithm << " compression failed" << std::endl;
                return 1;
            }
            if (compressor->decompress_cpu(X_ensemble_gpu, X_compressed_gpu) == 0) {
                std::cout << compression_algorithm << " decompression failed" << std::endl;
                return 1;
            }
        }

        HIP_ERRCHK(hipMemcpy(X_ensemble, X_ensemble_gpu, sizeof(double) * size, hipMemcpyDeviceToHost));

        if (compression_frequency == 0) {
            if (compressor->compress_cpu(X_ensemble, X_compressed) == 0) {
                std::cout << compression_algorithm << " compression failed" << std::endl;
                return 1;
            }
            if (compressor->decompress_cpu(X_ensemble, X_compressed) == 0) {
                std::cout << compression_algorithm << " decompression failed" << std::endl;
                return 1;
            }
        }

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

    delete compressor;

    HIP_ERRCHK(hipFree(X_ensemble_gpu));
    HIP_ERRCHK(hipFree(X_compressed_gpu));

    return 0;
}
