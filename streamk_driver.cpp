#include "half.hpp"
#include "utils.h"
#include "npy.h"

#include <chrono>
#include <fstream>
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

using float16 = half_float::half;

std::vector<float> load_from_npy(const std::string path) {
    npy::npy_data<float> d = npy::read_npy<float>(path);

    std::vector<float> data = d.data;
    std::vector<unsigned long> shape = d.shape;
    bool fortran_order = d.fortran_order;

    return data;
}

std::vector<char> readFileIntoVector(const char *filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);

    if (!file.is_open()) {
    std::cerr << "Unable to open file: " << filename << std::endl;
    return std::vector<char>();
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();
    return buffer;
}

// def streamk_gemm(
//         A (0), B(1), C(2),
//         M(3), N(4), K(5),
//         stride_am(6), stride_ak(7)(not), stride_bk(8)(not), stride_bn(9), stride_cm(10), stride_cn(11)(not),
//         total_full_tiles_streamk(12), total_partial_tiles_streamk(13), iters_per_tile(14),
//         total_tiles_streamk(15), total_programs_streamk(16), ACC_TYPE: tl.constexpr(17)(not),
//         GROUP_SIZE_M: tl.constexpr(18)(not),
//         BLOCK_M: tl.constexpr(19)(not), BLOCK_N: tl.constexpr(20)(not), BLOCK_K: tl.constexpr(21)(not),
// )

// 6 - 8256, 7 - 1, 8 - 1, 9 - 8256, 10 - 4096, 11 - 1

typedef struct {
    void *A;
    void *B;
    void *C;
    int32_t M;
    int32_t N;
    int32_t K;
    int32_t stride_am;
    int32_t stride_bn;
    int32_t stride_cm;
    int32_t total_full_tiles_streamk;
    int32_t total_partial_tiles_streamk;
    int32_t iters_per_tile;
    int32_t total_tiles_streamk;
    int32_t total_programs_streamk;
} streamk_arg_t;
// } __attribute__((packed)) streamk_arg_t;

void run(const char *kernel_name, const char *hsaco_file) {
    streamk_arg_t karg;

    auto A_npy = load_from_npy("/home/nmeganat/triton_kernel_integration/A_mxk_f32.npy");
    auto B_npy = load_from_npy("/home/nmeganat/triton_kernel_integration/B_kxn_f32.npy");

    float *A_f32 = A_npy.data();
    float *B_f32 = B_npy.data();

    karg.M = 4864;
    karg.N = 4096;
    karg.K = 8256;
    karg.stride_am = 8256;
    karg.stride_bn = 8256;
    karg.stride_cm = 4096;
    karg.total_full_tiles_streamk = 258;
    karg.total_partial_tiles_streamk = 0;
    karg.iters_per_tile = 129;
    karg.total_tiles_streamk = 608;
    karg.total_programs_streamk = 304;

    size_t data_byte = 2;
    void *A_half_host = malloc(karg.M * karg.K * data_byte);
    void *B_half_host = malloc(karg.N * karg.K * data_byte);
    void *C_half_host = malloc(karg.M * karg.N * data_byte);

    void *A_half_d, *B_half_d, *C_half_d;

    CHECK_HIP_ERROR(hipMalloc(&A_half_d, static_cast<size_t>(karg.M) * karg.K * data_byte));
    CHECK_HIP_ERROR(hipMalloc(&B_half_d, static_cast<size_t>(karg.N) * karg.K * data_byte));
    CHECK_HIP_ERROR(hipMalloc(&C_half_d, static_cast<size_t>(karg.M) * karg.N * data_byte));

    tensor_copy<float16, float>(static_cast<float16*>(A_half_host), A_f32, static_cast<size_t>(karg.M) * karg.K);
    tensor_copy<float16, float>(static_cast<float16*>(B_half_host), B_f32, static_cast<size_t>(karg.N) * karg.K);

    CHECK_HIP_ERROR(hipMemcpy(A_half_d, A_half_host, static_cast<size_t>(karg.M) * karg.K * data_byte, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(B_half_d, B_half_host, static_cast<size_t>(karg.N) * karg.K * data_byte, hipMemcpyHostToDevice));

    CHECK_HIP_ERROR(hipMemset(C_half_d,
                0, static_cast<size_t>(karg.M) * karg.N * data_byte));

    karg.A = (void *)A_half_d;
    karg.B = (void *)B_half_d;
    karg.C = (void *)C_half_d;

    hipModule_t module;
    hipFunction_t kernelFunc;
    std::vector<char> hsacoVec = readFileIntoVector(hsaco_file);
    if (hipModuleLoadDataEx(&module, hsacoVec.data(), 0, NULL, NULL) !=
        hipSuccess) {
        std::cout << "Failed to load module!\n";
        return;
    }
    if (hipModuleGetFunction(&kernelFunc, module, kernel_name) != hipSuccess) {
        std::cout << "Failed to get function!\n";
        return;
    }

    int warp_size = 64, num_warps = 4, shared_memory = 49152;
    std::vector<size_t> grid_size{304, 1, 1};
    std::vector<size_t> block_size{static_cast<size_t>(warp_size) * num_warps, 1, 1};
    void *params[] = {&karg.A, &karg.B, &karg.C,
                        &karg.M,
                        &karg.N,
                        &karg.K,
                        &karg.stride_am,
                        &karg.stride_bn,
                        &karg.stride_cm,
                        &karg.total_full_tiles_streamk,
                        &karg.total_partial_tiles_streamk,
                        &karg.iters_per_tile,
                        &karg.total_tiles_streamk,
                        &karg.total_programs_streamk};

    CHECK_HIP_ERROR(hipModuleLaunchKernel(kernelFunc, grid_size[0], grid_size[1], grid_size[2], block_size[0], block_size[1], block_size[2], shared_memory, 0, params, 0));

    // size_t arg_size = sizeof(karg);
    // void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, (void *)&karg,
    //             HIP_LAUNCH_PARAM_BUFFER_SIZE, &arg_size,
    //             HIP_LAUNCH_PARAM_END};
    // hipEvent_t start;
    // hipEvent_t stop;
    // CHECK_HIP_ERROR(hipEventCreate(&start));
    // CHECK_HIP_ERROR(hipEventCreate(&stop));
    // float ms = .0;

    // CHECK_HIP_ERROR(hipExtModuleLaunchKernel(
    //     kernelFunc, grid_size[0], grid_size[1], grid_size[2], block_size[0],
    //     block_size[1], block_size[2], 0, 0, NULL, (void **)&config, start, stop));

    // CHECK_HIP_ERROR(hipEventSynchronize(stop));
    // CHECK_HIP_ERROR(hipEventElapsedTime(&ms, start, stop));
    // CHECK_HIP_ERROR(hipEventDestroy(start));
    // CHECK_HIP_ERROR(hipEventDestroy(stop));

    CHECK_HIP_ERROR(hipMemcpy(C_half_host, karg.C, static_cast<size_t>(karg.N) * karg.M * data_byte, hipMemcpyDeviceToHost));

    std::vector<float> op;
    op.reserve(karg.M * karg.N);
    const std::vector<unsigned long> op_shape{static_cast<unsigned long>(karg.M), static_cast<unsigned long>(karg.N)};

    for(int i=0; i < karg.M * karg.N ; i++) {
        // std::cout<<*(((float16*)karg.C)+i) <<", ";
        op[i] = (float)*(((float16*)C_half_host)+i);
        // op[i] = (float)*(((float16*)karg.C)+i);
    }

    const npy::npy_data_ptr<float> op_npy{op.data(), op_shape, false};
    write_npy("/home/nmeganat/triton_kernel_integration/out_driver.npy", op_npy);

    free(A_half_host);
    free(B_half_host);
    free(C_half_host);

    hipFree(A_half_d);
    hipFree(B_half_d);
    hipFree(C_half_d);
}
int main() {

  const char *hsaco_file = "/home/nmeganat/triton_kernel_integration/streamk_gemm.hsaco";
  const char *kernel_name = "streamk_gemm";
  
  run(kernel_name, hsaco_file);

  return 0;
}