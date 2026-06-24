#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <vector>
#include <stdexcept>
#include <string>

using namespace nvcuda;

constexpr int M = 16;
constexpr int N = 16;
constexpr int K = 16;
constexpr int BASE = 10000;
constexpr int FRAG_ELEMS = 8;

__device__ __forceinline__ float make_unique_value(int row, int col) {
    return static_cast<float>(BASE + row * 100 + col);
}

__global__ void wmma_fragment_probe_kernel(
    float* dump,
    float* mat_original,
    float* mat_old_group,
    float* mat_new_group
) {
    __shared__ float smem[M * N];

    int tid  = threadIdx.x;
    int lane = tid & 31;

    // 关键修复：
    // 只有 32 个线程，但要初始化 256 个元素，所以必须 stride 初始化。
    for (int idx = tid; idx < M * N; idx += blockDim.x) {
        int row = idx / N;
        int col = idx % N;
        smem[idx] = make_unique_value(row, col);
    }

    __syncthreads();

    // ------------------------------------------------------------
    // 1. 原始 fragment 映射 dump
    // ------------------------------------------------------------
    wmma::fragment<wmma::accumulator, M, N, K, float> frag;
    wmma::load_matrix_sync(frag, smem, N, wmma::mem_row_major);

    // 对 m16n16k16 accumulator float，Ampere 上通常是 8 个元素。
    // 这里显式假设 8，用于和 host 侧 Tensor shape 对齐。
    #pragma unroll
    for (int i = 0; i < FRAG_ELEMS; ++i) {
        dump[lane * FRAG_ELEMS + i] = frag.x[i];
    }

    wmma::store_matrix_sync(mat_original, frag, N, wmma::mem_row_major);

    // ------------------------------------------------------------
    // 2. 旧错误分组污染测试：
    //    group A = x[0..3]
    //    group B = x[4..7]
    // 如果这个分组对应的不是整齐行，store 出来会交错污染。
    // ------------------------------------------------------------
    wmma::fragment<wmma::accumulator, M, N, K, float> frag_old;
    wmma::load_matrix_sync(frag_old, smem, N, wmma::mem_row_major);

    #pragma unroll
    for (int i = 0; i < FRAG_ELEMS; ++i) {
        if (i < 4) {
            frag_old.x[i] += 100000.0f;
        } else {
            frag_old.x[i] += 200000.0f;
        }
    }

    wmma::store_matrix_sync(mat_old_group, frag_old, N, wmma::mem_row_major);

    // ------------------------------------------------------------
    // 3. 新候选分组污染测试：
    //    group A = x[0], x[1], x[4], x[5]
    //    group B = x[2], x[3], x[6], x[7]
    // 如果这个分组正确，对应行应该更干净。
    // ------------------------------------------------------------
    wmma::fragment<wmma::accumulator, M, N, K, float> frag_new;
    wmma::load_matrix_sync(frag_new, smem, N, wmma::mem_row_major);

    frag_new.x[0] += 100000.0f;
    frag_new.x[1] += 100000.0f;
    frag_new.x[4] += 100000.0f;
    frag_new.x[5] += 100000.0f;

    frag_new.x[2] += 200000.0f;
    frag_new.x[3] += 200000.0f;
    frag_new.x[6] += 200000.0f;
    frag_new.x[7] += 200000.0f;

    wmma::store_matrix_sync(mat_new_group, frag_new, N, wmma::mem_row_major);
}

std::vector<torch::Tensor> probe() {
    auto opts = torch::TensorOptions()
        .device(torch::kCUDA)
        .dtype(torch::kFloat32);

    auto dump         = torch::empty({32, FRAG_ELEMS}, opts);
    auto mat_original = torch::empty({M, N}, opts);
    auto mat_old      = torch::empty({M, N}, opts);
    auto mat_new      = torch::empty({M, N}, opts);

    wmma_fragment_probe_kernel<<<1, 32>>>(
        dump.data_ptr<float>(),
        mat_original.data_ptr<float>(),
        mat_old.data_ptr<float>(),
        mat_new.data_ptr<float>()
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("kernel launch failed: ") + cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("kernel execution failed: ") + cudaGetErrorString(err));
    }

    return {dump, mat_original, mat_old, mat_new};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("probe", &probe, "Full WMMA fragment layout probe");
}