#include <torch/extension.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <math_constants.h>

using namespace nvcuda;

//V2A的Tiled大小是64tokenx128维度
#define Block_Size 64
#define Head_Dim 128

__global__ void flash_atten_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    const int* __restrict__ cu_seqlens,
    const int* __restrict__ cu_seqlens_blocks,
    half* __restrict__ O,
    int num_batches) 
{
    // 网格坐标系
    int q_chunk_idx = blockIdx.x; 
    int head_idx = blockIdx.z; 
    int tid = threadIdx.x;

    // 1D Warp 拓扑
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int action_Q_offset = warp_id * 16; 
    
    // 二分查找锁定用户归属
    int left = 0, right = num_batches - 1, batch_idx = 0;
    while(left <= right){
        int mid = left + ((right - left) >> 1);
        if (cu_seqlens_blocks[mid] <= q_chunk_idx) {
            batch_idx = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    int seq_start = cu_seqlens[batch_idx];
    int seq_end = cu_seqlens[batch_idx+1];
    int relative_chunk_idx = q_chunk_idx - cu_seqlens_blocks[batch_idx];
    
    int global_q_start = seq_start + relative_chunk_idx * Block_Size;
    if (global_q_start >= seq_end) return; 

    // 48KB 物理阵地划分
    extern __shared__ half smem[];
    half* s_Q_ptr = smem;                 // 0 ~ 16KB
    half* s_K_ptr = smem + 8192;          // 16KB ~ 32KB
    half* s_V_ptr = smem + 16384;         // 32KB ~ 48KB

    int num_heads = gridDim.z; 
    const int f4_stride = Head_Dim / 8; // 16

    // 1. 搬运 Q 矩阵
    int token_offset_q = global_q_start * num_heads + head_idx;
    const float4* Q_f4 = reinterpret_cast<const float4*>(Q + token_offset_q * Head_Dim);
    float4* s_Q_f4 = reinterpret_cast<float4*>(s_Q_ptr);
    
    const int q_stride_8_rows = 8 * (num_heads * f4_stride);
    int q_row_base = tid / 16;
    int q_col_f4 = tid % 16;
    const float4* my_Q_ptr = Q_f4 + q_row_base * (num_heads * f4_stride) + q_col_f4;
    float4* my_s_Q_ptr = s_Q_f4 + tid;

    #pragma unroll
    for(int i = 0; i < 8; ++i) {
        if (global_q_start + q_row_base + i * 8 < seq_end) {
            *my_s_Q_ptr = *my_Q_ptr;
        } else {
            *my_s_Q_ptr = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        my_Q_ptr += q_stride_8_rows;
        my_s_Q_ptr += 128;
    }
    __syncthreads();

    // 寄存器核心：历史 O 初始化
    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_O[8];
    #pragma unroll
    for(int j = 0; j < 8; ++j){
        wmma::fill_fragment(frag_O[j], 0.0f);
    }

    float4* s_K_f4 = reinterpret_cast<float4*>(s_K_ptr);
    float4* s_V_f4 = reinterpret_cast<float4*>(s_V_ptr);

    // ========================================================================
    // K_CHUNK 大循环
    // ========================================================================
    for(int k_chunk_start = seq_start; k_chunk_start < seq_end; k_chunk_start += Block_Size) {
        
        int KV_global_idx = k_chunk_start * num_heads + head_idx;
        const float4* K_f4 = reinterpret_cast<const float4*>(K + KV_global_idx * Head_Dim);

        // 2. 搬运 K
        #pragma unroll
        for(int i = 0; i < 8; ++i){
            int index = i * 128 + tid;
            int token_offset = index / 16;
            int Dim_offset = index % 16;
            if(k_chunk_start + token_offset < seq_end){
                int global_read_idx = token_offset * (num_heads * f4_stride) + Dim_offset;
                s_K_f4[index] = K_f4[global_read_idx]; 
            } else {
                s_K_f4[index] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        __syncthreads();

        // 3. 计算 S = Q * K^T
        {   
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_S[4];
            #pragma unroll
            for(int j = 0; j < 4; ++j) wmma::fill_fragment(frag_S[j], 0.0f);

            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_Q;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_K[4];

            #pragma unroll
            for(int t_step = 0; t_step < 8; ++t_step){ 
                wmma::load_matrix_sync(frag_Q, s_Q_ptr + action_Q_offset * 128 + t_step * 16, 128);
                #pragma unroll
                for(int j = 0; j < 4; ++j) {
                    wmma::load_matrix_sync(frag_K[j], s_K_ptr + j * 16 * 128 + t_step * 16, 128);
                }
                #pragma unroll
                for(int j = 0; j < 4; ++j) {
                    wmma::mma_sync(frag_S[j], frag_Q, frag_K[j], frag_S[j]);
                }
            }
            __syncthreads();


            float* s_S_full_buffer = reinterpret_cast<float*>(s_K_ptr); 
            #pragma unroll
            for(int j = 0; j < 4; ++j){
                wmma::store_matrix_sync(s_S_full_buffer + action_Q_offset * 64 + j * 16, frag_S[j], 64, wmma::mem_row_major);
            }
        }   
        __syncthreads();

        // ====================================================================
        // 4. Online Softmax (彻底抛弃 Smem 公告板，回归纯粹的寄存器闭环)
        // ====================================================================
        float* s_S_full_buffer = reinterpret_cast<float*>(s_K_ptr); 
        const float scale_factor = 1.0f / sqrtf(128.0f);
        half p_temp[64]; 
        
        // 【物理真理】：它就在这，一直活在寄存器里，绝不会被 if 吞噬！
        float row_scale = 1.0f; 

        if (tid < 64) {
            float* my_row = s_S_full_buffer + tid * 64; 
            float m_local = -INFINITY;
            
            #pragma unroll 4
            for(int a = 0; a < 64; ++a){
                my_row[a] *= scale_factor;
                m_local = fmaxf(m_local, my_row[a]);
            }

            float m_now = fmaxf(m_prev, m_local);
            
            // 【因果律锁定】：这 64 个线程的物理寄存器已被永久更新
            row_scale = expf(m_prev - m_now); 
            l_prev *= row_scale;

            float row_sum = 0.0f;
            #pragma unroll 4
            for(int a = 0; a < 64; ++a){
                float exp_val = expf(my_row[a] - m_now);
                p_temp[a] = __float2half(exp_val); 
                row_sum += exp_val;
            }

            l_prev += row_sum; 
            m_prev = m_now;    
        }//online soft Max结束
        //必须把sync放在if外，它的意义是让一个block的线程都到齐，但是if里只有64个线程
        __syncthreads(); // 此时，S 矩阵的历史使命彻底结束，可以被覆盖了

        // 写入 P 到 s_K 前 8KB
        if (tid < 64) {
            #pragma unroll 4
            for(int a = 0; a < 64; ++a){
                s_K_ptr[tid * 64 + a] = p_temp[a]; 
            }
        }
        __syncthreads();

        // ====================================================================
        // 5. 历史 O 补偿 (完美的寄存器直接补偿)
        // ====================================================================
        float* s_O_buffer = reinterpret_cast<float*>(s_V_ptr); 

        #pragma unroll
        for(int phase = 0; phase < 2; ++phase){
            bool is_my_time = (warp_id / 2 == phase); 
            
            if(is_my_time){
                #pragma unroll
                for(int j = 0; j < 8; ++j){
                    int local_row = (warp_id % 2) * 16; 
                    int local_col = j * 16;  
                    wmma::store_matrix_sync(s_O_buffer + local_row * 128 + local_col, frag_O[j], 128, wmma::mem_row_major);
                }
            }
            __syncthreads();

            // 完美的因果闭环：
            // tid 在 0~31 时（Phase 0），它用的就是自己刚才在 Softmax 里存下来的 row_scale
            // tid 在 32~63 时（Phase 1），它用的也是自己存下来的 row_scale
            if (tid >= phase * 32 && tid < (phase + 1) * 32) {
                int local_row = tid % 32; 
                float* my_O_row = s_O_buffer + local_row * 128;
                
                #pragma unroll 4
                for(int c = 0; c < 128; ++c){
                    my_O_row[c] *= row_scale; // 直接从寄存器砸向显存，零延迟，零碰撞！
                }
            }
            __syncthreads();

            if(is_my_time){
                #pragma unroll
                for(int j = 0; j < 8; ++j){
                    int local_row = (warp_id % 2) * 16;
                    int local_col = j * 16;
                    wmma::load_matrix_sync(frag_O[j], s_O_buffer + local_row * 128 + local_col, 128, wmma::mem_row_major);
                }
            }
            __syncthreads(); 
        }

        // 6. 搬运 V 
        const float4* V_f4 = reinterpret_cast<const float4*>(V + KV_global_idx * Head_Dim);
        #pragma unroll
        for(int i = 0; i < 8; ++i){
            int index = i * 128 + tid;
            int token_offset = index / 16;
            int Dim_offset = index % 16;

            if(k_chunk_start + token_offset < seq_end){
                int global_read_idx = token_offset * (num_heads * f4_stride) + Dim_offset;
                s_V_f4[index] = V_f4[global_read_idx]; 
            } else {
                s_V_f4[index] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        __syncthreads();

        // 7. 计算 O = P * V
        {   
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_P;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_V[8];

            #pragma unroll
            for(int t_step = 0; t_step < 4; ++t_step){  
                wmma::load_matrix_sync(frag_P, s_K_ptr + action_Q_offset * 64 + t_step * 16, 64); 
                
                #pragma unroll
                for(int j = 0; j < 8; ++j) {
                    wmma::load_matrix_sync(frag_V[j], s_V_ptr + t_step * 16 * 128 + j * 16, 128); 
                }
                
                #pragma unroll
                for(int j = 0; j < 8; ++j) {
                    wmma::mma_sync(frag_O[j], frag_P, frag_V[j], frag_O[j]);
                } 
            } 
        }
        __syncthreads(); 
    }

    // ========================================================================
    // 卸磨杀驴与归一化
    // ========================================================================
    float* s_O_dump = reinterpret_cast<float*>(smem); 
    float* s_l_prev_dump = s_O_dump + (64 * 128); 

    if (tid < 64) {
        s_l_prev_dump[tid] = l_prev;
    }

    #pragma unroll
    for(int j = 0; j < 8; ++j){ 
        int local_row = action_Q_offset; 
        int local_col = j * 16;
        wmma::store_matrix_sync(s_O_dump + local_row * 128 + local_col, frag_O[j], 128, wmma::mem_row_major);
    }
    __syncthreads();

    // 轴心翻转
    #pragma unroll 4
    for (int row = 0; row < 64; ++row) { 
        int current_global_token = global_q_start + row;
        if (current_global_token < seq_end) {
            int global_O_base = current_global_token * (num_heads * Head_Dim) + (head_idx * Head_Dim);
            
            float numerator = s_O_dump[row * 128 + tid];
            float denominator = s_l_prev_dump[row];
            
            half final_val = __float2half(numerator / denominator);
            O[global_O_base + tid] = final_val; 
        }
    }
}

// ============================================================================
// PyTorch C++ 绑定桥梁
// ============================================================================
torch::Tensor forward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor cu_seqlens, torch::Tensor cu_seqlens_blocks,
    int total_blocks)
{
    auto O = torch::empty_like(Q);
    int num_batches = cu_seqlens.size(0) - 1;
    int num_heads = Q.size(1);
    
    dim3 grid(total_blocks, num_heads, 1);
    dim3 block(128);
    
    int smem_size = 48 * 1024;

    flash_atten_kernel<<<grid, block, smem_size>>>(
        reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
        cu_seqlens.data_ptr<int>(),
        cu_seqlens_blocks.data_ptr<int>(),
        reinterpret_cast<half*>(O.data_ptr<at::Half>()),
        num_batches
    );

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "V2A Flash Attention Forward");
}