#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <math_constants.h>

using namespace nvcuda;

//V2A的Tiled大小是64tokenx128维度
#define Block_Size 64
#define Head_Dim 128

//控制申请的动态内存大小（由token数和头维度在Launch函数里决定）
constexpr int TILE_ELEMS = Block_Size*Head_Dim;



__global__ void flash_atten_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    const int* __restrict__ cu_seqlens,
    const int* __restrict__ cu_seqlens_blocks,
    half* __restrict__ O,
    int num_batches) 
{
    
     /*V2A里，将原先128个token128个维度拆分成64个token128个维度，所以导致原本一个128token的任务被拆分成两个
        我们的选择是加倍Block数，让一个block负责一个任务里的一半（空间换时间，且这两个任务本身不互相依赖）*/

    int q_chunk_idx = blockIdx.x;  //线程属于第几个block（不是每个用户，以全局block来算的）（一个block负责64个token）
    int head_idx = blockIdx.z;     //负责哪个头（32之一）
    int tid = threadIdx.x;        //我的工号是多少（负责64个token里的具体哪一个）

    // 新架构下的 1D Warp拓扑
    // 128个线程分4个Warp，QKV现在只有64行。
    // 每个Warp负责16行的全部列，消除了前几个版本里的错位
    int warp_id = tid / 32;  // 0, 1, 2, 3（算线程属于哪个Warp）
    int lane_id = tid % 32;  // 0 ~ 31 （算线程在warp里的偏移）
    int action_Q_offset = warp_id * 16;   // Q的行偏移，每个Warp负责16行
    /* 彻底消灭掉原来分配不均匀一个warp不负责完整的列导致出现的action_col了 
      无论哪个warp，它算S时都从K的开头读取*/




        /*常规的写法是让一个线程负责一个token的128个维度，在交织排布上会造成严重的访存不合并
        所以我们这里用轴心翻转，让一个线程负责一个维度（最后是一个线程负责一个float4）*/          

        /* 由于N的长度是b个用户的token凑一起的。所以一个block里可能参杂了两个用户甚至多个
        所以我们必须要查一个前缀和用户token的数组来找到当前线程到底负责哪个用户*/
    
   /* 用2分查找去求出每个人在total tokens里负责哪个token以及在totoal blocks里哪个block
      假设前缀和数组是[0,100,350],如果我的起点是150我会找到batch1*/
    int left = 0, right = num_batches - 1, batch_idx = 0;
    while(left <= right){
        // 右移位运算替代除法指令，直接在ALU层面最快速计算
        int mid = left + ((right - left) >> 1);

        if (cu_seqlens_blocks[mid] <= q_chunk_idx) {
            batch_idx = mid;   // 记录当前合法的最大批次索引
            left = mid + 1;    // 继续向右侧显存域逼近
        } else {
            right = mid - 1;   // 收缩左侧显存域
        }
    } //二分查找结束

    //找到当前用户的合法边界
    int seq_start = cu_seqlens[batch_idx];//起点为100
    int seq_end = cu_seqlens[batch_idx+1];//终点为350

    //计算我在这个用户的第几个block里？（以用户的block个数为背景）
    int relative_chunk_idx = q_chunk_idx - cu_seqlens_blocks[batch_idx];
    
    //从用户token的起点跳过前面无关的block（每个block有128个线程）和他们负责的token，来到了线程负责的token（128个数值上），也就是说算线程负责用户的哪个token      
    int global_q_start = seq_start + relative_chunk_idx * Block_Size;
    //简而言之就是在token的数组里算我到底负责哪个block里的token（跳过其他block的token）

    if (global_q_start >= seq_end) return; //越界直接斩杀

    //先划定一个sharedMemory大小是多大，128个token128个维度
    extern __shared__ half smem[]; //只调用声明但不定义，只为了过编译期，等Linker链接
    half* s_Q_ptr = smem;                // Q 从 0 开始
    half* s_K_ptr = smem + 8192;         // K 从 8192 开始
    half* s_V_ptr = smem + 16384;        // V 从 16384 开始

    int num_heads = gridDim.z; //避开传参开销，具体见Launch函数



    //一个float4任务可以搬16个字节即8个half数据所以除8
    const int f4_stride = Head_Dim / 8;



     //准备搬运Q进s_Q。再for搬运K，然后点积算S        
    /*128个线程按行来搬这样访存合并，且一个线程一次不搬一个half，而是一次搬运一个float4任务（一个float4是8个half），整个Q一共64行*16个任务，所以一次即可搬走8行
    所以64*16/128，只需要让128个线程每人搬8次即可完成*/


    // 1. 搬运 Q 矩阵
    //直接降落到当前线程负责的token的block里第0个token的开头，具体见笔记
    int token_offset_q = global_q_start * num_heads + head_idx;  //此背景是Q里的偏移而非显存全局偏移，计算在token的哪个头上）
    const float4* Q_f4 = reinterpret_cast<const float4*>(Q + token_offset_q * Head_Dim);  //读取改为float4型
    float4* s_Q_f4 = reinterpret_cast<float4*>(s_Q_ptr);
    
    // 时间线推进，每次吞8行，在显存里的偏移是多少
    const int q_stride_8_rows = 8 * (num_heads * f4_stride);  //8行。一个token有32个头和本行有16个float4任务

    // 循环外就算出物理基址(8次循环吞噬64行)，不让在循环里一直重复计算循环不变量
    int q_row_base = tid / 16;  // 0~7（计算线程负责的float4任务属于8个token行号里的哪一行）
    int q_col_f4 = tid % 16;    // 0~15（计算线程负责本行16个float4任务的哪一个）

    //算出线程负责的float4任务在Q_f4里的物理偏移 (循环外一次性算死这些循环不变量)
    //跳过前面的行后，偏移个负责的float4大小
    const float4* my_Q_ptr = Q_f4 + q_row_base * (num_heads * f4_stride) + q_col_f4; //最后加上Q的显存地址，就是在Base Q里的偏移
    
    // 再映射到SQ上
    //SQ是我们自己开的不是显存里是交织排布，所以写的逻辑简单不绕
    float4* my_s_Q_ptr = s_Q_f4 + tid;  //从SQPTR开始，一个线程负责放一个float4任务且都是float4指标，所以直接加线程编号即可（这里会有4Way BankConflict，32个Bank同一时钟周期只有128byte，所以会分4批吃下一个warp） 


    #pragma unroll
     //循环8次搬完64行Q矩阵
    for(int i = 0; i < 8; ++i) {

        //保证线程们搬运的token的任务是在当前用户里的合法token数的，每循环一次即表示搬完了8个token力保不能越界                     
        if (global_q_start + q_row_base + i * 8 < seq_end) {

            //如果合法，就把Q地址里的东西解引用拿给SQ里去存储
            *my_s_Q_ptr = *my_Q_ptr;
        } else {
            //越界不合法则直接置放0
            *my_s_Q_ptr = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        my_Q_ptr += q_stride_8_rows;  //往前推进8行在Q里的真实偏移量
        my_s_Q_ptr += 128;  //往前推进128个float4的偏移为下次循环准备
    }

    __syncthreads();  //Q搬运结束

    // 寄存器核心：历史 O 初始化
    float m_prev = -INFINITY;  //先定义出历史的最大行值
    float l_prev = 0.0f;       //定义出历史的行和
    
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

    }//将归一化后的O写回显存结束


} //核函数结束




void launch_flash_atten_forward(
    const half* Q, const half* K, const half* V, half* O,
    const int* cu_seqlens, 
    const int* cu_seqlens_blocks,
    int total_tokens, 
    int total_blocks, 
    int num_heads, 
    int num_batches,
    cudaStream_t stream) 
{
    // 强制征用 48KB 动态共享内存
    int smem_size = 3 * TILE_ELEMS * sizeof(half); 
    
    // 【物理越权】：突破 48KB 硬件静态限制，向 GPU 申请支配权
    cudaFuncSetAttribute(
        flash_atten_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    );

    // 【网格降维映射】
    int grid_x = total_blocks;
    int grid_z = num_heads;
    
    // X轴跑块，Z轴跑头
    dim3 grid(grid_x, 1, grid_z);
    dim3 block(128); // 一维步兵连

    // 【点火执行】：注意这里的 smem_size 和 stream 必须塞入第三、第四个配置位！
    flash_atten_kernel<<<grid, block, smem_size, stream>>>(
        Q, K, V, 
        cu_seqlens, cu_seqlens_blocks, 
        O, num_batches
    );
}


// ============================================================================
// PyTorch C++ 绑定桥梁
// ============================================================================



torch::Tensor forward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor cu_seqlens, torch::Tensor cu_seqlens_blocks,
    int total_blocks)
{
    //  定义O矩阵
    auto O = torch::empty_like(Q);
    int num_batches = cu_seqlens.size(0) - 1; //提取真实的用户批次数
    int num_heads = Q.size(1);
    int total_tokens = Q.size(0); // 取出压扁后的token总长度

    // 抓取PyTorch当前的计算流，防止CPU/GPU异步调度灾难
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // 让C++调度引擎去调用launch函数解耦三者
    launch_flash_atten_forward(
        reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<half*>(O.data_ptr<at::Half>()),
        cu_seqlens.data_ptr<int>(),
        cu_seqlens_blocks.data_ptr<int>(),
        total_tokens,
        total_blocks,
        num_heads,
        num_batches,
        stream
    );

    // 测性能时把这里的CudaDeviceSync注释掉
    // 因为Pythonbenchmark脚本里已经自带了更精准的torch.cuda.Event 异步事件同步
    // cudaDeviceSynchronize(); 

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "V2A Flash Attention Forward");
}