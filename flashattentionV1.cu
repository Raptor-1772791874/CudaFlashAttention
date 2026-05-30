#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <iostream>
using  namespace nvcuda;



#define Block_Size 128
#define Head_Dim 128


constexpr int TILE_ELEMS = Block_Size*Head_Dim;




    __global__ void flash_atten_kernel(
        const half* __restrict__ Q,
        const half* __restrict__ K,
        const half* __restrict__ V,
        half* __restrict__ O,
        float* s_dump,
        const int* __restrict__ cu_seqlens,
        const int* __restrict__ cu_seqlens_blocks,
        int total_tokens,
        int num_heads){

            int q_chunk_idx=blockIdx.x; //线程属于第几个切块的包工队（一个包工队负责128个token）
            int  head_idx=blockIdx.z;   //负责哪个头（32之一）
            int tid=threadIdx.x;       //我的工号是多少（负责128个token里的具体哪一个）


            
            //常规的写法是让一个线程负责一个token的1个维度，在此处我们让一个线程负责了一个token的128维度
            //虽然不能玩线程束洗牌，但比洗牌mold时用的FMAX的汇编，1个时钟周期就能吐出结果

            /*  由于N的长度是b个用户的token凑一起的
            所以一个block里可能参杂了两个用户甚至多个
            所以我们必须要查一个前缀和用户token的数组来找到当前线程到底负责哪个用户*/


            int batch_idx =0;
            /*  用2分查找节省时间，此处暂用线性查找后续再优化
            假设前缀和数组是[0,100,350],如果我的起点是150我会找到batch1*/
            while(cu_seqlens_blocks[batch_idx+1]<=q_chunk_idx){
                batch_idx++;
            }
            //找到当前用户的合法边界
            int seq_start =cu_seqlens[batch_idx];//起点为100
            int seq_end =cu_seqlens[batch_idx+1];//终点为350

            //计算我是这个用户的第几个block？
            int relative_chunk_idx=q_chunk_idx-cu_seqlens_blocks[batch_idx];

            //用第几个block乘以大小后这里才是线程负责的token所在的全规格矩阵里第一个token全局起点（此背景是全局偏移且是4096开头）
            //接上，意思是它没有走到自己负责的头的真实地址上还要写偏移公式。
            int global_q_start=seq_start+(relative_chunk_idx*Block_Size);//在Q的全局偏移背景下
            
            //先划定一个sharedMemory大小是多大，128个token128个维度
            
            extern __shared__ half smem[];

        // 2. 架构师的物理切割：手动分配指针偏移量！
        // 128 * 128 = 16384 个 half 数据
         half* s_Q_ptr = smem;                         // Q 从 0 开始
         half* s_K_ptr = smem + 16384;                 // K 从 16384 开始
         half* s_V_ptr = smem + (16384 * 2);           // V 从 32768 开始
            

            //并未在sharedmemory里，这是每个线程自己寄存器里的
            float m_old=-INFINITY; //历史最大王
            float l_old=0.0f;      //历史指数和（目前为0）
            
            //用来装最终的累加器O_local，这个线程负责当前词的128个维度所以开了一个一维数组存
            float o_local[Head_Dim]={0.0f};

            //一个float4任务可以搬16个字节即8个half数据所以除8
            const int f4_stride=Head_Dim/8;

            //直接降落到当前线程负责的token的block里第0个token的开头，具体见笔记
            int Not_sure_index=global_q_start *num_heads+head_idx;//（此背景是Q的全局偏移而非显存全局偏移）

            const float4* Q_f4 = reinterpret_cast<const float4*>(Q + Not_sure_index * Head_Dim);
            float4* s_Q_f4 = reinterpret_cast<float4*>(s_Q_ptr);

            
            float4* s_K_f4 =reinterpret_cast<float4*>(s_K_ptr);
            float4* s_V_f4 =reinterpret_cast<float4*>(s_V_ptr);


            //准备搬运Q进s_Q。再for搬运KV，然后点积

            #pragma unroll
            for(int i=0;i<16;++i){
                //1个token有16个float4要搬一共128token则有2048个任务，一次搬运了128个float4所以16次循环就搬完了，tid是每个线程在当前搬运里的相对偏移
                int index=i*128+tid;//每个线程单独再编号，一次i循环会搬完128个float4，用这个保持偏移的精准
                //index表示当前线程负责的具体哪个float4任务

                int token_offset=index/16;
                int Dim_offset=index%16;
                // 计算当前正在搬运的Token 的全局索引
                int current_global_token = global_q_start + token_offset;//此背景也是全局token数偏移下的
                
                // 越界防线：绝不跨入下一个用户的领地，也不超出当前 seq_end！
                if(current_global_token < seq_end) {
                    
                    // 【核心架构密码：跨越 num_heads 的维度撕裂】
                    // 下一个词的同一个头，物理上隔了整整 num_heads 个头
                    int global_read_idx = token_offset * (num_heads * f4_stride) + Dim_offset;
                    
                    // 执行搬运：全局显存 (跳跃读) -> SRAM (绝对连续写)
                    s_Q_f4[index] = Q_f4[global_read_idx];
                    
                } else {
                    // 越界区域直接填 0，防止脏数据污染后续的点积
                    s_Q_f4[index] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
                
                
            } 
            // for 循环结束

            // 绝对同步屏障：Q 矩阵已经分块全部躺在 SRAM 里了，必须等 128 个人全部搬完！
            __syncthreads();
                

            for(int k_chunk_start=seq_start;k_chunk_start<seq_end;k_chunk_start+=Block_Size){
                //重新寻找KV的开头的地址，globalqstart是大循环外面求的而且有Q的血如果用它每一次小循环转动都会重新计算巨大的地址偏移
                //为了符合基址寄存器折叠我们必须重新给KV算起始
                int KV_global_idx=k_chunk_start*num_heads+head_idx;
                const float4* K_f4=reinterpret_cast<const float4*>(K+KV_global_idx*Head_Dim);
                const float4* V_f4=reinterpret_cast<const float4*>(V+KV_global_idx*Head_Dim);

                //重复一下在搬运Q已经实践过的逻辑，把KV也搬走
                #pragma  unroll
                for(int i=0;i<16;++i){
                    int  index =i*128+tid;//算自己搬运2048个floa4任务里的哪一个（背景是4096偏移），一个token有16个float4任务
                    int token_offset=index/16;//算自己负责哪个token
                    int Dim_offset=index%16;//算自己负责token里的哪个float4任务


                    //查询当前token在数值上是否合法（在用户范围内）
                    int current_K_token =k_chunk_start+token_offset;
                    //合法词直接搬运即可，这里我们让KV直接一起搬运了因为偏移计算的逻辑是一致的直接用即可
                    //一起搬还能掩盖内存延迟，减少了一半的V偏移的乘法计算开销
                    if(current_K_token<seq_end){
                        //计算公式见补充5，与Q一致不再此过多赘述
                        int global_read_idx=token_offset*(num_heads*f4_stride)+Dim_offset;
                        s_K_f4[index]=K_f4[global_read_idx];
                        s_V_f4[index]=V_f4[global_read_idx];

                    }//如果越界还是填0，不允许不填（破坏了矩阵完整性宁愿计算无效值也要保证矩阵乘法时满足法则）
                    else{
                        s_K_f4[index]=make_float4(0.0f ,0.0f ,0.0f ,0.0f);
                        s_V_f4[index]=make_float4(0.0f ,0.0f ,0.0f ,0.0f);

                    }
                }
                //必须统一停下等待全部的sharedmemory被填满拒绝脏读
                __syncthreads();

                //gcc犯蠢会把2维数组的首元素直接当作一整行的大小的偏移，下面把S_Q,S_K重新换成步长为1个half类型否则越界
                //先分配任务，128*128/4为4个64*64，每个warp负责64个Q对K的打分点积。然后就是分时间线分碎块的搬了
                //给128个线程分工，分为4个warp，各自负责最终S128，128里的64，64也就是把S分为一共4个碎片
                int action_S_id =tid/32;
                int action_row=action_S_id/2;
                int action_col=action_S_id%2;

                //计算每个块的起始比如warp0的id为0，row为0，col为0即代表它的矩阵的开始是从Q的0和K的0开始取的
                int action_Q_offset=action_row*64;
                int action_K_offset=action_col*64;



                //正经算S矩阵了这下......

                //先声明FragS在线程寄存器里
                wmma::fragment<wmma::accumulator,16,16,16,float>frag_S[4][4];

                //将存放结果的结构体数组都先初始化为0
                #pragma unroll
                for(int i=0;i<4;++i){
                    #pragma unroll
                    for(int j=0;j<4;j++){
                        wmma::fill_fragment(frag_S[i][j],0.0f);
                    }
                }

                /*为QK各声明4个格子，总大小为64与16（选64是要算64*64每个warp，选16是为了迎合tensor core）
                为什么QK各要4个还是因为tensor core，每个大小为16*16（64/16为4）
                类型为half半精度还是为了迎合tensor core*/


                wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> frag_Q[4];
                wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> frag_K[4];/*虽然K矩阵是行主序存在内存里但是sgemm时用的是转置的（为了满足matrix mutiply的规则
                                                                                       这里0开销仅通过申明列主序类型就完成了假装转置（Tensor Core还是会顺着读但当成一列）*/

                /*时间线推进放在最外面，是因为fragS层面其他格子可以有数据的复用虽然单个fragS格子用不着
                为啥是8次因为一次只能算16个token的16个维度*具体见补_充_2*/
                #pragma unroll
                for(int t_step=0;t_step<8;++t_step){

                   //固定好SRAM大小，为128，也可以直接复用一个头的大小已在宏定义写好改为Head_Dim即可
                    const int Stride_Sram=128;

                    #pragma unroll
                    for(int i=0;i<4;++i){
                        wmma::load_matrix_sync(frag_Q[i],s_Q_ptr+(action_Q_offset+i*16)*Stride_Sram+t_step*16,Stride_Sram);//偏移公式看白皮书的版本1一页
                    }

                    #pragma unroll
                    for(int j=0;j<4;++j){
                        wmma::load_matrix_sync(frag_K[j],s_K_ptr+(action_K_offset+j*16)*Stride_Sram+t_step*16,Stride_Sram);
                    }

                    #pragma unroll
                    for(int i=0;i<4;++i){
                        #pragma unroll
                        for(int j=0;j<4;++j){
                            wmma::mma_sync(frag_S[i][j],frag_Q[i],frag_K[j],frag_S[i][j]);
                        }
                    }

                }


                //V1架构：采用分2批次处理FragS的规约。即Warp01先搬进SK的Smem地址里来

                //先将原本的halfsKptr强转为float型巧妙偷天换日地复用
                float* s_S_reduce=reinterpret_cast<float*>(s_K_ptr);

                //时间线推进2次，由于规约的时候需要全部的维度在场所以我们选择让同一行的一起规约
                
                #pragma unroll
                for(int phase=0;phase<2;++phase){

                    //检查轮到哪个Warp上场
                    //action_row==0 表示Warp01上场   ,action_row==1  表示Warp23上场

                    bool is_my_time =(action_row==phase);

                    if(is_my_time){
                        #pragma unroll
                        for(int i=0;i<4;++i){
                            #pragma unroll
                            for(int j=0;j<4;++j){
                                int local_row=i*16;
                                int local_col=action_col*64+j*16;

                                float* target_ptr=s_S_reduce+local_row*128+local_col;


                                wmma::store_matrix_sync(target_ptr,frag_S[i][j],128,wmma::mem_row_major);
                            }
                        }
                    }
                    //先一步做完的给我停着等其他人全部落齐了才行也就是凑满64，128后才走
                    __syncthreads();


                    //OnlineSoftMax开始，一个线程负责一行的数据


                    if(tid<64){
                int row_idx=tid;  //找行号64的
                float* my_row=s_S_reduce+row_idx*128;

                float row_max=-INFINITY;
                for(int a=0;a<128;++a){
                    row_max=fmax(row_max,my_row[a]);
                }

                float row_sum=0.0f;
                for(int a=0;a<128;++a){
                    float exp_val =expf(my_row[a]-row_max);
                    my_row[a]=exp_val;
                    row_sum+=exp_val;
                }

                int global_row=phase*64+row_idx;
                for(int a=0;a<128;++a){
                    float p_val=my_row[a]/row_sum;
                    s_dump[global_row*128+a]=p_val;
                }
            }
            __syncthreads();

            return;
               }

                
}

               


            





                __syncthreads();
}




            void launch_flash_atten_forward(
    half* Q,half* K,half* V,half* O,
    float* s_dump,
    int* cu_seqlens,
    int* cu_seqlen_blocks,
    int total_tokens,
    int total_blocks,
    int num_heads,
    cudaStream_t stream){

        int smem_size = 3 * 128 * 128 * sizeof(half); 

    // 2. 绝对越权指令：强迫 GPU 解锁 100KB 级别的动态共享内存！
    cudaFuncSetAttribute(
    flash_atten_kernel, 
    cudaFuncAttributeMaxDynamicSharedMemorySize, 
    smem_size
);

        //原本的z轴由b*h混合组成现在彻底踢掉b
        int grid_z=num_heads;

        /*由笛卡尔积下将网格总共需要多少个block计算出来    
        int grid_x=(total_tokens+Block_Size-1)/Block_Size;*/
        int grid_x=total_blocks;
        //但在verlen版中有多少个block早已在main里计算好了
        
        dim3 grid(grid_x,1,grid_z);
        /*不同于推导时的threadidx.x和threadidx.y，我们选择让线程粗化一个线程负责一个token的全流程了。
        不再需要128*128来设计block了（本身编译器也不会允许这么设计），这是线程粗化的好处
        如果是Tensor Core那要改写32，4的形态*/
        dim3 block(Block_Size);

        flash_atten_kernel<<<grid,block,smem_size,stream>>>(Q,K,V,O,s_dump,cu_seqlens,cu_seqlen_blocks,total_tokens,num_heads);
          
        //全部停下来等待
        cudaDeviceSynchronize();

       
    }
   

// 包装器，暴露给 Python
    torch::Tensor forward_debug(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, 
    torch::Tensor cu_seqlens, torch::Tensor cu_seqlen_blocks) 
{
    int total_tokens = Q.size(0);
    int num_heads = Q.size(1);
    
    // 初始化 O 矩阵 (目前用不到，但为了凑齐参数)
    auto O = torch::empty_like(Q);
    // 初始化 S_dump 靶场！大小为 [128, 128]，类型为 float32
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
    auto S_dump = torch::zeros({128, 128}, options);

    // 提取指针
    half* q_ptr = reinterpret_cast<half*>(Q.data_ptr<at::Half>());
    half* k_ptr = reinterpret_cast<half*>(K.data_ptr<at::Half>());
    half* v_ptr = reinterpret_cast<half*>(V.data_ptr<at::Half>());
    half* o_ptr = reinterpret_cast<half*>(O.data_ptr<at::Half>());
    float* s_dump_ptr = S_dump.data_ptr<float>();
    int* seqlens_ptr = cu_seqlens.data_ptr<int>();
    int* blocks_ptr = cu_seqlen_blocks.data_ptr<int>();

    // Varlen 单 Block 极简测试：强行设定只测 1 个 Block
    dim3 grid(1, 1, 1);
    dim3 block(128); // Block_Size = 128
    //定义掉128*128不要用纯数字避免后续暴雷
    int smem_size = 3 * TILE_ELEMS * sizeof(half);
    cudaFuncSetAttribute(
        flash_atten_kernel, 
        cudaFuncAttributeMaxDynamicSharedMemorySize, 
        smem_size
    );

    // 启动机甲
    flash_atten_kernel<<<grid, block,smem_size>>>(
        q_ptr, k_ptr, v_ptr, o_ptr, s_dump_ptr,
        seqlens_ptr, blocks_ptr, total_tokens, num_heads
    );

    cudaDeviceSynchronize();
    return S_dump;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_debug, "FlashAttention Varlen Debug Dump");
}
    
