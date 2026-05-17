#include <cuda_runtime.h>
#include <mma.h>
#include <iostream>
using  namespace nvcuda::wmma;


wmma::fragment<wmma::accumulator,16,16,16,float>frag_S[4][4];


#define Block_Size 128
#define Head_Dim 128


void launch_flash_atten_forward(
    float* Q,float* K,float* V,float* O,
    int* cu_seqlens,
    int* cu_seqlen_blocks
    int total_tokens,
    int total_blocks
    int num_heads,
    cudaStream_t stream){

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

        flash_atten_kernel<<<grid,block,0,stream>>>(Q,K,V,O,cu_seqlens,cu_seqlen_blocks,total_tokens,num_heads);
          
        //全部停下来等待
        cudaDeviceSynchronize();

       
    }
   //restrict是告诉编译器指针不重叠继续压榨性能

    __global__ void flash_atten_kernel(
        const float* __restrict__ Q,
        const float* __restrict__ K,
        const float* __restrict__ V,
        float* __restrict__ O,
        const int* __restrict__ cu_seqlens,
        const int* __restrict__ cu_seqlens_blocks;
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
            int global_q_start=seq_start+(relative_chunk_idx*Block_Size);
            
            //先划定一个sharedMemory大小是多大，128个token128个维度
            __shared__ float s_Q[Block_Size][Head_Dim];
            __shared__ float s_K[Block_Size][Head_Dim];
            __shared__ float s_V[Block_Size][Head_Dim];
            

            //并未在sharedmemory里，这是每个线程自己寄存器里的
            float m_old=-INFINITY; //历史最大王
            float l_old=0.0f;      //历史指数和（目前为0）
            
            //用来装最终的累加器O_local，这个线程负责当前词的128个维度所以开了一个一维数组存
            float o_local[Head_Dim]={0.0f};

            const int f4_stride=Head_Dim/4;

            //直接降落到当前线程负责的token的block里第0个token的开头，具体见笔记
            int Not_sure_index=global_q_start *num_heads+head_idx;//（此背景也是全局偏移）

            const float4* Q_f4 = reinterpret_cast<const float4*>(Q + Not_sure_index * Head_Dim);
            float4* s_Q_f4 = reinterpret_cast<float4*>(s_Q);

            
            float4* s_K_f4 =reinterpret_cast<float4*>(s_K);
            float4* s_V_f4 =reinterpret_cast<float4*>(s_V);


            //准备搬运Q进s_Q。再for搬运KV，然后点积

            #pragma unroll
            for(int i=0;i<32;++i){
                //128个token有32个float4要搬一共4096次，一次搬运了128个float4，tid是每个线程在当前搬运里的相对偏移
                int index=i*128+tid;//每个线程单独再编号，一次i循环会搬完128个float4，用这个保持偏移的精准
                //index表示当前线程负责的具体哪个float4任务

                int token_offset=index/32;
                int Dim_offset=index%32;
                // 计算当前正在搬运的Token 的全局索引
                int current_global_token = global_q_start + token_offset;//此背景也是全局偏移下的
                
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
            } // for 循环结束

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
                for(int i=0;i<32;++i){
                    int  index =i*128+tid;//算自己搬运4096个floa4任务里的哪一个（背景是4096偏移），一个token有32个float4任务
                    int token_offset=index/32;//算自己负责哪个token
                    int Dim_offset=index%32;//算自己负责token里的哪个float4任务


                    //查询当前token在数值上是否合法（在用户范围内）
                    int current_K_token =k_chunk_start+token_offset;
                    //合法词直接搬运即可，这里我们让KV直接一起搬运了因为偏移计算的逻辑是一致的直接用即可
                    //一起搬还能掩盖内存延迟，减少了一半的V偏移的乘法计算开销
                    if(current_K_token<seq_end){
                        //计算公式见补充5，与Q一致不再此过多赘述
                        int global_read_idx=token_offset*(num_heads*f4_stride)+Dim_offset;
                        s_K_f4[index]=K_f4[global_read_idx];
                        s_V_f4[index]=Q_f4[global_read_idx];

                    }//如果越界还是填0，不允许不填（破坏了矩阵完整性宁愿计算无效值也要保证矩阵乘法时满足法则）
                    else{
                        s_K_f4[index]=make_float4(0.0f ,0.0f ,0.0f ,0.0f);
                        s_V_f4[index]=make_float4(0.0f ,0,0f ,0,0f ,0,0f);

                    }
                }

                //必须统一停下等待全部的sharedmemory被填满拒绝脏读
                __syncthreads();


                //算S矩阵了这下......

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
                wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> frag_K[4];

                /*时间线推进放在最外面，是因为fragS层面其他格子可以有数据的复用虽然单个fragS格子用不着
                为啥是8次因为一次只能算16个token的16个维度*/
                #pragma unroll
                for(int t_step=0;t_step<8;++t_step){

                    #pragma unroll
                    for(int i=0;i<4;++i){
                        wmma::load_matrix_sync(frag_Q[i],s_Q+i*16*stride_Q+t_step*16,stride_Q);
                    }

                    #pragma unroll
                    for(int i=0;i<4;++i){
                        #pragma unroll
                        for(int j=0;j<4;++j){
                            wmma::mma_sync(frag_S[i][j],frag_Q[i],frag_K[j],frag_S[i][j]);
                        }
                    }

                }

                


                __syncthreads();

            }


        
        

    }



    int  main(){



        std::vector<int> h_cu_seqlen_blocks(batchsize+1);
        int  total_blocks = 0;

        for(int b=0;b<batchsize;++b){
            int  seq_len=h_cu_seqlen_blocks[b+1]-h_cu_seqlen_blocks;
            //即使只有一个token也要占据一整个block
            int blocks_for_this_seq=(seq_len+Blcok_Size-1)/Blcok_Size;

            total_blocks+=blocks_for_this_seq;
            h_cu_seqlen_blocks[b+1]=total_blocks;

        }
        //将其copy到GPU显存去变为d_cu_seqlen_blocks.传参时传走

        int* d_cu_seqlens_blocks=nullptr;
        cudaMalloc((void**)&d_cu_seqlens_blocks,(batchsize+1)*sizeof(int));
        //语法问题见Grammar 补充第一页

        cudaMemcpy(d_cu_seqlens_blocks,h_cu_seqlen_blocks.data(),(batchsize+1)*sizeof(int),cudaMemcpyHostToDevice);




        launch_flash_atten_forward(d_Q,d_K,d_V,d_O,d_cu_seqlens,d_cu_seqlens_blocks,total_tokens,total_blocks,num_heads,stream);



        return 0;
    }
