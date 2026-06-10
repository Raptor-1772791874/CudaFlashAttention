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
        int num_heads,
    int num_batches){

            int q_chunk_idx=blockIdx.x; //线程属于第几个block（不是每个用户，以全局block来算的）（一个包工队负责128个token）
            int  head_idx=blockIdx.z;   //负责哪个头（32之一）
            int tid=threadIdx.x;       //我的工号是多少（负责128个token里的具体哪一个）

//原本这段在kchunk内但由于kchunk循环外在最后搬回显存时用到了，所以我移到最开头来避免作用域和生命周期不全
            //先分配任务，128*128/4为4个64*64，每个warp负责64个Q对K的打分点积。然后就是分时间线分碎块的搬了
                //给128个线程分工，分为4个warp，各自负责最终S128，128里的64，64也就是把S分为一共4个碎片
                int action_S_id =tid/32;
                int action_row=action_S_id/2;
                int action_col=action_S_id%2;

                //计算每个块的起始比如warp0的S_id为0，所以row为0，而col也为0即代表它的矩阵的开始是从Q的0和K的0开始取的
                int action_Q_offset=action_row*64;
                int action_K_offset=action_col*64;



            
            //常规的写法是让一个线程负责一个token的1个维度，在此处我们让一个线程负责了一个token的128维度
            //虽然不能玩线程束洗牌，但比洗牌mold时用的FMAX的汇编，1个时钟周期就能吐出结果

            /*  由于N的长度是b个用户的token凑一起的
            所以一个block里可能参杂了两个用户甚至多个
            所以我们必须要查一个前缀和用户token的数组来找到当前线程到底负责哪个用户*/

            int left = 0;
            int right = num_batches - 1; 
            int batch_idx = 0;

        
            /*  用2分查找节省时间
            假设前缀和数组是[0,100,350],如果我的起点是150我会找到batch1*/
            while(left<=right){
                // 右移位运算替代除法指令，直接在 ALU 层面完成降维
                int mid = left + ((right - left) >> 1); 
                
                if (cu_seqlens_blocks[mid] <= q_chunk_idx) {
                    batch_idx = mid;     // 记录当前合法的最大批次索引
                    left = mid + 1;      // 继续向右侧显存域逼近
                } else {
                    right = mid - 1;     // 收缩左侧显存域
                }
            }
            //找到当前用户的合法边界
            int seq_start =cu_seqlens[batch_idx];//起点为100
            int seq_end =cu_seqlens[batch_idx+1];//终点为350

            //计算我是=在这个用户的第几个block里？（以用户的block个数为背景）
            int relative_chunk_idx=q_chunk_idx-cu_seqlens_blocks[batch_idx];

            //从用户token的起点跳过前面无关的block（每个block有128个线程）和他们负责的token，来到了线程负责的token的数值上，也就是说算线程负责用户的哪个token
            int global_q_start=seq_start+(relative_chunk_idx*Block_Size);//在Q的Block级全局偏移背景下。算当前线程负责哪个block里的token
            
            //先划定一个sharedMemory大小是多大，128个token128个维度
            
            extern __shared__ half smem[];

        // 2. 架构师的物理切割：手动分配指针偏移量！
        // 128 * 128 = 16384 个 half 数据
         half* s_Q_ptr = smem;                         // Q 从 0 开始
         half* s_K_ptr = smem + 16384;                 // K 从 16384 开始
         half* s_V_ptr = smem + (16384 * 2);           // V 从 32768 开始
            
           

            //一个float4任务可以搬16个字节即8个half数据所以除8
            const int f4_stride=Head_Dim/8;

            //直接降落到当前线程负责的token的block里第0个token的开头，具体见笔记
            int Not_sure_index=global_q_start *num_heads+head_idx;//（此背景是Q里的偏移而非显存全局偏移，计算在token的哪个头上）

            const float4* Q_f4 = reinterpret_cast<const float4*>(Q + Not_sure_index * Head_Dim);
            float4* s_Q_f4 = reinterpret_cast<float4*>(s_Q_ptr);

            
            float4* s_K_f4 =reinterpret_cast<float4*>(s_K_ptr);
            float4* s_V_f4 =reinterpret_cast<float4*>(s_V_ptr);


            //准备搬运Q进s_Q。再for搬运KV，然后点积

            #pragma unroll
            for(int i=0;i<16;++i){
                //1个token有16个float4要搬一共128token则有2048个任务，一次搬运了128个float4（每人搬1个）所以16次循环就搬完了，tid是每个线程在当前搬运里的相对偏移
                int index=i*128+tid;//每个线程单独再编号，一次i循环会搬完128个float4，用这个保持偏移的精准
                //index表示当前线程负责的具体哪个float4任务

                int token_offset=index/16;
                int Dim_offset=index%16;
                // 计算当前正在搬运的Token 的全局索引
                int current_global_token = global_q_start + token_offset;//此背景也是全局token数偏移下的
                
                // 越界防线：绝不跨入下一个用户的领地，也不超出当前 seq_end！
                if(current_global_token < seq_end) {
                    
                    // 【核心架构密码：跨越 num_heads 的维度撕裂】
                    // 由于布局为标准交织，下一个词的同一个头，物理上隔了整整 num_heads 个头
                    int global_read_idx = token_offset * (num_heads * f4_stride) + Dim_offset;
                    
                    // 执行搬运：全局显存 (跳跃读) -> Smem (绝对连续写)
                    s_Q_f4[index] = Q_f4[global_read_idx];
                    
                } else {
                    // 越界区域直接填 0，防止脏数据污染后续的点积
                    s_Q_f4[index] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
                
                
            } 
            // for 循环结束

            // 绝对同步屏障：Q 矩阵已经分块全部躺在 Smem 里了，必须等 128 个人全部搬完！
            __syncthreads();


            float m_prev = -INFINITY;//先定义出历史的最大行值
            float l_prev = 0.0f;//定义出历史的行和
            //O必须在kv循环开始之前就一直存在，存储于寄存器中
                
            wmma::fragment<wmma::accumulator,16,16,16,float> frag_O[4][4];

               #pragma unroll
               for(int i=0;i<4;++i){
                #pragma unroll
                for(int j=0;j<4;++j){
                    wmma::fill_fragment(frag_O[i][j],0.0f);
                }
               }



               //kv循环无需显式写循环两次，因为这里由start到end就是256个token而一次循环会搬运128个token完成一次直接跳过128个token。
            for(int k_chunk_start=seq_start;k_chunk_start<seq_end;k_chunk_start+=Block_Size){
                
                //重新寻找KV的开头的地址，globalqstart是大循环外面求的而且有Q的血如果用它每一次小循环转动都会重新计算巨大的地址偏移
                //为了符合基址寄存器折叠我们必须重新给KV算起始
                int KV_global_idx=k_chunk_start*num_heads+head_idx;//计算当前线程在负责的token的哪个头里所以得加上这个头的数字为偏移（在显存排布里，token0的num_heads(一般是32个）个头是排在一起的然后才是token1的32个头所以我们要算它到底在哪个头上）
                const float4* K_f4=reinterpret_cast<const float4*>(K+KV_global_idx*Head_Dim);//乘以维度就是真实的内存地址了

                //新架构中只搬运K，把V的内存留出来不急着搬运V等着后面给S的存在复用
                #pragma unroll
         for(int i=0; i<16; ++i){
            int index = i*128 + tid;
            int token_offset = index/16;
            int Dim_offset = index%16;
            if(k_chunk_start + token_offset < seq_end){
                int global_read_idx = token_offset * (num_heads * f4_stride) + Dim_offset;
                s_K_f4[index] = K_f4[global_read_idx]; 
            }   
            else {
                s_K_f4[index] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
              }
          }
         __syncthreads(); // K 阵地准备完毕

    
    // 2计算S，为Q*K转置
    
    {   //严格控制寄存器生命周期】
        wmma::fragment<wmma::accumulator,16,16,16,float> frag_S[4][4];
        #pragma unroll
        for(int i=0; i<4; ++i){
            #pragma unroll
            for(int j=0; j<4; ++j) wmma::fill_fragment(frag_S[i][j], 0.0f);
        }

        wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> frag_Q[4];
        wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> frag_K[4];/*虽然K矩阵是行主序存在内存里但是sgemm时用的是转置的（为了满足matrix mutiply的规则
                                                                                这里0开销仅通过申明列主序类型就完成了假装转置（Tensor Core还是会顺着读但当成一列）*/

        #pragma unroll
        for(int t_step=0; t_step<8; ++t_step){
            #pragma unroll
            for(int i=0; i<4; ++i) wmma::load_matrix_sync(frag_Q[i], s_Q_ptr + (action_Q_offset + i*16)*128 + t_step*16, 128);
            #pragma unroll
            for(int j=0; j<4; ++j) wmma::load_matrix_sync(frag_K[j], s_K_ptr + (action_K_offset + j*16)*128 + t_step*16, 128);
            #pragma unroll
            for(int i=0; i<4; ++i){
                #pragma unroll
                for(int j=0; j<4; ++j) wmma::mma_sync(frag_S[i][j], frag_Q[i], frag_K[j], frag_S[i][j]);
            }
        }

        //K的生命周期结束了，后面的计算与他无关，现在征用SK和SV打通64KB，作为S的存放地
        float* s_S_full_buffer = reinterpret_cast<float*>(s_K_ptr); 
        #pragma unroll
        for(int i=0; i<4; ++i){
            #pragma unroll
            for(int j=0; j<4; ++j){
                int local_row = action_row * 64 + i * 16;
                int local_col = action_col * 64 + j * 16;
                //将S全量存储在SK起的Smem里
                wmma::store_matrix_sync(s_S_full_buffer + local_row * 128 + local_col, frag_S[i][j], 128, wmma::mem_row_major);
            }
        }
    }   //计算S生命周期结束
        __syncthreads(); //64KB的S矩阵全部出生保存在Smem里

    
    // 3做OnlineSoftmax
    //重新指向SK是因为上一个是在括号里的，出了括号就无作用域了
    
    float* s_S_full_buffer = reinterpret_cast<float*>(s_K_ptr); 
    float* my_row = s_S_full_buffer + tid * 128; // 每人处理一行
    const float scale_factor = 1.0f / sqrtf(128.0f);//清洗数据做梯度消失的防爆处理，避免数值分布极度分散方差过大

    //找局部最大值
    float m_local = -INFINITY;
    #pragma unroll 4
    for(int a=0; a<128; ++a){
        my_row[a]*=scale_factor;
        m_local = fmaxf(m_local, my_row[a]);}

    //Online核心,找到过去和现在谁是新王，产生动态补偿
    float m_now = fmaxf(m_prev, m_local);
    float row_scale = expf(m_prev - m_now);//算出补偿指数
    

    //补偿历史的行总和，再计算当前指数与新总和
    l_prev *= row_scale;

    //存放本次的行和
    float row_sum = 0.0f;
    #pragma unroll 4
    for(int a=0; a<128; ++a){
        /*即使有了梯度防爆，但极深的神经网络中还是有可能面临大数据
        假设S为1000，在scale（约为11.3）后约为100，如果不减去最大会直接变为e的100次方，撑爆数据类型的上限*/
        float exp_val = expf(my_row[a] - m_now);//softMax中的数据保底防爆
        my_row[a] = exp_val;//刷新下防爆后的每个数字。这里省去中间变量expval后在某些无优化的情况下会多一条读出Smem的指令和读取
        row_sum += exp_val;//卷走行和
    }

    l_prev += row_sum; //累积新旧行和
    m_prev = m_now;    //更新，到当前K的行最大值
    
    //在baseline original中我们下面是直接把数据写回去SK内存去了，但这会造成数据踩踏内存竞争污染
    //算出P先按半精度存在自己的寄存器里开销64个float型，根写回SK这步分开避免在一起执行时造成内存竞争复写
    half p_temp[128];
    #pragma unroll 4  //减少L1cache存指令开销，底层只会有4个寄存器开销存储SaSS指令
    for(int a=0; a<128; ++a){
        //OnlinesoftMax中不在写回的时候归一化概率（这会将无意义的token的概率拔高到有意义的token的地步上导致整体的相对权重被打乱，所以Online里的每个K块的e指数总和只为了一只累加得到最后整体的总和）
        //只转化为半精度，此时P的大小只有32KB减少了Smem的占用
        p_temp[a] = __float2half(my_row[a]); 
    }
    
    //保证所有线程彻底读完了64KB的S矩阵
    __syncthreads(); 

    //将强转half型的S全部写回SK的32KB内存中 
    #pragma unroll 4
    for(int a=0; a<128; ++a){
        s_K_ptr[tid * 128 + a] = p_temp[a]; 
    }
    __syncthreads(); 

  
    //4闲置的V的Smem地址(32KB)在此处登场，用来存储然后缩放历史O矩阵，而由于O矩阵此时未归一化且是单精度
    //所以被迫重新引入了之前的phase分上下半场放入O矩阵
   
    float* s_O_buffer = reinterpret_cast<float*>(s_V_ptr); 

    #pragma unroll
    //上下半场循环开始
    for(int phase=0; phase<2; ++phase){

        //检查轮到哪个Warp上场
        //action_row==0 表示Warp01上场   ,action_row==1  表示Warp23上场
        bool is_my_time = (action_row == phase);
        
        // 吐出历史的O
        //每人负责64*64的历史O，一个fragment是16*16所以一共16次搬进去。
        if(is_my_time){
            #pragma unroll
            for(int i=0; i<4; ++i){
                #pragma unroll
                for(int j=0; j<4; ++j){
                    int local_row = i * 16; //无论是上半场还是下半场，他们都在一样的内存里倒腾所以不需要加上64行偏移（即下半场也是占满上半场同样位置的64行128列）
                    int local_col = action_col * 64 + j * 16;  //actioncol可以代替专门给从64-127列的线程写一个+64偏移的语句
                    wmma::store_matrix_sync(s_O_buffer + local_row * 128 + local_col, frag_O[i][j], 128, wmma::mem_row_major);
                }
            }
        }
        __syncthreads();

        //缩放
        if(is_my_time){
            //由于分上下半场且我们是让同一actionrow的wrap工作的（即同64行但不同列），所以让一个线程负责1行128个s打分数据
            //当前K块得到的每行缩放指数在求行最大时已经得到且存储在寄存器里 

            //定位到每个线程负责的行的开始，从SV首地址开始算偏移
            float* my_O_row = s_O_buffer + (tid - phase * 64) * 128;//当下半场执行时这里会减去64，是因为下半场也是复用的上半场那64行根本没有什么128行所以扣掉（区分逻辑工号tid和真实行数）
                                                             //用is my time2元对立巧妙的让不在场的warp不工作，但上下半场都用的同一SK地址
            #pragma unroll 4
            for(int c=0; c<128; ++c){
                my_O_row[c]*=row_scale;//补偿历史O矩阵的每行里128个数字
            }
        }
        __syncthreads();//务必等每个数都补偿完

        //写回
        if(is_my_time){
            #pragma unroll
            for(int i=0; i<4; ++i){
                #pragma unroll
                for(int j=0; j<4; ++j){
                    int local_row = i * 16;
                    int local_col = action_col * 64 + j * 16;
                    wmma::load_matrix_sync(frag_O[i][j], s_O_buffer + local_row * 128 + local_col, 128, wmma::mem_row_major);
                }
            }
        }
        __syncthreads(); 
    } //上下半场循环结束
     //历史的O在上下半场下缩放完成。s_V_ptr再次闲置了现在可以搬V算PV了

     

    //5搬运真实的V矩阵进场
    //直接照搬搬运K的逻辑和数值即可
    const float4* V_f4 = reinterpret_cast<const float4*>(V + KV_global_idx * Head_Dim);
    #pragma unroll
    for(int i=0; i<16; ++i){

        int index = i*128 + tid;//算自己搬运2048个floa4任务里的哪一个（背景是4096偏移），一个token有16个float4任务
        int token_offset = index/16;//算自己负责哪个token
        int Dim_offset = index%16;//算自己负责token里的哪个float4任务


        //查询当前token在数值上是否合法（在用户范围内）
        if(k_chunk_start + token_offset < seq_end){
            //合法词直接搬运即可，这里我们让KV直接一起搬运了因为偏移计算的逻辑是一致的直接用即可

            //计算公式见补充5，与Q一致不再此过多赘述
            int global_read_idx = token_offset * (num_heads * f4_stride) + Dim_offset;
            s_V_f4[index] = V_f4[global_read_idx]; 
        } else {
            //如果越界还是填0，不允许不填（破坏了矩阵完整性宁愿计算无效值也要保证矩阵乘法时满足法则）
            s_V_f4[index] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);//空的还是先置0
        }
    }
    __syncthreads();  //确保V完整的存在于SV内存里了

    
      //6计算本次的P*V，其结果直接累加进frag_O（无论多少次算PV，由于硬件底层排列不变所以结果完全可以直接叠加原来的O）
    
    {   //花括号限制生命周期强行压短
        wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> frag_P[4];
        wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> frag_V[4];

        #pragma unroll
        for(int t_step=0; t_step<8; ++t_step){  //与算S时采用的同一种套路让1鱼4吃寄存器级复用，用t_step充当时间线循环8次累加答案，利用外积重叠解决P*V的维度不满128的问题
            #pragma unroll
            //此时 P 永远躺在 s_K_ptr 的最开头！
            for(int i=0; i<4; ++i) wmma::load_matrix_sync(frag_P[i], s_K_ptr + (action_row * 64 + i * 16) * 128 + t_step * 16, 128);
            #pragma unroll
            // V 的读取是128行64列，P是64行128列
            //warp0负责的是0-64列的V，warp1负责的是64-127列的V
            for(int j=0; j<4; ++j) wmma::load_matrix_sync(frag_V[j], s_V_ptr + (t_step * 16 * 128) + (action_col * 64 + j * 16), 128);
            #pragma unroll
            for(int i=0; i<4; ++i){
                #pragma unroll
                for(int j=0; j<4; ++j) wmma::mma_sync(frag_O[i][j], frag_P[i], frag_V[j], frag_O[i][j]);
                //这里是把新算出的O直接累加进历史的O矩阵里也就是线程自己的fragment里的没有挪去其他地方存储
            } 

        } //PV计算结束
        
    }//括号结束严格限制其生命周期不允许多占用寄存器
        
    __syncthreads();  //清理战场，进入下一个KChunk循环

} //kv循环结束
            

 
//终极归一化与卸磨杀驴
//利用 s_O_dump (复用最初的 Q 阵地，Q的生命周期只存在于算S，等待KV循环完之后QKVSP都没用了可以复用内存做最终的归一化了)

float* s_O_dump = reinterpret_cast<float*>(smem); 


//霸占紧挨着的512Bytes作为l_prev的共享公告板。每个线程捏着一个本行的经过历史累加的行总和
//O矩阵是128*128总共16384个float，正好是64KB
float* s_l_prev_dump=s_O_dump+16384; //从完整的O在Smem里空间的末尾插根标，从标后512byte存储每行最终的历史行和（经过历史补偿累加）

//解除寄存器死锁
//128个线程同时交出自己物理寄存器里的l_prev，钉在公告板上，因为要执行翻转轴心了
s_l_prev_dump[tid] = l_prev;


#pragma unroll
for(int i=0; i<4; ++i){
    #pragma unroll
    for(int j=0; j<4; ++j){
        //把FragO降临在从SQ开始的Smem里，这是最后一次出现在Smem中
        int local_row = action_row * 64 + i * 16;//因为是一次性全部倾倒在Smem里所以行也必须加上偏移因为FragO是128行与128列的
        int local_col = action_col * 64 + j * 16;
        wmma::store_matrix_sync(s_O_dump + local_row * 128 + local_col, frag_O[i][j], 128, wmma::mem_row_major);
    }
}
__syncthreads();


//核心馈赠，不再让一个线程负责一行的数据的归一化与写回显存的搬运
//我们让一个线程负责一列也叫翻转轴心在Smem里bankconflict的惩罚比写回显存时访存不合并小
// 外层循环去遍历128个行 (row = 0~127)

#pragma unroll 4
for (int row = 0; row < 128; ++row) {//表示目前来到哪一行
    // 计算当前行对应的用户里的Token的数值，即当前来到了哪一个token的行里
    int current_global_token = global_q_start+row;
    
    //不准越界读其他用户的token，越界即斩杀
    if (current_global_token < seq_end) {
        
        // 算当前头里的Token在显存里的真实偏移
        int global_O_base = current_global_token * (num_heads*Head_Dim)+(head_idx*Head_Dim);//交织布局下一个token要把自己的头全排列后才轮到下一个token的头
        
        // 核心提取
        //现在，tid不再表示一个线程负责一行了，现在表示一个线程负责一列，128个线程完整的负责了一行里的全部列
        // 1：根据 tid (列) 和 row (行) 从 Smem 中提取分子
        float numerator = s_O_dump[row * 128 + tid];//即算出每个线程负责本行的哪一列，行号乘128是算偏移，即行与行之间步长为128个数

        //知道了当前线程在哪一行和哪一行的数值还不够，还得知道当前行的历史行总和是多少
        //如果没有前面的写入Smem里共享的话，线程0存行0的行总和但当这个翻转轴心来到行0外时除了当时负责本行的线程根本没人知道这行的行和是多少而且也不能用因为是存在寄存器里的没法直接读也要经过Smem
      
        // 2：从公告板中提取这一行专属的分母
        float denominator = s_l_prev_dump[row];
        

        //终极归一化与降维：
        //本行的每一个数都除以本行的历史行和做归一化并且最终转变为半精度half型
        half final_val = __float2half(numerator / denominator);
        

        //翻转轴心的奖励是完美的访存合并
        //注意看指针偏移，同一个 Warp里的32个线程，此时row是一样的
        //就表示它们的偏移全是global_O_base+tid。
        //这意味着这32个线程在写回显存时，地址是绝对连续的 0,1,2...31，是完美的访存合并而原先按照每个线程负责一行时，32个线程是巨大的步长用cpu的话说就是极差的局部性
        //GPU里的内存控制器会直接将它们打包成一次完美的32byteBurst写入操作
        O[global_O_base + tid] = final_val;
        }//将循环的每行O矩阵里的数据写回到显存去
   
    }//完整循环128次即完整写完128行
                
}//核函数边界




            void launch_flash_atten_forward(
    half* Q,half* K,half* V,half* O,
    float* s_dump,
    int* cu_seqlens,
    int* cu_seqlen_blocks,
    int total_tokens,
    int total_blocks,
    int num_heads,
    int num_batches,
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

        flash_atten_kernel<<<grid,block,smem_size,stream>>>(Q,K,V,O,s_dump,cu_seqlens,cu_seqlen_blocks,total_tokens,num_heads,num_batches);
          
        //全部停下来等待
       //cudaDeviceSynchronize();

       
    }
   

torch::Tensor forward_debug(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, 
    torch::Tensor cu_seqlens, torch::Tensor cu_seqlen_blocks,int total_blocks) 
{
    int total_tokens = Q.size(0);
    int num_heads = Q.size(1);
    int num_batches = cu_seqlens.size(0) - 1; // 提取真实的用户批次数

    // 初始化 O 矩阵 
    auto O = torch::empty_like(Q);
    
    // 初始化 S_dump 靶场！
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
    auto S_dump = torch::zeros({128, 128}, options);

    // 提取底层物理指针
    half* q_ptr = reinterpret_cast<half*>(Q.data_ptr<at::Half>());
    half* k_ptr = reinterpret_cast<half*>(K.data_ptr<at::Half>());
    half* v_ptr = reinterpret_cast<half*>(V.data_ptr<at::Half>());
    half* o_ptr = reinterpret_cast<half*>(O.data_ptr<at::Half>());
    float* s_dump_ptr = S_dump.data_ptr<float>();
    int* seqlens_ptr = cu_seqlens.data_ptr<int>();
    int* blocks_ptr = cu_seqlen_blocks.data_ptr<int>();

    
    //  从 blocks_ptr 的末尾直接提取全量网格所需的 Block 总数
    //  在你的 256 Token 测试中，num_batches 为 1，blocks_ptr[1] 会精准吐出 2
     

    // 砸碎 (1, 1, 1) 的绑定，让网格随着Token数和Head数横向增加
    dim3 grid(total_blocks, 1, num_heads); 
    dim3 block(Block_Size); // 128

    // 动态共享内存重分配 (3 * 128 * 128 * 2 Bytes = 96KB)
    int smem_size = 3 * TILE_ELEMS * sizeof(half);
    cudaFuncSetAttribute(
        flash_atten_kernel, 
        cudaFuncAttributeMaxDynamicSharedMemorySize, 
        smem_size
    );

    // 启动完全体硬件网格
    flash_atten_kernel<<<grid, block, smem_size>>>(
        q_ptr, k_ptr, v_ptr, o_ptr, s_dump_ptr,
        seqlens_ptr, blocks_ptr, total_tokens, num_heads, num_batches
    );

    // 测性能时把这里的CudaDeviceSync注释掉
    // 因为Pythonbenchmark脚本里已经自带了更精准的torch.cuda.Event 异步事件同步
    // cudaDeviceSynchronize(); 

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_debug, "FlashAttention Varlen Debug Dump");
}