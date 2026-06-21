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
    
    //O必须在kv循环开始之前就一直存在，且存储于寄存器中
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_O[8];  //初始化FragO时，由于每个warp分到16*128的O矩阵，所以只需要8个格子即可
    
    #pragma unroll
    for(int j = 0; j < 8; ++j){
        wmma::fill_fragment(frag_O[j], 0.0f);
    }

    //SK V也做转型
    float4* s_K_f4 = reinterpret_cast<float4*>(s_K_ptr);
    float4* s_V_f4 = reinterpret_cast<float4*>(s_V_ptr);

    




    // K_CHUNK 大循环
    
    //kv循环无需显式写循环几次，因为这里由start到end就是N个token而一次循环会搬运64个token完成一次直接跳过64个token。
    for(int k_chunk_start = seq_start; k_chunk_start < seq_end; k_chunk_start += Block_Size) {
        
        //重新寻找KV的开头的地址，globalqstart是大循环外面求的而且有Q的血如果用它每一次小循环转动都会重新计算巨大的地址偏移
        //为了符合基址寄存器折叠我们必须重新给KV算起始
        int KV_global_idx = k_chunk_start * num_heads + head_idx;  //计算当前线程在负责的token的哪个头里所以得加上这个头的数字为偏移（在显存排布里，token0的num_heads(一般是32个）个头是排在一起的然后才是token1的32个头所以我们要算它到底在哪个头上）
        const float4* K_f4 = reinterpret_cast<const float4*>(K + KV_global_idx * Head_Dim);  //乘以维度就是真实的内存地址

        // 2. 搬运 K（V2A中要占用SV的内存所以此处不急着把V搬进来）
        #pragma unroll
        for(int i = 0; i < 8; ++i){        //一次搬走了8行所以只循环8次即可搬完
            int index = i * 128 + tid;     //算1024个float4任务里我负责哪个
            int token_offset = index / 16; //算线程负责搬哪个token的搬运任务
            int Dim_offset = index % 16;   //算线程负责本行的哪一个float4任务
            if(k_chunk_start + token_offset < seq_end){   //防止越界
                int global_read_idx = token_offset * (num_heads * f4_stride) + Dim_offset;
                s_K_f4[index] = K_f4[global_read_idx]; 
            } else {  //越界置0
                s_K_f4[index] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }  //K搬运结束

        __syncthreads();  //同步锁



        // 3. 计算 S = Q * K^T （S大小为64*64的float单精度）
        {   //用花括号严格控制寄存器生命周期
            //每个warp分到16*64的S，所以只需要4个格子即可
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_S[4];
            #pragma unroll
            for(int j = 0; j < 4; ++j) wmma::fill_fragment(frag_S[j], 0.0f);//初始化S矩阵

            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_Q;  //用外积所以只需要一个Q
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_K[4]; /*虽然K矩阵是行主序存在内存里但是sgemm时用的是转置的（为了满足矩阵乘法的规则）但还是按行主序理解即可
                                                                                       这里0开销仅通过申明列主序类型就完成了假装转置（Tensor Core还是会顺着读但当成一列）*/

            //Q每次只装16个维度，但由于它要一次性看完64个token（节约访存）所以要给K开4个格子装16维度64token进去
            //时间线推进8次，会算完Q的16token128维度与K的128维度64token点积
            #pragma unroll
            for(int t_step = 0; t_step < 8; ++t_step){  
                
                wmma::load_matrix_sync(frag_Q, s_Q_ptr + action_Q_offset * 128 + t_step * 16, 128);
                
                /*4个warp无论是谁都得完整读完整个K（64token128维度），所以都得从SK开头读
                warp1的16-31的Q也得看完0-15的K和16-63的K*/
                #pragma unroll
                for(int j = 0; j < 4; ++j) {
                    wmma::load_matrix_sync(frag_K[j], s_K_ptr + j * 16 * 128 + t_step * 16, 128);
                }

                #pragma unroll
                for(int j = 0; j < 4; ++j) {
                    wmma::mma_sync(frag_S[j], frag_Q, frag_K[j], frag_S[j]);
                }

            }  //S计算结束

            /*计算结束了但复用了SK的内存当作S存放的地方（满足跨warp时序依赖，地址复用）
            所以这里必须加同步锁，避免有的warp开始往SK里写数据了但还有warp正在读完整的SK*/
            __syncthreads();


            //K的生命周期结束了，后面的计算与他无关，复用SK内存作为S的存放地
            float* s_S_full_buffer = reinterpret_cast<float*>(s_K_ptr); 

            #pragma unroll
            for(int j = 0; j < 4; ++j){
                wmma::store_matrix_sync(s_S_full_buffer + action_Q_offset * 64 + j * 16, frag_S[j], 64, wmma::mem_row_major);
            }
        
        } //算S结束，已放回SK内存地址

        __syncthreads();  //16KB的S矩阵全部保存在SmemK
        




        // 4. Online Softmax 
        float* s_S_full_buffer = reinterpret_cast<float*>(s_K_ptr);  //重新指向SK是因为上一个是在括号里的，出了括号就无作用域了
        const float scale_factor = 1.0f / sqrtf(128.0f);  //清洗数据做梯度消失的防爆处理，避免数值分布极度分散方差过大，刚搬来K就做清洗未必好因为当时是半精度
        half p_temp[64]; //为了避免算出P和把P写回时内存竞争所以这两步必须分开做，所以每行64个P要先放在寄存器里
        
        // 它就在这，一直活在寄存器里就算进入if也会被更改
        float row_scale = 1.0f; //先定义补偿指数

        //只有进来的前64个线程知道每行的补偿row scale是多少（后续补偿历史O会用到
        if (tid < 64) {
            //放在循环里，只有有效的才会计算自己负责哪一行其他64个线程不会开销寄存器
            float* my_row = s_S_full_buffer + tid * 64; // 每人处理一行
            float m_local = -INFINITY;  //先定义出当前最大（用来算64个打分里谁最大）历史最大是prev
            
            //找局部最大值
            #pragma unroll 4
            for(int a = 0; a < 64; ++a){
                my_row[a] *= scale_factor;
                m_local = fmaxf(m_local, my_row[a]);  //本行里最大值
            }

            //Online核心,找到过去和现在谁是新最大值，产生动态补偿
            float m_now = fmaxf(m_prev, m_local); 
            
            // 进if的64个线程的寄存器rowscale已被永久更新
            row_scale = expf(m_prev - m_now);  //算出补偿值（e为底）
            l_prev *= row_scale;               //补偿缩放历史的行和

            float row_sum = 0.0f;           //定义当前行和
            #pragma unroll 4
            for(int a = 0; a < 64; ++a){

                /*即使有了梯度防爆，但极深的神经网络中还是有可能面临大数据。假设S为1000，
             在scale（约为11.3）后约为100，如果不减去最大会直接变为e的100次方，会撑爆数据类型的上限*/
                float exp_val = expf(my_row[a] - m_now); //softMax中的数据保底防爆（打分减去当前行最大值）
                p_temp[a] = __float2half(exp_val);       //先放在寄存器里，避免直接写回造成内存竞争
                row_sum += exp_val;                     //卷走数据累计行和
            }

            l_prev += row_sum;     //把新旧行和累积起来
            m_prev = m_now;        //更新历史行里的最大值

        }  //online soft Max结束
           //必须把sync放在if外，它的意义是让一个block的线程都到齐，但是if里只有64个线程
        __syncthreads(); // 此时，S矩阵的历史使命彻底结束，可以将P写回了




        // 写入P到s_K前8KB（强转half后只有8KB）
        if (tid < 64) {
            #pragma unroll 4
            for(int a = 0; a < 64; ++a){
                s_K_ptr[tid * 64 + a] = p_temp[a]; 
            }
        }
        __syncthreads();//同步。必须等所有线程把数据写完了才走






        // 5. 历史 O 补偿

        //闲置的SV内存(16KB)在此处登场，用来存储然后缩放历史O矩阵，而由于O矩阵此时未归一化且是单精度（32KB）
        //所以被迫重新引入phase分上下半场放入O矩阵做补偿
        
        float* s_O_buffer = reinterpret_cast<float*>(s_V_ptr); 

        #pragma unroll

        //搬运时是本半场的64个线程在搬，补偿时是32个对应行线程在计算
        for(int phase = 0; phase < 2; ++phase){  //上下半场循环开始

            bool is_my_time = (warp_id / 2 == phase); 
            
            //ismytime只控制谁搬运
            if(is_my_time){

                // 吐出历史的O
                //每人负责16*128的历史O，所以搬8次即可
                #pragma unroll
                for(int j = 0; j < 8; ++j){
                    int local_row = (warp_id % 2) * 16; //算自己负责哪16行（warp0得出0，在store时，Tensor Core会以0行0列开始搬数据，warp1是1，Tensor Core会在16行0列开始搬数据
                    //因为上下半场搬运的数据不同了，即使warp02在同一个偏移下找数据，但数据已经变了
                    int local_col = j * 16;             //控制搬运多少次
                    wmma::store_matrix_sync(s_O_buffer + local_row * 128 + local_col, frag_O[j], 128, wmma::mem_row_major);
                }
            }

            __syncthreads();



            /* 每半场只负责32行的数据，而下半场搬运时是warp23在工作
            但是下半场负责的32-63行的数据只有warp1的32个线程知道知道
            上半场同理，只有warp0的32个线程知道rowscale是多少*/

            if (tid >= phase * 32 && tid < (phase + 1) * 32) { //if判断会让前面参与OnlineSoftMax的线程取出本行的rowscale来补偿
                int local_row = tid % 32;  //上下半场会算出一样的偏移（但是在前一个搬运阶段时说数据就不一样了会补偿到正确的行的）
                //即下半场算出的第1行是补偿的第33行（前面吐出历史O时确保了不会补偿错）

                float* my_O_row = s_O_buffer + local_row * 128;
                
                #pragma unroll 4
                for(int c = 0; c < 128; ++c){
                    my_O_row[c] *= row_scale; // 直接从寄存器砸向显存，零延迟，零碰撞
                }
            }

            __syncthreads();  // 必须等每一个字都被补偿完



            //补偿完后写回寄存器里存着
            if(is_my_time){
                #pragma unroll
                for(int j = 0; j < 8; ++j){
                    int local_row = (warp_id % 2) * 16;
                    int local_col = j * 16;
                    wmma::load_matrix_sync(frag_O[j], s_O_buffer + local_row * 128 + local_col, 128, wmma::mem_row_major);
                }
            }

            __syncthreads(); 
        } //历史补偿O结束。下一步算PV。






        // 6. 搬运 V 
        const float4* V_f4 = reinterpret_cast<const float4*>(V + KV_global_idx * Head_Dim);
        #pragma unroll
        for(int i = 0; i < 8; ++i){  // 128人在8轮里就能吃完64x128的V
            int index = i * 128 + tid;  //与K搬运一致
            int token_offset = index / 16;
            int Dim_offset = index % 16;

            if(k_chunk_start + token_offset < seq_end){ //防越界
                int global_read_idx = token_offset * (num_heads * f4_stride) + Dim_offset;
                s_V_f4[index] = V_f4[global_read_idx]; 
            } else {
                s_V_f4[index] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        __syncthreads();  // 必须等V矩阵全员躺在s_V_ptr里






        // 7. 计算 O = P * V
        {   
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_P;  //1鱼8吃减少访存
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_V[8];

            #pragma unroll
            for(int t_step = 0; t_step < 4; ++t_step){  // 为什么t_step只有4步？因为P矩阵的列只有 64 (64/16）
                wmma::load_matrix_sync(frag_P, s_K_ptr + action_Q_offset * 64 + t_step * 16, 64); 
                
                #pragma unroll
                for(int j = 0; j < 8; ++j) {
                    wmma::load_matrix_sync(frag_V[j], s_V_ptr + t_step * 16 * 128 + j * 16, 128); 
                }
                
                /*每个1个P的token要跟V里的全部维度（128列）提取出结果（即一个token对64token的打分
                要提取出自己128维度里的数值，所以让16行token16维度与16行token128列算出16行128列
                再由tstep循环4次得到16行token64打分与64token128维的最终结果*/
                #pragma unroll
                for(int j = 0; j < 8; ++j) {
                    wmma::mma_sync(frag_O[j], frag_P, frag_V[j], frag_O[j]);
                } 

            } 

        }//PV计算结束

        __syncthreads(); //同步锁






    }//Kchunk大循环结束

    
    /* 卸磨杀驴与最终归一化
    复用最初的 Q 阵地，Q的生命周期只存在于算kchunk循环，等待KV循环完之后QKVSP都没用了可以复用内存做最终的归一化了*/
    float* s_O_dump = reinterpret_cast<float*>(smem); 
    float* s_l_prev_dump = s_O_dump + (64 * 128); //从放O的内存往后512bytes，作为lPrev的公告板，每个线程捏着一个本行最终的历史行和

    
    //前64个线程同时交出自己物理寄存器里的l_prev，钉在公告板上，因为要执行翻转轴心了
    if (tid < 64) {
        s_l_prev_dump[tid] = l_prev;
    }



    #pragma unroll
    for(int j = 0; j < 8; ++j){ 

        //把FragO降临在从SQ开始的Smem里，这是最后一次出现在Smem中
        int local_row = action_Q_offset;  //因为是一次性全部倾倒在Smem里所以行也必须加上偏移因为每个warp负责的行不同
        int local_col = j * 16; 
        wmma::store_matrix_sync(s_O_dump + local_row * 128 + local_col, frag_O[j], 128, wmma::mem_row_major);
    }

    __syncthreads();



   //核心馈赠，不再让一个线程负责一行的数据的归一化与写回显存的搬运
  //我们让一个线程负责一列也叫翻转轴心在Smem里bankconflict的惩罚比写回显存时访存不合并小
  // 外层循环去遍历64个行 (row = 0~64)
    #pragma unroll 4
    for (int row = 0; row < 64; ++row) { //表示目前来到哪一行
         // 计算当前行对应的用户里的Token的数值，即当前来到了哪一个token的行里
        int current_global_token = global_q_start + row;


         //不准越界读其他用户的token，越界即斩杀
        if (current_global_token < seq_end) {
            // 算当前头里的Token在显存里的真实偏移
            int global_O_base = current_global_token * (num_heads * Head_Dim) + (head_idx * Head_Dim);
            
        
        /*现在tid不再表示一个线程负责一行了，现在表示一个线程负责一列，128个线程完整的负责了一行里的全部列
        1：根据 tid (列) 和 row (行) 从 Smem 中提取分子*/
            float numerator = s_O_dump[row * 128 + tid];


        //知道了当前线程在哪一行和哪一行的数值还不够，还得知道当前行的历史行总和是多少
        //如果没有前面的写入Smem里共享的话，线程0存行0的行总和但当这个翻转轴心来到行0外时除了当时负责本行的线程根本没人知道这行的行和是多少而且也不能用因为是存在寄存器里的没法直接读也要经过Smem
        //2：从公告板中提取这一行专属的分母
            float denominator = s_l_prev_dump[row];
            
            //本行的每一个数都除以本行的历史行和做归一化并且最终转变为半精度half型
            half final_val = __float2half(numerator / denominator);

        /*翻转轴心的奖励是完美的访存合并
        注意看指针偏移，同一个 Warp里的32个线程，此时row是一样的
        就表示它们的偏移全是global_O_base+tid。
        这意味着这32个线程在写回显存时，地址是绝对连续的 0,1,2...31，是完美的访存合并而原先按照每个线程负责一行时，32个线程是巨大的步长用cpu的话说就是极差的局部性
        GPU里的内存控制器会直接将它们打包成一次完美的32byteBurst写入操作*/
            O[global_O_base + tid] = final_val; 
        }  /将循环的每行O矩阵里的数据写回到显存去






    }//将归一化后的O写回显存结束，完整循环64次即完整搬完64行






} //核函数边界






     //launch端
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
    // 开启动态共享内存过编译期
    int smem_size = 3 * TILE_ELEMS * sizeof(half); 
    
    // 突破48KB硬件静态限制，向GPU直接申请支配权
    cudaFuncSetAttribute(
        flash_atten_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    );



    /*由笛卡尔积下将网格总共需要多少个block计算出来    
    int grid_x=(total_tokens+Block_Size-1)/Block_Size;*/
    int grid_x = total_blocks;  //但在verlen版中有多少个block早已在Python里计算好了

    //原本的z轴由b*h混合组成现在彻底踢掉b
    int grid_z = num_heads; /*gridDim、blockIdx、threadIdx 这些东西
    在GPU硬件底层，是直接刻在流多处理器（SM）的特殊只读寄存器上的*/
    /*举例，当核函数启动时：
    int num_heads = gridDim.z;
    这行代码没有任何访存开销。线程直接从自己身边的硬件内置寄存器里提取了这个数字*/
    
    // 总结：X轴跑块，Z轴跑头



    /*不同于推导时的threadidx.x和threadidx.y，我们选择让线程粗化一个线程负责一个token的全流程了。
        不再需要128*128来设计block了（本身编译器也不会允许这么设计），这是线程粗化的好处
        如果是Tensor Core那要改写32，4的形态*/
    dim3 grid(grid_x, 1, grid_z);  //虽然Y轴闲置但必须得占位
    dim3 block(128); // 每个block人数多少

    // 点火执行注意这里的smem_size和stream必须塞入第三、第四个配置位
    flash_atten_kernel<<<grid, block, smem_size, stream>>>(
        Q, K, V, 
        cu_seqlens, cu_seqlens_blocks, 
        O, num_batches
    );





}


// PyTorch C++ 绑定桥梁




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