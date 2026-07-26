Varlen Flash Attention CUDA 从零实现
自学 7 个月完成的工业级精度 Flash Attention 前向内核，完整支持变长序列，精度与 PyTorch 原生完全对齐
✅ V2B 终极版（基线实现）
核心功能
完整的 Varlen 变长序列支持（基于 cu_seqlens 前缀和）
基于 WMMA 的 Tensor Core 加速（FP16）
4 个 warp 分块计算 128x128 S 矩阵
float4 向量化全局内存加载
零开销 K 矩阵转置
完整的 Softmax 计算与数值稳定性处理
精度验证（工业级标准）
测试环境：RTX 5090 32GB, CUDA 12.8, PyTorch 2.4
FP16 最大误差：0.00048828125
FP16 平均误差：2.9385e-05
FP32 最大误差：0.0002927184
FP32 平均误差：3.1210e-06
✅ 无 NaN，无 Inf，与 PyTorch 原生 Attention 输出完全对齐
性能基准

实现	         序列长度	头维度	前向耗时	相对 PyTorch SDPA
PyTorch SDPA	128	128	0.014 ms	100%
本项目 V1	128	128	0.062 ms	22.6%
🚀 V1.5（OnlineSoftMax) 正式版（2026.06.10 发布）
核心优化
✅ Online Softmax：消除 S 矩阵全局 SMEM 存储，长序列性能提升 3 倍
✅ 花括号强制寄存器生命周期：寄存器溢出减少 60%
✅ 延迟 V 加载：SMEM 利用率翻倍，从 4 块降至 3 块
✅ 翻转轴心写回：实现 100% 显存访存合并，写回延迟减少 80%
✅ 二分查找批次边界：长序列查找耗时减少 90%
✅ 移除冗余同步：去掉 3 个不必要的全局同步
性能基准

实现	         序列长度	头维度	前向耗时	相对 PyTorch SDPA
PyTorch SDPA	1024	128	0.0334 ms	100%
本项目 V1.5	1024	128	0.0828 ms	40%

快速开始：
路线图
 V1: 单 tile 基线实现 
 V1.5: Online Softmax + 访存优化 + 变长序列支持
 V2A: 64x128 分块 + 重构部分OnlineSoftMax代码逻辑
 V2.5B: 实现寄存器内规约S，以warplevel级原语做蝴蝶洗牌规约历史O矩阵
 V3: 预计做异步拷贝 + 双缓冲
 V4: 反向传播实现
关于我
正在寻找深圳地区CUDA 算子开发 / 大模型推理优化实习岗位。
自学 CUDA 7 个月，从零实现完整 Flash Attention 内核
擅长 GPU 底层优化、Tensor Core 编程、高性能计算
邮箱：fuzhangsheng89@gmail.com / 1772791874@qq.com
GitHub：https://github.com/Raptor-1772791874/CudaFlashAttention

 Varlen Flash Attention CUDA Implementation from Scratch
An industrial-precision forward kernel for Flash Attention completed over 7 months of self-study, with full support for variable-length sequences and numerically identical outputs to native PyTorch implementation.

 ✅ Ultimate V2B Version (Baseline Implementation)
 Core Features
- Full variable-length sequence support via cumulative sequence length prefix sums (`cu_seqlens`)
- Tensor Core acceleration with WMMA intrinsics for FP16 computation
- 128×128 S matrix blocked computation across 4 warps
- Vectorized global memory loading using `float4`
- Zero-overhead transposition of the K matrix
- Complete Softmax computation with numerical stability safeguards

 Industrial-Grade Precision Validation
Test Environment: RTX 5090 32GB, CUDA 12.8, PyTorch 2.4
- FP16 Max Absolute Error: 0.00048828125
- FP16 Mean Absolute Error: 2.9385e-05
- FP32 Max Absolute Error: 0.0002927184
- FP32 Mean Absolute Error: 3.1210e-06

✅ Zero NaNs, zero Infs, fully aligned outputs with PyTorch native Attention

 Performance Benchmark
| Implementation | Seq Length | Head Dim | Forward Latency | Relative to PyTorch SDPA |
|----------------|------------|----------|-----------------|-------------------------|
| PyTorch SDPA   | 128        | 128      | 0.014 ms        | 100%                    |
| Our Project V1 | 128        | 128      | 0.062 ms        | 22.6%                   |

 🚀 V1.5 Official Release (OnlineSoftMax) | Released Jun 10, 2026
 Core Optimizations
- Online Softmax: Eliminates full S matrix storage in shared memory (SMEM), delivering 3× performance boost for long sequences
- Enforced register lifetime via brace scoping: 60% reduction in register spilling
- Lazy V matrix loading: Doubled SMEM utilization, reduced required blocks from 4 to 3
- Axis-flipped writeback scheme: 100% coalesced global memory access, 80% lower write latency
- Binary search for batch boundaries: 90% reduction in long-sequence boundary lookup overhead
- Redundant barrier elimination: Removed 3 unnecessary global thread block synchronizations

 Performance Benchmark
| Implementation | Seq Length | Head Dim | Forward Latency | Relative to PyTorch SDPA |
|----------------|------------|----------|-----------------|-------------------------|
| PyTorch SDPA   | 1024       | 128      | 0.0334 ms       | 100%                    |
| Our Project V1.5 | 1024     | 128      | 0.0828 ms       | 40%                     |

 Quick Start
*(Reserved for subsequent documentation)*

 Roadmap
- V1: Single-tile baseline implementation
- V1.5: Online Softmax + memory access optimization + variable-length sequence support
- V2A: 64×128 tiling + refactored OnlineSoftMax logic
- V2.5B: In-register reduction for S matrix, warp-level butterfly shuffle reduction for historical O matrix
- V3: Planned asynchronous copy + double buffering optimization
- V4: Backward pass implementation

About Me
Looking for full-time positions in Shenzhen in CUDA kernel development / large model inference optimization.
Self-taught CUDA for 5 months, implemented complete Flash Attention kernel from scratch
Specialized in GPU low-level optimization, Tensor Core programming, high-performance computing

Email: fuzhangsheng89@gmail.com / 1772791874@qq.com
GitHub: https://github.com/Raptor-1772791874/CudaFlashAttention



