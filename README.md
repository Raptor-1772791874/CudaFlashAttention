# Varlen Flash Attention V2 CUDA从0实现
✅ 已完成功能（V1 终极版）
 完整的 Varlen 变长序列支持（基于 cu_seqlens 前缀和）
 基于 WMMA 的 Tensor Core 加速（FP16）
 4 个 warp 分块计算 128x128 S 矩阵
 float4 向量化全局内存加载
 零开销 K 矩阵转置
 完整的 Softmax 计算与数值稳定性处理
📊 精度验证（工业级标准）
测试环境：5090 32GB，CUDA 12.8，PyTorch 2.4
FP16 最大误差：0.00048828125
FP16 平均误差：2.9385e-05
FP32 最大误差：0.0002927184
FP32 平均误差：3.1210e-06
✅ 无 NaN，无 Inf，与 PyTorch 原生 Attention 输出完全对齐


邮箱：fuzhangsheng89@gmail.com/1772791874@qq.com

