import torch
import math
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# ============================================================
# 0. 固定硬件随机状态（保证因果一致性）
# ============================================================
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# ============================================================
# 1. 编译并加载 CUDA V2A 算子 (严丝合缝咬合底层物理)
# ============================================================
flash_module = load(
    name="flash_varlen_v2a",
    sources=["test1.cu"],   # ← 确保对应你的最新 V2A 算子文件名
    extra_cuda_cflags=[
        "-O3",
        "-arch=native",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-Xptxas=-v",              # 极限观测物理寄存器分配量与 Spill
        "-lineinfo",
    ],
    verbose=True,
)

# ============================================================
# 2. 拓扑级可调参数
# ============================================================
total_tokens = 4096   # 物理总长
num_heads = 6        # 算子 Y 轴网格步进
head_dim = 128        # wmma 16x16x16 核心基准

block_size = 64       # 【注意】：V2A 物理滑窗完全解耦为 64 tokens/block

assert head_dim == 128, "V2A 架构深层依赖 WMMA 物理对齐"

# ============================================================
# 3. 构造低级显存连续张量
# ============================================================
Q = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.float16, device="cuda")
K = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.float16, device="cuda")
V = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.float16, device="cuda")

# ============================================================
# 4. 指针索引阵地划分 (Var Len 协议)
# ============================================================
cu_seqlens = torch.tensor([0, total_tokens], dtype=torch.int32, device="cuda")

# 物理网格数自动推导公式
num_blocks = (total_tokens + block_size - 1) // block_size
cu_seqlen_blocks = torch.tensor([0, num_blocks], dtype=torch.int32, device="cuda")
total_blocks = int(cu_seqlen_blocks[-1].item())

torch.cuda.synchronize()

# ============================================================
# 5. Reference (PyTorch 纯正多头标准闭环)
# ============================================================
# 转换为标准 4D 格式 [Batch=1, Heads, Tokens, Dim] 喂给 PyTorch
q_4d = Q.transpose(0, 1).unsqueeze(0) # -> [1, num_heads, total_tokens, head_dim]
k_4d = K.transpose(0, 1).unsqueeze(0)
v_4d = V.transpose(0, 1).unsqueeze(0)

# 5.1 手动多头高精度比对底座 (FP32)
q_4d_f32 = q_4d.float()
k_4d_f32 = k_4d.float()
v_4d_f32 = v_4d.float()
scores = (q_4d_f32 @ k_4d_f32.transpose(-2, -1)) / math.sqrt(head_dim)
o_manual_4d = torch.softmax(scores, dim=-1) @ v_4d_f32
# 还原为算子输出相同的 3D Layout -> [total_tokens, num_heads, head_dim]
o_manual = o_manual_4d.squeeze(0).transpose(0, 1).contiguous()

# 5.2 官方极速 F.scaled_dot_product_attention 比对底座
# 【彻底根除类型崩溃】：显式传递关键字参数，阻绝位置参数引发的垃圾比特乱码
o_sdpa_4d = F.scaled_dot_product_attention(
    q_4d, k_4d, v_4d,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False
)
o_sdpa = o_sdpa_4d.squeeze(0).transpose(0, 1).contiguous().float()

# ============================================================
# 6. Launch Custom CUDA V2A Kernel
# ============================================================
o_cuda = flash_module.forward(
    Q, K, V,
    cu_seqlens,
    cu_seqlen_blocks,
    total_blocks
)

torch.cuda.synchronize()
o_cuda_f32 = o_cuda.float()

# ============================================================
# 7. 全量维度精度清算
# ============================================================
err_manual = (o_manual - o_cuda_f32).abs()
err_sdpa = (o_sdpa - o_cuda_f32).abs()

print("================ Correctness ================")
print("o_cuda shape :", o_cuda.shape)
print("CUDA NaN     :", torch.isnan(o_cuda_f32).any().item())

print("manual vs CUDA max err :", err_manual.max().item())
print("SDPA   vs CUDA max err :", err_sdpa.max().item())

# 半精度固有截断误差误差容忍度设为 1e-2
if torch.allclose(o_manual, o_cuda_f32, atol=1e-2, rtol=1e-2) and \
   torch.allclose(o_sdpa, o_cuda_f32, atol=1e-2, rtol=1e-2):
    print("V2A correctness: PASS")
else:
    print("V2A correctness: FAIL")

# ============================================================
# 8. Benchmark Pipeline
# ============================================================
def benchmark_cuda_event(fn, warmup=30, iters=200, name="kernel"):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()

    torch.cuda.synchronize()
    avg = start.elapsed_time(end) / iters

    print(f"{name:<30}: {avg:.6f} ms")
    return avg


print("\n================ Benchmark ================")
print(f"shape: total_tokens={total_tokens}, num_heads={num_heads}, head_dim={head_dim}")

sdpa_time = benchmark_cuda_event(
    lambda: F.scaled_dot_product_attention(q_4d, k_4d, v_4d, attn_mask=None, dropout_p=0.0, is_causal=False),
    name="PyTorch SDPA FP16"
)

manual_time = benchmark_cuda_event(
    lambda: torch.softmax((q_4d_f32 @ k_4d_f32.transpose(-2, -1)) / math.sqrt(head_dim), dim=-1) @ v_4d_f32,
    name="PyTorch Manual FP32"
)

v2a_time = benchmark_cuda_event(
    lambda: flash_module.forward(Q, K, V, cu_seqlens, cu_seqlen_blocks, total_blocks),
    name="Custom CUDA V2A"
)

print("\n================ SUMMARY ================")
print(f"SDPA        : {sdpa_time:.6f} ms")
print(f"Manual      : {manual_time:.6f} ms")
print(f"V2A CUDA    : {v2a_time:.6f} ms")
print(f"Speed ratio : {(sdpa_time / v2a_time) * 100:.2f}%")