import torch
import math
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# ============================================================
# 0. 固定随机种子
# ============================================================
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# ============================================================
# 1. 编译并加载 CUDA 算子
# ============================================================
flash_module = load(
    name="flash_varlen",
    sources=["baseLineV1.5.cu"],
    extra_cuda_cflags=[
        "-O3",
        "-arch=native",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-Xptxas=-v",
        "-lineinfo",
    ],
    verbose=True,
)

# ============================================================
# 2. 可调参数
# ============================================================
total_tokens = 1024
num_heads = 4
head_dim = 128

# 构造随机 FP16 张量
Q = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.float16, device="cuda")
K = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.float16, device="cuda")
V = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.float16, device="cuda")

# 架构校准: 按 block=128 分块
cu_seqlens = torch.tensor([0, total_tokens], dtype=torch.int32, device="cuda")
cu_seqlen_blocks = torch.tensor([0, total_tokens // 128], dtype=torch.int32, device="cuda")

# total_blocks 传入 CUDA 前向
total_blocks = int(cu_seqlen_blocks[-1].item())

torch.cuda.synchronize()

# ============================================================
# 3. PyTorch reference (manual + SDPA)
# ============================================================
q = Q[:, 0, :].float()
k = K[:, 0, :].float()
v = V[:, 0, :].float()
o_manual = torch.softmax((q @ k.T) / math.sqrt(head_dim), dim=-1) @ v

q_4d = Q[:, 0, :].unsqueeze(0).unsqueeze(0).float()
k_4d = K[:, 0, :].unsqueeze(0).unsqueeze(0).float()
v_4d = V[:, 0, :].unsqueeze(0).unsqueeze(0).float()
o_sdpa = F.scaled_dot_product_attention(q_4d, k_4d, v_4d, dropout_p=0.0, is_causal=False).squeeze(0).squeeze(0)

q_4d_hp = Q[:, 0, :].unsqueeze(0).unsqueeze(0)
k_4d_hp = K[:, 0, :].unsqueeze(0).unsqueeze(0)
v_4d_hp = V[:, 0, :].unsqueeze(0).unsqueeze(0)

# ============================================================
# 4. CUDA forward
# ============================================================
o_cuda = flash_module.forward(Q, K, V, cu_seqlens, cu_seqlen_blocks, total_blocks)
torch.cuda.synchronize()
o_cuda_h0 = o_cuda[:, 0, :].float()

# ============================================================
# 5. correctness
# ============================================================
err_manual = (o_manual - o_cuda_h0).abs()
err_sdpa = (o_sdpa - o_cuda_h0).abs()

print("o_cuda shape: ", o_cuda.shape)
print("CUDA has NaN:", torch.isnan(o_cuda_h0).any().item())
print("manual vs CUDA max err:", err_manual.max().item())
print("SDPA   vs CUDA max err:", err_sdpa.max().item())

if torch.allclose(o_manual, o_cuda_h0, atol=1e-2, rtol=1e-2) and torch.allclose(o_sdpa, o_cuda_h0, atol=1e-2, rtol=1e-2):
    print("✅ V1 forward correctness passed.")
else:
    print("❌ V1 correctness failed.")

# ============================================================
# 6. Benchmark 工具函数
# ============================================================
def benchmark_cuda_event(fn, warmup=50, iters=500, name="kernel"):
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

    avg_ms = start.elapsed_time(end) / iters
    print(f"{name:<28}: {avg_ms:.6f} ms")
    return avg_ms

# ============================================================
# 7. Benchmark
# ============================================================
print(f"shape: seq_len={total_tokens}, num_heads={num_heads}, head_dim={head_dim}")

sdpa_time = benchmark_cuda_event(lambda: F.scaled_dot_product_attention(q_4d_hp, k_4d_hp, v_4d_hp, dropout_p=0.0, is_causal=False), name="PyTorch SDPA (FP16)")
manual_time = benchmark_cuda_event(lambda: torch.softmax((q @ k.T) / math.sqrt(head_dim), dim=-1) @ v, name="PyTorch manual (FP32)")
v1_time = benchmark_cuda_event(lambda: flash_module.forward(Q, K, V, cu_seqlens, cu_seqlen_blocks, total_blocks), name="Custom CUDA V1")

print("\n================ Summary ================")
print(f"PyTorch SDPA (FP16) : {sdpa_time:.6f} ms")
print(f"PyTorch manual (FP32): {manual_time:.6f} ms")
print(f"Custom CUDA V1      : {v1_time:.6f} ms")
print(f"V1 speed vs SDPA    : {sdpa_time / v1_time * 100:.2f}%")