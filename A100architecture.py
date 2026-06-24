import os
import json
import torch
from torch.utils.cpp_extension import load

BASE = 10000
FRAG_ELEMS = 8

torch.set_printoptions(linewidth=220, threshold=10000)

device_name = torch.cuda.get_device_name()
cap = torch.cuda.get_device_capability()
torch_cuda = torch.version.cuda
torch_ver = torch.__version__

opt_flag = os.environ.get("WMMA_PROBE_OPT", "-O3")
opt_tag = opt_flag.replace("-", "").replace("=", "_")

ext_name = f"wmma_probe_full_sm{cap[0]}{cap[1]}_{opt_tag}"

print("=== Build info ===")
print(f"GPU           : {device_name}")
print(f"Capability    : sm_{cap[0]}{cap[1]}")
print(f"Torch         : {torch_ver}")
print(f"Torch CUDA    : {torch_cuda}")
print(f"Compile opt   : {opt_flag}")
print(f"Extension name: {ext_name}")

mod = load(
    name=ext_name,
    sources=["A100architecture.cu"],    
    extra_cuda_cflags=[
        opt_flag,
        "-lineinfo",
        "-Xptxas=-v",
    ],
    verbose=True,
)

dump, mat_original, mat_old, mat_new = mod.probe()
torch.cuda.synchronize()

dump = dump.cpu()
mat_original = mat_original.cpu()
mat_old = mat_old.cpu()
mat_new = mat_new.cpu()

# ------------------------------------------------------------
# 1. 检查 store_matrix_sync 是否完整正确
# ------------------------------------------------------------
expected = torch.empty((16, 16), dtype=torch.float32)
for r in range(16):
    for c in range(16):
        expected[r, c] = BASE + r * 100 + c

max_abs_err = (mat_original - expected).abs().max().item()

print("\n=== store_matrix_sync 原始 16x16 矩阵 ===")
print(mat_original.to(torch.int32))

print("\n=== original matrix check ===")
print(f"max_abs_err = {max_abs_err}")

if max_abs_err != 0:
    print("❌ 原始 store_matrix_sync 矩阵不对，先别分析 fragment layout。")
else:
    print("✅ 原始 store_matrix_sync 矩阵完整正确，可以分析 fragment layout。")


# ------------------------------------------------------------
# 2. 解码 dump 值
# ------------------------------------------------------------
def decode(v: float):
    iv = int(round(float(v)))
    raw = iv - BASE
    if raw < 0:
        return None
    row = raw // 100
    col = raw % 100
    if not (0 <= row < 16 and 0 <= col < 16):
        return None
    return row, col


print("\n=== lane / x[i] -> logical(row, col) ===")

mapping = {}
for lane in range(32):
    xs = []
    mapping[lane] = {}
    for i in range(FRAG_ELEMS):
        rc = decode(dump[lane, i])
        mapping[lane][i] = rc
        if rc is None:
            xs.append(f"x[{i}]=INVALID({float(dump[lane, i]):.1f})")
        else:
            r, c = rc
            xs.append(f"x[{i}]=({r:2d},{c:2d})")
    print(f"lane {lane:2d}: " + "  ".join(xs))


# ------------------------------------------------------------
# 3. 以 row 为单位反查：每一行由哪些 lane/x[i] 持有
# ------------------------------------------------------------
rows = {r: [] for r in range(16)}
for lane in range(32):
    for i in range(FRAG_ELEMS):
        rc = mapping[lane][i]
        if rc is None:
            continue
        r, c = rc
        rows[r].append((c, lane, i))

print("\n=== row ownership: row -> (col, lane, x_index) ===")
for r in range(16):
    items = sorted(rows[r], key=lambda x: x[0])
    s = "  ".join([f"c{c:02d}:L{lane:02d}.x{i}" for c, lane, i in items])
    print(f"row {r:2d}: {s}")


# ------------------------------------------------------------
# 4. 检查分组是否行纯净
# ------------------------------------------------------------
OLD_A = [0, 1, 2, 3]
OLD_B = [4, 5, 6, 7]

NEW_A = [0, 1, 4, 5]
NEW_B = [2, 3, 6, 7]


def row_set_for(lane, group):
    rs = set()
    for i in group:
        rc = mapping[lane][i]
        if rc is not None:
            rs.add(rc[0])
    return rs


def check_group_purity(name, group_a, group_b):
    pure_a = 0
    pure_b = 0
    bad = []

    for lane in range(32):
        ra = row_set_for(lane, group_a)
        rb = row_set_for(lane, group_b)

        ok_a = len(ra) == 1
        ok_b = len(rb) == 1

        pure_a += int(ok_a)
        pure_b += int(ok_b)

        if not (ok_a and ok_b):
            bad.append((lane, sorted(ra), sorted(rb)))

    print(f"\n=== {name} grouping purity ===")
    print(f"group A pure lanes: {pure_a}/32")
    print(f"group B pure lanes: {pure_b}/32")

    if len(bad) == 0:
        print("✅ 这个分组在当前 mapping 下是 lane-local row-pure。")
    else:
        print("❌ 这个分组不是 row-pure，前几个坏例子：")
        for lane, ra, rb in bad[:12]:
            print(f"  lane {lane:2d}: A rows={ra}, B rows={rb}")

    return len(bad) == 0


old_ok = check_group_purity("OLD x[0..3] / x[4..7]", OLD_A, OLD_B)
new_ok = check_group_purity("NEW x[0,1,4,5] / x[2,3,6,7]", NEW_A, NEW_B)


# ------------------------------------------------------------
# 5. 污染测试矩阵
# ------------------------------------------------------------
print("\n=== OLD grouping pollution matrix ===")
print("x[0..3] += 100000, x[4..7] += 200000")
print(mat_old.to(torch.int32))

print("\n=== NEW grouping pollution matrix ===")
print("x[0,1,4,5] += 100000, x[2,3,6,7] += 200000")
print(mat_new.to(torch.int32))


# ------------------------------------------------------------
# 6. 保存 JSON，方便 3090 和 A100 diff
# ------------------------------------------------------------
safe_gpu = device_name.replace(" ", "_").replace("/", "_")
json_name = f"wmma_mapping_{safe_gpu}_sm{cap[0]}{cap[1]}_{opt_tag}.json"

json_mapping = {
    "gpu": device_name,
    "capability": f"sm_{cap[0]}{cap[1]}",
    "torch": torch_ver,
    "torch_cuda": torch_cuda,
    "opt_flag": opt_flag,
    "mapping": {
        str(lane): {
            str(i): None if mapping[lane][i] is None else {
                "row": mapping[lane][i][0],
                "col": mapping[lane][i][1],
            }
            for i in range(FRAG_ELEMS)
        }
        for lane in range(32)
    },
    "old_group_ok": old_ok,
    "new_group_ok": new_ok,
}

with open(json_name, "w", encoding="utf-8") as f:
    json.dump(json_mapping, f, indent=2, ensure_ascii=False)

print(f"\n=== saved mapping ===")
print(json_name)

print("\n=== final verdict ===")
if new_ok and not old_ok:
    print("✅ 当前环境下：旧分组错，新分组可用于 experimental V2B patch。")
elif old_ok:
    print("⚠️ 当前环境下：旧分组居然是 row-pure，需要人工复核输出。")
elif not new_ok:
    print("❌ 当前环境下：新分组也不 row-pure，不要在 WMMA fragment.x 上做 row-dependent softmax。")