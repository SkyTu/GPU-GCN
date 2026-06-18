#!/usr/bin/env python3
# Assemble a standard_gcn retrain dataset from the OBLIVIOUS-SELECT output.
# The select wrote secret shares of the X-removed selected subgraph (shard hidden):
#   Asel (=Amask, X's adj column zeroed), featsel, yohsel, trainmask (=tmeff, X dropped),
#   testmask  -- all in cora_shards_canon/select/, scale=12, 2-of-2 additive shares.
#
# standard_gcn converges at scale=24 (scale-12 stalls on truncation noise), so we
# re-quantize the secret shares 12->24 by multiplying EACH SHARE by 2^12 (local,
# distributes over the additive reconstruction). The train_mask stays a SECRET share
# (the trainer beaver-multiplies the gradient by it). The public train/test masks +
# labels are reconstructed for the trainer's (debug) reveal-eval only.
import numpy as np, os, sys

CANON = os.environ.get("FSS_DATA_ROOT") or os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "cora_shards_canon"))
SEL = CANON + "/select"
OUT = sys.argv[1] if len(sys.argv) > 1 else CANON + "/retrain"
def _meta(key, default):
    try:
        for line in open(CANON + "/meta.txt"):
            if line.startswith(key + "="):
                return int(line.split("=", 1)[1])
    except OSError:
        pass
    return default
N = _meta("Ns_max", 775); F = _meta("F", 1433); C = _meta("C", 7); H = 64
SRC_SCALE = 12; TRAIN_SCALE = 24
SC = np.uint64(1 << (TRAIN_SCALE - SRC_SCALE))     # 4096
rng = np.random.default_rng(0)

os.makedirs(OUT + "/graph", exist_ok=True)
os.makedirs(OUT + "/weights", exist_ok=True)
os.makedirs(OUT + "/outputs", exist_ok=True)

def rd(name): return np.fromfile(f"{SEL}/{name}.bin", np.uint64)

# --- secret graph shares, re-quantized 12 -> 24 (x SC, per share, mod 2^64) ---
for nm_in, nm_out in [("Asel", "adj"), ("featsel", "feat"), ("yohsel", "y_onehot")]:
    for p in (0, 1):
        s = rd(f"{nm_in}_share{p}")
        (s * SC).tofile(f"{OUT}/graph/{nm_out}_share{p}.bin")     # uint64 wraps mod 2^64

# --- train_mask: SECRET share (the trainer's gradient beaver-mul operand) ---
for p in (0, 1):
    rd(f"trainmask_share{p}").tofile(f"{OUT}/graph/train_mask_share{p}.bin")

# --- public masks + labels for the (debug) reveal-eval ---
tm = ((rd("trainmask_share0") + rd("trainmask_share1")) & np.uint64(1)).astype(np.uint8)
te = ((rd("testmask_share0") + rd("testmask_share1")) & np.uint64(1)).astype(np.uint8)
tm.tofile(f"{OUT}/graph/train_mask.bin")
te.tofile(f"{OUT}/graph/test_mask.bin")
yoh = (rd("yohsel_share0") + rd("yohsel_share1")).reshape(N, C)   # scale-12 one-hot
labels = np.argmax(yoh.astype(np.int64), axis=1).astype(np.int64)
labels.tofile(f"{OUT}/graph/labels.bin")

# --- fresh Glorot weights (retrain from scratch = GraphEraser semantics), scale-24 ---
def to_fixed(a): return np.round(a * (1 << TRAIN_SCALE)).astype(np.int64).view(np.uint64)
def share(u):
    s0 = rng.integers(0, 1 << 64, size=u.shape, dtype=np.uint64)
    return s0, (u - s0).astype(np.uint64)
l1 = np.sqrt(6.0 / (F + H)); l2 = np.sqrt(6.0 / (H + C))
W = {"W1": to_fixed(rng.uniform(-l1, l1, (F, H))), "b1": to_fixed(np.zeros(H)),
     "W2": to_fixed(rng.uniform(-l2, l2, (H, C))), "b2": to_fixed(np.zeros(C))}
for nm, u in W.items():
    s0, s1 = share(u.ravel())
    s0.tofile(f"{OUT}/weights/{nm}_share0.bin"); s1.tofile(f"{OUT}/weights/{nm}_share1.bin")

with open(f"{OUT}/meta.txt", "w") as f:
    f.write(f"N={N}\nF={F}\nC={C}\nH={H}\nscale={TRAIN_SCALE}\n"
            f"train_count={int(tm.sum())}\ntest_count={int(te.sum())}\nnormalized=1\nsecret_mask=1\n")

print(f"[assemble] retrain dataset -> {OUT}")
print(f"  N={N} F={F} H={H} C={C} scale={TRAIN_SCALE} (from select, re-quantized 12->24)")
print(f"  train_mask (SECRET share) sum={int(tm.sum())}  test sum={int(te.sum())}  (public copies are debug-eval only)")
print(f"  X-removed: adj column zeroed + X dropped from train_mask, all under the oblivious select")
