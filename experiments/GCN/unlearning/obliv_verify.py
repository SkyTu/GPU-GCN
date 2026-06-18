#!/usr/bin/env python3
# prep: secret-share X into qnode_share{0,1}.  verify: my independent oracle vs the FSS output.
import sys, os, numpy as np
# Portable: honor FSS_DATA_ROOT (set by the run scripts), else derive from this file's location.
ROOT = os.environ.get("FSS_DATA_ROOT") or os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "cora_shards_canon"))
def _meta(key, default):
    try:
        for line in open(ROOT + "/meta.txt"):
            if line.startswith(key + "="):
                return int(line.split("=", 1)[1])
    except OSError:
        pass
    return default
# query node X + its ground-truth shard/row + padded shard size, from canonical meta
# (canonical_export writes X/true_shard/row; fall back to cora's values for an old meta).
X = _meta("X", 270); Ns = _meta("Ns_max", 775); ROW = _meta("row", 74); TRUE_SHARD = _meta("true_shard", 0)
mode = sys.argv[1]
if mode == "prep":
    import os; os.makedirs(ROOT + "/select", exist_ok=True)
    rng = np.random.default_rng(1)
    r = rng.integers(0, 1 << 64, dtype=np.uint64)
    np.array([r], np.uint64).tofile(ROOT + "/select/qnode_share0.bin")
    np.array([np.uint64(X) - r], np.uint64).tofile(ROOT + "/select/qnode_share1.bin")
    print(f"prep: X={X} secret-shared -> qnode_share0/1.bin")
elif mode == "verify":
    # ORACLE (independent, canonical ordering): every oblivious gather of shard data
    # must equal shard 0's data (X∈shard0); keep must be 1 except X's row (74).
    def recon(name):
        s0 = np.fromfile(f"{ROOT}/shards/shard_{TRUE_SHARD}_{name}_share0.bin", np.uint64)
        s1 = np.fromfile(f"{ROOT}/shards/shard_{TRUE_SHARD}_{name}_share1.bin", np.uint64)
        return s0 + s1
    def nzcount(oracle, sel):
        d = (oracle.astype(object) - sel.astype(object))
        return int(np.count_nonzero(np.array([int(x) % (1 << 64) for x in d], dtype=object)))
    ok = True
    # raw gathers must be bit-exact vs shard 0
    for name, sel_f in [("feat", "featsel"), ("y_onehot", "yohsel")]:
        sel = np.fromfile(f"{ROOT}/select/{sel_f}_clear.bin", np.uint64)
        nz = nzcount(recon(name), sel)
        ok = ok and (nz == 0)
        print(f"GATHER {name:9s}: nonzero-count={nz:6d}  (0 = bit-exact select, shard hidden)")
    te = np.fromfile(f"{ROOT}/shards/shard_{TRUE_SHARD}_test_mask.bin", np.uint64)
    nz = nzcount(te, np.fromfile(f"{ROOT}/select/testmask_clear.bin", np.uint64))
    ok = ok and (nz == 0); print(f"GATHER test_mask : nonzero-count={nz:6d}  (local secret-oh x public-mask)")
    # keep-applied adjacency: shard 0 adj with X's COLUMN (index ROW) zeroed.
    A = recon("adj").reshape(Ns, Ns).copy(); A[:, ROW] = 0
    Amask = np.fromfile(f"{ROOT}/select/Amask_clear.bin", np.uint64).reshape(Ns, Ns)
    nz = nzcount(A.ravel(), Amask.ravel())
    ok = ok and (nz == 0)
    print(f"KEEP   adj_colzero: nonzero-count={nz:6d}  (0 = X's adjacency column removed obliviously)")
    # keep-applied train_mask: shard 0 train_mask with X (index ROW) dropped.
    tm = np.fromfile(f"{ROOT}/shards/shard_{TRUE_SHARD}_train_mask.bin", np.uint64).copy(); tm[ROW] = 0
    tmeff = np.fromfile(f"{ROOT}/select/tmeff_clear.bin", np.uint64)
    nz = nzcount(tm, tmeff)
    ok = ok and (nz == 0)
    print(f"KEEP   train_drop : nonzero-count={nz:6d}  (0 = X dropped from the loss mask)")
    keep = np.fromfile(f"{ROOT}/select/keep_clear.bin", np.uint64) & 1
    bad = [j for j in range(Ns) if keep[j] != (0 if j == ROW else 1)]
    ok = ok and not bad
    print(f"REMOVE  : keep[{ROW}]={int(keep[ROW])} (expect 0); other rows !=1 at: {bad[:10]}  (empty = perfect one-hot)")
    print(f"RESULT  : {'PASS — oblivious X-removed subgraph correct (gather + keep removal; X & shard hidden)' if ok else 'FAIL'}")
