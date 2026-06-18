#!/usr/bin/env python3
"""Single-node FSS-ciphertext unlearning for the GraphEraser GCN pipeline.

GraphEraser = SISA-on-graphs: the train graph is partitioned (OpenGU LPA) into
k=10 shards, one GCN per shard, posteriors mean-aggregated. To *unlearn* one
node X we (a) find the single shard s that owns X, (b) drop X from shard s and
**retrain that one shard from scratch** (GraphEraser semantics), (c) re-infer
shard s' and re-aggregate. Shards != s are untouched, so their posteriors are
reused verbatim. This is the single-node specialization of GraphEraser's batch
unlearning -- it REPLACES the OpenGU-cleartext 270-node batch path.

This script does the *cleartext data-plumbing* (numpy):
  build  : emit the retrain dataset (standard_gcn format) + the unlearned
           shard-set skeleton (cloned from cora_shards with shard s replaced).
  finalize: import the FSS-retrained weight shares into the unlearned shard-set,
            then re-share / verify.

The actual ciphertext compute (retrain, L1 inference, L2 aggregate, eval) is
2-party FSS and is driven by the companion runner `run_unlearn_single_node.sh`.

  *** Note on the shard-routing step ***
  Resolving "which shard owns X" is, in the real protocol, a secure DPF lookup
  over the secret-shared community map (the user's query node X is itself
  secret-shared and a DPF selects the matching shard without revealing X). Here
  -- for the experiment -- we resolve it in the clear from the public community
  pickle. The crypto DPF routing is a separate, orthogonal primitive; this
  driver assumes its output (the shard id) is given.

Scale: the GraphEraser FSS pipeline (cora_shards) is fixed-point scale=12, so
the retrain dataset and the imported weight shares are ALL scale=12 (NOT 24).
The standard_gcn trainer reads `scale` from meta.txt, so it trains at 12 too.

Format reuse: feature row-normalization, D^-1/2(A+I)D^-1/2 adjacency, and the
2-out-of-2 additive sharing all mirror prepare_standard_gcn.py / prepare_shards.py.
"""

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Paths (remote box <server>)
# --------------------------------------------------------------------------- #
import os
_OPENGU = os.environ.get("OPENGU_ROOT", "")   # OpenGU install base (contains GULib-master/)
# OblivGU tests/GNN: env override, else sibling of the GPU-GCN repo (../../../../OblivGU/tests/GNN)
OG = os.environ.get("OBLIVGU_GNN") or os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "OblivGU", "tests", "GNN"))
GU_ROOT = _OPENGU + "/GULib-master/data/GraphEraser/processed/cora"
COMMUNITY_PICKLE = "community_lpa_base_10_0"
TRAIN_DATA_PICKLE = "train_data"

# --------------------------------------------------------------------------- #
# Fixed-point + additive sharing (identical convention to prepare_*.py)
# --------------------------------------------------------------------------- #
def float_to_fixed(arr, scale):
    fixed = np.round(np.asarray(arr, dtype=np.float64) * (1 << scale)).astype(np.int64)
    return fixed.view(np.uint64)


def split_shares(data_u64, rng):
    a0 = rng.integers(0, 2**63, size=data_u64.shape, dtype=np.uint64)
    a0 |= rng.integers(0, 2, size=data_u64.shape, dtype=np.uint64) << np.uint64(63)
    a1 = data_u64 - a0  # wraps mod 2^64
    return a0, a1


def save_bin(path, arr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(arr).tofile(path)


def to_numpy(x, dtype=None):
    arr = x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
    return arr.astype(dtype) if dtype is not None else arr


# --------------------------------------------------------------------------- #
# Graph helpers (mirror prepare_standard_gcn.py / prepare_shards.py)
# --------------------------------------------------------------------------- #
def normalize_features(x):
    row_sum = x.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return x / row_sum


def edge_index_to_dense_adj(edge_index, n, normalize=True):
    A = np.zeros((n, n), dtype=np.float64)
    ei = to_numpy(edge_index)
    if ei.size:
        src, dst = ei[0], ei[1]
        m = (src < n) & (dst < n) & (src >= 0) & (dst >= 0)
        src, dst = src[m], dst[m]
        A[src, dst] = 1.0
        A[dst, src] = 1.0
    A += np.eye(n, dtype=np.float64)
    if normalize:
        deg = A.sum(axis=1)
        deg[deg == 0] = 1.0
        d_inv_sqrt = 1.0 / np.sqrt(deg)
        A = (A * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]
    return A


def induced_subgraph(global_x, global_y, global_edge_index, indices):
    """Induced subgraph on `indices` (returned sorted)."""
    idx = np.sort(np.asarray(indices, dtype=np.int64))
    n = len(idx)
    pos = -np.ones(int(global_x.shape[0]), dtype=np.int64)
    pos[idx] = np.arange(n)
    x_sub = global_x[idx]
    y_sub = global_y[idx]
    ei = to_numpy(global_edge_index)
    if ei.size:
        src, dst = ei[0], ei[1]
        mask = (pos[src] >= 0) & (pos[dst] >= 0)
        new_ei = np.stack([pos[src[mask]], pos[dst[mask]]], axis=0)
    else:
        new_ei = np.zeros((2, 0), dtype=np.int64)
    return x_sub, y_sub, new_ei, idx


def build_node_to_shard(community):
    n2s = {}
    for shard, nodes in community.items():
        for node in to_numpy(nodes).tolist():
            n2s[int(node)] = int(shard)
    return n2s


def load_globals():
    with open(Path(GU_ROOT) / TRAIN_DATA_PICKLE, "rb") as f:
        td = pickle.load(f)
    with open(Path(GU_ROOT) / COMMUNITY_PICKLE, "rb") as f:
        community = pickle.load(f)
    x = to_numpy(td.x, np.float64)
    y = to_numpy(td.y, np.int64)
    edge_index = to_numpy(td.edge_index, np.int64)
    train_idx = np.asarray(td.train_indices, dtype=np.int64)
    test_idx = np.asarray(td.test_indices, dtype=np.int64)
    return x, y, edge_index, train_idx, test_idx, community


# --------------------------------------------------------------------------- #
# Step 1: resolve affected shard for X (clear; models the secure DPF routing)
# --------------------------------------------------------------------------- #
def resolve_shard(node, community):
    n2s = build_node_to_shard(community)
    if node not in n2s:
        raise SystemExit(
            f"ERROR: node {node} is not in the public community train "
            f"partition ({COMMUNITY_PICKLE}); cannot unlearn it.")
    return n2s[node]


# --------------------------------------------------------------------------- #
# Step 2: build the standard_gcn retrain dataset for shard s minus X
# --------------------------------------------------------------------------- #
def build_retrain_dataset(node, shard, out_dir, scale, hidden, seed):
    """Emit a standard_gcn dataset (graph/ + weights/ + meta.txt) for the
    induced graph (shard-s TRAIN minus X) ∪ all TEST nodes.

    train_mask = retained shard-s train nodes; test_mask = the test nodes.
    Weights are FRESH Glorot (retrain-from-scratch = GraphEraser semantics).
    This induced graph is IDENTICAL to the unlearned L1 shard graph, so the
    learned weights are trained on exactly the topology they're applied to.
    """
    rng = np.random.default_rng(seed)
    x_all, y_all, edge_index, train_idx, test_idx, community = load_globals()

    shard_train = np.asarray(community[shard], dtype=np.int64)
    if node not in set(shard_train.tolist()):
        raise SystemExit(f"ERROR: node {node} not a TRAIN node of shard {shard}")
    remaining_train = np.setdiff1d(shard_train, np.asarray([node], dtype=np.int64))
    node_set = np.union1d(remaining_train, test_idx)

    x_sub, y_sub, ei_sub, glob_idx = induced_subgraph(x_all, y_all, edge_index, node_set)
    n, f = x_sub.shape
    c = int(y_all.max()) + 1
    h = hidden

    x_norm = normalize_features(x_sub)
    adj = edge_index_to_dense_adj(ei_sub, n, normalize=True)

    train_mask = np.isin(glob_idx, remaining_train).astype(np.uint8)
    test_mask = np.isin(glob_idx, test_idx).astype(np.uint8)

    out = Path(out_dir)
    (out / "graph").mkdir(parents=True, exist_ok=True)
    (out / "weights").mkdir(parents=True, exist_ok=True)

    x_fixed = float_to_fixed(x_norm, scale)
    adj_fixed = float_to_fixed(adj, scale)
    y_onehot = np.zeros((n, c), dtype=np.uint64)
    y_onehot[np.arange(n), y_sub.astype(np.int64)] = np.uint64(1 << scale)

    for name, arr in (("feat", x_fixed), ("adj", adj_fixed), ("y_onehot", y_onehot)):
        s0, s1 = split_shares(arr, rng)
        save_bin(out / "graph" / f"{name}_share0.bin", s0)
        save_bin(out / "graph" / f"{name}_share1.bin", s1)

    save_bin(out / "graph" / "labels.bin", y_sub.astype(np.int64))
    save_bin(out / "graph" / "train_mask.bin", train_mask)
    save_bin(out / "graph" / "test_mask.bin", test_mask)

    # Fresh Glorot init (retrain from scratch). zero biases.
    limit1 = np.sqrt(6.0 / (f + h))
    limit2 = np.sqrt(6.0 / (h + c))
    w1 = rng.uniform(-limit1, limit1, size=(f, h))
    w2 = rng.uniform(-limit2, limit2, size=(h, c))
    b1 = np.zeros(h, dtype=np.float64)
    b2 = np.zeros(c, dtype=np.float64)
    for name, arr in (
        ("W1", float_to_fixed(w1, scale)),
        ("b1", float_to_fixed(b1, scale)),
        ("W2", float_to_fixed(w2, scale)),
        ("b2", float_to_fixed(b2, scale)),
    ):
        s0, s1 = split_shares(arr, rng)
        save_bin(out / "weights" / f"{name}_share0.bin", s0)
        save_bin(out / "weights" / f"{name}_share1.bin", s1)

    with open(out / "meta.txt", "w") as fp:
        fp.write(f"N={n}\nF={f}\nC={c}\nH={h}\nscale={scale}\n"
                 f"train_count={int(train_mask.sum())}\n"
                 f"test_count={int(test_mask.sum())}\nnormalized=1\n")

    print(f"[build-retrain] dataset {out}")
    print(f"[build-retrain] node={node} shard={shard} N={n} (={len(remaining_train)} "
          f"train + {len(test_idx)} test) F={f} H={h} C={c} scale={scale}")
    print(f"[build-retrain] train_mask={int(train_mask.sum())} "
          f"test_mask={int(test_mask.sum())}")
    return n


# --------------------------------------------------------------------------- #
# Step 4a: build the unlearned shard-set skeleton (clone cora_shards, replace s)
# --------------------------------------------------------------------------- #
def build_unlearned_shardset(node, shard, retrain_dir, src_shardset, out_shardset,
                             scale, seed):
    """Clone the baseline shard-set and replace shard `s` with the
    (shard-s minus X) induced graph. Shards != s keep their original
    adj/feat/y/test_mask. Weights for shard s are filled in later (finalize)
    from the FSS-retrained shares; here we just write the shard-s graph.
    """
    import shutil

    rng = np.random.default_rng(seed + 1)
    x_all, y_all, edge_index, train_idx, test_idx, community = load_globals()

    src = Path(src_shardset)
    dst = Path(out_shardset)
    if dst.exists():
        shutil.rmtree(dst)
    print(f"[build-shardset] cloning {src} -> {dst}")
    shutil.copytree(src, dst)

    # Rebuild shard-s graph = induced (shard-s train minus X) ∪ test, scale=12.
    shard_train = np.asarray(community[shard], dtype=np.int64)
    remaining_train = np.setdiff1d(shard_train, np.asarray([node], dtype=np.int64))
    node_set = np.union1d(remaining_train, test_idx)
    x_sub, y_sub, ei_sub, glob_idx = induced_subgraph(x_all, y_all, edge_index, node_set)
    ns, f = x_sub.shape
    c = int(y_all.max()) + 1

    x_norm = normalize_features(x_sub)
    adj = edge_index_to_dense_adj(ei_sub, ns, normalize=True)
    test_mask = np.isin(glob_idx, test_idx).astype(np.uint8)

    A_u64 = float_to_fixed(adj, scale)
    X_u64 = float_to_fixed(x_norm, scale)
    A0, A1 = split_shares(A_u64, rng)
    X0, X1 = split_shares(X_u64, rng)

    y_u64 = y_sub.astype(np.int64).view(np.uint64).copy()
    Y0, Y1 = split_shares(y_u64, rng)
    tm_u64 = test_mask.astype(np.uint64)
    TM0, TM1 = split_shares(tm_u64, rng)

    sd = dst / "shards"
    prefix = sd / f"shard_{shard}"
    save_bin(f"{prefix}_adj_share0.bin", A0)
    save_bin(f"{prefix}_adj_share1.bin", A1)
    save_bin(f"{prefix}_feat_share0.bin", X0)
    save_bin(f"{prefix}_feat_share1.bin", X1)
    save_bin(f"{prefix}_y_share0.bin", Y0)
    save_bin(f"{prefix}_y_share1.bin", Y1)
    save_bin(f"{prefix}_y.bin", y_sub.astype(np.int64))
    save_bin(f"{prefix}_test_mask.bin", test_mask)
    save_bin(f"{prefix}_test_mask_share0.bin", TM0)
    save_bin(f"{prefix}_test_mask_share1.bin", TM1)
    with open(f"{prefix}_meta.txt", "w") as fp:
        fp.write(f"Ns={ns}\nF={f}\nC={c}\nscale={scale}\n"
                 f"num_test={int(test_mask.sum())}\nnormalized=1\nunlearned=1\n")

    # the global meta keeps k/scale; flag unlearned + record the request.
    gmeta = (dst / "meta.txt").read_text().splitlines()
    gmeta = [ln for ln in gmeta if not ln.startswith("unlearned=")]
    gmeta.append("unlearned=1")
    (dst / "meta.txt").write_text("\n".join(gmeta) + "\n")
    with open(dst / "unlearn_summary.txt", "w") as fp:
        fp.write("unlearning_mode=node\nunlearning_batching=single\n")
        fp.write(f"unlearn_node={node}\naffected_shards=[{shard}]\n")
        fp.write(f"num_unlearning_nodes=1\nnum_retrained_shards=1\n")
        fp.write(f"shard_{shard}_Ns={ns}\nscale={scale}\n")

    print(f"[build-shardset] shard {shard} replaced: Ns={ns} "
          f"num_test={int(test_mask.sum())} (X={node} removed)")
    # sanity: shard-s test rows must equal the baseline (so L2 num_test agrees)
    base_tm = np.fromfile(src / "shards" / f"shard_{shard}_test_mask.bin", dtype=np.uint8)
    if int(base_tm.sum()) != int(test_mask.sum()):
        raise SystemExit(
            f"ERROR: test-node count changed for shard {shard} "
            f"({int(base_tm.sum())} -> {int(test_mask.sum())}); L2 would reject")
    return ns


# --------------------------------------------------------------------------- #
# Step 4b (finalize): import FSS-retrained weight shares into the shard-set
# --------------------------------------------------------------------------- #
def _fixed_u64_to_float(arr_u64, scale):
    return arr_u64.view(np.int64).astype(np.float64) / float(1 << scale)


def finalize_weights(shard, retrain_dir, out_shardset, train_scale, pipe_scale, seed):
    """Map standard_gcn weight shares -> grapheraser shard_<s>_* layout.

    The standard_gcn trainer trains at `train_scale` (=24: scale-12 truncation
    noise stalls convergence on this small shard -- empirically verified, the
    full-graph FSS result reached the same conclusion). The GraphEraser L1
    pipeline is `pipe_scale` (=12). So this step is NOT a pure copy: we
      1. reconstruct the scale-`train_scale` weights from the two FSS shares,
      2. re-quantize to fixed-point scale-`pipe_scale` (round(w * 2^pipe)),
      3. re-split into fresh 2-out-of-2 additive shares,
    and write them in the grapheraser shard_<s>_{W1,b1,W2,b2}_share<p>.bin
    layout. The public reconstructed shard_<s>_*.bin is refreshed too (legacy
    cleartext readers). When train_scale == pipe_scale this is a faithful
    re-share with no value change.
    """
    rt = Path(retrain_dir) / "weights"
    wd = Path(out_shardset) / "weights"
    wd.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed + 2)
    for name in ("W1", "b1", "W2", "b2"):
        s0p = rt / f"{name}_share0.bin"
        s1p = rt / f"{name}_share1.bin"
        if not s0p.exists() or not s1p.exists():
            raise SystemExit(f"ERROR: retrained weight share missing: {s0p}/{s1p} "
                             f"(did the FSS retrain run + dump?)")
        s0 = np.fromfile(s0p, dtype=np.uint64)
        s1 = np.fromfile(s1p, dtype=np.uint64)
        recon_u64 = s0 + s1  # public reconstruction (wraps mod 2^64)
        w_float = _fixed_u64_to_float(recon_u64, train_scale)
        w_pipe_u64 = float_to_fixed(w_float, pipe_scale)
        a0, a1 = split_shares(w_pipe_u64, rng)
        save_bin(wd / f"shard_{shard}_{name}_share0.bin", a0)
        save_bin(wd / f"shard_{shard}_{name}_share1.bin", a1)
        save_bin(wd / f"shard_{shard}_{name}.bin", w_pipe_u64)  # public recon at pipe scale
    print(f"[finalize] retrained weights re-quantized scale {train_scale}->{pipe_scale}, "
          f"re-shared -> {wd}/shard_{shard}_*")


# --------------------------------------------------------------------------- #
# Privacy-preserving SELECT (FSS DPF) -- prep + verify
# --------------------------------------------------------------------------- #
def _select_bin(n_nodes):
    """smallest DPF bitwidth (>=8) covering node ids 0..n_nodes-1."""
    b = 8
    while (1 << b) < n_nodes:
        b += 1
    return b


def select_prep(node, out_shardset, seed):
    """Write the FSS inputs for grapheraser_fss_select into <shardset>/select/:
      qnode_share<p>.bin  : 2-out-of-2 additive shares of the query node id X
      community_tab.bin   : node -> shard id, PUBLIC, padded to 2^bin (u64)
      shard_nodes.bin     : PUBLIC node ids of the affected shard (u64)
      select_meta.txt     : N, bin, affected_ns, affected_shard
    The community partition is public in GraphEraser; ONLY X is secret-shared.
    """
    rng = np.random.default_rng(seed + 7)
    _, _, _, _, _, community = load_globals()
    n_total = 2708  # cora node count (DPF domain is padded to 2^bin)
    bin_ = _select_bin(n_total)
    dom = 1 << bin_

    shard = resolve_shard(node, community)  # ground-truth (the FSS proto recomputes this obliviously)
    shard_nodes = np.sort(np.asarray(community[shard], dtype=np.int64))
    affected_ns = len(shard_nodes)

    sd = Path(out_shardset) / "select"
    sd.mkdir(parents=True, exist_ok=True)

    # secret-shared query node X (additive shares over u64)
    x_u64 = np.asarray([node], dtype=np.uint64)
    s0, s1 = split_shares(x_u64, rng)
    save_bin(sd / "qnode_share0.bin", s0)
    save_bin(sd / "qnode_share1.bin", s1)

    # public community table node->shard, padded to 2^bin
    comm_tab = np.zeros(dom, dtype=np.uint64)
    n2s = build_node_to_shard(community)
    for nd, sh in n2s.items():
        comm_tab[nd] = np.uint64(sh)
    save_bin(sd / "community_tab.bin", comm_tab)

    # public affected-shard node ids
    save_bin(sd / "shard_nodes.bin", shard_nodes.astype(np.uint64))

    with open(sd / "select_meta.txt", "w") as fp:
        fp.write(f"N={n_total}\nbin={bin_}\naffected_ns={affected_ns}\n"
                 f"affected_shard={shard}\nquery_node={node}\n")
    print(f"[select-prep] node={node} -> (ground-truth shard {shard}); "
          f"N={n_total} bin={bin_} dom={dom} affected_ns={affected_ns}")
    print(f"[select-prep] wrote {sd}/{{qnode_share*,community_tab,shard_nodes,select_meta}}")


def select_verify(node, out_shardset):
    """Validate grapheraser_fss_select's output against the cleartext oracle:
      - routed_shard.bin must equal resolve_shard(X).
      - reconstruct keep_share / isx_share (XOR of the two boolean shares) and
        check isX is a one-hot at X's position, keep = NOT isX.
    """
    _, _, _, _, _, community = load_globals()
    sd = Path(out_shardset) / "select"
    gt_shard = resolve_shard(node, community)
    shard_nodes = np.sort(np.asarray(community[gt_shard], dtype=np.int64))
    affected_ns = len(shard_nodes)

    routed = int(np.fromfile(sd / "routed_shard.bin", dtype=np.int32)[0])
    ok_route = (routed == gt_shard)

    isx0 = np.fromfile(sd / "isx_share0.bin", dtype=np.uint64)
    isx1 = np.fromfile(sd / "isx_share1.bin", dtype=np.uint64)
    keep0 = np.fromfile(sd / "keep_share0.bin", dtype=np.uint64)
    keep1 = np.fromfile(sd / "keep_share1.bin", dtype=np.uint64)
    # arithmetic additive shares at bout=16 -> reconstruct mod 2^16
    MOD = np.uint64(1 << 16)
    isX = ((isx0 + isx1) % MOD).astype(np.int64)
    keep = ((keep0 + keep1) % MOD).astype(np.int64)

    # ground-truth one-hot: 1 exactly at the position of X within shard_nodes
    gt_isX = (shard_nodes == node).astype(np.int64)
    ok_isx = np.array_equal(isX, gt_isX)
    ok_keep = np.array_equal(keep, 1 - gt_isX)
    pos = int(np.where(shard_nodes == node)[0][0]) if node in shard_nodes else -1

    print(f"[select-verify] routed_shard={routed} (oracle {gt_shard}) -> "
          f"{'OK' if ok_route else 'MISMATCH'}")
    print(f"[select-verify] isX one-hot sum={int(isX.sum())} at pos={int(np.argmax(isX)) if isX.sum() else -1} "
          f"(oracle pos={pos}) -> {'OK' if ok_isx else 'MISMATCH'}")
    print(f"[select-verify] keep == NOT isX over {affected_ns} nodes -> "
          f"{'OK' if ok_keep else 'MISMATCH'}; kept={int(keep.sum())} (expect {affected_ns - 1})")
    all_ok = ok_route and ok_isx and ok_keep
    print(f"[select-verify] {'ALL OK -- FSS select matches cleartext oracle' if all_ok else 'FAILED'}")
    if not all_ok:
        raise SystemExit("select-verify FAILED")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_r = sub.add_parser("resolve", help="print the affected shard for node X")
    ap_r.add_argument("--node", type=int, default=270)

    ap_b = sub.add_parser("build", help="build retrain dataset + unlearned shard-set skeleton")
    ap_b.add_argument("--node", type=int, default=270)
    ap_b.add_argument("--scale", type=int, default=12,
                      help="pipeline / shard-set fixed-point scale (GraphEraser L1/L2 = 12)")
    ap_b.add_argument("--train-scale", type=int, default=24,
                      help="retrain-dataset scale for standard_gcn_fss_train (24 converges; "
                           "12 stalls on truncation noise -- verified)")
    ap_b.add_argument("--hidden", type=int, default=64)
    ap_b.add_argument("--seed", type=int, default=0)
    ap_b.add_argument("--retrain-dir", default=None)
    ap_b.add_argument("--src-shardset", default=f"{OG}/datasets/cora_shards")
    ap_b.add_argument("--out-shardset", default=None)

    ap_f = sub.add_parser("finalize", help="import FSS-retrained weights into shard-set")
    ap_f.add_argument("--node", type=int, default=270)
    ap_f.add_argument("--scale", type=int, default=12, help="pipeline scale")
    ap_f.add_argument("--train-scale", type=int, default=24, help="retrain dataset scale")
    ap_f.add_argument("--seed", type=int, default=0)
    ap_f.add_argument("--retrain-dir", default=None)
    ap_f.add_argument("--out-shardset", default=None)

    ap_sp = sub.add_parser("select-prep", help="write FSS inputs for grapheraser_fss_select")
    ap_sp.add_argument("--node", type=int, default=270)
    ap_sp.add_argument("--seed", type=int, default=0)
    ap_sp.add_argument("--out-shardset", default=None)

    ap_sv = sub.add_parser("select-verify", help="check FSS select output vs cleartext oracle")
    ap_sv.add_argument("--node", type=int, default=270)
    ap_sv.add_argument("--out-shardset", default=None)

    args = ap.parse_args()

    _, _, _, _, _, community = load_globals()
    shard = resolve_shard(args.node, community)

    if args.cmd == "resolve":
        print(f"node {args.node} -> shard {shard}")
        return

    retrain_dir = (getattr(args, "retrain_dir", None)
                   or f"{OG}/datasets/unlearn/shard{shard}_minus_{args.node}")
    out_shardset = (getattr(args, "out_shardset", None)
                    or f"{OG}/datasets/cora_shards_unlearned_single_{args.node}")

    if args.cmd == "build":
        print(f"[resolve] node {args.node} -> shard {shard} "
              f"(secure DPF routing resolved in clear for the experiment)")
        # retrain dataset is built at train_scale (24, converges); the shard-set
        # graph (adj/feat/y) is at pipeline scale (12). finalize() bridges the
        # two by re-quantizing the learned weights 24 -> 12.
        build_retrain_dataset(args.node, shard, retrain_dir,
                              args.train_scale, args.hidden, args.seed)
        build_unlearned_shardset(args.node, shard, retrain_dir,
                                 args.src_shardset, out_shardset, args.scale, args.seed)
        print(f"[build] retrain_dir   = {retrain_dir}  (train scale={args.train_scale})")
        print(f"[build] out_shardset  = {out_shardset}  (pipeline scale={args.scale})")
        print(f"[build] affected_shard={shard}")
    elif args.cmd == "finalize":
        finalize_weights(shard, retrain_dir, out_shardset,
                         args.train_scale, args.scale, args.seed)
    elif args.cmd == "select-prep":
        select_prep(args.node, out_shardset, args.seed)
    elif args.cmd == "select-verify":
        select_verify(args.node, out_shardset)


if __name__ == "__main__":
    main()
