import argparse
import ast
import pickle
import sys
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    print("ERROR: torch is required to read OpenGU pickles/checkpoints.", file=sys.stderr)
    raise


import os
_OPENGU = os.environ.get("OPENGU_ROOT", "")   # OpenGU install base; or pass --gu-root / --weight-dir
GU_ROOT_FMT = _OPENGU + "/data/GraphEraser/processed/{ds}"
WEIGHT_DIR_FMT = _OPENGU + "/data/GraphEraser/{ds}"
UNLEARN_SUMMARY_FMT = "datasets/{ds}_shards_unlearned/run0/unlearn_summary.txt"
SHARD_PICKLE = "shard_data_lpa_base_10_0.005_0"
COMMUNITY_PICKLE = "community_lpa_base_10_0"
TRAIN_DATA_PICKLE = "train_data"
WEIGHT_FILE_FMT = "GCN_lpa_base_10_0.005_0_{s}_0"
UNLEARNED_WEIGHT_FILE_FMT = WEIGHT_FILE_FMT + "_{n}_unlearned"


def parse_meta(path):
    out = {}
    with open(path) as f:
        for raw in f:
            if "=" not in raw:
                continue
            k, v = raw.strip().split("=", 1)
            out[k] = v
    return out


def parse_int_list_text(text):
    return [int(x) for x in ast.literal_eval(text)]


def read_unlearn_summary(path):
    meta = parse_meta(path)
    if "unlearning_node_ids" not in meta:
        raise SystemExit(f"ERROR: {path} missing unlearning_node_ids")
    nodes = np.asarray(parse_int_list_text(meta["unlearning_node_ids"]), dtype=np.int64)
    affected = parse_int_list_text(meta.get("affected_shards", "[]"))
    return meta, nodes, affected


def to_numpy(x, dtype=None):
    arr = x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
    return arr.astype(dtype) if dtype is not None else arr


def float_to_fixed(arr, scale):
    fixed = np.round(np.asarray(arr, dtype=np.float64) * (1 << scale)).astype(np.int64)
    return fixed.view(np.uint64)


def split_shares(data_u64, rng):
    a0 = rng.integers(0, 2**63, size=data_u64.shape, dtype=np.uint64)
    a0 |= rng.integers(0, 2, size=data_u64.shape, dtype=np.uint64) << np.uint64(63)
    return a0, data_u64 - a0


def save_bin(path, arr):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.asarray(arr).tofile(path)


def induced_subgraph(global_x, global_y, global_edge_index, indices):
    idx = np.sort(np.asarray(indices, dtype=np.int64))
    pos = -np.ones(int(global_x.shape[0]), dtype=np.int64)
    pos[idx] = np.arange(idx.size)

    x_sub = global_x[idx]
    y_sub = global_y[idx]
    ei = global_edge_index.cpu().numpy() if hasattr(global_edge_index, "cpu") else np.asarray(global_edge_index)
    if ei.size:
        src, dst = ei[0], ei[1]
        mask = (pos[src] >= 0) & (pos[dst] >= 0)
        new_ei = np.stack([pos[src[mask]], pos[dst[mask]]], axis=0)
    else:
        new_ei = np.zeros((2, 0), dtype=np.int64)
    return x_sub, y_sub, new_ei, idx


def edge_index_to_dense_adj(edge_index, n, add_self_loops=True, normalize=True):
    a = np.zeros((n, n), dtype=np.float64)
    if edge_index.size:
        src, dst = edge_index[0], edge_index[1]
        m = (src >= 0) & (dst >= 0) & (src < n) & (dst < n)
        a[src[m], dst[m]] = 1.0
        a[dst[m], src[m]] = 1.0
    if add_self_loops:
        a += np.eye(n, dtype=np.float64)
    if normalize:
        deg = a.sum(axis=1)
        deg[deg == 0] = 1.0
        d = 1.0 / np.sqrt(deg)
        a = (a * d[:, None]) * d[None, :]
    return a


def export_weights(weight_dir, out_root, shards, model_kind, affected, num_unlearn, f_dim, c_dim, scale, rng):
    weights_dir = out_root / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    h_dim = None
    c_seen = None
    sources = {}

    for s in shards:
        if model_kind == "unlearned" and int(s) in affected:
            wp = Path(weight_dir) / UNLEARNED_WEIGHT_FILE_FMT.format(s=s, n=num_unlearn)
            kind = "unlearned"
        else:
            wp = Path(weight_dir) / WEIGHT_FILE_FMT.format(s=s)
            kind = "original"
        if not wp.exists():
            raise SystemExit(f"ERROR: missing weight file for shard {s}: {wp}")

        sd = torch.load(wp, map_location="cpu")
        w1 = sd["convs.0.lin.weight"].cpu().numpy()
        b1 = sd["convs.0.bias"].cpu().numpy()
        w2 = sd["convs.1.lin.weight"].cpu().numpy()
        b2 = sd["convs.1.bias"].cpu().numpy()
        w1_io = w1.T.copy()
        w2_io = w2.T.copy()
        if w1.shape[1] != f_dim:
            raise SystemExit(f"ERROR: shard {s} W1 in_dim {w1.shape[1]} != F={f_dim}")
        if h_dim is None:
            h_dim = int(w1.shape[0])
        if c_seen is None:
            c_seen = int(w2.shape[0])
        if h_dim != int(w1.shape[0]) or h_dim != int(w2.shape[1]) or c_seen != c_dim:
            raise SystemExit(f"ERROR: inconsistent weight shape at shard {s}")

        arrays = {
            "W1": float_to_fixed(w1_io, scale),
            "b1": float_to_fixed(b1, scale),
            "W2": float_to_fixed(w2_io, scale),
            "b2": float_to_fixed(b2, scale),
        }
        for name, arr in arrays.items():
            a0, a1 = split_shares(arr, rng)
            prefix = weights_dir / f"shard_{s}_{name}"
            save_bin(f"{prefix}.bin", arr)
            save_bin(f"{prefix}_share0.bin", a0)
            save_bin(f"{prefix}_share1.bin", a1)
        sources[int(s)] = f"{kind}: {wp}"

    with open(weights_dir / "weights_meta.txt", "w") as fp:
        fp.write(f"H={h_dim}\nF={f_dim}\nC={c_seen}\nscale={scale}\n")
    return sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cora", choices=["cora", "citeseer", "pubmed"])
    ap.add_argument("--model-kind", choices=["original", "unlearned"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gu-root", default=None)
    ap.add_argument("--weight-dir", default=None)
    ap.add_argument("--unlearn-summary", default=None)
    ap.add_argument("--num-pos", type=int, default=100)
    ap.add_argument("--num-neg", type=int, default=100)
    ap.add_argument("--scale", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-normalize", action="store_true")
    args = ap.parse_args()

    gu_root = Path(args.gu_root or GU_ROOT_FMT.format(ds=args.dataset))
    weight_dir = Path(args.weight_dir or WEIGHT_DIR_FMT.format(ds=args.dataset))
    summary_path = Path(args.unlearn_summary or UNLEARN_SUMMARY_FMT.format(ds=args.dataset))
    out_root = Path(args.out_dir)
    rng = np.random.default_rng(args.seed)

    with open(gu_root / SHARD_PICKLE, "rb") as f:
        shard_data = pickle.load(f)
    with open(gu_root / COMMUNITY_PICKLE, "rb") as f:
        community = pickle.load(f)
    with open(gu_root / TRAIN_DATA_PICKLE, "rb") as f:
        train_data = pickle.load(f)

    summary, unlearn_nodes, affected = read_unlearn_summary(summary_path)
    test_pool = np.asarray(train_data.test_indices, dtype=np.int64)
    n_pos = min(args.num_pos, unlearn_nodes.size)
    n_neg = min(args.num_neg, test_pool.size)
    if n_pos < args.num_pos:
        print(f"WARN: only {unlearn_nodes.size} unlearned nodes available, using {n_pos}")
    if n_neg < args.num_neg:
        print(f"WARN: only {test_pool.size} test nodes available, using {n_neg}")
    pos = rng.choice(unlearn_nodes, size=n_pos, replace=False)
    neg = rng.choice(test_pool, size=n_neg, replace=False)
    inter = np.intersect1d(pos, neg)
    if inter.size:
        raise SystemExit(
            f"ERROR: pos/neg query sets overlap on {inter.size} node(s): "
            f"{inter[:10].tolist()}{'...' if inter.size > 10 else ''}")
    query_nodes = np.unique(np.concatenate([pos, neg]).astype(np.int64))
    query_nodes.sort()
    membership = np.isin(query_nodes, pos).astype(np.uint8)

    global_x = to_numpy(train_data.x)
    global_y = to_numpy(train_data.y, dtype=np.int64)
    n_nodes, f_dim = global_x.shape
    c_dim = int(global_y.max()) + 1
    k = len(shard_data)

    print(f"dataset={args.dataset} model={args.model_kind} out={out_root}")
    print(f"query nodes={query_nodes.size} pos={int(membership.sum())} neg={query_nodes.size - int(membership.sum())}")
    print(f"affected shards={affected}")

    shards_dir = out_root / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    save_bin(out_root / "mia_query_nodes.bin", query_nodes.astype(np.int64))
    save_bin(out_root / "mia_membership_labels.bin", membership)
    with open(out_root / "mia_query_meta.txt", "w") as fp:
        fp.write(f"dataset={args.dataset}\nmodel_kind={args.model_kind}\n"
                 f"num_query={query_nodes.size}\nnum_pos={int(membership.sum())}\n"
                 f"num_neg={query_nodes.size - int(membership.sum())}\n"
                 f"seed={args.seed}\n"
                 f"unlearn_summary={summary_path}\n")

    for s in sorted(shard_data.keys()):
        shard_indices = np.union1d(np.asarray(community[s], dtype=np.int64), query_nodes)
        x, y, ei, shard_global_idx = induced_subgraph(global_x, global_y, train_data.edge_index, shard_indices)
        ns = x.shape[0]
        test_mask = np.isin(shard_global_idx, query_nodes).astype(np.uint8)
        if int(test_mask.sum()) != query_nodes.size:
            raise SystemExit(f"ERROR: shard {s} has {int(test_mask.sum())} query rows, expected {query_nodes.size}")

        adj = edge_index_to_dense_adj(ei, ns, normalize=not args.no_normalize)
        a_u64 = float_to_fixed(adj, args.scale)
        x_u64 = float_to_fixed(x, args.scale)
        y_u64 = y.astype(np.int64).view(np.uint64).copy()
        tm_u64 = test_mask.astype(np.uint64)

        a0, a1 = split_shares(a_u64, rng)
        x0, x1 = split_shares(x_u64, rng)
        y0, y1 = split_shares(y_u64, rng)
        tm0, tm1 = split_shares(tm_u64, rng)

        prefix = shards_dir / f"shard_{s}"
        save_bin(f"{prefix}_adj_share0.bin", a0)
        save_bin(f"{prefix}_adj_share1.bin", a1)
        save_bin(f"{prefix}_feat_share0.bin", x0)
        save_bin(f"{prefix}_feat_share1.bin", x1)
        save_bin(f"{prefix}_y_share0.bin", y0)
        save_bin(f"{prefix}_y_share1.bin", y1)
        save_bin(f"{prefix}_test_mask_share0.bin", tm0)
        save_bin(f"{prefix}_test_mask_share1.bin", tm1)
        save_bin(f"{prefix}_y.bin", y.astype(np.int64))
        save_bin(f"{prefix}_test_mask.bin", test_mask)
        save_bin(f"{prefix}_global_idx.bin", shard_global_idx.astype(np.int64))
        with open(f"{prefix}_meta.txt", "w") as fp:
            fp.write(f"Ns={ns}\nF={f_dim}\nC={c_dim}\nscale={args.scale}\n"
                     f"num_test={query_nodes.size}\n"
                     f"normalized={int(not args.no_normalize)}\n"
                     f"mia_query=1\n")
        print(f"  shard {s}: Ns={ns} query_rows={int(test_mask.sum())}")

    sources = export_weights(weight_dir, out_root, sorted(shard_data.keys()),
                             args.model_kind, set(affected), len(unlearn_nodes),
                             f_dim, c_dim, args.scale, rng)
    with open(out_root / "meta.txt", "w") as fp:
        fp.write(f"dataset={args.dataset}\nN={n_nodes}\nF={f_dim}\nC={c_dim}\n"
                 f"k={k}\nscale={args.scale}\nv0_size=0\n"
                 f"normalized={int(not args.no_normalize)}\n"
                 f"mia_query=1\nmodel_kind={args.model_kind}\n"
                 f"num_query={query_nodes.size}\n")
    with open(out_root / "weight_sources.txt", "w") as fp:
        for s in sorted(sources):
            fp.write(f"{s}: {sources[s]}\n")

    print(f"Done. Output dir: {out_root}")


if __name__ == "__main__":
    main()
