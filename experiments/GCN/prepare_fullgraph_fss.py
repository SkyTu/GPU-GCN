#!/usr/bin/env python3
# Dataset-general full-graph FSS dataset for standard_gcn_fss_train.
# Loads via PyG Planetoid (Cora/CiteSeer/PubMed), builds the symmetric-normalized
# adjacency D^-1/2 (A+I) D^-1/2 + row-normalized features, quantizes to scale-24
# fixed point as 2-of-2 additive shares, and adds fresh Glorot weights. No cora
# hardcoding; dims (N,F,C) come from the dataset. Public train/test masks (plain
# full-graph training, not the oblivious-unlearning secret-mask variant).
import argparse, os, numpy as np
from pathlib import Path
from torch_geometric.datasets import Planetoid


def float_to_fixed(arr, scale):
    return np.round(np.asarray(arr, np.float64) * (1 << scale)).astype(np.int64).view(np.uint64)


def split_shares(u, rng):
    s0 = rng.integers(0, 1 << 64, size=u.shape, dtype=np.uint64)
    return s0, (u.astype(np.uint64) - s0).astype(np.uint64)


def normalize_features(x):
    s = x.sum(1, keepdims=True); s[s == 0] = 1.0
    return x / s


def build_norm_adj(edge_index, n):
    a = np.zeros((n, n), np.float64)
    a[edge_index[0], edge_index[1]] = 1.0
    a = np.maximum(a, a.T)            # symmetrize (undirected)
    a += np.eye(n)                    # self-loops
    deg = a.sum(1); deg[deg == 0] = 1.0
    di = 1.0 / np.sqrt(deg)
    return (a * di[:, None]) * di[None, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=("Cora", "CiteSeer", "PubMed"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scale", type=int, default=24)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pyg-root", default="/tmp/pyg")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    d = Planetoid(root=os.path.join(a.pyg_root, a.dataset), name=a.dataset)[0]
    x = normalize_features(d.x.numpy().astype(np.float64))
    y = d.y.numpy().astype(np.int64)
    n, f = x.shape; c = int(y.max() + 1); h = a.hidden
    train_mask = d.train_mask.numpy().astype(np.uint8)
    test_mask = d.test_mask.numpy().astype(np.uint8)
    adj = build_norm_adj(d.edge_index.numpy(), n)

    out = Path(a.out_dir)
    (out / "graph").mkdir(parents=True, exist_ok=True)
    (out / "weights").mkdir(exist_ok=True)
    (out / "outputs").mkdir(exist_ok=True)

    yoh = np.zeros((n, c), np.uint64); yoh[np.arange(n), y] = np.uint64(1 << a.scale)
    for nm, arr in (("feat", float_to_fixed(x, a.scale)),
                    ("adj", float_to_fixed(adj, a.scale)),
                    ("y_onehot", yoh)):
        s0, s1 = split_shares(arr.ravel(), rng)
        s0.tofile(out / "graph" / f"{nm}_share0.bin")
        s1.tofile(out / "graph" / f"{nm}_share1.bin")
    y.astype(np.int64).tofile(out / "graph" / "labels.bin")
    train_mask.tofile(out / "graph" / "train_mask.bin")
    test_mask.tofile(out / "graph" / "test_mask.bin")

    l1 = np.sqrt(6.0 / (f + h)); l2 = np.sqrt(6.0 / (h + c))
    W = {"W1": float_to_fixed(rng.uniform(-l1, l1, (f, h)), a.scale),
         "b1": float_to_fixed(np.zeros(h), a.scale),
         "W2": float_to_fixed(rng.uniform(-l2, l2, (h, c)), a.scale),
         "b2": float_to_fixed(np.zeros(c), a.scale)}
    for nm, u in W.items():
        s0, s1 = split_shares(u.ravel(), rng)
        s0.tofile(out / "weights" / f"{nm}_share0.bin")
        s1.tofile(out / "weights" / f"{nm}_share1.bin")

    with open(out / "meta.txt", "w") as fm:
        fm.write(f"N={n}\nF={f}\nC={c}\nH={h}\nscale={a.scale}\n"
                 f"train_count={int(train_mask.sum())}\ntest_count={int(test_mask.sum())}\n"
                 f"normalized=1\nsecret_mask=0\n")
    print(f"[prep] {a.dataset}: N={n} F={f} C={c} H={h} scale={a.scale} "
          f"train={int(train_mask.sum())} test={int(test_mask.sum())} -> {out}")


if __name__ == "__main__":
    main()
