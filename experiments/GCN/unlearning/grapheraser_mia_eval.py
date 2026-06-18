#!/usr/bin/env python3
"""
Evaluate GraphEraser MIA from two FSS query roots.

The score matches OpenGU's GraphEraser_MIA definition:

    score(node) = || posterior_before(node) - posterior_after(node) ||_2

Labels are 1 for unlearned nodes and 0 for held-out test nodes.  The query
roots must have matching mia_query_nodes.bin / mia_membership_labels.bin files
and aggregate_post_clear_<mode>.bin outputs from grapheraser_fss_l2_aggregate.
"""

import argparse
import os
import numpy as np


def parse_meta(path):
    out = {}
    with open(path) as f:
        for line in f:
            if "=" not in line:
                continue
            k, v = line.strip().split("=", 1)
            try:
                out[k] = int(v)
            except ValueError:
                out[k] = v
    return out


def u64_to_float(arr, scale):
    return arr.view(np.int64).astype(np.float64) / float(1 << scale)


def rank_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise SystemExit("ERROR: AUC needs at least one positive and one negative sample")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    # Average ranks for ties.
    sorted_scores = scores[order]
    i = 0
    while i < scores.size:
        j = i + 1
        while j < scores.size and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            avg = (i + 1 + j) / 2.0
            ranks[order[i:j]] = avg
        i = j
    pos_rank_sum = ranks[labels == 1].sum()
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def load_aggregate(root, mode):
    meta = parse_meta(os.path.join(root, "meta.txt"))
    qmeta = parse_meta(os.path.join(root, "mia_query_meta.txt"))
    scale = meta["scale"]
    c = meta["C"]
    n = qmeta["num_query"]
    path = os.path.join(root, "posteriors", f"aggregate_post_clear_{mode}.bin")
    if not os.path.exists(path):
        raise SystemExit(f"ERROR: missing {path}; run FSS L2 aggregate first")
    arr = np.fromfile(path, dtype=np.uint64)
    if arr.size != n * c:
        raise SystemExit(f"ERROR: {path} has {arr.size} values, expected {n*c}")
    return u64_to_float(arr, scale).reshape(n, c), {"scale": scale, "C": c, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before-root", required=True)
    ap.add_argument("--after-root", required=True)
    ap.add_argument("--mode", default="mean", choices=["mean", "weighted"])
    ap.add_argument("--dump-scores", default=None)
    args = ap.parse_args()

    nodes_before = np.fromfile(os.path.join(args.before_root, "mia_query_nodes.bin"), dtype=np.int64)
    nodes_after = np.fromfile(os.path.join(args.after_root, "mia_query_nodes.bin"), dtype=np.int64)
    labels = np.fromfile(os.path.join(args.before_root, "mia_membership_labels.bin"), dtype=np.uint8)
    labels_after = np.fromfile(os.path.join(args.after_root, "mia_membership_labels.bin"), dtype=np.uint8)
    if not np.array_equal(nodes_before, nodes_after):
        raise SystemExit("ERROR: before/after query node order differs")
    if not np.array_equal(labels, labels_after):
        raise SystemExit("ERROR: before/after membership labels differ")

    before, meta_b = load_aggregate(args.before_root, args.mode)
    after, meta_a = load_aggregate(args.after_root, args.mode)
    for key in ("scale", "C", "n"):
        if meta_b[key] != meta_a[key]:
            raise SystemExit(
                f"ERROR: before/after meta disagree on {key}: "
                f"{meta_b[key]} vs {meta_a[key]}")

    if args.mode == "weighted":
        alpha_paths = [os.path.join(r, "alpha.bin")
                       for r in (args.before_root, args.after_root)]
        missing = [p for p in alpha_paths if not os.path.exists(p)]
        if missing:
            raise SystemExit(
                f"ERROR: weighted mode requires alpha.bin in both roots; missing: {missing}")
        sigs = [np.fromfile(p, dtype=np.uint64).tobytes() for p in alpha_paths]
        if sigs[0] != sigs[1]:
            raise SystemExit(
                "ERROR: before/after alpha.bin differ; weighted MIA needs the "
                "same alpha applied to both roots (typically L3 trained on "
                "baseline, copied to query roots before L2).")

    scores = np.linalg.norm(before - after, axis=1)
    auc = rank_auc(labels, scores)

    pos = scores[labels == 1]
    neg = scores[labels == 0]
    print(f"mode={args.mode}  n={labels.size}  pos={pos.size}  neg={neg.size}")
    print(f"Attack AUC = {auc:.6f}")
    print(f"score mean: pos={pos.mean():.8f} neg={neg.mean():.8f}")
    print(f"score median: pos={np.median(pos):.8f} neg={np.median(neg):.8f}")

    if args.dump_scores:
        with open(args.dump_scores, "w") as fp:
            fp.write("node,label,score\n")
            for node, label, score in zip(nodes_before, labels, scores):
                fp.write(f"{int(node)},{int(label)},{score:.12g}\n")
        print(f"wrote {args.dump_scores}")


if __name__ == "__main__":
    main()
