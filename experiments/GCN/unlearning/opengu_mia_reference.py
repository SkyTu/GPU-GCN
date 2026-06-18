#!/usr/bin/env python3
"""
Standalone OpenGU/PyG GraphEraser MIA reference.

This reproduces the core of OpenGU's
GULib-master/attack/Attack_methods/GraphEraser_MIA.py without
entering the full OpenGU pipeline:

  score(node) = || mean_s posterior_original_s(node)
                  - mean_s posterior_unlearned_s(node) ||_2

Modes:
  batch: query all selected nodes in one induced subgraph per shard.  This
         matches prepare_mia_query.py and the current FSS adapter.
  exact: query one node at a time, matching OpenGU's _query_target_model loop
         more closely.
"""

import argparse
import ast
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


import os
_OPENGU = os.environ.get("OPENGU_ROOT", "")   # OpenGU install base; or pass --gu-root / --weight-dir
GU_ROOT_FMT = _OPENGU + "/data/GraphEraser/processed/{ds}"
WEIGHT_DIR_FMT = _OPENGU + "/data/GraphEraser/{ds}"
UNLEARN_SUMMARY_FMT = "datasets/{ds}_shards_unlearned/run0/unlearn_summary.txt"
COMMUNITY_PICKLE = "community_lpa_base_10_0"
TRAIN_DATA_PICKLE = "train_data"
WEIGHT_FILE_FMT = "GCN_lpa_base_10_0.005_0_{s}_0"
UNLEARNED_WEIGHT_FILE_FMT = WEIGHT_FILE_FMT + "_{n}_unlearned"


class GCNNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            GCNConv(in_channels, 64),
            GCNConv(64, out_channels),
        ])

    def forward(self, x, edge_index):
        x = self.convs[0](x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        return self.convs[-1](x, edge_index)


def parse_meta(path):
    out = {}
    with open(path) as f:
        for raw in f:
            if "=" not in raw:
                continue
            k, v = raw.strip().split("=", 1)
            out[k] = v
    return out


def read_summary(path):
    meta = parse_meta(path)
    nodes = np.asarray(ast.literal_eval(meta["unlearning_node_ids"]), dtype=np.int64)
    affected = set(int(x) for x in ast.literal_eval(meta.get("affected_shards", "[]")))
    return nodes, affected


def rank_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < scores.size:
        j = i + 1
        while j < scores.size and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def filter_edge_index_1(edge_index, node_indices):
    node_indices = np.sort(np.asarray(node_indices, dtype=np.int64))
    ei = edge_index.cpu().numpy() if hasattr(edge_index, "cpu") else np.asarray(edge_index)
    node_index = np.isin(ei, node_indices)
    col_index = np.nonzero(np.logical_and(node_index[0], node_index[1]))[0]
    kept = ei[:, col_index]
    return np.searchsorted(node_indices, kept).astype(np.int64)


def load_models(weight_dir, shards, affected, num_unlearn, f_dim, c_dim, device):
    original = {}
    unlearned = {}
    for s in shards:
        m0 = GCNNet(f_dim, c_dim).to(device)
        p0 = Path(weight_dir) / WEIGHT_FILE_FMT.format(s=s)
        m0.load_state_dict(torch.load(p0, map_location=device))
        m0.eval()
        original[int(s)] = m0

        if int(s) in affected:
            p1 = Path(weight_dir) / UNLEARNED_WEIGHT_FILE_FMT.format(s=s, n=num_unlearn)
        else:
            p1 = p0
        m1 = GCNNet(f_dim, c_dim).to(device)
        m1.load_state_dict(torch.load(p1, map_location=device))
        m1.eval()
        unlearned[int(s)] = m1
    return original, unlearned


@torch.no_grad()
def query_batch(train_data, community, models, query_nodes, posterior_kind, device):
    query_nodes = np.sort(np.asarray(query_nodes, dtype=np.int64))
    outs = []
    for s in sorted(community.keys()):
        shard_indices = np.union1d(np.asarray(community[s], dtype=np.int64), query_nodes)
        x = train_data.x[shard_indices].to(device)
        ei = filter_edge_index_1(train_data.edge_index, shard_indices)
        edge_index = torch.from_numpy(ei).long().to(device)
        mask = np.isin(shard_indices, query_nodes)
        logits = models[int(s)](x, edge_index)
        if posterior_kind == "logsoftmax":
            post = F.log_softmax(logits[mask], dim=-1)
        else:
            post = F.softmax(logits[mask], dim=-1)
        outs.append(post.cpu().numpy())
    return np.mean(np.stack(outs, axis=0), axis=0)


@torch.no_grad()
def query_exact(train_data, community, models, query_nodes, posterior_kind, device):
    rows = []
    for node in query_nodes:
        outs = []
        one = np.asarray([int(node)], dtype=np.int64)
        for s in sorted(community.keys()):
            shard_indices = np.union1d(np.asarray(community[s], dtype=np.int64), one)
            x = train_data.x[shard_indices].to(device)
            ei = filter_edge_index_1(train_data.edge_index, shard_indices)
            edge_index = torch.from_numpy(ei).long().to(device)
            mask = np.isin(shard_indices, one)
            logits = models[int(s)](x, edge_index)
            if posterior_kind == "logsoftmax":
                post = F.log_softmax(logits[mask], dim=-1)
            else:
                post = F.softmax(logits[mask], dim=-1)
            outs.append(post.squeeze(0).cpu().numpy())
        rows.append(np.mean(np.stack(outs, axis=0), axis=0))
    return np.stack(rows, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cora", choices=["cora", "citeseer", "pubmed"])
    ap.add_argument("--gu-root", default=None)
    ap.add_argument("--weight-dir", default=None)
    ap.add_argument("--unlearn-summary", default=None)
    ap.add_argument("--num-pos", type=int, default=100)
    ap.add_argument("--num-neg", type=int, default=100)
    ap.add_argument("--mode", choices=["batch", "exact"], default="batch")
    ap.add_argument("--posterior", choices=["softmax", "logsoftmax"], default="logsoftmax")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for sampling pos/neg query nodes (must match prepare_mia_query)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    gu_root = Path(args.gu_root or GU_ROOT_FMT.format(ds=args.dataset))
    weight_dir = Path(args.weight_dir or WEIGHT_DIR_FMT.format(ds=args.dataset))
    summary_path = Path(args.unlearn_summary or UNLEARN_SUMMARY_FMT.format(ds=args.dataset))

    with open(gu_root / TRAIN_DATA_PICKLE, "rb") as f:
        train_data = pickle.load(f)
    with open(gu_root / COMMUNITY_PICKLE, "rb") as f:
        community = pickle.load(f)

    unlearn_nodes, affected = read_summary(summary_path)
    test_pool = np.asarray(train_data.test_indices, dtype=np.int64)
    n_pos = min(args.num_pos, unlearn_nodes.size)
    n_neg = min(args.num_neg, test_pool.size)
    pos = rng.choice(unlearn_nodes, size=n_pos, replace=False)
    neg = rng.choice(test_pool, size=n_neg, replace=False)
    inter = np.intersect1d(pos, neg)
    if inter.size:
        raise SystemExit(
            f"ERROR: pos/neg overlap on {inter.size} node(s): {inter[:10].tolist()}")
    query_nodes = np.unique(np.concatenate([pos, neg]).astype(np.int64))
    query_nodes.sort()
    labels = np.isin(query_nodes, pos).astype(np.uint8)

    f_dim = int(train_data.x.shape[1])
    c_dim = int(train_data.y.max().item()) + 1
    device = torch.device(args.device)
    original, unlearned = load_models(weight_dir, sorted(community.keys()),
                                      affected, len(unlearn_nodes),
                                      f_dim, c_dim, device)

    query_fn = query_batch if args.mode == "batch" else query_exact
    before = query_fn(train_data, community, original, query_nodes, args.posterior, device)
    after = query_fn(train_data, community, unlearned, query_nodes, args.posterior, device)
    scores = np.linalg.norm(before - after, axis=1)
    auc = rank_auc(labels, scores)
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]

    print(f"OpenGU reference: mode={args.mode} posterior={args.posterior} device={device}")
    print(f"n={labels.size} pos={pos_scores.size} neg={neg_scores.size}")
    print(f"Attack AUC = {auc:.6f}")
    print(f"score mean: pos={pos_scores.mean():.8f} neg={neg_scores.mean():.8f}")
    print(f"score median: pos={np.median(pos_scores):.8f} neg={np.median(neg_scores):.8f}")


if __name__ == "__main__":
    main()
