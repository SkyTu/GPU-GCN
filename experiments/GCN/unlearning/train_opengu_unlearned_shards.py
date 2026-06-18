#!/usr/bin/env python3
"""
Train missing OpenGU GraphEraser node-unlearning shard checkpoints.

This is a small standalone equivalent of GraphEraser_Attack._train_shard_model
for the batch node-unlearning case used by the local FSS tests.  It reads
OpenGU's saved partition/train_data, removes the default unlearning batch from
the affected shard train sets, retrains those shard GCNs, and writes checkpoints
with the same names OpenGU expects:

  $OPENGU_ROOT/data/GraphEraser/<dataset>/
    GCN_lpa_base_10_0.005_0_<shard>_0_<N>_unlearned
"""

import argparse
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


SUPPORTED_DATASETS = ("cora", "citeseer", "pubmed")
_OPENGU = os.environ.get("OPENGU_ROOT", "")   # OpenGU install base (contains data/, GULib-master/)
GU_ROOT_FMT = _OPENGU + "/data/GraphEraser/processed/{ds}"
WEIGHT_DIR_FMT = _OPENGU + "/data/GraphEraser/{ds}"
UNLEARN_LIST_FMT = (
    _OPENGU + "/GULib-master/data/unlearning_task/transductive/imbalanced/"
    "unlearning_nodes_0.1_{ds}_{run}.txt"
)
SUMMARY_FMT = "datasets/{ds}_shards_unlearned/run{run}/unlearn_summary.txt"
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

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        x = self.convs[0](x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        return self.convs[-1](x, edge_index)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def read_nodes(path):
    nodes = []
    with open(path) as fp:
        for raw in fp:
            line = raw.split("#", 1)[0].strip()
            if line:
                nodes.append(int(line))
    if not nodes:
        raise SystemExit(f"ERROR: empty unlearning list: {path}")
    return np.asarray(nodes, dtype=np.int64)


def build_node_to_shard(community):
    out = {}
    for shard, nodes in community.items():
        for node in np.asarray(nodes, dtype=np.int64):
            # Matches OpenGU dataset_utils.c2n_to_n2c overwrite behavior.
            out[int(node)] = int(shard)
    return out


def filter_edge_index_1(edge_index, node_indices):
    node_indices = np.sort(np.asarray(node_indices, dtype=np.int64))
    ei = edge_index.cpu().numpy() if hasattr(edge_index, "cpu") else np.asarray(edge_index)
    node_index = np.isin(ei, node_indices)
    col_index = np.nonzero(np.logical_and(node_index[0], node_index[1]))[0]
    kept = ei[:, col_index]
    return np.searchsorted(node_indices, kept).astype(np.int64)


def format_int_list(values):
    return "[" + ", ".join(str(int(v)) for v in values) + "]"


def write_summary(path, ds, run, source, nodes, node_to_shard, by_shard, weight_dir):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    affected = sorted(by_shard.keys())
    with open(path, "w") as fp:
        fp.write("unlearning_mode=node\n")
        fp.write("unlearning_batching=batch\n")
        fp.write(f"run_id=run{run}\n")
        fp.write(f"unlearn_source={source}\n")
        fp.write(f"num_unlearning_nodes={len(nodes)}\n")
        fp.write(f"num_retrained_shards={len(affected)}\n")
        fp.write(f"unlearning_node_ids={format_int_list(nodes)}\n")
        fp.write("node_to_shard={\n")
        for node in nodes:
            fp.write(f"  {int(node)}: {node_to_shard[int(node)]}\n")
        fp.write("}\n")
        fp.write(f"affected_shards={format_int_list(affected)}\n")
        fp.write("unlearning_indices_by_shard={\n")
        for shard in affected:
            fp.write(f"  {shard}: {format_int_list(by_shard[shard])}\n")
        fp.write("}\n")
        fp.write("weight_source_per_shard={\n")
        for shard in range(10):
            base = Path(weight_dir) / WEIGHT_FILE_FMT.format(s=shard)
            unl = Path(weight_dir) / UNLEARNED_WEIGHT_FILE_FMT.format(s=shard, n=len(nodes))
            if shard in affected:
                fp.write(f"  {shard}: unlearned: {unl}\n")
            else:
                fp.write(f"  {shard}: original: {base}\n")
        fp.write("}\n")
    print(f"[{ds}] wrote summary: {path}")


def train_one_shard(train_data, community, shard, remove_nodes, epochs, device):
    shard_train = np.asarray(community[shard], dtype=np.int64)
    remaining = np.setdiff1d(shard_train, np.asarray(remove_nodes, dtype=np.int64))
    test_indices = np.asarray(train_data.test_indices, dtype=np.int64)
    shard_indices = np.union1d(remaining, test_indices)

    x = train_data.x[shard_indices].to(device)
    y = train_data.y[shard_indices].to(device)
    ei = filter_edge_index_1(train_data.edge_index, shard_indices)
    edge_index = torch.from_numpy(ei).long().to(device)
    train_mask = torch.from_numpy(np.isin(shard_indices, remaining)).bool().to(device)
    test_mask = torch.from_numpy(np.isin(shard_indices, test_indices)).bool().to(device)

    model = GCNNet(int(train_data.x.shape[1]), int(train_data.y.max().item()) + 1).to(device)
    model.reset_parameters()
    opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.000001)
    best_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[train_mask], y[train_mask])
        loss.backward()
        opt.step()
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                pred = model(x, edge_index).argmax(dim=1)
                best_f1 = float((pred[test_mask] == y[test_mask]).float().mean().item())
    return model.cpu().state_dict(), best_f1, int(train_mask.sum().item()), int(test_mask.sum().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    seed_everything(args.seed)
    ds = args.dataset
    gu_root = Path(GU_ROOT_FMT.format(ds=ds))
    weight_dir = Path(WEIGHT_DIR_FMT.format(ds=ds))
    unlearn_path = Path(UNLEARN_LIST_FMT.format(ds=ds, run=args.run))
    summary_path = Path(SUMMARY_FMT.format(ds=ds, run=args.run))

    with open(gu_root / TRAIN_DATA_PICKLE, "rb") as f:
        train_data = pickle.load(f)
    with open(gu_root / COMMUNITY_PICKLE, "rb") as f:
        community = pickle.load(f)

    nodes = read_nodes(unlearn_path)
    node_to_shard = build_node_to_shard(community)
    missing_nodes = [int(n) for n in nodes if int(n) not in node_to_shard]
    if missing_nodes:
        raise SystemExit(f"ERROR: {len(missing_nodes)} unlearning nodes are not in community map")

    by_shard = defaultdict(list)
    for node in nodes:
        by_shard[node_to_shard[int(node)]].append(int(node))
    affected = sorted(by_shard.keys())
    print(f"[{ds}] unlearn batch={len(nodes)} affected_shards={affected}")

    write_summary(summary_path, ds, args.run, unlearn_path, nodes,
                  node_to_shard, by_shard, weight_dir)

    device = torch.device(args.device)
    for shard in affected:
        out_path = weight_dir / UNLEARNED_WEIGHT_FILE_FMT.format(s=shard, n=len(nodes))
        if out_path.exists() and not args.force:
            print(f"[{ds}] shard {shard}: exists, skip {out_path}")
            continue
        print(f"[{ds}] shard {shard}: training, remove={len(by_shard[shard])}")
        state, f1, n_train, n_test = train_one_shard(
            train_data, community, shard, by_shard[shard], args.epochs, device)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, out_path)
        print(f"[{ds}] shard {shard}: wrote {out_path} train={n_train} test={n_test} f1={f1:.4f}")


if __name__ == "__main__":
    main()
