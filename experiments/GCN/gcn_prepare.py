#!/usr/bin/env python3
# Prepare sharded Cora for the clean FSS GCN (GPU-GCN/experiments/GCN).
# numpy-only (no torch/torch_geometric). Produces, per shard <s>:
#   datasets/<out>/shard<s>/meta.txt           (Ns,F,H,C,scale)
#   datasets/<out>/shard<s>/feat_share{0,1}.bin (Ns*F  u64, fixed scale)
#   datasets/<out>/shard<s>/adj_share{0,1}.bin  (Ns*Ns u64, normalized adj)
#   datasets/<out>/shard<s>/y_onehot_share{0,1}.bin (Ns*C u64)
#   datasets/<out>/shard<s>/labels.bin          (Ns int64, public)
#   datasets/<out>/shard<s>/weights.dat         (float32 W1[F*H] then W2[H*C], Glorot)
# Sharding: --shards 1 => whole graph as one shard; >1 => balanced LPA-style
# community partition (numpy connected-greedy; placeholder for OpenGU select).
import argparse, os, tarfile, urllib.request
from pathlib import Path
import numpy as np

CORA_URL = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
CLASS_MAP = {"Case_Based":0,"Genetic_Algorithms":1,"Neural_Networks":2,
             "Probabilistic_Methods":3,"Reinforcement_Learning":4,"Rule_Learning":5,"Theory":6}

def float_to_fixed(arr, scale):
    return np.round(np.asarray(arr,dtype=np.float64)*(1<<scale)).astype(np.int64).view(np.uint64)

def split_shares(u, rng):
    a0 = rng.integers(0,2**63,size=u.shape,dtype=np.uint64) | (rng.integers(0,2,size=u.shape,dtype=np.uint64)<<np.uint64(63))
    return a0, (u - a0)

def save_bin(path, arr):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.asarray(arr).tofile(path)

def load_raw_cora(raw_dir):
    raw_dir = Path(raw_dir)
    content = raw_dir/"cora"/"cora.content"; cites = raw_dir/"cora"/"cora.cites"
    if not (content.exists() and cites.exists()):
        raw_dir.mkdir(parents=True, exist_ok=True)
        tgz = raw_dir/"cora.tgz"
        if not tgz.exists():
            print(f"downloading {CORA_URL}"); urllib.request.urlretrieve(CORA_URL, tgz)
        with tarfile.open(tgz,"r:gz") as t: t.extractall(raw_dir)
    pid, feats, labels = [], [], []
    for line in open(content, encoding="utf-8"):
        p = line.strip().split("\t"); pid.append(int(p[0]))
        feats.append([int(v) for v in p[1:-1]]); labels.append(CLASS_MAP[p[-1]])
    idx = {p:i for i,p in enumerate(pid)}
    edges=[]
    for line in open(cites, encoding="utf-8"):
        a,b = [int(v) for v in line.strip().split("\t")]
        if a in idx and b in idx: edges.append((idx[a], idx[b]))
    X = np.asarray(feats, np.float64); y = np.asarray(labels, np.int64)
    E = np.asarray(edges, np.int64).T if edges else np.zeros((2,0),np.int64)
    return X, y, E

def norm_feats(x):
    s = x.sum(1, keepdims=True); s[s==0]=1.0; return x/s

def norm_adj(sub_edges, n):
    a = np.zeros((n,n), np.float64)
    if sub_edges.size:
        a[sub_edges[0], sub_edges[1]] = 1.0; a[sub_edges[1], sub_edges[0]] = 1.0
    a += np.eye(n)
    d = a.sum(1); d[d==0]=1.0; dis = 1.0/np.sqrt(d)
    return (a*dis[:,None])*dis[None,:]

def shard_nodes(y, k, rng):
    # balanced random partition by class (stand-in for OpenGU constrained-LPA select).
    n = len(y); order = rng.permutation(n)
    return [order[i::k] for i in range(k)]  # k roughly-equal node lists

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets/cora_gcn_shards")
    ap.add_argument("--raw-dir", default="datasets/cora_raw")
    ap.add_argument("--scale", type=int, default=24)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    X, y, E = load_raw_cora(a.raw_dir)
    X = norm_feats(X); N, F = X.shape; C = int(y.max()+1); H = a.hidden
    print(f"full graph: N={N} F={F} C={C} shards={a.shards} scale={a.scale}")
    parts = shard_nodes(y, a.shards, rng) if a.shards>1 else [np.arange(N)]
    glob2new = {}
    for s, nodes in enumerate(parts):
        nodes = np.sort(nodes); Ns = len(nodes)
        remap = {int(g):i for i,g in enumerate(nodes)}
        nodeset = set(int(g) for g in nodes)
        se = [(remap[int(u)], remap[int(v)]) for u,v in zip(E[0],E[1])
              if int(u) in nodeset and int(v) in nodeset] if E.size else []
        se = np.asarray(se, np.int64).T if se else np.zeros((2,0),np.int64)
        Xs = X[nodes]; ys = y[nodes]; As = norm_adj(se, Ns)
        Yoh = np.zeros((Ns,C), np.uint64); Yoh[np.arange(Ns), ys] = np.uint64(1<<a.scale)
        d = Path(a.out)/f"shard{s}"; d.mkdir(parents=True, exist_ok=True)
        for nm,arr in (("feat",float_to_fixed(Xs,a.scale)),("adj",float_to_fixed(As,a.scale))):
            s0,s1 = split_shares(arr.reshape(-1), rng)
            save_bin(d/f"{nm}_share0.bin", s0); save_bin(d/f"{nm}_share1.bin", s1)
        s0,s1 = split_shares(Yoh.reshape(-1), rng)
        save_bin(d/"y_onehot_share0.bin", s0); save_bin(d/"y_onehot_share1.bin", s1)
        save_bin(d/"labels.bin", ys.astype(np.int64))
        # Glorot float init weights: W1[F,H], W2[H,C]
        l1 = np.sqrt(6.0/(F+H)); l2 = np.sqrt(6.0/(H+C))
        W1 = rng.uniform(-l1,l1,(F,H)).astype(np.float32); W2 = rng.uniform(-l2,l2,(H,C)).astype(np.float32)
        with open(d/"weights.dat","wb") as f: f.write(W1.tobytes()); f.write(W2.tobytes())
        with open(d/"meta.txt","w") as f:
            f.write(f"Ns={Ns}\nF={F}\nH={H}\nC={C}\nscale={a.scale}\n")
        print(f"  shard{s}: Ns={Ns} edges={se.shape[1] if se.size else 0}")
    print(f"done -> {a.out}")

if __name__ == "__main__":
    main()
