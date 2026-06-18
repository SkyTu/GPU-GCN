#!/usr/bin/env python3
# Canonical, node-id-tracked shard export for the oblivious SELECT (dataset-general).
# Each shard's nodes = sorted(comm[s] U test_indices) -> a KNOWN ordering so isX /
# per-shard sum / gather / removal / oracle all align. Reads OpenGU GraphEraser
# processed data (community partition + train_data) for ANY dataset; writes FSS
# shares (scale 12) + a PUBLIC padded per-shard node-id table + meta.txt (incl. the
# chosen query node X and its ground-truth shard/row for the verifier).
import argparse, pickle, numpy as np, os, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cora")
    ap.add_argument("--opengu-processed", default=None,
                    help="OpenGU processed/<ds> dir; default $OPENGU_ROOT/data/GraphEraser/processed/<ds>")
    ap.add_argument("--out-dir", default=None,
                    help="default ../datasets/<ds>_shards_canon next to this script")
    ap.add_argument("--node", type=int, default=-1,
                    help="query node X to unlearn; -1 = auto (smallest train node)")
    ap.add_argument("--scale", type=int, default=12)
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    B = a.opengu_processed
    if B is None:
        root = os.environ.get("OPENGU_ROOT")
        if not root:
            sys.exit("set --opengu-processed or OPENGU_ROOT (OpenGU install base).")
        B = os.path.join(root, "data", "GraphEraser", "processed", a.dataset)
    OUT = a.out_dir or os.path.normpath(os.path.join(here, "..", "datasets", f"{a.dataset}_shards_canon"))
    SCALE = a.scale
    rng = np.random.default_rng(0)

    comm = pickle.load(open(B + "/community_lpa_base_10_0", "rb"))
    td = pickle.load(open(B + "/train_data", "rb"))
    x_all = np.asarray(td.x, dtype=np.float64); y_all = np.asarray(td.y).ravel().astype(np.int64)
    ei = np.asarray(td.edge_index, dtype=np.int64)
    test = sorted(int(v) for v in np.asarray(td.test_indices).ravel())
    N, F = x_all.shape; C = int(y_all.max() + 1); K = len(comm)
    BIN = max(8, (N).bit_length())          # smallest 2^BIN > max node id (ids in [0,N))
    if (1 << BIN) <= N:
        BIN += 1
    PAD_ID = (1 << BIN) - 1                  # never equals a real id (<N) nor any X

    E = set()
    for u, v in zip(ei[0], ei[1]):
        u, v = int(u), int(v)
        if u != v:
            E.add((min(u, v), max(u, v)))

    def norm_adj(nodes):
        n = len(nodes); idx = {g: i for i, g in enumerate(nodes)}
        A = np.eye(n)
        for (u, v) in E:
            if u in idx and v in idx:
                A[idx[u], idx[v]] = 1; A[idx[v], idx[u]] = 1
        d = 1.0 / np.sqrt(A.sum(1)); return (A * d[:, None]) * d[None, :]

    def to_fixed(arr): return np.round(arr * (1 << SCALE)).astype(np.int64).view(np.uint64)

    def share(u):
        s0 = rng.integers(0, 1 << 64, size=u.shape, dtype=np.uint64)
        return s0, (u - s0).astype(np.uint64)

    shards = {s: sorted(set(int(v) for v in np.asarray(comm[s]).ravel()) | set(test))
              for s in sorted(comm.keys())}
    Ns_max = max(len(v) for v in shards.values())
    os.makedirs(OUT + "/shards", exist_ok=True)
    trainset = set(int(v) for v in np.asarray(td.train_indices).ravel())
    testset = set(test)

    for s in sorted(shards.keys()):
        nodes = shards[s]; n = len(nodes)
        A = norm_adj(nodes); feat = x_all[nodes]; lab = y_all[nodes]
        tr = np.array([1 if g in trainset and g not in testset else 0 for g in nodes], np.uint64)
        te = np.array([1 if g in testset else 0 for g in nodes], np.uint64)
        Ap = np.zeros((Ns_max, Ns_max)); Ap[:n, :n] = A
        Fp = np.zeros((Ns_max, F)); Fp[:n] = feat
        Lp = np.zeros(Ns_max, np.int64); Lp[:n] = lab
        yoh = np.zeros((Ns_max, C), np.float64); yoh[np.arange(n), lab.astype(np.int64)] = 1.0
        nodeids = np.full(Ns_max, PAD_ID, np.uint64); nodeids[:n] = np.array(nodes, np.uint64)
        trp = np.zeros(Ns_max, np.uint64); trp[:n] = tr
        tep = np.zeros(Ns_max, np.uint64); tep[:n] = te
        p = f"{OUT}/shards/shard_{s}"
        for nm, u in [("adj", to_fixed(Ap).ravel()), ("feat", to_fixed(Fp).ravel()),
                      ("y", Lp.view(np.uint64)), ("y_onehot", to_fixed(yoh).ravel())]:
            a0, a1 = share(u); a0.tofile(f"{p}_{nm}_share0.bin"); a1.tofile(f"{p}_{nm}_share1.bin")
        nodeids.tofile(f"{p}_nodeids.bin")
        trp.tofile(f"{p}_train_mask.bin"); tep.tofile(f"{p}_test_mask.bin")

    # choose query node X: explicit, else smallest train node (in some shard, meaningful to unlearn)
    X = a.node if a.node >= 0 else min(trainset)
    s_true = next(s for s in shards if X in shards[s])
    row = shards[s_true].index(X)

    with open(f"{OUT}/meta.txt", "w") as f:
        f.write(f"N={N}\nF={F}\nC={C}\nk={K}\nNs_max={Ns_max}\nscale={SCALE}\nbin={BIN}\n"
                f"pad_id={PAD_ID}\nnum_test={len(test)}\nX={X}\ntrue_shard={s_true}\nrow={row}\n")

    print(f"canonical export ({a.dataset}) -> {OUT}")
    print(f"  N={N} F={F} C={C} K={K}  Ns per shard={[len(shards[s]) for s in sorted(shards)]} Ns_max={Ns_max}")
    print(f"  scale={SCALE} bin={BIN} pad_id={PAD_ID}")
    print(f"  ORACLE: X={X} -> shard {s_true}, row {row}")


if __name__ == "__main__":
    main()
