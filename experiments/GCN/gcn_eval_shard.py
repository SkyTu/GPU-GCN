#!/usr/bin/env python3
# Evaluate FSS-trained shard weights vs numpy, on the shard's own nodes.
import sys, re
import numpy as np
D = sys.argv[1] if len(sys.argv) > 1 else "datasets/cora_gcn_shards/shard0"
TRAINED = sys.argv[2] if len(sys.argv) > 2 else "/tmp/trained_shard0.dat"
scale = 24
meta = open(D + "/meta.txt").read()
Ns = int(re.search(r"Ns=(\d+)", meta)[1]); F = int(re.search(r"F=(\d+)", meta)[1])
H = int(re.search(r"H=(\d+)", meta)[1]); C = int(re.search(r"C=(\d+)", meta)[1])
def Lu(p, n): return np.fromfile(p, dtype=np.uint64, count=n)
def s2f(u): return u.astype(np.int64).astype(np.float64) / (1 << scale)
def rec(nm, n): return Lu(f"{D}/{nm}_share0.bin", n) + Lu(f"{D}/{nm}_share1.bin", n)
X = s2f(rec("feat", Ns * F)).reshape(Ns, F)
A = s2f(rec("adj", Ns * Ns)).reshape(Ns, Ns)
Yh = s2f(rec("y_onehot", Ns * C)).reshape(Ns, C)
labels = np.fromfile(D + "/labels.bin", dtype=np.int64, count=Ns)

def fwd(W1, W2):
    T1 = X @ W1; U1 = A @ T1; H1 = np.maximum(U1, 0); T2 = H1 @ W2; Z = A @ T2
    return Z
def acc(Z): return (Z.argmax(1) == labels).mean()

# init float weights (Glorot) used by the FSS trainer
wi = np.fromfile(D + "/weights.dat", dtype=np.float32)
W1i = wi[:F*H].reshape(F, H).astype(np.float64); W2i = wi[F*H:F*H+H*C].reshape(H, C).astype(np.float64)
print(f"shard {D}: Ns={Ns} F={F} H={H} C={C}")
print(f"  init      acc = {acc(fwd(W1i, W2i)):.3f}")

# FSS-trained weights (u64 fixed scale24: W1[F*H] then W2[H*C])
w = np.fromfile(TRAINED, dtype=np.uint64)
W1f = s2f(w[:F*H]).reshape(F, H); W2f = s2f(w[F*H:F*H+H*C]).reshape(H, C)
print(f"  FSS-train acc = {acc(fwd(W1f, W2f)):.3f}")

# numpy SGD reference from the same init (no bias, lr swept)
nt = Ns
for lr in [1.0, 4.0, 16.0]:
    w1, w2 = W1i.copy(), W2i.copy()
    for ep in range(20):
        T1 = X@w1; U1 = A@T1; H1 = np.maximum(U1, 0); T2 = H1@w2; Z = A@T2
        Zc = Z - Z.max(1, keepdims=True); P = np.exp(Zc); P /= P.sum(1, keepdims=True)
        dZ = (P - Yh) / nt
        dT2 = A.T@dZ; dW2 = H1.T@dT2; dH1 = dT2@w2.T; dU1 = (U1 > 0)*dH1; dT1 = A.T@dU1; dW1 = X.T@dT1
        w1 -= lr*dW1; w2 -= lr*dW2
    print(f"  numpy SGD lr={lr} (20ep) acc = {acc(fwd(w1, w2)):.3f}")
