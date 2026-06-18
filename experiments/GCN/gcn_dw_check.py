#!/usr/bin/env python3
# Compare FSS 1-epoch weight update vs numpy 1-step (matching FSS dZ/Bp, lr).
import sys, re, numpy as np
D = sys.argv[1]; TRAINED = sys.argv[2]; lr = float(sys.argv[3])
scale = 24
meta = open(D + "/meta.txt").read()
Ns=int(re.search(r"Ns=(\d+)",meta)[1]); F=int(re.search(r"F=(\d+)",meta)[1]); H=int(re.search(r"H=(\d+)",meta)[1]); C=int(re.search(r"C=(\d+)",meta)[1])
Bp = 1
while Bp < Ns: Bp <<= 1
def Lu(p,n): return np.fromfile(p,dtype=np.uint64,count=n)
def s2f(u): return u.astype(np.int64).astype(np.float64)/(1<<scale)
def rec(nm,n): return Lu(f"{D}/{nm}_share0.bin",n)+Lu(f"{D}/{nm}_share1.bin",n)
X=s2f(rec("feat",Ns*F)).reshape(Ns,F); A=s2f(rec("adj",Ns*Ns)).reshape(Ns,Ns); Yh=s2f(rec("y_onehot",Ns*C)).reshape(Ns,C)
wi=np.fromfile(D+"/weights.dat",dtype=np.float32); W1=wi[:F*H].reshape(F,H).astype(float); W2=wi[F*H:F*H+H*C].reshape(H,C).astype(float)
# numpy 1 step (FSS-matching: dZ=(P-Y)/Bp)
T1=X@W1; U1=A@T1; H1=np.maximum(U1,0); T2=H1@W2; Z=A@T2
Zc=Z-Z.max(1,keepdims=True); P=np.exp(Zc); P/=P.sum(1,keepdims=True)
dZ=(P-Yh)/Bp
dT2=A.T@dZ; dW2=H1.T@dT2; dH1=dT2@W2.T; dU1=(U1>0)*dH1; dT1=A.T@dU1; dW1=X.T@dT1
nW1=W1-lr*dW1; nW2=W2-lr*dW2
# FSS-trained (1 epoch) weights
w=np.fromfile(TRAINED,dtype=np.uint64); fW1=s2f(w[:F*H]).reshape(F,H); fW2=s2f(w[F*H:F*H+H*C]).reshape(H,C)
print(f"shard {D}: Ns={Ns} Bp={Bp} lr={lr}")
print(f"  W1 step: numpy dW range [{(-lr*dW1).min():.4f},{(-lr*dW1).max():.4f}]  fss dW range [{(fW1-W1).min():.4f},{(fW1-W1).max():.4f}]")
print(f"  W1: max|numpy_new - fss| = {np.abs(nW1-fW1).max():.5f}  mean = {np.abs(nW1-fW1).mean():.6f}")
print(f"  W2: max|numpy_new - fss| = {np.abs(nW2-fW2).max():.5f}  mean = {np.abs(nW2-fW2).mean():.6f}")
print(f"  W1 init->fss change mean|dW| = {np.abs(fW1-W1).mean():.6f} ; numpy step mean|dW| = {np.abs(lr*dW1).mean():.6f}")
# cosine of the update directions
du_f=(fW1-W1).ravel(); du_n=(nW1-W1).ravel()
cos=du_f@du_n/(np.linalg.norm(du_f)*np.linalg.norm(du_n)+1e-12)
print(f"  W1 update-direction cosine(numpy, fss) = {cos:.4f}")
