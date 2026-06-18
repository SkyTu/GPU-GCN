#!/usr/bin/env python3
# numpy 2-layer GCN on full Cora (transductive), no bias, plain SGD -- matches
# the FSS model. Sweeps H and lr to see what gives good test accuracy.
import numpy as np, sys
from pathlib import Path
RAW = sys.argv[1] if len(sys.argv) > 1 else "datasets/cora_raw"
CLASS = {"Case_Based":0,"Genetic_Algorithms":1,"Neural_Networks":2,"Probabilistic_Methods":3,
         "Reinforcement_Learning":4,"Rule_Learning":5,"Theory":6}
def load(raw):
    raw = Path(raw); pid,feats,lab = [],[],[]
    for ln in open(raw/"cora"/"cora.content"):
        p=ln.strip().split("\t"); pid.append(int(p[0])); feats.append([int(v) for v in p[1:-1]]); lab.append(CLASS[p[-1]])
    idx={p:i for i,p in enumerate(pid)}; E=[]
    for ln in open(raw/"cora"/"cora.cites"):
        a,b=[int(v) for v in ln.strip().split("\t")]
        if a in idx and b in idx: E.append((idx[a],idx[b]))
    return np.asarray(feats,float), np.asarray(lab), np.asarray(E).T
X,y,E = load(RAW); N,F = X.shape; C=int(y.max()+1)
X = X / np.clip(X.sum(1,keepdims=True),1,None)          # row-normalize feats
A = np.zeros((N,N)); A[E[0],E[1]]=1; A[E[1],E[0]]=1; A+=np.eye(N)
d = 1/np.sqrt(A.sum(1)); A = (A*d[:,None])*d[None,:]      # sym-normalized adj
rng = np.random.default_rng(0)
train = np.zeros(N,bool); test = np.zeros(N,bool); rem=[]
for c in range(C):
    ci = np.where(y==c)[0]; rng.shuffle(ci); train[ci[:20]]=True; rem += list(ci[20:])
rem=np.array(rem); rng.shuffle(rem); test[rem[:1000]]=True
nt = train.sum()
def run(H, lr, epochs=300, seed=1):
    r=np.random.default_rng(seed)
    W1=r.uniform(-np.sqrt(6/(F+H)),np.sqrt(6/(F+H)),(F,H)); W2=r.uniform(-np.sqrt(6/(H+C)),np.sqrt(6/(H+C)),(H,C))
    best=0
    for ep in range(epochs):
        T1=X@W1; U1=A@T1; H1=np.maximum(U1,0); T2=H1@W2; Z=A@T2
        Zc=Z-Z.max(1,keepdims=True); P=np.exp(Zc); P/=P.sum(1,keepdims=True)
        dZ=P.copy(); dZ[np.arange(N),y]-=1; dZ[~train]=0; dZ/=nt
        dT2=A.T@dZ; dW2=H1.T@dT2; dH1=dT2@W2.T; dU1=(U1>0)*dH1; dT1=A.T@dU1; dW1=X.T@dT1
        W1-=lr*dW1; W2-=lr*dW2
        te=(Z.argmax(1)[test]==y[test]).mean(); best=max(best,te)
    return (Z.argmax(1)[test]==y[test]).mean(), best
print(f"Cora full: N={N} F={F} C={C} train={nt} test={test.sum()} (no-bias SGD, 300 epochs)")
print(f"{'H':>4} {'lr':>10} {'test@end':>9} {'test@best':>9}")
for H in (64,128):
    for lr in (0.00125, 0.0125, 0.125, 1.0, 4.0, 16.0):
        e,b = run(H,lr); print(f"{H:>4} {lr:>10.5f} {e:>9.3f} {b:>9.3f}")
