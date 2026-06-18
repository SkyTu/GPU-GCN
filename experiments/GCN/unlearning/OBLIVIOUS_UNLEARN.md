# Oblivious single-node graph unlearning (FSS / 2-PC)

End-to-end **ciphertext** unlearning of one node `X` from a GraphEraser-sharded GCN, with
the routing, the selected shard, and the train mask all kept **secret-shared** throughout.
Everything runs as a 2-party FSS protocol on loopback (single L40S vGPU, party 0/1, trusted
dealer offline keygen).

```
X (secret-shared)
  │  DPF equality            isX[k,j] = 1{nodeid_k[j] == X}        1-bit boolean
  │  XOR reduce              shard_oh[k] = XOR_j isX[k,j]          which shard owns X (hidden)
  │                          isX_sel[j]  = XOR_k isX[k,j]          X's row in that shard (hidden)
  ▼
GATHER  (one oblivious select over the concatenated [adj | feat | y_onehot | train | test])
  │  -> A_sel, feat_sel, yoh_sel, train_sel, test_sel             selected shard's data (shard hidden)
  ▼
KEEP    (X removal, still oblivious)
  │  A_masked      = A_sel .* keep_col   (keep = 1 - isX_sel)      zero X's adjacency COLUMN
  │  train_eff     = train_sel .* keep                            drop X from the loss
  ▼
ASSEMBLE  ->  standard_gcn dataset (re-quantize data scale 12 -> 24, fresh Glorot weights)
  ▼
RETRAIN   standard_gcn_fss_train --secret-mask
          gradient dZ is BEAVER-MULTIPLIED by the SECRET train mask (never revealed);
          secure piranha softmax.  == "retrain shard-minus-X from scratch" (GraphEraser),
          but X / shard / mask stay hidden.
```

Nothing about `X` or its shard is ever revealed: the DPF output is a masked 1-bit, the gather
opens only *masked* values, the keep removal is a Beaver product, and the retrain multiplies
the gradient by the *secret* mask. Revealing the train mask would fingerprint the shard (the
community partition is public), so it is kept a share — hence `--secret-mask`.

## Gather: two interchangeable variants

The 1-of-K gather (pick the shard that owns `X`) is implemented two ways; both are **bit-exact**
(produce the identical X-removed subgraph) and have the **same I/O**, so they are drop-in swaps:

| variant | file | gather primitive |
|---|---|---|
| **gpuSelect** (default) | `oblivious_select.cu` | 1-bit equality **DPF** -> XOR shard one-hot -> **`gpuSelect`** (FSS MUX) over the concatenated `[adj\|feat\|y_onehot\|train\|test]` |
| **beaver** (ablation) | `oblivious_select_beaver.cu` | **DPF-LUT** (arithmetic isX) -> open-once **Beaver matmul** `oh[1,K]·stack[K,W]` |

Both touch all `K` shards' data (you cannot select 1-of-K *secret* vectors without touching all
`K`, else the choice leaks) — so the dominant cost, opening `K×(N2+NF+NC)` of data, is identical.
The gpuSelect variant needs **fewer reconstruct rounds** (`gpuDpf` is local; one concatenated
select) -> lower latency.

### Measured (cora, K=10 shards, Ns=775, F=1433, C=7; 3 runs each)

| metric | gpuSelect | beaver | note |
|---|---|---|---|
| communication (total) | **303.81 MB** | 303.80 MB | equal — `K×data` open dominates both |
| &nbsp;&nbsp;– gather / keep | 274.93 / 28.88 MB | 274.93 / 28.87 MB | identical data open |
| reconstruct **rounds** | **11** | 14 | gpuSelect ~21% fewer |
| online **time** | **~290 ms** | ~368 ms | gpuSelect ~21% faster |

Correctness (`obliv_verify.py`, numpy oracle): both **PASS** — gathered adj/feat/y_onehot/masks
are bit-exact vs shard(X), X's adjacency column is zeroed, X is dropped from the train mask, all
with `X` and its shard hidden.

### Retrain (secret-mask, on the gpuSelect-selected subgraph)

```
epoch=1  train=0.0978 test=0.1476     (fresh Glorot init)
epoch=20 train=0.4667 test=0.3875
epoch=40 train=0.5644 test=0.4631
epoch=80 train=0.5822 test=0.4483     secure piranha softmax, clean exit
```

Verified **bit-identical** to a public-mask baseline trained from the same fresh weights (the
Beaver mask-multiply equals the public `rowZero`), i.e. the secret mask costs **zero** accuracy.
A single GraphEraser shard is intentionally weak (it sees 1/10 of the graph; the 10-shard mean
aggregate reaches ~0.83) — the deliverable is the working *ciphertext* mechanism, not one shard's
absolute score.

## How to run  (on <server>)

```bash
cd ~/GPU-GCN/experiments/GCN/unlearning

# (A) oblivious SELECT only — runs + verifies + reports comm/rounds/time
./run_oblivious_select.sh            # both variants (ablation)
./run_oblivious_select.sh gpuselect  # or just one

# (B) full end-to-end unlearning: SELECT -> ASSEMBLE -> secret-mask RETRAIN
./run_unlearn_e2e.sh                 # 80 epochs, lr 16
./run_unlearn_e2e.sh 80 16
```

Both scripts source the `opengu` conda env, build what's missing, and run the two FSS parties on
loopback (single GPU).

## Files (`experiments/GCN/unlearning/`)

| file | role |
|---|---|
| `oblivious_select.cu` | gpuSelect gather variant (main) |
| `oblivious_select_beaver.cu` | Beaver-matmul gather variant (ablation) |
| `canonical_export.py` | export the canonical `sorted(comm[s]∪test)` shard-set (node-id tracked) |
| `obliv_verify.py` | numpy oracle: secret-share `X`, check the revealed select output |
| `assemble_retrain.py` | select shares -> standard_gcn dataset (scale 12->24, fresh Glorot) |
| `run_oblivious_select.sh` | run + verify + benchmark the select (both variants) |
| `run_unlearn_e2e.sh` | full pipeline: select -> assemble -> secret-mask retrain |

Trainer (in `tests/GNN/`): `standard_gcn_fss_train` with `--secret-mask` — reads the
train mask as a secret share and beaver-multiplies it into the backward gradient
(`gcnBackwardRun`), fixed `--train-count` normalizer, secure piranha softmax.

## Notes / gotchas

- **Scale bridge** — the GraphEraser pipeline is fixed-point scale-12; standard_gcn converges at
  scale-24 (scale-12 stalls on truncation noise). `assemble_retrain.py` re-quantizes the secret
  shares 12->24 locally (×4096 per share, distributes over additive reconstruction).
- **`opMasked=false`** in `gpuKeyGenSelect` — otherwise the select carries a `randomMaskOut`
  output mask that must be subtracted; with `opMasked=false` the share is clean.
- **`LBW=64`** in the beaver variant — a 16-bit LUT output leaks `2^16` carries from the zero
  entries into the per-shard arithmetic sum; the gpuSelect variant avoids this entirely (boolean
  XOR has no carry).
- The comm-round counter is `g_commRounds` in `utils/sigma_comms.cpp` (incremented per
  `exchangeShares`); the `[RESULT ...]` line reports bytes / rounds / online-time.
