#!/usr/bin/env bash
# =============================================================================
# run_unlearn_e2e.sh  —  End-to-end oblivious single-node UNLEARNING (all ciphertext)
#
#   1. SELECT   : DPF equality (1-bit isX) -> XOR shard one-hot -> gpuSelect over the
#                 concatenated [adj|feat|y_onehot|train_mask|test_mask] -> keep removal
#                 (B2A + gpuMul) zeros X's adjacency column and drops X from the train
#                 mask.  Output = secret shares of the X-removed subgraph; X & shard hidden.
#   2. ASSEMBLE : repackage the select shares into a standard_gcn dataset, re-quantizing
#                 the data scale 12 -> 24 (per-share x4096) and adding fresh Glorot weights.
#   3. RETRAIN  : standard_gcn_fss_train --secret-mask -- the gradient is beaver-multiplied
#                 by the SECRET train_mask (never revealed), with the secure piranha softmax.
#
#   Result == retraining "shard-minus-X from scratch" (GraphEraser node unlearning), but
#   the routing, the selected shard and the train mask are all secret-shared throughout.
#
# Usage:  ./run_unlearn_e2e.sh [EPOCHS] [LR]          (default: 80 16)
# =============================================================================
set -eo pipefail

# --- portable paths: derive the repo root from this script's own location ---
UNL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../experiments/GCN/unlearning
GCN="$(cd "$UNL/../../.." && pwd)"                     # GPU-GCN repo root (self-contained, no OblivGU dep)
DATA="${FSS_DATA_ROOT:-$GCN/experiments/GCN/datasets/cora_shards_canon}"
RETRAIN="$DATA/retrain"
CUDA_VERSION_BUILD="${CUDA_VERSION:-11.8}"; GPU_ARCH="${GPU_ARCH:-89}"
EPOCHS=${1:-80}; LR=${2:-16}
SEL_PORT="${SEL_PORT:-42310}"; TRAIN_PORT="${TRAIN_PORT:-42314}"
CONDA_ENV="${CONDA_ENV:-opengu}"

if [[ -z "${CONDA_SH:-}" ]]; then
    _cb="$(conda info --base 2>/dev/null || true)"
    [[ -n "$_cb" ]] && CONDA_SH="$_cb/etc/profile.d/conda.sh"
fi
if [[ -n "${CONDA_SH:-}" && -f "$CONDA_SH" ]]; then
    source "$CONDA_SH"; conda activate "$CONDA_ENV"
fi

[[ -f "$DATA/meta.txt" ]] || python "$UNL/canonical_export.py" "$DATA"

echo "==================== build (native GPU-GCN) ===================="
LOG="${TMPDIR:-/tmp}"
(cd "$GCN" && CUDA_VERSION=$CUDA_VERSION_BUILD GPU_ARCH=$GPU_ARCH make oblivious_select)        >"$LOG/b_sel.log" 2>&1 || { tail -20 "$LOG/b_sel.log"; exit 1; }
(cd "$GCN" && CUDA_VERSION=$CUDA_VERSION_BUILD GPU_ARCH=$GPU_ARCH make standard_gcn_fss_train)  >"$LOG/b_tr.log"  2>&1 || { tail -20 "$LOG/b_tr.log";  exit 1; }

echo "==================== 1. oblivious SELECT (X-removed subgraph) ===================="
export FSS_DATA_ROOT="$DATA"
python "$UNL/obliv_verify.py" prep >/dev/null          # secret-share X=270
pkill -9 -f oblivious_select 2>/dev/null || true; sleep 1
CUDA_VISIBLE_DEVICES=0 "$UNL/oblivious_select" 1 127.0.0.1 --port $SEL_PORT >"$LOG/sel_p1.log" 2>&1 &
sleep 2
CUDA_VISIBLE_DEVICES=0 "$UNL/oblivious_select" 0 127.0.0.1 --port $SEL_PORT 2>&1 | grep -E "RESULT|wrote"
wait

echo "==================== 2. ASSEMBLE retrain dataset (scale 12->24) ===================="
python "$UNL/assemble_retrain.py"

echo "==================== 3. secret-mask ciphertext RETRAIN ===================="
export FSS_DATA_ROOT="$RETRAIN"
pkill -9 -f "/tests/GNN/standard_gcn_fss_train " 2>/dev/null || true; sleep 1
B="$GCN/tests/GNN/standard_gcn_fss_train"          # native GPU-GCN trainer (no OblivGU dep)
CUDA_VISIBLE_DEVICES=0 "$B" 0 127.0.0.1 --port $TRAIN_PORT --epochs "$EPOCHS" --lr "$LR" --secret-mask --reveal-eval >"$LOG/rt_p0.log" 2>&1 &
P0=$!
sleep 3
CUDA_VISIBLE_DEVICES=0 "$B" 1 127.0.0.1 --port $TRAIN_PORT --epochs "$EPOCHS" --lr "$LR" --secret-mask --reveal-eval >"$LOG/rt_p1.log" 2>&1
wait $P0

echo "==================== retrain convergence (secret-mask, piranha softmax) ===================="
grep -qE "softmax=piranha" "$LOG/rt_p0.log" && echo "[ok] secure piranha softmax" || echo "[WARN] piranha softmax not detected"
grep -E "epoch=(1|10|20|40|60|80) " "$LOG/rt_p0.log" | \
    sed -E "s/.*(epoch=[0-9]+).*(train_acc=[0-9.]+) (test_acc=[0-9.]+).*/  \1 \2 \3/"
echo "[e2e] done.  weights -> $RETRAIN/weights"
