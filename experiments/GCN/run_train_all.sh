#!/usr/bin/env bash
# =============================================================================
# run_train_all.sh — full-graph ciphertext GCN training (2-party FSS) over datasets.
#
# For each dataset:  PyG Planetoid -> scale-24 FSS dataset (prepare_fullgraph_fss.py)
#   -> native standard_gcn_fss_train (loopback 2PC) -> report per-epoch test accuracy.
#
# Usage:   ./run_train_all.sh [DATASET ...]          (default: Cora CiteSeer)
#          EPOCHS=80 LR=16 PORT=42400 ./run_train_all.sh Cora
# Env:     CONDA_ENV (opengu), CUDA_VERSION (11.8), GPU_ARCH (89) — override as needed.
# Note:    PubMed (N=19717) full-graph dense FSS is IMPRACTICAL (O(N^2) keygen blows up);
#          it is skipped. Use the sharded GraphEraser pipeline for large graphs.
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"      # experiments/GCN
GCN="$(cd "$SCRIPT_DIR/../.." && pwd)"                          # GPU-GCN repo root
EPOCHS="${EPOCHS:-80}"; LR="${LR:-16}"; PORT="${PORT:-42400}"
CONDA_ENV="${CONDA_ENV:-opengu}"; CUDA_VERSION_BUILD="${CUDA_VERSION:-11.8}"; GPU_ARCH="${GPU_ARCH:-89}"
LOG="${TMPDIR:-/tmp}"
DATASETS=("$@"); [ ${#DATASETS[@]} -eq 0 ] && DATASETS=(Cora CiteSeer)

if [ -z "${CONDA_SH:-}" ]; then
    for _c in "$(conda info --base 2>/dev/null)" "${CONDA_PREFIX:-}" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniconda" /opt/conda; do
        [ -n "$_c" ] && [ -f "$_c/etc/profile.d/conda.sh" ] && { CONDA_SH="$_c/etc/profile.d/conda.sh"; break; }
    done
fi
[ -n "${CONDA_SH:-}" ] && [ -f "$CONDA_SH" ] && { source "$CONDA_SH"; conda activate "$CONDA_ENV"; }

echo "==================== build standard_gcn_fss_train ===================="
( cd "$GCN" && CUDA_VERSION=$CUDA_VERSION_BUILD GPU_ARCH=$GPU_ARCH make standard_gcn_fss_train ) >"$LOG/build_train.log" 2>&1 \
    || { tail -20 "$LOG/build_train.log"; exit 1; }
B="$GCN/tests/GNN/standard_gcn_fss_train"

for DS in "${DATASETS[@]}"; do
    dl="$(echo "$DS" | tr '[:upper:]' '[:lower:]')"
    if [ "$dl" = "pubmed" ]; then
        echo "[skip] PubMed full-graph dense FSS impractical (O(N^2)); use the sharded GraphEraser pipeline."
        continue
    fi
    DSDIR="$SCRIPT_DIR/datasets/${dl}_standard_gcn"
    echo "==================== $DS : prepare FSS dataset ===================="
    python3 "$SCRIPT_DIR/prepare_fullgraph_fss.py" --dataset "$DS" --out-dir "$DSDIR" || { echo "[$DS] prep failed"; continue; }
    echo "==================== $DS : train (epochs=$EPOCHS lr=$LR) ===================="
    pkill -9 -f "/tests/GNN/standard_gcn_fss_train " 2>/dev/null || true; sleep 1
    FSS_DATA_ROOT="$DSDIR" CUDA_VISIBLE_DEVICES=0 "$B" 0 127.0.0.1 --port $PORT --epochs "$EPOCHS" --lr "$LR" --reveal-eval >"$LOG/train_${dl}_p0.log" 2>&1 &
    P0=$!; sleep 3
    FSS_DATA_ROOT="$DSDIR" CUDA_VISIBLE_DEVICES=0 "$B" 1 127.0.0.1 --port $PORT --epochs "$EPOCHS" --lr "$LR" --reveal-eval >"$LOG/train_${dl}_p1.log" 2>&1
    wait $P0
    echo "--- $DS test accuracy (sampled epochs) ---"
    grep -E "epoch=" "$LOG/train_${dl}_p0.log" \
        | sed -E "s/.*(epoch=[0-9]+).*(train_acc=[0-9.]+) (test_acc=[0-9.]+).*/  \1 \2 \3/" \
        | awk 'NR==1 || NR%10==0 || NR==total' total="$EPOCHS" | tail -12
    PORT=$((PORT + 2))
done
echo "[train-all] done."
