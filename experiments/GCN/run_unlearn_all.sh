#!/usr/bin/env bash
# =============================================================================
# run_unlearn_all.sh — oblivious single-node unlearning (FSS) over datasets.
#
# For each dataset:  ensure canonical node-id-tracked shards (canonical_export.py
#   from the OpenGU GraphEraser data) -> oblivious SELECT (DPF routing + gpuSelect
#   gather, query node & its shard stay hidden) -> e2e (assemble X-removed subgraph
#   + secret-mask ciphertext retrain).  Calls unlearning/run_unlearn_e2e.sh per ds.
#
# Usage:   ./run_unlearn_all.sh [DATASET ...]        (default: cora citeseer)
#          EPOCHS=80 LR=16 ./run_unlearn_all.sh cora
# Env:     OPENGU_ROOT  — OpenGU install base (…/GULib-master); needed to (re)generate
#          canonical shards if datasets/<ds>_shards_canon is missing (it is gitignored).
#          CONDA_ENV (opengu).
# Note:    PubMed (per-shard Ns~5800) oblivious select is impractical (O(K*Ns^2) select
#          keygen blows up); it is skipped.
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"      # experiments/GCN
UNL="$SCRIPT_DIR/unlearning"
EPOCHS="${EPOCHS:-80}"; LR="${LR:-16}"
CONDA_ENV="${CONDA_ENV:-opengu}"
DATASETS=("$@"); [ ${#DATASETS[@]} -eq 0 ] && DATASETS=(cora citeseer)

if [ -z "${CONDA_SH:-}" ]; then
    for _c in "$(conda info --base 2>/dev/null)" "${CONDA_PREFIX:-}" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniconda" /opt/conda; do
        [ -n "$_c" ] && [ -f "$_c/etc/profile.d/conda.sh" ] && { CONDA_SH="$_c/etc/profile.d/conda.sh"; break; }
    done
fi
[ -n "${CONDA_SH:-}" ] && [ -f "$CONDA_SH" ] && { source "$CONDA_SH"; conda activate "$CONDA_ENV"; }

for DS in "${DATASETS[@]}"; do
    dl="$(echo "$DS" | tr '[:upper:]' '[:lower:]')"
    if [ "$dl" = "pubmed" ]; then
        echo "[skip] PubMed oblivious select impractical (O(K*Ns^2), per-shard Ns~5800)."
        continue
    fi
    CANON="$SCRIPT_DIR/datasets/${dl}_shards_canon"
    echo "==================== $DS : canonical shards ===================="
    if [ ! -f "$CANON/meta.txt" ]; then
        if [ -z "${OPENGU_ROOT:-}" ]; then
            echo "[$DS] canonical shards missing and OPENGU_ROOT unset -> cannot regenerate; skip."
            echo "      (generate OpenGU GraphEraser data first, then set OPENGU_ROOT.)"
            continue
        fi
        python3 "$UNL/canonical_export.py" --dataset "$dl" \
            --opengu-processed "$OPENGU_ROOT/data/GraphEraser/processed/$dl" --out-dir "$CANON" \
            || { echo "[$DS] canonical_export failed"; continue; }
    fi
    echo "==================== $DS : oblivious unlearning (select -> assemble -> secret-mask retrain) ===================="
    FSS_DATA_ROOT="$CANON" bash "$UNL/run_unlearn_e2e.sh" "$EPOCHS" "$LR"
done
echo "[unlearn-all] done."
