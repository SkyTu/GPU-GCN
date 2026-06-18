#!/usr/bin/env bash
# =============================================================================
# run_oblivious_select.sh  —  Oblivious single-node SELECT (routing + gather + X removal)
#
#   DPF equality (1-bit isX) -> XOR shard one-hot -> gpuSelect over the concatenated
#   [adj|feat|y_onehot|train_mask|test_mask] -> keep (B2A + gpuMul) zeros X's adjacency
#   column and drops X from the train mask.  NOTHING about X or its shard is revealed.
#
#   Two interchangeable gather variants (same I/O, both bit-exact):
#     gpuselect  : 1-bit boolean DPF + gpuSelect (the proper FSS primitive)   [default]
#     beaver     : DPF-LUT (arith) + open-once Beaver matmul                  [ablation]
#
#   Each run secret-shares X, runs both 2-party processes on loopback (single GPU),
#   reveals the result for the numpy oracle (obliv_verify.py) and prints the comm.
#
# Usage:  ./run_oblivious_select.sh [gpuselect|beaver|both]   (default: both)
# =============================================================================
set -eo pipefail

# --- portable paths: derive the repo root from this script's own location ---
UNL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../experiments/GCN/unlearning
GCN="$(cd "$UNL/../../.." && pwd)"                     # GPU-GCN repo root
DATA="${FSS_DATA_ROOT:-$GCN/experiments/GCN/datasets/cora_shards_canon}"
# build/runtime knobs (override via env if your CUDA/GPU/conda differ)
CUDA_VERSION_BUILD="${CUDA_VERSION:-11.8}"
GPU_ARCH="${GPU_ARCH:-89}"
PORT="${PORT:-42300}"
CONDA_ENV="${CONDA_ENV:-opengu}"
VARIANT=${1:-both}

# activate conda: prefer an explicit CONDA_SH, else discover via `conda info --base`
if [[ -z "${CONDA_SH:-}" ]]; then
    _cb="$(conda info --base 2>/dev/null || true)"
    [[ -n "$_cb" ]] && CONDA_SH="$_cb/etc/profile.d/conda.sh"
fi
if [[ -n "${CONDA_SH:-}" && -f "$CONDA_SH" ]]; then
    source "$CONDA_SH"; conda activate "$CONDA_ENV"
fi
export FSS_DATA_ROOT="$DATA"

# canonical shard-set (sorted(comm[s] U test), node-id tracked) must exist
if [[ ! -f "$DATA/meta.txt" ]]; then
    echo "[select] exporting canonical shard-set ..."
    python "$UNL/canonical_export.py" "$DATA"
fi

build() { (cd "$GCN" && CUDA_VERSION=$CUDA_VERSION_BUILD GPU_ARCH=$GPU_ARCH make "$1") >/tmp/build_$1.log 2>&1 \
            || { tail -20 /tmp/build_$1.log; echo "[select] build $1 FAILED"; exit 1; }; }

run_one() {   # $1 = binary, $2 = label, $3 = base port
    local bin="$UNL/$1"
    [[ -x "$bin" ]] || { echo "[select] building $1 ..."; build "$1"; }
    pkill -9 -f "/$1 " 2>/dev/null || true; sleep 1   # match the binary path, NOT this script
    rm -f "$DATA"/select/*_clear.bin
    python "$UNL/obliv_verify.py" prep >/dev/null          # secret-share X=270
    echo "================ $2 ================"
    CUDA_VISIBLE_DEVICES=0 "$bin" 1 127.0.0.1 --port "$3" --reveal-asel >/tmp/osel_${2}_p1.log 2>&1 &
    sleep 2
    CUDA_VISIBLE_DEVICES=0 "$bin" 0 127.0.0.1 --port "$3" --reveal-asel 2>&1 | grep -E "RESULT|wrote"
    wait
    python "$UNL/obliv_verify.py" verify
    echo
}

case "$VARIANT" in
    gpuselect) run_one oblivious_select        "gpuSelect" "$PORT" ;;
    beaver)    run_one oblivious_select_beaver "beaver"    "$PORT" ;;
    both)      run_one oblivious_select        "gpuSelect" "$PORT"
               run_one oblivious_select_beaver "beaver"    "$((PORT + 2))"
               echo "================ ablation (gpuSelect vs Beaver-matmul) ================"
               echo "Both variants are bit-exact (same X-removed subgraph). They open the same"
               echo "K x (N2+NF+NC) data => equal BYTES; gpuSelect uses fewer reconstruct ROUNDS"
               echo "(gpuDpf is local + one concatenated select) => lower latency. See [RESULT ...]." ;;
    *) echo "usage: $0 [gpuselect|beaver|both]"; exit 1 ;;
esac
echo "[select] done."
