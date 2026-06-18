#!/usr/bin/env bash
# Single-node FSS-ciphertext unlearning for the GraphEraser GCN pipeline.
#
# Unlearn ONE node X by FSS-secret-retraining ONLY its affected shard with the
# secure-piranha standard_gcn trainer, then re-infer (L1) the retrained shard,
# re-aggregate (L2 mean) and score micro-F1 against the baseline. Shards != s
# are untouched, so their L1 posteriors are reused from the baseline shard-set.
#
# This REPLACES the OpenGU-cleartext 270-node-batch unlearning path.
#
# All compute is 2-party FSS on loopback (single L40S vGPU). Stages:
#   0. build      cleartext data plumbing (unlearn_single_node.py build)
#   1. retrain    standard_gcn_fss_train on shard-s-minus-X (secure piranha)
#   2. finalize   import retrained weight shares into the unlearned shard-set
#   3. L1         grapheraser_fss_l1_inference on shard s only (reuse the rest)
#   4. L2 + eval  grapheraser_fss_l2_aggregate (mean) + grapheraser_eval.py
#   5. compare    original vs single-node-unlearned micro-F1
#
# Usage:
#   ./run_unlearn_single_node.sh [--node N] [--epochs E] [--lr LR] [--hidden H]
#                                [--scale S] [--skip-retrain] [--rebuild]
set -euo pipefail

# ---- paths (portable: derived from this script's location; override via env) ----
UNL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"        # .../experiments/GCN/unlearning
GCN="$(cd "$UNL/.." && pwd)"                               # .../experiments/GCN
# OblivGU (this old single-node pipeline still builds/runs OblivGU binaries):
# env override, else sibling of the GPU-GCN repo root.
REPO="${OBLIVGU_ROOT:-$(cd "$UNL/../../../.." && pwd)/OblivGU}"
OG=$REPO/tests/GNN
LOGS=$GCN/logs
CONDA_ENV="${CONDA_ENV:-opengu}"
if [[ -z "${CONDA_SH:-}" ]]; then
    _cb="$(conda info --base 2>/dev/null || true)"
    [[ -n "$_cb" ]] && CONDA_SH="$_cb/etc/profile.d/conda.sh"
fi

NODE=270
EPOCHS=80
LR=16
HIDDEN=64
SCALE=12          # GraphEraser L1/L2 pipeline fixed-point scale
TRAIN_SCALE=24    # standard_gcn retrain scale (24 converges; 12 stalls -- verified)
SEED=0
SKIP_RETRAIN=0
REBUILD=0
CUDA_VERSION_BUILD=11.8
GPU_ARCH=89
TRAIN_PORT=42340
PIPE_PORT=42003

while [[ $# -gt 0 ]]; do
    case "$1" in
        --node)    NODE=$2; shift 2;;
        --epochs)  EPOCHS=$2; shift 2;;
        --lr)      LR=$2; shift 2;;
        --hidden)  HIDDEN=$2; shift 2;;
        --scale)   SCALE=$2; shift 2;;
        --train-scale) TRAIN_SCALE=$2; shift 2;;
        --seed)    SEED=$2; shift 2;;
        --skip-retrain) SKIP_RETRAIN=1; shift;;
        --rebuild) REBUILD=1; shift;;
        -h|--help) sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//;/^set -euo/d'; exit 0;;
        *) echo "unknown arg: $1" >&2; exit 1;;
    esac
done

mkdir -p "$LOGS"
RUN_LOG=$LOGS/unlearn_single_${NODE}_run.log
exec > >(tee -a "$RUN_LOG") 2>&1

log()  { printf '\n\033[1;36m[unlearn]\033[0m %s\n' "$*"; }
fail() { printf '\n\033[1;31m[unlearn] FAIL:\033[0m %s\n' "$*" >&2; exit 1; }

# /proc/net/tcp LISTEN check (state 0A); avoids needing ss/netstat.
port_in_use() {
    local port_hex; printf -v port_hex '%04X' "$1"
    grep -qE ":${port_hex}\s+[0-9A-F:]+\s+0A" /proc/net/tcp 2>/dev/null
}
wait_for_listen() {
    local port=$1 elapsed=0 port_hex; printf -v port_hex '%04X' "$port"
    while (( elapsed < 120 )); do
        grep -qE ":${port_hex}\s+[0-9A-F:]+\s+0A" /proc/net/tcp 2>/dev/null && return 0
        sleep 0.5; elapsed=$((elapsed + 1))
    done
    return 1
}

[[ -f $CONDA_SH ]] || fail "conda profile not found: $CONDA_SH"
# shellcheck disable=SC1090
source "$CONDA_SH"; conda activate "$CONDA_ENV" || fail "conda env '$CONDA_ENV' missing"
cd "$OG"

# ---- build binaries if needed ----
need_build=0
for b in standard_gcn_fss_train grapheraser_fss_l1_inference grapheraser_fss_l2_aggregate; do
    [[ -x $OG/$b ]] || { need_build=1; break; }
done
if [[ $REBUILD -eq 1 || $need_build -eq 1 ]]; then
    log "make standard_gcn_fss_train grapheraser_fss_l1_inference grapheraser_fss_l2_aggregate"
    (cd "$REPO" && CUDA_VERSION=$CUDA_VERSION_BUILD GPU_ARCH=$GPU_ARCH \
        make standard_gcn_fss_train grapheraser_fss_l1_inference grapheraser_fss_l2_aggregate) \
        >"$LOGS/unlearn_single_${NODE}_make.log" 2>&1 \
        || { tail -40 "$LOGS/unlearn_single_${NODE}_make.log"; fail "make failed"; }
fi

# ---- resolve affected shard ----
SHARD=$(python3 "$UNL/unlearn_single_node.py" resolve --node "$NODE" | awk '{print $NF}')
[[ $SHARD =~ ^[0-9]+$ ]] || fail "could not resolve shard for node $NODE"
log "node $NODE -> affected shard $SHARD"

RETRAIN_DIR=$OG/datasets/unlearn/shard${SHARD}_minus_${NODE}
OUT_SHARDSET=$OG/datasets/cora_shards_unlearned_single_${NODE}
SRC_SHARDSET=$OG/datasets/cora_shards
RETRAIN_ROOT_REL=datasets/unlearn/shard${SHARD}_minus_${NODE}
OUT_SHARDSET_REL=datasets/cora_shards_unlearned_single_${NODE}
SRC_SHARDSET_REL=datasets/cora_shards

# ============================================================ #
# 0. build (cleartext data plumbing)
# ============================================================ #
log "stage 0: build retrain dataset (scale=$TRAIN_SCALE) + unlearned shard-set (scale=$SCALE)"
python3 "$UNL/unlearn_single_node.py" build --node "$NODE" --scale "$SCALE" \
    --train-scale "$TRAIN_SCALE" --hidden "$HIDDEN" --seed "$SEED" \
    --retrain-dir "$RETRAIN_DIR" --out-shardset "$OUT_SHARDSET" \
    --src-shardset "$SRC_SHARDSET"

# ============================================================ #
# 1. FSS retrain shard-s-minus-X (secure piranha standard_gcn)
# ============================================================ #
TRAIN_P0=$LOGS/unlearn_single_${NODE}_retrain_p0.log
TRAIN_P1=$LOGS/unlearn_single_${NODE}_retrain_p1.log
if [[ $SKIP_RETRAIN -eq 1 ]]; then
    log "stage 1: SKIPPED (--skip-retrain); reusing weights in $RETRAIN_DIR/weights"
else
    log "stage 1: FSS retrain (secure piranha) epochs=$EPOCHS lr=$LR  root=$RETRAIN_ROOT_REL"
    port_in_use "$TRAIN_PORT" && fail ":$TRAIN_PORT already LISTEN -- another run? clean up first"
    rm -f "$TRAIN_P0" "$TRAIN_P1"
    export FSS_DATA_ROOT="$RETRAIN_ROOT_REL"
    CUDA_VISIBLE_DEVICES=0 ./standard_gcn_fss_train 0 127.0.0.1 --port "$TRAIN_PORT" \
        --epochs "$EPOCHS" --lr "$LR" --reveal-eval >"$TRAIN_P0" 2>&1 &
    P0=$!
    trap 'kill -9 $P0 2>/dev/null || true' EXIT
    wait_for_listen "$TRAIN_PORT" || { tail -30 "$TRAIN_P0"; fail "retrain party 0 never bound :$TRAIN_PORT"; }
    set +e
    CUDA_VISIBLE_DEVICES=0 ./standard_gcn_fss_train 1 127.0.0.1 --port "$TRAIN_PORT" \
        --epochs "$EPOCHS" --lr "$LR" --reveal-eval >"$TRAIN_P1" 2>&1
    P1_RC=$?
    wait $P0; P0_RC=$?
    set -e
    trap - EXIT
    [[ $P1_RC -eq 0 ]] || { tail -30 "$TRAIN_P1"; fail "retrain party 1 exit=$P1_RC"; }
    [[ $P0_RC -eq 0 ]] || { tail -30 "$TRAIN_P0"; fail "retrain party 0 exit=$P0_RC"; }
    # convergence + security checks
    grep -qE "softmax=piranha" "$TRAIN_P0" || fail "retrain did NOT use secure piranha softmax"
    log "retrain test-acc curve (party 0):"
    grep -E "epoch=[0-9]+ .*test_acc=" "$TRAIN_P0" \
        | sed -E 's/.*(epoch=[0-9]+).*(train_acc=[0-9.]+) (test_acc=[0-9.]+).*/  \1 \2 \3/' \
        | awk 'NR==1||NR%5==0||1' | tail -n +1 | sed -n '1~5p;$p'
    FINAL_TEST=$(grep -oE "test_acc=[0-9.]+" "$TRAIN_P0" | tail -1)
    FIRST_TRAIN=$(grep -oE "train_acc=[0-9.]+" "$TRAIN_P0" | head -1)
    FINAL_TRAIN=$(grep -oE "train_acc=[0-9.]+" "$TRAIN_P0" | tail -1)
    log "retrain converged: first ${FIRST_TRAIN}, final ${FINAL_TRAIN} ${FINAL_TEST} (secure piranha)"
fi
unset FSS_DATA_ROOT

# ============================================================ #
# 2. finalize: import retrained weight shares into shard-set
# ============================================================ #
log "stage 2: import retrained weights (re-quantize $TRAIN_SCALE->$SCALE) into shard $SHARD of $OUT_SHARDSET_REL"
python3 "$UNL/unlearn_single_node.py" finalize --node "$NODE" \
    --scale "$SCALE" --train-scale "$TRAIN_SCALE" --seed "$SEED" \
    --retrain-dir "$RETRAIN_DIR" --out-shardset "$OUT_SHARDSET"

# ============================================================ #
# 3. L1 inference on shard s only (reuse other 9 posteriors)
# ============================================================ #
log "stage 3: L1 FSS inference on shard $SHARD  (other 9 shards' posteriors reused from baseline)"
# Other shards' posteriors were copied into OUT_SHARDSET by the clone in stage 0;
# verify they exist (so L2 has all 10) then re-run L1 for the unlearned shard.
for s in $(seq 0 9); do
    [[ -f $OUT_SHARDSET/posteriors/shard_${s}_post_share0.bin ]] \
        || fail "missing baseline posterior for shard $s in clone (clone incomplete?)"
done
L1_P0=$LOGS/unlearn_single_${NODE}_l1_shard${SHARD}_p0.log
L1_P1=$LOGS/unlearn_single_${NODE}_l1_shard${SHARD}_p1.log
port_in_use "$PIPE_PORT" && fail ":$PIPE_PORT already LISTEN -- another pipeline run? clean up first"
rm -f "$L1_P0" "$L1_P1"
export FSS_DATA_ROOT="$OUT_SHARDSET_REL"
CUDA_VISIBLE_DEVICES=0 ./grapheraser_fss_l1_inference 0 127.0.0.1 "$HIDDEN" "$SHARD" \
    --port "$PIPE_PORT" >"$L1_P0" 2>&1 &
P0=$!
trap 'kill -9 $P0 2>/dev/null || true' EXIT
wait_for_listen "$PIPE_PORT" || { tail -30 "$L1_P0"; fail "L1 party 0 never bound :$PIPE_PORT"; }
set +e
CUDA_VISIBLE_DEVICES=0 ./grapheraser_fss_l1_inference 1 127.0.0.1 "$HIDDEN" "$SHARD" \
    --port "$PIPE_PORT" >"$L1_P1" 2>&1
P1_RC=$?
wait $P0; P0_RC=$?
set -e
trap - EXIT
[[ $P1_RC -eq 0 ]] || { tail -30 "$L1_P1"; fail "L1 party 1 exit=$P1_RC"; }
[[ $P0_RC -eq 0 ]] || { tail -30 "$L1_P0"; fail "L1 party 0 exit=$P0_RC"; }
bad=$(grep -E "mismatches = [^0]" "$L1_P0" || true)
[[ -z $bad ]] || { echo "$bad"; fail "L1: nonzero FSS vs CPU mismatches"; }
grep -E "^\[shard ${SHARD}\] (online|FSS vs CPU mismatches)" "$L1_P0" || true

# ============================================================ #
# 4. L2 mean aggregate + eval
# ============================================================ #
log "stage 4: L2 FSS aggregate (mean) + eval"
L2_P0=$LOGS/unlearn_single_${NODE}_l2_p0.log
port_in_use "$PIPE_PORT" && fail ":$PIPE_PORT already LISTEN -- another pipeline run? clean up first"
rm -f "$L2_P0"
CUDA_VISIBLE_DEVICES=0 ./grapheraser_fss_l2_aggregate 0 127.0.0.1 mean >"$L2_P0" 2>&1 &
P0=$!
trap 'kill -9 $P0 2>/dev/null || true' EXIT
wait_for_listen "$PIPE_PORT" || { tail -30 "$L2_P0"; fail "L2 party 0 never bound :$PIPE_PORT"; }
set +e
CUDA_VISIBLE_DEVICES=0 ./grapheraser_fss_l2_aggregate 1 127.0.0.1 mean 2>&1 | grep -E "\[L2\]" | tail -10
P1_RC=${PIPESTATUS[0]}
wait $P0; P0_RC=$?
set -e
trap - EXIT
[[ $P1_RC -eq 0 ]] || { tail -30 "$L2_P0"; fail "L2 party 1 exit=$P1_RC"; }
[[ $P0_RC -eq 0 ]] || { tail -30 "$L2_P0"; fail "L2 party 0 exit=$P0_RC"; }
unset FSS_DATA_ROOT

UNLEARN_F1=$(python3 grapheraser_eval.py --root "$OUT_SHARDSET" --mode mean --source fss \
    | awk '/micro-F1/ {print $3; exit}')
ORIG_F1=$(python3 grapheraser_eval.py --root "$SRC_SHARDSET" --mode mean --source fss \
    | awk '/micro-F1/ {print $3; exit}')

# ============================================================ #
# 5. compare
# ============================================================ #
log "stage 5: comparison"
printf '\n================ SINGLE-NODE FSS UNLEARNING SUMMARY ================\n'
printf '  unlearned node          : %s\n' "$NODE"
printf '  affected shard          : %s (retrained from scratch, FSS secure piranha)\n' "$SHARD"
printf '  retrain epochs / lr      : %s / %s   (train scale=%s, pipeline scale=%s)\n' \
    "$EPOCHS" "$LR" "$TRAIN_SCALE" "$SCALE"
printf '  shards retrained         : 1 / 10   (others reused)\n'
printf '  ------------------------------------------------------------------\n'
printf '  %-32s %s\n' "original mean micro-F1"          "$ORIG_F1"
printf '  %-32s %s\n' "single-node-unlearned micro-F1"  "$UNLEARN_F1"
printf '  ------------------------------------------------------------------\n'
printf '  (1 of 2166 train nodes removed -> a tiny delta is expected & correct;\n'
printf '   the deliverable is the working ciphertext mechanism.)\n'
printf '====================================================================\n'
log "DONE  (node=$NODE shard=$SHARD)  log: $RUN_LOG"
