#!/usr/bin/env bash
set -euo pipefail

# Resilient launcher for Dino track phase-1 training.
# - Always resumes from latest epoch_*.pth if present
# - Appends logs to train_log.txt
# - Auto-restarts on non-zero exit with short backoff

ROOT_DIR="/DATA/G17/Group17_Who_is_Talking_To_Me/new_visual"
EXP_DIR="$ROOT_DIR/experiments/dino_track_phase1"
LOG_FILE="$EXP_DIR/train_log.txt"

MAX_RESTARTS="${MAX_RESTARTS:-20}"
RESTART_WAIT_SEC="${RESTART_WAIT_SEC:-15}"

mkdir -p "$EXP_DIR"

find_latest_ckpt() {
  local latest
  latest=$(ls -1 "$EXP_DIR"/epoch_*.pth 2>/dev/null | sort -V | tail -n 1 || true)
  if [[ -n "$latest" ]]; then
    echo "$latest"
  else
    echo ""
  fi
}

build_cmd() {
  local ckpt="$1"
  local cmd=(
    python run.py
    --source_path /DATA/G17/Data/video/
    --json_path /DATA/G17/Group17_Who_is_Talking_To_Me/starter_code/data/json_original
    --gt_path /DATA/G17/Group17_Who_is_Talking_To_Me/starter_code/data/result_TTM
    --train_file /DATA/G17/Group17_Who_is_Talking_To_Me/starter_code/data/split/train.list
    --val_file /DATA/G17/Group17_Who_is_Talking_To_Me/starter_code/data/split/val.list
    --model DinoViTTrackTTM
    --variant vit_base_patch16_224
    --clip_frames 8
    --batch_size 32
    --img_size 224
    --epochs 30
    --lr 1e-4
    --backbone_lr_scale 0.0
    --weight_decay 0.05
    --warmup_epochs 0
    --weights 0.598 3.052
    --grad_clip 1.0
    --num_workers 16
    --train_stride 8
    --val_stride 8
    --freeze_backbone
    --temporal_depth 2
    --use_track
    --amp
    --ema
    --exp_path "$EXP_DIR"
  )

  if [[ -n "$ckpt" ]]; then
    cmd+=(--checkpoint "$ckpt")
  fi

  printf '%q ' "${cmd[@]}"
}

cd "$ROOT_DIR"

restart_count=0
while true; do
  latest_ckpt="$(find_latest_ckpt)"
  if [[ -n "$latest_ckpt" ]]; then
    echo "[$(date '+%F %T')] Resuming from $latest_ckpt" | tee -a "$LOG_FILE"
  else
    echo "[$(date '+%F %T')] Starting fresh training" | tee -a "$LOG_FILE"
  fi

  run_cmd="$(build_cmd "$latest_ckpt")"
  echo "[$(date '+%F %T')] Launch: $run_cmd" | tee -a "$LOG_FILE"

  # With set -e + pipefail, a failing pipeline would exit before restart logic.
  # Temporarily disable errexit so we can capture the training exit code.
  set +e
  # shellcheck disable=SC2086
  eval "$run_cmd" 2>&1 | tee -a "$LOG_FILE"
  exit_code=${PIPESTATUS[0]}
  set -e

  if [[ "$exit_code" -eq 0 ]]; then
    echo "[$(date '+%F %T')] Training completed successfully." | tee -a "$LOG_FILE"
    exit 0
  fi

  restart_count=$((restart_count + 1))
  echo "[$(date '+%F %T')] Training exited with code $exit_code (restart $restart_count/$MAX_RESTARTS)." | tee -a "$LOG_FILE"

  if [[ "$restart_count" -ge "$MAX_RESTARTS" ]]; then
    echo "[$(date '+%F %T')] Reached MAX_RESTARTS=$MAX_RESTARTS. Stopping." | tee -a "$LOG_FILE"
    exit "$exit_code"
  fi

  echo "[$(date '+%F %T')] Waiting ${RESTART_WAIT_SEC}s before restart..." | tee -a "$LOG_FILE"
  sleep "$RESTART_WAIT_SEC"
done
