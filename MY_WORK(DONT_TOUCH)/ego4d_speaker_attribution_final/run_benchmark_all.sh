#!/bin/bash
set -euo pipefail

PROJECT_DIR="/DATA/G17/Group17_Who_is_Talking_To_Me/MY_WORK(DONT_TOUCH)/ego4d_speaker_attribution_final"
LOG_DIR="$PROJECT_DIR/benchmark_logs"
ENV_NAME="Group17"

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
else
  echo "ERROR: conda.sh not found"
  exit 1
fi

run_model() {
  local model_type="$1"
  local timestamp
  timestamp="$(date +%Y%m%d_%H%M%S)"
  local log_file="$LOG_DIR/${model_type}_${timestamp}.log"

  echo "[$(date)] Starting ${model_type}"
  python scripts/4_train_benchmark.py --model_type "$model_type" |& tee "$log_file"
  echo "[$(date)] Finished ${model_type}"
}

run_model "audio_only"
run_model "visual_only"
run_model "fusion_cross_attention"

echo "[$(date)] All benchmark runs finished."
