#!/bin/bash
# run_pipeline.sh - Robust pipeline runner with automatic retry

cd "$(dirname "$0")"

# Activate the Group17 conda environment
ENV_NAME=Group17
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
else
  echo "WARNING: conda.sh not found; unable to activate $ENV_NAME"
fi
if ! conda env list | grep -q "^$ENV_NAME"; then
  echo "ERROR: conda environment $ENV_NAME not found"
  exit 1
fi

MAX_RETRIES=3
RETRY_DELAY=60  # seconds

run_extraction() {
    echo "[$(date)] Starting feature extraction..."
    conda run -n "$ENV_NAME" python scripts/1_extract_features.py
    return $?
}

run_training() {
    echo "[$(date)] Starting training..."
    conda run -n "$ENV_NAME" python scripts/2_train.py
    return $?
}

# Feature extraction with retry
for i in $(seq 1 $MAX_RETRIES); do
    if run_extraction; then
        echo "[$(date)] Feature extraction completed successfully."
        break
    else
        echo "[$(date)] Feature extraction failed (attempt $i/$MAX_RETRIES)."
        if [ $i -lt $MAX_RETRIES ]; then
            echo "Retrying in $RETRY_DELAY seconds..."
            sleep $RETRY_DELAY
        else
            echo "[$(date)] Feature extraction failed after $MAX_RETRIES attempts. Exiting."
            exit 1
        fi
    fi
done

# Training with retry
for i in $(seq 1 $MAX_RETRIES); do
    if run_training; then
        echo "[$(date)] Training completed successfully."
        break
    else
        echo "[$(date)] Training failed (attempt $i/$MAX_RETRIES)."
        if [ $i -lt $MAX_RETRIES ]; then
            echo "Retrying in $RETRY_DELAY seconds..."
            sleep $RETRY_DELAY
        else
            echo "[$(date)] Training failed after $MAX_RETRIES attempts. Exiting."
            exit 1
        fi
    fi
done

echo "[$(date)] Pipeline finished."
