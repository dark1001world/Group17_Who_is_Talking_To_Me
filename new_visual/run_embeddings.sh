#!/bin/bash

# Activate conda and set up environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Group17

LOG_DIR="/DATA/G17/outputs"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "Starting embedding generation"
echo "Train split: $(date)"
echo "========================================"

# Run train split
cd /DATA/G17/Group17_Who_is_Talking_To_Me/new_visual
python embedding_generation.py --split train --overwrite >> "$LOG_DIR/embedding_train.log" 2>&1

echo "========================================"
echo "Train split completed: $(date)"
echo "Starting validation split"
echo "========================================"

# Run val split
python embedding_generation.py --split val --overwrite >> "$LOG_DIR/embedding_val.log" 2>&1

echo "========================================"
echo "All done: $(date)"
echo "========================================"
echo "Train embeddings: /DATA/G17/outputs/visual_segment_embeddings/train/visual_embedding.json"
echo "Val embeddings:   /DATA/G17/outputs/visual_segment_embeddings/val/visual_embedding.json"
