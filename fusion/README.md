# Fusion Module

This folder trains an audio-visual fusion model for Talking-To-Me prediction
using the merged file:

- /DATA/G17/outputs/final_embedding.json

## Key Technologies

- Visual branch: precomputed visual segment embeddings
- Audio branch: precomputed audio segment embeddings
- Fusion model: cross-modal attention (PyTorch MultiheadAttention)
- Optimization: AdamW + BCEWithLogitsLoss
- Evaluation: validation accuracy per epoch

## Input Format

`final_embedding.json` must contain:

```
{
	"segments": [
		{
			"uid": "...",
			"person_id": 2,
			"start_frame": 100,
			"end_frame": 140,
			"label": 0,
			"audio_embedding": [...],
			"visual_embedding": [...]
		}
	]
}
```

## Phase-wise Workflow

1. Read merged embeddings from `final_embedding.json`.
2. Split internally into train/validation using `val_ratio` in `utils/config.py`.
3. Build mini-batches with masks via `utils/collate.py`.
4. Forward pass through `models/fusion_model.py`.
5. Compute loss against `label` and backpropagate.
6. Validate each epoch and report accuracy.
7. Save trained checkpoint as `fusion_model.pt`.

## Main Files

- `data/dataset.py`: loads JSON, splits train/val, returns tensors
- `scripts/run_train.py`: training entry point
- `engine/train.py`: one-epoch backprop loop
- `engine/eval.py`: validation loop
- `models/fusion_model.py`: fusion architecture
- `utils/config.py`: paths and hyperparameters

## Run Training

From this folder:

```
python scripts/run_train.py
```

## Current Output

- Model checkpoint: `fusion_model.pt`