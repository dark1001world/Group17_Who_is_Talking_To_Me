# Ego4D Audio-Visual Speaker Attribution

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Place .wav files in `data/audio/`
3. Place frame folders in `data/frames/` (each folder named as clip ID)
4. Place annotation JSON files in `data/annotations/`

## Feature Extraction
`python scripts/1_extract_features.py`

## Training
`python scripts/2_train.py`

## Evaluation
`python scripts/3_evaluate.py`
