# Benchmark-Driven Fusion Structure

This project should treat the starter-generated artifacts as the benchmark data
contract and build all new modeling work inside this folder only.

## Data Contract

Raw benchmark source:
- `/DATA/G17/ego4d_data/v2/annotations/av_train.json`
- `/DATA/G17/ego4d_data/v2/annotations/av_val.json`
- `/DATA/G17/ego4d_data/v2/clips/*.mp4`

Prepared cache used for training:
- `/DATA/G17/Data/video/<clip_uid>/img_00001.jpg`
- `/DATA/G17/Data/wave/<clip_uid>.wav`
- `/DATA/G17/Group17_Who_is_Talking_To_Me/starter_code/data/json_original/<clip_uid>/*.json`
- `/DATA/G17/Group17_Who_is_Talking_To_Me/starter_code/data/result_TTM/<clip_uid>.json`
- `/DATA/G17/Group17_Who_is_Talking_To_Me/starter_code/data/split/{train,val}.list`

## Decision Target

The benchmark decision is:
- given `(clip_uid, person_id, start_frame, end_frame)`
- predict whether that person is `talking to me`

Training stays binary per person-segment.
Final "who is talking to me" selection comes from scoring all candidate persons in
the same temporal region and choosing the highest positive score.

## Exact Project Layout

```text
ego4d_speaker_attribution_final/
├── BENCHMARK_STRUCTURE.md
├── README.md
├── configs/
│   └── default.yaml
├── scripts/
│   ├── 0_validate_benchmark_dataset.py
│   ├── 1_extract_features.py              # legacy dense extractor experiment
│   ├── 2_train.py                         # legacy dense-feature trainer
│   └── 3_evaluate.py                      # legacy dense-feature evaluation
├── src/
│   ├── audio/
│   │   └── embedding_extractor.py
│   ├── dataset/
│   │   ├── ego4d_track_dataset.py         # legacy dense-feature dataset
│   │   └── ttm_benchmark_dataset.py       # benchmark-first segment dataset
│   ├── fusion/
│   │   ├── speaker_model.py
│   │   └── temporal_encoder.py
│   ├── losses/
│   │   ├── focal_loss.py
│   │   ├── lip_audio_sync.py
│   │   └── smoothness_loss.py
│   ├── tracking/
│   │   └── face_tracker.py
│   ├── utils/
│   │   ├── alignment.py
│   │   ├── config.py
│   │   └── logger.py
│   └── visual/
│       ├── face_extractor.py
│       ├── feature_extractor.py
│       ├── lip_encoder.py
│       └── reid_model.py
└── models/
```

## Recommended Build Order

1. `src/dataset/ttm_benchmark_dataset.py`
- Read benchmark splits, segment labels, tracklets, frame cache, and wave cache.

2. Audio-only baseline
- Reuse wav2vec2 or HuBERT embeddings over each labeled segment.

3. Visual-only baseline
- Crop the labeled person's face track and encode temporal face windows.

4. Fused benchmark model
- Audio encoder + visual encoder + cross-attention + binary classifier.

5. Final person decision
- For each temporal region, run all candidate persons through the classifier.
- Return the person with the highest `talking-to-me` probability.

## Known Legacy vs Benchmark Difference

`ttm_train.json` is a dense frame-level supervision source.
The official starter benchmark pipeline is segment-based and uses:
- `json_original` for person tracklets
- `result_TTM` for labeled speaking segments

For the benchmark pipeline, prefer the segment-based contract first.
