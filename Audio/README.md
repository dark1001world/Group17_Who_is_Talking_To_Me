# Audio Encoder for Ego4D TTM Task

**HuBERT-based audio encoder for frame-level speech detection in egocentric video.**

## 📁 Structure

```
├── audio_encoder.py       # HuBERT encoder module
├── preprocessing.py       # Audio loading, resampling, alignment
├── dataset.py            # PyTorch Dataset classes
├── config.py             # Configuration
├── train_audio_encoder.py # Training loop
├── utils.py              # Utilities & helpers
└── requirements.txt      # Dependencies
```

## 🚀 Quick Start

### Install
```bash
pip install -r requirements.txt
```

### Extract Embeddings
```python
from audio_encoder import AudioEncoder
from preprocessing import AudioProcessor
from config import AudioConfig

config = AudioConfig()
encoder = AudioEncoder(config).eval()
processor = AudioProcessor(config)

# Load and preprocess
audio, mask = processor.preprocess_audio("audio.wav")

# Extract: (1, num_frames, 768)
embeddings, _ = encoder(audio.unsqueeze(0), mask.unsqueeze(0))
```

### Train
```bash
python train_audio_encoder.py \
    --annotation-file annotations.json \
    --audio-root ./audio \
    --num-epochs 10
```

## 📊 Modules

| Module | Purpose |
|--------|---------|
| `AudioEncoder` | Loads HuBERT, extracts frame embeddings |
| `AudioProcessor` | Load → resample (16kHz) → pad/truncate with mask |
| `Ego4D_TTM_Dataset` | PyTorch Dataset for video segments + labels |
| `TTM_Classifier` | Frame-level classification head |
| `Trainer` | Training loop with validation, checkpointing |

## 🔀 Audio-Visual Fusion

```python
from audio_encoder import AudioEncoder
from utils import EmbeddingAlignment

# Audio embeddings: (batch, 400, 768)
audio_emb, _ = encoder(audio, mask)

# Video embeddings: (batch, 300, 512)
video_emb = vision_model(frames)

# Align & concatenate
audio_aligned, video_aligned = EmbeddingAlignment.match_sequence_lengths(
    audio_emb, video_emb, align_to="video"
)

multimodal = torch.cat([audio_aligned, video_aligned], dim=-1)
predictions = cross_modal_transformer(multimodal)
```

## ⚙️ Configuration

```python
from config import AudioConfig, TrainingConfig

audio_cfg = AudioConfig(
    sample_rate=16000,
    model_name="facebook/hubert-base-ls960",
    embedding_dim=768,
    freeze_encoder=False
)

train_cfg = TrainingConfig(
    batch_size=32,
    learning_rate=1e-4,
    num_epochs=10
)
```

## 📋 Dataset Format

```json
{
    "video_001": {
        "audio_path": "audio_001.wav",
        "duration_frames": 300,
        "segments": [
            {"start_frame": 10, "end_frame": 50, "label": 1},
            {"start_frame": 100, "end_frame": 150, "label": 0}
        ]
    }
}
```

## 🧪 Test

```bash
python audio_encoder.py        # Test encoder
python preprocessing.py        # Test audio loading
python dataset.py             # Test DataLoader
python utils.py               # Test utilities
```

## 💾 Save/Load

```python
# Save
torch.save(encoder.state_dict(), "encoder.pt")

# Load
encoder.load_state_dict(torch.load("encoder.pt"))
```

## 🔧 Utilities

**Padding:**
```python
from utils import PaddingUtils
padded, mask = PaddingUtils.pad_batch_audio(audio_list)
```

**Alignment:**
```python
from utils import EmbeddingAlignment
audio_aligned, video_aligned = EmbeddingAlignment.match_sequence_lengths(...)
```

**Loss:**
```python
from utils import LossFunctions
loss = LossFunctions.frame_level_bce_loss(predictions, targets, mask)
```

**Metrics:**
```python
from utils import MetricsUtils
metrics = MetricsUtils.compute_metrics(predictions, targets)
```

## 📈 Performance

- Model: HuBERT-base (~126M params)
- Output: `(batch, num_frames, 768)` where frames ≈ samples/4
- Memory: ~500MB model + 3GB GPU for batch_size=32
- Speed: ~50-100ms per minute (GPU)
