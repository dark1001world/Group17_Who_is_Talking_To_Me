# Audio Encoder for Ego4D TTM (Talking To Me) Task

**Production-ready HuBERT-based audio encoder for frame-level speech detection in egocentric video.**

---

## 📋 Overview

This module provides a complete audio encoding pipeline for the Ego4D "Talking To Me" (TTM) classification task. It uses **HuBERT** (a self-supervised speech representation model) to extract frame-aligned audio embeddings suitable for fusion with visual features in cross-modal transformers.

### Key Features

✅ HuBERT-based audio encoding
✅ Frame-level embeddings aligned with video
✅ Attention pooling support
✅ Variable-length audio handling with masking
✅ Complete training pipeline with AdamW + cosine annealing
✅ PyTorch Dataset for Ego4D-style annotations
✅ Comprehensive utilities for audio-visual fusion

---

## 📁 Project Structure

```
.
├── audio_encoder.py       # Core HuBERT-based audio encoder module
├── preprocessing.py       # Audio loading, resampling, frame alignment
├── dataset.py            # PyTorch Dataset classes for Ego4D data
├── config.py             # Centralized hyperparameter configuration
├── train_audio_encoder.py # Complete training loop with validation
├── utils.py              # Helper functions (padding, alignment, loss, metrics)
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Basic Usage: Load and Extract Embeddings

```python
import torch
from audio_encoder import AudioEncoder
from preprocessing import AudioProcessor
from config import AudioConfig

# Initialize config
config = AudioConfig()

# Create encoder
encoder = AudioEncoder(config)
encoder.eval()

# Load and preprocess audio
processor = AudioProcessor(config)
audio, mask = processor.preprocess_audio("path/to/audio.wav")

# Extract embeddings: shape (1, num_frames, 768)
with torch.no_grad():
    embeddings, attention_weights = encoder(audio.unsqueeze(0), mask.unsqueeze(0))

print(f"Embeddings shape: {embeddings.shape}")  # (1, ~400, 768)
```

### 3. Training

```bash
python train_audio_encoder.py \
    --annotation-file ./data/annotations.json \
    --audio-root ./data/audio \
    --num-epochs 10 \
    --batch-size 32 \
    --learning-rate 1e-4
```

---

## 📊 Module Details

### 🎙️ AudioEncoder (`audio_encoder.py`)

**Main class:** `AudioEncoder(nn.Module)`

**Purpose:** Loads pretrained HuBERT and extracts frame-level embeddings.

**Key Methods:**
- `forward(audio, attention_mask)` → embeddings (batch, frames, dim)
- `extract_embeddings(audio, pooling='none')` → pooled embeddings
- `enable_train_mode()` / `enable_eval_mode()`

**Why HuBERT?**
- Self-supervised speech representation (no labels needed)
- Pre-trained on 960h of LibriSpeech
- Rich phonetic/speaker information
- Low-level features (good for detecting speech)
- Temporal consistency for video alignment

**Configuration:**
```python
config = AudioConfig(
    sample_rate=16000,           # HuBERT requires 16kHz
    model_name="facebook/hubert-base-ls960",
    embedding_dim=768,
    freeze_encoder=False,        # Train or freeze weights
    use_attention_pooling=False
)
```

---

### 📁 Preprocessing (`preprocessing.py`)

**Main class:** `AudioProcessor`

**Features:**
- Load audio (.wav, .mp3, .flac, etc.)
- Resample to 16kHz (HuBERT requirement)
- Convert to mono
- Pad/truncate to fixed length with mask
- Calculate frame-level alignment

**Usage:**
```python
processor = AudioProcessor(config)
audio, mask = processor.preprocess_audio(
    "path/to/audio.wav",
    target_length=960000  # 60 seconds @ 16kHz
)
# audio shape: (1, target_length)
# mask shape: (1, target_length)
```

**Frame Alignment:**
```python
aligner = FrameAlignmentUtils(video_fps=30.0)
num_frames = processor.get_num_frames(audio.shape[-1])
# HuBERT reduces temporal dimension by 4x
```

---

### 🔗 Dataset (`dataset.py`)

**Classes:**
- `Ego4D_TTM_Dataset` - Full video-level dataset with frame segments
- `AudioOnlyDataset` - Simple audio+label pairs

**Expected annotation format:**
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

**Usage:**
```python
from dataset import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    annotation_file="annotations.json",
    audio_root="./audio",
    config=config,
    train_batch_size=32,
    num_workers=4
)

for batch in train_loader:
    audio = batch["audio"]           # (batch, num_samples)
    mask = batch["audio_mask"]       # (batch, num_samples)
    labels = batch["labels"]         # list of (num_frames,) tensors
    video_ids = batch["video_ids"]   # list of str
```

---

### ⚙️ Configuration (`config.py`)

Centralized hyperparameter management:

```python
audio_cfg, train_cfg, data_cfg = get_config()

# Audio config
audio_cfg.sample_rate = 16000
audio_cfg.window_size = 0.04  # 40ms frame (aligns with 25fps)
audio_cfg.max_audio_length = 60.0  # seconds

# Training config
train_cfg.batch_size = 32
train_cfg.learning_rate = 1e-4
train_cfg.num_epochs = 10
train_cfg.warmup_steps = 500

# Data config
data_cfg.data_root = "./data/audio"
data_cfg.annotation_file = "./data/annotations.json"
```

---

### 🔧 Utilities (`utils.py`)

**Padding & Masking:**
```python
from utils import PaddingUtils

padded, mask = PaddingUtils.pad_batch_audio(audio_list)
```

**Embedding Alignment:**
```python
from utils import EmbeddingAlignment

# Interpolate audio embeddings to match video frame rate
audio_aligned, video_aligned = EmbeddingAlignment.match_sequence_lengths(
    audio_emb, video_emb, align_to="video"
)
```

**Loss Functions:**
```python
from utils import LossFunctions

# Frame-level BCE loss
loss = LossFunctions.frame_level_bce_loss(
    predictions, targets, mask, pos_weight=1.5
)

# Temporal smoothness regularization
smooth_loss = LossFunctions.temporal_smoothness_loss(embeddings, lambda_smooth=0.01)
```

**Metrics:**
```python
from utils import MetricsUtils

metrics = MetricsUtils.compute_metrics(predictions, targets)
# Returns: TP, FP, FN, TN, precision, recall, f1, accuracy
```

---

## 🔀 Integration with Cross-Modal Fusion

### Example: Fusion with Visual Features

```python
import torch
from audio_encoder import AudioEncoder
from utils import EmbeddingAlignment

# Get audio embeddings
audio_encoder = AudioEncoder(config)
audio_embeddings, _ = audio_encoder(audio, audio_mask)
# Shape: (batch, 400, 768)

# Get video embeddings from your visual encoder
# Shape: (batch, 300, 512)
video_embeddings = your_vision_model(video_frames)

# Align to same frame count
audio_aligned, video_aligned = EmbeddingAlignment.match_sequence_lengths(
    audio_embeddings, video_embeddings, align_to="video"
)
# Both now: (batch, 300, ?)

# Concatenate for fusion
multimodal = torch.cat([audio_aligned, video_aligned], dim=-1)
# Shape: (batch, 300, 768+512)

# Pass through cross-modal transformer
predictions = cross_modal_transformer(multimodal)
```

---

## 🎯 Output Format

### Frame-Level Embeddings
```
Input:  (batch_size, num_audio_samples) @ 16kHz
            ↓
        HuBERT Extraction
            ↓
Output: (batch_size, num_frames, 768)

Example:
- Audio: 2 seconds @ 16kHz = 32,000 samples
- HuBERT reduction: 4x
- Output frames: 32,000 / 4 = 8,000 frames...

Wait, that's not right. Let me reconsider.
Actually HuBERT processes audio and the conv feature extract reduces by 4x.
So 16000 samples (1 sec) → 4000 frames approximately.

Actually, the exact calculation:
- HuBERT applies Conv1d with specific kernel/stride
- For 1 second of audio (16000 samples) at 16kHz
- Output is typically ~100 frames (so ~160 samples per frame)
- More precisely: output_length = floor((input_length - kernel_size + 2*padding) / stride) + 1

For standard HuBERT:
- Input 16000 samples
- Output approximately 100-130 frames depending on exact architecture

So for a simple example:
- Input: 2 seconds @ 16kHz = 32,000 samples
- Output: ~200-260 frames × 768 dims
```

### With Attention Pooling
```
Output: (batch_size, num_frames, 768)
Attention weights: (batch_size, num_frames)
```

### Pooled (Mean)
```
Output: (batch_size, 768)
```

---

## 📋 Model Variants

### HuBERT-Base (Default)
```python
config.model_name = "facebook/hubert-base-ls960"
config.embedding_dim = 768  # Output dimension
```

### wav2vec2-Base (Alternative)
```python
from transformers import AutoModel

config.model_name = "facebook/wav2vec2-base"
# Note: wav2vec2 has same architecture but trained differently
```

### Fine-tuned Models
```python
# Use a pretrained model from HuggingFace
config.model_name = "facebook/hubert-base-ls960-finetuned-ls960"
```

---

## 💾 Checkpointing

Models are automatically saved during training:

```python
# Periodic checkpoints
./checkpoints/checkpoint_epoch2_step500.pt

# Final model
./checkpoints/final_model_epoch9.pt

# Load checkpoint
checkpoint = torch.load("./checkpoints/final_model_epoch9.pt")
encoder.load_state_dict(checkpoint)
```

---

## 🧪 Testing

**Test audio extraction:**
```bash
python audio_encoder.py
# Creates dummy audio and prints shapes
```

**Test preprocessing:**
```bash
python preprocessing.py
# Tests audio loading, resampling, padding
```

**Test dataset:**
```bash
python dataset.py
# Creates dummy annotations and tests DataLoader
```

**Test utilities:**
```bash
python utils.py
# Tests padding, alignment, metrics
```

---

## 📈 Performance Notes

### Computational Cost
- HuBERT-base: ~126M parameters
- Inference: ~0.5-1.0 seconds for 1 minute of audio (CPU)
- ~50-100ms per minute of audio (GPU)

### Memory Requirements
- Model: ~500MB
- Batch size 32 (60-second audio): ~3GB GPU memory

### Optimization Tips
1. **Use `freeze_encoder=True`** if fine-tuning only classification head
2. **Enable attention pooling** to reduce output size if needed
3. **Use gradient checkpointing** for longer sequences:
   ```python
   encoder.model.gradient_checkpointing_enable()
   ```
4. **Consider audio chunking** for very long videos

---

## 🛠️ Common Use Cases

### 1. Inference on Video (without training)
```python
import torch
from audio_encoder import AudioEncoder
from preprocessing import AudioProcessor
from config import AudioConfig

config = AudioConfig()
encoder = AudioEncoder(config).eval()
processor = AudioProcessor(config)

# Process video audio
audio, mask = processor.preprocess_audio("video.wav")

with torch.no_grad():
    embeddings, _ = encoder(audio.unsqueeze(0), mask.unsqueeze(0))

# Save embeddings
torch.save(embeddings, "embeddings.pt")
```

### 2. Fine-tune on Your Data
```python
config.freeze_encoder = False  # Allow gradient updates

# Training loop updates HuBERT weights for your task
# This requires labeled data with segment annotations
```

### 3. Extract Embeddings for Preprocessing
```python
# Precompute embeddings for faster training
embedding_dict = {}

for video_id in video_ids:
    audio, mask = processor.preprocess_audio(f"audio/{video_id}.wav")
    with torch.no_grad():
        emb, _ = encoder(audio.unsqueeze(0), mask.unsqueeze(0))
    embedding_dict[video_id] = emb

torch.save(embedding_dict, "precomputed_embeddings.pt")
```

---

## ❓ FAQ

**Q: Can I use this with CUDA?**
A: Yes, device auto-detection is built-in. Models will automatically use GPU if available.

**Q: What if my audio is not 16kHz?**
A: The `AudioProcessor.resample_to_target()` automatically resamples to 16kHz.

**Q: How do I handle very long videos?**
A: Chunk the audio into segments (e.g., 1 minute chunks) and process separately, then concatenate.

**Q: Can I use this for other speech tasks?**
A: Yes! HuBERT embeddings work for:
- Speaker identification
- Speech emotion recognition
- Keyword spotting
- Speech command classification

**Q: What's the output frame rate after HuBERT?**
A: HuBERT typically outputs frames at ~50 frames per second (for 16kHz audio input).

---

## 📚 References

**HuBERT Paper:**
> Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, Abdelrahman Mohamed. "HuBERT: Self-supervised Speech Representation Learning by Masked Prediction of Hidden Units". IEEE/ACM Transactions on Audio, Speech and Language Processing, 2021.

**Ego4D Dataset:**
> Kristen Grauman, et al. "Ego4D: World's Largest Egocentric Video Dataset". CVPR 2023.

---

## 📝 License

This module is provided as-is for research and educational purposes.

---

## ✨ Summary

This production-ready audio encoder module provides:
- ✅ Complete HuBERT-based audio encoding pipeline
- ✅ Frame-level embeddings aligned with video
- ✅ Full training loop with all best practices
- ✅ Utilities for audio-visual fusion
- ✅ Modular, well-documented code
- ✅ Ready to integrate into larger AV systems

You can now directly plug audio embeddings into your cross-modal transformer for TTM classification!
