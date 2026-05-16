import os
import json
import torch
import numpy as np
from tqdm import tqdm
import librosa
from transformers import WhisperFeatureExtractor
from model import WhisperTTM

# ──────────────────────────────────────────────────────────────
#  1. Setup & Config
# ──────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
JSON_PATH = "/DATA/G17/Data/extract_data/ttm_validation_data.json"
WAVE_DIR = "/DATA/G17/Data/wave/"
OUT_DIR = "/DATA/G17/Data/extracted_features/audio/val_full/"
CKPT_PATH = "experiments/whisper_audio_baseline/best_model.pth"
BATCH_SIZE = 64  # Adjust based on VRAM (64 is safe for T4/A5000)

os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────
#  2. Load Weights & Model
# ──────────────────────────────────────────────────────────────
model = WhisperTTM(model_size="base", temporal_depth=2)
print(f"Loading weights from {CKPT_PATH}...")
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(ckpt['state_dict'])
model.to(DEVICE).eval()

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")

@torch.no_grad()
def get_features(mel_batch):
    # mel_batch shape: [Batch, 80, 3000]
    x = model.encoder(mel_batch).last_hidden_state
    x = model.temporal_transformer(x)
    return x.mean(dim=1) 

# ──────────────────────────────────────────────────────────────
#  3. Group JSON by Clip
# ──────────────────────────────────────────────────────────────
with open(JSON_PATH, 'r') as f:
    master_data = json.load(f)

clips_map = {}
for r in master_data:
    uid = r['clip_uid']
    if uid not in clips_map: clips_map[uid] = []
    clips_map[uid].append(r)

print(f"Total Clips: {len(clips_map)} | Total Frames: {len(master_data)}")

# ──────────────────────────────────────────────────────────────
#  4. The Batched Extraction Loop
# ──────────────────────────────────────────────────────────────
print(f"\n[!] FORCING EXTRACTION TO: {OUT_DIR}")

for uid, records in tqdm(clips_map.items(), desc="Extracting Clips"):
    # Clean the UID just in case it has '.wav' or spaces in the JSON
    clean_uid = str(uid).replace(".wav", "").strip()
    wav_path = os.path.join(WAVE_DIR, f"{clean_uid}.wav")
    
    if not os.path.exists(wav_path):
        # Fallback: check if the UID already has the extension in the JSON
        wav_path = os.path.join(WAVE_DIR, str(uid).strip())
        if not os.path.exists(wav_path):
            continue
    
    try:
        audio, _ = librosa.load(wav_path, sr=16000)
    except:
        continue

    for i in range(0, len(records), BATCH_SIZE):
        batch_records = records[i : i + BATCH_SIZE]
        chunks, valid_records = [], []

        for r in batch_records:
            pid, fid = r['person_id'], int(r['frame'])
            
            # --- THE "NUKE": COMMENT OUT THE SKIP LOGIC ---
            # if os.path.exists(os.path.join(OUT_DIR, save_name)): continue

            center_sample = int((fid / 30.0) * 16000) 
            start_s = max(0, center_sample - 2400)
            chunk = audio[start_s : start_s + 4800]
            
            if len(chunk) < 4800:
                chunk = np.pad(chunk, (0, 4800 - len(chunk)))
            
            chunks.append(chunk)
            valid_records.append(r)

        if not chunks: continue

        # Forward Pass
        mel_batch = feature_extractor(chunks, sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE)
        features = get_features(mel_batch)

        # Save Results
        for j, feat in enumerate(features):
            rec = valid_records[j]
            # Use the clean_uid for saving to ensure matching with Eval script
            s_name = f"{clean_uid}_{rec['person_id']}_{int(rec['frame']):06d}.pt"
            save_path = os.path.join(OUT_DIR, s_name)
            
            torch.save(feat.cpu(), save_path)
            
        # PRINT EVERY 50 BATCHES TO PROVE IT'S ALIVE
        if i == 0:
            print(f" -> Saving to: {os.path.abspath(save_path)}")

print(f"\nEXTRACTION COMPLETE! Now run the evaluation.")