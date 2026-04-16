import torch
import torchaudio
import numpy as np

class AudioEmbeddingExtractor:
    def __init__(self, model_name="wav2vec2", device='cuda'):
        self.device = device
        if model_name == "wav2vec2":
            bundle = torchaudio.pipelines.WAV2VEC2_BASE
        elif model_name == "hubert":
            bundle = torchaudio.pipelines.HUBERT_BASE
        else:
            raise ValueError(f"Unknown model: {model_name}")
        self.model = bundle.get_model().to(device)
        self.sample_rate = bundle.sample_rate
        self.model.eval()

        # For Wav2Vec2 base, the total stride of the convolutional frontend is 320 samples.
        # This is the number of audio samples per feature frame.
        self.hop_length = 320

    def extract_features_full(self, audio_path):
        waveform, orig_sr = torchaudio.load(audio_path)
        if orig_sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, orig_sr, self.sample_rate)
        waveform = waveform.to(self.device)
        with torch.no_grad():
            # extract_features returns (list of layer outputs, lengths)
            features_list, lengths = self.model.extract_features(waveform)
            # Take the output of the last transformer layer
            features = features_list[-1]  # shape: [1, time_steps, feature_dim]
        return features.squeeze(0), waveform.shape[1]

    def get_embedding_at_time(self, features, total_samples, time_center, window_duration):
        center_sample = int(time_center * self.sample_rate)
        center_idx = center_sample // self.hop_length
        window_samples = int(window_duration * self.sample_rate)
        window_idxs = window_samples // self.hop_length
        start_idx = max(0, center_idx - window_idxs // 2)
        end_idx = min(features.shape[0], start_idx + window_idxs)
        if end_idx - start_idx < window_idxs:
            chunk = features[start_idx:end_idx]
            if chunk.shape[0] == 0:
                chunk = torch.zeros((window_idxs, features.shape[1]), device=features.device)
            elif chunk.shape[0] < window_idxs:
                chunk = torch.nn.functional.pad(chunk, (0,0,0, window_idxs - chunk.shape[0]))
        else:
            chunk = features[start_idx:end_idx]
        embedding = chunk.mean(dim=0)
        return embedding.cpu().numpy()
