from .audio_only_model import AudioOnlyTTMModel
from .fusion_cross_attention_model import FusedCrossAttentionTTMModel
from .visual_only_model import VisualOnlyTTMModel


def build_benchmark_model(model_type: str, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.1):
    model_type = model_type.lower()
    if model_type == "audio_only":
        return AudioOnlyTTMModel(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    if model_type == "visual_only":
        return VisualOnlyTTMModel(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    if model_type == "fusion_cross_attention":
        return FusedCrossAttentionTTMModel(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    raise ValueError(f"Unknown benchmark model_type: {model_type}")
