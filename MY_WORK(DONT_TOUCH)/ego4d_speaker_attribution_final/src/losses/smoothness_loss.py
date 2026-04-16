import torch

def temporal_smoothness_loss(probs, track_mask):
    diff = torch.abs(probs[:, 1:] - probs[:, :-1])
    mask = track_mask[:, 1:] & track_mask[:, :-1]
    if mask.sum() > 0:
        return diff[mask].mean()
    return torch.tensor(0.0, device=probs.device)
