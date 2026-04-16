import torch
import torch.nn.functional as F

def lip_audio_contrastive_loss(lip_emb, audio_emb, temperature=0.07):
    B, T, D = lip_emb.shape
    if T < 2:
        return torch.tensor(0.0, device=lip_emb.device)
    lip_emb = F.normalize(lip_emb, p=2, dim=-1)
    audio_emb = F.normalize(audio_emb, p=2, dim=-1)
    pos_sim = (lip_emb * audio_emb).sum(dim=-1).view(B*T, 1)
    lip_flat = lip_emb.view(B*T, D)
    audio_flat = audio_emb.view(B*T, D)
    neg_sim = torch.matmul(lip_flat, audio_flat.t())
    mask = torch.eye(B*T, device=lip_emb.device).bool()
    neg_sim = neg_sim.masked_fill(mask, -1e9)
    logits = torch.cat([pos_sim, neg_sim], dim=1) / temperature
    labels = torch.zeros(B*T, dtype=torch.long, device=lip_emb.device)
    return F.cross_entropy(logits, labels)
