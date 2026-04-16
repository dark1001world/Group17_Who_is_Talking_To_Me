import torch
from utils.metrics import compute_all_metrics


def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for visual, audio, labels, mask_v, mask_a in loader:
            visual = visual.to(device)
            audio = audio.to(device)
            labels = labels.to(device)
            mask_v = mask_v.to(device)
            mask_a = mask_a.to(device)

            logits = model(visual, audio, mask_v, mask_a).squeeze(-1)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.detach().cpu())
            all_labels.append(labels.detach().cpu())

    probs = torch.cat(all_probs, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return compute_all_metrics(probs, labels)