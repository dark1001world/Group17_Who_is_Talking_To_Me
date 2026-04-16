import torch
import time


def train_one_epoch(model, loader, optimizer, criterion, device, epoch_idx=None, log_interval=50):
    model.train()
    total_loss = 0
    total_correct = 0
    total_count = 0
    start_t = time.perf_counter()

    for step, (visual, audio, labels, mask_v, mask_a) in enumerate(loader, start=1):
        visual = visual.to(device)
        audio = audio.to(device)
        labels = labels.to(device)
        mask_v = mask_v.to(device)
        mask_a = mask_a.to(device)

        optimizer.zero_grad()

        logits = model(visual, audio, mask_v, mask_a).squeeze(-1)

        loss = criterion(logits, labels)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        total_correct += (preds == labels).sum().item()
        total_count += labels.numel()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % max(log_interval, 1) == 0 or step == len(loader):
            pct = 100.0 * step / max(len(loader), 1)
            elapsed = int(time.perf_counter() - start_t)
            lr = optimizer.param_groups[0]["lr"]
            avg_loss = total_loss / step
            if epoch_idx is None:
                prefix = "ep=?"
            else:
                prefix = f"ep={epoch_idx:02d}"
            print(
                f"{prefix} {pct:5.1f}% | {step}/{len(loader)} "
                f"loss={avg_loss:.4f} lr={lr:.2e} t={elapsed}s"
            )

    mean_loss = total_loss / len(loader)
    train_acc = total_correct / max(total_count, 1)
    return mean_loss, train_acc