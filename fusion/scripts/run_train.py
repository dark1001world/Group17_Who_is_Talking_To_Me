import torch
import json
import sys
from pathlib import Path
from torch.utils.data import DataLoader

FUSION_ROOT = Path(__file__).resolve().parents[1]
if str(FUSION_ROOT) not in sys.path:
    sys.path.insert(0, str(FUSION_ROOT))

from models.fusion_model import AVFusion
from data.dataset import FusionDataset
from utils.collate import collate_fn
from engine.train import train_one_epoch
from engine.eval import evaluate
from utils.config import Config


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    training_log_path = Path(__file__).resolve().parents[1] / "training.json"

    train_data = FusionDataset(
        Config.final_embedding_path,
        split="train",
        val_ratio=Config.val_ratio,
        seed=Config.split_seed,
    )
    val_data = FusionDataset(
        Config.final_embedding_path,
        split="val",
        val_ratio=Config.val_ratio,
        seed=Config.split_seed,
    )

    sample_visual, sample_audio, _ = train_data[0]
    dim_v = sample_visual.shape[-1]
    dim_a = sample_audio.shape[-1]

    print(f"Train samples: {len(train_data)}")
    print(f"Val samples:   {len(val_data)}")
    print(f"Input dims: visual={dim_v}, audio={dim_a}")

    train_loader = DataLoader(
        train_data,
        batch_size=Config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=Config.num_workers,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=Config.batch_size,
        collate_fn=collate_fn,
        num_workers=Config.num_workers,
    )

    model = AVFusion(dim_v, dim_a).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    epoch_logs = []

    for epoch in range(Config.epochs):
        loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch_idx=epoch + 1,
            log_interval=50,
        )
        metrics = evaluate(model, val_loader, device)

        epoch_record = {
            "epoch": epoch + 1,
            "loss": float(loss),
            "train_acc": float(train_acc),
            "val_acc": float(metrics["accuracy"]),
            "auc_roc": float(metrics["auc_roc"]),
            "average_precision": float(metrics["average_precision"]),
            "frame_f1": float(metrics["f1"]),
        }
        epoch_logs.append(epoch_record)

        with open(training_log_path, "w") as f:
            json.dump(
                {
                    "config": {
                        "final_embedding_path": Config.final_embedding_path,
                        "val_ratio": Config.val_ratio,
                        "split_seed": Config.split_seed,
                        "batch_size": Config.batch_size,
                        "lr": Config.lr,
                        "epochs": Config.epochs,
                    },
                    "epochs": epoch_logs,
                },
                f,
                indent=2,
            )

        print(
            f"Epoch {epoch+1}: "
            f"Loss={loss:.4f}, "
            f"TrainAcc={train_acc:.4f}, "
            f"ValAcc={metrics['accuracy']:.4f}, "
            f"AUC-ROC={metrics['auc_roc']:.4f}, "
            f"AP={metrics['average_precision']:.4f}, "
            f"Frame-F1={metrics['f1']:.4f}"
        )

    torch.save(model.state_dict(), Config.checkpoint_path)
    print(f"Saved model to {Config.checkpoint_path}")
    print(f"Saved training metrics to {training_log_path}")


if __name__ == "__main__":
    main()