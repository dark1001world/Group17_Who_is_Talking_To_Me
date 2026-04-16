import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.dataset.ego4d_track_dataset import Ego4DTrackDataset
from src.fusion.speaker_model import SpeakerAttributionModel
from src.losses.focal_loss import FocalLoss
from src.losses.smoothness_loss import temporal_smoothness_loss


def train():
    config = load_config('configs/default.yaml')
    logger = setup_logger('train', config['logging']['log_dir'])
    device = torch.device(config['training']['device'])
    os.makedirs('models', exist_ok=True)

    dataset = Ego4DTrackDataset(config['data']['output_dir'])
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

    vis_dim = config['visual']['proj_vit_dim'] + config['visual']['proj_reid_dim'] + config['visual']['proj_lip_dim']
    model = SpeakerAttributionModel(
        audio_dim=768,
        visual_dim=vis_dim,
        fusion_dim=config['fusion']['fusion_dim'],
        num_heads=config['fusion']['num_heads'],
        num_temporal_layers=config['fusion']['num_temporal_layers'],
        dropout=config['fusion']['dropout']
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    focal_loss = FocalLoss(alpha=config['training']['focal_loss']['alpha'],
                           gamma=config['training']['focal_loss']['gamma'])
    lambda_smooth = config['training']['lambda_smooth']

    logger.info("Starting training...")
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            audio = batch['audio'].to(device)
            visual = batch['visual'].to(device)
            mask = batch['track_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(audio, visual, mask)

            pos_frames = (labels.sum(dim=-1) > 0).unsqueeze(-1)
            window = config['training']['pos_frame_window']
            if window > 1:
                pos_frames = pos_frames.float().permute(0,2,1)
                pos_frames = F.max_pool1d(pos_frames, kernel_size=window, stride=1, padding=window//2)
                pos_frames = pos_frames.permute(0,2,1) > 0
            loss_mask = pos_frames & mask

            fl = focal_loss(outputs[loss_mask], labels[loss_mask])
            sm = temporal_smoothness_loss(outputs, mask)
            loss = fl + lambda_smooth * sm

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), os.path.join('models', f'epoch_{epoch+1}.pth'))

    logger.info("Training completed.")


if __name__ == "__main__":
    train()
