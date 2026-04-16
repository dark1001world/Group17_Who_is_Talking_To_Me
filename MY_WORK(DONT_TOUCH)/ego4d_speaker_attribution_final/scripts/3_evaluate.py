import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.dataset.ego4d_track_dataset import Ego4DTrackDataset
from src.fusion.speaker_model import SpeakerAttributionModel


def evaluate():
    config = load_config('configs/default.yaml')
    logger = setup_logger('evaluate', config['logging']['log_dir'])
    device = torch.device(config['training']['device'])

    dataset = Ego4DTrackDataset(config['data']['output_dir'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    vis_dim = config['visual']['proj_vit_dim'] + config['visual']['proj_reid_dim'] + config['visual']['proj_lip_dim']
    model = SpeakerAttributionModel(
        audio_dim=768, visual_dim=vis_dim, fusion_dim=config['fusion']['fusion_dim'],
        num_heads=config['fusion']['num_heads'], num_temporal_layers=config['fusion']['num_temporal_layers'],
        dropout=config['fusion']['dropout']
    ).to(device)

    checkpoint = 'models/epoch_20.pth'
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            audio = batch['audio'].to(device)
            visual = batch['visual'].to(device)
            mask = batch['track_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(audio, visual, mask)
            all_probs.append(outputs[mask].cpu().numpy())
            all_labels.append(labels[mask].cpu().numpy())

    logger.info("Evaluation completed.")


if __name__ == "__main__":
    evaluate()
