import os
import torch
from tqdm import tqdm
from pathlib import Path

# Import your existing visual pipeline tools
from model import DinoViTTrackTTM

# CRITICAL IMPORT UPDATE: Bring in the Config dataclass
from data_loader import get_loader, ViTVisualConfig

@torch.no_grad()
def extract_visual_embeddings():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the Decapitated Model (Visual Only)
    model = DinoViTTrackTTM(
        vit_variant="vit_base_patch16_224",
        pretrained=False, # We are loading your trained weights
        num_frames=16,
    )
    
    ckpt_path = "experiments/new_visual_context2/best_model.pth"
    print(f"Loading visual weights from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    
    # Clean state dict (removes "module." and "_orig_mod.")
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    # ==========================================
    # 2. Setup DataLoaders (The Config Trick)
    # ==========================================
    
    # Base paths (Verify these match your system)
    IMG_SRC = "/DATA/G17/Data/video"
    TRACK_SRC = "/DATA/G17/starter_code/data/json_original"

    # Config for extracting Training data deterministically
    # TRICK: We put the train JSON into 'val_annotations' so we can use mode="val"
    cfg_train_extract = ViTVisualConfig(
        source_path=IMG_SRC,
        json_path=TRACK_SRC,
        val_annotations="/DATA/G17/Data/extract_data/ttm_train_data.json", 
        batch_size=32,
        num_workers=16,
        clip_frames=16,
        use_track=True,
        extracted_training_mode="context_clip",
        img_size=224
    )

    # Config for extracting Validation data deterministically
    cfg_val_extract = ViTVisualConfig(
        source_path=IMG_SRC,
        json_path=TRACK_SRC,
        val_annotations="/DATA/G17/Data/extract_data/ttm_validation_data.json",
        batch_size=32,
        num_workers=16,
        clip_frames=16,
        use_track=True,
        extracted_training_mode="context_clip",
        img_size=224
    )

    # By calling mode="val" for both, we guarantee shuffle=False and deterministic cropping
    loaders = {
        "train": get_loader(cfg_train_extract, mode="val"),
        "val":   get_loader(cfg_val_extract, mode="val")
    }
    # ==========================================

    # Dictionary to hold our final mapped vectors
    visual_embeddings = {}

    # 3. Extraction Loop
    for split, loader in loaders.items():
        print(f"\nExtracting {split} split...")
        pbar = tqdm(loader, desc=f"Visual Ext ({split})", dynamic_ncols=True)
        
        # We access the raw samples list to get the exact metadata for the current batch
        dataset_samples = loader.dataset.samples
        sample_idx = 0

        for batch in pbar:
            if len(batch) == 3:
                clips, track, labels = batch
            else:
                clips, labels = batch
                track = None

            clips = clips.to(device, non_blocking=True)
            if track is not None:
                track = track.to(device, non_blocking=True)

            # Pass through the decapitation bypass
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out_dict = model.extract_tokens(clips, track)
            
            # This is your golden [Batch, 768] tensor!
            cls_embeddings = out_dict["cls_embedding"].cpu().float()

            # Map each vector to its specific segment ID using the dataset index
            batch_len = clips.size(0)
            for i in range(batch_len):
                sample_meta = dataset_samples[sample_idx]
                
                # FIXED KEY NAMES: Matched to your visual ViTImagerLoader
                uid = sample_meta["uid"]
                pid = sample_meta["personid"]
                
                # Mathematically find the center frame ID from the list of fids
                fids_list = sample_meta["fids"]
                fid = fids_list[len(fids_list) // 2] 
                
                # Format perfectly matches audio: uid_pID_fID
                unique_key = f"{uid}_p{pid}_f{fid}"
                visual_embeddings[unique_key] = cls_embeddings[i].clone()
                
                sample_idx += 1

    # 4. Save to Disk
    out_dir = Path("/DATA/G17/outputs/new_dino_visual")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "visual_embeddings_dino.pt"
    
    print(f"\nSaving {len(visual_embeddings)} visual embeddings to {out_file}...")
    torch.save(visual_embeddings, out_file)
    print("Visual extraction complete. Check file size to confirm.")

if __name__ == "__main__":
    extract_visual_embeddings()