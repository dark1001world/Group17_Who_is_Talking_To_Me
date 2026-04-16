class Config:
    final_embedding_path = "/DATA/G17/outputs/final_embedding.json"
    val_ratio = 0.2
    split_seed = 42

    batch_size = 4
    lr = 1e-4
    epochs = 30
    num_workers = 0

    dim_v = 512
    dim_a = 512

    device = "cuda"
    checkpoint_path = "fusion_model.pt"