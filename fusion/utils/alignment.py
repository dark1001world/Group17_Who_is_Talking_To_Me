import torch


def pad_sequence(x, max_len):
    pad_size = max_len - x.shape[0]
    return torch.cat([x, torch.zeros(pad_size, x.shape[1])], dim=0)


def collate_fn(batch):
    visuals, audios, labels = zip(*batch)

    max_v = max(v.shape[0] for v in visuals)
    max_a = max(a.shape[0] for a in audios)

    visuals = torch.stack([pad_sequence(v, max_v) for v in visuals])
    audios = torch.stack([pad_sequence(a, max_a) for a in audios])
    labels = torch.stack([
        torch.cat([l, torch.zeros(max_v - l.shape[0])]) for l in labels
    ])

    return visuals, audios, labels