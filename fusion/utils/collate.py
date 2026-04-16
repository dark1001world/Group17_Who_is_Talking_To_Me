import torch


def pad_tensor(x, max_len):
    pad_len = max_len - x.size(0)
    if pad_len == 0:
        return x, torch.zeros(max_len, dtype=torch.bool)

    pad = torch.zeros(pad_len, x.size(1))
    mask = torch.cat([torch.zeros(x.size(0)), torch.ones(pad_len)]).bool()

    return torch.cat([x, pad], dim=0), mask


def collate_fn(batch):
    visuals, audios, labels = zip(*batch)

    max_v = max(v.size(0) for v in visuals)
    max_a = max(a.size(0) for a in audios)

    v_out, v_mask = [], []
    a_out, a_mask = [], []
    l_out = []

    for v, a, l in zip(visuals, audios, labels):
        v_pad, v_m = pad_tensor(v, max_v)
        a_pad, a_m = pad_tensor(a, max_a)

        l_pad = torch.cat([l, torch.zeros(max_v - l.size(0))])

        v_out.append(v_pad)
        v_mask.append(v_m)

        a_out.append(a_pad)
        a_mask.append(a_m)

        l_out.append(l_pad)

    return (
        torch.stack(v_out),
        torch.stack(a_out),
        torch.stack(l_out),
        torch.stack(v_mask),
        torch.stack(a_mask),
    )