import numpy as np
import torch


def accuracy(preds, labels, threshold=0.5):
    preds = (preds >= threshold).float()
    return (preds == labels).float().mean().item()


def frame_f1(preds, labels, threshold=0.5):
    pred_bin = (preds >= threshold).float()

    tp = ((pred_bin == 1) & (labels == 1)).sum().item()
    fp = ((pred_bin == 1) & (labels == 0)).sum().item()
    fn = ((pred_bin == 0) & (labels == 1)).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def roc_auc(preds, labels):
    y_score = preds.detach().cpu().numpy().reshape(-1)
    y_true = labels.detach().cpu().numpy().reshape(-1).astype(np.int32)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]

    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)

    tpr = np.concatenate(([0.0], tps / n_pos, [1.0]))
    fpr = np.concatenate(([0.0], fps / n_neg, [1.0]))

    return float(np.trapezoid(tpr, fpr))


def average_precision(preds, labels):
    y_score = preds.detach().cpu().numpy().reshape(-1)
    y_true = labels.detach().cpu().numpy().reshape(-1).astype(np.int32)

    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)

    positive_idx = np.where(y_sorted == 1)[0]
    if positive_idx.size == 0:
        return 0.0

    return float(precision[positive_idx].sum() / n_pos)


def compute_all_metrics(preds, labels, threshold=0.5):
    return {
        "accuracy": accuracy(preds, labels, threshold=threshold),
        "f1": frame_f1(preds, labels, threshold=threshold),
        "auc_roc": roc_auc(preds, labels),
        "average_precision": average_precision(preds, labels),
    }