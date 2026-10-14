"""Multi-task loss with per-sample label availability masks (partial labels on real data)."""
import torch
import torch.nn.functional as F

W = dict(den=1.0, cnt=0.5, roi=0.5, mask=1.0, beh=1.0, stats=0.5)


def _masked_mean(x, m):
    return (x * m).sum() / m.sum().clamp(min=1.0)


def dice_bce(logit, target):
    p = torch.sigmoid(logit)
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none").mean((1, 2, 3))
    inter = (p * target).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return bce + dice


def compute_loss(out, b, model_den_gt_mask=True):
    hp, hm, hs, hst = b["has_points"], b["has_mask"], b["has_state"], b["has_stats"]
    den, gt = out["density"], b["density"]
    l_den = _masked_mean(((den - gt) ** 2).mean((1, 2, 3)) * 10, hp)
    l_cnt = _masked_mean((out["count"] - b["count"]).abs() / (b["count"] + 10), hp)
    m_gt = F.interpolate(b["mask"], size=den.shape[2:], mode="area")
    roi_gt_mask = (den * m_gt).sum((1, 2, 3))
    l_roi = _masked_mean((roi_gt_mask - b["roi_count"]).abs() / (b["roi_count"] + 10), hp * hm)
    tgt8 = F.interpolate(b["mask"], size=out["region"].shape[2:], mode="area")
    l_mask = _masked_mean(dice_bce(out["region"], tgt8), hm)
    l_beh = _masked_mean(F.cross_entropy(out["logits"], b["state"], reduction="none"), hs)
    l_st = _masked_mean(F.smooth_l1_loss(out["stats"], b["stats"], reduction="none").mean(1), hst)
    parts = dict(den=l_den, cnt=l_cnt, roi=l_roi, mask=l_mask, beh=l_beh, stats=l_st)
    total = sum(W[k] * v for k, v in parts.items())
    return total, {k: float(v.detach()) for k, v in parts.items()}
