"""Evaluation: counting (full image and moving window), window IoU, behaviour."""
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate(model, loader, device="cpu", zero_motion=False, oracle_mask=False, n_states=3):
    model.eval()
    c_err, r_err, r_err_oracle, r_rel, ious, y, p, rows = [], [], [], [], [], [], [], []
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        flow, hm = b["flow"], b["has_motion"]
        if zero_motion:
            flow, hm = torch.zeros_like(flow), torch.zeros_like(hm)
        override = None
        if oracle_mask:
            override = F.interpolate(b["mask"], size=(b["img"].shape[2] // 8, b["img"].shape[3] // 8), mode="area")
        out = model(b["img"], flow, hm, mask_override=override)
        m4 = F.interpolate(b["mask"], size=out["density"].shape[2:], mode="area")
        roi_oracle = (out["density"] * m4).sum((1, 2, 3))
        pm = (torch.sigmoid(out["region"]) > 0.5).float()
        gm = (F.interpolate(b["mask"], size=pm.shape[2:], mode="area") > 0.5).float()
        inter, union = (pm * gm).sum((1, 2, 3)), ((pm + gm) > 0).float().sum((1, 2, 3))
        for i in range(len(pm)):
            if b["has_points"][i] > 0:
                c_err.append(float(out["count"][i] - b["count"][i]))
                if b["has_mask"][i] > 0:
                    r_err.append(float(out["roi_count"][i] - b["roi_count"][i]))
                    r_err_oracle.append(float(roi_oracle[i] - b["roi_count"][i]))
                    r_rel.append(abs(r_err[-1]) / max(1.0, float(b["roi_count"][i])))
            if b["has_mask"][i] > 0:
                ious.append(float(inter[i] / union[i].clamp(min=1)))
            if b["has_state"][i] > 0:
                y.append(int(b["state"][i]))
                p.append(int(out["logits"][i].argmax()))
            rows.append(dict(count=float(b["count"][i]), pred=float(out["count"][i]),
                             roi=float(b["roi_count"][i]), roi_pred=float(out["roi_count"][i])))
    res = {}
    if c_err:
        c = np.array(c_err)
        res.update(count_mae=np.abs(c).mean(), count_rmse=np.sqrt((c ** 2).mean()))
    if r_err:
        r, ro = np.array(r_err), np.array(r_err_oracle)
        res.update(roi_mae=np.abs(r).mean(), roi_rmse=np.sqrt((r ** 2).mean()), roi_mape=float(np.mean(r_rel)),
                   roi_mae_oracle_mask=np.abs(ro).mean())
    if ious:
        res["window_miou"] = float(np.mean(ious))
    if y:
        y, p = np.array(y), np.array(p)
        cm = np.zeros((n_states, n_states), int)
        np.add.at(cm, (y, p), 1)
        f1 = []
        for k in range(n_states):
            tp, fp, fn = cm[k, k], cm[:, k].sum() - cm[k, k], cm[k].sum() - cm[k, k]
            f1.append(2 * tp / max(1, 2 * tp + fp + fn))
        res.update(beh_acc=float((y == p).mean()), beh_macro_f1=float(np.mean(f1)), confusion=cm.tolist())
    res["mean_gt_count"] = float(np.mean([r["count"] for r in rows])) if rows else 0.0
    res["mean_gt_roi"] = float(np.mean([r["roi"] for r in rows])) if rows else 0.0
    return {k: (round(float(v), 4) if not isinstance(v, list) else v) for k, v in res.items()}
