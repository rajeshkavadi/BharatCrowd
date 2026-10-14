"""Run BharatCrowdNet on a video (sliding clips) or on still images.

Video: every --stride frames, the last --clip frames are processed; the moving-window
mask is EMA-tracked over time so the window follows the procession; the alert engine
consumes behaviour posteriors and window counts.
Still images: counting + appearance-only window; behaviour is reported as unknown.
"""
import argparse
import glob
import json
import os
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from bharatcrowd.alert import AlertEngine
from bharatcrowd.data import compute_flows, to_tensor_img
from bharatcrowd.model import BharatCrowdNet
from bharatcrowd.synth import STATES


def fit_size(h, w, H):
    W = int(round(w * H / h / 32)) * 32
    return H, max(32, W)


@torch.no_grad()
def run_clip(model, frames, has_motion, clip):
    if has_motion:
        flow = torch.from_numpy(compute_flows(frames))[None]
    else:
        H, W = frames[-1].shape[:2]
        flow = torch.zeros(1, clip - 1, 2, H // 2, W // 2)
    out = model(to_tensor_img(frames[-1])[None], flow, torch.tensor([float(has_motion)]))
    mask = torch.sigmoid(out["region"])[0, 0].numpy()
    return out, mask


def draw(frame, mask, out, alert, has_motion):
    vis = frame.copy()
    m = cv2.resize(mask, (frame.shape[1], frame.shape[0])) > 0.5
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    col = [(0, 200, 0), (0, 200, 255), (0, 0, 255)][alert["level"]] if alert else (255, 255, 0)
    cv2.drawContours(vis, cnts, -1, col, 2)
    if cnts:
        box = cv2.boxPoints(cv2.minAreaRect(np.concatenate(cnts)))
        cv2.polylines(vis, [box.astype(np.int32)], True, col, 1)
    p = torch.softmax(out["logits"][0], 0).numpy()
    txt = [f"total~{out['count'].item():.0f}  window~{out['roi_count'].item():.0f}"]
    txt.append(f"{STATES[p.argmax()]} {p.max():.2f}" if has_motion else "behaviour: n/a (no motion)")
    if alert:
        txt.append(f"ALERT {alert['name']} {'; '.join(alert['reasons'])}")
    for i, t in enumerate(txt):
        cv2.putText(vis, t, (8, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, t, (8, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--video")
    ap.add_argument("--images", help="folder of still images")
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--clip", type=int, default=8)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--frame_step", type=int, default=0, help="subsample video to ~3 fps if 0")
    ap.add_argument("--ppm", type=float, default=None, help="pixels/metre at the window (calibrated camera)")
    ap.add_argument("--out", default="runs/infer")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model = BharatCrowdNet(pretrained=False)
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    model.eval()
    log = open(os.path.join(a.out, "log.jsonl"), "w")

    if a.images:
        for p in sorted(glob.glob(os.path.join(a.images, "*.jpg")) + glob.glob(os.path.join(a.images, "*.png"))):
            if p.endswith(("_mask.png", "_overlay.jpg")):
                continue
            img = cv2.imread(p)
            H, W = fit_size(*img.shape[:2], a.height)
            fr = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            out, mask = run_clip(model, [fr], False, a.clip)
            cv2.imwrite(os.path.join(a.out, os.path.basename(p)), draw(fr, mask, out, None, False))
            rec = dict(file=os.path.basename(p), count=out["count"].item(), roi_count=out["roi_count"].item(),
                       window_frac=float((mask > 0.5).mean()))
            log.write(json.dumps(rec) + "\n"); print(rec)
        return

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = a.frame_step or max(1, int(round(fps / 3)))
    buf, ema, engine, writer, i = deque(maxlen=a.clip), None, AlertEngine(), None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i % step:
            continue
        H, W = fit_size(*frame.shape[:2], a.height)
        buf.append(cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA))
        if len(buf) < a.clip or (i // step) % a.stride:
            continue
        out, mask = run_clip(model, list(buf), True, a.clip)
        ema = mask if ema is None else 0.5 * ema + 0.5 * mask
        area = float((cv2.resize(ema, (W, H)) > 0.5).sum())
        alert = engine.update(torch.softmax(out["logits"][0], 0).numpy(), out["roi_count"].item(), area, a.ppm)
        vis = draw(buf[-1], ema, out, alert, True)
        if writer is None:
            writer = cv2.VideoWriter(os.path.join(a.out, "out.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                     max(1.0, 3.0 / a.stride), (W, H))
        writer.write(vis)
        log.write(json.dumps(dict(frame=i, count=out["count"].item(), roi_count=out["roi_count"].item(), **alert)) + "\n")
    if writer:
        writer.release()


if __name__ == "__main__":
    main()
