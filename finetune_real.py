"""Stage 2: fine-tune the synthetic model on real Indian festival data (partial labels).

Data: folders in RealFrames json format (see bharatcrowd/data.py), e.g. the output
of tools/yellow_roi.py (mask only), point-annotated frames (counting), and short
clips with behaviour labels. Each loss term is enabled only where its label exists.

With tiny real sets, freeze the backbone BatchNorm statistics (--freeze_bn) and
optionally the motion branch (--freeze_motion) when the real data has no video.
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from bharatcrowd.data import RealFrames, SynthClips
from bharatcrowd.evaluate import evaluate
from bharatcrowd.losses import compute_loss
from bharatcrowd.model import BharatCrowdNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train", nargs="+", required=True, help="RealFrames folders")
    ap.add_argument("--val", nargs="*", default=[])
    ap.add_argument("--synth", default="", help="mix in synthetic clips (replay) to limit forgetting")
    ap.add_argument("--synth_n", type=int, default=300)
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--width", type=int, default=576)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--freeze_bn", action="store_true")
    ap.add_argument("--freeze_motion", action="store_true")
    ap.add_argument("--out", default="runs/finetune")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    size = (a.height, a.width)

    sets = [RealFrames(r, size=size, train=True) for r in a.train]
    if a.synth:
        s = SynthClips(a.synth, "train", train=True, p_drop_motion=0.25)
        s.dirs = list(np.random.default_rng(0).permutation(s.dirs)[:a.synth_n])
        sets.append(s)
    # synthetic clips are square: batch real and synthetic separately via bs=1 when mixing sizes
    bs = 1 if a.synth else a.bs
    dl = DataLoader(ConcatDataset(sets), bs, shuffle=True, num_workers=a.workers)
    dv = [DataLoader(RealFrames(r, size=size), 1) for r in a.val]

    model = BharatCrowdNet(pretrained=False).to(dev)
    model.load_state_dict(torch.load(a.ckpt, map_location=dev))
    if a.freeze_motion:
        for n, p in model.named_parameters():
            if n.startswith(("motion.", "gru.")):
                p.requires_grad = False
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=1e-4)
    accum = max(1, a.bs // bs)
    log = []
    for ep in range(a.epochs):
        model.train()
        if a.freeze_bn:
            for m in model.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()
        agg = {}
        for it, b in enumerate(dl):
            b = {k: v.to(dev) for k, v in b.items()}
            out = model(b["img"], b["flow"], b["has_motion"])
            loss, parts = compute_loss(out, b)
            (loss / accum).backward()
            if (it + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step(); opt.zero_grad()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0) + v / len(dl)
        res = [evaluate(model, d, dev) for d in dv]
        log.append(dict(epoch=ep, train=agg, val=res))
        print(json.dumps(log[-1]), flush=True)
        torch.save(model.state_dict(), os.path.join(a.out, "last.pt"))
    json.dump(log, open(os.path.join(a.out, "log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
