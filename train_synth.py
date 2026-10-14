"""Stage 1: train BharatCrowdNet on the synthetic procession dataset."""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bharatcrowd.data import SynthClips
from bharatcrowd.evaluate import evaluate
from bharatcrowd.losses import compute_loss
from bharatcrowd.model import BharatCrowdNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth")
    ap.add_argument("--out", default="runs/synth")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--p_drop_motion", type=float, default=0.25)
    ap.add_argument("--p_teacher", type=float, default=0.5, help="P(use GT mask for behaviour pooling)")
    ap.add_argument("--max_train", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.set_num_threads(max(1, os.cpu_count() - a.workers))
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr = SynthClips(a.data, "train", train=True, p_drop_motion=a.p_drop_motion)
    if a.max_train:
        tr.dirs = tr.dirs[:a.max_train]
    va = SynthClips(a.data, "val")
    dl = DataLoader(tr, a.bs, shuffle=True, num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0,
                    worker_init_fn=lambda w: setattr(tr, "rng", np.random.default_rng(a.seed * 1000 + w +
                                                                                      int(time.time()))))
    dv = DataLoader(va, a.bs, num_workers=a.workers)

    model = BharatCrowdNet(pretrained=True).to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    bb = [p for n, p in model.named_parameters() if n.startswith(("s4.", "s8.", "s16."))]
    hd = [p for n, p in model.named_parameters() if not n.startswith(("s4.", "s8.", "s16."))]
    opt = torch.optim.AdamW([{"params": bb, "lr": a.lr * 0.3}, {"params": hd, "lr": a.lr}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr * 0.3, a.lr], total_steps=a.epochs * len(dl),
                                                pct_start=0.1)
    best, log = -1e9, []
    for ep in range(a.epochs):
        model.train()
        t0, agg = time.time(), {}
        for b in dl:
            b = {k: v.to(dev) for k, v in b.items()}
            override = None
            if np.random.rand() < a.p_teacher:
                override = F.interpolate(b["mask"], size=(b["img"].shape[2] // 8, b["img"].shape[3] // 8),
                                         mode="area")
            out = model(b["img"], b["flow"], b["has_motion"], mask_override=override)
            loss, parts = compute_loss(out, b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0) + v / len(dl)
        res = evaluate(model, dv, dev)
        # model selection: behaviour F1 minus relative ROI counting error
        score = res.get("beh_macro_f1", 0) - res.get("roi_mape", 0) + res.get("window_miou", 0)
        log.append(dict(epoch=ep, train=agg, val=res, sec=time.time() - t0))
        print(json.dumps(log[-1]), flush=True)
        torch.save(model.state_dict(), os.path.join(a.out, "last.pt"))
        if score > best:
            best = score
            torch.save(model.state_dict(), os.path.join(a.out, "best.pt"))
    json.dump(log, open(os.path.join(a.out, "log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
