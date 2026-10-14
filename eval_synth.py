"""Synthetic test-set evaluation with the controls needed to trust the numbers.

* full model
* motion removed at test time      -> behaviour must collapse if it is really motion-driven
* oracle window mask               -> isolates window-localisation error from counting/behaviour error
* density-only baseline            -> how much of the behaviour label is decodable from
                                      ground-truth window density alone (generator shortcut check)
* physics-stats baseline           -> GT (density, speed, dispersion, pressure) -> state; a
                                      ceiling for what the labels encode
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from bharatcrowd.data import SynthClips
from bharatcrowd.evaluate import evaluate
from bharatcrowd.model import BharatCrowdNet


def gt_stats(root, split):
    X, y = [], []
    for d in sorted(glob.glob(os.path.join(root, split, "seq_*"))):
        if not os.path.exists(os.path.join(d, "ann.npz")):
            continue
        a = np.load(os.path.join(d, "ann.npz"))
        X.append(np.log1p(np.maximum(a["stats"][-1], 0))); y.append(int(a["state"][-1]))
    return np.array(X, np.float32), np.array(y)


def logreg(Xtr, ytr, Xte, yte, cols, epochs=2000):
    Xtr, Xte = torch.tensor(Xtr[:, cols]), torch.tensor(Xte[:, cols])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    # small MLP so non-monotone density->state relations are not under-fit
    m = torch.nn.Sequential(torch.nn.Linear(len(cols), 32), torch.nn.ReLU(), torch.nn.Linear(32, 3))
    opt = torch.optim.Adam(m.parameters(), 1e-2, weight_decay=1e-4)
    yt = torch.tensor(ytr)
    for _ in range(epochs):
        opt.zero_grad(); torch.nn.functional.cross_entropy(m(Xtr), yt).backward(); opt.step()
    return float((m(Xte).argmax(1).numpy() == yte).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth")
    ap.add_argument("--ckpt", default="runs/synth/best.pt")
    ap.add_argument("--out", default="runs/synth/test_results.json")
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    model = BharatCrowdNet(pretrained=False)
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    dl = DataLoader(SynthClips(a.data, "test"), 8, num_workers=2)
    res = dict(full=evaluate(model, dl), no_motion=evaluate(model, dl, zero_motion=True),
               oracle_mask=evaluate(model, dl, oracle_mask=True))
    Xtr, ytr = gt_stats(a.data, "train")
    Xte, yte = gt_stats(a.data, "test")
    res["baseline_density_only_acc"] = logreg(Xtr, ytr, Xte, yte, [0])
    res["baseline_gt_physics_acc"] = logreg(Xtr, ytr, Xte, yte, [0, 1, 2, 3])
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
