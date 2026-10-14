"""Generate a synthetic procession dataset to disk.

Layout: <out>/<split>/seq_XXXXXX/{f00.jpg..fNN.jpg, ann.npz}
Seeds are disjoint across splits; classes are balanced (state = seq_index % 3).
"""
import argparse, json, os, sys
from multiprocessing import Pool
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bharatcrowd.synth import SynthConfig, generate_sequence

SPLIT_SEED = {"train": 0, "val": 10_000_000, "test": 20_000_000}


def save(args):
    out, split, i, size, frames = args
    d = os.path.join(out, split, f"seq_{i:06d}")
    if os.path.exists(os.path.join(d, "ann.npz")):
        return
    os.makedirs(d, exist_ok=True)
    seed = SPLIT_SEED[split] + i
    cfg = SynthConfig(H=size, W=size, frames=frames)
    fr, an = generate_sequence(cfg, i % 3, seed)
    q = int(np.random.default_rng(seed).integers(50, 96))  # broadcast-style compression
    for k, f in enumerate(fr):
        cv2.imwrite(os.path.join(d, f"f{k:02d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, q])
    fidx = np.concatenate([np.full(len(p), k, np.int16) for k, p in enumerate(an["pts"])])
    np.savez_compressed(os.path.join(d, "ann.npz"), pts=np.concatenate(an["pts"]), frame=fidx,
                        group=np.concatenate(an["group"]), ppm=np.concatenate(an["ppm"]),
                        mask=np.packbits(an["mask"], axis=-1), box=an["box"], state=an["state"],
                        stats=an["stats"], shape=np.array(an["mask"].shape), meta=json.dumps(an["meta"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--train", type=int, default=1500)
    ap.add_argument("--val", type=int, default=150)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    jobs = [(a.out, s, i, a.size, a.frames) for s in ("val", "test", "train") for i in range(getattr(a, s))]
    with Pool(a.workers) as p:
        for n, _ in enumerate(p.imap_unordered(save, jobs, chunksize=4)):
            if n % 200 == 0:
                print(f"{n}/{len(jobs)}", flush=True)
    print("done")
