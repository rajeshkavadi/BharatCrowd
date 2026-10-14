"""Precompute ego-motion-compensated Farneback flow for synthetic clips (float16 flow.npy).

Saves ~0.2 s CPU per clip per epoch. Trade-off: photometric augmentation no longer
perturbs the flow input (flip is applied consistently in the dataset).
"""
import glob, os, sys
from multiprocessing import Pool
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bharatcrowd.data import compute_flows


def one(d):
    out = os.path.join(d, "flow.npy")
    if os.path.exists(out) or not os.path.exists(os.path.join(d, "ann.npz")):
        return
    frames = np.stack([cv2.imread(f) for f in sorted(glob.glob(os.path.join(d, "f*.jpg")))])
    np.save(out, compute_flows(frames).astype(np.float16))


if __name__ == "__main__":
    dirs = sorted(glob.glob(os.path.join(sys.argv[1], "*", "seq_*")))
    with Pool(os.cpu_count()) as p:
        p.map(one, dirs, chunksize=8)
    print("cached", len(dirs))
