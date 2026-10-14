"""Datasets, optical flow and target construction."""
import glob
import json
import os

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset

IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)
DEN_STRIDE = 4      # density map stride
FLOW_SCALE = 0.5    # flow is computed at half resolution
STATS_NAMES = ("density_p_m2", "mean_speed_m_s", "dispersion_m_s", "pressure")


# ----------------------------------------------------------------------------- flow
def _affine_fit(flow, n=24, keep=0.7):
    """Robust global affine motion (camera pan/zoom/jitter) from a dense flow field."""
    h, w = flow.shape[:2]
    ys, xs = np.meshgrid(np.linspace(2, h - 3, n), np.linspace(2, w - 3, n), indexing="ij")
    ys, xs = ys.ravel().astype(int), xs.ravel().astype(int)
    F = flow[ys, xs]
    A = np.stack([xs, ys, np.ones_like(xs)], 1).astype(np.float32)
    sel = np.ones(len(A), bool)
    for _ in range(3):  # trimmed least squares: the static crowd dominates the frame
        P, *_ = np.linalg.lstsq(A[sel], F[sel], rcond=None)
        r = np.linalg.norm(A @ P - F, axis=1)
        sel = r <= np.quantile(r, keep)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return gx[..., None] * P[0] + gy[..., None] * P[1] + P[2]


def compute_flows(frames_bgr, compensate=True):
    """Farneback flow between consecutive frames at FLOW_SCALE, ego-motion removed.

    Returns (T-1, 2, h, w) float32 in pixels of the *flow* resolution per frame.
    """
    g = [cv2.cvtColor(cv2.resize(f, None, fx=FLOW_SCALE, fy=FLOW_SCALE, interpolation=cv2.INTER_AREA),
                      cv2.COLOR_BGR2GRAY) for f in frames_bgr]
    out = []
    for a, b in zip(g[:-1], g[1:]):
        fl = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 9, 3, 5, 1.1, 0)
        if compensate:
            fl = fl - _affine_fit(fl)
        out.append(fl.transpose(2, 0, 1))
    return np.stack(out).astype(np.float32)


# ----------------------------------------------------------------------------- targets
def density_map(pts, H, W, stride=DEN_STRIDE, sigma=1.0):
    """Count-preserving density at 1/stride resolution (bilinear splat + Gaussian)."""
    h, w = H // stride, W // stride
    d = np.zeros((h, w), np.float32)
    if len(pts):
        x = np.clip(pts[:, 0] / stride - 0.5, 0, w - 1.001)
        y = np.clip(pts[:, 1] / stride - 0.5, 0, h - 1.001)
        x0, y0 = x.astype(int), y.astype(int)
        fx, fy = x - x0, y - y0
        for dx, dy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            np.add.at(d, (y0 + dy, x0 + dx), wt)
        s = d.sum()
        d = gaussian_filter(d, sigma, mode="constant")
        d *= s / max(d.sum(), 1e-6)
    return d


def to_tensor_img(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(((rgb - IMNET_MEAN) / IMNET_STD).transpose(2, 0, 1).copy())


def photometric(frames, rng):
    """Same colour jitter on every frame of the clip (keeps flow consistent)."""
    f = frames.astype(np.float32)
    f = f * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25)
    f = f * rng.uniform(0.85, 1.15, 3)
    if rng.random() < 0.3:
        f = f.mean(-1, keepdims=True).repeat(3, -1)
    return np.clip(f, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- synthetic
class SynthClips(Dataset):
    """Clips from tools/generate_synth.py. Targets refer to the last (key) frame."""

    def __init__(self, root, split, train=False, p_drop_motion=0.0, clip_len=None):
        self.dirs = sorted(glob.glob(os.path.join(root, split, "seq_*")))
        self.dirs = [d for d in self.dirs if os.path.exists(os.path.join(d, "ann.npz"))]
        self.train, self.p_drop_motion, self.clip_len = train, p_drop_motion, clip_len
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, i):
        d = self.dirs[i]
        a = np.load(os.path.join(d, "ann.npz"))
        T, H, W = a["shape"]
        L = self.clip_len or T
        files = sorted(glob.glob(os.path.join(d, "f*.jpg")))[T - L:]
        frames = np.stack([cv2.imread(f) for f in files])
        k = T - 1
        sel = a["frame"] == k
        pts, group = a["pts"][sel].copy(), a["group"][sel]
        mask = np.unpackbits(a["mask"][k], axis=-1)[:, :W].astype(np.float32)
        rng = self.rng
        flipped = False
        if self.train:
            frames = photometric(frames, rng)
            if rng.random() < 0.5:  # horizontal flip
                flipped = True
                frames = frames[:, :, ::-1].copy()
                mask = mask[:, ::-1].copy()
                pts[:, 0] = W - 1 - pts[:, 0]
        fcache = os.path.join(d, "flow.npy")
        if os.path.exists(fcache) and L == T:
            flows = np.load(fcache).astype(np.float32)
            if flipped:
                flows = flows[..., ::-1].copy()
                flows[:, 0] *= -1
        else:
            flows = compute_flows(frames)
        has_motion = 1.0
        if self.train and rng.random() < self.p_drop_motion:
            flows[:] = 0
            has_motion = 0.0
        inroi = mask[np.clip(pts[:, 1].astype(int), 0, H - 1), np.clip(pts[:, 0].astype(int), 0, W - 1)] > 0
        stats = a["stats"][k]
        return dict(
            img=to_tensor_img(frames[-1]), flow=torch.from_numpy(flows), has_motion=torch.tensor(has_motion),
            density=torch.from_numpy(density_map(pts, H, W))[None], mask=torch.from_numpy(mask)[None],
            count=torch.tensor(float(len(pts))), roi_count=torch.tensor(float(inroi.sum())),
            state=torch.tensor(int(a["state"][k])), has_state=torch.tensor(1.0),
            stats=torch.from_numpy(np.log1p(np.maximum(stats, 0))), has_stats=torch.tensor(has_motion),
            has_points=torch.tensor(1.0),
            has_mask=torch.tensor(1.0))


# ----------------------------------------------------------------------------- real data
class RealFrames(Dataset):
    """Real images / short clips with *partial* labels.

    Expected layout (one sample per json):
        <root>/<name>.json  {"frames": ["a.jpg", ...] or "image": "a.jpg",
                             "points": [[x, y], ...]   (optional, key-frame heads),
                             "roi_mask": "a_mask.png"  (optional, moving-region mask),
                             "state": 0|1|2            (optional, needs >=2 frames)}
    Missing labels are masked out of the loss. Single images get zero flow and
    has_motion=0, so behaviour is neither trained nor predicted for them.
    """

    def __init__(self, root, size=(320, 576), train=False, clip_len=8):
        self.items = sorted(glob.glob(os.path.join(root, "*.json")))
        self.root, self.size, self.train, self.clip_len = root, size, train, clip_len
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        j = json.load(open(self.items[i]))
        names = j.get("frames") or [j["image"]]
        names = names[-self.clip_len:]
        frames = [cv2.imread(os.path.join(self.root, n)) for n in names]
        H0, W0 = frames[-1].shape[:2]
        H, W = self.size
        frames = np.stack([cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA) for f in frames])
        sx, sy = W / W0, H / H0
        pts = np.array(j.get("points", []), np.float32).reshape(-1, 2) * [sx, sy]
        has_points = float("points" in j)
        mask = np.zeros((H, W), np.float32)
        has_mask = 0.0
        if j.get("roi_mask"):
            m = cv2.imread(os.path.join(self.root, j["roi_mask"]), cv2.IMREAD_GRAYSCALE)
            mask = (cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 127).astype(np.float32)
            has_mask = 1.0
        if self.train:
            frames = photometric(frames, self.rng)
            if self.rng.random() < 0.5:
                frames, mask = frames[:, :, ::-1].copy(), mask[:, ::-1].copy()
                pts[:, 0] = W - 1 - pts[:, 0]
        if len(frames) > 1:
            flows, has_motion = compute_flows(frames), 1.0
        else:
            flows = np.zeros((self.clip_len - 1, 2, int(H * FLOW_SCALE), int(W * FLOW_SCALE)), np.float32)
            has_motion = 0.0
        inroi = mask[np.clip(pts[:, 1].astype(int), 0, H - 1), np.clip(pts[:, 0].astype(int), 0, W - 1)] > 0 \
            if len(pts) else np.zeros(0, bool)
        state = j.get("state", -1)
        return dict(
            img=to_tensor_img(frames[-1]), flow=torch.from_numpy(flows), has_motion=torch.tensor(has_motion),
            density=torch.from_numpy(density_map(pts, H, W))[None], mask=torch.from_numpy(mask)[None],
            count=torch.tensor(float(len(pts))), roi_count=torch.tensor(float(inroi.sum())),
            state=torch.tensor(max(state, 0)), has_state=torch.tensor(float(state >= 0 and has_motion > 0)),
            stats=torch.zeros(4), has_stats=torch.tensor(0.0), has_points=torch.tensor(has_points), has_mask=torch.tensor(has_mask))
