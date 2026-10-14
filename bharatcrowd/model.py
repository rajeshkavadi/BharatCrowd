"""BharatCrowdNet: moving-window crowd counting and behaviour estimation.

Factorisation (the region is unknown and can be anywhere):

    D  = density(appearance(key frame))                 full-image density, stride 4
    M  = sigmoid(region(appearance, motion, context))   moving-window mask, stride 8
    N_roi = sum(D * up(M))                              count inside the window
    y  = behaviour(mask-pooled spatio-temporal features, mask-pooled flow statistics)

Counting is appearance-only on purpose: stationary spectators must be counted as
well as movers, and it lets the density branch be fine-tuned on image-only
datasets (JHU-Crowd++, ShanghaiTech-A, UCF-QNRF) where no video exists. Behaviour
is only defined when motion is available (has_motion=1).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def cbr(i, o, k=3, s=1, d=1):
    return nn.Sequential(nn.Conv2d(i, o, k, s, padding=d * (k // 2), dilation=d, bias=False),
                         nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class ConvGRUCell(nn.Module):
    def __init__(self, i, h):
        super().__init__()
        self.h = h
        self.zr = nn.Conv2d(i + h, 2 * h, 3, padding=1)
        self.n = nn.Conv2d(i + h, h, 3, padding=1)

    def forward(self, x, h):
        if h is None:
            h = x.new_zeros(x.shape[0], self.h, *x.shape[2:])
        z, r = torch.sigmoid(self.zr(torch.cat([x, h], 1))).chunk(2, 1)
        n = torch.tanh(self.n(torch.cat([x, r * h], 1)))
        return (1 - z) * h + z * n


class MotionEncoder(nn.Module):
    """Flow (2ch, 1/2 res) + magnitude -> 1/16 res motion features."""

    def __init__(self, o=32):
        super().__init__()
        self.net = nn.Sequential(cbr(3, 16), cbr(16, 32, s=2), cbr(32, 32), cbr(32, o, s=2))

    def forward(self, f):
        f = F.avg_pool2d(f, 2)  # 1/4 res is enough for crowd-scale motion
        return self.net(torch.cat([f, f.norm(dim=1, keepdim=True)], 1))


def flow_stats(flow, m):
    """Mask-weighted physics descriptors of the flow inside the window.

    flow (B,T,2,h,w), m (B,1,h,w) -> (B,8): mean speed, speed std, |mean velocity|,
    velocity dispersion, |divergence|, |curl|, temporal speed change, direction coherence.
    """
    B, T = flow.shape[:2]
    w = m / (m.sum((2, 3), keepdim=True) + 1e-6)                   # (B,1,h,w)
    w = w[:, None]                                                  # (B,1,1,h,w)
    spd = (flow ** 2).sum(2, keepdim=True).clamp_min(1e-8).sqrt()
    mspd = (spd * w).sum((3, 4))                                    # (B,T,1)
    sspd = (((spd - mspd[..., None, None]) ** 2) * w).sum((3, 4)).clamp_min(1e-8).sqrt()
    mv = (flow * w).sum((3, 4))                                     # (B,T,2)
    disp = ((((flow - mv[..., None, None]) ** 2).sum(2, keepdim=True)) * w).sum((3, 4)).clamp_min(1e-8).sqrt()
    du_dx = flow[:, :, 0, :, 1:] - flow[:, :, 0, :, :-1]
    dv_dy = flow[:, :, 1, 1:, :] - flow[:, :, 1, :-1, :]
    dv_dx = flow[:, :, 1, :, 1:] - flow[:, :, 1, :, :-1]
    du_dy = flow[:, :, 0, 1:, :] - flow[:, :, 0, :-1, :]
    div = (du_dx[:, :, 1:, :] + dv_dy[:, :, :, 1:]).abs().unsqueeze(2)
    curl = (dv_dx[:, :, 1:, :] - du_dy[:, :, :, 1:]).abs().unsqueeze(2)
    wc = w[..., 1:, 1:]
    wc = wc / (wc.sum((3, 4), keepdim=True) + 1e-6)
    mdiv, mcurl = (div * wc).sum((3, 4)), (curl * wc).sum((3, 4))
    dt = (mspd[:, 1:] - mspd[:, :-1]).abs().mean(1) if T > 1 else mspd.new_zeros(B, 1)
    coh = (mv ** 2).sum(2, keepdim=True).clamp_min(1e-8).sqrt() / (mspd + 1e-6)
    feats = torch.cat([mspd.mean(1), sspd.mean(1), (mv ** 2).sum(2).clamp_min(1e-8).sqrt().mean(1, keepdim=True), disp.mean(1),
                       mdiv.mean(1), mcurl.mean(1), dt, coh.mean(1)], 1)
    return feats


class BharatCrowdNet(nn.Module):
    def __init__(self, pretrained=True, n_states=3, n_stats=4, ch=64):
        super().__init__()
        try:
            mb = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        except Exception as e:  # offline: fall back to random init, but say so
            print(f"[BharatCrowdNet] ImageNet weights unavailable ({e}); random init")
            mb = mobilenet_v3_small(weights=None)
        f = mb.features
        self.s4, self.s8, self.s16 = f[:2], f[2:4], f[4:9]          # 16ch/4, 24ch/8, 48ch/16
        c4, hc = ch // 2, 48
        self.lat4, self.lat8, self.lat16 = nn.Conv2d(16, c4, 1), nn.Conv2d(24, ch, 1), nn.Conv2d(48, ch, 1)
        self.top4 = nn.Conv2d(ch, c4, 1)
        self.smooth8, self.smooth4 = cbr(ch, ch), cbr(c4, c4)
        # density head (stride 4), dilations widen context for dense, perspective-varying heads
        self.density = nn.Sequential(cbr(c4, c4, d=2), cbr(c4, c4 // 2, d=3), nn.Conv2d(c4 // 2, 1, 1))
        nn.init.constant_(self.density[-1].bias, -4.0)
        # motion branch: per-pair encoder at 1/16, ConvGRU over time at 1/16
        self.motion = MotionEncoder(32)
        self.gru = ConvGRUCell(32, hc)
        # region (moving-window) head on fused features + global context
        self.fuse = cbr(ch + hc + 1, ch)
        self.ctx = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch, ch, 1), nn.Sigmoid())
        self.region = nn.Sequential(cbr(ch, 48, d=2), cbr(48, 32, d=4), nn.Conv2d(32, 1, 1))
        # behaviour head
        self.beh = nn.Sequential(nn.Linear(ch + hc + 8 + 1, 128), nn.ReLU(inplace=True), nn.Dropout(0.2),
                                 nn.Linear(128, n_states + n_stats))
        self.n_states = n_states

    def encode(self, img):
        x4 = self.s4(img)
        x8 = self.s8(x4)
        x16 = self.s16(x8)
        p8 = self.smooth8(self.lat8(x8) + F.interpolate(self.lat16(x16), size=x8.shape[2:], mode="bilinear",
                                                        align_corners=False))
        p4 = self.smooth4(self.lat4(x4) + F.interpolate(self.top4(p8), size=x4.shape[2:], mode="bilinear",
                                                        align_corners=False))
        return p4, p8

    def forward(self, img, flow, has_motion, mask_override=None):
        """img (B,3,H,W) key frame; flow (B,T-1,2,H/2,W/2); has_motion (B,)."""
        p4, p8 = self.encode(img)
        den = F.softplus(self.density(p4)) * 0.1                    # (B,1,H/4,W/4), sum = count

        B, T = flow.shape[:2]
        mf = self.motion(flow.flatten(0, 1))
        mf = mf.view(B, T, *mf.shape[1:])
        h = None
        for t in range(T):
            h = self.gru(mf[:, t], h)
        h = F.interpolate(h, size=p8.shape[2:], mode="bilinear", align_corners=False)
        hm = has_motion.view(B, 1, 1, 1)
        h = h * hm                                                   # no motion -> appearance-only region cue
        fused = self.fuse(torch.cat([p8, h, hm.expand(-1, 1, *p8.shape[2:])], 1))
        fused = fused * self.ctx(fused)
        reg = self.region(fused)                                     # (B,1,H/8,W/8) logits

        m = torch.sigmoid(reg) if mask_override is None else mask_override
        w = m / (m.sum((2, 3), keepdim=True) + 1e-6)
        pooled = torch.cat([(fused * w).sum((2, 3)), (h * w).sum((2, 3))], 1)
        m_flow = F.interpolate(m, size=flow.shape[3:], mode="bilinear", align_corners=False)
        fs = flow_stats(flow, m_flow) * has_motion.view(B, 1)
        out = self.beh(torch.cat([pooled, fs, has_motion.view(B, 1)], 1))

        m4 = F.interpolate(torch.sigmoid(reg), size=den.shape[2:], mode="bilinear", align_corners=False)
        return dict(density=den, region=reg, logits=out[:, :self.n_states], stats=out[:, self.n_states:],
                    count=den.sum((1, 2, 3)), roi_count=(den * m4).sum((1, 2, 3)))
