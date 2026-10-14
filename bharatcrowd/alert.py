"""Alert state machine over per-window model outputs.

Levels: 0 GREEN, 1 AMBER (pre-congestion / watch), 2 RED (intervene).

Evidence per step:
  * behaviour posterior of the moving window (EMA-smoothed),
  * window density in persons/m^2 -- only when the camera is calibrated
    (pixels-per-metre); otherwise density triggers are disabled, not guessed,
  * relative growth of the window count over a horizon (works uncalibrated).

Escalation needs `k_up` consecutive supporting steps, de-escalation `k_down`
(hysteresis against flicker). The thresholds below are literature-anchored
placeholders (Fruin LoS; Still 2014 ~4-5 p/m^2 for moving crowds becoming
critical; Helbing et al. 2007 crowd turbulence) and MUST be re-calibrated on
annotated real footage before any operational use.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

LEVELS = ("GREEN", "AMBER", "RED")


@dataclass
class AlertConfig:
    ema: float = 0.6
    p_amber: float = 0.55      # P(congested)+P(turbulent)
    p_red: float = 0.6         # P(turbulent)
    rho_amber: float = 4.0     # p/m^2 (calibrated cameras only)
    rho_red: float = 5.5
    growth_amber: float = 0.25  # +25% window count over the horizon
    horizon: int = 10
    k_up: int = 3
    k_down: int = 6


class AlertEngine:
    def __init__(self, cfg: AlertConfig = AlertConfig()):
        self.cfg = cfg
        self.p = None
        self.level = 0
        self.up = self.down = 0
        self.counts = deque(maxlen=cfg.horizon)

    def update(self, probs, roi_count, window_area_px=None, ppm=None, has_motion=True):
        """probs: behaviour posterior (3,) or None; ppm: pixels per metre at the window (scalar)."""
        c = self.cfg
        reasons = []
        target = 0
        if has_motion and probs is not None:
            probs = np.asarray(probs, float)
            self.p = probs if self.p is None else c.ema * self.p + (1 - c.ema) * probs
            if self.p[2] >= c.p_red:
                target, _ = 2, reasons.append(f"P(turbulent)={self.p[2]:.2f}")
            elif self.p[1] + self.p[2] >= c.p_amber:
                target, _ = 1, reasons.append(f"P(cong+turb)={self.p[1] + self.p[2]:.2f}")
        rho = None
        if ppm and window_area_px:
            rho = roi_count / (window_area_px / ppm ** 2)
            if rho >= c.rho_red:
                target = max(target, 2); reasons.append(f"density={rho:.1f}/m2")
            elif rho >= c.rho_amber:
                target = max(target, 1); reasons.append(f"density={rho:.1f}/m2")
        self.counts.append(float(roi_count))
        if len(self.counts) == self.counts.maxlen and self.counts[0] > 20:
            g = self.counts[-1] / self.counts[0] - 1
            if g >= c.growth_amber:
                target = max(target, 1); reasons.append(f"count growth={g:+.0%}")
        if target > self.level:
            self.up += 1; self.down = 0
            if self.up >= c.k_up:
                self.level, self.up = target, 0
        elif target < self.level:
            self.down += 1; self.up = 0
            if self.down >= c.k_down:
                self.level, self.down = self.level - 1, 0
        else:
            self.up = self.down = 0
        return dict(level=self.level, name=LEVELS[self.level], target=target, density=rho,
                    probs=None if self.p is None else self.p.tolist(), reasons=reasons)
