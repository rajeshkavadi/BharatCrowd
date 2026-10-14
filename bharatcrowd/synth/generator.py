"""Synthetic generator for Indian festival processions.

Scene model (from the reference screenshots: Rath Yatra, Ganesh visarjan, night
palanquin processions):

* An elevated, oblique camera (rooftop / tethered drone / news crane) looking at a
  ground plane, so person scale varies strongly with depth.
* A dense, mostly *stationary* spectator crowd lining and filling the road.
* A *procession*: a float (chariot / idol / palanquin) with an escort group, often
  in uniform colours, optionally ringed by a cordon, moving along a road. Its
  footprint is the "moving window" whose position is arbitrary in the image.
* Distractors that break naive "motion = procession" and "head-like blob = person"
  shortcuts: spectator sway, spectators displaced by the passing procession,
  random walkers, camera pan/zoom/jitter, hoardings with printed faces, vehicles,
  buildings, broadcaster logos and tickers, night lighting, JPEG artefacts.

All dynamics run in world metres / seconds so density (p/m^2), speed (m/s) and
crowd pressure (Helbing et al. 2007: rho * Var(v)) are physically meaningful.

Behaviour state of the moving region (index -> name), compatible with CrowdSTNet:
    0 normal     coherent advance, formation kept
    1 congested  stop-and-go, compression, spectators encroach
    2 turbulent  dense, spatially correlated surges, momentum dispersion

Class-conditional density ranges overlap on purpose: the state must be read from
motion, not inferred from headcount.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.spatial import ConvexHull, cKDTree

STATES = ("normal", "congested", "turbulent")

# group ids for annotated points
SPECTATOR, ESCORT, CORDON = 0, 1, 2

PALETTE = np.array([  # BGR clothing colours common in the footage
    (245, 245, 245), (220, 235, 240), (40, 120, 245), (40, 40, 200), (60, 160, 60),
    (160, 80, 30), (180, 105, 255), (30, 30, 30), (40, 200, 230), (120, 120, 120),
    (90, 60, 140), (200, 200, 90), (20, 90, 160)], np.float32)
UNIFORMS = np.array([(245, 245, 245), (30, 130, 250), (60, 170, 60), (40, 40, 210),
                     (150, 80, 40), (60, 110, 140)], np.float32)
FLOAT_COLS = np.array([(40, 40, 220), (30, 160, 250), (200, 80, 230), (40, 210, 240),
                       (60, 160, 60), (230, 230, 240)], np.float32)


@dataclass
class SynthConfig:
    H: int = 320
    W: int = 320
    frames: int = 8                # saved frames per sequence
    fps: float = 3.0               # saved frames per second
    steps_per_frame: int = 3       # physics sub-steps per saved frame
    warmup_s: float = 6.0          # simulated seconds before the first saved frame
    max_agents: int = 9000
    p_transition: float = 0.25     # probability of an escalation inside a clip
    p_night: float = 0.35
    p_overlay: float = 0.8
    class_params: dict = field(default_factory=lambda: {
        # v0: float speed m/s, turb: correlated-force amplitude m/s^2,
        # kp: formation stiffness 1/s, clear: spectator clearance m, rho: escort density p/m^2
        0: dict(v0=(0.5, 1.1), turb=(0.0, 0.25), kp=(0.8, 1.5), clear=(0.8, 3.0), rho=(0.8, 3.5)),
        1: dict(v0=(0.3, 0.9), turb=(0.2, 0.7), kp=(0.5, 1.0), clear=(0.0, 1.0), rho=(1.5, 4.5)),
        2: dict(v0=(0.0, 0.5), turb=(1.2, 3.0), kp=(0.2, 0.5), clear=(0.0, 0.8), rho=(2.5, 5.5)),
    })


# ----------------------------------------------------------------------------- camera
class Camera:
    """Pinhole camera at height h, tilted theta below the horizon, over plane Y=0."""

    def __init__(self, rng, H, W):
        self.H, self.W = H, W
        vfov = np.deg2rad(rng.uniform(38, 55))
        self.f0 = (H / 2) / np.tan(vfov / 2)
        lo = np.rad2deg(vfov / 2) + 18          # keeps the horizon well above the frame
        self.theta = np.deg2rad(rng.uniform(lo, 85))
        self.h = rng.uniform(20, 45)
        self.cx0, self.cy0 = W / 2, H / 2
        self.pan = rng.normal(0, 0.8, 2) if rng.random() < 0.5 else np.zeros(2)
        self.zoom = rng.normal(0, 0.005) if rng.random() < 0.3 else 0.0
        self.jit = rng.uniform(0.2, 0.8) if rng.random() < 0.4 else 0.0
        self.rng = rng
        self.set_frame(0.0)

    def set_frame(self, k):
        j = self.rng.normal(0, self.jit, 2) if self.jit else np.zeros(2)
        self.f = self.f0 * (1 + self.zoom * k)
        self.cx, self.cy = self.cx0 + self.pan[0] * k + j[0], self.cy0 + self.pan[1] * k + j[1]
        st, ct = np.sin(self.theta), np.cos(self.theta)
        self.up = np.array([0, ct, st])
        self.fwd = np.array([0, -st, ct])

    def project(self, X, Z, Y=0.0):
        """World (X lateral, Y up, Z forward) -> pixel u, v and camera depth zc."""
        vy = np.asarray(Y) - self.h
        yc = vy * self.up[1] + Z * self.up[2]
        zc = vy * self.fwd[1] + Z * self.fwd[2]
        zc = np.maximum(zc, 1e-3)
        return self.cx + self.f * X / zc, self.cy - self.f * yc / zc, zc

    def backproject(self, u, v):
        """Pixel -> ground (X, Z)."""
        du = (u - self.cx) / self.f
        dv = (self.cy - v) / self.f
        dx = du
        dy = dv * self.up[1] + self.fwd[1]
        dz = dv * self.up[2] + self.fwd[2]
        t = self.h / np.maximum(-dy, 1e-6)
        return t * dx, t * dz

    def ppm(self, X, Z):
        """Pixels per metre at a ground point (for density in p/m^2)."""
        return self.f / self.project(X, Z, 1.0)[2]


# ----------------------------------------------------------------------------- road path
class Path:
    def __init__(self, C, phi, kappa, L=160.0, ds=0.25):
        s = np.arange(-L, L + ds, ds)
        t = np.array([np.cos(phi), np.sin(phi)])
        n = np.array([-t[1], t[0]])
        self.pts = C + s[:, None] * t + (kappa * s ** 2)[:, None] * n
        seg = np.linalg.norm(np.diff(self.pts, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        self.arc = arc - arc[len(s) // 2]

    def at(self, a):
        a = np.atleast_1d(a)
        i = np.clip(np.searchsorted(self.arc, a) - 1, 0, len(self.arc) - 2)
        w = ((a - self.arc[i]) / (self.arc[i + 1] - self.arc[i]))[:, None]
        p = self.pts[i] * (1 - w) + self.pts[i + 1] * w
        tan = self.pts[i + 1] - self.pts[i]
        tan /= np.linalg.norm(tan, axis=1, keepdims=True)
        nor = np.stack([-tan[:, 1], tan[:, 0]], 1)
        return p, tan, nor


# ----------------------------------------------------------------------------- helpers
def _u(rng, rngpair):
    return rng.uniform(*rngpair)


def _value_noise(rng, n=128, octaves=3):
    out = np.zeros((n, n), np.float32)
    for o in range(octaves):
        k = 4 * 2 ** o
        out += cv2.resize(rng.standard_normal((k, k)).astype(np.float32), (n, n),
                          interpolation=cv2.INTER_CUBIC) / 2 ** o
    return out / (np.abs(out).max() + 1e-6)


def _poly_in_image(poly, H, W):
    return (poly[:, 0].max() > 0 and poly[:, 0].min() < W and poly[:, 1].max() > 0 and poly[:, 1].min() < H)


def _box_corners(cx, cz, ang, w, l, h):
    """World box (centre, heading, width, length, height) -> 8 corners (X, Y, Z)."""
    c, s = np.cos(ang), np.sin(ang)
    dx = np.array([-1, 1, 1, -1]) * l / 2
    dz = np.array([-1, -1, 1, 1]) * w / 2
    X = cx + c * dx - s * dz
    Z = cz + s * dx + c * dz
    return np.concatenate([X, X]), np.concatenate([np.zeros(4), np.full(4, h)]), np.concatenate([Z, Z])


def _random_text(rng, lo=3, hi=12):
    chars = "ABCDEFGHIJKLMNOPRSTUVWXYZ 0123456789"
    return "".join(rng.choice(list(chars), rng.integers(lo, hi + 1)))


# ----------------------------------------------------------------------------- scene
class Scene:
    def __init__(self, cfg: SynthConfig, state0: int, rng):
        self.cfg, self.rng = cfg, rng
        H, W = cfg.H, cfg.W
        self.cam = Camera(rng, H, W)
        self.state = state0
        self.night = rng.random() < cfg.p_night

        # visible ground footprint (bbox) from image corners
        uu = np.array([0, W, W, 0], float)
        vv = np.array([0, 0, H, H], float)
        gx, gz = self.cam.backproject(uu, vv)
        self.fp = (gx.min() - 3, gx.max() + 3, gz.min() - 3, gz.max() + 3)

        # road path through a random interior point -> the moving window can be anywhere
        cu, cv_ = rng.uniform(0.15, 0.85) * W, rng.uniform(0.2, 0.85) * H
        C = np.array(self.cam.backproject(cu, cv_))
        self.path = Path(C, rng.uniform(0, 2 * np.pi), rng.uniform(-0.012, 0.012))
        self.road_w = rng.uniform(8, 20)

        self._init_procession()
        self._init_static_objects()
        self._init_spectators()
        self._init_appearance()
        self.t = 0.0

    # -- procession -------------------------------------------------------------
    def _init_procession(self):
        rng, cp = self.rng, self.cfg.class_params[self.state]
        self.float_kind = rng.choice(["chariot", "idol", "palanquin"], p=[0.3, 0.45, 0.25])
        fw = {"chariot": (5, 8), "idol": (3, 5), "palanquin": (1.8, 2.6)}[self.float_kind]
        self.float_w = rng.uniform(*fw)
        self.float_l = self.float_w * rng.uniform(1.0, 1.4)
        self.float_h = {"chariot": (9, 13), "idol": (4, 7), "palanquin": (2.6, 3.4)}[self.float_kind]
        self.float_h = rng.uniform(*self.float_h)
        self.float_col = FLOAT_COLS[rng.integers(len(FLOAT_COLS))]
        self.float_col2 = FLOAT_COLS[rng.integers(len(FLOAT_COLS))]

        self.L_e = rng.uniform(10, 32)                       # formation length (m)
        self.W_e = rng.uniform(4, max(4.5, self.road_w - 1))  # formation width (m)
        self.s_f = rng.uniform(-4, 4)                        # float arc position
        self.kappa = 1.0                                     # along-track spacing factor
        self._set_params(self.state, first=True)

        rho = _u(rng, cp["rho"])
        area = np.pi * self.L_e / 2 * self.W_e / 2
        n = int(np.clip(rho * area, 15, 900))
        a, b = [], []
        while len(a) < n:  # sample ellipse, excluding the float footprint
            aa = rng.uniform(-0.75, 0.25, 4 * n) * self.L_e
            bb = rng.uniform(-0.5, 0.5, 4 * n) * self.W_e
            ac = (aa + 0.25 * self.L_e) / (0.5 * self.L_e)
            ok = (ac ** 2 + (bb / (0.5 * self.W_e)) ** 2) < 1
            ok &= ~((np.abs(aa) < self.float_l / 2 + 0.4) & (np.abs(bb) < self.float_w / 2 + 0.4))
            a.extend(aa[ok]); b.extend(bb[ok])
        self.e_a, self.e_b = np.array(a[:n]), np.array(b[:n])
        self.e_group = np.full(n, ESCORT)
        if rng.random() < 0.45:  # cordon: ring of uniformed police / volunteers
            m = int(np.clip(np.pi * (self.L_e + self.W_e) / 2 / rng.uniform(0.6, 1.0), 20, 160))
            th = np.linspace(0, 2 * np.pi, m, endpoint=False)
            self.e_a = np.concatenate([self.e_a, -0.25 * self.L_e + 0.5 * self.L_e * 1.05 * np.cos(th)])
            self.e_b = np.concatenate([self.e_b, 0.5 * self.W_e * 1.05 * np.sin(th)])
            self.e_group = np.concatenate([self.e_group, np.full(m, CORDON)])
        p, tan, nor = self.path.at(self.s_f + self.e_a)
        self.e_pos = p + nor * self.e_b[:, None]
        self.e_vel = tan * self.v0
        # turbulence field: K plane waves with independent force directions (shear + compression)
        K = 3
        self.tw_k = rng.normal(0, 1, (K, 2)); self.tw_k /= np.linalg.norm(self.tw_k, axis=1, keepdims=True)
        self.tw_k *= (2 * np.pi / rng.uniform(3, 9, K))[:, None]
        self.tw_e = rng.normal(0, 1, (K, 2)); self.tw_e /= np.linalg.norm(self.tw_e, axis=1, keepdims=True)
        self.tw_w = 2 * np.pi * rng.uniform(0.3, 1.0, K)
        self.tw_ph = rng.uniform(0, 2 * np.pi, K)

    def _set_params(self, state, first=False):
        rng, cp = self.rng, self.cfg.class_params[state]
        self.state = state
        self.v0 = _u(rng, cp["v0"])
        self.turb = _u(rng, cp["turb"])
        self.kp = _u(rng, cp["kp"])
        self.clear = _u(rng, cp["clear"])
        self.sg_T = rng.uniform(3, 7)
        self.sg_duty = rng.uniform(0.3, 0.6)
        self.sg_ph = rng.uniform(0, 1)
        self.kappa_stop = rng.uniform(0.5, 0.7)

    def float_speed(self):
        if self.state == 1:  # stop-and-go
            phase = (self.t / self.sg_T + self.sg_ph) % 1.0
            return self.v0 if phase > self.sg_duty else 0.03 * self.v0
        if self.state == 2:
            return max(0.0, self.v0 * (0.6 + 0.4 * np.sin(1.3 * self.t)))
        return self.v0

    def turb_force(self, pos):
        arg = pos @ self.tw_k.T - self.tw_w * self.t + self.tw_ph  # (N, K)
        return (np.sin(arg)[:, :, None] * self.tw_e[None]).sum(1) * self.turb / np.sqrt(len(self.tw_w))

    def region_frame(self):
        c, tan, nor = self.path.at(self.s_f - 0.25 * self.L_e)
        return c[0], tan[0], nor[0]

    # -- static scenery ---------------------------------------------------------
    def _init_static_objects(self):
        rng = self.rng
        x0, x1, z0, z1 = self.fp
        self.objects = []  # dicts: kind, corners (X,Y,Z), colour

        def off_road_point(min_d):
            for _ in range(50):
                X, Z = rng.uniform(x0, x1), rng.uniform(z0, z1)
                d = np.min(np.linalg.norm(self.path.pts[::8] - [X, Z], axis=1))
                if d > min_d:
                    return X, Z
            return None

        for _ in range(rng.integers(0, 4)):  # vehicles (ambulances, trucks, vans)
            p = off_road_point(self.road_w / 2 + 2)
            if p:
                w, l = rng.uniform(1.8, 2.5), rng.uniform(4, 8)
                col = np.array((235, 235, 235)) if rng.random() < 0.6 else PALETTE[rng.integers(len(PALETTE))]
                self.objects.append(dict(kind="vehicle", box=(p[0], p[1], rng.uniform(0, np.pi), w, l,
                                                             rng.uniform(1.8, 3.0)), col=col))
        for _ in range(rng.integers(0, 3)):  # buildings at the road side
            p = off_road_point(self.road_w / 2 + 8)
            if p:
                self.objects.append(dict(kind="building", box=(p[0], p[1], rng.uniform(0, np.pi), rng.uniform(6, 14),
                                                              rng.uniform(8, 20), rng.uniform(6, 15)),
                                         col=rng.uniform(80, 220, 3)))
        for _ in range(rng.integers(0, 3)):  # hoardings with printed faces (head-like distractors)
            p = off_road_point(self.road_w / 2 + 1)
            if p:
                self.objects.append(dict(kind="hoarding", box=(p[0], p[1], rng.uniform(-0.4, 0.4), 0.3,
                                                              rng.uniform(4, 9), rng.uniform(4, 7)),
                                         col=rng.uniform(40, 230, 3), faces=int(rng.integers(1, 5)),
                                         seed=int(rng.integers(1 << 30))))
        self.obstacles = [o["box"] for o in self.objects]

    def _inside_obstacle(self, X, Z):
        bad = np.zeros(len(X), bool)
        for (cx, cz, ang, w, l, _) in self.obstacles:
            c, s = np.cos(ang), np.sin(ang)
            dx, dz = X - cx, Z - cz
            a, b = c * dx + s * dz, -s * dx + c * dz
            bad |= (np.abs(a) < l / 2 + 0.3) & (np.abs(b) < w / 2 + 0.3)
        return bad

    # -- spectators -------------------------------------------------------------
    def _init_spectators(self):
        rng, cfg = self.rng, self.cfg
        x0, x1, z0, z1 = self.fp
        rho_off = rng.uniform(1.5, 4.0)
        road_factor = rng.uniform(0.0, 1.0)  # 0: cleared corridor (Rath Yatra), 1: packed road
        area = (x1 - x0) * (z1 - z0)
        n_cand = int(min(rho_off * area * 1.5, cfg.max_agents * 6))
        X, Z = rng.uniform(x0, x1, n_cand), rng.uniform(z0, z1, n_cand)
        u, v, _ = self.cam.project(X, Z, 1.6)
        keep = (u > -4) & (u < cfg.W + 4) & (v > -4) & (v < cfg.H + 4)
        X, Z = X[keep], Z[keep]
        # rejection by local target density (road vs. off-road)
        d_road = cKDTree(self.path.pts[::2]).query(np.stack([X, Z], 1))[0]
        on_road = d_road < self.road_w / 2
        rho = np.where(on_road, rho_off * road_factor, rho_off)
        keep = rng.random(len(X)) < rho / (rho_off * 1.5)
        X, Z = X[keep], Z[keep]
        X, Z = X[~self._inside_obstacle(X, Z)], Z[~self._inside_obstacle(X, Z)]
        # exclude the procession footprint (+ clearance)
        q = self._region_q(np.stack([X, Z], 1), self.clear)
        keep = q > 1.0
        X, Z = X[keep], Z[keep]
        budget = cfg.max_agents - len(self.e_pos)
        if len(X) > budget:
            idx = rng.choice(len(X), budget, replace=False)
            X, Z = X[idx], Z[idx]
        self.s_anchor = np.stack([X, Z], 1)
        self.s_disp = np.zeros_like(self.s_anchor)
        self.s_vel = np.zeros_like(self.s_anchor)
        n = len(X)
        self.s_walker = rng.random(n) < rng.uniform(0.01, 0.05)
        ang = rng.uniform(0, 2 * np.pi, n)
        self.s_wvel = np.stack([np.cos(ang), np.sin(ang)], 1) * rng.uniform(0.7, 1.3, n)[:, None]

    def _region_q(self, pos, clear):
        c, tan, nor = self.region_frame()
        d = pos - c
        a, b = d @ tan, d @ nor
        A, B = 0.5 * self.L_e * 1.05 + clear + 0.4, 0.5 * self.W_e * 1.05 + clear + 0.4
        return np.sqrt((a / A) ** 2 + (b / B) ** 2)

    # -- appearance -------------------------------------------------------------
    def _init_appearance(self):
        rng = self.rng
        ne, ns = len(self.e_pos), len(self.s_anchor)
        n = ne + ns
        cols = PALETTE[rng.integers(len(PALETTE), size=n)] + rng.normal(0, 18, (n, 3))
        if rng.random() < 0.65:  # uniform-wearing escort group
            ucol = UNIFORMS[rng.integers(len(UNIFORMS))]
            m = (self.e_group == ESCORT) & (rng.random(ne) < rng.uniform(0.5, 0.95))
            cols[:ne][m] = ucol + rng.normal(0, 10, (m.sum(), 3))
        ccol = UNIFORMS[rng.integers(len(UNIFORMS))]
        m = self.e_group == CORDON
        cols[:ne][m] = ccol + rng.normal(0, 8, (m.sum(), 3))
        self.cloth = np.clip(cols, 0, 255)
        self.hair = np.clip(rng.normal(25, 10, n), 5, 70)
        self.skin = np.clip(np.array((90, 130, 180)) * rng.uniform(0.55, 1.1, (n, 1)), 0, 255)
        self.hscale = np.where(rng.random(n) < 0.08, rng.uniform(0.65, 0.8, n), rng.uniform(0.92, 1.08, n))
        self.face_p = rng.random(n)
        self.flag = np.zeros(n, bool)
        self.flag[:ne] = (self.e_group == ESCORT) & (rng.random(ne) < rng.uniform(0.0, 0.06))
        self.flag_col = FLOAT_COLS[rng.integers(3, size=n)]
        self.ground_tex = _value_noise(rng)
        self.ground_col = np.array((100, 110, 120)) if rng.random() < 0.5 else np.array((90, 120, 150))
        self.road_col = rng.uniform(55, 100) * np.array((1, 1, 1.03))
        self.overlay = self._make_overlay() if rng.random() < self.cfg.p_overlay else None
        self.haze = rng.uniform(0, 0.35) if not self.night else 0.0
        self.glows = [(rng.uniform(0, self.cfg.W), rng.uniform(0, self.cfg.H), rng.uniform(10, 50),
                       FLOAT_COLS[rng.integers(len(FLOAT_COLS))]) for _ in range(rng.integers(2, 7))]
        self.night_gain = rng.uniform(0.35, 0.6)

    def _make_overlay(self):
        rng, H, W = self.rng, self.cfg.H, self.cfg.W
        ov = np.zeros((H, W, 3), np.uint8)
        alpha = np.zeros((H, W), np.float32)
        s = H / 320
        corner = rng.integers(4)
        r = int(rng.uniform(14, 24) * s)
        cx = r + 6 if corner in (0, 3) else W - r - 6
        cy = r + 6 if corner in (0, 1) else H - r - 6
        col = tuple(int(c) for c in FLOAT_COLS[rng.integers(len(FLOAT_COLS))])
        cv2.circle(ov, (cx, cy), r, col, -1, cv2.LINE_AA)
        cv2.circle(alpha, (cx, cy), r, 1.0, -1, cv2.LINE_AA)
        cv2.putText(ov, _random_text(rng, 1, 3), (cx - r // 2, cy + r // 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * s, (255, 255, 255), max(1, int(2 * s)), cv2.LINE_AA)
        if rng.random() < 0.6:  # ticker band
            y0 = int(H - rng.uniform(14, 22) * s)
            cv2.rectangle(ov, (0, y0), (W, H), (int(rng.uniform(0, 120)),) * 3, -1)
            alpha[y0:] = rng.uniform(0.6, 1.0)
            cv2.putText(ov, _random_text(rng, 10, 30), (4, H - int(5 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4 * s, (255, 255, 255), 1, cv2.LINE_AA)
        if rng.random() < 0.6:  # watermark
            x, y = int(rng.uniform(0.3, 0.7) * W), int(rng.uniform(0.2, 0.8) * H)
            tmp = np.zeros((H, W), np.uint8)
            cv2.putText(tmp, _random_text(rng, 3, 8), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, 255, 2, cv2.LINE_AA)
            ov[tmp > 0] = 230
            alpha = np.maximum(alpha, tmp.astype(np.float32) / 255 * rng.uniform(0.3, 0.6))
        return ov, alpha

    # -- dynamics ---------------------------------------------------------------
    def step(self, dt):
        rng = self.rng
        v_f = self.float_speed()
        self.s_f += v_f * dt
        k_tgt = self.kappa_stop if (self.state == 1 and v_f < 0.1) or self.state == 2 else 1.0
        self.kappa += (k_tgt - self.kappa) * min(1.0, dt / 1.5)

        # escorts: formation following + pairwise repulsion + correlated turbulence + noise
        p, tan, nor = self.path.at(self.s_f + self.e_a * self.kappa)
        target = p + nor * (self.e_b * (0.85 + 0.15 * self.kappa))[:, None]
        v_des = tan * v_f + self.kp * (target - self.e_pos)
        sp = np.linalg.norm(v_des, axis=1, keepdims=True)
        v_des *= np.minimum(1.0, 1.6 / (sp + 1e-6))
        acc = (v_des - self.e_vel) / 0.5
        pairs = cKDTree(self.e_pos).query_pairs(0.75, output_type="ndarray")
        if len(pairs):
            d = self.e_pos[pairs[:, 0]] - self.e_pos[pairs[:, 1]]
            dist = np.linalg.norm(d, axis=1, keepdims=True) + 1e-6
            f = 2.0 * np.exp((0.45 - dist) / 0.15) * d / dist
            f = np.minimum(f, 8.0)
            np.add.at(acc, pairs[:, 0], f)
            np.add.at(acc, pairs[:, 1], -f)
        acc += self.turb_force(self.e_pos) + rng.normal(0, 0.25, acc.shape)
        self.e_vel += acc * dt
        sp = np.linalg.norm(self.e_vel, axis=1, keepdims=True)
        self.e_vel *= np.minimum(1.0, 2.5 / (sp + 1e-6))
        self.e_pos += self.e_vel * dt

        # spectators: anchored springs, displaced by the passing procession, sway, spill-over
        pos = self.s_anchor + self.s_disp
        c, rt, rn = self.region_frame()
        d = pos - c
        a, b = d @ rt, d @ rn
        A, B = 0.5 * self.L_e * 1.05 + self.clear + 0.4, 0.5 * self.W_e * 1.05 + self.clear + 0.4
        q = np.sqrt((a / A) ** 2 + (b / B) ** 2) + 1e-6
        g = (a / A ** 2)[:, None] * rt + (b / B ** 2)[:, None] * rn
        g /= np.linalg.norm(g, axis=1, keepdims=True) + 1e-6
        push = (np.clip(1 - q, 0, 1) * 6.0)[:, None] * g
        acc = -1.0 * self.s_disp - 2.0 * self.s_vel + push + rng.normal(0, 0.15, pos.shape)
        if self.state == 2:
            band = np.clip((1.35 - q) / 0.35, 0, 1)[:, None]
            acc += 0.5 * band * self.turb_force(pos)
        w = self.s_walker
        self.s_vel += acc * dt
        self.s_vel[w] = self.s_wvel[w] - self.s_disp[w] * 0  # walkers: constant velocity
        self.s_disp += self.s_vel * dt
        self.t += dt

    # -- rendering --------------------------------------------------------------
    def _draw_box(self, img, depth, box, col, kind, zbuf_val=None, extra=None):
        cam = self.cam
        X, Y, Z = _box_corners(*box)
        u, v, zc = cam.project(X, Z, Y)
        pts = np.stack([u, v], 1)
        if not _poly_in_image(pts, self.cfg.H, self.cfg.W):
            return None
        hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
        col = tuple(float(c) for c in col)
        cv2.fillConvexPoly(img, np.round(hull * 8).astype(np.int32), col, cv2.LINE_AA, 3)
        top = np.stack([u[4:], v[4:]], 1)
        dark = tuple(0.8 * c for c in col)
        cv2.fillConvexPoly(img, np.round(top * 8).astype(np.int32), dark if kind != "hoarding" else col,
                           cv2.LINE_AA, 3)
        z = float(zc.min())
        mask = np.zeros(depth.shape, np.uint8)
        cv2.fillConvexPoly(mask, np.round(hull).astype(np.int32), 1)
        depth[(mask > 0) & (depth > z)] = z
        return hull, z

    def _decorate(self, img, hull, o):
        rng = np.random.default_rng(o.get("seed", 0))
        x0, y0 = hull.min(0)
        x1, y1 = hull.max(0)
        if o["kind"] == "building":
            for _ in range(12):
                x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
                cv2.rectangle(img, (int(x), int(y)), (int(x + (x1 - x0) / 12), int(y + (y1 - y0) / 10)),
                              tuple(float(c) * 0.6 for c in o["col"]), -1)
        elif o["kind"] == "hoarding":
            for k in range(o["faces"]):
                fx = x0 + (k + 0.5) * (x1 - x0) / o["faces"]
                fy = y0 + rng.uniform(0.35, 0.6) * (y1 - y0)
                r = max(1.0, (x1 - x0) / o["faces"] * rng.uniform(0.18, 0.3))
                skin = np.array((90, 130, 180)) * rng.uniform(0.6, 1.1)
                cv2.circle(img, (int(fx), int(fy - 0.5 * r)), int(r * 1.05), (20, 20, 20), -1, cv2.LINE_AA)
                cv2.ellipse(img, (int(fx), int(fy)), (int(r * 0.85), int(r)), 0, 0, 360,
                            tuple(float(c) for c in skin), -1, cv2.LINE_AA)
                cv2.ellipse(img, (int(fx), int(fy + 2.2 * r)), (int(1.6 * r), int(1.0 * r)), 0, 180, 360,
                            tuple(float(c) for c in rng.uniform(30, 240, 3)), -1, cv2.LINE_AA)
        elif o["kind"] == "vehicle":
            cv2.line(img, (int(x0), int((y0 + y1) / 2)), (int(x1), int((y0 + y1) / 2)), (40, 40, 200),
                     max(1, int((y1 - y0) / 10)))

    def _float_poly(self):
        p, tan, _ = self.path.at(self.s_f)
        return (p[0, 0], p[0, 1], np.arctan2(tan[0, 1], tan[0, 0]), self.float_w, self.float_l, self.float_h)

    def render(self, frame_idx):
        cfg, cam, rng = self.cfg, self.cam, self.rng
        H, W = cfg.H, cfg.W
        cam.set_frame(frame_idx)
        # ground, textured in world coordinates so it stays consistent under camera motion
        uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        gx, gz = cam.backproject(uu, vv)
        n = self.ground_tex.shape[0]
        mx = ((gx / 2.0) % n).astype(np.float32)
        mz = ((gz / 2.0) % n).astype(np.float32)
        tex = cv2.remap(self.ground_tex, mx, mz, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        d_road = cKDTree(self.path.pts[::4]).query(np.stack([gx.ravel(), gz.ravel()], 1))[0].reshape(H, W)
        on = (d_road < self.road_w / 2)[..., None]
        img = np.where(on, self.road_col, self.ground_col) * (1 + 0.12 * tex[..., None])
        img = img.astype(np.float32)
        depth = np.full((H, W), np.inf, np.float32)

        # drawables sorted far -> near (painter's algorithm)
        draw = []
        for o in self.objects:
            X, Y, Z = _box_corners(*o["box"])
            draw.append((float(cam.project(X, Z, Y)[2].max()), "obj", o))
        fb = self._float_poly()
        X, Y, Z = _box_corners(*fb)
        draw.append((float(cam.project(X, Z, Y)[2].max()), "float", fb))

        pos = np.concatenate([self.e_pos, self.s_anchor + self.s_disp])
        hs = self.hscale
        uh, vh, zh = cam.project(pos[:, 0], pos[:, 1], 1.55 * hs)
        us, vs, _ = cam.project(pos[:, 0], pos[:, 1], 1.35 * hs)
        uhip, vhip, _ = cam.project(pos[:, 0], pos[:, 1], 0.9 * hs)
        sc = cam.f / zh
        vis = (uh > -10) & (uh < W + 10) & (vh > -10) & (vh < H + 10)
        order = np.argsort(-zh)
        order = order[vis[order]]
        objs = sorted(draw, key=lambda d: -d[0])

        S = 8  # sub-pixel shift (3 bits)
        oi = 0
        for i in order:
            while oi < len(objs) and objs[oi][0] > zh[i]:
                self._render_obj(img, depth, objs[oi])
                oi += 1
            r = max(0.35, 0.11 * sc[i] * hs[i])
            bw = max(0.5, 0.23 * sc[i] * hs[i])
            bh = max(0.5, abs(vhip[i] - vs[i]) / 2 + 0.1 * sc[i])
            cx, cy = (us[i] + uhip[i]) / 2, (vs[i] + vhip[i]) / 2
            cv2.ellipse(img, (int(cx * S), int(cy * S)), (int(bw * S), int(bh * S)), 0, 0, 360,
                        tuple(float(c) for c in self.cloth[i]), -1, cv2.LINE_AA, 3)
            hc = (int(uh[i] * S), int(vh[i] * S))
            cv2.circle(img, hc, int(r * S), (float(self.hair[i]),) * 3, -1, cv2.LINE_AA, 3)
            if self.face_p[i] < 0.55 and r > 1.2:
                cv2.ellipse(img, (hc[0], hc[1] + int(0.35 * r * S)), (int(0.6 * r * S), int(0.55 * r * S)), 0, 0,
                            360, tuple(float(c) for c in self.skin[i]), -1, cv2.LINE_AA, 3)
        while oi < len(objs):
            self._render_obj(img, depth, objs[oi])
            oi += 1
        for i in np.where(self.flag[: len(self.e_pos)])[0]:  # flags above escorts
            u0, v0, _ = cam.project(pos[i, 0], pos[i, 1], 1.4)
            u1, v1, _ = cam.project(pos[i, 0], pos[i, 1], 3.4)
            w = 0.9 * sc[i]
            cv2.line(img, (int(u0), int(v0)), (int(u1), int(v1)), (40, 60, 90), 1, cv2.LINE_AA)
            tri = np.array([[u1, v1], [u1 + w, v1 + 0.3 * w], [u1, v1 + 0.6 * w]], np.float32)
            cv2.fillConvexPoly(img, np.round(tri * S).astype(np.int32), tuple(float(c) for c in self.flag_col[i]),
                               cv2.LINE_AA, 3)

        # photometrics: haze, night, glows, blur, noise, overlays
        if self.haze > 0:
            zc_row = cam.project(gx, gz)[2]
            a = np.clip((zc_row - zc_row.min()) / (zc_row.max() - zc_row.min() + 1e-6), 0, 1)[..., None] * self.haze
            img = img * (1 - a) + 200 * a
        if self.night:
            img = img * self.night_gain * np.array((0.85, 0.95, 1.1))
            for (gxp, gyp, gr, gcol) in self.glows:
                d2 = (uu - gxp) ** 2 + (vv - gyp) ** 2
                img += np.exp(-d2 / (2 * gr ** 2))[..., None] * gcol * 0.35
        if rng.random() < 0.3:
            img = cv2.GaussianBlur(img, (3, 3), 0)
        img += rng.normal(0, rng.uniform(1, 6 if not self.night else 10), img.shape)
        if self.overlay is not None:
            ov, al = self.overlay
            img = img * (1 - al[..., None]) + ov * al[..., None]
        img = np.clip(img, 0, 255).astype(np.uint8)

        # annotations: visible heads (people hidden behind objects are not counted)
        ui = np.clip(np.round(uh).astype(int), 0, W - 1)
        vi = np.clip(np.round(vh).astype(int), 0, H - 1)
        inside = (uh >= 0) & (uh < W) & (vh >= 0) & (vh < H)
        unocc = depth[vi, ui] > zh
        keep = inside & unocc
        groups = np.concatenate([self.e_group, np.full(len(self.s_anchor), SPECTATOR)])
        return img, dict(pts=np.stack([uh, vh], 1)[keep].astype(np.float32), group=groups[keep].astype(np.int8),
                         ppm=sc[keep].astype(np.float32), idx=np.where(keep)[0],
                         mask=self._region_mask(uh, vh, us, uhip, vhip, sc))

    def _render_obj(self, img, depth, item):
        _, kind, o = item
        if kind == "float":
            res = self._draw_box(img, depth, o, self.float_col, "float")
            if res is not None:
                hull, _ = res
                x0, y0 = hull.min(0)
                x1, y1 = hull.max(0)
                for k in range(1, 5):  # decorative bands
                    y = y0 + k * (y1 - y0) / 5
                    cv2.line(img, (int(x0), int(y)), (int(x1), int(y)), tuple(float(c) for c in self.float_col2),
                             max(1, int((y1 - y0) / 25)), cv2.LINE_AA)
                if self.float_kind == "idol":
                    cv2.ellipse(img, (int((x0 + x1) / 2), int(y0 + (y1 - y0) * 0.25)),
                                (max(1, int((x1 - x0) * 0.3)), max(1, int((y1 - y0) * 0.22))), 0, 0, 360,
                                (60, 150, 240), -1, cv2.LINE_AA)
                elif self.float_kind == "palanquin":
                    tri = np.array([[x0, y0 + 0.5 * (y1 - y0)], [x1, y0 + 0.5 * (y1 - y0)], [(x0 + x1) / 2, y0]])
                    cv2.fillConvexPoly(img, np.round(tri).astype(np.int32), tuple(float(c) for c in self.float_col2))
        else:
            res = self._draw_box(img, depth, o["box"], o["col"], o["kind"])
            if res is not None:
                self._decorate(img, res[0], o)

    def _region_mask(self, uh, vh, us, uhip, vhip, sc):
        """Moving-window mask: convex hull of procession bodies + float footprint (image space)."""
        ne = len(self.e_pos)
        pts = [np.stack([uh[:ne], vh[:ne]], 1), np.stack([us[:ne], vhip[:ne]], 1)]
        r = 0.6 * sc[:ne]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            pts.append(np.stack([uh[:ne] + dx * r, vh[:ne] + dy * r], 1))
        X, Y, Z = _box_corners(*self._float_poly())
        u, v, _ = self.cam.project(X, Z, Y)
        pts.append(np.stack([u, v], 1))
        P = np.concatenate(pts).astype(np.float32)
        mask = np.zeros((self.cfg.H, self.cfg.W), np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(P).reshape(-1, 2).round().astype(np.int32), 1)
        return mask

    def region_stats(self, prev_pos, dt):
        """Physical ground truth of the moving region (escorts + cordon)."""
        v = (self.e_pos - prev_pos) / dt
        vm = v.mean(0)
        disp = np.sqrt(((v - vm) ** 2).sum(1).mean())
        try:
            hull = ConvexHull(self.e_pos)
            area = hull.volume + hull.area * 0.4 + np.pi * 0.16
        except Exception:
            area = np.pi * self.L_e * self.W_e / 4
        rho = len(self.e_pos) / area
        return np.array([rho, np.linalg.norm(vm), disp, rho * disp ** 2], np.float32)


# ----------------------------------------------------------------------------- API
def generate_sequence(cfg: SynthConfig, state: int, seed: int):
    """Simulate and render one clip.

    Returns frames (T,H,W,3) uint8 BGR and an annotation dict with per-frame head
    points, groups, region masks, oriented boxes, behaviour states and physics stats.
    """
    rng = np.random.default_rng(seed)
    states = np.full(cfg.frames, state, np.int64)
    if state < 2 and rng.random() < cfg.p_transition:
        k = int(rng.integers(1, max(2, cfg.frames - 3)))
        states[k:] = state + 1
    sc = Scene(cfg, int(states[0]), rng)
    dt = 1.0 / (cfg.fps * cfg.steps_per_frame)
    for _ in range(int(cfg.warmup_s / dt)):
        sc.step(dt)
    frames, ann = [], dict(pts=[], group=[], ppm=[], mask=[], box=[], state=states, stats=[])
    for k in range(cfg.frames):
        if states[k] != sc.state:
            sc._set_params(int(states[k]))
        prev = sc.e_pos.copy()
        for _ in range(cfg.steps_per_frame):
            sc.step(dt)
        img, a = sc.render(k)
        frames.append(img)
        ys, xs = np.nonzero(a["mask"])
        box = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32)) if len(xs) else ((0, 0), (0, 0), 0)
        ann["pts"].append(a["pts"]); ann["group"].append(a["group"]); ann["ppm"].append(a["ppm"])
        ann["mask"].append(a["mask"])
        ann["box"].append(np.array([*box[0], *box[1], box[2]], np.float32))
        ann["stats"].append(sc.region_stats(prev, 1.0 / cfg.fps))
    ann["mask"] = np.stack(ann["mask"])
    ann["box"] = np.stack(ann["box"])
    ann["stats"] = np.stack(ann["stats"])
    ann["meta"] = dict(night=bool(sc.night), float_kind=str(sc.float_kind), cam_h=float(sc.cam.h),
                       cam_tilt_deg=float(np.rad2deg(sc.cam.theta)), n_escort=int(len(sc.e_pos)),
                       n_spectator=int(len(sc.s_anchor)))
    return np.stack(frames), ann
