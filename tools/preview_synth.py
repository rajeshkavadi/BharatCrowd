"""Render a few synthetic clips into a montage (key frame + mask contour + head points)."""
import argparse, time, sys, os
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bharatcrowd.synth import SynthConfig, generate_sequence, STATES

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--size", type=int, default=320)
ap.add_argument("--points", action="store_true")
a = ap.parse_args()
cfg = SynthConfig(H=a.size, W=a.size)
tiles = []
for i in range(a.n):
    t0 = time.time()
    fr, an = generate_sequence(cfg, i % 3, seed=1000 + i)
    k = cfg.frames - 1
    img = fr[k].copy()
    cnts, _ = cv2.findContours(an["mask"][k], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, (0, 255, 255), 1)
    if a.points:
        for (x, y), g in zip(an["pts"][k], an["group"][k]):
            img[int(y), int(x)] = (0, 0, 255) if g else (255, 0, 0)
    inroi = an["mask"][k][an["pts"][k][:, 1].astype(int), an["pts"][k][:, 0].astype(int)].sum()
    st = an["stats"][k]
    print(f"seq{i} state={STATES[an['state'][k]]} states={an['state'].tolist()} count={len(an['pts'][k])} roi={inroi} "
          f"rho={st[0]:.2f} speed={st[1]:.2f} disp={st[2]:.2f} P={st[3]:.3f} {an['meta']} {time.time()-t0:.2f}s")
    cv2.putText(img, f"{STATES[an['state'][k]]} N={len(an['pts'][k])} roi={inroi}", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    tiles.append(img)
cols = 3
rows = [np.concatenate(tiles[r * cols:(r + 1) * cols], 1) for r in range(len(tiles) // cols)]
cv2.imwrite(a.out, np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 80])
