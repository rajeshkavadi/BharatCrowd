"""Convert hand-drawn yellow ROI strokes into (clean image, region mask, sample json).

The stroke is burned into the pixels; training on it as-is lets the network learn
"find the yellow curve". So we (1) segment the stroke, (2) take the convex hull of
its pixels as the moving-window mask (strokes are often open at the frame border),
(3) inpaint the stroke away. Inspect the overlays: yellow clothing, marigolds and
signboards can be picked up; tune --min_len / HSV bounds when that happens.
"""
import argparse, glob, json, os
import cv2, numpy as np


def extract(img, h_lo=18, h_hi=38, s_min=110, v_min=140, min_len=0.25):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yel = cv2.inRange(hsv, (h_lo, s_min, v_min), (h_hi, 255, 255))
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(yel, np.ones((3, 3), np.uint8)), 8)
    H, W = yel.shape
    best, score = None, 0
    for i in range(1, n):
        x, y, w, h, area = st[i]
        extent = max(w, h) / max(H, W)
        fill = area / max(1, w * h)  # strokes are thin: low fill of their bbox
        if extent >= min_len and fill < 0.25 and extent / (fill + 1e-3) > score:
            best, score = i, extent / (fill + 1e-3)
    if best is None:
        return None, None
    stroke = ((lab == best) & (yel > 0)).astype(np.uint8)
    ys, xs = np.nonzero(stroke)
    hull = cv2.convexHull(np.stack([xs, ys], 1).astype(np.int32))
    mask = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return stroke, mask


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="folder with annotated screenshots")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(a.inp, "*.png")) + glob.glob(os.path.join(a.inp, "*.jp*g"))):
        img = cv2.imread(p)
        stroke, mask = extract(img)
        name = os.path.splitext(os.path.basename(p))[0].replace(" ", "_")
        if stroke is None:
            print("no stroke found:", p)
            continue
        clean = cv2.inpaint(img, cv2.dilate(stroke, np.ones((5, 5), np.uint8)), 5, cv2.INPAINT_TELEA)
        cv2.imwrite(os.path.join(a.out, name + ".jpg"), clean, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(a.out, name + "_mask.png"), mask)
        json.dump({"image": name + ".jpg", "roi_mask": name + "_mask.png"},
                  open(os.path.join(a.out, name + ".json"), "w"))
        ov = clean.copy()
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cnts, -1, (255, 0, 255), 3)
        cv2.imwrite(os.path.join(a.out, name + "_overlay.jpg"), ov)
        print(f"{name}: mask covers {mask.mean() / 255:.1%} of frame")
