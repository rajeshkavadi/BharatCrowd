# BharatCrowd — moving-window crowd counting, behaviour estimation and alerts for Indian festival processions

Indian festival footage (Rath Yatra, Ganesh visarjan, Muharram, night palanquin
processions) has a structure that existing benchmarks such as DroneCrowd and
ShanghaiTech don't capture. A **procession moves continuously through a much denser,
mostly stationary crowd**: a float (chariot, idol or palanquin) with an escort, often
in uniform colours and often ringed by a police or volunteer cordon. The risk sits
in that moving window, and the window can be anywhere in the frame.

This repo contains:

1. `bharatcrowd/synth/` — a physics-based **synthetic procession generator**
   (perspective camera, world-metre dynamics, labelled behaviour states).
2. `bharatcrowd/model.py` — **BharatCrowdNet** (0.51 M params), which segments the
   moving window and counts and classifies behaviour *inside it*.
3. `bharatcrowd/alert.py` — a hysteresis **alert state machine** (GREEN / AMBER / RED).
4. Tooling for real data: `tools/yellow_roi.py` turns the hand-drawn yellow-circle
   screenshots into masks and removes the stroke from the pixels. `finetune_real.py`
   trains on partial labels.

Reference: the CrowdSTNet repo (same author), where a similar exercise was done on DroneCrowd.

---

## 1. What the reference screenshots imply (and why the design looks like this)

| Observation in the 4 screenshots | Consequence |
|---|---|
| Elevated **oblique** camera (rooftop or tilted drone), not nadir | Strong perspective. Heads range from ~1 px (far) to ~15 px (near). The generator uses a ground-plane pinhole camera, and density targets are count-preserving. |
| The procession is **not the densest region**. At Puri the spectators are far denser than the rope-pullers. | The window can't be found by density. It needs motion **and** appearance cues (float, flags, uniform escorts, cordon). |
| In a still frame the window looks almost the same as the surrounding crowd (Lalbaug, night palanquin) | From still images the behaviour state is **unobservable**. The model refuses to output behaviour when `has_motion=0`. |
| Spectators near the procession move as they make way. There are also vendors and walkers. | "Moving pixels = procession" is wrong. The generator includes displaced spectators, random walkers and camera motion as distractors. |
| News overlays (V6, NDTV logos, tickers, watermarks), hoardings with **printed faces** | Head-like false positives. The generator renders hoardings with faces, and these are never counted. |
| The yellow annotation strokes are **burned into the pixels** | Training on them as-is teaches the network to find yellow curves. `tools/yellow_roi.py` extracts the mask and inpaints the stroke. |

## 2. Synthetic generator

```
python tools/preview_synth.py --out preview.jpg --n 6          # visual check
python tools/generate_synth.py --out data/synth                 # 1500/150/300 clips, 320x320, 8 frames @ 3 fps
python tools/cache_flow.py data/synth                           # optional, speeds up training
```

**Camera.** A pinhole camera at height 20–45 m, tilted 37–85° below the horizon,
with a 38–55° vertical field of view. Optional pan, zoom and jitter, which the flow
pipeline must compensate.

**Scene.** A curved road through a random interior point, so the window can be
anywhere. Ground texture is anchored in world coordinates. Vehicles, buildings and
hoardings with printed faces are depth-sorted (painter's algorithm), and occluded
heads are removed from the ground truth. Broadcaster logos, tickers and watermarks
are drawn on top. Night lighting, haze, blur, noise, and JPEG compression at
quality 50–95.

**Agents** (all dynamics in metres and seconds):
* *Escort / cordon*: formation-following along the road, plus pairwise exponential
  repulsion, plus a **spatially correlated turbulence field** (sum of plane waves
  with independent force directions, giving shear and compression), plus noise.
* *Spectators*: anchored damped springs, pushed aside by the passing window (an
  elliptical clearance zone), with sway noise and 1–5% random walkers. In the
  turbulent state, spectators near the window also get spill-over forcing.

**Behaviour states** (CrowdSTNet-compatible indices):

| idx | state | physics |
|---|---|---|
| 0 | normal | float 0.5–1.1 m/s, formation kept, weak noise |
| 1 | congested | stop-and-go (period 3–7 s), along-track compression, spectators encroach |
| 2 | turbulent | float ≤ 0.5 m/s, correlated surges at 1.2–3 m/s², weak formation, spill-over |

25% of normal and congested clips **escalate mid-clip** (0→1 or 1→2). Per-frame
labels are stored, and the clip label is the key-frame state.

**Labels per frame:** head points (spectator / escort / cordon), the moving-window
mask (convex hull of procession bodies plus the float), an oriented box,
pixels-per-metre at each head, the behaviour state, and the window's physical
statistics: density (p/m²), mean speed, velocity dispersion, and pressure
ρ·Var(v) (Helbing 2007).

**Shortcut control.** Class-conditional escort density ranges overlap on purpose
(0.8–3.5 / 1.5–4.5 / 2.5–5.5 p/m²). Measured on the generated data: ground-truth
window density alone predicts the state **59.7%** of the time (chance 33%). All four
ground-truth physics statistics together reach **89.7%** (`eval_synth.py`
baselines). Some density–state correlation is physically real, since turbulence
requires high density. The model has to beat the density-only number to show it
uses motion.

## 3. BharatCrowdNet

```
key frame (RGB) ─► MobileNetV3-S (ImageNet) ─► FPN-lite ─┬─► p4 ─► dilated density head ─► D  (stride 4, ΣD = count)
                                                         └─► p8 ─┐
flows (T-1, ego-motion removed) ─► MotionEncoder ─► ConvGRU ─────┼─► fuse(+has_motion) ─► SE ctx ─► region head ─► M (stride 8)
                                                                 │
             mask-pooled fused features + mask-pooled GRU state + flow physics stats (8)
                                         ─► MLP ─► behaviour logits (3) + window stats (4)
window count = Σ D · up(M)
```

Design decisions and the reasoning behind them:

* **Factorised window counting** (`Σ D·M`) instead of regressing a count for an
  unknown region. Counting stays shared across the whole image, the window only
  selects, and the oracle-mask evaluation separates localisation error from
  counting error.
* **Counting is appearance-only.** Stationary spectators must be counted too, and
  the density branch can then be fine-tuned on image-only dense datasets
  (JHU-Crowd++, which has many Indian festival images; ShanghaiTech-A; UCF-QNRF)
  without video.
* **Ego-motion compensation.** A robust trimmed-least-squares affine fit on
  Farneback flow, then subtracted. This assumes the static crowd dominates the
  frame, which holds in this footage.
* **Modality dropout** (p=0.25) teaches the region head appearance cues for still
  images. Behaviour loss is disabled on those samples, so the model can't learn an
  appearance shortcut for behaviour.
* **Mask-weighted pooling** for behaviour, with teacher forcing (ground-truth mask
  at p=0.5) during training. Flow physics descriptors (speed, dispersion,
  divergence, curl, temporal change, coherence) are pooled inside the window, which
  keeps the classifier tied to measurable quantities.

## 4. Results on synthetic data

Setup: 1500 / 150 / 300 clips (train / val / test, disjoint seeds, balanced classes),
320×320, 8 frames at 3 fps. 14 epochs on a 4-core CPU (~3 min per epoch, no GPU).
The checkpoint was selected on validation (epoch 9). All numbers below are on the
**test** set (`eval_synth.py`), single seed.

| Metric (test, n=300) | Full model | Motion removed at test | Oracle window mask |
|---|---|---|---|
| Total count MAE / RMSE (mean GT 2721) | 276 / 381 (≈10%) | 276 / 381 | 276 / 381 |
| Window count MAE / RMSE (mean GT 324) | 69.8 / 95.9 | 72.0 / 100.8 | 65.8 (oracle) |
| Window count mean relative error | 0.60 | 0.72 | — |
| Window mIoU | **0.865** | 0.821 | — |
| Behaviour accuracy / macro-F1 | **0.927 / 0.924** | 0.553 / 0.519 | 0.917 / 0.913 |
| Baseline: GT window density only → state | 0.590 | | |
| Baseline: GT physics (ρ, speed, dispersion, pressure) → state | 0.897 | | |

Confusion matrix (full model; rows are true normal / congested / turbulent):
`[[73,6,1],[7,90,3],[0,5,115]]`. Most errors are normal↔congested; turbulent is
rarely missed (5 of 120).

How to read this:
* **Behaviour is motion-driven, not a density shortcut.** Removing motion at test
  time drops accuracy from 92.7% to 55.3%, close to the 59% a density-only
  classifier reaches. The full model beats a classifier given the *ground-truth*
  physics summaries (89.7%), so it is using spatial motion structure beyond
  4 scalar statistics.
* **Window localisation transfers to stills on synthetic data** (mIoU 0.82 without
  motion), because of the appearance cues learned under modality dropout.
* **Counting is the weak point.** The oracle mask barely improves window MAE
  (69.8 → 65.8), so the error is in density estimation, not in localisation. Heads
  are 1–3 px at 320 px. The fix is higher input resolution (≥640 px, or tiled
  inference) and/or density pre-training on JHU-Crowd++, not a better window head.
  The mean relative error is inflated by clips with small windows (tens of people),
  so read MAE alongside it.
* Caveats: single seed, no confidence intervals, CPU-budget training. Before
  publication, run ≥3 seeds and bootstrap the test set for CIs.

**Zero-shot on the 4 real screenshots** (stills, so behaviour is not predicted):
window IoU against the yellow-stroke masks is **0.09 / 0.00 / 0.02 / 0.30** (mean
0.10), compared with 0.82 on synthetic stills. The window head fires on hoardings,
lights and dense spectator patches, and misses the Lalbaug procession entirely.
Predicted totals (4.5k, 8.7k, 0.8k, 0.7k) can't be scored without head
annotations; the last two are visibly undercounted. **The synthetic-to-real
appearance gap is large, and nothing here shows real-world efficacy yet**; see §5
and §8.

## 5. Real data: what exists and what it can and can't validate

`tools/yellow_roi.py --inp <screenshots> --out data/real_roi` handles the four
shared screenshots: it extracts the stroke, takes its convex hull as the window
mask (strokes are often open at the frame border), and inpaints the stroke. The
extracted masks cover 19–34% of each frame and match the drawn strokes on all four
images.

**Fine-tuning on this data is not yet possible in any meaningful sense.**
* 4 images with window masks and **no head points** → no counting supervision and
  no counting evaluation.
* Still images → no motion → behaviour can't be trained or evaluated.
* 4 samples can't validate anything. Any "fine-tuned" number would be overfitting.

What a defensible real-data stage needs, in priority order:
1. **Short video clips** (5–10 s, ≥3 fps) of processions, from the same public
   sources, with the yellow window drawn on the key frame. This enables window
   segmentation and motion.
2. **Head points** on key frames (CVAT, LabelMe points). Inside the window is
   enough if full-frame annotation is too costly; the loss supports partial masks.
   Alternatively, pre-train the density branch on JHU-Crowd++ first.
3. **Behaviour labels** on clips by ≥2 annotators, with agreement reported
   (Cohen's κ). Real "turbulent" footage is rare, so expect heavy class imbalance
   and plan for it.
4. A split **by event or location**, not by frame. Frames from the same video are
   near-duplicates and leak across splits.

`finetune_real.py` already supports all of this through the `RealFrames` json
format (optional `points`, `roi_mask`, `state`, `frames`), synthetic replay
(`--synth`), and frozen BatchNorm and motion branch for small sets.

## 6. Alert mechanism

`AlertEngine` (consumed by `infer.py`) escalates on EMA-smoothed P(congested or
turbulent) / P(turbulent), on window density when the camera is calibrated
(`--ppm`), and on window-count growth over a horizon. Escalation needs 3
consecutive supporting steps and de-escalation needs 6. **The thresholds are
literature-anchored placeholders** (Fruin LoS, Still 2014, Helbing 2007) and have
not been validated. Without calibration, density in p/m² is not computed at all
rather than guessed. Operational use needs: calibration per camera, threshold
fitting on annotated incident and near-miss footage, and a false-alarm-rate target
agreed with the authorities.

## 7. Usage

```
python train_synth.py --data data/synth --out runs/synth --epochs 14
python eval_synth.py  --data data/synth --ckpt runs/synth/best.pt
python tools/yellow_roi.py --inp screenshots/ --out data/real_roi
python infer.py --ckpt runs/synth/best.pt --images data/real_roi --out runs/infer_real
python infer.py --ckpt runs/synth/best.pt --video procession.mp4 --out runs/infer_video [--ppm 12]
python finetune_real.py --ckpt runs/synth/best.pt --train data/real_clips --val data/real_val \
       --synth data/synth --freeze_bn
```

## 8. Known limitations (be sceptical of the synthetic numbers)

* The rendering is stylised (ellipse bodies, disc heads). The appearance domain gap
  to real footage is large, and the zero-shot real-image results in §4 show it.
  Next step: texture or sprite-based people, or diffusion-based image-to-image
  refinement of rendered frames with labels preserved.
* Behaviour states are *defined by the generator's physics*. High synthetic
  accuracy shows the model can read those dynamics. It does not show that the
  definitions match how real crowd crushes develop.
* The camera is a single static pose per clip with mild pan and zoom. Real news
  footage has cuts, fast pans and zooms, and cuts break the flow.
* Flow is computed at 160×160 from 320 px frames. Far-field motion (heads under
  2 px) is effectively invisible.
* Turbulence is a prescribed force field, not emergent from contact forces at high
  density (as in Helbing–Johansson). It is a phenomenological proxy.
