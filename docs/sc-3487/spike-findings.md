# sc-3487 — DWPose → Rust path-selection spike (GO: `ort` + CoreML)

**Decision: GO on the `ort` (onnxruntime) + CoreML path.** It reproduces the shipped
Python rtmlib detector at sub-pixel parity and matches its latency. The fallbacks the
story listed (a different Rust onnx runtime, or an MLX pose model) are **not needed**.

Epic 3482 (Python Eradication). The DWPose "create from photo" flow + InstantID pose
conditioning must keep working on a Python-free Mac → the detector (rtmlib RTMW
whole-body on onnxruntime/CoreML) is ported to Rust.

## What was validated

Ported rtmlib's **performance preset** end-to-end into Rust and compared raw
COCO-WholeBody-133 keypoints + SimCC scores against the Python reference, on the
**exact same cached weights** production pins:

- detector `yolox_m_8xb8-300e_humanart-c2c7a14a.onnx` (embedded-NMS export → `dets`/`labels`)
- pose `rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx` (SimCC, split 2.0)

Faithful port of the rtmlib math: YOLOX letterbox (ratio=min(640/h,640/w), pad 114,
**no BGR→RGB swap, no mean/std**), the `shape[-1]==5` embedded-NMS branch (boxes/ratio,
keep score>0.3); RTMPose xyxy→center/scale (padding 1.25) → aspect-fix to 288/384 →
top-down affine crop (border 0) → `(px−mean)/std` on **BGR** channels → SimCC argmax
decode → rescale to original px. cv2 resampling conventions matched (half-pixel for
resize; direct index for warpAffine).

## Results (3× 1024² photos, 1 person each)

**A) Algorithm-port parity — rust.cpu vs python.cpu** (same ORT numerics, so any delta
is the Rust pre/post port):

| image  | kp mean err | p95   | max    | score \|Δ\| mean |
|--------|-------------|-------|--------|------------------|
| img_00 | 0.42 px     | 2.07  | 2.13   | 0.005            |
| img_01 | 0.16 px     | 1.65  | 1.67   | 0.011            |
| img_02 | 0.11 px     | 0.28  | 1.02   | 0.032            |

Sub-pixel mean error (≤0.03% of the image diagonal). The port is numerically faithful.

**B) Production-path parity — rust.coreml vs python.coreml** (the real A/B: Rust+CoreML
replacing Python+CoreML):

| image  | kp mean err | p95   | max          |
|--------|-------------|-------|--------------|
| img_00 | 1.05 px     | 2.72  | 4.36 @kp101  |
| img_01 | 0.22 px     | 1.66  | 3.71 @kp20   |
| img_02 | 0.12 px     | 0.38  | 0.64 @kp114  |

**C) The 217px "outlier" is CoreML EP nondeterminism, not a port bug.** CPU-vs-CoreML
*within Python* flips img_00/kp15 (left ankle) by 217px; CPU-vs-CoreML within Rust flips
it by 219px — identical behavior. Two CoreML runs (rust.coreml vs python.coreml) agree
on it (max 4.4px). It is a genuinely bistable low-value keypoint; gated by the
`DEFAULT_POSE_MIN_CONF` floor in production.

## Latency (release build, M-series, after one-time ~4.4s CoreML graph compile)

| path           | det      | pose     | per image |
|----------------|----------|----------|-----------|
| Rust CoreML    | 17–25 ms | 18–29 ms | ~35–55 ms |
| Python CoreML  | 12–30 ms | 18–29 ms | ~30–60 ms |
| Rust CPU       | 83–130ms | 46–191ms | slower    |

CoreML ≈3–5× faster than CPU on the models; Rust matches the Python CoreML path. (Rust
det is marginally higher only because the 640² letterbox is single-threaded scalar; not
worth optimizing for the spike.)

## Engineering notes for the implementation

- `ort = "=2.0.0-rc.12"` with `features=["coreml"]`. Default `download-binaries` fetches a
  prebuilt onnxruntime that **includes the CoreML EP** — no system onnxruntime needed.
  **Shipping caveat:** the packaged macOS app must bundle that `libonnxruntime*.dylib`
  (today the Python wheel ships it). Verify Tauri bundling or pin `ORT_DYLIB_PATH`.
- The YOLOX export here has **embedded NMS** (`dets`(1,N,5) f32 + `labels`(1,N) i64).
  Must skip the i64 output when extracting, and use the `shape[-1]==5` branch (keep
  score>0.3), **not** grid-decode.
- rtmlib feeds **BGR** to both models (never converts from cv2's BGR); mean/std apply in
  BGR-channel order. Matching this is required for parity.
- macOS-only dep (cfg(target_os="macos")), like the mlx-gen crates. Win/Linux keep the
  Python rtmlib path.
- The skeleton renderer (`openpose_skeleton::draw_wholebody`) is **already ported** in
  `crates/sceneworks-worker` — only the detector half is new.

## Reproduce

```
# Python reference (dwpose-spike venv has rtmlib+onnxruntime):
~/.dwpose-spike/venv/bin/python scripts/spikes/sc3487_reference.py \
    --images "/tmp/sc3487/sources/*.png" --out /tmp/sc3487/ref --device cpu   # and --device mps
# Rust detector:
cargo run --release --manifest-path scripts/spikes/sc3487_ort_pose/Cargo.toml --bin detect -- \
    --images "/tmp/sc3487/sources/*.png" --out /tmp/sc3487/rust --device cpu   # and --device coreml
# Parity report:
python3 scripts/spikes/sc3487_compare.py --ref /tmp/sc3487/ref --rust /tmp/sc3487/rust --b-device coreml
```

Standalone spike crate: `scripts/spikes/sc3487_ort_pose/` (detached `[workspace]`).
