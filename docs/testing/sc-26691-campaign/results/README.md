# sc-26691 terminal campaign verification (final, 2026-11-02)

**Result: evidence verified on both backends.** The paired campaign at SceneWorks `1e23ef16e`
passed every allocated case: MLX 25/25 and CUDA 18/18, each lane in a single run with normal anchor
pairing. The standard consolidation (`verify` over a master plan with `backendReadbacks`) re-validates
both native readbacks, re-verifies every retained case, and re-measures all reduced cross-case causal
controls and the rendered-content gate from the media. Every step passes. The verifier's last step,
`validateDelivery`, stops at `missing integrated delivery records`, which is expected at this point:
it needs post-merge records (final CI, inference and SceneWorks main merges, Shortcut readbacks) that
cannot exist before the epic merges. Nothing in the evidence failed.

This replaces the earlier FAILED record. That record combined split runs (41213935677 + 41310045937
for CUDA, 41331644188 + 41364105218 for MLX), and it is **superseded**: every video in those runs was
degenerate grey noise, caused by the QKV-layout bug described below. None of that evidence is used.

Files:

- `verify-campaign-pair.json` holds the consolidation result: the identity, matrix decision, all 49
  verification steps with their outcomes, the coverage report and the delivery-step result.
- `verify-campaign-mlx.json` and `verify-campaign-cuda.json` hold the per-backend case list, the cross-case
  controls and the rendered-content values, all re-measured here. Each control also carries the lane's
  own on-host readback delta.
- `verify-pair.mjs` is the driver that produced the three JSON files. It builds the master plan from the
  selected backend's plan: hash-bound retained witness and anchor paths are remapped to the downloaded
  artifacts, and `backendReadbacks` points at both lanes. It then runs `verifyCampaign`'s steps in order.
  Usage: `node verify-pair.mjs <sceneworks checkout> <evidence root> <out dir>`, where the evidence root
  holds `gh run download` of both runs under `mlx/` and `cuda/`. Running the CLI (`node
  scripts/h3-finetune-terminal.mjs verify master-plan.json`) on the same master plan exits only at the
  delivery step.
- `output-frames-contact.jpg` shows real frames. Each tile is 336×192. Row 1 is CUDA
  `real-hybrid-first-last`: first keyframe (red grid), output frame 0, last keyframe (blue grid), output
  frame 123. Row 2 is the same for MLX. Row 3 is MLX `artifact-0-ordered-mixed-baseline-ref-turbo`
  (a talking head) at frames 0, 41, 82 and 123.

## Runs, SHA, pin

| Lane | Run/attempt | Runner | SceneWorks | Inference pin | Native evidence | Cases |
|---|---|---|---|---|---|---|
| MLX (selected) | 41379525090/1 | `nax-macos-2` | `1e23ef16e` | `a43724b8` | `0bada8f7` | 25/25 passed |
| CUDA (displaced) | 41379549283/1 | `cuda-windows` | `1e23ef16e` | `a43724b8` | `0bada8f7` | 18/18 passed |

Both lanes ran at the same SHA and pin. There was no prior decision, no `inferenceChange` and no split
segment. Every case's execution record names its own run and `1e23ef16e`. The numeric artifact
probes carry device proof `mlx-default-gpu-no-fallback` (MLX) and `cuda-device-no-fallback` (CUDA, GPU 1).

## Matrix decision

The decision was measured in this pair from its own inline anchors: artifact-0 (reference) and
real-hybrid-text (prompt). The weights are reference 5 and prompt 2.

| Backend | Reference wall | Prompt wall | Weighted cost |
|---|---:|---:|---:|
| mlx | 2,008,028 ms | 658,822 ms | 11,357,784 |
| cuda | 3,161,060 ms | 978,762 ms | 17,762,824 |

The result is `mlx` selected and `cuda` displaced. `verifyMatrixAdmission` re-hashed all four anchor
receipts and both native encoded witnesses (`cheap-causal-witness`, mlx and cuda) from the downloaded
artifacts.

## Per-case status

All listed cases are `passed` and re-verified (`verifyPortableCase`). `·` means not allocated, because
the case belongs only to the selected backend.

| Case | MLX | CUDA |
|---|---|---|
| artifact-0-ordered-mixed-baseline-ref-turbo | passed | passed |
| artifact-1-reference | passed (8 steps) | passed (50 steps) |
| artifact-2-lokr-ref-turbo-baseline | passed | passed |
| artifact-3-opaque-renamed-linked-source | passed (8 steps) | passed (50 steps) |
| real-mixed-repeat | passed | passed |
| real-drop-images / real-drop-video | passed | · |
| real-drop-audio | passed | passed |
| real-reverse-images / -video / -audio | passed | · |
| real-drop-lora | passed | passed |
| real-lokr-repeat / real-drop-lokr | passed | passed |
| real-hybrid-text | passed | passed |
| real-hybrid-prompt-change / real-hybrid-text-repeat | passed | · |
| real-hybrid-image / -last-only / -first-last | passed | passed |
| real-hybrid-reference | passed (8 steps) | passed (50 steps) |
| real-independent-fl2va | passed | passed |
| numeric-pruned-artifact / numeric-w4a8-artifact | passed | passed |
| lifecycle-generation-source-prep-cancel | passed | passed |

The coverage report lists 84 catalog cells: 40 are proven by real-weight cases and 44 are reported as
explicit `unproven`. Those 44 are the reduced inventory's out-of-scope generalization, lifecycle and
numeric cells, plus CUDA's 4 capability cells, which are not allocated to the displaced backend. The run
had 37 unique executions with no shared or conflicting job IDs. Once delivery is recorded, the status
will read `reduced-evidence-reverified-with-explicit-unproven`.

## Cross-case causal controls

The metric is decoded RMSE: 16×16 frames at 1 fps for video, 8 kHz mono for audio. A control passes
when the intervention RMSE exceeds `max(minimumControlDelta = 0.01, 3 × repeat RMSE)`. Every repeat
(real-mixed-repeat, real-lokr-repeat, real-hybrid-text-repeat) is bit-identical to its baseline, so its
repeat RMSE is 0 and the threshold is 0.01. The values below were re-measured on this Windows host. The
lane's on-host readbacks agree to within 4e-4 (MLX, a different host ffmpeg) and exactly (CUDA).

| Control | Field | Stream | MLX RMSE | CUDA RMSE |
|---|---|---|---:|---:|
| real-drop-images | referenceAssetIds | video | 0.1893 pass | · |
| real-drop-video | sourceClipAssetIds | video | 0.2620 pass | · |
| real-drop-audio | referenceAudioAssetIds | audio | 0.1873 pass | 0.1554 pass |
| real-reverse-images | referenceAssetIds | video | 0.0359 pass | · |
| real-reverse-video | sourceClipAssetIds | video | 0.2421 pass | · |
| real-reverse-audio | referenceAudioAssetIds | audio | 0.1018 pass | · |
| real-drop-lora | loras | video | 0.2623 pass | 0.2527 pass |
| real-drop-lokr | loras | video | 0.0343 pass | 0.0358 pass |
| real-hybrid-prompt-change | prompt | video | 0.2800 pass | · |

Every control passes. The smallest margins are real-drop-lokr (0.034–0.036) and real-reverse-images
(0.036), about 3.5× the threshold. Both results are expected. Reordering two image references changes
less than dropping them. The controlled LoKr is a small rank adapter on top of the turbo baseline.
Under the failed record, the same video controls measured 0.003–0.009 on grey noise.

## Rendered-content gate (`RENDERED_CONTENT`, 8e0806cdb)

The gate works on 32×32 area-downsampled RGB. The thresholds are a median per-frame spatial std of at
least 0.07, and, for first/last-frame conditioning, a keyframe luma correlation of at least 0.4 with a
grey-relative RMSE ratio of at most 0.6. The values below were re-measured from each bound output and
its bound keyframe files. They are identical to the receipt each case recorded at run time
(`matchesRunReceipt: true` for all 22 MLX and 16 CUDA AV outputs).

**Median spatial std per case (124 frames each):**

| Case | MLX | CUDA |
|---|---:|---:|
| artifact-0 / real-mixed-repeat | 0.2308 / 0.2308 | 0.2087 / 0.2087 |
| artifact-1-reference | 0.1652 | 0.2137 |
| artifact-2 / real-lokr-repeat | 0.1247 / 0.1247 | 0.1454 / 0.1454 |
| artifact-3 | 0.1356 | 0.1918 |
| real-drop-images | 0.2240 | · |
| real-drop-video | 0.1713 | · |
| real-drop-audio | 0.1533 | 0.1399 |
| real-reverse-images / -video / -audio | 0.2290 / 0.2063 / 0.2302 | · |
| real-drop-lora | 0.1089 | **0.0982** (lowest) |
| real-drop-lokr | 0.1277 | 0.1452 |
| real-hybrid-text / -text-repeat | 0.2518 / 0.2518 | 0.2823 / · |
| real-hybrid-prompt-change | 0.2763 | · |
| real-hybrid-image | 0.2462 | 0.2448 |
| real-hybrid-last-only | 0.3029 | 0.2900 |
| real-hybrid-first-last | 0.2456 | 0.3395 |
| real-hybrid-reference | 0.1621 | 0.2457 |
| real-independent-fl2va | 0.2592 | 0.2548 |

**Keyframe resemblance (good side, first measured here):**

| Case | Position | MLX corr / ratio | CUDA corr / ratio |
|---|---|---|---|
| real-hybrid-image | first | 0.997 / 0.068 | 0.997 / 0.075 |
| real-hybrid-last-only | last | 0.875 / 0.186 | 0.876 / 0.184 |
| real-hybrid-first-last | first | 0.999 / 0.065 | 0.998 / 0.067 |
| real-hybrid-first-last | last | 0.787 / 0.201 | 0.788 / 0.200 |

**Margins:**

- **Keyframe thresholds are well placed.** These thresholds were set before any good render existed.
  The good side measures a correlation of 0.787–0.999 and a ratio of 0.065–0.201. The degenerate side,
  from the calibration fixture, measures a correlation of −0.077 to 0.131 and a ratio of 0.928–0.994.
  The 0.4 correlation bound sits 0.39 below the weakest good frame and 0.27 above the strongest
  degenerate one. The 0.6 ratio bound is 3× the weakest good frame and 0.33 under the best degenerate
  one. Both bounds are near the middle of the gap. The last frame always resembles its keyframe less
  than the first frame does (0.79–0.88 against ~0.998), because the endpoint is generated toward the
  keyframe rather than reproduced. Row 1 of the contact sheet shows the last output frame's grid at a
  slightly different scale from its keyframe. That is the expected behaviour and does not argue for
  moving the bound.
- **The spatial-std floor holds, but its good-side margin is thinner than calibrated.** The floor of
  0.07 is 2.5× the worst degenerate frame (0.0281). The weakest good median is 0.0982 (CUDA
  real-drop-lora, LoRA removed from the turbo baseline; MLX 0.1089), which is 1.4× the floor. Stock
  ref2va calibration had suggested ≥0.182. Per-frame minimums reach 0.070–0.071 in the keyframe cases
  (the near-uniform red grid at 32×32), which the median correctly ignores. The floor is not misplaced
  for these cases. A legitimately flat prompt, such as a plain backdrop, could still approach it, so the
  floor should be re-read if one appears. It was not changed.

## The QKV-layout bug and its fix

Every render in the earlier campaign runs was grey structured noise on both backends, from every
checkpoint, at both 8 and 50 steps, and every keyframe was lost. Even so, each case passed its decode,
lattice and sensitivity checks. The cause was the importer's codec. It split native fused
`attn.qkv_proj` in the **per-head interleaved** row order of the official MiniMax raw shards, but no
campaign checkpoint declares `h3_qkv_layout`, and all of them are **contiguous** `[q; k; v]`. That
covers the Comfy-Org single-file repack, the smhfacct hybrid, Singularity v1.3 INT8/W4A8 and PulpCut
ConvRot. The checkpoints were measured against the official Diffusers transformer
(MiniMaxAI/MiniMax-H3@6818f6c3) by per-row cosine.

`ec6d66ce2` makes the rule provenance plus metadata. An explicit `h3_qkv_layout` wins. Conflicting
declarations and the undeclared official raw-shard spelling are refused. Every other single-file native
source is treated as contiguous. The derivative adapter version moves to 2, so caches normalized under
the old default regenerate. An opt-in real-artifact test measures 0.9999–1.0000 per-row cosine on six
real checkpoints, against ≤0.033 for the old gather.

`8e0806cdb` adds the rendered-content gate above, so a degenerate render now fails its case at run time
and again at readback, instead of being caught only by the cross-case controls once a backend
completes. `fd9a79f75` documents both changes.

## Scope note

The user approved backend-specific step counts on 2026-10-30. The compatible-base cases
(artifact-1-reference, artifact-3-opaque-renamed-linked-source and real-hybrid-reference) ran 50 denoise
steps on CUDA and 8 on MLX, because a 50-step MLX case does not fit the 7200 s case bound. The
source-preparation cancel case declares the same counts but runs no denoise step. Everything else is
identical, and no output of these cases is compared across backends. The turbo, LoRA/LoKr, text and
keyframe recipes keep their own step counts (4 or 8) on both backends.

## Superseded evidence

| Runs | Status |
|---|---|
| CUDA 41213935677/1 + 41310045937/1 (`4c3625898` / `70f093db7`, pin `354b41cc`) | Superseded: all video degenerate (QKV layout) |
| MLX 41331644188/1 + 41364105218/1 (`ae148c29a` / `d8101a715`, pin `a43724b8`) | Superseded: all video degenerate (QKV layout) |
| 41303430895 | Superseded earlier (first-last keyframe bug) |

The split-run tooling (`verify-split-backend`, prior-decision adoption, lane continuation) stays in the
harness for future campaigns. This result does not use it.

## Follow-ups

- Record delivery after the epic merges (final-head CI, inference and SceneWorks main merges, Shortcut
  readbacks for 26679 and 26691), then run `verify` to its final status.
- Re-read the 0.07 spatial-std floor if a legitimately low-detail prompt is added. The thinnest good
  margin today is 1.4× (CUDA real-drop-lora).
- In this pair, CUDA anchors measured slower than MLX: 3,161 s against 2,008 s for the reference anchor.
  That is why MLX was selected. CUDA runs on GPU 1, the display GPU. Look into this if the CUDA lane's
  throughput matters for future campaigns.
- Old roots under `E:\huggingface\sc26691-private` are the operator's to clear. This verification wrote
  nothing to `E:`.
