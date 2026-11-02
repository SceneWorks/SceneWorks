# sc-26691 combined terminal campaign verification (2026-11-01)

**Result: FAILED on both lanes.** `node scripts/h3-finetune-terminal.mjs verify-split-backend <manifest>`
exits 1 for CUDA and for MLX with `conditioning/adapters ignored or below baseline noise floor`.
Every allocated case is present, and each one re-verifies against the plan it ran under: 25/25 CUDA
and 18/18 MLX, with run and SHA attribution, the shared decision, pins, native evidence, reused
witnesses and pre-decision anchors all checked. The verifier then refuses the reduced cross-case
causal controls. Those controls had never been evaluated on real evidence before this readback,
because no single run completed a backend. Visual inspection shows why they fail: the generated
videos are near-uniform grey noise, and they do not reproduce their keyframes
(`output-frames-contact.jpg`). The campaign is therefore **not** accepted, even though every lane
run reported green.

Files:

- `verify-split-backend-cuda.json` and `verify-split-backend-mlx.json` hold the verifier's own
  output fields, produced with its cross-case control step instrumented, plus `verifierError` and a
  `controlTable` that evaluates every control on its own, using the verifier's `decodedDelta` and the
  predeclared thresholds. `status` is `failed`. Neither file is a passing split readback.
- `output-frames-contact.jpg` has three rows. Top: the CUDA `real-hybrid-first-last` first keyframe
  (red grid) next to output frame 0. Middle: the last keyframe (blue grid) next to the final output
  frame. Bottom: frame 2.5 s of `artifact-0-ordered-mixed-baseline-ref-turbo`, CUDA on the left and
  MLX on the right.

## Runs, SHAs, pins

| Lane | Segment (role) | Run/attempt | SceneWorks | Inference | Cases |
|---|---|---|---|---|---|
| CUDA | decision-run | 41213935677/1 | `4c3625898` | `354b41cc` | 18 |
| CUDA | prior-decision-subset | 41310045937/1 | `70f093db7` | `354b41cc` | 7 |
| MLX | prior-lane-run | 41331644188/1 (`nax-macos-2`) | `ae148c29a` | `a43724b8` | 15 |
| MLX | prior-decision-subset (continuation) | 41364105218/1 (`nax-macos-2`) | `d8101a715` | `a43724b8` | 3 |

Run 41303430895 failed on the first-last keyframe bug and is superseded; nothing from it is used.
The native evidence SHA for every segment is `0bada8f7`.

### Case to run

- **CUDA, run 41213935677 (`4c3625898`):** artifact-0/1/2/3, real-mixed-repeat,
  real-drop-images/video/audio, real-reverse-images/video/audio, real-drop-lora, real-lokr-repeat,
  real-drop-lokr, real-hybrid-text, real-hybrid-prompt-change, real-hybrid-image,
  real-hybrid-last-only.
- **CUDA, run 41310045937 (`70f093db7`):** real-hybrid-first-last, real-hybrid-reference,
  real-independent-fl2va, real-hybrid-text-repeat, numeric-pruned-artifact, numeric-w4a8-artifact,
  lifecycle-generation-source-prep-cancel.
- **MLX, run 41331644188 (`ae148c29a`):** artifact-0/1/2/3, real-mixed-repeat, real-drop-audio,
  real-drop-lora, real-lokr-repeat, real-drop-lokr, real-hybrid-text, real-hybrid-image,
  real-hybrid-last-only, real-hybrid-first-last, real-hybrid-reference, real-independent-fl2va.
- **MLX, run 41364105218 (`d8101a715`):** numeric-pruned-artifact, numeric-w4a8-artifact,
  lifecycle-generation-source-prep-cancel. The two numeric cases carry device proof
  `mlx-default-gpu-no-fallback`.

Every listed case has execution status `passed` and re-verifies individually.

## Matrix decision

There is one measured decision, from run 41213935677/1. It was recomputed from the retained anchor
packets of runs 41213935677 (cuda) and 41213914294 (mlx).

- The decision is `cuda` selected, `mlx` displaced.
- The weighted cost is 16,628,470 for cuda against 22,750,796 for mlx.
- The anchors are artifact-0 (reference) and real-hybrid-text (prompt).

The CUDA subset run and both MLX runs adopted that decision rather than exchanging anchors again.
The decision run's two CUDA anchors ran before the decision existed, and their plans carry no
`matrixDecision`. The verifier now binds them to the decision run's retained anchor receipts and
reports them as `preDecisionAnchorCases`.

## Recorded SHA and pin changes

- **CUDA:** `sourceShaDifference` runs from `4c3625898` to `70f093db7`, covering 28 files. The
  inference pin is unchanged (`354b41cc`).
- **MLX:** `sourceShaDifference` runs from `4c3625898` to `d8101a715`, covering 141 files. The
  continuation's `priorLaneRun` is run 41331644188 at `ae148c29a`, with 15 passed cases.
- **MLX `inferenceChange`:** the pin moves from `354b41cc` to `a43724b8`, with `descends: true`. The
  11-entry first-parent log runs from inference #1394 to #1405, the shared H3 keyframe contract.
- **`caseDefinitionDifferences`:** only `inputIdentities` differs, with no workload fields.

## Reused native witnesses

The subset segments ran no native witnesses of their own. Instead they reused the decision's anchor
packets, which were measured at the same native evidence SHA `0bada8f7`, and re-hashed them from
retained files:

- `native-encoded-surface-mlx-0bada8f72…` (actions:41213914294/1)
- `native-encoded-surface-cuda-0bada8f72…` (actions:41213935677/1)

## Scope decision

MLX ran its compatible-base cases at 8 steps, a user-approved reduction, where CUDA used 50. These
are artifact-1-reference, artifact-3-opaque-renamed-linked-source and real-hybrid-reference. The
recorded sampling policies are:

- Turbo, LoRA/LoKr and text cases: 4 steps, scheduler shift 12.
- real-hybrid-first-last: 4 steps, shift 6.
- real-hybrid-image, real-hybrid-last-only and real-independent-fl2va: 8 steps, shift 12, on both
  backends.

## Cross-case controls (the failure)

The metric is decoded RMSE: 16×16 frames at 1 fps for video, and 8 kHz mono for audio. A control
passes only when the intervention RMSE exceeds `max(minimumControlDelta = 0.01, 3 × repeat RMSE)`.
Every repeat is bit-identical (repeat RMSE 0), so the threshold is 0.01.

| Control | Field | Stream | CUDA RMSE | MLX RMSE |
|---|---|---|---|---|
| real-drop-images | referenceAssetIds | video | 0.0036 FAIL | not allocated (displaced) |
| real-drop-video | sourceClipAssetIds | video | 0.0070 FAIL | not allocated |
| real-drop-audio | referenceAudioAssetIds | audio | 0.2070 pass | 0.1967 pass |
| real-reverse-images | referenceAssetIds | video | 0.0034 FAIL | not allocated |
| real-reverse-video | sourceClipAssetIds | video | 0.0044 FAIL | not allocated |
| real-reverse-audio | referenceAudioAssetIds | audio | 0.0924 pass | not allocated |
| real-drop-lora | loras | video | 0.0092 FAIL | 0.0092 FAIL |
| real-drop-lokr | loras | video | 0.0029 FAIL | 0.0030 FAIL |
| real-hybrid-prompt-change | prompt | video | 0.0030 FAIL | not allocated |

Every video control fails on both backends, and every audio control passes. Three observations
explain this:

- The baseline videos are nearly flat grey, with a spatial std of about 0.03 on a 0..1 scale. At
  128 px over all frames the deltas stay small: real-drop-images is 0.0046 and real-hybrid-prompt-change
  is 0.0030.
- In real-hybrid-first-last, frame 0 and the final frame are flat grey, not the red-grid and
  blue-grid keyframes.
- CUDA and MLX produce near-identical noise with near-identical LoRA/LoKr deltas, so the cause is
  shared upstream, not specific to one backend.

The 0.01 threshold is a predeclared operator knob (`bootstrap.operator.minimumControlDelta`), and it
was not changed. Lowering it after seeing these numbers would weaken the check.

## Verifier correction made during this readback

The CUDA readback first failed for a reason unrelated to the evidence, with
`segment prior-4c3625898 case cuda.reduced.artifact-0-… ran under another matrix decision`. The
verifier required every decision-run case to carry the decision. By design, though, the decision
run's own anchors ran before it was measured. `matrixDecision` and `matrixAnchors` sit outside the
plan digest, so all 18 cases share planSha256 `a69b422e…`.

The verifier now accepts a no-decision case only under all of these conditions:

- It is in the decision-run segment, not a lane-run segment.
- It is one of the decision's own `anchorIds` for this backend.
- Its plan carries no anchors and no prior record.
- It has the same planSha256 as the run's decided cases.
- Its `execution.json` hashes to the anchor receipt in the retained, re-hashed anchor packet.
- That anchor receipt is also an anchor the subset plan adopted.

The test fixture was changed to match the real artifact shape, and tests were added for the
refusals.

## Known follow-ups

- **Blocker:** find out why H3 generation emits grey noise on both backends, and why keyframes are
  not reproduced. Then re-run the affected AV cases. Step count is not the cause: the 50-step CUDA
  compatible-base cases (artifact-1-reference, real-hybrid-reference) and the 8-step MLX ones are
  grey as well. Suspects shared across both backends and all checkpoints include VAE decode or latent
  scaling, the noise/scheduler setup, the conditioning path at 1088×544, and the reduced canvas or
  frame geometry.
- Per-case `verifyPortableCase` checks AV decodability and the contract, not visual content. The
  reduced cross-case controls are the only content gate, and they run only when a backend completes.
  A cheap per-case non-degenerate-output check would have caught this in the first run.
- Old roots under `E:\huggingface\sc26691-private` are the operator's to clear. This readback wrote
  nothing to `E:`.
