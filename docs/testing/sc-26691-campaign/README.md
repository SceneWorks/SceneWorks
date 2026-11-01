# sc-26691 H3 terminal campaign — dispatch kit

The request payloads and dispatch script the reduced terminal campaign (epic sc-26679, story
sc-26691) was run with. The full procedure and evidence rules are in
[`../sc-26691-h3-terminal.md`](../sc-26691-h3-terminal.md); the current state and open findings are
in the closeout handoff on the epic in Shortcut.

## Today's plan (user-approved 2026-11-01)

- **CUDA, 18 passed cases at `4c3625898` stand.** CUDA run 41213935677 (attempt 1, `cuda-windows-2`)
  passed 18 of its 25 allocated cases (retained in `h3-terminal-cuda-reduced-41213935677-1`), then
  refused `real-hybrid-first-last` for disk space: it had written ~500 GB under
  `E:\huggingface\sc26691-private`. That evidence is not re-run.
- **CUDA, the 7 remaining cases at the new SHA**: `real-hybrid-first-last`, `real-hybrid-reference`,
  `real-independent-fl2va`, `real-hybrid-text-repeat`, `numeric-pruned-artifact`,
  `numeric-w4a8-artifact`, `lifecycle-generation-source-prep-cancel`. The lane adopts run
  41213935677's measured matrix decision (cuda selected, mlx displaced; recomputed from the retained
  anchors of runs 41213935677 and 41213914294) instead of a new anchor exchange.
- **MLX, the full displaced lane (18 cases) on `nax-macos-2` at the new SHA**, on the same declared
  decision. `nax-macos` keeps two orphaned processes from an ejected-drive run and is not used.
  `/Volumes/Models` on `nax-macos-2` is full: the lane only READS it; every write (including any
  pinned weight it must download) goes to `~/sc26691-campaign` on the main drive.
- The new SHA differs from `4c3625898`; each lane records `git diff --name-only 4c3625898..<new>` in
  `prior-matrix-decision.json`. No same-SHA re-run is implied or claimed.

```bash
SW="$(git rev-parse origin/feature/sc-26679-h3-finetune-import)"   # the pushed new head
# CUDA: the 7 remaining cases on run 41213935677's decision (work root D:\sc26691-campaign by default).
BACKENDS=cuda PRIOR_MATRIX_RUN=41213935677/1 \
  CASE_IDS=cuda.reduced.real-hybrid-first-last,cuda.reduced.real-hybrid-reference,cuda.reduced.real-independent-fl2va,cuda.reduced.real-hybrid-text-repeat,cuda.reduced.numeric-pruned-artifact,cuda.reduced.numeric-w4a8-artifact,cuda.reduced.lifecycle-generation-source-prep-cancel \
  docs/testing/sc-26691-campaign/dispatch.sh --ack "$SW"
# MLX: the full displaced lane on nax-macos-2, same decision, weights read from /Volumes/Models,
# writes (and downloads) under ~/sc26691-campaign on the main drive.
BACKENDS=mlx MLX_RUNNER=nax-macos-2 PRIOR_MATRIX_RUN=41213935677/1 \
  docs/testing/sc-26691-campaign/dispatch.sh --ack "$SW"
```

### Where the campaign reads and writes

- **Reads weights** only from the HF hub: `E:\huggingface\hub` on Windows; `mlx_hf_home` (default
  `/Volumes/Models/huggingface`, `~/` allowed) on macOS. The stage copies a present, SHA-verified file
  into the run's private staging (`copyFile`, never a hard link), so the hub may sit on another volume.
- **Writes** only under `work_root` (default `D:\sc26691-campaign` on Windows, `~/sc26691-campaign` on
  macOS): `sc26691-private/<run>-<attempt>` (staging), `...-bootstrap-<sha>` (case roots and
  `services/<case>`), `...-target`/`-native`/`-fixture` (cargo targets), `sc26691-python` (Windows
  portable CPython), the prior-run cleanup and the leftover-process scan. The lane refuses, loudly,
  a work root on `E:` or under the macOS HF home (`H3_WRITE_ON_WEIGHTS_VOLUME` /
  `H3_WRITE_UNDER_WEIGHTS_ROOT`).
- **Downloads**: every pinned file the bootstrap needs (singularity, hybrid, pulpcut, turbo, speech
  and the catalog components) that is absent from the hub is downloaded automatically, SHA-verified,
  into the run's private staging under the work root; nothing is written into the hub. The raw
  builtin/community shard sets stay cache-probe-only (`allow_fresh_builtin_download`), and the
  bootstrap does not use them.
- Old roots under `E:\huggingface\sc26691-private` (and `E:\huggingface\sc26691-python`) are left
  untouched by the lanes; nothing writes to or deletes from `E:`. Clearing them is the operator's call.
- Per-case reclaim now also removes each assembly's `transformer` directory (full private copies for
  hybrid cases, hard links otherwise; links are never followed), recorded in `storage-reclaim.json`.

### Combining the two CUDA runs (explicit, never a silent merge)

The CUDA allocation is then split across two runs at two SceneWorks SHAs. Re-validate it as such:

```bash
gh run download 41213935677 --repo SceneWorks/SceneWorks --name h3-terminal-cuda-reduced-41213935677-1 --dir prior
gh run download <new-run> --repo SceneWorks/SceneWorks --name h3-terminal-cuda-reduced-<new-run>-1 --dir subset
cat > split.json <<EOF
{"backend":"cuda","segments":[
  {"label":"prior-4c3625898","caseRoot":"$PWD/prior","runId":41213935677,"attempt":1},
  {"label":"subset-new-sha","caseRoot":"$PWD/subset","runId":<new-run>,"attempt":1}]}
EOF
node scripts/h3-finetune-terminal.mjs verify-split-backend split.json
```

`verify-split-backend` re-validates every passed case against the exact plan it ran under (its
retained `adapter-plan.json`, bound by `planSha256`), attributes each case to its run and SHA,
requires one shared measured decision, pin and native evidence, refuses a case that passed in both
segments, reports any allocation gap as `split-backend-incomplete`, and flags cross-case controls
whose cases come from different segments (`crossSegment: true`, e.g. `real-hybrid-text-repeat`
against `real-hybrid-text`). Its output is a `split-backend-content-readback` with
`singleShaBackendReadback: false`; it is not accepted where a single-SHA backend readback is required.

## Dispatch

```bash
# Both lanes, paired (each lane waits for the peer's anchor artifact while the peer lane is alive,
# bounded by its work deadline (lane budget and job timeout) less its remaining cases; it fails at once if the peer ended or was never dispatched).
docs/testing/sc-26691-campaign/dispatch.sh --ack "$(git rev-parse origin/feature/sc-26679-h3-finetune-import)"

# One lane (the peer must be dispatched at the SAME SceneWorks SHA to pair).
BACKENDS=cuda docs/testing/sc-26691-campaign/dispatch.sh --ack <sceneworks-sha>
```

- `--ack` records the operator's scoped MiniMax-H3 Community License acknowledgment (sc-26691
  activity 27075) and must name the branch's current head; the script refuses a moved head.
- `inference_sha` must be reachable from inference `origin/main` — the bootstrap refuses an interim
  feature pin. `dispatch.sh` stamps it from the reviewed head's `crates/sceneworks-worker/Cargo.toml`
  pin (`354b41cc` today: inference main, which carries SceneWorks/inference#1389); the checked-in requests carry the
  same value.
- `native_evidence_sha` (the separate `inference-evidence` checkout that builds the native probes and
  runs the CPU oracle bridge) is inference main `0bada8f7` (SceneWorks/inference#1385: the bridge binds
  the pruned reduced-projection weight with its `[out, rank]` header shape; at `1f004238` that case
  always refused). It must differ from the product pin and is not stamped by `dispatch.sh`. Its frozen
  producer digests must equal `FROZEN_PROVENANCE` in `scripts/h3-finetune-terminal-artifact-probe.mjs`
  (the ConvRot oracle digest moved with main's fetched-source verification). `native_evidence_ref`
  is `main`, which the bootstrap asserts; the evidence checkout is by SHA.
- Requires `bash`, `node`, `git` and an authenticated `gh` (on Windows: Git Bash); no standalone
  `jq`. `DRY_RUN=1` runs every read and stamps/validates both requests, printing them instead of
  dispatching.

## Knobs

- `bootstrap.operator.width` / `height` (1088x544) set the canvas of the turbo/ref cases;
  `timeoutSeconds` (7200) is the per-case bound. See the epic handoff for why these are under review.
- `expected_runner`: `cuda-windows` / `cuda-windows-2` (the job is pinned by the runner-name label);
  `nax-macos-2` (label `rw-mage`, the default `MLX_RUNNER`) or `nax-macos` (`rw-starvector`) for MLX.
  Both Macs carry `real-weights`; the lane asserts `RUNNER_NAME` equals the request's runner.
- `work_root`, `mlx_hf_home`, `case_ids`, `prior_matrix_decision`: stamped by `dispatch.sh` from
  `WORK_ROOT` (or `CUDA_WORK_ROOT` / `MLX_WORK_ROOT`), `MLX_HF_HOME`, `CASE_IDS` and
  `PRIOR_MATRIX_RUN=<run>[/<attempt>]`; see the header of `dispatch.sh`.
- CUDA uses `gpuId`/`requestedGpu` `1` (physical GPU 1 in PCI-bus order: `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
  `CUDA_VISIBLE_DEVICES=1`, `SCENEWORKS_GPU_ID=1`; device evidence is the GPU-1 UUID). GPU 0 is the
  owner's display/work GPU and is never used. MLX uses `mlx`.

## Representative case geometry

Coverage is unchanged: all four Singularity artifacts on both backends, decoded video + audio, and
the full reference/conditioning surface (ordered image + clip + voice references, drop/reverse
interventions, keyframes, hybrid and the independent fl2va merge). What changed is how heavy each
reference is, because H3 packs every reference into the DiT sequence and per-step attention is
quadratic in it:

- **Generated clip:** every case renders 124 frames (5.17 s, H3's shortest legal duration) at
  1088x544, or 1344x768 for the 768p first/last recipe (768 short edge inside H3's
  `maxPixels` 1,032,192; 1536x768 was silently fitted to 1408x704). Not reducible.
- **Image references** (`imageA`/`imageB`, 576x320 plates): the request sets no
  `referenceImageShortEdge` or `referenceImageResizeMode`, so the engine defaults apply. At the
  current inference pin (`354b41cc`, which carries the sc-26691 engine change) the default resize
  mode is `match` (`min(1, sqrt(targetArea / refArea))`, sides on the 32 lattice): a 576x320 plate
  against the 1088x544 canvas stays 576x320 (180 + 180), where the pre-sc-26691 pin (`232f3f24`)
  upscaled every plate to the 2048 short edge (3680x2048, 7,360 latent rows + 7,360 vision tokens
  each). Do not add an explicit short edge to the requests.
- **Clip references** (`clipA`/`clipB`): 2 s / 48 frames (was 6 s / 144 frames, truncated by the
  engine to the full 124 generated frames). On the engine's 768-short-edge canvas (1376x768) a clip
  snaps to 39 frames = 12 latents (12,384 rows) and 2 vision blocks, instead of 37 latents (38,184)
  and 6 blocks. Still above the 22-frame VAE floor and the 13-frame vision-sampling floor.
- **Voice references**: the ~5.9 s LibriSpeech row (235 audio rows; the engine change truncates a
  reference soundtrack to the generated 5.17 s, 207 rows).

Packed DiT rows per case (`S`; target video + target audio + reference latents + reference audio +
vision tokens + ~100 text rows), and the worker's stall budget. "Engine S" follows the candle H3
normalisation rules; "pin" is the current inference pin with the old 6 s clips (the measured
ordered-mixed run packed ~141k rows), "kit" is these 2 s clips under the `match` default. The worker
budgets on `gen_core::veda::workspace::h3_row_bound`, an upper envelope that prices every reference
clip at the generated latent-frame count, so it does not shrink with the clip; the step budget is
`ceil(600 s * S(26880 + S) / (S0(26880 + S0)))` with `S0` = 38,222 (+ prompt) for 1344x768x124
text-only, floored at 600 s and capped at 4 h; the pre-first-step window is 45 min + 600 s per `S0`
reference rows + one step, capped at 8 h.

| Cases | Engine S (pin) | Engine S (kit) | Worker envelope S (pin) | Step budget | Pre-step window |
|---|---:|---:|---:|---:|---:|
| 1, 5, 9-12 ordered-mixed (2 img, 2 clip, 2 voice) | 141,042 | 52,090 | 148,358 | 6,228 s | 10,896 s |
| 6 drop-images | 111,602 | 51,370 | 118,790 | 4,145 s | 8,351 s |
| 7 drop-video | 51,810 | 23,034 | 52,510 | 999 s | 4,169 s |
| 8 drop-audio | 140,572 | 51,676 | 147,888 | 6,191 s | 10,852 s |
| 2-4, 13, 14, 20 reference (1 img, 1 voice) | 36,855 | 22,467 | 37,491 | 600 s | 3,535 s |
| 15, 16, 21, 22 text | 21,900 | 21,900 | 22,382 | 600 s | 3,300 s |
| 17 image / 18 last-frame (1 keyframe) | 23,056 | 23,056 | 23,602 | 600 s | 3,320 s |
| 19 first-last (2 keyframes, estimated at 1536x768; now 1344x768) | 47,746 | 47,746 | 48,356 | 875 s | 3,650 s |

**Step counts per backend (user-approved scope decision, 2026-10-30).** The compatible-base cases
(2, 4, 20, and the cancel case 25) run 50 denoise steps on CUDA and 8 on MLX. Nothing else differs:
same inputs, geometry, adapters, seed and shift. At the tiled MLX kernel one H3 step measured about
5.4–6.6 min (run 41061562586). A 50-step MLX case (about 5 h) cannot fit the 7200 s case bound or the
24 h lane, while 8 steps (about 50 min) can. The count is declared per backend in the case manifest
(`backendSteps`) and recorded in the plan's `sampling.stepPolicy`. Validators check the requested
steps per backend, and no MLX/CUDA output of these cases is compared across backends. See *Backend-specific
step count* in `docs/testing/sc-26691-h3-terminal.md`. The turbo cases keep their recipe's 4 or 8
steps on both backends.

Cases 23-25 (artifact probes, source-preparation cancel) run no denoise step. The worker-envelope
columns use the pin's image rule; with the engine change the envelope's image rows drop to the
`match` size too (that function moves with the pin). `timeoutSeconds` (7200 s) is the whole-case
bound: at the pin, an ordered-mixed case's 4-step ref-turbo recipe may exceed it; at the kit
geometry it fits.
