# MOSS Image comparison (sc-26322)

This is an offline reference evaluation package, **not completed comparison
results, production integration, or native MOSS support**. No quality or cost
numbers are checked in. The story remains open until measured outputs/failures,
independent judgments, DMD precision validation, and analyst conclusions exist.
Calibration must precede a measured wall-clock estimate and scored launch.

## Fixed budget and corpus

`corpus.json` contains 12 scenarios, two distinct prompts each, exact requested
text, object/text boxes, task requirements, raw and separately enhanced prompt
bytes, two seeds (173,991), and explicit wide/tall canvases. Its 24 cases cover:
English signs; Chinese/mixed text; dense whiteboards; posters/covers/thumbnails;
infographics/diagrams; JSON text placement; JSON object placement; revised layout
specifications; long prompts; tag/anime prompts; photographic portraits; products.

Each configuration selects exactly one frozen raw/enhanced prompt condition;
this never doubles the render budget. Primary defaults use raw prompts for all
six configurations. Enhanced prompts must be reviewed for semantic equivalence
and selected before freezing. Report differing enhancement conditions explicitly.
The image count is 12×2×2×6=288 scored outputs, at most six serial 1024² calibration
outputs, and at most 24 later native probes (not launched by this package):318
maximum generated images across phases. Prediction parity fixtures add no images.
Every one of the three MOSS configurations is paired against every one of the
three baselines on all 48 cases: 3×3×48=432 blind comparisons.

The configuration identities are MOSS Base BF16, MOSS SFT BF16, SFT BF16 plus unmerged FP32 DMD adapters, and native Qwen Image 2.1 / FLUX.2 Klein 9B / Z-Image Turbo at q4.
The template freezes exact revisions discovered for this evaluation. It uses
matched case resolutions and each model's declared steps/guidance, rather than
silently changing the native defaults to match MOSS. The q4 baselines and MOSS
BF16 and mixed-precision candidate paths have different precision/runtime costs; comparisons must
retain those conditions and cannot establish architecture-only or model-intrinsic efficiency; this is screening
of concrete product options under declared conditions.

For the six JSON text/object/revised-layout cases, MOSS uses the author's
`summary/items`, `abs_centralized`, `yxyx` pixel schema. The three incumbents
receive the same reviewed narrative representations, retaining every literal
string, object, integer box, coordinate frame and restriction. Their official
guidance documents prose; it does not establish interpretation of the MOSS
schema. Numeric placement remains a target pending measurement. The other
eighteen source prompt forms are unchanged. Each configuration freezes both
raw/enhanced representation bytes, their UTF-8 SHA256s and the original source
prompt SHA256s; each cell records the selected form/hash. `corpus.json` records
source guidance and the accepted proposal digest. Objective annotations retain
normalized `xyxy` boxes (integer layout rounding differs by at most0.4pixel).
Revisions regenerate final specs. DMD's author-linked demo claims the SFT layout
forms are supported; transfer remains unvalidated until outputs are scored.

## Preparation and execution

Dependencies are Python3.12+, torch>=2.9, diffusers0.38, transformers5.3,
accelerate>=1.10, safetensors, Pillow and psutil. Freeze **actual resolved
versions**, not these ranges, together with source/code/model hashes and device,
CUDA/driver/host identities. Provisioning is a separate explicit bootstrap action;
`run.py` never acquires weights or installs dependencies.

Use absolute paths for all run state, outside the repository. The bootstrap
transport supplies the concrete host paths, approved byte budget and exclusive
device lease. Rendering also requires its live ancestor-held device lock and
MOSS_DEVICE_LOCK/MOSS_LEASE_ID/MOSS_PROTOCOL_SHA256 environment.
The direct CLI imports the same ExecutionGuard as bootstrap: frozen physical GPU index/UUID
GPU-e4b79931-7be6-f216-460a-f5405cfafffe, live ancestor-held lease, reconciled
background ownership, continuous foreign-process checks and deadline enforcement.
Native resolver authority includes exact refs/main, the complete snapshot set and
isolated cache closure; changed refs or an added alternate snapshot fail before
requests. Full input hashes run at configuration boundaries; inexpensive resolver
identity and file fingerprints run during cells. Scored execution additionally
requires accepted calibration receipts/estimate and explicit coordinator
authorization bound to the same protocol. The explicitly authorized fresh v9 replacement uses GPU0 with its own frozen index/UUID; original GPU1 evidence stays historical. Direct run.py
commands below execute only inside that coordinator-owned lease. An expired,
wrong-device, wrong-freeze, or missing coordinator process fails before spawning. Standalone preparation follows this sequence:

```text
python scripts/evaluation/moss/evaluate.py template /ABS/draft.json
# Fill runtime, hardware, ownership, local snapshots, config.settings.
python scripts/evaluation/moss/evaluate.py inventory /ABS/draft.json /ABS/inventoried.json
python scripts/evaluation/moss/evaluate.py freeze /ABS/inventoried.json /ABS/lock.json
python scripts/evaluation/moss/evaluate.py plan --lock /ABS/lock.json --output /ABS/plan.json
python scripts/evaluation/moss/run.py /ABS/lock.json /ABS/run --config moss_base --phase calibration
```

Execute the other five configuration IDs serially with the same lock/run root.
Each calibration command permits exactly one output. Then validate:

```text
python scripts/evaluation/moss/evaluate.py validate /ABS/lock.json /ABS/run --phase calibration
```

Use measured calibration time, actual full-resolution token/step counts, and
observed memory to estimate the sampled square condition. Calibration uses the
actual first scored1024² fixture; wide/tall canvas costs and entire-campaign
wall-clock remain uncertain until measured. No six-square linear extrapolation
is represented as a complete campaign estimate. Only the coordinator authorizes the scored
campaign, using the same command with `--phase scored` for each configuration.
All six calibration receipts must exist first. A failed calibration cannot become
scored success: explicit unavailable status requires a new reviewed freeze and
reproducible evidence. Carrying calibration to a new freeze requires review;
this tool never silently rewrites receipt identities.

Every invocation has a bounded initialization/per-output timeout. Failure stops
that configuration's runtime and emits explicit failure receipts for its remaining
cells without more GPU work. No automatic retry occurs. Existing receipts/logs
are never overwritten. Resume skips only validated successful cells; failed
receipts and orphan images block automatic repeats. Every attempt uses its own
log. Input/ownership mutations create durable invalidations, and affected prior
receipts cannot subsequently be accepted or resumed. Review the failure and use a fresh approved run directory
for a replacement; retain the original attempt. Ownership records include job,
revision/lock, responsible chat, deadline, resume and next action. A pending
cell is missing evidence, not an unavailable or successful result.

## Reference/runtime contracts

`reference.py` imports only the local reviewed `pipeline_moss_image.py` with
SHA256 `1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31`.
It never imports the Space's `app.py` (which has top-level download/server side
effects). All component loads require safetensors and local files; Transformers
remote code is disabled. Shared tokenizer/encoder/VAE use the frozen SFT
snapshot, whose shared weight bytes match Base at the researched revisions.

Base/SFT use the actual published training-grid-snapped scheduler, shift5,
50steps, CFG3, negative branch skip22, BF16 transformer/encoder and FP32
noise/Euler/VAE. Actual sigma/timestep arrays and initial noise hashes are saved.
TF32 matmul/convolution are explicitly disabled and effective precision/device
flags retained. DMD uses the author's eight-step FP32 fixed-noise recursion with no autocast,
CFG, negative branch, release Euler scheduler, re-noising or resampled codes:
`x=z+Q*x0; v=transformer(x,step_codes[r],...); x0=z-v`. Prefixes retain the
trained order. Q/codes shape, dtype and finiteness are checked; VAE decode uses
the published batch-normalization denormalization and 2×2 unpatchify.

DMD `settings.adapter_validation=true` is mandatory. The scored arm loads the
SFT BF16 backbone, strictly covers every rank128/alpha128 LoRA target, and
applies unmerged FP32 residuals. All time embedding tensors and **both** modulation
biases are replaced; the three time/modulation modules stay FP32. The scored
configuration is labeled `bf16+fp32-unmerged`, never full-transformer FP32.

A separate owned parity process loads candidate and full FP32 oracle, without a
VAE or decoded images. It compares eight predictions on identical fixed noise,
conditioning, Q/codes and candidate-recurrence inputs, saving input/context/prediction
hashes, numerical errors and forward counts. That process must exit and join
before a fresh **candidate-only** process measures generation; oracle tensors,
CUDA allocator state and host RSS cannot contaminate candidate costs.

Five representative existing fixtures are frozen: English square, Chinese square,
JSON text layout square, wide product and tall product, one seed each. Calibration
runs the first fixture once (16 forwards); scored preparation reuses that exact
hash-bound receipt and adds only the four missing fixtures (64 forwards). Total
controlled parity is at most80 forwards and zero images. Existing parity receipts
are immutable and validated before reuse. Recorded errors must meet frozen
thresholds for recommendation eligibility; CPU tests cannot establish real-weight
parity. Residual arithmetic is explicit rather than assumed equivalent.

Native baselines use `native.py` and the existing registered product route.
The frozen settings require `binary`, `binary_sha256`, `runtime_versions`
(exact SceneWorks source and inference SHA), `hub_cache`, `state_root`,
`device_id`, `port`, `worker_id`, and `job_timeout_seconds`. `state_root` is an
empty owned parent; one fresh API/worker pair is created per configuration.
No extra native services, builds or downloads run for telemetry.

Build the existing binary at the frozen SceneWorks source/inference identity:
`cargo build --release --locked -p sceneworks-rust-api --features backend-candle --bin sceneworks-rust-api`.
`native_services.py` isolates DATA/CONFIG/JOBS_DB/CREDENTIALS directories,
uses an owned loopback API with `SCENEWORKS_TRUST_LOOPBACK=true`, backend Candle
required, MLX disabled, `CUDA_VISIBLE_DEVICES` restricted, and the frozen isolated
HF hub. API and worker use the same binary; only the worker gets
`SCENEWORKS_WORKER_ONLY=1`. Worker PID/log are dynamic attempt evidence. The frozen deviceUUID maps to
the physical nvidia-smi index for worker telemetry/admission, while CUDA device0
is remapped by UUID; another card's peaks cannot substitute.
The bootstrap owns preparation and source/build receipts; no pin bump occurs.

Requests contain only the fixed text-to-image fields; imported/modelPath and
optional-component overrides are excluded. Native resolution uses the isolated
cache directly; offline environment flags alone are not Rust network-denial proof.
Requests put steps, guidance and `mlxQuantize` inside `advanced`, with
`mlxQuantizeExplicit=true`, count1 and enhancement disabled (enhanced prompt
bytes, if selected, are already frozen). Completion must include settled assets;
`assetWrites` alone is insufficient. The driver retains actual metrics,
worker/model/tier/steps/seed/canvas, image recipe, engine load/inference events,
and PNG bytes. A recipe's logical repo is **not** the physical artifact revision.
Native revision binding comes indirectly from exact isolated cache inventory,
source/binary hashes and worker identity; it is never claimed as a direct
runtime checkpoint-revision receipt.

First image latency includes initialization/model load from the staged cache;
prior checksum reads can warm filesystem caches, so this is not an OS-cold disk
measurement. Later images in the same
runtime are warm observations. Single-output calibration has no measured warm
latency (`null`, never zero). GPU peak includes cold model loading; PyTorch
allocated/reserved peaks are preserved. Native peaks are one-second sampled
**device** usage, not allocator peaks. Host RSS samples include owned descendants.
Native load timing includes prompt encoding until the first step; its separate
encode timing stays `null`. These metric definitions must accompany any cost
comparison. Per-render download bytes are zero under offline execution; actual
preparation download bytes belong to the bootstrap's acquisition receipt.

## Independent scoring and blind review

Before scored runs, review/freeze the rubric and thresholds in the manifest.
Defaults are proposed protocol constants, not measured model quality. Text uses
Unicode edit similarity with case/punctuation retained and whitespace collapsed;
layout uses per-element normalized-box IoU, with missing elements scored zero.
Obtain independent transcriptions and object boxes from the pixels, preferably
before exposing expected strings/boxes to the annotator. Save evaluator method,
version, output hash and annotator identity. OCR misses and ambiguous glyphs are
uncertainty, never silently replaced with requested text.

After all288 receipts exist (success or explicit reproducible failure):

```text
python scripts/evaluation/moss/evaluate.py pairs /ABS/lock.json /ABS/run /ABS/blind/pairs.json
```

Give judges only `pairs.json` and neutral `blind-images` paths. Keep the adjacent
`.private.json` mapping private. Every pair needs a judgment: `pair_id`, `winner`
(`A`,`B`,`tie`,`unavailable`), `reviewer`, `reviewer_kind` (`human` or
`machine_assisted`), `judge_version`, `notes`, `uncertainty` (`low`,`medium`,`high`),
and numeric1–5 `rubric` fields `visual_quality`, `instruction_following`,
`composition`. The frozen manifest defines anchors; judge task requirements per
case. Machine-assisted judging is explicitly labeled and retains model/version
and instruction provenance; it never becomes a claimed human rating. If either
arm failed, the judgment is `unavailable`, not a win. Missing judgments remain
pending and block reporting/eligibility.

Annotations are a JSON object keyed by successful cell ID, with `output_sha256`,
`annotator`, `method`, `tool_revision`, `text` (one transcription per expected
line), and `boxes` (elementID→normalized xyxy, or null for missing). Preserve raw
observations, assessor instructions and tool logs alongside these receipts.

```text
python scripts/evaluation/moss/evaluate.py report /ABS/lock.json /ABS/run /ABS/blind/pairs.json /ABS/judgments.json /ABS/annotations.json /ABS/report.json
```

The report retains per-scenario/per-configuration statuses, costs, independent
objective scores, blind wins, failures and uncertainty. Recommendation eligibility
requires all scored outputs,432 usable blind pairs, every successful output's
objective annotation, and all five controlled DMD step-level adapter fixtures within frozen
tolerances. It is deliberately conservative; failure rates still count against
availability and must be reported even when a scenario looks promising.
The analyst writes the use-case shortlist and tradeoffs; **Michael makes the
inclusion decision**. Source-only claims stay separate from measured quality,
reference runtime and native-feasibility conclusions. No native assessment starts
before the comparison is complete.

## CPU checks and evidence limits

```text
python3 -m pytest -q tests/test_moss_reference_evaluation.py
```

The actual CPU tensor test additionally needs torch. On the current preparation
Mac, XcodePython3.9 lacks torch. The existing Python3.14/torch2.9.1 interpreter
can use the already-installed pure-Python pytest library without installing:

```bash
MOSS_CPU_FIXTURE_RECEIPT=/ABS/fresh-cpu-receipt.json /opt/homebrew/bin/python3.14 - <<'PY'
import sys
sys.path.append('/Users/michael/Library/Python/3.9/lib/python/site-packages')
import pytest
raise SystemExit(pytest.main(['-q', 'tests/test_moss_reference_evaluation.py']))
PY
```

That run exercises the complete CPU suite without skips and can write an explicit CPU
fixture receipt: observed FP32 residual1.0001000166, unchanged BF16 backbone,
both modulation biases FP32. It uses no model weights, GPU allocation or image
quality observations. Controlled unavailable test fixtures are only tests, never
campaign failures or real evaluation evidence.

Checksums detect incomplete or changed evidence against the frozen lock and
actual artifact bytes; they are not signatures or a substitute for trusted
execution, frozen lock custody, and independent review. Changing source/weights,
corpus/settings, hardware, precision or scoring rules requires a new explicit
freeze; do not overwrite accepted evidence.

A guarded calibration runtime failure remains an actual failed/timeout attempt;
independent configurations can finish their one existing fixture. Before scored
execution, the coordinator must explicitly disposition every such failure in the
hash-bound availability control documented in `bootstrap.md`. This preserves the
original protocol and successful calibration receipts. Healthy configurations
cannot be excluded, code/acquisition defects must be fixed, and guard/input
invalidations cannot be approved away. Approved unavailable configurations emit
all48 failure records without GPU work. The matrix remains288 scored cells and
432 pairs; availability losses never become positive quality evidence.
