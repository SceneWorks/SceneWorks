# Local film harness — phase-2 evaluation and decision report (2026-09-15/16)

Epic [sc-23401](https://app.shortcut.com/trefry/epic/23401) (film harness phase 2), terminal story
[sc-23406](https://app.shortcut.com/trefry/story/23406). One finite pass of the `film-harness` on
stationary code, five configurations of the same six-shot courier film scored under the phase-1
rubric, ending in a continue / revise / stop recommendation. **This report recommends; it does not
authorize the next phase.** Same structure as the phase-1 report
([film-harness-evaluation-2026-09-14.md](film-harness-evaluation-2026-09-14.md)); the same rubric,
the same shots, the same evaluator method, so the numbers are comparable column for column.

Evidence directory (every run record, take, frame, review document, export, probe, log, socket
sample and timing log named below): `~/SceneWorks/film-harness-evidence/sc-23406/` — `README.md`
there is the index and the state of record, written after every cell. Paths below are relative to it
unless they start with `config/` or `docs/`.

**Michael should view the exports himself before acting on any verdict here.** The evaluator was an
AI agent acting as the human under the phase-1 rubric (`RUBRIC.md`, copied verbatim and declared
before any phase-2 frame was viewed), scoring from three extracted frames per take; it cannot watch
motion, and the phase-2 differences it could not score — texture, motion smear, how a face reads in
motion — are exactly the ones a person would weigh. The files to view: `exports/cell-a-v1-31s.mp4`
(the reference-conditioned film, 50 steps, 31 s, spoken lines), `exports/cell-e-turbo-v1-31s.mp4`
(the same film on the 4-step turbo recipe), `exports/cell-c-sh010-1344x768.mp4` (one shot at full
resolution, 50 steps), `exports/cell-e-sh010-1344x768-turbo.mp4` (the same shot, turbo),
`exports/cell-d-sh010-sh050-edge1024.mp4` (two shots with the reference short edge at 1024), and the
per-take frames under `frames/<cell>/` beside the plates they were conditioned on in
`pack/references/`.

## 1. Tested configuration

| item | value |
| --- | --- |
| SceneWorks code, cells (a)–(d) | `d347180963` = `origin/story/sc-23402-epic-23401-film-harness-phase-2-ref-edge` (the feature-end review passed on this head). **Stationary**: no source edit during the cells. |
| SceneWorks code, cell (e) | `de5eba32a` = `origin/story/sc-23406-epic-23401-film-harness-phase-2-turbo`, two commits on top of d347180963 (`4cd8f4f6a` the `model.loras` / `advanced.steps` plumbing and `plan.v2.turbo.jsonc`; `de5eba32a` the accelerator-offer rule). **Why a second commit**: at d347180963 the plan schema cannot express a LoRA or a step count, so the turbo cell — added to the scope by Michael mid-pass — needed the plumbing; only `sceneworks-rust-api` and `film-harness` were rebuilt (1 m 06 s), the stack was restarted from them on the same data dir after a SIGTERM teardown, and nothing in (a)–(d) was re-run. |
| inference pin | `e497db468073752568e4d3b83afbc79b0aaa9466` in both commits (the epic's pin; up from phase 1's `fff98b05`). No pin bump, no calibration / memory-matrix campaign. |
| Hardware / OS | Apple M5 Max, 128 GB unified, macOS 26.6.2; one native MLX worker (`SCENEWORKS_GPU_ID=mlx`, worker id `sc23406-eval-gpu`), one GPU process at a time, SIGTERM-only teardown. The host was **shared** for parts of the pass (a UTM VM, an rsync pair, other agents' builds; load average 16–20 during cell (a)'s SH050 and part of SH060) — flagged where it moved a number. |
| MLX | prebuilt libmlx `d5a7fc018d71` Release (`aarch64-apple-darwin-dt26.2-Release-accelerate-metal`) |
| Renderer | MiniMax-H3, family `minimax_h3`, **reference partition `minimax_h3_ref`** (`reference_to_video`, `ref2va`), tier **q4** (installed q4/q8/bf16; the plan's q4 is the catalog default), lane `mlx`, weights `SceneWorks/minimax-h3-mlx` @ `137ce668c55a20bc0935fd1cf2a3de8448abb7f4` (`q4/transformer_ref/*`), 24 fps, 5.1667 s clips |
| Turbo (cell e) | `minimax_h3_ref2v_turbo_4step` (lightx2v `minimax_h3_ref2v_turbo_4step_v0.1_bf16`), 4 NFE, video shift 12.0, audio shift 3.0; `minimax_h3_turbo_4step_v01` declared for the base partition and reached by no shot (every shot resolves to the reference partition) |
| Reference pack | `pack/references.jsonc` (id `courier-workshop-refs` v3): the sc-23403 Krea 2 Turbo q8 plates byte for byte (courier, recipient, red_parcel, workshop_location, workbench_table; `house_style` and `workshop_plate` placeholders unbound) plus the checked-in pack's sound block — three `dialogue` entries carrying `text` (Kokoro `am_michael` / `af_heart`) and the two tone beds. Copied out of sc-23403's evidence because a run writes its spoken clips into the pack's `sound/` and because the sc-23403 pack predates spoken dialogue (no `recipient_reveal_line`) |
| Plans | `config/film-harness/courier-workshop/plan.v2.jsonc` (cell a; six shots, all `reference_to_video`, four roles each, seeds 23405…23410, 576x320); derived in the evidence dir and byte-identical otherwise: `plans/plan.c.1344x768.jsonc` (`model.resolution: 1344x768`, ceilings 36000 s), `plans/plan.d.edge1024.jsonc` (`model.advanced.referenceImageShortEdge: 1024`), `plans/plan.bad.edge1023.jsonc` / `…2049.jsonc`; `config/film-harness/courier-workshop/plan.v2.turbo.jsonc` @ de5eba32a (cell e: plan.v2 + `model.loras`) and `plans/plan.e.turbo.1344x768.jsonc` |
| Review | `config/film-harness/courier-workshop/review.jsonc` v3; SenseNova-U1-8B (`sensenova_u1_8b`, MLX) through `image_vqa`; `frame_extract` on the in-process utility worker; labeled set `cell-a/review-eval/cell-a-takes.jsonc` (added to the repo by this PR as `config/film-harness/review-eval/evaluation-phase-2-cell-a-takes.jsonc`) |
| Stack | API `127.0.0.1:8797` with the in-process utility worker, fresh data dir (`data/`, receipts self-populated from the HF cache, no carried `data/cache/`), `SCENEWORKS_TRUST_LOOPBACK=1`; `stack.sh`, `stack.env`, `stack.pids`, `workers.json`, `host-capabilities.json`, `catalog-minimax_h3_ref.json`, `loras.json` |

## 2. Budgets declared before dispatch

`BUDGETS.md` was written before the first job, from the sc-23402 run records and phase 1. In short:
one reference shot at 576x320 / 4 refs @ 2048 expected 90–125 min under the plan's 10800 s cap
(cold load 3–21 min, 107–143 s/step); six shots + speech + export 9–12.5 h; review 10–15 min; one
bounded repair only if the rubric rejected a take; 1344x768 unmeasured, estimated **5–7.5 h** at
3–4.5x the step cost with a peak of 70–110 GB against the plan's 96 GB cap and a refusal or memory
stop accepted as the cell's result; edge 1024 estimated 35–60 min per shot at 0.3–0.5x; the whole
pass 18–25 h of the GPU loan. Time Machine local snapshots pruned every 2 h by the stack
(`snapshots.log`: 49 → 164 GB free after the first prune). Actual GPU time: **84 706 s = 23.5 h** of
renders — the five run records' `elapsedSeconds` in `manual/metrics.json`, 50 733.0 (a) + 23 977.5 (c)
+ 3 407.7 (d) + 4 765.5 (e, six shots) + 1 822.3 (e, 1344x768 turbo) = 84 706.0 s — plus 762 s of
review / review-eval.

**Cell (a) overran its declared budget.** 50 733 s (14 h 05 m) against `BUDGETS.md`:33's **9–12.5 h**,
and per-shot walls of 7 127–10 474 s (119–175 min) against :32's **90–125 min** — two shots inside
that range, four above it (`manual/metrics.json`). Two causes, both in the evidence and neither a
harness fault: the host's load average of 16–20 during SH050 and part of SH060 (`README.md`), which
by itself puts those two shots ~5 200 s above the four clean shots' 7 577 s mean — more than the
5 733 s by which the run exceeded the top of the budget — and the per-job checkpoint re-selection,
about 45 min of cold load spread over the six shots rather than paid once (§3.1).

## 3. Engineering correctness — did the harness do what its documents say?

### 3.1 Cell (a): six reference-conditioned shots with speech (E1, E2)

`film-harness run` over all six shots of `plan.v2.jsonc` against the generated pack —
`cell-a/run/run.json` (`run_6e9facfb1835471c90a928b58109e27d`), `cell-a/run.log`:

| shot | seed | wall (s) | load | s/step (median of progress-POST gaps) | peak (`metrics.peakMemoryBytes`) | resolved | edge | rubric |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SH010 | 23405 | 7682 | ~8 min | 140 | 24.31 GB | `minimax_h3_ref` | 2048 | 12 |
| SH020 | 23406 | 7397 | ~6 min | 139 | 24.30 GB | `minimax_h3_ref` | 2048 | 10 |
| SH030 | 23407 | 7127 | ~6 min | 137 | 24.30 GB | `minimax_h3_ref` | 2048 | 12 |
| SH040 | 23408 | 8102 | ~5 min | 138 | 24.31 GB | `minimax_h3_ref` | 2048 | 11 |
| SH050 | 23409 | 10474 † | ~9 min | 197 † | 24.30 GB | `minimax_h3_ref` | 2048 | 10 |
| SH060 | 23410 | 9888 † | ~10 min | 184 † | 24.30 GB | `minimax_h3_ref` | 2048 | 11 |

† host load average 16–20 from other processes during these two shots (`README.md`); the clean
figure for this configuration is 137–140 s/step and ~2 h per shot including the per-shot load.

Run outcome **`completed`** in **50 733 s** (14 h 05 m) of automatic work, 0 s human-requested;
6 attempts, 6 completed, 0 failed, 0 worker or harness errors, 0 rejections. Three spoken lines
synthesized by Kokoro in ~48 s before the first render (`sound/*.tts-*.wav`, adopted as project
assets and recorded in `run.json.sound[]`). Timeline 31.000 s, four tracks (picture; dialogue with
three items placed at 6.37 s, 23.27 s and 28.83 s; ambience and music 0–31 s), export `completed`:
`exports/cell-a-v1-31s.mp4`, 1138x640 letterboxed h264 + AAC 48 kHz stereo, container 31.00 s =
timeline = sidecar, decoded audio 30.997 s, `droppedAudioLayers: []`.

**Provenance, verified on every reference attempt** (`manual/metrics.json`): `resolvedModelId:
minimax_h3_ref`; `partitionReason` = "4 reference role(s); minimax_h3 declares no reference
conditioning, so the shot renders on its family's reference partition minimax_h3_ref";
`referenceImageShortEdge: 2048` (the effective value, recorded although the plan named none);
`conditioningAssets.referenceAssetIds` = four ids in the plan's role order for every shot, including
the two close-ups that list the bench before the room (SH030, SH060); the take's `model` and the
dispatched payload's `model` agree (`rawAdapterSettings.model: minimax_h3_ref`,
`minimaxH3Task: ref2va`); `model.partitionWeights` names the `q4/transformer_ref/*` row. Peak memory
came from the metrics route on every attempt (`peakMemorySource: metrics.peakMemoryBytes`).

Two behaviours worth naming. **The checkpoint is re-selected per job**
(`model_source_tier_selected` at each dispatch), so the 5–10 min load is paid on every shot, not once
per run — about 45 min of the 14 h. And **the observed peak is 24.3 GB**, not the 55.06 GB the
sc-23402 rerun recorded for the same partition at the same geometry on inference `f215cd22` (before
the packed token table and the coherence work landed on `main`); the difference is inference-side
and is noted, not explained, here.

### 3.2 Defects hit during the pass

**One open defect, on the expected user path — the planner (§3.6).** Nothing in the render, record or
assembly path: no crash, no wrong record field, no missing provenance value, no engine
refusal, no memory stop, no coherence-guard retry (`grep` of the worker log for `IncoherentLoad`,
"visibility recovered", `degenerate`, `SubmissionsIgnored` and `"level":"error"` over the whole
pass: 0, 0, 0, 0, 0 — `manual/metrics.json` `workerLogSignatures`; the guard is silent on the happy
path, as the S1 rerun also saw). One record-shape observation: an in-flight `run.json` reads
`outcome: "failed"` beside `state: "running"`; that is documented (`RunState::Running`: "`outcome`
is not meaningful yet") and every record flipped to `completed` on finish.

**The defect: the default brief does not yield a valid plan.** With `brief.jsonc` unmodified — the
expected user path, and the one where the accelerators are the default offer (§3.6) — the real
Anubis-Mini-8B planner produced **no valid plan in 2 repair rounds, nor in 5** (exit 2;
`cell-e/plan-check/plan-default.log`, `plan-default-rounds5.log`,
`planned-default*/planner-rejected.txt`). Every finding was about content — `continuityRoles` coverage
for beats naming `workbench_table`, one off-menu `targetDurationSeconds` — and none named `loras`, so
this is phase 1's "poor continuity author" finding carried into phase 2, not a fault in the turbo
plumbing. It is a defect on the expected path all the same: today the turbo default is reachable only
by a hand-authored plan. A fix is in progress on branch
`story/sc-23406-epic-23401-film-harness-phase-2-planner` (no PR open at the time of writing); **being
fixed in `story/sc-23406-epic-23401-film-harness-phase-2-planner`; result not part of this
evaluation's evidence.** Nothing was fixed in this branch; `FIXED_EXTRA` is empty.

One document-consistency finding, not a harness defect: `plan.v2.jsonc`'s SH050/SH060 prompts say
"she"/"her" and the phase-1 rubric names the recipient "a woman", while the approved `recipient`
plate is a man. The reference won every time (§4.1). The harness validates roles and files, not
prose against plates; a person supplying references would notice.

### 3.3 Offline execution

Every process of the pass ran with `HF_HUB_OFFLINE=1`, `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY` =
`http://127.0.0.1:9` (a closed port), `NO_PROXY=127.0.0.1,localhost`, no hosted-LLM variable
(`stack.env`). `lsof +c 0 -i -P -n` was sampled every 60 s (`lsof/sockets.log`). Recounted from that file:
**1 457 samples** (`grep -c '^### '`) spanning `2026-09-15T03:49:55Z` → `2026-09-16T04:10:31Z` =
**24 h 20 m (24.3 h)**, and **13 171 socket lines** (every line that is not a sample header; the file is
14 628 lines). **13 147 of them are `127.0.0.1`**; the remaining **24 are all pid 57030** — a
*different* `sceneworks-rust-api` process, not in `stack.pids` and not started by this pass, holding
**8 distinct file descriptors** (22u, 24u, 26u–31u) to `3.166.152.110:443` from the host's LAN address,
seen in exactly **three samples, at 04:01:56Z, 04:02:56Z and 04:03:56Z on 15 Sep** (8 × 3 = 24). Every
socket belonging to this stack's API, worker and `film-harness` processes is loopback. The foreign
process is recorded because the sampler matched it by name; it is not evidence about this stack. The API and worker logs carry no proxy, DNS or download error. What this
proves and does not prove is as in phase 1 §3.4.

### 3.4 Assisted review and the human loop

`review` over the six cell (a) takes: 37 questions, 72 `image_vqa` answers, 532 s, exit 3 (flags),
`realModelInference: true` in every document (`cell-a/run/reviews/`). Flags: SH020 identity (frame 1
is an empty room — true), SH040 cut (SH030 is a close-up with no wall: "pegboard: no" — false alarm,
the phase-1 close-up weakness again), SH050 parcel (out of view on frames 1–2 — true), custody (`hands`
read for the rag — false alarm) and cut (no pegboard in SH050's last frame — false alarm), SH060
sleeve (`white`/`black` for a grey sleeve — false alarm) and custody (`no` on frame 1 while the hands
enter — arguable). Human decisions: `accept-take` ×6 with reasons (12 decision-log entries,
`run.json.v3-after-decisions`). **No repair**: the declared rule grants `request-repair` only to a
take the rubric rejects, and none was (every criterion ≥ 1, every total ≥ 8), so the brief's "may do
one bounded repair on the weakest shot" had nothing to act on and 2 h of the loan was not spent.

`review-eval` on the new labeled set (`cell-a/review-eval/`): 6 cases, **37 scored, 30 correct
(81 %), 2 detections, 1 miss, 4 false alarms, 0 abstentions, 2 overclaims**, 230 s. Against phase 1's
hand-film set (34/51, 67 %, 8 false alarms, 4 overclaims) the reviewer looks better on these takes
because the takes are better — every room is the same room — not because it changed.

### 3.5 The `referenceImageShortEdge` knob (E5 plumbing)

`validate` refused `model.advanced.referenceImageShortEdge: 1023` and `2049` before any weight was
read, naming the field and the range — `referenceImageShortEdge must be from 1024 to 2048, got 1023`
(exit 2; `cell-d/validate-edge1023.log`, `…2049.log`) — and accepted 1024. The cell (d) attempt
records carry `referenceImageShortEdge: 1024`; the cell (a) records carry `2048` (effective, plan
named none).

### 3.6 Cell (e) plumbing (`model.loras`, de5eba32a)

`plan.v2.turbo.jsonc` validated against the live catalog; every attempt record carries `loras:
["minimax_h3_ref2v_turbo_4step"]`, `effectiveSteps: 4`, `turboSchedulerShift: 12.0`, and the take's
recipe `effectiveSteps 4`, `effectiveSchedulerShift 12.0`, `effectiveAudioSchedulerShift 3.0`,
`minimaxH3Turbo: minimax_h3_ref2v_turbo_4step` — the record and the payload agree. A new run was
used (compiled schema v3); nothing was resumed across the commit change. **Planner default (no GPU render; the real Anubis-Mini-8B planner, `cell-e/plan-check/`).** With
`brief.jsonc` + `"preferQuality": true` the planner wrote a **valid** plan (exit 0): no `model.loras`,
no `advanced.steps`, six `reference_to_video` shots on `minimax_h3_ref`, compiled schema 3 with
`effectiveSteps: 50` on every request — the full-step path is kept. With the unmodified brief the
draft **declared `"loras": ["minimax_h3_turbo_4step_v01", "minimax_h3_ref2v_turbo_4step"]`** — the
accelerators are the default offer and the planner took them — but no valid plan came out within 2
repair rounds, nor within 5 (`planned-default*/planner-rejected.txt`, 21 min of LLM time in all):
every finding was about **content** — beats about `workbench_table` with no shot binding it in
`continuityRoles`, and one off-menu `targetDurationSeconds` (6.8333 s); none named `loras`. That is
the phase-1 planner weakness (it does not honour the continuity contract) surfacing again, not the
turbo plumbing; the default is proven by the draft, the 50-step opt-out by a valid plan. The failure
to reach a valid default plan is recorded as an **open defect** in §3.2, a limitation in §5.2 and a
candidate in §5.4.

## 4. Research quality — did the takes tell the story?

Scored per shot under `RUBRIC.md` (identity, costume, location, parcel continuity, action
completion, cut; 0/1/2 each; sound once at sequence level), with the frame that justified each score
in `manual/SCORES.md`. One interpretation was declared there before any recipient take was viewed:
identity is scored against the **approved plate**, because holding the approved person is what
reference conditioning is for; the prompt's "she" is recorded as a document inconsistency.

### 4.1 Cell (a) — references on every shot, 50 steps (E1)

| take | id | cost | loc | parcel | action | cut | total | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SH010-a1 | 2 | 2 | 2 | 2 | 2 | 2 (n/a) | 12 | accept |
| SH020-a1 | 1 | 2 | 2 | 2 | 2 | 1 | 10 | accept |
| SH030-a1 | 2 | 2 | 2 | 2 | 2 | 2 | 12 | accept |
| SH040-a1 | 1 | 2 | 2 | 2 | 2 | 2 | 11 | accept |
| SH050-a1 | 2 | 2 | 2 | 1 | 1 | 2 | 10 | accept |
| SH060-a1 | 2 | 2 | 2 | 2 | 1 | 2 | 11 | accept |

Picture **66/72**, sound **2/2** (three buses; beds at identical levels either side of all five
cuts; each spoken line measurable inside its slot — RMS 0.049 / 0.050 / 0.044 against 0.0145 beds-only
outside — and absent outside it; `exports/cell-a-v1-31s-export-probe.json`): **68/74**.

What changed against phase 1 is exactly what phase 1 said was missing. **The room is one room**:
every wide is the `workshop_location` plate re-staged — the dog-holed bench with the vice, the tool
pegboard, the half-glazed door camera-left, the sash window right, the fluorescent battens — and
every close-up is the `workbench_table` plate. **The people are the plates' people**: the courier's
face is the courier plate's in SH010, and his blue work jacket and canvas satchel identify him in
every other shot; the recipient is the recipient plate's man in the grey apron with rolled sleeves.
**The parcel is one object**: the same small red lidded box with the paper label, from SH010's
doorway to SH060's open lid. The five cuts scored 2, 2, 2, 2, 1 — phase 1 scored 0, 1, 1, 0, 1. The
points lost are all stills-limitations or staging: a face never turned to camera (SH020, SH040), a
head cropped out of frame (SH050), an empty first 0.8 s (SH020), a parcel out of view on two frames
(SH050), a lid still held (SH060). The prompt's "soft warm glow" did not render in SH060 at 50 steps
(it did on the turbo, §4.5).

### 4.2 Cell (b) — the no-reference baseline: reused, 58/72

Not re-rendered. The decision and its evidence are in `README.md` §(b) and
`manual/inference-pin-diff-fff98b05-to-e497db468.txt` (the GitHub compare of the two pins: 15
commits, 28 files, full `mlx-gen` patches). Every change on the MLX MiniMax-H3 path between the pins
is either the load-boundary coherence guard (`materialize_accessed` / `verify_gpu_view` — a checksum
comparison that reads tensors and refuses on disagreement, transforming nothing) or the reference
short-edge knob (reference-only, defaulting to the previous constant); the base partition's DiT,
VAE, text-encoder forward, scheduler and seeds are untouched, and the MLX render is deterministic for
a seed (phase 1 §3.2 fix 1). The same `plan.jsonc`, seeds, rubric and method would reproduce the
same frames for ~2 h of the loan. Reused numbers: picture 58/72 at the capped verdict, 8 attempts
within the cap, walls 590–1000 s, peaks 13.5–14.2 GB, every room re-imagined per shot. Confidence
that the reuse is sound: **high** on the code reading; the residual risk is a different evaluator
session scoring phase 1.

### 4.3 Cell (c) — one full shot at 1344x768, 50 steps (E4)

SH010, references bound as in (a), edge 2048, installed tier q4 — `cell-c/run/run.json`
(`run_83893890b585499d8c1c5cb626438933`):

| item | 576x320 (cell a) | 1344x768 (cell c) | ratio |
| --- | --- | --- | --- |
| preflight | `validate` OK; worker admission `generic_mlx_cold_load` Resident, raw floor 67.8 GB → widened 127.8 GB vs 135.3 GB ceiling | identical (`cell-c/admission.log`) — the estimate is a per-tier floor, not geometry-aware; plan `maxMemoryGb` 96 ≥ `mlx.minMemoryGb` 64 ≤ host 128 | — |
| load | ~8 min | 13 min | 1.6x |
| s/step | 140 | **457** (~540 early, ~450 late) | **3.3x** |
| wall | 7682 s | **23 961 s (6 h 39 m)** | 3.1x |
| peak | 24.31 GB | **28.83 GB** | 1.19x |
| rubric | 12/12 | **12/12** | — |

Neither refused nor stopped: the peak sat 67 GB under the plan's cap and 35 GB under the estimate's
own raw floor, and **2.4–3.8x below `BUDGETS.md`:24's 70–110 GB expectation — which is an unexplained
miss, not a win.** That expectation was extrapolated from sc-23402's 55.06 GB for this partition at
576x320, and that baseline no longer reproduces: cell (a) peaked at 24.3 GB at the same geometry on
the current pin (§3.1). The same inference-side shift therefore explains both numbers, and it is
unexplained here; until someone reads the inference diff for it (§5.4's last candidate) the 28.8 GB
should be treated as a figure whose cause is unknown, not as evidence that full resolution is cheap in
memory. The take is the same staging as (a)'s SH010 at the same seed with the plate's room now resolved
tool by tool and the courier's face legible (`frames/cell-c/`). Tokens ~202k per step against ~96k at
576x320 — **`BUDGETS.md`:23's pre-dispatch arithmetic, not a measurement from this pass** (the
reference share is extrapolated from the 36.6k rows sc-23402 measured for two refs at edge 2048); the
measured 3.3x per step sits between the linear (2.1x) and quadratic (4.4x) scalings those token counts
imply. Export `exports/cell-c-sh010-1344x768.mp4` (1822x1024 letterboxed).

### 4.4 Cell (d) — reference short edge 1024 vs 2048 (Michael's fourth cell)

SH010 and SH050 of `plan.d.edge1024.jsonc` (576x320, same seeds) against the same shots of cell (a)
— `cell-d/run/run.json` (`run_d4eb009d52bc4dd490d8d0dd39089ff8`):

| shot | edge | wall (s) | load | s/step | peak | rubric |
| --- | --- | --- | --- | --- | --- | --- |
| SH010 | 2048 | 7682 | ~8 min | 140 | 24.31 GB | 12 |
| SH010 | **1024** | **1740** | ~3 min | **32** | **16.31 GB** | **12** |
| SH050 | 2048 | 10474 † | ~9 min | 197 † | 24.30 GB | 10 |
| SH050 | **1024** | **1635** | ~1 min | **30** | **16.30 GB** | **12** |

Per step 1024 is **4.4x cheaper** on SH010 (32 vs 140 s), the attempt 4.4x cheaper, and peak memory
8 GB lower; on `BUDGETS.md`:23's arithmetic (again derived, not measured here) the reference tokens fall
to about a quarter (~18k vs ~73k for four plates) and the whole sequence from ~96k to ~41k, and the
measured 4.4x sits near the quadratic ratio (5.5x) those counts imply — at 2048 the
reference tokens dominate the step. **Identity, costume and location fidelity: no difference the
rubric can score on either shot** (2/2/2 both ways; the plate's room, person and box in both). The
+2 on SH050 comes from parcel and action, two criteria that were out of frame / unobservable in the
2048 draw; the 1024 take is a different draw at the same seed (the conditioning tokens differ), so
per-shot deltas mix sampling with fidelity and the honest claim is "no loss visible at a 576x320
output", not "identical". Whether 1024 loses detail that matters at 1344x768 was not measured (§5.2).

### 4.5 Cell (e) — the 4-step turbo recipe (added scope; commit de5eba32a)

The same six shots, same seeds, same pack, `plan.v2.turbo.jsonc` — `cell-e/run/run.json`
(`run_db5a3f3aaeed40c19d5b4cc01d9fd8a1`):

| take | wall (s) | s/step | peak | id | cost | loc | parcel | action | cut | total | cell (a) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SH010 | 810 | 136 | 24.31 GB | 2 | 2 | 2 | 2 | 2 | 2 (n/a) | 12 | 12 |
| SH020 | 840 | 153 | 24.30 GB | 1 | 2 | 2 | 2 | 1 | 1 | 9 | 10 |
| SH030 | 890 | 167 | 24.30 GB | 2 | 2 | 2 | 2 | 2 | 2 | 12 | 12 |
| SH040 | 720 | 135 | 24.31 GB | 1 | 2 | 2 | 2 | 2 | 2 | 11 | 11 |
| SH050 | 740 | 144 | 24.30 GB | 1 | 2 | 2 | 1 | 1 | 2 | 9 | 10 |
| SH060 | 720 | 135 | 24.30 GB | 2 | 2 | 2 | 2 | 1 | 2 | 11 | 11 |

Run `completed` in **4 765 s** (1 h 19 m) against cell (a)'s 50 733 s — **10.6x** for the film; the
saving is 4 steps instead of 50. The `s/step` column is the **median of ~5 progress-POST gaps per
attempt** (`manual/metrics.json` `stepCadence`), a much thinner sample than the 50-step cells' ~56.
Those gaps also settle what load costs here: SH010's are 194 / 180 / 136 / 138 / 136 s (`api.log`), so
the pre-stepping load is ≈ **194 s** against ≈ 590 s of stepping — the **smaller** share of the 810 s
attempt, not the larger one. Per step the turbo is close to the 50-step cost at this geometry: 136–180 s
here against 137–140 s across cell (a)'s four uncontaminated shots, i.e. no large per-step change that
these gaps can show, in either direction. Peaks identical. Picture **64/72** against
66/72 (per shot 0, −1, 0, 0, −1, 0), sound 2/2 (`exports/cell-e-turbo-v1-31s-export-probe.json`):
**66/74**. The same rooms, people and props; the two points lost are SH020's set-down starting a shot
early and SH050's face never leaving the frame edge. What the rubric does **not** score and the
stills show plainly: harder contrast and crunchier edges on every turbo take, and a motion smear on
the two walking shots (SH020, SH050) that the 50-step takes do not have. One gain: SH060's "soft
warm glow" rendered on the turbo and on neither 50-step take.

**The 1344x768 pair (Michael's comparison).** The same shot as cell (c) — SH010, references bound,
edge 2048 — on the turbo recipe (`plans/plan.e.turbo.1344x768.jsonc`, `cell-e/run-1344x768/run.json`,
`run_f0571cd446084643b12648e54c5208c9`):

| item | (c) 50-step | (e) turbo 4-step | ratio |
| --- | --- | --- | --- |
| load (first pre-stepping progress gap, `api.log`) ‡ | 269 s (4.5 min) | **155 s (2.6 min)** | 0.6x |
| s/step | 457 (median of ~56 gaps) | 371 (median of ~5 gaps: 402 / 371 / 407 / 389) | 0.8x |
| wall | 23 961 s (6 h 39 m) | **1 811 s (30 min)** | **13.2x** |
| peak | 28.83 GB | **30.96 GB** | +2.1 GB (the adapter) |
| rubric | 12/12 | **12/12** | — |
| provenance | `effectiveSteps` absent (50), no `loras` | `loras: [minimax_h3_ref2v_turbo_4step]`, `effectiveSteps: 4`, `turboSchedulerShift: 12.0` | — |

‡ Both load figures are the first gap between progress POSTs, before stepping begins, counted from
`api.log`; that is a narrower quantity than the dispatch-to-step-1 figure quoted for cell (c) in §4.3's
table (13 min), and the only one derivable from the logs kept here for both jobs. The earlier "~9.5 min"
turbo load cannot be right: four step gaps of 371–407 s plus 9.5 min exceed the job's 1 811 s wall.
The 1 811 s and 23 961 s walls, the two peaks and the rubric scores are unaffected by which load
definition is used. No host-load sample was taken for cell (c)'s window, so no idle-versus-loaded claim
is made for this pair; `README.md` flags host load only for cell (a)'s SH050 and SH060.

Same room, same beat, same person and box (`frames/cell-e-1344/` against `frames/cell-c/`); a
slightly different camera (the door fully in frame, two windows), the satchel not visible in the
sampled frames, and the turbo look again — harder edges, flatter window highlights, a more saturated
jacket. On stills the rubric cannot separate them; a person watching the two exports
(`exports/cell-c-sh010-1344x768.mp4`, `exports/cell-e-sh010-1344x768-turbo.mp4`) should be the one to.

### 4.6 Side by side

| configuration | code | shots | picture /72 | sound /2 | render wall | s/step | peak | wall per accepted second |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (b) phase-1 baseline, no references, `minimax_h3`, 50 steps | c4c696d55 @ fff98b05 (reused) | 6 (+2 repairs) | **58** | 2 (v1) / 0 (FINAL) | 6 332 s over 9 attempts | ~13 | 14.2 GB | 255 s |
| (a) references on every shot, `minimax_h3_ref`, 50 steps, edge 2048 | d347180963 @ e497db468 | 6 | **66** | 2 | 50 669 s | 137–140 (197 †) | 24.3 GB | **1 634 s** |
| (c) SH010 at 1344x768, 50 steps | same | 1 | 12/12 | — | 23 961 s | 457 | 28.8 GB | 4 638 s |
| (d) SH010 + SH050, edge 1024, 50 steps | same | 2 | 24/24 (vs 22/24 at 2048) | — | 3 376 s | 30–32 | 16.3 GB | 327 s |
| (e) six shots, turbo 4-step, edge 2048 | de5eba32a @ e497db468 | 6 | **64** | 2 | 4 722 s | 135–167 | 24.3 GB | **152 s** |
| (e) SH010 at 1344x768, turbo | same | 1 | 12/12 | — | 1 811 s | 371 | 31.0 GB | 350 s |

Accepted duration is the sum of the selected takes' spans in the final timeline (31.0 s for the
six-shot cells, no trims); every attempt counts in the wall, and there were no rejected or failed
attempts in phase 2. Hands-on: **722 s = 12.0 min** of agent wall clock over **17** timed
scoring/decision phases (`manual/metrics.json` `handsOn`: `phasesSeconds` has 17 entries summing to 722,
the largest being `E.score.SH010-1344` at 111 s; `manual/TIMING.log` is the raw marks). Phase 1 recorded
12.5 min, but over a different set of phases — six shots plus two repairs, no turbo or 1344x768 cells —
so the two are not a like-for-like comparison and no trend should be read into the 0.5 min: what phase 2
shows is that scoring sixteen takes across five configurations (plus cell (a)'s decisions) cost about a quarter of an hour of
human-equivalent attention.

## 5. Decision

### 5.1 Supported conclusions (and confidence)

1. **Engineering: the render, record and assembly paths did everything their documents say on the first
   pass; the planner did not.** Six reference shots with spoken dialogue rendered, assembled and exported
   offline; every attempt record carries the resolved partition, the reason, the effective short
   edge, the reference ids in role order, the metrics-route peak and (on de5eba32a) the LoRA ids,
   effective steps and shift; the knob's range is refused by name at `validate`; the review, the
   human decisions and `review-eval` ran through the real routes. Confidence **high** — every claim
   is read off a record, a job table, a document or a decoded file in the evidence directory. Unlike
   phase 1, nothing on those paths had to be fixed to get here. **The exception is the expected user
   path into them**: with the default brief the planner yielded no valid plan in 2 or 5 repair rounds
   (§3.2, §3.6), so every plan run in this pass was hand-authored or hand-derived. That is one open
   defect, carried from phase 1's continuity-author weakness, being fixed in
   `story/sc-23406-epic-23401-film-harness-phase-2-planner`; result not part of this evaluation's
   evidence.
2. **Research: reference conditioning fixes the thing phase 1 said was broken.** With real plates
   bound on every shot the room, the people and the parcel held across all six shots and five cuts
   (66/72 against 58/72, cuts 2-2-2-2-1 against 0-1-1-0-1, no rejection), on the first attempt, with
   no human repair. The remaining losses are things stills cannot see or the model did not stage
   (a face turned away, a head cropped, a lid still held). Confidence **medium-high**: one film, one
   seed set, one pack, the same evaluator method as phase 1 but a different session; and the
   recipient's gender came from the plate, not the prompt, which shows how strongly the reference
   dominates. This is a positive finding about the tested configuration and says nothing about hosted
   systems; **no Seedance-equivalence is claimed**.
3. **Cost is the problem the references introduce.** The reference partition at 50 steps costs
   ~2 h per 5-second shot on this Mac (1 634 s of GPU per accepted second, 6.4x phase 1's 255 s).
   The token account for *why* — four references at short edge 2048 contributing ~73k of the ~96k
   tokens per step — is `BUDGETS.md`:23's pre-dispatch arithmetic, extrapolated from the 36.6k rows
   sc-23402 measured for two references; **it was not measured in this pass**, and the measured
   quantities are the walls, the per-step gaps, the peaks and the rubric scores. The
   two knobs measured here both cut it without a fidelity loss the rubric can see: **edge 1024 is
   4.4x cheaper per step** (and 8 GB lighter) on two shots, and the **4-step turbo is 10.6x cheaper
   for the film** at a cost of two rubric points, a harder look and motion smear on the walking
   shots. Confidence **high** on the *measured* numbers — walls, per-step gaps, peaks — **low** on the
   token counts, which are derived arithmetic, **medium** on "no fidelity loss" (stills, two shots for
   edge 1024, one evaluator).
4. **1344x768 is admitted and works, at 3.3x the step cost and 1.2x the memory** — 6 h 39 m for one
   50-step shot, 28.8 GB peak, rubric 12/12, no refusal, no memory stop; the admission estimate did
   not move with geometry and the observed peak sat far under every cap — **and far under the 70–110 GB
   the budget expected, for reasons nobody has established** (the sc-23402 baseline it was derived from
   no longer reproduces; §3.1, §4.3, §5.4), so the memory headroom is an unexplained observation rather
   than a demonstrated property. On the turbo recipe the same shot took 30 min at 31.0 GB for the same rubric — 13.2x — so full resolution is affordable on this Mac only through the accelerator, and a 50-step 1344x768 six-shot film would be ~40 h.
   Confidence **high** for the numbers on this hardware, **low** for anything about quality at this
   resolution beyond one shot.
5. **Assisted review is unchanged in character**: useful on identity and custody, unreliable on
   sleeves, structurally unable to judge a cut against a close-up; 81 % correct on the new set
   because the takes were better, not the reviewer. Confidence **high** on direction, **low** on the
   numbers (one set of six).

### 5.2 Limitations

- The evaluator was an AI agent, not Michael, scoring three stills per take; motion, texture and how
  a face reads in a moving shot — the very things the turbo and short-edge comparisons change — were
  judged from stills or noted outside the rubric. The exports are linked at the top.
- One film, one brief, one seed set, one generated pack, one Mac; the 1344x768 cells are one shot
  each; the edge-1024 cell is two shots; the 1024 takes are different draws from the 2048 takes at
  the same seed, so their per-shot deltas mix sampling with fidelity.
- **The turbo default was not reached through the planner in this evaluation**:
  `plan.v2.turbo.jsonc` and its 1344x768 derivative were authored by hand, because the default brief
  produced no valid plan in 2 or 5 repair rounds (§3.2, §3.6). Everything cell (e) shows is therefore
  about the recipe, not about a user getting to that recipe. Being fixed in
  `story/sc-23406-epic-23401-film-harness-phase-2-planner`; result not part of this evaluation's evidence.
- The per-step figures for the turbo cells are medians over ~5 progress gaps per attempt, against ~56
  for the 50-step cells; the token counts quoted throughout are `BUDGETS.md`:23 arithmetic, not
  measurements from this pass.
- Cell (c)'s 28.8 GB peak against a 70–110 GB expectation is unexplained, and shares its cause with the
  24.3 GB-versus-55.06 GB shift in §3.1; no conclusion about memory at full resolution should rest on it.
- Cell (b) was not re-rendered; its comparability rests on a code reading of the inference diff and
  on the determinism phase 1 measured, not on a fresh render.
- Cell (e) ran on a different commit than (a)–(d) because the plan schema could not carry LoRAs
  before it; the render path (inference pin, weights, worker) is the same.
- The host was shared for parts of the pass; two cell (a) shots are flagged as contaminated and are
  excluded from the per-step figures quoted.
- Not measured: edge 1024 at 1344x768 (whether the quarter-token reference loses detail that
  matters at full resolution); q8 and bf16 tiers; more than one seed; a real human viewing; the
  interrupt/resume and replace-take paths on the reference partition (phase 1 covered them on the
  base partition and no repair was needed here).
- The rubric's "sound" criterion was measured by RMS and tone probes, not listened to; the spoken
  lines are Kokoro's and were not judged for delivery.

### 5.3 Recommendation — **continue** (confidence medium-high)

Not *revise*: the experiment phase 1 asked for was run — real plates, a checkpoint with reference
conditioning — and it answered the question in the affirmative on the first attempt, with the render,
record and assembly paths needing no fix (the one open defect is the planner, §3.2, already being fixed
on its own branch). Not *stop*: the negative finding is cost, and two measured knobs already
bring a six-shot film from 14 h to 1 h 19 m (turbo) or would bring a 50-step film to roughly 3 h
(edge 1024) on this Mac. Continue, on the understanding that the next questions are about cost and
motion quality, which need Michael's eyes on the exports and more than one seed.

### 5.4 Candidate follow-ups (evidence-backed, **not authorized by this report**)

| candidate | evidence | dependencies / unresolved decisions |
| --- | --- | --- |
| Make edge 1024 the plan default for 576x320 work and measure it once at 1344x768 | §4.4: 4.4x per step, 8 GB lighter, no rubric loss on two shots; unmeasured at full resolution | one 1344x768 shot (~2 h at 50 steps); whether the default belongs in the plan or the catalog |
| A human viewing of the turbo vs 50-step exports before choosing a default recipe | §4.5: −2 rubric points, motion smear on walking shots, harder texture; the glow rendered only on turbo | Michael's time; possibly the 8-step file (`minimax_h3_turbo_8step`) as a middle point — not measured here |
| Keep the checkpoint resident across a run's jobs on one partition | §3.1: the checkpoint is re-selected per job, ~45 min of load across cell (a)'s 14 h; on a turbo attempt it is ~194 s of an 810 s shot (§4.5) — a visible share, though still smaller than the stepping | worker-side; whether the hot-cache work (epic 19703) already covers the reference partition |
| Make the default brief yield a valid plan (the planner's continuity contract) | §3.2, §3.6: no valid plan in 2 or 5 repair rounds on the default brief (exit 2, `cell-e/plan-check/plan-default*.log`); phase 1's same finding | **already in progress** on `story/sc-23406-epic-23401-film-harness-phase-2-planner` (no PR at the time of writing); its result is not part of this evaluation's evidence, so this row records the defect, not an unstarted idea |
| Reconcile the prompt/rubric gender with the approved plate (or regenerate the recipient plate) | §3.2, §4.1: the reference won every time | a one-line plan/rubric edit or one Krea plate; Michael's call on which is the intended recipient |
| A review question that abstains when the neighbour is a close-up, and a sleeve question that accepts "bare" | §3.4: the same two false-alarm classes as phase 1 | small review-plan change; re-measure on the two labeled sets |
| Explain the 24.3 GB vs 55.1 GB peak between inference f215cd22 and e497db468 on the reference partition — the same shift that put cell (c) at 28.8 GB against a 70–110 GB expectation | §3.1, §4.3 | inference-side reading; matters for the memory ladder and for any future budget derived from the sc-23402 baseline, not for this decision |

### 5.5 Reproducible artifacts

`~/SceneWorks/film-harness-evidence/sc-23406/README.md` indexes: `BUDGETS.md`, `RUBRIC.md`,
`BASE_COMMIT.txt`, `stack.sh` / `start-stack.sh` / `fh.sh` / `mark.sh` / `watch.sh`, `stack.env`,
`stack.pids`, `stack.log` (+ `.1-cells-a-d`), `commands.log` (every controller command with
start/end/exit), `api.log`, `worker.log`, `lsof/sockets.log`, `snapshots.log`, `build/`, `pack/`,
`plans/`, `cell-a/` … `cell-e/` (run records with snapshots, review documents, decisions, validate
logs, admission lines, NOTE.md handoffs), `frames/<cell>/`, `exports/` (five MP4s, sidecars,
timeline documents, audio probes), `manual/` (`SCORES.md`, `TIMING.log`, `metrics.py` /
`metrics.json`, `review-r1-flags.txt`, the inference pin diff, the README patch scripts).
