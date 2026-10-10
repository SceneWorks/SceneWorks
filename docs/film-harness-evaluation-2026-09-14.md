# Local film harness — POC evaluation and decision report (2026-09-14)

Epic [sc-22708](https://app.shortcut.com/trefry/epic/22708), terminal story
[sc-22715](https://app.shortcut.com/trefry/story/22715). One finite integrated pass of the
`film-harness` on the stationary feature-branch code, evaluated against the epic's acceptance
items, ending in a continue / revise / stop recommendation. **This report recommends; it does
not authorize the next phase.**

Evidence directory (every run record, timeline, MP4, frame, review document, evaluation output,
log, socket sample and timing log named below):
`~/SceneWorks/film-harness-evidence/sc-22715/evaluation-2026-09-14/` — `README.md` there is the
index. Paths below are relative to it unless they start with `config/`, `docs/` or `apps/`.

**Michael should view the exports himself before acting on any verdict here.** The evaluator was
an AI agent acting as the human under a rubric it declared before looking (`RUBRIC.md`), scoring
from three extracted frames per take; it cannot watch motion. The files to view:
`exports/hand-film-FINAL-30s.mp4` (the delivered sequence), `exports/hand-film-v1-before-review.mp4`
(the first assembly, 31 s, all three dialogue lines mixed — see §3.4 for why the final has none),
`exports/planner-SH010-SH020-13.9s.mp4`, and the per-take frames under `frames/`.

## 1. Tested configuration

| item | value |
| --- | --- |
| SceneWorks code | `feature/sc-22708-local-film-harness-poc` @ `c4c696d55` (every code story + the feature-end fixes) plus this story's fixes, committed as they were found: `489dd3225` (Step 0: re-derived harness audio items keep the editor's volume/fades), `cb22fdb11` (attempt *n* renders at seed + *n* − 1), `dde338536` (a replacement adopts the run's sound assets before re-assembling), `8b7725d3e` (SIGTERM is a stop signal). Which binary ran what: the six-shot hand-authored `run` controller was the `489dd3225` build; `review`, the decisions, both `request-repair`s and `trim` ran `cb22fdb11` (so the replacements rendered at seed + 1 but still dropped the dialogue — §3.4); the planner `plan`/`run`/`resume` and the LTX-2.5 run ran `dde338536`/`8b7725d3e` builds. The API and GPU-worker processes ran the `489dd3225` build throughout; all four fixes are client-side (`film-harness` binary / `sceneworks-core`). |
| inference pin | `fff98b05b31e24423c517f05553d1379bf24fc3a` — **unchanged**. Inference is out of scope for this epic: no pin bump, no calibration / memory-matrix / VRAM campaign. |
| Hardware / OS | Apple M5 Max, 128 GB unified memory, macOS 26.6.2 (25G83); one native MLX worker (`SCENEWORKS_GPU_ID=mlx`, worker id `sc22715-eval-gpu`), one GPU process at a time, SIGTERM-only teardown |
| MLX | prebuilt libmlx `d5a7fc018d71` Release (`aarch64-apple-darwin-dt26.2-Release-accelerate-metal`) |
| Renderer A | MiniMax-H3 (`minimax_h3`), tier **q4**, lane `mlx`, weights `SceneWorks/minimax-h3-mlx` @ `137ce668c55a20bc0935fd1cf2a3de8448abb7f4` (`q4/transformer/*`), 576x320, 24 fps, 5.1667 s clips (hand plan) / 6.58–10.13 s (planner plan) |
| Renderer B | LTX-2.5 (`ltx_2_5`), tier **q4**, lane `mlx`, weights `SceneWorks/ltx-2.5-mlx` @ `081658ce6886cacba20817ce0359bbefef706ff2` (`distilled/q4/*`, `dev/q4/*`), 768x512, 25 fps, 6 s clips |
| Planner LLM | `TheDrummer/Anubis-Mini-8B-v1` (`prompt_refine_anubis_8b`, bf16 MLX) through `POST /api/v1/prompts/refine` |
| Reviewer VLM | SenseNova-U1-8B (`sensenova_u1_8b`, MLX) through `image_vqa`; `frame_extract` on the in-process utility worker |
| Documents | `config/film-harness/courier-workshop/{plan.jsonc, plan.ltx25.jsonc, brief.jsonc, references.jsonc, review.jsonc}` at the commit above (placeholder plates and tone clips); labeled sets `config/film-harness/review-eval/{labels.jsonc, real-takes.jsonc}` plus the two written here (§3.6) |
| Stack | API `127.0.0.1:8791` with the in-process utility worker (`frame_extract`, `timeline_export`), fresh data dir seeded with `data/models/` receipts only (no `data/cache/`), `SCENEWORKS_TRUST_LOOPBACK=1`; `stack.sh`, `stack.env`, `workers.json`, `host-capabilities.json` |

## 2. Budgets declared before dispatch

`BUDGETS.md` was written before the first job. In short: an H3 shot 11–14 min expected under the
plan's 2700 s / 96 GB caps, one automatic attempt (`maxAttemptsPerShot: 1`); at most two human
replacements on the hand film; planner ≤ 1200 s per decode, ≤ 2 repair rounds, 24 GB; review
1200 s / 16 GB per the review plan; LTX-2.5 20–45 min per shot allowed, a refused or errored cell
INCONCLUSIVE and not retried; whole-evaluation GPU ceiling 6 h; the full six-shot planner film
conditional on ≥ 60 min remaining under the ceiling (it was not run — §3.5).

GPU time is the sum of job wall clocks in the run records and `compiled.json.planner`; hands-on
time is the agent's own wall clock between `mark.sh` start/stop marks (`manual/TIMING.log`).

## 3. Engineering correctness — did the harness do what its documents say?

### 3.1 The hand-authored film (acceptance 1)

`film-harness run` over all six shots of `plan.jsonc` (MiniMax-H3 q4, sound fixture, budgets as
declared) — `hand-film/run/run.json`, `hand-film/run.log`:

| shot | attempt | wall | peak | seed | result |
| --- | --- | --- | --- | --- | --- |
| SH010 | 1 (auto) | 670 s | 13.53 GB | 22710 | selected, accepted |
| SH020 | 1 (auto) | 660 s | 14.21 GB | 22711 | rejected by the human (§4) |
| SH020 | 2 (`request-repair`) | 630 s | 14.21 GB | 22712 | selected, accepted with a 1.0 s head trim |
| SH030 | 1 (auto) | 630 s | 13.49 GB | 22712 | selected, accepted |
| SH040 | 1 (auto) | 710 s | 14.24 GB | 22713 | selected, accepted (continuity defect noted) |
| SH050 | 1 (auto) | 590 s | 13.47 GB | 22714 | rejected by the human |
| SH050 | 2 (`request-repair`) | 790 s | 13.50 GB | 22715 | rejected by the human; cap reached; stays in the cut, record says rejected |
| SH060 | 1 (auto) | 650 s | 13.53 GB | 22715 | selected, accepted |
| SH040 | 2 (`replace-take`, outside the cap — §3.2) | 1000 s | 13.82 GB | 22714 | selected, accepted (9/12) |

Run outcome `completed` in 3924 s of automatic work (`elapsedSeconds`) and 1437 s of
human-requested work (`humanRequestedElapsedSeconds`: two replacements + one re-export) at the
capped verdict; 2448 s human-requested after the out-of-cap SH040 replacement. 9 attempts, 9
completed, 0 failed, 0 worker or harness errors.

**Which file says `completed`.** The snapshots `hand-film/run.json.v1-after-run` through
`run.json.v5-before-sh040-replacement` all read `outcome: "completed"`, `stop: null` — that is the
run's verdict, and it is what the sentence above is about. The record a reader opens today,
`hand-film/run/run.json` (== `run.json.v6-after-sh040-replacement`), reads `outcome: "failed"`,
`stop.reason: "attempts_exhausted"`, `resumable: false`. **That is a defect of the binary that
wrote it, fixed in this PR** (§3.2 fix 5), not a second verdict: the out-of-cap `replace-take` on
SH040 succeeded, and `finish_replacement` fell through to the run-level classifier, which
re-derived the outcome over EVERY selected shot. SH050 has `selectedAttempt: null` — the human
rejected both its takes — so "all rendered" was false and `completed` was overwritten by a stop
about a shot the replacement never touched. v5 and v6 carry identical SH050 state, which is the
proof that nothing about SH050 changed and only the verdict did. On the fixed binary that
replacement leaves `outcome: "completed"` with no stop, and only `export.stale` moves. No file in
the evidence directory was rewritten after the fact; v6 is left exactly as the pass produced it.

Timeline `timeline_7d8ba8c6d9be4fd3d7dd4675aaa681ec`
(`exports/hand-film-timeline.json`): 6 picture items in plan order, one `trim` edit (SH020
`sourceIn 0 → 1.0`), duration **30.000 s** (the rubric's floor; the plate head of SH020-a2 is
~1.5 s, so 0.5 s of dissolve remains — §4). Three exports: `hand-film-v1-before-review.mp4`
(31.0 s, first assembly), `hand-film-v2-after-repairs-31s.mp4`, `hand-film-FINAL-30s.mp4` (1138x640
letterboxed h264 + AAC 48 kHz stereo, container 30.00 s = timeline = sidecar, decoded audio
29.995 s, `droppedAudioLayers: []`).

Retained in the record and the project: every plan / pack / compiled hash, all 8 takes with
provenance (job id, seed, recipe, weights revision, backend), 3 rejections with reasons, 21
decision-log entries, 8 review documents, the three timeline versions and their edits, both bed
clips and the three line clips as project assets, the export jobs and their sidecars.

### 3.2 Defects hit during the pass, fixed in this PR (each with a test that the mutant fails)

| # | commit | what the evaluation showed | fix |
| --- | --- | --- | --- |
| 0 | `489dd3225` | (feature-end re-review minor) `merge_harness_audio_track` reset the editor's per-item `volume` / fades on every re-assembly | a fresh harness item merges onto the saved item of the same role + shot and keeps those fields; unit test |
| 1 | `cb22fdb11` | `replace-take` dispatched the compiled request's seed unchanged, and the MLX render is **deterministic for a seed**: today's SH010 and the sc-22715 sound-smoke SH010 (same seed 22710, four hours apart) are pixel-identical in all three sampled frames (`manual/determinism_check.txt`, mean abs diff 0.0 per channel). A replacement would have re-rendered the very take it rejected. | attempt *n* dispatches seed + *n* − 1, stamped into `advanced.filmHarness.seed` and read back from the take's recipe (SH020-a2 = 22712, SH050-a2 = 22715 in the record); unit test in `film_compile` |
| 2 | `dde338536` | after the first `request-repair`, the saved timeline's `track_dialogue` had **0 items** (3 before): `replace_take` re-assembled without `ensure_sound`, so the merge re-derived an empty dialogue track over the saved one. The beds survived only because a bed track with no clip is skipped and then kept as "not the harness's". Consequence: `hand-film-FINAL-30s.mp4` mixes ambience + music but **no dialogue lines**; `hand-film-v1-before-review.mp4` has all three (`hand-film/export-probe-v1.json`: 400 Hz line 0.0793 inside SH020's slot, 500 Hz 0.0793 / 0.0635 inside SH050's and SH060's, 0.0 outside; beds 0.00795 / 0.00631 identical either side of all five cuts). | `replace_take` calls `ensure_sound` (adopts the recorded clips, uploads nothing) before `assemble_timeline`; integration test over the sound-carrying fixture through the real routes |
| 3 | `8b7725d3e` | `kill <pid>` (SIGTERM) of the `run` controller took the crash path: the process died in 1 s with the record `running`, `stop: null`, the in-flight job unmentioned (`planner/run.json.after-sigterm`) — only SIGINT was listened for | SIGINT and SIGTERM are both stop signals, registered up front; test raises SIGTERM at the listener |

### 3.2b Defects found by the adversarial review OF this report, fixed in this PR

Fix 2 above was an incomplete fix for its own defect, and reading the evidence for the rest of the
report turned up five more. Each has a test that fails on the code the finding is about.

| # | what the evidence showed | fix |
| --- | --- | --- |
| 4 | Fix 2 restored the dialogue track but not every line on it. The picture track is MERGED onto the saved sequence (so a shot whose take was rejected keeps its item and stays in the cut) while the audio tracks were re-derived from `selected_takes()` — which no longer names that shot. The evaluation's own re-assembly shows it: `run.json.v1` `track_dialogue` **3** items → `v3` **0** (fix 2's defect) → `v6` **2**, and `hand-film-FINAL-with-dialogue-30s.mp4` is permanently missing SH050's line while SH050's picture sits in the cut at 19.67–24.83 s. Reachable by any successful `replace-take` on an unrelated shot after a `reject-take`. | the harness's dialogue items are derived from the merged PICTURE's shot list (`order`), not from the selection, so a line survives exactly as long as its shot is on screen; a line whose shot HAS left the cut is still dropped by `relayout_timeline`. Integration test: reject SH050, replace SH060, both lines survive and SH050 is not re-selected |
| 5 | The record a reader opens says `failed` / `attempts_exhausted` while the run's own snapshots say `completed` (§3.1). A successful `replace-take` on SH040 routed through `finish_replacement` → `finish()`, which re-derives the RUN's outcome from every selected shot — so a shot the human had rejected flipped the whole run's verdict because a different shot was replaced. | `finish_replacement` leaves a `completed` run's verdict alone when the replacement it was asked about landed (the `edit_timeline` rule, stated for the generation side); only `export.stale` moves. A landed replacement whose re-export failed records `export_failed` (resumable), not a stop about another shot. Test: completed run → reject SH050 → `replace-take` SH060 → outcome stays `completed`, SH050 still `selectedAttempt: null` |
| 6 | `StopSignals::next` built a fresh `tokio::signal::ctrl_c()` inside its `select!`, so the SIGINT stream lived for one call and was dropped whenever the SIGTERM arm won. A second Ctrl-C arriving between the first `next()` returning and the second being awaited was delivered to nothing — the operator's only way out of a 45-minute render, swallowed. (True for SIGTERM, which was held in the struct; false for SIGINT, which the doc comment claimed.) | a `signal(SignalKind::interrupt())` stream is held beside `terminate` for the life of the watcher. Test raises SIGTERM, observes it, raises SIGINT while nothing is awaiting, and requires the next call to return it |
| 7 | Registering SIGTERM installs a process-wide tokio handler that is never uninstalled, and the watcher was `abort()`ed as soon as the command returned — so for the tail of the process (printing the record, the last flush) a plain `kill <pid>` was swallowed and did nothing at all. | the watcher is told the run is over through a oneshot instead of being aborted, and goes on watching: a stop signal in the tail terminates the process with the code a shell reports (130 / 143). Test drives `watch_stop_signals` with an injected exit and a SIGTERM after completion |
| 8 | Attempt offsets of `+1` overlapped the plan's per-shot seed stride: the shipped fixture seeds its shots 22710…22715, and this run dispatched **22712, 22714 and 22715 twice each** — SH020-a2 = SH030-a1 = 22712, SH040-a2 = SH050-a1 = 22714, SH050-a2 = SH060-a1 = 22715. A dispatched seed did not identify the render it came from. | attempt *n* dispatches `seed + (n − 1) × 1000` (`film_compile::ATTEMPT_SEED_STRIDE`, documented in `docs/film-harness.md`); per-attempt seed provenance in the record is unchanged. Unit test asserts the stride, not just the first two attempts |
| 9 | `reject-take` left the shot `outcome: "rendered"` with `selectedAttempt: null` and nothing anywhere saying that it stays in the sequence carrying the rejected take — unlike the failed-replacement path, which writes exactly that note. A reader of the record could not tell what the cut shows for SH050. | the reject path writes the same explicit decision note; test asserts it names both what the cut shows and the command that changes it |
| — | `config/film-harness/review-eval/evaluation-2026-09-14-takes.jsonc` and `…-planner-takes.jsonc` (added by this PR) were loaded by no test | a test enumerates every `*.jsonc` in that directory, parses it, resolves the `reviewPlan` it declares, validates it, and checks every frame path is a plain name under the media root; the four checked-in sets are asserted present so an empty enumeration cannot pass |

Fixes 4 and 5 are **not** reflected in any exported file in the evidence directory, and cannot be
without a GPU: `assemble_timeline` — the only code that re-derives the harness's sound items — is
reached from `run`, `resume` and a successful `replace-take` only. The hand film's record is
`finished` and not resumable (whether it reads `failed`/`attempts_exhausted` as it does today or
`completed` as the fixed binary would write), so `resume` refuses it; `run` and `replace-take` each
dispatch a `video_generate` job. The `--export` re-layout path (`trim` / `reorder` / `swap-take`
with `--export`) reads the SAVED timeline document and never re-derives sound, so it re-exports the
two-line sequence unchanged — it cannot restore an item that is already absent. Fix 4 is therefore
proved by the integration test through the real routes rather than by a new MP4, and
`hand-film-FINAL-with-dialogue-30s.mp4` stays what the pass produced: two of the three lines.

A completed run is not resumable and an edit only re-lays what is saved, so nothing short of
another render re-assembles the hand film's dialogue after fix 2. The declared cap of two
quality replacements was kept and the capped result (`hand-film-FINAL-30s.mp4`, no lines) is the
verdict of record. **Outside that cap**, after the verdict was recorded and with GPU budget left,
one more `replace-take --export` was run on SH040 with the fixed binary — for the engineering
purpose of exercising fix 2 on the real film (and SH040 was the take a human editor would replace
next, the deviant tiled room). It rendered SH040-a2 (seed 22714, 1000 s, 13.82 GB; rubric 9/12,
same as a1, now in SH010's room family, door left open at the end), re-assembled with **2 dialogue
items** (SH020's and SH060's lines; SH050 has no selected take so no line is derived for it) and
re-exported `hand-film-FINAL-with-dialogue-30s.mp4`: `hand-film/export-probe-v6-sh040.json` shows
the 400 Hz line at 0.0793 inside SH020's slot, the 500 Hz line at 0.0635 inside SH060's, 0.0
outside both, beds continuous at every cut. Fix 2 verified on real data; the film's scores are
reported with and without this take (§4.1). Total attempts on the hand film: 9.

### 3.3 Interrupt / resume and isolated take replacement (acceptance 2)

Interrupt: the planner-plan run (`planner/run`) was sent SIGTERM at 10:24:50Z with SH010 complete
and SH020's job at 30 % (`planner/interrupt.log`). On the then-current binary this was a crash,
not a cancel (fix 3 above); the record it left names the running job (`planner/run.json.after-sigterm`).
`film-harness resume` at 10:25:41Z: "resumed with 5948s of the plan's 7200s budget left", **adopted
`job_b9472b56` at 32 %**, polled it to completion (1505 s), assembled and exported. The job table
before and after (`planner/jobs.before-resume.json`, `planner/jobs.after-resume.json`) holds
exactly two `video_generate` jobs for the run's idempotency keys — `SH010:a1` and `SH020:a1` —
before and after: SH010's take was reused, **no job was enqueued twice**, the interrupted render
was not wasted. Run outcome `completed`, 2433 s automatic, timeline 13.875 s, export completed.

Isolated replacement: the two `request-repair`s on the hand film are `replace-take` with the
review's flags folded in. `manual/isolation-check.txt` hashes every shot's attempts / takes /
conditioning / intended block before (`run.json.v2-after-decisions`) and after
(`run.json.v3-after-repairs`) the replacements: SH010, SH030, SH040, SH060 **unchanged** (same
digest, same single job each); only SH020 and SH050 changed (1 → 2 attempts, one new job each).
Every video job the API holds for the run maps to exactly one idempotency key (8 keys, 8 jobs,
0 duplicates). The dependents (SH030 on SH020, SH060 on SH050) were flagged `needsReview` and
never re-rendered; the flags were cleared by `accept-take` after a human look.

### 3.4 Offline execution (acceptance 2)

Every process of the evaluation ran with `HF_HUB_OFFLINE=1`, `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY`
= `http://127.0.0.1:9` (a closed port), `NO_PROXY=127.0.0.1,localhost`, and no hosted-LLM
variable (`stack.env`, `ps eww` of the API pid in the README). `lsof +c 0 -i -P -n` was sampled
every 60 s for the API, the worker and every `film-harness` process (`lsof/sockets.log`): 315
samples over the 5 h 17 m the stack was up, thousands of socket lines, **every one loopback**
(`127.0.0.1:8791` and the worker's/harness's loopback client ports); no other destination appears
once. The API and worker logs contain no
proxy, DNS or download error, i.e. nothing tried to leave. What this proves: planning (7 LLM
decodes), 12 renders on two models, 8 reviews (100 VQA calls), 3 exports and every controller
command completed with no reachable network. What it does not prove: that the binaries contain no
network code path (a request would have failed against the closed proxy and been logged, and none
was), or anything about provisioning — the weights and receipts were on disk before the run.

### 3.5 Planner (acceptance 3, engineering half)

`film-harness plan` over `brief.jsonc` (`planner/planned/`, `planner/plan.log`): 6 shots, **0
repair rounds**, **351 s** over 7 `prompt_refine` jobs (1 draft + 6 rewrites), planner peak
**16.87 GiB** (`compiled.json.planner`, budget 24 GB), no planner error, `validate` accepted the
result against the live catalog. Rendering: SH010 1250 s / **28.90 GB** peak (6.58 s clip), SH020
1505 s / 14.06 GB (7.29 s) — the longer clips cost ~190 s per clip-second against ~128 s for the
5.17 s hand shots, and SH010's peak is twice the hand film's. SH030–SH060 of the planner plan were
rendered **after** the required work, as a separate run (`planner/run-rest`: 1020 / 1831 / 1741 /
1461 s, peaks 13.8–14.9 GB, export 34.1 s, `completed`), once the LTX-2.5 cells and the review
measurements were in and the ceiling allowed (BUDGETS.md step 8 was conditional from the start).
Two things about reviewing the planner's takes: the default `review.jsonc` lookup found nothing
beside the generated plan (`--review-plan` is the documented answer), and the shipped review
plan then **refused the planner's plan outright** — its `acrossCut` questions need a `dependsOn`
edge and the planner wrote no chains, so its plan has no edges (`planner/review-then-ltx.log`).
The takes were reviewed with `planner/review.noedges.jsonc` (the shipped plan minus the five cut
questions); `review-eval` was unaffected because the labeled set carries its own neighbour frames.
Documented in `docs/film-harness.md`; the planner emitting no continuity edges is itself a finding
(§4.2).

**One worker failure** in the whole pass: at 13:17Z, after ~5 h of continuous GPU work, the MLX
worker's first `image_vqa` call after the last planner render died with
`kIOGPUCommandBufferCallbackErrorSubmissionsIgnored` (Metal ignoring the process's submissions
after prior GPU errors — process-scope poisoning; `worker.log.1-before-restart`). Both stack
processes were stopped through the stack's own SIGTERM path (the worker was idle), restarted on
the same data dir (`stack.log.2`, `restart-and-finish.log`, `finish2.log`), and the three
interrupted review steps re-ran cleanly. No render was affected, nothing was lost, and the
harness reported the failure as a transport error naming the job (exit 1) rather than recording
an `unobserved` verdict.

### 3.6 Assisted review (acceptance 3, engineering half)

`review` over all 8 hand-film takes: 100 `image_vqa` calls in 470 s (first call carries the cold
load), `realModelInference: true` in every document, no timeouts, no review stop. `review-eval`
totals per set are in §4.4; the two new labeled sets are
`config/film-harness/review-eval/evaluation-2026-09-14-takes.jsonc` (8 hand-film takes) and
`evaluation-2026-09-14-planner-takes.jsonc` (2 planner takes), media under `frames/`.

### 3.7 LTX-2.5 (acceptance 4, engineering half)

`validate` accepted `plan.ltx25.jsonc` against the live catalog (q4 installed, every value on the
entry's menus). The two-shot run (`ltx/run`) is reported cell by cell in §4.5.

## 4. Research quality — did the takes tell the story?

Scored per shot under `RUBRIC.md` (identity, costume, location, parcel continuity, action
completion, cut; 0/1/2 each; sound once at sequence level), with the frame that justified each
score in `manual/SCORES.md`. Decision rule declared in advance: reject on any 0 or a total < 8; at
most two replacements; a rejected replacement stays rejected.

### 4.1 Hand-authored film — rubric per take

| take | id | cost | loc | parcel | action | cut | total | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SH010-a1 | 2 | 2 | 2 | 2 | 2 | 2 (n/a) | 12 | accept |
| SH020-a1 | 1 | 2 | 1 | 1 | 2 | 0 | 7 | reject |
| SH020-a2 | 1 | 2 | 2 | 1 | 1 | 1 | 8 | accept + 1.0 s head trim |
| SH030-a1 | 2 | 2 | 2 | 2 | 2 | 1 | 11 | accept |
| SH040-a1 | 1 | 2 | 1 | 2 | 2 | 1 | 9 | accept (third room) |
| SH050-a1 | 2 | 2 | 2 | 1 | 2 | 0 | 9 | reject (cut 0) |
| SH050-a2 | 2 | 2 | 2 | 1 | 1 | 0 | 8 | reject (cut 0); cap reached |
| SH060-a1 | 2 | 2 | 2 | 1 | 2 | 1 | 10 | accept |
| SH040-a2 (outside the cap, §3.2) | 1 | 2 | 2 | 2 | 1 | 1 | 9 | accept |

Final cut at the capped verdict (SH010, SH020-a2, SH030, SH040-a1, SH050-a2*, SH060): picture
12 + 8 + 11 + 9 + 8 + 10 = **58/72**; sound **2/2 on v1** (three buses, beds continuous across
every cut, each line inside its slot only), **0/2 on FINAL** (dialogue bus empty — fix 2, an
engineering defect, not a model result). *SH050-a2 is in the cut but rejected in the record. With
the out-of-cap SH040-a2 in place of a1 the picture total is unchanged (9 for 9) and the sound on
`hand-film-FINAL-with-dialogue-30s.mp4` is **1/2** (both beds and two of the three lines; SH050's
line is not derived for a rejected take).

What the takes got right, every time: the courier's blue jacket, the red parcel's presence and
colour, the recipient's grey apron and rolled sleeves, and the **action of each shot** (door
opens, parcel set down and hands withdrawn, courier leaves and the door closes, parcel opened) —
7 of 8 takes reached their `endState` or nearly so. What they got wrong, structurally: **the room
is a different workshop in almost every shot** (SH010 panelled + pegboard; SH020-a1 a lumber shop;
SH020-a2 a pegboard room, closer framing; SH040 green tiles; SH050-a1 whitewashed brick; SH050-a2
another pegboard room), and **the parcel changes object** (plain box → flat envelope → box → ribboned
gift box → paper-wrapped) even while staying red. Both come from the same place: the reference
pack's plates are flat placeholders, the `image_to_video` shots (SH020, SH040) condition on a
brown plate that carries no room, and the MiniMax-H3 base checkpoint has no reference conditioning
— so every shot re-imagines the set from prose alone. The rubric's cut criterion, scored against
the immediate predecessor, then penalises whichever shot follows the deviant one (SH050-a2 lost its
cut score to SH040's tiled room while itself being in SH010's room family); a human editor would
more likely replace SH040 next.

Yield: 6 of 8 attempts accepted (75 %); 2 replacements bought one improvement (SH020: 7 → 8, and
the room family now matches SH010) and one non-improvement (SH050: 9 → 8, different faults). At
seed + 1 a replacement is a genuinely new draw, not a re-render — but with no continuity
conditioning it is a new draw of the *room* too.

### 4.2 Planner plan vs hand-authored plan (acceptance 3)

`manual/planner-vs-manual.md` has the full side-by-side (prompts, states, refined text).

| dimension | hand-authored | planner (0 repair rounds, 351 s) |
| --- | --- | --- |
| beat coverage | 6/6 | 6/6, every shot carries its `beatId`; refused nothing |
| running time | 31.0 s (6 × 5.1667) | 48.0 s (5.9–10.1 s per shot) — inside the brief's 30–60 s |
| conditioning | 4 t2v + 2 i2v on the plate | 6 t2v; no keyframes, no chains, no `dependsOn`, no seeds |
| roles | 4–5 per shot incl. `house_style` | exactly the beat's required roles, `house_style` never |
| sound | 3 dialogue clips, 2 beds | none placed (the brief carries no sound block) |
| prose | static framings, layout stated (door camera-left, bench centre) | "the camera follows …" in every shot; layout never stated; SH010's prompt closes the door behind the courier, contradicting SH020's start; SH020's prompt omits the jacket colour |
| rendered shots (rubric) | SH010 12, SH020 7 → 8 (all six with the human loop: 58/72) | P-SH010 **10** (three framings inside one take), P-SH020 **4** (black jacket, different room, parcel set down early — costume 0, cut 0); the four rendered afterwards: P-SH030 5 (dark sleeve, gift-wrapped box, bench by a window), P-SH040 6 (black hoodie, cinder-block store), P-SH050 **9** (the best planner take, in yet another room), P-SH060 5 (a ribboned gift box in a room with fairy lights — the glow beat itself was perfect); **39/72 with every cut scored 0**, no human loop |
| render cost | 128 s per clip-second, 13.5–14.2 GB | 181 s per clip-second over six shots (8810 s for 48.0 s), peaks 13.8–14.9 GB and one 28.9 GB |

The planner is competent at the contract (all beats, valid schema, sensible shot list, first
draft) and poor at what the contract does not force: it declares no continuity mechanism (no
keyframes, chains, edges, seeds, sound), writes moving-camera prose that MiniMax-H3 turns into
several framings per take, and drops the details the pack fixes (jacket colour, door side). Its
plan would cost ~55 % more GPU per second than the hand plan and its two rendered shots scored
14/24 against the hand plan's 20/24 for the same beats. Planner errors: 0 refusals, 0 repair
rounds, 0 schema errors; 2 content errors (door closed in SH010's prompt; SH020's endState
contradicted by its own take's early set-down).

### 4.3 Sound

Measured off the exports' own bytes (`hand-film/export-probe-v1.json`, `export-probe-final.json`):
two streams, container = timeline = sidecar duration, 100 Hz ambience and 250 Hz music at identical
levels 0.4 s either side of all five cuts (one continuous bed each, no restart), each placed line
measurable inside its slot and absent 0.3 s outside it, no generated clip audio (policy `mute`).
The final export lacks the lines for the reason in §3.2 (#2).

### 4.4 Assisted review measured (acceptance 3)

Review is **assistive, not QA**: nothing it reports approves, rejects, conditions or re-renders
anything; every human decision above was recorded through the controller after looking at the
frames. On this film the reviewer's flags agreed with the human on the two rejections (SH020-a1:
character_identity + cut_continuity; SH050-a1: cut_continuity) and disagreed on the costume of
SH010 (navy read as black — the known low-light limitation), SH060 (bare forearms read as white
sleeves), and SH050-a1's custody (a reaching hand read as holding).

`review-eval` per set (detections / misses / false alarms / abstentions / overclaims; the
per-topic tables are in `review-eval/*/review-eval.txt`):

| set | backend | cases / scored | correct | detections | misses | false alarms | abstentions | overclaims | wall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `labels.jsonc` (synthetic plates) | scripted, no weights | 11 / 69 | 4 | 0 | 13 | 0 | 52 | 0 | 0.1 s |
| `real-takes.jsonc` (sc-22710 takes) | SenseNova-U1-8B | 2 / 13 | 11 | 3 | 1 | 1 | 0 | **0** | 140 s |
| `evaluation-2026-09-14-takes.jsonc` (8 hand-film takes, new) | SenseNova-U1-8B | 8 / 51 | 34 | 6 | 3 | 8 | 2 | **4** | 337 s |
| `evaluation-2026-09-14-planner-takes.jsonc` (2 planner takes, new) | SenseNova-U1-8B | 2 / 13 | 10 | 3 | 0 | 0 | 0 | **3** | 76 s |

The synthetic row measures the scripted CPU stub (it abstains on nearly everything), not a
model — it is there because the story asked for the shipped sets, and the docs already say a real
model on flat plates measures nothing. The real-takes row reproduces the sc-22714 measurement
exactly (same 13 verdicts). The new hand-film set is the informative one: 34/51 correct (67 %),
6 detections (the SH020-a1 late entry and cut, SH020-a2's plate head, SH050-a2's unfinished
action), **8 false alarms** (SH010 jacket navy→black and custody; SH040 room→"kitchen" on one
frame and door state; SH050 custody ×2 and apron→"shirt"; SH060 bench), 3 misses (SH020-a1's
missing parcel on frame 1, and the SH040→SH050 cut on both SH050 takes, where the reviewer
answered "pegboard: yes" on SH040's tiled wall of hooks), and **4 overclaims** — SH020-a2 jacket
"black" and custody "surface" read off frames that show neither, SH040's cut judged against a
close-up, SH060's bare forearms read as "white" sleeves. Per-question detail:
`review-eval/hand-film-takes/review-eval.txt`. On the planner takes the reviewer caught all three
labeled faults of P-SH020 (black jacket, parcel absent from a frame, parcel on the surface) and
overclaimed three times — a room verdict from a close-up of a parcel, a door state it could not
see, and a cut judged against that close-up. Across the two new sets: 64 scored, 44 correct
(69 %), 9 detections, 3 misses, 8 false alarms, 2 abstentions, 7 overclaims.

### 4.5 MiniMax-H3 vs LTX-2.5 (acceptance 4)

Two declared representative shots: SH010 (`text_to_video`, the establishing beat) and SH020
(`image_to_video` on the plate, the approach beat), rendered from `plan.ltx25.jsonc` at LTX-2.5's
own menus (768x512, 25 fps, 6 s, q4) and from `plan.jsonc` at MiniMax-H3's (576x320, 24 fps,
5.1667 s, q4). **Different resolutions, durations and tiers: no ranking is made from these cells.**

| cell | model / settings | wall | peak | attempts | rubric | what the frames show |
| --- | --- | --- | --- | --- | --- | --- |
| SH010 t2v | MiniMax-H3 q4, 576x320, 24 fps, 5.17 s | 670 s | 13.53 GB | 1 | **12/12** | the plan's room and action, exactly |
| SH010 t2v | LTX-2.5 q4, 768x512, 25 fps, 6.12 s | **200 s** | 24.89 GB | 1 | **12/12** | a rustic pegboard workshop, door camera-left, the courier in a blue jacket walks in with the red box; locked off, one framing |
| SH020 i2v (first frame = flat plate) | MiniMax-H3 q4, 576x320, 24 fps, 5.17 s | 660 s (a1) / 630 s (a2) | 14.21 GB | 2 | 7/12 → **8/12** | the prompt's courier and parcel in a re-imagined workshop; the plate visible for ~1.5 s at the head of a2 |
| SH020 i2v (first frame = flat plate) | LTX-2.5 q4, 768x512, 25 fps, 6.12 s | 320 s | 24.89 GB | 1 | **0/12** | **off-prompt**: three unrelated framings of people talking outdoors (letterboxed); no workshop, parcel, courier or blue jacket. The job completed normally with the same 274-char prompt and a `sourceAssetId` for the plate (`ltx/sh020-job-payload.json`). Whether the q4 LTX-2.5 i2v path dropped the prompt or the placeholder plate dominated is an inference-side question, out of scope here. |

Both LTX cells ran (`outcome: completed`, no refusal — the validate-time menus held; `ltx/run/run.json`);
neither is inconclusive. Read as two cells, not a ranking: on the text-to-video establishing shot
LTX-2.5 q4 produced a rubric-perfect take in 30 % of the H3 wall clock at 1.8x the memory and a
larger frame; on the image-to-video approach shot conditioned on a placeholder plate it produced
nothing of the prompt, where H3 kept the subject and the prop. Peak memory was measured off
`metrics.peakMemoryBytes` for every cell.

With budget left, the remaining four LTX-2.5 shots were rendered as a second run (`ltx/run-rest`,
4 × 200 s, 24.89 GB each, export 24.48 s) to see whether the i2v failure was one bad draw: it was
not. **Every `text_to_video` take (4/4) was on-prompt and every `image_to_video` take (0/2) was
off-prompt** — L-SH040 is another talking-head clip against a mud wall. Under the same rubric the
six LTX takes score 12 / 0 / 8 / 0 / 10 / 7 = 37/72 with no human loop and no repairs: SH030 and
SH060 moved the bench outdoors at sunset (location 0), SH050 is a good take in yet another
workshop, SH060 never lifts the lid. Same continuity story as H3 — the room is re-imagined per
shot — plus a hard i2v failure on placeholder plates. Still two unequal configurations; still no
ranking.

## 5. Decision

### 5.1 Supported conclusions (and confidence)

1. **Engineering: every documented step of the harness ran end to end, offline, on this Mac** —
   plan, compile, validate, run, review, human decisions, bounded replacement, interrupt/resume
   with adoption, edit + re-export, and a record that explains every take. Confidence **high** for
   that: every claim above is read off a run record, a job table, a timeline document or a decoded
   file that is in the evidence directory. It did **not** work as documented on the first pass:
   four defects were found by the pass itself (§3.2) and six more by the adversarial review of this
   report (§3.2b), all fixed with tests in this PR. Two of them reached delivered artifacts — the
   dialogue drop reached the exported film (and its incomplete first fix left `…-with-dialogue-30s.mp4`
   missing SH050's line), and the outcome flip left the run record a reader opens saying `failed`
   where the run had `completed`. That is exactly why the integrated pass was worth running; a
   reader should take "works as documented" to mean "works as documented at this PR's HEAD", not
   "worked at the start of the pass".
2. **Research: a coherent 30-second film did not come out of this configuration, and the reason is
   specific.** Per-shot action and costume were reliable; **location and prop identity were not
   held across shots**, because nothing in the pipeline conditions one shot on another — placeholder
   plates, a base checkpoint with no reference conditioning, and prose-only continuity. Confidence
   **medium-high** that this is the cause (every cut failure traces to it; the one image-conditioned
   pair that got a real-looking plate — none did — cannot be tested until the pack has real plates).
   This is a valid negative finding about the tested configuration; it says nothing about hosted
   systems and makes no Seedance claim.
3. **The local planner is a serviceable draft writer and a poor continuity author.** Confidence
   **medium** (one brief, one draft, two rendered shots).
4. **Assisted review earns its "assistive" label and no more**: useful on identity/custody/cut
   flags, unreliable on costume colour in low light and on sleeves, and structurally unable to judge
   a cut whose neighbour is a close-up. Confidence **high** on the direction, **low** on the
   numbers (small labeled sets, one evaluator).
5. **Cost** (acceptance 3, `manual/metrics.json`). Accepted duration is defined as the sum of the
   selected takes' spans in the final timeline after trims: 24.83 s at the capped verdict (SH050
   has no selected take; the exported file is 30.00 s). Generation wall clock per accepted second,
   counting every attempt (rejected, replaced and out-of-cap): 6332 s / 24.83 s = **255 s**
   (211 s per exported second; 178 s per exported second at the capped 8-attempt verdict). Total
   attempts 21 (hand 9, planner 6, LTX 6), 21 completed, 0 failed; rejected takes 4 on the hand
   film (SH020-a1, SH040-a1 out-of-cap, SH050-a1, SH050-a2 — reasons in `manual/SCORES.md` and
   the record's `rejection` fields), 5 of 6 planner takes and 5 of 6 LTX takes would be rejected
   under the rubric (no loop was run on them). Peak memory per attempt in the tables above
   (`metrics.peakMemoryBytes` for every one); maxima 14.24 GB (H3 5.17 s), 28.90 GB (H3 6.58 s
   planner shot), 24.89 GB (LTX-2.5). Hands-on repair minutes: **12.5 min** of agent wall clock
   over 18 timed steps (rubric scoring, decisions, edits — `manual/TIMING.log` has 20 start/stop
   pairs, 2 of which are GPU waits and excluded; `manual/metrics.json` carries the 18); a person viewing
   36 frames and five exports would take longer. Planner decode: 351 s, 16.87 GiB. Reviewer: 122
   VQA calls in 600 s of `review` plus 553 s of `review-eval` on real weights. Failures: harness
   0, planner 0, reviewer 0, worker **1** (Metal submissions-ignored after ~5 h, §3.5, recovered by
   a process restart). GPU time used: **≈ 5.0 h** of the declared 6 h ceiling (16,460 s of renders,
   600 s of reviews, 553 s of review-eval, 351 s of planner decodes, ~90 s of exports).
   Confidence **high** for these numbers on this hardware.

### 5.2 Limitations

- The evaluator was an AI agent, not Michael, scoring stills, not motion; identity across takes
  was judged from three frames each. The exports are linked at the top for a human viewing.
- One film, one brief, one seed set, one Mac. Six shots is the smallest film the fixture allows and
  30.0 s is the rubric's floor after a single 1 s trim.
- The reference pack is placeholder art; the `image_to_video` shots were conditioned on a brown
  plate, which is the harshest possible test of continuity and not what a real production would do.
- The planner film was rendered to two shots only; the LTX-2.5 comparison is two cells at unequal
  settings; the review-eval sets are 2–11 cases each.
- The capped final export lacks dialogue (fix 2 landed after it); the with-dialogue export was
  produced by one replacement outside the declared cap and carries two of the three lines.
- One worker failure (Metal `SubmissionsIgnored` after ~5 h) needed a process restart; the pass
  does not say whether a longer session would hit it again sooner.
- Hands-on minutes are an agent's; the rubric's "sound" criterion was measured by tone probes, not
  listened to.

### 5.3 Recommendation — **revise** (confidence medium-high)

Not *continue* as is: the next phase should not render more films on this configuration, because
the dominant failure (no cross-shot continuity mechanism) is known and would recur. Not *stop*: the
harness itself did everything the epic asked, cheaply and reproducibly, and the negative finding is
about the conditioning path, which is the one thing the POC deliberately left out (no last-frame
chaining, placeholder plates). Revise the experiment, then decide again.

### 5.4 Candidate follow-ups (evidence-backed, **not authorized by this report**)

| candidate | evidence | dependencies / unresolved decisions |
| --- | --- | --- |
| Real reference plates (a rendered or photographed workshop plate, character plates) and re-run the six shots with the same seeds | every cut failure was a re-imagined room; the i2v shots got a brown card | who authors the plates; whether a generated plate counts as "local" |
| Last-frame chaining as `conditioning: chain` (a `conditioning` edge that actually conditions) | SH020-a2 → SH030 held the parcel object when the take before it did; nothing else did | MiniMax-H3 keyframe mode exists; harness design chose not to wire it (docs: "nothing conditions on the previous shot's last frame") — a product decision |
| Reference conditioning on a checkpoint that has it (LTX-2.5 references, or an H3 reference checkpoint) | H3 base has `maxReferenceAssets: 0`; the harness already resolves `referenceRoles` | which model; §4.5 cells |
| Planner prompt contract: forbid camera moves unless asked, require jacket/door/bench facts from the pack, emit seeds and `dependsOn` | P-SH020 black jacket, three framings per take, no seeds | one more planner story; measure against the same brief |
| Re-render any one hand-film shot on this PR's HEAD and re-export, to get a final carrying all **three** lines | §3.2b fix 4: the SH040 re-render already done under fix 2 produced only two, and no CPU-only path re-assembles sound | ~13 min GPU; changes nothing in the conclusions |
| Review questions that abstain on close-ups (a "can the wall be seen" gate before the pegboard question) | `sh040_cut`, planner `sh020_cut` overclaims | small review-plan change; re-measure on the new sets |

### 5.5 Reproducible artifacts

`README.md` in the evidence directory indexes: `BUDGETS.md`, `RUBRIC.md`, `stack.sh` / `fh.sh` /
`mark.sh`, `commands.log` (every command with start/end/exit), `api.log`, `worker.log`,
`lsof/sockets.log`, `hand-film/` (run record and its six snapshots, `run.json.v1-after-run`
through `run.json.v6-after-sh040-replacement` — v1–v5 read `completed`, v6 reads
`failed`/`attempts_exhausted` for the reason in §3.1 and §3.2b fix 5, and `run/run.json` is the
same document as v6 — review docs, three probes, decision logs), `planner/` (plan, compiled, run
record before/after SIGTERM and resume, job
tables, interrupt log), `ltx/`, `review-eval/`, `frames/`, `exports/`, `manual/` (scores, timing,
metrics, isolation and determinism checks, plan comparison).
