# Local film harness — phase-3 evaluation report (2026-09-22)

Epic [sc-24017](https://app.shortcut.com/trefry/epic/24017) (film harness phase 3, prompt
anchoring), terminal story [sc-24029](https://app.shortcut.com/trefry/story/24029). One finite GPU
pass over the mechanisms the epic adds — compiler-inserted `<Picture N>` binding sentences, shared-image
locators, the text identity lock for described-only roles, and the required per-shot `audio` sentence
— measured on the dev Mac against the feature head. **This report records what was measured; it does
not authorize the next phase.** Same structure and the same rubric as the phase-2 report
([film-harness-evaluation-phase-2.md](film-harness-evaluation-phase-2.md)), so the picture scores are
comparable column for column.

Evidence directory (every run record, take, frame, compiled document, guard log, probe and command
log named below): `~/SceneWorks/film-harness-evidence/sc-24029/` — `RESULTS.md` there is the state of
record. Paths below are relative to it unless they start with `config/` or `docs/`.

**Michael should view the exports himself before acting on anything here.** The evaluator was an AI
agent scoring three extracted frames per take under the phase-1/2 rubric (`RUBRIC.phase1-2.md`,
copied verbatim); it cannot watch motion and it cannot listen. The files to view:
`a6-image/export-a6-image-31s.mp4` (the reference turbo film, with sound) and
`a6-described/export-a6-described-31s.mp4` (the no-reference film, picture only). One take is
explicitly flagged as wanting a human ear: A6a SH060 (§6).

## 1. Tested configuration

| item | value |
| --- | --- |
| SceneWorks code | worktree `/Users/michael/.claude/worktrees/sc-24029-eval` @ **`3fe5f6067`** (the merge of PR 2907, the MLX free-buffer-cache fix), clean tree. `sceneworks-rust-api` and `film-harness` built 2026-09-22 08:17. **Stationary**: no source edit during the pass. |
| GPU loan | **2026-09-22T12:20:28Z → 14:23:44Z = 2 h 03 m 16 s (7 396 s)** of an authorised ~3 h |
| Measured GPU work inside it | 6 653 s (A5 756 s + A6a 5 022 s + A6b 875 s) + 202 s locator compile |
| Hardware / OS | Apple M5 Max, 128 GB unified, macOS 26.6.2; one MLX worker, one GPU process at a time, SIGTERM-only teardown |
| Renderer | MiniMax-H3 q4; reference partition `minimax_h3_ref` for A6a, base `minimax_h3` for A6b; turbo 4-step LoRAs; 576x320, 24 fps, 5.1667 s clips |
| Packs | A5/A6a: `courier-refs/references.jsonc` (schemaVersion 2, 7 roles / 7 images, the sc-23403 Krea plates + the checked-in sound block — §8.1). A5/A6b: `references.described.jsonc` (6 roles, **0 images**, all described-only). Locator cell: `locator-refs/` (a fixture built for this pass). |
| Guard | external 1 s RSS loop (`guard.sh`), 40 GB cap for planner phases, 80 GB for renders, `kill -TERM` only. **Never fired on any run.** |
| Disk | 382 → 378 GB free throughout (declared floor 40 GB) |
| CI | no `Runner.Worker` active at any point; the runner *listener* was idle-resident and is not a GPU process |

**Verdict: every mechanism the epic adds is present, correctly placed and recorded, on all 18
compiled requests across three packs.** A6a scored above the phase-2 turbo baseline; A6b confirmed
the identity lock's expected behaviour and, unexpectedly, showed exactly *why* it fails where it
fails. The defects found are in §8 and **none of them is in the anchoring code**.

A1–A4 are automated and run in CI (`apps/rust-api/src/tests/film_harness_anchoring.rs`). A5 and A6
are the GPU acceptance tests, and they are what this pass measured.

## 2. A5 — planner smoke from the unmodified brief, full repair budget

Both cases **PASS**. PASS = smoke exit 0 (which includes `film-harness validate`) AND an `audio`
sentence on every shot AND no `<Picture`/`<Audio`/`<Video` label anywhere in the planner's draft AND
turbo declared as the default recipe. Checked by `check_a5.py`; results in `a5-*/CHECK.json`.

| case | pack | exit | wall | repair rounds | audio on all 6 | engine labels in draft | turbo default | peak RSS | peak `cacheBytes` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `image` | `courier-refs/references.jsonc` (7 refs / 7 images) | **0** | 453 s | **1** of 2 | yes | **none** | `minimax_h3_turbo_4step_v01` + `minimax_h3_ref2v_turbo_4step` | 15.39 GB | 2.76 GB |
| `described` | `references.described.jsonc` (6 refs, 0 images) | **0** | 303 s | **0** of 2 | yes | **none** | `minimax_h3_turbo_4step_v01` | 15.38 GB | 4.49 GB |

Three things worth naming beyond the pass:

1. **Phase 2's open planner defect is fixed.** Phase 2 recorded that the *unmodified* brief produced
   no valid plan in 2 repair rounds nor in 5, so the turbo recipe was reachable only through a
   hand-authored plan ([phase 2](film-harness-evaluation-phase-2.md) §3.2, §3.6, §5.2). Here the
   unmodified brief yielded a valid plan in **one** repair round on the image pack and **zero** on
   the described pack, compiling to `minimax_h3_ref` with `effectiveSteps: 4`. The expected user path
   into the turbo recipe now works.
2. **The repair-round decode path is now exercised under the memory fix.** The 2026-09-22 10:43Z
   fix-confirm run used `MAX_REPAIR_ROUNDS=0` and flagged that as its main caveat. The `image` case
   here took a repair round at the default budget and the freed-buffer cache still sawtoothed under
   3 GB (§7).
3. **The pack decides the shape, as documented.** The image pack produced six `reference_to_video`
   shots binding approved roles; the described pack produced six shots with zero reference roles and
   the identity lock instead.

## 3. Inserted-text verification — the core of the story

`check_a6.py` re-implements the compiler's own composition rule
(`film_compile.rs::apply_inserted_text`) and checks each compiled request against the plan and pack
it claims. **All 18 requests across three cells PASS** (`*/INSERTED_CHECK.json`).

Per request it proves:

1. `promptSource == "refined"` — the rewrite actually happened;
2. `authoredPrompt` is byte-identical to the plan's own prompt — the author's words are kept beside
   the rewrite, unedited;
3. `prompt` reproduces **exactly** as leading insertions + recovered middle + trailing insertions,
   with the trailing sentence break the compiler supplies;
4. the recovered middle (= the refiner's output) is non-empty, differs from `authoredPrompt`, and
   **contains no `<Picture`/`<Audio`/`<Video` label** — which is what "inserted after the rewrite"
   means operationally: the labels live only in `insertedText`, never in the model's output;
5. every bound role has a `<Picture N>` binding sentence numbered in picture order;
6. the `Audio:` sentence is exactly `"Audio: " + ` the shot's own audio text, whitespace-normalized.

| cell | schemaVersion | requests | kinds present | result |
| --- | --- | --- | --- | --- |
| A6a (image pack) | **7** | 6 | `reference_binding`, `continuity_description`, `audio`, and `no_speech` on exactly SH020/SH050/SH060 | PASS |
| A6b (described pack) | **7** | 6 | `continuity_description`, `audio` (no bindings — nothing is bound; no no-speech — no dialogue clips) | PASS |
| locator fixture | **7** | 6 | `reference_binding` **with locators**, `continuity_description`, `audio`, `no_speech` | PASS |

**`no_speech` placement is exactly right.** It appears on SH020, SH050 and SH060 of A6a — the three
and only three shots carrying a `dialogueClip` — and on no shot of A6b, which has none. That is
`no_speech_text`'s documented rule (keyed on `dialogue_clip`, the placement, not on `dialogue` prose)
holding on real data.

A worked example (A6a SH060, all four kinds):

```
[reference_binding]  The recipient is the person shown in <Picture 1>. The recipient: grey work
                     apron, rolled sleeves. The red parcel is the object shown in <Picture 2>. …
                     The workshop location is the place shown in <Picture 4>. …
[continuity_description]  Warm late-afternoon light, film grain, natural colour.
[audio]              Audio: Tape tearing, cardboard flaps, a soft musical swell as the lid lifts.
[no_speech]          No spoken dialogue in the generated audio; no voices on the soundtrack.
```

The identity lock here carries `house_style` — a `continuityRoles` entry the shot does not bind to an
image — which is the lock working on a *reference* plan, not only on a described-only one.

### 3.1 The shared-image locator path (added check, compile-only)

**No pack in the repository exercises this path.** Every role in `courier-refs`, the shipped
`references.jsonc`, `references.spec.jsonc` and `references.described.jsonc` has its own distinct
file or no file, and not one declares a `locator`. The brief's "locators where roles share a file" is
therefore *vacuous* against the v2 pack — A6a can only prove the non-locator branch.

To close that gap at 202 s of GPU rather than a third film, `locator-refs/` binds `courier` and
`recipient` to one plate (`references/pair.png` — the two portraits hstacked, 2048x1024) and tells
them apart by locator; `plan.locator.jsonc` adds `recipient` to SH010's `continuityRoles` so one shot
supplies a picture that a second role rides. Compile only, no render. The compiler emitted:

> The courier is **the figure on the left** in `<Picture 1>`. The courier: blue jacket, carries the
> parcel. The recipient is **the figure on the right** in `<Picture 1>`. The recipient: grey work
> apron, rolled sleeves. The red parcel is the object shown in `<Picture 2>`. …

Exactly as `reference_binding_text` and `pictures_with_shared_continuity` document: both roles bound
to the **same** `<Picture 1>`, each located within it, the picture numbering unaffected (1, 1, 2, 3,
4), and the second role given a **binding** sentence rather than an identity-lock sentence because
its file is already being supplied. **Verified, but on a synthetic fixture and without a render — no
claim is made here about what the engine does with a located pair.**

## 4. A6a — reference turbo film: **65/72 picture, 2/2 sound = 67/74**

Phase 2's turbo cell (e) baseline: 64/72 and 66/74. Full scoring, per-shot justification and frame
citations in `SCORES.a6-image.md`.

| take | wall (s) | id | cost | loc | parcel | action | cut | total | phase-2 cell (e) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SH010 | 870 | 1 | 2 | 2 | 2 | 2 | 2 (n/a) | 11 | 12 |
| SH020 | 830 | 2 | 2 | 2 | 2 | 1 | 1 | 10 | 9 |
| SH030 | 780 | 2 | 2 | 2 | 2 | 2 | 1 | 11 | 12 |
| SH040 | 770 | 1 | 2 | 2 | 2 | 2 | 2 | 11 | 11 |
| SH050 | 810 | 1 | 2 | 2 | 2 | 2 | 2 | 11 | 9 |
| SH060 | 770 | 2 | 2 | 2 | 2 | 1 | 2 | 11 | 11 |
| **total** | **4 830** | 9 | **12** | **12** | **12** | 10 | 10 | **65/72** | 64/72 |

Run `completed`, 6 attempts, 6 completed, 0 failed, 0 rejections, 0 repairs. Render wall 4 876 s
against cell (e)'s 4 722 s — the same cost for the same film.

**Costume, location and parcel each scored a perfect 12/12**: the plates held on every one of six
shots. Every point lost is in identity (faces smeared or turned away), action (two end-state
overshoots) or cut. The room is one room, the parcel is one object, and the cuts carry.

**How comparable is this to 64/72? Only loosely**, and the scores file says so before the table. The
prompts are not cell (e)'s prompts — every one now carries the binding sentences, the identity lock
and the audio sentence — so each take is a **different draw at the same seed**, and per-shot deltas
mix sampling with the anchoring change. **The honest reading of 65 vs 64 is "no regression, and the
plates still dominate", not "+1 from anchoring."**

### 4.1 Sound — 2/2

`a6-image/export-probe.json` over the 31.000 s export (1138x640 h264 + AAC 48 kHz stereo):

- **Both beds continuous across all five cuts.** Ambience (100 Hz) after/before ratios 0.96, 1.01,
  0.92, 1.01, 1.00; music (250 Hz) 1.01, 1.00, 1.00, 0.99, 1.01. No bed restarts at a cut.
- **Every dialogue line measurable in its slot and absent outside.** RMS 0.048 / 0.041 / 0.033 at
  6.37 s, 23.27 s, 28.83 s against a beds-only 0.0200 — ratios 2.39, 2.03, 1.62. Phase 2 cell (a)
  measured 0.049 / 0.050 / 0.044 against 0.0145; the shapes agree.
- **No generated clip audio in the mix** (policy `mute`): outside the slots the level is steady
  (std/mean 0.15), i.e. the beds alone.
- All five sound roles adopted, including the three Kokoro syntheses (`run.json.sound[]`).

## 5. A6b — no-reference turbo film: **49/72 picture**, sound n.a.

Six `text_to_video` shots on the base checkpoint, **no plates at all**. Full scoring in
`SCORES.a6-described.md`. There is no comparable baseline — phase 1's no-reference 58/72 was a
different plan, pack, step count and attempt count — so **49/72 is a characterization, not a
regression**.

| take | wall (s) | id | cost | loc | parcel | action | cut | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SH010 | 210 | 2 | 2 | 2 | 2 | 1 | 2 (n/a) | 11 |
| SH020 | 70 | 1 | 2 | 2 | 2 | 2 | 1 | 10 |
| SH030 | 70 | 1 | 1 | **0** | 2 | 1 | **0** | 5 |
| SH040 | 70 | **0** | 2 | 2 | 2 | 1 | **0** | 7 |
| SH050 | 70 | 2 | 2 | **0** | 2 | 2 | **0** | 8 |
| SH060 | 80 | 2 | 2 | 1 | 2 | 1 | **0** | 8 |
| **total** | **570** | 8 | 11 | 7 | **12** | 8 | 3 | **49/72** |

Under the rubric's decision rule (any criterion 0 ⇒ reject) SH030, SH040, SH050 and SH060 are
rejections. No repair was requested — the loan was for measurement, and a repair would have replaced
the draws this cell exists to characterize.

Only SH010 pays the checkpoint load (210 s); the remaining five run in 70–80 s each, so **the
checkpoint stays resident across a run's jobs on this partition** — phase 2 observed the opposite on
the reference partition and raised it as a candidate.

Sound is **n.a.**: the described pack declares no `sound` block, so `run.json.sound[]` is empty and
the export carries **no audio stream**. That is the pack's construction, not a fault. No shot carries
the no-speech sentence either, so A6b contributes nothing to that question.

### 5.1 Role × shot consistency — the expectation, tested

| attribute | SH010 | SH020 | SH030 | SH040 | SH050 | SH060 |
| --- | --- | --- | --- | --- | --- | --- |
| **wardrobe** | held | held | **partial** | held | held | held |
| **props** (red parcel) | held | held | held | held | held | held |
| **setting** (workshop) | held | held | **drifted** | held | **drifted** | **drifted** |
| **face** | n.a. (baseline) | **drifted** | n.a. | **drifted** | n.a. | **drifted** |

**The brief's expectation is confirmed: the lock holds wardrobe, props and setting, and does not hold
a face.**

- **Wardrobe, word for word.** The recipient's "grey canvas work apron over a rust-coloured jumper,
  sleeves rolled to the elbow, reading glasses pushed up into short grey hair" renders with **all
  four** features in SH040, SH050 and SH060 — the glasses pushed up into the hair in every one. The
  courier's "blue *quilted* jacket over a grey shirt" renders quilted. SH030 is the only partial:
  jacket holds, trousers turn olive.
- **Props everywhere.** "A small bright red cardboard box … taped along one seam, with a plain white
  label on its lid" is legible as exactly that in all six, taped seam and lid label included.
- **Face nowhere.** Every pair of comparable faces is two different people (`zoom-courier-SH010` vs
  `-SH020`; `zoom-recipient-SH040` vs `-SH060`), and SH020 changes face between its own frames 2 and
  3. The pack describes garments, colours, objects and rooms; it says nothing that individuates a
  face beyond "a young woman" / "an older woman … short grey hair". **The lock holds what it says,
  and a face is not something it says.**

### 5.2 The sharpest result: the lock's reach is exactly what the plan lists

| shot | `workshop_location` in `continuityRoles`? | lock says "The workshop: …"? | rendered setting |
| --- | --- | --- | --- |
| SH010 | yes | yes | cluttered woodworking room, pegboard tool walls |
| SH020 | yes | yes | same room, bench centre, window right |
| SH030 | **no** | **no** | **an outdoor brick alley** |
| SH040 | yes | yes | cluttered woodworking room, pegboard, bench |
| SH050 | **no** | **no** | **a domestic room — radiator, floorboards, framed pictures** |
| SH060 | **no** | **no** | a workroom by a window; not the pegboard room |

All three shots whose plan omits `workshop_location` left the workshop; all three that name it
stayed. Mechanism, per-shot inserted text and rendered result agree on all six. Confidence **high**.

This is the lock working as designed *and* a warning about the surrounding system: **the lock cannot
hold a role the plan does not list**, and the planner is the thing that writes `continuityRoles`.
Phase 1 and phase 2 both recorded the planner as a "poor continuity author"; this cell shows what
that costs now that the lock exists to act on its output. **The lock makes the plan's omissions
visible and expensive rather than fixing them.**

## 6. Audio findings

Per-take analysis in `audio/`, spectrogram PNGs beside each JSON. **What these can and cannot
settle**: they measure whether there is a soundtrack, where its energy sits, and whether it carries
the harmonic-plus-syllabic signature of speech. They **cannot transcribe**, and they cannot tell a
door latch from a dropped tool — "the stated sound" is a semantic claim needing a person listening.
**No ASR was available**: `whisper_base` is in the catalog and the HF cache but declares
`capabilities: []`, so it is not dispatchable as a job in this build. **Every claim below is by
spectrogram and level probe only.**

- **Every one of the 12 takes has a real generated audio stream.** The `Audio:` sentence reaches a
  soundtrack in both cells.
- **No take shows speech** — in either film, including all three A6a shots that carry the no-speech
  sentence (SH020, SH050, SH060). The no-speech sentence is not contradicted anywhere it was used.
- **Transient structure is consistent with the stated sounds** where it is legible: SH010's four
  sharp broadband transients in the first 1.4 s then a regular impulse train (latch, creak,
  footsteps); SH020's evenly spaced transients ≈ every 0.45 s (footsteps); SH030's sub-85 Hz-dominant
  impacts (thump of cardboard on wood). Stated as consistency, **not identification**.
- **SH060 is the one worth a human ear.** It is strongly tonal — voiced-frame fraction 0.58, the
  highest of any take — which a voiced-frame count alone would flag as a possible voice. But its
  syllabic (4–8 Hz) modulation is only 0.12 and its spectrogram shows **steady horizontal harmonic
  bands rather than formant transitions**: a held musical tone, matching its own "a soft musical
  swell as the lid lifts", not speech. Confidence that this is music and not a voice: **medium-high**
  on the measurements; a person should confirm it by ear.

## 7. Memory — the reason this pass was guarded

The unguarded run of this evaluation on **2026-09-21 kernel-panicked the host**. Root cause: during
LLM decode in the prompt-refine path the **MLX freed-buffer cache grew ~92 MB/token**, reaching
63.7 GB and ~90 GB resident. The fix is **PR 2907** (`fix/sc-24029-prompt-refine-mlx-cache-growth`),
merged as `3fe5f6067`, which is the head this pass ran on.

| | panic run (2026-09-21) | fix-confirm (0 repair rounds) | **A5 image** | **A5 described** |
| --- | --- | --- | --- | --- |
| peak `cacheBytes` | 63.7 GB | 2.51 GB | **2.76 GB** | **4.49 GB** |
| peak worker RSS | ~90 GB | 15.35 GB | **15.39 GB** | **15.38 GB** |
| outcome | **kernel panic** | exit 0 | exit 0 | exit 0 |

The fix holds across both packs and across a repair round. `activeBytes` sits flat at the ~16.8 GB
the 8B checkpoint occupies, so only the freed-buffer pool is being cleared.

### Peak memory per run (from the guard logs)

| run | samples | guard cap | peak RSS | peak `activeBytes` | peak `peakBytes` | peak `cacheBytes` | guard fired |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A5 image | 439 | 40 GB | 15.39 GB | 16.83 GB | 16.97 GB | 2.76 GB | **no** |
| A5 described | 294 | 40 GB | 15.38 GB | 16.73 GB | 16.97 GB | 4.49 GB | **no** |
| A6a + A6b + locator (one stack) | 6 352 | 80 GB | **33.68 GB** | 39.09 GB | 39.59 GB | **21.96 GB** | **no** |

7 085 guard samples across the loan; the guard never fired. The A6 peak of 33.7 GB RSS is a transient
at the refiner→H3 handover (the 8B checkpoint still resident alongside the video model, ~16 + 24 GB);
steady-state rendering sat at 15–29 GB.

**`cacheBytes` reached 21.96 GB during *renders*.** PR 2907 bounds the **LLM decode path, not the
render path** — so this number is not covered by that fix. It was **stable rather than growing**, and
`active + cache` never exceeded 42.7 GB against a 135.3 GB ceiling. Worth knowing; **not a leak on
the evidence here.**

## 8. Defects and observations

**None of these is in the anchoring code the epic adds.**

1. **The briefed pack copy could not run `plan.v2.turbo.jsonc` (recipe gap, fixed in the evidence
   dir).** `validate` refused before dispatch: *`[SH060] dialogueClip.role: sound role
   "recipient_reveal_line" is not in reference pack "courier-workshop-refs"`*
   (`a6-image/validate-prefix.log`, exit 2). The sc-23403 pack predates the reveal line. Fixed
   exactly as phase 2 did — grafted the checked-in pack's sound block (three dialogue entries with
   `text`, Kokoro-synthesized, plus the two beds), original preserved as
   `courier-refs/references.jsonc.orig-no-reveal-line`. This is also what makes A6a's sound
   comparable with the 64/72 baseline. **Not a product defect**; the refusal names the field and the
   role correctly.
2. **The refiner truncated mid-sentence.** A6a SH060's refined text ends `"overall_soundscape:\n  The
   only"` and the compiler's sentence break then supplies the period. Total prompt 3 026 chars
   against `MAX_PROMPT_CHARS` 4 000, so **nothing was refused or trimmed by the limit** — the LLM's
   own generation stopped short. The compiler's own sentences were unaffected and still verified.
   The refine job's token budget is examined in §8.1.
3. **The prose beat the plate on A6a SH050 — the recipient rendered as a woman.** The approved
   `recipient` plate is a man; phase 2 recorded "the reference won every time" over two draws. Here
   the dispatched prompt carries **15** occurrences of she/her/woman and **0** masculine, opening "A
   woman in a grey work apron…", and the render followed it. **The compiler did not cause this**: its
   binding sentence is gender-neutral ("The recipient is the person shown in `<Picture 1>`."); the
   refiner's rewrite of the plan's own "she" did. Confidence that sc-24029's insertions caused the
   flip: **low** — one draw, and two things changed at once (insertions added *and* a different
   refiner rewrite); SH060's prompt has zero gendered words and shows only hands, so it is not a
   control either way. Confidence that the plan/plate inconsistency is real and can now visibly flip
   the rendered person: **high**. Fixed in the checked-in fixtures by this story — see §8.2.
4. **An uninvited person in A6b SH040.** `continuityRoles` name the recipient and the startState is
   "empty workshop, parcel alone on the workbench, door closed", but the take puts the **courier in
   the doorway** three shots after she left. Scored identity 0 by the rubric's own wording ("an extra
   person the plan did not ask for"). **A model/plan-authoring outcome, not a harness fault.**
5. **`check_a6.py` had a bug of the evaluator's own, found and fixed during the pass.** Its locator
   regex rejected a locator beginning with "the" ("the figure on the left"), and its first version
   matched insertion kinds in camelCase where the document uses snake_case. Both were corrected and
   **all three cells re-verified from scratch** with the corrected checker; the results in §3 are the
   corrected ones.

### 8.1 The refiner's token budget

RESULTS.md left §8.2 as "worth a look at the refine job's token budget". It was looked at, and **it
is a defect, fixed in this story.**

The per-shot film refine is dispatched with **no `task` field** — `apps/rust-api/src/film_planner.rs`
sends `task: None` for a shot rewrite, while the whole-plan calls pass `Some(FILM_PLAN_TASK)`. A
payload with no task classifies as `RefineTask::Rewrite`
(`crates/sceneworks-worker/src/prompt_refine_jobs.rs`, `RefineTask::from_payload`), and `Rewrite`
carried the smallest of the four budgets:

| task | budget (tokens) | reachable chars at ~2.83 chars/token |
| --- | --- | --- |
| `MagicPrompt` / `ImageCaption` | 4096 | ~11 600 |
| `FilmPlan` | 4096 | ~11 600 |
| `ImageDescribe` | 1024 | ~2 900 |
| **`Rewrite`** (the film shot refine) | **512** | **~1 450** |

The output contract for that rewrite is `sceneworks_core::MAX_PROMPT_CHARS` = **4000** chars, which
needs **~1414 tokens** at the ratio this file itself states ("4096 ≈ ~11.6k chars"). **512 tokens
could not reach the contract**, so the generation ended on `MaxTokens` rather than on EOS — exactly
the mid-sentence cut observed on SH060. The budget's comment still described the caller it was sized
for in 2018 ("the free-text rewrite is a one-liner (512 is ample)"); the per-shot film refine was
never revisited when film shots began emitting multi-field MiniMax-H3 blocks. The override path does
not rescue it either: `PromptRefineRequest` has no `maxNewTokens` field, so 512 was effectively
hard-coded on this route.

**Fix (minimum change):** `DEFAULT_REFINE_MAX_NEW_TOKENS` 512 → **1536** (≈4350 chars, the contract
plus headroom). A rewrite that has said what it has to say still emits EOS far below the cap, so this
only rescues the truncating cases and costs nothing on the normal path. Guarded by a new unit test
that asserts the **contract** — budget × the file's own chars-per-token figure ≥ `MAX_PROMPT_CHARS` —
rather than the literal, so shrinking the budget or raising the cap fails in CI rather than in a
render.

**One adjacent finding is left open and is Michael's call, not an agent's.** A `Rewrite` that
finishes on `FinishReason::Length` is treated as an ordinary success: the "exhausted its
{max_new_tokens}-token output budget" message is raised only for `FilmPlan` **and** only when the
output is empty, and `finishReason` is recorded on the *failure* result, not the success one. So a
truncated-but-non-empty rewrite completes silently — which is why this went unnoticed until a human
read SH060's prompt. Raising the budget removes the trigger, not the blind spot. Surfacing it rather
than fixing it here because it changes the job-result shape, which is beyond this story's scope.

### 8.2 The recipient's gender in the checked-in fixtures (fixed here)

The `recipient` plate is **approved and is a man** (`references/recipient.png`; the plate's own
generator prompt is gender-neutral — "a workshop owner … arms relaxed at their sides" — so the man is
the image model's draw, and the plate is the approved artifact). Every checked-in document that binds
that role against `references.jsonc` nevertheless called the recipient "she" or "a woman". Phase 2
recorded this as a document-consistency finding and it stayed open; §8.3 shows it is no longer
cosmetic, because the refiner now amplifies the plan's pronouns into the dispatched prompt.

Corrected in this story so the checked-in plans, brief, review plan and pack agree with the approved
plate:

| file | what changed |
| --- | --- |
| `config/film-harness/courier-workshop/plan.v2.turbo.jsonc` | SH050, SH060 prompts: her/She/she/her → his/He/he/his |
| `config/film-harness/courier-workshop/plan.v2.jsonc` | the same two prompts (identical prose) |
| `config/film-harness/courier-workshop/plan.jsonc` | SH050 "A woman…" → "A man…"; SH060 "A woman's hands…" → "A man's hands…" |
| `config/film-harness/courier-workshop/plan.ltx25.jsonc` | the same two prompts |
| `config/film-harness/courier-workshop/brief.jsonc` | synopsis and the SH060 beat — the text the planner reads |
| `config/film-harness/courier-workshop/review.jsonc` | "The recipient — a woman, not the courier —" → "a man" |
| `config/film-harness/courier-workshop/references.jsonc` | `recipient_line` description "as she notices the parcel" → "as he notices" |

`plan.v2.turbo.jsonc` and `plan.v2.jsonc` received byte-identical edits, so the invariant that the
turbo plan "differs from `plan.v2.jsonc` in `model.loras` and nothing else" still holds.

**`plan.described.jsonc` was deliberately left alone.** It runs against
`references.described.jsonc`, a *different* cast that describes the courier and the recipient as
women in words, with no plate to contradict. That pack is internally consistent and the lock quoted
it verbatim (§5.1).

**One inconsistency is left standing, and it needs Michael's decision, not an agent's.** The
recipient's two dialogue roles in `references.jsonc` still use the Kokoro voice `af_heart`, a
female voice. Changing it re-synthesizes the spoken lines and would break comparability with the
phase-2 sound baseline that §4.1 is scored against, so it was not changed here. The two coherent
end states are: keep the male plate and move the recipient to a male voice, or regenerate the plate
and revert the pronouns. Phase 2's candidate "reconcile the prompt/rubric gender with the approved
plate (or regenerate the recipient plate)" remains open on that one point.

## 9. Limitations — what this pass does not establish

- Motion, texture and how a face reads in movement: **not measured** (stills only, three per take).
  One seed set, one Mac, one evaluator, **no human viewing**.
- **The locator path was compiled but never rendered**, so nothing here says what the engine does
  with a located pair. The fixture is synthetic.
- A6b has no sound to score, and contributes nothing to the no-speech question.
- Whether the stated sound is *semantically* what was rendered: **no ASR, no listening** (§6).
- A6a's 65/72 against cell (e)'s 64/72 is **not** a controlled comparison: different prompts, same
  seed, so per-shot deltas mix sampling with the anchoring change (§4).
- A6b's 49/72 has **no baseline**; it is a characterization of the described-only path.
- The `cacheBytes` 21.96 GB observation during renders is one pass's evidence of a *stable* pool, not
  a demonstration that the render path is bounded.

## 10. Artifacts

Everything below is under `~/SceneWorks/film-harness-evidence/sc-24029/`.

| path | what |
| --- | --- |
| `RESULTS.md` | the state of record this report summarizes |
| `PRERUN_FINDINGS.md` | the two recipe findings established before any render |
| `SCORES.a6-image.md`, `SCORES.a6-described.md` | per-shot scoring with frame citations |
| `RUBRIC.phase1-2.md` | the phase-1/2 rubric, copied verbatim before any frame was viewed |
| `a5-image/`, `a5-described/` | planner smokes: `planned/plan.json`, `compiled.json`, `smoke.log`, `CHECK.json`, `guard.log` |
| `a6-image/` | compiled docs, `run/run.json`, `validate-prefix.log` (the refusal), `validate.log`, `INSERTED_CHECK.json`, `export-a6-image-31s.mp4`, `export-probe.json` |
| `a6-described/` | same shape; `export-a6-described-31s.mp4` (picture only) |
| `a6-locator/` | compile-only locator fixture: `compiled/compiled.json`, `INSERTED_CHECK.json` |
| `locator-refs/`, `plan.locator.jsonc` | the shared-plate fixture (`references/pair.png`) and its plan |
| `courier-refs/` | the v2 pack copy; `references.jsonc.orig-no-reveal-line` is the unmodified original |
| `frames/a6-image/`, `frames/a6-described/` | three frames per take + `index.json` + the face/courier zoom crops |
| `audio/a6-image/`, `audio/a6-described/` | per-take `*.audio.json` and `*.spectrogram.png` |
| `guard.sh`, `run_a5_guarded.sh`, `run_stack_guarded.sh` | the guard and its harnesses |
| `check_a5.py`, `check_a6.py`, `extract_frames.py`, `audio_probe.py`, `export_probe.py` | the checkers |
| `logs/guard-a6.log`, `a5-*/guard.log` | 7 085 guard samples across the loan |
| `commands.log`, `logs/stack.log`, `logs/api.log`, `logs/worker.log` | every command with start/end/exit, and the stack logs |

**Teardown.** `touch STOP` → `stack.sh` SIGTERMed worker then API (`stack exit 0`, 14:23:27Z). **No
SIGKILL was used at any point.** The guard observed zero `sceneworks-rust-api` pids and exited
(`fired=0`). `pgrep -x sceneworks-rust-api` and `pgrep -fl film-harness` both empty afterwards; no MLX
process remains and the GPU is free. Disk 378 GB free.
