# Calibration measurement runbook

**One parameterized procedure for measuring one `<backend>:<provider>` lane, end to end.**

Substitute a single lane name — for example `mlx:z_image_turbo` or `candle:krea_2_turbo` — and follow
this top to bottom. Everywhere below, `<lane>` means that string, `<backend>` its part before the
colon (`mlx` or `candle`) and `<provider>` its part after.

This replaces the pattern of filing a story per model. Measurements are made by following this
document; only lanes someone actually cares about get measured.

**Validation status.** This document was walked end to end by a fresh session holding only the
runbook and a lane name (`mlx:flux2_dev`, sc-18104). §1, §2, §4a, §5, §7b, §7c `--check` and §8 were
**executed**; the fixes from that walk are folded in below. The capture itself (§3, §6, §7a, §7d,
§9-§10) was **not** reached: §2 correctly stopped the lane in about a minute as an
adapter-implementation task, which is the outcome §2 exists to produce. Sections still marked
"transcribed, not executed" in their own provenance notes remain so.

Companions, not prerequisites — you should not need to open either to finish a measurement:

- [memory-calibration-harness.md](memory-calibration-harness.md) — the reference: schema semantics,
  identities, resume, provider protocol, truth status.
- [calibration-invalidation-sc-17774.md](calibration-invalidation-sc-17774.md) — what a closure
  digest is, what it deliberately does not see, and the pin-bump procedure.

## 0. What measuring buys, and what it does not

Measuring a lane **improves prediction**. It does not unlock anything.

Since epic 18093 (sc-18095/18096/18097), a lane whose provider compile closure has moved keeps
serving its measured numbers behind a **widened admission margin**, and unmeasured cells are admitted
from **fitted estimates** behind a wider margin still
(`crates/sceneworks-worker/src/ladder_margin_policy.rs`). Nothing in `npm run check`,
`npm run rust:check`, the pre-push hook or CI demands a re-capture — **there are zero staleness gates
in CI**. So the payoff of a capture is narrower margins and better-grounded admission on that lane,
not a ladder that was previously refused.

Read that as permission to measure the lane that matters and leave the rest on estimates.

## 1. Choose the lane

```bash
npm run report:stale-lanes          # human table, ranked by impact
npm run report:stale-lanes -- --json   # same data, machine-readable
```

Verified output shape (run on the sc-18212 branch at inference pin `40fa7583`):

```
9 declared lanes: 8 stale, 0 current, 0 pending capture; 6 lanes (declared or planned) have no adapter arm, and 2 planned lanes were never declared.
33 shipped calibration bindings are serving under a widened margin; 65 eligible evidence records
are stale corpus debt, not runtime inputs.

#  LANE                         BINDINGS  RECORDS  MARGIN  ESTIMATE  IMPACT  CAPTURE  MODELS
1  mlx:qwen_image               9/9       41/41    5.00%   10.00%    0.450   yes      qwen_image
2  mlx:z_image_turbo            5/5       5/5      5.00%   10.00%    0.250   yes      z_image_turbo
...
4  candle:flux1_dev             5/5       5/5      2.00%   4.00%     0.100   NO ARM   flux_dev
...
DECLARED/PLANNED BUT UNCAPTURABLE — no adapter arm can serve these; a capture host booked for
one is wasted (docs/calibration-runbook.md §2c). A declared lane needs adapter work before
measurement; a planned-but-undeclared lane needs both an adapter arm and a closure declaration:
  candle:z_image  declared=yes  plan=90 entries (90 authoritative)  bindings=0 shipped  records=0 eligible  status=uncapturable
  ...
```

`IMPACT` is stale bindings × the margin the runtime is applying to them right now, so the top row is
the lane where a capture buys the most. `BINDINGS` is the **shipped admission surface** (what the
worker's fit gates consult today); `RECORDS` is corpus debt. Prefer bindings.

`CAPTURE` (sc-18212) is §2c answered mechanically: the report parses the two adapter binaries'
provider-dispatch match arms (the same source of truth that decides whether `run()` can serve the
provider — `scripts/stale-lane-report.mjs#adapterCapturableProviders`) and flags every declared or
planned lane with no arm as **uncapturable**, in its own section, visibly distinct from "pending
capture". `candle:z_image` — declared, 90 authoritative plan entries, no arm — used to print as
"declared but never captured", which read as pending measurement work; it is an
adapter-implementation task, and the report now says so. Planned-but-undeclared lanes
(`candle:qwen_image_edit`, `candle:qwen_image`) are enumerated too. **§2c/§2d below remain the
diagnosis**; the report is now an index into them, not a substitute for them.

🔴 **This report enumerates DECLARED and PLANNED lanes only, so it still cannot tell you a lane does
not exist.** A lane is keyed exactly as `config/inference-provider-closures.json` keys it
(`scripts/stale-lane-report.mjs#SOURCE_PATHS`), with `config/memory-calibration-plan.json` supplying
the planned-but-undeclared rows (sc-18212); a lane absent from **both** files appears **nowhere in
the output**. Measured while validating this runbook (sc-18104): `mlx:flux2_dev` produces zero rows
even though FLUX.2 [dev] ships an MLX lane in `builtin.models.jsonc` — still true after sc-18212,
because no plan entry names it either. If you arrived here with a lane name from outside this report
— a product decision, a story, "the most popular model's Mac lane" — **do not read its absence as
"nothing to do"**. Go straight to §2 and check it explicitly.

The gate inventory behind the report — what does and does not grade currency — lives in
[calibration-invalidation-sc-17774.md](calibration-invalidation-sc-17774.md) ("What grades what",
"What does NOT grade currency").

## 2. Can this lane actually be captured today?

Four things must all be true. Check them **before** booking a GPU — every check below is a few
seconds, and together they are the difference between finding out now and finding out after a
multi-hour sweep.

They are **not independent, and the order here is diagnostic order, not build order.** For a lane that
does not exist yet the dependency runs the other way: an adapter arm (§2c) is what makes plan entries
(§2b) meaningful, and plan entries are what make a closure declaration (§2a) worth deriving. Read
§2a-§2d to find out *where* the lane stands; if it fails §2c or §2d, that is an implementation task
and §3 onward does not apply — see §2d.

### 2a. The lane is declared in the closure table

```bash
node -e 'const c=require("./config/inference-provider-closures.json");
  console.log(c.providers["<lane>"] ?? "NOT DECLARED")'
```

If it prints `NOT DECLARED`, the capture will fail in seconds — the harness derives one closure
digest per lane **before the first capture invocation** precisely so this does not surface after a
multi-hour sweep. Add the lane per §7c first.

### 2b. The plan has entries for the lane, and you know its fixture

`--fixture` — not `--provider` — is how a multi-rung ladder is selected as one reproducible capture.
Derive it from the plan rather than guessing:

```bash
node -e 'const [backend,provider]=process.argv[1].split(":");
  const plan=require("./config/memory-calibration-plan.json");
  const rows=plan.providers.filter(p=>p.backend===backend&&p.target.provider===provider);
  if(!rows.length) throw new Error("no plan entries for "+process.argv[1]);
  console.log([...new Set(rows.map(p=>`${p.evidenceScope}  ${p.fixture}`))].join("\n"))' <lane>
```

Verified for `mlx:z_image_turbo` → `authoritative  fresh-five-rung-z-image-q4-768-seed16402-step2`.

Only `authoritative` scope can ever become `current`. A `fixture`/`candidate` selection derives no
digest and can never be current evidence.

### 2c. A provider adapter covers the lane

The adapters are `crates/sceneworks-memory-adapter/src/bin/mlx.rs` and `.../bin/candle.rs`. Provider
ids they cover today:

| binary | providers covered | how it dispatches an unknown provider |
| --- | --- | --- |
| `memory-mlx-adapter` | `qwen_image`, `z_image_turbo`, `krea_2_turbo`, `sdxl`, `krea_2_turbo_control`, `flux2_dev` (SDXL exposes only Resident, Staged, and bounded-transformer residency; its decode/attention rungs are measured `Missing`. FLUX.2-dev is **resident rung only**.) | `mlx.rs` `run` — `MLX five-rung calibration does not implement provider "<id>"`; `validate_z_image_batch` (`assess_batch`) — `…five-rung batch assessment does not implement provider "<id>"` (cited by function name; the line numbers this table used to carry went stale the first time the file grew) |
| `memory-candle-adapter` | `qwen_image`, `krea_2_turbo` | `candle.rs:540-548` — `Candle five-rung calibration does not implement provider "<id>"` |

Since sc-18212 the stale-lane report answers this gate for you: its `CAPTURE` column and
"DECLARED/PLANNED BUT UNCAPTURABLE" section are derived by parsing these dispatch matches
(`scripts/stale-lane-report.mjs#adapterCapturableProviders` — anchored on the refusal phrase below,
so a provider counts as capturable only if every dispatch gate admits it). The table above is the
human copy; the report is the derived one. Grep before you schedule anyway:

```bash
grep -n '<provider>' crates/sceneworks-memory-adapter/src/bin/<backend>.rs
```

Both adapters now refuse an unimplemented provider **by name, before any environment or model work**,
on **both** MLX actions — `run`, and `assess_batch`, where the check lives inside
`validate_z_image_batch` so it fires before `runtime_macos::catalog()` is built.
Trust that message: it means the arm is missing, not that your environment is wrong.

> **This was not true until sc-18104, and the old behaviour is worth knowing** because it may still be
> what an older adapter binary or a stale build does. The MLX `run()` used to test for
> `z_image_turbo` and `krea_2_turbo_control` and send *everything else* to `run_qwen_provider` — so an
> unimplemented MLX lane silently **misrouted into the Qwen arm** and died further in on a Qwen-shaped
> complaint. Measured by reverting the fix and re-running the guard test: capturing `flux2_dev`
> produced `planned.target.overlay must be a string`. That names neither FLUX.2 nor the missing arm,
> reads like a malformed plan entry, and is exactly the kind of message that sends an operator off
> fixing fixtures or provisioning weights for the wrong model.
>
> **`assess_batch` had the identical hole, and it is the more dangerous of the two.**
> `validate_z_image_batch` checked batch length, canonical rung order and target-tuple stability but
> never read `target.provider`, while `assess_z_image_batch` hardcodes `Z_IMAGE_PROVIDER` when it
> reads the memory-strategy contract — so a foreign five-rung batch passed validation and was
> misrouted into the **Z-Image** contract, failing on a Z-Image-shaped fingerprint complaint *after*
> `runtime_macos::catalog()` had already done real environment work. It was reachable through the
> documented path: `assessProviderReuse` (`scripts/memory-calibration-harness.mjs`) selects candidates
> by backend and optional fixture only, **never by provider**. Measured by deleting the new guard: a
> `flux2_dev` five-rung batch validated *successfully* and proceeded toward the Z-Image contract.
> Candle has no equivalent hole — its batch path refuses by name at `candle.rs:1108`.

**How much this gate stops, measured (sc-18104, on `origin/main`).** Only `authoritative` scope can
ever become current evidence (§2b), and the plan holds **155 authoritative entries across 8 lanes**.
Of those 8 lanes, **4 have no adapter arm** — `candle:flux1_dev`, `candle:flux1_schnell`,
`candle:flux2_dev` and `candle:z_image` — which is **105 of the 155 authoritative entries**. Three of
those four (the flux lanes) **already carry 5 committed evidence records each**: evidence captured by
an adapter that no longer implements them, so they can be *reported* stale but cannot be re-captured
today. `candle:z_image` is the only one the stale-lane report surfaces, because it is the only one
with no records to be stale.

> Snapshot drift note: sc-18218 has since added **4 authoritative `mlx:flux2_dev` entries** (q4/q8
> at 768² and 1024²) with an adapter arm and a closure declaration, so the live totals are 159
> authoritative entries across 9 lanes. The measured proportions above are the
> sc-18104 snapshot and are kept as measured.

Two more traps in the same area:

- **A named arm is not automatically a current-evidence lane.** `candle:qwen_image` has an adapter arm
  but its 5 plan entries are all `candidate` scope, so no capture through it can ever be `current`
  (§2b). Check the arm *and* the scope.
- **A closure-table entry is not an arm.** `candle:flux2_dev` is declared, digested and has committed
  records, and still has no arm. Declaration, evidence and capturability are three separate facts.

A lane with no adapter arm is an adapter-implementation task, not a measurement task. Stop here, file
it as such, and say so — see §2d.

### 2d. If the lane is missing entirely

§2a-§2c can each fail on their own, but the interesting case is a lane that fails several at once:
**it does not exist yet.** Diagnose it precisely before writing any story, because the five states
below need very different work. Read all three gate results together — **row one and row four differ
only in §2b**, and mistaking one for the other prescribes writing plan entries that already exist.

| §2a | §2b | §2c | what it means | the work |
| --- | --- | --- | --- | --- |
| `NOT DECLARED` | throws | empty | the lane does not exist anywhere | adapter arm → plan entries → closure stub + `--write` (§7c), **in that order** |
| declared | throws | — | orphan closure entry | author plan entries |
| declared | OK | empty | declared and planned, not implementable | adapter arm only |
| `NOT DECLARED` | OK | empty | planned but never declared **or** implemented — usually a lane whose entries are all `candidate` scope, which no closure entry would make current anyway | adapter arm, then decide whether the entries should be re-scoped `authoritative`; only then a closure stub |
| declared | OK | non-empty | capturable | continue to §3 |

Row four is instantiated today: `candle:qwen_image_edit` has **9 plan entries, all `candidate`**, is
absent from the 10 declared lanes, and has no adapter arm (its rejection is pinned by
`candle.rs:1668-1673`). Its plan entries being candidate-scope is why nobody has missed the closure
entry — per §2b, candidate scope can never become current evidence.

**Worked example, and the reason this section exists** — `mlx:flux2_dev`, screened for sc-18104:

```
§2a  mlx:flux2_dev → NOT DECLARED   (9 declared lanes; the mlx side has 3)
§2b  Error: no plan entries for mlx:flux2_dev   (170 plan entries; 0 name it)
§2c  grep -rn flux crates/sceneworks-memory-adapter/src/ → no matches   (the WHOLE crate)
```

Row one. Note what is *not* wrong: `crates/media/mlx-gen/mlx-gen-flux2` exists at the pin and declares
`flux2_dev` (`mlx-gen-flux2/src/lib.rs:141`), FLUX.2 [dev] ships an `mlx` block in
`builtin.models.jsonc`, and a q8 tier directory is present on this Mac (unstamped — see §4a). **The
engine is real and the model shipped; only the calibration apparatus is missing.** That combination is
easy to mistake for "should
be a quick capture", which is exactly why §2 is four greps and not a judgement call.

Do **not** part-build the lane as consolation. Adding a closure stub for a lane with no plan entries
and no arm creates a digest nothing consumes, which then has to be re-derived when the arm actually
lands. Land the three pieces together or land none of them.

> The worked example is now wired for a real capture: sc-18218 adds the resident-only arm against
> the T2I provider's own reference-free contract, four authoritative q4/q8 plan entries at 768² and
> 1024², and the closure declaration together. SC-18218 then completed all four real-weight captures
> and bound them to the current provider closure: only q4/q8, reference-free T2I Resident cells are
> Verified. BF16 and every sibling mode, overlay, and rung remain Missing pending independent evidence
> on suitable hardware (sc-18104).

## 3. Pick the host

| lane | host | how CI reaches it |
| --- | --- | --- |
| `mlx:*` | Apple silicon — the dev Mac, which **is** the self-hosted `nax-macos` runner | `.github/workflows/macos-mlx.yml`, `runs-on: [self-hosted, macOS, ARM64, nax, weights]` when `run_memory_calibration` or `run_five_rung_reference` is set (macos-mlx.yml:429) |
| `candle:*` | a CUDA box | `.github/workflows/windows-candle.yml`, `runs-on: [self-hosted, Windows, X64, cuda]` (windows-candle.yml:266) |

Note the extra `weights` label on the MLX calibration dispatch: the ordinary MLX CI job runs on the
`nax` pool without it. A capture dispatch that cannot find a `weights`-labelled runner queues
forever rather than failing.

You can run the capture **locally on the host** or **through a guarded workflow dispatch**. §6 gives
both. The dispatch is preferable because it re-validates the identities for you, but only three lanes
are wired: `mlx:qwen_image` and `mlx:z_image_turbo` (two separate inputs on `macos-mlx.yml`) and
`candle:krea_2_turbo` (`windows-candle.yml`). Any other lane must go the local route.

## 4. Weights must really be present for the target tier

### 4a. First, list the tiers that are actually on disk

Before any verifier, answer the cheaper question: **which tier subdirectories exist at all?** A
multi-tier capture is quoted per tier, and a tier that is not on disk is a download, not a
measurement.

```bash
ls "$HOME/.cache/huggingface/hub/models--<Org>--<Repo>/snapshots/"*/
```

Measured on this Mac while validating this runbook (sc-18104):
`models--SceneWorks--flux2-dev-mlx/snapshots/2868b1461b2b…/` contains **only `q8/`** — 57 GiB on
disk. The manifest declares three tiers for that artifact, so a three-tier story would first have to
fetch `q4` (33.6 GB, and it is the `default: true` tier) and `bf16` (113 GB)
(`config/manifests/builtin.models.jsonc`, the `flux2_dev` `downloads` array). Neither absence is
visible from the repo directory or from the marker discussed next.

🔴 **`ls` the snapshot is not enough — the tier may not exist at the revision the manifest pins.**
Second worked example, `SceneWorks/ltx-2.3-mlx`, provisioned for sc-18809. The snapshot on this Mac
held `gemma/` + `q8/` only; the manifest declares four rows. `hf download … --include 'bf16/*'`
**exited 0 having fetched nothing**, because the pinned revision `254989c3…` (2026-06-14) predates
the upstream `bf16/` upload `01df27d3…` (2026-07-05) by three weeks. The glob was correct and the
tier was published — just not at the pin. So the download reported success, no directory appeared,
and the failure would have surfaced later as the engine's `missing transformer.safetensors`.

Two things follow, and both are now enforced rather than remembered:

- **Verify the pin, not the repo.** `node scripts/check-download-patterns.mjs` used to resolve every
  glob against the repo's *default branch*, so it passed this entry for weeks. As of sc-18809 it
  fetches `/api/models/<repo>/revision/<rev>` per entry and prints the revision it checked
  (`ltx_2_3/bf16  SceneWorks/ltx-2.3-mlx@254989c3ca7e  bf16/*`). Run it before quoting a tier as
  fetchable.

  As of sc-18854 this is **enforced on every PR**, without putting the HF API on a required lane.
  The check is split in two: `--write` is a networked RECORDER that transcribes one file listing per
  `repo@revision` key into `config/download-pattern-evidence.json`, and `--check` is a hermetic GATE
  that re-derives the claims from the manifests and grades them against those committed listings
  with no network at all. `npm run check` runs the gate, so it reaches check.yml's `parity-scaffold`
  and the required `parity` aggregator.

  **So the workflow when you touch a `downloads[]` entry or a LoRA `source` is: re-record, then
  commit the evidence with your manifest change.**

  ```
  npm run record:download-patterns   # networked; rewrites config/download-pattern-evidence.json
  npm run check:download-patterns:offline   # what CI runs; no network
  ```

  🔴 **`config/download-pattern-evidence.json` is GENERATED. Regenerate it — never hand-edit it.**
  The gate's entire trust chain terminates in that one multi-thousand-line file, and it is trusted
  unconditionally: a one-line edit to any `files` array turns a real zero-match green, and a `gated`
  flipped to `false` un-tracks an unfetchable repo. Neither is plausible to catch by eye in a review
  of a machine-generated diff. The only honest edit is a re-record. If you find yourself wanting to
  change the artifact by hand, the thing that is actually wrong is the manifest entry or the
  waiver list in `scripts/check-download-patterns.mjs`.

  **Reviewing someone else's evidence diff** — there is a no-write verify mode for exactly this:

  ```
  node scripts/check-download-patterns.mjs --write --dry-run
  ```

  It performs the same networked re-record, writes nothing, and ends with an
  `=== ARTIFACT VERDICT: … ===` banner. **The banner, not the absence of noise above it, is the
  signal:**

  - `=== ARTIFACT VERDICT: UP TO DATE (N key(s)) — exit 0 ===` — the committed artifact is
    byte-identical to what a re-record produces. That is what distinguishes an honest re-record
    from a hand edit.
  - `=== ARTIFACT VERDICT: WOULD CHANGE — exit 1 ===` — it is not. Re-record and **read the diff
    before concluding anything**: a manifest/evidence change or a hand edit both red this mode.

  🔴 **In this mode the exit code carries that verdict and nothing else** (sc-18854 review). The
  live zero-match / access-gated / redirect / untranslatable lists still print *above* the banner —
  today's tree prints `SERVED BY A DIFFERENT REPO (1)` on every clean run, with no
  `ACCESS-GATED` or `ZERO-MATCH PATTERNS` line — but they deliberately do not move the
  exit status, because an anti-tamper mode that reds unconditionally signals nothing. Grade those
  lists with `npm run check:download-patterns:offline`; that is the gate CI runs.

  Do **not** read the absent zero-match line as the two modes having converged. They answer
  different questions, and `scripts/check-download-patterns.test.mjs` no longer relies on a live
  catalog defect to prove it: the collision case now injects a bogus declared pattern into a
  throwaway copy of the catalog, which moves a live verdict without moving the artifact.

  `--dry-run` is rejected outright when it is not paired with `--write`, including in combination
  with `--check` or `--self-test` (both of which ignore it — they never touch the network or the
  artifact). Same reason: a flag that is silently swallowed lets a reviewer believe they verified
  the artifact when they did not.

  It is a **human tool and is deliberately wired into nothing**: it needs huggingface.co, and the
  entire point of the record/grade split is that no required lane depends on that.

  Forgetting the re-record is not a silent pass: adding or re-pinning an entry changes its
  `repo@revision` key, and a claim with no recorded listing reds the gate with the exact command to
  run. The recorder never evaluates a pattern — it only transcribes — so you cannot record your way
  out of a real zero-match. All 95 of 95 keys are pinned to immutable lowercase 40-hex revisions,
  whose listings cannot go stale. sc-18924 closed the last 11-key moving-default-branch window and
  made missing, branch, or tag revisions a hard offline-gate failure, so a new manifest entry cannot
  silently reopen it.

  Each recorded key also carries `gated` and `servedRepo`, and the gate hard-fails on both
  (`evidence-gated`, `evidence-repo-id-mismatch`) with tracked waivers in `KNOWN_REPO_CONDITIONS`.
  A green metadata listing does **not** mean a repo is fetchable: `gated: "auto"` answers 200 and
  then 401s the actual download unless the token's account has accepted that repo's licence.
  sc-18923 removed the two catalog instances by pinning checksum-identical public rehosts; any new
  gated source fails the offline gate unless it receives an explicit tracked waiver.
- **A tier that lands at a different revision is a resolution problem too.** The cache then holds two
  `snapshots/<rev>/` dirs with the tiers split across them, and `huggingface_snapshot_dir` selects
  exactly one. `ltx_bundle_subdir_across_revisions` (sc-18809) scans siblings with tier preference
  dominating revision, mirroring `bundled_ltx_gemma_dir`'s sc-14377 fix for the co-requisite TE.
  Without that a bf16 request silently renders at whatever tier sits in the selected snapshot.

Inventory after provisioning, following symlinks (`du -shL`), all four manifest rows now pinned to
`01df27d3…`: `gemma/` 26.4 GB, `q4/` 20.5 GB, `q8/` 29.7 GB, `bf16/` 47.1 GB. Blob-level dedup means
the bump itself re-downloads nothing — the cache is keyed by LFS digest and `01df27d3…` only *added*
`bf16/`.

🔴 **Record WHICH snapshot each tier landed in, not just that it landed.** "Provisioned at the
manifest revision" and "provisioned" are different claims, and this host is the worked example of the
gap. Actual layout, `models--SceneWorks--ltx-2.3-mlx/snapshots/`, with no `refs/main`:

| snapshot | holds | files |
| --- | --- | --- |
| `254989c3…` (pre-bump) | `gemma/` `q4/` `q8/` | **39** ← selected |
| `01df27d3…` (the manifest pin) | `gemma/` `bf16/` | 27 |

So `q4` and `q8` are on disk at the **old** revision, not the manifest's. Because blobs are shared by
LFS digest the bytes are identical either way, and `hf download` at the new pin would only re-link
them — this is bookkeeping, not a re-fetch. But it is exactly the layout that breaks a naive presence
check: with no `refs/main`, `resolve_huggingface_snapshot_dir` falls to "most files" and selects the
39-file **pre-bump** snapshot, which has no `bf16/` and never will. Any probe that looks only there is
permanently false — see `ensure_ltx_{q8,bf16}_present`, which sc-18809 had to move onto the same
cross-revision scan as the resolver for precisely this reason. When you write "tier X is provisioned"
into a handoff, name the snapshot.

⚠ **This step answers "which tiers exist", NOT "which tiers are usable" — do not let it become the
latter.** In that same worked example, `find … -name .sceneworks-model-revision` over the entire
`models--SceneWorks--flux2-dev-mlx` tree returns **nothing**, so by the very rule stated immediately
below, that `q8/` directory is *not* established as provisioned — a present tier directory is not a
complete one. The honest reading is "a q8 directory exists, unverified", and it still owes the §4
verifier. Write it that way in any story you hand off, because "the weights are already there" is
exactly the sentence a downstream capture plan will inherit as settled fact.

Then, and only then:

**`.sceneworks-model-revision` is the only sound POSITIVE signal.** It lives inside
`snapshots/<revision>/`, not in the repo directory — looking in the repo dir reads as a false
negative. SceneWorks never writes it. Its **presence** means the snapshot was provisioned by tooling
that stamps it, and it authoritatively names the revision when the directory itself is not named by
one. Its **absence proves nothing** — an app-installed or hand-fetched snapshot is complete and
unmarked. Do not treat a missing marker as a missing snapshot, and do not treat a present tier
directory as a complete one.

> Authorship, stated precisely because it is easy to get wrong: at pin `40fa7583` every reference to
> the marker in the inference tree **reads** it — `scripts/release/verify_model_snapshot.py:77-80`,
> `scripts/release/provision_mage_oracles.py:155-168`, `.github/workflows/real-weights.yml:1318-1325`
> — and `git grep` finds no writer at that revision. SceneWorks only reads it too
> (`scripts/audit-hf-cache-liveness.mjs:408`). So do not attribute the write to `verify_model_snapshot.py`;
> the load-bearing claims above hold regardless of which provisioning path stamped it.

The authoritative completeness test is the pinned verifier, run from a checkout of inference **at the
exact pinned SHA** (§5):

```bash
python3.12 scripts/release/verify_model_snapshot.py \
  --model <model-key> \
  --snapshot "$HOME/.cache/huggingface/hub/models--<Org>--<Repo>/snapshots/<exact-revision>" \
  --manifest release/real-weight-models.toml
```

It checks every `expected_files` entry **and** every shard named by any `*.index.json` those files
reference, and prints `model snapshot: OK (<key>@<revision>)` on success. Add
`--inventory-output <path>` to also emit a canonical dereferenced content inventory with an
`inventory_sha256` — that is the artifact to cite when binding a fixture to real weights.

Two caveats, both measured:

- **A partial tier fails loudly, which is the point.** Run against this Mac's
  `SceneWorks/qwen-image-mlx` snapshot `8080a417…`, which holds only `q8/`, the verifier exits 1 with
  `qwen-image-mlx snapshot is incomplete; missing: bf16/model_index.json, …`. That is the check
  earning its keep — the directory exists and looks fine.
- **Not every calibration lane has a manifest entry.** At pin `40fa7583`,
  `release/real-weight-models.toml` has 62 `[[models]]` keys, and `SceneWorks/z-image-turbo-mlx`,
  `SceneWorks/krea-2-turbo-mlx` and `SceneWorks/flux2-dev-mlx` are **not** among them
  (`z-image-turbo` there is the upstream `Tongyi-MAI` repo, a different artifact; the only flux2 key
  is `flux2-klein-9b-mlx-q4`, which is **klein**, a different model from FLUX.2 [dev] — do not
  mistake it for a binding on this lane). For those lanes there is no `verify_model_snapshot`
  binding to run — fall back to the workflow's own resolver contract below and say plainly in the PR
  that no manifest-bound verification exists for the lane. sc-18213 tracks closing this gap; check it
  before concluding a lane is permanently unbound.

### Where the adapters look — one resolver block PER LANE

Every adapter arm canonicalizes its root and requires a fixed
`/models--<Org>--<Repo>/snapshots/<exact-revision>/<tier>` suffix before loading, so a stale override
for another tier is rejected rather than silently used. But **each lane has its own resolver block
with its own hardcoded tier** — take the one for your lane, not the first one in the file:

| lane | workflow resolver | canonical locations |
| --- | --- | --- |
| `mlx:qwen_image` | macos-mlx.yml:726-762 (tier from the `qwen_tier` input) | `$HOME/.cache/huggingface/hub/models--SceneWorks--qwen-image-mlx/snapshots/<rev>/<tier>`, then `$HOME/Library/Application Support/SceneWorks/data/cache/huggingface/hub/…/<rev>/<tier>` (macos-mlx.yml:742-747) |
| `mlx:z_image_turbo` | macos-mlx.yml:763-789 (**tier hardcoded `q4`**) | the same two roots under `models--SceneWorks--z-image-turbo-mlx/snapshots/<rev>/q4` (macos-mlx.yml:772-773) |
| `candle:krea_2_turbo` | windows-candle.yml:430-450 (**tier hardcoded `q4`**) | a **single** root, no app-data fallback: `%USERPROFILE%\.cache\huggingface\hub\models--SceneWorks--krea-2-turbo-mlx\snapshots\<rev>\q4` (windows-candle.yml:440) |

Each also honours an optional repository-secret override (`SCENEWORKS_QWEN_IMAGE_ROOT`,
`SCENEWORKS_Z_IMAGE_ROOT`, …), used only when it canonicalizes to a path ending in that lane's exact
suffix.

### Adapter environment — one family per provider arm

The derivation rule: **each provider arm reads `SCENEWORKS_<ARTIFACT>_{REPOSITORY,REVISION,ROOT}`**,
where `<ARTIFACT>` names the artifact family the arm loads, not the provider id verbatim
(`z_image_turbo` → `Z_IMAGE`, `krea_2_turbo_control` → `KREA_CONTROL`). Confirm yours rather than
inferring it:

```bash
grep -oE 'SCENEWORKS_[A-Z_]+' crates/sceneworks-memory-adapter/src/bin/<backend>.rs | sort -u
```

Verified complete set today:

```bash
# memory-mlx-adapter — qwen_image
SCENEWORKS_QWEN_IMAGE_REPOSITORY=SceneWorks/qwen-image-mlx   # fixed; validated against QWEN_REPOSITORY
SCENEWORKS_QWEN_IMAGE_REVISION=<exact artifact revision>
SCENEWORKS_QWEN_IMAGE_ROOT=/abs/path/.../snapshots/<rev>/<tier>    # bf16 | q4 | q8

# memory-mlx-adapter — z_image_turbo   (mlx.rs:1063-1076)
SCENEWORKS_Z_IMAGE_REPOSITORY=SceneWorks/z-image-turbo-mlx   # fixed; validated against Z_IMAGE_REPOSITORY
SCENEWORKS_Z_IMAGE_REVISION=<exact artifact revision>
SCENEWORKS_Z_IMAGE_ROOT=/abs/path/.../snapshots/<rev>/q4     # tier hardcoded q4

# memory-mlx-adapter — krea_2_turbo (plain text-to-image)
SCENEWORKS_KREA_REPOSITORY=SceneWorks/krea-2-turbo-mlx       # fixed; validated against KREA_REPOSITORY
SCENEWORKS_KREA_REVISION=<exact base artifact revision>
SCENEWORKS_KREA_ROOT=/abs/path/.../snapshots/<rev>/<tier>    # bf16 | q4 | q8, derived from the plan target

# memory-mlx-adapter — sdxl (plain text-to-image)
SCENEWORKS_SDXL_REPOSITORY=SceneWorks/sdxl-base-mlx          # fixed; validated against SDXL_REPOSITORY
SCENEWORKS_SDXL_REVISION=<exact base artifact revision>
SCENEWORKS_SDXL_ROOT=/abs/path/.../snapshots/<rev>/<tier>    # bf16 | q4 | q8, derived from the plan target

# memory-mlx-adapter — krea_2_turbo_control   (mlx.rs:1717-1735)
SCENEWORKS_KREA_CONTROL_REPOSITORY=SceneWorks/krea-2-turbo-mlx  # validated against KREA_REPOSITORY
SCENEWORKS_KREA_CONTROL_REVISION=<exact base artifact revision>
SCENEWORKS_KREA_CONTROL_ROOT=/abs/path/.../snapshots/<rev>/q4   # tier hardcoded q4
SCENEWORKS_KREA_CONTROL_OVERLAY=/abs/path/to/control_step5000.safetensors
SCENEWORKS_KREA_CONTROL_OVERLAY_REVISION=<exact overlay revision>
# overlay artifact: SceneWorks/krea2-pose-controlnet-beta / control_step5000.safetensors
# (mlx.rs:47-48); the overlay path is validated against that repo + revision before load

# memory-mlx-adapter — flux2_dev   (sc-18218; resident rung only)
SCENEWORKS_FLUX2_REPOSITORY=SceneWorks/flux2-dev-mlx         # fixed; validated against FLUX2_REPOSITORY
SCENEWORKS_FLUX2_REVISION=<exact artifact revision>
SCENEWORKS_FLUX2_ROOT=/abs/path/.../snapshots/<rev>/<tier>   # q4 | q8 — tier DERIVED from the plan target

# memory-mlx-adapter — ltx_2_3   (sc-18808; the MLX video arm. FOUR vars, not three)
SCENEWORKS_LTX_REPOSITORY=SceneWorks/ltx-2.3-mlx             # fixed; validated against LTX_REPOSITORY
SCENEWORKS_LTX_REVISION=<exact artifact revision>
SCENEWORKS_LTX_ROOT=/abs/path/.../snapshots/<rev>/<tier>     # bf16 | q4 | q8, derived from the plan target
SCENEWORKS_LTX_TEXT_ENCODER_ROOT=/abs/path/.../snapshots/<rev>/gemma
# The Gemma-3-12B co-requisite is a HARD load-time requirement of the pinned provider
# (`resolve_gemma_dir`; sc-13664 removed the env/HF-cache fallbacks), rides `LoadSpec::text_encoder`,
# and is snapshot-validated against the SAME repository and revision as the tier root — a mismatched
# TE silently changes the measured conditioning peak. Both roots must therefore resolve under one
# revision. On this host q4/q8 originally materialised under the pre-bump snapshot `254989c3…` while
# the manifest pins `01df27d3…`; `hf download --revision 01df27d3… --include 'q8/*' --include 'q4/*'`
# re-links them at the manifest revision for **zero bytes**, because the blobs are shared (sc-18810).

# memory-mlx-adapter — any lane, optional
SCENEWORKS_MLX_WIRED_LIMIT_BYTES=<explicit wired-ceiling override>

# memory-candle-adapter — krea_2_turbo
SCENEWORKS_KREA_REPOSITORY=SceneWorks/krea-2-turbo-mlx
SCENEWORKS_KREA_REVISION=<exact artifact revision>
SCENEWORKS_KREA_ROOT=/abs/path/.../snapshots/<rev>/q4

# memory-candle-adapter — wan2_2_ti2v_5b   (sc-19057; the only candle VIDEO arm. See §12)
SCENEWORKS_WAN_REPOSITORY=SceneWorks/wan2.2-ti2v-5b-candle  # fixed; validated against WAN_CANDLE_REPOSITORY
SCENEWORKS_WAN_REVISION=<exact artifact revision>
SCENEWORKS_WAN_ROOT=/abs/path/.../snapshots/<rev>/<tier>    # q4 | q8 ONLY — bf16 is the upstream
# dense Wan-AI/Wan2.2-TI2V-5B-Diffusers snapshot, which carries no per-tier subdirectory for
# validate_huggingface_snapshot_root to bind an artifact identity to, so that tier has no arm.
```

All three of each family are **required** (`protocol::required_env`) — a missing one fails before
model load, not after. The plain MLX Krea arm is reference-free, rejects overlays and PiD, and runs
the production deferred-materialization shape across q4, q8, and bf16. Its plan deliberately has no
evidence records or manifest bindings until a physical Apple-Silicon capture is executed and reviewed.
The plain SDXL arm has the same artifact and surface guards, but plans only its three implemented
rungs: Resident, Staged, and bounded-transformer residency across the exact cadence domain
`1, 2, 5, 10`. Its measured-Missing decode and attention rungs are absent by design, and SC-18379
likewise adds no evidence or binding.

### If the snapshot is absent

Provision it through the workflow rather than by hand — that path is pinned and resumable. On the
MLX lane, `provision_qwen_snapshot=true` (rejected unless `run_memory_calibration=true`) creates a
job-local venv and pins `huggingface_hub==0.36.0` to fetch only `<tier>/**` from the fixed public
repository at the exact revision, with `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`. Only a provisioning
dispatch gets the extended 240-minute timeout; ordinary calibration keeps 45 minutes
(macos-mlx.yml:452).

**`python3` is NOT `python3.12` on this host.** Measured here: `python3 --version` → `Python 3.14.6`,
`python3.12 --version` → `Python 3.12.13`. The workflow asserts the exact spelling —
`command -v python3.12` then `python3.12 --version | grep -E '^Python 3\.12\.'` (macos-mlx.yml:633-635)
— and any local provisioning you do by hand must use `python3.12` too. The Windows CUDA lane is
looser: it accepts `python` at 3.12 **or newer** (windows-candle.yml:393-403).

## 5. Clean checkouts at exact SHAs

The harness stamps every record with each repository's `dirty` flag from `git status --porcelain`,
**which counts untracked paths**, and rejects complete evidence from a dirty repository:
`complete evidence cannot come from a dirty repository`
(memory-calibration-harness.mjs:288-289, and :412-413 for `runtime_complete`). It re-probes after
every provider case and aborts with `repository HEAD or dirty state changed during provider
execution` if anything moved (memory-calibration-harness.mjs:967-969).

So:

1. **Commit or stash everything first**, in *both* repositories.
2. **Change nothing in either repository while the capture runs.** Not a scratch file, not a log.
3. Check out inference at the adapter's compiled pin, not at whatever your clone happens to be on:

```bash
sed -nE 's/^pub const INFERENCE_PIN: &str = "([0-9a-f]{40})";/\1/p' \
  crates/sceneworks-memory-adapter/src/lib.rs
# 40fa7583a01974617e2a7275052d6d446688c956 at the time of writing
git -C <inference-clone> checkout <that SHA>
```

Both capture workflows enforce that equality themselves and refuse the dispatch on mismatch
(macos-mlx.yml:601-605, windows-candle.yml:386-389).

**There are two spellings of the pin, and different tools read different ones.** The adapter's
`INFERENCE_PIN` above is what the built binary stamps into its records;
`scripts/inference-closure-digest.mjs` instead defaults `--revision` to the pin it parses from the
workspace `Cargo.toml` (`inferencePinFromCargo`, `inference-closure-digest.mjs:671`). They agree today
— both read `40fa7583`, and `scripts/bump-inference.mjs` moves them in lockstep and asserts the
constant is rewritten (`bump-inference.mjs:99,161`) — so this is a thing to know, not a thing to fix.
Do not bump either by hand; use `bump-inference.mjs`. And keep this distinct from §7d's
`inferenceRevision`, which is a **manifest binding field** naming the revision a measurement was taken
at, not a build pin.

## 6. The capture

`--output` and `--resume` should point **outside the repository**. `.tmp/` is gitignored today
(`.gitignore:298-301`, added deliberately for this), so it also works — but the failure mode when an
ignore rule changes is a multi-hour sweep discarded at the end, and the CI lanes themselves write to
`$RUNNER_TEMP`. Use a path outside the tree and the question does not arise.

For physical MLX capture, the raw-log directory must also stay outside the checkout. Run
`scripts/hash-artifact-inventory.mjs --root <exact-tier-root> --github-env <env-file>` once before
the campaign, export the two inventory values it reports, and set
`SCENEWORKS_MEMORY_CAPTURE_DIR` plus `SCENEWORKS_MEMORY_SOURCE_PATH_PREFIX`. The adapter persists the
selected and reference RGB bytes under `<capture-dir>/<source-prefix>` using role- and
content-addressed filenames with exclusive creation. The adapter separately attests each RGB
receipt's SHA-256 and byte count; the harness requires those values to match the role/dimensions/hash
encoded in the filename and the bytes it reads after the provider exits. The harness adds one
`physical_mlx` source session per fresh case and writes the exact request and provider response beside
them. A physical session is invalid unless it carries exactly one typed `request`, `selected_rgb`,
and `reference_rgb` receipt at three distinct paths, so removing an entry cannot make a missing file
disappear from validation. The request receipt must be canonical JSON for the one record bound to the
session, and both RGB receipts must match that record's logical case and geometry. The temporary
directory therefore mirrors the repository-relative tree. Validation also reconstructs the full
evidence record from the immutable provider response, request, and artifact input, requires every
adapter-owned measurement/quality/sweep/scenario field to match, and re-derives the session ID from
the response digest and capture provenance. Session repositories, capture time, hardware, and compact
target/rung must exactly match the request and record. Changing a log and merely updating `stdoutSha256` is not
valid provenance. The inventory command
streams and hashes every artifact byte, including Hugging Face symlink targets.

### 6a. Locally on the host

> Provenance: the adapter builds and `harness run` were **not** executed while writing this runbook —
> a real capture is a multi-hour GPU sweep on dedicated hardware. Every flag below is transcribed
> from `memory-calibration-harness.mjs:1186-1233` and from the two checked-in capture workflows,
> which run exactly these commands. `harness check` WAS executed (on the committed bundle).

```bash
# Apple silicon
cargo build --release --locked -p sceneworks-memory-adapter \
  --features mlx --bin memory-mlx-adapter

# CUDA host (in the supported CUDA + host-compiler environment)
cargo build --release --locked -p sceneworks-memory-adapter \
  --features candle --bin memory-candle-adapter
```

```bash
node scripts/memory-calibration-harness.mjs run \
  --config config/memory-calibration-plan.json \
  --backend <backend> \
  --fixture <fixture from §2b> \
  --fresh-per-case \
  --provider-command '["/abs/path/to/target/release/memory-<backend>-adapter"]' \
  --sceneworks-repo /abs/path/to/SceneWorks \
  --inference-repo /abs/path/to/inference \
  --resume docs/generated/memory-calibration-evidence.json \
  --output /abs/path/OUTSIDE/the/repo/<lane>-evidence.json
```

🔴 **On an MLX physical-source arm only, add the raw-log pair.** These two flags are the
physical-source-capture path and they are MLX-only, so they are NOT in the invocation above:

```bash
  --raw-log-dir /abs/path/OUTSIDE/the/repo/raw-receipts \
  --source-path-prefix docs/calibration/<story-or-campaign> \
```

The harness refuses any case where `--raw-log-dir` is set and the provider returned no
`sourceCapture`, and accepts only `sourceCapture.kind === "physical_mlx"`. Only
`crates/sceneworks-memory-adapter/src/bin/mlx.rs` emits that key; `bin/candle.rs` emits none, so on
a candle lane the sweep dies *inside the per-case loop* — after the first row's cold model load —
rather than at argument parsing. The two are also validated as a pair, so supply both or neither.

Then schema-check the raw bundle before doing anything else:

```bash
node scripts/memory-calibration-harness.mjs check \
  --input /abs/path/OUTSIDE/the/repo/<lane>-evidence.json
```

Flag notes, all from `memory-calibration-harness.mjs:1186-1233`:

- `--backend` is **required** whenever the plan contains both backends; one provider process probes
  one backend-specific hardware shape.
- `--provider-command` is a **JSON argv array**, quoted as one shell word.
- `--fresh-per-case` forces one fresh process per case (the oracle shape both CI lanes use);
  `--batch-rungs` forces one target's rungs into an experimental batch.
- `--resume <bundle>` suppresses already-complete cases. The runner **atomically checkpoints the
  accumulated schema-valid bundle after every successful provider response**, so an interrupted
  sweep can be resumed by pointing `--resume` at the partial output and re-running. Only schema-valid
  `complete` records suppress their executed passing cases; a gated or candidate record suppresses a
  repeated *attempt* only when its logical case, harness version, repository receipts and hardware
  probe all match.

### 6b. Through the guarded dispatch

> Provenance: transcribed from the workflow files and **not** dispatched while writing this runbook —
> either dispatch starts a real capture on a self-hosted runner.

Three dispatches exist, one per wired lane. `run_memory_calibration` and `run_five_rung_reference`
are **separate inputs on the same MLX workflow**; the latter is not "the Windows one".

**`mlx:qwen_image`** — `run_memory_calibration` (macos-mlx.yml:113-149, 799-831):

```bash
gh workflow run macos-mlx.yml --ref main \
  -f run_memory_calibration=true \
  -f provision_qwen_snapshot=false \
  -f qwen_tier=bf16 \
  -f inference_revision=<exact adapter INFERENCE_PIN> \
  -f qwen_repository=SceneWorks/qwen-image-mlx \
  -f qwen_revision=<exact 40-hex artifact revision> \
  -f qwen_source_path_prefix=docs/calibration/<story-or-campaign>
```

Validates the identities, resolves (without printing) the snapshot root, checks out inference at the
exact revision into `.calibration/inference`, builds the release adapter, runs the harness with
`--fresh-per-case` on fixture `qwen-image-<tier>-seed<15511|16353>-step2` (seed 15511 for `bf16`,
16353 for the packed tiers — macos-mlx.yml:809-813), schema-checks the bundle and uploads it as
`memory-mlx-evidence-<tier>-<run_id>`. That artifact contains the bundle plus the immutable raw log
and selected/reference outputs in their repository-relative tree. Validate a downloaded artifact
with `memory-calibration-harness.mjs check --input <bundle> --source-root <unpacked-raw-root>`, then
copy its `docs/calibration/...` tree into the checkout before ingest. Both `check` and `ingest` verify
the provider log, exact request, and selected/reference output bytes for every `physical_mlx` session;
a missing or altered file fails closed.

**`mlx:z_image_turbo`** — `run_five_rung_reference` on the **same** workflow (macos-mlx.yml:150-163,
832-868):

```bash
gh workflow run macos-mlx.yml --ref main \
  -f run_five_rung_reference=true \
  -f provision_z_image_snapshot=false \
  -f inference_revision=<exact adapter INFERENCE_PIN> \
  -f z_image_repository=SceneWorks/z-image-turbo-mlx \
  -f z_image_revision=<exact 40-hex artifact revision>
```

No tier input — the fixture is fixed at `fresh-five-rung-z-image-q4-768-seed16402-step2`. It first
runs `assess-reuse` and **asserts the verdict is `unable_to_amortize`** before capturing
(macos-mlx.yml:841-848), then runs `--fresh-per-case` and uploads
`sc-16059-mlx-reuse-{run_id}` containing the bundle and the assessment.

**`candle:krea_2_turbo`** — `run_five_rung_reference` on `windows-candle.yml` (:134-163, 460-476):

```bash
gh workflow run windows-candle.yml --ref main \
  -f run_five_rung_reference=true \
  -f provision_krea_snapshot=false \
  -f inference_revision=<exact adapter INFERENCE_PIN> \
  -f krea_repository=SceneWorks/krea-2-turbo-mlx \
  -f krea_revision=<exact 40-hex artifact revision>
```

Captures fixture `fresh-five-rung-krea-q4-1024-seed16402-step2` **twice** — once `--fresh-per-case`,
once `--batch-rungs` — schema-checks both and runs `compare-reuse` between them.

Retrieve any of them with `gh run download <run-id>`.

### 6c. What it costs

Originally derived from `docs/generated/calibration-cost-model.json`. **Provenance note:** sc-18100
deleted that generator and its artifacts, which is why these figures were inlined here as facts
rather than linked. They are a snapshot of that model's last published run and should be re-derived
from a real capture rather than trusted forward.

- **Per-run seconds is an ASSUMPTION, not a measurement.** The model defaults to **300 s per
  certifying record** and states plainly that nothing in the repository establishes it. The published
  sweep runs 30 / 60 / 120 / 300 / 600 / 1200 s per run; the sweep, not the default, is the answer.
- **The one real anchor**: 8 real MLX provider invocations plus one hardware probe and a schema check,
  on an Apple M5 Max self-hosted runner — **30.76 s wall clock** for the whole harness step, i.e.
  **3.85 s per invocation**. This is a hard **floor** and known not to cover the dominant cost: those
  records are `gated`, exercising the VAE decode seam only — no text-encoder or DiT load, no
  conditioning, no denoise, no lifecycle injection, no warm A→B→A.
- **A five-rung lane is 5 certifying records** (one per cell), each a fresh model load under
  `--fresh-per-case`. At the assumed 300 s that is ~25 minutes of provider time plus five cold model
  loads; the floor anchor says the harness overhead itself is seconds. Budget hours, not minutes, and
  do not promise a number — measure yours and report it.
- **I/O is not the bottleneck**: warm-page-cache sequential reads of real safetensors shards on this
  Mac measured 11.3 GB/s and 12.3 GB/s, so a 20 GiB tier streams in under 2 s. Dequantisation, graph
  construction and denoise are unmeasured.
- **Disk**: the largest MLX calibration snapshot is approximately **57 GiB**
  ([memory-calibration-harness.md](memory-calibration-harness.md)), which is why only a provisioning
  dispatch gets the 240-minute timeout. Have that much free before provisioning, per tier.
- 🔴 **"All three tiers" is a claim about the HOST, not just the model.** A tier can be shipped,
  downloadable and still unmeasurable on the machine in front of you, and the manifest usually already
  says so. Check the tier's own memory note against the host before promising three tiers. Worked
  example (sc-18104): this Mac is an M5 Max with **128 GiB** unified memory (`sysctl hw.memsize` →
  `137438953472`), and `flux2_dev`'s bf16 download comment records an on-device finding from sc-8513
  — *"256² fits a 128 GB box, production 1024² wants >128 GB"*. So on this host the bf16 tier of that
  lane cannot be measured at the production geometry at all; three-tier coverage there needs a
  ≥192 GB Mac, or bf16 rows captured at a reduced geometry and labelled as such. Establish this
  before you accept a three-tier scope, not after two tiers have already been swept.
- 🔴 **…but establish it by MEASURING, because the obvious arithmetic over-counts a staged pipeline.**
  Counter-example, `mlx:ltx_2_3` bf16, settled in sc-18809. The arithmetic said no: 47.1 GB of dense
  bf16 tier plus the 26.4 GB Gemma co-requisite is **73.5 GB of weights** before a single video
  activation, on a 128 GiB box, across an envelope reaching 449 frames. That is the same shape as the
  `flux2_dev` finding above, and the epic carried it as a live risk. **It was wrong**, because the two
  giants never co-reside: sc-10976 stages the text phase (build TE → `encode_av` → `eval` → drop →
  `clear_cache()`) *before* the AvDiT materializes, so the weights floor is `max(TE, DiT)`, not their
  sum. Measured with the committed real-weights test (below), the bf16 co-residence estimate is
  69.20 GiB while the actual staged peak at the same geometry is **36.89 GiB** — 47% lower. Whenever a
  provider stages components, an additive weights sum is an upper bound that can be off by nearly 2×;
  read the provider's load path before quoting it, and prefer one cheap measured load over the sum.

### bf16 feasibility, measured — the shape of an answer this section wants

Host: Apple M5 Max, **128 GiB** unified (`sysctl hw.memsize` → `137438953472`). The ceiling that
actually binds is Metal's `recommendedMaxWorkingSetSize` = `115448725504` B = **107.52 GiB**
(`MTLCreateSystemDefaultDevice().recommendedMaxWorkingSetSize`); `maxBufferLength` is 80.64 GiB, which
bounds any *single* allocation and is nowhere near binding here. Quote both — `hw.memsize` alone
overstates what Metal will wire.

Instrument, on real weights at inference pin `b965641e`:

```bash
cargo test -p mlx-gen-ltx --release --test sequential_residency_real_weights -- --ignored --nocapture
# with LTX_MODEL_DIR=<snapshot>/<tier> and LTX_GEMMA_DIR=<snapshot>/gemma
```

It brackets a real staged `generate` in `reset_peak_memory()` / `get_peak_memory()`. Confirm the run
was not skipped — an `#[ignore]`d test that returns early still reports `ok`, so require the printed
line and a non-trivial wall time.

🔴 **Read the reproducibility column before you use any of these numbers.** The command above
reproduces the **first two rows only**. That test's geometry is *hardcoded* at 256×256×9 — verified at
inference pin `b965641e` and byte-identical at inference `HEAD` (`1ee44388`), with a clean working
tree. The two larger rows were captured by hand-editing that geometry onto env vars locally; **that
edit was never committed and no longer exists anywhere**, so those rows cannot be re-run from any
revision of either repo as written.

| tier | geometry | frames | stage-2 latent tokens | DiT weights | **staged peak** | wall | reproducible? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| q4 | 256×256 | 9 | 128 | 10.57 GiB | 33.10 GiB | 15.6 s | ✅ via the command above |
| bf16 | 256×256 | 9 | 128 | 35.37 GiB | 36.89 GiB | 13.6 s | ✅ via the command above |
| bf16 | 768×512 | 145 | 7 296 | 35.37 GiB | 43.87 GiB | 90.6 s | ❌ one-off, uncommitted instrumentation |
| bf16 | **1280×704** | **449** | **50 160** | 35.37 GiB | **87.07 GiB** | 847.8 s | ❌ one-off, uncommitted instrumentation |

Lifting the geometry onto env vars in the committed inference test is **sc-18856**; until that lands,
treat the bottom two rows as **anecdote with a number attached** — good enough to have retired a scope
risk, not good enough to fit a curve on.

**Verdict: bf16 ran to completion at the manifest maximum on this host, and the reason the planning
arithmetic said it could not is structural.** The last row is that maximum — the largest
`limits.resolutions` entry at `hardMaxDuration` 15 s × the fastest declared 30 fps, so 450 raw frames
snapped to 449 by LTX's `8k + 1` stride. It completed and returned all 449 frames.

Separate the two claims, because they are not equally earned:

- **The structural finding is solid**: the weights floor is `max(TE, DiT)`, not `TE + DiT`, because
  sc-10976 stages and drops the text phase before the AvDiT materializes. This is visible in the
  *reproducible* rows alone (36.89 GiB staged vs a 69.20 GiB co-residence estimate) and it does not
  depend on the two one-off captures at all.
- 🔴 **The "19% headroom" figure is NOT yet earned.** It reads `87.07` against the 107.52 GiB ceiling,
  and every one of the caveats below cuts in the same direction — the true production peak is
  *higher* than 87.07, by an unmeasured amount. Do not plan against 20 GiB of slack.

⚠ **Caveats that bound what this table can be used for.** All four apply to every row:

1. **All four rows ran `video_mode: "no_audio"`.** Per inference `mlx-gen-ltx/src/model.rs`, that
   gate (`if Self::no_audio(req)`) skips `decode_audio_track` — the audio VAE *and* the vocoder — which
   a **default** production LTX job runs. The table measures a non-default path, and the omitted decode
   sits at the 449-frame tail where memory is already highest.
2. **`get_peak_memory()` is MLX *active* memory and excludes the allocator cache**, but the 107.52 GiB
   `recommendedMaxWorkingSetSize` ceiling bounds active **+** cache. `clear_cache()` only runs at phase
   boundaries, so cache growth *inside* the 449-frame decode is unaccounted for on the very row that
   defines the envelope. `get_cache_memory()` was not logged.
3. **n = 1, at the maximum envelope.** One sample, no variance, no re-run — on the row carrying the
   entire feasibility claim.
4. **Captured on an otherwise-idle host**, with nothing else resident. A real worker shares the box
   with the API process, the OS, and anything else the user is running; none of that is accounted for.

Two things fell out that the sweep should **re-measure before building on**, not consume:

- **Peak looks linear in stage-2 latent token count**, `T_lat · (H/32) · (W/32)` with
  `T_lat = 1 + (frames − 1)/8`. Across the three bf16 rows the marginal cost is 0.997 then
  1.032 MiB/token, and `36.89 GiB + 1.0 MiB × tokens` predicts the max-envelope peak as 85.87 GiB
  against 87.07 GiB measured — 1.4% error over a **392×** token range. Neither raw frame count nor
  area alone can do that; the product is the right regressor *shape*.
  🔴 **But two of the three points are the non-reproducible rows, so this is a hypothesis with n = 3,
  not a fit.** sc-18810 must **re-measure these points itself** once sc-18856 commits the geometry
  knob, and fit on its own captures — do not inherit `1.0 MiB/token`, `85.87 GiB`, or the 1.4% error
  as settled inputs.
- **Which phase is the floor flips with tier.** For q4 the text phase dominates (peak 33.10 GiB with a
  10.57 GiB DiT); for bf16 the DiT does (35.37 GiB). A model that assumes one or the other is wrong
  for half the tiers of the same lane. This one rests entirely on the two **reproducible** rows.

**What a sweep may assume on this host.** bf16 completed at the maximum `limits` combination — the
five declared resolutions (0.41–0.90 MP) crossed with durations 4–15 s and fps 24/25/30, i.e. 96–449
frames — and q8 (20.6 GiB DiT) and q4 (10.6 GiB) sit strictly below it in the DiT phase. So the lane
needs **no reduced-geometry caveat and no ≥192 GB Mac**, and three-tier coverage is in scope.

🔴 **What it may NOT assume: a specific headroom number.** The measured 87.07 GiB was taken with audio
decode skipped, without cache accounting, once. Re-measure at the geometry you actually intend to
sweep, with audio on, before treating any margin as spendable. Budget wall clock too: the
max-envelope bf16 row took **14.1 minutes**, and `nax-worker` caps at 240 minutes per dispatch.

### sc-18810 re-measured it through the committed apparatus — three of those claims did not survive

Every number below is **measured** — from `docs/generated/ltx-mlx-geometry-sweep-sc-18810.json`,
captured through the sc-18808 MLX arm and `scripts/memory-calibration-harness.mjs` on the
**production full-A/V path** (`video_mode` unset, audio track decoded — the row above used
`no_audio`), q8, inference pin `b965641e` — **except where a paragraph says PREDICTED**, which means
it is computed from the engine's committed `LTX_VAE_*` cost model and has no record behind it. §5 is
the only such paragraph. 13 records over **8 captured fixtures spanning 7 distinct {w,h,frames}
geometries plus one fps probe** — the probe repeats `{768x512, f241}` at 24 fps, and §3 below argues
that is the same geometry, so it may not also be counted as an eighth one. Replicates on four of them
give a measured noise floor rather than an assumed one: decode is byte-identical across repeats,
denoise varies by 0.001% (2.7e-4 GiB), and the text phase by 0.33% (0.110 GiB).

🔴 **That noise floor is CROSS-SESSION and CROSS-REVISION, and every "× noise" statement below
inherits it.** The capture crashed the host after four records and resumed later, so it spans two
driver sessions and **four SceneWorks revisions** — `f301d712` (session 1, 4 records), then
`817cf550`, `41e8151f` and `c30c4974` (session 2, 9 records). Both sessions' logs ship
(`docs/calibration/sc-18810/precrash-q8-run.log` and `sweep-run.log`) and
`ltx-temporal-form-fit-sc-18810.json`'s `sourceSessions` says which records came from which; the
original PR committed only the second log, so four records had no terminal line anywhere. **Every
one of the four replicate groups contains one record from each session**, so not one of them is a
back-to-back repeat: the floor bounds repeat-plus-revision variation, not repeat variation alone. The
text floor's own maximum spread happens to fall between two session-2 records (`41e8151f` 33.2283
vs `817cf550` 33.1188 GiB), so it is cross-revision even setting the crashed session aside. This is
reported rather than corrected because correcting it means re-rendering, and a floor that is too
WIDE is the conservative direction for a residual to be compared against.

⚠️ The evidence **bundle**'s own `sourceSessions` is `[]` for this lane, as it is for sc-18808: the
harness only populates it on its capture path and these records were ingested without it. The
per-session provenance above is derived from the committed logs by
`scripts/fit-ltx-temporal-form.mjs`, and a captured record with no `OK` terminal in a committed log
now **throws** rather than shipping — which is the guard that would have caught the original gap.

**1. The peak is not one curve — it is the max of three, and only the phases are fittable.** The
shipped structure already does this (`KreaTurboPhasePeaks::peak_gb`), and it is load-bearing rather
than incidental. Fitting the *overall* peak with any single form leaves a held-out error of
**≥10.26 GiB** (94× the text-phase noise floor) for all five candidate forms, because a max of linear
pieces is not linear. Fitting each phase separately, and taking each phase's OWN best form, lands at
**0.019–0.30 GiB** — decode 0.019 (`cross`), denoise 0.13 (`latent_tokens`), text 0.30 (`area_only`).
⚠️ That band is a per-phase *best-of*, and it is **not** the band any single shipped form achieves:
the `cross` form sc-18812 actually adopts spans **0.019–0.44 GiB** (0.0185 / 0.1741 / 0.4438). Citing
0.019–0.30 as `cross`'s own accuracy is the mis-read that shipped once already, in the
`perMpxFrameGb` schema description. Add the temporal term *per phase*; never to the aggregate.

**Admission-margin verdict (SC-18829): keep the ratified constants unchanged.** Runtime adds each
phase's maximum fit/held-out absolute residual before taking the maximum over all three phases, and
only then applies the ordinary backend estimate margin. This construction is explicitly exempt from
the measured-binding-phase pin: it does not assume that one phase remains binding, because every
phase is independently represented and residual-bounded at the request geometry. The largest
adopted q8 `cross` residual is 0.4438 GiB; the 10% MLX estimate margin over the observed 33–40 GiB
phase envelope contributes roughly 3.3–4.0 GiB on top of that bound. The residual is therefore not
being spent as margin, and the margin is not being used to conceal a missing phase. Candle has no
promoted temporal curve yet, so this verdict changes no Candle constant or evidence claim.

**Currency at the final provider-contract pin:** the SC-18808 q8 curve is historical by design.
SC-19109 changes the loaded provider contract/carrier fingerprint and the final inference feature
closure changes the provider digest, so runtime must reject the old fingerprint/`87a27d…` closure
instead of making that artifact reachable by alias. SC-18946 owns the frozen-closure q8 reseed/refit
and the first q4, bf16, and rung-2 captures. The schema-v2 producer groups those new records by their
complete identity and can promote them without code changes; until those physical captures exist,
the final atomic inference pin will route q8, q4, and bf16 through the provider-owned conservative
geometry/temporal fallback. The historical capture pin `b965641e` had neither the loaded LTX
contract nor that profile API. This branch's frozen preparation pin `b4a29108` contains both and
exercises the fallback, but it is not the permanent inference-`main` pin and carries no replacement
physical captures yet; this checkpoint therefore must not be reported as production-complete.

**Multi-curve promotion is selector-complete, not tier-pooled.** The fit report's canonical
`selectorFits` partitions records by model, catalog family, provider, backend, tier, mode, rung,
load shape, batch, inference closure, calibration ABI/fingerprint, and decode pass before fitting
any coefficient. Its legacy `fits.<phase>.<tier>` view is emitted only when that tier maps to one
complete selector; a campaign containing (for example) q8 staged and q8 bounded-decode records
omits the ambiguous q8 legacy slice instead of pooling them. `video-memory-curves.json` consumes the
matching selector fit and exact sorted record IDs. Its `sourceCatalog` hashes every immutable source
file's exact bytes, and each curve separately names the path, digest, and record subset it consumed;
the Rust loader recomputes those source handshakes and rejects missing, extra, cross-curve-reused, or
selector-mismatched records before any curve can evaluate.

**2. The phase coefficients are the right order of magnitude for the staged components, and the
per-voxel one matches the engine's own fit to 0.3%.** 🔴 Every coefficient below is fitted on **four
q8 geometries (seven records) of the six declared `fit` rows** — one was attempted and killed, one
was never begun; see *Coverage is derived* below — and scored on six held-out records. The
"× noise" figures are against the cross-session, cross-revision floor described above.

| phase | best form | coefficients (q8) | held-out max residual |
| --- | --- | --- | --- |
| text | `fixedGb + perMpxGb·mpx` (no temporal term) | 32.92 + 0.68·mpx | 0.30 GiB (2.8× noise) |
| denoise | `fixedGb + perLatentTokenGb·T_lat·(W/32)·(H/32)` | 20.52 + 0.000986/token | **0.13 GiB** |
| decode | `fixedGb + perMpxGb·mpx + perMpxFrameGb·(mpx·frames)` | 2.52 + 0.12·mpx + 0.2998·mpx·frames | **0.019 GiB** |

🔴 **The intercepts are NOT identities, and an earlier revision of this section said they were by
comparing GiB against GB.** The coefficient column is GiB; `stagedTextEncoderBytes` and
`stagedTransformerBytes` are decimal bytes. In **one unit (GB)**:

- denoise intercept **22.03 GB** against a **20.61 GB** transformer — **1.42 GB unexplained (6.9%)**
- text intercept **35.35 GB** against a **32.73 GB** staged text encoder — **2.62 GB unexplained (8.0%)**

The GiB→GB factor is **7.37%**, so the apparent agreement WAS that factor. The residue is plausibly
other resident state (the connector, small components, allocator overhead), but nothing here
establishes that — and the denoise gap is ~10× that fit's own 0.13 GiB held-out residual, so it is
not measurement slack either. Treat the intercepts as fitted parameters near the component sizes, not
as the component sizes.

The **per-voxel** result is unit-clean and stronger than first claimed. `perMpxFrameGb` 0.2998 GiB per
`mpx·frames` is **322 B per output voxel**. The engine's own single-pass decode cost is
`LTX_VAE_ACCUM_BYTES_PER_VOXEL + LTX_VAE_TILE_BYTES_PER_OUT_VOXEL` (a single pass is one tile, so
both terms apply), and those two are documented as fitted at **~36 + ~287 = 323 B/voxel**, then
rounded *up* to 40 + 300 = 340 for headroom (`mlx-gen-ltx/src/pipeline.rs:218-228`). Measured 322
against the engine's fitted **323** is **0.3%** — an independent reproduction of that fit, not of the
rounded constant. `perLatentTokenGb` 0.000986 is **1.009 MiB/token**, which reproduces the withdrawn
`0.997–1.032 MiB/token` above — as a **denoise-phase** relation, not an overall-peak one.

🔴 **The additive `perFrameGb` form should not be adopted, but "15–350× worse" over-sells it and is
withdrawn.** Held-out max residuals, additive against the `cross` form actually adopted by sc-18812:
decode **6.5722 vs 0.0185** (355×), denoise **2.6848 vs 0.1741** (15.4×) — and text **0.3445 vs
0.4438**, where additive is **1.29× BETTER**. The range holds on the two phases that carry the
temporal response, not on all three. The adoption still stands: text is nearly flat in frames (all
five candidates land within 0.164 GiB of each other there, 0.3015–0.4650, against a 22.6 GiB spread
on decode), `cross` strictly contains `area_only` and `output_voxels` so neither can be refitted to
beat it, and the 0.0993 GiB `cross` concedes on text is *below* that phase's own 0.1095 GiB
replicate floor — the largest of the three floors by orders of magnitude (denoise 2.7e-4, decode 0).

**3. 🔴 fps is a real but negligible memory axis — not an unmeasurable one.** Measured at identical
`{768x512, 241 frames}`, fps 30 vs 24 (a 25% difference in audio-latent length): conditioning
**identical to the byte**, decode identical to within 400 B, denoise **+26.0 MB (+0.075%)**. The
argument for ignoring it is MAGNITUDE, not resolution: 26.0 MB is **90× this dataset's own denoise
replicate floor** (2.7e-4 GiB) and is comfortably resolvable — it is under the *text*-phase floor
only. It is 0.075% of the denoise phase and 0.07% of the run's 33.23 GiB active peak, and at this
geometry the admission quantity (`max` over phases, here the text phase) is **identical to the byte**
across the two fps values. So the joint audio denoise does cost memory, and the cost is far below any
headroom a budget would carry. `GeometryEnvelope`'s missing fps axis is not a correctness gap for
memory — but sc-18812 should record it as *small*, not as *unmeasured*.

**4. 🔴 `recommendedMaxWorkingSetSize` does not bound active + cache.** Measured directly here:
`recommendedMaxWorkingSetSize` = 115,448,725,504 B = **107.52 GiB**, `maxBufferLength` = 80.64 GiB,
`hw.memsize` = 128 GiB, MLX's own `get_memory_limit()` = 130,567,005,798 B = **121.60 GiB**
(0.95 × `hw.memsize`). q8 at 640x640 x 177 reached **24.30 GiB peak active + 90.68 GiB end-of-phase
allocator cache = 114.98 GiB**, i.e. 7.46 GiB *above* the 107.52 GiB ceiling, on a render that
completed and was bit-identical on warm repeat.

> **Provenance of the 114.98 GiB.** It is the **session-1** record (`f301d712`, 08:31:49Z). Its
> session-2 replicate (`817cf550`) reads **115.47 GiB** — 0.49 GiB higher, the largest replicate
> spread in this series and, like every replicate here, cross-revision. Either record clears the
> 107.52 GiB ceiling by more than 7 GiB, so the finding does not turn on which one is quoted; the
> lower one is quoted deliberately, because the claim is that even the *smaller* reading exceeds the
> ceiling.

⚠️ That 114.98 GiB is an **upper bound on co-existence, not a simultaneous maximum**.
`PhaseMemory::capture()` pairs a phase-**window** peak (`get_peak_memory`) with an **instantaneous**
end-of-phase cache reading (`get_cache_memory`); MLX exposes no "cache at the active peak". Cache
enters the decode at ~0 and grows monotonically to 90.68 GiB, so the two readings are furthest apart
exactly where the number is largest, and the 7.46 GiB excess is 6.5% of the reading — inside that
uncertainty. The **qualitative** finding is not in doubt (it is corroborated by the driver log's
free-disk trace, 81 → 16 GiB across two large decodes): MLX's cache is elastic, grows past the
recommended working set, and is released under pressure. Log it — but a feasibility ceiling is not
made of active+cache, and the cache series is unfittable (held-out error ≥20 GiB for every form).
Tightening 114.98 GiB into a true simultaneous reading would need cache sampled at the active peak —
an MLX-side hook this apparatus does not have — and a re-render, so the bound is stated rather than
closed.

**5. 🔴 Peak memory is NON-MONOTONIC in frames, and the worst geometry is the write cap. PREDICTED —
no f297 or f305 record was captured.** Rung 2 engages on TWO independent bounds:

- the **write bound**, `VaeTiling::LTX.writable_frame_cap(h,w) = i32::MAX/(8·h·w)`
  (`gen-core/src/tiling.rs:167`, `full_res_channels: 8`) = 682 / 682 / 655 / **297** / **297** over
  the five declared resolutions. It does not move with the host.
- the **memory bound**, single-pass `3.3 GB + 340 B/voxel` against `get_memory_limit() × 0.85`. It
  does, and it binds **earlier** on a smaller machine.

🔴 So the claim that holds everywhere is **ONE-SIDED: no host can exceed 297 single-pass output frames
at 0.90 MP, and smaller hosts tile earlier via the memory bound.** "The 0.90 MP buckets tile from 298
frames on every host" is FALSE and this repository's own CI falsifies it — the hosted `macos-26`
runner tiles `768x512 x 97`, **585 frames below that bucket's 682 cap**, purely for memory.

There are in fact **three** outcomes at a given geometry, not two, and the third is only visible on a
small host: below the full-output **accumulator floor** (`3.3 GB + 40 B/voxel`) no tiling helps —
the accumulators hold the assembled video — so `budgeted_plan` **refuses before any render**. That
runner reports `~13 GB just for the output buffers, over this machine's ~6 GB safe budget` at
1280x704 f297. All three outcomes are now pinned at fixed budgets in
`the_ltx_arm_follows_the_engine_across_the_decode_tiling_boundary`, and the live host's own outcome
is asserted as a total function of its budget rather than assumed.

Predicted from the committed cost model: single-pass decode climbs to **94.3 GB** at 1280x704 f297
(`3.3 GB + 340 B/voxel × 267,632,640`). One lattice step above, at f305, the decode must tile, and its
cost is `3.3 GB + 40 B/voxel × 274,841,600 = 14.3 GB` of unavoidable full-output accumulators plus
`300 B` per tile-voxel. The selector keeps the **largest** tile that fits, so the tiled cost *rises*
with host memory: **≈15.0 GB** where only the smallest selectable tile (192 px × 64 frames) fits,
**18.5–21.8 GB** across the 384–512 px × 96-frame range, and **≈63.8 GB** on this 128 GiB host, which
affords 768 px × the full 305 frames under its 103.4 GiB ceiling. The collapse across the cap is real
on every host but much smaller on a large one (94.3 → 63.8 GB here). The most expensive geometry in
the declared envelope is still the cap, not the maximum, and a curve fitted across that boundary fits
through a capability change.

**6. 🔴 What actually bounds this host is FREE DISK, via swap.** Because the allocator cache grows
past physical memory, large single-pass decodes push the box into swap. From the committed driver log
(`docs/calibration/sc-18810/sweep-run.log`): 704x1280 x 177 (159,498,240 output voxels) completed;
768x512 x 449 (176,553,984 voxels) was **killed by a signal** after 1238 s with free disk falling
81 → 16 GiB; 1280x704 x 241 (217,169,920 voxels) was begun at 16 GiB free and left **no terminal line
at all** — the driver itself did not survive it; and 768x512 x 361 (141,950,976 voxels) was killed
later in the same session at lower free space. Across that one session free space went
**95 → 15 GiB against the driver's 25 GiB floor**, and the driver halted itself on that floor rather
than continuing. The safe ceiling therefore *moves with free disk*, and degraded as APFS local
snapshots pinned the churn. Check `df -h /System/Volumes/Data` before AND during, and treat any
single-pass decode above ~150M output voxels as needing tens of GiB of headroom. This is a HOST
verdict, not a model verdict.

**Candle adoption/measurement state (explicitly not a fitted-curve claim).** No Candle LTX or Wan
geometry sweep has been promoted into `video-memory-curves.json`; the only fitted curve in this
section remains historical MLX LTX q8. SceneWorks now has the synchronous post-load CUDA
`mem_get_info` snapshot and fixed cold-load provider attribution needed for a truthful live budget.
At the historical `b965641e` capture pin the provider-owned geometry/frames decode profile and the
loaded Candle generator contract were absent, so Candle failed open before selection. The frozen
preparation pin `b4a29108` includes the reviewed SC-19117 profile and SC-19223 contract: unmeasured
Candle routes now use the provider-owned decode working set plus each contract component exactly
once and the ordinary 4% estimate margin. This is schema-capable estimate fallback, **not** a fitted
curve, an optimization claim, or Candle calibration; the permanent inference-`main` pin must still
consume the same APIs. A fitted Candle curve remains blocked on a real GPU sweep with the same
fit/held-out, per-phase, closure-bound evidence discipline above.

**Coverage is derived, not asserted.** `scripts/fit-ltx-temporal-form.mjs` buckets every planned entry
from two artifacts — the dataset (was a record captured?) and the driver logs (was it ever begun, and
did it terminate?) — and **throws** when the plan and the logs disagree. It used to take "attempted
and killed" from a hardcoded two-element array, and that array was wrong: `1280x704 f241` was
published as `not_attempted_host_limit` while the log shows it was begun. Final counts, all read from
`coverage.byState` rather than typed here: **8 captured fixtures over 7 distinct geometries /
13 records**, 3 attempted-and-not-survived, 7 `not_attempted_host_limit`, **1 `stopped_before`**, and
25 never reached (**12 bf16, 11 q4, 2 q8** — the two q8 rows are the rung-2 boundary pair).

🔴 **One declared `fit` row was never captured, so the realized fit design is smaller than the plan
declares.** Of the six q8 rows pre-registered as `fit`, **four produced records**. `768x512 f361` was
attempted and killed. `1280x704 f177` was **never begun at all** — it is the geometry the driver
named on its `STOP` line when free disk fell through the floor, and it now has its own
`stopped_before` bucket rather than being buried among the unreached rows. An earlier revision of
this paragraph said the unreached rows were "all bf16 and q4"; three were q8, and one of those three
was this `fit` row, so the false parenthetical was also what concealed it. The row keeps its `fit`
role — re-labelling a pre-registered role after seeing which points survived is exactly the
after-the-fact redraw the role vocabulary exists to prevent — and the fit script's test
*"the REALIZED fit design is smaller than the declared one"* pins the 6-declared / 4-realized split
against the shipped bundle. **Every q8 coefficient in §2 is therefore fitted on four geometries
(seven records), not six**, and re-capturing either missing row is a change to the fit, not a
confirmation of it.

## 6d. What the emitted memory counters mean (schema v5, sc-18864)

A phase carries **three** numbers, and only two of them are measurements.

| field | MLX | CUDA | kind |
|---|---|---|---|
| `activeBytes` | `mlx_rs::memory::get_peak_memory()` | `nvidia-smi memory.used` delta | peak over the phase window |
| `reclaimableBytes` | `mlx_rs::memory::get_cache_memory()` | `0` (no caching allocator) | instantaneous at the phase boundary |
| `allocatorBytes` | `activeBytes + reclaimableBytes` | same | **derived**, and the validators enforce the identity |

`allocatorBytes` is an **upper bound on co-existence, not a simultaneous maximum**: it adds a
peak-over-window to an instantaneous-at-boundary reading, and MLX exposes no "cache at the active
peak". During an LTX decode the cache is ~0 on entry and grows monotonically to its end-of-phase
value, so the bound is loosest exactly where it is largest — up to **142.6 GB on a 137.4 GB host**
for a q8 render that completed and was bit-identical on warm repeat.

**Only `activeBytes` may be compared against a hardware or wired ceiling.** The allocator cache is
elastic; MLX releases it under pressure, which is why a completing render co-existed **7.46 GiB
above** Metal's `recommendedMaxWorkingSetSize` (§6c). Both `validate_complete` and
`validate_runtime_complete` — and their JS mirrors — now run that check against `activeBytes`.

Schema v4 also carried `deviceBytes` and `wiredBytes`. **Both adapters set them to verbatim copies
of `allocatorBytes`**, provably so across all 456 committed phase objects that carried them (404
across the five bundles plus 52 in the thirteen immutable v4 receipts), and MLX exposes no third
counter they could have carried. Schema v5 removes them rather than inventing readings for them, so
a record can no longer *represent* `wiredBytes > hardware.wiredLimitBytes` — the shape every
committed MLX record used to carry, and the reason none could be promoted past `gated`.

### What happened to the records captured under v4

**Migrated in place, not re-captured and not tombstoned.** No capture was ever physically
impossible: across every committed MLX record, `activeBytes` is under both `memoryBytes` and
`wiredLimitBytes` — the inversion lived entirely in the aliased names. Dropping the two aliases is
lossless, so the corrected record is exactly what the fixed adapter would have emitted for the same
measurement, and re-running renders would have destroyed information (fresh weights, fresh host
state) to recover numbers already retained. `docs/generated/memory-matrix.json` re-generates
**bitwise identical outside its provenance block**, which is the check that the migration moved no
published cell.

The immutable provider-stdout receipts under `docs/calibration/` still carry the v4 shape and
**must not be rewritten** — they are byte-addressed provenance. `recordFromPhysicalMlxResponse`
projects v4 → v5 during reconstruction, and **refuses** a receipt whose aliases are not copies of
`allocatorBytes`, because that would mean the adapter measured something the names claimed.

## 7. Ingest, stamping, and a new lane

A capture has **two halves**, and both must land or the lane does not move. §7a-7b update the
**evidence corpus**; §7d updates the **shipped calibration bindings** in
`config/manifests/builtin.models.jsonc`. Doing only the first is the single most likely way to finish
this runbook and change nothing the runtime reads — §7d quantifies exactly what that costs.

### 7a. Merge into the committed bundle

> Provenance: written from `memory-calibration-harness.mjs:1197-1201`, not re-executed while writing
> this runbook — `ingest` needs a real captured bundle as input.

```bash
node scripts/memory-calibration-harness.mjs ingest \
  --input /abs/path/OUTSIDE/the/repo/<lane>-evidence.json \
  --resume docs/generated/memory-calibration-evidence.json \
  --output /abs/path/OUTSIDE/the/repo/merged.json
```

Merge is commutative and stable-sorted; **different content under the same resolved identity is
rejected** rather than resolved by arrival order. Copy `merged.json` over
`docs/generated/memory-calibration-evidence.json` once it validates.

Record ids are content-derived and re-validated on read, so a capture at a new revision necessarily
produces **new record ids**; a bundle whose ids do not re-derive fails with
`<id>: deterministic identity mismatch` (`memory-calibration-harness.mjs:503`). Never hand-edit a
record field and keep its id.

### 7b. Confirm the closure digest is on every record

The runner stamps it at capture time from the inference checkout it was given
(memory-calibration-harness.mjs:927-957), one digest **per lane**, keyed on each plan entry's own
`<backend>:<target.provider>`. Verify rather than assume — `evidenceSemantics` fails **loudly** if it
is absent (memory-calibration-harness.mjs:656-662):

```
<id>: no repositories.inference.closureDigest. Every complete record must carry the provider
closure digest it was captured under (sc-17774); re-run the backfill in
scripts/backfill-closure-digests.mjs against an inference clone.
```

The remedy is the backfill, which needs a clone containing every **captured** revision (not just the
pin):

```bash
node scripts/backfill-closure-digests.mjs --revisions            # which revisions must be present
node scripts/backfill-closure-digests.mjs --repo <inference> --write
```

> Provenance: `--revisions` and `--verify` were executed against a real inference clone; `--write`
> was **not** re-executed while writing this runbook — it mutates checked-in state, and `--verify`
> exercises the identical derivation.

A dry run (no `--write`, no `--verify`) prints what it would change and exits 0 — note its own
closing line, `(dry run — pass --write to update the bundle and manifest)`: `--write` updates
**both** `docs/generated/memory-calibration-evidence.json` and `config/manifests/builtin.models.jsonc`
(`backfill-closure-digests.mjs:541-547`). That is the mechanism §7d relies on. `--verify` is the CI
spelling and fails on any drift. Verified locally against `/Users/michael/Repos/inference`:

```
records: 0 stamped, 65 already correct, 65 total
manifest bindings: 0 stamped
captured closure digests match their revisions (65 records + every one of the manifest's digests,
none unreachable)
```

A **conflict** — a recorded digest that disagrees with the derivation at its own revision — is
reported and refused, not overwritten, because overwriting would launder a stale measurement into a
current one. `--restamp` exists only for a `CLOSURE_DIGEST_VERSION` bump and is not a way past a
genuine conflict.

### 7c. If the lane is NEW

> Provenance: `--check` was executed against a real inference clone; `--write` was **not**
> re-executed while writing this runbook — it rewrites checked-in state, and `--check` runs the
> identical derivation.

Every authoritative lane in the plan must have an entry in
`config/inference-provider-closures.json`, and `memory-calibration-harness.test.mjs` asserts it.

⚠ **Exactly one field is hand-added; everything else is derived.** "Regenerate, never by hand" is the
rule for *digests*, and it is easy to over-read into "run `--write` and the new lane appears" — it
will not. The script builds its work list from the checked-in config's own `crate` fields —
`declared = Object.entries(existing.providers).map(([p, e]) => [p, e.crate])`
(`inference-closure-digest.mjs:665-668`) — so a lane that is not already in the file is simply not
digested, and `--write` rewrites the same set it started with. Seed the stub first, then derive:

```bash
# 1. Hand-add ONLY the key and its crate directory to config/inference-provider-closures.json:
#      "<backend>:<provider>": { "crate": "crates/media/<backend>-gen/<engine-crate>" }
#    Every other field (digest, closureCrates, sourceFileCount, lockedPackageCount) is derived —
#    do not invent them, and do not copy a sibling lane's.
#
# 2. Derive the whole table from a real clone.
node scripts/inference-closure-digest.mjs --repo <inference clone> --write

# 3. Prove it took.
node scripts/inference-closure-digest.mjs --repo <inference clone> --check   # verified: prints
                                                # "config/inference-provider-closures.json matches 40fa7583"
```

`--write` and `--check` both digest at the pin parsed from the workspace `Cargo.toml`, not at the
adapter constant (§5).

The key is `<backend>:<provider>`; a bare provider id is ambiguous (`krea_2_turbo_control` exists on
both backends with different crates). `buildClosureConfig` also asserts the provider is really
declared in the crate you named, so a wrong crate fails rather than digesting the wrong tree.

### 7d. 🔴 Move the lane's SHIPPED bindings — the half that actually changes the runtime

**Do not skip this. Without it your capture promotes nothing.**

§1 called `BINDINGS` the shipped admission surface — the calibration objects under each model's
`mlx`/`candle` block in `config/manifests/builtin.models.jsonc`, which are what the worker's fit
gates read. They carry their **own** `inferenceRevision` / `inferenceClosureDigest` pair, independent
of the evidence corpus. Ingesting evidence does not touch them.

Point them at the revision you captured at, then let the backfill derive the digest:

```bash
# 1. In config/manifests/builtin.models.jsonc, for every binding whose "provider" is <provider>
#    under the <backend> block, set "inferenceRevision" to the revision you captured at.
#    Leave "inferenceClosureDigest" alone — step 2 derives it. Do NOT hand-write a digest.
#
#    Find them:
grep -n '"provider": "<provider>"' config/manifests/builtin.models.jsonc

# 2. Re-derive and stamp both halves from a real clone (§7b): this rewrites the bundle AND the
#    manifest, using the CI gate's own brace-depth-accurate, orphan-checked locator.
node scripts/backfill-closure-digests.mjs --repo <inference clone> --write

# 3. Prove it took.
node scripts/backfill-closure-digests.mjs --repo <inference clone> --verify
```

A hand-written digest is not an option: `--verify` re-derives every manifest digest at its own
revision and exits 1 on any disagreement, and that check is a required CI step (`check.yml:105-123`).

**Measured proof that this step is load-bearing.** Both paths were simulated on `origin/main` for
`mlx:z_image_turbo` — restamp the 5 evidence records to the live pin and digest, regenerate, and
compare:

| | evidence half only (§7a-7c) | both halves (§7a-7d) |
| --- | --- | --- |
| `report:stale-lanes` totals | `8 stale, 0 current` — **unchanged** | `7 stale, 1 current` |
| the lane on the stale list | still ranked **#2**, `5/5` bindings `0/5` records, `status=partially-stale` | **gone** — listed under `CURRENT (no widening applied)` |
| shipped stale bindings | 33 | 28 |
| `summary.currentCalibrationRuns` | 5 | 5 |
| `z_image_turbo` mlx cell states | `Implemented/unverified: 90`, **`Verified: 0`** | `Implemented/unverified: 85`, **`Verified: 5`** |

The evidence-only path produces `current` records that promote **nothing**. The report does not clear
the lane; it reports `status=partially-stale` with `captured=066ff9c6a26e,5b9092c67e0f` — the two
digests being the record half you moved and the binding half you did not. §10 step 1's success
criterion is unreachable without §7d.

## 8. Regenerate the derived docs

```bash
npm run generate:memory-matrix
npm run check:rust-derived-docs           # memory-matrix + tier integrity, both --check
npm run check:memory-calibration          # schema-checks the committed bundle
```

Verified green on `origin/main` today. `check:rust-derived-docs` ends with
`tier integrity OK: 111 declared exceptions (111 measured, 0 unmeasured).`

**Nothing in §8 is downstream of the adapter sources, so touching an adapter needs no regeneration
here.** Measured in sc-18104: a comment-and-`match` edit to `mlx.rs` left
`docs/generated/memory-matrix.json` byte-identical. Regenerate because you ingested evidence or moved
bindings, not because you touched Rust — what an adapter edit owes instead is `cargo fmt --all` and a
clippy-clean build, since the macOS lane lints that binary. (This was *not* true before sc-18100: the
cost model it deleted did fingerprint `.../bin/{mlx,candle}.rs`, so a Rust-only edit used to red a
generated-doc check. If you are on an older branch and see that failure, this is why.)

**Coordinate before regenerating.** These artifacts are regenerated by several stories at once and
are a classic merge-queue conflict. If a sibling PR is already regenerating them, rebase onto it
rather than racing.

## 9. ⚠ The trap: landing a genuinely CURRENT lane REDS pinned doc-fact tests

**Read this before you ingest.** Today the corpus is uniformly historical — verified:

```
mlx:qwen_image historical 41   mlx:krea_2_turbo_control historical 4
candle:flux1_dev historical 5  candle:flux1_schnell historical 5
candle:flux2_dev historical 5  mlx:z_image_turbo historical 5
```

`matrix.summary.currentCalibrationRuns` is `0`. Several tests **pin that as an exact set**, not as an
upper bound, so a capture that lands a `current` lane turns them red. This is the tests being right
about yesterday, not your measurement being wrong — but it means **a measurement PR is never
docs-only**, and you must plan to update the affected ones in the same commit.

**Which tests red is lane-dependent and step-dependent.** The table below is the measured result of
simulating an `mlx:z_image_turbo` capture on `origin/main`, both ways (§7d):

| test | assertion | evidence half only | + §7d bindings |
| --- | --- | --- | --- |
| `scripts/generate-memory-matrix.test.mjs:717` | `"no historical Z-Image capture may be promoted across an exact inference-pin change"` | 🔴 reds | (superseded by :670 below) |
| `scripts/generate-memory-matrix.test.mjs:670` | `"historical bindings must not mask the pinned MLX provider contract"` | green | 🔴 reds |
| `scripts/generate-memory-matrix.test.mjs:2506` | `"current evidence cannot promote through a historical exact manifest binding"` | **green** — it cannot red until a binding moves | 🔴 reds |
| `tests/test_memory_matrix.py:189-190` | `{… for run in full_runs} == {"historical"}` and current count `== 0` | 🔴 reds | 🔴 reds |
| `tests/test_memory_matrix.py:531` | `test_historical_records_remain_unverified_after_the_z_image_pin_advance` | 🔴 reds | 🔴 reds |
| `scripts/generate-memory-matrix.test.mjs` (`"the published summary re-derives from the evidence bundle and the closure ledger"`) | `summary.calibrationRunsByStatus` and `summary.currentCalibrationRuns`, recomputed from the bundle and `config/inference-provider-closures.json` | 🔴 reds on the status tally the moment a record lands | 🔴 reds until §8 regenerates |

sc-18100 deleted the flux1/flux2 evidence one-shots, whose lane-specific `semantics === "historical"`
pins used to appear in this table. The row above replaces them and is lane-agnostic: it fails
whenever the committed matrix summary disagrees with the bundle and closure ledger it is derived
from, so it is the general form of "you have not run §8 yet".

Two lessons in that table. First, the `"current evidence cannot promote through a historical exact
manifest binding"` assertion in `scripts/generate-memory-matrix.test.mjs` is the reviewer's canary
for §7d (cited by name, not line number — the row has already been renumbered once): if you finish a
capture and it is still green, **you skipped the binding half**. Second, the set is per-lane — enumerate
yours by running the suites rather than trusting this list:

```bash
node --test scripts/generate-memory-matrix.test.mjs \
             scripts/memory-calibration-harness.test.mjs
```

🔴 **The Python one runs ONLY in the required `parity` CI lane. A local green does NOT clear it.**
`npm run check` never invokes pytest; `check.yml` line 81 runs
`python -m pytest -q tests/ -m "not e2e and not parity" --strict-markers` inside the `parity` job,
and `tests/test_memory_matrix.py` carries no marker, so that is where it executes. Run it yourself
before pushing:

```bash
# once — put the venv OUTSIDE the repo so it cannot dirty a capture (§5)
python3.12 -m venv /abs/path/OUTSIDE/the/repo/venv
/abs/path/OUTSIDE/the/repo/venv/bin/python -m pip install \
  pytest==9.0.2 httpx==0.28.1 jsonschema==4.25.1

/abs/path/OUTSIDE/the/repo/venv/bin/python -m pytest -q tests/test_memory_matrix.py
```

Verified on `origin/main` today: `9 passed in 19.16s`. Note the bare `python3.12` on this host has no
`pytest` — the venv is not optional.

How to handle a red: **relax the pin to the new truth, do not delete the assertion.** The comment
above `tests/test_memory_matrix.py:189` explains why it is pinned as an exact set and count — "a bare
`<= {"current", "historical"}` would accept any mixture, and a count alone would let one family's
promotion mask another's demotion". Preserve that property: assert the exact new set and the exact
new count for your lane, and leave every other family's pin untouched. Say in the PR body which pins
moved and why.

## 10. Verification

Four checks. Do all four; the first three are cheap.

**1. The new record reads as `current` for its lane.**

```bash
node -e 'const m=require("./docs/generated/memory-matrix.json");
  const c={}; for (const r of m.calibrationRuns) {
    const k=`${r.record.backend}:${r.record.target.provider} ${r.semantics}`; c[k]=(c[k]||0)+1; }
  console.log(c); console.log("currentCalibrationRuns:", m.summary.currentCalibrationRuns)'
```

Your lane must now appear as `current`, and `currentCalibrationRuns` must have risen by the number of
records you landed. But **that alone is not success** — it is true on the evidence-only path too
(§7d). All three of these must hold:

```bash
npm run report:stale-lanes    # your lane must have LEFT the stale list, and the header's
                              # "N stale, M current" must show one lane moved across
```

```bash
node -e 'const m=require("./docs/generated/memory-matrix.json");
  const z=m.cells.filter(c=>c.modelId==="<modelId>"&&c.backend==="<backend>");
  const s={}; for(const c of z) s[c.state]=(s[c.state]||0)+1; console.log(s)'
# the measured rungs must now be Verified, not Implemented/unverified
```

Measured on the `mlx:z_image_turbo` simulation: the corrected procedure gives
`7 stale, 1 current`, the lane absent from the table, `28` (not 33) stale bindings, and
`{ 'Implemented/unverified': 85, Verified: 5 }`. The evidence-only path gives `8 stale, 0 current`,
the lane still ranked #2 as `5/5` bindings `0/5` records, and `Verified: 0`.

Failure modes:

- **Records stay `historical`** — the captured digest does not match the live one for the lane:
  usually the inference checkout was not at the pin (§5), or the closure table was regenerated after
  the capture.
- **Records are `current` but the lane is still listed, as `partially-stale`** — you did §7a-7c and
  skipped §7d. Go back and move the manifest bindings.
- **Records are `current`, the lane is clear, but no cell is `Verified`** — the binding moved but does
  not match the record on fingerprint, geometry, parameters, artifact revision or `engagedRungs`;
  `calibrationBinding` requires an exact ordered match. Compare the binding object against the record
  field by field.

**2. `verify_model_snapshot` binds the fixture to real weights** — §4. Re-run it after the capture,
with `--inventory-output`, and cite the `inventory_sha256` in the PR. Where the lane's artifact has no
`[[models]]` entry at the pin (§4), say so explicitly instead of skipping quietly.

**3. Schema and derived docs** — §8, all green.

**4. One real render through the newly measured cell.** The full API → queue → worker → backend path
runs natively; `npm run dev` being docker-compose does not mean otherwise.

```bash
cargo build -p sceneworks-rust-api -p sceneworks-rust-worker

SCENEWORKS_API_HOST=127.0.0.1 SCENEWORKS_API_PORT=8733 \
SCENEWORKS_DATA_DIR=<scratch>/data SCENEWORKS_CONFIG_DIR=<repo>/config \
SCENEWORKS_JOBS_DB_PATH=<scratch>/data/jobs.db \
HF_HOME=~/.cache/huggingface HF_HUB_CACHE=~/.cache/huggingface/hub \
  target/debug/sceneworks-rust-api

SCENEWORKS_API_URL=http://127.0.0.1:8733 SCENEWORKS_GPU_ID=mlx \
SCENEWORKS_DATA_DIR=<scratch>/data SCENEWORKS_CONFIG_DIR=<repo>/config \
HF_HOME=~/.cache/huggingface HF_HUB_CACHE=~/.cache/huggingface/hub \
  target/debug/sceneworks-rust-worker
```

Then `POST /api/v1/projects` → `POST /api/v1/image/jobs` with the measured model, tier, geometry and
mode → poll `GET /api/v1/jobs/:id`. Three traps that each cost half an hour:

- `SCENEWORKS_GPU_ID` **must be the literal `mlx`** on Apple silicon. Unset defaults to `cpu` and you
  get utility workers with no `image_generate` capability — the job sits `queued` forever with nothing
  logged. `auto` is NVIDIA-only and dead-ends the same way. Check `GET /api/v1/workers` first.
- Terminal job status is **`completed`**, not `succeeded`.
- Use a **scratch** `SCENEWORKS_DATA_DIR` with only the model's install marker copied in, and leave
  `HF_HUB_CACHE` on the real cache, so the real install is never mutated.

This path is documented from a prior verified session, not re-executed while writing this runbook —
treat the env-var contract as the load-bearing part and re-verify `GET /api/v1/workers` before
concluding anything from a stuck job.

## 11. Land it

One commit, referencing the story. Every one of these goes in it — a bundle without its moved
bindings promotes nothing (§7d), a moved binding without its regenerated matrix is a red CI, and a
regenerated matrix without the relaxed pins is another:

| file | why it is in this commit | step |
| --- | --- | --- |
| `docs/generated/memory-calibration-evidence.json` | the measurement itself | §7a-7b |
| `config/manifests/builtin.models.jsonc` | 🔴 the **shipped bindings** — without this the capture changes nothing the runtime reads | §7d |
| `config/inference-provider-closures.json` | only if the lane is new, or the pin moved | §7c |
| `docs/generated/memory-matrix.json` + `.md` | derived | §8 |
| `crates/sceneworks-memory-adapter/src/bin/<backend>.rs` | only if you had to add or fix an arm (§2c); run `cargo fmt --all`. Since sc-18100 deleted the cost model, an adapter edit no longer moves any generated doc — but the macOS lane clippies this file, so a warning here is a red CI | §2c |
| `scripts/generate-memory-matrix.test.mjs`, `tests/test_memory_matrix.py`, and any other pin your lane reds | the corpus is no longer uniformly historical | §9 |

Before pushing:

```bash
git status                # a new file can be silently .gitignored — confirm it is actually staged
npm run check             # must exit 0
<venv>/bin/python -m pytest -q tests/test_memory_matrix.py   # §9 — npm run check does NOT cover this
```

Merge the latest `origin/main` (merge, not rebase) before final verification, then open the PR. In the
PR body, state:

- which lane was measured, on which host, at which inference SHA and artifact revision;
- the **measured** wall clock and disk, so the next operator has a real number instead of the 300 s
  assumption;
- which test pins moved and why;
- the `verify_model_snapshot` `inventory_sha256`, or that the lane has no manifest binding.

This subsystem is mutation-checked throughout — the harness refuses dirty repositories, refuses
identity collisions, refuses laundered digests, and the generators refuse hand-edits. Expect a review
that asks what would have failed if your change were wrong, and answer it in the PR body.

## 12. The candle VIDEO lane — `candle:wan2_2_ti2v_5b` (sc-19057)

> Provenance: the arm, its guards, its plan and this section were written and unit-tested off-GPU.
> **The capture itself has NOT been run.** Per epic 18093's posture, measurement is a runbook rather
> than a per-model story: this section is the runbook, and the capture is a terminal-phase dispatch
> the epic coordinator performs once every code story has merged. A capture taken mid-epic
> fingerprints source paths later stories change, and the harness aborts if the repository moves
> under it (§5), so running it early wastes the lane rather than saving time.

### 12a. What this lane is, and why it is Wan and not LTX

`candle-gen-wan` is the **only** candle video crate that registers a `MemoryStrategyContract` at the
pinned inference revision (`memory_strategy::MEMORY_REGISTRATION`, SC-19223): three Implemented rungs
(`resident`, `staged_residency`, `bounded_decode`), a `PhaseEnvelope` formula over `FrameCount`,
per-load-policy calibration identities, and a `MemoryMode::Other("text_to_video")` route gate.

`candle-gen-ltx` registers none — no `memory_strategy.rs`, no `register_memory_strategy` call, and
`config/rung4-applicability-survey.json` already records that (*"there is no `MemoryStrategyContract`
impl in the crate at all"*). An LTX capture would therefore have no admission seam to interrogate and
no calibration identity to bind, and would first need the provider contract built in `inference` — a
two-repo pin-lockstep change. Verify that this still holds before assuming it:

```bash
git -C <inference-at-pin> grep -l memory_strategy_contract -- crates/media/candle-gen
# expect: …/candle-gen-wan/src/lib.rs and …/candle-gen-wan/src/memory_strategy.rs, and NO -ltx entry
```

### 12b. The envelope this arm accepts, and what it refuses

The arm validates Wan's own geometry envelope through `protocol::validate_video_geometry` — the video
counterpart of the `validate_still_geometry` hoist sc-18808 did for the image arms. Every other
Candle arm keeps its `frames == 1` refusal untouched.

| axis | value | source |
| --- | --- | --- |
| resolutions | `832x480`, `1280x704`, `704x1280` | `limits.resolutions` |
| spatial lattice | multiple of 32 | core's default floor; `wan_2_2` declares no `requiresDimensionsMultipleOf` |
| area cap | 901 120 px | `limits.maxPixels`, enforced by the engine as `MAX_AREA_5B` |
| temporal lattice | `frames = 1 + 4k` | the z48 VAE's `VAE_STRIDE_TEMPORAL` |
| frame envelope | `[93, 189]` | derived over `durations [4,5,6,7,8]` at the **capturable** cadence (24 fps) through `wan_frame_count` |
| batch | 1 | the provider advertises `max_count 1` |

🔴 **The frame envelope deliberately excludes the 16 fps rungs.** `61`, `77`, `109` and `125` are
real product geometries — the 4s–8s clips at 16 fps — but 16 fps is the cadence the calibrated route
refuses (next bullet), so no admissible plan row can produce them. Deriving the envelope over the
full `durations x fps` cross product would have admitted `832x480 f61 fps24`, a geometry no product
request can ask for, which is the same drift the fps refusal exists to prevent. The two rules now
agree. It is still a closed *span*, not a set — `protocol::VideoGeometryEnvelope` carries an interval
— so it also admits on-lattice counts between the rungs (`97`, `121`, …); the exact ladder membership
of every committed plan row is bound separately, by `the Candle Wan arm's manifest constants match
the shipped wan_2_2 limits` in `scripts/platform-review-contracts.test.mjs`. Write rows on the ladder:
`93, 117, 141, 165, 189`.

Three refusals are worth knowing before you write a row, because each one is a fact about the
**calibrated route** rather than about the product:

- **fps must be 24.** `limits.fps` declares `[16, 24]`, but `WanMemoryRequestScope::validate_request`
  rejects any cadence other than `DEFAULT_FPS`. The 16 fps column is shipped and not capturable; a
  16 fps fixture is refused by name rather than dying inside the provider.
- **q4 and q8 only.** `bf16` is the upstream dense `Wan-AI/Wan2.2-TI2V-5B-Diffusers` snapshot, which
  has no per-tier subdirectory, so `validate_huggingface_snapshot_root` has nothing to bind an
  artifact identity to and a bf16 record could not name the bytes it measured.
- **`bounded_attention` / `bounded_transformer_residency` are Missing** in the pinned contract. A row
  naming one is refused before the model load, not after it.

`planned.target` carries BOTH ids and both are validated: `provider` is the engine id
`wan2_2_ti2v_5b`, `modelId` is the builtin-catalog id `wan_2_2`. They genuinely differ on this lane
(unlike `mlx:ltx_2_3`, where they coincide), and `scripts/fit-ltx-temporal-form.mjs` resolves
`modelFamily` from the **catalog** id — a record carrying the engine id there makes the fit throw
`model wan2_2_ti2v_5b is absent from builtin.models.jsonc` hours after the capture burned.

### 12c. Preconditions

1. **Weights.** `SceneWorks/wan2.2-ti2v-5b-candle` at the manifest revision, tier subdirectory on
   disk. Confirm with the §4a tier listing; the manifest row is the authority for the revision.
2. **Closure entry.** `candle:wan2_2_ti2v_5b` is already declared in
   `config/inference-provider-closures.json` (sc-19057). `runProviderPlan` derives its digest eagerly
   and fails closed with `lane "candle:wan2_2_ti2v_5b" has no entry` **before** the capture burns, so
   confirm it survived any pin bump:

   ```bash
   node scripts/inference-closure-digest.mjs --repo <inference-at-pin> --check
   ```
3. **Clean checkouts at exact SHAs, in BOTH repositories** — §5. The harness re-probes after every
   provider case and aborts on `repository HEAD or dirty state changed during provider execution`.
4. **An idle GPU.** `VramProbe::start_rendered().assert_idle(1.0)` refuses to start against a card
   already holding more than 1 GB. On this host the two RTX PRO 6000s are **also** the four
   self-hosted CI runners, so take the runner tokens out of rotation (or dispatch through §12e) — a
   co-tenant CI job both poisons the measurement and is degraded by it.

Adapter environment — three variables, the same derivation rule as every other arm (§4):

```bash
# memory-candle-adapter — wan2_2_ti2v_5b   (sc-19057; the only candle VIDEO arm)
SCENEWORKS_WAN_REPOSITORY=SceneWorks/wan2.2-ti2v-5b-candle   # fixed; validated against WAN_CANDLE_REPOSITORY
SCENEWORKS_WAN_REVISION=<exact 40-hex artifact revision>
SCENEWORKS_WAN_ROOT=/abs/path/.../snapshots/<rev>/<tier>     # q4 | q8, derived from the plan target
```

### 12d. The capture

Build the adapter in the supported CUDA + host-compiler environment (MSVC 14.44 vcvars on this host),
then run the harness against the **standalone** plan. The plan is deliberately not in
`config/memory-calibration-plan.json`, for the reason sc-18808 established empirically: the matrix
generator resolves its universe from `manifest.models.filter(m => m.type === "image")`, so a video
row matches no coordinate and the check gate fails closed.

```bash
cargo build --release --locked -p sceneworks-memory-adapter \
  --features candle --bin memory-candle-adapter

node scripts/memory-calibration-harness.mjs run \
  --config docs/calibration/sc-19057/wan-candle-video-capture-plan.json \
  --backend candle \
  --fresh-per-case \
  --provider-command '["/abs/path/to/target/release/memory-candle-adapter.exe"]' \
  --sceneworks-repo /abs/path/to/SceneWorks \
  --inference-repo /abs/path/to/inference \
  --output /abs/path/OUTSIDE/the/repo/wan-candle-video-sc-19057.json
```

🔴 **There is no `--raw-log-dir` on this lane, and adding it aborts the sweep after the first cold
model load.** The raw-log provenance arm is MLX-only by construction, in two places that agree:
`memory-calibration-harness.mjs` fails any case where `--raw-log-dir` is configured and the provider
returned no `sourceCapture` (*"configured raw-log provenance requires provider sourceCapture"*), and
the one it does accept must be `sourceCapture.kind === "physical_mlx"`. Only
`crates/sceneworks-memory-adapter/src/bin/mlx.rs` ever emits that key — for the Qwen physical-source
arm — and `bin/candle.rs` emits none at all, so a candle case can never satisfy it. The flags are
also enforced as a PAIR (*"--raw-log-dir and --source-path-prefix must be supplied together"*), so
supply both or neither; supplying both still dies, and it dies *inside the per-case fragment loop*,
i.e. after the first row has already paid its ~10-minute cold load. Do not re-add them. Nothing is
lost by omitting them: `sourceProvenance` is a plan-declared field valid only for authoritative Qwen
MLX evidence, this plan declares none, and every other non-Qwen arm ships without it.

🔴 **`--output` and any `--resume` path MUST be outside the checkout.** The harness stamps every
record with each repository's `dirty` flag from `git status --porcelain`, **which counts untracked
paths**, so a bundle written inside the tree dirties the very repository the record is attesting to
and the whole sweep is discarded at the end (§5, §6). Resume an interrupted sweep by pointing
`--resume` at the partial output — the runner atomically checkpoints after every successful provider
response.

Six rows, ordered cheapest-first. The three-coefficient cross form needs at least three distinct
non-collinear geometry points plus at least one held-out row, so if the lane budget runs out the last
row (`704x1280 f189`, the heaviest) may be dropped and the remaining five still fit. Dropping more
than that makes the slice singular.

**q8 runs from a separate copy of the plan, never an in-place edit.** The committed plan's
`target.tier` must stay `q4` — `platform-review-contracts.test.mjs` pins every row's tier to
`candleDefaults[0].variant`, so a committed q8 swap reds that gate. An in-place *uncommitted* swap
is just as unusable: the harness stamps every record with each repository's dirty flag from `git
status --porcelain`, which counts untracked changes, so editing the plan inside this checkout
dirties SceneWorks and the whole q8 sweep is discarded at `check` (§5, and the `--output` note
above). Instead, copy `wan-candle-video-capture-plan.json` to a path OUTSIDE
the checkout, swap `q4` → `q8` in that copy's `target.tier`, `name`, and `fixture` tokens, and point
`--config` at the copy.

**Cost is unmeasured on this lane — measure yours and report it, do not promise a number.** What is
known: the shipped `candle.vramGbByTier` for this model was measured on an idle RTX PRO 6000 at
`832x480 x 121` frames, 20 steps, CFG on (sc-13175), and the peak is the denoise attention transient
rather than the decode. This arm renders at **2 steps** — the peak is per-step, so the ceiling is the
same, but the wall clock is not comparable to that measurement. Each row is a cold model load plus
two full clips (the measured render and its warm repeat).

### 12e. Through the guarded dispatch

There is no `run_memory_calibration`-style input for this lane on `windows-candle.yml` yet; the
existing `run_five_rung_reference` input drives `candle:krea_2_turbo` only (§6b). Until one is added,
run §12d directly on the CUDA host with the runner tokens paused. If you add the workflow input,
mirror the existing one's shape: validate the artifact identity, resolve the snapshot root without
printing it, check inference out at the exact `INFERENCE_PIN`, build the release adapter, run the
harness, `check` the bundle, and upload it.

### 12f. Verify the receipt before believing it

```bash
node scripts/memory-calibration-harness.mjs check \
  --input /abs/path/OUTSIDE/the/repo/wan-candle-video-sc-19057.json
```

Then read the receipt rather than assuming it. The arm is built so that each of these is a measured
claim, and each has a way to be wrong:

| field | what it must say | what it means if it does not |
| --- | --- | --- |
| `status` | `runtime_complete` for the six zero-axis rows | a `gated` row means an admission scenario did not execute or the sweep range was not verified — read `scenarios[].reason` |
| `scenarios.loadability` | `passed` | the arm only reaches this after a completed real load; `not_run` alongside any other `passed` scenario is refused structurally and cannot be emitted |
| `scenarios.exact_fit` / `unknown_budget` / `stale_evidence` | `passed`, with `predictedBytes == effectiveBudgetBytes` | these interrogate `memory_strategy_safety_check` on the live generator; a failure aborts the capture rather than downgrading the receipt |
| `scenarios.cancel` / `error` | `not_run`, carrying the named blocker | this arm executes no lifecycle injection; a `passed` here would be fabricated |
| `quality.maximumError` etc. | inside 3/255, 1/255, 1.5/255 | the warm repeat is not deterministic — do not ingest |
| `diagnostics.negativeMutation*Per255` | all three breaching the thresholds above | the thresholds are measuring nothing; the capture aborts rather than emitting this |
| `diagnostics.decodeTilingEngaged` | `0` on a `staged_residency` row | `1` means bounded decode engaged and the emitted curve id will carry `tiled`, not `single_pass` |
| `diagnostics.outputFps` | `24` | required by the fit; a mismatch against the request aborts the capture |
| `diagnostics.firstFrameNondegenerate` | `1` | measured, not asserted — a flat first frame is a decoder failure and the capture aborts, so a record can only ever carry the 1 |

### 12g. Where the numbers land

The record is committed **standalone**, the `docs/generated/ltx-mlx-video-sc-18808.json` precedent, and
wired into `npm run check` so it stays schema-valid:

1. Copy the bundle to `docs/generated/wan-candle-video-sc-19057.json`. There is **no** raw-receipt
   tree to copy — this lane runs without `--raw-log-dir` (see §12d), so `docs/calibration/sc-19057/`
   holds the capture plan and nothing else.
2. Add it to the `check:memory-calibration` chain in `package.json`, beside the LTX video record.
   Confirm the gate protects its **existence** by deleting the file and watching the check red.
3. Feed it to the fitter. `scripts/fit-ltx-temporal-form.mjs` is backend-neutral **in its plumbing**
   — it reads `record.backend`, `record.target.modelId` (→ `modelFamily` from the manifest),
   `record.repositories.inference.closureDigest`, and the `_role` labels from the committed plan —
   so it emits a `backend: "candle"` curve into `docs/generated/video-memory-curves.json` beside the
   existing MLX LTX one. That bundle currently contains exactly one curve and it is MLX; this is the
   producer gap this lane closes.

   The container is **multi-lane** as of schema v3 (epic 19048): fit provenance lives on the CURVE
   (`sourceFit`), not on the bundle, so the two lanes each name their own report. The fitter reads
   the bundle it is about to write and REPLACES only its own measurement lane, preserving every
   other one — so promoting candle does not delete the MLX curve, and a later MLX re-promotion does
   not delete candle. Two consequences:

   * Name the report from its own family: `--source-fit docs/generated/wan-temporal-form-fit-sc-19057.json`
     (with a matching `--write`). The pattern is
     `docs/generated/<family>-temporal-form-fit-sc-<story>.json`; a report outside it is refused by
     the producer, by `packages/schemas/video-memory-curves.schema.json`, and by the runtime reader.
   * One evidence file may not straddle two lanes, and a promotion must carry exactly one lane —
     both throw rather than merging ambiguously. Every curve in a lane must also name the SAME
     report; the reader rejects a lane that blends two campaigns.

   🔴 **Every default in this script is LTX-shaped — pass all four flags explicitly, or the first
   candle invocation dies on a missing file.** `--story sc-19057` alone resolves `--dataset` to
   `docs/generated/ltx-mlx-single-pass-sc-19057.json` and `--plan` to
   `docs/calibration/sc-19057/ltx-mlx-single-pass-sweep.json`, neither of which exists on this
   lane. The candle invocation is:

   ```bash
   node scripts/fit-ltx-temporal-form.mjs \
     --story sc-19057 \
     --dataset docs/generated/wan-candle-video-sc-19057.json \
     --plan docs/calibration/sc-19057/wan-candle-video-capture-plan.json \
     --write docs/generated/wan-temporal-form-fit-sc-19057.json \
     --source-fit docs/generated/wan-temporal-form-fit-sc-19057.json
   ```

   There is no `--driver-log` for this lane (the runbook's §12d invocation writes none), so the
   coverage block is derived from the plan and dataset alone.

   Then add `docs/generated/wan-candle-video-sc-19057.json` to
   `PACKAGED_VIDEO_MEMORY_CURVE_SOURCES` in `crates/sceneworks-core/src/video_memory_curves.rs`.
   That const is the compiled source catalog the loader validates the bundle against, and it must
   list **exactly** the sources the committed curves consume. Omitting it fails the whole bundle
   closed — taking the MLX curve down with it — with *"video-memory curve source catalog is not the
   exact sorted packaged catalog"*, which is the length check in `validate_source_catalog` firing
   before the later record-consumption check.

   🔴 **Three gates fire deliberately when the candle curve lands. All three are correct; clear
   them, do not weaken them.** The list was derived by promoting a synthetic candle lane end to end
   and diffing the whole `npm run check` failure set against a clean-head baseline (41 clean, 44
   with two lanes); no fourth gate appeared by that method:

   * `platform-review-contracts.test.mjs` — *"Rust Docker builders copy every production generated
     embed from sceneworks-core"*. The new `include_str!` needs a matching
     `COPY docs/generated/wan-candle-video-sc-19057.json ./docs/generated/` in **both** Rust builder
     stages of `docker/rust.Dockerfile`. Without it the crate compiles locally and fails in Docker,
     which is the standing `include_str!`-outside-the-build-context trap.
   * `generate-candle-admission-inventory.test.mjs` — *"the packaged video curves are MLX-only,
     which is why no candle route reaches the bundle"*. This is the recorded-gap tripwire; its own
     assertion says *"a candle curve now exists — update the recorded gap"*. Retire the
     `zero-candle-video-memory-curves-packaged` known gap and let
     `summary.byMechanism.video_memory_curve_bundle` reflect the route that now reaches the bundle.
   * `generate-candle-admission-inventory.test.mjs` — *"the committed artifacts match a fresh
     build"*. The candle-admission inventory hashes both
     `crates/sceneworks-core/src/video_memory_curves.rs` and
     `docs/generated/video-memory-curves.json` as sources, so adding the `include_str!` and
     promoting the curve rotates two source digests plus the aggregate `sceneWorksRevision` and the
     committed inventory goes stale. Re-run `npm run generate:candle-admission` and commit the
     result. Do this **after** the two edits above, or it will need doing twice.

   🔴 **`--check` verifies ONE lane. Run it once per lane.** The producer reads the committed
   bundle and copies the other lanes' curves verbatim into the artifact it compares against, so a
   candle `--check` can never detect tampering or rot in the MLX curve, or vice versa. Each lane's
   round-trip is only a statement about that lane. (`check:ltx-temporal-fit` is wired into no CI
   lane and is not a blocker; this is about what a manual run does and does not prove.)

   🔴 **Its REGRESSORS are not backend-neutral. Constrain the fit to the `cross` form on this lane.**
   `latentTemporalDepth()` hardcodes LTX's `1 + (frames - 1) / 8` and `latentTokens()` LTX's `/32`
   spatial factor, and `pointsFrom` recomputes both from the raw geometry rather than reading the
   arm's own `latentTemporalDepth` diagnostic. Wan's z48 VAE is **4x** causal, so for a Wan record
   the reported `tLat` is non-integral and physically meaningless, and every coefficient derived
   from it (`latent_tokens`, and the `tLat`-dependent half of the reported spread) describes
   nothing. Two consequences for the terminal-phase operator:

   * Read and promote the `cross` candidate only — `fixedGb + perMpxGb*mpx + perMpxFrameGb*mpx*frames`.
     That is what the six-row plan is designed for (two pixel counts x three frame counts, three fit
     rows and two held-out), and it is the ONLY form the runtime reader evaluates:
     `crates/sceneworks-core/src/video_memory_curves.rs` implements the affine cross form and nothing
     else, so a winning `latent_tokens` fit could not be read back even if it were meaningful.
   * Do **not** quote a `tLat`- or `latent_tokens`-derived number in the acceptance write-up. If the
     fitter reports one as the winner, that is an artifact of the hardcoded LTX stride, not a result.

   Teaching the fitter a per-model temporal stride is a real change and belongs in the fitter, not in
   a capture story; until then this caveat is the binding.
4. The consumer side is already wired: `video_admission.rs`'s `curve_backend` maps `VideoLane::Candle`
   to `VideoCurveBackend::Candle`, and `VideoMemoryCurveBundle::evaluate` fails closed on a foreign
   lane — so a candle reader picks up the candle curve and can never read the MLX coefficients.
5. Regenerate the derived docs (§8) and re-run `npm run check`.

Per-lane numbers only. Do **not** average, blend, or fall back across lanes: the whole point of the
lane tag is that an MLX coefficient is not evidence about a CUDA card.
