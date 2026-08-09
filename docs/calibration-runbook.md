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

Verified output shape (run on `origin/main` at inference pin `40fa7583`):

```
9 declared lanes: 8 stale, 0 current, 1 unmeasured (declared but never captured).
33 shipped calibration bindings and 65 evidence records are serving under a widened margin.

#  LANE                         BINDINGS  RECORDS  MARGIN  ESTIMATE  IMPACT  MODELS
1  mlx:qwen_image               9/9       41/41    5.00%   10.00%    0.450   qwen_image
2  mlx:z_image_turbo            5/5       5/5      5.00%   10.00%    0.250   z_image_turbo
...
DECLARED BUT NEVER CAPTURED (not stale — no measurement to be stale): candle:z_image
```

`IMPACT` is stale bindings × the margin the runtime is applying to them right now, so the top row is
the lane where a capture buys the most. `BINDINGS` is the **shipped admission surface** (what the
worker's fit gates consult today); `RECORDS` is corpus debt. Prefer bindings.

🔴 **This report enumerates DECLARED lanes only, so it cannot tell you a lane does not exist.** A lane
is keyed exactly as `config/inference-provider-closures.json` keys it
(`scripts/stale-lane-report.mjs:23-28`); a lane absent from that file appears **nowhere in the output**
— not in the stale table, and not under "declared but never captured", which lists only lanes that
*are* declared. Measured while validating this runbook (sc-18104): `mlx:flux2_dev` produces zero rows
even though FLUX.2 [dev] ships an MLX lane in `builtin.models.jsonc`. If you arrived here with a lane
name from outside this report — a product decision, a story, "the most popular model's Mac lane" —
**do not read its absence as "nothing to do"**. Go straight to §2 and check it explicitly.

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
| `memory-mlx-adapter` | `qwen_image`, `z_image_turbo`, `krea_2_turbo_control`, `flux2_dev` (sc-18218 — **resident rung only**; every other strategy is `Missing` on the pinned FLUX.2-dev contract, and the arm refuses a non-resident rung by name) | `mlx.rs` `run` — `MLX five-rung calibration does not implement provider "<id>"`; `validate_z_image_batch` (`assess_batch`) — `…five-rung batch assessment does not implement provider "<id>"` (cited by function name; the line numbers this table used to carry went stale the first time the file grew) |
| `memory-candle-adapter` | `qwen_image`, `krea_2_turbo` | `candle.rs:540-548` — `Candle five-rung calibration does not implement provider "<id>"` |

Grep before you schedule:

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

> Snapshot drift note: sc-18218 has since added **5 authoritative `mlx:flux2_dev` entries** (q4/q8
> at 768² and 1024², bf16 at a reduced 256²) with an adapter arm and a closure declaration, so the
> live totals are 160 authoritative entries across 9 lanes. The measured proportions above are the
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

> The worked example has since been closed out exactly that way: sc-18218 landed the arm
> (resident-only — the pinned FLUX.2-dev contract marks every other strategy `Missing`, registers on
> the edit provider, opens no request scope and has no fault-injection site, so captures are
> `runtime_complete`-shaped), the 5 authoritative plan entries, and the closure declaration in one
> change. `mlx:flux2_dev` now screens as row five: capturable, pending its first capture (sc-18104).

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

### Adapter environment — six families, one per provider arm

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
SCENEWORKS_FLUX2_ROOT=/abs/path/.../snapshots/<rev>/<tier>   # bf16 | q4 | q8 — tier DERIVED from the plan target

# memory-mlx-adapter — any lane, optional
SCENEWORKS_MLX_WIRED_LIMIT_BYTES=<explicit wired-ceiling override>

# memory-candle-adapter — krea_2_turbo
SCENEWORKS_KREA_REPOSITORY=SceneWorks/krea-2-turbo-mlx
SCENEWORKS_KREA_REVISION=<exact artifact revision>
SCENEWORKS_KREA_ROOT=/abs/path/.../snapshots/<rev>/q4
```

All three of each family are **required** (`protocol::required_env`) — a missing one fails before
model load, not after. Note that `memory-calibration-harness.md` documents only the Qwen and Krea
families; the Z-Image and Krea-control blocks above exist only here.

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
  -f qwen_revision=<exact 40-hex artifact revision>
```

Validates the identities, resolves (without printing) the snapshot root, checks out inference at the
exact revision into `.calibration/inference`, builds the release adapter, runs the harness with
`--fresh-per-case` on fixture `qwen-image-<tier>-seed<15511|16353>-step2` (seed 15511 for `bf16`,
16353 for the packed tiers — macos-mlx.yml:809-813), schema-checks the bundle and uploads it as
`memory-mlx-evidence-<tier>-<run_id>`.

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
