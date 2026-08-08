# Calibration measurement runbook

**One parameterized procedure for measuring one `<backend>:<provider>` lane, end to end.**

Substitute a single lane name — for example `mlx:z_image_turbo` or `candle:krea_2_turbo` — and follow
this top to bottom. Everywhere below, `<lane>` means that string, `<backend>` its part before the
colon (`mlx` or `candle`) and `<provider>` its part after.

This replaces the pattern of filing a story per model. Measurements are made by following this
document; only lanes someone actually cares about get measured.

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

The gate inventory behind the report — what does and does not grade currency — lives in
[calibration-invalidation-sc-17774.md](calibration-invalidation-sc-17774.md) ("What grades what",
"What does NOT grade currency").

## 2. Can this lane actually be captured today?

Three independent things must all be true. Check them **before** booking a GPU.

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
ids they name today:

| binary | providers named in the source |
| --- | --- |
| `memory-mlx-adapter` | `qwen_image`, `z_image_turbo`, `krea_2_turbo_control` |
| `memory-candle-adapter` | `krea_2_turbo`, `qwen_image` |

Grep before you schedule:

```bash
grep -n '"<provider>"' crates/sceneworks-memory-adapter/src/bin/<backend>.rs
```

**This is the gate that stops most lanes.** `candle:z_image` is declared in the closure table and has
five plan entries, and `grep -c z_image crates/sceneworks-memory-adapter/src/bin/candle.rs` returns
`0` — which is exactly why the stale-lane report lists it as "declared but never captured". A lane
with no adapter arm is an adapter-implementation task, not a measurement task. Stop here and say so.

## 3. Pick the host

| lane | host | how CI reaches it |
| --- | --- | --- |
| `mlx:*` | Apple silicon — the dev Mac, which **is** the self-hosted `nax-macos` runner | `.github/workflows/macos-mlx.yml`, `runs-on: [self-hosted, macOS, ARM64, nax, weights]` when `run_memory_calibration` or `run_five_rung_reference` is set (macos-mlx.yml:429) |
| `candle:*` | a CUDA box | `.github/workflows/windows-candle.yml`, `runs-on: [self-hosted, Windows, X64, cuda]` (windows-candle.yml:266) |

Note the extra `weights` label on the MLX calibration dispatch: the ordinary MLX CI job runs on the
`nax` pool without it. A capture dispatch that cannot find a `weights`-labelled runner queues
forever rather than failing.

You can run the capture **locally on the host** or **through the guarded workflow dispatch**. §6
gives both. The dispatch is preferable for MLX because it re-validates the identities for you; the
local path is the only option for a lane the dispatch does not wire (the MLX dispatch wires Qwen and
the Z-Image five-rung reference; the Windows dispatch wires only the Krea five-rung reference).

## 4. Weights must really be present for the target tier

**`.sceneworks-model-revision` is the only sound POSITIVE signal.** It is written inside
`snapshots/<revision>/` by the inference repo's release tooling
(`scripts/release/verify_model_snapshot.py`, `MARKER`) and never by the app. Its **presence** proves a
snapshot was CI-provisioned. Its **absence proves nothing** — an app-installed or hand-fetched
snapshot is complete and unmarked. Do not treat a missing marker as a missing snapshot, and do not
treat a present tier directory as a complete one.

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
  `release/real-weight-models.toml` has 62 `[[models]]` keys, and `SceneWorks/z-image-turbo-mlx` and
  `SceneWorks/krea-2-turbo-mlx` are **not** among them (`z-image-turbo` there is the upstream
  `Tongyi-MAI` repo, a different artifact). For those lanes there is no `verify_model_snapshot`
  binding to run — fall back to the workflow's own resolver contract below and say plainly in the PR
  that no manifest-bound verification exists for the lane.

### Where the adapters look

The MLX adapter canonicalizes its root and requires a fixed
`/models--SceneWorks--<repo>/snapshots/<exact-revision>/<tier>` suffix before loading, so a stale
override for another tier is rejected rather than silently used. Without an override the macOS
workflow checks exactly two locations, in order (macos-mlx.yml:742-747):

```
$HOME/.cache/huggingface/hub/models--SceneWorks--qwen-image-mlx/snapshots/<rev>/<tier>
$HOME/Library/Application Support/SceneWorks/data/cache/huggingface/hub/models--SceneWorks--qwen-image-mlx/snapshots/<rev>/<tier>
```

Adapter environment (from [memory-calibration-harness.md](memory-calibration-harness.md)):

```bash
# MLX
SCENEWORKS_QWEN_IMAGE_ROOT=/abs/path/to/<tier>
SCENEWORKS_QWEN_IMAGE_REPOSITORY=SceneWorks/qwen-image-mlx
SCENEWORKS_QWEN_IMAGE_REVISION=<exact artifact revision>
SCENEWORKS_MLX_WIRED_LIMIT_BYTES=<optional explicit override>

# Candle
SCENEWORKS_KREA_ROOT=/abs/path/to/q4
SCENEWORKS_KREA_REPOSITORY=SceneWorks/krea-2-turbo-mlx
SCENEWORKS_KREA_REVISION=<exact artifact revision>
```

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

## 6. The capture

`--output` and `--resume` should point **outside the repository**. `.tmp/` is gitignored today
(`.gitignore:298-301`, added deliberately for this), so it also works — but the failure mode when an
ignore rule changes is a multi-hour sweep discarded at the end, and the CI lanes themselves write to
`$RUNNER_TEMP`. Use a path outside the tree and the question does not arise.

### 6a. Locally on the host

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

### 6b. Through the guarded dispatch (MLX)

```bash
gh workflow run macos-mlx.yml --ref main \
  -f run_memory_calibration=true \
  -f provision_qwen_snapshot=false \
  -f qwen_tier=bf16 \
  -f inference_revision=<exact adapter INFERENCE_PIN> \
  -f qwen_repository=SceneWorks/qwen-image-mlx \
  -f qwen_revision=<exact 40-hex artifact revision>
```

The job validates the identities, resolves (without printing) the snapshot root, checks out inference
at the exact revision into `.calibration/inference`, builds the release adapter, runs the harness with
`--fresh-per-case`, schema-checks the bundle, and uploads it as
`memory-mlx-evidence-<tier>-<run_id>` (macos-mlx.yml:799-831). Retrieve it with `gh run download`.

The Windows equivalent is `run_five_rung_reference=true` on `windows-candle.yml`
(`provision_krea_snapshot`, `inference_revision`, `krea_repository`, `krea_revision`); it captures
fresh **and** batched bundles and runs `compare-reuse` between them (windows-candle.yml:460-475).

### 6c. What it costs

Cited from `docs/generated/calibration-cost-model.json` (generated by
`scripts/calibration-cost-model.mjs`). **Provenance note:** that generator and its artifact may be
retired by sc-18100, so these figures are inlined here as facts rather than linked, and should be
re-derived from a real capture rather than trusted forward.

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

## 7. Ingest, stamping, and a new lane

### 7a. Merge into the committed bundle

```bash
node scripts/memory-calibration-harness.mjs ingest \
  --input /abs/path/OUTSIDE/the/repo/<lane>-evidence.json \
  --resume docs/generated/memory-calibration-evidence.json \
  --output /abs/path/OUTSIDE/the/repo/merged.json
```

Merge is commutative and stable-sorted; **different content under the same resolved identity is
rejected** rather than resolved by arrival order. Copy `merged.json` over
`docs/generated/memory-calibration-evidence.json` once it validates.

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

A dry run (no `--write`, no `--verify`) prints what it would change and exits 0. `--verify` is the CI
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

Every authoritative lane in the plan must have an entry in
`config/inference-provider-closures.json`, and `memory-calibration-harness.test.mjs` asserts it. Add
the lane's inference crate directory and regenerate the whole table from a real clone — never by
hand:

```bash
node scripts/inference-closure-digest.mjs --repo <inference clone> --write
node scripts/inference-closure-digest.mjs --repo <inference clone> --check   # verified: prints
                                                # "config/inference-provider-closures.json matches 40fa7583"
```

The key is `<backend>:<provider>`; a bare provider id is ambiguous (`krea_2_turbo_control` exists on
both backends with different crates). `buildClosureConfig` also asserts the provider is really
declared in the crate you named, so a wrong crate fails rather than digesting the wrong tree.

## 8. Regenerate the derived docs

```bash
npm run generate:memory-matrix
npm run generate:calibration-cost-model   # only while that generator still exists (sc-18100)
npm run check:rust-derived-docs           # memory-matrix + cost model + tier integrity, all --check
npm run check:memory-calibration          # schema-checks the committed bundle
```

Verified green on `origin/main` today. `check:rust-derived-docs` ends with
`tier integrity OK: 111 declared exceptions (111 measured, 0 unmeasured).`

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
docs-only**, and you must plan to update these in the same commit:

| test | assertion | notes |
| --- | --- | --- |
| `scripts/generate-memory-matrix.test.mjs:2439` | `"current evidence cannot promote through a historical exact manifest binding"` | |
| `scripts/generate-memory-matrix.test.mjs:653` | `"historical bindings must not mask the pinned MLX provider contract"` (Z-Image MLX static contracts) | reds on an `mlx:z_image_turbo` capture in particular |
| `scripts/sc-15823-flux1-evidence.test.mjs:189` | `runs.every(({ semantics }) => semantics === "historical")` | flux1 lanes |
| `tests/test_memory_matrix.py:147-148` | `{run["semantics"] for run in full_runs} == {"historical"}` and `… == "current") == 0` | **Python — see below** |

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
above `tests/test_memory_matrix.py:147` explains why it is pinned as an exact set and count — "a bare
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
records you landed. Then confirm the signal agrees:

```bash
npm run report:stale-lanes    # your lane should have left the stale list
```

A record that stays `historical` means its captured digest does not match the live one for its lane —
usually the inference checkout was not at the pin (§5), or the closure table was regenerated after the
capture.

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

One commit, referencing the story. The evidence bundle, the regenerated derived docs, any closure-table
change and any relaxed test pin **go together** — a bundle without its regenerated matrix, or a
regenerated matrix without the relaxed pins, is a red CI and a misleading intermediate state.

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
