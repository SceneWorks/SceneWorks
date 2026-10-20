# sc-26243 installed-product terminal driver

This is executable acceptance **infrastructure**, not an acceptance receipt. No real
case is implied to have run. The parent coordinator owns actual resource approval,
paired MAIN integration, the final inference pin, campaigns, review, and merges.
Resource anchors/conversion observations are separate from saved-image proof.

## Safe planning and CPU tests

From the SceneWorks checkout, with Node 20:

```sh
node scripts/flux3-image-klein-terminal-driver.mjs plan
CUDA_VISIBLE_DEVICES=-1 node --test scripts/flux3-image-klein-terminal.test.mjs scripts/flux3-image-klein-terminal-budget.test.mjs scripts/flux3-image-klein-terminal-review-fixes.test.mjs scripts/flux3-image-klein-app-lifecycle.test.mjs scripts/flux3-image-klein-candle-app-lifecycle.test.mjs
```

No arguments also means `plan`. Planning only reads the shipped model manifest and
runtime capability catalogs. It never launches a service, reads a token, downloads,
builds, writes a cache, or captures a measurement. The ordinary historical MLX and
Candle smoke entrypoints keep their existing default request and behavior. The
terminal driver shares their nonce/env/health/owned-API/accepted-job/saved-asset/PNG
assertions; it does not impersonate a worker or publish provider images itself.

`plan` is a SMALL REPRESENTATIVE code default: **26 product generations + 12
resource generations = 38 total across both backends**, including the two training
reloads. The hard global ceiling is **48**, not 48 per machine. The remaining ten
are **unallocated, not authorized**, and cannot be used automatically. There are
zero implicit warmups, repeats, seeds/rungs, or retries. Exactly two short training
lifecycles are listed separately: MLX edit and Candle captioned, four steps each,
save every step, rank 4, resolution 512. Their generation reloads already count in
26; training previews are disabled.

The default named product list uses these suffixes with `mlx.` and `candle.`:

| Case suffix | Generations per backend | Combined obligations |
|---|---:|---|
| distilled.representative-repro.bf16 | 2 | Distilled T2I/PUP-off, persisted settings, same-seed fresh-process reproduction |
| distilled.representative-pup-text.q4 | 1 | Native q4, PUP t2i, portrait |
| distilled.representative-ref-1.bf16 | 1 | Ordered single distilled reference, PUP i2i, landscape |
| distilled.representative-layout-max.q8 | 1 | Native q8, PUP t2i_bbox, layout placement, 4096x4096 |
| base.representative-move-remove-10.bf16 | 1 | Ordered ten base references, PUP i2i_bbox, source/target moves and removals |
| base.representative-base-ref-1.bf16 | 1 | Single base reference, PUP bypass, portrait |
| distilled.representative-distilled-ref-10.bf16 | 1 | Ten distilled references, PUP bypass, landscape |
| distilled.lora-zero.bf16 / distilled.lora-on.bf16 | 2 total | Compatible real adapter identity, scales 0/0.75 and image-effect comparison |
| base.train-interrupt-resume-export-edit.bf16 (MLX), or ...-captioned.bf16 (Candle) | 1 reload | Four-step TRAIN, interrupt, actual checkpoint state, fresh resume/export, fresh ordinary base T2I adapter reload |
| base.cancel.bf16 | 1 | Progress then real cancellation |
| base.representative-native.fp8r (Candle only) | 1 | Actual CUDA FP8R package |
| distilled.representative-native.nvfp4 (Candle only) | 1 | Actual CUDA NVFP4 package |

This is **12 MLX + 14 Candle product generations**. Resource allocations are
explicitly **5 MLX + 7 Candle**: bf16 distilled/base on each backend (four anchors),
MLX distilled q4/q8 and Candle distilled q4/q8/nvfp4 plus base fp8r (six native
package anchors), and base `pup_edit.bf16` (exact native supplemental contract ID, ten references) once per backend (two
supplemental PUP observations, never extra anchors). The exact names and expanded
request templates are emitted as `generationInvocations`; each row is ONE actual
inner `Generator.generate`, not one command that may hide multiple samples.

Resource rows remain **BLOCKED** until the authorized resource owner verifies the
actual native single-call permit boundary and binds its real verification receipt
and normalized request digest before approval. A `nativeGenerationCalls:1` flag is
NOT execution proof. Hidden harness warmups/rungs/seeds must be disabled legitimately
or expanded into explicit permits, reducing the plan before approval. This driver
does not edit or claim verification of the native resource harness.

API-only checks (bad references/count/boxes/checkpoint compatibility, separate
installation/auth faults, and explicitly approved provisioning) add **zero image
generations**. Negative image/training ingress probes stop ONLY this session's GPU
worker. Any unexpectedly accepted job is canceled and terminal-verified while the
worker remains paused; unsafe acceptance aborts rather than resuming unknown jobs.

### Optional diagnostic catalog — NEVER the default campaign

`node scripts/flux3-image-klein-terminal-driver.mjs catalog` lists the larger
available obligation inventory without executing anything. It cannot itself be an
approved execution plan or automatically expand the representative profile. IDs
are `<mlx|candle>.<distilled|base>.<scenario>.<tier>`. Rows retain source identity,
ordered bindings, options, expected observables, and NOT_RUN/BLOCKED. The following
table describes that diagnostic inventory, **not a request to render every cell**:

| Phase | Cases per checkpoint/backend | Required observation |
|---|---|---|
| prerequisites | install-identity | Catalog-installed pinned package plus real inventoried files; **not** download/auth success |
| prerequisites | install-provision; install-provision-pup once per backend | Explicit approved whole-repo ordinary download, authenticated fixed pinned config, actual job-bound v2 receipt and complete inventory |
| ordinary | t2i, cancel, saved-settings-repro | Running progress, cancel transition; persisted recipe reconstructs request; fresh-process deterministic output |
| references-layout | refs-1, refs-10, layout-placement, box-move-remove | Ordered source IDs/content hashes/instance IDs; authored coordinates; move and explicit null removal |
| upsampler | pup-t2i, pup-i2i, pup-t2i_bbox, pup-i2i_bbox | Enhancement on; exact mode, enhanced outcome, original/effective prompt, no unexpected fallback, locked annotations |
| adaptation | lora-zero, lora-on | Real provisioned compatible adapter/hash/rank; actual API `weight` 0/0.75; observable image difference |
| adaptation, base only | train-interrupt-resume-export-captioned/edit | Bounded real TRAIN, checkpoint state, interruption, fresh-process resume, advancing factors/optimizer/RNG state, export, fresh ordinary-generation reload |
| native-tiers | native-tier.q4/q8; Candle also fp8r/nvfp4 | Actual named package, no silent downgrade/conversion-as-success |
| limits | portrait, landscape, max-resolution | 768x1024, 1024x768, 4096x4096 PNG proof; each needs its own admitted cost |
| failures | bad-reference, reference-count, bad-box, checkpoint-compatibility | Intended actionable rejection, not unrelated missing-model failure |
| failures | install-missing, authentication-denied | Separate empty-cache fault profile; injected denial is not successful upstream account authentication |

Base q4/q8 have no production catalog rows/trusted real conversion hashes yet.
Their cells stay BLOCKED, with an API refusal check when selected. Do not fabricate
rows/hashes or reinterpret this gap as an unsupported exception. Missing base
access (including existing HF403), packages, PUP, adapters, or checkpoint provenance
must be actionable BLOCKED, never fallback/skip-as-PASS.

## Future execution contract — not authorization to run

The concrete CLI accepts four local JSON files, not an arbitrary launcher module:

```sh
node scripts/flux3-image-klein-terminal-driver.mjs run /absolute/admission.json /absolute/inputs.json /absolute/case-ids.json /absolute/approved-plan.json
```

`case-ids.json` is an explicit array of IDs copied from plan output. Select a small
phase/backend batch, not the whole matrix blindly. No workflow is dispatched and
no build is performed. A separately built exact-source native rust-api binary is
required. The driver verifies its SHA256 and the Cargo inference pins.

The admission file requires:

- `actualRun:true`, `admissionId`, `approvedBy`, `backend`, explicitly allowed
  `phases`, and bounded `timeoutSeconds` (60..10800). `runnerAccount` must match
  the actual OS account name; it is non-secret provenance, not a token.
- Exact clean checkout `sourceSha`, exact `inferenceMainSha`, local
  `inferenceRepository`, and `inferenceMainTipSha`. SceneWorks source must be the
  approved clean, frozen **feature** commit (SceneWorks MAIN ancestry is not
  required). The actual resolved inference pin must be reachable from its locally
  fetched `origin/main`; a field merely named MAIN is insufficient. The driver
  never fetches or merges.
- Absolute sealed `binary`, its `binarySha256`, redundant `binarySourceSha ===
  sourceSha`, and `binaryBuildReceipt:{root,path,sha256}`. A caller source label or
  current lock beside an old executable is not compiled provenance. The global
  approval's `buildReceipts[backend]` binds the real producer receipt/transcript,
  binary, lock, tool executables and flags (see below).
- Existing absolute `outputRoot` outside the checkout and existing absolute
  `hfHome`. Stable repositories, retained PNGs, sidecars, and reports must survive
  worktree removal. Ordinary execution uses the installed cache offline and never
  calls download/relocate/reverify endpoints.
- `inputSpecSha256`: SHA256 of `JSON.stringify(parsedInputsJson)` (UTF-8), binding
  the approved provisioned inputs, package inventories, and adapter metadata.
- `disk`: `freeBytes`, `present` keyed by `repo@revision`, and explicit
  `conversionOutputBytes` keyed by artifact key. The budget deduplicates all
  requested whole repositories, includes optional PUP when requested, conversion
  outputs, 32 GiB build and 16 GiB state safety margins. Margins are **not
  measurements**. Actual free space is checked conservatively on both roots.
  Ordinary operation does not initiate a missing download even if disk permits it.

### Separately admitted build producer — never executed by planning/run

`flux3-image-klein-terminal-build.mjs` supplies the concrete future build producer;
`produce-build /absolute/build-admission.json` requires separate explicit
`actualBuild:true`, `userBuildApprovalId` and `resourceAdmissionId`. It observes
clean exact source/manifests/lock before and after a fixed offline locked release
build, jobs 2, CVD=-1, native platform target, Candle feature on Windows. Actual
Cargo metadata must resolve every inference edge to the approved MAIN-reachable
pin; real compiler-artifact/build-finished messages must identify the actual
executable. Tool executable hashes, actual rustc host/version, flags, and on
Windows actual MSVC 19.44 `/Bv` output bind the transcript. The admitted warm target
and exclusive sealed-copy output stay outside the checkout. Nothing here grants
permission to build or use the resource owner's target.

The parent-approved `buildReceipts[backend]` includes `receiptSha256`,
`transcriptSha256`, `binarySha256`, `cargoLockSha256`, `rustflagsSha256`,
`cargoExecutableSha256`, `rustcExecutableSha256`, plus Windows
`msvcCompilerSha256` and `cudaCompilerSha256` (actual nvcc 12.9 version/hash,
selected CUDA_PATH/host compiler). Build scratch uses an exclusive producer-owned
temp directory, not ambient shared TEMP. The launcher verifies current producer source/lock, observed
source/pin/recipe, and actual sealed binary bytes. Arbitrary older executables,
mixed pins, source changes during build and re-labelled receipts are rejected.

The fixed `binaryRole` is `api-worker` (default) or `resource-adapter`, not a caller
build command. The resource role builds package `sceneworks-memory-adapter`, bin
`memory-candle-adapter` with feature `candle` on Windows, or bin
`memory-mlx-adapter` with feature `mlx` on the secondary Mac, using the same locked
offline native-target release recipe. Its separately sealed receipt is bound in
`approvedPlan.resourceBuildReceipts[backend]`; an API/worker build receipt cannot
stand in for the actual resource adapter executable. Both role receipts are needed
before the combined physical campaign is approved.

### Immutable global budget and local single-use ledgers

`approved-plan.json` has `schemaVersion:1`, `kind:"flux3-terminal-plan-approval"`,
non-secret `approvalId`, **explicit user** `userPlanApprovalId`, `approvedBy`, exact
final `sourceSha` and `inferenceMainSha`, stable `ledgerId` (12..100 safe characters),
fixed absolute `ledgerRoots:{mlx,candle}`, both per-backend `inputSpecSha256` values,
and `plan` containing the exact expanded representative plan. Bind real resource
boundary verification receipts/normalized request hashes before approving and
initializing; do not initialize an incomplete plan and later reset it. Diagnostic
catalogs are never executable approval files. The admission additionally carries
`ledgerId`, its backend's exact `ledgerRoot`, and `approvedPlanSha256` (the helper's
SHA256 of `JSON.stringify(parsedApprovalJson)`).

The same GLOBAL approval assigns disjoint fixed quotas: **17 MLX and 21 Candle**,
including resource calls. There is no distributed service, second 48-call quota,
or automatic allocation of ten spare permits. Each backend's product, resource,
and resumed sessions use its SAME stable local ledger outside the checkout.
Explicit parent/user-approved CPU setup, performed once per backend, is:

```sh
node scripts/flux3-image-klein-terminal-driver.mjs init-ledger /absolute/approved-plan.json candle
# On the physical secondary Mac, use mlx and its fixed approved ledger path.
```

The parent directory must already exist; the ledger directory must be new. A
sibling exclusive initialization claim prevents reinitialization even if the ledger
directory is deleted. Actual `run` NEVER creates, resets, repairs, or refunds a
ledger. A missing/busy/stale/crash-mismatched ledger is a fail-closed blocker for
coordinator reconciliation, not permission to use a new directory or new counter.

Before any real launch, the CLI prints the **exact expanded physical invocation
list, request templates, resource phases, zero warmups/repeats/retries, fixed
training steps, and approved-plan hash**. It requires the explicit plan approval,
MAIN/source identity, input hashes, per-phase resource admission, and existing
ledger. Before every ordinary generation POST, the concrete launcher durably
consumes the assigned invocation ID; `count:1` is enforced and hidden hires passes
are refused. Training starts/resume are separately single-use and previews disabled;
the reload uses a generation permit. Failed/canceled/never-dispatched reservations
stay spent. Extra attempts, alternative ledger roots, automatic retries, and spare
allocation are refused. Any retry needs a new explicit parent/user decision and
remaining global budget; this driver does not amend an already-running plan.

`flux3-image-klein-terminal-budget.mjs` exports `reserveInvocation`, `auditLedger`,
and `aggregateLedgerReceipts`. The resource owner must consume a permit immediately
before EACH actual native `Generator.generate`, with `kind:"resource_anchor"` or
`"resource_supplemental"`, exact approved invocation/backend/case ID and normalized
`requestSha256`. Resource reservation refuses the current BLOCKED boundary state;
a self-reported one-call flag is insufficient. This contract is **not** a claim
that the unverified native harness already enforces it.

The synchronous adapter-child permit CLI must first call shared
`verifyResourceProducerHardware(admission, context)`, with exact `adapterPid`,
`caseId`, `invocationId`, normalized `requestSha256`, `approvedPlanSha256`,
`sourceSha` and `binarySha256`. It verifies that its **actual parent PID** is the
adapter, inspects that process's real executable path/start identity, verifies
that executable against the sealed resource-role build, and observes hardware
using the existing guard. Candle requires that actual adapter PID in GPU1's
compute-apps UUID observation; Mac requires the observed secondary host/arm64
runtime and monitored exclusive lease. Re-labelled metadata alone never passes.
A second process observation rejects exited/reused PIDs. The returned bound
hardware observation is retained with the reservation, before native generation;
this verifier never spends a second permit. A missing collector/proof blocks
resource execution rather than accepting request labels. Real boundary/hardware/
build verification remains required before resource rows become READY.

Return the latest audited
receipt per backend to the coordinator, who aggregates unique IDs against the same
global plan and total <=38 approved / <=48 hard ceiling. Full exact manifests,
contiguous event sequences/hash chains, unique backend receipts, final state
counts/head and single-use IDs are verified; truncated, replayed or contradictory
receipts are rejected. This is `reservation-accounting-only` with
`caseCompletionProven:false`, not case completion evidence. Do not add cumulative
snapshots together as though their repeated event histories were new runs.
Final `aggregateLedgerReceipts(approval, receipts, closure)` requires **both**
backend snapshots and a separately trusted parent closure: `kind:
"flux3-terminal-finalized-ledger-heads"`, `coordinatorClosureId`,
`producersStopped:true`, `collectionManifestSha256`, exact `ledgerId` and
`approvedPlanSha256`, and `backends:{mlx:{sequence,headSha256},candle:{sequence,
headSha256}}`. The parent obtains those latest heads from observed finalized
ledger closure/sealed collected artifact manifests after producers stop, never
from the candidate receipt's own freshness label. A valid-chain older snapshot
with a different trusted latest head is rejected. Partial run reports may still
audit their local ledger, but missing backend accounting remains INCOMPLETE/unknown,
never zero or campaign PASS. There is no network service or signing framework.

Candle additionally requires `physicalGpuId:"1"`, `cudaVisibleDevices:"1"`,
`logicalCudaOrdinal:0`, admitted `gpuUuid`, and `runner`. The controller checks
physical GPU1 against masked CUDA0 UUID, rejects foreign Runner.Worker processes
(only this controller's hosting ancestor is allowed) and every foreign GPU1
compute-app PID, **including C+G**. Graphics-only desktop processes remain allowed.
It monitors the owned worker PID/device and never selects user-reserved GPU0. MLX requires physical Apple Silicon, `runner:"nax-macos-2"`,
`runnerTarget:"secondary"`, `runnerLabel:"rw-mage"`, and exact admitted `hostName`.
`RUNNER_NAME` must be `nax-macos-2`; `rw-mage` is a label, not the runner name. The
primary Mac is forbidden. Labels alone never prove Metal exclusivity: MLX also
requires `macWorkloadLease:{root,path,sha256}` binding a coordinator-issued
`flux3-secondary-mac-exclusive-workload-lease`, exact admission/source/pin/account/
host identity, `exclusiveGpuWorkload:true`, `noForeignGpuWorkload:true`,
`observedAt`/`expiresAt` covering the fixed run deadline, actual
`ancestorRunnerPids`, and `baselineProcesses:{pid,ppid,command,startedAt}[]`.
New/reused foreign processes or another runner invalidate the lease. Unprovable
exclusivity is BLOCKED, never a blind launch or permission to kill others.

Each session uses fresh cryptographic nonce state, a real port-zero API listener
announcement, canonical owned-health paths before **every** request, and only its
own child handles for cleanup. Training resume changes PID **and** nonce-derived
worker identity before submitting ordinary retry with `resume:true`; reload starts
another fresh process. Children receive scrubbed offline environments, never
ambient credentials. No global process-name kill is used.

### Input file

`inputs.json` fields (see adjacent schema):

- `images`: binding name to `{root,path,sha256}` for real provisioned PNG files.
  Reference cases use `reference-0` through `reference-9`. The driver hashes and
  uploads them through the owned project's ordinary multipart assets API; input
  IDs from another project are never reused. Encoded file SHA is retained separately
  from the worker's actual carrier digest: SHA256(RGB=0/RGBA=1 byte + oriented
  width/height little-endian + original decoded pixels, including hidden alpha RGB).
  The proof fixtures use noninterlaced explicit RGB/RGBA8 PNG, at most 16 Mpixels;
  EXIF/tRNS ambiguity is BLOCKED for reprovisioning, never silently reoriented or
  flattened. This is a bounded driver-fixture contract, not a claim that the product
  rejects other supported image formats. Ordinal, oriented size, instance IDs and
  decoded carrier hash must agree with actual worker facts. Duplicate asset IDs in
  conditioning are legal only as separate ordered instance IDs.
- `packages`: artifact key `<modelId>:<backend>:<tier>` to
  `{root,repository,resolvedRevision,authority:"parent-approved-pinned-inventory",
  files:[{path,bytes,sha256}]}`. These are approved pinned inventories, **not**
  self-authorizing conversion observation receipts. The approved input hash binds
  them. Local derivations must additionally match the production catalog's real
  primary weight hash/size. Dense packages include encoder, autoencoder, license,
  notices and manifests. Non-dense packages require `externalComponents` for
  `text_encoder` and `autoencoder`, each with pinned identity and full inventory.
  Publisher `variants/fp8r`/`variants/nvfp4` packages bind their root repository's
  separate `license` inventory (LICENSE.md/NOTICE.md), rather than invent license
  files inside the variant subdirectory; the actual worker must advertise the CUDA
  format's device floor. PUP-on additionally requires `components.prompt_upsampler` with the separately
  pinned PUP inventory. Files are confined to physical roots and hashed; unchanged
  files are stat-checked rather than rehashing tens of GB for every case. Where
  an ordinary HF snapshot links into its blob store, explicitly approved
  `containmentRoot` names that physical owned hub; the package `root` remains the
  exact pinned snapshot, and escaping the admitted hub is refused. Dense snapshots
  and PUP additionally need the genuine original `installationReceipt:{root,path,
  sha256}`: v2, actual completed download job ID, exact repo/revision/model ID,
  `manifestFiles:[]`, and complete approved `resolvedFiles`. The launcher copies
  that verified provenance into its nonce-owned marker, never fabricates a receipt.
  q4/q8 require the real `flux3-conversion-receipt.json`, catalog weight hash/size,
  source/component inventories, and exact native quantization config. Packages are
  materialized **once** under `data/models/derived/<model>/<tier>/<conversion>`
  before any service spawn; fresh workers reuse them. The printed staging plan
  gives exact unique copy count/bytes and checks actual free space plus a 30 GiB
  floor before copying. Dense snapshots are read-only, not copied. TE/AE/PUP roots
  and used files must match the actual worker config/resolver-selected pinned
  snapshot, not simply the same repo/revision labels on another tree.
- `adapters`: model ID to `{root,path,sha256,bytes,rank,repository,resolvedRevision}`.
  The actual safetensors rank is read, and the ordinary LoRA import API verifies
  the file hash. No fixture is accepted as real adapter evidence. Run `lora-zero`
  before matching `lora-on` in the same session for image-effect comparison.
- `training.captioned` and `.edit`: `{steps,saveEvery,rank,items}`. The executable
  global profile permits exactly **4 steps, saveEvery 1, rank 4, resolution 512**,
  one lifecycle per backend and two total. Supply real item bindings/captions. Each item is
  `{binding,caption}`; edit items also require `referenceBinding`. Both bindings
  must exist in `images`. Config derives from `/training/targets`, never a made-up
  provider payload. Requests explicitly use `dryRun:false`; the admission must set
  `trainingLicenseAcknowledged:true` for the actual adaptation terms, never infer
  commercial-training permission from an output exception. Run base training
  before checkpoint-compatibility cases; no synthetic checkpoint is substituted.

### Explicit positive provisioning

Six provision cells add successful installation/authentication to the matrix: each
backend's distilled and base `install-provision`, and one
`distilled.install-provision-pup` per backend. The PUP cell retains its generation
target in `modelId`; `options.provisionModelId` selects the component-only shipped
`flux3_image_klein_pup` catalog entry. Artifact identity names the actual PUP repo.

Default execution remains offline. Selecting a provision cell without an exact
approval leaves it BLOCKED **before launcher or credential read**. For each
requested repo the admission additionally needs `downloadApprovals` entries:

```json
{
  "repository": "black-forest-labs/flux-3-image-klein-base",
  "resolvedRevision": "1469d27cfb9bb1bd81d3a93f5a023b39dbf136e4",
  "userDownloadApprovalId": "parent-recorded-user-approval-id",
  "wholeRepository": true,
  "accountGateAcknowledged": true,
  "diskAcknowledged": true,
  "downloadBytes": 43229093855
}
```

Only the three shipped frozen FLUX3 repositories/revisions are allowed, always
`files:[]`; no arbitrary repo/host/token argument is accepted. The shared legacy
runner-account credential boundary reads the selected account's `hfHome/token`
**only** in this explicitly admitted gated-model provisioning execution path.
Public PUP-only provisioning never reads a token and records a public pinned-repo
mechanism instead of pretending there was a gate. Planning and installed-only runs
never read credentials. The credential goes only to the owned
API's utility-worker child environment and the fixed Hugging Face pinned-config
request, not to the GPU worker, CLI arguments, JSON receipts, or artifacts. Raw and
URL-encoded forms are redacted from final private logs/reports. A failed runner
login/gate (including HF403) stays actionable BLOCKED, never a fallback PASS.

Positive proof requires the actual ordinary `model_download` job, its owned
utility worker, a schema-v2 receipt with the exact actual job ID/repo/revision and
`manifestFiles:[]`, exact complete `resolvedFiles` equality with the approved pinned
inventory, all file hashes, installed catalog identity, license terms, and the
installed config hash matching the real authenticated pinned config. An existing
installed flag or unrelated old receipt cannot pass this cell. No resource anchor
or synthetic unit-test receipt can replace this actual-run chain.

Whole pinned repository policy is `files:[]`: distilled 43,317,649,602 bytes,
base 43,229,093,855 bytes, optional PUP 18,839,746,673 bytes as declared in shipped
catalogs. This driver does **not** grant download approval. Parent/user approval,
runner-account gated access, and separate provisioning remain prerequisites.

### Explicit installation-failure profiles

A separate admission with `installationFaultAdmission:true` and
`faultProfile:"missing-install"` or `"hf403"` selects only the corresponding
failure case(s). It creates a fresh task-owned **empty** hub, never edits an
existing cache. Task-owned loopback responders return missing-package 404 or gated
403 and never read/log credentials or serve weight bytes. Both fault probes use the
ordinary CPU utility `model_download` path, never an unbudgeted image-generation
job. The product's real API/job must provide the intended actionable denial. Receipts identify the injected failure; they
cannot stand in for a successful upstream login/download. Positive download/auth
proof still comes from the separately approved provisioning run.

## Proposed required-CI budget and final delivery — PLAN ONLY

**No execution authorization is granted by this section.** The final MAIN PR and
MAIN push remain held until the parent/user approves the complete expanded plan
and GPU1 admission. The executable FLUX3 representative profile remains **38**
physical attempts with its unchanged 17 MLX / 21 Candle single-use permits.

The existing Windows workflow also automatically runs one **real-weight Krea**
smoke after its ordinary Candle job succeeds. It is not a weightless fixture and
is not silently outside the global budget:

| Proposed additional attempt | Trigger | Exact native job |
|---|---|---|
| `ci.windows.imported-nvfp4.initial-main-pr` | First approved same-repository MAIN PR head | One linked Krea 2 Turbo NVFP4 image, 256×256, count 1, step 1, seed 42 |
| `ci.windows.imported-nvfp4.main-push` | Approved resulting MAIN push | The same one-image mandatory smoke at the actual MAIN commit |

**Proposed combined total: 38 FLUX3 + 2 required CI = 40; hard ceiling 48;
8 unallocated and NOT automatically usable** (17 MLX + 23 Candle including CI).
These Krea outputs do not substitute for any FLUX3 product, resource or semantic
acceptance obligation. These two CI entries are separate
from the 38 FLUX3 permit IDs, but are included in the same global pre-run
accounting/admission. They do not consume the former ten spare slots implicitly.
This proposal does not initialize or amend an approved FLUX3 ledger. The
coordinator records each actual GitHub run ID, run attempt, event, source/checkout
SHA, inference pin, runner identity and the one-call source proof; no CI ledger
service/framework is added. A failed/canceled admitted CI attempt must not be
silently refunded or retried. **Every additional qualifying PR-head update,
workflow/job rerun or qualifying MAIN push requires a new explicit count and
admission BEFORE pushing/rerunning.** If the user declines the additional two,
retain the hold and obtain a newly approved smaller campaign or other explicit
user decision; never disable mandatory checks or start an unbudgeted attempt.

### Exact existing automatic CI behavior

- `.github/workflows/windows-candle.yml` watches `push: main` and
  `pull_request: main` with Candle path filters including `crates/**`,
  `apps/rust-api/**`, Cargo files and model manifests. Its
  `imported-nvfp4-worker-smoke` job needs `candle-worker` success and permits a push
  or same-repository PR. It has no job matrix, seed/rung expansion, warmup loop or
  automatic retry step. It invokes only
  `cargo test -p sceneworks-worker --features backend-candle --release imported_nvfp4_worker_gpu_smoke -- --ignored --nocapture --test-threads=1`.
- That ignored test requires the real Comfy-Org Krea NVFP4 checkpoint at the
  frozen `952f49d49653cb42e7d6cf7cbfad74738073ec7d` snapshot and the cached
  `SceneWorks/krea-2-turbo-mlx` bf16 companions. It performs one linked import
  followed by exactly one `run_image_generate_job` with the table's request.
  The checkpoint-plan route creates a work list from `request.count` (one),
  invokes one `generate_one`, and passes a `FnOnce` through
  `memory_strategy::generate_with_scope` to the actual `Generator.generate`.
  Import/load and kernel operations are not additional image generations.
- This job selects `[self-hosted, Windows, X64, cuda, real-weights]`, masks
  `CUDA_VISIBLE_DEVICES=1`, and serializes its real-weight group using
  `windows-candle-gpu-real-weights` with cancellation disabled. **The mask and
  labels are not an owned PID/GPU-UUID observation.** There is no terminal
  single-use permit or actual GPU1 UUID/foreign-compute guard in this existing
  smoke. The concurrency group serializes participating real-weight jobs, not
  every ordinary Candle job: ordinary PR/push `candle-worker` uses a per-run
  group and can otherwise share GPU1. The parent must establish the real runner,
  GPU1 identity and an exclusive admitted execution window before delivery.
- Normal Candle unit tests do not pass `--ignored`; their weightless CUDA-quant
  kernel tests can touch GPU1 but do not generate model images. Other five-rung,
  LTX, YuE2 and Iris real-weight campaigns in this workflow require explicit
  `workflow_dispatch` booleans (default false), not an ordinary PR/push.
- `.github/workflows/macos-mlx.yml` runs hosted `macos-checks` for MAIN and
  feature PRs. Its self-hosted `nax-worker` is enabled for relevant same-repository
  **MAIN-base** PRs and relevant MAIN pushes, never feature-base PRs or merge
  groups. Normal NAX CI builds the worker, runs only `--test nax_guard`, and dumps
  capabilities. The guard exercises random tensors (three attention shapes,
  two layouts, three dtypes), not model weights or `Generator.generate`:
  **zero image-generation attempts**. Real calibration/five-rung/provisioning
  paths require explicit dispatch inputs and remain unauthorized here.

### Mandatory secondary-Mac routing and parent-only commands

Use the **existing** documented `[Mac2]` routing, not a workflow/policy change:
put the operator-required exact `[Mac2]` prefix in the **initial PR title**, and also at the
start of the **actual MAIN merge-commit subject**. The workflow tests
`startsWith(PR.title, '[Mac2]')` / `startsWith(head_commit.message, '[Mac2]')` and
selects `[self-hosted, macOS, ARM64, rw-mage]`; a body mention or later label does
not suffice. Without the marker its bare `nax` pool may reach the forbidden
primary `nax-macos`. Dispatch calibration routing uses `nax,weights` instead and
is not covered by this marker. **Do not use a plain `macos-mlx.yml`
`workflow_dispatch` as a hosted-only metadata rescue:** even with calibration and
five-rung inputs false, its non-PR job condition enables `nax-worker` and the
runner expression falls back to bare `nax`. A title/body `[Mac2]` marker does NOT
route a `workflow_dispatch`. No such **unguarded/default** manual dispatch is authorized here. The normal
**feature-base PR hosted producer** is the ordinary collection route. The one
narrow c8 metadata-bootstrap profile described below was separately approved
because the genuine local matrix gate cannot bootstrap without combined MLX
facts; it is not a hook exception, a model campaign, or a NAX verdict. Verify the live runner-label/physical identity is
actually **secondary `nax-macos-2` before execution**; labels alone are not proof.
See the existing routing explanation in `docs/rust-mlx-build.md`.

### Narrow hosted-only c8 metadata bootstrap

One hosted build/lint/unit/facts capture is separately approved to resolve the
real `matching-platform production model mappings differ` bootstrap gate. The
new `workflow_dispatch.hosted_only` boolean defaults to **false**. Only an
explicit dispatch with it true excludes the **entire NAX job before runner
assignment**; normal PRs, pushes and default manual routing are unchanged. The
early hosted guard rejects any true/unexpected value for
`run_memory_calibration`, `run_five_rung_reference`, `provision_qwen_snapshot` or
`provision_z_image_snapshot`. No self-hosted machine, weights, calibration,
materialization or model-image generation is admitted by this profile. Its
hosted `macos-26` build/test and `always()` facts/upload remain normal; a collected
artifact is **not** a NAX verdict, a green preflight stamp or campaign acceptance.

The source-only bootstrap may retain genuine old MLX facts and old SW-generated
snapshots with their original provenance, explicitly stale pending that producer.
For this bootstrap, the stale inputs are the two original MLX media/runtime
files under `config/engine-capabilities/`, and the exact ce4 MAIN snapshots
`config/backend-capabilities/matrix.json`,
`config/manifests/builtin.preview-support.jsonc`, and
`docs/generated/memory-matrix.{json,md}`. Their old source hashes/revision labels
must not be relabeled as c8 or current. The local Candle media/runtime/audio files
are actual compiled c8 producer outputs, not a cross-family JSON union.
It must preserve the actual c8 Cargo edges, authoritative declarations and both
merge parents. Commit/push the plain no-PR
bootstrap ref through **normal hooks**. No-PR avoids the PR preflight-stamp
requirement, but the actual pre-push hook still runs applicable tier, neither and
Candle checks; it is not an exemption from those checks. If any normal hook
blocks, stop and report it—never use skip flags, `--no-verify`, alternate API file
writes or another session to push.

The existing workflow must be present on default main, but the operator must
explicitly select the **patched bootstrap ref** containing this input and NAX
exclusion. Verify its remote YAML blob against that exact local commit before
dispatch. Send typed JSON `hosted_only:true` and all four conflicting flags false.
If the server rejects the ref/input schema, stop; never omit the flag or fall
back to default-main dispatch. For the one approved run:

```sh
BOOTSTRAP_REF='bootstrap/sc-26243-hosted-facts-c8'
printf '%s\n' '{"hosted_only":true,"run_memory_calibration":false,"run_five_rung_reference":false,"provision_qwen_snapshot":false,"provision_z_image_snapshot":false}' | gh workflow run macos-mlx.yml --repo SceneWorks/SceneWorks --ref "$BOOTSTRAP_REF" --json
# Record the returned URL/id and exact checkout SHA. Verify actual jobs:
# NAX must be skipped/unassigned; only hosted jobs may execute. No rerun without approval.
# Follow this ONE run, uncapped, at the approved interval:
gh run watch "$RUN_ID" --repo SceneWorks/SceneWorks --interval 120 --exit-status
```

Collect only its genuine `backend-capability-facts-mlx` artifact into a new empty
external directory. Verify actual source/run/pin provenance, c8 stamps and both
FLUX3+MOSS media/runtime entries; adopt raw producer JSON, never hand-union or
relabel old snapshots. Then rerun the SW-source generators and genuine preflight
on a clean committed head after the outstanding bundle-budget user decision.
A failed hosted check may still produce valid evidence through `always()`; do
not describe that metadata-bootstrap workflow as green certification.

The following are parent-only **MAIN delivery** examples for after complete
plan/GPU1 approval, not instructions to run them now:

```sh
# Inspect live label placement; admit only the actual secondary Mac and GPU1 runner.
gh api repos/SceneWorks/SceneWorks/actions/runners --paginate --jq '.runners[] | {id,name,status,busy,labels:[.labels[].name]}'

# The initial title must already contain the routing marker.
gh pr create --base main --head '<parent-approved-final-head>' --title '[Mac2] FLUX3 terminal integration' --body-file '<approved-body-file>'

# Record PR/source identity before admitting the first one-image CI attempt.
gh pr view '<PR>' --json number,title,baseRefName,headRefName,headRefOid,url

# Only after reviewed green checks and explicit MAIN-push attempt admission.
# Do not rely on GitHub's default merge subject or a marker only in the body.
gh pr merge '<PR>' --merge --subject '[Mac2] FLUX3 terminal integration'

# Collect actual run/attempt/checkout and runner receipts, not assumed attempt counts.
RUN_ID='<actual-run-id>'
gh api "repos/SceneWorks/SceneWorks/actions/runs/$RUN_ID" --jq '{id,event,head_sha,run_attempt,conclusion,html_url}'
gh api "repos/SceneWorks/SceneWorks/actions/runs/$RUN_ID/jobs" --jq '.jobs[] | {id,name,runner_id,runner_name,status,conclusion}'
```

The parent must reconcile both proposed CI attempts and the 38 FLUX3 ledger
reservations against the approved global plan; normal weights-free CI must never
be confused with the real-weight job above. No actual CI image attempt or
production campaign execution is asserted by this document.

## Reports and evidence

Each run retains a report under `outputRoot/terminal-<nonce>/report.json` including
all plan cells: unselected cases remain NOT_RUN/BLOCKED. Selected cases retain
actual API requests/digests, job/progress events, package identities, ordered
source hashes, and separate per-job PNG/sidecar copies. Finished-image proof
requires GET-library, raw sidecar, job asset, generation-set recipe, served PNG
hash, full PNG CRC/IDAT/IEND/decoded dimensions, and worker-device identity to
agree. A queued job or provider-only image is not evidence.

Checkpoint proof retains actual safetensors format/identity, factor/optimizer/RNG
state and content hashes, checkpoint-compatibility refusal, final advanced factor
manifest, registered exported adapter, and fresh ordinary-generation image.
Missing or unrelated errors do not pass negative tests. Before first training and
again after resume, the same sealed binary runs the model-less early CLI
`--flux3-training-config-witness <owned-plan.json>` while this session's GPU worker
is paused, with explicit CVD=-1. It returns the exact Rust-mapped Debug config SHA,
input-file-bytes SHA, expected factor shapes/active workflow keys and RNG policy;
no weights, tensors, registry, provider, API or GPU initialization occurs. Neither
backend is typed unsupported. This is not a JS-emulated config hash.

The driver binds that witness to the actual persisted native plan/config, dataset
paths/order/captions/content, dense DIT/TE/AE inventories and immutable native
identity in both checkpoints. Accumulation is fixed at one, pending gradients
must be empty, sampler/update/micro-step and RNG seed/next-step lineage must agree,
active factors and optimizer moments must advance, Candle's actual `optim.step`
metadata must match the update cursor (never infer the counter from it), and
initial/final optimizer configuration must remain identical. Inactive workflow factors
must remain unchanged. Native MLX may legitimately omit optimizer moments for
inactive captioned image_cond factors; no fake zero moments are invented. Synthetic
Node fixtures test these checks but are **not** native Rust compile/CLI evidence;
that CPU verification remains a separate admitted step.

Images finish `AWAITING_VISUAL_INSPECTION`, **not PASSED**, with
`executionStatus:PASSED` separate from `semanticAcceptance:PENDING`. Per-case
`visualObligations` spell out quality/reference identity/placement/moves/removals
and adapter/training semantics. Required source PNG views are retained once under
`inputs/`, with exact uploaded content hashes; training also retains target/reference
bindings and caption hashes. A human or approved oracle must inspect these exact
retained input/output hashes. A valid, nonblank PNG or different adapter-on pixels
is only a binary/mechanism observation, not semantic image acceptance. The driver
does not invent or accept an unbound visual attestation; parent reconciliation
must bind any human/oracle evidence to the actual input/output hashes. The driver
intentionally cannot declare the whole epic PASS from one backend batch. The
parent reconciles per-phase receipts, visual-inspection evidence, positive
installation/auth/CI receipts, and the separate single resource campaign. None of
this document is a measurement, a regenerated artifact fact, or an executed case.
