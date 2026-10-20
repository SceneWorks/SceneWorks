# sc-26243 installed-product terminal driver

This is executable acceptance **infrastructure**, not an acceptance receipt. No real
case is implied to have run. The parent coordinator owns actual resource approval,
paired MAIN integration, the final inference pin, campaigns, review, and merges.
Resource anchors/conversion observations are separate from saved-image proof.

## Safe planning and CPU tests

From the SceneWorks checkout, with Node 20:

```sh
node scripts/flux3-image-klein-terminal-driver.mjs plan
CUDA_VISIBLE_DEVICES=-1 node --test scripts/flux3-image-klein-terminal.test.mjs scripts/flux3-image-klein-terminal-budget.test.mjs scripts/flux3-image-klein-app-lifecycle.test.mjs scripts/flux3-image-klein-candle-app-lifecycle.test.mjs
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
package anchors), and base `resource-pup-edit-10.bf16` once per backend (two
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
  `inferenceRepository`, and `inferenceMainTipSha`. Git ancestry is checked against
  locally fetched `origin/main` in **both** repositories. A feature-only SHA or a
  field merely named MAIN is insufficient. The driver never fetches or merges.
- Absolute `binary`, its `binarySha256`, and `binarySourceSha === sourceSha`.
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
that the unverified native harness already enforces it. Return the latest audited
receipt per backend to the coordinator, who aggregates unique IDs against the same
global plan and total <=38 approved / <=48 hard ceiling. Do not add cumulative
snapshots together as though their repeated event histories were new runs.

Candle additionally requires `physicalGpuId:"1"`, `cudaVisibleDevices:"1"`,
`logicalCudaOrdinal:0`, admitted `gpuUuid`, and `runner`. The controller checks
physical GPU1 against masked CUDA0 UUID, rejects active Runner.Worker and foreign
pure-compute processes, monitors the owned worker PID/device, and never selects
user-reserved GPU0. MLX requires physical Apple Silicon, `runner:"nax-macos-2"`,
`runnerTarget:"secondary"`, `runnerLabel:"rw-mage"`, and exact admitted `hostName`.
`RUNNER_NAME` must be `nax-macos-2`; `rw-mage` is a label, not the runner name. The
primary Mac is forbidden.

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
  exact pinned snapshot, and escaping the admitted hub is refused.
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
Missing or unrelated errors do not pass negative tests.

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
