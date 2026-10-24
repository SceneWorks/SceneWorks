# MOSS edit terminal campaign (sc-26757)

`scripts/moss-edit-terminal.mjs` is the distinct product harness for this epic.
It uploads original RGB/tensor inputs to an authorized project, drives the actual
typed REST API and existing Streamable HTTP `/mcp`, retains actual native jobs,
progress, provider reports, assets, recipes, workflow resolution and result bytes.
The API owns persistence; the harness has no alternate generator or job store.
The historical sc-26325/GPU0 entrypoints are unrelated and must never be invoked.

This source alone is not execution evidence. Implementation/runtime verification,
the final merged contract refresh, independent review, final paired source/pin,
the single campaign are still required. No source fixes
are allowed while a candidate is frozen. A source fix invalidates that candidate.

## Prepare and freeze

Current unexecuted protocol is **v5**, retaining v3's exact **50 renders,
58 refusal calls and ten required external evidence items**. v3/v4 locks/receipts
cannot certify this version. v5 separates the shared immutable candidate from
backend-local deployment paths and validates the Editor's actual scratch upload.

Use Node 22 or newer. No new packages are required. Preparation writes only a
fresh directory outside both repositories and never contacts a service/device:

```sh
node scripts/moss-edit-terminal.mjs prepare /ABSOLUTE/quality-inputs-v1/inputs.json /ABSOLUTE/FRESH-CANDIDATE
```

This copies the four byte-identical original PNGs and prepares small nontrivial
FP32 embeddings `[2,3,4096]`, binary masks `[2,3]` with different valid lengths,
and exact FP32 noise `[4,128,4,4]`. Input identities retain NPY file SHA, tensor
content SHA and decoded original RGB SHA separately. Values are fixed by source.
No values, prompts, rubrics or seeds may be selected after observing results.

Review `cases.json` and complete `launch.example.json` into a private launcher.
Original protocol v2 allocated **42 actual render executions**: 8 primary quality,
4 advanced tensor/latent plus 4 actual replay, 2 advanced cancel plus 2 same-input
actual retries, and 22 bounded image lifecycle/cache/reference/ordinary/DMD/
isolation executions. Each backend has 21. Four advanced
replays alternate actual same-project duplicate and real UI portable workflow
reacquisition. Quality allocation includes Full Studio, Simple Studio, Editor,
REST, MCP and backend-changing history/shared replay. Original quality settings
are unchanged: Base/SFT × MLX/CUDA × single/four, seed173, 30 steps, CFG4,
inference shift5, edge512, cachetrue. Exact poster lettering is excluded before
rendering. These generated sources do not establish camera-photo coverage.

Freeze `cacheComparisonPolicy.metric:"rgb8-mae/v1"`, a normalized [0,1] MAE
tolerance, and the executing `scripts/lib/moss-edit-terminal.mjs` SHA before
rendering. The harness computes RGB8 mean absolute error over the exact retained
cached/uncached output files with identical requests except cache policy.
Supply every predeclared guard request,
the exact expected HTTP status/message fragments and the source-specific actual
no-provider-load probe. JSON cannot represent NaN: a nonfinite guard must use
an inert NPY containing nonfinite values or the actual supported HTTP lexical
rejection; do not accidentally serialize NaN into null and claim native coverage.
Original v2 guards numbered 27 per backend, **54 actual refusal calls**.
Protocol v3 preserves them and adds latent-output + enabled inline-upscale
refusals for RealESRGAN and SeedVR2 on each backend: **58 calls, 29/backend**.
Each new exact request must return HTTP 400 with the actionable choice, before
provider load or device allocation. Images with inline upscale and latent output
with upscale disabled remain valid; bind finalized Story56 discriminator results
to final source carry-forward.
Host/device admission errors must be bounded refusals before provider load;
never provoke a real OOM. Corrupt/missing package tests require isolated owned
fixture/configuration supplied by the coordinator, never editing shared models.

Required private launcher identities include clean stationary SceneWorks and
inference checkouts, the final inference-main ref (refreshed by the coordinator),
actual API/worker binaries, web package inventory, verified weights/package
inventory, actual toolchains and source-bound numerical carry-forward receipts.
Set `localBackend` to `mlx` or `cuda`; `files` names the actual local files.
`artifactInventories.mlx` and `.cuda` point to source-bound build inventories:
`{backend,sources:{sceneworks:{head,tree},inference:{head,tree}},files:{apiBinary,
workerBinary,webPackage,packages,toolchain}}`, where every file identity is exactly
`{byteLength,sha256}`. Both inventories must bind the same clean paired source
identities. They contain identities, not platform paths; retain actual source
build/toolchain observations in the existing external provenance evidence.
No SSH or remote process enumeration is performed.
The coordinator owns server/worker launch and backend routing. Each backend
requires its actual isolated project ID, API URL, owned `workerId` and
`workerGpuId` (CUDA
must be `"1"`). JobsStore's first queue backend label is distinct from native
`execution_device`: pin progress can retain `workerGpuId`, while generation uses
`mlx` or `candle`. The native receipt must still prove `mlx` or `cuda`, and GPU1
admission independently binds UUID/PCI/logical0. Authentication is read from
the declared environment variable and sent as `X-SceneWorks-Token`; never put a
credential in JSON, an API URL or argv. The existing product `/mcp` is mounted
in the API; it has no separate stdio binary. Initialize/session/version headers,
SSE progress, tool invocation and result links use that existing transport.

After merged54/55/56 source review, final integration review and the coordinator's
frozen-protocol approval have actual SHA identities:

```sh
node scripts/moss-edit-terminal.mjs freeze /ABSOLUTE/launch.json
```

The lock verifies clean source/tree IDs, the inference-main ancestry and all three
SceneWorks manifest pins plus Cargo.lock, all files and exact case/input identities. Preparation
and freeze do not execute the campaign. The launcher contains private paths;
keep the complete candidate directory private (`0700`, evidence files `0600`).
Only `verify` stdout is a portable summary; raw job/recipe data may contain
private source paths. Raw MCP media tickets and inline image payloads are not
retained. No public export is implied.

`protocol.lock.json` contains shared source/protocol/input/plan/guard identities,
both backend artifact inventories and the existing approval/report provenance;
it contains no execution paths, API URLs or project IDs. `freeze` also writes
the first local `bindings/BACKEND.json`. Transfer the unchanged lock, cases,
original manifest and all original input bytes to the other native job, retaining
prior backend evidence. Supply its own local launcher paths/files/endpoint and:

```sh
node scripts/moss-edit-terminal.mjs bind /ABSOLUTE/LOCAL-launch.json
```

`bind` rechecks actual clean paired checkouts, every pin, original input bytes,
the currently executing harness/library and all local binary/package/toolchain
hashes against the shared lock. It writes a create-only local binding containing
the shared lock SHA, actual paths, API/project/worker identity and verified files.
Existing bindings must never be overwritten; relocating existing evidence is not
permission to rebase its binding. A new native backend uses its separate binding.
Keep source files and manifests clean and unchanged while building the deployment.
Every submission/row/refusal binds the shared lock and its backend-binding SHA.
Final collection retains both backend bindings and full evidence; `verify`
uses each binding's actual project and worker identity, not a summary PASS.
No new approval/signature gate or hardware ownership mechanism is introduced.

## Device identity and execution

The runner validates a candidate-bound device receipt before uploads/submissions.
Its admission JSON contains `version`, `lockSha256`, `backend`, and
`device:{uuid,pciOrRegistryId,name}`. MLX also requires
`device.platform:"apple-silicon"`. CUDA additionally requires actual
`physicalIndex:1`, `logicalIndex:0`, `cudaVisibleDevices` equal to that device UUID,
`sceneWorksGpuId:"1"`, and `postCompile:true`. GPU0 is never admitted.
Use existing Actions runner configuration and scheduling; no exclusive GPU lease,
host-wide process census, cross-repository resource lock, or agent-managed
hardware token is required before submission or execution. Freeze the final
clean stationary paired sources/pin before submitting the single campaign,
even when its configured runner is busy. Candidate/backend attempt locks below
prevent duplicate campaign submissions and preserve evidence.
The individual CLI harness does not inspect foreign argv/env, signal processes, launch a server,
install/download weights, or manipulate a device.

```sh
node scripts/moss-edit-terminal.mjs upload /ABSOLUTE/launch.json mlx /ABSOLUTE/device-admission.json
node scripts/moss-edit-terminal.mjs ui-task /ABSOLUTE/launch.json quality-mlx-base-single
node scripts/moss-edit-terminal.mjs run /ABSOLUTE/launch.json quality-mlx-base-single /ABSOLUTE/device-admission.json /ABSOLUTE/ui-capture/capture.json
node scripts/moss-edit-terminal.mjs run /ABSOLUTE/launch.json quality-mlx-base-four /ABSOLUTE/device-admission.json
node scripts/moss-edit-terminal.mjs guard /ABSOLUTE/launch.json edit-dmd mlx /ABSOLUTE/device-admission.json
```

Follow manifest order **within each backend**. There is one exclusive attempt
directory per row/upload/guard; no automatic retries or overwriting evidence.
Cancellation waits for actual `running`/`generating` progress with source-owned
`Image ... — step N/M.` or `Batch of ... — step N/M.` and 0<N<M before
requesting cancel and requires no published outputs/reports. Reentry is a real
retry of that canceled job, after the coordinator retains actual owned cleanup
proof in `external/cleanup-before-ROW_ID.json` containing candidate
`lockSha256`, `canceledJobId`, `verdict:"PASS"`, observer and hashed evidence
`files`. Reference changes, order/duplicates and uncached comparison follow.
Each backend has both an image cancellation/retry and an advanced
embeddings/masks/noise/Latents cancellation/retry. The latter preserves
B2/count2, seed:null, all tensors/reference order/output choice and the identical
request through the real retry endpoint.
Ordinary Base/SFT and both SFT DMD precisions run before the final return-to-edit
row. Source/identities are checked again after execution. MCP timeout or network
failure does not prove cancellation. Stop, inspect the authoritative job and
document reconciliation before permitting another dispatch. A failed row remains
visible; it must not be silently rerun under the same accepted candidate.

Every possible submission reserves an immutable candidate/backend attempt and
exclusive dispatch lock. An unexpected enqueue retains its full response/job ID;
network/refusal mismatches retain the original error. An unresolved attempt
prevents later uploads, renders, UI tasks or guards on that backend.

After actual authoritative reconciliation use
`reconcile LAUNCH_JSON BACKEND ATTEMPT_ID PROOF_JSON`. The proof binds version,
lock SHA, attempt SHA, backend/coordinator/date, `state:"NO_ACTIVE_OWNED_JOB"`,
all known terminal `jobs` with completed publication and hashed `rawEvidence`.
A lost response/job ID also requires
`authoritativeSearch:{requestSha256,since,complete:true}` covering the exact
request and start time. This command performs no cancel, retry or inference.
It permits safe later dispatch while preserving failure permanently; `verify`
still FAILs that candidate. An idle device is not authoritative reconciliation.

## Actual UI work and evidence contract

`ui-task` reserves an exclusive unresolved UI intent **before** emitting the exact
authored request and concrete observation tasks. It requires predecessor rows
complete. `run` adopts that same reserved request; it cannot create a second UI
attempt or change its request. The actual UI job must use the frozen
backend and device identity. The intent
stays `AWAITING_REAL_UI` until the coordinator operates the real supported UI.
`run` adoption requires actual observed network POST request/response, running
job snapshots, before/after PNG screenshots and action observations. A manual
job ID alone fails. The server job must independently retain the exact authored
model, controls, seed/noise, ordered inputs and effective native backend.

Capture JSON fields: `version,rowId,lockSha256,bindingSha256,observer,capturedAt`,
`network:{method:"POST",path:"/api/v1/image/jobs",observedAt,request,response:{id}}`,
`screenshots:[{phase:"before"|"after",file,sha256}]`, `progress:[JobSnapshot]`,
`actions:[observed action]`. Files are relative to the capture JSON directory.
Portable replay adds `workflow:{file,sha256,inspection,originRowId,originJobId,originSampleIndex,reacquired:[{ordinal,inputId,assetId,...observations}]}`
and retains the actual image PNG with its embedded workflow, or the actual tensor
`.sceneworks-workflow.json` companion, plus real ordered input
reacquisition. No generated output may replace an original reference/tensor.
Original exported sanitized reports/sample indexes remain intact as display
provenance and are checked against the original job/sample. Reports/sample
indexes must be absent from the **new request and canonical job payload**.
Before/after screenshots must
show actual input selection/options and the actual job/result respectively.

Primary CUDA Base four must be an actual portable replay of prior MLX Base four.
Primary CUDA SFT single must show actual Editor history hydration of MLX SFT
single. Its `history` contains `originJobId,originAssetId,originSampleIndex:0,
hydrationObserved:true,selectedRecipe` plus actual observations. The original
job/asset/recipe must remain available through the target API after the real
cross-backend state handoff; a fabricated imported history row cannot pass.

The Editor stages its working raster through a real project upload. Retain
`sourceUpload:{inputId:"B",file,sha256,observedAt,network:{method:"POST",
path:"/api/v1/projects/PROJECT/assets",request:{fileSha256},response:{id}}}`.
The harness authenticates to the target API and reads that actual asset/media
and original history job/asset back. The observed upload and served bytes must
match; the scratch must preserve the frozen original's exact decoded RGB and
geometry with fully opaque alpha. It must be a fresh project image uploaded
after the reserved UI intent and before the observed job POST. Original and
scratch PNG encoding hashes may differ; pixels cannot. Only this allocated
Editor first source slot can bind a new upload ID to original B. Other UI/REST/
MCP rows keep exact original IDs, and tensor identities remain exact.

`request.json` for Editor retains its actual POST with actual scratch ID;
`semanticRequestSha256` separately binds the frozen original-input request.
`source-binding.json` hashes the retained upload, authenticated asset/media and
history job/asset read-backs and is revalidated by `verify`. The canonical job,
POST, history IDs, recipes and native reports are never rewritten. Wrong pixels,
alpha, geometry, project, slot order, chronology, history or evidence fail.
The cached/uncached comparison uses the same frozen source semantics only after
revalidating this actual Editor source proof; it retains actual request/output
hashes in the metric record.
The two advanced workflow replays must exercise the real UI's supplied
embeddings/masks, negative mode, noise, Latents, count, max sequence and distinct
training/inference shifts; these rows provide advanced UI execution without a
second advanced matrix. Record install/select, defaults and supported custom
controls in the corresponding external UI evidence. The harness does not infer
product reachability from serializers, mocks or direct native calls.

## Required final evidence

Every row retains request, submission, canonical terminal job, progress, actual
asset/recipe/workflow, provider report and output file SHA. Native report checks
bind the exact published adapter/config, 252 projections, device, effective
scalars, original RGB slot order, no-crop aligned dimensions, output kind/shape,
caller-noise sample digests/seed:null and exact F32 embedding+mask row digests.
Each latent file is single-sample F32 NCHW `[1,128,H/16,W/16]`, finite real NPY.
Every successful advanced row retrieves results through both REST and actual
MCP ticket links and requires the same bytes. The MCP JSON asset summary binds
media by canonical asset ID; accompanying workflow links are downloaded and
validated separately. Resource-block ordering is never sample ordering.
Ordinary regression artifacts remain real
jobs; final source-bound regression/isolation observations are also required.

For each row add `memory.json` from an **actual** coordinator-owned sampler:
`{version,jobId,samplerSha256,device:{uuid},samples:[{at,phase,usedBytes}]}` with
baseline/active/drain phases. The report labels its peak as sampled whole-device
usage. It never calls an admission estimate or process allocator peak a measured
whole-device peak. Keep raw sampling source/tool identities, actual durations,
load/conditioning/denoise/decode timestamps and admission estimate/credit
separately in external evidence; null API metrics do not prove timing coverage.

Each quality row additionally needs `quality-review.json` with exact ordered
`outputSha256`, reviewer/date and all frozen `anchors:[{name,verdict,observation}]`.
Any anchor FAIL fails the candidate. Missing review is AWAITING. Retain full
original/output pairs and at least three detail crops in `visual-pairs.json`
(`version,lockSha256,rowId,inputPairs,detailCrops`, entries `{file,sha256}`;
each ordered inputPair also binds `inputId,inputSha256,outputSha256`).
PNG production alone cannot pass quality.

After both cached/uncached rows complete, run
`compare-cache LAUNCH_JSON cachefalse-BACKEND`. It writes `cache-comparison.json`
with ordered `{rowId,jobId,requestSha256,outputSha256}` inputs, decoded RGB pixel
SHA/geometry, executing library SHA, policy SHA and computed normalized MAE.
`verify` recomputes that exact measurement over retained bytes and rejects
changed inputs, copied results, another implementation or failed tolerance.
Each refusal requires
`no-provider-load.json` with candidate/case/backend/request SHA, observer, actual
source-bound before/after allocation-capable boundary/provider-load/provider-run
observations and raw append-integrity journal files. These are conservative model
entry observations, not individual device allocator call counts.
The actual nonchanged counters must come from the frozen source-specific probe.

Every `requiredExternal` name in cases.json needs `external/NAME.json` containing
`{version,lockSha256,verdict,observer,observations,rowIds,files:[{file,sha256}]}`.
These are actual observed lifecycle/UI, numerical source-carry-forward,
publication/cleanup and memory evidence; a prose assertion without retained
observations/files does not satisfy the contract. Retain actual unresolved-input
workflow refusal, retry semantics (canceled/error original), cleanup-before-reentry,
asset visibility during pending/cancel/error and cache identity transition proof.
Root owns these explicit real product/device tasks and integrated review.

```sh
node --test scripts/moss-edit-terminal.test.mjs
node scripts/moss-edit-terminal.mjs verify /ABSOLUTE/launch.json
```

Pure tests use source-contract fixtures explicitly labeled as non-runtime
evidence, deny live fetch, and use bounded private temporary files. The
prepare/freeze test uses tiny inert Git fixture data in TMPDIR plus read-only
Git inspection and a fixture index; it runs no git commit or remote operation.
These tests discriminate tampering, omitted anchors/rows, wrong physical device,
wrong candidates, wrong tensor/noise digests, receipt import, pending publication and
bad canceled output. They do not claim model quality or real product execution.
`verify` returns PASS only with every required row/evidence item present and
passing, FAIL for detected failures, AWAITING for missing evidence (exit2).

## Protocol v3 Story56 contract synchronization

All existing42 rows stay unchanged. Add8 renders (4/backend), for **50 renders,
25/backend**: seeded text B2/count2 Images and Latents originals, each followed by
one actual row-two viewer Keep-seed recipe replay in Full Studio. MLX Images/CUDA
Latents use Base; MLX Latents/CUDA Images use SFT. These are bounded two-step
64x64/CFG4 protocol checks, not new quality cells. All original eight quality
cells/inputs/rubrics and tensors/noise/transport/cache/lifecycle/regression/
publication/cancellation/memory requirements stay intact.

Select original output index2 (conditioning row two, effective seed175), retaining
its actual asset ID/recipe and row-two displayed prompt/sample provenance. Actual
Use this recipe with Keep seed must reopen the whole B2/count2 batch with main
prompt=row0, both distinct prompts and negative rows, original ordered references,
base seed173 and unchanged output kind. Capture the history fields from ui-task,
existing before/after screenshots and actual submitted network request. A REST
duplicate does not exercise this UI path. Reports and sample indexes stay absent
from the new request. Existing caller-noise rows remain seed:null and keep exact
caller-tensor digests.

On each backend's latent replay UI, retain observations/screenshots and an actual
network trace while selecting each inline-upscale engine: Generate must disable,
the actionable Disable upscale or select images message must appear, and no POST
may occur. Disabling upscale restores latent submission; selecting Images with
enabled upscale clears the incompatibility. Restore the exact frozen latent
request before submitting. These no-dispatch UI interactions are not extra
renders or REST refusal calls. Bind traces/screenshots to advanced-controls-ui
with the affected new row IDs. API refusals remain separate exact guards with
unchanged no-provider-load proof rules.

Version v3 source must be applied, reconciled and validated before any freeze.
v2 receipts/locks do not certify v3. Preparation proves no runtime/UI/quality/
capacity acceptance. Coordinator owns final source acceptance and campaign.

## Protocol v5 native Actions orchestration

`.github/workflows/moss-edit-terminal.yml` is manual and final-campaign only.
Dispatch only after 55/56/57 and the feature code have merged, the integrated
requirements review and inference source review have finished, inference is on
main, and the clean stationary SceneWorks feature candidate has the permanent
inference pin. SceneWorks final source review and its main merge follow the
accepted campaign and final paired validation.
`source_sha`, `inference_sha`, and `candidate_id` must match the reviewed input
artifact. Actions scheduling is sufficient; there is no GPU ownership census,
lease, cross-repository lock, or agent token. Never dispatch this campaign to
validate an intermediate implementation change.

Before this new workflow reaches main, use the already registered Memory catalog
dispatcher at the exact stationary feature commit. Its MOSS profile calls this
reusable workflow from the same commit and skips both catalog jobs. The dispatcher
rejects unknown/malformed JSON fields and a `ref` differing from its actual SHA
before either native backend starts. Keep dispatch inputs in a private JSON file:

```json
{
  "profile": "moss-edit-terminal",
  "ref": "EXACT_SCENEWORKS_SHA40",
  "inference_sha": "EXACT_INFERENCE_MAIN_SHA40",
  "moss_terminal": "{\"candidate_id\":\"sc-26752-final\",\"bundle_run_id\":\"RUN_ID\",\"bundle_artifact_id\":\"ARTIFACT_ID\",\"bundle_manifest_sha256\":\"SHA256\",\"phase\":\"execute\"}"
}
```

```sh
gh workflow run memory-catalog-campaign.yml --repo SceneWorks/SceneWorks --ref EXACT_SCENEWORKS_SHA40 --json < /ABSOLUTE/moss-dispatch.json
```

For retained verification, use the same source/pin/bundle/candidate with
`phase:"verify"` and the exact `execution_run_id`, `execution_run_attempt`,
`review_run_id` and `review_artifact_id` inside `moss_terminal`. No main bootstrap
or catalog measurement run is required for this route.

Create the first input artifact without an earlier artifact or credential copy.
The prepared corpus is
`/Users/michael/.codex/epic-runs/sc-26752/quality-inputs-v1/inputs.json`; its four
original PNGs and source receipts retain the generated-source caveat, exact
prompts, seed173/settings and ordered B,L,P,G roles. Complete the existing
`campaign.json` contract below against final source, then run:

```sh
node scripts/moss-edit-terminal-actions.mjs bundle /ABSOLUTE/REVIEWED/campaign.json /Users/michael/.codex/epic-runs/sc-26752/quality-inputs-v1/inputs.json /Users/michael/.codex/epic-runs/sc-26752/FRESH-SEALED-BUNDLE
```

The command copies the exact original PNG/receipt bytes, numerical carry-forward
and declared guard fixtures, changes only their manifest paths to bundle-relative
paths, and writes the complete `bundle-inventory.json`. It prints that manifest's
SHA256. Declared numerical/fixture paths resolve beside `campaign.json`; no model
packages, caches or credentials are copied. Reusing a destination fails.

Using the same outer `profile`, `ref` and `inference_sha`, set `moss_terminal` to
this JSON string and dispatch the registered command above:

```json
{
  "candidate_id": "sc-26752-final",
  "bundle_manifest_sha256": "PRINTED_SHA256",
  "phase": "upload",
  "upload_kind": "bundle",
  "upload_path": "/Users/michael/.codex/epic-runs/sc-26752/FRESH-SEALED-BUNDLE"
}
```

The upload job requires runner `nax-macos`, the actual preparation Mac carrying
`rw-starvector`, and reads the exact sealed directory from that same filesystem.
It verifies all identities/hashes before uploading `moss-bundle-CANDIDATE-RUN-ATTEMPT`.
Read its real run/artifact IDs and use them for `phase:"execute"`. Upload is CPU
file preparation; it starts no native build, stack, browser or model.

After execution, the coordinator inspects actual pairs/crops and writes the real
quality/external review files. Seal that existing review directory separately:

```sh
node scripts/moss-edit-terminal-actions.mjs inventory-review /Users/michael/.codex/epic-runs/sc-26752/ACTUAL-REVIEWS SOURCE_SHA40 INFERENCE_SHA40 CANDIDATE
```

Dispatch the same upload phase with `upload_kind:"review"`, its directory and
printed review inventory SHA. The resulting later run/artifact IDs become
`review_run_id`/`review_artifact_id` for verify. Inventory/upload do not create a
verdict; retained verification still requires actual observer evidence for every
existing obligation.

The workflow runs these concrete phases:

1. `build-native` uses the existing `rw-starvector` ARM64 Mac and Windows
   `cuda,real-weights` labels. It checks out both exact sources as siblings,
   validates all manifest/Cargo.lock pin sites and main ancestry, builds the
   actual React web bundle and release embedded API/worker, and inventories
   both native binaries, web build, complete model packages, and toolchains.
   Windows reuses the existing MSVC 2022 BuildTools `vcvars64.bat`/NVCC host
   compiler setup and native `CUDA_COMPUTE_CAP=120`; MLX reuses Release libmlx
   fetch with its source-build fallback. Each job has its own cargo target and
   HF projection. Shared preprovisioned package files are read-only sources.
2. Each native build deliberately omits the declared edit adapter files from
   its private HF projection, starts its own authenticated source-built API and
   native worker, and drives the actual React acknowledgement/installer. It
   retains the observed download POST, actual jobs/progress, installed status,
   and screenshots. A preinstalled adapter cannot substitute for installation.
   Omitted adapter files require no preinstalled source copy. The final package
   SHA inventory is written only after actual installation
   and verification of every expected component. Installation may download the
   exact pinned package. Execution verifies the source-bound build inventory and
   copies its actual installed private projection into a fresh owned offline cache.
3. Both immutable native artifacts must exist before `execute-mlx`. That job
   creates the owned campaign project, prepares the original image/tensor
   inputs, resolves actual refusal fixture uploads/readbacks, retains installer
   captures, and freezes the shared path-independent candidate exactly once.
   The controller runs all 25 MLX rows and 29 allocated refusal cases through
   the existing CLI (26 malformed image-job HTTP refusals, the declared
   per-artifact host-budget import refusal, and actual device-budget/corrupt-adapter
   queued-worker failures; all remain one allocated call each). UI rows reserve one intent and call the real React
   Playwright driver; they do not use direct API mutations as UI evidence.
4. The controller stops only the processes it spawned, requires their exit,
   checkpoints every owned SQLite DB, checks integrity, and retains a complete
   file/hash manifest for the project/media/jobs state. CUDA imports these
   exact bytes and relocates only campaign-owned recent-project registry
   locations. It preserves project, asset, job, recipe, payload and result IDs.
   Actual target API readbacks must recover the original MLX history before
   CUDA binds the existing lock and runs its 25 rows/29 refusals. The original
   lock and the MLX backend binding are never rebased or overwritten.
5. `phase: verify` downloads one exact execute run/attempt and the actual
   external/visual review artifact. `verify-retained` checks all paired source,
   native artifact, input, request, output, UI, journal and review identities
   while allowing the read-only verifier's retained files to move. Original
   execution endpoint/project/worker identities and bindings remain mandatory.
   It never launches a stack, model or browser, freezes again, or resubmits.
   Missing/failed evidence returns nonzero; an execution job's success is not
   campaign acceptance. All eight individual visual anchors and all ten named
   external evidence obligations remain mandatory.

The only browser dependency is explicitly provisioned **Playwright 1.56.1** with
its pinned Chromium revision into each owned job directory, outside the source
checkout. Its package, browser manifest and actual executable SHA are included
in the native toolchain inventory and rechecked before execution. Chromium is
headless with GPU/WebGL disabled. No browser is installed or started by importing
any harness/controller module, or by the inert tests.

The controller CLI is `scripts/moss-edit-terminal-actions.mjs`:

```text
validate-bundle BUNDLE_ROOT INVENTORY_SHA SOURCE_SHA INFERENCE_SHA CANDIDATE_ID
bundle CAMPAIGN_JSON INPUT_MANIFEST FRESH_BUNDLE_ROOT
inventory-review REVIEW_ROOT SOURCE_SHA INFERENCE_SHA CANDIDATE_ID
validate-upload ROOT KIND INVENTORY_SHA SOURCE_SHA INFERENCE_SHA CANDIDATE_ID
configure BUNDLE_JSON SOURCE_ROOT INFERENCE_ROOT FRESH_ROOT BACKEND BUILD_ROOT
configure-env
build RUNTIME_JSON
arrange-builds DOWNLOAD_ROOT
provision-browser RUNTIME_JSON
execute RUNTIME_JSON
finalize RUNTIME_JSON
```

`configure-env` consumes the workflow's `MOSS_*` path/identity variables and first
checks the immutable input bundle. The artifact root contains `campaign.json`,
`bundle-inventory.json`, the original input manifest/files, numerical source
carry-forward artifact, and any explicit refusal fixture bytes. The inventory
has `files:[{file,byteLength,sha256}]`, sorted exactly as `filesUnder()` produces,
and covers every file except itself. Its SHA256 is a dispatch input. Unexpected,
missing, substituted or changed bytes fail before native build/execution.

`campaign.json` supplies:

- `sourceSha`, `inferenceSha`, `candidateId`, `inputManifest`,
  `numericalCarryForward:{file,sha256}`, and the existing approved report
  provenance fields `approvedBy`, `approvalEvidenceSha256`,
  `phase18ReviewSha256`, `finalSourceReviewSha256`.
- `backends.mlx` and `.cuda`: actual host-local read-only source `hfHome`, unused
  loopback `port`, complete `packageFiles:[{file,sha256}]`,
  `installAdapter:true`, and exact relative `editAdapterFiles` (both `adapter_config.json` and
  `adapter_model.safetensors` under the integrated manifest's exact pinned
  snapshot; the weight SHA must match that manifest). The controller
  copies verified package bytes to its own `ROOT/hf`; it never repairs or
  corrupts the shared source cache. All files needed for native resolution,
  including model config/tokenizer/decoder/DMD/edit components, must be listed.
- The reviewed `cacheTolerance` and **all 29** exact `guardRequests`, each with
  its actual request, expected HTTP status, actionable `errorIncludes`, and
  `noProviderLoadProbe:"source-bound-moss-boundaries/v1"`. The two latent
  inline-upscale requests remain exactly source-owned by `latentUpscaleGuard`.
  Guard names do not excuse a request that tests a different failure condition.
  The host slot is source-owned as described below; the device-budget slot uses
  `expectedStatus:201,expectedTerminal:"failed"` and exact actual memory-admission
  error text. The corrupt-adapter slot also uses actual201→failed worker preparation,
  the exact frozen-component identity error, and zero provider load/run entries.
  Other malformed image-job cases must refuse at HTTP and cannot
  substitute queued failures.
- `guardFixtures:[{alias,cases,project,file,sha256,filename,mime,type,corruptWith?}]`.
  `project` is `campaign` or `foreign`; foreign assets are actually uploaded to
  a second owned project and read back. `$fixture:ALIAS` resolves only to that
  actual uploaded ID; `$fixture-sha:ALIAS` resolves to its actual registered
  tensor SHA. `$campaignProject` resolves to the actual project. Required
  foreign-reference/tensor and corrupt/nonfinite/nonbinary tensor cases must
  name their actual fixture aliases. The setup retains uploaded/served bytes,
  asset metadata and SHA identities, all bound into the shared lock.

Missing-reference/tensor fixtures use `deleteAfterUpload:true`: upload and hash-read
the fresh campaign-owned asset, permanently purge that exact ID through the real
API, and require actual HTTP404 readback. The missing-input guard retains that
same real ID. No synthetic missing IDs or empty-input substitutes are accepted.

The product rejects nonfinite/nonbinary NPY during import. To test the image-job
boundary, `corruptWith:{file,sha256}` first uploads a valid owned tensor, then
replaces only that asset's media bytes inside the campaign-owned project with
frozen invalid bytes and retains an authenticated readback. Metadata and IDs
stay original. Source56 `tensor_artifact::read_data` calls `inspect` before
metadata equality, so a finite/binary error can be observed before any provider
entry. Do not invent a foreign/corrupt ID or accept a hash-mismatch response as
proof of a distinct finite/shape/mask requirement.

`training-shift-nonfinite` may instead contain an exact bounded `rawJson` string
with the actual HTTP lexical nonfinite spelling and no `request` object. The
controller resolves only quoted actual fixture/project aliases before freezing.
The harness sends those UTF-8 bytes with `application/json`, retains
`request.raw.json`, its byte SHA and the real refusal response. It never runs
`JSON.stringify(NaN)` and calls the resulting `null` a nonfinite test. This is
explicit HTTP lexical rejection, not native scalar execution.

`missing-adapter`/`corrupt-adapter` additionally declare `adapterFile`, one of
the exact owned edit files. The latter declares `corruptBytesBase64` and
`corruptSha256`. The controller retains actual missing/corrupt-state evidence,
performs the allocated real API refusal and restores the exact original bytes
in `finally`. It never changes a shared package or accepts a cached provider
run as refusal-before-load evidence.

The optional native journal is enabled only for the owned campaign worker.
Its source/binary/worker/device/candidate identity, actual startup/ready marker,
live heartbeat, job scope, monotonic counters and chained append hashes are
mandatory for refusal proof. `allocationCapableEntries` counts conservative
actual MOSS provider construction and native generate/edit calls. It is not an
allocator call count or an NVML memory delta. `boundedContextEntries` records
cache context entry before CUDA binding; `cacheDispatchEntries` records generic
cold/warm run dispatch before admission. Neither is called a model allocation.
Bounded startup/health/metrics contexts remain distinct as well. Actual
`provider_run` hooks sit immediately before the two MOSS generate/edit calls,
after all warm admission checks; `provider_load` sits at the admitted runtime
constructor. The device-budget and corrupt-adapter cases drain and restart only the owned idle
worker, retains actual exit/restart identity, observes its new ready journal,
and requires the actual queued job to fail with no output and no provider entry.
Only that exact job may add bounded context/cache dispatch entries during its
refusal window; an HTTP-refusal window cannot enter any worker/cache scope.
Missing ready, unknown coverage, stale/truncated
logs, tampering, failed writes, missing job scope or any actual provider/allocation
entry fails closed.

Actual original/output review PNGs preserve both full unscaled rasters and
retain three deterministic output detail crops with source/output SHA and crop
coordinates. These artifacts carry no quality verdict. A reviewer must still
inspect every mandatory anchor and supply actual `quality-review.json` and
source-bound external observations. Browser disabled-state guards retain real
no-POST observation windows and screenshots; all ten UI rows retain their
actual POST/job/progress and history/export/reacquisition evidence.

Interrupted attempts are durable and never automatically replayed. The
controller stops with `execution-start.json`, per-attempt records and partial
artifacts; use the existing authoritative reconciliation contract before any
continuation. A reconciled failed attempt remains a failed acceptance record.
Do not rerun the whole Actions execution to hide an ambiguous submission.

### Integration and runtime validation still required

This implementation was written on sc-26757's pre-55/56 base. Source56's actual
MLX/Candle image routes converge through `image_jobs/stream.rs` cached-request
seams, the two `generator_cache` cold/warm closures and
`inference_runtime::load`; advanced batch/embedding/noise/latent behavior adds no
separate model constructor in the inspected source. Merge 55/56 into 57 once,
resolve the actual hook positions against the merged source, and recheck every
MOSS route including the two actual native generate/edit helpers, warm admission,
CUDA context entry and cold admitted constructor before formal review,
compilation and freeze. Current static checks are not final integrated coverage
proof. Unknown/new allocation-capable routes must be covered or fail closed.

No API, worker, browser, compiler, model or campaign was started while writing
this source. Inert tests exercise protocol/transport/state/fixture/journal
failures with pure bytes and owned CPU subprocesses. Actual Actions execution
must still prove the real selectors, Chromium provisioning on both hosts,
installer refresh/download behavior, exact HF inventory completeness, embedded
web/auth, worker routing, preserved cross-host SQLite/project history, all
50 renders/58 distinct refusals, progress/cancel/publication/cache behavior,
whole-device memory observations and all external/visual acceptance evidence.

### Host tensor invariant and the real import-budget guard

The defensive aggregate request-cap branch is unreachable through otherwise-valid
current public inputs. At B=8/count1/context4096/W=H4096, the largest two FP32
embedding arrays are 512 MiB each, two uint8 masks are 32 KiB each, and FP32 noise
`[8,128,256,256]` is 256 MiB. Their exact sum is 1,342,242,816 bytes
(1280 MiB + 64 KiB), below `MAX_REQUEST_BYTES=1,342,308,352` by 65,536.
Reference-image pixels and prompt strings are not in that sum. Even adding five
maximum permitted 4096+12-byte NPY framings leaves 44,996 bytes of headroom;
framing is actually excluded from measured `metadata.byte_length`. Accounted
host retention/staging is a separate 1,879,117,836-byte bound, added to the
worker's native peak, rather than the aggregate comparison operand.

The controller rereads the actual integrated Base/SFT manifest, tensor-carrier,
MOSS envelope, NPY dtype/framing and cap source before freeze. It writes
`host-tensor-bound.json` with exact source file SHA and the shape/dtype/byte
derivation. Carrier/cap/envelope changes fail closed and require protocol
reconciliation. The shared lock freezes that receipt; `verify-retained`
recomputes it against the same clean integrated sources. This is an invariant
proof, **not an executed aggregate-cap refusal**.

The legacy `host-budget-refusal` slot instead performs one real authenticated
`POST /api/v1/projects/:id/assets` multipart call. It sends the exact retained
128-byte inert NPY header declaring FP32 shape `[1,134217729]`, or 512 MiB + 4 data
bytes, with **no payload**. This is explicitly an unsupported declaration, not a
complete otherwise-valid tensor. The existing parser returns
`tensor payload exceeds budget` before the later payload-length check and before
any data materialization. Exact submitted header bytes/SHA, actual HTTP 400
response, before/after full asset lists and their SHA are retained; no asset or
output may appear. The actual ready source-bound worker journal must show no
new job/cache/provider/allocation entry. Only this exact guard slot may leave
the image-job route; no new product setting, lowered cap or test bypass exists.
The profile still makes 58 refusal calls and 50 render executions.
