# SDXL DDIM exact replay witness (SW-7148)

This is a prepared, unexecuted proof harness. It does not declare a deterministic route.
The current source checkpoint retains its existing inference pin. Integrate the reviewed
receipt implementation and final inference pin before building a live candidate.
Normal tests import only metadata validators. Only `workflow_dispatch` starts real weights.

## Registered dispatch route (required before a live run)

GitHub only enables `workflow_dispatch` when the workflow file exists on the default branch;
`--ref` does not register a brand-new workflow filename. See the
[official event contract](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#workflow_dispatch).
The initial standalone `sdxl-exact-replay-proof.yml` was therefore not a viable pre-main route
and has been removed.

The proof now uses the **existing** `.github/workflows/server-candle-linux.yml` dispatcher.
At source audit, remote default branch `main` was `5e88855d6c3890524f0ec4989a90115b14e1a2fe`,
the workflow was registered and active as ID `297197519`, and manual run `42342133418`
was completed. Its default-branch schema already accepts `starvector_terminal_operation`
value `standard` and the string `campaign_run_id`. No new input or choice value is
required, and no enabling main PR is needed. Recheck registration before live use.

On the **reviewed candidate ref**, that existing choice carries a JSON object with exactly:

```json
{
  "proof": "sdxl-exact-replay",
  "phase": "qualify",
  "planPath": "C:\\private\\approved-sdxl-plan.json",
  "deviceUuid": "GPU-00000000-0000-0000-0000-000000000001",
  "ownerGo": true
}
```

This is a shape example, not approved host data. The ordinary Linux compile job explicitly excludes
this marker, including when `ownerGo` is false. The selector is `standard`, so an old or
default-branch workflow can only perform its ordinary compile; it cannot select H3 or any
other learned model operation. Root routing checks only the string prefix
`{"proof":"sdxl-exact-replay",` and never parses ordinary campaign IDs as JSON. A recognized
prefix with malformed JSON fails the SDXL condition before model work while excluding the
ordinary compile. Other inputs, including plain StarVector campaign IDs, keep ordinary
compile behavior. The SDXL job requires the marker and owner
go, and its parser refuses extra keys, unknown phases and any command/script/ref field.
Other H3/StarVector selections retain their existing jobs and queues. If a candidate lacks
this bridge, do not dispatch it. Never substitute a different operation to get a run.

After independent review, required candidate CI, coordinator publication, final pin/plan
preflight and the owner's GPU go window, prepare a local `dispatch-inputs.json` containing
`starvector_terminal_operation: "standard"` and `campaign_run_id` as the serialized
five-field JSON object above, compactly encoded with `proof` first (for example,
`JSON.stringify({proof:"sdxl-exact-replay",phase,planPath,deviceUuid,ownerGo:true})`).
Whitespace before/inside the prefix or reordered keys do not select the SDXL job. The following is the concrete dispatch route, **not permission
to execute it now**:

```sh
gh workflow run server-candle-linux.yml --repo SceneWorks/SceneWorks \
  --ref REVIEWED_CANDIDATE_BRANCH --json < dispatch-inputs.json
```

Read back the run's `head_sha` and require the private plan's exact `sourceSha`; the harness
also enforces that hash before build and execution. A moving branch cannot silently qualify
a different candidate. Use the same dispatcher with `phase: "replay"` only after separately
reviewing the deterministic declaration and obtaining another go window. Nothing in this
route bypasses the feature PR/CI contract or turns proof work into PR/push-triggered inference.

## Approval and phases

Before dispatch, inform the owner and obtain an explicit Windows real-model go window.
CodeGraph rule `9fdbfea4-e2fc-4854-8342-79f5b831737f` applies even when a runner is idle.
Discover both organization and repository runner registrations and group access; select a
runner with Windows/X64/cuda/**real-weights** labels. Do not infer physical GPU capacity
from listener count. Reuse the existing `windows_cuda_execution.py` strict-idle UUID lease,
context-free ordinal mapping, owned Windows Job and post-run drain. No bypass or relaxed
admission policy is added here.

1. **Qualify:** dispatch `qualify` against the final reviewed app/inference candidate and a
   reviewed private plan. The harness runs two fresh count-one jobs through the production
   image API and worker, then compares complete execution identities and final PNG SHA-256.
   Equal bytes from this bounded sample are evidence for review, not a universal guarantee.
2. Review that evidence and the pinned provider/runtime implementation. If justified,
   implement and independently review a **route-specific** deterministic declaration for
   SDXL/DDIM on this exact device/runtime. Do not alter old receipts or label all Candle
   execution deterministic. If evidence differs, stop: qualification failed.
3. **Replay:** obtain another owner go window. A new private plan names the reviewed
   declaration PR, new app/inference revisions and immutable qualification digest. The
   harness runs two fresh jobs on that candidate, verifies actual provider support is
   `same_device_pinned_runtime`, and compares their execution identity (excluding only the
   support declaration) and PNG bytes with qualification. It then calls the separate production
   `/api/v1/jobs/:id/exact-replay` action and compares the third completed output. A new
   source receipt is necessary because old receipts remain bound to their old runtime pin.

The provider's actual report must have the declared scope. A plan or matching hashes never
rewrites a receipt or grants replay permission. The exact action still owns its native
source, device, worker, runtime, seed and completed-receipt checks.

## Private plan (runner-local, never upload)

Prepare an absolute JSON path outside the repository and artifact output. Version 1 fields:

- `sourceSha`, `inferenceRevision`: exact reviewed 40-hex candidate revisions.
- `requestSha256`: `digest(REQUEST)` from the harness. The request fixes SDXL text-to-image,
  1024 square, count one, seed 16402, DDIM/normal, 20 steps, CFG 7.5, no LoRAs or enhancement.
- `deviceUuid`, `physicalIndex`: approved NVIDIA UUID and its current NVML physical index
  (0 or 1). The child exposes exactly that UUID; Candle uses visible ordinal zero. The
  harness independently checks the context-free mapping and NVML label before starting
  processes. The final guard receipt must confirm the same ordinal, UUID and index.
- `ownerGo: true`, `goExpiresAt`: recorded approval and an ISO timestamp bounding the window.
  The harness rechecks expiry while waiting for work, not just at startup.
- `hub`: absolute, preprovisioned HF hub directory. Offline flags forbid fetching weights.
- `sourceIdentitySha256`: digest of `sourceIdentity(selectedRoute)` from a metadata-only,
  server-issued binding for the approved installed SDXL source. That projection preserves
  exact member hashes/roles/source identities and provider operation, excluding only the
  project-dependent authored-options hash. Only one base SDXL slot is accepted.
- `files`: exact selected-member mapping, one entry per bound file:
  `{slot, memberIndex, relativePath, path}`. Paths must resolve under `hub`; the harness
  verifies size and SHA-256 against the **server-issued** private binding before releasing
  a claim and after completion. No adapter or optional component is allowed in this witness.
- `scheduleSha256`: digest of the expected resolved schedule array from the final pinned
  provider's **metadata-only schedule calculation**, including sampler, numeric schedule
  digest and estimator. Audit `candle-gen-sdxl::pipeline::denoise_curated`: SDXL
  `AlphaSchedule::scaled_linear` uses shared `SDXL_*` constants, `DiscreteModelSampling::sdxl`,
  `schedule_sigmas(Normal, ..., 20)` and `resolve_schedule("normal", ...)`. No sampler is run
  to prepare this hash. Do not copy the first live output's hash into the plan to waive a
  mismatch. Final-pin schedule metadata is a live-preparation prerequisite, not an invented
  constant in this harness.
- `runtimeFiles`: `{path, sha256}` for the complete approved loaded DLL census matching
  `nv*`, `cuda*`, `cublas*`, `cudnn*`, `cufft*`, `curand*`, `cusolver*`, `cusparse*`,
  `vcruntime*`, `msvcp*`. Include the trusted System32 NVIDIA driver. The worker's actual
  module list must match exactly after every render; do not relax the list after failure.
- For replay only: `qualificationFile`, `qualificationSha256` (canonical digest of accepted
  qualification JSON), and `declaration: {scope:"same_device_pinned_runtime",
  candidateSourceSha, candidateInferenceRevision, reviewUrl}`. The review URL must identify
  the SceneWorks/inference PR that reviewed this narrow declaration.

The app binary is built from the clean candidate. The harness records its SHA-256 and
requires it to remain unchanged within the run. Replay may use a newly reviewed binary
because support is added in a separate commit; driver/device, OS and loaded DLL hashes must
remain identical to qualification, as must the observed execution identity excluding only
its deterministic-support field. Private route, runtime and plan preparation require the
actual final pin and provisioned host. They cannot be truthfully filled from this source-only
checkpoint or a metadata test fixture.

## Isolation and evidence

The harness creates its own API, one GPU worker, project, data/config directories, credentials
and jobs database. It generates an ephemeral master and derives a worker key using the
production ASCII-hex HMAC contract in `worker_proof.rs`. Registration must be privately
authenticated. A loopback proxy holds claims until the read-only queued source/member checks
pass. It forwards real API claims unchanged and verifies their route and capability; it does
not manufacture execution reports or authenticated claims. The API and worker perform all
normal admission and receipt validation.

SQLite opens with `DatabaseSync(..., {readOnly:true})`. The only SQL operations are SELECTs
against this task's database. Private logs/database/credentials are removed after processes
stop. The upload step names only `evidence.json`; it never uploads the plan, source route,
master/key, capability, raw results, model files, PNG, or logs. The evidence publishes hashes,
job/attempt identities, device UUID and provider/runtime/scope identifiers. Generic failure
text avoids printing private assertion values. The physical guard's receipt is represented
by its hash, and evidence remains rejected unless the guard proves completion and drain.

The runtime DLL census and stable driver/device/OS digests bound this witness; they do not
claim cross-device, cross-driver, cross-build or statistical determinism. This lane intentionally
refuses fallback samplers, stages/hires, extra outputs, source changes and missing reports.

## Source anchors and metadata checks

- API: `apps/rust-api/src/generation.rs::create_image_job`, `router/jobs.rs` receipt/exact action,
  `workers.rs::register_worker` private source-proof admission.
- Store: `jobs_store.rs::create_exact_replay_job`, claim binding and completion receipt paths.
- Contracts: `execution_receipt.rs`, `model_artifacts/portable_binding.rs`, `worker_proof.rs`.
- CUDA mapping: `sceneworks-worker/src/lib.rs::cuda_startup_health` documents visible ordinal
  zero; `windows_cuda_metadata.py` maps it without creating a CUDA context.
- Existing lifecycle pattern: `scripts/measure-sdxl-cuda-endpoints.mjs`; this witness does not
  inherit that measurement's artificial memory cap.

Run `node --test scripts/ci/sdxl-exact-replay-proof.test.mjs` on Node 22 or newer. This tests
completed identity/output binding, changed bytes/source/runtime facts, retry/cancel refusal,
read-only SQLite enforcement, owner-plan gating, separate qualification, exact physical
mapping and manual-only workflow. Test PNG and identity values are synthetic metadata and
are never submitted to a worker. No model computation occurs in this suite.
