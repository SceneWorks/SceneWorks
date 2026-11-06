# SW-F329 long-clip CUDA acceptance

Run once after the inference feature has merged to main, all focused numerical
and lifecycle proof is accepted, and the clean final SceneWorks candidate pins
that exact inference-main commit. No ordinary CI calls this campaign.

The registered `server-candle-linux.yml` workflow has one new explicit operation:
`h3-cuda-memory`. Pass its existing `campaign_run_id` string as the JSON plan
below. It calls the same-ref `h3-cuda-memory-terminal.yml`, preserving the other
dispatch choices and input keys. The request defaults to no authorization; the
plan must explicitly set `stationary_pair_authorized:true`.

The workflow checks the exact pair and script startup before native setup, binds
the selected physical GPU by UUID (default GPU1), uses the existing real-weight concurrency group, builds
the shipped release profile, and creates separate owned API/worker epochs. All
generated state, staged payloads, builds, logs, and media live on D:. Existing
exported lifecycle, request/poll, host/device observation, media decode and
content-check primitives are reused; the prior25case campaign is not run.
The explicit `sceneworks-rust-api/h3-memory-proof` feature enables the scoped
validation controls; normal product feature defaults remain unchanged. The
exact native build command/features are retained in each build receipt.

The whole campaign additionally takes the accepted engine's existing
`PUBLIC/SceneWorks/gpu-locks/<uuid>.lock` lease through `execution_claim`.
This coordinates separate proof workflows on the same physical host. A busy
lease waits within the original590minute campaign deadline; it cannot start an
anchor early or reset that deadline after waiting. The plan's pool plus external
reserve supplies the24GiB minimum free-memory check, while the resident anchor
keeps its own provider admission. No device ordinal fallback is permitted.
The existing suspended-child Windows Job owns Node and all descendants. Normal
exit, nonzero exit, timeout and interruption reap owned descendants before the
physical post-idle check and lease release. Sixty seconds are reserved for
cleanup, and ten workflow minutes remain for upload. The sibling `-lease`
directory retains claim/release samples, child logs and `campaign-lease.json`;
an unsuccessful child or unverified release cannot produce a passing receipt.

## Required plan fields

| Field | Value or meaning |
| --- | --- |
| `version` | `1` |
| `stationary_pair_authorized` | `true` only after the final pair is stationary |
| `sceneworks`, `inference` | Exact40hex candidate and accepted inference-main commit |
| `expectedRunner` | One live `cuda-windows` or `cuda-windows-2` listener |
| `physicalGpuIndex`, `physicalGpuUuid` | Optional exact numeric index `0` or `1`, default `1`. Selecting `0` requires UUID `GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0`; an explicit GPU1 UUID must be `GPU-e4b79931-7be6-f216-460a-f5405cfafffe`. The workflow, lease, request, service and inventory must agree. |
| `requestPath`, `requestSha256` | Absolute path and actual SHA256 of the recovered original request |
| `licenseAcknowledgmentsPath` | Existing base-model acknowledgment JSON; no fabricated consent |
| `seed` | Explicit u32, recorded alongside the original null seed |
| `poolBytes`, `externalReserveBytes` | Positive byte counts summing to24GiB; reserve at least2GiB |
| `hostBudgetBytes` | Separate positive host-RSS observation budget |
| `workstationDeviceBytes` | Actual chosen workstation observation envelope, no more than96GiB |
| `timeoutSeconds` | Per-anchor deadline, maximum7200seconds |
| `additionalSparse24`, `additionalSparse24Reason` | `true` and the explicit E5 gap: normal streamed Veda fit plus resident/window media diagnostics and matched-input component parity |
| `sources` | `tier`, `base`, `predictor`, `turbo`, each `{root,revision}` identifying the actual installed HF snapshot |
| `primitiveProof` | `{directory,runUrl,revision?}`; original measured revision defaults to the final pin, otherwise must be its ancestor with an identical complete Git tree |
| `materializingProof` | `{logPath,revision,runUrl}` for the accepted exact forced-materializing numerical test; the existing45 numerical/layout/getter/build selections must match final inference, subject only to the two exact non-numerical-file mappings below |
| `retainedWorkstation` | Optional, narrowly committed accepted workstation control described below; both24GiB anchors still run fresh |

GPU0 selection refuses `retainedWorkstation`: all three product anchors run fresh
on GPU0, and full video/audio diagnostics compare that fresh workstation output
with the fresh normal24GiB output on the same card. `device-selection.json`
records physical index/UUID and logical CUDA ordinal0 under its single-UUID mask.
The accepted B9 DiT, raw Veda and512MiB primitive receipts remain their original
GPU1 component/source evidence (B9 and M4 have identical whole trees). They are
not relabeled as GPU0 measurements; the new full-model anchors establish the
actual GPU0 cap enforcement and fit; E5 component parity remains separate.
GPU selection changes neither
the21GiB pool plus3GiB outside reserve nor the384GiB host,95GiB workstation,
7200second deadlines or original request/seed42. Other H3
manual callers retain their GPU1 default.

The source revisions and minimal file domains are fixed in the runner. The real
q4 tier is `SceneWorks/minimax-h3-mlx@137ce668…`, Veda predictor `9a1fd3a4…`,
Turbo8step `5d1d4829…`; supported installed upstream companion snapshots are
`6818f6c3…` and `939557dc…`. Placeholder `111…` fixture snapshots are refused.
Source payloads are copied only once if E:→D: hardlinks are impossible, hashed
once in the owned staging root, and hardlinked from D: into each anchor epoch.
Shard-index completeness is checked before model execution. No shared cache is
modified, no checkpoint is uploaded, and no model grid is executed.

## Three anchors in one bounded campaign

1. Actual workstation budget, normal Veda90% execution. Preserve original
   1344×768,345frames,24fps,14.375seconds,q4,balanced/crop,T2V,prompt and
   Turbo8step LoRA at weight1. Verify actual sparse calls, effective8steps,
   request/queued equality, full decoded video and generated stereo32kHz audio,
   exact dimensions/framecount/duration, and changing nonblack frames.
2. Same request with24GiB admission and a distinct enforced private pool. Force
   **Veda-to-dense** while leaving fused attention eligible. Verify actual
   fallback reason, continuous capped-pool RESERVED high-water, sampled whole
   device/RSS bounds, context/library/foreign reserve, full outputs, and cleanup.

3. Same original request, normal Veda90%, under the same enforced24GiB pool.
   Verify positive sparse execution and actual one-block transfer receipts for
   **both** the text encoder and DiT. The workstation must execute the resident
   policy and this anchor must execute the windowed policy. Their completed
   execution-policy traces, source pair, predictor/plan/kernel identities,
   actual queued creative settings and bound job/media identities must agree.

The third normal-sparse anchor is required by E5: the forced-dense anchor has
zero sparse calls and cannot establish Veda resident/window numerical
correctness. Set `additionalSparse24:true` with that concrete reason. Startup
rejects a two-anchor plan. This remains one bounded campaign, with no grid or
extra repeated-baseline model render.

After all three anchors pass their existing full decode, dimensions,345-frame
count, duration, stereo32kHz audio and nondegenerate-content checks, compare the
workstation and normal-sparse24GiB outputs. The forced-dense output is excluded
from this equivalent-route diagnostic. Both source-bound normal executions must
report
positive sparse calls; a CPU/skipped/fallback-only witness cannot pass.

The diagnostic reuses the terminal `decodedDelta` witness and adds full
native-resolution RGB24 decoding of **all345 frames** plus native-rate,
interleaved floating-point PCM for **both audio channels**. RGB values are
normalized by255; audio keeps native floating-point amplitude. Retain stream and
channel RMS, maximum differences and nonfinite/length checks. These metrics
describe output differences; they do not establish an inherited acceptance bound.
The historical [`maximumBaselineDelta:0.001`](sc-26691-campaign/h3-terminal-cuda.json)
tests unchanged-repeat stability on 1 fps, 16x16 RGB and mono 8 kHz audio. It is neither
a tensor tolerance nor a universal full-pixel/native-stereo residency tolerance.
No new full-media threshold is selected here.

Raw vectors are compared in64KiB file chunks and removed after comparison,
including failure. They are never retained as multi-gigabyte JSON arrays or
uploaded artifacts. `veda-equivalence.json` retains actual element counts,
per-stream/per-channel RMS, maximum absolute differences, original harness bounds,
sampled-comparator deltas, executed policy/kernel witnesses and the original
bound media hashes. A failed comparison writes a failed receipt and prevents
the campaign's successful aggregate result in the current CGPU controller.

### Numerical acceptance and current failed-run disposition

E5 requires identical fixed inputs through equivalent affected production paths,
with dtype, task, adapter stack and workspace policy held equivalent. Compare
residency separately from cuBLAS default versus explicit 32 MiB workspace. That
workspace difference is a source-backed, unproven confound; no causal explanation
or acceptable-drift claim follows from it alone. Reuse valid unchanged component
proofs and fill demonstrated coverage gaps with the bounded real-q4/Turbo BF16
projection and large-Q/K witness before another full render.

Existing Q/K RMS/partial-RoPE assertions are exact for their fixed inputs. Keep
exact Veda mask/score assertions for common inputs and the source-defined output
error checks. Differing upstream inputs cannot be treated as identical-input
mask proof. The small F32 DiT fixture's 1e-5 relative bound does not certify the
production BF16 projection/FFN path, and no general BF16 projection/FFN acceptance
bound is established here. Locate the first differing intermediate, apply the
relevant established component contract, and repair any violation. Full media
content/geometry and actual normal-Veda24 execution remain mandatory alongside
this numerical evidence; neither two valid clips nor diagnostic RMS alone passes E5.

[Campaign 42243645213](https://github.com/SceneWorks/SceneWorks/actions/runs/42243645213),
measured at consumer `c8815c7cd`/inference `2ff035382`, remains failed under the
predeclared 0.001 harness rule. All three fresh GPU0 children passed fit/media
checks and the lease verified
post-idle. Preserve those receipts and their source/binary/media hashes unchanged.
The normal-pair full-video RMS 0.09263594898258952 is a recorded divergence. The
original comparator stopped before audio; a later CPU-only native-stereo diagnostic
recorded RMS 0.015946034316153112, separately from that failed execution. Neither
number is declared acceptable. This docs correction changes no controller or test;
a reviewed acceptance implementation still awaits the component findings/schema.
No higher post-result threshold or retrospective campaign pass is authorized.

An earlier focused forced-materializing numerical proof remains required:
Veda-to-dense can still use FlashAttention. Job-stage sampled peaks retain their
latency/blind spots; the pool watermark is continuous and includes the whole
process epoch, without resetting away loading/conditioning/decode.

## Retaining the accepted uncapped control after the allocator repair

The optional `retainedWorkstation` applies only to the accepted workstation job
`job_b29fc9a4c1047efdabebe4eaaea1b754` from run42184091082, consumer
`66b5d26d94676914eba718a9af3d4c2ad3561099` and inference
`073e34f21382dc25ef711e0620ebf16bc7e74b6d`. It never reuses a24GiB, failed,
partial, forced-dense or control-mismatched result. The controller owns exact
commitments for the original result/build/exits/hardware/staged-model/request
receipts, binary and media. Original measured source identities remain in the
copied result; revalidation and current-target equivalence have separate receipts.

Supply `{campaignDirectory,runUrl,sourceAuditPath,sourceAuditSha256,binaryPath?}`.
`campaignDirectory` contains the retained original campaign; `runUrl` must name
42184091082. `binaryPath`, when supplied, relocates the original executable only;
its bytes must still match the committed original hash. All paths are absolute.

Before dispatch, the coordinator must semantically review the actual final
allocator repair and bind a D:-only source audit with this shape:

```json
{
  "kind": "sw-f329-uncapped-workstation-source-equivalence-v1",
  "originalSourcePair": {"sceneworks": "<original66b5 SHA>", "inference": "<original M2 SHA>"},
  "targetSourcePair": {"sceneworks": "<final consumer SHA>", "inference": "<accepted final inference-main SHA>"},
  "uncappedAllocatorPreserved": true,
  "boundedQkResidentPreserved": true,
  "boundedQkRefinerPreserved": true,
  "boundedQkResidentPricingPreserved": true,
  "reviewUrl": "https://github.com/SceneWorks/inference/pull/<reviewed repair>",
  "engineChanges": [{"path": "<code-owned permitted path>", "before": "<original Git object or null for new file>", "after": "<reviewed target Git object>"}],
  "consumerPinMetadata": [{"path": "<code-owned generated pin metadata path>", "before": "<original Git object>", "after": "<reviewed target Git object>"}],
  "consumerCiChanges": [{"path": "<code-owned CI-only path>", "before": "<original Git object or null>", "after": "<reviewed target Git object>"}]
}
```

This example is not evidence. Populate exact40hex objects after reviewing the
uncapped early-return allocations and accepting the final source. The controller
rechecks ancestry, actual objects and the complete engine changed-path set before
reuse and again before the final success receipt. Its permitted paths are owned
by code: capped allocator/recovery, their primitive/pure tests and registration,
the four explicit streamed-Q/K operator/planner files,
documentation, generated consumer pin metadata, and the reviewed standard Linux
cache CI helper. There are no caller-supplied path exemptions. The CI workflow
must remain byte-identical outside `build-candle`. All other application/operator/
kernel/build source stays exact; manifests/lock may change only by substituting
the accepted inference SHA. The audit's semantic assertion is bound to those exact
reviewed objects; it is not inferred merely from filenames or a new commit id.

The M4 generated provenance change is limited to the window planner's blob cell
in `config/inference-provenance-candidates.tsv`, matching the actual original and
target engine Git blobs. Its marker hash/classification and every other row stay
exact. The third-party audit may then update only pin labels and the population/
audit digests derived from that exact table/payload; counts, classifications and
all other fields remain identical. Both generated files need reviewed object
bindings. This is not a general TSV/JSON exemption or a new provenance scan.

When the four Q/K repair files change, all three `boundedQk*Preserved` review
assertions are mandatory. The controller also reads the original/target sources:
it checks the unchanged Resident/refiner QKV arm and surrounding operators, the
exact CUDA/Both-window guard, projection release only in a CUDA streamed model,
unchanged legacy allocation arithmetic and full/None pricing, and bounded calls
only in the explicit selected-window branches. The selected-window FFN5/legacy4
and norm9/legacy8 buffer counts, reduction and
norm-weight allowances are checked as exact conditional expressions; their false
arms must reduce to the original allocation arithmetic.
The original request-policy file is outside the permitted diff. Unknown paths,
changed guards, modified Resident
math/pricing and incomplete/incorrect object bindings refuse reuse.

These assertions establish only original95 source applicability. They do not
assert that the new branch passed parity. Before dispatch, the coordinator must
accept the fresh registered DiT `cuda0_materialized` receipt and its entire late
BODY/device/numerical/chunk/tail/cancellation controls at the reviewed source.
The primitive whole-tree and raw Veda45-source guards stay strict; changed source
must satisfy them with fresh affected evidence. Both full24GiB anchors and native
media diagnostics and matched-input component parity remain mandatory. Preserve
the historical audit/parameters and
write a new source-bound audit for the repaired final pair.

The accepted record must still match the original seed42, request/queued recipe,
q4/Turbo8/Veda controls, Resident execution, all345frames/audio/content, successful
cleanup, physical GPU1 UUID and driver, exact weight-file commitments, native
features/flags and original binary/media hashes. Media is hardlinked/copied into
the new output and hashed before and after full native AV comparison. The old
video is decoded and its content is checked again. New staging hashes must match
all original source roles/files. Each fresh24GiB service still binds its own
current executable/hash/source and must use the equivalent native configuration.
The acceptance artifact includes the six original receipt files under
`workstation-veda/original/**/*.json`, preserving their bytes for inspection
alongside the recorded hash commitments.

Only the workstation generation is skipped. Both forced-dense and normal sparse
24GiB anchors run fresh. The complete original95GiB versus fresh normal24GiB RGB/
stereo PCM diagnostics and the production-path component acceptance above remain
mandatory. Output reports both
actual measured source pairs and the explicit source mapping; nothing relabels
the old binary or measured peaks as a new-source execution.

The allocator primitive must report a nonnegative exact integer
`successfulRecoveryCount` and the boolean
`fragmentationRecoveryExercised:(successfulRecoveryCount > 0)`. Zero/false is an
honest observation and makes no claim that a CUDA retry recovered. Over-cap
refusal, both constructors and the cuBLAS stream-reset check remain mandatory.
The failed stress-fixture run is not accepted evidence; fresh full-model24GiB
anchors and their reserve/AV checks remain mandatory regardless of this count.
A reviewed-branch primitive may be retained across its ancestor/main merge only
when the whole source tree is identical, with original revision/run recorded.
The raw Veda45-object comparison keeps every numerical, getter, kernel and build
object strict. A differing Candle core tree may map only an actual
`VENDORED.md` content change; a differing candle-gen tree may map only
`tests/cuda_proof_pool.rs`. The latter is included by the candle-gen integration
target, not the dependency library linked by the H3 Veda unit-test binary.
Each mapping requires the actual Git diff to contain exactly that one file,
and every remaining recursive path, mode and blob to be identical. Reports
retain the original measured tree/file objects and record target objects and
the explicit mapping. Any other change, including core code, refuses reuse.
The passed Veda run42200518860 remains an execution of source4e3039c, not a
new-source run or a full-checkpoint24GiB result.

Failed anchors retain results and raw traces. Do not retry unchanged candidates
or enlarge a routine grid. Diagnose and repair the failing boundary, then rerun
only evidence actually invalidated. An uncapped Blackwell render or sampled
cancellation is never reported as enforced24GiB fit or RTX3090 validation.

## Validation and CI cost

`node --test scripts/h3-cuda-memory-terminal.test.mjs` runs only metadata,
request preservation, sparse/policy/media bindings, full raw-vector RMS,
whitelist and refusal tests, without services/model work. Fake FFmpeg fixtures
exercise late-frame/channel changes, NaNs, truncated streams and cleanup.
`node --check scripts/h3-cuda-memory-terminal.mjs` verifies the assembled entry
point. The manual workflow runs these before native/model setup. Runtime/build
cost of the long anchors is unknown before the first bounded campaign. Three
maximum2hour request windows plus native build and CPU media comparison use
the existing10hour bound selected by `additionalSparse24:true`.
No automatic CI or per-pin measurement cost is added. Artifacts exclude model
payloads, app databases, build caches, and access tokens.

## Conditional matched Resident control

The existing manual memory workflow also accepts the explicit
`mode: "matched-resident-control"`. This is one conditional diagnostic after the
component results are assessed. It is not a campaign rerun, a 24 GiB fit proof,
feature acceptance, or a new final-pixel tolerance. Its output and artifact scope
is `matched-resident-control`; the original failed aggregate remains unchanged.

This mode requires `matchedResidentControlAuthorized: true`, physical GPU0 with
its full canonical UUID, seed 42, the original request digest, inference M4
`2ff0353820e370d004305ae99a035ab38b93de30`, admission 95 GiB,
pool 92 GiB and outside reserve 3 GiB. Omit campaign-only fields
`additionalSparse24`, `additionalSparse24Reason`, `retainedWorkstation`,
`primitiveProof` and `materializingProof`. Other common source, request, runner,
timeout and host-budget fields retain their existing validation. The license
path names the original frozen acknowledgment under the prior service's config.
The workflow is still manual, uses the same UUID lease/owned Windows Job and
observations, and runs no model in ordinary CI. The one-control lease deadline
is the requested render timeout plus 30 minutes, within the 180 minute workflow.

The driver reuses only the immutable retained executable
SHA256`a5e83b704d67b999b1919d7308654d42dc9e1bc2de66347c67b0f07242f9b621`
and original build receipt
SHA256`8ac02a48dd1e1c5c1b523dd45684c6704658402808b357804ef46ca7758a3343`.
There is no general prebuilt API, Cargo invocation, old run-ID/cache-key reuse or
write to that target. A separate exact CGPU checkout supplies runtime registry
and configuration; new source records distinguish controllerSourcePair from
the original CGPU/M4 runtimeSourcePair. The original frozen license acknowledgment
and complete staged-source collection are independently bound. Current loaded
CUDA/MSVC DLL versions and hashes are recorded; historical loaded-DLL hashes
were not recorded and are not inferred from the executable hash.

Success of this diagnostic's execution requires actual normal Veda, Resident,
windowNone, zero streamed-block transfers, complete native video/audio/content,
92+3 bounds and owned cleanup. Admission labels alone cannot force Resident.
Complete AV comparisons to both prior GPU0 anchors are observations without a
new equality threshold. The existing physical-GPU lease must additionally
verify post-execution idle state. Component or source findings should be
resolved before deciding to run this conditional control.
