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
physical GPU1 by UUID, uses the existing real-weight concurrency group, builds
the shipped release profile, and creates separate owned API/worker epochs. All
generated state, staged payloads, builds, logs, and media live on D:. Existing
exported lifecycle, request/poll, host/device observation, media decode and
content-check primitives are reused; the prior25case campaign is not run.
The explicit `sceneworks-rust-api/h3-memory-proof` feature enables the scoped
validation controls; normal product feature defaults remain unchanged. The
exact native build command/features are retained in each build receipt.

## Required plan fields

| Field | Value or meaning |
| --- | --- |
| `version` | `1` |
| `stationary_pair_authorized` | `true` only after the final pair is stationary |
| `sceneworks`, `inference` | Exact40hex candidate and accepted inference-main commit |
| `expectedRunner` | One live `cuda-windows` or `cuda-windows-2` listener |
| `requestPath`, `requestSha256` | Absolute path and actual SHA256 of the recovered original request |
| `licenseAcknowledgmentsPath` | Existing base-model acknowledgment JSON; no fabricated consent |
| `seed` | Explicit u32, recorded alongside the original null seed |
| `poolBytes`, `externalReserveBytes` | Positive byte counts summing to24GiB; reserve at least2GiB |
| `hostBudgetBytes` | Separate positive host-RSS observation budget |
| `workstationDeviceBytes` | Actual chosen workstation observation envelope, no more than96GiB |
| `timeoutSeconds` | Per-anchor deadline, maximum7200seconds |
| `sources` | `tier`, `base`, `predictor`, `turbo`, each `{root,revision}` identifying the actual installed HF snapshot |
| `primitiveProof` | `{directory,runUrl}` of downloaded accepted allocator primitive artifacts at the exact inference-main commit |
| `materializingProof` | `{logPath,revision,runUrl}` for the accepted exact forced-materializing numerical test; H3 source tree must match final inference |

The source revisions and minimal file domains are fixed in the runner. The real
q4 tier is `SceneWorks/minimax-h3-mlx@137ce668…`, Veda predictor `9a1fd3a4…`,
Turbo8step `5d1d4829…`; supported installed upstream companion snapshots are
`6818f6c3…` and `939557dc…`. Placeholder `111…` fixture snapshots are refused.
Source payloads are copied only once if E:→D: hardlinks are impossible, hashed
once in the owned staging root, and hardlinked from D: into each anchor epoch.
Shard-index completeness is checked before model execution. No shared cache is
modified, no checkpoint is uploaded, and no model grid is executed.

## Two anchors and a bounded conditional addition

1. Actual workstation budget, normal Veda90% execution. Preserve original
   1344×768,345frames,24fps,14.375seconds,q4,balanced/crop,T2V,prompt and
   Turbo8step LoRA at weight1. Verify actual sparse calls, effective8steps,
   request/queued equality, full decoded video and generated stereo32kHz audio,
   exact dimensions/framecount/duration, and changing nonblack frames.
2. Same request with24GiB admission and a distinct enforced private pool. Force
   **Veda-to-dense** while leaving fused attention eligible. Verify actual
   fallback reason, continuous capped-pool RESERVED high-water, sampled whole
   device/RSS bounds, context/library/foreign reserve, full outputs, and cleanup.

These prove different routes. An earlier focused forced-materializing numerical
proof remains required: Veda-to-dense alone does not prove the no-fused route.
The memory counters name their scope. Job-stage sampled peaks retain their
latency/blind spots; the pool watermark is continuous and includes the whole
process epoch, without resetting away loading/conditioning/decode.

The two anchors do not automatically establish normal sparse24GiB execution if
the workstation selected a different configuration. Where existing allocation
bounds/evidence cannot establish that outcome, set `additionalSparse24:true`
with a concrete `additionalSparse24Reason` to add exactly one focused capped
normal-sparse render. Keep that decision/evidence explicit; no inferred sparse
success from the forced-dense anchor.

Failed anchors retain results and raw traces. Do not retry unchanged candidates
or enlarge a routine grid. Diagnose and repair the failing boundary, then rerun
only evidence actually invalidated. An uncapped Blackwell render or sampled
cancellation is never reported as enforced24GiB fit or RTX3090 validation.

## Validation and CI cost

`node --test scripts/h3-cuda-memory-terminal.test.mjs` runs only metadata,
request preservation, whitelist and refusal tests, without services/model work.
`node --check scripts/h3-cuda-memory-terminal.mjs` verifies the assembled entry
point. The manual workflow runs these before native/model setup. Runtime/build
cost of the long anchors is unknown before the first bounded campaign; two
maximum2hour request windows plus native build fit within the6hour job limit.
An explicitly justified third sparse24GiB case raises that job bound to10hours.
No automatic CI or per-pin measurement cost is added. Artifacts exclude model
payloads, app databases, build caches, and access tokens.
