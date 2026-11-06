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
| `additionalSparse24`, `additionalSparse24Reason` | `true` and the explicit E5 gap: normal resident versus normal streamed Veda AV equivalence |
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
from this equality comparison. Both source-bound normal executions must report
positive sparse calls; a CPU/skipped/fallback-only witness cannot pass.

The comparison reuses the terminal `decodedDelta` RMS witness and adds full
native-resolution RGB24 decoding of **all345 frames** plus native-rate,
interleaved floating-point PCM for **both audio channels**. It applies the
existing CUDA H3 campaign's predeclared same-backend
[`maximumBaselineDelta:0.001`](sc-26691-campaign/h3-terminal-cuda.json) RMS
ceiling, including each channel separately. RGB values are normalized by255;
audio keeps native floating-point amplitude. This is a decoded-media RMS
criterion, separate from the retained intermediate tensor tolerance checks;
no cross-backend allowance or post-result tolerance adjustment is used.

Raw vectors are compared in64KiB file chunks and removed after comparison,
including failure. They are never retained as multi-gigabyte JSON arrays or
uploaded artifacts. `veda-equivalence.json` retains actual element counts,
per-stream/per-channel RMS, maximum absolute differences, fixed bounds,
sampled-comparator deltas, executed policy/kernel witnesses and the original
bound media hashes. A failed comparison writes a failed receipt and prevents
the campaign's successful aggregate result.

An earlier focused forced-materializing numerical proof remains required:
Veda-to-dense can still use FlashAttention. Job-stage sampled peaks retain their
latency/blind spots; the pool watermark is continuous and includes the whole
process epoch, without resetting away loading/conditioning/decode.

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
