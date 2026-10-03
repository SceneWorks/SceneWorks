# Dense KV baseline harness (SC-20671)

`scripts/kv-baseline-harness.mjs` defines the sealed receipt boundary for the
compressed-KV POC. A producer supplies model/toolchain provenance, matrix
coordinate, geometry, Darwin `phys_footprint` and `phys_footprint_peak`, MLX allocator readings,
timings, the checked-in quality/statistical contract, complete lifecycle
outcomes, cancellation cleanup, and typed allocation events. The harness
rejects incomplete runs, identity drift, a missing or hash-mismatched model
file at record time, any non-nominal thermal state, failed cleanup, and any
detected full-cache temporary in a compressed run. Dense receipts retain the
product's measured cache allocation and transient coexistence as baseline evidence.

The receipt shape is described by
`packages/schemas/kv-baseline-receipt.schema.json`. Producers write each receipt
as an atomically renamed **directory** under `docs/calibration/`; a set contains
`receipt.json`, `receipt.md`, their exact-byte sidecars, and the four required
`fixtures/<name>.json` artifacts with sidecars that name those full relative
paths. The v2 producer also seals every fixture type for all five repeats.
The Markdown embeds the JSON semantic hash, and readers rehash and parse
every fixture, so an interrupted, missing, or mixed-generation set is rejected.
Comparison reports are generated with `compare`:

```text
node scripts/kv-baseline-harness.mjs record input.json dense-receipt
node scripts/kv-baseline-harness.mjs compare dense-receipt compressed-receipt comparison.json
```

`record` requires the local-only `provenance.modelFilePath` input and verifies
either that exact file or every resolved file in a sharded snapshot directory;
directory identity is the sorted path/byte-count/content-hash aggregate used by
the inference producer. It then verifies
its byte count and SHA-256 before removing the path from the portable receipt.
The four local fixture paths are likewise required only while recording; their
exact bytes are validated, copied, and rehashed when the published set is read.
This is the same layout emitted by the inference producer, so its genuine
receipt bundles can be consumed directly by `compare` and `campaign`.
`record` atomically generates the complete receipt set; `compare` generates its
JSON artifact, Markdown report, and exact-byte SHA-256 sidecars. The
checked-in quality contract is itself verified against its exact-byte sidecar
before the module accepts any receipt.

`readDarwinMemory()` is the only platform-specific reader: it obtains the
kernel-maintained current and lifetime-peak physical footprints through
`footprint -p`. The inference producer reads the same ledger with one
`proc_pid_rusage` syscall (phase `source: "proc_pid_rusage"`; `footprint -p`
receipts stay valid). MLX values are
inputs because the producer owns the MLX session and must sample active,
cached, and peak allocator counters at the defined phase boundaries. The
typed event stream uses high-water snapshots, not additive phase totals:
sequential token/layer events reconcile by their maximum, and the persistent
successor is counted once. A
`product-cache_release` event with lifetime `released` records exact released
K+V ownership. It is required lifecycle evidence and is excluded from both
workspace attribution and full-cache-temporary detection. Post-run release is
measured against the loaded-model boundary: MLX active and allocator-cache
tolerances are exactly zero, while a frozen 512 MiB Darwin-footprint allowance
bounds process-resident Metal/JIT runtime pages that MLX does not report as live
tensors or allocator cache.

`geometry.kvLength` is the maximum live token offset; `geometry.capacity` is
the producer-observed allocated token capacity of the retained K/V buffers.
Block growth, trim, and restored prefixes can make capacity exceed live length
without a universal rounding formula. Dense persistent bytes must equal
`batch × layers × kvHeads × capacity × headDimension × elementBytes × 2`
exactly, with zero reconciliation tolerance. Fit-boundary occupancy uses live
`kvLength`, so allocation beyond the native context does not masquerade as
additional usable context.

Schema v5 (sc-20671) changes what decode throughput means. Each of the five
timing samples carries a dedicated **fixed-length steady decode**: after the
repeat's four fixtures, the candidate arm prefills the row context (the kernel
fixture's raw prompt) into a fresh cache of the row's KV representation and
greedily decodes exactly 256 tokens with stop tokens forced through
(`steadyDecodeForcedStopTokens` counts them). The first token is untimed (TTFT
is separate); the other 255 are timed from GPU completion to GPU completion, so
`decodeTokensPerSecond` is exactly `steadyDecodeTimedTokens × 1000 /
steadyDecodeMs`. 256 is the longest fixed length every frozen band admits (the
fit-boundary prompt is refused above `native context − 256`). Under v4 the
value was a phase delta of the coordinate's own EOS/budget-terminated
generation — one token for chunked rows — and measured mostly the
`footprint -p` sampling at the `decode-steady` boundary; phase stamps now also
exclude the observer's own memory sampling.

Compile attribution `first-dispatch-minus-steady-v2` (sc-20671) records compile
cost instead of asserting it. `noiseSamplesMs` are the steady dispatches of the
same operation: the four post-first repeats of a cold row, or the five measured
repeats after a warm row's warmups. `noiseBandMs` is their `max − min`.
`compileCostResolved` is true only when `firstDispatchExcessMs` exceeds the
band, and then `compileCostMs` (and `timings.coldCompileMs`) carry the excess.
Otherwise both are omitted/null, and `compileCostUnresolvedReason` is
`first-dispatch-not-slower-than-steady` or `excess-within-steady-noise-band`.
At large geometries compute dominates and kernel compile cost is below
run-to-run noise, so a non-positive excess is a measurement, never a refused
row. Missing or non-finite probes still fail closed. The retired `-v1` method is
refused (its receipts belong to older producer revisions and cannot resume). The steady decode runs outside every
coordinate observer on a cache released before the next repeat, and its
`prompt + 256` live tokens are admitted by the producer's preflight bound.
Batch rows are timed as one sequence, since the compressed arm has no batch
route. `provenance.powerMode` is the active energy mode (`automatic`,
`low-power`, or `high-power` from `pmset -g`), and `provenance.hostStates`
records it with the thermal state at `row-start` and `row-end`, bracketing the
row's phase samples; a non-nominal `pmset -g therm` or
`NSProcessInfo.thermalState` at either boundary, or a mid-row power-mode change,
refuses the row.

`compare` refuses mismatched source/model/toolchain/hardware/power/thermal
identity, matrix, contract, or geometry and reports KV reduction (only for a
coordinate that ran compressed; see *Compressed rows*),
decode-steady/lifetime-peak process-footprint deltas, and throughput ratio.
The current `campaign` reader requires a producer-published v2 aggregate with
exactly eight sealed receipts: short, medium, memory-material, and fit-boundary
for each of Llama and Qwen. Both memory-material selectors are warm, single
request, single-shot prefill. The other rows cover cold starts, chunked prefill,
and supported batching under the fixed schedule. It rejects missing, extra,
duplicate, cross-schedule, cross-source/model, and artifact-mismatched rows.
`campaign-legacy` is an explicit read path for historical v1 64-row aggregates;
it does not classify them as current v2 evidence. No complete v2 campaign
result is checked in yet.

| family | band | request | prefill | process |
| --- | --- | --- | --- | --- |
| llama | short | single | chunked | cold |
| llama | medium | supported-batch | single-shot | warm |
| llama | memory-material | single | single-shot | warm |
| llama | fit-boundary | single | chunked | cold |
| qwen | short | single | single-shot | cold |
| qwen | medium | supported-batch | chunked | warm |
| qwen | memory-material | single | single-shot | warm |
| qwen | fit-boundary | single | chunked | cold |

The inference producer requires a safety policy file before parent or worker
execution. Its schema version 1 fields are `rowDeadlineSeconds`, `pollMillis`,
`termGraceMillis`, `hostFreeReserveBytes`, `childFootprintCapBytes`,
`maxContextTokens`, `maxRequestTokens`, `stdoutCapBytes`, and `stderrCapBytes`.
All values are positive integers within the exact JavaScript integer range;
unknown fields fail. The v2 manifest binds the policy hash, computed from
sorted-key, two-space JSON without a trailing newline, and a resume identity
covering captured source, executable, model and
reference inventories, prompt, schedule, and policy. SceneWorks requires the
same trusted policy and sealed `identity.json` from the captured local resume
directory when reading a v2 aggregate. The reader rehashes that identity and
binds captured source and model inventories to every receipt. The manifest's
self-reported resume hash alone is not independent provenance. Current checkout or inference-pin
movement is not a reason to invalidate a captured measurement.

```text
node scripts/kv-baseline-harness.mjs campaign producer-campaign summary.json \
  --safety-policy policy.json --resume-identity trusted-resume-dir/identity.json
node scripts/kv-baseline-harness.mjs campaign-legacy historical-v1-campaign summary.json
```

The inference producer is the standalone `sc20671-kv-baseline` executable. Its
`parent` mode consumes no caller-supplied coordinate list: it reads the frozen
eight-row covering schedule, forks a new child for every cold row, and requires
an in-process warm-up in each warm worker. Workers collect identity, geometry,
Darwin/MLX phase samples, allocation events, timings, output, prefix reuse, and
cancellation from the product route; the parent accepts only the child’s sealed
JSON/Markdown set. It copies all eight validated child sets to a hidden staging
directory and atomically renames one aggregate directory containing
`campaign.json`, the captured policy and resume identity, and their required
sidecars. Each row binds all four quality fixtures across five repeats. A
missing worker, non-product field, stale
sidecar, duplicate coordinate, reused cold PID, or safety refusal leaves no
successful campaign directory. Intermediate row artifacts can support a
same-identity resume, but they are not complete campaign publications.

Before the parent or a worker loads a snapshot, inference commit
`c781c09d2789538d3cff459257274cbc22109b25` seals these exact model identities and the
candidate-only materialized-weight residency boundary used by the receipt producer.
SceneWorks mirrors the same rule when it validates a portable receipt: a local
snapshot path is not an identity, and `modelId`/`referenceModelId` must contain
the matching repository, immutable revision, architecture, and producer
inventory digest. A mismatched revision, family, candidate/reference role, or
native context fails receipt validation; no result may replace one arm with a
caller-selected model.

| family | candidate | higher-precision reference | native context |
| --- | --- | --- | --- |
| Llama | `mlx-community/Llama-3.2-3B-Instruct-4bit@7f0dc925e0d0afb0322d96f9255cfddf2ba5636e` | `mlx-community/Llama-3.2-3B-Instruct-bf16@6d88ba43024fef71b10e52e101c7cd4598322601` | 131,072 |
| Qwen | `mlx-community/Qwen3-1.7B-4bit@3b1b1768f8f8cf8351c712464f906e86c2b8269e` | `mlx-community/Qwen3-1.7B-bf16@9cd6692855d3e06772228e9a962b2606359b2d24` | 40,960 |

The producer deliberately fails closed if its numeric quality-reference hooks
cannot supply all four checked-in quality fixtures. It must never substitute a
synthetic zero-error or caller-authored quality result merely to publish a
matrix. The kernel fixture is a real MLX dense-attention dispatch checked
against a separately accumulated host-fp32 reference; 4-bit-versus-bf16 model
perplexity and agreement are recorded as raw dense-baseline characterization,
not mislabeled as kernel error. A dense row is never rejected on model
behaviour: the structured-tool and needle artifacts record, for both the
candidate and the bf16 reference, tool validity, needle recovery, and output
agreement as booleans (with output hashes in the binding), and
`quality.needleDiscriminating` / `quality.toolDiscriminating` state whether the
dense run itself recovered the needle / emitted the valid tool call. Missing weights, identity drift, incomplete runs, and non-nominal
thermal state still fail closed.

Frozen quality thresholds gate only compressed receipts, and only against the
dense-KV run on the **same weights**: every fixture's `independentReference` is
`dense-kv-same-weights:<model inventory>` and every fixture artifact's
`binding.reference` coordinate/quality inventory must equal the receipt's
`modelFileSha256` (a bf16 reference is refused as a compressed denominator, and a
dense receipt may not claim one). Readers re-derive each tool/needle metric and
flag from the recorded outcomes of every repeat. This isolates
the effect of KV compression from weight quantization. When the same-weights
dense run missed the needle, the compressed needle check requires exact
agreement with the dense output and the receipt carries
`needleDiscriminating: false`; likewise identical invalid tool calls record
`toolDiscriminating: false`. Each repeat's artifact records its own flag and the
receipt flag is the AND over all five repeats. `compare` reports such a row as
non-discriminating instead of counting a shared miss as agreement.

Every row, including memory-material and fit-boundary, is admitted by its
runtime guards rather than a static whole-process MLX peak proof, which lazy
long-context graphs cannot supply. The mandatory policy must configure the
supervised worker's deadline, sampling, termination grace, host free-RAM
reserve, and `phys_footprint` watchdog cap; the supervisor refuses before spawn
unless host available RAM covers cap plus reserve, and a conservative static
load-plus-KV floor above the cap still refuses. Host available RAM is the
`vm_stat` measure `darwin-vm-stat-available-v3`, Activity Monitor's "Cached
Files" model: `(free + speculative + purgeable + max(0, file-backed -
speculative)) * page size`. File-backed page cache, such as the model bytes the
parent just hashed, is reclaimable without the compressor or swap; anonymous
pages are never credited. Known limitation: file pages another process has
mapped count as available; the child's own mapped weights are bounded by its
`phys_footprint` cap. The live watchdog compares the same
available measure with the reserve, so page cache alone never aborts a row. Each receipt records
`memory.admission` (`runtime-guarded`, `rule: estimate-plus-reserve-v1`, the stated cap,
reserve, the static floor, the admission `estimateSource`/`estimateBytes` -- at least the floor,
at most the cap, and exactly the cap for `child-footprint-cap-fallback` -- and
`hostMemoryComponents`: every `vm_stat` component (including the audit-only inactive, anonymous, throttled and active pages) and the derived
file-cache credit and available bytes); the reader recomputes the measure and
requires it to cover the row's estimate plus the reserve (the rule the inference supervisor
admitted the row on; its cap and reserve watchdogs still abort a row that outgrows the
estimate), and requires that cap and reserve to equal the captured safety policy. A pre-spawn refusal, watchdog abort, or failed worker is written as
a sealed `logs/<row>.attempt-<n>.unaccepted.json` record with its reason and is
never an accepted row.

## Compressed rows (SC-20676)

`sc20671-kv-baseline parent --mode compressed --kv-method <method>` runs the same
frozen eight-row schedule, fixtures, and geometry with each row's KV held in the
method's compressed cache and fused decode attention (`group-affine`, the
SC-20675 packed 2-bit cache read by the SC-20676 Metal kernel, and
`group-affine-4` and `group-affine-8`, the same cache and reader with 4- and
8-bit codes; all group 32. Each method is bound to exactly one
`compression.bits` and `representationIdentity`, and a receipt naming any other
pair is refused). Each
worker runs the compressed arm, then a dense-KV reference arm on the **same
candidate snapshot**, and gates quality against it (contract v3). The resume
identity carries `mode: "compressed"` and `kvMethod`, so dense and compressed
rows can never resume into each other, and a published campaign is uniformly
one mode (`validateCampaign` reports `mode`/`kvMethod`).

A compressed receipt carries a `compression` block (dense receipts never do):
the method and cache-reported representation identity; fused call and dense
fallback counts with every fallback's operation and reason, summed over every
compressed-arm dispatch of the row (warmups, fixtures, and each coordinate's
prompt-cache-reuse and cancellation probes); `fullCacheDequantizations`; and
the classification and storage of the **coordinate operation alone**.

The block also names the fused reader's `kernelGpuFamily` and its `kernelPaths`:
every kernel the reader actually dispatched (`sc20676_split_kv_simdgroup`
per-row, `sc20676_tiled_multi_row_simdgroup_matrix` fp32 tiled, or
`sc20676_nax_tiled_matmul2d` Neural Accelerator), with its selection token, the
reason, the dispatched `queryDtype`, and the accepted calls, sorted by (kernel,
selection, dtype). The calls sum to `fusedCalls`; the NAX kernel appears exactly
with the `nax-selected` selection and 16-bit queries, the fp32 tiled kernel only
with `nax-unavailable`, `f32-query` (f32 queries), or `nax-head-dimension`, and
the per-row kernel only with `below-multi-row-threshold` (qualified family) or
`conservative-family`. A NAX and a non-NAX run therefore never produce the same
receipt.

`persistentKvRepresentation` is decided only from evidence recorded during the
coordinate operation (its setup and measured dispatch); the producer closes
that scope before the lifecycle probes run, so their reasoned fallbacks never
reclassify the row. It is `compressed` only when the coordinate's own cache
stayed on the fused path with no fallback, and `dense-fallback` otherwise (the
supported-batch decoder and the prefix-cache reuse path are dense today).

For a `compressed` coordinate the block records that cache's storage at its
persistent-KV peak: `deviceCodeBytes` and `deviceMetadataBytes` (the retained
MLX arrays), `hostPayloadBytes` (the allocated host copy of codes/metadata plus
the staged dense key tail), `physicalKvBytes` (exactly their sum: the whole
physical representation), and `storageTokens`. Readers require
`memory.persistentKvBytes` (the MLX device share, which the memory-containment
checks use) to equal `deviceCodeBytes + deviceMetadataBytes`, `storageTokens`
to equal `geometry.kvLength`, and `physicalKvBytes` to be below
`memory.denseTheoreticalKvBytes`. A `dense-fallback` coordinate claims no
compressed storage: all byte fields and `storageTokens` are zero and it must
list at least one fallback.

`compare` claims a persistent-KV reduction only for a `compressed` coordinate,
as `(dense persistentKvBytes - compressed physicalKvBytes) / dense
persistentKvBytes`, so the host copy and staged tail count against the claim.
A `dense-fallback` row reports `persistentKvReductionEligible: false`, a
`persistentKvReductionIneligibleReason`, and `persistentKvReduction: null`;
eligibility is read from the recorded representation, never from the row's
request or prefill mode. Readers also reject a compressed receipt with no
fused calls, an unreasoned or uncounted fallback, a dense full-cache
reconstruction (counted by the cache or witnessed by a
`full_cache_materialization`/`dense_cache_temporary` allocation event), or
physical bytes that do not reconcile. One-command GPU-window launch, from the
inference checkout:

```text
eval "$(scripts/fetch-prebuilt-mlx.sh --build-type Release)" && export PMETAL_MLX_PREBUILT_DIR PMETAL_METALLIB_PATH && \
SCENEWORKS_ROOT=/abs/SceneWorks cargo run --locked --release -p mlx-llm --bin sc20671_kv_baseline -- \
  parent --mode compressed --kv-method group-affine \
  --llama-snapshot <llama-4bit> --qwen-snapshot <qwen-4bit> \
  --llama-fp32-reference-snapshot <llama-bf16> --qwen-fp32-reference-snapshot <qwen-bf16> \
  --prompt-file <prompt.txt> --safety-policy <policy.json> \
  --resume-dir /abs/sc20676-compressed-resume --out /abs/sc20676-compressed-campaign
```

## Contract v3 change record

Contract v3 (`config/kv-baseline-quality-contract.json`) replaced v2 before any
compressed-row result existed, so it is not post-hoc threshold tuning; every
threshold number is unchanged.

- The v2 needle fixture asked the model to remember a "harmless passphrase".
  The dense Q4 Llama baseline answered with a safety refusal about recovering
  passwords while the bf16 reference answered; an independent MLX-LM run
  reproduced the identical refusal, so it was model behaviour triggered by
  credential-like wording. A needle check the dense baseline fails cannot
  detect KV-induced retrieval loss. v3 uses neutral RULER/NIAH-style wording
  ("The special magic identifier is {needle}." / "What is the special magic
  identifier mentioned in the text above? Reply with only the identifier.") and
  keeps the exact-match needle token.
- The v2 compressed gate compared the Q4 candidate with the bf16 reference, so
  it measured weight quantization rather than the KV cache and would reject
  every compressed candidate by construction. v3 gates compressed rows against
  the dense-KV run on the same weights; the bf16 model remains dense-row
  characterization only.
- v3 greedy agreement is teacher-forced (decided before any compressed result
  existed). At each position of the reference kernel stream, the candidate's
  own greedy choice is compared with the reference token, the forced token is
  fed back, and the denominator is the reference length. A free-running
  comparison would count every token after one argmax flip as a mismatch.
  Receipts record `quality.greedyAgreementMethod: "teacher-forced"`, and the
  free-running first divergence is recorded separately
  (`quality.freeRunningFirstDivergence` and the kernel fixture evidence) as an
  observation. The reference stream keeps its stop token, so the stop position
  counts: the candidate agrees there when it also chooses a stop token. The
  gate is the minimum over the five repeats
  (`quality.greedyTokenAgreementByRepeat` records each), and the forced pass
  runs once per distinct reference stream. The contract JSON text and every
  threshold (0.999) are unchanged.

## Contract v4 change record

Contract v4 changes one metric, `multiTurnPromptCache`. It was changed **after
compressed results were visible**: the A2 2-bit and 4-bit campaigns and the
K8V8 (8-bit) single-row run. It was changed because the v3 metric did not
measure what its name claimed, not to move a result.

What v3 measured:

- The v3 `multi-turn-prompt-cache` fixture was a single turn, and it was cold.
  Its quality output came from the ordinary observed generation, with no prompt
  cache in either arm. It never exercised multiple turns or prompt-cache reuse.
  The name was wrong.
- The score was a free-running, position-by-position exact match over 64
  tokens, with threshold 1.0. One argmax flip at a near-tie made every later
  token count as a mismatch.
- On `llama-memory-material-single-single-shot-warm`, the 4-bit and 8-bit rows
  both scored 0.5625 (36/64). That is one flip at generated token 34 (" baseline"
  vs " context"), plus two tokens that matched by chance.
- The same-weights dense run flips between those two branches from one build to
  the next. Inference `aa1ea42c0` picks " baseline"; `c7b1553c0` and
  `4b9adc47d` pick " context". The 8-bit compressed output is byte-identical to
  the dense output of `aa1ea42c0`. So the metric would fail dense against
  itself across builds.

What v4 measures:

- **Fixture sizing.** Turn 1 uses the row's context-band payload, but the
  multi-turn fixture caps it so turn 2 plus the 1024-token continuation fits
  the native window. The cap is native − 1024 − 64 (turn-1 answer) − 512
  (prompt text, chat template, follow-up, re-tokenization slack).
  - Only the fit-boundary band is capped: Llama 130,560 → 129,472 payload
    tokens, Qwen 40,448 → 39,360.
  - Short, medium and memory-material keep their full band payload.
  - The row's coordinate operations always keep the full band prompt, so the
    row's measured geometry is unchanged.
  - Each turn's rendered prompt tokens are recorded in its turn record.
  - Preflight refuses a row whose turn 2 cannot leave 1024 tokens before any
    model loads. The provider refuses it again before decoding.
- **Two real turns.** Turn 1 is the frozen cache-fixture prompt. It is
  generated and stored through the product prompt cache (`PrefixCache`). Turn 2
  is turn 1's conversation, turn 1's answer, and a fixed follow-up
  (`multiTurnFixture.followUp`). Turn 2 is served by a prompt-cache hit over
  turn 1.
- **The same flow in both arms.** A compressed arm imports the reused prefix
  into its compressed cache by quantize-on-append. A dense arm seeds its dense
  cache. Turn 1 is dense in both arms, because the provider prefix store holds
  shared prefixes as dense K/V.
- **Cache records, fail closed.** Each arm records every turn's cache hit,
  reused-prefix tokens, and prompt token digest (`quality.multiTurnCache`,
  which must equal the primary repeat's sealed `turns` evidence). Turn 1 must
  miss its fresh store. Turn 2 must be served by a hit that reuses at least the
  prompt it shares with turn 1. Anything else refuses the row.
- **Scored like greedy agreement.** The score is teacher-forced, per-position
  argmax agreement on turn 2, taken after that turn's own prompt-cache hit
  (`quality.multiTurnPromptCacheMethod`).
  - Compressed rows: the same-weights dense-KV session gives a turn-2 forced
    continuation of 1024 tokens, greedy, with stop tokens decoded through. It is
    always 1024 tokens: a row whose turn 2 cannot leave them is refused, never
    shortened. The compressed session is teacher-forced on it.
  - The continuation is measured once per row, not once per repeat. That one
    measurement is copied into every measured repeat
    (`quality.multiTurnForcedContinuation`, and each repeat's fixture evidence),
    so every repeat records the same `multiTurnPromptCache`.
  - Both sessions' turn records for that pass are sealed
    (`quality.multiTurnForcedPass`). The two sessions must have rendered the same
    turn-2 prompt token ids (`promptSha256`), and that prompt must be the
    fixture's own turn-2 prompt. Otherwise the row is refused. The two arms of a
    compressed row's observed fixture must render the same turn-2 prompt too.
  - Dense rows: the reference repeat's natural turn-2 stream.
- **Threshold aligned with greedy agreement.** The threshold is 0.999. Over 1024
  positions that allows one flip. Every measured repeat must meet it.
- **Observations, never gated.** Turn 2's free-running first divergence
  (`quality.multiTurnFreeRunningFirstDivergence`) and matched-prefix length
  (`quality.multiTurnMatchedPrefixTokens`) are recorded but not gated.

Every other threshold, and all other contract wording, is unchanged from v3.
Receipt schema v7 (`sc-20671-kv-baseline-v7`) carries the new fields. The v4
contract hash is the compatibility fence: any receipt bound to v3 is refused, so
every campaign row is re-measured under v4.

## Measurement paths (compressed rows)

`compression.measurementPaths` (additive; absent on receipts produced before inference
fix/sc-20669 measurement paths) names the KV path each block of a compressed row ran on:
`memory` (memory, `compression` storage and `persistentKvRepresentation`) and
`prefillFirstToken` (`timings.prefillMs`, `ttftMs`, `firstTokenMs`) are the coordinate operation
(`supported-batch` with 2 sequences on a batch row, else `chunked-prefix-reuse` or
`single-shot-generation`), whose `kvPath` is the row's `persistentKvRepresentation`;
`decodeTiming` (`timings.decodeTokensPerSecond`, the fixed-length steady decode) and `quality`
are always one sequence on the compressed reader. A supported-batch row therefore records a
`dense-fallback` memory path beside a `compressed` decode and quality path. Readers refuse any
other value.

## Contract v5 change record

Contract v5 (`config/kv-baseline-quality-contract.json`, hash
`8461b072e36493f4f4559e7fc858a286a239a43e8ca8d8ea93179bcf33f344ca`, recorded in
its own `changeRecord`) keeps every v4 threshold. It was made after compressed
A2 results (run 36923403549) were visible; each change is a measurement
correction, not a relaxation:

- **Quality is measured once per arm per process.** Every fixture, forced
  continuation and teacher-forced pass is deterministic within a process: run
  36923403549 sealed five identical quality repeats (evidence and metrics) for
  all four fixtures of all 16 rows, so repeating them re-measured nothing. A row
  now runs its timing first (a warm row's two warmups, then the five repeats:
  the kernel fixture's coordinate operation and a fixed-length steady decode,
  feeding the coefficient-of-variation rule), then one quality measurement per
  arm. The candidate's kernel fixture reuses timing repeat 0's coordinate
  operation (same prompt, operation and session); every other fixture runs its
  own coordinate operation once. Receipts record `quality.statistics.qualityMeasuredOnce:
  true`, `greedyTokenAgreementByRepeat` and the gate carry one measurement
  (`repeat` 0), and a set publishes only `fixtures/<name>.json` (no
  `fixtures/repeat-N/`). Cross-process variation of the dense arm is real; the
  dense noise-floor control (inference `sc20671-kv-baseline noise-floor-parent`,
  kv-poc phase `nf`) quantifies it rather than in-process repeats hiding it.
- **A non-discriminating needle is an observation.** When the same-weights dense
  run misses the needle, `needleRetrieval` records the compressed run's own exact
  recovery, `needleDiscriminating` is false, both outputs stay in the artifact,
  and the gate never evaluates it (a gate recording it is refused). When the
  dense run recovers the needle, the compressed run must recover it exactly, as
  before. Agreement with a dense miss text measured free-running continuation
  agreement after the miss, which one near-tie argmax flip breaks.
- **`perplexityDelta` is scored on the dense reference's tokens** (inference
  4bd0e2429): both arms' mean per-token negative log-likelihood of one stream,
  the candidate teacher-forced on it. v4 producers scored each arm's own
  free-running stream, so after one flip they compared two texts (llama
  fit-boundary group-affine-8: 479/479 teacher-forced agreement beside a 0.290
  delta).

The v5 contract hash is the compatibility fence: v4 receipts are refused, so
every campaign row is re-measured under v5.

## Measured quality gate (compressed rows)

A compressed row whose quality is validly measured but misses a frozen
threshold is evidence for the Go/No-Go decision (sc-20678), not an invalid run.
The row is accepted and records `quality.qualityGate`:
`{passed, failures: [{metric, fixture, repeat, value, threshold, comparison}]}`.
The gate evaluates every measured repeat against the contract v4 thresholds
(`greedyTokenAgreement`, `perplexityDelta`, `structuredToolAgreement`,
`needleRetrieval`, `multiTurnPromptCache`); `passed` is true only when no repeat
missed. Validators reject a gate that omits, edits, reorders, or claims a pass
over any failing value. Campaign readers re-derive every repeat's metrics from
the sealed fixture artifacts and require the gate to equal their evaluation.
`compare` reports `qualityGatePassed` and the failures. The campaign manifest and
summary carry each row's outcome, and a compressed campaign is
`qualityGatePassed: false` when any row failed. Dense rows carry no gate.

Some failures still refuse the row: kernel parity (`parityMaxError`, which
compares the fused reader with its independent host-fp32
dequantize-then-attend reference over the stored codes), fixture-evidence and
binding integrity, identity, silent fallback, dense reconstruction, and a
discrimination flag its outcomes do not derive. Kernel parity is a correctness
check of the kernel, not a quality-versus-dense metric. Each refusal names its
metric, value, threshold, and fixture.

Greedy agreement on compressed rows is measured over a forced continuation, not
the short natural fixture streams. Over ~16 natural tokens, one argmax flip is
already 0.9375, so the frozen 0.999 would mean zero flips. The same-weights
dense-KV session greedily continues the kernel fixture prompt through every stop
token for 1024 tokens. A fit-boundary row uses the window left after the prompt,
never fewer than 256 tokens. This runs once per row because the stream is
deterministic. The compressed session is teacher-forced on that stream in one
pass and compared by argmax at every position. `quality.forcedContinuation`
records the length, matches, agreement, flip count, the first 32 flip positions,
and stream digests. `greedyTokenAgreement` is that agreement, shared by every
repeat. The free-running first divergence stays an observation. The threshold
(0.999) is unchanged; over 1024 positions it allows one flip.

## Receipt v6: recorded, not refused (sc-20671 hardware audit)

Receipt schema v6 (`sc-20671-kv-baseline-v6`) records real-hardware observations
that v5 refused rows on:

- **Host state.** Power mode and `NSProcessInfo.thermalState`
  (nominal/fair/serious/critical) are recorded at row start, after every timing
  sample (`timings.samples[].hostState`), and at row end. Each observation also
  carries the verbatim `pmset -g therm` text and its `CPU_Speed_Limit`. Only a
  throttled row start refuses the row (serious/critical, or CPU_Speed_Limit
  < 100). Later changes are recorded as `provenance.thermalChangedDuringRow` /
  `powerModeChangedDuringRow`. Unknown pmset note lines are tolerated. Power and
  thermal state are not part of the campaign's global identity; the manifest
  instead records `hostStateVaried`.
- **Memory-material share.** `memory.denseKvShareBps` (dense KV over the
  prefill-peak footprint) is recorded. `memory.belowMemoryMaterialShare` flags a
  memory-material row below 10% (the band is defined by geometry).
- **Post-release allocator slack.** MLX active/cache may exceed the
  weights-loaded boundary by `max(1 MiB, 0.1% of baseline)`. The tolerances and
  the actual residuals (`mlxActiveResidualBytes`, `mlxCacheResidualBytes`) are
  recorded, and a larger residual still fails as a leak.
- **Block growth.** The producer's dense cache records the retired pre-growth
  buffer as the transient coexisting with the grown persistent cache. The peak
  floor `baseline + persistent + transient` therefore counts the new buffer
  once, including on chunked rows that land on a 256-token boundary.

The v3 contract hash is the compatibility fence: receipts bound to v2 are
refused, and v5 receipts produced under v3 carry the required
`quality.needleDiscriminating`, `quality.toolDiscriminating`, and
`memory.admission` fields. The inference producer keeps a byte-exact copy of the
contract (`crates/llm/mlx-llm/testdata/`) whose hash and needle wording its tests
pin.

## Product compressed KV (sc-20682)

The POC's qualified K8V8 rows ship as an **opt-in** on SceneWorks' local LLM
generations. The setting is off by default:

- `SCENEWORKS_LLM_KV_COMPRESSION` (`off` | `qualified`) is the worker-wide
  default. Unset, empty or unrecognized is `off`.
- A prompt-refine request may set `kvCompression` (`off` | `qualified`) on
  `POST /api/v1/prompts/refine`. The API refuses any other value and forwards it
  as the job payload's `kvCompression`, which overrides the worker default.
  Catalog vision analysis and StarVector requests carry the worker default.

`qualified` never forces compression. The inference engine's qualification
table (`KV_COMPRESSION_QUALIFICATIONS`) has two K8V8 (`group-affine-8`) rows,
each measured on one model. A generation runs compressed only when it is a
single sequence and matches one of them:

| Measured model | Decoder it applies to | Prompt | Final context (prompt + max new tokens) |
| --- | --- | --- | --- |
| Llama-3.2-3B-Instruct | Llama | ≥ 32 768 | < 130 560 |
| Qwen3-1.7B (dense) | dense Qwen3 | ≥ 10 240 | ≤ 40 960 |

Batched requests, prompts below the minimum, final contexts past the maximum and
every other decoder run dense, deterministically, with a stable reason
(`batched_decode`, `below_minimum_context`, `above_qualified_context`,
`unqualified_model`, ...).

SceneWorks keeps no LLM KV pricing of its own. Request memory admission is the
in-process engine's, and it prices the cache the request actually runs on:

- Dense K/V for a dense plan.
- The format-derived compressed bytes for a qualified plan. These cover packed
  codes, scale/zero per group, residual rows and the fused path's transients;
  see the inference repo's `docs/reference/qwen38/native-memory-admission.md`.
- A request whose compressed selection falls back to dense is re-admitted at the
  dense price.
- A mid-generation dense transition is admitted against fresh memory, failing
  typed (`RequestResourceExhausted`) rather than oversubscribing.

Every prompt-refine result carries `generation.kvCache`. The worker emits the
same block as the `llm_kv_cache` telemetry event for every local LLM generation:
prompt refine, catalog vision analysis and StarVector. The event carries
`jobId` and `engine` (the lane, `mlx` or `candle`). It never includes prompt or
output text.

The block's `outcome` says how the generation ended. A `completed` one carries
the engine's report. A generation the engine ends with an error still emits the
event (sc-20688), with the requested `policy`, `reported: false`, a stable
`reason` and the engine's `error` message:

- `refused`: request memory admission refused it while pricing the dense or
  compressed cache (`request_resource_exhausted`). It adds a `refusal` object
  with the prompt and generation token counts, the context limit, and the
  required and available bytes.
- `canceled`: the request was canceled.
- `failed`: any other engine error (`unsupported`, `invalid_request` or
  `engine_error`). StarVector, JoyCaption and other multimodal-wrapped decoders have
no qualification-table family, so they always report dense, with
`policy_disabled` or `unqualified_model`.

| Key | Meaning |
|---|---|
| `policy` | `off` or `qualified`, as the job resolved it. |
| `reported` | `false` when the provider reports no KV cache. In that case, no other key follows. |
| `formatVersion` | The engine's `KV_CACHE_FORMAT_VERSION`. |
| `format` | `group-affine-k8v8`, or `null` when the generation ran dense throughout. |
| `ranCompressed` | `true` only when the whole generation ran on the compressed cache. |
| `fallbackReason` | A `KvCacheFallbackReason` id, or `null` exactly when `ranCompressed` is true. The ids are `policy_disabled`, `unqualified_model`, `unsupported_request`, `batched_decode`, `below_minimum_context`, `above_qualified_context`, `unsupported_geometry`, `reader_unavailable`, `runtime_fallback` and `dense_gather`. |
| `detail` | The engine's operation and reason words, or `null`. |
| `counters` | `fusedAttentionCalls`, `denseFallbackEvents`, `fullCacheDequantizations`, `denseGatherFallbacks` and `compressedCacheBytes`. |
