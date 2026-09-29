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
`footprint -p`. MLX values are
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
additional usable context. Existing exact-length v4 receipts remain readable.

`compare` refuses mismatched source/model/toolchain/hardware/power/thermal
identity, matrix, contract, or geometry and reports KV reduction,
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
`quality.needleDiscriminating` states whether the dense run itself recovered the
needle. Missing weights, identity drift, incomplete runs, and non-nominal
thermal state still fail closed.

Frozen quality thresholds gate only compressed receipts, and only against the
dense-KV run on the **same weights** (every fixture's `independentReference` is
`dense-kv-same-weights:<model inventory>`; a bf16 reference is refused as a
compressed denominator, and a dense receipt may not claim one). This isolates
the effect of KV compression from weight quantization. When the same-weights
dense run missed the needle, the compressed needle check requires exact
agreement with the dense output and the receipt carries
`needleDiscriminating: false`; `compare` reports that row as
non-discriminating instead of counting a shared miss as retrieval.

Every row, including memory-material and fit-boundary, is admitted by its
runtime guards rather than a static whole-process MLX peak proof, which lazy
long-context graphs cannot supply. The mandatory policy must configure the
supervised worker's deadline, sampling, termination grace, host free-RAM
reserve, and `phys_footprint` watchdog cap; the supervisor refuses before spawn
unless host free RAM covers cap plus reserve, and a conservative static
load-plus-KV floor above the cap still refuses. Each receipt records
`memory.admission` (`runtime-guarded`, the stated cap, reserve, and the static
estimate). A pre-spawn refusal, watchdog abort, or failed worker is written as
a sealed `logs/<row>.attempt-<n>.unaccepted.json` record with its reason and is
never an accepted row.

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

The v3 contract hash is the compatibility fence: receipts bound to v2 are
refused, and v4 receipts produced under v3 carry the required
`quality.needleDiscriminating` and `memory.admission` fields.
