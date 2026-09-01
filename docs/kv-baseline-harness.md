# Dense KV baseline harness (SC-20671)

`scripts/kv-baseline-harness.mjs` defines the sealed receipt boundary for the
compressed-KV POC. A producer supplies model/toolchain provenance, matrix
coordinate, geometry, Darwin `phys_footprint` and `phys_footprint_peak`, MLX allocator readings,
timings, the checked-in quality/statistical contract, complete lifecycle
outcomes, cancellation cleanup, and typed allocation events. The harness
rejects incomplete runs, identity drift, a missing or hash-mismatched model
file at record time, any non-nominal thermal state, failed cleanup, and any
detected full-cache temporary in a compressed run. Dense receipts retain the
product's measured immutable-concat coexistence as deliberate baseline evidence.

The receipt shape is described by
`packages/schemas/kv-baseline-receipt.schema.json`. Producers write each receipt
as an atomically renamed **directory** under `docs/calibration/`; a set contains
`receipt.json`, `receipt.md`, their exact-byte sidecars, and the four required
`fixtures/<name>.json` artifacts with sidecars that name those full relative
paths. The Markdown embeds the JSON semantic hash, and readers rehash and parse
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
immutable dense concat reports only the old-plus-incoming transient overhead
above its retained merged successor, sequential token/layer events reconcile
by their maximum, and the persistent successor is counted once. A
`product-cache_release` event with lifetime `released` records exact released
K+V ownership. It is required lifecycle evidence and is excluded from both
workspace attribution and full-cache-temporary detection.
comparison refuses mismatched source/model/toolchain/hardware/power/thermal
identity, matrix, contract, or geometry and reports KV reduction,
decode-steady/lifetime-peak process-footprint deltas, and throughput ratio.
`campaign` requires sealed
receipts for both Llama and Qwen across every declared context/request/prefill
and cold/warm coordinate. No campaign result is checked in yet: the real
matrix remains open work in SC-20671, not a terminal-epic-only task.

The inference producer is the standalone `sc20671-kv-baseline` executable. Its
`parent` mode consumes no caller-supplied coordinate list: it reads the frozen
2 × 4 × 2 × 2 × 2 matrix, forks a new child for every cold row, and requires an
in-process warm-up in each warm worker. Workers collect identity, geometry,
Darwin/MLX phase samples, allocation events, timings, output, prefix reuse, and
cancellation from the product route; the parent accepts only the child’s sealed
JSON/Markdown set. It copies all 64 validated child sets to a hidden staging
directory and atomically renames one aggregate directory containing
`campaign.json` and its sidecar. A missing worker, non-product field, stale
sidecar, duplicate coordinate, or reused cold PID leaves no campaign directory.

Before the parent or a worker loads a snapshot, inference commit
`98adfde3e1aec3c31d39ec7442af7d7f5917db28` seals these exact model identities.
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
matrix; a successful real-model campaign remains the final SC-20671 gate.
