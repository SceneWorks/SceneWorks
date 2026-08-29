# Dense KV baseline harness (SC-20671)

`scripts/kv-baseline-harness.mjs` defines the sealed receipt boundary for the
compressed-KV POC. A producer supplies model/toolchain provenance, matrix
coordinate, geometry, Darwin `phys_footprint` and `phys_footprint_peak`, MLX allocator readings,
timings, the checked-in quality/statistical contract, complete lifecycle
outcomes, cancellation cleanup, and typed allocation events. The harness
rejects incomplete runs, identity drift, a missing or hash-mismatched model
file at record time, any non-nominal thermal state, failed cleanup, and any
detected full-cache temporary.

The receipt shape is described by
`packages/schemas/kv-baseline-receipt.schema.json`. Producers should write
receipts atomically under `docs/calibration/` with `record`; comparison reports
are generated with `compare`:

```text
node scripts/kv-baseline-harness.mjs record input.json receipt.json
node scripts/kv-baseline-harness.mjs compare dense.json compressed.json comparison.json
```

`record` requires the local-only `provenance.modelFilePath` input and verifies
its byte count and SHA-256 before removing the path from the portable receipt.
Both `record` and `compare` atomically generate the JSON artifact, a Markdown
human receipt with the same basename, and exact-byte SHA-256 sidecars. The
checked-in quality contract is itself verified against its exact-byte sidecar
before the module accepts any receipt.

`readDarwinMemory()` is the only platform-specific reader: it obtains the
kernel-maintained current and lifetime-peak physical footprints through
`footprint -p`. MLX values are
inputs because the producer owns the MLX session and must sample active,
cached, and peak allocator counters at the defined phase boundaries. The
comparison refuses mismatched source/model/toolchain/hardware/power/thermal
identity, matrix, contract, or geometry and reports KV reduction,
decode-steady/lifetime-peak process-footprint deltas, and throughput ratio.
`campaign` requires sealed
receipts for both Llama and Qwen across every declared context/request/prefill
and cold/warm coordinate. No campaign result is checked in yet: the real
matrix remains open work in SC-20671, not a terminal-epic-only task.
