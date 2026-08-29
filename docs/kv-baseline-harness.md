# Dense KV baseline harness (SC-20671)

`scripts/kv-baseline-harness.mjs` defines the sealed receipt boundary for the
compressed-KV POC. A producer supplies model/toolchain provenance, geometry,
Darwin `phys_footprint`, MLX allocator readings, timings, quality thresholds,
lifecycle outcomes, cancellation cleanup, and allocation events. The harness
rejects incomplete runs, missing model identity or thermal state, failed
cleanup, and any detected full-cache temporary.

The receipt shape is described by
`packages/schemas/kv-baseline-receipt.schema.json`. Producers should write
receipts atomically under `docs/calibration/` with `record`; comparison reports
are generated with `compare`:

```text
node scripts/kv-baseline-harness.mjs record input.json receipt.json
node scripts/kv-baseline-harness.mjs compare dense.json compressed.json comparison.json
```

`readDarwinMemory()` is the only platform-specific reader: it obtains the
kernel-maintained `phys_footprint` value through `footprint -p`. MLX values are
inputs because the producer owns the MLX session and must sample active,
cached, and peak allocator counters at the defined phase boundaries. The
comparison refuses mismatched SceneWorks/inference revisions or geometry and
reports KV reduction, process-footprint delta, throughput ratio, and quality
values. It is deliberately not a real-weight campaign runner; the final device
matrix belongs to the terminal epic story.
