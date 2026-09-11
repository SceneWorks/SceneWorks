# MiniMax-H3 load-error fix adoption (sc-23053)

SceneWorks adopts inference `8fb95eb05cee4cf1e482aa050874fae98ccb3f78`
([PR #970](https://github.com/SceneWorks/inference/pull/970)) and MLX fork
`d5a7fc018d713a37091e1cd102873eab355a00c6`
([PR #29](https://github.com/SceneWorks/mlx-rs/pull/29)).

An asynchronous native weight read could fail while host evaluation reported
success. Controlled EFAULT injection reproduced all-zero H3 embeddings. The
native fix transports the read exception to the host, preserves file/offset/errno,
handles interrupted and partial reads, and isolates independent arrays. H3 now
checks dense CPU embedding values before verifying their GPU view. A failed
context still stops generation; the worker does not silently retry the request.

Validation on the development Mac covered all 50 text layers for text-only and
grounded conditioning, fresh and warm encoders, bf16 deferred/resident and q4/q8
deferred tiers. All 16 forwards were finite and nonzero, with exact fresh/warm
agreement. Injecting failures into all 47 embedding reads now returns a
recoverable error. Native regression tests cover EOF, EFAULT, EINTR, partial
reads, sync/async CPU/GPU evaluation and independent-array isolation. The full
inference CI passed on head `233c7273615e8476573fb34f36282ef15257f987`
([run 34595278981](https://github.com/SceneWorks/inference/actions/runs/34595278981)).

The original job did not capture read status or a CPU buffer view, so its exact
cause remains unproven. Fresh encoder construction is not a claim of a cold OS
file cache. These checks validate conditioning, not a complete rendered video
or installation of a new desktop build. Detailed source evidence is in the
pinned inference repository's `docs/diagnostics/sc-23053-mlx-load-errors.md`.

## Memory records and provenance

The pin's changed inference files have an empty intersection with the loader
closures for the four retained Candle currency attestations (Krea Turbo q4 and
Z-Image Turbo bf16/q4/q8); their existing bounded claims extend to this pin.
Ten MLX attestations are removed and their records derive currency at their
original measurement revisions. The native loading path changed, and the H3
checks are not replacement memory witnesses for those other models. Historical
measurements remain intact. The static memory matrix was regenerated from these
records to update its currency labels; no calibration campaign or disabled gate
was run.
The loader digest deliberately excludes external dependency versions; the
provider digest records the MLX dependency change.

The source/license inventory was regenerated. Its candidate population is
unchanged (683 candidates, 97 crate prefixes). Existing report-only licensing
findings remain the separate compliance worklist.
