# Candle capture repairs for sc-23026

These changes address individual captures in
[run 34493340342](https://github.com/SceneWorks/SceneWorks/actions/runs/34493340342),
which started on SceneWorks `fc3c6dc81` with inference `e11fd9f0f`.
The running checkout, executable, cache and evidence branch are unchanged.

## Repair boundary

| Failure | Repair | Verification before another campaign |
| --- | --- | --- |
| Flux2 dev and Klein: request changed after admission | In inference, consume the original request's admission before cloning it to resolve the seed. Keep rejection of copied, mutated and replayed requests. | Registered-generator admission regression through `generate`, canceled before weight materialization. |
| MiniMax q4/q8: provider reports BF16 identity | In inference, resolve memory accounting's reference partition using the loader's staged-DiT sibling rule. | Sparse split-install fixtures with a larger, irrelevant upstream dense reference partition; check both byte accounting and q4/q8 identity. |
| MiniMax: `stagedTierArtifact` rejected by schema | Emit the schema's standard load-root artifact; preserve staged repository/revision/tier and selected partition in the loadability fingerprint. | Adapter serialization checks. |
| InstantID: invalid status | Emit `gated`, a measured-tuple sweep and `quality.result=not_run`; retain observed peaks and explicitly mark unexecuted lifecycle scenarios. | Fragment regression checks against schema properties. No promotion claim is added. |
| Kolors: Windows main-thread stack overflow | Reserve 32 MiB for the Candle adapter executable with a binary-specific MSVC linker argument. | Native build and PE stack-reserve inspection. A real Kolors capture remains necessary to establish that this resolves the observed overflow. |
| Bernini still: unsupported geometry/request | Plan the admitted 512×512 still geometry and explicitly send `video_mode=t2i`, `frames=1`. | Plan/request regression across all three tiers. |
| PuLID: no face detected | Bundle the existing generated face used by inference's successful sc-16956 PuLID renders. Version the fixture name with `face-sc16956`. | Decode, dimensions and SHA-256 checks; reject the old gradient fixture identity. |
| SANA base/Sprint: wrong request evidence | Read the provider's `REQUEST_EVIDENCE_REVISION` rather than sending a generic story/pin token. | Both routes tested against the linked provider constant. |
| Mage and Lens: config-only component directories | Require nonempty safetensors files before classifying these components as runnable; also check Mage component shard indexes. | Missing weights, incomplete shards and repaired-directory classification tests. |

## Remaining host or route work

- Two Mage files are truncated on the running host. At inspection, the BF16
  transformer was 6,138,102 bytes versus 8,231,536,784 bytes declared by its own
  header; the shared q8 text encoder was 90,319,125 versus 4,719,576,437 bytes.
  These need artifact repair after the current campaign releases the cache.
  The preflight change detects absent weights, not arbitrary corruption or
  checksum mismatches in present files.
- The Lens q4 and Lens-Turbo q8 transformer directories contain only
  `config.json`. The next `--download-missing` walk can now classify and fetch
  them rather than reporting a missing provider identity after loading.
- Bernini's video probe asks for `text_to_video`, but the pinned memory
  contract admits reference/clip-conditioned video routes. LTX-2.3 similarly
  plans T2V and omits the packed quant assertion its contract requires; that
  contract admits I2V, first/last and specific editing routes. Correcting these
  requires a deliberate representative-probe change with matching plan,
  conditioning and provider receipts, not weakening admission or relabeling a
  rejected T2V probe as measured evidence.
- The campaign is still running. This is an interim failure inventory, not the
  story's final residue enumeration or evidence closeout.

## Integration order

The inference fixes are prepared on `codex/sc-23026-capture-repairs` in the
inference repository. Land those first, then update SceneWorks' inference pin
to their merged main revision together with these adapter fixes. No local-only
inference revision is pinned here. Resume measurements using the successful
evidence already collected, after repairing the truncated artifacts. These
patches do not themselves establish new GPU measurements or close sc-23026.
