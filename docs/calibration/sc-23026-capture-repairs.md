# Candle capture repairs for sc-23026

These changes address individual captures in
[run 34493340342](https://github.com/SceneWorks/SceneWorks/actions/runs/34493340342),
which started on SceneWorks `fc3c6dc81` with inference `e11fd9f0f`.
The campaign checkout, executable and evidence branch were left unchanged. The two
truncated Mage cache blobs are repaired separately, with their original bytes
retained as backups.

## Repair boundary

| Failure | Repair | Verification before another campaign |
| --- | --- | --- |
| Flux2 dev and Klein: request changed after admission | In inference, consume the original request's admission before cloning it to resolve the seed. Keep rejection of copied, mutated and replayed requests. | Registered-generator admission regression through `generate`, canceled before weight materialization. |
| MiniMax q4/q8: provider reports BF16 identity | In inference, resolve memory accounting's reference partition using the loader's staged-DiT sibling rule. | Sparse split-install fixtures with a larger, irrelevant upstream dense reference partition; check both byte accounting and q4/q8 identity. |
| MiniMax: `stagedTierArtifact` rejected by schema | Emit the schema's standard load-root artifact; preserve staged repository/revision/tier and selected partition in the loadability fingerprint. | Adapter serialization checks. |
| InstantID: invalid status | Emit `gated`, a measured-tuple sweep and `quality.result=not_run`; retain observed peaks and explicitly mark unexecuted lifecycle scenarios. | Fragment regression checks against schema properties. No promotion claim is added. |
| Kolors: Windows main-thread stack overflow | Reserve 32 MiB for the Candle adapter executable with a binary-specific MSVC linker argument. | Native build and PE stack-reserve inspection. A real Kolors capture remains necessary to establish that this resolves the observed overflow. |
| Bernini still: unsupported geometry/request | Plan the admitted 512×512 still geometry and explicitly send `video_mode=t2i`, `frames=1`. | Plan/request regression across all three tiers. |
| Bernini video: unsupported T2V request and 49-frame geometry | Use a 45-frame `reference_to_video` probe with one `MultiReference`, `video_mode=r2v` and the provider's content-sealed conditioning/adapter receipts. Version the fixture identity. | The actual provider safety check and request scope accept all three tiers and reject a changed reference or 49 frames. |
| LTX-2.3: missing packed quant assertion and unsupported T2V request | Load explicit q4/q8 and measure I2V with one fitted reference and its conditioning receipt. In inference, admit artifact-checked q8 in the public loader and retain the loaded numeric tier through safety and request scope. | Public loader and generator regressions accept matching q4/q8 and reject crossed tiers and missing conditioning. |
| Reference captures lose conditioning cardinality | Add optional `referenceCount` to plan and record schemas; preserve it through planning, exceeded bounds, typed records and anchor validation. | Schema, identity-hash, extraction and typed-record regressions. Older records remain unchanged. |
| PuLID: no face detected | Bundle the existing generated face used by inference's successful sc-16956 PuLID renders. Version the fixture name with `face-sc16956`. | Decode, dimensions and SHA-256 checks; reject the old gradient fixture identity. |
| SANA base/Sprint: wrong request evidence | Read the provider's `REQUEST_EVIDENCE_REVISION` rather than sending a generic story/pin token. | Both routes tested against the linked provider constant. |
| Mage and Lens: config-only component directories | Require nonempty safetensors files before classifying these components as runnable; also check Mage component shard indexes. | Missing weights, incomplete shards and repaired-directory classification tests. |

## Remaining host or route work

- Two Mage files were truncated on the running host. At inspection, the BF16
  transformer was 6,138,102 bytes versus 8,231,536,784 bytes declared by its own
  header; the shared q8 text encoder was 90,319,125 versus 4,719,576,437 bytes.
  Replacements were fetched at the exact same revisions into a separate staging
  directory and checked against the publisher's SHA-256 and safetensors length.
  Installation acquired exclusive access to each original blob and preserved
  `.corrupt-sc23026` backups; snapshot symlinks keep the same content address.
  Both installed files passed SHA-256 readback through those symlinks, and a
  subsequent header/length audit passed for all eight safetensors files in the
  two affected snapshots.
  The preflight change detects absent weights, not arbitrary corruption or
  checksum mismatches in present files.
- The Lens q4 and Lens-Turbo q8 transformer directories contain only
  `config.json`. The next `--download-missing` walk can now classify and fetch
  them rather than reporting a missing provider identity after loading.
- The corrected Bernini/LTX probes measure reference-conditioned video. Their
  changed modes, reference counts and fixture names keep the new workload
  distinct from the old rejected T2V requests. They still need real captures.
- The shared LTX loader change makes three older LTX-2.5 Candle anchors stale
  in the regenerated matrix; this patch does not extend those measurements.
- The campaign finished all 133 scheduled cells: 47 committed, 85 capture failures,
  and one runtime-budget stop. All 47 commits reached
  `story/sc-22738-candle-campaign-34493340342` at `76807148c1ff30889764adeaeaa0810566c3fa02`.
  The missing results PR was a repository Actions-permission warning; the capture
  failures caused the final nonzero exit. The evidence fold remains separate.

## Late failures and follow-up repairs

| Failure class | Cells | Repair / disposition |
| --- | ---: | --- |
| SD3.5 component storage dtype rejected | 9 | Accept F16/BF16/F32 component storage that the loader converts, while pricing the realized dtype. Integer components and crossed packed transformer tiers remain refused. |
| SenseNova reports no decode phase | 13 | Emit the decode boundary before pixel-space output is converted to RGB8 and transferred to the host. No completion requirement is relaxed. |
| SenseNova missing or truncated weights | 5 | Restore the exact pinned artifacts; detect config-only roots and safetensors header/length mismatches during preflight. Base BF16 also lacked its shard index and tokenizer files. |
| Wan TI2V uses A14B receipt inventory | 3 | Use TI2V's standard memory contract and an adapter revision token, rather than opening the unrelated A14B receipt inventory. |
| Wan T2V omits prepared memory contract | 2 | Publish the sealed contract from both A14B public loaders. Regression covers BF16/q4/q8 on both routes. |
| Wan I2V stack overflow | 3 | The existing 32 MiB reserve gets past the previous crash. A bounded physical probe then exposed the next A14B adapter defect: missing request memory carrier and sealed adapter identity in the admission context. |
| A14B receipt/context incomplete | Both A14B routes | Derive the explicit memory carrier from the provider's mode contract before minting its receipt, and carry the provider's sealed adapter identity into admission. Tests enter the actual request scope at every tier and reject mutated requests and missing identities. |
| SCAIL2 BF16 runtime budget | 1 | The q4/q8 cells completed in 6,244/6,365 seconds. This recipe uses 20 steps with CFG, 832×480, 77 frames and no Lightning. Follow the story's explicit `video=330` budget on the next BF16 attempt; another full render is not required to diagnose this stop. |

This repair pass does not publish new memory anchors. The header preflight checks
file completeness, not content hashes; a corrupt existing blob is named explicitly
and still requires repair if the normal Hugging Face fetch treats it as cached.
Host replacements are verified separately against publisher SHA-256 values.

All five SenseNova installs were repaired at their manifest revisions on
2026-09-11: base BF16 (eight shards plus missing tokenizer/index files), Fast q8,
Infographic v2 q4, and Infographic v2 Fast BF16/q4. All twelve replacement weight
files passed publisher SHA-256 and safetensors length checks, including hash
readback through their installed snapshot symlinks. The last two were recovered
by copying the existing prefixes into separate staging files and fetching their
missing tails; both complete-file hashes matched. Their original 28,333,502,721-
and 5,222,604,064-byte blobs remain as `.corrupt-sc23026-late` backups. The live
preflight now classifies all eighteen SenseNova Candle cells as runnable.

The Windows q4 I2V diagnostic used the existing inference pin `81fda3bd` and
the adapter's verified 33,554,432-byte PE stack reserve. After the receipt repair
it passed actual admission, text encoding and renderer loading, with GPU execution
observed, then hit the diagnostic's ten-minute bound before a completed denoise
step. This verifies progress beyond the original crash, not a complete render or
a new memory anchor. The diagnostic executable and logs are separate from the
campaign's original executable and evidence.

## Integration order

The first inference fixes landed in [inference PR #968](https://github.com/SceneWorks/inference/pull/968),
followed by the SD3.5, SenseNova and Wan T2V repairs in
[inference PR #969](https://github.com/SceneWorks/inference/pull/969).
SceneWorks pins their merged main revision
`8168c4b52243863655e1e62ffc2bdf14cea42bdb` together with these adapter fixes.
The two truncated Mage artifacts and all five SenseNova installs are repaired and verified. Resume measurements
using the successful evidence already collected. These patches do not themselves
establish new GPU measurements or close sc-23026.
