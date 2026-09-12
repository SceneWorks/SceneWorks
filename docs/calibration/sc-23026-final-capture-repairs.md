# Remaining Candle capture repairs (sc-23026)

Scope: the 17 failures in five groups from run 34616864537. PR #2821 owns
the successful evidence fold. This change does not rerun the campaign or change
the Bernini/SCAIL2 probe budgets.

## Acceptance and validation

| Route | Tiers | Required result |
| --- | --- | --- |
| Kolors | bf16, q4, q8 | Adapter evidence revision passes the real provider's admission; the old revision is rejected. |
| PuLID FLUX | bf16, q4, q8 | Identity inventory survives in an artifact accepted by the existing schema and Rust reader. |
| Qwen Edit 2511 Lightning | bf16, q4, q8 | A pinned Hugging Face snapshot symlink passes recipe validation; crossed route, scale, and changed receipts remain rejected. |
| Mage variants | q4, all six variants | Shared text encoder matches the publisher SHA-256 and has complete safetensors data. |
| MiniMax H3 Reference | q4, q8 | Both damaged reference transformer shards match publisher SHA-256 and have complete safetensors data. |

Risks: accepting a foreign Lightning adapter, losing identity provenance, treating
an unchanged file length as proof of valid weights, or claiming a weights-free
test proves a completed render. Tests must exercise production admission and
serialization; host repairs must verify full publisher hashes and preserve backups.

Before publication: focused adapter and Qwen provider regressions, Node harness
and catalog tests, formatting, generated-file freshness, pin consistency if the
engine changes, and independent review of the complete diff. Full GPU catalog
measurement remains a separate operator dispatch.

## Findings and repairs

- Kolors used `sc-22732@<pin>` instead of its provider-owned request evidence
  revision. The adapter now reads that constant directly. The regression drives
  the real provider safety check for all three tiers and rejects the old token.
- PuLID emitted an `identityBundle` object absent from both the JSON schema and
  Rust artifact type. Both adapters now emit the existing `inventorySha256` field:
  the SHA-256 of the ordered `<file>:<content hash>` inventory. The loadability
  fingerprint retains the same composite identity; no schema relaxation is made.
- Qwen Lightning canonicalized a snapshot file symlink into `blobs/`, losing its
  pinned recipe path. The engine fix resolves the parent and preserves the filename;
  prepared file receipts continue to bind the resolved bytes. Tests cover the real
  symlink layout, route and scale crossings, raw blob paths, and changed contents.

The host repairs replaced these truncated blobs only after downloading and
verifying their complete publisher SHA-256. The prior blobs remain as backups.

| File | Previous bytes | Verified bytes | Publisher SHA-256 |
| --- | ---: | ---: | --- |
| Mage components q4 `text_encoder/model.safetensors` | 233319257 | 4319577455 | `1ac6f4bc5ec603ff8f95cad190effefc6ecb1c68ed49ce789daeb8ded04fe7b2` |
| MiniMax q4 `transformer_ref/diffusion_pytorch_model-00001-of-00014.safetensors` | 176733569 | 1488457995 | `135c2385128d55c8d24545df51d8559fe01ebb7eeab874bbc6ff3519870908e0` |
| MiniMax q8 `transformer_ref/diffusion_pytorch_model-00001-of-00014.safetensors` | 1578248925 | 2649330053 | `3a2777b0833099b877f0bbe3d9c4044318c66ccdc5ca2e9f061bc0d78243080a` |

Revisions are unchanged: Mage components `c936de2a107ee8d0869137e73943f6414f23adaa`
and MiniMax `137ce668c55a20bc0935fd1cf2a3de8448abb7f4`. The post-install audit checked
163 safetensors files across these snapshots and the MiniMax upstream snapshot;
all headers, contiguous tensor offsets, and declared lengths passed. All 17 scoped
catalog rows are runnable in the host preflight. This is not a completed render.
