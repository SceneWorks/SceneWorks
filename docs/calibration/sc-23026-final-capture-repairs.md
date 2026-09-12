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
- After the path repair, a real-file registry probe exposed a second Qwen refusal:
  the nonzero overlay estimate lacked a typed auxiliary component. Nonzero adapter
  bytes now use a whole-render `AdapterStack` component, with the same byte estimate
  and no transformer-window reduction. Zero-overlay contracts stay unchanged.

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
An expanded audit of every affected row's weight roots checked 689 safetensors
paths without header, offset, or length errors. The actual Qwen Lightning snapshot
and prepared LoRA receipts pass the production registry contract at all three tiers,
using a temporary CPU-only probe; no model render or GPU campaign was executed.

## Local verification

- Adapter tests: 47 library and 96 Candle binary tests passed. Kolors admission and
  PuLID production-reader regressions failed before the code fixes.
- Capture harness: 65 tests passed, including inventory identity and strict schema
  validation. The full catalog suite has nine Windows fixture failures reproduced
  unchanged on base `68ad4cb1c` (133 pass, 22 platform skips); neither its code nor
  its tests changed in this repair.
- Qwen engine: 16 edit tests passed, including a red/green reproduction of the exact
  Lightning refusal. All-target package Clippy with warnings denied passed, as did
  formatting, the workspace gate, and the clock-assertion ratchet.
- The additional nonzero-overlay conformance regression reproduced the real-host
  failure before the fix and passes afterward; independent repair review passed.
- Adapter Clippy passed with the existing `-A dead_code` baseline allowance and
  other warnings denied. macOS adapter execution requires its platform CI lane.
- Independent review passed for the code and host repairs. Final engine pin and
  CI integration are checked separately before publication of the SceneWorks PR.

## Pin integration

SceneWorks pins `290fa1f3e7dbf2039dd7f7434cf58f3c29eec35f` from
[inference PR #974](https://github.com/SceneWorks/inference/pull/974). The standard
pin updater completed, including dependency-skew and source-audit checks. Existing
capability dump revision labels remain historical; their revision-only differences
are informational under the repository's gate teardown policy.

At this exact pin, 143 adapter tests and 1,335 core tests passed (four core tests
ignored). Adapter Clippy passed with its existing dead-code allowance. Generated
anchors and matrix are current; loader/currency tests passed 59 with 17 platform
skips. The four existing Krea/Z-Image attestations were extended after source review:
Z-Image has an empty changed-file intersection, and Krea's intersection is confined
to Qwen Edit path validation/accounting that its VAE/preview use does not execute.
No new attestation or measurement revision was invented.
