# SC-20686 cache attribution contract

This source-only lane records where a future real-weight receipt must measure persistent/reused
cross-attention state, transient read/dequant/workspace, process/allocator peak, and generation
timing. Changing self-attention is explicitly excluded: it is request-dependent and is never counted
as a reusable cache opportunity.

## Supported route map

| Family | Route | K/V creation and position handling | Reuse boundary |
| --- | --- | --- | --- |
| FLUX.2 Klein | `flux2_klein_9b_edit` reference edit | `candle-gen-flux2/src/transformer.rs:365-381` creates image/text K/V for `DoubleAttention`; `sc20686_observer.rs:620-621` isolates the exact reference slice | `transformer.rs:393` consumes the dense joint context on every denoise evaluation. No persistent reference-K/V reader exists, so this route is a measured no-go rather than a promotable cache. |
| Wan | TI2V-5B, T2V-14B, I2V-14B, `wan_vace`, and VACE-Fun 14B | `candle-gen-wan/src/transformer.rs:347` registers each prepared cross-K/V cache and `:413` records its reads | Each request-scoped cache is reused across denoise steps and released by its owning prepared-cache lifetime; latent self-attention remains excluded. |

The paired inference reducer is `scripts/sc20686_cache_attribution.py`. It accepts injected runtime
rows only, rejects inferred rows and self-attention, applies the tracker-recorded thresholds, and
produces separate family decisions. No real-weight claim or Go/No-go is emitted without rows.

## Campaign producer boundary

The paired producer is `inference/scripts/sc20686_campaign_adapter.py`. It deliberately does not
pin a fixed inference commit in this document: every run requires
`--inference-revision <40-hex-commit>`, verifies that value against the inference checkout's
`git rev-parse HEAD`, and seals it into the resolved inputs, command transcript, observer metadata,
and row receipt. The adapter rechecks the checkout around every child run. It is inert unless called
with `--campaign`; normal generation has no observer or receipt overhead.

For every arm, the adapter creates a separate `sealed-run` directory, uses it as the child working
directory, and passes absolute sibling paths ending in `sealed-run/events.jsonl` for
`--sc20686-events` and either `sealed-run/media.png` (FLUX) or `sealed-run/media` (Wan) for `--out`.
This keeps default or explicit images and video frames inside the exact child-run closure. Before
that private directory is deleted, every normal-arm output is copied into the final campaign
bundle. A canonical per-run manifest seals its output kind, relative paths, byte counts, and exact
content hashes; the row and aggregate receipts bind both the manifest hash and each media hash.
Cancellation arms instead seal an `absent` manifest and fail if partial media remains. The reducer
also rejects command evidence when the event and media paths do not share the isolated parent.
Provider stdout and stderr remain diagnostics and are never parsed as event evidence.

The adapter rejects a missing, non-JSONL, or carriage-return-containing event stream before
requiring product-owned metadata, creation/reuse/invalidation/release, allocator samples, and
process samples, then atomically writes the canonical raw row and its SHA-256 sidecar. A cancellation
row must contain its product-owned arm identity and exactly one `cancelled` terminal followed by
metrics, invalidation, and release in product order. Every coordinate decision is explicitly bound
to that verified cancellation arm. Missing geometry, identity, exact source binding, product
residency, or explicit campaign mode also fails closed.

### Independent source and model provenance

`source_ref` identifies the verified inference repository commit. It is independent from both
`model_snapshot_revision`, which identifies the immutable Hugging Face model revision, and
`model_snapshot_sha256`, which hashes the exact selected model/tier contents. A selected FLUX or
VACE model root may be either a component/tier directory containing `config.json`, or a Diffusers
pipeline root containing `model_index.json` plus component `config.json` files. For a nested tier
such as `<snapshot-revision>/q4`, the revision is resolved from the nearest two ancestors while the
content hash remains scoped to `q4`; unrelated sibling tiers cannot change or satisfy that identity.

### Product-equivalent residency

`residency_strategy` is a sealed route axis and is passed to the real entrypoint as
`--sc20686-residency`. The entrypoint applies it to `LoadSpec`; FLUX.2 edit additionally enables its
request-scoped generation staging. The exact SceneWorks-equivalent map is:

| Product route | Residency |
| --- | --- |
| `flux2_klein_9b_edit` | `sequential` |
| `wan2_2_ti2v_5b` | `sequential` |
| `wan2_2_t2v_14b` | `sequential` |
| `wan2_2_i2v_14b` | `sequential` |
| `wan_vace` | `resident` |
| `wan2_2_vace_fun_14b` | `sequential` |

The manifest, adapter, entrypoints, observers, receipts, and reducer reject any other route/strategy
pair rather than accepting evidence from a non-product memory shape. The Wan 14B ComfyUI-expert
entrypoint passes the same campaign policy through its explicit-residency external-expert loader;
it cannot silently use that loader's ordinary resident default while sealing a sequential receipt.

The Rust entrypoints require a dedicated event file when campaign mode is selected, so provider
progress output cannot corrupt the observer transcript. The default observer remains `None`, and
self-attention is never emitted as a reusable event.

### FLUX.2 Klein route limitation

The supported reference-image route enters `Flux2Edit::generate_inner` at
`crates/media/candle-gen/candle-gen-flux2/src/edit_provider.rs:506`, with campaign activation at
`:524`. The edit transformer's `DoubleAttention` projects image K/V at
`transformer.rs:365-367`, projects the text additions at `:376-381`, and evaluates their dense joint
attention at `:393`. The observer isolates the reference portion at
`sc20686_observer.rs:620-621`. It does not create a persistent reference-image K/V cache or packed
reader. Caption-upsample `ContiguousKvCache` self-attention is outside this route and is never
campaign evidence. This is the evidence-based SC-20686 no-go boundary: no persistent reference-K/V
productization or promotable FLUX cache is claimed unless a product-owned persistent cross-attention
K/V boundary and reader are actually introduced and measured.

### Wan variant closure

The registered campaign surface is asserted by `candle-gen-wan/src/sc20686_observer.rs:43-48`:
TI2V-5B, T2V-14B, I2V-14B, `wan_vace`, and VACE-Fun 14B. Shared cross-K/V ownership registers caches
at `transformer.rs:347`, records reads at `:413`, and releases them with the prepared-cache owner.
Product activation occurs in `lib.rs:1147`, `wan14b.rs:1171`, `model_vace.rs:471`, and
`model_vace_fun.rs:576`. The checked-in coverage manifest lists all five IDs and requires exact
geometry axes before reduction. No terminal family decision is valid until each listed variant has
a real full-generation row for every representative geometry coordinate and its matching deliberate
cancellation arm.

Thresholds: opportunity ≥512 MiB and ≥5% peak with reuse ≥2; projected saving ≥256 MiB and ≥3%
peak without replacement transient; runtime-only opportunity ≥5% generation time.

Real-weight generation receipts remain blocked until the appropriate FLUX.2 Klein and Wan assets are
available on an uncontended runner. No measurement is fabricated by this source-only lane.

### Wan producer context is wired

The former Wan producer API blocker is superseded by the live runtime context sealed to each run's
verified inference revision. `sc20686_observer::activate_requested` activates only after the product
route has reached its snapshot-backed runtime, and `bind_cross_kv_geometry` threads the exact
product-owned cross-K/V geometry into the observer before it emits metadata. The five registered Wan
routes therefore emit lifecycle, allocator, model identity, source identity, and residency facts
from their real producer context; caller-authored JSON remains non-evidence and is rejected by the
adapter and reducer.
