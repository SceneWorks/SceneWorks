# SC-20686 cache attribution contract

This source-only lane records where a future real-weight receipt must measure persistent/reused
cross-attention state, transient read/dequant/workspace, process/allocator peak, and generation
timing. Changing self-attention is explicitly excluded: it is request-dependent and is never counted
as a reusable cache opportunity.

## Supported route map

| Family | Route | K/V creation and position handling | Reuse boundary |
| --- | --- | --- | --- |
| FLUX.2 Klein | `candle-gen-flux2` Klein caption-upsample route | `crates/media/candle-gen/candle-gen-flux2/src/text_encoder.rs:171-198` (`forward_step` creates q/k/v, applies `rotary.apply_at`, then `ContiguousKvCache::update`); `:484-504` drives every layer | `:524` creates one decoder cache per generation; repeated `generate_from_embeds` calls recompute the decoder cache, so only a separately proven prompt/conditioning reuse row may count. Self-attention is excluded. |
| Wan | `candle-gen-wan` A14B T2V/I2V route | `crates/media/candle-gen/candle-gen-wan/src/wan14b.rs:533-570` (`denoise_range` prepares cross K/V before the loop and reuses it across denoise steps); `:1025-1056` owns lazy component reuse | `prepare_cross_kv` at `:552-554` is request-scoped and reused within the denoise range; changing latent/self-attention state is recomputed per step and excluded. |

The paired inference reducer is `scripts/sc20686_cache_attribution.py`. It accepts injected runtime
rows only, rejects inferred rows and self-attention, applies the tracker-recorded thresholds, and
produces separate family decisions. No real-weight claim or Go/No-go is emitted without rows.

## Campaign producer boundary

The paired campaign producer is `inference/scripts/sc20686_campaign_adapter.py`. It is inert unless
called with `--campaign`; normal generation has no observer or receipt overhead. At the mapped
FLUX.2 Klein/Wan entrypoint, the producer receives the product-owned observer event stream and
exact geometry/configuration, hashes the complete snapshot inventory, requires creation/reuse/
invalidation/release and allocator/process phase samples, then atomically writes one canonical raw
row plus its SHA-256 sidecar. It invokes the reducer with `--sidecar`, which verifies both hashes
before any family decision. `--fake` uses the same path for weightless tests; it is not real-weight
evidence. Missing events, geometry, identity, or explicit campaign mode fail closed.

The Rust providers expose the same observer seam through `sc20686_observer::install_jsonl`: the
campaign launcher passes `--sc20686-campaign --sc20686-events -` to the selected provider entrypoint,
which emits JSONL for generation start/end, cross-KV creation/read, invalidation, and release. The
default observer remains `None`; changing self-attention is never emitted as a reusable event.

### FLUX.2 Klein route limitation

The supported reference-image route is `Flux2Edit::generate_inner` at
`crates/media/candle-gen/candle-gen-flux2/src/edit_provider.rs:580-647`: references are VAE-encoded,
then concatenated into the joint `[target, refs]` stream at `:630` and re-concatenated for every
denoise prediction at `:685-686`. It does not create a persistent reference-image K/V cache; the
existing `ContiguousKvCache` hook in `text_encoder.rs` is caption-upsample self-attention and is
not valid evidence for this story. The reference opportunity is blocked until a product-owned
persistent cross-attention K/V boundary exists.

### Wan variant closure

The registered Wan generator surface is asserted in
`crates/media/candle-gen/candle-gen-wan/src/lib.rs:1488-1492`: TI2V-5B, T2V-14B, I2V-14B,
`wan_vace`, and VACE-Fun 14B. The shared cross-K/V ownership is
`transformer.rs:389-439` for block projection/read and `transformer.rs:621-653` for the
multi-block prepared cache; the denoise lifecycle is `wan14b.rs:533-581`. The checked-in
coverage manifest lists all five IDs (plus the historical adapter alias) and requires exact
geometry axes before reduction. No terminal family decision is valid until each listed variant
has a real full-generation row for its representative geometry coordinates.

Thresholds: opportunity ≥512 MiB and ≥5% peak with reuse ≥2; projected saving ≥256 MiB and ≥3%
peak without replacement transient; runtime-only opportunity ≥5% generation time.

Real-weight generation receipts remain blocked until the appropriate FLUX.2 Klein and Wan assets are
available on an uncontended runner. No measurement is fabricated by this source-only lane.

### Wan producer API blocker

Loaded Wan identity is owned by private `WanGenerator` fields (`root`, `descriptor`, and
`dit_source`) at `candle-gen-wan/src/lib.rs:853-880`; exact configuration is held by private
`Pipeline::{te_cfg,dit_cfg,vae_cfg,variant,root}` at `candle-gen-wan/src/wan14b.rs:146-163`.
The current public observer accepts only a sink path and receives no generator, pipeline,
request, or tensor metadata. A truthful producer therefore requires a campaign context threaded
through this private API; caller-authored JSON/config is not evidence and remains rejected.
