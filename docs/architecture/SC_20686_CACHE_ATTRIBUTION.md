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

The paired campaign producer is `inference/scripts/sc20686_campaign_adapter.py` at inference commit
`f9a0ab078b906cb0b46c56b24d16bfebb8ea5138`. It is inert unless called with `--campaign`; normal
generation has no observer or receipt overhead. For every arm the adapter creates an adapter-owned
private `events.jsonl` file and passes its path with `--sc20686-events`; provider stdout and stderr
are sealed only as diagnostics and are never parsed as event evidence. The adapter rejects a missing,
non-JSONL, or carriage-return-containing event stream before requiring product-owned metadata,
creation/reuse/invalidation/release, allocator samples, and process samples, then atomically writes
the canonical raw row and its SHA-256 sidecar. Missing geometry, identity, or explicit campaign mode
also fails closed.

The Rust entrypoints require a dedicated event file when campaign mode is selected, so provider
progress output cannot corrupt the observer transcript. The default observer remains `None`, and
self-attention is never emitted as a reusable event.

### FLUX.2 Klein route limitation

The supported reference-image route is `Flux2Edit::generate_inner` at
`crates/media/candle-gen/candle-gen-flux2/src/edit_provider.rs:580-647`: references are VAE-encoded,
then concatenated into the joint `[target, refs]` stream at `:630` and re-concatenated for every
denoise prediction at `:685-686`. It does not create a persistent reference-image K/V cache; the
existing `ContiguousKvCache` hook in `text_encoder.rs` is caption-upsample self-attention and is
not valid evidence for this story. This is the evidence-based SC-20686 No-go boundary: no persistent
reference-K/V productization or promotable FLUX cache is claimed unless a product-owned persistent
cross-attention K/V boundary and reader are actually introduced and measured.

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

### Wan producer context is wired

The former Wan producer API blocker is superseded by the live runtime context in inference commit
`f9a0ab078b906cb0b46c56b24d16bfebb8ea5138`. `sc20686_observer::activate_requested` now activates
only after the product route has reached its snapshot-backed runtime, and
`bind_cross_kv_geometry` threads the exact product-owned cross-K/V geometry into the observer before
it emits metadata. The five registered Wan routes therefore emit lifecycle, allocator, and immutable
identity facts from their real producer context; caller-authored JSON or configuration remains
non-evidence and is rejected by the adapter and reducer.
