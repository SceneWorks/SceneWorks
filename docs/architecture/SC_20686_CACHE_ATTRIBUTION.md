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

Thresholds: opportunity ≥512 MiB and ≥5% peak with reuse ≥2; projected saving ≥256 MiB and ≥3%
peak without replacement transient; runtime-only opportunity ≥5% generation time.

Real-weight generation receipts remain blocked until the appropriate FLUX.2 Klein and Wan assets are
available on an uncontended runner. No measurement is fabricated by this source-only lane.
