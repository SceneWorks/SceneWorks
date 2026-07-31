# sc-16195 — Resolution calibration of the generic MLX request estimator

The measurement campaign behind the change to `MlxRequestPlan::generic_total_peak_bytes`
(`crates/sceneworks-worker/src/mlx_fit_gate.rs`). Raw rows in
[`resolution-sweep.json`](./resolution-sweep.json); the harness that produced them is
`crates/sceneworks-worker/src/resolution_sweep.rs`.

---

## 1. What was wrong

The request-scoped generic estimator modeled

```rust
peak = asset_bytes + activation_headroom_bytes * (w*h / 1024²).max(1.0) * batch
```

`activation_headroom_bytes` is `HEADROOM_GB` (18) minus the 2 GiB legacy unified reserve = **16 GiB**.
That constant came from sc-10863, which measured four tiers **at 1024² only**, and its own doc comment
decomposes it as *the max common-case transient (14.04) + a ~4 GiB macOS/app reserve*. Multiplying the
whole thing by megapixels is wrong twice over:

1. **The OS/app share is fixed overhead.** macOS and other apps draw the same amount from the unified
   pool whether this request renders 512² or 2048². Scaling it is pure invention.
2. **The transient is not linear in area.** The only prior evidence (sc-5567, SenseNova-U1 8B Q8) had
   16× the area producing 3.8× the memory.

sc-16194 fixed the dominant term (the job image count was being charged as a batch dimension). This is
the second-order half, and the story that filed it said explicitly that picking a replacement curve
without measurements would be guesswork on a gate whose permissive-side failure mode is an OS Jetsam
SIGKILL. Hence the sweep.

---

## 2. Method

`resolution_sweep.rs` is the sibling of `footprint_measure.rs` with the axis rotated: that harness
measures **one resolution across tiers**, this one measures **one tier across resolutions**.

Per resolution cell, on a **warm** generator (weights already materialized):

```text
clear_cache(); reset_peak_memory();        // peak re-based to the resident weights
generate(w, h)                             // one real render through the worker's own load seam
peak = get_peak_memory()                   // weights + this cell's activation high-water
clear_cache(); resident = get_active_memory()
transient = peak − resident                // the area-dependent term under test
```

Notes on why each piece is what it is:

- **In-process counters, not `ps`.** `mlx_rs::memory::get_{active,peak,cache}_memory` are the only
  observers of the Metal allocator's high-water mark. RSS and `getrusage` maxrss do not see it, so no
  out-of-process sampler could produce these numbers.
- **`clear_cache()` before sampling resident** is the sc-8516 credibility fix. Without it the
  generation's freeable scratch is folded into resident and the transient is understated.
- **Warm is the regime that matters here.** `evaluate_request` runs per request, after the
  generator-cache lookup and immediately before generation. The cold load+gen ceiling is gated
  separately and resolution-blind (`decide_residency_for_spec`, the flat `HEADROOM_GB`), so it is
  recorded as a `"cell":"cold"` row for reference rather than mixed into the fit.
- **Each tier's own recipe** (steps / guidance from `engines.rs`) is used, because a guidance-distilled
  engine rejects a guidance value outright and a true-CFG engine runs a second uncond forward whose
  activations are part of what is being measured. Step *count* does not move the peak — the high-water
  mark is set by one denoise step's working set plus the VAE decode, not by how many steps run.
- **A blank frame is the only hard failure.** A low-contrast-but-real render at an out-of-distribution
  aspect is still a genuine activation profile, so it is flagged on the row (`lowContrast`) rather than
  dropped — several of these families list only 1024²-ish resolutions in the catalog, and the sweep
  deliberately probes past them.

Cells: **512², 1024², 1024×1536, 1152×2048, 2048²** — the story's required minimum. Square and
non-square points at nearby areas are both present on purpose: they separate "scales with AREA" from
"scales with the LONGEST SIDE".

**Instrument check.** Three tiers reproduce their sc-10863 numbers bit-for-bit through the new
harness before any new cell is trusted: illustrious q8 resident 4.74 GiB and 1024² transient 14.04;
qwen-image q8 1024² transient 7.66 and peak 41.11. The instrument agrees with the calibration it is
meant to refine.

---

## 3. Results

**7 tiers across 5 families, 5 cells each.** Activation transient (GiB), warm, `peak − resident`:

| tier | 512² | 1024² | 1024×1536 | 1152×2048 | 2048² |
|---|---:|---:|---:|---:|---:|
| *megapixels* | *0.25* | *1.00* | *1.50* | *2.25* | *4.00* |
| illustrious_xl_v1 q8 — SDXL UNet, packed | 4.28 | 14.04 | 21.02 | 31.51 | 55.06 |
| sdxl bf16 — SDXL UNet, **dense** | 4.28 | 14.04 | 21.02 | 31.51 | 55.06 |
| z_image_turbo q4 — DiT, packed | 4.17 | 14.04 | 21.02 | 31.52 | 55.07 |
| lens q4 — DiT, packed | 4.41 | 14.04 | 21.03 | 31.52 | 55.08 |
| qwen_image q8 — DiT, **tiled** VAE decode | 3.92 | 7.66 | 11.48 | 17.21 | 30.60 |
| krea_2_turbo q8 — DiT, tiled VAE, packed | 3.92 | 7.67 | 11.48 | 17.22 | 30.61 |
| krea_2_turbo bf16 — DiT, tiled VAE, **dense** | 3.92 | 7.67 | 11.48 | 17.22 | 30.61 |

Normalised to each tier's own 1024² anchor — this is the **shape**, independent of the family's
absolute level:

| tier | 0.25 MP | 1.00 MP | 1.50 MP | 2.25 MP | 4.00 MP |
|---|---:|---:|---:|---:|---:|
| illustrious_xl_v1 q8 | 0.305 | 1.000 | 1.497 | 2.245 | 3.922 |
| sdxl bf16 | 0.305 | 1.000 | 1.497 | 2.245 | 3.922 |
| z_image_turbo q4 | 0.297 | 1.000 | 1.497 | 2.244 | 3.921 |
| lens q4 | 0.314 | 1.000 | 1.497 | 2.245 | 3.922 |
| qwen_image q8 | 0.511 | 1.000 | 1.498 | 2.246 | 3.994 |
| krea_2_turbo q8 | 0.511 | 1.000 | 1.498 | 2.245 | 3.992 |
| krea_2_turbo bf16 | 0.511 | 1.000 | 1.498 | 2.245 | 3.992 |
| **proportional-to-area** | *0.250* | *1.000* | *1.500* | *2.250* | *4.000* |

### Finding 1 — above 1024² the transient is proportional to area, to within 2%

**Maximum deviation from `ratio = megapixels` across every cell above the anchor, all seven tiers:
1.97%.** At 1.5 MP the seven tiers span 1.497–1.498. They are an SDXL UNet, three single-stream DiTs,
and three tiers whose VAE decode is tiled (sc-11747) — architectures with no reason to agree. Whatever
sets a family's absolute level, the *slope in area* is not architecture-bound over this range.

### Finding 2 — this REFUTES the premise the story was filed on

sc-16195 was written expecting a sublinear curve, reasoning from the two sc-5567 points (16× area →
3.8× memory, an exponent near **0.48**) and calling linear-in-area "the wrong model". The sweep says
the opposite: the exponent is **≈0.98**, and fitting 0.48 would have predicted illustrious q8 at 2048²
as `14.04 × 4^0.48 = 27.3 GiB` against a measured **55.06** — a 28 GiB under-prediction on a gate
whose permissive-side failure mode is an OS Jetsam SIGKILL.

The sc-5567 pair is not wrong, it is *two points spanning the anchor*: 512² (below, where the floor
dominates) and 2048² (above). Fitting a single power law across the floor's knee yields an exponent
that describes neither side. The lesson is the story's own — measure, don't extrapolate — and it cut
against the story's own hypothesis.

### Finding 3 — below 1024² there is a floor, and the existing clamp is the right shape

At 0.25 MP every tier sits **above** the 0.250 a proportional term would give (0.297–0.511). Something
resolution-independent — text-encoder activations, allocator working set — dominates down there. The
estimator's pre-existing `.max(1.0)` clamp on the scale is therefore the conservative reading of the
data, and is kept.

### Finding 4 — the transient is independent of the quant tier

`sdxl bf16` (dense) matches `illustrious_xl_v1 q8` (packed) to the byte at every cell, and
`krea_2_turbo bf16` matches `krea_2_turbo q8` likewise. Activations are bf16 regardless of how the
weights are encoded, so the transient is a property of *architecture × resolution* only. This is why
the story's "dense and packed" requirement resolves to a single answer per family rather than two —
and it is what makes a per-family anchor (§4) characterisable from one tier.

### What was actually wrong, and what this changes

Only one thing: the fixed macOS/app share of the flat allowance was being multiplied by megapixels.

```text
before:  asset + 16·MP
after:   asset + 2 + 14·MP          (16 GiB allowance = 2 fixed + 14 area)
```

Identical at the 1024² anchor for every family, by construction — the sweep re-derived the *shape*,
it did not re-cut the safety margin. Above the anchor:

| cell | before | after | Δ |
|---|---:|---:|---:|
| 1024×1536 | 24.00 | 23.00 | −1.00 |
| 1152×2048 | 36.00 | 33.50 | −2.50 |
| 2048² | 64.00 | 58.00 | −6.00 |

---

## 4. The story's impact claim is NOT delivered by this fix — and the reason is a different axis

sc-16195 predicted that krea-2-turbo bf16 (33.22 GiB) at 1152×2048, modeled at 69.22 GiB, "would fit
a 64 GiB Mac" once the estimator was corrected. It does not — and because that exact cell is in the
sweep, this is measured rather than argued:

| | GiB | on a 64 GiB Mac (62 GiB budget) |
|---|---:|---|
| **measured real peak** | **49.27** | would have completed |
| modeled before sc-16194 (count 4) | 177.22 | rejected |
| modeled after sc-16194 | 69.22 | rejects |
| modeled after sc-16195 (this change) | 66.72 | **still rejects** |
| modeled with krea's *measured* anchor (7.67) | 52.45 | admits |

The story's over-rejection claim is real — the gate refuses a request that fits with 12.7 GiB to
spare. But the resolution shape was never the dominant error. **The dominant error is the flat
per-family anchor**: every family is charged the same 14 GiB 1024² transient, while krea and
qwen-image both measure **7.66–7.67** — 45% under it, because their shared Qwen-Image VAE decodes
tiled (sc-11747).

That is sc-11924's axis (per-architecture transient terms), not this story's, and it is filed as
**sc-16209** with these measurements attached. Finding 4 above is what makes it tractable: because the
transient is tier-independent, one measured tier characterises a family.

Why it is not folded in here: this change is a pure *shape* correction that provably cannot lower the
modeled peak below any measured point, whereas lowering a family's anchor reduces a SIGKILL-guarding
margin and must be argued family by family — including for the families this sweep did not reach,
which must keep the conservative 14. Two different risk profiles, two different reviews.

---

## 5. Conservativeness verification (story requirement 4)

The story requires the estimator to stay conservative at the measured points — the sweep sets the
shape, not a smaller safety margin. Checked against all 35 warm cells, `modeled − measured peak`:

| tier | 1024² | 1024×1536 | 1152×2048 | 2048² |
|---|---:|---:|---:|---:|
| sdxl bf16 | **+1.95** | +1.97 | +1.97 | +2.92 |
| z_image_turbo q4 | +2.07 | +2.09 | +2.09 | +3.04 |
| illustrious_xl_v1 q8 | +2.23 | +2.25 | +2.25 | +3.21 |
| lens q4 | +3.17 | +3.18 | +3.19 | +4.13 |
| krea_2_turbo q8 | +9.41 | +12.59 | +17.36 | +28.47 |
| krea_2_turbo bf16 | +9.50 | +12.68 | +17.45 | +28.56 |
| qwen_image q8 | +10.80 | +13.98 | +18.75 | +29.85 |

**No cell is under-predicted.** The tightest margin over the whole sweep is **+1.95 GiB**, and it
falls at the 1024² anchor — where this change is a no-op by construction. The margin the gate actually
runs closest to is therefore *unchanged* by sc-16195; only the slack above the anchor shrinks, and it
shrinks toward the measurement rather than past it.

The two margin regimes visible here are the same finding as §4 from the other side: the four families
whose anchor really is ~14 sit on a ~2–4 GiB margin, while the three tiled-VAE tiers carry 9–30 GiB of
unused slack because they are charged an anchor they never approach. That slack is sc-16209's to
reclaim, and reclaiming it is exactly what would move a family from the second group into the first —
which is why it needs its own review rather than riding along here.
