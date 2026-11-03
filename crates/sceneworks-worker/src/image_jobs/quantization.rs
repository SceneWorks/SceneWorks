use super::artifact_resolution::resolve_weights_dir;
use super::artifact_resolution::tier_key_for_resolved_dir;
use super::is_dense_te_tier;
use super::nvfp4_host_eligible;
use super::nvfp4_selected;
use super::tier_static_name;
use super::ImageRequest;
use super::Quant;
#[cfg(not(target_os = "macos"))]
use super::INT8_CONVROT_TIER;
#[cfg(target_os = "macos")]
use crate::emit_event;
use crate::quant_int;
use crate::JsonObject;
use crate::Path;
use crate::PathBuf;
use crate::Settings;
use crate::Value;
#[cfg(target_os = "macos")]
use serde_json::json;
#[cfg(native_lane)]
/// Resolve quantization: the explicit `advanced.quantTier: "nvfp4"` label → `advanced.mlxQuantize` →
/// `manifest.mlx.quantize` → Q8 default. The engine supports Q4/Q8/NVFP4; map (<=0 → dense, <=4 → Q4,
/// else Q8). Returns the engine quant + the effective bit count for the recipe (None = dense bf16).
///
/// Shared by the MLX path and the candle lane (sc-5126). On the candle lane it is called ONLY for a
/// family whose descriptor advertises `supported_quants` (i.e. Lens — see `generate_candle_stream`'s
/// `model.supports_quant()` gate), so the Q8 default applies to Lens exactly like the MLX families;
/// the sc-3675/sc-5096 candle families advertise no quant and never reach this resolver (stay dense).
///
/// **NVFP4 (sc-11042, epic 11037 SC#5)** is selected ONLY on a full [`nvfp4_selected`] — an explicit
/// pick, a Blackwell-eligible host, AND the `nvfp4/` tier actually resolved on disk — never from
/// `bits`, a manifest default, or hardware detection alone, and never on a `tier_dir` that resolved to
/// some other tier (that would load FP4 against q8 weights and record the wrong tier). Its recipe
/// bit count is `None`, not `Some(4)`: `Quant::Nvfp4::bits()` returns 4 (its E2M1 elements are 4-bit),
/// but NVFP4 is ~4.5 EFFECTIVE bits/weight and is a different tier from int4-affine `q4`, so reporting
/// `4` here would stamp an NVFP4 render with `q4`'s bit count in the recipe — the aliasing SC#5 forbids
/// (the same footgun `video_quant_label` carries; see its note). Every non-NVFP4 request takes the
/// unchanged path below.
///
/// `tier_dir` is the RESOLVED tier subdir this job will load (`standard_tier_subdir`'s output), or
/// `None` on a lane with no standard-tier layout in scope — see [`tier_dir_is_nvfp4`] for why `None`
/// conservatively means "not NVFP4".
#[cfg(native_lane)]
pub(crate) fn resolve_quant(
    request: &ImageRequest,
    tier_dir: Option<&Path>,
) -> (Option<Quant>, Option<i64>) {
    resolve_quant_gated(request, nvfp4_host_eligible(), tier_dir)
}

/// [`resolve_quant`] with the NVFP4 **host** gate passed in rather than probed (sc-11042). Split out
/// for testability, exactly like [`standard_tier_subdir_gated`] — and necessarily so: the candle CI
/// lane runs ON the sm_120 rig, so a test that let this probe the live cap would assert different
/// things on the rig and on a developer box. Production has one caller.
#[cfg(native_lane)]
pub(super) fn resolve_quant_gated(
    request: &ImageRequest,
    nvfp4_host: bool,
    tier_dir: Option<&Path>,
) -> (Option<Quant>, Option<i64>) {
    // Dense-TE turnkeys (FLUX.2-klein, sc-8711): the tier subdir already holds a packed transformer
    // + a DENSE bf16 text encoder, so the load Quant must be None — quantizing here would crush the
    // dense bf16 TE we deliberately kept full-precision. The packed transformer self-describes its
    // quant regardless. Tier selection (q4/q8/bf16) is driven by the resolved subdir, not this.
    //
    // Ordered FIRST, ahead of the NVFP4 arm (sc-11042): `flux2_klein_9b`/`_kv` are BOTH in
    // declared dense-TE (`mlx.denseTextEncoderTier`) and on the candle txt2img lane (whose `resolve_quant` call is gated only
    // by `model.supports_quant()`), so an NVFP4 arm placed above this could return `Some(Nvfp4)` for a
    // crafted `quantTier: "nvfp4"` on a dense-TE turnkey and skip the carve-out — re-quantizing the
    // bf16 text encoder sc-8711/sc-9362 deliberately kept dense. The carve-out is the wider invariant
    // (the TE must never be quantized by ANY tier), so it wins outright rather than being an exception
    // the NVFP4 arm has to remember.
    if is_dense_te_tier(request) {
        return (None, None);
    }
    // The distinct NVFP4 tier (sc-11042): an EXPLICIT `advanced.quantTier: "nvfp4"` pick AND a
    // Blackwell-eligible host AND the `nvfp4/` tier resolved on disk — all three halves, the SC#5
    // opt-in (see [`nvfp4_selected`]). Checked before the bits map: it is a tier identity, not a point
    // on the bits ladder. Off Blackwell / off the candle lane `nvfp4_host` is false, and no shipping
    // model packs an `nvfp4/` dir today, so this arm is unreachable there and every request resolves
    // exactly as it always has.
    if nvfp4_selected(request, nvfp4_host, tier_dir) {
        return (Some(Quant::Nvfp4), None);
    }
    let raw = request.quant_bits().or_else(|| {
        request
            .model_manifest_entry
            .get("mlx")
            .and_then(|mlx| mlx.get("quantize"))
            .and_then(quant_int)
    });
    match crate::payload::tier_for_bits(raw, "q8") {
        "bf16" => (None, None),
        "q4" => (Some(Quant::Q4), Some(4)),
        _ => (Some(Quant::Q8), Some(8)),
    }
}

/// The transformer-tier bit count a dense-TE turnkey (FLUX.2-klein — a declared
/// `mlx.denseTextEncoderTier` entry) actually asked for, derived from `advanced.mlxQuantize` the SAME
/// way [`standard_tier_subdir`]
/// picks its `bf16`/`q8`/`q4` tier (no explicit pick → `q8`; `<=0 → bf16`; `>4 → q8`; else `q4`).
/// Returns the recipe bit count of the REQUESTED tier: `None` (bf16) / `Some(8)` / `Some(4)`. Kept in
/// lockstep with the q8 default (sc-10726) so a straight default dense-TE job that resolves the q8 tier
/// isn't mis-reported as a bf16/q4→q8 tier change by [`reconcile_resolved_tier_quant`].
///
/// sc-9362 (F-018 follow-up): [`resolve_quant`] returns `(None, None)` for every dense-TE job (the
/// load quant must stay `None` so the deliberately-dense bf16 text encoder is never re-quantized), so
/// the request-derived recipe bits are ALWAYS bf16 even though the transformer is packed at q4/q8. If
/// [`reconcile_resolved_tier_quant`] compared the resolved transformer tier against that always-`None`
/// value, every straight (non-fallback) dense-TE job would look like a bf16→qN "downgrade" — firing a
/// spurious `quant_tier_downgraded` event while the asset telemetry still hid the true transformer
/// precision. Comparing the resolved tier against THIS requested-tier value instead lets the reconcile
/// record the actual transformer precision on every job and warn/emit ONLY on a genuine fallback
/// (requested tier absent → resolver fell through to an adjacent tier).
///
/// macOS-only: consumed on the MLX `generate_stream` reconcile path (the candle lane has no tier
/// layout), alongside [`tier_quant_from_resolved_dir`].
#[cfg(target_os = "macos")]
pub(super) fn dense_te_requested_tier_bits(request: &ImageRequest) -> Option<i64> {
    match crate::payload::tier_for_bits(request.quant_bits(), "q8") {
        "bf16" => None,
        "q4" => Some(4),
        _ => Some(8),
    }
}

/// The generation tier (`bf16`/`q8`/`q4`) a resolved turnkey tier subdir ACTUALLY loads at, parsed
/// from its basename (sc-8820 / sc-12090). The tier resolvers ([`standard_tier_subdir`],
/// [`ideogram_model_subdir`], [`boogu_model_subdir`], [`krea_model_subdir`]) fall through q4→q8→bf16
/// (or the Boogu `<variant>`-shaped names) when the requested tier isn't downloaded, so the resolved
/// path can be a DIFFERENT precision than the request asked for. This maps the resolved basename back
/// to its tier:
///   - `bf16` / `<variant>-bf16` → `bf16`
///   - `q4`   / `<variant>-q4`   → `q4`
///   - `q8`   / `<variant>-q8`   → `q8`
///   - a bare Boogu `<variant>/` (no `-q4`/`-q8`/`-bf16` suffix) → the shipped **Q8** default
///
/// `None` when the basename is not a recognizable tier name — e.g. the resolver fell all the way back
/// to the repo `root` (a partial/absent turnkey the engine will error on), or the dir is a `modelPath`
/// override. In that case the caller keeps its request-derived tier rather than inventing one.
///
/// Available on the candle lane too (sc-12090): the candle VRAM fit-gate reads the tier the
/// disk-probing resolver landed on here — instead of re-deriving from the manifest — so it names and
/// budgets against the tier that would actually load, never an uninstalled one.
/// The tier key the candle VRAM fit-gate sizes a render against — the ONE place that decision is made
/// (sc-12425). Pure: extracted out of `generate_candle_stream` so the ConvRot identity below is
/// unit-testable; that function takes an api/settings/job and cannot be exercised from a unit test, so
/// the mapping had no gate of its own.
///
/// Resolution order:
///
/// 1. **A resolved ConvRot load ⇒ its tier IDENTITY** (`convrot_resolved` = the base dir AND the int8
///    DiT are both on disk). This is the sc-12425 fix. It used to hand ConvRot to
///    [`vram_gate::requested_tier_key`](crate::vram_gate::requested_tier_key), which is BITS-derived —
///    and a ConvRot request carries no `mlxQuantize` ([`wants_krea_convrot`]), so it fell to that
///    function's `None => "q8"` arm. That sized a **measured 42.9 GB** render (sc-12381) against q8's
///    35.9 + 2.0 = 37.9 GB and admitted loads that OOM. Identical aliasing to the one sc-11042 fixed for
///    NVFP4, except q8 OVER-predicts NVFP4 ("never an OOM") and UNDER-predicts ConvRot.
/// 2. Else the **on-disk tier** the resolver landed on (sc-12090) — budget the tier that will load, not
///    one the user never installed. A `modelPath`/flat root has no recognizable basename ⇒ `None`.
/// 3. Else the manifest/request key (`requested_tier_key`), whose own `nvfp4` arm is the sibling of (1).
// Candle-lane only — NOT `any(macos, candle)` like `tier_key_from_resolved_dir`, because this fn calls
// `crate::vram_gate`, which is itself `#[cfg(all(not(macos), backend-candle))]`. On the macOS/MLX build
// `vram_gate` doesn't exist, so an `any(macos, ...)` gate here fails to compile (E0433). Its only caller,
// `generate_candle_stream`, is candle-only too, so this loses nothing.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn gate_tier_key(
    convrot_resolved: bool,
    weights_dir: &Path,
    advanced: &JsonObject,
    manifest_entry: &JsonObject,
    nvfp4: bool,
) -> &'static str {
    if convrot_resolved {
        return INT8_CONVROT_TIER;
    }
    tier_key_for_resolved_dir(manifest_entry, weights_dir)
        .unwrap_or_else(|| crate::vram_gate::requested_tier_key(advanced, manifest_entry, nvfp4))
}

#[cfg(native_lane)]
pub(super) fn tier_key_from_resolved_dir(dir: &Path) -> Option<&'static str> {
    let name = dir.file_name()?.to_str()?;
    // Match the trailing tier token: the whole basename (standard/ideogram/krea) or the suffix of a
    // Boogu `<variant>-<tier>` folder; a bare Boogu `<variant>` IS the packed Q8 default.
    match name.rsplit('-').next().unwrap_or(name) {
        "bf16" => Some("bf16"),
        "q4" => Some("q4"),
        "q8" | "base" | "turbo" | "edit" => Some("q8"),
        _ => None,
    }
}

/// The `(engine Quant, recipe bit count)` a resolved turnkey tier subdir ACTUALLY loads at (sc-8820) —
/// the `Quant`-typed view of [`tier_key_from_resolved_dir`], consumed by
/// [`reconcile_resolved_tier_quant`] on the MLX `generate_stream` path. `None` for an unrecognizable
/// basename (kept identical to the pre-sc-12090 behavior: the caller keeps the request-derived quant).
///
/// macOS-only: the candle lane has no quant-tier layout to reconcile, so this would be dead code there.
#[cfg(target_os = "macos")]
pub(super) fn tier_quant_from_resolved_dir(
    manifest_entry: &JsonObject,
    dir: &Path,
) -> Option<(Option<Quant>, Option<i64>)> {
    match tier_key_for_resolved_dir(manifest_entry, dir)? {
        "bf16" => Some((None, None)),
        "q4" => Some((Some(Quant::Q4), Some(4))),
        "q8" => Some((Some(Quant::Q8), Some(8))),
        _ => None,
    }
}

/// Resolve the on-disk weights dir for `request` FORCED to a specific generation `tier`
/// (`bf16`/`q8`/`q4`), reusing the SAME family disk-probing resolver the default path uses
/// ([`resolve_weights_dir`]). `Some(dir)` only when that tier is actually INSTALLED — the resolver
/// lands on that exact tier; `None` when it isn't (the resolver falls through to a different/absent
/// tier) or the model has no quant-tier layout. This lets the capability downtier (sc-10733) and the
/// reject-message tier suggestions (sc-12090) enumerate installed tiers WITHOUT duplicating each
/// family's bespoke `present()` logic — a forced-bits probe is byte-for-byte what the loader would do.
///
/// The forced `mlxQuantize` is an explicit tier probe, so it bypasses the per-model quality floor
/// (explicit picks are always honored) — the caller filters candidates to `>= floor` before probing.
#[cfg(native_lane)]
pub(super) fn resolve_tier_dir(
    request: &ImageRequest,
    settings: &Settings,
    tier: &str,
) -> Option<PathBuf> {
    let bits: i64 = match tier {
        "bf16" => 0,
        "q4" => 4,
        _ => 8, // q8
    };
    let mut probe = request.clone();
    probe
        .advanced
        .insert("mlxQuantize".to_owned(), Value::from(bits));
    let dir = resolve_weights_dir(&probe, settings).ok().flatten()?;
    (tier_key_for_resolved_dir(&request.model_manifest_entry, &dir) == Some(tier_static_name(tier)))
        .then_some(dir)
}

/// Reconcile the request-derived `(quant, quant_bits)` against the tier subdir the resolver ACTUALLY
/// landed on (sc-8820). The tier resolvers fall through q4→q8→bf16 when the preferred tier isn't
/// downloaded, but the recipe quant is derived from the REQUEST — so a user who selected bf16 with
/// only `q4/` present would silently render Q4 while the sidecar records dense. That makes the epic
/// 8506 quant A/B workflow lie about precision (and can compare a tier against itself). This corrects
/// the recorded quant to the resolved tier, and — when a downgrade actually happened — `warn!`s and
/// emits a `quant_tier_downgraded` event so the UI/telemetry surfaces the fallback instead of hiding
/// it. We do NOT hard-error: a working render at an adjacent tier beats failing because the preferred
/// tier is missing (the finding prefers warn+fallback+correct-recording). Returns the
/// `(quant, quant_bits)` to record + load with.
///
/// `allow_quant_change` gates whether the LOAD quant may be rewritten to match the resolved tier.
/// It's `true` for the ordinary packed-turnkey families (the load-quant is a no-op on already-packed
/// weights, so correcting it to the resolved tier is safe/right). It's `false` for the DENSE-TE
/// turnkeys (FLUX.2-klein, sc-8711): their load quant MUST stay `None` so the deliberately-dense bf16
/// text encoder is never re-quantized — but the *recorded* bit count still gets corrected to the
/// packed transformer's resolved tier so the sidecar tells the truth. The event/`warn!` still fire so
/// the fallback is surfaced either way.
#[cfg(target_os = "macos")]
pub(super) fn reconcile_resolved_tier_quant(
    requested: (Option<Quant>, Option<i64>),
    weights_dir: &Path,
    manifest_entry: &JsonObject,
    allow_quant_change: bool,
    model_id: &str,
    job_id: &str,
    engine: &str,
) -> (Option<Quant>, Option<i64>) {
    let Some((actual_quant, actual_bits)) =
        tier_quant_from_resolved_dir(manifest_entry, weights_dir)
    else {
        // Not a recognizable tier dir (fell back to the repo root, or a modelPath override) — keep
        // the request-derived quant; the engine will surface any missing-weights error itself.
        return requested;
    };
    if actual_bits == requested.1 {
        return requested;
    }
    // The resolved tier differs from what the request asked for → a silent fallback. Surface it and
    // record the precision that actually ran.
    let requested_label = requested.1.map_or("bf16".to_owned(), |b| format!("q{b}"));
    let actual_label = actual_bits.map_or("bf16".to_owned(), |b| format!("q{b}"));
    tracing::warn!(
        "{engine}: requested quant tier {requested_label} for {model_id} is not downloaded; \
         fell back to {actual_label} — recording the tier that actually ran"
    );
    emit_event(
        "quant_tier_downgraded",
        json!({
            "jobId": job_id,
            "engine": engine,
            "model": model_id,
            "requested": requested_label,
            "resolved": actual_label,
            "requestedBits": requested.1,
            "resolvedBits": actual_bits,
        }),
    );
    // Always correct the recorded bits; only rewrite the load quant when it's safe to (packed
    // turnkeys), so a dense-TE turnkey keeps its `None` load quant while still recording the truth.
    let load_quant = if allow_quant_change {
        actual_quant
    } else {
        requested.0
    };
    (load_quant, actual_bits)
}

/// Krea's MLX base-DiT turnkeys are already packed in the resolved `bf16/q4/q8`
/// artifact directory. `LoadSpec::quantize` is a load-time transform request, not
/// that artifact-tier identity, so forwarding the UI tier as a quant transform
/// would make the provider predicate reject the otherwise exact production spec.
/// Keep the resolved tier on the request plan/declaration axes instead.
#[cfg(target_os = "macos")]
pub(crate) fn mlx_load_quant_for_resolved_artifact(
    engine_id: &str,
    quant: Option<Quant>,
) -> Option<Quant> {
    // FLUX.2 Klein's packed-tier rule is the MLX provider crate's product decision (`product_load`).
    if matches!(
        engine_id,
        "krea_2_raw" | "krea_2_turbo" | "krea_2_edit" | "krea_2_turbo_edit"
    ) || runtime_macos::providers::flux2::product_load::loads_packed_tier_unquantized(engine_id)
    {
        None
    } else {
        quant
    }
}

/// Fixed converted artifacts whose resolved precision is independent of a request preference.
/// Returning `Some` lets callers skip tier reconciliation entirely, avoiding both a false recipe
/// identity and a false downgrade event before the provider sees the load.
#[cfg(target_os = "macos")]
pub(crate) fn fixed_mlx_artifact_quant(model_id: &str) -> Option<(Option<Quant>, Option<i64>)> {
    (model_id == "flux2_klein_9b_true_v2").then_some((None, None))
}

#[cfg(target_os = "macos")]
pub(super) fn fixed_mlx_artifact_tier(model_id: &str) -> Option<&'static str> {
    (model_id == "flux2_klein_9b_true_v2").then_some("bf16")
}

# [cfg (native_lane)]
# [cfg (test)]
# [cfg (target_os = "macos")]
#[path="tests/quant_tier_reconcile_tests.rs"]
mod quant_tier_reconcile_tests;
