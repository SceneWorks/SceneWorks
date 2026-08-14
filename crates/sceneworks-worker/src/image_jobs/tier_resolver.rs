#[cfg(target_os = "macos")]
use super::is_anima_model;
use super::{ImageRequest, Path, PathBuf, Value, STANDARD_TIER_MODELS};

// sc-15799 (tier integrity): the hardcoded `DENSE_TE_TIER_MODELS` registry is DELETED. Keeping a dense
// bf16 text encoder resident on a q4/q8 tier is an above-tier residency exception, and an exception the
// shared decision cannot see is exactly what that story exists to eliminate — the ids lived in this file
// while the catalog declared nothing, so `config/tier-integrity.jsonc` had no way to know they existed.
// The manifest flag `mlx.denseTextEncoderTier` is now the ONLY way in, it must carry a declared exception
// row (enforced by `scripts/check-tier-integrity.mjs`), and `tests/gpu_and_manifest.rs` pins that the
// shipped catalog still declares it for the FLUX.2-klein pair so the deletion cannot silently re-quantize
// a TE that sc-8711/sc-9362 deliberately kept dense.

/// Whether a request's model ships the standard SceneWorks quant-matrix turnkey layout (sc-8508):
/// true when it is registered in [`STANDARD_TIER_MODELS`] OR its manifest entry declares
/// `mlx.standardTierLayout: true`. The manifest flag is the first-class, catalog-driven form of the
/// hardcoded registry (epic 8506) — a new quant-matrix model can opt in from the manifest alone,
/// while the registry keeps every already-wired model working with zero manifest change.
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
pub(super) fn uses_standard_tier_layout(request: &ImageRequest) -> bool {
    STANDARD_TIER_MODELS.contains(&request.model.as_str())
        || request
            .model_manifest_entry
            .get("mlx")
            .and_then(|mlx| mlx.get("standardTierLayout"))
            .and_then(Value::as_bool)
            .unwrap_or(false)
}

/// Whether a standard-tier request keeps a DENSE bf16 text encoder in every tier (sc-8508): true iff its
/// manifest entry declares `mlx.denseTextEncoderTier: true`.
///
/// **Manifest-only since sc-15799.** This is an above-tier residency exception — on a q4 or q8 tier the
/// text encoder is resident at a HIGHER precision than the user selected — so it may only be declared
/// where the shared decision, the audit table, and the parity lane can all see it. The hardcoded id list
/// this used to fall back to is gone (see the note at the top of this file); a model that needs the
/// carve-out declares the flag and carries a measured exception row in `config/tier-integrity.jsonc`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn is_dense_te_tier(request: &ImageRequest) -> bool {
    request
        .model_manifest_entry
        .get("mlx")
        .and_then(|mlx| mlx.get("denseTextEncoderTier"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// Quality rank of a generation quant tier: higher = more faithful (`bf16` = 3, `q8` = 2, `q4` = 1,
/// anything else = 0). Used to CLAMP a resolver's DEFAULT tier UP to a model's per-model quality floor
/// (`mlx.minQualityTier`, sc-10731) — the clamp only ever RAISES, never lowers.
pub(super) fn tier_quality_rank(tier: &str) -> u8 {
    match tier {
        "bf16" => 3,
        "q8" => 2,
        "q4" => 1,
        _ => 0,
    }
}

/// Normalize a floor tier string to its canonical `&'static str` name (`bf16`/`q8`/`q4`), so a floor
/// borrowed from the request manifest can be returned as a static tier subdir name. An unknown value
/// falls to `q4` (harmless — it never outranks the `q8` default, so it is never actually selected).
pub(super) fn tier_static_name(tier: &str) -> &'static str {
    match tier {
        "bf16" => "bf16",
        "q8" => "q8",
        _ => "q4",
    }
}

/// The tier subdir name of the NVFP4 tier (sc-11042, epic 11037), and the value of the
/// `advanced.quantTier` label that selects it.
///
/// A DISTINCT, user-selectable tier — **not** an int4-affine equivalent and never an auto-swap of `q4`
/// (epic 11037 SC#5 / the sc-11042 Option A decision). NVFP4 is E2M1 4-bit elements over 16-element
/// blocks with FP8-E4M3 micro-scales + an FP32 per-tensor scale (~4.5 effective bits/weight), a
/// different numeric regime from `q4`; auto-selecting it for a `q4` pick on Blackwell would silently
/// change that tier's output.
pub(crate) const NVFP4_TIER: &str = "nvfp4";

/// The `candle.vramGbByTier` key of the INT8-ConvRot tier (sc-9300), and the tier IDENTITY the VRAM
/// gate sizes a ConvRot render against (sc-12425).
///
/// Like [`NVFP4_TIER`], this is a tier identity with **no honest `mlxQuantize` integer** — the
/// online-rotation int8 DiT is not a point on the bits ladder, so the picker sends
/// `advanced.convRot: true` instead ([`wants_krea_convrot`]). NVFP4's doc calls that "exactly the
/// sc-9300 `convRot` precedent"; sc-12425 is that precedent finally reaching the gate.
///
/// **Why this const has to exist (sc-12425).** `vram_gate::requested_tier_key` is bits-derived and
/// returns only `nvfp4`/`bf16`/`q8`/`q4`, so a ConvRot request — carrying no `mlxQuantize` — fell to
/// its `None => "q8"` arm and was sized against `vramGbByTier["q8"]`. That is the identical aliasing
/// sc-11042 fixed for NVFP4, **but with the sign flipped**: q8 OVER-predicts NVFP4 (a spurious
/// `TooBig`/`Offload`, never an OOM), and UNDER-predicts INT8-ConvRot. Measured on a real trunk
/// (sc-12381, sm_120, 1024²/8-step): the tier peaks at **42.9 GB** while the q8 row predicts
/// 35.9 + 2.0 headroom = 37.9 GB — permissive by 5.0 GB, i.e. it admits a load that OOMs.
/// `vram_gate::predicted_peak_gb`'s own doc names the hazard: "an under-prediction admits a load that
/// can OOM".
///
/// The `vramGbByTier["int8-convrot"]` row existed since sc-9300 but **nothing ever read it** — which is
/// why its unmeasured 31.0 estimate survived without a symptom: a dead row cannot be wrong out loud.
///
/// Candle-lane only — UNLIKE [`NVFP4_TIER`], which is un-gated because macOS-compiled fns
/// (`nvfp4_selected`, `preferred_tier`, …) use it. This const's ONLY users are candle-only
/// (`gate_tier_key`, `vram_gate`), so on the macOS/MLX build it would be dead code (clippy `-D warnings`
/// → error). ConvRot is a candle-only tier (sm_89, sc-9300), so nothing on the MLX path references it.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) const INT8_CONVROT_TIER: &str = "int8-convrot";

/// Whether the request EXPLICITLY asked for the NVFP4 tier, via the `advanced.quantTier: "nvfp4"`
/// label (sc-12006 established the label as asset telemetry; sc-11042 makes it a selection input).
///
/// NVFP4 rides `quantTier` and NOT `advanced.mlxQuantize` because `mlxQuantize` is a BITS-VALUED knob:
/// every consumer parses it as an integer that NAMES A TIER (`quant_int` → `<= 0` ⇒ bf16, `<= 4` ⇒ q4,
/// else q8), so **no integer is honest for NVFP4** — `4` would select the int4-affine `q4` tier, and
/// every other value names bf16/q8. So `mlxQuantize` stays `null` for NVFP4 and the tier identity rides
/// this distinct label, exactly the sc-9300 `convRot` precedent.
///
/// PURE + UNGATED (like [`preferred_tier`], its caller): reads only the request, so the "did the user
/// ask for NVFP4" question is testable on every platform without a GPU. The sm_120 host gate is the
/// separate [`nvfp4_host_eligible`], and the on-disk gate is [`tier_dir_is_nvfp4`]; the tier is
/// SELECTED only when all three hold — see [`nvfp4_selected`].
pub(super) fn nvfp4_requested(request: &ImageRequest) -> bool {
    request
        .advanced
        .get("quantTier")
        .and_then(Value::as_str)
        .is_some_and(|tier| tier.trim().eq_ignore_ascii_case(NVFP4_TIER))
}

/// Whether the RESOLVED tier dir is the distinct `nvfp4/` tier — i.e. whether the NVFP4 tier is
/// actually what [`standard_tier_subdir`] landed on (sc-11042).
///
/// The DISK half of the NVFP4 gate. `standard_tier_subdir` only returns the `nvfp4/` dir when that dir
/// exists with weights in it; a request for a tier that isn't converted yet (sc-11043 owns the
/// convert-at-install loop — **no shipping model packs an `nvfp4/` dir today**) rejoins the clean
/// q8 → bf16 → q4 fallback chain. So "the resolved basename is `nvfp4`" is exactly "the NVFP4 tier is
/// installed AND was chosen", read off the same value the loader will read.
///
/// `None` (tier dir unknown — a lane that resolves no standard-tier subdir: a flat diffusers snapshot,
/// a `modelPath` override, or a caller with no dir in scope) ⇒ **false**. A tier we cannot verify is a
/// tier we must not claim: the conservative direction is to fall through to the request-derived
/// q4/q8/bf16 (which is what such a lane actually loads), never to stamp NVFP4 on it.
///
/// Gated to the lanes that HAVE a quant resolver ([`resolve_quant`]'s cfg): the neither-MLX-nor-candle
/// build resolves no load quant at all, so this would be dead code there (`-D warnings`).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn tier_dir_is_nvfp4(tier_dir: Option<&Path>) -> bool {
    tier_dir
        .and_then(Path::file_name)
        .and_then(|name| name.to_str())
        == Some(NVFP4_TIER)
}

/// Whether this job actually SELECTS the distinct NVFP4 tier (sc-11042, epic 11037 SC#5) — the single
/// predicate behind the tier's load quant ([`resolve_quant`]), its recorded label
/// ([`effective_quant_label`]), and its VRAM sizing (`vram_gate::requested_tier_key`), so those three
/// can never disagree about what ran.
///
/// **All THREE halves are required:**
/// 1. [`nvfp4_requested`] — the user EXPLICITLY named the tier. The SC#5 opt-in: never inferred from
///    `bits`, a manifest default, or hardware detection. Being on Blackwell alone selects nothing.
/// 2. [`nvfp4_host_eligible`] — this host can serve FP4 (sm_120 on the candle lane).
/// 3. [`tier_dir_is_nvfp4`] — the `nvfp4/` tier is INSTALLED and is the dir that resolved.
///
/// Half 3 is why this takes the RESOLVED dir rather than trusting the request: halves 1+2 alone say
/// only what the user WANTED on hardware that COULD serve it, and `standard_tier_subdir` independently
/// falls back to q8 when the `nvfp4/` dir is absent — the shipping case on every model today. Deriving
/// the label from 1+2 therefore stamped `"nvfp4"` on a render that actually ran the **q8** weights: a
/// creative choice falsified in the asset record, precisely the SC#5 aliasing this tier exists to
/// avoid. The label must describe what RAN, so it is read off the same dir the loader loads.
///
/// Gated to [`resolve_quant`]'s cfg, like [`tier_dir_is_nvfp4`]: every caller (the quant resolver, the
/// label, and the candle fit gate) lives on the MLX or candle lane, so compiling it on the
/// neither-backend build would be dead code under `-D warnings`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn nvfp4_selected(
    request: &ImageRequest,
    nvfp4_host: bool,
    tier_dir: Option<&Path>,
) -> bool {
    nvfp4_requested(request) && nvfp4_host && tier_dir_is_nvfp4(tier_dir)
}

/// Whether THIS HOST can serve the NVFP4 tier: the candle lane on a GPU clearing the sm_120
/// consumer-Blackwell compute-cap floor (sc-11042).
///
/// The RUNTIME half of the Blackwell gate, and deliberately defence-in-depth. The web picker already
/// hides the tier unless a live worker advertises the `nvfp4` capability (`gpu.rs`), but `advanced` is
/// free-form pass-through (`rawAdapterSettings` has no strict deserializer), so a hand-crafted API call
/// can put `quantTier: "nvfp4"` on a request to ANY worker. Checking here means such a request falls
/// back cleanly to an installed tier instead of routing an FP4 load at hardware with no FP4 tensor
/// cores. Mirrors ConvRot's belt-and-braces (`int8_convrot` capability + engine-side `ensure_int8_floor`).
///
/// Always `false` off the candle lane: macOS/MLX (Metal has no FP4 hardware — explicitly out of scope
/// for epic 11037, and the runtime's MLX side `reject_quant`s NVFP4) and the non-candle build (no FP4
/// compute) can never serve it, so the tier is never selected there.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn nvfp4_host_eligible() -> bool {
    crate::gpu::compute_cap_meets_nvfp4(crate::gpu::cached_compute_cap())
}

#[cfg(not(all(not(target_os = "macos"), feature = "backend-candle")))]
pub(super) fn nvfp4_host_eligible() -> bool {
    false
}

/// The tier subdir name a request prefers, given its explicit `advanced.mlxQuantize` `bits`, the
/// model's per-model quality `floor` (`mlx.minQualityTier`, sc-10731 — `None` = no floor), and whether
/// the request is a CANDIDATE for the distinct NVFP4 tier (`nvfp4`, sc-11042 — an explicit
/// [`nvfp4_requested`] label AND an [`nvfp4_host_eligible`] host).
///
/// `nvfp4` here is deliberately the TWO-part gate, NOT the fully-resolved [`nvfp4_selected`]: this
/// function is what CHOOSES the tier dir, so it necessarily runs before any dir exists to probe (asking
/// for `nvfp4_selected` would be circular). The third half of the gate — the `nvfp4/` tier dir actually
/// being installed — is the caller's own `present()` fallback chain, which is exactly what
/// [`nvfp4_selected`] reads back afterwards to confirm what this resolver landed on. So a host-eligible
/// NVFP4 pick with no converted tier on disk falls through this function's chain to an installed tier,
/// and `nvfp4_selected` then reports `false` for it — the two agree by construction.
///
/// An explicit, host-eligible **NVFP4** pick (`nvfp4 == true`) resolves the distinct [`NVFP4_TIER`] and
/// short-circuits the bits map below — NVFP4 has no honest `mlxQuantize` integer, so it cannot be
/// expressed there (epic 11037 SC#5 / sc-11042 Option A). It is NOT floor-clamped: like every explicit
/// pick it is a deliberate creative choice, honored as asked. `nvfp4 == false` — every request that did
/// not explicitly ask for NVFP4, which is all of them today — leaves this function's behavior **exactly**
/// as it was: the mapping below is untouched, so no existing tier's resolution changes (guarded by
/// `preferred_tier_bits_map_is_unchanged_by_the_nvfp4_tier`).
///
/// An EXPLICIT bits pick maps directly (`<= 0` → bf16, `> 4` → q8, `1..=4` → q4) and is HONORED even below
/// the floor — a quant tier is a deliberate quality/creative choice, so the worker never silently
/// overrides it (the web surfaces a non-blocking advisory on a below-floor pick instead). With NO
/// explicit pick, the app-wide default (Q8, epic 10721 / sc-10726) is CLAMPED UP to the floor: a floored
/// model (Anima base/aesthetic = q8) never lets the plain default land below the floor. The floor only
/// ever RAISES the default — a floor at or below Q8 leaves the Q8 default unchanged — and the caller's
/// clean-tier fallback chain still caps the result at what's installed (a floor tier not on disk falls
/// to the best installed tier). This is the SHARED default-tier logic behind both [`standard_tier_subdir`]
/// and [`anima_tier_subdir`]; it REPLACES the sc-10714 anima-specific `None => "q8"` hardcode so Anima's
/// q8 default is now floor-driven from the manifest, not a resolver special-case.
pub(super) fn preferred_tier(bits: Option<i64>, floor: Option<&str>, nvfp4: bool) -> &'static str {
    // The distinct NVFP4 tier (sc-11042). Checked FIRST and returned as-is: it is not a point on the
    // bf16/q8/q4 fidelity ladder, so it takes no part in the floor clamp or the bits map below.
    if nvfp4 {
        return NVFP4_TIER;
    }
    match bits {
        Some(b) if b <= 0 => "bf16",
        Some(b) if b > 4 => "q8",
        Some(_) => "q4",
        None => match floor {
            Some(f) if tier_quality_rank(f) > tier_quality_rank("q8") => tier_static_name(f),
            _ => "q8",
        },
    }
}

/// The model's per-model quality FLOOR (`mlx.minQualityTier`, sc-10731): the MINIMUM-fidelity tier a
/// DEFAULT resolution may land on, read from the request's forwarded manifest entry. `None` (field
/// absent) means no floor — the app-wide default stands (e.g. Anima turbo is q4-tolerant and declares
/// none). Only `bf16`/`q8`/`q4` are honored; any other value is ignored. Ungated (like
/// [`standard_tier_subdir`], its ungated caller) so every build config that compiles the resolver also
/// compiles this.
pub(super) fn min_quality_floor(request: &ImageRequest) -> Option<&str> {
    request
        .model_manifest_entry
        .get("mlx")
        .and_then(|mlx| mlx.get("minQualityTier"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|tier| tier_quality_rank(tier) > 0)
}

/// The component subdirs a diffusers-style tier snapshot declares in its own `model_index.json`.
///
/// `None` when the tier ships no readable `model_index.json` — a flat unified turnkey (SenseNova-U1
/// MoT, sc-8771) roots its weights + `tokenizer.json` directly in the tier dir and has no component
/// tree to verify, so callers treat `None` as "nothing to check" and keep the backbone-only probe.
///
/// KNOWN LIMIT (sc-12279): that also means a tier torn badly enough to lack `model_index.json`
/// ITSELF is unverifiable and still resolves on its backbone alone. Deliberate — absence of the index
/// is not evidence the tier is torn, and inferring "component-shaped but no index ⇒ torn" would let a
/// perfectly good tier lose the chain to a sibling that merely ships an index, which is a worse bug
/// than the one this fixes. We only ever demote a tier on POSITIVE evidence: its own index naming a
/// component that is not on disk.
/// Moved to `sceneworks_core::mlx_tier_completeness` (sc-14980) so this resolver, rust-api's catalog
/// completeness, and the training-base gate all read a tier's declared component set the SAME way —
/// a split tier that ships only the components it owns must look identical to all three. Re-exported
/// here rather than reimplemented, for the same anti-drift reason the per-family predicates are
/// shared (sc-13513).
use sceneworks_core::mlx_tier_completeness::tier_declared_components;

/// Whether every component `dir`'s own `model_index.json` declares is actually on disk (sc-12279).
///
/// Presence is "the subdir holds at least one non-hidden entry", NOT "holds weights": `tokenizer/`
/// and `scheduler/` are config-only. Hidden entries don't count for the same reason the backbone
/// probes skip them — a dir holding only an AppleDouble `._tokenizer.json` sidecar has no tokenizer
/// (SceneWorks#1333).
pub(super) fn tier_components_present(dir: &Path) -> bool {
    let Some(components) = tier_declared_components(dir) else {
        return true;
    };
    components
        .iter()
        .all(|component| dir_has_visible_entry(&dir.join(component)))
}

/// Whether `dir` is a directory holding at least one non-hidden entry.
fn dir_has_visible_entry(dir: &Path) -> bool {
    std::fs::read_dir(dir).is_ok_and(|entries| {
        entries
            .flatten()
            .any(|entry| !sceneworks_core::lora_family::is_hidden_file(&entry.path()))
    })
}

// The per-family tier-completeness predicates for the no-`model_index` MLX turnkeys
// (`anima_tier_complete` / `boogu_tier_complete` / `sana_tier_complete`, plus their
// `dir_has_visible_file_ending` helper) live in `sceneworks_core::mlx_tier_completeness` and are
// called fully-qualified below. They are shared with rust-api's catalog completeness so the worker's
// tier resolvers and the /models `installed`-vs-`incomplete` report cannot drift apart (sc-13513).

/// Whether the tier dir `resolve_weights_dir` resolved for `request` is COMPLETE — every component its
/// loader reads is on disk. Used ONLY by [`mlx_weights_gap`] to turn a torn-tier load (which otherwise
/// dies mid-generation with a raw "No such file or directory") into an actionable PRE-FLIGHT message.
/// After the sc-12279-generalized resolvers, `resolve_weights_dir` already falls back to a complete
/// sibling tier whenever one exists, so this only ever returns `false` when NO complete tier is
/// installed for the model.
///
/// Dispatches to the SAME family completeness predicate the resolvers use. An unrecognized family falls
/// back to [`tier_components_present`] (the `model_index.json` guard), which is `true` for a layout
/// without one — so a dispatch miss degrades to "no extra message" (today's raw error at load), NEVER a
/// false "incomplete" that would block a loadable tier.
#[cfg(target_os = "macos")]
pub(super) fn resolved_tier_is_complete(request: &ImageRequest, dir: &Path) -> bool {
    if is_anima_model(&request.model) {
        return sceneworks_core::mlx_tier_completeness::anima_tier_complete(dir);
    }
    if matches!(
        request.model.as_str(),
        "boogu_image" | "boogu_image_turbo" | "boogu_image_edit"
    ) {
        return sceneworks_core::mlx_tier_completeness::boogu_tier_complete(dir);
    }
    if matches!(request.model.as_str(), "sana_1600m" | "sana_sprint_1600m") {
        return tier_components_present(dir)
            && sceneworks_core::mlx_tier_completeness::sana_tier_complete(dir);
    }
    if let Some(complete) = sensenova_tier_predicate(&request.model) {
        return tier_components_present(dir) && complete(dir);
    }
    tier_components_present(dir)
}

// The SenseNova-U1 id list + its per-tier predicate dispatcher live beside the predicates themselves in
// `sceneworks_core::mlx_tier_completeness` (`SENSENOVA_MODELS` / `sensenova_tier_predicate`), so this
// resolver and rust-api's catalog gate on ONE list — a per-crate copy is how one of the two gates ends
// up un-wired (sc-14432).
use sceneworks_core::mlx_tier_completeness::sensenova_tier_predicate;

/// Walk `chain` (a tier-name preference order) and return the first tier that is safe to load: one
/// that is COMPLETE (`complete` returns true) if any qualifies, else the first that merely clears
/// `present`'s backbone probe. `None` when no tier is installed at all.
///
/// The shared tail of every tier resolver (sc-12279). `present` alone accepts a tier on its backbone,
/// so a torn tier — the transformer landed, `tokenizer/` did not — short-circuited the chain and the
/// loader died on `tokenizer: No such file or directory` even when a complete sibling tier was
/// installed (issue #850's symptom). Running the chain twice, rather than folding completeness into
/// `present`, is deliberate: if `complete` ever misjudges a tier shape we haven't seen, pass 2 lands
/// exactly where the pre-sc-12279 code did, so this can never strand a model that loads today.
/// Duplicates in `chain` (the preferred tier is usually also a fallback) are harmless.
///
/// `complete` is family-specific: the diffusers-turnkey families pass [`tier_components_present`]
/// (reads the tier's own `model_index.json`); families with a bespoke on-disk layout and no
/// `model_index.json` (Anima's `diffusion_models/ + text_encoders/ + vae/`, Boogu's packed subfolders,
/// InstantID's `unet/`) pass their own predicate so the completeness half is not a silent no-op for
/// them (the pre-generalization gap that let those families still short-circuit onto a torn tier).
pub(super) fn pick_loadable_tier(
    chain: &[&str],
    present: &dyn Fn(&str) -> Option<PathBuf>,
    complete: &dyn Fn(&Path) -> bool,
) -> Option<PathBuf> {
    let first = |probe: &dyn Fn(&str) -> Option<PathBuf>| chain.iter().find_map(|name| probe(name));
    if let Some(dir) = first(&|name: &str| present(name).filter(|dir| complete(dir))) {
        return Some(dir);
    }
    // No complete tier: load the torn one anyway (pre-sc-12279 behavior) but say so — the loader
    // error it usually raises names a missing file, never the tier that lacks it.
    let torn = first(present)?;
    tracing::warn!(
        tier = %torn.display(),
        "Tier is missing components it should ship, and no complete tier is installed. Loading it \
         anyway; re-download the tier if load fails."
    );
    Some(torn)
}

/// Pick the engine-complete tier subdir of a standard SceneWorks quant-matrix turnkey `root`:
/// `bf16/` when the request opts out of quantization (`advanced.mlxQuantize <= 0` / "none"), `q8/`
/// when it opts into Q8 (`> 4`), `q4/` for an explicit Q4 pick (`1..=4`), else — with NO explicit
/// `mlxQuantize` — the **`q8/`** default (epic 10721 / sc-10726: the app-wide gen-time default tier,
/// matching [`resolve_quant`]'s Q8 default and [`anima_tier_subdir`], replacing the old blind q4).
/// Falls back through the clean tiers first (q8 → bf16 → q4 → `root`) so a partial install never
/// silently lands on the low-fidelity q4 (and a fully-absent turnkey surfaces as a load error). The
/// Q8 default is CLAMPED to what's installed: with only `q4/` on disk it resolves q4, so a heavy model
/// never OOMs on a tier the user didn't download.
///
/// Tier presence is filename-agnostic: a tier is "present" when its backbone component holds any
/// `*.safetensors` (packed single-file OR a `*-00001-of-*.safetensors` shard) or a `*.index.json`
/// (dense sharded). The backbone component is `transformer/` for the DiT turnkeys
/// (flux/qwen/z-image/sd3.5) or `unet/` for the SDXL-family turnkeys (sc-8746) — SDXL packs its UNet
/// under `unet/`, never `transformer/`. This covers every backbone regardless of its packed filename
/// (`diffusion_pytorch_model.safetensors`, `model.safetensors`, …), so a new model needs only a
/// [`STANDARD_TIER_MODELS`] entry (or `mlx.standardTierLayout`), no bespoke resolver.
///
/// Unified-model turnkeys (SenseNova-U1 MoT, sc-8771) have NO component subdir: the whole backbone
/// is a flat `model.safetensors` (or sharded `*.index.json`) directly in the tier dir. The presence
/// check also accepts weights at the tier root so a flat unified tier resolves like a component one.
pub(super) fn standard_tier_subdir(root: &Path, request: &ImageRequest) -> PathBuf {
    standard_tier_subdir_gated(root, request, nvfp4_host_eligible())
}

/// [`standard_tier_subdir`] with the NVFP4 **host** gate passed in rather than probed (sc-11042).
///
/// Split out ONLY for testability, and the injected value is deliberately the HARDWARE fact
/// (`nvfp4_host` = "this host clears the sm_120 floor"), not the finished decision: the live
/// compute-cap probe is the one thing a test can't control, while reading the request's `quantTier`
/// label stays real. So a test can drive both host classes and still exercise the actual
/// request-parsing + SC#5 opt-in. Production has exactly one caller ([`standard_tier_subdir`]), which
/// passes the real probe.
pub(super) fn standard_tier_subdir_gated(
    root: &Path,
    request: &ImageRequest,
    nvfp4_host: bool,
) -> PathBuf {
    let bits = request
        .advanced
        .get("mlxQuantize")
        .and_then(|v| v.as_i64().or_else(|| v.as_str()?.trim().parse().ok()));
    // A component dir "has weights" when it holds a packed/dense safetensors or a shard index.
    // Hidden entries don't count: a dir holding only a `._model.safetensors` AppleDouble sidecar has
    // no weights, and reporting otherwise routes the loader at a tier it cannot load
    // (SceneWorks#1333).
    let component_has_weights = |dir: &Path| -> bool {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return false;
        };
        entries.flatten().any(|entry| {
            if sceneworks_core::lora_family::is_hidden_file(&entry.path()) {
                return false;
            }
            let file = entry.file_name();
            let name = file.to_string_lossy();
            name.ends_with(".safetensors") || name.ends_with(".index.json")
        })
    };
    let present = |name: &str| -> Option<PathBuf> {
        let dir = root.join(name);
        // DiT turnkeys pack the backbone under `transformer/`; SDXL-family turnkeys under `unet/`.
        // Unified-model turnkeys (SenseNova-U1 MoT, sc-8771) have NO component subdirs — the whole
        // backbone is a flat `model.safetensors` (or sharded `*.index.json`) directly in the tier
        // dir. Accept any of the three so a flat unified tier resolves like a component one.
        let has_backbone = component_has_weights(&dir.join("transformer"))
            || component_has_weights(&dir.join("unet"))
            || component_has_weights(&dir);
        has_backbone.then_some(dir)
    };
    // No explicit selection → the app-wide q8 default (epic 10721 / sc-10726), CLAMPED UP to the model's
    // per-model quality floor (`mlx.minQualityTier`, sc-10731 — raises only, never lowers); bits<=0
    // (advanced.mlxQuantize: 0 / "none") → bf16; bits>4 → q8; else (an explicit 1..=4) the q4 the user
    // asked for (an explicit below-floor pick is honored — the web flags it, the worker never overrides).
    // Fallback prefers the clean tiers (q8 → bf16 → q4) so a partial install never silently lands on the
    // washed q4, and the (possibly floored) default is clamped to what's on disk.
    //
    // An explicit, host-eligible NVFP4 pick (sc-11042) prefers the distinct `nvfp4/` tier dir and then
    // rejoins the SAME clean-tier fallback chain, so a request for a tier that isn't converted yet
    // (sc-11043 owns the convert-at-install loop; no shipping model packs an `nvfp4/` dir today) lands
    // on an installed tier exactly like an uninstalled q4/bf16 pick does — never a load error, and never
    // an FP4 load on hardware that can't serve it (`nvfp4` is already false off Blackwell).
    //
    // That NVFP4 fallback is currently NOT event-surfaced, and deliberately so — say what is true here:
    // `reconcile_resolved_tier_quant` (which `warn!`s + fires `quant_tier_downgraded` on a tier
    // downgrade) is `#[cfg(target_os = "macos")]`, while `nvfp4_host_eligible()` is hard-`false` on
    // macOS — so the reconcile path and an NVFP4 pick are MUTUALLY EXCLUSIVE by construction and nothing
    // reconciles NVFP4 on any lane. What keeps the fallback HONEST instead is [`nvfp4_selected`]: the
    // recorded label + the load quant are gated on this resolver's OWN output (the resolved dir), so a
    // pick that lands on q8 is recorded `"q8"` — the asset record tells the truth even with no event.
    // Wiring a candle-lane reconcile so the downgrade is also OBSERVABLE (an event, not just an honest
    // record) is worth doing when a tier is actually converted; sc-11043 owns that tier.
    //
    // BOTH halves are required HERE (the SC#5 opt-in): the user explicitly named the tier AND the host
    // can serve it. Neither alone selects NVFP4. The third half — the tier is actually installed — is
    // this function's own `present()` chain below, which is what [`nvfp4_selected`] reads back.
    let preferred = preferred_tier(
        bits,
        min_quality_floor(request),
        nvfp4_requested(request) && nvfp4_host,
    );
    // sc-12279: prefer a tier whose component tree is fully on disk, so a torn tier falls through to a
    // complete sibling instead of reaching the loader. Diffusers-shaped turnkeys (flux/qwen/sd3.5/…)
    // ship a per-tier `model_index.json` that `tier_components_present` reads. SANA is the exception:
    // its `SceneWorks/Sana_*_mlx` turnkeys ship NO `model_index.json`, so that guard is a no-op for it —
    // fold in the concrete `sana_tier_complete` check (transformer + VAE + Gemma TE + tokenizer) so a
    // torn SANA tier is demoted too. Flat unified tiers (SenseNova-U1 MoT) ship no `model_index.json`
    // either, and their backbone probe passes on the tier root, so fold in their predicate
    // ([`sensenova_tier_predicate`]: whole backbone + config + an own-or-sibling `tokenizer.json`, plus
    // the distill marker for the `_fast` twins) for the same reason — otherwise a torn SenseNova tier
    // short-circuits the chain and dies at load (sc-14432).
    let is_sana = matches!(request.model.as_str(), "sana_1600m" | "sana_sprint_1600m");
    let sensenova_complete = sensenova_tier_predicate(&request.model);
    let complete = |dir: &Path| {
        tier_components_present(dir)
            && (!is_sana || sceneworks_core::mlx_tier_completeness::sana_tier_complete(dir))
            && match sensenova_complete {
                Some(complete) => complete(dir),
                None => true,
            }
    };
    pick_loadable_tier(&[preferred, "q8", "bf16", "q4"], &present, &complete)
        .unwrap_or_else(|| root.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preferred_tier_honors_explicit_bits_and_clamps_only_the_default() {
        assert_eq!(preferred_tier(Some(4), Some("bf16"), false), "q4");
        assert_eq!(preferred_tier(Some(8), Some("bf16"), false), "q8");
        assert_eq!(preferred_tier(None, Some("bf16"), false), "bf16");
        assert_eq!(preferred_tier(None, Some("q4"), false), "q8");
    }

    #[test]
    fn loadable_tier_prefers_complete_then_preserves_present_fallback() {
        let present = |tier: &str| matches!(tier, "q8" | "q4").then(|| PathBuf::from(tier));
        let complete = |tier: &Path| tier == Path::new("q4");

        assert_eq!(
            pick_loadable_tier(&["q8", "q4"], &present, &complete),
            Some(PathBuf::from("q4"))
        );
        assert_eq!(
            pick_loadable_tier(&["q8"], &present, &|_| false),
            Some(PathBuf::from("q8"))
        );
    }
}
