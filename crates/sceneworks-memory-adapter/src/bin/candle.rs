#[cfg(target_os = "macos")]
compile_error!("memory-candle-adapter is supported only on CUDA hosts");

use candle_gen::testkit::VramProbe;
use runtime_cuda::gen_core::{
    GenerationRequest, LoadShape, LoadSpec, MemoryBudget, MemoryCacheState, MemoryGeometry,
    MemoryMode, MemoryNumericTier, MemoryOptimizationAuthority, MemoryPhase, MemoryRunContext,
    MemoryRunOutcome, MemorySafetyDecision, MemorySelection, MemoryStrategy,
    MemoryStrategyParameters, OffloadPolicy, Precision, Progress, Quant, TransformerComponent,
    WeightsSource,
};
use sceneworks_memory_adapter as protocol;
use serde_json::{json, Map, Value};
use std::path::PathBuf;
use std::process::Command;

const KREA_ID: &str = "krea_2_turbo";
const KREA_PLAIN_EXECUTION_PATH: &str = "the Candle Krea base-only text-to-image path";
/// The label the Krea arm refuses a non-still geometry under (sc-18808); see
/// [`still_calibration_label`].
const KREA_STILL_CALIBRATION: &str = "Candle Krea base calibration";
const QWEN_ID: &str = "qwen_image";
const QWEN_PLAIN_EXECUTION_PATH: &str = "the Candle Qwen-Image base-only text-to-image path";
/// The label the Qwen arm refuses a non-still geometry under (sc-18808); see
/// [`still_calibration_label`].
const QWEN_STILL_CALIBRATION: &str = "Candle Qwen base calibration";
const GIB: u64 = 1024 * 1024 * 1024;
const MIB: u64 = 1024 * 1024;

// ─── sc-19057: the FIRST candle VIDEO arm (`candle:wan2_2_ti2v_5b`) ──────────────────────────────
//
// Every arm above this line is an image arm and keeps its `geometry.frames == 1` refusal
// (`protocol::validate_still_geometry`, sc-18808). This one is the single Candle arm allowed to
// accept a multi-frame geometry, and it pays for that by validating Wan's OWN declared envelope
// through `protocol::validate_video_geometry` — the video counterpart hoist (sc-19057) of the same
// shared guard, so the two video lanes cannot drift into reporting different reasons for the same
// class of bad plan row.
//
// WHY WAN AND NOT LTX. The story left the family open ("likely LTX or wan"). At the pinned inference
// revision `candle-gen-ltx` registers NO `MemoryStrategyContract` at all — no `memory_strategy.rs`,
// no `register_memory_strategy` call, and its `Components` struct holds the Gemma TE, AvDiT and both
// VAEs in one cache it never clears (`supports_sequential_offload: false`). `candle-gen-wan` is the
// ONLY candle video crate that registers one (`memory_strategy::MEMORY_REGISTRATION`, SC-19223):
// three Implemented rungs, a phase-envelope formula over `FrameCount`, per-load-policy calibration
// identities, and a `MemoryMode::Other("text_to_video")` route gate. A capture needs an admission
// seam to interrogate, so LTX would have meant building the provider contract in `inference` first —
// a two-repo pin-lockstep change this story does not own.

/// The candle Wan2.2 TI2V-5B ENGINE id — the key `catalog.media().load(..)` resolves.
const WAN_PROVIDER: &str = "wan2_2_ti2v_5b";
/// The BUILTIN-CATALOG model id whose `limits` block is the source of truth for the envelope below,
/// and which `scripts/fit-ltx-temporal-form.mjs` resolves `modelFamily` from
/// (`manifest.models.find(model => model.id === record.target.modelId)`).
///
/// For `mlx:ltx_2_3` the catalog id and the engine id coincide, which let that arm check one field
/// twice. Here they genuinely differ, so the plan target carries BOTH and this arm validates the
/// exact pair — a record whose `modelId` were the engine id would make the fit throw
/// `model wan2_2_ti2v_5b is absent from builtin.models.jsonc` hours after the capture burned.
const WAN_MANIFEST_MODEL_ID: &str = "wan_2_2";
const WAN_PLAIN_EXECUTION_PATH: &str = "the Candle Wan2.2 TI2V-5B base-only text-to-video path";
/// How this arm names itself in every geometry refusal.
const WAN_CALIBRATION_LABEL: &str = "Candle Wan2.2 TI2V-5B calibration";
/// The four constants below are copies of the `limits` block of the `wan_2_2` entry in
/// **`config/manifests/builtin.models.jsonc`**, which is their source of truth.
///
/// The binding lives on the node side for the same reason the MLX LTX arm's does: this crate carries
/// two dependencies on purpose and cannot reach `sceneworks-core`'s JSONC reader. `npm run check`
/// runs `the Candle Wan arm's manifest constants match the shipped wan_2_2 limits` in
/// `scripts/platform-review-contracts.test.mjs`, which parses the manifest, parses these
/// declarations out of this file, and re-derives [`WAN_FRAME_ENVELOPE`] from the manifest's own
/// durations x fps.
const WAN_RESOLUTIONS: [(u32, u32); 3] = [(832, 480), (1280, 704), (704, 1280)];
/// `limits.durations` and `limits.fps`, verbatim. Together they span the frame envelope below.
const WAN_DURATIONS_SECONDS: [u32; 5] = [4, 5, 6, 7, 8];
const WAN_FPS: [u32; 2] = [16, 24];
/// `limits.maxPixels`, verbatim — upstream's own budget for the 5B (`MAX_AREA_CONFIGS`), and the
/// bound `candle-gen-wan`'s `safety_check` enforces as `MAX_AREA_5B`.
const WAN_MAX_PIXELS: u64 = 901_120;
/// The `wan_2_2` entry deliberately declares NO `limits.requiresDimensionsMultipleOf`: core's
/// default floor is already 32, which is exactly `vae_stride_spatial (16) x patch (2)` — candle's
/// `SIZE_MULTIPLE`. The manifest binding test asserts that elision rather than a value, so
/// re-declaring a different stride there reds instead of silently widening this arm.
const WAN_DIMENSION_MULTIPLE: u32 = 32;
/// The z48 VAE's temporal downsample (`VAE_STRIDE_TEMPORAL`): latent `T = (frames - 1) / 4 + 1`, and
/// the engine's `safety_check` hard-rejects any `frames` where `(frames - 1) % 4 != 0`.
const WAN_TEMPORAL_SCALE: u32 = 4;
/// `limits.fps` declares `[16, 24]`, but `WanMemoryRequestScope::validate_request` rejects any
/// cadence other than `DEFAULT_FPS`. The 16 fps column is therefore *shipped* and *not capturable*,
/// which is a fact about the calibrated route rather than about the product — recorded here so a
/// plan row that asks for it is refused with that sentence instead of dying inside the provider.
const WAN_CALIBRATED_FPS: u32 = 24;
/// The cadences [`wan_frame_envelope`] derives over — the capturable subset of [`WAN_FPS`], not the
/// whole declared column.
///
/// sc-19057 review: deriving over the full `durations x fps` cross product admitted frame counts
/// that only the refused 16 fps cadence reaches (61, 77, 109, 125), so a row like
/// `832x480 f61 fps24` passed the geometry guard for a geometry no product request can produce —
/// the same drift class the fps refusal exists to prevent. The envelope an arm validates against is
/// a statement about what that arm may CAPTURE, so it is derived over the capturable cadence alone
/// and the two rules now agree.
const WAN_CAPTURABLE_FPS: [u32; 1] = [WAN_CALIBRATED_FPS];
/// One fixed seed for every `candle:wan2_2_ti2v_5b` fixture
/// (`wan2-2-ti2v-5b-candle-<tier>-<width>x<height>-f<frames>-fps<fps>-seed19057`).
const WAN_SEED: u64 = 19057;
/// Two steps are intentional, exactly as in the five-rung reference arm: the first `Step` callback
/// closes a conservative conditioning envelope and the second gives denoise its own measured
/// interval before `Decoding`. The denoise peak is the per-step attention transient, so it is
/// reached inside any step count — sc-13175 measured this model's ceiling at 20 steps and the
/// mechanism does not change at 2.
const WAN_STEPS: u32 = 2;
const WAN_PROMPT: &str =
    "a slow dolly shot along a rain-slicked city street at night, neon reflections on wet asphalt";
/// Candle Wan quality here is repeat determinism on ONE loaded provider: the measured render and a
/// clean warm repeat select the same code path, so they must agree to within allocator jitter. The
/// envelope is deliberately the same numeric envelope the MLX LTX video arm adopted rather than a
/// looser one invented for CUDA — 3/255 max and 1/255 mean sit far above same-process floating-point
/// jitter and far below the mandatory +64/255 broad-bias mutation, which must breach all three. The
/// values are restated under WAN names because the record embeds them as `maximumErrorThreshold` and
/// friends: a `candle:wan2_2_ti2v_5b` receipt must not be traceable to a constant asserting an LTX
/// or FLUX.2 provenance.
const WAN_MAX_THRESHOLD: f64 = 3.0 / 255.0;
const WAN_MEAN_THRESHOLD: f64 = 1.0 / 255.0;
const WAN_RMS_THRESHOLD: f64 = 1.5 / 255.0;
/// The broad bias the falsifiability check applies to the measured clip. It must breach all three
/// thresholds; if it does not, the thresholds are not measuring anything and the capture fails.
const WAN_NEGATIVE_MUTATION_BIAS: u8 = 64;
/// What this arm does NOT execute, named on every scenario it leaves `not_run`.
const WAN_LIFECYCLE_BLOCKER: &str = concat!(
    "sc-19057 executes the loaded-contract admission scenarios plus a clean warm repeat; typed ",
    "cancellation, authorized-error injection, and their recovery renders are additional full-clip ",
    "video renders this arm does not perform, so they have no measurement behind them here"
);

/// Port of `sceneworks_core::video_request::wan_frame_count` — frames FLOOR to the largest `4k + 1`
/// at or below the raw count, with a 5-frame minimum. Duplicated rather than depended on for the
/// same reason the LTX ladder is in `bin/mlx.rs`: `sceneworks-core` pulls a bundled SQLite, an image
/// codec stack and a trash binding into what is otherwise a two-dependency calibration binary.
///
/// Because it is a port and NOT a call, the binding to the shipped ladder takes TWO tests to close:
/// `wan_frame_ladder_port_matches_the_transcribed_shipped_ladder` here pins the 10 shipped
/// `(duration, fps)` pairs against this port, and
/// `wan_frame_count_matches_the_sc_19057_calibration_ladder` in
/// `crates/sceneworks-core/src/video_request.rs` pins the SAME 10 pairs against `wan_frame_count`
/// itself. That crate is a workspace default member, so a change to the shipped ladder reds under a
/// plain `cargo test`.
const fn wan_snapped_frame_count(raw_frames: u32) -> u32 {
    let raw = if raw_frames < 1 { 1 } else { raw_frames };
    let floored = raw - ((raw - 1) % WAN_TEMPORAL_SCALE);
    if floored < 5 {
        5
    } else {
        floored
    }
}

/// The closed frame envelope this arm can actually CAPTURE: the declared `limits.durations` crossed
/// with [`WAN_CAPTURABLE_FPS`], through the ladder the product itself uses. Derived rather than
/// written down so the bounds cannot drift away from the arrays above.
///
/// It is deliberately NOT the full `durations x fps` product — see [`WAN_CAPTURABLE_FPS`]. It is
/// still a SPAN and therefore still a superset of the five rungs `[93, 117, 141, 165, 189]` the
/// capturable cadence actually reaches; the shared [`protocol::VideoGeometryEnvelope`] carries a
/// closed interval rather than a set, and the exact ladder membership of every committed plan row
/// is bound instead by `the Candle Wan arm's manifest constants match the shipped wan_2_2 limits`
/// in `scripts/platform-review-contracts.test.mjs`.
const fn wan_frame_envelope() -> (u32, u32) {
    let (mut minimum, mut maximum) = (u32::MAX, 0);
    let mut duration = 0;
    while duration < WAN_DURATIONS_SECONDS.len() {
        let mut fps = 0;
        while fps < WAN_CAPTURABLE_FPS.len() {
            let frames =
                wan_snapped_frame_count(WAN_DURATIONS_SECONDS[duration] * WAN_CAPTURABLE_FPS[fps]);
            if frames < minimum {
                minimum = frames;
            }
            if frames > maximum {
                maximum = frames;
            }
            fps += 1;
        }
        duration += 1;
    }
    (minimum, maximum)
}

const WAN_FRAME_ENVELOPE: (u32, u32) = wan_frame_envelope();

/// Wan's own geometry envelope, which REPLACES the image arms' `frames == 1` refusal for this arm
/// alone. Five independent constraints, each from a stated source: the declared resolution pairs,
/// the 32-px spatial lattice, the `limits.maxPixels` area cap the engine enforces as `MAX_AREA_5B`,
/// the `1 + 4k` temporal lattice the z48 VAE requires, and the `[93, 189]` span the declared
/// durations produce at the one CAPTURABLE cadence through the shipped Wan frame ladder.
///
/// A still geometry (`frames == 1`) is on the lattice but below the envelope floor, so it is refused
/// here too: this arm may not silently capture a single-frame record for a video model.
fn wan_video_envelope() -> protocol::VideoGeometryEnvelope<'static> {
    // The rationale spells the stride out in prose, so a change to the constant that left the
    // sentence behind would make the refusal message lie. Fail at compile time instead.
    const _: () = assert!(
        WAN_TEMPORAL_SCALE == 4,
        "update the temporal rationale prose with the stride"
    );
    protocol::VideoGeometryEnvelope {
        calibration_label: WAN_CALIBRATION_LABEL,
        resolutions: &WAN_RESOLUTIONS,
        dimension_multiple: WAN_DIMENSION_MULTIPLE,
        max_pixels: Some(WAN_MAX_PIXELS),
        temporal_scale: WAN_TEMPORAL_SCALE,
        temporal_rationale: "the Wan z48 video VAE is 4x causal in time",
        frame_envelope: WAN_FRAME_ENVELOPE,
        batch_rationale: "the provider advertises max_count 1",
    }
}

/// Defense-in-depth mirror of [`still_calibration_label`], plus the T2V-specific target shape.
/// [`run`] routes by provider id, but this arm hardcodes the Wan contract, so a foreign caller must
/// be refused BY NAME here rather than misrouted into it.
fn validate_wan_target(request: &Value) -> Result<protocol::VideoGeometry, String> {
    let target = protocol::planned(request)?
        .get("target")
        .and_then(Value::as_object)
        .ok_or_else(|| "planned.target must be an object".to_owned())?;
    let field = |name: &str| {
        target
            .get(name)
            .and_then(Value::as_str)
            .ok_or_else(|| format!("planned.target.{name} must be a string"))
    };
    let provider = field("provider")?;
    if provider != WAN_PROVIDER {
        return Err(format!(
            "{WAN_CALIBRATION_LABEL} does not implement provider {provider:?}"
        ));
    }
    let model_id = field("modelId")?;
    if model_id != WAN_MANIFEST_MODEL_ID {
        return Err(format!(
            "{WAN_CALIBRATION_LABEL} requires modelId {WAN_MANIFEST_MODEL_ID:?}, the builtin-catalog \
             id the video-curve fit reads modelFamily from; the ENGINE id {WAN_PROVIDER:?} is \
             planned.target.provider. Got {model_id:?}"
        ));
    }
    let mode = field("mode")?;
    if mode != "text_to_video" {
        return Err(format!(
            "{WAN_CALIBRATION_LABEL} requires reference-free text_to_video mode, got {mode:?}"
        ));
    }
    for name in ["referenceCount", "reference_count"] {
        if let Some(value) = target.get(name) {
            if value.as_u64() != Some(0) {
                return Err(format!(
                    "{WAN_CALIBRATION_LABEL} requires {name} == 0 when declared"
                ));
            }
        }
    }
    for name in ["hasReference", "has_reference"] {
        if let Some(value) = target.get(name) {
            if value.as_bool() != Some(false) {
                return Err(format!(
                    "{WAN_CALIBRATION_LABEL} requires {name} == false when declared"
                ));
            }
        }
    }
    protocol::video_target_geometry(request, &wan_video_envelope())
}

/// The numeric tier this case measures. Scoped to the two SceneWorks-hosted PACKED tiers on purpose:
/// `bf16` is served by the upstream `Wan-AI/Wan2.2-TI2V-5B-Diffusers` snapshot, which has no
/// per-tier subdirectory, so [`protocol::validate_huggingface_snapshot_root`] has nothing to bind an
/// artifact identity to and a bf16 record could not name the bytes it measured.
fn wan_planned_tier(request: &Value) -> Result<&str, String> {
    match protocol::planned(request)?
        .pointer("/target/tier")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.tier must be a string".to_owned())?
    {
        tier @ ("q4" | "q8") => Ok(tier),
        "bf16" => Err(format!(
            "{WAN_CALIBRATION_LABEL} has no bf16 arm: that tier is the upstream dense \
             Wan-AI/Wan2.2-TI2V-5B-Diffusers snapshot, which carries no per-tier subdirectory to \
             bind an artifact identity to. Capture it once it is re-hosted under the SceneWorks \
             tier layout"
        )),
        tier => Err(format!(
            "unsupported {WAN_CALIBRATION_LABEL} numeric tier {tier:?}"
        )),
    }
}

fn wan_numeric_tier(tier: &str) -> Result<MemoryNumericTier, String> {
    let quant = match tier {
        "q4" => Quant::Q4,
        "q8" => Quant::Q8,
        other => {
            return Err(format!(
                "unsupported {WAN_CALIBRATION_LABEL} numeric tier {other:?}"
            ))
        }
    };
    Ok(MemoryNumericTier {
        precision: Precision::Bf16,
        quant: Some(quant),
        component_precision_floors: &[],
    })
}

/// Bind the fixture to the planned tier AND the full rendered geometry, and recover the two request
/// parameters the geometry envelope cannot carry: the output cadence `fps` and the seed.
fn planned_wan_capture(
    request: &Value,
    tier: &str,
    geometry: protocol::VideoGeometry,
) -> Result<(u32, u64), String> {
    let fixture = protocol::planned(request)?
        .get("fixture")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.fixture must be a string".to_owned())?;
    let prefix = format!(
        "wan2-2-ti2v-5b-candle-{tier}-{}x{}-f{}-fps",
        geometry.width, geometry.height, geometry.frames
    );
    let remainder = fixture
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("planned.fixture {fixture:?} must start with {prefix:?}"))?;
    let (fps, seed) = remainder
        .split_once("-seed")
        .ok_or_else(|| format!("planned.fixture {fixture:?} must end with -seed<seed>"))?;
    let fps = fps
        .parse::<u32>()
        .map_err(|error| format!("parse Wan fixture fps {fps:?}: {error}"))?;
    let seed = seed
        .parse::<u64>()
        .map_err(|error| format!("parse Wan fixture seed {seed:?}: {error}"))?;
    if !WAN_FPS.contains(&fps) {
        return Err(format!(
            "planned.fixture declares fps {fps}, which is not one of the declared limits.fps {WAN_FPS:?}"
        ));
    }
    if fps != WAN_CALIBRATED_FPS {
        return Err(format!(
            "planned.fixture declares fps {fps}, but the calibrated Candle Wan route is fixed at \
             {WAN_CALIBRATED_FPS} fps (the provider's request scope rejects any other cadence), so \
             that column of limits.fps is shipped but not capturable"
        ));
    }
    if seed != WAN_SEED {
        return Err(format!(
            "planned.fixture seed {seed} does not match the Candle Wan calibration seed {WAN_SEED}"
        ));
    }
    Ok((fps, seed))
}

/// The rung this case measures. Wan declares only three Implemented mechanisms; the other two are
/// `Missing` in the pinned contract, so a plan row naming one is refused here rather than discovered
/// after a cold model load.
fn wan_planned_memory_strategy(request: &Value) -> Result<MemoryStrategy, String> {
    match protocol::planned_rung(request)? {
        "resident" => Ok(MemoryStrategy::Resident),
        "staged_residency" => Ok(MemoryStrategy::StagedResidency),
        "bounded_decode" => Ok(MemoryStrategy::BoundedDecode),
        missing @ ("bounded_attention" | "bounded_transformer_residency") => Err(format!(
            "{WAN_CALIBRATION_LABEL} rung {missing:?} is Missing in the pinned provider contract \
             (Wan declares neither attention chunking nor transformer-window materialization), so \
             no capture can measure it"
        )),
        other => Err(format!(
            "unsupported {WAN_CALIBRATION_LABEL} rung {other:?}"
        )),
    }
}

/// Read the selected parameter tuple, refusing knobs this provider does not implement.
///
/// The five-rung image arms accept every parameter because Krea implements all four. Wan implements
/// exactly one — the decode tuple — so an `attentionChunkSize` on a Wan plan row is a plan defect,
/// and accepting it silently would emit a record whose `strategy.parameters` advertise a mechanism
/// nothing engaged.
fn wan_planned_selection(request: &Value, tier: &str) -> Result<MemorySelection, String> {
    let strategy = wan_planned_memory_strategy(request)?;
    for unsupported in ["attentionChunkSize", "transformerWindowSize"] {
        if protocol::optional_parameter(request, unsupported)?.is_some() {
            return Err(format!(
                "{WAN_CALIBRATION_LABEL} declares no {unsupported} mechanism; a plan row that \
                 selects one would record a parameter nothing engaged"
            ));
        }
    }
    let decode_tile_edge = protocol::optional_parameter(request, "decodeTileEdge")?;
    let decode_overlap = protocol::optional_parameter(request, "decodeOverlap")?;
    match (strategy, decode_tile_edge, decode_overlap) {
        (MemoryStrategy::BoundedDecode, Some(_), Some(_)) => {}
        (MemoryStrategy::BoundedDecode, _, _) => {
            return Err(format!(
                "{WAN_CALIBRATION_LABEL} bounded_decode requires both decodeTileEdge and \
                 decodeOverlap"
            ))
        }
        (_, None, None) => {}
        _ => {
            return Err(format!(
                "{WAN_CALIBRATION_LABEL} rung {:?} engages no decode tiler, so it must declare no \
                 decode tuple",
                protocol::planned_rung(request)?
            ))
        }
    }
    Ok(MemorySelection {
        strategy,
        parameters: MemoryStrategyParameters {
            decode_tile_edge,
            decode_overlap,
            attention_chunk_size: None,
            transformer_window_size: None,
            transformer_window_component: None,
        },
        tier: wan_numeric_tier(tier)?,
    })
}

/// The exercised sweep domain for one Wan plan row.
///
/// `rangeVerified` is DERIVED, not asserted: a row with no parameter axes has a singleton domain,
/// so its one executed case genuinely is the whole range; a bounded-decode row selects one point out
/// of `DECODE_TILE_EDGES` and has NOT verified the range this record exercised. A parameterized row
/// therefore reports `false`, which is also what stops [`wan_receipt_status`] from promoting it —
/// the harness requires a verified range for `runtime_complete`.
fn wan_complete_sweep(request: &Value) -> Result<Value, String> {
    let parameters = protocol::strategy_parameters(request)?;
    let mut sweep = protocol::reference_sweep(request, "passed")?;
    sweep["rangeVerified"] = json!(parameters.is_empty());
    Ok(sweep)
}

/// Which of the eight required scenarios this run ACTUALLY executed and passed.
///
/// The point of the struct is that a receipt cannot be written except from measured outcomes.
/// sc-18808's own review caught the inverse error on the MLX side — a fragment marking `loadability`
/// as `not_run` while every number in it came out of that very load — and the same class of
/// dishonesty in the other direction (a `passed` scenario nothing executed) is what
/// [`wan_scenarios`] refuses structurally.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct WanScenarioOutcomes {
    loadability: bool,
    exact_fit: bool,
    unknown_budget: bool,
    stale_evidence: bool,
    warm_repeat: bool,
    cancel: bool,
    error: bool,
}

/// Build the scenario array, failing closed on the one shape that would be a well-formed lie.
///
/// Every scenario except `loadability` is a claim about the LOADED provider: the three admission
/// verdicts interrogate `memory_strategy_safety_check` on a live generator, and `warm_repeat`
/// compares two renders from it. So a receipt that reports any of them as `passed` while reporting
/// no completed load is internally impossible, and is refused rather than emitted.
fn wan_scenarios(
    outcomes: WanScenarioOutcomes,
    predicted_bytes: u64,
    blocker: &str,
) -> Result<Value, String> {
    if !outcomes.loadability
        && (outcomes.exact_fit
            || outcomes.unknown_budget
            || outcomes.stale_evidence
            || outcomes.warm_repeat
            || outcomes.cancel
            || outcomes.error)
    {
        return Err(
            "a Candle Wan receipt cannot report an executed scenario without a completed load: \
             every scenario but loadability is a claim about the loaded provider"
                .to_owned(),
        );
    }
    let settled = |name: &str, executed: bool, reason: &str| {
        if executed {
            json!({ "name": name, "result": "passed", "reason": reason })
        } else {
            json!({ "name": name, "result": "not_run", "reason": blocker })
        }
    };
    let exact_fit = if outcomes.exact_fit {
        json!({
            "name": "exact_fit",
            "result": "passed",
            "reason": "the loaded provider contract admitted a budget exactly equal to the measured ceiling",
            "predictedBytes": predicted_bytes,
            "effectiveBudgetBytes": predicted_bytes,
        })
    } else {
        json!({ "name": "exact_fit", "result": "not_run", "reason": blocker })
    };
    Ok(json!([
        exact_fit,
        settled(
            "unknown_budget",
            outcomes.unknown_budget,
            "the loaded provider contract rejected a zero/unknown budget",
        ),
        settled(
            "stale_evidence",
            outcomes.stale_evidence,
            "the loaded provider contract rejected a mutated calibration fingerprint",
        ),
        settled(
            "warm_repeat",
            outcomes.warm_repeat,
            "the selected request scope repeated deterministically within the declared clip-wide envelope",
        ),
        settled(
            "cancel",
            outcomes.cancel,
            "typed cancellation at the selected rung boundary cleaned up and recovered",
        ),
        settled(
            "error",
            outcomes.error,
            "provider fault injection at the selected rung boundary cleaned up and recovered",
        ),
        if outcomes.loadability {
            json!({ "name": "loadability", "result": "passed" })
        } else {
            json!({ "name": "loadability", "result": "not_run", "reason": blocker })
        },
        json!({
            "name": "overlay",
            "result": "not_applicable",
            "reason": "settled below from the declared reference-free target",
        }),
    ]))
}

/// The status a receipt with these outcomes is ENTITLED to, mirroring the two clauses of
/// `memory-calibration-harness.mjs#validateRuntimeComplete` this arm can actually satisfy rather
/// than guessing at the third.
///
/// Runtime activation needs the four admission/loadability scenarios passed, a verified sweep range,
/// and a lifecycle triple the harness accepts. The harness accepts three lifecycle shapes; this arm
/// can emit two of them:
///
/// * entirely `not_run`, and
/// * parity-only — `warm_repeat` passed, `cancel`/`error` `not_run`.
///
/// The harness's third shape, "fully passed", additionally requires `cleanupVerified == true` and
/// `warmFollowUpPassed == true` on the `cancel` and `error` scenarios, and [`wan_scenarios`]'
/// `settled` helper emits neither field — it has no lifecycle injection behind it to measure them
/// from. sc-19057 review: this function previously carried a `lifecycle_passed` branch anyway, which
/// would have promoted a receipt the harness then rejects. Nothing can reach it today, so it was not
/// a live bug, but a mirror that mirrors a clause its own emitter cannot honour is worse than no
/// mirror. A later story that adds cancel/error injection must extend `settled` with those two
/// measured fields and add the branch back in the same change, so the two halves cannot separate.
///
/// So today a zero-axis row reaches runtime activation through the parity-only branch and a
/// bounded-decode row stays gated on its unverified range — both derived rather than hardcoded.
fn wan_receipt_status(outcomes: WanScenarioOutcomes, range_verified: bool) -> &'static str {
    let admitted = outcomes.loadability
        && outcomes.exact_fit
        && outcomes.unknown_budget
        && outcomes.stale_evidence;
    let lifecycle_not_run = !outcomes.warm_repeat && !outcomes.cancel && !outcomes.error;
    let parity_only = outcomes.warm_repeat && !outcomes.cancel && !outcomes.error;
    if admitted && range_verified && (lifecycle_not_run || parity_only) {
        "runtime_complete"
    } else {
        "gated"
    }
}

fn wan_quality_passes(maximum: f64, mean: f64, rms: f64) -> bool {
    maximum <= WAN_MAX_THRESHOLD && mean <= WAN_MEAN_THRESHOLD && rms <= WAN_RMS_THRESHOLD
}

/// Maximum, mean, and root-mean-square absolute error between two clips, in [0,1] units, over every
/// frame. The `runtime_complete` quality shape requires all three.
fn wan_clip_max_mean_rms(
    left: &[runtime_cuda::gen_core::Image],
    right: &[runtime_cuda::gen_core::Image],
) -> Result<(f64, f64, f64), String> {
    if left.len() != right.len() {
        return Err(format!(
            "Wan clip frame-count mismatch: measured={} repeat={}",
            left.len(),
            right.len()
        ));
    }
    let mut maximum = 0.0_f64;
    let mut sum = 0.0_f64;
    let mut sum_squares = 0.0_f64;
    let mut samples = 0_usize;
    for (index, (left, right)) in left.iter().zip(right).enumerate() {
        if left.width != right.width || left.height != right.height {
            return Err(format!(
                "Wan clip frame {index} changed dimensions between renders"
            ));
        }
        if left.pixels.len() != right.pixels.len() {
            return Err(format!(
                "Wan clip frame {index} changed pixel length between renders"
            ));
        }
        for (&left, &right) in left.pixels.iter().zip(&right.pixels) {
            let difference = (f64::from(left) - f64::from(right)).abs() / 255.0;
            maximum = maximum.max(difference);
            sum += difference;
            sum_squares += difference * difference;
            samples += 1;
        }
    }
    if samples == 0 {
        return Err("Wan clip comparison had no samples".to_owned());
    }
    Ok((
        maximum,
        sum / samples as f64,
        (sum_squares / samples as f64).sqrt(),
    ))
}

fn wan_negative_mutation(frame: &runtime_cuda::gen_core::Image) -> runtime_cuda::gen_core::Image {
    let mut mutated = frame.clone();
    for channel in &mut mutated.pixels {
        *channel = channel.wrapping_add(WAN_NEGATIVE_MUTATION_BIAS);
    }
    mutated
}

/// A clip whose first frame is a single flat colour is a decoder failure, not evidence. The MLX LTX
/// receipt asserts `firstFrameNondegenerate: true`; here it is MEASURED before it is asserted.
fn wan_frame_is_nondegenerate(frame: &runtime_cuda::gen_core::Image) -> bool {
    frame
        .pixels
        .first()
        .is_some_and(|first| frame.pixels.iter().any(|channel| channel != first))
}

/// Unwrap the video output, refusing an image-shaped or audio-shaped return.
fn wan_video_frames(
    output: runtime_cuda::gen_core::GenerationOutput,
) -> Result<(Vec<runtime_cuda::gen_core::Image>, u32), String> {
    match output {
        runtime_cuda::gen_core::GenerationOutput::Video { frames, fps, .. } => {
            if frames.is_empty() {
                return Err("Candle Wan render returned no frames".to_owned());
            }
            Ok((frames, fps))
        }
        runtime_cuda::gen_core::GenerationOutput::Images(_) => {
            Err("Candle Wan render returned images, not a video clip".to_owned())
        }
        runtime_cuda::gen_core::GenerationOutput::Audio(_) => {
            Err("Candle Wan render returned an audio track, not a video clip".to_owned())
        }
    }
}

fn wan_generation_request(
    geometry: protocol::VideoGeometry,
    fps: u32,
    seed: u64,
) -> GenerationRequest {
    // Every field the provider's request scope forbids stays at its default: no conditioning, no
    // prompt enhancement, no `video_mode`, no phases. `guidance` is left unset so the engine applies
    // its own `DEFAULT_GUIDANCE` — CFG ON, the shape sc-13175 measured the shipped ceiling under.
    GenerationRequest {
        prompt: WAN_PROMPT.to_owned(),
        width: geometry.width,
        height: geometry.height,
        count: 1,
        seed: Some(seed),
        steps: Some(WAN_STEPS),
        frames: Some(geometry.frames),
        fps: Some(fps),
        ..Default::default()
    }
}

#[derive(Clone)]
struct NvidiaSmi {
    executable: PathBuf,
    physical_id: String,
}

impl NvidiaSmi {
    fn resolve() -> Result<Self, String> {
        let executable = if cfg!(windows) {
            let system_root =
                std::env::var("SystemRoot").unwrap_or_else(|_| "C:\\Windows".to_owned());
            PathBuf::from(system_root)
                .join("System32")
                .join("nvidia-smi.exe")
        } else {
            PathBuf::from("/usr/bin/nvidia-smi")
        };
        if !executable.is_file() {
            return Err(format!(
                "trusted nvidia-smi path does not exist: {}",
                executable.display()
            ));
        }
        let physical_id = std::env::var("CUDA_VISIBLE_DEVICES")
            .ok()
            .and_then(|value| value.split(',').next().map(str::trim).map(str::to_owned))
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "0".to_owned());
        Ok(Self {
            executable,
            physical_id,
        })
    }

    fn query(&self, fields: &str) -> Result<String, String> {
        let output = Command::new(&self.executable)
            .arg(format!("--id={}", self.physical_id))
            .arg(format!("--query-gpu={fields}"))
            .arg("--format=csv,noheader,nounits")
            .output()
            .map_err(|error| format!("start {}: {error}", self.executable.display()))?;
        if !output.status.success() {
            return Err(format!(
                "{} query failed: {}",
                self.executable.display(),
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
    }

    fn used_bytes(&self) -> Result<u64, String> {
        self.query("memory.used")?
            .parse::<u64>()
            .map(|used_mib| used_mib * MIB)
            .map_err(|error| format!("parse nvidia-smi memory.used: {error}"))
    }
}

fn decimal_gb_to_bytes(value: f64) -> u64 {
    (value * 1.0e9).round() as u64
}

fn cuda_phase_metrics(device_bytes: u64) -> Value {
    // Candle's exact CUDA backend allocates directly through cudarc/CUDA and has no caching
    // allocator counter. On the required idle single-process GPU the `nvidia-smi memory.used` delta
    // is therefore the one truthful residency counter, and it is non-reclaimable: discrete CUDA
    // device allocations are physically non-pageable. So `activeBytes` carries the reading,
    // `reclaimableBytes` is a measured zero, and `allocatorBytes` is their sum by the schema-v5
    // identity. sc-18864 dropped `deviceBytes` and `wiredBytes`, which were further copies of this
    // same number under names claiming to be distinct quantities.
    json!({
        "activeBytes": device_bytes,
        "allocatorBytes": device_bytes,
        "reclaimableBytes": 0,
    })
}

fn nvcc_runtime() -> Result<String, String> {
    let executable = if cfg!(windows) {
        PathBuf::from(protocol::required_env("CUDA_PATH")?)
            .join("bin")
            .join("nvcc.exe")
    } else {
        PathBuf::from("/usr/local/cuda/bin/nvcc")
    };
    let output = Command::new(&executable)
        .arg("--version")
        .output()
        .map_err(|error| format!("start {}: {error}", executable.display()))?;
    if !output.status.success() {
        return Err(format!("{} --version failed", executable.display()));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .split("release ")
        .nth(1)
        .and_then(|tail| tail.split(',').next())
        .map(str::trim)
        .map(str::to_owned)
        .ok_or_else(|| format!("cannot parse CUDA runtime from {}", executable.display()))
}

fn probe() -> Result<Value, String> {
    let smi = NvidiaSmi::resolve()?;
    let fields = smi.query("index,name,compute_cap,driver_version,memory.total")?;
    let columns: Vec<_> = fields.split(',').map(str::trim).collect();
    if columns.len() != 5 {
        return Err(format!(
            "nvidia-smi returned {} fields instead of 5: {fields:?}",
            columns.len()
        ));
    }
    let total_mib: u64 = columns[4]
        .parse()
        .map_err(|error| format!("parse nvidia-smi memory.total: {error}"))?;
    Ok(json!({
        "hardware": {
            "probe": format!("{} selected through CUDA_VISIBLE_DEVICES", smi.executable.display()),
            "memoryBytes": total_mib * MIB,
            "deviceId": columns[0],
            "name": columns[1],
            "computeCapability": columns[2],
            "driverVersion": columns[3],
            "runtimeVersion": nvcc_runtime()?,
        }
    }))
}

fn sweep(request: &Value, parameters: &Map<String, Value>, result: &str) -> Result<Value, String> {
    let fingerprint = protocol::planned(request)?
        .get("calibrationFingerprint")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.calibrationFingerprint must be a string".to_owned())?;
    let candidates: &[(u64, u64, u64, u64)] = match fingerprint {
        "krea-turbo-cuda-phase-curves-v1" => &[(512, 128, 134_217_728, 1)],
        "krea-turbo-cuda-phase-curves-v2" => {
            &[(384, 64, 67_108_864, 1), (640, 128, 134_217_728, 2)]
        }
        other => return Err(format!("unknown Krea calibration fingerprint {other:?}")),
    };
    let current = |name: &str| parameters.get(name).and_then(Value::as_u64);
    let rows = candidates
        .iter()
        .map(|(edge, overlap, attention, window)| {
            let selected = current("decodeTileEdge") == Some(*edge)
                && current("decodeOverlap") == Some(*overlap)
                && current("attentionChunkSize") == Some(*attention)
                && current("transformerWindowSize") == Some(*window);
            json!({
                "parameters": {
                    "decodeTileEdge": edge,
                    "decodeOverlap": overlap,
                    "attentionChunkSize": attention,
                    "transformerWindowSize": window,
                },
                "result": if selected { result } else { "not_run" },
            })
        })
        .collect::<Vec<_>>();
    let values = |index: usize| {
        let mut values = candidates
            .iter()
            .map(|candidate| match index {
                0 => candidate.0,
                1 => candidate.1,
                2 => candidate.2,
                3 => candidate.3,
                _ => unreachable!("Krea sweep has exactly four axes"),
            })
            .collect::<Vec<_>>();
        values.sort_unstable();
        values.dedup();
        values
    };
    Ok(json!({
        "axes": [
            { "parameter": "decodeTileEdge", "testedValues": values(0) },
            { "parameter": "decodeOverlap", "testedValues": values(1) },
            { "parameter": "attentionChunkSize", "testedValues": values(2) },
            { "parameter": "transformerWindowSize", "testedValues": values(3) }
        ],
        "cases": rows,
        "rangeVerified": false,
    }))
}

fn artifact(repository: &str, revision: &str, tier: &str) -> Value {
    json!({
        "repository": repository,
        "resolvedRevision": revision,
        "variant": tier,
    })
}

fn loadability_fingerprint(repository: &str, revision: &str, tier: &str) -> String {
    format!("{repository}@{revision}:{tier}")
}

#[allow(clippy::too_many_arguments)]
fn execute_lifecycle_request(
    generator: &dyn runtime_cuda::gen_core::Generator,
    context: &MemoryRunContext,
    edge: u32,
    overlap: u32,
    attention: u32,
    window: u32,
    fault_phase: Option<MemoryPhase>,
    cancel_phase: Option<MemoryPhase>,
) -> Result<(), String> {
    let mut scope = generator
        .begin_memory_strategy_request(context)
        .map_err(|error| format!("begin lifecycle Krea scope: {error}"))?
        .ok_or_else(|| "lifecycle Krea selection did not create a provider scope".to_owned())?;
    scope
        .configure_decode(edge, overlap, context.geometry)
        .map_err(|error| format!("configure lifecycle decode tuple: {error}"))?;
    scope
        .configure_attention(attention)
        .map_err(|error| format!("configure lifecycle attention tuple: {error}"))?;
    scope
        .materialize_transformer_window(0, window)
        .map_err(|error| format!("configure lifecycle transformer tuple: {error}"))?;

    let mut generation = GenerationRequest {
        prompt: "a photorealistic red apple on a wooden table, studio lighting".to_owned(),
        width: context.geometry.width,
        height: context.geometry.height,
        count: 1,
        seed: Some(42),
        steps: Some(8),
        ..Default::default()
    };
    scope
        .configure_request(&mut generation)
        .map_err(|error| format!("apply lifecycle request strategy: {error}"))?;
    let memory = generation
        .memory
        .as_mut()
        .ok_or_else(|| "optimized lifecycle request did not receive GenerationMemory".to_owned())?;
    if let Some(phase) = fault_phase {
        memory.authorize_calibration_fault(phase);
    }
    scope
        .enter_phase(MemoryPhase::Conditioning)
        .map_err(|error| format!("enter lifecycle conditioning phase: {error}"))?;

    let cancel = generation.cancel.clone();
    let mut phase = MemoryPhase::Conditioning;
    let result = generator.generate(&generation, &mut |progress| match progress {
        Progress::Loading(runtime_cuda::gen_core::LoadPhase::TextEncoder)
            if cancel_phase == Some(MemoryPhase::Conditioning) =>
        {
            cancel.cancel();
        }
        Progress::Loading(runtime_cuda::gen_core::LoadPhase::Renderer)
            if phase == MemoryPhase::Conditioning =>
        {
            let _ = scope.leave_phase(MemoryPhase::Conditioning);
            let _ = scope.enter_phase(MemoryPhase::Denoise);
            phase = MemoryPhase::Denoise;
            if cancel_phase == Some(MemoryPhase::Denoise) {
                cancel.cancel();
            }
        }
        Progress::Loading(runtime_cuda::gen_core::LoadPhase::Renderer)
            if phase == MemoryPhase::Denoise =>
        {
            let _ = scope.leave_phase(MemoryPhase::Denoise);
            let _ = scope.enter_phase(MemoryPhase::Decode);
            phase = MemoryPhase::Decode;
        }
        Progress::Decoding if cancel_phase == Some(MemoryPhase::Decode) => {
            cancel.cancel();
        }
        _ => {}
    });

    match (fault_phase, cancel_phase, result) {
        (None, None, Ok(runtime_cuda::gen_core::GenerationOutput::Images(images)))
            if images.len() == 1 =>
        {
            scope
                .leave_phase(phase)
                .map_err(|error| format!("leave successful lifecycle phase: {error}"))?;
            scope
                .finish(MemoryRunOutcome::Complete)
                .map_err(|error| format!("finish successful lifecycle request: {error}"))
        }
        (Some(expected), None, Err(error))
            if error.to_string().contains("injected memory-strategy calibration error")
                && error.to_string().contains(&format!("{expected:?}")) =>
        {
            scope
                .finish(MemoryRunOutcome::Error {
                    message: error.to_string(),
                })
                .map_err(|finish| format!("finish injected-error lifecycle request: {finish}"))
        }
        (None, Some(_), Err(runtime_cuda::gen_core::Error::Canceled)) => scope
            .finish(MemoryRunOutcome::Canceled)
            .map_err(|error| format!("finish canceled lifecycle request: {error}")),
        (expected_fault, expected_cancel, actual) => Err(format!(
            "lifecycle outcome mismatch: fault={expected_fault:?}, cancel={expected_cancel:?}, actual={}",
            match actual {
                Ok(_) => "success".to_owned(),
                Err(error) => format!("error: {error}"),
            }
        )),
    }
}

fn execute_parity_request(
    generator: &dyn runtime_cuda::gen_core::Generator,
    baseline_context: &MemoryRunContext,
    strategy: MemoryStrategy,
    parameters: MemoryStrategyParameters,
) -> Result<runtime_cuda::gen_core::Image, String> {
    let mut context = baseline_context.clone();
    context.selection.strategy = strategy;
    context.selection.parameters = parameters;
    let mut scope = generator
        .begin_memory_strategy_request(&context)
        .map_err(|error| format!("begin parity Krea scope for {strategy:?}: {error}"))?
        .ok_or_else(|| format!("parity Krea strategy {strategy:?} did not create a scope"))?;
    if strategy.is_optimized() {
        let edge = parameters
            .decode_tile_edge
            .ok_or_else(|| format!("parity {strategy:?} is missing decode_tile_edge"))?;
        let overlap = parameters
            .decode_overlap
            .ok_or_else(|| format!("parity {strategy:?} is missing decode_overlap"))?;
        let attention = parameters
            .attention_chunk_size
            .ok_or_else(|| format!("parity {strategy:?} is missing attention_chunk_size"))?;
        let window = parameters
            .transformer_window_size
            .ok_or_else(|| format!("parity {strategy:?} is missing transformer_window_size"))?;
        scope
            .configure_decode(edge, overlap, context.geometry)
            .map_err(|error| format!("configure parity decode tuple: {error}"))?;
        scope
            .configure_attention(attention)
            .map_err(|error| format!("configure parity attention tuple: {error}"))?;
        scope
            .materialize_transformer_window(0, window)
            .map_err(|error| format!("configure parity transformer tuple: {error}"))?;
    }

    let mut generation = GenerationRequest {
        prompt: "a photorealistic red apple on a wooden table, studio lighting".to_owned(),
        width: context.geometry.width,
        height: context.geometry.height,
        count: 1,
        seed: Some(42),
        steps: Some(8),
        ..Default::default()
    };
    scope
        .configure_request(&mut generation)
        .map_err(|error| format!("apply parity request strategy: {error}"))?;
    scope
        .enter_phase(MemoryPhase::Conditioning)
        .map_err(|error| format!("enter parity conditioning phase: {error}"))?;
    let mut phase = MemoryPhase::Conditioning;
    let result = generator.generate(&generation, &mut |progress| match progress {
        Progress::Loading(runtime_cuda::gen_core::LoadPhase::Renderer)
            if phase == MemoryPhase::Conditioning =>
        {
            let _ = scope.leave_phase(MemoryPhase::Conditioning);
            let _ = scope.enter_phase(MemoryPhase::Denoise);
            phase = MemoryPhase::Denoise;
        }
        Progress::Loading(runtime_cuda::gen_core::LoadPhase::Renderer)
            if phase == MemoryPhase::Denoise =>
        {
            let _ = scope.leave_phase(MemoryPhase::Denoise);
            let _ = scope.enter_phase(MemoryPhase::Decode);
            phase = MemoryPhase::Decode;
        }
        _ => {}
    });
    let output = match result {
        Ok(output) => output,
        Err(error) => {
            let message = error.to_string();
            let _ = scope.finish(MemoryRunOutcome::Error {
                message: message.clone(),
            });
            return Err(format!(
                "parity Krea {strategy:?} generation failed: {message}"
            ));
        }
    };
    scope
        .leave_phase(phase)
        .map_err(|error| format!("leave parity terminal phase: {error}"))?;
    scope
        .finish(MemoryRunOutcome::Complete)
        .map_err(|error| format!("finish parity Krea request: {error}"))?;
    match output {
        runtime_cuda::gen_core::GenerationOutput::Images(mut images) if images.len() == 1 => {
            Ok(images.remove(0))
        }
        runtime_cuda::gen_core::GenerationOutput::Images(images) => Err(format!(
            "parity Krea {strategy:?} run returned {} images, expected 1",
            images.len()
        )),
        _ => Err(format!(
            "parity Krea {strategy:?} run returned non-image output"
        )),
    }
}

fn pixel_error(
    reference: &runtime_cuda::gen_core::Image,
    candidate: &runtime_cuda::gen_core::Image,
) -> Result<(u64, u64), String> {
    if (reference.width, reference.height, reference.pixels.len())
        != (candidate.width, candidate.height, candidate.pixels.len())
    {
        return Err(format!(
            "parity image shape mismatch: reference={}x{}x{}, candidate={}x{}x{}",
            reference.width,
            reference.height,
            reference.pixels.len(),
            candidate.width,
            candidate.height,
            candidate.pixels.len()
        ));
    }
    if reference.pixels.is_empty() {
        return Err("parity image is empty".to_owned());
    }
    let mut maximum = 0_u64;
    let mut total = 0_u64;
    for (&left, &right) in reference.pixels.iter().zip(&candidate.pixels) {
        let error = u64::from(left.abs_diff(right));
        maximum = maximum.max(error);
        total += error;
    }
    let mean_micro_units =
        total.saturating_mul(1_000_000) / u64::try_from(reference.pixels.len()).unwrap_or(u64::MAX);
    Ok((maximum, mean_micro_units))
}

fn preflight_fragment(
    request: &Value,
    strategy: &Value,
    load_shape: LoadShape,
    blocker: String,
    measurement_name: &'static str,
    repository: &str,
    revision: &str,
) -> Result<Value, String> {
    let mut fragment = protocol::plain_gated_fragment(
        request,
        KREA_PLAIN_EXECUTION_PATH,
        protocol::PlainGatedFragment {
            artifact: artifact(repository, revision, planned_tier(request)?),
            sweep: sweep(request, protocol::strategy_parameters(request)?, "failed")?,
            blocker: &blocker,
            quality: json!({ "result": "not_run" }),
            negative_mutation: Value::Null,
            loadability: json!({ "result": "not_run", "resolvedPathFingerprint": null }),
            diagnostics: protocol::diagnostics(
                "memory-candle-adapter",
                "gated_before_execution",
                [blocker.clone()],
                [(measurement_name, "count", 1)],
            ),
        },
    )?;
    fragment["strategy"] = strategy.clone();
    fragment["loadShape"] = json!(load_shape_key(load_shape));
    Ok(fragment)
}

/// Persisted spelling of `gen_core::LoadShape` for the schema-v4 receipt field.
///
/// Callers pass the shape the run actually executed under — in practice
/// `contract.calibration.load_shape` from the LOADED provider, never the plan's declared value and
/// never a literal. A receipt may only testify to its own run (sc-16482).
fn load_shape_key(load_shape: LoadShape) -> &'static str {
    match load_shape {
        LoadShape::EagerMaterialization => protocol::LOAD_SHAPE_EAGER,
        LoadShape::DeferredMaterialization => protocol::LOAD_SHAPE_DEFERRED,
    }
}

fn strategy_name(strategy: MemoryStrategy) -> &'static str {
    match strategy {
        MemoryStrategy::Resident => "resident",
        MemoryStrategy::StagedResidency => "staged_residency",
        MemoryStrategy::BoundedDecode => "bounded_decode",
        MemoryStrategy::BoundedAttention => "bounded_attention",
        MemoryStrategy::BoundedTransformerResidency => "bounded_transformer_residency",
    }
}

fn planned_memory_strategy(request: &Value) -> Result<MemoryStrategy, String> {
    match protocol::planned_rung(request)? {
        "resident" => Ok(MemoryStrategy::Resident),
        "staged_residency" => Ok(MemoryStrategy::StagedResidency),
        "bounded_decode" => Ok(MemoryStrategy::BoundedDecode),
        "bounded_attention" => Ok(MemoryStrategy::BoundedAttention),
        "bounded_transformer_residency" => Ok(MemoryStrategy::BoundedTransformerResidency),
        other => Err(format!("unsupported Candle fresh-reference rung {other:?}")),
    }
}

fn planned_provider(request: &Value) -> Result<&str, String> {
    protocol::planned(request)?
        .pointer("/target/provider")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.provider must be a string".to_owned())
}

fn plain_execution_path(request: &Value) -> Result<&'static str, String> {
    match planned_provider(request)? {
        "qwen_image" => Ok(QWEN_PLAIN_EXECUTION_PATH),
        "krea_2_turbo" => Ok(KREA_PLAIN_EXECUTION_PATH),
        provider => Err(format!(
            "Candle five-rung calibration does not implement provider {provider:?}"
        )),
    }
}

/// The calibration label this Candle target refuses a non-still geometry under (sc-18808).
///
/// BOTH Candle arms are image arms, and both carried the same latent defect the MLX image arms did:
/// they read only `width`/`height` through [`protocol::target_geometry`] and then wrote `frames: 1`
/// straight into `MemoryGeometry`. A plan row declaring `frames: 2` would therefore have rendered ONE
/// frame and emitted a well-formed record whose geometry envelope claimed a single frame it was never
/// asked for — the exact defect class this apparatus exists to make impossible.
///
/// No Candle plan row declares a non-unit frames axis today (all 154 rows are `frames: 1`), so this
/// is not a live exposure. It is added anyway because epic 18803 IS the video lane and
/// `ltx_2_3_distilled` is a Candle engine id: the shape becomes reachable, and a refusal is the only
/// thing that keeps the record honest when it does. Mirrors [`plain_execution_path`] so a provider
/// this adapter does not implement is rejected by the same sentence in both.
fn still_calibration_label(request: &Value) -> Result<&'static str, String> {
    match planned_provider(request)? {
        QWEN_ID => Ok(QWEN_STILL_CALIBRATION),
        KREA_ID => Ok(KREA_STILL_CALIBRATION),
        provider => Err(format!(
            "Candle five-rung calibration does not implement provider {provider:?}"
        )),
    }
}

/// The numeric tier this case plans to measure, read from the plan rather than assumed.
///
/// sc-17097: this used to be hardcoded `q4`, which silently capped the Candle lane at one tier — the
/// `krea_2_turbo` turbo fit ships `q4`, `q8` and `bf16` phase curves, so two thirds of it could not be
/// re-measured at all. The MLX adapter has always derived its tier from `/target/tier`; this mirrors
/// that, and [`planned_tier_variant`] keeps the on-disk artifact bound to the same token so a q8 plan
/// can never be satisfied by q4 weights.
fn planned_tier(request: &Value) -> Result<&str, String> {
    match protocol::planned(request)?
        .pointer("/target/tier")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.tier must be a string".to_owned())?
    {
        tier @ ("bf16" | "q4" | "q8") => Ok(tier),
        tier => Err(format!("unsupported Candle numeric tier {tier:?}")),
    }
}

/// The fixture must name the tier and geometry it measured, so a bf16 record can never be emitted
/// against a q4 capture that merely reused the fixture string.
///
/// Scoped to `krea_2_turbo` DELIBERATELY. Krea is the only provider here whose plan spans several
/// (tier, geometry) legs through one adapter path — six of them, which is exactly how a mislabelled
/// capture would arise. The Qwen legs declare a single tier and geometry each and their fixture names
/// (`qwen-image-candle-q4-seed15817-step2`) predate this convention: applying the geometry token
/// requirement to them would reject five plan rows that measure correctly today. Widen this when
/// those fixtures are renamed, not before.
fn validate_fixture_binds_tier_and_geometry(request: &Value) -> Result<(), String> {
    if planned_provider(request)? != KREA_ID {
        return Ok(());
    }
    let planned = protocol::planned(request)?;
    let fixture = planned
        .get("fixture")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.fixture must be a string".to_owned())?;
    let tier = planned_tier(request)?;
    let (width, height) = protocol::target_geometry(request)?;
    if width != height {
        return Err(format!(
            "Candle Krea calibration fixtures are square; planned geometry is {width}x{height}"
        ));
    }
    for token in [format!("-{tier}-"), format!("-{width}-")] {
        if !fixture.contains(&token) {
            return Err(format!(
                "planned.fixture {fixture:?} must contain {token:?} so the capture cannot be \
                 attributed to another tier or geometry"
            ));
        }
    }
    Ok(())
}

fn numeric_tier(tier: &str) -> Result<MemoryNumericTier, String> {
    // Matches the worker's `tier_to_quant`: bf16 is the dense base, q4/q8 are the packed tiers.
    let quant = match tier {
        "bf16" => None,
        "q4" => Some(Quant::Q4),
        "q8" => Some(Quant::Q8),
        other => return Err(format!("unsupported Candle numeric tier {other:?}")),
    };
    Ok(MemoryNumericTier {
        precision: Precision::Bf16,
        quant,
        component_precision_floors: &[],
    })
}

fn planned_selection(request: &Value) -> Result<MemorySelection, String> {
    let strategy = planned_memory_strategy(request)?;
    let transformer_window_size = protocol::optional_parameter(request, "transformerWindowSize")?;
    Ok(MemorySelection {
        strategy,
        parameters: MemoryStrategyParameters {
            decode_tile_edge: protocol::optional_parameter(request, "decodeTileEdge")?,
            decode_overlap: protocol::optional_parameter(request, "decodeOverlap")?,
            attention_chunk_size: protocol::optional_parameter(request, "attentionChunkSize")?,
            transformer_window_size,
            transformer_window_component: transformer_window_size
                .map(|_| TransformerComponent::Dit),
        },
        tier: numeric_tier(planned_tier(request)?)?,
    })
}

fn reference_phase(phase: MemoryPhase) -> protocol::ReferencePhase {
    match phase {
        MemoryPhase::Conditioning => protocol::ReferencePhase::Conditioning,
        MemoryPhase::Denoise => protocol::ReferencePhase::Denoise,
        MemoryPhase::Decode => protocol::ReferencePhase::Decode,
    }
}

fn memory_phase(phase: protocol::ReferencePhase) -> MemoryPhase {
    match phase {
        protocol::ReferencePhase::Conditioning => MemoryPhase::Conditioning,
        protocol::ReferencePhase::Denoise => MemoryPhase::Denoise,
        protocol::ReferencePhase::Decode => MemoryPhase::Decode,
    }
}

fn measured_strategy(
    request: &Value,
    selection: &MemorySelection,
    engaged: &[MemoryStrategy],
) -> Result<Value, String> {
    let measured = json!({
        "rung": strategy_name(selection.strategy),
        "engagedRungs": engaged.iter().copied().map(strategy_name).collect::<Vec<_>>(),
        "parameters": protocol::strategy_parameters(request)?,
    });
    let planned = protocol::planned(request)?
        .get("strategy")
        .ok_or_else(|| "planned.strategy must be present".to_owned())?;
    if planned != &measured {
        return Err(format!(
            "plan/provider strategy mismatch: plan={planned}, pinned provider measured={measured}"
        ));
    }
    Ok(measured)
}

/// Everything one five-rung capture needs after the artifact identity is validated and the real
/// generator is resident: `(provider id, plain execution path, repository, resolved revision,
/// generator, VRAM probe already holding the load sample)`.
type LoadedFiveRungGenerator = (
    &'static str,
    &'static str,
    String,
    String,
    Box<dyn runtime_cuda::gen_core::Generator>,
    VramProbe,
);

fn load_five_rung_generator(request: &Value) -> Result<LoadedFiveRungGenerator, String> {
    let (provider_id, execution_path, repository_env, revision_env, root_env, expected_repository) =
        match planned_provider(request)? {
            "qwen_image" => (
                QWEN_ID,
                QWEN_PLAIN_EXECUTION_PATH,
                "SCENEWORKS_QWEN_IMAGE_REPOSITORY",
                "SCENEWORKS_QWEN_IMAGE_REVISION",
                "SCENEWORKS_QWEN_IMAGE_ROOT",
                protocol::QWEN_REPOSITORY,
            ),
            "krea_2_turbo" => (
                KREA_ID,
                KREA_PLAIN_EXECUTION_PATH,
                "SCENEWORKS_KREA_REPOSITORY",
                "SCENEWORKS_KREA_REVISION",
                "SCENEWORKS_KREA_ROOT",
                protocol::KREA_REPOSITORY,
            ),
            provider => {
                return Err(format!(
                    "Candle five-rung calibration does not implement provider {provider:?}"
                ))
            }
        };
    let tier = planned_tier(request)?;
    validate_fixture_binds_tier_and_geometry(request)?;
    let repository = protocol::required_env(repository_env)?;
    let revision = protocol::required_env(revision_env)?;
    protocol::validate_artifact_identity(&repository, &revision, expected_repository)?;
    let root = std::fs::canonicalize(PathBuf::from(protocol::required_env(root_env)?))
        .map_err(|error| format!("canonicalize {root_env}: {error}"))?;
    // The root must end in the PLANNED tier's directory, so a stale `…/q4` export cannot satisfy a
    // q8 or bf16 plan and quietly re-label another tier's peaks.
    protocol::validate_huggingface_snapshot_root(
        &root,
        &repository,
        &revision,
        tier,
        expected_repository,
    )?;
    let spec = LoadSpec::new(WeightsSource::Dir(root))
        .with_offload_policy(OffloadPolicy::Sequential)
        .with_load_shape(LoadShape::DeferredMaterialization);
    let spec = match (provider_id, numeric_tier(tier)?.quant) {
        // Krea's loader takes the packed tier's quant explicitly; bf16 is the dense base and must
        // carry no quant at all (`Quant::None` — the same shape the worker's `tier_to_quant` uses).
        (KREA_ID, Some(quant)) => spec.with_quant(quant),
        (KREA_ID, None) => spec,
        // Qwen packed Diffusers snapshots declare their device-format quantization in
        // transformer/config.json. Passing LoadSpec.quant would request a second, unsupported
        // runtime quantization pass instead of loading the packed artifact as authored.
        _ => spec,
    };
    let catalog =
        runtime_cuda::catalog().map_err(|error| format!("build CUDA catalog: {error}"))?;
    let mut vram = VramProbe::start_rendered().assert_idle(1.0);
    let load_sample = vram.phase();
    let generator = catalog
        .media()
        .load(provider_id, &spec)
        .map_err(|error| format!("load real {provider_id} {tier} generator: {error}"))?;
    vram.end_load(load_sample);
    Ok((
        provider_id,
        execution_path,
        repository,
        revision,
        generator,
        vram,
    ))
}

fn run_five_rung_reference_loaded(
    request: &Value,
    provider_id: &str,
    execution_path: &str,
    generator: &dyn runtime_cuda::gen_core::Generator,
    vram: &mut VramProbe,
    repository: &str,
    revision: &str,
) -> Result<Value, String> {
    protocol::validate_plain_overlay_target(request, execution_path)?;
    protocol::validate_still_geometry(request, still_calibration_label(request)?)?;
    let contract = generator
        .memory_strategy_contract()
        .ok_or_else(|| format!("loaded {provider_id} has no memory-strategy contract"))?;
    let selection = planned_selection(request)?;
    contract.validate_selection(&selection).map_err(|error| {
        format!("pinned {provider_id} provider rejected planned selection: {error}")
    })?;
    let strategy = measured_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    let calibration = contract
        .calibration
        .as_ref()
        .ok_or_else(|| "pinned Krea provider has no calibration identity".to_owned())?;
    let planned_fingerprint = protocol::planned(request)?
        .get("calibrationFingerprint")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.calibrationFingerprint must be a string".to_owned())?;
    if planned_fingerprint != calibration.fingerprint {
        return Err(format!(
            "plan/provider calibration mismatch: plan={planned_fingerprint}, pinned provider={}",
            calibration.fingerprint
        ));
    }
    let planned_load_shape = protocol::planned(request)?
        .get("loadShape")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.loadShape must be a string".to_owned())?;
    let actual_load_shape = match calibration.load_shape {
        LoadShape::EagerMaterialization => "eager_materialization",
        LoadShape::DeferredMaterialization => "deferred_materialization",
    };
    if planned_load_shape != actual_load_shape {
        return Err(format!(
            "plan/provider load-shape mismatch: plan={planned_load_shape}, pinned provider={actual_load_shape}"
        ));
    }
    let (width, height) = protocol::target_geometry(request)?;
    let hardware_bytes = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.memoryBytes must be an integer".to_owned())?;
    let context = MemoryRunContext {
        selection,
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
        calibration_abi: calibration.abi,
        calibration_fingerprint: calibration.fingerprint.clone(),
        load_shape: calibration.load_shape,
        mode: MemoryMode::TextToImage,
        has_reference: false,
        use_pid: false,
        has_phases: false,
        geometry: MemoryGeometry {
            width,
            height,
            batch: 1,
            frames: 1,
            reference_count: 0,
        },
        overlay: None,
        budget: MemoryBudget {
            total_bytes: hardware_bytes,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes: 1,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: format!("sc-16402@{}", protocol::INFERENCE_PIN),
    };
    let mut scope = generator
        .begin_memory_strategy_request(&context)
        .map_err(|error| format!("begin {provider_id} fresh-reference scope: {error}"))?
        .ok_or_else(|| {
            format!("{provider_id} fresh-reference selection did not create a provider scope")
        })?;
    let parameters = context.selection.parameters;
    match (parameters.decode_tile_edge, parameters.decode_overlap) {
        (Some(edge), Some(overlap)) => scope
            .configure_decode(edge, overlap, context.geometry)
            .map_err(|error| format!("configure {provider_id} fresh-reference decode: {error}"))?,
        (None, None) => {}
        _ => {
            return Err(format!(
                "{provider_id} decode edge and overlap must be selected together"
            ))
        }
    }
    if let Some(attention) = parameters.attention_chunk_size {
        scope.configure_attention(attention).map_err(|error| {
            format!("configure {provider_id} fresh-reference attention: {error}")
        })?;
    }
    if let Some(window) = parameters.transformer_window_size {
        scope
            .materialize_transformer_window(0, window)
            .map_err(|error| {
                format!("configure {provider_id} fresh-reference transformer: {error}")
            })?;
    }
    let mut generation = GenerationRequest {
        prompt: "a photorealistic red apple on a wooden table, studio lighting".to_owned(),
        width,
        height,
        count: 1,
        seed: Some(16402),
        // Two steps are intentional: resident Krea has no provider loading boundary between text
        // encode and denoise. The first Step callback closes a conservative conditioning envelope;
        // the second step then gives denoise its own measured interval before Decoding.
        steps: Some(2),
        ..Default::default()
    };
    scope
        .configure_request(&mut generation)
        .map_err(|error| format!("apply {provider_id} fresh-reference strategy: {error}"))?;
    scope
        .enter_phase(MemoryPhase::Conditioning)
        .map_err(|error| format!("enter {provider_id} fresh-reference conditioning: {error}"))?;
    let generation_sample = vram.phase();
    let mut phase_sample = Some(vram.phase());
    let mut phase = MemoryPhase::Conditioning;
    let mut conditioning_peak_gb = None;
    let mut denoise_peak_gb = None;
    let mut decode_peak_gb = None;
    let mut phase_error = None;
    let result = generator.generate(&generation, &mut |progress| {
        if phase_error.is_some() {
            return;
        }
        let boundary = match progress {
            Progress::Loading(runtime_cuda::gen_core::LoadPhase::Renderer) => {
                protocol::ReferenceBoundary::RendererLoad
            }
            Progress::Step { current: 1, .. } => protocol::ReferenceBoundary::FirstDenoiseStep,
            Progress::Decoding => protocol::ReferenceBoundary::Decoding,
            _ => return,
        };
        let Some(next) = protocol::next_reference_phase(reference_phase(phase), boundary) else {
            return;
        };
        let peak = phase_sample.take().map(|sample| vram.end_observed(sample));
        match phase {
            MemoryPhase::Conditioning => conditioning_peak_gb = peak,
            MemoryPhase::Denoise => denoise_peak_gb = peak,
            MemoryPhase::Decode => decode_peak_gb = peak,
        }
        if let Err(error) = scope.leave_phase(phase) {
            phase_error = Some(format!("leave {provider_id} {phase:?}: {error}"));
            return;
        }
        let next = memory_phase(next);
        if let Err(error) = scope.enter_phase(next) {
            phase_error = Some(format!("enter {provider_id} {next:?}: {error}"));
            return;
        }
        phase = next;
        phase_sample = Some(vram.phase());
    });
    if let Some(sample) = phase_sample.take() {
        let terminal_peak_gb = vram.end_observed(sample);
        match phase {
            MemoryPhase::Conditioning => conditioning_peak_gb = Some(terminal_peak_gb),
            MemoryPhase::Denoise => denoise_peak_gb = Some(terminal_peak_gb),
            MemoryPhase::Decode => decode_peak_gb = Some(terminal_peak_gb),
        }
    }
    vram.end_gen(generation_sample);
    if let Some(message) = phase_error {
        let _ = scope.finish(MemoryRunOutcome::Error {
            message: message.clone(),
        });
        return Err(message);
    }
    match result {
        Ok(runtime_cuda::gen_core::GenerationOutput::Images(images)) if images.len() == 1 => {}
        Ok(runtime_cuda::gen_core::GenerationOutput::Images(images)) => {
            return Err(format!(
                "{provider_id} fresh reference returned {} images",
                images.len()
            ));
        }
        Ok(_) => {
            return Err(format!(
                "{provider_id} fresh reference returned non-image output"
            ))
        }
        Err(error) => {
            let message = error.to_string();
            let _ = scope.finish(MemoryRunOutcome::Error {
                message: message.clone(),
            });
            return Err(format!(
                "{provider_id} fresh-reference generation failed: {message}"
            ));
        }
    }
    scope
        .leave_phase(phase)
        .map_err(|error| format!("leave {provider_id} fresh-reference terminal phase: {error}"))?;
    scope
        .finish(MemoryRunOutcome::Complete)
        .map_err(|error| format!("finish {provider_id} fresh-reference scope: {error}"))?;
    let conditioning_bytes = decimal_gb_to_bytes(conditioning_peak_gb.ok_or_else(|| {
        format!("{provider_id} fresh reference did not expose conditioning boundary")
    })?);
    let denoise_bytes =
        decimal_gb_to_bytes(denoise_peak_gb.ok_or_else(|| {
            format!("{provider_id} fresh reference did not expose denoise boundary")
        })?);
    let decode_bytes = decimal_gb_to_bytes(
        decode_peak_gb
            .ok_or_else(|| format!("{provider_id} fresh reference did not complete decode"))?,
    );
    let overall_bytes = conditioning_bytes.max(denoise_bytes).max(decode_bytes);
    let blocker = if provider_id == QWEN_ID {
        concat!(
            "SC-15817 five-rung conformance measures exact per-rung memory, strategy identity, ",
            "and loadability; it intentionally remains gated because this run does not repeat ",
            "each sibling story's promotion-quality, negative-mutation, and lifecycle suite"
        )
    } else {
        concat!(
            "five-rung oracle capture measures exact per-rung memory and strategy identity for ",
            "SC-16059; it intentionally remains gated because this run does not repeat the full ",
            "promotion-quality, negative-mutation, and lifecycle scenario suite"
        )
    };
    let mut fragment = protocol::plain_gated_fragment(
        request,
        execution_path,
        protocol::PlainGatedFragment {
            artifact: artifact(repository, revision, planned_tier(request)?),
            sweep: protocol::reference_sweep(request, "passed")?,
            blocker,
            quality: json!({ "result": "not_run" }),
            negative_mutation: Value::Null,
            loadability: json!({
                "result": "passed",
                "resolvedPathFingerprint": loadability_fingerprint(
                    repository,
                    revision,
                    planned_tier(request)?,
                ),
            }),
            diagnostics: protocol::diagnostics(
                &format!("memory-candle-adapter:{provider_id}-five-rung-reference"),
                "executed",
                [blocker.to_owned()],
                [
                    ("conditioningDevicePeakDelta", "bytes", conditioning_bytes),
                    ("denoiseDevicePeakDelta", "bytes", denoise_bytes),
                    ("decodeDevicePeakDelta", "bytes", decode_bytes),
                    ("overallDevicePeakDelta", "bytes", overall_bytes),
                ],
            ),
        },
    )?;
    fragment["strategy"] = strategy;
    fragment["loadShape"] = json!(load_shape_key(calibration.load_shape));
    fragment["observedMemory"] = json!({
        "conditioning": cuda_phase_metrics(conditioning_bytes),
        "denoise": cuda_phase_metrics(denoise_bytes),
        "decode": cuda_phase_metrics(decode_bytes),
        "overall": cuda_phase_metrics(overall_bytes),
    });
    Ok(fragment)
}

fn run_five_rung_reference(request: &Value) -> Result<Value, String> {
    let execution_path = plain_execution_path(request)?;
    protocol::validate_plain_overlay_target(request, execution_path)?;
    // Before `load_five_rung_generator`, for the same reason the overlay check is duplicated here:
    // a geometry this arm cannot honour must be refused before it costs a real weight load.
    protocol::validate_still_geometry(request, still_calibration_label(request)?)?;
    let (provider_id, execution_path, repository, revision, generator, mut vram) =
        load_five_rung_generator(request)?;
    run_five_rung_reference_loaded(
        request,
        provider_id,
        execution_path,
        generator.as_ref(),
        &mut vram,
        &repository,
        &revision,
    )
}

fn update_warmed_retention_baseline(
    settled_after_resident: &mut Option<u64>,
    after: u64,
) -> Result<(), String> {
    if let Some(baseline) = *settled_after_resident {
        if after > baseline.saturating_add(64 * MIB) {
            return Err(format!(
                "reused Krea rung retained {} bytes above the warmed resident baseline; refusing contaminated batching",
                after - baseline
            ));
        }
    } else {
        *settled_after_resident = Some(after);
    }
    Ok(())
}

fn run_five_rung_batch(request: &Value) -> Result<Value, String> {
    let planned = request
        .get("planned")
        .and_then(Value::as_array)
        .ok_or_else(|| "run_batch request.planned must be an array".to_owned())?;
    let expected_rungs = [
        "resident",
        "staged_residency",
        "bounded_decode",
        "bounded_attention",
        "bounded_transformer_residency",
    ];
    let actual_rungs = planned
        .iter()
        .map(|item| {
            item.pointer("/strategy/rung")
                .and_then(Value::as_str)
                .ok_or_else(|| "batched planned strategy.rung must be a string".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if actual_rungs != expected_rungs {
        return Err(format!(
            "run_batch requires one canonical five-rung target, got {actual_rungs:?}"
        ));
    }
    let first_target = planned[0]
        .get("target")
        .ok_or_else(|| "batched planned target is missing".to_owned())?;
    if planned
        .iter()
        .any(|item| item.get("target") != Some(first_target))
    {
        return Err("run_batch cannot mix calibration targets in one model load".to_owned());
    }
    for item in planned {
        let mut per_rung_request = request.clone();
        per_rung_request["action"] = json!("run");
        per_rung_request["planned"] = item.clone();
        let execution_path = plain_execution_path(&per_rung_request)?;
        protocol::validate_plain_overlay_target(&per_rung_request, execution_path)?;
        protocol::validate_still_geometry(
            &per_rung_request,
            still_calibration_label(&per_rung_request)?,
        )?;
    }

    let mut first_request = request.clone();
    first_request["action"] = json!("run");
    first_request["planned"] = planned[0].clone();
    let (provider_id, execution_path, repository, revision, generator, mut vram) =
        load_five_rung_generator(&first_request)?;
    let smi = NvidiaSmi::resolve()?;
    // Krea uses DeferredMaterialization, so loading the generator does not establish its
    // steady-state device residency. The canonical batch starts with `resident`; use the
    // memory retained after that first rung as the contamination baseline, then require every
    // later rung to release its transient allocations back to that warmed state.
    let mut settled_after_resident = None;
    let mut fragments = Vec::with_capacity(planned.len());
    for item in planned {
        let mut per_rung_request = request.clone();
        per_rung_request["action"] = json!("run");
        per_rung_request["planned"] = item.clone();
        fragments.push(run_five_rung_reference_loaded(
            &per_rung_request,
            provider_id,
            execution_path,
            generator.as_ref(),
            &mut vram,
            &repository,
            &revision,
        )?);
        let after = smi.used_bytes()?;
        update_warmed_retention_baseline(&mut settled_after_resident, after)?;
    }
    Ok(json!({ "modelLoads": 1, "fragments": fragments }))
}

/// The fixture prefix that marks a plan row as a five-rung reference capture.
const FIVE_RUNG_FIXTURE_PREFIX: &str = "fresh-five-rung-";

/// Which of [`run`]'s two branches a plan row takes: the five-rung reference path, or the inline
/// Krea arm.
///
/// Named rather than inlined so the decision is testable on its own (sc-18808 re-review). It is what
/// determines which arm [`run`]'s geometry guard is standing in front of, and every case in the
/// original regression table happened to answer `true` — so the inline arm, and with it `run`'s own
/// guard, went unexercised while the redundant copy at the head of [`run_five_rung_reference`]
/// produced the byte-identical message. Five shipped Candle plan rows answer `false`
/// (`krea-q4-1024-seed42` and its q8/bf16/768/v2 siblings).
fn routes_to_five_rung_reference(request: &Value) -> Result<bool, String> {
    let is_five_rung_fixture = protocol::planned(request)?
        .get("fixture")
        .and_then(Value::as_str)
        .is_some_and(|fixture| fixture.starts_with(FIVE_RUNG_FIXTURE_PREFIX));
    Ok(is_five_rung_fixture || planned_provider(request)? == QWEN_ID)
}

fn run(request: &Value) -> Result<Value, String> {
    if protocol::planned(request)?
        .get("backend")
        .and_then(Value::as_str)
        != Some("candle")
    {
        return Err(
            "Candle adapter received a non-Candle planned case; run the harness with --backend candle"
                .to_owned(),
        );
    }
    let provider = planned_provider(request)?;
    // sc-19057: the first candle VIDEO arm, dispatched BEFORE the still-geometry guard below. It is
    // the one Candle arm entitled to a multi-frame geometry, and it validates Wan's own declared
    // envelope through the shared video guard instead. Every other provider keeps the `frames == 1`
    // refusal untouched.
    if provider == WAN_PROVIDER {
        return run_wan(request);
    }
    let execution_path = plain_execution_path(request)?;
    protocol::validate_plain_overlay_target(request, execution_path)?;
    // Both remaining dispatch targets are image arms; refuse a non-still geometry here, before
    // either of them resolves an environment variable or touches a weight snapshot (sc-18808).
    protocol::validate_still_geometry(request, still_calibration_label(request)?)?;
    if routes_to_five_rung_reference(request)? {
        return run_five_rung_reference(request);
    }
    if provider != KREA_ID {
        return Err(format!(
            "unsupported Candle calibration provider {provider:?}"
        ));
    }
    let parameters = protocol::strategy_parameters(request)?;
    let tier = planned_tier(request)?;
    validate_fixture_binds_tier_and_geometry(request)?;
    let repository = protocol::required_env("SCENEWORKS_KREA_REPOSITORY")?;
    let revision = protocol::required_env("SCENEWORKS_KREA_REVISION")?;
    protocol::validate_artifact_identity(&repository, &revision, protocol::KREA_REPOSITORY)?;
    let root = std::env::var("SCENEWORKS_KREA_ROOT")
        .map(PathBuf::from)
        .unwrap_or_default();
    let root = if root.is_dir() {
        let canonical = std::fs::canonicalize(root)
            .map_err(|error| format!("canonicalize SCENEWORKS_KREA_ROOT: {error}"))?;
        protocol::validate_huggingface_snapshot_root(
            &canonical,
            &repository,
            &revision,
            tier,
            protocol::KREA_REPOSITORY,
        )?;
        canonical
    } else {
        root
    };
    let spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
        .with_offload_policy(OffloadPolicy::Sequential);
    let spec = match numeric_tier(tier)?.quant {
        Some(quant) => spec.with_quant(quant),
        None => spec,
    };
    let catalog =
        runtime_cuda::catalog().map_err(|error| format!("build CUDA catalog: {error}"))?;
    let contract = catalog
        .media()
        .memory_strategy_contract(KREA_ID, &spec)
        .map_err(|error| format!("read {KREA_ID} memory-strategy contract: {error}"))?
        .ok_or_else(|| {
            format!(
                "{KREA_ID} has no memory-strategy contract at {}",
                protocol::INFERENCE_PIN
            )
        })?;
    let edge = protocol::parameter(request, "decodeTileEdge")?;
    let overlap = protocol::parameter(request, "decodeOverlap")?;
    let attention = protocol::parameter(request, "attentionChunkSize")?;
    let window = protocol::parameter(request, "transformerWindowSize")?;
    let selected = MemoryStrategyParameters {
        decode_tile_edge: Some(edge),
        decode_overlap: Some(overlap),
        attention_chunk_size: Some(attention),
        transformer_window_size: Some(window),
        transformer_window_component: Some(TransformerComponent::Dit),
    };
    let selection = MemorySelection {
        strategy: MemoryStrategy::BoundedTransformerResidency,
        parameters: selected,
        tier: numeric_tier(tier)?,
    };
    let strategy = measured_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    let planned_fingerprint = protocol::planned(request)?
        .get("calibrationFingerprint")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.calibrationFingerprint must be a string".to_owned())?;
    let actual_calibration = contract.calibration.as_ref().ok_or_else(|| {
        format!(
            "{KREA_ID} has no calibration identity at {}",
            protocol::INFERENCE_PIN
        )
    })?;
    let actual_fingerprint = actual_calibration.fingerprint.as_str();
    if planned_fingerprint != actual_fingerprint {
        return preflight_fragment(
            request,
            &strategy,
            actual_calibration.load_shape,
            format!(
                "plan/provider calibration mismatch: plan={planned_fingerprint}, pinned provider={actual_fingerprint} at {}",
                protocol::INFERENCE_PIN
            ),
            "contractFingerprintMismatch",
            &repository,
            &revision,
        );
    }

    if let Err(reason) = contract.validate_selection(&selection) {
        return preflight_fragment(
            request,
            &strategy,
            actual_calibration.load_shape,
            format!("pinned provider rejected planned parameters before load: {reason}"),
            "contractParameterRejection",
            &repository,
            &revision,
        );
    }
    if !root.is_dir() {
        return preflight_fragment(
            request,
            &strategy,
            actual_calibration.load_shape,
            format!(
                "supported provider tuple requires real weights; set SCENEWORKS_KREA_ROOT to                  the validated {tier} snapshot"
            ),
            "missingWeights",
            &repository,
            &revision,
        );
    }

    let hardware = request
        .get("hardware")
        .and_then(Value::as_object)
        .ok_or_else(|| "run request.hardware must be an object".to_owned())?;
    let total_bytes = hardware
        .get("memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.memoryBytes must be an integer".to_owned())?;
    let (width, height) = protocol::target_geometry(request)?;
    let context = MemoryRunContext {
        selection,
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
        calibration_abi: actual_calibration.abi,
        calibration_fingerprint: actual_calibration.fingerprint.clone(),
        load_shape: actual_calibration.load_shape,
        mode: MemoryMode::TextToImage,
        has_reference: false,
        use_pid: false,
        has_phases: false,
        geometry: MemoryGeometry {
            width,
            height,
            batch: 1,
            frames: 1,
            reference_count: 0,
        },
        overlay: None,
        budget: MemoryBudget {
            total_bytes,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 2 * GIB,
        },
        predicted_peak_bytes: total_bytes.saturating_sub(2 * GIB),
        cache_state: MemoryCacheState::Cold,
        evidence_revision: format!("sc-15508-adapter@{}", protocol::INFERENCE_PIN),
    };

    let mut vram = VramProbe::start_rendered().assert_idle(1.0);
    let load_sample = vram.phase();
    let generator = catalog
        .media()
        .load(KREA_ID, &spec)
        .map_err(|error| format!("load real {KREA_ID} {tier} generator: {error}"))?;
    vram.end_load(load_sample);
    let mut scope = generator
        .begin_memory_strategy_request(&context)
        .map_err(|error| format!("begin real Krea memory-strategy scope: {error}"))?
        .ok_or_else(|| "optimized Krea selection did not create a provider scope".to_owned())?;
    scope
        .configure_decode(edge, overlap, context.geometry)
        .map_err(|error| format!("configure Krea decode tuple: {error}"))?;
    scope
        .configure_attention(attention)
        .map_err(|error| format!("configure Krea attention tuple: {error}"))?;
    scope
        .materialize_transformer_window(0, window)
        .map_err(|error| format!("configure Krea transformer tuple: {error}"))?;

    let mut generation = GenerationRequest {
        prompt: "a photorealistic red apple on a wooden table, studio lighting".to_owned(),
        width,
        height,
        count: 1,
        seed: Some(42),
        steps: Some(8),
        ..Default::default()
    };
    scope
        .configure_request(&mut generation)
        .map_err(|error| format!("apply Krea request-scoped strategy: {error}"))?;
    scope
        .enter_phase(MemoryPhase::Conditioning)
        .map_err(|error| format!("enter Krea conditioning phase: {error}"))?;

    let generation_sample = vram.phase();
    let mut phase_sample = Some(vram.phase());
    let mut phase = MemoryPhase::Conditioning;
    let mut conditioning_peak_gb = None;
    let mut denoise_peak_gb = None;
    let mut decode_peak_gb = None;
    let result = generator.generate(&generation, &mut |progress| match progress {
        Progress::Loading(runtime_cuda::gen_core::LoadPhase::Renderer)
            if phase == MemoryPhase::Conditioning =>
        {
            conditioning_peak_gb = phase_sample.take().map(|sample| vram.end_observed(sample));
            let _ = scope.leave_phase(MemoryPhase::Conditioning);
            let _ = scope.enter_phase(MemoryPhase::Denoise);
            phase = MemoryPhase::Denoise;
            phase_sample = Some(vram.phase());
        }
        Progress::Loading(runtime_cuda::gen_core::LoadPhase::Renderer)
            if phase == MemoryPhase::Denoise =>
        {
            denoise_peak_gb = phase_sample.take().map(|sample| vram.end_observed(sample));
            let _ = scope.leave_phase(MemoryPhase::Denoise);
            let _ = scope.enter_phase(MemoryPhase::Decode);
            phase = MemoryPhase::Decode;
            phase_sample = Some(vram.phase());
        }
        _ => {}
    });
    if let Some(sample) = phase_sample.take() {
        let terminal_peak_gb = vram.end_observed(sample);
        match phase {
            MemoryPhase::Conditioning => conditioning_peak_gb = Some(terminal_peak_gb),
            MemoryPhase::Denoise => denoise_peak_gb = Some(terminal_peak_gb),
            MemoryPhase::Decode => decode_peak_gb = Some(terminal_peak_gb),
        }
    }
    vram.end_gen(generation_sample);
    let report = vram.report();
    let output = match result {
        Ok(output) => output,
        Err(error) => {
            let message = error.to_string();
            let _ = scope.finish(MemoryRunOutcome::Error {
                message: message.clone(),
            });
            return Err(format!("real Krea {tier} generation failed: {message}"));
        }
    };
    scope
        .leave_phase(phase)
        .map_err(|error| format!("leave terminal Krea phase: {error}"))?;
    scope
        .finish(MemoryRunOutcome::Complete)
        .map_err(|error| format!("finish real Krea memory-strategy scope: {error}"))?;
    let image_count = match output {
        runtime_cuda::gen_core::GenerationOutput::Images(images) => images.len(),
        _ => 0,
    };
    if image_count != 1 {
        return Err(format!(
            "real Krea run returned {image_count} images, expected 1"
        ));
    }

    let conditioning_bytes = decimal_gb_to_bytes(conditioning_peak_gb.ok_or_else(|| {
        "Krea run did not expose a conditioning-to-denoise phase boundary".to_owned()
    })?);
    let denoise_bytes =
        decimal_gb_to_bytes(denoise_peak_gb.ok_or_else(|| {
            "Krea run did not expose a denoise-to-decode phase boundary".to_owned()
        })?);
    let decode_bytes = decimal_gb_to_bytes(
        decode_peak_gb.ok_or_else(|| "Krea run did not complete decode sampling".to_owned())?,
    );
    let overall_bytes = decimal_gb_to_bytes(report.peak_gb);
    let baseline = decimal_gb_to_bytes(report.baseline_gb);
    let lifecycle_phases = [
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    let smi = NvidiaSmi::resolve()?;
    let cleanup_tolerance_bytes = 64 * MIB;
    let mut maximum_cleanup_growth_bytes = 0_u64;
    for lifecycle_phase in lifecycle_phases {
        let before_fault_bytes = smi.used_bytes()?;
        execute_lifecycle_request(
            generator.as_ref(),
            &context,
            edge,
            overlap,
            attention,
            window,
            None,
            Some(lifecycle_phase),
        )?;
        let after_fault_bytes = smi.used_bytes()?;
        let cleanup_growth_bytes = after_fault_bytes.saturating_sub(before_fault_bytes);
        maximum_cleanup_growth_bytes = maximum_cleanup_growth_bytes.max(cleanup_growth_bytes);
        if cleanup_growth_bytes > cleanup_tolerance_bytes {
            return Err(format!(
                "{lifecycle_phase:?} cancellation retained {cleanup_growth_bytes} device bytes above its pre-request baseline"
            ));
        }
        execute_lifecycle_request(
            generator.as_ref(),
            &context,
            edge,
            overlap,
            attention,
            window,
            None,
            None,
        )?;
    }
    for lifecycle_phase in lifecycle_phases {
        let before_fault_bytes = smi.used_bytes()?;
        execute_lifecycle_request(
            generator.as_ref(),
            &context,
            edge,
            overlap,
            attention,
            window,
            Some(lifecycle_phase),
            None,
        )?;
        let after_fault_bytes = smi.used_bytes()?;
        let cleanup_growth_bytes = after_fault_bytes.saturating_sub(before_fault_bytes);
        maximum_cleanup_growth_bytes = maximum_cleanup_growth_bytes.max(cleanup_growth_bytes);
        if cleanup_growth_bytes > cleanup_tolerance_bytes {
            return Err(format!(
                "{lifecycle_phase:?} injected error retained {cleanup_growth_bytes} device bytes above its pre-request baseline"
            ));
        }
        execute_lifecycle_request(
            generator.as_ref(),
            &context,
            edge,
            overlap,
            attention,
            window,
            None,
            None,
        )?;
    }
    let resident_parameters = MemoryStrategyParameters::default();
    let resident_a = execute_parity_request(
        generator.as_ref(),
        &context,
        MemoryStrategy::Resident,
        resident_parameters,
    )?;
    let bounded_b = execute_parity_request(
        generator.as_ref(),
        &context,
        MemoryStrategy::BoundedTransformerResidency,
        selected,
    )?;
    let resident_a_repeat = execute_parity_request(
        generator.as_ref(),
        &context,
        MemoryStrategy::Resident,
        resident_parameters,
    )?;
    let (resident_repeat_max_error, resident_repeat_mean_error) =
        pixel_error(&resident_a, &resident_a_repeat)?;
    if resident_repeat_max_error != 0 {
        return Err(format!(
            "resident A-B-A repeat was not deterministic: maximum pixel error {resident_repeat_max_error}"
        ));
    }
    let (bounded_max_error, bounded_mean_error) = pixel_error(&resident_a, &bounded_b)?;
    let blocker = concat!(
        "real Krea phase telemetry executed, but complete evidence still requires predicted phase ",
        "curves, bounded-output tolerance approval, exact-fit/stale/unknown worker selection, and ",
        "a measured negative mutation"
    );
    let mut fragment = protocol::plain_gated_fragment(
        request,
        KREA_PLAIN_EXECUTION_PATH,
        protocol::PlainGatedFragment {
            artifact: artifact(&repository, &revision, tier),
            sweep: sweep(request, parameters, "passed")?,
            blocker,
            quality: json!({ "result": "not_run" }),
            negative_mutation: Value::Null,
            loadability: json!({
                "result": "passed",
                "resolvedPathFingerprint": loadability_fingerprint(&repository, &revision, tier),
            }),
            diagnostics: protocol::diagnostics(
                "memory-candle-adapter",
                "executed",
                [blocker.to_owned()],
                [
                    ("preLoadDeviceUsed", "bytes", baseline),
                    (
                        "loadDevicePeakDelta",
                        "bytes",
                        decimal_gb_to_bytes(report.load_peak_gb),
                    ),
                    ("conditioningDevicePeakDelta", "bytes", conditioning_bytes),
                    ("denoiseDevicePeakDelta", "bytes", denoise_bytes),
                    ("decodeDevicePeakDelta", "bytes", decode_bytes),
                    ("overallDevicePeakDelta", "bytes", overall_bytes),
                    ("allocatorCounterAliasesDeviceDelta", "boolean", 1),
                    ("wiredCounterAliasesDiscreteDeviceDelta", "boolean", 1),
                    ("cudaCachingAllocatorPresent", "boolean", 0),
                    ("phaseCancelInjections", "count", 3),
                    ("phaseErrorInjections", "count", 3),
                    ("postFaultWarmFollowUps", "count", 6),
                    (
                        "maximumPostFaultDeviceGrowth",
                        "bytes",
                        maximum_cleanup_growth_bytes,
                    ),
                    (
                        "postFaultDeviceGrowthTolerance",
                        "bytes",
                        cleanup_tolerance_bytes,
                    ),
                    (
                        "abaResidentRepeatMaximumPixelError",
                        "u8",
                        resident_repeat_max_error,
                    ),
                    (
                        "abaResidentRepeatMeanPixelError",
                        "pixel-micro-units",
                        resident_repeat_mean_error,
                    ),
                    ("abaBoundedMaximumPixelError", "u8", bounded_max_error),
                    (
                        "abaBoundedMeanPixelError",
                        "pixel-micro-units",
                        bounded_mean_error,
                    ),
                ],
            ),
        },
    )?;
    fragment["strategy"] = strategy;
    fragment["loadShape"] = json!(load_shape_key(actual_calibration.load_shape));
    fragment["observedMemory"] = json!({
        "conditioning": cuda_phase_metrics(conditioning_bytes),
        "denoise": cuda_phase_metrics(denoise_bytes),
        "decode": cuda_phase_metrics(decode_bytes),
        "overall": cuda_phase_metrics(overall_bytes),
    });
    if let Some(scenarios) = fragment["scenarios"].as_array_mut() {
        for scenario in scenarios {
            match scenario.get("name").and_then(Value::as_str) {
                Some("cancel") | Some("error") => {
                    let name = scenario["name"].clone();
                    *scenario = json!({
                        "name": name,
                        "result": "passed",
                        "cleanupVerified": true,
                        "warmFollowUpPassed": true,
                    });
                }
                Some("loadability") => {
                    *scenario = json!({ "name": "loadability", "result": "passed" });
                }
                _ => {}
            }
        }
    }
    Ok(fragment)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct WanPhasePeaks {
    conditioning: u64,
    denoise: u64,
    decode: u64,
    overall: u64,
}

impl WanPhasePeaks {
    fn from_phases(conditioning: u64, denoise: u64, decode: u64) -> Self {
        Self {
            conditioning,
            denoise,
            decode,
            overall: conditioning.max(denoise).max(decode),
        }
    }

    fn json(self) -> Value {
        json!({
            "conditioning": cuda_phase_metrics(self.conditioning),
            "denoise": cuda_phase_metrics(self.denoise),
            "decode": cuda_phase_metrics(self.decode),
            "overall": cuda_phase_metrics(self.overall),
        })
    }

    fn predicted_json(self) -> Value {
        json!({
            "conditioning": self.conditioning,
            "denoise": self.denoise,
            "decode": self.decode,
            "overall": self.overall,
        })
    }
}

/// One phase-sampled Candle Wan clip under an active memory-strategy scope.
///
/// The boundary walk is the SAME `protocol::next_reference_phase` sequence the five-rung image arm
/// uses — resident exposes `Step`/`Decoding`, staged exposes `Renderer`/`Decoding` — so the two
/// candle arms cannot disagree about what "the denoise phase" means in a record.
fn wan_scoped_render(
    generator: &dyn runtime_cuda::gen_core::Generator,
    context: &MemoryRunContext,
    geometry: protocol::VideoGeometry,
    fps: u32,
    seed: u64,
    vram: &mut VramProbe,
) -> Result<(Vec<runtime_cuda::gen_core::Image>, u32, WanPhasePeaks), String> {
    let mut scope = generator
        .begin_memory_strategy_request(context)
        .map_err(|error| format!("begin {WAN_PROVIDER} video scope: {error}"))?
        .ok_or_else(|| format!("{WAN_PROVIDER} selection did not create a provider scope"))?;
    let parameters = context.selection.parameters;
    if let (Some(edge), Some(overlap)) = (parameters.decode_tile_edge, parameters.decode_overlap) {
        scope
            .configure_decode(edge, overlap, context.geometry)
            .map_err(|error| format!("configure {WAN_PROVIDER} decode tuple: {error}"))?;
    }
    let mut generation = wan_generation_request(geometry, fps, seed);
    scope
        .configure_request(&mut generation)
        .map_err(|error| format!("apply {WAN_PROVIDER} request-scoped strategy: {error}"))?;
    scope
        .enter_phase(MemoryPhase::Conditioning)
        .map_err(|error| format!("enter {WAN_PROVIDER} conditioning phase: {error}"))?;

    let generation_sample = vram.phase();
    let mut phase_sample = Some(vram.phase());
    let mut phase = MemoryPhase::Conditioning;
    let mut conditioning_peak_gb = None;
    let mut denoise_peak_gb = None;
    let mut decode_peak_gb = None;
    let mut phase_error = None;
    let result = generator.generate(&generation, &mut |progress| {
        if phase_error.is_some() {
            return;
        }
        let boundary = match progress {
            Progress::Loading(runtime_cuda::gen_core::LoadPhase::Renderer) => {
                protocol::ReferenceBoundary::RendererLoad
            }
            Progress::Step { current: 1, .. } => protocol::ReferenceBoundary::FirstDenoiseStep,
            Progress::Decoding => protocol::ReferenceBoundary::Decoding,
            _ => return,
        };
        let Some(next) = protocol::next_reference_phase(reference_phase(phase), boundary) else {
            return;
        };
        let peak = phase_sample.take().map(|sample| vram.end_observed(sample));
        match phase {
            MemoryPhase::Conditioning => conditioning_peak_gb = peak,
            MemoryPhase::Denoise => denoise_peak_gb = peak,
            MemoryPhase::Decode => decode_peak_gb = peak,
        }
        if let Err(error) = scope.leave_phase(phase) {
            phase_error = Some(format!("leave {WAN_PROVIDER} {phase:?}: {error}"));
            return;
        }
        let next = memory_phase(next);
        if let Err(error) = scope.enter_phase(next) {
            phase_error = Some(format!("enter {WAN_PROVIDER} {next:?}: {error}"));
            return;
        }
        phase = next;
        phase_sample = Some(vram.phase());
    });
    if let Some(sample) = phase_sample.take() {
        let terminal_peak_gb = vram.end_observed(sample);
        match phase {
            MemoryPhase::Conditioning => conditioning_peak_gb = Some(terminal_peak_gb),
            MemoryPhase::Denoise => denoise_peak_gb = Some(terminal_peak_gb),
            MemoryPhase::Decode => decode_peak_gb = Some(terminal_peak_gb),
        }
    }
    vram.end_gen(generation_sample);
    if let Some(message) = phase_error {
        let _ = scope.finish(MemoryRunOutcome::Error {
            message: message.clone(),
        });
        return Err(message);
    }
    let output = match result {
        Ok(output) => output,
        Err(error) => {
            let message = error.to_string();
            let _ = scope.finish(MemoryRunOutcome::Error {
                message: message.clone(),
            });
            return Err(format!("{WAN_PROVIDER} video generation failed: {message}"));
        }
    };
    scope
        .leave_phase(phase)
        .map_err(|error| format!("leave {WAN_PROVIDER} terminal phase: {error}"))?;
    scope
        .finish(MemoryRunOutcome::Complete)
        .map_err(|error| format!("finish {WAN_PROVIDER} video scope: {error}"))?;

    let (frames, reported_fps) = wan_video_frames(output)?;
    if frames.len() != geometry.frames as usize {
        return Err(format!(
            "{WAN_PROVIDER} rendered {} frames, but the calibrated geometry declares {}",
            frames.len(),
            geometry.frames
        ));
    }
    let peaks =
        WanPhasePeaks::from_phases(
            decimal_gb_to_bytes(conditioning_peak_gb.ok_or_else(|| {
                format!("{WAN_PROVIDER} render did not expose a conditioning boundary")
            })?),
            decimal_gb_to_bytes(denoise_peak_gb.ok_or_else(|| {
                format!("{WAN_PROVIDER} render did not expose a denoise boundary")
            })?),
            decimal_gb_to_bytes(
                decode_peak_gb
                    .ok_or_else(|| format!("{WAN_PROVIDER} render did not complete decode"))?,
            ),
        );
    Ok((frames, reported_fps, peaks))
}

/// sc-19057 — the executed Candle Wan2.2 TI2V-5B video capture arm.
///
/// Everything that can be decided from the plan alone is decided BEFORE the artifact roots are
/// resolved, and everything that needs the pinned provider is decided before the cold load, so a
/// malformed row costs a millisecond rather than a multi-minute video render.
fn run_wan(request: &Value) -> Result<Value, String> {
    let geometry = validate_wan_target(request)?;
    protocol::validate_plain_overlay_target(request, WAN_PLAIN_EXECUTION_PATH)?;
    let tier = wan_planned_tier(request)?.to_owned();
    let (fps, seed) = planned_wan_capture(request, &tier, geometry)?;
    let selection = wan_planned_selection(request, &tier)?;
    let planned_fingerprint = protocol::planned(request)?
        .get("calibrationFingerprint")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.calibrationFingerprint must be a string".to_owned())?
        .to_owned();
    let planned_load_shape = protocol::planned(request)?
        .get("loadShape")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.loadShape must be a string".to_owned())?
        .to_owned();
    let hardware_bytes = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.memoryBytes must be an integer".to_owned())?;

    let repository = protocol::required_env("SCENEWORKS_WAN_REPOSITORY")?;
    let revision = protocol::required_env("SCENEWORKS_WAN_REVISION")?;
    let root = std::fs::canonicalize(PathBuf::from(protocol::required_env(
        "SCENEWORKS_WAN_ROOT",
    )?))
    .map_err(|error| format!("canonicalize SCENEWORKS_WAN_ROOT: {error}"))?;
    // The root must end in the PLANNED tier's directory, so a stale `…/q4` export cannot satisfy a
    // q8 plan and quietly re-label another tier's peaks.
    protocol::validate_huggingface_snapshot_root(
        &root,
        &repository,
        &revision,
        &tier,
        protocol::WAN_CANDLE_REPOSITORY,
    )?;

    // `validate_contract_route` refuses anything but a snapshot directory loaded eagerly at bf16
    // with no controls, PiD, external components or adapters — which is exactly `LoadSpec::new`'s
    // default shape plus the packed tier's quant. `Sequential` is the policy `video_jobs::candle`
    // forces for this model in production (sc-13175), and it also selects the provider's
    // `…-candle-sequential-load-v1` calibration identity, so the capture measures the shipped load.
    let spec = LoadSpec::new(WeightsSource::Dir(root))
        .with_offload_policy(OffloadPolicy::Sequential)
        .with_load_shape(LoadShape::EagerMaterialization)
        .with_quant(
            wan_numeric_tier(&tier)?
                .quant
                .ok_or_else(|| format!("{WAN_CALIBRATION_LABEL} tier {tier} carries no quant"))?,
        );
    let catalog =
        runtime_cuda::catalog().map_err(|error| format!("build CUDA catalog: {error}"))?;

    let mut vram = VramProbe::start_rendered().assert_idle(1.0);
    let load_sample = vram.phase();
    let generator = catalog
        .media()
        .load(WAN_PROVIDER, &spec)
        .map_err(|error| format!("load real {WAN_PROVIDER} {tier} generator: {error}"))?;
    vram.end_load(load_sample);
    // Past this line a load has demonstrably happened, which is the ONLY thing that entitles the
    // receipt to report `loadability: passed` (sc-18808's review finding, in the honest direction).
    let mut outcomes = WanScenarioOutcomes {
        loadability: true,
        ..WanScenarioOutcomes::default()
    };

    let contract = generator.memory_strategy_contract().ok_or_else(|| {
        format!(
            "loaded {WAN_PROVIDER} exposes no memory-strategy contract at {}",
            protocol::INFERENCE_PIN
        )
    })?;
    contract.validate_selection(&selection).map_err(|error| {
        format!("pinned {WAN_PROVIDER} provider rejected the planned selection: {error}")
    })?;
    let calibration = contract
        .calibration
        .as_ref()
        .ok_or_else(|| format!("pinned {WAN_PROVIDER} provider has no calibration identity"))?;
    if planned_fingerprint != calibration.fingerprint {
        return Err(format!(
            "plan/provider calibration mismatch: plan={planned_fingerprint}, pinned provider={}",
            calibration.fingerprint
        ));
    }
    let actual_load_shape = load_shape_key(calibration.load_shape);
    if planned_load_shape != actual_load_shape {
        return Err(format!(
            "plan/provider load-shape mismatch: plan={planned_load_shape}, pinned provider={actual_load_shape}"
        ));
    }
    let engaged = contract.engaged_composition(selection.strategy);
    let decode_tiling_engaged = engaged.contains(&MemoryStrategy::BoundedDecode);
    let strategy = measured_strategy(request, &selection, &engaged)?;

    let context = MemoryRunContext {
        selection,
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
        calibration_abi: calibration.abi,
        calibration_fingerprint: calibration.fingerprint.clone(),
        load_shape: calibration.load_shape,
        // The provider's route gate compares `mode.as_key()` against this exact string; there is no
        // video variant on `MemoryMode`, and a `TextToImage` context is rejected outright.
        mode: MemoryMode::Other("text_to_video".to_owned()),
        has_reference: false,
        use_pid: false,
        has_phases: false,
        geometry: MemoryGeometry {
            width: geometry.width,
            height: geometry.height,
            batch: 1,
            frames: geometry.frames,
            reference_count: 0,
        },
        overlay: None,
        budget: MemoryBudget {
            total_bytes: hardware_bytes,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 2 * GIB,
        },
        predicted_peak_bytes: hardware_bytes.saturating_sub(2 * GIB),
        cache_state: MemoryCacheState::Cold,
        evidence_revision: format!("sc-19057@{}", protocol::INFERENCE_PIN),
    };

    let (measured, reported_fps, peaks) =
        wan_scoped_render(generator.as_ref(), &context, geometry, fps, seed, &mut vram)?;
    if reported_fps != fps {
        return Err(format!(
            "{WAN_PROVIDER} returned a {reported_fps} fps clip for a {fps} fps request"
        ));
    }
    if [peaks.conditioning, peaks.denoise, peaks.decode].contains(&0) {
        return Err(format!(
            "a synchronized {WAN_PROVIDER} phase reported a zero device peak"
        ));
    }
    let first = measured
        .first()
        .ok_or_else(|| format!("{WAN_PROVIDER} clip has no first frame"))?;
    let first_frame_nondegenerate = wan_frame_is_nondegenerate(first);
    if !first_frame_nondegenerate {
        return Err(format!(
            "{WAN_PROVIDER} clip's first frame is a single flat colour; a degenerate decode is not \
             calibration evidence"
        ));
    }

    // Admission scenarios against the LOADED provider, at the measured ceiling.
    let mut exact = context.clone();
    exact.predicted_peak_bytes = peaks.overall;
    exact.budget.total_bytes = peaks.overall;
    exact.budget.reserved_headroom_bytes = 0;
    outcomes.exact_fit = matches!(
        generator.memory_strategy_safety_check(&exact),
        MemorySafetyDecision::Accept
    );
    if !outcomes.exact_fit {
        return Err(format!(
            "{WAN_PROVIDER} rejected an exact-fit calibrated budget at the measured ceiling"
        ));
    }
    let mut unknown = context.clone();
    unknown.budget.total_bytes = 0;
    outcomes.unknown_budget = matches!(
        generator.memory_strategy_safety_check(&unknown),
        MemorySafetyDecision::Reject { .. }
    );
    if !outcomes.unknown_budget {
        return Err(format!(
            "{WAN_PROVIDER} accepted an unknown/zero memory budget"
        ));
    }
    let mut stale = context.clone();
    stale.calibration_fingerprint = format!("{}-stale", calibration.fingerprint);
    outcomes.stale_evidence = matches!(
        generator.memory_strategy_safety_check(&stale),
        MemorySafetyDecision::Reject { .. }
    );
    if !outcomes.stale_evidence {
        return Err(format!(
            "{WAN_PROVIDER} accepted stale calibration evidence"
        ));
    }

    // Warm repeat determinism on the same loaded provider, then the falsifiability check that makes
    // the thresholds mean something.
    let (repeat, _, repeat_peaks) =
        wan_scoped_render(generator.as_ref(), &context, geometry, fps, seed, &mut vram)?;
    let (maximum_error, mean_error, rms_error) = wan_clip_max_mean_rms(&measured, &repeat)?;
    if !wan_quality_passes(maximum_error, mean_error, rms_error) {
        return Err(format!(
            "{WAN_PROVIDER} warm repeat exceeded the determinism envelope: max={maximum_error:.6}, \
             mean={mean_error:.6}, rms={rms_error:.6}"
        ));
    }
    outcomes.warm_repeat = true;
    let mutated = measured
        .iter()
        .map(wan_negative_mutation)
        .collect::<Vec<_>>();
    let (mutated_maximum, mutated_mean, mutated_rms) = wan_clip_max_mean_rms(&mutated, &measured)?;
    if wan_quality_passes(mutated_maximum, mutated_mean, mutated_rms) {
        return Err(format!(
            "{WAN_PROVIDER} output mutation did not breach the determinism envelope"
        ));
    }

    let smi = NvidiaSmi::resolve()?;
    let post_render_device_bytes = smi.used_bytes()?;
    let sweep = wan_complete_sweep(request)?;
    let range_verified = sweep
        .get("rangeVerified")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let status = wan_receipt_status(outcomes, range_verified);
    let scenarios = wan_scenarios(outcomes, peaks.overall, WAN_LIFECYCLE_BLOCKER)?;
    let mut fragment = json!({
        "status": status,
        "strategy": strategy,
        "loadShape": actual_load_shape,
        "artifact": artifact(&repository, &revision, &tier),
        "sweep": sweep,
        "scenarios": scenarios,
        "predictedPeakBytes": peaks.predicted_json(),
        "observedMemory": peaks.json(),
        "quality": {
            "contract": "identical artifact, prompt, seed, geometry, frames, fps, tier, provider contract and selected request scope; measured clip versus a clean warm repeat, compared over every frame",
            "identicalInputs": true,
            "result": "passed",
            "maximumError": maximum_error,
            "meanError": mean_error,
            "rootMeanSquareError": rms_error,
            "maximumErrorThreshold": WAN_MAX_THRESHOLD,
            "meanErrorThreshold": WAN_MEAN_THRESHOLD,
            "rootMeanSquareErrorThreshold": WAN_RMS_THRESHOLD,
        },
        "negativeMutation": null,
        "loadability": {
            "result": "passed",
            "resolvedPathFingerprint": loadability_fingerprint(&repository, &revision, &tier),
        },
        // NB there is no top-level `output` block here, deliberately. The MLX LTX arm emits one, but
        // `packages/schemas/memory-calibration.schema.json`'s record object is
        // `additionalProperties: false` and declares no `output` property — the harness copies a
        // fixed field set out of the fragment, so that block never reaches a record and its
        // `firstFrameNondegenerate` claim is validated by nobody. The same facts are carried below,
        // in `diagnostics.measurements`, which DOES land in the record and which the video-curve fit
        // already reads.
        "diagnostics": protocol::diagnostics(
            "memory-candle-adapter:wan2-2-ti2v-5b-video",
            "executed",
            [WAN_LIFECYCLE_BLOCKER.to_owned()],
            [
                ("conditioningDevicePeakDelta", "bytes", peaks.conditioning),
                ("denoiseDevicePeakDelta", "bytes", peaks.denoise),
                ("decodeDevicePeakDelta", "bytes", peaks.decode),
                ("overallDevicePeakDelta", "bytes", peaks.overall),
                ("predictedOverallCeiling", "bytes", peaks.overall),
                ("warmRepeatOverallDevicePeakDelta", "bytes", repeat_peaks.overall),
                ("postRenderDeviceUsed", "bytes", post_render_device_bytes),
                // Required by scripts/fit-ltx-temporal-form.mjs: the regressors and the decode-pass
                // axis of the emitted curve id.
                ("renderedFrames", "count", u64::from(geometry.frames)),
                ("latentTemporalDepth", "count", u64::from(geometry.latent_frames)),
                ("outputFps", "count", u64::from(reported_fps)),
                ("decodeTilingEngaged", "count", u64::from(decode_tiling_engaged)),
                ("audioTrackDecoded", "count", 0),
                // Measured, not asserted: this is the value `wan_frame_is_nondegenerate` returned
                // for the rendered first frame, and the capture aborts above when it is false — so
                // a record can only ever carry the 1, and it carries it because it was measured
                // rather than because a literal said so.
                (
                    "firstFrameNondegenerate",
                    "count",
                    u64::from(first_frame_nondegenerate),
                ),
                (
                    "negativeMutationMaximumErrorPer255",
                    "count",
                    (mutated_maximum * 255.0).round() as u64,
                ),
                (
                    "negativeMutationMeanErrorPer255",
                    "count",
                    (mutated_mean * 255.0).round() as u64,
                ),
                (
                    "negativeMutationRootMeanSquareErrorPer255",
                    "count",
                    (mutated_rms * 255.0).round() as u64,
                ),
            ],
        ),
        "capturedAt": protocol::captured_at(),
    });
    protocol::settle_plain_overlay_scenario(request, &mut fragment, WAN_PLAIN_EXECUTION_PATH)?;
    Ok(fragment)
}

fn main() {
    let request = protocol::request_from_stdin().unwrap_or_else(|error| protocol::fail(error));
    let response = match protocol::action(&request).unwrap_or_else(|error| protocol::fail(error)) {
        "probe" => probe(),
        "run" => run(&request),
        "run_batch" => run_five_rung_batch(&request),
        other => Err(format!("unsupported action {other:?}")),
    }
    .unwrap_or_else(|error| protocol::fail(error));
    protocol::write_response(&response).unwrap_or_else(|error| protocol::fail(error));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen_request() -> Value {
        json!({
            "planned": {
                "target": { "provider": "qwen_image", "overlay": "none" },
                "strategy": { "rung": "resident", "parameters": {} },
                "loadShape": "deferred_materialization"
            }
        })
    }

    #[test]
    fn qwen_plan_routes_to_the_qwen_base_execution_path() {
        let request = qwen_request();
        assert_eq!(planned_provider(&request).unwrap(), "qwen_image");
        assert_eq!(
            plain_execution_path(&request).unwrap(),
            QWEN_PLAIN_EXECUTION_PATH
        );
        assert_eq!(
            planned_memory_strategy(&request).unwrap(),
            MemoryStrategy::Resident
        );
    }

    #[test]
    fn edit_plan_is_not_mislabeled_as_base_qwen_conformance() {
        let mut request = qwen_request();
        request["planned"]["target"]["provider"] = json!("qwen_image_edit");
        let error = plain_execution_path(&request).unwrap_err();
        assert!(error.contains("qwen_image_edit"));
        assert!(error.contains("does not implement"));
    }

    #[test]
    fn deferred_materialization_establishes_retention_baseline_after_resident_rung() {
        let mut baseline = None;
        update_warmed_retention_baseline(&mut baseline, 12 * GIB).unwrap();
        assert_eq!(baseline, Some(12 * GIB));
        update_warmed_retention_baseline(&mut baseline, 12 * GIB + 64 * MIB).unwrap();
    }

    #[test]
    fn warmed_retention_baseline_rejects_later_growth() {
        let mut baseline = Some(12 * GIB);
        let error =
            update_warmed_retention_baseline(&mut baseline, 12 * GIB + 64 * MIB + 1).unwrap_err();
        assert!(error.contains("above the warmed resident baseline"));
    }

    /// A single planned case at `frames`, minimal enough that the geometry guard is the FIRST thing
    /// that can reject it — no weight root, no environment.
    ///
    /// `fixture` is LOAD-BEARING here, not decoration. [`run`] picks its dispatch branch from it: a
    /// `fresh-five-rung-` prefix routes into [`run_five_rung_reference`], and ANY OTHER fixture on
    /// `krea_2_turbo` falls through to the inline Krea arm instead. A table that only ever passes
    /// one prefix therefore exercises only one of the two branches, which is precisely how this
    /// test shipped blind to [`run`]'s own guard once (sc-18808 re-review).
    fn still_planned_case_with_fixture(
        provider: &str,
        rung: &str,
        frames: u64,
        fixture: &str,
    ) -> Value {
        json!({
            "backend": "candle",
            "target": {
                "provider": provider,
                "modelId": provider,
                "tier": "q4",
                "mode": "text_to_image",
                "overlay": "none",
                "geometry": { "width": 1024, "height": 1024, "batch": 1, "frames": frames }
            },
            "loadShape": "deferred_materialization",
            "strategy": { "rung": rung, "parameters": {} },
            "calibrationFingerprint": "unused",
            "fixture": fixture
        })
    }

    /// The five-rung shape — the only one [`run_five_rung_batch`] accepts.
    fn still_planned_case(provider: &str, rung: &str, frames: u64) -> Value {
        still_planned_case_with_fixture(provider, rung, frames, "fresh-five-rung-unused")
    }

    /// The canonical five-rung batch shape `run_five_rung_batch` requires, at `frames`.
    fn still_batch_request(provider: &str, frames: u64) -> Value {
        let planned: Vec<Value> = [
            "resident",
            "staged_residency",
            "bounded_decode",
            "bounded_attention",
            "bounded_transformer_residency",
        ]
        .into_iter()
        .map(|rung| still_planned_case(provider, rung, frames))
        .collect();
        json!({ "action": "run_batch", "planned": planned })
    }

    /// sc-18808 — the Candle twin of the MLX adapter's
    /// `every_image_arm_still_refuses_a_multi_frame_geometry`.
    ///
    /// BOTH Candle arms hardcoded `frames: 1` into `MemoryGeometry` while reading only
    /// `width`/`height` from the plan, so a plan row declaring any other frame count would have
    /// rendered ONE frame and emitted a record claiming a geometry it was never asked for.
    ///
    /// Two entry points are reachable from `main`, and both must refuse with the exact pinned
    /// wording before any environment or weight work:
    ///
    /// * `run` — the dispatcher. Its guard stands in front of BOTH of its branches, and the branch
    ///   is chosen by `planned.fixture`, which is why the table below carries the fixture instead of
    ///   hardcoding one. A `fresh-five-rung-` prefix (or the `qwen_image` provider) routes into
    ///   `run_five_rung_reference`; ANY OTHER fixture on `krea_2_turbo` falls through to the INLINE
    ///   Krea arm — the arm that resolves `SCENEWORKS_KREA_REPOSITORY` and then writes its own
    ///   `MemoryGeometry { frames: 1 }`. Five SHIPPED Candle plan rows carry the second shape
    ///   (`krea-q4-1024-seed42` and its q8/bf16/768/v2 siblings), so it is the live one; the third
    ///   row below is one of them verbatim. Until it was added, every case in this table began
    ///   `fresh-five-rung-`, so all of them short-circuited into `run_five_rung_reference` and
    ///   `run`'s own guard was shadowed by the redundant copy at the head of that function —
    ///   deleting `run`'s guard left this test green.
    /// * `run_five_rung_batch` — reached straight from `main`, so `run`'s guard never sees it. Its
    ///   per-item pre-load loop is the guard under test; the fixture is irrelevant to it, so the
    ///   canonical five-rung shape is the only one it is exercised with.
    ///
    /// The other two copies of the refusal are redundant defense-in-depth and are NOT what this
    /// test pins: the one at the head of `run_five_rung_reference` (whose only caller is `run`,
    /// which already refused) and the one in `run_five_rung_reference_loaded` (reachable only after
    /// a real generator load, so no unit test can enter it). Removing either of those alone leaves
    /// this suite green — which is exactly why a future reader must not "clean up" `run`'s or
    /// `run_five_rung_batch`'s on the strength of them still being there.
    #[test]
    fn every_candle_arm_still_refuses_a_multi_frame_geometry() {
        for (provider, label, fixture) in [
            (QWEN_ID, QWEN_STILL_CALIBRATION, "fresh-five-rung-unused"),
            (KREA_ID, KREA_STILL_CALIBRATION, "fresh-five-rung-unused"),
            // The inline Krea arm — a real shipped plan fixture, which the two rows above cannot
            // reach.
            (KREA_ID, KREA_STILL_CALIBRATION, "krea-q4-1024-seed42"),
        ] {
            for frames in [0_u64, 2, 97] {
                let expected = format!("{label} requires geometry.frames == 1, got {frames}");
                let request = json!({
                    "action": "run",
                    "planned": still_planned_case_with_fixture(
                        provider, "resident", frames, fixture,
                    )
                });
                assert_eq!(
                    run(&request).expect_err("the Candle dispatcher must refuse a video geometry"),
                    expected,
                    "run: {provider} at frames={frames} via fixture {fixture:?}"
                );
            }
        }
        for (provider, label) in [
            (QWEN_ID, QWEN_STILL_CALIBRATION),
            (KREA_ID, KREA_STILL_CALIBRATION),
        ] {
            for frames in [0_u64, 2, 97] {
                let expected = format!("{label} requires geometry.frames == 1, got {frames}");
                assert_eq!(
                    run_five_rung_batch(&still_batch_request(provider, frames))
                        .expect_err("the Candle batch arm must refuse a video geometry"),
                    expected,
                    "run_batch: {provider} at frames={frames}"
                );
            }
        }
    }

    /// The third row above is not decoration: the fixtures the SHIPPED Candle plan gives the inline
    /// Krea arm really do take that branch, so `run`'s own guard — not the redundant copy inside
    /// [`run_five_rung_reference`] — is the one standing in front of them.
    ///
    /// Asserted against the dispatch predicate itself rather than by observing an error, because
    /// both branches resolve the same `SCENEWORKS_KREA_REPOSITORY` and would report the same
    /// sentence: the error is not a routing witness, the predicate is. Widening the prefix (or
    /// renaming these fixtures into it) would re-shadow `run`'s guard, and this reds when it does.
    #[test]
    fn the_shipped_krea_fixtures_take_the_inline_arm_not_the_five_rung_branch() {
        for fixture in [
            "krea-q4-1024-seed42",
            "krea-q8-1024-seed42",
            "krea-bf16-1024-seed42",
            "krea-q4-768-seed42",
            "krea-q4-1024-seed42-v2-candidate",
        ] {
            let request = json!({
                "planned": still_planned_case_with_fixture(KREA_ID, "resident", 1, fixture)
            });
            assert!(
                !routes_to_five_rung_reference(&request).unwrap(),
                "{fixture} must reach the inline Krea arm"
            );
        }
        for (provider, fixture) in [
            (KREA_ID, "fresh-five-rung-krea-q4-1024-seed16402-step2"),
            (QWEN_ID, "qwen-image-candle-q4-seed15817-step2"),
        ] {
            let request = json!({
                "planned": still_planned_case_with_fixture(provider, "resident", 1, fixture)
            });
            assert!(
                routes_to_five_rung_reference(&request).unwrap(),
                "{fixture} must reach the five-rung reference path"
            );
        }
    }

    // ─── sc-19057: the Candle Wan video arm ─────────────────────────────────────────────────────

    /// A `candle:wan2_2_ti2v_5b` plan row at the sc-19057 capture shape. Minimal enough that every
    /// assertion below reaches its guard before any environment variable or weight snapshot is
    /// touched: `run_wan` validates the target, overlay, tier, fixture and selection first, and only
    /// then resolves `SCENEWORKS_WAN_*`.
    fn wan_planned_case(width: u32, height: u32, frames: u32, tier: &str, fps: u32) -> Value {
        json!({
            "backend": "candle",
            "target": {
                "provider": WAN_PROVIDER,
                "modelId": WAN_MANIFEST_MODEL_ID,
                "tier": tier,
                "mode": "text_to_video",
                "overlay": "none",
                "geometry": { "width": width, "height": height, "batch": 1, "frames": frames }
            },
            "loadShape": "eager_materialization",
            "strategy": {
                "rung": "staged_residency",
                "engagedRungs": ["resident", "staged_residency"],
                "parameters": {}
            },
            "calibrationFingerprint": "sc-19223-wan2-2-ti2v-5b-candle-sequential-load-v1",
            "fixture": format!(
                "wan2-2-ti2v-5b-candle-{tier}-{width}x{height}-f{frames}-fps{fps}-seed{WAN_SEED}"
            )
        })
    }

    fn wan_run_request(width: u32, height: u32, frames: u32) -> Value {
        json!({
            "action": "run",
            "hardware": { "memoryBytes": 96 * GIB },
            "planned": wan_planned_case(width, height, frames, "q4", WAN_CALIBRATED_FPS)
        })
    }

    /// The transcription half of the ladder binding. `wan_frame_count_matches_the_sc_19057_
    /// calibration_ladder` in `crates/sceneworks-core/src/video_request.rs` pins the SAME ten pairs
    /// against the shipped function, so a drift on either side reds.
    #[test]
    fn wan_frame_ladder_port_matches_the_transcribed_shipped_ladder() {
        for (duration, fps, expected) in [
            (4, 16, 61),
            (5, 16, 77),
            (6, 16, 93),
            (7, 16, 109),
            (8, 16, 125),
            (4, 24, 93),
            (5, 24, 117),
            (6, 24, 141),
            (7, 24, 165),
            (8, 24, 189),
        ] {
            assert_eq!(
                wan_snapped_frame_count(duration * fps),
                expected,
                "{duration}s x {fps}fps"
            );
            assert_eq!(
                (expected - 1) % WAN_TEMPORAL_SCALE,
                0,
                "every ladder rung is on the 1 + 4k lattice"
            );
        }
        // The envelope spans the CAPTURABLE cadence only, so its floor is the shortest duration at
        // 24 fps and not the 61-frame 4s-at-16fps rung in the table above. The 16 fps column stays
        // in that table because it is a real product geometry the ladder port must still reproduce.
        assert_eq!(WAN_FRAME_ENVELOPE, (93, 189));
        let (minimum, maximum) = WAN_FRAME_ENVELOPE;
        assert_eq!(minimum, wan_snapped_frame_count(4 * WAN_CALIBRATED_FPS));
        assert_eq!(maximum, wan_snapped_frame_count(8 * WAN_CALIBRATED_FPS));
        assert_eq!(WAN_CAPTURABLE_FPS, [WAN_CALIBRATED_FPS]);
        for fps in WAN_CAPTURABLE_FPS {
            assert!(
                WAN_FPS.contains(&fps),
                "a capturable cadence must be shipped"
            );
        }
    }

    /// Every frame count exercised here is a rung the CAPTURABLE 24 fps cadence actually reaches
    /// (93 = 4s, 117 = 5s, 189 = 8s), so this cannot pass by admitting a geometry no product request
    /// can produce — which is exactly what the pre-review envelope did with the 16 fps rungs.
    #[test]
    fn the_wan_arm_accepts_every_declared_resolution_across_the_frame_envelope() {
        for (width, height) in WAN_RESOLUTIONS {
            for frames in [WAN_FRAME_ENVELOPE.0, 117, WAN_FRAME_ENVELOPE.1] {
                let request =
                    json!({ "planned": wan_planned_case(width, height, frames, "q4", 24) });
                let geometry = validate_wan_target(&request)
                    .unwrap_or_else(|error| panic!("{width}x{height} f{frames}: {error}"));
                assert_eq!(geometry.frames, frames);
                assert_eq!(
                    geometry.latent_frames,
                    1 + (frames - 1) / WAN_TEMPORAL_SCALE
                );
                assert!(
                    geometry.latent_frames > 1,
                    "a video record is multi-latent-frame"
                );
            }
        }
    }

    /// The negative half, mirroring the shape of the MLX LTX arm's pinned envelope test. The still
    /// case is the load-bearing one: `1 % 4 == 1` puts it ON the lattice, so only the envelope floor
    /// can stop this arm capturing a single-frame record for a video model.
    #[test]
    fn the_wan_arm_rejects_out_of_envelope_geometry_with_a_named_reason() {
        for (width, height, frames, expected) in [
            (768, 512, 117, "declared limits.resolutions"),
            (1024, 1024, 117, "declared limits.resolutions"),
            (832, 480, 118, "1 + 4k"),
            (832, 480, 120, "1 + 4k"),
            (832, 480, 1, "duration/fps envelope"),
            (832, 480, 5, "duration/fps envelope"),
            (832, 480, 193, "duration/fps envelope"),
            // sc-19057 review: real product geometries (4s and 5s at 16 fps) that this arm still
            // may not capture, because the cadence reaching them is the one the route refuses.
            (832, 480, 61, "duration/fps envelope"),
            (832, 480, 77, "duration/fps envelope"),
        ] {
            let request = json!({ "planned": wan_planned_case(width, height, frames, "q4", 24) });
            let error = validate_wan_target(&request)
                .expect_err("an out-of-envelope Wan geometry must be refused");
            assert!(
                error.contains(expected),
                "{width}x{height} f{frames}: {error}"
            );
        }
    }

    /// Reached through `run`, so it also proves the dispatcher routes the video provider to the
    /// video envelope rather than to the image arms' `frames == 1` refusal — and that it does so
    /// before any environment resolution.
    #[test]
    fn the_wan_arm_refuses_a_single_frame_capture_through_the_dispatcher() {
        let error =
            run(&wan_run_request(832, 480, 1)).expect_err("a video arm must not capture a still");
        assert_eq!(
            error,
            "Candle Wan2.2 TI2V-5B calibration requires geometry.frames within the declared \
             duration/fps envelope [93, 189], got 1"
        );
        // ...and the multi-frame geometry the image arms refuse is exactly what this one accepts.
        let admitted = validate_wan_target(&json!({
            "planned": wan_planned_case(832, 480, 117, "q4", 24)
        }));
        assert!(admitted.is_ok(), "{admitted:?}");
    }

    /// The video arm must not be reachable through the five-rung batch entry point, which `main`
    /// dispatches directly and which carries no video geometry validation at all.
    #[test]
    fn the_five_rung_batch_arm_refuses_the_video_provider_by_name() {
        let error = run_five_rung_batch(&still_batch_request(WAN_PROVIDER, 117))
            .expect_err("the batch arm has no video envelope and must refuse by name");
        assert!(error.contains(WAN_PROVIDER), "{error}");
        assert!(error.contains("does not implement"), "{error}");
    }

    #[test]
    fn the_wan_arm_refuses_a_foreign_provider_and_an_engine_id_used_as_the_catalog_model_id() {
        let mut foreign = json!({ "planned": wan_planned_case(832, 480, 117, "q4", 24) });
        foreign["planned"]["target"]["provider"] = json!("ltx_2_3_distilled");
        let error = validate_wan_target(&foreign).unwrap_err();
        assert!(error.contains("ltx_2_3_distilled"), "{error}");
        assert!(error.contains("does not implement"), "{error}");

        // The exact confusion this arm exists to prevent: the video-curve fit resolves modelFamily
        // from `target.modelId` through builtin.models.jsonc, where only `wan_2_2` exists.
        let mut engine_id_as_model =
            json!({ "planned": wan_planned_case(832, 480, 117, "q4", 24) });
        engine_id_as_model["planned"]["target"]["modelId"] = json!(WAN_PROVIDER);
        let error = validate_wan_target(&engine_id_as_model).unwrap_err();
        assert!(error.contains("requires modelId \"wan_2_2\""), "{error}");
        assert!(error.contains("modelFamily"), "{error}");

        let mut still_mode = json!({ "planned": wan_planned_case(832, 480, 117, "q4", 24) });
        still_mode["planned"]["target"]["mode"] = json!("text_to_image");
        assert!(validate_wan_target(&still_mode)
            .unwrap_err()
            .contains("reference-free text_to_video mode"));
    }

    #[test]
    fn the_wan_fixture_binds_the_tier_geometry_cadence_and_seed() {
        let geometry = protocol::VideoGeometry {
            width: 832,
            height: 480,
            frames: 117,
            latent_frames: 30,
        };
        let request = json!({ "planned": wan_planned_case(832, 480, 117, "q4", 24) });
        assert_eq!(
            planned_wan_capture(&request, "q4", geometry).unwrap(),
            (24, WAN_SEED)
        );

        // A q8 plan may not be satisfied by a fixture that names another tier or geometry.
        for (tier, other) in [("q8", "q4"), ("q4", "q8")] {
            let mislabelled = json!({ "planned": wan_planned_case(832, 480, 117, other, 24) });
            assert!(planned_wan_capture(&mislabelled, tier, geometry)
                .unwrap_err()
                .contains("must start with"));
        }
        let wrong_frames = json!({ "planned": wan_planned_case(832, 480, 141, "q4", 24) });
        assert!(planned_wan_capture(&wrong_frames, "q4", geometry)
            .unwrap_err()
            .contains("must start with"));

        // 16 fps is SHIPPED but not capturable: the provider's request scope rejects any cadence
        // other than 24, so the arm says so instead of dying inside the provider hours later.
        let sixteen = json!({ "planned": wan_planned_case(832, 480, 117, "q4", 16) });
        let error = planned_wan_capture(&sixteen, "q4", geometry).unwrap_err();
        assert!(error.contains("fixed at 24 fps"), "{error}");
        assert!(error.contains("not capturable"), "{error}");

        // A cadence outside limits.fps entirely is refused as an undeclared cadence.
        let thirty = json!({ "planned": wan_planned_case(832, 480, 117, "q4", 30) });
        assert!(planned_wan_capture(&thirty, "q4", geometry)
            .unwrap_err()
            .contains("declared limits.fps"));

        let mut wrong_seed = json!({ "planned": wan_planned_case(832, 480, 117, "q4", 24) });
        wrong_seed["planned"]["fixture"] =
            json!("wan2-2-ti2v-5b-candle-q4-832x480-f117-fps24-seed42");
        assert!(planned_wan_capture(&wrong_seed, "q4", geometry)
            .unwrap_err()
            .contains("does not match the Candle Wan calibration seed"));
    }

    #[test]
    fn the_wan_arm_measures_only_the_two_packed_tiers_it_can_name_the_bytes_of() {
        for tier in ["q4", "q8"] {
            let request = json!({ "planned": wan_planned_case(832, 480, 117, tier, 24) });
            assert_eq!(wan_planned_tier(&request).unwrap(), tier);
            assert!(wan_numeric_tier(tier).unwrap().quant.is_some());
        }
        let dense = json!({ "planned": wan_planned_case(832, 480, 117, "bf16", 24) });
        let error = wan_planned_tier(&dense).unwrap_err();
        assert!(error.contains("no bf16 arm"), "{error}");
        assert!(error.contains("Wan-AI/Wan2.2-TI2V-5B-Diffusers"), "{error}");

        let unknown = json!({ "planned": wan_planned_case(832, 480, 117, "nvfp4", 24) });
        assert!(wan_planned_tier(&unknown)
            .unwrap_err()
            .contains("unsupported"));
    }

    #[test]
    fn the_wan_selection_admits_only_the_three_implemented_rungs_and_their_own_knobs() {
        for rung in ["resident", "staged_residency"] {
            let mut request = json!({ "planned": wan_planned_case(832, 480, 117, "q4", 24) });
            request["planned"]["strategy"]["rung"] = json!(rung);
            assert!(wan_planned_selection(&request, "q4").is_ok(), "{rung}");
        }
        for missing in ["bounded_attention", "bounded_transformer_residency"] {
            let mut request = json!({ "planned": wan_planned_case(832, 480, 117, "q4", 24) });
            request["planned"]["strategy"]["rung"] = json!(missing);
            let error = wan_planned_selection(&request, "q4").unwrap_err();
            assert!(
                error.contains("is Missing in the pinned provider contract"),
                "{error}"
            );
        }

        // bounded_decode needs BOTH halves of the decode tuple ...
        let mut bounded = json!({ "planned": wan_planned_case(832, 480, 117, "q4", 24) });
        bounded["planned"]["strategy"]["rung"] = json!("bounded_decode");
        assert!(wan_planned_selection(&bounded, "q4")
            .unwrap_err()
            .contains("requires both decodeTileEdge and decodeOverlap"));
        bounded["planned"]["strategy"]["parameters"] =
            json!({ "decodeTileEdge": 384, "decodeOverlap": 64 });
        let selection = wan_planned_selection(&bounded, "q4").unwrap();
        assert_eq!(selection.strategy, MemoryStrategy::BoundedDecode);
        assert_eq!(selection.parameters.decode_tile_edge, Some(384));

        // ... and a non-bounded rung must declare none of it.
        let mut stray = json!({ "planned": wan_planned_case(832, 480, 117, "q4", 24) });
        stray["planned"]["strategy"]["parameters"] =
            json!({ "decodeTileEdge": 384, "decodeOverlap": 64 });
        assert!(wan_planned_selection(&stray, "q4")
            .unwrap_err()
            .contains("engages no decode tiler"));

        // Wan implements neither knob, so a row that selects one is a plan defect, not a widening.
        for unsupported in ["attentionChunkSize", "transformerWindowSize"] {
            let mut request = json!({ "planned": wan_planned_case(832, 480, 117, "q4", 24) });
            request["planned"]["strategy"]["parameters"] = json!({ unsupported: 64 });
            let error = wan_planned_selection(&request, "q4").unwrap_err();
            assert!(error.contains(unsupported), "{error}");
            assert!(error.contains("nothing engaged"), "{error}");
        }
    }

    /// `rangeVerified` is DERIVED from the row's parameter axes, not asserted. A zero-axis row has a
    /// singleton domain and genuinely covers it; a bounded-decode row picked one tile edge out of
    /// the production domain and has not.
    #[test]
    fn the_wan_sweep_only_claims_a_verified_range_for_a_singleton_domain() {
        let plain = json!({ "planned": wan_planned_case(832, 480, 117, "q4", 24) });
        assert_eq!(
            wan_complete_sweep(&plain).unwrap()["rangeVerified"],
            json!(true)
        );

        let mut bounded = json!({ "planned": wan_planned_case(832, 480, 117, "q4", 24) });
        bounded["planned"]["strategy"]["parameters"] =
            json!({ "decodeTileEdge": 384, "decodeOverlap": 64 });
        let sweep = wan_complete_sweep(&bounded).unwrap();
        assert_eq!(sweep["rangeVerified"], json!(false));
        assert_eq!(sweep["cases"][0]["result"], json!("passed"));
    }

    /// Receipt honesty, in the direction sc-18808's review caught on the MLX side and in the
    /// opposite one. A scenario is `passed` only when the run executed it, an unexecuted one carries
    /// the named blocker, and the one internally impossible shape — an executed scenario without a
    /// completed load — is refused rather than emitted.
    #[test]
    fn a_wan_receipt_cannot_claim_a_scenario_it_did_not_execute() {
        let executed = WanScenarioOutcomes {
            loadability: true,
            exact_fit: true,
            unknown_budget: true,
            stale_evidence: true,
            warm_repeat: true,
            cancel: false,
            error: false,
        };
        let scenarios = wan_scenarios(executed, 42_000, WAN_LIFECYCLE_BLOCKER).unwrap();
        let by_name = |name: &str| {
            scenarios
                .as_array()
                .unwrap()
                .iter()
                .find(|scenario| scenario["name"] == name)
                .unwrap()
                .clone()
        };
        assert_eq!(scenarios.as_array().unwrap().len(), 8);
        assert_eq!(by_name("loadability")["result"], "passed");
        assert_eq!(by_name("exact_fit")["result"], "passed");
        assert_eq!(by_name("exact_fit")["predictedBytes"], json!(42_000));
        assert_eq!(by_name("exact_fit")["effectiveBudgetBytes"], json!(42_000));
        assert_eq!(by_name("warm_repeat")["result"], "passed");
        for unexecuted in ["cancel", "error"] {
            assert_eq!(by_name(unexecuted)["result"], "not_run");
            assert_eq!(by_name(unexecuted)["reason"], WAN_LIFECYCLE_BLOCKER);
        }

        // Nothing executed: every scenario is not_run and loadability is NOT quietly promoted.
        let nothing = WanScenarioOutcomes::default();
        let scenarios = wan_scenarios(nothing, 0, WAN_LIFECYCLE_BLOCKER).unwrap();
        for scenario in scenarios.as_array().unwrap() {
            if scenario["name"] == "overlay" {
                continue;
            }
            assert_eq!(scenario["result"], "not_run", "{scenario}");
        }

        // The impossible shape.
        for impossible in [
            WanScenarioOutcomes {
                exact_fit: true,
                ..WanScenarioOutcomes::default()
            },
            WanScenarioOutcomes {
                warm_repeat: true,
                ..WanScenarioOutcomes::default()
            },
            WanScenarioOutcomes {
                stale_evidence: true,
                ..WanScenarioOutcomes::default()
            },
        ] {
            let error = wan_scenarios(impossible, 1, WAN_LIFECYCLE_BLOCKER)
                .expect_err("an executed scenario without a load is internally impossible");
            assert!(error.contains("without a completed load"), "{error}");
        }
    }

    /// The status law, mirroring the clauses of
    /// `memory-calibration-harness.mjs#validateRuntimeComplete` this arm can honour rather than
    /// restating a constant. Both verdicts are exercised, so this cannot pass by asserting a
    /// default the production path never reaches.
    ///
    /// The name is precise on purpose (sc-19057 review): the harness's third lifecycle shape —
    /// "fully passed" — additionally requires `cleanupVerified`/`warmFollowUpPassed` on `cancel` and
    /// `error`, and [`wan_scenarios`] emits neither, so this arm must NOT promote that shape. The
    /// assertions below pin that it stays `gated`, and that the emitter really lacks those fields.
    #[test]
    fn the_wan_receipt_status_mirrors_the_harness_law_for_the_shapes_this_arm_can_emit() {
        let admitted = WanScenarioOutcomes {
            loadability: true,
            exact_fit: true,
            unknown_budget: true,
            stale_evidence: true,
            warm_repeat: true,
            cancel: false,
            error: false,
        };
        // Parity-only lifecycle over a singleton domain — what this arm actually produces today.
        assert_eq!(wan_receipt_status(admitted, true), "runtime_complete");
        // Entirely-not_run lifecycle is also activation-eligible ...
        assert_eq!(
            wan_receipt_status(
                WanScenarioOutcomes {
                    warm_repeat: false,
                    ..admitted
                },
                true
            ),
            "runtime_complete"
        );
        // ... but a fully executed lifecycle is NOT, here. The harness would accept that shape only
        // with `cleanupVerified`/`warmFollowUpPassed` on cancel and error, and `wan_scenarios` has
        // no measurement to emit those from — so promoting it would produce a `runtime_complete`
        // the harness then rejects. Fail closed instead.
        assert_eq!(
            wan_receipt_status(
                WanScenarioOutcomes {
                    cancel: true,
                    error: true,
                    ..admitted
                },
                true
            ),
            "gated"
        );
        // The same for a half-executed lifecycle.
        assert_eq!(
            wan_receipt_status(
                WanScenarioOutcomes {
                    cancel: true,
                    ..admitted
                },
                true
            ),
            "gated"
        );
        // An unverified sweep range is not, because the harness refuses it.
        assert_eq!(wan_receipt_status(admitted, false), "gated");
        // And neither is a missing admission verdict.
        for missing in [
            WanScenarioOutcomes {
                exact_fit: false,
                ..admitted
            },
            WanScenarioOutcomes {
                unknown_budget: false,
                ..admitted
            },
            WanScenarioOutcomes {
                stale_evidence: false,
                ..admitted
            },
            WanScenarioOutcomes {
                loadability: false,
                ..admitted
            },
        ] {
            assert_eq!(wan_receipt_status(missing, true), "gated");
        }

        // The structural reason the "fully passed" clause is absent above: the emitter carries no
        // field to satisfy it with. If a later story adds lifecycle injection, this reds and the
        // branch and the emitter must be restored together.
        let scenarios = wan_scenarios(admitted, 1, WAN_LIFECYCLE_BLOCKER).unwrap();
        for name in ["cancel", "error"] {
            let scenario = scenarios
                .as_array()
                .and_then(|items| items.iter().find(|item| item["name"] == json!(name)))
                .unwrap_or_else(|| panic!("the {name} scenario must be emitted"));
            assert!(scenario.get("cleanupVerified").is_none(), "{scenario}");
            assert!(scenario.get("warmFollowUpPassed").is_none(), "{scenario}");
        }
    }

    /// The determinism envelope has to be falsifiable: the mandatory broad-bias mutation must breach
    /// all three thresholds, or the thresholds are measuring nothing. Computed over real pixel
    /// arithmetic rather than asserted.
    #[test]
    fn the_wan_determinism_envelope_admits_jitter_and_is_breached_by_the_mandatory_mutation() {
        let frame = runtime_cuda::gen_core::Image {
            width: 2,
            height: 1,
            pixels: vec![10, 40, 90, 200, 130, 60],
        };
        let clip = vec![frame.clone(), frame.clone()];

        // A bit-identical repeat is the expected observation.
        let (maximum, mean, rms) = wan_clip_max_mean_rms(&clip, &clip).unwrap();
        assert_eq!((maximum, mean, rms), (0.0, 0.0, 0.0));
        assert!(wan_quality_passes(maximum, mean, rms));

        // One channel of jitter at 2/255 stays inside the envelope, 4/255 does not.
        for (delta, admitted) in [(2_u8, true), (4, false)] {
            let mut jittered = frame.clone();
            jittered.pixels[0] += delta;
            let (maximum, mean, rms) =
                wan_clip_max_mean_rms(&clip, &[jittered.clone(), frame.clone()]).unwrap();
            assert_eq!(
                wan_quality_passes(maximum, mean, rms),
                admitted,
                "delta {delta}: max={maximum}"
            );
        }

        let mutated = clip.iter().map(wan_negative_mutation).collect::<Vec<_>>();
        let (maximum, mean, rms) = wan_clip_max_mean_rms(&mutated, &clip).unwrap();
        assert!(!wan_quality_passes(maximum, mean, rms));
        assert!(maximum > WAN_MAX_THRESHOLD, "max {maximum}");
        assert!(mean > WAN_MEAN_THRESHOLD, "mean {mean}");
        assert!(rms > WAN_RMS_THRESHOLD, "rms {rms}");

        // A clip whose frame count changed between renders is a comparison error, not a pass.
        assert!(wan_clip_max_mean_rms(&clip, std::slice::from_ref(&frame))
            .unwrap_err()
            .contains("frame-count mismatch"));
    }

    #[test]
    fn a_flat_first_frame_is_not_calibration_evidence() {
        assert!(!wan_frame_is_nondegenerate(
            &runtime_cuda::gen_core::Image {
                width: 2,
                height: 1,
                pixels: vec![7; 6],
            }
        ));
        assert!(!wan_frame_is_nondegenerate(
            &runtime_cuda::gen_core::Image::default()
        ));
        assert!(wan_frame_is_nondegenerate(&runtime_cuda::gen_core::Image {
            width: 2,
            height: 1,
            pixels: vec![7, 7, 7, 7, 7, 8],
        }));
    }

    /// And the guard is the frames axis rather than a blanket rejection: the same still geometry
    /// passes it on both Candle labels, so the refusals above cannot be an unconditional error.
    #[test]
    fn the_candle_still_geometry_guard_is_not_a_blanket_refusal() {
        for provider in [QWEN_ID, KREA_ID] {
            let request = json!({ "planned": still_planned_case(provider, "resident", 1) });
            let label = still_calibration_label(&request).unwrap();
            protocol::validate_still_geometry(&request, label)
                .unwrap_or_else(|error| panic!("{provider}: {error}"));
        }
    }
}
