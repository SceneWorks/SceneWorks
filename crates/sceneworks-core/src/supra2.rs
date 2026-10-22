//! Supra2-IMG request boundary (sc-24215, epic 24212): the one place the API and the worker refuse
//! what the published Supra2 interface does not have, BEFORE a job exists (API) or a model loads
//! (worker).
//!
//! The published interface is text-to-image at 256x256 with prompt, seed, guidance, steps and image
//! count. The backend-neutral rules (native size, `steps >= 1`, finite guidance, at most
//! [`gen_core::supra2::MAX_IMAGES`] images, no negative prompt / prompt batch / conditioning) are
//! inference's own `Supra2Resolved::resolve`, run here on the request the worker would build.
//! SceneWorks adds only what the engine never sees because the generic lane would drop or rewrite
//! it first: an edit/reference mode, a reference/mask/control carrier, LoRAs, a quantization tier, a
//! hires-fix pass, a sampler/scheduler/guidance-method choice and every other lane knob the engine
//! has no input for, plus the lane's step ceiling (the worker clamps above it rather than refusing).
use crate::contracts::JsonObject;
use gen_core::supra2::{self as s2, Supra2Resolved};
use gen_core::GenerationRequest;
use serde_json::Value;

/// Is `model` the Supra2-IMG catalog id?
pub fn is_supra2(model: &str) -> bool {
    model == s2::PROVIDER_ID
}

/// `advanced` knobs other lanes honor and the Supra2 engine has no input for. A meaningful value is
/// refused by name rather than dropped, so a recipe or headless call never renders something other
/// than what it asked for.
const UNSUPPORTED_ADVANCED: &[&str] = &[
    "schedulerShift",
    // The legacy alias the generic lane still reads into the schedule shift (`base.rs`).
    "timestepShift",
    "cfgInterval",
    "captionOverflow",
    "enhancePrompt",
    "mlxQuantize",
    "quantTier",
    "convRot",
    "usePid",
    "pidTarget",
    "strength",
    "ipAdapterScale",
    // The InstantID identity-structure slider's WIRE key (`imageJobAdvanced.js` /
    // `CharacterAdvancedOptions.jsx`); `identityStructure` / `controlnetScale` /
    // `variationStrength` are studio UI-state names that never reach a request.
    "controlnetConditioningScale",
    "trueCfgScale",
    "textStyleGain",
    "viewAngle",
    "viewAngles",
    "poses",
    "controlMode",
    "controlImage",
    "controlScale",
    "controlWeights",
    "phases",
    "transparentBackground",
    "textEncoderModel",
    // The Pose Library's face-restoration pass (sent with `poses`).
    "faceRestore",
    // The tier-pick marker that rides with `mlxQuantize`.
    "mlxQuantizeExplicit",
    // MOSS-Image's own wire objects (`imageJobAdvanced.js` / the MCP image tool).
    "mossImage",
    "mossOrdinarySettings",
    // The character lane's Angle Set / Key Point Library extras (`characterPanels.jsx`).
    "angleSet",
    "keypointCollectionId",
];

/// Top-level carriers of a source, reference, mask, control, adapter or prompt batch.
const UNSUPPORTED_CARRIERS: &[&str] = &[
    "sourceAssetId",
    "referenceAssetId",
    "referenceAssetIds",
    "maskAssetId",
    "controls",
    "controlnets",
    "loras",
    "prompts",
    // A character identity (the studio sends these on `character_image`; the MCP image tool
    // forwards `characterId` with any mode). `fitMode` is NOT here: the API stamps its default on
    // every image job, and it only shapes an edit source, which text-to-image has none of.
    "characterId",
    "characterLookId",
];

/// A value that asks for something: not `null`, `""`, `[]`, `{}` or `false`.
fn meaningful(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::String(value) => !value.trim().is_empty(),
        Value::Array(values) => !values.is_empty(),
        Value::Object(object) => !object.is_empty(),
        Value::Number(_) => true,
    }
}

/// A JSON number or numeric string (the worker's own `advanced` reading); `Err` for any other shape.
fn number(object: &JsonObject, key: &str) -> Result<Option<f64>, String> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_f64()
            .or_else(|| value.as_str()?.trim().parse().ok())
            .filter(|value| value.is_finite())
            .map(Some)
            .ok_or_else(|| format!("advanced.{key} must be a finite number, got {value}")),
    }
}

/// An optional menu choice that is only acceptable as the engine's own (or the `default` alias).
fn menu_choice(advanced: &JsonObject, key: &str, accepted: &[&str]) -> Result<(), String> {
    match advanced.get(key) {
        None | Some(Value::Null) => Ok(()),
        Some(Value::String(value))
            if value.trim().is_empty()
                || value == "default"
                || accepted.contains(&value.as_str()) =>
        {
            Ok(())
        }
        Some(value) => Err(format!(
            "advanced.{key} {value} is not offered (Supra2-IMG runs {} only)",
            accepted.join(" / ")
        )),
    }
}

/// Worker-owned `rawAdapterSettings` key: the pinned `{repo, revision, componentId?}` of every
/// component the render loaded (denoiser checkpoint, text encoder, VAE), so each asset names the
/// exact weights that produced it.
pub const MODEL_REVISIONS_KEY: &str = "modelRevisions";

/// The pinned download rows of `entry` as `[{repo, revision, componentId?}]` (catalog order). Every
/// Supra2 row loads on every render, so the list is the asset's full weight provenance.
pub fn pinned_revisions(entry: &JsonObject) -> Value {
    let rows = entry
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|row| {
            let repo = row.get("repo").and_then(Value::as_str)?;
            let revision = row.get("revision").and_then(Value::as_str)?;
            let mut pin = serde_json::Map::new();
            pin.insert("repo".to_owned(), Value::from(repo));
            pin.insert("revision".to_owned(), Value::from(revision));
            if let Some(component) = row.get("componentId").and_then(Value::as_str) {
                pin.insert("componentId".to_owned(), Value::from(component));
            }
            Some(Value::Object(pin))
        })
        .collect();
    Value::Array(rows)
}

/// Worker-owned `rawAdapterSettings` key: which unconditional context classifier-free guidance
/// steered against — [`UNCONDITIONAL_STORED`], [`UNCONDITIONAL_EMPTY_PROMPT`] or
/// [`UNCONDITIONAL_NONE`] (see [`unconditional_source`]).
pub const UNCONDITIONAL_KEY: &str = "unconditional";
/// CFG used the checkpoint-stored unconditional embedding (the pinned bundle's conditioning file).
pub const UNCONDITIONAL_STORED: &str = "stored";
/// CFG used upstream's fallback, the empty-prompt encoding (a bundle that stores no conditioning;
/// the pin contract refuses such a bundle at load, so a published asset never records it).
pub const UNCONDITIONAL_EMPTY_PROMPT: &str = "emptyPrompt";
/// Guidance `<= 1.0`: upstream's unguided branch evaluates no unconditional context at all.
pub const UNCONDITIONAL_NONE: &str = "none";
/// Worker-owned `rawAdapterSettings` key: the backend that rendered the asset (`mlx` / `candle`).
pub const BACKEND_KEY: &str = "backend";

/// Which unconditional context a render at `guidance` (the resolved value; `None` ⇒ the release
/// default) steers against, derived deterministically the way the engine decides it: CFG runs iff
/// `guidance > 1` (`gen_core::supra2::cfg_enabled`), and then uses the bundle's stored conditioning
/// when its manifest records one, else the empty-prompt fallback. The engine's own
/// `used_stored_unconditional` flag stays inside the provider (the `Generator` report does not carry
/// it), so the worker reads the same fact off the converted bundle's manifest at the entry's
/// `modelPath`. `Null` when the manifest cannot be read (that render fails at load anyway).
pub fn unconditional_source(entry: &JsonObject, guidance: Option<f32>) -> Value {
    if !s2::cfg_enabled(guidance.unwrap_or(s2::DEFAULT_GUIDANCE)) {
        return Value::from(UNCONDITIONAL_NONE);
    }
    let manifest = entry
        .get("modelPath")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|path| !path.is_empty())
        .and_then(|dir| std::fs::read(std::path::Path::new(dir).join(s2::BUNDLE_MANIFEST)).ok())
        .and_then(|bytes| serde_json::from_slice::<s2::BundleManifest>(&bytes).ok());
    match manifest {
        Some(manifest) if manifest.conditioning.is_some() => Value::from(UNCONDITIONAL_STORED),
        Some(_) => Value::from(UNCONDITIONAL_EMPTY_PROMPT),
        None => Value::Null,
    }
}

/// The largest `advanced.steps` Supra2 accepts: the entry's declared `limits.hardMaxSteps` (the
/// shipped row declares the provider's own sanity cap, so the published interface is not narrowed),
/// else the generic lane's clamp. A larger request is refused, never clamped.
pub fn max_steps(entry: &JsonObject) -> u32 {
    crate::video_request::hard_max_steps(entry)
        .unwrap_or_else(|| s2::DEFAULT_STEPS.max(crate::image_request::BASE_MAX_IMAGE_STEPS))
}

/// Validate the FINAL (post-preset, post-merge) image payload. `Ok(None)` for any other model;
/// `Ok(Some(resolved))` with the published defaults applied; `Err(message)` naming the first
/// unsupported input. Width, height and count fall back to `entry`'s declared defaults when the
/// payload does not name them yet (the API fills them later from the same keys).
pub fn validate_request(
    payload: &JsonObject,
    entry: &JsonObject,
) -> Result<Option<Supra2Resolved>, String> {
    let model = payload.get("model").and_then(Value::as_str).unwrap_or("");
    if !is_supra2(model) {
        return Ok(None);
    }
    let named = |message: String| format!("{model}: {message}");
    match payload.get("mode") {
        None | Some(Value::Null) => {}
        Some(Value::String(mode)) if mode.is_empty() || mode == "text_to_image" => {}
        Some(mode) => {
            return Err(named(format!(
                "mode {mode} is not offered; Supra2-IMG is text-to-image only"
            )))
        }
    }
    if let Some(key) = UNSUPPORTED_CARRIERS
        .iter()
        .find(|key| payload.get(**key).is_some_and(meaningful))
    {
        return Err(named(format!(
            "`{key}` is not supported; Supra2-IMG renders from one text prompt only"
        )));
    }
    let negative = payload
        .get("negativePrompt")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or_default();
    if !negative.is_empty() {
        return Err(named(
            "a negative prompt is not supported (guidance steers against the checkpoint's own \
             unconditional embedding); clear it"
                .to_owned(),
        ));
    }
    let hires = payload
        .get("hiresFix")
        .and_then(Value::as_object)
        .and_then(|hires| hires.get("enabled"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if hires {
        return Err(named(
            "hires fix is not supported; Supra2-IMG renders at its native 256x256 only".to_owned(),
        ));
    }
    let empty = JsonObject::new();
    let advanced = match payload.get("advanced") {
        None | Some(Value::Null) => &empty,
        Some(Value::Object(advanced)) => advanced,
        Some(_) => return Err(named("advanced must be an object".to_owned())),
    };
    if let Some(key) = UNSUPPORTED_ADVANCED
        .iter()
        .find(|key| advanced.get(**key).is_some_and(meaningful))
    {
        return Err(named(format!(
            "advanced.{key} is not supported (Supra2-IMG takes prompt, seed, guidance, steps and \
             image count only)"
        )));
    }
    menu_choice(advanced, "sampler", &[s2::SAMPLER])
        .and_then(|()| menu_choice(advanced, "scheduler", &[s2::SCHEDULER]))
        .and_then(|()| menu_choice(advanced, "guidanceMethod", &["cfg"]))
        .and_then(|()| menu_choice(advanced, "decoder", &["native"]))
        .map_err(named)?;
    let steps = number(advanced, "steps").map_err(named)?;
    if let Some(steps) = steps {
        if steps.fract() != 0.0 || steps < 1.0 || steps > f64::from(max_steps(entry)) {
            return Err(named(format!(
                "advanced.steps {steps} is outside 1..={}",
                max_steps(entry)
            )));
        }
    }
    let guidance = number(advanced, "guidanceScale").map_err(named)?;
    let (default_width, default_height) =
        crate::image_request::default_resolution(entry).unwrap_or((s2::IMAGE_SIZE, s2::IMAGE_SIZE));
    let dimension = |key: &str, default: u32| match payload.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(value) => value
            .as_u64()
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| named(format!("{key} must be a positive integer, got {value}"))),
    };
    let width = dimension("width", default_width)?;
    let height = dimension("height", default_height)?;
    let count = match payload.get("count") {
        None | Some(Value::Null) => crate::image_request::default_count(entry).unwrap_or(1),
        Some(value) => value
            .as_u64()
            .and_then(|value| u32::try_from(value).ok())
            .filter(|count| *count >= 1)
            .ok_or_else(|| named(format!("count must be a positive integer, got {value}")))?,
    };
    // The engine renders up to `MAX_IMAGES` (64) per call, but a SceneWorks job is capped at
    // `image_request::MAX_COUNT` (8): `ImageRequest::from_payload` CLAMPS above it, so a larger
    // count (e.g. through retry/duplicate, which skips the typed route's 1..=8 check) would
    // silently render fewer images than asked. Refuse it here instead.
    if count > crate::image_request::MAX_COUNT {
        return Err(named(format!(
            "count {count} exceeds the {} images a job renders (the engine allows up to {} per \
             call, SceneWorks caps a job at {})",
            crate::image_request::MAX_COUNT,
            s2::MAX_IMAGES,
            crate::image_request::MAX_COUNT
        )));
    }
    let request = GenerationRequest {
        prompt: payload
            .get("prompt")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        width,
        height,
        count,
        steps: steps.map(|steps| steps as u32),
        guidance: guidance.map(|guidance| guidance as f32),
        ..Default::default()
    };
    Supra2Resolved::resolve(&request)
        .map(Some)
        .map_err(|error| named(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn object(value: Value) -> JsonObject {
        value.as_object().cloned().expect("object literal")
    }

    fn entry() -> JsonObject {
        object(json!({
            "id": "supra2_img",
            "defaults": { "resolution": "256x256", "steps": 50, "guidanceScale": 3.0, "count": 1 },
            "image": { "supportsNegativePrompt": false, "nativeCountBatch": true },
            "limits": { "hardMaxSteps": 100000 }
        }))
    }

    fn validate(payload: Value) -> Result<Option<Supra2Resolved>, String> {
        let mut payload = object(payload);
        payload
            .entry("model".to_owned())
            .or_insert(json!("supra2_img"));
        validate_request(&payload, &entry())
    }

    #[test]
    fn pinned_revisions_name_every_loaded_component() {
        let entry = object(json!({ "downloads": [
            { "repo": "SupraLabs/Supra2-IMG", "revision": "r1", "default": true },
            { "repo": "google/flan-t5-base", "revision": "r2", "componentId": "text_encoder" },
            { "repo": "unpinned" }
        ]}));
        assert_eq!(
            pinned_revisions(&entry),
            json!([
                { "repo": "SupraLabs/Supra2-IMG", "revision": "r1" },
                { "repo": "google/flan-t5-base", "revision": "r2", "componentId": "text_encoder" }
            ])
        );
        assert_eq!(pinned_revisions(&JsonObject::new()), json!([]));
    }

    /// The studio's off-state for the keys it always spells (a `false` toggle, a `null` id) is not
    /// a request for the knob, so it renders.
    #[test]
    fn studio_off_states_are_accepted() {
        assert!(validate(json!({
            "prompt": "a fox", "characterId": null, "characterLookId": null,
            "advanced": { "faceRestore": false, "flashAttn": false, "resolution": "256x256" }
        }))
        .is_ok());
    }

    /// CFG <= 1 evaluates no unconditional context; CFG > 1 uses the bundle's stored conditioning
    /// when its manifest records one, else the empty-prompt fallback; no readable manifest ⇒ null.
    #[test]
    fn the_unconditional_source_follows_guidance_and_the_bundle_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let file = |path: &str| s2::BundleFile {
            file: path.to_owned(),
            bytes: 1,
            sha256: "0".repeat(64),
        };
        let manifest = |conditioning: Option<s2::BundleFile>| s2::BundleManifest {
            schema: s2::BUNDLE_SCHEMA.to_owned(),
            provider_id: s2::PROVIDER_ID.to_owned(),
            source: s2::BundleSource {
                repo: None,
                revision: None,
                file: "model_final_ema.pt".to_owned(),
                bytes: 1,
                sha256: "0".repeat(64),
                weights_key: "ema".to_owned(),
            },
            config: s2::Supra2Config::published(),
            vae_scale: s2::VAE_SCALE,
            denoiser: file(s2::DENOISER_FILE),
            denoiser_tensors: 1,
            conditioning,
            components: Vec::new(),
        };
        let entry = object(json!({ "modelPath": dir.path() }));
        let write = |m: &s2::BundleManifest| {
            std::fs::write(
                dir.path().join(s2::BUNDLE_MANIFEST),
                serde_json::to_vec(m).unwrap(),
            )
            .unwrap()
        };
        assert_eq!(unconditional_source(&entry, Some(3.0)), Value::Null);
        write(&manifest(Some(file(s2::CONDITIONING_FILE))));
        assert_eq!(unconditional_source(&entry, Some(3.0)), json!("stored"));
        assert_eq!(unconditional_source(&entry, None), json!("stored"));
        assert_eq!(unconditional_source(&entry, Some(1.0)), json!("none"));
        assert_eq!(unconditional_source(&entry, Some(0.5)), json!("none"));
        write(&manifest(None));
        assert_eq!(
            unconditional_source(&entry, Some(3.0)),
            json!("emptyPrompt")
        );
        assert_eq!(
            unconditional_source(&JsonObject::new(), Some(1.0)),
            json!("none")
        );
    }

    #[test]
    fn other_models_are_untouched() {
        let payload = object(json!({ "model": "flux_schnell", "negativePrompt": "x", "width": 9 }));
        assert_eq!(validate_request(&payload, &entry()), Ok(None));
    }

    #[test]
    fn the_published_controls_resolve_with_the_release_defaults() {
        let resolved = validate(json!({ "prompt": "a fox" })).unwrap().unwrap();
        assert_eq!(
            (resolved.steps, resolved.guidance, resolved.count),
            (s2::DEFAULT_STEPS, s2::DEFAULT_GUIDANCE, 1)
        );
        assert!(resolved.use_cfg);
        let resolved = validate(json!({
            "prompt": "a fox", "width": 256, "height": 256, "count": 8, "seed": 7,
            "mode": "text_to_image",
            "advanced": { "steps": "12", "guidanceScale": 1.0, "resolution": "256x256",
                          "sampler": "default", "guidanceMethod": "cfg", "flashAttn": false,
                          "styleId": "s", "stylePrompt": "" },
            "negativePrompt": "  ", "loras": [], "referenceAssetIds": [],
            "hiresFix": { "enabled": false }
        }))
        .unwrap()
        .unwrap();
        assert_eq!((resolved.steps, resolved.count), (12, 8));
        // CFG <= 1 is upstream's unguided branch, not an error.
        assert!(!resolved.use_cfg);
        assert_eq!(max_steps(&entry()), 100_000);
        let at_max = validate(json!({ "advanced": { "steps": 100_000 } }));
        assert_eq!(
            at_max.unwrap().unwrap().steps,
            100_000,
            "the ceiling itself renders"
        );
        // An entry declaring no ceiling keeps the generic lane's limit.
        assert_eq!(max_steps(&JsonObject::new()), 80);
    }

    #[test]
    fn every_unsupported_input_is_refused_by_name() {
        for (payload, needle) in [
            (json!({ "width": 512, "height": 512 }), "256x256"),
            (json!({ "width": 256, "height": 384 }), "256x256"),
            (json!({ "count": 9 }), "exceeds the 8 images"),
            (json!({ "count": 65 }), "exceeds the 8 images"),
            (
                json!({ "advanced": { "timestepShift": 3 } }),
                "timestepShift",
            ),
            (json!({ "count": 0 }), "count"),
            (json!({ "advanced": { "steps": 0 } }), "advanced.steps"),
            (json!({ "advanced": { "steps": 100_001 } }), "1..=100000"),
            (json!({ "advanced": { "steps": 2.5 } }), "advanced.steps"),
            (
                json!({ "advanced": { "guidanceScale": "x" } }),
                "guidanceScale",
            ),
            (json!({ "negativePrompt": "blurry" }), "negative prompt"),
            (
                json!({ "mode": "edit_image", "sourceAssetId": "a" }),
                "text-to-image",
            ),
            (json!({ "mode": "character_image" }), "text-to-image"),
            (json!({ "sourceAssetId": "a" }), "sourceAssetId"),
            (json!({ "referenceAssetId": "a" }), "referenceAssetId"),
            (json!({ "referenceAssetIds": ["a"] }), "referenceAssetIds"),
            (json!({ "maskAssetId": "a" }), "maskAssetId"),
            (json!({ "controls": [{}] }), "controls"),
            (json!({ "loras": [{ "id": "l" }] }), "loras"),
            (json!({ "prompts": ["a", "b"] }), "prompts"),
            (json!({ "hiresFix": { "enabled": true } }), "hires fix"),
            (json!({ "advanced": { "sampler": "dpmpp_2m" } }), "sampler"),
            (
                json!({ "advanced": { "scheduler": "karras" } }),
                "scheduler",
            ),
            (
                json!({ "advanced": { "guidanceMethod": "apg" } }),
                "guidanceMethod",
            ),
            (json!({ "advanced": { "decoder": "pid" } }), "decoder"),
            (json!({ "advanced": { "mlxQuantize": 4 } }), "mlxQuantize"),
            (json!({ "advanced": { "quantTier": "q8" } }), "quantTier"),
            (
                json!({ "advanced": { "schedulerShift": 3 } }),
                "schedulerShift",
            ),
            (json!({ "advanced": { "poses": [{}] } }), "poses"),
            (
                json!({ "advanced": { "controlImage": "a" } }),
                "controlImage",
            ),
            (json!({ "advanced": { "usePid": true } }), "usePid"),
            // The studio's WIRE names (epic-review E4): the identity-structure slider and the pose
            // library's face restoration, as `imageJobAdvanced.js` sends them.
            (
                json!({ "advanced": { "controlnetConditioningScale": 0.8 } }),
                "controlnetConditioningScale",
            ),
            (
                json!({ "advanced": { "faceRestore": true } }),
                "faceRestore",
            ),
            (
                json!({ "advanced": { "mlxQuantizeExplicit": true } }),
                "mlxQuantizeExplicit",
            ),
            (
                json!({ "advanced": { "mossImage": { "acceleration": "off" } } }),
                "mossImage",
            ),
            (json!({ "advanced": { "angleSet": true } }), "angleSet"),
            (json!({ "characterId": "c1" }), "characterId"),
            (json!({ "characterLookId": "l1" }), "characterLookId"),
            (json!({ "advanced": "x" }), "advanced must be an object"),
        ] {
            let error = validate(payload.clone()).expect_err(&payload.to_string());
            assert!(
                error.starts_with("supra2_img: ") && error.contains(needle),
                "{payload}: {error}"
            );
        }
    }
}
