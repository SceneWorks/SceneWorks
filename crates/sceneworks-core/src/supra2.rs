//! Native Supra2 artifact/provenance adapter; portable options owns request policy.
use crate::contracts::JsonObject;
use gen_core_checkpoint::supra2::{self as s2, Supra2Resolved};
pub use sceneworks_options::supra2::{is_supra2, max_steps};
use serde_json::Value;
pub fn validate_request(
    payload: &JsonObject,
    entry: &JsonObject,
) -> Result<Option<Supra2Resolved>, String> {
    sceneworks_options::supra2::validate_request(payload, entry).map(|resolved| {
        resolved.map(|value| Supra2Resolved {
            steps: value.steps,
            guidance: value.guidance,
            use_cfg: value.use_cfg,
            seed: value.seed,
            count: value.count,
        })
    })
}
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
/// `guidance > 1` (`gen_core_checkpoint::supra2::cfg_enabled`), and then uses the bundle's stored conditioning
/// when its manifest records one, else the empty-prompt fallback. The engine's own
/// `used_stored_unconditional` flag stays inside the provider (the `Generator` report does not carry
/// it), so the worker reads the same fact off the manifest of `bundle` — the confined directory the
/// render LOADED (`resolve_weights_dir`), never a path re-read off the request. `Null` when the
/// manifest cannot be read (that render fails at load anyway).
pub fn unconditional_source(bundle: Option<&std::path::Path>, guidance: Option<f32>) -> Value {
    if !s2::cfg_enabled(guidance.unwrap_or(s2::DEFAULT_GUIDANCE)) {
        return Value::from(UNCONDITIONAL_NONE);
    }
    let manifest = bundle
        .and_then(|dir| std::fs::read(dir.join(s2::BUNDLE_MANIFEST)).ok())
        .and_then(|bytes| serde_json::from_slice::<s2::BundleManifest>(&bytes).ok());
    match manifest {
        Some(manifest) if manifest.conditioning.is_some() => Value::from(UNCONDITIONAL_STORED),
        Some(_) => Value::from(UNCONDITIONAL_EMPTY_PROMPT),
        None => Value::Null,
    }
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
        let bundle = Some(dir.path());
        let write = |m: &s2::BundleManifest| {
            std::fs::write(
                dir.path().join(s2::BUNDLE_MANIFEST),
                serde_json::to_vec(m).unwrap(),
            )
            .unwrap()
        };
        assert_eq!(unconditional_source(bundle, Some(3.0)), Value::Null);
        write(&manifest(Some(file(s2::CONDITIONING_FILE))));
        assert_eq!(unconditional_source(bundle, Some(3.0)), json!("stored"));
        assert_eq!(unconditional_source(bundle, None), json!("stored"));
        assert_eq!(unconditional_source(bundle, Some(1.0)), json!("none"));
        assert_eq!(unconditional_source(bundle, Some(0.5)), json!("none"));
        write(&manifest(None));
        assert_eq!(
            unconditional_source(bundle, Some(3.0)),
            json!("emptyPrompt")
        );
        assert_eq!(unconditional_source(None, Some(1.0)), json!("none"));
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
            // sc-24217 pre-merge review: a caller-chosen weights dir would load a bundle the
            // diagnosis and the provenance record do not describe.
            (
                json!({ "advanced": { "modelPath": "/elsewhere/supra2_img" } }),
                "advanced.modelPath",
            ),
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
