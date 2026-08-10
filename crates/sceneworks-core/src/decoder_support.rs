//! Generated alternate-decoder support derived from the checked-in engine capability facts.
//!
//! The API process does not always link a media engine, so it cannot inspect descriptors at serve
//! time. The stage-1 facts files are the portable, pinned answer: the MLX dumper emits only decoder
//! options whose provider id, backend, and typed latent-space contract all agree. This module embeds
//! those facts and stamps the relevant options onto model catalog entries without mirroring an
//! eligibility list in SceneWorks.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use serde_json::Value;

const FACTS: &[&str] = &[
    include_str!("../../../config/engine-capabilities/capabilities.candle.json"),
    include_str!("../../../config/engine-capabilities/capabilities.mlx.json"),
];

pub const DECODERS_FIELD: &str = "decoders";
pub const BY_BACKEND_FIELD: &str = "byBackend";

type DecoderTable = BTreeMap<String, BTreeMap<String, Vec<Value>>>;

const KNOWN_BACKENDS: &[&str] = &["candle", "mlx"];
const IMPORTED_KREA_PROVIDER: &str = "krea_2_turbo";

static TABLE: OnceLock<Result<DecoderTable, String>> = OnceLock::new();

fn parse() -> Result<DecoderTable, String> {
    let mut table = DecoderTable::new();
    for contents in FACTS {
        let facts: Value = serde_json::from_str(contents)
            .map_err(|error| format!("engine capability facts are malformed: {error}"))?;
        let backend = facts
            .get("backend")
            .and_then(Value::as_str)
            .ok_or_else(|| "engine capability facts have no backend".to_owned())?;
        let engines = facts
            .get("engines")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("{backend} engine capability facts have no engines array"))?;
        for engine in engines {
            let Some(options) = engine.get("decoderOptions").and_then(Value::as_array) else {
                continue;
            };
            if options.is_empty() {
                continue;
            }
            let id = engine
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("{backend} decoder-capable engine has no id"))?;
            for option in options {
                for field in ["id", "label", "componentId", "licenseComponent"] {
                    if option.get(field).and_then(Value::as_str).is_none() {
                        return Err(format!(
                            "{backend} engine {id} decoder option is missing string field {field}"
                        ));
                    }
                }
                if option
                    .get("experimental")
                    .and_then(Value::as_bool)
                    .is_none()
                {
                    return Err(format!(
                        "{backend} engine {id} decoder option is missing boolean experimental"
                    ));
                }
            }
            table
                .entry(id.to_owned())
                .or_default()
                .insert(backend.to_owned(), options.clone());
        }
    }
    Ok(table)
}

fn table() -> Result<&'static DecoderTable, &'static str> {
    TABLE.get_or_init(parse).as_ref().map_err(String::as_str)
}

/// Resolve the registered provider whose decoder contract applies to one model row on `backend`.
///
/// Builtins resolve by their engine id or their existing memory-contract provider alias. Imported
/// Krea checkpoints are the one family-routed surface: their arbitrary catalog ids reuse the
/// registered Turbo provider for generation, so derive that same deterministic provider from the
/// existing route-by-family contract. Raw is deliberately not guessed from the family — imported
/// Krea generation has always paired its single-file DiT with the Turbo base/provider.
pub fn provider_id_for_backend(
    object: &serde_json::Map<String, Value>,
    backend: &str,
) -> Option<String> {
    let table = table().ok()?;
    let model_id = object.get("id").and_then(Value::as_str)?;
    if table
        .get(model_id)
        .and_then(|by_backend| by_backend.get(backend))
        .is_some_and(|options| !options.is_empty())
    {
        return Some(model_id.to_owned());
    }

    let declared_provider = object
        .get(backend)
        .and_then(Value::as_object)
        .and_then(|config| config.get("memoryStrategyContract"))
        .and_then(Value::as_object)
        .and_then(|contract| contract.get("provider"))
        .and_then(Value::as_str);
    if let Some(provider) = declared_provider.filter(|provider| {
        table
            .get(*provider)
            .and_then(|by_backend| by_backend.get(backend))
            .is_some_and(|options| !options.is_empty())
    }) {
        return Some(provider.to_owned());
    }

    let imported_krea = backend == "mlx"
        && object.get("type").and_then(Value::as_str) == Some("image")
        && object.get("family").and_then(Value::as_str) == Some("krea_2")
        && !crate::jobs_store::is_builtin_image_model(model_id)
        && crate::jobs_store::image_family_is_mlx_routed("krea_2");
    imported_krea
        .then_some(IMPORTED_KREA_PROVIDER)
        .filter(|provider| {
            table
                .get(*provider)
                .and_then(|by_backend| by_backend.get(backend))
                .is_some_and(|options| !options.is_empty())
        })
        .map(str::to_owned)
}

/// Component ids required by the compatible decoder options for a resolved model/provider row.
/// SceneWorks uses these ids to inherit only the optional decoder donor declarations from the
/// canonical builtin provider; unrelated soft co-requisites remain untouched.
pub fn component_ids_for_backend(
    object: &serde_json::Map<String, Value>,
    backend: &str,
) -> Vec<String> {
    let Ok(table) = table() else {
        return Vec::new();
    };
    let Some(provider) = provider_id_for_backend(object, backend) else {
        return Vec::new();
    };
    table
        .get(&provider)
        .and_then(|by_backend| by_backend.get(backend))
        .into_iter()
        .flatten()
        .filter_map(|option| option.get("componentId").and_then(Value::as_str))
        .map(str::to_owned)
        .collect()
}

/// Stamp `decoders.byBackend` onto a model entry when a dumped engine with the same catalog id
/// advertises one or more compatible choices. Absence means unsupported/unknown; it is never turned
/// into an invented empty capability.
pub fn apply_to_model_entry(object: &mut serde_json::Map<String, Value>) {
    let Ok(table) = table() else { return };
    let by_backend = KNOWN_BACKENDS
        .iter()
        .filter_map(|backend| {
            let provider = provider_id_for_backend(object, backend)?;
            let options = table.get(&provider)?.get(*backend)?;
            Some(((*backend).to_owned(), Value::Array(options.clone())))
        })
        .collect::<serde_json::Map<String, Value>>();
    if by_backend.is_empty() {
        return;
    }
    let mut decoders = serde_json::Map::new();
    decoders.insert(BY_BACKEND_FIELD.to_owned(), Value::Object(by_backend));
    object.insert(DECODERS_FIELD.to_owned(), Value::Object(decoders));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embeds_experimental_wan_decoder_only_for_dumped_compatible_mlx_engines() {
        let table = table().expect("decoder facts parse");
        for id in [
            "qwen_image",
            "qwen_image_control",
            "qwen_image_edit",
            "krea_2_turbo",
            "krea_2_raw",
            "krea_2_edit",
            "krea_2_turbo_edit",
            "krea_2_turbo_control",
        ] {
            let option = &table[id]["mlx"][0];
            assert_eq!(option["id"], "wan_2_1_vae", "{id}");
            assert_eq!(option["componentId"], "vae", "{id}");
            assert_eq!(option["experimental"], true, "{id}");
            assert!(table[id].get("candle").is_none(), "{id}");
        }
        assert!(
            table.get("wan2_2_ti2v_5b").is_none(),
            "z48 must fail closed"
        );
    }

    #[test]
    fn catalog_stamp_is_additive() {
        let mut model = serde_json::json!({ "id": "qwen_image", "name": "Qwen Image" });
        apply_to_model_entry(model.as_object_mut().unwrap());
        assert_eq!(
            model["decoders"]["byBackend"]["mlx"][0]["id"],
            "wan_2_1_vae"
        );

        let mut unknown = serde_json::json!({ "id": "external_model" });
        let before = unknown.clone();
        apply_to_model_entry(unknown.as_object_mut().unwrap());
        assert_eq!(unknown, before);

        let mut aliased_edit = serde_json::json!({
            "id": "qwen_image_edit_2511",
            "mlx": { "memoryStrategyContract": { "provider": "qwen_image_edit" } }
        });
        apply_to_model_entry(aliased_edit.as_object_mut().unwrap());
        assert_eq!(
            aliased_edit["decoders"]["byBackend"]["mlx"][0]["id"],
            "wan_2_1_vae"
        );

        let mut imported = serde_json::json!({
            "id": "user_kreamania_variant5",
            "type": "image",
            "family": "krea_2",
            "downloads": []
        });
        assert_eq!(
            provider_id_for_backend(imported.as_object().unwrap(), "mlx").as_deref(),
            Some("krea_2_turbo")
        );
        assert_eq!(
            component_ids_for_backend(imported.as_object().unwrap(), "mlx"),
            vec!["vae"]
        );
        apply_to_model_entry(imported.as_object_mut().unwrap());
        assert_eq!(
            imported["decoders"]["byBackend"]["mlx"][0]["id"],
            "wan_2_1_vae"
        );
        assert!(imported["decoders"]["byBackend"].get("candle").is_none());

        let mut imported_wrong_family = serde_json::json!({
            "id": "user_kreamania_variant5",
            "type": "image",
            "family": "z-image"
        });
        apply_to_model_entry(imported_wrong_family.as_object_mut().unwrap());
        assert!(imported_wrong_family.get("decoders").is_none());
    }
}
