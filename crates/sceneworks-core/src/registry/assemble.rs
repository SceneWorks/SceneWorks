//! Consume the current descriptor dumps. ENG-7 replaces this input adapter at its producer cutover.
use super::policy;
use super::spec::{
    BackendEvidence, GeneratorRoute, ModelSpec, ProductModel, ProviderRoute, VideoRoute,
};
use serde::Deserialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct DescriptorDump {
    schema_version: u32,
    model_mappings: BTreeMap<String, String>,
    video_model_mappings: Vec<VideoMapping>,
    snapshot: Snapshot,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct VideoMapping {
    model_id: String,
    mode: String,
    engine_ids: Vec<String>,
}

#[derive(Deserialize)]
struct Snapshot {
    backend: String,
    generator_capabilities: Vec<GeneratorRoute>,
    audio_generator_capabilities: Vec<GeneratorRoute>,
    captioner_ids: Vec<String>,
    image_embedder_ids: Vec<String>,
    text_embedder_ids: Vec<String>,
    text_llm_ids: Vec<String>,
    audio_voice_embedder_ids: Vec<String>,
    audio_transform_ids: Vec<String>,
    audio_transcriber_ids: Vec<String>,
    audio_embedder_ids: Vec<String>,
    transform_ids: Vec<String>,
}

impl Snapshot {
    fn generator(&self, id: &str, backend: &str) -> Option<&GeneratorRoute> {
        self.generator_capabilities
            .iter()
            .chain(&self.audio_generator_capabilities)
            .find(|route| route.id == id && route.backend == backend)
    }

    fn provider(&self, id: &str, backend: &str) -> Vec<ProviderRoute> {
        let mut routes = Vec::new();
        if let Some(generator) = self
            .audio_generator_capabilities
            .iter()
            .find(|route| route.id == id && route.backend == backend)
        {
            routes.push(ProviderRoute {
                kind: "audio_generator".to_owned(),
                id: id.to_owned(),
                generator: Some(generator.clone()),
            });
        }
        for (kind, ids) in [
            ("captioner", &self.captioner_ids),
            ("image_embedder", &self.image_embedder_ids),
            ("text_embedder", &self.text_embedder_ids),
            ("text_llm", &self.text_llm_ids),
            ("audio_voice_embedder", &self.audio_voice_embedder_ids),
            ("audio_transform", &self.audio_transform_ids),
            ("audio_transcriber", &self.audio_transcriber_ids),
            ("audio_embedder", &self.audio_embedder_ids),
            ("transform", &self.transform_ids),
        ] {
            if ids.iter().any(|candidate| candidate == id) {
                routes.push(ProviderRoute {
                    kind: kind.to_owned(),
                    id: id.to_owned(),
                    generator: None,
                });
            }
        }
        routes
    }
}

pub fn assemble(
    products: &str,
    evidence: &str,
    route_policy: &str,
    preview_features: &str,
    dumps: &[(&str, &str)],
) -> Result<BTreeMap<String, ModelSpec>, String> {
    let document: Value = serde_json::from_str(products).map_err(|e| e.to_string())?;
    let products = document
        .as_array()
        .or_else(|| document.get("models").and_then(Value::as_array))
        .ok_or("registry product document needs a models array")?;
    let route_policy = policy::parse(route_policy)?;
    let preview_document: Value =
        serde_json::from_str(preview_features).map_err(|e| e.to_string())?;
    if preview_document["version"] != 1 {
        return Err("unsupported preview feature export version".to_owned());
    }
    let preview: BTreeMap<String, BTreeMap<String, bool>> = serde_json::from_value(
        preview_document
            .get("models")
            .cloned()
            .ok_or("preview feature export has no models")?,
    )
    .map_err(|e| format!("invalid preview feature export: {e}"))?;
    if preview
        .values()
        .flat_map(|lanes| lanes.keys())
        .any(|backend| backend != "mlx" && backend != "candle")
    {
        return Err("unknown preview feature backend".to_owned());
    }
    let product_types: BTreeMap<_, _> = products
        .iter()
        .filter_map(|row| Some((row.get("id")?.as_str()?, row.get("type")?.as_str()?)))
        .collect();
    for (id, kind, legacy) in route_policy
        .image
        .iter()
        .map(|row| (row.id.as_str(), "image", row.legacy_alias))
        .chain(
            route_policy
                .video
                .iter()
                .map(|row| (row.id.as_str(), "video", row.legacy_alias)),
        )
    {
        if legacy != !product_types.contains_key(id)
            || product_types
                .get(id)
                .is_some_and(|product_kind| *product_kind != kind)
        {
            return Err(format!(
                "invalid legacy marker or modality for route policy {id}"
            ));
        }
    }
    let mut evidence: BTreeMap<String, Value> =
        serde_json::from_str(evidence).map_err(|e| e.to_string())?;
    let mut exports = BTreeMap::new();
    for &(backend, contents) in dumps {
        let dump: DescriptorDump = serde_json::from_str(contents).map_err(|e| e.to_string())?;
        if dump.schema_version != 2 || dump.snapshot.backend != backend {
            return Err(format!(
                "unsupported {backend} descriptor schema or backend"
            ));
        }
        if exports.insert(backend, dump).is_some() {
            return Err(format!("duplicate {backend} descriptor"));
        }
    }
    let mut specs = BTreeMap::new();
    for row in products {
        let mut product: ProductModel =
            serde_json::from_value(row.clone()).map_err(|e| format!("invalid product: {e}"))?;
        let id = product.id.clone();
        let downloads = product
            .fields
            .get("downloads")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("builtin {id} needs downloads"))?;
        if downloads.iter().any(|download| {
            download.get("provider").and_then(Value::as_str) != Some("huggingface")
                || download
                    .get("repo")
                    .and_then(Value::as_str)
                    .is_none_or(|repo| crate::hf_repo_id::namespaced(repo) != Ok(repo))
                || download
                    .get("revision")
                    .and_then(Value::as_str)
                    .is_some_and(|revision| {
                        revision.len() != 40
                            || !revision
                                .bytes()
                                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
                    })
        }) {
            return Err(format!(
                "builtin {id} needs valid download repositories and revisions"
            ));
        }
        let raw_evidence = evidence
            .remove(&id)
            .unwrap_or_else(|| Value::Object(Default::default()));
        let model_evidence = raw_evidence
            .get("model")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let download_evidence = raw_evidence
            .get("downloads")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|row| {
                row.as_object()
                    .cloned()
                    .ok_or_else(|| format!("invalid {id} download evidence"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut lanes = BTreeMap::new();
        for backend in ["mlx", "candle"] {
            if let Some(fields) = raw_evidence.get(backend) {
                let parsed: BackendEvidence = serde_json::from_value(fields.clone())
                    .map_err(|error| format!("invalid {id}/{backend} evidence: {error}"))?;
                lanes.insert(backend.to_owned(), parsed);
            }
        }
        let mut routes = BTreeMap::new();
        let mut video_routes = BTreeMap::new();
        let mut provider_routes = BTreeMap::new();
        for (&backend, dump) in &exports {
            if let Some(provider) = dump.model_mappings.get(&id) {
                if let Some(route) = dump.snapshot.generator(provider, backend) {
                    routes.insert(backend.to_owned(), route.clone());
                }
            }
            let mut video = Vec::new();
            for mapping in dump
                .video_model_mappings
                .iter()
                .filter(|mapping| mapping.model_id == id)
            {
                let mut engines = Vec::new();
                for engine in &mapping.engine_ids {
                    if let Some(route) = dump.snapshot.generator(engine, backend) {
                        engines.push(route.clone());
                    }
                }
                if !engines.is_empty() {
                    video.push(VideoRoute {
                        mode: mapping.mode.clone(),
                        engines,
                    });
                }
            }
            if !video.is_empty() {
                video_routes.insert(backend.to_owned(), video);
            }
            let providers = dump.snapshot.provider(&id, backend);
            let mut providers = providers;
            if let Some(vector) = product
                .fields
                .get("vector")
                .and_then(|vector| vector.get("providers"))
                .and_then(|providers| providers.get(backend))
            {
                let provider = vector
                    .get("id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| format!("invalid {id}/{backend} vector provider"))?;
                if vector.get("available").and_then(Value::as_bool) == Some(true) {
                    if !dump
                        .snapshot
                        .text_llm_ids
                        .iter()
                        .any(|candidate| candidate == provider)
                    {
                        return Err(format!(
                            "unregistered {id}/{backend} vector provider {provider}"
                        ));
                    }
                    providers.push(ProviderRoute {
                        kind: "text_llm".to_owned(),
                        id: provider.to_owned(),
                        generator: None,
                    });
                }
            }
            if !providers.is_empty() {
                provider_routes.insert(backend.to_owned(), providers);
            }
        }
        // Shared sampler menus cannot advertise a named method absent on a native image route.
        if product.model_type == "image" && !routes.is_empty() {
            if let Some(limits) = product
                .fields
                .get_mut("limits")
                .and_then(Value::as_object_mut)
            {
                for field in ["samplers", "schedulers"] {
                    if let Some(values) = limits.get_mut(field).and_then(Value::as_array_mut) {
                        values.retain(|value| {
                            value == "default"
                                || value.as_str().is_some_and(|name| {
                                    routes.values().all(|route| {
                                        let native = if field == "samplers" {
                                            &route.samplers
                                        } else {
                                            &route.schedulers
                                        };
                                        native.iter().any(|item| item == name)
                                    })
                                })
                        });
                    }
                }
            }
        }
        if specs
            .insert(
                id.clone(),
                ModelSpec {
                    product,
                    routing: route_policy.for_model(&id),
                    preview_by_backend: preview.get(&id).cloned().unwrap_or_default(),
                    routes,
                    video_routes,
                    provider_routes,
                    evidence: lanes,
                    model_evidence,
                    download_evidence,
                },
            )
            .is_some()
        {
            return Err(format!("duplicate product {id}"));
        }
    }
    if !evidence.is_empty() {
        let unknown: BTreeSet<_> = evidence.keys().collect();
        return Err(format!("orphan evidence rows: {unknown:?}"));
    }
    Ok(specs)
}
