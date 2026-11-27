//! APP-1 static layer: assemble typed [`ModelSpec`] records from the v2 product manifest,
//! measured evidence store, router policy, preview support, and canonical `sga.capabilities/2`
//! hardware-free exports (`SW-6983`, `REG-3`, `HOOK-4`).
use super::policy;
use super::route_bindings;
use super::spec::{
    BackendEvidence, GeneratorRoute, ModelSpec, PagRouteCapability, PagSpec, ProductModel,
    ProviderRoute, VideoRoute,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

const SGA_SCHEMA_VERSION: &str = "sga.capabilities/2";

#[derive(Deserialize)]
struct SgaCapabilitiesExport {
    schema_version: String,
    backend: String,
    #[serde(default)]
    routes: BTreeMap<String, SgaRoute>,
    #[serde(default)]
    audio_routes: BTreeMap<String, SgaRoute>,
    #[serde(default)]
    captioners: BTreeMap<String, Value>,
    #[serde(default)]
    embedders: SgaEmbedders,
    #[serde(default)]
    llm_models: BTreeMap<String, Value>,
    #[serde(default)]
    transcribers: BTreeMap<String, Value>,
    #[serde(default)]
    transforms: SgaTransforms,
}

#[derive(Default, Deserialize)]
struct SgaEmbedders {
    #[serde(default)]
    image: BTreeMap<String, Value>,
    #[serde(default)]
    text: BTreeMap<String, Value>,
    #[serde(default)]
    voice: BTreeMap<String, Value>,
    #[serde(default)]
    audio: BTreeMap<String, Value>,
}

#[derive(Default, Deserialize)]
struct SgaTransforms {
    #[serde(default)]
    image: BTreeMap<String, Value>,
    #[serde(default)]
    audio: BTreeMap<String, Value>,
}

#[derive(Deserialize)]
struct SgaRoute {
    route_id: String,
    family: String,
    backend: String,
    modality: String,
    #[serde(default)]
    solvers: Vec<String>,
    #[serde(default)]
    schedules: Vec<String>,
    guidance: SgaGuidance,
    conditioning: SgaConditioning,
    adapters: SgaAdapters,
    quant: SgaQuant,
    #[serde(default)]
    features: Vec<SgaFeatureApplicability>,
    capabilities: SgaCapabilities,
}

#[derive(Deserialize)]
struct SgaGuidance {
    #[serde(default)]
    methods: Vec<String>,
    #[serde(default)]
    pag: Option<PagRouteCapability>,
}

#[derive(Deserialize)]
struct SgaConditioning {
    #[serde(default)]
    kinds: Vec<String>,
}

#[derive(Deserialize)]
struct SgaAdapters {
    #[serde(default)]
    supports_lora: bool,
    #[serde(default)]
    supports_lokr: bool,
}

#[derive(Deserialize)]
struct SgaQuant {
    #[serde(default)]
    supported_tiers: Vec<String>,
}

#[derive(Deserialize)]
struct SgaCapabilities {
    #[serde(default)]
    supports_preview: bool,
    #[serde(default)]
    supports_prompt_enhancement: bool,
}

#[derive(Clone, Deserialize, Serialize)]
struct SgaFeatureApplicability {
    feature_id: String,
    version: u32,
    class: String,
    applicable: bool,
    parameter_schema: Value,
    relations: Value,
    hook_points: Vec<String>,
    memory_effect: Value,
    reasons: Vec<Value>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    conditioned_routes: BTreeMap<String, Value>,
}

impl SgaRoute {
    fn to_generator_route(&self) -> GeneratorRoute {
        let mut features = BTreeMap::new();
        for feature in &self.features {
            if feature.feature_id == "low_res_start" && feature.applicable {
                let value = serde_json::to_value(feature)
                    .expect("serializable ExportedFeatureApplicabilityV2");
                features.insert(feature.feature_id.clone(), value);
            }
        }
        GeneratorRoute {
            id: self.route_id.clone(),
            family: self.family.clone(),
            backend: self.backend.clone(),
            modality: self.modality.clone(),
            conditioning: self.conditioning.kinds.clone(),
            supported_quants: self.quant.supported_tiers.clone(),
            supports_lora: self.adapters.supports_lora,
            supports_lokr: self.adapters.supports_lokr,
            supports_preview: self.capabilities.supports_preview,
            supports_prompt_enhancement: self.capabilities.supports_prompt_enhancement,
            samplers: self.solvers.clone(),
            schedulers: self.schedules.clone(),
            supported_guidance_methods: self.guidance.methods.clone(),
            pag: self.guidance.pag.clone(),
            features,
        }
    }
}

struct ParsedExport {
    backend: String,
    routes: BTreeMap<String, GeneratorRoute>,
    audio_routes: BTreeMap<String, GeneratorRoute>,
    captioner_ids: BTreeSet<String>,
    image_embedder_ids: BTreeSet<String>,
    text_embedder_ids: BTreeSet<String>,
    text_llm_ids: BTreeSet<String>,
    audio_voice_embedder_ids: BTreeSet<String>,
    audio_transform_ids: BTreeSet<String>,
    audio_transcriber_ids: BTreeSet<String>,
    audio_embedder_ids: BTreeSet<String>,
    transform_ids: BTreeSet<String>,
}

impl ParsedExport {
    fn from_sga(expected_backend: &str, raw: SgaCapabilitiesExport) -> Result<Self, String> {
        if raw.schema_version != SGA_SCHEMA_VERSION || raw.backend != expected_backend {
            return Err(format!(
                "unsupported {expected_backend} sga.capabilities/2 export schema ({:?}) or backend ({:?})",
                raw.schema_version, raw.backend
            ));
        }
        let routes = raw
            .routes
            .into_iter()
            .map(|(id, route)| (id, route.to_generator_route()))
            .collect();
        let audio_routes = raw
            .audio_routes
            .into_iter()
            .map(|(id, route)| (id, route.to_generator_route()))
            .collect();
        Ok(Self {
            backend: raw.backend,
            routes,
            audio_routes,
            captioner_ids: raw.captioners.into_keys().collect(),
            image_embedder_ids: raw.embedders.image.into_keys().collect(),
            text_embedder_ids: raw.embedders.text.into_keys().collect(),
            text_llm_ids: raw.llm_models.into_keys().collect(),
            audio_voice_embedder_ids: raw.embedders.voice.into_keys().collect(),
            audio_transform_ids: raw.transforms.audio.into_keys().collect(),
            audio_transcriber_ids: raw.transcribers.into_keys().collect(),
            audio_embedder_ids: raw.embedders.audio.into_keys().collect(),
            transform_ids: raw.transforms.image.into_keys().collect(),
        })
    }

    fn generator(&self, id: &str, backend: &str) -> Option<&GeneratorRoute> {
        self.routes
            .get(id)
            .filter(|route| route.backend == backend)
            .or_else(|| {
                self.audio_routes
                    .get(id)
                    .filter(|route| route.backend == backend)
            })
    }

    fn provider(&self, id: &str, backend: &str) -> Vec<ProviderRoute> {
        let mut routes = Vec::new();
        if let Some(generator) = self
            .audio_routes
            .get(id)
            .filter(|route| route.backend == backend)
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
            if ids.contains(id) {
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
        let raw: SgaCapabilitiesExport =
            serde_json::from_str(contents).map_err(|e| e.to_string())?;
        let parsed = ParsedExport::from_sga(backend, raw)?;
        if exports.insert(backend, parsed).is_some() {
            return Err(format!("duplicate {backend} sga.capabilities/2 export"));
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
        let depth_evidence = raw_evidence
            .get("depth")
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
        for (&backend, export) in &exports {
            if product.model_type == "image" {
                if let Some(provider) = route_bindings::image_route_id(&id, |candidate| {
                    export.routes.contains_key(candidate)
                }) {
                    if let Some(route) = export.generator(provider, backend) {
                        routes.insert(backend.to_owned(), route.clone());
                    }
                }
            }
            if product.model_type == "video" {
                let mut video = Vec::new();
                for &mode in route_bindings::VIDEO_MODES {
                    let engine_ids = route_bindings::video_route_ids(backend, &id, mode);
                    if engine_ids.is_empty() {
                        continue;
                    }
                    let mut engines = Vec::new();
                    for &engine in engine_ids {
                        if let Some(route) = export.generator(engine, backend) {
                            engines.push(route.clone());
                        }
                    }
                    if !engines.is_empty() {
                        video.push(VideoRoute {
                            mode: mode.to_owned(),
                            engines,
                        });
                    }
                }
                if !video.is_empty() {
                    video_routes.insert(backend.to_owned(), video);
                }
            }
            let mut providers = export.provider(&id, backend);
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
                    let vector_route_registered = export.routes.get(&id).is_some_and(|route| {
                        route.modality == "vector" && route.backend == export.backend
                    });
                    if !vector_route_registered || !export.text_llm_ids.contains(provider) {
                        return Err(format!(
                            "unregistered {id}/{backend} vector route or backing LLM provider {provider}"
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
        let pag = if product.model_type == "image" {
            let pag_routes: Vec<(&str, &Option<PagRouteCapability>)> = if !routes.is_empty() {
                routes
                    .iter()
                    .map(|(backend, route)| (backend.as_str(), &route.pag))
                    .collect()
            } else if let Some(adapter) = product.fields.get("adapter").and_then(Value::as_str) {
                exports
                    .iter()
                    .filter_map(|(&backend, export)| {
                        export
                            .generator(adapter, backend)
                            .map(|route| (backend, &route.pag))
                    })
                    .collect()
            } else {
                Vec::new()
            };
            if let Some((_, first)) = pag_routes.first() {
                for (backend, candidate) in pag_routes.iter().skip(1) {
                    if *candidate != *first {
                        return Err(format!(
                            "PAG capability mismatch across backends for {id} ({backend})"
                        ));
                    }
                }
                first.as_ref().map(PagSpec::from)
            } else {
                None
            }
        } else {
            None
        };
        let mut features = BTreeMap::new();
        let mut feature_details = BTreeMap::new();
        if product.model_type == "image" && !routes.is_empty() {
            let low_res_applicable = routes.values().all(|route| {
                route
                    .features
                    .get("low_res_start")
                    .and_then(|detail| detail.get("applicable"))
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
            });
            if low_res_applicable {
                features.insert("low_res_start".to_owned(), true);
                if let Some(detail) = routes
                    .get("mlx")
                    .or_else(|| routes.get("candle"))
                    .and_then(|route| route.features.get("low_res_start"))
                {
                    feature_details.insert("low_res_start".to_owned(), detail.clone());
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
                    pag,
                    features,
                    feature_details,
                    routes,
                    video_routes,
                    provider_routes,
                    evidence: lanes,
                    model_evidence,
                    download_evidence,
                    depth_evidence,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removing_a_routed_video_engine_removes_its_typed_route() {
        let products = crate::jsonc::strip_jsonc_comments(include_str!(
            "../../../../config/manifests/builtin.models.jsonc"
        ));
        let evidence = include_str!("../../../../config/manifests/registry/evidence.json");
        let policy = include_str!("../../../../config/manifests/registry/route-policy.json");
        let preview = crate::jsonc::strip_jsonc_comments(include_str!(
            "../../../../config/manifests/builtin.preview-support.jsonc"
        ));
        let mlx = include_str!("../../../../config/engine-capabilities/sga/macos-mlx.json");
        let candle = include_str!("../../../../config/engine-capabilities/sga/candle-cuda.json");
        let original = assemble(
            &products,
            evidence,
            policy,
            &preview,
            &[("mlx", mlx), ("candle", candle)],
        )
        .unwrap();
        for (backend, dump, model, route_id, mode) in [
            ("mlx", mlx, "ltx_2_5", "ltx_2_5", "text_to_video"),
            (
                "candle",
                candle,
                "ltx_2_5",
                "ltx_2_5_distilled",
                "text_to_video",
            ),
            (
                "mlx",
                mlx,
                "krea_realtime_14b",
                "krea_realtime_14b",
                "image_to_video",
            ),
            (
                "candle",
                candle,
                "minimax_h3",
                "minimax_h3",
                "first_last_frame",
            ),
        ] {
            assert!(original[model]
                .video_routes
                .get(backend)
                .is_some_and(|routes| routes.iter().any(|route| route.mode == mode)));
            let mut changed: Value = serde_json::from_str(dump).unwrap();
            assert!(changed["routes"]
                .as_object_mut()
                .unwrap()
                .remove(route_id)
                .is_some());
            let changed = serde_json::to_string(&changed).unwrap();
            let dumps = if backend == "mlx" {
                [("mlx", changed.as_str()), ("candle", candle)]
            } else {
                [("mlx", mlx), ("candle", changed.as_str())]
            };
            let projected = assemble(&products, evidence, policy, &preview, &dumps).unwrap();
            assert!(
                !projected[model]
                    .video_routes
                    .get(backend)
                    .is_some_and(|routes| routes.iter().any(|route| route.mode == mode)),
                "{backend}/{model}/{mode} must disappear when route {route_id} is removed from sga.capabilities/2"
            );
        }
    }
}
