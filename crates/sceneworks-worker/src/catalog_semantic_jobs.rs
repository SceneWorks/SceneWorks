//! Production catalog pipeline for sc-14958.
//!
//! One GPU-routed job owns the shared catalog-processing lease while it runs
//! bounded acquisition, objective analysis, and survivor-only semantic work.
//! Vectors and their searchable manifest stay outside SQLite; record metadata
//! contains only small results, pointers, digests, and exact provenance.
#![cfg_attr(
    all(not(target_os = "macos"), not(feature = "backend-candle")),
    allow(dead_code)
)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use std::sync::Arc;

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use sceneworks_core::catalog_store::CatalogProcessingLease;
use sceneworks_core::catalog_store::{
    Catalog, CatalogAnalyzerConfig, CatalogRecord, CatalogRegistry, NewCatalogRecord,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use sha2::{Digest, Sha256};

use crate::{WorkerError, WorkerResult};

const SEMANTIC_SCHEMA_VERSION: u32 = 1;
const VISION_ANALYZER_VERSION: &str = "catalog-vision-taxonomy-v1";
const EMBEDDING_ANALYZER_VERSION: &str = "catalog-semantic-embedding-v1";
const TAXONOMY_VERSION: &str = "sceneworks-training-tags-v1";
const VISION_MODEL_ID: &str = "huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated";
const VISION_MODEL_REVISION: &str = "b47a0690b22eaf1d9a63874d967a03781c90f9cf";
const CLIP_MODEL_ID: &str = "openai/clip-vit-large-patch14";
const CLIP_MODEL_REVISION: &str = "32bd64288804d66eefd0ccbe215aa642df71cc41";
const CLIP_EMBEDDER_ID: &str = "clip_vit_l14";
const CLIP_PROVIDER: &str = CLIP_EMBEDDER_ID;
const CLIP_SPACE: &str = "clip-vit-l14";
pub(crate) const INFERENCE_RUNTIME_REVISION: &str = "98dcbf3490c877c89114e7ba5836b2882a477d4a";
const DEFAULT_BATCH_SIZE: usize = 16;
const MAX_BATCH_SIZE: usize = 64;
const PAGE_SIZE: u32 = 250;
const MAX_ERROR_CHARS: usize = 2_048;
const CANCEL_MESSAGE: &str = "Catalog analysis canceled by user.";
const SEMANTIC_CHECKPOINT_KEY: &str = "catalog_semantic.checkpoint.v1";

const TAG_ALLOWLIST: &[&str] = &[
    "single_subject",
    "multiple_subjects",
    "close_up",
    "full_body",
    "portrait",
    "wide_shot",
    "centered_subject",
    "off_center_subject",
    "standing",
    "sitting",
    "walking",
    "running",
    "posing",
    "holding_object",
    "interacting",
    "casual_clothing",
    "formal_clothing",
    "outerwear",
    "dress",
    "uniform",
    "costume",
    "sportswear",
    "indoor",
    "outdoor",
    "studio",
    "urban",
    "nature",
    "landscape",
    "beach",
    "forest",
    "mountain",
    "street",
    "home",
    "workplace",
    "event",
    "front_view",
    "side_view",
    "back_view",
    "three_quarter_view",
    "low_angle",
    "high_angle",
    "aerial_view",
    "eye_level",
    "clean_background",
    "simple_background",
    "complex_background",
    "occluded",
    "cropped",
    "text_overlay",
    "watermark",
    "low_quality",
    "high_detail",
];

const VISION_POLICY_PROMPT: &str = r#"Classify this training image using only the fixed schema below.
Return one JSON object and no prose or markdown:
{"medium":{"value":"photograph|painting|sketch_drawing|illustration_cartoon_anime|render|sculpture|unknown","confidence":0.0},"tags":[{"value":"fixed_tag","confidence":0.0}]}

Tags must come only from this allowlist:
single_subject, multiple_subjects, close_up, full_body, portrait, wide_shot,
centered_subject, off_center_subject, standing, sitting, walking, running, posing,
holding_object, interacting, casual_clothing, formal_clothing, outerwear, dress,
uniform, costume, sportswear, indoor, outdoor, studio, urban, nature, landscape,
beach, forest, mountain, street, home, workplace, event, front_view, side_view,
back_view, three_quarter_view, low_angle, high_angle, aerial_view, eye_level,
clean_background, simple_background, complex_background, occluded, cropped,
text_overlay, watermark, low_quality, high_detail.

Choose tags only for visible composition, action, clothing, setting, viewpoint,
and training utility. Never infer or mention a person's identity, name, age,
race, ethnicity, nationality, religion, disability, sexual orientation, gender
identity, health, or other sensitive demographic attributes. Confidence must be
a finite number from 0 to 1. Omit uncertain tags."#;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VisualMedium {
    Photograph,
    Painting,
    SketchDrawing,
    IllustrationCartoonAnime,
    Render,
    Sculpture,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConfidentMedium {
    pub value: VisualMedium,
    pub confidence: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConfidentTag {
    pub value: String,
    pub confidence: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NormalizedVisionResult {
    pub medium: ConfidentMedium,
    pub tags: Vec<ConfidentTag>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawVisionResult {
    medium: RawMedium,
    #[serde(default)]
    tags: Vec<RawTag>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawMedium {
    value: String,
    confidence: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTag {
    value: String,
    confidence: f64,
}

pub fn parse_normalized_vision_result(text: &str) -> Result<NormalizedVisionResult, String> {
    let cleaned = clean_json_output(text);
    let raw: RawVisionResult = serde_json::from_str(&cleaned)
        .map_err(|_| "vision response did not match the required schema".to_owned())?;
    let medium = match normalize_token(&raw.medium.value).as_str() {
        "photograph" | "photo" => VisualMedium::Photograph,
        "painting" => VisualMedium::Painting,
        "sketch_drawing" | "sketch" | "drawing" => VisualMedium::SketchDrawing,
        "illustration_cartoon_anime" | "illustration" | "cartoon" | "anime" => {
            VisualMedium::IllustrationCartoonAnime
        }
        "render" | "3d_render" => VisualMedium::Render,
        "sculpture" => VisualMedium::Sculpture,
        "unknown" => VisualMedium::Unknown,
        _ => return Err("vision response medium was outside the allowed taxonomy".to_owned()),
    };
    let medium_confidence = valid_confidence(raw.medium.confidence, "medium")?;
    let allowed = TAG_ALLOWLIST.iter().copied().collect::<BTreeSet<_>>();
    let mut tags = BTreeMap::<String, f64>::new();
    for raw_tag in raw.tags {
        let value = normalize_token(&raw_tag.value);
        if !allowed.contains(value.as_str()) {
            continue;
        }
        let confidence = valid_confidence(raw_tag.confidence, &value)?;
        tags.entry(value)
            .and_modify(|current| *current = current.max(confidence))
            .or_insert(confidence);
    }
    Ok(NormalizedVisionResult {
        medium: ConfidentMedium {
            value: medium,
            confidence: medium_confidence,
        },
        tags: tags
            .into_iter()
            .map(|(value, confidence)| ConfidentTag { value, confidence })
            .collect(),
    })
}

fn clean_json_output(text: &str) -> String {
    let trimmed = text.trim();
    let unfenced = if trimmed.starts_with("```") && trimmed.ends_with("```") {
        let after_header = trimmed
            .find('\n')
            .map(|index| &trimmed[index + 1..])
            .unwrap_or(trimmed);
        after_header
            .strip_suffix("```")
            .unwrap_or(after_header)
            .trim()
    } else {
        trimmed
    };
    match (unfenced.find('{'), unfenced.rfind('}')) {
        (Some(start), Some(end)) if end >= start => unfenced[start..=end].to_owned(),
        _ => unfenced.to_owned(),
    }
}

fn normalize_token(value: &str) -> String {
    value
        .trim()
        .to_ascii_lowercase()
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character
            } else {
                '_'
            }
        })
        .collect::<String>()
        .split('_')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("_")
}

fn valid_confidence(value: f64, _field: &str) -> Result<f64, String> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err("vision response confidence was outside the allowed range".to_owned())
    }
}

fn inference_backend() -> &'static str {
    if cfg!(target_os = "macos") {
        "mlx"
    } else if cfg!(feature = "backend-candle") {
        "candle"
    } else {
        "unavailable"
    }
}

fn vision_provider() -> &'static str {
    if cfg!(target_os = "macos") {
        "mlx-llama"
    } else if cfg!(feature = "backend-candle") {
        "candle-llama"
    } else {
        "unavailable"
    }
}

fn is_structured_survivor(record: &CatalogRecord) -> bool {
    record
        .metadata
        .pointer("/analysis/structured/derived/qualifiedSingleFullBody")
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn structured_input_fingerprint(record: &CatalogRecord) -> Option<&str> {
    record
        .metadata
        .pointer("/analysis/structured/inputFingerprint")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty() && *value != "unavailable")
}

fn vision_is_current(record: &CatalogRecord, input_fingerprint: &str) -> bool {
    let value = record.metadata.pointer("/analysis/visionLanguage");
    value
        .and_then(|value| value.get("status"))
        .and_then(Value::as_str)
        == Some("succeeded")
        && value
            .and_then(|value| value.get("inputFingerprint"))
            .and_then(Value::as_str)
            == Some(input_fingerprint)
        && value
            .and_then(|value| value.get("analyzerVersion"))
            .and_then(Value::as_str)
            == Some(VISION_ANALYZER_VERSION)
        && value
            .and_then(|value| value.get("modelRevision"))
            .and_then(Value::as_str)
            == Some(VISION_MODEL_REVISION)
        && value
            .and_then(|value| value.get("taxonomyVersion"))
            .and_then(Value::as_str)
            == Some(TAXONOMY_VERSION)
        && value
            .and_then(|value| value.get("runtimeRevision"))
            .and_then(Value::as_str)
            == Some(INFERENCE_RUNTIME_REVISION)
        && value
            .and_then(|value| value.get("backend"))
            .and_then(Value::as_str)
            == Some(inference_backend())
        && value
            .and_then(|value| value.get("provider"))
            .and_then(Value::as_str)
            == Some(vision_provider())
}

fn embedding_is_current(record: &CatalogRecord, input_fingerprint: &str) -> bool {
    let value = record.metadata.pointer("/analysis/semanticEmbedding");
    record.embedding_path.is_some()
        && value
            .and_then(|value| value.get("status"))
            .and_then(Value::as_str)
            == Some("succeeded")
        && value
            .and_then(|value| value.get("inputFingerprint"))
            .and_then(Value::as_str)
            == Some(input_fingerprint)
        && value
            .and_then(|value| value.get("analyzerVersion"))
            .and_then(Value::as_str)
            == Some(EMBEDDING_ANALYZER_VERSION)
        && value
            .and_then(|value| value.get("modelRevision"))
            .and_then(Value::as_str)
            == Some(CLIP_MODEL_REVISION)
        && value
            .and_then(|value| value.get("runtimeRevision"))
            .and_then(Value::as_str)
            == Some(INFERENCE_RUNTIME_REVISION)
        && value
            .and_then(|value| value.get("backend"))
            .and_then(Value::as_str)
            == Some(inference_backend())
        && value
            .and_then(|value| value.get("provider"))
            .and_then(Value::as_str)
            == Some(CLIP_PROVIDER)
}

fn bounded_error(error: impl std::fmt::Display) -> String {
    error.to_string().chars().take(MAX_ERROR_CHARS).collect()
}

fn metadata_analysis_mut(metadata: &mut Value) -> WorkerResult<&mut Map<String, Value>> {
    if !metadata.is_object() {
        *metadata = json!({});
    }
    let root = metadata.as_object_mut().expect("metadata object");
    let analysis = root.entry("analysis").or_insert_with(|| json!({}));
    analysis.as_object_mut().ok_or_else(|| {
        WorkerError::InvalidPayload("catalog metadata.analysis must be an object".to_owned())
    })
}

fn record_update(record: CatalogRecord) -> NewCatalogRecord {
    NewCatalogRecord {
        id: record.id,
        image_path: record.image_path,
        thumbnail_path: record.thumbnail_path,
        embedding_path: record.embedding_path,
        artifact_path: record.artifact_path,
        metadata: record.metadata,
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct SemanticReconcileReport {
    records_reconciled: u64,
    embedding_required: bool,
}

fn check_semantic_cancel(cancel: &gen_core::CancelFlag) -> WorkerResult<()> {
    if cancel.is_cancelled() {
        Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned()))
    } else {
        Ok(())
    }
}

fn reconcile_semantics_blocking(
    registry: &CatalogRegistry,
    catalog_id: &str,
    semantic_embeddings_enabled: bool,
    cancel: &gen_core::CancelFlag,
) -> WorkerResult<SemanticReconcileReport> {
    let metadata =
        reconcile_semantic_records(registry, catalog_id, semantic_embeddings_enabled, cancel);
    let index = rebuild_embedding_index_streaming(
        registry,
        catalog_id,
        metadata.as_ref().ok().map(|_| cancel),
    );
    match (metadata, index) {
        (Ok(report), Ok(())) => Ok(report),
        (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
        (Err(primary), Err(index)) => Err(WorkerError::Engine(format!(
            "{primary}; embedding index reconciliation also failed: {index}"
        ))),
    }
}

fn reconcile_semantic_records(
    registry: &CatalogRegistry,
    catalog_id: &str,
    semantic_embeddings_enabled: bool,
    cancel: &gen_core::CancelFlag,
) -> WorkerResult<SemanticReconcileReport> {
    let mut cursor = None;
    let mut examined = 0_u64;
    let mut report = SemanticReconcileReport::default();
    loop {
        check_semantic_cancel(cancel)?;
        let page = registry
            .open_attached(catalog_id)
            .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?
            .page_records_after(cursor, PAGE_SIZE)
            .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?;
        if page.records.is_empty() {
            break;
        }
        for mut record in page.records {
            check_semantic_cancel(cancel)?;
            examined = examined.saturating_add(1);
            let survivor = is_structured_survivor(&record);
            let structured_fingerprint = structured_input_fingerprint(&record).map(str::to_owned);
            let vision_current = survivor
                && structured_fingerprint
                    .as_deref()
                    .is_some_and(|fingerprint| vision_is_current(&record, fingerprint));
            let embedding_current = survivor
                && structured_fingerprint
                    .as_deref()
                    .is_some_and(|fingerprint| embedding_is_current(&record, fingerprint));
            if semantic_embeddings_enabled
                && survivor
                && structured_fingerprint.is_some()
                && !embedding_current
            {
                report.embedding_required = true;
            }

            let mut changed = false;
            if !embedding_current {
                changed |= record.embedding_path.take().is_some();
            }
            if let Some(analysis) = record
                .metadata
                .get_mut("analysis")
                .and_then(Value::as_object_mut)
            {
                if !vision_current {
                    for key in ["visionLanguage", "medium", "tagMembership", "tagConfidence"] {
                        changed |= analysis.remove(key).is_some();
                    }
                }
                if !embedding_current {
                    changed |= analysis.remove("semanticEmbedding").is_some();
                }
                let selection = if survivor {
                    json!({
                        "status": "selected",
                        "predicate": "structured.derived.qualifiedSingleFullBody",
                        "structuredInputFingerprint": structured_fingerprint,
                    })
                } else {
                    json!({
                        "status": "filtered_out",
                        "predicate": "structured.derived.qualifiedSingleFullBody",
                        "structuredInputFingerprint": structured_fingerprint,
                    })
                };
                if analysis.get("semanticSelection") != Some(&selection) {
                    analysis.insert("semanticSelection".to_owned(), selection);
                    changed = true;
                }
            }
            if changed {
                registry
                    .open_attached(catalog_id)
                    .and_then(|mut catalog| catalog.append_records(&[record_update(record)]))
                    .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?;
                report.records_reconciled = report.records_reconciled.saturating_add(1);
            }
        }
        cursor = page.next_cursor;
        persist_semantic_checkpoint(
            registry,
            catalog_id,
            "semantic_reconcile",
            cursor,
            examined,
            report.records_reconciled,
        )?;
        if cursor.is_none() {
            break;
        }
    }
    Ok(report)
}

fn resolve_catalog_image(catalog: &Catalog, record: &CatalogRecord) -> WorkerResult<PathBuf> {
    let relative = record
        .metadata
        .pointer("/acquisition/cachePath")
        .and_then(Value::as_str)
        .or_else(|| {
            (!record.image_path.contains("://") && !record.image_path.trim().is_empty())
                .then_some(record.image_path.as_str())
        })
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "catalog record {} has no available local image",
                record.id
            ))
        })?;
    let path = Path::new(relative);
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        return Err(WorkerError::InvalidPayload(format!(
            "catalog record {} has an unsafe image path",
            record.id
        )));
    }
    let candidate = catalog.root().join(path);
    let resolved = candidate.canonicalize().map_err(WorkerError::Io)?;
    let root = catalog.root().canonicalize().map_err(WorkerError::Io)?;
    if !resolved.starts_with(&root) || !resolved.is_file() {
        return Err(WorkerError::InvalidPayload(format!(
            "catalog record {} image escapes its catalog",
            record.id
        )));
    }
    Ok(resolved)
}

fn l2_normalize(mut vector: Vec<f32>) -> Result<Vec<f32>, String> {
    if vector.is_empty() || vector.iter().any(|value| !value.is_finite()) {
        return Err("embedding is empty or non-finite".to_owned());
    }
    let norm = vector
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() || norm <= f64::EPSILON {
        return Err("embedding has zero or invalid norm".to_owned());
    }
    for value in &mut vector {
        *value = (f64::from(*value) / norm) as f32;
    }
    Ok(vector)
}

fn vector_bytes(vector: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(vector));
    for value in vector {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn atomic_write(path: &Path, bytes: &[u8]) -> WorkerResult<()> {
    let parent = path.parent().ok_or_else(|| {
        WorkerError::InvalidPayload("artifact path has no parent directory".to_owned())
    })?;
    fs::create_dir_all(parent)?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    temporary.write_all(bytes)?;
    temporary.as_file_mut().sync_all()?;
    temporary
        .persist(path)
        .map_err(|error| WorkerError::Io(error.error))?;
    Ok(())
}

fn persist_semantic_checkpoint(
    registry: &CatalogRegistry,
    catalog_id: &str,
    stage: &str,
    next_cursor: Option<i64>,
    examined: u64,
    updated: u64,
) -> WorkerResult<()> {
    let catalog = registry
        .open_attached(catalog_id)
        .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?;
    let mut state = catalog
        .contract_state()
        .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?;
    state.checkpoints.insert(
        SEMANTIC_CHECKPOINT_KEY.to_owned(),
        json!({
            "schemaVersion": SEMANTIC_SCHEMA_VERSION,
            "stage": stage,
            "nextCursor": next_cursor,
            "examined": examined,
            "updated": updated,
            "visionAnalyzerVersion": VISION_ANALYZER_VERSION,
            "visionModelRevision": VISION_MODEL_REVISION,
            "embeddingAnalyzerVersion": EMBEDDING_ANALYZER_VERSION,
            "embeddingModelRevision": CLIP_MODEL_REVISION,
            "taxonomyVersion": TAXONOMY_VERSION,
        }),
    );
    catalog
        .set_contract_state(&state)
        .map_err(|error| WorkerError::InvalidPayload(error.to_string()))
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn generate_vision_json(
    api: &crate::ApiClient,
    settings: &crate::Settings,
    job: &sceneworks_core::contracts::JobSnapshot,
    weights_dir: PathBuf,
    image_path: PathBuf,
) -> WorkerResult<String> {
    use gen_core::core_llm::{
        CancelFlag, Constraint, Content, Message, ModelRequirements, Role, Sampling, TextLlmRequest,
    };

    let image = tokio::task::spawn_blocking(move || {
        crate::prompt_refine_jobs::load_caption_image_ref(&image_path)
    })
    .await
    .map_err(|error| crate::task_join_error("catalog vision image decode", error))??;
    let cancel = CancelFlag::new();
    let blocking_cancel = cancel.clone();
    let spec = gen_core::core_llm::LoadSpec {
        source: weights_dir.to_string_lossy().into_owned(),
        quantize: None,
    };
    let requirements = ModelRequirements::default().with_constraint(Constraint::Json);
    let generation = crate::refine_model_cache::with_cached_refiner(
        spec,
        requirements,
        "catalog vision model load failed",
        move |model| {
            let request = TextLlmRequest {
                messages: vec![Message {
                    role: Role::User,
                    content: vec![
                        Content::Image(image),
                        Content::text(VISION_POLICY_PROMPT.to_owned()),
                    ],
                    thinking: None,
                    tool_calls: Vec::new(),
                }],
                sampling: Sampling {
                    temperature: 0.1,
                    top_p: 0.8,
                    ..Sampling::default()
                },
                max_new_tokens: 512,
                constraint: Some(Constraint::Json),
                cancel: blocking_cancel,
                ..Default::default()
            };
            model
                .generate(&request, &mut |_| {})
                .map(|output| output.text)
                .map_err(|error| {
                    WorkerError::Engine(format!("catalog vision inference failed: {error}"))
                })
        },
    );
    tokio::pin!(generation);
    loop {
        tokio::select! {
            result = &mut generation => return result,
            _ = tokio::time::sleep(std::time::Duration::from_secs(2)) => {
                if crate::shutdown_requested()
                    || crate::cancel_requested_peek(api, &job.id).await
                {
                    cancel.cancel();
                    return Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned()));
                }
                crate::heartbeat(api, settings, sceneworks_core::contracts::WorkerStatus::Busy, Some(&job.id)).await?;
            }
        }
    }
}

fn apply_vision_outcome(
    record: &mut CatalogRecord,
    input_fingerprint: &str,
    outcome: WorkerResult<NormalizedVisionResult>,
) -> WorkerResult<()> {
    let analysis = metadata_analysis_mut(&mut record.metadata)?;
    match outcome {
        Ok(result) => {
            let membership = result
                .tags
                .iter()
                .map(|tag| (tag.value.clone(), Value::Bool(true)))
                .collect::<Map<_, _>>();
            let confidence = result
                .tags
                .iter()
                .map(|tag| (tag.value.clone(), json!(tag.confidence)))
                .collect::<Map<_, _>>();
            analysis.insert(
                "visionLanguage".to_owned(),
                json!({
                    "schemaVersion": SEMANTIC_SCHEMA_VERSION,
                    "status": "succeeded",
                    "inputFingerprint": input_fingerprint,
                    "analyzerVersion": VISION_ANALYZER_VERSION,
                    "taxonomyVersion": TAXONOMY_VERSION,
                    "modelId": VISION_MODEL_ID,
                    "modelRevision": VISION_MODEL_REVISION,
                    "runtimeRevision": INFERENCE_RUNTIME_REVISION,
                    "backend": inference_backend(),
                    "provider": vision_provider(),
                    "medium": result.medium,
                    "tags": result.tags,
                }),
            );
            analysis.insert("medium".to_owned(), json!(result.medium.value));
            analysis.insert("tagMembership".to_owned(), Value::Object(membership));
            analysis.insert("tagConfidence".to_owned(), Value::Object(confidence));
        }
        Err(error) => {
            analysis.remove("medium");
            analysis.remove("tagMembership");
            analysis.remove("tagConfidence");
            analysis.insert(
                "visionLanguage".to_owned(),
                json!({
                    "schemaVersion": SEMANTIC_SCHEMA_VERSION,
                    "status": "failed",
                    "inputFingerprint": input_fingerprint,
                    "analyzerVersion": VISION_ANALYZER_VERSION,
                    "taxonomyVersion": TAXONOMY_VERSION,
                    "modelId": VISION_MODEL_ID,
                    "modelRevision": VISION_MODEL_REVISION,
                    "runtimeRevision": INFERENCE_RUNTIME_REVISION,
                    "backend": inference_backend(),
                    "provider": vision_provider(),
                    "error": bounded_error(error),
                }),
            );
        }
    }
    Ok(())
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn run_vision_pass(
    api: &crate::ApiClient,
    settings: &crate::Settings,
    job: &sceneworks_core::contracts::JobSnapshot,
    registry: &CatalogRegistry,
    catalog_id: &str,
    weights_dir: &Path,
) -> WorkerResult<(u64, u64)> {
    let mut cursor = None;
    let mut examined = 0_u64;
    let mut updated = 0_u64;
    loop {
        let page = registry
            .open_attached(catalog_id)
            .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?
            .page_records_after(cursor, PAGE_SIZE)
            .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?;
        if page.records.is_empty() {
            break;
        }
        for mut record in page.records {
            if !is_structured_survivor(&record) {
                continue;
            }
            examined = examined.saturating_add(1);
            let Some(input_fingerprint) = structured_input_fingerprint(&record).map(str::to_owned)
            else {
                continue;
            };
            if vision_is_current(&record, &input_fingerprint) {
                continue;
            }
            crate::check_cancel(api, &job.id, CANCEL_MESSAGE).await?;
            let catalog = registry
                .open_attached(catalog_id)
                .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?;
            let image_path = resolve_catalog_image(&catalog, &record);
            drop(catalog);
            let outcome = match image_path {
                Ok(image_path) => {
                    generate_vision_json(api, settings, job, weights_dir.to_path_buf(), image_path)
                        .await
                        .and_then(|text| {
                            parse_normalized_vision_result(&text).map_err(WorkerError::Engine)
                        })
                }
                Err(error) => Err(error),
            };
            if let Err(WorkerError::Canceled(message)) = &outcome {
                return Err(WorkerError::Canceled(message.clone()));
            }
            apply_vision_outcome(&mut record, &input_fingerprint, outcome)?;
            registry
                .open_attached(catalog_id)
                .and_then(|mut catalog| catalog.append_records(&[record_update(record)]))
                .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?;
            updated = updated.saturating_add(1);
        }
        cursor = page.next_cursor;
        persist_semantic_checkpoint(registry, catalog_id, "vision", cursor, examined, updated)?;
        if cursor.is_none() {
            break;
        }
    }
    Ok((examined, updated))
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn fetch_catalog_images_with_heartbeat(
    api: &crate::ApiClient,
    settings: &crate::Settings,
    job_id: &str,
    registry: &CatalogRegistry,
    catalog_id: &str,
    options: &crate::catalog_image_fetch::CatalogImageFetchOptions,
    processing: &CatalogProcessingLease,
) -> WorkerResult<crate::catalog_image_fetch::CatalogImageFetchReport> {
    use sceneworks_core::contracts::WorkerStatus;

    let fetch = crate::catalog_image_fetch::fetch_attached_catalog_images_under_processing_lease(
        registry, catalog_id, options, processing,
    );
    tokio::pin!(fetch);
    let mut interval = tokio::time::interval(crate::progress_report_interval(settings));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            result = &mut fetch => {
                return result
                    .map_err(|error| WorkerError::Engine(format!("catalog image fetch failed: {error}")));
            }
            _ = interval.tick() => {
                crate::heartbeat(api, settings, WorkerStatus::Busy, Some(job_id)).await?;
                crate::check_cancel(api, job_id, CANCEL_MESSAGE).await?;
            }
        }
    }
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn run_embedding_pass_blocking(
    registry: CatalogRegistry,
    catalog_id: String,
    weights_dir: PathBuf,
    batch_size: usize,
    cancel: gen_core::CancelFlag,
) -> WorkerResult<(u64, u64)> {
    use gen_core::{Image, LoadSpec, WeightsSource};

    let embedder = crate::inference_runtime::load_image_embedder(
        CLIP_EMBEDDER_ID,
        &LoadSpec::new(WeightsSource::Dir(weights_dir)),
    )
    .map_err(|error| WorkerError::Engine(format!("CLIP embedder load failed: {error}")))?;
    let descriptor = embedder.descriptor().clone();
    if descriptor.space != CLIP_SPACE {
        return Err(WorkerError::Engine(format!(
            "CLIP embedding space mismatch: expected {CLIP_SPACE}, got {}",
            descriptor.space
        )));
    }
    let mut cursor = None;
    let mut examined = 0_u64;
    let mut updated = 0_u64;
    loop {
        if cancel.is_cancelled() {
            return Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned()));
        }
        let page = registry
            .open_attached(&catalog_id)
            .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?
            .page_records_after(cursor, PAGE_SIZE)
            .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?;
        if page.records.is_empty() {
            break;
        }
        let mut pending = Vec::new();
        for record in &page.records {
            if !is_structured_survivor(record) {
                continue;
            }
            examined = examined.saturating_add(1);
            let Some(input_fingerprint) = structured_input_fingerprint(record).map(str::to_owned)
            else {
                continue;
            };
            if embedding_is_current(record, &input_fingerprint) {
                continue;
            }
            let catalog = registry
                .open_attached(&catalog_id)
                .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?;
            match resolve_catalog_image(&catalog, record) {
                Ok(path) => pending.push((record.clone(), input_fingerprint, path)),
                Err(error) => {
                    persist_embedding_failure(
                        &registry,
                        &catalog_id,
                        record.clone(),
                        &input_fingerprint,
                        error,
                    )?;
                    updated = updated.saturating_add(1);
                }
            }
        }
        for batch in pending.chunks(batch_size) {
            if cancel.is_cancelled() {
                return Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned()));
            }
            let mut images = Vec::with_capacity(batch.len());
            let mut decoded_records = Vec::with_capacity(batch.len());
            for (record, input_fingerprint, path) in batch {
                match crate::image_decode::decode_image_any(path) {
                    Ok(decoded) => {
                        let decoded = decoded.to_rgb8();
                        images.push(Image {
                            width: decoded.width(),
                            height: decoded.height(),
                            pixels: decoded.into_raw(),
                        });
                        decoded_records.push((record.clone(), input_fingerprint.clone()));
                    }
                    Err(error) => {
                        persist_embedding_failure(
                            &registry,
                            &catalog_id,
                            record.clone(),
                            input_fingerprint,
                            error,
                        )?;
                        updated = updated.saturating_add(1);
                    }
                }
            }
            if images.is_empty() {
                continue;
            }
            let vectors = match embedder.embed_batch(&images) {
                Ok(vectors) if vectors.len() == decoded_records.len() => vectors,
                Ok(_) => {
                    for (record, input_fingerprint) in decoded_records {
                        persist_embedding_failure(
                            &registry,
                            &catalog_id,
                            record,
                            &input_fingerprint,
                            "CLIP batch output count did not match its input count",
                        )?;
                        updated = updated.saturating_add(1);
                    }
                    continue;
                }
                Err(error) => {
                    let message = format!("CLIP batch embed failed: {error}");
                    for (record, input_fingerprint) in decoded_records {
                        persist_embedding_failure(
                            &registry,
                            &catalog_id,
                            record,
                            &input_fingerprint,
                            &message,
                        )?;
                        updated = updated.saturating_add(1);
                    }
                    continue;
                }
            };
            for ((mut record, input_fingerprint), vector) in
                decoded_records.into_iter().zip(vectors)
            {
                if cancel.is_cancelled() {
                    return Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned()));
                }
                let normalized = match l2_normalize(vector) {
                    Ok(value) if value.len() == descriptor.embedding_dim => value,
                    Ok(value) => {
                        persist_embedding_failure(
                            &registry,
                            &catalog_id,
                            record,
                            &input_fingerprint,
                            format!(
                                "CLIP embedding dimension mismatch: expected {}, got {}",
                                descriptor.embedding_dim,
                                value.len()
                            ),
                        )?;
                        updated = updated.saturating_add(1);
                        continue;
                    }
                    Err(error) => {
                        persist_embedding_failure(
                            &registry,
                            &catalog_id,
                            record,
                            &input_fingerprint,
                            error,
                        )?;
                        updated = updated.saturating_add(1);
                        continue;
                    }
                };
                let bytes = vector_bytes(&normalized);
                let digest_hex = format!("{:x}", Sha256::digest(&bytes));
                let digest = format!("sha256:{digest_hex}");
                let relative = PathBuf::from("embeddings")
                    .join(CLIP_SPACE)
                    .join(format!("{digest_hex}.f32"));
                let catalog = registry
                    .open_attached(&catalog_id)
                    .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?;
                atomic_write(&catalog.root().join(&relative), &bytes)?;
                record.embedding_path = Some(relative.to_string_lossy().replace('\\', "/"));
                metadata_analysis_mut(&mut record.metadata)?.insert(
                    "semanticEmbedding".to_owned(),
                    json!({
                        "schemaVersion": SEMANTIC_SCHEMA_VERSION,
                        "status": "succeeded",
                        "inputFingerprint": input_fingerprint,
                        "analyzerVersion": EMBEDDING_ANALYZER_VERSION,
                        "modelId": CLIP_MODEL_ID,
                        "modelRevision": CLIP_MODEL_REVISION,
                        "runtimeRevision": INFERENCE_RUNTIME_REVISION,
                        "embedderId": CLIP_EMBEDDER_ID,
                        "backend": descriptor.backend,
                        "provider": CLIP_PROVIDER,
                        "space": descriptor.space,
                        "dimension": descriptor.embedding_dim,
                        "digest": digest,
                        "path": record.embedding_path,
                    }),
                );
                registry
                    .open_attached(&catalog_id)
                    .and_then(|mut catalog| catalog.append_records(&[record_update(record)]))
                    .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?;
                updated = updated.saturating_add(1);
            }
        }
        cursor = page.next_cursor;
        persist_semantic_checkpoint(
            &registry,
            &catalog_id,
            "embedding",
            cursor,
            examined,
            updated,
        )?;
        if cursor.is_none() {
            break;
        }
    }
    Ok((examined, updated))
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn run_embedding_pass_with_index_reconciliation(
    registry: CatalogRegistry,
    catalog_id: String,
    weights_dir: PathBuf,
    batch_size: usize,
    cancel: gen_core::CancelFlag,
) -> WorkerResult<(u64, u64)> {
    let embedding = run_embedding_pass_blocking(
        registry.clone(),
        catalog_id.clone(),
        weights_dir,
        batch_size,
        cancel,
    );
    // Always publish the SQLite-derived index, including after a persisted
    // record-local failure followed by cancellation or another fatal error.
    finish_with_index_reconciliation(embedding, &registry, &catalog_id)
}

fn finish_with_index_reconciliation<T>(
    primary: WorkerResult<T>,
    registry: &CatalogRegistry,
    catalog_id: &str,
) -> WorkerResult<T> {
    let index = rebuild_embedding_index_streaming(registry, catalog_id, None);
    match (primary, index) {
        (Ok(counts), Ok(())) => Ok(counts),
        (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
        (Err(primary), Err(index)) => Err(WorkerError::Engine(format!(
            "{primary}; embedding index reconciliation also failed: {index}"
        ))),
    }
}

fn persist_embedding_failure(
    registry: &CatalogRegistry,
    catalog_id: &str,
    mut record: CatalogRecord,
    input_fingerprint: &str,
    error: impl std::fmt::Display,
) -> WorkerResult<()> {
    record.embedding_path = None;
    metadata_analysis_mut(&mut record.metadata)?.insert(
        "semanticEmbedding".to_owned(),
        json!({
            "schemaVersion": SEMANTIC_SCHEMA_VERSION,
            "status": "failed",
            "inputFingerprint": input_fingerprint,
            "analyzerVersion": EMBEDDING_ANALYZER_VERSION,
            "modelId": CLIP_MODEL_ID,
            "modelRevision": CLIP_MODEL_REVISION,
            "runtimeRevision": INFERENCE_RUNTIME_REVISION,
            "embedderId": CLIP_EMBEDDER_ID,
            "backend": inference_backend(),
            "provider": CLIP_PROVIDER,
            "error": bounded_error(error),
        }),
    );
    registry
        .open_attached(catalog_id)
        .and_then(|mut catalog| catalog.append_records(&[record_update(record)]))
        .map(|_| ())
        .map_err(|error| WorkerError::InvalidPayload(error.to_string()))
}

fn rebuild_embedding_index_streaming(
    registry: &CatalogRegistry,
    catalog_id: &str,
    cancel: Option<&gen_core::CancelFlag>,
) -> WorkerResult<()> {
    let catalog = registry
        .open_attached(catalog_id)
        .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?;
    let index = catalog
        .root()
        .join("embeddings")
        .join(CLIP_SPACE)
        .join("index.jsonl");
    let parent = index.parent().ok_or_else(|| {
        WorkerError::InvalidPayload("embedding index has no parent directory".to_owned())
    })?;
    fs::create_dir_all(parent)?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    let mut cursor = None;
    let mut examined = 0_u64;
    let mut indexed = 0_u64;
    let mut canceled = false;
    loop {
        canceled |= cancel.is_some_and(gen_core::CancelFlag::is_cancelled);
        let page = catalog
            .page_records_after(cursor, PAGE_SIZE)
            .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?;
        for record in page.records {
            examined = examined.saturating_add(1);
            let Some(embedding) = record.metadata.pointer("/analysis/semanticEmbedding") else {
                continue;
            };
            if embedding.get("status").and_then(Value::as_str) != Some("succeeded")
                || record.embedding_path.is_none()
            {
                continue;
            }
            serde_json::to_writer(
                &mut temporary,
                &json!({
                    "recordId": record.id,
                    "path": record.embedding_path,
                    "inputFingerprint": embedding.get("inputFingerprint"),
                    "digest": embedding.get("digest"),
                    "dimension": embedding.get("dimension"),
                    "space": embedding.get("space"),
                    "modelId": embedding.get("modelId"),
                    "modelRevision": embedding.get("modelRevision"),
                }),
            )
            .map_err(|error| WorkerError::Engine(error.to_string()))?;
            temporary.write_all(b"\n")?;
            indexed = indexed.saturating_add(1);
        }
        cursor = page.next_cursor;
        persist_semantic_checkpoint(
            registry,
            catalog_id,
            "embedding_index",
            cursor,
            examined,
            indexed,
        )?;
        if cursor.is_none() {
            break;
        }
    }
    temporary.as_file_mut().sync_all()?;
    temporary
        .persist(&index)
        .map_err(|error| WorkerError::Io(error.error))?;
    if canceled {
        Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned()))
    } else {
        Ok(())
    }
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) async fn run_catalog_analysis_job(
    api: &crate::ApiClient,
    settings: &crate::Settings,
    job: &sceneworks_core::contracts::JobSnapshot,
) -> WorkerResult<()> {
    use sceneworks_core::contracts::{JobStatus, ProgressStage, WorkerStatus};

    let catalog_id = job
        .payload
        .get("catalogId")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload("catalog analysis requires catalogId".to_owned())
        })?
        .to_owned();
    let expected_revision = job
        .payload
        .get("analyzerConfigRevision")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "catalog analysis requires analyzerConfigRevision".to_owned(),
            )
        })?;
    let batch_size = job
        .payload
        .get("batchSize")
        .and_then(Value::as_u64)
        .unwrap_or(DEFAULT_BATCH_SIZE as u64) as usize;
    if batch_size == 0 || batch_size > MAX_BATCH_SIZE {
        return Err(WorkerError::InvalidPayload(format!(
            "catalog batchSize must be between 1 and {MAX_BATCH_SIZE}"
        )));
    }

    let registry = CatalogRegistry::new(&settings.config_dir);
    let catalog = registry
        .open_attached(&catalog_id)
        .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?;
    let analyzer_config = catalog
        .analyzer_config()
        .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?;
    if analyzer_config.revision != expected_revision {
        return Err(WorkerError::InvalidPayload(
            "catalog analyzer configuration changed; create a fresh job".to_owned(),
        ));
    }
    validate_pipeline_settings(&analyzer_config)?;
    let record_count = catalog
        .storage_accounting()
        .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?
        .record_count;
    drop(catalog);

    crate::heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    crate::update_job(
        api,
        &job.id,
        crate::progress_payload(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.02,
            "Preparing catalog analysis pipeline.",
            None,
            None,
            None,
        ),
    )
    .await?;

    let mut structured_resources = None;
    if analyzer_config.settings.structured_analysis_enabled {
        let person_detector = crate::person_jobs::require_detector_weights(settings)?;
        let face_weights_directory =
            crate::image_jobs::ensure_face_stack_dir(api, settings, job).await?;
        let (pose_detector, pose_model) = crate::pose_jobs::require_dwpose_weights(settings)?;
        structured_resources = Some(crate::catalog_analysis::ModelBackedCatalogAnalyzerPaths {
            person_detector,
            face_weights_directory,
            pose_detector,
            pose_model,
        });
    }
    let vision_weights = analyzer_config
        .settings
        .vision_analysis_enabled
        .then(|| -> WorkerResult<PathBuf> {
            crate::model_jobs::huggingface_pinned_snapshot_dir(
                &settings.data_dir,
                VISION_MODEL_ID,
                VISION_MODEL_REVISION,
            )
            .ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "catalog vision model exact snapshot {VISION_MODEL_ID}@{VISION_MODEL_REVISION} is not cached"
                ))
            })
        })
        .transpose()?;
    let catalog = registry
        .open_attached(&catalog_id)
        .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?;
    let processing = Arc::new(
        CatalogProcessingLease::try_acquire(&catalog)
            .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?,
    );
    drop(catalog);

    crate::update_job(
        api,
        &job.id,
        crate::progress_payload(
            JobStatus::Running,
            ProgressStage::Running,
            0.08,
            "Fetching catalog images.",
            None,
            None,
            None,
        ),
    )
    .await?;
    let fetch_options = crate::catalog_image_fetch::CatalogImageFetchOptions {
        accepted_target: record_count,
        ..crate::catalog_image_fetch::CatalogImageFetchOptions::default()
    };
    let fetch = fetch_catalog_images_with_heartbeat(
        api,
        settings,
        &job.id,
        &registry,
        &catalog_id,
        &fetch_options,
        processing.as_ref(),
    )
    .await?;

    let structured = if let Some(resources) = structured_resources {
        crate::update_job(
            api,
            &job.id,
            crate::progress_payload(
                JobStatus::Running,
                ProgressStage::Running,
                0.32,
                "Running structured survivor analysis.",
                None,
                None,
                None,
            ),
        )
        .await?;
        let mut config = crate::catalog_analysis::CatalogAnalysisConfig::default();
        config.thresholds.person_min_confidence =
            analyzer_config.settings.thresholds.person_min_confidence;
        config.thresholds.face_min_confidence =
            analyzer_config.settings.thresholds.face_min_confidence;
        config.thresholds.pose_min_keypoint_confidence = analyzer_config
            .settings
            .thresholds
            .pose_min_keypoint_confidence;
        config.thresholds.prominent_frame_fraction =
            analyzer_config.settings.thresholds.prominent_frame_fraction;
        config.thresholds.frame_edge_margin = analyzer_config.settings.thresholds.frame_edge_margin;
        config.thresholds.min_pose_coverage = analyzer_config.settings.thresholds.min_pose_coverage;
        let mut analyzers = crate::catalog_analysis::ModelBackedCatalogAnalyzers::new(resources);
        let structured_processing = processing.clone();
        let cancel = gen_core::CancelFlag::new();
        let blocking_cancel = cancel.clone();
        Some(
            crate::run_blocking_with_heartbeat(
                api,
                settings,
                &job.id,
                Some(cancel),
                CANCEL_MESSAGE,
                "catalog structured analysis",
                crate::no_cancel_ack(),
                tokio::task::spawn_blocking({
                let registry = registry.clone();
                let catalog_id = catalog_id.clone();
                move || {
                    let mut on_page = |_progress: crate::catalog_analysis::CatalogAnalysisPageProgress| {
                        Ok(())
                    };
                    crate::catalog_analysis::analyze_attached_catalog_under_processing_lease_with_control(
                        &registry,
                        &catalog_id,
                        &config,
                        &mut analyzers,
                        structured_processing.as_ref(),
                        &blocking_cancel,
                        &mut on_page,
                    )
                    .map_err(|error| match error {
                        crate::catalog_analysis::CatalogAnalysisError::Canceled(message) => {
                            WorkerError::Canceled(message)
                        }
                        error => WorkerError::Engine(error.to_string()),
                    })
                }
            }),
            )
            .await?,
        )
    } else {
        None
    };

    let reconcile_cancel = gen_core::CancelFlag::new();
    let blocking_reconcile_cancel = reconcile_cancel.clone();
    let reconcile_registry = registry.clone();
    let reconcile_catalog_id = catalog_id.clone();
    let semantic_embeddings_enabled = analyzer_config.settings.semantic_embeddings_enabled;
    let semantic_reconcile = crate::run_blocking_with_heartbeat(
        api,
        settings,
        &job.id,
        Some(reconcile_cancel),
        CANCEL_MESSAGE,
        "catalog semantic reconciliation",
        crate::no_cancel_ack(),
        tokio::task::spawn_blocking(move || {
            reconcile_semantics_blocking(
                &reconcile_registry,
                &reconcile_catalog_id,
                semantic_embeddings_enabled,
                &blocking_reconcile_cancel,
            )
        }),
    )
    .await?;
    let filtered_semantic_records = semantic_reconcile.records_reconciled;

    let mut vision_counts = (0, 0);
    if let Some(weights) = vision_weights {
        crate::update_job(
            api,
            &job.id,
            crate::progress_payload(
                JobStatus::Running,
                ProgressStage::Running,
                0.60,
                "Classifying structured-filter survivors.",
                None,
                None,
                None,
            ),
        )
        .await?;
        vision_counts =
            run_vision_pass(api, settings, job, &registry, &catalog_id, &weights).await?;
    }

    let mut embedding_counts = (0, 0);
    let embedding_required = semantic_reconcile.embedding_required;
    let clip_weights = embedding_required
        .then(|| -> WorkerResult<PathBuf> {
            crate::model_jobs::huggingface_pinned_snapshot_dir(
                &settings.data_dir,
                CLIP_MODEL_ID,
                CLIP_MODEL_REVISION,
            )
            .ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "catalog CLIP model exact snapshot {CLIP_MODEL_ID}@{CLIP_MODEL_REVISION} is not cached"
                ))
            })
        })
        .transpose()?;
    if let Some(weights) = clip_weights {
        crate::update_job(
            api,
            &job.id,
            crate::progress_payload(
                JobStatus::Running,
                ProgressStage::Running,
                0.80,
                "Embedding structured-filter survivors in batches.",
                None,
                None,
                None,
            ),
        )
        .await?;
        let registry_for_embeddings = registry.clone();
        let catalog_for_embeddings = catalog_id.clone();
        let cancel = gen_core::CancelFlag::new();
        let blocking_cancel = cancel.clone();
        let blocking = tokio::task::spawn_blocking(move || {
            run_embedding_pass_with_index_reconciliation(
                registry_for_embeddings,
                catalog_for_embeddings,
                weights,
                batch_size,
                blocking_cancel,
            )
        });
        embedding_counts = crate::run_blocking_with_heartbeat(
            api,
            settings,
            &job.id,
            Some(cancel),
            CANCEL_MESSAGE,
            "catalog semantic embeddings",
            crate::no_cancel_ack(),
            blocking,
        )
        .await?;
    }

    let catalog = registry
        .open_attached(&catalog_id)
        .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?;
    let mut state = catalog
        .contract_state()
        .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?;
    if analyzer_config.settings.vision_analysis_enabled {
        state.analyzer_versions.insert(
            "vision_language".to_owned(),
            format!(
                "{VISION_ANALYZER_VERSION}:{VISION_MODEL_ID}@{VISION_MODEL_REVISION}:{TAXONOMY_VERSION}"
            ),
        );
    }
    if analyzer_config.settings.semantic_embeddings_enabled {
        state.analyzer_versions.insert(
            "semantic_embedding".to_owned(),
            format!(
                "{EMBEDDING_ANALYZER_VERSION}:{CLIP_MODEL_ID}@{CLIP_MODEL_REVISION}:{CLIP_SPACE}"
            ),
        );
    }
    catalog
        .set_contract_state(&state)
        .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?;
    drop(processing);

    let result = json!({
        "catalogId": catalog_id,
        "analyzerConfigRevision": expected_revision,
        "fetch": fetch,
        "structured": structured,
        "filteredSemanticRecordsCleared": filtered_semantic_records,
        "vision": {
            "survivorsExamined": vision_counts.0,
            "recordsUpdated": vision_counts.1,
            "modelId": VISION_MODEL_ID,
            "modelRevision": VISION_MODEL_REVISION,
            "taxonomyVersion": TAXONOMY_VERSION,
        },
        "semanticEmbeddings": {
            "survivorsExamined": embedding_counts.0,
            "recordsUpdated": embedding_counts.1,
            "modelId": CLIP_MODEL_ID,
            "modelRevision": CLIP_MODEL_REVISION,
            "space": CLIP_SPACE,
        },
    })
    .as_object()
    .cloned()
    .expect("catalog result object");
    crate::update_job(
        api,
        &job.id,
        crate::progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Catalog analysis completed.",
            None,
            Some(result),
            None,
        ),
    )
    .await?;
    Ok(())
}

fn validate_pipeline_settings(config: &CatalogAnalyzerConfig) -> WorkerResult<()> {
    if (config.settings.vision_analysis_enabled || config.settings.semantic_embeddings_enabled)
        && !config.settings.structured_analysis_enabled
    {
        return Err(WorkerError::InvalidPayload(
            "Vision analysis and semantic embeddings require structured analysis so expensive models run only on survivors."
                .to_owned(),
        ));
    }
    Ok(())
}

#[cfg(not(any(target_os = "macos", feature = "backend-candle")))]
pub(crate) async fn run_catalog_analysis_job(
    _api: &crate::ApiClient,
    _settings: &crate::Settings,
    _job: &sceneworks_core::contracts::JobSnapshot,
) -> WorkerResult<()> {
    Err(WorkerError::InvalidPayload(
        "Catalog semantic analysis needs the macOS MLX backend or the candle backend.".to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sceneworks_core::catalog_store::CatalogRecordFilter;

    #[test]
    fn semantic_provenance_matches_linked_inference_revision() {
        let manifest = include_str!("../Cargo.toml");
        for dependency in ["sceneworks-gen-core", "runtime-macos", "runtime-cuda"] {
            let prefix = format!("{dependency} =");
            let declaration = manifest
                .lines()
                .find(|line| line.trim_start().starts_with(&prefix))
                .unwrap_or_else(|| panic!("missing {dependency} dependency declaration"));
            assert!(
                declaration.contains(&format!("rev = \"{INFERENCE_RUNTIME_REVISION}\"")),
                "{dependency} must stay pinned to the semantic provenance revision"
            );
        }
    }

    fn survivor_record(id: &str) -> CatalogRecord {
        CatalogRecord {
            id: id.to_owned(),
            image_path: format!("images/{id}.jpg"),
            thumbnail_path: None,
            embedding_path: None,
            artifact_path: None,
            metadata: json!({
                "analysis": {
                    "structured": {
                        "inputFingerprint": "sha256:abc",
                        "derived": {"qualifiedSingleFullBody": true}
                    }
                }
            }),
            created_at: String::new(),
            updated_at: String::new(),
        }
    }

    #[test]
    fn parser_normalizes_medium_tags_deduplicates_and_drops_open_vocabulary() {
        let parsed = parse_normalized_vision_result(
            r#"```json
            {"medium":{"value":"Photo","confidence":0.91},"tags":[
              {"value":"Full body","confidence":0.7},
              {"value":"full-body","confidence":0.94},
              {"value":"red_hair","confidence":1.0},
              {"value":"outdoor","confidence":0.8}
            ]}
            ```"#,
        )
        .expect("valid normalized result");
        assert_eq!(parsed.medium.value, VisualMedium::Photograph);
        assert_eq!(
            parsed.tags,
            vec![
                ConfidentTag {
                    value: "full_body".to_owned(),
                    confidence: 0.94,
                },
                ConfidentTag {
                    value: "outdoor".to_owned(),
                    confidence: 0.8,
                },
            ]
        );
    }

    #[test]
    fn demographic_and_identity_terms_can_never_escape_the_fixed_allowlist() {
        let parsed = parse_normalized_vision_result(
            r#"{"medium":{"value":"painting","confidence":1.0},"tags":[
              {"value":"person_name","confidence":1.0},
              {"value":"race_asian","confidence":1.0},
              {"value":"young_age","confidence":1.0},
              {"value":"religion","confidence":1.0},
              {"value":"front_view","confidence":0.75}
            ]}"#,
        )
        .expect("schema remains valid");
        assert_eq!(
            parsed.tags,
            vec![ConfidentTag {
                value: "front_view".to_owned(),
                confidence: 0.75,
            }]
        );
    }

    #[test]
    fn invalid_medium_or_non_finite_out_of_range_confidence_fails_closed() {
        for payload in [
            r#"{"medium":{"value":"video","confidence":0.8},"tags":[]}"#,
            r#"{"medium":{"value":"photo","confidence":1.1},"tags":[]}"#,
            r#"{"medium":{"value":"photo","confidence":0.8},"tags":[{"value":"outdoor","confidence":-0.1}]}"#,
        ] {
            assert!(parse_normalized_vision_result(payload).is_err());
        }
    }

    #[test]
    fn schema_errors_reject_unknown_fields_without_echoing_generated_text() {
        for (payload, secret) in [
            (
                r#"{"medium":{"value":"private_person_name","confidence":0.8},"tags":[]}"#,
                "private_person_name",
            ),
            (
                r#"{"medium":{"value":"photo","confidence":0.8,"private_identity":"Alice"},"tags":[]}"#,
                "Alice",
            ),
            (
                r#"{"medium":{"value":"photo","confidence":0.8},"tags":[],"private_identity":"Bob"}"#,
                "Bob",
            ),
        ] {
            let error = parse_normalized_vision_result(payload).expect_err("schema must reject");
            assert!(
                !error.contains(secret),
                "failure metadata must not echo output"
            );
        }
    }

    #[test]
    fn failed_reanalysis_clears_stale_query_scalars_and_records_exact_runtime() {
        let mut record = survivor_record("failure");
        record.metadata["analysis"]["medium"] = json!("photograph");
        record.metadata["analysis"]["tagMembership"] = json!({"full_body": true});
        record.metadata["analysis"]["tagConfidence"] = json!({"full_body": 0.99});

        apply_vision_outcome(
            &mut record,
            "sha256:abc",
            Err(WorkerError::Engine(
                "vision response did not match the required schema".to_owned(),
            )),
        )
        .expect("failure persists");

        assert!(record.metadata.pointer("/analysis/medium").is_none());
        assert!(record.metadata.pointer("/analysis/tagMembership").is_none());
        assert!(record.metadata.pointer("/analysis/tagConfidence").is_none());
        let failure = record
            .metadata
            .pointer("/analysis/visionLanguage")
            .expect("failure metadata");
        assert_eq!(failure["runtimeRevision"], INFERENCE_RUNTIME_REVISION);
        assert_eq!(failure["backend"], inference_backend());
        assert_eq!(failure["provider"], vision_provider());
    }

    #[test]
    fn normalized_tag_confidence_is_bounded_and_queryable_by_dotted_path() {
        let temporary = tempfile::tempdir().expect("temporary");
        let registry = CatalogRegistry::new(temporary.path().join("state"));
        let root = temporary.path().join("catalog");
        let mut catalog = registry
            .create_catalog(&root, "confidence")
            .expect("catalog");
        let mut record = survivor_record("confidence");
        apply_vision_outcome(
            &mut record,
            "sha256:abc",
            Ok(NormalizedVisionResult {
                medium: ConfidentMedium {
                    value: VisualMedium::Photograph,
                    confidence: 0.87,
                },
                tags: vec![ConfidentTag {
                    value: "full_body".to_owned(),
                    confidence: 0.94,
                }],
            }),
        )
        .expect("success applies");
        catalog
            .append_records(&[record_update(record)])
            .expect("record persists");
        let confidence = catalog
            .page_records(0, 1)
            .expect("record reads")
            .pop()
            .expect("record")
            .metadata
            .pointer("/analysis/tagConfidence/full_body")
            .and_then(Value::as_f64)
            .expect("confidence scalar");
        assert!((0.0..=1.0).contains(&confidence));
        let filtered = catalog
            .query_records_after(
                None,
                10,
                &[CatalogRecordFilter {
                    field: "analysis.tagConfidence.full_body".to_owned(),
                    values: vec!["0.94".to_owned()],
                }],
            )
            .expect("dotted scalar query");
        assert_eq!(filtered.records.len(), 1);
    }

    #[test]
    fn survivor_and_freshness_gates_are_record_local_and_exact() {
        let mut record = survivor_record("r");
        record.embedding_path = Some("embeddings/clip-vit-l14/r.f32".to_owned());
        record.metadata["analysis"]["visionLanguage"] = json!({
            "status": "succeeded",
            "inputFingerprint": "sha256:abc",
            "analyzerVersion": VISION_ANALYZER_VERSION,
            "modelRevision": VISION_MODEL_REVISION,
            "taxonomyVersion": TAXONOMY_VERSION,
            "runtimeRevision": INFERENCE_RUNTIME_REVISION,
            "backend": inference_backend(),
            "provider": vision_provider(),
        });
        record.metadata["analysis"]["semanticEmbedding"] = json!({
            "status": "succeeded",
            "inputFingerprint": "sha256:abc",
            "analyzerVersion": EMBEDDING_ANALYZER_VERSION,
            "modelRevision": CLIP_MODEL_REVISION,
            "runtimeRevision": INFERENCE_RUNTIME_REVISION,
            "backend": inference_backend(),
            "provider": CLIP_PROVIDER,
        });
        assert!(is_structured_survivor(&record));
        assert!(vision_is_current(&record, "sha256:abc"));
        assert!(embedding_is_current(&record, "sha256:abc"));
        assert!(!vision_is_current(&record, "sha256:changed"));
        assert!(!embedding_is_current(&record, "sha256:changed"));

        for path in [
            "/analysis/visionLanguage/runtimeRevision",
            "/analysis/visionLanguage/backend",
            "/analysis/visionLanguage/provider",
            "/analysis/semanticEmbedding/runtimeRevision",
            "/analysis/semanticEmbedding/backend",
            "/analysis/semanticEmbedding/provider",
        ] {
            let mut stale = record.clone();
            *stale.metadata.pointer_mut(path).expect("provenance field") = json!("stale");
            assert!(
                !vision_is_current(&stale, "sha256:abc")
                    || !embedding_is_current(&stale, "sha256:abc"),
                "{path} must participate in freshness"
            );
        }
    }

    #[test]
    fn filtered_pointer_cleanup_immediately_reconciles_external_index() {
        let temporary = tempfile::tempdir().expect("temporary");
        let registry = CatalogRegistry::new(temporary.path().join("state"));
        let root = temporary.path().join("catalog");
        let mut catalog = registry.create_catalog(&root, "filtered").expect("catalog");
        let mut record = survivor_record("filtered");
        record.metadata["analysis"]["structured"]["derived"]["qualifiedSingleFullBody"] =
            json!(false);
        record.embedding_path = Some("embeddings/clip-vit-l14/stale.f32".to_owned());
        record.metadata["analysis"]["semanticEmbedding"] = json!({
            "status": "succeeded",
            "inputFingerprint": "sha256:abc",
            "digest": "sha256:stale",
            "dimension": 768,
            "space": CLIP_SPACE,
            "modelId": CLIP_MODEL_ID,
            "modelRevision": CLIP_MODEL_REVISION,
        });
        catalog
            .append_records(&[record_update(record)])
            .expect("record");
        let catalog_id = catalog.descriptor().id.clone();
        drop(catalog);

        let cancel = gen_core::CancelFlag::new();
        assert_eq!(
            reconcile_semantics_blocking(&registry, &catalog_id, false, &cancel)
                .expect("cleanup")
                .records_reconciled,
            1
        );
        let catalog = registry.open_attached(&catalog_id).expect("catalog");
        assert!(catalog
            .page_records(0, 1)
            .expect("record")
            .pop()
            .expect("record")
            .embedding_path
            .is_none());
        assert_eq!(
            fs::read_to_string(root.join("embeddings").join(CLIP_SPACE).join("index.jsonl"))
                .expect("index"),
            ""
        );
    }

    #[test]
    fn zero_survivors_require_no_clip_resolution_or_load() {
        let temporary = tempfile::tempdir().expect("temporary");
        let registry = CatalogRegistry::new(temporary.path().join("state"));
        let root = temporary.path().join("catalog");
        let mut catalog = registry.create_catalog(&root, "empty").expect("catalog");
        let mut filtered = survivor_record("filtered");
        filtered.metadata["analysis"]["structured"]["derived"]["qualifiedSingleFullBody"] =
            json!(false);
        catalog
            .append_records(&[record_update(filtered)])
            .expect("record");
        let catalog_id = catalog.descriptor().id.clone();
        drop(catalog);
        let report = reconcile_semantics_blocking(
            &registry,
            &catalog_id,
            true,
            &gen_core::CancelFlag::new(),
        )
        .expect("eligibility");
        assert!(!report.embedding_required);
    }

    #[test]
    fn changed_survivor_clears_stale_semantics_when_optional_analyzers_are_disabled() {
        let temporary = tempfile::tempdir().expect("temporary");
        let registry = CatalogRegistry::new(temporary.path().join("state"));
        let root = temporary.path().join("catalog");
        let mut catalog = registry.create_catalog(&root, "changed").expect("catalog");
        let mut record = survivor_record("changed");
        record.metadata["analysis"]["structured"]["inputFingerprint"] = json!("sha256:new");
        record.metadata["analysis"]["visionLanguage"] = json!({
            "status": "succeeded",
            "inputFingerprint": "sha256:old",
            "analyzerVersion": VISION_ANALYZER_VERSION,
            "modelRevision": VISION_MODEL_REVISION,
            "taxonomyVersion": TAXONOMY_VERSION,
            "runtimeRevision": INFERENCE_RUNTIME_REVISION,
            "backend": inference_backend(),
            "provider": vision_provider(),
        });
        record.metadata["analysis"]["medium"] = json!("photograph");
        record.metadata["analysis"]["tagMembership"] = json!({"full_body": true});
        record.metadata["analysis"]["tagConfidence"] = json!({"full_body": 0.9});
        record.metadata["analysis"]["semanticEmbedding"] = json!({
            "status": "succeeded",
            "inputFingerprint": "sha256:old",
            "analyzerVersion": EMBEDDING_ANALYZER_VERSION,
            "modelRevision": CLIP_MODEL_REVISION,
            "runtimeRevision": INFERENCE_RUNTIME_REVISION,
            "backend": inference_backend(),
            "provider": CLIP_PROVIDER,
        });
        record.embedding_path = Some("embeddings/clip-vit-l14/old.f32".to_owned());
        catalog
            .append_records(&[record_update(record)])
            .expect("record");
        let catalog_id = catalog.descriptor().id.clone();
        drop(catalog);

        let report = reconcile_semantics_blocking(
            &registry,
            &catalog_id,
            false,
            &gen_core::CancelFlag::new(),
        )
        .expect("reconcile");
        assert_eq!(report.records_reconciled, 1);
        assert!(!report.embedding_required);
        let record = registry
            .open_attached(&catalog_id)
            .expect("catalog")
            .page_records(0, 1)
            .expect("record")
            .remove(0);
        assert!(record.embedding_path.is_none());
        for path in [
            "/analysis/visionLanguage",
            "/analysis/medium",
            "/analysis/tagMembership",
            "/analysis/tagConfidence",
            "/analysis/semanticEmbedding",
        ] {
            assert!(record.metadata.pointer(path).is_none(), "{path} is stale");
        }
    }

    #[test]
    fn clip_failure_clears_pointer_and_persists_exact_runtime_backend_provider() {
        let temporary = tempfile::tempdir().expect("temporary");
        let registry = CatalogRegistry::new(temporary.path().join("state"));
        let root = temporary.path().join("catalog");
        let mut catalog = registry
            .create_catalog(&root, "clip-failure")
            .expect("catalog");
        let mut record = survivor_record("clip-failure");
        record.embedding_path = Some("embeddings/clip-vit-l14/stale.f32".to_owned());
        record.metadata["analysis"]["semanticEmbedding"] = json!({
            "status": "succeeded",
            "inputFingerprint": "sha256:abc",
            "digest": "sha256:stale",
            "dimension": 768,
            "space": CLIP_SPACE,
            "modelId": CLIP_MODEL_ID,
            "modelRevision": CLIP_MODEL_REVISION,
        });
        catalog
            .append_records(&[record_update(record.clone())])
            .expect("record");
        let catalog_id = catalog.descriptor().id.clone();
        drop(catalog);
        rebuild_embedding_index_streaming(&registry, &catalog_id, None).expect("stale index");

        persist_embedding_failure(
            &registry,
            &catalog_id,
            record,
            "sha256:abc",
            "generic embedding failure",
        )
        .expect("failure persists");
        let record = registry
            .open_attached(&catalog_id)
            .expect("catalog")
            .page_records(0, 1)
            .expect("record")
            .pop()
            .expect("record");
        assert!(record.embedding_path.is_none());
        let failure = record
            .metadata
            .pointer("/analysis/semanticEmbedding")
            .expect("failure");
        assert_eq!(failure["runtimeRevision"], INFERENCE_RUNTIME_REVISION);
        assert_eq!(failure["backend"], inference_backend());
        assert_eq!(failure["provider"], CLIP_PROVIDER);

        let terminal = finish_with_index_reconciliation::<()>(
            Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned())),
            &registry,
            &catalog_id,
        )
        .expect_err("cancel remains terminal");
        assert!(matches!(terminal, WorkerError::Canceled(_)));
        assert_eq!(
            fs::read_to_string(root.join("embeddings").join(CLIP_SPACE).join("index.jsonl"))
                .expect("index"),
            "",
            "index is reconciled even when the embedding pass ends canceled"
        );
    }

    #[test]
    fn multi_page_reconcile_and_index_are_checkpointed_and_cancel_aware() {
        let temporary = tempfile::tempdir().expect("temporary");
        let registry = CatalogRegistry::new(temporary.path().join("state"));
        let root = temporary.path().join("catalog");
        let mut catalog = registry
            .create_catalog(&root, "multi-page")
            .expect("catalog");
        let records = (0..=PAGE_SIZE)
            .map(|index| NewCatalogRecord {
                id: format!("record-{index:04}"),
                image_path: format!("images/{index:04}.jpg"),
                thumbnail_path: None,
                embedding_path: None,
                artifact_path: None,
                metadata: json!({
                    "analysis": {
                        "structured": {
                            "inputFingerprint": format!("sha256:{index:04}"),
                            "derived": {"qualifiedSingleFullBody": false}
                        }
                    }
                }),
            })
            .collect::<Vec<_>>();
        catalog.append_records(&records).expect("records");
        let catalog_id = catalog.descriptor().id.clone();
        drop(catalog);

        let report = reconcile_semantics_blocking(
            &registry,
            &catalog_id,
            true,
            &gen_core::CancelFlag::new(),
        )
        .expect("multi-page reconcile");
        assert_eq!(report.records_reconciled, u64::from(PAGE_SIZE) + 1);
        assert!(!report.embedding_required);
        let checkpoint = registry
            .open_attached(&catalog_id)
            .expect("catalog")
            .contract_state()
            .expect("state")
            .checkpoints[SEMANTIC_CHECKPOINT_KEY]
            .clone();
        assert_eq!(checkpoint["stage"], "embedding_index");
        assert_eq!(checkpoint["examined"], u64::from(PAGE_SIZE) + 1);
        assert!(checkpoint["nextCursor"].is_null());

        let cancel = gen_core::CancelFlag::new();
        cancel.cancel();
        let error = reconcile_semantics_blocking(&registry, &catalog_id, true, &cancel)
            .expect_err("pre-canceled reconcile");
        assert!(matches!(error, WorkerError::Canceled(_)));
        assert!(
            root.join("embeddings")
                .join(CLIP_SPACE)
                .join("index.jsonl")
                .is_file(),
            "cancel still leaves a reconciled atomic index"
        );
    }

    #[test]
    fn catalog_pipeline_wires_fetch_and_blocking_analysis_keepalives() {
        let source = include_str!("catalog_semantic_jobs.rs");
        let fetch = source
            .split("async fn fetch_catalog_images_with_heartbeat")
            .nth(1)
            .and_then(|body| body.split("fn run_embedding_pass_blocking").next())
            .expect("fetch helper source");
        assert!(fetch.contains("crate::heartbeat("));
        assert!(fetch.contains("crate::check_cancel("));
        let handler = source
            .split("pub(crate) async fn run_catalog_analysis_job")
            .nth(1)
            .expect("handler source");
        assert!(handler.contains("crate::run_blocking_with_heartbeat("));
        assert!(handler.contains("analyze_attached_catalog_under_processing_lease_with_control"));
        assert!(handler.contains("\"catalog semantic reconciliation\""));
    }

    #[test]
    fn vectors_are_normalized_and_encoded_little_endian() {
        let vector = l2_normalize(vec![3.0, 4.0]).expect("normalizes");
        assert!((vector[0] - 0.6).abs() < 1e-6);
        assert!((vector[1] - 0.8).abs() < 1e-6);
        assert_eq!(&vector_bytes(&[1.0])[..], &1.0_f32.to_le_bytes());
        assert!(l2_normalize(vec![0.0, 0.0]).is_err());
        assert!(l2_normalize(vec![f32::NAN]).is_err());
    }
}
