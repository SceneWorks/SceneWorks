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
const CLIP_SPACE: &str = "clip-vit-l14";
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
struct RawVisionResult {
    medium: RawMedium,
    #[serde(default)]
    tags: Vec<RawTag>,
}

#[derive(Deserialize)]
struct RawMedium {
    value: String,
    confidence: f64,
}

#[derive(Deserialize)]
struct RawTag {
    value: String,
    confidence: f64,
}

pub fn parse_normalized_vision_result(text: &str) -> Result<NormalizedVisionResult, String> {
    let cleaned = clean_json_output(text);
    let raw: RawVisionResult =
        serde_json::from_str(&cleaned).map_err(|error| format!("invalid vision JSON: {error}"))?;
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
        value => return Err(format!("unsupported visual medium: {value}")),
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

fn valid_confidence(value: f64, field: &str) -> Result<f64, String> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err(format!(
            "{field} confidence must be finite and between zero and one"
        ))
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

fn clear_filtered_semantics(registry: &CatalogRegistry, catalog_id: &str) -> WorkerResult<u64> {
    let mut cursor = None;
    let mut cleared = 0_u64;
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
            if is_structured_survivor(&record) {
                continue;
            }
            let structured_fingerprint = structured_input_fingerprint(&record).map(str::to_owned);
            let mut changed = record.embedding_path.take().is_some();
            if let Some(analysis) = record
                .metadata
                .get_mut("analysis")
                .and_then(Value::as_object_mut)
            {
                for key in [
                    "visionLanguage",
                    "semanticEmbedding",
                    "medium",
                    "tagMembership",
                ] {
                    changed |= analysis.remove(key).is_some();
                }
                let selection = json!({
                    "status": "filtered_out",
                    "predicate": "structured.derived.qualifiedSingleFullBody",
                    "structuredInputFingerprint": structured_fingerprint,
                });
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
                cleared = cleared.saturating_add(1);
            }
        }
        cursor = page.next_cursor;
        if cursor.is_none() {
            break;
        }
    }
    Ok(cleared)
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
    let mut bytes = Vec::with_capacity(vector.len() * std::mem::size_of::<f32>());
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
            let analysis = metadata_analysis_mut(&mut record.metadata)?;
            match outcome {
                Ok(result) => {
                    let membership = result
                        .tags
                        .iter()
                        .map(|tag| (tag.value.clone(), Value::Bool(true)))
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
                            "medium": result.medium,
                            "tags": result.tags,
                        }),
                    );
                    analysis.insert("medium".to_owned(), json!(result.medium.value));
                    analysis.insert("tagMembership".to_owned(), Value::Object(membership));
                }
                Err(error) => {
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
                            "error": bounded_error(error),
                        }),
                    );
                }
            }
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
                        "runtimeRevision": "1d80161ad9c4a86445725e4dab69ba7b460f4101",
                        "embedderId": CLIP_EMBEDDER_ID,
                        "backend": descriptor.backend,
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
    rebuild_embedding_index(&registry, &catalog_id)?;
    Ok((examined, updated))
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
            "runtimeRevision": "1d80161ad9c4a86445725e4dab69ba7b460f4101",
            "embedderId": CLIP_EMBEDDER_ID,
            "error": bounded_error(error),
        }),
    );
    registry
        .open_attached(catalog_id)
        .and_then(|mut catalog| catalog.append_records(&[record_update(record)]))
        .map(|_| ())
        .map_err(|error| WorkerError::InvalidPayload(error.to_string()))
}

fn rebuild_embedding_index(registry: &CatalogRegistry, catalog_id: &str) -> WorkerResult<()> {
    let catalog = registry
        .open_attached(catalog_id)
        .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?;
    let mut cursor = None;
    let mut lines = Vec::new();
    loop {
        let page = catalog
            .page_records_after(cursor, PAGE_SIZE)
            .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?;
        for record in page.records {
            let Some(embedding) = record.metadata.pointer("/analysis/semanticEmbedding") else {
                continue;
            };
            if embedding.get("status").and_then(Value::as_str) != Some("succeeded") {
                continue;
            }
            lines.push(
                serde_json::to_string(&json!({
                    "recordId": record.id,
                    "path": record.embedding_path,
                    "inputFingerprint": embedding.get("inputFingerprint"),
                    "digest": embedding.get("digest"),
                    "dimension": embedding.get("dimension"),
                    "space": embedding.get("space"),
                    "modelId": embedding.get("modelId"),
                    "modelRevision": embedding.get("modelRevision"),
                }))
                .map_err(|error| WorkerError::Engine(error.to_string()))?,
            );
        }
        cursor = page.next_cursor;
        if cursor.is_none() {
            break;
        }
    }
    lines.sort();
    let mut payload = lines.join("\n");
    if !payload.is_empty() {
        payload.push('\n');
    }
    atomic_write(
        &catalog
            .root()
            .join("embeddings")
            .join(CLIP_SPACE)
            .join("index.jsonl"),
        payload.as_bytes(),
    )
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
        let client = crate::downloads::streaming_download_client();
        let context = crate::downloads::DownloadContext {
            api,
            client: &client,
            settings,
            job_id: &job.id,
            cancel_message: CANCEL_MESSAGE,
            fresh_download: false,
        };
        let person_detector =
            crate::person_jobs::ensure_detector_weights(settings, &context).await?;
        let face_weights_directory =
            crate::image_jobs::ensure_face_stack_dir(api, settings, job).await?;
        let (pose_detector, pose_model) =
            crate::pose_jobs::ensure_dwpose_weights(api, settings, &client, job).await?;
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
    let clip_weights = analyzer_config
        .settings
        .semantic_embeddings_enabled
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
    let mut fetch_options = crate::catalog_image_fetch::CatalogImageFetchOptions::default();
    fetch_options.accepted_target = record_count;
    let fetch = crate::catalog_image_fetch::fetch_attached_catalog_images_under_processing_lease(
        &registry,
        &catalog_id,
        &fetch_options,
        processing.as_ref(),
    )
    .await
    .map_err(|error| WorkerError::Engine(format!("catalog image fetch failed: {error}")))?;

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
        Some(
            tokio::task::spawn_blocking({
                let registry = registry.clone();
                let catalog_id = catalog_id.clone();
                move || {
                    crate::catalog_analysis::analyze_attached_catalog_under_processing_lease(
                        &registry,
                        &catalog_id,
                        &config,
                        &mut analyzers,
                        structured_processing.as_ref(),
                    )
                }
            })
            .await
            .map_err(|error| crate::task_join_error("catalog structured analysis", error))?
            .map_err(|error| WorkerError::Engine(error.to_string()))?,
        )
    } else {
        None
    };

    let filtered_semantic_records = clear_filtered_semantics(&registry, &catalog_id)?;

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
            run_embedding_pass_blocking(
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
    fn survivor_and_freshness_gates_are_record_local_and_exact() {
        let record = CatalogRecord {
            id: "r".to_owned(),
            image_path: "images/r.jpg".to_owned(),
            thumbnail_path: None,
            embedding_path: Some("embeddings/clip-vit-l14/r.f32".to_owned()),
            artifact_path: None,
            metadata: json!({
                "analysis": {
                    "structured": {
                        "inputFingerprint": "sha256:abc",
                        "derived": {"qualifiedSingleFullBody": true}
                    },
                    "visionLanguage": {
                        "status": "succeeded",
                        "inputFingerprint": "sha256:abc",
                        "analyzerVersion": VISION_ANALYZER_VERSION,
                        "modelRevision": VISION_MODEL_REVISION,
                        "taxonomyVersion": TAXONOMY_VERSION
                    },
                    "semanticEmbedding": {
                        "status": "succeeded",
                        "inputFingerprint": "sha256:abc",
                        "analyzerVersion": EMBEDDING_ANALYZER_VERSION,
                        "modelRevision": CLIP_MODEL_REVISION
                    }
                }
            }),
            created_at: String::new(),
            updated_at: String::new(),
        };
        assert!(is_structured_survivor(&record));
        assert!(vision_is_current(&record, "sha256:abc"));
        assert!(embedding_is_current(&record, "sha256:abc"));
        assert!(!vision_is_current(&record, "sha256:changed"));
        assert!(!embedding_is_current(&record, "sha256:changed"));
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
