//! ProjectStore asset sidecars aggregate.

use super::*;

/// Assemble the on-disk asset sidecar from the worker-reported flat facts. Rust
/// is the single owner of this envelope schema now (story 1656): the worker
/// ships values (paths, dimensions, seed, recipe inputs) and Rust builds the
/// `file`/`status`/`recipe`/`lineage` structure, matching the shape pinned by
/// `resource_sidecars.json` and the worker's former Python `build_asset_sidecar`.
/// `type` is derived from the mime so the video slice can reuse this.
pub(super) fn build_generated_asset_sidecar(
    project_id: &str,
    job_id: &str,
    generation_set_id: &str,
    fact: &Value,
) -> Value {
    let get = |key: &str| fact.get(key).cloned().unwrap_or(Value::Null);
    // The worker sends an explicit `type` (a procedural video preview is a .webp,
    // so mime alone can't classify it); fall back to mime for image facts that
    // predate it.
    let mime = fact
        .get("mimeType")
        .and_then(Value::as_str)
        .unwrap_or("image/png");
    let media_type = fact
        .get("type")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| {
            if mime.starts_with("video/") {
                "video".to_owned()
            } else {
                "image".to_owned()
            }
        });
    let (file, recipe, lineage) = match media_type.as_str() {
        "video" => build_video_sidecar_parts(job_id, fact),
        "audio" => build_audio_sidecar_parts(job_id, fact),
        "document" => build_document_sidecar_parts(job_id, fact),
        "vector" => build_vector_sidecar_parts(job_id, fact),
        "depth" => build_depth_sidecar_parts(job_id, fact),
        _ => build_image_sidecar_parts(job_id, fact),
    };
    let mut asset = json!({
        "schemaVersion": 1,
        "id": get("assetId"),
        "projectId": project_id,
        "generationSetId": generation_set_id,
        "type": media_type,
        "displayName": get("displayName"),
        "createdAt": get("createdAt"),
        "file": file,
        "status": { "favorite": false, "rating": 0, "rejected": false, "trashed": false },
        "recipe": recipe,
        "lineage": lineage,
    });
    // The API replaces this fact from its immutable job/plan binding before any producer reaches
    // this common sidecar boundary. Keep the same projection in the saved recipe and asset row.
    if let Some(provenance) = fact.get(crate::model_artifacts::provenance::MODEL_PROVENANCE_KEY) {
        asset["modelProvenance"] = provenance.clone();
        asset["recipe"]["modelProvenance"] = provenance.clone();
    }
    // Record the originating studio so the Asset Library can exclude Character
    // Studio test outputs (sc-2024). Derived from recipe mode + media type.
    let origin = asset_origin(&asset);
    if let Some(object) = asset.as_object_mut() {
        object.insert("origin".to_owned(), Value::String(origin));
    }
    if let Some(extra) = fact.get("extra") {
        if let Some(object) = asset.as_object_mut() {
            object.insert("extra".to_owned(), extra.clone());
        }
    }
    // The API strips this field from every worker report and restores it only from the persisted
    // server-owned child relationship. Applying it during sidecar construction closes the window
    // in which a composed workflow's unpublished raster could appear in the Asset Library.
    if media_type == "image" {
        if let Some(ownership) = fact
            .get("vectorWorkflowOwnership")
            .and_then(Value::as_object)
        {
            let workflow_id = ownership.get("workflowId").and_then(Value::as_str);
            let parent_job_id = ownership.get("parentJobId").and_then(Value::as_str);
            let child_job_id = ownership.get("childJobId").and_then(Value::as_str);
            let valid = [workflow_id, parent_job_id, child_job_id]
                .into_iter()
                .all(|id| id.is_some_and(is_safe_id))
                && ownership.get("role").and_then(Value::as_str) == Some("retained_intermediate")
                && ownership.get("publication").and_then(Value::as_str) == Some("unpublished")
                && ownership.get("hidden").and_then(Value::as_bool) == Some(true);
            if valid {
                let object = asset.as_object_mut().expect("generated asset is an object");
                object.insert(
                    "origin".to_owned(),
                    Value::String("vector_workflow_intermediate".to_owned()),
                );
                let extra = object.entry("extra").or_insert_with(|| json!({}));
                if !extra.is_object() {
                    *extra = json!({});
                }
                extra
                    .as_object_mut()
                    .expect("workflow asset extra is an object")
                    .insert(
                        "vectorWorkflow".to_owned(),
                        Value::Object(ownership.clone()),
                    );
            }
        }
    }
    // SVG is never a browser-previewable media type. The worker supplies a separately stored,
    // bounded PNG fact after sanitizing/rasterizing the SVG; keep it on the vector sidecar rather
    // than asking clients to infer a sibling filename or to fetch the active source document.
    //
    // A depth map (sc-25682) has the same split: its `file` is the raw float32 `.npy`, never
    // rewritten, and the `preview` is a separately stored PNG derived from it. Its `depth` block is
    // the reload contract — value convention, layout, preprocessing and model/config identity — so a
    // consumer can read the map without knowing which job or model produced it.
    if media_type == "vector" || media_type == "depth" {
        if let Some(preview) = fact.get("preview") {
            if let Some(object) = asset.as_object_mut() {
                object.insert("preview".to_owned(), preview.clone());
            }
        }
    }
    if media_type == "tensor" {
        if let Some(tensor) = fact.get("tensor") {
            asset["tensor"] = tensor.clone();
            if let Some(hash) = tensor.get("sha256").and_then(Value::as_str) {
                asset["file"]["contentHash"] = json!(format!("sha256:{hash}"));
            }
        }
        if let Some(index) = fact.get("batchIndex") {
            asset["batchIndex"] = index.clone();
        }
    }
    if media_type == "depth" {
        if let Some(depth) = fact.get("depth") {
            if let Some(object) = asset.as_object_mut() {
                object.insert("depth".to_owned(), depth.clone());
            }
        }
    }
    asset
}

/// `(file, recipe, lineage)` for a depth-map asset (Iris-3B, sc-25682). The file is the raw `.npy`
/// at the EXIF-corrected source size; the recipe records the job's own settings
/// (`rawAdapterSettings.depth`) so the run is reopenable; lineage names the source image.
fn build_depth_sidecar_parts(job_id: &str, fact: &Value) -> (Value, Value, Value) {
    let get = |key: &str| fact.get(key).cloned().unwrap_or(Value::Null);
    let parents = fact
        .get("parents")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_else(|| match fact.get("sourceAssetId").and_then(Value::as_str) {
            Some(source) => vec![Value::String(source.to_owned())],
            None => Vec::new(),
        });
    let file = json!({
        "path": get("mediaPath"),
        "mimeType": get("mimeType"),
        "width": get("width"),
        "height": get("height"),
        "duration": Value::Null,
        "fps": Value::Null,
    });
    let recipe = json!({
        "mode": get("mode"),
        "model": get("model"),
        "adapter": get("adapter"),
        "prompt": "",
        "negativePrompt": "",
        "seed": Value::Null,
        "loras": [],
        "stylePreset": Value::Null,
        "normalizedSettings": {
            "width": get("width"),
            "height": get("height"),
            "count": 1,
        },
        "rawAdapterSettings": fact.get("rawAdapterSettings").cloned().unwrap_or_else(|| json!({})),
    });
    let lineage = json!({
        "parents": parents,
        "sourceAssetId": get("sourceAssetId"),
        "sourceTimestamp": Value::Null,
        "jobId": job_id,
    });
    (file, recipe, lineage)
}

/// A list-valued id key as an array, even on facts written before the key existed, so a replay
/// reader never has to distinguish "absent" from "empty" (sc-12345).
fn fact_id_list(fact: &Value, key: &str) -> Value {
    match fact.get(key) {
        Some(value @ Value::Array(_)) => value.clone(),
        _ => json!([]),
    }
}

/// The imported checkpoint a generation selected (sc-26687), or `None` when it ran from a builtin.
///
/// `recipe.model` names the selection; this records WHICH import answered it, so a replay or an API
/// consumer re-resolves the same checkpoint rather than a same-named builtin. Every field is read
/// off the worker's own fact — the plan identity the imported route stamped into
/// `rawAdapterSettings.importPlan` at load time, and the assembly partition beside it. Nothing is
/// synthesized from a catalog entry: a fact carrying no plan identity is not an imported
/// generation and gets no `importedModelRef` at all, rather than an object claiming "an import,
/// unnamed".
fn imported_model_ref(fact: &Value) -> Option<Map<String, Value>> {
    let model = fact
        .get("model")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())?;
    let plan = fact
        .pointer("/rawAdapterSettings/importPlan")
        .and_then(Value::as_object);
    let checkpoint_id = plan
        .and_then(|plan| plan.get("checkpointId"))
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())?;
    let mut reference = Map::new();
    reference.insert("modelId".to_owned(), Value::String(model.to_owned()));
    if let Some(family) = fact
        .get("family")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        reference.insert("family".to_owned(), Value::String(family.to_owned()));
    }
    // The declared task partition, taken from the bound assembly the route recorded — never from
    // a filename or a directory name, which prove nothing about task compatibility.
    if let Some(partition) = fact
        .pointer("/rawAdapterSettings/importPlan/assembly/partition")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        reference.insert(
            "declaredPartition".to_owned(),
            Value::String(partition.to_owned()),
        );
    }
    if let Some(base_model) = fact
        .get("baseModel")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        reference.insert("baseModel".to_owned(), Value::String(base_model.to_owned()));
    }
    reference.insert(
        "checkpointId".to_owned(),
        Value::String(checkpoint_id.to_owned()),
    );
    if let Some(plan_id) = plan
        .and_then(|plan| plan.get("planId"))
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        reference.insert("planId".to_owned(), Value::String(plan_id.to_owned()));
    }
    if let Some(semantic_digest) = plan
        .and_then(|plan| plan.get("semanticDigest"))
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        reference.insert(
            "semanticDigest".to_owned(),
            Value::String(semantic_digest.to_owned()),
        );
    }
    Some(reference)
}

/// Attach the imported-checkpoint identity to a recipe when the generation carried one.
fn stamp_imported_model_ref(recipe: &mut Value, fact: &Value) {
    if let (Some(reference), Some(object)) = (imported_model_ref(fact), recipe.as_object_mut()) {
        object.insert("importedModelRef".to_owned(), Value::Object(reference));
    }
}

/// `(file, recipe, lineage)` for a generated image asset, matching the worker's
/// former `build_asset_sidecar` (story 1656).
pub(super) fn build_image_sidecar_parts(job_id: &str, fact: &Value) -> (Value, Value, Value) {
    let get = |key: &str| fact.get(key).cloned().unwrap_or(Value::Null);
    let parents = fact
        .get("parents")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_else(|| match fact.get("sourceAssetId").and_then(Value::as_str) {
            Some(source) => vec![Value::String(source.to_owned())],
            None => Vec::new(),
        });
    let file = json!({
        "path": get("mediaPath"),
        "mimeType": get("mimeType"),
        "width": get("width"),
        "height": get("height"),
        "duration": Value::Null,
        "fps": Value::Null,
    });
    let mut recipe = json!({
        "mode": get("mode"),
        "model": get("model"),
        "adapter": get("adapter"),
        "prompt": get("prompt"),
        "negativePrompt": get("negativePrompt"),
        "seed": get("seed"),
        "loras": fact.get("loras").cloned().unwrap_or_else(|| json!([])),
        "stylePreset": get("stylePreset"),
        "normalizedSettings": {
            "width": get("normalizedWidth"),
            "height": get("normalizedHeight"),
            "count": get("count"),
            "family": get("family"),
            "characterId": get("characterId"),
            "characterLookId": get("characterLookId"),
            "characterConditioningActive": false,
            "characterConditioningNote": "Character metadata is recorded, but adapter-level character conditioning is not active in this build.",
            // sc-26687: the ORDERED image conditioning the replay path restores. `referenceAssetIds`
            // is a list whose element order is semantic — the engine labels the supplied images
            // positionally — so it is recorded verbatim, never sorted or de-duplicated. The image
            // recipe recorded neither the edit source nor the reference list before this, so
            // "re-run this generation" silently rebuilt a job with different conditioning.
            "sourceAssetId": get("sourceAssetId"),
            "referenceAssetId": get("referenceAssetId"),
            "referenceAssetIds": fact_id_list(fact, "referenceAssetIds"),
        },
        "rawAdapterSettings": fact.get("rawAdapterSettings").cloned().unwrap_or_else(|| json!({})),
    });
    stamp_imported_model_ref(&mut recipe, fact);
    let lineage = json!({
        "parents": parents,
        "sourceAssetId": get("sourceAssetId"),
        "sourceTimestamp": Value::Null,
        "jobId": job_id,
    });
    (file, recipe, lineage)
}

/// Vector Studio replay is intentionally explicit rather than inheriting the image recipe's
/// width/height/LoRA shape. These are the complete generation inputs and the two versioned
/// post-processing boundaries required to reproduce or explain a stored SVG.
fn build_vector_sidecar_parts(job_id: &str, fact: &Value) -> (Value, Value, Value) {
    let get = |key: &str| fact.get(key).cloned().unwrap_or(Value::Null);
    let source_asset_id = get("sourceAssetId");
    let parents = source_asset_id
        .as_str()
        .map(|source| vec![Value::String(source.to_owned())])
        .unwrap_or_default();
    let file = json!({
        "path": get("mediaPath"),
        "mimeType": get("mimeType"),
        "width": get("width"),
        "height": get("height"),
        "duration": Value::Null,
        "fps": Value::Null,
    });
    let recipe = json!({
        "mode": get("mode"),
        "model": get("model"),
        "adapter": get("adapter"),
        "prompt": get("prompt"),
        "sampling": get("sampling"),
        "detailBudget": get("detailBudget"),
        "sanitizerVersion": get("sanitizerVersion"),
        "rendererVersion": get("rendererVersion"),
        "workflow": get("workflow"),
    });
    let workflow_id = fact.pointer("/workflow/id").cloned().unwrap_or(Value::Null);
    let lineage = json!({
        "parents": parents,
        "sourceAssetId": source_asset_id,
        "sourceTimestamp": Value::Null,
        "jobId": job_id,
        "workflowId": workflow_id,
        "intermediateAssetId": get("sourceAssetId"),
    });
    (file, recipe, lineage)
}

/// `(file, recipe, lineage)` for a generated video asset, matching the worker's
/// former `build_video_asset_sidecar` (story 1656): video file carries
/// duration/fps, the recipe has no `stylePreset`, normalizedSettings are
/// video-shaped (and fold in the honest `replacement_status` for replace_person),
/// and lineage tracks the four source clip/frame ids + the timeline context.
pub(super) fn build_video_sidecar_parts(job_id: &str, fact: &Value) -> (Value, Value, Value) {
    let get = |key: &str| fact.get(key).cloned().unwrap_or(Value::Null);
    // List-valued ids keep an array shape even on facts written before sc-12345, so the replay
    // reader never has to distinguish "absent" from "empty".
    let list_or_empty = |key: &str| fact_id_list(fact, key);
    let source_keys = [
        "sourceAssetId",
        "lastFrameAssetId",
        "sourceClipAssetId",
        "bridgeRightClipAssetId",
    ];
    // The list-valued source ids are parents too — mv2v's clips, the reference-driven modes'
    // subject images, and ads2v's reference video. Without them those modes' provenance names
    // only a subset of what the clip was actually derived from (sc-12345).
    // sc-17160 adds the audio references: media the clip was genuinely derived from, so leaving
    // them out would name only a subset of its provenance — the same gap sc-12345 closed for the
    // clip and image lists.
    let list_source_keys = [
        "sourceClipAssetIds",
        "referenceAssetIds",
        "referenceAudioAssetIds",
    ];
    let parents: Vec<Value> = source_keys
        .iter()
        .chain(std::iter::once(&"referenceClipAssetId"))
        .filter_map(|key| fact.get(*key).and_then(Value::as_str))
        .map(|id| Value::String(id.to_owned()))
        .chain(list_source_keys.iter().flat_map(|key| {
            fact.get(*key)
                .and_then(Value::as_array)
                .map(|ids| {
                    ids.iter()
                        .filter_map(Value::as_str)
                        .map(|id| Value::String(id.to_owned()))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        }))
        .collect();
    // sc-12371: `file` describes the mp4 on disk, so it takes the worker's MEASURED clip facts —
    // counted off the frames that were encoded, not the duration/fps the user asked for. Those
    // differ whenever the model's temporal stride snaps the frame count, which is usually: a 6 s x
    // 25 fps ask on a Wan engine renders 149 frames = 5.96 s, and `file.duration` claimed 6.0.
    //
    // Falls back to the requested values for facts written before the worker measured (assets
    // already on disk, and the non-video callers of this shape) — a slightly-wrong duration on an
    // old asset beats a missing one, and it is what those assets have always reported.
    let measured = |encoded: &str, requested: &str| match fact.get(encoded) {
        Some(value) if !value.is_null() => value.clone(),
        _ => get(requested),
    };
    let mut file = json!({
        "path": get("mediaPath"),
        "mimeType": get("mimeType"),
        "width": get("width"),
        "height": get("height"),
        "duration": measured("encodedDuration", "duration"),
        "fps": measured("encodedFps", "fps"),
        "frameCount": fact.get("encodedFrameCount").cloned().unwrap_or(Value::Null),
    });
    // sc-19577: whether the mp4 carries a soundtrack, measured off what the worker actually MUXED.
    //
    // Written ONLY when the fact carries it, and deliberately not defaulted to `false`. Every asset
    // generated before sc-19577 has no such fact, and a `false` there would be a fresh claim that
    // those clips are silent — which for the LTX renders among them is untrue. Absent means UNKNOWN
    // and the UI shows nothing; `false` means measured-and-silent. Same shape `frameCount` would
    // have needed had a wrong default been possible for it.
    if let (Some(has_audio), Some(object)) = (
        fact.get("hasAudio").filter(|value| value.is_boolean()),
        file.as_object_mut(),
    ) {
        object.insert("hasAudio".to_owned(), has_audio.clone());
    }
    // ...whereas `normalizedSettings` is the REPLAY record: the knobs the user picked off the
    // model's `limits.durations` / `limits.fps` menus, which "re-run this generation" (sc-12324 /
    // sc-12345) rebuilds the payload from. It must round-trip the ask, not the measurement.
    let mut normalized = json!({
        "duration": get("duration"),
        "fps": get("fps"),
        "width": get("width"),
        "height": get("height"),
        "quality": get("quality"),
        "family": get("family"),
        "sourceAssetId": get("sourceAssetId"),
        "lastFrameAssetId": get("lastFrameAssetId"),
        "sourceClipAssetId": get("sourceClipAssetId"),
        "bridgeRightClipAssetId": get("bridgeRightClipAssetId"),
        // sc-12345: the fit and the list-valued source ids the replay path needs. Older facts
        // predate them — `get` yields null and the studio falls back to its own defaults.
        "fitMode": get("fitMode"),
        "sourceClipAssetIds": list_or_empty("sourceClipAssetIds"),
        "referenceAssetIds": list_or_empty("referenceAssetIds"),
        // sc-17160. `list_or_empty` keeps the array shape on facts written before this key
        // existed, so the replay reader never distinguishes "absent" from "empty".
        "referenceAudioAssetIds": list_or_empty("referenceAudioAssetIds"),
        "referenceClipAssetId": get("referenceClipAssetId"),
        "characterId": get("characterId"),
        "characterLookId": get("characterLookId"),
        "personTrackId": get("personTrackId"),
        "replacementMode": get("replacementMode"),
        "timelineContextRef": "lineage.timeline",
    });
    if fact.get("mode").and_then(Value::as_str) == Some("replace_person") {
        if let Some(object) = normalized.as_object_mut() {
            object.insert("personDetectionActive".to_owned(), Value::Bool(false));
            object.insert("personTrackingActive".to_owned(), Value::Bool(false));
            object.insert("replacementActive".to_owned(), Value::Bool(false));
            if let Some(status) = fact.get("replacementStatus").and_then(Value::as_object) {
                for (key, value) in status {
                    object.insert(key.clone(), value.clone());
                }
            }
        }
    }
    let mut recipe = json!({
        "mode": get("mode"),
        "model": get("model"),
        "adapter": get("adapter"),
        "prompt": get("prompt"),
        "negativePrompt": get("negativePrompt"),
        "seed": get("seed"),
        "loras": fact.get("loras").cloned().unwrap_or_else(|| json!([])),
        "normalizedSettings": normalized,
        "rawAdapterSettings": fact.get("rawAdapterSettings").cloned().unwrap_or_else(|| json!({})),
    });
    stamp_imported_model_ref(&mut recipe, fact);
    let lineage = json!({
        "parents": parents,
        "sourceAssetId": get("sourceAssetId"),
        "lastFrameAssetId": get("lastFrameAssetId"),
        "sourceClipAssetId": get("sourceClipAssetId"),
        "bridgeRightClipAssetId": get("bridgeRightClipAssetId"),
        "personTrackId": get("personTrackId"),
        "characterId": get("characterId"),
        "characterLookId": get("characterLookId"),
        "replacementMode": get("replacementMode"),
        "sourceTimestamp": Value::Null,
        "timeline": fact.get("timelineContext").cloned().unwrap_or_else(|| json!({})),
        "jobId": job_id,
    });
    (file, recipe, lineage)
}

/// `(file, recipe, lineage)` for a generated audio asset (SceneWorks Audio Studio, epic 13400 /
/// sc-13404) — the audio twin of [`build_video_sidecar_parts`]. The worker writes a WAV (via
/// `write_wav_pcm16`) and reports the measured `duration`/`sampleRate`/`channels` off the produced
/// `AudioTrack`, so the `file` block is honest about what is on disk; the recipe's
/// `normalizedSettings` round-trips the ASK (voice / language / targetDuration) so "re-run this
/// generation" can rebuild the payload. A pure-audio asset has no width/height (unlike an image or
/// video), so those are null.
fn build_audio_sidecar_parts(job_id: &str, fact: &Value) -> (Value, Value, Value) {
    let get = |key: &str| fact.get(key).cloned().unwrap_or(Value::Null);
    let parents = fact
        .get("parents")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_else(|| match fact.get("sourceAssetId").and_then(Value::as_str) {
            Some(source) => vec![Value::String(source.to_owned())],
            None => Vec::new(),
        });
    let file = json!({
        "path": get("mediaPath"),
        "mimeType": get("mimeType"),
        "width": Value::Null,
        "height": Value::Null,
        // MEASURED off the produced AudioTrack (the WAV on disk), mirroring the video file block's
        // encoded facts. `duration` is the clip's real running time in seconds; `sampleRate` and
        // `channels` describe the PCM the worker wrote.
        "duration": get("duration"),
        "fps": Value::Null,
        "sampleRate": get("sampleRate"),
        "channels": get("channels"),
    });
    let recipe = json!({
        "mode": get("mode"),
        "model": get("model"),
        "adapter": get("adapter"),
        "prompt": get("prompt"),
        "negativePrompt": get("negativePrompt"),
        "seed": get("seed"),
        "loras": fact.get("loras").cloned().unwrap_or_else(|| json!([])),
        // The REPLAY record: the audio knobs the user picked (voice / language / requested
        // duration) so a re-run rebuilds the same request. `sampleRate` records the rate actually
        // produced (Kokoro's native 24 kHz today).
        "normalizedSettings": {
            "voice": get("voice"),
            "language": get("language"),
            "targetDurationSecs": get("targetDurationSecs"),
            "sampleRate": get("sampleRate"),
            "family": get("family"),
        },
        "rawAdapterSettings": fact.get("rawAdapterSettings").cloned().unwrap_or_else(|| json!({})),
    });
    let lineage = json!({
        "parents": parents,
        "sourceAssetId": get("sourceAssetId"),
        "sourceTimestamp": Value::Null,
        "jobId": job_id,
    });
    (file, recipe, lineage)
}

/// `(file, recipe, lineage)` for an interleaved document asset (story 1656,
/// slice 4). The worker writes the document body JSON (the "media"); Rust builds
/// this sidecar from the document fact. The recipe mode is `interleave` and
/// lineage parents are the embedded image asset ids.
fn build_document_sidecar_parts(job_id: &str, fact: &Value) -> (Value, Value, Value) {
    let get = |key: &str| fact.get(key).cloned().unwrap_or(Value::Null);
    let file = json!({
        "path": get("mediaPath"),
        "mimeType": get("mimeType"),
        "width": Value::Null,
        "height": Value::Null,
        "duration": Value::Null,
        "fps": Value::Null,
    });
    let recipe = json!({
        "mode": get("mode"),
        "model": get("model"),
        "adapter": get("adapter"),
        "prompt": get("prompt"),
        "negativePrompt": get("negativePrompt"),
        "seed": get("seed"),
        "loras": fact.get("loras").cloned().unwrap_or_else(|| json!([])),
        "normalizedSettings": {
            "maxImages": get("maxImages"),
            "resolution": get("resolution"),
            "imageCount": get("imageCount"),
        },
        "rawAdapterSettings": fact.get("rawAdapterSettings").cloned().unwrap_or_else(|| json!({})),
    });
    let lineage = json!({
        "parents": fact.get("parents").cloned().unwrap_or_else(|| json!([])),
        "sourceAssetId": Value::Null,
        "sourceTimestamp": Value::Null,
        "jobId": job_id,
    });
    (file, recipe, lineage)
}

impl ProjectStore {
    /// Persist a generated asset the worker reported as flat facts: build the
    /// sidecar (Rust owns this schema now — story 1656), write it next to the
    /// media the worker already saved, write the recipe, and index project.db.
    /// Idempotent — re-applied progress updates upsert the same row/files.
    /// Returns the built sidecar so the API can re-inject it into the job result.
    pub fn persist_generated_asset(
        &self,
        project_id: &str,
        job_id: &str,
        generation_set_id: &str,
        fact: &Value,
    ) -> ProjectStoreResult<Value> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        let media_rel = fact
            .get("mediaPath")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ProjectStoreError::BadRequest("generated asset fact missing mediaPath".to_owned())
            })?;
        // mediaPath comes from the worker result fact and is joined into the project
        // path below for sidecar/dir writes; reject traversal/absolute paths before any
        // create_dir_all/write so the worker->API boundary can't write outside the
        // project root (sc-5721 / CORE-002->F). Mirrors the is_safe_relative_path guard
        // used at every other path-from-outside site in this store.
        if !is_safe_relative_path(media_rel) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid media path".to_owned(),
            ));
        }
        let asset_id = fact.get("assetId").and_then(Value::as_str).ok_or_else(|| {
            ProjectStoreError::BadRequest("generated asset fact missing assetId".to_owned())
        })?;
        // The asset id comes from the worker's result fact and is joined into the
        // recipe path below, so charset-check it before path use (sc-4211 / F-CORE-7).
        if !is_safe_id(asset_id) {
            return Err(ProjectStoreError::BadRequest("Invalid asset id".to_owned()));
        }
        // A preview path (vector / depth) is joined into the project path by every reader, so it
        // gets the same traversal guard as mediaPath (sc-25682).
        if let Some(preview_rel) = fact.pointer("/preview/path") {
            if !preview_rel.as_str().is_some_and(is_safe_relative_path) {
                return Err(ProjectStoreError::BadRequest(
                    "Invalid preview path".to_owned(),
                ));
            }
        }
        let mut asset = build_generated_asset_sidecar(project_id, job_id, generation_set_id, fact);
        self.inherit_model_provenance(&project_path, &mut asset)?;
        let media_path = project_path.join(media_rel);
        let sidecar_path = media_path.with_extension("sceneworks.json");
        let index_mutation = AssetIndexMutation::begin(&project_path)?;
        if let Some(parent) = sidecar_path.parent() {
            fs::create_dir_all(parent)?;
        }
        write_json(&sidecar_path, &asset)?;
        let recipes_dir = project_path.join("recipes");
        fs::create_dir_all(&recipes_dir)?;
        write_json(
            &recipes_dir.join(format!("{asset_id}.recipe.json")),
            &asset["recipe"],
        )?;
        index_asset(project_id, &project_path, &asset, Some(&sidecar_path))?;
        index_mutation.commit();
        Ok(asset)
    }

    /// Keep immutable model ancestry through edits, media conversion, and multi-input generation,
    /// including producer types that do not themselves load a checkpoint. Flatten and deduplicate
    /// commitments rather than recursively copying entire parent assets or mutable recipes.
    pub(super) fn inherit_model_provenance(
        &self,
        project_path: &Path,
        asset: &mut Value,
    ) -> ProjectStoreResult<()> {
        let parents = asset
            .pointer("/lineage/parents")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut inherited = std::collections::BTreeMap::new();
        for provenance in asset
            .pointer("/lineage/modelProvenance")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            inherited.insert(serde_json::to_string(provenance)?, provenance.clone());
        }
        for parent in parents.iter().filter_map(Value::as_str) {
            let path = match self.find_asset_sidecar(project_path, parent) {
                Ok(path) => path,
                Err(ProjectStoreError::NotFound(_)) => continue,
                Err(error) => return Err(error),
            };
            let source: Value = read_json(&path)?;
            if let Some(provenance) = source.get("modelProvenance") {
                inherited.insert(serde_json::to_string(provenance)?, provenance.clone());
            }
            for provenance in source
                .pointer("/lineage/modelProvenance")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                inherited.insert(serde_json::to_string(provenance)?, provenance.clone());
            }
        }
        if !inherited.is_empty() {
            asset["lineage"]["modelProvenance"] = Value::Array(inherited.into_values().collect());
        }
        Ok(())
    }
}
