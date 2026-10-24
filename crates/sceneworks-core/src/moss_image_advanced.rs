//! Edit-only product transport for the native advanced ABI. Project-owned inert files,
//! exact identities and checked shapes cross this boundary; caller paths/inline tensors do not.
use crate::{
    contracts::JsonObject,
    project_store::ProjectStore,
    tensor_artifact::{self, TensorMetadata},
};
use gen_core::{
    moss_image::{
        MossImageAdvancedInputs, MossImageEmbeddings, MossImageOutputKind, MossImageResolved,
    },
    GenerationRequest,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::PathBuf;
pub const KEY: &str = "mossImageAdvanced";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TensorRef {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub asset_id: String,
    pub sha256: String,
    /// Measured by the server. Never trusted without re-reading the owned artifact.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<TensorMetadata>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct EmbeddingsRef {
    pub values: TensorRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mask: Option<TensorRef>,
}
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct EditAdvanced {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub positive: Option<EmbeddingsRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub negative: Option<EmbeddingsRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub negative_prompts: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub noise: Option<TensorRef>,
    #[serde(default)]
    pub output: MossImageOutputKind,
}
pub fn options(payload: &JsonObject) -> Result<Option<EditAdvanced>, String> {
    let Some(value) = payload.get("advanced").and_then(|v| v.get(KEY)) else {
        return Ok(None);
    };
    if !crate::moss_image::is_moss(
        payload
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or_default(),
    ) || !crate::moss_image::is_edit(payload)
    {
        return Err("mossImageAdvanced is supported only by explicit MOSS edit_image".into());
    }
    let options: EditAdvanced = serde_json::from_value(value.clone())
        .map_err(|e| format!("invalid mossImageAdvanced: {e}"))?;
    if options.positive.is_some()
        && (payload
            .get("prompt")
            .and_then(Value::as_str)
            .is_some_and(|s| !s.is_empty())
            || payload
                .get("prompts")
                .and_then(Value::as_array)
                .is_some_and(|p| !p.is_empty()))
    {
        return Err(
            "positive embeddings replace both prompt and prompts; clear both text carriers".into(),
        );
    }
    if options.negative.is_some() && options.negative_prompts.is_some() {
        return Err("negative embeddings and negativePrompts are mutually exclusive".into());
    }
    if (options.negative.is_some() || options.negative_prompts.is_some())
        && payload
            .get("negativePrompt")
            .and_then(Value::as_str)
            .is_some_and(|s| !s.is_empty())
    {
        return Err("advanced negative inputs replace the broadcast negativePrompt".into());
    }
    if options.output == MossImageOutputKind::Latents
        && payload
            .get("upscale")
            .and_then(|v| v.get("enabled"))
            .and_then(Value::as_bool)
            == Some(true)
    {
        return Err(
            "MOSS latent output cannot use inline upscale; disable upscale or select images output"
                .into(),
        );
    }
    Ok(Some(options))
}
pub fn batch_hint(payload: &JsonObject) -> usize {
    payload
        .get("advanced")
        .and_then(|v| v.get(KEY))
        .and_then(|v| v.get("positive"))
        .and_then(|v| v.get("values"))
        .and_then(|v| v.get("metadata"))
        .and_then(|v| v.get("shape"))
        .and_then(|v| v.get(0))
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .unwrap_or(1)
}
pub fn schema() -> Value {
    let metadata = json!({"type":"object","additionalProperties":false,"required":["dtype","shape","byteLength","sha256"],"properties":{"dtype":{"type":"string","enum":["float32","uint8"]},"shape":{"type":"array","minItems":2,"maxItems":4,"items":{"type":"integer","minimum":1}},"byteLength":{"type":"integer","minimum":1,"maximum":tensor_artifact::MAX_DATA_BYTES},"sha256":{"type":"string","pattern":"^[0-9a-f]{64}$"}}});
    let tensor = json!({"type":"object","additionalProperties":false,"required":["assetId","sha256"],"properties":{"assetId":{"type":"string","minLength":1},"sha256":{"type":"string","pattern":"^[0-9a-f]{64}$"},"metadata":metadata},"description":"Exact project-owned inert NPY artifact. Optional measured metadata is reverified against the bytes; it never authorizes a shape or identity."});
    let embeddings = json!({"type":"object","additionalProperties":false,"required":["values"],"properties":{"values":tensor,"mask":tensor}});
    json!({"type":"object","additionalProperties":false,"properties":{"positive":embeddings,"negative":embeddings,"noise":tensor,"negativePrompts":{"type":"array","minItems":1,"maxItems":8,"items":{"type":"string","maxLength":4000}},"output":{"type":"string","enum":["images","latents"],"default":"images"}}})
}

fn resolve(
    store: &ProjectStore,
    project: &str,
    reference: &mut TensorRef,
) -> Result<(PathBuf, TensorMetadata), String> {
    if reference.sha256.len() != 64
        || !reference
            .sha256
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err("tensor reference requires its exact lowercase SHA256".into());
    }
    let asset = store
        .get_asset(project, &reference.asset_id)
        .map_err(|e| format!("tensor asset {}: {e}", reference.asset_id))?;
    if asset.get("type").and_then(Value::as_str) != Some("tensor") {
        return Err("advanced inputs require project-owned tensor assets".into());
    }
    let path = asset
        .pointer("/file/path")
        .and_then(Value::as_str)
        .ok_or("tensor file missing")?;
    let path = store
        .project_file(project, path)
        .map_err(|e| e.to_string())?
        .path;
    let metadata = tensor_artifact::inspect(&path)?;
    if metadata.sha256 != reference.sha256 {
        return Err(format!(
            "tensor {} identity changed; reacquire the exact bytes",
            reference.asset_id
        ));
    }
    if reference.metadata.as_ref().is_some_and(|m| m != &metadata) {
        return Err("tensor metadata does not match its exact artifact".into());
    }
    reference.metadata = Some(metadata.clone());
    Ok((path, metadata))
}
fn embedding_metadata(
    store: &ProjectStore,
    project: &str,
    reference: &mut EmbeddingsRef,
    context: usize,
) -> Result<usize, String> {
    let (_, meta) = resolve(store, project, &mut reference.values)?;
    if meta.dtype != "float32"
        || meta.shape.len() != 3
        || meta.shape[0] > 8
        || meta.shape[1] > context
        || meta.shape[2] != 4096
    {
        return Err("embeddings must be finite FP32 [B,S,4096], B1–8 and S within context".into());
    }
    if let Some(mask) = reference.mask.as_mut() {
        let (path, m) = resolve(store, project, mask)?;
        if m.dtype != "uint8" || m.shape != meta.shape[..2] {
            return Err("binary attention mask must match [B,S] embeddings".into());
        }
        let bytes = tensor_artifact::read_data(&path, &m)?;
        if !bytes
            .chunks_exact(meta.shape[1])
            .all(|row| row.contains(&1))
        {
            return Err("each embedding row requires at least one valid mask token".into());
        }
    }
    Ok(meta.shape[0])
}
/// Resolve/stream-check before enqueue. Only tiny masks are materialized here.
pub fn canonicalize(
    store: &ProjectStore,
    payload: &mut JsonObject,
    entry: &JsonObject,
) -> Result<(), String> {
    let Some(mut advanced) = options(payload)? else {
        return Ok(());
    };
    if let Some(prompts) = payload
        .get("prompts")
        .and_then(Value::as_array)
        .filter(|p| !p.is_empty())
        .cloned()
    {
        payload
            .get_mut("advanced")
            .and_then(Value::as_object_mut)
            .ok_or("advanced must be an object")?
            .insert("promptBatch".into(), Value::Array(prompts));
    }
    let project = payload
        .get("projectId")
        .and_then(Value::as_str)
        .ok_or("advanced tensors require projectId")?
        .to_owned();
    let context = payload
        .get("advanced")
        .and_then(|v| v.get("mossImage"))
        .and_then(|v| v.get("max_sequence_length"))
        .and_then(Value::as_u64)
        .unwrap_or(4096) as usize;
    if !(1..=4096).contains(&context) {
        return Err("context limit must be in1–4096".into());
    }
    let batch = if let Some(positive) = advanced.positive.as_mut() {
        embedding_metadata(store, &project, positive, context)?
    } else {
        payload
            .get("prompts")
            .and_then(Value::as_array)
            .map_or(1, |p| p.len().max(1))
    };
    if let Some(negative) = advanced.negative.as_mut() {
        if embedding_metadata(store, &project, negative, context)? != batch {
            return Err("negative embedding batch must match positive conditioning".into());
        }
    }
    if let Some(negatives) = &advanced.negative_prompts {
        if negatives.len() != batch || negatives.iter().any(|s| s.chars().count() > 4000) {
            return Err("negativePrompts must match positive batch and existing 4000-character prompt bounds".into());
        }
    }
    payload
        .get_mut("advanced")
        .and_then(Value::as_object_mut)
        .ok_or("advanced must be an object")?
        .insert(
            KEY.into(),
            serde_json::to_value(&advanced).map_err(|e| e.to_string())?,
        );
    let native =
        crate::moss_image::native_request(payload, entry)?.ok_or("missing MOSS edit request")?;
    let images = batch
        .checked_mul(native.count as usize)
        .filter(|n| *n > 0 && *n <= 8)
        .ok_or("MOSS advanced batch×count must be1–8")?;
    if let Some(noise) = advanced.noise.as_mut() {
        let (_, meta) = resolve(store, &project, noise)?;
        if meta.dtype != "float32"
            || meta.shape
                != [
                    images,
                    128,
                    native.height as usize / 16,
                    native.width as usize / 16,
                ]
        {
            return Err("noise must be exact finite FP32 [B×count,128,H/16,W/16]".into());
        }
        payload.insert("seed".into(), Value::Null);
    }
    let total = references(&advanced)
        .iter()
        .try_fold(0usize, |n, r| {
            n.checked_add(r.metadata.as_ref().expect("resolved metadata").byte_length)
        })
        .filter(|n| *n <= tensor_artifact::MAX_REQUEST_BYTES)
        .ok_or("advanced tensor materialization exceeds the1280MiB request budget")?;
    let _ = total;
    payload
        .get_mut("advanced")
        .and_then(Value::as_object_mut)
        .expect("advanced object")
        .insert(
            KEY.into(),
            serde_json::to_value(advanced).map_err(|e| e.to_string())?,
        );
    Ok(())
}
pub fn references(options: &EditAdvanced) -> Vec<&TensorRef> {
    let mut refs = Vec::new();
    for e in [&options.positive, &options.negative].into_iter().flatten() {
        refs.push(&e.values);
        if let Some(mask) = &e.mask {
            refs.push(mask);
        }
    }
    if let Some(noise) = &options.noise {
        refs.push(noise);
    }
    refs
}

/// Cross-library envelopes retain exact identities and shapes, never installation-local IDs.
pub fn share_options(value: &Value) -> Option<Value> {
    let mut options: EditAdvanced = serde_json::from_value(value.clone()).ok()?;
    let clear = |r: &mut TensorRef| -> Option<()> {
        let m = r.metadata.as_ref()?;
        if m.sha256 != r.sha256
            || m.sha256.len() != 64
            || !m
                .sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            || !(2..=4).contains(&m.shape.len())
            || m.shape.contains(&0)
            || m.byte_length > tensor_artifact::MAX_DATA_BYTES
        {
            return None;
        }
        let bytes = m
            .shape
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))?
            .checked_mul(if m.dtype == "float32" {
                4
            } else if m.dtype == "uint8" {
                1
            } else {
                return None;
            })?;
        if bytes != m.byte_length {
            return None;
        }
        r.asset_id.clear();
        Some(())
    };
    for e in [&mut options.positive, &mut options.negative]
        .into_iter()
        .flatten()
    {
        clear(&mut e.values)?;
        if let Some(mask) = e.mask.as_mut() {
            clear(mask)?;
        }
    }
    if let Some(noise) = options.noise.as_mut() {
        clear(noise)?;
    }
    if options
        .negative_prompts
        .as_ref()
        .is_some_and(|p| p.len() > 8 || p.iter().any(|s| s.chars().count() > 4000))
    {
        return None;
    }
    if let Some(negatives) = options.negative_prompts.as_mut() {
        for value in negatives {
            *value = crate::workflow_share::shareable_prose(value);
        }
    }
    serde_json::to_value(options).ok()
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TensorRequirement {
    pub role: String,
    pub sha256: String,
    pub metadata: TensorMetadata,
    pub detail: String,
}
pub fn requirements(value: &Value) -> Vec<TensorRequirement> {
    let Ok(options) = serde_json::from_value::<EditAdvanced>(value.clone()) else {
        return Vec::new();
    };
    let mut result = Vec::new();
    let mut add = |role: &str, r: &TensorRef| {
        if let Some(metadata) = r.metadata.as_ref() {
            result.push(TensorRequirement {
                role: role.into(),
                sha256: r.sha256.clone(),
                metadata: metadata.clone(),
                detail: format!(
                    "Reacquire exact {role} tensor (SHA256 {}, {} {:?}).",
                    r.sha256, metadata.dtype, metadata.shape
                ),
            });
        }
    };
    if let Some(e) = &options.positive {
        add("positive.values", &e.values);
        if let Some(m) = &e.mask {
            add("positive.mask", m);
        }
    }
    if let Some(e) = &options.negative {
        add("negative.values", &e.values);
        if let Some(m) = &e.mask {
            add("negative.mask", m);
        }
    }
    if let Some(n) = &options.noise {
        add("noise", n);
    }
    result
}

#[derive(Default)]
pub struct OwnedInputs {
    pub options: EditAdvanced,
    pub positive: Option<Vec<f32>>,
    pub positive_mask: Option<Vec<u8>>,
    pub negative: Option<Vec<f32>>,
    pub negative_mask: Option<Vec<u8>>,
    pub noise: Option<Vec<f32>>,
    pub host_bytes: usize,
}
impl OwnedInputs {
    pub fn load(
        store: &ProjectStore,
        payload: &JsonObject,
        entry: &JsonObject,
    ) -> Result<Self, String> {
        let mut checked = payload.clone();
        canonicalize(store, &mut checked, entry)?;
        let options = options(&checked)?.unwrap_or_default();
        let project = checked
            .get("projectId")
            .and_then(Value::as_str)
            .ok_or("projectId missing")?;
        let bytes = |r: &TensorRef| -> Result<Vec<u8>, String> {
            let mut r = r.clone();
            let (path, meta) = resolve(store, project, &mut r)?;
            tensor_artifact::read_data(&path, &meta)
        };
        let floats = |r: &TensorRef| -> Result<Vec<f32>, String> {
            Ok(bytes(r)?
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().expect("four bytes")))
                .collect())
        };
        let positive = options
            .positive
            .as_ref()
            .map(|e| floats(&e.values))
            .transpose()?;
        let positive_mask = options
            .positive
            .as_ref()
            .and_then(|e| e.mask.as_ref())
            .map(bytes)
            .transpose()?;
        let negative = options
            .negative
            .as_ref()
            .map(|e| floats(&e.values))
            .transpose()?;
        let negative_mask = options
            .negative
            .as_ref()
            .and_then(|e| e.mask.as_ref())
            .map(bytes)
            .transpose()?;
        let noise = options.noise.as_ref().map(floats).transpose()?;
        let sizes = references(&options)
            .iter()
            .map(|r| r.metadata.as_ref().expect("validated metadata").byte_length)
            .collect::<Vec<_>>();
        // Retained native host inputs plus the largest exact-file decode staging buffer.
        let host_bytes = sizes.iter().sum::<usize>()
            + sizes.iter().max().copied().unwrap_or(0)
            + if sizes.is_empty() {
                0
            } else {
                tensor_artifact::MAX_HEADER_BYTES + 12
            };
        Ok(Self {
            options,
            positive,
            positive_mask,
            negative,
            negative_mask,
            noise,
            host_bytes,
        })
    }
    pub fn inputs(&self) -> MossImageAdvancedInputs<'_> {
        fn embeddings<'a>(
            e: &'a Option<EmbeddingsRef>,
            values: &'a [f32],
            mask: Option<&'a [u8]>,
        ) -> Option<MossImageEmbeddings<'a>> {
            e.as_ref().map(|e| {
                let shape = &e
                    .values
                    .metadata
                    .as_ref()
                    .expect("validated metadata")
                    .shape;
                MossImageEmbeddings {
                    values,
                    batch: shape[0],
                    sequence: shape[1],
                    mask,
                }
            })
        }
        MossImageAdvancedInputs {
            positive: embeddings(
                &self.options.positive,
                self.positive.as_deref().unwrap_or_default(),
                self.positive_mask.as_deref(),
            ),
            negative: embeddings(
                &self.options.negative,
                self.negative.as_deref().unwrap_or_default(),
                self.negative_mask.as_deref(),
            ),
            negative_prompts: self.options.negative_prompts.as_deref(),
            noise: self.noise.as_deref(),
            output: self.options.output,
        }
    }
    pub fn validate(
        &self,
        request: &GenerationRequest,
        resolved: &MossImageResolved,
    ) -> Result<usize, String> {
        self.inputs()
            .validate(request, resolved)
            .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn upload(
        store: &ProjectStore,
        project: &str,
        path: &std::path::Path,
        bytes: &[u8],
    ) -> TensorRef {
        std::fs::write(path, bytes).unwrap();
        let asset = store
            .import_asset(
                project,
                crate::project_store::UploadAsset {
                    filename: "exact.npy".into(),
                    content_type: Some(tensor_artifact::MIME.into()),
                    source_path: path.to_owned(),
                    source_asset_id: None,
                    provenance: None,
                },
            )
            .unwrap();
        TensorRef {
            asset_id: asset["id"].as_str().unwrap().into(),
            sha256: asset["tensor"]["sha256"].as_str().unwrap().into(),
            metadata: None,
        }
    }
    #[test]
    fn owned_masks_context_shapes_modes_and_identities_fail_before_host_materialization() {
        let temp = tempfile::tempdir().unwrap();
        let store = ProjectStore::new(temp.path().to_owned(), "test");
        let project = store.create_project("mask bounds").unwrap();
        let positive = upload(
            &store,
            &project.id,
            &temp.path().join("positive.npy"),
            &tensor_artifact::encode_f32(&[2, 2, 4096], &vec![0.; 16384]).unwrap(),
        );
        let header = b"{'descr': '|u1', 'fortran_order': False, 'shape': (2,2), }\n";
        let mut mask = b"\x93NUMPY\x01\x00".to_vec();
        mask.extend((header.len() as u16).to_le_bytes());
        mask.extend(header);
        mask.extend([1, 0, 0, 0]);
        let mask = upload(&store, &project.id, &temp.path().join("mask.npy"), &mask);
        let payload=json!({"projectId":project.id,"model":"moss_image_sft","mode":"edit_image","prompt":"","referenceAssetIds":["rgb"],"width":16,"height":16,"advanced":{"mossImage":{"max_sequence_length":2},"mossImageAdvanced":{"positive":{"values":positive,"mask":mask}}}}).as_object().unwrap().clone();
        let entry = json!({"limits":{"minDimension":16}})
            .as_object()
            .unwrap()
            .clone();
        assert!(canonicalize(&store, &mut payload.clone(), &entry)
            .unwrap_err()
            .contains("each embedding row"));
        let mut value = payload.clone();
        value.get_mut("advanced").unwrap()[KEY]["positive"]
            .as_object_mut()
            .unwrap()
            .remove("mask");
        value.get_mut("advanced").unwrap()["mossImage"]["max_sequence_length"] = json!(1);
        assert!(canonicalize(&store, &mut value, &entry)
            .unwrap_err()
            .contains("within context"));
        let mut value = payload.clone();
        value.insert("mode".into(), json!("text_to_image"));
        assert!(canonicalize(&store, &mut value, &entry)
            .unwrap_err()
            .contains("only"));
        let mut value = payload.clone();
        value.get_mut("advanced").unwrap()[KEY]["positive"]["values"]["sha256"] =
            json!("b".repeat(64));
        assert!(canonicalize(&store, &mut value, &entry)
            .unwrap_err()
            .contains("identity changed"));
        let foreign = store.create_project("foreign").unwrap();
        let mut value = payload;
        value.insert("projectId".into(), json!(foreign.id));
        assert!(canonicalize(&store, &mut value, &entry).is_err());
    }
    #[test]
    fn latent_inline_upscale_is_rejected_before_enqueue_without_rejecting_valid_outputs() {
        let temp = tempfile::tempdir().unwrap();
        let store = ProjectStore::new(temp.path().to_owned(), "test");
        let project = store.create_project("output combinations").unwrap();
        let noise = upload(
            &store,
            &project.id,
            &temp.path().join("noise.npy"),
            &tensor_artifact::encode_f32(&[1, 128, 1, 1], &vec![0.; 128]).unwrap(),
        );
        let payload = json!({"projectId":project.id,"model":"moss_image_sft","mode":"edit_image","prompt":"edit","referenceAssetIds":["rgb"],"width":16,"height":16,"advanced":{"mossImageAdvanced":{"noise":noise,"output":"latents"}}})
            .as_object().unwrap().clone();
        let entry = json!({"limits":{"minDimension":16,"maxDimension":4096}})
            .as_object()
            .unwrap()
            .clone();
        for engine in ["real-esrgan", "seedvr2"] {
            let mut invalid = payload.clone();
            invalid.insert(
                "upscale".into(),
                json!({"enabled":true,"engine":engine,"factor":2}),
            );
            let unchanged = invalid.clone();
            let error = canonicalize(&store, &mut invalid, &entry).unwrap_err();
            assert!(
                error.contains("disable upscale or select images"),
                "{error}"
            );
            assert_eq!(
                invalid, unchanged,
                "refuse before enqueue canonicalization mutates inputs"
            );
            invalid.get_mut("advanced").unwrap()[KEY]["output"] = json!("images");
            canonicalize(&store, &mut invalid, &entry).unwrap();
        }
        canonicalize(&store, &mut payload.clone(), &entry).unwrap();
        let mut disabled = payload;
        disabled.insert(
            "upscale".into(),
            json!({"enabled":false,"engine":"real-esrgan"}),
        );
        canonicalize(&store, &mut disabled, &entry).unwrap();
    }
    #[test]
    fn owned_artifacts_reach_native_views_with_exact_noise_bits_and_replay_identities() {
        let temp = tempfile::tempdir().unwrap();
        let store = ProjectStore::new(temp.path().to_owned(), "test");
        let project = store.create_project("advanced edit").unwrap();
        let positive = upload(
            &store,
            &project.id,
            &temp.path().join("positive.npy"),
            &tensor_artifact::encode_f32(&[2, 2, 4096], &vec![0.; 2 * 2 * 4096]).unwrap(),
        );
        let mut noise_values = vec![0.; 4 * 128];
        noise_values[0] = -0.;
        noise_values[1] = f32::from_bits(1);
        let noise = upload(
            &store,
            &project.id,
            &temp.path().join("noise.npy"),
            &tensor_artifact::encode_f32(&[4, 128, 1, 1], &noise_values).unwrap(),
        );
        let mut payload=json!({"projectId":project.id,"model":"moss_image_sft","mode":"edit_image","prompt":"","referenceAssetIds":["rgb"],"count":2,"width":16,"height":16,"seed":42,"advanced":{"mossImageAdvanced":{"positive":{"values":positive},"noise":noise,"negativePrompts":["blur","text"],"output":"latents"}}}).as_object().unwrap().clone();
        let entry = json!({"limits":{"minDimension":16,"maxDimension":4096}})
            .as_object()
            .unwrap()
            .clone();
        canonicalize(&store, &mut payload, &entry).unwrap();
        assert_eq!(payload["seed"], Value::Null);
        assert_eq!(batch_hint(&payload), 2);
        let native = crate::moss_image::native_request(&payload, &entry)
            .unwrap()
            .unwrap();
        let resolved =
            MossImageResolved::resolve(gen_core::moss_image::MossImageVariant::Sft, &native)
                .unwrap();
        let owned = OwnedInputs::load(&store, &payload, &entry).unwrap();
        assert_eq!(owned.validate(&native, &resolved).unwrap(), 4);
        assert_eq!(
            owned.inputs().noise.unwrap()[0].to_bits(),
            (-0f32).to_bits()
        );
        assert_eq!(owned.inputs().noise.unwrap()[1].to_bits(), 1);
        assert_eq!(owned.inputs().output, MossImageOutputKind::Latents);
        let share = share_options(&payload["advanced"][KEY]).unwrap();
        assert!(share["positive"]["values"].get("assetId").is_none());
        assert_eq!(
            share["noise"]["sha256"],
            payload["advanced"][KEY]["noise"]["sha256"]
        );
        assert_eq!(requirements(&share).len(), 2);
        payload.insert("prompt".into(), json!("conflicting text"));
        assert!(canonicalize(&store, &mut payload, &entry).is_err());
    }
}
