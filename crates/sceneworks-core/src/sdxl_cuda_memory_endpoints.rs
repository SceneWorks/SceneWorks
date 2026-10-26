//! Two measured SDXL CUDA endpoints, independently of full runtime certification.
//!
//! The ordinary anchor store deliberately retains one derivation basis per catalog cell. This
//! bounded pair retains both cold renders without changing that store's identity or its callers.
//! Raw peaks are measurements; using them for a future admission still requires the worker's
//! estimate validation, uncertainty and reserve. No quality/parity or runtime-complete verdict is
//! created here. The immutable capture summary binds the bundle bytes, actual CUDA context,
//! compiled capture pin, artifact inventory, exact execution settings and retained image receipts.

use std::collections::BTreeSet;
use std::sync::OnceLock;

use serde::Deserialize;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use super::{validate_anchor_with_sources, AnchorBackend, AnchorLoadShape, MemoryAnchor};
use crate::memory_calibration::{load_bundle, RequiredNullable};

pub const PACKAGED_SDXL_CUDA_MEMORY_ENDPOINTS: &str =
    include_str!("../../../config/sdxl-cuda-memory-endpoints.json");

// Populated only with actual retained captures. These resources intentionally do not participate
// in the ordinary anchor extractor: a fully bounded endpoint is not its unbounded derivation row.
const PACKAGED_ENDPOINT_SOURCES: &[(&str, &str)] = &[];
const GPU_UUID: &str = "GPU-e4b79931-7be6-f216-460a-f5405cfafffe";
const FINGERPRINT: &str = "sdxl-candle-sdxl-staged-decode-attention-v2";
const FIXTURE: &str = "fresh-five-rung-sdxl-q4-1024-seed16402-default-steps";
const PROMPT: &str = "a photorealistic red apple on a wooden table, studio lighting";
const RGB_BYTES: u64 = 1024 * 1024 * 3;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EndpointArtifact {
    pub repository: String,
    pub resolved_revision: String,
    pub variant: String,
    pub inventory_sha256: String,
    pub resolved_path_fingerprint: String,
    /// Exact capture order; the composite digest commits to this order as well as each file.
    pub files: Vec<EndpointArtifactFile>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EndpointArtifactFile {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}

/// Validated raw measurements. Neither endpoint is a runtime-certification record.
#[derive(Debug, Clone)]
pub struct SdxlCudaMemoryEndpoints {
    pub resident: MemoryAnchor,
    pub sequential: MemoryAnchor,
    pub artifact: EndpointArtifact,
    pub sequential_parameters: Map<String, Value>,
    /// Actual request settings and resolved provider defaults, authenticated by the summary.
    pub settings: Value,
    pub resolved_defaults: Value,
    /// Model load is outside the three render phases. Derivations must retain this lower bound.
    pub resident_load_peak_bytes: u64,
    pub sequential_load_peak_bytes: u64,
    pub summary_path: String,
    pub summary_sha256: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Config {
    schema_version: u32,
    #[serde(deserialize_with = "required_pair")]
    pair: RequiredNullable<StoredPair>,
}

// A custom field deserializer makes omission an error at the containing struct. Without it,
// RequiredNullable delegates to Option and Serde treats a missing field like explicit null.
fn required_pair<'de, D>(deserializer: D) -> Result<RequiredNullable<StoredPair>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    RequiredNullable::deserialize(deserializer)
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct StoredPair {
    resident: MemoryAnchor,
    sequential: MemoryAnchor,
    summary: SourceReceipt,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceReceipt {
    path: String,
    sha256: String,
}

/// Invalid or absent packaged evidence supplies no candidate. The explicit loader returns the
/// diagnostic for tooling/tests; product fallback never borrows one half of a malformed pair.
pub fn packaged_sdxl_cuda_memory_endpoints() -> Option<&'static SdxlCudaMemoryEndpoints> {
    static PAIR: OnceLock<Option<SdxlCudaMemoryEndpoints>> = OnceLock::new();
    PAIR.get_or_init(|| {
        load_sdxl_cuda_memory_endpoints(
            PACKAGED_SDXL_CUDA_MEMORY_ENDPOINTS,
            PACKAGED_ENDPOINT_SOURCES,
        )
        .ok()
        .flatten()
    })
    .as_ref()
}

/// Load against an explicit immutable resource table, analogous to the ordinary anchor store.
/// Production calls this only with compiled resources; no runtime filesystem trust is involved.
pub fn load_sdxl_cuda_memory_endpoints(
    raw: &str,
    sources: &[(&str, &str)],
) -> Result<Option<SdxlCudaMemoryEndpoints>, String> {
    let config: Config =
        serde_json::from_str(raw).map_err(|e| format!("SDXL endpoint config: {e}"))?;
    require(config.schema_version == 1, "unsupported schemaVersion")?;
    let RequiredNullable::Value(pair) = config.pair else {
        return Ok(None);
    };
    require(
        pair.resident.id != pair.sequential.id,
        "duplicate anchor id",
    )?;
    require(
        pair.resident.source.record_id != pair.sequential.source.record_id,
        "duplicate record id",
    )?;
    let summary = authenticated_json(sources, &pair.summary.path, &pair.summary.sha256)?;
    require(
        summary["schemaVersion"] == 1 && summary["profile"] == "sdxl-cuda-endpoints",
        "capture profile",
    )?;
    require(
        summary["reserveIncludedInMeasuredPeaks"] == false
            && summary["reserveBytes"] == 2u64 * 1024 * 1024 * 1024,
        "raw peak reserve accounting",
    )?;
    let pin = text(&summary, "/inferencePin")?;
    require(is_hex(pin, 40), "capture inference pin")?;
    let files = validate_inventory(&summary["artifactInventory"])?;
    let inventory = text(&summary, "/artifactInventory/sha256")?;
    require(is_hex(inventory, 64), "artifact inventory digest")?;
    require(
        number(&summary, "/artifactInventory/files")? > 0
            && number(&summary, "/artifactInventory/bytes")? > 0,
        "empty artifact inventory",
    )?;
    let endpoints = summary["endpoints"]
        .as_array()
        .ok_or("SDXL endpoints: missing endpoint receipts")?;
    require(
        endpoints.len() == 2,
        "requires exactly two endpoint receipts",
    )?;
    let mut records = Vec::new();
    let mut receipts = Vec::new();
    for (name, anchor) in [
        ("resident", &pair.resident),
        ("sequential", &pair.sequential),
    ] {
        validate_anchor_with_sources(anchor, sources)?;
        let source = authenticated_json(sources, &anchor.source.path, &anchor.source.sha256)?;
        let source_raw = resource(sources, &anchor.source.path)?;
        load_bundle(source_raw).map_err(|e| format!("SDXL endpoint evidence: {e}"))?;
        let rows = source["records"]
            .as_array()
            .ok_or("SDXL endpoints: missing source records")?;
        require(rows.len() == 1, "endpoint bundle must contain one record")?;
        let record = &rows[0];
        require(
            record["id"] == anchor.source.record_id,
            "source record identity",
        )?;
        let matches: Vec<_> = endpoints.iter().filter(|e| e["endpoint"] == name).collect();
        require(matches.len() == 1, "duplicate or missing endpoint receipt")?;
        let receipt = matches[0];
        validate_endpoint(name, anchor, record, receipt, &summary)?;
        records.push(record.clone());
        receipts.push(receipt.clone());
    }
    let resident_record = &records[0];
    let sequential_record = &records[1];
    for field in [
        "artifact",
        "hardware",
        "repositories",
        "loadShape",
        "target",
        "calibrationFingerprint",
        "fixture",
        "loadability",
    ] {
        require(
            resident_record[field] == sequential_record[field],
            &format!("pair differs in {field}"),
        )?;
    }
    let mut resident_settings = receipts[0]["settings"].clone();
    let mut sequential_settings = receipts[1]["settings"].clone();
    // Execution memory differs by design; all generation settings must be identical.
    resident_settings
        .as_object_mut()
        .ok_or("SDXL endpoints: settings object")?
        .remove("memory");
    sequential_settings
        .as_object_mut()
        .ok_or("SDXL endpoints: settings object")?
        .remove("memory");
    require(
        receipts[0]["engineBuild"] == receipts[1]["engineBuild"],
        "endpoint builds differ",
    )?;
    require(
        resident_settings == sequential_settings,
        "generation settings differ",
    )?;
    require(
        pair.sequential.overall_allocator_envelope_bytes
            <= pair.resident.overall_allocator_envelope_bytes,
        "optimized endpoint exceeds resident endpoint",
    )?;
    validate_comparison(&summary["qualityComparison"])?;
    let artifact = EndpointArtifact {
        repository: text(resident_record, "/artifact/repository")?.to_owned(),
        resolved_revision: text(resident_record, "/artifact/resolvedRevision")?.to_owned(),
        variant: text(resident_record, "/artifact/variant")?.to_owned(),
        inventory_sha256: inventory.to_owned(),
        resolved_path_fingerprint: text(resident_record, "/loadability/resolvedPathFingerprint")?
            .to_owned(),
        files,
    };
    Ok(Some(SdxlCudaMemoryEndpoints {
        resident: pair.resident,
        sequential: pair.sequential,
        artifact,
        sequential_parameters: expected_strategy("sequential")["parameters"]
            .as_object()
            .unwrap()
            .clone(),
        settings: receipts[0]["settings"].clone(),
        resolved_defaults: summary["resolvedDefaults"].clone(),
        resident_load_peak_bytes: number(&receipts[0], "/loadPeakBytes")?,
        sequential_load_peak_bytes: number(&receipts[1], "/loadPeakBytes")?,
        summary_path: pair.summary.path,
        summary_sha256: pair.summary.sha256,
    }))
}

fn validate_endpoint(
    name: &str,
    anchor: &MemoryAnchor,
    record: &Value,
    receipt: &Value,
    summary: &Value,
) -> Result<(), String> {
    require(
        anchor.model_id == "sdxl"
            && anchor.route == "sdxl"
            && anchor.provider == "sdxl"
            && anchor.backend == AnchorBackend::Candle
            && anchor.tier == "q4"
            && anchor.mode == "text_to_image",
        "endpoint model identity",
    )?;
    // Stock SDXL uses eager LoadSpec materialization. Phase staging remains an independent
    // strategy axis; a deferred capture cannot stand in for the ordinary product request.
    require(
        anchor.load_shape == AnchorLoadShape::EagerMaterialization,
        "ordinary SDXL eager load shape",
    )?;
    require(
        anchor.overlay.is_none()
            && anchor.reference_count == 0
            && anchor.transformer_variant.is_none()
            && anchor.decoder.is_none(),
        "endpoint overlay/reference identity",
    )?;
    require(
        anchor.geometry.width == 1024
            && anchor.geometry.height == 1024
            && anchor.geometry.frames == 1
            && anchor.geometry.fps.is_none(),
        "endpoint geometry",
    )?;
    require(
        record["target"]["geometry"] == json!({"width":1024,"height":1024,"batch":1,"frames":1}),
        "source batch/geometry",
    )?;
    require(
        record["target"]["overlay"] == "none"
            && record["target"]
                .get("referenceCount")
                .is_none_or(|v| v == 0),
        "source overlay/reference identity",
    )?;
    require(
        record["fixture"] == FIXTURE && record["calibrationFingerprint"] == FINGERPRINT,
        "endpoint fixture/fingerprint",
    )?;
    require(
        record["diagnostics"]["execution"] == "executed",
        "endpoint was not executed",
    )?;
    require(
        record["strategy"] == expected_strategy(name),
        "exact endpoint strategy parameters",
    )?;
    require(
        receipt["planned"]["strategy"] == record["strategy"]
            && receipt["planned"]["target"] == record["target"]
            && receipt["planned"]["fixture"] == record["fixture"]
            && receipt["planned"]["loadShape"] == record["loadShape"]
            && receipt["planned"]["calibrationFingerprint"] == record["calibrationFingerprint"],
        "planned/executed identity differs",
    )?;
    require(
        receipt["recordId"] == anchor.source.record_id
            && receipt["evidenceSha256"] == anchor.source.sha256,
        "summary/source digest or record binding",
    )?;
    // Capture provenance only: the adapter that ran must match the captured source checkout.
    // No active runtime pin is consulted; a later merge/pin does not invalidate this measurement.
    require(
        receipt["inferencePin"] == summary["inferencePin"]
            && record["repositories"]["inference"]["revision"] == summary["inferencePin"],
        "compiled capture pin differs from source",
    )?;
    for repository in ["sceneWorks", "inference"] {
        require(
            record["repositories"][repository]["dirty"] == false
                && is_hex(
                    text(record, &format!("/repositories/{repository}/revision"))?,
                    40,
                ),
            "dirty or unidentified capture source",
        )?;
    }
    require(
        record["artifact"]["inventorySha256"] == summary["artifactInventory"]["sha256"]
            && record["artifact"]["variant"] == "q4"
            && is_hex(text(record, "/artifact/resolvedRevision")?, 40),
        "artifact inventory identity",
    )?;
    let fingerprint = format!(
        "{}@{}:q4",
        text(record, "/artifact/repository")?,
        text(record, "/artifact/resolvedRevision")?
    );
    require(
        record["loadability"]["result"] == "passed"
            && record["loadability"]["resolvedPathFingerprint"] == fingerprint,
        "artifact loadability identity",
    )?;
    require(
        receipt["deviceUuid"] == GPU_UUID
            && receipt["cudaContextUuid"] == GPU_UUID
            && receipt["logicalCudaDevice"] == 0
            && receipt["offloadPolicy"] == name,
        "physical CUDA context/offload identity",
    )?;
    require(
        receipt["engineBuild"]["adapterCandleFeature"] == true
            && receipt["engineBuild"]["sdxlFlashAttentionKernelCompiled"] == false
            && receipt["engineBuild"]["cudaComputeCapabilityCompileTarget"] == "120",
        "captured CUDA build/attention kernel policy",
    )?;
    require(
        record["hardware"]["deviceId"] == GPU_UUID
            && text(record, "/hardware/probe")?.contains("nvidia-smi")
            && number(record, "/hardware/memoryBytes")? > 0,
        "CUDA hardware receipt",
    )?;
    let whole = number(receipt, "/wholeRunPeakBytes")?;
    let load = number(receipt, "/loadPeakBytes")?;
    let baseline = number(receipt, "/baselineBytes")?;
    let device = number(receipt, "/wholeDevicePeakBytes")?;
    require(
        whole > 0 && load > 0 && whole >= load && whole == anchor.overall_allocator_envelope_bytes,
        "load-inclusive whole-run peak",
    )?;
    // Decimal-GB f64 -> integer-byte conversion may round the combined sample by one byte.
    require(
        baseline
            .checked_add(whole)
            .is_some_and(|sum| sum.abs_diff(device) <= 1),
        "physical baseline/peak arithmetic",
    )?;
    require(
        device <= number(record, "/hardware/memoryBytes")?,
        "device peak exceeds physical capacity",
    )?;
    for (phase, bytes) in [
        ("conditioning", anchor.phase_active_peak_bytes.conditioning),
        ("denoise", anchor.phase_active_peak_bytes.denoise),
        ("decode", anchor.phase_active_peak_bytes.decode),
        ("overall", whole),
    ] {
        require(
            bytes <= whole
                && record["observedMemory"][phase]
                    == json!({"activeBytes":bytes,"allocatorBytes":bytes,"reclaimableBytes":0}),
            "physical-device delta accounting",
        )?;
    }
    let settings = &receipt["settings"];
    require(
        settings["prompt"] == PROMPT
            && settings["seed"] == 16402
            && settings["steps"] == 30
            && settings["guidance"].as_f64() == Some(7.0),
        "prompt/seed/default generation settings",
    )?;
    require(
        settings["flashAttn"] == true && settings["count"] == 1,
        "attention kernel/count setting",
    )?;
    for key in [
        "negativePrompt",
        "sampler",
        "scheduler",
        "cfgInterval",
        "guidanceMethod",
        "schedulerShift",
    ] {
        require(
            settings.get(key) == Some(&Value::Null),
            "unexpected generation override",
        )?;
    }
    require(
        summary["resolvedDefaults"]["steps"] == 30
            && summary["resolvedDefaults"]["guidance"].as_f64() == Some(7.0)
            && !text(summary, "/resolvedDefaults/sampler")?.is_empty(),
        "resolved provider defaults",
    )?;
    let output = &receipt["output"];
    require(
        output["width"] == 1024
            && output["height"] == 1024
            && output["channels"] == 3
            && output["rgbBytes"] == RGB_BYTES
            && output["spatiallyNonFlat"] == true,
        "retained RGB shape",
    )?;
    require(
        number(output, "/minimumChannel")? < number(output, "/maximumChannel")?
            && number(output, "/maximumChannel")? <= 255,
        "retained RGB range",
    )?;
    for field in ["rgbSha256", "pngSha256"] {
        require(
            is_hex(text(output, &format!("/{field}"))?, 64),
            "retained output digest",
        )?;
    }
    require(
        number(output, "/pngBytes")? > 0
            && !text(output, "/path")?.is_empty()
            && !text(output, "/rgbPath")?.is_empty(),
        "retained output receipt",
    )?;
    let cleanup = &receipt["cleanup"];
    require(
        cleanup["adapterProcessExited"] == true
            && receipt["providerDrop"]["generatorDropped"] == true,
        "endpoint process/provider cleanup",
    )?;
    let tolerance = number(cleanup, "/toleranceBytes")?;
    require(
        tolerance <= 64 * 1024 * 1024
            && number(cleanup, "/beforeBytes")?
                .checked_add(tolerance)
                .is_some_and(|limit| {
                    number(cleanup, "/afterBytes").is_ok_and(|after| after <= limit)
                }),
        "endpoint cleanup residency",
    )?;
    Ok(())
}

fn validate_inventory(value: &Value) -> Result<Vec<EndpointArtifactFile>, String> {
    let files: Vec<EndpointArtifactFile> = serde_json::from_value(value["entries"].clone())
        .map_err(|error| format!("SDXL endpoints: artifact inventory entries: {error}"))?;
    require(
        !files.is_empty() && files.len() as u64 == number(value, "/files")?,
        "inventory file count",
    )?;
    let mut seen = BTreeSet::new();
    let mut total = 0u64;
    for file in &files {
        require(
            !file.path.is_empty()
                && !file
                    .path
                    .chars()
                    .any(|c| c.is_control() || c == '\\' || c == ':')
                && file
                    .path
                    .split('/')
                    .all(|part| !part.is_empty() && part != "." && part != "..")
                && seen.insert(file.path.as_str()),
            "inventory path must be unique and confined",
        )?;
        require(
            file.bytes > 0 && is_hex(&file.sha256, 64),
            "inventory file size/digest",
        )?;
        total = total
            .checked_add(file.bytes)
            .ok_or("SDXL endpoints: inventory byte overflow")?;
    }
    // Shared snapshots can contain unrelated cached files. The provider consumes exactly these
    // three shared files; only the base tier is a complete directory source in its artifact seal.
    let shared_files = [
        "components/tokenizer_clip_l/tokenizer.json",
        "components/tokenizer_clip_bigg/tokenizer.json",
        "components/vae_fp16_fix/diffusion_pytorch_model.safetensors",
    ];
    require(
        files.iter().any(|file| file.path.starts_with("base/"))
            && shared_files
                .iter()
                .all(|path| files.iter().any(|file| file.path == *path))
            && files.iter().all(|file| {
                file.path.starts_with("base/") || shared_files.contains(&file.path.as_str())
            }),
        "inventory must bind the full base tier and exactly the consumed shared files",
    )?;
    require(total == number(value, "/bytes")?, "inventory byte total")?;
    require(
        inventory_digest(&files) == text(value, "/sha256")?,
        "ordered inventory composite digest",
    )?;
    Ok(files)
}

fn inventory_digest(files: &[EndpointArtifactFile]) -> String {
    let mut hash = Sha256::new();
    // Consume the recorded order. Do not reproduce the capture host's path sorting/locale here.
    for file in files {
        hash.update(file.path.as_bytes());
        hash.update(b"\0");
        hash.update(file.bytes.to_string().as_bytes());
        hash.update(b"\0");
        hash.update(file.sha256.as_bytes());
        hash.update(b"\n");
    }
    format!("{hash:x}", hash = hash.finalize())
}

fn expected_strategy(name: &str) -> Value {
    if name == "resident" {
        json!({"rung":"resident","engagedRungs":["resident"],"parameters":{}})
    } else {
        json!({"rung":"bounded_attention","engagedRungs":["resident","staged_residency","bounded_decode","bounded_attention"],"parameters":{"stageResidency":true,"decodeTileEdge":512,"decodeOverlap":64,"attentionChunkSize":67_108_864}})
    }
}

fn validate_comparison(value: &Value) -> Result<(), String> {
    require(
        value["result"] == "measured"
            && value["reference"] == "resident"
            && value["selected"] == "sequential"
            && value["channels"] == RGB_BYTES,
        "RGB comparison identity",
    )?;
    let changed = number(value, "/changedChannels")?;
    let maximum = number(value, "/maximumAbsoluteRgbDelta255")?;
    let mean = value["meanAbsoluteRgbDelta255"]
        .as_f64()
        .ok_or("SDXL endpoints: RGB mean")?;
    require(
        changed <= RGB_BYTES
            && maximum <= 255
            && mean >= 0.0
            && mean <= maximum as f64
            && value["identical"] == (changed == 0)
            && (maximum == 0) == (changed == 0),
        "RGB comparison measurements",
    )?;
    require(
        value["maximumAbsoluteError"]
            .as_f64()
            .is_some_and(|v| (v - maximum as f64 / 255.0).abs() < 1e-12)
            && value["meanAbsoluteError"]
                .as_f64()
                .is_some_and(|v| (v - mean / 255.0).abs() < 1e-12),
        "RGB comparison normalization",
    )
}

fn resource<'a>(sources: &'a [(&str, &str)], path: &str) -> Result<&'a str, String> {
    let matches: Vec<_> = sources.iter().filter(|(name, _)| *name == path).collect();
    require(matches.len() == 1, "source is not uniquely compiled")?;
    Ok(matches[0].1)
}
fn authenticated_json(sources: &[(&str, &str)], path: &str, sha256: &str) -> Result<Value, String> {
    let raw = resource(sources, path)?;
    require(
        is_hex(sha256, 64) && format!("{:x}", Sha256::digest(raw.as_bytes())) == sha256,
        "source digest mismatch",
    )?;
    serde_json::from_str(raw).map_err(|e| format!("SDXL endpoint source JSON: {e}"))
}
fn is_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn text<'a>(value: &'a Value, pointer: &str) -> Result<&'a str, String> {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("SDXL endpoints: missing string {pointer}"))
}
fn number(value: &Value, pointer: &str) -> Result<u64, String> {
    value
        .pointer(pointer)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("SDXL endpoints: missing integer {pointer}"))
}
fn require(ok: bool, reason: &str) -> Result<(), String> {
    if ok {
        Ok(())
    } else {
        Err(format!("SDXL endpoints: {reason}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Synthetic mutations of a retained gated record exercise validation only. No fixture is
    // written to the production config and none of these byte values claims a physical capture.
    struct Fixture {
        config: Value,
        bundles: [Value; 2],
        summary: Value,
    }

    fn fixture_inventory() -> Value {
        let entries: Vec<EndpointArtifactFile> = [
            "base/weights.safetensors",
            "components/tokenizer_clip_l/tokenizer.json",
            "components/tokenizer_clip_bigg/tokenizer.json",
            "components/vae_fp16_fix/diffusion_pytorch_model.safetensors",
        ]
        .into_iter()
        .map(|path| EndpointArtifactFile {
            path: path.into(),
            bytes: 1000,
            sha256: "a".repeat(64),
        })
        .collect();
        json!({"files":entries.len(), "bytes":4000, "sha256":inventory_digest(&entries), "entries":entries.iter().map(|file| json!({"path":file.path,"bytes":file.bytes,"sha256":file.sha256})).collect::<Vec<_>>()})
    }

    fn fixture() -> Fixture {
        let inventory = fixture_inventory();
        let base: Value = serde_json::from_str(include_str!(
            "../../../docs/calibration/sc-22738/sdxl-q4-candle-evidence.json"
        ))
        .unwrap();
        let mut bundles = [base.clone(), base];
        let mut anchors = Vec::new();
        let mut receipts = Vec::new();
        for (index, name) in ["resident", "sequential"].into_iter().enumerate() {
            let bounded = index == 1;
            let peak = if bounded {
                4_000_000_000u64
            } else {
                12_000_000_000u64
            };
            let record = &mut bundles[index]["records"][0];
            record["id"] = json!(format!("imc-{:020x}", index + 1));
            record["logicalCaseId"] = json!(format!("implan-{:020x}", index + 1));
            record["calibrationFingerprint"] = json!(FINGERPRINT);
            record["fixture"] = json!(FIXTURE);
            record["loadShape"] = json!("eager_materialization");
            record["artifact"]["inventorySha256"] = inventory["sha256"].clone();
            record["hardware"]["deviceId"] = json!(GPU_UUID);
            record["target"]["geometry"] = json!({"width":1024,"height":1024,"batch":1,"frames":1});
            record["strategy"] = expected_strategy(name);
            record["sweep"] = json!({"axes":[],"cases":[{"parameters": expected_strategy(name)["parameters"], "result":"passed"}],"rangeVerified":false});
            for phase in ["conditioning", "denoise", "decode", "overall"] {
                record["observedMemory"][phase] =
                    json!({"activeBytes":peak,"allocatorBytes":peak,"reclaimableBytes":0});
            }
            for measurement in record["diagnostics"]["measurements"]
                .as_array_mut()
                .unwrap()
            {
                if measurement["unit"] == "bytes" {
                    measurement["value"] = json!(peak);
                }
            }
            anchors.push(json!({
                "id":format!("sdxl-{name}"), "modelId":"sdxl", "modelFamily":"sdxl",
                "route":"sdxl", "provider":"sdxl", "backend":"candle", "tier":"q4",
                "transformerVariant":null, "decoder":null, "mode":"text_to_image",
                "overlay":null, "referenceCount":0, "loadShape":record["loadShape"],
                "measuredRegime":{"decodeTiled":bounded,"transformerWindowed":false,"staged":bounded,"attentionChunked":bounded},
                "source":{"path":format!("{name}.json"),"sha256":"", "recordId":record["id"],"calibrationFingerprint":FINGERPRINT,"loaderClosureDigest":"b".repeat(64)},
                "geometry":{"width":1024,"height":1024,"frames":1,"fps":null},
                "phaseActivePeakBytes":{"conditioning":peak,"denoise":peak,"decode":peak},
                "phaseAllocatorEnvelopeBytes":{"conditioning":peak,"denoise":peak,"decode":peak},
                "overallAllocatorEnvelopeBytes":peak
            }));
            receipts.push(json!({
                "engineBuild":{"adapterCandleFeature":true,"sdxlFlashAttentionKernelCompiled":false,"cudaComputeCapabilityCompileTarget":"120"},
                "endpoint":name,"deviceUuid":GPU_UUID,"cudaContextUuid":GPU_UUID,"logicalCudaDevice":0,
                "inferencePin":record["repositories"]["inference"]["revision"],"offloadPolicy":name,
                "recordId":record["id"],"evidenceSha256":"",
                "loadPeakBytes":1_000_000_000u64,"wholeRunPeakBytes":peak,"baselineBytes":1024,"wholeDevicePeakBytes":peak+1024,
                "settings":{"prompt":PROMPT,"negativePrompt":null,"seed":16402,"steps":30,"guidance":7.0,"sampler":null,"scheduler":null,"flashAttn":true,"count":1,"cfgInterval":null,"guidanceMethod":null,"schedulerShift":null,"memory":name},
                "planned":{"target":record["target"],"strategy":record["strategy"],"fixture":record["fixture"],"loadShape":record["loadShape"],"calibrationFingerprint":record["calibrationFingerprint"]},
                "output":{"path":format!("{name}.png"),"rgbPath":format!("{name}.rgb"),"width":1024,"height":1024,"channels":3,"rgbBytes":RGB_BYTES,"spatiallyNonFlat":true,"minimumChannel":0,"maximumChannel":255,"rgbSha256":"c".repeat(64),"pngSha256":"d".repeat(64),"pngBytes":1000},
                "cleanup":{"adapterProcessExited":true,"beforeBytes":1024,"afterBytes":1024,"toleranceBytes":64*1024*1024},
                "providerDrop":{"generatorDropped":true,"beforeBytes":1024,"afterBytes":1024}
            }));
        }
        Fixture {
            config: json!({"schemaVersion":1,"pair":{"resident":anchors[0],"sequential":anchors[1],"summary":{"path":"summary.json","sha256":""}}}),
            summary: json!({"schemaVersion":1,"profile":"sdxl-cuda-endpoints","inferencePin":bundles[0]["records"][0]["repositories"]["inference"]["revision"],"artifactInventory":inventory,"reserveBytes":2u64*1024*1024*1024,"reserveIncludedInMeasuredPeaks":false,"resolvedDefaults":{"steps":30,"guidance":7.0,"sampler":"euler"},"endpoints":receipts,"qualityComparison":{"result":"measured","reference":"resident","selected":"sequential","channels":RGB_BYTES,"maximumAbsoluteRgbDelta255":0,"meanAbsoluteRgbDelta255":0.0,"maximumAbsoluteError":0.0,"meanAbsoluteError":0.0,"changedChannels":0,"identical":true}}),
            bundles,
        }
    }

    impl Fixture {
        fn sealed(&self) -> (String, [String; 3]) {
            let mut config = self.config.clone();
            let mut summary = self.summary.clone();
            let resident = self.bundles[0].to_string();
            let sequential = self.bundles[1].to_string();
            for (index, (name, raw)) in [("resident", &resident), ("sequential", &sequential)]
                .into_iter()
                .enumerate()
            {
                let digest = format!("{:x}", Sha256::digest(raw.as_bytes()));
                config["pair"][name]["source"]["sha256"] = json!(digest);
                summary["endpoints"][index]["evidenceSha256"] = json!(digest);
            }
            let summary = summary.to_string();
            config["pair"]["summary"]["sha256"] =
                json!(format!("{:x}", Sha256::digest(summary.as_bytes())));
            (config.to_string(), [resident, sequential, summary])
        }
        fn load(&self) -> Result<Option<SdxlCudaMemoryEndpoints>, String> {
            let (config, resources) = self.sealed();
            load_sdxl_cuda_memory_endpoints(
                &config,
                &[
                    ("resident.json", &resources[0]),
                    ("sequential.json", &resources[1]),
                    ("summary.json", &resources[2]),
                ],
            )
        }
    }

    #[test]
    fn sdxl_endpoint_empty_package_has_no_authority() {
        assert!(
            load_sdxl_cuda_memory_endpoints(r#"{"schemaVersion":1,"pair":null}"#, &[])
                .unwrap()
                .is_none()
        );
        assert!(load_sdxl_cuda_memory_endpoints(r#"{"schemaVersion":1}"#, &[]).is_err());
        assert!(
            load_sdxl_cuda_memory_endpoints(r#"{"schemaVersion":2,"pair":null}"#, &[]).is_err()
        );
    }

    #[test]
    fn sdxl_endpoint_packaged_pair_validates_against_compiled_sources() {
        let declared: Value = serde_json::from_str(PACKAGED_SDXL_CUDA_MEMORY_ENDPOINTS).unwrap();
        let loaded = load_sdxl_cuda_memory_endpoints(
            PACKAGED_SDXL_CUDA_MEMORY_ENDPOINTS,
            PACKAGED_ENDPOINT_SOURCES,
        )
        .expect("packaged endpoint data must pass the full source and summary handshake");
        assert_eq!(loaded.is_some(), !declared["pair"].is_null());
    }

    #[test]
    fn sdxl_endpoint_gated_pair_preserves_raw_measurements_without_certification() {
        let fixture = fixture();
        let pair = fixture.load().unwrap().unwrap();
        assert_eq!(
            pair.resident.overall_allocator_envelope_bytes,
            12_000_000_000
        );
        assert_eq!(
            pair.sequential.overall_allocator_envelope_bytes,
            4_000_000_000
        );
        assert_eq!(pair.resident_load_peak_bytes, 1_000_000_000);
        assert_eq!(pair.sequential_parameters["stageResidency"], true);
        assert_eq!(
            pair.resident.load_shape,
            AnchorLoadShape::EagerMaterialization
        );
        assert_eq!(
            pair.sequential.load_shape,
            AnchorLoadShape::EagerMaterialization
        );
        for bundle in &fixture.bundles {
            assert_eq!(bundle["records"][0]["status"], "gated");
            assert_eq!(bundle["records"][0]["predictedPeakBytes"], Value::Null);
            assert_eq!(bundle["records"][0]["quality"]["result"], "not_run");
        }
    }

    #[test]
    fn sdxl_endpoint_rejects_unbound_sources_and_summary() {
        let (config, mut sources) = fixture().sealed();
        for index in 0..3 {
            sources[index].push(' ');
            assert!(load_sdxl_cuda_memory_endpoints(
                &config,
                &[
                    ("resident.json", &sources[0]),
                    ("sequential.json", &sources[1]),
                    ("summary.json", &sources[2])
                ]
            )
            .is_err());
            sources[index].pop();
        }
        assert!(load_sdxl_cuda_memory_endpoints(&config, &[]).is_err());
        let mut missing_pair = fixture().config;
        missing_pair["pair"]
            .as_object_mut()
            .unwrap()
            .remove("sequential");
        assert!(load_sdxl_cuda_memory_endpoints(&missing_pair.to_string(), &[]).is_err());
    }

    #[test]
    fn sdxl_endpoint_rejects_resealed_identity_and_measurement_mutations() {
        let mutations = [
            ("/endpoints/1/cudaContextUuid", json!("GPU-other")),
            (
                "/endpoints/1/engineBuild/sdxlFlashAttentionKernelCompiled",
                json!(true),
            ),
            ("/endpoints/1/recordId", json!("imc-00000000000000000001")),
            ("/endpoints/1/inferencePin", json!("f".repeat(40))),
            (
                "/endpoints/1/planned/strategy/parameters/attentionChunkSize",
                json!(1024),
            ),
            ("/endpoints/1/planned/target/geometry/batch", json!(2)),
            ("/endpoints/1/settings/seed", json!(42)),
            ("/endpoints/1/settings/flashAttn", json!(false)),
            ("/endpoints/1/settings/prompt", json!("a different prompt")),
            ("/endpoints/1/output/pngSha256", Value::Null),
            ("/endpoints/1/output/spatiallyNonFlat", json!(false)),
            ("/endpoints/1/loadPeakBytes", json!(5_000_000_000u64)),
            ("/endpoints/1/wholeDevicePeakBytes", json!(1)),
            ("/endpoints/1/cleanup/adapterProcessExited", json!(false)),
            ("/endpoints/1/cleanup/afterBytes", json!(1024 * 1024 * 1024)),
            ("/artifactInventory/sha256", json!("f".repeat(64))),
            ("/reserveIncludedInMeasuredPeaks", json!(true)),
            ("/qualityComparison/result", json!("passed")),
        ];
        for (path, value) in mutations {
            let mut fixture = fixture();
            *fixture.summary.pointer_mut(path).unwrap() = value;
            assert!(fixture.load().is_err(), "accepted mutation {path}");
        }
        for (path, value) in [
            ("/records/0/hardware/deviceId", json!("1")),
            ("/records/0/repositories/inference/dirty", json!(true)),
            ("/records/0/artifact/inventorySha256", json!("f".repeat(64))),
            ("/records/0/target/geometry/batch", json!(2)),
            (
                "/records/0/strategy/parameters/stageResidency",
                json!(false),
            ),
            ("/records/0/observedMemory/overall/activeBytes", json!(1)),
        ] {
            let mut fixture = fixture();
            *fixture.bundles[1].pointer_mut(path).unwrap() = value;
            assert!(fixture.load().is_err(), "accepted source mutation {path}");
        }
    }

    #[test]
    fn sdxl_endpoint_inventory_binds_order_contents_and_complete_source_assembly() {
        let inventory = fixture_inventory();
        assert_eq!(validate_inventory(&inventory).unwrap().len(), 4);
        for (pointer, replacement) in [
            ("/entries/0/path", json!("../weights.safetensors")),
            (
                "/entries/0/path",
                json!("components/tokenizer_clip_l/tokenizer.json"),
            ),
            ("/entries/0/path", json!("other/weights.safetensors")),
            ("/entries/0/bytes", json!(1001)),
            ("/entries/0/sha256", json!("b".repeat(64))),
            ("/files", json!(3)),
            ("/bytes", json!(3999)),
        ] {
            let mut changed = inventory.clone();
            *changed.pointer_mut(pointer).unwrap() = replacement;
            assert!(validate_inventory(&changed).is_err(), "accepted {pointer}");
        }
        let mut extra = inventory.clone();
        extra["entries"].as_array_mut().unwrap().push(json!({
            "path":"components/tokenizer_clip_l/config.json", "bytes":1000, "sha256":"a".repeat(64)
        }));
        extra["files"] = json!(5);
        extra["bytes"] = json!(5000);
        let entries: Vec<EndpointArtifactFile> =
            serde_json::from_value(extra["entries"].clone()).unwrap();
        extra["sha256"] = json!(inventory_digest(&entries));
        assert!(validate_inventory(&extra)
            .unwrap_err()
            .contains("exactly the consumed shared files"));
        let mut reordered = inventory;
        reordered["entries"].as_array_mut().unwrap().swap(0, 1);
        assert!(validate_inventory(&reordered)
            .unwrap_err()
            .contains("ordered inventory composite"));
    }

    #[test]
    fn sdxl_endpoint_rejects_consistently_deferred_capture_for_stock_eager_request() {
        let mut fixture = fixture();
        for (index, name) in ["resident", "sequential"].into_iter().enumerate() {
            fixture.config["pair"][name]["loadShape"] = json!("deferred_materialization");
            fixture.bundles[index]["records"][0]["loadShape"] = json!("deferred_materialization");
            fixture.summary["endpoints"][index]["planned"]["loadShape"] =
                json!("deferred_materialization");
        }
        let error = fixture.load().unwrap_err();
        assert!(error.contains("ordinary SDXL eager load shape"), "{error}");
    }

    #[test]
    fn sdxl_endpoint_does_not_relax_runtime_complete_validation() {
        let mut fixture = fixture();
        fixture.bundles[1]["records"][0]["status"] = json!("runtime_complete");
        assert!(fixture.load().is_err());
    }
}
