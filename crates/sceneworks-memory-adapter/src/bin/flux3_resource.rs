//! Supplemental FLUX3 request binding, never a calibration-anchor replacement.
use super::*;
use sha2::{Digest, Sha256};
use std::path::Path;

pub(super) struct BoundCase {
    pub value: Value,
    pub digest: String,
    pub inputs: Vec<Value>,
    pub request: GenerationRequest,
    pub adapters: Vec<core::AdapterSpec>,
    pub pup_root: Option<PathBuf>,
    pub reference_resources: Option<core::MemoryReferenceResourceFacts>,
}

pub(super) fn canonical_bytes(value: &Value) -> Result<Vec<u8>, String> {
    // Match the shared Node canonical JSON, including integral floats and ECMAScript's
    // decimal/exponent thresholds (serde_json otherwise hashes 1.0 differently from 1).
    fn write(value: &Value, out: &mut String) -> Result<(), String> {
        match value {
            Value::Object(object) => {
                out.push('{');
                for (index, (key, value)) in object
                    .iter()
                    .collect::<std::collections::BTreeMap<_, _>>()
                    .into_iter()
                    .enumerate()
                {
                    if index != 0 {
                        out.push(',');
                    }
                    out.push_str(&serde_json::to_string(key).map_err(blocked)?);
                    out.push(':');
                    write(value, out)?;
                }
                out.push('}');
            }
            Value::Array(items) => {
                out.push('[');
                for (index, item) in items.iter().enumerate() {
                    if index != 0 {
                        out.push(',');
                    }
                    write(item, out)?;
                }
                out.push(']');
            }
            Value::Number(number) => {
                if let Some(n) = number.as_i64() {
                    if n.unsigned_abs() > 9_007_199_254_740_991 {
                        return Err(blocked("canonical integer exceeds shared safe range"));
                    }
                    out.push_str(&n.to_string());
                } else if let Some(n) = number.as_u64() {
                    if n > 9_007_199_254_740_991 {
                        return Err(blocked("canonical integer exceeds shared safe range"));
                    }
                    out.push_str(&n.to_string());
                } else {
                    let n = number
                        .as_f64()
                        .ok_or_else(|| blocked("nonfinite canonical number"))?;
                    if n == 0. {
                        out.push('0');
                    } else if n.abs() < 1e-6 || n.abs() >= 1e21 {
                        let s = format!("{n:e}");
                        let (mantissa, exponent) = s
                            .split_once('e')
                            .ok_or_else(|| blocked("canonical exponent"))?;
                        let exponent: i32 = exponent.parse().map_err(blocked)?;
                        out.push_str(mantissa);
                        out.push('e');
                        if exponent >= 0 {
                            out.push('+');
                        }
                        out.push_str(&exponent.to_string());
                    } else {
                        out.push_str(&n.to_string());
                    }
                }
            }
            _ => out.push_str(&serde_json::to_string(value).map_err(blocked)?),
        }
        Ok(())
    }
    let mut out = String::new();
    write(value, &mut out)?;
    Ok(out.into_bytes())
}
pub(super) fn json_digest(domain: &[u8], value: &Value) -> Result<String, String> {
    let mut hash = Sha256::new();
    hash.update(domain);
    hash.update(canonical_bytes(value)?);
    Ok(format!("{:x}", hash.finalize()))
}

fn blocked(detail: impl std::fmt::Display) -> String {
    format!("BLOCKED: FLUX3 supplemental resource case: {detail}")
}
fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| blocked(format!("{key} must be a nonempty string")))
}
fn exact(value: &Value, allowed: &[&str]) -> Result<(), String> {
    let object = value
        .as_object()
        .ok_or_else(|| blocked("expected object"))?;
    if let Some(key) = object.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(blocked(format!("unknown field {key}")));
    }
    Ok(())
}
fn confined(root: &Path, raw: &str) -> Result<PathBuf, String> {
    if raw.contains("://") || raw.starts_with("//") || raw.starts_with("\\\\") {
        return Err(blocked(
            "network paths are not caller-provisioned local inputs",
        ));
    }
    let path = std::fs::canonicalize(root.join(raw)).map_err(blocked)?;
    if !path.starts_with(root) {
        return Err(blocked("input escaped caller-provided root"));
    }
    Ok(path)
}
fn file_names(
    root: &Path,
    prefix: &Path,
    names: &mut std::collections::BTreeSet<String>,
) -> Result<(), String> {
    for entry in std::fs::read_dir(root.join(prefix)).map_err(blocked)? {
        let entry = entry.map_err(blocked)?;
        let relative = prefix.join(entry.file_name());
        let ty = entry.file_type().map_err(blocked)?;
        if ty.is_dir() {
            file_names(root, &relative, names)?;
        } else {
            let resolved = confined(
                root,
                relative
                    .to_str()
                    .ok_or_else(|| blocked("non-UTF8 inventory path"))?,
            )?;
            if !resolved.is_file() {
                return Err(blocked("directory links are not inventory files"));
            }
            names.insert(relative.to_string_lossy().replace('\\', "/"));
        }
    }
    Ok(())
}
fn binding(root: &Path, value: &Value) -> Result<(PathBuf, Value), String> {
    let path = confined(root, string(value, "path")?)?;
    let expected = string(value, "sha256")?;
    if expected.len() != 64
        || !expected
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(blocked("full lowercase SHA256 required"));
    }
    let actual = core::flux3::conversion::hash_source_file(&path).map_err(blocked)?;
    if actual.sha256 != expected
        || value
            .get("bytes")
            .and_then(Value::as_u64)
            .is_some_and(|n| n != actual.bytes)
    {
        return Err(blocked(
            "input bytes/checksum differ from provided inventory",
        ));
    }
    let inventory = json!({"path":path, "bytes":actual.bytes, "sha256":actual.sha256});
    Ok((path, inventory))
}

/// Full actual inventories of the exact package and external TE/AE opened by LoadSpec. These
/// observations do not replace trusted catalog hashes, and are never emitted before generation.
pub(super) fn model_inventory() -> Result<Value, String> {
    fn tree(root: &Path, prefix: &Path, files: &mut Vec<Value>) -> Result<(), String> {
        let mut entries = std::fs::read_dir(root.join(prefix))
            .map_err(blocked)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(blocked)?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let relative = prefix.join(entry.file_name());
            let ty = entry.file_type().map_err(blocked)?;
            if ty.is_dir() {
                tree(root, &relative, files)?;
            } else {
                if ty.is_symlink() && entry.path().is_dir() {
                    return Err(blocked("directory links are not inventory files"));
                }
                let file =
                    core::flux3::conversion::hash_source_file(&entry.path()).map_err(blocked)?;
                files.push(json!({"path":relative.to_string_lossy().replace('\\',"/"),"bytes":file.bytes,"sha256":file.sha256}));
            }
        }
        Ok(())
    }
    let original = PathBuf::from(protocol::required_env(
        "SCENEWORKS_FLUX3_KLEIN_ORIGINAL_ROOT",
    )?);
    let package = PathBuf::from(protocol::required_env("SCENEWORKS_FLUX3_KLEIN_ROOT")?);
    let mut package_files = Vec::new();
    // Dense originals include external components themselves; count them as separate roles rather
    // than accidentally pricing the entire source repository as the DiT package.
    if package == original {
        for name in ["config.json", "model.safetensors"] {
            let file =
                core::flux3::conversion::hash_source_file(&package.join(name)).map_err(blocked)?;
            package_files.push(json!({"path":name,"bytes":file.bytes,"sha256":file.sha256}));
        }
    } else {
        tree(&package, Path::new(""), &mut package_files)?;
    }
    let mut encoder_files = Vec::new();
    tree(
        &original.join("text_encoder"),
        Path::new(""),
        &mut encoder_files,
    )?;
    let ae = core::flux3::conversion::hash_source_file(&original.join("ae.safetensors"))
        .map_err(blocked)?;
    Ok(json!({"package":{"root":package,"files":package_files},
        "text_encoder":{"root":original.join("text_encoder"),"files":encoder_files},
        "autoencoder":{"root":original,"files":[{"path":"ae.safetensors","bytes":ae.bytes,"sha256":ae.sha256}]}}))
}

/// Input is the normalized shared Node contract, but the native consumer independently checks
/// identities and actual local bytes before any registry load. No receipt supplies trusted pins.
pub(super) fn bind(
    value: &Value,
    lane: &str,
    checkpoint: Checkpoint,
    tier: &str,
    root: &Path,
) -> Result<BoundCase, String> {
    exact(
        value,
        &[
            "schemaVersion",
            "caseId",
            "backend",
            "modelId",
            "tier",
            "scenario",
            "generation",
            "references",
            "adapters",
            "components",
        ],
    )?;
    if value["schemaVersion"] != 1
        || value["backend"] != lane
        || value["modelId"] != checkpoint.provider
        || value["tier"] != tier
    {
        return Err(blocked(
            "case model/backend/tier differs from loaded target",
        ));
    }
    let label = if checkpoint.provider == BASE_PROVIDER {
        "base"
    } else {
        "distilled"
    };
    let scenario = string(value, "scenario")?;
    if !scenario
        .bytes()
        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
    {
        return Err(blocked("invalid explicit scenario"));
    }
    if string(value, "caseId")? != format!("{lane}.{label}.{scenario}.{tier}") {
        return Err(blocked("caseId identity mismatch"));
    }
    let generation = &value["generation"];
    exact(
        generation,
        &[
            "prompt", "width", "height", "seed", "steps", "guidance", "pup", "layout",
        ],
    )?;
    let integer = |name: &str| {
        generation[name]
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| blocked(format!("{name} must be u32")))
    };
    let width = integer("width")?;
    let height = integer("height")?;
    if width < 64 || height < 64 || width % 16 != 0 || height % 16 != 0 {
        return Err(blocked("invalid native geometry"));
    }
    let guidance = match generation.get("guidance") {
        None => None,
        Some(value) => Some(
            value
                .as_f64()
                .filter(|v| v.is_finite() && *v >= 0.)
                .ok_or_else(|| blocked("invalid guidance"))? as f32,
        ),
    };
    if checkpoint.provider == PROVIDER && guidance.is_some() {
        return Err(blocked("distilled guidance must be omitted"));
    }
    if checkpoint.provider == BASE_PROVIDER && guidance.is_none() {
        return Err(blocked("normalized base guidance is required"));
    }
    let mut request = GenerationRequest {
        prompt: string(generation, "prompt")?.into(),
        width,
        height,
        count: 1,
        seed: Some(
            generation["seed"]
                .as_u64()
                .ok_or_else(|| blocked("normalized seed required"))?,
        ),
        steps: Some(integer("steps")?),
        guidance,
        enhance_prompt: generation["pup"]
            .as_bool()
            .ok_or_else(|| blocked("normalized boolean pup required"))?,
        ..Default::default()
    };
    if request.steps == Some(0) {
        return Err(blocked("steps must be positive"));
    }
    let root = std::fs::canonicalize(root).map_err(blocked)?;
    let references = value["references"]
        .as_array()
        .ok_or_else(|| blocked("ordered references required"))?;
    if references.len() > 10 {
        return Err(blocked("more than 10 references"));
    }
    let mut ids = std::collections::BTreeSet::new();
    let mut inputs = Vec::new();
    for (index, reference) in references.iter().enumerate() {
        exact(
            reference,
            &["instanceId", "assetId", "index", "path", "sha256"],
        )?;
        let id = string(reference, "instanceId")?;
        if !ids.insert(id) || reference["index"].as_u64() != Some(index as u64) {
            return Err(blocked("duplicate instance or reference order mismatch"));
        }
        string(reference, "assetId")?;
        let (path, mut inventory) = binding(&root, reference)?;
        let conditioning =
            core::flux3::references::decode_reference_file(&path).map_err(blocked)?;
        let size = match &conditioning {
            core::Conditioning::Reference { image, .. } => (image.width, image.height),
            core::Conditioning::ReferenceRgba { image, .. } => (image.width, image.height),
            _ => return Err(blocked("decoder did not return a source-mode reference")),
        };
        if generation.get("layout").is_some() {
            request
                .flux3_layout_bindings
                .push(core::flux3::layouts::ReferenceBinding::oriented(id, size));
        }
        inventory["decodedCarrier"] = reference_carrier(&conditioning)
            .ok_or_else(|| blocked("missing native reference carrier"))?;
        request.conditioning.push(conditioning);
        inventory["role"] = json!("reference");
        inventory["instanceId"] = json!(id);
        inventory["assetId"] = reference["assetId"].clone();
        inventory["index"] = json!(index);
        inputs.push(inventory);
    }
    if let Some(layout) = generation.get("layout") {
        request.flux3_layout =
            Some(core::flux3::layouts::Layout::from_json(layout.clone()).map_err(blocked)?);
    }
    core::flux3::references::normalize_request(&mut request).map_err(blocked)?;
    let mut adapters = Vec::new();
    for (index, adapter) in value["adapters"]
        .as_array()
        .ok_or_else(|| blocked("normalized adapters required"))?
        .iter()
        .enumerate()
    {
        exact(adapter, &["path", "sha256", "scale"])?;
        let (path, mut inventory) = binding(&root, adapter)?;
        let scale = adapter["scale"]
            .as_f64()
            .filter(|n| n.is_finite())
            .ok_or_else(|| blocked("finite adapter scale required"))? as f32;
        if !scale.is_finite() {
            return Err(blocked("adapter scale exceeds native precision"));
        }
        adapters.push(core::AdapterSpec::new(path, scale, core::AdapterKind::Lora));
        inventory["role"] = json!("adapter");
        inventory["index"] = json!(index);
        inventory["scale"] = json!(scale);
        inputs.push(inventory);
    }
    exact(&value["components"], &["prompt_upsampler"])?;
    let pup_root = if request.enhance_prompt {
        let component = &value["components"]["prompt_upsampler"];
        exact(
            component,
            &["root", "repository", "resolvedRevision", "inventory"],
        )?;
        if component["repository"] != core::flux3::pup::UPSAMPLER_REPO
            || component["resolvedRevision"] != core::flux3::pup::UPSAMPLER_REVISION
        {
            return Err(blocked("PUP checkpoint identity mismatch"));
        }
        let component_root = confined(&root, string(component, "root")?)?;
        let files = component["inventory"]
            .as_array()
            .filter(|v| !v.is_empty())
            .ok_or_else(|| blocked("PUP full inventory required"))?;
        let mut names = std::collections::BTreeSet::new();
        for file in files {
            exact(file, &["path", "bytes", "sha256"])?;
            if !names.insert(string(file, "path")?) {
                return Err(blocked("duplicate PUP inventory path"));
            }
            let (_, mut inventory) = binding(&component_root, file)?;
            inventory["role"] = json!("prompt_upsampler");
            inputs.push(inventory);
        }
        for name in core::flux3::pup::UPSAMPLER_REQUIRED_FILES {
            if !names.contains(name) {
                return Err(blocked(format!("missing PUP inventory {name}")));
            }
        }
        let mut actual_names = std::collections::BTreeSet::new();
        file_names(&component_root, Path::new(""), &mut actual_names)?;
        if actual_names != names.into_iter().map(str::to_owned).collect() {
            return Err(blocked(
                "PUP inventory does not cover complete actual directory",
            ));
        }
        Some(component_root)
    } else {
        if value["components"].get("prompt_upsampler").is_some() {
            return Err(blocked("PUP bypass must omit auxiliary component"));
        }
        None
    };
    if tier == "bf16" {
        let count = request.image_reference_count();
        if scenario == "reference1"
            && (count != 1 || request.enhance_prompt || !adapters.is_empty())
        {
            return Err(blocked("reference1 obligation requires its distinct single-reference preprocessing request"));
        }
        if scenario == "reference10"
            && (count != 10 || request.enhance_prompt || !adapters.is_empty())
        {
            return Err(blocked(
                "reference10 obligation is not its declared actual request",
            ));
        }
        if scenario == "adapter" && (adapters.is_empty() || count != 0 || request.enhance_prompt) {
            return Err(blocked(
                "adapter obligation is not its declared actual request",
            ));
        }
        if checkpoint.provider == BASE_PROVIDER && scenario.starts_with("pup_") {
            let editing = scenario.contains("edit");
            let boxes = scenario.ends_with("_boxes");
            if !request.enhance_prompt
                || count != if editing { 10 } else { 0 }
                || boxes != request.flux3_layout.is_some()
                || !adapters.is_empty()
            {
                return Err(blocked(
                    "PUP obligation mode differs from actual references/layout",
                ));
            }
        }
    }
    let mut hasher = Sha256::new();
    hasher.update(b"sceneworks-flux3-supplemental-request-v1\0");
    hasher.update(canonical_bytes(value)?);
    let reference_resources = core::flux3::references::request_plan(&request)
        .map_err(blocked)?
        .map(|plan| plan.memory_resources())
        .transpose()
        .map_err(blocked)?;
    Ok(BoundCase {
        value: value.clone(),
        digest: format!("{:x}", hasher.finalize()),
        inputs,
        request,
        adapters,
        pup_root,
        reference_resources,
    })
}

fn reference_carrier(value: &core::Conditioning) -> Option<Value> {
    let (kind, width, height, pixels) = match value {
        core::Conditioning::Reference { image, .. } => {
            ("RGB8", image.width, image.height, image.pixels.as_slice())
        }
        core::Conditioning::ReferenceRgba { image, .. } => {
            ("RGBA8", image.width, image.height, image.pixels.as_slice())
        }
        _ => return None,
    };
    let mut hash = Sha256::new();
    hash.update(b"flux3_native_reference_carrier_v1\0");
    hash.update(kind.as_bytes());
    hash.update([0]);
    hash.update(width.to_le_bytes());
    hash.update(height.to_le_bytes());
    hash.update((pixels.len() as u64).to_le_bytes());
    hash.update(pixels);
    Some(
        json!({"kind":kind,"width":width,"height":height,"bytes":pixels.len(),"sha256":format!("{:x}",hash.finalize())}),
    )
}
pub(super) fn actual_request_observation(request: &GenerationRequest) -> Value {
    json!({"prompt":request.prompt,"negativePrompt":request.negative_prompt,"width":request.width,"height":request.height,
        "count":request.count,"seed":request.seed,"steps":request.steps,"guidance":request.guidance,"trueCfg":request.true_cfg,
        "sampler":request.sampler,"scheduler":request.scheduler,"schedulerShift":request.scheduler_shift,"enhancePrompt":request.enhance_prompt,
        "layout":request.flux3_layout.as_ref().map(|layout|layout.to_json()),
        "layoutBindings":request.flux3_layout_bindings.iter().map(|binding|json!({"instanceId":binding.id,
            "originalSize":binding.original_size,"orientation":binding.orientation})).collect::<Vec<_>>(),
        "referenceCarriers":request.conditioning.iter().filter_map(reference_carrier).collect::<Vec<_>>(),
        "conditioningCount":request.conditioning.len()})
}

pub(super) fn anchor_digest(plan: &Plan, request: &GenerationRequest) -> Result<String, String> {
    let value = json!({"backend":plan.lane,"provider":plan.checkpoint.provider,"tier":plan.tier,
        "prompt":request.prompt,"width":request.width,"height":request.height,"count":request.count,
        "seed":request.seed,"steps":request.steps,"guidance":request.guidance,"enhancePrompt":false});
    json_digest(b"sceneworks-flux3-anchor-request-v1\0", &value)
}

/// Reserve inside the native call chain, after load/configuration and before EACH generate.
/// The fixed source helper imports the same atomic ledger as product workers. Missing approval,
/// runtime/helper, busy ledger, reused permit or child failure refuses; never initialize/retry.
pub(super) fn reserve(plan: &Plan, request: &GenerationRequest) -> Result<Value, String> {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let input = reservation_input(
        plan,
        request,
        &protocol::required_env("SCENEWORKS_FLUX3_INVOCATION")?,
    )?;
    spawn_reservation(input, |input| {
        let script =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/flux3-resource-reserve.mjs");
        let mut child = Command::new(protocol::required_env("SCENEWORKS_FLUX3_BUDGET_NODE")?)
            .arg(script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(blocked)?;
        let write = child
            .stdin
            .take()
            .ok_or_else(|| blocked("reservation stdin unavailable"))?
            .write_all(&serde_json::to_vec(input).map_err(blocked)?);
        let result = child.wait_with_output().map_err(blocked)?;
        write.map_err(blocked)?;
        if !result.status.success() {
            return Err(blocked(String::from_utf8_lossy(&result.stderr)));
        }
        serde_json::from_slice(&result.stdout).map_err(blocked)
    })
}

/// The production reservation identity check, independent of the ledger transport: the configured
/// request must equal the approved inputs, the invocation must carry exactly its binding fields,
/// and its backend/case/kind/request digest must equal this native invocation's own. Shared by the
/// real reservation and the zero-budget native-boundary self-test.
pub(super) fn reservation_input(
    plan: &Plan,
    request: &GenerationRequest,
    invocation: &str,
) -> Result<Value, String> {
    // Configuration may set memory policy, but may not silently change approved generation inputs.
    let intended = generation_request(plan);
    if request.count != 1
        || actual_request_observation(request) != actual_request_observation(&intended)
    {
        return Err(blocked(
            "configured generation differs from approved inputs",
        ));
    }
    let mut input: Value = serde_json::from_str(invocation).map_err(blocked)?;
    exact(
        &input,
        &[
            "approvalFile",
            "admissionFile",
            "adapterPid",
            "sourceSha",
            "binarySha256",
            "ledgerRoot",
            "ledgerId",
            "approvedPlanSha256",
            "invocationId",
            "backend",
            "caseId",
            "kind",
            "requestSha256",
        ],
    )?;
    let label = if plan.checkpoint.provider == BASE_PROVIDER {
        "base"
    } else {
        "distilled"
    };
    let (case_id, kind, digest) = if let Some(case) = &plan.resource_case {
        (
            string(&case.value, "caseId")?.to_owned(),
            "resource_supplemental",
            case.digest.clone(),
        )
    } else {
        (
            format!("{}.{label}.resource-native.{}", plan.lane, plan.tier),
            "resource_anchor",
            anchor_digest(plan, request)?,
        )
    };
    if input["backend"] != plan.lane
        || input["caseId"] != case_id
        || input["kind"] != kind
        || input["requestSha256"] != digest
    {
        return Err(blocked(
            "reservation target/request differs from actual native invocation",
        ));
    }
    // Replace no caller-controlled identities: the CLI independently checks the immutable plan.
    input["requestSha256"] = json!(digest);
    input["adapterPid"] = json!(std::process::id());
    Ok(input)
}

/// Consume ONE permit through `transport` and verify the returned ledger event binds exactly the
/// validated invocation (and, for the real ledger, the observed hardware/build/PID).
pub(super) fn spawn_reservation(
    input: Value,
    transport: impl FnOnce(&Value) -> Result<Value, String>,
) -> Result<Value, String> {
    let event = transport(&input)?;
    for field in [
        "ledgerId",
        "approvedPlanSha256",
        "invocationId",
        "backend",
        "caseId",
        "kind",
        "requestSha256",
    ] {
        if event[field] != input[field] {
            return Err(blocked("reservation response identity mismatch"));
        }
    }
    string(&event, "eventSha256")?;
    for field in [
        "adapterPid",
        "sourceSha",
        "binarySha256",
        "approvedPlanSha256",
        "invocationId",
        "caseId",
        "requestSha256",
    ] {
        if event["hardwareObservation"][field] != input[field] {
            return Err(blocked("observed hardware/build/PID binding mismatch"));
        }
    }
    Ok(event)
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "flux3-resource-{}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
                SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn case(provider: &str) -> Value {
        json!({"schemaVersion":1,"caseId":format!("candle.{}.lora.bf16",if provider==BASE_PROVIDER {"base"} else {"distilled"}),
            "backend":"candle","modelId":provider,"tier":"bf16","scenario":"lora",
            "generation":{"prompt":"specific non-default scene","width":1536,"height":768,"seed":9182,"steps":7,"pup":false},
            "references":[],"adapters":[],"components":{}})
    }
    #[test]
    fn supplemental_binds_actual_generation_and_load_adapters_without_loading_any_provider() {
        let fixture = Fixture::new();
        let baseline = bind(
            &case(PROVIDER),
            "candle",
            checkpoint(PROVIDER).unwrap(),
            "bf16",
            &fixture.0,
        )
        .unwrap();
        assert_eq!(
            baseline.digest, "4ec330d847867e0973d3cf263ee8f314cb67f17be62f65daf4da152116e747b9",
            "shared Node/Rust canonical domain-separated digest"
        );
        std::fs::write(
            fixture.0.join("adapter.safetensors"),
            b"tiny fixture NOT model evidence",
        )
        .unwrap();
        let file =
            core::flux3::conversion::hash_source_file(&fixture.0.join("adapter.safetensors"))
                .unwrap();
        for provider in [PROVIDER, BASE_PROVIDER] {
            let mut value = case(provider);
            if provider == BASE_PROVIDER {
                value["generation"]["guidance"] = json!(3.5);
            }
            value["adapters"] =
                json!([{"path":"adapter.safetensors","sha256":file.sha256,"scale":0.625}]);
            let checkpoint = checkpoint(provider).unwrap();
            let bound = bind(&value, "candle", checkpoint, "bf16", &fixture.0).unwrap();
            assert_eq!(bound.request.prompt, "specific non-default scene");
            assert_eq!((bound.request.width, bound.request.height), (1536, 768));
            assert_eq!(
                (bound.request.seed, bound.request.steps),
                (Some(9182), Some(7))
            );
            assert_eq!(bound.request.guidance, checkpoint.guidance);
            assert_eq!(bound.adapters.len(), 1);
            assert_eq!(bound.adapters[0].scale, 0.625);
            assert_eq!(bound.inputs[0]["sha256"], file.sha256);
            assert_eq!(
                bound.digest,
                bind(&value, "candle", checkpoint, "bf16", &fixture.0)
                    .unwrap()
                    .digest
            );
            let mut changed = value.clone();
            changed["generation"]["seed"] = json!(9183);
            assert_ne!(
                bound.digest,
                bind(&changed, "candle", checkpoint, "bf16", &fixture.0)
                    .unwrap()
                    .digest
            );
            assert!(bind(&value, "mlx", checkpoint, "bf16", &fixture.0).is_err());
            changed = value.clone();
            changed["modelId"] = json!("another_checkpoint");
            assert!(bind(&changed, "candle", checkpoint, "bf16", &fixture.0).is_err());
            changed = value;
            changed["adapters"][0]["sha256"] = json!("0".repeat(64));
            assert!(bind(&changed, "candle", checkpoint, "bf16", &fixture.0).is_err());
        }
    }
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn supplemental_decodes_ten_real_local_reference_files_preserving_duplicate_asset_instances() {
        let fixture = Fixture::new();
        image::save_buffer(
            fixture.0.join("reference.png"),
            &[255, 0, 0, 0, 255, 0].repeat(32 * 16 / 2),
            32,
            16,
            image::ColorType::Rgb8,
        )
        .unwrap();
        let file =
            core::flux3::conversion::hash_source_file(&fixture.0.join("reference.png")).unwrap();
        let mut value = case(PROVIDER);
        value["references"]=json!((0..10).map(|index| json!({"instanceId":format!("instance-{index}"),"assetId":"same-asset","index":index,"path":"reference.png","sha256":file.sha256})).collect::<Vec<_>>());
        let bound = bind(
            &value,
            "candle",
            checkpoint(PROVIDER).unwrap(),
            "bf16",
            &fixture.0,
        )
        .unwrap();
        assert_eq!(bound.request.image_reference_count(), 10);
        assert!(
            bound.request.flux3_layout_bindings.is_empty(),
            "ordinary references are ordered attachments, not a layout document"
        );
        assert_eq!(bound.inputs[9]["instanceId"], "instance-9");
        let mut boxed = value.clone();
        boxed["generation"]["layout"] = json!({"version":1,"boxes":[]});
        let boxed = bind(
            &boxed,
            "candle",
            checkpoint(PROVIDER).unwrap(),
            "bf16",
            &fixture.0,
        )
        .unwrap();
        assert_eq!(boxed.request.flux3_layout_bindings.len(), 10);
        assert_eq!(boxed.request.flux3_layout_bindings[9].id, "instance-9");
        assert!(bound.reference_resources.is_some());
        assert_eq!(bound.inputs[9]["index"], 9);
        assert_eq!(bound.inputs[9]["assetId"], "same-asset");
        assert_eq!(bound.inputs[9]["decodedCarrier"]["width"], 32);
        let original = actual_request_observation(&bound.request);
        let mut changed_request = bound.request.clone();
        if let core::Conditioning::Reference { image, .. } = &mut changed_request.conditioning[0] {
            image.pixels[0] ^= 1;
        }
        assert_ne!(original, actual_request_observation(&changed_request));
        if let core::Conditioning::Reference { image, .. } = &mut changed_request.conditioning[0] {
            image.pixels[0] ^= 1;
        }
        assert_eq!(original, actual_request_observation(&changed_request));
        let mut invalid = value.clone();
        invalid["references"][1]["instanceId"] = json!("instance-0");
        assert!(bind(
            &invalid,
            "candle",
            checkpoint(PROVIDER).unwrap(),
            "bf16",
            &fixture.0
        )
        .is_err());
        invalid = value.clone();
        invalid["references"][1]["index"] = json!(0);
        assert!(bind(
            &invalid,
            "candle",
            checkpoint(PROVIDER).unwrap(),
            "bf16",
            &fixture.0
        )
        .is_err());
        std::fs::write(fixture.0.join("reference.png"), b"corrupt reference").unwrap();
        assert!(bind(
            &value,
            "candle",
            checkpoint(PROVIDER).unwrap(),
            "bf16",
            &fixture.0
        )
        .is_err());
    }
}
