//! CANDLE sdxl capture owner.
use super::*;

/// The revision `candle-gen-sdxl` pins for this member, read out of the ENGINE's own
/// `SDXL_ROUTES` at the compiled pin.
///
/// Fail-closed in both directions: a member the engine declares no route for, and a route whose
/// repository disagrees with the one this adapter stages, are both errors rather than a silently
/// skipped comparison. `SdxlRoute` is not nameable from outside the crate (only the const is
/// re-exported), which is why this returns the field rather than the row.
pub(super) fn sdxl_candle_route_revision(arm: &SdxlCandleArm) -> Result<&'static str, String> {
    let route = candle_gen_sdxl::SDXL_ROUTES
        .iter()
        .find(|route| route.id == arm.model_id)
        .ok_or_else(|| {
            format!(
                "candle-gen-sdxl's SDXL_ROUTES declares no route {:?} at {}, so this adapter has \
                 no engine revision to bind {} against",
                arm.model_id,
                protocol::INFERENCE_PIN,
                arm.expected_repository
            )
        })?;
    if route.repository != arm.expected_repository {
        return Err(format!(
            "candle-gen-sdxl routes {:?} to {} at {}, but this adapter stages {} through {}; the \
             two disagree about which repository the route loads",
            arm.model_id,
            route.repository,
            protocol::INFERENCE_PIN,
            arm.expected_repository,
            arm.repository_env
        ));
    }
    Ok(route.revision)
}

/// Which SDXL family member the plan asks for, from `planned.target.modelId`.
///
/// Refuses by name: the engine id is not an artifact identity — all five members load through the
/// same `sdxl` provider — so a model id no member serves must not be measured as base SDXL.
pub(super) fn sdxl_candle_arm(request: &Value) -> Result<SdxlCandleArm, String> {
    let model_id = protocol::planned_model_id(request)?;
    let arm = SDXL_CANDLE_FAMILY
        .into_iter()
        .find(|arm| arm.model_id == model_id)
        .ok_or_else(|| {
            format!(
                "the Candle SDXL arm does not implement modelId {model_id:?} on provider \
                 {SDXL_ID:?} (family: {})",
                SDXL_CANDLE_FAMILY
                    .iter()
                    .map(|arm| arm.model_id)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
    Ok(arm)
}

/// The exact calibration fingerprint `candle-gen-sdxl` mints for one route
/// (`memory_strategy.rs` `route_fingerprint` / `route_identity_tokens` at the pin): the route id in
/// kebab tokens, with a trailing `vN` token rewritten to `revN` so the shared
/// `validate_calibration_fingerprint` still sees exactly one version token.
///
/// This is derived rather than transcribed so a plan row can be checked against it without a
/// weights root, and `run_five_rung_reference_loaded` still compares the LOADED contract's own
/// string — this function is never the authority, only the plan-time expectation.
#[cfg(test)]
pub(super) fn sdxl_candle_route_fingerprint(model_id: &str) -> String {
    let tokens = model_id
        .replace('_', "-")
        .split('-')
        .map(|token| match token.strip_prefix('v') {
            Some(digits) if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) => {
                format!("rev{digits}")
            }
            _ => token.to_owned(),
        })
        .collect::<Vec<_>>()
        .join("-");
    format!("sdxl-candle-{tokens}-staged-decode-attention-v3")
}

/// The SDXL-specific half of the shared five-rung spec: the resolved catalog route the engine
/// requires, and the three caller-staged components it validates at exact upstream revisions.
///
/// Also the seam that refuses a member whose engine-side route revision has drifted from the
/// revision this repository ships. `candle-gen-sdxl`'s `path_has_snapshot` matches the staged root
/// against `SdxlRoute::revision` verbatim, so for such a member NO root that the manifest can
/// resolve will ever seal — the load fails inside the engine with a message about the engine's own
/// pinned revision. Refusing here says which two revisions disagree instead.
pub(super) fn sdxl_candle_spec(request: &Value, spec: LoadSpec) -> Result<LoadSpec, String> {
    let arm = sdxl_candle_arm(request)?;
    let revision = protocol::required_env(arm.revision_env)?;
    let route_revision = sdxl_candle_route_revision(&arm)?;
    if revision != route_revision {
        return Err(format!(
            "candle-gen-sdxl pins route {:?} at {} ({}), but {} names {revision}; the engine's \
             `path_has_snapshot` matches that literal, so no staged root can seal this route at \
             {}. This cell is not capturable until the two revisions agree.",
            arm.model_id,
            route_revision,
            arm.expected_repository,
            arm.revision_env,
            protocol::INFERENCE_PIN
        ));
    }
    let mut spec = spec.with_resolved_route(arm.model_id);
    for (component, env) in SDXL_COMPONENTS {
        let root = std::fs::canonicalize(PathBuf::from(protocol::required_env(env)?))
            .map_err(|error| format!("canonicalize {env}: {error}"))?;
        spec = spec.with_component(component, WeightsSource::Dir(root));
    }
    Ok(spec)
}

pub(super) fn inspect_sdxl_output(request: &Value) -> Result<Value, String> {
    let input = request
        .get("inputPath")
        .and_then(Value::as_str)
        .ok_or("PNG inputPath is required")?;
    let rgb = image::open(input)
        .map_err(|error| format!("decode served SDXL PNG: {error}"))?
        .into_rgb8();
    let output = Image {
        width: rgb.width(),
        height: rgb.height(),
        pixels: rgb.into_raw(),
    };
    Ok(json!({ "output": retain_sdxl_endpoint_output(request, &output)? }))
}

pub(super) fn probe_sdxl_endpoint() -> Result<Value, String> {
    let expected = sdxl_endpoint_expected_uuid()?;
    let uuid = NvidiaSmi::resolve()?.query_bounded("uuid", Duration::from_secs(5))?;
    if uuid != expected {
        return Err(format!(
            "SDXL endpoint probe physical UUID mismatch: expected {expected}, observed {uuid}"
        ));
    }
    let mut response = probe()?;
    response["hardware"]["deviceId"] = json!(uuid);
    Ok(response)
}

pub(super) fn valid_sdxl_gpu_uuid(value: &str) -> bool {
    let Some(uuid) = value.strip_prefix("GPU-") else {
        return false;
    };
    uuid.len() == 36
        && uuid.bytes().enumerate().all(|(index, byte)| {
            if [8, 13, 18, 23].contains(&index) {
                byte == b'-'
            } else {
                byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
            }
        })
}

pub(super) fn validate_sdxl_endpoint_binding(
    visible: &str,
    physical: &str,
    uuid: &str,
) -> Result<(), String> {
    if !matches!(visible, "0" | "1") || visible != physical || !valid_sdxl_gpu_uuid(uuid) {
        return Err("SDXL endpoint requires identical physical index 0/1 in CUDA_VISIBLE_DEVICES and SCENEWORKS_GPU_ID, plus a canonical SCENEWORKS_SDXL_EXPECTED_GPU_UUID".into());
    }
    Ok(())
}

pub(super) fn sdxl_endpoint_expected_uuid() -> Result<String, String> {
    let visible = protocol::required_env("CUDA_VISIBLE_DEVICES")?;
    let physical = protocol::required_env("SCENEWORKS_GPU_ID")?;
    let expected = protocol::required_env("SCENEWORKS_SDXL_EXPECTED_GPU_UUID")?;
    validate_sdxl_endpoint_binding(&visible, &physical, &expected)?;
    Ok(expected)
}

pub(super) fn sdxl_endpoint_context_uuid() -> Result<String, String> {
    // SDXL's loader uses this same production device constructor (logical CUDA 0).
    let device = candle_gen::default_device()
        .map_err(|error| format!("SDXL endpoint CUDA context: {error}"))?;
    let cuda = device.as_cuda_device().map_err(|error| error.to_string())?;
    let uuid = cuda
        .cuda_stream()
        .context()
        .uuid()
        .map_err(|error| error.to_string())?;
    let bytes = uuid.bytes.map(|byte| byte as u8);
    let hex = sceneworks_core::hashing::hex(&bytes);
    Ok(format!(
        "GPU-{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..]
    ))
}

pub(super) fn wait_for_sdxl_idle(
    mut query: impl FnMut(Duration) -> Result<String, String>,
    elapsed: impl Fn() -> Duration,
    mut sleep: impl FnMut(Duration),
) -> Result<Vec<u64>, String> {
    let timeout = Duration::from_secs(30);
    let mut samples = Vec::new();
    let mut used = Vec::new();
    while elapsed() < timeout {
        let fields = query(
            timeout
                .saturating_sub(elapsed())
                .min(Duration::from_secs(5)),
        )
        .map_err(|error| {
            format!("SDXL endpoint activity telemetry: {error}; samples={samples:?}")
        })?;
        samples.push(format!("{}ms:{fields}", elapsed().as_millis()));
        eprintln!(
            "SDXL endpoint selected-device idle sample: {}",
            samples.last().unwrap()
        );
        let values = fields
            .split(',')
            .map(|value| value.trim().parse::<u64>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| {
                format!("SDXL endpoint invalid activity telemetry: {error}; samples={samples:?}")
            })?;
        if values.len() != 4
            || values[2] > 100
            || values[3] > 100
            || values[0].checked_mul(MIB).is_none()
            || values[1].checked_mul(MIB).is_none()
        {
            return Err(format!(
                "SDXL endpoint invalid activity telemetry; samples={samples:?}"
            ));
        }
        if values[1] < 16 * 1024 || values[2] != 0 || values[3] != 0 {
            used.clear();
        } else {
            used.push(values[0]);
            if used.iter().max().unwrap() - used.iter().min().unwrap() > 64 {
                used.clear();
                used.push(values[0]);
            }
            if used.len() == 5 && elapsed() <= timeout {
                return Ok(used);
            }
        }
        sleep(Duration::from_millis(200).min(timeout.saturating_sub(elapsed())));
    }
    Err(format!("SDXL endpoint did not settle within 30s to five consecutive zero-utilization samples with at least 16 GiB free and <=64 MiB drift; samples={samples:?}"))
}

pub(super) fn retain_sdxl_endpoint_output(
    request: &Value,
    output: &Image,
) -> Result<Value, String> {
    let path = request
        .get("outputPath")
        .and_then(Value::as_str)
        .ok_or("SDXL endpoint requires outputPath")?;
    let (width, height) = protocol::target_geometry(request)?;
    if output.width != width
        || output.height != height
        || output.pixels.len() != output.width as usize * output.height as usize * 3
        || output.pixels.len() < 3
    {
        return Err("SDXL endpoint output is not the requested RGB8 geometry".into());
    }
    if !output
        .pixels
        .as_chunks::<3>()
        .0
        .iter()
        .any(|pixel| pixel.as_slice() != &output.pixels[..3])
    {
        return Err("SDXL endpoint output is spatially flat".into());
    }
    image::save_buffer_with_format(
        path,
        &output.pixels,
        output.width,
        output.height,
        image::ColorType::Rgb8,
        image::ImageFormat::Png,
    )
    .map_err(|error| format!("retain SDXL endpoint PNG: {error}"))?;
    let rgb_path = PathBuf::from(path).with_extension("rgb");
    std::fs::write(&rgb_path, &output.pixels)
        .map_err(|error| format!("retain SDXL endpoint RGB: {error}"))?;
    let minimum = output.pixels.iter().min().copied().unwrap_or(0);
    let maximum = output.pixels.iter().max().copied().unwrap_or(0);
    Ok(json!({
        "path": path, "rgbPath": rgb_path, "width": output.width, "height": output.height,
        "channels": 3, "rgbBytes": output.pixels.len(), "spatiallyNonFlat": true,
        "minimumChannel": minimum, "maximumChannel": maximum,
        "rgbSha256": format!("{:x}", Sha256::digest(&output.pixels)),
    }))
}

pub(super) fn run_sdxl_endpoint(request: &Value) -> Result<Value, String> {
    let endpoint = request
        .get("sdxlEndpoint")
        .and_then(Value::as_str)
        .ok_or("SDXL endpoint name is required")?;
    let expected_rung = match endpoint {
        "resident" => "resident",
        "sequential" => "bounded_attention",
        _ => return Err("SDXL endpoint must be resident or sequential".into()),
    };
    if planned_provider(request)? != SDXL_ID
        || planned_tier(request)? != "q4"
        || protocol::planned_rung(request)? != expected_rung
        || protocol::planned_optional_str(request, "/target/modelId")? != Some("sdxl")
    {
        return Err("SDXL endpoint only accepts base SDXL q4 and its exact endpoint rung".into());
    }
    let expected = sdxl_endpoint_expected_uuid()?;
    let (width, height) = protocol::target_geometry(request)?;
    let parameters = protocol::strategy_parameters(request)?;
    let expected_parameters = if endpoint == "resident" {
        json!({})
    } else {
        json!({
            "stageResidency": true, "decodeTileEdge": 256, "decodeOverlap": 64,
            "attentionChunkSize": 16_777_216,
        })
    };
    if (width, height) != (1024, 1024) || json!(parameters) != expected_parameters {
        return Err("SDXL endpoint requires 1024-square geometry and the exact resident/fully-optimized composition".into());
    }
    let smi = NvidiaSmi::resolve()?;
    let uuid = smi.query_bounded("uuid", Duration::from_secs(5))?;
    if uuid != expected {
        return Err(format!(
            "SDXL endpoint physical UUID mismatch: expected {expected}, observed {uuid}"
        ));
    }
    let used_bytes = || -> Result<u64, String> {
        let mib = smi
            .query_bounded("memory.used", Duration::from_secs(5))?
            .parse::<u64>()
            .map_err(|error| format!("SDXL endpoint memory.used: {error}"))?;
        mib.checked_mul(MIB)
            .ok_or_else(|| "SDXL endpoint memory.used overflow".into())
    };
    let before = used_bytes()?;
    let mut fragment = run_five_rung_reference(request)?;
    // run_five_rung_reference drops the request scope and generator before returning.
    let after = used_bytes()?;
    fragment["endpointMeasurement"]["providerDrop"] = json!({
        "beforeBytes": before, "afterBytes": after, "generatorDropped": true,
    });
    fragment["endpointMeasurement"]["deviceUuid"] = json!(uuid);
    let context_uuid = sdxl_endpoint_context_uuid()?;
    if context_uuid != expected {
        return Err(format!("SDXL endpoint logical CUDA0 context UUID mismatch: expected {expected}, observed {context_uuid}"));
    }
    fragment["endpointMeasurement"]["cudaContextUuid"] = json!(context_uuid);
    fragment["endpointMeasurement"]["logicalCudaDevice"] = json!(0);
    fragment["endpointMeasurement"]["endpoint"] = json!(endpoint);
    Ok(fragment)
}
