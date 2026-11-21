use gen_core::{LoadSpec, Quant, WeightsSource};
use serde_json::Value;

use crate::{WorkerError, WorkerResult};

/// Select the existing ConvRot assembly from the checkpoint's declarations, never its filename or
/// a request flag. Both legacy imports and checkpoint plans pass here before encoder preparation,
/// admission, and cache lookup, so every consumer sees the same source set and retained file pin.
/// The engine's ConvRot loader still owns descriptor/scale validation and the CUDA capability floor.
pub(super) fn route_imported_krea_checkpoint(
    spec: &mut LoadSpec,
    engine_id: &str,
    pins: &[gen_core::PinnedWeightsFile],
) -> WorkerResult<()> {
    if !matches!(
        engine_id,
        "krea_2_turbo"
            | "krea_2_raw"
            | "krea_2_edit"
            | "krea_2_turbo_edit"
            | "krea_2_turbo_control"
    ) {
        return Ok(());
    }
    let WeightsSource::File(dit) = &spec.weights else {
        return Ok(());
    };
    let Some(WeightsSource::Dir(base)) = spec.components.get(gen_core::BASE_SNAPSHOT_COMPONENT)
    else {
        return Ok(());
    };
    let pin = pins
        .iter()
        .find(|pin| pin.loader_path() == dit)
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "Imported Krea checkpoint requires its prepared file pin".into(),
            )
        })?;
    let convrot = pin
        .read_unchanged(|path| -> gen_core::Result<bool> {
            let declares_rotation = |value: &Value| {
                value.get("convrot").is_some() || value.get("convrot_groupsize").is_some()
            };
            if let Some(metadata) = gen_core::safetensors_path_quantization_metadata(path)? {
                let metadata: Value = serde_json::from_str(&metadata).map_err(|error| {
                    gen_core::Error::Msg(format!(
                        "Invalid checkpoint quantization metadata: {error}"
                    ))
                })?;
                if metadata
                    .get("layers")
                    .and_then(Value::as_object)
                    .is_some_and(|layers| layers.values().any(declares_rotation))
                {
                    return Ok(true);
                }
            }
            let descriptors = gen_core::read_safetensors_tensor_payloads(
                path,
                |header| header.name.ends_with(".comfy_quant"),
                65_536,
            )?;
            for (key, payload) in descriptors {
                let descriptor: Value = serde_json::from_slice(&payload).map_err(|error| {
                    gen_core::Error::Msg(format!("Invalid checkpoint descriptor {key:?}: {error}"))
                })?;
                if declares_rotation(&descriptor) {
                    return Ok(true);
                }
            }
            Ok(false)
        })
        .map_err(|error| {
            crate::generation_error("Krea checkpoint format detection failed", error)
        })?;
    if !convrot {
        return Ok(());
    }
    if cfg!(target_os = "macos") {
        return Err(WorkerError::InvalidPayload(
            "Imported Krea INT8-ConvRot checkpoints require the Candle/CUDA backend".into(),
        ));
    }
    if !matches!(engine_id, "krea_2_turbo" | "krea_2_turbo_control") {
        return Err(WorkerError::InvalidPayload(
            "Imported Krea INT8-ConvRot checkpoints support Turbo generation and pose control; edit and multi-phase generation are not supported".into(),
        ));
    }
    if spec.quantize.is_some_and(|quant| quant != Quant::Q8) {
        return Err(WorkerError::InvalidPayload(
            "Imported Krea checkpoint is stored as INT8-ConvRot; remove the load-time quantization override or select Q8".into(),
        ));
    }
    if spec
        .components
        .contains_key(gen_core::KREA_CONVROT_DIT_COMPONENT)
    {
        return Err(WorkerError::InvalidPayload(
            "Select only one imported Krea DiT checkpoint".into(),
        ));
    }
    let dit = dit.clone();
    spec.weights = WeightsSource::Dir(base.clone());
    spec.components.remove(gen_core::BASE_SNAPSHOT_COMPONENT);
    spec.components.insert(
        gen_core::KREA_CONVROT_DIT_COMPONENT.into(),
        WeightsSource::File(dit),
    );
    spec.quantize = Some(Quant::Q8);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn checkpoint(
        descriptor: Option<Value>,
        metadata: Option<Value>,
    ) -> (tempfile::TempDir, std::path::PathBuf) {
        let temp = tempfile::tempdir().unwrap();
        // A misleading name must not turn plain INT8 or dense weights into rotated weights.
        let path = temp.path().join("community_INT8-ConvRot.safetensors");
        let stored_int8 = descriptor.is_some() || metadata.is_some();
        let mut data = vec![0_u8; 16 * 256 * if stored_int8 { 1 } else { 2 }];
        let mut header = json!({
            "blocks.0.attn.gate.weight": {"dtype":if stored_int8 { "I8" } else { "F16" }, "shape":[16,256], "data_offsets":[0,data.len()]}
        });
        if stored_int8 {
            let start = data.len();
            for _ in 0..16 {
                data.extend_from_slice(&1_f32.to_le_bytes());
            }
            header["blocks.0.attn.gate.weight_scale"] = json!({
                "dtype":"F32", "shape":[16,1], "data_offsets":[start,data.len()]
            });
        }
        if let Some(descriptor) = descriptor {
            let bytes = serde_json::to_vec(&descriptor).unwrap();
            let start = data.len();
            data.extend_from_slice(&bytes);
            header["blocks.0.attn.gate.comfy_quant"] = json!({
                "dtype":"U8", "shape":[bytes.len()], "data_offsets":[start,data.len()]
            });
        }
        if let Some(metadata) = metadata {
            header["__metadata__"] = json!({"_quantization_metadata":metadata.to_string()});
        }
        let encoded = serde_json::to_vec(&header).unwrap();
        let mut file = (encoded.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(&encoded);
        file.extend_from_slice(&data);
        std::fs::write(&path, file).unwrap();
        (temp, path)
    }

    fn spec(path: &std::path::Path) -> LoadSpec {
        LoadSpec::new(WeightsSource::File(path.to_path_buf())).with_component(
            gen_core::BASE_SNAPSHOT_COMPONENT,
            WeightsSource::Dir(path.parent().unwrap().join("bf16")),
        )
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn imported_convrot_uses_the_existing_loader_and_retains_source_pins() {
        let (_temp, path) = checkpoint(
            Some(json!({
                "format":"int8_tensorwise", "per_row":true, "convrot":true, "convrot_groupsize":256
            })),
            None,
        );
        let pin = gen_core::PinnedWeightsFile::pin(&path).unwrap();
        let adapter_path = path.parent().unwrap().join("adapter.safetensors");
        std::fs::write(&adapter_path, b"adapter fixture").unwrap();
        let adapter_pin = gen_core::PinnedWeightsFile::pin(&adapter_path).unwrap();
        let adapter =
            gen_core::AdapterSpec::new(adapter_path.clone(), 0.6, gen_core::AdapterKind::Lora);
        for engine in ["krea_2_turbo", "krea_2_turbo_control"] {
            let mut routed = spec(&path).with_adapters(vec![adapter.clone()]);
            routed.offload_policy = gen_core::OffloadPolicy::Sequential;
            route_imported_krea_checkpoint(&mut routed, engine, std::slice::from_ref(&pin))
                .unwrap();
            assert_eq!(
                routed.weights,
                WeightsSource::Dir(path.parent().unwrap().join("bf16"))
            );
            assert_eq!(
                routed.components.get(gen_core::KREA_CONVROT_DIT_COMPONENT),
                Some(&WeightsSource::File(path.clone()))
            );
            assert!(!routed
                .components
                .contains_key(gen_core::BASE_SNAPSHOT_COMPONENT));
            assert_eq!(routed.quantize, Some(Quant::Q8));
            assert_eq!(routed.adapters[0].path, adapter_path);
            assert_eq!(routed.adapters[0].scale, 0.6);
            assert_eq!(routed.offload_policy, gen_core::OffloadPolicy::Sequential);
            routed
                .prepare_with_file_pins([pin.clone(), adapter_pin.clone()])
                .unwrap();
            assert_eq!(routed.prepared_file_pin_for(&path).unwrap(), Some(&pin));
            routed.validate_prepared_file_pins().unwrap();
        }
        std::fs::write(&path, b"replaced after routing").unwrap();
        assert!(route_imported_krea_checkpoint(&mut spec(&path), "krea_2_turbo", &[pin]).is_err());
    }

    #[test]
    fn plain_checkpoint_and_other_families_keep_their_original_assembly() {
        for descriptor in [
            None,
            Some(json!({"format":"int8_tensorwise", "per_row":true})),
        ] {
            let (_temp, path) = checkpoint(descriptor, None);
            let pin = gen_core::PinnedWeightsFile::pin(&path).unwrap();
            let mut routed = spec(&path);
            route_imported_krea_checkpoint(&mut routed, "krea_2_turbo", &[pin]).unwrap();
            assert_eq!(routed.weights, WeightsSource::File(path));
            assert!(routed
                .components
                .contains_key(gen_core::BASE_SNAPSHOT_COMPONENT));
            assert_eq!(routed.quantize, None);
        }
        let (_temp, path) = checkpoint(Some(json!({"convrot":true})), None);
        let mut routed = spec(&path);
        route_imported_krea_checkpoint(&mut routed, "flux2_klein_9b", &[]).unwrap();
        assert_eq!(routed.weights, WeightsSource::File(path));
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn file_metadata_selects_convrot_and_incompatible_overrides_fail_clearly() {
        let (_temp, path) = checkpoint(
            None,
            Some(json!({
                "format_version":"1.0", "layers":{"blocks.0.attn.gate":{
                    "format":"int8_tensorwise", "convrot":true, "convrot_groupsize":256
                }}
            })),
        );
        let pin = gen_core::PinnedWeightsFile::pin(&path).unwrap();
        let mut routed = spec(&path);
        route_imported_krea_checkpoint(&mut routed, "krea_2_turbo", std::slice::from_ref(&pin))
            .unwrap();
        assert!(routed
            .components
            .contains_key(gen_core::KREA_CONVROT_DIT_COMPONENT));
        let error = route_imported_krea_checkpoint(
            &mut spec(&path).with_quant(Quant::Q4),
            "krea_2_turbo",
            std::slice::from_ref(&pin),
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("remove the load-time quantization override"));
        for engine in ["krea_2_raw", "krea_2_turbo_edit", "krea_2_edit"] {
            let error = route_imported_krea_checkpoint(
                &mut spec(&path),
                engine,
                std::slice::from_ref(&pin),
            )
            .unwrap_err();
            assert!(error.to_string().contains("edit and multi-phase"));
        }
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    #[ignore = "requires SCENEWORKS_TEST_KREA_CONVROT_PATH; verifies routing without GPU execution"]
    fn real_imported_convrot_checkpoint_selects_the_existing_loader() {
        let path = std::path::PathBuf::from(
            std::env::var_os("SCENEWORKS_TEST_KREA_CONVROT_PATH").expect("checkpoint path"),
        );
        let pin = gen_core::PinnedWeightsFile::pin(&path).unwrap();
        let mut routed = spec(&path);
        route_imported_krea_checkpoint(&mut routed, "krea_2_turbo", std::slice::from_ref(&pin))
            .unwrap();
        assert!(matches!(routed.weights, WeightsSource::Dir(_)));
        assert_eq!(
            routed.components.get(gen_core::KREA_CONVROT_DIT_COMPONENT),
            Some(&WeightsSource::File(path))
        );
        routed.prepare_with_file_pins([pin]).unwrap();
        routed.validate_prepared_file_pins().unwrap();
    }
}
