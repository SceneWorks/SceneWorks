//! Surface the **base** weights sitting in an operator's ComfyUI `models/` tree as
//! assembled virtual models, read in place (epic 10451 Phase 2, sc-10667).
//!
//! The scan itself lives in `sceneworks_core::external_base_models` so the API
//! catalog and the worker's portable source inventory assemble the same
//! `external_base_…` identities from the same files (SW-7030); this module only
//! re-exports it under the API's historical path.

#[cfg(test)]
use std::path::{Path, PathBuf};

#[cfg(test)]
use serde_json::{json, Value};

#[cfg(test)]
pub(crate) use sceneworks_core::external_base_models::EXTERNAL_SCOPE;
pub(crate) use sceneworks_core::external_base_models::{
    scan_external_base_models, ExternalBaseModelCache, EXTERNAL_ID_PREFIX,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn flux2_retry_and_duplicate_keep_verified_precision_with_frozen_source() {
        use crate::portable_admission::{capture_local_model_entry, portable_authored_options};
        use crate::tests::support::{create_project, request, test_settings};
        use axum::http::StatusCode;
        use sceneworks_core::contracts::{JobType, WorkerCapability};
        use sceneworks_core::jobs_store::{CreateJob, RegisterWorker};
        use sceneworks_core::model_artifacts::portable_binding::{
            canonical_authored_options_sha256, registered_digest_member_claim_id,
            PortableJobBinding, PortableModelBinding, PortableRouteBinding, PortableSourceClaim,
            PortableSourceProvenance,
        };
        use sceneworks_core::source_precision::SourcePrecision;

        let temp = tempfile::tempdir().unwrap();
        let root = models_root(temp.path());
        let path = root.join("diffusion_models/flux2_dev_fp8mixed.safetensors");
        write_safetensors(&path, &flux2_dev_dit_inline_scale_keys());
        let mut settings = test_settings(&temp);
        settings.external_model_roots = vec![root];
        settings.candle_required = true;
        let (app, state) = crate::create_app_with_state(settings).unwrap();
        let (_, project) = create_project(app.clone(), &"External FLUX.2 replay").await;
        let id = "external_base_flux2-dev-fp8mixed";
        let current = crate::models::resolve_model_manifest_entry(&state, id)
            .await
            .unwrap();
        assert_eq!(current["importQuantFormat"], "fp8_e4m3_scalar");

        let mut saved = current.as_object().unwrap().clone();
        saved.remove("importQuantFormat");
        saved.remove("sourcePrecision");
        saved.insert("quant".into(), json!("fp8_inline_scale"));
        saved.get_mut("components").unwrap()[0]["quant"] = json!("fp8_inline_scale");
        saved.insert("name".into(), json!("Frozen pre-upgrade carrier"));
        let project_id = project["id"].as_str().unwrap();
        let payload = json!({
            "projectId": project_id, "mode": "text_to_image", "model": id, "prompt": "a fox",
            "variant": "q4", "advanced": {"quant": "q4", "steps": 12},
            "modelManifestEntry": saved,
        });
        let authored = portable_authored_options(payload.as_object().unwrap());
        let model = PortableModelBinding::new(
            "modelManifestEntry",
            id,
            "candle",
            "q4",
            None,
            canonical_authored_options_sha256(&authored).unwrap(),
            capture_local_model_entry(&state, &current, "modelManifestEntry").unwrap(),
        )
        .unwrap();
        let claims: Vec<_> = model
            .members
            .iter()
            .map(|member| {
                PortableSourceClaim::new(
                    registered_digest_member_claim_id(&model, member, &payload).unwrap(),
                    PortableSourceProvenance::OperatorOverride,
                    member.files.clone(),
                )
                .unwrap()
            })
            .collect();
        let route = PortableRouteBinding::new("candle", "windows", vec![model]).unwrap();
        let frozen = PortableJobBinding::new(authored, [(route.key(), route)].into()).unwrap();
        state
            .jobs_store
            .register_worker_with_portable_sources(
                RegisterWorker {
                    worker_id: "flux2-fixture".into(),
                    gpu_id: "cuda:0".into(),
                    gpu_name: None,
                    capabilities: vec![
                        WorkerCapability::Gpu,
                        WorkerCapability::ImageGenerate,
                        WorkerCapability::Unknown("candle".into()),
                    ],
                    loaded_models: Vec::new(),
                    utilization: None,
                },
                Some("windows"),
                Some(&claims),
            )
            .unwrap();
        let job = state
            .jobs_store
            .create_job_with_portable_binding(
                CreateJob {
                    job_type: JobType::ImageGenerate,
                    project_id: Some(project_id.to_owned()),
                    project_name: None,
                    payload: payload.as_object().unwrap().clone(),
                    requested_gpu: "auto".into(),
                    source_job_id: None,
                    duplicate_of_job_id: None,
                    attempts: 0,
                    initial_status: None,
                },
                None,
                None,
                Some(&serde_json::to_value(&frozen).unwrap()),
            )
            .unwrap();
        let expected_binding =
            serde_json::to_value(frozen.with_fixed_source_precision(id).unwrap()).unwrap();
        let mut expected_entry = saved;
        expected_entry.insert(
            "importQuantFormat".into(),
            current["importQuantFormat"].clone(),
        );
        SourcePrecision::project(&mut expected_entry);

        for operation in ["retry", "duplicate"] {
            let (status, replay) = request(
                app.clone(),
                "POST",
                &format!("/api/v1/jobs/{}/{operation}", job.id),
                json!({"payloadChanges": {}}),
            )
            .await;
            assert_eq!(status, StatusCode::CREATED, "{operation}: {replay}");
            let replay_id = replay["id"].as_str().unwrap();
            let replay = state.jobs_store.get_job(replay_id).unwrap();
            assert_eq!(replay.payload["modelManifestEntry"], json!(expected_entry));
            assert!(replay.payload["variant"].is_null());
            assert_eq!(replay.payload["advanced"], json!({"steps": 12}));
            assert_eq!(
                state.jobs_store.portable_model_binding(replay_id).unwrap(),
                Some(expected_binding.clone())
            );
            let (status, override_error) = request(
                app.clone(),
                "POST",
                &format!("/api/v1/jobs/{}/{operation}", job.id),
                json!({"payloadChanges": {"advanced": {"quant": "q4"}}}),
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{override_error}");
            assert_eq!(override_error["context"]["field"], "advanced.quant");
        }

        // The worker still advertises the admitted bytes. The local replacement has the same
        // FLUX.2 format and catalog id, but its precision facts cannot be grafted onto that ledger.
        let mut changed = std::fs::read(&path).unwrap();
        *changed.last_mut().unwrap() ^= 1;
        std::fs::write(&path, changed).unwrap();
        let (status, error) = request(
            app,
            "POST",
            &format!("/api/v1/jobs/{}/retry", job.id),
            json!({}),
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT, "{error}");
        assert!(
            error["detail"]
                .as_str()
                .unwrap()
                .contains("different bytes"),
            "{error}"
        );
    }

    /// Write a safetensors file whose header declares `(name, dtype)` tensors. Only
    /// the header is read, but the declared data must be present or the header is
    /// rejected as truncated (sc-6072).
    fn write_safetensors(path: &Path, entries: &[(&str, &str)]) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("parent dir");
        }
        std::fs::write(
            path,
            crate::tests::support::test_safetensors_bytes_with_typed_keys(entries),
        )
        .expect("write safetensors");
    }

    fn z_image_keys() -> Vec<(&'static str, &'static str)> {
        vec![
            ("cap_embedder.0.weight", "BF16"),
            ("noise_refiner.0.attention.qkv.weight", "BF16"),
            ("layers.0.attention.qkv.weight", "BF16"),
            ("layers.0.feed_forward.w1.weight", "BF16"),
        ]
    }
    fn qwen3_encoder_keys() -> Vec<(&'static str, &'static str)> {
        vec![
            ("model.embed_tokens.weight", "BF16"),
            ("model.layers.0.self_attn.q_proj.weight", "BF16"),
            ("model.layers.0.self_attn.q_norm.weight", "BF16"),
            ("model.layers.0.self_attn.k_norm.weight", "BF16"),
            ("model.layers.0.mlp.gate_proj.weight", "BF16"),
        ]
    }
    fn vae_keys() -> Vec<(&'static str, &'static str)> {
        vec![
            ("encoder.conv_in.weight", "F32"),
            ("encoder.down.0.block.0.conv1.weight", "F32"),
            ("decoder.conv_out.weight", "F32"),
            ("decoder.up.0.block.0.conv1.weight", "F32"),
        ]
    }
    /// A ComfyUI UMT5-XXL text encoder (`text_encoders/umt5_xxl_fp8_e4m3fn_scaled`): T5 encoder keys +
    /// a `.scale_weight` companion → detector `(t5, text_encoder, scaled_fp8_companion)`.
    fn umt5_scaled_te_keys() -> Vec<(&'static str, &'static str)> {
        vec![
            ("shared.weight", "F32"),
            ("encoder.block.0.layer.0.SelfAttention.q.weight", "F8_E4M3"),
            (
                "encoder.block.0.layer.0.SelfAttention.q.scale_weight",
                "F32",
            ),
            (
                "encoder.block.0.layer.1.DenseReluDense.wi_0.weight",
                "F8_E4M3",
            ),
            ("encoder.final_layer_norm.weight", "F32"),
        ]
    }

    fn models_root(temp: &Path) -> PathBuf {
        temp.join("ComfyUI").join("models")
    }

    fn scan(roots: &[PathBuf]) -> Vec<Value> {
        scan_external_base_models(roots, &mut ExternalBaseModelCache::default())
    }

    #[test]
    fn no_roots_configured_yields_no_rows() {
        assert!(scan(&[]).is_empty());
    }

    #[test]
    fn a_root_without_base_subtrees_is_empty() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = models_root(temp.path());
        std::fs::create_dir_all(root.join("loras")).expect("mkdir");
        assert!(scan(&[root]).is_empty());
    }

    #[test]
    fn z_image_bf16_with_encoder_and_vae_assembles_complete_and_runnable() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = models_root(temp.path());
        write_safetensors(
            &root.join("unet").join("z_image_turbo_bf16.safetensors"),
            &z_image_keys(),
        );
        write_safetensors(
            &root.join("text_encoders").join("qwen_3_4b.safetensors"),
            &qwen3_encoder_keys(),
        );
        write_safetensors(&root.join("vae").join("ae.safetensors"), &vae_keys());

        let rows = scan(&[root]);
        assert_eq!(
            rows.len(),
            1,
            "one anchored model, encoder+vae folded in as components"
        );
        let row = &rows[0];
        assert_eq!(row["family"], "z-image");
        assert_eq!(row["type"], "image");
        assert_eq!(row["catalogScope"], EXTERNAL_SCOPE);
        assert_eq!(row["importSourceShape"], "comfy_ui_tree");
        assert_eq!(row["removable"], false);
        assert_eq!(row["assembly"], "complete");
        // sc-10668: z-image bf16 complete is now runnable (the candle loader exists),
        // so it is usable and the picker offers it. No unusable reason.
        assert_eq!(row["usable"], true);
        assert_eq!(row["unusableReason"], Value::Null);
        assert_eq!(row["quant"], "bf16");
        assert_eq!(row["components"].as_array().unwrap().len(), 3);
        assert!(row["id"].as_str().unwrap().starts_with(EXTERNAL_ID_PREFIX));
    }

    #[test]
    fn z_image_without_encoder_is_incomplete_with_reason() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = models_root(temp.path());
        write_safetensors(
            &root.join("unet").join("z_image_turbo_bf16.safetensors"),
            &z_image_keys(),
        );
        write_safetensors(&root.join("vae").join("ae.safetensors"), &vae_keys());
        // No text_encoders/ at all.

        let rows = scan(&[root]);
        let row = &rows[0];
        assert_eq!(row["assembly"], "incomplete");
        assert_eq!(row["usable"], false);
        let reason = row["unusableReason"].as_str().unwrap();
        assert!(reason.contains("text encoder"), "reason: {reason}");
        // The VAE it did find is still listed as a component.
        assert_eq!(row["components"].as_array().unwrap().len(), 2);
    }

    /// A ComfyUI Qwen-Image DiT (`diffusion_models/*_fp8_e4m3fn`): dual-stream MMDiT keys under the
    /// BFL `model.diffusion_model.` prefix, all `F8_E4M3` → detector `(qwen-image, transformer,
    /// fp8_e4m3)`.
    fn qwen_image_dit_fp8_keys() -> Vec<(&'static str, &'static str)> {
        vec![
            ("model.diffusion_model.img_in.weight", "F8_E4M3"),
            (
                "model.diffusion_model.transformer_blocks.0.attn.add_q_proj.weight",
                "F8_E4M3",
            ),
            (
                "model.diffusion_model.transformer_blocks.0.img_mlp.net.0.proj.weight",
                "F8_E4M3",
            ),
        ]
    }

    #[test]
    fn qwen_image_fp8_dit_is_snapshot_backed_complete_and_runnable() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = models_root(temp.path());
        // Only the DiT is on disk — no tree text encoder / VAE. Snapshot-backed families (sc-10670)
        // source the TE / VAE / tokenizer from a resident SceneWorks snapshot, so a bare DiT anchor is
        // still complete and runnable.
        write_safetensors(
            &root
                .join("diffusion_models")
                .join("qwen_image_2512_fp8_e4m3fn.safetensors"),
            &qwen_image_dit_fp8_keys(),
        );

        let rows = scan(&[root]);
        assert_eq!(rows.len(), 1, "one anchored DiT model");
        let row = &rows[0];
        assert_eq!(row["family"], "qwen-image");
        assert_eq!(row["catalogScope"], EXTERNAL_SCOPE);
        assert_eq!(row["importSourceShape"], "comfy_ui_tree");
        assert_eq!(row["type"], "image");
        assert_eq!(row["quant"], "fp8_e4m3_scalar");
        assert_eq!(row["importQuantFormat"], row["quant"]);
        assert_eq!(row["sourcePrecision"]["codecId"], "fp8-e4m3-scalar-v1");
        let refusal = sceneworks_core::request_validation::validate(
            "image",
            json!({"projectId":"p", "model": row["id"], "prompt":"snow", "advanced": {"quantTier":"q4"}})
                .as_object()
                .unwrap(),
            row.as_object().unwrap(),
        ).unwrap_err();
        assert_eq!(refusal.field, "advanced.quantTier");
        assert_eq!(row["assembly"], "complete");
        // sc-10670: plain-fp8 qwen DiT has a loader → usable, no reason. No tree VAE here, so
        // components is the DiT alone (the TE is always snapshot-sourced, sc-10671).
        assert_eq!(row["usable"], true);
        assert_eq!(row["unusableReason"], Value::Null);
        assert_eq!(row["components"].as_array().unwrap().len(), 1);
    }

    /// A ComfyUI FLUX.2-dev fp8-mixed DiT (`diffusion_models/flux2_dev_fp8mixed`): BFL-native MMDiT keys
    /// with **inline-scale fp8** MLPs (`.weight` F8_E4M3 + `.weight_scale`/`.input_scale` F32 siblings)
    /// and BF16 attention/modulation → detector `(flux2, transformer, fp8_inline_scale)` (sc-10662).
    fn flux2_dev_dit_inline_scale_keys() -> Vec<(&'static str, &'static str)> {
        vec![
            ("guidance_in.in_layer.weight", "BF16"),
            ("double_stream_modulation_img.lin.weight", "BF16"),
            ("single_stream_modulation.lin.weight", "BF16"),
            ("double_blocks.0.img_attn.qkv.weight", "BF16"),
            ("double_blocks.0.img_mlp.0.weight", "F8_E4M3"),
            ("double_blocks.0.img_mlp.0.weight_scale", "F32"),
            ("double_blocks.0.img_mlp.0.input_scale", "F32"),
            ("single_blocks.0.linear1.weight", "F8_E4M3"),
            ("single_blocks.0.linear1.weight_scale", "F32"),
            ("single_blocks.0.linear1.input_scale", "F32"),
        ]
    }

    #[test]
    fn a_flux2_backbone_without_dev_guidance_does_not_claim_the_dev_tree_route() {
        let temp = tempfile::tempdir().unwrap();
        let root = models_root(temp.path());
        let mut keys = flux2_dev_dit_inline_scale_keys();
        keys.retain(|(name, _)| !name.starts_with("guidance_in."));
        write_safetensors(&root.join("diffusion_models/klein.safetensors"), &keys);
        let rows = scan(&[root]);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["usable"], false);
        assert!(rows[0]["unusableReason"]
            .as_str()
            .unwrap()
            .contains("Dev guidance"));
    }

    #[test]
    fn flux2_inline_scale_dit_is_snapshot_backed_complete_and_runnable() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = models_root(temp.path());
        // Only the DiT is on disk — no tree text encoder / VAE. flux2 is snapshot-backed (sc-10680): the
        // Mistral-3 TE / AutoencoderKL-Flux2 / tokenizer come from a resident SceneWorks snapshot, so a
        // bare inline-scale-fp8 DiT anchor is still complete and runnable.
        write_safetensors(
            &root
                .join("diffusion_models")
                .join("flux2_dev_fp8mixed.safetensors"),
            &flux2_dev_dit_inline_scale_keys(),
        );

        let rows = scan(&[root]);
        assert_eq!(rows.len(), 1, "one anchored DiT model");
        let row = &rows[0];
        assert_eq!(row["family"], "flux2");
        assert_eq!(row["type"], "image");
        assert_eq!(row["quant"], "fp8_e4m3_scalar");
        assert_eq!(row["assembly"], "complete");
        // sc-10680: inline-scale-fp8 flux2 DiT has a loader → usable, no reason. No tree VAE here, so
        // components is the DiT alone (the TE / VAE are snapshot-sourced).
        assert_eq!(row["usable"], true);
        assert_eq!(row["unusableReason"], Value::Null);
        assert_eq!(row["components"].as_array().unwrap().len(), 1);
    }

    /// A ComfyUI Wan2.2 A14B expert (`unet/wan2.2_*_fp8_scaled`): native-Wan dual-stream keys + a
    /// `.scale_weight` companion + the `scaled_fp8` marker → detector `(wan-video, transformer,
    /// scaled_fp8_companion)`.
    fn wan_expert_scaled_fp8_keys() -> Vec<(&'static str, &'static str)> {
        vec![
            ("patch_embedding.weight", "F16"),
            ("blocks.0.self_attn.q.weight", "F8_E4M3"),
            ("blocks.0.self_attn.q.scale_weight", "F32"),
            ("blocks.0.cross_attn.q.weight", "F8_E4M3"),
            ("blocks.0.ffn.0.weight", "F8_E4M3"),
            ("scaled_fp8", "F8_E4M3"),
        ]
    }

    #[test]
    fn wan_moe_high_low_experts_pair_into_one_complete_runnable_model() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = models_root(temp.path());
        write_safetensors(
            &root
                .join("unet")
                .join("wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"),
            &wan_expert_scaled_fp8_keys(),
        );
        write_safetensors(
            &root
                .join("unet")
                .join("wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"),
            &wan_expert_scaled_fp8_keys(),
        );

        let rows = scan(&[root]);
        assert_eq!(rows.len(), 1, "two experts pair into one MoE model");
        let row = &rows[0];
        assert_eq!(row["family"], "wan-video");
        assert_eq!(row["type"], "video");
        assert_eq!(row["quant"], "fp8_e4m3_scalar");
        assert_eq!(row["assembly"], "complete");
        assert_eq!(row["usable"], true);
        assert_eq!(row["unusableReason"], Value::Null);
        // Both experts as components, tagged with their MoE roles.
        let components = row["components"].as_array().unwrap();
        assert_eq!(components.len(), 2);
        let roles: Vec<&str> = components
            .iter()
            .map(|c| c["role"].as_str().unwrap())
            .collect();
        assert!(roles.contains(&"transformer_high"));
        assert!(roles.contains(&"transformer_low"));
    }

    #[test]
    fn wan_experts_keep_independently_verified_precisions() {
        let temp = tempfile::tempdir().unwrap();
        let root = models_root(temp.path());
        let high = wan_expert_scaled_fp8_keys();
        let low: Vec<_> = high
            .iter()
            .map(|&(key, dtype)| {
                (
                    key,
                    if key.ends_with(".weight") && dtype == "F8_E4M3" {
                        "F8_E5M2"
                    } else {
                        dtype
                    },
                )
            })
            .collect();
        write_safetensors(&root.join("unet/wan_high_noise.safetensors"), &high);
        write_safetensors(&root.join("unet/wan_low_noise.safetensors"), &low);
        let rows = scan(&[root]);
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row["usable"], true);
        assert_eq!(row["sourcePrecision"]["format"], "fp8_e4m3_scalar");
        let components = row["components"].as_array().unwrap();
        for (role, format) in [
            ("transformer_high", "fp8_e4m3_scalar"),
            ("transformer_low", "fp8_e5m2_scalar"),
        ] {
            assert_eq!(
                components
                    .iter()
                    .find(|component| component["role"] == role)
                    .unwrap()["quant"],
                format
            );
        }
    }

    #[test]
    fn wan_moe_folds_in_tree_umt5_te_and_vae() {
        // sc-10909: a complete MoE pair + the tree's UMT5 TE (t5, scaled-fp8) + one VAE → both are
        // folded in as in-place `text_encoder` / `vae` components (worker reads them in place). Still
        // complete + runnable (the tokenizer stays snapshot-sourced).
        let temp = tempfile::tempdir().expect("tempdir");
        let root = models_root(temp.path());
        write_safetensors(
            &root
                .join("unet")
                .join("wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"),
            &wan_expert_scaled_fp8_keys(),
        );
        write_safetensors(
            &root
                .join("unet")
                .join("wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"),
            &wan_expert_scaled_fp8_keys(),
        );
        write_safetensors(
            &root
                .join("text_encoders")
                .join("umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
            &umt5_scaled_te_keys(),
        );
        write_safetensors(
            &root.join("vae").join("wan_2.1_vae.safetensors"),
            &vae_keys(),
        );

        let rows = scan(&[root]);
        assert_eq!(rows.len(), 1, "experts + TE + VAE → one MoE model");
        let row = &rows[0];
        assert_eq!(row["family"], "wan-video");
        assert_eq!(row["assembly"], "complete");
        assert_eq!(row["usable"], true);
        let components = row["components"].as_array().unwrap();
        let roles: Vec<&str> = components
            .iter()
            .map(|c| c["role"].as_str().unwrap())
            .collect();
        assert!(roles.contains(&"transformer_high"));
        assert!(roles.contains(&"transformer_low"));
        // The TE + VAE ride along as in-place components (worker reads them in place, sc-10909).
        assert!(roles.contains(&"text_encoder"), "roles: {roles:?}");
        assert!(roles.contains(&"vae"), "roles: {roles:?}");
    }

    #[test]
    fn wan_lone_expert_without_sibling_is_incomplete() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = models_root(temp.path());
        // Only the high-noise expert — no low-noise sibling.
        write_safetensors(
            &root
                .join("unet")
                .join("wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"),
            &wan_expert_scaled_fp8_keys(),
        );
        let rows = scan(&[root]);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["family"], "wan-video");
        assert_eq!(rows[0]["assembly"], "incomplete");
        assert_eq!(rows[0]["usable"], false);
        let reason = rows[0]["unusableReason"].as_str().unwrap();
        assert!(reason.contains("MoE expert"), "reason: {reason}");
    }

    #[test]
    fn qwen_image_fp8_dit_folds_in_an_unambiguous_tree_vae() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = models_root(temp.path());
        // sc-10830: a ComfyUI Qwen-Image DiT + exactly one tree VAE → the VAE is folded into the row
        // (read in place, native WAN-VAE keys → diffusers remap) while the DiT stays snapshot-backed
        // for the TE + tokenizer. Runnable, and the VAE rides along as a `vae` component.
        write_safetensors(
            &root
                .join("diffusion_models")
                .join("qwen_image_2512_fp8_e4m3fn.safetensors"),
            &qwen_image_dit_fp8_keys(),
        );
        write_safetensors(
            &root.join("vae").join("qwen_image_vae.safetensors"),
            &vae_keys(),
        );

        let rows = scan(&[root]);
        assert_eq!(rows.len(), 1, "one anchored DiT model");
        let row = &rows[0];
        assert_eq!(row["family"], "qwen-image");
        assert_eq!(row["assembly"], "complete");
        assert_eq!(row["usable"], true);
        let components = row["components"].as_array().unwrap();
        assert_eq!(components.len(), 2, "DiT + the folded tree VAE");
        assert!(
            components
                .iter()
                .any(|component| component["role"] == "vae"),
            "the tree VAE rides along as a `vae` component for the in-place loader"
        );
    }

    #[test]
    fn all_in_one_checkpoint_is_complete_standalone() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = models_root(temp.path());
        // LTX all-in-one: DiT + an embedded audio_vae → ComponentRole::Checkpoint.
        write_safetensors(
            &root.join("checkpoints").join("ltx.safetensors"),
            &[
                ("model.diffusion_model.scale_shift_table", "F32"),
                ("model.diffusion_model.patchify_proj.weight", "BF16"),
                (
                    "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight",
                    "BF16",
                ),
                ("audio_vae.encoder.conv_in.conv.weight", "F32"),
                ("audio_vae.decoder.conv_out.conv.weight", "F32"),
            ],
        );

        let rows = scan(&[root]);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["family"], "ltx-video");
        assert_eq!(rows[0]["type"], "video");
        assert_eq!(rows[0]["assembly"], "complete");
        // No component pool needed: the checkpoint carries every role.
        assert_eq!(rows[0]["components"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn text_encoders_and_vaes_are_components_never_anchors() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = models_root(temp.path());
        // Only encoders + VAEs, no transformer/checkpoint → nothing to anchor a model.
        write_safetensors(
            &root.join("text_encoders").join("qwen_3_4b.safetensors"),
            &qwen3_encoder_keys(),
        );
        write_safetensors(&root.join("vae").join("ae.safetensors"), &vae_keys());
        assert!(scan(&[root]).is_empty());
    }

    #[test]
    fn unrecognized_family_transformer_is_surfaced_unusable() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = models_root(temp.path());
        write_safetensors(
            &root.join("diffusion_models").join("mystery.safetensors"),
            &[("some.unknown.tensor", "BF16"), ("another.mystery", "BF16")],
        );
        let rows = scan(&[root]);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["usable"], false);
        assert_eq!(rows[0]["assembly"], "unrecognized");
        assert!(rows[0].get("family").is_none());
    }

    #[test]
    fn a_family_without_a_component_spec_is_unassemblable_but_listed() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = models_root(temp.path());
        // Ideogram is a recognized DiT family with no component spec here yet.
        write_safetensors(
            &root.join("diffusion_models").join("ideo.safetensors"),
            &[
                ("embed_image_indicator.weight", "BF16"),
                ("layers.0.attention.qkv.weight", "BF16"),
                ("layers.0.adaln_modulation.weight", "BF16"),
            ],
        );
        let rows = scan(&[root]);
        assert_eq!(rows[0]["family"], "ideogram");
        assert_eq!(rows[0]["assembly"], "unassemblable");
        assert_eq!(rows[0]["usable"], false);
    }

    #[test]
    fn gguf_anchor_is_surfaced() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = models_root(temp.path());
        let unet = root.join("unet");
        std::fs::create_dir_all(&unet).expect("mkdir");
        // GGUF magic + padding; detected by magic, classified quant=gguf.
        std::fs::write(unet.join("wan_Q4_K_S.gguf"), b"GGUF\x00\x00\x00\x00padding")
            .expect("write");
        let rows = scan(&[root]);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["quant"], "gguf");
        assert_eq!(rows[0]["usable"], false);
    }

    #[test]
    fn scan_populates_then_prunes_the_cache() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = models_root(temp.path());
        let anchor = root.join("unet").join("z_image_turbo_bf16.safetensors");
        write_safetensors(&anchor, &z_image_keys());

        let mut cache = ExternalBaseModelCache::default();
        let rows = scan_external_base_models(std::slice::from_ref(&root), &mut cache);
        assert_eq!(rows.len(), 1);
        assert_eq!(cache.len(), 1);

        let again = scan_external_base_models(std::slice::from_ref(&root), &mut cache);
        assert_eq!(
            again, rows,
            "unchanged tree yields identical rows from cache"
        );

        std::fs::remove_file(&anchor).expect("remove");
        assert!(scan_external_base_models(&[root], &mut cache).is_empty());
        assert!(cache.is_empty(), "vanished files pruned");
    }

    /// Manual smoke against a real ComfyUI tree — the only test over real key
    /// conventions. Ignored by default:
    ///
    /// ```text
    /// SCENEWORKS_EXTERNAL_MODEL_ROOTS='C:\Users\Michael\ComfyUI-Shared\models' \
    ///   cargo test -p sceneworks-rust-api --lib external_base_models::tests::real_comfyui_base_tree -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "requires a real ComfyUI models tree via SCENEWORKS_EXTERNAL_MODEL_ROOTS"]
    fn real_comfyui_base_tree_smoke() {
        let roots = sceneworks_core::external_roots::parse_external_model_roots(
            std::env::var("SCENEWORKS_EXTERNAL_MODEL_ROOTS")
                .ok()
                .as_deref(),
        );
        assert!(!roots.is_empty(), "set SCENEWORKS_EXTERNAL_MODEL_ROOTS");
        let rows = scan(&roots);
        println!("\n{} external base models assembled:", rows.len());
        for row in &rows {
            println!(
                "  {:<10} {:<14} {:<8} {}",
                row.get("family")
                    .and_then(Value::as_str)
                    .unwrap_or("(none)"),
                row["assembly"].as_str().unwrap_or_default(),
                row.get("quant").and_then(Value::as_str).unwrap_or("-"),
                row["name"].as_str().unwrap_or_default(),
            );
        }
        assert!(
            !rows.is_empty(),
            "the real tree should assemble base models"
        );
    }
}
