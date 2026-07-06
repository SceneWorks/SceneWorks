#[test]
fn nvidia_smi_parsing_and_visible_device_filtering_match_python_worker() {
    let gpus = parse_nvidia_smi_gpus(
        "0, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887, 4096, 93791, 12\n\
             1, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887, 8192, 89695, 25\n",
    );

    assert_eq!(
        gpus.iter().map(|gpu| gpu.id.as_str()).collect::<Vec<_>>(),
        ["0", "1"]
    );
    assert_eq!(
        gpus[0].name,
        "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (97887 MB)"
    );
    assert!(gpus[1]
        .capabilities
        .iter()
        .any(|capability| capability.as_str() == "nvidia"));
    assert!(gpus[1]
        .capabilities
        .iter()
        .any(|capability| capability.as_str() == "placeholder"));
    assert_eq!(
        gpus[0].utilization,
        Some(WorkerUtilizationSnapshot {
            memory_total_mb: Some(97887),
            memory_used_mb: Some(4096),
            memory_free_mb: Some(93791),
            gpu_load_percent: Some(12.0),
        })
    );

    assert_eq!(visible_gpu_ids(None), None);
    assert_eq!(visible_gpu_ids(Some("all")), None);
    assert_eq!(visible_gpu_ids(Some("none")), Some(Vec::new()));
    assert_eq!(
        visible_gpu_ids(Some("0, GPU-abcd")),
        Some(vec!["0".to_owned(), "GPU-abcd".to_owned()])
    );
}

// sc-9300 (epic 9083): the INT8-ConvRot sm_89 compute-cap probe parses `nvidia-smi
// --query-gpu=compute_cap` and takes the HIGHEST cap across GPUs (a multi-GPU box may mix an Ada card
// with an older one). Blank / unparseable rows are ignored; nothing parseable ⇒ `None` (ineligible).
#[test]
fn compute_cap_parse_takes_the_highest_and_tolerates_junk() {
    // Single Ada card (RTX 4090 = 8.9) clears the floor.
    assert_eq!(parse_max_compute_cap("8.9\n"), Some(8.9));
    // Blackwell (12.0).
    assert_eq!(parse_max_compute_cap("12.0\n"), Some(12.0));
    // Multi-GPU: an A100 (8.0) + an RTX 4090 (8.9) → the max (8.9) is what the gate sees.
    assert_eq!(parse_max_compute_cap("8.0\n8.9\n"), Some(8.9));
    // Junk / blank rows are ignored; a trailing blank line doesn't break the max.
    assert_eq!(parse_max_compute_cap("  \n7.5\nN/A\n"), Some(7.5));
    // No parseable cap (nvidia-smi absent / empty) ⇒ None ⇒ ConvRot-ineligible.
    assert_eq!(parse_max_compute_cap(""), None);
    assert_eq!(parse_max_compute_cap("\n[N/A]\n"), None);
}

#[test]
fn auto_worker_ids_and_child_environment_match_python_supervisor() {
    assert_eq!(gpu_worker_id("worker-gpu-auto-0", "0"), "worker-gpu-auto-0");
    assert_eq!(gpu_worker_id("worker-gpu-auto-0", "1"), "worker-gpu-auto-1");
    assert_eq!(cpu_worker_id("worker-gpu-auto-0"), "worker-gpu-auto-cpu");

    let gpus = vec![fallback_gpu("0"), fallback_gpu("1")];
    let specs = auto_worker_specs("worker-gpu-auto-0", &gpus);
    assert_eq!(
        specs
            .iter()
            .map(|spec| spec.worker_id.as_str())
            .collect::<Vec<_>>(),
        [
            "worker-gpu-auto-0",
            "worker-gpu-auto-1",
            "worker-gpu-auto-cpu"
        ]
    );
    assert_eq!(
        specs
            .iter()
            .map(|spec| spec.gpu_id.as_str())
            .collect::<Vec<_>>(),
        ["0", "1", "cpu"]
    );

    let gpu_env = child_environment(&WorkerSpec {
        worker_id: "worker-gpu-auto-1".to_owned(),
        gpu_id: "1".to_owned(),
    });
    assert_eq!(gpu_env["SCENEWORKS_UTILITY_JOBS"], "0");
    assert_eq!(gpu_env["CUDA_VISIBLE_DEVICES"], "1");

    let cpu_env = child_environment(&WorkerSpec {
        worker_id: "worker-gpu-auto-cpu".to_owned(),
        gpu_id: "cpu".to_owned(),
    });
    assert_eq!(cpu_env["SCENEWORKS_UTILITY_JOBS"], "1");
    assert_eq!(cpu_env["CUDA_VISIBLE_DEVICES"], "");
}

#[test]
fn utility_worker_specs_scale_to_requested_count() {
    let single = utility_worker_specs("rust-utility-worker-0", 1);
    assert_eq!(
        single
            .iter()
            .map(|spec| spec.worker_id.as_str())
            .collect::<Vec<_>>(),
        ["rust-utility-worker-cpu"]
    );

    let pool = utility_worker_specs("rust-utility-worker-0", 4);
    assert_eq!(
        pool.iter()
            .map(|spec| spec.worker_id.as_str())
            .collect::<Vec<_>>(),
        [
            "rust-utility-worker-cpu",
            "rust-utility-worker-cpu-1",
            "rust-utility-worker-cpu-2",
            "rust-utility-worker-cpu-3",
        ]
    );
    assert!(pool.iter().all(|spec| spec.gpu_id == "cpu"));

    // A count of 0 must still yield a single worker rather than an empty pool.
    assert_eq!(utility_worker_specs("rust-utility-worker-0", 0).len(), 1);
}

#[test]
fn rust_cpu_capabilities_do_not_claim_gpu_generation_jobs() {
    let cpu_capabilities = worker_capabilities_with_utility(&cpu_gpu(), true);

    assert!(cpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "model_download"));
    assert!(cpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "placeholder"));
    assert!(cpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "timeline_export"));
    // The CPU utility worker advertises only the procedural *preview*
    // capabilities; real detection/tracking route to the Python GPU worker.
    assert!(cpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "person_detect_preview"));
    assert!(cpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "person_track_preview"));
    assert!(!cpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "person_detect"));
    assert!(!cpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "person_track"));
    assert!(!cpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "image_generate"));
    assert!(!cpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "video_generate"));

    let gpu_capabilities = worker_capabilities_with_utility(&fallback_gpu("0"), false);
    assert!(gpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "gpu"));
    assert!(gpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "placeholder"));
    assert!(!gpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "model_download"));
    assert!(!gpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "image_generate"));
}

/// The Apple-Silicon MLX GPU worker (epic 3018) advertises `image_generate`,
/// `image_edit` (sc-3513), + `video_generate` so the API routes generation/editing to
/// it, but it must NOT pick up CPU utility jobs (those stay on the CPU worker) — the
/// inverse of the CPU-worker contract above. `video_generate` lands with the video
/// runtime (sc-3033).
#[cfg(target_os = "macos")]
#[test]
fn mlx_gpu_advertises_generation_capabilities_only() {
    let mlx = mlx_gpu(&crate::Settings::from_env());
    assert_eq!(mlx.id, "mlx");
    let capabilities = worker_capabilities_with_utility(&mlx, true);
    assert!(capabilities
        .iter()
        .any(|capability| capability.as_str() == "image_generate"));
    // Plain Image Edit (sc-3513): without this the API's worker_supports_job would
    // reject an `image_edit` claim and the job would silently fall back to torch.
    assert!(capabilities
        .iter()
        .any(|capability| capability.as_str() == "image_edit"));
    assert!(capabilities
        .iter()
        .any(|capability| capability.as_str() == "video_generate"));
    assert!(capabilities
        .iter()
        .any(|capability| capability.as_str() == "training_caption"));
    // Real, model-backed person detect/track are ported to the MLX worker (sc-3709): the
    // worker advertises the non-preview capabilities so the API routes real jobs here.
    assert!(capabilities
        .iter()
        .any(|capability| capability.as_str() == "person_detect"));
    assert!(capabilities
        .iter()
        .any(|capability| capability.as_str() == "person_track"));
    assert!(capabilities
        .iter()
        .any(|capability| capability.as_str() == "gpu"));
    // No CPU utility capabilities, even with utility jobs enabled — only the CPU
    // worker (which carries `Cpu`) gets those extended onto it.
    for utility in [
        "model_download",
        "model_import",
        "timeline_export",
        "cpu",
        "placeholder",
    ] {
        assert!(
            !capabilities
                .iter()
                .any(|capability| capability.as_str() == utility),
            "MLX GPU worker should not advertise utility capability {utility}"
        );
    }
}

/// sc-3723 acceptance gate: with default settings (mlx enabled) and ALL provider crates
/// linked on macOS, the registry-DERIVED advertisement must equal exactly today's hardcoded
/// MLX capability set (order-independent). This is the invariant that lets the dispatch table
/// move + the flags become descriptor-derived without changing what the worker advertises.
#[cfg(target_os = "macos")]
#[test]
fn mlx_gpu_capability_set_matches_expected_full_set() {
    use sceneworks_core::contracts::WorkerCapability;
    use std::collections::BTreeSet;
    let mlx = mlx_gpu(&crate::Settings::from_env());
    let actual: BTreeSet<_> = mlx.capabilities.iter().cloned().collect();
    let expected: BTreeSet<_> = [
        // seed
        WorkerCapability::Gpu,
        // 7 registry-derived
        WorkerCapability::ImageGenerate,
        WorkerCapability::VideoGenerate,
        WorkerCapability::LoraTrain,
        WorkerCapability::LoraTrainExecute,
        WorkerCapability::TrainingCaption,
        // sc-5552: the native MLX `prompt_refine` TextLlm provider (mlx-gen-prompt-refine) is
        // force-linked on macOS, so the registry now derives PromptRefine from its descriptor.
        WorkerCapability::PromptRefine,
        // sc-6535: mlx-gen-clip registers the CLIP `clip_vit_l14` ImageEmbedder (force-linked in
        // dataset_analysis_jobs.rs), so the registry derives DatasetAnalysis from its descriptor.
        WorkerCapability::DatasetAnalysis,
        // carve-outs
        WorkerCapability::ImageEdit,
        WorkerCapability::ImageDetail,
        WorkerCapability::ImageVqa,
        WorkerCapability::ImageInterleave,
        WorkerCapability::VideoExtend,
        WorkerCapability::VideoBridge,
        WorkerCapability::PersonReplace,
        WorkerCapability::PoseDetect,
        WorkerCapability::KpsExtract,
        // sc-6538: the native SCRFD+ArcFace face stack (mlx-gen-face) is hardcoded in mlx_gpu (no
        // gen-core registry for FaceEmbedder), so DatasetFaceAnalysis is advertised on Mac like
        // KpsExtract.
        WorkerCapability::DatasetFaceAnalysis,
        // sc-4415: on-demand face-likeness compare — same hardcoded-in-mlx_gpu native face stack as
        // DatasetFaceAnalysis (no gen-core registry for FaceEmbedder), advertised on Mac like KpsExtract.
        WorkerCapability::FaceLikenessCompare,
        WorkerCapability::ImageUpscale,
        // sc-6539: Dataset Doctor one-tap upscale — reuses the Real-ESRGAN engine, advertised
        // wherever image_upscale is.
        WorkerCapability::DatasetUpscale,
        // sc-6105: smart-select segmentation (native-MLX SAM3 box-prompt) — Mac-only, advertised
        // only here so an `image_segment` job routes to the MLX worker by construction.
        WorkerCapability::ImageSegment,
        WorkerCapability::VideoUpscale,
        WorkerCapability::PersonDetect,
        WorkerCapability::PersonTrack,
    ]
    .into_iter()
    .collect();
    assert_eq!(
        actual, expected,
        "registry-derived MLX capability set drifted from the expected full set"
    );
}

/// sc-3723: every MODEL_TABLE row resolves through the registry-joined `mlx_model` lookup
/// (its engine id is registered by a linked provider crate), and the descriptor-derived
/// guidance/negative-prompt flags match the pre-deletion hardcoded values — proving the two
/// removed row flags were faithfully replaced by the engine's own advertised surface.
#[cfg(target_os = "macos")]
#[test]
fn model_table_rows_resolve_and_flags_match_descriptor() {
    use crate::engines::{mlx_model, MODEL_TABLE};
    // (sceneworks_id, supports_guidance, supports_negative_prompt) — the exact pre-sc-3723
    // values that used to live on each MlxModel row.
    let expected: &[(&str, bool, bool)] = &[
        ("z_image_turbo", false, false),
        // Base (non-distilled) Z-Image (sc-8320): the undistilled foundation model is full real CFG —
        // supports_guidance=true + supports_negative_prompt=true (vs Turbo's CFG-free distill).
        ("z_image", true, true),
        // Ideogram 4 (epic 4725): asymmetric-CFG guidance (supports_guidance=true) with no user
        // negative prompt (the "negative" is a fixed unconditional DiT, not a prompt).
        ("ideogram_4", true, false),
        // Ideogram 4 Turbo (mlx-gen #488): CFG-free single-DiT — the descriptor drops guidance
        // (supports_guidance=false), no negative prompt. Requires the mlx-gen pin to include the
        // `ideogram_4_turbo` engine (PR #489) for this row to resolve through the registry.
        ("ideogram_4_turbo", false, false),
        ("z_image_edit", false, false),
        ("flux_schnell", false, false),
        ("flux_dev", true, false),
        ("qwen_image", true, true),
        ("qwen_image_edit", true, true),
        ("qwen_image_edit_2509", true, true),
        ("qwen_image_edit_2511", true, true),
        // sc-3723 finding: the lightning variant shares the `qwen_image_edit` engine id, whose
        // descriptor advertises supports_negative_prompt=true. The old row hardcoded `false`
        // (the CFG-off recipe), but the engine itself drops the negative branch under the
        // `lightning` sampler (model_edit.rs `neg = None` when is_lightning), so the
        // descriptor-derived `true` is behavior-equivalent — the CFG-off recipe is
        // engine-enforced, not a model capability the worker has to suppress.
        ("qwen_image_edit_2511_lightning", true, true),
        ("flux2_klein_9b", true, false),
        ("flux2_klein_9b_kv", true, false),
        ("flux2_klein_9b_true_v2", true, false),
        // FLUX.2-dev (epic 5914 / sc-5921): its own `flux2_dev` engine id, embedded distilled
        // guidance (supports_guidance=true) with no negative prompt / true-CFG.
        ("flux2_dev", true, false),
        ("sdxl", true, true),
        ("realvisxl", true, true),
        // RealVisXL Lightning (sc-6075): shares the `sdxl` engine id, whose descriptor advertises
        // guidance + negative prompt (true, true). The few-step recipe runs CFG-off (guidance 1.0,
        // negative inert) via the worker-forced `lightning` sampler, but that's a recipe default,
        // not a capability the descriptor drops — so the descriptor-derived flags stay (true, true).
        ("realvisxl_lightning", true, true),
        ("kolors", true, true),
        ("chroma1_hd", false, true),
        ("chroma1_base", false, true),
        ("chroma1_flash", false, true),
        ("sensenova_u1_8b", true, false),
        // Infographic-V2 (epic 9959): rides the base `sensenova_u1_8b` engine id, so its descriptor
        // flags are identical to the base (guidance true, negative false).
        ("sensenova_u1_8b_infographic_v2", true, false),
        ("sensenova_u1_8b_fast", true, false),
        // Infographic-V2 fast (epic 9959): rides the base `sensenova_u1_8b_fast` engine id → same flags.
        ("sensenova_u1_8b_infographic_v2_fast", true, false),
        // Lens / Lens-Turbo (epic 3164 / sc-5105): the `mlx-gen-lens` descriptor advertises the
        // norm-rescaled CFG path (`supports_guidance=true`) + a negative prompt
        // (`supports_negative_prompt=true`) — a standard guidance family (NOT true-CFG), so the worker
        // forwards the CFG scale via `guidance`. Turbo simply defaults guidance to 1.0 (≈ no CFG).
        ("lens", true, true),
        ("lens_turbo", true, true),
        // Bernini still-image companion (epic 4699 / sc-5424): the image-typed `bernini_image` id
        // maps to the SAME `bernini` engine the video id uses (`Modality::Both`). Standard guidance
        // family — `supports_guidance=true` (omega_txt) + `supports_negative_prompt=true`.
        ("bernini_image", true, true),
        // Boogu-Image-0.1 (epic 6387 / sc-6399): Base + Edit are true-CFG (supports_guidance=true)
        // with no user negative prompt (the CFG-negative is a fixed empty/drop instruction). Turbo is
        // the DMD few-step, CFG-free distill (supports_guidance=false). None take a negative prompt.
        ("boogu_image", true, false),
        ("boogu_image_turbo", false, false),
        ("boogu_image_edit", true, false),
        // Krea 2 Turbo (epic 7565 / sc-7572): TDM-distilled few-step, CFG-free
        // (supports_guidance=false) with no user negative prompt (supports_negative_prompt=false) — the
        // z_image_turbo / boogu_image_turbo distilled-turbo pattern.
        ("krea_2_turbo", false, false),
        // SD3.5 Large (epic 7841 / sc-7871): true-CFG MMDiT flagship — supports_guidance=true +
        // supports_negative_prompt=true (the `sd3_5_large` descriptor advertises supports_true_cfg).
        ("sd3_5_large", true, true),
        // SD3.5 Large Turbo (epic 7841 / sc-7871): the ADD-distilled few-step, CFG-off sibling — the
        // `sd3_5_large_turbo` descriptor drops guidance + negative prompt (supports_guidance=false,
        // supports_negative_prompt=false), the distilled-turbo pattern.
        ("sd3_5_large_turbo", false, false),
        // SD3.5 Medium (epic 7841 / sc-7869 M3, wired sc-7871): the MMDiT-X true-CFG variant —
        // supports_guidance=true + supports_negative_prompt=true (the `sd3_5_medium` descriptor advertises
        // supports_true_cfg, same as Large; only the transformer + step/guidance recipe differ).
        ("sd3_5_medium", true, true),
        // SANA 1600M (epic 8485 / sc-8489): true-CFG Linear-DiT — supports_guidance=true +
        // supports_negative_prompt=true (the `sana_1600m` descriptor advertises supports_true_cfg).
        ("sana_1600m", true, true),
        // SANA-Sprint 1.6B (epic 8485 / sc-8490): the CFG-free few-step distillation — the guidance
        // scalar is folded into the trunk via a guidance-embedding (supports_guidance=true) but there is
        // no cond/uncond combine, so the descriptor advertises supports_true_cfg=false +
        // supports_negative_prompt=false (the distilled-turbo "guidance is an embedding, no negative"
        // shape; cf. boogu_image_turbo's CFG-free-without-negative pattern).
        ("sana_sprint_1600m", true, false),
    ];
    // Every row is covered by the expectation table (no row added without a flag pair here).
    assert_eq!(MODEL_TABLE.len(), expected.len());
    for (id, guidance, negative) in expected {
        let m = mlx_model(id).unwrap_or_else(|| panic!("{id} resolves through the registry"));
        assert_eq!(
            m.supports_guidance(),
            *guidance,
            "{id} supports_guidance descriptor drift"
        );
        assert_eq!(
            m.supports_negative_prompt(),
            *negative,
            "{id} supports_negative_prompt descriptor drift"
        );
        assert_eq!(m.backend(), "mlx", "{id} backend");
    }
}

/// Parse the embedded `builtin.models.jsonc` manifest (the exact bytes shipped) and return its
/// `models` array. Shared by the manifest-gating catalog-eligibility tests so the ~15-line
/// load→strip-comments→parse→get-array boilerplate lives in ONE place (sc-8926 dedupe of the
/// near-verbatim copies that had drifted between the SD3.5 / SANA / SANA-Sprint gate tests).
fn builtin_models_manifest() -> Vec<Value> {
    use sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS;
    use sceneworks_core::jsonc::strip_jsonc_comments;

    let raw = BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .map(|(_, contents)| *contents)
        .expect("builtin.models.jsonc present");
    let manifest: Value =
        serde_json::from_str(&strip_jsonc_comments(raw)).expect("builtin models parses as JSON");
    manifest
        .get("models")
        .and_then(Value::as_array)
        .expect("models array")
        .clone()
}

/// The builtin-manifest entry for `id`, panicking with a clear hint if the id is absent. Shared by
/// the manifest-gating tests (sc-8926) so each gate test asserts on the entry directly instead of
/// re-implementing the manifest lookup.
fn builtin_model_entry(id: &str) -> Value {
    builtin_models_manifest()
        .into_iter()
        .find(|m| m.get("id").and_then(Value::as_str) == Some(id))
        .unwrap_or_else(|| panic!("{id} present in builtin.models.jsonc"))
}

/// sc-7875 (SD3.5 S6, MLX-path validation boundary): the three SD3.5 builtin-manifest entries gate
/// correctly at the catalog layer — `macOnly: false` (cross-platform now that the candle off-Mac lane
/// is wired, sc-7880/epic 7982; availability is driven by the routing tables, not this flag),
/// `capabilities == ["text_to_image"]` only (edit/reference rejected), the family `sd3`, the gated
/// stabilityai/* download with `gated: true` + `credentialHost: huggingface.co`, and the per-tier
/// `mlx.minMemoryGb` (Large/Turbo 64, Medium 56) that drives the memory-eligibility gate. Parses the
/// embedded builtin manifest (the exact bytes shipped) so manifest drift on any of these eligibility
/// levers fails CI without a real download. (The descriptor-derived guidance/negative/backend surface
/// is covered by `model_table_rows_resolve_and_flags_match_descriptor`; the credential-host derivation
/// by the rust-api `gated_credential_tests`; this is the catalog-eligibility counterpart.)
#[test]
fn sd3_5_manifest_entries_gate_correctly() {
    // (id, expected minMemoryGb) — Large/Turbo flagship-tier 64, Medium light-tier 56
    // (S6 worker-lane footprint ~52 GB Q8 / ~48.6 GB Q4 + headroom, below the 64 flagship tier).
    let expected: &[(&str, u64)] = &[
        ("sd3_5_large", 64),
        ("sd3_5_large_turbo", 64),
        ("sd3_5_medium", 56),
    ];
    for (id, min_mem) in expected {
        let entry = builtin_model_entry(id);

        assert_eq!(
            entry.get("family").and_then(Value::as_str),
            Some("sd3"),
            "{id} family"
        );
        // Cross-platform: the candle off-Mac lane is now wired (sc-7880, epic 7982), so `macOnly` is a
        // no-op label flipped to false (mirroring krea/flux2_dev) — availability is driven by the
        // routing tables (`MLX_ROUTED_MODELS` / `CANDLE_ROUTED_MODELS`), not this flag.
        assert_eq!(
            entry.get("macOnly").and_then(Value::as_bool),
            Some(false),
            "{id} macOnly"
        );
        // Capability gate: text_to_image ONLY — edit/reference are rejected (no img2img/inpaint path).
        let caps: Vec<&str> = entry
            .get("capabilities")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        assert_eq!(caps, vec!["text_to_image"], "{id} capabilities");
        // UN-gated SceneWorks MLX re-host (sc-8513, epic 8506): the pre-built quant-matrix turnkey
        // carries the Stability AI Community License + "Powered by Stability AI" NOTICE, so no HF
        // credential host / license-click. (Was the gated stabilityai/* source + install-time convert.)
        assert_eq!(
            entry.get("gated").and_then(Value::as_bool),
            Some(false),
            "{id} gated"
        );
        assert_eq!(
            entry.get("credentialHost"),
            None,
            "{id} credentialHost (dropped on re-host)"
        );
        // Every tier download is the SceneWorks re-host (q4 default + q8 + bf16), and the default
        // (first) entry is q4.
        let downloads = entry
            .get("downloads")
            .and_then(Value::as_array)
            .unwrap_or_else(|| panic!("{id} downloads array"));
        for dl in downloads {
            let repo = dl.get("repo").and_then(Value::as_str).unwrap_or("");
            assert!(
                repo.starts_with("SceneWorks/sd3.5-"),
                "{id} tier downloads from the SceneWorks re-host, got {repo:?}"
            );
        }
        let first_files: Vec<&str> = downloads
            .first()
            .and_then(|d| d.get("files"))
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        assert_eq!(first_files, vec!["q4/*"], "{id} default tier is q4");
        // Per-tier memory-eligibility gate (drives the Studio admit/hide-by-available-memory).
        assert_eq!(
            entry
                .get("mlx")
                .and_then(|m| m.get("minMemoryGb"))
                .and_then(Value::as_u64),
            Some(*min_mem),
            "{id} mlx.minMemoryGb"
        );
        // sd3 LoRA family declared (S5): the picker offers ONLY sd3-family LoRAs (an empty list would
        // match every LoRA, sc-1927).
        let lora_families: Vec<&str> = entry
            .get("loraCompatibility")
            .and_then(|c| c.get("families"))
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        assert_eq!(
            lora_families,
            vec!["sd3"],
            "{id} loraCompatibility.families"
        );
    }
}

/// sc-8489 (SANA Phase B2): the SANA builtin-manifest entry gates correctly at the catalog layer —
/// family `sana`, `capabilities == ["text_to_image"]` only (edit/reference rejected), the UN-gated
/// `SceneWorks/Sana_1600M_1024px_mlx` MLX re-host (NOT gated — the mirror carries the NVIDIA
/// non-commercial NOTICE), dense bf16 (NO `mlx.quantize` — the load path rejects a quant), the
/// `mlx.minMemoryGb` memory-eligibility lever, the sana LoRA family, and the NVIDIA non-commercial
/// notice surfaced in the UI description. Parses the embedded builtin manifest (the exact bytes
/// shipped) so manifest drift on any of these levers fails CI without a real download. The
/// descriptor-derived guidance/negative/backend surface is covered by
/// `model_table_rows_resolve_and_flags_match_descriptor`.
#[test]
fn sana_manifest_entry_gates_correctly() {
    let entry = builtin_model_entry("sana_1600m");

    assert_eq!(
        entry.get("family").and_then(Value::as_str),
        Some("sana"),
        "sana family"
    );
    // Capability gate: text_to_image ONLY — edit/reference are rejected (base SANA is plain t2i).
    let caps: Vec<&str> = entry
        .get("capabilities")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    assert_eq!(caps, vec!["text_to_image"], "sana capabilities");
    // UN-gated SceneWorks/* MLX re-host (the mirror carries the NVIDIA non-commercial NOTICE; OK to
    // ship un-gated with notice, the Krea/Boogu precedent) — so NO `gated: true`.
    assert_ne!(
        entry.get("gated").and_then(Value::as_bool),
        Some(true),
        "sana is un-gated (re-host carries the notice)"
    );
    let repo = entry
        .get("downloads")
        .and_then(Value::as_array)
        .and_then(|d| d.first())
        .and_then(|d| d.get("repo"))
        .and_then(Value::as_str)
        .unwrap_or("");
    assert_eq!(
        repo, "SceneWorks/Sana_1600M_1024px_mlx",
        "sana downloads from the un-gated SceneWorks/* MLX mirror, got {repo:?}"
    );
    // Quant matrix (sc-8489/sc-8513): the q4/q8/bf16 turnkey tiers are packed-detected on load, so the
    // descriptor advertises Q4/Q8 and the manifest defaults to the packed q4 tier (`mlx.quantize: 4`;
    // `standard_tier_subdir` resolves `q4/`). minMemoryGb drives the admit/hide gate.
    let mlx = entry.get("mlx").expect("sana mlx block");
    assert_eq!(
        mlx.get("quantize").and_then(Value::as_u64),
        Some(4),
        "sana defaults to the packed q4 tier"
    );
    // Per-tier variants present (q4 default + q8 + bf16), each its own installable artifact (sc-8508).
    let variants: Vec<&str> = entry
        .get("downloads")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|d| d.get("variant").and_then(Value::as_str))
                .collect()
        })
        .unwrap_or_default();
    assert_eq!(
        variants,
        vec!["q4", "q8", "bf16"],
        "sana ships the q4/q8/bf16 tier matrix"
    );
    assert!(
        mlx.get("minMemoryGb").and_then(Value::as_u64).is_some(),
        "sana mlx.minMemoryGb present"
    );
    // sana LoRA family declared (reserved; no SANA LoRA wired yet) — an empty list would match every
    // LoRA (sc-1927).
    let lora_families: Vec<&str> = entry
        .get("loraCompatibility")
        .and_then(|c| c.get("families"))
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    assert_eq!(
        lora_families,
        vec!["sana"],
        "sana loraCompatibility.families"
    );
    // NVIDIA non-commercial notice surfaced in the UI description (the gated-with-notice carrier).
    let desc = entry
        .get("ui")
        .and_then(|u| u.get("description"))
        .and_then(Value::as_str)
        .unwrap_or("");
    assert!(
        desc.contains("NVIDIA") && desc.to_lowercase().contains("non-commercial"),
        "sana UI description carries the NVIDIA non-commercial notice"
    );
}

/// sc-8490: the SANA-Sprint builtin entry gates exactly like base SANA — `sana` family, text_to_image
/// only (CFG-free few-step distillation, no edit/reference surface), un-gated `SceneWorks/*` MLX
/// re-host carrying the NVIDIA non-commercial notice, the q4/q8/bf16 quant matrix (default q4), and the
/// SANA LoRA family reserved. The few-step default (2 steps) is asserted so a manifest drift to the base 20-step
/// loop fails CI. Descriptor-derived guidance/negative/backend flags are covered by
/// `model_table_rows_resolve_and_flags_match_descriptor`.
#[test]
fn sana_sprint_manifest_entry_gates_correctly() {
    let entry = builtin_model_entry("sana_sprint_1600m");

    assert_eq!(
        entry.get("family").and_then(Value::as_str),
        Some("sana"),
        "sana-sprint family"
    );
    // Capability gate: text_to_image ONLY — edit/reference are rejected (Sprint is plain few-step t2i).
    let caps: Vec<&str> = entry
        .get("capabilities")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    assert_eq!(caps, vec!["text_to_image"], "sana-sprint capabilities");
    // UN-gated SceneWorks/* MLX re-host (the mirror carries the NVIDIA non-commercial NOTICE) — no `gated`.
    assert_ne!(
        entry.get("gated").and_then(Value::as_bool),
        Some(true),
        "sana-sprint is un-gated (re-host carries the notice)"
    );
    let repo = entry
        .get("downloads")
        .and_then(Value::as_array)
        .and_then(|d| d.first())
        .and_then(|d| d.get("repo"))
        .and_then(Value::as_str)
        .unwrap_or("");
    assert_eq!(
        repo, "SceneWorks/Sana_Sprint_1.6B_1024px_mlx",
        "sana-sprint downloads from the un-gated SceneWorks/* MLX mirror, got {repo:?}"
    );
    // Few-step distillation: default steps = 2 (a drift to base SANA's 20-step loop fails here).
    assert_eq!(
        entry
            .get("defaults")
            .and_then(|d| d.get("steps"))
            .and_then(Value::as_u64),
        Some(2),
        "sana-sprint is a few-step (2-step) distillation"
    );
    // Quant matrix (sc-8490/sc-8513): default q4 (`mlx.quantize: 4`); q4/q8/bf16 tiers packed-detected.
    let mlx = entry.get("mlx").expect("sana-sprint mlx block");
    assert_eq!(
        mlx.get("quantize").and_then(Value::as_u64),
        Some(4),
        "sana-sprint defaults to the packed q4 tier"
    );
    let variants: Vec<&str> = entry
        .get("downloads")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|d| d.get("variant").and_then(Value::as_str))
                .collect()
        })
        .unwrap_or_default();
    assert_eq!(
        variants,
        vec!["q4", "q8", "bf16"],
        "sana-sprint ships the q4/q8/bf16 tier matrix"
    );
    assert!(
        mlx.get("minMemoryGb").and_then(Value::as_u64).is_some(),
        "sana-sprint mlx.minMemoryGb present"
    );
    // sana LoRA family declared (reserved; no SANA LoRA wired yet).
    let lora_families: Vec<&str> = entry
        .get("loraCompatibility")
        .and_then(|c| c.get("families"))
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    assert_eq!(
        lora_families,
        vec!["sana"],
        "sana-sprint loraCompatibility.families"
    );
    // NVIDIA non-commercial notice surfaced in the UI description (the gated-with-notice carrier).
    let desc = entry
        .get("ui")
        .and_then(|u| u.get("description"))
        .and_then(Value::as_str)
        .unwrap_or("");
    assert!(
        desc.contains("NVIDIA") && desc.to_lowercase().contains("non-commercial"),
        "sana-sprint UI description carries the NVIDIA non-commercial notice"
    );
}

/// sc-9942 (epic 8506): the Wan2.2 T2V-A14B builtin entry ships the macOS quant matrix as per-tier
/// installable artifacts — `q4` (default) + `q8` + `bf16`, each a self-contained SceneWorks HF
/// download tagged with a `variant` and carrying an `estimatedSizeBytes` + `footprint` (sc-8508). The
/// video load path (`video_jobs::wan_a14b_tier_subdir`) descends into the chosen tier, so a drift
/// that drops a tier or its size would break the download UI + the RAM→tier suggestion — assert the
/// shape here so it fails CI.
#[test]
fn wan_t2v_14b_manifest_ships_the_quant_matrix() {
    let entry = builtin_model_entry("wan_2_2_t2v_14b");
    assert_eq!(
        entry.get("family").and_then(Value::as_str),
        Some("wan-video"),
        "wan T2V-14B family"
    );
    assert_eq!(
        entry.get("type").and_then(Value::as_str),
        Some("video"),
        "wan T2V-14B is a video model"
    );
    let downloads = entry
        .get("downloads")
        .and_then(Value::as_array)
        .expect("wan T2V-14B downloads");
    // The macOS tiers, in order, from the SceneWorks quant-matrix repo.
    let macos: Vec<&Value> = downloads
        .iter()
        .filter(|d| {
            d.get("platforms")
                .and_then(Value::as_array)
                .map(|p| p.iter().any(|x| x.as_str() == Some("macos")))
                .unwrap_or(false)
        })
        .collect();
    let variants: Vec<&str> = macos
        .iter()
        .filter_map(|d| d.get("variant").and_then(Value::as_str))
        .collect();
    assert_eq!(
        variants,
        vec!["q4", "q8", "bf16"],
        "wan T2V-14B ships the q4/q8/bf16 tier matrix on macOS"
    );
    for tier in &macos {
        let variant = tier.get("variant").and_then(Value::as_str).unwrap();
        assert_eq!(
            tier.get("repo").and_then(Value::as_str),
            Some("SceneWorks/wan2.2-t2v-a14b-mlx"),
            "{variant} tier hosts on the SceneWorks quant-matrix repo"
        );
        let files: Vec<String> = tier
            .get("files")
            .and_then(Value::as_array)
            .map(|f| f.iter().filter_map(Value::as_str).map(String::from).collect())
            .unwrap_or_default();
        assert_eq!(
            files,
            vec![format!("{variant}/*")],
            "{variant} tier installs only its own subdir"
        );
        assert!(
            tier.get("estimatedSizeBytes")
                .and_then(Value::as_u64)
                .is_some_and(|b| b > 0),
            "{variant} tier declares a nonzero estimatedSizeBytes"
        );
        assert!(
            tier.get("footprint")
                .and_then(|f| f.get("diskSizeBytes"))
                .and_then(Value::as_u64)
                .is_some_and(|b| b > 0),
            "{variant} tier declares a footprint.diskSizeBytes"
        );
    }
    // Exactly one macOS default, and it is the lean q4 tier (the big bf16 is a deliberate opt-in).
    let defaults: Vec<&str> = macos
        .iter()
        .filter(|d| d.get("default").and_then(Value::as_bool) == Some(true))
        .filter_map(|d| d.get("variant").and_then(Value::as_str))
        .collect();
    assert_eq!(defaults, vec!["q4"], "the macOS default tier is q4");
}

/// sc-9943 (epic 8506): the Wan2.2 I2V-A14B builtin entry ships the SAME macOS quant matrix as its
/// T2V sibling — per-tier `q4` (default) + `q8` + `bf16` self-contained SceneWorks HF downloads
/// tagged with a `variant`, each with an `estimatedSizeBytes` + `footprint` (sc-8508). The video load
/// path (`video_jobs::wan_a14b_tier_subdir`) descends into the chosen tier for BOTH A14B models, so a
/// drift that drops an I2V tier or its size would break the download UI + the RAM→tier suggestion —
/// assert the shape here so it fails CI.
#[test]
fn wan_i2v_14b_manifest_ships_the_quant_matrix() {
    let entry = builtin_model_entry("wan_2_2_i2v_14b");
    assert_eq!(
        entry.get("family").and_then(Value::as_str),
        Some("wan-video"),
        "wan I2V-14B family"
    );
    assert_eq!(
        entry.get("type").and_then(Value::as_str),
        Some("video"),
        "wan I2V-14B is a video model"
    );
    let downloads = entry
        .get("downloads")
        .and_then(Value::as_array)
        .expect("wan I2V-14B downloads");
    // The macOS tiers, in order, from the SceneWorks quant-matrix repo.
    let macos: Vec<&Value> = downloads
        .iter()
        .filter(|d| {
            d.get("platforms")
                .and_then(Value::as_array)
                .map(|p| p.iter().any(|x| x.as_str() == Some("macos")))
                .unwrap_or(false)
        })
        .collect();
    let variants: Vec<&str> = macos
        .iter()
        .filter_map(|d| d.get("variant").and_then(Value::as_str))
        .collect();
    assert_eq!(
        variants,
        vec!["q4", "q8", "bf16"],
        "wan I2V-14B ships the q4/q8/bf16 tier matrix on macOS"
    );
    for tier in &macos {
        let variant = tier.get("variant").and_then(Value::as_str).unwrap();
        assert_eq!(
            tier.get("repo").and_then(Value::as_str),
            Some("SceneWorks/wan2.2-i2v-a14b-mlx"),
            "{variant} tier hosts on the SceneWorks I2V quant-matrix repo"
        );
        let files: Vec<String> = tier
            .get("files")
            .and_then(Value::as_array)
            .map(|f| f.iter().filter_map(Value::as_str).map(String::from).collect())
            .unwrap_or_default();
        assert_eq!(
            files,
            vec![format!("{variant}/*")],
            "{variant} tier installs only its own subdir"
        );
        assert!(
            tier.get("estimatedSizeBytes")
                .and_then(Value::as_u64)
                .is_some_and(|b| b > 0),
            "{variant} tier declares a nonzero estimatedSizeBytes"
        );
        assert!(
            tier.get("footprint")
                .and_then(|f| f.get("diskSizeBytes"))
                .and_then(Value::as_u64)
                .is_some_and(|b| b > 0),
            "{variant} tier declares a footprint.diskSizeBytes"
        );
    }
    // Exactly one macOS default, and it is the lean q4 tier (the big bf16 is a deliberate opt-in).
    let defaults: Vec<&str> = macos
        .iter()
        .filter(|d| d.get("default").and_then(Value::as_bool) == Some(true))
        .filter_map(|d| d.get("variant").and_then(Value::as_str))
        .collect();
    assert_eq!(defaults, vec!["q4"], "the macOS default tier is q4");
}
