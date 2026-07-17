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

/// sc-11042 (epic 11037): the NVFP4 tier's Blackwell gate floors at compute cap **12.0** (consumer
/// Blackwell sm_120 — the FP4 tensor cores the cuBLASLt NVFP4 GEMM dispatches on).
///
/// This is the ONLY place hardware maps onto NVFP4 eligibility, so the boundary is pinned here: the
/// tier-select tests inject the resulting bool rather than probing a live GPU.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
fn nvfp4_gate_floors_at_blackwell_sm_120() {
    use crate::gpu::compute_cap_meets_nvfp4;
    // Consumer Blackwell (RTX PRO 6000 / RTX 50-series = 12.0) — the epic's target, and eligible.
    assert!(compute_cap_meets_nvfp4(Some(12.0)));
    // A hypothetical future cap above the floor stays eligible (a floor, matching the ConvRot shape).
    assert!(compute_cap_meets_nvfp4(Some(12.8)));
    // Ada (8.9) and Ampere (8.6) have no FP4 tensor cores → ineligible, and fall back cleanly.
    assert!(!compute_cap_meets_nvfp4(Some(8.9)));
    assert!(!compute_cap_meets_nvfp4(Some(8.6)));
    // DATACENTER Blackwell sm_100 (B100/B200) probes 10.0. It HAS FP4 hardware but is explicitly out of
    // scope for epic 11037 — the lane is neither built for it (`CUDA_COMPUTE_CAP=120` emits sm_120 SASS
    // + compute_120 PTX) nor validated on it — so the 12.0 floor keeps it off the tier BY CONSTRUCTION.
    assert!(!compute_cap_meets_nvfp4(Some(10.0)));
    // No probe (nvidia-smi absent / CPU / non-NVIDIA) ⇒ ineligible. Fail-safe: an unprobed host must
    // never route an FP4 load at hardware that may not have FP4 tensor cores.
    assert!(!compute_cap_meets_nvfp4(None));
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
        // sc-5552: runtime-macos includes the native MLX `prompt_refine` provider, so the registry
        // derives PromptRefine from its descriptor.
        WorkerCapability::PromptRefine,
        // sc-6535: runtime-macos includes the CLIP `clip_vit_l14` ImageEmbedder, so the registry
        // derives DatasetAnalysis from its descriptor.
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
        // Illustrious-XL v1.0 / v2.0 (epic 10609): plain SDXL engine, real CFG + negative prompt.
        ("illustrious_xl_v1", true, true),
        ("illustrious_xl_v2", true, true),
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
        // Krea 2 Raw (epic 9992): the UNDISTILLED base run with TRUE CFG — supports_guidance=true +
        // supports_negative_prompt=true (unlike the CFG-free distilled Turbo). The Boogu-base pattern,
        // but Raw ALSO accepts a user negative prompt (the reference `sample()` takes `negative_prompts`).
        ("krea_2_raw", true, true),
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
        // Anima 2B anime t2i (epic 10512 / sc-10523): Base + Aesthetic run true classifier-free
        // guidance — the descriptor derives supports_guidance = supports_negative_prompt = `uses_cfg()`,
        // which is true for both (mlx-gen-anima `descriptor_for` / `Variant::uses_cfg`). Turbo is the
        // merged CFG-free few-step student (`uses_cfg() == false`), so its descriptor drops both flags.
        ("anima_base", true, true),
        ("anima_aesthetic", true, true),
        ("anima_turbo", false, false),
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

/// epic 10765 sc-10920: the FLUX.2 `candle` blocks drive the dynamic fit-gate end-to-end against the
/// SHIPPED manifest bytes — resident-fits → sequential-offload → reject-before-OOM. The pure
/// `vram_gate` unit tests cover the decision logic on synthetic fixtures; THIS guards the data half:
/// a manifest edit that drops the flux2 `vramGbByTier` / `sequentialPeakGb` makes `predicted_*` return
/// `None` → the whole gate goes INERT for flux2 (the exact "candle block missing" failure the story
/// targets). Exercises the same `SCENEWORKS_CUDA_VRAM_CAP_GB` small-card emulation the worker honors via
/// `apply_vram_cap`. Gated to the candle lane where `vram_gate` compiles (sc-10920 measured q4/q8;
/// bf16 + klein are carried, and are asserted to reject on real cards, which is the intended outcome).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
fn flux2_candle_blocks_drive_the_fit_gate_and_reject() {
    use crate::vram_gate::{
        apply_vram_cap, fit_decision, predicted_peak_gb, predicted_sequential_peak_gb,
        resolve_offload, sequential_overflow_gb, FitDecision,
    };

    let dev = builtin_model_entry("flux2_dev");
    // `predicted_*` take the MODEL ENTRY (they read `.candle` internally), not the candle sub-object.
    let dev_entry = dev.as_object().expect("flux2_dev entry object");
    assert!(
        dev_entry.get("candle").and_then(Value::as_object).is_some(),
        "flux2_dev candle block present (absent ⇒ fit-gate inert for flux2)"
    );

    // Measured q4 (sc-10920): resident 44.0 / sequential 35.6, each + the gate's 2 GB headroom.
    let q4_res = predicted_peak_gb(dev_entry, "q4").expect("q4 resident predicted");
    let q4_seq = predicted_sequential_peak_gb(dev_entry, "q4").expect("q4 sequential predicted");
    assert!((q4_res - 46.0).abs() < 1e-6, "q4 resident 44.0 + 2 headroom, got {q4_res}");
    assert!((q4_seq - 37.6).abs() < 1e-6, "q4 sequential 35.6 + 2 headroom, got {q4_seq}");
    assert!(q4_seq < q4_res, "sequential is the lower peak");

    // A 96 GB card fits q4 resident outright — no offload.
    let card96 = apply_vram_cap(None, Some(96.0));
    assert_eq!(fit_decision(Some(q4_res), card96), FitDecision::Fits);

    // Emulate a 40 GB card: resident 46 won't fit, but sequential 37.6 does → OFFLOAD, run sequentially.
    let card40 = apply_vram_cap(None, Some(40.0));
    let at40 = resolve_offload(fit_decision(Some(q4_res), card40), /* sequential_capable */ true);
    assert!(matches!(at40, FitDecision::Offload { .. }), "40 GB → offload, got {at40:?}");
    assert_eq!(sequential_overflow_gb(Some(q4_seq), card40), None, "sequential fits 40 GB → run");

    // Emulate a 30 GB card: even the sequential 37.6 peak won't fit → REJECT-before-OOM (sc-10856 gate).
    let card30 = apply_vram_cap(None, Some(30.0));
    let at30 = resolve_offload(fit_decision(Some(q4_res), card30), true);
    assert!(matches!(at30, FitDecision::Offload { .. }), "30 GB → offload attempt, got {at30:?}");
    assert_eq!(
        sequential_overflow_gb(Some(q4_seq), card30),
        Some(q4_seq),
        "sequential 37.6 > 30 GB → reject carrying the number"
    );

    // Measured q8 is present too (resident 70.7 / sequential 64.9 + headroom).
    assert!((predicted_peak_gb(dev_entry, "q8").unwrap() - 72.7).abs() < 1e-6);
    assert!((predicted_sequential_peak_gb(dev_entry, "q8").unwrap() - 66.9).abs() < 1e-6);

    // The carried bf16 tier (128 / 97) rejects on a 96 GB card even sequentially — the intended outcome
    // (113 GB dense weights can't run off-Mac), reject-before-OOM instead of a silent load-time OOM.
    let bf16_seq = predicted_sequential_peak_gb(dev_entry, "bf16").expect("bf16 sequential carried");
    assert_eq!(
        sequential_overflow_gb(Some(bf16_seq), card96),
        Some(bf16_seq),
        "bf16 sequential 99 > 96 GB → reject"
    );

    // klein carries a candle block so ITS fit-gate is live: on a 16 GB epic-target card klein rejects
    // even sequentially (the ~32 GB F32-resident Qwen3 TE is the sequential floor). q4 sequential is the
    // MEASURED ~24 GB (sc-11031 klein DiT-quant A/B) — well over the 16 GB card either way.
    let klein = builtin_model_entry("flux2_klein_9b");
    let klein_entry = klein.as_object().expect("flux2_klein_9b entry object");
    assert!(
        klein_entry.get("candle").and_then(Value::as_object).is_some(),
        "flux2_klein_9b candle block present"
    );
    let klein_seq =
        predicted_sequential_peak_gb(klein_entry, "q4").expect("klein q4 sequential carried");
    let card16 = apply_vram_cap(None, Some(16.0));
    assert_eq!(
        sequential_overflow_gb(Some(klein_seq), card16),
        Some(klein_seq),
        "klein sequential ~24 > 16 GB → reject on the epic's small-card target"
    );
}

/// sc-12130/sc-12131: the shipped Krea base block feeds both stages of the generic Candle gate. This
/// pins the provider-derived capability handoff and the measured `sequentialPeakGb` values that turn a
/// small-card staged attempt into a clean reject instead of relying on reactive OOM containment.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
fn krea_candle_block_drives_the_registry_and_second_stage_gate() {
    use crate::vram_gate::{
        apply_vram_cap, fit_decision, predicted_peak_gb, predicted_sequential_peak_gb,
        resolve_offload, sequential_overflow_gb, FitDecision,
    };

    let krea = builtin_model_entry("krea_2_turbo");
    let entry = krea.as_object().expect("krea_2_turbo entry object");
    assert_eq!(
        entry
            .get("candle")
            .and_then(Value::as_object)
            .and_then(|candle| candle.get("measured"))
            .and_then(Value::as_bool),
        Some(true),
        "the published Krea base resident + sequential rows are measured"
    );
    assert!(
        crate::mlx_fit_gate::engine_supports_sequential("krea_2_turbo"),
        "the generic Candle gate must derive Krea support from the registered descriptor"
    );

    // Manifest values plus the generic gate's fixed 2 GB runtime headroom.
    let q4_resident = predicted_peak_gb(entry, "q4").expect("q4 resident peak");
    let q4_sequential =
        predicted_sequential_peak_gb(entry, "q4").expect("q4 sequential peak");
    assert!((q4_resident - 28.4).abs() < 1e-6);
    assert!((q4_sequential - 24.7).abs() < 1e-6);
    assert!((predicted_sequential_peak_gb(entry, "q8").unwrap() - 31.5).abs() < 1e-6);
    assert!((predicted_sequential_peak_gb(entry, "bf16").unwrap() - 41.8).abs() < 1e-6);
    // sc-12425: int8-convrot NOW has a measured sequential peak (28.6 + 2.0 headroom = 30.6). It was
    // `None` because the ConvRot lane was pinned Resident; inference PR #67 makes it drop the f32 TE like
    // every other Turbo request. 30.6 sits next to q8's 31.5 — once the shared TE drops, the same-size
    // int8 DiT costs the same as q8. GATED: this PR must not merge before the runtime pin honors that
    // sequential lane (else the worker predicts 30.6 while the runtime runs the 42.9 resident peak).
    assert!((predicted_sequential_peak_gb(entry, "int8-convrot").unwrap() - 30.6).abs() < 1e-6);

    // A 27 GB card cannot hold resident q4 but can run staged, so the registry bit selects Offload.
    let card27 = apply_vram_cap(None, Some(27.0));
    assert!(matches!(
        resolve_offload(fit_decision(Some(q4_resident), card27), true),
        FitDecision::Offload { .. }
    ));
    assert_eq!(sequential_overflow_gb(Some(q4_sequential), card27), None);

    // A 12 GB card cannot hold even the measured staged working set: carry the honest number into the
    // worker's second-stage reject-before-load path instead of attempting a process-killing allocation.
    let card12 = apply_vram_cap(None, Some(12.0));
    assert_eq!(
        sequential_overflow_gb(Some(q4_sequential), card12),
        Some(24.7)
    );
}

/// sc-11754 + sc-11744 (epic 8459 → epic 10765): the Krea 2 Turbo `candle.control` block drives the
/// pose-ControlNet VRAM fit LADDER end-to-end against the SHIPPED manifest bytes. The control-lane sibling
/// of `flux2_candle_blocks_drive_the_fit_gate_and_reject`: guards the DATA half — dropping the `control`
/// block makes `predicted_control_peak_gb` return `None` → the control fit-gate goes INERT (bf16 branch
/// always, the pre-sc-11754 behavior). Exercises the `SCENEWORKS_CUDA_VRAM_CAP_GB` small-card emulation via
/// `apply_vram_cap` across the cost-ordered rungs: big card → resident untiled bf16 branch (no penalty);
/// sequential residency (sc-12176) first; then VAE tiling, chunking, branch quant, and reject-before-OOM.
/// Expectations are expressed relative to the
/// SHIPPED deltas (a manifest edit that drifts `decodeTileSaveGb` / `branchQuantSaveGb` fails here) rather
/// than hard-coded arithmetic.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
fn krea_control_candle_block_drives_the_fit_ladder() {
    use crate::krea_control_fit::{
        branch_quant_save_gb, chunk_attn_save_gb, decode_tile_save_gb, fit_ladder,
        predicted_control_peak_gb, predicted_control_sequential_peak_gb, KreaControlFit,
    };
    use crate::vram_gate::apply_vram_cap;
    use gen_core::{OffloadPolicy, Quant};

    let krea = builtin_model_entry("krea_2_turbo");
    let entry = krea.as_object().expect("krea_2_turbo entry object");
    let candle = entry
        .get("candle")
        .and_then(Value::as_object)
        .expect("krea_2_turbo candle block");
    assert!(
        candle.get("control").and_then(Value::as_object).is_some(),
        "krea_2_turbo candle.control block present (absent ⇒ control fit-gate inert → always bf16 branch)"
    );

    // Shipped MEASURED branch-quant deltas (sc-11743): q8 −8.4 (near-lossless), q4 −10.2 (pose-locked).
    let q8 = branch_quant_save_gb(entry, "q8");
    let q4 = branch_quant_save_gb(entry, "q4");
    assert_eq!(q8, Some(8.4));
    assert_eq!(q4, Some(10.2));

    // Shipped MEASURED activation-chunking delta (sc-11745): the denoise-peak rung between tiling and
    // branch-quant — a SCALAR (tier-independent), speed-only, byte-identical output.
    let chunk = chunk_attn_save_gb(entry);
    let chunk_save = chunk.expect("chunkAttnSaveGb shipped (the sc-11745 chunking rung)");
    assert!(chunk_save > 0.0, "chunking must recover VRAM, got {chunk_save}");

    // The common small-card install: q4 BASE tier. Resident and Sequential are both measured; every
    // deeper rung composes from the Sequential baseline because residency is the cheapest adaptation.
    let q4_peak = predicted_control_peak_gb(entry, "q4");
    let peak = q4_peak.expect("q4 control peak");
    assert!((peak - 38.2).abs() < 1e-6);
    let q4_sequential = predicted_control_sequential_peak_gb(entry, "q4");
    let sequential_peak = q4_sequential.expect("q4 sequential control peak");
    assert!((sequential_peak - 34.5).abs() < 1e-6);
    let tile = decode_tile_save_gb(entry, "q4");
    let tile_save = tile.expect("q4 decodeTileSaveGb shipped (the sc-11744 tiling rung)");
    assert!(tile_save > 0.0, "tiling must recover VRAM, got {tile_save}");
    let tiled_peak = sequential_peak - tile_save;
    let chunked_peak = tiled_peak - chunk_save;

    // 96 GB card: the monolithic peak fits outright — nothing engages, untiled full-speed bf16 branch.
    assert_eq!(
        fit_ladder(
            q4_peak,
            q4_sequential,
            apply_vram_cap(None, Some(96.0)),
            tile,
            chunk,
            q8,
            q4,
        ),
        KreaControlFit::Fits {
            offload_policy: OffloadPolicy::Resident,
            tile_vae_decode: false,
            chunk_attention: false,
            branch_quant: None,
        }
    );
    // A card that fits the measured Sequential peak but not Resident engages residency only.
    assert_eq!(
        fit_ladder(
            q4_peak,
            q4_sequential,
            apply_vram_cap(None, Some(sequential_peak)),
            tile,
            chunk,
            q8,
            q4,
        ),
        KreaControlFit::Fits {
            offload_policy: OffloadPolicy::Sequential,
            tile_vae_decode: false,
            chunk_attention: false,
            branch_quant: None,
        }
    );
    // A card at the staged+tiled peak engages Sequential then VAE tiling, keeping the bf16 branch.
    assert_eq!(
        fit_ladder(
            q4_peak,
            q4_sequential,
            apply_vram_cap(None, Some(tiled_peak)),
            tile,
            chunk,
            q8,
            q4,
        ),
        KreaControlFit::Fits {
            offload_policy: OffloadPolicy::Sequential,
            tile_vae_decode: true,
            chunk_attention: false,
            branch_quant: None,
        }
    );
    // Just below the tiled peak: tiling alone no longer fits, but the next speed-only rung (chunking) does
    // (chunked_peak fits) ⇒ tiling + chunking, STILL a bf16 branch — sc-11745's win over dropping to q8.
    assert_eq!(
        fit_ladder(
            q4_peak,
            q4_sequential,
            apply_vram_cap(None, Some(tiled_peak - 0.5)),
            tile,
            chunk,
            q8,
            q4,
        ),
        KreaControlFit::Fits {
            offload_policy: OffloadPolicy::Sequential,
            tile_vae_decode: true,
            chunk_attention: true,
            branch_quant: None,
        }
    );
    // Just below the chunked peak: both speed rungs stay on and q8 composes (chunked_peak − 8.4 fits) ⇒
    // tiling + chunking + q8, near-lossless — a shallower quant than the chunk-less ladder would have taken.
    assert_eq!(
        fit_ladder(
            q4_peak,
            q4_sequential,
            apply_vram_cap(None, Some(chunked_peak - 0.5)),
            tile,
            chunk,
            q8,
            q4,
        ),
        KreaControlFit::Fits {
            offload_policy: OffloadPolicy::Sequential,
            tile_vae_decode: true,
            chunk_attention: true,
            branch_quant: Some(Quant::Q8),
        }
    );
    // A card between (…+q8) and (…+q4): tiling + chunking + q4 (chunked_peak − 10.2) fits ⇒ the deepest rung.
    assert_eq!(
        fit_ladder(
            q4_peak,
            q4_sequential,
            apply_vram_cap(None, Some(chunked_peak - 8.9)),
            tile,
            chunk,
            q8,
            q4,
        ),
        KreaControlFit::Fits {
            offload_policy: OffloadPolicy::Sequential,
            tile_vae_decode: true,
            chunk_attention: true,
            branch_quant: Some(Quant::Q4),
        }
    );
    // A card below even (tiling + chunking + q4) ⇒ reject-before-OOM at the best-case peak.
    match fit_ladder(
        q4_peak,
        q4_sequential,
        apply_vram_cap(None, Some(chunked_peak - 11.0)),
        tile,
        chunk,
        q8,
        q4,
    ) {
        KreaControlFit::TooBig { needed_gb, .. } => {
            assert!(
                (needed_gb - (chunked_peak - 10.2)).abs() < 1e-6,
                "best-case (tiling + chunking + q4) peak, got {needed_gb}"
            );
        }
        other => panic!("below tiling+chunking+q4 → reject, got {other:?}"),
    }

    // The bf16 BASE tier carries no tiling saving (its denoise
    // steady-state, not the decode, is the peak), but the SCALAR chunking rung applies to every tier: a
    // 41 GB card engages chunking (speed-only) then q8 (48.2 − chunk_save − 8.4 ≤ 41), the near-lossless
    // preference before q4 — a shallower quant than the chunk-less walk (which took bare q8 at 39.8).
    let bf16_peak = predicted_control_peak_gb(entry, "bf16");
    assert!((bf16_peak.expect("bf16 control peak") - 67.8).abs() < 1e-6);
    let bf16_sequential = predicted_control_sequential_peak_gb(entry, "bf16");
    assert!((bf16_sequential.expect("bf16 sequential control peak") - 52.0).abs() < 1e-6);
    assert_eq!(decode_tile_save_gb(entry, "bf16"), None);
    assert_eq!(
        fit_ladder(
            bf16_peak,
            bf16_sequential,
            apply_vram_cap(None, Some(42.0)),
            decode_tile_save_gb(entry, "bf16"),
            chunk,
            q8,
            q4
        ),
        KreaControlFit::Fits {
            offload_policy: OffloadPolicy::Sequential,
            tile_vae_decode: false,
            chunk_attention: true,
            branch_quant: Some(Quant::Q8),
        }
    );
}

/// Live real-hardware validation (sc-11754): the REAL `nvidia-smi` VRAM reading on GPU 0 + the cap →
/// predict → ladder chain against the SHIPPED krea_2_turbo control block — the one piece the pure/data
/// tests can't cover (they use a synthetic budget). Mirrors `vram_gate`'s
/// `live_cuda_budget_drives_a_real_fit_decision`. Ignored by default (needs a CUDA GPU); run on the 96 GB
/// box with `cargo test -p sceneworks-worker --lib --features backend-candle -- --ignored --nocapture
/// krea_control_live`.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[tokio::test]
#[ignore]
async fn krea_control_live_ladder_on_a_real_card() {
    use crate::krea_control_fit::{
        branch_quant_save_gb, chunk_attn_save_gb, decode_tile_save_gb, fit_ladder,
        predicted_control_peak_gb, predicted_control_sequential_peak_gb, KreaControlFit,
    };
    use crate::vram_gate::apply_vram_cap;
    use gen_core::OffloadPolicy;

    let krea = builtin_model_entry("krea_2_turbo");
    let entry = krea.as_object().expect("krea_2_turbo entry object");
    let tile = decode_tile_save_gb(entry, "q4");
    let chunk = chunk_attn_save_gb(entry);
    let q8 = branch_quant_save_gb(entry, "q8");
    let q4 = branch_quant_save_gb(entry, "q4");
    // The common small-card install: q4 base tier.
    let peak = predicted_control_peak_gb(entry, "q4");
    let sequential = predicted_control_sequential_peak_gb(entry, "q4");
    let tiled_peak = sequential.unwrap() - tile.expect("q4 decodeTileSaveGb shipped");
    let chunked_peak = tiled_peak - chunk.expect("chunkAttnSaveGb shipped");

    let real = crate::gpu::nvidia_vram_budget_gb("0")
        .await
        .expect("GPU 0 should report a live VRAM budget on a CUDA box");
    eprintln!("live CUDA budget GPU0: {real:?}");

    // Uncapped real 96 GB card → untiled monolithic decode, unchunked, bf16 branch, no rung engages.
    assert_eq!(
        fit_ladder(
            peak,
            sequential,
            apply_vram_cap(Some(real), None),
            tile,
            chunk,
            q8,
            q4,
        ),
        KreaControlFit::Fits {
            offload_policy: OffloadPolicy::Resident,
            tile_vae_decode: false,
            chunk_attention: false,
            branch_quant: None,
        },
        "uncapped 96 GB card keeps the untiled bf16 branch"
    );
    // Cap just at the tiled peak (off the REAL reading) → residency and the first speed-cost rung engage:
    // sequential + tiling, with the bf16 branch kept (no quality penalty).
    assert_eq!(
        fit_ladder(
            peak,
            sequential,
            apply_vram_cap(Some(real), Some(tiled_peak)),
            tile,
            chunk,
            q8,
            q4,
        ),
        KreaControlFit::Fits {
            offload_policy: OffloadPolicy::Sequential,
            tile_vae_decode: true,
            chunk_attention: false,
            branch_quant: None,
        },
        "cap at the tiled peak → VAE tiling keeps the bf16 branch"
    );
    // Cap below (tiling + chunking + q8) off the REAL reading → all three cheaper rungs on + q4 to fit,
    // where the old tiling-only ladder rejected-before-OOM.
    assert_eq!(
        fit_ladder(
            peak,
            sequential,
            apply_vram_cap(Some(real), Some(chunked_peak - 8.9)),
            tile,
            chunk,
            q8,
            q4,
        ),
        KreaControlFit::Fits {
            offload_policy: OffloadPolicy::Sequential,
            tile_vae_decode: true,
            chunk_attention: true,
            branch_quant: Some(gen_core::Quant::Q4),
        },
        "cap below tiling+chunking+q8 → tiling + chunking + q4 fits"
    );
    eprintln!(
        "krea control fit ladder on a real card: 96→untiled bf16, tiled-peak→tiling, \
         below-chunk-peak→tiling+chunking, deep→+q4 ✓"
    );
}

/// epic 10765 sc-11019: the Qwen-Image-Edit `candle` blocks drive the EDIT fit-gate (qwen_edit_candle.rs,
/// sc-10968) end-to-end against the SHIPPED manifest bytes — resident-fits → sequential-offload →
/// reject-before-OOM. The edit sibling of `flux2_candle_blocks_drive_the_fit_gate_and_reject`: guards the
/// DATA half — a manifest edit dropping the qwen-edit `vramGbByTier` / `sequentialPeakGb` makes
/// `predicted_*` return `None` → the edit gate goes INERT (the exact "candle block missing" failure the
/// story targets). The edit lane is unconditionally sequential-capable (sc-10968 wired
/// `QwenEdit::generate_sequential`), so this exercises `resolve_offload` with `sequential_capable=true`.
/// Both the base entry (measured q4/q8/bf16) and the Lightning entry (the same conservative base numbers,
/// `measured:false`) are asserted to carry a candle block so BOTH edit gates are live.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
fn qwen_edit_candle_blocks_drive_the_fit_gate_and_reject() {
    use crate::vram_gate::{
        apply_vram_cap, fit_decision, predicted_peak_gb, predicted_sequential_peak_gb,
        resolve_offload, sequential_overflow_gb, FitDecision,
    };

    let edit = builtin_model_entry("qwen_image_edit_2511");
    // `predicted_*` take the MODEL ENTRY (they read `.candle` internally), not the candle sub-object.
    let edit_entry = edit.as_object().expect("qwen_image_edit_2511 entry object");
    assert!(
        edit_entry.get("candle").and_then(Value::as_object).is_some(),
        "qwen_image_edit_2511 candle block present (absent ⇒ edit fit-gate inert)"
    );

    // Measured q4 (sc-10968): resident 56.7 / sequential 36.9, each + the gate's 2 GB headroom.
    let q4_res = predicted_peak_gb(edit_entry, "q4").expect("q4 resident predicted");
    let q4_seq = predicted_sequential_peak_gb(edit_entry, "q4").expect("q4 sequential predicted");
    assert!(
        (q4_res - 58.7).abs() < 1e-6,
        "q4 resident 56.7 + 2 headroom, got {q4_res}"
    );
    assert!(
        (q4_seq - 38.9).abs() < 1e-6,
        "q4 sequential 36.9 + 2 headroom, got {q4_seq}"
    );
    assert!(q4_seq < q4_res, "sequential is the lower peak");

    // A 96 GB card fits q4 resident outright — no offload.
    let card96 = apply_vram_cap(None, Some(96.0));
    assert_eq!(fit_decision(Some(q4_res), card96), FitDecision::Fits);

    // A 48 GB card: resident 58.7 won't fit, but sequential 38.9 does → OFFLOAD, run sequentially.
    let card48 = apply_vram_cap(None, Some(48.0));
    let at48 = resolve_offload(fit_decision(Some(q4_res), card48), true);
    assert!(matches!(at48, FitDecision::Offload { .. }), "48 GB → offload");
    assert_eq!(
        sequential_overflow_gb(Some(q4_seq), card48),
        None,
        "sequential 38.9 fits 48 GB → run"
    );

    // A 30 GB card: even the sequential 38.9 peak won't fit → REJECT-before-OOM (sc-10856 gate).
    let card30 = apply_vram_cap(None, Some(30.0));
    let at30 = resolve_offload(fit_decision(Some(q4_res), card30), true);
    assert!(matches!(at30, FitDecision::Offload { .. }), "30 GB → offload attempt");
    assert_eq!(
        sequential_overflow_gb(Some(q4_seq), card30),
        Some(q4_seq),
        "sequential 38.9 > 30 GB → reject carrying the number"
    );

    // Measured q8 (69.0 / 39.3) + bf16 (81.7 / 52.2) present too, each + headroom.
    assert!((predicted_peak_gb(edit_entry, "q8").unwrap() - 71.0).abs() < 1e-6);
    assert!((predicted_sequential_peak_gb(edit_entry, "q8").unwrap() - 41.3).abs() < 1e-6);
    let bf16_res = predicted_peak_gb(edit_entry, "bf16").expect("bf16 resident");
    let bf16_seq = predicted_sequential_peak_gb(edit_entry, "bf16").expect("bf16 seq");
    assert!(
        (bf16_res - 83.7).abs() < 1e-6,
        "bf16 resident 81.7 + 2 headroom, got {bf16_res}"
    );
    assert!(
        (bf16_seq - 54.2).abs() < 1e-6,
        "bf16 sequential 52.2 + 2 headroom, got {bf16_seq}"
    );
    // Dense bf16 fits a 96 GB card resident, but on a 48 GB card even sequential overflows → reject.
    assert_eq!(fit_decision(Some(bf16_res), card96), FitDecision::Fits);
    assert_eq!(
        sequential_overflow_gb(Some(bf16_seq), card48),
        Some(bf16_seq),
        "bf16 sequential 54.2 > 48 GB → reject on the smaller card"
    );

    // The Lightning entry carries its own MEASURED additive-path block (sc-11666) so ITS edit gate is
    // live too — q4 additive sequential 33.1 + 2 headroom = 35.1 > 30 → reject on the small card.
    let lightning = builtin_model_entry("qwen_image_edit_2511_lightning");
    let lightning_entry = lightning.as_object().expect("lightning entry object");
    assert!(
        lightning_entry.get("candle").and_then(Value::as_object).is_some(),
        "qwen_image_edit_2511_lightning candle block present"
    );
    let l_seq = predicted_sequential_peak_gb(lightning_entry, "q4").expect("lightning q4 seq");
    assert_eq!(
        sequential_overflow_gb(Some(l_seq), card30),
        Some(l_seq),
        "lightning sequential 35.1 > 30 GB → reject (gate live for the distill entry too)"
    );
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

/// sc-9946 (epic 8506): the Kolors builtin entry ships the standard q4/q8/bf16 quant matrix from the
/// un-gated `SceneWorks/kolors-mlx` re-host (was upstream `Kwai-Kolors/Kolors-diffusers` dense +
/// install-time quant). Unlike SANA, Kolors keeps its full capability surface. Asserts the flip to the
/// SceneWorks repo, the per-tier variants (q4 default + q8 + bf16) each an installable artifact with a
/// `footprint`, `mlx.quantize: 4` (packed q4 default) + `minMemoryGb`, and the reserved kolors LoRA
/// family — so a manifest drift fails CI without a real download. Descriptor guidance/steps are covered
/// by `model_table_rows_resolve_and_flags_match_descriptor`.
#[test]
fn kolors_manifest_entry_gates_correctly() {
    let entry = builtin_model_entry("kolors");

    assert_eq!(
        entry.get("family").and_then(Value::as_str),
        Some("kolors"),
        "kolors family"
    );
    // Kolors keeps its full surface (unlike base SANA's t2i-only): edit/character/style variations.
    let caps: Vec<&str> = entry
        .get("capabilities")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    assert_eq!(
        caps,
        vec![
            "text_to_image",
            "edit_image",
            "character_image",
            "style_variations"
        ],
        "kolors keeps its full capability surface"
    );
    // Un-gated SceneWorks/* MLX re-host (the tier LICENSE travels with the weights) — NO `gated: true`.
    assert_ne!(
        entry.get("gated").and_then(Value::as_bool),
        Some(true),
        "kolors is un-gated (re-host carries the license)"
    );
    // Every tier downloads from the flipped SceneWorks re-host (was Kwai-Kolors/Kolors-diffusers).
    let downloads = entry
        .get("downloads")
        .and_then(Value::as_array)
        .expect("kolors downloads");
    for d in downloads {
        assert_eq!(
            d.get("repo").and_then(Value::as_str),
            Some("SceneWorks/kolors-mlx"),
            "kolors tier downloads from the SceneWorks re-host"
        );
        // Each tier is an installable artifact with a footprint (sc-8508).
        assert!(
            d.get("footprint")
                .and_then(|f| f.get("diskSizeBytes"))
                .and_then(Value::as_u64)
                .is_some(),
            "each kolors tier carries a footprint.diskSizeBytes"
        );
    }
    let variants: Vec<&str> = downloads
        .iter()
        .filter_map(|d| d.get("variant").and_then(Value::as_str))
        .collect();
    assert_eq!(
        variants,
        vec!["q4", "q8", "bf16"],
        "kolors ships the q4/q8/bf16 tier matrix"
    );
    // Packed q4 default + memory-eligibility lever, matching flux1/sana (ChatGLM3 TE is packed).
    let mlx = entry.get("mlx").expect("kolors mlx block");
    assert_eq!(
        mlx.get("quantize").and_then(Value::as_u64),
        Some(4),
        "kolors defaults to the packed q4 tier"
    );
    assert_eq!(
        mlx.get("repo").and_then(Value::as_str),
        Some("SceneWorks/kolors-mlx"),
        "kolors mlx.repo points at the re-host"
    );
    assert!(
        mlx.get("minMemoryGb").and_then(Value::as_u64).is_some(),
        "kolors mlx.minMemoryGb present"
    );
    // kolors LoRA family reserved (empty list would match every LoRA, sc-1927).
    let lora_families: Vec<&str> = entry
        .get("loraCompatibility")
        .and_then(|c| c.get("families"))
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    assert_eq!(lora_families, vec!["kolors"], "kolors loraCompatibility.families");
}

/// sc-9942 (epic 8506): the Wan2.2 T2V-A14B builtin entry ships the macOS quant matrix as per-tier
/// installable artifacts — `q4` (default) + `q8` + `bf16`, each a self-contained SceneWorks HF
/// download tagged with a `variant` and carrying an `estimatedSizeBytes` + `footprint` (sc-8508). The
/// video load path (`video_jobs::wan_tier_subdir`) descends into the chosen tier, so a drift
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
            let is_macos = d
                .get("platforms")
                .and_then(Value::as_array)
                .map(|p| p.iter().any(|x| x.as_str() == Some("macos")))
                .unwrap_or(false);
            // The Lightning coRequisite (sc-10030) is a macOS download too, but it is a mandatory
            // dependency, not a selectable quant tier — exclude it from the tier assertions.
            let is_corequisite = d.get("coRequisite").and_then(Value::as_bool) == Some(true);
            is_macos && !is_corequisite
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
/// path (`video_jobs::wan_tier_subdir`) descends into the chosen tier for BOTH A14B models, so a
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
            let is_macos = d
                .get("platforms")
                .and_then(Value::as_array)
                .map(|p| p.iter().any(|x| x.as_str() == Some("macos")))
                .unwrap_or(false);
            // The Lightning coRequisite (sc-10030) is a macOS download too, but it is a mandatory
            // dependency, not a selectable quant tier — exclude it from the tier assertions.
            let is_corequisite = d.get("coRequisite").and_then(Value::as_bool) == Some(true);
            is_macos && !is_corequisite
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

/// sc-9941 (epic 8506): the single-expert Wan2.2 TI2V-5B builtin entry (`wan_2_2`) ships the SAME
/// macOS quant matrix as its A14B siblings — per-tier `q4` (default) + `q8` + `bf16` self-contained
/// SceneWorks HF downloads tagged with a `variant`, each with an `estimatedSizeBytes` + `footprint`
/// (sc-8508). The video load path (`video_jobs::wan_tier_subdir`) descends into the chosen tier, so a
/// drift that drops a tier or its size would break the download UI + the RAM→tier suggestion — assert
/// the shape here so it fails CI.
#[test]
fn wan_ti2v_5b_manifest_ships_the_quant_matrix() {
    let entry = builtin_model_entry("wan_2_2");
    assert_eq!(
        entry.get("family").and_then(Value::as_str),
        Some("wan-video"),
        "wan TI2V-5B family"
    );
    assert_eq!(
        entry.get("type").and_then(Value::as_str),
        Some("video"),
        "wan TI2V-5B is a video model"
    );
    let downloads = entry
        .get("downloads")
        .and_then(Value::as_array)
        .expect("wan TI2V-5B downloads");
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
        "wan TI2V-5B ships the q4/q8/bf16 tier matrix on macOS"
    );
    for tier in &macos {
        let variant = tier.get("variant").and_then(Value::as_str).unwrap();
        assert_eq!(
            tier.get("repo").and_then(Value::as_str),
            Some("SceneWorks/wan2.2-ti2v-5b-mlx"),
            "{variant} tier hosts on the SceneWorks TI2V-5B quant-matrix repo"
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

/// sc-9945 (epic 8506): BOTH Bernini catalog ids — the `bernini` video id and the `bernini_image`
/// still-image companion — ship the same macOS quant matrix (per-tier `q4` default + `q8` + `bf16`
/// self-contained SceneWorks HF downloads, each `variant`-tagged with an `estimatedSizeBytes` +
/// `footprint`, all from the shared `SceneWorks/bernini-mlx` repo). Both load paths descend into the
/// chosen tier (`video_jobs::bernini_tier_subdir`), so a drift that drops a tier, its size, or lets the
/// two ids diverge would break the download UI + the RAM→tier suggestion — assert the shape here.
#[test]
fn bernini_manifest_ships_the_quant_matrix() {
    for (id, ty) in [("bernini", "video"), ("bernini_image", "image")] {
        let entry = builtin_model_entry(id);
        assert_eq!(
            entry.get("family").and_then(Value::as_str),
            Some("bernini"),
            "{id} family"
        );
        assert_eq!(entry.get("type").and_then(Value::as_str), Some(ty), "{id} type");
        let downloads = entry
            .get("downloads")
            .and_then(Value::as_array)
            .unwrap_or_else(|| panic!("{id} downloads"));
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
            "{id} ships the q4/q8/bf16 tier matrix on macOS"
        );
        for tier in &macos {
            let variant = tier.get("variant").and_then(Value::as_str).unwrap();
            assert_eq!(
                tier.get("repo").and_then(Value::as_str),
                Some("SceneWorks/bernini-mlx"),
                "{id} {variant} tier hosts on the shared SceneWorks Bernini repo"
            );
            let files: Vec<String> = tier
                .get("files")
                .and_then(Value::as_array)
                .map(|f| f.iter().filter_map(Value::as_str).map(String::from).collect())
                .unwrap_or_default();
            assert_eq!(
                files,
                vec![format!("{variant}/*")],
                "{id} {variant} tier installs only its own subdir"
            );
            assert!(
                tier.get("estimatedSizeBytes")
                    .and_then(Value::as_u64)
                    .is_some_and(|b| b > 0),
                "{id} {variant} tier declares a nonzero estimatedSizeBytes"
            );
            assert!(
                tier.get("footprint")
                    .and_then(|f| f.get("diskSizeBytes"))
                    .and_then(Value::as_u64)
                    .is_some_and(|b| b > 0),
                "{id} {variant} tier declares a footprint.diskSizeBytes"
            );
        }
        let defaults: Vec<&str> = macos
            .iter()
            .filter(|d| d.get("default").and_then(Value::as_bool) == Some(true))
            .filter_map(|d| d.get("variant").and_then(Value::as_str))
            .collect();
        assert_eq!(defaults, vec!["q4"], "{id} macOS default tier is q4");
    }
}

/// sc-11991 (epic 1788): the Mochi 1 builtin entry. Mochi deliberately DIVERGES from every
/// quant-matrix sibling above in ways a well-meaning "make it consistent with LTX" edit would undo,
/// so each divergence is pinned here:
///
///  * **One repo, BOTH backends.** Every download points at `SceneWorks/mochi-1-mlx` and NO entry
///    carries `platforms`. A6 (sc-11990) gave candle a `.scales`-detect seam that ingests the
///    mlx-affine tiers directly, CUDA-validated on Blackwell by downloading this repo byte-exact;
///    `SceneWorks/mochi-1-candle` was never published. Re-adding an LTX-style
///    `genmo/mochi-1-preview` Windows entry would also reintroduce sc-12113 (upstream ships bf16 AND
///    fp32 `vae/`; the loader's dir-glob picks fp32).
///  * **Tiers hold ONLY the transformer.** T5-XXL / VAE / tokenizer are a shared `coRequisite`
///    (sc-9696) resolved from the tier dir's PARENT, so they must stay OUT of the tier matrix and out
///    of the per-tier footprint.
///  * **No sampler/scheduler axis.** Both descriptors advertise `samplers: []` + `schedulers: []`
///    (one fixed flow-match Euler integrator) — advertising a menu would be a false capability.
///  * **t2v only, no LoRA.** `conditioning: []` and `supports_lora`/`supports_lokr` = false.
#[test]
fn mochi_manifest_ships_the_one_repo_tier_matrix() {
    let entry = builtin_model_entry("mochi_1");
    assert_eq!(
        entry.get("family").and_then(Value::as_str),
        Some("mochi"),
        "mochi family"
    );
    assert_eq!(
        entry.get("type").and_then(Value::as_str),
        Some("video"),
        "mochi is a video model"
    );
    // t2v ONLY — both descriptors declare `conditioning: []`, so there is no i2v/FLF/extend surface.
    let capabilities: Vec<&str> = entry
        .get("capabilities")
        .and_then(Value::as_array)
        .map(|c| c.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    assert_eq!(
        capabilities,
        vec!["text_to_video"],
        "mochi advertises text_to_video and nothing else"
    );
    // Apache-2.0 => ungated, no credential host.
    assert_ne!(
        entry.get("gated").and_then(Value::as_bool),
        Some(true),
        "mochi is Apache-2.0, not gated"
    );
    // ~20.1 GB for the default q4 install, ~53.5 GB for all three tiers + shared — a deliberate
    // opt-in. Mochi carries no `recommended: true`, so it is neither badged nor pre-checked in the
    // wizard; `autoDownload: false` states that intent explicitly.
    assert_eq!(
        entry.get("autoDownload").and_then(Value::as_bool),
        Some(false),
        "mochi is not auto-downloaded"
    );

    let downloads = entry
        .get("downloads")
        .and_then(Value::as_array)
        .expect("mochi downloads");
    // ONE repo, BOTH backends: every entry is the rehost and NONE is platform-tagged, so
    // `retain_downloads_for_os` keeps them all on macOS, Windows and Linux alike.
    for download in downloads {
        assert_eq!(
            download.get("repo").and_then(Value::as_str),
            Some("SceneWorks/mochi-1-mlx"),
            "every mochi download comes from the one rehost repo (candle ingests the mlx tiers)"
        );
        assert!(
            download.get("platforms").is_none(),
            "mochi downloads stay platform-agnostic — one repo serves both backends"
        );
    }

    // The shared T5-XXL / tokenizer / VAE co-requisite: fetched with any tier, excluded from the
    // tier matrix, and NOT a selectable variant.
    let corequisites: Vec<&Value> = downloads
        .iter()
        .filter(|d| d.get("coRequisite").and_then(Value::as_bool) == Some(true))
        .collect();
    assert_eq!(
        corequisites.len(),
        1,
        "exactly one mochi co-requisite (the shared text_encoder/tokenizer/vae)"
    );
    let shared_files: Vec<&str> = corequisites[0]
        .get("files")
        .and_then(Value::as_array)
        .map(|f| f.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    assert_eq!(
        shared_files,
        vec!["text_encoder/*", "tokenizer/*", "vae/*"],
        "the co-requisite carries the shared components the tier dirs resolve from their parent"
    );
    assert!(
        corequisites[0].get("variant").is_none(),
        "the shared co-requisite is not a selectable quant tier"
    );

    // The tier matrix: q4 (default) / q8 / bf16, each installing only its own transformer subdir,
    // with the EXACT hosted subdir byte sizes (HF API, repo rev 90a87786).
    let tiers: Vec<&Value> = downloads
        .iter()
        .filter(|d| d.get("coRequisite").and_then(Value::as_bool) != Some(true))
        .collect();
    let variants: Vec<&str> = tiers
        .iter()
        .filter_map(|d| d.get("variant").and_then(Value::as_str))
        .collect();
    assert_eq!(
        variants,
        vec!["q4", "q8", "bf16"],
        "mochi ships the q4/q8/bf16 tier matrix"
    );
    let expected_sizes = [
        ("q4", 9_670_883_602_u64),
        ("q8", 13_282_967_046),
        ("bf16", 20_055_485_000),
    ];
    for (tier, expected) in expected_sizes {
        let entry = tiers
            .iter()
            .find(|d| d.get("variant").and_then(Value::as_str) == Some(tier))
            .unwrap_or_else(|| panic!("mochi {tier} tier"));
        let files: Vec<String> = entry
            .get("files")
            .and_then(Value::as_array)
            .map(|f| f.iter().filter_map(Value::as_str).map(String::from).collect())
            .unwrap_or_default();
        assert_eq!(
            files,
            vec![format!("{tier}/*")],
            "{tier} tier installs only its own subdir (the shared components ride the co-requisite)"
        );
        assert_eq!(
            entry.get("estimatedSizeBytes").and_then(Value::as_u64),
            Some(expected),
            "{tier} tier declares the exact hosted subdir size"
        );
        assert_eq!(
            entry
                .get("footprint")
                .and_then(|f| f.get("diskSizeBytes"))
                .and_then(Value::as_u64),
            Some(expected),
            "{tier} tier footprint.diskSizeBytes matches its download size"
        );
    }
    let defaults: Vec<&str> = tiers
        .iter()
        .filter(|d| d.get("default").and_then(Value::as_bool) == Some(true))
        .filter_map(|d| d.get("variant").and_then(Value::as_str))
        .collect();
    assert_eq!(defaults, vec!["q4"], "the default mochi tier is q4");

    // limits: the engine hard-rejects width/height not divisible by 16, so every advertised bucket
    // must satisfy it. NO sampler/scheduler axis (both descriptors advertise neither).
    let limits = entry
        .get("limits")
        .and_then(Value::as_object)
        .expect("mochi limits");
    assert!(
        limits.get("samplers").is_none() && limits.get("schedulers").is_none(),
        "mochi advertises NO sampler/scheduler axis — one fixed flow-match Euler integrator"
    );
    assert_eq!(
        limits.get("requiresDimensionsMultipleOf").and_then(Value::as_u64),
        Some(16),
        "mochi requires dimensions divisible by 16"
    );
    let resolutions: Vec<&str> = limits
        .get("resolutions")
        .and_then(Value::as_array)
        .map(|r| r.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    assert_eq!(
        resolutions,
        vec!["848x480", "480x848"],
        "mochi advertises only its native 480p bucket"
    );
    for resolution in &resolutions {
        let (w, h) = resolution.split_once('x').expect("WxH resolution");
        for axis in [w, h] {
            let value: u32 = axis.parse().expect("numeric axis");
            assert_eq!(value % 16, 0, "{resolution} axis {axis} divides by 16");
        }
    }
    // 30 fps only: `frames = duration * fps`, so a second value would silently change clip length.
    let fps: Vec<u64> = limits
        .get("fps")
        .and_then(Value::as_array)
        .map(|f| f.iter().filter_map(Value::as_u64).collect())
        .unwrap_or_default();
    assert_eq!(fps, vec![30], "mochi is a 30 fps model");

    // defaults must be drawn from the advertised menus, or the studio opens on an invalid request.
    let defaults_block = entry
        .get("defaults")
        .and_then(Value::as_object)
        .expect("mochi defaults");
    assert_eq!(
        defaults_block.get("resolution").and_then(Value::as_str),
        Some("848x480"),
        "mochi defaults to its native bucket"
    );
    assert_eq!(
        defaults_block.get("fps").and_then(Value::as_u64),
        Some(30),
        "mochi defaults to 30 fps"
    );
    assert_eq!(
        defaults_block.get("steps").and_then(Value::as_u64),
        Some(64),
        "mochi defaults to the engine's DEFAULT_STEPS"
    );
    let durations: Vec<u64> = limits
        .get("durations")
        .and_then(Value::as_array)
        .map(|d| d.iter().filter_map(Value::as_u64).collect())
        .unwrap_or_default();
    let default_duration = defaults_block
        .get("duration")
        .and_then(Value::as_u64)
        .expect("mochi default duration");
    assert!(
        durations.contains(&default_duration),
        "the default duration {default_duration} is one of the advertised {durations:?}"
    );

    // No Mochi adapter path on either backend, but the family is declared so the picker never offers
    // cross-architecture LoRAs (sc-1927) — the SVD posture.
    let lora = entry
        .get("loraCompatibility")
        .and_then(Value::as_object)
        .expect("mochi loraCompatibility");
    assert_eq!(
        lora.get("families")
            .and_then(Value::as_array)
            .map(|f| f.iter().filter_map(Value::as_str).collect::<Vec<_>>()),
        Some(vec!["mochi"]),
        "mochi declares its own LoRA family"
    );
    assert_eq!(
        lora.get("types")
            .and_then(Value::as_array)
            .map(|t| t.len()),
        Some(0),
        "mochi supports no LoRA types (supports_lora/supports_lokr are false on both backends)"
    );

    let mlx = entry
        .get("mlx")
        .and_then(Value::as_object)
        .expect("mochi mlx block");
    assert_eq!(
        mlx.get("repo").and_then(Value::as_str),
        Some("SceneWorks/mochi-1-mlx"),
        "mochi mlx repo"
    );
    assert_eq!(
        mlx.get("standardTierLayout").and_then(Value::as_bool),
        Some(true),
        "mochi ships the standard q4/q8/bf16 turnkey tier layout"
    );
    assert!(
        mlx.get("minMemoryGb").and_then(Value::as_u64).is_some(),
        "mochi declares mlx.minMemoryGb"
    );

    // The prompt guide is served as a static file from apps/web/public; a typo'd path is a silent
    // 404 in the studio (nothing else in the repo reads `promptGuide`), so pin THIS entry's path.
    let guide = entry
        .get("ui")
        .and_then(|ui| ui.get("promptGuide"))
        .and_then(|g| g.get("path"))
        .and_then(Value::as_str)
        .expect("mochi ui.promptGuide.path");
    let guide_file = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../apps/web/public")
        .join(guide.trim_start_matches('/'));
    assert!(
        guide_file.is_file(),
        "mochi prompt guide {guide} resolves to a real file ({})",
        guide_file.display()
    );
}
