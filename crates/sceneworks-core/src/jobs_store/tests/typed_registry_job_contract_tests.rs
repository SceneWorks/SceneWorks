use super::*;
use serde_json::json;

fn job(job_type: JobType, payload: Value) -> JobSnapshot {
    serde_json::from_value(json!({
        "id": "native-contract-probe", "type": job_type.as_str(), "status": "queued",
        "payload": payload, "result": {}, "requestedGpu": "auto", "progress": 0,
        "stage": "queued", "message": "", "attempts": 1, "cancelRequested": false,
        "createdAt": "2026-11-07T00:00:00Z", "updatedAt": "2026-11-07T00:00:00Z"
    }))
    .unwrap()
}

fn worker(gpu_id: &str, capabilities: &[&str]) -> WorkerSnapshot {
    serde_json::from_value(json!({
        "id": format!("worker-{gpu_id}"), "gpuId": gpu_id, "status": "idle",
        "capabilities": capabilities, "loadedModels": [],
        "registeredAt": "2026-11-07T00:00:00Z", "lastSeenAt": "2026-11-07T00:00:00Z"
    }))
    .unwrap()
}

#[test]
fn vector_mode_selects_native_worker_capability_and_requires_acceleration() {
    let image = job(
        JobType::VectorGenerate,
        json!({
            "model": "starvector_1b", "mode": "image_to_svg", "sourceAssetId": "source"
        }),
    );
    let text = job(
        JobType::VectorGenerate,
        json!({
            "model": "starvector_1b", "mode": "text_to_svg", "prompt": "a logo"
        }),
    );
    assert_eq!(required_capability(&image), "vector_image_to_svg");
    assert_eq!(required_capability(&text), "vector_text_to_svg");
    assert!(job_requires_gpu(&JobType::VectorGenerate));
    assert!(!worker_supports_job(
        &worker("cpu", &["vector_image_to_svg"]),
        &image
    ));
    assert!(worker_supports_job(
        &worker("cuda", &["gpu", "vector_image_to_svg"]),
        &image
    ));
    assert!(!worker_supports_job(
        &worker("cuda", &["gpu", "vector_image_to_svg"]),
        &text
    ));
}

#[test]
fn utility_jobs_use_their_actual_job_type_and_native_claim() {
    let detail = job(
        JobType::ImageDetail,
        json!({
            "model": "sdxl", "engine": "controlnet-tile", "sourceAssetId": "source"
        }),
    );
    assert!(crate::jobs_store::image_detail_native_eligible(&detail));
    assert_eq!(required_capability(&detail), "image_detail");
    for backend in ["mlx", "candle"] {
        assert!(
            worker_supports_job(&worker(backend, &["gpu", "image_detail"]), &detail),
            "{backend} detail"
        );
    }
    let vision = job(
        JobType::PromptRefine,
        json!({
            "model": "vision_caption_qwen3vl_8b", "task": "image_caption",
            "imagePath": "probe"
        }),
    );
    assert_eq!(required_capability(&vision), "prompt_refine");
    for backend in ["mlx", "candle"] {
        assert!(
            worker_supports_job(&worker(backend, &["prompt_refine"]), &vision),
            "{backend} caption"
        );
    }
    let film = job(
        JobType::PromptRefine,
        json!({
            "prompt": "a shot", "task": "film_plan", "workflow": "video",
            "model": crate::film_workspace::QWEN36_FILM_PLANNER_REPO,
            "modelId": "minimax_h3", "thinkingMode": "disabled"
        }),
    );
    assert_eq!(required_capability(&film), "prompt_refine");
    for backend in ["mlx", "candle"] {
        assert!(
            worker_supports_job(&worker(backend, &["prompt_refine"]), &film),
            "{backend} film planner"
        );
    }
    let training = job(
        JobType::TrainingCaption,
        json!({"captioner": "vision_caption_qwen3vl_8b"}),
    );
    assert!(!crate::jobs_store::caption_job_is_mlx_eligible(&training));
}

#[test]
fn derived_audio_tiers_require_audio_capability() {
    let derived = job(
        JobType::ModelDownload,
        json!({
            "modelId": "yue2", "localDerivation": {
                "fromVariant": "bf16", "conversion": "yue2-ggml-tier-v1"
            }
        }),
    );
    assert_eq!(required_capability(&derived), "audio_generate");
    assert!(!worker_supports_job(
        &worker("cpu", &["cpu", "model_download"]),
        &derived
    ));
    assert!(worker_supports_job(
        &worker("cuda", &["gpu", "audio_generate"]),
        &derived
    ));
}

#[test]
fn gpu_backed_jobs_reject_cpu_claims_while_plain_download_remains_utility_work() {
    for job_type in [
        JobType::ImageGenerate,
        JobType::ImageEdit,
        JobType::ImageDetail,
        JobType::VectorGenerate,
        JobType::VideoGenerate,
        JobType::VideoExtend,
        JobType::VideoBridge,
        JobType::VideoUpscale,
        JobType::PersonReplace,
        JobType::LoraTrain,
        JobType::ControlTraining,
        JobType::TrainingCaption,
        JobType::DatasetAnalysis,
        JobType::CatalogAnalysis,
        JobType::DatasetUpscale,
        JobType::DatasetFaceAnalysis,
        JobType::DatasetSubjectMask,
        JobType::FaceLikenessCompare,
    ] {
        assert!(job_requires_gpu(&job_type), "{}", job_type.as_str());
        let request = job(job_type.clone(), json!({}));
        assert!(
            !worker_supports_job(&worker("cpu", &["cpu", job_type.as_str()]), &request),
            "{}",
            job_type.as_str()
        );
    }
    assert!(!job_requires_gpu(&JobType::ModelDownload));
    let download = job(JobType::ModelDownload, json!({"modelId": "yue2"}));
    assert!(worker_supports_job(
        &worker("cpu", &["cpu", "model_download"]),
        &download
    ));
}

#[test]
fn native_only_training_policy_refuses_generic_workers_without_stranding_targets() {
    use crate::jobs_store::routing::catalog::{
        CANDLE_ROUTED_TRAINING_KERNELS, MLX_ONLY_TRAINING_KERNELS, MLX_ROUTED_TRAINING_KERNELS,
    };

    let generic = worker("cuda:0", &["gpu", "lora_train", "lora_train_execute"]);
    let targets = crate::training::builtin_training_targets();
    for kernel in MLX_ONLY_TRAINING_KERNELS {
        assert!(
            MLX_ROUTED_TRAINING_KERNELS.contains(*kernel)
                || CANDLE_ROUTED_TRAINING_KERNELS.contains(*kernel),
            "native-only {kernel} must have a native serving lane"
        );
        let matching: Vec<_> = targets
            .targets
            .iter()
            .filter(|target| target.kernel == *kernel)
            .collect();
        assert!(!matching.is_empty(), "{kernel} must have a shipped target");
        for target in matching {
            let job_type = if *kernel == "krea_control" {
                JobType::ControlTraining
            } else {
                JobType::LoraTrain
            };
            let request = job(
                job_type,
                json!({
                    "dryRun": false,
                    "plan": {"target": {
                        "targetId": target.id, "kernel": target.kernel,
                        "baseModel": target.base_model
                    }, "config": {"advanced": {"networkType": "lora"}}}
                }),
            );
            assert!(crate::jobs_store::training_kernel_is_mlx_only(&request));
            assert!(
                !worker_supports_job(&generic, &request),
                "generic worker claimed {kernel}"
            );
        }
    }
}
