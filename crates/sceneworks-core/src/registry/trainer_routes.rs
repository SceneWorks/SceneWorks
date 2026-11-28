//! Backend-local trainer identity joins product training targets with the canonical
//! `sga.capabilities/2` export (`SW-6983`, `REG-3`). The product target supplies its logical id,
//! base model and kernel; `route_bindings::trainer_engine_id` binds it to the backend-local
//! trainer id; the `sga.capabilities/2` export owns the registered trainer descriptor.
use std::collections::{BTreeMap, BTreeSet};
use std::sync::LazyLock;

use crate::sga_capabilities::{
    CapabilitiesExportV2, ExportedTrainerEntry, CAPABILITIES_EXPORT_SCHEMA_VERSION,
};

#[derive(Clone)]
struct TrainerFacts {
    backend: String,
    trainer_mappings: BTreeMap<String, String>,
    trainers: BTreeMap<String, ExportedTrainerEntry>,
}

static EXPORTS: LazyLock<BTreeMap<String, TrainerFacts>> = LazyLock::new(|| {
    let mut exports = BTreeMap::new();
    for (backend, source) in [
        ("mlx", crate::sga_capabilities::MACOS_MLX_JSON),
        ("candle", crate::sga_capabilities::CANDLE_CUDA_JSON),
    ] {
        exports.insert(
            backend.to_owned(),
            parse_trainer_facts(source, backend)
                .expect("canonical sga.capabilities/2 trainer export"),
        );
    }
    exports
});

fn parse_trainer_facts(source: &str, backend: &str) -> Result<TrainerFacts, String> {
    let export = CapabilitiesExportV2::from_json_bytes(source.as_bytes())
        .map_err(|error| error.to_string())?;
    if export.schema_version != CAPABILITIES_EXPORT_SCHEMA_VERSION || export.backend != backend {
        return Err(format!("invalid {backend} trainer export schema/backend"));
    }
    let mut trainers = BTreeMap::new();
    for (key, trainer) in export.trainers {
        if trainer.backend != backend || trainer.id.trim().is_empty() || trainer.id != key {
            return Err(format!(
                "invalid or wrong-backend {backend} trainer descriptor {key:?}"
            ));
        }
        trainers.insert(key, trainer);
    }
    let mut trainer_mappings = BTreeMap::new();
    for &(target_id, mlx_engine_id, candle_engine_id) in
        super::route_bindings::TRAINER_TARGET_BINDINGS
    {
        let engine_id = match backend {
            "mlx" => mlx_engine_id,
            "candle" => candle_engine_id,
            _ => continue,
        };
        if target_id.trim().is_empty() || engine_id.trim().is_empty() {
            return Err(format!("incomplete {backend} trainer mapping"));
        }
        trainer_mappings.insert(target_id.to_owned(), engine_id.to_owned());
    }
    // The product binding also names deliberately unavailable mappings (for example Krea Control on
    // MLX). Claimability is a separate feature fact; keep the product identity available to
    // validation without treating every mapping as a registered route.
    Ok(TrainerFacts {
        backend: export.backend,
        trainer_mappings,
        trainers,
    })
}

pub(crate) fn expected_backend_local_trainer_id(
    target: &crate::training::TrainingTarget,
    backend: &str,
) -> Result<&'static str, String> {
    let facts = EXPORTS.get(backend).ok_or_else(|| {
        format!(
            "production training target {:?} requested an identity for unknown backend {backend:?}",
            target.id
        )
    })?;
    facts.trainer_mappings.get(&target.id).map(String::as_str).ok_or_else(|| {
        format!(
            "production training target {:?} (base {:?}, kernel {:?}) has no backend-local trainer in {backend} feature export",
            target.id, target.base_model, target.kernel
        )
    })
}

/// Kernels with a registered native trainer that supports at least one network type offered by a
/// shipped target on this backend. Product targets own the kernel/technique choices; the
/// `sga.capabilities/2` export owns the executable descriptor. An exported mapping alone is
/// insufficient: Krea Control intentionally maps on MLX without an MLX control trainer.
pub(crate) fn native_training_kernels(backend: &str) -> BTreeSet<String> {
    let Some(facts) = EXPORTS.get(backend) else {
        return BTreeSet::new();
    };
    let targets = crate::training::builtin_training_targets();
    training_kernels_from_facts(facts, &targets.targets)
}

fn training_kernels_from_facts(
    facts: &TrainerFacts,
    targets: &[crate::training::TrainingTarget],
) -> BTreeSet<String> {
    targets
        .iter()
        .filter(|target| {
            offered_network_types(target)
                .iter()
                .any(|kind| trainer_serves(facts, target, kind))
        })
        .map(|target| target.kernel.clone())
        .collect()
}

/// The network types a shipped target offers. A target that declares no `networkTypes` offers
/// its kernel's one technique: control for Krea Control, LoRA otherwise.
pub(crate) fn offered_network_types(target: &crate::training::TrainingTarget) -> Vec<String> {
    match target
        .limits
        .get("networkTypes")
        .and_then(serde_json::Value::as_array)
    {
        Some(types) => types
            .iter()
            .filter_map(serde_json::Value::as_str)
            .map(str::to_owned)
            .collect(),
        None if target.kernel == "krea_control" => vec!["control".to_owned()],
        None => vec!["lora".to_owned()],
    }
}

/// Whether `backend`'s native export serves `target` with network type `kind`.
#[cfg(test)]
pub(crate) fn native_trainer_serves(
    backend: &str,
    target: &crate::training::TrainingTarget,
    kind: &str,
) -> bool {
    EXPORTS
        .get(backend)
        .is_some_and(|facts| trainer_serves(facts, target, kind))
}

/// The target maps to a trainer registered on this backend, that trainer accepts the target's
/// reference cap (an instruction-edit target must never land on a text-to-image-only trainer under
/// the same engine id), and it supports the technique.
fn trainer_serves(
    facts: &TrainerFacts,
    target: &crate::training::TrainingTarget,
    kind: &str,
) -> bool {
    let Some(engine) = facts.trainer_mappings.get(&target.id) else {
        return false;
    };
    let Some(descriptor) = facts
        .trainers
        .get(engine)
        .filter(|descriptor| descriptor.id == *engine && descriptor.backend == facts.backend)
    else {
        return false;
    };
    if descriptor.max_reference_images
        < crate::training::training_target_max_reference_images(target)
    {
        return false;
    }
    match kind {
        "lora" => descriptor.supports_lora,
        "lokr" => descriptor.supports_lokr,
        "control" => descriptor.supports_control,
        "full" => descriptor.supports_full_finetune,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::{training_target_max_reference_images, TrainingTarget};
    use serde_json::{json, Value};

    const MLX: &str = crate::sga_capabilities::MACOS_MLX_JSON;
    const CANDLE: &str = crate::sga_capabilities::CANDLE_CUDA_JSON;

    fn native(backend: &str) -> TrainerFacts {
        parse_trainer_facts(if backend == "mlx" { MLX } else { CANDLE }, backend).unwrap()
    }

    fn registered<'a>(
        facts: &'a TrainerFacts,
        target: &TrainingTarget,
    ) -> Option<&'a ExportedTrainerEntry> {
        let engine = facts.trainer_mappings.get(&target.id)?;
        facts
            .trainers
            .get(engine)
            .filter(|trainer| trainer.backend == facts.backend && trainer.id == *engine)
    }

    /// The production per-(target, technique) predicate the kernel projection is built from.
    fn supports(facts: &TrainerFacts, target: &TrainingTarget, kind: &str) -> bool {
        trainer_serves(facts, target, kind)
    }

    fn training_job(target: &TrainingTarget, kind: &str) -> crate::contracts::JobSnapshot {
        serde_json::from_value(json!({
            "id": "typed-trainer-route-probe",
            "type": "lora_train",
            "status": "queued",
            "payload": {
                "dryRun": false,
                "plan": {
                    "target": {
                        "targetId": target.id,
                        "kernel": target.kernel,
                        "baseModel": target.base_model
                    },
                    "config": { "advanced": { "networkType": kind } }
                }
            },
            "result": {},
            "requestedGpu": "auto",
            "progress": 0,
            "stage": "queued",
            "message": "",
            "attempts": 1,
            "cancelRequested": false,
            "createdAt": "2026-11-07T00:00:00Z",
            "updatedAt": "2026-11-07T00:00:00Z"
        }))
        .unwrap()
    }

    #[test]
    fn qwen_image_training_targets_are_claimable_on_both_native_lanes() {
        let targets = crate::training::builtin_training_targets();
        for id in ["qwen_image_2_1_lora", "qwen_image_2_1_edit_lora"] {
            let target = targets
                .targets
                .iter()
                .find(|target| target.id == id)
                .unwrap();
            for kind in ["lora", "lokr"] {
                let job = training_job(target, kind);
                assert!(
                    crate::jobs_store::training_job_is_mlx_eligible(&job),
                    "MLX {id}/{kind}"
                );
                assert!(
                    crate::jobs_store::training_job_is_candle_eligible(&job),
                    "Candle {id}/{kind}"
                );
                for backend in ["mlx", "candle"] {
                    let facts = native(backend);
                    assert_eq!(
                        expected_backend_local_trainer_id(target, backend).unwrap(),
                        "qwen_image_2_1"
                    );
                    assert!(
                        supports(&facts, target, kind),
                        "{backend} {id}/{kind} descriptor"
                    );
                }
            }
        }
    }

    /// SW-7103 (ported from the retired capability matrix): both runtimes register the
    /// `minimax_h3_image` trainer (SW-7097 MLX, SW-7098 Candle) with the generator's 16-reference
    /// edit cap, both H3-Image targets map onto it, and their LoRA and LoKr jobs are claimable on
    /// both native lanes.
    #[test]
    fn minimax_h3_image_training_targets_are_claimable_on_both_native_lanes() {
        let targets = crate::training::builtin_training_targets();
        for id in ["minimax_h3_image_lora", "minimax_h3_image_edit_lora"] {
            let target = targets
                .targets
                .iter()
                .find(|target| target.id == id)
                .unwrap();
            for backend in ["mlx", "candle"] {
                let facts = native(backend);
                assert_eq!(
                    expected_backend_local_trainer_id(target, backend).unwrap(),
                    "minimax_h3_image"
                );
                assert!(
                    facts
                        .trainers
                        .get("minimax_h3_image")
                        .is_some_and(|trainer| trainer.max_reference_images >= 16),
                    "{backend} registers the edit-capable minimax_h3_image trainer"
                );
            }
            for kind in ["lora", "lokr"] {
                let job = training_job(target, kind);
                assert!(
                    crate::jobs_store::training_job_is_mlx_eligible(&job),
                    "MLX {id}/{kind}"
                );
                assert!(
                    crate::jobs_store::training_job_is_candle_eligible(&job),
                    "Candle {id}/{kind}"
                );
                for backend in ["mlx", "candle"] {
                    assert!(
                        supports(&native(backend), target, kind),
                        "{backend} {id}/{kind} descriptor"
                    );
                }
            }
        }
    }

    #[test]
    fn native_exports_cover_exactly_the_product_training_targets() {
        let targets = crate::training::builtin_training_targets();
        let product_ids: BTreeSet<_> = targets
            .targets
            .iter()
            .map(|target| target.id.as_str())
            .collect();
        for backend in ["mlx", "candle"] {
            let facts = native(backend);
            let exported_ids: BTreeSet<_> =
                facts.trainer_mappings.keys().map(String::as_str).collect();
            assert_eq!(
                exported_ids, product_ids,
                "{backend} target population drift"
            );
            for target in &targets.targets {
                assert_eq!(
                    expected_backend_local_trainer_id(target, backend).unwrap(),
                    facts.trainer_mappings[&target.id],
                    "{backend}/{} must use the product's native trainer identity",
                    target.id
                );
            }
        }
        // A target the Candle export no longer maps is unclaimable there: the production kernel
        // projection drops it rather than borrowing another target's trainer.
        for id in ["ltx_video_lora", "ltx_2_5_video_lora"] {
            let target = targets
                .targets
                .iter()
                .find(|target| target.id == id)
                .unwrap()
                .clone();
            let mut missing = native("candle");
            assert!(
                training_kernels_from_facts(&missing, std::slice::from_ref(&target))
                    .contains(&target.kernel),
                "{id} is served before the mapping is withdrawn"
            );
            missing.trainer_mappings.remove(id);
            assert!(
                training_kernels_from_facts(&missing, std::slice::from_ref(&target)).is_empty(),
                "a withdrawn Candle mapping for {id} must not stay claimable"
            );
        }
    }

    #[test]
    fn native_trainer_exports_refuse_incomplete_or_wrong_backend_descriptors() {
        for (backend, source) in [("mlx", MLX), ("candle", CANDLE)] {
            parse_trainer_facts(source, backend).unwrap();

            let mut wrong_backend: Value = serde_json::from_str(source).unwrap();
            let first_key = wrong_backend["trainers"]
                .as_object()
                .unwrap()
                .keys()
                .next()
                .unwrap()
                .clone();
            wrong_backend["trainers"][&first_key]["backend"] = json!("other");
            assert!(parse_trainer_facts(&wrong_backend.to_string(), backend).is_err());

            let mut mismatched_id: Value = serde_json::from_str(source).unwrap();
            mismatched_id["trainers"][&first_key]["id"] = json!("wrong_trainer_id");
            assert!(parse_trainer_facts(&mismatched_id.to_string(), backend).is_err());

            let mut wrong_schema: Value = serde_json::from_str(source).unwrap();
            wrong_schema["schema_version"] = json!("sga.capabilities/999");
            assert!(parse_trainer_facts(&wrong_schema.to_string(), backend).is_err());
        }
    }

    #[test]
    fn native_training_mapping_refuses_unregistered_engine_and_unknown_target() {
        let target = crate::training::builtin_training_targets()
            .targets
            .into_iter()
            .find(|target| target.id == "qwen_image_2_1_lora")
            .unwrap();
        assert!(expected_backend_local_trainer_id(&target, "unknown").is_err());
        let mut unknown = target.clone();
        unknown.id = "missing_training_target".to_owned();
        assert!(expected_backend_local_trainer_id(&unknown, "mlx").is_err());
        for backend in ["mlx", "candle"] {
            let mut facts = native(backend);
            facts
                .trainer_mappings
                .insert(target.id.clone(), "unregistered_native_trainer".to_owned());
            assert!(!supports(&facts, &target, "lora"));
            facts
                .trainer_mappings
                .insert(target.id.clone(), "qwen_image_2_1".to_owned());
            facts.trainers.get_mut("qwen_image_2_1").unwrap().backend = "wrong_backend".to_owned();
            assert!(!supports(&facts, &target, "lora"));
        }
    }

    #[test]
    fn routed_kernel_projection_requires_a_registered_capable_native_trainer() {
        let targets = crate::training::builtin_training_targets();
        let mlx = native("mlx");
        let candle = native("candle");
        let mlx_kernels = training_kernels_from_facts(&mlx, &targets.targets);
        let candle_kernels = training_kernels_from_facts(&candle, &targets.targets);
        for kernels in [&mlx_kernels, &candle_kernels] {
            for kernel in [
                "mage_flow_lora",
                "ltx_mlx_lora",
                "qwen_image_2_1_lora",
                "qwen_image_2_1_edit_lora",
            ] {
                assert!(kernels.contains(kernel), "{kernel} needs a native trainer");
            }
        }
        assert!(!mlx_kernels.contains("krea_control"));
        assert!(candle_kernels.contains("krea_control"));

        let mut missing = mlx.clone();
        missing.trainers.remove("qwen_image_2_1");
        let missing_kernels = training_kernels_from_facts(&missing, &targets.targets);
        assert!(!missing_kernels.contains("qwen_image_2_1_lora"));
        assert!(!missing_kernels.contains("qwen_image_2_1_edit_lora"));
        assert!(missing_kernels.contains("mage_flow_lora"));

        let mut incapable = candle.clone();
        let qwen = incapable.trainers.get_mut("qwen_image_2_1").unwrap();
        qwen.supports_lora = false;
        qwen.supports_lokr = false;
        let incapable_kernels = training_kernels_from_facts(&incapable, &targets.targets);
        assert!(!incapable_kernels.contains("qwen_image_2_1_lora"));
        assert!(!incapable_kernels.contains("qwen_image_2_1_edit_lora"));
    }

    #[test]
    fn ltx_2_3_target_rejects_the_registered_ltx_2_5_trainer() {
        let target = crate::training::builtin_training_targets()
            .targets
            .into_iter()
            .find(|target| target.id == "ltx_video_lora")
            .unwrap();
        let job = training_job(&target, "lora");
        assert!(crate::jobs_store::training_job_is_mlx_eligible(&job));
        assert!(crate::jobs_store::training_job_is_candle_eligible(&job));
        for backend in ["mlx", "candle"] {
            assert_eq!(
                expected_backend_local_trainer_id(&target, backend).unwrap(),
                "ltx_2_3",
                "a same-capability newer trainer cannot satisfy the LTX 2.3 product target"
            );
            let only = std::slice::from_ref(&target);
            let facts = native(backend);
            assert_eq!(
                training_kernels_from_facts(&facts, only),
                BTreeSet::from([target.kernel.clone()]),
                "{backend} serves the LTX 2.3 target"
            );
            let mut unregistered = native(backend);
            unregistered
                .trainer_mappings
                .insert(target.id.clone(), "unregistered_native_trainer".to_owned());
            assert!(
                training_kernels_from_facts(&unregistered, only).is_empty(),
                "{backend} unregistered trainer id"
            );
            let mut missing = native(backend);
            missing.trainers.remove("ltx_2_3");
            assert!(
                training_kernels_from_facts(&missing, only).is_empty(),
                "{backend} missing descriptor"
            );
            let mut wrong_backend = native(backend);
            wrong_backend.trainers.get_mut("ltx_2_3").unwrap().backend = "wrong_backend".to_owned();
            assert!(
                training_kernels_from_facts(&wrong_backend, only).is_empty(),
                "{backend} wrong-backend descriptor"
            );
            let mut no_lora = native(backend);
            no_lora.trainers.get_mut("ltx_2_3").unwrap().supports_lora = false;
            assert!(
                training_kernels_from_facts(&no_lora, only).is_empty(),
                "{backend} missing LoRA support"
            );
        }
    }

    #[test]
    fn paired_native_trainers_preserve_x0_decoder_and_edit_reference_cap() {
        let targets = crate::training::builtin_training_targets();
        for target in &targets.targets {
            let mlx = expected_backend_local_trainer_id(target, "mlx").unwrap();
            let candle = expected_backend_local_trainer_id(target, "candle").unwrap();
            assert_eq!(
                crate::training::depth_anchoring::x0_decoder_for_trainer(mlx),
                crate::training::depth_anchoring::x0_decoder_for_trainer(candle),
                "{}: {mlx} vs {candle}",
                target.id
            );
        }
        let edit = targets
            .targets
            .iter()
            .find(|target| target.id == "qwen_image_2_1_edit_lora")
            .unwrap();
        let t2i = targets
            .targets
            .iter()
            .find(|target| target.id == "qwen_image_2_1_lora")
            .unwrap();
        let required = training_target_max_reference_images(edit);
        assert_eq!(required, 10);
        assert_eq!(training_target_max_reference_images(t2i), 0);
        for backend in ["mlx", "candle"] {
            let facts = native(backend);
            assert_eq!(registered(&facts, edit).unwrap().id, "qwen_image_2_1");
            for (cap, edit_served) in [
                (0, false),
                (required - 1, false),
                (required, true),
                (required + 2, true),
            ] {
                let mut capped = facts.clone();
                capped
                    .trainers
                    .get_mut("qwen_image_2_1")
                    .unwrap()
                    .max_reference_images = cap;
                for kind in ["lora", "lokr"] {
                    assert_eq!(
                        supports(&capped, edit, kind),
                        edit_served,
                        "{backend} edit/{kind} with a trainer cap of {cap}"
                    );
                    assert!(
                        supports(&capped, t2i, kind),
                        "{backend} t2i/{kind} has no ref cap"
                    );
                }
                let kernels = training_kernels_from_facts(&capped, &targets.targets);
                assert_eq!(
                    kernels.contains(&edit.kernel),
                    edit_served,
                    "{backend} edit kernel projection with a trainer cap of {cap}"
                );
                assert!(kernels.contains(&t2i.kernel), "{backend} t2i kernel");
            }
        }
    }

    #[test]
    fn ltx_2_5_has_distinct_backend_local_identity() {
        let target = crate::training::builtin_training_targets()
            .targets
            .into_iter()
            .find(|target| target.id == "ltx_2_5_video_lora")
            .expect("shipped target");
        assert_eq!(
            expected_backend_local_trainer_id(&target, "mlx").unwrap(),
            "ltx_2_5"
        );
        assert_eq!(
            expected_backend_local_trainer_id(&target, "candle").unwrap(),
            "ltx_2_5_distilled"
        );
    }

    #[test]
    fn moss_trainer_routes_match_product_targets() {
        let targets = crate::training::builtin_training_targets();
        for (id, base) in [
            ("moss_image_base_lora", "moss_image_base"),
            ("moss_image_sft_lora", "moss_image_sft"),
        ] {
            let target = targets
                .targets
                .iter()
                .find(|target| target.id == id)
                .unwrap();
            assert_eq!(target.base_model, base);
            assert_eq!(target.kernel, "moss_image_lora");
            for backend in ["mlx", "candle"] {
                assert_eq!(
                    expected_backend_local_trainer_id(target, backend).unwrap(),
                    base
                );
            }
        }
    }
}
