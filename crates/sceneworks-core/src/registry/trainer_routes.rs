//! Backend-local trainer identity comes from the typed runtime capability export. The product
//! target supplies its logical id, base model and kernel; the export binds that id to a registered
//! native trainer on each backend.
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::LazyLock;

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TrainerFacts {
    schema_version: u32,
    trainer_mappings: BTreeMap<String, String>,
    snapshot: TrainerSnapshot,
}

#[derive(Clone, Deserialize)]
struct TrainerSnapshot {
    backend: String,
    trainer_capabilities: Vec<TrainerDescriptor>,
}

#[derive(Clone, Deserialize)]
struct TrainerDescriptor {
    id: String,
    backend: String,
    #[serde(default)]
    supports_lora: bool,
    #[serde(default)]
    supports_lokr: bool,
    #[serde(default)]
    supports_control: bool,
    #[serde(default)]
    supports_full_finetune: bool,
    #[serde(default)]
    max_reference_images: u32,
}

static EXPORTS: LazyLock<BTreeMap<String, TrainerFacts>> = LazyLock::new(|| {
    let mut exports = BTreeMap::new();
    for (backend, source) in [
        (
            "mlx",
            include_str!("../../../../config/engine-capabilities/runtime/capabilities.mlx.json"),
        ),
        (
            "candle",
            include_str!("../../../../config/engine-capabilities/runtime/capabilities.candle.json"),
        ),
    ] {
        exports.insert(
            backend.to_owned(),
            parse_trainer_facts(source, backend).expect("typed runtime trainer export"),
        );
    }
    exports
});

fn parse_trainer_facts(source: &str, backend: &str) -> Result<TrainerFacts, String> {
    let facts: TrainerFacts = serde_json::from_str(source).map_err(|error| error.to_string())?;
    if facts.schema_version != 2 || facts.snapshot.backend != backend {
        return Err(format!("invalid {backend} trainer export schema/backend"));
    }
    let registered: BTreeSet<_> = facts
        .snapshot
        .trainer_capabilities
        .iter()
        .filter(|trainer| trainer.backend == backend && !trainer.id.trim().is_empty())
        .map(|trainer| trainer.id.as_str())
        .collect();
    if registered.len() != facts.snapshot.trainer_capabilities.len() {
        return Err(format!(
            "duplicate or wrong-backend {backend} trainer descriptor"
        ));
    }
    if facts
        .trainer_mappings
        .iter()
        .any(|(target, engine)| target.trim().is_empty() || engine.trim().is_empty())
    {
        return Err(format!("incomplete {backend} trainer mapping"));
    }
    // The export also names deliberately unavailable mappings (for example Krea Control on
    // MLX). Claimability is a separate feature fact; keep the product identity available to
    // validation without treating every mapping as a registered route.
    Ok(facts)
}

#[cfg(test)]
fn parse_trainer_routes(source: &str, backend: &str) -> Result<BTreeMap<String, String>, String> {
    parse_trainer_facts(source, backend).map(|facts| facts.trainer_mappings)
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
/// shipped target on this backend. Product targets own the kernel/technique choices; the runtime
/// export owns the backend-local identity and executable descriptor. An exported mapping alone is
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
            let Some(engine) = facts.trainer_mappings.get(&target.id) else {
                return false;
            };
            let Some(descriptor) = facts
                .snapshot
                .trainer_capabilities
                .iter()
                .find(|descriptor| {
                    descriptor.id == *engine && descriptor.backend == facts.snapshot.backend
                })
            else {
                return false;
            };
            if descriptor.max_reference_images
                < crate::training::training_target_max_reference_images(target)
            {
                return false;
            }
            let offered = target
                .limits
                .get("networkTypes")
                .and_then(serde_json::Value::as_array);
            match offered {
                Some(types) => types.iter().any(|kind| match kind.as_str() {
                    Some("lora") => descriptor.supports_lora,
                    Some("lokr") => descriptor.supports_lokr,
                    Some("control") => descriptor.supports_control,
                    Some("full") => descriptor.supports_full_finetune,
                    _ => false,
                }),
                None if target.kernel == "krea_control" => descriptor.supports_control,
                None => descriptor.supports_lora,
            }
        })
        .map(|target| target.kernel.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::{training_target_max_reference_images, TrainingTarget};
    use serde_json::{json, Value};

    const MLX: &str =
        include_str!("../../../../config/engine-capabilities/runtime/capabilities.mlx.json");
    const CANDLE: &str =
        include_str!("../../../../config/engine-capabilities/runtime/capabilities.candle.json");

    fn native(backend: &str) -> TrainerFacts {
        serde_json::from_str(if backend == "mlx" { MLX } else { CANDLE }).unwrap()
    }

    fn registered<'a>(
        facts: &'a TrainerFacts,
        target: &TrainingTarget,
    ) -> Option<&'a TrainerDescriptor> {
        let engine = facts.trainer_mappings.get(&target.id)?;
        facts
            .snapshot
            .trainer_capabilities
            .iter()
            .find(|trainer| trainer.backend == facts.snapshot.backend && trainer.id == *engine)
    }

    fn supports(facts: &TrainerFacts, target: &TrainingTarget, kind: &str) -> bool {
        let Some(trainer) = registered(facts, target) else {
            return false;
        };
        trainer.max_reference_images >= training_target_max_reference_images(target)
            && match kind {
                "lora" => trainer.supports_lora,
                "lokr" => trainer.supports_lokr,
                "control" => trainer.supports_control,
                "full" => trainer.supports_full_finetune,
                _ => false,
            }
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
        let mut missing = native("candle");
        missing.trainer_mappings.remove("ltx_video_lora");
        assert_ne!(
            missing
                .trainer_mappings
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>(),
            product_ids,
            "a missing Candle target must not satisfy the paired population"
        );
    }

    #[test]
    fn native_trainer_exports_refuse_incomplete_or_wrong_backend_descriptors() {
        for (backend, source) in [("mlx", MLX), ("candle", CANDLE)] {
            parse_trainer_routes(source, backend).unwrap();
            let mut incomplete: Value = serde_json::from_str(source).unwrap();
            incomplete["trainerMappings"]["ltx_video_lora"] = json!("");
            assert!(parse_trainer_routes(&incomplete.to_string(), backend).is_err());

            let mut wrong_backend: Value = serde_json::from_str(source).unwrap();
            wrong_backend["snapshot"]["trainer_capabilities"][0]["backend"] = json!("other");
            assert!(parse_trainer_routes(&wrong_backend.to_string(), backend).is_err());

            let mut duplicate: Value = serde_json::from_str(source).unwrap();
            let first = duplicate["snapshot"]["trainer_capabilities"][0].clone();
            duplicate["snapshot"]["trainer_capabilities"]
                .as_array_mut()
                .unwrap()
                .push(first);
            assert!(parse_trainer_routes(&duplicate.to_string(), backend).is_err());
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
            let trainer = facts
                .snapshot
                .trainer_capabilities
                .iter_mut()
                .find(|trainer| trainer.id == "qwen_image_2_1")
                .unwrap();
            trainer.backend = "wrong_backend".to_owned();
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
        missing
            .snapshot
            .trainer_capabilities
            .retain(|trainer| trainer.id != "qwen_image_2_1");
        let missing_kernels = training_kernels_from_facts(&missing, &targets.targets);
        assert!(!missing_kernels.contains("qwen_image_2_1_lora"));
        assert!(!missing_kernels.contains("qwen_image_2_1_edit_lora"));
        assert!(missing_kernels.contains("mage_flow_lora"));

        let mut incapable = candle.clone();
        let qwen = incapable
            .snapshot
            .trainer_capabilities
            .iter_mut()
            .find(|trainer| trainer.id == "qwen_image_2_1")
            .unwrap();
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
            let mut facts = native(backend);
            assert!(supports(&facts, &target, "lora"));
            facts.trainer_mappings.insert(
                target.id.clone(),
                if backend == "mlx" {
                    "ltx_2_5"
                } else {
                    "ltx_2_5_distilled"
                }
                .to_owned(),
            );
            assert_ne!(
                facts.trainer_mappings[&target.id],
                expected_backend_local_trainer_id(&target, backend).unwrap(),
                "a newer registered trainer is not the target's native identity"
            );
            facts
                .trainer_mappings
                .insert(target.id.clone(), "ltx_2_3".to_owned());
            facts
                .snapshot
                .trainer_capabilities
                .retain(|trainer| trainer.id != "ltx_2_3");
            assert!(
                !supports(&facts, &target, "lora"),
                "{backend} missing descriptor"
            );
            let mut facts = native(backend);
            facts
                .snapshot
                .trainer_capabilities
                .iter_mut()
                .find(|trainer| trainer.id == "ltx_2_3")
                .unwrap()
                .supports_lora = false;
            assert!(
                !supports(&facts, &target, "lora"),
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
            for kind in ["lora", "lokr"] {
                assert!(supports(&facts, edit, kind), "{backend} edit/{kind}");
                let mut too_small = facts.clone();
                too_small
                    .snapshot
                    .trainer_capabilities
                    .iter_mut()
                    .find(|trainer| trainer.id == "qwen_image_2_1")
                    .unwrap()
                    .max_reference_images = required - 1;
                assert!(
                    !supports(&too_small, edit, kind),
                    "{backend} edit/{kind} cap"
                );
                assert!(
                    supports(&too_small, t2i, kind),
                    "{backend} t2i/{kind} has no ref cap"
                );
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
