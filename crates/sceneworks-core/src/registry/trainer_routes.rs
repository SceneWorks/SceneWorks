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
    #[cfg(test)]
    #[serde(default)]
    supports_lora: bool,
    #[cfg(test)]
    #[serde(default)]
    supports_lokr: bool,
    #[cfg(test)]
    #[serde(default)]
    supports_control: bool,
    #[cfg(test)]
    #[serde(default)]
    supports_full_finetune: bool,
    #[cfg(test)]
    #[serde(default)]
    max_reference_images: u32,
}

static ROUTES: LazyLock<BTreeMap<String, BTreeMap<String, String>>> = LazyLock::new(|| {
    let mut routes = BTreeMap::new();
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
        routes.insert(
            backend.to_owned(),
            parse_trainer_routes(source, backend).expect("typed runtime trainer export"),
        );
    }
    routes
});

fn parse_trainer_routes(source: &str, backend: &str) -> Result<BTreeMap<String, String>, String> {
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
    Ok(facts.trainer_mappings)
}

pub(crate) fn expected_backend_local_trainer_id(
    target: &crate::training::TrainingTarget,
    backend: &str,
) -> Result<&'static str, String> {
    let mappings = ROUTES.get(backend).ok_or_else(|| {
        format!(
            "production training target {:?} requested an identity for unknown backend {backend:?}",
            target.id
        )
    })?;
    mappings.get(&target.id).map(String::as_str).ok_or_else(|| {
        format!(
            "production training target {:?} (base {:?}, kernel {:?}) has no backend-local trainer in {backend} feature export",
            target.id, target.base_model, target.kernel
        )
    })
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
