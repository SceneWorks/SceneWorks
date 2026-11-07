//! Backend-local trainer identity comes from the typed runtime capability export. The product
//! target supplies its logical id, base model and kernel; the export binds that id to a registered
//! native trainer on each backend.
use serde::Deserialize;
use std::collections::BTreeMap;
use std::sync::LazyLock;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TrainerFacts {
    schema_version: u32,
    trainer_mappings: BTreeMap<String, String>,
    snapshot: TrainerSnapshot,
}

#[derive(Deserialize)]
struct TrainerSnapshot {
    backend: String,
    trainer_capabilities: Vec<TrainerDescriptor>,
}

#[derive(Deserialize)]
struct TrainerDescriptor {
    id: String,
    backend: String,
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
        let facts: TrainerFacts =
            serde_json::from_str(source).expect("typed runtime trainer export");
        assert_eq!(facts.schema_version, 2, "trainer export schema");
        assert_eq!(facts.snapshot.backend, backend, "trainer export backend");
        let registered: std::collections::BTreeSet<_> = facts
            .snapshot
            .trainer_capabilities
            .iter()
            .filter(|trainer| trainer.backend == backend)
            .map(|trainer| trainer.id.as_str())
            .collect();
        assert_eq!(
            registered.len(),
            facts.snapshot.trainer_capabilities.len(),
            "duplicate or wrong-backend trainer"
        );
        // The export also names deliberately unavailable mappings (for example Krea Control on
        // MLX). Claimability is a separate feature fact; keep the product identity available to
        // validation without treating every mapping as a registered route.
        routes.insert(backend.to_owned(), facts.trainer_mappings);
    }
    routes
});

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
