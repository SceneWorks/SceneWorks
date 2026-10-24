//! Compact provenance projections of the existing immutable checkpoint plan. Catalog and client
//! metadata are never generation-time authority. Unknown historical declarations stay unknown.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::{json, Map, Value};

use crate::checkpoint_import::ImportPlanV1;
use crate::checkpoint_plan_store::CheckpointPlanStore;

pub const MODEL_PROVENANCE_KEY: &str = "modelProvenance";
pub const MODEL_DECLARATION_KEY: &str = "modelDeclaration";

/// Called once by the import compiler, with the server-normalized import declaration. H3 license
/// owners are read from the trusted builtin component catalog, never from author-provided flags.
pub fn import_declaration_context(
    entry: &Map<String, Value>,
) -> Result<BTreeMap<String, Value>, String> {
    let mut normalized = Map::new();
    let h3 = crate::minimax_h3_import::entry_requires_minimax_h3_family(entry);
    let components = if h3 {
        crate::minimax_h3_import::MiniMaxH3Partition::from_entry(entry)?;
        let components = crate::minimax_h3_import::inherited_component_entries(entry);
        if components.is_empty() {
            return Err("inherited H3 component catalog is unavailable".into());
        }
        components
    } else {
        entry
            .get("baseModel")
            .and_then(Value::as_str)
            .and_then(crate::builtin_manifests::builtin_model_entry)
            .map(|base| {
                vec![crate::license_acknowledgments::component_license_metadata(
                    base,
                )]
            })
            .unwrap_or_default()
    };
    // Clear arbitrary author-provided ancestry even when no trusted base is known.
    crate::license_acknowledgments::apply_inherited_component_licenses(
        &mut normalized,
        &components,
    );
    let mut lineage = Map::new();
    if let Some(base) = entry
        .get("baseModel")
        .filter(|v| v.as_str().is_some_and(|s| !s.is_empty()))
    {
        lineage.insert("baseModel".into(), base.clone());
    }
    // Model IDs here identify license/component ancestry, not the imported denoiser's owner.
    lineage.insert(
        "componentLicenseOwners".into(),
        Value::Array(
            components
                .iter()
                .filter_map(|c| c.get("id").cloned())
                .collect(),
        ),
    );
    let mut semantics = Map::new();
    for key in [
        "family",
        "minimaxH3Partition",
        "importQuantFormat",
        "importComponentTier",
    ] {
        if let Some(value) = entry.get(key) {
            semantics.insert(key.into(), value.clone());
        }
    }
    Ok(BTreeMap::from([(
        MODEL_DECLARATION_KEY.into(),
        json!({
            "schemaVersion": 1,
            "baseLineage": lineage,
            "declaredSemantics": semantics,
            "inheritedLicenseAcknowledgments": normalized.get("inheritedLicenseAcknowledgments").cloned().unwrap_or_else(|| json!([])),
        }),
    )]))
}

/// A portable, compact explanation of a validated immutable plan. The full configuration stays
/// in the plan; assets carry only its content commitment and original source/base provenance.
pub fn checkpoint_model_provenance(plan: &ImportPlanV1, model_id: &str) -> Result<Value, String> {
    let reference = plan.plan_reference().map_err(|e| e.to_string())?;
    let sources = plan
        .layers
        .iter()
        .map(|layer| {
            Ok(json!({"layerId":layer.layer_id,"role":layer.role,
            "sha256":layer.source.content_digest().map_err(|e| e.to_string())?,
            "source":layer.source}))
        })
        .collect::<Result<Vec<Value>, String>>()?;
    let mut provenance = json!({"schemaVersion":1,"modelId":model_id,"family":plan.family,
        "plan":reference,"sources":sources,"inheritedLicenseAcknowledgments":[],"derivativeHistory":[]});
    if let Some(declaration) = plan.context.get(MODEL_DECLARATION_KEY) {
        for key in [
            "baseLineage",
            "declaredSemantics",
            "inheritedLicenseAcknowledgments",
        ] {
            provenance[key] = declaration[key].clone();
        }
    }
    if let Some(encoding) = plan.context.get("sourceEncoding") {
        let semantics = provenance
            .as_object_mut()
            .expect("object above")
            .entry("declaredSemantics")
            .or_insert_with(|| json!({}));
        semantics["sourceEncoding"] = encoding.clone();
    }
    if let Some(binding) = plan.context.get("transformerConfig") {
        // The source may legitimately differ from the base. Check its own binding, never JSON
        // equality against stock configuration, and do not duplicate its payload in every asset.
        crate::minimax_h3_import::bound_config(&plan.context)?;
        provenance["configuration"] =
            json!({"sha256":binding["sha256"],"provenance":binding["provenance"]});
    }
    Ok(provenance)
}

/// Resolve the record/plan commitment before projecting any provenance. Old imported entries with
/// no plan retain their route and return None. Old plans retain their hashes without invented
/// base/partition declarations. A new bound declaration cannot be overwritten by catalog edits.
pub fn resolved_model_provenance(
    data_dir: &Path,
    entry: &Map<String, Value>,
) -> Result<Option<Value>, String> {
    let Some(checkpoint_id) = crate::jobs_store::checkpoint_plan_checkpoint_id(entry) else {
        return Ok(None);
    };
    let store = CheckpointPlanStore::open(data_dir);
    let record = store.record(checkpoint_id).map_err(|e| e.to_string())?;
    let plan = store
        .plan(checkpoint_id, &record.plan.plan_id)
        .map_err(|e| e.to_string())?;
    record
        .validate_loaded_plan(&plan)
        .map_err(|e| e.to_string())?;
    let binding = entry
        .get("importPlan")
        .ok_or("missing import plan binding")?;
    if binding
        .get("planId")
        .and_then(Value::as_str)
        .is_some_and(|id| id != plan.plan_id)
        || binding
            .get("semanticDigest")
            .and_then(Value::as_str)
            .is_some_and(|digest| digest != record.plan.semantic_digest)
    {
        return Err("model entry no longer matches its immutable import plan".into());
    }
    validate_entry_declaration(&plan, entry)?;
    checkpoint_model_provenance(
        &plan,
        entry.get("id").and_then(Value::as_str).unwrap_or_default(),
    )
    .map(Some)
}

/// Shared by admission and the native provider's pre-load guard. A catalog edit cannot reinterpret
/// the same checkpoint as another partition, compatible base, or resident-component precision.
pub fn validate_entry_declaration(
    plan: &ImportPlanV1,
    entry: &Map<String, Value>,
) -> Result<(), String> {
    if let Some(declaration) = plan.context.get(MODEL_DECLARATION_KEY) {
        if declaration.get("schemaVersion").and_then(Value::as_f64) != Some(1.0) {
            return Err("unsupported immutable model declaration".into());
        }
        for (path, key) in [
            ("/baseLineage/baseModel", "baseModel"),
            (
                "/declaredSemantics/minimaxH3Partition",
                "minimaxH3Partition",
            ),
            (
                "/declaredSemantics/importComponentTier",
                "importComponentTier",
            ),
        ] {
            if declaration
                .pointer(path)
                .is_some_and(|bound| entry.get(key) != Some(bound))
            {
                return Err(format!(
                    "model {key} differs from the immutable import declaration"
                ));
            }
        }
    }
    Ok(())
}

/// Worker facts are mutable reports, not source authority. Every producer uses this same boundary;
/// an absent historical binding clears a forged claim instead of fabricating provenance.
pub fn stamp_asset_provenance(facts: &mut [Value], provenance: Option<&Value>) {
    for fact in facts {
        let Some(object) = fact.as_object_mut() else {
            continue;
        };
        object.remove(MODEL_PROVENANCE_KEY);
        for key in ["extra", "rawAdapterSettings"] {
            if let Some(nested) = object.get_mut(key).and_then(Value::as_object_mut) {
                nested.remove(MODEL_PROVENANCE_KEY);
            }
        }
        if let Some(provenance) = provenance {
            object.insert(MODEL_PROVENANCE_KEY.into(), provenance.clone());
        }
    }
}

/// Redeem only a concrete conversion receipt. No cache scan, guessed codec, or mutable metadata
/// is substituted for evidence of a derivative actually held by the producer.
pub fn verify_asset_derivative_provenance(
    data_dir: &Path,
    facts: &mut [Value],
) -> Result<(), String> {
    let mut store = None;
    let mut receipts: BTreeMap<String, Value> = BTreeMap::new();
    for fact in facts {
        let claim = fact
            .get_mut("rawAdapterSettings")
            .and_then(Value::as_object_mut)
            .and_then(|raw| raw.remove("checkpointDerivative"));
        let Some(claim) = claim else {
            continue;
        };
        let cache_key = claim
            .get("cacheKey")
            .and_then(Value::as_str)
            .ok_or("invalid checkpoint derivative identity")?;
        let provenance = fact
            .get_mut(MODEL_PROVENANCE_KEY)
            .ok_or("checkpoint derivative has no immutable source binding")?;
        let receipt = if let Some(receipt) = receipts.get(cache_key) {
            receipt.clone()
        } else {
            if store.is_none() {
                store = Some(
                    crate::checkpoint_derivative::CheckpointDerivativeStore::open(data_dir)
                        .map_err(|e| e.to_string())?,
                );
            }
            let receipt = store
                .as_ref()
                .expect("opened above")
                .verified_provenance_receipt(cache_key, provenance)
                .map_err(|e| e.to_string())?;
            receipts.insert(cache_key.to_owned(), receipt.clone());
            receipt
        };
        provenance["derivativeHistory"] = json!([receipt]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint_import::{CheckpointContainerV1, ImportLayerV1, SourceLocatorV1};

    fn plan() -> ImportPlanV1 {
        ImportPlanV1::new(
            format!("checkpoint-plan-{}", "1".repeat(64)),
            "ltx2",
            vec![ImportLayerV1 {
                layer_id: "dit".into(),
                role: "transformer".into(),
                target_path: "model.safetensors".into(),
                container: CheckpointContainerV1::Safetensors,
                source: SourceLocatorV1::linked("root-test", "model.safetensors", "a".repeat(64))
                    .unwrap(),
            }],
        )
        .unwrap()
    }

    #[test]
    fn old_plans_keep_identity_without_forging_historical_declarations() {
        let plan = plan();
        let before = plan.canonical_json().unwrap();
        let provenance = checkpoint_model_provenance(&plan, "old-import").unwrap();
        assert_eq!(provenance["sources"][0]["sha256"], "a".repeat(64));
        assert!(provenance.get("baseLineage").is_none());
        assert!(provenance.get("declaredSemantics").is_none());
        assert_eq!(plan.canonical_json().unwrap(), before);
        let decoded: ImportPlanV1 = serde_json::from_str(&before).unwrap();
        assert_eq!(
            decoded.plan_reference().unwrap(),
            plan.plan_reference().unwrap()
        );
    }

    #[test]
    fn declaration_and_provenance_roundtrip_are_immutable_content() {
        let mut plan = plan();
        let entry = json!({"baseModel":"ltx-base", "family":"ltx2"});
        plan.context = import_declaration_context(entry.as_object().unwrap()).unwrap();
        let committed = plan.semantic_digest().unwrap();
        let roundtrip: ImportPlanV1 =
            serde_json::from_str(&plan.canonical_json().unwrap()).unwrap();
        let provenance = checkpoint_model_provenance(&roundtrip, "custom").unwrap();
        assert_eq!(provenance["modelId"], "custom");
        assert_eq!(provenance["baseLineage"]["baseModel"], "ltx-base");
        plan.context.get_mut(MODEL_DECLARATION_KEY).unwrap()["baseLineage"]["baseModel"] =
            json!("another-base");
        assert_ne!(plan.semantic_digest().unwrap(), committed);
    }

    #[test]
    fn every_asset_kind_replaces_worker_claims_and_absent_evidence_stays_absent() {
        let kinds = ["image", "video", "audio", "document", "vector", "depth"];
        let mut facts: Vec<_> = kinds.iter().map(|kind| json!({"type":kind,"modelProvenance":{"forged":true},
            "extra":{"modelProvenance":{"forged":true}}, "rawAdapterSettings":{"modelProvenance":{}}})).collect();
        let trusted = checkpoint_model_provenance(&plan(), "actual-model").unwrap();
        stamp_asset_provenance(&mut facts, Some(&trusted));
        for fact in &facts {
            assert_eq!(fact[MODEL_PROVENANCE_KEY], trusted);
            assert!(fact["extra"].get(MODEL_PROVENANCE_KEY).is_none());
            assert!(fact["rawAdapterSettings"]
                .get(MODEL_PROVENANCE_KEY)
                .is_none());
        }
        stamp_asset_provenance(&mut facts, None);
        assert!(facts
            .iter()
            .all(|fact| fact.get(MODEL_PROVENANCE_KEY).is_none()));
    }

    #[test]
    fn new_h3_declaration_retains_trusted_component_obligations_not_author_flags() {
        let entry = json!({"family":"minimax-h3","baseModel":"minimax_h3_ref",
            "minimaxH3Partition":"ref2va","license":"MIT","requiresLicenseAcknowledgment":false,
            "inheritedLicenseAcknowledgments":[]});
        let context = import_declaration_context(entry.as_object().unwrap()).unwrap();
        let declaration = &context[MODEL_DECLARATION_KEY];
        let components = declaration["inheritedLicenseAcknowledgments"]
            .as_array()
            .unwrap();
        assert!(components
            .iter()
            .any(|component| component["modelId"] == "minimax_h3"
                && component["requiresLicenseAcknowledgment"] == true));
        assert_eq!(declaration["baseLineage"]["baseModel"], "minimax_h3_ref");
        assert_eq!(
            declaration["declaredSemantics"]["minimaxH3Partition"],
            "ref2va"
        );
    }

    #[test]
    fn catalog_and_plan_tampering_cannot_relabel_an_immutable_declaration() {
        use crate::checkpoint_import::{CheckpointCatalogRecordV1, CheckpointInventoryV1};
        let dir = tempfile::tempdir().unwrap();
        let mut plan = plan();
        plan.context =
            import_declaration_context(json!({"baseModel":"declared-base"}).as_object().unwrap())
                .unwrap();
        let checkpoint = "linked/root-test/model.safetensors";
        let record = CheckpointCatalogRecordV1::from_plan(checkpoint, &plan).unwrap();
        let inventory = CheckpointInventoryV1::new(vec![record.clone()]).unwrap();
        let plans_dir = dir.path().join("checkpoints/plans");
        std::fs::create_dir_all(&plans_dir).unwrap();
        let plan_path = plans_dir.join(format!("{}.json", plan.plan_id));
        std::fs::write(&plan_path, plan.canonical_json().unwrap()).unwrap();
        std::fs::write(
            dir.path().join("checkpoints/inventory.json"),
            inventory.canonical_json().unwrap(),
        )
        .unwrap();
        let mut entry = json!({"id":"custom","baseModel":"declared-base","importPlan":{
            "checkpointId":checkpoint,"planId":plan.plan_id,"semanticDigest":record.plan.semantic_digest}});
        assert!(
            resolved_model_provenance(dir.path(), entry.as_object().unwrap())
                .unwrap()
                .is_some()
        );
        entry["baseModel"] = json!("stock-replacement");
        assert!(
            resolved_model_provenance(dir.path(), entry.as_object().unwrap())
                .unwrap_err()
                .contains("immutable import declaration")
        );
        entry["baseModel"] = json!("declared-base");
        plan.context.get_mut(MODEL_DECLARATION_KEY).unwrap()["baseLineage"]["baseModel"] =
            json!("stock-replacement");
        std::fs::write(&plan_path, plan.canonical_json().unwrap()).unwrap();
        assert!(resolved_model_provenance(dir.path(), entry.as_object().unwrap()).is_err());
        assert_eq!(
            resolved_model_provenance(
                dir.path(),
                json!({"id":"pre-plan-import"}).as_object().unwrap()
            )
            .unwrap(),
            None
        );
    }

    #[test]
    fn job_provenance_migration_stays_unknown_and_frozen_binding_survives_restart() {
        use crate::jobs_store::{CreateJob, JobsStore};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("jobs.db");
        let store = JobsStore::new(&path);
        store.initialize().unwrap();
        let job = store
            .create_job(CreateJob {
                job_type: crate::contracts::JobType::ImageGenerate,
                project_id: None,
                project_name: None,
                payload: Map::new(),
                requested_gpu: "auto".into(),
                source_job_id: None,
                duplicate_of_job_id: None,
                attempts: 1,
                initial_status: None,
            })
            .unwrap();
        drop(store);
        let connection = rusqlite::Connection::open(&path).unwrap();
        connection
            .execute("alter table jobs drop column model_provenance_json", [])
            .unwrap();
        drop(connection);
        let store = JobsStore::new(&path);
        store.initialize().unwrap();
        assert_eq!(store.model_provenance(&job.id).unwrap(), None);
        let provenance = checkpoint_model_provenance(&plan(), "custom").unwrap();
        store.freeze_model_provenance(&job.id, &provenance).unwrap();
        let mut forged = provenance.clone();
        forged["sources"][0]["sha256"] = json!("b".repeat(64));
        assert!(store.freeze_model_provenance(&job.id, &forged).is_err());
        let retry = store
            .retry_job(
                &job.id,
                crate::jobs_store::RetryJob {
                    payload_changes: Map::new(),
                },
            )
            .unwrap();
        assert_eq!(
            store.model_provenance(&retry.id).unwrap(),
            Some(provenance.clone())
        );
        let duplicate = store
            .duplicate_job(
                &job.id,
                crate::jobs_store::DuplicateJob {
                    payload_changes: json!({"modelProvenance":forged})
                        .as_object()
                        .unwrap()
                        .clone(),
                    requested_gpu: None,
                },
            )
            .unwrap();
        assert_eq!(
            store.model_provenance(&duplicate.id).unwrap(),
            Some(provenance.clone())
        );
        let different_model = store
            .duplicate_job(
                &job.id,
                crate::jobs_store::DuplicateJob {
                    payload_changes: json!({"model":"other-model"}).as_object().unwrap().clone(),
                    requested_gpu: None,
                },
            )
            .unwrap();
        assert_eq!(store.model_provenance(&different_model.id).unwrap(), None);
        drop(store);
        let restarted = JobsStore::new(&path);
        restarted.initialize().unwrap();
        assert_eq!(
            restarted.model_provenance(&job.id).unwrap(),
            Some(provenance)
        );
    }

    #[test]
    fn managed_source_revision_and_hash_survive_without_claiming_linked_ownership() {
        use crate::checkpoint_import::ManagedProvenanceV1;
        let linked = plan();
        let mut managed = linked.clone();
        let revision = "1234567890abcdef1234567890abcdef12345678";
        let mut origin = ManagedProvenanceV1::of_source("huggingface");
        origin.version_id = Some(revision.into());
        origin.reference = Some(format!("author/fine-tune@{revision}"));
        origin.file_id = Some("model.safetensors".into());
        managed.layers[0].source =
            SourceLocatorV1::managed("my-install", "model.safetensors", "a".repeat(64), origin)
                .unwrap();
        let linked = checkpoint_model_provenance(&linked, "user-id").unwrap();
        let managed = checkpoint_model_provenance(&managed, "user-id").unwrap();
        assert_eq!(
            managed["sources"][0]["source"]["provenance"]["versionId"],
            revision
        );
        assert_eq!(
            managed["sources"][0]["sha256"],
            linked["sources"][0]["sha256"]
        );
        assert_eq!(
            managed["plan"]["semanticDigest"],
            linked["plan"]["semanticDigest"]
        );
        assert_ne!(
            managed["plan"]["sourceBindingIdentity"],
            linked["plan"]["sourceBindingIdentity"]
        );
        assert_eq!(managed["sources"][0]["source"]["kind"], "managed");
        assert_eq!(linked["sources"][0]["source"]["kind"], "linked");
        assert_eq!(managed["modelId"], "user-id");
    }

    #[test]
    fn provenance_schema_accepts_legacy_unknowns_but_rejects_invalid_hashes_and_versions() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../packages/schemas/model-manifest.schema.json");
        let schema: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let schema = json!({"$defs":schema["$defs"],"$ref":"#/$defs/modelProvenance"});
        let validator = jsonschema::validator_for(&schema).unwrap();
        let mut value = checkpoint_model_provenance(&plan(), "legacy-import").unwrap();
        assert!(validator.is_valid(&value));
        value["sources"][0]["sha256"] = json!("mutable-main");
        assert!(!validator.is_valid(&value));
        value["sources"][0]["sha256"] = json!("a".repeat(64));
        value["schemaVersion"] = json!(2);
        assert!(!validator.is_valid(&value));
    }

    #[test]
    fn recomputing_mutable_assembly_cannot_change_a_bound_h3_partition() {
        use crate::minimax_h3_import::{
            config_binding, stamp_imported_minimax_h3_entry, verify_entry_assembly,
            MiniMaxH3ImportAssembly,
        };
        let mut plan = plan();
        plan.family = "minimax-h3".into();
        let mut entry = json!({"id":"my-import","family":"minimax-h3","baseModel":"minimax_h3_ref","minimaxH3Partition":"ref2va"}).as_object().unwrap().clone();
        plan.context = import_declaration_context(&entry).unwrap();
        plan.context.insert(
            "transformerConfig".into(),
            config_binding(
                crate::minimax_h3_codec::H3Config::published().0,
                json!({"kind":"source"}),
            ),
        );
        entry.insert(
            "importPlan".into(),
            json!({"planId":plan.plan_id,"semanticDigest":plan.semantic_digest().unwrap()}),
        );
        let assembly = MiniMaxH3ImportAssembly::from_declared_plan(&plan, &entry).unwrap();
        stamp_imported_minimax_h3_entry(&mut entry, &assembly);
        verify_entry_assembly(&entry, &plan).unwrap();
        entry.insert("baseModel".into(), json!("minimax_h3"));
        entry.insert("minimaxH3Partition".into(), json!("hybrid"));
        let forged = MiniMaxH3ImportAssembly::from_declared_plan(&plan, &entry).unwrap();
        stamp_imported_minimax_h3_entry(&mut entry, &forged);
        assert!(verify_entry_assembly(&entry, &plan)
            .unwrap_err()
            .contains("immutable import declaration"));
    }

    #[test]
    fn declaration_schema_and_plan_serde_agree_without_closing_unrelated_context() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../packages/schemas/checkpoint-import.schema.json");
        let schema: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let validator = jsonschema::validator_for(&schema).unwrap();
        let mut document = serde_json::to_value(plan()).unwrap();
        assert!(validator.is_valid(&document));
        let valid = json!({"schemaVersion":1,"baseLineage":{},"declaredSemantics":{},"inheritedLicenseAcknowledgments":[]});
        for (key, value) in [
            ("schemaVersion", json!(2)),
            ("baseLineage", json!([])),
            ("declaredSemantics", json!(null)),
            ("inheritedLicenseAcknowledgments", json!([false])),
            ("schemaVersion", json!(1.0)),
        ] {
            let mut declaration = valid.clone();
            declaration[key] = value;
            document["context"] = json!({"modelDeclaration":declaration,"unrelatedLoaderContext":{"opaque":"retained"}});
            assert_eq!(
                serde_json::from_value::<ImportPlanV1>(document.clone()).is_ok(),
                validator.is_valid(&document),
                "{key}: {document}"
            );
        }
        let decoded: ImportPlanV1 = serde_json::from_value(document).unwrap();
        assert_eq!(
            decoded.context["unrelatedLoaderContext"],
            json!({"opaque":"retained"})
        );
    }
}
