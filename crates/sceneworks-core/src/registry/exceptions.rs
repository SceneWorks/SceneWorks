//! Product decisions for deliberate MLX/Candle differences. These are not inferred from
//! runtime routes: changing a route does not revoke an owner's approved product decision.
use serde::Deserialize;
use std::collections::BTreeSet;
use std::sync::LazyLock;

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ApprovedBackendException {
    pub id: String,
    pub category: String,
    pub decision_type: DecisionType,
    pub evidence: String,
    pub approver: String,
    pub authority: String,
    pub approved_date: String,
    pub user_facing_behavior: String,
    pub revisit_condition: String,
    pub cells: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionType {
    TechnicalBlocker,
    LicensingBlocker,
    HardwareFloor,
    SequencingChoice,
    ProductDecision,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct DecisionStore {
    schema_version: u32,
    authorized_approvers: Vec<AuthorizedApprover>,
    records: Vec<ApprovedBackendException>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AuthorizedApprover {
    name: String,
    authority: String,
}

static APPROVED: LazyLock<Vec<ApprovedBackendException>> = LazyLock::new(|| {
    let store: DecisionStore = serde_json::from_str(include_str!(
        "../../../../config/manifests/registry/approved-exceptions.json"
    ))
    .expect("typed approved backend exceptions");
    validate(&store).expect("valid approved backend exceptions");
    store.records
});

pub fn approved_backend_exceptions() -> &'static [ApprovedBackendException] {
    &APPROVED
}

fn validate(store: &DecisionStore) -> Result<(), String> {
    if store.schema_version != 2 {
        return Err(format!(
            "unsupported exception schema {}",
            store.schema_version
        ));
    }
    let authorities: BTreeSet<_> = store
        .authorized_approvers
        .iter()
        .map(|a| (a.name.as_str(), a.authority.as_str()))
        .collect();
    if authorities.len() != store.authorized_approvers.len() {
        return Err("duplicate approval authority".into());
    }
    let mut ids = BTreeSet::new();
    let mut cells = BTreeSet::new();
    for record in &store.records {
        if !ids.insert(record.id.as_str()) {
            return Err(format!("duplicate decision {}", record.id));
        }
        if [
            &record.id,
            &record.category,
            &record.evidence,
            &record.approver,
            &record.authority,
            &record.approved_date,
            &record.user_facing_behavior,
            &record.revisit_condition,
        ]
        .iter()
        .any(|value| value.trim().is_empty())
            || record.cells.is_empty()
        {
            return Err(format!("incomplete decision {}", record.id));
        }
        if !authorities.contains(&(record.approver.as_str(), record.authority.as_str())) {
            return Err(format!("unapproved decision {}", record.id));
        }
        if !valid_iso_date(&record.approved_date) {
            return Err(format!("invalid approval date for {}", record.id));
        }
        if !matches!(
            record.category.as_str(),
            "operation"
                | "conditioning"
                | "adapter"
                | "precision"
                | "guidance"
                | "preview"
                | "training"
                | "utility"
        ) {
            return Err(format!("unknown decision category {}", record.category));
        }
        for cell in &record.cells {
            if category_for_cell(cell) != Some(record.category.as_str()) {
                return Err(format!("decision {} has mismatched cell {cell}", record.id));
            }
            if !cells.insert(cell.as_str()) {
                return Err(format!("duplicate exception cell {cell}"));
            }
        }
    }
    Ok(())
}

fn valid_iso_date(value: &str) -> bool {
    let mut parts = value.split('-');
    let (Some(year), Some(month), Some(day), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return false;
    };
    if year.len() != 4 || month.len() != 2 || day.len() != 2 {
        return false;
    }
    let (Ok(year), Ok(month), Ok(day)) = (
        year.parse::<u32>(),
        month.parse::<u32>(),
        day.parse::<u32>(),
    ) else {
        return false;
    };
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let max_day = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return false,
    };
    (1..=max_day).contains(&day)
}

fn category_for_cell(cell: &str) -> Option<&'static str> {
    if cell.starts_with("gpuJobTypes/") {
        return Some("utility");
    }
    if cell.starts_with("trainingKernels/") {
        return Some("training");
    }
    if !cell.starts_with("models/") {
        return None;
    }
    if cell.ends_with("/preview") {
        return Some("preview");
    }
    [
        ("/operationAndMode/", "operation"),
        ("/conditioningShape/", "conditioning"),
        ("/userAdapters/", "adapter"),
        ("/precisionTier/", "precision"),
        ("/guidanceMethod/", "guidance"),
    ]
    .into_iter()
    .find_map(|(segment, category)| cell.contains(segment).then_some(category))
}

/// Decisions are valid only while the named MLX-only product surface still exists. The
/// structural validator above protects approval provenance; this check joins that provenance to
/// the compiled product policy and native descriptor facts used by the actual router.
#[cfg(test)]
fn validate_live_cells(records: &[ApprovedBackendException]) -> Result<(), String> {
    use crate::registry::{builtin_model_spec, RoutePolicy};

    for record in records {
        for cell in &record.cells {
            let ["models", id, axis, value] = cell.split('/').collect::<Vec<_>>()[..] else {
                return Err(format!("unrecognized approved cell {cell}"));
            };
            let spec =
                builtin_model_spec(id).ok_or_else(|| format!("approved model {id} is absent"))?;
            let policy = spec
                .routing
                .as_ref()
                .ok_or_else(|| format!("approved model {id} has no product route"))?;
            let mlx_video = spec.video_routes.get("mlx").into_iter().flatten();
            let candle_video = spec.video_routes.get("candle").into_iter().flatten();
            let supported = match axis {
                "operationAndMode" => {
                    mlx_video.clone().any(|route| route.mode == value)
                        && !candle_video.clone().any(|route| route.mode == value)
                }
                "conditioningShape" => {
                    mlx_video
                        .clone()
                        .flat_map(|route| &route.engines)
                        .any(|engine| engine.conditioning.iter().any(|kind| kind == value))
                        && !candle_video
                            .clone()
                            .flat_map(|route| &route.engines)
                            .any(|engine| engine.conditioning.iter().any(|kind| kind == value))
                }
                "userAdapters" => {
                    let has = |backend: &str| {
                        spec.video_routes
                            .get(backend)
                            .into_iter()
                            .flatten()
                            .flat_map(|route| &route.engines)
                            .any(|engine| match value {
                                "lora" => engine.supports_lora,
                                "lokr" => engine.supports_lokr,
                                _ => false,
                            })
                    };
                    has("mlx") && !has("candle")
                }
                "precisionTier" => match policy {
                    RoutePolicy::Image {
                        mlx_routed: true,
                        candle_quant: false,
                        ..
                    } => spec.routes.get("mlx").is_some_and(|route| {
                        route.supported_quants.iter().any(|tier| tier == value)
                    }),
                    RoutePolicy::Video { .. } => {
                        let mlx_tier = mlx_video
                            .clone()
                            .flat_map(|route| &route.engines)
                            .any(|engine| engine.supported_quants.iter().any(|tier| tier == value));
                        let candle_tier = candle_video
                            .clone()
                            .flat_map(|route| &route.engines)
                            .any(|engine| engine.supported_quants.iter().any(|tier| tier == value));
                        (mlx_tier && !candle_tier)
                            || (value == "bf16"
                                && mlx_video.clone().next().is_some()
                                && candle_video.clone().next().is_none())
                    }
                    _ => false,
                },
                "guidanceMethod" => {
                    spec.routes.get("mlx").is_some_and(|route| {
                        route
                            .supported_guidance_methods
                            .iter()
                            .any(|method| method == value)
                    }) && !spec.routes.get("candle").is_some_and(|route| {
                        route
                            .supported_guidance_methods
                            .iter()
                            .any(|method| method == value)
                            || (value == "cfg" && policy.candle_routed())
                    })
                }
                _ => false,
            };
            if !supported {
                return Err(format!("approved cell is no longer MLX-only: {cell}"));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{builtin_model_spec, RoutePolicy};

    #[test]
    fn approved_product_decisions_keep_the_exact_reviewed_surface() {
        let records = approved_backend_exceptions();
        assert_eq!(records.len(), 10);
        assert_eq!(
            records
                .iter()
                .map(|record| record.cells.len())
                .sum::<usize>(),
            42
        );
        assert_eq!(
            records
                .iter()
                .filter(|record| matches!(record.decision_type, DecisionType::ProductDecision))
                .count(),
            4
        );
        assert!(records.iter().any(|record| record.id
            == "epic-18803-eros-candle-operation-withdrawal"
            && record
                .cells
                .contains(&"models/ltx_2_3_eros/operationAndMode/text_to_video".to_owned())));
        let cfg_pp = records
            .iter()
            .find(|record| record.id == "epic-7434-cfg-pp-guidance-sequencing")
            .unwrap();
        assert_eq!(cfg_pp.authority, "epic-7434");
        assert!(cfg_pp.evidence.contains("sc-8257"));
        assert_eq!(cfg_pp.cells.len(), 4);
        assert!(cfg_pp
            .cells
            .iter()
            .all(|cell| cell.ends_with("/guidanceMethod/cfg_pp")));
    }

    /// Every residual cell is owned by its live specialist authority: Krea Realtime by epic-8433,
    /// the Eros Candle withdrawal by epic-18803, CFG++ by epic-7434 (tracked as sc-8257), and the
    /// remaining precision sequencing by epic-9083.
    #[test]
    fn residual_cells_are_owned_by_their_live_specialist_authority() {
        for record in approved_backend_exceptions() {
            for cell in &record.cells {
                let expected = if cell.starts_with("models/krea_realtime_14b/") {
                    "epic-8433"
                } else if cell.starts_with("models/ltx_2_3_eros/") {
                    "epic-18803"
                } else {
                    match record.category.as_str() {
                        "guidance" => "epic-7434",
                        "precision" => "epic-9083",
                        other => panic!("{cell}: no specialist owner for category {other}"),
                    }
                };
                assert_eq!(record.authority, expected, "{cell}");
            }
            if record.category == "guidance" {
                assert!(record.evidence.contains("sc-8257"), "{}", record.id);
            }
        }
    }

    #[test]
    fn malformed_or_duplicate_decisions_are_rejected() {
        let mut store: DecisionStore = serde_json::from_str(include_str!(
            "../../../../config/manifests/registry/approved-exceptions.json"
        ))
        .unwrap();
        let duplicate = store.records[1].cells[0].clone();
        store.records[0].cells.push(duplicate);
        assert!(validate(&store).is_err());
        store.records[0].cells.pop();
        store.records[0].authority = "unapproved".into();
        assert!(validate(&store).is_err());

        // A decision with blank evidence is incomplete, and an unknown decision type is refused
        // when the store is read.
        let mut store: DecisionStore = serde_json::from_str(include_str!(
            "../../../../config/manifests/registry/approved-exceptions.json"
        ))
        .unwrap();
        store.records[0].evidence = "  ".into();
        assert!(validate(&store).is_err(), "blank evidence was admitted");
        let mut raw: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../config/manifests/registry/approved-exceptions.json"
        ))
        .unwrap();
        raw["records"][0]["decisionType"] = serde_json::json!("capability_axis");
        assert!(
            serde_json::from_value::<DecisionStore>(raw).is_err(),
            "unknown decision type was admitted"
        );
    }

    #[test]
    fn approval_provenance_and_live_cell_mutations_fail_closed() {
        let raw = include_str!("../../../../config/manifests/registry/approved-exceptions.json");
        let mut store: DecisionStore = serde_json::from_str(raw).unwrap();
        validate(&store).unwrap();
        validate_live_cells(&store.records).unwrap();

        let original_date = store.records[0].approved_date.clone();
        store.records[0].approved_date = "2026-02-30".into();
        assert!(
            validate(&store).is_err(),
            "invalid calendar date was admitted"
        );
        store.records[0].approved_date = original_date;

        let original_authority = store.records[0].authority.clone();
        store.records[0].authority = "team:unapproved".into();
        assert!(
            validate(&store).is_err(),
            "unapproved authority was admitted"
        );
        store.records[0].authority = original_authority;

        let original_category = store.records[0].category.clone();
        store.records[0].category = "operation".into();
        assert!(
            validate(&store).is_err(),
            "wrong cell category was admitted"
        );
        store.records[0].category = original_category;

        let original_cell = store.records[0].cells[0].clone();
        store.records[0].cells[0] = "models/does-not-exist/precisionTier/q4".into();
        assert!(
            validate_live_cells(&store.records).is_err(),
            "unknown product cell was admitted"
        );
        store.records[0].cells[0] = original_cell.clone();

        // FLUX.1 Schnell q4 already has an admitted Candle product route.
        store.records[0].cells[0] = "models/flux_schnell/precisionTier/q4".into();
        assert!(
            validate_live_cells(&store.records).is_err(),
            "non-residual product cell was admitted"
        );
        store.records[0].cells[0] = original_cell;
        store.records[5].cells[0] = "models/sdxl/guidanceMethod/cfg".into();
        assert!(
            validate_live_cells(&store.records).is_err(),
            "shared default CFG was admitted as a gap"
        );
    }

    #[test]
    fn every_approved_cell_matches_typed_route_and_native_capability() {
        let mut checked = 0;
        for decision in approved_backend_exceptions() {
            for cell in &decision.cells {
                let ["models", id, axis, value] = cell.split('/').collect::<Vec<_>>()[..] else {
                    panic!("unexpected approved cell {cell}");
                };
                let spec = builtin_model_spec(id).unwrap_or_else(|| panic!("missing spec {id}"));
                let policy = spec
                    .routing
                    .as_ref()
                    .expect("approved model has route policy");
                match axis {
                    "operationAndMode" => {
                        assert!(policy.mlx_routed() && !policy.candle_routed(), "{cell}");
                        assert!(
                            spec.video_routes
                                .get("mlx")
                                .into_iter()
                                .flatten()
                                .any(|r| r.mode == value),
                            "{cell}"
                        );
                        assert!(
                            !spec
                                .video_routes
                                .get("candle")
                                .into_iter()
                                .flatten()
                                .any(|r| r.mode == value),
                            "{cell}"
                        );
                    }
                    "conditioningShape" => {
                        assert!(policy.mlx_routed() && !policy.candle_routed(), "{cell}");
                        assert!(
                            spec.video_routes
                                .get("mlx")
                                .into_iter()
                                .flatten()
                                .flat_map(|r| &r.engines)
                                .any(|e| e.conditioning.iter().any(|c| c == value)),
                            "{cell}"
                        );
                    }
                    "userAdapters" => {
                        assert!(policy.mlx_routed() && !policy.candle_routed(), "{cell}");
                        assert!(
                            spec.video_routes
                                .get("mlx")
                                .into_iter()
                                .flatten()
                                .flat_map(|r| &r.engines)
                                .any(|e| match value {
                                    "lora" => e.supports_lora,
                                    "lokr" => e.supports_lokr,
                                    _ => false,
                                }),
                            "{cell}"
                        );
                    }
                    "precisionTier" => match policy {
                        RoutePolicy::Image {
                            mlx_routed: true,
                            candle_quant: false,
                            ..
                        } => {
                            assert!(spec.routes.contains_key("mlx"), "{cell}");
                        }
                        RoutePolicy::Video {
                            video_mlx_routed: true,
                            candle_video_routed: false,
                            ..
                        } => {
                            assert!(spec.video_routes.contains_key("mlx"), "{cell}");
                        }
                        _ => panic!("approved precision cell disagrees with route policy: {cell}"),
                    },
                    "guidanceMethod" => {
                        let mlx = spec.routes.get("mlx").unwrap_or_else(|| panic!("{cell}"));
                        let candle = spec
                            .routes
                            .get("candle")
                            .unwrap_or_else(|| panic!("{cell}"));
                        assert!(
                            mlx.supported_guidance_methods.iter().any(|g| g == value),
                            "{cell}"
                        );
                        assert!(
                            !candle.supported_guidance_methods.iter().any(|g| g == value),
                            "{cell}"
                        );
                    }
                    _ => panic!("unhandled approved cell {cell}"),
                }
                checked += 1;
            }
        }
        assert_eq!(checked, 42);
    }
}
