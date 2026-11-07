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

#[cfg(test)]
mod tests {
    use super::*;

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
    }
}
