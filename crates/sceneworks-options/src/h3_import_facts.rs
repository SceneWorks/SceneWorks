//! Pure imported-H3 declaration policy. Native code still verifies source plans and artifacts.
use serde_json::{Map, Value};
const MINIMAX_H3_IMPORT_FAMILY: &str = "minimax-h3";
const MINIMAX_H3_IMPORT_SOURCE_SHAPE: &str = "transformer_directory";
const MINIMAX_H3_ASSEMBLY_SCHEMA_VERSION: u64 = 1;
const MINIMAX_H3_BASE_PARTITION_MODEL: &str = "minimax_h3";
const MINIMAX_H3_REFERENCE_PARTITION_MODEL: &str = "minimax_h3_ref";
fn entry_str<'a>(entry: &'a Map<String, Value>, key: &str) -> Option<&'a str> {
    entry
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
}
fn checkpoint_plan_checkpoint_id(entry: &Map<String, Value>) -> Option<&str> {
    entry
        .get("importPlan")?
        .get("checkpointId")?
        .as_str()
        .map(str::trim)
        .filter(|id| !id.is_empty())
}
fn entry_assembly(entry: &Map<String, Value>) -> Option<&Map<String, Value>> {
    entry.get("importPlan")?.get("assembly")?.as_object()
}
/// Task compatibility declared independently of architecture and tensor shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiniMaxH3Partition {
    Fl2va,
    Ref2va,
    Hybrid,
}

impl MiniMaxH3Partition {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fl2va => "fl2va",
            Self::Ref2va => "ref2va",
            Self::Hybrid => "hybrid",
        }
    }

    pub fn supports_base(self) -> bool {
        matches!(self, Self::Fl2va | Self::Hybrid)
    }

    pub fn supports_references(self) -> bool {
        matches!(self, Self::Ref2va | Self::Hybrid)
    }

    pub fn from_entry(entry: &Map<String, Value>) -> Result<Self, String> {
        let partition = match entry_str(entry, "minimaxH3Partition") {
            Some("fl2va") => Self::Fl2va,
            Some("ref2va") => Self::Ref2va,
            Some("hybrid") => Self::Hybrid,
            _ => return Err("MiniMax-H3 task compatibility is ambiguous: explicitly declare minimaxH3Partition fl2va, ref2va, or hybrid; shapes and directory names do not prove compatibility".to_owned()),
        };
        let expected = if partition == Self::Ref2va {
            MINIMAX_H3_REFERENCE_PARTITION_MODEL
        } else {
            MINIMAX_H3_BASE_PARTITION_MODEL
        };
        if entry_str(entry, "baseModel") != Some(expected) {
            return Err(format!("MiniMax-H3 {} requires exact compatible baseModel {expected}; incompatible or missing base declaration", partition.as_str()));
        }
        Ok(partition)
    }
}

pub fn entry_is_imported_minimax_h3(entry: &Map<String, Value>) -> bool {
    entry_str(entry, "family") == Some(MINIMAX_H3_IMPORT_FAMILY)
        && entry_str(entry, "importSourceShape") == Some(MINIMAX_H3_IMPORT_SOURCE_SHAPE)
        && checkpoint_plan_checkpoint_id(entry).is_some()
        && entry_assembly(entry).is_some_and(|assembly| {
            assembly.get("schemaVersion").and_then(Value::as_u64)
                == Some(MINIMAX_H3_ASSEMBLY_SCHEMA_VERSION)
                && MiniMaxH3Partition::from_entry(entry).is_ok_and(|partition| {
                    assembly.get("partition").and_then(Value::as_str) == Some(partition.as_str())
                })
        })
}
