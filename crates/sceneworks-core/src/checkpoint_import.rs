//! Versioned, backend-neutral checkpoint-import contracts.
//!
//! Inventory rows retain a compact [`ImportPlanSummaryV1`] plus a stable
//! [`ImportPlanReferenceV1`]. The immutable, complete [`ImportPlanV1`] is a
//! separate atomically published document, so catalog records never duplicate
//! individual layer locations.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
};

use serde::{
    de::{DeserializeOwned, Error as _, MapAccess, Visitor},
    ser::SerializeStruct,
    Deserialize, Deserializer, Serialize, Serializer,
};
use serde_json::value::RawValue;
use sha2::{Digest, Sha256};

/// Wire version shared by all checkpoint-import contracts in this module.
pub const CHECKPOINT_IMPORT_CONTRACT_VERSION: u32 = 1;

const LOCATOR_SEMANTIC_DOMAIN: &str = "sceneworks.checkpoint-import.v1.locator-semantic";
const LOCATOR_BINDING_DOMAIN: &str = "sceneworks.checkpoint-import.v1.locator-binding";
const PLAN_SEMANTIC_DOMAIN: &str = "sceneworks.checkpoint-import.v1.plan-semantic";
const PLAN_BINDING_DOMAIN: &str = "sceneworks.checkpoint-import.v1.plan-binding";

/// Error returned when a checkpoint-import document is malformed or too new.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckpointImportContractError(pub String);

impl std::fmt::Display for CheckpointImportContractError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for CheckpointImportContractError {}

fn invalid(message: impl Into<String>) -> CheckpointImportContractError {
    CheckpointImportContractError(message.into())
}

fn validate_version(found: u32) -> Result<(), CheckpointImportContractError> {
    if found != CHECKPOINT_IMPORT_CONTRACT_VERSION {
        return Err(invalid(format!(
            "checkpoint-import schema version {found} is unsupported; recompile/rescan required"
        )));
    }
    Ok(())
}

/// Maximum supported nesting for a v1 body duplicate scan. `serde_json` applies
/// its own equivalent recursion guard while capturing [`RawValue`]; this second
/// explicit bound keeps the contract-owned traversal deterministic too.
const CHECKPOINT_IMPORT_MAX_JSON_DEPTH: usize = 128;

/// A JSON object whose decoded keys remain ordered and whose values stay as raw
/// tokens. Unlike `serde_json::Value`, this does not collapse duplicate keys or
/// convert numbers through `f64`.
struct RawObject(Vec<(String, Box<RawValue>)>);

impl<'de> Deserialize<'de> for RawObject {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_map(RawObjectVisitor)
    }
}

struct RawObjectVisitor;

impl<'de> Visitor<'de> for RawObjectVisitor {
    type Value = RawObject;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON object")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut fields = Vec::new();
        while let Some((key, value)) = map.next_entry::<String, Box<RawValue>>()? {
            fields.push((key, value));
        }
        Ok(RawObject(fields))
    }
}

fn raw_object(value: &RawValue) -> Option<RawObject> {
    serde_json::from_str(value.get()).ok()
}

fn raw_array(value: &RawValue) -> Option<Vec<Box<RawValue>>> {
    serde_json::from_str(value.get()).ok()
}

fn object_values<'a>(object: &'a RawObject, key: &'a str) -> impl Iterator<Item = &'a RawValue> {
    object
        .0
        .iter()
        .filter_map(move |(candidate, value)| (candidate == key).then_some(value.as_ref()))
}

/// Parses a syntactically valid JSON number as an exact, nonnegative u32.
///
/// The decimal point and exponent are applied to the significand as decimal
/// digits, so representations such as `1.0`, `10e-1`, and `0.10e1` agree with
/// JSON Schema integer equality without any floating-point or range leakage.
fn exact_json_u32(token: &str) -> Option<u32> {
    if token.starts_with('-') {
        return None;
    }

    let (significand, exponent_token) = token
        .split_once(['e', 'E'])
        .map_or((token, None), |(left, right)| (left, Some(right)));
    let (integer, fraction) = significand
        .split_once('.')
        .map_or((significand, ""), |(left, right)| (left, right));
    if integer.is_empty()
        || !integer.bytes().all(|byte| byte.is_ascii_digit())
        || !fraction.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    let mut digits = String::with_capacity(integer.len() + fraction.len());
    digits.push_str(integer);
    digits.push_str(fraction);
    if digits.bytes().all(|byte| byte == b'0') {
        return Some(0);
    }

    let exponent = match exponent_token {
        None => 0_i64,
        Some(token) => {
            let (negative, magnitude) = match token.as_bytes().first() {
                Some(b'+') => (false, &token[1..]),
                Some(b'-') => (true, &token[1..]),
                _ => (false, token),
            };
            if magnitude.is_empty() || !magnitude.bytes().all(|byte| byte.is_ascii_digit()) {
                return None;
            }
            let magnitude = magnitude.bytes().try_fold(0_i64, |value, digit| {
                value.checked_mul(10)?.checked_add(i64::from(digit - b'0'))
            })?;
            if negative {
                magnitude.checked_neg()?
            } else {
                magnitude
            }
        }
    };
    let fraction_len = i64::try_from(fraction.len()).ok()?;
    let decimal_shift = exponent.checked_sub(fraction_len)?;
    let significant = digits.trim_start_matches('0');

    let integer_digits = if decimal_shift < 0 {
        let removed = usize::try_from(decimal_shift.checked_neg()?).ok()?;
        if removed > significant.len()
            || !significant[significant.len() - removed..]
                .bytes()
                .all(|byte| byte == b'0')
        {
            return None;
        }
        &significant[..significant.len() - removed]
    } else {
        significant
    };
    let integer_digits = integer_digits.trim_start_matches('0');
    if integer_digits.is_empty() {
        return Some(0);
    }
    let appended_zeros = usize::try_from(decimal_shift.max(0)).ok()?;
    if integer_digits.len().checked_add(appended_zeros)? > 10 {
        return None;
    }
    let mut value = integer_digits.bytes().try_fold(0_u64, |value, digit| {
        value.checked_mul(10)?.checked_add(u64::from(digit - b'0'))
    })?;
    for _ in 0..appended_zeros {
        value = value.checked_mul(10)?;
    }
    u32::try_from(value).ok()
}

fn deserialize_schema_version<'de, D: Deserializer<'de>>(deserializer: D) -> Result<u32, D::Error> {
    struct SupportedSchemaVersionVisitor;

    impl<'de> Visitor<'de> for SupportedSchemaVersionVisitor {
        type Value = u32;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("an exact checkpoint-import u32 schemaVersion")
        }

        fn visit_u64<E: serde::de::Error>(self, value: u64) -> Result<Self::Value, E> {
            u32::try_from(value)
                .map_err(|_| E::custom("checkpoint-import schemaVersion must be a u32"))
        }

        fn visit_i64<E: serde::de::Error>(self, value: i64) -> Result<Self::Value, E> {
            u32::try_from(value)
                .map_err(|_| E::custom("checkpoint-import schemaVersion must be a u32"))
        }

        fn visit_f64<E: serde::de::Error>(self, value: f64) -> Result<Self::Value, E> {
            if value.is_finite()
                && value >= 0.0
                && value <= f64::from(u32::MAX)
                && value.fract() == 0.0
            {
                Ok(value as u32)
            } else {
                Err(E::custom("checkpoint-import schemaVersion must be a u32"))
            }
        }
    }

    // Exact lexical acceptance and range checks already happened in the raw
    // envelope preflight. This visitor only lets serde's internally tagged enum
    // content representation carry accepted spellings such as `1.0` into v1.
    deserializer.deserialize_any(SupportedSchemaVersionVisitor)
}

fn deserialize_layer_count<'de, D: Deserializer<'de>>(deserializer: D) -> Result<u32, D::Error> {
    let raw = Box::<RawValue>::deserialize(deserializer)?;
    exact_json_u32(raw.get())
        .ok_or_else(|| D::Error::custom("checkpoint-import layerCount must be a u32"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VersionedSurface {
    CheckpointInventory,
    SourceLocator,
    ImportPlan,
    PlanReference,
    PlanSummary,
    CatalogRecord,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum VersionDiagnostic {
    Duplicate,
    Unsupported(u32),
    Invalid,
}

impl VersionDiagnostic {
    fn precedence_key(&self) -> (u8, u32) {
        match self {
            Self::Duplicate => (0, 0),
            Self::Unsupported(found) => (1, *found),
            Self::Invalid => (2, 0),
        }
    }

    fn into_contract_error(self) -> CheckpointImportContractError {
        match self {
            Self::Duplicate => {
                invalid("checkpoint-import JSON contains duplicate object key `schemaVersion`")
            }
            Self::Unsupported(found) => invalid(format!(
                "checkpoint-import schema version {found} is unsupported; recompile/rescan required"
            )),
            Self::Invalid => invalid("checkpoint-import schemaVersion must be a u32"),
        }
    }
}

fn preflight_versioned_tree(
    value: &RawValue,
    surface: VersionedSurface,
) -> Result<(), VersionDiagnostic> {
    // The current envelope owns precedence over everything inside it. Once it
    // selects supported v1, inspect every schema-defined child occurrence and
    // choose a diagnostic by kind/value rather than input order. Ordinary body
    // validation is deliberately deferred until the whole version graph passes.
    let object = raw_object(value).ok_or(VersionDiagnostic::Invalid)?;
    let versions: Vec<_> = object_values(&object, "schemaVersion").collect();
    if versions.len() > 1 {
        return Err(VersionDiagnostic::Duplicate);
    }
    let version = versions
        .first()
        .and_then(|value| exact_json_u32(value.get()))
        .ok_or(VersionDiagnostic::Invalid)?;
    if version != CHECKPOINT_IMPORT_CONTRACT_VERSION {
        return Err(VersionDiagnostic::Unsupported(version));
    }

    let mut nested_diagnostics = Vec::new();
    let mut preflight_child = |child: &RawValue, child_surface: VersionedSurface| {
        if let Err(diagnostic) = preflight_versioned_tree(child, child_surface) {
            nested_diagnostics.push(diagnostic);
        }
    };

    match surface {
        VersionedSurface::ImportPlan => {
            for layers in object_values(&object, "layers") {
                if let Some(layers) = raw_array(layers) {
                    for layer in layers {
                        if let Some(layer) = raw_object(&layer) {
                            for source in object_values(&layer, "source") {
                                preflight_child(source, VersionedSurface::SourceLocator);
                            }
                        }
                    }
                }
            }
        }
        VersionedSurface::CatalogRecord => {
            for plan in object_values(&object, "plan") {
                preflight_child(plan, VersionedSurface::PlanReference);
            }
            for summary in object_values(&object, "summary") {
                preflight_child(summary, VersionedSurface::PlanSummary);
            }
        }
        VersionedSurface::CheckpointInventory => {
            for records in object_values(&object, "records") {
                if let Some(records) = raw_array(records) {
                    for record in records {
                        preflight_child(&record, VersionedSurface::CatalogRecord);
                    }
                }
            }
        }
        VersionedSurface::SourceLocator
        | VersionedSurface::PlanReference
        | VersionedSurface::PlanSummary => {}
    }

    nested_diagnostics
        .into_iter()
        .min_by_key(VersionDiagnostic::precedence_key)
        .map_or(Ok(()), Err)
}

fn collect_duplicate_keys(
    value: &RawValue,
    depth: usize,
    duplicates: &mut BTreeSet<String>,
) -> Result<(), CheckpointImportContractError> {
    if depth > CHECKPOINT_IMPORT_MAX_JSON_DEPTH {
        return Err(invalid(format!(
            "checkpoint-import JSON exceeds nesting depth limit of {CHECKPOINT_IMPORT_MAX_JSON_DEPTH}"
        )));
    }
    if let Some(object) = raw_object(value) {
        let mut counts = BTreeMap::new();
        for (key, child) in &object.0 {
            *counts.entry(key.as_str()).or_insert(0_u32) += 1;
            collect_duplicate_keys(child, depth + 1, duplicates)?;
        }
        duplicates.extend(
            counts
                .into_iter()
                .filter(|(_, count)| *count > 1)
                .map(|(key, _)| key.to_owned()),
        );
    } else if let Some(values) = raw_array(value) {
        for child in values {
            collect_duplicate_keys(&child, depth + 1, duplicates)?;
        }
    }
    Ok(())
}

fn reject_duplicate_keys(value: &RawValue) -> Result<(), CheckpointImportContractError> {
    let mut duplicates = BTreeSet::new();
    collect_duplicate_keys(value, 0, &mut duplicates)?;
    if let Some(key) = duplicates.into_iter().next() {
        return Err(invalid(format!(
            "checkpoint-import JSON contains duplicate object key `{key}`"
        )));
    }
    Ok(())
}

/// Runs recursive envelope preflight and whole-document duplicate-key rejection
/// before selecting a v1 body decoder. An unambiguous future version therefore
/// wins over all diagnostics belonging to that version-specific body, while a
/// supported graph reaches strict typed decoding only after every key is unique.
fn deserialize_versioned_v1<'de, D, T>(
    deserializer: D,
    surface: VersionedSurface,
) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: DeserializeOwned,
{
    let value = Box::<RawValue>::deserialize(deserializer)?;
    preflight_versioned_tree(&value, surface)
        .map_err(VersionDiagnostic::into_contract_error)
        .map_err(D::Error::custom)?;
    reject_duplicate_keys(&value).map_err(D::Error::custom)?;
    serde_json::from_str(value.get()).map_err(D::Error::custom)
}

fn checked_layer_count(layer_count: usize) -> Result<u32, CheckpointImportContractError> {
    u32::try_from(layer_count)
        .map_err(|_| invalid("import plan contains more layers than v1 can represent"))
}

fn validate_nonempty(value: &str, label: &str) -> Result<(), CheckpointImportContractError> {
    if value.trim().is_empty() {
        return Err(invalid(format!("{label} must not be empty")));
    }
    Ok(())
}

/// The document contract's relative-path rule. The RULES live in exactly one place —
/// [`crate::checkpoint_plan_store::portable_relative_path_parts`] — and every checkpoint-seam
/// validator delegates there, so none of them can drift looser than this contract and admit a path
/// the others reject (feature-end round 1: this was the last of the five near-copies).
///
/// Only the refusal TYPE is local: the shared rule answers a `&'static str` reason so each caller
/// wraps it in its own error. The message is unchanged so the contract's published refusal text
/// stays byte-identical.
fn validate_relative_path(value: &str, label: &str) -> Result<(), CheckpointImportContractError> {
    validate_nonempty(value, label)?;
    crate::checkpoint_plan_store::portable_relative_path_parts(value)
        .map_err(|_| invalid(format!("{label} must be a portable confined relative path")))?;
    Ok(())
}

fn validate_sha256(value: &str, label: &str) -> Result<(), CheckpointImportContractError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(invalid(format!(
            "{label} must be a lowercase SHA-256 hex digest"
        )));
    }
    Ok(())
}

fn validate_sha256_prefixed(value: &str, label: &str) -> Result<(), CheckpointImportContractError> {
    let Some(digest) = value.strip_prefix("sha256:") else {
        return Err(invalid(format!("{label} must use the sha256: prefix")));
    };
    validate_sha256(digest, label)
}

/// A domain-separated SHA-256 identity over a struct with a fixed field order.
/// No map values participate in these contracts, so insertion order cannot affect
/// canonical bytes or a digest.
fn identity<T: Serialize>(domain: &str, value: &T) -> String {
    let payload = serde_json::to_vec(value).expect("checkpoint-import identity payload serializes");
    let mut hasher = Sha256::new();
    hasher.update(domain.as_bytes());
    hasher.update([0]);
    hasher.update(payload);
    format!("sha256:{:x}", hasher.finalize())
}

fn canonical_json<T: Serialize>(value: &T) -> Result<String, CheckpointImportContractError> {
    serde_json::to_string(value).map_err(|error| invalid(error.to_string()))
}

/// Provenance retained for an application-owned installed checkpoint.
///
/// Every field beyond `source` is optional and omitted from the canonical bytes when absent, so a
/// document written before a field existed and one written after with the field unset are the same
/// bytes and the same [`SourceLocatorV1::source_binding_identity`].
///
/// This records WHERE the bytes came from, never HOW they were authorized: `credential_host` names
/// the host whose stored credential was applied, and [`Self::validate`] refuses a `url` that
/// carries userinfo, so no token, key, or password can reach a persisted plan through here
/// (sc-20636).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ManagedProvenanceV1 {
    /// The ingest source kind: `upload`, `local-copy`, `url`, `huggingface`, `civitai`.
    pub source: String,
    /// The source-scoped human reference: an HF `repo@revision`, a Civitai model/version name, the
    /// uploaded filename.
    pub reference: Option<String>,
    /// The fetch URL, with userinfo and secret-bearing query parameters already stripped by the
    /// caller (`checkpoint_ingest::sanitize_provenance_url`). Validated here as a second line.
    pub url: Option<String>,
    /// The source-scoped version identity: a Civitai `modelVersionId`, an HF revision.
    pub version_id: Option<String>,
    /// The source-scoped file identity: a Civitai file id, an HF filename.
    pub file_id: Option<String>,
    /// The host whose stored credential authorized the fetch. The credential itself is never
    /// recorded; this is only the fact that one was used, and against which host.
    pub credential_host: Option<String>,
}

impl Serialize for ManagedProvenanceV1 {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.validate().map_err(serde::ser::Error::custom)?;
        let optional = [
            ("reference", &self.reference),
            ("url", &self.url),
            ("versionId", &self.version_id),
            ("fileId", &self.file_id),
            ("credentialHost", &self.credential_host),
        ];
        let present = optional.iter().filter(|(_, value)| value.is_some()).count();
        let mut state = serializer.serialize_struct("ManagedProvenanceV1", 1 + present)?;
        state.serialize_field("source", &self.source)?;
        for (key, value) in optional {
            if let Some(value) = value {
                state.serialize_field(key, value)?;
            }
        }
        state.end()
    }
}

impl<'de> Deserialize<'de> for ManagedProvenanceV1 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Wire {
            source: String,
            #[serde(default)]
            reference: Option<String>,
            #[serde(default)]
            url: Option<String>,
            #[serde(default)]
            version_id: Option<String>,
            #[serde(default)]
            file_id: Option<String>,
            #[serde(default)]
            credential_host: Option<String>,
        }
        let wire = Wire::deserialize(deserializer)?;
        let provenance = Self {
            source: wire.source,
            reference: wire.reference,
            url: wire.url,
            version_id: wire.version_id,
            file_id: wire.file_id,
            credential_host: wire.credential_host,
        };
        provenance.validate().map_err(D::Error::custom)?;
        Ok(provenance)
    }
}

impl ManagedProvenanceV1 {
    /// Provenance carrying only its source kind. Every other field is set by the ingest that knows
    /// it, so a caller can never leave one silently defaulted to a wrong value.
    pub fn of_source(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            ..Self::default()
        }
    }

    pub fn validate(&self) -> Result<(), CheckpointImportContractError> {
        validate_nonempty(&self.source, "managed provenance source")?;
        for (value, label) in [
            (&self.reference, "managed provenance reference"),
            (&self.url, "managed provenance url"),
            (&self.version_id, "managed provenance version id"),
            (&self.file_id, "managed provenance file id"),
            (&self.credential_host, "managed provenance credential host"),
        ] {
            if let Some(value) = value {
                validate_nonempty(value, label)?;
            }
        }
        if let Some(url) = &self.url {
            validate_secret_free_url(url)?;
        }
        Ok(())
    }
}

/// Refuses a provenance URL that embeds a credential. A persisted plan is world-readable app state
/// and is shown back to the user, so `https://user:token@host/file` must never reach it — the
/// authority component is checked directly rather than trusting the caller to have stripped it.
fn validate_secret_free_url(url: &str) -> Result<(), CheckpointImportContractError> {
    let authority = url
        .split_once("://")
        .map(|(_, rest)| rest)
        .unwrap_or(url)
        .split(['/', '?', '#'])
        .next()
        .unwrap_or_default();
    if authority.contains('@') {
        return Err(invalid(
            "managed provenance url must not embed credentials (userinfo)",
        ));
    }
    Ok(())
}

/// The physical location of one source file.
///
/// [`Self::semantic_identity`] deliberately excludes physical ownership and
/// location; [`Self::source_binding_identity`] includes both.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SourceLocatorV1 {
    Linked {
        schema_version: u32,
        root_id: String,
        relative_path: String,
        fingerprint: String,
    },
    Managed {
        schema_version: u32,
        install_id: String,
        relative_path: String,
        sha256: String,
        provenance: ManagedProvenanceV1,
    },
}

impl Serialize for SourceLocatorV1 {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.validate().map_err(serde::ser::Error::custom)?;
        match self {
            Self::Linked {
                schema_version,
                root_id,
                relative_path,
                fingerprint,
            } => {
                let mut state = serializer.serialize_struct("SourceLocatorV1", 5)?;
                state.serialize_field("kind", "linked")?;
                state.serialize_field("schemaVersion", schema_version)?;
                state.serialize_field("rootId", root_id)?;
                state.serialize_field("relativePath", relative_path)?;
                state.serialize_field("fingerprint", fingerprint)?;
                state.end()
            }
            Self::Managed {
                schema_version,
                install_id,
                relative_path,
                sha256,
                provenance,
            } => {
                let mut state = serializer.serialize_struct("SourceLocatorV1", 6)?;
                state.serialize_field("kind", "managed")?;
                state.serialize_field("schemaVersion", schema_version)?;
                state.serialize_field("installId", install_id)?;
                state.serialize_field("relativePath", relative_path)?;
                state.serialize_field("sha256", sha256)?;
                state.serialize_field("provenance", provenance)?;
                state.end()
            }
        }
    }
}

#[derive(Deserialize)]
#[serde(
    tag = "kind",
    rename_all = "snake_case",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
enum SourceLocatorV1Wire {
    Linked {
        #[serde(deserialize_with = "deserialize_schema_version")]
        schema_version: u32,
        root_id: String,
        relative_path: String,
        fingerprint: String,
    },
    Managed {
        #[serde(deserialize_with = "deserialize_schema_version")]
        schema_version: u32,
        install_id: String,
        relative_path: String,
        sha256: String,
        provenance: ManagedProvenanceV1,
    },
}

impl From<SourceLocatorV1Wire> for SourceLocatorV1 {
    fn from(value: SourceLocatorV1Wire) -> Self {
        match value {
            SourceLocatorV1Wire::Linked {
                schema_version,
                root_id,
                relative_path,
                fingerprint,
            } => Self::Linked {
                schema_version,
                root_id,
                relative_path,
                fingerprint,
            },
            SourceLocatorV1Wire::Managed {
                schema_version,
                install_id,
                relative_path,
                sha256,
                provenance,
            } => Self::Managed {
                schema_version,
                install_id,
                relative_path,
                sha256,
                provenance,
            },
        }
    }
}

impl<'de> Deserialize<'de> for SourceLocatorV1 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let locator = Self::from(deserialize_versioned_v1::<_, SourceLocatorV1Wire>(
            deserializer,
            VersionedSurface::SourceLocator,
        )?);
        locator.validate().map_err(D::Error::custom)?;
        Ok(locator)
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct LocatorSemanticV1<'a> {
    schema_version: u32,
    content_digest: &'a str,
}

impl SourceLocatorV1 {
    pub fn linked(
        root_id: impl Into<String>,
        relative_path: impl Into<String>,
        fingerprint: impl Into<String>,
    ) -> Result<Self, CheckpointImportContractError> {
        let locator = Self::Linked {
            schema_version: CHECKPOINT_IMPORT_CONTRACT_VERSION,
            root_id: root_id.into(),
            relative_path: relative_path.into(),
            fingerprint: fingerprint.into(),
        };
        locator.validate()?;
        Ok(locator)
    }

    pub fn managed(
        install_id: impl Into<String>,
        relative_path: impl Into<String>,
        sha256: impl Into<String>,
        provenance: ManagedProvenanceV1,
    ) -> Result<Self, CheckpointImportContractError> {
        let locator = Self::Managed {
            schema_version: CHECKPOINT_IMPORT_CONTRACT_VERSION,
            install_id: install_id.into(),
            relative_path: relative_path.into(),
            sha256: sha256.into(),
            provenance,
        };
        locator.validate()?;
        Ok(locator)
    }

    pub fn content_digest(&self) -> Result<&str, CheckpointImportContractError> {
        self.validate()?;
        Ok(match self {
            Self::Linked { fingerprint, .. } => fingerprint,
            Self::Managed { sha256, .. } => sha256,
        })
    }

    /// Locator-independent identity of the source bytes.
    pub fn semantic_identity(&self) -> Result<String, CheckpointImportContractError> {
        self.validate()?;
        Ok(identity(
            LOCATOR_SEMANTIC_DOMAIN,
            &LocatorSemanticV1 {
                schema_version: self.schema_version(),
                content_digest: self.content_digest()?,
            },
        ))
    }

    /// Identity of the source bytes and its physical owner, path, and provenance.
    pub fn source_binding_identity(&self) -> Result<String, CheckpointImportContractError> {
        self.validate()?;
        Ok(identity(LOCATOR_BINDING_DOMAIN, self))
    }

    pub fn canonical_json(&self) -> Result<String, CheckpointImportContractError> {
        self.validate()?;
        canonical_json(self)
    }

    pub fn validate(&self) -> Result<(), CheckpointImportContractError> {
        match self {
            Self::Linked {
                schema_version,
                root_id,
                relative_path,
                fingerprint,
            } => {
                validate_version(*schema_version)?;
                validate_nonempty(root_id, "linked root id")?;
                validate_relative_path(relative_path, "linked relative path")?;
                validate_sha256(fingerprint, "linked fingerprint")
            }
            Self::Managed {
                schema_version,
                install_id,
                relative_path,
                sha256,
                provenance,
            } => {
                validate_version(*schema_version)?;
                validate_nonempty(install_id, "managed install id")?;
                validate_relative_path(relative_path, "managed relative path")?;
                validate_sha256(sha256, "managed SHA-256")?;
                provenance.validate()
            }
        }
    }

    fn schema_version(&self) -> u32 {
        match self {
            Self::Linked { schema_version, .. } | Self::Managed { schema_version, .. } => {
                *schema_version
            }
        }
    }
}

/// One logical layer of an import plan. The source remains separate from its
/// logical shape, so linked and managed copies can compile to the same plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportLayerV1 {
    pub layer_id: String,
    pub role: String,
    pub target_path: String,
    pub source: SourceLocatorV1,
}

impl Serialize for ImportLayerV1 {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.validate().map_err(serde::ser::Error::custom)?;
        let mut state = serializer.serialize_struct("ImportLayerV1", 4)?;
        state.serialize_field("layerId", &self.layer_id)?;
        state.serialize_field("role", &self.role)?;
        state.serialize_field("targetPath", &self.target_path)?;
        state.serialize_field("source", &self.source)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ImportLayerV1Wire {
    layer_id: String,
    role: String,
    target_path: String,
    source: SourceLocatorV1,
}

impl From<ImportLayerV1Wire> for ImportLayerV1 {
    fn from(value: ImportLayerV1Wire) -> Self {
        Self {
            layer_id: value.layer_id,
            role: value.role,
            target_path: value.target_path,
            source: value.source,
        }
    }
}

impl<'de> Deserialize<'de> for ImportLayerV1 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let layer = Self::from(ImportLayerV1Wire::deserialize(deserializer)?);
        layer.validate().map_err(D::Error::custom)?;
        Ok(layer)
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SemanticLayerV1<'a> {
    layer_id: &'a str,
    role: &'a str,
    target_path: &'a str,
    source_semantic_identity: String,
}

impl ImportLayerV1 {
    pub fn validate(&self) -> Result<(), CheckpointImportContractError> {
        validate_nonempty(&self.layer_id, "import layer id")?;
        validate_nonempty(&self.role, "import layer role")?;
        validate_relative_path(&self.target_path, "import layer target path")?;
        self.source.validate()
    }

    fn semantic_form(&self) -> Result<SemanticLayerV1<'_>, CheckpointImportContractError> {
        self.validate()?;
        Ok(SemanticLayerV1 {
            layer_id: &self.layer_id,
            role: &self.role,
            target_path: &self.target_path,
            source_semantic_identity: self.source.semantic_identity()?,
        })
    }
}

/// The complete, immutable loading plan stored separately from catalog rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportPlanV1 {
    pub schema_version: u32,
    pub plan_id: String,
    pub family: String,
    pub layers: Vec<ImportLayerV1>,
}

impl Serialize for ImportPlanV1 {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.validate().map_err(serde::ser::Error::custom)?;
        let mut state = serializer.serialize_struct("ImportPlanV1", 4)?;
        state.serialize_field("schemaVersion", &self.schema_version)?;
        state.serialize_field("planId", &self.plan_id)?;
        state.serialize_field("family", &self.family)?;
        state.serialize_field("layers", &self.layers)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ImportPlanV1Wire {
    #[serde(deserialize_with = "deserialize_schema_version")]
    schema_version: u32,
    plan_id: String,
    family: String,
    layers: Vec<ImportLayerV1>,
}

impl From<ImportPlanV1Wire> for ImportPlanV1 {
    fn from(value: ImportPlanV1Wire) -> Self {
        Self {
            schema_version: value.schema_version,
            plan_id: value.plan_id,
            family: value.family,
            layers: value.layers,
        }
    }
}

impl<'de> Deserialize<'de> for ImportPlanV1 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let plan = Self::from(deserialize_versioned_v1::<_, ImportPlanV1Wire>(
            deserializer,
            VersionedSurface::ImportPlan,
        )?);
        plan.validate().map_err(D::Error::custom)?;
        Ok(plan)
    }
}

/// The semantic form deliberately carries NO `plan_id` (sc-20636).
///
/// A plan id is an assigned document name, not content: the inspector derives it from the
/// checkpoint id, which encodes ownership (`linked/<rootId>/<path>` vs `managed/<installId>`).
/// Including it made the "locator-independent" semantic digest differ between a linked checkpoint
/// and a managed copy of the identical bytes — the exact equality E1 requires and duplicate
/// detection is built on. Document identity is still bound by
/// [`ImportPlanV1::source_binding_identity`], which hashes the whole canonical plan.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SemanticPlanV1<'a> {
    schema_version: u32,
    family: &'a str,
    layers: Vec<SemanticLayerV1<'a>>,
}

impl ImportPlanV1 {
    pub fn new(
        plan_id: impl Into<String>,
        family: impl Into<String>,
        mut layers: Vec<ImportLayerV1>,
    ) -> Result<Self, CheckpointImportContractError> {
        layers.sort_by(|left, right| left.layer_id.cmp(&right.layer_id));
        let plan = Self {
            schema_version: CHECKPOINT_IMPORT_CONTRACT_VERSION,
            plan_id: plan_id.into(),
            family: family.into(),
            layers,
        };
        plan.validate()?;
        Ok(plan)
    }

    /// Content and logical-routing identity, deliberately excluding all physical
    /// source owners, paths, and managed provenance.
    pub fn semantic_digest(&self) -> Result<String, CheckpointImportContractError> {
        self.validate()?;
        Ok(identity(PLAN_SEMANTIC_DOMAIN, &self.semantic_form()?))
    }

    /// Identity that also binds each layer's exact physical owner, path, and provenance.
    pub fn source_binding_identity(&self) -> Result<String, CheckpointImportContractError> {
        self.validate()?;
        Ok(identity(PLAN_BINDING_DOMAIN, &self.canonicalized()))
    }

    pub fn plan_reference(&self) -> Result<ImportPlanReferenceV1, CheckpointImportContractError> {
        self.validate()?;
        let reference = ImportPlanReferenceV1 {
            schema_version: self.schema_version,
            plan_id: self.plan_id.clone(),
            semantic_digest: self.semantic_digest()?,
            source_binding_identity: self.source_binding_identity()?,
        };
        reference.validate()?;
        Ok(reference)
    }

    pub fn summary(&self) -> Result<ImportPlanSummaryV1, CheckpointImportContractError> {
        self.validate()?;
        let summary = ImportPlanSummaryV1 {
            schema_version: self.schema_version,
            family: self.family.clone(),
            layer_count: checked_layer_count(self.layers.len())?,
            layer_roles: self.layers.iter().map(|layer| layer.role.clone()).collect(),
            semantic_digest: self.semantic_digest()?,
        };
        summary.validate()?;
        Ok(summary)
    }

    /// Checked conversion shared by plan publication and callers that preflight
    /// an incoming layer count without materializing an impractically large plan.
    pub fn checked_layer_count(layer_count: usize) -> Result<u32, CheckpointImportContractError> {
        checked_layer_count(layer_count)
    }

    pub fn canonical_json(&self) -> Result<String, CheckpointImportContractError> {
        self.validate()?;
        canonical_json(&self.canonicalized())
    }

    pub fn validate(&self) -> Result<(), CheckpointImportContractError> {
        validate_version(self.schema_version)?;
        validate_nonempty(&self.plan_id, "import plan id")?;
        validate_nonempty(&self.family, "import plan family")?;
        if self.layers.is_empty() {
            return Err(invalid("import plan must contain at least one layer"));
        }
        checked_layer_count(self.layers.len())?;
        let mut previous = None;
        for layer in &self.layers {
            layer.validate()?;
            if previous.as_deref() >= Some(layer.layer_id.as_str()) {
                return Err(invalid(
                    "import plan layers must be uniquely sorted by layer id",
                ));
            }
            previous = Some(layer.layer_id.clone());
        }
        Ok(())
    }

    fn semantic_form(&self) -> Result<SemanticPlanV1<'_>, CheckpointImportContractError> {
        let mut layers: Vec<_> = self
            .layers
            .iter()
            .map(ImportLayerV1::semantic_form)
            .collect::<Result<_, _>>()?;
        layers.sort_by(|left, right| left.layer_id.cmp(right.layer_id));
        Ok(SemanticPlanV1 {
            schema_version: self.schema_version,
            family: &self.family,
            layers,
        })
    }

    fn canonicalized(&self) -> Self {
        let mut plan = self.clone();
        plan.layers
            .sort_by(|left, right| left.layer_id.cmp(&right.layer_id));
        plan
    }
}

/// Stable handle to a separately atomically stored [`ImportPlanV1`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportPlanReferenceV1 {
    pub schema_version: u32,
    pub plan_id: String,
    pub semantic_digest: String,
    pub source_binding_identity: String,
}

impl Serialize for ImportPlanReferenceV1 {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.validate().map_err(serde::ser::Error::custom)?;
        let mut state = serializer.serialize_struct("ImportPlanReferenceV1", 4)?;
        state.serialize_field("schemaVersion", &self.schema_version)?;
        state.serialize_field("planId", &self.plan_id)?;
        state.serialize_field("semanticDigest", &self.semantic_digest)?;
        state.serialize_field("sourceBindingIdentity", &self.source_binding_identity)?;
        state.end()
    }
}

/// Short spelling for APIs that conventionally call this a plan reference.
pub type ImportPlanRefV1 = ImportPlanReferenceV1;

impl<'de> Deserialize<'de> for ImportPlanReferenceV1 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Wire {
            #[serde(deserialize_with = "deserialize_schema_version")]
            schema_version: u32,
            plan_id: String,
            semantic_digest: String,
            source_binding_identity: String,
        }
        let wire =
            deserialize_versioned_v1::<_, Wire>(deserializer, VersionedSurface::PlanReference)?;
        let reference = Self {
            schema_version: wire.schema_version,
            plan_id: wire.plan_id,
            semantic_digest: wire.semantic_digest,
            source_binding_identity: wire.source_binding_identity,
        };
        reference.validate().map_err(D::Error::custom)?;
        Ok(reference)
    }
}

impl ImportPlanReferenceV1 {
    pub fn canonical_json(&self) -> Result<String, CheckpointImportContractError> {
        self.validate()?;
        canonical_json(self)
    }
    pub fn validate(&self) -> Result<(), CheckpointImportContractError> {
        validate_version(self.schema_version)?;
        validate_nonempty(&self.plan_id, "import plan reference id")?;
        validate_sha256_prefixed(&self.semantic_digest, "import plan semantic digest")?;
        validate_sha256_prefixed(
            &self.source_binding_identity,
            "import plan source-binding identity",
        )
    }
}

/// Small catalog-facing projection of a plan; it intentionally contains no layers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportPlanSummaryV1 {
    pub schema_version: u32,
    pub family: String,
    pub layer_count: u32,
    pub layer_roles: Vec<String>,
    pub semantic_digest: String,
}

impl Serialize for ImportPlanSummaryV1 {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.validate().map_err(serde::ser::Error::custom)?;
        let mut state = serializer.serialize_struct("ImportPlanSummaryV1", 5)?;
        state.serialize_field("schemaVersion", &self.schema_version)?;
        state.serialize_field("family", &self.family)?;
        state.serialize_field("layerCount", &self.layer_count)?;
        state.serialize_field("layerRoles", &self.layer_roles)?;
        state.serialize_field("semanticDigest", &self.semantic_digest)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for ImportPlanSummaryV1 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Wire {
            #[serde(deserialize_with = "deserialize_schema_version")]
            schema_version: u32,
            family: String,
            #[serde(deserialize_with = "deserialize_layer_count")]
            layer_count: u32,
            layer_roles: Vec<String>,
            semantic_digest: String,
        }
        let wire =
            deserialize_versioned_v1::<_, Wire>(deserializer, VersionedSurface::PlanSummary)?;
        let summary = Self {
            schema_version: wire.schema_version,
            family: wire.family,
            layer_count: wire.layer_count,
            layer_roles: wire.layer_roles,
            semantic_digest: wire.semantic_digest,
        };
        summary.validate().map_err(D::Error::custom)?;
        Ok(summary)
    }
}

impl ImportPlanSummaryV1 {
    pub fn canonical_json(&self) -> Result<String, CheckpointImportContractError> {
        self.validate()?;
        canonical_json(self)
    }
    pub fn validate(&self) -> Result<(), CheckpointImportContractError> {
        validate_version(self.schema_version)?;
        validate_nonempty(&self.family, "import plan summary family")?;
        if self.layer_count == 0 || self.layer_count as usize != self.layer_roles.len() {
            return Err(invalid(
                "import plan summary layer count must match its non-empty roles",
            ));
        }
        for role in &self.layer_roles {
            validate_nonempty(role, "import plan summary layer role")?;
        }
        validate_sha256_prefixed(&self.semantic_digest, "import plan summary semantic digest")
    }
}

/// A catalog row with a plan reference and compact projection, never per-layer data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckpointCatalogRecordV1 {
    pub schema_version: u32,
    pub checkpoint_id: String,
    pub plan: ImportPlanReferenceV1,
    pub summary: ImportPlanSummaryV1,
}

impl Serialize for CheckpointCatalogRecordV1 {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.validate().map_err(serde::ser::Error::custom)?;
        let mut state = serializer.serialize_struct("CheckpointCatalogRecordV1", 4)?;
        state.serialize_field("schemaVersion", &self.schema_version)?;
        state.serialize_field("checkpointId", &self.checkpoint_id)?;
        state.serialize_field("plan", &self.plan)?;
        state.serialize_field("summary", &self.summary)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for CheckpointCatalogRecordV1 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Wire {
            #[serde(deserialize_with = "deserialize_schema_version")]
            schema_version: u32,
            checkpoint_id: String,
            plan: ImportPlanReferenceV1,
            summary: ImportPlanSummaryV1,
        }
        let wire =
            deserialize_versioned_v1::<_, Wire>(deserializer, VersionedSurface::CatalogRecord)?;
        let record = Self {
            schema_version: wire.schema_version,
            checkpoint_id: wire.checkpoint_id,
            plan: wire.plan,
            summary: wire.summary,
        };
        record.validate().map_err(D::Error::custom)?;
        Ok(record)
    }
}

impl CheckpointCatalogRecordV1 {
    pub fn from_plan(
        checkpoint_id: impl Into<String>,
        plan: &ImportPlanV1,
    ) -> Result<Self, CheckpointImportContractError> {
        plan.validate()?;
        let record = Self {
            schema_version: CHECKPOINT_IMPORT_CONTRACT_VERSION,
            checkpoint_id: checkpoint_id.into(),
            plan: plan.plan_reference()?,
            summary: plan.summary()?,
        };
        record.validate()?;
        record.validate_loaded_plan(plan)?;
        Ok(record)
    }
    pub fn canonical_json(&self) -> Result<String, CheckpointImportContractError> {
        self.validate()?;
        canonical_json(self)
    }
    pub fn validate(&self) -> Result<(), CheckpointImportContractError> {
        validate_version(self.schema_version)?;
        validate_nonempty(&self.checkpoint_id, "checkpoint id")?;
        self.plan.validate()?;
        self.summary.validate()?;
        if self.plan.semantic_digest != self.summary.semantic_digest {
            return Err(invalid(
                "catalog plan reference and summary semantic digests differ",
            ));
        }
        Ok(())
    }

    /// Verifies that an atomically loaded plan is exactly the plan this catalog
    /// record advertised, rather than trusting the record's claimed digests.
    pub fn validate_loaded_plan(
        &self,
        loaded_plan: &ImportPlanV1,
    ) -> Result<(), CheckpointImportContractError> {
        self.validate()?;
        loaded_plan.validate()?;
        let expected_reference = loaded_plan.plan_reference()?;
        let expected_summary = loaded_plan.summary()?;
        if self.plan.plan_id != expected_reference.plan_id {
            return Err(invalid(
                "catalog plan reference does not match loaded plan id",
            ));
        }
        if self.plan.semantic_digest != expected_reference.semantic_digest {
            return Err(invalid(
                "catalog plan reference does not match loaded plan semantic digest",
            ));
        }
        if self.plan.source_binding_identity != expected_reference.source_binding_identity {
            return Err(invalid(
                "catalog plan reference does not match loaded plan source binding",
            ));
        }
        if self.summary.family != expected_summary.family {
            return Err(invalid(
                "catalog plan summary does not match loaded plan family",
            ));
        }
        if self.summary.layer_count != expected_summary.layer_count {
            return Err(invalid(
                "catalog plan summary does not match loaded plan layer count",
            ));
        }
        if self.summary.layer_roles != expected_summary.layer_roles {
            return Err(invalid(
                "catalog plan summary does not match loaded plan layer roles",
            ));
        }
        if self.summary.semantic_digest != expected_summary.semantic_digest {
            return Err(invalid(
                "catalog plan summary does not match loaded plan semantic digest",
            ));
        }
        Ok(())
    }
}

/// Canonical inventory of catalog records used by discovery and routing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckpointInventoryV1 {
    pub schema_version: u32,
    pub records: Vec<CheckpointCatalogRecordV1>,
}

impl Serialize for CheckpointInventoryV1 {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.validate().map_err(serde::ser::Error::custom)?;
        let mut records: Vec<_> = self.records.iter().collect();
        records.sort_by(|left, right| left.checkpoint_id.cmp(&right.checkpoint_id));
        let mut state = serializer.serialize_struct("CheckpointInventoryV1", 2)?;
        state.serialize_field("schemaVersion", &self.schema_version)?;
        state.serialize_field("records", &records)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for CheckpointInventoryV1 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Wire {
            #[serde(deserialize_with = "deserialize_schema_version")]
            schema_version: u32,
            records: Vec<CheckpointCatalogRecordV1>,
        }
        let wire = deserialize_versioned_v1::<_, Wire>(
            deserializer,
            VersionedSurface::CheckpointInventory,
        )?;
        let inventory = Self {
            schema_version: wire.schema_version,
            records: wire.records,
        };
        inventory.validate().map_err(D::Error::custom)?;
        Ok(inventory)
    }
}

impl CheckpointInventoryV1 {
    pub fn new(
        mut records: Vec<CheckpointCatalogRecordV1>,
    ) -> Result<Self, CheckpointImportContractError> {
        records.sort_by(|left, right| left.checkpoint_id.cmp(&right.checkpoint_id));
        let inventory = Self {
            schema_version: CHECKPOINT_IMPORT_CONTRACT_VERSION,
            records,
        };
        inventory.validate()?;
        Ok(inventory)
    }

    pub fn canonical_json(&self) -> Result<String, CheckpointImportContractError> {
        self.validate()?;
        let mut inventory = self.clone();
        inventory
            .records
            .sort_by(|left, right| left.checkpoint_id.cmp(&right.checkpoint_id));
        canonical_json(&inventory)
    }

    pub fn validate(&self) -> Result<(), CheckpointImportContractError> {
        validate_version(self.schema_version)?;
        let mut ids = BTreeSet::new();
        let mut plan_ids = BTreeSet::new();
        for record in &self.records {
            record.validate()?;
            if !ids.insert(&record.checkpoint_id) {
                return Err(invalid(
                    "checkpoint inventory contains duplicate checkpoint ids",
                ));
            }
            if !plan_ids.insert(&record.plan.plan_id) {
                return Err(invalid(
                    "checkpoint inventory contains duplicate import plan ids",
                ));
            }
        }
        Ok(())
    }
}
