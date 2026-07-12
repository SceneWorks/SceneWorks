// Generic manifest-entity CRUD helpers shared by `prompt_batches` and `recipe_presets`
// (sc-11217, F-029). Both modules persist a named, scoped entity into the same
// global/project JSONC-manifest storage with the same list/get/create/update/
// delete-as-archive/duplicate shape. That CRUD skeleton used to be copy-adapted into two
// ~400-line clones; it now lives here once, parameterized by the per-entity differences.
//
// IMPORTANT (behavior preservation): the two entities deliberately differ in several
// runtime details — the recipe-preset flavor validates against the model/LoRA catalogs,
// finalizes its response, supports a read-only `builtin` scope, and strips a DIFFERENT set
// of runtime fields on update/duplicate than the prompt-batch flavor does. Every such
// difference is carried in through a parameter or closure so each endpoint behaves
// EXACTLY as it did before this extraction. In particular the update/duplicate field-strip
// sets are passed by the caller verbatim: their drift is a separate finding (F-056,
// sc-11277) and is intentionally left untouched here.

use super::recipe_presets::{
    next_duplicate_preset_id, next_duplicate_preset_name, slugify_preset_id,
};
use super::*;

/// Resolved write target for a manifest entity: the concrete scope it lives in and the
/// manifest file to mutate. Shared by both entities (replaced the two identical per-entity
/// `*WriteLocation` structs).
#[derive(Debug, Clone)]
pub(crate) struct ManifestWriteLocation {
    pub scope: String,
    pub manifest_path: PathBuf,
}

fn entry_has_id(entry: &Value, id: &str) -> bool {
    entry.get("id").and_then(Value::as_str) == Some(id)
}

/// Both entities store the archived flag under the same `archived` boolean.
pub(crate) fn manifest_entry_archived(entry: &Value) -> bool {
    entry
        .get("archived")
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// Shared list-filter: drop archived entries (unless requested), then keep only the
/// requested scope, then apply the caller's entity-specific extra predicate (prompt
/// batches have none; recipe presets additionally filter by model/workflow). The three
/// retains commute, so ordering matches the prior per-entity code exactly.
pub(crate) fn filter_manifest_catalog(
    mut entries: Vec<Value>,
    include_archived: bool,
    scope: Option<&str>,
    extra: impl Fn(&Value) -> bool,
) -> Vec<Value> {
    if !include_archived {
        entries.retain(|entry| !manifest_entry_archived(entry));
    }
    if let Some(scope) = scope {
        entries.retain(|entry| entry.get("scope").and_then(Value::as_str) == Some(scope));
    }
    entries.retain(extra);
    entries
}

/// Shared single-entity GET lookup: find by id, then apply the optional scope filter and
/// the archived filter, else the entity's not-found error.
pub(crate) fn find_manifest_entry(
    catalog: Vec<Value>,
    entity_id: &str,
    scope: Option<&str>,
    include_archived: bool,
    not_found: impl Fn() -> ApiError,
) -> Result<Value, ApiError> {
    catalog
        .into_iter()
        .find(|entry| entry_has_id(entry, entity_id))
        .filter(|entry| {
            scope.map_or(true, |scope| {
                entry.get("scope").and_then(Value::as_str) == Some(scope)
            })
        })
        .filter(|entry| include_archived || !manifest_entry_archived(entry))
        .ok_or_else(not_found)
}

/// Shared "does this scope's manifest already contain the id?" probe used by the per-entity
/// `find_*_write_location` resolvers. The caller resolves `manifest_path` for `scope`
/// (filenames differ per entity), we scan for the id and wrap it into a location.
pub(crate) async fn manifest_location_if_present(
    state: &AppState,
    entity_id: &str,
    scope: &str,
    manifest_field: &str,
    manifest_path: PathBuf,
    not_found: impl Fn() -> ApiError,
) -> Result<ManifestWriteLocation, ApiError> {
    let entries = load_manifest_entries(state, &manifest_path, manifest_field).await?;
    if entries.iter().any(|entry| entry_has_id(entry, entity_id)) {
        Ok(ManifestWriteLocation {
            scope: scope.to_owned(),
            manifest_path,
        })
    } else {
        Err(not_found())
    }
}

/// Shared CREATE: derive the id (from an explicit id or a slugified name), stamp
/// created/updated timestamps, then finalize + dedup-check + append inside the shared
/// manifest read→merge→save lock. `finalize` carries the entity's normalize (and, for
/// recipe presets, model/LoRA validation); it is called with `require_all = true`.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn create_manifest_entry(
    state: &AppState,
    manifest_path: &FsPath,
    manifest_field: &str,
    entry: Value,
    scope: &str,
    must_be_object_msg: &'static str,
    name_required_msg: &'static str,
    already_exists_msg: &'static str,
    finalize: impl FnOnce(Value, &str, bool) -> Result<Value, ApiError>,
) -> Result<Value, ApiError> {
    let mut entry = entry;
    let object = entry
        .as_object_mut()
        .ok_or_else(|| ApiError::bad_request(must_be_object_msg))?;
    let id = object
        .get("id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .or_else(|| {
            object
                .get("name")
                .and_then(Value::as_str)
                .map(slugify_preset_id)
        })
        .ok_or_else(|| ApiError::bad_request(name_required_msg))?;
    object.insert("id".to_owned(), Value::String(id.clone()));
    let timestamp = now_rfc3339();
    object
        .entry("createdAt".to_owned())
        .or_insert_with(|| Value::String(timestamp.clone()));
    object.insert("updatedAt".to_owned(), Value::String(timestamp));
    mutate_manifest_entries(state, manifest_path, manifest_field, move |mut entries| {
        let entry = finalize(entry, scope, true)?;
        if entries.iter().any(|existing| entry_has_id(existing, &id)) {
            return Err(ApiError::bad_request(already_exists_msg));
        }
        entries.push(entry.clone());
        Ok((entries, entry))
    })
    .await
}

/// Shared UPDATE: merge the caller's already-stripped patch onto the stored entry, pin the
/// id, restamp `updatedAt`, then finalize (`require_all = false`) and write back in place.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn update_manifest_entry(
    state: &AppState,
    manifest_path: &FsPath,
    manifest_field: &str,
    entity_id: &str,
    scope: &str,
    patch: Value,
    not_found: impl Fn() -> ApiError,
    finalize: impl FnOnce(Value, &str, bool) -> Result<Value, ApiError>,
) -> Result<Value, ApiError> {
    let entity_id = entity_id.to_owned();
    mutate_manifest_entries(state, manifest_path, manifest_field, move |mut entries| {
        let Some(index) = entries
            .iter()
            .position(|entry| entry_has_id(entry, &entity_id))
        else {
            return Err(not_found());
        };
        let mut entry = entries[index].clone();
        merge_object(&mut entry, patch);
        if let Some(object) = entry.as_object_mut() {
            object.insert("id".to_owned(), Value::String(entity_id.clone()));
            object.insert("updatedAt".to_owned(), Value::String(now_rfc3339()));
        }
        let entry = finalize(entry, scope, false)?;
        entries[index] = entry.clone();
        Ok((entries, entry))
    })
    .await
}

/// Shared DELETE-as-archive: flip `archived` to true, restamp `updatedAt`, finalize
/// (`require_all = false`) and write back. Recipe-preset delete deliberately finalizes with
/// normalize only (no catalog validation); prompt-batch delete likewise — the caller
/// supplies the exact finalize it used before.
pub(crate) async fn delete_manifest_entry(
    state: &AppState,
    manifest_path: &FsPath,
    manifest_field: &str,
    entity_id: &str,
    scope: &str,
    not_found: impl Fn() -> ApiError,
    finalize: impl FnOnce(Value, &str, bool) -> Result<Value, ApiError>,
) -> Result<Value, ApiError> {
    let entity_id = entity_id.to_owned();
    mutate_manifest_entries(state, manifest_path, manifest_field, move |mut entries| {
        let Some(index) = entries
            .iter()
            .position(|entry| entry_has_id(entry, &entity_id))
        else {
            return Err(not_found());
        };
        let mut entry = entries[index].clone();
        if let Some(object) = entry.as_object_mut() {
            object.insert("archived".to_owned(), Value::Bool(true));
            object.insert("updatedAt".to_owned(), Value::String(now_rfc3339()));
        }
        let entry = finalize(entry, scope, false)?;
        entries[index] = entry.clone();
        Ok((entries, entry))
    })
    .await
}

/// Shared DUPLICATE: clone the source, strip its runtime fields (the per-entity set is
/// injected via `strip_runtime` — prompt batches drop only `manifestPath`; recipe presets
/// also drop `builtInLoras`/`appliedDefaults`/`lastUsedAt`), assign a fresh copy id/name,
/// reset scope/archived/timestamps, finalize (`require_all = true`) and append.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn duplicate_manifest_entry(
    state: &AppState,
    manifest_path: &FsPath,
    manifest_field: &str,
    entity_id: &str,
    scope: &str,
    not_found: impl Fn() -> ApiError,
    strip_runtime: impl Fn(&mut Value),
    finalize: impl FnOnce(Value, &str, bool) -> Result<Value, ApiError>,
) -> Result<Value, ApiError> {
    let entity_id = entity_id.to_owned();
    let scope = scope.to_owned();
    mutate_manifest_entries(state, manifest_path, manifest_field, move |mut entries| {
        let Some(source) = entries
            .iter()
            .find(|entry| entry_has_id(entry, &entity_id))
            .cloned()
        else {
            return Err(not_found());
        };
        let mut duplicate = source;
        strip_runtime(&mut duplicate);
        let base_id = duplicate
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or(entity_id.as_str());
        let duplicate_id = next_duplicate_preset_id(&entries, base_id);
        let duplicate_name = next_duplicate_preset_name(
            &entries,
            duplicate
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or(base_id),
        );
        let timestamp = now_rfc3339();
        if let Some(object) = duplicate.as_object_mut() {
            object.insert("id".to_owned(), Value::String(duplicate_id));
            object.insert("name".to_owned(), Value::String(duplicate_name));
            object.insert("scope".to_owned(), Value::String(scope.clone()));
            object.insert("archived".to_owned(), Value::Bool(false));
            object.insert("createdAt".to_owned(), Value::String(timestamp.clone()));
            object.insert("updatedAt".to_owned(), Value::String(timestamp));
        }
        let duplicate = finalize(duplicate, &scope, true)?;
        entries.push(duplicate.clone());
        Ok((entries, duplicate))
    })
    .await
}
