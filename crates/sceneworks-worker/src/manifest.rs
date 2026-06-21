//! JSONC manifest read/upsert/write helpers shared by the LoRA and model manifests.
use super::*;

pub(crate) async fn read_json_value(path: &Path) -> WorkerResult<Value> {
    Ok(serde_json::from_slice(&tokio::fs::read(path).await?)?)
}

/// Upsert `entry` (keyed by its `id`) into the `collection_key` array of a JSONC
/// manifest at `path`, creating the manifest when absent. An existing entry with
/// the same id is merged (incoming fields win) but keeps its original `createdAt`.
/// Shared by the LoRA (`"loras"`) and model (`"models"`) manifests, which differed
/// only by this array key (sc-4279 / F-MLXW-15).
pub(crate) async fn upsert_manifest_entry(
    path: &Path,
    collection_key: &str,
    entry: serde_json::Map<String, Value>,
) -> WorkerResult<()> {
    let mut manifest = match tokio::fs::read_to_string(path).await {
        Ok(payload) => serde_json::from_str(&strip_jsonc_comments(&payload))?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let mut object = serde_json::Map::new();
            object.insert("schemaVersion".to_owned(), json!(1));
            object.insert(collection_key.to_owned(), Value::Array(Vec::new()));
            Value::Object(object)
        }
        Err(error) => return Err(error.into()),
    };
    let entry_id = entry.get("id").and_then(Value::as_str).ok_or_else(|| {
        WorkerError::InvalidPayload(format!("{collection_key} manifest entry requires id"))
    })?;
    let collection = manifest
        .as_object_mut()
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!("{collection_key} manifest must be an object"))
        })?
        .entry(collection_key.to_owned())
        .or_insert_with(|| Value::Array(Vec::new()));
    let collection = collection.as_array_mut().ok_or_else(|| {
        WorkerError::InvalidPayload(format!("{collection_key} manifest array must be an array"))
    })?;
    let mut found = false;
    for item in collection.iter_mut() {
        if item.get("id").and_then(Value::as_str) != Some(entry_id) {
            continue;
        }
        found = true;
        let created_at = item.get("createdAt").cloned();
        let Some(object) = item.as_object_mut() else {
            return Err(WorkerError::InvalidPayload(format!(
                "{collection_key} manifest entry must be an object"
            )));
        };
        for (key, value) in entry.clone() {
            object.insert(key, value);
        }
        if let Some(created_at) = created_at {
            object.insert("createdAt".to_owned(), created_at);
        }
    }
    if !found {
        collection.push(Value::Object(entry));
    }
    write_json_value(path, &manifest).await
}

pub(crate) async fn write_json_value(path: &Path, value: &Value) -> WorkerResult<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let mut output = serde_json::to_vec_pretty(value)?;
    output.push(b'\n');
    let tmp_path = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("json")
    ));
    tokio::fs::write(&tmp_path, output).await?;
    tokio::fs::rename(tmp_path, path).await?;
    Ok(())
}
