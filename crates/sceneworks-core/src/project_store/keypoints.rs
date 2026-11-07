//! ProjectStore keypoints aggregate.

use super::*;

/// Reserved project that holds the GLOBAL Key Point Library (epic 4422, sc-4434). User-created
/// face-angle presets live here as ordinary `type:"keypoint"` assets (kps + the retained source
/// image), and the user's angle-set collections live in a `keypoint-collections.json` sidecar.
/// Hidden from `list_projects` like the pose library. The built-in 11 angle presets stay in
/// [`crate::angle_kps`] (virtual, always-present, protected — never stored here).
pub const GLOBAL_KEYPOINTS_PROJECT_ID: &str = "project_global_keypoints";

pub const GLOBAL_KEYPOINTS_PROJECT_NAME: &str = "Key Point Library";

/// The user angle-set collections store, relative to the global keypoints project.
const KEYPOINT_COLLECTIONS_FILE: &str = "keypoint-collections.json";

/// Staging cache dir (under `<data>/cache`) for Key Point Library source uploads. One
/// constant shared by the API staging route (`create_keypoint_sources`), the store's
/// save-preset guard ([`ProjectStore`]'s keypoint upload check), and the worker's
/// `kps sourcePath` confinement — sc-13713 was these drifting apart (the worker confined
/// to the pose lane's dir, rejecting every capture).
pub const KEYPOINT_UPLOADS_CACHE_DIR: &str = "keypoint-uploads";

fn parse_normalized_kps(value: Option<&Value>) -> ProjectStoreResult<Vec<Value>> {
    let points = value.and_then(Value::as_array).ok_or_else(|| {
        ProjectStoreError::BadRequest("keypoint spec missing kps array".to_owned())
    })?;
    if points.len() != 5 {
        return Err(ProjectStoreError::BadRequest(format!(
            "keypoint kps must have 5 points, got {}",
            points.len()
        )));
    }
    let mut out = Vec::with_capacity(5);
    for point in points {
        let pair = point
            .as_array()
            .filter(|values| values.len() == 2)
            .ok_or_else(|| {
                ProjectStoreError::BadRequest("each kps point must be [x, y]".to_owned())
            })?;
        let x = pair[0]
            .as_f64()
            .ok_or_else(|| ProjectStoreError::BadRequest("kps x must be a number".to_owned()))?;
        let y = pair[1]
            .as_f64()
            .ok_or_else(|| ProjectStoreError::BadRequest("kps y must be a number".to_owned()))?;
        if !(0.0..=1.0).contains(&x) || !(0.0..=1.0).contains(&y) {
            return Err(ProjectStoreError::BadRequest(
                "kps must be normalized to [0,1]".to_owned(),
            ));
        }
        out.push(json!([x, y]));
    }
    Ok(out)
}

/// Parse a stored/served kps record (`[[x,y]×5]`) into a fixed `[(f32,f32);5]`, or `None` if it
/// isn't exactly 5 numeric pairs. Used to resolve a collection's presets for generation (sc-4450).
fn parse_kps_tuple(value: Option<&Value>) -> Option<[(f32, f32); 5]> {
    let points = value.and_then(Value::as_array)?;
    if points.len() != 5 {
        return None;
    }
    let mut out = [(0.0f32, 0.0f32); 5];
    for (slot, point) in out.iter_mut().zip(points) {
        let pair = point.as_array().filter(|values| values.len() == 2)?;
        *slot = (pair[0].as_f64()? as f32, pair[1].as_f64()? as f32);
    }
    Some(out)
}

/// The built-in 11 as resolved angle presets (the generation fallback / default set, sc-4450).
fn builtin_resolved_presets() -> Vec<ResolvedAnglePreset> {
    crate::angle_kps::BUILTIN_ANGLE_KPS
        .iter()
        .map(|(angle, kps)| ResolvedAnglePreset {
            preset_id: crate::angle_kps::builtin_preset_id(angle),
            name: crate::angle_kps::builtin_angle_display_name(angle),
            angle: Some((*angle).to_owned()),
            kps: *kps,
        })
        .collect()
}

/// Map a stored `type:"keypoint"` asset → a Key Point Library preset record.
fn keypoint_asset_to_preset(asset: &Value) -> Value {
    json!({
        "id": asset.get("id").cloned().unwrap_or(Value::Null),
        "name": asset.get("displayName").cloned().unwrap_or(Value::Null),
        "kps": asset
            .get("keypoint")
            .and_then(|keypoint| keypoint.get("kps"))
            .cloned()
            .unwrap_or(Value::Null),
        "builtin": false,
        "sourceImageRef": asset
            .get("file")
            .and_then(|file| file.get("path"))
            .cloned()
            .unwrap_or(Value::Null),
        "sourceAssetId": asset
            .get("lineage")
            .and_then(|lineage| lineage.get("sourceAssetId"))
            .cloned()
            .unwrap_or(Value::Null),
    })
}

/// Read the user angle-set collections (`keypoint-collections.json`) → `[]` when absent.
fn read_user_collections(project_path: &Path) -> ProjectStoreResult<Vec<Value>> {
    let path = project_path.join(KEYPOINT_COLLECTIONS_FILE);
    if !path.exists() {
        return Ok(Vec::new());
    }
    let document = read_json(&path)?;
    Ok(document
        .get("collections")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default())
}

/// Persist the user angle-set collections sidecar.
fn write_user_collections(project_path: &Path, collections: &[Value]) -> ProjectStoreResult<()> {
    let path = project_path.join(KEYPOINT_COLLECTIONS_FILE);
    write_json(
        &path,
        &json!({ "schemaVersion": 1, "collections": collections }),
    )
}

impl ProjectStore {
    /// Ensure the reserved global Key Point Library project exists (idempotent), returning its
    /// summary. Created lazily on first preset/collection write (epic 4422, sc-4434).
    pub fn ensure_global_keypoints_project(&self) -> ProjectStoreResult<ProjectSummary> {
        let mut registry_cache = self.lock.lock();
        self.ensure_data_dirs()?;
        let project_path = self.projects_dir().join("global-keypoints.sceneworks");
        if project_path.exists() {
            return read_project_summary(&project_path);
        }
        self.provision_project_locked(
            GLOBAL_KEYPOINTS_PROJECT_ID,
            GLOBAL_KEYPOINTS_PROJECT_NAME,
            &project_path,
            &mut registry_cache,
        )
    }

    // ---- Key Point Library (epic 4422, sc-4434) -----------------------------------------

    /// Guard a staged source-image path under the keypoint-uploads cache (mirrors
    /// [`Self::pose_preview_path`]) so a save can't copy an arbitrary file into the library.
    fn keypoint_upload_path(&self, path: &str) -> ProjectStoreResult<PathBuf> {
        let cache_root = self.data_dir.join("cache").join(KEYPOINT_UPLOADS_CACHE_DIR);
        let canonical_root = fs::canonicalize(&cache_root).map_err(|_| {
            ProjectStoreError::NotFound("Key Point upload cache is unavailable".to_owned())
        })?;
        let canonical = fs::canonicalize(path).map_err(|_| {
            ProjectStoreError::NotFound("Key Point source image not found".to_owned())
        })?;
        if !canonical.starts_with(&canonical_root) || !canonical.is_file() {
            return Err(ProjectStoreError::BadRequest(
                "Key Point source path is invalid".to_owned(),
            ));
        }
        Ok(canonical)
    }

    /// Persist a user keypoint preset: the 5-point kps + the retained source image (copied from
    /// a staged upload into the library). Built-in presets are NOT stored here — they live in
    /// [`crate::angle_kps`] (virtual, protected). Spec:
    /// `{ name, kps[[x,y]×5], sourceUploadPath, sourceAssetId? }`.
    pub fn create_keypoint_asset(&self, spec: &Value) -> ProjectStoreResult<Value> {
        let name = spec
            .get("name")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| ProjectStoreError::BadRequest("keypoint spec missing name".to_owned()))?
            .to_owned();
        let kps = parse_normalized_kps(spec.get("kps"))?;
        let upload_path = spec
            .get("sourceUploadPath")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ProjectStoreError::BadRequest("keypoint spec missing sourceUploadPath".to_owned())
            })?;
        let canonical_source = self.keypoint_upload_path(upload_path)?;

        self.ensure_global_keypoints_project()?;
        let (project_path, _project_guard) = self.lock_project(GLOBAL_KEYPOINTS_PROJECT_ID)?;

        let asset_id = format!("asset_{}", random_hex(16)?);
        let created_at = utc_now();
        let dir = project_path.join("assets").join("keypoints");
        fs::create_dir_all(&dir)?;
        // Determine the format from the file CONTENT, not the staged path's extension: uploads
        // arrive as `upload-<uuid>.tmp` (the generic temp writer keeps no extension), so keying
        // off the extension would mislabel every capture as PNG. A valid-but-unsupported format
        // (AVIF/HEIC/HEIF/TIFF/BMP/GIF) is transcoded to PNG so the retained source image is always
        // decodable and never mislabeled (sc-6143) — without this an AVIF capture would be copied
        // verbatim under a `.png` name. A sniffable supported format keeps its bytes; when the magic
        // bytes aren't recognized at all (e.g. SVG) we fall back to the extension, then PNG.
        let index_mutation = AssetIndexMutation::begin(&project_path)?;
        let ingested = crate::media_convert::ingest_image_with_naming(
            &canonical_source,
            &dir.join(&asset_id),
            crate::media_convert::ImageCopyNaming::Canonical,
        )?;
        let media_path = ingested.path;
        let mime = ingested.mime;
        let media_rel = relative_string(&project_path, &media_path)?;

        let source_asset_id = spec
            .get("sourceAssetId")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let parents = source_asset_id
            .clone()
            .map(|id| vec![Value::String(id)])
            .unwrap_or_default();

        let asset = json!({
            "schemaVersion": 1,
            "id": asset_id,
            "projectId": GLOBAL_KEYPOINTS_PROJECT_ID,
            "generationSetId": Value::Null,
            "type": "keypoint",
            "displayName": name,
            "createdAt": created_at,
            "origin": "keypoint_library",
            "tags": Vec::<Value>::new(),
            "file": {
                "path": media_rel,
                "mimeType": mime,
                "width": spec.get("sourceWidth").cloned().unwrap_or(Value::Null),
                "height": spec.get("sourceHeight").cloned().unwrap_or(Value::Null),
                "duration": Value::Null,
                "fps": Value::Null
            },
            "status": { "favorite": false, "rating": 0, "rejected": false, "trashed": false },
            "recipe": {
                "mode": "upload",
                "model": "scrfd",
                "adapter": "api-upload",
                "prompt": name,
                "negativePrompt": "",
                "seed": 0,
                "loras": [],
                "stylePreset": "none",
                "normalizedSettings": {},
                "rawAdapterSettings": { "detector": "scrfd" }
            },
            "lineage": {
                "parents": parents,
                "sourceAssetId": source_asset_id,
                "sourceTimestamp": Value::Null,
                "jobId": Value::Null
            },
            "keypoint": {
                "kps": kps,
                "builtin": false
            }
        });
        let sidecar_path = media_path.with_extension("sceneworks.json");
        write_json(&sidecar_path, &asset)?;
        index_asset(
            GLOBAL_KEYPOINTS_PROJECT_ID,
            &project_path,
            &asset,
            Some(&sidecar_path),
        )?;
        index_mutation.commit();
        normalize_asset(GLOBAL_KEYPOINTS_PROJECT_ID, &project_path, &sidecar_path)
    }

    /// All keypoint presets: the built-in 11 (from [`crate::angle_kps`]) followed by the user's
    /// stored `type:"keypoint"` assets, each as a `{ id, name, kps, builtin, sourceImageRef }`
    /// record. The single read the Key Point Library renders from.
    pub fn list_keypoint_presets(&self) -> ProjectStoreResult<Vec<Value>> {
        let mut presets = crate::angle_kps::builtin_preset_records();
        presets.extend(self.list_user_keypoint_presets()?);
        Ok(presets)
    }

    fn list_user_keypoint_presets(&self) -> ProjectStoreResult<Vec<Value>> {
        self.ensure_global_keypoints_project()?;
        let assets = self.list_assets_filtered(
            GLOBAL_KEYPOINTS_PROJECT_ID,
            false,
            false,
            AssetScope::All,
            Some("keypoint"),
        )?;
        Ok(assets.iter().map(keypoint_asset_to_preset).collect())
    }

    /// The set of preset ids a collection may reference: the built-in ids + the user's stored
    /// keypoint asset ids.
    fn known_preset_ids(&self) -> ProjectStoreResult<std::collections::HashSet<String>> {
        let mut ids: std::collections::HashSet<String> = crate::angle_kps::BUILTIN_ANGLE_SET_ORDER
            .iter()
            .map(|angle| crate::angle_kps::builtin_preset_id(angle))
            .collect();
        for preset in self.list_user_keypoint_presets()? {
            if let Some(id) = preset.get("id").and_then(Value::as_str) {
                ids.insert(id.to_owned());
            }
        }
        Ok(ids)
    }

    /// All angle-set collections: the virtual built-in default (the 11 in order) followed by the
    /// user's collections. The built-in's `isDefault` is true unless a user collection claims it.
    pub fn list_keypoint_collections(&self) -> ProjectStoreResult<Vec<Value>> {
        self.ensure_global_keypoints_project()?;
        let project_path = self.find_project_path(GLOBAL_KEYPOINTS_PROJECT_ID)?;
        let user = read_user_collections(&project_path)?;
        let any_user_default = user
            .iter()
            .any(|collection| collection.get("isDefault").and_then(Value::as_bool) == Some(true));
        let mut builtin = crate::angle_kps::builtin_default_collection();
        builtin["isDefault"] = Value::Bool(!any_user_default);
        let mut all = vec![builtin];
        all.extend(user);
        Ok(all)
    }

    /// Create or update a user angle-set collection: `{ id?, name, orderedPresetIds[], isDefault? }`.
    /// Validates every referenced preset exists; marking it default clears the flag on the others
    /// (and on the built-in, which then reports `isDefault:false`).
    pub fn upsert_keypoint_collection(&self, spec: &Value) -> ProjectStoreResult<Value> {
        let name = spec
            .get("name")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| ProjectStoreError::BadRequest("collection missing name".to_owned()))?
            .to_owned();
        let ordered = spec
            .get("orderedPresetIds")
            .and_then(Value::as_array)
            .filter(|values| !values.is_empty())
            .ok_or_else(|| {
                ProjectStoreError::BadRequest(
                    "collection needs a non-empty orderedPresetIds".to_owned(),
                )
            })?;
        let ordered_ids: Vec<String> = ordered
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect();
        if ordered_ids.len() != ordered.len() {
            return Err(ProjectStoreError::BadRequest(
                "orderedPresetIds must be strings".to_owned(),
            ));
        }
        let known = self.known_preset_ids()?;
        for id in &ordered_ids {
            if !known.contains(id) {
                return Err(ProjectStoreError::BadRequest(format!(
                    "collection references unknown preset id {id}"
                )));
            }
        }
        let requested_id = spec
            .get("id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty());
        if requested_id == Some(crate::angle_kps::BUILTIN_DEFAULT_COLLECTION_ID) {
            return Err(ProjectStoreError::BadRequest(
                "the built-in default collection is read-only".to_owned(),
            ));
        }
        let id = match requested_id {
            Some(value) if is_safe_id(value) => value.to_owned(),
            Some(_) => {
                return Err(ProjectStoreError::BadRequest(
                    "invalid collection id".to_owned(),
                ))
            }
            None => format!("kpc_{}", random_hex(8)?),
        };
        let make_default = spec.get("isDefault").and_then(Value::as_bool) == Some(true);

        self.ensure_global_keypoints_project()?;
        let (project_path, _project_guard) = self.lock_project(GLOBAL_KEYPOINTS_PROJECT_ID)?;
        let mut collections = read_user_collections(&project_path)?;
        let record = json!({
            "id": id,
            "name": name,
            "orderedPresetIds": ordered_ids,
            "isDefault": make_default,
            "builtin": false,
        });
        match collections
            .iter_mut()
            .find(|collection| collection.get("id").and_then(Value::as_str) == Some(id.as_str()))
        {
            Some(existing) => *existing = record.clone(),
            None => collections.push(record.clone()),
        }
        if make_default {
            for collection in collections.iter_mut() {
                if collection.get("id").and_then(Value::as_str) != Some(id.as_str()) {
                    collection["isDefault"] = Value::Bool(false);
                }
            }
        }
        write_user_collections(&project_path, &collections)?;
        Ok(record)
    }

    /// Mark one collection the default (clearing the others). Passing the built-in id clears all
    /// user defaults so the built-in default becomes effective again.
    pub fn set_default_keypoint_collection(&self, id: &str) -> ProjectStoreResult<Vec<Value>> {
        self.ensure_global_keypoints_project()?;
        {
            let (project_path, _project_guard) = self.lock_project(GLOBAL_KEYPOINTS_PROJECT_ID)?;
            let mut collections = read_user_collections(&project_path)?;
            if id == crate::angle_kps::BUILTIN_DEFAULT_COLLECTION_ID {
                for collection in collections.iter_mut() {
                    collection["isDefault"] = Value::Bool(false);
                }
            } else {
                if !collections
                    .iter()
                    .any(|collection| collection.get("id").and_then(Value::as_str) == Some(id))
                {
                    return Err(ProjectStoreError::NotFound(format!(
                        "collection {id} not found"
                    )));
                }
                for collection in collections.iter_mut() {
                    let is_target = collection.get("id").and_then(Value::as_str) == Some(id);
                    collection["isDefault"] = Value::Bool(is_target);
                }
            }
            write_user_collections(&project_path, &collections)?;
        }
        self.list_keypoint_collections()
    }

    /// Delete a user collection. The built-in default collection cannot be deleted.
    pub fn delete_keypoint_collection(&self, id: &str) -> ProjectStoreResult<()> {
        if id == crate::angle_kps::BUILTIN_DEFAULT_COLLECTION_ID {
            return Err(ProjectStoreError::BadRequest(
                "the built-in default collection cannot be deleted".to_owned(),
            ));
        }
        self.ensure_global_keypoints_project()?;
        let (project_path, _project_guard) = self.lock_project(GLOBAL_KEYPOINTS_PROJECT_ID)?;
        let mut collections = read_user_collections(&project_path)?;
        let before = collections.len();
        collections.retain(|collection| collection.get("id").and_then(Value::as_str) != Some(id));
        if collections.len() == before {
            return Err(ProjectStoreError::NotFound(format!(
                "collection {id} not found"
            )));
        }
        write_user_collections(&project_path, &collections)
    }

    /// Resolve the active angle-set collection to its ordered presets (kps + name + optional
    /// canonical angle) for generation (sc-4450). Selection: an explicit `collection_id` override
    /// → the user's default → the built-in default (the 11). Referenced presets that no longer
    /// resolve (e.g. a custom preset deleted after being added) are skipped; if nothing resolves,
    /// falls back to the built-in 11. Returns `(collection_id_used, presets)`.
    pub fn resolve_angle_collection(
        &self,
        collection_id: Option<&str>,
    ) -> ProjectStoreResult<(String, Vec<ResolvedAnglePreset>)> {
        let collections = self.list_keypoint_collections()?;
        let target = match collection_id {
            Some(id) => collections
                .iter()
                .find(|collection| collection.get("id").and_then(Value::as_str) == Some(id)),
            None => collections.iter().find(|collection| {
                collection.get("isDefault").and_then(Value::as_bool) == Some(true)
            }),
        };
        let used_id = target
            .and_then(|collection| collection.get("id").and_then(Value::as_str))
            .unwrap_or(crate::angle_kps::BUILTIN_DEFAULT_COLLECTION_ID)
            .to_owned();
        let ordered: Vec<String> = target
            .and_then(|collection| collection.get("orderedPresetIds"))
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();

        let user_presets = self.list_user_keypoint_presets()?;
        let mut resolved = Vec::with_capacity(ordered.len());
        for id in &ordered {
            if let Some(angle) = id.strip_prefix("builtin_") {
                if let Some(kps) = crate::angle_kps::angle_kps(angle) {
                    resolved.push(ResolvedAnglePreset {
                        preset_id: id.clone(),
                        name: crate::angle_kps::builtin_angle_display_name(angle),
                        angle: Some(angle.to_owned()),
                        kps,
                    });
                }
            } else if let Some(record) = user_presets
                .iter()
                .find(|preset| preset.get("id").and_then(Value::as_str) == Some(id.as_str()))
            {
                if let Some(kps) = parse_kps_tuple(record.get("kps")) {
                    resolved.push(ResolvedAnglePreset {
                        preset_id: id.clone(),
                        name: record
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or(id)
                            .to_owned(),
                        angle: None,
                        kps,
                    });
                }
            }
        }
        if resolved.is_empty() {
            return Ok((
                crate::angle_kps::BUILTIN_DEFAULT_COLLECTION_ID.to_owned(),
                builtin_resolved_presets(),
            ));
        }
        Ok((used_id, resolved))
    }
}
