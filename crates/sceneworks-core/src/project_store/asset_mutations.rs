//! ProjectStore asset mutations aggregate.

use super::*;

impl ProjectStore {
    /// Write the generation-set JSON for a job from the worker-reported facts.
    /// Idempotent — overwrites the same `<id>.json` on re-applied updates.
    pub fn write_generation_set(
        &self,
        project_id: &str,
        job_id: &str,
        generation_set: &Value,
        recipe_fact: Option<&Value>,
    ) -> ProjectStoreResult<()> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        let id = generation_set
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ProjectStoreError::BadRequest("generation set fact missing id".to_owned())
            })?;
        // The id comes from the worker-reported fact and is joined into the
        // generation-sets path below, so charset-check it before path use — the
        // worker->API boundary must not write outside the project root
        // (sc-13582; mirrors the persist_generated_asset guards, sc-5721/sc-4211).
        if !is_safe_id(id) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid generation set id".to_owned(),
            ));
        }
        let get = |key: &str| generation_set.get(key).cloned().unwrap_or(Value::Null);
        let record = json!({
            "schemaVersion": 1,
            "id": id,
            "projectId": project_id,
            "jobId": job_id,
            "mode": get("mode"),
            "model": get("model"),
            "prompt": get("prompt"),
            "negativePrompt": get("negativePrompt"),
            "count": get("count"),
            "createdAt": get("createdAt"),
        });
        let embedded_recipe = generation_set.get("recipe").cloned().or_else(|| {
            recipe_fact.and_then(|fact| {
                build_generated_asset_sidecar(project_id, job_id, id, fact)
                    .get("recipe")
                    .cloned()
            })
        });
        let mut record = record;
        if let Some(recipe) = embedded_recipe {
            if let Some(object) = record.as_object_mut() {
                object.insert("recipe".to_owned(), recipe);
            }
        }
        let index_mutation = AssetIndexMutation::begin(&project_path)?;
        let dir = project_path.join("generation-sets");
        fs::create_dir_all(&dir)?;
        write_json(&dir.join(format!("{id}.json")), &record)?;
        maybe_fail_asset_index_db_mutation()?;
        let mut connection = connect_project_db(&project_path)?;
        apply_project_migrations(&connection)?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        transaction.execute(
            "
            insert or replace into generation_sets (id, mode, model, prompt, created_at, job_id)
            values (?1, ?2, ?3, ?4, ?5, ?6)
            ",
            params![
                id,
                optional_str(&record, "mode").unwrap_or("unknown"),
                optional_str(&record, "model").unwrap_or("unknown"),
                optional_str(&record, "prompt").unwrap_or(""),
                optional_str(&record, "createdAt").unwrap_or(""),
                optional_str(&record, "jobId"),
            ],
        )?;
        let indexed_assets = {
            let mut statement = transaction
                .prepare("select id, asset_json from assets where generation_set_id = ?1")?;
            let rows = statement
                .query_map(params![id], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?))
                })?
                .collect::<Result<Vec<_>, _>>()?;
            rows
        };
        for (asset_id, asset_json) in indexed_assets {
            let Some(asset_json) = asset_json else {
                continue;
            };
            let Ok(mut asset) = serde_json::from_str::<Value>(&asset_json) else {
                continue;
            };
            let Some(object) = asset.as_object_mut() else {
                continue;
            };
            object.insert("generationSet".to_owned(), record.clone());
            transaction.execute(
                "update assets set asset_json = ?1 where id = ?2",
                params![serde_json::to_string(&asset)?, asset_id],
            )?;
        }
        transaction.commit()?;
        index_mutation.commit();
        Ok(())
    }

    pub fn update_asset_status(
        &self,
        project_id: &str,
        asset_id: &str,
        patch: AssetStatusPatch,
    ) -> ProjectStoreResult<Value> {
        if patch.rating.is_some_and(|rating| rating > 5) {
            return Err(ProjectStoreError::BadRequest(
                "Rating must be between 0 and 5".to_owned(),
            ));
        }
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        let sidecar_path = self.find_asset_sidecar(&project_path, asset_id)?;
        let mut asset = read_json(&sidecar_path)?;
        let status = asset
            .as_object_mut()
            .ok_or_else(|| {
                ProjectStoreError::BadRequest("Asset sidecar must be an object".to_owned())
            })?
            .entry("status")
            .or_insert_with(|| json!({}));
        let status = status.as_object_mut().ok_or_else(|| {
            ProjectStoreError::BadRequest("Asset status must be an object".to_owned())
        })?;
        if let Some(value) = patch.favorite {
            status.insert("favorite".to_owned(), Value::Bool(value));
        }
        if let Some(value) = patch.rating {
            status.insert("rating".to_owned(), json!(value));
        }
        if let Some(value) = patch.rejected {
            status.insert("rejected".to_owned(), Value::Bool(value));
        }
        if let Some(value) = patch.trashed {
            status.insert("trashed".to_owned(), Value::Bool(value));
        }
        let index_mutation = AssetIndexMutation::begin(&project_path)?;
        write_json(&sidecar_path, &asset)?;
        index_asset(project_id, &project_path, &asset, Some(&sidecar_path))?;
        index_mutation.commit();
        normalize_asset(project_id, &project_path, &sidecar_path)
    }

    pub fn update_asset_tags(
        &self,
        project_id: &str,
        asset_id: &str,
        patch: AssetTagsPatch,
    ) -> ProjectStoreResult<Value> {
        let tags = normalize_asset_tags(&patch.tags)?;
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        let sidecar_path = self.find_asset_sidecar(&project_path, asset_id)?;
        let mut asset = read_json(&sidecar_path)?;
        let object = asset.as_object_mut().ok_or_else(|| {
            ProjectStoreError::BadRequest("Asset sidecar must be an object".to_owned())
        })?;
        object.insert(
            "tags".to_owned(),
            Value::Array(tags.into_iter().map(Value::String).collect()),
        );
        let index_mutation = AssetIndexMutation::begin(&project_path)?;
        write_json(&sidecar_path, &asset)?;
        index_asset(project_id, &project_path, &asset, Some(&sidecar_path))?;
        index_mutation.commit();
        normalize_asset(project_id, &project_path, &sidecar_path)
    }

    /// Promote a character asset into the Main Asset Library (sc-8341). This is a
    /// true *move*: the asset leaves every character and surfaces in the Library.
    /// Because the Library excludes `origin == "character_studio"` and the Character
    /// Assets grid keys off `recipe.normalizedSettings.characterId` +
    /// `metadata.characterReferences`, all three must change: flip `origin` to a
    /// library-visible studio value, drop the recipe's `characterId`, and strip the
    /// asset's `characterReferences`, then unlink it from every character sidecar.
    ///
    /// The move carries the asset's whole upscale-fold group (sc-10205), so a
    /// folded original/upscaled pair never gets split across collections. Returns
    /// the array of updated normalized assets, the requested one first.
    pub fn move_asset_to_library(
        &self,
        project_id: &str,
        asset_id: &str,
    ) -> ProjectStoreResult<Value> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        let character_store = CharacterStore::new(&self.data_dir, project_path.clone());
        let index_mutation = AssetIndexMutation::begin(&project_path)?;
        let mut moved = Vec::new();
        for member_id in upscale_lineage_group(&project_path, asset_id)? {
            let sidecar_path = self.find_asset_sidecar(&project_path, &member_id)?;
            let mut asset = read_json(&sidecar_path)?;
            {
                let object = asset.as_object_mut().ok_or_else(|| {
                    ProjectStoreError::BadRequest("Asset sidecar must be an object".to_owned())
                })?;
                // Promote to a library-visible origin by media type so the
                // `character_studio` exclusion (LibraryScreen) no longer applies.
                let asset_type = object
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let origin = match asset_type {
                    "video" => "video_studio",
                    "document" => "document_studio",
                    _ => "image_studio",
                };
                object.insert("origin".to_owned(), Value::String(origin.to_owned()));
                // Detach the character association the grid filters on.
                if let Some(settings) = object
                    .get_mut("recipe")
                    .and_then(Value::as_object_mut)
                    .and_then(|recipe| recipe.get_mut("normalizedSettings"))
                    .and_then(Value::as_object_mut)
                {
                    settings.remove("characterId");
                }
                if let Some(metadata) = object.get_mut("metadata").and_then(Value::as_object_mut) {
                    metadata.remove("characterReferences");
                }
            }
            write_json(&sidecar_path, &asset)?;
            index_asset(project_id, &project_path, &asset, Some(&sidecar_path))?;
            // Drop the asset from every character's references[] (character sidecars + index).
            character_store.remove_asset_references(&member_id)?;
            moved.push(normalize_asset(project_id, &project_path, &sidecar_path)?);
        }
        index_mutation.commit();
        Ok(Value::Array(moved))
    }

    /// Move a library asset into a character's assets (sc-10200): the true-move
    /// twin of [`Self::move_asset_to_library`]. A character-sidecar `references[]`
    /// entry is NOT the right vehicle for a move — it leaves the asset in the Main
    /// Asset Library (its `origin` stays library-visible) and surfaces it in the
    /// curated "Approved set" panel (which renders every `references[]` entry). So
    /// a move must: flip `origin` to `character_studio` (the Library's allow-list
    /// then excludes it), anchor the target association in
    /// `metadata.characterReferences` only (the Character assets grid and the
    /// `?character=` scope both match on it), and detach the asset from everything
    /// else — the recipe's `characterId` and every character's curated
    /// `references[]` — without adding a curated reference to the target.
    ///
    /// The move carries the asset's whole upscale-fold group (sc-10205): the
    /// Library renders a linked original/upscaled pair as ONE folded tile, so a
    /// move of the visible tile must take the hidden fold-mate along or it stays
    /// stranded in the Library. Returns the array of updated normalized assets,
    /// the requested one first.
    pub fn move_asset_to_character(
        &self,
        project_id: &str,
        asset_id: &str,
        character_id: &str,
    ) -> ProjectStoreResult<Value> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        let character_store = CharacterStore::new(&self.data_dir, project_path.clone());
        // Reject an unknown/deleted target before mutating anything.
        character_store.get_character(project_id, character_id)?;
        let index_mutation = AssetIndexMutation::begin(&project_path)?;
        let mut moved = Vec::new();
        for member_id in upscale_lineage_group(&project_path, asset_id)? {
            let sidecar_path = self.find_asset_sidecar(&project_path, &member_id)?;
            let mut asset = read_json(&sidecar_path)?;
            {
                let object = asset.as_object_mut().ok_or_else(|| {
                    ProjectStoreError::BadRequest("Asset sidecar must be an object".to_owned())
                })?;
                object.insert(
                    "origin".to_owned(),
                    Value::String("character_studio".to_owned()),
                );
                // The recipe's characterId is a competing association vector (it would
                // keep the asset in the previous character's grid), so clear it and let
                // the metadata anchor own the membership.
                if let Some(settings) = object
                    .get_mut("recipe")
                    .and_then(Value::as_object_mut)
                    .and_then(|recipe| recipe.get_mut("normalizedSettings"))
                    .and_then(Value::as_object_mut)
                {
                    settings.remove("characterId");
                }
                let metadata = object
                    .entry("metadata".to_owned())
                    .or_insert_with(|| json!({}));
                let metadata = metadata.as_object_mut().ok_or_else(|| {
                    ProjectStoreError::BadRequest("Asset metadata must be an object".to_owned())
                })?;
                // `source` distinguishes this move anchor from the "character-sidecar"
                // mirror entries `update_asset_character_link` manages — that rebuild
                // must not drop the anchor when a curated reference is added/removed.
                metadata.insert(
                    "characterReferences".to_owned(),
                    json!([{
                        "characterId": character_id,
                        "source": "library-move",
                        "approved": false,
                        "role": "asset",
                        "linkedAt": utc_now(),
                    }]),
                );
            }
            write_json(&sidecar_path, &asset)?;
            index_asset(project_id, &project_path, &asset, Some(&sidecar_path))?;
            // Leave no curated membership behind: the move detaches the asset from every
            // character's references[] (including the target's — the Approved set is
            // hand-curated and a bulk move must not populate it).
            character_store.remove_asset_references(&member_id)?;
            moved.push(normalize_asset(project_id, &project_path, &sidecar_path)?);
        }
        index_mutation.commit();
        Ok(Value::Array(moved))
    }

    /// Mark the raster produced by a disclosed prompt-to-vector workflow as hidden but retained.
    /// The exact workflow and parent ids are the deletion authority used by cleanup; callers can
    /// never turn an arbitrary project asset into a cleanup target by naming only its id.
    pub fn mark_vector_workflow_intermediate(
        &self,
        project_id: &str,
        asset_id: &str,
        workflow_id: &str,
        parent_job_id: &str,
        child_job_id: &str,
    ) -> ProjectStoreResult<Value> {
        if !is_safe_id(workflow_id) || !is_safe_id(parent_job_id) || !is_safe_id(child_job_id) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid vector workflow ownership id".to_owned(),
            ));
        }
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        let sidecar_path = self.find_asset_sidecar(&project_path, asset_id)?;
        let mut asset = read_json(&sidecar_path)?;
        let object = asset.as_object_mut().ok_or_else(|| {
            ProjectStoreError::BadRequest("Asset sidecar must be an object".to_owned())
        })?;
        if object.get("type").and_then(Value::as_str) != Some("image") {
            return Err(ProjectStoreError::BadRequest(
                "Vector workflow intermediate must be a raster image".to_owned(),
            ));
        }
        object.insert(
            "origin".to_owned(),
            Value::String("vector_workflow_intermediate".to_owned()),
        );
        let extra = object
            .entry("extra")
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .ok_or_else(|| {
                ProjectStoreError::BadRequest("Asset extra must be an object".to_owned())
            })?;
        extra.insert(
            "vectorWorkflow".to_owned(),
            json!({
                "role": "retained_intermediate",
                "publication": "unpublished",
                "workflowId": workflow_id,
                "parentJobId": parent_job_id,
                "childJobId": child_job_id,
                "hidden": true,
            }),
        );
        let index_mutation = AssetIndexMutation::begin(&project_path)?;
        write_json(&sidecar_path, &asset)?;
        index_asset(project_id, &project_path, &asset, Some(&sidecar_path))?;
        index_mutation.commit();
        normalize_asset(project_id, &project_path, &sidecar_path)
    }

    /// Permanently purge a workflow-owned intermediate only when every ownership fact matches.
    /// Returns false for a user asset, another workflow's asset, or an already-cleaned id.
    pub fn purge_vector_workflow_intermediate(
        &self,
        project_id: &str,
        asset_id: &str,
        workflow_id: &str,
        parent_job_id: &str,
    ) -> ProjectStoreResult<bool> {
        let asset = match self.get_asset(project_id, asset_id) {
            Ok(asset) => asset,
            Err(ProjectStoreError::NotFound(_)) => return Ok(false),
            Err(error) => return Err(error),
        };
        let owned = asset.get("origin").and_then(Value::as_str)
            == Some("vector_workflow_intermediate")
            && asset
                .pointer("/extra/vectorWorkflow/workflowId")
                .and_then(Value::as_str)
                == Some(workflow_id)
            && asset
                .pointer("/extra/vectorWorkflow/parentJobId")
                .and_then(Value::as_str)
                == Some(parent_job_id)
            && asset
                .pointer("/extra/vectorWorkflow/role")
                .and_then(Value::as_str)
                == Some("retained_intermediate")
            && asset
                .pointer("/extra/vectorWorkflow/publication")
                .and_then(Value::as_str)
                == Some("unpublished");
        if !owned {
            return Ok(false);
        }
        self.purge_asset(project_id, asset_id, true)?;
        Ok(true)
    }

    pub fn index_asset_sidecar(
        &self,
        project_id: &str,
        sidecar_path: &Path,
    ) -> ProjectStoreResult<()> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        let asset = read_json(sidecar_path)?;
        let index_mutation = AssetIndexMutation::begin(&project_path)?;
        index_asset(project_id, &project_path, &asset, Some(sidecar_path))?;
        index_mutation.commit();
        Ok(())
    }

    /// Persist a native worker-produced asset sidecar and recipe, then index the
    /// envelope as one repair-guarded project mutation.
    ///
    /// Native workers must use this instead of writing the sidecar and later
    /// calling [`Self::index_asset_sidecar`]: the durable marker and reentrant
    /// project lock begin before either JSON write and remain owned until the
    /// separate SQLite transaction commits.
    pub fn persist_native_asset_sidecar(
        &self,
        project_id: &str,
        sidecar_path: &Path,
        mut asset: Value,
    ) -> ProjectStoreResult<Value> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        let sidecar_rel = relative_string(&project_path, sidecar_path)?;
        if !is_safe_relative_path(&sidecar_rel) {
            return Err(ProjectStoreError::BadRequest(
                "Asset sidecar path must be project-relative".to_owned(),
            ));
        }
        self.inherit_model_provenance(&project_path, &mut asset)?;
        let asset_id = required_str(&asset, "id")?;
        if !is_safe_id(asset_id) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid native asset id".to_owned(),
            ));
        }
        let recipe = asset.get("recipe").ok_or_else(|| {
            ProjectStoreError::BadRequest("Native asset recipe is required".to_owned())
        })?;
        let index_mutation = AssetIndexMutation::begin(&project_path)?;
        write_json(sidecar_path, &asset)?;
        write_json(
            &project_path
                .join("recipes")
                .join(format!("{asset_id}.recipe.json")),
            recipe,
        )?;
        index_asset(project_id, &project_path, &asset, Some(sidecar_path))?;
        index_mutation.commit();
        Ok(asset)
    }

    pub fn delete_asset(
        &self,
        project_id: &str,
        asset_id: &str,
    ) -> ProjectStoreResult<AssetMutationResult> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        let sidecar_path = self.find_asset_sidecar(&project_path, asset_id)?;
        let mut asset = read_json(&sidecar_path)?;
        let media_rel = asset
            .pointer("/file/path")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        if !media_rel.is_empty() && !is_safe_relative_path(&media_rel) {
            return Err(ProjectStoreError::BadRequest(
                "Asset media path must be project-relative".to_owned(),
            ));
        }
        let media_path = project_path.join(&media_rel);
        let status = asset
            .as_object_mut()
            .ok_or_else(|| {
                ProjectStoreError::BadRequest("Asset sidecar must be an object".to_owned())
            })?
            .entry("status")
            .or_insert_with(|| json!({}));
        let status = status.as_object_mut().ok_or_else(|| {
            ProjectStoreError::BadRequest("Asset status must be an object".to_owned())
        })?;
        status.insert("trashed".to_owned(), Value::Bool(true));

        let index_mutation = AssetIndexMutation::begin(&project_path)?;
        let trash_dir = project_path.join("trash").join(asset_id);
        fs::create_dir_all(&trash_dir)?;
        remove_sibling_poster_no_symlinks(&project_path, &media_rel)?;
        if media_path.exists() && media_path.is_file() {
            let trashed_media_path = trash_dir.join(media_path.file_name().ok_or_else(|| {
                ProjectStoreError::BadRequest("Invalid asset media path".to_owned())
            })?);
            fs::rename(&media_path, &trashed_media_path)?;
            if let Some(file) = asset.get_mut("file").and_then(Value::as_object_mut) {
                file.insert(
                    "path".to_owned(),
                    Value::String(relative_string(&project_path, &trashed_media_path)?),
                );
            }
        }
        let trashed_sidecar_path = trash_dir.join(sidecar_path.file_name().ok_or_else(|| {
            ProjectStoreError::BadRequest("Invalid asset sidecar path".to_owned())
        })?);
        write_json(&trashed_sidecar_path, &asset)?;
        if sidecar_path != trashed_sidecar_path {
            fs::remove_file(&sidecar_path).ok();
        }
        index_asset(
            project_id,
            &project_path,
            &asset,
            Some(&trashed_sidecar_path),
        )?;
        index_mutation.commit();
        Ok(AssetMutationResult {
            id: asset_id.to_owned(),
            status: "trashed".to_owned(),
        })
    }

    /// Permanently remove a (usually already-trashed) asset. With `permanent = false`
    /// the on-disk media/sidecar are moved to the OS trash (recoverable); if that move
    /// fails nothing is deleted and the result carries `status = "trash_unavailable"`
    /// so the caller can prompt the user to confirm a permanent delete (`permanent = true`).
    pub fn purge_asset(
        &self,
        project_id: &str,
        asset_id: &str,
        permanent: bool,
    ) -> ProjectStoreResult<AssetMutationResult> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        let sidecar_path = self.find_asset_sidecar(&project_path, asset_id)?;
        let sidecar_rel = relative_string(&project_path, &sidecar_path)?;
        if !is_safe_relative_path(&sidecar_rel) {
            return Err(ProjectStoreError::BadRequest(
                "Asset sidecar path must be project-relative".to_owned(),
            ));
        }
        let sidecar_path = confined_deletion_target(&project_path, &sidecar_rel, "Asset sidecar")?;
        let asset = read_json(&sidecar_path)?;
        let media_rel = asset
            .pointer("/file/path")
            .and_then(Value::as_str)
            .unwrap_or_default();
        // The sidecar is durable, externally writable project state. Validate its
        // media path before constructing any deletion or trash target from it.
        if !media_rel.is_empty() && !is_safe_relative_path(media_rel) {
            return Err(ProjectStoreError::BadRequest(
                "Asset media path must be project-relative".to_owned(),
            ));
        }
        let media_path = if media_rel.is_empty() {
            None
        } else {
            Some(confined_deletion_target(
                &project_path,
                media_rel,
                "Asset media",
            )?)
        };
        // A separately stored derived preview (vector PNG, depth preview — sc-25682) is part of the
        // asset's footprint: soft delete leaves it in place for the trashed sidecar to keep
        // rendering, and purge removes it with the media instead of orphaning it.
        let preview_rel = asset
            .pointer("/preview/path")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let preview_path = if preview_rel.is_empty() || !is_safe_relative_path(preview_rel) {
            None
        } else {
            Some(confined_deletion_target(
                &project_path,
                preview_rel,
                "Asset preview",
            )?)
        };
        let project_root = fs::canonicalize(&project_path)?;
        let trash_dir = project_root.join("trash");
        let asset_trash_dir = sidecar_path
            .parent()
            .filter(|parent| {
                parent.file_name().and_then(|name| name.to_str()) == Some(asset_id)
                    && parent.parent() == Some(trash_dir.as_path())
            })
            .map(Path::to_path_buf);
        if !permanent {
            // Move the asset's footprint to the OS trash rather than unlinking it. When
            // the asset already lives in `trash/<id>/`, trash that directory in one shot;
            // otherwise trash the media file and sidecar directly.
            let mut targets: Vec<PathBuf> = if let Some(dir) = asset_trash_dir.clone() {
                vec![dir]
            } else {
                let mut targets = Vec::new();
                if let Some(media_path) = media_path
                    .as_ref()
                    .filter(|path| path.exists() && path.is_file())
                {
                    targets.push(media_path.clone());
                }
                if sidecar_path.exists() {
                    targets.push(sidecar_path.clone());
                }
                targets
            };
            if let Some(preview_path) = preview_path
                .as_ref()
                .filter(|path| path.exists() && path.is_file())
            {
                targets.push(preview_path.clone());
            }
            let index_mutation = AssetIndexMutation::begin(&project_path)?;
            remove_sibling_poster_no_symlinks(&project_path, media_rel)?;
            if !targets.is_empty() && trash::delete_all(&targets).is_err() {
                // The trash API does not guarantee an atomic multi-target operation.
                // Keep the marker so the next read repairs any partial deletion before
                // serving indexed rows.
                return Ok(AssetMutationResult {
                    id: asset_id.to_owned(),
                    status: "trash_unavailable".to_owned(),
                });
            }
            CharacterStore::new(&self.data_dir, project_path.clone())
                .remove_asset_references(asset_id)?;
            purge_asset_record(&project_path, asset_id)?;
            index_mutation.commit();
            return Ok(AssetMutationResult {
                id: asset_id.to_owned(),
                status: "purged".to_owned(),
            });
        }
        let index_mutation = AssetIndexMutation::begin(&project_path)?;
        remove_sibling_poster_no_symlinks(&project_path, media_rel)?;
        CharacterStore::new(&self.data_dir, project_path.clone())
            .remove_asset_references(asset_id)?;
        if let Some(media_path) = media_path.filter(|path| path.exists() && path.is_file()) {
            fs::remove_file(media_path)?;
        }
        if let Some(preview_path) = preview_path.filter(|path| path.exists() && path.is_file()) {
            fs::remove_file(preview_path)?;
        }
        remove_asset_sidecar(&sidecar_path)?;
        if let Some(parent) = asset_trash_dir {
            fs::remove_dir_all(parent).ok();
        }
        purge_asset_record(&project_path, asset_id)?;
        index_mutation.commit();
        Ok(AssetMutationResult {
            id: asset_id.to_owned(),
            status: "purged".to_owned(),
        })
    }
}
