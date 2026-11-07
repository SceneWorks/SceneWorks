//! ProjectStore training aggregate.

use super::*;

impl ProjectStore {
    pub fn list_training_datasets(
        &self,
        project_id: &str,
    ) -> ProjectStoreResult<Vec<TrainingDatasetSummary>> {
        let project_path = self.find_project_path(project_id)?;
        TrainingDatasetStore::new(project_path).list_datasets(project_id)
    }

    pub fn create_training_dataset(
        &self,
        project_id: &str,
        input: TrainingDatasetCreateInput,
    ) -> ProjectStoreResult<TrainingDataset> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        TrainingDatasetStore::new(project_path).create_dataset(project_id, input)
    }

    pub fn create_training_dataset_from_external(
        &self,
        project_id: &str,
        input: TrainingDatasetCreateInput,
        items: Vec<ExternalTrainingDatasetItemInput>,
        extra: ExtraFields,
        idempotency_key: &str,
    ) -> ProjectStoreResult<(TrainingDataset, bool)> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        TrainingDatasetStore::new(project_path).create_dataset_from_external(
            project_id,
            input,
            items,
            extra,
            idempotency_key,
        )
    }

    pub fn get_catalog_materialization(
        &self,
        project_id: &str,
        idempotency_key: &str,
    ) -> ProjectStoreResult<Option<TrainingDataset>> {
        let project_path = self.find_project_path(project_id)?;
        TrainingDatasetStore::new(project_path)
            .get_catalog_materialization(project_id, idempotency_key)
    }

    pub fn upload_training_dataset_item(
        &self,
        project_id: &str,
        upload: TrainingDatasetUpload,
    ) -> ProjectStoreResult<Value> {
        if fs::metadata(&upload.source_path)?.len() == 0 {
            return Err(ProjectStoreError::BadRequest(
                "Uploaded file is empty".to_owned(),
            ));
        }
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        let upload_dir = project_path.join("training").join("uploads");
        fs::create_dir_all(&upload_dir)?;

        let guessed_mime = guess_mime_from_filename(&upload.filename);
        let content_type = upload
            .content_type
            .as_deref()
            .filter(|value| !value.is_empty() && *value != "application/octet-stream")
            .map(str::to_owned)
            .or(guessed_mime)
            .unwrap_or_else(|| "application/octet-stream".to_owned());
        // SVG is active document syntax, not a generic upload format. Vector assets may enter
        // only through the worker-owned `vector_generate` sanitizer/rasterizer boundary, which
        // writes a canonical SVG plus a bounded PNG preview together (sc-22251).
        if content_type.eq_ignore_ascii_case("image/svg+xml")
            || Path::new(&upload.filename)
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("svg"))
        {
            return Err(ProjectStoreError::BadRequest(
                "SVG uploads are not supported; use Vector Studio".to_owned(),
            ));
        }
        if !content_type.starts_with("image/") {
            return Err(ProjectStoreError::BadRequest(
                "Only image dataset uploads are supported".to_owned(),
            ));
        }

        // sc-6143: training reads dataset images straight through the engine (no worker decode
        // backstop on this path), so normalize a valid-but-unsupported format to lossless PNG here
        // — the only lever that makes AVIF/HEIC training images decode.
        let normalized = normalize_image_upload(
            &upload.source_path,
            &content_type,
            &upload.filename,
            &upload_dir,
            // A dataset item is not an asset: it has no `extra` slot, and no reader offers its
            // recipe back. So this lane opts out of the sc-15949 chunk scan rather than walking the
            // metadata of every image in a bulk dataset import for an answer nobody reads.
            WorkflowScan::Skip,
        )?;
        let content_type = normalized.content_type.clone();

        let upload_id = format!("dataset_upload_{}", random_hex(16)?);
        let extension = normalized.extension.clone();
        let suffix = &upload_id[upload_id.len().saturating_sub(8)..];
        let filename = format!(
            "{}-{suffix}{extension}",
            safe_filename(&upload.filename, &upload_id)
        );
        let media_path = upload_dir.join(filename);
        move_normalized_upload(&normalized, &upload.source_path, &media_path)?;
        let media_rel = relative_string(&project_path, &media_path)?;
        let display_name = Path::new(&upload.filename)
            .file_name()
            .and_then(|value| value.to_str())
            .filter(|value| !value.is_empty())
            .unwrap_or("dataset-upload")
            .to_owned();

        // sc-6531 (Dataset Doctor): emit the stored image's real pixel dimensions + a content
        // hash instead of the historical `Null`s, so the asset layer / pre-train readout
        // (`datasetHelpers.js`) and Tier-0 checks have them up front. Read from the normalized
        // stored file — header-only dims + a streamed hash, no full decode (core stays codec-free).
        let (width, height) = crate::media_convert::image_dimensions(&media_path)
            .map_or((Value::Null, Value::Null), |(w, h)| (json!(w), json!(h)));
        let content_hash = crate::media_convert::file_content_hash(&media_path)
            .ok()
            .map_or(Value::Null, Value::String);

        Ok(json!({
            "id": upload_id,
            "projectId": project_id,
            "datasetOnly": true,
            "type": "image",
            "displayName": display_name,
            "file": {
                "path": media_rel,
                "mimeType": content_type,
                "width": width,
                "height": height,
                "contentHash": content_hash
            },
            "url": format!("/api/v1/projects/{project_id}/files/{media_rel}")
        }))
    }

    pub fn get_training_dataset(
        &self,
        project_id: &str,
        dataset_id: &str,
    ) -> ProjectStoreResult<TrainingDataset> {
        let project_path = self.find_project_path(project_id)?;
        TrainingDatasetStore::new(project_path).get_dataset(project_id, dataset_id)
    }

    /// Loads a training dataset together with its absolute on-disk root and the
    /// project stem — the inputs the Rust API needs to resolve a
    /// [`crate::training::TrainingPlan`] and label the queued job. Resolves the
    /// project path once rather than re-reading the registry per value.
    pub fn training_dataset_for_plan(
        &self,
        project_id: &str,
        dataset_id: &str,
    ) -> ProjectStoreResult<(TrainingDataset, PathBuf, String)> {
        let project_path = self.find_project_path(project_id)?;
        let dataset =
            TrainingDatasetStore::new(project_path.clone()).get_dataset(project_id, dataset_id)?;
        let root = crate::training_store::dataset_root(&project_path, dataset_id);
        let project_stem = project_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or_default()
            .to_owned();
        Ok((dataset, root, project_stem))
    }

    pub fn update_training_dataset(
        &self,
        project_id: &str,
        dataset_id: &str,
        input: TrainingDatasetUpdateInput,
    ) -> ProjectStoreResult<TrainingDataset> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        TrainingDatasetStore::new(project_path).update_dataset(project_id, dataset_id, input)
    }

    /// Finalize an empty import against its observed revision under the same
    /// project lock as materialization and the dataset write.
    pub fn finalize_training_dataset_import(
        &self,
        project_id: &str,
        dataset_id: &str,
        expected_version: u32,
        items: Vec<crate::training_store::TrainingDatasetItemInput>,
    ) -> ProjectStoreResult<TrainingDataset> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        let store = TrainingDatasetStore::new(project_path);
        let current = store.get_dataset(project_id, dataset_id)?;
        if current.version != expected_version {
            return Err(ProjectStoreError::Conflict(format!(
                "Dataset changed during Parquet import (expected version {}, found {}).",
                expected_version, current.version
            )));
        }
        if !current.items.is_empty() {
            return Err(ProjectStoreError::Conflict(
                "Dataset is no longer empty; Parquet import was not applied.".to_owned(),
            ));
        }
        store.update_dataset(
            project_id,
            dataset_id,
            TrainingDatasetUpdateInput {
                name: None,
                status: None,
                character_id: None,
                items: Some(items),
                preserve_caption_item_ids: Vec::new(),
            },
        )
    }

    pub fn install_ltx_prepared_bundle(
        &self,
        project_id: &str,
        dataset_id: &str,
        item_id: &str,
        source_path: &Path,
    ) -> ProjectStoreResult<TrainingDataset> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        TrainingDatasetStore::new(project_path).install_ltx_prepared_bundle(
            project_id,
            dataset_id,
            item_id,
            source_path,
        )
    }

    /// Install or remove an Iris training sidecar of a dataset item (sc-25687 / sc-25688).
    pub fn set_iris_item_sidecar(
        &self,
        project_id: &str,
        dataset_id: &str,
        item_id: &str,
        sidecar: &crate::training_store::IrisItemSidecar,
    ) -> ProjectStoreResult<TrainingDataset> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        TrainingDatasetStore::new(project_path)
            .set_iris_item_sidecar(project_id, dataset_id, item_id, sidecar)
    }

    /// Persist freshly-extracted Tier-0 scalars onto dataset items as the readiness cache (sc-6533).
    /// Locked like any dataset mutation; a metadata-only write (no version/`updated_at` bump).
    pub fn cache_dataset_tier0_scalars(
        &self,
        project_id: &str,
        dataset_id: &str,
        updates: &[(String, CachedTier0Scalars)],
    ) -> ProjectStoreResult<()> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        TrainingDatasetStore::new(project_path).cache_tier0_scalars(project_id, dataset_id, updates)
    }

    /// Persist (or clear) a per-image quality override (sc-6534). Locked wrapper over
    /// [`TrainingDatasetStore::set_item_quality_ack`].
    pub fn set_dataset_item_quality_ack(
        &self,
        project_id: &str,
        dataset_id: &str,
        item_id: &str,
        checks: &[QualityCheck],
        expected_content_hash: Option<&str>,
        expected_caption_hash: Option<&str>,
    ) -> ProjectStoreResult<Option<QualityAck>> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        TrainingDatasetStore::new(project_path).set_item_quality_ack(
            project_id,
            dataset_id,
            item_id,
            checks,
            expected_content_hash,
            expected_caption_hash,
        )
    }

    /// Persist the dataset's CLIP embeddings sidecar (sc-6535). Locked wrapper over
    /// [`TrainingDatasetStore::write_dataset_embeddings`].
    pub fn write_dataset_embeddings(
        &self,
        project_id: &str,
        dataset_id: &str,
        embeddings: &DatasetEmbeddings,
    ) -> ProjectStoreResult<()> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        TrainingDatasetStore::new(project_path)
            .write_dataset_embeddings(project_id, dataset_id, embeddings)
    }

    /// Re-point dataset items at derived (e.g. upscaled) child assets (sc-6539). Locked wrapper over
    /// [`TrainingDatasetStore::repoint_dataset_items`].
    pub fn repoint_dataset_items(
        &self,
        project_id: &str,
        dataset_id: &str,
        repoints: &[DatasetItemRepoint],
    ) -> ProjectStoreResult<TrainingDataset> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        TrainingDatasetStore::new(project_path)
            .repoint_dataset_items(project_id, dataset_id, repoints)
    }

    /// Backfill a training dataset's valid-but-unsupported images (AVIF/HEIC/…) to PNG in place
    /// (sc-8225) — the one-time fix for datasets built before normalization, run pre-train so the
    /// engine (which reads dataset images with no decode backstop) never sees an undecodable format.
    /// Locked wrapper over [`TrainingDatasetStore::normalize_unsupported_images`].
    pub fn normalize_training_dataset_unsupported_images(
        &self,
        project_id: &str,
        dataset_id: &str,
    ) -> ProjectStoreResult<TrainingDataset> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        TrainingDatasetStore::new(project_path).normalize_unsupported_images(project_id, dataset_id)
    }

    /// Read the dataset's CLIP embeddings sidecar (`None` if the analysis job hasn't run). Locked
    /// wrapper over [`TrainingDatasetStore::read_dataset_embeddings`].
    pub fn read_dataset_embeddings(
        &self,
        project_id: &str,
        dataset_id: &str,
    ) -> ProjectStoreResult<Option<DatasetEmbeddings>> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        TrainingDatasetStore::new(project_path).read_dataset_embeddings(project_id, dataset_id)
    }

    /// Persist the dataset's face sidecar (sc-6538). Locked wrapper over
    /// [`TrainingDatasetStore::write_dataset_faces`].
    pub fn write_dataset_faces(
        &self,
        project_id: &str,
        dataset_id: &str,
        faces: &DatasetFaceRecords,
    ) -> ProjectStoreResult<()> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        TrainingDatasetStore::new(project_path).write_dataset_faces(project_id, dataset_id, faces)
    }

    /// Read the dataset's face sidecar (`None` if the worker face pass hasn't run). Locked wrapper
    /// over [`TrainingDatasetStore::read_dataset_faces`].
    pub fn read_dataset_faces(
        &self,
        project_id: &str,
        dataset_id: &str,
    ) -> ProjectStoreResult<Option<DatasetFaceRecords>> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        TrainingDatasetStore::new(project_path).read_dataset_faces(project_id, dataset_id)
    }

    /// The images a `dataset_subject_mask` job segments (sc-2126). Locked wrapper over
    /// [`TrainingDatasetStore::subject_mask_targets`] (it may backfill legacy content hashes).
    pub fn training_dataset_subject_mask_targets(
        &self,
        project_id: &str,
        dataset_id: &str,
        item_ids: Option<&[String]>,
    ) -> ProjectStoreResult<(
        Vec<crate::training_subject_masks::SubjectMaskTarget>,
        PathBuf,
        String,
    )> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        let targets = TrainingDatasetStore::new(project_path.clone())
            .subject_mask_targets(project_id, dataset_id, item_ids)?;
        let root = crate::training_store::dataset_root(&project_path, dataset_id);
        let project_stem = project_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or_default()
            .to_owned();
        Ok((targets, root, project_stem))
    }

    /// Persist generated or uploaded subject masks (sc-2126). Locked wrapper over
    /// [`TrainingDatasetStore::write_subject_masks`].
    pub fn write_training_dataset_subject_masks(
        &self,
        project_id: &str,
        dataset_id: &str,
        writes: Vec<crate::training_subject_masks::SubjectMaskWrite>,
        source: crate::training_subject_masks::SubjectMaskSource,
    ) -> ProjectStoreResult<crate::training_subject_masks::SubjectMaskReport> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        TrainingDatasetStore::new(project_path)
            .write_subject_masks(project_id, dataset_id, writes, source)
    }

    /// Install a user-uploaded replacement subject mask (sc-2126). Locked wrapper over
    /// [`TrainingDatasetStore::upload_subject_mask`].
    pub fn upload_training_dataset_subject_mask(
        &self,
        project_id: &str,
        dataset_id: &str,
        item_id: &str,
        source_path: &Path,
    ) -> ProjectStoreResult<crate::training_subject_masks::SubjectMaskReport> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        TrainingDatasetStore::new(project_path).upload_subject_mask(
            project_id,
            dataset_id,
            item_id,
            source_path,
        )
    }

    /// A dataset's subject-mask coverage (sc-2126). Locked wrapper over
    /// [`TrainingDatasetStore::subject_mask_report`].
    pub fn training_dataset_subject_mask_report(
        &self,
        project_id: &str,
        dataset_id: &str,
    ) -> ProjectStoreResult<crate::training_subject_masks::SubjectMaskReport> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        TrainingDatasetStore::new(project_path).subject_mask_report(project_id, dataset_id)
    }

    pub fn batch_rename_training_dataset_items(
        &self,
        project_id: &str,
        dataset_id: &str,
        input: TrainingDatasetBatchRenameInput,
    ) -> ProjectStoreResult<TrainingDataset> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        TrainingDatasetStore::new(project_path)
            .batch_rename_dataset_items(project_id, dataset_id, input)
    }

    pub fn write_training_dataset_caption_sidecars(
        &self,
        project_id: &str,
        dataset_id: &str,
        input: TrainingDatasetCaptionSidecarsInput,
    ) -> ProjectStoreResult<TrainingCaptionSidecarsResult> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        TrainingDatasetStore::new(project_path)
            .write_caption_sidecars(project_id, dataset_id, input)
    }

    pub fn delete_training_dataset(
        &self,
        project_id: &str,
        dataset_id: &str,
    ) -> ProjectStoreResult<TrainingDatasetMutationResult> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        TrainingDatasetStore::new(project_path).delete_dataset(project_id, dataset_id)
    }
}
