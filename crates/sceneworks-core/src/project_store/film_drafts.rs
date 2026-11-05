//! ProjectStore film drafts aggregate.

use super::*;

pub(super) fn copy_film_reference_files(
    project_path: &Path,
    draft_id: &str,
    pack: &ReferencePack,
    destination_root: &Path,
    include_sound: bool,
) -> ProjectStoreResult<()> {
    // Nothing to copy when no entry names a file on disk. A DESCRIBED-ONLY reference role is words
    // and nothing else (sc-24025), exactly as a synthesized dialogue line is, so a pack made only
    // of them stages no images — and must not be sent through the staging directory lookup below,
    // which refuses a draft whose immutable input directory was never created.
    if pack
        .references
        .iter()
        .all(|reference| reference.file().is_none())
        && (!include_sound || pack.sound.iter().all(|sound| sound.file.is_none()))
    {
        return Ok(());
    }
    let destination_relative = destination_root.strip_prefix(project_path).map_err(|_| {
        ProjectStoreError::BadRequest(
            "Film reference destination must remain inside its project".to_owned(),
        )
    })?;
    if destination_relative
        .components()
        .any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        return Err(ProjectStoreError::BadRequest(
            "Film reference destination must remain inside its project".to_owned(),
        ));
    }

    let canonical_project = fs::canonicalize(project_path)?;
    fs::create_dir_all(destination_root)?;
    let canonical_destination = fs::canonicalize(destination_root)?;
    if !canonical_destination.starts_with(&canonical_project) {
        return Err(ProjectStoreError::BadRequest(
            "Film reference destination must remain inside its project".to_owned(),
        ));
    }

    let draft_assets = project_path.join("films/draft-assets").join(draft_id);
    let canonical_assets = fs::canonicalize(&draft_assets).map_err(|_| {
        ProjectStoreError::BadRequest(format!(
            "Film draft {draft_id:?} is missing its immutable reference directory"
        ))
    })?;
    if !canonical_assets.starts_with(&canonical_project) {
        return Err(ProjectStoreError::BadRequest(
            "Film reference source must remain inside its project".to_owned(),
        ));
    }
    for reference in &pack.references {
        // A described-only role stages no image, so there is nothing to find or copy (sc-24025).
        let Some(file) = reference.file() else {
            continue;
        };
        stage_draft_file(
            &canonical_assets,
            &canonical_destination,
            file,
            &reference.role,
            "reference",
            "image",
            false,
        )?;
    }
    if include_sound {
        for sound in &pack.sound {
            let Some(file) = sound.file.as_deref() else {
                continue;
            };
            // Generated dialogue may reserve a filename that the harness creates later.
            stage_draft_file(
                &canonical_assets,
                &canonical_destination,
                file,
                &sound.role,
                "sound",
                "audio",
                sound.text.is_some(),
            )?;
        }
    }
    Ok(())
}

/// Copy one immutable draft input; described-only references are skipped by the caller.
pub(super) fn stage_draft_file(
    draft_assets: &Path,
    destination_root: &Path,
    file: &str,
    role: &str,
    kind: &str,
    media: &str,
    allow_missing: bool,
) -> ProjectStoreResult<()> {
    if !is_safe_relative_path(file) {
        return Err(ProjectStoreError::BadRequest(format!(
            "Film {kind} {role:?} has an unsafe file path"
        )));
    }
    let source = draft_assets.join(file);
    if allow_missing && !source.is_file() {
        return Ok(());
    }
    let canonical_source = fs::canonicalize(&source).map_err(|_| {
        ProjectStoreError::BadRequest(format!(
            "Film {kind} {role:?} is missing its staged {media} {file}"
        ))
    })?;
    if !canonical_source.starts_with(draft_assets) || !canonical_source.is_file() {
        return Err(ProjectStoreError::BadRequest(format!(
            "Film {kind} {role:?} resolves outside its immutable draft input"
        )));
    }
    let destination = destination_root.join(file);
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::copy(canonical_source, destination)?;
    Ok(())
}

impl ProjectStore {
    pub fn list_film_drafts(&self, project_id: &str) -> ProjectStoreResult<Vec<FilmDraft>> {
        let project_path = self.find_project_path(project_id)?;
        let drafts_dir = project_path.join("films/drafts");
        if !drafts_dir.exists() {
            return Ok(Vec::new());
        }
        let mut drafts = Vec::new();
        for path in read_dir_paths(&drafts_dir)? {
            if path.extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            let draft = Self::carry_film_draft_forward(read_json(&path)?)?;
            if draft.project_id != project_id {
                return Err(ProjectStoreError::BadRequest(format!(
                    "film draft {} belongs to a different project",
                    draft.id
                )));
            }
            drafts.push(draft);
        }
        drafts.sort_by(|left, right| right.updated_at.cmp(&left.updated_at));
        Ok(drafts)
    }

    pub fn create_film_draft(
        &self,
        project_id: &str,
        draft_id: &str,
        title: &str,
    ) -> ProjectStoreResult<FilmDraft> {
        self.create_film_draft_document(
            project_id,
            FilmDraft::manual_one_shot(project_id, draft_id, title),
        )
    }

    /// Persist a freshly constructed draft without a revision bump. The API uses this after it has
    /// resolved host-dependent render defaults; ordinary edits still go through
    /// [`Self::save_film_draft`] and optimistic revision checks.
    pub fn create_film_draft_document(
        &self,
        project_id: &str,
        draft: FilmDraft,
    ) -> ProjectStoreResult<FilmDraft> {
        let draft_id = draft.id.as_str();
        if !is_safe_id(draft_id) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid film draft ID".to_owned(),
            ));
        }
        if draft.project_id != project_id || draft.revision != 1 {
            return Err(ProjectStoreError::BadRequest(
                "New film draft identity or revision does not match its project".to_owned(),
            ));
        }
        let (project_path, _guard) = self.lock_project(project_id)?;
        let path = project_path
            .join("films/drafts")
            .join(format!("{draft_id}.json"));
        if path.exists() {
            return Err(ProjectStoreError::BadRequest(
                "Film draft already exists".to_owned(),
            ));
        }
        write_json(&path, &draft)?;
        Ok(draft)
    }

    /// Carry a persisted film draft forward to the schema versions this build reads, then decode
    /// it (sc-24024, sc-24026).
    ///
    /// THE one place a draft read from `films/drafts/<id>.json` is brought up to date, called from
    /// every read path so they cannot drift, and the place a later schema bump extends rather than
    /// adding a fourth copy of the same idea.
    ///
    /// It takes the raw [`Value`] rather than a decoded [`FilmDraft`] because part of the
    /// carry-forward has to happen BEFORE serde sees it. Plan schema version 3 renamed the optional
    /// `shots[].sound` to a required `shots[].audio`, and [`crate::film_plan::Shot`] is
    /// `deny_unknown_fields`: a draft written by an earlier build has both faults at once and each
    /// is a hard decode error, so no helper that accepts a `FilmDraft` can ever run on one. The
    /// reference-pack stamp is applied after the decode, on the typed value, because a version 1
    /// pack decodes perfectly well.
    ///
    /// A draft is project-store STATE, not a document a person authors: it is deserialized verbatim
    /// and the workspace offers no way to edit a `schemaVersion` key inside it. Left alone, a
    /// pre-existing draft breaks in two ways at once. The plan faults above make a single stale
    /// draft 500 its own GET — and, because `list_film_drafts` propagates, take the whole Films
    /// list for that project down with it, so the user cannot even reach the draft to repair it.
    /// And when the reference pack went to version 2 (sc-24024), every draft on disk still said
    /// version 1, while `validate_reference_pack` — which `add_film_reference`, `add_film_sound`
    /// and the pack PUT all run — refuses a pack by version, so each of those would fail on every
    /// pre-existing draft with a refusal telling the user to make an edit they cannot make.
    ///
    /// Carrying forward is honest here and only here, and for a different reason per field. Pack
    /// version 2 only ADDS the optional `locator`, so a version 1 draft pack IS a structurally
    /// valid version 2 pack. The shot's old `sound` prose moves into `audio` when it said something
    /// and blanks when it did not, which is the truth in both cases: the version 2 field meant the
    /// same thing, and a draft that never had one has an author who has not yet said what the shot
    /// sounds like. Blank is not silently accepted — it surfaces as the shot-named `audio` finding,
    /// in the Audio field the workspace now shows, which is exactly where it gets fixed.
    ///
    /// DOCUMENTS on disk stay refused by version, both kinds: a pack document under E6, and a plan
    /// document in [`crate::film_plan::parse_plan_document`]. Those have an author who can edit
    /// them.
    pub(super) fn carry_film_draft_forward(mut stored: Value) -> ProjectStoreResult<FilmDraft> {
        Self::carry_film_draft_plan_forward(&mut stored);
        let mut draft: FilmDraft = serde_json::from_value(stored)?;
        draft.reference_pack.schema_version = REFERENCE_PACK_SCHEMA_VERSION;
        Ok(draft)
    }

    /// The JSON-level half of [`Self::carry_film_draft_forward`]: the production-plan fixups that
    /// must land before the typed decode, because each of them is a hard serde error (sc-24026).
    fn carry_film_draft_plan_forward(stored: &mut Value) {
        let Some(plan) = stored
            .get_mut("productionPlan")
            .and_then(Value::as_object_mut)
        else {
            return;
        };
        if let Some(shots) = plan.get_mut("shots").and_then(Value::as_array_mut) {
            for shot in shots {
                let Some(shot) = shot.as_object_mut() else {
                    continue;
                };
                // Removed either way: version 3 has no `sound` on a shot, and leaving the key would
                // be an unknown field whichever way `audio` resolved.
                let carried = shot.remove("sound");
                if !shot.contains_key("audio") {
                    let audio = carried
                        .as_ref()
                        .and_then(Value::as_str)
                        .filter(|value| !value.trim().is_empty())
                        .unwrap_or_default()
                        .to_owned();
                    shot.insert("audio".to_owned(), Value::String(audio));
                }
            }
        }
        // Stamped last, and only from a version this carry-forward actually handles: an unknown
        // FUTURE version must stay unstamped so it is still reported rather than claimed.
        if plan
            .get("schemaVersion")
            .and_then(Value::as_u64)
            .is_some_and(|version| version == 1 || version == 2)
        {
            plan.insert("schemaVersion".to_owned(), Value::from(PLAN_SCHEMA_VERSION));
        }
    }

    pub fn get_film_draft(
        &self,
        project_id: &str,
        draft_id: &str,
    ) -> ProjectStoreResult<FilmDraft> {
        if !is_safe_id(draft_id) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid film draft ID".to_owned(),
            ));
        }
        let project_path = self.find_project_path(project_id)?;
        let path = project_path
            .join("films/drafts")
            .join(format!("{draft_id}.json"));
        if !path.exists() {
            return Err(ProjectStoreError::NotFound(
                "Film draft not found".to_owned(),
            ));
        }
        let draft = Self::carry_film_draft_forward(read_json(&path)?)?;
        if draft.id != draft_id || draft.project_id != project_id {
            return Err(ProjectStoreError::BadRequest(
                "Film draft identity does not match its project path".to_owned(),
            ));
        }
        Ok(draft)
    }

    pub fn save_film_draft(
        &self,
        project_id: &str,
        draft_id: &str,
        mut draft: FilmDraft,
    ) -> ProjectStoreResult<FilmDraft> {
        if !is_safe_id(draft_id) || draft.id != draft_id || draft.project_id != project_id {
            return Err(ProjectStoreError::BadRequest(
                "Film draft identity cannot be changed".to_owned(),
            ));
        }
        let (project_path, _guard) = self.lock_project(project_id)?;
        let path = project_path
            .join("films/drafts")
            .join(format!("{draft_id}.json"));
        if !path.exists() {
            return Err(ProjectStoreError::NotFound(
                "Film draft not found".to_owned(),
            ));
        }
        let current = Self::carry_film_draft_forward(read_json(&path)?)?;
        if draft.revision != current.revision {
            return Err(ProjectStoreError::Conflict(format!(
                "Film draft revision conflict: expected {}, got {}",
                current.revision, draft.revision
            )));
        }
        // A production+compiled import is one document transaction. If the supplied compile
        // exactly matches the supplied plan, carry it across the store-owned revision bump by
        // updating only its identity hash. If the plan was edited independently, preserve the old
        // identity so preflight reports it stale and rendering cannot silently use old prompts.
        //
        // The question asked here is about the PLAN only (sc-24029). The bump this carries across
        // rewrites the plan's identity — `version`, and therefore its hash — and nothing else, so
        // a pack that no longer matches must survive the save as exactly that: one finding naming
        // the pack. Counting the pack here instead would withhold the plan carry-across, and a
        // pack edit would then surface as a STALE PLAN as well, putting the wrong cause first in
        // the refusal the operator reads.
        let supplied_plan_sha = production_plan_sha256(&draft.production_plan)?;
        let compiled_was_current = draft.compiled_plan.as_ref().is_some_and(|compiled| {
            compiled
                .staleness_findings(
                    &draft.production_plan,
                    &supplied_plan_sha,
                    &draft.reference_pack,
                )
                .iter()
                // Everything BUT the pack: a pack finding is not an answer about the plan, and
                // it is deliberately carried through the save rather than resolved by it.
                .all(|finding| finding.field == COMPILED_PACK_STALENESS_FIELD)
        });
        draft.reconcile_review_plan();
        draft.revision = current.revision.saturating_add(1);
        draft.created_at = current.created_at;
        draft.updated_at = utc_now();
        draft.production_plan.id = draft.id.clone();
        draft.production_plan.version = draft.revision;
        draft.production_plan.title = draft.title.clone();
        if compiled_was_current {
            let normalized_sha = production_plan_sha256(&draft.production_plan)?;
            if let Some(compiled) = draft.compiled_plan.as_mut() {
                compiled.plan_id = draft.production_plan.id.clone();
                compiled.plan_version = draft.production_plan.version;
                compiled.plan_sha256 = normalized_sha;
            }
        }
        write_json(&path, &draft)?;
        Ok(draft)
    }

    /// Copy a project-library image into a draft-owned reference pack and persist the new draft
    /// revision. Runs later copy from this immutable draft input rather than reading mutable asset
    /// library state.
    ///
    /// With no [`FilmReferenceInput::asset_id`] this authors a DESCRIBED-ONLY role instead
    /// (sc-24025): no image is resolved, none is copied, and the entry names neither a `file` nor a
    /// `sourceAssetId`. Both shapes are checked by the one [`validate_reference_pack`] below, so a
    /// described-only role with no `description` is refused in the core's own words rather than by
    /// a rule this route states a second time.
    pub fn add_film_reference(
        &self,
        project_id: &str,
        draft_id: &str,
        input: FilmReferenceInput,
    ) -> ProjectStoreResult<FilmDraft> {
        if !is_safe_id(draft_id)
            || input
                .asset_id
                .as_deref()
                .is_some_and(|asset_id| !is_safe_id(asset_id))
        {
            return Err(ProjectStoreError::BadRequest(
                "Invalid film draft or source asset ID".to_owned(),
            ));
        }
        if !REFERENCE_KINDS.contains(&input.kind.as_str()) {
            return Err(ProjectStoreError::BadRequest(format!(
                "Reference kind must be one of {}",
                REFERENCE_KINDS.join(", ")
            )));
        }
        // Resolved BEFORE the project lock exactly as it always was, and only for a role that
        // names an image. A described-only role has no bytes to find, stat or copy.
        let image = match input.asset_id.as_deref() {
            Some(asset_id) => {
                let source = self.resolve_asset_media_path(project_id, asset_id)?;
                let extension = source
                    .extension()
                    .and_then(|value| value.to_str())
                    .map(str::to_ascii_lowercase)
                    .filter(|value| matches!(value.as_str(), "png" | "jpg" | "jpeg" | "webp"))
                    .ok_or_else(|| {
                        ProjectStoreError::BadRequest(
                            "Film references must use a PNG, JPEG, or WebP image asset".to_owned(),
                        )
                    })?;
                Some((source, format!("references/{asset_id}.{extension}")))
            }
            None => None,
        };

        let (project_path, _guard) = self.lock_project(project_id)?;
        let draft_path = project_path
            .join("films/drafts")
            .join(format!("{draft_id}.json"));
        if !draft_path.exists() {
            return Err(ProjectStoreError::NotFound(
                "Film draft not found".to_owned(),
            ));
        }
        let mut draft = Self::carry_film_draft_forward(read_json(&draft_path)?)?;
        if draft.id != draft_id || draft.project_id != project_id {
            return Err(ProjectStoreError::BadRequest(
                "Film draft identity does not match its project path".to_owned(),
            ));
        }
        if input.draft_revision != draft.revision {
            return Err(ProjectStoreError::Conflict(format!(
                "Film draft revision conflict: expected {}, got {}",
                draft.revision, input.draft_revision
            )));
        }

        draft.reference_pack.references.push(ReferenceEntry {
            role: input.role,
            kind: input.kind,
            // BOTH or NEITHER (sc-24025). An entry naming a `sourceAssetId` with no `file` is
            // refused by `validate_reference_pack` as an image-backed role that forgot where its
            // picture is, so the described-only shape has to drop the asset id too — which it
            // does by construction here, because both are read from the one `image` above.
            file: image.as_ref().map(|(_, file)| file.clone()),
            source_asset_id: input.asset_id,
            description: input.description,
            locator: input.locator,
            approved: input.approved,
            generated: false,
            generation: None,
        });
        draft.reference_pack.version = draft.reference_pack.version.saturating_add(1);
        let findings = validate_reference_pack(&draft.reference_pack);
        if !findings.is_empty() {
            return Err(ProjectStoreError::BadRequest(
                findings
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("; "),
            ));
        }

        if let Some((source, relative_file)) = image {
            let destination = project_path
                .join("films/draft-assets")
                .join(draft_id)
                .join(&relative_file);
            if let Some(parent) = destination.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(&source, &destination)?;
        }
        draft.revision = draft.revision.saturating_add(1);
        draft.production_plan.version = draft.revision;
        draft.updated_at = utc_now();
        write_json(&draft_path, &draft)?;
        Ok(draft)
    }

    /// Pin a draft revision's immutable reference bytes beside one planning operation. The caller
    /// writes the matching `references.json`; this method guarantees every file named by that
    /// document is copied from the draft-owned store into the contained operation directory.
    pub fn stage_film_planning_references(
        &self,
        project_id: &str,
        draft_id: &str,
        draft_revision: u32,
        operation_id: &str,
    ) -> ProjectStoreResult<()> {
        if !is_safe_id(draft_id) || !is_safe_id(operation_id) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid film draft or planning operation ID".to_owned(),
            ));
        }
        let (project_path, _guard) = self.lock_project(project_id)?;
        let draft_path = project_path
            .join("films/drafts")
            .join(format!("{draft_id}.json"));
        if !draft_path.exists() {
            return Err(ProjectStoreError::NotFound(
                "Film draft not found".to_owned(),
            ));
        }
        let draft = Self::carry_film_draft_forward(read_json(&draft_path)?)?;
        if draft.id != draft_id || draft.project_id != project_id {
            return Err(ProjectStoreError::BadRequest(
                "Film draft identity does not match its project path".to_owned(),
            ));
        }
        if draft.revision != draft_revision {
            return Err(ProjectStoreError::Conflict(format!(
                "Film draft revision conflict: expected {}, got {draft_revision}",
                draft.revision
            )));
        }
        let operation_dir = project_path
            .join("films/planning")
            .join(draft_id)
            .join("operations")
            .join(operation_id);
        fs::create_dir_all(&operation_dir)?;
        copy_film_reference_files(
            &project_path,
            draft_id,
            &draft.reference_pack,
            &operation_dir,
            false,
        )
    }

    /// Copy a project-library audio asset into a draft-owned sound pack and persist the new draft
    /// revision. The immutable staged file is what a later run pins, even if the library asset is
    /// renamed, replaced, or removed in the meantime.
    pub fn add_film_sound(
        &self,
        project_id: &str,
        draft_id: &str,
        input: FilmSoundInput,
    ) -> ProjectStoreResult<FilmDraft> {
        if !is_safe_id(draft_id) || !is_safe_id(&input.asset_id) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid film draft or source asset ID".to_owned(),
            ));
        }
        if !crate::film_plan::SOUND_KINDS.contains(&input.kind.as_str()) {
            return Err(ProjectStoreError::BadRequest(format!(
                "Sound kind must be one of {}",
                crate::film_plan::SOUND_KINDS.join(", ")
            )));
        }
        let source = self.resolve_asset_media_path(project_id, &input.asset_id)?;
        let extension = source
            .extension()
            .and_then(|value| value.to_str())
            .map(str::to_ascii_lowercase)
            .filter(|value| {
                matches!(
                    value.as_str(),
                    "wav" | "mp3" | "m4a" | "aac" | "flac" | "ogg" | "opus"
                )
            })
            .ok_or_else(|| {
                ProjectStoreError::BadRequest(
                    "Film sound must use a WAV, MP3, M4A, AAC, FLAC, OGG, or Opus audio asset"
                        .to_owned(),
                )
            })?;

        let (project_path, _guard) = self.lock_project(project_id)?;
        let draft_path = project_path
            .join("films/drafts")
            .join(format!("{draft_id}.json"));
        if !draft_path.exists() {
            return Err(ProjectStoreError::NotFound(
                "Film draft not found".to_owned(),
            ));
        }
        let mut draft = Self::carry_film_draft_forward(read_json(&draft_path)?)?;
        if draft.id != draft_id || draft.project_id != project_id {
            return Err(ProjectStoreError::BadRequest(
                "Film draft identity does not match its project path".to_owned(),
            ));
        }
        if input.draft_revision != draft.revision {
            return Err(ProjectStoreError::Conflict(format!(
                "Film draft revision conflict: expected {}, got {}",
                draft.revision, input.draft_revision
            )));
        }

        let relative_file = format!("sound/{}.{}", input.asset_id, extension);
        draft
            .reference_pack
            .sound
            .push(crate::film_plan::SoundEntry {
                role: input.role,
                kind: input.kind,
                file: Some(relative_file.clone()),
                description: input.description,
                text: None,
                voice: None,
                model: None,
            });
        draft.reference_pack.version = draft.reference_pack.version.saturating_add(1);
        let findings = validate_reference_pack(&draft.reference_pack);
        if !findings.is_empty() {
            return Err(ProjectStoreError::BadRequest(
                findings
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("; "),
            ));
        }

        let destination = project_path
            .join("films/draft-assets")
            .join(draft_id)
            .join(&relative_file);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(&source, &destination)?;
        draft.revision = draft.revision.saturating_add(1);
        draft.production_plan.version = draft.revision;
        draft.updated_at = utc_now();
        write_json(&draft_path, &draft)?;
        Ok(draft)
    }
}
