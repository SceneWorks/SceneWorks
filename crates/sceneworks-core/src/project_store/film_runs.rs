//! ProjectStore film runs aggregate.

use super::*;

impl ProjectStore {
    pub fn create_film_run(
        &self,
        project_id: &str,
        run_locator_id: &str,
        draft_id: &str,
        selected_shot_ids: Vec<String>,
        compiled: Option<CompiledPlan>,
    ) -> ProjectStoreResult<FilmRunLocator> {
        self.pin_film_run(
            project_id,
            run_locator_id,
            draft_id,
            None,
            selected_shot_ids,
            compiled,
        )
    }

    /// Pin exactly the draft the caller validated; the comparison and all documents share a lock.
    pub fn create_film_run_at_revision(
        &self,
        project_id: &str,
        run_locator_id: &str,
        draft_id: &str,
        expected_revision: u32,
        selected_shot_ids: Vec<String>,
    ) -> ProjectStoreResult<FilmRunLocator> {
        self.pin_film_run(
            project_id,
            run_locator_id,
            draft_id,
            Some(expected_revision),
            selected_shot_ids,
            None,
        )
    }

    fn pin_film_run(
        &self,
        project_id: &str,
        run_locator_id: &str,
        draft_id: &str,
        expected_revision: Option<u32>,
        selected_shot_ids: Vec<String>,
        compiled: Option<CompiledPlan>,
    ) -> ProjectStoreResult<FilmRunLocator> {
        if !is_safe_id(run_locator_id) || !is_safe_id(draft_id) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid film run ID".to_owned(),
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
        if expected_revision.is_some_and(|expected| expected != draft.revision) {
            return Err(ProjectStoreError::Conflict(format!(
                "Film draft revision conflict: expected {}, found {}; reload and preflight the draft before rendering",
                expected_revision.expect("checked"), draft.revision)));
        }
        let selected_shot_ids = if selected_shot_ids.is_empty() {
            draft
                .production_plan
                .shots
                .iter()
                .map(|shot| shot.id.clone())
                .collect()
        } else {
            selected_shot_ids
        };
        if selected_shot_ids.iter().any(|id| {
            !draft
                .production_plan
                .shots
                .iter()
                .any(|shot| &shot.id == id)
        }) {
            return Err(ProjectStoreError::BadRequest(
                "Film run selection contains an unknown shot".to_owned(),
            ));
        }
        let compiled = compiled.or_else(|| draft.compiled_plan.clone());
        if let Some(compiled) = &compiled {
            let sha = production_plan_sha256(&draft.production_plan)?;
            let findings =
                compiled.staleness_findings(&draft.production_plan, &sha, &draft.reference_pack);
            if !findings.is_empty() {
                // `Display`, not `{:?}` — this string reaches an operator through the API, and a
                // `Vec<PlanDiagnostic>` struct dump is the store's private notation (sc-24029).
                return Err(ProjectStoreError::BadRequest(format!(
                    "Film compiled plan does not match the pinned draft: {}",
                    findings
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join("; ")
                )));
            }
        }
        let mut draft = draft;
        draft.reconcile_review_plan();
        let relative_dir = format!("films/runs/{run_locator_id}");
        let run_dir = project_path.join(&relative_dir);
        if run_dir.exists() {
            return Err(ProjectStoreError::BadRequest(
                "Film run already exists".to_owned(),
            ));
        }
        fs::create_dir_all(&run_dir)?;
        let pin_result = (|| -> ProjectStoreResult<()> {
            copy_film_reference_files(
                &project_path,
                draft_id,
                &draft.reference_pack,
                &run_dir,
                true,
            )?;
            write_json(&run_dir.join("authoring.json"), &draft)?;
            write_json(&run_dir.join("plan.json"), &draft.production_plan)?;
            write_json(&run_dir.join("references.json"), &draft.reference_pack)?;
            write_json(&run_dir.join("review.jsonc"), &draft.review_plan_for_run())?;
            if let Some(compiled) = compiled.as_ref() {
                write_json(&run_dir.join("compiled.json"), compiled)?;
            }
            Ok(())
        })();
        if let Err(error) = pin_result {
            let _ = fs::remove_dir_all(&run_dir);
            return Err(error);
        }
        let locator = FilmRunLocator {
            schema_version: crate::film_workspace::FILM_RUN_LOCATOR_SCHEMA_VERSION,
            id: run_locator_id.to_owned(),
            project_id: project_id.to_owned(),
            draft_id: draft_id.to_owned(),
            draft_revision: draft.revision,
            selected_shot_ids,
            record_directory: relative_dir,
            authoring_snapshot_path: Some(format!("films/runs/{run_locator_id}/authoring.json")),
            created_at: utc_now(),
        };
        write_json(&run_dir.join("locator.json"), &locator)?;
        Ok(locator)
    }

    pub fn get_film_run(
        &self,
        project_id: &str,
        run_locator_id: &str,
    ) -> ProjectStoreResult<FilmRunLocator> {
        let files = self.film_run_files(project_id, run_locator_id)?;
        let path = files.directory.join("locator.json");
        if !path.exists() {
            return Err(ProjectStoreError::NotFound("Film run not found".to_owned()));
        }
        let locator: FilmRunLocator = serde_json::from_value(read_json(&path)?)?;
        if locator.id != run_locator_id || locator.project_id != project_id {
            return Err(ProjectStoreError::BadRequest(
                "Film run identity does not match its project path".to_owned(),
            ));
        }
        Ok(locator)
    }

    /// Read the complete authoring document retained for a run. Locators created before authoring
    /// snapshots were introduced return `None`; derived render documents are never used to invent
    /// a historical draft.
    pub fn get_film_run_authoring_snapshot(
        &self,
        project_id: &str,
        run_locator_id: &str,
    ) -> ProjectStoreResult<Option<FilmDraft>> {
        let locator = self.get_film_run(project_id, run_locator_id)?;
        let Some(relative_path) = locator.authoring_snapshot_path.as_deref() else {
            return Ok(None);
        };
        if !is_safe_relative_path(relative_path) {
            return Err(ProjectStoreError::BadRequest(
                "Film run authoring snapshot path must be project-relative".to_owned(),
            ));
        }
        let files = self.film_run_files(project_id, run_locator_id)?;
        let expected_path = format!("{}/authoring.json", locator.record_directory);
        if relative_path != expected_path {
            return Err(ProjectStoreError::BadRequest(
                "Film run authoring snapshot path does not match its run directory".to_owned(),
            ));
        }
        let snapshot = Self::carry_film_draft_forward(read_json(&files.authoring_snapshot)?)?;
        if snapshot.id != locator.draft_id
            || snapshot.project_id != locator.project_id
            || snapshot.revision != locator.draft_revision
        {
            return Err(ProjectStoreError::BadRequest(
                "Film run authoring snapshot identity does not match its locator".to_owned(),
            ));
        }
        Ok(Some(snapshot))
    }

    pub fn list_film_runs(&self, project_id: &str) -> ProjectStoreResult<Vec<FilmRunLocator>> {
        let project_path = self.find_project_path(project_id)?;
        let runs_dir = project_path.join("films/runs");
        if !runs_dir.exists() {
            return Ok(Vec::new());
        }
        let mut runs = Vec::new();
        for directory in read_dir_paths(&runs_dir)? {
            let locator_path = directory.join("locator.json");
            if !locator_path.is_file() {
                continue;
            }
            let locator: FilmRunLocator = serde_json::from_value(read_json(&locator_path)?)?;
            if locator.project_id != project_id
                || directory.file_name().and_then(|value| value.to_str())
                    != Some(locator.id.as_str())
            {
                return Err(ProjectStoreError::BadRequest(format!(
                    "Film run identity does not match {}",
                    locator_path.display()
                )));
            }
            runs.push(locator);
        }
        runs.sort_by(|left, right| right.created_at.cmp(&left.created_at));
        Ok(runs)
    }

    pub fn film_run_files(
        &self,
        project_id: &str,
        run_locator_id: &str,
    ) -> ProjectStoreResult<FilmRunFiles> {
        if !is_safe_id(run_locator_id) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid film run ID".to_owned(),
            ));
        }
        let project_path = self.find_project_path(project_id)?;
        let directory = project_path.join("films/runs").join(run_locator_id);
        Ok(FilmRunFiles {
            authoring_snapshot: directory.join("authoring.json"),
            plan: directory.join("plan.json"),
            reference_pack: directory.join("references.json"),
            compiled: directory.join("compiled.json"),
            review_plan: directory.join("review.jsonc"),
            directory,
        })
    }
}
