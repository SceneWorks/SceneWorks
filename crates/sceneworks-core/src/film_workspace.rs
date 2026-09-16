//! Project-owned film authoring documents used by the Video Editor.
//!
//! Execution state deliberately does not live here. A [`FilmRunLocator`] points at the canonical
//! film-harness directory; `run.json` remains the only mutable shot/attempt/timeline record.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::film_plan::{
    PlanLimits, PlanModel, PlanSound, ProductionPlan, ReferencePack, Shot, ShotConditioning,
    PLAN_SCHEMA_VERSION, REFERENCE_PACK_SCHEMA_VERSION,
};
use crate::time::utc_now;

pub const FILM_DRAFT_SCHEMA_VERSION: u32 = 1;
pub const FILM_RUN_LOCATOR_SCHEMA_VERSION: u32 = 1;
pub const DEFAULT_FILM_PLANNING_PROVIDER: &str = "prompt_refiner";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FilmPlanningSelection {
    pub provider: String,
}

impl Default for FilmPlanningSelection {
    fn default() -> Self {
        Self {
            provider: DEFAULT_FILM_PLANNING_PROVIDER.to_owned(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FilmDraft {
    pub schema_version: u32,
    pub id: String,
    pub project_id: String,
    pub revision: u32,
    pub title: String,
    #[serde(default)]
    pub original_script: String,
    #[serde(default)]
    pub brief: String,
    #[serde(default)]
    pub planning: FilmPlanningSelection,
    pub production_plan: ProductionPlan,
    pub reference_pack: ReferencePack,
    /// Reserved, versioned authoring document for the later review slice. Persisting the empty
    /// shape now keeps a draft self-contained without implementing review behavior in S1.
    pub review_plan: Value,
    pub created_at: String,
    pub updated_at: String,
}

impl FilmDraft {
    pub fn manual_one_shot(project_id: &str, draft_id: &str, title: &str) -> Self {
        let now = utc_now();
        let title = title.trim();
        let title = if title.is_empty() {
            "Untitled film"
        } else {
            title
        };
        Self {
            schema_version: FILM_DRAFT_SCHEMA_VERSION,
            id: draft_id.to_owned(),
            project_id: project_id.to_owned(),
            revision: 1,
            title: title.to_owned(),
            original_script: String::new(),
            brief: String::new(),
            planning: FilmPlanningSelection::default(),
            production_plan: ProductionPlan {
                schema_version: PLAN_SCHEMA_VERSION,
                id: draft_id.to_owned(),
                version: 1,
                title: title.to_owned(),
                synopsis: String::new(),
                model: PlanModel {
                    id: "minimax_h3".to_owned(),
                    tier: Some("q4".to_owned()),
                    loras: Vec::new(),
                    fps: Some(24),
                    resolution: Some("576x320".to_owned()),
                    advanced: None,
                },
                limits: PlanLimits {
                    max_run_seconds: 3600,
                    max_shot_seconds: 2700,
                    max_attempts_per_shot: 1,
                    max_memory_gb: 96.0,
                    planner_max_memory_gb: None,
                },
                sound: PlanSound::default(),
                shots: vec![Shot {
                    id: "SH010".to_owned(),
                    beat_id: None,
                    beat: "Opening shot".to_owned(),
                    framing: "wide".to_owned(),
                    prompt: String::new(),
                    negative_prompt: None,
                    target_duration_seconds: 5.1667,
                    resolution: None,
                    start_state: "Opening state".to_owned(),
                    end_state: "Closing state".to_owned(),
                    dialogue: None,
                    sound: None,
                    generated_audio: None,
                    dialogue_clip: None,
                    conditioning: ShotConditioning {
                        mode: "text_to_video".to_owned(),
                        first_frame_role: None,
                        last_frame_role: None,
                        reference_roles: Vec::new(),
                        chain_from_shot_id: None,
                    },
                    seed: None,
                    continuity_roles: Vec::new(),
                    depends_on: Vec::new(),
                }],
            },
            reference_pack: ReferencePack {
                schema_version: REFERENCE_PACK_SCHEMA_VERSION,
                id: format!("{draft_id}-references"),
                version: 1,
                description: "No references supplied".to_owned(),
                references: Vec::new(),
                sound: Vec::new(),
            },
            review_plan: json!({"schemaVersion": 1, "questions": []}),
            created_at: now.clone(),
            updated_at: now,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FilmRunLocator {
    pub schema_version: u32,
    pub id: String,
    pub project_id: String,
    pub draft_id: String,
    pub draft_revision: u32,
    /// Project-relative directory containing pinned inputs and canonical `run.json`.
    pub record_directory: String,
    pub created_at: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::film_plan::{
        validate_plan_against_pack, validate_plan_structure, validate_reference_pack,
    };

    #[test]
    fn manual_draft_defaults_to_prompt_refiner_and_needs_no_reference_roles() {
        let mut draft = FilmDraft::manual_one_shot("project_1", "film_1", "Film");
        draft.production_plan.shots[0].prompt = "A courier enters a workshop.".to_owned();
        assert_eq!(draft.planning.provider, DEFAULT_FILM_PLANNING_PROVIDER);
        assert!(draft.reference_pack.references.is_empty());
        assert!(draft.production_plan.shots[0]
            .conditioning
            .reference_roles
            .is_empty());
        assert!(validate_plan_structure(&draft.production_plan).is_empty());
        assert!(validate_reference_pack(&draft.reference_pack).is_empty());
        assert!(
            validate_plan_against_pack(&draft.production_plan, &draft.reference_pack).is_empty()
        );
    }
}
