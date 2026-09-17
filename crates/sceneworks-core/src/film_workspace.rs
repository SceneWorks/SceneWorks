//! Project-owned film authoring documents used by the Video Editor.
//!
//! Execution state deliberately does not live here. A [`FilmRunLocator`] points at the canonical
//! film-harness directory; `run.json` remains the only mutable shot/attempt/timeline record.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::film_compile::CompiledPlan;
use crate::film_plan::{
    PlanLimits, PlanModel, PlanSound, ProductionPlan, ReferencePack, Shot, ShotConditioning,
    PLAN_SCHEMA_VERSION, REFERENCE_PACK_SCHEMA_VERSION,
};
use crate::time::utc_now;

pub const FILM_DRAFT_SCHEMA_VERSION: u32 = 1;
pub const FILM_RUN_LOCATOR_SCHEMA_VERSION: u32 = 1;
pub const DEFAULT_FILM_PLANNING_PROVIDER: &str = "prompt_refiner";
pub const QWEN36_FILM_PLANNER_MODEL_ID: &str = "film_planner_qwen3_6_27b";
pub const QWEN36_FILM_PLANNER_REPO: &str = "Qwen/Qwen3.6-27B";

/// How the film's plan-level adapters and step count are chosen.
///
/// Legacy drafts omit this field. Their existing `model.loras` and `model.advanced.steps` remain
/// authoritative until the user explicitly chooses one of these regimes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FilmRenderRegime {
    RecommendedTurbo,
    Quality,
    Custom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FilmTurboUnavailableReason {
    ModelUnavailable,
    NoInstalledCompatibleAdapter,
    IncompletePartitionCoverage,
    IncompatibleResolution,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FilmRenderChoice {
    pub adapter_ids: Vec<String>,
    pub effective_steps: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FilmRecommendedTurbo {
    pub available: bool,
    pub adapter_ids: Vec<String>,
    pub effective_steps: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unavailable_reason: Option<FilmTurboUnavailableReason>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FilmRenderOptions {
    pub selected_regime: FilmRenderRegime,
    pub recommended_turbo: FilmRecommendedTurbo,
    pub quality: FilmRenderChoice,
    pub effective: FilmRenderChoice,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FilmPlanningSelection {
    pub provider: String,
    /// Catalog identity of an explicitly selected native planner. The built-in provider keeps this
    /// empty and therefore continues to use the small prompt-refiner checkpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    /// Saved OpenAI-compatible connection selected for external planning. This is a non-secret
    /// settings identity; credentials remain in the host secret facility and never enter a film
    /// draft or its exports.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connection_id: Option<String>,
    /// `disabled`, `enabled`, or `auto`. Thinking is an LLM control and never changes the target
    /// video model stored in `productionPlan.model.id`.
    #[serde(default = "default_planning_thinking_mode")]
    pub thinking_mode: String,
    /// Prompt refinement after planning is a separate compile choice, not an implicit side effect
    /// of selecting a more capable planning LLM.
    #[serde(default)]
    pub refine_prompts: bool,
    /// Include approved reference image bytes in external planner requests. The backend also
    /// requires the selected connection to declare image-input support, so a stale or edited
    /// project document cannot enable pixel egress by itself.
    #[serde(default)]
    pub send_reference_pixels: bool,
}

fn default_planning_thinking_mode() -> String {
    "disabled".to_owned()
}

impl Default for FilmPlanningSelection {
    fn default() -> Self {
        Self {
            provider: DEFAULT_FILM_PLANNING_PROVIDER.to_owned(),
            model_id: None,
            connection_id: None,
            thinking_mode: default_planning_thinking_mode(),
            refine_prompts: false,
            send_reference_pixels: false,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FilmBriefDocument {
    #[serde(default)]
    pub synopsis: String,
    #[serde(default)]
    pub style_notes: String,
    #[serde(default = "default_target_seconds")]
    pub target_total_seconds: f64,
    #[serde(default)]
    pub beats: Vec<FilmBeat>,
    #[serde(default)]
    pub dialogue: Vec<FilmDialogueLine>,
}

fn default_target_seconds() -> f64 {
    30.0
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FilmBeat {
    pub id: String,
    pub summary: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FilmDialogueLine {
    pub id: String,
    pub beat_id: String,
    pub speaker: String,
    pub text: String,
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
    pub structured_brief: FilmBriefDocument,
    #[serde(default)]
    pub planning: FilmPlanningSelection,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub render_regime: Option<FilmRenderRegime>,
    pub production_plan: ProductionPlan,
    /// Last explicitly imported or planner-produced compile. Edits intentionally leave this in
    /// place so preflight can explain that it is stale instead of silently replacing refined
    /// prompts. A manual draft may omit it and compile authored prompts at preflight.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compiled_plan: Option<CompiledPlan>,
    pub reference_pack: ReferencePack,
    /// Versioned authoring document for advisory take review, pinned independently into each run.
    pub review_plan: Value,
    pub created_at: String,
    pub updated_at: String,
}

impl FilmDraft {
    /// Missing means a legacy draft. Preserve its authored adapter/step fields exactly instead of
    /// retroactively opting it into a newly introduced default.
    pub fn effective_render_regime(&self) -> FilmRenderRegime {
        self.render_regime.unwrap_or(FilmRenderRegime::Custom)
    }

    pub fn manual_one_shot(project_id: &str, draft_id: &str, title: &str) -> Self {
        let now = utc_now();
        let title = title.trim();
        let title = if title.is_empty() {
            "Untitled film"
        } else {
            title
        };
        let shot = Shot {
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
        };
        let review_plan = default_review_plan(draft_id, std::slice::from_ref(&shot));
        Self {
            schema_version: FILM_DRAFT_SCHEMA_VERSION,
            id: draft_id.to_owned(),
            project_id: project_id.to_owned(),
            revision: 1,
            title: title.to_owned(),
            original_script: String::new(),
            brief: String::new(),
            structured_brief: FilmBriefDocument::default(),
            planning: FilmPlanningSelection::default(),
            render_regime: Some(FilmRenderRegime::RecommendedTurbo),
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
                shots: vec![shot],
            },
            compiled_plan: None,
            reference_pack: ReferencePack {
                schema_version: REFERENCE_PACK_SCHEMA_VERSION,
                id: format!("{draft_id}-references"),
                version: 1,
                description: "No references supplied".to_owned(),
                references: Vec::new(),
                sound: Vec::new(),
            },
            review_plan,
            created_at: now.clone(),
            updated_at: now,
        }
    }

    /// The review document pinned into a run. Early film drafts carried the placeholder
    /// `{schemaVersion, questions}` shape; turn only that known legacy seam into the typed plan so
    /// reopening an old draft does not make review unavailable. Authored modern documents remain
    /// byte-for-byte inputs, including invalid ones that review should report by field.
    pub fn review_plan_for_run(&self) -> Value {
        if self.review_plan.is_null()
            || (self.review_plan.get("questions").is_some()
                && self.review_plan.get("sampling").is_none()
                && self.review_plan.get("shots").is_none())
        {
            default_review_plan(&self.id, &self.production_plan.shots)
        } else {
            self.review_plan.clone()
        }
    }
}

fn default_review_plan(draft_id: &str, shots: &[Shot]) -> Value {
    let questions = shots
        .iter()
        .map(|shot| {
            (
                shot.id.clone(),
                json!({
                    "questions": [{
                        "id": format!("{}_action", shot.id),
                        "topic": "action_completion",
                        "intended": shot.end_state,
                        "ask": "Does the final frame show the authored action completed? Answer yes or no.",
                        "expect": ["yes"],
                        "contradict": ["no"],
                        "frames": "last",
                        "mustObserve": true
                    }]
                }),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    json!({
                "schemaVersion": 1,
                "id": format!("{draft_id}-review"),
                "version": 1,
                "description": "Advisory review questions for generated takes",
                "sampling": { "positions": [0.1, 0.5, 0.9] },
                "limits": {
                    "maxSeconds": 120,
                    "maxFramesPerShot": 3,
                    "maxQuestionsPerShot": 8,
                    "maxAnswerSeconds": 30,
                    "maxNewTokens": 192,
                    "maxMemoryGb": 16.0
                },
                "shots": questions,
                "uncertainBelow": 0.5
    })
}

/// Convert pasted prose or screenplay text into an editable starting document. This is deliberately
/// deterministic and conservative: it identifies scene/action paragraphs and screenplay dialogue,
/// but leaves every extracted field editable before an LLM is asked to plan shots.
pub fn parse_film_script(script: &str) -> FilmBriefDocument {
    let lines = script.lines().map(str::trim).collect::<Vec<_>>();
    let mut beats = Vec::new();
    let mut dialogue = Vec::new();
    let mut paragraph = Vec::<String>::new();
    let mut index = 0_usize;

    let push_beat = |parts: &mut Vec<String>, beats: &mut Vec<FilmBeat>| {
        let summary = parts.join(" ").trim().to_owned();
        parts.clear();
        if summary.is_empty() || beats.len() >= crate::film_planner::MAX_PLANNER_SHOTS {
            return;
        }
        beats.push(FilmBeat {
            id: format!("B{:03}", beats.len() + 1),
            summary,
        });
    };

    while index < lines.len() {
        let line = lines[index];
        if line.is_empty() {
            push_beat(&mut paragraph, &mut beats);
            index += 1;
            continue;
        }
        let next = lines.get(index + 1).copied().unwrap_or_default();
        let screenplay_speaker = is_screenplay_speaker(line) && !next.is_empty();
        if screenplay_speaker {
            push_beat(&mut paragraph, &mut beats);
            if beats.is_empty() {
                beats.push(FilmBeat {
                    id: "B001".to_owned(),
                    summary: "Opening dialogue".to_owned(),
                });
            }
            dialogue.push(FilmDialogueLine {
                id: format!("D{:03}", dialogue.len() + 1),
                beat_id: beats.last().expect("created above").id.clone(),
                speaker: line.trim_matches(['(', ')']).to_owned(),
                text: next.to_owned(),
            });
            index += 2;
            continue;
        }
        if is_scene_heading(line) {
            push_beat(&mut paragraph, &mut beats);
            paragraph.push(line.to_owned());
            push_beat(&mut paragraph, &mut beats);
        } else {
            paragraph.push(line.to_owned());
        }
        index += 1;
    }
    push_beat(&mut paragraph, &mut beats);

    if beats.is_empty() && !script.trim().is_empty() {
        beats.push(FilmBeat {
            id: "B001".to_owned(),
            summary: script.split_whitespace().collect::<Vec<_>>().join(" "),
        });
    }
    let synopsis = beats
        .iter()
        .take(3)
        .map(|beat| beat.summary.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    FilmBriefDocument {
        synopsis,
        style_notes: String::new(),
        target_total_seconds: (beats.len().max(1) as f64 * 5.1667).clamp(5.1667, 120.0),
        beats,
        dialogue,
    }
}

fn is_scene_heading(line: &str) -> bool {
    let upper = line.to_ascii_uppercase();
    upper.starts_with("INT.")
        || upper.starts_with("EXT.")
        || upper.starts_with("INT/")
        || upper.starts_with("EXT/")
}

fn is_screenplay_speaker(line: &str) -> bool {
    let cleaned = line.trim_matches(['(', ')']);
    !cleaned.is_empty()
        && cleaned.len() <= 48
        && cleaned.chars().any(char::is_alphabetic)
        && cleaned
            .chars()
            .all(|character| !character.is_alphabetic() || character.is_uppercase())
        && !is_scene_heading(cleaned)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FilmRunLocator {
    pub schema_version: u32,
    pub id: String,
    pub project_id: String,
    pub draft_id: String,
    pub draft_revision: u32,
    /// Empty on legacy locators means all shots. New locators always pin the effective selection.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub selected_shot_ids: Vec<String>,
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
        assert_eq!(
            draft.render_regime,
            Some(FilmRenderRegime::RecommendedTurbo)
        );
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

    #[test]
    fn legacy_draft_without_a_render_regime_preserves_authored_controls() {
        let mut document =
            serde_json::to_value(FilmDraft::manual_one_shot("project_1", "film_1", "Film"))
                .expect("draft serializes");
        document.as_object_mut().unwrap().remove("renderRegime");
        document["productionPlan"]["model"]["loras"] = json!(["minimax_h3_turbo_8step"]);
        document["productionPlan"]["model"]["advanced"] = json!({ "steps": 7 });
        let draft: FilmDraft = serde_json::from_value(document).expect("legacy draft parses");
        assert_eq!(draft.render_regime, None);
        assert_eq!(draft.effective_render_regime(), FilmRenderRegime::Custom);
        assert_eq!(
            draft.production_plan.model.loras,
            vec!["minimax_h3_turbo_8step"]
        );
        assert_eq!(
            draft
                .production_plan
                .model
                .advanced
                .as_ref()
                .and_then(|advanced| advanced.steps),
            Some(7)
        );
    }

    #[test]
    fn pasted_screenplay_extracts_editable_beats_and_dialogue_without_changing_source() {
        let source = "INT. WORKSHOP - NIGHT\nA courier enters with a parcel.\n\nMARA\nPut it on the bench.\n\nThe lights fail.";
        let parsed = parse_film_script(source);
        assert_eq!(parsed.beats[0].summary, "INT. WORKSHOP - NIGHT");
        assert!(parsed
            .beats
            .iter()
            .any(|beat| beat.summary.contains("courier")));
        assert_eq!(parsed.dialogue[0].speaker, "MARA");
        assert_eq!(parsed.dialogue[0].text, "Put it on the bench.");
        assert!(parsed
            .beats
            .iter()
            .any(|beat| beat.summary.contains("lights fail")));
    }
}
