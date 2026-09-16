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
pub const QWEN36_FILM_PLANNER_MODEL_ID: &str = "film_planner_qwen3_6_27b";
pub const QWEN36_FILM_PLANNER_REPO: &str = "Qwen/Qwen3.6-27B";

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
            structured_brief: FilmBriefDocument::default(),
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
