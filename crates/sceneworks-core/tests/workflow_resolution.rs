//! Contract tests for the cross-install resolution report (sc-15950, epic 15945).
//!
//! The load-bearing ones are [`a_catalog_known_but_absent_model_is_installable_not_missing`]
//! (the actionable middle case the story is built around), [`nothing_is_ever_substituted`] (a
//! requirement that does not resolve is NEVER swapped for something that does), and
//! [`an_omitted_collection_withholds_one_click_replay`] (a recipe whose `loras` were dropped is
//! not a recipe with no LoRAs).

use sceneworks_core::project_store::IMPORTED_WORKFLOW_KEY;
use sceneworks_core::workflow_resolution::{
    build_resolution_report, CatalogEntry, InstallAction, LoraMatchedBy, RequirementState,
    StaticCatalogs, StyleField, INERT_STYLE_PRESETS,
};
use sceneworks_core::workflow_share::{
    parse_workflow_share, parse_workflow_share_json, WorkflowShare, OMITTED_LORAS, OMITTED_POSES,
};
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// A parsed envelope built from `overrides` merged onto a minimal valid body, so every test
/// exercises the SAME reduction the reader applies to a stranger's PNG rather than a hand-built
/// struct the parser would have rejected.
fn envelope(overrides: Value) -> WorkflowShare {
    let mut body = json!({
        "sceneworksWorkflow": "image",
        "schemaVersion": 1,
        "producer": { "name": "SceneWorks", "url": "https://example.invalid", "version": "0.8.1" },
        "mode": "text_to_image",
        "model": "krea_2_turbo",
        "prompt": "a lighthouse in heavy fog",
    });
    let object = body.as_object_mut().expect("fixture body is an object");
    for (key, value) in overrides
        .as_object()
        .expect("overrides is an object")
        .clone()
    {
        object.insert(key, value);
    }
    parse_workflow_share(&body).expect("the fixture envelope parses")
}

/// A catalog in which the fixture model is installed and nothing else is known.
fn model_installed() -> StaticCatalogs {
    StaticCatalogs {
        models: vec![CatalogEntry::new("krea_2_turbo")
            .with_name("Krea 2 Turbo")
            .installed()],
        ..StaticCatalogs::default()
    }
}

fn download_action(path: &str) -> InstallAction {
    InstallAction {
        method: "POST".to_owned(),
        path: path.to_owned(),
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

#[test]
fn an_installed_catalog_model_resolves() {
    let report = build_resolution_report(&envelope(json!({})), &model_installed());
    assert_eq!(report.model.state, RequirementState::Resolved);
    assert_eq!(report.model.slug, "krea_2_turbo");
    assert_eq!(report.model.catalog_id.as_deref(), Some("krea_2_turbo"));
    assert_eq!(report.model.name.as_deref(), Some("Krea 2 Turbo"));
    assert!(report.model.install.is_none(), "nothing to install");
    assert!(report.replayable, "a resolved, input-free recipe replays");
}

#[test]
fn a_catalog_known_but_absent_model_is_installable_not_missing() {
    // The middle case the story is built around: a cache-only resolver never auto-downloads, so
    // "in the catalog" and "on disk" are different answers — and the middle one must be ACTIONABLE.
    let catalogs = StaticCatalogs {
        models: vec![CatalogEntry::new("krea_2_turbo")
            .with_name("Krea 2 Turbo")
            .with_install(download_action("/api/v1/models/krea_2_turbo/download"))],
        ..StaticCatalogs::default()
    };
    let report = build_resolution_report(&envelope(json!({})), &catalogs);
    assert_eq!(report.model.state, RequirementState::Installable);
    assert_ne!(
        report.model.state,
        RequirementState::Missing,
        "a model the catalog knows is not missing"
    );
    assert_eq!(
        report
            .model
            .install
            .as_ref()
            .map(|action| action.path.as_str()),
        Some("/api/v1/models/krea_2_turbo/download"),
        "the actionable middle case must name the flow that fetches it"
    );
    assert!(
        !report.replayable,
        "a model that is not on disk cannot be replayed in one click"
    );
}

#[test]
fn a_model_the_catalog_does_not_know_is_missing_and_names_the_slug() {
    let report = build_resolution_report(&envelope(json!({})), &StaticCatalogs::default());
    assert_eq!(report.model.state, RequirementState::Missing);
    assert_eq!(report.model.slug, "krea_2_turbo");
    assert!(report.model.catalog_id.is_none());
    assert!(report.model.install.is_none());
    assert!(
        report.model.detail.contains("krea_2_turbo"),
        "the sentence must name what could not be resolved: {}",
        report.model.detail
    );
}

#[test]
fn a_catalog_known_model_with_no_install_flow_is_missing_rather_than_falsely_actionable() {
    // An entry the catalog knows but cannot fetch (a `downloadable: false` row, an external
    // ComfyUI base that vanished). Reporting it as installable would offer a button that
    // cannot work, which is its own kind of silent substitution.
    let catalogs = StaticCatalogs {
        models: vec![CatalogEntry::new("krea_2_turbo").with_name("Krea 2 Turbo")],
        ..StaticCatalogs::default()
    };
    let report = build_resolution_report(&envelope(json!({})), &catalogs);
    assert_eq!(report.model.state, RequirementState::Missing);
    assert!(report.model.install.is_none());
    assert_eq!(
        report.model.catalog_id.as_deref(),
        Some("krea_2_turbo"),
        "the catalog match is still reported, only the action is not"
    );
}

#[test]
fn nothing_is_ever_substituted() {
    // One model in the catalog, installed, and it is NOT the one the envelope names. The report
    // must not reach for it — no default model, no nearest match, no quiet success.
    let catalogs = StaticCatalogs {
        models: vec![CatalogEntry::new("z_image_turbo")
            .with_name("Z-Image Turbo")
            .installed()],
        loras: vec![CatalogEntry::new("film_grain")
            .with_name("Film Grain")
            .installed()],
        styles: vec![CatalogEntry::new("ghibli-style")],
        ..StaticCatalogs::default()
    };
    let share = envelope(json!({
        "loras": [{ "name": "Aurora Portrait", "weight": 0.8 }],
        "styleId": "akira-style",
    }));
    let report = build_resolution_report(&share, &catalogs);
    assert_eq!(report.model.state, RequirementState::Missing);
    assert_eq!(report.model.slug, "krea_2_turbo");
    assert!(report.model.catalog_id.is_none());
    assert_eq!(report.loras.len(), 1);
    assert_eq!(report.loras[0].state, RequirementState::Missing);
    assert!(report.loras[0].catalog_id.is_none());
    assert!(report.loras[0].matched_by.is_none());
    assert_eq!(report.styles.len(), 1);
    assert_eq!(report.styles[0].state, RequirementState::Missing);
    assert!(!report.replayable);
}

// ---------------------------------------------------------------------------
// LoRAs
// ---------------------------------------------------------------------------

#[test]
fn a_lora_matches_by_hugging_face_repo_id() {
    let catalogs = StaticCatalogs {
        loras: vec![CatalogEntry::new("film_grain")
            .with_name("Film Grain (local rename)")
            .with_repo("acme/film-grain")
            .installed()],
        ..model_installed()
    };
    let share = envelope(json!({
        "loras": [{ "name": "Whatever the sender called it", "repo": "acme/film-grain", "weight": 0.6 }],
    }));
    let report = build_resolution_report(&share, &catalogs);
    assert_eq!(report.loras.len(), 1);
    assert_eq!(report.loras[0].state, RequirementState::Resolved);
    assert_eq!(report.loras[0].matched_by, Some(LoraMatchedBy::Repo));
    assert_eq!(report.loras[0].catalog_id.as_deref(), Some("film_grain"));
    assert_eq!(report.loras[0].weight, Some(0.6));
    assert!(report.replayable);
}

#[test]
fn a_lora_matches_by_name_when_the_envelope_carried_no_repo() {
    let catalogs = StaticCatalogs {
        loras: vec![CatalogEntry::new("film_grain")
            .with_name("Film Grain")
            .installed()],
        ..model_installed()
    };
    let share = envelope(json!({ "loras": [{ "name": "Film Grain" }] }));
    let report = build_resolution_report(&share, &catalogs);
    assert_eq!(report.loras[0].state, RequirementState::Resolved);
    assert_eq!(report.loras[0].matched_by, Some(LoraMatchedBy::Name));
    assert_eq!(report.loras[0].catalog_id.as_deref(), Some("film_grain"));
}

#[test]
fn a_name_match_ignores_case_and_separators() {
    // "Film Grain" / "film-grain" / "film_grain" are one name — a display name typed on one
    // install and a catalog id slugged on another are the same LoRA, and refusing to see that
    // would report a resolvable LoRA as user-trained.
    let catalogs = StaticCatalogs {
        loras: vec![CatalogEntry::new("film_grain")
            .with_name("Film  Grain")
            .installed()],
        ..model_installed()
    };
    for sent in ["film-grain", "FILM GRAIN", "Film_Grain", "film grain"] {
        let share = envelope(json!({ "loras": [{ "name": sent }] }));
        let report = build_resolution_report(&share, &catalogs);
        assert_eq!(
            report.loras[0].state,
            RequirementState::Resolved,
            "`{sent}` must match the catalog name"
        );
    }
}

#[test]
fn a_name_match_also_accepts_the_catalog_id() {
    let catalogs = StaticCatalogs {
        loras: vec![CatalogEntry::new("film_grain").installed()],
        ..model_installed()
    };
    let share = envelope(json!({ "loras": [{ "name": "film_grain" }] }));
    let report = build_resolution_report(&share, &catalogs);
    assert_eq!(report.loras[0].state, RequirementState::Resolved);
    assert_eq!(report.loras[0].matched_by, Some(LoraMatchedBy::Name));
}

#[test]
fn a_repo_match_wins_over_a_name_match() {
    // A repo id is unambiguous and a display name is not, so the repo is tried first.
    let catalogs = StaticCatalogs {
        loras: vec![
            CatalogEntry::new("by_name")
                .with_name("Film Grain")
                .installed(),
            CatalogEntry::new("by_repo")
                .with_name("Something else entirely")
                .with_repo("acme/film-grain")
                .installed(),
        ],
        ..model_installed()
    };
    let share = envelope(json!({
        "loras": [{ "name": "Film Grain", "repo": "acme/film-grain" }],
    }));
    let report = build_resolution_report(&share, &catalogs);
    assert_eq!(report.loras[0].catalog_id.as_deref(), Some("by_repo"));
    assert_eq!(report.loras[0].matched_by, Some(LoraMatchedBy::Repo));
}

#[test]
fn an_unresolvable_lora_is_listed_rather_than_dropped() {
    // A user-trained LoRA is the EXPECTED unresolvable case. Dropping the entry would make the
    // report claim a recipe with no LoRAs, which is the silent-loss failure this epic exists for.
    let share = envelope(json!({
        "loras": [
            { "name": "Aurora Portrait v3", "weight": 0.9 },
            { "repo": "someone/private-style" },
        ],
    }));
    let report = build_resolution_report(&share, &model_installed());
    assert_eq!(
        report.loras.len(),
        2,
        "both entries survive into the report"
    );
    assert!(report
        .loras
        .iter()
        .all(|lora| lora.state == RequirementState::Missing));
    assert!(
        report.loras[0].detail.contains("Aurora Portrait v3"),
        "the sentence names the LoRA: {}",
        report.loras[0].detail
    );
    assert!(
        report.loras[1].detail.contains("someone/private-style"),
        "a repo-only entry is named by its repo: {}",
        report.loras[1].detail
    );
    assert!(!report.replayable);
}

#[test]
fn a_catalog_known_but_uninstalled_lora_is_installable() {
    let catalogs = StaticCatalogs {
        loras: vec![CatalogEntry::new("film_grain")
            .with_name("Film Grain")
            .with_repo("acme/film-grain")
            .with_install(download_action("/api/v1/loras/film_grain/download"))],
        ..model_installed()
    };
    let share = envelope(json!({ "loras": [{ "repo": "acme/film-grain" }] }));
    let report = build_resolution_report(&share, &catalogs);
    assert_eq!(report.loras[0].state, RequirementState::Installable);
    assert_eq!(
        report.loras[0]
            .install
            .as_ref()
            .map(|action| action.path.as_str()),
        Some("/api/v1/loras/film_grain/download")
    );
}

// ---------------------------------------------------------------------------
// Style
// ---------------------------------------------------------------------------

#[test]
fn a_style_id_resolves_against_the_style_catalog() {
    let catalogs = StaticCatalogs {
        styles: vec![CatalogEntry::new("ghibli-style").with_name("Ghibli Style")],
        ..model_installed()
    };
    let report =
        build_resolution_report(&envelope(json!({ "styleId": "ghibli-style" })), &catalogs);
    assert_eq!(report.styles.len(), 1);
    assert_eq!(report.styles[0].field, StyleField::StyleId);
    assert_eq!(report.styles[0].state, RequirementState::Resolved);
    assert_eq!(report.styles[0].name.as_deref(), Some("Ghibli Style"));
    assert!(report.replayable);

    let unknown =
        build_resolution_report(&envelope(json!({ "styleId": "not-a-style" })), &catalogs);
    assert_eq!(unknown.styles[0].state, RequirementState::Missing);
    assert!(
        unknown.styles[0].install.is_none(),
        "a style is not something this install can fetch"
    );
    assert!(!unknown.replayable);
}

#[test]
fn a_style_preset_resolves_against_the_merged_recipe_preset_catalog() {
    let catalogs = StaticCatalogs {
        recipe_presets: vec![CatalogEntry::new("preset_noir").with_name("Noir")],
        ..model_installed()
    };
    let report = build_resolution_report(
        &envelope(json!({ "stylePreset": "preset_noir" })),
        &catalogs,
    );
    assert_eq!(report.styles.len(), 1);
    assert_eq!(report.styles[0].field, StyleField::StylePreset);
    assert_eq!(report.styles[0].state, RequirementState::Resolved);
    assert!(report.replayable);

    let unknown = build_resolution_report(
        &envelope(json!({ "stylePreset": "preset_from_their_install" })),
        &catalogs,
    );
    assert_eq!(unknown.styles[0].state, RequirementState::Missing);
    assert!(!unknown.replayable);
}

#[test]
fn the_inert_style_preset_defaults_are_not_requirements() {
    // `stylePreset` is only a preset id when a recipe preset actually ran; otherwise it is the
    // wire default that `ImageRequest::from_payload` fills in, which EVERY generated envelope
    // carries. Treating that as an unresolved preset would mark every shared image unreplayable.
    for inert in INERT_STYLE_PRESETS {
        let report = build_resolution_report(
            &envelope(json!({ "stylePreset": inert })),
            &model_installed(),
        );
        assert!(
            report.styles.is_empty(),
            "`{inert}` names no preset, so it is not a requirement"
        );
        assert!(report.replayable, "`{inert}` must not block replay");
    }
}

#[test]
fn both_style_axes_are_reported_with_the_live_one_first() {
    let catalogs = StaticCatalogs {
        styles: vec![CatalogEntry::new("ghibli-style")],
        recipe_presets: vec![CatalogEntry::new("preset_noir")],
        ..model_installed()
    };
    let share = envelope(json!({ "styleId": "ghibli-style", "stylePreset": "preset_noir" }));
    let report = build_resolution_report(&share, &catalogs);
    let fields: Vec<StyleField> = report.styles.iter().map(|style| style.field).collect();
    assert_eq!(fields, vec![StyleField::StyleId, StyleField::StylePreset]);
}

// ---------------------------------------------------------------------------
// Input images
// ---------------------------------------------------------------------------

#[test]
fn input_images_are_user_supplied_and_named_by_shape() {
    let share = envelope(json!({
        "mode": "edit_image",
        "inputs": [
            { "kind": "source", "count": 1 },
            { "kind": "reference", "count": 2 },
            { "kind": "mask", "count": 1 },
            { "kind": "control", "count": 1, "controlMode": "canny" },
        ],
    }));
    let report = build_resolution_report(&share, &model_installed());
    let details: Vec<&str> = report
        .inputs
        .iter()
        .map(|input| input.detail.as_str())
        .collect();
    assert_eq!(
        details,
        vec![
            "Needs a source image.",
            "Needs 2 reference images.",
            "Needs a mask.",
            "Needs a canny control image.",
        ]
    );
    assert!(report
        .inputs
        .iter()
        .all(|input| input.state == RequirementState::UserSupplied));
    assert!(
        !report.replayable,
        "an input image the user must pick is not a one-click replay"
    );
}

#[test]
fn a_control_input_without_a_mode_still_reads_as_a_control_image() {
    let share = envelope(json!({ "inputs": [{ "kind": "control", "count": 1 }] }));
    let report = build_resolution_report(&share, &model_installed());
    assert_eq!(report.inputs[0].detail, "Needs a control image.");
    assert!(report.inputs[0].control_mode.is_none());
}

// ---------------------------------------------------------------------------
// The "name what does not resolve" roll-up
// ---------------------------------------------------------------------------

#[test]
fn unresolved_collects_every_sentence_a_reader_has_to_show() {
    let share = envelope(json!({
        "loras": [{ "name": "Aurora Portrait" }],
        "styleId": "not-a-style",
        "inputs": [{ "kind": "source", "count": 1 }],
        "omitted": [OMITTED_POSES],
    }));
    // Model resolved, everything else not: the roll-up must carry four sentences and not the
    // model's.
    let report = build_resolution_report(&share, &model_installed());
    let unresolved = report.unresolved();
    assert_eq!(unresolved.len(), 4, "got {unresolved:?}");
    assert!(unresolved
        .iter()
        .any(|line| line.contains("Aurora Portrait")));
    assert!(unresolved.iter().any(|line| line.contains("not-a-style")));
    assert!(unresolved.contains(&"Needs a source image."));
    assert!(unresolved.iter().any(|line| line.contains("pose-free")));
    assert!(!unresolved
        .iter()
        .any(|line| line.contains("krea_2_turbo") || line.contains("Krea 2 Turbo")));
}

#[test]
fn a_fully_resolved_report_has_nothing_to_name() {
    assert!(
        build_resolution_report(&envelope(json!({})), &model_installed())
            .unresolved()
            .is_empty()
    );
}

// ---------------------------------------------------------------------------
// Omitted collections
// ---------------------------------------------------------------------------

#[test]
fn an_omitted_collection_withholds_one_click_replay() {
    // A recipe whose `loras` were omitted is NOT a recipe with no LoRAs — the envelope says so
    // and the report has to pass that on, or sc-15952 offers a replay that silently drops five
    // LoRAs.
    let share = envelope(json!({ "omitted": [OMITTED_LORAS, OMITTED_POSES] }));
    let report = build_resolution_report(&share, &model_installed());
    assert!(report.loras.is_empty(), "there were none to classify");
    let fields: Vec<&str> = report
        .omitted
        .iter()
        .map(|entry| entry.field.as_str())
        .collect();
    // The envelope's `omitted` is a sorted SET (`reduce_omitted`), so the report preserves that
    // order rather than inventing one.
    assert_eq!(fields, vec![OMITTED_POSES, OMITTED_LORAS]);
    assert!(
        report.omitted.iter().all(|entry| !entry.detail.is_empty()),
        "each omission carries a sentence"
    );
    assert!(
        report
            .omitted
            .iter()
            .any(|entry| entry.field == OMITTED_LORAS && entry.detail.contains("NOT")),
        "the LoRA omission has to say what the absence does not mean"
    );
    assert!(
        !report.replayable,
        "an omitted collection withholds one-click replay"
    );
}

#[test]
fn an_envelope_with_nothing_omitted_reports_no_omissions() {
    let report = build_resolution_report(&envelope(json!({})), &model_installed());
    assert!(report.omitted.is_empty());
    assert!(report.replayable);
}

// ---------------------------------------------------------------------------
// The whole report
// ---------------------------------------------------------------------------

#[test]
fn the_report_reads_an_imported_assets_stored_envelope() {
    // sc-15949 stores the envelope at `extra.importedWorkflow`. The report must work on THAT
    // value, not only on one freshly read out of a PNG, because sc-15951/sc-15952 read the
    // asset row.
    let share = envelope(json!({ "loras": [{ "repo": "acme/film-grain" }] }));
    let stored =
        json!({ "extra": { IMPORTED_WORKFLOW_KEY: serde_json::to_value(&share).unwrap() } });
    let round_tripped = parse_workflow_share(&stored["extra"][IMPORTED_WORKFLOW_KEY])
        .expect("the stored envelope re-parses");
    assert_eq!(round_tripped, share);

    let catalogs = StaticCatalogs {
        loras: vec![CatalogEntry::new("film_grain")
            .with_repo("acme/film-grain")
            .installed()],
        ..model_installed()
    };
    let report = build_resolution_report(&round_tripped, &catalogs);
    assert_eq!(report.loras[0].state, RequirementState::Resolved);
    assert!(report.replayable);
}

#[test]
fn the_report_serializes_as_camel_case_json_for_the_web() {
    let catalogs = StaticCatalogs {
        models: vec![CatalogEntry::new("krea_2_turbo")
            .with_install(download_action("/api/v1/models/krea_2_turbo/download"))],
        ..StaticCatalogs::default()
    };
    let share = envelope(json!({ "inputs": [{ "kind": "source", "count": 1 }] }));
    let report = build_resolution_report(&share, &catalogs);
    let value = serde_json::to_value(&report).expect("the report serializes");
    assert_eq!(value["model"]["state"], json!("installable"));
    assert_eq!(value["model"]["slug"], json!("krea_2_turbo"));
    assert_eq!(value["model"]["install"]["method"], json!("POST"));
    assert_eq!(value["inputs"][0]["state"], json!("userSupplied"));
    assert_eq!(value["inputs"][0]["kind"], json!("source"));
    assert_eq!(value["replayable"], json!(false));
    // Every declared key is present even when empty, so a reader never has to distinguish
    // "absent" from "none".
    for key in [
        "model",
        "loras",
        "styles",
        "inputs",
        "omitted",
        "replayable",
    ] {
        assert!(value.get(key).is_some(), "`{key}` must be serialized");
    }
}

#[test]
fn a_report_round_trips_through_json() {
    let report = build_resolution_report(
        &envelope(json!({ "loras": [{ "name": "Aurora" }], "styleId": "ghibli-style" })),
        &model_installed(),
    );
    let text = serde_json::to_string(&report).expect("serializes");
    let parsed: sceneworks_core::workflow_resolution::ResolutionReport =
        serde_json::from_str(&text).expect("deserializes");
    assert_eq!(parsed, report);
}

#[test]
fn every_omitted_field_the_envelope_can_carry_has_a_sentence() {
    // The closed vocabulary is `workflow_share::OMITTED_FIELDS`; a member with no sentence here
    // would reach the user as a bare field name.
    for field in sceneworks_core::workflow_share::OMITTED_FIELDS {
        let share = envelope(json!({ "omitted": [field] }));
        let report = build_resolution_report(&share, &model_installed());
        assert_eq!(report.omitted.len(), 1, "`{field}` survives the parse");
        let detail = &report.omitted[0].detail;
        assert!(
            !detail.is_empty() && !detail.contains("was declared but not recorded"),
            "`{field}` needs its own sentence, got: {detail}"
        );
    }
}

#[test]
fn a_share_read_out_of_json_text_is_reportable_end_to_end() {
    // The shape sc-15951 hands us: a chunk's text straight off a dropped file.
    let share = parse_workflow_share_json(
        r#"{
            "sceneworksWorkflow": "image",
            "schemaVersion": 1,
            "producer": { "name": "SceneWorks", "url": "https://example.invalid", "version": "0.8.1" },
            "mode": "text_to_image",
            "model": "z_image_turbo",
            "prompt": "a fox in the snow",
            "loras": [{ "name": "Aurora Portrait", "weight": 0.7 }]
        }"#,
    )
    .expect("the envelope parses");
    let report = build_resolution_report(&share, &StaticCatalogs::default());
    assert_eq!(report.model.slug, "z_image_turbo");
    assert_eq!(report.model.state, RequirementState::Missing);
    assert_eq!(report.loras.len(), 1);
    assert_eq!(report.loras[0].name.as_deref(), Some("Aurora Portrait"));
}
