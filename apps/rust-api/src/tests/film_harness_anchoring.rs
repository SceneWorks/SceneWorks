//! Epic 24017 — prompt anchoring: the acceptance tests for the text the COMPILER writes into a
//! film-harness prompt.
//!
//! The harness's prompts used to say nothing about the images they were conditioned on. MiniMax-H3
//! labels each supplied reference `<Picture 1>`, `<Picture 2>`, … ahead of the prompt, in supply
//! order, and the model's own prompt guide is explicit that a reference needs a job in the text
//! ("the woman from `<Picture 1>`"); a bare pile of files gives the model a guess. The compiler now
//! writes that job in, after the refine rewrite so no language model can paraphrase a label.
//!
//! A1 below is the epic's first acceptance test, driven against the SHIPPED courier documents
//! rather than a fixture, because the thing being proved is an agreement between two lists — the
//! `<Picture N>` in the prompt and the position of that reference's asset in the dispatched
//! `referenceAssetIds` — and there is nothing downstream that can detect them disagreeing. Later
//! stories of this epic extend this file with their own acceptance tests.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use sceneworks_core::film_compile::{
    normalized_description, DispatchContext, InsertedTextKind, NO_SPEECH_SENTENCE,
};
use sceneworks_core::film_plan::{ProductionPlan, ReferencePack, Shot};
use serde_json::{json, Value};

use crate::film_harness;
use crate::film_planner;
use crate::tests::film_harness::{planner_llm, planner_options, request_job, Harness, FIXTURE_DIR};

/// The role as a binding sentence names it, mirroring the compiler's own rule. Written out here
/// rather than imported so the assertion below is an independent statement of the expected text.
///
/// BOTH separators, like `film_compile::role_phrase`: a pack role legally contains `-`
/// (`film_plan`'s `is_safe_plan_id` admits `[A-Za-z0-9_-]`), and mirroring only `_` would make a
/// hyphenated role panic here with "is never named" instead of failing on the thing under test.
fn role_phrase(role: &str) -> String {
    role.replace(['_', '-'], " ")
}

/// Every reference role in `pack`, mapped to a distinct stand-in asset id — what `ensure_references`
/// produces once the pack's plates have been imported into the project.
fn role_assets(pack: &ReferencePack) -> BTreeMap<String, String> {
    pack.references
        .iter()
        .map(|entry| (entry.role.clone(), format!("asset_{}", entry.role)))
        .collect()
}

/// A1 (epic 24017). Compiling the shipped courier reference plan gives every reference shot one
/// binding sentence per bound role, and the `<Picture N>` each sentence names is the 1-based
/// position that role's asset takes in the request's dispatched `referenceAssetIds`.
///
/// Both numbers come from `sceneworks_core::film_compile::shot_reference_pictures`, which is the
/// point: the engine's labelling is POSITIONAL, so a prompt that binds the courier to `<Picture 2>`
/// while the courier's plate is dispatched first renders a confidently wrong film and no validator,
/// record or reviewer downstream can tell. Asserting the agreement against the shipped plan — six
/// shots, four references each, two of them in a different order — is what keeps the two in step.
///
/// The phase-1 baseline `plan.jsonc` is the control: its shots bind nothing, resolve to the base
/// checkpoint, and must come out with the prompt the plan wrote and no picture talk at all.
#[tokio::test]
async fn a1_every_reference_shot_names_its_pictures_in_the_order_its_assets_are_dispatched() {
    let harness = Harness::start(true, Vec::new()).await;
    let v2_path = Path::new(FIXTURE_DIR).join("plan.v2.jsonc");

    let (plan, pack) = film_harness::validate(
        Some(&harness.transport),
        &harness.options(v2_path.clone(), harness.fixture_pack(), None),
    )
    .await
    .expect("plan.v2.jsonc validates against the live catalog");

    let options = planner_options(&harness, "anchoring-v2");
    let artifacts = film_planner::compile_existing(
        &harness.transport,
        &planner_llm(&harness),
        &options,
        &v2_path,
    )
    .await
    .expect("plan.v2.jsonc compiles");
    assert_eq!(artifacts.compiled.requests.len(), plan.shots.len());

    let assets = role_assets(&pack);
    let mut bound_shots = 0;
    for request in &artifacts.compiled.requests {
        assert_eq!(request.model, "minimax_h3_ref", "{}", request.shot_id);
        assert!(
            !request.reference_roles.is_empty(),
            "{} binds nothing",
            request.shot_id
        );
        bound_shots += 1;

        // The FIRST insertion is the reference-binding kind, recorded apart from the authored
        // prompt. The audio insertions that trail it are asserted by A4 below.
        assert_eq!(
            request.inserted_text[0].kind,
            InsertedTextKind::ReferenceBinding,
            "{}",
            request.shot_id
        );
        let bindings = request.inserted_text[0].text.as_str();

        // It leads the dispatched prompt, and the plan's own words are still behind it.
        assert!(
            request.prompt.starts_with(bindings),
            "{}: the bindings must lead the prompt: {}",
            request.shot_id,
            request.prompt
        );
        let shot = plan
            .shots
            .iter()
            .find(|shot| shot.id == request.shot_id)
            .expect("every request is a shot of the plan");
        assert!(
            request.prompt.contains(&shot.prompt),
            "{}: the authored prompt must survive the insertion",
            request.shot_id
        );

        // The dispatched list, read out of the job body this COMPILED REQUEST builds — not out of a
        // run's own dispatch. Both are the same list: the harness resolves the record's
        // `conditioning_assets` through the very `resolve_conditioning` called below and posts
        // `to_job_body_with` from it. The agreement on the body a real run SENT is asserted
        // end-to-end in `film_harness.rs`; what this loop adds is every shot of the shipped plan.
        let context = DispatchContext {
            project_id: "proj_anchoring",
            run_id: "run_anchoring",
            plan_id: &plan.id,
            plan_version: plan.version,
            attempt: 1,
            tier: plan.model.tier.as_deref(),
            idempotency_key: None,
            pack: &pack,
            role_assets: &assets,
        };
        let body = request
            .to_job_body(&context)
            .unwrap_or_else(|findings| panic!("{}: {findings:?}", request.shot_id));
        let dispatched: Vec<String> = body["referenceAssetIds"]
            .as_array()
            .unwrap_or_else(|| panic!("{} dispatches reference assets: {body}", request.shot_id))
            .iter()
            .map(|id| id.as_str().expect("asset ids are strings").to_owned())
            .collect();
        assert_eq!(
            dispatched.len(),
            request.reference_roles.len(),
            "{}",
            request.shot_id
        );

        // THE acceptance criterion, role by role: the asset at 1-based position N belongs to the
        // role the sentence bound to `<Picture N>`. Pinned through the ORDER of the sentences —
        // role name, then its picture number, then the next role — so a sentence that named the
        // wrong number, or numbers that were right while the assets moved, both go red.
        let mut cursor = 0usize;
        for (index, role) in request.reference_roles.iter().enumerate() {
            let number = index + 1;
            assert_eq!(
                dispatched[index],
                *assets
                    .get(role)
                    .unwrap_or_else(|| panic!("{role} is a pack role")),
                "{}: {role} is dispatched at position {number} of {dispatched:?}",
                request.shot_id
            );
            // Anchored on the full sentence opening, not the bare phrase: a pack description is
            // repeated verbatim into the binding text, so a bare phrase could match inside someone
            // else's description and the scan would walk past the sentence it meant to check.
            let phrase = format!("{} is the ", role_phrase(role));
            let role_at = bindings[cursor..]
                .find(&phrase)
                .map(|at| at + cursor)
                .unwrap_or_else(|| {
                    panic!(
                        "{}: {phrase:?} is never named in {bindings:?}",
                        request.shot_id
                    )
                });
            let picture_at = bindings[role_at..]
                .find(&format!("<Picture {number}>"))
                .map(|at| at + role_at)
                .unwrap_or_else(|| {
                    panic!(
                        "{}: {phrase:?} must be bound to <Picture {number}>, the position its \
                         asset takes in {dispatched:?}: {bindings:?}",
                        request.shot_id
                    )
                });
            cursor = picture_at;
        }

        // No picture the shot does not send is ever named.
        let over = request.reference_roles.len() + 1;
        assert!(
            !bindings.contains(&format!("<Picture {over}>")),
            "{}: names a picture it does not dispatch: {bindings:?}",
            request.shot_id
        );
    }
    // Every compiled request went through the loop above — a shape assertion, not a corpus count:
    // the plan may gain or lose shots without this test caring, but none may skip the criterion.
    assert_eq!(bound_shots, artifacts.compiled.requests.len());

    // Two of the six order `workbench_table` and `workshop_location` the other way round, so the
    // agreement above is exercised against genuinely different orderings of the same files rather
    // than one order repeated six times.
    let orders: Vec<Vec<String>> = artifacts
        .compiled
        .requests
        .iter()
        .map(|request| request.reference_roles.clone())
        .collect();
    assert!(
        orders.iter().any(|order| order != &orders[0]),
        "the shipped plan must exercise more than one reference order: {orders:?}"
    );

    // The phase-1 baseline: no bound references, so no binding sentence and no picture talk.
    let baseline_path = Path::new(FIXTURE_DIR).join("plan.jsonc");
    let baseline_options = planner_options(&harness, "anchoring-baseline");
    let baseline = film_planner::compile_existing(
        &harness.transport,
        &planner_llm(&harness),
        &baseline_options,
        &baseline_path,
    )
    .await
    .expect("plan.jsonc compiles");
    for request in &baseline.compiled.requests {
        assert_eq!(request.model, "minimax_h3", "{}", request.shot_id);
        assert!(
            request
                .inserted_text
                .iter()
                .all(|piece| piece.kind != InsertedTextKind::ReferenceBinding),
            "{}: a base-checkpoint shot sends no pictures, so it names none: {:?}",
            request.shot_id,
            request.inserted_text
        );
        assert!(
            !request.prompt.contains("<Picture"),
            "{}: {}",
            request.shot_id,
            request.prompt
        );
    }
}

/// The sentences one compiled prompt must END with, given the shot it was compiled from.
///
/// Written out here rather than imported from the compiler, so the assertion is an independent
/// statement of the expected text: only `normalized_description` is shared, and only because it is
/// the whitespace rule the plan document itself is validated against.
fn expected_audio_tail(shot: &Shot) -> String {
    let audio = format!("Audio: {}", normalized_description(&shot.audio));
    match shot.dialogue_clip {
        // A PLACED line — spoken by TTS or imported from disk, both put our own voice on the
        // dialogue bus — so H3 is told not to add a second one, after the audio sentence.
        Some(_) => format!("{audio} {NO_SPEECH_SENTENCE}"),
        None => audio,
    }
}

/// A4 (epic 24017), the compiled half. EVERY shot of EVERY shipped plan ends with its own `Audio:`
/// sentence, on both partitions, and only a shot that PLACES a dialogue line also ends with the
/// fixed no-speech sentence.
///
/// Driven against the shipped documents rather than a fixture for the same reason A1 is: the thing
/// being proved is that no shot was missed, and a fixture can only prove it about itself. The
/// courier plans carry both cases — three shots place a line, three do not — so the negative half
/// of the criterion is exercised by the same loop as the positive one.
#[tokio::test]
async fn a4_every_shipped_shot_ends_with_its_audio_sentence_and_only_placed_lines_silence_h3() {
    let harness = Harness::start(true, Vec::new()).await;
    let mut placed = 0usize;
    let mut unplaced = 0usize;

    for (document, expected_model) in [
        ("plan.jsonc", "minimax_h3"),
        ("plan.v2.jsonc", "minimax_h3_ref"),
    ] {
        let path = Path::new(FIXTURE_DIR).join(document);
        let (plan, _) = film_harness::validate(
            Some(&harness.transport),
            &harness.options(path.clone(), harness.fixture_pack(), None),
        )
        .await
        .unwrap_or_else(|error| panic!("{document} validates: {error:?}"));

        let options = planner_options(&harness, &format!("audio-{document}"));
        let artifacts = film_planner::compile_existing(
            &harness.transport,
            &planner_llm(&harness),
            &options,
            &path,
        )
        .await
        .unwrap_or_else(|error| panic!("{document} compiles: {error:?}"));
        assert_eq!(artifacts.compiled.requests.len(), plan.shots.len());

        for request in &artifacts.compiled.requests {
            assert_eq!(request.model, expected_model, "{}", request.shot_id);
            let shot = plan
                .shots
                .iter()
                .find(|shot| shot.id == request.shot_id)
                .expect("every request is a shot of the plan");
            let tail = expected_audio_tail(shot);
            assert!(
                request.prompt.ends_with(&tail),
                "{document} {}: must end with {tail:?}: {}",
                request.shot_id,
                request.prompt
            );
            // Recorded as INSERTED text, separately from the authored prompt, and written after the
            // refine rewrite — the authored text the plan wrote never contains either sentence.
            let kinds: Vec<InsertedTextKind> = request
                .inserted_text
                .iter()
                .map(|piece| piece.kind)
                .filter(|kind| *kind != InsertedTextKind::ReferenceBinding)
                .collect();
            let authored = request
                .authored_prompt
                .as_deref()
                .unwrap_or(shot.prompt.as_str());
            assert!(
                !authored.contains("Audio:") && !authored.contains(NO_SPEECH_SENTENCE),
                "{document} {}: the authored prompt stays the plan's own words: {authored}",
                request.shot_id
            );

            if shot.dialogue_clip.is_some() {
                placed += 1;
                assert_eq!(
                    kinds,
                    vec![InsertedTextKind::Audio, InsertedTextKind::NoSpeech],
                    "{document} {}",
                    request.shot_id
                );
            } else {
                unplaced += 1;
                assert_eq!(
                    kinds,
                    vec![InsertedTextKind::Audio],
                    "{document} {}",
                    request.shot_id
                );
                assert!(
                    !request.prompt.contains(NO_SPEECH_SENTENCE),
                    "{document} {}: no line is placed on this shot, so none is silenced: {}",
                    request.shot_id,
                    request.prompt
                );
            }
        }
    }
    // Shape, not a corpus count: the plans may gain or lose shots, but BOTH branches of the
    // dialogue rule must have been exercised or the negative half proved nothing.
    assert!(
        placed > 0 && unplaced > 0,
        "{placed} placed, {unplaced} not"
    );
}

/// A4, the document half. A shot that says nothing about sound is refused by name, a shot that says
/// it is silent is accepted, and both older plan schema versions are refused BY VERSION.
///
/// Against the SHIPPED plan rather than a fixture, so the refusals are proved on the document an
/// operator actually edits.
#[test]
fn a4_a_shot_without_audio_and_an_older_plan_version_are_both_refused() {
    let text = std::fs::read_to_string(Path::new(FIXTURE_DIR).join("plan.v2.jsonc"))
        .expect("the shipped reference plan");
    let stripped = sceneworks_core::jsonc::strip_jsonc_comments(&text);
    let original: serde_json::Value = serde_json::from_str(&stripped).expect("it parses");
    let shot_id = original["shots"][2]["id"]
        .as_str()
        .expect("a shot id")
        .to_owned();

    // As shipped: clean.
    let plan: ProductionPlan = serde_json::from_value(original.clone()).expect("it decodes");
    assert!(
        sceneworks_core::film_plan::validate_plan_structure(&plan).is_empty(),
        "the shipped plan must validate: {:?}",
        sceneworks_core::film_plan::validate_plan_structure(&plan)
    );

    // Blanked: refused, NAMING the shot.
    let mut blanked = original.clone();
    blanked["shots"][2]["audio"] = serde_json::json!("   ");
    let plan: ProductionPlan = serde_json::from_value(blanked).expect("it decodes");
    let findings = sceneworks_core::film_plan::validate_plan_structure(&plan);
    assert_eq!(findings.len(), 1, "{findings:?}");
    assert_eq!(findings[0].shot_id.as_deref(), Some(shot_id.as_str()));
    assert_eq!(findings[0].field, "audio");

    // Silence, stated: accepted. The validator never reads the prose.
    let mut silent = original.clone();
    silent["shots"][2]["audio"] = serde_json::json!("No audio. Silence.");
    let plan: ProductionPlan = serde_json::from_value(silent).expect("it decodes");
    assert!(
        sceneworks_core::film_plan::validate_plan_structure(&plan).is_empty(),
        "a shot that states its silence is a complete answer"
    );

    // Versions 1 and 2: refused BY VERSION, with the remedy named.
    for stale in [1, 2] {
        let mut older = original.clone();
        older["schemaVersion"] = serde_json::json!(stale);
        let plan: ProductionPlan = serde_json::from_value(older).expect("it decodes");
        let findings = sceneworks_core::film_plan::validate_plan_structure(&plan);
        assert!(
            findings.iter().any(|finding| {
                finding.field == "schemaVersion"
                    && finding.message.contains(&stale.to_string())
                    && finding.message.contains("audio")
            }),
            "version {stale}: {findings:?}"
        );
    }
}

/// `[courier, recipient]` — the phrase each of the two sharing roles picks itself out of the one
/// plate with. Written here, not read off the pack under test, so the assertions below are an
/// independent statement of what must reach the prompt.
const LOCATORS: [&str; 2] = ["the woman on the left", "the man on the right"];

/// The shipped pack with `recipient` re-pointed at the COURIER's plate — one photograph holding two
/// people, which is what sc-24024 exists for. Written beside the copied plates so the relative
/// `file` paths still resolve; `locators` decides whether the two sharing roles declare the
/// locators the pack now requires of them.
fn shared_file_pack(harness: &Harness, locators: bool) -> PathBuf {
    let base = harness.fixture_pack_without_sound();
    let text = std::fs::read_to_string(&base).expect("fixture pack");
    let mut pack: Value = serde_json::from_str(&text).expect("fixture pack parses");
    let entries = pack["references"]
        .as_array_mut()
        .expect("the pack declares references");
    let courier_file = entries
        .iter()
        .find(|entry| entry["role"] == "courier")
        .and_then(|entry| entry["file"].as_str())
        .expect("the shipped pack approves a courier")
        .to_owned();
    for entry in entries.iter_mut() {
        let role = entry["role"].as_str().unwrap_or_default().to_owned();
        if role == "recipient" {
            assert_ne!(
                entry["file"].as_str(),
                Some(courier_file.as_str()),
                "the shipped pack must give the two characters their own plates, or this fixture \
                 proves nothing"
            );
            entry["file"] = json!(courier_file);
        }
        if locators && matches!(role.as_str(), "courier" | "recipient") {
            entry["locator"] = json!(LOCATORS[usize::from(role == "recipient")]);
        }
    }
    let path = base.with_file_name(format!(
        "references.shared{}.json",
        if locators { "" } else { ".unlocated" }
    ));
    std::fs::write(&path, serde_json::to_string_pretty(&pack).unwrap()).unwrap();
    path
}

/// The mixed fixture with SH010 binding the two sharing roles AND a role with a plate of its own.
/// `workshop_location` stays LAST so the shared pair cannot pass by numbering everything 1.
fn shared_file_plan(harness: &Harness) -> PathBuf {
    harness.mixed_partition_plan(|plan| {
        plan["shots"][0]["conditioning"]["referenceRoles"] =
            json!(["courier", "recipient", "workshop_location"]);
    })
}

/// A2 (epic 24017, sc-24024). Two roles on ONE file, each with a locator, dispatch ONE asset id
/// under ONE `<Picture N>`, and both binding sentences carry their own locator.
///
/// Driven through a REAL run — `film_harness::run` against the in-process router — and asserted on
/// the body `POST /api/v1/video/jobs` actually received, because what is at stake is an agreement
/// between the prompt and the payload and nothing downstream can detect them disagreeing.
#[tokio::test]
async fn a2_two_roles_on_one_file_share_one_picture_and_one_dispatched_asset() {
    let harness = Harness::start(true, Vec::new()).await;
    let pack_path = shared_file_pack(&harness, true);
    let plan_path = shared_file_plan(&harness);

    let record = film_harness::run(
        &harness.transport,
        &harness.options(plan_path, pack_path, Some(&["SH010"])),
    )
    .await
    .expect("the shared-file run completes");

    // ONE asset for the shared plate: the two roles resolve to the same imported id, and the role
    // with its own plate keeps its own.
    let asset_of = |role: &str| -> String {
        record
            .references
            .iter()
            .find(|reference| reference.role == role)
            .unwrap_or_else(|| panic!("{role} is imported"))
            .asset_id
            .clone()
    };
    let shared = asset_of("courier");
    assert_eq!(
        asset_of("recipient"),
        shared,
        "two roles on one file must resolve to ONE imported asset"
    );
    let location = asset_of("workshop_location");
    assert_ne!(location, shared, "a role with its own file keeps its own");

    let shot = record
        .shots
        .iter()
        .find(|shot| shot.shot_id == "SH010")
        .expect("SH010 ran");
    let job_id = shot
        .attempts
        .last()
        .and_then(|attempt| attempt.job_id.clone())
        .expect("SH010 dispatched a job");
    let (status, job) = request_job(&harness, &job_id).await;
    assert_eq!(status, axum::http::StatusCode::OK, "{job}");

    // THE dispatched list: three bound roles, TWO supplied images, in picture order.
    assert_eq!(
        job["payload"]["referenceAssetIds"],
        json!([shared, location]),
        "the shared plate is supplied once: {}",
        job["payload"]
    );
    let prompt = job["payload"]["prompt"]
        .as_str()
        .unwrap_or_else(|| panic!("a dispatched prompt: {}", job["payload"]));

    // Both sharing roles are bound to <Picture 1> — the 1-based position of their ONE asset in the
    // payload above — and each says WHICH subject in it it is.
    for (role, locator) in [("courier", LOCATORS[0]), ("recipient", LOCATORS[1])] {
        let sentence = format!("The {role} is {locator} in <Picture 1>.");
        assert!(
            prompt.contains(&sentence),
            "{sentence:?} must lead the dispatched prompt: {prompt:?}"
        );
    }
    // And the role with its own file is <Picture 2>: the shared pair did not consume two numbers,
    // and it keeps the wording a reference with an image to itself has always had.
    assert!(
        prompt.contains("The workshop location is the place shown in <Picture 2>."),
        "a role with its own file keeps the unlocated wording at its own number: {prompt:?}"
    );
    assert!(
        !prompt.contains("<Picture 3>"),
        "three roles over two files name two pictures: {prompt:?}"
    );

    // The record says the same thing the route was sent.
    assert_eq!(
        shot.conditioning_assets.reference_asset_ids,
        vec![shared, location]
    );
}

/// A2's refusal half. The SAME pack with no locator on the roles that share a file is refused, and
/// the finding names both roles and the file — an author looking at that document knows its entries
/// by name, and "two of your references collide" would send them counting array elements.
///
/// And the schema gate (E6): a version 1 document is refused BY VERSION.
#[tokio::test]
async fn a2_a_shared_file_without_locators_and_a_version_1_pack_are_both_refused() {
    let harness = Harness::start(false, Vec::new()).await;
    let plan_path = shared_file_plan(&harness);

    let unlocated = shared_file_pack(&harness, false);
    let findings = refusal_findings(
        film_harness::validate(
            Some(&harness.transport),
            &harness.options(plan_path.clone(), unlocated.clone(), None),
        )
        .await,
    );
    let shared = findings
        .iter()
        .find(|message| message.contains("locator"))
        .unwrap_or_else(|| panic!("a shared file without locators must be refused: {findings:?}"));
    for named in ["\"courier\"", "\"recipient\"", "references/courier.png"] {
        assert!(
            shared.contains(named),
            "the refusal must name {named}: {shared:?}"
        );
    }

    // Version: the same document, otherwise valid, declared as version 1.
    let text = std::fs::read_to_string(shared_file_pack(&harness, true)).expect("shared pack");
    let mut pack: Value = serde_json::from_str(&text).expect("pack parses");
    pack["schemaVersion"] = json!(1);
    let old = unlocated.with_file_name("references.v1.json");
    std::fs::write(&old, serde_json::to_string_pretty(&pack).unwrap()).unwrap();
    let findings = refusal_findings(
        film_harness::validate(
            Some(&harness.transport),
            &harness.options(plan_path, old, None),
        )
        .await,
    );
    assert!(
        findings.iter().any(|message| {
            message.contains("referencePack.schemaVersion") && message.contains("version 1")
        }),
        "a version 1 pack must be refused by version: {findings:?}"
    );
}

/// The refusal messages of a `validate` that must not have succeeded.
fn refusal_findings<T>(result: Result<T, film_harness::HarnessError>) -> Vec<String> {
    match result {
        Ok(_) => panic!("validate accepted a pack it must refuse"),
        Err(film_harness::HarnessError::Validation(findings)) => {
            findings.iter().map(ToString::to_string).collect()
        }
        Err(other) => panic!("expected a validation refusal, got {other}"),
    }
}
