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
                // The two LEADING kinds are other stories' subjects: the bindings are A1's and the
                // identity text is A3's (sc-24025). What A4 is about is the audio tail.
                .filter(|kind| {
                    !matches!(
                        kind,
                        InsertedTextKind::ReferenceBinding
                            | InsertedTextKind::ContinuityDescription
                    )
                })
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

/// The described-only fixture pair: a pack whose roles are words alone, and the six-shot film that
/// tells the courier story against it with no conditioning of any kind.
///
/// Both are CHECKED IN, and pointed at where they sit. The pack declares no `file` on any role, so
/// there is nothing to copy beside it and nothing for a run to import — which is the whole of what
/// is being proved, and a copied-into-a-tempdir version could hide a path resolution this must not
/// need.
fn described_fixture() -> (PathBuf, PathBuf) {
    (
        Path::new(FIXTURE_DIR).join("plan.described.jsonc"),
        Path::new(FIXTURE_DIR).join("references.described.jsonc"),
    )
}

/// The identity text a shot must carry for `role`, built from the PACK rather than from the
/// compiler: the description as written, whitespace-normalized, with the closing `.` the compiler
/// supplies when the author left one off.
///
/// An independent statement of the rule documented on `ReferenceEntry::description` — only
/// `normalized_description` is shared, because that is the whitespace rule the pack document is
/// itself validated against.
fn expected_identity_sentence(pack: &ReferencePack, role: &str) -> String {
    let entry = pack
        .references
        .iter()
        .find(|entry| entry.role == role)
        .unwrap_or_else(|| panic!("{role} is a role of the described pack"));
    let mut text = normalized_description(&entry.description);
    assert!(
        !text.is_empty(),
        "{role} is described-only, so its description is the whole of it"
    );
    if !text.ends_with(['.', '!', '?']) {
        text.push('.');
    }
    text
}

/// A3 (epic 24017, sc-24025). THE TEXT IDENTITY LOCK, on a film with no references at all.
///
/// A pack role may be DESCRIBED-ONLY — words, no image — and a shot that lists such a role in
/// `continuityRoles` gets the pack's description of it written into the dispatched prompt word for
/// word. The criterion is BYTE-IDENTITY across shots: this exists because nothing conditions these
/// shots on a picture, so the only thing keeping the courier one courier over six cuts is that the
/// compiler says the same thing about her every time. A shot that does not list the role says
/// nothing about it.
///
/// Asserted on BOTH halves: the compiled document, over the whole six-shot film, and the body
/// `POST /api/v1/video/jobs` actually received in a real run — because a document that says the
/// right thing while the route is sent something else is exactly the failure no reviewer
/// downstream could see.
#[tokio::test]
async fn a3_a_described_only_role_is_locked_word_for_word_into_every_shot_that_names_it() {
    let harness = Harness::start(true, Vec::new()).await;
    // The fixture declares the base-partition turbo adapter (sc-24029): turbo is the path a user
    // takes, so the no-reference evaluation film is the turbo film, and the run below needs the
    // adapter it names to be installed.
    harness.install_turbo_loras();
    let (plan_path, pack_path) = described_fixture();

    let (plan, pack) = film_harness::validate(
        Some(&harness.transport),
        &harness.options(plan_path.clone(), pack_path.clone(), None),
    )
    .await
    .expect("the described-only pack and its plan validate against the live catalog");

    // The fixture is what it claims to be, or nothing below is about described-only roles at all.
    assert!(
        pack.references
            .iter()
            .all(sceneworks_core::film_plan::ReferenceEntry::is_described_only),
        "every role of the described pack must declare no file"
    );
    assert!(
        plan.shots.iter().all(|shot| {
            shot.conditioning.reference_roles.is_empty()
                && shot.conditioning.first_frame_role.is_none()
                && shot.conditioning.last_frame_role.is_none()
        }),
        "the described plan must condition on nothing at all"
    );

    // The role under test, and the shots that do and do not name it — read off the PLAN, so the
    // fixture may be re-cut without this test asserting a count it invented.
    const ROLE: &str = "recipient";
    let naming: Vec<String> = plan
        .shots
        .iter()
        .filter(|shot| shot.continuity_roles.iter().any(|role| role == ROLE))
        .map(|shot| shot.id.clone())
        .collect();
    let silent: Vec<String> = plan
        .shots
        .iter()
        .filter(|shot| !shot.continuity_roles.iter().any(|role| role == ROLE))
        .map(|shot| shot.id.clone())
        .collect();
    assert!(
        naming.len() >= 3 && !silent.is_empty(),
        "the fixture must lock {ROLE} in at least three shots and omit it from at least one: \
         naming {naming:?}, silent {silent:?}"
    );
    let sentence = expected_identity_sentence(&pack, ROLE);

    let mut options = planner_options(&harness, "anchoring-described");
    options.reference_pack_path = pack_path.clone();
    let artifacts = film_planner::compile_existing(
        &harness.transport,
        &planner_llm(&harness),
        &options,
        &plan_path,
    )
    .await
    .expect("the described plan compiles");
    assert_eq!(artifacts.compiled.requests.len(), plan.shots.len());

    // THE TURBO REGIME the fixture declares (sc-24029), asserted on the compiled document so the
    // evaluation film cannot silently revert to the 50-step base regime. Every shot resolves to
    // the base partition — nothing here binds a reference — so every shot takes the base
    // partition's adapter and renders at the count that adapter was distilled for.
    for request in &artifacts.compiled.requests {
        assert_eq!(
            request.loras,
            vec!["minimax_h3_turbo_4step_v01".to_owned()],
            "{} dispatches the base-partition turbo adapter",
            request.shot_id
        );
        assert_eq!(
            request.effective_steps,
            Some(4),
            "{} renders at the recipe's own count, not the model default",
            request.shot_id
        );
        assert!(
            request.steps.is_none(),
            "{}: the plan sets no step override, so the recipe governs",
            request.shot_id
        );
    }

    // THE criterion: every shot that names the role carries the pack's own words for it, and the
    // text is the same text in each of them.
    let mut locked: BTreeMap<String, String> = BTreeMap::new();
    for request in &artifacts.compiled.requests {
        // No shot of this plan binds anything, so none carries a binding sentence or a picture.
        assert_eq!(request.model, "minimax_h3", "{}", request.shot_id);
        assert!(
            !request.prompt.contains("<Picture"),
            "{}: nothing is conditioned, so no picture is named: {}",
            request.shot_id,
            request.prompt
        );
        let identity: Vec<&str> = request
            .inserted_text
            .iter()
            .filter(|piece| piece.kind == InsertedTextKind::ContinuityDescription)
            .map(|piece| piece.text.as_str())
            .collect();

        if naming.contains(&request.shot_id) {
            assert_eq!(
                identity.len(),
                1,
                "{}: one identity insertion: {:?}",
                request.shot_id,
                request.inserted_text
            );
            assert!(
                identity[0].contains(&sentence),
                "{}: must carry the pack's own words for {ROLE} — {sentence:?} — got {:?}",
                request.shot_id,
                identity[0]
            );
            // It reaches the PROMPT, and the authored text never contained it: the compiler wrote
            // it, after the refine rewrite.
            assert!(
                request.prompt.contains(identity[0]),
                "{}: the identity text must reach the prompt: {}",
                request.shot_id,
                request.prompt
            );
            // The PLAN's own words for this shot — the refiner's input when there was one, and the
            // authored prompt otherwise. Never `request.prompt`, which is the composed text the
            // insertion has already been written into and would make this assertion vacuous.
            let shot = plan
                .shots
                .iter()
                .find(|shot| shot.id == request.shot_id)
                .expect("every request is a shot of the plan");
            let authored = request
                .authored_prompt
                .as_deref()
                .unwrap_or(shot.prompt.as_str());
            assert!(
                !authored.contains(&sentence),
                "{}: the identity text is the COMPILER's, not the plan's: {authored}",
                request.shot_id
            );
            locked.insert(request.shot_id.clone(), identity[0].to_owned());
        } else {
            assert!(
                !request.prompt.contains(&sentence),
                "{}: does not name {ROLE}, so nothing describes {ROLE}: {}",
                request.shot_id,
                request.prompt
            );
        }
    }

    // BYTE-IDENTICAL, shot to shot. The role's sentence sits inside a larger identity block (these
    // shots lock several roles), so the role's own sentence is compared across every shot that
    // locks it, and each shot must state it exactly once.
    assert_eq!(locked.len(), naming.len());
    for (shot_id, text) in &locked {
        assert_eq!(
            text.matches(sentence.as_str()).count(),
            1,
            "{shot_id}: {ROLE} is described exactly once: {text:?}"
        );
    }

    // THE REAL DISPATCH PATH. One shot that locks the role and one that does not, driven through
    // `film_harness::run` against the in-process router, asserted on the body the video route
    // actually received.
    let with = naming.first().expect("a shot that names the role").clone();
    let without = silent.first().expect("a shot that does not").clone();
    let record = film_harness::run(
        &harness.transport,
        &harness.options(
            plan_path,
            pack_path,
            Some(&[with.as_str(), without.as_str()]),
        ),
    )
    .await
    .expect("the described-only run completes");

    // A described-only role is never imported as an asset: there are no bytes to import.
    assert!(
        record.references.is_empty(),
        "a pack of described-only roles imports nothing: {:?}",
        record.references
    );

    for (shot_id, expected) in [(&with, true), (&without, false)] {
        let shot = record
            .shots
            .iter()
            .find(|shot| &shot.shot_id == shot_id)
            .unwrap_or_else(|| panic!("{shot_id} ran"));
        let job_id = shot
            .attempts
            .last()
            .and_then(|attempt| attempt.job_id.clone())
            .unwrap_or_else(|| panic!("{shot_id} dispatched a job"));
        let (status, job) = request_job(&harness, &job_id).await;
        assert_eq!(status, axum::http::StatusCode::OK, "{job}");
        let prompt = job["payload"]["prompt"]
            .as_str()
            .unwrap_or_else(|| panic!("a dispatched prompt: {}", job["payload"]));
        assert_eq!(
            prompt.contains(&sentence),
            expected,
            "{shot_id}: the route must{} have been sent {ROLE}'s identity text {sentence:?}: \
             {prompt:?}",
            if expected { "" } else { " NOT" }
        );
        if expected {
            // And it is the SAME text the compiled document recorded: the route and the document
            // agree on the exact words, which is the agreement nothing downstream could check.
            let compiled = locked
                .get(shot_id.as_str())
                .unwrap_or_else(|| panic!("{shot_id} was locked at compile time"));
            assert!(
                prompt.contains(compiled.as_str()),
                "{shot_id}: the dispatched prompt must carry the compiled identity text verbatim \
                 — {compiled:?} — got {prompt:?}"
            );
        }
        // Nothing is conditioned, so nothing is supplied — on the payload, not only the document.
        assert!(
            job["payload"]["referenceAssetIds"]
                .as_array()
                .is_none_or(|ids| ids.is_empty()),
            "{shot_id}: a described-only film supplies no images: {}",
            job["payload"]
        );
    }
}

/// A3's refusal half (sc-24025). A described-only role may not be BOUND; a role that is neither an
/// image nor a description is not a role at all; and a locator on a role with no image points into
/// a picture that does not exist.
///
/// All three are proved against the SHIPPED described documents, edited in memory, so they are
/// statements about the document an author actually writes.
#[tokio::test]
async fn a3_a_described_only_role_cannot_be_bound_and_an_empty_role_is_refused() {
    let harness = Harness::start(false, Vec::new()).await;
    let (plan_path, pack_path) = described_fixture();
    let pack: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(
        &std::fs::read_to_string(&pack_path).expect("the described pack"),
    ))
    .expect("the pack parses");
    let plan: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(
        &std::fs::read_to_string(&plan_path).expect("the described plan"),
    ))
    .expect("the plan parses");
    let dir = harness.temp_dir.path();

    // BOUND: SH010 names the described-only courier in `conditioning.referenceRoles`. Refused,
    // naming the shot and the role.
    let mut bound = plan.clone();
    let shot_id = bound["shots"][0]["id"]
        .as_str()
        .expect("a shot id")
        .to_owned();
    bound["shots"][0]["conditioning"] =
        json!({ "mode": "reference_to_video", "referenceRoles": ["courier"] });
    let bound_path = dir.join("plan.described.bound.json");
    std::fs::write(&bound_path, serde_json::to_string_pretty(&bound).unwrap()).unwrap();
    let findings = refusal_findings(
        film_harness::validate(
            Some(&harness.transport),
            &harness.options(bound_path, pack_path.clone(), None),
        )
        .await,
    );
    let refusal = findings
        .iter()
        .find(|message| message.contains("DESCRIBED-ONLY"))
        .unwrap_or_else(|| panic!("binding a described-only role must be refused: {findings:?}"));
    for named in [shot_id.as_str(), "\"courier\"", "continuityRoles"] {
        assert!(
            refusal.contains(named),
            "the refusal must name {named}: {refusal:?}"
        );
    }

    // NEITHER: a role with no file and no description. Refused, naming the role.
    let mut empty = pack.clone();
    empty["references"][0]["description"] = json!("   ");
    let empty_path = dir.join("references.described.empty.json");
    std::fs::write(&empty_path, serde_json::to_string_pretty(&empty).unwrap()).unwrap();
    let findings = refusal_findings(
        film_harness::validate(
            Some(&harness.transport),
            &harness.options(plan_path.clone(), empty_path, None),
        )
        .await,
    );
    assert!(
        findings.iter().any(|message| {
            message.contains("\"courier\"") && message.contains("no `file` and no `description`")
        }),
        "a role that is neither an image nor words must be refused by name: {findings:?}"
    );

    // A LOCATOR on a role with no image: it picks a subject out of a picture there is none of.
    let mut located = pack.clone();
    located["references"][0]["locator"] = json!("the woman on the left");
    let located_path = dir.join("references.described.located.json");
    std::fs::write(
        &located_path,
        serde_json::to_string_pretty(&located).unwrap(),
    )
    .unwrap();
    let findings = refusal_findings(
        film_harness::validate(
            Some(&harness.transport),
            &harness.options(plan_path, located_path, None),
        )
        .await,
    );
    assert!(
        findings
            .iter()
            .any(|message| message.contains("\"courier\"") && message.contains("locator")),
        "a locator on a role with no image must be refused by name: {findings:?}"
    );
}

/// sc-24027. The PLANNER's half of the epic, end to end on BOTH shipped pack shapes, with the
/// model's answer scripted: what this asserts is the request the planner is given, the plan the
/// harness accepts from it and the prompts that plan compiles to — not whether a real 8B model
/// follows the instruction, which is the epic's own real-LLM acceptance test.
///
/// The two shapes are the two the epic ships. A pack with IMAGES plans reference shots: the labels
/// in the dispatched prompt are the COMPILER's, written after the planner answered, and the
/// planner's own prose carries none. A pack of DESCRIBED-ONLY roles approves words rather than
/// pixels, so nothing about a reference may be offered — not the mode, not the cap, not the
/// accelerator for a partition the film never dispatches — and the plan stays on the base
/// checkpoint throughout.
#[tokio::test]
async fn the_planner_plans_both_shipped_pack_shapes_with_audio_and_no_labels_of_its_own() {
    use crate::tests::film_harness::{
        draft_text, full_draft, reference_draft, refine_job_payloads, set_plan_replies,
    };

    // ── The image pack ──────────────────────────────────────────────────────────────────────
    let harness = Harness::start(true, Vec::new()).await;
    set_plan_replies(&harness, vec![draft_text(&reference_draft())]);
    let artifacts = film_planner::generate(
        &harness.transport,
        &planner_llm(&harness),
        &planner_options(&harness, "anchoring-planned-images"),
    )
    .await
    .expect("a reference-binding draft with audio on every shot is a plan");
    assert_eq!(artifacts.repair_rounds, 0);

    let planning = refine_job_payloads(&harness, true)
        .first()
        .map(|payload| payload["prompt"].as_str().unwrap_or_default().to_owned())
        .expect("a planning job was created");
    assert!(
        planning.contains("audio is REQUIRED on every shot")
            && planning.contains("audio NEVER begins with \"Audio:\""),
        "{planning}"
    );
    assert!(
        planning.contains("Never write \"<Picture 1>\", \"<Audio 1>\", \"<Video 1>\""),
        "{planning}"
    );

    for shot in &artifacts.plan.shots {
        assert!(!shot.audio.trim().is_empty(), "{}", shot.id);
        assert!(
            !shot.prompt.contains("<Picture"),
            "{}: the planner writes no labels: {}",
            shot.id,
            shot.prompt
        );
    }
    for request in &artifacts.compiled.requests {
        let shot = artifacts
            .plan
            .shots
            .iter()
            .find(|shot| shot.id == request.shot_id)
            .expect("every request is a shot");
        assert_eq!(request.model, "minimax_h3_ref", "{}", request.shot_id);
        // The labels in the dispatched prompt are the compiler's, and the audio sentence trails it.
        assert!(
            request.prompt.contains("<Picture 1>"),
            "{}: {}",
            request.shot_id,
            request.prompt
        );
        assert!(
            request
                .prompt
                .trim_end()
                .ends_with(&expected_audio_tail(shot)),
            "{}: {}",
            request.shot_id,
            request.prompt
        );
    }

    // ── The described-only pack ─────────────────────────────────────────────────────────────
    let harness = Harness::start(true, Vec::new()).await;
    // Every shipped accelerator installed, so "the reference partition's adapter is not offered"
    // is a statement about the pack rather than about an empty host.
    harness.install_turbo_loras();
    let mut draft = full_draft();
    // The base partition's offered recipe, which is the only one this film could use.
    draft["loras"] = json!(["minimax_h3_turbo_4step_v01"]);
    set_plan_replies(&harness, vec![draft_text(&draft)]);
    let mut options = planner_options(&harness, "anchoring-planned-described");
    options.reference_pack_path = Path::new(FIXTURE_DIR).join("references.described.jsonc");
    let artifacts = film_planner::generate(&harness.transport, &planner_llm(&harness), &options)
        .await
        .expect("a text-to-video draft against a described-only pack is a plan");
    assert_eq!(artifacts.repair_rounds, 0);

    let planning = refine_job_payloads(&harness, true)
        .first()
        .map(|payload| payload["prompt"].as_str().unwrap_or_default().to_owned())
        .expect("a planning job was created");
    assert!(
        planning.contains("Reference conditioning: THIS CHECKPOINT HAS NONE"),
        "a pack of words approves no picture to condition on: {planning}"
    );
    assert!(
        !planning.contains("minimax_h3_ref"),
        "neither the reference partition nor its adapter is offered for a film that never \
         dispatches it: {planning}"
    );
    assert!(
        planning.contains("(no image: name this role in continuityRoles only"),
        "the described-only roles are still offered, for continuity: {planning}"
    );

    assert_eq!(
        artifacts.plan.model.loras,
        vec!["minimax_h3_turbo_4step_v01"]
    );
    for request in &artifacts.compiled.requests {
        let shot = artifacts
            .plan
            .shots
            .iter()
            .find(|shot| shot.id == request.shot_id)
            .expect("every request is a shot");
        assert_eq!(request.model, "minimax_h3", "{}", request.shot_id);
        assert_eq!(request.mode, "text_to_video", "{}", request.shot_id);
        assert!(
            request.reference_roles.is_empty(),
            "{}: nothing is bound",
            request.shot_id
        );
        assert!(
            !request.prompt.contains("<Picture"),
            "{}: nothing is conditioned, so no picture is named: {}",
            request.shot_id,
            request.prompt
        );
        assert!(
            request
                .prompt
                .trim_end()
                .ends_with(&expected_audio_tail(shot)),
            "{}: {}",
            request.shot_id,
            request.prompt
        );
    }
}

/// sc-24029. A per-shot rewrite that ran out of output budget stops mid-sentence, and the job still
/// reports success because its text is non-empty — that is how a film shot shipped with a prompt cut
/// at 3026 chars. The compile must REFUSE such a rewrite rather than bake half a sentence into the
/// dispatched request: the shot falls back to the prompt its author wrote, exactly as `--no-refine`
/// would leave it, and the run says so through a non-fatal finding naming the shot.
///
/// Driven through the shipped `run_fake_refine_job` seam, so the `generation.finishReason` under
/// test is read off the same job snapshot the GPU path reads. No weights, no GPU.
#[tokio::test]
async fn a_rewrite_that_exhausted_its_token_budget_falls_back_to_the_authored_prompt() {
    let harness = Harness::start(true, Vec::new()).await;
    let plan_path = Path::new(FIXTURE_DIR).join("plan.jsonc");
    let plan = sceneworks_core::film_plan::read_plan_file(&plan_path).expect("plan.jsonc parses");

    // Control: the same compile with an ordinary EOS finish keeps the rewrite.
    {
        let mut script = harness.script.lock();
        script.refine_template = Some("rewritten: {prompt}".to_owned());
        script.refine_finish_reason = None;
    }
    let mut options = planner_options(&harness, "refine-stop");
    options.refine_prompts = true;
    let accepted = film_planner::compile_existing(
        &harness.transport,
        &planner_llm(&harness),
        &options,
        &plan_path,
    )
    .await
    .expect("the plan compiles with a completed rewrite");
    assert!(
        accepted.findings.is_empty(),
        "a rewrite that finished on EOS raises nothing: {:?}",
        accepted.findings
    );
    for request in &accepted.compiled.requests {
        assert_eq!(
            request.prompt_source,
            sceneworks_core::film_compile::PromptSource::Refined,
            "{}",
            request.shot_id
        );
        assert!(
            request.prompt.contains("rewritten:"),
            "{}: {}",
            request.shot_id,
            request.prompt
        );
    }

    // The defect: every rewrite stops on `length`.
    harness.script.lock().refine_finish_reason = Some("length".to_owned());
    let mut options = planner_options(&harness, "refine-length");
    options.refine_prompts = true;
    let artifacts = film_planner::compile_existing(
        &harness.transport,
        &planner_llm(&harness),
        &options,
        &plan_path,
    )
    .await
    .expect("a truncated rewrite is not fatal — the authored prompt is a valid request");

    assert_eq!(
        artifacts.compiled.requests.len(),
        plan.shots.len(),
        "the film still compiles in full"
    );
    for request in &artifacts.compiled.requests {
        assert_eq!(
            request.prompt_source,
            sceneworks_core::film_compile::PromptSource::Authored,
            "{} kept a truncated rewrite",
            request.shot_id
        );
        assert!(
            !request.prompt.contains("rewritten:"),
            "{}: the discarded rewrite reached the dispatched prompt: {}",
            request.shot_id,
            request.prompt
        );
        let authored = &plan
            .shots
            .iter()
            .find(|shot| shot.id == request.shot_id)
            .expect("the request names a shot of the plan")
            .prompt;
        assert!(
            request.prompt.contains(authored.as_str()),
            "{}: the author's own words must survive: {}",
            request.shot_id,
            request.prompt
        );
    }

    // And the run SAYS so, per shot, rather than dropping the rewrite silently.
    let flagged: Vec<&str> = artifacts
        .findings
        .iter()
        .map(|finding| finding.shot_id.as_deref().unwrap_or("plan"))
        .collect();
    let expected: Vec<&str> = plan.shots.iter().map(|shot| shot.id.as_str()).collect();
    assert_eq!(flagged, expected, "{:?}", artifacts.findings);
    assert!(
        artifacts
            .findings
            .iter()
            .all(|finding| finding.message.contains("output budget")),
        "{:?}",
        artifacts.findings
    );
}
