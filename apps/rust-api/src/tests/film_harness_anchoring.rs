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
use std::path::Path;

use sceneworks_core::film_compile::{DispatchContext, InsertedTextKind};
use sceneworks_core::film_plan::ReferencePack;

use crate::film_harness;
use crate::film_planner;
use crate::tests::film_harness::{planner_llm, planner_options, Harness, FIXTURE_DIR};

/// The role as a binding sentence names it, mirroring the compiler's own rule. Written out here
/// rather than imported so the assertion below is an independent statement of the expected text.
fn role_phrase(role: &str) -> String {
    role.replace('_', " ")
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

        // ONE insertion, of the reference-binding kind, recorded apart from the authored prompt.
        assert_eq!(
            request.inserted_text.len(),
            1,
            "{}: {:?}",
            request.shot_id,
            request.inserted_text
        );
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
            request.prompt.ends_with(&shot.prompt),
            "{}: the authored prompt must survive the insertion",
            request.shot_id
        );

        // The dispatched list, read out of the job body the route actually receives.
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
            let phrase = role_phrase(role);
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
    assert!(bound_shots >= 6, "the shipped reference plan has six shots");

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
            request.inserted_text.is_empty(),
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
