use super::*;
use crate::contracts::JsonObject;
use crate::video_request::test_fixtures::builtin_video_models;
use serde_json::{json, Value};

fn payload(value: Value) -> JsonObject {
    value.as_object().cloned().unwrap()
}

/// The step-menu refusal for `steps` against `entry`, asserting there IS one (sc-19502).
/// Names a fixed model so the message assertions can look for it.
fn steps_menu_message(entry: &JsonObject, steps: u32) -> String {
    steps_limit_error("some_distilled", steps, entry)
        .unwrap_or_else(|| panic!("{steps} steps must be refused by the declared menu"))
}

/// Each shipped video model's declared `limits.hardMinSteps`, TRANSCRIBED from the manifest —
/// deliberately independent of [`hard_min_steps`], the function under test, exactly as
/// [`DURATION_FLOORS`] is of [`hard_min_duration`].
///
/// `None` is the load-bearing half, and it is a SURVEY RESULT rather than a default: sc-19426
/// read every video engine's schedule construction at the then-pinned inference rev `014134e3`
/// (sc-19721 moved the pin to `75d66db5`; the survey's finding below is unchanged, and the rev
/// is kept because it dates the READING)
/// looking for a step count below which it errors or builds an unusable schedule. What it found:
///
/// * **Every** engine shares gen-core's `steps == 0` rejection (`generator.rs`, "steps must be
///   >= 1"), so 1 is the universal floor and declaring `hardMinSteps: 1` would be a no-op.
/// * Wan / Bernini / SCAIL-2 / Krea-Realtime / SVD all build a structurally valid grid at a
///   single step — SVD's Karras ramp even branches on `n == 1` deliberately — so a floor there
///   would invent a constraint their engines do not have.
/// * MiniMax-H3 is the one family whose scheduler REFUSES below 2.
///
/// ✅ **LTX's floor-shaped hole is now filled — by the third column, not this one.** sc-19426
/// recorded LTX as a `None` it could not express: `candle-gen-ltx` runs a baked 8-step distilled
/// schedule and rejects any `steps` that is not exactly `NATIVE_STEPS` (8), so `hardMinSteps: 8`
/// would have said "at least 8" and wrongly admitted 30. sc-19502 added `limits.steps`, the
/// membership-tested menu, which states it exactly. LTX therefore stays `None` on the FLOOR axis
/// — correctly, it has no floor — and carries `Some(&[8])` on the MENU axis.
///
/// The lane divergence sc-19426 also recorded (mlx never reading `req.steps` and silently
/// rendering the baked schedule anyway) was fixed in inference rather than modelled here: both
/// lanes now derive the count from a byte-identical `STAGE1_SIGMAS` table and share gen-core's
/// `Capabilities::supported_steps` floor. That is what makes ONE lane-agnostic declaration
/// honest; `both_ltx_lanes_agree_on_one_legal_step_count` is the guard.
///
/// The menu column is a survey result on the same footing as the floor column: `None` means the
/// engine genuinely accepts a range of counts, and every model but LTX does.
/// One row of [`STEP_FLOORS`]: the model id, its `limits.hardMinSteps` FLOOR, and its
/// `limits.steps` MENU. Both `None` for a model whose engine samples at any count.
///
/// Named rather than written inline because `clippy::type_complexity` refuses the bare
/// three-column tuple — and the alias is the better read anyway: the two `Option`s are
/// different axes, not a pair.
type StepConstraints = (&'static str, Option<u32>, Option<&'static [u32]>);

const STEP_FLOORS: &[StepConstraints] = &[
    // Distilled: the σ waypoints are baked into training, so 8 is the only renderable count
    // (sc-19502). A menu, not a floor — 30 is as unrenderable as 1.
    ("ltx_2_3", None, Some(&[8])),
    ("ltx_2_3_eros", None, Some(&[8])),
    // LTX-2.5 ships both the 8-step CFG-free distilled recipe and the guided 30-step dev recipe.
    ("ltx_2_5", None, Some(&[8, 30])),
    ("svd", None, None),
    ("wan_2_2", None, None),
    ("wan_2_2_t2v_14b", None, None),
    ("wan_2_2_i2v_14b", None, None),
    ("wan_2_2_vace_fun_14b", None, None),
    ("bernini", None, None),
    ("scail2_14b", None, None),
    ("krea_realtime_14b", None, None),
    // `steps` counts MODEL EVALUATIONS (NFE): the engine appends the terminal sigma itself
    // (`linspace(1, 0, steps + 1)`), so 1 renders — as noise, a single Euler jump from pure
    // randomness. The floor of 2 is the corrected sc-18726 product judgement (see
    // `hard_min_steps`), not an engine limit. A FLOOR, not a menu: everything at or above 2
    // renders fine.
    ("minimax_h3", Some(2), None),
    ("minimax_h3_ref", Some(2), None),
];

/// sc-19426 / sc-19502 — the step twin of
/// `shipped_manifest_duration_floors_admit_everything_they_advertise`, and the test that makes
/// `limits.hardMinSteps` **and** `limits.steps` live constraints on the REAL manifest bytes
/// rather than schema entries nothing exercises.
///
/// Same four-part skeleton on each axis: core reads exactly what is declared (including
/// "nothing"), the model's own shipped default is admitted, the constraint actually refuses
/// something, and the refusal names the model. The menu axis adds a fifth part the floor axis
/// structurally cannot have — it must refuse an OVER-menu count too, which is the precise gap
/// that made `hardMinSteps` unable to express LTX.
#[test]
fn shipped_manifest_step_floors_admit_everything_they_advertise() {
    let models = builtin_video_models();
    assert_eq!(
        models.len(),
        STEP_FLOORS.len(),
        "a video model was added/removed; decide whether its engine REFUSES below a step count \
             (a floor), can render only specific counts (a menu), or samples happily at any count, \
             and add it to STEP_FLOORS — an undeclared constraint is silently NO constraint"
    );

    for (id, want_floor, want_menu) in STEP_FLOORS {
        let entry = models
            .iter()
            .find(|m| m.get("id").and_then(Value::as_str) == Some(*id))
            .unwrap_or_else(|| panic!("{id} present in builtin.models.jsonc"))
            .as_object()
            .expect("model entry object");

        // 1. Core reads exactly what the manifest declares — including "nothing". This is the
        //    half that catches an authored `2.0`, which JSON Schema's `integer` admits and
        //    `as_u64` does not, i.e. a declaration with no reader.
        assert_eq!(hard_min_steps(entry), *want_floor, "{id}: hard min steps");
        assert_eq!(
            allowed_steps(entry).as_deref(),
            *want_menu,
            "{id}: step menu"
        );

        let declared_default = entry
            .get("defaults")
            .and_then(Value::as_object)
            .and_then(|defaults| defaults.get("steps"))
            .and_then(Value::as_u64)
            .map(|steps| steps as u32);

        // The MENU axis (sc-19502). Checked before the floor because a model carrying a menu is
        // governed entirely by it — the exact statement wins over the floor's `<`.
        if let Some(menu) = want_menu {
            // 2a. FIXED POINT: the model's own `defaults.steps` must be ON the menu, or the app
            //     would refuse the request it composed itself (the sc-12400 class).
            let default_steps = declared_default.unwrap_or_else(|| {
                panic!("{id} declares a step menu, so it must declare a default step count")
            });
            assert!(
                menu.contains(&default_steps),
                "{id} defaults to {default_steps} steps, which is off its own menu {menu:?}"
            );
            for steps in *menu {
                assert_eq!(
                    steps_limit_error(id, *steps, entry),
                    None,
                    "{id}: the advertised {steps}-step schedule must be admitted"
                );
            }

            // 2b. MUTATION CHECK, on BOTH sides. The over-menu case is the whole reason this key
            //     exists: `hardMinSteps: 8` would have admitted 30, which the engine refuses.
            for off in [1u32, menu.iter().max().expect("non-empty menu") + 22] {
                let message = steps_limit_error(id, off, entry).unwrap_or_else(|| {
                    panic!("{id}: {off} steps is off the menu {menu:?} and must be rejected")
                });
                assert!(message.contains(id), "{id}: names the model: {message}");
                assert!(
                    message.contains(&off.to_string()),
                    "{id}: names the request: {message}"
                );
                assert!(
                    menu.iter().all(|s| message.contains(&s.to_string())),
                    "{id}: names the legal value(s): {message}"
                );
            }
            continue;
        }

        let Some(floor) = want_floor else {
            // A model with neither menu nor floor admits everything, including the degenerate
            // single step.
            assert_eq!(steps_limit_error(id, 1, entry), None, "{id}: no floor");
            continue;
        };

        // 2. FIXED POINT: the model's own shipped `defaults.steps` — the value the studio puts
        //    in the form and therefore the likeliest number to arrive — must be admitted, or
        //    the gate would refuse a request the app itself composed (the sc-12400 class).
        let default_steps = declared_default.unwrap_or_else(|| {
            panic!("{id} declares a step floor, so it must declare a default step count")
        });
        assert_eq!(
            steps_limit_error(id, default_steps, entry),
            None,
            "{id} defaults to {default_steps} steps; its own floor must admit that"
        );
        // …and at the floor admits too: the bound is `<`, like every other bound here.
        assert_eq!(steps_limit_error(id, *floor, entry), None, "{id}: at-floor");

        // 3. MUTATION CHECK: the floor actually refuses something, and names the model.
        let under = floor - 1;
        let message = steps_limit_error(id, under, entry).unwrap_or_else(|| {
            panic!("{id}: {under} steps is under the {floor}-step floor and must be rejected")
        });
        assert!(message.contains(id), "{id}: names the model: {message}");
        assert!(
            message.contains("Raise"),
            "{id}: gives the lever: {message}"
        );
    }
}

/// sc-19426 — the step floor's message, held to the same house convention as the duration
/// bounds': name the model, state what was asked and what the limit is, and give the lever.
///
/// The floor is deliberately 4 rather than MiniMax-H3's shipped 2, so every assertion below is
/// on a number that appears nowhere else in the entry — a message built from the wrong field
/// cannot coincidentally satisfy it.
#[test]
fn steps_limit_error_names_the_model_the_request_and_the_floor() {
    let turbo = payload(json!({ "limits": { "hardMinSteps": 4 } }));
    let message =
        steps_limit_error("minimax_h3_turbo", 1, &turbo).expect("1 is under the 4-step floor");
    assert!(
        message.contains("minimax_h3_turbo"),
        "names the model: {message}"
    );
    assert!(
        message.contains("at least 4 sampling steps"),
        "states the floor: {message}"
    );
    assert!(
        message.contains("asks for 1."),
        "states what was asked: {message}"
    );
    assert!(message.contains("Raise"), "gives the lever: {message}");

    // At the floor admits (`<`, matching every other bound in this module being strict), and
    // there is no ceiling arm to swallow it — a large count is somebody's slow render, not an
    // error.
    assert_eq!(steps_limit_error("minimax_h3_turbo", 4, &turbo), None);
    assert_eq!(steps_limit_error("minimax_h3_turbo", 500, &turbo), None);
    // …and it refuses, never clamps: the caller gets a message, not a rewritten step count.
    assert!(steps_limit_error("minimax_h3_turbo", 3, &turbo).is_some());
}

/// The infallibility contract [`hard_min_duration`] holds to, applied to the step floor: an
/// absent or unhonorable value is NO floor, so every video model that declares nothing — all ten
/// of them before sc-19426 — is byte-for-byte unchanged.
#[test]
fn absent_or_unhonorable_hard_min_steps_means_no_floor() {
    let bare = payload(json!({}));
    assert_eq!(hard_min_steps(&bare), None);
    assert_eq!(steps_limit_error("bernini", 1, &bare), None);

    for bad in [
        json!(0),
        json!(-1),
        json!(2.5),
        // An authored `2.0` is an `integer` to JSON Schema and a float to serde — so it reads as
        // NO floor, which is why the shipped-manifest test transcribes the bytes.
        json!(2.0),
        json!("2"),
        json!(null),
        json!(u64::from(u32::MAX) + 1),
    ] {
        let entry = payload(json!({ "limits": { "hardMinSteps": bad } }));
        assert_eq!(hard_min_steps(&entry), None, "unhonorable floor {bad}");
        assert_eq!(
            steps_limit_error("some_model", 1, &entry),
            None,
            "an unhonorable floor must not brick the model: {bad}"
        );
    }

    // The two hard floors are independent: declaring one must not conjure the other.
    let duration_only = payload(json!({ "limits": { "hardMinDuration": 5.1667 } }));
    assert_eq!(hard_min_steps(&duration_only), None);
    let steps_only = payload(json!({ "limits": { "hardMinSteps": 2 } }));
    assert_eq!(hard_min_duration(&steps_only), None);
    assert_eq!(duration_limit_error("steps_only", 0.5, &steps_only), None);
    assert!(steps_limit_error("steps_only", 1, &steps_only).is_some());
}

/// sc-24215 — the declared step CEILING refuses above it (never clamps), admits the ceiling
/// itself, composes with a floor, and is absent for every entry that declares nothing.
#[test]
fn a_declared_hard_max_steps_refuses_above_the_ceiling_only() {
    let capped = payload(json!({ "limits": { "hardMaxSteps": 300 } }));
    assert_eq!(hard_max_steps(&capped), Some(300));
    assert_eq!(steps_limit_error("capped", 300, &capped), None);
    let message = steps_limit_error("capped", 301, &capped).expect("301 is above 300");
    assert!(
        message.contains("capped") && message.contains("at most 300") && message.contains("301"),
        "{message}"
    );
    let both = payload(json!({ "limits": { "hardMinSteps": 4, "hardMaxSteps": 300 } }));
    assert!(steps_limit_error("both", 3, &both).is_some());
    assert_eq!(steps_limit_error("both", 4, &both), None);
    assert!(steps_limit_error("both", 301, &both).is_some());
    for bad in [json!(0), json!(2.5), json!("300"), json!(null)] {
        let entry = payload(json!({ "limits": { "hardMaxSteps": bad } }));
        assert_eq!(hard_max_steps(&entry), None, "{bad}");
    }
    assert_eq!(hard_max_steps(&payload(json!({}))), None);
    assert_eq!(
        steps_limit_error("bare", 100_000, &payload(json!({}))),
        None
    );
}

/// sc-19502 — the step MENU's message, held to the same house convention: name the model, what
/// was asked, the legal value, and the lever.
///
/// The menu is deliberately `[6]` rather than LTX's shipped `[8]`, so every assertion is on a
/// number appearing nowhere else in the entry — a message built from the wrong field cannot
/// coincidentally satisfy it.
#[test]
fn steps_menu_error_names_the_model_the_request_and_the_legal_value() {
    let distilled = payload(json!({ "limits": { "steps": [6] } }));
    let message = steps_menu_message(&distilled, 30);
    assert!(
        message.contains("some_distilled"),
        "names the model: {message}"
    );
    assert!(
        message.contains("fixed 6-step schedule"),
        "states the legal value: {message}"
    );
    assert!(
        message.contains("asks for 30 steps"),
        "states what was asked: {message}"
    );
    assert!(
        message.contains("Set steps to 6"),
        "gives the lever: {message}"
    );

    // ON the menu admits.
    assert_eq!(steps_limit_error("some_distilled", 6, &distilled), None);

    // THE POINT OF THE KEY: refused on BOTH sides. A floor would have admitted everything above
    // 6, which is exactly why `hardMinSteps` could not express LTX (sc-19426 → sc-19502).
    for off in [1u32, 5, 7, 8, 30, 500] {
        assert!(
            steps_limit_error("some_distilled", off, &distilled).is_some(),
            "{off} is off the menu and must be refused"
        );
    }
}

/// sc-19502 — a multi-value menu is a SET, not a range, and it is humanized for the caller.
#[test]
fn a_multi_value_step_menu_refuses_the_gap_between_its_entries() {
    let two = payload(json!({ "limits": { "steps": [4, 8] } }));
    assert_eq!(allowed_steps(&two).as_deref(), Some(&[4u32, 8][..]));
    assert_eq!(steps_limit_error("m", 4, &two), None);
    assert_eq!(steps_limit_error("m", 8, &two), None);
    let gap = steps_limit_error("m", 6, &two).expect("6 belongs to neither schedule");
    assert!(
        gap.contains("4 or 8"),
        "the menu must be humanized, not debug-printed: {gap}"
    );

    let three = payload(json!({ "limits": { "steps": [4, 8, 12] } }));
    let message = steps_menu_message(&three, 5);
    assert!(
        message.contains("renders at 4, 8, or 12 steps only"),
        "a multi-entry menu states the SET rather than taking the singular `-step` suffix, \
             which would read as one number: {message}"
    );
    assert!(
        message.contains("Set steps to 4, 8, or 12"),
        "…and still gives the lever: {message}"
    );
}

/// The infallibility contract [`allowed_fps`] holds to, applied to the step menu: an absent or
/// unsatisfiable declaration is NO menu, so every video model that declares nothing — all ten
/// of them before sc-19502 — is byte-for-byte unchanged.
///
/// The unsatisfiable half matters more here than for fps: a menu nothing can satisfy would
/// refuse EVERY request including the model's own `defaults.steps`, i.e. a typo would brick the
/// model rather than constrain it.
#[test]
fn absent_or_unsatisfiable_step_menu_means_no_menu() {
    let bare = payload(json!({}));
    assert_eq!(allowed_steps(&bare), None);
    assert_eq!(steps_limit_error("bernini", 1, &bare), None);

    for bad in [
        json!([]),
        json!([0]),
        // An authored `8.0` is an `integer` to JSON Schema and a float to serde — so it reads as
        // NO menu, which is why the shipped-manifest test transcribes the bytes.
        json!([8.0]),
        json!(["8"]),
        json!([null]),
        json!([u64::from(u32::MAX) + 1]),
        json!(8),
        json!(null),
    ] {
        let entry = payload(json!({ "limits": { "steps": bad } }));
        assert_eq!(allowed_steps(&entry), None, "unsatisfiable menu {bad}");
        assert_eq!(
            steps_limit_error("some_model", 1, &entry),
            None,
            "an unsatisfiable menu must not brick the model: {bad}"
        );
        assert_eq!(
            steps_limit_error("some_model", 8, &entry),
            None,
            "an unsatisfiable menu must not brick the model: {bad}"
        );
    }

    // A partially-unhonorable menu keeps its usable entries rather than collapsing to no menu.
    let mixed = payload(json!({ "limits": { "steps": [8, 0, "12"] } }));
    assert_eq!(allowed_steps(&mixed).as_deref(), Some(&[8u32][..]));

    // The menu and the floor are independent: declaring one must not conjure the other.
    let menu_only = payload(json!({ "limits": { "steps": [8] } }));
    assert_eq!(hard_min_steps(&menu_only), None);
    let floor_only = payload(json!({ "limits": { "hardMinSteps": 2 } }));
    assert_eq!(allowed_steps(&floor_only), None);
}

/// sc-19502 — when a model somehow declares BOTH, the exact menu governs and the floor is not
/// consulted. Pins the precedence so a future edit cannot make the two gates disagree with each
/// other on one entry.
#[test]
fn the_step_menu_governs_when_a_model_declares_both_axes() {
    let both = payload(json!({ "limits": { "steps": [8], "hardMinSteps": 2 } }));
    // 4 clears the floor but is off the menu — the menu must still refuse it, or declaring both
    // would be strictly weaker than declaring the menu alone.
    let message = steps_limit_error("m", 4, &both).expect("4 is off the menu");
    assert!(
        message.contains("fixed 8-step schedule"),
        "the menu's message must win: {message}"
    );
    assert_eq!(steps_limit_error("m", 8, &both), None);
}

/// sc-19502 — ONE catalog entry drives BOTH backends, so the single legal step count must be a
/// fact about the family rather than about whichever lane happens to run.
///
/// This guard lives here, in the consumer, because no crate in `inference` depends on both
/// `candle-gen-ltx` and `mlx-gen-ltx` (mlx is macOS-only), so neither lane can assert agreement
/// about the other. SceneWorks is where the two meet: `limits.steps` is lane-agnostic and is
/// enforced by `steps_limit_error` on the enqueue path regardless of which backend the router
/// picks, so if the lanes disagreed, one of them would be refusing a request the catalog
/// admits — or rendering something the caller did not ask for.
///
/// The transcribed 8 is `STAGE1_SIGMAS.len() - 1` in BOTH lanes (9 σ waypoints, byte-identical
/// tables). Changing this number requires re-baking both schedules.
#[test]
fn both_ltx_lanes_agree_on_one_legal_step_count() {
    let models = builtin_video_models();
    for id in ["ltx_2_3", "ltx_2_3_eros"] {
        let entry = models
            .iter()
            .find(|m| m.get("id").and_then(Value::as_str) == Some(id))
            .unwrap_or_else(|| panic!("{id} present"))
            .as_object()
            .expect("model entry object");

        // Exactly ONE legal count, and it is the distilled stage-1 schedule both lanes bake.
        assert_eq!(
            allowed_steps(entry).as_deref(),
            Some(&[8u32][..]),
            "{id}: both lanes run the same 8-step distilled stage-1 schedule"
        );

        // The declaration sits on the LANE-AGNOSTIC `limits` block, not on a per-backend one —
        // a per-lane declaration would let the two drift apart again, which is the defect
        // sc-19502 fixed rather than modelled.
        for backend in ["mlx", "candle"] {
            let per_lane = entry
                .get(backend)
                .and_then(Value::as_object)
                .and_then(|block| block.get("limits"))
                .and_then(Value::as_object)
                .and_then(|limits| limits.get("steps"));
            assert_eq!(
                per_lane, None,
                "{id}: {backend} must not carry a per-lane step menu; the constraint is a \
                     property of the distilled weights, not of the backend"
            );
        }

        // And the studio's own default is that one legal value, so the common path never trips
        // the gate.
        assert_eq!(
            entry
                .get("defaults")
                .and_then(Value::as_object)
                .and_then(|d| d.get("steps"))
                .and_then(Value::as_u64),
            Some(8),
            "{id}: defaults.steps must be the single legal count"
        );
    }
}

/// sc-19426 — the gate must read every shape the CONSUMER reads, or it is a gate with a hole.
///
/// `video_jobs::wan::advanced_opt_u32` is what the adapters call to turn `advanced.steps` into
/// the number handed to an engine. It accepts a JSON number *or* a numeric string and narrows
/// with `as u32`. Both quirks are load-bearing here: `"1"` reaches the engine as 1, and
/// `4294967297` WRAPS to 1 — so a gate that read only `as_u64`, or that used a checked
/// conversion, would admit two payloads that arrive at the engine as exactly the value the floor
/// exists to refuse.
#[test]
fn requested_steps_reads_every_shape_the_worker_reads() {
    let advanced = |value: Value| {
        let mut map = JsonObject::new();
        map.insert("steps".to_owned(), value);
        map
    };

    assert_eq!(requested_steps(&advanced(json!(20))), Some(20));
    assert_eq!(requested_steps(&advanced(json!("20"))), Some(20));
    assert_eq!(requested_steps(&advanced(json!(" 20 "))), Some(20));
    // The wrapping narrow, mirrored from `advanced_opt_u32`.
    assert_eq!(
        requested_steps(&advanced(json!(u64::from(u32::MAX) + 2))),
        Some(1)
    );
    // Nothing the worker cannot read is a value this gate may invent.
    assert_eq!(requested_steps(&JsonObject::new()), None);
    for unreadable in [json!(null), json!("many"), json!(-4), json!(true)] {
        assert_eq!(
            requested_steps(&advanced(unreadable.clone())),
            None,
            "unreadable steps {unreadable}"
        );
    }

    // …and the floor is applied to the value the worker would have passed down, not to the raw
    // JSON: both of these reach an engine as 1.
    let h3 = payload(json!({ "limits": { "hardMinSteps": 2 } }));
    for wire in [json!("1"), json!(u64::from(u32::MAX) + 2)] {
        let steps = requested_steps(&advanced(wire.clone())).expect("readable");
        assert!(
            steps_limit_error("minimax_h3", steps, &h3).is_some(),
            "{wire} arrives at the engine as {steps} and must be refused"
        );
    }
}
