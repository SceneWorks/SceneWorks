use crate::contracts::JsonObject;
use crate::video_request::*;
use serde_json::{json, Value};

fn payload(value: Value) -> JsonObject {
    value.as_object().cloned().unwrap()
}

/// The `limits` block Qwen-Image 2.1 ships, transcribed from the manifest. Kept as a literal
/// rather than read out of `builtin.models.jsonc` so a failure here says which VALUE moved,
/// while the manifest-audit tests own the "the file still says this" half.
fn qwen_image_2_1_limits() -> JsonObject {
    payload(json!({
        "limits": {
            "resolutions": [
                "2048x2048", "2400x1792", "1792x2400", "2528x1696",
                "1696x2528", "2752x1536", "1536x2752"
            ],
            "count": [1, 2, 4, 8],
            "minDimension": 32,
            "maxDimension": 2752,
            "requiresDimensionsMultipleOf": 32,
            "maxReferenceAssets": 10,
            "hardMinSteps": 2,
            "samplers": ["default", "euler", "euler_ancestral", "heun", "dpmpp_2m",
                         "dpmpp_2m_sde", "dpmpp_sde", "er_sde", "uni_pc", "lcm", "ddim"],
            "schedulers": ["default", "normal", "simple", "karras", "exponential",
                           "sgm_uniform", "beta", "ddim_uniform", "beta57"]
        }
    }))
}

/// sc-24114: the sampler / scheduler menu is enforced at enqueue — a name no lane advertises
/// is a 400 naming the menu; a member, the `"default"` sentinel, a blank, and a model that
/// declares no menu are all untouched. A per-backend override's names count too.
#[test]
fn image_sampling_knob_error_enforces_the_declared_menu() {
    let entry = qwen_image_2_1_limits();
    let advanced = |value: Value| payload(value);
    for ok in [
        json!({}),
        json!({ "sampler": "default", "scheduler": "default" }),
        json!({ "sampler": "  " }),
        json!({ "sampler": "er_sde", "scheduler": "beta57" }),
        json!({ "sampler": "dpmpp_2m", "scheduler": "karras" }),
    ] {
        assert_eq!(
            image_sampling_knob_error("qwen_image_2_1", &advanced(ok.clone()), &entry),
            None,
            "{ok}"
        );
    }
    let refused = image_sampling_knob_error(
        "qwen_image_2_1",
        &advanced(json!({ "sampler": "made_up" })),
        &entry,
    )
    .expect("an off-menu sampler is refused");
    assert!(
        refused.contains("\"made_up\"") && refused.contains("er_sde"),
        "{refused}"
    );
    assert!(image_sampling_knob_error(
        "qwen_image_2_1",
        &advanced(json!({ "scheduler": "made_up" })),
        &entry
    )
    .is_some());
    // Undeclared menu ⇒ inert.
    assert_eq!(
        image_sampling_knob_error(
            "m",
            &advanced(json!({ "sampler": "x" })),
            &payload(json!({}))
        ),
        None
    );
    // A per-backend override contributes its names (the API does not know the lane).
    let split = payload(json!({
        "limits": { "samplers": ["default"] },
        "candle": { "limits": { "samplers": ["default", "euler"] } }
    }));
    assert_eq!(
        image_sampling_knob_error("m", &advanced(json!({ "sampler": "euler" })), &split),
        None
    );
    assert!(
        image_sampling_knob_error("m", &advanced(json!({ "sampler": "heun" })), &split).is_some()
    );
}

#[test]
fn an_image_model_that_declares_no_limits_is_untouched_by_the_new_gates() {
    // The whole safety argument for running these on EVERY image enqueue. A model with no
    // `limits` block at all, and one with a `limits` block that names none of these keys, must
    // both behave exactly as they did before the readers existed.
    for entry in [payload(json!({})), payload(json!({ "limits": {} }))] {
        assert_eq!(image_max_reference_assets(&entry), None);
        assert_eq!(image_reference_limit_error("m", 99, &entry), None);
        assert_eq!(
            image_dimension_envelope(&entry),
            ImageDimensionEnvelope::default()
        );
        // With nothing declared, the caller's blanket bounds decide — and they decide exactly
        // as the old global check did.
        assert_eq!(
            image_dimension_error("m", 1024, 1024, &entry, 256, 4096),
            None
        );
        assert!(image_dimension_error("m", 255, 1024, &entry, 256, 4096).is_some());
        assert!(image_dimension_error("m", 1024, 4097, &entry, 256, 4096).is_some());
        // ... including an off-grid size, which nobody declared a grid for.
        assert_eq!(
            image_dimension_error("m", 1023, 777, &entry, 256, 4096),
            None
        );
    }
}

#[test]
fn the_declared_envelope_lets_a_model_render_below_the_blanket_floor() {
    // 32 px is BELOW the API's historical 256 floor. Declaring it is the only thing that
    // admits it, which is why the floor became a per-model fallback rather than a constant.
    let entry = qwen_image_2_1_limits();
    assert_eq!(
        image_dimension_envelope(&entry),
        ImageDimensionEnvelope {
            min: Some(32),
            max: Some(2752),
            multiple: Some(32),
        }
    );
    assert_eq!(
        image_dimension_error("qwen_image_2_1", 32, 32, &entry, 256, 4096),
        None
    );
    // ... and the declared ceiling binds well below the blanket 4096.
    let message = image_dimension_error("qwen_image_2_1", 2784, 2048, &entry, 256, 4096)
        .expect("2784 is past the declared 2752 ceiling");
    assert!(message.contains("2752"), "{message}");
    assert!(message.contains("width"), "{message}");
    // The smallest legal size is refused one step below.
    let message = image_dimension_error("qwen_image_2_1", 2048, 16, &entry, 256, 4096)
        .expect("16 is below the declared 32 floor");
    assert!(message.contains("32"), "{message}");
    assert!(message.contains("height"), "{message}");
}

#[test]
fn an_off_grid_size_is_refused_with_the_two_legal_neighbours_named() {
    // An off-stride size used to pass every check in the app and die inside the provider. The
    // message has to be actionable, which means naming where to go rather than only what is
    // wrong: 2050 sits between 2048 and 2080.
    let entry = qwen_image_2_1_limits();
    let message = image_dimension_error("qwen_image_2_1", 2050, 2048, &entry, 256, 4096)
        .expect("2050 is not a multiple of 32");
    assert!(message.contains("32-pixel grid"), "{message}");
    assert!(message.contains("2048"), "{message}");
    assert!(message.contains("2080"), "{message}");
    // Every shipped preset is on the grid and inside the envelope, which is the claim the
    // Aspect menu depends on.
    for preset in [
        (2048, 2048),
        (2400, 1792),
        (1792, 2400),
        (2528, 1696),
        (1696, 2528),
        (2752, 1536),
        (1536, 2752),
    ] {
        assert_eq!(
            image_dimension_error("qwen_image_2_1", preset.0, preset.1, &entry, 256, 4096),
            None,
            "{preset:?} is a shipped preset and must be legal"
        );
    }
}

#[test]
fn the_eleventh_reference_is_refused_by_count_rather_than_truncated() {
    // Truncation is not a smaller render for this family: the template numbers the images and
    // block-causal attention makes each visible only to what follows, so dropping one
    // renumbers the rest. Ten is legal, eleven is a refusal that names both numbers.
    let entry = qwen_image_2_1_limits();
    assert_eq!(image_max_reference_assets(&entry), Some(10));
    for count in 0..=10 {
        assert_eq!(
            image_reference_limit_error("qwen_image_2_1", count, &entry),
            None,
            "{count} references are within the declared cap"
        );
    }
    let message = image_reference_limit_error("qwen_image_2_1", 11, &entry)
        .expect("an 11th reference must be refused");
    assert!(message.contains("up to 10"), "{message}");
    assert!(message.contains("11"), "{message}");
    // sc-24110: the wording names the ORDERED CONDITIONING LIST, not `referenceAssetIds`.
    // The count is over the flattened list — source, mask and references together — so naming
    // a single carrier would send the caller to trim a field that may not be the one over
    // budget, or that may be empty while the request is still eleven images.
    assert!(
        message.contains("ordered conditioning list"),
        "the refusal must name what is actually counted: {message}"
    );
    assert!(
        message.contains("source, mask and references together"),
        "…and spell out which carriers that is: {message}"
    );
}

#[test]
fn a_declared_cap_of_zero_says_this_model_takes_none() {
    // Distinct from an ABSENT key, which means "no opinion". A model that declares 0 is saying
    // something, and the message has to be different or the user is told to "reduce to 0".
    let entry = payload(json!({ "limits": { "maxReferenceAssets": 0 } }));
    assert_eq!(image_max_reference_assets(&entry), Some(0));
    let message = image_reference_limit_error("plain_model", 1, &entry).expect("refused");
    assert!(message.contains("takes no reference images"), "{message}");
    assert_eq!(image_reference_limit_error("plain_model", 0, &entry), None);
}

/// sc-24113 REGRESSION: a bare `requiresDimensionsMultipleOf` must NOT become an enqueue
/// refusal for the models that already declared one.
///
/// Six shipped `mage_flow*` image entries declare `16` with no min/max, and on the image lane
/// that key had exactly ONE reader before this story — the Studio's free Width/Height control,
/// which SNAPS. Enforcing the grid for every declaring model turned that silent snap into a
/// 400 and would have rejected sizes those models have always accepted, including stored
/// recipes and workflow replays: 1000×1000 is not a multiple of 16.
///
/// Read off the REAL manifest rather than a fixture, because the claim is about what is
/// shipped. Dropping the `envelope.min.is_some() && envelope.max.is_some()` gate in
/// `image_dimension_error` reds every row below.
#[test]
fn a_bare_dimension_stride_stays_advisory_for_the_models_that_already_declared_one() {
    let manifest: Value = serde_json::from_str(&crate::jsonc::strip_jsonc_comments(include_str!(
        "../../../config/manifests/builtin.models.jsonc"
    )))
    .expect("builtin.models.jsonc parses");
    let models = manifest["models"].as_array().expect("models array");

    let striding: Vec<&Value> = models
        .iter()
        .filter(|model| model["type"] == "image")
        .filter(|model| model["limits"]["requiresDimensionsMultipleOf"].is_number())
        .collect();
    // Anti-collapse: if the filter ever matches nothing, every assertion below is vacuous.
    assert!(
        striding.len() >= 7,
        "expected the six mage_flow* entries plus qwen_image_2_1, got {}",
        striding.len()
    );

    for model in striding {
        let id = model["id"].as_str().expect("model id").to_owned();
        let entry = model.as_object().expect("model object").clone();
        let declares_envelope = model["limits"]["minDimension"].is_number()
            && model["limits"]["maxDimension"].is_number();

        if id.starts_with("mage_flow") {
            assert!(
                !declares_envelope,
                "{id} gained a size envelope — re-read this test before re-pinning it"
            );
            // 1000 is off the ÷16 grid and comfortably inside the blanket bounds: exactly the
            // shape a stored recipe replays.
            assert_eq!(
                image_dimension_error(&id, 1000, 1000, &entry, 256, 4096),
                None,
                "{id} must still enqueue an off-grid size"
            );
            // ... and an on-grid one, so the row cannot pass by refusing everything.
            assert_eq!(
                image_dimension_error(&id, 1024, 1024, &entry, 256, 4096),
                None
            );
            // The blanket bounds still apply — the gate narrows the GRID check, not the range.
            assert!(image_dimension_error(&id, 200, 1024, &entry, 256, 4096).is_some());
        } else {
            // A model that declares the full envelope DOES get its declared grid enforced
            // (shape, not a population: qwen_image_2_1 at 32 px, iris_3b at 16 px, ...).
            assert!(
                declares_envelope,
                "{id} declares a stride without its envelope"
            );
            let multiple = model["limits"]["requiresDimensionsMultipleOf"]
                .as_u64()
                .expect("integer stride") as u32;
            let off_grid = 1024 + multiple / 2;
            let message = image_dimension_error(&id, off_grid, 1024, &entry, 256, 4096)
                .unwrap_or_else(|| panic!("{id} enforces its declared grid"));
            assert!(
                message.contains(&format!("{multiple}-pixel grid")),
                "{id}: {message}"
            );
            assert_eq!(
                image_dimension_error(&id, 1024, 1024, &entry, 256, 4096),
                None,
                "{id} admits an on-grid size"
            );
        }
    }
}

#[test]
fn an_unsatisfiable_declared_bound_falls_back_to_undeclared_rather_than_bricking_the_model() {
    // The same typo'd-manifest tolerance `hard_min_steps` takes. A zero/negative/fractional
    // bound, or a grid of 0 or 1, is not a constraint anyone could act on — and enforcing one
    // would make the model un-renderable rather than merely un-constrained.
    for bad in [
        json!({ "limits": { "minDimension": 0, "maxDimension": 0, "requiresDimensionsMultipleOf": 0 } }),
        json!({ "limits": { "minDimension": -8, "maxDimension": -8, "requiresDimensionsMultipleOf": 1 } }),
        json!({ "limits": { "minDimension": 32.5, "maxDimension": 2752.0, "requiresDimensionsMultipleOf": 32.0 } }),
        json!({ "limits": { "minDimension": "32", "maxDimension": "2752", "requiresDimensionsMultipleOf": "32" } }),
    ] {
        let entry = payload(bad.clone());
        assert_eq!(
            image_dimension_envelope(&entry),
            ImageDimensionEnvelope::default(),
            "{bad}"
        );
    }
    // A `multiple` of 1 in particular is a no-op grid, not a grid that rejects everything.
    let entry = payload(json!({ "limits": { "requiresDimensionsMultipleOf": 1 } }));
    assert_eq!(
        image_dimension_error("m", 1023, 777, &entry, 256, 4096),
        None
    );
}
