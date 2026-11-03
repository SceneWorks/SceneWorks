use super::*;

#[test]
fn advertised_name_passes_through() {
    let advertised = ["euler", "dpmpp_2m", "uni_pc"];
    assert_eq!(
        normalize_sampling_knob(
            Some("dpmpp_2m".to_owned()),
            &advertised,
            "sampler",
            "qwen_image",
            "job-1",
            "mlx",
        ),
        Some("dpmpp_2m".to_owned())
    );
}

#[test]
fn unadvertised_name_falls_back_to_default() {
    // N3: a name the engine doesn't advertise (a legacy `dpmpp`/`unipc` recipe, or a candle
    // per-backend gap) is dropped to the engine default (`None`) instead of hard-failing the
    // generation in `validate_request`.
    let advertised = ["lightning"];
    assert_eq!(
        normalize_sampling_knob(
            Some("dpmpp".to_owned()),
            &advertised,
            "sampler",
            "qwen_image",
            "job-1",
            "mlx",
        ),
        None
    );
}

#[test]
fn unset_knob_stays_unset() {
    let advertised = ["euler"];
    assert_eq!(
        normalize_sampling_knob(None, &advertised, "scheduler", "m", "j", "mlx"),
        None
    );
}

// sc-7448 — the guidance method rides the same `normalize_sampling_knob` N3 guard as sampler/scheduler.
#[test]
fn guidance_method_advertised_passes_through() {
    let advertised = ["cfg", "cfg_pp"];
    assert_eq!(
        normalize_sampling_knob(
            Some("cfg_pp".to_owned()),
            &advertised,
            "guidanceMethod",
            "sdxl",
            "job-1",
            "mlx",
        ),
        Some("cfg_pp".to_owned())
    );
}

#[test]
fn guidance_method_unadvertised_falls_back_to_default() {
    // N3: cfg_pp requested on a model that doesn't advertise it (an engine that only does plain `cfg`,
    // or a stale recipe) drops to the engine default — never a `validate_request` hard-fail.
    let advertised = ["cfg"];
    assert_eq!(
        normalize_sampling_knob(
            Some("cfg_pp".to_owned()),
            &advertised,
            "guidanceMethod",
            "chroma",
            "job-1",
            "mlx",
        ),
        None
    );
}

// N1: the read strips the `"default"` sentinel + blanks to `None` (the absence of a method = the
// engine default = the guaranteed no-op), exactly like the sampler/scheduler read.
#[test]
fn read_guidance_method_strips_default_and_blank() {
    assert_eq!(
        read_advanced_guidance_method(&advanced(serde_json::json!({}))),
        None
    );
    assert_eq!(
        read_advanced_guidance_method(&advanced(serde_json::json!({"guidanceMethod": "default"}))),
        None
    );
    assert_eq!(
        read_advanced_guidance_method(&advanced(serde_json::json!({"guidanceMethod": "  "}))),
        None
    );
    assert_eq!(
        read_advanced_guidance_method(&advanced(serde_json::json!({"guidanceMethod": "cfg_pp"}))),
        Some("cfg_pp".to_owned())
    );
}

fn advanced(value: serde_json::Value) -> JsonObject {
    value.as_object().expect("object").clone()
}

// N1 (epic 7114): the guaranteed no-op default. A job with no sampling knobs — or the explicit
// `"default"` sentinel the UI sends for "Model default" — must resolve to ALL `None`, i.e. the engine
// runs its existing native path byte-for-byte. This guards the worker read against a future change
// that silently injects a non-default sampler onto the default path.
#[test]
fn n1_default_advanced_is_a_no_op() {
    assert_eq!(
        read_advanced_sampling_knobs(&advanced(serde_json::json!({}))),
        (None, None, None)
    );
    assert_eq!(
        read_advanced_sampling_knobs(&advanced(serde_json::json!({
            "sampler": "default",
            "scheduler": "default",
            "steps": 30
        }))),
        (None, None, None)
    );
    // Blank / whitespace-only names are also treated as the default (no name).
    assert_eq!(
        read_advanced_sampling_knobs(&advanced(
            serde_json::json!({"sampler": "  ", "scheduler": ""})
        )),
        (None, None, None)
    );
}

#[test]
fn read_passes_real_names_and_shift_through() {
    assert_eq!(
        read_advanced_sampling_knobs(&advanced(serde_json::json!({
            "sampler": "dpmpp_2m",
            "scheduler": "sgm_uniform",
            "schedulerShift": 2.5
        }))),
        (
            Some("dpmpp_2m".to_owned()),
            Some("sgm_uniform".to_owned()),
            Some(2.5)
        )
    );
    // schedulerShift accepts a numeric string and the legacy `timestepShift` key.
    let (_, _, shift) = read_advanced_sampling_knobs(&advanced(serde_json::json!({
        "timestepShift": "1.5"
    })));
    assert_eq!(shift, Some(1.5));
}
