use super::*;
use serde_json::json;
const PRODUCTS: &str = include_str!("../../../../config/manifests/registry/products.json");
const EVIDENCE: &str = include_str!("../../../../config/manifests/registry/evidence.json");
const MLX: &str =
    include_str!("../../../../config/engine-capabilities/runtime/capabilities.mlx.json");
const CANDLE: &str =
    include_str!("../../../../config/engine-capabilities/runtime/capabilities.candle.json");

#[test]
fn both_shipping_routes_and_the_v1_compatibility_projection_are_preserved() {
    let spec = builtin_model_spec("flux_schnell").unwrap();
    assert_eq!(spec.routes["mlx"].id, "flux1_schnell");
    assert_eq!(spec.routes["mlx"].conditioning, ["reference"]);
    assert_eq!(spec.routes["mlx"].supported_quants, ["q4", "q8"]);
    assert!(spec.routes["candle"].supported_quants.is_empty());
    assert!(spec.routes["candle"].conditioning.is_empty());
    let old: Value = serde_json::from_str(&crate::jsonc::strip_jsonc_comments(include_str!(
        "../../../../config/manifests/builtin.models.jsonc"
    )))
    .unwrap();
    let legacy = old["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|model| model["id"] == "flux_schnell")
        .unwrap();
    assert_eq!(
        &spec.project(None),
        legacy,
        "v1 raw readers must see the same public facts and exact calibration identities"
    );
    assert_eq!(
        crate::builtin_manifests::builtin_model_entry("flux_schnell"),
        legacy.as_object()
    );
    assert!(builtin_model_spec("flux_dev").is_none());
}

#[test]
fn author_can_override_product_choices_but_cannot_replace_native_or_evidence_authority() {
    let spec = builtin_model_spec("flux_schnell").unwrap();
    let overlay = json!({
        "name":"Family favorite","hidden":true,"ui":{"label":"Custom","pid":null,"promptGuide":{"path":"/tmp"}},
        "defaults":{"steps":8,"guidanceScale":99,"count":999,"sampler":"heun","resolution":"1024x1024"},
        "id":"alias","family":"sdxl","adapter":"external","downloads":[],"gated":true,"paths":{"model":"/tmp"},
        "limits":{"samplers":[]},"image":{"supportsGuidance":true},"mlx":{"minMemoryGb":0},"candle":{"calibrations":[]},
        "modelPath":"/tmp", "installed":true,"catalogScope":"user"
    });
    let mut expected = spec.project(None);
    expected["name"] = json!("Family favorite");
    expected["hidden"] = json!(true);
    expected["ui"]["label"] = json!("Custom");
    expected["defaults"]["steps"] = json!(8);
    expected["defaults"]["sampler"] = json!("heun");
    assert_eq!(spec.project(Some(&overlay)), expected);
}

#[test]
fn descriptor_contents_change_the_spec_while_provenance_does_not() {
    let mut mlx: Value = serde_json::from_str(MLX).unwrap();
    let build = |mlx: &Value| {
        assemble::assemble(
            PRODUCTS,
            EVIDENCE,
            &[("mlx", &mlx.to_string()), ("candle", CANDLE)],
        )
        .unwrap()
    };
    let before = build(&mlx);
    mlx["generatedFrom"]["inferenceRevision"] = json!("unrelated-pin");
    assert_eq!(
        serde_json::to_value(&before).unwrap(),
        serde_json::to_value(build(&mlx)).unwrap()
    );
    let desc = mlx["snapshot"]["generator_capabilities"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|row| row["id"] == "flux1_schnell")
        .unwrap();
    desc["supports_preview"] = json!(false);
    desc["samplers"]
        .as_array_mut()
        .unwrap()
        .retain(|entry| entry != "heun");
    let changed = build(&mlx);
    let spec = &changed["flux_schnell"];
    assert!(!spec.routes["mlx"].supports_preview);
    assert!(spec.routes["candle"].supports_preview);
    assert!(!spec
        .product
        .limits
        .samplers
        .iter()
        .any(|menu| menu == "heun"));
    let mut evidence: Value = serde_json::from_str(EVIDENCE).unwrap();
    evidence["flux_schnell"]["candle"]["calibrations"][0]["provider"] = json!("other");
    assert!(assemble::assemble(
        PRODUCTS,
        &evidence.to_string(),
        &[("mlx", MLX), ("candle", CANDLE)]
    )
    .unwrap_err()
    .contains("mismatched evidence provider"));
}

#[test]
fn shipped_snapshot_matches_the_build_time_join() {
    let shipped = include_str!("../../../../config/manifests/registry/specs.json").trim_end();
    let compiled = include_str!(concat!(env!("OUT_DIR"), "/model-registry.json"));
    assert_eq!(
        compiled, shipped,
        "refresh with cargo run -q -p sceneworks-core --bin dump-model-registry"
    );
}

#[test]
fn user_menus_only_narrow_native_product_choices_and_reconcile_defaults() {
    let spec = builtin_model_spec("flux_schnell").unwrap();
    let row = spec.project(Some(&json!({
        "limits": {"resolutions": ["768x768"], "count": [1, 2], "samplers": ["unshipped"], "schedulers": ["karras"]},
        "defaults": {"resolution": "1024x1024", "count": 2, "sampler": "unshipped"}
    })));
    assert_eq!(row["defaults"]["resolution"], "768x768");
    assert_eq!(row["defaults"]["count"], 2);
    assert_eq!(row["defaults"]["scheduler"], "karras");
    assert_eq!(
        row["limits"]["samplers"],
        spec.project(None)["limits"]["samplers"]
    );
    assert!(row["defaults"].get("sampler").is_none());
}

#[test]
fn mutable_distribution_revision_cannot_enter_the_static_catalog() {
    let mut products: Value = serde_json::from_str(PRODUCTS).unwrap();
    products[0]["downloads"][0]["revision"] = json!("main");
    let result = assemble::assemble(
        &products.to_string(),
        EVIDENCE,
        &[("mlx", MLX), ("candle", CANDLE)],
    );
    assert!(result.unwrap_err().contains("immutable download"));
}
