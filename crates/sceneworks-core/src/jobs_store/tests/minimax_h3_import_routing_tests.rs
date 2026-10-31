//! Imported dense MiniMax-H3 DiT claims (sc-26680, epic 26679): an entry-keyed plan-backed video
//! claim on BOTH native lanes, narrowed to the base partition the stamped assembly binds, and the
//! catalog support surfaces derived from those same claim predicates.
use super::*;
use crate::checkpoint_import::{
    CheckpointContainerV1, ImportLayerV1, ImportPlanV1, SourceLocatorV1,
};
use crate::minimax_h3_import::{stamp_imported_minimax_h3_entry, MiniMaxH3ImportAssembly};
use serde_json::{json, Value};

const IMPORTED_ID: &str = "singularity_fl2va_dense";

fn object(value: Value) -> Map<String, Value> {
    value.as_object().expect("test value is an object").clone()
}

fn dense_dit_plan() -> ImportPlanV1 {
    let locator = |path: &str| {
        SourceLocatorV1::linked(
            "root-0123456789abcdef",
            format!("h3/{path}"),
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        )
        .expect("valid locator")
    };
    ImportPlanV1::new(
        "checkpoint-plan-h3",
        "minimax-h3",
        vec![
            ImportLayerV1 {
                layer_id: "artifact:diffusion_pytorch_model.safetensors".to_owned(),
                role: "transformer".to_owned(),
                target_path: "diffusion_pytorch_model.safetensors".to_owned(),
                container: CheckpointContainerV1::Safetensors,
                source: locator("diffusion_pytorch_model.safetensors"),
            },
            ImportLayerV1 {
                layer_id: "artifact:config.json".to_owned(),
                role: "descriptor".to_owned(),
                target_path: "config.json".to_owned(),
                container: CheckpointContainerV1::JsonDescriptor,
                source: locator("config.json"),
            },
        ],
    )
    .expect("valid plan")
}

/// The catalog entry the import job stamps for a dense H3 DiT of one declared partition.
fn imported_entry_partition(partition: &str, base: &str) -> Map<String, Value> {
    let mut entry = object(json!({
        "id": IMPORTED_ID,
        "importPlan": { "checkpointId": format!("managed/singularity_{partition}") },
    }));
    let assembly = MiniMaxH3ImportAssembly::from_declared_plan(
        &dense_dit_plan(),
        serde_json::json!({"baseModel":base,"minimaxH3Partition":partition})
            .as_object()
            .unwrap(),
    )
    .expect("complete");
    stamp_imported_minimax_h3_entry(&mut entry, &assembly);
    crate::lora_family::apply_model_manifest_defaults(&mut entry, "video", Some("minimax-h3"));
    entry
}

/// The catalog entry the import job stamps for a dense H3 DiT.
fn imported_entry() -> Map<String, Value> {
    imported_entry_partition("fl2va", "minimax_h3")
}

fn payload(model: &str, mode: &str, extra: Value, entry: Map<String, Value>) -> Map<String, Value> {
    let mut payload = object(json!({
        "projectId": "p",
        "model": model,
        "mode": mode,
        "prompt": "a courier crosses a rain-lit square",
    }));
    payload.extend(object(extra));
    payload.insert("modelManifestEntry".to_owned(), Value::Object(entry));
    payload
}

fn claimed_on_both(payload: &Map<String, Value>) -> (bool, bool) {
    (
        video_request_is_mlx_eligible(&JobType::VideoGenerate, payload),
        video_request_is_candle_eligible(&JobType::VideoGenerate, payload),
    )
}

/// Terminal review Issue 1: a reference-partition (Ref2VA) import serves its adapter surface
/// THROUGH THE REFERENCE SHAPE. A text-only probe cannot be claimed by this partition, so an
/// advertisement derived from a single mode would withdraw adapters the reference shape
/// genuinely serves; the claim here proves both lanes carry it with real reference conditioning.
#[test]
fn a_ref2va_import_with_adapters_is_claimed_through_its_reference_shape() {
    let entry = imported_entry_partition("ref2va", "minimax_h3_ref");
    let turbo = json!({ "id": "minimax_h3_turbo_4step_v01" });

    // The text-only shape is outside the reference partition's bound surface, adapter or not.
    for loras in [json!({}), json!({ "loras": [turbo] })] {
        let claim = payload(IMPORTED_ID, "text_to_video", loras, entry.clone());
        assert_eq!(
            claimed_on_both(&claim),
            (false, false),
            "text_to_video stays refused for a reference-only partition"
        );
    }

    // reference_to_video WITH valid conditioning is the shape it serves — adapters included, on
    // BOTH native lanes, and through the enqueue gate.
    let claim = payload(
        IMPORTED_ID,
        "reference_to_video",
        json!({ "referenceAssetIds": ["r1"], "loras": [turbo] }),
        entry.clone(),
    );
    assert_eq!(
        claimed_on_both(&claim),
        (true, true),
        "both lanes must claim the ref2va reference shape with its adapter stack"
    );
    assert!(video_request_is_claimable_by_any_lane(
        &JobType::VideoGenerate,
        &claim
    ));
    assert!(
        crate::jobs_store::imported_video_adapter_surface_claimable(&entry, true, false)
            && crate::jobs_store::imported_video_adapter_surface_claimable(&entry, false, true),
        "the any-valid-shape advertisement must hold on each lane pair"
    );

    // Control: a partition with no claimable shape (the assembly lies) advertises nothing.
    let mut unclaimable = imported_entry_partition("ref2va", "minimax_h3_ref");
    unclaimable
        .entry("importPlan".to_owned())
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .unwrap()
        .remove("assembly");
    assert!(
        !crate::jobs_store::imported_video_adapter_surface_claimable(&unclaimable, true, true),
        "an entry that is no longer plan-backed advertises no adapter surface"
    );
}

#[test]
fn an_imported_h3_dit_is_claimed_on_both_lanes_for_its_bound_partition() {
    for (mode, extra) in [
        ("text_to_video", json!({})),
        ("image_to_video", json!({ "sourceAssetId": "first" })),
        (
            "first_last_frame",
            json!({ "sourceAssetId": "first", "lastFrameAssetId": "last" }),
        ),
    ] {
        let claim = payload(IMPORTED_ID, mode, extra, imported_entry());
        assert_eq!(
            claimed_on_both(&claim),
            (true, true),
            "{mode}: the imported DiT must be claimable on MLX and Candle"
        );
        assert!(
            video_request_is_claimable_by_any_lane(&JobType::VideoGenerate, &claim),
            "{mode}: the enqueue gate must admit it"
        );
    }
}

#[test]
fn an_imported_h3_id_prefix_does_not_select_the_resident_checkpoint() {
    let mut entry = imported_entry();
    entry.insert("id".to_owned(), json!("minimax_h3_custom"));
    let claim = payload(
        "minimax_h3_custom",
        "text_to_video",
        json!({}),
        entry.clone(),
    );
    assert_eq!(claimed_on_both(&claim), (true, true));
    assert!(imported_video_model_mac_support(&entry).supported);
    assert!(imported_video_model_candle_support(&entry).supported);

    // Only the exact resident ids are reserved. Without an import plan, each retains its own
    // partition's native route; attaching an imported assembly to either must still refuse.
    for (model, mode, extra) in [
        ("minimax_h3", "text_to_video", json!({})),
        (
            "minimax_h3_ref",
            "reference_to_video",
            json!({ "referenceAssetIds": ["ref"] }),
        ),
    ] {
        let imported = payload(model, mode, extra, entry.clone());
        assert_eq!(claimed_on_both(&imported), (false, false), "{model}");
        let mut resident = imported;
        resident.remove("modelManifestEntry");
        assert_eq!(claimed_on_both(&resident), (true, true), "{model}");
    }
}

#[test]
fn the_imported_h3_claim_is_narrow_on_every_axis() {
    let t2v = |entry| payload(IMPORTED_ID, "text_to_video", json!({}), entry);

    // Control for the mutations below.
    assert_eq!(claimed_on_both(&t2v(imported_entry())), (true, true));

    fn t2v_with_loras(loras: Value) -> Map<String, Value> {
        // `payload`'s `extra` block is an OBJECT whose keys extend the job payload — the adapter
        // list rides its `loras` key, exactly as an enqueued job's payload spells it. Passing the
        // raw array here is the merged-suite failure this shape pins.
        payload(
            IMPORTED_ID,
            "text_to_video",
            json!({ "loras": loras }),
            imported_entry(),
        )
    }

    // The plan identity is what makes the entry plan-backed at all.
    let mut no_plan = imported_entry();
    no_plan.remove("importPlan");
    assert_eq!(claimed_on_both(&t2v(no_plan)), (false, false));

    // An entry with a plan but no validated assembly is incomplete, never claimable.
    let mut no_assembly = imported_entry();
    no_assembly["importPlan"]
        .as_object_mut()
        .unwrap()
        .remove("assembly");
    assert_eq!(claimed_on_both(&t2v(no_assembly)), (false, false));

    // Another family's plan has no H3 lane — and on MLX it must not fall to the id ladder.
    let mut wan = imported_entry();
    wan.insert("family".to_owned(), json!("wan-video"));
    assert!(!video_request_is_mlx_eligible(
        &JobType::VideoGenerate,
        &t2v(wan)
    ));

    // A builtin id names the shipped weights: a plan riding under it is refused on both lanes,
    // even though the builtin id alone (no plan) is claimable.
    let under_builtin = payload("minimax_h3", "text_to_video", json!({}), imported_entry());
    assert_eq!(claimed_on_both(&under_builtin), (false, false));
    let builtin_alone = object(json!({ "model": "minimax_h3", "mode": "text_to_video" }));
    assert!(video_request_is_mlx_eligible(
        &JobType::VideoGenerate,
        &builtin_alone
    ));

    // Outside the bound base partition: references and mis-shaped keyframes.
    for (mode, extra) in [
        ("reference_to_video", json!({ "referenceAssetIds": ["r1"] })),
        ("text_to_video", json!({ "referenceAudioAssetIds": ["a1"] })),
        ("image_to_video", json!({})),
        (
            "image_to_video",
            json!({ "sourceAssetId": "first", "lastFrameAssetId": "last" }),
        ),
        ("first_last_frame", json!({ "sourceAssetId": "first" })),
    ] {
        let claim = payload(IMPORTED_ID, mode, extra.clone(), imported_entry());
        assert_eq!(
            claimed_on_both(&claim),
            (false, false),
            "{mode} {extra}: outside the imported DiT's bound surface"
        );
    }

    // The full native LoRA/LoKr/turbo surface is claimable on both backends — it reaches the same
    // per-render selected-task adapter install the builtin partitions use. The native installer
    // owns target/namespace/basis compatibility; this predicate owns only the request shape.
    for loras in [
        json!([{ "id": "minimax_h3_turbo_4step_v01" }]),
        json!([{ "id": "user_style_rank16" }]),
        json!([{ "id": "minimax_h3_turbo_4step_v01" }, { "id": "user_style_rank16" }]),
    ] {
        let claim = t2v_with_loras(loras.clone());
        assert_eq!(
            claimed_on_both(&claim),
            (true, true),
            "{loras}: compatible H3 adapters stay claimable on both native lanes"
        );
        assert!(video_request_is_claimable_by_any_lane(
            &JobType::VideoGenerate,
            &claim
        ));
    }

    // The advanced job types are never an imported-H3 shape.
    assert!(!video_request_is_mlx_eligible(
        &JobType::VideoExtend,
        &payload(
            IMPORTED_ID,
            "extend_clip",
            json!({ "sourceClipAssetId": "c" }),
            imported_entry()
        )
    ));
}

#[test]
fn the_catalog_surfaces_are_derived_from_the_same_claims() {
    let entry = imported_entry();
    let mac = imported_video_model_mac_support(&entry);
    let candle = imported_video_model_candle_support(&entry);
    assert!(mac.supported && candle.supported);
    for mode in ["text_to_video", "image_to_video", "first_last_frame"] {
        assert_eq!(
            mac.features.video_modes.get(mode),
            Some(&true),
            "mac {mode}"
        );
        assert_eq!(
            candle.features.video_modes.get(mode),
            Some(&true),
            "candle {mode}"
        );
    }
    for mode in ["reference_to_video", "extend_clip", "replace_person"] {
        assert_eq!(
            mac.features.video_modes.get(mode),
            Some(&false),
            "mac {mode}"
        );
        assert_eq!(
            candle.features.video_modes.get(mode),
            Some(&false),
            "candle {mode}"
        );
    }

    // The id-keyed surface cannot see the plan: the imported id alone is unsupported, which is
    // exactly why the entry-keyed form exists.
    assert!(!model_mac_support(IMPORTED_ID, "video", Some("minimax-h3")).supported);
    let mut unbound = entry.clone();
    unbound["importPlan"]
        .as_object_mut()
        .unwrap()
        .remove("assembly");
    assert!(!imported_video_model_mac_support(&unbound).supported);
    assert!(!imported_video_model_candle_support(&unbound).supported);
}

#[test]
fn explicit_ref_and_hybrid_imports_claim_only_their_declared_operations() {
    for partition in ["fl2va", "ref2va", "hybrid"] {
        let mut entry = imported_entry();
        entry.insert("minimaxH3Partition".to_owned(), json!(partition));
        entry.insert(
            "baseModel".to_owned(),
            json!(if partition == "ref2va" {
                "minimax_h3_ref"
            } else {
                "minimax_h3"
            }),
        );
        let assembly =
            MiniMaxH3ImportAssembly::from_declared_plan(&dense_dit_plan(), &entry).unwrap();
        stamp_imported_minimax_h3_entry(&mut entry, &assembly);
        crate::minimax_h3_import::apply_minimax_h3_studio_surface_defaults(&mut entry);
        for (mode, extra) in [
            ("text_to_video", json!({})),
            ("image_to_video", json!({"sourceAssetId":"first"})),
            // sc-26691: FL2VA's last-frame-only keyframe, which the API admits for fl2va/hybrid.
            ("image_to_video", json!({"lastFrameAssetId":"last"})),
            (
                "first_last_frame",
                json!({"sourceAssetId":"first","lastFrameAssetId":"last"}),
            ),
            (
                "reference_to_video",
                json!({"referenceAssetIds":["image1","image2"],"sourceClipAssetIds":["video1"],"referenceAudioAssetIds":["audio1"]}),
            ),
        ] {
            let expected =
                partition == "hybrid" || (partition == "ref2va") == (mode == "reference_to_video");
            assert_eq!(
                claimed_on_both(&payload(IMPORTED_ID, mode, extra, entry.clone())),
                (expected, expected),
                "{partition}/{mode}"
            );
        }
        // Explicit hybrid is an operation union, never permission to drop keyframes or refs.
        assert_eq!(
            claimed_on_both(&payload(
                IMPORTED_ID,
                "reference_to_video",
                json!({"sourceAssetId":"first","referenceAssetIds":["reference"]}),
                entry.clone()
            )),
            (false, false)
        );
        entry.remove("minimaxH3Partition");
        assert_eq!(
            claimed_on_both(&payload(IMPORTED_ID, "text_to_video", json!({}), entry)),
            (false, false)
        );
    }
}
