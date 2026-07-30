//! Contract tests for the sanitized workflow share envelope (sc-15946, epic 15945).
//!
//! Three are load-bearing. [`every_registered_builder_has_its_advanced_keys_classified`] parses
//! every builder in `ADVANCED_BUILDERS` so a new knob can neither silently leak nor silently
//! vanish; [`every_advanced_builder_in_the_web_app_is_accounted_for`] walks `apps/web/src` so a
//! new BUILDER cannot appear unclassified (sc-15946's lint read one file on the premise that it
//! was the only one — it was not); and [`no_value_in_a_built_envelope_is_path_shaped`] seeds the
//! request with paths on purpose and asserts none of them reach the file.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS;
use sceneworks_core::contracts::{Asset, JsonObject};
use sceneworks_core::jsonc::strip_jsonc_comments;
use sceneworks_core::workflow_share::{
    advanced_key_rule, build_workflow_share, build_workflow_share_from, is_path_shaped,
    parse_workflow_share, AdvancedBuilder, AdvancedBuilderShape, AdvancedDisposition,
    AdvancedKeySource, WorkflowAssetFacts, WorkflowShare, ADVANCED_BUILDERS, ADVANCED_KEY_RULES,
    DEFERRED_ADVANCED_BUILDERS, PERMANENT_EXEMPTION, PRODUCER_URL, PRODUCER_VERSION,
    WORKFLOW_SHARE_MAX_BYTES, WORKFLOW_SHARE_SCHEMA_VERSION,
};
use serde_json::{json, Value};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

fn read_repo_file(relative_path: &str) -> String {
    let path = repo_root().join(relative_path);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

// ---------------------------------------------------------------------------
// Golden input
// ---------------------------------------------------------------------------

/// A generated asset's sidecar, in the shape `project_store::build_image_sidecar_parts` writes.
fn golden_asset() -> Asset {
    serde_json::from_value(json!({
        "schemaVersion": 1,
        "id": "asset_9f2c",
        "projectId": "project_7a10",
        "generationSetId": "genset_31bb",
        "type": "image",
        "displayName": "Lighthouse in fog #2",
        "createdAt": "2026-07-29T13:04:11Z",
        "file": {
            "path": "assets/images/2026-07-29_krea_2_turbo_lighthouse_0002.png",
            "mimeType": "image/png",
            "width": 1024,
            "height": 1024,
            "duration": null,
            "fps": null
        },
        "status": { "favorite": true, "rating": 4, "rejected": false, "trashed": false },
        "recipe": {
            "mode": "edit_image",
            "model": "krea_2_turbo",
            "adapter": "flux_diffusers",
            "prompt": "a lighthouse in heavy fog, cinematic",
            "negativePrompt": "text, watermark",
            "seed": 880412,
            "loras": [],
            "stylePreset": "cinematic",
            "normalizedSettings": {},
            "rawAdapterSettings": {}
        },
        "lineage": {
            "parents": ["asset_source_1"],
            "sourceAssetId": "asset_source_1",
            "sourceTimestamp": null,
            "jobId": "job_5c8e"
        }
    }))
    .expect("golden asset parses")
}

/// The job row's `payload_json` for that asset — the exact `ImageJobRequest` the API stored,
/// including the fields it stamps on (`modelManifestEntry`, `seeds`, the resolved geometry).
fn golden_payload() -> JsonObject {
    json!({
        "projectId": "project_7a10",
        "projectName": "Michael's Unreleased Film",
        "mode": "edit_image",
        "prompt": "a lighthouse in heavy fog, cinematic",
        "negativePrompt": "text, watermark",
        "model": "krea_2_turbo",
        "count": 2,
        "seed": 880411,
        "seeds": [880411, 880412],
        "width": 1024,
        "height": 1024,
        "stylePreset": "cinematic",
        "styleId": "noir_bloom",
        "fitMode": "crop",
        "characterId": "character_c001",
        "characterLookId": "look_l001",
        "sourceAssetId": "asset_source_1",
        "referenceAssetIds": ["asset_ref_1", "asset_ref_2"],
        "maskAssetId": "asset_mask_1",
        "upscale": { "enabled": true, "factor": 2, "engine": "seedvr2", "softness": 0.25 },
        "modelManifestEntry": {
            "id": "krea_2_turbo",
            "installPath": "E:\\models\\krea_2_turbo",
            "downloads": [{ "repo": "kreaai/krea-2-turbo" }]
        },
        "loras": [{
            "id": "lora_1f0d",
            "name": "Foggy Coast",
            "weight": 0.65,
            "installedPath": "E:\\loras\\foggy-coast",
            "sourcePath": "/mnt/nas/loras/foggy-coast.safetensors",
            "source": { "provider": "huggingface", "repo": "acme/foggy-coast", "file": "v2.safetensors" }
        }],
        "advanced": {
            "resolution": "1024x1024",
            "sampler": "euler",
            "scheduler": "beta",
            "schedulerShift": 1.15,
            "steps": 28,
            "guidanceScale": 3.5,
            "guidanceMethod": "cfg_pp",
            "strength": 0.55,
            "styleId": "noir_bloom",
            "stylePrompt": "a lighthouse in heavy fog",
            "controlMode": "canny",
            "controlScale": 0.9,
            "controlImage": "asset_control_1",
            "controlWeights": { "overlayId": "overlay_7", "path": "E:\\overlays\\pose.safetensors" },
            "quantTier": "nvfp4",
            "mlxQuantize": 4,
            "flashAttn": false,
            "recipePresetId": "preset_local_1"
        }
    })
    .as_object()
    .cloned()
    .expect("golden payload is an object")
}

fn golden_fixture_path() -> PathBuf {
    repo_root()
        .join("tests")
        .join("fixtures")
        .join("workflow_share")
        .join("image-workflow-share.json")
}

fn load_golden_fixture() -> Value {
    let path = golden_fixture_path();
    let payload = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    serde_json::from_str(&payload)
        .unwrap_or_else(|error| panic!("failed to parse {}: {error}", path.display()))
}

// ---------------------------------------------------------------------------
// Golden fixture + round trip (matching tests/contract_roundtrip.rs)
// ---------------------------------------------------------------------------

// The three tests below compare parsed `Value`s and MUST keep doing so — never serialized text,
// and never a byte comparison against the fixture file. `serde_json::Map` is a `BTreeMap`
// (sorted) or an `IndexMap` (insertion-ordered) depending on the `preserve_order` feature, which
// Cargo unifies across the workspace, so the same envelope serializes its keys in a different
// ORDER under `cargo test -p sceneworks-core` than under the workspace build the `parity` job
// runs. `Value` equality is order-independent under both; a text comparison would not be, and
// would fail in exactly one of the two configurations while nothing was wrong (sc-15946).

#[test]
fn golden_envelope_round_trips_without_field_drift() {
    let original = load_golden_fixture();
    let typed: WorkflowShare =
        serde_json::from_value(original.clone()).expect("golden envelope deserializes");
    let encoded = serde_json::to_value(typed).expect("golden envelope serializes");

    assert_eq!(
        encoded, original,
        "the golden workflow-share envelope drifted after a typed round-trip"
    );
}

#[test]
fn the_builder_reproduces_the_golden_envelope() {
    let built = build_workflow_share(&golden_asset(), &golden_payload());
    let encoded = serde_json::to_value(&built).expect("built envelope serializes");
    let mut golden = load_golden_fixture();
    // The fixture pins the SHAPE, so a release version bump is not drift: the producer version
    // is asserted against `CARGO_PKG_VERSION` and the workspace manifest in
    // `producer_version_is_strict_semver_and_matches_the_workspace` instead.
    golden["producer"]["version"] = json!(PRODUCER_VERSION);

    assert_eq!(
        encoded,
        golden,
        "builder output drifted from the golden fixture.\nActual:\n{}",
        serde_json::to_string_pretty(&encoded).expect("pretty")
    );
}

/// The two build entry points are one builder (sc-15948).
///
/// sc-15948 embeds at the worker's write seam, where no `Asset` exists yet, so it calls
/// [`build_workflow_share_from`] with the per-image facts directly. If that were a second
/// implementation, the allow-list, the path guard and the payload-over-facts precedence could all
/// drift between "embedded when the file was written" and "rebuilt from a sidecar" — and the
/// embedded copy is the one that leaves the machine.
#[test]
fn build_workflow_share_is_the_facts_builder() {
    let asset = golden_asset();
    let payload = golden_payload();
    let via_asset = build_workflow_share(&asset, &payload);
    let via_facts = build_workflow_share_from(&WorkflowAssetFacts::from_asset(&asset), &payload);
    assert_eq!(
        serde_json::to_value(&via_facts).expect("serializes"),
        serde_json::to_value(&via_asset).expect("serializes"),
    );

    // And the facts really are the fallbacks: with an EMPTY payload nothing but them is left, so
    // each one has to show up in the envelope on its own.
    let facts = WorkflowAssetFacts {
        mode: "image_detail".to_owned(),
        model: "realvisxl".to_owned(),
        prompt: "ultra detailed, sharp focus".to_owned(),
        negative_prompt: "blurry, soft".to_owned(),
        seed: 7,
        width: Some(1536),
        height: Some(1024),
    };
    let bare = build_workflow_share_from(&facts, &JsonObject::new());
    assert_eq!(bare.mode, "image_detail");
    assert_eq!(bare.model, "realvisxl");
    assert_eq!(bare.prompt, "ultra detailed, sharp focus");
    assert_eq!(bare.negative_prompt, "blurry, soft");
    assert_eq!(bare.seed, Some(7));
    assert_eq!((bare.width, bare.height), (Some(1536), Some(1024)));
    assert!(bare.advanced.is_empty());
    assert!(bare.inputs.is_empty());

    // The payload wins over the facts wherever it speaks — except the seed, which is per-image and
    // is the one thing the payload cannot know.
    let payload = json!({ "mode": "text_to_image", "seed": 999, "seeds": [999, 1000] })
        .as_object()
        .cloned()
        .expect("object");
    let overridden = build_workflow_share_from(&facts, &payload);
    assert_eq!(overridden.mode, "text_to_image");
    assert_eq!(
        overridden.seed,
        Some(7),
        "the payload's batch base seed must never displace the produced image's own"
    );
}

/// The sanitizer runs in exactly ONE place, and it is the reducer both directions share.
///
/// `build_workflow_share` used to sanitize `advanced` and then hand the result to
/// `reduce_workflow_share`, which sanitizes it again. Every rule is idempotent today, so that
/// cost nothing but a second pass — and that is precisely why no assertion could see it. It is
/// pinned here as source structure because the failure it invites is a future shape rule that is
/// NOT idempotent (a truncation, a normalization, a de-duplication) quietly applying twice on
/// the build side and once on the parse side, which would put the two directions out of sync in
/// the one function written to keep them in step.
#[test]
fn the_builder_leaves_advanced_sanitizing_to_the_single_reducer() {
    let source = read_repo_file("crates/sceneworks-core/src/workflow_share.rs").replace('\r', "");
    // Comments stripped, so this reads CALLS and not the prose explaining them.
    let body = |name: &str| {
        let start = source
            .find(&format!("fn {name}("))
            .unwrap_or_else(|| panic!("workflow_share.rs no longer defines {name}"));
        let rest = &source[start..];
        let end = rest
            .find("\n}\n")
            .unwrap_or_else(|| panic!("could not find the end of {name}"));
        strip_comments(&rest[..end])
    };

    assert!(
        !body("build_workflow_share").contains("sanitize_advanced("),
        "`build_workflow_share` sanitizes `advanced` before handing it to \
         `reduce_workflow_share`, which sanitizes it again. Pass the raw map through and let the \
         reducer's single call do it."
    );
    assert!(
        body("reduce_workflow_share").contains("sanitize_advanced("),
        "the reducer stopped sanitizing `advanced`, so nothing does — this test's other half \
         would then pass while the allow-list was bypassed entirely"
    );
}

#[test]
fn the_golden_envelope_parses_back_through_the_reader() {
    let golden = load_golden_fixture();
    let parsed = parse_workflow_share(&golden).expect("the golden envelope parses");
    assert_eq!(
        serde_json::to_value(&parsed).expect("re-serializes"),
        golden,
        "parsing the golden envelope changed it"
    );
}

// ---------------------------------------------------------------------------
// Producer block
// ---------------------------------------------------------------------------

#[test]
fn producer_version_is_strict_semver_and_matches_the_workspace() {
    let share = build_workflow_share(&golden_asset(), &golden_payload());
    let version = share.producer.version;

    assert_eq!(version, PRODUCER_VERSION);

    // Strict MAJOR.MINOR.PATCH. A dirty-tree suffix (`0.8.1-dirty`), build metadata
    // (`0.8.1+ci.42`), a `v` prefix or a hostname must all fail here — that string ships in
    // every image and is the one field the allow-list cannot catch.
    let parts: Vec<&str> = version.split('.').collect();
    assert_eq!(
        parts.len(),
        3,
        "producer.version must be MAJOR.MINOR.PATCH, got {version:?}"
    );
    for part in &parts {
        assert!(
            !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()),
            "producer.version segment {part:?} is not numeric ({version:?}) — a git describe, \
             dirty-tree suffix, CI build number or hostname must never reach the envelope"
        );
        assert!(
            *part == "0" || !part.starts_with('0'),
            "producer.version segment {part:?} has a leading zero ({version:?})"
        );
    }

    // The canonical repo URL, and nothing derived from this machine.
    assert_eq!(share.producer.url, PRODUCER_URL);
    assert_eq!(share.producer.name, "SceneWorks");

    // The workspace version is the single source of truth (scripts/sync-version.mjs keeps the
    // web/desktop manifests in lockstep with it).
    let workspace_version = read_repo_file("Cargo.toml")
        .lines()
        .skip_while(|line| line.trim() != "[workspace.package]")
        .find_map(|line| {
            line.strip_prefix("version")
                .and_then(|rest| rest.split('"').nth(1))
                .map(str::to_owned)
        })
        .expect("root Cargo.toml declares [workspace.package] version");
    assert_eq!(
        version, workspace_version,
        "producer.version must be the workspace version"
    );
}

#[test]
fn the_parser_branches_on_schema_version_and_never_on_producer_version() {
    let base = serde_json::to_value(build_workflow_share(&golden_asset(), &golden_payload()))
        .expect("serializes");

    // A wildly different producer.version is irrelevant to parsing: same schemaVersion, so it
    // reads, and it reads to the SAME workflow.
    let mut alien_build = base.clone();
    alien_build["producer"]["version"] = json!("99.0.0");
    let parsed =
        parse_workflow_share(&alien_build).expect("producer.version must not gate parsing");
    assert_eq!(parsed.producer.version, "99.0.0");
    assert_eq!(parsed.prompt, base["prompt"].as_str().expect("prompt"));

    // Our own producer.version with a future schemaVersion must be rejected — proving the
    // branch is on schemaVersion alone and not on the build string.
    let mut future_contract = base;
    future_contract["schemaVersion"] = json!(WORKFLOW_SHARE_SCHEMA_VERSION + 1);
    let error = parse_workflow_share(&future_contract)
        .expect_err("a future schemaVersion must be rejected");
    let message = error.to_string();
    assert!(
        message.contains(&(WORKFLOW_SHARE_SCHEMA_VERSION + 1).to_string())
            && message.contains(&WORKFLOW_SHARE_SCHEMA_VERSION.to_string()),
        "the error must name both versions, got: {message}"
    );
}

// ---------------------------------------------------------------------------
// Coverage lint
// ---------------------------------------------------------------------------

/// The registry entry for one source tag, or a failure naming what is missing.
fn builder_for(source: AdvancedKeySource) -> &'static AdvancedBuilder {
    ADVANCED_BUILDERS
        .iter()
        .find(|builder| builder.source == source)
        .unwrap_or_else(|| panic!("no `ADVANCED_BUILDERS` entry for {source:?}"))
}

/// Every key `ADVANCED_KEY_RULES` classifies, in either direction.
fn classified_keys() -> BTreeSet<String> {
    ADVANCED_KEY_RULES
        .iter()
        .map(|rule| rule.key.to_owned())
        .collect()
}

/// The whole coverage lint for ONE registered builder, in both directions.
///
/// Split out of the per-builder tests so the registry-wide sweep and the two named lints
/// sc-15946/sc-15948 shipped assert exactly the same thing about their own entries — a builder
/// cannot be covered by the sweep and not by its own test, or vice versa.
fn assert_builder_is_fully_classified(builder: &AdvancedBuilder) {
    let path = builder.path;
    let function = builder.function;
    let source = read_repo_file(path);
    let emitted = emitted_keys(builder, &source);

    // A sanity floor plus named anchors: if an extractor ever stops understanding its file (a
    // refactor to a different shape, a move), it must fail loudly rather than pass with an empty
    // set. A lint that has quietly stopped looking protects nothing.
    assert!(
        emitted.len() >= builder.minimum_keys,
        "only {} keys were extracted from {function} in {path}, below the registry's declared \
         floor of {} — the extractor no longer understands that file, so this lint is not \
         protecting anything. Fix `emitted_keys` in this test.",
        emitted.len(),
        builder.minimum_keys
    );
    for anchor in builder.anchors {
        assert!(
            emitted.contains(*anchor),
            "the extractor missed the known key `{anchor}` in {function} ({path}), so this lint \
             is not protecting anything"
        );
    }

    let classified = classified_keys();
    let unclassified: Vec<&String> = emitted.difference(&classified).collect();
    assert!(
        unclassified.is_empty(),
        "{function} in {path} can emit {unclassified:?}, which `ADVANCED_KEY_RULES` in \
         crates/sceneworks-core/src/workflow_share.rs does not classify.\n\
         That builder feeds an embedding lane ({lane}), so an unclassified key is dropped \
         silently from every shared image it writes.\n\
         A new advanced knob must be classified before it ships: add it to the table as \
         `allow(..)` if it describes WHAT TO MAKE (it travels in shared images) or `deny(..)` \
         if it describes WHAT THIS MACHINE CAN AFFORD (tier, kernel, GPU) or names anything \
         local (an id, a path, a preset). Either way write the reason — an unclassified key \
         is dropped silently today and that is exactly what this lint exists to stop.",
        lane = builder.lane
    );

    let tagged: BTreeSet<String> = ADVANCED_KEY_RULES
        .iter()
        .filter(|rule| rule.source == builder.source)
        .map(|rule| rule.key.to_owned())
        .collect();
    if builder.shape == AdvancedBuilderShape::NoAdvancedMap {
        assert!(
            tagged.is_empty(),
            "{function} in {path} is registered as emitting no `advanced` map, but \
             `ADVANCED_KEY_RULES` tags {tagged:?} to it. Either it grew a map (change its \
             registry `shape`) or those rules belong to another builder."
        );
        return;
    }

    // Reverse-direction accountability. A rule's `source` is the builder whose JS the lint reads
    // to decide whether the rule is stale, so a key that ONLY this builder emits must be tagged to
    // it — otherwise nothing ever notices the knob being deleted from the UI. Keys several builders
    // emit are held by whichever one owns them; that builder's own stale check covers them.
    //
    // Stricter than the flat "at least one rule names this builder" this generalizes: it is
    // per-key, so a builder can gain an exclusive knob tagged to the wrong source and still fail.
    let elsewhere: BTreeSet<String> = ADVANCED_BUILDERS
        .iter()
        .filter(|other| other.source != builder.source)
        .flat_map(|other| emitted_keys(other, &read_repo_file(other.path)))
        .collect();
    for key in emitted.difference(&elsewhere) {
        let rule = advanced_key_rule(key).expect("classified above");
        assert_eq!(
            rule.source, builder.source,
            "`{key}` is emitted ONLY by {function} in {path}, but its rule is tagged \
             `AdvancedKeySource::{:?}`. The tag is what tells the lint which JS to re-read, so a \
             key nobody else emits must be tagged to its own builder or nothing will ever notice \
             it disappearing from the UI.",
            rule.source
        );
    }

    let stale: Vec<&String> = tagged.difference(&emitted).collect();
    assert!(
        stale.is_empty(),
        "`ADVANCED_KEY_RULES` classifies {stale:?} as coming from {function}, but {path} no \
         longer emits them.\n\
         Either the knob was removed (drop the rule), it moved to another builder (re-tag the \
         rule), or it moved to the server (`AdvancedKeySource::Server`). A rule that no longer \
         matches its source is a classification nobody is maintaining."
    );
}

/// Every builder in the registry has every key it can emit classified (sc-15948).
///
/// The generalized form of the two lints below. sc-15946 shipped one of them on the premise that
/// `buildImageJobAdvanced` was the only builder filling `advanced`; sc-15948 found three more, one
/// of which (`cnScale`) was already losing data and two of which (`angleSet`,
/// `imageGuidanceScale`) were live leaks on lanes that embed. Patching one file per discovery
/// closes nothing, so this walks the registry instead — and
/// [`every_advanced_builder_in_the_web_app_is_accounted_for`] is what makes the registry complete.
#[test]
fn every_registered_builder_has_its_advanced_keys_classified() {
    assert!(
        ADVANCED_BUILDERS.len() >= 8,
        "the builder registry shrank to {} entries — a lane was removed from the lint rather than \
         from the product?",
        ADVANCED_BUILDERS.len()
    );
    for builder in ADVANCED_BUILDERS {
        assert_builder_is_fully_classified(builder);
    }

    // The rules table must not carry a key twice with two dispositions.
    assert_eq!(
        classified_keys().len(),
        ADVANCED_KEY_RULES.len(),
        "`ADVANCED_KEY_RULES` has duplicate keys"
    );
    // Every rule explains itself — the failure message above tells an author to write one.
    for rule in ADVANCED_KEY_RULES {
        assert!(
            rule.reason.len() > 20,
            "rule for `{}` needs a real reason",
            rule.key
        );
    }
}

/// The registry and the source tag are one thing, asserted from both sides.
///
/// A `AdvancedKeySource` variant with no registry entry would silently opt its rules out of the
/// reverse-direction check; a registry entry sharing a tag with another would make the reverse
/// check read the wrong file.
#[test]
fn every_source_tag_names_exactly_one_registered_builder() {
    for source in AdvancedKeySource::ALL {
        let entries = ADVANCED_BUILDERS
            .iter()
            .filter(|builder| builder.source == *source)
            .count();
        let expected = usize::from(*source != AdvancedKeySource::Server);
        assert_eq!(
            entries, expected,
            "`AdvancedKeySource::{source:?}` has {entries} `ADVANCED_BUILDERS` entries, expected \
             {expected}. Every variant but `Server` names exactly one builder — that pairing is \
             what lets the lint read the right file for a rule's tag."
        );
    }
    for builder in ADVANCED_BUILDERS {
        assert!(
            AdvancedKeySource::ALL.contains(&builder.source),
            "`AdvancedKeySource::ALL` is missing {:?}, so the check above cannot see it",
            builder.source
        );
        assert!(
            builder.lane.len() > 20,
            "the registry entry for {} needs a real `lane` — it is what tells the next author why \
             those keys matter",
            builder.function
        );
    }
}

/// Every deferral either names a story or is an explicit permanent exemption, so "out of scope"
/// cannot be a shrug (sc-15956 owns video; the training namespace is exempt).
///
/// The permanent category exists because the alternative is worse. `trainingConfigSnapshot` builds
/// a ~20-key `advanced` map that no story will ever classify — it is trainer hyperparameters on a
/// lane that embeds nothing, a different namespace that shares a name — and a deferral list that
/// only accepts `sc-` ids would have forced either a fake story id or leaving the biggest
/// unaccounted builder in the tree unaccounted. So a permanent reason must say what would have to
/// change (an embedding write seam) and where the entry moves when it does.
#[test]
fn every_deferred_builder_names_the_story_that_owns_it() {
    assert!(
        !DEFERRED_ADVANCED_BUILDERS.is_empty(),
        "the deferral list emptied — if every builder is now classified, the video lanes embed \
         and their keys must be in `ADVANCED_KEY_RULES`"
    );
    for deferred in DEFERRED_ADVANCED_BUILDERS {
        let permanent = deferred.reason.starts_with(PERMANENT_EXEMPTION);
        assert!(
            permanent || deferred.reason.contains("sc-"),
            "the deferral for {} in {} must name the story that owns classifying its keys, or \
             start with `{PERMANENT_EXEMPTION}` and say why no story ever will, got {:?}",
            deferred.function,
            deferred.path,
            deferred.reason
        );
        if permanent {
            // A permanent exemption is the one shape nobody will revisit on a story's schedule,
            // so the reason has to carry the escape route itself.
            assert!(
                deferred.reason.contains("embed")
                    && deferred.reason.contains("ADVANCED_BUILDERS")
                    && deferred.reason.len() > 200,
                "the permanent exemption for {} in {} must state what would have to change (a \
                 write seam that EMBEDS) and that the entry then moves into `ADVANCED_BUILDERS`, \
                 got {:?}",
                deferred.function,
                deferred.path,
                deferred.reason
            );
        }
        assert!(
            !ADVANCED_BUILDERS
                .iter()
                .any(|builder| builder.path == deferred.path
                    && builder.function == deferred.function),
            "{} in {} is both registered and deferred",
            deferred.function,
            deferred.path
        );
    }
}

/// What to do about an `advanced` map nothing accounts for.
///
/// Three branches, and the third is not a formality. This sweep reads the NAME `advanced`, and
/// nothing in the text separates a UI-state local called `advanced` from a payload map — a JSX prop
/// and a parameter default can be recognized by their shape and are skipped, a plain local cannot
/// be. Telling that author to "classify every key it can emit" would be wrong advice on a
/// build-blocking failure, which is how a lint gets deleted instead of diagnosed. Renaming the local
/// is the honest fix, and it is offered here rather than left to be guessed.
const UNACCOUNTED_ADVANCED_ADVICE: &str = "\
    Every place the web app fills `advanced` must be accounted for, because four of those places \
    now feed a lane that embeds the map into the PNG (epic 15945). So either:\n\
    * add the (file, function) pair to `ADVANCED_BUILDERS` in \
    crates/sceneworks-core/src/workflow_share.rs and classify every key it can emit; or\n\
    * if its lane does not embed yet, add it to `DEFERRED_ADVANCED_BUILDERS` with the story that \
    owns turning it on; or\n\
    * if this is not a job payload at all, RENAME THE LOCAL. This lint reads the name `advanced`, \
    and a non-payload map that carries that name is indistinguishable from one that ships.";

/// Call heads an indirect `advanced` expression may name WITHOUT being a declared builder.
///
/// (file, call head, why). One entry, and it is not a builder: `useCharacterAdvancedOptions` is the
/// hook that returns the character panel's advanced-options CONTROLLER — the thing whose
/// `buildAdvanced` is registered as [`AdvancedKeySource::CharacterBuilder`] and is called at
/// `advanced.buildAdvanced(controller.advancedExtras)`. So `const advanced =
/// useCharacterAdvancedOptions(…)` names a controller, not a map: there are no keys behind it for
/// this lint to classify, and the keys that do exist are already covered by the entry for
/// `buildAdvanced`.
///
/// Path-scoped on purpose. The same helper name called from another screen is a NEW indirection and
/// has to be looked at rather than inherited.
const ADVANCED_INDIRECTION_CALL_HEADS: &[(&str, &str, &str)] = &[(
    "apps/web/src/screens/characterPanels.jsx",
    "useCharacterAdvancedOptions",
    "The hook that returns the CharacterAdvancedOptions controller. Its `buildAdvanced` is the \
     registered builder; the local this initializes is the controller itself.",
)];

/// An exemption that no longer describes the tree is how a lint loses a subject quietly.
///
/// The same staleness check the registry and the deferral list get: if the hook call moves or is
/// renamed, the entry must go rather than sit there excusing something that is not there.
#[test]
fn every_indirection_exemption_still_describes_the_tree() {
    for (path, head, reason) in ADVANCED_INDIRECTION_CALL_HEADS {
        assert!(
            reason.len() > 40,
            "the indirection exemption for `{head}` in {path} needs a real reason — it is the only \
             thing telling the next author why an unreadable expression is allowed"
        );
        let stripped = scannable(&read_repo_file(path));
        let found = advanced_map_signals(&stripped).into_iter().any(|signal| {
            signal.kind == AdvancedSignalKind::Indirect && signal.called.as_deref() == Some(*head)
        });
        assert!(
            found,
            "`ADVANCED_INDIRECTION_CALL_HEADS` exempts `{head}` in {path}, but no `advanced` \
             expression there comes from it any more — drop the exemption instead of leaving a \
             hole open for whatever gets called `{head}` next"
        );
    }
}

/// The trip-wire that closes the class: every place `apps/web/src` builds an `advanced` map — or
/// hands one in from a helper — is classified, deferred, or exempt by name (sc-15948).
///
/// This is the assertion sc-15946 was missing. Its lint read one file, so the second builder
/// (`buildDetailJobBody`) and then the third and fourth (`CharacterAdvancedOptions.buildAdvanced`
/// with the `advancedExtras` controllers, and `DocumentStudio.submit`) were each found by hand,
/// after shipping, one bug at a time. Instead of trusting a premise about how many builders exist,
/// this walks `apps/web/src` for every place an `advanced` map is built and demands that each one
/// sit inside a registered builder or a declared deferral.
///
/// # Exactly what it proves
///
/// In every non-test `.js`/`.jsx` file under `apps/web/src`, each of these fails the build unless
/// it falls inside the body of a function [`ADVANCED_BUILDERS`] or [`DEFERRED_ADVANCED_BUILDERS`]
/// declares:
///
/// * an inline literal — `advanced: { … }`, `advancedExtras: { … }`;
/// * an assigned literal — `advanced = { … }`;
/// * a property assignment — `advanced.key = …`;
/// * INDIRECTION — `advanced: buildFoo(state)`, `advanced: someVar`, `advanced =
///   compactObject({ … })`. The map is assembled somewhere this scan cannot read, so the call has
///   to name a declared builder (that builder's own entry accounts for the keys) or the signal
///   stands. This is the hole the first cut of this sweep had: it matched only a literal `{`, so
///   `trainingConfigSnapshot` built a ~20-key map through `compactObject` and the lint stayed
///   green — a live counterexample to the "we found all the builders" premise sc-15946 died of.
///
/// Plus, in every file it reads and not merely inside a declared builder, a write it could not
/// follow at all: `advanced["k"] = v`, `Object.assign(x.advanced, …)`, `advanced.push(…)`.
///
/// # What it does NOT prove
///
/// * That a declared builder's keys are classified. That is
///   [`every_registered_builder_has_its_advanced_keys_classified`], which also refuses a SECOND,
///   unread `advanced` map inside a declared span — a nested helper inside `DocumentStudio.submit`
///   is not accounted for by `submit`'s entry, because the extractor never reads it.
/// * That indirection is caught in assignment position when the right-hand side is not a call.
///   `const advanced = defaults.advanced ?? {}` is a READ of someone else's map, not a build, and
///   flagging member reads made this sweep fire on the training config's draft reader. Key
///   position (`advanced:`) is strict about every non-brace expression; assignment position needs
///   a call.
/// * Anything about `advanced` maps assembled outside `apps/web/src`.
///
/// A NEW builder therefore fails the build the moment it appears — a new file, a new function in a
/// file that is already covered, a new `advancedExtras` on a controller, or a new helper a payload
/// calls — and the author has to decide: classify its keys (move it into [`ADVANCED_BUILDERS`]) or
/// state which story owns it (add it to [`DEFERRED_ADVANCED_BUILDERS`]).
#[test]
fn every_advanced_builder_in_the_web_app_is_accounted_for() {
    // (path, function) → the declared body span, plus whether it is classified or deferred.
    let mut declared: Vec<(&str, &str, bool)> = ADVANCED_BUILDERS
        .iter()
        .map(|builder| (builder.path, builder.function, true))
        .collect();
    declared.extend(
        DEFERRED_ADVANCED_BUILDERS
            .iter()
            .map(|deferred| (deferred.path, deferred.function, false)),
    );

    let files = web_source_files();
    assert!(
        files.len() >= 100,
        "only {} web source files were walked — the discovery scan lost the tree it is supposed \
         to sweep, so it is not protecting anything",
        files.len()
    );

    // Every declared builder must resolve to a real function in a real file. `function_body_span`
    // panics with instructions when a rename or a move has left the registry pointing at nothing.
    let mut spans: Vec<(&str, &str, usize, usize)> = Vec::new();
    for (path, function, _) in &declared {
        let stripped = scannable(&read_repo_file(path));
        let (start, end) = function_body_span(&stripped, function, path);
        spans.push((path, function, start, end));
    }

    let declared_functions = declared_builder_functions();

    let mut covered_by: Vec<(&str, &str)> = Vec::new();
    for relative in &files {
        let stripped = scannable(&read_repo_file(relative));
        // Not only inside a declared builder: a write this lint cannot follow is invisible
        // wherever it is, and `body.payload.advanced["k"] = v` in a brand-new file was green
        // before this call moved out of the `AssignedObject` arm.
        refuse_invisible_advanced_writes(&stripped, relative);
        for signal in advanced_map_signals(&stripped) {
            if signal.calls_a_declared_builder(relative, &declared_functions) {
                continue;
            }
            // The INNERMOST declared span wins, so a nested helper that has been given its own
            // registry entry is credited to that entry rather than to the function around it.
            let owner = spans
                .iter()
                .filter(|(path, _, start, end)| {
                    *path == relative.as_str() && signal.offset >= *start && signal.offset < *end
                })
                .min_by_key(|(_, _, start, end)| end - start);
            let (_, function, _, _) = owner.unwrap_or_else(|| {
                let line = 1 + stripped[..signal.offset].matches('\n').count();
                panic!(
                    "{relative}:{line} builds an `advanced` map ({}) inside no builder that \
                     `ADVANCED_BUILDERS` or `DEFERRED_ADVANCED_BUILDERS` declares.{}\n{}",
                    signal.kind.describe(),
                    signal.indirection_advice(),
                    UNACCOUNTED_ADVANCED_ADVICE,
                )
            });
            covered_by.push((relative.as_str(), function));
        }
    }

    // The other direction: a declared builder that no longer builds an `advanced` map is a stale
    // entry, and a stale deferral is how a video lane could start embedding unnoticed.
    //
    // Two shapes are exempt by construction. `NoAdvancedMap` asserts the absence of a map — that is
    // its whole point. `ReturnedObject` builds one by RETURNING it (`buildImageJobAdvanced` never
    // writes the word `advanced`), so there is nothing for a signal scan to find; the floor and
    // anchors in `assert_builder_is_fully_classified` are what hold that shape honest.
    for builder in ADVANCED_BUILDERS {
        if matches!(
            builder.shape,
            AdvancedBuilderShape::NoAdvancedMap | AdvancedBuilderShape::ReturnedObject
        ) {
            continue;
        }
        assert!(
            covered_by
                .iter()
                .any(|(path, function)| *path == builder.path && *function == builder.function),
            "`ADVANCED_BUILDERS` registers {} in {} but the discovery scan finds no `advanced` map \
             built there any more",
            builder.function,
            builder.path
        );
    }
    for deferred in DEFERRED_ADVANCED_BUILDERS {
        assert!(
            covered_by
                .iter()
                .any(|(path, function)| *path == deferred.path && *function == deferred.function),
            "`DEFERRED_ADVANCED_BUILDERS` defers {} in {} but the discovery scan finds no \
             `advanced` map built there any more — drop the deferral",
            deferred.function,
            deferred.path
        );
    }
}

/// The lint sc-15946 shipped, still asserting exactly what it did, now through the registry.
#[test]
fn every_advanced_key_the_studio_can_emit_is_classified() {
    assert_builder_is_fully_classified(builder_for(AdvancedKeySource::StudioBuilder));
}

/// The Detail pass's own knobs are classified too (sc-15948).
///
/// `buildImageJobAdvanced` is not the only builder that fills `advanced`: the standalone
/// `image_detail` job has its own, and its keys land in the same untyped map and go through the
/// same allow-list. Before this lint existed, `cnScale` — half of what that UI exposes — was
/// unclassified and therefore silently dropped from every shared detail-refined image.
#[test]
fn every_advanced_key_the_detail_pass_can_emit_is_classified() {
    assert_builder_is_fully_classified(builder_for(AdvancedKeySource::DetailBuilder));
}

/// The character lane leaked `angleSet` — the knob that decides how many images exist (sc-15948).
///
/// `advanced.angleSet` is what makes the worker emit one image per view angle, and
/// `mode: "character_image"` posts to `/api/v1/image/jobs`, which embeds. Unclassified, it was
/// dropped, so a shared angle-set image replayed as a SINGLE image. Its neighbour
/// `keypointCollectionId` is a local id and is denied.
#[test]
fn every_advanced_key_the_character_lane_can_emit_is_classified() {
    for source in [
        AdvancedKeySource::CharacterBuilder,
        AdvancedKeySource::CharacterAngleExtras,
        AdvancedKeySource::CharacterPoseExtras,
    ] {
        assert_builder_is_fully_classified(builder_for(source));
    }
    assert_eq!(
        advanced_key_rule("angleSet").map(|rule| rule.disposition),
        Some(AdvancedDisposition::Allow),
        "`angleSet` decides how many images the job makes, so it must travel"
    );
    assert_eq!(
        advanced_key_rule("keypointCollectionId").map(|rule| rule.disposition),
        Some(AdvancedDisposition::Deny),
        "a Key Point Library collection id resolves to nothing on another install"
    );
}

/// The interleave lane BECAME an embedding lane in this story, so its keys are classified here.
///
/// `sensenova_jobs.rs` threads `workflow_source` into `ImagePlan`, which reaches
/// `write_image_asset` — so every shared interleaved document image carries whatever
/// `DocumentStudio.submit` put in `advanced`, and dropped whatever it did not classify. Making a
/// lane embed and not classifying its keys is the same bug as the `cnScale` one, on a lane this
/// story created.
#[test]
fn every_advanced_key_the_interleave_lane_can_emit_is_classified() {
    assert_builder_is_fully_classified(builder_for(AdvancedKeySource::InterleaveBuilder));
    for (key, disposition) in [
        ("systemMessage", AdvancedDisposition::Allow),
        ("imageGuidanceScale", AdvancedDisposition::Allow),
    ] {
        assert_eq!(
            advanced_key_rule(key).map(|rule| rule.disposition),
            Some(disposition),
            "`{key}` is authored generation intent on a lane that embeds"
        );
    }
}

/// The standalone upscale job posts NO `advanced` map — and that is now asserted, not assumed.
///
/// sc-15948 made that job embed (ISSUE 1), which puts `buildUpscaleJobBody` on the same footing as
/// the other three builders: the day it grows a knob, the lint demands a classification instead of
/// dropping it silently.
#[test]
fn the_standalone_upscale_builder_still_posts_no_advanced_map() {
    let builder = builder_for(AdvancedKeySource::UpscaleBuilder);
    assert_eq!(builder.shape, AdvancedBuilderShape::NoAdvancedMap);
    assert_builder_is_fully_classified(builder);
}

#[test]
fn the_key_extractor_reads_a_representative_builder() {
    // Pins the extractor itself against the exact JS shapes `imageJobAdvanced.js` uses:
    // shorthand keys, `key: value`, conditional spreads, a spread nested inside a spread,
    // an object VALUE (whose keys are not advanced keys), a call argument object (likewise),
    // and string literals that must not be mistaken for keys.
    let source = r#"
export function buildImageJobAdvanced(state) {
  const { resolution, sampler } = state;
  return {
    resolution,
    // A comment naming notAKey: 1
    /* block comment: alsoNotAKey: 2 */
    ...(sampler && sampler !== "default" ? { sampler } : {}),
    ...(steps !== "" ? { steps: Number(steps) } : {}),
    ...(flashAttn ? {} : { flashAttn: false }),
    ...(tier !== null
      ? {
          mlxQuantize: tierQuantize(tier),
          ...(tierExplicit ? { mlxQuantizeExplicit: true } : {}),
        }
      : {}),
    ...(pidTarget === "2k" ? { pidTarget: "2k" } : {}),
    ...(posePayload.length ? { poses: posePayload, faceRestore } : {}),
    ...(overlayId ? { controlWeights: { overlayId: controlOverlayId } } : {}),
    ...(sendStructured ? { structuredPrompt: buildRecipe({ intent: a, caption: b }) } : {}),
  };
}
"#;
    let keys = emitted_advanced_keys(source);
    let expected: BTreeSet<String> = [
        "resolution",
        "sampler",
        "steps",
        "flashAttn",
        "mlxQuantize",
        "mlxQuantizeExplicit",
        "pidTarget",
        "poses",
        "faceRestore",
        "controlWeights",
        "structuredPrompt",
    ]
    .iter()
    .map(|key| (*key).to_owned())
    .collect();
    assert_eq!(keys, expected);
}

/// A builder shaped the way the scanner cannot read, so the tests below can hand it one.
fn builder_with_return_body(body: &str) -> String {
    format!("export function buildImageJobAdvanced(state) {{\n  return {{\n{body}\n  }};\n}}\n")
}

#[test]
#[should_panic(expected = "cannot read")]
fn the_key_extractor_refuses_a_spread_of_a_call_expression() {
    // Verified against the real lint: adding this line plus a helper returning
    // `{ secretLocalPathKnob: state.weightsPath }` made the whole coverage lint PASS SILENTLY —
    // the `>= 25` floor and all five anchors still resolved while the new knob became invisible.
    emitted_advanced_keys(&builder_with_return_body(
        "    resolution,\n    ...buildFutureKnobs(state),",
    ));
}

#[test]
#[should_panic(expected = "cannot read")]
fn the_key_extractor_refuses_a_spread_of_a_bare_identifier() {
    // A parenthesis-less spread also used to leave a spread open forever, so the next nested
    // object VALUE would have been read as an emit scope — quiet corruption, not just a miss.
    emitted_advanced_keys(&builder_with_return_body(
        "    resolution,\n    ...defaults,\n    ...(a ? { controlWeights: { overlayId } } : {}),",
    ));
}

#[test]
#[should_panic(expected = "contains no object literal")]
fn the_key_extractor_refuses_a_spread_with_no_object_literal_in_it() {
    emitted_advanced_keys(&builder_with_return_body(
        "    resolution,\n    ...(buildFutureKnobs(state)),",
    ));
}

// ---------------------------------------------------------------------------
// The discovery scan's own unit tests: what it sees, and what it must not
// ---------------------------------------------------------------------------

/// Every signal `advanced_map_signals` reports for one snippet, as (kind, callee).
fn signal_shapes(source: &str) -> Vec<(AdvancedSignalKind, Option<String>)> {
    advanced_map_signals(&scannable(source))
        .into_iter()
        .map(|signal| (signal.kind, signal.called))
        .collect()
}

/// Indirection is a signal, not silence — the hole that let `trainingConfigSnapshot` stay green.
///
/// The first cut of this sweep only ever matched a literal `{` after `advanced:`, so every one of
/// these was invisible: the map was assembled in a helper, a variable or a ternary, and a ~20-key
/// builder sat in neither the registry nor the deferral list with the lint reporting 27/27.
#[test]
fn the_discovery_scan_sees_a_map_handed_in_from_somewhere_else() {
    for (source, called) in [
        (
            "const body = { advanced: probeExtrasMap(opts) };",
            Some("probeExtrasMap"),
        ),
        ("const body = { advanced: someVar };", None),
        (
            "const body = { advanced: cond ? { a: 1 } : { b: 2 } };",
            None,
        ),
        (
            "const advanced = compactObject({ probeKnob: 1 });",
            Some("compactObject"),
        ),
        ("const advanced = helpers.build(opts);", Some("build")),
    ] {
        assert_eq!(
            signal_shapes(source),
            vec![(AdvancedSignalKind::Indirect, called.map(str::to_owned))],
            "indirection went unreported for {source:?}, which is how a builder stays invisible"
        );
    }
}

/// A call site of a REGISTERED builder stays exempt, or every screen would need an entry.
#[test]
fn the_discovery_scan_lets_a_declared_builders_call_site_through() {
    let declared = declared_builder_functions();
    for source in [
        "const body = { advanced: buildImageJobAdvanced({ prompt }) };",
        "const body = { advanced: advanced.buildAdvanced(controller.advancedExtras) };",
    ] {
        let signals = advanced_map_signals(&scannable(source));
        assert_eq!(signals.len(), 1, "expected one signal for {source:?}");
        assert!(
            signals[0].calls_a_declared_builder("apps/web/src/imageJobRequest.js", &declared),
            "{source:?} calls a registered builder, so demanding a registry entry for the SCREEN \
             would say nothing about which keys exist"
        );
    }
    // ... and an unregistered helper of the same shape is NOT exempt.
    let signals = advanced_map_signals(&scannable("const b = { advanced: probeMap(o) };"));
    assert!(
        !signals[0].calls_a_declared_builder("apps/web/src/imageJobRequest.js", &declared),
        "an indirection through an unregistered helper must stand as a signal"
    );
}

/// A member READ is not a build, and flagging it fired on the training config's draft reader.
#[test]
fn the_discovery_scan_ignores_a_read_of_somebody_elses_advanced() {
    assert!(
        signal_shapes("const advanced = defaults.advanced ?? {};").is_empty(),
        "`defaults.advanced ?? {{}}` reads a saved config; the sweep must not demand a registry \
         entry for every screen that inspects one"
    );
}

/// A JSX prop is not a payload, and a build-blocking lint that says otherwise gets deleted.
///
/// `advanced` appears in 59 non-test files under `apps/web/src`. Failing the BUILD on
/// `<Child advanced={state.opts} />` with "classify every key it can emit" is advice that cannot be
/// followed, on work that has nothing to do with sharing.
#[test]
fn the_discovery_scan_ignores_a_jsx_prop() {
    for source in [
        "const row = <Child advanced={state.opts} />;",
        "const row = <Child title=\"a > b; not a tag\" advanced={{ open: false }} />;",
        "const row = <ProbeRow advanced = {renamed} label=\"spaced\" />;",
        "const row = <Outer>{items.map((item) => <Child advanced={item.opts} />)}</Outer>;",
    ] {
        assert!(
            signal_shapes(source).is_empty(),
            "{source:?} is a JSX prop, not a map this app builds"
        );
    }
}

/// A binding is not a build: a parameter default, and a destructuring rename.
#[test]
fn the_discovery_scan_ignores_a_binding_pattern() {
    for source in [
        "function Row({ advanced = {} }) { return advanced; }",
        "function Row({ advanced = {} }, ref) { return ref ?? advanced; }",
        "const Row = ({ advanced = {} }) => advanced;",
        "const { advanced = {} } = props;",
        "const { advanced: renamed } = props;",
    ] {
        assert!(
            signal_shapes(source).is_empty(),
            "{source:?} BINDS `advanced`, it does not build one"
        );
    }
}

/// A ternary's middle arm wears the same colon and builds nothing.
#[test]
fn the_discovery_scan_ignores_a_colon_that_is_not_a_key() {
    assert!(
        signal_shapes("const map = useSaved ? advanced : fallback;").is_empty(),
        "`cond ? advanced : fallback` is not an object key"
    );
}

/// Handing on a map that was already built is not building a second one.
///
/// `{ advanced: advanced }` is the long spelling of the `{ advanced }` shorthand three registered
/// builders use. Reporting it would fail the build on a rewrite that changes nothing, and it can
/// hide nothing: the map still has to be created somewhere, and creation is what this scan reads.
#[test]
fn the_discovery_scan_ignores_a_hand_on_of_the_same_map() {
    for source in [
        "const body = { mode, advanced: advanced };",
        "return { advancedExtras: advancedExtras };",
    ] {
        assert!(
            signal_shapes(source).is_empty(),
            "{source:?} hands on a map this scan already saw being built"
        );
    }
    // But a member read or a call on it is still indirection.
    assert_eq!(
        signal_shapes("const body = { advanced: advanced.buildAdvanced(extras) };"),
        vec![(
            AdvancedSignalKind::Indirect,
            Some("buildAdvanced".to_owned())
        )]
    );
}

/// The one shape nothing in the text can settle, so the failure has to offer the rename.
///
/// A plain local named `advanced` holding UI state is indistinguishable from a payload map — the
/// same `const advanced = { … }` that `buildEditJobBody` and `DocumentStudio.submit` use. Narrowing
/// the signal to "locals that visibly flow into a request" would let `const advanced = { probeKnob:
/// 1 }; body.advanced = advanced;` back through, so the signal stands and the message carries the
/// third branch instead.
#[test]
fn a_plain_local_still_signals_and_the_advice_offers_the_rename() {
    assert_eq!(
        signal_shapes("const advanced = { open: false };"),
        vec![(AdvancedSignalKind::AssignedLiteral, None)],
    );
    assert!(
        UNACCOUNTED_ADVANCED_ADVICE.contains("RENAME THE LOCAL"),
        "a local this lint cannot classify must be told the one fix that applies to it, or the \
         next author deletes the lint rather than diagnosing it"
    );
}

/// The literal shapes the sweep has always caught, pinned so a rewrite cannot drop one.
#[test]
fn the_discovery_scan_still_sees_every_literal_shape() {
    assert_eq!(
        signal_shapes("const body = { advanced: { steps: 4 } };"),
        vec![(AdvancedSignalKind::InlineLiteral, None)]
    );
    assert_eq!(
        signal_shapes("return { advancedExtras: { angleSet } };"),
        vec![(AdvancedSignalKind::ExtrasLiteral, None)]
    );
    assert_eq!(
        signal_shapes("const advanced = {};\nadvanced.steps = 4;"),
        vec![
            (AdvancedSignalKind::AssignedLiteral, None),
            (AdvancedSignalKind::PropertyAssignment, None)
        ]
    );
    // Never an equality test, and never a method call on the map.
    assert!(signal_shapes("if (advanced == null) return;").is_empty());
    assert!(signal_shapes("const n = advanced.steps;").is_empty());
}

#[test]
#[should_panic(expected = "a computed key")]
fn an_invisible_write_is_refused_through_any_object_path() {
    refuse_invisible_advanced_writes(
        &scannable("body.payload.advanced[\"probeKnob\"] = value;"),
        "a probe",
    );
}

#[test]
#[should_panic(expected = "Object.assign(body.payload.advanced, …)")]
fn an_object_assign_onto_a_nested_advanced_is_refused() {
    // `Object.assign(advanced` as a literal substring missed this: the target was nested.
    refuse_invisible_advanced_writes(
        &scannable("Object.assign(body.payload.advanced, { probeKnob: 1 });"),
        "a probe",
    );
}

#[test]
#[should_panic(expected = "`advanced.push(…)`")]
fn an_array_mutator_on_advanced_is_refused() {
    refuse_invisible_advanced_writes(&scannable("advanced.push(knob);"), "a probe");
}

/// The refusal must not fire on an `Object.assign` that has nothing to do with `advanced`.
#[test]
fn an_unrelated_object_assign_is_not_an_invisible_advanced_write() {
    for source in [
        "Object.assign(target, { a: 1 });",
        "const merged = Object.assign({}, base, extra);",
        "Object.assign(state.advancedOpen, { a: 1 });",
    ] {
        assert!(
            invisible_advanced_writes(&scannable(source)).is_empty(),
            "{source:?} does not write an `advanced` map"
        );
    }
}

/// A nested helper inside a declared span is not accounted for by that declaration.
#[test]
#[should_panic(expected = "builds a SECOND `advanced` map")]
fn a_second_map_inside_a_declared_builder_is_refused() {
    // Shaped like `DocumentStudio.submit`: the registered `AssignedObject` initializer, plus a
    // nested helper with its own literal that the arm never reads. Verified against the real lint:
    // inserting this into the real `submit` left 27/27 green before this refusal existed.
    emitted_keys(
        builder_for(AdvancedKeySource::InterleaveBuilder),
        r#"
async function submit(event) {
  const advanced = {};
  advanced.systemMessage = trimmedSystem;
  advanced.imageGuidanceScale = referenceGuidance;
  function probeNestedBody(probeKnob, probeSecret) {
    return { advanced: { probeKnob, probeSecret } };
  }
  return probeNestedBody(1, 2);
}
"#,
    );
}

// ---------------------------------------------------------------------------
// Leak tests
// ---------------------------------------------------------------------------

/// Every value in a built envelope, checked against an INDEPENDENTLY written path sniffer.
///
/// The seeds are the point: the guard only ever gets as strong as what this test throws at it,
/// and the first cut seeded only absolute paths — which is exactly the subset the first cut of
/// `is_path_shaped` already caught, so the two could agree while both being wrong. The relative,
/// traversing and drive-relative seeds below are the ones that used to travel verbatim, and the
/// top-level ones (`model`, `stylePreset`, `styleId`, `fitMode`, `mode`) cover the fields that
/// were copied with no check at all while `advanced.styleId` — literally the same value — was
/// guarded.
///
/// The deliberate PROSE exemption is not seeded here; it is pinned by
/// [`authored_prose_travels_verbatim_even_when_it_names_a_path`] so that each test asserts one
/// thing.
#[test]
fn no_value_in_a_built_envelope_is_path_shaped() {
    let mut payload = golden_payload();
    // Top-level request fields, copied straight onto the envelope. `mode` and `model` are
    // required strings, so a path-shaped one reduces to empty rather than vanishing.
    for (key, value) in [
        ("mode", "..\\..\\Users\\Michael"),
        ("model", "C:\\models\\evil"),
        ("stylePreset", "\\\\fileserver\\styles\\x"),
        ("styleId", "../../etc/passwd"),
        ("fitMode", "/etc/passwd"),
    ] {
        payload.insert(key.to_owned(), json!(value));
    }
    payload.insert(
        "upscale".to_owned(),
        json!({ "enabled": true, "factor": 2, "engine": "E:\\engines\\seedvr2" }),
    );
    payload.insert(
        "loras".to_owned(),
        json!([{
            "name": "Users\\Michael\\Desktop\\coast.safetensors",
            "weight": 0.65,
            "source": { "provider": "huggingface", "repo": "../../../etc/passwd" }
        }]),
    );

    let advanced = payload
        .get_mut("advanced")
        .and_then(Value::as_object_mut)
        .expect("advanced object");
    // Seeded on purpose: paths under allow-listed keys, under denied keys, under keys nobody
    // has classified, and nested inside an object smuggled under a scalar key.
    for (key, value) in [
        ("sampler", json!("C:\\Users\\Michael\\samplers\\euler.json")),
        ("scheduler", json!("/home/michael/schedules/beta")),
        ("controlScale", json!("file:///D:/maps/canny.png")),
        ("resolution", json!("~/renders/1024x1024")),
        (
            "steps",
            json!({ "path": "\\\\fileserver\\share\\steps.json" }),
        ),
        (
            "someFutureKnob",
            json!("E:\\models\\future\\weights.safetensors"),
        ),
        ("weightsPath", json!("/opt/sceneworks/weights.safetensors")),
        (
            "poses",
            json!([{ "id": "pose_1", "keypoints": "C:\\poses\\a.json" }]),
        ),
        // The escapes the first cut of the guard let through, one per family.
        ("guidanceMethod", json!("Users\\Michael\\Desktop\\cfg.json")),
        ("viewAngle", json!("..\\..\\Users\\Michael")),
        ("schedulerShift", json!("../../etc/passwd")),
        // A drive-RELATIVE path: a drive prefix with no separator after it.
        ("controlMode", json!("C:foo")),
        ("styleId", json!("c:secret\\noir")),
        ("faceRestore", json!("%USERPROFILE%\\restore.json")),
        ("textStyleGain", json!("~michael/gain.json")),
        // Percent-encoded `file://D:/x`.
        ("trueCfgScale", json!("file%3A%2F%2FD%3A%2Fx")),
        ("ipAdapterScale", json!("assets/images/michael/x.png")),
    ] {
        advanced.insert(key.to_owned(), value);
    }

    let share = build_workflow_share(&golden_asset(), &payload);
    let encoded = serde_json::to_value(&share).expect("serializes");

    let mut offenders = Vec::new();
    collect_strings(&encoded, String::from("$"), &mut offenders);
    for (pointer, value) in &offenders {
        // `producer.url` is the ONE URL the envelope deliberately carries. It names the
        // software's repository, never this installation.
        if pointer == "$.producer.url" {
            continue;
        }
        assert!(
            !looks_like_a_path(value),
            "{pointer} = {value:?} is path-shaped and must never reach a shared image"
        );
    }

    // And nothing from the seeded values survived as a substring anywhere.
    let text = serde_json::to_string(&encoded).expect("serializes");
    for fragment in [
        "Michael",
        "C:\\\\",
        "c:secret",
        "C:foo",
        "/home/",
        "file://",
        "file%3A",
        "fileserver",
        "E:\\\\",
        "/opt/",
        "safetensors",
        "USERPROFILE",
        "etc/passwd",
        "..",
        "~",
    ] {
        assert!(
            !text.contains(fragment),
            "{fragment:?} leaked into the envelope: {text}"
        );
    }
}

/// The one deliberate exception to "every filesystem path without exception", made explicit.
///
/// `stylePrompt` and the structured prompt's `intent` / `runtimePrompt` are the same class as
/// the top-level `prompt`, which the story puts IN: they are what the user typed. Silently
/// mangling authored text because it mentions a directory would be worse than the leak it
/// prevents — the user wrote it and can see it before sharing. That decision was previously
/// implicit in a `PROSE_KEYS` constant no test seeded; this pins it in both directions.
#[test]
fn authored_prose_travels_verbatim_even_when_it_names_a_path() {
    const STYLE_PROMPT: &str = "C:\\Users\\Michael\\Desktop\\secret_project\\brief.txt";
    const INTENT: &str = "/home/michael/clients/acme/nda.md";
    const RUNTIME_PROMPT: &str = "rendered from ..\\..\\briefs\\acme.json";
    const PROMPT: &str = "a lighthouse, per D:\\briefs\\fog.md";
    const NEGATIVE_PROMPT: &str = "no text, nothing like \\\\fileserver\\rejects\\list.txt";

    let mut payload = golden_payload();
    payload.insert("prompt".to_owned(), json!(PROMPT));
    payload.insert("negativePrompt".to_owned(), json!(NEGATIVE_PROMPT));
    let advanced = payload
        .get_mut("advanced")
        .and_then(Value::as_object_mut)
        .expect("advanced object");
    advanced.insert("stylePrompt".to_owned(), json!(STYLE_PROMPT));
    advanced.insert(
        "structuredPrompt".to_owned(),
        json!({ "intent": INTENT, "runtimePrompt": RUNTIME_PROMPT }),
    );

    let share = build_workflow_share(&golden_asset(), &payload);
    assert_eq!(share.prompt, PROMPT);
    assert_eq!(share.negative_prompt, NEGATIVE_PROMPT);
    assert_eq!(share.advanced["stylePrompt"], json!(STYLE_PROMPT));
    let recipe = share.advanced["structuredPrompt"]
        .as_object()
        .expect("structuredPrompt object");
    assert_eq!(recipe["intent"], json!(INTENT));
    assert_eq!(recipe["runtimePrompt"], json!(RUNTIME_PROMPT));

    // The exemption is per key, not a hole in the guard: a NON-prose neighbour with the same
    // text is still dropped, and the exempt pointers are exactly these five.
    let mut offenders = Vec::new();
    collect_strings(
        &serde_json::to_value(&share).expect("serializes"),
        String::from("$"),
        &mut offenders,
    );
    let path_shaped: BTreeSet<&str> = offenders
        .iter()
        .filter(|(_, value)| looks_like_a_path(value))
        .map(|(pointer, _)| pointer.as_str())
        .collect();
    assert_eq!(
        path_shaped,
        [
            "$.prompt",
            "$.negativePrompt",
            "$.advanced.stylePrompt",
            "$.advanced.structuredPrompt.intent",
            "$.advanced.structuredPrompt.runtimePrompt",
        ]
        .into_iter()
        .collect::<BTreeSet<&str>>(),
        "only the authored prose fields may carry a path"
    );
}

/// The false-positive half, pinned against REAL data rather than a hand-picked corpus.
///
/// The relative-tree tally read `"PiD 1.5 Decoder (FLUX.1 / Boogu / Chroma / Z-Image)"` — a
/// display name shipped in `config/manifests/builtin.models.jsonc` — as a four-deep path,
/// because it counted a ` / ` list separator as a path separator. Those display names do not
/// themselves travel (the envelope carries the model SLUG), but `loras[].name` is exactly this
/// class of free human label and DOES travel, and a dropped LoRA name is silent: the user chose
/// those words and nothing tells them the name went missing. So the manifest's own labels stand
/// in as the corpus of what people actually type.
#[test]
fn shipped_display_labels_that_list_with_slashes_are_not_path_shaped() {
    let source = BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .map(|(_, contents)| *contents)
        .expect("the builtin model manifest is embedded");
    let manifest: Value = serde_json::from_str(&strip_jsonc_comments(source))
        .expect("builtin.models.jsonc parses as JSON");

    let mut strings = Vec::new();
    collect_strings(&manifest, String::from("$"), &mut strings);
    let listed: Vec<&String> = strings
        .iter()
        .filter(|(pointer, _)| {
            [".name", ".label", ".displayName"]
                .iter()
                .any(|suffix| pointer.ends_with(suffix))
        })
        .map(|(_, value)| value)
        .filter(|value| value.contains(" / "))
        .collect();

    assert!(
        listed.len() >= 5,
        "the manifest no longer carries the ` / ` display labels this test pins ({} found). If \
         they were renamed, retarget this test at the labels that replaced them — do not delete \
         it: it is the only place the guard meets real human labels.",
        listed.len()
    );
    for label in listed {
        assert!(
            !is_path_shaped(label),
            "the shipped display label {label:?} reads as a filesystem path, so the same shape in \
             a `loras[].name` would be dropped from the user's own share with no signal"
        );
    }
}

/// The shipped guard must never be WEAKER than the sniffer this file asserts with.
///
/// That inversion is exactly how the first cut shipped: [`looks_like_a_path`] already treated
/// any backslash as a path while `is_path_shaped` tested only the FIRST character, so the two
/// disagreed on every relative Windows path — and the property test passed anyway, because its
/// seeds happened to be precisely the subset both agreed on. A seed corpus can only ever fail to
/// notice that; the relationship itself has to be asserted, so it is asserted here.
///
/// The other direction is deliberately NOT asserted: the shipped guard is allowed to be
/// stricter (it also catches relative POSIX trees and percent-encoded `file://`), because the
/// cost of a false positive is one dropped label and the cost of a false negative is a username
/// inside every copy of a shared image.
#[test]
fn the_shipped_path_guard_is_never_weaker_than_the_independent_sniffer() {
    for value in [
        // Paths, one per family.
        "/home/michael/x.png",
        "C:\\Users\\Michael\\x.png",
        "Users\\Michael\\Desktop\\secret.png",
        "models\\weights\\x.safetensors",
        "..\\..\\Users\\Michael",
        "../../etc/passwd",
        "./local/thing",
        "C:foo",
        "c:secret\\file",
        "%USERPROFILE%\\x",
        "assets/images/x.png",
        "~/models/x.png",
        "~michael/x",
        "\\\\fileserver\\share\\x.png",
        "file:///D:/x.png",
        "file%3A%2F%2FD%3A%2Fx",
        "engine loaded from D:/models/x",
        // Legitimate values, which neither may flag.
        "euler",
        "dpmpp_2m",
        "beta",
        "cfg_pp",
        "1024x1024",
        "2k",
        "acme/mira",
        "acme/foggy-coast",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "https://github.com/SceneWorks/SceneWorks",
        "text_to_image",
        "krea_2_turbo",
        "seedvr2",
        "noir_bloom",
        "cinematic",
        "crop",
        "canny",
        "",
    ] {
        if looks_like_a_path(value) {
            assert!(
                is_path_shaped(value),
                "the shipped `is_path_shaped` misses {value:?}, which this file's independent \
                 sniffer flags — so the leak tests here are checking a guard weaker than their \
                 own assertion and a bug in it can hide behind itself. Strengthen \
                 `workflow_share::is_path_shaped`; do NOT weaken `looks_like_a_path`."
            );
        }
    }
}

/// The two live leaks sc-15948's review found, asserted as BEHAVIOUR rather than as coverage.
///
/// The lint tests above would have caught these at build time; this catches them at the seam, so
/// the fix cannot regress by someone re-tagging a rule while leaving it denied. Both lanes post to
/// endpoints that embed, so each of these keys either travels inside a shared PNG or is lost:
///
/// * `angleSet` is what makes the worker emit one image per view angle. Dropped, a shared angle-set
///   image replays as a SINGLE image.
/// * `imageGuidanceScale` is the authored reference-guidance strength for an interleaved document,
///   and `systemMessage` is the system prompt the user typed. Dropped, a shared interleaved image
///   loses both.
/// * `keypointCollectionId` is a local Key Point Library id and must NOT travel.
#[test]
fn the_character_and_interleave_lane_knobs_survive_the_allow_list() {
    let mut payload = golden_payload();
    let advanced = payload
        .get_mut("advanced")
        .and_then(Value::as_object_mut)
        .expect("advanced object");
    advanced.insert("angleSet".to_owned(), json!(true));
    advanced.insert("keypointCollectionId".to_owned(), json!("kpc_7f31"));
    advanced.insert("imageGuidanceScale".to_owned(), json!(1.8));
    advanced.insert(
        "systemMessage".to_owned(),
        json!("Keep every panel on the same character."),
    );

    let share = build_workflow_share(&golden_asset(), &payload);
    assert_eq!(share.advanced.get("angleSet"), Some(&json!(true)));
    assert_eq!(share.advanced.get("imageGuidanceScale"), Some(&json!(1.8)));
    assert_eq!(
        share.advanced.get("systemMessage"),
        Some(&json!("Keep every panel on the same character."))
    );
    assert!(
        !share.advanced.contains_key("keypointCollectionId"),
        "a local Key Point Library id must not travel"
    );
    assert!(
        !serde_json::to_string(&share)
            .expect("serializes")
            .contains("kpc_7f31"),
        "the collection id leaked into the envelope"
    );
}

#[test]
fn input_ids_become_shape_descriptors_with_zero_id_leakage() {
    let mut payload = golden_payload();
    payload.insert("referenceAssetId".to_owned(), json!("asset_ref_solo"));
    let share = build_workflow_share(&golden_asset(), &payload);

    let kinds: Vec<(&str, u32, Option<&str>)> = share
        .inputs
        .iter()
        .map(|input| {
            (
                input.kind.as_str(),
                input.count,
                input.control_mode.as_deref(),
            )
        })
        .collect();
    assert_eq!(
        kinds,
        vec![
            ("source", 1, None),
            ("reference", 3, None),
            ("mask", 1, None),
            ("control", 1, Some("canny")),
        ]
    );

    let text = serde_json::to_string(&share).expect("serializes");
    for id in [
        "asset_source_1",
        "asset_ref_1",
        "asset_ref_2",
        "asset_ref_solo",
        "asset_mask_1",
        "asset_control_1",
        "asset_9f2c",
        "project_7a10",
        "genset_31bb",
        "job_5c8e",
        "character_c001",
        "look_l001",
        "lora_1f0d",
        "overlay_7",
        "preset_local_1",
        "Michael's Unreleased Film",
        "2026-07-29T13:04:11Z",
        "nvfp4",
        "krea_2_turbo\\\\", // the install path's model dir, not the catalog slug
    ] {
        assert!(!text.contains(id), "{id} leaked into the envelope: {text}");
    }

    // The catalog slug itself IS in — it names a model, not an installation.
    assert_eq!(share.model, "krea_2_turbo");
}

#[test]
fn denied_top_level_request_fields_never_travel() {
    let share = build_workflow_share(&golden_asset(), &golden_payload());
    let encoded = serde_json::to_value(&share).expect("serializes");
    let object = encoded.as_object().expect("envelope is an object");
    for denied in [
        "projectId",
        "projectName",
        "jobId",
        "assetId",
        "generationSetId",
        "characterId",
        "characterLookId",
        "requestedGpu",
        "quantTier",
        "tierExplicit",
        "modelManifestEntry",
        "seeds",
        "createdAt",
        "recipePresetId",
    ] {
        assert!(
            !object.contains_key(denied),
            "`{denied}` must not be a top-level envelope field"
        );
    }
    for denied in [
        "quantTier",
        "mlxQuantize",
        "flashAttn",
        "controlWeights",
        "controlImage",
    ] {
        assert!(
            !share.advanced.contains_key(denied),
            "`advanced.{denied}` must not travel"
        );
    }
}

// ---------------------------------------------------------------------------
// Collection bounds (sc-15949 review)
// ---------------------------------------------------------------------------

/// `workflow_share::MAX_SHARE_PHASES` against the two validators it claims to be.
///
/// `sceneworks-worker` depends on `sceneworks-core`, so the constant cannot be imported and the
/// derivation would otherwise be a comment nobody checks. Read from source instead, the same
/// technique the `ADVANCED_BUILDERS` lint uses on `apps/web/src`: if either validator's cap moves,
/// this fails and the envelope's cap becomes a decision rather than a stale copy.
#[test]
fn the_phase_cap_matches_the_multi_phase_validators() {
    let worker = read_repo_file("crates/sceneworks-worker/src/image_jobs/krea_multiphase.rs");
    assert!(
        worker.contains("const MAX_MULTIPHASE_PHASES: usize = 8;"),
        "the worker's multi-phase cap moved; `workflow_share::MAX_SHARE_PHASES` (8) has to move \
         with it or the envelope will drop a phase set the worker would have run"
    );
    let web = read_repo_file("apps/web/src/imageMultiPhase.js");
    assert!(
        web.contains("export const MULTIPHASE_MAX_PHASES = 8;"),
        "the web's multi-phase cap moved; see `workflow_share::MAX_SHARE_PHASES`"
    );
}

/// The bounds against REAL requests, not fixtures written to pass them.
///
/// A cap is only correct if nothing legitimate touches it, and the failure mode of getting that
/// wrong is silent: the envelope simply arrives without its LoRAs. So the three widest shapes the
/// studio actually submits are run through the builder and every collection is asserted to have
/// survived whole.
#[test]
fn no_real_request_is_truncated_by_the_collection_bounds() {
    // 1. The golden fixture — an edit with a source, two references, a mask, a control map and a
    //    LoRA, i.e. all four input kinds at once.
    let golden = build_workflow_share(&golden_asset(), &golden_payload());
    assert_eq!(golden.loras.len(), 1, "{:?}", golden.loras);
    assert_eq!(golden.inputs.len(), 4, "{:?}", golden.inputs);
    // And the fixture ON DISK, which is what a reader will actually be handed.
    let parsed = parse_workflow_share(&load_golden_fixture()).expect("the golden fixture parses");
    assert_eq!(parsed.loras.len(), 1);
    assert_eq!(parsed.inputs.len(), 4);

    // 2. A Krea multi-phase recipe at the validators' own ceiling: 8 phases, each naming phase
    //    LoRAs by index into a full 5-LoRA stack.
    let mut krea = golden_payload();
    krea.insert("model".to_owned(), json!("krea_2_turbo"));
    krea.insert(
        "loras".to_owned(),
        json!((0..5)
            .map(|index| json!({
                "id": format!("lora_{index}"),
                "name": format!("Foggy Coast {index}"),
                "weight": 0.6,
                "source": { "provider": "huggingface", "repo": format!("acme/foggy-{index}") }
            }))
            .collect::<Vec<Value>>()),
    );
    krea.insert(
        "advanced".to_owned(),
        json!({
            "steps": 28,
            "phases": (0..8)
                .map(|index| json!({
                    "steps": index + 2,
                    "guidance": 3.5,
                    "loras": (0..5).map(|lora| json!({ "index": lora, "weight": 0.5 })).collect::<Vec<Value>>()
                }))
                .collect::<Vec<Value>>(),
        }),
    );
    let krea_share = build_workflow_share(&golden_asset(), &krea);
    assert_eq!(krea_share.loras.len(), 5, "{:?}", krea_share.loras);
    let phases = krea_share.advanced["phases"].as_array().expect("phases");
    assert_eq!(phases.len(), 8);
    for phase in phases {
        assert_eq!(
            phase["loras"].as_array().expect("phase loras").len(),
            5,
            "a phase's LoRA indices point into the job's own 5-LoRA stack and must all survive"
        );
    }

    // 3. A multi-reference FLUX.2 request: many reference images, which the builder records as ONE
    //    input entry with a count — the reason `MAX_SHARE_INPUTS` can be the number of kinds.
    let mut flux = golden_payload();
    flux.insert("model".to_owned(), json!("flux2_dev"));
    flux.insert("mode".to_owned(), json!("edit_image"));
    flux.remove("maskAssetId");
    flux.insert(
        "referenceAssetIds".to_owned(),
        json!((0..10)
            .map(|index| json!(format!("asset_ref_{index}")))
            .collect::<Vec<Value>>()),
    );
    let flux_share = build_workflow_share(&golden_asset(), &flux);
    let references = flux_share
        .inputs
        .iter()
        .find(|input| input.kind == "reference")
        .expect("the reference inputs travel");
    assert_eq!(references.count, 10, "the COUNT carries the breadth");
    assert!(flux_share.inputs.len() <= 4, "{:?}", flux_share.inputs);

    // Every one of the three round-trips through the reader unchanged, so the bounds cannot be
    // asymmetric between the direction that wrote the file and the direction that reads it.
    for share in [golden, krea_share, flux_share] {
        let envelope = serde_json::to_value(&share).expect("serializes");
        assert_eq!(
            parse_workflow_share(&envelope).expect("parses"),
            share,
            "a real envelope must survive its own reader"
        );
    }
}

/// Nothing legitimate is anywhere near the recording ceiling.
///
/// A ceiling is only correct if no real envelope reaches it, and getting that wrong is silent in the
/// worst way: the image simply arrives with no recipe. So the four widest real shapes are measured
/// against it in ABSOLUTE bytes, and the margin is printed rather than described.
#[test]
fn no_legitimate_envelope_comes_close_to_the_recording_ceiling() {
    let mut widest = 0;
    let mut measure = |label: &str, share: &WorkflowShare| {
        let bytes = serde_json::to_string(share).expect("serializes").len();
        println!(
            "sc-15949: {label} is {bytes} bytes, {:.0}x under the {WORKFLOW_SHARE_MAX_BYTES} byte \
             ceiling",
            WORKFLOW_SHARE_MAX_BYTES as f64 / bytes as f64
        );
        assert!(
            bytes * 4 < WORKFLOW_SHARE_MAX_BYTES,
            "{label} is {bytes} bytes, within 4x of the {WORKFLOW_SHARE_MAX_BYTES} byte ceiling — \
             the ceiling has stopped having real headroom over what this app produces"
        );
        assert!(
            share.omitted.is_empty(),
            "{label} lost something: {share:?}"
        );
        widest = widest.max(bytes);
    };

    // 1. The golden fixture, both as the builder makes it and as it sits on disk.
    measure(
        "the golden envelope",
        &build_workflow_share(&golden_asset(), &golden_payload()),
    );
    let parsed = parse_workflow_share(&load_golden_fixture()).expect("the golden fixture parses");
    measure("the golden fixture on disk", &parsed);

    // 2. A Krea multi-phase recipe at the validators' own ceiling: 8 phases, a full 5-LoRA stack,
    //    every phase naming every LoRA.
    let mut krea = golden_payload();
    krea.insert("model".to_owned(), json!("krea_2_turbo"));
    krea.insert(
        "loras".to_owned(),
        json!((0..5)
            .map(|index| json!({
                "id": format!("lora_{index}"),
                "name": format!("Foggy Coast {index}"),
                "weight": 0.6,
                "source": { "provider": "huggingface", "repo": format!("acme/foggy-{index}") }
            }))
            .collect::<Vec<Value>>()),
    );
    krea.insert(
        "advanced".to_owned(),
        json!({
            "steps": 28,
            "guidanceScale": 3.5,
            "sampler": "euler",
            "scheduler": "beta",
            "phases": (0..8)
                .map(|index| json!({
                    "steps": index + 2,
                    "guidance": 3.5,
                    "loras": (0..5).map(|lora| json!({ "index": lora, "weight": 0.5 })).collect::<Vec<Value>>()
                }))
                .collect::<Vec<Value>>(),
        }),
    );
    measure(
        "a Krea multi-phase recipe at the validators' ceiling",
        &build_workflow_share(&golden_asset(), &krea),
    );

    // 3. A multi-reference FLUX.2 request: ten reference images, all four input kinds in play.
    let mut flux = golden_payload();
    flux.insert("model".to_owned(), json!("flux2_dev"));
    flux.insert(
        "referenceAssetIds".to_owned(),
        json!((0..10)
            .map(|index| json!(format!("asset_ref_{index}")))
            .collect::<Vec<Value>>()),
    );
    measure(
        "a multi-reference FLUX.2 request",
        &build_workflow_share(&golden_asset(), &flux),
    );

    // 4. A realistic long non-Latin prompt — the case a BYTE bound on prose could have truncated
    //    where a character bound would not. 3 bytes per character, at a length people really type.
    let cjk = "霧の中の灯台、シネマティック、レンズに雨、35mmフィルムで撮影、\
               ボリューメトリックライト、夜明けの海岸線、遠くの汽笛、"
        .repeat(12);
    assert!(
        cjk.chars().count() > 700 && cjk.len() > 2_000,
        "the fixture must be a long non-Latin prompt: {} chars, {} bytes",
        cjk.chars().count(),
        cjk.len()
    );
    let mut japanese = golden_payload();
    japanese.insert("prompt".to_owned(), json!(cjk.clone()));
    japanese.insert("negativePrompt".to_owned(), json!(cjk.clone()));
    let advanced = japanese
        .get_mut("advanced")
        .and_then(Value::as_object_mut)
        .expect("advanced object");
    advanced.insert("stylePrompt".to_owned(), json!(cjk.clone()));
    advanced.insert("systemMessage".to_owned(), json!(cjk.clone()));
    let japanese_share = build_workflow_share(&golden_asset(), &japanese);
    measure("a long CJK prompt in every prose slot", &japanese_share);
    // Character for character, not "roughly the same length": a byte bound must not have cut it.
    assert_eq!(japanese_share.prompt, cjk);
    assert_eq!(japanese_share.negative_prompt, cjk);
    assert_eq!(japanese_share.advanced["stylePrompt"], json!(cjk));
    assert_eq!(japanese_share.advanced["systemMessage"], json!(cjk));

    // Every one of them round-trips through the reader unchanged, so the ceiling and the marker are
    // not asymmetric between the direction that wrote the file and the direction that reads it.
    for share in [
        build_workflow_share(&golden_asset(), &golden_payload()),
        build_workflow_share(&golden_asset(), &krea),
        build_workflow_share(&golden_asset(), &flux),
        japanese_share,
    ] {
        let envelope = serde_json::to_value(&share).expect("serializes");
        assert_eq!(
            parse_workflow_share(&envelope).expect("a real envelope survives its own reader"),
            share
        );
    }
    println!("sc-15949: the widest legitimate envelope measured here is {widest} bytes");
}

/// Every write seam goes through the GATED builder.
///
/// `build_workflow_share_from` does not run the recording ceiling — `embeddable_workflow_share` is
/// the form that does, and it is the one the worker must call. A seam that called the ungated builder
/// would embed a chunk our own reader then refuses with `TooLarge`, which is the writer-and-reader
/// drift every guard in `workflow_share.rs` is arranged to prevent. Pinned as source structure
/// because `sceneworks-worker` depends on this crate and cannot be inspected any other way.
#[test]
fn every_write_seam_embeds_through_the_gated_builder() {
    for relative in [
        "crates/sceneworks-worker/src/image_jobs.rs",
        "crates/sceneworks-worker/src/image_jobs/detail.rs",
        "crates/sceneworks-worker/src/upscale_jobs.rs",
        "crates/sceneworks-worker/src/single_child_asset.rs",
    ] {
        let source = read_repo_file(relative);
        // Only the code: the prose in these files names the function it does not call.
        let code: String = source
            .lines()
            .map(str::trim_start)
            .filter(|line| !line.starts_with("//") && !line.starts_with("///"))
            .collect::<Vec<&str>>()
            .join("\n");
        assert!(
            !code.contains("build_workflow_share_from("),
            "{relative} builds an embedded envelope with `build_workflow_share_from`, which does \
             not run the recording ceiling. Call `embeddable_workflow_share` and treat `None` as \
             `write the file exactly as today`."
        );
    }
    let worker = read_repo_file("crates/sceneworks-worker/src/image_jobs.rs");
    assert!(
        worker.contains("embeddable_workflow_share("),
        "the worker stopped building envelopes through the gated builder"
    );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn collect_strings(value: &Value, pointer: String, out: &mut Vec<(String, String)>) {
    match value {
        Value::String(text) => out.push((pointer, text.clone())),
        Value::Array(values) => {
            for (index, item) in values.iter().enumerate() {
                collect_strings(item, format!("{pointer}[{index}]"), out);
            }
        }
        Value::Object(map) => {
            for (key, item) in map {
                collect_strings(item, format!("{pointer}.{key}"), out);
            }
        }
        _ => {}
    }
}

/// An independent path sniffer for the assertion side, deliberately written without reusing
/// `workflow_share::is_path_shaped` so a bug in that function cannot hide behind itself.
fn looks_like_a_path(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    if lower.contains("file://") {
        return true;
    }
    if value.starts_with('/') || value.starts_with('\\') || value.starts_with("~/") {
        return true;
    }
    if value.contains('\\') {
        return true;
    }
    // `X:/…` or `X:\…` where the letter is not part of a URL scheme.
    lower
        .split(|character: char| !character.is_ascii_alphanumeric() && character != ':')
        .any(|token| {
            let bytes = token.as_bytes();
            bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':'
        })
}

// ---------------------------------------------------------------------------
// The JS key extractor
// ---------------------------------------------------------------------------

/// Comments removed and every non-ASCII character replaced by a space.
///
/// The ASCII normalization is not cosmetic: the discovery scan compares byte offsets of `advanced`
/// signals against byte offsets of function-body spans, and every builder file carries em-dashes in
/// its comments and prose in its string literals. Normalizing first makes byte offsets and
/// character offsets the same number, so a `&str` slice can never land mid-codepoint and the two
/// halves of the scan cannot silently disagree. Identifiers are ASCII, so nothing the extractors
/// read is affected.
fn scannable(source: &str) -> String {
    strip_comments(source)
        .chars()
        .map(|character| if character.is_ascii() { character } else { ' ' })
        .collect()
}

/// Every web source file the discovery scan reads, repo-relative with forward slashes.
///
/// Test scaffolding is excluded by name: `main.testSupport.jsx` builds request bodies with
/// `advanced` maps on purpose (they are fixtures for the tests, not payloads a user can send), and
/// `*.test.js(x)` do the same.
fn web_source_files() -> Vec<String> {
    let root = repo_root().join("apps").join("web").join("src");
    let mut out = Vec::new();
    let mut stack = vec![root.clone()];
    while let Some(dir) = stack.pop() {
        let entries = fs::read_dir(&dir)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", dir.display()));
        for entry in entries {
            let path = entry.expect("dir entry").path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default()
                .to_owned();
            if !(name.ends_with(".js") || name.ends_with(".jsx"))
                || name.contains(".test.")
                || name.contains(".testSupport.")
            {
                continue;
            }
            let relative = path
                .strip_prefix(repo_root())
                .expect("under the repo root")
                .to_string_lossy()
                .replace('\\', "/");
            out.push(relative);
        }
    }
    out.sort_unstable();
    out
}

/// What a web file did with the name `advanced`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AdvancedSignalKind {
    /// `advanced: { … }` — a map written inline into a request body.
    InlineLiteral,
    /// `advancedExtras: { … }` — a character-lane controller's contribution.
    ExtrasLiteral,
    /// `advanced = { … }` — the initializer half of an assignment-style builder.
    AssignedLiteral,
    /// `advanced.key = …` — the assignment half.
    PropertyAssignment,
    /// `advanced: buildFoo(state)`, `advanced: someVar`, `advanced = compactObject({ … })` — the
    /// map exists but is assembled somewhere this scan cannot read.
    Indirect,
}

impl AdvancedSignalKind {
    fn describe(self) -> &'static str {
        match self {
            Self::InlineLiteral => "an inline object literal",
            Self::ExtrasLiteral => "an inline `advancedExtras` object literal",
            Self::AssignedLiteral => "an assigned object literal",
            Self::PropertyAssignment => "a property assignment",
            Self::Indirect => "an expression this lint cannot read",
        }
    }
}

/// One place a web file builds an `advanced` map.
struct AdvancedSignal {
    /// Offset into the [`scannable`] text of the `advanced` / `advancedExtras` identifier.
    offset: usize,
    /// What was matched, for the failure message.
    kind: AdvancedSignalKind,
    /// For [`AdvancedSignalKind::Indirect`], the callee the map comes from:
    /// `advanced: buildImageJobAdvanced({ … })` → `buildImageJobAdvanced`,
    /// `advanced: advanced.buildAdvanced(extras)` → `buildAdvanced`. `None` when the expression is
    /// not a call at all (`advanced: someVar`), which can never be a builder call site.
    called: Option<String>,
}

impl AdvancedSignal {
    /// True when this is indirection through a builder the registry already accounts for.
    ///
    /// `advanced: buildImageJobAdvanced({ … })` and `advanced: advanced.buildAdvanced(extras)` are
    /// CALL SITES of a registered builder: demanding a registry entry for every screen that
    /// submits a job would say nothing about which keys exist. The difference from before is that
    /// the callee now has to be NAMED somewhere — an indirection through an unregistered helper is
    /// a signal, not silence.
    fn calls_a_declared_builder(&self, path: &str, declared: &BTreeSet<&str>) -> bool {
        let Some(called) = self.called.as_deref() else {
            return false;
        };
        declared.contains(called)
            || ADVANCED_INDIRECTION_CALL_HEADS
                .iter()
                .any(|(exempt_path, head, _)| *exempt_path == path && *head == called)
    }

    /// The indirection-specific half of the failure message.
    fn indirection_advice(&self) -> String {
        if self.kind != AdvancedSignalKind::Indirect {
            return String::new();
        }
        match self.called.as_deref() {
            Some(called) => format!(
                "\nThe map comes from `{called}(…)`, so its keys are assembled where this scan \
                 never looks. Either build it inline (`advanced: {{ … }}`) inside a builder that \
                 is declared, or declare `{called}` itself."
            ),
            None => "\nThe map comes from a bare expression (a variable, a member read, a \
                     ternary), so its keys are assembled where this scan never looks. Build it \
                     inline (`advanced: { … }`) inside a builder that is declared, or move the \
                     assembly into a function and declare that."
                .to_owned(),
        }
    }
}

/// Every place `stripped` builds an `advanced` map, or takes one from somewhere unreadable.
///
/// Five shapes; see [`AdvancedSignalKind`]. The first four are literal writes. The fifth,
/// [`AdvancedSignalKind::Indirect`], is what makes the sweep a class-closer rather than a
/// pattern-matcher: any indirection is enough to hide a map, and this scan reports it so the
/// caller can insist the callee be named.
///
/// # Not every `advanced` is a payload
///
/// Three ordinary React shapes carry the name and build nothing, and a build-blocking lint that
/// misfires on them gets deleted rather than diagnosed:
///
/// * a JSX prop — `<Child advanced={state.opts} />`;
/// * a destructuring default or rename — `function Row({ advanced = {} })`,
///   `const { advanced: opts } = props`;
/// * (not suppressible) a plain local named `advanced` that holds something else. Nothing in the
///   text distinguishes it from a payload map, so it still fails, and the failure message offers
///   the only honest fix: rename the local.
///
/// A computed write (`advanced["key"] = …`) or an `Object.assign(x.advanced, …)` is invisible here
/// no matter where it sits, so [`refuse_invisible_advanced_writes`] refuses those over every file
/// the sweep reads rather than reading past them.
fn advanced_map_signals(stripped: &str) -> Vec<AdvancedSignal> {
    let bytes = stripped.as_bytes();
    let mask = string_literal_mask(bytes);
    let mut out = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'"' | b'\'' | b'`' => {
                index = skip_string_ascii(bytes, index);
                continue;
            }
            byte if is_identifier_start(byte as char) => {
                let start = index;
                while index < bytes.len() && is_identifier_char(bytes[index] as char) {
                    index += 1;
                }
                let word = &stripped[start..index];
                if word != "advanced" && word != "advancedExtras" {
                    continue;
                }
                let extras = word == "advancedExtras";
                let lookahead = skip_ascii_whitespace(bytes, index);
                let mut called = None;
                let kind = match bytes.get(lookahead) {
                    // `advanced: {` / `advancedExtras: {`, or `advanced: <expression>`. Only in KEY
                    // position: `cond ? advanced : fallback` wears the same colon and builds
                    // nothing, and `const { advanced: opts } = props` is a rename, not a build.
                    Some(b':')
                        if is_object_key_position(bytes, &mask, start)
                            && !is_binding_pattern_position(bytes, &mask, start) =>
                    {
                        let after = skip_ascii_whitespace(bytes, lookahead + 1);
                        if bytes.get(after) == Some(&b'{') {
                            Some(if extras {
                                AdvancedSignalKind::ExtrasLiteral
                            } else {
                                AdvancedSignalKind::InlineLiteral
                            })
                        } else {
                            called = call_head(bytes, stripped, after);
                            (called.is_some() || !is_pass_through(bytes, stripped, after))
                                .then_some(AdvancedSignalKind::Indirect)
                        }
                    }
                    // `advanced = {` or `advanced = someBuilder(…)`, but never `advanced == …`, a
                    // JSX prop, or a parameter default.
                    Some(b'=')
                        if bytes.get(lookahead + 1) != Some(&b'=')
                            && !is_jsx_attribute_position(bytes, &mask, start)
                            && !is_binding_pattern_position(bytes, &mask, start) =>
                    {
                        let after = skip_ascii_whitespace(bytes, lookahead + 1);
                        if bytes.get(after) == Some(&b'{') {
                            Some(AdvancedSignalKind::AssignedLiteral)
                        } else {
                            // A CALL can build a map; `defaults.advanced ?? {}` and friends are
                            // reads of someone else's, and flagging those fires on every screen
                            // that merely inspects a saved config.
                            called = call_head(bytes, stripped, after);
                            called.is_some().then_some(AdvancedSignalKind::Indirect)
                        }
                    }
                    // `advanced.key = …`, but never `advanced.key ==` or `advanced.key(…)`.
                    Some(b'.') if word == "advanced" => {
                        let key_start = lookahead + 1;
                        let mut key_end = key_start;
                        while key_end < bytes.len() && is_identifier_char(bytes[key_end] as char) {
                            key_end += 1;
                        }
                        let after = skip_ascii_whitespace(bytes, key_end);
                        (key_end > key_start
                            && bytes.get(after) == Some(&b'=')
                            && bytes.get(after + 1) != Some(&b'=')
                            && bytes.get(after + 1) != Some(&b'>'))
                        .then_some(AdvancedSignalKind::PropertyAssignment)
                    }
                    _ => None,
                };
                if let Some(kind) = kind {
                    out.push(AdvancedSignal {
                        offset: start,
                        kind,
                        called,
                    });
                }
            }
            _ => index += 1,
        }
    }
    out
}

/// True when the expression at `from` is just the tracked name again: `advanced: advanced`.
///
/// The long spelling of the `{ advanced }` shorthand every builder already uses. It hands on a map
/// that had to be BUILT somewhere first — and building it is what this scan catches — so reporting
/// the hand-on adds nothing and would fail the build on rewriting `advanced,` as `advanced:
/// advanced` inside a builder that is already registered.
fn is_pass_through(bytes: &[u8], text: &str, from: usize) -> bool {
    let (name, end) = identifier_at(bytes, text, from);
    if name != "advanced" && name != "advancedExtras" {
        return false;
    }
    // `advanced: advanced.buildAdvanced(x)` and `advanced: advanced[key]` are not hand-ons.
    !matches!(
        bytes.get(skip_ascii_whitespace(bytes, end)),
        Some(b'.') | Some(b'(') | Some(b'[') | Some(b'?')
    )
}

/// The callee of a call expression starting at `from`: `buildFoo(x)` → `buildFoo`,
/// `advanced.buildAdvanced(x)` → `buildAdvanced`. `None` when this is not a plain call.
fn call_head(bytes: &[u8], text: &str, from: usize) -> Option<String> {
    let mut index = from;
    loop {
        if index >= bytes.len() || !is_identifier_start(bytes[index] as char) {
            return None;
        }
        let start = index;
        while index < bytes.len() && is_identifier_char(bytes[index] as char) {
            index += 1;
        }
        let next = skip_ascii_whitespace(bytes, index);
        match bytes.get(next) {
            // `Object.keys(x)` → keep walking, the callee is the LAST name in the chain.
            Some(b'.') => index = skip_ascii_whitespace(bytes, next + 1),
            Some(b'(') => return Some(text[start..index].to_owned()),
            _ => return None,
        }
    }
}

/// Every position inside a string literal, so a BACKWARD scan can skip them.
///
/// The forward scanners skip strings by jumping over them; the context checks below walk backwards
/// and cannot, so a `<` or a `;` inside a string would otherwise decide whether a signal fires.
fn string_literal_mask(bytes: &[u8]) -> Vec<bool> {
    let mut mask = vec![false; bytes.len()];
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'"' | b'\'' | b'`' => {
                let end = skip_string_ascii(bytes, index).min(bytes.len());
                for slot in mask[index..end].iter_mut() {
                    *slot = true;
                }
                index = end.max(index + 1);
            }
            _ => index += 1,
        }
    }
    mask
}

/// The nearest significant character before `at`, skipping whitespace and string bodies.
fn previous_significant(bytes: &[u8], mask: &[bool], at: usize) -> Option<u8> {
    let mut index = at;
    while index > 0 {
        index -= 1;
        if mask[index] {
            return Some(bytes[index]);
        }
        if !bytes[index].is_ascii_whitespace() {
            return Some(bytes[index]);
        }
    }
    None
}

/// True when the identifier at `at` starts an object-literal entry (`{ advanced:`, `, advanced:`).
///
/// The discriminator against a ternary's middle arm (`cond ? advanced : fallback`) and against a
/// labelled statement, both of which put a colon after the name and build nothing.
fn is_object_key_position(bytes: &[u8], mask: &[bool], at: usize) -> bool {
    match previous_significant(bytes, mask, at) {
        Some(b'{') | Some(b',') => true,
        // A body slice can begin mid-literal, so "nothing before it" stays permissive.
        None => true,
        _ => false,
    }
}

/// True when the identifier at `at` sits in JSX attribute position: `<Child advanced={…} />`.
///
/// Walks back to the tag that would own the attribute. Anything that cannot appear between a tag
/// name and one of its attributes — a statement end, an open brace or paren at this level — means
/// this is ordinary JS.
fn is_jsx_attribute_position(bytes: &[u8], mask: &[bool], at: usize) -> bool {
    let mut depth = 0_usize;
    let mut index = at;
    while index > 0 {
        index -= 1;
        if mask[index] {
            continue;
        }
        match bytes[index] {
            b'}' => depth += 1,
            b'{' => {
                if depth == 0 {
                    return false;
                }
                depth -= 1;
            }
            b'(' | b')' | b';' | b',' if depth == 0 => return false,
            // `=>` is an arrow, not a tag that just closed.
            b'>' if depth == 0 && bytes.get(index.wrapping_sub(1)) != Some(&b'=') => return false,
            b'<' if depth == 0 => {
                return bytes
                    .get(index + 1)
                    .is_some_and(|next| is_identifier_start(*next as char));
            }
            _ => {}
        }
    }
    false
}

/// True when the identifier at `at` is being BOUND rather than built: a parameter destructuring
/// default (`function Row({ advanced = {} })`, `({ advanced = {} }) => …`) or a destructuring
/// declaration (`const { advanced = {} } = props`, `const { advanced: opts } = props`).
///
/// Decided by what follows the pattern, because what PRECEDES it cannot tell a parameter list from
/// a call argument: `foo({ advanced: 1 })` and `function foo({ advanced = {} })` both put a `(`
/// before the brace. A pattern's closing brace is followed by `=` (a binding), or by the `)` of a
/// parameter list that a body or an `=>` follows.
fn is_binding_pattern_position(bytes: &[u8], mask: &[bool], at: usize) -> bool {
    let Some(open) = enclosing_open_brace(bytes, mask, at) else {
        return false;
    };
    let Some(close) = matching_close_brace(bytes, open) else {
        return false;
    };
    let after = skip_ascii_whitespace(bytes, close + 1);
    match bytes.get(after) {
        // `const { advanced = {} } = props`
        Some(b'=') => bytes.get(after + 1) != Some(&b'=') && bytes.get(after + 1) != Some(&b'>'),
        // `function Row({ advanced = {} }) {` / `({ advanced = {} }) => …`
        Some(b')') => opens_a_function_body(bytes, after + 1),
        // `function Row({ advanced = {} }, ref) {` — a non-final parameter.
        Some(b',') => enclosing_close_paren(bytes, after + 1)
            .is_some_and(|paren| opens_a_function_body(bytes, paren + 1)),
        _ => false,
    }
}

/// True when a function body (`{`) or an arrow (`=>`) starts at or after `from`.
fn opens_a_function_body(bytes: &[u8], from: usize) -> bool {
    let at = skip_ascii_whitespace(bytes, from);
    match bytes.get(at) {
        Some(b'{') => true,
        Some(b'=') => bytes.get(at + 1) == Some(&b'>'),
        _ => false,
    }
}

/// The innermost `{` still open at `at`.
fn enclosing_open_brace(bytes: &[u8], mask: &[bool], at: usize) -> Option<usize> {
    let mut depth = 0_usize;
    let mut index = at;
    while index > 0 {
        index -= 1;
        if mask[index] {
            continue;
        }
        match bytes[index] {
            b'}' => depth += 1,
            b'{' => {
                if depth == 0 {
                    return Some(index);
                }
                depth -= 1;
            }
            _ => {}
        }
    }
    None
}

/// The `}` that closes the `{` at `open`.
fn matching_close_brace(bytes: &[u8], open: usize) -> Option<usize> {
    let mut depth = 0_usize;
    let mut index = open;
    while index < bytes.len() {
        match bytes[index] {
            b'"' | b'\'' | b'`' => {
                index = skip_string_ascii(bytes, index);
                continue;
            }
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
        index += 1;
    }
    None
}

/// The `)` that closes the parenthesis `from` is already inside.
fn enclosing_close_paren(bytes: &[u8], from: usize) -> Option<usize> {
    let mut depth = 0_usize;
    let mut index = from;
    while index < bytes.len() {
        match bytes[index] {
            b'"' | b'\'' | b'`' => {
                index = skip_string_ascii(bytes, index);
                continue;
            }
            b'(' | b'[' | b'{' => depth += 1,
            b']' | b'}' => {
                if depth == 0 {
                    return None;
                }
                depth -= 1;
            }
            b')' => {
                if depth == 0 {
                    return Some(index);
                }
                depth -= 1;
            }
            _ => {}
        }
        index += 1;
    }
    None
}

/// The offsets of `function <name>`'s body in `stripped` — the text BETWEEN its braces.
///
/// Panics with instructions rather than returning nothing when the function is gone: a registry
/// entry pointing at a renamed function must fail the lint, not quietly stop covering it.
fn function_body_span(stripped: &str, function: &str, path: &str) -> (usize, usize) {
    let needle = format!("function {function}");
    // On a NAME boundary, not a prefix: `function buildDetailJobBody` matches inside
    // `function buildDetailJobBodyV2`, so a rename could leave the registry pointing at a
    // different function while the lint reported one clean match.
    let occurrences: Vec<usize> = stripped
        .match_indices(needle.as_str())
        .filter(
            |(at, _)| match stripped[at + needle.len()..].chars().next() {
                Some(next) => !is_identifier_char(next),
                None => true,
            },
        )
        .map(|(at, _)| at)
        .collect();
    assert_eq!(
        occurrences.len(),
        1,
        "`{needle}` appears {} times in {path}; this coverage lint needs exactly one so it reads an \
         unambiguous body. Either the function was renamed or moved — point the \
         `ADVANCED_BUILDERS` entry at wherever that `advanced` map is built now — or a second \
         function shares the name, in which case rename one.",
        occurrences.len()
    );
    let at = occurrences[0];
    let bytes = stripped.as_bytes();

    // Walk the parameter list first: a default value carries braces of its own
    // (`buildAdvanced(base = {})`), so the body's `{` is not simply the next one.
    let mut index = at + needle.len();
    while index < bytes.len() && bytes[index] != b'(' {
        index += 1;
    }
    assert!(
        index < bytes.len(),
        "`{needle}` in {path} has no parameter list"
    );
    let mut depth = 0_usize;
    while index < bytes.len() {
        match bytes[index] {
            b'"' | b'\'' | b'`' => {
                index = skip_string_ascii(bytes, index);
                continue;
            }
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    index += 1;
                    break;
                }
            }
            _ => {}
        }
        index += 1;
    }
    while index < bytes.len() && bytes[index] != b'{' {
        index += 1;
    }
    assert!(
        index < bytes.len(),
        "`{needle}` in {path} has no body — an arrow-bodied function is a shape this lint cannot \
         read"
    );
    let open = index;
    let mut depth = 0_usize;
    while index < bytes.len() {
        match bytes[index] {
            b'"' | b'\'' | b'`' => {
                index = skip_string_ascii(bytes, index);
                continue;
            }
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return (open + 1, index);
                }
            }
            _ => {}
        }
        index += 1;
    }
    panic!("`{needle}` in {path} has an unbalanced body");
}

/// The balanced text inside the first `{` at or after `from`.
fn object_literal_body(text: &str, from: usize, what: &str) -> String {
    let bytes = text.as_bytes();
    let open = from
        + text[from..]
            .find('{')
            .unwrap_or_else(|| panic!("{what} is not followed by an object literal"));
    let mut depth = 0_usize;
    let mut index = open;
    while index < bytes.len() {
        match bytes[index] {
            b'"' | b'\'' | b'`' => {
                index = skip_string_ascii(bytes, index);
                continue;
            }
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return text[open + 1..index].to_owned();
                }
            }
            _ => {}
        }
        index += 1;
    }
    panic!("{what}'s object literal is unbalanced");
}

/// Every `advanced` key one registered builder can emit, read out of its own JS shape.
///
/// The dispatch the registry's [`AdvancedBuilderShape`] exists for. Each arm fails loudly when the
/// file stops matching the shape it claims, because a green lint that has quietly stopped reading
/// anything is worse than no lint — it is the exact failure that let `cnScale` ship unclassified.
fn emitted_keys(builder: &AdvancedBuilder, source: &str) -> BTreeSet<String> {
    let stripped = scannable(source);
    let (start, end) = function_body_span(&stripped, builder.function, builder.path);
    let body = &stripped[start..end];
    let what = format!("`{}` in {}", builder.function, builder.path);

    // Every shape, not just `AssignedObject`: a `FlatAdvancedLiteral` builder can grow a computed
    // write too, and reading past one is how a knob goes missing with the lint still green.
    refuse_invisible_advanced_writes(body, &what);

    // Which signals each arm actually READ. Anything left over is a second map the registry entry
    // does not account for, refused below.
    let signals = advanced_map_signals(body);
    let (keys, consumed): (BTreeSet<String>, Vec<usize>) = match builder.shape {
        AdvancedBuilderShape::ReturnedObject => {
            let return_at = body
                .find("return {")
                .unwrap_or_else(|| panic!("{what} no longer returns an object literal"))
                + "return ".len();
            let literal = object_literal_body(body, return_at, &what);
            (scan_object_literal(&literal, &what, false).0, Vec::new())
        }
        AdvancedBuilderShape::ExtrasLiteral => {
            let at = body.find("advancedExtras:").unwrap_or_else(|| {
                panic!(
                    "{what} no longer carries an `advancedExtras:` literal, so this lint cannot \
                     see which knobs that form contributes"
                )
            });
            let literal = object_literal_body(body, at + "advancedExtras:".len(), &what);
            (scan_object_literal(&literal, &what, false).0, vec![at])
        }
        AdvancedBuilderShape::FlatAdvancedLiteral => {
            let at = body
                .find("advanced:")
                .unwrap_or_else(|| panic!("{what} no longer carries an `advanced:` literal"));
            let literal = object_literal_body(body, at + "advanced:".len(), &what);
            assert!(
                !literal.contains('{') && !literal.contains("..."),
                "{what}'s `advanced` literal is no longer flat — re-tag its registry `shape` (the \
                 conditional-spread scanner reads that shape) rather than letting this arm \
                 misread it"
            );
            let keys = literal
                .split(',')
                // `key` and `key: value` alike — take the identifier before any colon.
                .filter_map(|entry| entry.split(':').next())
                .map(str::trim)
                .filter(|key| !key.is_empty() && key.chars().all(is_identifier_char))
                .map(str::to_owned)
                .collect();
            (keys, vec![at])
        }
        AdvancedBuilderShape::AssignedObject => {
            let at = body
                .find("advanced =")
                .unwrap_or_else(|| panic!("{what} no longer initializes an `advanced` object"));
            let literal = object_literal_body(body, at + "advanced =".len(), &what);
            let (mut keys, spreads) = scan_object_literal(&literal, &what, true);
            let declared: BTreeSet<String> = builder
                .spread_of
                .iter()
                .map(|name| (*name).to_owned())
                .collect();
            assert_eq!(
                spreads, declared,
                "{what} spreads {spreads:?} into its `advanced` initializer, but its registry \
                 entry declares `spread_of: {declared:?}`. A spread nobody declared is a whole \
                 set of keys this lint cannot see — either point `spread_of` at it and register \
                 the builder that produces those keys, or stop spreading it."
            );
            let (assigned, mut consumed) = assigned_advanced_keys(body, &signals);
            keys.extend(assigned);
            consumed.push(at);
            (keys, consumed)
        }
        AdvancedBuilderShape::NoAdvancedMap => {
            assert!(
                !body.contains("advanced"),
                "{what} is registered as posting NO `advanced` map, but its body now mentions \
                 `advanced`. It feeds a lane that embeds the map into every PNG it writes, so \
                 give it a real registry `shape` and classify every key it can emit."
            );
            (BTreeSet::new(), Vec::new())
        }
    };
    refuse_unread_advanced_signals(body, &what, builder, &signals, &consumed);
    keys
}

/// A second `advanced` map inside a declared builder is NOT accounted for by that declaration.
///
/// [`every_advanced_builder_in_the_web_app_is_accounted_for`] credits every signal inside a declared
/// function's body span to that function, but the extractors above read exactly one shape each — the
/// one the registry names. So a nested helper declared inside `DocumentStudio.submit`, with its own
/// `advanced: { probeKnob, probeSecret }`, counted as accounted for while its keys were never
/// classified and the sweep stayed green. This is the refusal
/// [`refuse_invisible_advanced_writes`] makes for a write it cannot follow, applied to a map the arm
/// can see and did not read.
fn refuse_unread_advanced_signals(
    body: &str,
    what: &str,
    builder: &AdvancedBuilder,
    signals: &[AdvancedSignal],
    consumed: &[usize],
) {
    let declared = declared_builder_functions();
    for signal in signals {
        if consumed.contains(&signal.offset)
            || signal.calls_a_declared_builder(builder.path, &declared)
        {
            continue;
        }
        let line = 1 + body[..signal.offset].matches('\n').count();
        panic!(
            "{what} builds a SECOND `advanced` map at line {line} of its body ({}), which the \
             `{:?}` arm of this extractor never reads. Its keys are invisible here, so they would \
             be dropped from every shared image exactly as if the builder were unregistered. A \
             nested helper is not accounted for by the entry around it: fold those keys into the \
             map this arm does read, or give the helper its own `ADVANCED_BUILDERS` entry — the \
             innermost declared span wins, so a nested entry is credited to itself.",
            signal.kind.describe(),
            builder.shape,
        );
    }
}

/// Every function name the registry declares, registered or deferred.
fn declared_builder_functions() -> BTreeSet<&'static str> {
    ADVANCED_BUILDERS
        .iter()
        .map(|builder| builder.function)
        .chain(
            DEFERRED_ADVANCED_BUILDERS
                .iter()
                .map(|deferred| deferred.function),
        )
        .collect()
}

/// A write this lint cannot follow must fail rather than be scanned past.
///
/// Three shapes set keys without ever naming one: a computed key (`advanced["k"] = v`),
/// `Object.assign(x.advanced, { … })`, and an array mutator. Called over EVERY file the sweep
/// reads, not only inside an `AssignedObject` body: while this ran in that one arm,
/// `body.payload.advanced["probeKnob"] = v` was green both in a brand-new file and inside
/// `buildDetailJobBody`.
///
/// The `Object.assign` check reads the first ARGUMENT instead of matching the text
/// `Object.assign(advanced`, which `Object.assign(body.payload.advanced, { … })` walked straight
/// past.
fn refuse_invisible_advanced_writes(text: &str, what: &str) {
    if let Some((offset, how)) = invisible_advanced_writes(text).into_iter().next() {
        let line = 1 + text[..offset].matches('\n').count();
        panic!(
            "{what} writes `advanced` through {how} at line {line}, which this coverage lint \
             cannot read — the keys it sets are invisible here and would be silently dropped from \
             every shared image. Write them as plain `advanced.key = …` assignments, or as an \
             `advanced: {{ … }}` literal, inside a builder `ADVANCED_BUILDERS` declares."
        );
    }
}

/// Mutators that reshape a map/array without naming a key.
const INVISIBLE_ADVANCED_MUTATORS: &[&str] = &["push", "unshift", "splice"];

/// Every `advanced` write in `text` that this lint cannot follow, as (offset, how).
fn invisible_advanced_writes(text: &str) -> Vec<(usize, String)> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'"' | b'\'' | b'`' => {
                index = skip_string_ascii(bytes, index);
                continue;
            }
            byte if is_identifier_start(byte as char) => {
                let start = index;
                while index < bytes.len() && is_identifier_char(bytes[index] as char) {
                    index += 1;
                }
                let word = &text[start..index];
                let after = skip_ascii_whitespace(bytes, index);
                match word {
                    "advanced" | "advancedExtras" => match bytes.get(after) {
                        Some(b'[') => out.push((start, format!("a computed key (`{word}[…]`)"))),
                        Some(b'.') => {
                            let (method, method_end) = identifier_at(bytes, text, after + 1);
                            if INVISIBLE_ADVANCED_MUTATORS.contains(&method)
                                && bytes.get(skip_ascii_whitespace(bytes, method_end))
                                    == Some(&b'(')
                            {
                                out.push((start, format!("`{word}.{method}(…)`")));
                            }
                        }
                        _ => {}
                    },
                    "Object" => {
                        if bytes.get(after) != Some(&b'.') {
                            continue;
                        }
                        let (method, method_end) = identifier_at(bytes, text, after + 1);
                        let open = skip_ascii_whitespace(bytes, method_end);
                        if method != "assign" || bytes.get(open) != Some(&b'(') {
                            continue;
                        }
                        let Some(target) = first_argument(bytes, text, open + 1) else {
                            continue;
                        };
                        let target = target.trim();
                        if ["advanced", "advancedExtras"].contains(&target)
                            || target.ends_with(".advanced")
                            || target.ends_with(".advancedExtras")
                        {
                            out.push((start, format!("`Object.assign({target}, …)`")));
                        }
                    }
                    _ => {}
                }
            }
            _ => index += 1,
        }
    }
    out
}

/// The identifier starting at or after `from`, plus the offset just past it.
fn identifier_at<'text>(bytes: &[u8], text: &'text str, from: usize) -> (&'text str, usize) {
    let start = skip_ascii_whitespace(bytes, from);
    let mut end = start;
    while end < bytes.len() && is_identifier_char(bytes[end] as char) {
        end += 1;
    }
    (&text[start..end], end)
}

/// The text of the first argument of a call whose `(` ends just before `from`.
fn first_argument(bytes: &[u8], text: &str, from: usize) -> Option<String> {
    let mut depth = 0_usize;
    let mut index = from;
    while index < bytes.len() {
        match bytes[index] {
            b'"' | b'\'' | b'`' => {
                index = skip_string_ascii(bytes, index);
                continue;
            }
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' | b',' if depth == 0 => {
                return (bytes[index] != b']' && bytes[index] != b'}')
                    .then(|| text[from..index].to_owned());
            }
            b')' | b']' | b'}' => depth -= 1,
            _ => {}
        }
        index += 1;
    }
    None
}

/// Every `advanced.key = …` assignment in one function body, with the signal offsets it read.
fn assigned_advanced_keys(
    body: &str,
    signals: &[AdvancedSignal],
) -> (BTreeSet<String>, Vec<usize>) {
    let mut keys = BTreeSet::new();
    let mut consumed = Vec::new();
    for signal in signals
        .iter()
        .filter(|signal| signal.kind == AdvancedSignalKind::PropertyAssignment)
    {
        let after_dot = signal.offset + "advanced.".len();
        let bytes = body.as_bytes();
        let mut end = after_dot;
        while end < bytes.len() && is_identifier_char(bytes[end] as char) {
            end += 1;
        }
        keys.insert(body[after_dot..end].to_owned());
        consumed.push(signal.offset);
    }
    (keys, consumed)
}

/// Every key an object literal built from conditional spreads can contribute, plus the bare
/// identifiers it spreads.
///
/// A small brace-aware scanner rather than a regex, because the studio builder's whole shape is
/// conditional spreads: `...(cond ? { key: value } : {})` contributes a TOP-LEVEL advanced
/// key from brace depth two, while `{ controlWeights: { overlayId } }` and a call argument
/// object like `buildStructuredPromptRecipe({ intent })` do not. A regex cannot tell those
/// apart, and getting it wrong in the permissive direction would make this lint pass on a key
/// that is never emitted while missing one that is.
///
/// Rule: a brace is an "emit scope" when it is the literal itself, or when it is an
/// object literal at the top level of a spread expression (`...( … )`) written inside another
/// emit scope — which covers BOTH branches of the `cond ? { a } : { b }` the builder uses.
/// Only identifiers in key position inside an emit scope count.
///
/// # Fails loud, never quiet
///
/// The scanner understands ONE shape, and a lint that silently understands nothing is worse
/// than no lint: `...buildFutureKnobs(state)` in the return object would contribute no keys, the
/// key floor and every anchor would still resolve, and a brand-new knob would stop
/// travelling with zero signal. So a spread this scanner cannot follow — a call expression, or a
/// parenthesized expression with no object literal in it — PANICS with instructions instead of
/// scanning past it. Fail-safe on the privacy axis (the key is dropped, not leaked) is only half
/// of what this lint is for; the other half is noticing that a knob went missing.
///
/// `allow_identifier_spreads` is the one exception, for `{ ...base, key }`: an
/// [`AdvancedBuilderShape::AssignedObject`] builder merges another registered builder's map in by
/// name. Those identifiers are RETURNED rather than ignored, so the caller can assert they are
/// exactly the ones the registry declares.
fn scan_object_literal(
    body: &str,
    what: &str,
    allow_identifier_spreads: bool,
) -> (BTreeSet<String>, BTreeSet<String>) {
    let chars: Vec<char> = body.chars().collect();
    let mut keys = BTreeSet::new();
    let mut identifier_spreads = BTreeSet::new();
    // (is_emit_scope, expecting_a_key)
    let mut scopes: Vec<(bool, bool)> = vec![(true, true)];
    // Open spread expressions, as (open scope count, paren depth at the `...`, produced an
    // object literal). Nested, because `...(a ? { k, ...(b ? { j } : {}) } : {})` happens.
    let mut spreads: Vec<(usize, usize, bool)> = Vec::new();
    let mut paren_depth = 0_usize;
    let mut index = 0;

    while index < chars.len() {
        let character = chars[index];
        if character.is_whitespace() {
            index += 1;
            continue;
        }
        match character {
            '{' => {
                let in_spread = spreads
                    .last()
                    .is_some_and(|(scope_count, _, _)| *scope_count == scopes.len());
                let is_emit = in_spread && scopes.last().is_some_and(|(emit, _)| *emit);
                if let (true, Some(spread)) = (is_emit, spreads.last_mut()) {
                    spread.2 = true;
                }
                if let Some(scope) = scopes.last_mut() {
                    scope.1 = false;
                }
                scopes.push((is_emit, true));
                index += 1;
            }
            '}' => {
                scopes.pop();
                index += 1;
                if scopes.is_empty() {
                    break;
                }
            }
            '(' => {
                paren_depth += 1;
                if let Some(scope) = scopes.last_mut() {
                    scope.1 = false;
                }
                index += 1;
            }
            ')' => {
                paren_depth = paren_depth.saturating_sub(1);
                while let Some((_, spread_paren, produced)) = spreads.last().copied() {
                    if spread_paren < paren_depth {
                        break;
                    }
                    assert!(
                        produced,
                        "a spread in {what} contains no object literal, so `scan_object_literal` \
                         in this test scanned past it and contributed nothing. Every key inside \
                         it is invisible to this lint. Write the spread as \
                         `...(cond ? {{ key }} : {{}})`, or teach the scanner the new shape."
                    );
                    spreads.pop();
                }
                if let Some(scope) = scopes.last_mut() {
                    scope.1 = false;
                }
                index += 1;
            }
            ',' => {
                if let Some(scope) = scopes.last_mut() {
                    scope.1 = true;
                }
                index += 1;
            }
            '.' => {
                if index + 2 < chars.len() && chars[index + 1] == '.' && chars[index + 2] == '.' {
                    if scopes.last().is_some_and(|(emit, _)| *emit) {
                        // The scanner follows `...( … )`, plus a bare identifier where the caller
                        // has declared one. Anything else — a call, a member expression — would
                        // leave a spread open forever, so the next nested object VALUE would read
                        // as an emit scope: refuse rather than guess.
                        let mut lookahead = index + 3;
                        while lookahead < chars.len() && chars[lookahead].is_whitespace() {
                            lookahead += 1;
                        }
                        let identifier_spread = allow_identifier_spreads
                            && chars
                                .get(lookahead)
                                .copied()
                                .is_some_and(is_identifier_start);
                        if identifier_spread {
                            let start = lookahead;
                            while lookahead < chars.len() && is_identifier_char(chars[lookahead]) {
                                lookahead += 1;
                            }
                            let mut after = lookahead;
                            while after < chars.len() && chars[after].is_whitespace() {
                                after += 1;
                            }
                            assert!(
                                matches!(chars.get(after), Some(',') | Some('}') | None),
                                "{what} spreads something this coverage lint cannot read: a \
                                 member or call expression (`...{}`). Every key it contributes is \
                                 invisible to this lint, so a new knob would silently stop \
                                 travelling.",
                                chars[start..]
                                    .iter()
                                    .take(24)
                                    .collect::<String>()
                                    .trim_end()
                            );
                            identifier_spreads
                                .insert(chars[start..lookahead].iter().collect::<String>());
                            index = lookahead;
                            if let Some(scope) = scopes.last_mut() {
                                scope.1 = false;
                            }
                            continue;
                        }
                        assert!(
                            chars.get(lookahead) == Some(&'('),
                            "{what} spreads something this coverage lint cannot read: a bare \
                             identifier or a call expression (`...{}`). Every key it contributes \
                             is invisible to this lint, so a new knob would silently stop \
                             travelling. Write it as `...(cond ? {{ key }} : {{}})`, or teach the \
                             scanner the new shape.",
                            chars[lookahead..]
                                .iter()
                                .take(24)
                                .collect::<String>()
                                .trim_end()
                        );
                        spreads.push((scopes.len(), paren_depth, false));
                    }
                    index += 3;
                } else {
                    index += 1;
                }
                if let Some(scope) = scopes.last_mut() {
                    scope.1 = false;
                }
            }
            '"' | '\'' | '`' => {
                index = skip_string(&chars, index);
                if let Some(scope) = scopes.last_mut() {
                    scope.1 = false;
                }
            }
            character if is_identifier_start(character) => {
                let start = index;
                while index < chars.len() && is_identifier_char(chars[index]) {
                    index += 1;
                }
                let (is_emit, expecting) = *scopes.last().expect("a scope is always open");
                if is_emit && expecting {
                    let mut lookahead = index;
                    while lookahead < chars.len() && chars[lookahead].is_whitespace() {
                        lookahead += 1;
                    }
                    // End of the literal counts as a terminator: the caller passes the text
                    // INSIDE the braces, so a final entry with no trailing comma
                    // (`{ ...base, ipAdapterScale }`) ends at the slice's end rather than at `}`.
                    if lookahead >= chars.len() || matches!(chars[lookahead], ':' | ',' | '}') {
                        keys.insert(chars[start..index].iter().collect::<String>());
                    }
                }
                if let Some(scope) = scopes.last_mut() {
                    scope.1 = false;
                }
            }
            _ => {
                if let Some(scope) = scopes.last_mut() {
                    scope.1 = false;
                }
                index += 1;
            }
        }
    }
    assert!(
        spreads.is_empty(),
        "a spread in {what} never closed, so `scan_object_literal` in this test lost track of the \
         literal's shape. Fix the scanner rather than letting it scan a shape it does not \
         understand."
    );
    (keys, identifier_spreads)
}

/// Every key `buildImageJobAdvanced` can put into the `advanced` payload — the studio builder's
/// extractor, exposed on its own so the scanner's own unit tests can feed it a synthetic builder.
fn emitted_advanced_keys(source: &str) -> BTreeSet<String> {
    emitted_keys(
        ADVANCED_BUILDERS
            .iter()
            .find(|builder| builder.source == AdvancedKeySource::StudioBuilder)
            .expect("the studio builder is registered"),
        source,
    )
}

/// Blank out `//` and `/* */` comments without disturbing string literals.
fn strip_comments(source: &str) -> String {
    let chars: Vec<char> = source.chars().collect();
    let mut out = String::with_capacity(source.len());
    let mut index = 0;
    while index < chars.len() {
        let character = chars[index];
        match character {
            '/' if index + 1 < chars.len() && chars[index + 1] == '/' => {
                while index < chars.len() && chars[index] != '\n' {
                    index += 1;
                }
            }
            '/' if index + 1 < chars.len() && chars[index + 1] == '*' => {
                index += 2;
                while index + 1 < chars.len() && !(chars[index] == '*' && chars[index + 1] == '/') {
                    index += 1;
                }
                index = (index + 2).min(chars.len());
                out.push(' ');
            }
            '"' | '\'' | '`' => {
                let end = skip_string(&chars, index);
                out.extend(&chars[index..end]);
                index = end;
            }
            _ => {
                out.push(character);
                index += 1;
            }
        }
    }
    out
}

/// Index just past the string literal starting at `start`.
fn skip_string(chars: &[char], start: usize) -> usize {
    let quote = chars[start];
    let mut index = start + 1;
    while index < chars.len() {
        match chars[index] {
            '\\' => index += 2,
            character if character == quote => return index + 1,
            _ => index += 1,
        }
    }
    chars.len()
}

/// [`skip_string`] over the ASCII-normalized text the offset-based scanners work on.
fn skip_string_ascii(bytes: &[u8], start: usize) -> usize {
    let quote = bytes[start];
    let mut index = start + 1;
    while index < bytes.len() {
        match bytes[index] {
            b'\\' => index += 2,
            byte if byte == quote => return index + 1,
            _ => index += 1,
        }
    }
    bytes.len()
}

/// The first non-whitespace offset at or after `from`.
fn skip_ascii_whitespace(bytes: &[u8], from: usize) -> usize {
    let mut index = from;
    while index < bytes.len() && bytes[index].is_ascii_whitespace() {
        index += 1;
    }
    index
}

fn is_identifier_start(character: char) -> bool {
    character.is_ascii_alphabetic() || character == '_' || character == '$'
}

fn is_identifier_char(character: char) -> bool {
    is_identifier_start(character) || character.is_ascii_digit()
}
