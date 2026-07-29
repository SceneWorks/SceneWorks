//! Contract tests for the sanitized workflow share envelope (sc-15946, epic 15945).
//!
//! The two load-bearing ones are [`every_advanced_key_the_studio_can_emit_is_classified`] —
//! which parses `apps/web/src/imageJobAdvanced.js` so a new knob cannot silently leak OR
//! silently vanish — and [`no_value_in_a_built_envelope_is_path_shaped`], which seeds the
//! request with paths on purpose and asserts none of them reach the file.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use sceneworks_core::contracts::{Asset, JsonObject};
use sceneworks_core::workflow_share::{
    build_workflow_share, is_path_shaped, parse_workflow_share, AdvancedKeySource, WorkflowShare,
    ADVANCED_KEY_RULES, PRODUCER_URL, PRODUCER_VERSION, WORKFLOW_SHARE_SCHEMA_VERSION,
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

const ADVANCED_BUILDER_PATH: &str = "apps/web/src/imageJobAdvanced.js";

#[test]
fn every_advanced_key_the_studio_can_emit_is_classified() {
    let source = read_repo_file(ADVANCED_BUILDER_PATH);
    let emitted = emitted_advanced_keys(&source);

    // A sanity floor: if the extractor ever stops understanding the file (a refactor to a
    // different builder shape, a move), it must fail loudly rather than pass with an empty set.
    assert!(
        emitted.len() >= 25,
        "only {} keys were extracted from {ADVANCED_BUILDER_PATH} — the extractor no longer \
         understands that file, so this lint is not protecting anything. Fix \
         `emitted_advanced_keys` in this test.",
        emitted.len()
    );
    for anchor in ["sampler", "steps", "styleId", "poses", "resolution"] {
        assert!(
            emitted.contains(anchor),
            "the extractor missed the known key `{anchor}` in {ADVANCED_BUILDER_PATH}"
        );
    }

    let classified: BTreeSet<String> = ADVANCED_KEY_RULES
        .iter()
        .map(|rule| rule.key.to_owned())
        .collect();
    let from_builder: BTreeSet<String> = ADVANCED_KEY_RULES
        .iter()
        .filter(|rule| rule.source == AdvancedKeySource::StudioBuilder)
        .map(|rule| rule.key.to_owned())
        .collect();

    let unclassified: Vec<&String> = emitted.difference(&classified).collect();
    assert!(
        unclassified.is_empty(),
        "buildImageJobAdvanced can emit {unclassified:?}, which `ADVANCED_KEY_RULES` in \
         crates/sceneworks-core/src/workflow_share.rs does not classify.\n\
         A new advanced knob must be classified before it ships: add it to the table as \
         `allow(..)` if it describes WHAT TO MAKE (it travels in shared images) or `deny(..)` \
         if it describes WHAT THIS MACHINE CAN AFFORD (tier, kernel, GPU) or names anything \
         local (an id, a path, a preset). Either way write the reason — an unclassified key \
         is dropped silently today and that is exactly what this lint exists to stop."
    );

    let stale: Vec<&String> = from_builder.difference(&emitted).collect();
    assert!(
        stale.is_empty(),
        "`ADVANCED_KEY_RULES` classifies {stale:?} as coming from buildImageJobAdvanced, but \
         {ADVANCED_BUILDER_PATH} no longer emits them.\n\
         Either the knob was removed (drop the rule) or it moved to the server (re-tag the \
         rule `AdvancedKeySource::Server`). A rule that no longer matches its source is a \
         classification nobody is maintaining."
    );

    // The rules table must not carry a key twice with two dispositions.
    assert_eq!(
        classified.len(),
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

/// Every key `buildImageJobAdvanced` can put into the `advanced` payload.
///
/// A small brace-aware scanner rather than a regex, because the builder's whole shape is
/// conditional spreads: `...(cond ? { key: value } : {})` contributes a TOP-LEVEL advanced
/// key from brace depth two, while `{ controlWeights: { overlayId } }` and a call argument
/// object like `buildStructuredPromptRecipe({ intent })` do not. A regex cannot tell those
/// apart, and getting it wrong in the permissive direction would make this lint pass on a key
/// that is never emitted while missing one that is.
///
/// Rule: a brace is an "emit scope" when it is the return object itself, or when it is an
/// object literal at the top level of a spread expression (`...( … )`) written inside another
/// emit scope — which covers BOTH branches of the `cond ? { a } : { b }` the builder uses.
/// Only identifiers in key position inside an emit scope count.
///
/// # Fails loud, never quiet
///
/// The scanner understands ONE shape, and a lint that silently understands nothing is worse
/// than no lint: `...buildFutureKnobs(state)` in the return object would contribute no keys, the
/// `>= 25` floor and every anchor would still resolve, and a brand-new knob would stop
/// travelling with zero signal. So a spread this scanner cannot follow — a bare identifier
/// (`...defaults,`), a call expression, or a parenthesized expression with no object literal in
/// it — PANICS with instructions instead of scanning past it. Fail-safe on the privacy axis
/// (the key is dropped, not leaked) is only half of what this lint is for; the other half is
/// noticing that a knob went missing.
fn emitted_advanced_keys(source: &str) -> BTreeSet<String> {
    let body = builder_return_body(source);
    let chars: Vec<char> = body.chars().collect();
    let mut keys = BTreeSet::new();
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
                        "a spread in buildImageJobAdvanced's return object contains no object \
                         literal, so `emitted_advanced_keys` in this test scanned past it and \
                         contributed nothing. Every key inside it is invisible to this lint. \
                         Write the spread as `...(cond ? {{ key }} : {{}})`, or teach the \
                         scanner the new shape."
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
                        // The scanner follows `...( … )` and nothing else. A bare identifier or
                        // a call would leave a spread open forever — the next nested object
                        // VALUE would then read as an emit scope — so refuse rather than guess.
                        let mut lookahead = index + 3;
                        while lookahead < chars.len() && chars[lookahead].is_whitespace() {
                            lookahead += 1;
                        }
                        assert!(
                            chars.get(lookahead) == Some(&'('),
                            "buildImageJobAdvanced spreads something `emitted_advanced_keys` in \
                             this test cannot read: a bare identifier or a call expression \
                             (`...{}`). Every key it contributes is invisible to this lint, so a \
                             new knob would silently stop travelling. Write it as \
                             `...(cond ? {{ key }} : {{}})`, or teach the scanner the new shape.",
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
                    if lookahead < chars.len() && matches!(chars[lookahead], ':' | ',' | '}') {
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
        "a spread in buildImageJobAdvanced's return object never closed, so \
         `emitted_advanced_keys` in this test lost track of the builder's shape. Fix the \
         scanner rather than letting it scan a shape it does not understand."
    );
    keys
}

/// The text inside `buildImageJobAdvanced`'s `return { … }`, comments removed.
fn builder_return_body(source: &str) -> String {
    let stripped = strip_comments(source);
    let function_at = stripped
        .find("function buildImageJobAdvanced")
        .unwrap_or_else(|| {
            panic!(
                "`buildImageJobAdvanced` is gone from {ADVANCED_BUILDER_PATH} — this coverage \
                 lint must be pointed at wherever the advanced payload is built now"
            )
        });
    let return_at = stripped[function_at..]
        .find("return {")
        .map(|offset| function_at + offset + "return ".len())
        .expect("buildImageJobAdvanced returns an object literal");

    let chars: Vec<char> = stripped[return_at..].chars().collect();
    let mut depth = 0_usize;
    let mut index = 0;
    let mut end = None;
    while index < chars.len() {
        match chars[index] {
            '"' | '\'' | '`' => {
                index = skip_string(&chars, index);
                continue;
            }
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(index);
                    break;
                }
            }
            _ => {}
        }
        index += 1;
    }
    let end = end.expect("the returned object literal is balanced");
    chars[1..end].iter().collect()
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

fn is_identifier_start(character: char) -> bool {
    character.is_ascii_alphabetic() || character == '_' || character == '$'
}

fn is_identifier_char(character: char) -> bool {
    is_identifier_start(character) || character.is_ascii_digit()
}
