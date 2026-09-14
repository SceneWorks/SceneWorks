//! `film-harness make-references` end to end (sc-23403): the REAL image route in-process, a
//! scripted worker that claims `image_generate` and writes a deterministic plate where the GPU
//! would, and the generator driving both through `ApiTransport`.
//!
//! What the fake replaces is only the render. Job creation, asset persistence, the sidecar and
//! recipe, the asset index, the two-phase result handoff, the metrics block the memory budget is
//! read from and the file route the plate is downloaded back through are all production code paths.
//!
//! The two questions these tests exist to answer, both from the story's acceptance criteria:
//!
//! 1. a run against a spec writes N plates and a `references.jsonc` that `film-harness validate`
//!    accepts **for the courier plan**, with `generated: true` and full provenance on every
//!    generated entry — and the `generated` flag reaches the imported asset when the harness later
//!    runs that pack;
//! 2. a spec role with no prompt, a job that fails, a job that overruns the spec's own budget and a
//!    job that blows its memory budget are each refused BY NAME and leave no half-written pack —
//!    not the `--out` directory, and not the temporary directory it would have been renamed from.

use std::path::{Path, PathBuf};
use std::time::Duration;

use sceneworks_core::film_plan::{self, ReferencePack};
use serde_json::{json, Value};

use crate::film_harness::references::{make_references, MakeReferencesOptions};
use crate::film_harness::{self, sha256_hex, HarnessError};
use crate::tests::film_harness::{Harness, FIXTURE_DIR};

/// These tests are SERIALIZED against each other.
///
/// Each one stands up an API plus a claiming worker and drives five image jobs through it, and they
/// run in the same binary as the rest of the film-harness suite — which holds tests with hard
/// wall-clock budgets measured in single seconds (a review's `limits.maxSeconds`, a plan's
/// `maxShotSeconds`). Four more harnesses rendering CONCURRENTLY was enough extra load on this
/// machine to spend one of those budgets on scheduling alone
/// (`a_review_that_spends_its_wall_clock_budget…` measured 0 of 6 answers instead of a partial
/// set). One at a time keeps what this story adds to a single extra harness. It is a load bound,
/// not shared state: these tests share nothing but the machine.
static GENERATOR_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Take [`GENERATOR_LOCK`] for the rest of the test.
fn serialized() -> std::sync::MutexGuard<'static, ()> {
    GENERATOR_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The shipped spec, rendered with the shipped model id against the in-process API.
fn options(harness: &Harness, spec_path: PathBuf, out: &str) -> MakeReferencesOptions {
    let mut options = MakeReferencesOptions::new(spec_path, harness.temp_dir.path().join(out));
    // Neither Krea tier is installed on a CI runner, and this suite is about the generator, not
    // about the install gate (which has its own test below).
    options.require_installed = false;
    options.poll_interval = Duration::from_millis(100);
    options
}

fn shipped_spec() -> PathBuf {
    Path::new(FIXTURE_DIR).join("references.spec.jsonc")
}

/// A copy of the whole shipped fixture directory in the temp dir, with `edit` applied to the spec,
/// so a test can break one field without touching the checked-in documents. The source pack and its
/// files come along because the spec INHERITS from them by a relative path.
fn edited_spec(harness: &Harness, name: &str, edit: impl FnOnce(&mut Value)) -> PathBuf {
    let dir = harness.temp_dir.path().join(name);
    std::fs::create_dir_all(dir.join("references")).expect("references dir");
    std::fs::create_dir_all(dir.join("sound")).expect("sound dir");
    for sub in ["references", "sound"] {
        for entry in std::fs::read_dir(Path::new(FIXTURE_DIR).join(sub)).expect("fixture subdir") {
            let entry = entry.expect("directory entry");
            std::fs::copy(entry.path(), dir.join(sub).join(entry.file_name()))
                .expect("file copies");
        }
    }
    std::fs::copy(
        Path::new(FIXTURE_DIR).join("references.jsonc"),
        dir.join("references.jsonc"),
    )
    .expect("source pack copies");
    let text = std::fs::read_to_string(shipped_spec()).expect("shipped spec");
    let mut spec: Value =
        serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(&text))
            .expect("shipped spec parses");
    edit(&mut spec);
    let path = dir.join("references.spec.jsonc");
    std::fs::write(&path, serde_json::to_string_pretty(&spec).unwrap()).expect("spec writes");
    path
}

fn read_pack(path: &Path) -> ReferencePack {
    film_plan::read_reference_pack_file(path).expect("the written pack parses")
}

/// Nothing of a failed run survives: neither `--out` nor the hidden sibling it would have been
/// renamed from.
fn assert_nothing_published(options: &MakeReferencesOptions) {
    assert!(
        !options.out_dir.exists(),
        "{} exists after a refused run",
        options.out_dir.display()
    );
    let parent = options.out_dir.parent().expect("out has a parent");
    let leftovers: Vec<String> = std::fs::read_dir(parent)
        .expect("parent readable")
        .filter_map(Result::ok)
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.contains("make-references-pending"))
        .collect();
    assert!(
        leftovers.is_empty(),
        "a refused run left a pending directory behind: {leftovers:?}"
    );
}

#[tokio::test]
async fn make_references_writes_a_pack_the_courier_plan_validates() {
    let _serial = serialized();
    let harness = Harness::start(true, Vec::new()).await;
    let options = options(&harness, shipped_spec(), "generated-pack");
    let build = make_references(&harness.transport, &options)
        .await
        .expect("the shipped spec renders");

    // One job per declared role, and nothing else.
    assert_eq!(build.generated.len(), 5, "{:#?}", build.generated);
    assert_eq!(image_jobs(&harness), 5, "one image job per role");
    assert_eq!(
        build.inherited,
        vec!["house_style".to_owned(), "workshop_plate".to_owned()]
    );

    // The files are really there, at the geometry the spec asked for, and the document's sha256 is
    // the sha256 of the bytes on disk.
    let pack = read_pack(&build.pack_path);
    assert_eq!(pack.id, "courier-workshop-refs");
    assert_eq!(pack.references.len(), 7, "5 generated + 2 inherited");
    assert_eq!(pack.sound.len(), 4, "the source pack's sound is copied");
    for entry in &pack.sound {
        assert!(
            options.out_dir.join(&entry.file).is_file(),
            "sound {:?} was not copied",
            entry.role
        );
    }
    let generated: Vec<_> = pack.references.iter().filter(|e| e.generated).collect();
    assert_eq!(generated.len(), 5);
    for entry in &generated {
        let path = options.out_dir.join(&entry.file);
        let bytes = std::fs::read(&path).expect("plate written");
        assert!(
            bytes.starts_with(b"\x89PNG"),
            "{:?} is not a PNG",
            entry.role
        );
        assert!(entry.approved, "a generated plate is approved");
        let generation = entry
            .generation
            .as_ref()
            .unwrap_or_else(|| panic!("{:?} carries no provenance", entry.role));
        assert_eq!(generation.model, "krea_2_turbo");
        assert_eq!(generation.tier.as_deref(), Some("q8"));
        assert_eq!(generation.backend.as_deref(), Some("mlx"));
        assert_eq!(generation.mode, "text_to_image");
        assert_eq!((generation.width, generation.height), (1024, 1024));
        assert!(!generation.prompt.trim().is_empty());
        assert_eq!(
            generation.negative_prompt, None,
            "krea_2_turbo declares supportsNegativePrompt: false, so none was sent"
        );
        assert!(
            generation.seed.is_some(),
            "the seed that rendered is recorded"
        );
        assert!(generation.job_id.starts_with("job"), "{generation:?}");
        assert!(generation.asset_id.starts_with("asset_"), "{generation:?}");
        assert_eq!(
            generation.sha256,
            sha256_hex(&bytes),
            "the recorded digest is the digest of the published file"
        );
        assert!(!generation.created_at.is_empty());
    }
    // Seeds come off the spec's `seedBase`, in declaration order, so a re-run asks for the same
    // five images.
    let seeds: Vec<i64> = generated
        .iter()
        .filter_map(|entry| entry.generation.as_ref().and_then(|g| g.seed))
        .collect();
    assert_eq!(seeds, vec![4200, 4201, 4202, 4203, 4204]);

    // An inherited plate is NOT marked generated: a supplied plate stays supplied.
    for entry in pack.references.iter().filter(|e| !e.generated) {
        assert!(entry.generation.is_none(), "{:?}", entry.role);
        assert!(
            ["house_style", "workshop_plate"].contains(&entry.role.as_str()),
            "{:?} should have been generated",
            entry.role
        );
    }

    // THE acceptance criterion: `film-harness validate` accepts it for the courier plan.
    let run_options = harness.options(harness.fixture_plan(), build.pack_path.clone(), None);
    let (plan, validated) = film_harness::validate(Some(&harness.transport), &run_options)
        .await
        .expect("the generated pack validates for the courier plan");
    assert_eq!(plan.id, "courier-workshop");
    assert_eq!(validated.references.len(), 7);

    // A published pack is not silently replaced: the second run is refused, and the pack on disk
    // is untouched by the refusal.
    let published = std::fs::read_to_string(&build.pack_path).expect("pack readable");
    let error = make_references(&harness.transport, &options)
        .await
        .expect_err("a populated --out is refused");
    assert!(
        matches!(&error, HarnessError::Refused(message) if message.contains("--force")),
        "{error}"
    );
    assert_eq!(
        std::fs::read_to_string(&build.pack_path).expect("pack readable"),
        published,
        "the refused run must not have touched the published pack"
    );
    let mut forced = options.clone();
    forced.force = true;
    let second = make_references(&harness.transport, &forced)
        .await
        .expect("--force replaces the published pack");
    assert_eq!(second.pack.references.len(), 7);
    assert!(read_pack(&second.pack_path)
        .references
        .iter()
        .any(|entry| entry.generated));
}

#[tokio::test]
async fn the_generated_flag_reaches_the_imported_assets_provenance() {
    let _serial = serialized();
    let harness = Harness::start(true, Vec::new()).await;
    let options = options(&harness, shipped_spec(), "generated-pack");
    let build = make_references(&harness.transport, &options)
        .await
        .expect("the shipped spec renders");

    // Run ONE shot of the courier plan against the generated pack: the import is what carries the
    // provenance onto the asset.
    let mut run_options = harness.options(
        harness.edited_plan(|_| {}),
        build.pack_path.clone(),
        Some(&["SH010"]),
    );
    run_options.export = false;
    let record = film_harness::run(&harness.transport, &run_options)
        .await
        .expect("the run completes");
    let project_id = record.project_id.clone().expect("the run made a project");
    let (_, assets) = crate::tests::support::request(
        harness.app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/assets"),
        Value::Null,
    )
    .await;
    let assets = assets.as_array().cloned().unwrap_or_default();
    let mut seen = 0;
    for reference in &record.references {
        let asset = assets
            .iter()
            .find(|asset| asset["id"].as_str() == Some(reference.asset_id.as_str()))
            .unwrap_or_else(|| panic!("no asset for role {:?}", reference.role));
        let provenance = asset.pointer("/extra/filmHarness").unwrap_or_else(|| {
            panic!(
                "role {:?} has no filmHarness block: {asset}",
                reference.role
            )
        });
        let generated = provenance["generated"].as_bool();
        let expected = pack_entry_is_generated(&build.pack, &reference.role);
        assert_eq!(
            generated,
            Some(expected),
            "role {:?} provenance: {provenance}",
            reference.role
        );
        if expected {
            assert_eq!(
                provenance
                    .pointer("/generation/model")
                    .and_then(Value::as_str),
                Some("krea_2_turbo"),
                "role {:?} lost its generation block: {provenance}",
                reference.role
            );
            seen += 1;
        }
    }
    assert!(seen > 0, "the run imported no generated reference");
}

fn pack_entry_is_generated(pack: &ReferencePack, role: &str) -> bool {
    pack.references
        .iter()
        .find(|entry| entry.role == role)
        .is_some_and(|entry| entry.generated)
}

/// Every way one run can fail partway through, on ONE harness: a role with no prompt (refused
/// before anything is dispatched), a job that fails, a job that overruns the spec's own
/// `maxJobSeconds`, and a job whose peak blows the memory budget. Each must name the role and
/// leave `--out` — and the pending directory it would have been renamed from — absent.
///
/// One harness rather than four because each one is an API plus a claiming worker, and this suite
/// runs beside the rest of the film-harness tests, some of which are timing sensitive.
#[tokio::test]
async fn every_mid_run_failure_is_refused_by_name_and_publishes_nothing() {
    let _serial = serialized();
    let harness = Harness::start(true, Vec::new()).await;

    // (1) A role with no prompt: refused before a single job.
    let spec = edited_spec(&harness, "no-prompt", |spec| {
        spec["references"][2]["prompt"] = json!("   ");
    });
    let refused = options(&harness, spec, "no-prompt-out");
    let error = make_references(&harness.transport, &refused)
        .await
        .expect_err("a role with no prompt is refused");
    let HarnessError::Validation(findings) = &error else {
        panic!("expected a validation refusal, got {error}");
    };
    assert!(
        findings.iter().any(|finding| {
            finding.field == "referenceSpec.references[2].prompt"
                && finding.message.contains("red_parcel")
        }),
        "the refusal must name the role: {findings:#?}"
    );
    assert_eq!(
        image_jobs(&harness),
        0,
        "nothing may be dispatched for a spec that cannot be rendered"
    );
    assert_nothing_published(&refused);

    // (2) A job that fails. The THIRD role fails, so two plates are already written into the
    // pending directory when the run gives up — exactly the half-written pack that must not
    // survive.
    harness.script.lock().image_fails = vec!["red_parcel".to_owned()];
    let spec = edited_spec(&harness, "failing", |spec| {
        spec["limits"]["maxAttemptsPerRole"] = json!(1);
    });
    let failing = options(&harness, spec, "failing-out");
    let error = make_references(&harness.transport, &failing)
        .await
        .expect_err("a failed job refuses the run");
    assert!(
        matches!(&error, HarnessError::Refused(message)
            if message.contains("red_parcel") && message.contains("fake image engine fault")),
        "{error}"
    );
    assert_nothing_published(&failing);
    harness.script.lock().image_fails.clear();

    // (3) A job that overruns the spec's own per-job budget: cancelled through the API, refused
    // with the limit named.
    harness.script.lock().image_hangs = vec!["courier".to_owned()];
    let spec = edited_spec(&harness, "hanging", |spec| {
        spec["limits"]["maxJobSeconds"] = json!(1);
        spec["limits"]["maxAttemptsPerRole"] = json!(1);
    });
    let hanging = options(&harness, spec, "hanging-out");
    let error = make_references(&harness.transport, &hanging)
        .await
        .expect_err("a job that overruns the budget refuses the run");
    assert!(
        matches!(&error, HarnessError::Refused(message)
            if message.contains("courier") && message.contains("maxJobSeconds")),
        "{error}"
    );
    assert_nothing_published(&hanging);
    harness.script.lock().image_hangs.clear();

    // (4) A peak over the declared memory budget: 60% of the fake host's 128 GB is 76.8 GB, over
    // the spec's declared 64 GB. Terminal, not retried — a retry re-renders at the same cost.
    harness.script.lock().image_peak_pct = Some(60.0);
    let over_memory = options(&harness, shipped_spec(), "over-memory-out");
    let error = make_references(&harness.transport, &over_memory)
        .await
        .expect_err("an over-budget peak refuses the run");
    assert!(
        matches!(&error, HarnessError::Refused(message)
            if message.contains("courier") && message.contains("64 GB budget")),
        "{error}"
    );
    assert_nothing_published(&over_memory);
}

fn image_jobs(harness: &Harness) -> usize {
    harness
        .script
        .lock()
        .claimed
        .iter()
        .filter(|(kind, _, _)| kind == "image_generate")
        .count()
}

#[tokio::test]
async fn the_catalog_gates_are_checked_before_the_first_render() {
    let _serial = serialized();
    let harness = Harness::start(true, Vec::new()).await;

    // A model the catalog does not hold.
    let spec = edited_spec(&harness, "unknown-model", |spec| {
        spec["model"]["id"] = json!("not_a_model");
    });
    let error = make_references(&harness.transport, &options(&harness, spec, "unknown-out"))
        .await
        .expect_err("an unknown model is refused");
    let HarnessError::Validation(findings) = &error else {
        panic!("expected a validation refusal, got {error}");
    };
    assert!(
        findings
            .iter()
            .any(|finding| finding.field == "referenceSpec.model.id"),
        "{findings:#?}"
    );

    // A negative prompt for a model that declares it does not take one. Krea 2 Turbo is CFG-free:
    // the engine never forwards the text, so a spec that declares one is refused rather than
    // rendered against a prompt the model never saw.
    let spec = edited_spec(&harness, "negative-prompt", |spec| {
        spec["model"]["negativePrompt"] = json!("blurry, text, watermark");
    });
    let error = make_references(&harness.transport, &options(&harness, spec, "negative-out"))
        .await
        .expect_err("a negative prompt for a CFG-free model is refused");
    let HarnessError::Validation(findings) = &error else {
        panic!("expected a validation refusal, got {error}");
    };
    assert!(
        findings.iter().any(|finding| {
            finding.field.ends_with(".negativePrompt")
                && finding.message.contains("supportsNegativePrompt")
        }),
        "{findings:#?}"
    );

    // A geometry the model does not declare.
    let spec = edited_spec(&harness, "geometry", |spec| {
        spec["model"]["resolution"] = json!("999x999");
    });
    let error = make_references(&harness.transport, &options(&harness, spec, "geometry-out"))
        .await
        .expect_err("an undeclared geometry is refused");
    let HarnessError::Validation(findings) = &error else {
        panic!("expected a validation refusal, got {error}");
    };
    assert!(
        findings
            .iter()
            .any(|finding| finding.field.ends_with(".resolution")),
        "{findings:#?}"
    );

    // The install gate, which every other test in this file turns off.
    let mut installed = options(&harness, shipped_spec(), "install-out");
    installed.require_installed = true;
    let error = make_references(&harness.transport, &installed)
        .await
        .expect_err("an uninstalled tier is refused when the gate is on");
    let HarnessError::Validation(findings) = &error else {
        panic!("expected a validation refusal, got {error}");
    };
    assert!(
        findings.iter().any(|finding| {
            finding.field == "referenceSpec.model.tier" && finding.message.contains("not installed")
        }),
        "{findings:#?}"
    );

    assert_eq!(
        image_jobs(&harness),
        0,
        "no gate may be discovered after a render has already been paid for"
    );
}
