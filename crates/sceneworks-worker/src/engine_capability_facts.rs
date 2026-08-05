//! **Stage 1** of the engine-capability pipeline (sc-16965, epic 16948): the engine-linked
//! dumper that turns the *linked* provider registry into a checked-in, engine-keyed facts file.
//!
//! # Why a dumper and not a serve-time lookup
//!
//! `Capabilities::supports_preview` (gen-core, sc-16677) is the only weights-free way to tell
//! "this route cannot live-preview" from "it can, but no frame has arrived yet". But the process
//! that serves `GET /api/v1/models` is **not** always the process that links an engine:
//!
//! - **Desktop** builds `sceneworks-rust-api --features embed-web,backend-candle` — registry
//!   in-process.
//! - **Docker / RunPod** builds `sceneworks-rust-api --features embed-web` with **no**
//!   `backend-candle`, while the *worker* image is built with it. Separate images.
//!
//! [`crate::inference_runtime::media`] returns an **empty** `ProviderRegistry` unless
//! `target_os = "macos"` or `feature = "backend-candle"`, so a serve-time derivation would answer
//! truthfully on desktop and report "nothing supports preview" on every server — a *wrong* answer,
//! not a missing one, in the topology least exercised locally. `sceneworks-core`, which owns the
//! manifests, has no engine dependency at all and can never derive.
//!
//! # Two stages
//!
//! 1. **This module** walks `registry.generators()` weights-free and writes one checked-in facts
//!    file *per backend* — `capabilities.candle.json` from a `backend-candle` lane,
//!    `capabilities.mlx.json` from the macOS lane. One file per backend means two lanes never
//!    rewrite the same file. These are the analogue of `documents/style.txt`: a checked-in source.
//! 2. `apps/web/scripts/generate-preview-support.mjs` derives the served catalog block from those
//!    files, joined to SceneWorks model ids through [`crate::engines::MODEL_TABLE`], and
//!    `apps/web/src/data/previewSupportCatalog.test.js` re-derives it as a drift guard.
//!
//! Stage 2 depends only on checked-in files, so its guard runs on **every PR**. Stage 1 needs a
//! linked registry, which exists only on macOS and `backend-candle` lanes — so *writing* a facts
//! file stays a manual step run on the matching box.
//!
//! **Both files are nevertheless VERIFIED on every PR** that can invalidate them. `macos-mlx.yml`
//! re-dumps to a scratch dir and diffs `capabilities.mlx.json` against it (sc-17119); the
//! self-hosted CUDA lane `windows-candle.yml` does the same for `capabilities.candle.json` under
//! `shell: cmd` (sc-17592). Both lanes watch `config/engine-capabilities/**`, so the one edit these
//! guards exist to catch — a **restamp**, where the `inferenceRevision` line is rewritten and the
//! engine list is left stale — cannot slip through by touching that file alone. Without those steps
//! every remaining guard reads this file as a *source* and is satisfied by editing one line. (The
//! other candle lanes still verify nothing: `desktop-windows.yml` builds the sidecar on main/release
//! only, and `server-candle-linux.yml` is `workflow_dispatch`-only.)
//!
//! **Nor was the dispatch-only era an unguarded gap.** sc-16951's `candle-gen-catalog` bidirectional
//! test runs in the *inference* repo's CI on every PR, so descriptor-level truth is guarded
//! continuously upstream. SceneWorks only needs to re-dump when the descriptors can have moved —
//! i.e. at an **inference pin bump**, where `scripts/bump-inference.mjs` fails closed on a stale
//! facts file, beside the licence re-scan in the same generated-artifact cascade.
//!
//! # Running it
//!
//! ```text
//! cargo run -p sceneworks-worker --bin dump-engine-capabilities \
//!     --no-default-features --features backend-candle     # Windows / Linux CUDA → capabilities.candle.json
//! cargo run -p sceneworks-worker --bin dump-engine-capabilities   # macOS → capabilities.mlx.json
//! ```
//!
//! # The vacuous-green trap
//!
//! [`facts_from_descriptors`] **refuses** to produce anything from an empty descriptor list. On a
//! lane without engines the dumper would otherwise emit a valid, empty, entirely wrong facts file,
//! and stage 2 would happily derive "no route supports preview" from it — the same failure class as
//! a candle CI smoke that proves PASS rather than kill. The assertion is on the pure function, not
//! on the I/O wrapper, so it is unit-testable on **every** lane including the ones with no engines.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Every backend SceneWorks can link a **media** registry for, and therefore every backend stage 2
/// requires a checked-in facts file for (sc-17119).
///
/// Scoped to [`crate::inference_runtime::media`] because that is the only registry this module
/// dumps. [`crate::inference_runtime::audio`] is a separate registry whose descriptors are dumped
/// **nowhere** — its `supports_preview` is unknown to the catalog on every platform. That gap is not
/// closed here and is not visible to this list: audio is candle-native everywhere, so backend-level
/// coverage is satisfied coincidentally rather than because anything checked. See sc-17593.
///
/// Without this list, "which backends are covered?" could only be answered by *the files on disk* —
/// so a backend that was **never dumped** passed every guard forever, silently. That is not
/// hypothetical: `capabilities.mlx.json` was absent for four consecutive pins beginning with the one
/// sc-16965 itself shipped on (`d4802320`, `bf06bb56`, `5b6d6aa0`, `5f973a73`), and the vitest drift
/// guard and the generator — which both glob this directory — were green throughout, as
/// `verifyEngineCapabilityFacts` would have been had anything invoked it. Absence rendered as
/// "unknown", which renders exactly as before — i.e. it looked fine.
///
/// This is the **union across both lanes**, so no single build can verify it whole: a macOS build
/// links only `mlx`, a `backend-candle` build only `candle` (see [`crate::inference_runtime::media`],
/// whose two cfg arms are the definition this list mirrors). What each lane *can* enforce is that
/// the backend it actually links is declared here — see `the_linked_backend_is_declared`. Between
/// the two lanes every entry is covered by a real registry.
///
/// Sorted and unique; stage 2 compares it against the sorted set of dumped files.
pub const SCENEWORKS_BACKENDS: &[&str] = &["candle", "mlx"];

/// One engine's weights-free preview facts, as written to a per-backend facts file.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EngineFact {
    /// The gen-core registry id (`ModelDescriptor::id`) — the join key stage 2 resolves
    /// `MODEL_TABLE.engine_id` against.
    pub id: String,
    /// `"image"` | `"video"` | `"both"` | `"audio"` (`ModelDescriptor::modality`).
    pub modality: String,
    /// `Capabilities::supports_preview` — whether this engine emits `PreviewSink` frames.
    pub supports_preview: bool,
}

/// Provenance stamped onto every facts file so a stale dump is detectable without rebuilding.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FactsProvenance {
    /// The inference revision the descriptors were read from — the same constant
    /// `scripts/bump-inference.mjs` repins, so a bump makes every facts file stale by construction
    /// and `verifyEngineCapabilityFacts` in that script says so.
    pub inference_revision: String,
    /// How to reproduce this file.
    pub dumper: String,
}

/// One backend's complete, weights-free engine facts.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EngineCapabilityFacts {
    /// `"mlx"` | `"candle"` — `ModelDescriptor::backend`.
    pub backend: String,
    pub generated_from: FactsProvenance,
    /// Every generator this backend registered, sorted by id so the file is byte-stable across runs.
    pub engines: Vec<EngineFact>,
}

impl EngineCapabilityFacts {
    /// `capabilities.<backend>.json` — the file name this backend owns. One file per backend is
    /// what keeps the macOS and candle lanes from overwriting each other's dump.
    pub fn file_name(&self) -> String {
        format!("capabilities.{}.json", self.backend)
    }
}

fn modality_label(modality: &gen_core::Modality) -> &'static str {
    // Exhaustive on purpose: a new gen-core modality must break this build loudly at the next pin
    // bump rather than be silently flattened into a catch-all label.
    match modality {
        gen_core::Modality::Image => "image",
        gen_core::Modality::Video => "video",
        gen_core::Modality::Both => "both",
        gen_core::Modality::Audio => "audio",
    }
}

/// Group descriptors into one [`EngineCapabilityFacts`] per backend.
///
/// Pure — no registry, no filesystem — so the empty-registry refusal below is exercised on lanes
/// that have no engines at all, which is exactly where the mistake it prevents would be made.
///
/// # Errors
///
/// - the descriptor list is **empty** (the vacuous-green trap: an empty registry must never be
///   mistaken for "no engine supports preview");
/// - a backend registered the same engine id twice, which would make the stage-2 join ambiguous.
pub fn facts_from_descriptors(
    descriptors: &[gen_core::ModelDescriptor],
    inference_revision: &str,
    dumper: &str,
) -> Result<Vec<EngineCapabilityFacts>, String> {
    if descriptors.is_empty() {
        return Err(
            "refusing to write engine-capability facts: the provider registry is EMPTY. This lane \
             links no engines, so the dump would be a valid, empty, entirely wrong facts file that \
             stage 2 would read as \"no route supports live preview\". Re-run on a lane that links \
             a registry: macOS (mlx), or off-Mac with --no-default-features --features \
             backend-candle."
                .to_owned(),
        );
    }

    let mut by_backend: BTreeMap<String, Vec<EngineFact>> = BTreeMap::new();
    for descriptor in descriptors {
        by_backend
            .entry(descriptor.backend.to_owned())
            .or_default()
            .push(EngineFact {
                id: descriptor.id.to_owned(),
                modality: modality_label(&descriptor.modality).to_owned(),
                supports_preview: descriptor.capabilities.supports_preview,
            });
    }

    let mut out = Vec::with_capacity(by_backend.len());
    for (backend, mut engines) in by_backend {
        engines.sort_by(|left, right| left.id.cmp(&right.id));
        if let Some(duplicate) = engines
            .windows(2)
            .find(|pair| pair[0].id == pair[1].id)
            .map(|pair| pair[0].id.clone())
        {
            return Err(format!(
                "backend {backend:?} registered engine id {duplicate:?} more than once; the \
                 stage-2 join through MODEL_TABLE.engine_id would be ambiguous"
            ));
        }
        out.push(EngineCapabilityFacts {
            backend,
            generated_from: FactsProvenance {
                inference_revision: inference_revision.to_owned(),
                dumper: dumper.to_owned(),
            },
            engines,
        });
    }
    Ok(out)
}

/// How this build's facts files are reproduced, named in the file itself.
fn dumper_invocation() -> &'static str {
    if cfg!(target_os = "macos") {
        "cargo run -p sceneworks-worker --bin dump-engine-capabilities"
    } else {
        "cargo run -p sceneworks-worker --bin dump-engine-capabilities --no-default-features \
         --features backend-candle"
    }
}

/// Walk the linked provider registry weights-free and group it per backend.
///
/// Reads only each registration's `descriptor` closure — no model load, no weights on disk — the
/// same introspection `mlx-gen-catalog` and [`crate::engines::registry_capabilities`] do.
pub fn collect_engine_capability_facts() -> Result<Vec<EngineCapabilityFacts>, String> {
    let descriptors: Vec<gen_core::ModelDescriptor> = crate::inference_runtime::media()
        .generators()
        .map(|registration| (registration.descriptor)())
        .collect();
    facts_from_descriptors(
        &descriptors,
        crate::catalog_semantic_jobs::INFERENCE_RUNTIME_REVISION,
        dumper_invocation(),
    )
}

/// Serialize one backend's facts as the exact bytes checked in (pretty JSON + trailing newline),
/// matching `apps/web/scripts/generate-styles.mjs`'s `JSON.stringify(value, null, 2) + "\n"` so a
/// JS-side rewrite and a Rust-side dump agree byte-for-byte.
pub fn facts_json(facts: &EngineCapabilityFacts) -> Result<String, String> {
    let mut json = serde_json::to_string_pretty(facts)
        .map_err(|error| format!("engine capability facts do not serialize: {error}"))?;
    json.push('\n');
    Ok(json)
}

/// The checked-in facts directory, resolved from this crate's manifest dir so the dumper works from
/// any cwd: `<repo>/config/engine-capabilities`.
///
/// Under `config/`, which `docker/rust.Dockerfile` `COPY config ./config`s wholesale — so nothing
/// here needs a Dockerfile line, unlike an `include_str!` reaching outside a crate.
pub fn default_facts_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("config")
        .join("engine-capabilities")
}

/// Dump this build's facts files into `dir`, returning the paths written.
///
/// Writes **nothing** when [`collect_engine_capability_facts`] refuses, so a lane with no engines
/// cannot leave a wrong file behind.
pub fn dump_to(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let facts = collect_engine_capability_facts()?;
    std::fs::create_dir_all(dir)
        .map_err(|error| format!("cannot create {}: {error}", dir.display()))?;
    let mut written = Vec::with_capacity(facts.len());
    for entry in &facts {
        let path = dir.join(entry.file_name());
        std::fs::write(&path, facts_json(entry)?)
            .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
        written.push(path);
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor(
        id: &'static str,
        backend: &'static str,
        supports_preview: bool,
    ) -> gen_core::ModelDescriptor {
        gen_core::ModelDescriptor {
            id,
            family: id,
            backend,
            modality: gen_core::Modality::Image,
            capabilities: gen_core::Capabilities {
                supports_preview,
                ..Default::default()
            },
            required_components: &[],
            control_kinds: None,
        }
    }

    // The vacuous-green trap (epic 16948 decision, activity-16974). An empty registry is the
    // *normal* state on the plain Linux / non-candle lane, so a dumper that happily wrote an empty
    // file there would ship "no route supports live preview" as fact. Assert the refusal on the
    // pure function so this test runs on EVERY lane — including the ones with no engines, which is
    // precisely where the vacuous file would be produced.
    #[test]
    fn refuses_to_dump_an_empty_registry() {
        let error = facts_from_descriptors(&[], "d48023204cd3a4f3f8eb060f79803dccaddcb482", "dump")
            .expect_err("an empty descriptor list must not produce facts");
        assert!(
            error.contains("EMPTY"),
            "the refusal must name the empty registry so the operator knows the lane is wrong, got: {error}"
        );
        assert!(
            error.contains("backend-candle"),
            "the refusal must name a lane that CAN dump, got: {error}"
        );
    }

    #[test]
    fn groups_by_backend_and_sorts_engines_by_id() {
        let facts = facts_from_descriptors(
            &[
                descriptor("z_image_turbo", "candle", false),
                descriptor("krea_2_turbo", "candle", true),
                descriptor("krea_2_turbo", "mlx", true),
            ],
            "d48023204cd3a4f3f8eb060f79803dccaddcb482",
            "dump",
        )
        .expect("non-empty descriptors produce facts");

        assert_eq!(
            facts.iter().map(|f| f.backend.as_str()).collect::<Vec<_>>(),
            ["candle", "mlx"],
            "one facts file per backend, in a stable order"
        );
        assert_eq!(
            facts[0]
                .engines
                .iter()
                .map(|e| e.id.as_str())
                .collect::<Vec<_>>(),
            ["krea_2_turbo", "z_image_turbo"],
            "engines are sorted by id so the checked-in file is byte-stable"
        );
        assert!(facts[0].engines[0].supports_preview);
        assert!(!facts[0].engines[1].supports_preview);
        assert_eq!(facts[0].file_name(), "capabilities.candle.json");
        assert_eq!(facts[1].file_name(), "capabilities.mlx.json");
    }

    // The same engine id twice in one backend would make the stage-2 join through
    // MODEL_TABLE.engine_id resolve to whichever row won — a silently wrong answer for every
    // SceneWorks id mapping onto it.
    #[test]
    fn rejects_a_duplicate_engine_id_within_one_backend() {
        let error = facts_from_descriptors(
            &[
                descriptor("krea_2_turbo", "candle", true),
                descriptor("krea_2_turbo", "candle", false),
            ],
            "d48023204cd3a4f3f8eb060f79803dccaddcb482",
            "dump",
        )
        .expect_err("a duplicate engine id must be refused");
        assert!(error.contains("krea_2_turbo"), "got: {error}");
    }

    // Stage 2 compares SCENEWORKS_BACKENDS against the sorted list of dumped files, so an unsorted
    // or duplicated entry would fail the coverage check for a reason that has nothing to do with a
    // missing dump.
    #[test]
    fn sceneworks_backends_is_sorted_and_unique() {
        let mut sorted = SCENEWORKS_BACKENDS.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted, SCENEWORKS_BACKENDS,
            "SCENEWORKS_BACKENDS must be sorted and duplicate-free"
        );
        assert!(
            !SCENEWORKS_BACKENDS.is_empty(),
            "an empty backend list would make the stage-2 coverage assertion vacuous — the exact \
             class of hole it exists to close"
        );
    }

    // The half of SCENEWORKS_BACKENDS this lane can actually measure. The list is a union across
    // two mutually exclusive lanes, so neither build sees it whole; what each build proves is that
    // the backend IT links is declared. An undeclared backend would ship with no stage-2 coverage
    // assertion at all — a third engine could be added, never dumped, and stay invisible exactly
    // the way `mlx` did.
    #[cfg(any(target_os = "macos", feature = "backend-candle"))]
    #[test]
    fn the_linked_backend_is_declared() {
        let facts = collect_engine_capability_facts()
            .expect("this lane links a media registry, so collecting facts must succeed");
        assert!(
            !facts.is_empty(),
            "a lane that links a registry must produce at least one backend's facts"
        );
        for entry in &facts {
            assert!(
                SCENEWORKS_BACKENDS.contains(&entry.backend.as_str()),
                "this build links backend {:?}, which SCENEWORKS_BACKENDS ({:?}) does not declare. \
                 Stage 2 requires one dumped facts file per DECLARED backend, so an undeclared one \
                 is covered by nothing. Add it to the list and dump {} on this lane.",
                entry.backend,
                SCENEWORKS_BACKENDS,
                entry.file_name(),
            );
        }
    }

    #[test]
    fn serializes_as_camel_case_json_with_a_trailing_newline() {
        let facts = facts_from_descriptors(
            &[descriptor("krea_2_turbo", "candle", true)],
            "d48023204cd3a4f3f8eb060f79803dccaddcb482",
            "cargo run …",
        )
        .expect("facts");
        let json = facts_json(&facts[0]).expect("serializes");
        assert!(json.ends_with("}\n"), "one trailing newline, got: {json:?}");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("parses");
        assert_eq!(parsed["backend"], "candle");
        assert_eq!(
            parsed["generatedFrom"]["inferenceRevision"],
            "d48023204cd3a4f3f8eb060f79803dccaddcb482"
        );
        assert_eq!(parsed["engines"][0]["id"], "krea_2_turbo");
        assert_eq!(parsed["engines"][0]["supportsPreview"], true);
        assert_eq!(parsed["engines"][0]["modality"], "image");
    }
}
