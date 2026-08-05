//! The SC-15833 FLUX.2 inference-compatibility audit, validated against frozen expectations.
//!
//! sc-17497: this lives OUTSIDE `candle_memory_strategy` on purpose. That module compiles only under
//! `all(not(target_os = "macos"), feature = "backend-candle")`, and a `cargo test` of that lane
//! links libcuda — so its tests first execute on CI, which is how sc-12306 shipped a candle test
//! asserting the opposite of its intent. This validation is pure JSON logic with no CUDA dependency,
//! so it is compiled (and its tests run) on every platform instead.

use serde_json::Value;

pub(crate) const FLUX2_CAPTURED_INFERENCE_REVISION: &str =
    "5ffd7612e7de4e76b6db00a7148ed3d9c15b4c0d";
pub(crate) const FLUX2_COMPATIBLE_INFERENCE_REVISION: &str =
    "a4f409ae8ce73eda2ee8117b89b5f479666606b8";
pub(crate) const FLUX2_INFERENCE_COMPATIBILITY_AUDIT: &str =
    include_str!("../../../docs/calibration/sc-15833/inference-compatibility-a4f4.json");
/// sc-17607: v5 is the NINE-path closure — exactly what the measurement binary compiles, plus the
/// workspace inputs to that compile — carrying sc-17606's shipped-feature witness.
///
/// Every older version is refused rather than re-graded, and the refusals run in both directions.
/// v1 and v2 describe sc-15833's seven-path closure, which omits `Cargo.lock`,
/// `rust-toolchain.toml` and `.cargo/config.toml`; v3 has the ten-path closure but never looked at
/// how the shipped bundle resolves features — accepting either would read it as evidence about
/// inputs it never audited. v4 is the opposite case and is refused just as firmly: it audited
/// `crates/bundles/runtime-cuda`, a path no build of the audited target can adjudicate and one this
/// schema deliberately stopped asking about, so accepting one means dropping an entry out of
/// someone else's record to make it fit — the same unearned re-reading, by subtraction.
pub(crate) const FLUX2_AUDIT_SCHEMA_VERSION: u64 = 5;
pub(crate) const FLUX2_V5_AUDIT_METHOD: &str = concat!(
    "compiled artifact identity for changed paths, git object identity for unchanged paths, ",
    "and shipped-bundle resolved-feature identity, across the Candle FLUX.2 measurement binary's ",
    "compile closure and its workspace build inputs"
);
/// sc-17497: the compiled-artifact proof, frozen here so the packaged record cannot authorize
/// itself. `None` means every closure object is byte-identical at the live pin and object identity
/// still decides on its own; `Some((digest, adjudicates))` is required the moment one of them moves.
///
/// `adjudicates` is the closure subset the audited binary can speak for, and it is load-bearing:
/// `runtime-cuda` depends on `candle-gen-flux2`, not the reverse, so a commit into the CUDA bundle
/// leaves the measurement binary byte-identical. Without this the unchanged digest would be read as
/// proof over a path that was never compiled into it.
///
/// sc-17524: it also carries the four workspace build inputs, which cargo can never name because
/// they are not packages. They belong there for a different reason than the crate trees — not
/// "compiled into the binary" but "an input to the build that produced it", which is the only route
/// they have to the measured code at all.
///
/// See `scripts/inference-artifact-audit.mjs` and `docs/inference-artifact-audit-sc-17497.md`.
pub(crate) const FLUX2_AUDIT_ARTIFACT_PROOF: Option<(&str, &[&str])> = Some((
    "sha256:d80844f24dcb95f957c1cd893f9238c9d753db8e1e40c5deefe9f6b6f740f9aa",
    &[
        "Cargo.toml",
        "Cargo.lock",
        "rust-toolchain.toml",
        ".cargo/config.toml",
        "crates/contracts/gen-core",
        "crates/media/candle-gen/candle-gen",
        "crates/media/candle-gen/candle-gen-pid",
        "crates/media/candle-gen/candle-gen-flux2",
        "crates/media/candle-gen/vendor/candle-kernels",
    ],
));
/// What the resolved-feature witness has to describe for its digest to mean anything (sc-17606).
///
/// The digest alone is not enough. A witness taken over another bundle, another target triple, or
/// with dev edges folded in is an answer to a different question, and nothing about a bare sha256
/// says which question it answered.
pub(crate) struct FeatureWitnessExpectation<'a> {
    pub(crate) digest: &'a str,
    pub(crate) shipped_package: &'a str,
    pub(crate) scope_root: &'a str,
    pub(crate) scope_features: &'a str,
    pub(crate) target: &'a str,
    pub(crate) edges: &'a str,
}

/// sc-17606: the shipped-bundle feature resolution, frozen here so the record cannot authorize
/// itself — the same bargain `FLUX2_AUDIT_ARTIFACT_PROOF` makes.
///
/// This layer exists because the other two cannot see it. The audited binary is built
/// `-p candle-gen-flux2`, which unifies features over a strictly smaller graph than the shipped
/// `-p runtime-cuda` bundle does; a crate outside FLUX.2's closure — `candle-llm`, the audio lane,
/// another provider — can enable a feature on a dependency FLUX.2 SHARES and change what it
/// executes in the shipped runtime. Feature flags are not in `Cargo.lock`, so widening the closure
/// to the lockfile did not catch it either: every audited object stays byte-identical, the artifact
/// digest stays byte-identical, and the record takes the free path.
///
/// Which is why, unlike the artifact proof, this is never `None`. The change it exists to catch
/// arrives exclusively on the path where nothing else runs.
pub(crate) const FLUX2_FEATURE_WITNESS: FeatureWitnessExpectation<'static> =
    FeatureWitnessExpectation {
        digest: "sha256:321625ed01f837fd0682b2020d92cde068e8792db103d4e8ba3bf3ff8bff500b",
        shipped_package: "runtime-cuda",
        scope_root: "candle-gen-flux2",
        scope_features: "cuda",
        target: "x86_64-pc-windows-msvc",
        edges: "normal,build",
    };

/// The CAPTURED side of the closure — the code the measurements were taken against. Immutable.
///
/// sc-17524 added three more workspace build inputs, bringing that kind to four. They are not
/// packages, so cargo's
/// `compiler-artifact` stream can never name them, but they feed every build in the closure — and
/// `Cargo.lock` had already moved inside the authorized window while all seven crate trees stayed
/// byte-identical, which is a changed build input sailing through the free path unexamined.
///
/// sc-17607 removed `crates/bundles/runtime-cuda`. It is not compiled into the measurement binary,
/// so nothing here could ever clear it; what a bundle edit actually raises is whether `flux2_dev`
/// is still registered by `candle-gen-flux2`, which `flux2_composition_audit.rs` asks of the LINKED
/// bundle instead. Its sibling `candle-gen-catalog` — equally "linked by the worker", and in no
/// list at all until that story — is covered there too. The FEATURE half of what the bundle's own
/// manifest could change is covered by `FLUX2_FEATURE_WITNESS`. (Named as a file rather than linked
/// as an item: that module is candle-lane-only, and an intra-doc link to it breaks rustdoc
/// everywhere else.)
pub(crate) const FLUX2_AUDITED_OBJECTS: [(&str, &str); 9] = [
    ("Cargo.toml", "8f5af6b9d53bbfe3be5d9d79b8949364138a087c"),
    ("Cargo.lock", "8ab01e00f01607a99845d875ed60275ae033450c"),
    (
        "rust-toolchain.toml",
        "ae829f875c68c03c367ce92cc05e041036a92d0a",
    ),
    (
        ".cargo/config.toml",
        "61d7be37632a60aea10dc3c25b8ad5bec0a5fa45",
    ),
    (
        "crates/contracts/gen-core",
        "9a7e86f5893e584a8d0d656147abc4ae93af6922",
    ),
    (
        "crates/media/candle-gen/candle-gen",
        "e8b8b3f0787fac49539a2ef1085c48c9fdc9ec57",
    ),
    (
        "crates/media/candle-gen/candle-gen-pid",
        "f3c8db10f1a872fc8fdb2c7243e607591886a5fa",
    ),
    (
        "crates/media/candle-gen/candle-gen-flux2",
        "f91cd1a302f0d27f82bbc9c60bd4e578390e44b1",
    ),
    (
        "crates/media/candle-gen/vendor/candle-kernels",
        "3b8327cf01d346c8068a5e9d096dcdddca440e99",
    ),
];

/// The packaged closure audit, validated against the frozen expectations.
///
/// sc-17497: `body` and `expected_proof` are parameters rather than the constants directly so the
/// tests can drive the ACCEPTING side of the compiled-artifact layer while the shipped constant is
/// still `None`. A validator only ever exercised against its own default passes with the feature
/// removed.
pub(crate) fn compatibility_audit_authorizes(
    body: &str,
    expected_proof: Option<(&str, &[&str])>,
    expected_witness: &FeatureWitnessExpectation<'_>,
) -> Option<()> {
    let audit: Value = serde_json::from_str(body).ok()?;
    if audit.get("schemaVersion")?.as_u64()? != FLUX2_AUDIT_SCHEMA_VERSION
        || audit.get("story")?.as_str()? != "SC-15833"
        || audit.get("capturedInferenceRevision")?.as_str()? != FLUX2_CAPTURED_INFERENCE_REVISION
        || audit.get("compatibleInferenceRevision")?.as_str()?
            != FLUX2_COMPATIBLE_INFERENCE_REVISION
        || audit.get("method")?.as_str()? != FLUX2_V5_AUDIT_METHOD
        // JS requires this too; a record one language accepts and the other rejects is a split brain.
        || audit.get("command")?.as_str().is_none()
    {
        return None;
    }
    feature_witness_authorizes(&audit, expected_witness)?;
    let objects = audit.get("auditedObjects")?.as_array()?;
    if objects.len() != FLUX2_AUDITED_OBJECTS.len() {
        return None;
    }
    // sc-17497: only the captured side is pinned. A moved compatible object is adjudicated by the
    // compiled-artifact proof below, not rejected outright — that rejection is what turned a
    // doc-comment commit into a 47.6 GB re-capture.
    let mut changed = Vec::new();
    for (path, object_id) in FLUX2_AUDITED_OBJECTS {
        let matching = objects
            .iter()
            .filter(|entry| entry.get("path").and_then(Value::as_str) == Some(path))
            .collect::<Vec<_>>();
        if matching.len() != 1
            || matching[0].get("capturedObject").and_then(Value::as_str) != Some(object_id)
        {
            return None;
        }
        let compatible = matching[0]
            .get("compatibleObject")
            .and_then(Value::as_str)?;
        if compatible.len() != 40 || !compatible.bytes().all(|b| b.is_ascii_hexdigit()) {
            return None;
        }
        if compatible != object_id {
            changed.push(path);
        }
    }
    let declared = audit.get("changedClosurePaths")?.as_array()?;
    let mut declared = declared
        .iter()
        .map(Value::as_str)
        .collect::<Option<Vec<_>>>()?;
    declared.sort_unstable();
    let mut observed = changed.clone();
    observed.sort_unstable();
    if declared != observed {
        return None;
    }
    match (changed.is_empty(), expected_proof) {
        // Nothing moved: object identity carries the proof on its own, and a digest must not be
        // expected — a stale one left behind would silently keep demanding a build that is not due.
        (true, None) => {}
        (false, Some((digest, adjudicates))) => {
            let artifact = audit.get("auditedArtifact")?.as_object()?;
            let (algorithm, hex) = digest.split_once(':')?;
            if algorithm != "sha256"
                || hex.len() != 64
                || !hex.bytes().all(|byte| byte.is_ascii_hexdigit())
            {
                return None;
            }
            // Metal builds the same closure and is NOT proof of the CUDA artifact the capture ran.
            if artifact.get("lane").and_then(Value::as_str) != Some("cuda")
                || artifact.get("package").and_then(Value::as_str) != Some("candle-gen-flux2")
                || artifact.get("test").and_then(Value::as_str)
                    != Some("tests::flux2_dev_probed_generate_for_offload_ab")
                || artifact.get("profile").and_then(Value::as_str) != Some("release")
                || artifact.get("capturedDigest").and_then(Value::as_str) != Some(digest)
                || artifact.get("compatibleDigest").and_then(Value::as_str) != Some(digest)
            {
                return None;
            }
            // The digest speaks only for what the binary linked; anything else that moved is
            // unproven, and silently accepting it would be strictly worse than the false positive
            // this story removes.
            //
            // Intersected with the RECORD's own `adjudicates`, not just the frozen set. The frozen
            // half is a human transcription, and unlike the digest — where a typo fails closed — an
            // over-wide set fails OPEN. Intersecting lets the record narrow the claim, never widen
            // it, so both halves must be wrong the same way to do harm.
            let recorded = artifact.get("adjudicates")?.as_array()?;
            let recorded = recorded
                .iter()
                .map(Value::as_str)
                .collect::<Option<Vec<_>>>()?;
            if changed
                .iter()
                .any(|path| !adjudicates.contains(path) || !recorded.contains(path))
            {
                return None;
            }
        }
        _ => return None,
    }
    Some(())
}

/// The resolved-feature half of a v4 record (sc-17606).
///
/// Unconditional, and that is the design rather than an oversight. The artifact proof is demanded
/// exactly when a closure object moved; the gap this closes moves NONE of them, so it can only ever
/// present itself on a record whose `changedClosurePaths` is empty. Gating the witness on a build
/// would have checked it on precisely the runs where the thing it looks for cannot be.
///
/// `packages` must be a positive integer for one specific false green: a witness computed over an
/// EMPTY resolution hashes to a stable digest that is identical at every revision. The emitting
/// script refuses to produce one; this refuses to accept a hand-written one.
fn feature_witness_authorizes(
    audit: &Value,
    expected: &FeatureWitnessExpectation<'_>,
) -> Option<()> {
    let witness = audit.get("featureWitness")?.as_object()?;
    let (algorithm, hex) = expected.digest.split_once(':')?;
    if algorithm != "sha256" || hex.len() != 64 || !hex.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return None;
    }
    let field = |key: &str| witness.get(key).and_then(Value::as_str);
    if field("capturedDigest") != Some(expected.digest)
        || field("compatibleDigest") != Some(expected.digest)
        // The resolution IS the question. Another bundle, another target triple or dev edges folded
        // in answer something else, and their digests would compare on perfectly equal terms.
        || field("shippedPackage") != Some(expected.shipped_package)
        || field("scopeRoot") != Some(expected.scope_root)
        // `cuda`, not `metal`: the provider's own feature selection is half of which code the
        // bundle compiles for it, and an ungraded declaration beside a graded digest reads as
        // though both were checked.
        || field("scopeFeatures") != Some(expected.scope_features)
        || field("target") != Some(expected.target)
        || field("edges") != Some(expected.edges)
        || witness.get("packages").and_then(Value::as_u64).unwrap_or(0) == 0
    {
        return None;
    }
    Some(())
}

#[cfg(test)]
mod sc_17497_artifact_audit_tests {
    use super::*;
    use serde_json::json;

    const CANDLE_GEN: &str = "crates/media/candle-gen/candle-gen";
    /// The two crates above the provider, in the closure no longer (sc-17607) and never in it.
    const COMPOSITION_ONLY: [&str; 2] = [
        "crates/bundles/runtime-cuda",
        "crates/media/candle-gen/candle-gen-catalog",
    ];
    const CARGO_LOCK: &str = "Cargo.lock";
    const RUST_TOOLCHAIN: &str = "rust-toolchain.toml";
    const CARGO_CONFIG: &str = ".cargo/config.toml";
    const DIGEST: &str = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    /// What the `candle-gen-flux2` lib test binary speaks for. The crate trees are the ones it
    /// COMPILES; the four workspace build inputs are there because they are inputs to that very
    /// build, so anything they change about the measured code lands in its digest. Since sc-17607
    /// this is the whole closure — no member is left that the binary cannot answer for.
    const ADJUDICATES: &[&str] = &[
        "Cargo.toml",
        "Cargo.lock",
        "rust-toolchain.toml",
        ".cargo/config.toml",
        "crates/contracts/gen-core",
        "crates/media/candle-gen/candle-gen",
        "crates/media/candle-gen/candle-gen-pid",
        "crates/media/candle-gen/candle-gen-flux2",
        "crates/media/candle-gen/vendor/candle-kernels",
    ];
    const PROOF: Option<(&str, &[&str])> = Some((DIGEST, ADJUDICATES));
    const WITNESS_DIGEST: &str =
        "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    /// The witness expectation the fixtures are graded against — deliberately NOT the shipped
    /// constant, so these tests cannot pass by agreeing with whatever happens to be frozen.
    const WITNESS: FeatureWitnessExpectation<'static> = FeatureWitnessExpectation {
        digest: WITNESS_DIGEST,
        shipped_package: "runtime-cuda",
        scope_root: "candle-gen-flux2",
        scope_features: "cuda",
        target: "x86_64-pc-windows-msvc",
        edges: "normal,build",
    };

    fn feature_witness() -> Value {
        json!({
            "shippedPackage": "runtime-cuda",
            "scopeRoot": "candle-gen-flux2",
            "scopeFeatures": "cuda",
            "target": "x86_64-pc-windows-msvc",
            "edges": "normal,build",
            "packages": 179,
            "measurementDelta": [],
            "capturedDigest": WITNESS_DIGEST,
            "compatibleDigest": WITNESS_DIGEST,
        })
    }

    /// A v5 record in which `moved` closure paths carry a different compatible object.
    ///
    /// A `moved` entry that is not a closure path would silently do nothing here, so callers may
    /// only name real ones — asserted rather than trusted, because a typo would otherwise turn a
    /// rejection test green for the wrong reason.
    fn v5(moved: &[&str], artifact: Option<Value>) -> String {
        for path in moved {
            assert!(
                FLUX2_AUDITED_OBJECTS.iter().any(|(known, _)| known == path),
                "{path} is not a closure path, so moving it in a fixture proves nothing"
            );
        }
        let objects = FLUX2_AUDITED_OBJECTS
            .iter()
            .map(|(path, object_id)| {
                let compatible = if moved.contains(path) {
                    "e".repeat(40)
                } else {
                    (*object_id).to_owned()
                };
                json!({ "path": path, "capturedObject": object_id, "compatibleObject": compatible })
            })
            .collect::<Vec<_>>();
        let mut record = json!({
            "schemaVersion": FLUX2_AUDIT_SCHEMA_VERSION,
            "story": "SC-15833",
            "capturedInferenceRevision": FLUX2_CAPTURED_INFERENCE_REVISION,
            "compatibleInferenceRevision": FLUX2_COMPATIBLE_INFERENCE_REVISION,
            "method": FLUX2_V5_AUDIT_METHOD,
            "command": "node scripts/inference-artifact-audit.mjs",
            "changedClosurePaths": moved,
            "auditedObjects": objects,
            "featureWitness": feature_witness(),
        });
        if let Some(artifact) = artifact {
            record["auditedArtifact"] = artifact;
        }
        record.to_string()
    }

    fn cuda_artifact() -> Value {
        json!({
            "package": "candle-gen-flux2",
            "kind": "lib test binary",
            "test": "tests::flux2_dev_probed_generate_for_offload_ab",
            "profile": "release",
            "lane": "cuda",
            "adjudicates": ADJUDICATES,
            "capturedDigest": DIGEST,
            "compatibleDigest": DIGEST,
        })
    }

    #[test]
    fn the_shipped_record_authorizes_its_own_pin_only_with_the_frozen_proof() {
        // sc-17524: the packaged record's closure is NOT quiet. `Cargo.lock` and `candle-gen`
        // both moved between the captured revision and the live pin, so this record
        // authorizes only in company with the compiled-artifact proof frozen above — and the
        // negative half is the load-bearing one, because a validator that accepts the shipped
        // record no matter what is frozen would pass with the whole artifact layer deleted.
        assert!(
            compatibility_audit_authorizes(
                FLUX2_INFERENCE_COMPATIBILITY_AUDIT,
                FLUX2_AUDIT_ARTIFACT_PROOF,
                &FLUX2_FEATURE_WITNESS
            )
            .is_some(),
            "the shipped SC-15833 audit must authorize its own pin"
        );
        assert!(
            FLUX2_AUDIT_ARTIFACT_PROOF.is_some(),
            "closure paths moved at the live pin, so a build is owed and its digest must be frozen"
        );
        assert!(
            compatibility_audit_authorizes(
                FLUX2_INFERENCE_COMPATIBILITY_AUDIT,
                None,
                &FLUX2_FEATURE_WITNESS
            )
            .is_none(),
            "a moved closure path cannot be waved through with no digest frozen in source"
        );
        // sc-17606: and the same for the witness. Flipping the first nibble to a value the frozen
        // digest demonstrably is not, rather than to a fixed letter — a fixed one silently becomes
        // a no-op the day the real digest starts with it, which is what it did in sc-17524.
        let frozen = FLUX2_FEATURE_WITNESS.digest;
        let flipped = format!(
            "sha256:{}{}",
            if frozen.as_bytes()[7] == b'0' {
                '1'
            } else {
                '0'
            },
            &frozen[8..]
        );
        assert_ne!(flipped, frozen, "the mutation must actually mutate");
        assert!(
            compatibility_audit_authorizes(
                FLUX2_INFERENCE_COMPATIBILITY_AUDIT,
                FLUX2_AUDIT_ARTIFACT_PROOF,
                &FeatureWitnessExpectation {
                    digest: &flipped,
                    ..FLUX2_FEATURE_WITNESS
                }
            )
            .is_none(),
            "one character of the shipped feature witness is fatal"
        );
    }

    /// sc-17607: `crates/bundles/runtime-cuda` and `crates/media/candle-gen/candle-gen-catalog`
    /// both sit ABOVE the provider, so the measurement binary compiles neither and its digest can
    /// neither convict nor clear either one. The bundle was in this table anyway (on the argument
    /// that the worker links it) and the catalog — the intermediate node on that very edge — was in
    /// no list at all, which is one crate failing loud and its sibling failing silent for no
    /// recorded reason. Both are out now: the composition question they raise is asked of the
    /// linked bundle by `flux2_composition_audit.rs`, and the feature half of what the bundle's
    /// manifest could change is carried by `FLUX2_FEATURE_WITNESS`.
    ///
    /// The positive half matters as much as the negative: every remaining entry is a path the
    /// audited binary compiles (or an input to that compile), so no closure member can move without
    /// a digest being able to answer for it. Asserted as an EQUALITY, both directions — an audited
    /// path missing from the frozen set has no remedy but a re-capture, and a frozen entry that is
    /// no longer audited is a claim about a path nothing checks.
    #[test]
    fn the_closure_holds_no_path_the_audited_binary_cannot_answer_for() {
        for path in COMPOSITION_ONLY {
            assert!(
                !FLUX2_AUDITED_OBJECTS.iter().any(|(known, _)| *known == path),
                "{path} is a composition crate; a digest cannot adjudicate it, so auditing it here                  can only demand re-captures it will never authorize"
            );
        }
        let (_, adjudicates) =
            FLUX2_AUDIT_ARTIFACT_PROOF.expect("the shipped window moved paths, so a proof is owed");
        for (path, _) in FLUX2_AUDITED_OBJECTS {
            assert!(
                adjudicates.contains(&path),
                "{path} is audited but outside the adjudicable set, so a move in it would have no                  remedy but a re-capture"
            );
        }
        for path in adjudicates {
            assert!(
                FLUX2_AUDITED_OBJECTS.iter().any(|(known, _)| known == path),
                "{path} is claimed as adjudicable but is not in the closure, so nothing ever                  compares its objects and the claim is unfalsifiable"
            );
        }
    }

    #[test]
    fn the_superseded_schema_versions_are_refused_rather_than_re_graded() {
        // sc-17524: v1 and v2 record sc-15833's seven-path closure, which never looked at
        // `Cargo.lock`, `rust-toolchain.toml` or `.cargo/config.toml`. sc-17606: v3 has the
        // ten-path closure but never looked at how the shipped bundle resolves features.
        // sc-17607: v4 is refused from the OTHER direction — it audited one path MORE than this
        // schema does, so re-grading it means dropping an entry out of someone else's record.
        // Accepting any of them reads a record as evidence about inputs it did not audit, so the
        // version itself is the refusal.
        for stale in [1, 2, 3, 4] {
            let mut audit: Value =
                serde_json::from_str(&v5(&[CANDLE_GEN], Some(cuda_artifact()))).unwrap();
            audit["schemaVersion"] = json!(stale);
            assert!(
                compatibility_audit_authorizes(&audit.to_string(), PROOF, &WITNESS).is_none(),
                "a v{stale} record describes a closure this validator no longer recognizes"
            );
        }
    }

    #[test]
    fn a_doc_comment_move_is_authorized_by_a_matching_artifact_digest() {
        // The sc-16961 shape exactly: one crate tree moved, identical compiled code.
        assert!(compatibility_audit_authorizes(
            &v5(&[CANDLE_GEN], Some(cuda_artifact())),
            PROOF,
            &WITNESS
        )
        .is_some());
        assert!(
            compatibility_audit_authorizes(&v5(&[], None), None, &WITNESS).is_some(),
            "a v4 record over a quiet closure still takes the fast path"
        );
    }

    /// sc-17606: the layer with no fast path.
    ///
    /// The gap it closes — a crate outside FLUX.2's closure enabling a feature on a dependency
    /// FLUX.2 shares — moves no closure object and no lockfile entry, so it can only ever present
    /// itself on a record whose closure is QUIET. Every case here is therefore driven over the
    /// quiet record, which is exactly the one the other two layers wave straight through.
    #[test]
    fn the_shipped_feature_witness_is_required_even_when_nothing_moved() {
        let quiet = || serde_json::from_str::<Value>(&v5(&[], None)).unwrap();
        assert!(compatibility_audit_authorizes(&quiet().to_string(), None, &WITNESS).is_some());

        let mut absent = quiet();
        absent.as_object_mut().unwrap().remove("featureWitness");
        assert!(
            compatibility_audit_authorizes(&absent.to_string(), None, &WITNESS).is_none(),
            "a record with no witness is a v3 record wearing a v4 version number"
        );

        // The load-bearing negative: the witness must be graded against what is FROZEN, or the
        // record authorizes its own feature resolution and the layer is decorative.
        let elsewhere = format!("sha256:{}", "e".repeat(64));
        let mut foreign = quiet();
        foreign["featureWitness"]["capturedDigest"] = json!(elsewhere);
        foreign["featureWitness"]["compatibleDigest"] = json!(elsewhere);
        assert!(
            compatibility_audit_authorizes(&foreign.to_string(), None, &WITNESS).is_none(),
            "a witness signed by some other resolution is not signed at all"
        );

        // Two different digests IS the finding: something outside the closure changed how the
        // bundle compiles code FLUX.2 links, and the calibration describes the old resolution.
        let mut disagreeing = quiet();
        disagreeing["featureWitness"]["compatibleDigest"] = json!(elsewhere);
        assert!(
            compatibility_audit_authorizes(&disagreeing.to_string(), None, &WITNESS).is_none(),
            "witness digests that disagree are a re-capture, not an authorization"
        );
        // The mirror, and the case that keeps the CAPTURED comparison from being redundant: a
        // compatible side that matches the frozen digest says nothing about the captured side, and
        // dropping that comparison survived every other assertion here.
        let mut captured_off = quiet();
        captured_off["featureWitness"]["capturedDigest"] = json!(elsewhere);
        assert!(
            compatibility_audit_authorizes(&captured_off.to_string(), None, &WITNESS).is_none(),
            "the captured side is graded against the frozen digest, not inferred from the other one"
        );

        // The resolution is the question. A witness over the provider itself rather than the
        // shipped bundle would be exactly the narrower unification this story exists to escape,
        // and its digest would compare on perfectly equal terms.
        for (key, wrong) in [
            ("shippedPackage", "candle-gen-flux2"),
            ("scopeRoot", "runtime-cuda"),
            ("scopeFeatures", "metal"),
            ("target", "aarch64-apple-darwin"),
            ("edges", "normal,build,dev"),
        ] {
            let mut reframed = quiet();
            reframed["featureWitness"][key] = json!(wrong);
            assert!(
                compatibility_audit_authorizes(&reframed.to_string(), None, &WITNESS).is_none(),
                "a witness with {key} = {wrong} answers a different question"
            );
        }

        // A witness over an EMPTY resolution hashes to a stable digest identical at every
        // revision — the one false green this layer could manufacture on its own.
        for empty in [json!(0), json!("179"), Value::Null] {
            let mut nothing = quiet();
            nothing["featureWitness"]["packages"] = empty.clone();
            assert!(
                compatibility_audit_authorizes(&nothing.to_string(), None, &WITNESS).is_none(),
                "packages = {empty} cannot describe a resolution that was actually walked"
            );
        }
    }

    /// sc-17524: the build inputs the seven-path closure omitted. `.cargo/config.toml` is the
    /// only closure entry containing a path separator, so it is the transcription most likely to
    /// break — it is exercised here rather than assumed to behave like its siblings.
    #[test]
    fn a_moved_build_input_demands_a_digest_and_is_adjudicated_by_one() {
        for input in [CARGO_LOCK, RUST_TOOLCHAIN, CARGO_CONFIG] {
            assert!(
                compatibility_audit_authorizes(&v5(&[input], None), None, &WITNESS).is_none(),
                "{input} moving must force the artifact layer, not sail through the free path"
            );
            assert!(
                compatibility_audit_authorizes(
                    &v5(&[input], Some(cuda_artifact())),
                    PROOF,
                    &WITNESS
                )
                .is_some(),
                "{input} reaches the measured binary only through the build, so its digest decides"
            );
        }
        // ...and the digest still has to be the frozen one. `Cargo.lock` is the path this story
        // exists for, so it gets the mutation check rather than trusting the shared helper.
        let mut moved_digest = cuda_artifact();
        let other = format!("sha256:{}", "d".repeat(64));
        moved_digest["capturedDigest"] = json!(other);
        moved_digest["compatibleDigest"] = json!(other);
        assert!(
            compatibility_audit_authorizes(&v5(&[CARGO_LOCK], Some(moved_digest)), PROOF, &WITNESS)
                .is_none(),
            "a lockfile move signed by some other build is not signed at all"
        );
        // A binary that never reported linking the closure cannot vouch for a build input either:
        // the record's own `adjudicates` is intersected, so narrowing it withdraws the claim.
        let mut narrowed = cuda_artifact();
        narrowed["adjudicates"] = json!([CANDLE_GEN]);
        assert!(
            compatibility_audit_authorizes(&v5(&[CARGO_LOCK], Some(narrowed)), PROOF, &WITNESS)
                .is_none(),
            "a record that does not claim the lockfile must not have it granted by the frozen set"
        );
    }

    #[test]
    fn the_artifact_proof_cannot_be_faked() {
        let mutated = format!("sha256:d{}", "c".repeat(63));
        let reject = |label: &str, body: String, proof: Option<(&str, &[&str])>| {
            assert!(
                compatibility_audit_authorizes(&body, proof, &WITNESS).is_none(),
                "{label}"
            );
        };
        let mut one_character_off = cuda_artifact();
        one_character_off["capturedDigest"] = json!(mutated);
        one_character_off["compatibleDigest"] = json!(mutated);
        reject(
            "one character of the digest must be fatal",
            v5(&[CANDLE_GEN], Some(one_character_off)),
            PROOF,
        );
        let mut metal = cuda_artifact();
        metal["lane"] = json!("metal");
        reject(
            "a Metal build is not proof of the CUDA artifact the capture ran",
            v5(&[CANDLE_GEN], Some(metal)),
            PROOF,
        );
        let mut disagreeing = cuda_artifact();
        disagreeing["compatibleDigest"] = json!(format!("sha256:{}", "d".repeat(64)));
        reject(
            "digests that disagree are a re-capture, not an authorization",
            v5(&[CANDLE_GEN], Some(disagreeing)),
            PROOF,
        );
        let mut other_crate = cuda_artifact();
        other_crate["package"] = json!("candle-gen");
        reject(
            "auditing some other crate's binary",
            v5(&[CANDLE_GEN], Some(other_crate)),
            PROOF,
        );
        let mut debug = cuda_artifact();
        debug["profile"] = json!("debug");
        reject(
            "auditing a debug build the capture never used",
            v5(&[CANDLE_GEN], Some(debug)),
            PROOF,
        );
        reject(
            "a moved path with no artifact block at all",
            v5(&[CANDLE_GEN], None),
            PROOF,
        );
        reject(
            "the record may not authorize itself with no digest frozen in source",
            v5(&[CANDLE_GEN], Some(cuda_artifact())),
            None,
        );
        // sc-17607: with `crates/bundles/runtime-cuda` out of the closure there is no audited path
        // the binary fails to link, so the "unproven move" case is driven by withdrawing a path
        // from the adjudicable set rather than by naming one that was never in it. Same guard, and
        // it stays exercised for the day a future closure entry is not compile-covered.
        const WITHOUT_CANDLE_GEN: &[&str] = &[
            "Cargo.toml",
            "Cargo.lock",
            "rust-toolchain.toml",
            ".cargo/config.toml",
            "crates/contracts/gen-core",
            "crates/media/candle-gen/candle-gen-pid",
            "crates/media/candle-gen/candle-gen-flux2",
            "crates/media/candle-gen/vendor/candle-kernels",
        ];
        reject(
            "a move in a path the frozen set does not claim is unproven, not authorized",
            v5(&[CANDLE_GEN], Some(cuda_artifact())),
            Some((DIGEST, WITHOUT_CANDLE_GEN)),
        );
        reject(
            "and it is still unproven when it moves alongside one that IS adjudicable",
            v5(&[CANDLE_GEN, CARGO_LOCK], Some(cuda_artifact())),
            Some((DIGEST, WITHOUT_CANDLE_GEN)),
        );
        let mut understated: Value =
            serde_json::from_str(&v5(&[CANDLE_GEN], Some(cuda_artifact()))).unwrap();
        understated["changedClosurePaths"] = json!([]);
        reject(
            "understating which paths moved hides a second, unproven change",
            understated.to_string(),
            PROOF,
        );
        let mut captured_tampered: Value =
            serde_json::from_str(&v5(&[CANDLE_GEN], Some(cuda_artifact()))).unwrap();
        captured_tampered["auditedObjects"][0]["capturedObject"] = json!("a".repeat(40));
        reject(
            "a captured object that does not match the code the measurements ran on",
            captured_tampered.to_string(),
            PROOF,
        );
        let mut future: Value =
            serde_json::from_str(&v5(&[CANDLE_GEN], Some(cuda_artifact()))).unwrap();
        future["schemaVersion"] = json!(FLUX2_AUDIT_SCHEMA_VERSION + 1);
        reject(
            "a schema version nobody has defined",
            future.to_string(),
            PROOF,
        );
        let mut other_test = cuda_artifact();
        other_test["test"] = json!("tests::flux2_dev_smoke");
        reject(
            "auditing some other test than the one that produced the measurements",
            v5(&[CANDLE_GEN], Some(other_test)),
            PROOF,
        );
        let mut captured_off = cuda_artifact();
        captured_off["capturedDigest"] = json!(format!("sha256:{}", "d".repeat(64)));
        reject(
            "a captured digest that does not match the frozen one",
            v5(&[CANDLE_GEN], Some(captured_off)),
            PROOF,
        );
        let mut not_a_list = cuda_artifact();
        not_a_list["adjudicates"] = json!("everything");
        reject(
            "a record whose own adjudicable set is not a list of paths",
            v5(&[CANDLE_GEN], Some(not_a_list)),
            PROOF,
        );
        // The frozen set is a HUMAN TRANSCRIPTION and, unlike the digest, an over-wide one fails
        // OPEN. Intersecting it with what the build reported linking means both halves have to be
        // wrong the same way before anything is authorized.
        const OVER_WIDE: &[&str] = &[
            "crates/contracts/gen-core",
            "crates/media/candle-gen/candle-gen",
            "crates/media/candle-gen/candle-gen-pid",
            "crates/media/candle-gen/candle-gen-flux2",
            "crates/media/candle-gen/vendor/candle-kernels",
            "crates/bundles/runtime-cuda",
        ];
        let mut narrow_record = cuda_artifact();
        narrow_record["adjudicates"] = json!(WITHOUT_CANDLE_GEN);
        reject(
            "an over-wide frozen set is still checked against what the build reported linking",
            v5(&[CANDLE_GEN], Some(narrow_record)),
            Some((DIGEST, OVER_WIDE)),
        );
        assert!(
            compatibility_audit_authorizes(
                &v5(&[CANDLE_GEN], Some(cuda_artifact())),
                Some((DIGEST, OVER_WIDE)),
                &WITNESS
            )
            .is_some(),
            "a path BOTH halves claim is still adjudicated — the rule is coverage, not a blocklist"
        );
        let mut no_command: Value =
            serde_json::from_str(&v5(&[CANDLE_GEN], Some(cuda_artifact()))).unwrap();
        no_command.as_object_mut().unwrap().remove("command");
        reject(
            "a record with no command is not a record either language accepts",
            no_command.to_string(),
            PROOF,
        );
    }
}
