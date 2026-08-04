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
    "06e0c5e919918aeb7cec966a83ce6fe394feec5e";
pub(crate) const FLUX2_INFERENCE_COMPATIBILITY_AUDIT: &str =
    include_str!("../../../docs/calibration/sc-15833/inference-compatibility-06e0.json");
/// sc-17524: v3 is the nine-path closure. v1 and v2 describe sc-15833's seven-path one, which omits
/// `Cargo.lock` and `rust-toolchain.toml`; accepting one would read it as evidence about two build
/// inputs it never looked at.
pub(crate) const FLUX2_AUDIT_SCHEMA_VERSION: u64 = 3;
pub(crate) const FLUX2_V3_AUDIT_METHOD: &str = concat!(
    "compiled artifact identity for changed paths, git object identity for unchanged paths, ",
    "across the complete Candle FLUX.2 runtime dependency closure and its workspace build inputs"
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
/// sc-17524: it also carries the three workspace build inputs, which cargo can never name because
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
        "crates/contracts/gen-core",
        "crates/media/candle-gen/candle-gen",
        "crates/media/candle-gen/candle-gen-pid",
        "crates/media/candle-gen/candle-gen-flux2",
        "crates/media/candle-gen/vendor/candle-kernels",
    ],
));
/// The CAPTURED side of the closure — the code the measurements were taken against. Immutable.
///
/// sc-17524 added the two workspace build inputs. They are not packages, so cargo's
/// `compiler-artifact` stream can never name them, but they feed every build in the closure — and
/// `Cargo.lock` had already moved inside the authorized window while all seven crate trees stayed
/// byte-identical, which is a changed build input sailing through the free path unexamined.
pub(crate) const FLUX2_AUDITED_OBJECTS: [(&str, &str); 9] = [
    ("Cargo.toml", "8f5af6b9d53bbfe3be5d9d79b8949364138a087c"),
    ("Cargo.lock", "8ab01e00f01607a99845d875ed60275ae033450c"),
    (
        "rust-toolchain.toml",
        "ae829f875c68c03c367ce92cc05e041036a92d0a",
    ),
    (
        "crates/contracts/gen-core",
        "9a7e86f5893e584a8d0d656147abc4ae93af6922",
    ),
    (
        "crates/bundles/runtime-cuda",
        "aba807f775872760e72fd98f28a5a0d2853cf00f",
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
) -> Option<()> {
    let audit: Value = serde_json::from_str(body).ok()?;
    if audit.get("schemaVersion")?.as_u64()? != FLUX2_AUDIT_SCHEMA_VERSION
        || audit.get("story")?.as_str()? != "SC-15833"
        || audit.get("capturedInferenceRevision")?.as_str()? != FLUX2_CAPTURED_INFERENCE_REVISION
        || audit.get("compatibleInferenceRevision")?.as_str()?
            != FLUX2_COMPATIBLE_INFERENCE_REVISION
        || audit.get("method")?.as_str()? != FLUX2_V3_AUDIT_METHOD
        // JS requires this too; a record one language accepts and the other rejects is a split brain.
        || audit.get("command")?.as_str().is_none()
    {
        return None;
    }
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

#[cfg(test)]
mod sc_17497_artifact_audit_tests {
    use super::*;
    use serde_json::json;

    const CANDLE_GEN: &str = "crates/media/candle-gen/candle-gen";
    const RUNTIME_CUDA: &str = "crates/bundles/runtime-cuda";
    const CARGO_LOCK: &str = "Cargo.lock";
    const RUST_TOOLCHAIN: &str = "rust-toolchain.toml";
    const DIGEST: &str = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    /// What the `candle-gen-flux2` lib test binary speaks for. The crate trees are the ones it
    /// COMPILES; the three workspace build inputs are there because they are inputs to that very
    /// build, so anything they change about the measured code lands in its digest. `runtime-cuda` is
    /// deliberately absent: it depends on the provider, not the other way round.
    const ADJUDICATES: &[&str] = &[
        "Cargo.toml",
        "Cargo.lock",
        "rust-toolchain.toml",
        "crates/contracts/gen-core",
        "crates/media/candle-gen/candle-gen",
        "crates/media/candle-gen/candle-gen-pid",
        "crates/media/candle-gen/candle-gen-flux2",
        "crates/media/candle-gen/vendor/candle-kernels",
    ];
    const PROOF: Option<(&str, &[&str])> = Some((DIGEST, ADJUDICATES));

    /// A v3 record in which `moved` closure paths carry a different compatible object.
    fn v3(moved: &[&str], artifact: Option<Value>) -> String {
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
            "method": FLUX2_V3_AUDIT_METHOD,
            "command": "node scripts/inference-artifact-audit.mjs",
            "changedClosurePaths": moved,
            "auditedObjects": objects,
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
        // sc-17524: the packaged record's closure is NOT quiet. `Cargo.lock`, gen-core and
        // candle-gen all moved between the captured revision and the live pin, so this record
        // authorizes only in company with the compiled-artifact proof frozen above — and the
        // negative half is the load-bearing one, because a validator that accepts the shipped
        // record no matter what is frozen would pass with the whole artifact layer deleted.
        assert!(
            compatibility_audit_authorizes(
                FLUX2_INFERENCE_COMPATIBILITY_AUDIT,
                FLUX2_AUDIT_ARTIFACT_PROOF
            )
            .is_some(),
            "the shipped SC-15833 audit must authorize its own pin"
        );
        assert!(
            FLUX2_AUDIT_ARTIFACT_PROOF.is_some(),
            "closure paths moved at the live pin, so a build is owed and its digest must be frozen"
        );
        assert!(
            compatibility_audit_authorizes(FLUX2_INFERENCE_COMPATIBILITY_AUDIT, None).is_none(),
            "a moved closure path cannot be waved through with no digest frozen in source"
        );
    }

    #[test]
    fn the_seven_path_schema_versions_are_refused_rather_than_re_graded() {
        // sc-17524: v1 and v2 record sc-15833's seven-path closure, which never looked at
        // `Cargo.lock` or `rust-toolchain.toml`. Accepting one would read it as evidence about two
        // build inputs it did not audit, so the version itself is the refusal.
        for stale in [1, 2] {
            let mut audit: Value =
                serde_json::from_str(&v3(&[CANDLE_GEN], Some(cuda_artifact()))).unwrap();
            audit["schemaVersion"] = json!(stale);
            assert!(
                compatibility_audit_authorizes(&audit.to_string(), PROOF).is_none(),
                "a v{stale} record describes a closure this validator no longer recognizes"
            );
        }
    }

    #[test]
    fn a_doc_comment_move_is_authorized_by_a_matching_artifact_digest() {
        // The sc-16961 shape exactly: one crate tree moved, identical compiled code.
        assert!(
            compatibility_audit_authorizes(&v3(&[CANDLE_GEN], Some(cuda_artifact())), PROOF)
                .is_some()
        );
        assert!(
            compatibility_audit_authorizes(&v3(&[], None), None).is_some(),
            "a v3 record over a quiet closure still takes the fast path"
        );
    }

    /// sc-17524: the two build inputs the seven-path closure omitted.
    #[test]
    fn a_moved_build_input_demands_a_digest_and_is_adjudicated_by_one() {
        for input in [CARGO_LOCK, RUST_TOOLCHAIN] {
            assert!(
                compatibility_audit_authorizes(&v3(&[input], None), None).is_none(),
                "{input} moving must force the artifact layer, not sail through the free path"
            );
            assert!(
                compatibility_audit_authorizes(&v3(&[input], Some(cuda_artifact())), PROOF)
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
            compatibility_audit_authorizes(&v3(&[CARGO_LOCK], Some(moved_digest)), PROOF).is_none(),
            "a lockfile move signed by some other build is not signed at all"
        );
        // A binary that never reported linking the closure cannot vouch for a build input either:
        // the record's own `adjudicates` is intersected, so narrowing it withdraws the claim.
        let mut narrowed = cuda_artifact();
        narrowed["adjudicates"] = json!([CANDLE_GEN]);
        assert!(
            compatibility_audit_authorizes(&v3(&[CARGO_LOCK], Some(narrowed)), PROOF).is_none(),
            "a record that does not claim the lockfile must not have it granted by the frozen set"
        );
    }

    #[test]
    fn the_artifact_proof_cannot_be_faked() {
        let mutated = format!("sha256:d{}", "c".repeat(63));
        let reject = |label: &str, body: String, proof: Option<(&str, &[&str])>| {
            assert!(
                compatibility_audit_authorizes(&body, proof).is_none(),
                "{label}"
            );
        };
        let mut one_character_off = cuda_artifact();
        one_character_off["capturedDigest"] = json!(mutated);
        one_character_off["compatibleDigest"] = json!(mutated);
        reject(
            "one character of the digest must be fatal",
            v3(&[CANDLE_GEN], Some(one_character_off)),
            PROOF,
        );
        let mut metal = cuda_artifact();
        metal["lane"] = json!("metal");
        reject(
            "a Metal build is not proof of the CUDA artifact the capture ran",
            v3(&[CANDLE_GEN], Some(metal)),
            PROOF,
        );
        let mut disagreeing = cuda_artifact();
        disagreeing["compatibleDigest"] = json!(format!("sha256:{}", "d".repeat(64)));
        reject(
            "digests that disagree are a re-capture, not an authorization",
            v3(&[CANDLE_GEN], Some(disagreeing)),
            PROOF,
        );
        let mut other_crate = cuda_artifact();
        other_crate["package"] = json!("candle-gen");
        reject(
            "auditing some other crate's binary",
            v3(&[CANDLE_GEN], Some(other_crate)),
            PROOF,
        );
        let mut debug = cuda_artifact();
        debug["profile"] = json!("debug");
        reject(
            "auditing a debug build the capture never used",
            v3(&[CANDLE_GEN], Some(debug)),
            PROOF,
        );
        reject(
            "a moved path with no artifact block at all",
            v3(&[CANDLE_GEN], None),
            PROOF,
        );
        reject(
            "the record may not authorize itself with no digest frozen in source",
            v3(&[CANDLE_GEN], Some(cuda_artifact())),
            None,
        );
        reject(
            "a move in a path the audited binary never links is unproven, not authorized",
            v3(&[RUNTIME_CUDA], Some(cuda_artifact())),
            PROOF,
        );
        reject(
            "and it is still unproven when it moves alongside one that IS adjudicable",
            v3(&[CANDLE_GEN, RUNTIME_CUDA], Some(cuda_artifact())),
            PROOF,
        );
        let mut understated: Value =
            serde_json::from_str(&v3(&[CANDLE_GEN], Some(cuda_artifact()))).unwrap();
        understated["changedClosurePaths"] = json!([]);
        reject(
            "understating which paths moved hides a second, unproven change",
            understated.to_string(),
            PROOF,
        );
        let mut captured_tampered: Value =
            serde_json::from_str(&v3(&[CANDLE_GEN], Some(cuda_artifact()))).unwrap();
        captured_tampered["auditedObjects"][0]["capturedObject"] = json!("a".repeat(40));
        reject(
            "a captured object that does not match the code the measurements ran on",
            captured_tampered.to_string(),
            PROOF,
        );
        let mut future: Value =
            serde_json::from_str(&v3(&[CANDLE_GEN], Some(cuda_artifact()))).unwrap();
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
            v3(&[CANDLE_GEN], Some(other_test)),
            PROOF,
        );
        let mut captured_off = cuda_artifact();
        captured_off["capturedDigest"] = json!(format!("sha256:{}", "d".repeat(64)));
        reject(
            "a captured digest that does not match the frozen one",
            v3(&[CANDLE_GEN], Some(captured_off)),
            PROOF,
        );
        let mut not_a_list = cuda_artifact();
        not_a_list["adjudicates"] = json!("everything");
        reject(
            "a record whose own adjudicable set is not a list of paths",
            v3(&[CANDLE_GEN], Some(not_a_list)),
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
        reject(
            "an over-wide frozen set is still checked against what the build reported linking",
            v3(&[RUNTIME_CUDA], Some(cuda_artifact())),
            Some((DIGEST, OVER_WIDE)),
        );
        let mut links_the_bundle = cuda_artifact();
        links_the_bundle["adjudicates"] = json!(OVER_WIDE);
        assert!(
            compatibility_audit_authorizes(
                &v3(&[RUNTIME_CUDA], Some(links_the_bundle)),
                Some((DIGEST, OVER_WIDE))
            )
            .is_some(),
            "an artifact that DOES link it may adjudicate it — the rule is coverage, not a blocklist"
        );
        let mut no_command: Value =
            serde_json::from_str(&v3(&[CANDLE_GEN], Some(cuda_artifact()))).unwrap();
        no_command.as_object_mut().unwrap().remove("command");
        reject(
            "a record with no command is not a record either language accepts",
            no_command.to_string(),
            PROOF,
        );
    }
}
