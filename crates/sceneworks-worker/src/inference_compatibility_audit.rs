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
    "277f423822bf1899340ed3d867c3d6a773473d7b";
pub(crate) const FLUX2_INFERENCE_COMPATIBILITY_AUDIT: &str =
    include_str!("../../../docs/calibration/sc-15833/inference-compatibility-277f.json");
pub(crate) const FLUX2_V1_AUDIT_METHOD: &str =
    "git object identity across the complete Candle FLUX.2 runtime dependency closure";
pub(crate) const FLUX2_V2_AUDIT_METHOD: &str = concat!(
    "compiled artifact identity for changed paths, git object identity for unchanged paths, ",
    "across the complete Candle FLUX.2 runtime dependency closure"
);
/// sc-17497: the compiled-artifact proof, frozen here so the packaged record cannot authorize
/// itself. `None` means every closure object is byte-identical at the live pin and object identity
/// still decides on its own; `Some((digest, adjudicates))` is required the moment one of them moves.
///
/// `adjudicates` is the closure subset the audited binary LINKS, and it is load-bearing:
/// `runtime-cuda` depends on `candle-gen-flux2`, not the reverse, so a commit into the CUDA bundle
/// leaves the measurement binary byte-identical. Without this the unchanged digest would be read as
/// proof over a path that was never compiled into it.
///
/// See `scripts/inference-artifact-audit.mjs` and `docs/inference-artifact-audit-sc-17497.md`.
pub(crate) const FLUX2_AUDIT_ARTIFACT_PROOF: Option<(&str, &[&str])> = None;
/// The CAPTURED side of the closure — the code the measurements were taken against. Immutable.
pub(crate) const FLUX2_AUDITED_OBJECTS: [(&str, &str); 7] = [
    ("Cargo.toml", "8f5af6b9d53bbfe3be5d9d79b8949364138a087c"),
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
    let schema_version = audit.get("schemaVersion")?.as_u64()?;
    let expected_method = match schema_version {
        1 => FLUX2_V1_AUDIT_METHOD,
        2 => FLUX2_V2_AUDIT_METHOD,
        _ => return None,
    };
    if audit.get("story")?.as_str()? != "SC-15833"
        || audit.get("capturedInferenceRevision")?.as_str()? != FLUX2_CAPTURED_INFERENCE_REVISION
        || audit.get("compatibleInferenceRevision")?.as_str()?
            != FLUX2_COMPATIBLE_INFERENCE_REVISION
        || audit.get("method")?.as_str()? != expected_method
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
    // A v1 record has no artifact layer, so it can only ever assert object identity.
    if schema_version == 1 && !changed.is_empty() {
        return None;
    }
    if schema_version == 2 {
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
            if changed.iter().any(|path| !adjudicates.contains(path)) {
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
    const DIGEST: &str = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    /// What the `candle-gen-flux2` lib test binary links. `runtime-cuda` is deliberately absent: it
    /// depends on the provider, not the other way round.
    const ADJUDICATES: &[&str] = &[
        "crates/contracts/gen-core",
        "crates/media/candle-gen/candle-gen",
        "crates/media/candle-gen/candle-gen-pid",
        "crates/media/candle-gen/candle-gen-flux2",
        "crates/media/candle-gen/vendor/candle-kernels",
    ];
    const PROOF: Option<(&str, &[&str])> = Some((DIGEST, ADJUDICATES));

    /// A v2 record in which `moved` closure paths carry a different compatible object.
    fn v2(moved: &[&str], artifact: Option<Value>) -> String {
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
            "schemaVersion": 2,
            "story": "SC-15833",
            "capturedInferenceRevision": FLUX2_CAPTURED_INFERENCE_REVISION,
            "compatibleInferenceRevision": FLUX2_COMPATIBLE_INFERENCE_REVISION,
            "method": FLUX2_V2_AUDIT_METHOD,
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
            "capturedDigest": DIGEST,
            "compatibleDigest": DIGEST,
        })
    }

    #[test]
    fn shipped_record_authorizes_and_needs_no_artifact_proof() {
        // The packaged record is v1 with an object-identical closure; that is the free fast path,
        // and it must keep working unchanged.
        assert!(
            compatibility_audit_authorizes(FLUX2_INFERENCE_COMPATIBILITY_AUDIT, None).is_some(),
            "the shipped SC-15833 audit must still authorize its own pin"
        );
        assert_eq!(
            FLUX2_AUDIT_ARTIFACT_PROOF, None,
            "the live pin's closure is object-identical, so no build is owed"
        );
        // A digest frozen while the closure is quiet demands a build that is not due.
        assert!(
            compatibility_audit_authorizes(FLUX2_INFERENCE_COMPATIBILITY_AUDIT, PROOF).is_none()
        );
    }

    #[test]
    fn a_v1_record_can_never_absorb_a_moved_object() {
        let mut audit: Value = serde_json::from_str(FLUX2_INFERENCE_COMPATIBILITY_AUDIT).unwrap();
        for entry in audit["auditedObjects"].as_array_mut().unwrap() {
            if entry["path"] == CANDLE_GEN {
                entry["compatibleObject"] = json!("e".repeat(40));
            }
        }
        assert!(compatibility_audit_authorizes(&audit.to_string(), None).is_none());
        // ...and it is not rescued by bolting an artifact block onto a v1 record.
        audit["auditedArtifact"] = cuda_artifact();
        assert!(compatibility_audit_authorizes(&audit.to_string(), PROOF).is_none());
    }

    #[test]
    fn a_doc_comment_move_is_authorized_by_a_matching_artifact_digest() {
        // The sc-16961 shape exactly: one crate tree moved, identical compiled code.
        assert!(
            compatibility_audit_authorizes(&v2(&[CANDLE_GEN], Some(cuda_artifact())), PROOF)
                .is_some()
        );
        assert!(
            compatibility_audit_authorizes(&v2(&[], None), None).is_some(),
            "a v2 record over a quiet closure still takes the fast path"
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
            v2(&[CANDLE_GEN], Some(one_character_off)),
            PROOF,
        );
        let mut metal = cuda_artifact();
        metal["lane"] = json!("metal");
        reject(
            "a Metal build is not proof of the CUDA artifact the capture ran",
            v2(&[CANDLE_GEN], Some(metal)),
            PROOF,
        );
        let mut disagreeing = cuda_artifact();
        disagreeing["compatibleDigest"] = json!(format!("sha256:{}", "d".repeat(64)));
        reject(
            "digests that disagree are a re-capture, not an authorization",
            v2(&[CANDLE_GEN], Some(disagreeing)),
            PROOF,
        );
        let mut other_crate = cuda_artifact();
        other_crate["package"] = json!("candle-gen");
        reject(
            "auditing some other crate's binary",
            v2(&[CANDLE_GEN], Some(other_crate)),
            PROOF,
        );
        let mut debug = cuda_artifact();
        debug["profile"] = json!("debug");
        reject(
            "auditing a debug build the capture never used",
            v2(&[CANDLE_GEN], Some(debug)),
            PROOF,
        );
        reject(
            "a moved path with no artifact block at all",
            v2(&[CANDLE_GEN], None),
            PROOF,
        );
        reject(
            "the record may not authorize itself with no digest frozen in source",
            v2(&[CANDLE_GEN], Some(cuda_artifact())),
            None,
        );
        reject(
            "a move in a path the audited binary never links is unproven, not authorized",
            v2(&[RUNTIME_CUDA], Some(cuda_artifact())),
            PROOF,
        );
        reject(
            "and it is still unproven when it moves alongside one that IS adjudicable",
            v2(&[CANDLE_GEN, RUNTIME_CUDA], Some(cuda_artifact())),
            PROOF,
        );
        let mut understated: Value =
            serde_json::from_str(&v2(&[CANDLE_GEN], Some(cuda_artifact()))).unwrap();
        understated["changedClosurePaths"] = json!([]);
        reject(
            "understating which paths moved hides a second, unproven change",
            understated.to_string(),
            PROOF,
        );
        let mut captured_tampered: Value =
            serde_json::from_str(&v2(&[CANDLE_GEN], Some(cuda_artifact()))).unwrap();
        captured_tampered["auditedObjects"][0]["capturedObject"] = json!("a".repeat(40));
        reject(
            "a captured object that does not match the code the measurements ran on",
            captured_tampered.to_string(),
            PROOF,
        );
        let mut future: Value =
            serde_json::from_str(&v2(&[CANDLE_GEN], Some(cuda_artifact()))).unwrap();
        future["schemaVersion"] = json!(3);
        reject("a v3 nobody has defined", future.to_string(), PROOF);
    }
}
