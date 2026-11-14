pub use sceneworks_options::admission_geometry::*;
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn shipped_entry(id: &str) -> JsonObject<String, Value> {
        let manifest = crate::builtin_manifests::builtin_manifest("builtin.models.jsonc")
            .expect("valid embedded model catalog");
        manifest["models"]
            .as_array()
            .expect("models array")
            .iter()
            .find(|model| model["id"] == id)
            .unwrap_or_else(|| panic!("{id} is in the shipped catalog"))
            .as_object()
            .expect("a model entry is an object")
            .clone()
    }

    fn qwen_2_1() -> AdmissionGeometry {
        AdmissionGeometry::from_manifest(&shipped_entry("qwen_image_2_1"))
            .expect("qwen_image_2_1 declares an admission envelope")
    }

    /// The catalog block must reproduce the provider's own `admission_geometry()` field for field.
    /// Spelled as literals rather than re-derived, because a derivation here would just be a second
    /// copy of the engine's arithmetic that agrees with itself while both drift from the engine.
    #[test]
    fn the_shipped_envelope_mirrors_the_providers_own_declaration() {
        let geometry = qwen_2_1();
        assert_eq!(geometry.max_side, 2752);
        assert_eq!(geometry.max_preset_area, 2400 * 1792);
        // Neither the widest preset nor the square default is the largest by AREA. If this ever
        // reads 2752*1536 or 2048*2048, someone has budgeted from the wrong preset.
        assert!(geometry.max_preset_area > 2752 * 1536);
        assert!(geometry.max_preset_area > 2048 * 2048);
        assert_eq!(geometry.max_target_image_tokens, 150 * 112);
        assert_eq!(geometry.max_reference_images, 10);
        // The FITTED 1024² grid, not the target's own token count.
        assert_eq!(geometry.max_batch_reference_tokens, 64 * 64);
        assert_eq!(geometry.pixels_per_token, 16);
        assert_eq!(geometry.max_batch, 8);
        assert_eq!(geometry.max_joint_tokens, 58_016);
        // …and the total is the sum of its parts, so a field cannot be edited in isolation.
        assert_eq!(
            geometry.max_joint_tokens,
            DECLARED_CONDITIONING_TOKENS
                + geometry.max_target_image_tokens
                + geometry.max_reference_images as u64 * geometry.max_batch_reference_tokens
        );
    }

    /// THE STORY'S OWN ACCEPTANCE CASE. Ten references at the default preset is a legal request and
    /// must be ADMITTED — it is 57 600 joint tokens against a declared 58 016.
    ///
    /// *Mutation that reds this:* pricing a reference at the target's token count (16 384 here)
    /// instead of the fitted 4 096. That is the over-pricing failure the engine's own fix pass
    /// withdrew, and it refuses a request that fits.
    #[test]
    fn a_ten_reference_request_at_the_default_preset_is_admitted() {
        let geometry = qwen_2_1();
        let tokens = geometry
            .admit(2048, 2048, 10, 1)
            .expect("ten references at the default preset is inside the envelope");
        assert_eq!(tokens, 256 + 16_384 + 10 * 4_096);
        assert_eq!(tokens, 57_600);
        assert!(tokens <= geometry.max_joint_tokens);
    }

    /// The worst case the envelope is stated at is admitted EXACTLY, with nothing to spare — which
    /// is what makes `max_joint_tokens` a real bound rather than a round number above the truth.
    #[test]
    fn the_declared_worst_case_is_admitted_exactly() {
        let geometry = qwen_2_1();
        let tokens = geometry
            .admit(2400, 1792, 10, 1)
            .expect("the largest-area preset with ten references is the declared worst case");
        assert_eq!(tokens, geometry.max_joint_tokens);
    }

    /// An over-size request is REFUSED, and the refusal names the number. Each axis is exercised,
    /// including the composite case no single axis catches.
    #[test]
    fn over_envelope_requests_are_refused_and_never_shrunk() {
        let geometry = qwen_2_1();

        // One side past the longest any preset uses.
        assert_eq!(
            geometry.admit(3072, 1024, 0, 1),
            Err(AdmissionRefusal::Side {
                requested: 3072,
                max: 2752
            })
        );
        // AREA over the envelope with BOTH sides individually legal — the case a `max_side`-only
        // check waves through. 2752x2048 is 5.63 Mpx against a 4.30 Mpx envelope.
        assert_eq!(
            geometry.admit(2752, 2048, 0, 1),
            Err(AdmissionRefusal::Area {
                requested: 2752 * 2048,
                max: 2400 * 1792
            })
        );
        // One reference past what the joint layout can express.
        assert_eq!(
            geometry.admit(2048, 2048, 11, 1),
            Err(AdmissionRefusal::ReferenceCount {
                requested: 11,
                max: 10
            })
        );
        assert_eq!(
            geometry.admit(2048, 2048, 0, 9),
            Err(AdmissionRefusal::Batch {
                requested: 9,
                max: 8
            })
        );

        // Just past the largest-area preset, with a legal reference count and batch: still an AREA
        // refusal, because for THIS envelope the area bound is the tighter of the two (see
        // `the_joint_bound_is_implied_by_the_axis_bounds_for_this_envelope`).
        assert!(matches!(
            geometry.admit(2400, 1808, 10, 1),
            Err(AdmissionRefusal::Area { .. })
        ));
    }

    /// For `qwen_image_2_1` the joint-token bound is IMPLIED by the area and reference bounds — and
    /// saying so is more useful than pretending otherwise.
    ///
    /// `max_joint_tokens` is defined as conditioning + the largest-area preset's tokens + ten
    /// fitted references, and image tokens are `floor(w/16) * floor(h/16) <= area / 256`. So any
    /// request inside the area and reference bounds is inside the joint bound too, and the composite
    /// check cannot fire for this model. It is kept because it is the quantity the memory model
    /// actually consumes: a route whose scheduler caps the sequence BELOW what its presets imply
    /// (a `max_image_seq_len` ceiling, say) would declare a tighter `maxJointTokens`, and that is
    /// the case the next test covers. Claiming here that the composite check "catches what the
    /// per-axis ones miss" would be a comment that is false for the only model that uses it.
    #[test]
    fn the_joint_bound_is_implied_by_the_axis_bounds_for_this_envelope() {
        let geometry = qwen_2_1();
        assert_eq!(
            geometry.max_joint_tokens,
            DECLARED_CONDITIONING_TOKENS
                + geometry.max_preset_area
                    / (geometry.pixels_per_token * geometry.pixels_per_token)
                + geometry.max_reference_images as u64 * geometry.max_batch_reference_tokens
        );
        // Swept over every shipped preset at every legal reference count: always admitted, which is
        // the statement above made executable rather than asserted in prose.
        for (width, height) in [
            (2048, 2048),
            (2400, 1792),
            (1792, 2400),
            (2528, 1696),
            (1696, 2528),
            (2752, 1536),
            (1536, 2752),
        ] {
            for references in 0..=geometry.max_reference_images {
                assert!(
                    geometry.admit(width, height, references, 1).is_ok(),
                    "{width}x{height} with {references} references is a legal shipped request"
                );
            }
        }
    }

    /// The composite refusal, on an envelope whose `maxJointTokens` is genuinely tighter than its
    /// axes imply — the shape a scheduler-capped route would declare. Every individual axis is
    /// satisfied and the request is still refused, and the message explains which combination.
    ///
    /// *Mutation that reds this:* dropping the joint-token check and keeping the per-axis ones.
    #[test]
    fn a_tighter_joint_bound_refuses_a_request_every_axis_admits() {
        let mut entry = JsonObject::new();
        entry.insert(
            "admissionGeometry".to_owned(),
            json!({
                "maxSide": 2752,
                "maxPresetArea": 4300800,
                "maxTargetImageTokens": 16800,
                "maxReferenceImages": 10,
                "tokensPerMaxReference": 4096,
                // Tighter than 256 + 16_800 + 10 * 4_096 = 58_016.
                "maxJointTokens": 40000,
                "pixelsPerToken": 16,
                "maxBatch": 8
            }),
        );
        let geometry = AdmissionGeometry::from_manifest(&entry).expect("a complete block parses");
        // Every axis on its own is satisfied…
        assert!(2048 <= geometry.max_side);
        assert!(2048u64 * 2048 <= geometry.max_preset_area);
        assert!(10 <= geometry.max_reference_images);
        // …and the combination is not.
        let refusal = geometry
            .admit(2048, 2048, 10, 1)
            .expect_err("57_600 tokens against a declared 40_000");
        assert_eq!(
            refusal,
            AdmissionRefusal::JointTokens {
                requested: 57_600,
                max: 40_000
            }
        );
        let message = refusal.message("scheduler_capped");
        assert!(message.contains("40000"), "{message}");
        assert!(message.contains("57600"), "{message}");
        assert!(
            message.contains("refused rather than silently shrunk"),
            "a refusal must say it is a refusal: {message}"
        );
    }

    /// Every message names both the bound and the request, so a user can act on it.
    #[test]
    fn every_refusal_names_the_bound_and_the_request() {
        let geometry = qwen_2_1();
        for (width, height, references, batch) in [
            (3072, 1024, 0, 1),
            (2752, 2048, 0, 1),
            (2048, 2048, 11, 1),
            (2048, 2048, 0, 9),
        ] {
            let refusal = geometry
                .admit(width, height, references, batch)
                .expect_err("over-envelope")
                .message("qwen_image_2_1");
            assert!(refusal.starts_with("qwen_image_2_1: "), "{refusal}");
            assert!(
                refusal.contains("exceed"),
                "a refusal must say what was exceeded: {refusal}"
            );
        }
    }

    /// The seam both gates call, over the SHIPPED entry.
    #[test]
    fn the_shared_seam_admits_the_legal_request_and_refuses_the_oversize_one() {
        let entry = shipped_entry("qwen_image_2_1");
        assert_eq!(
            refuse_over_envelope("qwen_image_2_1", &entry, 2048, 2048, 10, 1),
            None,
            "a legal ten-reference request at the default preset must be admitted"
        );
        assert!(
            refuse_over_envelope("qwen_image_2_1", &entry, 2752, 2048, 10, 1).is_some(),
            "an over-envelope request must be refused"
        );
    }

    /// INERT for a model that declares nothing — which is every other catalog entry today. A gate
    /// that started refusing renders for 99 models because one model gained a declaration would be
    /// a far worse bug than the one it fixes.
    #[test]
    fn a_model_with_no_declared_envelope_is_never_refused() {
        let entry = shipped_entry("qwen_image");
        assert!(entry.get("admissionGeometry").is_none());
        assert!(AdmissionGeometry::from_manifest(&entry).is_none());
        assert_eq!(
            refuse_over_envelope("qwen_image", &entry, 8192, 8192, 99, 99),
            None
        );
        // …and so is an empty entry, which is what a synthetic/imported request carries.
        assert_eq!(
            refuse_over_envelope("whatever", &JsonObject::new(), 8192, 8192, 99, 99),
            None
        );
    }

    /// A PARTIAL declaration fails OPEN rather than half-gating. A malformed block must not start
    /// refusing renders a correct one would admit; the manifest audit is what keeps it well-formed.
    #[test]
    fn a_partial_declaration_is_inert_rather_than_half_enforced() {
        let mut entry = JsonObject::new();
        entry.insert(
            "admissionGeometry".to_owned(),
            json!({ "maxSide": 2752, "maxPresetArea": 4300800 }),
        );
        assert!(AdmissionGeometry::from_manifest(&entry).is_none());
        assert_eq!(
            refuse_over_envelope("partial", &entry, 8192, 8192, 99, 99),
            None
        );
    }
}
