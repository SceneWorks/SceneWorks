//! Request-geometry admission against a model's declared envelope (sc-24112).
//!
//! # Why `limits.resolutions` is not enough
//!
//! Most image routes are bounded by their output size alone, and for those the resolution menu plus
//! the worker's stride checks are the whole story. `qwen_image_2_1` is the first catalog route where
//! that is false: its DiT attends over a **joint sequence** made of the conditioning tokens, the
//! target image's latent tokens, **and** one block of latent tokens per reference image — and the
//! activation transient is linear in that sequence. A consumer that budgeted from the largest side,
//! or from the default preset, would under-budget the real worst case; one that priced a reference
//! at the *target's* token count would over-budget it by more than 3x at the largest preset and
//! refuse requests that would in fact have fitted.
//!
//! So the provider publishes an explicit envelope (`memory_strategy::admission_geometry()`), the
//! catalog mirrors it verbatim under `admissionGeometry`, and this module is the consumer side.
//!
//! # It refuses; it never shrinks
//!
//! Every answer here is admit-or-refuse with a message that names the number. Nothing in this
//! module resizes an image, drops a reference, lowers a batch count or downgrades a tier. A silent
//! shrink is the failure mode this envelope exists to prevent: the user asked for a geometry, and
//! being handed a different one without being told is worse than being told no.
//!
//! # Declaration-driven, and inert for every model that declares nothing
//!
//! [`AdmissionGeometry::from_manifest`] returns `None` for an entry with no `admissionGeometry`
//! block, and [`refuse_over_envelope`] then answers `None` — so this is a no-op for the ~99 models
//! that do not declare one, exactly like the fit gates' behaviour on a model with no `candle` block.

use serde_json::{Map as JsonObject, Value};

/// A model's declared request-admission envelope, mirrored from its provider's own
/// `admission_geometry()`. Every field is a structural count off frozen engine geometry — none of
/// this is measured, and none of it is a budget in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AdmissionGeometry {
    /// Longest side any declared preset uses.
    pub(crate) max_side: u32,
    /// Largest preset AREA in pixels. Deliberately separate from [`Self::max_side`]: the widest
    /// preset and the largest-AREA preset are usually different objects, and budgeting from the
    /// former under-counts the latter.
    pub(crate) max_preset_area: u64,
    /// Latent tokens the largest-area preset contributes.
    pub(crate) max_target_image_tokens: u64,
    /// Most reference images the engine's joint layout can express.
    pub(crate) max_reference_images: u32,
    /// Latent tokens ONE reference adds — constant in the target size for a route that fits every
    /// condition image to a fixed output resolution first.
    pub(crate) max_batch_reference_tokens: u64,
    /// The worst-case joint sequence: conditioning + [`Self::max_target_image_tokens`] +
    /// [`Self::max_reference_images`] x [`Self::max_batch_reference_tokens`].
    pub(crate) max_joint_tokens: u64,
    /// Pixels one latent token covers on each axis.
    pub(crate) pixels_per_token: u64,
    /// Images one request may ask for. They render sequentially, so this multiplies time, not peak.
    pub(crate) max_batch: u32,
}

/// The conditioning-token count the published envelope is stated at. The engine's own table uses a
/// full prompt through its template, and [`AdmissionGeometry::max_joint_tokens`] already includes
/// it — so admitting a request means comparing like with like, which is why this is a named
/// constant rather than a zero.
pub(crate) const DECLARED_CONDITIONING_TOKENS: u64 = 256;

/// Why a request is outside the envelope. Each variant carries the declared bound AND the requested
/// value, because a refusal a user cannot act on is barely better than an OOM.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum AdmissionRefusal {
    /// A side longer than any preset uses.
    Side { requested: u32, max: u32 },
    /// An area larger than the largest preset's, even if both sides are individually legal. This is
    /// the one a `max_side`-only check misses.
    Area { requested: u64, max: u64 },
    /// More reference images than the joint layout can express.
    ReferenceCount { requested: u32, max: u32 },
    /// A batch larger than the engine accepts.
    Batch { requested: u32, max: u32 },
    /// The composite failure: every individual axis is legal, but together they exceed the joint
    /// sequence the attention runs over. This is the whole reason the envelope is not a set of
    /// independent limits.
    JointTokens { requested: u64, max: u64 },
}

impl AdmissionRefusal {
    /// The message the worker surfaces. It names the bound, the request, and — for the composite
    /// case — how the number was arrived at, so the user can see which axis to reduce.
    pub(crate) fn message(&self, model: &str) -> String {
        match self {
            Self::Side { requested, max } => format!(
                "{model}: {requested} px exceeds the model's {max} px longest side. Pick a smaller \
                 preset; the request is refused rather than silently resized."
            ),
            Self::Area { requested, max } => format!(
                "{model}: {requested} pixels exceeds the model's largest preset area of {max} \
                 pixels. Both sides can be individually legal and the area still be too large. The \
                 request is refused rather than silently resized."
            ),
            Self::ReferenceCount { requested, max } => format!(
                "{model}: {requested} reference images exceed the {max} the model's joint layout \
                 can express. The request is refused rather than silently dropping references."
            ),
            Self::Batch { requested, max } => {
                format!("{model}: a batch of {requested} exceeds the model's maximum of {max}.")
            }
            Self::JointTokens { requested, max } => format!(
                "{model}: this request needs {requested} joint attention tokens, above the model's \
                 declared {max}. Every limit on its own is satisfied — it is the combination of \
                 output size and reference count that is too large, because each reference adds a \
                 fixed block of image tokens. Reduce the output size or the number of references; \
                 the request is refused rather than silently shrunk."
            ),
        }
    }
}

impl AdmissionGeometry {
    /// Read the envelope a manifest entry declares, or `None` when it declares none.
    ///
    /// Every field is required by the schema, so a partial block is a manifest bug rather than a
    /// half-envelope to be guessed at: any missing field answers `None` and the gate becomes inert,
    /// which is the same way a model with no `candle` block skips the VRAM gate. It fails OPEN
    /// deliberately — a malformed declaration must not start refusing renders that a correct
    /// declaration would have admitted; the manifest audit is what makes the block well-formed.
    pub(crate) fn from_manifest(entry: &JsonObject<String, Value>) -> Option<Self> {
        let block = entry.get("admissionGeometry")?;
        let u32_field = |key: &str| -> Option<u32> {
            block
                .get(key)
                .and_then(Value::as_u64)
                .and_then(|value| u32::try_from(value).ok())
        };
        let u64_field = |key: &str| block.get(key).and_then(Value::as_u64);
        Some(Self {
            max_side: u32_field("maxSide")?,
            max_preset_area: u64_field("maxPresetArea")?,
            max_target_image_tokens: u64_field("maxTargetImageTokens")?,
            max_reference_images: u32_field("maxReferenceImages")?,
            max_batch_reference_tokens: u64_field("tokensPerMaxReference")?,
            max_joint_tokens: u64_field("maxJointTokens")?,
            pixels_per_token: u64_field("pixelsPerToken")?,
            max_batch: u32_field("maxBatch")?,
        })
    }

    /// Latent tokens one `width` x `height` image contributes.
    pub(crate) fn image_tokens(&self, width: u32, height: u32) -> u64 {
        if self.pixels_per_token == 0 {
            return 0;
        }
        (width as u64 / self.pixels_per_token) * (height as u64 / self.pixels_per_token)
    }

    /// The joint sequence a request of this shape produces: conditioning + target image +
    /// one fixed block per reference.
    ///
    /// References are priced at [`Self::max_batch_reference_tokens`] — the FITTED grid — and
    /// deliberately NOT at the target's own token count. The engine resizes every condition image
    /// to its own output resolution before the vision tower and the VAE see it, so a reference
    /// costs the same whatever the target size; charging the target's count instead would over-
    /// state the worst case by more than 3x at the largest preset and refuse legal requests.
    pub(crate) fn joint_tokens(&self, width: u32, height: u32, reference_count: u32) -> u64 {
        DECLARED_CONDITIONING_TOKENS
            + self.image_tokens(width, height)
            + reference_count as u64 * self.max_batch_reference_tokens
    }

    /// Admit a request, or say exactly why not.
    ///
    /// The per-axis checks run first so a single over-large axis is named directly rather than
    /// surfacing as an opaque token total; the composite joint-token check runs last and is the one
    /// that catches a request every individual axis admits.
    pub(crate) fn admit(
        &self,
        width: u32,
        height: u32,
        reference_count: u32,
        batch: u32,
    ) -> Result<u64, AdmissionRefusal> {
        let longest = width.max(height);
        if longest > self.max_side {
            return Err(AdmissionRefusal::Side {
                requested: longest,
                max: self.max_side,
            });
        }
        let area = width as u64 * height as u64;
        if area > self.max_preset_area {
            return Err(AdmissionRefusal::Area {
                requested: area,
                max: self.max_preset_area,
            });
        }
        if reference_count > self.max_reference_images {
            return Err(AdmissionRefusal::ReferenceCount {
                requested: reference_count,
                max: self.max_reference_images,
            });
        }
        if batch > self.max_batch {
            return Err(AdmissionRefusal::Batch {
                requested: batch,
                max: self.max_batch,
            });
        }
        let tokens = self.joint_tokens(width, height, reference_count);
        if tokens > self.max_joint_tokens {
            return Err(AdmissionRefusal::JointTokens {
                requested: tokens,
                max: self.max_joint_tokens,
            });
        }
        Ok(tokens)
    }
}

/// The consumer seam both fit gates and the worker's image entry point share: `Some(message)` when
/// the request is outside the model's declared envelope, `None` when it is admitted OR when the
/// model declares no envelope at all.
///
/// Backend-neutral on purpose. The envelope is a property of the engine's attention layout, which
/// MLX and Candle share exactly — a per-lane copy would be two declarations of one fact, free to
/// diverge.
pub(crate) fn refuse_over_envelope(
    model: &str,
    entry: &JsonObject<String, Value>,
    width: u32,
    height: u32,
    reference_count: u32,
    batch: u32,
) -> Option<String> {
    let geometry = AdmissionGeometry::from_manifest(entry)?;
    geometry
        .admit(width, height, reference_count, batch)
        .err()
        .map(|refusal| refusal.message(model))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn shipped_entry(id: &str) -> JsonObject<String, Value> {
        let manifest: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(
            sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
                .iter()
                .find(|(name, _)| *name == "builtin.models.jsonc")
                .expect("builtin.models.jsonc embedded")
                .1,
        ))
        .expect("builtin.models.jsonc parses");
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
        assert!(2048u32.max(2048) <= geometry.max_side);
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
