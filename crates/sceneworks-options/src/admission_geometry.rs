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
pub struct AdmissionGeometry {
    /// Longest side any declared preset uses.
    pub max_side: u32,
    /// Largest preset AREA in pixels. Deliberately separate from [`Self::max_side`]: the widest
    /// preset and the largest-AREA preset are usually different objects, and budgeting from the
    /// former under-counts the latter.
    pub max_preset_area: u64,
    /// Latent tokens the largest-area preset contributes.
    pub max_target_image_tokens: u64,
    /// Most reference images the engine's joint layout can express.
    pub max_reference_images: u32,
    /// Latent tokens ONE reference adds — constant in the target size for a route that fits every
    /// condition image to a fixed output resolution first.
    pub max_batch_reference_tokens: u64,
    /// The worst-case joint sequence: conditioning + [`Self::max_target_image_tokens`] +
    /// [`Self::max_reference_images`] x [`Self::max_batch_reference_tokens`].
    pub max_joint_tokens: u64,
    /// Pixels one latent token covers on each axis.
    pub pixels_per_token: u64,
    /// Images one request may ask for. They render sequentially, so this multiplies time, not peak.
    pub max_batch: u32,
}

/// The conditioning-token count the published envelope is stated at. The engine's own table uses a
/// full prompt through its template, and [`AdmissionGeometry::max_joint_tokens`] already includes
/// it — so admitting a request means comparing like with like, which is why this is a named
/// constant rather than a zero.
pub const DECLARED_CONDITIONING_TOKENS: u64 = 256;

/// Why a request is outside the envelope. Each variant carries the declared bound AND the requested
/// value, because a refusal a user cannot act on is barely better than an OOM.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdmissionRefusal {
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
    pub fn message(&self, model: &str) -> String {
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
    pub fn from_manifest(entry: &JsonObject<String, Value>) -> Option<Self> {
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
    pub fn image_tokens(&self, width: u32, height: u32) -> u64 {
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
    pub fn joint_tokens(&self, width: u32, height: u32, reference_count: u32) -> u64 {
        DECLARED_CONDITIONING_TOKENS
            + self.image_tokens(width, height)
            + reference_count as u64 * self.max_batch_reference_tokens
    }

    /// Admit a request, or say exactly why not.
    ///
    /// The per-axis checks run first so a single over-large axis is named directly rather than
    /// surfacing as an opaque token total; the composite joint-token check runs last and is the one
    /// that catches a request every individual axis admits.
    pub fn admit(
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
pub fn refuse_over_envelope(
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
