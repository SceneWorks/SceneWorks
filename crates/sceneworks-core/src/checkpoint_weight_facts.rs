//! **Three separate, correlated facts about one loaded checkpoint**, SceneWorks side (sc-21484,
//! epic 11037).
//!
//! The inference half (inference PR #806, `gen_core::checkpoint_facts`) owns the *measurement*: it
//! compiles a source-codec inventory from a verified import-plan binding, splits the load receipt
//! per execution representation, and refuses to let a receipt label a run native that the host's
//! probed capability does not license. This module owns the *SceneWorks* half — the shape those
//! facts take once they leave the engine and enter admission, an asset receipt, a telemetry row, or
//! a model fact a user reads.
//!
//! # Why SceneWorks needs its own type rather than re-exporting the engine's
//!
//! Three reasons, in increasing order of how much they matter.
//!
//! 1. **`sceneworks-core` has no engine dependency at all.** It owns the manifests, the routing
//!    catalog and the job contracts, and it is compiled into `sceneworks-rust-api` builds that link
//!    no backend (the Docker/RunPod api image is built without `backend-candle`). A fact type that
//!    only exists where an engine is linked cannot be the type admission and telemetry speak.
//! 2. **These facts get *persisted*.** They ride an asset sidecar and a `generation_metrics` row and
//!    are read back months later by a different build. That needs an owned, versioned, serde shape
//!    with stable field names — not a `#[derive(Debug)]` engine struct whose field set moves with
//!    the pin.
//! 3. **The labels must survive the trip unmodified.** Everything here is *carried*, never
//!    re-derived. See [the honesty rules](#the-honesty-rules).
//!
//! The **wire strings are shared verbatim** with the engine and are part of the cross-repo handoff
//! contract: the codec ids ([`NVFP4_CODEC_ID`] and friends) and the representation labels
//! ([`ExecutionRepresentation::label`], `"native-packed"` / `"dense-fallback"`). Those strings do
//! not change with refactors on either side.
//!
//! # The three facts
//!
//! 1. **What is the source?** — [`SourceBindingFact`], the host-independent identity token of the
//!    artifact the load actually read (`"<file-name>@<size-bytes>"`, the same rendering as the
//!    engine's `SourceBinding::stable_token`).
//! 2. **What codecs does the source store?** — [`SourceCodecFact`] rows. A checkpoint whose
//!    projections are stored `nvfp4-v1` says so **on every host**, including a Mac, a CPU-only box,
//!    and a pre-Blackwell GPU. This is the fact that must never be collapsed into a request tier.
//! 3. **What did this host actually materialize?** — [`MaterializationFact`], which is *either* a
//!    set of measured [`MaterializedCodecFact`] rows split per [`ExecutionRepresentation`], *or* the
//!    explicit [`MaterializationFact::Unavailable`] arm. There is no third state and no default.
//!
//! Alongside them rides the host's [`NativeExecutionCapabilityFact`] — what this box *could* execute
//! packed, stated without any checkpoint in hand. It is what licenses a native label; a
//! representation is never inferred from it.
//!
//! # The honesty rules
//!
//! [`CheckpointWeightFactsV1::try_new`] is the only constructor, so no consumer is ever handed an
//! unvalidated set. It enforces exactly the invariants whose violation would tell a user something
//! untrue, and each mirrors a rule the engine enforces on its own side:
//!
//! * **A dense fallback may not be labelled native.** A [`ExecutionRepresentation::NativePacked`]
//!   row with a non-zero tensor count requires a capability that lists the codec
//!   ([`CheckpointWeightFactsError::NativeWithoutCapability`]). A non-native host — a Mac, a CPU
//!   box, a pre-Blackwell GPU, and *datacenter Blackwell `sm_100`*, which is outside this leg's
//!   kernel — lists no `nvfp4-v1`, and that absence makes the native label unrepresentable rather
//!   than merely unlikely.
//! * **A receipt may not alias the source.** Every codec a materialization row reports must be one
//!   the source inventory declares ([`CheckpointWeightFactsError::UnplannedCodec`]). This is the
//!   rule that refuses `nvfp4-v1` → `q4`: a *request tier* is not a codec id, and a file stored
//!   `nvfp4-v1` cannot report having materialized `int8-per-row-v1` either.
//! * **A tier is never a codec and a codec is never a tier.** [`is_codec_id`] rejects every request
//!   tier spelling (`q4`, `q8`, `bf16`, `nvfp4`, `int8-convrot`, a bit count) at the boundary, so an
//!   aliasing bug cannot enter through a `String` field. E2's five historical failures were all
//!   `bits => tier` re-derivations; nothing here reads a bit count.
//! * **Unavailable is not dense and is not native.** When the runtime supplied no receipt,
//!   [`CheckpointWeightFactsV1::executes_natively`] answers `None` — "not known" — never `false`
//!   (which would assert a dense run nobody measured) and never `true`.
//!
//! # Not every source produces facts
//!
//! The engine's accessors return `Ok(None)` for a **directory-sourced import** — a diffusers tree,
//! or a packed-tier variant resolved to a folder — because those have no single pinned file and no
//! compiled plan. That arm is represented here by simply having no
//! [`CheckpointWeightFactsV1`] at all; every consumer takes an `Option`. `None` means "this source
//! shape does not answer these questions", never "the source stores nothing quantized", and in
//! particular a consumer must not fall back to a request tier to fill the hole.
//!
//! # What this module deliberately does not do
//!
//! It does not *probe* anything. It has no compute-capability check, no `sm_120` floor, no dtype
//! inspection. The capability is rendered by whoever probed the host (the worker's candle lane, from
//! the `nvidia-smi` compute-cap probe `discover_gpu` already runs at startup) and the
//! materialization is measured by the engine. This module is the shape they are carried in and the
//! set of lies it refuses to represent.

use std::collections::BTreeSet;
use std::fmt;

use serde::{Deserialize, Serialize};

/// Schema version of the persisted fact set. Bump only for a breaking field change; new optional
/// fields do not need one.
pub const CHECKPOINT_WEIGHT_FACTS_VERSION: u32 = 1;

/// NVIDIA NVFP4 (packed E2M1 nibbles + blocked FP8 scales + a scalar FP32 second scale).
///
/// **This is a source codec id, not a request tier.** The tier a user picks is spelled `"nvfp4"`
/// (see `apps/web/src/quantTier.js`); the codec a file is *stored in* is spelled `"nvfp4-v1"`. The
/// two are deliberately different strings so a copy-paste cannot silently turn one into the other.
pub const NVFP4_CODEC_ID: &str = "nvfp4-v1";
/// Dense BF16 — the resident encoding every quantized row in this leg decodes to when it takes the
/// dense fallback.
pub const DENSE_BF16_CODEC_ID: &str = "dense-bf16-v1";
/// ComfyUI `comfy_quant` int8 tensorwise/per-row.
pub const INT8_PER_ROW_CODEC_ID: &str = "int8-per-row-v1";
/// Dense F16.
pub const DENSE_F16_CODEC_ID: &str = "dense-f16-v1";
/// Dense F32.
pub const DENSE_F32_CODEC_ID: &str = "dense-f32-v1";
/// FP8 E4M3 with a scalar scale.
pub const FP8_E4M3_SCALAR_CODEC_ID: &str = "fp8-e4m3-scalar-v1";
/// FP8 E5M2 with a scalar scale.
pub const FP8_E5M2_SCALAR_CODEC_ID: &str = "fp8-e5m2-scalar-v1";
/// Microscaling FP8.
pub const MXFP8_CODEC_ID: &str = "mxfp8-v1";
/// A GGUF container (the ggml block codecs live inside it).
pub const GGUF_CONTAINER_CODEC_ID: &str = "gguf-container-v1";

/// Every codec id this build knows, sorted. Mirrors the engine's registered codec set; the guard
/// [`is_codec_id`] is what keeps a request tier from entering a codec field.
pub const KNOWN_CODEC_IDS: &[&str] = &[
    DENSE_BF16_CODEC_ID,
    DENSE_F16_CODEC_ID,
    DENSE_F32_CODEC_ID,
    FP8_E4M3_SCALAR_CODEC_ID,
    FP8_E5M2_SCALAR_CODEC_ID,
    GGUF_CONTAINER_CODEC_ID,
    INT8_PER_ROW_CODEC_ID,
    MXFP8_CODEC_ID,
    NVFP4_CODEC_ID,
];

/// Whether `value` is a codec id this build recognizes.
///
/// The point of the check is **not** completeness — it is that every request-tier spelling
/// (`"q4"`, `"q8"`, `"bf16"`, `"dense"`, `"nvfp4"`, `"int8-convrot"`, `"4"`) fails it. A codec id
/// and a tier are different vocabularies and this is where they are kept apart; an unknown *codec*
/// from a newer engine is rejected here too, which is the conservative direction (a fact set is
/// refused rather than a stranger string being rendered to a user as if it were understood).
pub fn is_codec_id(value: &str) -> bool {
    KNOWN_CODEC_IDS.contains(&value)
}

/// The representation a codec row was **actually materialized as** on this host.
///
/// Serializes as its [`label`](Self::label) — the stable wire string shared with the engine's
/// `ExecutionRepresentation`. These two strings appear in persisted asset sidecars and telemetry
/// rows; they do not change.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ExecutionRepresentation {
    /// The stored packing itself was the execution operand — NVFP4 W4A4, packed FP8 E4M3 GEMM.
    #[serde(rename = "native-packed")]
    NativePacked,
    /// The stored bytes were decoded to the codec's dense resident encoding (BF16 for every
    /// quantized row in this leg) and executed dense.
    #[serde(rename = "dense-fallback")]
    DenseFallback,
}

impl ExecutionRepresentation {
    /// The stable wire label.
    pub fn label(self) -> &'static str {
        match self {
            Self::NativePacked => "native-packed",
            Self::DenseFallback => "dense-fallback",
        }
    }

    /// Parse a wire label back. Anything else — including a tier name — is `None`.
    pub fn from_label(label: &str) -> Option<Self> {
        match label {
            "native-packed" => Some(Self::NativePacked),
            "dense-fallback" => Some(Self::DenseFallback),
            _ => None,
        }
    }
}

impl fmt::Display for ExecutionRepresentation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// What **this host** can execute in a codec's stored packing — a host fact, stated without any
/// checkpoint in hand.
///
/// # Who may construct a non-empty one
///
/// Only code rendering an actual **probe** of the host. In this repository that is the worker's
/// candle lane (`crate::checkpoint_weight_facts_host`), which renders the `nvidia-smi` compute-cap
/// probe `discover_gpu` already performs at startup. Every dense-only host — macOS/MLX, CPU, a
/// non-candle build, an unprobed host — uses [`Self::dense_only`], which says *why* the set is
/// empty.
///
/// The AC3 guarantee (a host below the native floor cannot produce facts labelling its run native)
/// rests entirely on this being a rendering of a probe rather than an assertion by a consumer who
/// would like the answer to be yes. Tests may construct the host they are simulating; nothing else
/// should.
/// # Deserialization revalidates
///
/// The `Deserialize` impl is routed through [`Self::new`] rather than derived onto the field
/// (sc-21484 review). A derived impl would accept any string set, so a persisted receipt carrying
/// a *tier* spelling — `"nvfp4"`, the request vocabulary — would round-trip through the asset
/// receipt and back out as a native-execution claim that [`Self::new`] refuses to build. The
/// guarantee has to hold on the way in as well as the way out or it is only a construction-time
/// convention.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeExecutionCapabilityFact {
    /// The codec ids this host executes in their stored packing, sorted and de-duplicated.
    native_codec_ids: BTreeSet<String>,
}

impl<'de> Deserialize<'de> for NativeExecutionCapabilityFact {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct Wire {
            #[serde(default)]
            native_codec_ids: BTreeSet<String>,
        }
        let wire = Wire::deserialize(deserializer)?;
        Self::new(wire.native_codec_ids).map_err(serde::de::Error::custom)
    }
}

impl NativeExecutionCapabilityFact {
    /// The dense-only host: no codec executes in its stored packing.
    pub fn dense_only() -> Self {
        Self {
            native_codec_ids: BTreeSet::new(),
        }
    }

    /// Declare the codec rows this host executes natively. Every id must pass [`is_codec_id`]; a
    /// tier spelling is rejected rather than silently admitted as an unknown codec.
    pub fn new<I, S>(codec_ids: I) -> Result<Self, CheckpointWeightFactsError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut native_codec_ids = BTreeSet::new();
        for codec_id in codec_ids {
            let codec_id = codec_id.into();
            if !is_codec_id(&codec_id) {
                return Err(CheckpointWeightFactsError::NotACodecId { value: codec_id });
            }
            native_codec_ids.insert(codec_id);
        }
        Ok(Self { native_codec_ids })
    }

    /// Whether this host executes `codec_id` in its stored packing.
    pub fn executes_natively(&self, codec_id: &str) -> bool {
        self.native_codec_ids.contains(codec_id)
    }

    /// Whether this host executes nothing natively — the dense-fallback host.
    pub fn is_dense_only(&self) -> bool {
        self.native_codec_ids.is_empty()
    }

    /// The declared codec ids, sorted.
    pub fn native_codec_ids(&self) -> impl ExactSizeIterator<Item = &str> {
        self.native_codec_ids.iter().map(String::as_str)
    }
}

/// The host-independent identity of the artifact a load read.
///
/// Mirrors the engine's `SourceBinding::stable_token` rendering — `"<file-name>@<size-bytes>"` —
/// which is deliberately free of the full canonical path (machine-local, and separators differ) and
/// of every `cfg`-gated stat field (inode on unix, file id on windows, so a token carrying them
/// would compare unequal across hosts for identical bytes).
///
/// This is an **identity label, not a content hash**. Two different files of equal name and size
/// render the same token; the actual verification is the engine's pin re-check, or SceneWorks' own
/// `SourceLocatorV1` sha256, both of which ran before this token was built.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceBindingFact {
    /// `"<file-name>@<size-bytes>"`.
    pub stable_token: String,
    /// The resolved target's size in bytes — the one identity field every platform has.
    pub size_bytes: u64,
}

impl SourceBindingFact {
    /// Render the token from a file name and size, matching the engine's format exactly. A name
    /// that is not valid UTF-8 renders as `"<non-utf8-name>"`, so the token is always printable.
    pub fn new(file_name: Option<&str>, size_bytes: u64) -> Self {
        let name = file_name.unwrap_or("<non-utf8-name>");
        Self {
            stable_token: format!("{name}@{size_bytes}"),
            size_bytes,
        }
    }
}

impl fmt::Display for SourceBindingFact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.stable_token)
    }
}

/// One row of the **source inventory**: what the file stores, answered the same on every host.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceCodecFact {
    pub codec_id: String,
    /// Logical tensors stored in this codec. `None` when the producer knows the codec but not the
    /// topology — a SceneWorks-side classification from the safetensors header proves *which* codec
    /// a checkpoint is stored in without compiling a plan, and inventing a count there would be
    /// asserting something never counted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tensor_count: Option<usize>,
    /// Source bytes of those tensors plus their companions. `None` for the same reason as above.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_bytes: Option<u64>,
}

impl SourceCodecFact {
    /// A codec row with no topology — the header-classification case.
    pub fn declared(codec_id: impl Into<String>) -> Self {
        Self {
            codec_id: codec_id.into(),
            tensor_count: None,
            source_bytes: None,
        }
    }

    /// A codec row counted from a compiled plan.
    pub fn counted(codec_id: impl Into<String>, tensor_count: usize, source_bytes: u64) -> Self {
        Self {
            codec_id: codec_id.into(),
            tensor_count: Some(tensor_count),
            source_bytes: Some(source_bytes),
        }
    }
}

/// One **measured** row: a `(codec, representation)` pair and what it left resident.
///
/// A `nvfp4-v1` row that ran the packed W4A4 operand and a `nvfp4-v1` row that decoded to dense
/// BF16 are *different rows here*, never one indistinguishable total.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MaterializedCodecFact {
    pub codec_id: String,
    pub representation: ExecutionRepresentation,
    pub tensor_count: usize,
    pub source_bytes: u64,
    pub resident_bytes: u64,
}

/// Why this host could not report what it materialized.
///
/// An explicit reason rather than a bare absence: "the runtime did not hand back a receipt" and
/// "this source shape has no receipt to give" are different situations, and a consumer that renders
/// them identically at least does so knowingly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MaterializationUnavailable {
    /// The linked runtime exposes no checkpoint-facts accessor on the handle this lane holds, so
    /// nothing measured the load. Distinct from a dense run: nobody looked.
    NoRuntimeReceipt,
    /// The source is a directory (a diffusers tree, or a packed-tier variant resolved to a folder).
    /// The engine's accessors answer `Ok(None)` for these by construction — there is no single
    /// pinned file and no compiled plan to measure against.
    DirectorySourcedImport,
}

impl MaterializationUnavailable {
    pub fn label(self) -> &'static str {
        match self {
            Self::NoRuntimeReceipt => "no-runtime-receipt",
            Self::DirectorySourcedImport => "directory-sourced-import",
        }
    }
}

impl fmt::Display for MaterializationUnavailable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Fact 3: what this host actually materialized — or an explicit statement that nothing measured it.
///
/// There is no `Default`. A consumer must handle both arms, which is the whole point: the failure
/// this story exists to prevent is a missing measurement rendering as a confident one.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "camelCase")]
pub enum MaterializationFact {
    /// Measured rows, sorted by `(codec_id, representation)`. Never empty — see
    /// [`CheckpointWeightFactsError::EmptyReportedMaterialization`].
    #[serde(rename = "reported")]
    Reported {
        rows: Vec<MaterializedCodecFact>,
        /// Whether the receipt covers the plan's whole tensor surface.
        complete: bool,
    },
    /// Nothing measured this load. Never rendered as dense and never as native.
    #[serde(rename = "unavailable")]
    Unavailable { reason: MaterializationUnavailable },
}

/// Why a set of checkpoint facts is not self-consistent. None of these is recoverable by rounding:
/// each one means a consumer was about to be told something untrue about the load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CheckpointWeightFactsError {
    /// A codec field was given something that is not a codec id — most importantly a *request
    /// tier* (`q4`, `nvfp4`, `bf16`), which is the exact aliasing this story forbids.
    NotACodecId { value: String },
    /// The source inventory lists one codec twice.
    DuplicateSourceCodec { codec_id: String },
    /// The source inventory is empty. A fact set that declares nothing about the source cannot
    /// correlate anything with it; omit the facts entirely instead.
    EmptySourceInventory,
    /// A materialization row reports a codec the source does not declare — the receipt is
    /// *aliasing* the source.
    UnplannedCodec { codec_id: String },
    /// The same `(codec, representation)` pair is reported twice.
    DuplicateMaterializedRow {
        codec_id: String,
        representation: ExecutionRepresentation,
    },
    /// A row labels tensors natively packed that this host declares no native execution for. With a
    /// dense-only capability this is precisely "a dense fallback labelled native".
    NativeWithoutCapability {
        codec_id: String,
        tensor_count: usize,
    },
    /// A `Reported` materialization carries no rows at all. "Measured, and it materialized nothing"
    /// is not a thing a load can be; the honest statement for a load nobody measured is
    /// [`MaterializationFact::Unavailable`], and an empty `Reported` was silently equivalent to a
    /// measured DENSE run — [`CheckpointWeightFactsV1::executes_natively`] answered `Some(false)`
    /// and the label rendered "dense fallback" — which is the fabricated-measurement mirror of the
    /// lie this module exists to prevent (sc-11045 review).
    EmptyReportedMaterialization,
    /// The payload declares a `schemaVersion` this build cannot read. A future version may add
    /// fields whose absence changes what the rest of the payload MEANS, so decoding it as v1 and
    /// dropping them would produce a confidently-wrong fact set — exactly the class this module
    /// exists to make unrepresentable. Refuse instead (sc-21484 review).
    UnsupportedSchemaVersion { version: u32 },
}

impl fmt::Display for CheckpointWeightFactsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotACodecId { value } => write!(
                f,
                "checkpoint weight facts: {value:?} is not a codec id; a source codec \
                 (e.g. {NVFP4_CODEC_ID:?}) and a request tier (e.g. \"nvfp4\", \"q4\") are \
                 different vocabularies and must not be substituted for one another"
            ),
            Self::DuplicateSourceCodec { codec_id } => write!(
                f,
                "checkpoint weight facts: the source inventory lists codec {codec_id:?} twice; one \
                 codec is one row"
            ),
            Self::EmptySourceInventory => write!(
                f,
                "checkpoint weight facts: the source inventory is empty; facts that declare \
                 nothing about the source cannot correlate anything with it"
            ),
            Self::UnplannedCodec { codec_id } => write!(
                f,
                "checkpoint weight facts: a materialization row reports codec {codec_id:?}, which \
                 the source inventory does not declare; a receipt may report how the source was \
                 materialized, never re-label what the source is"
            ),
            Self::DuplicateMaterializedRow {
                codec_id,
                representation,
            } => write!(
                f,
                "checkpoint weight facts: codec {codec_id:?} reports two `{representation}` rows; \
                 one (codec, representation) pair is one row"
            ),
            Self::NativeWithoutCapability {
                codec_id,
                tensor_count,
            } => write!(
                f,
                "checkpoint weight facts: {tensor_count} tensor(s) of codec {codec_id:?} are \
                 labelled `{}`, but this host declares no native execution for that codec — a \
                 non-native host runs the declared dense fallback and its receipt must say so",
                ExecutionRepresentation::NativePacked
            ),
            Self::EmptyReportedMaterialization => write!(
                f,
                "checkpoint weight facts: a `reported` materialization carries no rows; a load \
                 nobody measured is `unavailable`, not a measurement of nothing"
            ),
            Self::UnsupportedSchemaVersion { version } => write!(
                f,
                "checkpoint weight facts: payload declares schemaVersion {version}; this build \
                 reads {CHECKPOINT_WEIGHT_FACTS_VERSION}"
            ),
        }
    }
}

impl std::error::Error for CheckpointWeightFactsError {}

/// The three correlated facts about one loaded checkpoint, validated against each other.
///
/// Construct with [`Self::try_new`]; the fields are private and there is no field-wise constructor,
/// so a consumer can never be handed an unvalidated set. Deserialization goes through the same
/// validation (see the `Deserialize` impl), so a hand-edited sidecar cannot smuggle a lie back in
/// either.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CheckpointWeightFactsV1 {
    schema_version: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_binding: Option<SourceBindingFact>,
    capability: NativeExecutionCapabilityFact,
    source: Vec<SourceCodecFact>,
    materialization: MaterializationFact,
}

/// The wire form, used only as the `Deserialize` staging shape so decoding runs the same validation
/// `try_new` does.
///
/// `schema_version` is carried explicitly rather than ignored (sc-21484 review). The serialized
/// form has always emitted `schemaVersion`, but a staging struct without the field silently
/// DROPPED it, so a v2 payload decoded as a v1 fact set — a future revision's added fields would
/// vanish while the object still claimed to be a complete, validated receipt. A payload that
/// predates the field (or omits it) is v1 by [`Default`]; a payload from a newer writer is
/// refused.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CheckpointWeightFactsWire {
    #[serde(default = "default_schema_version")]
    schema_version: u32,
    #[serde(default)]
    source_binding: Option<SourceBindingFact>,
    capability: NativeExecutionCapabilityFact,
    source: Vec<SourceCodecFact>,
    materialization: MaterializationFact,
}

/// An absent `schemaVersion` is v1 — the only version that has ever been written.
fn default_schema_version() -> u32 {
    1
}

impl Default for CheckpointWeightFactsWire {
    fn default() -> Self {
        Self {
            schema_version: default_schema_version(),
            source_binding: None,
            capability: NativeExecutionCapabilityFact::dense_only(),
            source: Vec::new(),
            materialization: MaterializationFact::Unavailable {
                reason: MaterializationUnavailable::NoRuntimeReceipt,
            },
        }
    }
}

impl<'de> Deserialize<'de> for CheckpointWeightFactsV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = CheckpointWeightFactsWire::deserialize(deserializer)?;
        if wire.schema_version == 0 || wire.schema_version > CHECKPOINT_WEIGHT_FACTS_VERSION {
            return Err(serde::de::Error::custom(
                CheckpointWeightFactsError::UnsupportedSchemaVersion {
                    version: wire.schema_version,
                },
            ));
        }
        Self::try_new(
            wire.source_binding,
            wire.capability,
            wire.source,
            wire.materialization,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl CheckpointWeightFactsV1 {
    /// Assemble and validate. See [the honesty rules](self#the-honesty-rules) for what is refused.
    pub fn try_new(
        source_binding: Option<SourceBindingFact>,
        capability: NativeExecutionCapabilityFact,
        source: Vec<SourceCodecFact>,
        materialization: MaterializationFact,
    ) -> Result<Self, CheckpointWeightFactsError> {
        if source.is_empty() {
            return Err(CheckpointWeightFactsError::EmptySourceInventory);
        }
        let mut declared: BTreeSet<&str> = BTreeSet::new();
        for entry in &source {
            if !is_codec_id(&entry.codec_id) {
                return Err(CheckpointWeightFactsError::NotACodecId {
                    value: entry.codec_id.clone(),
                });
            }
            if !declared.insert(entry.codec_id.as_str()) {
                return Err(CheckpointWeightFactsError::DuplicateSourceCodec {
                    codec_id: entry.codec_id.clone(),
                });
            }
        }
        if let MaterializationFact::Reported { rows, .. } = &materialization {
            if rows.is_empty() {
                return Err(CheckpointWeightFactsError::EmptyReportedMaterialization);
            }
            let mut seen: BTreeSet<(&str, ExecutionRepresentation)> = BTreeSet::new();
            for row in rows {
                if !is_codec_id(&row.codec_id) {
                    return Err(CheckpointWeightFactsError::NotACodecId {
                        value: row.codec_id.clone(),
                    });
                }
                if !declared.contains(row.codec_id.as_str()) {
                    return Err(CheckpointWeightFactsError::UnplannedCodec {
                        codec_id: row.codec_id.clone(),
                    });
                }
                if !seen.insert((row.codec_id.as_str(), row.representation)) {
                    return Err(CheckpointWeightFactsError::DuplicateMaterializedRow {
                        codec_id: row.codec_id.clone(),
                        representation: row.representation,
                    });
                }
                if row.representation == ExecutionRepresentation::NativePacked
                    && row.tensor_count > 0
                    && !capability.executes_natively(&row.codec_id)
                {
                    return Err(CheckpointWeightFactsError::NativeWithoutCapability {
                        codec_id: row.codec_id.clone(),
                        tensor_count: row.tensor_count,
                    });
                }
            }
        }
        Ok(Self {
            schema_version: CHECKPOINT_WEIGHT_FACTS_VERSION,
            source_binding,
            capability,
            source,
            materialization,
        })
    }

    /// The convenience shape for the common SceneWorks producer: a source classified from its
    /// header (so one declared codec, no topology), a probed host capability, and no receipt.
    pub fn declared_source(
        source_binding: Option<SourceBindingFact>,
        capability: NativeExecutionCapabilityFact,
        codec_id: impl Into<String>,
        reason: MaterializationUnavailable,
    ) -> Result<Self, CheckpointWeightFactsError> {
        Self::try_new(
            source_binding,
            capability,
            vec![SourceCodecFact::declared(codec_id)],
            MaterializationFact::Unavailable { reason },
        )
    }

    pub fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Fact 1: the verified source binding, when the producer supplied one.
    pub fn source_binding(&self) -> Option<&SourceBindingFact> {
        self.source_binding.as_ref()
    }

    /// This host's native-execution declaration.
    pub fn capability(&self) -> &NativeExecutionCapabilityFact {
        &self.capability
    }

    /// Fact 2: what the source stores.
    pub fn source(&self) -> &[SourceCodecFact] {
        &self.source
    }

    /// The source inventory row for one codec.
    pub fn source_entry(&self, codec_id: &str) -> Option<&SourceCodecFact> {
        self.source.iter().find(|entry| entry.codec_id == codec_id)
    }

    /// Whether the source stores any tensor in this codec — the **source** question, answered the
    /// same on every host, including hosts that cannot execute it natively.
    pub fn declares(&self, codec_id: &str) -> bool {
        self.source_entry(codec_id).is_some()
    }

    /// Fact 3: what this host materialized, or why that is unknown.
    pub fn materialization(&self) -> &MaterializationFact {
        &self.materialization
    }

    /// The measured row for one `(codec, representation)` pair. `None` when unavailable *or* when
    /// no such row was reported — both mean "do not claim this happened".
    pub fn materialized_as(
        &self,
        codec_id: &str,
        representation: ExecutionRepresentation,
    ) -> Option<&MaterializedCodecFact> {
        match &self.materialization {
            MaterializationFact::Reported { rows, .. } => rows
                .iter()
                .find(|row| row.codec_id == codec_id && row.representation == representation),
            MaterializationFact::Unavailable { .. } => None,
        }
    }

    /// Whether **any** tensor of this codec actually executed in its stored packing.
    ///
    /// **Tri-state on purpose.** `Some(true)` = measured native, `Some(false)` = measured and it
    /// took the dense fallback, `None` = nothing measured it. This is the question a user-visible
    /// model fact must ask before saying "native NVFP4", and answering `false` for the unmeasured
    /// case would assert a dense run nobody observed — the mirror image of the lie this story
    /// exists to prevent.
    pub fn executes_natively(&self, codec_id: &str) -> Option<bool> {
        match &self.materialization {
            MaterializationFact::Reported { rows, .. } => Some(rows.iter().any(|row| {
                row.codec_id == codec_id
                    && row.representation == ExecutionRepresentation::NativePacked
                    && row.tensor_count > 0
            })),
            MaterializationFact::Unavailable { .. } => None,
        }
    }

    /// Total measured resident bytes across every reported row; `None` when unavailable.
    pub fn resident_bytes(&self) -> Option<u64> {
        match &self.materialization {
            MaterializationFact::Reported { rows, .. } => {
                Some(rows.iter().map(|row| row.resident_bytes).sum())
            }
            MaterializationFact::Unavailable { .. } => None,
        }
    }

    /// **How much** of this codec ran packed and how much took the dense fallback, as a single
    /// summary. `None` when nothing measured the load.
    ///
    /// This is the fact [`Self::executes_natively`] deliberately loses. That predicate is
    /// `any`-shaped — the right shape for "may this run be called native at all" — but a receipt
    /// that reports BOTH a non-empty `native-packed` row and a non-empty `dense-fallback` row
    /// describes a genuinely mixed load, and collapsing it to `true` rendered as "native (packed)"
    /// on a user-facing surface. That is untrue about the majority of the tensors whenever the
    /// engine's shipping policy is mixed, and the pinned engine documents exactly such a policy for
    /// this leg (a minority of projections executing the packed W4A4 operand, the remainder
    /// decoded to dense W4A16).
    ///
    /// Zero-count rows assert nothing (the same boundary [`Self::try_new`] draws when it lets an
    /// empty native row skip the capability check), so they never make a load "mixed".
    pub fn execution_mix(&self, codec_id: &str) -> Option<ExecutionMix> {
        let MaterializationFact::Reported { rows, .. } = &self.materialization else {
            return None;
        };
        let tensors = |representation| {
            rows.iter()
                .filter(|row| row.codec_id == codec_id && row.representation == representation)
                .map(|row| row.tensor_count)
                .sum::<usize>()
        };
        let native_tensors = tensors(ExecutionRepresentation::NativePacked);
        let dense_tensors = tensors(ExecutionRepresentation::DenseFallback);
        Some(match (native_tensors, dense_tensors) {
            (0, _) => ExecutionMix::DenseFallback,
            (_, 0) => ExecutionMix::NativePacked,
            (native_tensors, dense_tensors) => ExecutionMix::Mixed {
                native_tensors,
                dense_tensors,
            },
        })
    }

    /// The single representation label to render for a codec, or `None` when unknown. Never
    /// synthesized from the capability — a host that *could* run packed may still have taken the
    /// dense path for a reason only the receipt knows.
    ///
    /// A mixed load renders as [`ExecutionMix::Mixed`]'s counted label rather than as either pure
    /// arm; see [`Self::execution_mix`].
    pub fn representation_label(&self, codec_id: &str) -> Option<String> {
        Some(self.execution_mix(codec_id)?.label())
    }
}

/// The `"mixed"` summary kind, and the separator its counted form uses.
///
/// **Not an [`ExecutionRepresentation`].** The two engine wire strings label individual measured
/// ROWS and are shared verbatim across the repo boundary; this is a SceneWorks-side summary OVER
/// those rows, and it is spelled differently on purpose so a reader (and
/// [`ExecutionRepresentation::from_label`]) cannot mistake one vocabulary for the other.
pub const MIXED_EXECUTION_LABEL: &str = "mixed";

/// What a codec's measured rows add up to, across representations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionMix {
    /// Every measured tensor of this codec executed in its stored packing.
    NativePacked,
    /// No measured tensor of this codec executed packed.
    DenseFallback,
    /// Both, with the tensor counts that make the split legible.
    Mixed {
        native_tensors: usize,
        dense_tensors: usize,
    },
}

impl ExecutionMix {
    /// The value a telemetry column carries and a client renders.
    ///
    /// The two pure arms are the engine's own row labels, unchanged — every existing consumer and
    /// every persisted row keeps reading exactly what it read before. The mixed arm is
    /// `"mixed:<packed>/<dense>"`: a distinct value (so a query can separate mixed runs from either
    /// pure one) that also carries the counts, so a surface can say *how* mixed without a second
    /// column.
    pub fn label(self) -> String {
        match self {
            Self::NativePacked => ExecutionRepresentation::NativePacked.label().to_owned(),
            Self::DenseFallback => ExecutionRepresentation::DenseFallback.label().to_owned(),
            Self::Mixed {
                native_tensors,
                dense_tensors,
            } => format!("{MIXED_EXECUTION_LABEL}:{native_tensors}/{dense_tensors}"),
        }
    }

    /// Whether any measured tensor executed in its stored packing — the `any` question, asked
    /// explicitly rather than by collapsing the mix.
    pub fn has_native(self) -> bool {
        matches!(self, Self::NativePacked | Self::Mixed { .. })
    }

    pub fn is_mixed(self) -> bool {
        matches!(self, Self::Mixed { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sm_120 host: consumer Blackwell, the only capability in this leg that lists NVFP4.
    fn sm_120() -> NativeExecutionCapabilityFact {
        NativeExecutionCapabilityFact::new([NVFP4_CODEC_ID]).expect("nvfp4-v1 is a codec id")
    }

    fn nvfp4_source() -> Vec<SourceCodecFact> {
        vec![SourceCodecFact::counted(
            NVFP4_CODEC_ID,
            588,
            32_700_000_000,
        )]
    }

    fn row(
        representation: ExecutionRepresentation,
        tensor_count: usize,
        resident_bytes: u64,
    ) -> MaterializedCodecFact {
        MaterializedCodecFact {
            codec_id: NVFP4_CODEC_ID.to_owned(),
            representation,
            tensor_count,
            source_bytes: 32_700_000_000,
            resident_bytes,
        }
    }

    fn reported(rows: Vec<MaterializedCodecFact>) -> MaterializationFact {
        MaterializationFact::Reported {
            rows,
            complete: true,
        }
    }

    // ---------------------------------------------------------------------------------------
    // The host fact matrix: sm_120 / sm_100 / pre-Blackwell / CPU.
    //
    // Each row constructs the capability that host's probe renders — no model is loaded and no
    // hardware is touched. What the matrix pins is that the SOURCE fact is identical on all four
    // (a checkpoint stored nvfp4-v1 says so everywhere) while only sm_120 can even represent a
    // native run.
    // ---------------------------------------------------------------------------------------

    #[test]
    fn sm_120_may_report_a_native_nvfp4_run() {
        let facts = CheckpointWeightFactsV1::try_new(
            Some(SourceBindingFact::new(
                Some("kreamania_v7.safetensors"),
                32_700_000_000,
            )),
            sm_120(),
            nvfp4_source(),
            reported(vec![row(
                ExecutionRepresentation::NativePacked,
                588,
                6_100_000_000,
            )]),
        )
        .expect("an sm_120 host may license a native NVFP4 row");

        assert!(facts.declares(NVFP4_CODEC_ID));
        assert_eq!(facts.executes_natively(NVFP4_CODEC_ID), Some(true));
        assert_eq!(
            facts.representation_label(NVFP4_CODEC_ID).as_deref(),
            Some("native-packed")
        );
        assert!(facts.capability().executes_natively(NVFP4_CODEC_ID));
        assert_eq!(facts.resident_bytes(), Some(6_100_000_000));
    }

    #[test]
    fn sm_120_may_also_report_a_dense_fallback() {
        // Capability is a licence, not a prediction: an sm_120 host that took the dense path for
        // some other reason still reports what the receipt says.
        let facts = CheckpointWeightFactsV1::try_new(
            None,
            sm_120(),
            nvfp4_source(),
            reported(vec![row(
                ExecutionRepresentation::DenseFallback,
                588,
                32_700_000_000,
            )]),
        )
        .expect("a capable host that ran dense reports dense");
        assert_eq!(facts.executes_natively(NVFP4_CODEC_ID), Some(false));
        assert_eq!(
            facts.representation_label(NVFP4_CODEC_ID).as_deref(),
            Some("dense-fallback")
        );
    }

    #[test]
    fn sm_100_is_dense_only_and_still_declares_the_nvfp4_source() {
        // Datacenter Blackwell HAS FP4 tensor cores, but this leg's kernel is not built or
        // validated for sm_100, so its probe renders a dense-only capability. The source fact is
        // unchanged — the file is still stored nvfp4-v1.
        let sm_100 = NativeExecutionCapabilityFact::dense_only();
        let facts = CheckpointWeightFactsV1::try_new(
            None,
            sm_100,
            nvfp4_source(),
            reported(vec![row(
                ExecutionRepresentation::DenseFallback,
                588,
                32_700_000_000,
            )]),
        )
        .expect("an sm_100 host runs the declared dense fallback");

        assert!(
            facts.declares(NVFP4_CODEC_ID),
            "the source is nvfp4-v1 on every host"
        );
        assert!(facts.capability().is_dense_only());
        assert_eq!(facts.executes_natively(NVFP4_CODEC_ID), Some(false));
    }

    #[test]
    fn pre_blackwell_and_cpu_hosts_declare_the_source_and_never_execute_natively() {
        for host in ["pre-blackwell-sm89", "cpu"] {
            let facts = CheckpointWeightFactsV1::try_new(
                None,
                NativeExecutionCapabilityFact::dense_only(),
                nvfp4_source(),
                reported(vec![row(
                    ExecutionRepresentation::DenseFallback,
                    588,
                    32_700_000_000,
                )]),
            )
            .unwrap_or_else(|error| panic!("{host} must produce facts: {error}"));
            assert!(
                facts.declares(NVFP4_CODEC_ID),
                "{host} still declares the source codec"
            );
            assert_eq!(
                facts.executes_natively(NVFP4_CODEC_ID),
                Some(false),
                "{host} never executes NVFP4 natively"
            );
        }
    }

    #[test]
    fn an_unmeasured_load_is_neither_native_nor_dense() {
        let facts = CheckpointWeightFactsV1::declared_source(
            None,
            sm_120(),
            NVFP4_CODEC_ID,
            MaterializationUnavailable::NoRuntimeReceipt,
        )
        .expect("a declared source with no receipt is a valid fact set");

        assert!(facts.declares(NVFP4_CODEC_ID));
        // Capable host, unmeasured load: still not a native claim.
        assert!(facts.capability().executes_natively(NVFP4_CODEC_ID));
        assert_eq!(
            facts.executes_natively(NVFP4_CODEC_ID),
            None,
            "an unmeasured load must answer `unknown`, never `true` and never `false`"
        );
        assert_eq!(facts.representation_label(NVFP4_CODEC_ID), None);
        assert_eq!(facts.resident_bytes(), None);
    }

    #[test]
    fn a_directory_sourced_import_is_its_own_explicit_arm() {
        let facts = CheckpointWeightFactsV1::declared_source(
            None,
            sm_120(),
            NVFP4_CODEC_ID,
            MaterializationUnavailable::DirectorySourcedImport,
        )
        .expect("a directory-sourced import still has a source fact");
        assert!(matches!(
            facts.materialization(),
            MaterializationFact::Unavailable {
                reason: MaterializationUnavailable::DirectorySourcedImport
            }
        ));
        assert_eq!(facts.executes_natively(NVFP4_CODEC_ID), None);
    }

    // ---------------------------------------------------------------------------------------
    // The two mutations the story names. Both MUST be refused at construction.
    // ---------------------------------------------------------------------------------------

    #[test]
    fn mutation_labelling_a_dense_receipt_native_is_refused() {
        // The mutation: take the dense-only host's fact set and relabel its row `native-packed`.
        let error = CheckpointWeightFactsV1::try_new(
            None,
            NativeExecutionCapabilityFact::dense_only(),
            nvfp4_source(),
            reported(vec![row(
                ExecutionRepresentation::NativePacked,
                588,
                6_100_000_000,
            )]),
        )
        .expect_err("a dense-only host may not label its run native");

        assert_eq!(
            error,
            CheckpointWeightFactsError::NativeWithoutCapability {
                codec_id: NVFP4_CODEC_ID.to_owned(),
                tensor_count: 588,
            }
        );
        assert!(error.to_string().contains("native-packed"));
    }

    #[test]
    fn mutation_aliasing_the_source_to_q4_is_refused() {
        // The mutation, in both directions it can be attempted.
        //
        // (a) Write the request tier into the source inventory.
        for tier in ["q4", "q8", "bf16", "dense", "nvfp4", "int8-convrot", "4"] {
            let error = CheckpointWeightFactsV1::try_new(
                None,
                sm_120(),
                vec![SourceCodecFact::declared(tier)],
                MaterializationFact::Unavailable {
                    reason: MaterializationUnavailable::NoRuntimeReceipt,
                },
            )
            .expect_err("a request tier is not a codec id");
            assert_eq!(
                error,
                CheckpointWeightFactsError::NotACodecId {
                    value: tier.to_owned()
                }
            );
        }

        // (b) Keep a real source codec but have the receipt report a different one — the receipt
        //     re-labelling what the source IS.
        let error = CheckpointWeightFactsV1::try_new(
            None,
            sm_120(),
            nvfp4_source(),
            reported(vec![MaterializedCodecFact {
                codec_id: INT8_PER_ROW_CODEC_ID.to_owned(),
                representation: ExecutionRepresentation::DenseFallback,
                tensor_count: 588,
                source_bytes: 1,
                resident_bytes: 1,
            }]),
        )
        .expect_err("a receipt may not alias the source codec");
        assert_eq!(
            error,
            CheckpointWeightFactsError::UnplannedCodec {
                codec_id: INT8_PER_ROW_CODEC_ID.to_owned()
            }
        );
    }

    #[test]
    fn a_capability_cannot_be_declared_with_a_tier_name() {
        let error = NativeExecutionCapabilityFact::new(["nvfp4"])
            .expect_err("the tier spelling is not the codec id");
        assert_eq!(
            error,
            CheckpointWeightFactsError::NotACodecId {
                value: "nvfp4".to_owned()
            }
        );
        // The codec spelling is what works.
        assert!(NativeExecutionCapabilityFact::new([NVFP4_CODEC_ID]).is_ok());
    }

    // ---------------------------------------------------------------------------------------
    // Wire stability + round trip.
    // ---------------------------------------------------------------------------------------

    #[test]
    fn representation_labels_are_the_engines_wire_strings() {
        assert_eq!(
            ExecutionRepresentation::NativePacked.label(),
            "native-packed"
        );
        assert_eq!(
            ExecutionRepresentation::DenseFallback.label(),
            "dense-fallback"
        );
        for representation in [
            ExecutionRepresentation::NativePacked,
            ExecutionRepresentation::DenseFallback,
        ] {
            assert_eq!(
                ExecutionRepresentation::from_label(representation.label()),
                Some(representation)
            );
            // The label is also what serde writes.
            assert_eq!(
                serde_json::to_value(representation).expect("serializes"),
                serde_json::Value::String(representation.label().to_owned())
            );
        }
        // A tier spelling never parses as a representation.
        for tier in ["nvfp4", "q4", "bf16", "native", "dense"] {
            assert_eq!(ExecutionRepresentation::from_label(tier), None);
        }
    }

    #[test]
    fn the_source_codec_id_is_not_the_tier_spelling() {
        assert_eq!(NVFP4_CODEC_ID, "nvfp4-v1");
        assert!(is_codec_id(NVFP4_CODEC_ID));
        assert!(
            !is_codec_id("nvfp4"),
            "the tier spelling must not pass as a codec id"
        );
    }

    #[test]
    fn facts_round_trip_through_json_and_revalidate_on_the_way_back() {
        let facts = CheckpointWeightFactsV1::try_new(
            Some(SourceBindingFact::new(
                Some("kreamania_v7.safetensors"),
                32_700_000_000,
            )),
            sm_120(),
            nvfp4_source(),
            reported(vec![row(
                ExecutionRepresentation::NativePacked,
                588,
                6_100_000_000,
            )]),
        )
        .expect("valid");

        let json = serde_json::to_string(&facts).expect("serializes");
        assert!(json.contains("\"native-packed\""));
        assert!(json.contains("\"nvfp4-v1\""));
        assert!(json.contains("\"kreamania_v7.safetensors@32700000000\""));
        let back: CheckpointWeightFactsV1 = serde_json::from_str(&json).expect("round trips");
        assert_eq!(back, facts);

        // A hand-edited sidecar that strips the capability but keeps the native row is refused on
        // the way back in, not silently trusted.
        let tampered = json.replace(
            "\"capability\":{\"nativeCodecIds\":[\"nvfp4-v1\"]}",
            "\"capability\":{\"nativeCodecIds\":[]}",
        );
        assert_ne!(tampered, json, "the tamper must actually apply");
        let error = serde_json::from_str::<CheckpointWeightFactsV1>(&tampered)
            .expect_err("a native row without a capability is refused on deserialize too");
        assert!(
            error.to_string().contains("no native execution"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn the_source_binding_token_matches_the_engine_rendering() {
        let binding = SourceBindingFact::new(Some("model.safetensors"), 4096);
        assert_eq!(binding.stable_token, "model.safetensors@4096");
        assert_eq!(binding.to_string(), "model.safetensors@4096");
        assert_eq!(
            SourceBindingFact::new(None, 7).stable_token,
            "<non-utf8-name>@7"
        );
    }

    #[test]
    fn an_empty_source_inventory_is_refused() {
        let error = CheckpointWeightFactsV1::try_new(
            None,
            sm_120(),
            Vec::new(),
            MaterializationFact::Unavailable {
                reason: MaterializationUnavailable::NoRuntimeReceipt,
            },
        )
        .expect_err("empty inventories are refused");
        assert_eq!(error, CheckpointWeightFactsError::EmptySourceInventory);
    }

    /// An empty `reported` receipt is refused — it is a FABRICATED measurement (sc-11045 review).
    ///
    /// It used to be accepted, and it was silently equivalent to a measured dense run:
    /// `executes_natively` answered `Some(false)` and the label rendered "dense fallback" for a
    /// load nobody measured. That is the same class of untruth as labelling a dense run native,
    /// only pointing the other way — and it is exactly the reading the explicit `Unavailable` arm
    /// exists to keep separate.
    ///
    /// Failing mutation (run): delete the `if rows.is_empty()` guard in `try_new`. The
    /// construction succeeds and the two assertions after it read "measured dense".
    #[test]
    fn an_empty_reported_materialization_is_refused_on_both_paths() {
        let error = CheckpointWeightFactsV1::try_new(
            None,
            sm_120(),
            nvfp4_source(),
            MaterializationFact::Reported {
                rows: Vec::new(),
                complete: true,
            },
        )
        .expect_err("a measurement of nothing is not a measurement");
        assert_eq!(
            error,
            CheckpointWeightFactsError::EmptyReportedMaterialization
        );
        assert!(error.to_string().contains("unavailable"));

        // The deserializer runs the same validation, so a hand-edited sidecar cannot smuggle one
        // back in either.
        let valid = CheckpointWeightFactsV1::try_new(
            None,
            sm_120(),
            nvfp4_source(),
            reported(vec![row(ExecutionRepresentation::NativePacked, 588, 1)]),
        )
        .expect("valid");
        let json = serde_json::to_string(&valid).expect("serializes");
        let mut value: serde_json::Value = serde_json::from_str(&json).expect("json");
        value["materialization"]["rows"] = serde_json::json!([]);
        let error = serde_json::from_value::<CheckpointWeightFactsV1>(value)
            .expect_err("an emptied receipt is refused on the way back in");
        assert!(
            error.to_string().contains("no rows"),
            "unexpected error: {error}"
        );

        // The honest statement for the same situation still constructs.
        CheckpointWeightFactsV1::try_new(
            None,
            sm_120(),
            nvfp4_source(),
            MaterializationFact::Unavailable {
                reason: MaterializationUnavailable::NoRuntimeReceipt,
            },
        )
        .expect("`unavailable` is how a load nobody measured is stated");
    }

    #[test]
    fn duplicate_rows_are_refused_on_both_sides() {
        let duplicate_source = CheckpointWeightFactsV1::try_new(
            None,
            sm_120(),
            vec![
                SourceCodecFact::declared(NVFP4_CODEC_ID),
                SourceCodecFact::declared(NVFP4_CODEC_ID),
            ],
            MaterializationFact::Unavailable {
                reason: MaterializationUnavailable::NoRuntimeReceipt,
            },
        )
        .expect_err("one codec is one source row");
        assert_eq!(
            duplicate_source,
            CheckpointWeightFactsError::DuplicateSourceCodec {
                codec_id: NVFP4_CODEC_ID.to_owned()
            }
        );

        let duplicate_row = CheckpointWeightFactsV1::try_new(
            None,
            sm_120(),
            nvfp4_source(),
            reported(vec![
                row(ExecutionRepresentation::DenseFallback, 1, 1),
                row(ExecutionRepresentation::DenseFallback, 2, 2),
            ]),
        )
        .expect_err("one (codec, representation) pair is one row");
        assert_eq!(
            duplicate_row,
            CheckpointWeightFactsError::DuplicateMaterializedRow {
                codec_id: NVFP4_CODEC_ID.to_owned(),
                representation: ExecutionRepresentation::DenseFallback,
            }
        );
    }

    /// A receipt reporting BOTH a non-empty packed row and a non-empty dense row describes a MIXED
    /// load, and it must render as one (sc-11045, epic 11037, major 2-SW).
    ///
    /// `executes_natively` is `any`-shaped, so a mixed receipt collapsed to `Some(true)` and the
    /// single label became `"native-packed"` → "native (packed)". The pinned engine documents a
    /// mixed shipping policy for this leg (a minority of projections on the packed W4A4 operand,
    /// the rest decoded to dense W4A16), so that reading was untrue about the MAJORITY of the
    /// tensors on the very hosts the label is most likely to be read on.
    ///
    /// Failing mutation (run): restore the `any` collapse — make `representation_label` go back to
    /// `match self.executes_natively(codec_id)? { true => native, false => dense }`. The counted
    /// label below becomes `"native-packed"`.
    #[test]
    fn a_receipt_with_both_representations_reports_mixed_with_counts() {
        // The engine README's shipping policy for this leg, as a receipt: 68 of 163 packed.
        let facts = CheckpointWeightFactsV1::try_new(
            None,
            sm_120(),
            nvfp4_source(),
            reported(vec![
                row(ExecutionRepresentation::NativePacked, 68, 2_400_000_000),
                row(ExecutionRepresentation::DenseFallback, 95, 9_100_000_000),
            ]),
        )
        .expect("a capable host may report a mixed load");

        assert_eq!(
            facts.execution_mix(NVFP4_CODEC_ID),
            Some(ExecutionMix::Mixed {
                native_tensors: 68,
                dense_tensors: 95,
            })
        );
        assert_eq!(
            facts.representation_label(NVFP4_CODEC_ID).as_deref(),
            Some("mixed:68/95"),
            "a mixed load must not render as either pure representation"
        );
        assert!(facts
            .execution_mix(NVFP4_CODEC_ID)
            .expect("measured")
            .is_mixed());
        // The `any` predicate is untouched — it still answers the question it asks.
        assert_eq!(facts.executes_natively(NVFP4_CODEC_ID), Some(true));
        assert_eq!(facts.resident_bytes(), Some(11_500_000_000));
    }

    /// The pure arms keep the engine's own wire strings, byte for byte, and the tri-state
    /// "not measured" arm is untouched.
    #[test]
    fn the_pure_and_unmeasured_arms_are_unchanged_by_the_mixed_state() {
        let pure = |representation, tensor_count| {
            CheckpointWeightFactsV1::try_new(
                None,
                sm_120(),
                nvfp4_source(),
                reported(vec![row(representation, tensor_count, 1)]),
            )
            .expect("valid")
        };

        let native = pure(ExecutionRepresentation::NativePacked, 588);
        assert_eq!(
            native.execution_mix(NVFP4_CODEC_ID),
            Some(ExecutionMix::NativePacked)
        );
        assert_eq!(
            native.representation_label(NVFP4_CODEC_ID).as_deref(),
            Some("native-packed")
        );

        let dense = pure(ExecutionRepresentation::DenseFallback, 588);
        assert_eq!(
            dense.execution_mix(NVFP4_CODEC_ID),
            Some(ExecutionMix::DenseFallback)
        );
        assert_eq!(
            dense.representation_label(NVFP4_CODEC_ID).as_deref(),
            Some("dense-fallback")
        );

        // A zero-count row asserts nothing, so it never makes a load "mixed".
        let empty_native = CheckpointWeightFactsV1::try_new(
            None,
            NativeExecutionCapabilityFact::dense_only(),
            nvfp4_source(),
            reported(vec![
                row(ExecutionRepresentation::NativePacked, 0, 0),
                row(ExecutionRepresentation::DenseFallback, 588, 1),
            ]),
        )
        .expect("an empty native row claims nothing");
        assert_eq!(
            empty_native.execution_mix(NVFP4_CODEC_ID),
            Some(ExecutionMix::DenseFallback)
        );
        assert_eq!(
            empty_native.representation_label(NVFP4_CODEC_ID).as_deref(),
            Some("dense-fallback")
        );

        // Unmeasured stays tri-state `None` — never "mixed", never either pure arm.
        let unmeasured = CheckpointWeightFactsV1::declared_source(
            None,
            sm_120(),
            NVFP4_CODEC_ID,
            MaterializationUnavailable::NoRuntimeReceipt,
        )
        .expect("valid");
        assert_eq!(unmeasured.execution_mix(NVFP4_CODEC_ID), None);
        assert_eq!(unmeasured.representation_label(NVFP4_CODEC_ID), None);

        // And a codec the receipt reported nothing for is dense-by-measurement, as before: the
        // rows exist, none of them are this codec's native rows.
        assert_eq!(
            native.execution_mix(DENSE_BF16_CODEC_ID),
            Some(ExecutionMix::DenseFallback)
        );
    }

    /// The mixed label is a SUMMARY, in a vocabulary the engine's row labels do not share.
    #[test]
    fn the_mixed_label_is_not_an_execution_representation() {
        assert_eq!(MIXED_EXECUTION_LABEL, "mixed");
        for label in ["mixed", "mixed:68/95"] {
            assert_eq!(
                ExecutionRepresentation::from_label(label),
                None,
                "{label} must not parse as a measured row's representation"
            );
        }
        assert_eq!(
            ExecutionMix::Mixed {
                native_tensors: 1,
                dense_tensors: 2,
            }
            .label(),
            "mixed:1/2"
        );
        assert!(ExecutionMix::NativePacked.has_native());
        assert!(ExecutionMix::Mixed {
            native_tensors: 1,
            dense_tensors: 2,
        }
        .has_native());
        assert!(!ExecutionMix::DenseFallback.has_native());
    }

    #[test]
    fn a_zero_tensor_native_row_needs_no_capability() {
        // A row that materialized nothing asserts nothing, so it is not a native claim. This is the
        // boundary the `tensor_count > 0` guard draws, pinned so a later refactor cannot widen it
        // into "any native row is fine".
        CheckpointWeightFactsV1::try_new(
            None,
            NativeExecutionCapabilityFact::dense_only(),
            nvfp4_source(),
            reported(vec![row(ExecutionRepresentation::NativePacked, 0, 0)]),
        )
        .expect("an empty native row claims nothing");
    }

    /// The wire form carries `schemaVersion` rather than dropping it, so a payload from a future
    /// writer is refused instead of quietly decoding as v1 with its added fields discarded.
    #[test]
    fn a_future_schema_version_is_refused_and_an_absent_one_is_v1() {
        let facts = CheckpointWeightFactsV1::declared_source(
            None,
            NativeExecutionCapabilityFact::dense_only(),
            NVFP4_CODEC_ID,
            MaterializationUnavailable::NoRuntimeReceipt,
        )
        .expect("a declared source with no receipt is valid");
        let json = serde_json::to_string(&facts).expect("serializes");
        assert!(
            json.contains("\"schemaVersion\":1"),
            "the version must be on the wire: {json}"
        );
        // Round-trip is unchanged: the version survives instead of being dropped and re-stamped.
        let decoded: CheckpointWeightFactsV1 = serde_json::from_str(&json).expect("round-trips");
        assert_eq!(decoded, facts);
        assert_eq!(decoded.schema_version(), CHECKPOINT_WEIGHT_FACTS_VERSION);

        // A payload predating the field decodes as v1 — the only version ever written.
        let mut value: serde_json::Value = serde_json::from_str(&json).expect("json");
        value
            .as_object_mut()
            .expect("object")
            .remove("schemaVersion");
        let without: CheckpointWeightFactsV1 =
            serde_json::from_value(value.clone()).expect("an absent version defaults to v1");
        assert_eq!(without.schema_version(), 1);

        // A newer writer's payload is refused rather than decoded as v1.
        for version in [
            serde_json::json!(CHECKPOINT_WEIGHT_FACTS_VERSION + 1),
            serde_json::json!(99),
            serde_json::json!(0),
        ] {
            let mut future = value.clone();
            future
                .as_object_mut()
                .expect("object")
                .insert("schemaVersion".to_owned(), version.clone());
            let error = serde_json::from_value::<CheckpointWeightFactsV1>(future)
                .expect_err("an unreadable schemaVersion is refused");
            assert!(
                error.to_string().contains("schemaVersion"),
                "unexpected error for {version}: {error}"
            );
        }
    }

    /// `NativeExecutionCapabilityFact::new` refuses a request-tier spelling; deserialization must
    /// refuse it too, or a persisted receipt round-trips a capability the constructor forbids.
    #[test]
    fn a_persisted_tier_spelling_is_refused_by_the_capability_deserializer() {
        let error = serde_json::from_str::<NativeExecutionCapabilityFact>(
            r#"{"nativeCodecIds":["nvfp4"]}"#,
        )
        .expect_err("`nvfp4` is a request tier, not a codec id");
        assert!(
            error.to_string().contains("nvfp4"),
            "unexpected error: {error}"
        );
        // The codec spelling still decodes, and an absent/empty set is the dense-only host.
        let ok = serde_json::from_str::<NativeExecutionCapabilityFact>(
            r#"{"nativeCodecIds":["nvfp4-v1"]}"#,
        )
        .expect("the codec spelling is accepted");
        assert!(ok.executes_natively(NVFP4_CODEC_ID));
        assert!(serde_json::from_str::<NativeExecutionCapabilityFact>("{}")
            .expect("an absent set decodes")
            .is_dense_only());

        // And the whole fact set inherits the refusal, so a tampered receipt cannot smuggle one in.
        let facts = CheckpointWeightFactsV1::declared_source(
            None,
            sm_120(),
            NVFP4_CODEC_ID,
            MaterializationUnavailable::NoRuntimeReceipt,
        )
        .expect("valid");
        let tampered = serde_json::to_string(&facts)
            .expect("serializes")
            .replace("\"nvfp4-v1\"]", "\"nvfp4\"]");
        serde_json::from_str::<CheckpointWeightFactsV1>(&tampered)
            .expect_err("a tier spelling in the capability is refused on deserialize");
    }
}
