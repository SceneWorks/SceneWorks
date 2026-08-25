//! The pinned managed NVFP4 checkpoint variants (sc-11043, epic 11037).
//!
//! This module is **registration, and only registration**. It names the exact upstream artifacts
//! SceneWorks offers to install for itself, and then hands every verb to the epic-20398 universal
//! checkpoint-import contracts rather than owning a second lifecycle:
//!
//! | verb | owner |
//! | --- | --- |
//! | install | [`ManagedIngest`] — staging, checksum, one atomic rename, `compile_managed` |
//! | remove | [`CheckpointPlanStore::remove_managed`] |
//! | identity | [`ImportPlanV1::semantic_digest`](crate::checkpoint_import::ImportPlanV1::semantic_digest) |
//! | format | the engine's registered `nvfp4-v1` codec, via the header classification in [`crate::base_weights`] |
//!
//! Three properties this registry keeps, each of which is a story acceptance criterion:
//!
//! * **Explicit selection only (E2).** The only lookup is [`managed_nvfp4_variant`], keyed on the
//!   variant id a caller already holds. There is deliberately NO `for_family`, `for_provider`,
//!   `default_variant`, or `best_for_host` — an automatic tier chooser has nothing here to reach,
//!   because no function in this module maps a *model* to a variant. That is what makes
//!   auto-selection structurally impossible rather than merely disabled, and it is why a caller
//!   that wants "the NVFP4 build of this model" must name it.
//! * **Never a bit count, never q4.** [`Self::quant_tier`] is the tier's own name and
//!   [`Self::source_codec`] is the engine's codec id. Both are validated against the one
//!   vocabulary ([`QuantFormat::Nvfp4`]), so an entry cannot be edited into `q4` — or into any
//!   tier derived from a number of bits — and still load.
//! * **No conversion.** A variant names bytes that are ALREADY NVFP4 upstream. There is no
//!   convert-at-install branch here and none downstream: the ingest hashes, renames, and compiles.
//!   `size_bytes` and `sha256` describe the pinned upstream file exactly as served.
//!
//! A LINKED copy of the identical file is a separate, equally valid ownership mode and stays
//! entirely outside SceneWorks: [`Self::linked_locator`] exists so a caller (and the test suite)
//! can prove the two compile to the same semantic plan without SceneWorks ever writing to the
//! user's library.

use std::collections::BTreeSet;
use std::sync::OnceLock;

use crate::base_weights::QuantFormat;
use crate::checkpoint_import::{ManagedProvenanceV1, SourceLocatorV1};
use crate::checkpoint_ingest::{ManagedIngest, ManagedIngestError, ManagedInstallV1};
use crate::checkpoint_plan_store::{
    managed_checkpoint_id, CheckpointPlanError, CheckpointPlanStore,
};

/// The provenance `source` kind every variant in this registry is served from.
pub const MANAGED_VARIANT_PROVENANCE_SOURCE: &str = "huggingface";

/// The `importSourceShape` a single pinned transformer file is recorded under.
pub const MANAGED_VARIANT_SOURCE_SHAPE: &str = "transformer_file";

/// The one backend that can execute these variants' packed weights. MLX has no consumer for
/// NVFP4's packed E2M1 nibbles and blocked scales.
pub const MANAGED_VARIANT_BACKEND: &str = "candle";

/// Everything that goes wrong before a variant can be acted on. Each is a registration defect —
/// a mis-typed pin or an entry edited into a tier it is not — never a runtime condition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManagedVariantError {
    reason: String,
}

impl ManagedVariantError {
    fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }

    pub fn reason(&self) -> &str {
        &self.reason
    }
}

impl std::fmt::Display for ManagedVariantError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "[managed-variant] {}", self.reason)
    }
}

impl std::error::Error for ManagedVariantError {}

/// One explicitly named managed variant: a single pinned upstream file, its verified digest, and
/// the identity it keeps once SceneWorks owns a copy of it.
///
/// Deliberately NOT `Default`: every field here is load-bearing, and a defaulted eligibility flag
/// or an empty digest is exactly the failure mode epic 11037's close-out named (a tier derived
/// from a number rather than declared, an eligibility flag that defaults open). Construction goes
/// through [`Self::new`], which validates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManagedCheckpointVariantV1 {
    /// Stable id, and — because a managed checkpoint's identity IS its install id — the directory
    /// SceneWorks owns the bytes under. `managed/<variantId>` is the checkpoint id.
    pub variant_id: String,
    /// What the variant is called in the product. Names the tier explicitly (E8): a user choosing
    /// it must be able to see that it is NVFP4 and not a q4 build.
    pub display_name: String,
    /// The engine provider these weights are served through once imported.
    pub provider: String,
    /// The architecture family the header classifier must agree the file is.
    pub family: String,
    /// The tier's own NAME. Never a bit count, never re-derived from one.
    pub quant_tier: String,
    /// The engine's stable source-codec id for the bytes as stored.
    pub source_codec: String,
    /// The upstream Hugging Face repo.
    pub repo: String,
    /// The pinned upstream commit. A 40-hex commit, never a branch: a moving `main` would let the
    /// bytes under a recorded provenance change without the plan noticing.
    pub revision: String,
    /// The file's path inside the pinned repo revision.
    pub repo_file: String,
    /// Where the file sits inside the managed install directory once committed.
    pub relative_path: String,
    /// The pinned SHA-256 of the upstream file. Verified against the staged bytes at finalize; a
    /// mismatch produces no install at all.
    pub sha256: String,
    /// The pinned file's exact served size, for a download estimate. Not a memory figure.
    pub size_bytes: u64,
}

/// The exact upstream file a variant is pinned to.
///
/// Grouped rather than passed field by field so the four facts that must move together — repo,
/// revision, file, digest — cannot be re-ordered or half-updated at a call site.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PinnedArtifactV1 {
    pub repo: String,
    pub revision: String,
    pub file: String,
    pub sha256: String,
    pub size_bytes: u64,
}

impl PinnedArtifactV1 {
    pub fn new(
        repo: impl Into<String>,
        revision: impl Into<String>,
        file: impl Into<String>,
        sha256: impl Into<String>,
        size_bytes: u64,
    ) -> Self {
        Self {
            repo: repo.into(),
            revision: revision.into(),
            file: file.into(),
            sha256: sha256.into(),
            size_bytes,
        }
    }
}

impl ManagedCheckpointVariantV1 {
    pub fn new(
        variant_id: impl Into<String>,
        display_name: impl Into<String>,
        provider: impl Into<String>,
        family: impl Into<String>,
        pin: PinnedArtifactV1,
    ) -> Result<Self, ManagedVariantError> {
        let relative_path = pin.file.rsplit('/').next().unwrap_or_default().to_owned();
        let variant = Self {
            variant_id: variant_id.into(),
            display_name: display_name.into(),
            provider: provider.into(),
            family: family.into(),
            quant_tier: QuantFormat::Nvfp4.as_str().to_owned(),
            source_codec: QuantFormat::Nvfp4
                .source_codec_id()
                .unwrap_or_default()
                .to_owned(),
            repo: pin.repo,
            revision: pin.revision,
            repo_file: pin.file,
            relative_path,
            sha256: pin.sha256,
            size_bytes: pin.size_bytes,
        };
        variant.validate()?;
        Ok(variant)
    }

    /// This variant's pin, as a unit.
    pub fn pin(&self) -> PinnedArtifactV1 {
        PinnedArtifactV1::new(
            &self.repo,
            &self.revision,
            &self.repo_file,
            &self.sha256,
            self.size_bytes,
        )
    }

    /// Every registration invariant, checked as a whole.
    ///
    /// The tier and codec checks are the mutation surface the story asks for: an entry redirected
    /// to `q4` (or to any tier a bit count could produce) fails here, before it can be offered,
    /// installed, or recorded — it does not silently become a q4 install of NVFP4 bytes.
    pub fn validate(&self) -> Result<(), ManagedVariantError> {
        for (value, label) in [
            (&self.variant_id, "variant id"),
            (&self.display_name, "display name"),
            (&self.provider, "provider"),
            (&self.family, "family"),
            (&self.repo, "repo"),
            (&self.repo_file, "repo file"),
            (&self.relative_path, "relative path"),
        ] {
            if value.trim().is_empty() {
                return Err(ManagedVariantError::new(format!(
                    "{label} must not be blank"
                )));
            }
        }
        if self.quant_tier != QuantFormat::Nvfp4.as_str() {
            return Err(ManagedVariantError::new(format!(
                "{} declares tier {:?}; a managed NVFP4 variant is only ever {:?} — it is never \
                 represented as q4 and never derived from a bit count",
                self.variant_id,
                self.quant_tier,
                QuantFormat::Nvfp4.as_str()
            )));
        }
        if Some(self.source_codec.as_str()) != QuantFormat::Nvfp4.source_codec_id() {
            return Err(ManagedVariantError::new(format!(
                "{} declares source codec {:?}, which is not the codec {:?} decodes",
                self.variant_id,
                self.source_codec,
                QuantFormat::Nvfp4.as_str()
            )));
        }
        if !self.repo.contains('/') || self.repo.starts_with('/') || self.repo.ends_with('/') {
            return Err(ManagedVariantError::new(format!(
                "{} names repo {:?}, which is not an owner/name pair",
                self.variant_id, self.repo
            )));
        }
        if self.revision.len() != 40 || !self.revision.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(ManagedVariantError::new(format!(
                "{} pins revision {:?}; a managed variant pins a 40-hex commit, never a branch",
                self.variant_id, self.revision
            )));
        }
        if self.sha256.len() != 64
            || !self
                .sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(ManagedVariantError::new(format!(
                "{} pins checksum {:?}, which is not a lowercase SHA-256",
                self.variant_id, self.sha256
            )));
        }
        if !self.repo_file.ends_with(&self.relative_path) {
            return Err(ManagedVariantError::new(format!(
                "{} installs {:?} from {:?}; the installed path must be the pinned file",
                self.variant_id, self.relative_path, self.repo_file
            )));
        }
        if crate::jobs_store::is_builtin_image_model(&self.variant_id) {
            return Err(ManagedVariantError::new(format!(
                "{} collides with a builtin model id; an imported variant must carry an id the \
                 family route can reach",
                self.variant_id
            )));
        }
        if self.size_bytes == 0 {
            return Err(ManagedVariantError::new(format!(
                "{} records a zero size for its pinned file",
                self.variant_id
            )));
        }
        Ok(())
    }

    /// The checkpoint id a managed copy of this variant carries.
    pub fn checkpoint_id(&self) -> String {
        managed_checkpoint_id(&self.variant_id)
    }

    /// The catalog model id an installed copy of this variant is registered under.
    ///
    /// Deliberately the variant id. It has to be an id no BUILTIN model owns, because the
    /// imported-provider gate applies only to non-builtin ids — a builtin is routed by its own
    /// id-keyed capabilities and would never consult the family route these variants need.
    /// [`Self::validate`] enforces that, so registering a variant under a builtin's id fails at
    /// registration rather than producing an entry the router silently ignores.
    pub fn imported_model_id(&self) -> &str {
        &self.variant_id
    }

    /// The URL the pinned bytes are served from. Derived from the pin, never stored separately, so
    /// provenance cannot name a revision the download did not use.
    pub fn source_url(&self) -> String {
        format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            self.repo, self.revision, self.repo_file
        )
    }

    /// What the install records about where its bytes came from.
    ///
    /// `reference` and `version_id` both carry the pinned commit, and `file_id` the exact file, so
    /// a persisted plan names the artifact rather than a repo that may have moved. No credential
    /// host: both pinned repos are public and ungated, so no stored credential authorizes them.
    pub fn provenance(&self) -> ManagedProvenanceV1 {
        ManagedProvenanceV1 {
            source: MANAGED_VARIANT_PROVENANCE_SOURCE.to_owned(),
            reference: Some(format!("{}@{}", self.repo, self.revision)),
            url: Some(self.source_url()),
            version_id: Some(self.revision.clone()),
            file_id: Some(self.repo_file.clone()),
            credential_host: None,
        }
    }

    /// The locator a managed copy of this variant resolves through.
    pub fn managed_locator(&self) -> Result<SourceLocatorV1, ManagedVariantError> {
        self.validate()?;
        SourceLocatorV1::managed(
            &self.variant_id,
            &self.relative_path,
            &self.sha256,
            self.provenance(),
        )
        .map_err(|error| ManagedVariantError::new(error.to_string()))
    }

    /// The locator a LINKED copy of the identical bytes resolves through — the user's own file, in
    /// the user's own library, which SceneWorks never writes to or moves.
    ///
    /// Its content digest is this variant's pinned SHA-256, which is precisely why the two
    /// ownership modes compile to the same semantic plan: the semantic identity of a source is its
    /// bytes, and ownership lives only in the binding identity.
    pub fn linked_locator(
        &self,
        root_id: &str,
        relative_path: &str,
    ) -> Result<SourceLocatorV1, ManagedVariantError> {
        self.validate()?;
        SourceLocatorV1::linked(root_id, relative_path, &self.sha256)
            .map_err(|error| ManagedVariantError::new(error.to_string()))
    }

    /// Open the epic-20398 managed ingest for this variant. The caller transfers the pinned file
    /// into [`ManagedIngest::staging_dir`] and then calls [`Self::finalize_install`].
    ///
    /// Pure delegation: the install id is the variant id and the provenance is this registration.
    /// Nothing about staging, atomicity, or refusal is re-owned here.
    pub fn begin_install(
        &self,
        store: &CheckpointPlanStore,
    ) -> Result<ManagedIngest, ManagedIngestError> {
        ManagedIngest::begin(store, &self.variant_id, self.provenance())
    }

    /// Commit a staged transfer, verifying the staged bytes against the PINNED digest.
    ///
    /// This is where "verified file checksum" is enforced. The digest handed to the ingest is the
    /// registry's pin, never a digest recomputed from whatever arrived, so a substituted or
    /// truncated download produces no install directory, no catalog record, and no plan.
    pub fn finalize_install(
        &self,
        ingest: ManagedIngest,
    ) -> Result<ManagedInstallV1, ManagedIngestError> {
        ingest.finalize(&self.relative_path, Some(&self.sha256))
    }

    /// Remove the managed install, its record, and its plan documents.
    ///
    /// Delegation again — and the reason it is safe: [`CheckpointPlanStore::remove_managed`] can
    /// only ever address a directory under the store's own installs root, so no linked source can
    /// be reached by it whatever this registry says.
    pub fn remove(&self, store: &CheckpointPlanStore) -> Result<bool, CheckpointPlanError> {
        store.remove_managed(&self.variant_id)
    }

    /// Confirm a finalized install really is THIS variant: the ownership SceneWorks recorded, the
    /// path it installed, and the digest it was pinned to.
    pub fn confirm_installed(&self, install: &ManagedInstallV1) -> Result<(), ManagedVariantError> {
        self.validate()?;
        if install.install_id != self.variant_id {
            return Err(ManagedVariantError::new(format!(
                "install {:?} is not {:?}",
                install.install_id, self.variant_id
            )));
        }
        if install.checkpoint_id != self.checkpoint_id() {
            return Err(ManagedVariantError::new(format!(
                "install {:?} resolved checkpoint {:?}, not {:?}",
                install.install_id,
                install.checkpoint_id,
                self.checkpoint_id()
            )));
        }
        if install.primary_relative_path != self.relative_path {
            return Err(ManagedVariantError::new(format!(
                "install {:?} committed {:?}, not the pinned {:?}",
                install.install_id, install.primary_relative_path, self.relative_path
            )));
        }
        if install.primary_sha256 != self.sha256 {
            return Err(ManagedVariantError::new(format!(
                "install {:?} committed bytes digesting {:?}, not the pinned {:?}",
                install.install_id, install.primary_sha256, self.sha256
            )));
        }
        Ok(())
    }
}

/// Every registered managed NVFP4 variant, in a stable order.
///
/// Enumeration is what makes them APPEAR (a client renders this list). It is deliberately not a
/// selection: nothing here answers "which variant should this model use", so listing the cohort
/// can never become choosing from it.
pub fn managed_nvfp4_variants() -> &'static [ManagedCheckpointVariantV1] {
    static VARIANTS: OnceLock<Vec<ManagedCheckpointVariantV1>> = OnceLock::new();
    VARIANTS.get_or_init(|| {
        let variants = vec![
            ManagedCheckpointVariantV1::new(
                "nvfp4-krea-2-turbo",
                "Krea 2 Turbo (NVFP4)",
                "krea_2_turbo",
                "krea_2",
                PinnedArtifactV1::new(
                    "Comfy-Org/Krea-2",
                    "952f49d49653cb42e7d6cf7cbfad74738073ec7d",
                    "diffusion_models/krea2_turbo_nvfp4.safetensors",
                    "61527003b2d537055494d01bc8efe51d6e86e64192ba23e3721a5647231fe394",
                    7_673_668_448,
                ),
            )
            .expect("the pinned Krea 2 Turbo NVFP4 variant is well formed"),
            ManagedCheckpointVariantV1::new(
                "nvfp4-flux2-klein-9b-true-v2",
                "FLUX.2 [klein] 9B True V2 (NVFP4)",
                "flux2_klein_9b",
                "flux2",
                PinnedArtifactV1::new(
                    "wikeeyang/Flux2-Klein-9B-True-V2",
                    "9c9fe9880029a4e0c4af5ca7d86e83cdb83eea83",
                    "Flux2-Klein-9B-True-v2-nvfp4mixed.safetensors",
                    "32ab833377c6a6052508ee3d29c1cb0f5cd2eeb369518fb6e740ee35645ecadb",
                    5_616_278_928,
                ),
            )
            .expect("the pinned FLUX.2 Klein NVFP4 variant is well formed"),
        ];
        let ids: BTreeSet<&str> = variants
            .iter()
            .map(|variant| variant.variant_id.as_str())
            .collect();
        assert_eq!(
            ids.len(),
            variants.len(),
            "managed variant ids must be unique: they are install directory names"
        );
        variants
    })
}

/// The ONE lookup: by the id a caller already holds.
///
/// There is no sibling that maps a model, family, backend, or host capability onto a variant, and
/// adding one would be the auto-selection this epic forbids (E2 / SC#5). A caller that wants NVFP4
/// states which artifact it means.
pub fn managed_nvfp4_variant(variant_id: &str) -> Option<&'static ManagedCheckpointVariantV1> {
    managed_nvfp4_variants()
        .iter()
        .find(|variant| variant.variant_id == variant_id)
}
