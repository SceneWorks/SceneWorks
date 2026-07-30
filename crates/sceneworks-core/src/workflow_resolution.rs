//! "Can this machine actually run this workflow?" — the cross-install resolution report
//! (sc-15950, epic 15945).
//!
//! A shared workflow references things that exist on the sender's install and may not exist here.
//! This module answers, per requirement, which of those three it is: **resolved** (usable right
//! now), **installable** (this install knows it and can fetch it), or **missing** (nothing here
//! matches). Input images are a fourth answer — [`RequirementState::UserSupplied`] — because they
//! are never auto-resolved.
//!
//! The doctrine, and the reason this is a report rather than a resolver: **import what resolves,
//! name what does not, never silently substitute.** Nothing here falls back to a default model, a
//! similar LoRA or a nearest resolution. A requirement that does not resolve is reported as not
//! resolved, and an entry that matches nothing stays in the list rather than being dropped —
//! dropping it would make the report claim a recipe that is not the one in the file, which is the
//! silent-loss failure epic 15945 exists to prevent.
//!
//! # Why the catalogs are injected
//!
//! The catalogs live behind `apps/rust-api`: the model catalog is the merged
//! `builtin.models.jsonc` + `user.models.jsonc` with a filesystem install-state sweep stapled on
//! (`models::model_catalog`), the LoRA catalog additionally merges the user's external ComfyUI
//! tree and a project manifest (`loras::lora_catalog`), the Style catalog is
//! `builtin.styles.jsonc` (`styles::styles_catalog`) and the recipe presets are the three-scope
//! merge in `recipe_presets::recipe_preset_catalog`. None of that is reachable from core, and
//! moving the report into the API crate would put it out of reach of import (sc-15949) and of the
//! web contract (sc-15951 / sc-15952). So the lookup is a trait — [`WorkflowCatalogs`] — and the
//! API injects it. [`StaticCatalogs`] is the ready-made implementation for a caller that already
//! has its rows in memory, which the API does.
//!
//! What stays in core is everything that must not drift between callers: the matching ORDER (a
//! Hugging Face repo id before a display name, because a repo id is unambiguous and a name is
//! not), the name normalization ([`normalized_label`]), the mapping from
//! (known, installed, fetchable) to a state, and the sentences.
//!
//! # "In the catalog" and "on disk" are different answers
//!
//! A cache-only resolver never auto-downloads, so a model the catalog knows perfectly well can
//! still be absent from this machine. That middle case is the difference between the feature
//! working and the feature being a tease, so it is [`RequirementState::Installable`] and it
//! carries the [`InstallAction`] that fetches it — never [`RequirementState::Missing`]. The
//! inverse guard matters too: a catalog row this install has no way to fetch is `Missing`, because
//! offering an action that cannot run is its own kind of lie.
//!
//! # The report is computed, never stored
//!
//! Deliberately not persisted beside the envelope sc-15949 writes to `extra.importedWorkflow`.
//! The envelope is a fact about the file and never changes; the report is a fact about THIS
//! machine at THIS moment and stops being true the instant the user installs the model. So it is
//! rebuilt on read, from the stored envelope, every time.

use serde::{Deserialize, Serialize};

use crate::image_request::DEFAULT_STYLE_PRESET;
use crate::workflow_share::{
    WorkflowInput, WorkflowLora, WorkflowShare, INPUT_KIND_CONTROL, INPUT_KIND_MASK,
    INPUT_KIND_REFERENCE, INPUT_KIND_SOURCE, OMITTED_INPUTS, OMITTED_LORAS, OMITTED_PHASES,
    OMITTED_PHASE_LORAS, OMITTED_POSES,
};

// ---------------------------------------------------------------------------
// States
// ---------------------------------------------------------------------------

/// What this install can do about one requirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RequirementState {
    /// Present and usable right now.
    Resolved,
    /// This install's catalog knows it, it is not on disk, and there is a flow that fetches it.
    /// The actionable middle case — see the module docs on why it is not [`Self::Missing`].
    Installable,
    /// Nothing here matches, or nothing here can fetch what does. Named, never substituted.
    Missing,
    /// The user supplies it and nothing resolves it — the input images. Not a failure; a
    /// prerequisite.
    UserSupplied,
}

impl RequirementState {
    /// The wire value, so a caller comparing against a string does not re-derive the mapping.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Resolved => "resolved",
            Self::Installable => "installable",
            Self::Missing => "missing",
            Self::UserSupplied => "userSupplied",
        }
    }

    /// Whether this state stops a one-click replay. Everything except [`Self::Resolved`] does:
    /// an installable requirement needs a download first, a missing one cannot be run at all, and
    /// a user-supplied image is by definition not one click.
    #[must_use]
    pub const fn blocks_replay(self) -> bool {
        !matches!(self, Self::Resolved)
    }
}

/// The existing flow that fetches a catalog-known but absent requirement.
///
/// Supplied by the injected [`WorkflowCatalogs`] rather than built here, so core hardcodes no HTTP
/// route and the report points at the Model Manager's real endpoints
/// (`POST /api/v1/models/:model_id/download`, `POST /api/v1/loras/:lora_id/download`) instead of
/// inventing a mechanism.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InstallAction {
    pub method: String,
    pub path: String,
}

// ---------------------------------------------------------------------------
// The injected catalog lookup
// ---------------------------------------------------------------------------

/// One catalog row, reduced to what the report needs.
///
/// Not a mirror of any manifest: the catalogs are untyped `serde_json::Value` everywhere in the
/// API, and the report only ever asks four questions of a row — what is it called, does it name a
/// Hugging Face repo, is it on disk, and can we fetch it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CatalogEntry {
    /// The local catalog id, so a caller can act on the match (select the model, apply the LoRA).
    pub id: String,
    /// Display name, when the row has one.
    pub name: Option<String>,
    /// `owner/name` on Hugging Face, when the row names one. Read by [`StaticCatalogs`]'s repo
    /// match; a lazier implementation may leave it unset and answer
    /// [`WorkflowCatalogs::lora_by_repo`] however it likes.
    pub repo: Option<String>,
    /// Whether it is usable RIGHT NOW — not merely known. See the module docs.
    pub installed: bool,
    /// The flow that fetches it when `installed` is false. `None` means this install has no way
    /// to get it, which makes the requirement [`RequirementState::Missing`].
    pub install: Option<InstallAction>,
}

impl CatalogEntry {
    /// A row keyed by `id`, not installed, with nothing else known.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            ..Self::default()
        }
    }

    /// Mark the row as present on disk.
    #[must_use]
    pub fn installed(mut self) -> Self {
        self.installed = true;
        self
    }

    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    #[must_use]
    pub fn with_repo(mut self, repo: impl Into<String>) -> Self {
        self.repo = Some(repo.into());
        self
    }

    #[must_use]
    pub fn with_install(mut self, install: InstallAction) -> Self {
        self.install = Some(install);
        self
    }
}

/// What this install's catalogs know. Implemented by the caller that can actually read them.
///
/// Deliberately lookups rather than whole lists: the matching rules (order, normalization) belong
/// in core so import, the `POST /api/v1/workflows/inspect` endpoint and the web all agree, while
/// the catalog READS stay where the catalogs are.
pub trait WorkflowCatalogs {
    /// The model catalog's row for an envelope's `model` slug.
    fn model(&self, slug: &str) -> Option<CatalogEntry>;

    /// The LoRA catalog row whose Hugging Face repo id is `repo`. Tried FIRST, because a repo id
    /// identifies one adapter and a display name does not.
    fn lora_by_repo(&self, repo: &str) -> Option<CatalogEntry>;

    /// The LoRA catalog row whose display name (or catalog id) matches `name`. Compare with
    /// [`normalized_label`] so both sides fold case and separators identically.
    fn lora_by_name(&self, name: &str) -> Option<CatalogEntry>;

    /// The Style catalog row for a `styleId` — a group id or a sub-style id, one flat id-space.
    fn style(&self, id: &str) -> Option<CatalogEntry>;

    /// The merged builtin/global/project recipe-preset row for a `stylePreset` id.
    fn recipe_preset(&self, id: &str) -> Option<CatalogEntry>;
}

/// A [`WorkflowCatalogs`] over plain lists, for a caller that already holds its rows in memory
/// (the API's cached catalog snapshots) and for tests.
///
/// Name matching goes through [`normalized_label`] against both `name` and `id`, so a LoRA a
/// sender displayed as `"Film Grain"` matches a local row whose id is `film_grain`.
#[derive(Debug, Clone, Default)]
pub struct StaticCatalogs {
    pub models: Vec<CatalogEntry>,
    pub loras: Vec<CatalogEntry>,
    pub styles: Vec<CatalogEntry>,
    pub recipe_presets: Vec<CatalogEntry>,
}

fn find_by_id(entries: &[CatalogEntry], id: &str) -> Option<CatalogEntry> {
    entries.iter().find(|entry| entry.id == id).cloned()
}

impl WorkflowCatalogs for StaticCatalogs {
    fn model(&self, slug: &str) -> Option<CatalogEntry> {
        find_by_id(&self.models, slug)
    }

    fn lora_by_repo(&self, repo: &str) -> Option<CatalogEntry> {
        let wanted = repo.trim().to_ascii_lowercase();
        self.loras
            .iter()
            .find(|entry| {
                entry
                    .repo
                    .as_deref()
                    .is_some_and(|candidate| candidate.trim().to_ascii_lowercase() == wanted)
            })
            .cloned()
    }

    fn lora_by_name(&self, name: &str) -> Option<CatalogEntry> {
        let wanted = normalized_label(name);
        if wanted.is_empty() {
            return None;
        }
        self.loras
            .iter()
            .find(|entry| {
                entry
                    .name
                    .as_deref()
                    .is_some_and(|candidate| normalized_label(candidate) == wanted)
                    || normalized_label(&entry.id) == wanted
            })
            .cloned()
    }

    fn style(&self, id: &str) -> Option<CatalogEntry> {
        find_by_id(&self.styles, id)
    }

    fn recipe_preset(&self, id: &str) -> Option<CatalogEntry> {
        find_by_id(&self.recipe_presets, id)
    }
}

/// The comparison key for a free label: case-folded, with every run of non-alphanumeric
/// characters collapsed to one space, trimmed.
///
/// So `"Film Grain"`, `"film-grain"`, `"film_grain"` and `"FILM  GRAIN"` are one name. That
/// forgiveness is deliberate and bounded: a display name typed on one install and a catalog id
/// slugged on another are routinely the same adapter, and refusing to see it would report a
/// perfectly resolvable LoRA as user-trained. It never crosses a word boundary, so
/// `"film grain heavy"` still does not match `"film grain"`.
#[must_use]
pub fn normalized_label(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut pending_space = false;
    for character in value.chars() {
        if character.is_alphanumeric() {
            if pending_space && !out.is_empty() {
                out.push(' ');
            }
            pending_space = false;
            out.extend(character.to_lowercase());
        } else {
            pending_space = true;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Requirements
// ---------------------------------------------------------------------------

/// The model the envelope names, classified.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelRequirement {
    /// The envelope's `model` — a catalog slug, never a resolved weights location. Preserved
    /// verbatim whatever the state, so an unresolved model is still nameable.
    pub slug: String,
    pub state: RequirementState,
    /// The local catalog id that matched. `None` when nothing did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub catalog_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Set only for [`RequirementState::Installable`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub install: Option<InstallAction>,
    /// A sentence fit to show the user.
    pub detail: String,
}

/// Which envelope field a LoRA resolved through.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum LoraMatchedBy {
    /// The Hugging Face repo id. Unambiguous, so it is tried first.
    Repo,
    /// The display name, normalized by [`normalized_label`].
    Name,
}

/// One LoRA the envelope named, classified. Never dropped, whatever the state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LoraRequirement {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<f64>,
    pub state: RequirementState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matched_by: Option<LoraMatchedBy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub catalog_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub install: Option<InstallAction>,
    pub detail: String,
}

/// Which style axis a [`StyleRequirement`] came from.
///
/// Two fields, two catalogs, and they are not interchangeable: `styleId` is the live Style-catalog
/// axis (`builtin.styles.jsonc`, one flat id-space of groups and sub-styles) while `stylePreset`
/// holds a RECIPE PRESET id whenever a preset ran (the API stamps it in
/// `apply_recipe_preset_to_image_payload`) and the inert wire default otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StyleField {
    StyleId,
    StylePreset,
}

/// A style axis the envelope carried, classified.
///
/// A style is never [`RequirementState::Installable`]: there is nothing to fetch. Either this
/// install's catalog has the id or it does not.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StyleRequirement {
    pub field: StyleField,
    pub id: String,
    pub state: RequirementState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Always `None`. Declared for symmetry with the other classes so a reader's
    /// "is there an action?" branch does not need a special case.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub install: Option<InstallAction>,
    pub detail: String,
}

/// An input image the recipe needs, by shape. Always [`RequirementState::UserSupplied`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InputRequirement {
    /// One of the `workflow_share::INPUT_KIND_*` constants.
    pub kind: String,
    pub count: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control_mode: Option<String>,
    pub state: RequirementState,
    /// "Needs a source image." / "Needs 2 reference images." / "Needs a mask."
    pub detail: String,
}

/// A collection the envelope declared but could not record.
///
/// The reason this is a first-class part of the report rather than a footnote: `loras` carries
/// `skip_serializing_if = "Vec::is_empty"`, so a six-LoRA envelope whose LoRAs were dropped over
/// the cap and a genuinely LoRA-free one serialize byte-identically. Without the marker sc-15952
/// would render "no LoRAs" and offer one-click replay for a recipe that had five.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OmittedCollection {
    /// A member of `workflow_share::OMITTED_FIELDS`.
    pub field: String,
    /// What the absence does, and does NOT, mean.
    pub detail: String,
}

/// Everything one shared workflow needs, and what this install can do about each of it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolutionReport {
    pub model: ModelRequirement,
    /// Every LoRA the envelope named, in envelope order. Unresolvable entries are listed, not
    /// dropped.
    pub loras: Vec<LoraRequirement>,
    /// The style axes the envelope carried, live axis (`styleId`) first. Empty when it carried
    /// neither, or only the inert `stylePreset` default (see [`INERT_STYLE_PRESETS`]).
    pub styles: Vec<StyleRequirement>,
    /// The input images the user must supply, by shape.
    pub inputs: Vec<InputRequirement>,
    /// Collections the envelope declared but did not record.
    pub omitted: Vec<OmittedCollection>,
    /// Whether a one-click "use this recipe" may be offered: every classified requirement
    /// [`RequirementState::Resolved`], no input image outstanding, and nothing omitted.
    ///
    /// Deliberately strict. Every `false` here is a case where replaying would quietly produce
    /// something other than what the file describes, which is the one thing this report exists to
    /// prevent.
    pub replayable: bool,
}

impl ResolutionReport {
    /// Every requirement that is not [`RequirementState::Resolved`], as sentences — the "name what
    /// does not resolve" half of the doctrine, in the order a reader would present them.
    #[must_use]
    pub fn unresolved(&self) -> Vec<&str> {
        let mut out = Vec::new();
        if self.model.state.blocks_replay() {
            out.push(self.model.detail.as_str());
        }
        out.extend(
            self.loras
                .iter()
                .filter(|lora| lora.state.blocks_replay())
                .map(|lora| lora.detail.as_str()),
        );
        out.extend(
            self.styles
                .iter()
                .filter(|style| style.state.blocks_replay())
                .map(|style| style.detail.as_str()),
        );
        out.extend(self.inputs.iter().map(|input| input.detail.as_str()));
        out.extend(self.omitted.iter().map(|entry| entry.detail.as_str()));
        out
    }
}

// ---------------------------------------------------------------------------
// The inert `stylePreset` values
// ---------------------------------------------------------------------------

/// The `stylePreset` values that name NOTHING and are therefore not requirements.
///
/// `stylePreset` is only a recipe-preset id when a preset actually ran — the API overwrites the
/// field with the preset id in `apply_recipe_preset_to_image_payload`. With no preset it keeps the
/// wire default [`DEFAULT_STYLE_PRESET`] that `ImageRequest::from_payload` fills in, which means
/// EVERY generated envelope carries it (`crates/sceneworks-worker/src/image_jobs.rs` writes
/// `request.style_preset` into the asset facts the builder reads). Older sidecars wrote the
/// literal `"none"` for the same "no preset" state.
///
/// Looking either up in the preset catalog would fail for every shared image and mark all of them
/// unreplayable, which is not an honest answer — the field names no preset, so there is nothing to
/// resolve and nothing to report.
pub const INERT_STYLE_PRESETS: &[&str] = &[DEFAULT_STYLE_PRESET, "none"];

/// Whether a `stylePreset` value names a preset at all.
fn names_a_preset(value: &str) -> bool {
    let trimmed = value.trim();
    !trimmed.is_empty()
        && !INERT_STYLE_PRESETS
            .iter()
            .any(|inert| trimmed.eq_ignore_ascii_case(inert))
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

/// Classify everything `share` needs against `catalogs`.
///
/// Total and infallible: there is no input for which the right answer is an error. A workflow
/// naming nothing this install has is a perfectly readable report that says so.
#[must_use]
pub fn build_resolution_report(
    share: &WorkflowShare,
    catalogs: &dyn WorkflowCatalogs,
) -> ResolutionReport {
    let model = classify_model(&share.model, catalogs);
    let loras = share
        .loras
        .iter()
        .map(|lora| classify_lora(lora, catalogs))
        .collect::<Vec<_>>();
    let mut styles = Vec::new();
    if let Some(id) = share.style_id.as_deref().map(str::trim) {
        if !id.is_empty() {
            styles.push(classify_style(StyleField::StyleId, id, catalogs));
        }
    }
    if let Some(preset) = share.style_preset.as_deref() {
        if names_a_preset(preset) {
            styles.push(classify_style(
                StyleField::StylePreset,
                preset.trim(),
                catalogs,
            ));
        }
    }
    let inputs = share.inputs.iter().map(classify_input).collect::<Vec<_>>();
    let omitted = share
        .omitted
        .iter()
        .map(|field| OmittedCollection {
            field: field.clone(),
            detail: omission_detail(field),
        })
        .collect::<Vec<_>>();

    let replayable = !model.state.blocks_replay()
        && loras.iter().all(|lora| !lora.state.blocks_replay())
        && styles.iter().all(|style| !style.state.blocks_replay())
        && inputs.is_empty()
        && omitted.is_empty();

    ResolutionReport {
        model,
        loras,
        styles,
        inputs,
        omitted,
        replayable,
    }
}

/// The state a catalog match implies. The one place (known, installed, fetchable) becomes a state,
/// so the model and LoRA classes cannot drift apart.
fn state_for(entry: Option<&CatalogEntry>) -> RequirementState {
    match entry {
        Some(entry) if entry.installed => RequirementState::Resolved,
        // Known but absent, and there is a flow. The actionable middle case.
        Some(entry) if entry.install.is_some() => RequirementState::Installable,
        // Known, absent, and unfetchable — an action here would be a button that cannot work.
        Some(_) | None => RequirementState::Missing,
    }
}

/// The install action to publish: only for [`RequirementState::Installable`], so a resolved
/// requirement never carries a download the caller might offer anyway.
fn install_for(entry: Option<&CatalogEntry>, state: RequirementState) -> Option<InstallAction> {
    match state {
        RequirementState::Installable => entry.and_then(|entry| entry.install.clone()),
        _ => None,
    }
}

fn classify_model(slug: &str, catalogs: &dyn WorkflowCatalogs) -> ModelRequirement {
    let entry = catalogs.model(slug);
    let state = state_for(entry.as_ref());
    let install = install_for(entry.as_ref(), state);
    let label = entry
        .as_ref()
        .and_then(|entry| entry.name.clone())
        .unwrap_or_else(|| format!("`{slug}`"));
    let detail = match state {
        RequirementState::Resolved => format!("{label} is installed on this machine."),
        RequirementState::Installable => format!(
            "{label} is in the model catalog but is not downloaded on this machine; the Model \
             Manager can fetch it."
        ),
        RequirementState::Missing if entry.is_some() => format!(
            "{label} is in the model catalog but this install has no way to fetch it, so this \
             workflow cannot be reproduced here."
        ),
        RequirementState::Missing | RequirementState::UserSupplied => format!(
            "No model matching `{slug}` is in this install's catalog, so this workflow cannot be \
             reproduced here."
        ),
    };
    ModelRequirement {
        slug: slug.to_owned(),
        state,
        catalog_id: entry.as_ref().map(|entry| entry.id.clone()),
        name: entry.and_then(|entry| entry.name),
        install,
        detail,
    }
}

fn classify_lora(lora: &WorkflowLora, catalogs: &dyn WorkflowCatalogs) -> LoraRequirement {
    // Repo before name: `owner/name` identifies one adapter, a display name is whatever the
    // sender's install called it.
    let matched = lora
        .repo
        .as_deref()
        .map(str::trim)
        .filter(|repo| !repo.is_empty())
        .and_then(|repo| catalogs.lora_by_repo(repo))
        .map(|entry| (LoraMatchedBy::Repo, entry))
        .or_else(|| {
            lora.name
                .as_deref()
                .map(str::trim)
                .filter(|name| !name.is_empty())
                .and_then(|name| catalogs.lora_by_name(name))
                .map(|entry| (LoraMatchedBy::Name, entry))
        });
    let entry = matched.as_ref().map(|(_, entry)| entry);
    let state = state_for(entry);
    let install = install_for(entry, state);
    let identity = match (lora.name.as_deref(), lora.repo.as_deref()) {
        (Some(name), _) => format!("`{name}`"),
        (None, Some(repo)) => format!("Hugging Face repo `{repo}`"),
        (None, None) => "an unnamed LoRA".to_owned(),
    };
    let label = entry
        .and_then(|entry| entry.name.clone())
        .unwrap_or_else(|| identity.clone());
    let detail = match state {
        RequirementState::Resolved => format!("{label} is installed."),
        RequirementState::Installable => format!(
            "{label} is in the LoRA catalog but is not downloaded; the Model Manager can fetch it."
        ),
        RequirementState::Missing if entry.is_some() => {
            format!("{label} is in the LoRA catalog but this install has no way to fetch it.")
        }
        RequirementState::Missing | RequirementState::UserSupplied => format!(
            "No LoRA on this install matches {identity}; it was most likely trained by whoever \
             shared the image."
        ),
    };
    LoraRequirement {
        name: lora.name.clone(),
        repo: lora.repo.clone(),
        weight: lora.weight,
        state,
        matched_by: matched.as_ref().map(|(how, _)| *how),
        catalog_id: entry.map(|entry| entry.id.clone()),
        install,
        detail,
    }
}

fn classify_style(
    field: StyleField,
    id: &str,
    catalogs: &dyn WorkflowCatalogs,
) -> StyleRequirement {
    let entry = match field {
        StyleField::StyleId => catalogs.style(id),
        StyleField::StylePreset => catalogs.recipe_preset(id),
    };
    // A style/preset is either in the catalog or it is not — there is nothing to download, so
    // `installed` never enters into it and `Installable` is unreachable by construction.
    let state = if entry.is_some() {
        RequirementState::Resolved
    } else {
        RequirementState::Missing
    };
    let detail = match (field, state) {
        (StyleField::StyleId, RequirementState::Resolved) => {
            format!("Style `{id}` is in this install's Style catalog.")
        }
        (StyleField::StyleId, _) => format!(
            "Style `{id}` is not in this install's Style catalog, so the style it applied cannot \
             be reproduced here."
        ),
        (StyleField::StylePreset, RequirementState::Resolved) => {
            format!("Recipe preset `{id}` is on this install.")
        }
        (StyleField::StylePreset, _) => format!(
            "Recipe preset `{id}` is not on this install, so the prompt wrapping and LoRAs it \
             contributed cannot be reproduced here."
        ),
    };
    StyleRequirement {
        field,
        id: id.to_owned(),
        state,
        name: entry.and_then(|entry| entry.name),
        install: None,
        detail,
    }
}

fn classify_input(input: &WorkflowInput) -> InputRequirement {
    let count = input.count.max(1);
    let label = input_label(input);
    let detail = if count == 1 {
        format!("Needs {} {label}.", article(&label))
    } else {
        format!("Needs {count} {label}s.")
    };
    InputRequirement {
        kind: input.kind.clone(),
        count: input.count,
        control_mode: input.control_mode.clone(),
        state: RequirementState::UserSupplied,
        detail,
    }
}

/// The noun for one input shape, singular. A control map names its conditioning when the original
/// run named one (`canny`, `depth`, …), which is what makes the descriptor actionable rather than
/// merely true.
fn input_label(input: &WorkflowInput) -> String {
    match input.kind.as_str() {
        INPUT_KIND_SOURCE => "source image".to_owned(),
        INPUT_KIND_REFERENCE => "reference image".to_owned(),
        INPUT_KIND_MASK => "mask".to_owned(),
        INPUT_KIND_CONTROL => match input.control_mode.as_deref().map(str::trim) {
            Some(mode) if !mode.is_empty() => format!("{mode} control image"),
            _ => "control image".to_owned(),
        },
        // `reduce_input` drops anything outside `INPUT_KINDS` on parse, so this is only reachable
        // from a hand-built envelope. Still answered rather than panicked on.
        other => format!("{other} image"),
    }
}

/// `a` / `an` for a label, so a `depth` control image does not read as "a edge control image".
fn article(label: &str) -> &'static str {
    match label.chars().next().map(|first| first.to_ascii_lowercase()) {
        Some('a' | 'e' | 'i' | 'o' | 'u') => "an",
        _ => "a",
    }
}

/// What a dropped collection means. One sentence per member of
/// `workflow_share::OMITTED_FIELDS`, each saying what the absence does NOT mean — which is the
/// whole point of the marker.
fn omission_detail(field: &str) -> String {
    match field {
        OMITTED_LORAS => "This workflow named more LoRAs than a job can carry, so none of them \
                          were recorded. This is NOT a LoRA-free recipe."
            .to_owned(),
        OMITTED_INPUTS => "This workflow named more input images than the envelope records, so \
                           none of them were recorded. This is NOT a recipe with no input images."
            .to_owned(),
        OMITTED_POSES => "This workflow carried a pose selection too large to record, so no poses \
                          were recorded. This is NOT a pose-free recipe."
            .to_owned(),
        OMITTED_PHASES => "This workflow carried a multi-phase denoise schedule too large to \
                           record, so no phases were recorded. This is NOT a single-phase recipe."
            .to_owned(),
        OMITTED_PHASE_LORAS => {
            "A denoise phase's own LoRA schedule was too large to record and was dropped. The \
             phases that remain are NOT the whole schedule."
                .to_owned()
        }
        // The parse side keeps only members of the closed vocabulary, so this is unreachable from
        // a file. Answered anyway rather than showing the user a bare field name.
        other => format!("The `{other}` collection was declared but not recorded."),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalized_label_folds_case_and_separators_but_not_words() {
        assert_eq!(normalized_label("Film  Grain"), "film grain");
        assert_eq!(normalized_label("film-grain"), "film grain");
        assert_eq!(normalized_label("FILM_GRAIN"), "film grain");
        assert_eq!(normalized_label("  film / grain  "), "film grain");
        assert_ne!(normalized_label("film grain heavy"), "film grain");
        assert_eq!(normalized_label("---"), "");
    }

    #[test]
    fn state_for_separates_known_absent_from_unknown() {
        assert_eq!(state_for(None), RequirementState::Missing);
        assert_eq!(
            state_for(Some(&CatalogEntry::new("x").installed())),
            RequirementState::Resolved
        );
        assert_eq!(
            state_for(Some(&CatalogEntry::new("x").with_install(InstallAction {
                method: "POST".to_owned(),
                path: "/x".to_owned(),
            }))),
            RequirementState::Installable
        );
        assert_eq!(
            state_for(Some(&CatalogEntry::new("x"))),
            RequirementState::Missing,
            "known but unfetchable is missing, not falsely actionable"
        );
    }

    #[test]
    fn only_the_installable_state_publishes_an_action() {
        let entry = CatalogEntry::new("x")
            .installed()
            .with_install(InstallAction {
                method: "POST".to_owned(),
                path: "/x".to_owned(),
            });
        assert!(install_for(Some(&entry), RequirementState::Resolved).is_none());
        assert!(install_for(Some(&entry), RequirementState::Missing).is_none());
        assert!(install_for(Some(&entry), RequirementState::Installable).is_some());
    }

    #[test]
    fn the_inert_style_presets_name_no_preset() {
        assert!(!names_a_preset(DEFAULT_STYLE_PRESET));
        assert!(!names_a_preset("None"));
        assert!(!names_a_preset("   "));
        assert!(names_a_preset("preset_noir"));
    }

    #[test]
    fn every_state_but_resolved_blocks_replay() {
        assert!(!RequirementState::Resolved.blocks_replay());
        for state in [
            RequirementState::Installable,
            RequirementState::Missing,
            RequirementState::UserSupplied,
        ] {
            assert!(state.blocks_replay(), "{state:?} must block replay");
        }
    }

    #[test]
    fn the_wire_value_matches_what_serde_emits() {
        for state in [
            RequirementState::Resolved,
            RequirementState::Installable,
            RequirementState::Missing,
            RequirementState::UserSupplied,
        ] {
            let serialized = serde_json::to_string(&state).expect("state serializes");
            assert_eq!(serialized, format!("\"{}\"", state.as_str()));
        }
    }

    #[test]
    fn article_reads_correctly_for_a_vowel_initial_control_mode() {
        assert_eq!(article("source image"), "a");
        assert_eq!(article("edge control image"), "an");
        assert_eq!(article("Openpose control image"), "an");
    }
}
