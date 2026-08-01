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
//! not; then the display name before the catalog id, because the envelope's `name` IS a display
//! name), the rule that an ambiguous pass resolves to NOTHING (see [`unique_match`]), the name
//! normalization ([`normalized_label`]), the mapping from (known, installed, fetchable) to a
//! state, and the sentences.
//!
//! # What a LoRA's `name` is allowed to decide
//!
//! `loras[].repo` and `loras[].name` are not two tries at the same question. The repo id names one
//! Hugging Face repo on every install; the display name is whatever the SENDER's install called
//! the adapter, and two installs of the same file routinely disagree about it. So the name is
//! never allowed to overrule the repo — only to say which row, among the ones the repo already
//! selected, was meant:
//!
//! - **The repo matches exactly one local row.** It resolves, and the name is not consulted at
//!   all. A `name` that CONTRADICTS that row is therefore ignored rather than acted on. That is a
//!   deliberate asymmetry, not an oversight: acting on the contradiction would mean either
//!   reporting a row the repo id positively identifies as unresolved, or letting a label the
//!   sender's install chose overturn the one key that means the same thing everywhere. The
//!   envelope does carry a signal here; this reader declines to treat it as one.
//! - **Several local rows claim the repo.** The repo has narrowed the answer to those rows but
//!   cannot pick one — `loras[].source.file` does not travel. The name then breaks the tie WITHIN
//!   that set ([`WorkflowCatalogs::lora_by_name_in_repo`]), because both parties installing the
//!   same multi-adapter pack is the ordinary way a repo becomes ambiguous and is exactly the case
//!   where the names agree. A name matching none of the tied rows, or two of them, resolves to
//!   nothing and is named.
//! - **No local row claims the repo** (or the envelope carries no repo). Nothing has been narrowed,
//!   so the name is the only key there is and it runs over the whole catalog.
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
    /// Exact adapter SHA-256 when this install has worker-verified it.
    pub hash: Option<String>,
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
    pub fn with_hash(mut self, hash: impl Into<String>) -> Self {
        self.hash = Some(hash.into());
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

    /// The unique LoRA catalog row with this exact full SHA-256. A digest-bearing envelope never
    /// falls back to repo or name: doing so could silently substitute different bytes.
    fn lora_by_hash(&self, _hash: &str) -> Option<CatalogEntry> {
        None
    }

    /// The LoRA catalog row whose Hugging Face repo id is `repo`. Tried FIRST, because a repo id
    /// identifies one adapter and a display name does not.
    fn lora_by_repo(&self, repo: &str) -> Option<CatalogEntry>;

    /// Whether MORE THAN ONE row claims `repo` — the question [`Self::lora_by_repo`] cannot
    /// answer, because "nothing matched" and "several matched" reduce to the same `None`.
    ///
    /// It is asked because [`classify_lora`] falls back to the display-name pass when the repo
    /// matched nothing, and an ambiguous repo has to send that fallback somewhere NARROWER than a
    /// plain miss does. It does not switch the fallback off: see
    /// [`Self::lora_by_name_in_repo`] for where it goes instead, and why.
    ///
    /// The ambiguity is structural rather than unlucky: `loras[].source.file` does **not** travel
    /// (see `docs/workflow-share-envelope.md`), so an envelope naming a repo two locally-installed
    /// adapters came out of cannot say which of them the sender used *by repo alone*. Guessing
    /// produces a plausible wrong image, which is worse than reporting the entry unresolvable.
    ///
    /// The converse limit is honest and unclosable from here: a repo holding several adapters of
    /// which this install registered only ONE is not ambiguous to the catalog, so it resolves —
    /// and it resolves even when the envelope's `name` names something else entirely, because the
    /// repo pass wins outright and the name is never consulted. That asymmetry is deliberate, not
    /// an oversight: see the module docs.
    fn lora_repo_ambiguous(&self, repo: &str) -> bool;

    /// The LoRA catalog row whose display name (or catalog id) matches `name`. Compare with
    /// [`normalized_label`] so both sides fold case and separators identically.
    fn lora_by_name(&self, name: &str) -> Option<CatalogEntry>;

    /// The same display-name lookup as [`Self::lora_by_name`], but over ONLY the rows claiming
    /// `repo` — the tie-break [`classify_lora`] runs when [`Self::lora_repo_ambiguous`] said yes.
    ///
    /// Why a restricted pass rather than either extreme. Refusing the name outright makes the
    /// ordinary case unresolvable: **both parties installing the same multi-adapter pack is the
    /// usual way a repo becomes ambiguous**, and it is exactly the case where the two installs'
    /// display names agree and the name is right. Running the UNRESTRICTED pass is the other
    /// error: it lets the weaker key pick a row from some other repo, contradicting the stronger
    /// key that had already narrowed the answer to this repo's rows.
    ///
    /// So the repo keeps its authority — the answer is one of the rows claiming it, or nothing —
    /// and the name is used only to say WHICH of those rows. A unique hit inside the tied set
    /// resolves; zero hits, or a name two of the tied rows share, stays unresolved and is named.
    fn lora_by_name_in_repo(&self, name: &str, repo: &str) -> Option<CatalogEntry>;

    /// The Style catalog row for a `styleId` — a group id or a sub-style id, one flat id-space.
    fn style(&self, id: &str) -> Option<CatalogEntry>;

    /// The merged builtin/global/project recipe-preset row for a `stylePreset` id.
    fn recipe_preset(&self, id: &str) -> Option<CatalogEntry>;
}

/// A [`WorkflowCatalogs`] over plain lists, for a caller that already holds its rows in memory
/// (the API's cached catalog snapshots) and for tests.
///
/// Name matching goes through [`normalized_label`] in TWO passes — display name first, catalog id
/// only if no row matched by name — and a pass that matches more than one row resolves to nothing.
/// See [`unique_match`] for why.
#[derive(Debug, Clone, Default)]
pub struct StaticCatalogs {
    pub models: Vec<CatalogEntry>,
    pub loras: Vec<CatalogEntry>,
    pub styles: Vec<CatalogEntry>,
    pub recipe_presets: Vec<CatalogEntry>,
}

/// The outcome of one matching pass over a catalog.
///
/// Three answers, not two, because "nothing matched" and "several matched" must lead to different
/// places: the first may fall through to the next, weaker pass, the second must not — a weaker key
/// cannot disambiguate what a stronger one already found ambiguous.
enum Match<'a> {
    None,
    One(&'a CatalogEntry),
    /// More than one row matched. Resolves to nothing: see [`unique_match`].
    Ambiguous,
}

/// The single row matching `predicate`, or nothing when zero or MORE THAN ONE row does.
///
/// The "more than one" half is the doctrine, not a nicety. `Iterator::find` returns the first row
/// in catalog order, which is an accident of manifest merge order — so with two rows a sender's
/// label could plausibly mean, picking one is a coin flip presented to the user as a fact, and
/// sc-15952 would offer one-click replay with an adapter the file never named. There is no honest
/// way to choose, so the entry is reported [`RequirementState::Missing`] and NAMED instead. Import
/// what resolves, name what does not, never silently substitute.
fn unique_match<'a>(
    entries: &'a [CatalogEntry],
    mut predicate: impl FnMut(&CatalogEntry) -> bool,
) -> Match<'a> {
    let mut found: Option<&CatalogEntry> = None;
    for entry in entries {
        if !predicate(entry) {
            continue;
        }
        if found.is_some() {
            return Match::Ambiguous;
        }
        found = Some(entry);
    }
    found.map_or(Match::None, Match::One)
}

/// An exact-id lookup, with a blank id matching NOTHING.
///
/// The blank guard is load-bearing: `workflow_share` scrubs an unshareable (path-shaped) `model`
/// to the empty string, and a catalog row whose manifest omitted `id` used to read as `id: ""` —
/// so an envelope that named no model at all matched a row the file never mentioned and was
/// reported "installed on this machine". Both ends are now closed; this is the reader's half.
///
/// Deliberately first-match rather than [`unique_match`]: `id` is the catalogs' primary key
/// (`merge_entries_by_id`), and the live resolvers this report describes — `style_text_for_id`,
/// the preset lookup in `apply_recipe_preset_to_image_payload` — also take the first row. Being
/// STRICTER than the resolver would report a style as missing that the app would in fact apply,
/// which is its own kind of lie.
fn find_by_id(entries: &[CatalogEntry], id: &str) -> Option<CatalogEntry> {
    // The guard is here rather than only at the two call sites so a THIRD id-keyed lookup cannot be
    // added without it. A row is dropped upstream when its manifest carried no id, so a blank here
    // can only come from a scrubbed envelope field — a question with no answer, not a key.
    if id.trim().is_empty() {
        return None;
    }
    entries
        .iter()
        .find(|entry| !entry.id.trim().is_empty() && entry.id == id)
        .cloned()
}

impl StaticCatalogs {
    /// The one repo pass, so [`WorkflowCatalogs::lora_by_repo`] and
    /// [`WorkflowCatalogs::lora_repo_ambiguous`] read the same three-way answer instead of two
    /// predicates that can drift into disagreeing about what "ambiguous" means.
    fn match_lora_repo(&self, repo: &str) -> Match<'_> {
        let wanted = repo.trim().to_ascii_lowercase();
        if wanted.is_empty() {
            return Match::None;
        }
        unique_match(&self.loras, |entry| {
            entry
                .repo
                .as_deref()
                .is_some_and(|candidate| candidate.trim().to_ascii_lowercase() == wanted)
        })
    }

    /// The two-pass name lookup, over the rows `restrict` admits.
    ///
    /// One body for both name methods so the whole-catalog pass and the tied-rows pass cannot
    /// drift into folding labels differently or ordering their two passes differently — the
    /// restriction is the ONLY difference between them.
    fn match_lora_name(
        &self,
        name: &str,
        restrict: &dyn Fn(&CatalogEntry) -> bool,
    ) -> Option<CatalogEntry> {
        let wanted = normalized_label(name);
        if wanted.is_empty() {
            return None;
        }
        // Pass 1: the DISPLAY NAME, which is what the envelope's `name` actually is. A single
        // predicate over `name || id` let a row whose id happened to equal the sender's name win
        // over the row actually CALLED that, purely because it came first in the catalog.
        match unique_match(&self.loras, |entry| {
            restrict(entry)
                && entry
                    .name
                    .as_deref()
                    .is_some_and(|candidate| normalized_label(candidate) == wanted)
        }) {
            Match::One(entry) => return Some(entry.clone()),
            // An ambiguous name pass stops here rather than falling through: an id match is the
            // WEAKER key, so it cannot break a tie the stronger one could not.
            Match::Ambiguous => return None,
            Match::None => {}
        }
        // Pass 2: the catalog id, so a name slugged on one install (`film_grain`) still matches a
        // row that carries no display name. Only reached when NOTHING is called `wanted`.
        match unique_match(&self.loras, |entry| {
            restrict(entry) && normalized_label(&entry.id) == wanted
        }) {
            Match::One(entry) => Some(entry.clone()),
            Match::None | Match::Ambiguous => None,
        }
    }
}

impl WorkflowCatalogs for StaticCatalogs {
    fn model(&self, slug: &str) -> Option<CatalogEntry> {
        find_by_id(&self.models, slug)
    }

    fn lora_by_hash(&self, hash: &str) -> Option<CatalogEntry> {
        let wanted = hash.trim().to_ascii_lowercase();
        if wanted.len() != 64 || !wanted.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return None;
        }
        match unique_match(&self.loras, |entry| {
            entry
                .hash
                .as_deref()
                .is_some_and(|candidate| candidate.trim().eq_ignore_ascii_case(&wanted))
        }) {
            Match::One(entry) => Some(entry.clone()),
            Match::None | Match::Ambiguous => None,
        }
    }

    fn lora_by_repo(&self, repo: &str) -> Option<CatalogEntry> {
        match self.match_lora_repo(repo) {
            Match::One(entry) => Some(entry.clone()),
            // Two rows off one repo (two adapter FILES from the same Hugging Face repo) are two
            // different adapters. The repo id no longer identifies one, so it identifies none.
            Match::None | Match::Ambiguous => None,
        }
    }

    fn lora_repo_ambiguous(&self, repo: &str) -> bool {
        matches!(self.match_lora_repo(repo), Match::Ambiguous)
    }

    fn lora_by_name(&self, name: &str) -> Option<CatalogEntry> {
        self.match_lora_name(name, &|_| true)
    }

    fn lora_by_name_in_repo(&self, name: &str, repo: &str) -> Option<CatalogEntry> {
        let wanted_repo = repo.trim().to_ascii_lowercase();
        if wanted_repo.is_empty() {
            return None;
        }
        // The SAME repo comparison `match_lora_repo` uses, so the set the name is asked to
        // choose within is exactly the set the repo pass called tied. Two spellings of "claims
        // this repo" would let a row be tied for one purpose and absent for the other.
        self.match_lora_name(name, &|entry| {
            entry
                .repo
                .as_deref()
                .is_some_and(|candidate| candidate.trim().to_ascii_lowercase() == wanted_repo)
        })
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
    /// Exact SHA-256 of the adapter bytes. Strongest identity and therefore tried exclusively.
    Hash,
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
    /// **Nothing is missing or omitted — this install can run it.** Every requirement this report
    /// had to look UP ([`model`](Self::model), [`loras`](Self::loras), [`styles`](Self::styles)) is
    /// [`RequirementState::Resolved`], and no collection was dropped on the way into the envelope.
    ///
    /// Deliberately says nothing about input images. They are not a resolution failure — nothing
    /// on this machine could ever have supplied them — so folding them in here would report every
    /// `edit_image` share as unrunnable even when the model and every LoRA are installed and the
    /// user has the source image open. Ask [`input_images_required`](Self::input_images_required)
    /// for that, and [`one_click_replayable`](Self::one_click_replayable) for both at once.
    pub runnable: bool,
    /// **How many images the user must supply before this can run**, summed over
    /// [`inputs`](Self::inputs). `0` for a text-to-image recipe.
    ///
    /// The other half of the old conflated `replayable`: a share this install can run but that
    /// needs one more click is a different answer from one it cannot reproduce at all, and a
    /// single boolean could not tell sc-15952 which it was holding.
    pub input_images_required: u32,
}

impl ResolutionReport {
    /// Both flags at once: this install can run it AND there is nothing left for the user to
    /// supply. The condition under which sc-15952 may offer a true one-click "use this recipe".
    ///
    /// Provided so the composition lives here rather than being re-derived (and eventually
    /// re-derived DIFFERENTLY) by each reader.
    #[must_use]
    pub const fn one_click_replayable(&self) -> bool {
        self.runnable && self.input_images_required == 0
    }

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

    // "This install can run it" — the LOOKED-UP requirements only. Input images are counted
    // separately rather than folded in; see `ResolutionReport::runnable`.
    let runnable = !model.state.blocks_replay()
        && loras.iter().all(|lora| !lora.state.blocks_replay())
        && styles.iter().all(|style| !style.state.blocks_replay())
        && omitted.is_empty();
    // `classify_input` floors a zero/absent count at 1, and the same floor is used here so the
    // total never says "0 images" for an input the report is simultaneously describing.
    let input_images_required = inputs
        .iter()
        .map(|input| input.count.max(1))
        .fold(0u32, u32::saturating_add);

    ResolutionReport {
        model,
        loras,
        styles,
        inputs,
        omitted,
        runnable,
        input_images_required,
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
    // A blank slug names nothing, so it must MATCH nothing. `workflow_share` scrubs a path-shaped
    // `model` to the empty string, and consulting the catalog with it once resolved against an
    // id-less catalog row — reporting a model the envelope never named as "installed on this
    // machine". The catalog is not asked at all rather than being asked a question with no answer.
    let names_a_model = !slug.trim().is_empty();
    let entry = if names_a_model {
        catalogs.model(slug)
    } else {
        None
    };
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
        // The envelope named no model AT ALL. `workflow_share` reduces an unshareable value — a
        // local file path, an over-long label — to the empty string, so there is no slug to have
        // missed with. "No model matching ``" would report a catalog miss for something that was
        // never a catalog key, and it reads to a user like a bug in the reader.
        RequirementState::Missing | RequirementState::UserSupplied if !names_a_model => {
            "This image does not name a model. The sender's model was a local file path or was \
             otherwise not shareable, so it was left out of the file and the recipe cannot be \
             reproduced here."
                .to_owned()
        }
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
    let repo = lora
        .repo
        .as_deref()
        .map(str::trim)
        .filter(|repo| !repo.is_empty());
    // A repo SEVERAL local rows claim is not a miss, and the difference decides WHERE the name
    // pass may look — not whether it runs at all. `source.file` does not travel, so two adapters
    // off one repo are two answers the repo id alone cannot choose between; the display name may
    // break that tie, but only among the tied rows. Letting it reach OUTSIDE the repo would be the
    // weaker key overruling the stronger one, which had already narrowed the answer to this repo.
    //
    // Both halves matter. Both parties installing the same multi-adapter pack is the ordinary way
    // a repo becomes ambiguous, and it is precisely the case where the names agree — refusing the
    // name there reports "applying one would be a guess" about a recipe the envelope named exactly.
    let repo_ambiguous = repo.is_some_and(|repo| catalogs.lora_repo_ambiguous(repo));
    let name = lora
        .name
        .as_deref()
        .map(str::trim)
        .filter(|name| !name.is_empty());
    let matched = if let Some(hash) = lora.hash.as_deref() {
        catalogs
            .lora_by_hash(hash)
            .map(|entry| (LoraMatchedBy::Hash, entry))
    } else {
        repo.and_then(|repo| catalogs.lora_by_repo(repo))
            .map(|entry| (LoraMatchedBy::Repo, entry))
            .or_else(|| {
                let name = name?;
                match repo.filter(|_| repo_ambiguous) {
                    Some(repo) => catalogs.lora_by_name_in_repo(name, repo),
                    None => catalogs.lora_by_name(name),
                }
                .map(|entry| (LoraMatchedBy::Name, entry))
            })
    };
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
        // Named separately from a plain miss because the user's next move is different: they have
        // the weights, and no amount of downloading will tell this install WHICH of them the file
        // meant.
        RequirementState::Missing | RequirementState::UserSupplied if lora.hash.is_some() => {
            format!(
                "No installed LoRA has the exact adapter hash recorded for {identity}; the entry \
                 stays unresolved rather than substituting another file with the same name."
            )
        }
        RequirementState::Missing | RequirementState::UserSupplied if repo_ambiguous => format!(
            "More than one LoRA installed here comes from Hugging Face repo `{}`, and a shared \
             recipe does not record which adapter file was used — so applying one would be a \
             guess.",
            repo.unwrap_or_default()
        ),
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

    /// The reviewer's exact collision: row A's ID is row B's NAME. `Iterator::find` over
    /// `name || id` returned row A — the wrong adapter, reported as installed, with a one-click
    /// replay offered on top of it.
    fn colliding_loras() -> Vec<CatalogEntry> {
        vec![
            CatalogEntry::new("film_grain")
                .with_name("Vintage Filmstock")
                .installed(),
            CatalogEntry::new("vintage_1")
                .with_name("Film Grain")
                .installed(),
        ]
    }

    #[test]
    fn the_display_name_pass_runs_before_the_id_pass() {
        let catalogs = StaticCatalogs {
            loras: colliding_loras(),
            ..StaticCatalogs::default()
        };
        // `film_grain` is one row's ID and another row's NAME. The row actually CALLED "Film
        // Grain" wins, whatever order the catalog merged them in.
        let matched = catalogs
            .lora_by_name("Film Grain")
            .expect("one row matches");
        assert_eq!(matched.id, "vintage_1");
        assert_eq!(matched.name.as_deref(), Some("Film Grain"));
        // Reversing the catalog order must not change the answer — that is the whole bug.
        let reversed = StaticCatalogs {
            loras: colliding_loras().into_iter().rev().collect(),
            ..StaticCatalogs::default()
        };
        assert_eq!(
            reversed.lora_by_name("Film Grain").map(|entry| entry.id),
            Some("vintage_1".to_owned()),
            "the winner must not be an accident of catalog order"
        );
    }

    #[test]
    fn a_name_matched_by_two_rows_resolves_to_nothing() {
        let catalogs = StaticCatalogs {
            loras: vec![
                CatalogEntry::new("film_grain_a")
                    .with_name("Film Grain")
                    .installed(),
                CatalogEntry::new("film_grain_b")
                    .with_name("film-grain")
                    .installed(),
            ],
            ..StaticCatalogs::default()
        };
        assert!(
            catalogs.lora_by_name("Film Grain").is_none(),
            "two rows share the name, so neither may be presented as the answer"
        );
    }

    #[test]
    fn an_ambiguous_name_pass_does_not_fall_through_to_the_id_pass() {
        // Two rows are CALLED "Film Grain" and a third has it as an id. The weaker key cannot
        // break a tie the stronger one could not, so nothing resolves.
        let catalogs = StaticCatalogs {
            loras: vec![
                CatalogEntry::new("a").with_name("Film Grain").installed(),
                CatalogEntry::new("b").with_name("Film Grain").installed(),
                CatalogEntry::new("film_grain").installed(),
            ],
            ..StaticCatalogs::default()
        };
        assert!(catalogs.lora_by_name("film grain").is_none());
    }

    fn multi_adapter_repo() -> StaticCatalogs {
        StaticCatalogs {
            loras: vec![
                CatalogEntry::new("union")
                    .with_repo("acme/pack")
                    .with_name("Union")
                    .installed(),
                CatalogEntry::new("hdr")
                    .with_repo("Acme/Pack")
                    .with_name("HDR")
                    .installed(),
            ],
            ..StaticCatalogs::default()
        }
    }

    #[test]
    fn two_rows_off_one_repo_resolve_to_nothing() {
        let catalogs = multi_adapter_repo();
        assert!(
            catalogs.lora_by_repo("acme/pack").is_none(),
            "two adapter files from one repo are two adapters"
        );
        assert!(
            catalogs.lora_repo_ambiguous("acme/pack"),
            "and the caller must be able to tell that from a plain miss"
        );
        assert!(
            !catalogs.lora_repo_ambiguous("acme/nothing"),
            "a repo nobody claims is a miss, not an ambiguity"
        );
        assert!(!catalogs.lora_repo_ambiguous("   "));
    }

    #[test]
    fn an_ambiguous_repo_lets_the_name_break_the_tie_among_the_tied_rows() {
        // Both installs have the same multi-adapter pack — the ORDINARY way a repo becomes
        // ambiguous, and the one case where the two installs' display names agree. The repo
        // narrowed the answer to `union` and `hdr`; the name says which. Refusing here reported
        // "applying one would be a guess" about a recipe the envelope had named exactly.
        let catalogs = multi_adapter_repo();
        let requirement = classify_lora(
            &WorkflowLora {
                name: Some("Union".to_owned()),
                repo: Some("acme/pack".to_owned()),
                weight: Some(0.8),
                hash: None,
            },
            &catalogs,
        );
        assert_eq!(requirement.state, RequirementState::Resolved);
        assert_eq!(requirement.catalog_id.as_deref(), Some("union"));
        assert_eq!(requirement.matched_by, Some(LoraMatchedBy::Name));
    }

    #[test]
    fn an_ambiguous_repo_confines_the_name_pass_to_the_rows_claiming_it() {
        // The leak sc-15952 closed, stated the way it is actually true. The tie-break is scoped to
        // the repo's own rows: a name matching NONE of them resolves to nothing rather than
        // reaching across the catalog for a row from some other repo — that would be the weaker
        // key overruling the stronger one, which had already said the answer is in `acme/pack`.
        let mut catalogs = multi_adapter_repo();
        catalogs.loras.push(
            CatalogEntry::new("film_grain")
                .with_repo("other/pack")
                .with_name("Film Grain")
                .installed(),
        );
        let requirement = classify_lora(
            &WorkflowLora {
                name: Some("Film Grain".to_owned()),
                repo: Some("acme/pack".to_owned()),
                weight: Some(0.8),
                hash: None,
            },
            &catalogs,
        );
        assert_eq!(requirement.state, RequirementState::Missing);
        assert!(requirement.catalog_id.is_none());
        assert!(requirement.matched_by.is_none());
        assert!(
            requirement.detail.contains("acme/pack") && requirement.detail.contains("guess"),
            "the sentence must name the repo and why nothing was chosen: {}",
            requirement.detail
        );
    }

    #[test]
    fn an_ambiguous_repo_whose_tied_rows_share_a_name_resolves_to_nothing() {
        // The tie-break is a UNIQUE hit inside the tied set, not a first-match. Two rows off one
        // repo that are also CALLED the same thing leave nothing to choose with, and catalog order
        // must not become the answer.
        let catalogs = StaticCatalogs {
            loras: vec![
                CatalogEntry::new("union_a")
                    .with_repo("acme/pack")
                    .with_name("Union")
                    .installed(),
                CatalogEntry::new("union_b")
                    .with_repo("acme/pack")
                    .with_name("union")
                    .installed(),
            ],
            ..StaticCatalogs::default()
        };
        let requirement = classify_lora(
            &WorkflowLora {
                name: Some("Union".to_owned()),
                repo: Some("acme/pack".to_owned()),
                weight: None,
                hash: None,
            },
            &catalogs,
        );
        assert_eq!(requirement.state, RequirementState::Missing);
        assert!(requirement.matched_by.is_none());
    }

    #[test]
    fn an_ambiguous_repo_with_no_name_at_all_resolves_to_nothing() {
        // Nothing to break the tie with. The repo pass declined and there is no second key.
        let catalogs = multi_adapter_repo();
        let requirement = classify_lora(
            &WorkflowLora {
                name: None,
                repo: Some("acme/pack".to_owned()),
                weight: None,
                hash: None,
            },
            &catalogs,
        );
        assert_eq!(requirement.state, RequirementState::Missing);
        assert!(requirement.matched_by.is_none());
    }

    #[test]
    fn the_restricted_name_pass_folds_labels_the_way_the_open_one_does() {
        // Both name methods run one body, so the tied-rows pass must normalize identically — and
        // must reach the ID pass for a tied row that carries no display name.
        let catalogs = multi_adapter_repo();
        assert_eq!(
            catalogs
                .lora_by_name_in_repo("  union  ", "ACME/Pack")
                .map(|entry| entry.id),
            Some("union".to_owned()),
            "case and surrounding space fold on both the name and the repo"
        );
        assert!(catalogs.lora_by_name_in_repo("Union", "   ").is_none());
        assert!(catalogs.lora_by_name_in_repo("   ", "acme/pack").is_none());
        let id_only = StaticCatalogs {
            loras: vec![
                CatalogEntry::new("film_grain")
                    .with_repo("acme/pack")
                    .installed(),
                CatalogEntry::new("other")
                    .with_repo("acme/pack")
                    .with_name("Other")
                    .installed(),
            ],
            ..StaticCatalogs::default()
        };
        assert_eq!(
            id_only
                .lora_by_name_in_repo("Film Grain", "acme/pack")
                .map(|entry| entry.id),
            Some("film_grain".to_owned()),
            "the id pass runs inside the restriction too, for a row with no display name"
        );
    }

    #[test]
    fn an_unambiguous_repo_still_falls_through_to_the_name_pass() {
        // The guard is scoped to the ambiguous case: a repo NOTHING here carries must still let a
        // display-name match resolve, which is how a locally-imported copy of a shared LoRA is
        // found at all.
        let catalogs = StaticCatalogs {
            loras: vec![CatalogEntry::new("film_grain")
                .with_name("Film Grain")
                .installed()],
            ..StaticCatalogs::default()
        };
        let requirement = classify_lora(
            &WorkflowLora {
                name: Some("Film Grain".to_owned()),
                repo: Some("acme/unknown-pack".to_owned()),
                weight: None,
                hash: None,
            },
            &catalogs,
        );
        assert_eq!(requirement.state, RequirementState::Resolved);
        assert_eq!(requirement.matched_by, Some(LoraMatchedBy::Name));
        assert_eq!(requirement.catalog_id.as_deref(), Some("film_grain"));
    }

    #[test]
    fn a_blank_id_matches_nothing_even_when_a_catalog_row_carries_one() {
        // The scrubbed-`model` case: an id-less catalog row used to read as `id: ""` and match an
        // envelope that named no model at all.
        let entries = vec![CatalogEntry::new("")
            .with_name("Some Other Model")
            .installed()];
        assert!(find_by_id(&entries, "").is_none());
        assert!(find_by_id(&entries, "   ").is_none());
    }

    #[test]
    fn a_blank_model_slug_never_consults_the_catalog() {
        struct Exploding;
        impl WorkflowCatalogs for Exploding {
            fn model(&self, _: &str) -> Option<CatalogEntry> {
                panic!("a blank slug must not reach the catalog");
            }
            fn lora_by_repo(&self, _: &str) -> Option<CatalogEntry> {
                None
            }
            fn lora_repo_ambiguous(&self, _: &str) -> bool {
                false
            }
            fn lora_by_name(&self, _: &str) -> Option<CatalogEntry> {
                None
            }
            fn lora_by_name_in_repo(&self, _: &str, _: &str) -> Option<CatalogEntry> {
                None
            }
            fn style(&self, _: &str) -> Option<CatalogEntry> {
                None
            }
            fn recipe_preset(&self, _: &str) -> Option<CatalogEntry> {
                None
            }
        }
        let requirement = classify_model("", &Exploding);
        assert_eq!(requirement.state, RequirementState::Missing);
        assert!(requirement.catalog_id.is_none());
        assert!(requirement.name.is_none());
    }

    #[test]
    fn article_reads_correctly_for_a_vowel_initial_control_mode() {
        assert_eq!(article("source image"), "a");
        assert_eq!(article("edge control image"), "an");
        assert_eq!(article("Openpose control image"), "an");
    }
}
