//! UI preferences (theme, accent, …) persisted as a small JSON file in the
//! config dir.
//! Served over plain HTTP because the bundled desktop UI runs at the API's
//! `http://127.0.0.1:<port>` origin, where both Tauri IPC and origin-keyed
//! `localStorage` are unreliable across launches (the port — and so the origin —
//! changes every launch). Routing through the API, the same channel the rest of
//! the app already uses, makes the choice durable.
//!
//! AUTH IS METHOD-ASYMMETRIC here — the routes are NOT simply public (sc-8869, F-067; see
//! `auth::requires_token`). The GET is public because the theme has to paint before the
//! access gate to avoid a flash, and the value is non-sensitive. The PUT writes this file to
//! DISK, so it is gated like every other write (epic 4484): an unauthenticated LAN caller
//! must not be able to overwrite it. A client that sends the PUT without the access token
//! gets a 401 on every remote-auth deployment — exactly the bug sc-15136 fixed, after this
//! doc's older "the routes are public" wording had been echoed into six web call sites.

use super::*;

use std::collections::BTreeMap;
use std::path::PathBuf;

// The filename lives in `sceneworks_core::app_paths::ui_preferences_file` — see
// `preferences_path`. A second literal here is what made the cross-crate name drift possible.

#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct UiPreferences {
    /// Last-used UI theme (`"light"` or `"dark"`); absent until first set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    theme: Option<String>,
    /// Last-used accent palette id (see `ACCENT_IDS`); absent until first set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    accent: Option<String>,
    /// "Use Simple UI by default" (the Simple UI design handoff): whether the reduced-surface
    /// Simple shell is the one the app opens on. Absent until first set, which the web reads as
    /// OFF — an existing install keeps the full workspace it already has. Made durable through
    /// this API for the same reason as theme/accent: the desktop shell's per-launch
    /// `127.0.0.1:<port>` origin changes every launch, wiping origin-keyed `localStorage`.
    /// A phone viewport forces the Simple shell regardless of this value; that rule is
    /// client-side (it depends on the viewport, not on stored state).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    simple_ui: Option<bool>,
    /// App-wide default generation quality (`auto`|`bf16`|`q8`|`q4`); the baseline the
    /// per-model default tier derives from when no per-(screen,model) sticky pick exists
    /// (sc-10728). `auto` (epic 10721 R3) is the default: each model defaults to the
    /// highest-fidelity tier that fits this machine's memory. Absent in the stored file
    /// until first set — but the GET response always resolves an absent/invalid value to
    /// `auto`, so the web always has a value to seed. Made durable through this API (not
    /// just origin-keyed `localStorage`) for the same reason as theme/accent: the desktop
    /// shell's `127.0.0.1:<port>` origin changes every launch, wiping `localStorage`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    default_generation_quality: Option<String>,
    /// One-time migration marker (epic 10721 R3). The quality setting predates the `auto`
    /// option, so every stored `q8` is the OLD forced default that rode along in unrelated
    /// PUTs, not a choice AGAINST Auto (which didn't exist). On GET we flip a stored `q8`
    /// to `auto` exactly once and set this true; a DELIBERATE quality PUT also sets it — so
    /// a q8 the user picks AFTER the upgrade sticks and is never re-migrated.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    migrated_quality_auto: bool,
    /// Per-(screen, modelId) last-picked quant tier — the DURABLE form of the studio tier sticky
    /// (epic 10721 R1). The desktop shell's per-launch `127.0.0.1:<port>` origin wipes localStorage, so
    /// a tier a user picks for a model in a studio would be lost on relaunch and they'd have to re-pick
    /// every session. Persisting it here (re-seeded into localStorage on launch) makes "your per-model
    /// picks are remembered" actually hold across restarts. Shape: `{ [screen]: { [modelId]: tier } }`;
    /// the frontend owns the tier vocabulary, so values are stored verbatim. A PUT carrying this field
    /// replaces the whole map — the web sends the full merged map it holds in its localStorage cache, so
    /// no server-side deep-merge is needed (and a PUT without the field leaves it untouched).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    per_model_tier: Option<BTreeMap<String, BTreeMap<String, String>>>,
    /// Last top-level advanced-shell destination. This deliberately excludes project
    /// state; it lets global workspaces such as Dataset Catalogs reopen after the
    /// desktop API chooses a new origin on restart.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    active_view: Option<String>,
    /// Last selected attached Dataset Catalog. The web reconciles this id against the
    /// registry on load, so a detached or relocated catalog degrades to another item.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    selected_catalog_id: Option<String>,
    /// Simple-UI studio settings — the DURABLE form of the Simple studios' own knobs (model,
    /// prompt, resolution, LoRA picks, …). The Simple shell renders one screen at a time, so a
    /// studio unmounts on navigation; the shell keeps a session copy, and this is what makes
    /// that copy outlive a relaunch. The advanced studios solve the same problem with a
    /// per-workspace `localStorage` blob (`hooks/useStudioSettings.js`), which is exactly what
    /// the desktop shell's per-launch `127.0.0.1:<port>` origin wipes — hence the server copy
    /// here rather than a second localStorage store.
    ///
    /// Shape: `{ [workspaceId]: { [studio]: { [field]: value } } }`. Keyed by WORKSPACE to match
    /// the advanced studios: a prompt written for one project must not follow the user into
    /// another. The field vocabulary is the frontend's and values are stored verbatim — this is
    /// the same stance as `per_model_tier`, so adding a Simple knob needs no server change.
    /// A PUT carrying this field replaces the whole map (the web sends the full merged map from
    /// its cache), and a PUT without it leaves the stored map untouched.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    simple_studio: Option<BTreeMap<String, serde_json::Value>>,
    /// Advanced-workspace studio settings — the same thing as `simple_studio`, for the full
    /// workspace's Image / Video / Audio / Character / Editor studios (sc-15425).
    ///
    /// The advanced studios keep their settings in a per-workspace `localStorage` blob
    /// (`hooks/useStudioSettings.js`) and survive NAVIGATION by never unmounting (`KEEP_ALIVE_VIEWS`),
    /// so until now they lost everything on relaunch — the desktop shell's `127.0.0.1:<port>` origin
    /// changes each launch and takes origin-keyed storage with it. This is the durable copy that
    /// `localStorage` cache is re-seeded from on launch.
    ///
    /// Shape: `{ [workspaceId]: { [studio]: { [field]: value } } }`, studios being
    /// `image`/`video`/`audio`/`character`/`editor-video`. Stored verbatim like the fields above.
    /// The web deliberately omits its two unbounded batch fields from what it sends here; see
    /// `MAX_STUDIO_ENTRY_BYTES`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    advanced_studio: Option<BTreeMap<String, serde_json::Value>>,
    /// Whether a generated PNG carries the sanitized workflow envelope so a shared image reloads
    /// its recipe on drop (epic 15945, sc-15948). Absent until first set, which every reader takes
    /// as ON — the feature is pointless off by default, and a user who does not want it should
    /// have to find the switch once rather than opt in forever.
    ///
    /// This is the ONE preference read outside the web app: the WORKER reads it straight off this
    /// file at its PNG write seam, through `sceneworks_core::app_paths::embed_workflow_in_images`,
    /// because the desktop hands every process it spawns the same `SCENEWORKS_CONFIG_DIR`. Renaming
    /// the field or moving the file therefore breaks a consumer the route cannot see.
    /// sc-15953 owns the Settings toggle and the first-run disclosure that write it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    embed_workflow_in_images: Option<bool>,
    /// Whether the one-time embedded-workflow disclosure has been shown and dismissed (sc-15953).
    ///
    /// Durable rather than a browser flag because the desktop shell serves the UI from a
    /// `127.0.0.1:<port>` origin that changes each launch, taking origin-keyed storage with it — a
    /// `localStorage`-only "already told them" flag would re-show the notice on every launch.
    ///
    /// Absent means NOT shown, and that is the state an install upgrading into this build is in:
    /// embedding defaults ON, so someone who has been generating for months has never been told,
    /// and the notice fires once for them rather than only for fresh installs. Nothing ever writes
    /// `false` back — this only goes from absent to `true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    workflow_embed_notice_seen: Option<bool>,
}

/// Retained workspaces in a studio-settings map. The map grows by one entry per workspace the
/// user opens a studio in and nothing ever removes an entry, so without a bound a long-lived
/// install would carry every workspace it has ever seen — each holding a full prompt — in a file
/// re-read and rewritten on every preference write. The web prunes to the same bound on its side;
/// this is the backstop for a client that doesn't.
const MAX_STUDIO_WORKSPACES: usize = 24;

/// Serialized bytes allowed for one workspace's studio snapshot. Prompts are free text, so this
/// bounds what a single entry can cost; an oversized entry is DROPPED rather than truncated,
/// because a half-parsed snapshot would restore a studio into a state the user never chose.
///
/// The ADVANCED entry is the sizing case, not the Simple one: a Simple workspace holds ~8 fields
/// per studio, while `ImageStudio` and `VideoStudio` each persist ~47. The web keeps its two
/// unbounded batch fields (`batchPromptsText`, `batchVariableValues`) out of the durable copy for
/// exactly this reason, which is what keeps a realistic advanced snapshot comfortably inside this.
const MAX_STUDIO_ENTRY_BYTES: usize = 128 * 1024;

/// Keep the well-formed, in-budget entries of a studio-settings map (`simple_studio` /
/// `advanced_studio`).
///
/// Entries must be JSON objects (the `{ [studio]: { … } }` shape the web writes); anything else is
/// a client bug or a hand-edited file, and storing it would hand the web back something it can't
/// read. Over-budget entries are dropped. When more than `MAX_STUDIO_WORKSPACES` survive, the
/// retained set is the first N by key order — arbitrary but STABLE, so repeated PUTs of the same
/// oversized map converge instead of thrashing the file.
fn normalize_studio_settings(
    input: BTreeMap<String, serde_json::Value>,
) -> BTreeMap<String, serde_json::Value> {
    input
        .into_iter()
        .filter(|(workspace, snapshot)| {
            !workspace.trim().is_empty()
                && snapshot.is_object()
                && serde_json::to_string(snapshot)
                    .map(|body| body.len() <= MAX_STUDIO_ENTRY_BYTES)
                    .unwrap_or(false)
        })
        .take(MAX_STUDIO_WORKSPACES)
        .collect()
}

/// User-selectable accent palettes. Keep in sync with web/src/accents.js.
const ACCENT_IDS: [&str; 7] = [
    "teal", "indigo", "cobalt", "violet", "coral", "amber", "emerald",
];

/// User-facing generation-quality tiers. Keep in sync with
/// web/src/quantTier.js `GENERATION_QUALITY_TIERS`.
const GENERATION_QUALITY_IDS: [&str; 3] = ["bf16", "q8", "q4"];

/// The capability-aware "Auto" mode (epic 10721 R3). Keep in sync with
/// web/src/generationQuality.js `AUTO_GENERATION_QUALITY`.
const AUTO_GENERATION_QUALITY: &str = "auto";

/// The app-wide default generation quality when unset or invalid: `auto` (epic 10721 R3),
/// so an untouched install gets the capability-aware default, not a flat q8. Keep in sync
/// with the web `AUTO_GENERATION_QUALITY` default in `normalizeGenerationQuality`.
const DEFAULT_GENERATION_QUALITY: &str = AUTO_GENERATION_QUALITY;

/// The stored preference file.
///
/// Goes through `sceneworks_core::app_paths` rather than joining a local literal, because this file
/// has a reader in ANOTHER crate: the worker reads `embedWorkflowInImages` off it at its PNG write
/// seam (sc-15948). Two literals for one cross-crate file is a silent-failure shape — a drift in
/// either would leave the worker reading a path that does not exist, which its reader resolves to
/// "embedding ON" with no error and no log, forever.
fn preferences_path(state: &AppState) -> PathBuf {
    sceneworks_core::app_paths::ui_preferences_file(&state.settings.config_dir)
}

fn load_preferences(path: &std::path::Path) -> UiPreferences {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|body| serde_json::from_str(&body).ok())
        .unwrap_or_default()
}

/// The valid stored theme for `input`, or `None` if it isn't one we recognize.
fn normalize_theme(input: Option<&str>) -> Option<String> {
    match input.map(str::trim) {
        Some("light") => Some("light".to_owned()),
        Some("dark") => Some("dark".to_owned()),
        _ => None,
    }
}

/// The valid stored accent for `input`, or `None` if it isn't a known palette.
fn normalize_accent(input: Option<&str>) -> Option<String> {
    let value = input.map(str::trim)?;
    ACCENT_IDS
        .iter()
        .find(|id| **id == value)
        .map(|id| (*id).to_owned())
}

fn normalize_preference_id(input: Option<&str>, max_len: usize) -> Option<String> {
    let value = input.map(str::trim)?;
    (!value.is_empty()
        && value.len() <= max_len
        && value
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || "_-".contains(character)))
    .then(|| value.to_owned())
}

/// The valid stored generation quality for `input`, falling back to the q8 default when
/// it's absent or not one of `bf16`|`q8`|`q4`. Unlike theme/accent (which return `None`
/// for an unknown value so a partial PUT leaves the stored value untouched), the quality
/// setting has a defined app-wide default, so this resolves to it. The PUT merge only
/// invokes this when the payload actually carries the field, so a theme-only PUT never
/// clobbers a previously-set value.
fn normalize_generation_quality(input: Option<&str>) -> String {
    match input.map(str::trim) {
        Some(value)
            if value == AUTO_GENERATION_QUALITY || GENERATION_QUALITY_IDS.contains(&value) =>
        {
            value.to_owned()
        }
        _ => DEFAULT_GENERATION_QUALITY.to_owned(),
    }
}

/// Apply the one-time `q8` → `auto` migration to `prefs` in place (epic 10721 R3). Returns `true` when
/// it changed something, so the caller persists. Idempotent: guarded by `migrated_quality_auto`, so a
/// deliberate q8 pick (which sets that marker via PUT) is never re-migrated.
fn migrate_quality_to_auto(prefs: &mut UiPreferences) -> bool {
    if !prefs.migrated_quality_auto && prefs.default_generation_quality.as_deref() == Some("q8") {
        prefs.default_generation_quality = Some(AUTO_GENERATION_QUALITY.to_owned());
        prefs.migrated_quality_auto = true;
        return true;
    }
    false
}

/// Current UI preferences (empty object on first run).
pub(crate) async fn get_ui_preferences(
    State(state): State<AppState>,
) -> Result<Json<UiPreferences>, ApiError> {
    // sc-4202 (F-API-3): keep the preferences-file read off the async executor.
    let path = preferences_path(&state);
    let lock = manifest_write_lock(&state, &path);
    let _guard = lock.lock().await;
    let load_path = path.clone();
    let (mut prefs, migrated_body) = tokio::task::spawn_blocking(move || {
        let mut prefs = load_preferences(&load_path);
        // One-time migration (epic 10721 R3): flip a stored `q8` (the old forced default, predating the
        // `auto` option) to `auto` and persist — see `migrate_quality_to_auto`. Only writes on a change.
        if migrate_quality_to_auto(&mut prefs) {
            let body = serde_json::to_string_pretty(&prefs).ok();
            return (prefs, body);
        }
        (prefs, None)
    })
    .await
    .map_err(|err| ApiError::internal(format!("UI preferences load task failed: {err}")))?;
    if let Some(body) = migrated_body {
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|error| {
                ApiError::internal(format!("Failed to create preferences directory: {error}"))
            })?;
        }
        write_manifest_atomic(&path, &body).await?;
    }
    // Always hand the web a concrete value to seed (`auto` when unset/invalid), so the durable
    // value survives a desktop relaunch even before the user changes it.
    prefs.default_generation_quality = Some(normalize_generation_quality(
        prefs.default_generation_quality.as_deref(),
    ));
    Ok(Json(prefs))
}

/// Merge the supplied preferences in and persist. Only recognized fields/values
/// are applied, so an unknown theme leaves the stored one untouched.
pub(crate) async fn set_ui_preferences(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<UiPreferences>,
) -> Result<Json<UiPreferences>, ApiError> {
    // sc-4202 (F-API-3): the read-modify-write of the preferences file runs on the
    // blocking pool so it can't stall a tokio worker thread.
    let path = preferences_path(&state);
    let lock = manifest_write_lock(&state, &path);
    let _guard = lock.lock().await;
    let load_path = path.clone();
    let (prefs, body) = tokio::task::spawn_blocking(move || -> Result<_, serde_json::Error> {
        let mut prefs = load_preferences(&load_path);
        if let Some(theme) = normalize_theme(payload.theme.as_deref()) {
            prefs.theme = Some(theme);
        }
        if let Some(accent) = normalize_accent(payload.accent.as_deref()) {
            prefs.accent = Some(accent);
        }
        // Only touch the Simple-UI default when the payload actually carries it, so a
        // theme/accent/quality-only PUT can't reset it (same partial-merge rule as the fields
        // above). It is a plain bool, so there is nothing to normalize — an out-of-vocabulary
        // value can't parse in the first place.
        if let Some(simple_ui) = payload.simple_ui {
            prefs.simple_ui = Some(simple_ui);
        }
        // Only touch the quality when the payload actually carries it, so a theme- or accent-only PUT
        // can't reset it. A present-but-invalid value coerces to `auto`. A deliberate quality PUT also
        // sets the migration marker, so a q8 the user picks AFTER the upgrade is never auto-migrated
        // (a theme-only PUT deliberately does NOT set it, so an existing q8 still migrates on GET).
        if payload.default_generation_quality.is_some() {
            prefs.default_generation_quality = Some(normalize_generation_quality(
                payload.default_generation_quality.as_deref(),
            ));
            prefs.migrated_quality_auto = true;
        }
        // The per-model tier map is replaced wholesale ONLY when the payload carries it (the web sends
        // the full merged map from its cache), so a theme/quality-only PUT never clobbers a stored map.
        if let Some(per_model_tier) = payload.per_model_tier {
            prefs.per_model_tier = Some(per_model_tier);
        }
        // Replaced wholesale ONLY when the payload carries it, like `per_model_tier` above — the
        // web sends the full merged map from its cache, so a theme- or view-only PUT never
        // clobbers a stored snapshot.
        if let Some(simple_studio) = payload.simple_studio {
            prefs.simple_studio = Some(normalize_studio_settings(simple_studio));
        }
        if let Some(advanced_studio) = payload.advanced_studio {
            prefs.advanced_studio = Some(normalize_studio_settings(advanced_studio));
        }
        // Same partial-merge rule as `simple_ui`: only touch it when the payload carries it, so an
        // unrelated PUT can neither turn embedding off nor turn it back on. A plain bool needs no
        // normalization — an out-of-vocabulary value cannot parse.
        if let Some(embed_workflow_in_images) = payload.embed_workflow_in_images {
            prefs.embed_workflow_in_images = Some(embed_workflow_in_images);
        }
        // One-way (sc-15953): once the disclosure has been shown it stays shown. A payload
        // carrying `false` is ignored rather than obeyed, so no unrelated client — or a stale
        // cached copy of an older preferences object PUT back wholesale — can make a user who has
        // already been told see the notice again.
        if payload.workflow_embed_notice_seen == Some(true) {
            prefs.workflow_embed_notice_seen = Some(true);
        }
        if let Some(active_view) = normalize_preference_id(payload.active_view.as_deref(), 48) {
            prefs.active_view = Some(active_view);
        }
        if let Some(selected_catalog_id) = payload.selected_catalog_id {
            if selected_catalog_id.trim().is_empty() {
                prefs.selected_catalog_id = None;
            } else if let Some(selected_catalog_id) =
                normalize_preference_id(Some(&selected_catalog_id), 128)
            {
                prefs.selected_catalog_id = Some(selected_catalog_id);
            }
        }
        let body = serde_json::to_string_pretty(&prefs)?;
        Ok((prefs, body))
    })
    .await
    .map_err(|err| ApiError::internal(format!("UI preferences save task failed: {err}")))?
    .map_err(|error| ApiError::internal(format!("Failed to encode UI preferences: {error}")))?;
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await.map_err(|error| {
            ApiError::internal(format!("Failed to create preferences directory: {error}"))
        })?;
    }
    write_manifest_atomic(&path, &body).await?;
    Ok(Json(prefs))
}

#[cfg(test)]
mod tests {
    use super::{
        migrate_quality_to_auto, normalize_accent, normalize_generation_quality,
        normalize_studio_settings, normalize_theme, UiPreferences, MAX_STUDIO_ENTRY_BYTES,
        MAX_STUDIO_WORKSPACES,
    };
    use std::collections::BTreeMap;

    fn snapshot(prompt: &str) -> serde_json::Value {
        serde_json::json!({ "image": { "prompt": prompt, "model": "z_image" } })
    }

    #[test]
    fn normalize_studio_settings_keeps_well_formed_entries_verbatim() {
        let mut input = BTreeMap::new();
        input.insert("project-1".to_owned(), snapshot("a lighthouse"));
        let kept = normalize_studio_settings(input);
        // Values are stored verbatim — the field vocabulary belongs to the frontend, so a new
        // Simple knob must not need a server change to survive a round trip.
        assert_eq!(kept.get("project-1"), Some(&snapshot("a lighthouse")));
    }

    #[test]
    fn normalize_studio_settings_drops_malformed_and_oversized_entries() {
        let mut input = BTreeMap::new();
        input.insert("ok".to_owned(), snapshot("fine"));
        // Not an object: a client bug or a hand-edited file. Storing it would hand the web back
        // something it cannot read.
        input.insert("scalar".to_owned(), serde_json::json!("not-an-object"));
        input.insert("null".to_owned(), serde_json::Value::Null);
        // Blank workspace key — nothing to key a restore on.
        input.insert("   ".to_owned(), snapshot("blank key"));
        // Over the per-entry budget: dropped whole, never truncated.
        input.insert(
            "huge".to_owned(),
            snapshot(&"x".repeat(MAX_STUDIO_ENTRY_BYTES + 1)),
        );

        let kept = normalize_studio_settings(input);
        assert_eq!(kept.keys().collect::<Vec<_>>(), vec!["ok"]);
    }

    #[test]
    fn normalize_studio_settings_bounds_the_workspace_count_stably() {
        let over = MAX_STUDIO_WORKSPACES + 5;
        let build = || {
            (0..over)
                .map(|n| (format!("project-{n:03}"), snapshot("p")))
                .collect::<BTreeMap<_, _>>()
        };
        let kept = normalize_studio_settings(build());
        assert_eq!(kept.len(), MAX_STUDIO_WORKSPACES);
        // Stable across repeated PUTs of the same map, so the file converges rather than
        // thrashing a different retained set on every write.
        assert_eq!(kept, normalize_studio_settings(build()));
    }

    #[test]
    fn simple_studio_round_trips_through_serde_as_camel_case() {
        let mut map = BTreeMap::new();
        map.insert("project-1".to_owned(), snapshot("round trip"));
        let prefs = UiPreferences {
            simple_studio: Some(map),
            ..Default::default()
        };
        let body = serde_json::to_string(&prefs).expect("serialize");
        assert!(body.contains("\"simpleStudio\""));
        let parsed: UiPreferences = serde_json::from_str(&body).expect("deserialize");
        assert_eq!(
            parsed
                .simple_studio
                .and_then(|m| m.get("project-1").cloned()),
            Some(snapshot("round trip"))
        );
    }

    #[test]
    fn an_absent_simple_studio_is_omitted_from_the_stored_file() {
        let body = serde_json::to_string(&UiPreferences::default()).expect("serialize");
        assert!(!body.contains("simpleStudio"));
        assert!(!body.contains("advancedStudio"));
    }

    /// The two studio maps are INDEPENDENT fields sharing one normalizer (sc-15425). A PUT
    /// carrying one must not disturb the other — the Simple shell and the advanced workspace
    /// write on their own schedules and would otherwise erase each other.
    #[test]
    fn simple_and_advanced_studio_are_independent_fields() {
        let mut simple = BTreeMap::new();
        simple.insert("project-1".to_owned(), snapshot("simple side"));
        let mut advanced = BTreeMap::new();
        advanced.insert("project-1".to_owned(), snapshot("advanced side"));
        let prefs = UiPreferences {
            simple_studio: Some(simple),
            advanced_studio: Some(advanced),
            ..Default::default()
        };
        let body = serde_json::to_string(&prefs).expect("serialize");
        let parsed: UiPreferences = serde_json::from_str(&body).expect("deserialize");
        assert_eq!(
            parsed.simple_studio.unwrap()["project-1"],
            snapshot("simple side")
        );
        assert_eq!(
            parsed.advanced_studio.unwrap()["project-1"],
            snapshot("advanced side")
        );
    }

    /// The workflow-embedding toggle has a reader outside this crate: the worker reads it off the
    /// stored file by name at its PNG write seam (sc-15948). So the wire name and the
    /// omitted-when-absent behaviour are contract, not detail — an absent field is what the worker
    /// resolves to ON, and a stored `false` is the only thing that turns embedding off.
    #[test]
    fn embed_workflow_in_images_round_trips_under_the_name_the_worker_reads() {
        let body = serde_json::to_string(&UiPreferences::default()).expect("serialize");
        assert!(
            !body.contains("embedWorkflowInImages"),
            "an untouched install must store nothing, so every reader falls back to ON: {body}"
        );

        let prefs = UiPreferences {
            embed_workflow_in_images: Some(false),
            ..Default::default()
        };
        let body = serde_json::to_string(&prefs).expect("serialize");
        assert!(body.contains("\"embedWorkflowInImages\":false"), "{body}");
        // The exact shape `sceneworks_core::app_paths::embed_workflow_in_images` parses.
        assert!(
            !sceneworks_core::app_paths::embed_workflow_in_images_from_json(&body),
            "the worker's reader must see the stored opt-out"
        );

        let parsed: UiPreferences = serde_json::from_str(&body).expect("deserialize");
        assert_eq!(parsed.embed_workflow_in_images, Some(false));

        // The FILE, not only the field. This test used to pin the wire name while the two crates
        // named the file with two separate literals, so a drift in either would have left the
        // worker opening a path that does not exist — which its reader resolves to "embedding ON",
        // with no error and no log. `preferences_path` now builds the path from this same helper,
        // and this pins the name the two crates share.
        let config = std::path::Path::new("config-dir");
        assert_eq!(
            sceneworks_core::app_paths::ui_preferences_file(config),
            config.join("ui-preferences.json"),
            "the worker finds this preference by opening this exact filename"
        );
    }

    #[test]
    fn normalize_theme_accepts_only_known_themes() {
        assert_eq!(normalize_theme(Some(" light ")), Some("light".to_owned()));
        assert_eq!(normalize_theme(Some("dark")), Some("dark".to_owned()));
        assert_eq!(normalize_theme(Some("blue")), None);
        assert_eq!(normalize_theme(None), None);
    }

    #[test]
    fn normalize_accent_accepts_only_known_palettes() {
        assert_eq!(normalize_accent(Some("teal")), Some("teal".to_owned()));
        assert_eq!(
            normalize_accent(Some(" emerald ")),
            Some("emerald".to_owned())
        );
        assert_eq!(normalize_accent(Some("amber")), Some("amber".to_owned()));
        assert_eq!(normalize_accent(Some("fuchsia")), None);
        assert_eq!(normalize_accent(None), None);
    }

    #[test]
    fn normalize_generation_quality_accepts_auto_and_tiers_and_defaults_to_auto() {
        assert_eq!(normalize_generation_quality(Some(" auto ")), "auto");
        assert_eq!(normalize_generation_quality(Some(" bf16 ")), "bf16");
        assert_eq!(normalize_generation_quality(Some("q8")), "q8");
        assert_eq!(normalize_generation_quality(Some("q4")), "q4");
        // Unknown/candle-only tiers and absent values fall back to the Auto default (epic 10721 R3).
        assert_eq!(normalize_generation_quality(Some("int8-convrot")), "auto");
        assert_eq!(normalize_generation_quality(Some("")), "auto");
        assert_eq!(normalize_generation_quality(None), "auto");
    }

    #[test]
    fn migrate_quality_to_auto_flips_a_stored_q8_exactly_once() {
        // A stored q8 (the old forced default) migrates to auto and sets the marker.
        let mut prefs = UiPreferences {
            default_generation_quality: Some("q8".to_owned()),
            ..Default::default()
        };
        assert!(migrate_quality_to_auto(&mut prefs));
        assert_eq!(prefs.default_generation_quality.as_deref(), Some("auto"));
        assert!(prefs.migrated_quality_auto);
        // Idempotent: a second pass does nothing.
        assert!(!migrate_quality_to_auto(&mut prefs));

        // A DELIBERATE q8 (marker already set, e.g. picked via PUT after the upgrade) is left alone.
        let mut deliberate = UiPreferences {
            default_generation_quality: Some("q8".to_owned()),
            migrated_quality_auto: true,
            ..Default::default()
        };
        assert!(!migrate_quality_to_auto(&mut deliberate));
        assert_eq!(deliberate.default_generation_quality.as_deref(), Some("q8"));

        // Non-q8 values (a deliberate bf16, or an unset install) never migrate.
        let mut bf16 = UiPreferences {
            default_generation_quality: Some("bf16".to_owned()),
            ..Default::default()
        };
        assert!(!migrate_quality_to_auto(&mut bf16));
        let mut unset = UiPreferences::default();
        assert!(!migrate_quality_to_auto(&mut unset));
    }
}
