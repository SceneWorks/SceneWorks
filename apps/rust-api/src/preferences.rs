//! UI preferences (theme, accent, …) persisted as a small JSON file in the
//! config dir.
//! Served over plain HTTP because the bundled desktop UI runs at the API's
//! `http://127.0.0.1:<port>` origin, where both Tauri IPC and origin-keyed
//! `localStorage` are unreliable across launches (the port — and so the origin —
//! changes every launch). Routing through the API, the same channel the rest of
//! the app already uses, makes the choice durable. Non-sensitive, so the routes
//! are public like `/health`.

use super::*;

use std::collections::BTreeMap;
use std::path::PathBuf;

const PREFERENCES_FILENAME: &str = "ui-preferences.json";

#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct UiPreferences {
    /// Last-used UI theme (`"light"` or `"dark"`); absent until first set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    theme: Option<String>,
    /// Last-used accent palette id (see `ACCENT_IDS`); absent until first set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    accent: Option<String>,
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

fn preferences_path(state: &AppState) -> PathBuf {
    state.settings.config_dir.join(PREFERENCES_FILENAME)
}

fn load_preferences(path: &std::path::Path) -> UiPreferences {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|body| serde_json::from_str(&body).ok())
        .unwrap_or_default()
}

fn save_preferences(path: &std::path::Path, prefs: &UiPreferences) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let body = serde_json::to_string_pretty(prefs)?;
    std::fs::write(path, body)
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
    let prefs = tokio::task::spawn_blocking(move || {
        let mut prefs = load_preferences(&path);
        // One-time migration (epic 10721 R3): flip a stored `q8` (the old forced default, predating the
        // `auto` option) to `auto` and persist — see `migrate_quality_to_auto`. Only writes on a change.
        if migrate_quality_to_auto(&mut prefs) {
            let _ = save_preferences(&path, &prefs);
        }
        // Always hand the web a concrete value to seed (`auto` when unset/invalid), so the durable
        // value survives a desktop relaunch even before the user changes it.
        prefs.default_generation_quality = Some(normalize_generation_quality(
            prefs.default_generation_quality.as_deref(),
        ));
        prefs
    })
    .await
    .map_err(|err| ApiError::internal(format!("UI preferences load task failed: {err}")))?;
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
    let prefs = tokio::task::spawn_blocking(move || -> std::io::Result<UiPreferences> {
        let mut prefs = load_preferences(&path);
        if let Some(theme) = normalize_theme(payload.theme.as_deref()) {
            prefs.theme = Some(theme);
        }
        if let Some(accent) = normalize_accent(payload.accent.as_deref()) {
            prefs.accent = Some(accent);
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
        save_preferences(&path, &prefs)?;
        Ok(prefs)
    })
    .await
    .map_err(|err| ApiError::internal(format!("UI preferences save task failed: {err}")))?
    .map_err(|error| ApiError::internal(format!("Failed to save UI preferences: {error}")))?;
    Ok(Json(prefs))
}

#[cfg(test)]
mod tests {
    use super::{
        migrate_quality_to_auto, normalize_accent, normalize_generation_quality, normalize_theme,
        UiPreferences,
    };

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
