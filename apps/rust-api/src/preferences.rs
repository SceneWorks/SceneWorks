//! UI preferences (theme, accent, …) persisted as a small JSON file in the
//! config dir.
//! Served over plain HTTP because the bundled desktop UI runs at the API's
//! `http://127.0.0.1:<port>` origin, where both Tauri IPC and origin-keyed
//! `localStorage` are unreliable across launches (the port — and so the origin —
//! changes every launch). Routing through the API, the same channel the rest of
//! the app already uses, makes the choice durable. Non-sensitive, so the routes
//! are public like `/health`.

use super::*;

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
    /// App-wide default generation quality (`bf16`|`q8`|`q4`); the baseline quant tier
    /// new generations use when no per-(screen,model) sticky pick exists (sc-10728).
    /// Absent in the stored file until first set — but the GET response always resolves
    /// an absent/invalid value to the q8 default, so the web always has a value to seed.
    /// Made durable through this API (not just origin-keyed `localStorage`) for the same
    /// reason as theme/accent: the desktop shell's `127.0.0.1:<port>` origin changes every
    /// launch, wiping `localStorage`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    default_generation_quality: Option<String>,
}

/// User-selectable accent palettes. Keep in sync with web/src/accents.js.
const ACCENT_IDS: [&str; 7] = [
    "teal", "indigo", "cobalt", "violet", "coral", "amber", "emerald",
];

/// User-facing generation-quality tiers. Keep in sync with
/// web/src/quantTier.js `GENERATION_QUALITY_TIERS`.
const GENERATION_QUALITY_IDS: [&str; 3] = ["bf16", "q8", "q4"];

/// The app-wide default generation quality when unset or invalid. Matches the worker's
/// generation default (sc-10726) and the web `DEFAULT_GENERATION_QUALITY`.
const DEFAULT_GENERATION_QUALITY: &str = "q8";

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
        Some(value) if GENERATION_QUALITY_IDS.contains(&value) => value.to_owned(),
        _ => DEFAULT_GENERATION_QUALITY.to_owned(),
    }
}

/// Current UI preferences (empty object on first run).
pub(crate) async fn get_ui_preferences(
    State(state): State<AppState>,
) -> Result<Json<UiPreferences>, ApiError> {
    // sc-4202 (F-API-3): keep the preferences-file read off the async executor.
    let path = preferences_path(&state);
    let prefs = tokio::task::spawn_blocking(move || {
        let mut prefs = load_preferences(&path);
        // Always hand the web a concrete quality tier to seed (q8 when unset/invalid), so the
        // durable value survives a desktop relaunch even before the user changes it.
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
        // Only touch the quality when the payload actually carries it, so a theme- or
        // accent-only PUT can't reset it. A present-but-invalid value coerces to q8.
        if payload.default_generation_quality.is_some() {
            prefs.default_generation_quality = Some(normalize_generation_quality(
                payload.default_generation_quality.as_deref(),
            ));
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
    use super::{normalize_accent, normalize_generation_quality, normalize_theme};

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
    fn normalize_generation_quality_accepts_only_known_tiers_and_defaults_to_q8() {
        assert_eq!(normalize_generation_quality(Some(" bf16 ")), "bf16");
        assert_eq!(normalize_generation_quality(Some("q8")), "q8");
        assert_eq!(normalize_generation_quality(Some("q4")), "q4");
        // Unknown/candle-only tiers and absent values fall back to the q8 default.
        assert_eq!(normalize_generation_quality(Some("int8-convrot")), "q8");
        assert_eq!(normalize_generation_quality(Some("")), "q8");
        assert_eq!(normalize_generation_quality(None), "q8");
    }
}
