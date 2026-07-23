use std::ffi::OsStr;

#[cfg(target_os = "linux")]
const SCENEWORKS_DMABUF_OPT_IN: &str = "SCENEWORKS_WEBKIT_DMABUF";
#[cfg(target_os = "linux")]
const WEBKIT_DISABLE_DMABUF_RENDERER: &str = "WEBKIT_DISABLE_DMABUF_RENDERER";

fn opt_in_enabled(value: Option<&OsStr>) -> bool {
    value
        .and_then(OsStr::to_str)
        .map(str::trim)
        .is_some_and(|value| {
            value.eq_ignore_ascii_case("1")
                || value.eq_ignore_ascii_case("true")
                || value.eq_ignore_ascii_case("yes")
                || value.eq_ignore_ascii_case("on")
        })
}

pub(crate) fn should_disable_dmabuf(
    sceneworks_opt_in: Option<&OsStr>,
    existing_webkit_value: Option<&OsStr>,
) -> bool {
    existing_webkit_value.is_none() && !opt_in_enabled(sceneworks_opt_in)
}

/// Configure WebKitGTK before Tauri creates the webview. The DMA-BUF renderer is
/// still driver/compositor-sensitive on the supported Ubuntu range and can yield
/// blank canvases or black video. Prefer the broadly compatible renderer by
/// default while preserving any explicit WebKit setting and providing an opt-in
/// for users who have validated DMA-BUF on their stack.
#[cfg(target_os = "linux")]
pub(crate) fn configure_environment() {
    if should_disable_dmabuf(
        std::env::var_os(SCENEWORKS_DMABUF_OPT_IN).as_deref(),
        std::env::var_os(WEBKIT_DISABLE_DMABUF_RENDERER).as_deref(),
    ) {
        std::env::set_var(WEBKIT_DISABLE_DMABUF_RENDERER, "1");
    }
}

#[cfg(test)]
mod tests {
    use super::should_disable_dmabuf;
    use std::ffi::OsStr;

    #[test]
    fn disables_dmabuf_by_default() {
        assert!(should_disable_dmabuf(None, None));
    }

    #[test]
    fn preserves_an_explicit_webkit_setting() {
        assert!(!should_disable_dmabuf(None, Some(OsStr::new("custom"))));
    }

    #[test]
    fn sceneworks_opt_in_accepts_common_truthy_values() {
        for value in ["1", "true", "YES", "on"] {
            assert!(!should_disable_dmabuf(Some(OsStr::new(value)), None));
        }
        assert!(should_disable_dmabuf(Some(OsStr::new("0")), None));
    }
}
