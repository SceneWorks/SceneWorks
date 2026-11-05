//! Stable display and URL-path spellings shared by the API and downloader.

/// Format binary multiples using the historical B/KB/MB/GB/TB labels.
/// Bytes are integral; larger units retain one decimal and stop at TB.
pub fn format_bytes(value: u64) -> String {
    let mut size = value as f64;
    for unit in ["B", "KB", "MB", "GB", "TB"] {
        if size < 1024.0 || unit == "TB" {
            if unit == "B" {
                return format!("{} {unit}", size as u64);
            }
            return format!("{size:.1} {unit}");
        }
        size /= 1024.0;
    }
    format!("{size:.1} TB")
}

/// Percent-encode UTF-8 URL path bytes, retaining `/` separators and unreserved ASCII.
/// This is URL escaping, not POSIX shell quoting or display quoting.
pub fn quote_path(value: &str) -> String {
    let mut output = String::new();
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~' | b'/') {
            output.push(char::from(byte));
        } else {
            output.push_str(&format!("%{byte:02X}"));
        }
    }
    output
}
