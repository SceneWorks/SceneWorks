//! A generator-owner panic must stop the process before Rust unwinds GPU values.
//! The auto-worker supervisor owns job attribution after the child exits.
use std::cell::Cell;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

thread_local! { static GENERATOR_OWNER: Cell<bool> = const { Cell::new(false) }; }
static INSTALLED: OnceLock<()> = OnceLock::new();
const MAX_DIAGNOSTIC_BYTES: usize = 512;

pub(crate) fn install(path: Option<PathBuf>) {
    INSTALLED.get_or_init(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if GENERATOR_OWNER.try_with(Cell::get).unwrap_or(false) {
                let payload = info
                    .payload()
                    .downcast_ref::<&str>()
                    .copied()
                    .or_else(|| info.payload().downcast_ref::<String>().map(String::as_str))
                    .unwrap_or("non-string panic payload");
                if let Some(path) = &path {
                    write_diagnostic(path, payload);
                }
                // Do not run the previous hook, destructors, or the cache's catch_unwind.
                // GPU values on this stack may be unsafe to release after a driver fault.
                std::process::abort();
            }
            previous(info);
        }));
    });
}

pub(crate) fn on_generator_owner<R>(run: impl FnOnce() -> R) -> R {
    GENERATOR_OWNER.with(|owner| {
        let old = owner.replace(true);
        struct Reset<'a>(&'a Cell<bool>, bool);
        impl Drop for Reset<'_> {
            fn drop(&mut self) {
                self.0.set(self.1);
            }
        }
        let _reset = Reset(owner, old);
        run()
    })
}

fn write_diagnostic(path: &Path, payload: &str) {
    let mut end = payload.len().min(MAX_DIAGNOSTIC_BYTES);
    while !payload.is_char_boundary(end) {
        end -= 1;
    }
    if let Ok(mut file) = OpenOptions::new().write(true).create_new(true).open(path) {
        let _ = file.write_all(payload[..end].as_bytes());
        let _ = file.sync_all();
    }
}

pub(crate) fn read_diagnostic(path: &Path) -> Option<String> {
    use std::io::Read;
    (|| {
        let mut file = std::fs::File::open(path).ok()?;
        let mut bytes = Vec::new();
        std::io::Read::by_ref(&mut file)
            .take(MAX_DIAGNOSTIC_BYTES as u64)
            .read_to_end(&mut bytes)
            .ok()?;
        if bytes.is_empty() {
            return Some("generator panic diagnostic unavailable".to_owned());
        }
        Some(String::from_utf8_lossy(&bytes).into_owned())
    })()
}

pub(crate) fn consume_diagnostic(path: &Path) {
    let _ = std::fs::remove_file(path);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owner_panic_aborts_before_drop() {
        if let Some(dir) = std::env::var_os("SCENEWORKS_PANIC_PROTOCOL_CHILD") {
            let dir = PathBuf::from(dir);
            install(Some(dir.join("diagnostic")));
            struct DropMarker(PathBuf);
            impl Drop for DropMarker {
                fn drop(&mut self) {
                    std::fs::write(&self.0, b"dropped").unwrap();
                }
            }
            on_generator_owner(|| {
                let _gpu_like_value = DropMarker(dir.join("dropped"));
                panic!("bounded panic receipt");
            });
            unreachable!();
        }
        let dir = tempfile::tempdir().unwrap();
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "generator_panic::tests::owner_panic_aborts_before_drop",
                "--nocapture",
            ])
            .env("SCENEWORKS_PANIC_PROTOCOL_CHILD", dir.path())
            .output()
            .unwrap();
        assert!(!output.status.success());
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            assert_eq!(output.status.signal(), Some(6), "owner panic must abort");
        }
        assert!(
            !dir.path().join("dropped").exists(),
            "owner stack must not unwind"
        );
        assert_eq!(
            read_diagnostic(&dir.path().join("diagnostic")).as_deref(),
            Some("bounded panic receipt")
        );
        consume_diagnostic(&dir.path().join("diagnostic"));
        assert!(
            !dir.path().join("diagnostic").exists(),
            "receipt is consumed once"
        );
    }

    #[test]
    fn bounded_diagnostic_rejects_oversize_and_preserves_utf8() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("receipt");
        write_diagnostic(&path, &format!("{}é", "x".repeat(MAX_DIAGNOSTIC_BYTES - 1)));
        assert_eq!(
            read_diagnostic(&path).unwrap().len(),
            MAX_DIAGNOSTIC_BYTES - 1
        );
        consume_diagnostic(&path);
    }
}
