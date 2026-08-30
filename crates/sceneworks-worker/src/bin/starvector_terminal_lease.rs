//! Cross-repository SC-22261 terminal-campaign advisory lease holder.
//!
//! The process deliberately holds the `fs2` lock until stdin closes. A dead
//! holder releases the OS lock automatically; callers must never delete or
//! overwrite a contended lock to "repair" an alleged stale owner.

use std::{
    env,
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::Path,
    process,
};

use fs2::FileExt;

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("starvector terminal lease: {message}");
    process::exit(2);
}

fn parent(path: &Path) -> &Path {
    path.parent()
        .unwrap_or_else(|| fail("lease path has no parent"))
}

fn main() {
    let args: Vec<_> = env::args().collect();
    if args.len() != 4 || args[1] != "hold" {
        fail("usage: starvector-terminal-lease hold <stable-lock-path> <owner-json>");
    }
    let path = Path::new(&args[2]);
    fs::create_dir_all(parent(path))
        .unwrap_or_else(|error| fail(format!("create lease root: {error}")));
    let mut file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(path)
        .unwrap_or_else(|error| fail(format!("open {}: {error}", path.display())));
    if let Err(error) = file.try_lock_exclusive() {
        let owner = fs::read_to_string(path).unwrap_or_else(|_| "unreadable".to_owned());
        fail(format!(
            "advisory lease is held; never auto-break it: {} owner={owner}; {error}",
            path.display()
        ));
    }
    file.set_len(0)
        .unwrap_or_else(|error| fail(format!("truncate owner: {error}")));
    file.write_all(args[3].as_bytes())
        .unwrap_or_else(|error| fail(format!("write owner: {error}")));
    file.sync_all()
        .unwrap_or_else(|error| fail(format!("sync owner: {error}")));
    println!("locked");
    io::stdout()
        .flush()
        .unwrap_or_else(|error| fail(format!("flush readiness: {error}")));
    let mut stdin = io::stdin();
    let mut discard = Vec::new();
    stdin
        .read_to_end(&mut discard)
        .unwrap_or_else(|error| fail(format!("wait for release: {error}")));
    FileExt::unlock(&file).unwrap_or_else(|error| fail(format!("unlock advisory lease: {error}")));
}
