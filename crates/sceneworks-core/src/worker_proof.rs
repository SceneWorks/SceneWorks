//! Separate worker-only proof key for request-time local model source checks.

use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::Path;

const MASTER_FILE: &str = "worker-source-proof-master";

fn valid_secret(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

pub fn derive_worker_key(master: &str, worker_id: &str) -> io::Result<String> {
    if !valid_secret(master) || worker_id.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid worker proof identity",
        ));
    }
    let mut inner_pad = [0x36u8; 64];
    let mut outer_pad = [0x5cu8; 64];
    for (index, byte) in master.bytes().enumerate() {
        inner_pad[index] ^= byte;
        outer_pad[index] ^= byte;
    }
    let inner = Sha256::new()
        .chain_update(inner_pad)
        .chain_update(worker_id.as_bytes())
        .finalize();
    let outer = Sha256::new()
        .chain_update(outer_pad)
        .chain_update(inner)
        .finalize();
    Ok(format!("{outer:x}"))
}

pub fn load_or_create_master(credentials_dir: &Path) -> io::Result<String> {
    let supplied = std::env::var("SCENEWORKS_WORKER_PROOF_MASTER")
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty());
    if supplied
        .as_deref()
        .is_some_and(|value| !valid_secret(value))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid worker proof master",
        ));
    }
    fs::create_dir_all(credentials_dir)?;
    let path = credentials_dir.join(MASTER_FILE);
    let secret = supplied.clone().unwrap_or_else(|| {
        format!(
            "{}{}",
            uuid::Uuid::new_v4().simple(),
            uuid::Uuid::new_v4().simple()
        )
    });
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    match options.open(&path) {
        Ok(mut file) => {
            file.write_all(secret.as_bytes())?;
            file.sync_all()?;
            Ok(secret)
        }
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            let existing = read_master(&path)?;
            if supplied.as_ref().is_some_and(|value| value != &existing) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "configured worker proof master differs from credentials file",
                ));
            }
            Ok(existing)
        }
        Err(error) => Err(error),
    }
}

fn read_master(path: &Path) -> io::Result<String> {
    let value = fs::read_to_string(path)?.trim().to_owned();
    if valid_secret(&value) {
        Ok(value)
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid worker proof master file",
        ))
    }
}

pub fn worker_key_from_env_or_file(
    config_dir: &Path,
    worker_id: &str,
) -> io::Result<Option<String>> {
    if let Some(value) = std::env::var("SCENEWORKS_WORKER_PROOF_KEY")
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
    {
        if valid_secret(&value) {
            return Ok(Some(value));
        }
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid worker proof key",
        ));
    }
    derived_key_from_master_env_or_file(config_dir, worker_id)
}

pub fn derived_key_from_master_env_or_file(
    config_dir: &Path,
    worker_id: &str,
) -> io::Result<Option<String>> {
    if let Some(master) = std::env::var("SCENEWORKS_WORKER_PROOF_MASTER")
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
    {
        return derive_worker_key(&master, worker_id).map(Some);
    }
    let credentials_dir = crate::credentials::credentials_dir(config_dir);
    match read_master(&credentials_dir.join(MASTER_FILE)) {
        Ok(master) => derive_worker_key(&master, worker_id).map(Some),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derived_keys_are_worker_scoped_and_master_is_private() {
        let temp = tempfile::tempdir().unwrap();
        let master = load_or_create_master(temp.path()).unwrap();
        assert_eq!(load_or_create_master(temp.path()).unwrap(), master);
        assert_ne!(
            derive_worker_key(&master, "worker-a").unwrap(),
            derive_worker_key(&master, "worker-b").unwrap()
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(temp.path().join(MASTER_FILE))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }
    }
}
