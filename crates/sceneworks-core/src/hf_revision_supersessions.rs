//! Exact cache-compatibility declarations for a pinned Hugging Face revision that a newer pinned
//! revision of the SAME repository supersedes without changing any of the bytes an install holds.
//!
//! A manifest pins every re-host tier to an immutable commit and the worker loads that exact
//! `snapshots/<sha>/`. Bumping the pin therefore strands an install made at the previous commit
//! even when the new commit only ADDS a small file: the old snapshot holds every carried byte, but
//! under a directory the worker no longer reads. These declarations are the sibling of
//! [`crate::hf_repo_renames`]: exact across repository, both revisions, the tier's `files` filter,
//! every carried file (path + size, read off the Hub tree API at both commits, where each carried
//! file's blob id is identical), and the content of every added file. They permit only a cache-local
//! materialization of the new snapshot from the old one; they never authorize a network fetch, a
//! partial carry, or revision drift.

use crate::hf_repo_renames::validate_huggingface_snapshot_files;
use std::path::{Path, PathBuf};

/// A file present at both revisions with an identical blob id. `size` is the Hub-reported byte
/// length; the legacy snapshot entry must match it exactly, which rejects a torn or truncated
/// blob without hashing multi-GB weights.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct CarriedRevisionFile {
    pub path: &'static str,
    pub size: u64,
}

/// A file the superseding revision adds. Its full content is declared here so the cache-only
/// migration never needs the network. `blob` is the Hub-listed git object id, which is the name
/// Hugging Face's cache (and SceneWorks' downloader, via the `etag` header) gives a non-LFS blob,
/// so a later download of the same revision dedupes against it.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct AddedRevisionFile {
    pub path: &'static str,
    pub contents: &'static [u8],
    pub sha256: &'static str,
    pub blob: &'static str,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct HuggingFaceRevisionSupersession {
    pub repo: &'static str,
    pub legacy_revision: &'static str,
    pub current_revision: &'static str,
    /// The manifest row's exact `files` filter (a whole tier subdir).
    pub files: &'static [&'static str],
    pub carried: &'static [CarriedRevisionFile],
    pub added: &'static [AddedRevisionFile],
}

impl HuggingFaceRevisionSupersession {
    /// Every repo-relative file the superseding snapshot holds for this tier, carried first.
    pub fn current_files(&self) -> Vec<&'static str> {
        self.carried
            .iter()
            .map(|file| file.path)
            .chain(self.added.iter().map(|file| file.path))
            .collect()
    }
}

const QWEN_IMAGE_2_1_REHOST_REPO: &str = "SceneWorks/qwen-image-2-1-mlx";
/// sc-24114's publication of the q8/q4 re-host (the v0.9.0 pin).
const QWEN_IMAGE_2_1_REHOST_LEGACY_REVISION: &str = "1691de01c24a070131e0a28bf4c065fd027f4fe9";
/// sc-25909's republication: adds ONLY `q8/NOTICE` and `q4/NOTICE`.
const QWEN_IMAGE_2_1_REHOST_CURRENT_REVISION: &str = "ef03535eac756dd21feb7ac47c4f2df3e9b5b3c0";

/// The §3 attribution notice the converter writes as `<tier>/NOTICE` (its `attribution_notice()`:
/// the upstream licence notice plus a trailing newline), byte-identical to the published file.
const QWEN_RESEARCH_ATTRIBUTION_NOTICE: &[u8] = b"Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026 Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights Reserved.\n";
const QWEN_RESEARCH_ATTRIBUTION_NOTICE_SHA256: &str =
    "efa09ff399c8e4426b6e57e707c14e5db35258a290a15368a5bacd3e1e8381ab";
const QWEN_RESEARCH_ATTRIBUTION_NOTICE_BLOB: &str = "1cddfd79b04dc770b8994708cfa18f4f2ba30308";

macro_rules! carried {
    ($tier:literal, $te:literal, $dit:literal) => {
        &[
            CarriedRevisionFile {
                path: concat!($tier, "/CHANGES.md"),
                size: 1802,
            },
            CarriedRevisionFile {
                path: concat!($tier, "/LICENSE"),
                size: 7831,
            },
            CarriedRevisionFile {
                path: concat!($tier, "/README.md"),
                size: 5358,
            },
            CarriedRevisionFile {
                path: concat!($tier, "/SHA256SUMS"),
                size: 1831,
            },
            CarriedRevisionFile {
                path: concat!($tier, "/model_index.json"),
                size: 447,
            },
            CarriedRevisionFile {
                path: concat!($tier, "/processor/added_tokens.json"),
                size: 707,
            },
            CarriedRevisionFile {
                path: concat!($tier, "/processor/chat_template.jinja"),
                size: 5292,
            },
            CarriedRevisionFile {
                path: concat!($tier, "/processor/merges.txt"),
                size: 1_671_853,
            },
            CarriedRevisionFile {
                path: concat!($tier, "/processor/preprocessor_config.json"),
                size: 782,
            },
            CarriedRevisionFile {
                path: concat!($tier, "/processor/special_tokens_map.json"),
                size: 613,
            },
            CarriedRevisionFile {
                path: concat!($tier, "/processor/tokenizer.json"),
                size: 11_422_654,
            },
            CarriedRevisionFile {
                path: concat!($tier, "/processor/tokenizer_config.json"),
                size: 5445,
            },
            CarriedRevisionFile {
                path: concat!($tier, "/processor/video_preprocessor_config.json"),
                size: 817,
            },
            CarriedRevisionFile {
                path: concat!($tier, "/processor/vocab.json"),
                size: 2_776_833,
            },
            CarriedRevisionFile {
                path: concat!($tier, "/scheduler/scheduler_config.json"),
                size: 485,
            },
            CarriedRevisionFile {
                path: concat!($tier, "/text_encoder/config.json"),
                size: 1576,
            },
            CarriedRevisionFile {
                path: concat!($tier, "/text_encoder/model.safetensors"),
                size: $te,
            },
            CarriedRevisionFile {
                path: concat!($tier, "/transformer/config.json"),
                size: 429,
            },
            CarriedRevisionFile {
                path: concat!($tier, "/transformer/model.safetensors"),
                size: $dit,
            },
            CarriedRevisionFile {
                path: concat!($tier, "/vae/config.json"),
                size: 2079,
            },
            CarriedRevisionFile {
                path: concat!($tier, "/vae/diffusion_pytorch_model.safetensors"),
                size: 1_350_989_512,
            },
        ]
    };
}

pub const QWEN_IMAGE_2_1_Q8_NOTICE_SUPERSESSION: HuggingFaceRevisionSupersession =
    HuggingFaceRevisionSupersession {
        repo: QWEN_IMAGE_2_1_REHOST_REPO,
        legacy_revision: QWEN_IMAGE_2_1_REHOST_LEGACY_REVISION,
        current_revision: QWEN_IMAGE_2_1_REHOST_CURRENT_REVISION,
        files: &["q8/*"],
        carried: carried!("q8", 11_022_747_191, 7_559_920_470),
        added: &[AddedRevisionFile {
            path: "q8/NOTICE",
            contents: QWEN_RESEARCH_ATTRIBUTION_NOTICE,
            sha256: QWEN_RESEARCH_ATTRIBUTION_NOTICE_SHA256,
            blob: QWEN_RESEARCH_ATTRIBUTION_NOTICE_BLOB,
        }],
    };

pub const QWEN_IMAGE_2_1_Q4_NOTICE_SUPERSESSION: HuggingFaceRevisionSupersession =
    HuggingFaceRevisionSupersession {
        repo: QWEN_IMAGE_2_1_REHOST_REPO,
        legacy_revision: QWEN_IMAGE_2_1_REHOST_LEGACY_REVISION,
        current_revision: QWEN_IMAGE_2_1_REHOST_CURRENT_REVISION,
        files: &["q4/*"],
        carried: carried!("q4", 7_549_863_030, 4_002_363_835),
        added: &[AddedRevisionFile {
            path: "q4/NOTICE",
            contents: QWEN_RESEARCH_ATTRIBUTION_NOTICE,
            sha256: QWEN_RESEARCH_ATTRIBUTION_NOTICE_SHA256,
            blob: QWEN_RESEARCH_ATTRIBUTION_NOTICE_BLOB,
        }],
    };

const SUPERSESSIONS: &[HuggingFaceRevisionSupersession] = &[
    QWEN_IMAGE_2_1_Q8_NOTICE_SUPERSESSION,
    QWEN_IMAGE_2_1_Q4_NOTICE_SUPERSESSION,
];

/// The declaration for a manifest row whose repository, CURRENT revision and `files` filter all
/// match exactly; `None` for every other row.
pub fn exact_huggingface_revision_supersession(
    repo: &str,
    current_revision: &str,
    files: &[String],
) -> Option<&'static HuggingFaceRevisionSupersession> {
    SUPERSESSIONS.iter().find(|supersession| {
        supersession.repo == repo
            && supersession.current_revision == current_revision
            && files.len() == supersession.files.len()
            && files
                .iter()
                .zip(supersession.files)
                .all(|(actual, expected)| actual == expected)
    })
}

/// The tier subdirectory a supersession covers (`["q8/*"]` → `q8`).
pub fn supersession_tier(supersession: &HuggingFaceRevisionSupersession) -> Option<&'static str> {
    match supersession.files {
        [pattern] => pattern.strip_suffix("/*"),
        _ => None,
    }
}

/// Prove the LEGACY snapshot holds every carried file, confined to the repository cache, at its
/// exact declared size. Returns the canonical snapshot and the confined plain source of each
/// carried file (in declaration order). `None` refuses the migration: a partial, truncated or
/// escaping legacy tier is never carried forward.
pub fn validate_superseded_legacy_snapshot(
    hub_cache_root: &Path,
    repo_root: &Path,
    legacy_snapshot: &Path,
    supersession: &HuggingFaceRevisionSupersession,
) -> Option<(PathBuf, Vec<PathBuf>)> {
    let paths = supersession
        .carried
        .iter()
        .map(|file| file.path)
        .collect::<Vec<_>>();
    let (snapshot, sources) =
        validate_huggingface_snapshot_files(hub_cache_root, repo_root, legacy_snapshot, &paths)?;
    let sizes_match = sources
        .iter()
        .zip(supersession.carried)
        .all(|(source, file)| {
            std::fs::metadata(source).is_ok_and(|metadata| metadata.len() == file.size)
        });
    sizes_match.then_some((snapshot, sources))
}

/// Prove the CURRENT snapshot holds this tier's complete superseding file set — every carried file
/// at its declared size and every added file with its declared length — confined to the
/// repository cache. Returns the canonical snapshot directory.
pub fn validate_superseding_current_snapshot(
    hub_cache_root: &Path,
    repo_root: &Path,
    current_snapshot: &Path,
    supersession: &HuggingFaceRevisionSupersession,
) -> Option<PathBuf> {
    let paths = supersession.current_files();
    let (snapshot, sources) =
        validate_huggingface_snapshot_files(hub_cache_root, repo_root, current_snapshot, &paths)?;
    let expected_sizes = supersession.carried.iter().map(|file| file.size).chain(
        supersession
            .added
            .iter()
            .map(|file| file.contents.len() as u64),
    );
    sources
        .iter()
        .zip(expected_sizes)
        .all(|(source, size)| std::fs::metadata(source).is_ok_and(|m| m.len() == size))
        .then_some(snapshot)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn lookup_requires_the_exact_repo_current_revision_and_tier_filter() {
        let q8 = vec!["q8/*".to_owned()];
        assert_eq!(
            exact_huggingface_revision_supersession(
                QWEN_IMAGE_2_1_REHOST_REPO,
                QWEN_IMAGE_2_1_REHOST_CURRENT_REVISION,
                &q8
            ),
            Some(&QWEN_IMAGE_2_1_Q8_NOTICE_SUPERSESSION)
        );
        assert!(exact_huggingface_revision_supersession(
            QWEN_IMAGE_2_1_REHOST_REPO,
            QWEN_IMAGE_2_1_REHOST_LEGACY_REVISION,
            &q8
        )
        .is_none());
        assert!(exact_huggingface_revision_supersession(
            "SceneWorks/qwen-image-2-1-turbo-mlx",
            QWEN_IMAGE_2_1_REHOST_CURRENT_REVISION,
            &q8
        )
        .is_none());
        assert!(exact_huggingface_revision_supersession(
            QWEN_IMAGE_2_1_REHOST_REPO,
            QWEN_IMAGE_2_1_REHOST_CURRENT_REVISION,
            &["q8/transformer/*".to_owned()]
        )
        .is_none());
        assert_eq!(
            supersession_tier(&QWEN_IMAGE_2_1_Q4_NOTICE_SUPERSESSION),
            Some("q4")
        );
    }

    /// The embedded NOTICE is byte-identical to the published `q8/NOTICE` / `q4/NOTICE` at the
    /// current revision (148 B, sha256 from the Hub tree listing).
    #[test]
    fn the_declared_notice_is_the_published_bytes() {
        for supersession in SUPERSESSIONS {
            for added in supersession.added {
                assert_eq!(added.contents.len(), 148, "{}", added.path);
                assert_eq!(
                    format!("{:x}", Sha256::digest(added.contents)),
                    added.sha256,
                    "{}",
                    added.path
                );
            }
        }
    }

    /// The declaration is pinned to what ships: each shipped re-host row with this repo + filter
    /// pins `current_revision`, its measured `estimatedSizeBytes` is exactly the declared carried +
    /// added bytes, and the recorded download evidence for that revision lists exactly the declared
    /// files under the tier. A carried file dropped from (or added to) the list fails here.
    #[test]
    fn declarations_match_the_shipped_rows_and_the_recorded_download_evidence() {
        use serde_json::Value;
        let manifest: Value = serde_json::from_str(&crate::jsonc::strip_jsonc_comments(
            crate::builtin_manifests::BUILTIN_MANIFESTS
                .iter()
                .find(|(name, _)| *name == "builtin.models.jsonc")
                .expect("builtin.models.jsonc embedded")
                .1,
        ))
        .expect("builtin.models.jsonc parses");
        let evidence: Value = serde_json::from_str(include_str!(
            "../../../config/download-pattern-evidence.json"
        ))
        .expect("download evidence parses");
        for supersession in SUPERSESSIONS {
            let tier = supersession_tier(supersession).expect("whole-tier filter");
            let rows = manifest["models"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|model| model["downloads"].as_array().cloned().unwrap_or_default())
                .filter(|row| {
                    row["repo"] == supersession.repo
                        && row["files"] == serde_json::json!(supersession.files)
                })
                .collect::<Vec<_>>();
            assert_eq!(rows.len(), 1, "{:?}", supersession.files);
            assert_eq!(rows[0]["revision"], supersession.current_revision);
            let total = supersession
                .carried
                .iter()
                .map(|file| file.size)
                .sum::<u64>()
                + supersession
                    .added
                    .iter()
                    .map(|file| file.contents.len() as u64)
                    .sum::<u64>();
            assert_eq!(
                rows[0]["estimatedSizeBytes"].as_u64(),
                Some(total),
                "{tier}"
            );

            let key = format!("{}@{}", supersession.repo, supersession.current_revision);
            let listed = evidence["repos"]
                .as_array()
                .expect("evidence repos array")
                .iter()
                .find(|entry| entry["key"] == key.as_str())
                .unwrap_or_else(|| panic!("no download evidence for {key}"));
            let mut evidenced = listed["files"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(Value::as_str)
                .filter(|path| path.starts_with(&format!("{tier}/")))
                .collect::<Vec<_>>();
            evidenced.sort_unstable();
            let mut declared = supersession.current_files();
            declared.sort_unstable();
            assert_eq!(declared, evidenced, "{tier}");
        }
    }
}
