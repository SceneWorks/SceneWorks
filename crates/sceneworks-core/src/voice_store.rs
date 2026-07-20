//! Per-project **saved voices** registry (sc-13517, epic 13400 C4 follow-up).
//!
//! A "saved voice" is a named, reusable pointer to a library **reference audio clip** plus the
//! Chatterbox-VE speaker **embedding** computed from it. The load-bearing element for generation is
//! the reference audio asset id: picking a saved voice supplies `referenceAudioAssetId` to the
//! existing Voice Clone pipeline (native `chatterbox_tts` with OpenVoice V2 fallback), both of which
//! drive off the reference AUDIO — the stored embedding is NOT a generation input (VoiceEmbedding-only
//! errors at S3Gen; see sc-13412). The embedding's real consumer is **near-duplicate detection**:
//! at register time we cosine-compare the new embedding against the project's existing saved voices
//! and flag a likely re-registration of the same speaker.
//!
//! Storage mirrors the training-dataset satellite precedent: a `saved_voices` table in the
//! project.db, migrated through `apply_project_migrations` (schema-version gated). Unlike
//! characters/assets/timelines there is no sidecar — a saved voice has no on-disk artifact of its
//! own beyond the referenced asset, so the DB row is authoritative and survives the reindex that a
//! schema bump triggers (the reindex only clears the sidecar-rebuilt tables, never this one).

use std::path::PathBuf;

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::project_store::{connect_project_db_migrated, ProjectStoreError, ProjectStoreResult};
use crate::store_util::{is_safe_id, random_hex};
use crate::time::utc_now;

/// The Chatterbox-VE (GE2E speaker encoder) embedding dimensionality.
pub const VOICE_EMBEDDING_DIM: usize = 256;

/// Default cosine-similarity threshold above which a newly-registered voice is flagged as a likely
/// near-duplicate of an existing saved voice. Chatterbox-VE embeddings of the SAME speaker cluster
/// high (the same clip ~1.0; the same speaker on a different clip typically > 0.9), while distinct
/// speakers sit well below — the sc-13411 smoke measured converted-vs-reference 0.79 and
/// base-vs-reference 0.65 across *different* Kokoro voices. 0.92 flags a re-registration of the same
/// voice without tripping on merely-similar-but-distinct speakers. Callers may override per request.
pub const DEFAULT_VOICE_DEDUP_THRESHOLD: f32 = 0.92;

const MAX_VOICE_NAME_LEN: usize = 200;

/// Inputs for registering a saved voice. The `embedding` is computed upstream (the worker's
/// Chatterbox-VE embed path) from the chosen reference clip; the store treats it as opaque identity
/// data for dedup and persistence.
#[derive(Debug, Clone)]
pub struct SavedVoiceCreateInput {
    pub name: String,
    pub reference_audio_asset_id: String,
    pub embedding: Vec<f32>,
}

/// A near-duplicate hit surfaced at register time so the UI can warn the user.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SavedVoiceDuplicate {
    pub id: String,
    pub name: String,
    pub similarity: f32,
}

/// The result of a delete.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SavedVoiceMutationResult {
    pub id: String,
    pub status: String,
}

/// Create the `saved_voices` table. Hooked into `apply_project_migrations` (schema-version gated), so
/// this replays idempotently on existing DBs once `PROJECT_SCHEMA_VERSION` advances. `text` ids +
/// `text` timestamps + the embedding as a JSON-encoded f32 array match every other project.db table.
pub fn apply_voice_migrations(connection: &Connection) -> ProjectStoreResult<()> {
    connection.execute_batch(
        "
        create table if not exists saved_voices (
          id text primary key,
          project_id text not null,
          name text not null,
          reference_audio_asset_id text not null,
          embedding text not null,
          created_at text not null
        );
        ",
    )?;
    Ok(())
}

/// L2-normalized cosine similarity in `[-1, 1]`. Chatterbox-VE returns raw (un-normalized) vectors, so
/// both sides are normalized here (matches the `cosine` helper the sc-13411 smoke uses as evidence).
/// Returns `0.0` when either vector has zero magnitude.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Per-project saved-voices data access. Constructed per call by [`crate::project_store::ProjectStore`]
/// delegating methods (mirrors `CharacterStore`).
pub struct SavedVoiceStore {
    project_path: PathBuf,
}

impl SavedVoiceStore {
    pub fn new(project_path: impl Into<PathBuf>) -> Self {
        Self {
            project_path: project_path.into(),
        }
    }

    /// List the project's saved voices, newest first. The raw embedding is intentionally omitted from
    /// the hydrated view — the UI needs only the identity + reference pointer, and the 256-float
    /// vector is dedup-only internal data.
    pub fn list_saved_voices(&self, project_id: &str) -> ProjectStoreResult<Vec<Value>> {
        let connection = connect_project_db_migrated(&self.project_path)?;
        let mut statement = connection.prepare(
            "select id, name, reference_audio_asset_id, created_at
             from saved_voices
             where project_id = ?1
             order by created_at desc, name asc",
        )?;
        let rows = statement
            .query_map(params![project_id], |row| {
                Ok(json!({
                    "id": row.get::<_, String>(0)?,
                    "name": row.get::<_, String>(1)?,
                    "referenceAudioAssetId": row.get::<_, String>(2)?,
                    "createdAt": row.get::<_, String>(3)?,
                }))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Register a saved voice. Runs the near-duplicate check against the project's existing voices
    /// FIRST (the embedding's real consumer), then persists the row. Returns the hydrated voice plus
    /// the best duplicate hit at/above `dedup_threshold`, if any — the UI surfaces it as a warning.
    pub fn create_saved_voice(
        &self,
        project_id: &str,
        input: SavedVoiceCreateInput,
        dedup_threshold: f32,
    ) -> ProjectStoreResult<(Value, Option<SavedVoiceDuplicate>)> {
        let name = input.name.trim();
        if name.is_empty() {
            return Err(ProjectStoreError::BadRequest(
                "Voice name must not be empty".to_owned(),
            ));
        }
        if name.chars().count() > MAX_VOICE_NAME_LEN {
            return Err(ProjectStoreError::BadRequest(format!(
                "Voice name must be at most {MAX_VOICE_NAME_LEN} characters"
            )));
        }
        if !is_safe_id(&input.reference_audio_asset_id) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid reference audio asset id".to_owned(),
            ));
        }
        if input.embedding.is_empty() {
            return Err(ProjectStoreError::BadRequest(
                "Voice embedding must not be empty".to_owned(),
            ));
        }

        let connection = connect_project_db_migrated(&self.project_path)?;
        let duplicate =
            nearest_existing_voice(&connection, project_id, &input.embedding, dedup_threshold)?;

        let id = format!("voice_{}", random_hex(16)?);
        let now = utc_now();
        let embedding_json = serde_json::to_string(&input.embedding).map_err(|error| {
            ProjectStoreError::BadRequest(format!("Voice embedding is not serializable: {error}"))
        })?;
        connection.execute(
            "insert into saved_voices
               (id, project_id, name, reference_audio_asset_id, embedding, created_at)
             values (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                id,
                project_id,
                name,
                input.reference_audio_asset_id,
                embedding_json,
                now
            ],
        )?;

        let voice = json!({
            "id": id,
            "name": name,
            "referenceAudioAssetId": input.reference_audio_asset_id,
            "createdAt": now,
        });
        Ok((voice, duplicate))
    }

    /// Permanently delete a saved voice. `NotFound` when the id doesn't belong to the project.
    pub fn delete_saved_voice(
        &self,
        project_id: &str,
        voice_id: &str,
    ) -> ProjectStoreResult<SavedVoiceMutationResult> {
        if !is_safe_id(voice_id) {
            return Err(ProjectStoreError::BadRequest("Invalid voice id".to_owned()));
        }
        let connection = connect_project_db_migrated(&self.project_path)?;
        let affected = connection.execute(
            "delete from saved_voices where id = ?1 and project_id = ?2",
            params![voice_id, project_id],
        )?;
        if affected == 0 {
            return Err(ProjectStoreError::NotFound(
                "Saved voice not found".to_owned(),
            ));
        }
        Ok(SavedVoiceMutationResult {
            id: voice_id.to_owned(),
            status: "deleted".to_owned(),
        })
    }
}

/// Return the existing saved voice whose embedding is most cosine-similar to `embedding`, but only
/// when that similarity is at/above `threshold`. Rows whose stored embedding is unparseable or a
/// different dimensionality are skipped (a saved voice from a different embedder must not false-match).
fn nearest_existing_voice(
    connection: &Connection,
    project_id: &str,
    embedding: &[f32],
    threshold: f32,
) -> ProjectStoreResult<Option<SavedVoiceDuplicate>> {
    let mut statement =
        connection.prepare("select id, name, embedding from saved_voices where project_id = ?1")?;
    let candidates = statement
        .query_map(params![project_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;

    let mut best: Option<SavedVoiceDuplicate> = None;
    for (id, name, embedding_json) in candidates {
        let Ok(other) = serde_json::from_str::<Vec<f32>>(&embedding_json) else {
            continue;
        };
        if other.len() != embedding.len() {
            continue;
        }
        let similarity = cosine_similarity(embedding, &other);
        let beats_current = match &best {
            None => true,
            Some(current) => similarity > current.similarity,
        };
        if similarity >= threshold && beats_current {
            best = Some(SavedVoiceDuplicate {
                id,
                name,
                similarity,
            });
        }
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    fn migrated_db() -> Connection {
        let connection = Connection::open_in_memory().expect("in-memory db");
        apply_voice_migrations(&connection).expect("voice migration runs");
        connection
    }

    /// The store's public delegating layer opens the real project.db; the unit tests here drive the
    /// SQL directly against an in-memory DB via small helpers that mirror the store methods, so they
    /// don't need a project registry on disk.
    fn insert(connection: &Connection, id: &str, project: &str, name: &str, embedding: &[f32]) {
        connection
            .execute(
                "insert into saved_voices (id, project_id, name, reference_audio_asset_id, embedding, created_at)
                 values (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    id,
                    project,
                    name,
                    "asset_ref",
                    serde_json::to_string(embedding).unwrap(),
                    "2026-07-20T00:00:00Z"
                ],
            )
            .expect("insert voice");
    }

    #[test]
    fn cosine_similarity_is_one_for_identical_and_zero_for_orthogonal() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
        // Scale-invariant: a parallel vector is still 1.0.
        let scaled: Vec<f32> = a.iter().map(|x| x * 7.5).collect();
        assert!((cosine_similarity(&a, &scaled) - 1.0).abs() < 1e-6);
        // Orthogonal → 0.
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        // Zero magnitude → 0, never NaN.
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }

    #[test]
    fn dedup_flags_a_near_identical_embedding_and_ignores_a_distinct_one() {
        let connection = migrated_db();
        // An existing "Narrator" voice.
        let narrator = vec![0.9_f32, 0.1, 0.2, 0.05, 0.3];
        insert(&connection, "voice_a", "project_1", "Narrator", &narrator);

        // A near-identical clip (tiny perturbation) must trip the threshold.
        let near: Vec<f32> = narrator.iter().map(|x| x + 0.001).collect();
        let hit = nearest_existing_voice(
            &connection,
            "project_1",
            &near,
            DEFAULT_VOICE_DEDUP_THRESHOLD,
        )
        .expect("dedup query");
        let hit = hit.expect("near-identical embedding should flag a duplicate");
        assert_eq!(hit.id, "voice_a");
        assert_eq!(hit.name, "Narrator");
        assert!(hit.similarity >= DEFAULT_VOICE_DEDUP_THRESHOLD);

        // A clearly distinct speaker (near-orthogonal) must NOT flag.
        let distinct = vec![-0.2_f32, 0.8, -0.6, 0.7, -0.9];
        let miss = nearest_existing_voice(
            &connection,
            "project_1",
            &distinct,
            DEFAULT_VOICE_DEDUP_THRESHOLD,
        )
        .expect("dedup query");
        assert!(
            miss.is_none(),
            "a distinct speaker embedding must not be flagged as a duplicate (got {miss:?})"
        );
    }

    #[test]
    fn dedup_is_scoped_per_project() {
        let connection = migrated_db();
        let embedding = vec![0.5_f32, 0.5, 0.5, 0.5];
        insert(&connection, "voice_a", "project_1", "Narrator", &embedding);
        // The identical embedding registered under a DIFFERENT project must not match.
        let hit =
            nearest_existing_voice(&connection, "project_2", &embedding, 0.5).expect("dedup query");
        assert!(hit.is_none(), "dedup must not cross project boundaries");
    }

    #[test]
    fn dedup_skips_mismatched_dimensionality() {
        let connection = migrated_db();
        insert(
            &connection,
            "voice_a",
            "project_1",
            "Narrator",
            &[1.0, 2.0, 3.0],
        );
        // A 4-d probe against a 3-d stored embedding must be skipped, not panic or false-match.
        let hit = nearest_existing_voice(&connection, "project_1", &[1.0, 2.0, 3.0, 4.0], 0.0)
            .expect("dedup query");
        assert!(hit.is_none());
    }

    /// End-to-end create → list → delete round-trip against a real (temp) project.db — the exact
    /// data path the rust-api create/list/delete handlers delegate to via `ProjectStore`. Also proves
    /// the create-time dedup consumer fires on a re-register and stays quiet for a distinct voice.
    #[test]
    fn store_round_trips_create_list_delete_with_dedup() {
        let temp = tempfile::tempdir().expect("temp dir");
        let store = SavedVoiceStore::new(temp.path());

        assert!(
            store
                .list_saved_voices("project_1")
                .expect("empty list")
                .is_empty(),
            "a fresh project has no saved voices"
        );

        // First register: no existing voices, so no duplicate.
        let (first, dup) = store
            .create_saved_voice(
                "project_1",
                SavedVoiceCreateInput {
                    name: "Narrator".to_owned(),
                    reference_audio_asset_id: "asset_ref_1".to_owned(),
                    embedding: vec![0.9, 0.1, 0.2, 0.05, 0.3],
                },
                DEFAULT_VOICE_DEDUP_THRESHOLD,
            )
            .expect("create first voice");
        assert!(dup.is_none(), "first voice can't duplicate anything");
        assert_eq!(first["name"], "Narrator");
        assert_eq!(first["referenceAudioAssetId"], "asset_ref_1");
        let first_id = first["id"].as_str().expect("voice id").to_owned();

        // Re-register a near-identical embedding: the dedup consumer flags the existing voice.
        let (_second, dup2) = store
            .create_saved_voice(
                "project_1",
                SavedVoiceCreateInput {
                    name: "Narrator (again)".to_owned(),
                    reference_audio_asset_id: "asset_ref_2".to_owned(),
                    embedding: vec![0.901, 0.099, 0.2, 0.05, 0.301],
                },
                DEFAULT_VOICE_DEDUP_THRESHOLD,
            )
            .expect("create near-duplicate voice");
        let dup2 = dup2.expect("a near-identical embedding must be flagged");
        assert_eq!(dup2.name, "Narrator");

        // A distinct voice registers cleanly.
        let (_third, dup3) = store
            .create_saved_voice(
                "project_1",
                SavedVoiceCreateInput {
                    name: "Villain".to_owned(),
                    reference_audio_asset_id: "asset_ref_3".to_owned(),
                    embedding: vec![-0.2, 0.8, -0.6, 0.7, -0.9],
                },
                DEFAULT_VOICE_DEDUP_THRESHOLD,
            )
            .expect("create distinct voice");
        assert!(dup3.is_none(), "a distinct speaker must not be flagged");

        // List reflects all three (embedding intentionally omitted from the view).
        let listed = store.list_saved_voices("project_1").expect("list");
        assert_eq!(listed.len(), 3);
        assert!(listed.iter().all(|voice| voice.get("embedding").is_none()));

        // Delete the first, then a re-delete is a clean NotFound.
        let result = store
            .delete_saved_voice("project_1", &first_id)
            .expect("delete voice");
        assert_eq!(result.status, "deleted");
        assert_eq!(store.list_saved_voices("project_1").expect("list").len(), 2);
        assert!(
            store.delete_saved_voice("project_1", &first_id).is_err(),
            "deleting a missing voice is NotFound"
        );
    }

    #[test]
    fn create_rejects_empty_name_and_bad_asset_id() {
        let temp = tempfile::tempdir().expect("temp dir");
        let store = SavedVoiceStore::new(temp.path());
        assert!(store
            .create_saved_voice(
                "project_1",
                SavedVoiceCreateInput {
                    name: "   ".to_owned(),
                    reference_audio_asset_id: "asset_ref".to_owned(),
                    embedding: vec![1.0, 2.0],
                },
                DEFAULT_VOICE_DEDUP_THRESHOLD,
            )
            .is_err());
        assert!(store
            .create_saved_voice(
                "project_1",
                SavedVoiceCreateInput {
                    name: "Ok".to_owned(),
                    reference_audio_asset_id: "../escape".to_owned(),
                    embedding: vec![1.0, 2.0],
                },
                DEFAULT_VOICE_DEDUP_THRESHOLD,
            )
            .is_err());
    }
}
