//! ProjectStore audio aggregate.

use super::*;

impl ProjectStore {
    /// List the project's saved voices (Voice Clone registry, sc-13517).
    pub fn list_saved_voices(&self, project_id: &str) -> ProjectStoreResult<Vec<Value>> {
        let project_path = self.find_project_path(project_id)?;
        crate::voice_store::SavedVoiceStore::new(project_path).list_saved_voices(project_id)
    }

    /// Register a saved voice; returns the hydrated voice plus any near-duplicate hit (sc-13517).
    pub fn create_saved_voice(
        &self,
        project_id: &str,
        input: crate::voice_store::SavedVoiceCreateInput,
        dedup_threshold: f32,
    ) -> ProjectStoreResult<(Value, Option<crate::voice_store::SavedVoiceDuplicate>)> {
        let project_path = self.find_project_path(project_id)?;
        crate::voice_store::SavedVoiceStore::new(project_path).create_saved_voice(
            project_id,
            input,
            dedup_threshold,
        )
    }

    /// Delete a saved voice (sc-13517).
    pub fn delete_saved_voice(
        &self,
        project_id: &str,
        voice_id: &str,
    ) -> ProjectStoreResult<crate::voice_store::SavedVoiceMutationResult> {
        let project_path = self.find_project_path(project_id)?;
        crate::voice_store::SavedVoiceStore::new(project_path)
            .delete_saved_voice(project_id, voice_id)
    }

    /// The project's YuE2 score version/render/comparison records (sc-22997).
    pub fn yue2_score_store(
        &self,
        project_id: &str,
    ) -> ProjectStoreResult<crate::yue2_score::store::Yue2ScoreStore> {
        let project_path = self.find_project_path(project_id)?;
        Ok(crate::yue2_score::store::Yue2ScoreStore::new(
            project_path,
            project_id,
        ))
    }
}
