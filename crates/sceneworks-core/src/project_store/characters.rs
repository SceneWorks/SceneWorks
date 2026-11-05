//! ProjectStore characters aggregate.

use super::*;

impl ProjectStore {
    pub fn list_characters(
        &self,
        project_id: &str,
        include_archived: bool,
    ) -> ProjectStoreResult<Vec<Value>> {
        let project_path = self.find_project_path(project_id)?;
        CharacterStore::new(&self.data_dir, project_path)
            .list_characters(project_id, include_archived)
    }

    pub fn create_character(
        &self,
        project_id: &str,
        input: CharacterCreateInput,
    ) -> ProjectStoreResult<Value> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).create_character(project_id, input)
    }

    pub fn get_character(&self, project_id: &str, character_id: &str) -> ProjectStoreResult<Value> {
        let project_path = self.find_project_path(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).get_character(project_id, character_id)
    }

    pub fn update_character(
        &self,
        project_id: &str,
        character_id: &str,
        input: CharacterUpdateInput,
    ) -> ProjectStoreResult<Value> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).update_character(
            project_id,
            character_id,
            input,
        )
    }

    pub fn archive_character(
        &self,
        project_id: &str,
        character_id: &str,
    ) -> ProjectStoreResult<CharacterMutationResult> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).archive_character(character_id)
    }

    pub fn purge_character(
        &self,
        project_id: &str,
        character_id: &str,
    ) -> ProjectStoreResult<CharacterMutationResult> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).purge_character(character_id)
    }

    pub fn add_character_reference(
        &self,
        project_id: &str,
        character_id: &str,
        input: CharacterReferenceInput,
    ) -> ProjectStoreResult<Value> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).add_reference(
            project_id,
            character_id,
            input,
        )
    }

    pub fn update_character_reference(
        &self,
        project_id: &str,
        character_id: &str,
        asset_id: &str,
        input: CharacterReferenceUpdateInput,
    ) -> ProjectStoreResult<Value> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).update_reference(
            project_id,
            character_id,
            asset_id,
            input,
        )
    }

    pub fn remove_character_reference(
        &self,
        project_id: &str,
        character_id: &str,
        asset_id: &str,
    ) -> ProjectStoreResult<Value> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).remove_reference(
            project_id,
            character_id,
            asset_id,
        )
    }

    pub fn create_character_look(
        &self,
        project_id: &str,
        character_id: &str,
        input: CharacterLookInput,
    ) -> ProjectStoreResult<Value> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).create_look(
            project_id,
            character_id,
            input,
        )
    }

    pub fn update_character_look(
        &self,
        project_id: &str,
        character_id: &str,
        look_id: &str,
        input: CharacterLookUpdateInput,
    ) -> ProjectStoreResult<Value> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).update_look(
            project_id,
            character_id,
            look_id,
            input,
        )
    }

    pub fn delete_character_look(
        &self,
        project_id: &str,
        character_id: &str,
        look_id: &str,
    ) -> ProjectStoreResult<Value> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).delete_look(
            project_id,
            character_id,
            look_id,
        )
    }

    pub fn attach_character_lora(
        &self,
        project_id: &str,
        character_id: &str,
        input: CharacterLoraInput,
    ) -> ProjectStoreResult<Value> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).attach_lora(
            project_id,
            character_id,
            input,
        )
    }

    pub fn update_character_lora(
        &self,
        project_id: &str,
        character_id: &str,
        link_id: &str,
        input: CharacterLoraUpdateInput,
    ) -> ProjectStoreResult<Value> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).update_lora_link(
            project_id,
            character_id,
            link_id,
            input,
        )
    }

    pub fn detach_character_lora(
        &self,
        project_id: &str,
        character_id: &str,
        link_id: &str,
    ) -> ProjectStoreResult<Value> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).detach_lora(
            project_id,
            character_id,
            link_id,
        )
    }
}
