import { useAppDomainValue } from "../context/useAppDomainValue.js";
import { useProjectScopedList } from "./useProjectScopedList.js";
import { charactersApi } from "../api/charactersResources.js";
import { useCallback, useState } from "react";

// Owns the project's character roster plus every character CRUD/reference/look/LoRA
// mutation. Extracted from App.jsx (sc-1651) to shrink the god module; behavior is
// unchanged — shared concerns (token, active project, error/navigation) are passed in
// so the hook stays a thin data layer. Character test jobs surface live via the SSE
// job stream (the create endpoint publishes job.updated), so no post-create refetch.
//
// sc-4194: every returned action is wrapped in useCallback so its identity is stable
// across App's SSE-driven re-renders, which lets appContextValue memoize instead of
// rebuilding (and changing identity) on every job.updated tick.
export function useCharacters({ token, activeProject, activeProjectRef, setError, requestedGpu, setActiveView }) {
  const [characters, setCharacters] = useState([]);

  const refreshCharacters = useProjectScopedList({
    token, activeProject, activeProjectRef,
    load: charactersApi.list,
    setItems: setCharacters,
    setError: setError,
  });

  const withCharacterApi = useCallback(
    async (callback) => {
      if (!activeProject) {
        setError("Create or open a project first.");
        return null;
      }
      try {
        const result = await callback(activeProject.id);
        setError("");
        return result;
      } catch (err) {
        setError(err.message);
        return null;
      }
    },
    [activeProject, setError],
  );

  const createCharacter = useCallback(
    async (payload) =>
      withCharacterApi(async (projectId) => {
        const created = await charactersApi.list(projectId, token, {
          method: "POST",
          body: JSON.stringify(payload),
        });
        setCharacters((items) => [created, ...items.filter((item) => item.id !== created.id)]);
        return created;
      }),
    [withCharacterApi, token],
  );

  const updateCharacter = useCallback(
    async (characterId, changes) =>
      withCharacterApi(async (projectId) => {
        const updated = await charactersApi.item(projectId, characterId, token, {
          method: "PATCH",
          body: JSON.stringify(changes),
        });
        setCharacters((items) => items.map((item) => (item.id === updated.id ? updated : item)));
        return updated;
      }),
    [withCharacterApi, token],
  );

  const archiveCharacter = useCallback(
    async (characterId) =>
      withCharacterApi(async (projectId) => {
        await charactersApi.archive(projectId, characterId, token, { method: "POST" });
        setCharacters((items) => items.filter((item) => item.id !== characterId));
        return { id: characterId, status: "archived" };
      }),
    [withCharacterApi, token],
  );

  // Restore an archived character (sc-6066). Archive is a soft flag, so the existing
  // update endpoint un-archives via `{ archived: false }`. The active roster
  // (`characters`) is fetched without archived ones, so the restored character isn't
  // in it — add it back rather than mapping an existing entry.
  const unarchiveCharacter = useCallback(
    async (characterId) =>
      withCharacterApi(async (projectId) => {
        const updated = await charactersApi.item(projectId, characterId, token, {
          method: "PATCH",
          body: JSON.stringify({ archived: false }),
        });
        setCharacters((items) => [updated, ...items.filter((item) => item.id !== updated.id)]);
        return updated;
      }),
    [withCharacterApi, token],
  );

  // Fetch the project's archived characters for the dedicated "Archived" view
  // (sc-6066). The list endpoint returns both active and archived when
  // `include_archived=true`; the active roster is shown elsewhere, so keep only the
  // archived ones here. Does not touch the active `characters` state.
  const listArchivedCharacters = useCallback(
    async ({ signal } = {}) =>
      withCharacterApi(async (projectId) => {
        const items = await charactersApi.listIncludingArchived(projectId, token, { signal });
        return (items ?? []).filter((item) => item.archived);
      }),
    [withCharacterApi, token],
  );

  const addCharacterReference = useCallback(
    async (characterId, payload) =>
      withCharacterApi(async (projectId) => {
        const updated = await charactersApi.references(projectId, characterId, token, {
          method: "POST",
          body: JSON.stringify(payload),
        });
        setCharacters((items) => items.map((item) => (item.id === updated.id ? updated : item)));
        return updated;
      }),
    [withCharacterApi, token],
  );

  const updateCharacterReference = useCallback(
    async (characterId, assetId, changes) =>
      withCharacterApi(async (projectId) => {
        const updated = await charactersApi.reference(projectId, characterId, assetId, token, {
          method: "PATCH",
          body: JSON.stringify(changes),
        });
        setCharacters((items) => items.map((item) => (item.id === updated.id ? updated : item)));
        return updated;
      }),
    [withCharacterApi, token],
  );

  const removeCharacterReference = useCallback(
    async (characterId, assetId) =>
      withCharacterApi(async (projectId) => {
        const updated = await charactersApi.reference(projectId, characterId, assetId, token, {
          method: "DELETE",
        });
        setCharacters((items) => items.map((item) => (item.id === updated.id ? updated : item)));
        return updated;
      }),
    [withCharacterApi, token],
  );

  const createCharacterLook = useCallback(
    async (characterId, payload) =>
      withCharacterApi(async (projectId) => {
        const updated = await charactersApi.looks(projectId, characterId, token, {
          method: "POST",
          body: JSON.stringify(payload),
        });
        setCharacters((items) => items.map((item) => (item.id === updated.id ? updated : item)));
        return updated;
      }),
    [withCharacterApi, token],
  );

  const updateCharacterLook = useCallback(
    async (characterId, lookId, changes) =>
      withCharacterApi(async (projectId) => {
        const updated = await charactersApi.look(projectId, characterId, lookId, token, {
          method: "PATCH",
          body: JSON.stringify(changes),
        });
        setCharacters((items) => items.map((item) => (item.id === updated.id ? updated : item)));
        return updated;
      }),
    [withCharacterApi, token],
  );

  const deleteCharacterLook = useCallback(
    async (characterId, lookId) =>
      withCharacterApi(async (projectId) => {
        const updated = await charactersApi.look(projectId, characterId, lookId, token, {
          method: "DELETE",
        });
        setCharacters((items) => items.map((item) => (item.id === updated.id ? updated : item)));
        return updated;
      }),
    [withCharacterApi, token],
  );

  const attachCharacterLora = useCallback(
    async (characterId, payload) =>
      withCharacterApi(async (projectId) => {
        const updated = await charactersApi.loras(projectId, characterId, token, {
          method: "POST",
          body: JSON.stringify(payload),
        });
        setCharacters((items) => items.map((item) => (item.id === updated.id ? updated : item)));
        return updated;
      }),
    [withCharacterApi, token],
  );

  const updateCharacterLora = useCallback(
    async (characterId, linkId, changes) =>
      withCharacterApi(async (projectId) => {
        const updated = await charactersApi.lora(projectId, characterId, linkId, token, {
          method: "PATCH",
          body: JSON.stringify(changes),
        });
        setCharacters((items) => items.map((item) => (item.id === updated.id ? updated : item)));
        return updated;
      }),
    [withCharacterApi, token],
  );

  const detachCharacterLora = useCallback(
    async (characterId, linkId) =>
      withCharacterApi(async (projectId) => {
        const updated = await charactersApi.lora(projectId, characterId, linkId, token, {
          method: "DELETE",
        });
        setCharacters((items) => items.map((item) => (item.id === updated.id ? updated : item)));
        return updated;
      }),
    [withCharacterApi, token],
  );

  const createCharacterTestJob = useCallback(
    async (characterId, payload) =>
      withCharacterApi(async (projectId) => {
        await charactersApi.testJobs(projectId, characterId, token, {
          method: "POST",
          body: JSON.stringify({ ...payload, requestedGpu }),
        });
        setActiveView("Queue");
        return { id: characterId, status: "queued" };
      }),
    [withCharacterApi, token, requestedGpu, setActiveView],
  );

  const context = useAppDomainValue({
    characters,
    createCharacter,
    updateCharacter,
    archiveCharacter,
    unarchiveCharacter,
    listArchivedCharacters,
    addCharacterReference,
    updateCharacterReference,
    removeCharacterReference,
    createCharacterLook,
    updateCharacterLook,
    deleteCharacterLook,
    attachCharacterLora,
    updateCharacterLora,
    detachCharacterLora,
    createCharacterTestJob
  });
  return { ...context, setCharacters, refreshCharacters, context };
}
