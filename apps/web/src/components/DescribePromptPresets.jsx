import React, { useCallback, useEffect, useState } from "react";
import { trainingApi } from "../api/trainingResources.js";
import { isAbortError } from "../api.js";
import { errorMessage } from "../errorMessage.js";

// Named describing-prompt presets (SW-7086) for the "Create Prompt Batch from Dataset" dialog. The
// server lists the built-in default first (built from the worker's prompt asset; it cannot be
// changed or deleted), then the user's saved presets. Selecting one hands its exact text to the
// dialog's prompt editor; "Save as preset" stores the editor's current text under a name.
export function DescribePromptPresets({ token, prompt, disabled = false, onSelect }) {
  const [presets, setPresets] = useState([]);
  const [selectedId, setSelectedId] = useState("default");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const load = useCallback(async (signal) => {
    const list = await trainingApi.describePromptPresets(token, { signal });
    setPresets(Array.isArray(list) ? list : []);
    return list;
  }, [token]);

  useEffect(() => {
    const controller = new AbortController();
    load(controller.signal).catch((loadError) => {
      if (!isAbortError(loadError)) setError(errorMessage(loadError, "Could not load the saved prompts."));
    });
    return () => controller.abort();
  }, [load]);

  const selected = presets.find((preset) => preset.id === selectedId) ?? null;

  function select(id) {
    setSelectedId(id);
    setError("");
    const preset = presets.find((entry) => entry.id === id);
    if (preset && typeof preset.prompt === "string") onSelect(preset.prompt);
  }

  async function save() {
    if (busy || !name.trim() || !prompt.trim()) return;
    setBusy(true);
    setError("");
    try {
      const created = await trainingApi.describePromptPresets(token, {
        method: "POST",
        body: JSON.stringify({ name: name.trim(), prompt }),
      });
      await load();
      if (created?.id) setSelectedId(created.id);
      setName("");
    } catch (saveError) {
      setError(errorMessage(saveError, "Could not save the prompt."));
    } finally {
      setBusy(false);
    }
  }

  async function remove() {
    if (busy || !selected || selected.builtin) return;
    setBusy(true);
    setError("");
    try {
      await trainingApi.describePromptPreset(selected.id, token, { method: "DELETE" });
      await load();
      setSelectedId("default");
    } catch (deleteError) {
      setError(errorMessage(deleteError, "Could not delete the prompt."));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="describe-prompt-presets">
      <label>
        Saved prompt
        <select disabled={disabled || busy || !presets.length} onChange={(event) => select(event.target.value)} value={selectedId}>
          {presets.map((preset) => (
            <option key={preset.id} value={preset.id}>
              {preset.builtin ? `${preset.name} (built-in)` : preset.name}
            </option>
          ))}
        </select>
      </label>
      <button
        className="secondary-action"
        disabled={disabled || busy || !selected || selected.builtin}
        onClick={remove}
        title={selected?.builtin ? "The built-in default cannot be deleted" : "Delete this saved prompt"}
        type="button"
      >
        Delete
      </button>
      <label>
        Save as
        <input
          disabled={disabled || busy}
          onChange={(event) => setName(event.target.value)}
          placeholder="Preset name"
          value={name}
        />
      </label>
      <button
        className="secondary-action"
        disabled={disabled || busy || !name.trim() || !prompt.trim()}
        onClick={save}
        type="button"
      >
        Save prompt
      </button>
      {error ? <p className="inline-warning" role="alert">{error}</p> : null}
    </div>
  );
}
