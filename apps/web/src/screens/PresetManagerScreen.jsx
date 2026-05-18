import React, { useEffect, useMemo, useState } from "react";

const workflowOptions = [
  ["text_to_image", "Text to Image"],
  ["edit_image", "Image Edit"],
  ["image_to_video", "Image to Video"],
  ["text_to_video", "Text to Video"],
  ["first_last_frame", "First/Last Frame"],
];

const defaultModesByWorkflow = {
  text_to_image: "text_to_image, character_image, style_variations",
  edit_image: "edit_image",
  image_to_video: "image_to_video",
  text_to_video: "text_to_video",
  first_last_frame: "first_last_frame",
};

function slugify(value) {
  return String(value ?? "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function loraId(lora) {
  return typeof lora === "string" ? lora : lora?.id;
}

function parseList(value) {
  return String(value ?? "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function modelOptions(models, workflow) {
  const type = workflow.includes("video") || workflow === "first_last_frame" ? "video" : "image";
  return models.filter((model) => model.type === type);
}

function formFromPreset(preset, fallbackModel) {
  return {
    id: preset?.id ?? "",
    name: preset?.name ?? "",
    scope: preset?.scope === "project" ? "project" : "global",
    workflow: preset?.workflow ?? "text_to_image",
    modes: (preset?.modes ?? []).join(", "),
    model: preset?.model ?? fallbackModel ?? "",
    order: preset?.order ?? "",
    count: preset?.defaults?.count ?? "",
    resolution: preset?.defaults?.resolution ?? "",
    negativePrompt: preset?.defaults?.negativePrompt ?? "",
    promptPrefix: preset?.prompt?.prefix ?? "",
    promptSuffix: preset?.prompt?.suffix ?? "",
    description: preset?.ui?.description ?? "",
    loraIds: (preset?.builtInLoras ?? preset?.loras ?? []).map(loraId).filter(Boolean),
  };
}

export function PresetManagerScreen({
  activeProject,
  createRecipePreset,
  deleteRecipePreset,
  duplicateRecipePreset,
  imageModels,
  loras = [],
  recipePresets = [],
  updateRecipePreset,
  videoModels,
}) {
  const models = useMemo(() => [...imageModels, ...videoModels], [imageModels, videoModels]);
  const [selectedPresetId, setSelectedPresetId] = useState(recipePresets.find((preset) => preset.scope !== "builtin")?.id ?? "");
  const selectedPreset = recipePresets.find((preset) => preset.id === selectedPresetId) ?? null;
  const [form, setForm] = useState(() => formFromPreset(selectedPreset, models[0]?.id));
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");
  const editable = !selectedPreset || selectedPreset.scope !== "builtin";
  const availableModels = modelOptions(models, form.workflow);
  const compatibleLoras = loras.filter((lora) => lora.presetManaged || lora.scope === "builtin");

  useEffect(() => {
    if (selectedPreset && !recipePresets.some((preset) => preset.id === selectedPreset.id)) {
      setSelectedPresetId(recipePresets.find((preset) => preset.scope !== "builtin")?.id ?? "");
    }
  }, [recipePresets, selectedPreset?.id]);

  useEffect(() => {
    setForm(formFromPreset(selectedPreset, modelOptions(models, selectedPreset?.workflow ?? "text_to_image")[0]?.id ?? models[0]?.id));
    setMessage("");
  }, [selectedPreset?.id, models.length]);

  useEffect(() => {
    if (!availableModels.length) {
      return;
    }
    if (!availableModels.some((model) => model.id === form.model)) {
      setForm((current) => ({ ...current, model: availableModels[0].id }));
    }
  }, [availableModels, form.model]);

  function updateField(field, value) {
    setForm((current) => {
      if (field === "workflow") {
        return {
          ...current,
          workflow: value,
          modes: current.modes || defaultModesByWorkflow[value] || value,
        };
      }
      if (field === "name" && !selectedPreset) {
        return { ...current, name: value, id: slugify(value) };
      }
      return { ...current, [field]: value };
    });
  }

  function toggleLora(id) {
    setForm((current) => {
      const hasLora = current.loraIds.includes(id);
      const loraIds = hasLora ? current.loraIds.filter((item) => item !== id) : [...current.loraIds, id].slice(0, 3);
      return { ...current, loraIds };
    });
  }

  function buildPayload() {
    const defaults = {};
    if (form.count !== "") {
      defaults.count = Number(form.count);
    }
    if (form.resolution.trim()) {
      defaults.resolution = form.resolution.trim();
    }
    if (form.negativePrompt.trim()) {
      defaults.negativePrompt = form.negativePrompt.trim();
    }
    const prompt = {};
    if (form.promptPrefix.trim()) {
      prompt.prefix = form.promptPrefix.trim();
    }
    if (form.promptSuffix.trim()) {
      prompt.suffix = form.promptSuffix.trim();
    }
    const payload = {
      id: slugify(form.id || form.name),
      name: form.name.trim(),
      scope: form.scope,
      workflow: form.workflow,
      modes: parseList(form.modes),
      model: form.model,
      builtInLoras: form.loraIds.map((id) => ({ id })),
      ui: { description: form.description.trim() },
    };
    if (form.order !== "") {
      payload.order = Number(form.order);
    }
    if (Object.keys(defaults).length) {
      payload.defaults = defaults;
    }
    if (Object.keys(prompt).length) {
      payload.prompt = prompt;
    }
    return payload;
  }

  async function savePreset(event) {
    event.preventDefault();
    setSaving(true);
    setMessage("");
    try {
      const payload = buildPayload();
      if (selectedPreset) {
        await updateRecipePreset(selectedPreset.id, payload);
        setMessage("Preset saved.");
      } else {
        const created = await createRecipePreset(payload);
        setSelectedPresetId(created?.id ?? payload.id);
        setMessage("Preset created.");
      }
    } catch (err) {
      setMessage(err.message);
    } finally {
      setSaving(false);
    }
  }

  async function duplicateSelected() {
    if (!selectedPreset) {
      return;
    }
    setSaving(true);
    setMessage("");
    try {
      if (selectedPreset.scope === "builtin") {
        const payload = buildPayload();
        payload.id = slugify(`${selectedPreset.id}_copy`);
        payload.name = `${selectedPreset.name ?? selectedPreset.id} Copy`;
        const created = await createRecipePreset(payload);
        setSelectedPresetId(created.id);
        setMessage("Preset duplicated.");
        return;
      }
      const duplicated = await duplicateRecipePreset(selectedPreset.id, form.scope);
      setSelectedPresetId(duplicated.id);
      setMessage("Preset duplicated.");
    } catch (err) {
      setMessage(err.message);
    } finally {
      setSaving(false);
    }
  }

  async function archiveSelected() {
    if (!selectedPreset || selectedPreset.scope === "builtin") {
      return;
    }
    setSaving(true);
    setMessage("");
    try {
      await deleteRecipePreset(selectedPreset.id);
      setSelectedPresetId("");
      setMessage("Preset archived.");
    } catch (err) {
      setMessage(err.message);
    } finally {
      setSaving(false);
    }
  }

  function startNewPreset() {
    setSelectedPresetId("");
    setForm(formFromPreset(null, modelOptions(models, "text_to_image")[0]?.id ?? models[0]?.id));
    setMessage("");
  }

  return (
    <section className="main-surface preset-manager">
      <div className="surface-header">
        <div className="section-heading">
          <p className="eyebrow">Preset Manager</p>
          <h2>{activeProject ? activeProject.name : "Global presets"}</h2>
        </div>
        <div className="toolbar">
          <button onClick={startNewPreset} type="button">
            New Preset
          </button>
          <button disabled={!selectedPreset || saving} onClick={duplicateSelected} type="button">
            Duplicate
          </button>
          <button disabled={!selectedPreset || selectedPreset.scope === "builtin" || saving} onClick={archiveSelected} type="button">
            Archive
          </button>
        </div>
      </div>

      <div className="preset-layout">
        <section className="preset-list" aria-label="Recipe presets">
          {recipePresets.length ? (
            recipePresets.map((preset) => (
              <button
                className={selectedPresetId === preset.id ? "preset-row active" : "preset-row"}
                key={`${preset.scope}-${preset.id}`}
                onClick={() => setSelectedPresetId(preset.id)}
                type="button"
              >
                <span>
                  <strong>{preset.name ?? preset.id}</strong>
                  <small>
                    {preset.scope ?? "global"} | {preset.workflow}
                  </small>
                </span>
                <span>{preset.model}</span>
              </button>
            ))
          ) : (
            <div className="empty-panel compact-panel">No presets</div>
          )}
        </section>

        <form className="preset-editor" onSubmit={savePreset}>
          <div className="control-grid compact-controls">
            <label>
              Name
              <input disabled={!editable} onChange={(event) => updateField("name", event.target.value)} required value={form.name} />
            </label>
            <label>
              ID
              <input disabled={Boolean(selectedPreset) || !editable} onChange={(event) => updateField("id", event.target.value)} required value={form.id} />
            </label>
          </div>

          <div className="control-grid">
            <label>
              Scope
              <select disabled={!editable} onChange={(event) => updateField("scope", event.target.value)} value={form.scope}>
                <option value="global">Global</option>
                <option disabled={!activeProject} value="project">
                  Project
                </option>
              </select>
            </label>
            <label>
              Workflow
              <select disabled={!editable} onChange={(event) => updateField("workflow", event.target.value)} value={form.workflow}>
                {workflowOptions.map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Order
              <input disabled={!editable} onChange={(event) => updateField("order", event.target.value)} type="number" value={form.order} />
            </label>
          </div>

          <div className="control-grid compact-controls">
            <label>
              Model
              <select disabled={!editable} onChange={(event) => updateField("model", event.target.value)} value={form.model}>
                {availableModels.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name ?? model.id}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Modes
              <input disabled={!editable} onChange={(event) => updateField("modes", event.target.value)} value={form.modes} />
            </label>
          </div>

          <div className="control-grid">
            <label>
              Count
              <input disabled={!editable} min="1" max="8" onChange={(event) => updateField("count", event.target.value)} type="number" value={form.count} />
            </label>
            <label>
              Resolution
              <input disabled={!editable} onChange={(event) => updateField("resolution", event.target.value)} placeholder="1024x1024" value={form.resolution} />
            </label>
            <label>
              Negative
              <input disabled={!editable} onChange={(event) => updateField("negativePrompt", event.target.value)} value={form.negativePrompt} />
            </label>
          </div>

          <label>
            Description
            <input disabled={!editable} onChange={(event) => updateField("description", event.target.value)} value={form.description} />
          </label>

          <div className="control-grid compact-controls">
            <label>
              Prompt Prefix
              <textarea disabled={!editable} onChange={(event) => updateField("promptPrefix", event.target.value)} value={form.promptPrefix} />
            </label>
            <label>
              Prompt Suffix
              <textarea disabled={!editable} onChange={(event) => updateField("promptSuffix", event.target.value)} value={form.promptSuffix} />
            </label>
          </div>

          <section className="lora-picker" aria-label="Preset LoRAs">
            <div>
              <strong>Managed LoRAs</strong>
              <span>{form.loraIds.length}/3 selected</span>
            </div>
            {compatibleLoras.length ? (
              <div className="lora-choice-list">
                {compatibleLoras.map((lora) => {
                  const checked = form.loraIds.includes(lora.id);
                  return (
                    <label className={checked ? "lora-choice active" : "lora-choice"} key={lora.id}>
                      <input checked={checked} disabled={!editable || (!checked && form.loraIds.length >= 3)} onChange={() => toggleLora(lora.id)} type="checkbox" />
                      <span>
                        <strong>{lora.name ?? lora.id}</strong>
                        <small>{lora.family ?? lora.scope ?? "global"}</small>
                      </span>
                    </label>
                  );
                })}
              </div>
            ) : (
              <div className="empty-panel compact-panel">No managed LoRAs</div>
            )}
          </section>

          {selectedPreset?.scope === "builtin" ? <p className="inline-warning">Built-in presets are read-only.</p> : null}
          {message ? <p className="inline-warning">{message}</p> : null}
          <button className="primary-action" disabled={!editable || saving || !form.name.trim() || !form.model} type="submit">
            {selectedPreset ? "Save Preset" : "Create Preset"}
          </button>
        </form>
      </div>
    </section>
  );
}
