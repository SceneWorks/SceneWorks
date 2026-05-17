import React, { useEffect, useMemo, useState } from "react";
import { AssetCard } from "../components/assetPanels.jsx";
import { AssetMedia } from "../components/assetMedia.jsx";

const characterTypes = [
  ["person", "Person"],
  ["creature", "Creature"],
  ["object", "Object"],
];

function typeLabel(value) {
  return characterTypes.find(([id]) => id === value)?.[1] ?? "Person";
}

function compatibleFamilies(item) {
  const compatibility = item?.compatibility ?? {};
  const values =
    item?.families ??
    item?.compatibleFamilies ??
    item?.modelFamilies ??
    compatibility.families ??
    (item?.family ? [item.family] : []);
  return Array.isArray(values) ? values : [values].filter(Boolean);
}

function summarizeCompatibility(item) {
  const values = compatibleFamilies(item);
  return values.length ? values.join(", ") : "Unspecified";
}

function editableLora(link) {
  return {
    name: link?.name ?? "",
    triggerWords: (link?.triggerWords ?? []).join(", "),
    defaultWeight: link?.defaultWeight ?? 0.8,
    families: compatibleFamilies(link).join(", "),
    scope: link?.scope ?? "project",
  };
}

export function CharacterStudio({
  activeProject,
  assets,
  characters,
  createCharacter,
  updateCharacter,
  archiveCharacter,
  addCharacterReference,
  updateCharacterReference,
  removeCharacterReference,
  createCharacterLook,
  updateCharacterLook,
  deleteCharacterLook,
  attachCharacterLora,
  updateCharacterLora,
  detachCharacterLora,
  createCharacterTestJob,
  deleteAsset,
  purgeAsset,
  imageModels,
  latestAssets,
  loras,
  onPreview,
  onSendImage,
  onSendVideo,
  updateAssetStatus,
}) {
  const [selectedCharacterId, setSelectedCharacterId] = useState(characters[0]?.id ?? "");
  const [draft, setDraft] = useState({ name: "", type: "person", description: "" });
  const [newCharacter, setNewCharacter] = useState({ name: "", type: "person", description: "" });
  const [referenceAssetId, setReferenceAssetId] = useState("");
  const [lookDraft, setLookDraft] = useState({ name: "", description: "" });
  const [selectedReferenceIds, setSelectedReferenceIds] = useState([]);
  const [loraId, setLoraId] = useState("");
  const [loraEdits, setLoraEdits] = useState({});
  const [testPrompt, setTestPrompt] = useState("A clean character reference portrait, consistent identity, studio lighting");
  const [testModel, setTestModel] = useState(imageModels[0]?.id ?? "z_image_turbo");
  const [testLookId, setTestLookId] = useState("");

  const imageAssets = useMemo(
    () => assets.filter((asset) => ["image", "frame", "upload"].includes(asset.type)),
    [assets],
  );
  const selectedCharacter = characters.find((item) => item.id === selectedCharacterId) ?? characters[0] ?? null;
  const approvedReferences = selectedCharacter?.approvedReferences ?? [];
  const selectedLook = selectedCharacter?.looks?.find((look) => look.id === testLookId) ?? null;

  useEffect(() => {
    if (!selectedCharacter && characters[0]?.id) {
      setSelectedCharacterId(characters[0].id);
    }
  }, [characters, selectedCharacter]);

  useEffect(() => {
    if (!selectedCharacter) {
      setDraft({ name: "", type: "person", description: "" });
      return;
    }
    setDraft({
      name: selectedCharacter.name ?? "",
      type: selectedCharacter.type ?? "person",
      description: selectedCharacter.description ?? "",
    });
    setSelectedReferenceIds((ids) =>
      ids.filter((id) => selectedCharacter.approvedReferences?.some((reference) => reference.assetId === id)),
    );
    setLoraEdits(
      Object.fromEntries((selectedCharacter.loras ?? []).map((link) => [link.id, editableLora(link)])),
    );
    if (testLookId && !selectedCharacter.looks?.some((look) => look.id === testLookId)) {
      setTestLookId("");
    }
  }, [selectedCharacter?.id, selectedCharacter?.updatedAt]);

  useEffect(() => {
    if (!imageModels.some((item) => item.id === testModel)) {
      setTestModel(imageModels[0]?.id ?? "z_image_turbo");
    }
  }, [imageModels, testModel]);

  async function submitNewCharacter(event) {
    event.preventDefault();
    const created = await createCharacter(newCharacter);
    if (created) {
      setSelectedCharacterId(created.id);
      setNewCharacter({ name: "", type: "person", description: "" });
    }
  }

  async function saveCharacter(event) {
    event.preventDefault();
    if (selectedCharacter) {
      await updateCharacter(selectedCharacter.id, draft);
    }
  }

  async function submitReference(event) {
    event.preventDefault();
    if (selectedCharacter && referenceAssetId) {
      await addCharacterReference(selectedCharacter.id, { assetId: referenceAssetId, approved: false });
      setReferenceAssetId("");
    }
  }

  async function submitLook(event) {
    event.preventDefault();
    if (!selectedCharacter || !lookDraft.name.trim()) {
      return;
    }
    await createCharacterLook(selectedCharacter.id, {
      name: lookDraft.name,
      description: lookDraft.description,
      approvedReferenceIds: selectedReferenceIds,
      recipeSettings: selectedLook?.recipeSettings ?? {},
    });
    setLookDraft({ name: "", description: "" });
    setSelectedReferenceIds([]);
  }

  async function submitLora(event) {
    event.preventDefault();
    if (!selectedCharacter || !loraId) {
      return;
    }
    const lora = loras.find((item) => item.id === loraId);
    await attachCharacterLora(selectedCharacter.id, {
      loraId: lora.id,
      name: lora.name ?? lora.id,
      triggerWords: lora.triggerWords ?? [],
      defaultWeight: lora.defaultWeight ?? 0.8,
      compatibility: { families: compatibleFamilies(lora) },
      scope: "global",
    });
    setLoraId("");
  }

  async function saveLora(link) {
    const edit = loraEdits[link.id] ?? editableLora(link);
    await updateCharacterLora(selectedCharacter.id, link.id, {
      name: edit.name,
      triggerWords: edit.triggerWords
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean),
      defaultWeight: Number(edit.defaultWeight),
      compatibility: {
        ...(link.compatibility ?? {}),
        families: edit.families
          .split(",")
          .map((item) => item.trim())
          .filter(Boolean),
      },
      scope: edit.scope,
    });
  }

  function setLoraEdit(linkId, key, value) {
    setLoraEdits((items) => ({
      ...items,
      [linkId]: {
        ...(items[linkId] ?? {}),
        [key]: value,
      },
    }));
  }

  async function submitTest(event) {
    event.preventDefault();
    if (!selectedCharacter) {
      return;
    }
    await createCharacterTestJob(selectedCharacter.id, {
      prompt: testPrompt,
      model: testModel,
      count: 4,
      width: 1024,
      height: 1024,
      lookId: testLookId || null,
    });
  }

  return (
    <section className="main-surface character-studio">
      <div className="surface-header">
        <div className="section-heading">
          <p className="eyebrow">Character Studio</p>
          <h2>{activeProject ? activeProject.name : "Create a project"}</h2>
        </div>
        <form className="inline-create" onSubmit={submitNewCharacter}>
          <input
            aria-label="Character name"
            onChange={(event) => setNewCharacter((item) => ({ ...item, name: event.target.value }))}
            placeholder="New character"
            value={newCharacter.name}
          />
          <select
            aria-label="Character type"
            onChange={(event) => setNewCharacter((item) => ({ ...item, type: event.target.value }))}
            value={newCharacter.type}
          >
            {characterTypes.map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
          <button disabled={!activeProject || !newCharacter.name.trim()} type="submit">
            Create
          </button>
        </form>
      </div>

      {!selectedCharacter ? (
        <div className="empty-panel">No characters yet</div>
      ) : (
        <div className="character-layout">
          <aside className="character-list">
            {characters.map((character) => (
              <button
                className={character.id === selectedCharacter.id ? "character-row active" : "character-row"}
                key={character.id}
                onClick={() => setSelectedCharacterId(character.id)}
                type="button"
              >
                <strong>{character.name}</strong>
                <span>{typeLabel(character.type)}</span>
                <small>{character.references?.length ?? 0} refs</small>
              </button>
            ))}
          </aside>

          <section className="character-detail">
            <form className="character-editor" onSubmit={saveCharacter}>
              <div className="control-grid">
                <label>
                  Name
                  <input onChange={(event) => setDraft((item) => ({ ...item, name: event.target.value }))} value={draft.name} />
                </label>
                <label>
                  Type
                  <select onChange={(event) => setDraft((item) => ({ ...item, type: event.target.value }))} value={draft.type}>
                    {characterTypes.map(([value, label]) => (
                      <option key={value} value={value}>
                        {label}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <label className="prompt-field">
                Notes
                <textarea
                  onChange={(event) => setDraft((item) => ({ ...item, description: event.target.value }))}
                  value={draft.description}
                />
              </label>
              <div className="detail-actions">
                <button type="submit">Save</button>
                <button onClick={() => archiveCharacter(selectedCharacter.id)} type="button">
                  Archive
                </button>
                <button onClick={() => onSendImage(selectedCharacter, testLookId || null)} type="button">
                  Image
                </button>
                <button onClick={() => onSendVideo(selectedCharacter, testLookId || null)} type="button">
                  Video
                </button>
              </div>
            </form>

            <section className="character-section">
              <div className="section-heading">
                <p className="eyebrow">References</p>
                <h2>Approved set</h2>
              </div>
              <form className="inline-create" onSubmit={submitReference}>
                <select onChange={(event) => setReferenceAssetId(event.target.value)} value={referenceAssetId}>
                  <option value="">Add image or frame</option>
                  {imageAssets.map((asset) => (
                    <option key={asset.id} value={asset.id}>
                      {asset.displayName}
                    </option>
                  ))}
                </select>
                <button disabled={!referenceAssetId} type="submit">
                  Add
                </button>
              </form>
              <div className="character-reference-grid">
                {(selectedCharacter.references ?? []).map((reference) => (
                  <article className={reference.approved ? "reference-card approved" : "reference-card"} key={reference.assetId}>
                    <button className="reference-media" onClick={() => reference.asset && onPreview(reference.asset)} type="button">
                      {reference.asset ? <AssetMedia asset={reference.asset} /> : <span>Missing asset</span>}
                    </button>
                    <div>
                      <strong>{reference.asset?.displayName ?? reference.assetId}</strong>
                      <span>{reference.role}</span>
                    </div>
                    <div className="review-actions">
                      <button
                        className={reference.approved ? "active" : ""}
                        onClick={() =>
                          updateCharacterReference(selectedCharacter.id, reference.assetId, { approved: !reference.approved })
                        }
                        type="button"
                      >
                        {reference.approved ? "Approved" : "Approve"}
                      </button>
                      <button onClick={() => removeCharacterReference(selectedCharacter.id, reference.assetId)} type="button">
                        Remove
                      </button>
                    </div>
                  </article>
                ))}
                {selectedCharacter.references?.length ? null : <div className="empty-panel compact-panel">No references</div>}
              </div>
            </section>

            <section className="character-section">
              <div className="section-heading">
                <p className="eyebrow">Looks</p>
                <h2>Saved recipes</h2>
              </div>
              <form className="look-composer" onSubmit={submitLook}>
                <input
                  aria-label="Look name"
                  onChange={(event) => setLookDraft((item) => ({ ...item, name: event.target.value }))}
                  placeholder="Look name"
                  value={lookDraft.name}
                />
                <input
                  aria-label="Look notes"
                  onChange={(event) => setLookDraft((item) => ({ ...item, description: event.target.value }))}
                  placeholder="Notes"
                  value={lookDraft.description}
                />
                <button disabled={!lookDraft.name.trim()} type="submit">
                  Save Look
                </button>
                <div className="reference-checks">
                  {approvedReferences.map((reference) => (
                    <label className="checkline" key={reference.assetId}>
                      <input
                        checked={selectedReferenceIds.includes(reference.assetId)}
                        onChange={(event) =>
                          setSelectedReferenceIds((ids) =>
                            event.target.checked
                              ? [...ids, reference.assetId]
                              : ids.filter((id) => id !== reference.assetId),
                          )
                        }
                        type="checkbox"
                      />
                      {reference.asset?.displayName ?? reference.assetId}
                    </label>
                  ))}
                </div>
              </form>
              <div className="look-list">
                {(selectedCharacter.looks ?? []).map((look) => (
                  <article className="look-row" key={look.id}>
                    <div>
                      <strong>{look.name}</strong>
                      <span>{look.description || "No notes"}</span>
                      <small>{look.approvedReferenceIds?.length ?? 0} approved refs</small>
                    </div>
                    <div className="review-actions">
                      <button onClick={() => setTestLookId(look.id)} type="button">
                        Select
                      </button>
                      <button
                        onClick={() =>
                          updateCharacterLook(selectedCharacter.id, look.id, {
                            ...look,
                            recipeSettings: { ...(look.recipeSettings ?? {}), touchedAt: new Date().toISOString() },
                          })
                        }
                        type="button"
                      >
                        Refresh
                      </button>
                      <button onClick={() => deleteCharacterLook(selectedCharacter.id, look.id)} type="button">
                        Delete
                      </button>
                    </div>
                  </article>
                ))}
                {selectedCharacter.looks?.length ? null : <div className="empty-panel compact-panel">No looks</div>}
              </div>
            </section>

            <section className="character-section">
              <div className="section-heading">
                <p className="eyebrow">LoRAs</p>
                <h2>Character adapters</h2>
              </div>
              <form className="inline-create" onSubmit={submitLora}>
                <select onChange={(event) => setLoraId(event.target.value)} value={loraId}>
                  <option value="">Attach imported LoRA</option>
                  {loras.map((lora) => (
                    <option key={lora.id} value={lora.id}>
                      {lora.name}
                    </option>
                  ))}
                </select>
                <button disabled={!loraId} type="submit">
                  Attach
                </button>
              </form>
              <div className="lora-editor-list">
                {(selectedCharacter.loras ?? []).map((link) => {
                  const edit = loraEdits[link.id] ?? editableLora(link);
                  return (
                    <article className="lora-editor" key={link.id}>
                      <div className="lora-editor-head">
                        <strong>{link.name}</strong>
                        <span>{link.copiedIntoProject ? "Project copy" : link.scope}</span>
                      </div>
                      <div className="control-grid compact-controls">
                        <label>
                          Name
                          <input onChange={(event) => setLoraEdit(link.id, "name", event.target.value)} value={edit.name} />
                        </label>
                        <label>
                          Families
                          <input onChange={(event) => setLoraEdit(link.id, "families", event.target.value)} value={edit.families} />
                        </label>
                        <label>
                          Triggers
                          <input
                            onChange={(event) => setLoraEdit(link.id, "triggerWords", event.target.value)}
                            value={edit.triggerWords}
                          />
                        </label>
                        <label>
                          Weight
                          <input
                            max="2"
                            min="-2"
                            onChange={(event) => setLoraEdit(link.id, "defaultWeight", event.target.value)}
                            step="0.05"
                            type="number"
                            value={edit.defaultWeight}
                          />
                        </label>
                      </div>
                      <small>Compatibility: {summarizeCompatibility(link)}</small>
                      <div className="review-actions">
                        <button onClick={() => saveLora(link)} type="button">
                          Save
                        </button>
                        <button onClick={() => detachCharacterLora(selectedCharacter.id, link.id)} type="button">
                          Detach
                        </button>
                      </div>
                    </article>
                  );
                })}
                {selectedCharacter.loras?.length ? null : <div className="empty-panel compact-panel">No linked LoRAs</div>}
              </div>
            </section>

            <section className="character-section test-character-panel">
              <div className="section-heading">
                <p className="eyebrow">Test Character</p>
                <h2>Sample outputs</h2>
              </div>
              <form className="test-character-form" onSubmit={submitTest}>
                <label className="prompt-field">
                  Prompt
                  <textarea onChange={(event) => setTestPrompt(event.target.value)} value={testPrompt} />
                </label>
                <div className="control-grid">
                  <label>
                    Look
                    <select onChange={(event) => setTestLookId(event.target.value)} value={testLookId}>
                      <option value="">Character defaults</option>
                      {(selectedCharacter.looks ?? []).map((look) => (
                        <option key={look.id} value={look.id}>
                          {look.name}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Model
                    <select onChange={(event) => setTestModel(event.target.value)} value={testModel}>
                      {imageModels.map((model) => (
                        <option key={model.id} value={model.id}>
                          {model.name}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
                <button className="primary-action" disabled={!testPrompt.trim()} type="submit">
                  Test Character
                </button>
              </form>
              <div className="review-grid">
                {latestAssets.map((asset) => (
                  <div className="test-result" key={asset.id}>
                    <AssetCard
                      asset={asset}
                      deleteAsset={deleteAsset}
                      onPreview={onPreview}
                      purgeAsset={purgeAsset}
                      updateAssetStatus={updateAssetStatus}
                    />
                    <button
                      onClick={() => addCharacterReference(selectedCharacter.id, { assetId: asset.id, approved: true, role: "test-output" })}
                      type="button"
                    >
                      Approve as Reference
                    </button>
                  </div>
                ))}
                {latestAssets.length ? null : <div className="empty-panel compact-panel">No test outputs yet</div>}
              </div>
            </section>
          </section>
        </div>
      )}
    </section>
  );
}
