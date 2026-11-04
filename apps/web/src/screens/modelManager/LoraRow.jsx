import React from "react";

export function LoraRow({ lora, cancelEditLora, deleteLora, deletingItem, editingLora, families, loraDownloadJobsFor, loraEditDraft, loraEditError, loraEditKey, loraEditSuggestions, loraHasResolvableFamily, onCancelJob, onDeleteLora, onDownloadLora, onOpenQueue, onUpdateLora, saveEditLora, savingLora, setLoraEditDraft, startEditLora, terminalStatuses, KeywordTagEditor, WorkerProgressCard }) {
    const installed = lora.installState === "installed";
    const missing = lora.installState === "missing";
    const isBuiltin = lora.scope === "builtin";
    // A built-in that isn't downloaded reads "not installed" (neutral) — it can be fetched.
    // A user LoRA with no local files reads "unavailable" (danger) — it's registered but broken.
    const userUnavailable = missing && !isBuiltin;
    const statusText = installed ? "installed" : missing ? (isBuiltin ? "not installed" : "unavailable") : "pending";
    const statusClass = installed ? "status-badge installed" : userUnavailable ? "status-badge danger" : "status-badge";
    const rowClass = userUnavailable ? "lora-row unavailable" : "lora-row";
    const deleteKey = `lora:${lora.scope ?? "global"}:${lora.id}`;
    // Built-in LoRAs with a Hugging Face source can be fetched on demand (sc-5944) —
    // user LoRAs are installed via the import form, so they get no Download affordance.
    const hfSource = (lora.source?.provider ?? lora.provider) === "huggingface";
    const canDownload = Boolean(onDownloadLora) && lora.scope === "builtin" && (!installed || lora.updateAvailable) && hfSource;
    const downloadJob = loraDownloadJobsFor(lora).find((job) => !terminalStatuses.has(job.status));
    const keywords = Array.isArray(lora.triggerWords) ? lora.triggerWords : [];
    const notes = typeof lora.notes === "string" ? lora.notes : "";
    const isEditing = editingLora === loraEditKey(lora);
    // A LoRA whose architecture family couldn't be resolved (no manifest family, no
    // detected signature — e.g. an external ComfyUI adapter in a format we don't yet
    // recognize) is listed so the user sees the file was found, but flagged unusable:
    // the generate-time gate refuses it for every model, so no picker offers it
    // (sc-10509). Built-in catalog LoRAs always declare a family, so this only marks
    // user/external rows.
    const unusable = !isBuiltin && !loraHasResolvableFamily(lora);
    // Built-in entries are read-only (their manifest is compiled in); the backend
    // rejects PATCH on them, so no Edit affordance is offered.
    const canEdit = Boolean(onUpdateLora) && lora.scope !== "builtin";
    // sc-24163: an unresolved family is chosen in place — except on an external (ComfyUI-folder)
    // row, which no manifest backs: the API refuses the write, so it gets an import hint instead.
    const isExternal = lora.scope === "external";
    const canChooseFamily = unusable && !isExternal;
    // What the adapter file itself declared in its safetensors `__metadata__` at import
    // (sc-14057). Each part is shown only when the file stated it — a great many third-party
    // adapters declare no rank/alpha, and showing an inferred value would be a lie about the
    // file. `rank`/`alpha` are numbers; `== null` keeps a legitimate alpha of 0.
    const declared = [
      lora.networkType,
      lora.rank == null ? null : `rank ${lora.rank}`,
      lora.alpha == null ? null : `alpha ${lora.alpha}`,
    ].filter(Boolean);
    return (
      <article className={rowClass} key={lora.id ?? lora.name}>
        <span>
          <strong>{lora.name ?? lora.id}</strong>
          <small>{[lora.scope, unusable ? "unrecognized format" : lora.family ?? "compatible", ...declared].filter(Boolean).join(" | ")}</small>
        </span>
        {unusable ? (
          <span className="status-badge warning">
            unusable
          </span>
        ) : (
          <>
            <span className={statusClass}>{statusText}</span>
            {lora.updateAvailable ? <span className="status-badge warning">update available</span> : null}
          </>
        )}
        {unusable && isExternal ? <small className="lora-family-hint">Import a copy to choose its family.</small> : null}
        <span className="lora-row-actions">
          {canDownload ? (
            <button disabled={Boolean(downloadJob)} onClick={() => onDownloadLora(lora)} type="button">
              {downloadJob ? downloadJob.status : lora.updateAvailable ? "Update" : "Download"}
            </button>
          ) : null}
          {canEdit && !isEditing ? (
            <button onClick={() => startEditLora(lora)} type="button">
              {canChooseFamily ? "Choose family" : "Edit"}
            </button>
          ) : null}
          <button
            className="danger-action"
            disabled={!onDeleteLora || lora.removable === false || deletingItem === deleteKey}
            onClick={() => deleteLora(lora)}
            type="button"
          >
            {lora.removable === false ? "Protected" : deletingItem === deleteKey ? "Deleting" : "Delete"}
          </button>
        </span>
        {!isEditing && (keywords.length || notes) ? (
          <div className="lora-row-meta">
            {keywords.length ? (
              <div className="lora-keywords">
                {keywords.map((keyword) => (
                  <span className="kw-chip" key={keyword}>
                    {keyword}
                  </span>
                ))}
              </div>
            ) : null}
            {notes ? <p className="lora-notes">{notes}</p> : null}
          </div>
        ) : null}
        {isEditing ? (
          <div className="lora-row-editor">
            {canChooseFamily ? (
              <label>
                Unresolved family — choose
                <select
                  disabled={savingLora || !families.length}
                  onChange={(event) => setLoraEditDraft((current) => ({ ...current, family: event.target.value }))}
                  value={loraEditDraft.family}
                >
                  <option value="">Choose family</option>
                  {families.map((family) => (
                    <option key={family} value={family}>
                      {family}
                    </option>
                  ))}
                </select>
              </label>
            ) : null}
            <label>
              Trigger keywords
              <KeywordTagEditor
                disabled={savingLora}
                onChange={(triggerWords) => setLoraEditDraft((current) => ({ ...current, triggerWords }))}
                suggestions={loraEditSuggestions}
                value={loraEditDraft.triggerWords}
              />
            </label>
            <label>
              Notes
              <textarea
                disabled={savingLora}
                onChange={(event) => setLoraEditDraft((current) => ({ ...current, notes: event.target.value }))}
                placeholder="How to use this LoRA — keyword combinations, recommended weight, and other tips."
                rows={2}
                value={loraEditDraft.notes}
              />
            </label>
            <div className="lora-row-editor-actions">
              <button disabled={savingLora} onClick={() => saveEditLora(lora)} type="button">
                {savingLora ? "Saving…" : "Save"}
              </button>
              <button disabled={savingLora} onClick={cancelEditLora} type="button">
                Cancel
              </button>
            </div>
            {loraEditError ? <p className="inline-warning">{loraEditError}</p> : null}
          </div>
        ) : null}
        {downloadJob ? (
          <div className="lora-row-progress">
            <WorkerProgressCard job={downloadJob} onCancel={onCancelJob} onOpenQueue={onOpenQueue} />
          </div>
        ) : null}
      </article>
    );
  }
