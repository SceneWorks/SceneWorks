import React, { useMemo, useState } from "react";
import { Icon } from "../components/Icons.jsx";
import { useAppContext } from "../context/AppContext.js";
import {
  loraHasResolvableFamily,
  loraMatchesModel,
  normalizeLoraFamily,
} from "../presetUtils.js";
import { loraMeta } from "./useSimpleLoras.js";

// The Add LoRA sheet body (epic 15404): the picker list plus a collapsed inline import,
// rendered inside Simple's ONE sheet chrome (`SimpleSheet`).
//
// The LoRA catalog is read LIVE from `useAppContext()` rather than handed in by the studio.
// The sheet descriptor is held in shell state, so a body captured at open time would freeze
// the list — and the one thing that must update while this is open is exactly that list: an
// import queued from the panel below has to appear here when it completes.

const FILE_ACCEPT = ".safetensors,.ckpt,.pt,.bin";

/**
 * @param {object} props
 * @param {object|null} props.selectedModel
 * @param {string[]} props.excludeIds - already-selected ids plus any LoRA the studio applies
 *   itself (Simple's managed `image_edit` LoRA), which must never be offered here.
 * @param {boolean} props.atLimit - at MAX_USER_JOB_LORAS.
 * @param {(lora: object) => void} props.onAdd - adds the LoRA AND closes the sheet.
 * @param {(job: object) => void} [props.onImportQueued]
 */
export function SimpleLoraSheet({ selectedModel, excludeIds = [], atLimit, onAdd, onImportQueued }) {
  const { loras = [], createLoraImportJob, activeProject } = useAppContext();
  const excluded = useMemo(() => new Set(excludeIds), [excludeIds]);

  // The same eligibility predicate `useSimpleLoras` applies, over the live catalog. It is a
  // filter over `presetUtils`, not a re-implementation: `loraMatchesModel` owns family
  // matching and `loraHasResolvableFamily` owns the fail-closed unknown-family rule.
  const availableLoras = useMemo(
    () =>
      loras.filter((lora) => {
        if (lora.presetManaged || lora.installState === "missing" || excluded.has(lora.id)) {
          return false;
        }
        return loraHasResolvableFamily(lora) && loraMatchesModel(lora, selectedModel);
      }),
    [loras, selectedModel, excluded],
  );

  const hasPendingCompatibleLoras =
    Boolean(selectedModel) &&
    loras.some((lora) => lora.installState === "missing" && loraMatchesModel(lora, selectedModel));
  const emptyMessage = !selectedModel
    ? "No model selected"
    : hasPendingCompatibleLoras
      ? "No installed compatible LoRAs. Imports appear after the Queue completes."
      : `No installed LoRAs match ${selectedModel.name ?? selectedModel.id}.`;

  // Nothing to pick ⇒ the import form starts open (sc-15373). The reason the list is empty
  // and the fix for it belong in the same place; a collapsed row would hide the fix behind
  // one more tap at exactly the moment it's needed.
  const [importOpen, setImportOpen] = useState(availableLoras.length === 0);

  return (
    <>
      {availableLoras.length ? (
        <div className="su-lora-pick-list">
          {availableLoras.map((lora) => {
            // Builtins don't count against the user cap (they're auto-applied), so they stay
            // pickable at the limit — the same rule the advanced picker's rows use.
            const disabled = lora.scope !== "builtin" && atLimit;
            return (
              <button
                className="su-lora-pick-row"
                disabled={disabled}
                key={lora.id}
                onClick={() => onAdd(lora)}
                type="button"
              >
                <span className="su-lora-pick-meta">
                  <strong>{lora.name ?? lora.id}</strong>
                  <small>{loraMeta(lora)}</small>
                </span>
                <span className="su-lora-pick-add">{disabled ? "Limit reached" : "Add"}</span>
              </button>
            );
          })}
        </div>
      ) : (
        <p className="su-lora-pick-empty">{emptyMessage}</p>
      )}

      <LoraImportSection
        activeProject={activeProject}
        createLoraImportJob={createLoraImportJob}
        onOpenChange={setImportOpen}
        onQueued={onImportQueued}
        open={importOpen}
      />
    </>
  );
}

// Model Manager's Import LoRA form reduced to the two cases Simple exposes — a file the user
// already has, or a URL. Scope and Family are NOT surfaced: the importer auto-detects the
// family and reports it back on the job, and the scope default (`project` when one is open,
// else `global`) is the one `StudioLoraImportPanel` already uses.
//
// A <div role="group">, not a <form>: this can end up nested inside a screen that is already
// a form, and nested forms are invalid HTML — so the submit rides a click handler.
function LoraImportSection({ open, onOpenChange, activeProject, createLoraImportJob, onQueued }) {
  const [form, setForm] = useState({
    // Upload by default, matching Model Manager: most LoRAs are a file the user already has.
    mode: "file",
    sourceUrl: "",
    file: null,
    name: "",
    triggerKeywords: [],
  });
  const [draftKeyword, setDraftKeyword] = useState("");
  const [importing, setImporting] = useState(false);
  const [message, setMessage] = useState({ tone: "neutral", text: "" });
  // Re-mounts the file input so choosing the SAME file twice still emits a change event.
  const [fileInputKey, setFileInputKey] = useState(0);

  const isFileImport = form.mode === "file";
  const patch = (changes) => setForm((current) => ({ ...current, ...changes }));
  const scope = activeProject ? "project" : "global";
  const importDisabled =
    importing || !createLoraImportJob || (isFileImport ? !form.file : !form.sourceUrl.trim());

  const addKeyword = (raw) => {
    const token = String(raw ?? "").trim();
    setDraftKeyword("");
    if (!token || form.triggerKeywords.some((keyword) => keyword.toLowerCase() === token.toLowerCase())) {
      return;
    }
    patch({ triggerKeywords: [...form.triggerKeywords, token] });
  };

  const removeKeyword = (index) => {
    patch({ triggerKeywords: form.triggerKeywords.filter((_, position) => position !== index) });
  };

  async function submit() {
    if (importDisabled) {
      return;
    }
    setImporting(true);
    setMessage({
      tone: "neutral",
      text: isFileImport ? "Uploading LoRA file before queueing import." : "",
    });
    try {
      const job = await createLoraImportJob({
        ...(isFileImport ? { file: form.file } : { sourceUrl: form.sourceUrl.trim() }),
        name: form.name.trim() || undefined,
        scope,
        triggerWords: form.triggerKeywords,
      });
      const loraId = job?.payload?.loraId;
      // No `family` was sent, so whatever the importer detected is news — report it in the
      // same normalized vocabulary every other surface uses.
      const resolvedFamily = job?.payload?.manifestEntry?.family;
      const detectionNote = resolvedFamily ? ` Detected family: ${normalizeLoraFamily(resolvedFamily)}.` : "";
      patch({ sourceUrl: "", file: null, name: "", triggerKeywords: [] });
      setFileInputKey((current) => current + 1);
      setMessage({
        tone: "success",
        text: `${loraId ? `LoRA import queued for ${loraId}.` : "LoRA import queued."}${detectionNote}`,
      });
      onQueued?.(job);
    } catch (err) {
      // Never swallowed: the import surface is the only place this reason is shown.
      setMessage({ tone: "error", text: err.message });
    } finally {
      setImporting(false);
    }
  }

  return (
    <div aria-label="Import LoRA" className="su-lora-import" role="group">
      <button
        aria-expanded={open}
        className="su-lora-import-trigger"
        onClick={() => onOpenChange(!open)}
        type="button"
      >
        <span aria-hidden="true" className="su-guide-num">
          <Icon.Plus size={14} />
        </span>
        <span className="su-lora-import-label">
          <strong>Import a LoRA</strong>
          <small>From a URL or a file — family and scope are detected</small>
        </span>
        <span className={open ? "su-lora-import-chev open" : "su-lora-import-chev"}>
          <Icon.ChevDown size={14} />
        </span>
      </button>

      {open ? (
        <div className="su-lora-import-panel">
          <div aria-label="LoRA import source" className="su-switch" role="group">
            <button
              className={form.mode === "url" ? "active" : ""}
              disabled={importing}
              onClick={() => patch({ mode: "url" })}
              type="button"
            >
              URL
            </button>
            <button
              className={isFileImport ? "active" : ""}
              disabled={importing}
              onClick={() => patch({ mode: "file" })}
              type="button"
            >
              Upload
            </button>
          </div>

          {isFileImport ? (
            <label className="su-lora-import-row">
              LoRA file
              <span className="su-lora-file">
                <span className="su-lora-file-btn">
                  Choose
                  <input
                    accept={FILE_ACCEPT}
                    disabled={importing}
                    key={fileInputKey}
                    onChange={(event) => patch({ file: event.target.files?.[0] ?? null })}
                    type="file"
                  />
                </span>
                <span className="su-lora-file-name">{form.file?.name ?? "No file selected"}</span>
              </span>
            </label>
          ) : (
            <label className="su-lora-import-row">
              Source URL
              <input
                className="su-input"
                disabled={importing}
                onChange={(event) => patch({ sourceUrl: event.target.value })}
                placeholder="https://..."
                value={form.sourceUrl}
              />
            </label>
          )}

          <label className="su-lora-import-row">
            Name
            <input
              className="su-input"
              disabled={importing}
              onChange={(event) => patch({ name: event.target.value })}
              placeholder="Optional"
              value={form.name}
            />
          </label>

          {/* KeywordTagEditor's behavior in Simple dress: commit on Enter or comma, and
              Backspace on an empty draft removes the last chip. */}
          <label className="su-lora-import-row">
            Trigger keywords
            {/* The wrapping <label> already focuses the draft input on a click anywhere in
                the box, so the chips need no focus handler of their own. */}
            <span className="su-lora-kw">
              {form.triggerKeywords.map((keyword, index) => (
                <span className="su-lora-kw-chip" key={`${keyword}-${index}`}>
                  {keyword}
                  <button
                    aria-label={`Remove ${keyword}`}
                    disabled={importing}
                    onClick={(event) => {
                      event.stopPropagation();
                      removeKeyword(index);
                    }}
                    type="button"
                  >
                    ×
                  </button>
                </span>
              ))}
              <input
                disabled={importing}
                onBlur={() => addKeyword(draftKeyword)}
                onChange={(event) => setDraftKeyword(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === ",") {
                    event.preventDefault();
                    addKeyword(draftKeyword);
                  } else if (event.key === "Backspace" && draftKeyword === "" && form.triggerKeywords.length) {
                    event.preventDefault();
                    removeKeyword(form.triggerKeywords.length - 1);
                  }
                }}
                placeholder={form.triggerKeywords.length ? "" : "e.g. sksStyle — press Enter or comma to add"}
                value={draftKeyword}
              />
            </span>
          </label>

          <button
            className={importing ? "su-accent-btn busy" : "su-accent-btn"}
            disabled={importDisabled}
            onClick={submit}
            type="button"
          >
            {importing ? (isFileImport ? "Uploading" : "Queueing…") : "Queue import"}
          </button>
          {importing ? (
            <div className="su-bar su-bar--indeterminate">
              <span />
            </div>
          ) : null}
          {message.text ? (
            <p className={`su-lora-import-msg su-lora-import-msg--${message.tone}`}>{message.text}</p>
          ) : null}
          <p className="su-lora-import-help">
            Imported LoRAs land in your library and become selectable here once the import job completes — no
            trip to Model Manager.
          </p>
        </div>
      ) : null}
    </div>
  );
}
