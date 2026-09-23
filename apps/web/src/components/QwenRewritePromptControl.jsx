import React, { useEffect, useRef, useState } from "react";
import { Icon } from "./Icons.jsx";

// Qwen-Image 2.1's OFFICIAL prompt rewriting (sc-24113, epic 24107).
//
// # Why this is not `RefinePromptControl`
//
// The generic refiner and this share a shape — ask a local LLM, review, apply — and differ in the
// three things this story's acceptance criteria are about:
//
//  1. The rewrite is EDITABLE. `RefinePromptControl` renders its suggestion in a `<p>`: take it or
//     leave it. Qwen's rewriter emits a long, dense paragraph that a user will almost always want
//     to adjust, so the suggestion lands in a textarea they can change before applying.
//  2. It carries an explicit ASPECT-RATIO suggestion alongside the prose, which the user accepts or
//     ignores INDEPENDENTLY of the prompt. Accepting it selects one of the model's own seven
//     presets — the same value the Aspect menu offers — never a computed size.
//  3. Which rewriter runs is decided by the REQUEST (references present ⇒ the editing checkpoint),
//     so there is no picker here and the two halves are separate optional downloads.
//
// # The guarantees this component is responsible for
//
// * The user's prompt is NEVER replaced silently. Nothing leaves this component until Apply, and
//   "Keep original" is always available beside it.
// * Rewriting is a USER ACTION. There is no `autoStart`: the panel opens idle with a button. (The
//   generic refiner has an autoStart mode for its prompt-tool tile; deliberately not mirrored,
//   because an automatic rewrite is exactly what the story forbids.)
// * With no rewriter installed there is no degraded path — the affordance is not rendered at all
//   (the caller gates on `installed`), and this component's missing-model branch exists only for
//   the race where a checkpoint is uninstalled between render and click.

// Humanize a byte count to a "18.8 GB" label; null when the size is unknown.
function formatGb(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return null;
  }
  return `${(bytes / 1e9).toFixed(1)} GB`;
}

// The worker fast-fails with "…snapshot is not cached…" when the checkpoint is absent. Prefer the
// catalog install state when the entry is supplied; fall back to the message otherwise.
function isModelMissing(rewriteModel, errorMessage) {
  if (rewriteModel?.installState) {
    return rewriteModel.installState === "missing";
  }
  return /not cached|not installed|snapshot is not/i.test(errorMessage ?? "");
}

export function QwenRewritePromptControl({
  prompt,
  modelId,
  projectId,
  // The ORDERED reference asset ids the render will use. Their presence selects the editing
  // rewriter over the text-to-image one, and their ORDER is what the rewrite's `<imageN>` numbering
  // refers to — so this must be the same list, in the same order, the Generate button will send.
  referenceAssetIds = [],
  rewritePrompt,
  onApply,
  // Accept the aspect suggestion: called with a "2752x1536"-style preset the caller can hand
  // straight to its resolution control. Optional — a caller with no resolution control (the Image
  // Editor, whose geometry comes from the working image) simply omits it and the offer is hidden.
  onApplyResolution,
  // The catalog entry for whichever rewriter this request will select, plus its downloader.
  rewriteModel,
  onDownloadRewriteModel,
}) {
  const [status, setStatus] = useState("idle"); // idle | loading | review | error
  // The suggestion, held as EDITABLE state. Seeded from the model's reply and then owned by the
  // user's keystrokes until they apply or discard it.
  const [draft, setDraft] = useState("");
  const [suggestion, setSuggestion] = useState(null);
  const [error, setError] = useState("");
  const [downloadRequested, setDownloadRequested] = useState(false);
  const controllerRef = useRef(null);

  const trimmed = (prompt ?? "").trim();
  const busy = status === "loading";
  const disabled = busy || !trimmed || typeof rewritePrompt !== "function";
  const modelMissing = status === "error" && isModelMissing(rewriteModel, error);
  const installState = rewriteModel?.installState;
  const editing = referenceAssetIds.length > 0;

  // When the checkpoint finishes downloading (a catalog refresh flips installState), clear the
  // missing-model error so the user can retry without reopening the panel.
  useEffect(() => {
    if (installState === "installed" && downloadRequested) {
      setDownloadRequested(false);
      setStatus((current) => (current === "error" ? "idle" : current));
      setError((current) => (current ? "" : current));
    }
  }, [installState, downloadRequested]);

  useEffect(
    () => () => {
      controllerRef.current?.abort();
    },
    [],
  );

  async function handleRewrite() {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    const { signal } = controller;
    setStatus("loading");
    setError("");
    try {
      const result = await rewritePrompt({
        prompt: trimmed,
        modelId,
        projectId,
        sourceAssetIds: referenceAssetIds,
        signal,
      });
      if (signal.aborted) {
        return;
      }
      setDraft(result?.refinedPrompt ?? "");
      setSuggestion(result?.rewriteSuggestion ?? null);
      setStatus("review");
    } catch (err) {
      if (err?.name === "AbortError") {
        return;
      }
      setError(err?.message || "Prompt rewriting failed.");
      setStatus("error");
    } finally {
      if (controllerRef.current === controller) {
        controllerRef.current = null;
      }
    }
  }

  async function handleDownloadModel() {
    if (typeof onDownloadRewriteModel !== "function") return;
    try {
      const job = await onDownloadRewriteModel();
      if (job) {
        setDownloadRequested(true);
      }
    } catch (err) {
      setError(err?.message || "Could not start the rewriter download.");
    }
  }

  const sizeLabel = formatGb(rewriteModel?.downloadSizeBytes);
  const modelName = rewriteModel?.name || "Qwen Image 2.1 Prompt Rewriter";
  // Offer the aspect change only when the reply actually named a preset. An edit rewrite that set
  // `ratioFollow` instead has no ratio to offer, and an unrecognised ratio was dropped worker-side
  // rather than guessed at — in both cases the prose is still worth having.
  const suggestedResolution = suggestion?.resolution || "";
  const ratioFollow = suggestion?.ratioFollow || "";
  const followsIndex = Number.isInteger(suggestion?.followsReferenceIndex)
    ? suggestion.followsReferenceIndex
    : null;

  return (
    <div className="refine-control qwen-rewrite-control">
      {status === "loading" || status === "review" ? null : (
        <button
          className="hero-link refine-button"
          disabled={disabled}
          onClick={handleRewrite}
          type="button"
        >
          <Icon.Wand size={14} />{" "}
          {editing ? "Rewrite edit instruction" : "Rewrite prompt"}
        </button>
      )}
      {status === "idle" ? (
        <p className="refine-hint">
          {editing
            ? `Qwen’s official editing rewriter reads your ${referenceAssetIds.length} reference${
                referenceAssetIds.length === 1 ? "" : "s"
              } in order and rewrites the instruction. You edit and accept the result — your prompt is not changed on its own.`
            : "Qwen’s official rewriter turns a short brief into a long descriptive prompt and suggests an aspect ratio. You edit and accept the result — your prompt is not changed on its own."}
        </p>
      ) : null}
      {busy ? (
        <p className="refine-status">
          <Icon.Wand size={14} /> Rewriting…
        </p>
      ) : null}

      {status === "error" && modelMissing ? (
        <div className="refine-missing-model" role="alert">
          {downloadRequested ? (
            <p className="refine-error">
              Downloading the rewriter… track progress on the Models screen, then try again.
            </p>
          ) : (
            <>
              <p className="refine-error">
                The {editing ? "editing" : "text-to-image"} rewriter
                {sizeLabel ? ` (${sizeLabel})` : ""} isn’t installed yet. Qwen Image 2.1 generates
                normally without it.
              </p>
              {typeof onDownloadRewriteModel === "function" ? (
                <button className="secondary-action" onClick={handleDownloadModel} type="button">
                  Download rewriter
                </button>
              ) : (
                <p className="refine-error">Open the Models screen to download “{modelName}”.</p>
              )}
            </>
          )}
        </div>
      ) : status === "error" ? (
        <p className="refine-error" role="alert">
          {error}
        </p>
      ) : null}

      {status === "review" ? (
        <div className="refine-review qwen-rewrite-review">
          <label className="refine-review-label" htmlFor="qwen-rewrite-draft">
            Rewritten prompt — edit before applying
          </label>
          {/* EDITABLE, unlike the generic refiner's read-only paragraph. This is the control the
              acceptance criteria name: the rewrite is a starting point the user adjusts, and their
              original prompt is untouched behind this panel until they press Apply. */}
          <textarea
            className="refine-review-text qwen-rewrite-draft"
            id="qwen-rewrite-draft"
            onChange={(event) => setDraft(event.target.value)}
            rows={8}
            value={draft}
          />
          {suggestedResolution ? (
            <div className="qwen-rewrite-aspect">
              <p className="qwen-rewrite-aspect-label">
                Suggested aspect ratio: {suggestion?.whRatio}{" "}
                <span className="qwen-rewrite-aspect-size">({suggestedResolution})</span>
              </p>
              {/* A SEPARATE accept. The prompt and the geometry are two decisions, and a user who
                  wants the rewrite at their own aspect ratio must not have to take both. */}
              {typeof onApplyResolution === "function" ? (
                <button
                  className="secondary-action"
                  onClick={() => onApplyResolution(suggestedResolution)}
                  type="button"
                >
                  Use this aspect ratio
                </button>
              ) : null}
            </div>
          ) : ratioFollow ? (
            <p className="qwen-rewrite-aspect-label">
              The rewriter expects the output to match reference{" "}
              {followsIndex == null ? ratioFollow : `image ${followsIndex + 1}`} rather than a new
              aspect ratio.
            </p>
          ) : null}
          <div className="refine-review-actions">
            <button
              className="secondary-action"
              disabled={!draft.trim()}
              onClick={() => {
                onApply(draft);
                setStatus("idle");
              }}
              type="button"
            >
              Apply
            </button>
            <button className="secondary-action" onClick={() => setStatus("idle")} type="button">
              Keep original
            </button>
            <button className="secondary-action" onClick={handleRewrite} type="button">
              Rewrite again
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
