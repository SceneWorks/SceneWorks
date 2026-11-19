import React, { useEffect, useRef, useState } from "react";
import { isModelMissing } from "../modelMissing.js";
import { joinPromptLines, splitPromptLines } from "../promptBatch.js";
import { Icon } from "./Icons.jsx";

// Run a prompt expander over every prompt of a batch (the batch-mode counterpart of
// QwenRewritePromptControl, used for MiniMax-H3-Image's expander).
//
// The single-prompt control's guarantees carry over: expansion is a user action, nothing in the
// batch changes until the user applies the result, and applying can be undone. The prompts are
// expanded one after another (each is a worker job); Stop keeps what has finished.
//
// Per prompt:
//  * A leading `[WxH]` / `[W:H]` size directive is kept and only the text after it is expanded.
//  * A prompt with `{{…}}` placeholders is left as written: the expander would rewrite the
//    placeholders along with the prose, and the batch would lose its variables.
//  * The expander's proposed size is offered as a `[WxH]` prefix for prompts with no directive of
//    their own, off by default — the same separate opt-in the single-prompt control gives.

const DIRECTIVE = /^\s*\[\s*\d{1,5}\s*[x×X:]\s*\d{1,5}\s*\]\s*/;

function planEntry(entry) {
  if (/\{\{[^{}]*\}\}/.test(entry)) return { skip: true };
  const match = DIRECTIVE.exec(entry);
  const directive = match ? match[0].trim() : "";
  const body = match ? entry.slice(match[0].length).trim() : entry.trim();
  return body ? { skip: false, directive, body } : { skip: true };
}

export function BatchExpandPromptsControl({
  promptsText,
  onPromptsTextChange,
  expandPrompt,
  modelId,
  projectId,
  referenceAssetIds = [],
  expandModel,
  copy,
  suggestedResolutionFor = () => "",
}) {
  const [status, setStatus] = useState("idle"); // idle | running | review | applied
  const [run, setRun] = useState(null); // { text, entries, plans, results, done, total }
  const [useSizes, setUseSizes] = useState(false);
  const [applied, setApplied] = useState(null); // { previous, next, count }
  const [error, setError] = useState("");
  const controllerRef = useRef(null);

  useEffect(() => () => controllerRef.current?.abort(), []);

  const entries = splitPromptLines(promptsText);
  const eligible = entries.filter((entry) => !planEntry(entry).skip).length;
  const skipped = entries.length - eligible;

  async function handleExpand() {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    const plans = entries.map(planEntry);
    const total = plans.filter((plan) => !plan.skip).length;
    const state = { text: promptsText, entries, plans, results: entries.map(() => null), done: 0, total };
    setRun({ ...state });
    setStatus("running");
    setError("");
    setApplied(null);
    for (let index = 0; index < plans.length; index += 1) {
      const plan = plans[index];
      if (plan.skip) continue;
      if (controller.signal.aborted) break;
      try {
        const result = await expandPrompt({
          prompt: plan.body,
          modelId,
          projectId,
          sourceAssetIds: referenceAssetIds,
          signal: controller.signal,
        });
        if (controller.signal.aborted) break;
        state.results[index] = { ok: true, text: String(result?.refinedPrompt ?? "").trim(), suggestion: result?.rewriteSuggestion ?? null };
      } catch (err) {
        if (err?.name === "AbortError" || controller.signal.aborted) break;
        state.results[index] = {
          ok: false,
          error: isModelMissing(expandModel, err?.message) ? copy.missing(false, expandModel?.name || copy.defaultModelName) : err?.message || copy.failure,
        };
      }
      state.done += 1;
      setRun({ ...state, results: [...state.results] });
    }
    if (controllerRef.current === controller) controllerRef.current = null;
    setRun({ ...state, results: [...state.results] });
    setStatus("review");
  }

  function sizeFor(index) {
    const result = run?.results[index];
    return result?.ok && !run.plans[index].directive ? suggestedResolutionFor(result.suggestion) || "" : "";
  }

  function handleApply() {
    if (promptsText !== run.text) {
      setError(copy.batch.changed);
      return;
    }
    let count = 0;
    const next = run.entries.map((entry, index) => {
      const result = run.results[index];
      if (!result?.ok || !result.text) return entry;
      count += 1;
      const size = useSizes ? sizeFor(index) : "";
      const prefix = run.plans[index].directive || (size ? `[${size}]` : "");
      return prefix ? `${prefix} ${result.text}` : result.text;
    });
    const nextText = joinPromptLines(next);
    onPromptsTextChange(nextText);
    setApplied({ previous: run.text, next: nextText, count });
    setRun(null);
    setStatus("applied");
  }

  const results = run?.results ?? [];
  const expanded = results.filter((result) => result?.ok && result.text).length;
  const failures = results
    .map((result, index) => (result && !result.ok ? { item: index + 1, error: result.error } : null))
    .filter(Boolean);
  const anySize = status === "review" && results.some((_, index) => sizeFor(index));

  return (
    <div className="refine-control batch-expand-control">
      {status === "idle" || status === "applied" ? (
        <>
          <button
            className="hero-link refine-button"
            disabled={!eligible || typeof expandPrompt !== "function"}
            onClick={handleExpand}
            type="button"
          >
            <Icon.Wand size={14} /> {copy.batch.button(eligible)}
          </button>
          <p className="refine-hint">
            {copy.batch.hint}
            {skipped ? ` ${copy.batch.skipped(skipped)}` : ""}
          </p>
        </>
      ) : null}

      {status === "applied" && applied ? (
        <p className="refine-status" role="status">
          {copy.batch.applied(applied.count)}{" "}
          {promptsText === applied.next ? (
            <button
              className="secondary-action"
              onClick={() => {
                onPromptsTextChange(applied.previous);
                setApplied(null);
                setStatus("idle");
              }}
              type="button"
            >
              Undo
            </button>
          ) : null}
        </p>
      ) : null}

      {status === "running" && run ? (
        <p className="refine-status" role="status">
          <Icon.Wand size={14} /> {copy.batch.progress(Math.min(run.done + 1, run.total), run.total)}{" "}
          <button className="secondary-action" onClick={() => controllerRef.current?.abort()} type="button">
            Stop
          </button>
        </p>
      ) : null}

      {status === "review" && run ? (
        <div className="refine-review batch-expand-review">
          <p className="refine-review-label" role="status">
            {copy.batch.summary(expanded, run.total)}
          </p>
          {failures.length ? (
            <ul className="refine-error batch-expand-failures" role="alert">
              {failures.map(({ item, error: message }) => (
                <li key={item}>
                  Prompt {item}: {message}
                </li>
              ))}
            </ul>
          ) : null}
          {expanded ? (
            <details className="batch-expand-preview">
              <summary>{copy.batch.reviewLabel}</summary>
              <ol>
                {results.map((result, index) =>
                  result?.ok && result.text ? (
                    <li key={index} value={index + 1}>
                      {run.plans[index].directive ? <code>{run.plans[index].directive}</code> : null} {result.text}
                    </li>
                  ) : null,
                )}
              </ol>
            </details>
          ) : null}
          {anySize ? (
            <label className="batch-expand-sizes">
              <input checked={useSizes} onChange={(event) => setUseSizes(event.target.checked)} type="checkbox" />{" "}
              {copy.batch.useSizes}
            </label>
          ) : null}
          {error ? (
            <p className="refine-error" role="alert">
              {error}
            </p>
          ) : null}
          <div className="refine-review-actions">
            <button className="secondary-action" disabled={!expanded} onClick={handleApply} type="button">
              {copy.batch.apply}
            </button>
            <button
              className="secondary-action"
              onClick={() => {
                setRun(null);
                setError("");
                setStatus("idle");
              }}
              type="button"
            >
              Discard
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
