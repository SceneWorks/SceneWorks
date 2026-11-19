import React, { useEffect, useRef, useState } from "react";
import { isModelMissing } from "../modelMissing.js";
import {
  cardinality,
  extractKeys,
  iterateBatch,
  joinPromptLines,
  linkedGroupIssues,
  missingKeys,
  splitPromptLines,
} from "../promptBatch.js";
import { Icon } from "./Icons.jsx";

// Run a prompt expander over every prompt of a batch (the batch-mode counterpart of
// QwenRewritePromptControl, used for MiniMax-H3-Image's expander).
//
// The single-prompt control's guarantees carry over: expansion is a user action, nothing in the
// batch changes until the user applies the result, and applying can be undone. The prompts are
// expanded one after another (each is a worker job) BEFORE any image renders, so the image model
// stays warm across the batch; Stop keeps what has finished.
//
// Variables are filled in first. `{{name}}` values, inline `{{a|b}}` choices and linked groups are
// resolved exactly as the batch run resolves them, and each resulting prompt is expanded on its
// own — every combination gets its own caption. Applying therefore replaces the templates with one
// line per resolved prompt (expanded, or as resolved when its expansion failed or was not reached),
// so the batch still renders the same set of prompts; Undo restores the templates.
//
// Per prompt, a leading `[WxH]` / `[W:H]` size directive is kept and only the text after it is
// expanded. The expander's proposed size is offered as a `[WxH]` prefix for prompts with no
// directive of their own, off by default — the same separate opt-in the single-prompt control gives.

const DIRECTIVE = /^\s*\[\s*\d{1,5}\s*[x×X:]\s*\d{1,5}\s*\]\s*/;
// The batch's placeholder syntax (promptBatch.js `keyPattern`, non-blank inner text).
const PLACEHOLDER = /\{\{[^{}]*\S[^{}]*\}\}/;
// Past this many resolved prompts the expander asks before starting (each is a worker job).
export const BATCH_EXPAND_CONFIRM_AT = 100;
// More than this is refused: the user should narrow the variables first.
export const BATCH_EXPAND_MAX = 2000;

function planEntry(entry) {
  const match = DIRECTIVE.exec(entry);
  const directive = match ? match[0].trim() : "";
  const body = match ? entry.slice(match[0].length).trim() : entry.trim();
  return body ? { skip: false, directive, body } : { skip: true };
}

function batchVariables(prompts, variableValues) {
  return extractKeys(prompts).map((key) => ({ key, values: variableValues?.[key] ?? [] }));
}

export function BatchExpandPromptsControl({
  promptsText,
  // The batch's `{{name}}` values ({ key: [values] }), as the batch panel edits them.
  variableValues = {},
  onPromptsTextChange,
  expandPrompt,
  modelId,
  projectId,
  referenceAssetIds = [],
  expandModel,
  copy,
  suggestedResolutionFor = () => "",
}) {
  const [status, setStatus] = useState("idle"); // idle | confirm | running | review | applied
  const [run, setRun] = useState(null); // { text, entries, plans, results, done, total }
  const [useSizes, setUseSizes] = useState(false);
  const [applied, setApplied] = useState(null); // { previous, next, count }
  const [error, setError] = useState("");
  const controllerRef = useRef(null);

  useEffect(() => () => controllerRef.current?.abort(), []);

  const prompts = splitPromptLines(promptsText);
  const variables = batchVariables(prompts, variableValues);
  const variablesKey = JSON.stringify(variables);
  const templated = prompts.some((prompt) => /\{\{[^{}]*\S[^{}]*\}\}/.test(prompt));
  const missing = missingKeys(prompts, variables);
  const groupIssues = linkedGroupIssues(prompts);
  const total = cardinality(prompts, variables, 1);
  const blocked = missing.length
    ? copy.batch.missing(missing)
    : groupIssues.length
      ? copy.batch.groups
      : total > BATCH_EXPAND_MAX
        ? copy.batch.tooMany(total, BATCH_EXPAND_MAX)
        : "";

  function handleStart() {
    if (total > BATCH_EXPAND_CONFIRM_AT && status !== "confirm") {
      setStatus("confirm");
      return;
    }
    handleExpand();
  }

  async function handleExpand() {
    // Admission is re-read here, not only on the first click: the batch list and variables stay
    // editable while the confirmation is open, and the product must never be materialised past
    // the maximum (or with an unfilled key / mismatched group).
    if (blocked || !total) {
      setStatus("idle");
      return;
    }
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    const entries = [...iterateBatch(prompts, variables)].map(({ prompt }) => prompt);
    const plans = entries.map(planEntry);
    const runTotal = plans.filter((plan) => !plan.skip).length;
    const state = { text: promptsText, variablesKey, entries, plans, results: entries.map(() => null), done: 0, total: runTotal };
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
    if (promptsText !== run.text || variablesKey !== run.variablesKey) {
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
    // The batch list has no escape for `{{…}}`: a written line that contains placeholder syntax
    // (from a variable's value or from the expansion itself) would be read as a template again and
    // change what the batch renders. Refuse and name those prompts rather than alter the batch.
    const reparsed = next
      .map((line, index) => (PLACEHOLDER.test(line) ? index + 1 : null))
      .filter((item) => item !== null);
    if (reparsed.length) {
      setError(copy.batch.literalPlaceholders(reparsed));
      return;
    }
    // An empty resolved choice (`{{|a fox}}`) is a prompt of its own in the batch, but the list
    // cannot hold an empty line; writing it would silently drop that item.
    const empty = next
      .map((line, index) => (line.trim() === "" ? index + 1 : null))
      .filter((item) => item !== null);
    if (empty.length) {
      setError(copy.batch.emptyPrompts(empty));
      return;
    }
    const nextText = joinPromptLines(next);
    // The written list must read back as exactly these prompts. A prompt that is (or contains) a
    // standalone `---` line would split or merge prompts in block mode, so the batch would render
    // a different set; refuse rather than alter it.
    const expected = next.map((line) => line.trim());
    const readBack = splitPromptLines(nextText);
    if (readBack.length !== expected.length || readBack.some((line, index) => line !== expected[index])) {
      const delimiters = next
        .map((line, index) => (line.split(/\r?\n/).some((part) => part.trim() === "---") ? index + 1 : null))
        .filter((item) => item !== null);
      setError(copy.batch.unrepresentable(delimiters));
      return;
    }
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
            disabled={!total || Boolean(blocked) || typeof expandPrompt !== "function"}
            onClick={handleStart}
            type="button"
          >
            <Icon.Wand size={14} /> {copy.batch.button(total)}
          </button>
          <p className="refine-hint">
            {copy.batch.hint}
            {templated && total && !blocked ? ` ${copy.batch.resolved(total)}` : ""}
          </p>
          {blocked ? (
            <p className="refine-error" role="alert">
              {blocked}
            </p>
          ) : null}
        </>
      ) : null}

      {status === "confirm" ? (
        <div className="batch-confirm" role="alertdialog">
          <span>{blocked || copy.batch.confirm(total)}</span>
          <button className="secondary-action" disabled={Boolean(blocked) || !total} onClick={handleExpand} type="button">
            {copy.batch.button(total)}
          </button>
          <button className="secondary-action" onClick={() => setStatus("idle")} type="button">
            Cancel
          </button>
        </div>
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
