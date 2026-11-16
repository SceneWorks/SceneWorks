import React, { useRef, useState } from "react";

export function completedTrainingSteps(job) {
  const recorded = job?.result?.stepsCompleted;
  return Number.isInteger(recorded) && recorded >= 0 ? recorded : job?.payload?.plan?.config?.steps;
}

export function canContinueTraining(job) {
  return job?.type === "lora_train" && job.status === "completed" &&
    job.payload?.dryRun === false && job.result?.dryRun !== true &&
    job.payload?.plan?.target?.outputKind === "lora" &&
    Number.isInteger(completedTrainingSteps(job)) && completedTrainingSteps(job) > 0;
}

export function ContinueTrainingAction({ job, onContinue }) {
  const [expanded, setExpanded] = useState(false);
  const [additionalSteps, setAdditionalSteps] = useState("1000");
  const [submitting, setSubmitting] = useState(false);
  const pending = useRef(false);
  const [error, setError] = useState("");
  const initialSteps = completedTrainingSteps(job);
  const additional = Number(additionalSteps);
  const total = initialSteps + additional;
  const valid = additionalSteps.trim() !== "" && Number.isSafeInteger(additional) && additional > 0 && total <= 4294967295;

  async function submit(event) {
    event.preventDefault();
    if (!valid || pending.current) return;
    pending.current = true;
    setSubmitting(true);
    setError("");
    try {
      const result = await onContinue(job, additional);
      if (result) setExpanded(false);
    } catch (cause) {
      setError(cause.message || "Could not continue training.");
    } finally {
      pending.current = false;
      setSubmitting(false);
    }
  }

  return expanded ? (
    <form className="training-continue-form" onSubmit={submit} aria-label="Continue training">
      <label>
        Additional steps
        <input type="number" min="1" step="1" value={additionalSteps} disabled={submitting}
          onChange={(event) => setAdditionalSteps(event.target.value)} />
      </label>
      <p>{valid ? `${initialSteps.toLocaleString()} → ${total.toLocaleString()} total steps` : "Enter a positive whole number of additional steps."}</p>
      <p>A new linked run preserves the original LoRA. The learning-rate schedule uses the extended total.</p>
      {error ? <p role="alert">{error}</p> : null}
      <button className="primary-action" type="submit" disabled={!valid || submitting}>
        {submitting ? "Queuing…" : "Start continued run"}
      </button>
      <button className="secondary-action" type="button" disabled={submitting} onClick={() => setExpanded(false)}>Cancel</button>
    </form>
  ) : (
    <button className="secondary-action" type="button" onClick={() => setExpanded(true)}>Continue training</button>
  );
}
