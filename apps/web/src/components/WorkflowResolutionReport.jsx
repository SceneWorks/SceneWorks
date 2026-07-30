import React from "react";

import { workflowUnresolved } from "../workflowShare.js";

// "Can this machine actually run this workflow?", rendered (sc-15951, epic 15945).
//
// The presentation half of `crates/sceneworks-core/src/workflow_resolution.rs`. Deliberately a
// component of its own and not part of the drop panel: sc-15952 offers the same recipe from an
// ALREADY-IMPORTED asset and has to say the same things about it, and two surfaces that describe
// the same report differently is exactly how "the model is missing" turns into "the model is
// fine" on one of them.
//
// It states, and never repairs. Every sentence rendered here is one the report already wrote —
// nothing is reworded, nothing is summarized away, and no requirement is dropped for being
// inconvenient. Two of them carry weight beyond their text:
//
// * **A missing model.** The recipe cannot be reproduced here, and the studio will keep whatever
//   model it is already on — which the panel has to SAY, because a prefilled prompt over someone
//   else's model looks like a successful replay.
// * **`omitted`.** A collection the envelope declared but could not record. A recipe whose LoRAs
//   were omitted is not a LoRA-free recipe, and it is indistinguishable from one in the JSON, so
//   this marker is the only thing standing between the user and a confidently wrong replay.

const STATE_LABELS = {
  resolved: "Ready",
  installable: "Not downloaded",
  missing: "Missing",
  userSupplied: "You supply it",
};

const STATE_CLASSES = {
  resolved: "workflow-req-ok",
  installable: "workflow-req-warn",
  missing: "workflow-req-bad",
  userSupplied: "workflow-req-warn",
};

function StateChip({ state }) {
  return (
    <span className={`workflow-req-chip ${STATE_CLASSES[state] ?? "workflow-req-warn"}`}>
      {STATE_LABELS[state] ?? state}
    </span>
  );
}

function Row({ title, state, detail }) {
  return (
    <li className="workflow-req-row">
      <span className="workflow-req-head">
        <strong>{title}</strong>
        <StateChip state={state} />
      </span>
      <span className="workflow-req-detail">{detail}</span>
    </li>
  );
}

export function WorkflowResolutionReport({ report }) {
  if (!report) {
    return (
      <p className="workflow-req-empty">
        This install could not be checked against the recipe, so nothing here says whether it can
        be reproduced.
      </p>
    );
  }
  const unresolved = workflowUnresolved(report);
  const model = report.model ?? null;
  const loras = report.loras ?? [];
  const styles = report.styles ?? [];
  const inputs = report.inputs ?? [];
  const omitted = report.omitted ?? [];
  return (
    <div className="workflow-req">
      <p className={report.runnable ? "workflow-req-verdict ok" : "workflow-req-verdict warn"}>
        {report.runnable
          ? report.inputImagesRequired > 0
            ? `Everything this recipe names is installed here. It still needs ${report.inputImagesRequired} image${report.inputImagesRequired === 1 ? "" : "s"} from you.`
            : "Everything this recipe names is installed here."
          : `${unresolved.length} thing${unresolved.length === 1 ? "" : "s"} in this recipe cannot be reproduced on this install.`}
      </p>
      <ul className="workflow-req-list">
        {model ? (
          <Row
            detail={model.detail}
            state={model.state}
            title={model.name ?? model.slug ?? "Model"}
          />
        ) : null}
        {loras.map((lora, index) => (
          <Row
            detail={lora.detail}
            key={`lora-${lora.catalogId ?? lora.repo ?? lora.name ?? index}`}
            state={lora.state}
            title={`LoRA — ${lora.name ?? lora.repo ?? "unnamed"}`}
          />
        ))}
        {styles.map((style, index) => (
          <Row
            detail={style.detail}
            key={`style-${style.field}-${style.id}-${index}`}
            state={style.state}
            title={`Style — ${style.name ?? style.id}`}
          />
        ))}
        {inputs.map((input, index) => (
          <Row
            detail={input.detail}
            key={`input-${input.kind}-${index}`}
            state={input.state}
            title={`Input image — ${input.kind}`}
          />
        ))}
        {omitted.map((entry) => (
          <Row
            detail={entry.detail}
            key={`omitted-${entry.field}`}
            state="missing"
            title={`Not recorded — ${entry.field}`}
          />
        ))}
      </ul>
    </div>
  );
}
