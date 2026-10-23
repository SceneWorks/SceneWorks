import React from "react";

// The imported-checkpoint product surface (sc-26687) on a Model Manager card: what was imported
// (source variant), what it can do (supported operations), where it can run (supported backends),
// whether its components are assembled (component readiness), and what to do about the ones that
// are not (actionable repair steps).
//
// Every value comes from the server's ONE derivation
// (`apps/rust-api/src/models.rs::apply_imported_checkpoint_product_surface`), so the repair this
// card offers is the same repair the API describes and the queue gate requires. Nothing here
// re-derives a verdict from a path, a filename or a repository name — an imported checkpoint's
// identity is its contents and its declared contract, not its spelling.
//
// Busy-card convention, same as `ModelComponentDetails`: one default-collapsed `<details>`, so a
// card that is mostly about install state is not buried under a table. Renders NOTHING for a
// builtin entry or a model with no import behind it.

const OPERATION_LABELS = {
  text_to_video: "Text → Video",
  image_to_video: "Image → Video",
  first_last_frame: "First → Last Frame",
  reference_to_video: "Reference → Video",
};

const BACKEND_LABELS = {
  mlx: "MLX / Metal",
  "candle-cuda": "Candle / CUDA",
};

const COMPONENT_LABELS = {
  transformer: "Imported DiT",
  transformer_ref: "Reference DiT",
  text_encoder: "Text encoder",
  tokenizer: "Tokenizer",
  vae: "Video VAE",
  video_vae: "Video VAE",
  audio_vae: "Audio VAE",
};

const STATE_LABELS = {
  installed: "Ready",
  incomplete: "Incomplete",
  missing: "Missing",
};

const STATE_BADGE = {
  installed: "status-badge installed",
  incomplete: "status-badge incomplete",
  missing: "status-badge",
};

// Only a repair the Model Manager can actually perform gets a button. `select-base` is a
// declaration the user makes in the checkpoint import panel, so it renders as a sentence naming
// where that happens rather than as a button that would do nothing.
const ACTIONABLE_REPAIR_KINDS = new Set(["download", "relink", "rescan"]);

export function hasImportedCheckpointSurface(model) {
  return Boolean(model?.importedSource);
}

export function componentLabel(componentId) {
  return COMPONENT_LABELS[componentId] ?? componentId ?? "Component";
}

export function operationLabel(operation) {
  return OPERATION_LABELS[operation] ?? operation;
}

export function backendLabel(backend) {
  return BACKEND_LABELS[backend] ?? backend;
}

// The one-line summary shown before the card is opened, so the row reads "what this is" without
// the user expanding anything. Split out because the focused behaviour tests assert the sentence a
// user reads rather than a field list.
export function importedSourceSummary(model) {
  const source = model?.importedSource ?? {};
  const parts = [];
  if (source.sourceVariant) {
    parts.push(String(source.sourceVariant));
  }
  if (source.componentTier) {
    parts.push(String(source.componentTier));
  }
  if (source.ownershipMode) {
    parts.push(source.ownershipMode === "linked" ? "linked source" : "managed install");
  }
  return parts.join(" · ");
}

export function ImportedCheckpointSurface({ model, onRepair }) {
  if (!hasImportedCheckpointSurface(model)) {
    return null;
  }
  const source = model.importedSource;
  const readiness = model.componentReadiness ?? {};
  const components = Array.isArray(readiness.components) ? readiness.components : [];
  const operations = Array.isArray(model.supportedOperations) ? model.supportedOperations : [];
  // `null` (unjudged) is NOT the same as `[]` (both lanes looked, neither will claim it): the
  // server omits the field when no per-lane verdict exists, and this card must not turn that into
  // a refusal nobody made.
  const backends = Array.isArray(model.supportedBackends) ? model.supportedBackends : null;
  const repairs = Array.isArray(model.repairActions) ? model.repairActions : [];

  return (
    <details className="model-component-details">
      <summary>Imported checkpoint</summary>
      <p className="imported-checkpoint-summary">{importedSourceSummary(model)}</p>
      <dl className="imported-checkpoint-source">
        <dt>Source variant</dt>
        <dd>{source.sourceVariant ?? "unknown"}</dd>
        <dt>Source shape</dt>
        <dd>
          {source.sourceShape ?? "unknown"}
          {source.sharded ? " (sharded)" : ""}
        </dd>
        <dt>Declared partition</dt>
        <dd>{source.declaredPartition ?? "not declared"}</dd>
        <dt>Compatible base</dt>
        <dd>{source.baseModel ?? "not declared"}</dd>
      </dl>
      <p className="imported-checkpoint-operations">
        <strong>Operations:</strong>{" "}
        {operations.length ? operations.map(operationLabel).join(", ") : "none declared"}
      </p>
      {backends ? (
        <p className="imported-checkpoint-backends">
          <strong>Backends:</strong>{" "}
          {backends.length ? backends.map(backendLabel).join(", ") : "none available here"}
        </p>
      ) : null}
      {components.length ? (
        <table className="model-component-table" aria-label="Component readiness">
          <thead>
            <tr>
              <th scope="col">Component</th>
              <th scope="col">Source</th>
              <th scope="col">State</th>
            </tr>
          </thead>
          <tbody>
            {components.map((component) => {
              const state = component.state ?? (component.present ? "installed" : "missing");
              return (
                <tr key={`${component.source ?? "plan"}:${component.componentId ?? "component"}`}>
                  <td>{componentLabel(component.componentId)}</td>
                  <td>{component.model ?? "imported checkpoint"}</td>
                  <td>
                    <span className={STATE_BADGE[state] ?? "status-badge"}>
                      {STATE_LABELS[state] ?? state}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      ) : null}
      {repairs.length ? (
        <ul className="imported-checkpoint-repairs" aria-label="Repair steps">
          {repairs.map((repair) => {
            const actionable =
              typeof onRepair === "function" && ACTIONABLE_REPAIR_KINDS.has(repair.kind);
            return (
              <li key={repair.id ?? repair.label}>
                {actionable ? (
                  <button type="button" onClick={() => onRepair(repair)}>
                    {repair.label}
                  </button>
                ) : (
                  <>
                    {repair.label}
                    {repair.kind === "select-base"
                      ? " (declare it in the checkpoint import panel)"
                      : ""}
                  </>
                )}
              </li>
            );
          })}
        </ul>
      ) : (
        <p className="imported-checkpoint-repairs">
          Nothing to repair: every component is assembled.
        </p>
      )}
    </details>
  );
}
