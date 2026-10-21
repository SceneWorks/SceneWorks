import React, { useState } from "react";

import { errorMessage } from "../../errorMessage.js";
import { IRIS_SIDECAR_KINDS, itemSidecarState, matchSidecarFiles } from "../../training/irisTaskOptions.js";

// Iris depth / restoration sidecars (sc-25687 / sc-25688): per saved dataset item, an aligned depth
// map (+ validity mask) for depth training or an aligned degraded input for restoration training.
// A collapsed disclosure (busy page), rendered only when opened. Per-item pickers plus a bulk
// "attach by filename" picker per kind that matches files to items by stem.

const ACCEPT = { depth: ".npy,.png", mask: ".npy,.png", degraded: ".png,.jpg,.jpeg,.webp" };
const LABEL = { depth: "Depth map", mask: "Mask", degraded: "Degraded" };
const DEPTH_FIELDS = [
  ["scale", "Depth scale (e.g. 0.001 for mm PNG)"],
  ["minDepth", "Min valid depth"],
  ["maxDepth", "Max valid depth"],
];

export function IrisSidecarPanel({ items = [], onSidecar }) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState({ message: "", error: "" });
  const [depthFields, setDepthFields] = useState({});
  const saved = items.filter((item) => item?.id);

  async function apply(jobs, unmatched = []) {
    setBusy(true);
    setStatus({ message: "", error: "" });
    const errors = [];
    let done = 0;
    for (const { item, kind, file } of jobs) {
      try {
        await onSidecar(item, kind, file, file && kind === "depth" ? depthFields : undefined);
        done += 1;
      } catch (error) {
        errors.push(`${item.displayName ?? item.id}: ${errorMessage(error, "failed")}`);
      }
    }
    setBusy(false);
    setStatus({
      message: `${done} sidecar change${done === 1 ? "" : "s"} applied.${unmatched.length ? ` No item matched: ${unmatched.join(", ")}.` : ""}`,
      error: errors.join(" · "),
    });
  }

  const picker = (kind, label, onFiles, multiple = false, aria = label) => (
    <label className="secondary-action" key={kind}>
      {label}
      <input
        accept={ACCEPT[kind]}
        aria-label={aria}
        disabled={busy || !onSidecar}
        hidden
        multiple={multiple}
        onChange={(event) => {
          const files = [...(event.target.files ?? [])];
          event.target.value = "";
          if (files.length) onFiles(files);
        }}
        type="file"
      />
    </label>
  );

  return (
    <details className="training-iris-options" onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary>Iris sidecars (depth / restoration training)</summary>
      {open ? (
        <>
          <p className="training-field-hint">
            Depth/mask: aligned .npy or grayscale PNG. Degraded: image at 1× or an integer downscale. Bulk files match by
            name (photo.npy, photo_depth.png, photo_mask.png).
            {onSidecar ? "" : " Save the dataset first."}
          </p>
          <div className="training-advanced-grid">
            {DEPTH_FIELDS.map(([key, label]) => (
              <label key={key}>
                {label}
                <input
                  onChange={(event) => setDepthFields({ ...depthFields, [key]: event.target.value })}
                  step="any"
                  type="number"
                  value={depthFields[key] ?? ""}
                />
              </label>
            ))}
          </div>
          <div className="training-caption-card-actions">
            {IRIS_SIDECAR_KINDS.map((kind) =>
              picker(kind, `Attach ${LABEL[kind].toLowerCase()}s by filename`, (files) => {
                const { matches, unmatched } = matchSidecarFiles(saved, files, kind);
                apply(matches.map(({ item, file }) => ({ item, kind, file })), unmatched);
              }, true),
            )}
          </div>
          {status.message ? <p className="inline-success">{status.message}</p> : null}
          {status.error ? <p className="inline-warning">{status.error}</p> : null}
          <ul aria-label="Iris sidecars per item">
            {saved.map((item) => {
              const state = itemSidecarState(item);
              const name = item.displayName ?? item.id;
              return (
                <li key={item.id}>
                  <strong>{name}</strong>{" "}
                  {IRIS_SIDECAR_KINDS.map((kind) => (
                    <React.Fragment key={kind}>
                      {picker(kind, `${LABEL[kind]}${state[kind] ? " ✓" : ""}`, ([file]) => apply([{ item, kind, file }]), false, `${LABEL[kind]} for ${name}`)}
                      {state[kind] ? (
                        <button
                          aria-label={`Remove ${LABEL[kind].toLowerCase()} of ${name}`}
                          className="secondary-action"
                          disabled={busy}
                          onClick={() => apply([{ item, kind, file: null }])}
                          type="button"
                        >
                          ×
                        </button>
                      ) : null}
                    </React.Fragment>
                  ))}
                </li>
              );
            })}
          </ul>
        </>
      ) : null}
    </details>
  );
}
