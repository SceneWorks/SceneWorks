import React from "react";
import { formatBytes } from "../formatting.js";

// Installation details for a model assembled from separately licensed, revision-pinned components
// (sc-24216, epic 24212 — Supra2-IMG's denoiser, Flan-T5-Base text encoder and SD VAE decoder).
//
// The licence row above it (`ModelLicenseSummary`) carries the whole-model summary; this lists what
// actually gets installed: each download row's role, repository, pinned revision, size and its own
// licence, the total the install fetches, and the limits the engine supports (resolution, images
// per job, prompt length). Busy-card convention: one DEFAULT-COLLAPSED `<details>` so a card that
// is mostly about install state is not buried under a component table.
//
// Rendered only for an entry that declares something here to say — a per-download `license` or an
// `image.maxPromptTokens` — so every other model card is unchanged.

function componentRole(download) {
  if (download?.coRequisite !== true) {
    return "Model weights";
  }
  switch (download.componentId) {
    case "text_encoder":
      return "Text encoder";
    case "vae":
      return "VAE decoder";
    default:
      return download.componentId ?? "Component";
  }
}

function downloadBytes(download) {
  return download?.expectedSizeBytes ?? download?.estimatedSizeBytes ?? download?.footprint?.diskSizeBytes ?? null;
}

export function modelComponentRows(model) {
  return (Array.isArray(model?.downloads) ? model.downloads : []).filter(
    (download) => download && typeof download.repo === "string",
  );
}

export function hasModelComponentDetails(model) {
  return (
    modelComponentRows(model).some((download) => typeof download.license === "string") ||
    Number.isInteger(model?.image?.maxPromptTokens)
  );
}

export function modelLimitLines(model) {
  const lines = [];
  const resolutions = Array.isArray(model?.limits?.resolutions) ? model.limits.resolutions : [];
  if (resolutions.length) {
    lines.push(
      `${resolutions.length === 1 ? "Native resolution" : "Resolutions"}: ${resolutions
        .map((value) => String(value).replace("x", "×"))
        .join(", ")}`,
    );
  }
  const counts = Array.isArray(model?.limits?.count) ? model.limits.count : [];
  if (counts.length) {
    lines.push(`Images per job: ${counts.join(", ")}`);
  }
  if (Number.isInteger(model?.image?.maxPromptTokens)) {
    lines.push(
      `Prompt: up to ${model.image.maxPromptTokens} tokens (longer prompts are truncated and flagged on the result)`,
    );
  }
  return lines;
}

export function ModelComponentDetails({ model, totalSizeText }) {
  if (!hasModelComponentDetails(model)) {
    return null;
  }
  const rows = modelComponentRows(model);
  const limits = modelLimitLines(model);
  return (
    <details className="model-component-details">
      <summary>Components and limits</summary>
      {rows.length ? (
        <table className="model-component-table">
          <thead>
            <tr>
              <th scope="col">Component</th>
              <th scope="col">Source</th>
              <th scope="col">Size</th>
              <th scope="col">License</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((download) => (
              <tr key={`${download.repo}:${download.componentId ?? "primary"}`}>
                <td>{componentRole(download)}</td>
                <td>
                  <span className="model-component-repo">{download.repo}</span>
                  {download.revision ? (
                    <span className="model-component-revision" title={download.revision}>
                      {" "}
                      @ {download.revision.slice(0, 12)}
                    </span>
                  ) : null}
                </td>
                <td>{formatBytes(downloadBytes(download))}</td>
                <td>{download.license ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
      {totalSizeText ? <p className="model-component-total">Total download: {totalSizeText}</p> : null}
      {limits.length ? (
        <ul className="model-component-limits" aria-label="Supported limits">
          {limits.map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>
      ) : null}
    </details>
  );
}
