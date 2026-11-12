import { readFile } from "node:fs/promises";
import { readFileSync } from "node:fs";
import path from "node:path";

export const MEMORY_EVIDENCE_PATH = "config/manifests/registry/evidence.json";

const IDENTITY_FIELDS = ["provider", "repo", "revision", "variant", "files"];
const MEASURED_FOOTPRINT_FIELDS = [
  "residentMemoryBytes",
  "peakMemoryBytes",
  "measuredPixels",
];

function object(value, label) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  return value;
}

function downloadIdentity(row) {
  return JSON.stringify(IDENTITY_FIELDS.map((field) =>
    Object.hasOwn(row, field) ? [field, row[field]] : [field]));
}

function evidenceIdentity(row) {
  return JSON.stringify([downloadIdentity(row), Object.hasOwn(row, "platforms") ? row.platforms : null]);
}

/**
 * Restore measured facts to a parsed product manifest for legacy script consumers.
 * The committed evidence store owns these facts; the product manifest owns the
 * download identity and disk size. This projection never changes either input.
 */
export function projectBuiltinMemoryEvidence(manifest, evidence) {
  object(manifest, "builtin manifest");
  object(evidence, "memory evidence");
  if (!Array.isArray(manifest.models)) throw new Error("builtin manifest models must be an array");

  const projected = structuredClone(manifest);
  const models = new Map();
  for (const model of projected.models) {
    if (typeof model?.id !== "string" || models.has(model.id)) {
      throw new Error(`duplicate or missing builtin model id: ${model?.id}`);
    }
    models.set(model.id, model);
  }

  for (const [modelId, raw] of Object.entries(evidence)) {
    const model = models.get(modelId);
    if (!model) throw new Error(`memory evidence names unknown builtin model ${modelId}`);
    const facts = object(raw, `memory evidence ${modelId}`);
    for (const field of ["model", "mlx", "candle"]) {
      if (!Object.hasOwn(facts, field)) continue;
      const measured = object(facts[field], `${modelId}.${field}`);
      const target = field === "model" ? model : model[field];
      if (field !== "model" && !target) {
        throw new Error(`memory evidence names absent ${modelId}.${field} backend`);
      }
      Object.assign(target, structuredClone(measured));
    }

    if (!Object.hasOwn(facts, "downloads")) continue;
    if (!Array.isArray(facts.downloads)) {
      throw new Error(`${modelId}.downloads memory evidence must be an array`);
    }
    const downloads = model.downloads ?? [];
    const available = new Map();
    for (const download of downloads) {
      const key = downloadIdentity(download);
      available.set(key, [...(available.get(key) ?? []), download]);
    }
    const seen = new Set();
    const claimed = new Set();
    for (const row of facts.downloads) {
      object(row, `${modelId}.downloads evidence row`);
      const key = downloadIdentity(row);
      const evidenceKey = evidenceIdentity(row);
      if (seen.has(evidenceKey)) throw new Error(`duplicate memory evidence download for ${modelId}: ${evidenceKey}`);
      seen.add(evidenceKey);
      const matches = (available.get(key) ?? []).filter((download) =>
        !Object.hasOwn(row, "platforms") ||
        JSON.stringify(download.platforms) === JSON.stringify(row.platforms));
      if (matches.length !== 1) {
        throw new Error(`memory evidence download for ${modelId} matches ${matches.length} product rows: ${key}`);
      }
      if (claimed.has(matches[0])) {
        throw new Error(`multiple memory evidence downloads match one product row for ${modelId}: ${key}`);
      }
      claimed.add(matches[0]);
      const footprint = object(row.footprint, `${modelId}.downloads footprint`);
      for (const field of Object.keys(footprint)) {
        if (!MEASURED_FOOTPRINT_FIELDS.includes(field)) {
          throw new Error(`memory evidence cannot own product download field ${field}`);
        }
      }
      const target = matches[0];
      target.footprint ??= {};
      for (const field of MEASURED_FOOTPRINT_FIELDS) delete target.footprint[field];
      Object.assign(target.footprint, structuredClone(footprint));
    }
  }
  return projected;
}

export async function readBuiltinMemoryEvidence(root) {
  return JSON.parse(await readFile(path.join(root, MEMORY_EVIDENCE_PATH), "utf8"));
}

export function readBuiltinMemoryEvidenceSync(root) {
  return JSON.parse(readFileSync(path.join(root, MEMORY_EVIDENCE_PATH), "utf8"));
}
