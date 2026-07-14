#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const root = process.cwd();
const manifestPath = path.join(root, "config/manifests/builtin.models.jsonc");
const enginesPath = path.join(root, "crates/sceneworks-worker/src/engines.rs");

function stripJsoncComments(body) {
  let result = "";
  let inString = false;
  let escaped = false;

  for (let index = 0; index < body.length; index += 1) {
    const char = body[index];
    const next = body[index + 1];

    if (inString) {
      result += char;
      if (escaped) {
        escaped = false;
      } else if (char === "\\") {
        escaped = true;
      } else if (char === '"') {
        inString = false;
      }
      continue;
    }

    if (char === '"') {
      inString = true;
      result += char;
      continue;
    }

    if (char === "/" && next === "/") {
      while (index < body.length && body[index] !== "\n") {
        index += 1;
      }
      result += "\n";
      continue;
    }

    if (char === "/" && next === "*") {
      index += 2;
      while (
        index < body.length &&
        !(body[index] === "*" && body[index + 1] === "/")
      ) {
        index += 1;
      }
      index += 1;
      continue;
    }

    result += char;
  }

  return result;
}

function sha256(body) {
  return createHash("sha256").update(body).digest("hex");
}

function modelSnapshot(model) {
  const downloads = Array.isArray(model.downloads) ? model.downloads : [];
  const variants = [
    ...new Set(
      downloads
        .map((download) => download?.variant)
        .filter((variant) => typeof variant === "string"),
    ),
  ].sort();

  return {
    id: model.id,
    family: model.family ?? null,
    type: model.type ?? null,
    adapter: model.adapter ?? null,
    capabilities: Array.isArray(model.capabilities)
      ? [...model.capabilities].sort()
      : [],
    routes: {
      mlx: model.mlx != null,
      candle: model.candle != null,
    },
    variants,
  };
}

function engineRows(source) {
  const rows = [];
  let sceneworksId = null;

  for (const line of source.split("\n")) {
    const sceneMatch = line.match(/^\s*sceneworks_id:\s*"([^"]+)"/);
    if (sceneMatch) {
      sceneworksId = sceneMatch[1];
      continue;
    }

    const engineMatch = line.match(/^\s*engine_id:\s*"([^"]+)"/);
    if (engineMatch && sceneworksId != null) {
      rows.push({
        sceneworksId,
        engineId: engineMatch[1],
      });
      sceneworksId = null;
    }
  }

  return rows;
}

const [manifestBody, enginesBody] = await Promise.all([
  readFile(manifestPath, "utf8"),
  readFile(enginesPath, "utf8"),
]);

const manifest = JSON.parse(stripJsoncComments(manifestBody));
const models = manifest.models.map(modelSnapshot).sort((left, right) =>
  left.id.localeCompare(right.id),
);
const rows = engineRows(enginesBody).sort((left, right) =>
  left.sceneworksId.localeCompare(right.sceneworksId),
);

const output = {
  schemaVersion: 1,
  sources: {
    builtinModels: {
      path: "config/manifests/builtin.models.jsonc",
      sha256: sha256(manifestBody),
      schemaVersion: manifest.schemaVersion,
    },
    workerEngineTable: {
      path: "crates/sceneworks-worker/src/engines.rs",
      sha256: sha256(enginesBody),
    },
  },
  counts: {
    models: models.length,
    mlxRoutedModels: models.filter((model) => model.routes.mlx).length,
    candleRoutedModels: models.filter((model) => model.routes.candle).length,
    workerEngineRows: rows.length,
  },
  modelSignatures: models.map(
    (model) =>
      [
        model.id,
        model.family ?? "-",
        model.type ?? "-",
        model.adapter ?? "-",
        model.capabilities.join(",") || "-",
        `mlx=${model.routes.mlx}`,
        `candle=${model.routes.candle}`,
        `variants=${model.variants.join(",") || "-"}`,
      ].join("|"),
  ),
  workerEngineRows: rows.map(
    (row) => `${row.sceneworksId}=>${row.engineId}`,
  ),
};

const rendered = `${JSON.stringify(output, null, 2)}\n`;
const checkIndex = process.argv.indexOf("--check");

if (checkIndex >= 0) {
  const expectedPath = process.argv[checkIndex + 1];
  if (expectedPath == null) {
    throw new Error("--check requires a snapshot path");
  }

  const expected = await readFile(path.resolve(root, expectedPath), "utf8");
  if (expected !== rendered) {
    process.stderr.write(
      `catalog baseline drift: regenerate ${expectedPath} and review the change\n`,
    );
    process.exitCode = 1;
  } else {
    process.stdout.write(`catalog baseline matches ${expectedPath}\n`);
  }
} else {
  process.stdout.write(rendered);
}
