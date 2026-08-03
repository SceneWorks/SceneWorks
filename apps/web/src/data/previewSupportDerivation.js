// Stage 2 of the engine-capability pipeline (sc-16965, epic 16948): the PURE derivation shared by
// the generator (apps/web/scripts/generate-preview-support.mjs) and its drift guard
// (previewSupportCatalog.test.js), so the committed artifacts and the test derive from one
// implementation — the parseStyleCatalog.js / styleManifest.js shape.
//
// ## What this derives
//
// `Capabilities.supports_preview` is set per ENGINE, on the provider descriptor. SceneWorks models
// are a different namespace, and the map is many-to-one: `z_image_turbo` backs both `z_image_turbo`
// and `z_image_edit`; `sdxl` backs six ids; `qwen_image_edit` backs four. The join is the static
// `MODEL_TABLE` in `crates/sceneworks-worker/src/engines.rs`, which we parse here rather than
// mirror — a checked-in text file, exactly like `documents/style.txt` is for the style catalog. That
// means adding a MODEL_TABLE row makes the committed artifacts stale on the SAME PR that adds it,
// and the drift guard says so, instead of the row silently inheriting "unknown" forever.
//
// ## Why the result is engine-KEYED and not a boolean
//
// `supports_streaming` is a property of the model, so the audio catalog ships one flag. Preview
// support is a property of **the engine that runs it**: during epic 16948's rollout `krea_2_turbo`
// is `true` on MLX/macOS and `false` on candle/Windows — same model id, same app version, two
// answers. It does not fully collapse when the epic completes either: SenseNova-U1 is candle-only
// and MLX never wired it (sc-16960), so at least one route stays split permanently. A single
// shipped boolean is wrong in every ordering, so the derived value is a `{ backend: bool }` map.
//
// ## Absence is UNKNOWN, never false
//
// A backend key is emitted only when that backend's facts file actually contains the engine id. A
// backend that never registered the engine (no candle Mage, no MLX SenseNova), or a backend whose
// facts file has not been dumped on this checkout, produces NO key — the consumer reads that as
// "unknown" and renders exactly as it did before this story. Under-claiming is invisible;
// over-claiming would leave a placeholder up forever on a route that never emits.

/** The artifact schema version, bumped when the derived shape changes incompatibly. */
export const PREVIEW_SUPPORT_VERSION = 1;

/** Named in the artifacts so a reader knows the one sanctioned way to update them. */
export const PREVIEW_SUPPORT_GENERATOR = "apps/web/scripts/generate-preview-support.mjs";

// The `MODEL_TABLE` const in engines.rs. Anchored on the type so a rename is a loud parse failure
// rather than a silently empty table.
const MODEL_TABLE_OPEN = "const MODEL_TABLE: &[ModelRow] = &[";

// Belt-and-braces only. The real guard is the exact `ModelRow`-occurrence count in
// {@link parseEngineModelTable} — the floor exists solely to catch an anchor that matched something
// tiny and structurally wrong. Kept just under the current row count (53) rather than at the old
// 40, whose 13 rows of slack grew with the table and let a truncated parse through.
const MIN_EXPECTED_MODEL_ROWS = 50;

/**
 * Strip Rust comments, leaving string literals and line structure intact.
 *
 * Comment handling is the difference between "this model no longer exists" and "this model
 * over-claims live preview". A row commented OUT with a block-comment span is not a row, and a
 * line-based `//` filter cannot see that; a `];` inside a comment is not the end of the table.
 * Both are handled here, once, before anything else looks at the text. Block comments nest in
 * Rust, so the depth counter is not decoration.
 *
 * String literals are copied verbatim (escapes honoured) so a `//` inside a repo URL — every
 * `default_repo` is one — can never be mistaken for a comment.
 */
function stripRustComments(source) {
  let out = "";
  let index = 0;
  let blockDepth = 0; // Rust block comments nest.
  while (index < source.length) {
    if (blockDepth > 0) {
      if (source.startsWith("/*", index)) {
        blockDepth += 1;
        index += 2;
      } else if (source.startsWith("*/", index)) {
        blockDepth -= 1;
        index += 2;
      } else {
        // Keep newlines so the caller's line-oriented error messages still line up.
        out += source[index] === "\n" ? "\n" : " ";
        index += 1;
      }
      continue;
    }
    if (source.startsWith("/*", index)) {
      blockDepth = 1;
      index += 2;
      continue;
    }
    if (source.startsWith("//", index)) {
      while (index < source.length && source[index] !== "\n") index += 1;
      continue;
    }
    if (source[index] === '"') {
      out += source[index];
      index += 1;
      while (index < source.length) {
        const char = source[index];
        out += char;
        index += 1;
        if (char === "\\") {
          if (index < source.length) {
            out += source[index];
            index += 1;
          }
          continue;
        }
        if (char === '"') break;
      }
      continue;
    }
    out += source[index];
    index += 1;
  }
  if (blockDepth > 0) {
    throw new Error("parseEngineModelTable: engines.rs has an unterminated `/*` block comment");
  }
  return out;
}

/**
 * The text between `MODEL_TABLE`'s opening `&[` and its MATCHING `]`.
 *
 * Bracket-matched rather than terminated on the first `\n];`, because that sequence can occur
 * inside the table (a nested literal, a re-indent) and truncating there drops every row after it —
 * which read downstream as a pile of legitimate "unknown"s. Comments are already gone by the time
 * this runs; string literals are skipped so a bracket inside one cannot close the table.
 */
function sliceModelTableBody(source) {
  const open = source.indexOf(MODEL_TABLE_OPEN);
  if (open === -1) {
    throw new Error(
      `parseEngineModelTable: no ${JSON.stringify(MODEL_TABLE_OPEN)} in engines.rs — the model ` +
        "table was renamed or restructured; update this parser and re-run `npm run gen:preview-support`",
    );
  }
  const bodyStart = open + MODEL_TABLE_OPEN.length;
  let depth = 1; // the `[` the anchor ends on
  let index = bodyStart;
  while (index < source.length) {
    const char = source[index];
    if (char === '"') {
      index += 1;
      while (index < source.length) {
        const inner = source[index];
        index += 1;
        if (inner === "\\") {
          index += 1;
          continue;
        }
        if (inner === '"') break;
      }
      continue;
    }
    if (char === "[" || char === "{" || char === "(") depth += 1;
    else if (char === "]" || char === "}" || char === ")") {
      depth -= 1;
      if (depth === 0) return source.slice(bodyStart, index);
    }
    index += 1;
  }
  throw new Error(
    "parseEngineModelTable: MODEL_TABLE's opening `&[` is never closed — the table is truncated " +
      "or the brackets are unbalanced",
  );
}

/**
 * Parse the `sceneworks_id` → `engine_id` rows out of `crates/sceneworks-worker/src/engines.rs`.
 *
 * Returns `[{ sceneworksId, engineId }]` in source order. Throws — never returns a partial or empty
 * table — because every failure mode here (renamed const, reformatted rows, a row missing a field)
 * would otherwise degrade into "this model has no engine", which reads downstream as a legitimate
 * "unknown" and hides the breakage. A DROPPED model is the dangerous direction precisely because
 * unknown renders exactly as it did before this story: it looks fine.
 */
export function parseEngineModelTable(rustSource) {
  if (typeof rustSource !== "string" || rustSource.length === 0) {
    throw new Error("parseEngineModelTable: expected the engines.rs source text");
  }
  // Comments first, then locate the table: a `];` or a `ModelRow` mention inside a comment must not
  // be able to end the table or be counted as a row.
  const body = sliceModelTableBody(stripRustComments(rustSource));

  // Rows carry only scalar/string fields today — no nested braces — so a brace-free match is exact.
  // The moment that stops being true this regex silently SKIPS the row, so the count below is what
  // makes the assumption enforceable rather than merely documented.
  const rows = [];
  for (const match of body.matchAll(/ModelRow\s*\{([^{}]*)\}/g)) {
    const fields = match[1];
    const sceneworksId = /\bsceneworks_id:\s*"([^"]+)"/.exec(fields)?.[1];
    const engineId = /\bengine_id:\s*"([^"]+)"/.exec(fields)?.[1];
    if (!sceneworksId || !engineId) {
      throw new Error(
        `parseEngineModelTable: a ModelRow is missing sceneworks_id/engine_id:\n${fields.trim()}`,
      );
    }
    rows.push({ sceneworksId, engineId });
  }

  // Every `ModelRow` the (comment-free) table declares must have produced a row. Without this, a
  // row that grows a nested-brace field — `limits: Limits { max: 4 }` — is silently DROPPED and the
  // model reads as "unknown" downstream, i.e. renders exactly as before and looks fine. The drift
  // guard would only catch it on a PR where nobody blind-regenerates, and blind-regenerating is
  // exactly the workflow each remaining family story (sc-16953…sc-16960) follows.
  const declared = (body.match(/\bModelRow\b/g) ?? []).length;
  if (rows.length !== declared) {
    throw new Error(
      `parseEngineModelTable: MODEL_TABLE declares ${declared} ModelRow entries but only ` +
        `${rows.length} parsed. A row the brace-free matcher cannot read (most likely a new field ` +
        "with a nested `{ … }` value) would otherwise be dropped silently and read as \"unknown\". " +
        "Teach this parser the new shape, then re-run `npm run gen:preview-support`.",
    );
  }

  if (rows.length < MIN_EXPECTED_MODEL_ROWS) {
    throw new Error(
      `parseEngineModelTable: parsed only ${rows.length} rows (expected at least ` +
        `${MIN_EXPECTED_MODEL_ROWS}) — the parse is almost certainly wrong`,
    );
  }
  const seen = new Set();
  for (const row of rows) {
    if (seen.has(row.sceneworksId)) {
      throw new Error(
        `parseEngineModelTable: MODEL_TABLE declares ${row.sceneworksId} twice; the SceneWorks id ` +
          "must be unique or the derived flag is ambiguous",
      );
    }
    seen.add(row.sceneworksId);
  }
  return rows;
}

/**
 * Validate one stage-1 facts file and index it as `Map<engineId, supportsPreview>`.
 *
 * Mirrors the Rust dumper's refusal (`engine_capability_facts::facts_from_descriptors`) on the JS
 * side, so a hand-edited or truncated facts file is caught here rather than becoming a confident
 * "nothing supports preview".
 */
function indexFacts(facts, sourceLabel) {
  if (!facts || typeof facts !== "object") {
    throw new Error(`${sourceLabel}: not a JSON object`);
  }
  if (typeof facts.backend !== "string" || facts.backend.length === 0) {
    throw new Error(`${sourceLabel}: missing the \`backend\` this file belongs to`);
  }
  if (!Array.isArray(facts.engines) || facts.engines.length === 0) {
    throw new Error(
      `${sourceLabel}: \`engines\` is empty. A facts file with no engines is the vacuous-green ` +
        "trap — it would derive as \"no route supports live preview\". Re-dump on a lane that " +
        "links a registry (`cargo run -p sceneworks-worker --bin dump-engine-capabilities " +
        "--features backend-candle`, or the plain command on macOS).",
    );
  }
  const index = new Map();
  for (const engine of facts.engines) {
    if (typeof engine?.id !== "string" || typeof engine?.supportsPreview !== "boolean") {
      throw new Error(
        `${sourceLabel}: every engine needs a string \`id\` and a boolean \`supportsPreview\`, got ` +
          JSON.stringify(engine),
      );
    }
    if (index.has(engine.id)) {
      throw new Error(`${sourceLabel}: duplicate engine id ${engine.id}`);
    }
    index.set(engine.id, engine.supportsPreview);
  }
  return index;
}

/**
 * Derive the served preview-support catalog from the MODEL_TABLE rows + the per-backend facts.
 *
 * @param {{sceneworksId: string, engineId: string}[]} rows from {@link parseEngineModelTable}
 * @param {{backend: string, generatedFrom?: object, engines: object[]}[]} factsFiles stage-1 dumps
 * @returns the canonical catalog object both artifacts are written from
 */
export function derivePreviewSupport(rows, factsFiles) {
  if (!Array.isArray(factsFiles) || factsFiles.length === 0) {
    throw new Error(
      "derivePreviewSupport: no stage-1 facts files. Nothing can be derived without at least one " +
        "`config/engine-capabilities/capabilities.<backend>.json`.",
    );
  }

  const indexed = factsFiles
    .map((facts) => ({
      backend: facts.backend,
      inferenceRevision: facts.generatedFrom?.inferenceRevision ?? null,
      engines: indexFacts(facts, `capabilities.${facts?.backend ?? "?"}.json`),
    }))
    .sort((left, right) => (left.backend < right.backend ? -1 : left.backend > right.backend ? 1 : 0));

  const backendNames = indexed.map((entry) => entry.backend);
  if (new Set(backendNames).size !== backendNames.length) {
    throw new Error(`derivePreviewSupport: two facts files claim the same backend (${backendNames})`);
  }

  const models = {};
  for (const row of [...rows].sort((left, right) =>
    left.sceneworksId < right.sceneworksId ? -1 : left.sceneworksId > right.sceneworksId ? 1 : 0,
  )) {
    const byBackend = {};
    for (const entry of indexed) {
      // Absence is UNKNOWN, not false: a backend that never registered this engine gets no key.
      if (!entry.engines.has(row.engineId)) continue;
      byBackend[entry.backend] = entry.engines.get(row.engineId);
    }
    if (Object.keys(byBackend).length > 0) {
      models[row.sceneworksId] = byBackend;
    }
  }

  const generatedFrom = {};
  for (const entry of indexed) {
    generatedFrom[entry.backend] = { inferenceRevision: entry.inferenceRevision };
  }

  return {
    version: PREVIEW_SUPPORT_VERSION,
    generatedBy: PREVIEW_SUPPORT_GENERATOR,
    backends: backendNames,
    generatedFrom,
    models,
  };
}

/**
 * The web app's slice of the catalog — the SAME derivation with the backend-only provenance dropped.
 *
 * The generator emits both artifacts from one `derivePreviewSupport` call, so `builtin.preview-
 * support.jsonc` and `previewSupport.json` can never disagree; this transform only decides what the
 * browser bundle has to carry (`generatedBy` / `generatedFrom` are audit fields the UI never reads).
 */
export function catalogToWebPreviewSupport(catalog) {
  return {
    version: catalog.version,
    backends: catalog.backends,
    models: catalog.models,
  };
}
