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
// support is a property of **the engine that runs it**. Provider routes and preview sinks can land
// independently on MLX and Candle, so the same model id and app version can temporarily have two
// answers. A single shipped boolean is wrong in that ordering, so the derived value is a
// `{ backend: bool }` map even when the current routes happen to agree.
//
// ## Absence is UNKNOWN, never false
//
// A backend key is emitted only when that backend's facts file actually contains the engine id. A
// backend that never registered a given engine, or a backend whose
// facts file has not been dumped on this checkout, produces NO key — the consumer reads that as
// "unknown" and renders exactly as it did before this story. Under-claiming is invisible;
// over-claiming would leave a placeholder up forever on a route that never emits.
//
// ## Two registries, two join rules (sc-17593)
//
// `crate::inference_runtime::audio` is a SEPARATE `ProviderRegistry` from the media one (epic 13400):
// audio is candle-native on every platform and rides its own lane rather than the mlx media graph.
// Until sc-17593 nothing dumped it, so every audio route was "unknown" everywhere — reported exactly
// like "not wired". Its facts arrive as a second input, from `config/engine-capabilities/audio/`.
//
// The join differs, which is why the inputs stay apart rather than being concatenated. Media engine
// ids reach SceneWorks model ids through MODEL_TABLE; audio engine ids ARE SceneWorks model ids
// (`kokoro_82m`, `chatterbox_tts`, `acestep_v15_turbo`, …) — `audio_jobs.rs` passes the payload's
// `model` straight into `inference_runtime::load_audio`, so for audio the join is the identity. The
// two id namespaces are independent, so an id claimed by both is a genuine ambiguity and refused.

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
function stripRustComments(source, sourceLabel = "engines.rs") {
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
    throw new Error(`${sourceLabel} has an unterminated \`/*\` block comment`);
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
  const body = sliceModelTableBody(stripRustComments(rustSource, "parseEngineModelTable: engines.rs"));

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
 * Parse preview support for the bespoke Candle Qwen-Edit lane.
 *
 * That provider is dispatched directly by SceneWorks rather than registered in the generic media
 * registry, so stage-1 engine facts cannot describe it. The declaration and the actual
 * `QwenEditRequest.preview` wiring live in one Rust source file; both must remain present or this
 * parser fails loudly instead of preserving a stale `true` claim.
 */
export function parseBespokePreviewRoutes(rustSource) {
  if (typeof rustSource !== "string" || rustSource.length === 0) {
    throw new Error("parseBespokePreviewRoutes: expected qwen_edit_candle.rs source text");
  }
  const source = stripRustComments(
    rustSource,
    "parseBespokePreviewRoutes: qwen_edit_candle.rs",
  );
  const anchor = "const QWEN_EDIT_CANDLE_PREVIEW_MODELS: &[&str] =";
  const declarations = source.split(anchor).length - 1;
  if (declarations !== 1) {
    throw new Error(
      `parseBespokePreviewRoutes: expected exactly one ${anchor}, found ${declarations}`,
    );
  }
  const afterAnchor = source.slice(source.indexOf(anchor) + anchor.length);
  const open = afterAnchor.indexOf("&[");
  const close = open === -1 ? -1 : afterAnchor.indexOf("]", open + 2);
  if (open === -1 || close === -1) {
    throw new Error("parseBespokePreviewRoutes: preview model list is not a closed `&[...]`");
  }
  const body = afterAnchor.slice(open + 2, close);
  const modelIds = [...body.matchAll(/"([a-z0-9_]+)"/g)].map((match) => match[1]);
  const quotes = (body.match(/"/g) ?? []).length;
  if (modelIds.length === 0 || quotes !== modelIds.length * 2) {
    throw new Error("parseBespokePreviewRoutes: preview model list is empty or only partly parsed");
  }
  if (new Set(modelIds).size !== modelIds.length) {
    throw new Error("parseBespokePreviewRoutes: preview model list contains a duplicate id");
  }
  if (
    !/move\s*\|\s*index,\s*\(seed,\s*prompt\),\s*preview,\s*on_progress\s*\|/.test(source) ||
    !/let\s+req\s*=\s*QwenEditRequest\s*\{[\s\S]*?\bpreview,\s*\};/.test(source)
  ) {
    throw new Error(
      "parseBespokePreviewRoutes: QwenEditRequest is not wired to drive_gen_items' live preview sink",
    );
  }
  return modelIds.map((modelId) => ({
    modelId,
    backend: "candle",
    supportsPreview: true,
  }));
}

/**
 * Parse the direct-dispatch PuLID preview routes from both platform worker implementations.
 * PuLID is intentionally absent from MODEL_TABLE: MLX loads the registry engine through its
 * identity route and Candle calls its bespoke provider by name. The exact model constants and live
 * request sinks in these two files are therefore the authority for the derived catalog entries.
 */
export function parsePulidPreviewRoutes(mlxSource, candleSource) {
  if (typeof mlxSource !== "string" || typeof candleSource !== "string") {
    throw new Error("parsePulidPreviewRoutes: expected MLX and Candle PuLID source text");
  }
  const mlx = stripRustComments(mlxSource, "parsePulidPreviewRoutes: pulid.rs");
  const candle = stripRustComments(candleSource, "parsePulidPreviewRoutes: pulid_candle.rs");
  const modelId = /const\s+PULID_MODEL:\s*&str\s*=\s*"([a-z0-9_]+)";/.exec(mlx)?.[1];
  const candleModelId =
    /const\s+PULID_CANDLE_MODEL:\s*&str\s*=\s*"([a-z0-9_]+)";/.exec(candle)?.[1];
  if (!modelId || modelId !== candleModelId) {
    throw new Error("parsePulidPreviewRoutes: platform model declarations are missing or disagree");
  }
  if (!/GenerationRequest\s*\{[\s\S]*?\bpreview,\s*[\s\S]*?\};/.test(mlx)) {
    throw new Error("parsePulidPreviewRoutes: MLX GenerationRequest has no live preview sink");
  }
  if (!/PulidFluxRequest\s*\{[\s\S]*?\bpreview:\s*preview\.clone\(\),[\s\S]*?\};/.test(candle)) {
    throw new Error("parsePulidPreviewRoutes: Candle PulidFluxRequest has no live preview sink");
  }
  return ["mlx", "candle"].map((backend) => ({
    modelId,
    backend,
    supportsPreview: true,
    directDispatch: true,
  }));
}

/**
 * Parse one declared `&[&str]` backend list out of
 * `crates/sceneworks-worker/src/engine_capability_facts.rs`.
 *
 * Parsed rather than mirrored, for the same reason `MODEL_TABLE` is: a JS-side copy of the list
 * would have to be kept in step by hand, and the failure mode of a stale copy is that it stops
 * requiring a backend — i.e. it re-opens the hole rather than closing it.
 *
 * Throws on anything it cannot read in full. A partial parse here is the DANGEROUS direction: a
 * dropped entry shrinks the required set, so {@link assertBackendCoverage} stops asking for a
 * backend and a never-dumped file passes again, silently.
 *
 * @param {string} rustSource the engine_capability_facts.rs text
 * @param {string} constName `SCENEWORKS_BACKENDS` (media) or `SCENEWORKS_AUDIO_BACKENDS` (audio)
 * @param {string} caller the exported function name, so the error names what the caller asked for
 */
function parseBackendConst(rustSource, constName, caller) {
  if (typeof rustSource !== "string" || rustSource.length === 0) {
    throw new Error(`${caller}: expected the engine_capability_facts.rs source text`);
  }
  const open_token = `const ${constName}: &[&str] = &[`;
  // Comments first: the doc comments above both consts name the backends in prose, and a
  // `SCENEWORKS_BACKENDS` mention inside a comment must not be able to anchor the parse.
  const source = stripRustComments(rustSource, `${caller}: engine_capability_facts.rs`);
  // Each anchor carries its const's FULL name including the leading `const `, so neither declaration
  // can match the other's search: `const SCENEWORKS_AUDIO_BACKENDS: …` does not contain the
  // substring `const SCENEWORKS_BACKENDS: …`. Without that, reading one const as the other would
  // require the wrong set — silently, and in the direction that requires less.
  const open = source.indexOf(open_token);
  if (open === -1) {
    throw new Error(
      `${caller}: no ${JSON.stringify(open_token)} in engine_capability_facts.rs — the backend ` +
        "list was renamed or restructured; update this parser, or the stage-2 coverage guard " +
        "silently stops requiring any backend at all",
    );
  }
  const bodyStart = open + open_token.length;
  const bodyEnd = source.indexOf("]", bodyStart);
  if (bodyEnd === -1) {
    throw new Error(`${caller}: ${constName}' opening \`&[\` is never closed`);
  }
  const body = source.slice(bodyStart, bodyEnd);

  const backends = [...body.matchAll(/"([a-z0-9_-]+)"/g)].map((match) => match[1]);
  // Every string literal in the body must have been read. An entry the matcher cannot parse (an
  // escape, an unexpected character class, a `concat!`) would otherwise be dropped — and dropping
  // one WEAKENS the guard, which is invisible by construction.
  const quotes = (body.match(/"/g) ?? []).length;
  if (quotes !== backends.length * 2) {
    throw new Error(
      `${caller}: ${constName} holds ${quotes / 2} string literal(s) but only ` +
        `${backends.length} parsed (${JSON.stringify(backends)}). A dropped entry would stop this ` +
        "guard requiring that backend's dump. Teach this parser the new shape.",
    );
  }
  if (backends.length === 0) {
    throw new Error(
      `${caller}: ${constName} is empty, which would make the coverage assertion vacuous — the ` +
        "exact class of hole it exists to close",
    );
  }
  const seen = new Set();
  for (const backend of backends) {
    if (seen.has(backend)) {
      throw new Error(`${caller}: ${constName} declares ${backend} twice`);
    }
    seen.add(backend);
  }
  return backends;
}

/** The declared MEDIA backend set — `SCENEWORKS_BACKENDS`. */
export function parseSceneworksBackends(rustSource) {
  return parseBackendConst(rustSource, "SCENEWORKS_BACKENDS", "parseSceneworksBackends");
}

/**
 * The declared AUDIO backend set — `SCENEWORKS_AUDIO_BACKENDS` (sc-17593).
 *
 * A second const rather than a reuse of the media one because the two answer different questions.
 * `SCENEWORKS_BACKENDS` contains `candle`, and the MEDIA dump satisfies it; that says nothing about
 * whether the audio registry was ever walked. Backend-level coverage was the wrong granularity, and
 * that is exactly how the audio gap survived sc-17119's new guard.
 */
export function parseSceneworksAudioBackends(rustSource) {
  return parseBackendConst(
    rustSource,
    "SCENEWORKS_AUDIO_BACKENDS",
    "parseSceneworksAudioBackends",
  );
}

/**
 * Assert the dumped facts files cover exactly the backends SceneWorks can run (sc-17119).
 *
 * Every other check in this pipeline is scoped to *what is on disk* — the generator globs the
 * directory, the drift guard globs it, and `verifyEngineCapabilityFacts` validates only files that
 * exist. So a backend that was NEVER dumped is not a failure anywhere; it is simply absent, and
 * absence derives as "unknown", which renders exactly as it did before the feature shipped. This is
 * the one assertion that can see a file that is not there.
 *
 * Both directions are errors:
 * - **missing** — a declared backend has no dump; the catalog is silently blind to it.
 * - **undeclared** — a dump exists for a backend the Rust list does not know about, so the list is
 *   stale and cannot be trusted to require anything.
 *
 * The `lane` parameter exists because the media and audio registries are dumped separately and a
 * media-shaped remediation message would send the operator to the wrong file (sc-17593). It only
 * changes the wording — the assertion is identical, and it has to be, since "audio was never
 * dumped" is the same invisible failure "mlx was never dumped" was.
 *
 * @param {string[]} declaredBackends from {@link parseSceneworksBackends} / {@link parseSceneworksAudioBackends}
 * @param {string[]} dumpedBackends the `backend` field of each checked-in facts file
 * @param {"media"|"audio"} lane which registry's dumps these are
 */
export function assertBackendCoverage(declaredBackends, dumpedBackends, lane = "media") {
  if (lane !== "media" && lane !== "audio") {
    throw new Error(`assertBackendCoverage: unknown lane ${JSON.stringify(lane)}`);
  }
  const constName = lane === "audio" ? "SCENEWORKS_AUDIO_BACKENDS" : "SCENEWORKS_BACKENDS";
  const what = lane === "audio" ? "audio engine-capability facts" : "engine-capability facts";
  // An empty declared set makes every check below vacuously true — the same hole
  // `parseSceneworksBackends` refuses, refused again here because this function is exported and
  // called from three places, not all of which get their list from that parser.
  if (!Array.isArray(declaredBackends) || declaredBackends.length === 0) {
    throw new Error(
      "assertBackendCoverage: the declared backend set is empty, so this assertion would pass for " +
        `ANY set of dumped files — including none. Read it from ${constName} in ` +
        "crates/sceneworks-worker/src/engine_capability_facts.rs.",
    );
  }
  // A facts file with no readable `backend` would otherwise flow through as `undefined` and be
  // reported as an unnamed entry — "facts exist for backend(s) []" — which tells the operator
  // nothing about which file is wrong.
  const nameless = dumpedBackends.filter(
    (backend) => typeof backend !== "string" || backend.length === 0,
  );
  if (nameless.length > 0) {
    throw new Error(
      `assertBackendCoverage: ${nameless.length} facts file(s) carry no readable \`backend\` ` +
        `field, so they cannot be matched against ${constName}. A facts file names the ` +
        "backend it belongs to; re-dump the offending file rather than hand-editing it.",
    );
  }
  const dumped = new Set(dumpedBackends);
  const missing = declaredBackends.filter((backend) => !dumped.has(backend));
  if (missing.length > 0) {
    throw new Error(
      `${what} are missing for backend(s) [${missing.join(", ")}]. SceneWorks can ` +
        "run them, so the served catalog reports \"unknown\" for every route they own — " +
        "indistinguishable from \"not wired\". " +
        (lane === "audio"
          ? "The audio registry is candle-native on every platform, so EITHER lane dumps it:\n" +
            "  cargo run -p sceneworks-worker --bin dump-engine-capabilities " +
            "[--no-default-features --features backend-candle]\n" +
            "which writes config/engine-capabilities/audio/capabilities.<backend>.json, "
          : "Dump each on the lane that owns it:\n" +
            "  mlx    (macOS) : cargo run -p sceneworks-worker --bin dump-engine-capabilities\n" +
            "  candle (off-Mac): cargo run -p sceneworks-worker --bin dump-engine-capabilities " +
            "--no-default-features --features backend-candle\n") +
        "then re-run `npm run gen:preview-support` from apps/web.",
    );
  }
  const undeclared = [...dumped].filter((backend) => !declaredBackends.includes(backend));
  if (undeclared.length > 0) {
    throw new Error(
      `${what} exist for backend(s) [${undeclared.join(", ")}] that ` +
        `${constName} (${JSON.stringify(declaredBackends)}) does not declare. Add them to ` +
        "the const in crates/sceneworks-worker/src/engine_capability_facts.rs — until then nothing " +
        "requires their dump, so deleting the file again would go unnoticed.",
    );
  }
}

/**
 * Validate one stage-1 facts file and index it as `Map<engineId, supportsPreview>`.
 *
 * Mirrors the Rust dumper's refusal (`engine_capability_facts::facts_from_descriptors`) on the JS
 * side, so a hand-edited or truncated facts file is caught here rather than becoming a confident
 * "nothing supports preview".
 */
function indexFacts(facts, sourceLabel, lane = "media") {
  if (!facts || typeof facts !== "object") {
    throw new Error(`${sourceLabel}: not a JSON object`);
  }
  if (typeof facts.backend !== "string" || facts.backend.length === 0) {
    throw new Error(`${sourceLabel}: missing the \`backend\` this file belongs to`);
  }
  // The `registry` discriminator (sc-17593). A media dump and an audio dump can carry the SAME
  // `backend` ("candle" on every platform for audio), so the file name alone cannot tell them
  // apart — only where it sits, which is exactly what a mistaken copy gets wrong. Media files carry
  // no `registry` key at all, deliberately: adding one would change their bytes and
  // `capabilities.mlx.json` can only be re-dumped on a Mac.
  const registry = facts.registry ?? "media";
  if (registry !== lane) {
    throw new Error(
      `${sourceLabel}: this is a ${JSON.stringify(registry)} dump but it was read as ` +
        `${JSON.stringify(lane)}. The media and audio registries have independent engine-id ` +
        "namespaces and different stage-2 joins, so reading one as the other silently answers for " +
        "the wrong routes. Media dumps live in config/engine-capabilities/, audio dumps in " +
        "config/engine-capabilities/audio/.",
    );
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

const byBackendName = (left, right) =>
  left.backend < right.backend ? -1 : left.backend > right.backend ? 1 : 0;

/**
 * Derive the served preview-support catalog from the MODEL_TABLE rows + the per-backend facts, plus
 * the audio registry's own dumps (sc-17593).
 *
 * The two registries are joined by different rules and merged into one `models` object:
 * - **media** — `MODEL_TABLE.sceneworks_id -> engine_id`, many-to-one.
 * - **audio** — the identity: an audio engine id IS the SceneWorks model id it serves.
 *
 * An id claimed by both is refused rather than resolved, because there is no correct answer: two
 * independent registries would be describing one catalog entry.
 *
 * Of the 14 dumped audio engines, 8 have a `builtin.models.jsonc` entry — the six `type: "audio"`
 * ids plus `mmaudio_small_16k` / `mmaudio_large_44k`, which are typed `"utility"` there but are
 * ordinary audio Generators in the registry, so they get a real served answer. The remaining six
 * (`stable_audio_3_*`) have no entry and are emitted all the same. They are never *looked up* —
 * `preview_support::apply_to_model_entry` only answers for ids the served catalog carries — and
 * emitting them means the answer is already right on the day such a model is added, with no
 * re-dump. Filtering through the manifest would buy nothing but a second parser, and would make a
 * new audio model's arrival a re-dump event on a box that may not be to hand.
 *
 * @param {{sceneworksId: string, engineId: string}[]} rows from {@link parseEngineModelTable}
 * @param {{backend: string, generatedFrom?: object, engines: object[]}[]} factsFiles media stage-1 dumps
 * @param {{backend: string, registry: string, generatedFrom?: object, engines: object[]}[]} audioFactsFiles audio stage-1 dumps
 * @param {{modelId: string, backend: string, supportsPreview: boolean, directDispatch?: boolean}[]} bespokePreviewRoutes exact direct-dispatch routes parsed from worker source
 * @returns the canonical catalog object both artifacts are written from
 */
export function derivePreviewSupport(rows, factsFiles, audioFactsFiles = [], bespokePreviewRoutes = []) {
  if (!Array.isArray(factsFiles) || factsFiles.length === 0) {
    throw new Error(
      "derivePreviewSupport: no stage-1 facts files. Nothing can be derived without at least one " +
        "`config/engine-capabilities/capabilities.<backend>.json`.",
    );
  }
  if (!Array.isArray(audioFactsFiles)) {
    throw new Error("derivePreviewSupport: audioFactsFiles must be an array of audio dumps");
  }
  if (!Array.isArray(bespokePreviewRoutes)) {
    throw new Error("derivePreviewSupport: bespokePreviewRoutes must be an array");
  }

  const indexed = factsFiles
    .map((facts) => ({
      backend: facts.backend,
      inferenceRevision: facts.generatedFrom?.inferenceRevision ?? null,
      engines: indexFacts(facts, `capabilities.${facts?.backend ?? "?"}.json`),
    }))
    .sort(byBackendName);

  const backendNames = indexed.map((entry) => entry.backend);
  if (new Set(backendNames).size !== backendNames.length) {
    throw new Error(`derivePreviewSupport: two facts files claim the same backend (${backendNames})`);
  }

  const audioIndexed = audioFactsFiles
    .map((facts) => ({
      backend: facts.backend,
      inferenceRevision: facts.generatedFrom?.inferenceRevision ?? null,
      engines: indexFacts(facts, `audio/capabilities.${facts?.backend ?? "?"}.json`, "audio"),
    }))
    .sort(byBackendName);

  const audioBackendNames = audioIndexed.map((entry) => entry.backend);
  if (new Set(audioBackendNames).size !== audioBackendNames.length) {
    throw new Error(
      `derivePreviewSupport: two audio facts files claim the same backend (${audioBackendNames})`,
    );
  }

  const models = {};
  for (const row of rows) {
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

  const modelIds = new Set(rows.map((row) => row.sceneworksId));
  const knownBackends = new Set(backendNames);
  const bespokeKeys = new Set();
  for (const route of bespokePreviewRoutes) {
    if (
      typeof route?.modelId !== "string" ||
      typeof route?.backend !== "string" ||
      route?.supportsPreview !== true ||
      ![undefined, true].includes(route?.directDispatch)
    ) {
      throw new Error(
        `derivePreviewSupport: malformed bespoke preview route ${JSON.stringify(route)}`,
      );
    }
    if (!modelIds.has(route.modelId) && route.directDispatch !== true) {
      throw new Error(
        `derivePreviewSupport: bespoke preview route names unknown MODEL_TABLE id ${route.modelId}`,
      );
    }
    if (!knownBackends.has(route.backend)) {
      throw new Error(
        `derivePreviewSupport: bespoke preview route names undumped backend ${route.backend}`,
      );
    }
    const key = `${route.modelId}\u0000${route.backend}`;
    if (bespokeKeys.has(key)) {
      throw new Error(`derivePreviewSupport: duplicate bespoke preview route ${key}`);
    }
    bespokeKeys.add(key);
    const existing = models[route.modelId]?.[route.backend];
    if (existing !== undefined && existing !== route.supportsPreview) {
      throw new Error(
        `derivePreviewSupport: bespoke preview route ${route.modelId}/${route.backend} conflicts ` +
          "with the generic media registry dump",
      );
    }
    models[route.modelId] = {
      ...(models[route.modelId] ?? {}),
      [route.backend]: route.supportsPreview,
    };
  }

  // Every MODEL_TABLE id, not merely the ones that produced an answer: the collision is between the
  // two NAMESPACES, and a media row whose engine happens to be in no facts file today would
  // otherwise let an audio engine quietly take its name and answer for it.
  const fromMedia = modelIds;
  for (const entry of audioIndexed) {
    for (const [engineId, supportsPreview] of entry.engines) {
      if (fromMedia.has(engineId)) {
        throw new Error(
          `derivePreviewSupport: ${engineId} is both a MODEL_TABLE model id and an audio engine ` +
            "id. The two registries have independent namespaces, so there is no correct answer " +
            "here — one of them would silently overwrite the other. Rename the colliding id, or " +
            "teach this derivation which registry owns it.",
        );
      }
      models[engineId] = { ...(models[engineId] ?? {}), [entry.backend]: supportsPreview };
    }
  }

  // Sorted so the artifact is byte-stable regardless of which registry contributed a key.
  const sortedModels = {};
  for (const id of Object.keys(models).sort()) sortedModels[id] = models[id];

  const generatedFrom = {};
  for (const entry of [...indexed, ...audioIndexed]) {
    const existing = generatedFrom[entry.backend];
    // The media and audio dumps of ONE backend must come from one revision — `candle` is both, on
    // every platform. A disagreement means one of the two was not re-dumped at the current pin, and
    // silently keeping either would make `generatedFrom` describe a checkout that never existed.
    if (existing && existing.inferenceRevision !== entry.inferenceRevision) {
      throw new Error(
        `derivePreviewSupport: backend ${entry.backend} was dumped at two different inference ` +
          `revisions (${existing.inferenceRevision} and ${entry.inferenceRevision}). Re-dump both ` +
          "its media and audio facts at the pin the worker currently carries.",
      );
    }
    generatedFrom[entry.backend] = { inferenceRevision: entry.inferenceRevision };
  }

  return {
    version: PREVIEW_SUPPORT_VERSION,
    generatedBy: PREVIEW_SUPPORT_GENERATOR,
    // The union, sorted: a backend that appears only in the audio lane must still be advertised, or
    // a consumer iterating `backends` has no key for its routes.
    backends: [...new Set([...backendNames, ...audioBackendNames])].sort(),
    generatedFrom,
    models: sortedModels,
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
