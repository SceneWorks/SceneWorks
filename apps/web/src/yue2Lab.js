// YuE2 Song Lab (sc-23000, epic 22988) — the pure half of the experimental surface.
//
// Everything the lab SENDS is built here, from the lab's settings, by `buildYue2JobRequest`, and
// everything it SHOWS about a run is read here, off the job the server returned, by `yue2RunView`.
// Keeping both pure is what lets the tests assert the actual request body for every control and the
// actual rendering of every server outcome without a GPU.
//
// The request contract is `crates/sceneworks-core/src/yue2_score/jobs.rs` (`Yue2JobSpec`). Its one
// rule that shapes this module: a field sent to a kind that does not read it is a typed
// `yue2_invalid_combination`, never silently dropped. So the builder computes every field the
// settings imply and then keeps only the ones the job kind reads (`YUE2_FIELD_KINDS`, a mirror of the
// core's `FIELD_KINDS`) — a control never leaks into a kind it would break.

import {
  YUE2_MODEL_ID,
  licenseFromNotice,
  usagePolicyChips,
  yue2ExportStem,
  yue2ModelIdentity,
  yue2TakeFilename,
} from "./yue2Policy.js";

export {
  YUE2_MODEL_ID,
  licenseFromNotice,
  usagePolicyChips,
  yue2ExportStem,
  yue2ModelIdentity,
  yue2TakeFilename,
};

// What a job does (core `Yue2JobKind`).
export const YUE2_KINDS = Object.freeze([
  "create",
  "plan",
  "fromPlan",
  "cover",
  "renderVersion",
  "decode",
  "transcribe",
]);

export const YUE2_KIND_LABELS = Object.freeze({
  create: "Song",
  plan: "Plan only",
  fromPlan: "From saved plan",
  cover: "Cover",
  renderVersion: "Score version render",
  decode: "Cached decode",
  transcribe: "Transcription",
});

// Which kinds read each optional field — core `FIELD_KINDS`, verbatim.
export const YUE2_FIELD_KINDS = Object.freeze({
  style: ["create", "plan", "fromPlan", "cover"],
  lyrics: ["create", "plan", "fromPlan", "cover"],
  seed: ["create", "plan", "cover"],
  cfgScale: ["create", "plan", "cover"],
  steps: ["create", "fromPlan", "cover", "renderVersion"],
  planning: ["create", "plan"],
  score: ["create", "plan"],
  scoreSampling: ["create", "plan"],
  semanticSampling: ["create", "fromPlan", "cover", "renderVersion"],
  decoder: ["create", "fromPlan", "cover", "renderVersion", "decode"],
  tier: ["create", "plan", "fromPlan", "cover", "renderVersion", "decode"],
  precision: ["create", "plan", "fromPlan", "cover", "renderVersion", "decode"],
  offloadPolicy: ["create", "fromPlan", "cover", "renderVersion"],
  "memory.acoustic": ["create", "fromPlan", "cover", "renderVersion"],
  "memory.decode": ["create", "fromPlan", "cover", "renderVersion", "decode"],
  planJobId: ["fromPlan"],
  sourceJobId: ["decode"],
  versionId: ["renderVersion"],
  cover: ["cover"],
  sourceAudioAssetId: ["transcribe"],
  transcription: ["transcribe"],
  count: ["create", "plan", "cover"],
});

export const MAX_TAKES = 8;

// Smallest NAR attention chunk the engine accepts (core `MIN_ATTENTION_CHUNK_ELEMENTS`: one query
// row at the full 24 576-token context across 16 heads), and the decode tile's upper bound (core
// `MAX_DECODE_TILE_FRAMES`).
export const MIN_ATTENTION_CHUNK_ELEMENTS = 16 * 24576;
export const MAX_DECODE_TILE_FRAMES = 1024;

// One autoregressive phase's sampling overrides (core `TokenSampling`), in display order.
export const SAMPLING_FIELDS = Object.freeze([
  { key: "temperature", label: "Temperature", kind: "float", min: 0, max: 5, step: 0.05 },
  { key: "topP", label: "Top-p", kind: "float", min: 0.01, max: 1, step: 0.01 },
  { key: "topK", label: "Top-k", kind: "int", min: 1, step: 1 },
  { key: "repetitionPenalty", label: "Repetition penalty", kind: "float", min: 0.05, step: 0.05 },
  { key: "penaltyWindow", label: "Penalty window", kind: "int", min: 1, max: 100, step: 1 },
  { key: "minTokens", label: "Min tokens", kind: "int", min: 0, step: 1 },
  { key: "maxTokens", label: "Max tokens", kind: "int", min: 1, max: 24576, step: 1 },
]);

// The protocol defaults (core `ABC_TOKEN_DEFAULTS` / `SEMANTIC_TOKEN_DEFAULTS`), shown as placeholders.
export const SAMPLING_TOKEN_DEFAULTS = Object.freeze({
  scoreSampling: { minTokens: 32, maxTokens: 4096 },
  semanticSampling: { minTokens: 200, maxTokens: 9000 },
});

export function emptySampling() {
  return Object.fromEntries(SAMPLING_FIELDS.map((field) => [field.key, ""]));
}

// The lab's settings, as persisted (server-side, through the studio-settings seam). Every value is
// the control's own raw value; "" means "not set — the model default applies".
export function defaultYue2Settings() {
  return {
    optIn: false,
    tab: "compose",
    style: "",
    lyrics: "",
    planning: "full",
    planSource: "sample",
    suppliedScore: "",
    restorePlanJobId: "",
    planOnly: false,
    count: "1",
    coverSource: "version",
    coverVersionId: "",
    coverScore: "",
    // The core contract requires a cover mode (melody or full); melody is the upstream default.
    coverMode: "melody",
    coverKeep: "",
    coverTranslatedFrom: "",
    // Cover from a recording (sc-23002): the recording to transcribe, the transcription under
    // review, and the transcriber's window settings ("" = the engine default).
    transcribeAssetId: "",
    transcriptionId: "",
    transcribeMaxSeconds: "",
    transcribeOverlapSeconds: "",
    transcribeLookaheadSeconds: "",
    seed: "",
    cfgScale: "",
    steps: "",
    scoreSampling: emptySampling(),
    semanticSampling: emptySampling(),
    tier: "",
    decoder: "",
    precision: "",
    offloadPolicy: "",
    stageResidency: "",
    chunkAttention: "",
    attentionChunkSize: "",
    tileVaeDecode: "",
    decodeTileEdge: "",
    advancedOpen: false,
    selectedVersionId: "",
    compareA: "",
    compareB: "",
    compareRenderA: "",
    compareRenderB: "",
    compareNotes: "",
    // The score workbench's drafts live with the lab settings, so a tab change, a surface switch or
    // a relaunch does not lose an in-progress edit (bounded by the durable byte budget).
    editDraft: defaultEditDraft(),
    importDraft: defaultImportDraft(),
    presets: [],
  };
}

// Merge a restored snapshot onto the defaults, keeping only known keys of the right shape so a
// snapshot written by an older build can never inject a value a control cannot display.
export function restoreYue2Settings(saved) {
  const base = defaultYue2Settings();
  if (!saved || typeof saved !== "object") {
    return base;
  }
  const out = { ...base };
  for (const [key, value] of Object.entries(base)) {
    const restored = saved[key];
    if (restored === undefined || restored === null) {
      continue;
    }
    if (key === "scoreSampling" || key === "semanticSampling") {
      out[key] = { ...emptySampling(), ...pickStrings(restored, Object.keys(value)) };
    } else if (key === "presets") {
      out.presets = Array.isArray(restored) ? restored.filter(isPreset) : [];
    } else if (key === "editDraft" || key === "importDraft") {
      out[key] = restored && typeof restored === "object" && !Array.isArray(restored) ? { ...value, ...restored } : value;
    } else if (typeof value === typeof restored) {
      out[key] = restored;
    }
  }
  return out;
}

function pickStrings(object, keys) {
  if (!object || typeof object !== "object") {
    return {};
  }
  return Object.fromEntries(
    keys.filter((key) => typeof object[key] === "string").map((key) => [key, object[key]]),
  );
}

function isPreset(preset) {
  return (
    preset &&
    typeof preset === "object" &&
    typeof preset.id === "string" &&
    typeof preset.name === "string" &&
    preset.settings &&
    typeof preset.settings === "object"
  );
}

// ---- number parsing -------------------------------------------------------------------------

function numberOrUndefined(raw) {
  if (raw === "" || raw === null || raw === undefined) {
    return undefined;
  }
  const value = Number(raw);
  return Number.isFinite(value) ? value : undefined;
}

function intOrUndefined(raw) {
  const value = numberOrUndefined(raw);
  return value === undefined ? undefined : Math.trunc(value);
}

function triState(raw) {
  if (raw === "on") return true;
  if (raw === "off") return false;
  return undefined;
}

function compact(object) {
  const out = {};
  for (const [key, value] of Object.entries(object)) {
    if (value !== undefined) {
      out[key] = value;
    }
  }
  return Object.keys(out).length ? out : undefined;
}

export function samplingBody(raw) {
  if (!raw || typeof raw !== "object") {
    return undefined;
  }
  return compact(
    Object.fromEntries(
      SAMPLING_FIELDS.map((field) => [
        field.key,
        field.kind === "int" ? intOrUndefined(raw[field.key]) : numberOrUndefined(raw[field.key]),
      ]),
    ),
  );
}

function memoryBody(settings, kind) {
  const acoustic = YUE2_FIELD_KINDS["memory.acoustic"].includes(kind);
  const decode = YUE2_FIELD_KINDS["memory.decode"].includes(kind);
  return compact({
    stageResidency: acoustic ? triState(settings.stageResidency) : undefined,
    chunkAttention: acoustic ? triState(settings.chunkAttention) : undefined,
    // A chunk size is read only with chunked attention on, a tile edge only with tiled decode on
    // (core `validate_common`), so neither is sent without its switch.
    attentionChunkSize:
      acoustic && settings.chunkAttention === "on" ? intOrUndefined(settings.attentionChunkSize) : undefined,
    tileVaeDecode: decode ? triState(settings.tileVaeDecode) : undefined,
    decodeTileEdge: decode && settings.tileVaeDecode === "on" ? intOrUndefined(settings.decodeTileEdge) : undefined,
  });
}

function planText(value) {
  return typeof value === "string" && value ? value : undefined;
}

function trimmed(value) {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

// The compose surface's job kind: a restored plan synthesizes from it; otherwise plan-only stops
// after the score, and a full create runs every phase.
export function composeKind(settings) {
  if (settings.planSource === "restore") {
    return "fromPlan";
  }
  return settings.planOnly ? "plan" : "create";
}

/**
 * The request body of `POST /api/v1/projects/:id/yue2/jobs` for `kind`, from the lab's settings.
 * `target` carries the ids a kind needs that are not settings (`versionId` for a render,
 * `sourceJobId` for a cached decode). `requestedGpu` is the envelope's GPU pick.
 *
 * Every candidate field is computed, then filtered to the kinds that read it, so a control never
 * reaches a kind that would refuse it. Empty controls are omitted — the model default applies.
 */
export function buildYue2JobRequest(kind, settings, target = {}, requestedGpu = undefined) {
  const s = settings ?? defaultYue2Settings();
  const suppliedScore = s.planSource === "supplied" ? trimmed(stripYue2ExportHeader(s.suppliedScore)) : undefined;
  const samplesPlan = s.planSource === "sample" && s.planning !== "off";
  const count = intOrUndefined(s.count);
  // A restored plan renders ITS OWN style and lyrics (the engine and the server refuse any other
  // words — an edited plan is a new request), so a fromPlan body carries the plan's recorded text
  // verbatim (`target.plan`, from `yue2PlanRequest`), never the Compose fields; omitted when the
  // plan's text is not known, which the server reads as "the plan's".
  const fromPlan = kind === "fromPlan";
  const candidates = {
    style: fromPlan ? planText(target.plan?.style) : trimmed(s.style),
    lyrics: fromPlan ? planText(target.plan?.lyrics) : trimmed(s.lyrics),
    seed: intOrUndefined(s.seed),
    cfgScale: numberOrUndefined(s.cfgScale),
    steps: intOrUndefined(s.steps),
    planning: s.planning || undefined,
    score: suppliedScore,
    scoreSampling: samplesPlan ? samplingBody(s.scoreSampling) : undefined,
    semanticSampling: samplingBody(s.semanticSampling),
    // A cached decode ("Decode again" on a run card) chooses its decoder on that card, never
    // through the lab-wide select.
    decoder: (target.decoder !== undefined ? target.decoder : s.decoder) || undefined,
    tier: s.tier || undefined,
    precision: s.precision || undefined,
    offloadPolicy: s.offloadPolicy || undefined,
    planJobId: trimmed(s.restorePlanJobId),
    sourceJobId: trimmed(target.sourceJobId),
    versionId: trimmed(target.versionId),
    sourceAudioAssetId: trimmed(target.sourceAudioAssetId),
    transcription: kind === "transcribe" ? transcriptionBody(s) : undefined,
    // A cover never names a recording (`cover.sourceAudioAssetId` is refused with
    // `yue2_transcription_review_required`): it follows a reviewed score version or a pasted score.
    cover: kind === "cover" ? coverBody(s) : undefined,
    count: count !== undefined && count !== 1 ? count : undefined,
  };
  const body = { kind };
  for (const [field, value] of Object.entries(candidates)) {
    if (value === undefined) {
      continue;
    }
    const kinds = YUE2_FIELD_KINDS[field];
    if (kinds && kinds.includes(kind)) {
      body[field] = value;
    }
  }
  const memory = memoryBody(s, kind);
  if (memory) {
    body.memory = memory;
  }
  if (requestedGpu) {
    body.requestedGpu = requestedGpu;
  }
  return body;
}

// The transcriber's window settings (core `TranscriptionSettings`); omitted when all are defaults.
function transcriptionBody(s) {
  return compact({
    maxSeconds: numberOrUndefined(s.transcribeMaxSeconds),
    overlapSeconds: numberOrUndefined(s.transcribeOverlapSeconds),
    lookaheadSeconds: numberOrUndefined(s.transcribeLookaheadSeconds),
  });
}

// SheetSage2 transcribes 300 s windows; core `check_transcription`'s defaults and bounds.
export const TRANSCRIPTION_WINDOW_SECONDS = 300;
export const TRANSCRIPTION_DEFAULT_OVERLAP_SECONDS = 200;
export const TRANSCRIPTION_DEFAULT_LOOKAHEAD_SECONDS = 100;

// Core `check_transcription` over the RESOLVED values (an unset field takes its default):
// `0 <= lookahead <= overlap < 300` and a positive length limit. The server is the authority.
export function transcriptionSettingsProblems(settings) {
  const s = settings ?? defaultYue2Settings();
  const read = (raw) => (raw === "" || raw === null || raw === undefined ? undefined : Number(raw));
  const max = read(s.transcribeMaxSeconds);
  const overlapRaw = read(s.transcribeOverlapSeconds);
  const lookaheadRaw = read(s.transcribeLookaheadSeconds);
  if ([max, overlapRaw, lookaheadRaw].some((value) => value !== undefined && !Number.isFinite(value))) {
    return ["Transcription settings must be numbers."];
  }
  const problems = [];
  if (max !== undefined && max <= 0) {
    problems.push("The transcription length limit must be more than 0 seconds.");
  }
  const overlap = overlapRaw ?? TRANSCRIPTION_DEFAULT_OVERLAP_SECONDS;
  const lookahead = lookaheadRaw ?? TRANSCRIPTION_DEFAULT_LOOKAHEAD_SECONDS;
  if (overlap < 0 || overlap >= TRANSCRIPTION_WINDOW_SECONDS) {
    problems.push(`The window overlap must be at least 0 and under ${TRANSCRIPTION_WINDOW_SECONDS} seconds.`);
  } else if (lookahead < 0 || lookahead > overlap) {
    problems.push(`The look-ahead must be between 0 and the window overlap (${overlap} s).`);
  }
  return problems;
}

function coverBody(s) {
  const cover = {};
  if (s.coverSource === "version") {
    const id = trimmed(s.coverVersionId);
    if (id) cover.versionId = id;
  } else {
    const score = trimmed(stripYue2ExportHeader(s.coverScore));
    if (score) cover.score = score;
  }
  if (s.coverMode) cover.mode = s.coverMode;
  // Only a melody cover chooses which melodies it keeps.
  if (s.coverKeep && s.coverMode === "melody") cover.keep = s.coverKeep;
  const translated = trimmed(s.coverTranslatedFrom);
  if (translated) cover.translatedFrom = translated;
  return cover;
}

/**
 * What must be fixed before `kind` can be submitted, as user-facing sentences (never the server's
 * internal field names). Empty ⇒ submittable. The server re-validates everything; this only keeps
 * the obviously incomplete request from being sent.
 */
export function yue2RequestProblems(kind, settings, target = {}, context = {}) {
  const s = settings ?? defaultYue2Settings();
  const problems = [];
  if (context.hasProject === false) {
    problems.push("Open or create a workspace first.");
  }
  if (YUE2_FIELD_KINDS.seed.includes(kind)) {
    const seedProblem = seedValueProblem(s.seed);
    if (seedProblem) problems.push(seedProblem);
  }
  const needsLyrics = kind === "create" || kind === "plan" || kind === "cover";
  if (needsLyrics && !trimmed(s.lyrics)) {
    problems.push("Write the lyrics the song sings.");
  }
  if ((kind === "create" || kind === "plan") && s.planSource === "supplied") {
    if (!trimmed(s.suppliedScore)) {
      problems.push("Paste the ABC score to plan from, or choose to sample a new plan.");
    }
    if (s.planning === "off") {
      problems.push("A supplied score needs full or melody planning.");
    }
  }
  if (kind === "plan" && s.planning === "off") {
    problems.push("Plan only needs full or melody planning — planning off makes no score.");
  }
  if (kind === "fromPlan" && !trimmed(s.restorePlanJobId)) {
    problems.push("Choose the saved plan to restore.");
  }
  if (kind === "cover") {
    if (s.coverMode !== "melody" && s.coverMode !== "full") {
      problems.push("Choose the cover mode (melody or full).");
    }
    if (s.coverSource === "version" && !trimmed(s.coverVersionId)) {
      problems.push("Choose the reviewed score version the cover follows.");
    }
    if (s.coverSource === "inline" && !trimmed(s.coverScore)) {
      problems.push("Paste the reviewed ABC score the cover follows.");
    }
  }
  if (kind === "cover" && s.coverSource === "version" && context.coverVersion?.transcription) {
    // A transcribed score is covered in its own mode: melody-only → melody, full (chords) → full.
    const cot = context.coverVersion.cot;
    if ((cot === "melody" || cot === "full") && cot !== s.coverMode) {
      problems.push(
        `This transcribed score is a ${cot === "melody" ? "melody-only" : "full"} score — cover it in ${cot} mode.`,
      );
    }
  }
  if (kind === "transcribe") {
    if (!trimmed(target.sourceAudioAssetId)) {
      problems.push("Choose the recording to transcribe.");
    }
    problems.push(...transcriptionSettingsProblems(s));
  }
  if (kind === "renderVersion" && !trimmed(target.versionId)) {
    problems.push("Choose the score version to render.");
  }
  if (kind === "decode" && !trimmed(target.sourceJobId)) {
    problems.push("Choose the finished run whose latents to decode.");
  }
  const chunk = intOrUndefined(s.attentionChunkSize);
  if (
    YUE2_FIELD_KINDS["memory.acoustic"].includes(kind) &&
    s.chunkAttention === "on" &&
    chunk !== undefined &&
    chunk < MIN_ATTENTION_CHUNK_ELEMENTS
  ) {
    problems.push(`The attention chunk size must be at least ${MIN_ATTENTION_CHUNK_ELEMENTS} score elements.`);
  }
  const tile = intOrUndefined(s.decodeTileEdge);
  if (
    YUE2_FIELD_KINDS["memory.decode"].includes(kind) &&
    s.tileVaeDecode === "on" &&
    tile !== undefined &&
    (tile < 1 || tile > MAX_DECODE_TILE_FRAMES)
  ) {
    problems.push(`The decode tile must be between 1 and ${MAX_DECODE_TILE_FRAMES} latent frames.`);
  }
  const count = intOrUndefined(s.count);
  if (YUE2_FIELD_KINDS.count.includes(kind) && count !== undefined && (count < 1 || count > MAX_TAKES)) {
    problems.push(`Takes must be between 1 and ${MAX_TAKES}.`);
  }
  return problems;
}

// A seed is shown, recorded and exported through JavaScript, where a Number is exact only up to
// 2^53 - 1: a larger typed seed would silently become a different one, so it is refused.
export const MAX_SAFE_SEED = Number.MAX_SAFE_INTEGER;

export function seedValueProblem(raw) {
  if (raw === "" || raw === null || raw === undefined) {
    return null;
  }
  const text = String(raw).trim();
  const value = Number(text);
  if (!/^\d+$/.test(text) || !Number.isSafeInteger(value)) {
    return `The seed must be a whole number from 0 to ${MAX_SAFE_SEED}.`;
  }
  return null;
}

// Why a control does nothing for a job kind — core `why_not`, in user terms. Null when the kind
// reads the field (or when there is no job to submit, `kind` null).
export function yue2FieldDisabledReason(field, kind) {
  if (!kind || !YUE2_FIELD_KINDS[field] || YUE2_FIELD_KINDS[field].includes(kind)) {
    return null;
  }
  let reason;
  if (["seed", "cfgScale", "planning", "score", "scoreSampling"].includes(field) && kind === "fromPlan") {
    reason = "a saved plan fixes it (an edited plan is a new request)";
  } else if (kind === "renderVersion") {
    reason = "the score version fixes its style, lyrics, planning, seed and guidance; edit the version to change them";
  } else if (kind === "decode") {
    reason = "a cached decode re-renders the source run's latents and generates nothing";
  } else if (kind === "plan") {
    reason = "a plan-only job stops after planning the score";
  } else if (kind === "cover") {
    reason = "a cover plans from its own reviewed score in its own mode";
  } else if (kind === "transcribe") {
    reason = "transcription takes only the source recording and its transcription settings";
  } else if (field === "count") {
    reason = "a restored plan, a score version and a cached decode render the same take every time";
  } else {
    reason = "this kind does not read it";
  }
  return `Not used by a ${YUE2_KIND_LABELS[kind] ?? kind} job: ${reason}.`;
}

// ---- model / licence ------------------------------------------------------------------------

// The commercial-use alternatives the catalog names (`commercialUse.alternatives`, ids), resolved
// to display names when the catalog has them.
export function commercialAlternatives(model, models = []) {
  const ids = Array.isArray(model?.commercialUse?.alternatives) ? model.commercialUse.alternatives : [];
  return ids.map((id) => {
    const match = (models ?? []).find((entry) => entry?.id === id);
    return { id, name: match?.ui?.label ?? match?.name ?? id };
  });
}

// ---- cover from a recording (sc-23002) ------------------------------------------------------

// The cover closure: the conditional components a cover from a recording needs (SheetSage2 +
// MERT-v2-FullSong), each with its install state, size and licence, and the purpose's summary
// (`conditionalPurposes.cover`). A component the catalog marks `blocked` again is reported with its
// reason and unblock condition — the install door would refuse it (`component_blocked`).
export function yue2CoverSetup(model) {
  const rows = (Array.isArray(model?.conditionalComponents) ? model.conditionalComponents : []).filter((row) =>
    (row?.requiredFor ?? []).includes("cover"),
  );
  const purpose = model?.conditionalPurposes?.cover ?? null;
  const components = rows.map((row) => ({
    componentId: row.componentId,
    repo: row.repo ?? "",
    revision: row.revision ?? "",
    estimatedSizeBytes: Number.isFinite(row.estimatedSizeBytes) ? row.estimatedSizeBytes : null,
    installState: row.installState ?? "missing",
    license: componentLicenseLabel(row.license),
    nonCommercial: row.nonCommercial === true,
    licenseBasis: row.licenseBasis ?? "",
    blocked: row.blocked ? { reason: row.blocked.reason ?? "", unblock: row.blocked.unblock ?? "" } : null,
  }));
  return {
    declared: components.length > 0,
    // Installed only when the catalog says so; a catalog without the summary proves nothing.
    installState: purpose?.installState ?? "missing",
    installed: purpose?.installState === "installed",
    blocked: Boolean(purpose?.blocked) || components.some((row) => row.blocked),
    components,
    totalBytes: components.reduce((sum, row) => sum + (row.estimatedSizeBytes ?? 0), 0),
  };
}

// `cc-by-nc-4.0` → `CC BY-NC 4.0`: the catalog's licence id, as a licence is printed.
export function componentLicenseLabel(id) {
  const match = typeof id === "string" ? id.match(/^cc-([a-z-]+)-(\d\.\d)$/i) : null;
  return match ? `CC ${match[1].toUpperCase()} ${match[2]}` : (id ?? "");
}

// A model_download job that installs the cover closure: one of the entry's downloads whose repo
// is a cover component's (the install door queues one job per component repo).
export function isCoverComponentDownload(job, model) {
  if (job?.type !== "model_download" || job?.payload?.modelId !== model?.id) {
    return false;
  }
  return yue2CoverSetup(model).components.some((row) => row.repo === job?.payload?.repo);
}

const PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

// MIDI 60 → "C4 (60)".
export function midiNoteLabel(midi) {
  const value = Number(midi);
  if (midi === null || midi === undefined || !Number.isFinite(value)) {
    return "—";
  }
  const rounded = Math.round(value);
  return `${PITCH_CLASSES[((rounded % 12) + 12) % 12]}${Math.floor(rounded / 12) - 1} (${value})`;
}

const EXPORT_KIND_ORDER = ["midi", "lab", "abc", "json", "text", "data"];
export const EXPORT_KIND_LABELS = Object.freeze({
  midi: "MIDI",
  lab: "LAB (timed labels)",
  abc: "ABC scores",
  json: "JSON",
  text: "Text",
  data: "Data",
});

function list(value) {
  return Array.isArray(value) ? value : [];
}

function warningRow(warning) {
  return typeof warning === "string"
    ? { code: "", message: warning }
    : { code: warning?.code ?? "", message: warning?.message ?? "" };
}

function readinessRow(row) {
  if (!row || typeof row !== "object") {
    return null;
  }
  return { ready: row.ready === true, reason: typeof row.reason === "string" ? row.reason : "" };
}

/**
 * Everything the transcription review shows, read off the server's `TranscriptionRecord` (whose
 * review / source / closure / octave sub-objects are the engine's snake_case JSON verbatim). Nothing
 * is invented: an absent field stays absent, and a refused mode keeps its reason.
 */
export function yue2TranscriptionView(record) {
  const review = record?.review && typeof record.review === "object" ? record.review : {};
  const voice = (raw) => ({
    notes: Number.isFinite(raw?.notes) ? raw.notes : 0,
    minPitch: raw?.min_pitch ?? null,
    maxPitch: raw?.max_pitch ?? null,
    medianPitch: raw?.median_pitch ?? null,
  });
  // The record's warnings are the review's; the review copy is the fallback for an older record.
  const warnings = (list(record?.warnings).length ? list(record.warnings) : list(review.warnings)).map(warningRow);
  const octave = record?.octaveEvidence && typeof record.octaveEvidence === "object" ? record.octaveEvidence : null;
  const groups = new Map();
  for (const file of list(record?.exports)) {
    const kind = EXPORT_KIND_ORDER.includes(file?.kind) ? file.kind : "data";
    groups.set(kind, [...(groups.get(kind) ?? []), file]);
  }
  const versionErrors = record?.versionErrors && typeof record.versionErrors === "object" ? record.versionErrors : {};
  const source = record?.source && typeof record.source === "object" ? record.source : {};
  const settings = record?.settings && typeof record.settings === "object" ? record.settings : {};
  return {
    id: record?.id ?? "",
    createdAt: record?.createdAt ?? null,
    jobId: record?.jobId ?? null,
    sourceAudioAssetId: record?.sourceAudioAssetId ?? null,
    sourceName: typeof source.name === "string" ? source.name : "",
    durationSeconds: Number.isFinite(source.duration_seconds) ? source.duration_seconds : null,
    device: record?.device ?? record?.closure?.device ?? "",
    settings: {
      overlapSeconds: settings.overlap_seconds ?? null,
      lookaheadSeconds: settings.lookahead_seconds ?? null,
      maxSeconds: settings.max_seconds ?? null,
    },
    vocal: voice(review.voices?.vocal),
    instrumental: voice(review.voices?.instrumental),
    chords: list(review.distinct_chords),
    chordRoots: Number.isFinite(review.distinct_chord_roots) ? review.distinct_chord_roots : null,
    keys: list(review.keys),
    sections: list(review.sections),
    bars: Number.isFinite(review.bars) ? review.bars : null,
    diagnostics: list(review.diagnostics),
    warnings,
    decodeWarnings: list(record?.decodeWarnings).map(warningRow),
    octave: octave
      ? {
          method: octave.method ?? "",
          range: Array.isArray(octave.transcribed_midi_range) ? octave.transcribed_midi_range : null,
          checked: octave.notes_checked ?? list(octave.notes).length,
          f0HalfDominant: octave.notes_with_more_energy_at_f0_half ?? null,
          fraction: Number.isFinite(octave.fraction_f0_half_dominant) ? octave.fraction_f0_half_dominant : null,
          notes: list(octave.notes),
        }
      : null,
    readiness: {
      melody: readinessRow(record?.readiness?.melody ?? review.cover?.melody),
      full: readinessRow(record?.readiness?.full ?? review.cover?.full),
    },
    versions: { melody: record?.versions?.melody ?? null, full: record?.versions?.full ?? null },
    versionErrors: Object.entries(versionErrors)
      .filter(([, message]) => message)
      .map(([mode, message]) => ({ mode, message: String(message) })),
    exportGroups: EXPORT_KIND_ORDER.filter((kind) => groups.has(kind)).map((kind) => ({
      kind,
      label: EXPORT_KIND_LABELS[kind],
      files: groups.get(kind),
    })),
    usagePolicy: record?.usagePolicy ?? null,
    replay: record?.replay ?? null,
    unload: record?.unload ?? null,
    closure: record?.closure ?? null,
  };
}

// Tier rows for the lab's install panel. A locally derived tier (`derivationPending`) is installable:
// its download fetches the original and derives the tier on this machine (sc-22999).
export function yue2TierRows(model) {
  const variants = Array.isArray(model?.variants) ? model.variants : [];
  return variants.map((variant) => {
    const installed = variant?.installState === "installed";
    const derived = variant?.derivationPending === true || variant?.installState === "derivationPending";
    const pending = variant?.pendingArtifact === true || variant?.installState === "pending";
    return {
      tier: variant?.variant,
      installed,
      derived,
      installable: !installed && !pending,
      state: installed
        ? "installed"
        : pending
          ? "not published yet"
          : derived
            ? "derived on this machine"
            : variant?.cacheState === "incomplete"
              ? "incomplete"
              : "not installed",
    };
  });
}

// The status of a model-download job, in words. A derived-tier install (q8 / q4) is claimed only by
// an audio-lane worker and may sit queued longer than a plain download, so it says so.
export function installJobStatusLabel(job) {
  const derived = Boolean(job?.payload?.localDerivation);
  const status = String(job?.status ?? "");
  if (derived && (status === "queued" || status === "pending")) {
    return "queued — waits for an audio-lane worker to derive it";
  }
  if (derived && status === "running") {
    return "deriving on this machine";
  }
  return status;
}

// Whether the decoder `option` (`standard` / `legacy`) is installed on this host — the catalog's
// per-option `installedChoices.decoder`. A catalog that does not report it proves nothing, so the
// option reads as not installed (the render would fail after the load otherwise).
export function yue2DecoderInstalled(model, option) {
  const installed = model?.installedChoices?.decoder;
  return Array.isArray(installed) && installed.includes(option);
}

export function yue2ModelInstalled(model) {
  return model?.installState === "installed";
}

// ---- runs -----------------------------------------------------------------------------------

const TRUNCATION_TEXT = {
  abc: "The score plan reached its token budget and was cut short — the song follows an incomplete plan.",
  semantic:
    "The semantic tokens reached their budget and were cut short — the recording ends before the lyrics do.",
};

export function isYue2Job(job) {
  return Boolean(job?.payload && typeof job.payload === "object" && job.payload.yue2);
}

const TERMINAL = new Set(["completed", "failed", "canceled", "interrupted"]);

/**
 * Everything a run card shows, read off the job the server returned: stage progress while it
 * runs, the error when it failed, and — once complete — truncation, warnings, side-effect errors,
 * the effective settings and the usage policy. Nothing is invented: an absent field stays absent.
 */
export function yue2RunView(job) {
  const spec = job?.payload?.yue2 ?? {};
  const block = job?.result?.yue2 && typeof job.result.yue2 === "object" ? job.result.yue2 : {};
  const status = job?.status ?? "queued";
  const running = !TERMINAL.has(status);
  const truncated = block.truncated && typeof block.truncated === "object" ? block.truncated : {};
  // `truncated: null` means the run could not tell (a failed render): unknown, never "not truncated".
  const truncationUnknown = "truncated" in block && block.truncated === null;
  const truncations = ["abc", "semantic"]
    .filter((key) => truncated[key] === true)
    .map((key) => ({ key, text: TRUNCATION_TEXT[key] }));
  const warnings = (Array.isArray(block.warnings) ? block.warnings : []).map((warning) =>
    typeof warning === "string"
      ? { code: "", message: warning }
      : { code: warning?.code ?? "", message: warning?.message ?? "" },
  );
  const progressNumber = Number(job?.progress);
  const error =
    status === "failed" || status === "interrupted"
      ? block.error || job?.error || "The run failed without an error message."
      : status === "canceled"
        ? job?.error || "Canceled."
        : null;
  return {
    id: job?.id,
    kind: spec.kind ?? block.kind ?? "create",
    kindLabel: YUE2_KIND_LABELS[spec.kind ?? block.kind] ?? spec.kind ?? "YuE2",
    status,
    running,
    cancelRequested: Boolean(job?.cancelRequested),
    message: typeof job?.message === "string" ? job.message : "",
    progress: Number.isFinite(progressNumber) ? Math.max(0, Math.min(1, progressNumber)) : 0,
    error,
    truncations,
    truncationUnknown,
    warnings,
    sideEffectErrors: Array.isArray(block.sideEffectErrors) ? block.sideEffectErrors : [],
    effectiveSettings: block.effectiveSettings ?? null,
    usagePolicy: block.usagePolicy ?? job?.payload?.usagePolicy ?? null,
    model: block.model ?? null,
    decoder: block.decoder ?? null,
    run: block.run ?? null,
    request: block.request ?? null,
    score: block.score ?? null,
    scoreVersionId: block.scoreVersionId ?? null,
    // A plan cut off by its token budget is kept on the run but never becomes a score version.
    scoreVersionSkipped:
      block.scoreVersionSkipped === "abc_truncated"
        ? "This plan was truncated, so it was not saved as a score version. Its partial score is still on the run."
        : block.scoreVersionSkipped
          ? `No score version was created (${block.scoreVersionSkipped}).`
          : null,
    renderRecordId: block.renderRecordId ?? null,
    // A finished transcription's review record and the recording it read (sc-23002).
    transcriptionId: block.transcriptionId ?? null,
    sourceAudioAssetId: spec.sourceAudioAssetId ?? null,
    versionId: block.versionId ?? spec.versionId ?? null,
    batch: spec.batch ?? block.batch ?? null,
    style: spec.style ?? block.effectiveSettings?.style ?? "",
    createdAt: job?.createdAt ?? null,
    // A completed run with a plan identity can be restored; a song / cached-decode run holds
    // latents that can be decoded again.
    restorable: status === "completed" && Boolean(block.run?.planIdentity),
    decodable:
      status === "completed" && ["song", "cached_decode"].includes(String(block.run?.kind ?? "")),
  };
}

// The style and lyrics a restorable run's plan was made with — its recorded request (the engine's
// `request.json`), else the settings it ran with. Null when the run recorded neither.
export function yue2PlanRequest(view) {
  const source = [view?.request, view?.effectiveSettings].find(
    (candidate) => candidate && typeof candidate === "object" && typeof candidate.lyrics === "string",
  );
  if (!source) {
    return null;
  }
  return { style: typeof source.style === "string" ? source.style : "", lyrics: source.lyrics };
}

// The YuE2 runs of a project, newest first.
export function yue2ProjectRuns(jobs, projectId) {
  return (Array.isArray(jobs) ? jobs : [])
    .filter((job) => isYue2Job(job) && (!projectId || job.projectId === projectId || job.payload?.projectId === projectId))
    .sort((a, b) => Date.parse(b.createdAt ?? 0) - Date.parse(a.createdAt ?? 0));
}

// The reproducibility + licence record of a run, as exported. It carries the usage policy the run
// was granted, so the noncommercial / experimental distinction travels with the export (E2).
export function yue2RunExport(job, assets = [], exportedAt = new Date().toISOString()) {
  const view = yue2RunView(job);
  return {
    schema: "sceneworks.yue2.runExport.v1",
    exportedAt,
    usagePolicy: view.usagePolicy,
    job: { id: view.id, kind: view.kind, status: view.status, createdAt: view.createdAt },
    model: view.model,
    decoder: view.decoder,
    effectiveSettings: view.effectiveSettings,
    request: view.request,
    submitted: job?.payload?.yue2 ?? null,
    run: view.run,
    truncated: job?.result?.yue2?.truncated ?? null,
    warnings: view.warnings,
    score: view.score,
    scoreVersionId: view.scoreVersionId,
    renderRecordId: view.renderRecordId,
    audio: (assets ?? []).map((asset) => ({
      id: asset?.id,
      displayName: asset?.displayName ?? null,
      path: asset?.file?.path ?? null,
      usagePolicy: asset?.extra?.usagePolicy ?? null,
    })),
  };
}

// Exported .abc files open with a `%` comment naming the licence, so a score file that leaves the
// app keeps the noncommercial distinction (E2). The native YuE2 dialect requires `X:1` on the
// first line, so the lab strips this header again whenever a score is pasted back in.
export const ABC_EXPORT_MARKER = "% SceneWorks YuE2 export:";

export function yue2AbcExportHeader(policy) {
  const license = policy?.license?.license || licenseFromNotice(policy?.license?.notice) || "see the model licence";
  const parts = [`${ABC_EXPORT_MARKER} weights licence ${license}`];
  if (policy?.nonCommercial !== false) {
    parts.push("NONCOMMERCIAL USE ONLY");
  }
  if (policy?.experimental !== false) {
    parts.push("experimental model");
  }
  return parts.join(" · ");
}

export function yue2AbcExport(abc, policy) {
  return `${yue2AbcExportHeader(policy)}\n${stripYue2ExportHeader(abc ?? "")}`;
}

export function stripYue2ExportHeader(abc) {
  if (typeof abc !== "string") {
    return abc;
  }
  const lines = abc.split("\n");
  let index = 0;
  while (index < lines.length && lines[index].startsWith(ABC_EXPORT_MARKER)) {
    index += 1;
  }
  return index ? lines.slice(index).join("\n") : abc;
}

// ---- score edits ----------------------------------------------------------------------------

export const EDIT_OPERATIONS = Object.freeze([
  { op: "reharmonize", label: "Reharmonize (chords only)" },
  { op: "strip_chords", label: "Strip chords" },
  { op: "set_tempo", label: "Set tempo" },
  { op: "arrange_sections", label: "Arrange sections" },
  { op: "set_lyrics", label: "Set lyrics" },
  { op: "set_style", label: "Set style" },
  { op: "replace_score", label: "Replace score" },
]);

export function defaultImportDraft() {
  return { open: false, abc: "", cot: "full", seed: "", cfgScale: "" };
}

export function defaultEditDraft() {
  return {
    op: "reharmonize",
    brief: "",
    chordChanges: [{ bar: "", onsetQuarters: "", chord: "" }],
    keepVoice: "both",
    bpm: "",
    style: "",
    sectionOrder: "",
    lyrics: "",
    abc: "",
    allowHarmony: false,
    allowTempo: false,
    allowMelody: false,
    melodyVoices: "vocal",
    melodyFromBar: "",
    melodyToBar: "",
    cot: "",
  };
}

/**
 * The bounded edit operation (core `ScoreEditOperation`, `op`-tagged snake_case) for a draft, or
 * `{ error }` naming what is missing. Only the fields the operation defines are sent.
 */
export function buildEditOperation(draft) {
  const d = draft ?? defaultEditDraft();
  const text = (value) => (typeof value === "string" && value.trim() ? value.trim() : null);
  switch (d.op) {
    case "reharmonize": {
      const changes = (d.chordChanges ?? [])
        .filter((row) => String(row.bar).trim() !== "" && String(row.onsetQuarters).trim() !== "")
        .map((row) => ({
          bar: Math.trunc(Number(row.bar)),
          onsetQuarters: String(row.onsetQuarters).trim(),
          chord: text(row.chord),
        }));
      if (!changes.length) return { error: "Add at least one chord change (bar and onset)." };
      return { operation: { op: "reharmonize", changes } };
    }
    case "strip_chords":
      return { operation: { op: "strip_chords", keepVoice: d.keepVoice || "both" } };
    case "set_tempo": {
      const bpm = intOrUndefined(d.bpm);
      if (!bpm || bpm < 1) return { error: "Give the new tempo in BPM." };
      const op = { op: "set_tempo", bpm };
      if (text(d.style)) op.style = text(d.style);
      return { operation: op };
    }
    case "arrange_sections": {
      const order = String(d.sectionOrder ?? "")
        .split(/[\s,]+/)
        .filter(Boolean)
        .map((value) => Number(value));
      if (!order.length || order.some((value) => !Number.isInteger(value) || value < 0)) {
        return { error: "List the section order as section numbers, e.g. 0, 1, 1, 2." };
      }
      if (!text(d.lyrics)) return { error: "Restate the lyrics for the new form." };
      const op = { op: "arrange_sections", sectionOrder: order, lyrics: text(d.lyrics) };
      if (text(d.style)) op.style = text(d.style);
      return { operation: op };
    }
    case "set_lyrics":
      if (!text(d.lyrics)) return { error: "Write the new lyrics." };
      return { operation: { op: "set_lyrics", lyrics: text(d.lyrics) } };
    case "set_style":
      if (!text(d.style)) return { error: "Write the new style." };
      return { operation: { op: "set_style", style: text(d.style) } };
    case "replace_score": {
      if (!text(d.abc)) return { error: "Paste the complete edited ABC score." };
      const allow = {};
      if (d.allowHarmony) allow.harmony = true;
      if (d.allowTempo) allow.tempo = true;
      if (d.allowMelody) {
        const from = intOrUndefined(d.melodyFromBar);
        const to = intOrUndefined(d.melodyToBar);
        if (from === undefined || to === undefined) {
          return { error: "Give the bar range the melody may change in." };
        }
        allow.melody = {
          voices: d.melodyVoices === "both" ? ["Vocal", "Ins"] : d.melodyVoices === "ins" ? ["Ins"] : ["Vocal"],
          fromBar: from,
          toBar: to,
        };
      }
      const op = { op: "replace_score", abc: stripYue2ExportHeader(d.abc) };
      if (Object.keys(allow).length) op.allow = allow;
      if (text(d.lyrics)) op.lyrics = text(d.lyrics);
      if (text(d.style)) op.style = text(d.style);
      if (d.cot) op.cot = d.cot;
      return { operation: op };
    }
    default:
      return { error: "Choose an edit operation." };
  }
}

export const UI_PROVENANCE = Object.freeze({ actor: "user", channel: "ui" });
