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

export const YUE2_MODEL_ID = "yue2";

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
  count: ["create", "plan", "cover"],
});

export const MAX_TAKES = 8;

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
    coverMode: "",
    coverKeep: "",
    coverTranslatedFrom: "",
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
    attentionChunkSize: acoustic ? intOrUndefined(settings.attentionChunkSize) : undefined,
    tileVaeDecode: decode ? triState(settings.tileVaeDecode) : undefined,
    decodeTileEdge: decode ? intOrUndefined(settings.decodeTileEdge) : undefined,
  });
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
  const suppliedScore = s.planSource === "supplied" ? trimmed(s.suppliedScore) : undefined;
  const samplesPlan = s.planSource === "sample" && s.planning !== "off";
  const count = intOrUndefined(s.count);
  const candidates = {
    style: trimmed(s.style),
    lyrics: trimmed(s.lyrics),
    seed: intOrUndefined(s.seed),
    cfgScale: numberOrUndefined(s.cfgScale),
    steps: intOrUndefined(s.steps),
    planning: s.planning || undefined,
    score: suppliedScore,
    scoreSampling: samplesPlan ? samplingBody(s.scoreSampling) : undefined,
    semanticSampling: samplingBody(s.semanticSampling),
    decoder: s.decoder || undefined,
    tier: s.tier || undefined,
    precision: s.precision || undefined,
    offloadPolicy: s.offloadPolicy || undefined,
    planJobId: trimmed(s.restorePlanJobId),
    sourceJobId: trimmed(target.sourceJobId),
    versionId: trimmed(target.versionId),
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

function coverBody(s) {
  const cover = {};
  if (s.coverSource === "version") {
    const id = trimmed(s.coverVersionId);
    if (id) cover.versionId = id;
  } else {
    const score = trimmed(s.coverScore);
    if (score) cover.score = score;
  }
  if (s.coverMode) cover.mode = s.coverMode;
  if (s.coverKeep) cover.keep = s.coverKeep;
  const translated = trimmed(s.coverTranslatedFrom);
  if (translated) cover.translatedFrom = translated;
  return cover;
}

/**
 * What must be fixed before `kind` can be submitted, as user-facing sentences (never the server's
 * internal field names). Empty ⇒ submittable. The server re-validates everything; this only keeps
 * the obviously incomplete request from being sent.
 */
export function yue2RequestProblems(kind, settings, target = {}) {
  const s = settings ?? defaultYue2Settings();
  const problems = [];
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
  if (kind === "fromPlan" && !trimmed(s.restorePlanJobId)) {
    problems.push("Choose the saved plan to restore.");
  }
  if (kind === "cover") {
    if (s.coverSource === "version" && !trimmed(s.coverVersionId)) {
      problems.push("Choose the reviewed score version the cover follows.");
    }
    if (s.coverSource === "inline" && !trimmed(s.coverScore)) {
      problems.push("Paste the reviewed ABC score the cover follows.");
    }
  }
  if (kind === "renderVersion" && !trimmed(target.versionId)) {
    problems.push("Choose the score version to render.");
  }
  if (kind === "decode" && !trimmed(target.sourceJobId)) {
    problems.push("Choose the finished run whose latents to decode.");
  }
  const count = intOrUndefined(s.count);
  if (YUE2_FIELD_KINDS.count.includes(kind) && count !== undefined && (count < 1 || count > MAX_TAKES)) {
    problems.push(`Takes must be between 1 and ${MAX_TAKES}.`);
  }
  return problems;
}

// ---- model / licence ------------------------------------------------------------------------

// The version / licence identity every lab surface prints: the picker, the presets and the runs.
// The licence name is the catalog's `license` when it declares one; YuE2's entry names its licence
// only inside `licenseNotice`, so the Creative Commons identifier is read from there (never assumed).
export function yue2ModelIdentity(model) {
  const declared = typeof model?.license === "string" && model.license.trim() ? model.license.trim() : null;
  return {
    id: model?.id ?? YUE2_MODEL_ID,
    name: model?.ui?.label ?? model?.name ?? "YuE2",
    version: "YuE2 (v2)",
    license: declared ?? licenseFromNotice(model?.licenseNotice) ?? "",
    licenseUrl: typeof model?.licenseUrl === "string" ? model.licenseUrl : "",
    experimental: model?.experimental === true,
    nonCommercial: model?.nonCommercial === true,
  };
}

export function licenseFromNotice(notice) {
  if (typeof notice !== "string") {
    return null;
  }
  const match = notice.match(/\bCC BY(?:-[A-Z]{2})*\s+\d\.\d\b/);
  return match ? match[0].replace(/\s+/g, " ") : null;
}

// Chips for a usage policy recorded on a job / asset (`sceneworks.usagePolicy.v1`).
export function usagePolicyChips(policy, fallbackModel = null) {
  const chips = [];
  const identity = yue2ModelIdentity(fallbackModel);
  chips.push(identity.version);
  if (policy?.experimental ?? identity.experimental) {
    chips.push("Experimental");
  }
  const nonCommercial = policy?.nonCommercial ?? identity.nonCommercial;
  if (nonCommercial) {
    chips.push("Noncommercial");
  }
  const license =
    policy?.license?.license || licenseFromNotice(policy?.license?.notice) || identity.license;
  if (license) {
    chips.push(String(license));
  }
  return chips;
}

// The commercial-use alternatives the catalog names (`commercialUse.alternatives`, ids), resolved
// to display names when the catalog has them.
export function commercialAlternatives(model, models = []) {
  const ids = Array.isArray(model?.commercialUse?.alternatives) ? model.commercialUse.alternatives : [];
  return ids.map((id) => {
    const match = (models ?? []).find((entry) => entry?.id === id);
    return { id, name: match?.ui?.label ?? match?.name ?? id };
  });
}

// The recorded block of cover transcription: every conditional component the catalog marks
// `blocked`, with its reason and unblock condition.
export function blockedTranscription(model) {
  const rows = Array.isArray(model?.conditionalComponents) ? model.conditionalComponents : [];
  return rows
    .filter((row) => row?.blocked && (row.requiredFor ?? []).includes("cover"))
    .map((row) => ({
      componentId: row.componentId,
      repo: row.repo,
      reason: row.blocked.reason ?? "",
      unblock: row.blocked.unblock ?? "",
    }));
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
    renderRecordId: block.renderRecordId ?? null,
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

// A download filename stem that keeps the licence distinction visible in the file itself.
export function yue2ExportStem(prefix, id, policy) {
  const safe = String(id ?? "run").replace(/[^A-Za-z0-9_-]+/g, "-");
  return `${prefix}-${safe}${policy?.nonCommercial ? "-noncommercial" : ""}`;
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
      const op = { op: "replace_score", abc: d.abc };
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
