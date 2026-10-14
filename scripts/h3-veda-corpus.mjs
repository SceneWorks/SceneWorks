// Approved finite corpus and source-backed boundary routes; no extra matrix dimension.
const check = (ok, why) => { if (!ok) throw new Error(`H3 S9 corpus: ${why}`); };
const FIELDS = ["first", "last", "images", "clips", "audio"];
export const CAP_CASES = {
  images_at_cap: { counts: [9, 0, 0] },
  clips_at_cap: { counts: [0, 3, 0] },
  audio_at_cap_with_visual: { counts: [1, 0, 3] },
  combined_at_cap: { counts: [6, 3, 3] },
  combined_images_clips_at_cap: { counts: [9, 3, 0] },
  combined_images_audio_at_cap: { counts: [9, 0, 3] },
  soundtrack_slot_accounting: { counts: [6, 3, 3], soundtracks: 3 },
  images_over: { counts: [10, 0, 0], refusal: ["referenceAssetIds must contain at most 9 ids"] },
  clips_over: { counts: [0, 4, 0], refusal: ["minimax_h3_ref takes up to 3 source clips"] },
  audio_over: { counts: [1, 0, 4], refusal: ["referenceAudioAssetIds must contain at most 3 ids"] },
  combined_over: { counts: [7, 3, 3], refusal: ["minimax_h3_ref takes up to 12 reference files in total", "supplies 13"] },
  audio_only: { counts: [0, 0, 1], refusal: ["Reference to Video requires at least one reference image or video clip"] },
  empty_refs: { counts: [0, 0, 0], refusal: ["Reference to Video requires at least one reference image or video clip"] },
};
export function carrierBinding(m, row) {
  const base = m.bindings[row.profile]; check(base, `profile binding absent: ${row.profile}`);
  if (base.byCanvas) check(base.byCanvas[row.canvas], `canvas binding absent: ${row.profile}/${row.canvas}`);
  const result = { seed: base.seed }, chosen = row.carriers ?? base.byCanvas?.[row.canvas] ?? base;
  check(Object.keys(chosen).every((field) => FIELDS.includes(field) || (chosen === base && ["seed", "byCanvas"].includes(field))), "unknown carrier override field");
  for (const field of FIELDS) if (chosen[field] !== undefined) result[field] = chosen[field];
  return result;
}
export function validateCarrierBinding(m, b, profile, { mixed = true, refusal = false } = {}) {
  check(Number.isSafeInteger(b.seed) && b.seed >= 0 && b.seed <= 4294967295, "seed outside uint32 domain");
  check(!!b.first === ["fl2va_first", "fl2va_both"].includes(profile) && !!b.last === ["fl2va_last", "fl2va_both"].includes(profile), `keyframe shape mismatch: ${profile}`);
  for (const [field, type] of [["first", "image"], ["last", "image"], ["images", "image"], ["clips", "video"], ["audio", "audio"]]) {
    const keys = ["first", "last"].includes(field) ? (b[field] ? [b[field]] : []) : (b[field] ?? []);
    check(Array.isArray(keys), `carrier list invalid: ${field}`);
    for (const key of keys) { const a = m.assets[key]; check(a?.assetId && !a.assetId.includes("UNBOUND") && a.type === type, `carrier ID/type unbound: ${key}/${type}`); }
  }
  const counts = [b.images?.length ?? 0, b.clips?.length ?? 0, b.audio?.length ?? 0];
  if (profile !== "r2va_mixed") check(counts.every((n) => n === 0), `unexpected reference carriers: ${profile}`);
  else if (!refusal) {
    check(counts[0] + counts[1] > 0 && counts[0] <= 9 && counts[1] <= 3 && counts[2] <= 3 && counts.reduce((a, n) => a + n, 0) <= 12, "reference set exceeds source caps or lacks visual reference");
    if (mixed) check(counts.every((n) => n > 0) && b.clips.some((key) => m.assets[key].hasAudio === true), "named mixed R2VA needs images, clip soundtrack and standalone audio");
  }
}
export function validateCorpusBindings(m, canvases) {
  for (const profile of Object.keys(m.prompts)) {
    const b = m.bindings[profile];
    if (b?.byCanvas) check(Object.keys(b.byCanvas).length === canvases.length && canvases.every((c) => Object.hasOwn(b.byCanvas, c)), `complete canvas bindings required: ${profile}`);
    for (const canvas of canvases) validateCarrierBinding(m, carrierBinding(m, { profile, canvas }), profile);
  }
  check(m.boundaryBindings && Object.keys(m.boundaryBindings).length === Object.keys(CAP_CASES).length, "complete finite cap boundary bindings required");
  for (const [id, spec] of Object.entries(CAP_CASES)) {
    const carriers = m.boundaryBindings[id]; check(carriers && Object.keys(carriers).every((k) => FIELDS.includes(k)), `unknown/missing cap binding: ${id}`);
    const b = { seed: m.bindings.r2va_mixed.seed, ...carriers };
    validateCarrierBinding(m, b, "r2va_mixed", { mixed: false, refusal: !!spec.refusal });
    check([b.images?.length ?? 0, b.clips?.length ?? 0, b.audio?.length ?? 0].every((n, i) => n === spec.counts[i]), `source cap counts differ: ${id}`);
    if (["images_at_cap", "clips_at_cap", "audio_at_cap_with_visual"].includes(id)) for (const field of ["images", "clips", "audio"]) check(new Set((b[field] ?? []).map((key) => m.assets[key].assetId)).size === (b[field]?.length ?? 0), `unique at-cap corpus required: ${id}/${field}`);
    if (spec.soundtracks) check(b.clips.filter((key) => m.assets[key].hasAudio === true).length === spec.soundtracks, "soundtrack slot proof must actually carry three attached soundtracks");
  }
}
export function publishedBoundaryRows(catalog, m) {
  const ref = catalog.models.find((x) => x.id === "minimax_h3_ref").limits;
  check(ref.maxReferenceAssets === 9 && ref.maxSourceClipAssets === 3 && ref.maxReferenceAudioAssets === 3 && ref.maxCombinedReferenceAssets === 12, "source reference caps changed; reconcile complete corpus");
  const rows = [], sampling = { steps: 50, schedulerShift: 12, audioSchedulerShift: 3 };
  const add = (profile, backend, id, values) => rows.push({ profile, model: profile === "t2va" ? "minimax_h3" : "minimax_h3_ref", backend, tier: "q4", recipe: "base_50", canvas: "768x768", frames: 124, duration: 5.1667, sampling, style: null, proof: "published_input_boundaries", kind: "product_proof", expectation: "sparse", boundary: id, id: `boundary/${backend}/${profile}/${id}`, ...values });
  for (const backend of ["mlx", "candle_cuda"]) {
    for (const [id, spec] of Object.entries(CAP_CASES)) add("r2va_mixed", backend, id, { carriers: m.boundaryBindings[id], ...(spec.refusal ? { kind: "expected_refusal", refusal: { category: "input_boundary", messageIncludes: spec.refusal } } : {}) });
    for (const model of catalog.models) {
      const profile = model.id === "minimax_h3" ? "t2va" : "r2va_mixed";
      check(model.limits.durations.length === 14 && model.limits.fps.length === 1 && model.limits.fps[0] === 24, "published duration/fps set changed");
      for (const duration of model.limits.durations) { const frames = Math.round(duration * 24); check((frames - 5) % 17 === 0, "published frame lattice changed"); add(profile, backend, `published_duration_${frames}`, { frames, duration }); }
      add(profile, backend, "in_window_duration_snap", { duration: 6, frames: 158 });
      for (const duration of [5, 15, 0, -1]) add(profile, backend, `duration_refuse_${duration}`, { duration, kind: "expected_refusal", refusal: { category: "input_boundary", messageIncludes: duration <= 0 ? ["duration must be between 1 and 30"] : [model.id, duration === 5 ? "renders clips of at least" : "renders clips up to"] } });
    }
  }
  return rows;
}
export function validateRefusal(row, response) {
  check([400, 422].includes(response.status), "required validation refusal absent (queued/failed jobs are not API refusal proof)");
  const text = JSON.stringify(response.value);
  if (row.refusal?.category === "input_boundary") check(row.refusal.messageIncludes.every((part) => text.includes(part)), "boundary refusal absent or unrelated");
  else check(text.includes(row.recipe) && /declared.*not|not.*compatible/i.test(text), "modelIds refusal absent or unrelated");
}

// Provisioning uses real normal-product import/readback receipts. This function never invents IDs.
export function bindCorpusPlan(m, plan, receipt) {
  check(plan.schemaVersion === 1 && plan.scope === "named-required" && Object.keys(plan.assets ?? {}).length === 33, "source finite33 corpus plan missing");
  check(receipt.projectId && !receipt.projectId.includes("UNBOUND") && receipt.planSha256 === m.corpusPlan?.sha256, "owned import project/plan receipt differs");
  check(Object.keys(receipt.assets ?? {}).length === 33 && Object.keys(plan.assets).every(key => Object.hasOwn(receipt.assets,key)), "complete33 product imports/readbacks required");
  check(plan.catalogSha256 === m.sourceFreeze.files["config/manifests/builtin.models.jsonc"], "corpus catalog source changed");
  const assets = {};
  for (const [key, declared] of Object.entries(plan.assets)) {
    const a = receipt.assets[key];
    check(a.assetId && !a.assetId.includes("UNBOUND") && a.bytes === declared.bytes && a.sha256 === declared.sha256 && a.servedBytes === declared.bytes && a.servedSha256 === declared.sha256, `import/readback bytes or actualID differ: ${key}`);
    check(a.normalized?.path && Number.isSafeInteger(a.normalized.bytes) && a.normalized.bytes > 0 && /^[a-f0-9]{64}$/.test(a.normalized.sha256 ?? ""), `prepared carrier bytes missing: ${key}`);
    assets[key] = { path:a.path,bytes:a.bytes,sha256:a.sha256,assetId:a.assetId,type:declared.type,hasAudio:declared.hasAudio,normalized:a.normalized };
  }
  const result = { ...m,projectId:receipt.projectId,assets,bindings:structuredClone(plan.bindings),boundaryBindings:structuredClone(plan.boundaryBindings) };
  validateCorpusBindings(result,plan.canvases);
  return result;
}
