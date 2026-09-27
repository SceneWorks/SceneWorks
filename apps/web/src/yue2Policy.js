// YuE2 usage-policy identity (sc-23000, epic 22988) — the licence / version chips and the
// licence-marked download names a YuE2 take carries. Split out of yue2Lab.js so the shared audio
// take helpers (audioTakes.js, which the initial bundle loads) do not pull the whole Song Lab
// module into the entry chunk; yue2Lab.js re-exports everything here.

export const YUE2_MODEL_ID = "yue2";

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

// A download filename stem that keeps the licence distinction visible in the file itself.
export function yue2ExportStem(prefix, id, policy) {
  const safe = String(id ?? "run").replace(/[^A-Za-z0-9_-]+/g, "-");
  return `${prefix}-${safe}${policy?.nonCommercial ? "-noncommercial" : ""}`;
}

// A YuE2 take's download name: never the style text, always the licence-marked export stem.
export function yue2TakeFilename(asset, policy = null) {
  return `${yue2ExportStem("yue2-song", asset?.id ?? "take", asset?.extra?.usagePolicy ?? policy)}.wav`;
}
