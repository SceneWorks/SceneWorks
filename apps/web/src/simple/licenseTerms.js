// Commercial-use badge for the Simple UI's Licenses screen (design handoff).
//
// The design shows one of two badges per model-licence row: "Commercial OK" (accent) or
// "Non-commercial" (danger). The real corpus (`bundledLicenses`, derived from
// apps/desktop/licenses/manifest.json) carries a free-text `license` string, not a flag —
// so the badge is DERIVED here, in one pure place, rather than duplicated per render site.
//
// The rule is deliberately conservative and one-directional: a licence is treated as
// non-commercial ONLY when it SAYS SO (the manifest names every restricted licence
// explicitly — "FLUX.1 [dev] Non-Commercial License v1.1.1", "Ideogram Non-Commercial Model
// Agreement", "CircleStone Labs Non-Commercial License v1.2", "Research / Non-Commercial
// (CC-BY-NC-4.0 · …)", "Stable Video Diffusion Non-Commercial Community License"). Anything
// else gets the permissive badge, which matches the design's own mapping (it badges the
// Stability AI Community License as "Commercial OK").
//
// The badge is a NAVIGATION aid, not legal advice: several "Commercial OK" licences
// (Stability Community, Krea 2 Community, LTX-2 Community, NVIDIA Open Model) carry
// revenue thresholds or other conditions. The row keeps the full licence NAME next to the
// badge, and the full text stays one tap away in the advanced Licenses screen.

// Matches "non-commercial", "non commercial", "noncommercial" and the SPDX-style
// "-NC-" / "NC-4.0" markers, case-insensitively.
const NON_COMMERCIAL_PATTERN = /non[\s-]?commercial|\bnc-\d|-nc-|\bcc-by-nc\b/i;

/**
 * True when a licence string restricts commercial use.
 * @param {string|null|undefined} license
 */
export function licenseIsNonCommercial(license) {
  return NON_COMMERCIAL_PATTERN.test(String(license ?? ""));
}

/**
 * The badge to render for a licence string.
 * @param {string|null|undefined} license
 * @returns {{ label: string, tone: "ok"|"danger" }}
 */
export function licenseBadge(license) {
  return licenseIsNonCommercial(license)
    ? { label: "Non-commercial", tone: "danger" }
    : { label: "Commercial OK", tone: "ok" };
}

/**
 * The model-licence rows the Simple Licenses screen lists: the bundled components that
 * actually ship MODEL WEIGHTS (`models[]` non-empty), which is what the design's
 * "MODEL LICENSES" group means. The binary components (FFmpeg, ONNX Runtime, the CUDA
 * runtime) are tooling, not model terms, and stay out — they remain in the advanced
 * Licenses screen, which lists the whole corpus with full text.
 * @param {Array<object>} components - `bundledLicenses`.
 */
export function modelLicenseRows(components) {
  return (components ?? [])
    .filter((component) => Array.isArray(component?.models) && component.models.length > 0)
    .map((component) => ({
      id: component.id,
      name: component.name,
      license: component.license ?? "",
      badge: licenseBadge(component.license),
    }));
}
