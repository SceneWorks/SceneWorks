// Commercial-use badge for the Simple UI's Licenses screen (design handoff).
//
// The design shows one of two badges per model-licence row: "Commercial OK" (accent) or
// "Non-commercial" (danger). The real corpus (`bundledLicenses`, derived from
// apps/desktop/licenses/manifest.json) carries a free-text `license` string, not a flag —
// so the badge is DERIVED here, in one pure place, rather than duplicated per render site.
//
// There are TWO inputs, in priority order (sc-24108):
//
//   1. `component.nonCommercial === true` — an EXPLICIT declaration on the manifest component.
//      This is authoritative and is checked first.
//   2. The licence NAME, matched against [`NON_COMMERCIAL_PATTERN`] — the historical rule, kept
//      as the fallback for every component that carries no flag.
//
// The name match alone was a real defect, not a hypothetical one. It assumed every restricted
// licence SAYS SO in its title ("FLUX.1 [dev] Non-Commercial License v1.1.1", "Ideogram
// Non-Commercial Model Agreement", "CircleStone Labs Non-Commercial License v1.2", "Research /
// Non-Commercial (CC-BY-NC-4.0 · …)", "Stable Video Diffusion Non-Commercial Community
// License") — and that held until the "Qwen RESEARCH LICENSE AGREEMENT" arrived, whose title
// contains no such marker while §1(i) defines "Non-Commercial" as research or evaluation only
// and §2(a) grants rights FOR NON-COMMERCIAL PURPOSES ONLY. Name-matching badged it
// "Commercial OK". A licence that restricts commercial use without saying so in its title is
// exactly what the flag exists for; reach for it rather than widening the regex, which cannot
// be made to read a licence it has not been shown.
//
// Everything with neither signal keeps the permissive badge, which matches the design's own
// mapping (it badges the Stability AI Community License as "Commercial OK").
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
 * The badge to render for a bundled licence component — or, for callers that only hold the
 * licence string, for that string.
 *
 * A component's explicit `nonCommercial: true` WINS over the name match, and is the only way a
 * licence whose title carries no marker (the Qwen RESEARCH LICENSE AGREEMENT) can be badged
 * correctly. A component without the flag falls through to matching its `license` name, so every
 * pre-existing component keeps the badge it had.
 *
 * @param {object|string|null|undefined} component - a `bundledLicenses` entry, or a licence string.
 * @returns {{ label: string, tone: "ok"|"danger" }}
 */
export function licenseBadge(component) {
  const restricted =
    component !== null && typeof component === "object"
      ? component.nonCommercial === true || licenseIsNonCommercial(component.license)
      : licenseIsNonCommercial(component);
  return restricted
    ? { label: "Non-commercial", tone: "danger" }
    : { label: "Commercial OK", tone: "ok" };
}

/**
 * The model-licence rows the Simple Licenses screen lists: the bundled components that
 * actually ship MODEL WEIGHTS (`models[]` non-empty), plus optional model components
 * whose terms compose with named products (`appliesToModels[]` non-empty). The latter
 * stays separate from `models[]` because license coverage treats that field as exclusive
 * primary attribution. Binary components (FFmpeg, ONNX Runtime, the CUDA runtime) have
 * neither relation and stay out — they remain in the advanced Licenses screen, which
 * lists the whole corpus with full text.
 * @param {Array<object>} components - `bundledLicenses`.
 */
export function modelLicenseRows(components) {
  return (components ?? [])
    .filter(
      (component) =>
        (Array.isArray(component?.models) && component.models.length > 0) ||
        (Array.isArray(component?.appliesToModels) && component.appliesToModels.length > 0),
    )
    .map((component) => ({
      id: component.id,
      name: component.name,
      license: component.license ?? "",
      // The whole component, not just its `license` string — so an explicit `nonCommercial`
      // declaration is visible to the badge (sc-24108).
      badge: licenseBadge(component),
    }));
}
