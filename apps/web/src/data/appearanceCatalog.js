import catalog from "../../../../config/manifests/builtin.appearance.json";

export const APPEARANCE_CATALOG = catalog;
export const APPEARANCE_AXES = [
  { key: "camera", label: "Camera" },
  { key: "film", label: "Film" },
  { key: "looks", label: "Looks", axis: "look", multiple: true },
  { key: "lighting", label: "Lighting" },
];
export function emptyAppearance() {
  return { camera: null, film: null, looks: [], lighting: null };
}
export function normalizeAppearance(value) {
  return {
    camera: typeof value?.camera === "string" && value.camera ? value.camera : null,
    film: typeof value?.film === "string" && value.film ? value.film : null,
    looks: [...new Set(Array.isArray(value?.looks) ? value.looks.filter((id) => typeof id === "string" && id) : [])],
    lighting: typeof value?.lighting === "string" && value.lighting ? value.lighting : null,
  };
}
export function hasAppearance(value) {
  return Boolean(value?.camera || value?.film || value?.lighting || value?.looks?.length);
}
export function appearanceProfile(model) {
  if (model?.captionStyle === "tags" || model?.structuredPrompt || model?.captionStyle === "structured") return null;
  const id = typeof model === "string" ? model : model?.id;
  return Object.keys(catalog.profiles).find((key) => catalog.profiles[key].models.includes(id) || catalog.profiles[key].families.includes(model?.family)) ?? null;
}
export function appearanceOptions(model, axis) {
  const profile = appearanceProfile(model);
  return profile ? catalog.entries.filter((entry) => entry.axis === axis && entry.variants[profile]).map((entry) => ({ ...entry, note: entry.variants[profile].note ?? entry.note })) : [];
}
export function pruneAppearance(value, model) {
  const next = normalizeAppearance(value);
  const removed = [];
  for (const { key, axis = key, multiple } of APPEARANCE_AXES) {
    const allowed = new Set(appearanceOptions(model, axis).map((entry) => entry.id));
    if (multiple) {
      removed.push(...next[key].filter((id) => !allowed.has(id)));
      next[key] = next[key].filter((id) => allowed.has(id));
    } else if (next[key] && !allowed.has(next[key])) {
      removed.push(next[key]);
      next[key] = null;
    }
  }
  return { selection: next, removed };
}
export function resolveAppearance(value, model) {
  const selection = normalizeAppearance(value);
  if (!hasAppearance(selection)) return { selection, directives: {}, negative: "", entries: [] };
  const profile = appearanceProfile(model);
  if (!profile) throw new Error("Camera, Film, Looks and Lighting recipes are unavailable for this model.");
  const entries = [];
  const directives = {};
  for (const { key, axis = key } of APPEARANCE_AXES) {
    const ids = key === "looks" ? selection.looks : selection[key] ? [selection[key]] : [];
    const prompts = [];
    for (const id of ids) {
      const entry = catalog.entries.find((item) => item.id === id && item.axis === axis);
      const variant = entry?.variants[profile];
      if (!variant) throw new Error(`Appearance recipe ${id} is unavailable for this model.`);
      entries.push({ ...entry, note: variant.note ?? entry.note, variant });
      prompts.push(variant.prompt);
    }
    if (prompts.length) directives[key === "looks" ? "Look" : key[0].toUpperCase() + key.slice(1)] = prompts.join(" ");
  }
  const negatives = entries.map(({ variant }) => variant.negative).filter(Boolean);
  return { selection, directives, entries, negative: catalog.profiles[profile].negative ? [...new Set(negatives)].join(", ") : "" };
}
export function appearanceWarnings(resolved, userPrompt = "") {
  const warnings = resolved.entries.map((entry) => entry.note).filter(Boolean);
  if (resolved.entries.some((entry) => entry.monochrome) && /\b(red|green|blue|yellow|purple|orange|pink|colorful|colourful)\b/i.test(userPrompt)) {
    warnings.unshift("Your scene names colors that can compete with the monochrome recipe. Your wording is preserved.");
  }
  if (resolved.selection.looks.some((id) => /warm|golden/.test(id)) && resolved.selection.looks.some((id) => /cool|teal/.test(id))) {
    warnings.unshift("The selected warm and cool looks may pull the color treatment in different directions.");
  }
  return [...new Set(warnings)];
}
export function composeNegativePrompt(userPrompt = "", recipeNegative = "") {
  if (!recipeNegative) return userPrompt;
  return userPrompt?.trim() ? `${userPrompt}, ${recipeNegative}` : recipeNegative;
}
