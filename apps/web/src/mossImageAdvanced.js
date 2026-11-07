// Project tensor assets contain inert NPY files. These identities describe exact bytes;
// neither replay nor a picker reconstructs values from metadata or substitutes an output.
export const mossAdvancedDefaults = () => ({ output: "images" });
export function mossTensorReferences(options = {}) {
  const refs = [];
  for (const branch of ["positive", "negative"]) {
    if (options[branch]?.values) refs.push([`${branch}.values`, options[branch].values]);
    if (options[branch]?.mask) refs.push([`${branch}.mask`, options[branch].mask]);
  }
  if (options.noise) refs.push(["noise", options.noise]);
  return refs;
}
export function mossTensorRef(asset) {
  if (asset?.type !== "tensor" || !asset.tensor?.sha256) throw new Error("Choose an owned tensor artifact with a measured SHA256.");
  return { assetId: asset.id, sha256: asset.tensor.sha256, metadata: { ...asset.tensor, shape: [...asset.tensor.shape] } };
}
export function mossTensorMatches(asset, reference) {
  return asset?.type === "tensor" && asset.tensor?.sha256 === reference?.sha256 &&
    (!reference.metadata || (asset.tensor.dtype === reference.metadata.dtype && asset.tensor.byteLength === reference.metadata.byteLength && JSON.stringify(asset.tensor.shape) === JSON.stringify(reference.metadata.shape)));
}
export function mossAdvancedWire(options = mossAdvancedDefaults()) {
  if (!options || typeof options !== "object" || Array.isArray(options) || Object.keys(options).some((key) => !["positive", "negative", "negativePrompts", "noise", "output"].includes(key))) throw new Error("Unsupported MOSS edit advanced control.");
  const result = { output: options.output ?? "images" };
  if (!["images", "latents"].includes(result.output)) throw new Error("MOSS output must be images or latents.");
  for (const key of ["positive", "negative"]) {
    if (options[key] == null) continue;
    const value = options[key];
    if (!value.values || Object.keys(value).some((field) => !["values", "mask"].includes(field))) throw new Error(`${key} embeddings require a values artifact and optional mask.`);
    result[key] = { values: { ...value.values }, ...(value.mask ? { mask: { ...value.mask } } : {}) };
  }
  if (options.noise) result.noise = { ...options.noise };
  if (options.negativePrompts != null) {
    if (!Array.isArray(options.negativePrompts) || !options.negativePrompts.length || options.negativePrompts.length > 8 || options.negativePrompts.some((text) => typeof text !== "string" || [...text].length > 4000)) throw new Error("Per-prompt negatives require 1–8 strings of at most 4000 characters.");
    result.negativePrompts = [...options.negativePrompts];
  }
  if (result.negative && result.negativePrompts) throw new Error("Choose negative embeddings or per-prompt negative strings.");
  for (const [role, ref] of mossTensorReferences(result)) {
    if (!ref.assetId || !/^[0-9a-f]{64}$/.test(ref.sha256 ?? "") || Object.keys(ref).some((key) => !["assetId", "sha256", "metadata"].includes(key))) throw new Error(`Reacquire the exact ${role} tensor artifact.`);
  }
  return result;
}
export function reacquireMossTensors(options, requirements = [], picks = {}, assets = []) {
  const result = structuredClone(options);
  for (const requirement of requirements) {
    const asset = assets.find((candidate) => candidate.id === picks[`tensor:${requirement.role}`]);
    if (!mossTensorMatches(asset, requirement)) throw new Error(`Reacquire the exact ${requirement.role} tensor before replay.`);
    const keys = requirement.role.split(".");
    if (keys.length === 1) result[keys[0]] = mossTensorRef(asset);
    else result[keys[0]][keys[1]] = mossTensorRef(asset);
  }
  return mossAdvancedWire(result);
}
