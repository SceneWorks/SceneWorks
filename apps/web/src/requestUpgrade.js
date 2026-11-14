import metadata from "./optionUpgradeMetadata.json";

// Aliases are generated from the Rust OptionSpec registry. This adapter only copies carriers;
// it never resolves defaults or makes model/route availability decisions.
export class UpgradeError extends Error {
  constructor(code, field, message) {
    super(`${field}: ${message}`);
    this.code = code;
    this.field = field;
  }
}
const object = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
function version(value, key, supported, path) {
  if (!Object.hasOwn(value, key)) return;
  const actual = value[key];
  if (!Number.isSafeInteger(actual) || actual < 0) throw new UpgradeError("InvalidSchemaVersion", `${path}${key}`, "Expected a positive integer schema version");
  if (actual === 0 || actual > supported) throw new UpgradeError("UnsupportedSchemaVersion", `${path}${key}`, `Schema version ${actual} is unsupported; this reader supports through ${supported}`);
}
function equal(left, right) {
  if (left === right) return true;
  if (Array.isArray(left) && Array.isArray(right)) return left.length === right.length && left.every((v, i) => equal(v, right[i]));
  return object(left) && object(right) && Object.keys(left).length === Object.keys(right).length && Object.keys(left).every((k) => Object.hasOwn(right, k) && equal(left[k], right[k]));
}
function aliases(value, location, path) {
  for (const rule of metadata.aliases.filter((rule) => rule.location === location)) {
    const names = [rule.key, ...rule.aliases].filter((key) => Object.hasOwn(value, key));
    if (!names.length) continue;
    if (names.some((name) => !equal(value[name], value[names[0]]))) throw new UpgradeError("ConflictingAliases", `${path}${rule.key}`, `${names.join(" and ")} contain different authored values`);
    value[rule.key] = value[names[0]];
  }
}
function walk(value, path = "", depth = 0) {
  if (depth > 64) throw new UpgradeError("CarrierTooDeep", path, "Option carrier exceeds 64 nested levels");
  if (Array.isArray(value)) { value.forEach((v, i) => walk(v, `${path}${i}.`, depth + 1)); return; }
  if (!object(value)) return;
  version(value, "optionsSchemaVersion", metadata.optionsSchemaVersion, path);
  if (Object.hasOwn(value, "planVersion") && Object.hasOwn(value, "target") && Object.hasOwn(value, "dataset")) {
    version(value, "planVersion", 1, path);
    version(value, "schemaVersion", 1, path);
  }
  for (const [key, child] of Object.entries(value)) {
    const childPath = `${path}${key}.`;
    if (Object.hasOwn(metadata.recipeVersions, key) && object(child)) {
      if (!Object.hasOwn(child, "version")) throw new UpgradeError("MissingRecipeVersion", `${childPath}version`, `${key} requires an explicit recipe version`);
      version(child, "version", metadata.recipeVersions[key], childPath);
    } else if (["advanced", "rawAdapterSettings"].includes(key)) {
      if (object(child)) aliases(child, "advanced", childPath);
      walk(child, childPath, depth + 1);
    } else if (key === "payload") {
      if (object(child)) aliases(child, "image", childPath);
      walk(child, childPath, depth + 1);
    } else if (["plan", "normalizedSettings", "settings", "recipe", "config", "model", "shots", "scenes", "segments", "generationSet"].includes(key)) walk(child, childPath, depth + 1);
  }
}
export function upgradeCarrier(input, carrier = "request") {
  if (!object(input)) throw new UpgradeError("InvalidCarrier", "", "Expected an object");
  const versions = { filmPlan: 3, compiledFilmPlan: 7, trainingPlan: 1, shareEnvelope: 2 };
  if (versions[carrier]) version(input, "schemaVersion", versions[carrier], "");
  const result = JSON.parse(JSON.stringify(input));
  if (["request", "job"].includes(carrier)) aliases(result, "image", "");
  if (carrier === "studioSettings") aliases(result, "advanced", "");
  walk(result);
  return result;
}
