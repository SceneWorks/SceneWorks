export function isModelMissing(model, message) {
  if (model?.installState) return model.installState === "missing";
  return /not cached|not installed|snapshot is not/i.test(message ?? "");
}
