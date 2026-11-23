// Imported models preserve their identity while reusing a declared base's request policy.
export const modelPolicyId = (model) => typeof model === "object" && model !== null
  ? model.policyModel ?? model.id : model;
export const isMossImage = (model) => ["moss_image_base", "moss_image_sft"].includes(modelPolicyId(model));
export const isMossEdit = (model, mode) => isMossImage(model) && mode === "edit_image";
