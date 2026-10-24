// Pure model identity shared by startup validation and lazy MOSS controls.
export const isMossImage = (id) => id === "moss_image_base" || id === "moss_image_sft";
export const isMossEdit = (model, mode) => isMossImage(model) && mode === "edit_image";
