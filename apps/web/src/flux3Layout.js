// Attachment identity is independent of asset identity and native ref_image_N ordinals.
// Matching occurrences preserves duplicate inputs; explicit rail moves move the slot itself.
export function reconcileLayoutReferences(previous = [], assetIds = [], ids = null) {
  const remaining = [...previous];
  return assetIds.map((assetId, index) => {
    const match = remaining.findIndex((slot) => slot.assetId === assetId);
    const existing = match < 0 ? null : remaining.splice(match, 1)[0];
    return { assetId, id: ids?.[index] ?? existing?.id ?? `attachment_${crypto.randomUUID()}` };
  });
}

export function isFlux3LayoutModel(model) {
  return ["flux3_klein", "flux3_klein_base"].includes(typeof model === "string" ? model : model?.id ?? "");
}

export function orderedLayoutAssetIds(references) {
  return [references.sourceAssetId, ...(references.referenceAssetIds ?? []), references.referenceAssetId].filter(Boolean);
}

// An explicit attachment edit upgrades ordinal-only imported controls against the OLD
// sequence. Loading/selecting a native document never changes it; its authored from stays intact.
export function bindLayoutReferenceIds(layout, slots) {
  return { ...layout, boxes: layout.boxes.map((box) => {
    if (Array.isArray(box.bbox) || box.reference_id) return box;
    const index = /^ref_image_(\d+)$/.exec(box.from ?? "")?.[1];
    const slot = index == null ? null : slots[Number(index)];
    return slot ? { ...box, reference_id: slot.id } : box;
  }) };
}

export function moveLayoutReference(slots, from, to) {
  const next = [...slots];
  next.splice(to, 0, ...next.splice(from, 1));
  return next;
}
