// The edit LoRA section's "+ Add" affordance (epic 10644 / sc-10653).
//
// `+ Add` disables when there is no next compatible LoRA to add, or the per-edit cap is
// reached. Before this it dimmed with no stated reason — the sc-10492 defect. This
// returns the note to show once at least one LoRA is applied, or null when Add is live
// (or the section is empty and its own empty-state note speaks).
//
// The note is neutral, not a validation error: a reached limit or an exhausted list is
// not a value the user broke. So it stays local to this section rather than joining the
// validation core, whose surfaced kinds are danger (error) and amber (advisory) — neither
// fits a neutral "you've added them all" (epic 10644, R6 boundary).
export function loraAddHint({ selectedCount, hasNext, max }) {
  const addDisabled = !hasNext || selectedCount >= max;
  if (!addDisabled || selectedCount === 0) {
    return null;
  }
  return selectedCount >= max ? `Up to ${max} LoRAs per edit.` : "No more compatible LoRAs to add.";
}
