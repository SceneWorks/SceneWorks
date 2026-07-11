// The app-wide form-validation vocabulary (epic 10644, sc-10645). One issue type,
// one summarize(), so a CTA's `disabled` and the messages it owes the user come from
// the same call and cannot drift apart.
//
// That drift is not hypothetical. sc-10492 deleted a screen's validation messages and
// left its `disabled` expression untouched, so clearing a number silently killed
// `Start training` with nothing on screen to say why; sc-10501 restored them. The
// same defect is still live in PaletteEditor and ImageEditor (sc-10653).
//
// Three kinds, and `blocking` / `surfaced` are DERIVED from the kind rather than
// stored beside it — a stored flag is a flag that can disagree with its kind.
//
//   requirement — a field the user hasn't filled in. Blocks. Stays silent: which
//                 fields are empty is plain from the form, and a chip reading
//                 "Name the output" beside an empty Name box is noise (sc-10492).
//   error       — a value the user actively broke, or a condition nothing on the
//                 form explains. Blocks, and says so: otherwise the dead CTA has
//                 no stated reason.
//   advisory    — worth saying, doesn't block. Dataset readiness "needs attention".
//
// The fourth cell of that 2x2 — doesn't block, doesn't surface — is not an issue at
// all, which is why three kinds is the complete set and not a convenient subset.
//
// Issues carry a `field` so a broken value can mark its own input: the training
// config grid holds ~25 numeric inputs, and a chip naming "Rank" down by the
// actions row cannot point at the Rank box (epic 10644, R5). Form-scoped issues
// that belong to no single input pass `null`.

// Constructors live on a namespace rather than as free functions: `error` as a bare
// import is shadowed by a `catch (error)` binding, which three of this epic's own
// target screens use. `issue.error(...)` names the kind and cannot collide.
export const issue = {
  requirement: (field, message) => ({ kind: "requirement", field, message }),
  error: (field, message) => ({ kind: "error", field, message }),
  advisory: (field, message) => ({ kind: "advisory", field, message }),
};

export function isBlocking(item) {
  return item.kind !== "advisory";
}

export function isSurfaced(item) {
  return item.kind !== "requirement";
}

// Roll a rule set's issues into what a form needs: whether its CTA may fire, what to
// say, and which inputs to mark.
//
//   ready        — over EVERY issue, silent requirements included.
//   surfaced     — the messages, in rule order. The ONE message channel: a chip row,
//                  rendered against the form's actions so the chips read as the reason
//                  the CTA is dead (sc-10501).
//   invalidFields — the inputs to outline. A Set of field names, no messages.
//
// `invalidFields` is a Set and not a `Map<field, Issue[]>` on purpose. Two containers
// both holding the same messages is two views that can disagree about who renders an
// issue — a field-scoped error would appear as a chip AND under its input, and nothing
// in the API would say which owns it. That is the same defect this module exists to
// prevent for `ready` (R2), and it must not re-enter through the render path. Messages
// live in exactly one place; a Set of strings cannot be rendered as text.
//
// So a broken field looks broken (R5) by being outlined, while the chip names it. The
// mark and the message can never contradict each other because there is only one of
// each.
//
// Only errors mark their field. A requirement is an untouched empty box, and a fresh
// form must not paint itself red before the user has typed; an advisory doesn't block,
// so outlining its input would overstate it.
export function summarize(issues) {
  const list = issues ?? [];
  const invalidFields = new Set();
  for (const item of list) {
    if (item.kind === "error" && item.field != null) {
      invalidFields.add(item.field);
    }
  }
  return {
    ready: !list.some(isBlocking),
    surfaced: list.filter(isSurfaced),
    invalidFields,
  };
}
