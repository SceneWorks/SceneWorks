// The CTA gate on "Use this recipe" for a shared workflow (sc-15952, epic 15945).
//
// The app-wide validation vocabulary (`validation/issues.js`, epic 10644), used for the same
// reason Image Studio's own Generate uses it: the button's `disabled` and the sentence it owes the
// user come out of ONE call, so a dead CTA can never be dead without a stated reason. This is not a
// second gate beside the studio's — it is the same mechanism applied one step earlier, at the point
// where the recipe is handed over.
//
// # What blocks, and what does not
//
// `RequirementState::blocks_replay()` is true for every state but `resolved`, and that is the right
// answer to "is this a one-click replay?". It is the WRONG rule for a button, because two of those
// states can never be cleared from here:
//
// * a **user-trained LoRA** the sender made is not on Hugging Face and not in any catalog — no
//   download exists, so blocking on it would make "Use this recipe" permanently dead for exactly
//   the recipes this feature was built to rescue;
// * an **omitted collection** is a fact about the file, frozen at the moment it was written.
//
// And one can only be cleared AFTER the hand-over: a mask is painted in the Image Editor and a
// control map is attached in the studio's Control panel, so blocking on either would refuse to
// perform the step its own message tells the user to take first.
//
// A permanently-dead CTA is not a safety property; it is a feature that does not work, and the user
// reaches for the prompt with a screenshot instead. So blocking is reserved for the requirements a
// user can actually resolve in front of the panel — the model, and the source/reference images the
// pickers offer — and everything else is SURFACED as an advisory that says plainly what the replay
// will not carry, or where the missing piece is supplied. Nothing is silent either way.

import { issue } from "./validation/issues.js";
import { workflowInputSlots } from "./workflowShare.js";

export function workflowReplayValidation({
  report = null,
  // The model id the user chose in the substitute picker, or "" for "not chosen". NEVER defaulted
  // to anything: a default here would be the silent substitution the whole epic exists to prevent.
  substituteModelId = "",
  // `{ [slotKey]: assetId }` from the input-image pickers.
  inputPicks = {},
} = {}) {
  const issues = [];
  if (!report) {
    // No report means this install was never checked against the recipe. Prefilling from an
    // unchecked envelope would seed a model id the studio's own picker may drop on the next render,
    // which is the silent loss the adapter refuses — so the CTA waits.
    issues.push(
      issue.error(
        "report",
        "This install has not been checked against the recipe yet, so nothing can be applied from it.",
      ),
    );
    return issues;
  }

  const model = report.model ?? null;
  if (model && model.state !== "resolved" && !substituteModelId) {
    // Two different sentences, because the user's next move is different. An installable model has
    // a download button beside it; an unknown one has only the picker.
    issues.push(
      issue.error(
        "model",
        model.state === "installable"
          ? `${model.name ?? model.slug} is not downloaded here. Install it, or choose a model to use instead — nothing is substituted for you.`
          : `This install has no model matching \`${model.slug}\`. Choose one to use instead — nothing is substituted for you, and the image will differ from the one you were sent.`,
      ),
    );
  }

  // Every image the recipe needs, counted the way the report counts them. A picker with nothing in
  // it is a requirement, not an error: the empty control is its own message, and a chip repeating
  // "pick a source image" beside an obviously empty picker is the noise sc-10492 removed.
  //
  // The UNPICKABLE kinds go the other way. A mask is painted in the Image Editor and a control map
  // is attached in the studio's own Control panel — both AFTER the recipe has been loaded — so
  // blocking on them would refuse to perform the step its own sentence tells the user to take
  // first. They are surfaced, loudly, and the recipe is handed over.
  for (const slot of workflowInputSlots(report)) {
    if (!slot.pickable) {
      issues.push(
        issue.advisory(
          `input:${slot.key}`,
          `${slot.detail || `Needs a ${slot.kind}.`} ${slot.elsewhere}`.trim(),
        ),
      );
    } else if (!inputPicks[slot.key]) {
      issues.push(issue.requirement(`input:${slot.key}`, `Choose ${slot.label.toLowerCase()}`));
    }
  }

  // The advisories: everything the replay will not carry that no action here can change. Each says
  // what the user will get instead, because "LoRA missing" without a consequence reads as a warning
  // about the file rather than about the image they are about to make.
  for (const lora of report.loras ?? []) {
    if (lora.state === "missing") {
      issues.push(
        issue.advisory(
          "lora",
          `${lora.name ?? lora.repo ?? "An unnamed LoRA"} is not on this install and will not be applied, so the image will differ from the one you were sent.`,
        ),
      );
    } else if (lora.state === "installable") {
      issues.push(
        issue.advisory(
          "lora",
          `${lora.name ?? lora.repo} is not downloaded yet and will not be applied until it is.`,
        ),
      );
    }
  }
  for (const style of report.styles ?? []) {
    if (style.state !== "resolved") {
      issues.push(issue.advisory("style", style.detail));
    }
  }
  for (const entry of report.omitted ?? []) {
    issues.push(issue.advisory("omitted", entry.detail));
  }
  return issues;
}

// Whether the substitute picker's value is one the panel may keep.
//
// A model chosen and then uninstalled (or a catalog refetch that drops the row) must not stay
// selected: the recipe would carry an id the studio's own list no longer offers, which is the blank
// control the adapter refuses to produce. Re-checked on every render against the live list rather
// than trusted once at pick time — the whole point of this panel is that the catalog changes under
// it while it is open.
export function keptSubstituteModelId(substituteModelId, models) {
  if (!substituteModelId) {
    return "";
  }
  return (models ?? []).some((model) => model.id === substituteModelId) ? substituteModelId : "";
}

// The reference-image ids the user picked, in slot order, ready for the recipe.
//
// Slot order is the order the recipe's `referenceAssetIds` are sent in, and the studio's
// multi-reference lane treats that order as meaningful (image 1, image 2), so it is preserved
// rather than being whatever `Object.values` happens to yield.
export function pickedReferenceAssetIds(report, inputPicks = {}) {
  return workflowInputSlots(report)
    .filter((slot) => slot.kind === "reference")
    .map((slot) => inputPicks[slot.key])
    .filter(Boolean);
}

// The source image the user picked, or "" — the launch's `sourceAssetId`, which is a LAUNCH field
// and not a recipe one, so it does not travel through `recipeFromWorkflowShare`.
export function pickedSourceAssetId(report, inputPicks = {}) {
  const slot = workflowInputSlots(report).find((entry) => entry.kind === "source");
  return (slot && inputPicks[slot.key]) || "";
}
