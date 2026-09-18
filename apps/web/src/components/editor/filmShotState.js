// Delivery state of one planned shot, read from a film run record. Shared by the Film
// workspace shot list, the cut strip, and the Timeline-mode toolbar chip.
const TERMINAL_FAILURES = new Set(["failed", "canceled", "canceled_by_operator", "interrupted", "timed_out"]);

export const FILM_SHOT_STATE_LABELS = {
  accepted: "Accepted",
  rejected: "Rejected",
  decide: "Needs decision",
  rendering: "Rendering",
  queued: "Queued",
  failed: "Failed",
  planned: "Not rendered",
};

export function filmShotState(shotId, run) {
  const record = run?.record;
  const shot = record?.shots?.find((item) => item.shotId === shotId);
  const active = run?.controllerActive === true;
  const selected = [...(run?.locator?.selectedShotIds ?? []), ...(record?.selectedShotIds ?? [])].includes(shotId);
  if (!shot) return active && selected ? "queued" : "planned";
  const attempts = shot.attempts ?? [];
  const last = attempts.at(-1);
  if (active && last && !last.take && !TERMINAL_FAILURES.has(last.status) && last.status !== "completed" && last.status !== "rejected") return "rendering";
  const delivered = attempts.some((attempt) => attempt.take);
  if (delivered) {
    const decision = String(shot.humanDecision?.state ?? "");
    if (decision.startsWith("accept")) return "accepted";
    if (decision.startsWith("reject")) return "rejected";
    return "decide";
  }
  if (active && selected) return "queued";
  // An automatic QC reject leaves a "rejected" attempt with no take: nothing was delivered.
  if (last && (TERMINAL_FAILURES.has(last.status) || last.status === "rejected")) return "failed";
  return "planned";
}

export function filmRunSummary(draft, run) {
  const shots = draft?.productionPlan?.shots ?? [];
  const states = shots.map((shot) => filmShotState(shot.id, run));
  const delivered = states.filter((state) => state === "accepted" || state === "rejected" || state === "decide").length;
  const renderingIndex = states.indexOf("rendering");
  return {
    states,
    delivered,
    total: shots.length,
    decide: states.filter((state) => state === "decide").length,
    renderingShotId: renderingIndex >= 0 ? shots[renderingIndex].id : null,
  };
}
