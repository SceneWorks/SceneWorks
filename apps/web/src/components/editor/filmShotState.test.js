import { describe, expect, it } from "vitest";
import { filmRunSummary, filmShotState } from "./filmShotState.js";

const take = { assetId: "asset_1" };

function run(shots, { active = false, selected = [] } = {}) {
  return { controllerActive: active, locator: { id: "run_1", selectedShotIds: selected }, record: { shots } };
}

describe("filmShotState", () => {
  it("reads a delivered take as needing a decision until a person decides", () => {
    const delivered = { shotId: "SH010", attempts: [{ attempt: 1, status: "completed", take }], humanDecision: null };
    expect(filmShotState("SH010", run([delivered]))).toBe("decide");
    expect(filmShotState("SH010", run([{ ...delivered, humanDecision: { state: "accepted" } }]))).toBe("accepted");
    expect(filmShotState("SH010", run([{ ...delivered, humanDecision: { state: "rejected" } }]))).toBe("rejected");
  });

  it("separates the shot in flight from shots waiting behind it, and only while a controller is active", () => {
    const shots = [{ shotId: "SH010", attempts: [{ attempt: 1, status: "running" }] }, { shotId: "SH020", attempts: [] }];
    const active = run(shots, { active: true, selected: ["SH010", "SH020"] });
    expect(filmShotState("SH010", active)).toBe("rendering");
    expect(filmShotState("SH020", active)).toBe("queued");
    expect(filmShotState("SH030", active)).toBe("planned");
    expect(filmShotState("SH020", run(shots, { selected: ["SH010", "SH020"] }))).toBe("planned");
  });

  it("reports a terminal attempt with no take as failed, and no run as planned", () => {
    expect(filmShotState("SH010", run([{ shotId: "SH010", attempts: [{ attempt: 1, status: "timed_out" }] }]))).toBe("failed");
    expect(filmShotState("SH010", null)).toBe("planned");
  });

  it("summarizes a draft against its run in plan order", () => {
    const draft = { productionPlan: { shots: [{ id: "SH010" }, { id: "SH020" }, { id: "SH030" }] } };
    const summary = filmRunSummary(draft, run([
      { shotId: "SH010", attempts: [{ attempt: 1, status: "completed", take }] },
      { shotId: "SH020", attempts: [{ attempt: 1, status: "running" }] },
    ], { active: true, selected: ["SH010", "SH020"] }));
    expect(summary).toMatchObject({ states: ["decide", "rendering", "planned"], delivered: 1, total: 3, decide: 1, renderingShotId: "SH020" });
  });
});
