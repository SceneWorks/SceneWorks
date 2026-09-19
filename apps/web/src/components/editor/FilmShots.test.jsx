import React, { useState } from "react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { FilmShots } from "./FilmShots.jsx";

function makeDraft() {
  const shot = (id, beat) => ({
    id, beat, framing: "wide", prompt: `${beat} prompt`, targetDurationSeconds: 5.1667,
    startState: "start", endState: "end", audio: "Room tone. No music.", conditioning: { mode: "text_to_video", referenceRoles: [] },
    continuityRoles: [], dependsOn: [],
  });
  return {
    id: "film_1",
    productionPlan: {
      schemaVersion: 3, id: "film_1", version: 2, title: "Film", synopsis: "",
      model: { id: "minimax_h3", tier: "q4", fps: 24, resolution: "576x320", loras: [] },
      limits: { maxRunSeconds: 3600, maxShotSeconds: 2700, maxAttemptsPerShot: 1, maxMemoryGb: 96 },
      sound: {}, shots: [shot("SH010", "Arrival"), shot("SH020", "Reveal")],
    },
    referencePack: { references: [
      { role: "hero", approved: true, file: "references/hero.png" },
      { role: "plate", approved: true, file: "references/plate.png" },
      // DESCRIBED-ONLY (sc-24025): approved, but with no image. It belongs in continuityRoles and
      // nowhere else — the server refuses it in every conditioning slot.
      { role: "recipient", approved: true, description: "Grey work apron." },
      { role: "rejected", approved: false, file: "references/rejected.png" },
    ] },
  };
}

function change(element, value) {
  const prototype = element.tagName === "TEXTAREA" ? HTMLTextAreaElement.prototype
    : element.tagName === "SELECT" ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
  Object.getOwnPropertyDescriptor(prototype, "value").set.call(element, value);
  element.dispatchEvent(new Event("input", { bubbles: true }));
  element.dispatchEvent(new Event("change", { bubbles: true }));
}

let container;
let root;
let latest;

beforeEach(() => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  container = document.createElement("div");
  document.body.appendChild(container);
});

afterEach(() => {
  act(() => root?.unmount());
  container.remove();
});

async function render(props = {}) {
  function Harness() {
    const [draft, setDraft] = useState(makeDraft());
    const [selection, setSelection] = useState(["SH010", "SH020"]);
    latest = { draft, selection };
    return <FilmShots
      capabilities={{ modes: ["text_to_video", "reference_to_video"], durations: [5.1667], resolutions: ["576x320"], maxReferenceImages: 3, turboLoras: [{ id: "turbo", name: "Turbo", steps: 4 }] }}
      compiled={null}
      disabled={false}
      draft={draft}
      findings={[]}
      onChange={(mutator) => setDraft((current) => { const next = structuredClone(current); mutator(next); return next; })}
      onImportError={vi.fn()}
      selectedShotIds={selection}
      setSelectedShotIds={setSelection}
      {...props}
    />;
  }
  root = createRoot(container);
  await act(async () => root.render(<Harness />));
}

describe("FilmShots", () => {
  it("edits conditioning, intent, dialogue placement and dependencies while reordering a subset", async () => {
    await render();
    const second = [...container.querySelectorAll(".ve-film-shot-list button")].find((button) => button.textContent.includes("SH020"));
    await act(async () => second.click());
    await act(async () => {
      change(container.querySelector('select[aria-label="Conditioning mode"]'), "reference_to_video");
      change(container.querySelector('input[aria-label="Reference roles"]'), "hero");
      change(container.querySelector('textarea[aria-label="Shot SH020 dialogue"]'), "Open it.");
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Add dependency").click();
    });
    await act(async () => {
      change(container.querySelector('select[aria-label="Dependency 1 shot"]'), "SH010");
      change(container.querySelector('select[aria-label="Dependency 1 kind"]'), "continuity");
      change(container.querySelector('input[aria-label="Dependency 1 note"]'), "Keep the courier screen-left");
      change([...container.querySelectorAll("label")].find((label) => label.textContent.startsWith("Audio role")).querySelector("input"), "line_1");
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Move up").click();
      const renderFirst = container.querySelector('input[aria-label="Render SH010"]');
      renderFirst.click();
    });
    expect(latest.draft.productionPlan.shots.map((shot) => shot.id)).toEqual(["SH020", "SH010"]);
    expect(latest.draft.productionPlan.shots[0]).toMatchObject({
      dialogue: "Open it.", dialogueClip: { role: "line_1" },
      conditioning: { mode: "reference_to_video", referenceRoles: ["hero"] },
      dependsOn: [{ shotId: "SH010", kind: "continuity", note: "Keep the courier screen-left" }],
    });
    expect(latest.selection).toEqual(["SH020"]);
  });

  // sc-24025. `validate_plan_against_pack` refuses a DESCRIBED-ONLY role in every conditioning
  // slot, so offering one in the two keyframe selects or in the reference-role suggestions would
  // let the editor build a draft the server rejects. It belongs in continuityRoles, and that is
  // the one input that suggests it.
  it("offers a described-only role for continuity only and never for conditioning", async () => {
    await render();
    const selectFor = (prefix) => [...container.querySelectorAll("label")]
      .find((label) => label.textContent.startsWith(prefix)).querySelector("select");
    // `option.value` falls back to the text for an option with no value attribute, so this is the
    // role list with the selects' empty "None" entry dropped.
    const optionValues = (element) => [...element.querySelectorAll("option")]
      .map((option) => option.value).filter(Boolean);

    for (const prefix of ["First frame role", "Last frame role"]) {
      expect(optionValues(selectFor(prefix))).toEqual(["hero", "plate"]);
    }
    expect(optionValues(container.querySelector("#film-reference-role-options")))
      .toEqual(["hero", "plate"]);
    expect(optionValues(container.querySelector("#film-continuity-role-options")))
      .toEqual(["hero", "plate", "recipient"]);
  });

  it("shows field findings, effective settings and recorded refinement identity", async () => {
    const compiled = {
      requests: [{ shotId: "SH010", model: "minimax_h3", width: 576, height: 320, fps: 24, durationSeconds: 5.1667, mode: "text_to_video", referenceRoles: [], effectiveSteps: 4, loras: ["turbo"], promptSource: "refined" }],
      planner: { executions: [{ jobId: "job_1", provider: "native", model: "Qwen/Qwen3.6-27B", thinkingMode: "enabled", targetVideoModelId: "minimax_h3" }] },
    };
    await render({ compiled, findings: [{ shotId: "SH010", field: "prompt", message: "prompt is required" }] });
    expect(container.textContent).toContain("prompt is required");
    expect(container.textContent).toContain("576×320");
    expect(container.textContent).toContain("Qwen/Qwen3.6-27B");
    expect(container.textContent).toContain("target minimax_h3");
  });

  // sc-24026. `audio` is required on every shot by plan schema 3, and this textarea is the only
  // place in the workspace it can be written or repaired.
  it("edits the required audio sentence into the shot the draft saves", async () => {
    await render();
    const audio = container.querySelector('textarea[aria-label="Shot SH010 audio"]');
    expect(audio.value).toBe("Room tone. No music.");
    await act(async () => change(audio, "A door latch clicking. No music."));
    expect(latest.draft.productionPlan.shots[0].audio).toBe("A door latch clicking. No music.");
    // The edit lands on the selected shot only.
    expect(latest.draft.productionPlan.shots[1].audio).toBe("Room tone. No music.");

    // Clearing it is allowed in the editor — a blank value is refused by the server as a finding
    // that names the shot, not blocked here, so the user can retype it in place.
    await act(async () => change(container.querySelector('textarea[aria-label="Shot SH010 audio"]'), ""));
    expect(latest.draft.productionPlan.shots[0].audio).toBe("");
  });

  it("shows a shot-named audio finding under the audio field", async () => {
    await render({ findings: [{ shotId: "SH010", field: "audio", message: "audio is required: say what this shot sounds like" }] });
    const audio = container.querySelector('textarea[aria-label="Shot SH010 audio"]');
    const findings = audio.closest("label").nextElementSibling;
    expect(findings?.className).toContain("ve-film-findings");
    expect(findings.textContent).toContain("audio is required: say what this shot sounds like");
  });

  it("imports supported production and compiled documents without a JSON-only editing path", async () => {
    await render();
    const importedPlan = makeDraft().productionPlan;
    importedPlan.shots[0].prompt = "Imported production prompt";
    const planInput = container.querySelector('input[aria-label="Production plan file"]');
    Object.defineProperty(planInput, "files", { configurable: true, value: [{ text: async () => JSON.stringify(importedPlan) }] });
    await act(async () => { planInput.dispatchEvent(new Event("change", { bubbles: true })); await Promise.resolve(); });
    expect(latest.draft.productionPlan.shots[0].prompt).toBe("Imported production prompt");
    expect(latest.draft.renderRegime).toBe("custom");

    const importedCompiled = { schemaVersion: 3, planSha256: "hash", requests: [{ shotId: "SH010" }] };
    const compiledInput = container.querySelector('input[aria-label="Compiled plan file"]');
    Object.defineProperty(compiledInput, "files", { configurable: true, value: [{ text: async () => JSON.stringify(importedCompiled) }] });
    await act(async () => { compiledInput.dispatchEvent(new Event("change", { bubbles: true })); await Promise.resolve(); });
    expect(latest.draft.compiledPlan).toEqual(importedCompiled);
    expect([...container.querySelectorAll("button")].find((button) => button.textContent === "Export compiled plan").disabled).toBe(false);
  });

  it("switches to custom when adapters or steps are edited manually", async () => {
    await render();
    const advanced = [...container.querySelectorAll("details")]
      .find((item) => item.querySelector(":scope > summary")?.textContent.startsWith("Advanced video and budget settings"));
    await act(async () => advanced.querySelector("summary").click());
    await act(async () => change(container.querySelector('input[aria-label="Plan adapters"]'), "turbo"));
    expect(latest.draft.renderRegime).toBe("custom");
    expect(latest.draft.productionPlan.model.loras).toEqual(["turbo"]);

    latest.draft.renderRegime = "quality";
    const steps = [...container.querySelectorAll("label")].find((label) => label.textContent.startsWith("Steps")).querySelector("input");
    await act(async () => change(steps, "8"));
    expect(latest.draft.renderRegime).toBe("custom");
    expect(latest.draft.productionPlan.model.advanced.steps).toBe(8);
  });
});
