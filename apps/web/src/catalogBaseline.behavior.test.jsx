import { renderBaseline } from "../../../scripts/rearchitecture/baseline-json.mjs";
import { readFileSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { currentCandidate } from "./testUtils/catalogBaseline/candidate.js";
import { act } from "react";
import { beforeAll, describe, expect, it, vi } from "vitest";
vi.mock("./api.js", async (original) => ({ ...await original(), apiFetch: vi.fn(async () => ({})) }));
import old from "./testUtils/catalogBaseline/old-catalog.json";
import { mount, click, change, controls } from "./testUtils/catalogBaseline/harness.jsx";
import { appearanceOptions } from "./data/appearanceCatalog.js";
const path = "src/testUtils/catalogBaseline/old-behavior.json";
const capture = process.env.CAPTURE_OLD_CATALOG === old.source.revision;
const expected = capture ? {} : JSON.parse(readFileSync(path, "utf8"));
const observed = {};
const candidate = currentCandidate();
const text = (node) => node.textContent.trim();
const button = (container, label) => [...container.querySelectorAll("button")].find((node) => text(node) === label);
const snapshots = (name, value) => { if (capture) observed[name] = value; else expect(value).toEqual(expected.rows[name]); };
beforeAll(() => {
  if (capture) {
    expect(execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim()).toBe(old.source.revision);
    expect(execFileSync("git", ["diff", old.source.revision, "--", ...Object.keys(old.source.trees)], { encoding: "utf8", maxBuffer: 4 * 1024 * 1024 })).toBe("");
  }
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  vi.spyOn(Date, "now").mockReturnValue(1794000000000);
  vi.spyOn(Math, "random").mockReturnValue(0.25);
  vi.stubGlobal("fetch", vi.fn(() => { throw new Error("Behavior baseline must not fetch or execute a model"); }));
});
describe("old public request behavior (SW-7028)", () => {
  for (const platform of Object.keys(old.catalog.platforms)) {
    for (const id of ["z_image_turbo", "z_image", "krea_2_turbo", "krea_2_raw", "moss_image_base", "moss_image_sft"]) {
      it(`${platform}/${id}: appearance menus, authored/effective batch and re-import`, async () => {
        const row = { platform, studio: "image", model: id, mode: "text_to_image" };
        const catalog = capture ? old.catalog : candidate;
        const mounted = await mount(catalog, row, { batchMode: true, batchPromptsText: "a fox\na cat", count: 1 });
        let original, saved;
        const menus = [];
        try {
          for (const [axis, names] of [["Camera", ["Disposable", "Digital compact"]], ["Film", ["Kodak Portra 400"]], ["Looks", ["Warm golden", "Matte faded film"]], ["Lighting", ["Golden hour"]]]) {
            // Use the old catalog's option names as the population: a removed candidate entry must
            // fail its interaction, rather than silently producing a smaller comparison.
            for (const [index, name] of names.entries()) {
              await click(mounted.container.querySelector(`button[aria-label="${axis}"]`));
              const search = mounted.container.querySelector('input[type="search"]');
              expect(search).toBe(document.activeElement);
              menus.push({ axis, stage: "groups", controls: controls(mounted.container.querySelector('[role="dialog"]'), mounted.persistenceKey) });
              if (axis === "Lighting") {
                await click(button(mounted.container, "Outdoor daylight"));
                menus.push({ axis, stage: "group", controls: controls(mounted.container.querySelector('[role="dialog"]'), mounted.persistenceKey) });
                await click(button(mounted.container, "Groups"));
              }
              const available = appearanceOptions(old.catalog.models.find((m) => m.id === id), axis === "Looks" ? "look" : axis.toLowerCase());
              const targetName = available.some((entry) => entry.name === name) ? name : available[index]?.name;
              await change(search, targetName);
              const option = [...mounted.container.querySelectorAll('.appearance-option')].find((node) => node.querySelector("strong")?.textContent === targetName);
              expect(option, `${id}: ${axis} ${targetName}`).toBeTruthy();
              await click(option.querySelector("input"));
              expect(mounted.calls).toHaveLength(0);
              if (axis === "Looks") {
                expect(mounted.container.querySelector('[role="dialog"]')).toBeTruthy();
                await click(button(mounted.container, "Done"));
              } else expect(mounted.container.querySelector('[role="dialog"]')).toBeNull();
            }
          }
          const selection = JSON.parse(localStorage.getItem(mounted.persistenceKey)).appearance;
          expect(selection.looks).toHaveLength(2);
          await click([...mounted.container.querySelectorAll("button")].find((node) => text(node).includes("Run batch")));
          expect(mounted.calls.length).toBeGreaterThan(0);
          original = JSON.parse(JSON.stringify(mounted.calls));
          saved = JSON.parse(localStorage.getItem(mounted.persistenceKey));
        } finally { await mounted.close(); }
        // Restore the public recipe carrier used by workflow/PNG imports and regenerate through
        // the real studio. Native packed and one-job-per-prompt batches retain their own shapes.
        const first = original[0].payload;
        const recipe = {
          model: id, mode: "text_to_image", prompt: first.prompt,
          normalizedSettings: { count: 1 },
          rawAdapterSettings: { ...first.advanced, ...(first.prompts ? { promptBatch: first.prompts } : {}) },
        };
        const replay = await mount(catalog, row, saved, { studioLaunch: { id: "baseline-workflow-import", view: "Image", recipe } });
        try {
          await click([...replay.container.querySelectorAll("button")].find((node) => text(node).includes("Run batch")) ?? button(replay.container, "Generate"));
          const effective = (calls) => calls.flatMap(({ payload }) => payload.prompts ?? [payload.prompt]);
          // An unpacked recipe represents one job, so importing it restores its single authored
          // row; packed carriers restore the complete authored batch.
          expect(effective(replay.calls)).toEqual(first.prompts ?? [first.prompt]);
          snapshots(`${platform}/${id}/appearance`, { menus, original, carrier: recipe, restored: JSON.parse(localStorage.getItem(replay.persistenceKey)), replay: replay.calls });
        } finally { await replay.close(); }
      });
    }
    for (const model of old.catalog.models.filter((entry) => ["image", "video"].includes(entry.type))) {
      const modes = model.type === "image" ? ["text_to_image", "edit_image", "character_image"] : ["text_to_video", "image_to_video", "first_last_frame", ...(model.capabilities ?? [])];
      const mode = modes.find((item) => model.capabilities?.includes(item));
      if (!mode) continue;
      it(`${platform}/${model.id}: replayed invalid provider sampling overrides`, async () => {
        const mounted = await mount(capture ? old.catalog : candidate, { platform, studio: model.type, model: model.id, mode }, {
          sampler: "baseline-unsupported-sampler", scheduler: "baseline-unsupported-scheduler",
          schedulerShift: -1, schedulerShiftExplicit: true, stepsOverride: -1, guidanceOverride: -1,
        });
        try {
          for (const toggle of [...mounted.container.querySelectorAll(".advanced-section-toggle")]) if (toggle.getAttribute("aria-expanded") !== "true") await click(toggle);
          const rendered = controls(mounted.container, mounted.persistenceKey).filter((control) => /sampler|scheduler|schedule shift|guidance|steps/i.test(control.label));
          const submit = [...mounted.container.querySelectorAll('button[type="submit"]')].find((node) => text(node).startsWith("Generate"));
          if (submit && !submit.disabled) await click(submit);
          snapshots(`${platform}/${model.id}/sampling-override`, { rendered, calls: mounted.calls, submitDisabled: submit?.disabled ?? null,
            hints: [...mounted.container.querySelectorAll('.field-hint,[role="alert"],.form-error')].map(text) });
        } finally { await mounted.close(); }
      });
    }
    it(`${platform}: file import, source identity and edit submission`, async () => {
      const row = { platform, studio: "image", model: "sdxl", mode: "edit_image" };
      const mounted = await mount(capture ? old.catalog : candidate, row, { sourceAssetId: "", referenceAssetId: "", referenceAssetIds: [] }, { assets: [] });
      try {
        const select = [...mounted.container.querySelectorAll(".asset-picker-head button")].find((node) => text(node) === "Select image");
        await click(select);
        const dialog = document.body.querySelector('[role="dialog"]');
        await click([...dialog.querySelectorAll('[role="tab"]')].find((node) => text(node) === "File Upload"));
        const input = dialog.querySelector('input[type="file"]');
        const file = new File(["image fixture"], "source.png", { type: "image/png" });
        await act(async () => { Object.defineProperty(input, "files", { configurable: true, value: [file] }); input.dispatchEvent(new Event("change", { bubbles: true })); });
        expect(mounted.context.importAsset).toHaveBeenCalledWith(file, { select: false, throwOnError: true });
        await click(button(mounted.container, "Generate"));
        expect(mounted.calls[0].payload.sourceAssetId).toBe("baseline-image");
        snapshots(`${platform}/file-import`, mounted.calls);
      } finally { await mounted.close(); }
    });
    it(`${platform}: native Iris sampling refuses invalid overrides before submission`, async () => {
      const mounted = await mount(capture ? old.catalog : candidate, { platform, studio: "image", model: "iris_3b", mode: "text_to_image" });
      try {
        const field = (label) => [...mounted.container.querySelectorAll("label")].find((node) => node.textContent.startsWith(label))?.querySelector("input,select");
        const interval = field("Guidance interval");
        expect(interval).toBeTruthy();
        await change(interval, "0.1, 0.9");
        await click(button(mounted.container, "Generate"));
        expect(mounted.calls[0].payload.advanced.cfgInterval).toEqual([0.1, 0.9]);
        const valid = JSON.parse(JSON.stringify(mounted.calls));
        mounted.calls.length = 0;
        await change(interval, "0.9, 0.1");
        await click(button(mounted.container, "Generate"));
        expect(mounted.calls).toHaveLength(0);
        expect(mounted.container.textContent).toContain("Need 0 ≤ start < end ≤ 1");
        snapshots(`${platform}/invalid-sampling`, { valid, invalid: controls(mounted.container, mounted.persistenceKey), calls: mounted.calls });
      } finally { await mounted.close(); }
    });
  }
  it("freezes source-identified behavior observations", () => {
    if (capture) writeFileSync(path, `${renderBaseline({ source: old.source, rows: observed })}\n`);
    else expect(Object.keys(observed).length).toBe(0);
  });
});
