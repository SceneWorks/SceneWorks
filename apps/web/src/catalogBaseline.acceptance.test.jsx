import { afterEach, beforeEach, expect, it, vi } from "vitest";
vi.mock("./api.js", async (original) => ({ ...await original(), apiFetch: vi.fn(async () => ({})) }));
import old from "./testUtils/catalogBaseline/old-catalog.json";
import behavior from "./testUtils/catalogBaseline/old-behavior.json";
import { click, contextFor, mount, observe, submissionButton } from "./testUtils/catalogBaseline/harness.jsx";
import { defaultTierSelection, installedTiers } from "./quantTier.js";
import { appearanceOptions } from "./data/appearanceCatalog.js";

beforeEach(() => {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  vi.spyOn(Date, "now").mockReturnValue(1794000000000);
  vi.spyOn(Math, "random").mockReturnValue(0.25);
  vi.stubGlobal("fetch", vi.fn(() => { throw new Error("Catalog acceptance must not fetch or execute a model"); }));
});
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

it("records actual full Video requests for Render clip", async () => {
  const result = await observe(old.catalog, { platform: "linux", studio: "video", model: "wan_2_2", mode: "text_to_video" });
  expect(result.submission.available).toBe(true);
  expect(result.submission.disabled).toBe(false);
  expect(result.submission.calls[0]).toMatchObject({ kind: "video", payload: { model: "wan_2_2", mode: "text_to_video" } });
  expect(result.changes.some((change) => change.calls.some(({ kind }) => kind === "video"))).toBe(true);
});

it("records actual full Training requests from its type=button action", async () => {
  const result = await observe(old.catalog, { platform: "linux", studio: "training", model: "anima_base_lora", mode: "anima_lora" });
  expect(result.submission.available).toBe(true);
  expect(result.submission.disabled).toBe(false);
  expect(result.submission.calls[0]).toMatchObject({ kind: "training", payload: { targetId: "anima_base_lora", dryRun: false } });
  expect(result.changes.some((change) => change.calls.some(({ kind }) => kind === "training"))).toBe(true);
});

it("recognizes the full Video Replace person action rather than its mode tab", async () => {
  const mounted = await mount(old.catalog, { platform: "macos", studio: "video", model: "ltx_2_3", mode: "replace_person" });
  try {
    const action = submissionButton(mounted.container, "video");
    expect(action?.textContent.trim()).toBe("Replace person");
    expect(action?.type).toBe("submit");
    expect(action?.classList.contains("prompt-cta")).toBe(true);
  } finally { await mounted.close(); }
});

it("renders all converted Anima tiers and submits the q8 default", async () => {
  const row = { platform: "macos", studio: "image", model: "anima_base", mode: "text_to_image" };
  const { context } = contextFor(old.catalog, row);
  const model = context.imageModels[0];
  expect(installedTiers(model)).toEqual(["q4", "q8", "bf16"]);
  expect(defaultTierSelection(model, null)).toBe("q8");
  const mounted = await mount(old.catalog, row);
  try {
    const picker = mounted.container.querySelector(".quant-tier-picker select");
    expect(picker).toBeTruthy();
    expect([...picker.options].map(({ value }) => value)).toEqual(["q4", "q8", "bf16"]);
    expect(picker.value).toBe("q8");
    await click([...mounted.container.querySelectorAll("button")].find((node) => node.textContent.trim() === "Generate"));
    expect(mounted.calls[0].payload.advanced.mlxQuantize).toBe(8);
  } finally { await mounted.close(); }
});

it("restores every appearance axis from a recipe into an independent conflicting draft", async () => {
  const witness = behavior.rows["macos/moss_image_base/appearance"];
  const recipe = witness.carrier;
  const expected = recipe.rawAdapterSettings.appearance;
  const model = old.catalog.models.find(({ id }) => id === recipe.model);
  const alternative = (axis, selected) => appearanceOptions(model, axis).find(({ id }) => !selected.includes(id)).id;
  const conflict = { camera: alternative("camera", [expected.camera]), film: alternative("film", [expected.film]),
    looks: [alternative("look", expected.looks)], lighting: alternative("lighting", [expected.lighting]) };
  for (const axis of ["camera", "film", "looks", "lighting"]) expect(conflict[axis]).not.toEqual(expected[axis]);
  const mounted = await mount(old.catalog, { platform: "macos", studio: "image", model: recipe.model, mode: "text_to_image" },
    { appearance: conflict, prompt: "An unrelated destination draft", batchMode: false, batchPromptsText: "unrelated row", count: 4 },
    { studioLaunch: { id: "independent-appearance-import", view: "Image", recipe } });
  try {
    expect(JSON.parse(localStorage.getItem(mounted.persistenceKey)).appearance).toEqual(expected);
    expect(mounted.calls).toHaveLength(0);
    await click([...mounted.container.querySelectorAll("button")].find((node) => node.textContent.includes("Run batch")));
    expect(mounted.calls[0].payload.appearance).toEqual(expected);
    expect(mounted.calls.flatMap(({ payload }) => payload.prompts ?? [payload.prompt])).toEqual(witness.original[0].payload.prompts);
  } finally { await mounted.close(); }
});
