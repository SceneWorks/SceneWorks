import { afterEach, beforeEach, expect, it, vi } from "vitest";
vi.mock("./api.js", async (original) => ({ ...await original(), apiFetch: vi.fn(async () => ({})) }));
import { currentCandidate } from "./testUtils/catalogBaseline/candidate.js";
import { change, click, contextFor, mount, observe, submissionButton } from "./testUtils/catalogBaseline/harness.jsx";
import { defaultTierSelection, installedTiers } from "./quantTier.js";

// Real studio screens mounted against the live catalog projection, driving actual handlers. These
// were the behavioural checks inside the retired SW-7028 frozen-catalog suite; they assert what a
// screen DOES, never that its control list matches a snapshot.

beforeEach(() => {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  vi.spyOn(Date, "now").mockReturnValue(1794000000000);
  vi.spyOn(Math, "random").mockReturnValue(0.25);
  vi.stubGlobal("fetch", vi.fn(() => { throw new Error("Studio surface tests must not fetch or execute a model"); }));
});
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

it("submits a full Video request from Render clip", async () => {
  const result = await observe(currentCandidate(), { platform: "linux", studio: "video", model: "wan_2_2", mode: "text_to_video" });
  expect(result.submission.available).toBe(true);
  expect(result.submission.disabled).toBe(false);
  expect(result.submission.calls[0]).toMatchObject({ kind: "video", payload: { model: "wan_2_2", mode: "text_to_video" } });
});

it("submits a full Training request from its type=button action", async () => {
  const result = await observe(currentCandidate(), { platform: "linux", studio: "training", model: "anima_base_lora", mode: "anima_lora" });
  expect(result.submission.available).toBe(true);
  expect(result.submission.disabled).toBe(false);
  expect(result.submission.calls[0]).toMatchObject({ kind: "training", payload: { targetId: "anima_base_lora", dryRun: false } });
});

it("recognizes the Video Replace person action rather than its mode tab", async () => {
  const mounted = await mount(currentCandidate(), { platform: "macos", studio: "video", model: "ltx_2_3", mode: "replace_person" });
  try {
    const action = submissionButton(mounted.container, "video");
    expect(action?.textContent.trim()).toBe("Replace person");
    expect(action?.type).toBe("submit");
    expect(action?.classList.contains("prompt-cta")).toBe(true);
  } finally { await mounted.close(); }
});

it("renders every converted Anima tier and submits the q8 default", async () => {
  const catalog = currentCandidate();
  const row = { platform: "macos", studio: "image", model: "anima_base", mode: "text_to_image" };
  const model = contextFor(catalog, row).context.imageModels[0];
  expect(installedTiers(model)).toEqual(["q4", "q8", "bf16"]);
  expect(defaultTierSelection(model, null)).toBe("q8");
  const mounted = await mount(catalog, row);
  try {
    const picker = mounted.container.querySelector(".quant-tier-picker select");
    expect([...picker.options].map(({ value }) => value)).toEqual(["q4", "q8", "bf16"]);
    expect(picker.value).toBe("q8");
    await click([...mounted.container.querySelectorAll("button")].find((node) => node.textContent.trim() === "Generate"));
    expect(mounted.calls[0].payload.advanced.mlxQuantize).toBe(8);
  } finally { await mounted.close(); }
});

// Upscale, Hires.fix and PiD are mutually exclusive post-passes: turning one on clears the others.
const postPasses = (mounted) => {
  const saved = mounted.getPersisted();
  return { usePid: saved.usePid, upscale: saved.upscaleEnabled, hiresFix: saved.hiresFixEnabled };
};
const checkbox = (container, label) => [...container.querySelectorAll('input[type="checkbox"]')]
  .find((node) => node.closest("label")?.textContent.trim() === label);
const sdxl = { platform: "linux", studio: "image", model: "sdxl", mode: "text_to_image" };

for (const [id, other, label] of [["upscale", "hiresFix", "Upscale"], ["hiresFix", "upscale", "Hires.fix"]]) {
  it(`enabling ${label} clears ${other}`, async () => {
    const mounted = await mount(currentCandidate(), sdxl, { upscaleEnabled: other === "upscale", hiresFixEnabled: other === "hiresFix" });
    try {
      expect(postPasses(mounted)[other]).toBe(true);
      const node = checkbox(mounted.container, label);
      expect(node?.disabled).toBe(false);
      await click(node);
      expect(postPasses(mounted)).toMatchObject({ [id]: true, [other]: false });
    } finally { await mounted.close(); }
  });
}

it("a restored PiD alongside Upscale or Hires.fix is cleared", async () => {
  for (const id of ["upscale", "hiresFix"]) {
    const mounted = await mount(currentCandidate(), sdxl, { usePid: true, upscaleEnabled: id === "upscale", hiresFixEnabled: id === "hiresFix" });
    try {
      expect(postPasses(mounted)).toMatchObject({ [id]: true, usePid: false });
    } finally { await mounted.close(); }
  }
});

it("choosing a non-native decoder clears a restored PiD", async () => {
  // Installed alternate-decoder host state: the public picker uses this API availability
  // projection. No decoder or model code executes in jsdom.
  const catalog = currentCandidate();
  const model = catalog.models.find((entry) => entry.id === "krea_2_raw");
  model.decoders.byBackend.mlx = model.decoders.byBackend.mlx.map((option) => ({ ...option, available: true }));
  const mounted = await mount(catalog, { platform: "macos", studio: "image", model: "krea_2_raw", mode: "text_to_image" }, { usePid: true });
  try {
    expect(postPasses(mounted).usePid).toBe(true);
    const decoder = [...mounted.container.querySelectorAll("select")].find((node) => node.closest("label")?.textContent.startsWith("Decoder"));
    const alternative = [...decoder.options].find((option) => option.value !== "native" && !option.disabled);
    expect(alternative).toBeTruthy();
    await change(decoder, alternative.value);
    expect(postPasses(mounted).usePid).toBe(false);
  } finally { await mounted.close(); }
});
