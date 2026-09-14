import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mountRoot, unmountRoot } from "../testUtils/dom.js";
import {
  ModeTabs,
  StyleAxisRow,
  TierPickerField,
  useQuantTierPicker,
} from "../components/generationStudio.jsx";
import { readLastTier, writeLastTier } from "../lastTierStore.js";

vi.mock("../lastTierStore.js", () => ({
  readLastTier: vi.fn(() => null),
  writeLastTier: vi.fn(),
}));
vi.mock("../generationQuality.js", () => ({
  readDefaultGenerationQuality: vi.fn(() => "auto"),
}));

function modelWithTiers(id, installed = ["q4", "q8", "bf16"]) {
  return {
    id,
    hasVariantMatrix: true,
    variants: ["q4", "q8", "bf16"].map((variant) => ({
      variant,
      installState: installed.includes(variant) ? "installed" : "missing",
    })),
  };
}

function TierHarness({
  model = "one",
  availableTiers = ["q4", "q8"],
  screen = "video",
  reseedOnModelChange = false,
  useGenerationQuality = false,
  autoTier = null,
  preferencesHydrated = true,
}) {
  const state = useQuantTierPicker({
    screen,
    model,
    selectedModel: modelWithTiers(model, availableTiers),
    availableTiers,
    tierOptions: { defaultQuality: "q4" },
    reseedOnModelChange,
    useGenerationQuality,
    autoTier,
    preferencesHydrated,
  });
  return (
    <div>
      <output>{state.quantTier}|{state.tierSwitching}</output>
      <button data-action="q8" onClick={() => state.handleTierChange("q8")}>q8</button>
      <button data-action="invalid" onClick={() => state.handleTierChange("bf16")}>invalid</button>
      <button
        data-action="recipe-q8"
        onClick={() => {
          state.skipNextReseed();
          state.setQuantTier("q8");
        }}
      >
        recipe q8
      </button>
      <button
        data-action="recipe-bf16"
        onClick={() => {
          state.skipNextReseed();
          state.setQuantTier("bf16");
        }}
      >
        recipe bf16
      </button>
    </div>
  );
}

describe("shared generation-studio controls", () => {
  let container;
  let root;
  beforeEach(() => {
    vi.mocked(readLastTier).mockReset();
    vi.mocked(readLastTier).mockReturnValue(null);
    vi.mocked(writeLastTier).mockReset();
    ({ container, root } = mountRoot());
  });
  afterEach(async () => {
    vi.useRealTimers();
    await unmountRoot(root, container);
  });

  it("keeps the tier switch hint for 1500ms and then clears it", async () => {
    vi.useFakeTimers();
    await act(async () => root.render(<TierHarness />));
    await act(async () => container.querySelector('[data-action="q8"]').click());
    expect(container.querySelector("output").textContent).toBe("q8|q8");
    await act(async () => vi.advanceTimersByTime(1500));
    expect(container.querySelector("output").textContent).toBe("q8|");
  });

  it("re-seeds Video A→B from B's sticky even when the installed tier list is identical", async () => {
    vi.mocked(readLastTier).mockImplementation((_, model) => model === "B" ? "q8" : null);
    await act(async () => root.render(<TierHarness model="A" reseedOnModelChange />));
    expect(container.querySelector("output").textContent).toBe("q4|");
    await act(async () => root.render(<TierHarness model="B" reseedOnModelChange />));
    expect(container.querySelector("output").textContent).toBe("q8|");
  });

  it("preserves an installed recipe tier exactly once, then re-seeds the next model change", async () => {
    vi.mocked(readLastTier).mockImplementation((_, model) => model === "C" ? "q4" : null);
    await act(async () => root.render(<TierHarness model="A" reseedOnModelChange />));
    await act(async () => container.querySelector('[data-action="recipe-q8"]').click());
    await act(async () => root.render(<TierHarness model="B" reseedOnModelChange />));
    expect(container.querySelector("output").textContent).toBe("q8|");
    await act(async () => root.render(<TierHarness model="C" reseedOnModelChange />));
    expect(container.querySelector("output").textContent).toBe("q4|");
  });

  it("falls through to the new model default when a recipe tier is unavailable", async () => {
    await act(async () => root.render(<TierHarness model="A" reseedOnModelChange />));
    await act(async () => container.querySelector('[data-action="recipe-bf16"]').click());
    await act(async () => root.render(
      <TierHarness model="B" availableTiers={["q4", "q8"]} reseedOnModelChange />,
    ));
    expect(container.querySelector("output").textContent).toBe("q4|");
  });

  it("keeps Image Auto capability-aware while Video retains its q4 base", async () => {
    await act(async () => root.render(
      <TierHarness
        model="image"
        availableTiers={["q4", "q8", "bf16"]}
        screen="image"
        useGenerationQuality
        autoTier="bf16"
      />,
    ));
    expect(container.querySelector("output").textContent).toBe("bf16|");
    await unmountRoot(root, container);
    ({ container, root } = mountRoot());
    await act(async () => root.render(
      <TierHarness model="video" availableTiers={["q4", "q8", "bf16"]} screen="video" />,
    ));
    expect(container.querySelector("output").textContent).toBe("q4|");
  });

  it("waits for durable preferences before seeding instead of locking in a catalog-race default", async () => {
    vi.mocked(readLastTier).mockReturnValue(null);
    await act(async () => root.render(
      <TierHarness
        model="imported-int8"
        availableTiers={["q4", "q8", "bf16"]}
        screen="image"
        useGenerationQuality
        autoTier="q8"
        preferencesHydrated={false}
      />,
    ));
    expect(container.querySelector("output").textContent).toBe("|");
    expect(readLastTier).not.toHaveBeenCalled();

    // App seeds the server's per-model map into lastTierStore before flipping this flag.
    vi.mocked(readLastTier).mockReturnValue("bf16");
    await act(async () => root.render(
      <TierHarness
        model="imported-int8"
        availableTiers={["q4", "q8", "bf16"]}
        screen="image"
        useGenerationQuality
        autoTier="q8"
        preferencesHydrated
      />,
    ));
    expect(container.querySelector("output").textContent).toBe("bf16|");
    expect(readLastTier).toHaveBeenCalledWith("image", "imported-int8");
  });

  it("persists only valid manual selections and rejects unavailable tiers", async () => {
    await act(async () => root.render(<TierHarness model="one" />));
    await act(async () => container.querySelector('[data-action="invalid"]').click());
    expect(container.querySelector("output").textContent).toBe("q4|");
    expect(writeLastTier).not.toHaveBeenCalled();
    await act(async () => container.querySelector('[data-action="q8"]').click());
    expect(writeLastTier).toHaveBeenCalledWith("video", "one", "q8");
  });

  it("preserves disabled and active mode-tab semantics", async () => {
    await act(async () => root.render(
      <ModeTabs
        label="Modes"
        options={[["text", "Text"], ["edit", "Edit"]]}
        mode="text"
        onChange={() => {}}
        blockFor={(_, active) => active ? null : { text: "Unavailable" }}
      />,
    ));
    const [text, edit] = container.querySelectorAll('[role="tab"]');
    expect(text.getAttribute("aria-selected")).toBe("true");
    expect(edit.disabled).toBe(true);
    expect(edit.title).toBe("Unavailable");
  });

  it("renders shared tier warnings and style/general preset chips", async () => {
    await act(async () => root.render(
      <TierPickerField
        value="q8"
        onChange={() => {}}
        items={[{ tier: "q8", label: "Q8", disabled: false }]}
        tierSwitching=""
        tierLabel={(value) => value}
        warning={<span>Memory warning</span>}
      />,
    ));
    expect(container.textContent).toContain("Memory warning");
    const onPresetChange = vi.fn();
    const onToggleGeneral = vi.fn();
    await act(async () => root.render(
      <StyleAxisRow
        available={false}
        groups={[]}
        styleId={null}
        onStyleChange={() => {}}
        selectedPreset={null}
        presets={[{ id: "film", name: "Film" }]}
        onPresetChange={onPresetChange}
        generalPresets={[{ id: "sharp", name: "Sharp" }]}
        generalStackIds={["sharp"]}
        onToggleGeneral={onToggleGeneral}
        noPresetValue="none"
      />,
    ));
    const buttons = [...container.querySelectorAll("button")];
    await act(async () => buttons.find((button) => button.textContent === "Film").click());
    await act(async () => buttons.find((button) => button.textContent === "Sharp").click());
    expect(onPresetChange).toHaveBeenCalledWith("film");
    expect(onToggleGeneral).toHaveBeenCalledWith("sharp");
    expect(buttons.find((button) => button.textContent === "Sharp").classList.contains("active")).toBe(true);
  });
});
