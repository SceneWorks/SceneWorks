import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { click, mountRoot, setInput, unmountRoot } from "../testUtils/dom.js";
import { LoraPickerList } from "./LoraPickerList.jsx";
import { LoraPickerField, useLoraSelection } from "./LoraPickerField.jsx";
import { LoraPickerSection } from "./generationStudio.jsx";
import { MAX_JOB_LORAS_TOTAL, MAX_USER_JOB_LORAS } from "../presetUtils.js";
import { CharacterLoras } from "../screens/characterPanels.jsx";

const loras = [
  { id: "analog-photo", name: "Film Grain", family: "sdxl", scope: "project", triggerWords: ["analog35"] },
  { id: "watercolor", name: "Watercolor", family: "sdxl", scope: "global" },
];
const model = { id: "sdxl", family: "sdxl", name: "SDXL" };
let root;
let container;
beforeEach(() => ({ root, container } = mountRoot()));
afterEach(() => unmountRoot(root, container));
const search = () => container.querySelector("input[type='search']");
const rowNames = () => [...container.querySelectorAll("button strong")].map((node) => node.textContent);
async function filter(query) {
  await act(async () => setInput(search(), query));
}

describe.each(["studio", "simple"])("shared LoRA list (%s)", (variant) => {
  it("puts a focused search first and filters names, IDs, metadata and trigger words", async () => {
    const onAdd = vi.fn();
    await act(async () => root.render(<LoraPickerList loras={loras} onAdd={onAdd} variant={variant} />));
    expect(container.firstElementChild.firstElementChild).toBe(search());
    expect(document.activeElement).toBe(search());
    for (const query of ["  FILM  ", "analog-photo", "PROJECT sdxl", "analog35"]) {
      await filter(query);
      expect(rowNames()).toEqual(["Film Grain"]);
    }
    await click(container.querySelector("button"));
    expect(onAdd).toHaveBeenCalledWith(loras[0]);
    await filter("no match");
    expect(rowNames()).toEqual([]);
    expect(container.querySelector("[role='status']").textContent).toBe("No LoRAs match your search.");
    await filter("");
    expect(rowNames()).toEqual(["Film Grain", "Watercolor"]);
  });

  it("keeps caller limits and live catalog updates while searching", async () => {
    const onAdd = vi.fn();
    const render = (items) => <LoraPickerList loras={items} onAdd={onAdd} isDisabled={(lora) => lora.id === "watercolor"} variant={variant} />;
    await act(async () => root.render(render(loras)));
    await filter("water");
    expect(container.querySelector("button").disabled).toBe(true);
    await click(container.querySelector("button"));
    expect(onAdd).not.toHaveBeenCalled();
    await act(async () => root.render(render([...loras, { id: "water-ink", name: "Water Ink" }])));
    expect(search().value).toBe("water");
    expect(rowNames()).toEqual(["Watercolor", "Water Ink"]);
  });
});

it("keeps the eligibility empty message and prevents search Enter from submitting a parent form", async () => {
  const onSubmit = vi.fn((event) => event.preventDefault());
  await act(async () => root.render(
    <form onSubmit={onSubmit}>
      <LoraPickerList loras={[]} emptyMessage="No installed compatible LoRAs." onAdd={vi.fn()} />
    </form>,
  ));
  expect(container.querySelector("[role='status']").textContent).toBe("No installed compatible LoRAs.");
  const event = new window.KeyboardEvent("keydown", { key: "Enter", bubbles: true, cancelable: true });
  await act(async () => search().dispatchEvent(event));
  expect(event.defaultPrevented).toBe(true);
  expect(onSubmit).not.toHaveBeenCalled();
});

function CharacterHarness({ catalog = loras, selectedModel = model }) {
  const selection = useLoraSelection(catalog, selectedModel);
  return <><LoraPickerField selection={selection} /><output>{JSON.stringify(selection.serializedLoras)}</output></>;
}

it("uses the studio picker in Character, keeps weights while filtering, and clears search on reopen", async () => {
  await act(async () => root.render(<CharacterHarness catalog={[
    ...loras,
    { id: "pending", family: "sdxl", installState: "missing" },
    { id: "managed", family: "sdxl", presetManaged: true },
    { id: "other", family: "flux" },
    { id: "unknown" },
  ]} />));
  await click(container.querySelector(".lora-add"));
  expect(rowNames()).toEqual(["Film Grain", "Watercolor"]);
  await filter("film");
  await click(container.querySelector(".lora-pick-row"));
  expect(search()).toBeNull();
  await act(async () => setInput(container.querySelector("input[type='range']"), "-0.5"));
  await click(container.querySelector(".lora-add"));
  expect(search().value).toBe("");
  expect(rowNames()).toEqual(["Watercolor"]);
  await filter("no match");
  expect(JSON.parse(container.querySelector("output").textContent)).toEqual([
    expect.objectContaining({ id: "analog-photo", weight: -0.5 }),
  ]);
  await click(container.querySelector(".lora-slot-remove"));
  await filter("");
  expect(rowNames()).toEqual(["Film Grain", "Watercolor"]);
  expect(JSON.parse(container.querySelector("output").textContent)).toEqual([]);
});

it("preserves Character's total cap and prunes selections when the model changes", async () => {
  const catalog = Array.from({ length: MAX_JOB_LORAS_TOTAL + 1 }, (_, index) => ({ id: `lora-${index}`, family: "sdxl" }));
  await act(async () => root.render(<CharacterHarness catalog={catalog} />));
  for (let index = 0; index < MAX_JOB_LORAS_TOTAL; index += 1) {
    await click(container.querySelector(".lora-add"));
    await click(container.querySelector(".lora-pick-row"));
  }
  await click(container.querySelector(".lora-add"));
  expect(container.querySelector(".lora-pick-row").disabled).toBe(true);
  await act(async () => root.render(<CharacterHarness catalog={catalog} selectedModel={{ id: "flux", family: "flux" }} />));
  expect(JSON.parse(container.querySelector("output").textContent)).toEqual([]);
});

it("preserves studio user caps with builtin LoRAs still selectable after search", async () => {
  const onAdd = vi.fn();
  await act(async () => root.render(<LoraPickerSection
    selectedModel={model}
    selectedLoras={[]}
    selectedLoraIds={[]}
    compatibleLoras={[...loras, { id: "builtin", name: "Builtin Film", scope: "builtin" }]}
    userSelectedLoraCount={MAX_USER_JOB_LORAS}
    toggleLora={onAdd}
  />));
  await click(container.querySelector(".lora-add"));
  expect(container.querySelector(".lora-picker-panel").firstElementChild.firstElementChild).toBe(search());
  await filter("film");
  const rows = [...container.querySelectorAll(".lora-pick-row")];
  expect(rows.map((row) => row.disabled)).toEqual([true, false]);
  await click(rows[1]);
  expect(onAdd).toHaveBeenCalledWith(expect.objectContaining({ id: "builtin" }));
  expect(search()).toBeNull();
});

it("uses the shared searchable list to select a Character adapter before attaching", async () => {
  const onAttach = vi.fn();
  function Harness() {
    const [loraId, setLoraId] = React.useState("");
    return <CharacterLoras
      loras={loras}
      selectedCharacter={{ id: "character", loras: [] }}
      loraId={loraId}
      setLoraId={setLoraId}
      submitLora={(event) => { event.preventDefault(); onAttach(loraId); }}
    />;
  }
  await act(async () => root.render(<Harness />));
  await click([...container.querySelectorAll("button")].find((node) => node.textContent === "Attach imported LoRA"));
  await filter("film");
  expect(rowNames()).toEqual(["Film Grain"]);
  await click(container.querySelector(".lora-pick-row"));
  expect(search()).toBeNull();
  expect(onAttach).not.toHaveBeenCalled();
  const attach = container.querySelector("button[type='submit']");
  expect(attach.disabled).toBe(false);
  await click(attach);
  expect(onAttach).toHaveBeenCalledWith("analog-photo");
});
