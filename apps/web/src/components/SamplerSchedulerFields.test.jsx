import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { mountRoot, setInput, setSelect, unmountRoot } from "../testUtils/dom.js";
import { useSamplerControls } from "../hooks/useSamplerControls.js";
import { SamplerSchedulerFields, SamplingSeedField, SamplingStepsField } from "./SamplerSchedulerFields.jsx";

let root, container, state;
beforeEach(() => {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  ({ root, container } = mountRoot());
});
afterEach(() => unmountRoot(root, container));
const model = { id: "a", limits: { samplers: ["euler", "heun"], schedulers: ["normal", "karras"] },
  defaults: { sampler: "heun", scheduler: "karras", steps: 12, schedulerShift: 4 },
  candle: { limits: { samplers: ["euler"], schedulers: ["normal"] } } };
function Harness({ model, backend = null, reconcile = "invalid", preset = false }) {
  const [sampler, setSampler] = React.useState(preset ? "" : "heun");
  const [scheduler, setScheduler] = React.useState(preset ? "" : "karras");
  const [schedulerShift, setSchedulerShift] = React.useState(1);
  const [steps, setSteps] = React.useState("");
  const [seed, setSeed] = React.useState("");
  const controls = useSamplerControls(model, backend, { sampler, scheduler, setSampler, setScheduler,
    setSchedulerShift, reconcile, emptyMenus: preset });
  state = { sampler, scheduler, schedulerShift, steps, seed };
  return <>
    <SamplerSchedulerFields sampler={sampler} scheduler={scheduler} {...controls}
      onSamplerChange={setSampler} onSchedulerChange={setScheduler} emptyLabel={preset ? "No default" : undefined} />
    <SamplingStepsField value={steps} onChange={setSteps} inputProps={{ type: "number", placeholder: String(controls.stepsDefault ?? "") }} />
    <SamplingSeedField value={seed} onChange={setSeed} inputProps={{ type: "number", placeholder: "Random" }} />
  </>;
}

describe("shared sampling authoring contract", () => {
  it("preserves restored choices until catalog resolution and clamps to the actual backend menu", async () => {
    await act(async () => root.render(<Harness model={null} />));
    expect(state).toMatchObject({ sampler: "heun", scheduler: "karras" });
    await act(async () => root.render(<Harness model={model} />));
    expect(container.querySelectorAll("select")).toHaveLength(2);
    expect(container.querySelector("input").placeholder).toBe("12");
    await act(async () => root.render(<Harness model={model} backend="candle" />));
    expect(state).toMatchObject({ sampler: "euler", scheduler: "normal", schedulerShift: 1 });
    expect(container.querySelectorAll("select")).toHaveLength(2);
    expect([...container.querySelectorAll("select")].map((select) => select.value)).toEqual(["euler", "normal"]);
    const inputs = container.querySelectorAll("input");
    await act(async () => { setInput(inputs[0], "8"); setInput(inputs[1], "0"); });
    expect(state).toMatchObject({ steps: "8", seed: "0" });
    await act(async () => { setInput(inputs[0], ""); setInput(inputs[1], ""); });
    expect(state).toMatchObject({ steps: "", seed: "" });
  });
  it.each(["invalid", "model-invalid", "model-reset"])("retains the %s owner's model-change policy", async (reconcile) => {
    await act(async () => root.render(<Harness model={model} reconcile={reconcile} />));
    const next = { ...model, id: "b", defaults: { sampler: "euler", scheduler: "normal", schedulerShift: 5 },
      limits: { samplers: ["euler", "heun"], schedulers: ["normal"] } };
    await act(async () => root.render(<Harness model={next} reconcile={reconcile} />));
    expect(state.sampler).toBe(reconcile === "model-reset" ? "euler" : "heun");
    expect(state.scheduler).toBe("normal");
    expect(state.schedulerShift).toBe(reconcile === "invalid" ? 1 : "");
  });
  it("keeps preset no-default values empty until the author chooses a value", async () => {
    await act(async () => root.render(<Harness model={null} reconcile="none" preset />));
    expect(container.querySelectorAll("select")).toHaveLength(0);
    await act(async () => root.render(<Harness model={model} reconcile="none" preset />));
    const select = container.querySelector("select");
    expect(select.value).toBe("");
    expect(select.options[0].textContent).toBe("No default");
    await act(async () => setSelect(select, "euler"));
    expect(state.sampler).toBe("euler");
  });
});
