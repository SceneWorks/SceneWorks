// Derive the historical side solely from pinned rendered observations, including the menu
// choices. The old option vocabulary lives here so future registry/UI edits cannot rename it.
const optionLabels = {
  "schedule shift": "schedulerShift", sampler: "sampler", scheduler: "scheduler", steps: "steps",
  guidance: "guidanceScale", "guidance (cfg)": "guidance", "width override": "width",
  "height override": "height", "custom width": "width", "custom height": "height",
  aspect: "resolution", variations: "count", "pid decoder non-commercial": "usePid",
  "hires.fix": "hiresFix", upscale: "upscale", "flash attention": "flashAttn",
  "quant tier": "mlxQuantize", "negative prompt": "negativePrompt", "caption overflow": "captionOverflow",
  "guidance method": "guidanceMethod", "transparent background (rgba)": "transparentBackground",
  "text style": "textStyleGain", "control strength": "controlScale", "enable multi-phase denoise": "phases",
};

const effects = {
  hiresFix: [{ when: "enabled", clear: "usePid" }, { when: "enabled", clear: "upscale" }],
  upscale: [{ when: "enabled", clear: "usePid" }, { when: "enabled", clear: "hiresFix" }],
  decoder: [{ when: "nonNative", clear: "usePid" }],
};

function optionId(control) {
  // The legacy DOM value attribute contains the current number/text. It cannot identify
  // a field. Remove values and select choices embedded by the wrapping HTML label.
  let label = control.label;
  if (control.options?.length) {
    const start = label.indexOf(control.options.map((option) => option.label).join(""));
    if (start >= 0) label = label.slice(0, start);
  }
  if (typeof control.value === "string" && control.value && label.endsWith(control.value)) {
    label = label.slice(0, -control.value.length);
  }
  const name = label.trim().toLowerCase();
  if (optionLabels[name]) return optionLabels[name];
  const prefix = ["negative prompt", "quant tier", "decoder experimental", "control strength", "text style", "system prompt", "prompta ", "aspect256"].find((key) => name.startsWith(key));
  if (prefix) return optionLabels[prefix] ?? { "decoder experimental": "decoder", "system prompt": "systemMessage", "prompta ": "prompt", aspect256: "resolution" }[prefix];
  // A navigation or picker action has a stable control identity but no wire option of its own.
  return (name || control.role).replace(/[^a-z0-9]+(.)?/g, (_, next) => next?.toUpperCase() ?? "");
}

export function backendFor(platform) {
  if (platform.macGatingActive) return "mlx";
  if (platform.candleGatingActive) return "candle";
  return "ungated";
}

function inventoryControls(controls) {
  return controls.map((control) => {
    const id = optionId(control);
    return { id, availability: control.disabled ? "disabled" : "enabled",
      reason: control.disabledReason, effects: effects[id] ?? [] };
  });
}

export function controlInventory(observation, platform) {
  return {
    backend: backendFor(platform),
    controls: inventoryControls(observation.controls),
    menus: observation.menus.map((menu) => ({ label: menu.label, controls: inventoryControls(menu.controls) })),
  };
}
