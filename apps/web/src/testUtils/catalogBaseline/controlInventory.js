import identities from "./controlIdentities.json";
import old from "./old-catalog.json";

// These names are a frozen vocabulary, not inferred candidate wire fields. Unknown
// controls fail closed until their identity is explicitly added to the vocabulary.
const fieldPrefixes = Object.keys(identities.fields).sort((a, b) => b.length - a.length);
const modelNames = new Set(old.catalog.models.map((model) => model.name ?? model.id));
export const declaredEffects = {
  hiresFix: [{ when: "enabled", clear: "usePid" }, { when: "enabled", clear: "upscale" }],
  upscale: [{ when: "enabled", clear: "usePid" }, { when: "enabled", clear: "hiresFix" }],
  decoder: [{ when: "nonNative", clear: "usePid" }],
};

function optionId(control, simple, menu) {
  if (!menu && simple && identities.simpleVoiceLabels.includes(control.label)) return "voice";
  if (!menu && simple) {
    const prefix = Object.keys(identities.actionPrefixes).find((name) => control.label.startsWith(name));
    if (prefix) return identities.actionPrefixes[prefix];
  }
  if (!menu && simple && modelNames.has(control.label)) return "model";
  if (!menu && simple && /^\d+:\d+.*[×x]/.test(control.label)) return "resolution";
  if (["button", "tab", "radio"].includes(control.role)) {
    const id = identities.actions[control.label];
    if (id) return `${menu ? "choice/" : ""}${id}`;
  } else {
    // Match only an explicit historical field name; wrapping label text can also
    // contain its value, choices and help text. None of those belongs in an ID.
    const name = fieldPrefixes.find((prefix) => control.label.startsWith(prefix));
    if (name) return identities.fields[name];
  }
  throw new Error(`No declared control identity: ${control.role}: ${control.label}`);
}

export function backendFor(platform) {
  if (platform.macGatingActive) return "mlx";
  if (platform.candleGatingActive) return "candle";
  return "ungated";
}

function inventoryControls(controls, scope, simple, menu = false) {
  const occurrences = new Map();
  return controls.map((control) => {
    const id = optionId(control, simple, menu);
    const instance = occurrences.get(id) ?? 0;
    occurrences.set(id, instance + 1);
    return { id, controlId: `${scope}/${control.role}/${id}/${instance}`,
      availability: control.disabled ? "disabled" : "enabled",
      reason: control.disabledReason, effects: declaredEffects[id] ?? [] };
  });
}

export function controlInventory(observation, platform, row = {}) {
  const scope = row.studio ?? "studio";
  return {
    backend: backendFor(platform),
    controls: inventoryControls(observation.controls, scope, scope.startsWith("simple")),
    menus: observation.menus.map((menu, index) => ({ label: menu.label,
      controls: inventoryControls(menu.controls, `${scope}/picker/${index}`, false, true) })),
  };
}

// Effect witnesses provide the precondition and final persisted state after a real
// interaction. Checking a declaration against itself cannot establish this contract.
export function assertDeclaredEffects(id, when, before, after) {
  for (const effect of declaredEffects[id] ?? []) {
    if (effect.when === when && before[effect.clear] && after[effect.clear] !== false) {
      throw new Error(`${id} must clear ${effect.clear}`);
    }
  }
}
