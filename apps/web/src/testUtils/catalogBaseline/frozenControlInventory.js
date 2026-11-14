import { createHash } from "node:crypto";
import frozen from "./old-control-inventory.json";
import identities from "./old-control-identities.json";
import observations from "./old-observations.json";

// The original capture predates the reviewed identity correction. This frozen
// migration was derived from identityRevision, never from the candidate vocabulary.
if (identities.sourceRevision !== frozen.source.revision ||
    identities.legacyInventorySha256 !== createHash("sha256").update(JSON.stringify(frozen)).digest("hex")) {
  throw new Error("Frozen control identity migration does not match its source inventory");
}

export function frozenControlInventory(key) {
  const row = frozen.rows[key];
  const observed = observations.rows[key];
  const studio = key.split("/")[1];
  function controls(group, roles, scope) {
    const ids = identities.groups[identities.byLegacyGroup[group]];
    if (ids.length !== frozen.groups[group].length || ids.length !== roles.length) {
      throw new Error(`Incomplete frozen control identity migration for ${key}`);
    }
    const occurrences = new Map();
    return frozen.groups[group].map((control, index) => {
      const id = identities.ids[ids[index]];
      const instance = occurrences.get(id) ?? 0;
      occurrences.set(id, instance + 1);
      return { ...control, id, controlId: `${scope}/${roles[index]}/${id}/${instance}` };
    });
  }
  return {
    backend: row.backend,
    controls: controls(row.controls, observed.controls.map((index) => observations.controls[index].role), studio),
    menus: row.menus.map((menu, index) => ({ ...menu,
      controls: controls(menu.controls, observed.menus[index].controls.map((control) => control.role), `${studio}/picker/${index}`),
    })),
  };
}
