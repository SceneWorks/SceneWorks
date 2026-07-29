#!/usr/bin/env node
// Split every dual-backend memory-strategy story into an MLX twin and a Candle twin (sc-15812).
//
// WHY A SCRIPT: the split is ~230 discrete Shortcut writes. The MCP surface available to an agent
// offers no idempotency, no transaction, and ambiguous failures, so a hand pass reliably produces
// silent misses in exactly the metadata whose purpose is stopping Candle work from being lost.
//
// This script is:
//   * PURE in --dry-run     — reads only the generated matrix; needs no token, writes nothing.
//   * IDEMPOTENT in --apply — every created object is recorded in a state file and re-checked
//                             against the live API, so a re-run after any failure resumes cleanly.
//   * TWO-PHASE             — nothing predicts a server-assigned id; cross-references are patched
//                             from the recorded id map after creation.
//   * VERIFIABLE in --verify — diffs the live story graph against what the matrix says must exist.
//
// Usage:
//   node scripts/split-memory-strategy-stories.mjs --dry-run
//   SHORTCUT_API_TOKEN=... node scripts/split-memory-strategy-stories.mjs --apply
//   SHORTCUT_API_TOKEN=... node scripts/split-memory-strategy-stories.mjs --verify
//
// The token is read from the environment and is never logged, echoed, or written to the state file.

import { readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, "..");
const MATRIX = resolve(REPO, "docs/generated/memory-matrix.json");
const STATE = resolve(REPO, ".shortcut-split-state.json");
const PLAN = resolve(REPO, ".shortcut-split-plan.json");

const EPIC_ID = 15448;
const API = "https://api.app.shortcut.com/api/v3";

// Upstream gates every Candle family story inherits from its MLX twin. Kept explicit rather than
// copied from the twin's links, because the twin also carries links we must NOT clone (its own
// model-story children, which the Candle family gets fresh).
const FAMILY_GATES = [15507, 15508, 15750, 15804, 15799, 15812];

const MLX = "mlx";
const CANDLE = "candle";

// A new Candle twin ALWAYS starts in To Do. It must never inherit its MLX twin's workflow state:
// the twin's state reflects work done on a Mac, and copying it would mark Candle work complete that
// nobody has done — the exact false green this epic exists to prevent. (This bit: SC-15510 is Done,
// so its twin SC-15815 was created Done, which in turn made its two children read as unblocked.)
const TODO_STATE = 500000007;

// ── invariants the matrix must satisfy before we touch anything ────────────────────────────────
const EXPECT = {
  totalModels: 53,
  dualModels: 33,
  mlxOnlyModels: 20,
  candleOnlyModels: 0,
  dualFamilies: 14,
  mlxOnlyFamilies: 6,
};

function die(message) {
  console.error(`\n  FAIL  ${message}\n`);
  process.exit(1);
}

// ── phase A: plan, computed purely from the generated matrix ───────────────────────────────────
function buildPlan() {
  if (!existsSync(MATRIX)) die(`matrix not found at ${MATRIX}`);
  const matrix = JSON.parse(readFileSync(MATRIX, "utf8"));

  const byModel = new Map();
  // sc-15812 made the matrix attribute cells PER (model, backend): a candle cell names the Candle
  // twin, not the MLX story. So ownership is read per backend, and the disagreement guard is scoped
  // within a backend — comparing across backends would now fire on a correct matrix.
  const OWNERSHIP = {
    mlx: { story: "mlxStory", family: "family" },
    candle: { story: "candleStory", family: "candleFamily" },
  };
  for (const cell of matrix.cells ?? []) {
    let entry = byModel.get(cell.modelId);
    if (!entry) {
      entry = {
        model: cell.modelId,
        mlxStory: null,
        family: null,
        candleStory: null,
        candleFamily: null,
        backends: new Set(),
      };
      byModel.set(cell.modelId, entry);
    }
    entry.backends.add(cell.backend);
    const keys = OWNERSHIP[cell.backend];
    if (!keys) die(`${cell.modelId}: unknown cell backend ${cell.backend}`);
    if (entry[keys.story] === null) {
      entry[keys.story] = cell.owningModelStory;
      entry[keys.family] = cell.owningFamilyStory;
    } else if (
      // A model whose same-backend cells disagree about ownership means the generator changed under us.
      entry[keys.story] !== cell.owningModelStory ||
      entry[keys.family] !== cell.owningFamilyStory
    ) {
      die(`${cell.modelId}: ${cell.backend} cells disagree about owning story/family — regenerate the matrix`);
    }
  }

  const models = [...byModel.values()].map((m) => ({ ...m, backends: [...m.backends].sort() }));
  const dualModels = models.filter((m) => m.backends.length > 1).sort((a, b) => a.mlxStory - b.mlxStory);
  const mlxOnly = models.filter((m) => m.backends.length === 1 && m.backends[0] === MLX);
  const candleOnly = models.filter((m) => m.backends.length === 1 && m.backends[0] === CANDLE);

  const dualFamilies = [...new Set(dualModels.map((m) => m.family))].sort((a, b) => a - b);
  const allFamilies = [...new Set(models.map((m) => m.family))];
  const mlxOnlyFamilies = allFamilies.filter((f) => !dualFamilies.includes(f)).sort((a, b) => a - b);

  // sc-15812: now that the matrix names both twins, the Candle ids are readable straight off it.
  // That matters because `.shortcut-split-state.json` is gitignored, so --verify used to report all
  // 47 twins missing from a fresh clone; the matrix is the durable source.
  // A plain object, not a Map: --dry-run serialises the plan, and a Map would silently write `{}`.
  const candleFamilyByFamily = {};
  for (const m of dualModels) {
    const seen = candleFamilyByFamily[m.family];
    if (seen === undefined) candleFamilyByFamily[m.family] = m.candleFamily;
    else if (seen !== m.candleFamily) {
      die(`family SC-${m.family}: models disagree about the Candle family twin (SC-${seen} vs SC-${m.candleFamily})`);
    }
  }

  // Assert the shape BEFORE any write. A drifted catalog must stop the run, not silently
  // half-apply — the whole point of the split is a trustworthy graph.
  const actual = {
    totalModels: models.length,
    dualModels: dualModels.length,
    mlxOnlyModels: mlxOnly.length,
    candleOnlyModels: candleOnly.length,
    dualFamilies: dualFamilies.length,
    mlxOnlyFamilies: mlxOnlyFamilies.length,
  };
  for (const [key, want] of Object.entries(EXPECT)) {
    if (actual[key] !== want) {
      die(
        `invariant ${key}: matrix says ${actual[key]}, sc-15812 says ${want}.\n` +
          `        The catalog changed. Re-derive the split and update EXPECT before applying.`,
      );
    }
  }
  // A dual model whose family is not itself dual would strand the Candle story with no parent.
  for (const m of dualModels) {
    if (!dualFamilies.includes(m.family)) die(`${m.model}: dual model under non-dual family SC-${m.family}`);
  }

  return {
    models,
    dualModels,
    mlxOnly,
    candleOnly,
    dualFamilies,
    mlxOnlyFamilies,
    candleFamilyByFamily,
    actual,
  };
}

// ── Shortcut client — minimal, rate-limited, never logs the token ───────────────────────────────
function token() {
  const t = process.env.CLAUDE_SHORTCUT_MCP_TOKEN || process.env.SHORTCUT_API_TOKEN;
  if (!t) {
    die(
      "No token. Set CLAUDE_SHORTCUT_MCP_TOKEN (or SHORTCUT_API_TOKEN).\n" +
        "        --dry-run needs no token; --apply and --verify do.",
    );
  }
  return t;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function api(method, path, body, tok) {
  // Shortcut allows 200 req/min; stay well under it rather than discovering the limit mid-migration.
  await sleep(350);
  const res = await fetch(`${API}${path}`, {
    method,
    headers: { "Content-Type": "application/json", "Shortcut-Token": tok },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    // Deliberately does not include the request body: story descriptions are large and a failure
    // dump that scrolls the real error off screen is how a 409 gets mistaken for a duplicate.
    throw new Error(`${method} ${path} -> ${res.status} ${res.statusText} ${text.slice(0, 300)}`);
  }
  return res.status === 204 ? null : res.json();
}

function loadState() {
  return existsSync(STATE) ? JSON.parse(readFileSync(STATE, "utf8")) : { candleFamilies: {}, candleModels: {}, rescoped: [], links: [] };
}
function saveState(state) {
  writeFileSync(STATE, `${JSON.stringify(state, null, 2)}\n`);
}

const SCOPE_MLX = (twinId) =>
  `BACKEND SCOPE — MLX/Metal ONLY. Evidence must be produced on a Mac. The Candle/CUDA half of ` +
  `this work is SC-${twinId} and is NOT covered here; do not mark this story Done on the strength ` +
  `of Candle results, and do not close SC-${twinId} from a Mac.`;

const SCOPE_CANDLE = (twinId) =>
  `BACKEND SCOPE — Candle/CUDA ONLY. Evidence must be produced on CUDA hardware. The MLX/Metal ` +
  `half is SC-${twinId}. Split from it under sc-15812 because a dual-backend story cannot be ` +
  `completed on one machine, which is how Candle work goes missing once MLX is green.`;

function candleName(mlxName) {
  return `${mlxName} — Candle/CUDA`;
}
function mlxName(name) {
  return name.endsWith(" — MLX/Metal") ? name : `${name} — MLX/Metal`;
}

// ── phase B: apply ─────────────────────────────────────────────────────────────────────────────
async function apply(plan) {
  const tok = token();
  const state = loadState();

  // Resolve every twin we intend to create, so a re-run adopts what already exists instead of
  // creating a second copy. Names are the natural key; they are unique within the epic.
  const epicStories = await api("GET", `/epics/${EPIC_ID}/stories`, null, tok);
  const byName = new Map(epicStories.map((s) => [s.name, s]));

  const createTwin = async (mlxId, kind) => {
    if (state[kind][mlxId]) {
      const existing = epicStories.find((s) => s.id === state[kind][mlxId]);
      if (existing) return state[kind][mlxId];
      console.warn(`  state names SC-${state[kind][mlxId]} for SC-${mlxId} but it is gone; recreating`);
    }
    const twin = await api("GET", `/stories/${mlxId}`, null, tok);
    const name = candleName(mlxName(twin.name).replace(" — MLX/Metal", ""));
    const adopted = byName.get(name);
    if (adopted) {
      state[kind][mlxId] = adopted.id;
      saveState(state);
      console.log(`  adopt   SC-${adopted.id}  ${name}`);
      return adopted.id;
    }
    const created = await api(
      "POST",
      "/stories",
      {
        name,
        story_type: twin.story_type,
        epic_id: EPIC_ID,
        group_id: twin.group_id ?? undefined,
        workflow_state_id: TODO_STATE,
        description: `## ${SCOPE_CANDLE(mlxId)}\n\n---\n\n${twin.description}`,
      },
      tok,
    );
    // The HARD AC calibration task (epic comment 15674) must exist on BOTH twins; a Candle story
    // without it can be closed on harness output alone, which is the defect that task exists for.
    for (const task of twin.tasks ?? []) {
      await api("POST", `/stories/${created.id}/tasks`, { description: task.description }, tok);
    }
    state[kind][mlxId] = created.id;
    saveState(state);
    console.log(`  create  SC-${created.id}  ${name}`);
    return created.id;
  };

  const link = async (subject, object) => {
    const key = `${subject}->${object}`;
    if (state.links.includes(key)) return;
    try {
      await api("POST", "/story-links", { subject_id: subject, verb: "blocks", object_id: object }, tok);
    } catch (error) {
      // A 409 here is ambiguous: it may mean "already linked" or a genuine failure. Verify against
      // the live story rather than assuming, because assuming is how an edge silently goes missing.
      const live = await api("GET", `/stories/${object}`, null, tok);
      const present = (live.story_links ?? []).some((l) => l.subject_id === subject && l.object_id === object);
      if (!present) throw error;
    }
    state.links.push(key);
    saveState(state);
  };

  const rescope = async (mlxId, twinId) => {
    if (state.rescoped.includes(mlxId)) return;
    const story = await api("GET", `/stories/${mlxId}`, null, tok);
    if (story.name !== mlxName(story.name)) {
      await api("PUT", `/stories/${mlxId}`, { name: mlxName(story.name) }, tok);
    }
    // A TASK, not a description edit: the descriptions are generated by sc-15506, and epic comment
    // 15674 established that convention so regeneration does not clobber hand edits.
    const already = (story.tasks ?? []).some((t) => t.description.startsWith("BACKEND SCOPE"));
    if (!already) await api("POST", `/stories/${mlxId}/tasks`, { description: SCOPE_MLX(twinId) }, tok);
    state.rescoped.push(mlxId);
    saveState(state);
    console.log(`  rescope SC-${mlxId} -> MLX/Metal`);
  };

  console.log("\nFamilies");
  for (const familyId of plan.dualFamilies) {
    const twinId = await createTwin(familyId, "candleFamilies");
    for (const gate of FAMILY_GATES) await link(gate, twinId);
    await rescope(familyId, twinId);
  }

  console.log("\nModels");
  for (const m of plan.dualModels) {
    const twinId = await createTwin(m.mlxStory, "candleModels");
    const candleFamily = state.candleFamilies[m.family];
    if (!candleFamily) die(`no Candle family recorded for SC-${m.family}; re-run after families succeed`);
    await link(candleFamily, twinId);
    await rescope(m.mlxStory, twinId);
  }

  console.log(`\nApplied. State: ${STATE}`);
}

// ── phase C: verify ────────────────────────────────────────────────────────────────────────────
async function verify(plan) {
  const tok = token();
  const state = loadState();
  const epicStories = await api("GET", `/epics/${EPIC_ID}/stories`, null, tok);
  const byId = new Map(epicStories.map((s) => [s.id, s]));
  const problems = [];

  // The matrix is the authority on which twin id belongs to which MLX story; the state file is only a
  // cross-check, because it is gitignored and therefore absent on a fresh clone.
  const expected = (fromMatrix, fromState, label) => {
    if (fromState && fromMatrix && fromState !== fromMatrix) {
      problems.push(`${label}: matrix names SC-${fromMatrix} but the state file names SC-${fromState}`);
    }
    return fromMatrix ?? fromState;
  };
  for (const familyId of plan.dualFamilies) {
    const twin = expected(
      plan.candleFamilyByFamily[familyId],
      state.candleFamilies[familyId],
      `family SC-${familyId}`,
    );
    if (!twin || !byId.has(twin)) problems.push(`family SC-${familyId}: no Candle twin`);
  }
  for (const m of plan.dualModels) {
    const twin = expected(m.candleStory, state.candleModels[m.mlxStory], `model SC-${m.mlxStory} (${m.model})`);
    if (!twin || !byId.has(twin)) problems.push(`model SC-${m.mlxStory} (${m.model}): no Candle twin`);
    const mlx = byId.get(m.mlxStory);
    if (mlx && !mlx.name.endsWith("— MLX/Metal")) problems.push(`SC-${m.mlxStory}: not rescoped to MLX`);
  }
  // No Candle work has been done yet, so a Candle twin in a completed state is a bug, not progress.
  // RELAX THIS the day genuine Candle evidence starts landing — but not before, because the state
  // was silently inherited from the MLX twin once already (SC-15815) and read as finished work.
  const everyTwin = new Set([
    ...Object.values(state.candleFamilies),
    ...Object.values(state.candleModels),
    ...Object.values(plan.candleFamilyByFamily),
    ...plan.dualModels.map((m) => m.candleStory),
  ]);
  for (const twin of everyTwin) {
    if (!twin) continue;
    const s = byId.get(twin);
    if (s?.completed) problems.push(`SC-${twin}: Candle twin is marked complete but no Candle work has landed`);
  }

  // The mlx-only set must be untouched: a Candle twin there could never be closed.
  for (const m of plan.mlxOnly) {
    if (state.candleModels[m.mlxStory]) problems.push(`SC-${m.mlxStory} (${m.model}): mlx-only but has a Candle twin`);
    const s = byId.get(m.mlxStory);
    if (s && s.name.endsWith("— MLX/Metal")) problems.push(`SC-${m.mlxStory}: mlx-only should not be renamed`);
  }

  if (problems.length) {
    console.error(`\n  ${problems.length} problem(s):`);
    for (const p of problems) console.error(`    - ${p}`);
    process.exit(1);
  }
  console.log(`\n  OK — ${plan.dualFamilies.length} family twins, ${plan.dualModels.length} model twins, ${plan.mlxOnly.length} mlx-only untouched.`);
}

// ── main ───────────────────────────────────────────────────────────────────────────────────────
const mode = process.argv.find((a) => ["--dry-run", "--apply", "--verify"].includes(a)) ?? "--dry-run";
const plan = buildPlan();

if (mode === "--dry-run") {
  console.log("\nInvariants (matrix vs sc-15812)");
  for (const [k, v] of Object.entries(plan.actual)) console.log(`  ok  ${k.padEnd(18)} ${v}`);

  console.log(`\nWould create ${plan.dualFamilies.length} Candle family stories:`);
  for (const f of plan.dualFamilies) {
    const kids = plan.dualModels.filter((m) => m.family === f);
    console.log(`  SC-${f} -> Candle twin, blocking ${kids.length} model twin(s)`);
  }
  console.log(`\nWould create ${plan.dualModels.length} Candle model stories:`);
  for (const m of plan.dualModels) console.log(`  SC-${m.mlxStory}  fam SC-${m.family}  ${m.model}`);

  console.log(`\nWould rescope ${plan.dualFamilies.length + plan.dualModels.length} existing stories to MLX/Metal.`);
  console.log(`Would leave UNTOUCHED: ${plan.mlxOnly.length} mlx-only models, ${plan.mlxOnlyFamilies.length} mlx-only families (SC-${plan.mlxOnlyFamilies.join(", SC-")}).`);

  const writes =
    (plan.dualFamilies.length + plan.dualModels.length) * 2 +
    plan.dualFamilies.length * FAMILY_GATES.length +
    plan.dualModels.length +
    (plan.dualFamilies.length + plan.dualModels.length);
  console.log(`\nApprox ${writes} writes. Re-runnable; state in ${STATE}`);

  writeFileSync(PLAN, `${JSON.stringify(plan, null, 2)}\n`);
  console.log(`Plan written to ${PLAN}`);
} else if (mode === "--apply") {
  await apply(plan);
} else {
  await verify(plan);
}
