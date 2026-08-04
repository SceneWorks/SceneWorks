#!/usr/bin/env node
/**
 * Answer, for a `~/.cache/huggingface/hub/models--*` directory: is anything in the installed
 * SceneWorks app still using it? (sc-17335, following sc-17271 / sc-17283.)
 *
 * WHY THIS EXISTS. On a dev Mac that is also a CI runner, that cache backs BOTH the real-weight
 * CI lanes and the locally installed app — `crates/sceneworks-core/src/hf_home.rs` points
 * `HF_HOME` at the OS Hugging Face home whenever `HF_HUB_CACHE` / `HUGGINGFACE_HUB_CACHE` /
 * `HF_HOME` are all unset, which is how the app runs natively. Reclaiming disk by deleting a
 * `models--*` directory that CI no longer needs can therefore silently uninstall something the
 * app is using: the app's own receipt survives the deletion, so it keeps reporting the thing as
 * installed and fails at load time instead.
 *
 * FOUR INDEPENDENT WAYS A REPO CAN BE APP-LIVE, and each is invisible to the others. Every
 * reclaim so far has re-derived them by hand and each has missed at least one:
 *
 *   1. MODEL INSTALL MARKER — `data/models/<Org>__<Repo>/.sceneworks-download-complete.json`
 *      (`safe_download_dir`, apps/rust-api/src/lib.rs). The directory holds ONLY that ~4 KB
 *      marker; the weights are in the cache. A model directory that looks empty is an INSTALLED
 *      model.
 *
 *   2. CO-REQUISITE OF AN INSTALLED MODEL — a pinned `"coRequisite": true` download never gets a
 *      marker of its own, because it belongs to its parent model's install. sc-17271 nearly
 *      deleted 6.6 GB of `MOSS-Audio-Tokenizer` on exactly this mistake.
 *
 *   3. LORA RECEIPT — `data/loras/<loraId>/`, keyed by LORA ID rather than by repo, so the model
 *      test returns "no marker" for a LoRA the app is rendering with (sc-17335). Measured here:
 *      `krea2_identity_edit` (conradlocke/krea2-identity-edit) and `krea2_turbo_accel`
 *      (Comfy-Org/Krea-2) hold NO local weights at all.
 *
 *   4. A REPO ID HARDCODED IN THE SOURCE, fetched on demand into this same cache and named in no
 *      manifest at all. This class is deliberate and documented — builtin.models.jsonc says of
 *      the InstantID adapter weights that they "are fetched on demand by the adapter, so they are
 *      not listed here". `QWEN_LIGHTNING_LORA_REPO`
 *      (crates/sceneworks-worker/src/image_jobs/qwen.rs) is the same shape: an installed model's
 *      required distill LoRA, materialized into the hub cache by the worker, declared nowhere.
 *      An earlier version of this script reported both as safe to delete.
 *
 * WHAT THIS TOOL CANNOT DO, STATED PLAINLY. It finds REFERENCES; it cannot prove their absence.
 * The source scan matches whole string literals, so a repo id assembled at runtime
 * (`format!("SceneWorks/{name}")`) is invisible to it. That is why the terminal verdict is
 * `unreferenced` and NOT "safe to delete", and why no field named `safeToDelete` exists: this
 * narrows 129 cached repos to a short candidate list for a human to confirm, and nothing more.
 * `unreferenced` also says nothing whatsoever about whether CI still needs the repo, or whether
 * another machine holds a copy — both are separate questions the operator answers per lane.
 *
 * NOT A LIVENESS SIGNAL, DELIBERATELY: `.sceneworks-model-revision` (written inside
 * `snapshots/<rev>/` by the inference repo's release tooling, never by the app). Its PRESENCE
 * shows a snapshot was CI-provisioned; its ABSENCE shows nothing. Reported as corroboration only.
 *
 * FAILS CLOSED. A missing app data dir or cache directory ABORTS rather than quietly yielding
 * zero markers and promoting live weights to `unreferenced` — an earlier version turned a typo in
 * `--data-dir` into 89 GiB of installed models looking unreferenced.
 */

import { readFile, readdir, stat } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { stripJsoncComments } from "./lib/jsonc.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

export const VERDICT = {
  MODEL: "app-live:model",
  COREQUISITE: "app-live:co-requisite",
  LORA: "app-live:lora",
  SOURCE: "app-live:source",
  DECLARED: "declared:not-installed",
  // Terminal verdict. NOT "safe to delete" -- see the header. It means the four mechanical
  // reference tests found nothing, which is the absence of evidence, not evidence of absence.
  UNREFERENCED: "unreferenced",
};

/** `models--Comfy-Org--Krea-2` -> `Comfy-Org/Krea-2`. */
export function repoFromCacheDir(name) {
  if (!name.startsWith("models--")) return null;
  const rest = name.slice("models--".length);
  const separator = rest.indexOf("--");
  if (separator <= 0) return null;
  // Only the FIRST `--` splits org from name: `models--Lightricks--LTX-2.3-22b-IC-LoRA-HDR`
  // is one org and one repo whose own name contains no `--`, but a repo name legitimately can.
  return `${rest.slice(0, separator)}/${rest.slice(separator + 2)}`;
}

/** `Comfy-Org/Krea-2` -> `Comfy-Org__Krea-2`, the app's `data/models/` directory name. */
export function appModelDirName(repo) {
  return repo.replace("/", "__");
}

function* walk(node) {
  if (Array.isArray(node)) {
    for (const item of node) yield* walk(item);
  } else if (node && typeof node === "object") {
    yield node;
    for (const value of Object.values(node)) yield* walk(value);
  }
}

/**
 * Map every repo declared in `builtin.models.jsonc` to the model(s) that download it.
 *
 * A model's PRIMARY repos are its non-co-requisite downloads — those are what the install marker
 * is keyed by. Its co-requisites ride on the same install and never get their own marker, which
 * is the whole reason this indirection exists.
 */
export function modelDownloadIndex(manifest) {
  const byRepo = new Map();
  for (const node of walk(manifest)) {
    if (!Array.isArray(node.downloads)) continue;
    const modelId = typeof node.id === "string" ? node.id : null;
    if (!modelId) continue;
    const primary = new Set();
    for (const download of node.downloads) {
      if (!download || typeof download.repo !== "string") continue;
      if (!download.coRequisite) primary.add(download.repo);
    }
    for (const download of node.downloads) {
      if (!download || typeof download.repo !== "string") continue;
      const entry = byRepo.get(download.repo) ?? [];
      entry.push({
        modelId,
        coRequisite: Boolean(download.coRequisite),
        componentId: download.componentId ?? null,
        primaryRepos: [...primary],
      });
      byRepo.set(download.repo, entry);
    }
  }
  return byRepo;
}

/** A `<org>/<repo>` value. Used to tell a repo id from a path or filename. */
export const REPO_SHAPE = /^[A-Za-z0-9][\w.-]*\/[\w.-]+$/;

/** Manifest keys whose value is a Hugging Face repo id. Pinned by a test so a new one reds. */
export const REPO_KEY = /repo(sitory)?$/i;

/**
 * Map every repo declared by ANY manifest to the declarations that name it.
 *
 * Two deliberate generalizations, each from a real miss:
 *
 *  - Scans every `config/manifests/*.jsonc` rather than an allow-list. The first version indexed
 *    the models and loras manifests only and reported `SceneWorks/krea2-pose-controlnet-beta` as
 *    unreferenced — it is declared in `builtin.control_overlays.jsonc`.
 *  - Scans every key matching `REPO_KEY`, not just `repo`. The second version read `repo` and
 *    `source.repo` only and reported `mlx-community/gemma-3-12b-it-bf16` (25 GiB) as
 *    unreferenced — it is declared as `textEncoderRepo`, and `convertBaseRepo` /
 *    `convertSourceRepo` (which models.rs actually consumes) were indexed only by luck.
 */
export function manifestRepoIndex(manifests) {
  const byRepo = new Map();
  for (const [manifestName, manifest] of Object.entries(manifests)) {
    for (const node of walk(manifest)) {
      const id = typeof node.id === "string" ? node.id : null;
      for (const [key, value] of Object.entries(node)) {
        if (typeof value !== "string" || !REPO_KEY.test(key) || !REPO_SHAPE.test(value.trim())) continue;
        const entry = byRepo.get(value.trim()) ?? { manifests: new Set(), ids: [] };
        entry.manifests.add(manifestName);
        if (id && !entry.ids.includes(id)) entry.ids.push(id);
        byRepo.set(value.trim(), entry);
      }
    }
  }
  return byRepo;
}

/**
 * Collect every `<org>/<repo>`-shaped string literal in the app source.
 *
 * This is the only mechanical handle on liveness source 4. It is intentionally over-broad --
 * `"image/png"` matches the shape too -- because the result is only ever INTERSECTED with repos
 * that are actually in the cache, so a false positive costs nothing and a false negative could
 * cost a user their weights. What it cannot see is a repo id assembled at runtime; the header
 * says so, and it is why the terminal verdict is `unreferenced` rather than "safe".
 */
export function sourceLiteralIndex(files) {
  const byRepo = new Map();
  const literal = /"([A-Za-z0-9][\w.-]*\/[\w.-]+)"/g;
  for (const [file, body] of Object.entries(files)) {
    for (const match of body.matchAll(literal)) {
      const entry = byRepo.get(match[1]) ?? [];
      if (!entry.includes(file)) entry.push(file);
      byRepo.set(match[1], entry);
    }
  }
  return byRepo;
}

/**
 * Classify each cached repo. Pure — every filesystem fact arrives as plain data so the tests can
 * feed fixtures and mutations instead of depending on whatever this particular box happens to
 * have installed.
 *
 * @param {object} input
 * @param {string[]} input.repos              repo ids present in the cache
 * @param {Set<string>} input.installedModels repo ids with a `data/models/<Org>__<Repo>/` marker
 * @param {{loraId: string, repo: string}[]} input.loraReceipts  installed LoRAs naming a repo
 * @param {Map<string, object[]>} input.modelDownloads  from `modelDownloadIndex`
 * @param {Map<string, object>} input.manifestDeclarations from `manifestRepoIndex`
 * @param {Map<string, string[]>} [input.sourceLiterals] from `sourceLiteralIndex`
 * @param {Set<string>} [input.ciProvisioned] repo ids carrying `.sceneworks-model-revision`
 */
export function classify({
  repos,
  installedModels,
  loraReceipts,
  modelDownloads,
  manifestDeclarations,
  sourceLiterals = new Map(),
  testLiterals = new Map(),
  ciProvisioned = new Set(),
}) {
  const loraByRepo = new Map();
  for (const receipt of loraReceipts) {
    const entry = loraByRepo.get(receipt.repo) ?? [];
    entry.push(receipt.loraId);
    loraByRepo.set(receipt.repo, entry);
  }

  return repos.map((repo) => {
    const ci = ciProvisioned.has(repo);

    if (installedModels.has(repo)) {
      return row(repo, VERDICT.MODEL, `install marker at data/models/${appModelDirName(repo)}/`, ci);
    }

    // Co-requisite: app-live only if the PARENT model is installed. A co-requisite of something
    // nobody installed is not keeping this repo alive, and saying otherwise would pin weights
    // forever on the strength of a manifest entry alone.
    for (const owner of modelDownloads.get(repo) ?? []) {
      const installedParent = owner.primaryRepos.find((parent) => installedModels.has(parent));
      if (!installedParent) continue;
      const role = owner.coRequisite
        ? `co-requisite (${owner.componentId ?? "component"}) of`
        : "download of";
      return row(repo, VERDICT.COREQUISITE, `${role} installed model ${owner.modelId} (${installedParent})`, ci);
    }

    const installedLoras = loraByRepo.get(repo);
    if (installedLoras?.length) {
      return row(repo, VERDICT.LORA, `installed LoRA receipt: ${installedLoras.join(", ")}`, ci);
    }

    const declaration = manifestDeclarations.get(repo);
    if (declaration) {
      const where = [...declaration.manifests].sort().join(", ");
      const named = declaration.ids.length ? ` (${declaration.ids.join(", ")})` : "";
      return row(
        repo,
        VERDICT.DECLARED,
        `declared in ${where}${named} but not installed — deleting costs a re-download`,
        ci,
      );
    }

    // Backstop, checked last: a manifest entry is the more specific explanation when one exists,
    // so this catches only what NO manifest declares -- which is exactly the on-demand class
    // (`QWEN_LIGHTNING_LORA_REPO`, the InstantID adapter weights) that the first version missed.
    const cited = sourceLiterals.get(repo);
    if (cited?.length) {
      const more = cited.length > 2 ? ` (+${cited.length - 2} more)` : "";
      return row(repo, VERDICT.SOURCE, `repo id hardcoded in ${cited.slice(0, 2).join(", ")}${more}`, ci);
    }

    const inTests = testLiterals.get(repo);
    const how = ci ? "; carries the CI provenance marker" : "";
    const testNote = inTests?.length ? `; named only in test sources (${inTests.slice(0, 2).join(", ")}) — likely a fixture` : "";
    return row(repo, VERDICT.UNREFERENCED, `no marker, receipt, manifest entry or source literal found${how}${testNote}`, ci);
  });
}

function row(repo, verdict, reason, ciProvisioned) {
  // Deliberately NOT named `safeToDelete`. The tool establishes that no reference was found, which
  // is a candidate for a human to confirm -- never a clearance to delete.
  return { repo, verdict, reason, ciProvisioned, referenced: verdict !== VERDICT.UNREFERENCED };
}

async function readJsonc(file) {
  const body = await readFile(file, "utf8");
  try {
    return JSON.parse(stripJsoncComments(body));
  } catch (error) {
    throw new Error(`${path.relative(ROOT, file)} is not parseable JSONC: ${error.message}`);
  }
}

/** Load every manifest in `config/manifests/`. Exported so the tests exercise THIS loader. */
export async function loadManifests(dir = path.join(ROOT, "config/manifests")) {
  const manifests = {};
  for (const name of (await readdir(dir)).sort()) {
    if (name.endsWith(".jsonc")) manifests[name] = await readJsonc(path.join(dir, name));
  }
  return manifests;
}

/** Source roots that can name a Hugging Face repo the app fetches on demand (liveness source 4). */
const SOURCE_ROOTS = ["crates", "apps/rust-api/src", "apps/web/src", "apps/desktop/src"];
const SOURCE_EXTENSIONS = new Set([".rs", ".ts", ".tsx", ".js", ".jsx", ".mjs"]);

/**
 * Split deliberately: a repo id that appears ONLY in a test is a fixture, not something the app
 * fetches, and letting test files vote would mark half the cache app-live and bury the on-demand
 * class this scan exists to catch. Test mentions are still reported on the `unreferenced` line so
 * a human confirming a candidate can see them.
 */
export function isTestSource(relativePath) {
  return (
    /(^|\/)tests?\//.test(relativePath) ||
    /\.(test|spec)\.[^.]+$/.test(relativePath) ||
    /(^|\/)fixtures?\//.test(relativePath)
  );
}

export async function loadSourceFiles(roots = SOURCE_ROOTS) {
  const files = {};
  async function visit(dir) {
    for (const entry of await readdir(dir, { withFileTypes: true }).catch(() => [])) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        if (entry.name === "node_modules" || entry.name === "target" || entry.name === ".git") continue;
        await visit(full);
      } else if (SOURCE_EXTENSIONS.has(path.extname(entry.name))) {
        files[path.relative(ROOT, full)] = await readFile(full, "utf8").catch(() => "");
      }
    }
  }
  for (const root of roots) await visit(path.join(ROOT, root));
  const production = {};
  const tests = {};
  for (const [file, body] of Object.entries(files)) (isTestSource(file) ? tests : production)[file] = body;
  return { production, tests };
}

async function exists(target) {
  try {
    await stat(target);
    return true;
  } catch {
    return false;
  }
}

/**
 * Read this box. FAILS CLOSED: a missing app data dir or cache is an abort, never an empty
 * result. Quietly returning zero install markers promotes every installed model to
 * `unreferenced` -- a `--data-dir` typo did exactly that, to 89 GiB of live weights.
 */
async function readBox(dataDir, hubDir) {
  const modelsDir = path.join(dataDir, "models");
  for (const [label, dir] of [["app data dir", modelsDir], ["Hugging Face cache", hubDir]]) {
    if (!(await exists(dir))) {
      throw new Error(
        `${label} not found: ${dir}\nRefusing to classify — with no ${label} every live repo would look unreferenced.`,
      );
    }
  }

  const repos = [];
  const unclassifiedDirs = [];
  for (const entry of await readdir(hubDir, { withFileTypes: true })) {
    if (!entry.isDirectory() || entry.name === ".locks") continue;
    const repo = repoFromCacheDir(entry.name);
    if (repo) repos.push(repo);
    // `datasets--*` / `spaces--*` are cache entries this tool has no rules for. Counting them as
    // zero would understate the cache; report them rather than silently dropping them.
    else unclassifiedDirs.push(entry.name);
  }

  const installedModels = new Set();
  for (const entry of await readdir(modelsDir, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    if (await exists(path.join(modelsDir, entry.name, ".sceneworks-download-complete.json"))) {
      const separator = entry.name.indexOf("__");
      if (separator > 0) {
        installedModels.add(`${entry.name.slice(0, separator)}/${entry.name.slice(separator + 2)}`);
      }
    }
  }

  const loraReceipts = [];
  for (const entry of await readdir(path.join(dataDir, "loras"), { withFileTypes: true }).catch(() => [])) {
    if (!entry.isDirectory()) continue;
    const body = await readFile(
      path.join(dataDir, "loras", entry.name, ".sceneworks-download-complete.json"),
      "utf8",
    ).catch(() => null);
    if (!body) continue;
    let parsed;
    try {
      parsed = JSON.parse(body);
    } catch {
      continue;
    }
    // An uploaded or locally trained LoRA has `repo: null` and a `sourcePath` — its weights sit
    // beside the receipt, so it holds nothing in the cache.
    if (typeof parsed.repo === "string" && parsed.repo.trim()) {
      loraReceipts.push({ loraId: entry.name, repo: parsed.repo.trim() });
    }
  }

  const ciProvisioned = new Set();
  for (const repo of repos) {
    const snapshots = path.join(hubDir, `models--${repo.replace("/", "--")}`, "snapshots");
    for (const rev of await readdir(snapshots).catch(() => [])) {
      if (await exists(path.join(snapshots, rev, ".sceneworks-model-revision"))) {
        ciProvisioned.add(repo);
        break;
      }
    }
  }

  return { repos, installedModels, loraReceipts, ciProvisioned, unclassifiedDirs };
}

const SELF_TEST_CASES = [
  {
    name: "an installed LoRA's repo is app-live with no model marker anywhere",
    mutate: (input) => ({ ...input, installedModels: new Set() }),
    repo: "Comfy-Org/Krea-2",
    expect: VERDICT.LORA,
  },
  {
    name: "dropping the LoRA receipt downgrades it to declared, never to unreferenced",
    mutate: (input) => ({ ...input, loraReceipts: [] }),
    repo: "Comfy-Org/Krea-2",
    expect: VERDICT.DECLARED,
  },
  {
    name: "a co-requisite of an installed model is app-live",
    repo: "OpenMOSS-Team/XY_Tokenizer_TTSD_V0",
    expect: VERDICT.COREQUISITE,
  },
  {
    name: "uninstalling the parent stops the co-requisite being app-live",
    mutate: (input) => ({ ...input, installedModels: new Set() }),
    repo: "OpenMOSS-Team/XY_Tokenizer_TTSD_V0",
    expect: VERDICT.DECLARED,
  },
  {
    // The BLOCKER this tool shipped with: a repo in no manifest at all, fetched on demand by the
    // worker into this cache, was reported unreferenced.
    name: "a repo hardcoded in the source is app-live though no manifest names it",
    repo: "lightx2v/Qwen-Image-Edit-2511-Lightning",
    expect: VERDICT.SOURCE,
  },
  {
    // The regression the manifest scan was generalized for: declared only in a THIRD manifest.
    name: "a control-overlay repo is never reported unreferenced",
    repo: "SceneWorks/krea2-pose-controlnet-beta",
    expect: VERDICT.DECLARED,
  },
  {
    name: "a repo nothing references at all is the only unreferenced verdict",
    repo: "acme/definitely-not-a-real-repo",
    expect: VERDICT.UNREFERENCED,
  },
];

async function selfTest(modelDownloads, manifestDeclarations, sourceLiterals, testLiterals) {
  const base = {
    repos: [
      "Comfy-Org/Krea-2",
      "OpenMOSS-Team/XY_Tokenizer_TTSD_V0",
      "lightx2v/Qwen-Image-Edit-2511-Lightning",
      "SceneWorks/krea2-pose-controlnet-beta",
      "acme/definitely-not-a-real-repo",
    ],
    installedModels: new Set(["OpenMOSS-Team/MOSS-TTSD-v0.5"]),
    loraReceipts: [{ loraId: "krea2_turbo_accel", repo: "Comfy-Org/Krea-2" }],
    modelDownloads,
    manifestDeclarations,
    sourceLiterals,
    testLiterals,
  };
  let failures = 0;
  for (const testCase of SELF_TEST_CASES) {
    const input = testCase.mutate ? testCase.mutate(base) : base;
    const actual = classify(input).find((entry) => entry.repo === testCase.repo);
    // Assert the consequence too, not just the label: a rule that returns the right verdict but
    // the wrong `referenced` is the failure that actually deletes weights.
    const wantReferenced = testCase.expect !== VERDICT.UNREFERENCED;
    const ok = actual?.verdict === testCase.expect && actual?.referenced === wantReferenced;
    if (!ok) failures += 1;
    console.log(
      `  ${ok ? "ok  " : "FAIL"} ${testCase.name} -> ${actual?.verdict ?? "<missing>"} (referenced=${actual?.referenced})`,
    );
  }
  if (failures) {
    console.error(`self-test: ${failures} case(s) failed`);
    process.exitCode = 1;
    return;
  }
  console.log(`self-test OK: ${SELF_TEST_CASES.length} cases`);
}

function valueOf(argv, flag) {
  const index = argv.indexOf(flag);
  return index >= 0 ? argv[index + 1] : undefined;
}

async function main() {
  const argv = process.argv.slice(2);
  const home = process.env.HOME ?? "";
  const dataDir = valueOf(argv, "--data-dir") ?? path.join(home, "SceneWorks/data");
  const hubDir = valueOf(argv, "--hub") ?? path.join(home, ".cache/huggingface/hub");

  const [manifests, sourceFiles] = await Promise.all([loadManifests(), loadSourceFiles()]);
  const modelDownloads = modelDownloadIndex(manifests["builtin.models.jsonc"]);
  const manifestDeclarations = manifestRepoIndex(manifests);
  const sourceLiterals = sourceLiteralIndex(sourceFiles.production);
  const testLiterals = sourceLiteralIndex(sourceFiles.tests);

  if (argv.includes("--self-test")) {
    await selfTest(modelDownloads, manifestDeclarations, sourceLiterals, testLiterals);
    return;
  }

  const box = await readBox(dataDir, hubDir);
  const only = valueOf(argv, "--repo");
  const target = only ? (repoFromCacheDir(only) ?? only) : null;
  if (target && !box.repos.includes(target)) {
    console.error(`note: ${target} is not present in ${hubDir} — classifying it anyway.`);
  }
  const rows = classify({
    ...box,
    repos: target ? [target] : box.repos,
    modelDownloads,
    manifestDeclarations,
    sourceLiterals,
    testLiterals,
  });
  rows.sort((a, b) => a.verdict.localeCompare(b.verdict) || a.repo.localeCompare(b.repo));

  if (argv.includes("--json")) {
    console.log(JSON.stringify({ dataDir, hubDir, unclassifiedDirs: box.unclassifiedDirs, rows }, null, 2));
    return;
  }

  const width = Math.max(4, ...rows.map((entry) => entry.repo.length));
  for (const entry of rows) {
    console.log(`${entry.referenced ? "*" : " "} ${entry.repo.padEnd(width)}  ${entry.verdict.padEnd(22)}  ${entry.reason}`);
  }
  const unreferenced = rows.filter((entry) => !entry.referenced).length;
  console.log(`\ndata dir ${dataDir}\ncache    ${hubDir}`);
  if (box.unclassifiedDirs?.length) {
    console.log(`${box.unclassifiedDirs.length} cache entries this tool has no rules for (not counted): ${box.unclassifiedDirs.slice(0, 5).join(", ")}`);
  }
  console.log(
    `${rows.length} cached repos: ${rows.length - unreferenced} referenced by the app (*, KEEP), ${unreferenced} with no reference found.`,
  );
  console.log(
    "`unreferenced` is NOT a clearance to delete. It means four mechanical tests found no reference —\n" +
      "a repo id assembled at runtime is invisible to all of them. Confirm each candidate by hand, and\n" +
      "separately confirm CI no longer needs it before removing anything.",
  );
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  // A refusal to classify is an ordinary outcome here, not a crash: print the reason and exit
  // non-zero. A stack trace would bury the one line the operator needs to read.
  try {
    await main();
  } catch (error) {
    console.error(`audit-hf-cache-liveness: ${error.message}`);
    process.exitCode = 1;
  }
}
