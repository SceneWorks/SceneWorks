// Job selection for Check. Unknown inputs and incomplete diffs run every job. Required contexts
// stay unfiltered at the trigger; parity grades both selection and the expected job verdicts.
import { appendFileSync, readFileSync, readdirSync } from "node:fs";
import { execFileSync } from "node:child_process";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { dockerSmokeDecision } from "./docker-smoke-relevance.mjs";

export const CHECK_JOBS = {
  web: "web",
  "parity-rust": "rust",
  "parity-scaffold": "scaffold",
  "parity-history": "history",
  "parity-digests": "scaffold",
  "parity-docker": "docker",
  candle: "rust",
};
const OUTPUTS = [...new Set(Object.values(CHECK_JOBS))];
const all = (reason) => ({ ...Object.fromEntries(OUTPUTS.map((key) => [key, true])), reason });

// Discover Rust's actual literal include inputs across source and integration tests. Embedded
// web files and documentation are native inputs too. Runtime document reads are named below;
// concat!-based config includes remain covered by the conservative native/data roots.
export function embeddedRustInputs(root) {
  const inputs = new Set();
  function walk(directory) {
    for (const item of readdirSync(path.join(root, directory), { withFileTypes: true })) {
      const file = path.posix.join(directory, item.name);
      if (item.isDirectory()) walk(file);
      else if (item.isFile() && file.endsWith(".rs")) {
        const source = readFileSync(path.join(root, file), "utf8");
        for (const match of source.matchAll(/include_(?:str|bytes)!\s*\(\s*"([^"\n]+)"/g)) {
          const target = path.posix.normalize(path.posix.join(path.posix.dirname(file), match[1]));
          if (!target.startsWith("../") && !path.posix.isAbsolute(target)) inputs.add(target);
        }
      }
    }
  }
  for (const directory of ["crates", "apps/rust-api", "apps/rust-worker"]) walk(directory);
  return inputs;
}

export function checkDecision(files, { embeddedRust = new Set(), uncertain = false, full = false } = {}) {
  if (full || uncertain || !files.length) return all("Explicit full check or uncertain/empty diff.");
  const result = Object.fromEntries(OUTPUTS.map((key) => [key, false]));
  for (const file of files) {
    if (typeof file !== "string" || !file || file.startsWith("/") || file.includes("..") || /[\r\n]/.test(file)) {
      return all("Unrecognized changed-path encoding.");
    }
    if (/^(?:\.github\/|\.cargo\/|Cargo\.(?:toml|lock)$|rust(?:-toolchain\.toml|fmt\.toml)$|package(?:-lock)?\.json$)/.test(file)) {
      return all(`Shared build/control input: ${file}`);
    }
    if (embeddedRust.has(file)) result.rust = true;
    // These roots contain native code, embedded data, and fixtures read across the Node/web/Rust
    // boundaries. Keep them conservative: a filename extension does not prove independence.
    if (/^(?:crates\/|apps\/(?:rust-api|rust-worker|desktop)\/|vendor\/|config\/|packages\/|tests\/|release\/)/.test(file)) {
      result.rust = result.scaffold = result.history = result.web = true;
    } else if (file.startsWith("scripts/")) {
      result.scaffold = result.history = true;
      // pytest's source contracts and rust:check's generated-doc prerequisite consume scripts too.
      result.rust = true;
    } else if (file.startsWith("apps/web/")) {
      result.web = result.scaffold = true;
    } else if (/^docs\/(?:calibration\/|generated\/)/.test(file)) {
      result.rust = result.scaffold = result.history = true;
    } else if (file.startsWith("docs/") && file.endsWith(".md")) {
      // Real source/build readers, not all Markdown. New non-documentation roots fail open below.
      if (/^docs\/(?:film-script-writing|film-editor|workflow-share-envelope)\.md$/.test(file)) result.web = true;
      // fs::read_to_string and pytest runtime reads are not include macros.
      if (["docs/workflow-share-envelope.md", "docs/sc-17139-minimax-h3-hosting-layout.md"].includes(file)) result.rust = true;
      if (file === "docs/epic-20738-terminal-cuda.md") result.scaffold = true;
      if (file === "docs/calibration-runbook.md") result.history = true;
    } else if (/^(?:README|AGENTS|CODEGRAPH|CONTRIBUTING|FEATURE_DEVELOPMENT|RELEASING|CHANGELOG|LICENSE)(?:\.[^/]+)?$/.test(file)) {
      // Contributor prose has no production/test reader in these jobs.
    } else if (/^(?:docker\/|docker-compose\.yml$|\.dockerignore$)/.test(file)) {
      result.scaffold = true;
      if (file === "docker/rust.Dockerfile") result.history = true;
    } else {
      return all(`Unclassified input: ${file}`);
    }
  }
  result.docker = dockerSmokeDecision(files.length, files).run;
  return { ...result, reason: `Selected from ${files.length} changed path(s).` };
}

export function selectCheckJobs({ eventName, event, root, git = execFileSync }) {
  if (eventName !== "pull_request") return all(`${eventName}: full acceptance.`);
  try {
    const base = event.pull_request?.base?.sha;
    const head = event.pull_request?.head?.sha;
    if (![base, head].every((sha) => /^[0-9a-f]{40}$/.test(sha ?? ""))) throw new Error("missing PR revisions");
    // NUL names + --no-renames cover both sides of a rename and avoid line-oriented truncation.
    const output = git("git", ["diff", "--name-only", "--no-renames", "-z", `${base}...${head}`], { cwd: root, encoding: "utf8" });
    if (output && !output.endsWith("\0")) throw new Error("incomplete NUL-delimited diff");
    const files = output.split("\0").filter(Boolean);
    return checkDecision(files, { embeddedRust: embeddedRustInputs(root) });
  } catch (error) {
    return all(`Could not prove relevance (${error.message}); running conservatively.`);
  }
}

export function assertCheckResults(needs) {
  const expected = ["changes", "supply-chain", ...Object.keys(CHECK_JOBS)].sort();
  if (JSON.stringify(Object.keys(needs).sort()) !== JSON.stringify(expected)) throw new Error("Check needs set changed or is incomplete");
  if (needs.changes.result !== "success") throw new Error("Check selection did not succeed");
  if (needs["supply-chain"].result !== "success") throw new Error("Advisory gate did not succeed");
  for (const [job, output] of Object.entries(CHECK_JOBS)) {
    const selected = needs.changes.outputs?.[output];
    if (selected !== "true" && selected !== "false") throw new Error(`Missing/invalid selection for ${output}`);
    const result = needs[job].result;
    if (result !== "success" && !(selected === "false" && result === "skipped")) {
      throw new Error(`${job}: ${result}, selected=${selected}`);
    }
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  if (process.argv.includes("--assert-results")) {
    assertCheckResults(JSON.parse(process.env.RESULTS));
  } else {
    const decision = selectCheckJobs({ eventName: process.env.GITHUB_EVENT_NAME, event: JSON.parse(readFileSync(process.env.GITHUB_EVENT_PATH, "utf8")), root: process.cwd() });
    console.log(decision.reason);
    const output = OUTPUTS.map((key) => `${key}=${decision[key]}\n`).join("");
    if (process.env.GITHUB_OUTPUT) appendFileSync(process.env.GITHUB_OUTPUT, output);
    else process.stdout.write(output);
  }
}
