// Syntax + portability gate for the repo's shell scripts.
//
// WHY: the repo's shell surface had no CI coverage at all — `check:docker:runpod:supervisor`
// runs one script's self-test, and nothing parsed the rest. scripts/setup-nax-runner.sh is
// several hundred lines of provisioning bash that must run under the bash macOS actually
// ships, and three of its defects were found by executing it rather than reading it.
//
// TWO CHECKS, because one alone is not enough:
//
//  1. `bash -n` on every script. Catches outright syntax errors.
//
//  2. A scan for bash-4-only constructs. This is the part `bash -n` CANNOT do here: this
//     gate runs on ubuntu-latest, whose bash is 5.x, so a bash-4 idiom parses cleanly in
//     CI and then explodes on a developer's or self-hosted runner's Mac. macOS ships bash
//     3.2.57 as /bin/bash (it has not shipped a newer GPL-licensed bash since 2007), and
//     `#!/usr/bin/env bash` resolves to exactly that on a stock Mac.
//
// SCRIPTS ARE CLASSIFIED BY WHERE THEY RUN, because a single rule cannot be right for
// both. `docker/runpod-entrypoint.sh` legitimately uses `[[ -v VAR ]]` (bash 4.2): it only
// ever executes inside a Linux container whose bash is 5.x. Holding it to 3.2 would be a
// false positive, while holding scripts/*.sh to 5.x would miss the real thing. Worse, a
// naive `bash -n` is PLATFORM-DEPENDENT — run on a Mac it rejects that container script,
// run on CI's ubuntu it accepts a bash-4 idiom in a macOS-only script. Both directions are
// wrong, so the classification below is what makes the gate mean the same thing anywhere.
//
// KNOWN GAP, stated rather than papered over: the specific bug that motivated this gate —
// expanding an empty array as "${a[@]}" under `set -u`, which is an "unbound variable"
// error on bash 3.2 but fine on bash 4.4+ — is not statically detectable without real
// dataflow analysis, since whether the array is ever empty depends on runtime branches.
// The idiom scan below cannot see it. Running a script's own `--check`/`--self-test` on a
// macOS lane is what actually catches that class.
//
// Usage:
//   node scripts/check-shell-scripts.mjs
//   node scripts/check-shell-scripts.mjs --self-test

import { execFileSync, spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import path from "node:path";

const ROOT = path.resolve(import.meta.dirname, "..");

// Constructs that are bash 4+ only and therefore fail on macOS's /bin/bash 3.2. Each entry
// is zero-false-positive by construction: these are syntax, not naming.
const BASH4_ONLY = [
  {
    // `declare -A` / `local -A` — associative arrays landed in bash 4.0.
    pattern: /^[^#]*\b(?:declare|local|typeset)\s+-[A-Za-z]*A[A-Za-z]*\b/,
    what: "associative array (declare/local -A)",
    since: "bash 4.0",
  },
  {
    // `mapfile` / `readarray` — bash 4.0.
    pattern: /^[^#]*\b(?:mapfile|readarray)\b/,
    what: "mapfile/readarray",
    since: "bash 4.0",
  },
  {
    // ${v^} ${v^^} ${v,} ${v,,} case modification — bash 4.0.
    pattern: /\$\{[A-Za-z_][A-Za-z0-9_]*(?:\[[^\]]*\])?(?:\^\^?|,,?)\}/,
    what: "case-modification expansion (${v^^} / ${v,,})",
    since: "bash 4.0",
  },
  {
    // `&>>` append-both-streams — bash 4.0. (`&>` alone is fine in 3.2.)
    pattern: /(?:^|[^0-9<>&])&>>/,
    what: "&>> append redirection",
    since: "bash 4.0",
  },
  {
    // `;;&` case fallthrough — bash 4.0.
    pattern: /;;&/,
    what: ";;& case fallthrough",
    since: "bash 4.0",
  },
  {
    // ${v:offset:len} on arrays is fine; negative substring `${v: -1}` is 3.2-safe too.
    // `shopt -s globstar` (**) is bash 4.0.
    pattern: /^[^#]*\bshopt\s+-s\s+globstar\b/,
    what: "globstar",
    since: "bash 4.0",
  },
  {
    // `[[ -v name ]]` — bash 4.2. On 3.2 this is not merely unsupported, it is a SYNTAX
    // error ("conditional binary operator expected"), so it kills the whole script rather
    // than one branch. The 3.2-safe form is `[[ -n "${name+x}" ]]`.
    pattern: /\[\[\s+-v\s/,
    what: "[[ -v name ]] variable-set test",
    since: "bash 4.2",
  },
];

// Scripts that only ever run inside a Linux container (bash 5.x) and are therefore exempt
// from the bash-3.2 portability scan. Everything else is treated as portable: the scripts/
// directory is developer- and CI-facing on both macOS and Linux, and setup-nax-runner.sh is
// macOS-only by definition. Keep this list minimal and justified — an entry here is an
// assertion that no Mac ever sources the file.
const LINUX_ONLY = new Set([
  // Executed as the RunPod image entrypoint; never run on a developer's Mac.
  "docker/runpod-entrypoint.sh",
]);

function localBashMajor() {
  const result = spawnSync("bash", ["-c", "echo $BASH_VERSINFO"], { encoding: "utf8" });
  const major = Number.parseInt((result.stdout || "").trim(), 10);
  return Number.isFinite(major) ? major : 0;
}

function trackedShellScripts() {
  const out = execFileSync("git", ["ls-files", "-z", "*.sh"], {
    cwd: ROOT,
    encoding: "utf8",
    maxBuffer: 1 << 24,
  });
  return out.split("\0").filter(Boolean);
}

function scanForBash4(relativePath, body) {
  const problems = [];
  const lines = body.split("\n");
  for (const [index, line] of lines.entries()) {
    for (const rule of BASH4_ONLY) {
      if (rule.pattern.test(line)) {
        problems.push(
          `${relativePath}:${index + 1} uses ${rule.what} (${rule.since}), which fails on ` +
            `macOS's /bin/bash 3.2: ${line.trim()}`,
        );
      }
    }
  }
  return problems;
}

function parseCheck(relativePath) {
  const result = spawnSync("bash", ["-n", path.join(ROOT, relativePath)], {
    encoding: "utf8",
  });
  if (result.error) {
    throw new Error(`could not run \`bash -n\` for ${relativePath}: ${result.error.message}`);
  }
  if (result.status !== 0) {
    return [`${relativePath} failed \`bash -n\`:\n${(result.stderr || "").trim()}`];
  }
  return [];
}

function selfTest() {
  // Prove the idiom scan actually fires, so a green gate is not a gate that scans nothing.
  const cases = [
    ["declare -A map", "associative array"],
    ["  local -A thing", "associative array"],
    ["mapfile -t lines < file", "mapfile"],
    ['echo "${name^^}"', "case-modification"],
    ["cmd &>> log.txt", "&>>"],
    ["  ;;&", ";;& case fallthrough"],
    ["shopt -s globstar", "globstar"],
  ];
  for (const [line, expected] of cases) {
    const found = scanForBash4("self-test.sh", line);
    if (found.length === 0) {
      throw new Error(`self-test: expected to flag ${expected} in: ${line}`);
    }
  }
  // And that it does NOT fire on 3.2-safe equivalents, including a commented-out idiom.
  const safe = [
    'declare -a list',
    'local -r frozen="x"',
    'printf "%s" "${name}"',
    'cmd >log.txt 2>&1',
    'cmd &> log.txt',
    '# declare -A map',
    'value="${a[@]+"${a[@]}"}"',
    'echo "${path%/*}"',
  ];
  for (const line of safe) {
    const found = scanForBash4("self-test.sh", line);
    if (found.length > 0) {
      throw new Error(`self-test: false positive on 3.2-safe line: ${line}\n  ${found[0]}`);
    }
  }
  console.log(`SceneWorks shell-script gate self-test passed (${cases.length} positive, ${safe.length} negative).`);
}

if (process.argv.includes("--self-test")) {
  selfTest();
  process.exit(0);
}

const scripts = trackedShellScripts();
if (scripts.length === 0) {
  throw new Error("no tracked *.sh files found — the discovery glob is probably wrong");
}

const bashMajor = localBashMajor();
const problems = [];
const skipped = [];

for (const relativePath of scripts) {
  const linuxOnly = LINUX_ONLY.has(relativePath);

  // `bash -n` uses whatever bash is on this host. For a container-only script that is
  // allowed to use bash-4 syntax, a 3.2 host would report a spurious syntax error — so
  // only trust the parse for those when the local bash can actually represent them.
  if (linuxOnly && bashMajor < 4) {
    skipped.push(`${relativePath} (container-only; local bash ${bashMajor}.x cannot parse bash 4+ syntax)`);
  } else {
    problems.push(...parseCheck(relativePath));
  }

  if (!linuxOnly) {
    problems.push(...scanForBash4(relativePath, readFileSync(path.join(ROOT, relativePath), "utf8")));
  }
}

if (problems.length > 0) {
  console.error(`SceneWorks shell-script check FAILED (${problems.length} problem(s)):\n`);
  for (const problem of problems) console.error(`  - ${problem}`);
  process.exit(1);
}

for (const note of skipped) {
  console.log(`  (parse skipped) ${note}`);
}
const portable = scripts.length - LINUX_ONLY.size;
console.log(
  `SceneWorks shell-script check passed (${scripts.length} scripts; ` +
    `${portable} held to bash-3.2 portability, ${LINUX_ONLY.size} container-only).`,
);
