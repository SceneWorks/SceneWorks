// Syntax + portability gate for the repo's shell scripts.
//
// WHY: the repo's shell surface had no CI coverage at all — `check:docker:runpod:supervisor`
// runs one script's self-test, and nothing parsed the rest. scripts/setup-nax-runner.sh is
// several hundred lines of provisioning bash that must run under the bash macOS actually
// ships, and three of its defects were found by executing it rather than reading it.
//
// TWO CHECKS, because one alone is not enough:
//
//  1. `bash -n`, run under BOTH the ambient bash and — when it exists and is old enough —
//     /bin/bash. That second pass is the point: macOS ships bash 3.2.57 at /bin/bash and
//     has since 2007, but a developer with Homebrew bash earlier on PATH gets 5.x from
//     plain `bash`, and CI's ubuntu is 5.x too. So without naming the interpreter
//     explicitly, the 3.2 parse never actually happens anywhere and the "portable" claim
//     rests entirely on the regexes below.
//
//  2. A scan for bash-4-only constructs. This is the part no `bash -n` can do in CI: on
//     ubuntu a bash-4 idiom parses cleanly and then explodes on a Mac.
//
// SCRIPTS ARE CLASSIFIED BY WHERE THEY RUN, because a single rule cannot be right for
// both. `docker/runpod-entrypoint.sh` legitimately uses `[[ -v VAR ]]` (bash 4.2): it only
// ever executes inside a Linux container whose bash is 5.x. Holding it to 3.2 would be a
// false positive, while holding scripts/*.sh to 5.x would miss the real thing. Worse, a
// naive `bash -n` is PLATFORM-DEPENDENT — run on a Mac it rejects that container script,
// run on CI's ubuntu it accepts a bash-4 idiom in a macOS-only script. Both directions are
// wrong, so the classification below is what makes the gate mean the same thing anywhere.
//
// DISCOVERY IS EXTENSION-BASED: `git ls-files '*.sh'`. A shell script without a `.sh`
// extension (or an inline `run:` block in a workflow) is out of scope and unchecked. That
// is a deliberate boundary, not an oversight — stated here so nobody assumes the gate is
// exhaustive over shell in this repo.
//
// KNOWN GAP, stated rather than papered over: the specific bug that motivated this gate —
// expanding an empty array as "${a[@]}" under `set -u`, which is an "unbound variable"
// error on bash 3.2 but fine on bash 4.4+ — is not statically detectable without real
// dataflow analysis, since whether the array is ever empty depends on runtime branches.
// The idiom scan cannot see it, and `bash -n` cannot either (it is valid syntax). Running a
// script's own `--check`/`--self-test` under a real 3.2 is what catches that class.
//
// Usage:
//   node scripts/check-shell-scripts.mjs
//   node scripts/check-shell-scripts.mjs --self-test

import { execFileSync, spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import path from "node:path";

const ROOT = path.resolve(import.meta.dirname, "..");

// Strip a trailing/whole-line shell comment so the rules below only ever see CODE.
//
// Every rule used to carry its own `^[^#]*` prefix guard, and four of seven were missing
// it — so a comment *warning against* an idiom flagged as a use of it. That is a live
// hazard in this repo specifically, whose scripts carry long explanatory comments; a note
// reading "use ${name+x}, not [[ -v name ]]" would have reddened CI. Doing it once, here,
// means a new rule cannot forget the guard.
//
// `#` only opens a comment at line start or after whitespace. That distinction matters:
// splitting on a bare `#` would truncate `${path#/prefix}` mid-expansion.
//
// Tradeoff, deliberate: an idiom appearing AFTER a `#` on a code line is not seen, so a
// contrived `printf '%s' "$x" # ${v^^}` is missed. Preferring false negatives to false
// positives is the right way round for a gate that can block a merge.
function codePortion(line) {
  const match = /(?:^|\s)#/.exec(line);
  return match ? line.slice(0, match.index) : line;
}

// Constructs that are bash 4+ only and therefore fail on macOS's /bin/bash 3.2. These match
// against codePortion(line), never the raw line.
const BASH4_ONLY = [
  {
    // `declare -A` / `local -A` — associative arrays landed in bash 4.0.
    pattern: /\b(?:declare|local|typeset)\s+-[A-Za-z]*A[A-Za-z]*\b/,
    what: "associative array (declare/local -A)",
    since: "bash 4.0",
  },
  {
    // `mapfile` / `readarray` — bash 4.0.
    pattern: /\b(?:mapfile|readarray)\b/,
    what: "mapfile/readarray",
    since: "bash 4.0",
  },
  {
    // ${v^} ${v^^} ${v,} ${v,,} case modification — bash 4.0.
    pattern: /\$\{[A-Za-z_][A-Za-z0-9_]*(?:\[[^\]]*\])?(?:\^\^?|,,?)\}/,
    // Escaped template literal, not a plain string. CodeQL's "template syntax in string
    // literal" rule fires on any ordinary string containing `${...}` — normally a forgotten
    // backtick, here deliberate SHELL syntax. A template literal with `\${` produces the
    // byte-identical string and is not what the rule looks for. Applies to every fixture
    // below too; please don't "simplify" them back to quotes.
    what: `case-modification expansion (\${v^^} / \${v,,})`,
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
    // `shopt -s globstar` (**) — bash 4.0.
    pattern: /\bshopt\s+-s\s+globstar\b/,
    what: "globstar",
    since: "bash 4.0",
  },
  {
    // `[[ -v name ]]` — bash 4.2. On 3.2 this is not merely unsupported, it is a SYNTAX
    // error ("conditional binary operator expected"), so it kills the whole script rather
    // than one branch. The 3.2-safe form uses the ${name+x} set-test.
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

function bashMajorOf(binary) {
  const result = spawnSync(binary, ["-c", "echo $BASH_VERSINFO"], { encoding: "utf8" });
  if (result.error || result.status !== 0) return null;
  const major = Number.parseInt((result.stdout || "").trim(), 10);
  return Number.isFinite(major) ? major : null;
}

// The interpreter that can actually reject bash-4 syntax, if this host has one. Returns null
// on Linux CI (where /bin/bash is 5.x), in which case the 3.2 parse simply does not run and
// the summary says so rather than implying coverage it does not have.
function legacyBash() {
  const major = bashMajorOf("/bin/bash");
  return major !== null && major < 4 ? { binary: "/bin/bash", major } : null;
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
  for (const [index, line] of body.split("\n").entries()) {
    const code = codePortion(line);
    for (const rule of BASH4_ONLY) {
      if (rule.pattern.test(code)) {
        problems.push(
          `${relativePath}:${index + 1} uses ${rule.what} (${rule.since}), which fails on ` +
            `macOS's /bin/bash 3.2: ${line.trim()}`,
        );
      }
    }
  }
  return problems;
}

function parseCheck(relativePath, binary, label) {
  const result = spawnSync(binary, ["-n", path.join(ROOT, relativePath)], { encoding: "utf8" });
  if (result.error) {
    throw new Error(`could not run \`${binary} -n\` for ${relativePath}: ${result.error.message}`);
  }
  if (result.status !== 0) {
    return [`${relativePath} failed \`${binary} -n\` (${label}):\n${(result.stderr || "").trim()}`];
  }
  return [];
}

function selfTest() {
  // Prove the idiom scan fires, so a green gate is not a gate that scans nothing.
  const positive = [
    ["declare -A map", "associative array"],
    ["  local -A thing", "associative array"],
    ["mapfile -t lines < file", "mapfile"],
    [`echo "\${name^^}"`, "case-modification"],
    ["cmd &>> log.txt", "&>>"],
    ["  ;;&", ";;& case fallthrough"],
    ["shopt -s globstar", "globstar"],
    ["if [[ -v name ]]; then", "[[ -v ]]"],
  ];
  for (const [line, expected] of positive) {
    if (scanForBash4("self-test.sh", line).length === 0) {
      throw new Error(`self-test: expected to flag ${expected} in: ${line}`);
    }
  }

  // EVERY idiom must be comment-immune, not just the two that happened to carry a prefix
  // guard. The previous self-test asserted comment-immunity using `# declare -A map` alone
  // — one of the three rules that already had the guard — and therefore reported
  // confidence it had not earned while four rules false-positived on comments.
  const negative = [
    "# declare -A map",
    "# mapfile is bash 4.0",
    `# use \${v^^} instead of tr on bash 4`,
    "# never write cmd &>> log.txt here",
    "  # ;;& is a bash 4 fallthrough",
    "# shopt -s globstar is bash 4",
    `# [[ -v name ]] is bash 4.2; use \${name+x} instead`,
    'cmd run   # and never [[ -v x ]] here either',
    // 3.2-safe code that must not trip anything.
    "declare -a list",
    'local -r frozen="x"',
    `printf "%s" "\${name}"`,
    "cmd >log.txt 2>&1",
    "cmd &> log.txt",
    `value="\${a[@]+"\${a[@]}"}"`,
    `echo "\${path%/*}"`,
    // `#` inside a parameter expansion is NOT a comment — must still be scanned, and must
    // not be truncated in a way that hides a later idiom on the same line.
    `echo "\${path#/prefix}"`,
  ];
  for (const line of negative) {
    const found = scanForBash4("self-test.sh", line);
    if (found.length > 0) {
      throw new Error(`self-test: false positive on 3.2-safe line: ${line}\n  ${found[0]}`);
    }
  }

  // And prove the comment stripper does not swallow a real idiom that precedes a comment.
  if (scanForBash4("self-test.sh", `x="\${v^^}"   # normalize`).length === 0) {
    throw new Error("self-test: comment stripping hid a real idiom on a code line");
  }

  console.log(
    `SceneWorks shell-script gate self-test passed ` +
      `(${positive.length} positive, ${negative.length} negative, 1 mixed).`,
  );
}

if (process.argv.includes("--self-test")) {
  selfTest();
  process.exit(0);
}

const scripts = trackedShellScripts();
if (scripts.length === 0) {
  throw new Error("no tracked *.sh files found — the discovery glob is probably wrong");
}

// A LINUX_ONLY entry that matches nothing is a silent lie: it exempts no file while implying
// it does, and it skews the summary count. Usually it means the file was renamed or deleted.
const staleExemptions = [...LINUX_ONLY].filter((entry) => !scripts.includes(entry));
if (staleExemptions.length > 0) {
  console.error(
    `SceneWorks shell-script check FAILED: LINUX_ONLY lists ${staleExemptions.length} path(s) ` +
      `that match no tracked file — remove or correct them:\n`,
  );
  for (const entry of staleExemptions) console.error(`  - ${entry}`);
  process.exit(1);
}

const ambient = { binary: "bash", major: bashMajorOf("bash") };
const legacy = legacyBash();
const problems = [];
let portableCount = 0;

for (const relativePath of scripts) {
  const linuxOnly = LINUX_ONLY.has(relativePath);
  if (!linuxOnly) portableCount += 1;

  // The ambient bash parse. For a container-only script allowed to use bash-4 syntax, a 3.x
  // ambient bash would report a spurious syntax error — only trust it when it can represent
  // the syntax.
  if (linuxOnly && (ambient.major ?? 0) < 4) {
    // Skipped; reported in the summary.
  } else {
    problems.push(...parseCheck(relativePath, ambient.binary, `bash ${ambient.major ?? "?"}.x`));
  }

  // The parse that makes the portability claim real. Only portable scripts, and only when a
  // genuine 3.x interpreter exists on this host.
  if (!linuxOnly && legacy) {
    problems.push(...parseCheck(relativePath, legacy.binary, `legacy bash ${legacy.major}.x`));
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

console.log(
  legacy
    ? `  legacy parse: ${legacy.binary} (bash ${legacy.major}.x) — bash-3.2 syntax verified directly`
    : `  legacy parse: SKIPPED (no bash 3.x at /bin/bash; ambient is ${ambient.major ?? "?"}.x) — ` +
      `portability rests on the ${BASH4_ONLY.length} idiom rules alone`,
);
console.log(
  `SceneWorks shell-script check passed (${scripts.length} scripts; ` +
    `${portableCount} held to bash-3.2 portability, ${scripts.length - portableCount} container-only).`,
);
