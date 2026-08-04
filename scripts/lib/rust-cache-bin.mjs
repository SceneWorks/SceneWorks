/**
 * `Swatinem/rust-cache` on a SELF-HOSTED runner must set `cache-bin: "false"`.
 *
 * WHY (inference PR #467, 2026-08-04) — this is the canonical writeup; the notes in
 * `.github/actions/prepare-rust-runner/action.yml` and `scripts/win/use-real-rust-toolchain.cmd`
 * point here. It is deliberately NOT filed under a Windows-only path: the failure was diagnosed on
 * an Apple Silicon Mac and the mechanism is cross-platform.
 *
 * The action's POST step runs `cleanBin()`:
 *
 *     if (dirent.isFile() && !bins.has(dirent.name)) { await rm(dir.path, dirent); }
 *
 * `bins` is built from `$CARGO_HOME/.crates2.json`, i.e. ONLY binaries installed via
 * `cargo install`. Everything else in `$CARGO_HOME/bin` that is a regular file is deleted —
 * including `rustup` itself, which is not a `cargo install` product.
 *
 * On an ephemeral GitHub-hosted runner that is harmless: the whole VM is discarded. On a
 * SELF-HOSTED runner `CARGO_HOME` is a real user profile, so the post step deletes the developer's
 * `rustup`, and every proxy in `$CARGO_HOME/bin` (`cargo`, `rustc`, `rustfmt`, `clippy-driver`, …)
 * dispatches through that one binary. The result is `cargo: command not found`, or the more
 * confusing `no such file or directory: .../.cargo/bin/cargo` while `ls` still SHOWS `cargo` —
 * the classic dangling-symlink signature.
 *
 * Exactly which entries survive depends on how rustup materialised its proxies, and is
 * privilege- and version-dependent — do not over-specify it. rustup prefers symlinks (skipped by
 * `isFile()`, so they survive as dangling links) but falls back to hardlinks where symlinks are
 * unavailable (e.g. unprivileged Windows), and a hardlink IS a regular file, so `cleanBin()` would
 * remove those entries outright rather than leave them dangling. Both shapes present as "the Rust
 * toolchain vanished"; only the forensics differ.
 *
 * Why it stays hidden: `cleanBin()` is wrapped in a try/catch that logs at debug level only and the
 * job still passes, and the next run's `dtolnay/rust-toolchain` step silently reinstalls rustup. So
 * nothing ever goes red — it presents as unreproducible flakiness in OTHER processes on the box,
 * including any concurrent local build or agent session.
 *
 * This check exists because the alternative — a prose note saying "audited, all our uses are
 * ephemeral" — is a point-in-time declaration, true until someone adds one `uses:` line to a
 * self-hosted job. An invariant survives that; a comment does not.
 */

const JOBS_HEADER = /^jobs:\s*$/;
const JOB_HEADER = /^ {2}([A-Za-z0-9_.-]+):\s*$/;
const RUST_CACHE_USES = /^\s*(?:-\s*)?uses:\s*Swatinem\/rust-cache@/;
const RUNS_ON_INLINE = /^\s{4}runs-on:\s*(.*)$/;
const BLOCK_SEQ_ITEM = /^\s+-\s*(.+?)\s*$/;
const CACHE_BIN = /^\s*cache-bin:\s*(.+?)\s*$/;

/** Strip a trailing `# comment` and surrounding quotes from a scalar. */
function scalar(raw) {
  const withoutComment = raw.replace(/\s+#.*$/, "").trim();
  return withoutComment.replace(/^["']|["']$/g, "");
}

/**
 * Resolve a job's `runs-on` into a list of labels. Handles the three shapes the repo uses:
 * `runs-on: ubuntu-latest`, `runs-on: [self-hosted, macOS, ARM64]`, and a block sequence.
 * Returns `null` when the job declares no `runs-on` at all.
 */
function runsOnLabels(lines) {
  for (let i = 0; i < lines.length; i += 1) {
    const match = RUNS_ON_INLINE.exec(lines[i]);
    if (!match) continue;
    const inline = scalar(match[1]);
    if (inline.startsWith("[")) {
      return inline
        .replace(/^\[|\]$/g, "")
        .split(",")
        .map((label) => scalar(label))
        .filter(Boolean);
    }
    if (inline) return [inline];
    // Block sequence: consume the following `- label` lines.
    const labels = [];
    for (let j = i + 1; j < lines.length; j += 1) {
      if (!lines[j].trim() || lines[j].trim().startsWith("#")) continue;
      const item = BLOCK_SEQ_ITEM.exec(lines[j]);
      if (!item) break;
      labels.push(scalar(item[1]));
    }
    return labels;
  }
  return null;
}

/** The lines belonging to the step that owns `usesIndex`, up to the next step at the same indent. */
function stepBody(lines, usesIndex) {
  let start = usesIndex;
  while (start > 0 && !/^\s*-\s/.test(lines[start])) start -= 1;
  const indent = lines[start].search(/\S/);
  const body = [lines[start]];
  for (let i = start + 1; i < lines.length; i += 1) {
    const line = lines[i];
    if (!line.trim()) continue;
    const lineIndent = line.search(/\S/);
    if (lineIndent <= indent && /^\s*-\s/.test(line)) break;
    if (lineIndent < indent) break;
    body.push(line);
  }
  return body;
}

/**
 * Find every `Swatinem/rust-cache` usage in a self-hosted job that does not disable bin caching.
 * Returns `[{ file, job, line, reason }]`.
 */
export function findRustCacheBinViolations(workflow, file = "") {
  const lines = workflow.split(/\r?\n/);
  const jobsAt = lines.findIndex((line) => JOBS_HEADER.test(line));
  if (jobsAt === -1) return [];

  const headers = [];
  for (let i = jobsAt + 1; i < lines.length; i += 1) {
    if (/^\S/.test(lines[i])) break; // dedent out of `jobs:`
    const match = JOB_HEADER.exec(lines[i]);
    if (match) headers.push({ name: match[1], index: i });
  }

  const violations = [];
  headers.forEach((header, position) => {
    const end = position + 1 < headers.length ? headers[position + 1].index : lines.length;
    const block = lines.slice(header.index, end);
    const cacheSteps = block
      .map((line, offset) => (RUST_CACHE_USES.test(line) ? offset : -1))
      .filter((offset) => offset !== -1);
    if (cacheSteps.length === 0) return;

    const labels = runsOnLabels(block);
    if (labels === null) {
      violations.push({
        file,
        job: header.name,
        line: header.index + 1,
        reason: `job uses Swatinem/rust-cache but declares no 'runs-on', so it cannot be shown to be ephemeral`,
      });
      return;
    }
    if (!labels.some((label) => label.toLowerCase() === "self-hosted")) return;

    for (const offset of cacheSteps) {
      const body = stepBody(block, offset);
      const setting = body.map((line) => CACHE_BIN.exec(line)).find(Boolean);
      const value = setting ? scalar(setting[1]).toLowerCase() : null;
      if (value === "false") continue;
      violations.push({
        file,
        job: header.name,
        line: header.index + offset + 1,
        reason:
          value === null
            ? `self-hosted job uses Swatinem/rust-cache without 'cache-bin: "false"' (it defaults to true)`
            : `self-hosted job sets 'cache-bin: "${value}"'; it must be "false"`,
      });
    }
  });
  return violations;
}

export function formatRustCacheBinViolations(violations) {
  return violations
    .map((violation) => `  ${violation.file}:${violation.line} [${violation.job}] ${violation.reason}`)
    .join("\n");
}
