// Shape rule for GitHub Actions `uses:` references.
//
// The supply-chain control is that every third-party action resolves to an
// immutable commit: a 40-character SHA, never a tag or branch, carrying a
// trailing comment that records the human-readable version. Dependabot rewrites
// both halves in place, so the rule stays true across automated bumps instead of
// pinning a review-time SHA that every bump then has to be re-approved against.

const USES_LINE = /^\s*(?:-\s+)?uses:\s*(.+?)\s*$/;
const PINNED = /^[A-Za-z0-9._-]+\/[A-Za-z0-9._/-]+@[0-9a-f]{40}\s+#\s*\S.*$/;

function describe(reference) {
  const separator = reference.indexOf("@");
  if (separator === -1) {
    return "is not pinned to a commit SHA";
  }
  const ref = reference.slice(separator + 1).split(/\s+#\s*/, 1)[0];
  if (!/^[0-9a-f]{40}$/.test(ref)) {
    return `is pinned to the mutable ref '${ref}' instead of a 40-character commit SHA`;
  }
  return "is missing the trailing '# version' comment Dependabot maintains";
}

/**
 * Collect every `uses:` reference in a workflow that is not pinned to an
 * immutable commit. Local composite actions (`./...`) are exempt: they live in
 * this repository and are versioned by the commit that runs them.
 *
 * @param {string} source workflow YAML
 * @param {string} path path reported in violations
 * @returns {{path: string, line: number, reference: string, reason: string}[]}
 */
export function findActionPinViolations(source, path = "<workflow>") {
  const violations = [];
  source.split(/\r?\n/).forEach((line, index) => {
    const match = USES_LINE.exec(line);
    if (!match) {
      return;
    }
    const reference = match[1];
    if (reference.startsWith("./")) {
      return;
    }
    if (PINNED.test(reference)) {
      return;
    }
    violations.push({
      path,
      line: index + 1,
      reference,
      reason: describe(reference),
    });
  });
  return violations;
}

export function formatViolations(violations) {
  return violations
    .map((v) => `  ${v.path}:${v.line}  ${v.reference}\n      ${v.reason}`)
    .join("\n");
}
