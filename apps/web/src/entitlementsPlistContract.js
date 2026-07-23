// Single source of truth for the entitlements-plist well-formedness rule that macOS
// codesign / AMFI enforces but plutil -lint and the JSON-ish linters do NOT.
//
// The break (sc-13609): apps/desktop/Entitlements.plist carried a comment containing
// `codesign --force --options runtime`. An XML comment body may not contain `--`
// (double-hyphen) — XML 1.0 production [15] allows `--` only as part of the closing
// `-->`. codesign's AMFI parser (AMFIUnserializeXML) is strict and rejected it at real
// signing time ("Failed to parse entitlements: syntax error near line 18"), yet
// `plutil -lint` is lenient and passed it, and NO CI lane codesigns — so it merged green
// and only broke when the release lane actually signed the app.
//
// This module is the PR-time detector for that class of defect. It is pure and
// dependency-free so both scripts/check-scaffold.mjs (the scaffold/parity CI gate) and the
// co-located vitest test can consume the SAME rule and it can never silently drift.

// Compute the 1-based line number of a character offset in `text`.
function lineAtOffset(text, offset) {
  let line = 1;
  for (let index = 0; index < offset && index < text.length; index += 1) {
    if (text[index] === "\n") {
      line += 1;
    }
  }
  return line;
}

// The single source line containing `offset`, trimmed — the actionable "offending text".
function lineTextAtOffset(text, offset) {
  const start = text.lastIndexOf("\n", offset - 1) + 1;
  const newline = text.indexOf("\n", offset);
  const end = newline === -1 ? text.length : newline;
  return text.slice(start, end).trim();
}

// Scan XML/plist `text` and return every comment-level defect that codesign/AMFI would
// reject. Returns an array of findings; an empty array means the comments are well-formed
// for AMFI's purposes.
//
// Faithful to the XML rule rather than a substring heuristic: inside a comment, `--` is
// legal ONLY as the leading two characters of the closing `-->`. Any other `--` (e.g.
// `--force`, or a `--->` close) is illegal, and a `<!--` with no closing `-->` is
// unterminated. Because a literal `<!--` can only be a comment start in XML (an unescaped
// `<` is illegal in element text/attributes), treating every `<!--` as a comment is correct.
//
// Each finding: { line, column, kind, snippet }
//   kind: "double-hyphen-in-comment" | "unterminated-comment"
export function findXmlCommentDefects(text) {
  const findings = [];
  const OPEN = "<!--";
  let cursor = 0;
  while (cursor < text.length) {
    const open = text.indexOf(OPEN, cursor);
    if (open === -1) {
      break;
    }
    const bodyStart = open + OPEN.length;
    // Walk the body looking for the next `--`. The only legal `--` is the one that
    // begins the closing `-->`; anything else is a defect.
    let scan = bodyStart;
    let closed = false;
    while (scan < text.length) {
      const dash = text.indexOf("--", scan);
      if (dash === -1) {
        break; // no `--` at all before EOF => unterminated
      }
      if (text[dash + 2] === ">") {
        // Valid closing `-->`; resume scanning after this comment.
        cursor = dash + 3;
        closed = true;
        break;
      }
      // An illegal `--` inside the comment body (the sc-13609 class).
      findings.push({
        line: lineAtOffset(text, dash),
        column: dash - text.lastIndexOf("\n", dash),
        kind: "double-hyphen-in-comment",
        snippet: lineTextAtOffset(text, dash),
      });
      // Resume after this `--` so a single comment can report at most sensibly and we
      // still find the real terminator (or flag it unterminated).
      scan = dash + 2;
    }
    if (!closed) {
      findings.push({
        line: lineAtOffset(text, open),
        column: open - text.lastIndexOf("\n", open),
        kind: "unterminated-comment",
        snippet: lineTextAtOffset(text, open),
      });
      break; // nothing well-formed can follow an unterminated comment
    }
  }
  return findings;
}

// Human-readable, actionable one-liner for a finding (used in the thrown CI message).
export function describeXmlCommentDefect(finding) {
  if (finding.kind === "unterminated-comment") {
    return `line ${finding.line}: unterminated XML comment (no closing \`-->\`) — ${JSON.stringify(finding.snippet)}`;
  }
  return `line ${finding.line}: \`--\` (double-hyphen) inside an XML comment — ${JSON.stringify(finding.snippet)}`;
}
