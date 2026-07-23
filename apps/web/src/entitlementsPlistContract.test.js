import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { describeXmlCommentDefect, findXmlCommentDefects } from "./entitlementsPlistContract.js";

// Guard for the sc-13609 class: an XML comment body containing `--` (double-hyphen) is
// illegal XML. codesign's AMFI parser (AMFIUnserializeXML) rejects it at signing time, but
// `plutil -lint` is lenient and no CI lane codesigns — so it merged green and only broke the
// release/notarize lane. scripts/check-scaffold.mjs (the scaffold/parity gate) calls
// findXmlCommentDefects on apps/desktop/Entitlements.plist; this test locks the detector's
// behaviour AND reads the REAL committed plist so a regression that reintroduces `--` reds.

const HERE = dirname(fileURLToPath(import.meta.url));
const ENTITLEMENTS_PATH = resolve(HERE, "../../../apps/desktop/Entitlements.plist");

describe("findXmlCommentDefects", () => {
  it("passes clean XML with a well-formed comment", () => {
    const xml = '<?xml version="1.0"?>\n<!-- hardened runtime notes -->\n<plist><dict/></plist>\n';
    expect(findXmlCommentDefects(xml)).toEqual([]);
  });

  it("passes an empty comment (`<!---->`)", () => {
    expect(findXmlCommentDefects("<!---->")).toEqual([]);
  });

  it("flags a `--` inside a comment body — the exact sc-13609 defect", () => {
    const xml =
      '<?xml version="1.0"?>\n' +
      "<!--\n  re-signs it (`codesign --force --options runtime`) with the SAME\n-->\n" +
      "<plist><dict/></plist>\n";
    const defects = findXmlCommentDefects(xml);
    expect(defects.length).toBeGreaterThan(0);
    expect(defects[0].kind).toBe("double-hyphen-in-comment");
    // Points at the real offending source line, not the comment open.
    expect(defects[0].line).toBe(3);
    expect(defects[0].snippet).toContain("codesign --force --options runtime");
    expect(describeXmlCommentDefect(defects[0])).toContain("double-hyphen");
  });

  it("does NOT flag `--` that appears in element text (comment-scoped only)", () => {
    // A double-hyphen in a <string> value is legal XML; the rule is comment-body-only.
    const xml = '<?xml version="1.0"?>\n<plist><dict><string>a--b</string></dict></plist>\n';
    expect(findXmlCommentDefects(xml)).toEqual([]);
  });

  it("does NOT mistake the closing `-->` for an illegal `--`", () => {
    expect(findXmlCommentDefects("<!-- ok -->")).toEqual([]);
  });

  it("flags an unterminated comment (matched-delimiter sanity)", () => {
    const defects = findXmlCommentDefects("<!-- never closed");
    expect(defects).toHaveLength(1);
    expect(defects[0].kind).toBe("unterminated-comment");
  });

  it("flags a `--->` close (a `--` not immediately closing the comment)", () => {
    const defects = findXmlCommentDefects("<!-- trailing dash --->");
    expect(defects.length).toBeGreaterThan(0);
    expect(defects[0].kind).toBe("double-hyphen-in-comment");
  });
});

describe("the shipped apps/desktop/Entitlements.plist", () => {
  it("is codesign/AMFI-safe (no `--` in any comment body)", () => {
    const body = readFileSync(ENTITLEMENTS_PATH, "utf8");
    expect(findXmlCommentDefects(body)).toEqual([]);
  });
});
