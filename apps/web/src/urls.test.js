import { describe, expect, it } from "vitest";
import { safeExternalUrl } from "./urls.js";

// sc-8881 [F-079]: manifest-supplied URL fields (licenseUrl, homepage,
// prompt-guide sources) are rendered into `<a href>`; `safeExternalUrl` must
// admit only absolute http(s) links and reject anything that could execute on
// click (javascript:, data:, vbscript:) or is otherwise unusable.
describe("safeExternalUrl", () => {
  it("accepts absolute http and https URLs unchanged", () => {
    expect(safeExternalUrl("https://huggingface.co/org/model")).toBe(
      "https://huggingface.co/org/model",
    );
    expect(safeExternalUrl("http://example.com/path?q=1#frag")).toBe(
      "http://example.com/path?q=1#frag",
    );
  });

  it("trims surrounding whitespace on valid URLs", () => {
    expect(safeExternalUrl("  https://example.com/x  ")).toBe("https://example.com/x");
  });

  it("is case-insensitive on the scheme", () => {
    expect(safeExternalUrl("HTTPS://example.com")).toBe("HTTPS://example.com");
  });

  it("rejects javascript: URLs (click-time script execution vector)", () => {
    expect(safeExternalUrl("javascript:alert(1)")).toBeUndefined();
    expect(safeExternalUrl("JavaScript:alert(1)")).toBeUndefined();
    // Leading whitespace must not smuggle the scheme past the check.
    expect(safeExternalUrl("  javascript:alert(1)")).toBeUndefined();
  });

  it("rejects data: URLs", () => {
    expect(safeExternalUrl("data:text/html,<script>alert(1)</script>")).toBeUndefined();
  });

  it("rejects vbscript: URLs", () => {
    expect(safeExternalUrl("vbscript:msgbox(1)")).toBeUndefined();
  });

  it("rejects other non-http(s) schemes", () => {
    expect(safeExternalUrl("mailto:someone@example.com")).toBeUndefined();
    expect(safeExternalUrl("file:///etc/passwd")).toBeUndefined();
    expect(safeExternalUrl("ftp://example.com/file")).toBeUndefined();
  });

  it("rejects relative and scheme-less URLs (manifest links must be absolute)", () => {
    expect(safeExternalUrl("/relative/path")).toBeUndefined();
    expect(safeExternalUrl("example.com/no-scheme")).toBeUndefined();
    expect(safeExternalUrl("#anchor")).toBeUndefined();
  });

  it("rejects malformed, empty, and non-string inputs", () => {
    expect(safeExternalUrl("")).toBeUndefined();
    expect(safeExternalUrl("   ")).toBeUndefined();
    expect(safeExternalUrl("http://")).toBeUndefined();
    expect(safeExternalUrl("ht!tp://bad")).toBeUndefined();
    expect(safeExternalUrl(null)).toBeUndefined();
    expect(safeExternalUrl(undefined)).toBeUndefined();
    expect(safeExternalUrl(42)).toBeUndefined();
    expect(safeExternalUrl({})).toBeUndefined();
  });
});
