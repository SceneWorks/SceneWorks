import { afterEach, describe, expect, it } from "vitest";

import {
  EMBEDDED_PROSE_FIELDS,
  WORKFLOW_SHARE_DOC_URL,
  proseFieldSentence,
  readEmbedWorkflowInImages,
  readWorkflowEmbedNoticeSeen,
  writeEmbedWorkflowInImages,
  writeWorkflowEmbedNoticeSeen,
} from "./workflowEmbed.js";

afterEach(() => {
  try {
    globalThis.localStorage?.clear();
  } catch {
    // no-op
  }
});

describe("the embed preference cache (sc-15953)", () => {
  it("defaults ON when nothing is stored, matching the worker's own reader", () => {
    // `embed_workflow_in_images_from_json` resolves an absent key to true. If this cache
    // disagreed, the toggle would render OFF on a fresh install while every image embedded.
    expect(readEmbedWorkflowInImages()).toBe(true);
  });

  it("round-trips both values", () => {
    expect(writeEmbedWorkflowInImages(false)).toBe(false);
    expect(readEmbedWorkflowInImages()).toBe(false);
    expect(writeEmbedWorkflowInImages(true)).toBe(true);
    expect(readEmbedWorkflowInImages()).toBe(true);
  });

  it("ignores a value it did not write rather than reading it as OFF", () => {
    globalThis.localStorage.setItem("sceneworks-embed-workflow", "maybe");
    expect(readEmbedWorkflowInImages()).toBe(true);
  });
});

describe("the first-run disclosure flag (sc-15953)", () => {
  it("defaults to NOT SEEN, so an upgrading install is told once", () => {
    // An install that has been generating for months has never been shown this. Defaulting to
    // "seen" would mean the silently-on default is only ever disclosed to fresh installs.
    expect(readWorkflowEmbedNoticeSeen()).toBe(false);
  });

  it("round-trips", () => {
    writeWorkflowEmbedNoticeSeen(true);
    expect(readWorkflowEmbedNoticeSeen()).toBe(true);
  });
});

describe("the copy's field list (sc-15953)", () => {
  it("lists the six path-exempt prose fields, keyed by their envelope path", () => {
    // The set itself is pinned against docs/workflow-share-envelope.md's `prose-fields` table by
    // `the_settings_copy_names_exactly_the_path_exempt_prose_fields` in
    // crates/sceneworks-core/tests/workflow_share_doc.rs. What is asserted HERE is only the shape
    // the UI relies on: pairs of (envelope path, human label), all non-empty.
    expect(EMBEDDED_PROSE_FIELDS.length).toBe(6);
    for (const entry of EMBEDDED_PROSE_FIELDS) {
      expect(entry).toHaveLength(2);
      expect(typeof entry[0]).toBe("string");
      expect(entry[0].length).toBeGreaterThan(0);
      expect(entry[1].length).toBeGreaterThan(0);
    }
    expect(EMBEDDED_PROSE_FIELDS.map(([key]) => key)).toContain("prompt");
  });

  it("renders every field into the sentence the settings copy shows", () => {
    const sentence = proseFieldSentence();
    for (const [, label] of EMBEDDED_PROSE_FIELDS) {
      expect(sentence).toContain(label);
    }
    // Read as a list rather than as a run-on: the last item is joined with "and".
    expect(sentence).toContain(` and ${EMBEDDED_PROSE_FIELDS.at(-1)[1]}`);
  });

  it("links to the contract document rather than to a marketing page", () => {
    expect(WORKFLOW_SHARE_DOC_URL).toContain("docs/workflow-share-envelope.md");
    expect(WORKFLOW_SHARE_DOC_URL.startsWith("https://")).toBe(true);
  });
});
