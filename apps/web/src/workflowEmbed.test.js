import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const putUiPreferences = vi.fn();
vi.mock("./uiPreferences.js", () => ({ putUiPreferences: (...args) => putUiPreferences(...args) }));

const {
  EMBEDDED_PROSE_FIELDS,
  PRODUCER_URL,
  SAVE_WITHOUT_WORKFLOW_LABEL,
  WORKFLOW_FIELDS_IN_FILE,
  WORKFLOW_FIELDS_NOT_IN_FILE,
  WORKFLOW_SHARE_DOC_URL,
  inFileItems,
  notInFileItems,
  persistWorkflowEmbedPreference,
  proseFieldSentence,
  readEmbedWorkflowInImages,
  readWorkflowEmbedNoticeSeen,
  subscribeWorkflowEmbedFlags,
  writeEmbedWorkflowInImages,
  writeWorkflowEmbedNoticeSeen,
} = await import("./workflowEmbed.js");

beforeEach(() => {
  putUiPreferences.mockReset();
  putUiPreferences.mockResolvedValue(undefined);
});

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
    // DERIVED from the URL the envelope itself carries, not written out beside it. The prefix used
    // to be a hardcoded string under a comment claiming it was rooted here, with nothing checking.
    expect(WORKFLOW_SHARE_DOC_URL.startsWith(`${PRODUCER_URL}/`)).toBe(true);
  });
});

describe("the two lists a user acts on (sc-15953)", () => {
  // The SET of keys is pinned against the doc's tables in both directions by
  // `the_settings_copy_accounts_for_every_shared_and_withheld_field`. What is asserted here is what
  // the user actually reads.

  it("names the pose, hand and FACE coordinates in what travels", () => {
    // The finding: `advanced.poses` reduces to the `keypoints` / `hands` / `face` arrays —
    // coordinates derived from a reference photo, facial landmarks included — and the copy named
    // none of it, folding them under "the shared generation settings … and the like".
    const sentence = inFileItems().join(" | ");
    for (const word of ["pose", "hand", "face"]) {
      expect(sentence).toContain(word);
    }
    expect(sentence).toContain("facial landmarks");
    // And the multi-phase schedule, the other thing a user would recognise but never saw named.
    expect(sentence).toContain("multi-phase denoise schedule");
  });

  it("does not let 'not in the file' imply nothing derived from a reference travels", () => {
    // "The input images themselves" sitting in a list of absences invites exactly that reading,
    // which is the forbidden direction: claiming more privacy than the allow-list delivers.
    const sentence = notInFileItems().join(" | ");
    expect(sentence).toContain("the input images themselves");
    expect(sentence).toContain("what a pose selection traced from one does travel");
  });

  it("does not claim the file says nothing about a person replacement", () => {
    // The finding: the bullet read "anything about a person replacement — …", and `mode` travels
    // as `replace_person` in every one. The enumerated tail was true; the leading clause claimed
    // more privacy than the allow-list delivers. What is withheld is the identity and the variant.
    // Asserted against a real envelope's bytes by
    // `the_settings_copy_does_not_overclaim_about_person_replacement` in
    // `crates/sceneworks-core/tests/workflow_share_doc.rs`.
    const sentence = notInFileItems().join(" | ");
    expect(sentence).not.toContain("anything about a person replacement");
    expect(sentence).toContain("who was replaced");
    expect(sentence).toContain("the person's name");
    expect(sentence).toContain("a replacement is what ran");
  });

  it("renders every declared label into its sentence, with repeats collapsed", () => {
    const rendered = inFileItems().join(" | ");
    for (const [, label] of WORKFLOW_FIELDS_IN_FILE) {
      expect(rendered).toContain(label);
    }
    for (const [, label] of WORKFLOW_FIELDS_NOT_IN_FILE) {
      expect(notInFileItems()).toContain(label);
    }
    // `width` and `height` share one phrase; the reader must see one bullet, not two.
    expect(inFileItems().length).toBeLessThan(WORKFLOW_FIELDS_IN_FILE.length);
    expect(new Set(inFileItems()).size).toBe(inFileItems().length);
  });

  it("keys both lists by envelope path, so the Rust pin has something to match", () => {
    for (const list of [WORKFLOW_FIELDS_IN_FILE, WORKFLOW_FIELDS_NOT_IN_FILE]) {
      const keys = list.map(([key]) => key);
      expect(new Set(keys).size).toBe(keys.length);
      for (const [key, label] of list) {
        expect(typeof key).toBe("string");
        expect(key.length).toBeGreaterThan(0);
        expect(label.length).toBeGreaterThan(0);
      }
    }
    expect(WORKFLOW_FIELDS_IN_FILE.map(([key]) => key)).toContain("advanced.poses");
    expect(WORKFLOW_FIELDS_NOT_IN_FILE.map(([key]) => key)).toContain("advanced.quantTier");
  });

  it("names the without-workflow control once, for every surface to reuse", () => {
    expect(SAVE_WITHOUT_WORKFLOW_LABEL.length).toBeGreaterThan(0);
    // No trailing ellipsis in the constant: the surfaces that open a dialog add their own.
    expect(SAVE_WITHOUT_WORKFLOW_LABEL.endsWith("…")).toBe(false);
  });
});

describe("persisting a flag durably (sc-15953)", () => {
  const noSleep = () => Promise.resolve();

  it("returns on the first success without retrying", async () => {
    await expect(
      persistWorkflowEmbedPreference({ embedWorkflowInImages: false }, { sleep: noSleep }),
    ).resolves.toBe(true);
    expect(putUiPreferences).toHaveBeenCalledTimes(1);
    expect(putUiPreferences).toHaveBeenCalledWith({ embedWorkflowInImages: false });
  });

  it("retries a transient failure rather than dropping the write", async () => {
    // The localStorage mirror is wiped by the desktop shell's per-launch origin, so a dropped PUT
    // is not "saved locally" — it is a privacy opt-out that reverts to ON at the next launch.
    putUiPreferences.mockRejectedValueOnce(new Error("offline")).mockResolvedValueOnce(undefined);
    await expect(
      persistWorkflowEmbedPreference({ workflowEmbedNoticeSeen: true }, { sleep: noSleep }),
    ).resolves.toBe(true);
    expect(putUiPreferences).toHaveBeenCalledTimes(2);
  });

  it("propagates the failure once the attempts are spent, instead of swallowing it", async () => {
    putUiPreferences.mockRejectedValue(new Error("401"));
    await expect(
      persistWorkflowEmbedPreference({ workflowEmbedNoticeSeen: true }, { sleep: noSleep }),
    ).rejects.toThrow("401");
    expect(putUiPreferences).toHaveBeenCalledTimes(3);
  });
});

describe("following another tab (sc-15953)", () => {
  function fireStorage(key) {
    globalThis.dispatchEvent(new globalThis.StorageEvent("storage", { key }));
  }

  it("carries a dismissal into a tab that was already mounted", async () => {
    // Two tabs are two React trees. Before this, a second tab went on holding `seen: false` and
    // re-showed the disclosure on its own next generation — "once" being per tab, not per user.
    const seen = [];
    const unsubscribe = subscribeWorkflowEmbedFlags((state) => seen.push(state));
    writeWorkflowEmbedNoticeSeen(true);
    fireStorage("sceneworks-embed-workflow-notice-seen");
    expect(seen.at(-1).seen).toBe(true);

    // The embed preference is mirrored BOTH ways: it is a setting with two states.
    writeEmbedWorkflowInImages(false);
    fireStorage("sceneworks-embed-workflow");
    expect(seen.at(-1).embed).toBe(false);
    unsubscribe();
  });

  it("never moves the disclosure flag back to unseen", async () => {
    // A record that something was said. Another tab clearing its storage is not evidence the user
    // was never told, and letting it move back would re-arm the notice everywhere.
    writeWorkflowEmbedNoticeSeen(true);
    const seen = [];
    const unsubscribe = subscribeWorkflowEmbedFlags((state) => seen.push(state));
    globalThis.localStorage.clear();
    fireStorage(null);
    expect(seen.at(-1).seen).toBeUndefined();
    unsubscribe();
  });

  it("ignores an unrelated key so a busy tab is not re-rendered by every write", () => {
    const seen = [];
    const unsubscribe = subscribeWorkflowEmbedFlags((state) => seen.push(state));
    fireStorage("sceneworks-theme");
    expect(seen).toHaveLength(0);
    unsubscribe();
  });
});
