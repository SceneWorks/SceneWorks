import React from "react";
import { Icon } from "./Icons.jsx";
import {
  WORKFLOW_SHARE_DOC_URL,
  proseFieldSentence,
} from "../workflowEmbed.js";

// The user-facing copy for the embedded-workflow feature (sc-15953, epic 15945): the block of
// detail under the Settings toggle, and the one-time first-run disclosure.
//
// Both live here so there is ONE set of claims about what leaves with an image. Every sentence
// below is checked against `docs/workflow-share-envelope.md`, which is itself pinned in both
// directions against the shipped sanitizer by `crates/sceneworks-core/tests/workflow_share_doc.rs`.
// The list of path-exempt prose fields is not written out as prose at all — it is rendered from
// `EMBEDDED_PROSE_FIELDS`, which that same Rust test pins against the doc's `prose-fields` table,
// so a seventh prose field cannot be added to the sanitizer without this copy failing a test.
//
// The standing failure mode this epic keeps hitting is an artifact claiming a stronger guarantee
// than the code delivers. So the wording is deliberately narrow: it says what IS in the file rather
// than promising the file is safe, it names the two-segment-path hole rather than claiming every
// path is caught, and it says the switch applies to the NEXT generation rather than implying
// anything about images already on disk.

// The paragraphs under the Settings toggle. Also rendered inside the first-run notice, so the two
// surfaces cannot drift.
export function WorkflowEmbedDetails() {
  return (
    <>
      <p className="settings-note">
        <strong>Your prompt travels as you wrote it.</strong> {proseFieldSentence()} are recorded
        exactly as authored, so a file path or a client&apos;s name typed into any of them leaves
        with every copy of the image. Every other value is dropped if it looks like a file path —
        except a two-segment name like <code>Clients/Acme</code>, which is the shape of a Hugging
        Face repo id and is not treated as a location.
      </p>
      <p className="settings-note">
        <strong>Also in the file:</strong> the generation mode, the model and style names, the seed
        for this image, its size, how it was fitted, how many images the run asked for, any upscale
        settings, LoRA names, weights and Hugging Face repo ids, the shared generation settings
        (sampler, steps, guidance and the like), which kinds of input image the recipe needs, and
        the SceneWorks version that wrote the file.
      </p>
      <p className="settings-note">
        <strong>Not in the file:</strong> your user or machine name, project, job and asset ids, any
        timestamp, the other seeds in a batch, the input images themselves, and this
        machine&apos;s quality tier, attention kernel or GPU.
      </p>
      <p className="settings-note">
        Turning this off applies from the next generation. Images already on disk keep the block
        they were written with — to share one of those without it, use{" "}
        <strong>Save without workflow</strong>.
      </p>
      <p className="settings-note">
        <a href={WORKFLOW_SHARE_DOC_URL} rel="noreferrer noopener" target="_blank">
          What travels, exactly
        </a>
      </p>
    </>
  );
}

// The one-time disclosure. Rendered the first time a generation is submitted while embedding is on
// and the user has never been told; dismissing it records a durable server-side flag, so it does
// not come back on the next launch (the desktop shell's per-launch origin makes a localStorage-only
// flag useless for this).
//
// Deliberately NOT a modal. It is a disclosure, not a decision — blocking a generation the user
// already asked for to make them read a paragraph would train them to dismiss it unread, which is
// the opposite of the point.
export function WorkflowEmbedNotice({ onDismiss, onOpenSettings }) {
  return (
    <section
      aria-labelledby="workflow-embed-notice-title"
      className="settings-notice workflow-embed-notice"
      role="region"
    >
      <Icon.Warning size={16} />
      <div>
        <div className="workflow-embed-notice-title" id="workflow-embed-notice-title">
          Your generated images carry their recipe
        </div>
        <WorkflowEmbedDetails />
        <div className="settings-button-row">
          {onOpenSettings ? (
            <button className="settings-btn" onClick={onOpenSettings} type="button">
              Open settings
            </button>
          ) : null}
          <button className="settings-btn" onClick={onDismiss} type="button">
            Got it
          </button>
        </div>
      </div>
    </section>
  );
}
