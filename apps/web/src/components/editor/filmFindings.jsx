import React from "react";

// A refused film-document write comes back as `PlanDiagnostic`'s `Display` output — `[plan] <field>:
// <message>`, or `[<shotId>] <field>: <message>` — and `ProjectStore::add_film_reference` joins
// several of them with `"; "` (`crates/sceneworks-core/src/film_plan.rs`,
// `crates/sceneworks-core/src/project_store.rs`).
//
// The scope and field path are the INTERNAL half of that record. This repo shows an operator the
// `error` and never the `requirement`, and every structured finding the workspace renders shows
// `finding.message` alone — so a refusal that arrives as one flat string is split back into the
// same messages rather than printed with its field path attached.
// The sound half of the reference pack, authored on the Sound step. Stated once so the panel that
// CLAIMS those findings and the panel that LEAVES them alone cannot drift into both or neither.
export const SOUND_FIELD_PREFIX = "referencePack.sound";

const PLAN_DIAGNOSTIC_PREFIX = /^\[[^\]]*\]\s*[^:\s]+:\s*/;
// Split ONLY where the next part opens a new diagnostic. A message may itself contain "; " and
// ": " — the shared-file refusal quotes role names and a path — so a blind split on "; " would cut
// one sentence in half.
const PLAN_DIAGNOSTIC_SEPARATOR = /;\s+(?=\[[^\]]*\]\s*[^:\s]+:\s*)/;

// The operator-facing messages inside a refusal detail, in order.
//
// A string that does not open with a diagnostic prefix is returned unchanged: an ordinary
// transport or route error ("Film draft not found") is already operator-facing, and stripping
// nothing is the safe reading.
export function planRefusalMessages(detail) {
  const text = typeof detail === "string" ? detail.trim() : "";
  if (!text) return [];
  if (!PLAN_DIAGNOSTIC_PREFIX.test(text)) return [text];
  return text
    .split(PLAN_DIAGNOSTIC_SEPARATOR)
    .map((part) => part.replace(PLAN_DIAGNOSTIC_PREFIX, "").trim())
    .filter(Boolean);
}

// One `ve-film-findings` list of messages, or nothing when there are none.
export function FindingList({ label, messages = [] }) {
  if (!messages.length) return null;
  return (
    <ul aria-label={label} className="ve-film-findings">
      {messages.map((message, index) => <li key={`${message}-${index}`}>{message}</li>)}
    </ul>
  );
}

// Every pack-level finding (`shotId == null`) this panel owns but has not already shown beside a
// field. Nothing the server reports may be counted in the step header and then displayed nowhere.
export function unroutedPackFindings(findings = [], renderedFields, owns = () => true) {
  return findings.filter((finding) => (
    finding.shotId == null && !renderedFields.has(finding.field) && owns(finding.field)
  ));
}
