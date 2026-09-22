import React from "react";
import { safeExternalUrl } from "../urls.js";

// The PERSISTENT licence row on a model's details (sc-24108).
//
// `LicenseGateNotice` is the PRE-DOWNLOAD half: it renders only while a download is on offer
// (`licenseGateApplies = requiresLicenseAcknowledgment(model) && downloadOnOffer`), because its job
// is to take an acknowledgment before bytes are fetched, and re-blocking a finished install would
// be noise. The consequence was that the restriction VANISHED from every surface the moment the
// model finished installing — the Models screen and the Setup Wizard were its only render sites —
// so a user could not go back and read what they had accepted, and the Simple UI never showed it at
// all. That is the half this component restores.
//
// It is deliberately NOT a gate: no checkbox, nothing disabled, nothing blocked. It is a summary —
// the licence name, a link to the full text, and the manifest's `licenseNotice` behind a
// DEFAULT-COLLAPSED `<details>` so a card that is mostly about tiers and disk usage is not buried
// under several paragraphs of licence prose the user has already read once.
//
// Its condition is the PRESENCE of licence data (`licenseNotice` or `licenseUrl`), never the
// install state. The one thing it does check is whether the gate is already on screen
// (`gateVisible`), because the gate renders the same `licenseNotice` and showing both at once would
// print the same several paragraphs twice on one card. The union of the two is therefore
// "whenever the model carries licence terms", which is the AC — shown before download (the gate)
// AND in model details (this row).
//
// `licenseName` is the human label ("Qwen RESEARCH LICENSE AGREEMENT"). The catalog does not carry
// one per model, so callers pass what they have and it falls back to a neutral phrase rather than
// inventing a licence name.
export function ModelLicenseSummary({
  licenseName,
  licenseUrl,
  licenseNotice,
  nonCommercial = false,
  gateVisible = false,
  className = "model-license-summary",
}) {
  if (gateVisible) {
    return null;
  }
  if (!licenseNotice && !licenseUrl) {
    return null;
  }
  const safeUrl = safeExternalUrl(licenseUrl);
  const label = licenseName || "Model license";
  return (
    <div className={className}>
      <p className="model-license-summary-head">
        <span className="model-license-summary-name">{label}</span>
        {nonCommercial ? (
          <span className="model-license-summary-tag">Non-commercial</span>
        ) : null}
        {safeUrl ? (
          <a href={safeUrl} target="_blank" rel="noreferrer noopener">
            Read the full license
          </a>
        ) : null}
      </p>
      {licenseNotice ? (
        <details className="model-license-summary-details">
          <summary>License restrictions</summary>
          <p className="model-license-terms">{licenseNotice}</p>
        </details>
      ) : null}
    </div>
  );
}
