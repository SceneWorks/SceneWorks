import React from "react";
import { safeExternalUrl } from "../urls.js";

// The pre-download licence gate's MARKUP, shared by every surface that can take the
// acknowledgment (sc-17137 review). The PREDICATE and the localStorage accessors live in
// `../licenseAcknowledgment.js`; this module is the other half — the notice, the links, and the
// checkbox — extracted from `screens/ModelManagerScreen.jsx` when the Setup Wizard grew its own
// copy of them. Two hand-maintained copies of a licence notice is exactly the text that must not
// drift: MiniMax H3 Community License §V.2 obliges us to notify each user that its use
// restrictions apply, and "the Models screen says it" is not a defence for the wizard.

// The Hugging Face page of a gated model's primary download repo — where the user clicks
// "Agree and access" to be granted access with their token (sc-5999). Derived from the first HF
// download repo (or the mlx repo), so it covers every gated model without a per-model manifest
// field. Returns null when no repo is known; each caller decides its own fallback (the Models
// screen falls back to `licenseUrl`, so a gated model with no HF row still links somewhere).
export function gatedRepoUrl(model) {
  const host = model.credentialHost || "huggingface.co";
  const repo =
    (model.downloads ?? []).find((entry) => entry.provider === "huggingface" && entry.repo)?.repo ??
    model.mlx?.repo;
  return repo ? `https://${host}/${repo}` : null;
}

// Two independent requirements share one box (sc-17227):
//
//   * CREDENTIAL (`credentialRequired`, from the catalog's `gated` — sc-1898). Gated models
//     (e.g. FLUX.1 [dev]) need a saved token on `credentialHost` plus access granted on the
//     model page before the download can succeed. When the matching credential is already
//     present we soften the notice to a ready state; otherwise we point the user at the
//     Settings credential screen. `present` is undefined while presence is still unknown
//     (e.g. the credential list hasn't loaded) — we still show the link then. `repoUrl` links
//     the gated repo so the user can request access (sc-5999); shown alongside `licenseUrl`
//     only when the license lives on a different page (e.g. Ideogram 4, whose terms are on
//     the source repo but access is on the SceneWorks repo).
//   * ACKNOWLEDGMENT (always, whenever this notice renders — sc-7872). The download button
//     stays disabled until `acknowledged`. `licenseNotice` is the manifest's statement of the
//     restrictions the user is accepting; MiniMax H3 Community License §V.2 obliges us to
//     notify each user that its use restrictions apply, and a bare "accept the license"
//     checkbox does not do that.
//
// A model can need the second without the first: a PUBLIC repo (no token, nothing to request)
// under a licence that still binds the user. Rendering the credential half unconditionally
// would tell that user to add a token that does not exist.
//
// `variant` is the ONE thing the two surfaces genuinely disagree about, and only on the
// credential half:
//
//   * `"card"` — the Models screen card. Settings exists and the credential list has loaded, so
//     the notice reports whether the token is already saved and offers the jump to Settings.
//   * `"onboarding"` — the first-run Setup Wizard. It overlays the whole app (Settings included)
//     and knows nothing about saved credentials, so it can only say the token will be needed and
//     defer it to after setup. Adds `setup-wizard-license` to the notice's classes.
//
// The acknowledgment half — the licence-only sentence, the manifest's `licenseNotice`, the links
// and the checkbox — is byte-identical across both, which is the drift this module prevents.
export function LicenseGateNotice({
  variant = "card",
  credentialRequired,
  host,
  repoUrl,
  licenseUrl,
  licenseNotice,
  present,
  acknowledged,
  onAcknowledgeChange,
  onOpenSettings,
}) {
  const onboarding = variant === "onboarding";
  const hostLabel = host || "the required service";
  const safeRepoUrl = credentialRequired ? safeExternalUrl(repoUrl) : null;
  const safeLicenseUrl = safeExternalUrl(licenseUrl);
  const showSeparateLicense = safeLicenseUrl && safeLicenseUrl !== safeRepoUrl;
  const ready = credentialRequired && present;
  const showSettings = !onboarding && credentialRequired && !present;
  // The wizard drops an EMPTY actions row rather than rendering a bare bordered strip; the card
  // always has the Settings button to put in it when it has nothing else.
  const showActions = !onboarding || Boolean(safeRepoUrl) || Boolean(safeLicenseUrl);
  const classNames = ["model-gated-notice"];
  if (ready) {
    classNames.push("ready");
  }
  if (onboarding) {
    classNames.push("setup-wizard-license");
  }
  return (
    <div className={classNames.join(" ")}>
      <p className={ready ? "inline-success" : "inline-warning"}>
        {!credentialRequired
          ? "License acknowledgment required. These weights carry a license that binds you directly — read it and accept before downloading."
          : onboarding
            ? `Gated download. It also needs a ${hostLabel} token with access granted on the model page (add one in Settings after setup). Accept the license below to download.`
            : present
              ? `Credential for ${hostLabel} saved — request access on the model page, then download.`
              : `Gated download. Add a ${hostLabel} token, then request access on the model page and accept the license before downloading.`}
      </p>
      {licenseNotice ? <p className="model-license-terms">{licenseNotice}</p> : null}
      {showActions ? (
        <div className="model-gated-actions">
          {showSettings ? (
            <button type="button" onClick={onOpenSettings}>
              Add token in Settings
            </button>
          ) : null}
          {safeRepoUrl ? (
            <a href={safeRepoUrl} target="_blank" rel="noreferrer noopener">
              Request access on Hugging Face
            </a>
          ) : null}
          {showSeparateLicense ? (
            <a href={safeLicenseUrl} target="_blank" rel="noreferrer noopener">
              Review license
            </a>
          ) : null}
        </div>
      ) : null}
      <label className="model-license-ack">
        <input
          type="checkbox"
          checked={acknowledged}
          onChange={(event) => onAcknowledgeChange(event.target.checked)}
        />
        <span>I have read and accept this model&rsquo;s license.</span>
      </label>
    </div>
  );
}
