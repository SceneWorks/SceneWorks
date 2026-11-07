import { safeExternalUrl } from "../../urls.js";
import { TIER_LABELS } from "./labConstants.js";
import React, { useState } from "react";
import { Icon } from "../../components/Icons.jsx";
import { commercialAlternatives, installJobStatusLabel, yue2TierRows } from "../../yue2Lab.js";
import { ErrorNotice, PolicyChips } from "../audioYue2Parts.jsx";

// ---- gate -----------------------------------------------------------------------------------

export function Yue2Gate({ model, models, identity, ack, onAccept, busy, error, optIn, onRetry }) {
  const [understood, setUnderstood] = useState(Boolean(optIn));
  const [accepted, setAccepted] = useState(Boolean(ack.acknowledged));
  const alternatives = commercialAlternatives(model, models);
  const canEnable = understood && accepted && !busy && ack.status === "ready";
  const licenseUrl = safeExternalUrl(identity.licenseUrl);
  return (
    <div className="yue2-gate" data-testid="yue2-gate">
      <div className="yue2-gate__head">
        <strong>Before you use {identity.name}</strong>
        <PolicyChips model={model} />
      </div>
      <p>
        YuE2 is a separate, <strong>experimental</strong> song model — not a newer YuE1 and never a substitute for
        it. Its weights are licensed for <strong>noncommercial use only</strong>. Nothing about the standard Audio
        Studio modes changes when you enable it.
      </p>
      {model.licenseNotice ? <p className="model-license-terms">{model.licenseNotice}</p> : null}
      {licenseUrl ? (
        <a href={licenseUrl} rel="noreferrer" target="_blank">
          Read the full licence
        </a>
      ) : null}
      {model.commercialUse && model.commercialUse.eligible === false ? (
        <div className="yue2-notice yue2-notice--warn" data-testid="yue2-commercial-note">
          <Icon.Warning size={15} />
          <span>
            Not for commercial use: {model.commercialUse.reason}
            {/[.!?]$/.test(String(model.commercialUse.reason ?? "").trim()) ? "" : "."}
            {alternatives.length ? ` For commercial work use ${alternatives.map((alt) => alt.name).join(", ")}.` : ""}
            {model.commercialUse.alternativeNote ? ` ${model.commercialUse.alternativeNote}` : ""}
          </span>
        </div>
      ) : null}
      {ack.status === "error" ? (
        <ErrorNotice error={ack.error} testId="yue2-ack-error">
          <button className="secondary-action" onClick={onRetry} type="button">
            Retry
          </button>
        </ErrorNotice>
      ) : null}
      <label className="model-license-ack">
        <input checked={understood} onChange={(event) => setUnderstood(event.target.checked)} type="checkbox" />I
        understand YuE2 is experimental and separate from YuE1
      </label>
      <label className="model-license-ack">
        <input checked={accepted} onChange={(event) => setAccepted(event.target.checked)} type="checkbox" />I accept
        the licence terms above — noncommercial use only
      </label>
      <button className="prompt-cta" disabled={!canEnable} onClick={onAccept} type="button">
        Enable the experimental Song Lab
      </button>
      <ErrorNotice error={error} testId="yue2-gate-error" />
    </div>
  );
}

// ---- install --------------------------------------------------------------------------------

export function Yue2Install({ model, downloads, onInstall, decoder }) {
  const rows = yue2TierRows(model);
  return (
    <div className="yue2-install" data-testid="yue2-install">
      <span className="eyebrow">Install YuE2</span>
      <p className="yue2-muted">
        BF16 is the released checkpoint. Q8 and Q4 are derived on this machine from it (the BF16 original is fetched
        first); they are never downloaded pre-quantized.
      </p>
      <ul className="yue2-tier-list">
        {rows.map((row) => (
          <li className="yue2-tier" key={row.tier}>
            <strong>{TIER_LABELS[row.tier] ?? row.tier}</strong>
            <span className={row.installed ? "status-badge installed" : "status-badge"}>{row.state}</span>
            {row.installable ? (
              <button
                className="secondary-action"
                disabled={downloads.some((job) => [job.payload?.variant, job.payload?.localDerivation?.variant].includes(row.tier))}
                onClick={() => onInstall(row.tier, decoder === "legacy" ? { decoder: "legacy" } : undefined)}
                type="button"
              >
                {row.derived ? `Derive ${row.tier} here` : `Download ${row.tier}`}
              </button>
            ) : null}
          </li>
        ))}
      </ul>
      {downloads.map((job) => (
        <p className="yue2-muted" key={job.id}>
          {job.payload?.localDerivation?.variant ?? job.payload?.variant ?? "install"}: {installJobStatusLabel(job)}
          {job.message ? ` — ${job.message}` : ""}
        </p>
      ))}
    </div>
  );
}
