import React from "react";
import {
  LOOP_UTILITY_REPO,
  loopTierAvailable,
  loopTierInstalled,
  loopTiers,
} from "../loopUtility.js";
import { useAppContextOptional } from "../context/AppContext.js";
import { RequiredModelsNotice } from "./RequiredModelsNotice.jsx";

export function LoopAccessHelp() {
  return (
    <p className="inline-warning">
      Full precision requires access from <a href={`https://huggingface.co/${LOOP_UTILITY_REPO}`} target="_blank" rel="noreferrer">Hugging Face</a>.
      {" "}Request access there and save a Hugging Face token in Settings → Credentials.
    </p>
  );
}

export function LoopUtilityTier({ model, tier, onChange, onDownload, disabled }) {
  const app = useAppContextOptional();
  const downloadJobs = (app?.jobs ?? []).filter((job) =>
    job.type === "model_download" && job.payload?.variant === tier,
  );
  const available = loopTierAvailable(model, tier);
  const row = model?.variants?.find((entry) => entry.variant === tier);
  const installed = loopTierInstalled(model, tier);
  return (
    <>
      <label>
        Tier
        <select disabled={disabled} onChange={(event) => onChange(event.target.value)} value={tier ?? ""}>
          {available ? null : <option value={tier ?? ""}>Unavailable{tier ? ` (${tier})` : ""}</option>}
          {loopTiers(model).map(([key, label]) => (
            <option key={key} value={key}>
              {label}{loopTierInstalled(model, key) ? "" : " — not installed"}
            </option>
          ))}
        </select>
      </label>
      {!available ? <p className="inline-warning" role="status">This tier is unavailable on this platform.</p> : null}
      {available && !installed ? (
        <>
          {tier === "bf16" ? <LoopAccessHelp /> : null}
          <RequiredModelsNotice
            key={tier}
            feature="This action"
            models={[{ ...model, name: `${model.name} ${tier}`, downloadSizeBytes: row?.downloadSizeBytes }]}
            downloadJobs={downloadJobs}
            requireDownloadJob
            onDownload={onDownload ? () => onDownload(model, { variant: tier }) : undefined}
          />
        </>
      ) : null}
    </>
  );
}
