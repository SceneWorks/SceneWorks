import { TIER_LABELS } from "./labConstants.js";
import { COMPUTE_PRECISION_TIP } from "./LabFields.jsx";
import React from "react";
import { Icon } from "../../components/Icons.jsx";
import { MAX_TAKES } from "../../yue2Lab.js";

export function Yue2GenerationSettings({
  identity,
  why,
  update,
  settings,
  tierRows,
  createModelDownloadJob,
  model
}) {
  return (
<div className="settings-bar audio-settings-bar yue2-settings">
              <div className="settings-bar-row">
                <label className="settings-field settings-field-model">
                  Model
                  <select aria-label="YuE2 model" disabled value={identity.id}>
                    <option value={identity.id}>
                      {identity.name} · {identity.version} · {identity.license || "licence restricted"}
                    </option>
                  </select>
                </label>
                <label className="settings-field settings-field-tier">
                  Weight tier
                  <select aria-label="Weight tier" disabled={Boolean(why("tier"))} title={why("tier") ?? undefined} onChange={(event) => update({ tier: event.target.value })} value={settings.tier}>
                    <option value="">Default (BF16 weights)</option>
                    {tierRows.map((row) => (
                      <option key={row.tier} value={row.tier}>
                        {TIER_LABELS[row.tier] ?? row.tier}
                        {row.installed ? "" : " — not installed"}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="settings-field">
                  <span className="yue2-field-label">
                    Compute precision
                    <span aria-label={COMPUTE_PRECISION_TIP} className="yue2-field-tip" role="img" tabIndex={0} title={COMPUTE_PRECISION_TIP}>
                      <Icon.Info size={13} />
                    </span>
                  </span>
                  <select aria-label="Compute precision" disabled={Boolean(why("computePolicy"))} title={why("computePolicy") ?? undefined} onChange={(event) => update({ computePolicy: event.target.value })} value={settings.computePolicy}>
                    <option value="">Choose compute precision</option>
                    <option value="auto">Auto (BF16 model + FP32 VAE on GPU; FP32 on CPU)</option>
                    <option value="bf16">BF16 (model + VAE on GPU)</option>
                    <option value="fp32">FP32 (model + VAE)</option>
                  </select>
                </label>
                <label className="settings-field">
                  Decoder
                  <select aria-label="Decoder" disabled={Boolean(why("decoder"))} title={why("decoder") ?? undefined} onChange={(event) => update({ decoder: event.target.value })} value={settings.decoder}>
                    <option value="">Default (standard)</option>
                    <option value="standard">Standard</option>
                    <option value="legacy">Legacy (add-on install)</option>
                  </select>
                </label>
                <label className="settings-field settings-field-count">
                  Takes
                  <input
                    aria-label="Takes"
                    disabled={Boolean(why("count"))}
                    title={why("count") ?? undefined}
                    max={MAX_TAKES}
                    min="1"
                    onChange={(event) => update({ count: event.target.value })}
                    type="number"
                    value={settings.count}
                  />
                </label>
              </div>
              {settings.decoder === "legacy" ? (
                <p className="yue2-muted">
                  The legacy decoder is an add-on.{" "}
                  <button
                    className="audio-link"
                    onClick={() =>
                      createModelDownloadJob?.(model, {
                        variant: settings.tier || undefined,
                        choices: { decoder: "legacy" },
                      })
                    }
                    type="button"
                  >
                    Install the legacy decoder
                  </button>
                </p>
              ) : null}
            </div>
  );
}
