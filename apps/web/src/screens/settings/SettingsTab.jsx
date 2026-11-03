import React from "react";
import { Icon } from "../../components/Icons.jsx";
import { SCHEME_LABELS } from "../../credentials.js";
import { WorkflowEmbedDetails } from "../../components/WorkflowEmbedNotice.jsx";
import { formatBytes } from "../../formatting.js";
import { isDesktop } from "../../runtime.js";
import { policyNeedsRestart } from "../../modelCache.js";

export function SettingsTab({
  DesktopUpdatesGroup,
  SCHEME_OPTIONS,
  addCredential,
  appUpdate,
  cache,
  cacheDays,
  cacheError,
  cacheLimitGb,
  canSaveCredential,
  changeCacheEnabled,
  changeDataDir,
  changeModelLibrary,
  commitCacheInterval,
  commitCacheLimit,
  credentialLocation,
  credentials,
  dataDirLabel,
  embedWorkflow,
  gpu,
  modelLibraryFromEnvironment,
  modelLibraryNotice,
  modelLibraryPath,
  newHost,
  newLabel,
  newScheme,
  newToken,
  onChangeModelLibrary,
  onEmbedWorkflowChange,
  persistedPolicy,
  removeCredential,
  revealDataDir,
  revealModelLibrary,
  setCacheDays,
  setCacheLimitGb,
  setNewHost,
  setNewLabel,
  setNewScheme,
  setNewToken,
  settings,
  sharingHeadingRef,
}) {
  return (
<div className="settings-tab-body">
            {/* Sharing (sc-15953). The toggle writes `embedWorkflowInImages` into
                ui-preferences.json, which the WORKER re-reads at its PNG write seam on every job —
                so the change lands on the next generation with no restart. The copy under it is
                the same block the first-run disclosure shows; see WorkflowEmbedNotice.jsx for why
                its claims are shaped the way they are. */}
            <div className="settings-group-title" ref={sharingHeadingRef} tabIndex={-1}>Sharing</div>
            <div className="settings-row">
              <div>
                <div className="settings-row-title">Include the recipe in generated images</div>
                <div className="settings-row-sub">
                  New PNGs carry a small block of JSON, so dropping one back into SceneWorks
                  reloads the recipe that made it.
                </div>
              </div>
              <button
                aria-checked={embedWorkflow}
                aria-label="Include the recipe in generated images"
                className={embedWorkflow ? "settings-toggle on" : "settings-toggle"}
                onClick={() => onEmbedWorkflowChange?.(!embedWorkflow)}
                role="switch"
                type="button"
              >
                <span />
              </button>
            </div>
            <WorkflowEmbedDetails />
            <div className="work-panel-divider" />

            {isDesktop ? <DesktopUpdatesGroup update={appUpdate} /> : null}

            {isDesktop ? (
              <>
                <div className="settings-group-title">Storage</div>
                <div>
                  <div className="settings-row-title">Data directory</div>
                  <div className="settings-row-sub">
                    Projects, generated assets, models and LoRAs. Moving it needs a restart.
                  </div>
                  <div className="settings-path">{dataDirLabel}</div>
                  <div className="settings-button-row">
                    <button className="settings-btn" type="button" onClick={changeDataDir}>
                      Change…
                    </button>
                    <button
                      className="settings-btn"
                      type="button"
                      onClick={revealDataDir}
                      disabled={!settings?.dataDir}
                    >
                      Reveal in {gpu?.platform === "windows" ? "Explorer" : "Finder"}
                    </button>
                  </div>
                </div>
                {/* The Hugging Face cache home (`HF_HOME`) — where downloaded model weights live.
                    Changing it runs the sc-19709 relocation: the folder is checked first (with
                    models installed it must hold every one this install recorded, so nothing is
                    orphaned; before anything is installed any folder is acceptable), then the
                    override is persisted and the library re-bound. Like the data directory, the
                    sidecars pick it up at their next spawn, so it needs a restart. */}
                <div>
                  <div className="settings-row-title">Model library (Hugging Face cache)</div>
                  <div className="settings-row-sub">
                    Downloaded model weights. Choose the Hugging Face cache folder (the one
                    containing <code>hub</code>) — if you moved your cache to another drive, point
                    SceneWorks at it here. Needs a restart.
                  </div>
                  <div className="settings-path">{modelLibraryPath ?? "…"}</div>
                  {modelLibraryFromEnvironment ? (
                    <div className="settings-row-sub">
                      Set by the <code>HF_HOME</code> environment variable, which overrides
                      anything chosen here. Unset it to change the location from Settings.
                    </div>
                  ) : null}
                  {modelLibraryNotice ? (
                    <p aria-live="polite" className="settings-status" role="status">
                      {modelLibraryNotice}
                    </p>
                  ) : null}
                  <div className="settings-button-row">
                    <button
                      className="settings-btn"
                      type="button"
                      onClick={changeModelLibrary}
                      disabled={!onChangeModelLibrary || modelLibraryFromEnvironment}
                    >
                      Change…
                    </button>
                    <button
                      className="settings-btn"
                      type="button"
                      onClick={revealModelLibrary}
                      disabled={!modelLibraryPath}
                    >
                      Reveal in {gpu?.platform === "windows" ? "Explorer" : "Finder"}
                    </button>
                  </div>
                </div>
                <div className="work-panel-divider" />
              </>
            ) : null}

            {/* Resolved-model hot cache (epic 19703, sc-19711). The controls are shell-backed and
                therefore desktop-only, exactly like the data directory above; the usage readout
                comes from the API and renders in every deployment, because a remote admin still
                needs to see what the host is holding. */}
            <div className="settings-group-title">Local model copies</div>
            <div className="settings-row settings-row--top">
              <div>
                <div className="settings-row-title">
                  Keep a working copy of models from external libraries
                </div>
                <div className="settings-row-sub">
                  Copies models loaded from an attached or network model library into SceneWorks’
                  own storage, so they keep working when that library is disconnected. Off by
                  default. Takes effect after a restart.
                </div>
              </div>
              <button
                aria-checked={Boolean(persistedPolicy?.enabled)}
                aria-label="Keep a working copy of models from external libraries"
                className={persistedPolicy?.enabled ? "settings-toggle on" : "settings-toggle"}
                disabled={!isDesktop || !persistedPolicy}
                onClick={() => changeCacheEnabled(!persistedPolicy?.enabled)}
                role="switch"
                type="button"
              >
                <span />
              </button>
            </div>
            {isDesktop && persistedPolicy ? (
              <div className="settings-field-row">
                <label className="settings-note" htmlFor="model-cache-limit">
                  Use at most
                </label>
                <input
                  aria-label="Maximum size for local model copies, in GiB"
                  className="settings-port"
                  id="model-cache-limit"
                  min="1"
                  onBlur={commitCacheLimit}
                  onChange={(event) => setCacheLimitGb(event.target.value)}
                  type="number"
                  value={cacheLimitGb}
                />
                <span className="settings-note">GiB, and remove copies unused for</span>
                <input
                  aria-label="Remove local model copies unused for this many days"
                  className="settings-port"
                  id="model-cache-days"
                  min="1"
                  onBlur={commitCacheInterval}
                  onChange={(event) => setCacheDays(event.target.value)}
                  type="number"
                  value={cacheDays}
                />
                <span className="settings-note settings-grow">days.</span>
              </div>
            ) : null}
            {!isDesktop ? (
              <p className="settings-note">
                These settings are configured on the machine running SceneWorks.
              </p>
            ) : null}
            {policyNeedsRestart(persistedPolicy, cache?.policy) ? (
              <p className="settings-note">
                Saved. SceneWorks is still running under the previous setting — restart to apply
                it.
              </p>
            ) : null}
            {cacheError ? (
              <p className="inline-warning">Couldn’t read local copy storage: {cacheError}</p>
            ) : cache?.error ? (
              <p className="inline-warning">Couldn’t read local copy storage: {cache.error}</p>
            ) : cache?.policy ? (
              <div className="settings-inset settings-inset--spaced">
                <div className="settings-inset-title">Storage</div>
                <div className="settings-kv">
                  <span>Used</span>
                  <span className="settings-mono">
                    {formatBytes(cache.usedBytes)} of {formatBytes(cache.policy.maxBytes)}
                  </span>
                </div>
                <div className="settings-kv">
                  <span>Copies</span>
                  <span className="settings-mono">{cache.entryCount}</span>
                </div>
                <div className="settings-kv">
                  <span>Can be reclaimed</span>
                  <span className="settings-mono">{formatBytes(cache.reclaimableBytes)}</span>
                </div>
                <div className="settings-kv">
                  <span>Kept</span>
                  <span className="settings-mono">{formatBytes(cache.pinnedBytes)}</span>
                </div>
                {/* The external library these copies are made FROM. It is the location every other
                    line here is relative to — and the one the same-volume warning below is about —
                    so a user who has to act on that warning can see what is actually configured
                    instead of hunting for "the model library". Reported by the API rather than
                    re-derived here: the status read compares against this exact path, so showing
                    anything else could name a location the comparison never used. */}
                {cache.sourceLibraryPath ? (
                  <div className="settings-kv">
                    <span>Library</span>
                    <span className="settings-mono" title={cache.sourceLibraryPath}>
                      {cache.sourceLibraryPath}
                    </span>
                  </div>
                ) : null}
              </div>
            ) : null}
            {/* Same-volume is not an error, but it silently removes the whole point of the
                feature, so it is stated plainly rather than left for the user to infer — and it
                names the configured location, because "point the model library elsewhere" is not
                an actionable instruction to someone who cannot see where it currently points. */}
            {cache?.sourceVolumeRelation === "same" ? (
              <div className="settings-notice">
                <Icon.Warning size={16} />
                <span>
                  Local copies are on the same disk as the model library
                  {cache.sourceLibraryPath ? ` (${cache.sourceLibraryPath})` : ""}, so they protect
                  against neither a disconnect nor a slow drive — they only use extra space. Point
                  the model library at a different volume to get the benefit.
                </span>
              </div>
            ) : null}
            <p className="settings-note">
              Remove individual copies, or mark one to keep, from a model’s card in Model Manager.
            </p>
            <div className="work-panel-divider" />

            <div className="settings-group-title">Service credentials</div>
            <p className="settings-note">
              API tokens for model &amp; LoRA downloads — Hugging Face, Civit.ai, any authenticated
              source. Stored in {credentialLocation}. Tokens are write-only: never shown again after
              saving, and a change takes effect on the next worker restart.
            </p>
            {credentials.length ? (
              <div className="settings-credential-list">
                {credentials.map((credential) => (
                  <div className="settings-credential" key={credential.host}>
                    <span
                      className={
                        credential.present === false
                          ? "settings-credential-glyph missing"
                          : "settings-credential-glyph"
                      }
                    >
                      {credential.present === false ? (
                        <Icon.Warning size={18} />
                      ) : (
                        <Icon.Model size={18} />
                      )}
                    </span>
                    <span className="settings-credential-text">
                      <span className="settings-credential-host">{credential.host}</span>
                      <span className="settings-credential-meta">
                        {credential.label ? `${credential.label} · ` : ""}
                        {SCHEME_LABELS[credential.scheme] ?? credential.scheme}
                        {credential.present === false ? " · token missing" : " · token saved"}
                      </span>
                    </span>
                    <button
                      className="settings-btn settings-btn--quiet settings-btn--compact"
                      type="button"
                      onClick={() => removeCredential(credential.host)}
                    >
                      Remove
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <p className="settings-note">No credentials saved.</p>
            )}
            <div className="settings-inset settings-credential-form">
              <div className="settings-inset-title">Add a token</div>
              <div className="settings-field-pair">
                <input
                  type="text"
                  placeholder="huggingface.co"
                  value={newHost}
                  onChange={(event) => setNewHost(event.target.value)}
                  aria-label="Credential host"
                />
                <input
                  type="text"
                  placeholder="Label (optional)"
                  value={newLabel}
                  onChange={(event) => setNewLabel(event.target.value)}
                  aria-label="Credential label"
                />
              </div>
              <div
                className="settings-chips settings-chips--narrow"
                role="radiogroup"
                aria-label="Authentication scheme"
              >
                {SCHEME_OPTIONS.map(([value, label]) => (
                  <button
                    aria-checked={newScheme === value}
                    className={newScheme === value ? "settings-chip active" : "settings-chip"}
                    key={value}
                    onClick={() => setNewScheme(value)}
                    role="radio"
                    type="button"
                  >
                    {label}
                  </button>
                ))}
              </div>
              <div className="settings-field-row">
                <input
                  className="settings-grow"
                  type="password"
                  placeholder="Access token"
                  value={newToken}
                  onChange={(event) => setNewToken(event.target.value)}
                  aria-label="Credential token"
                />
                <button
                  className="settings-btn settings-btn--primary"
                  type="button"
                  onClick={addCredential}
                  disabled={!canSaveCredential}
                >
                  Add
                </button>
              </div>
            </div>
          </div>
  );
}
