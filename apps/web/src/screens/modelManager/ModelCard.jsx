import { LinkedLibraryStatus } from "./LinkedLibraryStatus.jsx";
import React from "react";

export function ModelCard({ model, RETIRED_MODEL_CAPABILITIES, audioCapabilityChips, availabilityBadge, blanketFloorGb, cacheBusyKey, cacheConvergenceStalled, cacheEligibilityBadge, canHoldLocalCopy, capabilityLabel, convertJobsFor, createModelDownloadJob, credentials, deleteModel, deleteModelVariant, deletingItem, describeEntryState, describeMissingLocalCopy, downloadJobsFor, downloadSizeText, entriesForModel, formatBytes, gatedRepoUrl, handleUpdateModel, hasPresentCredential, isRecommendedModel, isTransitionalEntry, licenseAcks, licenseComponentForModel, lightestInstallableTier, linkedCorrection, linkedStatuses, localCopiesEnabled, macCapabilities, macModelBlock, memoryBackend, mlxStatusText, modelCache, modelLinkedStatus, modelProvenance, onCancelJob, onConvertModel, onDeleteModel, onDownloadModel, onDownloadVariant, onFreshDownloadJob, onOpenQueue, onResumeDownloadJob, orderedMatrixVariants, pendingUpdate, refreshModelCache, relinkLinkedModel, removeLocalCopy, requiresLicenseAcknowledgment, rescanLinkedModel, setActiveTab, setActiveView, setCheckpointRepairRequest, setLicenseAck, terminalStatuses, tierLabel, toggleKeepLocalCopy, unifiedMemoryGb, ConvertedTierList, ImportedCheckpointSurface, LicenseGateNotice, ModelComponentDetails, ModelLicenseSummary, ModelTierDownloadPanel, WorkerProgressCard }) {
    const downloadJobs = downloadJobsFor(model);
    const downloadJob = downloadJobs.find((job) => !terminalStatuses.has(job.status));
    const installed = model.installState === "installed";
    const cleanupOnly = model.platformCleanupOnly === true;
    const incomplete = model.cacheState === "incomplete" || model.repairAvailable;
    const repairVariant = model.variants?.find((variant) => !variant.installed && variant.cacheState === "incomplete");
    const missingRequiredFiles = Array.isArray(model.missingRequiredFiles) ? model.missingRequiredFiles : [];
    const localDownloadJob = cleanupOnly || installed ? null : downloadJobs.find((job) => job.status !== "completed");
    const failedDownload = localDownloadJob && terminalStatuses.has(localDownloadJob.status);
    const downloadSize = downloadSizeText(model, "Unavailable");
    const unassociated = !model.family;
    const capabilities = Array.isArray(model.capabilities)
      ? model.capabilities.filter((capability) => !RETIRED_MODEL_CAPABILITIES.has(capability))
      : [];
    // Audio-type cards describe themselves via the `audio` sub-block (voice count,
    // languages, edit modes, multi-speaker) instead of the generic capabilities[].
    const audioChips = audioCapabilityChips(model);
    const deleteKey = `model:${model.id}`;
    const canDelete = Boolean(onDeleteModel) && model.removable !== false;
    // MLX (macOS) variant: only present when the catalog computed mlxConversionState.
    // Conversion state is a platform capability supplied by the API, not a memory measurement.
    // Keep that control surface intact while guarding the MLX memory block by the active lane.
    const mlxState = cleanupOnly ? null : model.mlxConversionState;
    const mlxMinGb = memoryBackend === "mlx" ? blanketFloorGb(model, "mlx", lightestInstallableTier(model)) : null;
    const mlxEnoughMemory = unifiedMemoryGb == null || mlxMinGb == null || unifiedMemoryGb >= mlxMinGb;
    const convertJobs = convertJobsFor(model);
    const convertJob = convertJobs.find((job) => !terminalStatuses.has(job.status));
    const failedConvert = convertJobs.find((job) => terminalStatuses.has(job.status) && job.status !== "completed");
    const showConvertButton = mlxState === "needs_conversion" || mlxState === "converted";
    const gated = Boolean(model.gated);
    const credentialPresent = gated && hasPresentCredential(credentials, model.credentialHost);
    // Quant-matrix models (sc-8509): render the per-tier download panel with a RAM-based suggestion
    // + multi-select instead of the single Download button. Single-variant models are unchanged.
    const matrixVariants = orderedMatrixVariants(model);
    const hasTierMatrix = model.hasVariantMatrix === true && matrixVariants.length > 0;
    // License-acknowledgment gate (sc-7872, decoupled from the credential by sc-17227): a model
    // whose licence must be acknowledged can't be downloaded until the user accepts it in-app.
    // The requirement comes from `gated` OR the standalone `requiresLicenseAcknowledgment` — the
    // latter covers a public repo whose licence still binds the user (MiniMax-H3).
    //
    // The predicate follows the DOWNLOAD CONTROLS, and is derived from the same conditions that
    // decide whether any of them render — deliberately, because deriving it independently is how
    // this went wrong: `!installed || updateAvailable` read "installed" as "nothing left to
    // download", which is FALSE for a quant matrix. `install_state_for` (apps/rust-api/src/models.rs)
    // marks a matrix model installed when ANY tier is present, so a MiniMax-H3 with q4 installed and
    // q8/bf16 missing reported `installed: true`, `updateAvailable: false` — no notice, no
    // checkbox — while the tier panel still offered bf16 and q8 and the choke point
    // (`licenseAcknowledgmentBlocked`) still refused the click, telling the user to accept a licence
    // on a card that rendered nothing to accept. Unrecoverable through the UI, and hit on every
    // desktop relaunch, where localStorage does not survive.
    //
    // Each disjunct is a control that can start a download:
    //   !installed           → the footer Download button
    //   incomplete           → the footer Fix button (a torn cache re-fetches)
    //   updateAvailable      → the Update button (MLX block and footer); also the `breaking_update`
    //                          shape, which reports installed:false + updateAvailable:true
    //   an uninstalled tier  → the tier panel's checkboxes + its Download button
    // An installed model with none of those shows no notice: the acknowledgment is a pre-download
    // step, and re-blocking a finished install would be noise.
    const licenseGated = requiresLicenseAcknowledgment(model);
    const downloadOnOffer =
      !installed ||
      incomplete ||
      model.updateAvailable === true ||
      (hasTierMatrix && matrixVariants.some((variant) => variant.installState !== "installed"));
    const licenseGateApplies = licenseGated && downloadOnOffer;
    const licenseAcknowledged = licenseAcks[model.id] === true;
    const licenseAckRequired = licenseGateApplies && !licenseAcknowledged;
    const firstCapability = capabilities.length ? capabilityLabel(capabilities[0]) : null;
    const familyMeta = [model.family ?? "unassociated", firstCapability].filter(Boolean).join(" · ");
    const macBlock = macModelBlock(model, macCapabilities);
    // Header install-status badge: warn tones for an incomplete cache, accent for installed,
    // neutral for missing.
    // A linked checkpoint whose library is detached, or whose file drifted, is NOT missing: the
    // plans are intact and there is exactly one button to press (AC2, epic 20398). Rendering it as
    // "missing" invites the user to re-download bytes they already own.
    const linkedStatus = modelLinkedStatus(model, linkedStatuses);
    const linkedFix = linkedCorrection(linkedStatus);
    const provenance = modelProvenance(model);
    const statusClass = linkedFix
      ? "status-badge warning"
      : incomplete
        ? "status-badge warning"
        : installed
          ? "status-badge installed"
          : "status-badge";
    const statusText = linkedFix
      ? linkedFix.headline.toLowerCase()
      : incomplete
        ? "incomplete"
        : installed
          ? "installed"
          : "missing";
    // Where this model's files actually resolve from, straight off the typed judgement the one
    // shared resolver produced (sc-19708). NEVER re-derived here from paths or error text — that
    // discipline is the whole reason a second, drifting availability opinion can't exist.
    // `local_ready` is itself local-copy messaging, so suppress even a stale catalog judgement
    // until the current cache policy is known to be enabled. The other availability badges still
    // matter with caching off (for example, that the external library is disconnected).
    const availability =
      model.modelAvailability === "local_ready" && !localCopiesEnabled
        ? null
        : availabilityBadge(model);
    // How much of this model a local copy could ever serve, straight off the backend's typed
    // `cacheEligibility` (sc-19712 F-5). Null for a fully cacheable model and for a row with no
    // external requirement closure; a badge whenever the local-copy affordance would over-promise.
    const cacheCoverage = localCopiesEnabled ? cacheEligibilityBadge(model) : null;
    // Local copies of THIS model, joined by the backend.
    const localCopies = localCopiesEnabled ? entriesForModel(modelCache, model.id) : [];
    // The block renders ONLY when the cache state is actually known. `modelCache` is null when the
    // status read failed outright, and a returned snapshot carries `error` when the store exists
    // but could not be listed — in both cases the entry list is empty for want of an answer, not
    // because there are no copies. Gating on `cacheKnown` is what stops "No local copy yet." from
    // being rendered as a confident claim over a read that never succeeded.
    const cacheKnown = Boolean(modelCache) && !modelCache.error;
    const showLocalCopySection =
      localCopiesEnabled && cacheKnown && (localCopies.length > 0 || canHoldLocalCopy(model));
    return (
      <article className={model.recommended ? "model-card recommended" : "model-card"} key={model.id}>
        <div className="model-card-head">
          <span className="model-card-title">
            <strong>{model.name}</strong>
            <small>{familyMeta}</small>
          </span>
          <span className="model-card-status">
            <span className={statusClass}>{statusText}</span>
            {availability ? (
              <span
                className={availability.tone ? `status-badge ${availability.tone}` : "status-badge"}
                title={availability.title}
              >
                {availability.text}
              </span>
            ) : null}
            {cacheCoverage ? (
              <span
                className={cacheCoverage.tone ? `status-badge ${cacheCoverage.tone}` : "status-badge"}
                title={cacheCoverage.title}
              >
                {cacheCoverage.text}
              </span>
            ) : null}
            {model.updateAvailable ? <span className="status-badge warning">update available</span> : null}
            {unassociated ? (
              <span className="status-badge warning" title="Set this model's family in user.models.jsonc before using it for generation.">
                needs family
              </span>
            ) : null}
            {macBlock ? (
              <span className="status-badge warning" title={macBlock.text}>
                not on Mac
              </span>
            ) : null}
          </span>
        </div>
        {isRecommendedModel(model) ? <span className="model-card-rec-chip">★ Recommended</span> : null}
        {/* Licence-required UI attribution (sc-17227). Some upstream licences oblige the product
            to display a specific string prominently — MiniMax H3 Community License §IV.2 requires
            "MiniMax H3" on the user interface. Its own line above the description, not folded into
            prose, so it is legible as attribution rather than marketing copy. */}
        {model.ui?.attribution ? (
          <p className="model-card-attribution">{model.ui.attribution}</p>
        ) : null}
        <p className="model-card-description">{model.ui?.description ?? model.family ?? model.id}</p>
        {provenance ? (
          <p className="model-card-provenance">
            Source: {provenance.label}
            {provenance.reference ? ` · ${provenance.reference}` : ""}
          </p>
        ) : null}
        {linkedFix ? (
          <LinkedLibraryStatus model={model} linkedFix={linkedFix} linkedStatus={linkedStatus} deletingItem={deletingItem} relinkLinkedModel={relinkLinkedModel} rescanLinkedModel={rescanLinkedModel} />
        ) : null}
        {capabilities.length ? (
          <ul className="model-capabilities">
            {capabilities.map((capability) => (
              <li className="chip" key={capability}>
                {capabilityLabel(capability)}
              </li>
            ))}
          </ul>
        ) : null}
        {audioChips.length ? (
          <ul className="model-capabilities model-audio-capabilities">
            {audioChips.map((chip) => (
              <li className="chip" key={chip}>
                {chip}
              </li>
            ))}
          </ul>
        ) : null}
        {!cleanupOnly && licenseGateApplies ? (
          <LicenseGateNotice
            credentialRequired={gated}
            host={model.credentialHost}
            repoUrl={gated ? (gatedRepoUrl(model) ?? model.licenseUrl ?? null) : null}
            licenseUrl={model.licenseUrl}
            licenseNotice={model.licenseNotice}
            present={credentialPresent}
            acknowledged={licenseAcknowledged}
            onAcknowledgeChange={(checked) => setLicenseAck(model.id, checked)}
            onOpenSettings={() => setActiveView("Settings")}
          />
        ) : null}
        {/* The PERSISTENT half of the licence surface (sc-24108). The gate above disappears the
            moment the model finishes installing, which used to take the restriction with it — this
            row keeps the licence name, the link and the full notice reachable from model details
            for the whole life of the install. Its condition is the PRESENCE of licence terms, not
            the install state; `gateVisible` only stops the same paragraphs being printed twice
            while the gate is still on screen. */}
        {!cleanupOnly ? (
          <ModelLicenseSummary
            licenseName={licenseComponentForModel(model.id)?.license}
            licenseUrl={model.licenseUrl}
            licenseNotice={model.licenseNotice}
            nonCommercial={model.nonCommercial === true}
            gateVisible={licenseGateApplies}
          />
        ) : null}
        {/* Per-component installation details + supported limits (sc-24216). Collapsed by default;
            renders only for an entry that declares per-download licences or a prompt limit. */}
        {!cleanupOnly ? <ModelComponentDetails model={model} totalSizeText={model.downloadSizeLabel ? downloadSize : null} /> : null}
        {/* The imported-checkpoint product surface (sc-26687): source variant, supported
            operations/backends, component readiness and the repair steps that would complete the
            assembly. Values come from the server's single derivation, so a repair offered here is
            the repair the API describes. Collapsed by default, like the component table above. */}
        {!cleanupOnly ? (
          <ImportedCheckpointSurface
            model={model}
            onRepair={(repair) => {
              if (repair.kind === "download") {
                // An imported DiT is not downloadable. The server binds this resident component
                // to the import's compatible base and tier; never retry an unrelated old job.
                createModelDownloadJob(model, { componentId: repair.componentId });
                return;
              }
              if (repair.kind === "reimport") {
                if (model.importedSource?.ownershipMode === "managed") {
                  setCheckpointRepairRequest({ model });
                  setActiveTab(model.type === "video" ? "video" : "image");
                }
                return;
              }
              // relink / rescan: the linked-repair affordance above, which is the seam that decides
              // which of the two the library root index calls for.
              if (linkedStatus && linkedFix) {
                if (linkedFix.action === "relink") {
                  relinkLinkedModel(linkedStatus);
                } else {
                  rescanLinkedModel(linkedStatus);
                }
              }
            }}
          />
        ) : null}
        {incomplete ? (
          <p className="inline-warning">
            Cached files are incomplete
            {missingRequiredFiles.length ? `: ${missingRequiredFiles.slice(0, 3).join(", ")}${missingRequiredFiles.length > 3 ? "..." : ""}` : ""}.
          </p>
        ) : null}
        {cleanupOnly ? (
          <p className="inline-warning">
            This model is no longer available on this platform. Delete it to reclaim its existing local files.
          </p>
        ) : null}
        {!cleanupOnly && localDownloadJob ? (
          <WorkerProgressCard
            job={localDownloadJob}
            onCancel={onCancelJob}
            onRetry={onResumeDownloadJob}
            onFreshRetry={onFreshDownloadJob}
            onOpenQueue={onOpenQueue}
          />
        ) : null}
        {!cleanupOnly && mlxState ? (
          <div className="mlx-status">
            <div className="mlx-status-badges">
              {/* sc-20529: device-neutral, like `mlxStatusText`. This block renders off-Mac too
                  now (the candle convert lane), where an "MLX" chip mislabels a CUDA artifact. */}
              <span className="status-badge">Conversion</span>
              {/* The model-level `mlx.minMemoryGb` is a single blanket floor = the HEAVIEST tier's
                  worst case (e.g. Wan A14B bf16, both MoE experts dense = 133 GB). Showing it
                  tier-agnostically over-warns quant-matrix models whose default/installed tier is q4
                  (measured ~24 GiB, one MoE expert resident at a time) — it reads "won't run" on Macs
                  that run q4/q8 fine. For a matrix model the per-tier download panel below drives the
                  per-tier RAM guidance instead (sc-10042); keep the blanket badge/warning only for
                  single-variant MLX models, which have one legitimate footprint. */}
              {mlxMinGb != null && !hasTierMatrix ? (
                <span className={mlxEnoughMemory ? "status-badge" : "status-badge warning"}>needs ≥ {mlxMinGb} GB</span>
              ) : null}
            </div>
            <p>{mlxStatusText(model)}</p>
            {/* A conversion that left files the engine would refuse (sc-24216): the catalog reports
                it as needing conversion and names the engine's reason. */}
            {model.mlxConversionProblem ? (
              <div className="inline-warning" role="alert">
                <p>The converted files are damaged or incomplete, so this model cannot load. Convert again to rebuild them from the downloaded checkpoint; if that fails, delete the model and download it again.</p>
                <details className="model-conversion-problem">
                  <summary>Details</summary>
                  <p>{model.mlxConversionProblem}</p>
                </details>
              </div>
            ) : null}
            {/* The last conversion's failure, so "Retry Conversion" is never offered without the
                reason it failed (sc-24216). */}
            {!convertJob && failedConvert && mlxState !== "converted" && (failedConvert.error || failedConvert.message) ? (
              <p className="inline-warning" role="alert">
                Conversion {failedConvert.status === "canceled" ? "was canceled" : "failed"}
                {failedConvert.error ? `: ${failedConvert.error}` : failedConvert.message ? `: ${failedConvert.message}` : "."}
              </p>
            ) : null}
            {!mlxEnoughMemory && !hasTierMatrix ? (
              <p className="inline-warning">
                Needs ≥ {mlxMinGb} GB unified memory; this Mac has ~{Math.round(unifiedMemoryGb)} GB. It may run out of memory.
              </p>
            ) : null}
            {convertJob ? <WorkerProgressCard job={convertJob} onCancel={onCancelJob} onOpenQueue={onOpenQueue} /> : null}
            {model.updateAvailable ? (
              <>
                <p className="inline-warning">
                  A newer checkpoint is available. Update re-downloads it and re-converts it.
                </p>
                <button
                  disabled={
                    Boolean(downloadJob) ||
                    Boolean(convertJob) ||
                    Boolean(pendingUpdate[model.id]) ||
                    !mlxEnoughMemory ||
                    licenseAckRequired
                  }
                  onClick={() => handleUpdateModel(model)}
                  title={licenseAckRequired ? "Accept the license above before downloading." : undefined}
                  type="button"
                >
                  {convertJob
                    ? convertJob.status
                    : downloadJob || pendingUpdate[model.id]
                      ? "Updating…"
                      : "Update"}
                </button>
              </>
            ) : showConvertButton ? (
              <button
                disabled={mlxState === "converted" || Boolean(convertJob) || !mlxEnoughMemory}
                onClick={() => onConvertModel?.(model)}
                type="button"
              >
                {convertJob
                  ? convertJob.status
                  : mlxState === "converted"
                    ? "Converted"
                    : failedConvert
                      ? "Retry Conversion"
                      : model.mlxConversionProblem
                        ? "Convert again"
                        : "Convert"}
              </button>
            ) : null}
          </div>
        ) : null}
        {!cleanupOnly && hasTierMatrix ? (
          <ModelTierDownloadPanel
            model={model}
            unifiedMemoryGb={unifiedMemoryGb}
            backend={memoryBackend}
            downloadJobs={downloadJobs}
            licenseAckRequired={licenseAckRequired}
            onDownloadVariant={onDownloadVariant}
            onDeleteVariant={deleteModelVariant}
            deletingItem={deletingItem}
          />
        ) : null}
        {/* Convert-at-install models (mlxTiers) have no download panel; surface their installed
            tiers with a per-tier delete so unused convert outputs can be reclaimed (sc-12025). */}
        {!cleanupOnly && !hasTierMatrix ? (
          <ConvertedTierList
            model={model}
            onDeleteVariant={deleteModelVariant}
            deletingItem={deletingItem}
          />
        ) : null}
        {/* Local copies of this model in the resolved-model cache (sc-19711). Separate from the
            download/install job surface above on purpose: promoting or reclaiming a local copy is
            not an install, and conflating them would make "removed" read as "uninstalled". */}
        {showLocalCopySection ? (
          <div className="model-local-copy">
            <div className="model-local-copy-head">
              <span className="status-badge">Local copy</span>
              {model.modelAvailability === "installed_external_unavailable" && !localCopies.length ? (
                <span className="status-badge warning">library disconnected</span>
              ) : null}
              {/* Convergence is visible, and it is honest about having stopped. While the store is
                  still working the screen re-reads on its own; once the bounded refresh gives up it
                  says so, and "Check again" remains the way to re-read at any point. */}
              {localCopies.some(isTransitionalEntry) ? (
                <>
                  <span className="model-local-copy-progress">
                    {cacheConvergenceStalled
                      ? "Still in progress — SceneWorks has stopped checking automatically."
                      : "Checking for changes…"}
                  </span>
                  <button onClick={refreshModelCache} type="button">
                    Check again
                  </button>
                </>
              ) : null}
            </div>
            {localCopies.length === 0 ? (
              // The empty state must not promise a copy the cache can never serve: for a model the
              // backend excluded (soft co-requisites, or a requirement with no recorded revision)
              // this says so instead (sc-19712 F-5). The block itself stays VISIBLE — the exclusion
              // is the thing the user has to be able to see.
              <p>{describeMissingLocalCopy(model)}</p>
            ) : (
              <ul className="model-local-copy-list">
                {localCopies.map((entry) => {
                  const busy = cacheBusyKey === entry.cacheKey;
                  const ready = entry.state === "complete";
                  // What this state means, in words, straight off the store's typed value — with a
                  // remedy attached for the states nothing resolves on its own.
                  const stateNote = describeEntryState(entry);
                  return (
                    <li className="model-local-copy-entry" key={entry.cacheKey}>
                      <span className="model-local-copy-text">
                        <span>
                          {entry.tier ? `${tierLabel(entry.tier)} · ` : ""}
                          {formatBytes(entry.bytes)}
                        </span>
                        <small className={stateNote.failure ? "model-local-copy-failure" : undefined}>
                          {stateNote.text}
                        </small>
                      </span>
                      <span className="model-local-copy-actions">
                        <button
                          disabled={busy || !ready}
                          onClick={() => toggleKeepLocalCopy(entry)}
                          type="button"
                        >
                          {entry.pinned ? "Allow automatic removal" : "Keep locally"}
                        </button>
                        <button
                          className="danger-action"
                          disabled={busy}
                          onClick={() => removeLocalCopy(model, entry)}
                          type="button"
                        >
                          {busy ? "Working…" : "Remove local copy"}
                        </button>
                      </span>
                    </li>
                  );
                })}
              </ul>
            )}
            <p>
              Removing a local copy never uninstalls the model — the original files in the model
              library are untouched.
            </p>
          </div>
        ) : null}
        {!cleanupOnly && model.updateAvailable && !mlxState ? (
          <p className="inline-warning">A newer model download is available; the installed version remains usable.</p>
        ) : null}
        <div className="model-card-footer">
          <span className="model-card-size">{downloadSize}</span>
          <div className="model-card-footer-actions">
            {cleanupOnly ? null : hasTierMatrix ? (
              // A quant-matrix model installs its tiers from the panel above. Keep only a Fix
              // affordance here for an incomplete cache or soft co-requisite update; otherwise
              // there's no single-tier button. Repair the torn tier, including its companions.
              incomplete || model.updateAvailable ? (
                <button
                  className="model-card-primary"
                  disabled={!model.downloadable || Boolean(downloadJob) || licenseAckRequired}
                  title={licenseAckRequired ? "Accept the license above before downloading." : undefined}
                  onClick={() =>
                    failedDownload
                      ? onResumeDownloadJob(localDownloadJob, { payloadChanges: { downloadAction: "resume" } })
                      : repairVariant
                        ? onDownloadVariant(model, repairVariant.variant)
                        : onDownloadModel(model)
                  }
                  type="button"
                >
                  {downloadJob ? downloadJob.status : failedDownload ? "Resume Download" : model.updateAvailable ? "Update" : "Fix"}
                </button>
              ) : null
            ) : (
              <button
                className="model-card-primary"
                disabled={(installed && !incomplete && !model.updateAvailable) || !model.downloadable || Boolean(downloadJob) || licenseAckRequired}
                title={licenseAckRequired ? "Accept the license above before downloading." : undefined}
                onClick={() =>
                  failedDownload
                    ? onResumeDownloadJob(localDownloadJob, { payloadChanges: { downloadAction: "resume" } })
                    : onDownloadModel(model)
                }
                type="button"
              >
                {downloadJob
                  ? downloadJob.status
                  : failedDownload
                      ? "Resume Download"
                      : incomplete
                        ? "Fix"
                        : model.updateAvailable
                          ? "Update"
                        : installed
                          ? "Ready"
                          : model.downloadSizeLabel
                            ? `Download ${downloadSize}`
                            : "Download"}
              </button>
            )}
            <button className="danger-action" disabled={!canDelete || deletingItem === deleteKey} onClick={() => deleteModel(model)} type="button">
              {model.removable === false ? "Protected" : deletingItem === deleteKey ? "Deleting" : "Delete"}
            </button>
          </div>
        </div>
      </article>
    );
  }
