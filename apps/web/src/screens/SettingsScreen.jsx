import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  SCHEME_LABELS,
  loadCredentials,
  removeCredentialRequest,
  saveCredential,
  serverToken,
} from "../credentials.js";
import { apiFetch } from "../api.js";
import { putUiPreferences } from "../uiPreferences.js";
import { isDesktop, tauriInvoke as invoke } from "../runtime.js";
import {
  GENERATION_QUALITY_OPTIONS,
  generationQualityLabel,
  readDefaultGenerationQuality,
  shortQualityLabel,
  writeDefaultGenerationQuality,
} from "../generationQuality.js";
import { writeClipboardText } from "../clipboard.js";
import { appConfirm } from "../appConfirm.jsx";
import { formatBytes } from "../formatting.js";
import {
  bytesToGib,
  daysToSeconds,
  describeDisableConsequence,
  describeLimitConsequence,
  fetchModelCache,
  gibToBytes,
  policyNeedsRestart,
  secondsToDays,
} from "../modelCache.js";
import { useCacheConvergence } from "../hooks/useCacheConvergence.js";
import { WorkPanel } from "../components/WorkPanel.jsx";
import { Icon } from "../components/Icons.jsx";
import { ModeTabs } from "../components/generationStudio.jsx";
import { WorkflowEmbedDetails } from "../components/WorkflowEmbedNotice.jsx";
import { ACCENTS } from "../accents.js";
import { describeDevice } from "../deviceSummary.js";
import { useAppContext } from "../context/AppContext.js";

// The data-dir / GPU / worker / wizard / remote-access controls are desktop-only
// (backed by Tauri commands in the shell) and are gated behind `isDesktop` so a
// remote LAN browser (epic 4484) doesn't render buttons that would call a missing
// Tauri bridge. Service credentials work in both deployments and route through the
// shared ../credentials.js transport (keychain on desktop, authed REST on the
// server / remote browser). `isDesktop`/`invoke` come from the unified runtime
// helper (story 6).
//
// Settings-page redesign handoff: the screen is the Page-Frame standard (page-frame →
// WorkPanel → .prompt-hero-top → ModeTabs, the same arrangement as Image Studio), split
// into four tabs so it stops being one long scroll. Theme, accent, the Simple-mode
// default and the engine/GPU readout — which previously only existed in Simple mode —
// are surfaced here too, wired to the SAME durable stores (`PUT /api/v1/ui-preferences`
// via App.jsx's changeTheme/changeAccent/changeSimpleUiDefault), so a change made in
// either screen is the same change.

// GPU memory cap (epic 7819). The persisted value is a fraction (0.1–0.99) of total unified
// memory, or null/absent for "no limit". The slider works in whole percent; 100% means Off.
const GPU_LIMIT_MIN_PERCENT = 10;
function fractionToPercent(fraction) {
  if (typeof fraction !== "number" || !Number.isFinite(fraction) || fraction >= 1) return 100;
  return Math.min(100, Math.max(GPU_LIMIT_MIN_PERCENT, Math.round(fraction * 100)));
}

// GPU memory telemetry (epic 7819, sc-7825). The worker publishes byte counts; the readout is in GB.
function bytesToGb(bytes) {
  if (typeof bytes !== "number" || !Number.isFinite(bytes) || bytes <= 0) return 0;
  return bytes / (1024 * 1024 * 1024);
}

function remotePasswordMeetsMinimum(password, minimumPasswordLength) {
  return (
    Number.isSafeInteger(minimumPasswordLength) &&
    minimumPasswordLength > 0 &&
    Array.from(password.trim()).length >= minimumPasswordLength
  );
}

const SCHEME_OPTIONS = [
  ["bearer", "Bearer header"],
  ["query", "Query token"],
];

export function SettingsScreen({
  accent,
  onAccentChange,
  simpleDefault = false,
  onSimpleDefaultChange,
  lockedToSimple = false,
  embedWorkflow = true,
  onEmbedWorkflowChange,
  // Settings → Storage → Model library → Change…. Owned by App.jsx (it shares the relocation
  // sequence and the restart disclosure with the unavailable-library prompt, sc-19709): runs the
  // native picker, validates + persists + re-binds, and opens the restart dialog. Resolves to the
  // adopted target, `null` when the picker was dismissed; rejects with the message to show.
  onChangeModelLibrary,
  sharingFocusRequest = 0,
}) {
  // theme/changeTheme and the worker registry come from the app context (the same values the
  // topbar toggle and Simple Settings read); accent + the Simple-mode default are drilled from
  // App.jsx, which owns them alongside the sidebar's mode switch.
  const { theme, changeTheme, visibleWorkers = [], macCapabilities } = useAppContext();
  const [settings, setSettings] = useState(null);
  // The first-run storage snapshot, read only for `hfHomeDefault`: the model library row shows the
  // folder the app is ACTUALLY reading from when no override is set, not "Default location" — the
  // whole point of the row is telling a user whose cache moved where SceneWorks still thinks it is.
  const [storageSetup, setStorageSetup] = useState(null);
  const [gpu, setGpu] = useState(null);
  const [credentials, setCredentials] = useState([]);
  const [newHost, setNewHost] = useState("");
  const [newLabel, setNewLabel] = useState("");
  const [newScheme, setNewScheme] = useState("bearer");
  const [newToken, setNewToken] = useState("");
  const [status, setStatus] = useState("");
  // Which tab is showing. Not persisted — the screen is short-lived.
  const [tab, setTab] = useState("appearance");
  const sharingHeadingRef = useRef(null);
  useEffect(() => {
    if (!sharingFocusRequest) return undefined;
    setTab("settings");
    const timer = setTimeout(() => {
      sharingHeadingRef.current?.scrollIntoView?.({ block: "start" });
      sharingHeadingRef.current?.focus();
    }, 0);
    return () => clearTimeout(timer);
  }, [sharingFocusRequest]);
  // LAN remote access (epic 4484 stories 4/11). `remote` is the RemoteAccessStatus
  // snapshot from the shell; only fetched/rendered on desktop.
  const [remote, setRemote] = useState(null);
  const [remotePort, setRemotePort] = useState("");
  const [remotePassword, setRemotePassword] = useState("");
  // GPU memory cap (epic 7819). Local slider percent for smooth dragging; committed to the shell
  // on release. 100 = Off (no limit). Synced from persisted settings once they load.
  const [gpuLimitPercent, setGpuLimitPercent] = useState(100);
  // Live MLX memory telemetry (epic 7819, sc-7825). The worker's latest snapshot, polled while the
  // GPU card is visible; null until the first sample arrives (or on platforms without it).
  const [gpuTelemetry, setGpuTelemetry] = useState(null);
  // Global "default generation quality" (epic 10721 / sc-10728): the app-wide baseline quant tier new
  // generations use when a model has no per-(screen,model) sticky pick, read at generation time by
  // `defaultTierSelection` (precedence rung 3). Persisted like theme/accent — a durable server copy in
  // `ui-preferences.json` (write-through PUT in `changeDefaultQuality`) plus a localStorage instant-paint
  // cache — because on the desktop shell the UI's `127.0.0.1:<port>` origin changes each launch and wipes
  // origin-keyed localStorage. Seeded here from the cache (App.jsx re-primes it from the server on launch);
  // q8 by default.
  const [defaultQuality, setDefaultQuality] = useState(readDefaultGenerationQuality);
  const defaultQualityRequest = useRef(0);
  // Resolved-model hot cache (epic 19703, sc-19711). `cache` is the API's authoritative snapshot:
  // the policy the RUNNING sidecars captured at spawn, plus the store's own usage accounting.
  // Fetched on mount, re-fetched after any action that could have changed it, and re-read on a
  // bounded cadence ONLY while the store still has an entry in flight — so the storage totals
  // converge while a promotion runs, and a settled cache is still read exactly once. A timer that
  // ran unconditionally would put back the per-row read cost sc-19708 removed from the catalog.
  const [cache, setCache] = useState(null);
  const [cacheError, setCacheError] = useState("");
  // Draft limit/interval so typing doesn't fire a save per keystroke; committed on blur / Apply.
  const [cacheLimitGb, setCacheLimitGb] = useState("");
  const [cacheDays, setCacheDays] = useState("");

  const refresh = useCallback(async () => {
    try {
      if (isDesktop) {
        const [loadedSettings, gpuInfo, storedCredentials, remoteAccess, storage] =
          await Promise.all([
            invoke("get_app_settings"),
            invoke("get_gpu_info"),
            invoke("list_credentials"),
            invoke("get_remote_access"),
            invoke("get_storage_setup"),
          ]);
        setSettings(loadedSettings);
        setStorageSetup(storage);
        setGpu(gpuInfo);
        setCredentials(storedCredentials ?? []);
        if (remoteAccess) {
          setRemote(remoteAccess);
          setRemotePort(String(remoteAccess.port));
        }
      } else {
        setCredentials(await loadCredentials());
      }
    } catch (error) {
      setStatus(String(error));
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Keep the slider in sync with the persisted cap (also after a commit echoes settings back).
  useEffect(() => {
    if (settings) setGpuLimitPercent(fractionToPercent(settings.gpuMemoryLimitFraction));
  }, [settings]);

  const refreshCache = useCallback(async () => {
    try {
      const snapshot = await fetchModelCache();
      setCache(snapshot);
      setCacheError("");
      return snapshot;
    } catch (error) {
      // Report the failure rather than rendering a reassuring empty cache.
      setCache(null);
      setCacheError(String(error?.message ?? error));
      return null;
    }
  }, []);

  useEffect(() => {
    refreshCache();
  }, [refreshCache]);

  // Bounded convergence refresh — the same one the Model Manager runs, so the two surfaces cannot
  // disagree about when the app stops polling. It re-reads only while an entry is still moving.
  useCacheConvergence(cache, refreshCache);

  // The persisted (shell-owned) policy seeds the editable fields. On a non-desktop deployment
  // there is no shell to ask, so the running policy is both what is persisted and what applies.
  const persistedPolicy = isDesktop ? (settings?.resolvedCache ?? null) : (cache?.policy ?? null);
  // Re-seed only when the persisted VALUES change, never on the containing object's identity:
  // `settings` is a fresh object after every `get_app_settings`, so depending on the policy object
  // would clobber whatever the user is mid-way through typing on each refresh. The primitives are
  // lifted into their own bindings so that intent survives `react-hooks/exhaustive-deps` instead
  // of being expressed as a dependency list the rule can't verify.
  const persistedMaxBytes = persistedPolicy?.maxBytes;
  const persistedInactivitySeconds = persistedPolicy?.inactivitySeconds;
  useEffect(() => {
    if (persistedMaxBytes === undefined || persistedInactivitySeconds === undefined) return;
    setCacheLimitGb(String(Math.round(bytesToGib(persistedMaxBytes))));
    setCacheDays(String(Math.round(secondsToDays(persistedInactivitySeconds))));
  }, [persistedMaxBytes, persistedInactivitySeconds]);

  // Poll live MLX memory telemetry while the GPU card is visible (epic 7819, sc-7825). macOS-only —
  // the worker only publishes the snapshot on the MLX path; elsewhere the command returns null.
  // Lightweight: a small file read in the shell every 2s, cleaned up on unmount / platform change.
  useEffect(() => {
    if (!isDesktop || gpu?.platform !== "macos" || !gpu?.unifiedMemoryMb) return undefined;
    let cancelled = false;
    const poll = async () => {
      try {
        const telemetry = await invoke("get_gpu_telemetry");
        if (!cancelled) setGpuTelemetry(telemetry ?? null);
      } catch {
        if (!cancelled) setGpuTelemetry(null);
      }
    };
    poll();
    const id = setInterval(poll, 2000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [gpu]);

  const secretStore =
    gpu?.platform === "windows"
      ? "Credential Manager"
      : gpu?.platform === "linux"
        ? "Secret Service (GNOME Keyring or KWallet)"
        : "Keychain";
  const credentialLocation = isDesktop
    ? `the system ${secretStore}`
    : "the server's credential store (a restricted file in the config directory)";
  const dataDirLabel = settings?.dataDir ?? "Default location";

  async function changeDataDir() {
    try {
      const picked = await invoke("choose_data_dir");
      if (picked) {
        const updated = await invoke("set_data_dir", { path: picked });
        setSettings(updated);
        setStatus("Data directory updated — restart SceneWorks to apply.");
      }
    } catch (error) {
      setStatus(String(error));
    }
  }

  async function revealDataDir() {
    if (settings?.dataDir) {
      await invoke("reveal_in_os", { path: settings.dataDir });
    }
  }

  // The Hugging Face cache home SceneWorks is ACTUALLY reading models from, as the shell resolves
  // it for the sidecar spawn (ambient `HF_HOME` first, then the persisted override, then the
  // platform default). The persisted override alone would lie whenever an ambient `HF_HOME` wins —
  // which is exactly when it matters, so the row shows the resolved path and, in that case, says
  // the environment owns it instead of offering a change that would not take effect.
  const modelLibraryFromEnvironment = storageSetup?.hfHomeFromEnvironment === true;
  const modelLibraryPath =
    storageSetup?.hfHomeActive ?? settings?.hfHome ?? storageSetup?.hfHomeDefault ?? null;
  // Outcome of the last Change…/Reveal, rendered INLINE at the row. The screen's shared status
  // line sits at the top of the panel — off-screen when the user is scrolled down at the Storage
  // group — so a relocation refusal there reads as a silent no-op (the sc-21389 field failure:
  // the API refused with a typed 409 and the user saw "nothing happened").
  const [modelLibraryNotice, setModelLibraryNotice] = useState("");

  async function changeModelLibrary() {
    setModelLibraryNotice("");
    try {
      const adopted = await onChangeModelLibrary?.();
      if (adopted?.hfHome) {
        // The shell persisted the override inside the relocation; re-read it rather than guessing.
        const [reloaded, storage] = await Promise.all([
          invoke("get_app_settings"),
          invoke("get_storage_setup"),
        ]);
        setSettings(reloaded);
        setStorageSetup(storage);
        setModelLibraryNotice("Model library updated — restart SceneWorks to apply.");
      }
    } catch (error) {
      setModelLibraryNotice(error?.message || String(error));
    }
  }

  async function revealModelLibrary() {
    if (!modelLibraryPath) return;
    try {
      await invoke("reveal_in_os", { path: modelLibraryPath });
    } catch (error) {
      // Most likely: the default cache home was never created because nothing was downloaded yet.
      setModelLibraryNotice(error?.message || String(error));
    }
  }

  async function addCredential() {
    try {
      const updated = await saveCredential({
        host: newHost,
        label: newLabel,
        scheme: newScheme,
        token: newToken,
      });
      setCredentials(updated ?? []);
      setNewHost("");
      setNewLabel("");
      setNewScheme("bearer");
      setNewToken("");
      setStatus("Credential saved.");
    } catch (error) {
      setStatus(String(error));
    }
  }

  async function removeCredential(host) {
    try {
      const updated = await removeCredentialRequest(host);
      setCredentials(updated ?? []);
      setStatus(`Removed the credential for ${host}.`);
    } catch (error) {
      setStatus(String(error));
    }
  }

  // Persist the global default generation quality (sc-10728). Write-through: the localStorage cache gives
  // an instant read for the studio, and the PUT makes it durable across desktop relaunches (the shell's
  // 127.0.0.1:<port> origin changes each launch, wiping origin-keyed localStorage). Same contract as
  // theme/accent in App.jsx — the endpoint MERGES, so sending only this field can't disturb theme/accent.
  // `putUiPreferences` attaches the access token: only the ui-preferences GET is public, the disk-writing
  // PUT is gated (sc-8869, F-067), so a credential-less write 401s on any remote-auth deployment (sc-15136).
  async function changeDefaultQuality(value) {
    const request = defaultQualityRequest.current + 1;
    defaultQualityRequest.current = request;
    const previous = defaultQuality;
    const next = writeDefaultGenerationQuality(value);
    setDefaultQuality(next);
    try {
      await putUiPreferences({ defaultGenerationQuality: next });
      if (defaultQualityRequest.current === request) {
        setStatus(`Default generation quality set to ${generationQualityLabel(next)}.`);
      }
    } catch (error) {
      // An older request may finish after a newer click. Only the latest operation owns the
      // optimistic cache/state; a stale failure must not roll back a newer successful choice.
      if (defaultQualityRequest.current === request) {
        writeDefaultGenerationQuality(previous);
        setDefaultQuality(previous);
        setStatus(`Could not save default generation quality: ${String(error)}`);
      }
    }
  }

  async function restartWorker() {
    try {
      // Desktop kills the worker child directly via Tauri; a remote admin restarts it
      // over REST (epic 4484 story 12), which signals the desktop supervisor to respawn.
      if (isDesktop) {
        await invoke("restart_worker");
      } else {
        await apiFetch("/api/v1/worker/restart", serverToken(), { method: "POST" });
      }
      setStatus("Restarting the inference worker…");
    } catch (error) {
      setStatus(String(error));
    }
  }

  async function commitGpuMemoryLimit(percent) {
    try {
      // 100% = Off → send null so the shell clears it; otherwise a fraction (0.1–0.99).
      const fraction = percent >= 100 ? null : percent / 100;
      const updated = await invoke("set_gpu_memory_limit", { fraction });
      setSettings(updated);
      // sc-7824: the shell writes the new ceiling to a live-handoff file the running worker re-reads
      // between jobs, so no restart is needed — it applies within a couple of seconds.
      setStatus(
        fraction == null
          ? "GPU memory target turned off — applies within a couple of seconds."
          : `GPU memory target set to ${percent}% — applies within a couple of seconds.`,
      );
    } catch (error) {
      setStatus(String(error));
    }
  }

  // Persist a resolved-cache policy through the shell (the durable copy the sidecars read at
  // spawn). Every caller passes a COMPLETE policy so a partial write can never leave the store
  // running under a half-applied rule. The status line says "after a restart" because that is
  // literally when it takes effect — the three sidecar processes captured their policy at spawn.
  async function commitCachePolicy(policy, message) {
    try {
      const updated = await invoke("set_resolved_cache_policy", { policy });
      setSettings(updated);
      await refreshCache();
      setStatus(message);
    } catch (error) {
      setStatus(String(error));
    }
  }

  // Turning the cache OFF is not destructive and does not sweep. Say what actually happens to the
  // copies already on disk before committing, so nobody discovers it afterwards.
  async function changeCacheEnabled(enabled) {
    if (!persistedPolicy) return;
    if (!enabled) {
      const proceed = await appConfirm({
        title: "Stop keeping local copies?",
        message: describeDisableConsequence(cache),
        confirmLabel: "Turn off",
      });
      if (!proceed) return;
    }
    await commitCachePolicy(
      { ...persistedPolicy, enabled },
      enabled
        ? "Local model copies enabled — takes effect after a restart."
        : "Local model copies disabled — takes effect after a restart. Existing copies stay on disk.",
    );
  }

  // Lowering the limit below current usage schedules future cleanup; it never sweeps here. The
  // confirm spells out how much can actually be reclaimed and how much is kept.
  async function commitCacheLimit() {
    if (!persistedPolicy) return;
    const nextMaxBytes = gibToBytes(cacheLimitGb);
    if (nextMaxBytes <= 0) {
      setStatus("The size limit must be at least 1 GiB.");
      setCacheLimitGb(String(Math.round(bytesToGib(persistedPolicy.maxBytes))));
      return;
    }
    if (nextMaxBytes === Number(persistedPolicy.maxBytes)) return;
    const consequence = describeLimitConsequence(cache, nextMaxBytes);
    if (consequence) {
      const proceed = await appConfirm({
        title: "Lower the local copy limit?",
        message: consequence,
        confirmLabel: "Set limit",
      });
      if (!proceed) {
        setCacheLimitGb(String(Math.round(bytesToGib(persistedPolicy.maxBytes))));
        return;
      }
    }
    await commitCachePolicy(
      { ...persistedPolicy, maxBytes: nextMaxBytes },
      `Local copy limit set to ${formatBytes(nextMaxBytes)} — takes effect after a restart.`,
    );
  }

  async function commitCacheInterval() {
    if (!persistedPolicy) return;
    const nextSeconds = daysToSeconds(cacheDays);
    if (nextSeconds <= 0) {
      setStatus("The unused-for interval must be at least one day.");
      setCacheDays(String(Math.round(secondsToDays(persistedPolicy.inactivitySeconds))));
      return;
    }
    if (nextSeconds === Number(persistedPolicy.inactivitySeconds)) return;
    await commitCachePolicy(
      { ...persistedPolicy, inactivitySeconds: nextSeconds },
      `Local copies are removed after ${Math.round(secondsToDays(nextSeconds))} unused days — takes effect after a restart.`,
    );
  }

  async function rerunSetupWizard() {
    try {
      await invoke("reset_setup");
      window.location.reload();
    } catch (error) {
      setStatus(String(error));
    }
  }

  // --- Remote access (LAN) handlers (epic 4484 story 4) ---
  async function saveRemotePassword() {
    const minimumPasswordLength = remote?.minimumPasswordLength;
    if (!Number.isSafeInteger(minimumPasswordLength) || minimumPasswordLength < 1) {
      setStatus("Remote-access password policy is unavailable. Restart SceneWorks and try again.");
      return;
    }
    if (!remotePasswordMeetsMinimum(remotePassword, minimumPasswordLength)) {
      setStatus(
        `Use a remote-access password with at least ${minimumPasswordLength} characters after trimming surrounding whitespace.`,
      );
      return;
    }
    try {
      const updated = await invoke("set_remote_access_password", {
        password: remotePassword.trim(),
      });
      setRemote(updated);
      setRemotePassword("");
      setStatus("Remote access password saved.");
    } catch (error) {
      setStatus(String(error));
    }
  }

  async function clearRemotePassword() {
    try {
      const updated = await invoke("clear_remote_access_password");
      setRemote(updated);
      setStatus("Password cleared — remote access disabled.");
    } catch (error) {
      setStatus(String(error));
    }
  }

  async function applyRemoteAccess(enabled) {
    const port = Number.parseInt(remotePort, 10);
    if (!Number.isInteger(port) || port < 1024 || port > 65535) {
      setStatus("Choose a port between 1024 and 65535.");
      return;
    }
    try {
      const updated = await invoke("set_remote_access", { enabled, port });
      setRemote(updated);
      setRemotePort(String(updated.port));
      setStatus(
        enabled
          ? "Remote access enabled — restart SceneWorks to apply."
          : "Remote access disabled — restart SceneWorks to apply.",
      );
    } catch (error) {
      setStatus(String(error));
    }
  }

  async function copyRemoteUrl() {
    if (!remote?.url) {
      return;
    }
    if (await writeClipboardText(remote.url)) {
      setStatus(`Copied ${remote.url}`);
    } else {
      setStatus("Couldn't copy to clipboard.");
    }
  }

  const canSaveCredential = newHost.trim() && newToken.trim();

  // Engine / GPU are read off the live worker registry — the one signal available in every
  // deployment (desktop, Docker, remote browser), so "This machine" renders in both.
  const device = useMemo(
    () => describeDevice(visibleWorkers, macCapabilities),
    [visibleWorkers, macCapabilities],
  );

  // The Server tab exists only where remote access does (the same condition that gated the old
  // Remote-access card). When it disappears under a selected "server" tab, fall back to Appearance
  // rather than rendering an empty panel.
  const serverTabAvailable = Boolean(isDesktop && remote);
  const activeTab = tab === "server" && !serverTabAvailable ? "appearance" : tab;
  const tabOptions = [
    ["appearance", "Appearance"],
    ["settings", "Settings"],
    ["device", "Device"],
    ...(serverTabAvailable ? [["server", "Server"]] : []),
  ];

  const macGpuMemory = gpu?.platform === "macos" && Boolean(gpu?.unifiedMemoryMb);
  const unifiedGb = macGpuMemory ? Math.round(gpu.unifiedMemoryMb / 1024) : 0;
  const gpuTargetLabel =
    gpuLimitPercent >= 100
      ? "Use all available"
      : `~${Math.round(unifiedGb * (gpuLimitPercent / 100))} GB of ${unifiedGb} GB (${gpuLimitPercent}%)`;

  return (
    <section className="page-frame settings-screen">
      <WorkPanel className="settings-work-panel">
        <div className="prompt-hero-top">
          <ModeTabs
            label="Settings section"
            options={tabOptions}
            mode={activeTab}
            onChange={setTab}
          />
        </div>

        {/* One status line under the tab row — the old full-width band above the panel is gone.
            It is a live region so a change committed by keyboard is announced, not just painted
            (sc-19711 accessibility parity). */}
        {status ? (
          <p aria-live="polite" className="settings-status" role="status">
            {status}
          </p>
        ) : null}

        {activeTab === "appearance" ? (
          <div className="settings-tab-body">
            <div className="settings-group-title">Theme</div>
            <div className="settings-row">
              <span className="settings-row-title">Mode</span>
              <div className="settings-segment" role="radiogroup" aria-label="Theme">
                <button
                  aria-checked={theme === "light"}
                  className={theme === "light" ? "active" : ""}
                  onClick={() => changeTheme("light")}
                  role="radio"
                  type="button"
                >
                  <Icon.Sun size={14} />
                  Light
                </button>
                <button
                  aria-checked={theme === "dark"}
                  className={theme === "dark" ? "active" : ""}
                  onClick={() => changeTheme("dark")}
                  role="radio"
                  type="button"
                >
                  <Icon.Moon size={14} />
                  Dark
                </button>
              </div>
            </div>
            <div className="settings-row settings-row--top">
              <div>
                <div className="settings-row-title">Accent</div>
                <div className="settings-row-sub">Recolors the whole app — chips, buttons, the mark.</div>
              </div>
              <div className="settings-accents" role="radiogroup" aria-label="Accent">
                {ACCENTS.map((entry) => (
                  <button
                    aria-checked={accent === entry.id}
                    aria-label={entry.name}
                    className={
                      accent === entry.id ? "settings-accent-swatch active" : "settings-accent-swatch"
                    }
                    key={entry.id}
                    onClick={() => onAccentChange?.(entry.id)}
                    role="radio"
                    style={{ background: entry.swatch }}
                    type="button"
                  />
                ))}
              </div>
            </div>

            <div className="work-panel-divider" />
            <div className="settings-group-title">Simple mode</div>
            <div className="settings-row">
              <div>
                <div className="settings-row-title">Use Simple mode by default</div>
                <div className="settings-row-sub">
                  {lockedToSimple
                    ? "Phones always use it."
                    : "Phones always use it; the sidebar switch overrides it for this session."}
                </div>
              </div>
              <button
                aria-checked={simpleDefault}
                aria-label="Use Simple mode by default"
                className={simpleDefault ? "settings-toggle on" : "settings-toggle"}
                onClick={() => onSimpleDefaultChange?.(!simpleDefault)}
                role="switch"
                type="button"
              >
                <span />
              </button>
            </div>

            <div className="work-panel-divider" />
            <div className="settings-group-title">Generation</div>
            <div>
              <div className="settings-row-title">Default quality</div>
              {/* SHORT labels (the tier keys) — the descriptive wording ("Auto (best that fits this
                  Mac)") doesn't fit a 34px chip, so it rides the title attribute and the line below. */}
              <div className="settings-chips" role="radiogroup" aria-label="Default generation quality">
                {GENERATION_QUALITY_OPTIONS.map((value) => (
                  <button
                    aria-checked={defaultQuality === value}
                    className={defaultQuality === value ? "settings-chip active" : "settings-chip"}
                    key={value}
                    onClick={() => changeDefaultQuality(value)}
                    role="radio"
                    title={generationQualityLabel(value)}
                    type="button"
                  >
                    {shortQualityLabel(value)}
                  </button>
                ))}
              </div>
              <div className="settings-row-sub">{generationQualityLabel(defaultQuality)}</div>
              <p className="settings-note settings-note--spaced">
                The tier new generations use for a model you haven’t picked one for yet.{" "}
                <strong>Auto</strong> takes the highest-fidelity tier that fits this machine for each
                model. Per-model picks in a studio always win, and are remembered.
              </p>
            </div>
          </div>
        ) : null}

        {activeTab === "settings" ? (
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
        ) : null}

        {activeTab === "device" ? (
          <div className="settings-tab-body">
            <div className="settings-group-title">This machine</div>
            <div className="settings-kv-list">
              <div className="settings-kv">
                <span>Engine</span>
                <span>{device.engine}</span>
              </div>
              <div className="settings-kv">
                <span>GPU</span>
                <span>{device.gpu}</span>
              </div>
              {macGpuMemory ? (
                <div className="settings-kv">
                  <span>Unified memory</span>
                  <span>
                    {unifiedGb} GB
                    {typeof gpu.wiredLimitMb === "number"
                      ? ` · system cap ${Math.round(gpu.wiredLimitMb / 1024)} GB`
                      : ""}
                  </span>
                </div>
              ) : null}
            </div>

            {macGpuMemory ? (
              <>
                <div className="work-panel-divider" />
                <div className="settings-group-title">GPU memory</div>
                <div>
                  <div className="settings-row settings-row--baseline">
                    <div className="settings-row-title">Memory target</div>
                    <div className="settings-readout">{gpuTargetLabel}</div>
                  </div>
                  <input
                    className="settings-range"
                    type="range"
                    min={GPU_LIMIT_MIN_PERCENT}
                    max={100}
                    step={5}
                    value={gpuLimitPercent}
                    aria-label="GPU memory target for SceneWorks (percent of unified memory)"
                    onChange={(event) => setGpuLimitPercent(Number(event.target.value))}
                    onMouseUp={(event) => commitGpuMemoryLimit(Number(event.target.value))}
                    onKeyUp={(event) => commitGpuMemoryLimit(Number(event.target.value))}
                    onTouchEnd={(event) => commitGpuMemoryLimit(Number(event.target.value))}
                  />
                  <div className="settings-scale">
                    <span>{GPU_LIMIT_MIN_PERCENT}%</span>
                    <span>Use all available</span>
                  </div>
                  <p className="settings-note settings-note--spaced">
                    A soft target for how much unified memory SceneWorks holds, so more stays free
                    for the rest of your system. Not a hard limit — large models can still exceed it.
                    Applies within a couple of seconds, no restart.{" "}
                    <code>sudo sysctl iogpu.wired_limit_mb=&lt;bytes&gt;</code> raises the system cap
                    on 96/128 GB Macs.
                  </p>
                  {gpuTelemetry ? (
                    <div className="settings-inset settings-inset--spaced">
                      <div className="settings-inset-title">Live MLX memory</div>
                      <div className="settings-kv">
                        <span>Active</span>
                        <span className="settings-mono">
                          {bytesToGb(gpuTelemetry.activeBytes).toFixed(1)} GB
                        </span>
                      </div>
                      <div className="settings-kv">
                        <span>Peak</span>
                        <span className="settings-mono">
                          {bytesToGb(gpuTelemetry.peakBytes).toFixed(1)} GB
                        </span>
                      </div>
                      {gpuTelemetry.limitBytes ? (
                        <div className="settings-kv">
                          <span>Limit</span>
                          <span className="settings-mono">
                            {Math.round(bytesToGb(gpuTelemetry.limitBytes))} GB
                          </span>
                        </div>
                      ) : null}
                    </div>
                  ) : null}
                </div>
              </>
            ) : null}

            {gpu?.platform === "windows" || gpu?.platform === "linux" ? (
              <p className="settings-note">
                Requires current NVIDIA drivers with CUDA support. The GPU memory target is
                macOS-only — a discrete GPU has no per-process memory cap, so generations use
                whatever VRAM they need.
              </p>
            ) : null}

            <div className="work-panel-divider" />
            <div className="settings-group-title">Maintenance</div>
            {/* Available in both modes: desktop restarts via Tauri, a remote admin via REST
                (epic 4484 story 12). */}
            <div className="settings-row">
              <div>
                <div className="settings-row-title">Inference worker</div>
                <div className="settings-row-sub">
                  Restart to pick up credential changes or clear a stuck job.
                </div>
              </div>
              <button className="settings-btn" type="button" onClick={restartWorker}>
                Restart
              </button>
            </div>
            {isDesktop ? (
              <div className="settings-row">
                <div>
                  <div className="settings-row-title">Setup wizard</div>
                  <div className="settings-row-sub">
                    Re-open the guided setup to download more models or create a project.
                  </div>
                </div>
                <button className="settings-btn" type="button" onClick={rerunSetupWizard}>
                  Re-run
                </button>
              </div>
            ) : null}
          </div>
        ) : null}

        {activeTab === "server" && serverTabAvailable ? (
          <div className="settings-tab-body">
            <div className="settings-group-title">Remote access (LAN)</div>
            <div className="settings-row">
              <div>
                <div className="settings-row-title">Let other devices on this network in</div>
                <div className="settings-row-sub">
                  Generation still runs on this computer. Takes effect after a restart.
                </div>
              </div>
              <button
                aria-checked={Boolean(remote.enabled)}
                aria-label="Remote access"
                className={remote.enabled ? "settings-toggle on" : "settings-toggle"}
                disabled={!remote.passwordSet && !remote.enabled}
                onClick={() => applyRemoteAccess(!remote.enabled)}
                role="switch"
                type="button"
              >
                <span />
              </button>
            </div>

            {remote.url ? (
              <>
                <div className="settings-field-row">
                  <span className="settings-url">{remote.url}</span>
                  <button
                    className="settings-btn settings-btn--compact"
                    type="button"
                    onClick={copyRemoteUrl}
                  >
                    <Icon.Duplicate size={14} />
                    Copy
                  </button>
                </div>
                {remote.lanCandidates && remote.lanCandidates.length > 1 ? (
                  <div className="settings-note">
                    Also reachable at {remote.lanCandidates.slice(1).join(", ")}
                  </div>
                ) : null}
              </>
            ) : (
              <p className="settings-note">
                Couldn’t determine this computer’s LAN address — check your network connection.
              </p>
            )}

            <div className="settings-inset">
              <div className="settings-inset-title">Access</div>
              <div className="settings-field-row">
                <input
                  className="settings-grow"
                  type="password"
                  placeholder={remote.passwordSet ? "Change password" : "Set a password"}
                  value={remotePassword}
                  onChange={(event) => setRemotePassword(event.target.value)}
                  aria-label="Remote access password"
                />
                <button
                  className="settings-btn"
                  type="button"
                  onClick={saveRemotePassword}
                  disabled={!remotePasswordMeetsMinimum(
                    remotePassword,
                    remote.minimumPasswordLength,
                  )}
                >
                  Save
                </button>
                {remote.passwordSet ? (
                  <button
                    className="settings-btn settings-btn--quiet"
                    type="button"
                    onClick={clearRemotePassword}
                  >
                    Clear
                  </button>
                ) : null}
              </div>
              <div className="settings-field-row">
                <span className="settings-note settings-grow">
                  Use at least {remote.minimumPasswordLength} characters.{" "}
                  {remote.passwordSet
                    ? "A password is set — remote browsers must enter it."
                    : "Set a password before enabling remote access."}
                </span>
                <label className="settings-note" htmlFor="remote-port">
                  Port
                </label>
                <input
                  className="settings-port"
                  id="remote-port"
                  type="number"
                  min="1024"
                  max="65535"
                  value={remotePort}
                  onChange={(event) => setRemotePort(event.target.value)}
                  aria-label="Remote access port"
                />
              </div>
            </div>

            {/* Security note + platform firewall guidance (story 11), consolidated into one notice. */}
            <div className="settings-notice">
              <Icon.Warning size={16} />
              <span>
                Trusted local networks only. Traffic is plain HTTP, so the password and your content
                travel unencrypted — never port-forward this or expose it to the internet.{" "}
                {remote.platform === "windows"
                  ? "The first time SceneWorks binds the port, allow it on Private networks in the Windows Security prompt."
                  : "The first time SceneWorks binds the port, macOS may ask to allow incoming connections — click Allow."}
              </span>
            </div>
          </div>
        ) : null}
      </WorkPanel>
    </section>
  );
}
