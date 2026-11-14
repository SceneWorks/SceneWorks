import { systemApi } from "../api/systemResources.js";
import { AppearanceTab } from "./settings/AppearanceTab.jsx";
import { SettingsTab } from "./settings/SettingsTab.jsx";
import { DeviceTab } from "./settings/DeviceTab.jsx";
import { ServerTab } from "./settings/ServerTab.jsx";
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { loadCredentials, removeCredentialRequest, saveCredential } from "../credentials.js";
import { putUiPreferences } from "../uiPreferences.js";
import { isDesktop, tauriInvoke as invoke } from "../runtime.js";
import { generationQualityLabel, readDefaultGenerationQuality, writeDefaultGenerationQuality } from "../generationQuality.js";
import { writeClipboardText } from "../clipboard.js";
import { appConfirm } from "../appConfirm.jsx";
import { formatBytes } from "../formatting.js";
import { bytesToGib, daysToSeconds, describeDisableConsequence, describeLimitConsequence, gibToBytes, secondsToDays } from "../modelCache.js";
import { useModelCacheStatus } from "../hooks/useModelCacheStatus.js";
import { WorkPanel } from "../components/WorkPanel.jsx";

import { ModeTabs } from "../components/generationStudio.jsx";


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
// The slider's "Use all available" end: the shell's no-target state, not a memory amount.
const GPU_LIMIT_OFF_PERCENT = 100;
const APP_VERSION = import.meta.env.VITE_APP_VERSION ?? "";

// Settings → Updates (sc-8663): an on-demand version of the shell's startup/hourly update check.
// The shell caches any offer and emits `app-update-changed`, so App's `useAppUpdate` hook (passed
// in as `update`) picks it up and this group, like the sidebar, installs through that one flow —
// there is never a second download/install path to keep in step.
function DesktopUpdatesGroup({ update }) {
  const [checking, setChecking] = useState(false);
  const [result, setResult] = useState("");
  const check = async () => {
    setChecking(true);
    setResult("");
    try {
      const version = await invoke("check_for_app_update");
      setResult(version ? `SceneWorks ${version} is available.` : "You're up to date.");
    } catch (error) {
      setResult(`Couldn't check for updates: ${error?.message ?? String(error)}`);
    } finally {
      setChecking(false);
    }
  };
  const busy = checking || Boolean(update?.phase);
  return (
    <>
      <div className="settings-group-title">Updates</div>
      <div className="settings-row">
        <div>
          <div className="settings-row-title">SceneWorks {APP_VERSION || "desktop"}</div>
          <div className="settings-row-sub">
            SceneWorks checks for updates when it starts and every hour while it runs.
          </div>
        </div>
        <div className="settings-button-row">
          <button className="settings-btn" disabled={busy} onClick={check} type="button">
            {checking ? "Checking…" : "Check for updates"}
          </button>
          {update?.version ? (
            <button className="settings-btn" disabled={busy} onClick={update.install} type="button">
              Update to {update.version}
            </button>
          ) : null}
        </div>
      </div>
      {result ? (
        <p aria-live="polite" className="settings-note" role="status">
          {result}
        </p>
      ) : null}
      <div className="work-panel-divider" />
    </>
  );
}

// The slider position for the shell's served GPU memory target (SW-7032). The shell always serves
// `gpuMemoryLimitFraction`: `null` is its own "no target" state, drawn at the slider's "Use all
// available" end. Before settings have loaded (or from a shell that serves no key) there is no
// position: the slider is disabled and empty, and the web never invents a cap to save.
function gpuLimitPercentFromSettings(settings) {
  if (!settings || !Object.hasOwn(settings, "gpuMemoryLimitFraction")) return null;
  const fraction = settings.gpuMemoryLimitFraction;
  if (fraction === null) return GPU_LIMIT_OFF_PERCENT;
  if (typeof fraction !== "number" || !Number.isFinite(fraction)) return null;
  return fraction >= 1
    ? GPU_LIMIT_OFF_PERCENT
    : Math.min(GPU_LIMIT_OFF_PERCENT, Math.max(GPU_LIMIT_MIN_PERCENT, Math.round(fraction * 100)));
}

// GPU memory telemetry (epic 7819, sc-7825). The worker publishes byte counts;
// preserve a reported zero while distinguishing an absent reading from zero.
function telemetryBytesToGib(bytes) {
  if (typeof bytes !== "number" || !Number.isFinite(bytes) || bytes < 0) return null;
  return bytesToGib(bytes);
}

function telemetryGibLabel(bytes, digits = 1) {
  const gib = telemetryBytesToGib(bytes);
  return gib === null ? "Unavailable" : `${gib.toFixed(digits)} GiB`;
}

const REMOTE_PASSWORD_POLICY_V1 = "unicode-white-space-scalar-count-v1";

// Exact Unicode White_Space code-point set named by the native V1 policy. Do
// not use JavaScript trim(): it excludes U+0085 and includes U+FEFF, unlike this
// contract and Rust's corresponding explicit predicate.
function isRemotePasswordBoundaryWhitespaceV1(character) {
  const value = character.codePointAt(0);
  return (
    (value >= 0x0009 && value <= 0x000d) ||
    value === 0x0020 ||
    value === 0x0085 ||
    value === 0x00a0 ||
    value === 0x1680 ||
    (value >= 0x2000 && value <= 0x200a) ||
    (value >= 0x2028 && value <= 0x2029) ||
    value === 0x202f ||
    value === 0x205f ||
    value === 0x3000
  );
}

function normalizeRemotePassword(password, policy) {
  if (policy !== REMOTE_PASSWORD_POLICY_V1) return null;
  const scalars = Array.from(password);
  // Array.from combines valid surrogate pairs, but preserves an unpaired UTF-16
  // surrogate as one element. Rust strings cannot contain those non-scalars, so
  // fail closed instead of counting a value native validation cannot receive.
  if (scalars.some((character) => {
    const value = character.codePointAt(0);
    return value >= 0xd800 && value <= 0xdfff;
  })) return null;
  let start = 0;
  let end = scalars.length;
  while (start < end && isRemotePasswordBoundaryWhitespaceV1(scalars[start])) start += 1;
  while (end > start && isRemotePasswordBoundaryWhitespaceV1(scalars[end - 1])) end -= 1;
  return scalars.slice(start, end).join("");
}

function remotePasswordCandidate(password, remoteAccessStatus) {
  const normalized = normalizeRemotePassword(password, remoteAccessStatus?.passwordPolicy);
  const minimumPasswordLength = remoteAccessStatus?.minimumPasswordLength;
  if (
    normalized === null ||
    !Number.isSafeInteger(minimumPasswordLength) ||
    minimumPasswordLength < 1
  ) {
    return null;
  }
  return {
    normalized,
    meetsMinimum: Array.from(normalized).length >= minimumPasswordLength,
  };
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
  // App's `useAppUpdate` state (sc-8663): the cached offer + the one install flow it owns.
  appUpdate = null,
}) {
  // theme/changeTheme and the worker registry come from the app context (the same values the
  // topbar toggle and Simple Settings read); accent + the Simple-mode default are drilled from
  // App.jsx, which owns them alongside the sidebar's mode switch.
  const { token = "", theme, changeTheme, visibleWorkers = [], macCapabilities } = useAppContext();
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
  const [gpuLimitPercent, setGpuLimitPercent] = useState(null);
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
  const { cache, error: cacheError, refresh: refreshCache } = useModelCacheStatus(token);
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
    setGpuLimitPercent(gpuLimitPercentFromSettings(settings));
  }, [settings]);

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
        await systemApi.workerRestart(token, { method: "POST" });
      }
      setStatus("Restarting the inference worker…");
    } catch (error) {
      setStatus(String(error));
    }
  }

  async function commitGpuMemoryLimit(percent) {
    // Nothing to save until the shell's served target has loaded.
    if (gpuLimitPercent === null || !Number.isFinite(percent)) return;
    try {
      // 100% = Off → send null so the shell clears it; otherwise a fraction (0.1–0.99).
      const fraction = percent >= GPU_LIMIT_OFF_PERCENT ? null : percent / 100;
      const updated = await invoke("set_gpu_memory_limit", { fraction });
      setSettings(updated);
      // Native jobs read the atomic handoff between runs; an active job retains
      // its admitted configuration. Other CUDA jobs do not gain a global quota.
      setStatus(
        fraction == null
          ? "GPU memory target turned off for new jobs — no restart needed."
          : `GPU memory target set to ${percent}% for new jobs — no restart needed.`,
      );
    } catch (error) {
      setGpuLimitPercent(gpuLimitPercentFromSettings(settings));
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
    const candidate = remotePasswordCandidate(remotePassword, remote);
    if (!candidate) {
      setStatus("Remote-access password policy is unavailable. Restart SceneWorks and try again.");
      return;
    }
    if (!candidate.meetsMinimum) {
      setStatus(
        `Use a remote-access password with at least ${remote.minimumPasswordLength} characters after normalizing surrounding whitespace.`,
      );
      return;
    }
    try {
      const updated = await invoke("set_remote_access_password", {
        password: candidate.normalized,
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
  const cudaGpuMemory = ["windows", "linux"].includes(gpu?.platform) && Number(gpu?.gpuMemoryMb) > 0;
  const unifiedGb = macGpuMemory ? Math.round(gpu.unifiedMemoryMb / 1024) : 0;
  const gpuTargetLabel =
    gpuLimitPercent === null
      ? "Loading…"
      : gpuLimitPercent >= GPU_LIMIT_OFF_PERCENT
      ? "Use all available"
      : macGpuMemory
        ? `~${Math.round(unifiedGb * (gpuLimitPercent / 100))} GiB of ${unifiedGb} GiB (${gpuLimitPercent}%)`
        : `${gpuLimitPercent}% of each selected GPU's VRAM`;

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
          <AppearanceTab
            accent={accent}
            changeDefaultQuality={changeDefaultQuality}
            changeTheme={changeTheme}
            defaultQuality={defaultQuality}
            lockedToSimple={lockedToSimple}
            onAccentChange={onAccentChange}
            onSimpleDefaultChange={onSimpleDefaultChange}
            simpleDefault={simpleDefault}
            theme={theme}
          />
        ) : null}

        {activeTab === "settings" ? (
          <SettingsTab
            DesktopUpdatesGroup={DesktopUpdatesGroup}
            SCHEME_OPTIONS={SCHEME_OPTIONS}
            addCredential={addCredential}
            appUpdate={appUpdate}
            cache={cache}
            cacheDays={cacheDays}
            cacheError={cacheError}
            cacheLimitGb={cacheLimitGb}
            canSaveCredential={canSaveCredential}
            changeCacheEnabled={changeCacheEnabled}
            changeDataDir={changeDataDir}
            changeModelLibrary={changeModelLibrary}
            commitCacheInterval={commitCacheInterval}
            commitCacheLimit={commitCacheLimit}
            credentialLocation={credentialLocation}
            credentials={credentials}
            dataDirLabel={dataDirLabel}
            embedWorkflow={embedWorkflow}
            gpu={gpu}
            modelLibraryFromEnvironment={modelLibraryFromEnvironment}
            modelLibraryNotice={modelLibraryNotice}
            modelLibraryPath={modelLibraryPath}
            newHost={newHost}
            newLabel={newLabel}
            newScheme={newScheme}
            newToken={newToken}
            onChangeModelLibrary={onChangeModelLibrary}
            onEmbedWorkflowChange={onEmbedWorkflowChange}
            persistedPolicy={persistedPolicy}
            removeCredential={removeCredential}
            revealDataDir={revealDataDir}
            revealModelLibrary={revealModelLibrary}
            setCacheDays={setCacheDays}
            setCacheLimitGb={setCacheLimitGb}
            setNewHost={setNewHost}
            setNewLabel={setNewLabel}
            setNewScheme={setNewScheme}
            setNewToken={setNewToken}
            settings={settings}
            sharingHeadingRef={sharingHeadingRef}
          />
        ) : null}

        {activeTab === "device" ? (
          <DeviceTab
            GPU_LIMIT_MIN_PERCENT={GPU_LIMIT_MIN_PERCENT}
            commitGpuMemoryLimit={commitGpuMemoryLimit}
            cudaGpuMemory={cudaGpuMemory}
            device={device}
            gpu={gpu}
            gpuLimitPercent={gpuLimitPercent}
            gpuTargetLabel={gpuTargetLabel}
            gpuTelemetry={gpuTelemetry}
            macGpuMemory={macGpuMemory}
            rerunSetupWizard={rerunSetupWizard}
            restartWorker={restartWorker}
            setGpuLimitPercent={setGpuLimitPercent}
            telemetryGibLabel={telemetryGibLabel}
            unifiedGb={unifiedGb}
          />
        ) : null}

        {activeTab === "server" && serverTabAvailable ? (
          <ServerTab
            applyRemoteAccess={applyRemoteAccess}
            clearRemotePassword={clearRemotePassword}
            copyRemoteUrl={copyRemoteUrl}
            remote={remote}
            remotePassword={remotePassword}
            remotePasswordCandidate={remotePasswordCandidate}
            remotePort={remotePort}
            saveRemotePassword={saveRemotePassword}
            setRemotePassword={setRemotePassword}
            setRemotePort={setRemotePort}
          />
        ) : null}
      </WorkPanel>
    </section>
  );
}
