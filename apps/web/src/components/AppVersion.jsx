import React, { useCallback, useEffect, useRef, useState } from "react";
import { apiFetch } from "../api.js";
import { appConfirm } from "../appConfirm.jsx";
import { isDesktop, tauriInvoke } from "../runtime.js";
import { Modal } from "./Modal.jsx";

const APP_VERSION = import.meta.env.VITE_APP_VERSION ?? "";

// Re-read the global, uncapped queue: the visible jobs may be stale or project-filtered.
export async function prepareAppUpdate(token, confirmed = false, onCanceling = () => {}) {
  const queue = await apiFetch("/api/v1/queue", token);
  if (!Array.isArray(queue?.activeJobs)) throw new Error("Unable to check work in progress.");
  if (!queue.activeJobs.length) return { proceed: true, confirmed };
  if (!confirmed) {
    confirmed = await appConfirm({
      title: "Update SceneWorks",
      message: "There are work items in progress. Are you sure you want to update now?",
      confirmLabel: "Update now",
      cancelLabel: "Cancel",
      tone: "danger",
    });
    if (!confirmed) return { proceed: false, confirmed: false };
  }
  onCanceling();
  // Re-read after confirmation and wait for workers to acknowledge cancellation.
  // Never terminate the app while a cancellation request has merely been queued.
  const requested = new Set();
  const deadline = Date.now() + 60_000;
  while (true) {
    const current = await apiFetch("/api/v1/queue", token);
    if (!Array.isArray(current?.activeJobs)) throw new Error("Unable to check work in progress.");
    if (!current.activeJobs.length) break;
    if (Date.now() >= deadline) throw new Error("Work is still stopping. Please try updating again once it has stopped.");
    await Promise.all(current.activeJobs.filter((job) => !requested.has(job.id)).map(async (job) => {
      await apiFetch(`/api/v1/jobs/${encodeURIComponent(job.id)}/cancel`, token, { method: "POST" });
      requested.add(job.id);
    }));
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  return { proceed: true, confirmed };
}

// Owned by App so switching between Simple and Advanced cannot start two installs.
export function useAppUpdate(token, ready = true) {
  const [version, setVersion] = useState(null);
  const [phase, setPhase] = useState("");
  const [error, setError] = useState("");
  const busy = useRef(false);

  const install = useCallback(async () => {
    if (busy.current) return;
    busy.current = true;
    setError("");
    setPhase("Checking work…");
    let downloaded = false;
    try {
      const first = await prepareAppUpdate(token, false, () => setPhase("Canceling work…"));
      if (!first.proceed) return;
      setPhase("Downloading update…");
      await tauriInvoke("download_app_update");
      downloaded = true;
      // Work may have started during the download, including from another client.
      setPhase("Checking work…");
      const final = await prepareAppUpdate(token, first.confirmed, () => setPhase("Canceling work…"));
      if (!final.proceed) return;
      setPhase("Installing update…");
      await tauriInvoke("install_app_update");
    } catch (err) {
      setError(`Update failed: ${err?.message ?? String(err)}`);
    } finally {
      if (downloaded) await tauriInvoke("discard_app_update").catch(() => {});
      busy.current = false;
      setPhase("");
    }
  }, [token]);

  useEffect(() => {
    if (!isDesktop) return;
    let stopped = false;
    let timer;
    let unlisten;
    const read = async () => {
      try {
        const status = await tauriInvoke("get_update_status", { ready: ready && !busy.current });
        if (!stopped) {
          setVersion(status.version);
          if (status.startupRequested) await install();
        }
      } catch { /* Availability checks are fail-soft; installation errors are visible. */ }
    };
    const poll = async () => {
      await read();
      if (!stopped) timer = setTimeout(poll, 15_000);
    };
    // Events make startup acceptance immediate; polling also recovers missed events.
    window.__TAURI__?.event?.listen("app-update-changed", read).then((off) => {
      if (stopped) off();
      else unlisten = off;
    }).catch(() => {});
    poll();
    return () => { stopped = true; clearTimeout(timer); unlisten?.(); };
  }, [install, ready]);
  return { version, phase, error, install };
}

export function AppVersion({ update }) {
  return (
    <div className="app-version-section">
      {APP_VERSION ? <span className="app-version" title={`SceneWorks ${APP_VERSION}`}>Version: {APP_VERSION}</span> : null}
      {update.version ? (
        <button className="app-update-link" type="button" disabled={Boolean(update.phase)} onClick={update.install}>
          Update to {update.version}
        </button>
      ) : null}
      {update.error ? <span className="app-update-error" role="alert">{update.error}</span> : null}
      {update.phase && update.phase !== "Checking work…" ? (
        <Modal label="Updating SceneWorks" onClose={() => {}}>
          <p role="status">{update.phase}</p>
          <p>SceneWorks will restart when the update is installed.</p>
        </Modal>
      ) : null}
    </div>
  );
}
