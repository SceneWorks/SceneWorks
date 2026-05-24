import React, { useCallback, useEffect, useState } from "react";

// Desktop-only: these settings are backed by Tauri commands in the shell.
const isDesktop = typeof window !== "undefined" && !!window.__TAURI__;
const invoke = (command, args) => window.__TAURI__.core.invoke(command, args);

export function SettingsScreen() {
  const [settings, setSettings] = useState(null);
  const [gpu, setGpu] = useState(null);
  const [tokenPresent, setTokenPresent] = useState(false);
  const [tokenInput, setTokenInput] = useState("");
  const [status, setStatus] = useState("");

  const refresh = useCallback(async () => {
    if (!isDesktop) {
      return;
    }
    try {
      const [loadedSettings, gpuInfo, present] = await Promise.all([
        invoke("get_app_settings"),
        invoke("get_gpu_info"),
        invoke("hf_token_present"),
      ]);
      setSettings(loadedSettings);
      setGpu(gpuInfo);
      setTokenPresent(present);
    } catch (error) {
      setStatus(String(error));
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  if (!isDesktop) {
    return (
      <div className="settings-screen">
        <p className="settings-muted">
          Settings are managed by the SceneWorks desktop app.
        </p>
      </div>
    );
  }

  const secretStore = gpu?.platform === "windows" ? "Credential Manager" : "Keychain";
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

  async function saveToken() {
    try {
      await invoke("set_hf_token", { token: tokenInput });
      setTokenInput("");
      await refresh();
      setStatus(`Hugging Face token saved to the ${secretStore}.`);
    } catch (error) {
      setStatus(String(error));
    }
  }

  async function restartWorker() {
    try {
      await invoke("restart_worker");
      setStatus("Restarting the inference worker…");
    } catch (error) {
      setStatus(String(error));
    }
  }

  async function rerunSetupWizard() {
    try {
      await invoke("reset_setup");
      window.location.reload();
    } catch (error) {
      setStatus(String(error));
    }
  }

  return (
    <div className="settings-screen">
      {status ? <p className="settings-status">{status}</p> : null}

      <section className="settings-card">
        <h3>Data directory</h3>
        <p className="settings-value">{dataDirLabel}</p>
        <div className="settings-actions">
          <button type="button" onClick={changeDataDir}>
            Change…
          </button>
          <button type="button" onClick={revealDataDir} disabled={!settings?.dataDir}>
            Reveal in {gpu?.platform === "windows" ? "Explorer" : "Finder"}
          </button>
        </div>
      </section>

      <section className="settings-card">
        <h3>Hugging Face token</h3>
        <p className="settings-muted">
          Stored in the system {secretStore}. {tokenPresent ? "A token is currently set." : "No token set."}
        </p>
        <div className="settings-actions">
          <input
            type="password"
            placeholder="hf_…"
            value={tokenInput}
            onChange={(event) => setTokenInput(event.target.value)}
            aria-label="Hugging Face token"
          />
          <button type="button" onClick={saveToken}>
            {tokenInput.trim() ? "Save" : "Clear"}
          </button>
        </div>
      </section>

      <section className="settings-card">
        <h3>Detected GPU</h3>
        {gpu?.devices?.length ? (
          <ul className="settings-list">
            {gpu.devices.map((device) => (
              <li key={device}>{device}</li>
            ))}
          </ul>
        ) : (
          <p className="settings-muted">No accelerated GPU detected.</p>
        )}
        {gpu?.unifiedMemoryMb ? (
          <p className="settings-muted">
            Unified memory: {Math.round(gpu.unifiedMemoryMb / 1024)} GB
            {typeof gpu.wiredLimitMb === "number"
              ? ` · GPU cap: ${Math.round(gpu.wiredLimitMb / 1024)} GB`
              : ""}
          </p>
        ) : null}
        {gpu?.platform === "macos" ? (
          <p className="settings-help">
            On 96/128 GB Macs you can raise the GPU memory cap:{" "}
            <code>sudo sysctl iogpu.wired_limit_mb=&lt;bytes&gt;</code>
          </p>
        ) : null}
        {gpu?.platform === "windows" ? (
          <p className="settings-help">
            Requires current NVIDIA drivers with CUDA support.
          </p>
        ) : null}
      </section>

      <section className="settings-card">
        <h3>Inference worker</h3>
        <div className="settings-actions">
          <button type="button" onClick={restartWorker}>
            Restart worker
          </button>
        </div>
      </section>

      <section className="settings-card">
        <h3>Setup wizard</h3>
        <p className="settings-muted">
          Re-open the guided setup to download more models or create another project.
        </p>
        <div className="settings-actions">
          <button type="button" onClick={rerunSetupWizard}>
            Re-run setup wizard
          </button>
        </div>
      </section>
    </div>
  );
}
