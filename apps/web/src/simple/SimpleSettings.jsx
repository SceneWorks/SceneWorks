import React, { useEffect, useMemo, useState } from "react";
import { Icon } from "../components/Icons.jsx";
import { apiFetch } from "../api.js";
import { useAppContext } from "../context/AppContext.js";
import { ACCENTS } from "../accents.js";
import {
  AUTO_GENERATION_QUALITY,
  GENERATION_QUALITY_OPTIONS,
  generationQualityLabel,
  readDefaultGenerationQuality,
  writeDefaultGenerationQuality,
} from "../generationQuality.js";
import { loadCredentials, saveCredential } from "../credentials.js";
import { Chips } from "./studioParts.jsx";
import { useSimpleUi } from "./SimpleUiContext.js";

// Simple Settings (design handoff): Appearance (theme + accent), Generation (default
// quality + the Simple-UI default), Service credentials, Device.
//
// Every control writes to the SAME durable store the advanced Settings screen uses —
// theme/accent and the default quality through `PUT /api/v1/ui-preferences`, credentials
// through the shared keychain/REST transport — so a change made here is the same change
// made there. Nothing in this screen is Simple-only state.

export function SimpleSettings({ accent, onAccentChange, simpleDefault, onSimpleDefaultChange, lockedToSimple }) {
  const { theme, changeTheme, visibleWorkers = [], macCapabilities } = useAppContext();
  const { toast } = useSimpleUi();
  const [quality, setQuality] = useState(readDefaultGenerationQuality);
  const [credentials, setCredentials] = useState([]);
  const [host, setHost] = useState("");
  const [token, setToken] = useState("");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let cancelled = false;
    loadCredentials()
      .then((items) => {
        if (!cancelled) {
          setCredentials(items ?? []);
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  function changeQuality(next) {
    const value = writeDefaultGenerationQuality(next);
    setQuality(value);
    // ui-preferences is a public route (like /health) and MERGES partial updates, so
    // sending only this field can't disturb theme/accent — same contract as App.jsx.
    apiFetch("/api/v1/ui-preferences", "", {
      method: "PUT",
      body: JSON.stringify({ defaultGenerationQuality: value }),
    }).catch(() => {});
    toast(`Default quality set to ${generationQualityLabel(value)}`);
  }

  async function addCredential() {
    const trimmedHost = host.trim();
    if (!trimmedHost || !token.trim() || saving) {
      return;
    }
    setSaving(true);
    try {
      const updated = await saveCredential({
        host: trimmedHost,
        label: trimmedHost,
        scheme: "bearer",
        token: token.trim(),
      });
      setCredentials(updated ?? credentials);
      setHost("");
      setToken("");
      toast(`Credential saved for ${trimmedHost}`);
    } catch (error) {
      toast(String(error?.message ?? error));
    } finally {
      setSaving(false);
    }
  }

  // Engine / GPU are read off the live worker registry — the one signal available in
  // every deployment (desktop, Docker, remote browser).
  const device = useMemo(() => describeDevice(visibleWorkers, macCapabilities), [visibleWorkers, macCapabilities]);

  return (
    <div className="su-screen su-screen--loose">
      <div className="su-group">
        <div className="su-group-title">Appearance</div>
        <div className="su-card su-card--split">
          <span className="su-card-title">Theme</span>
          <div className="su-theme-toggle" role="radiogroup" aria-label="Theme">
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
        <div className="su-card">
          <div className="su-card-title">Accent</div>
          <div className="su-accents" role="radiogroup" aria-label="Accent">
            {ACCENTS.map((entry) => (
              <button
                aria-checked={accent === entry.id}
                aria-label={entry.name}
                className={accent === entry.id ? "su-accent-swatch active" : "su-accent-swatch"}
                key={entry.id}
                onClick={() => onAccentChange(entry.id)}
                role="radio"
                style={{ background: entry.swatch }}
                type="button"
              />
            ))}
          </div>
        </div>
      </div>

      <div className="su-group">
        <div className="su-group-title">Generation</div>
        <div className="su-card">
          <div className="su-card-title">Default quality</div>
          {/* Auto leads the row: it is the app-wide default (the highest-fidelity tier that
              fits this machine), and omitting it would strand a Simple user on a pinned tier
              with no way back. The three explicit tiers are the design's.
              SHORT labels (the tier keys) — the advanced Settings screen's descriptive ones
              ("Auto (best that fits this Mac)") don't fit a 34px chip. The full wording rides
              the title attribute, and the selected tier is spelled out underneath. */}
          <Chips
            ariaLabel="Default quality"
            capped
            onChange={changeQuality}
            options={GENERATION_QUALITY_OPTIONS.map((value) => ({
              value,
              label: shortQualityLabel(value),
              title: generationQualityLabel(value),
            }))}
            value={quality}
          />
          <div className="su-card-sub">{generationQualityLabel(quality)}</div>
        </div>
        <div className="su-card su-card--split">
          <div>
            <div className="su-card-title">Use Simple UI by default</div>
            <div className="su-card-sub">
              {lockedToSimple ? "Phones always use it." : "Phones always use it; the sidebar switch overrides it for this session."}
            </div>
          </div>
          <button
            aria-checked={simpleDefault}
            aria-label="Use Simple UI by default"
            className={simpleDefault ? "su-toggle on" : "su-toggle"}
            onClick={() => onSimpleDefaultChange(!simpleDefault)}
            role="switch"
            type="button"
          >
            <span />
          </button>
        </div>
      </div>

      <div className="su-group">
        <div className="su-group-title">Service credentials</div>
        <div className="su-card">
          <div className="su-card-body">
            <p className="su-card-note">
              Add a token for gated downloads (e.g. huggingface.co). Tokens are write-only — they are
              never shown again after saving.
            </p>
            {credentials.length ? (
              <p className="su-card-note">
                Saved for {credentials.map((credential) => credential.host).join(", ")}.
              </p>
            ) : null}
            <div className="su-inline-form">
              <input
                aria-label="Credential host"
                className="su-input"
                onChange={(event) => setHost(event.target.value)}
                placeholder="huggingface.co"
                value={host}
              />
            </div>
            <div className="su-inline-form">
              <input
                aria-label="Access token"
                className="su-input"
                onChange={(event) => setToken(event.target.value)}
                placeholder="Access token"
                type="password"
                value={token}
              />
              <button
                className="su-accent-btn"
                disabled={!host.trim() || !token.trim() || saving}
                onClick={addCredential}
                type="button"
              >
                {saving ? "Saving…" : "Add"}
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="su-group">
        <div className="su-group-title">Device</div>
        <div className="su-card">
          <div className="su-card-body">
            <div className="su-kv">
              <span>Engine</span>
              <span>{device.engine}</span>
            </div>
            <div className="su-kv">
              <span>GPU</span>
              <span>{device.gpu}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Chip-sized label for a quality value — the design's [bf16, Q8, Q4] plus Auto. Derived
// from the value rather than a lookup table, so a tier added to the vocabulary still gets a
// sensible chip; the full descriptive wording rides the chip's title and the line beneath.
export function shortQualityLabel(value) {
  if (value === AUTO_GENERATION_QUALITY) {
    return "Auto";
  }
  // bf16 is written lower-case everywhere in the product; the quant tiers are upper-case.
  return value === "bf16" ? "bf16" : String(value).toUpperCase();
}

// Engine + GPU summary from the live worker registry. An MLX worker means Apple Silicon
// unified memory; anything else is the discrete-GPU (candle) lane. Unknown values read
// "Detecting…" rather than a fabricated placeholder.
export function describeDevice(workers, macCapabilities) {
  const gpuWorkers = (workers ?? []).filter((worker) => worker?.gpuId && worker.gpuId !== "cpu");
  const mlx = gpuWorkers.find((worker) => worker.gpuId === "mlx");
  const primary = mlx ?? gpuWorkers[0] ?? null;
  const engine = mlx
    ? "MLX · Apple Silicon"
    : primary
      ? "Candle · discrete GPU"
      : macCapabilities?.platform === "macos"
        ? "MLX · Apple Silicon"
        : "Detecting…";
  if (!primary) {
    return { engine, gpu: "No worker connected" };
  }
  const totalMb = Number(primary.utilization?.memoryTotalMb);
  const memory = Number.isFinite(totalMb) && totalMb > 0 ? ` · ${(totalMb / 1024).toFixed(0)} GB` : "";
  return { engine, gpu: `${primary.gpuName ?? `GPU ${primary.gpuId}`}${memory}` };
}
