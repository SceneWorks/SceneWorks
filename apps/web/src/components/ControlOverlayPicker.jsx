import React, { useEffect, useState } from "react";
import { apiFetch, isAbortError } from "../api.js";
import { useAppContext } from "../context/AppContext.js";

// Trained ControlNet overlay picker (sc-10165 B4). Self-contained: fetches the registered overlays for
// the given backbone (`GET /api/v1/control-overlays?baseModel=…`, project-scoped) and lets the user pick
// one. The chosen id flows out via `onOverlayChange` → `advanced.controlWeights.overlayId`, which the API
// (`resolve_control_overlay_selection`) resolves to the overlay's `.safetensors` for the worker
// strict-control lane. Rendered only for backbones whose pose control rides a registered overlay
// (e.g. Krea 2 Turbo — the Fun-Union backbones carry built-in control weights and never show this).
export function ControlOverlayPicker({ baseModel, selectedOverlayId, onOverlayChange }) {
  const { token, activeProject } = useAppContext();
  const [overlays, setOverlays] = useState([]);
  const [status, setStatus] = useState("loading"); // loading | ready | error

  useEffect(() => {
    if (!baseModel) {
      return undefined;
    }
    const controller = new AbortController();
    setStatus("loading");
    const params = new URLSearchParams({ baseModel });
    if (activeProject?.id) {
      params.set("projectId", activeProject.id);
    }
    apiFetch(`/api/v1/control-overlays?${params.toString()}`, token, {
      signal: controller.signal,
    })
      .then((items) => {
        setOverlays(
          Array.isArray(items)
            ? items.filter((overlay) => overlay.installState === "installed")
            : [],
        );
        setStatus("ready");
      })
      .catch((err) => {
        if (isAbortError(err)) {
          return;
        }
        setStatus("error");
      });
    return () => controller.abort();
  }, [baseModel, activeProject?.id, token]);

  return (
    <div className="control-overlay-field">
      <label htmlFor="control-overlay-select">ControlNet overlay</label>
      {status === "loading" ? (
        <p className="muted">Loading overlays…</p>
      ) : status === "error" ? (
        <p className="muted">Couldn&apos;t load control overlays.</p>
      ) : overlays.length ? (
        <select
          id="control-overlay-select"
          onChange={(event) => onOverlayChange?.(event.target.value || null)}
          value={selectedOverlayId ?? ""}
        >
          <option value="">Select a trained overlay…</option>
          {overlays.map((overlay) => (
            <option key={overlay.id} value={overlay.id}>
              {overlay.name ?? overlay.id}
            </option>
          ))}
        </select>
      ) : (
        <p className="muted">
          No pose ControlNet installed for this model yet — train one in the Training Studio to
          enable pose control.
        </p>
      )}
    </div>
  );
}
