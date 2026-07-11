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
        // Show installed overlays (studio-trained/registered, or a cached hosted one) AND hosted
        // overlays not yet on disk (the built-in beta) — the latter lazy-download on first use, so they
        // are selectable. A studio-trained overlay that is merely "missing" (no HF repo) is hidden: the
        // API would 400 it. sc-8466.
        setOverlays(
          Array.isArray(items)
            ? items.filter(
                (overlay) =>
                  overlay.installState === "installed" ||
                  overlay.source?.provider === "huggingface",
              )
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

  // A built-in (hosted) overlay ships with the app for this backbone, so leaving the picker unselected
  // still gets pose control (the worker defaults to it); the picker is for choosing a specific overlay.
  const hasBuiltin = overlays.some((overlay) => overlay.scope === "builtin");

  return (
    <div className="control-overlay-field">
      <label htmlFor="control-overlay-select">ControlNet overlay</label>
      {status === "loading" ? (
        <p className="muted">Loading overlays…</p>
      ) : status === "error" ? (
        <p className="muted">Couldn&apos;t load control overlays.</p>
      ) : overlays.length ? (
        <>
          <select
            id="control-overlay-select"
            onChange={(event) => onOverlayChange?.(event.target.value || null)}
            value={selectedOverlayId ?? ""}
          >
            <option value="">
              {hasBuiltin ? "Built-in beta overlay (default)" : "Select a trained overlay…"}
            </option>
            {overlays.map((overlay) => (
              <option key={overlay.id} value={overlay.id}>
                {(overlay.name ?? overlay.id) +
                  (overlay.installState === "installed" ? "" : " (downloads on first use)")}
              </option>
            ))}
          </select>
          {hasBuiltin ? (
            <p className="muted">
              Leave unselected to use the built-in beta pose overlay (an experimental feasibility
              spike) — or train your own in the Training Studio for a custom one.
            </p>
          ) : null}
        </>
      ) : (
        <p className="muted">
          No pose ControlNet installed for this model yet — train one in the Training Studio to
          enable pose control.
        </p>
      )}
    </div>
  );
}
