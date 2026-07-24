import React, { useMemo } from "react";
import { bundledLicenses } from "../data/bundledLicenses.js";
import { modelLicenseRows } from "./licenseTerms.js";

// Simple Licenses (design handoff): the app's own licence card, then one row per
// bundled MODEL licence with a commercial-use badge. Backed by the real corpus
// (apps/desktop/licenses/manifest.json via data/bundledLicenses.js) — the same source
// the advanced Licenses screen renders, which is also where the full licence TEXT lives.

const APP_LICENSE = "AGPL-3.0-or-later · © 2026 Michael Trefry & contributors. Free and open source.";

export function SimpleLicenses() {
  const rows = useMemo(() => modelLicenseRows(bundledLicenses), []);

  return (
    <div className="su-screen su-screen--tight">
      <div className="su-license-app">
        <strong>SceneWorks</strong>
        <p>{APP_LICENSE}</p>
      </div>

      <div className="su-group-title">Model licenses</div>

      {rows.map((row) => (
        <div className="su-license-row" key={row.id}>
          <span className="su-row-text">
            <span className="su-license-name">{row.name}</span>
            <span className="su-license-terms">{row.license || "—"}</span>
          </span>
          <span
            className={
              row.badge.tone === "danger" ? "su-license-badge su-license-badge--danger" : "su-license-badge"
            }
          >
            {row.badge.label}
          </span>
        </div>
      ))}

      <p className="su-empty">
        Full licence text for every bundled component — including the tooling binaries — is under
        Licenses in the Advanced workspace.
      </p>
    </div>
  );
}
