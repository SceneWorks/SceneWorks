import React, { useMemo, useState } from "react";

import { bundledLicenses, licensesIntro } from "../data/bundledLicenses.js";
import { safeExternalUrl } from "../urls.js";

// About → Licenses (sc-3778). Aggregates the third-party components SceneWorks
// redistributes in the desktop bundle (ffmpeg GPLv3, onnxruntime MIT, …) and
// exposes their full license text plus any written offers, so the notices are
// reachable from inside the app. The corpus is imported from the tracked
// apps/desktop/licenses/ source of truth (see data/bundledLicenses.js), so this
// screen renders identically on desktop, web and Docker with no backend call.
export function LicensesScreen() {
  const components = bundledLicenses;
  const [selectedId, setSelectedId] = useState(components[0]?.id ?? null);
  const [docIndex, setDocIndex] = useState(0);

  const selected = useMemo(
    () => components.find((component) => component.id === selectedId) ?? components[0] ?? null,
    [components, selectedId],
  );

  const selectComponent = (id) => {
    setSelectedId(id);
    setDocIndex(0);
  };

  if (!selected) {
    return (
      <section className="page-frame licenses-screen">
        <p className="licenses-empty">No bundled third-party components are recorded.</p>
      </section>
    );
  }

  const activeDoc = selected.documents[docIndex] ?? selected.documents[0];

  return (
    <section className="page-frame licenses-screen">
      {licensesIntro ? <p className="licenses-intro">{licensesIntro}</p> : null}

      <div className="licenses-layout">
        <nav className="licenses-list" aria-label="Bundled components">
          {components.map((component) => (
            <button
              key={component.id}
              type="button"
              className={component.id === selected.id ? "licenses-item active" : "licenses-item"}
              aria-current={component.id === selected.id}
              onClick={() => selectComponent(component.id)}
            >
              <span className="licenses-item-name">{component.name}</span>
              <span className="licenses-item-meta">
                {component.license}
                {component.version ? ` · v${component.version}` : ""}
              </span>
            </button>
          ))}
        </nav>

        <div className="licenses-detail">
          <header className="licenses-detail-head">
            <h3>{selected.name}</h3>
            <dl className="licenses-facts">
              {selected.publisher ? (
                <>
                  <dt>Publisher</dt>
                  <dd>{selected.publisher}</dd>
                </>
              ) : null}
              <dt>License</dt>
              <dd>{selected.license}</dd>
              {selected.version ? (
                <>
                  <dt>Version</dt>
                  <dd>{selected.version}</dd>
                </>
              ) : null}
              {selected.homepage ? (
                <>
                  <dt>Homepage</dt>
                  <dd>
                    {safeExternalUrl(selected.homepage) ? (
                      <a href={safeExternalUrl(selected.homepage)} target="_blank" rel="noreferrer">
                        {selected.homepage}
                      </a>
                    ) : (
                      selected.homepage
                    )}
                  </dd>
                </>
              ) : null}
            </dl>
            {selected.usage ? <p className="licenses-usage">{selected.usage}</p> : null}
          </header>

          {selected.documents.length > 1 ? (
            <div className="segmented-control" role="group" aria-label="License documents">
              {selected.documents.map((doc, index) => (
                <button
                  key={doc.label}
                  type="button"
                  className={index === docIndex ? "active" : ""}
                  onClick={() => setDocIndex(index)}
                >
                  {doc.label}
                </button>
              ))}
            </div>
          ) : null}

          {activeDoc ? (
            <pre className="licenses-text" aria-label={`${selected.name} — ${activeDoc.label}`}>
              {activeDoc.text}
            </pre>
          ) : (
            <p className="licenses-empty">No license text on file for this component.</p>
          )}
        </div>
      </div>
    </section>
  );
}

export default LicensesScreen;
