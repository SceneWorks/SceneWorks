import React, { useMemo, useRef, useState } from "react";
import { Icon } from "../components/Icons.jsx";

const tabs = [
  { id: "dataset", label: "Dataset", title: "Dataset intake", status: "Rust dataset store" },
  { id: "rename-caption", label: "Rename & Caption", title: "Rename and caption pass", status: "Workflow reserved" },
  { id: "configure", label: "Configure Job", title: "Configure training job", status: "Queue disabled" },
];

function formatDatasetModality(dataset) {
  return String(dataset.modality ?? "image").replaceAll("_", " ");
}

function datasetItemCount(dataset) {
  const value = Number(dataset.itemCount ?? dataset.items?.length ?? 0);
  return Number.isFinite(value) ? value : 0;
}

function summarizeDatasets(datasets) {
  return datasets.reduce((summary, dataset) => ({ items: summary.items + datasetItemCount(dataset) }), { items: 0 });
}

export function TrainingStudio({
  activeProject,
  authenticated = true,
  datasets = [],
  datasetsError = "",
  loadingDatasets = false,
  onRefreshDatasets = () => {},
}) {
  const [activeTab, setActiveTab] = useState("dataset");
  const tabRefs = useRef({});
  const activeIndex = tabs.findIndex((tab) => tab.id === activeTab);
  const active = tabs[activeIndex] ?? tabs[0];
  const datasetSummary = useMemo(() => summarizeDatasets(datasets), [datasets]);

  function focusTab(index) {
    const next = tabs[(index + tabs.length) % tabs.length];
    setActiveTab(next.id);
    window.requestAnimationFrame(() => tabRefs.current[next.id]?.focus());
  }

  function onTabKeyDown(event) {
    if (event.key === "ArrowRight") {
      event.preventDefault();
      focusTab(activeIndex + 1);
    }
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      focusTab(activeIndex - 1);
    }
    if (event.key === "Home") {
      event.preventDefault();
      focusTab(0);
    }
    if (event.key === "End") {
      event.preventDefault();
      focusTab(tabs.length - 1);
    }
  }

  return (
    <section className="main-surface training-studio">
      <div className="training-studio-shell">
        <div className="training-summary-band">
          <div className="section-heading">
            <p className="eyebrow">Training Studio</p>
            <h2>Native LoRA training workflow</h2>
            <p className="view-copy">
              Build datasets, normalize captions, and prepare a Rust-owned training plan before any ML runtime work begins.
            </p>
          </div>
          <div className="training-metrics" aria-label="Training workspace summary">
            <div>
              <strong>{activeProject?.name ?? "No workspace"}</strong>
              <span>Project</span>
            </div>
            <div>
              <strong>{datasets.length}</strong>
              <span>Datasets</span>
            </div>
            <div>
              <strong>{datasetSummary.items}</strong>
              <span>Items</span>
            </div>
          </div>
        </div>

        {!authenticated ? (
          <div className="training-empty-state" role="status">
            <Icon.Train size={24} />
            <div>
              <strong>Pairing required</strong>
              <span>Unlock SceneWorks to load project training datasets.</span>
            </div>
          </div>
        ) : !activeProject ? (
          <div className="training-empty-state" role="status">
            <Icon.Folder size={24} />
            <div>
              <strong>No workspace open</strong>
              <span>Create or select a workspace before building a training dataset.</span>
            </div>
          </div>
        ) : (
          <>
            <div className="training-tabs" role="tablist" aria-label="Training workflow">
              {tabs.map((tab) => (
                <button
                  aria-controls={activeTab === tab.id ? `training-panel-${tab.id}` : undefined}
                  aria-selected={activeTab === tab.id}
                  className={activeTab === tab.id ? "active" : ""}
                  id={`training-tab-${tab.id}`}
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  onKeyDown={onTabKeyDown}
                  ref={(node) => {
                    tabRefs.current[tab.id] = node;
                  }}
                  role="tab"
                  tabIndex={activeTab === tab.id ? 0 : -1}
                  type="button"
                >
                  <span>{tab.label}</span>
                  <small>{tab.status}</small>
                </button>
              ))}
            </div>

            <section
              aria-labelledby={`training-tab-${active.id}`}
              className="training-panel"
              id={`training-panel-${active.id}`}
              role="tabpanel"
            >
              {activeTab === "dataset" ? (
                <>
                  <div className="training-panel-head">
                    <div>
                      <p className="eyebrow">Dataset</p>
                      <h3>{active.title}</h3>
                    </div>
                    <button className="secondary-action" disabled={loadingDatasets} onClick={onRefreshDatasets} type="button">
                      <Icon.Search size={14} />
                      {loadingDatasets ? "Refreshing" : "Refresh"}
                    </button>
                  </div>
                  {datasetsError ? <p className="inline-warning">{datasetsError}</p> : null}
                  {loadingDatasets ? (
                    <div className="empty-panel">Loading training datasets</div>
                  ) : datasets.length ? (
                    <div className="training-dataset-list">
                      {datasets.map((dataset) => {
                        const itemCount = datasetItemCount(dataset);
                        return (
                          <article className="training-dataset-row" key={dataset.id}>
                            <div>
                              <strong>{dataset.name ?? dataset.id}</strong>
                              <span>{formatDatasetModality(dataset)} dataset</span>
                            </div>
                            <span>{itemCount} item{itemCount === 1 ? "" : "s"}</span>
                          </article>
                        );
                      })}
                    </div>
                  ) : (
                    <div className="empty-panel">No training datasets yet</div>
                  )}
                </>
              ) : null}

              {activeTab === "rename-caption" ? (
                <>
                  <div className="training-panel-head">
                    <div>
                      <p className="eyebrow">Rename & Caption</p>
                      <h3>{active.title}</h3>
                    </div>
                    <span className="training-status-pill">Not queueable</span>
                  </div>
                  <div className="training-workflow-grid">
                    <div className="training-step-block">
                      <strong>Batch rename</strong>
                      <span>Stable filenames and ordered item ids will be prepared from the selected dataset.</span>
                    </div>
                    <div className="training-step-block">
                      <strong>Caption sidecars</strong>
                      <span>Caption metadata will stay attached to SceneWorks dataset items before sidecar export.</span>
                    </div>
                  </div>
                </>
              ) : null}

              {activeTab === "configure" ? (
                <>
                  <div className="training-panel-head">
                    <div>
                      <p className="eyebrow">Configure Job</p>
                      <h3>{active.title}</h3>
                    </div>
                    <span className="training-status-pill">Dry run pending</span>
                  </div>
                  <div className="training-config-preview" aria-label="Training job placeholder settings">
                    <label>
                      Target
                      <select defaultValue="z_image_turbo" disabled>
                        <option value="z_image_turbo">Z-Image Turbo LoRA</option>
                      </select>
                    </label>
                    <label>
                      Dataset
                      <select defaultValue="" disabled>
                        <option value="">Select after dataset workflow</option>
                      </select>
                    </label>
                    <label>
                      Preset
                      <select defaultValue="simple" disabled>
                        <option value="simple">Simple defaults</option>
                      </select>
                    </label>
                  </div>
                  <p className="inline-warning">Training submission is disabled until the dry-run plan story wires queue semantics.</p>
                </>
              ) : null}
            </section>
          </>
        )}
      </div>
    </section>
  );
}
