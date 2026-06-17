import React from "react";
import { Icon } from "../components/Icons.jsx";
import { useAppContext } from "../context/AppContext.js";

export function SimpleWorkflowScreen() {
  const { activeProject, recentImageAssets = [], recentVideoAssets = [], setActiveView } = useAppContext();
  const recentCount = recentImageAssets.length + recentVideoAssets.length;

  return (
    <section className="main-surface simple-workflow">
      <div className="simple-workflow-header">
        <div>
          <h2>Start a workflow</h2>
          <p>
            {activeProject?.name
              ? `Use ${activeProject.name} with a streamlined path into the existing studios.`
              : "Create or open a workspace to begin."}
          </p>
        </div>
        <span className="simple-workflow-count">{recentCount} recent</span>
      </div>

      <div className="simple-workflow-actions" aria-label="Simple workflow actions">
        <button type="button" onClick={() => setActiveView("Image")}>
          <Icon.Image />
          <span>Create image</span>
        </button>
        <button type="button" onClick={() => setActiveView("Video")}>
          <Icon.Video />
          <span>Create video</span>
        </button>
        <button type="button" onClick={() => setActiveView("Library")}>
          <Icon.Library />
          <span>Browse assets</span>
        </button>
        <button type="button" onClick={() => setActiveView("Queue")}>
          <Icon.Queue />
          <span>View queue</span>
        </button>
      </div>
    </section>
  );
}
