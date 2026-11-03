import React, { useEffect, useRef, useState } from "react";
import { Icon } from "../Icons.jsx";
import { Logo } from "../Logo.jsx";
import { ScreenActiveContext } from "../../context/ScreenActiveContext.js";

// Selective lazy keep-alive (sc-11959, backbone for epic 11949's edit persistence).
// These view ids mount on FIRST visit and then stay mounted (hidden) so their React
// state — in-progress prompt, settings, edits — survives leaving and returning, and
// the studios stop re-hydrating (and clobbering) from localStorage on every visit.
// Everything NOT listed here (Queue, Settings, Stats, Logs, Licenses, Models, and
// Library/Assets) keeps today's conditional unmount-on-navigation behavior. Training
// contributes two view ids — the default "Train" workspace and the "LibraryDataSets"
// Data Sets mode — both part of the Training Studio family.
export const KEEP_ALIVE_VIEWS = Object.freeze(
  new Set([
    "Image",
    "Video",
    "Audio",
    "Characters",
    "Document",
    "Train",
    "LibraryDataSets",
    "Presets",
    "Poses",
    "Keypoints",
    "Editor",
    "ImageEditor",
    "VectorStudio",
  ]),
);

// Wrapper for a kept-alive screen (sc-11959). The screen mounts once and then stays
// mounted; navigating away toggles `active` false, which hides the pane instead of
// unmounting it. `display: contents` (see .keep-alive-pane in styles.css) keeps the
// wrapper transparent to the .workspace flex column while visible — the child screen
// lays out exactly as a direct .workspace child would — and the `hidden` attribute
// drops it from layout and the accessibility tree while backgrounded. The pane also
// publishes `active` on ScreenActiveContext so the screen can pause expensive work
// when hidden (S2) or drop a leave-guard when it isn't the foreground view.
export function KeepAlivePane({ active, children }) {
  return (
    <div className="keep-alive-pane" hidden={!active}>
      <ScreenActiveContext.Provider value={active}>{children}</ScreenActiveContext.Provider>
    </div>
  );
}

// Exported so the nav registration (Workspace / Library / System sections) is assertable in
// unit tests without rendering the whole App (epic 13400 C0 mirrors the KEEP_ALIVE_VIEWS export).
export const navSections = [
  {
    label: "Workspace",
    items: [
      { id: "Image", icon: Icon.Image },
      { id: "Video", icon: Icon.Video },
      // Audio Studio (epic 13400) is a generative studio — it sits with Image/Video in Workspace.
      { id: "Audio", icon: Icon.Audio },
      // Character Studio is a generative studio (sc-2300) — it sits with Image/Video,
      // below Video and above Training, not in the Library section.
      { id: "Characters", icon: Icon.Character },
      { id: "Document", icon: Icon.Wand },
      { id: "Train", icon: Icon.Train },
      { id: "ImageEditor", label: "Image Editor", icon: Icon.ImageEditor },
      { id: "VectorStudio", label: "Vector Studio", icon: Icon.Wand },
      { id: "Editor", label: "Video Editor", icon: Icon.Editor },
    ],
  },
  {
    label: "Library",
    items: [
      { id: "Library", label: "Assets", icon: Icon.Library },
      { id: "LibraryDataSets", label: "Data Sets", icon: Icon.Train },
      { id: "DatasetCatalogs", label: "Dataset Catalogs", icon: Icon.Book },
      { id: "Poses", label: "Pose Library", icon: Icon.Character },
      { id: "Keypoints", label: "Key Point Library", icon: Icon.Character },
      { id: "Presets", icon: Icon.Preset },
      { id: "Models", icon: Icon.Model },
    ],
  },
  {
    label: "System",
    items: [
      { id: "Queue", icon: Icon.Queue },
      { id: "Stats", icon: Icon.Chart },
      { id: "Logs", icon: Icon.Logs },
      { id: "Settings", icon: Icon.Sliders },
      { id: "Licenses", icon: Icon.Info },
    ],
  },
];

// Exported alongside navSections so a view's title/blurb registration is unit-testable.
export const viewTitles = {
  Library: { title: "Assets", blurb: "Browse stills and clips across all your projects." },
  LibraryDataSets: { title: "Data Sets", blurb: "Create and caption training datasets." },
  DatasetCatalogs: {
    title: "Dataset Catalogs",
    blurb: "Index, analyze, and monitor large datasets independently of any workspace.",
  },
  Poses: { title: "Pose Library", blurb: "Manage whole-body pose skeletons and create new ones from photos." },
  Keypoints: {
    title: "Key Point Library",
    blurb: "Capture face-angle framing presets and compose angle-set collections for character turnarounds.",
  },
  Image: { title: "Image Studio", blurb: "Describe what you want — we'll render variations side by side." },
  Video: { title: "Video Studio", blurb: "Bring stills to life, or render new clips from scratch." },
  Audio: { title: "Audio Studio", blurb: "Generate speech, music and sound effects — or clone a voice." },
  Document: { title: "Document Studio", blurb: "Generate interleaved text-image documents — guides, storyboards, tutorials." },
  Train: { title: "Training Studio", blurb: "Build datasets and prepare LoRA training plans." },
  Editor: { title: "Video Editor", blurb: "Cut, sequence and export your timeline." },
  ImageEditor: { title: "Image Editor", blurb: "Crop, upscale and refine a single image on a canvas." },
  VectorStudio: { title: "Vector Studio", blurb: "Convert a project raster image or create one through a disclosed raster-to-vector workflow." },
  Characters: { title: "Characters", blurb: "Keep the same face across every shot." },
  Presets: { title: "Presets", blurb: "Save and share recurring generation setups." },
  Models: { title: "Models", blurb: "Download, import and manage local checkpoints." },
  Queue: { title: "Queue", blurb: "All running and recent jobs across workers." },
  Stats: { title: "Generation Stats", blurb: "Compare runs by model, quant, settings, timing and memory." },
  Logs: { title: "Logs", blurb: "This session's activity — routing decisions, worker phases and errors." },
  Settings: {
    title: "Settings",
    blurb: "Appearance, generation defaults, storage, credentials and this machine.",
  },
  Licenses: {
    title: "Licenses",
    blurb: "Third-party components bundled with SceneWorks and their license notices.",
  },
};

export function ProjectSwitcher({ activeProject, projects, onSelect, onCreate, disabled, onOpen, collapsed }) {
  const [open, setOpen] = useState(false);
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const containerRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    if (!open) {
      return undefined;
    }
    function onDocMouseDown(event) {
      if (!containerRef.current?.contains(event.target)) {
        setOpen(false);
        setCreating(false);
        setName("");
      }
    }
    function onDocKey(event) {
      if (event.key === "Escape") {
        setOpen(false);
        setCreating(false);
        setName("");
      }
    }
    document.addEventListener("mousedown", onDocMouseDown);
    document.addEventListener("keydown", onDocKey);
    return () => {
      document.removeEventListener("mousedown", onDocMouseDown);
      document.removeEventListener("keydown", onDocKey);
    };
  }, [open]);

  useEffect(() => {
    if (creating) {
      inputRef.current?.focus();
    }
  }, [creating]);

  useEffect(() => {
    if (collapsed) {
      setOpen(false);
      setCreating(false);
      setName("");
    }
  }, [collapsed]);

  async function submitNew(event) {
    event.preventDefault();
    const trimmed = name.trim();
    if (!trimmed || submitting) {
      return;
    }
    setSubmitting(true);
    const created = await onCreate(trimmed);
    setSubmitting(false);
    if (created) {
      setName("");
      setCreating(false);
      setOpen(false);
    }
  }

  return (
    <div className="project-switcher" ref={containerRef}>
      <button
        aria-expanded={open}
        aria-haspopup="listbox"
        aria-label={activeProject?.name ?? "Pick a workspace"}
        className="project-pill"
        disabled={disabled}
        onClick={() => {
          if (!open) onOpen?.();
          setOpen((value) => !value);
        }}
        title={activeProject?.name ?? "Pick a workspace"}
        type="button"
      >
        <span className="project-pill-thumb" aria-hidden="true" />
        <span className="project-pill-meta">
          <strong>{activeProject?.name ?? "No workspace open"}</strong>
          <span>
            {projects.length} workspace{projects.length === 1 ? "" : "s"}
          </span>
        </span>
        <Icon.ChevDown className="chev" />
      </button>

      {open ? (
        <div className="project-menu" role="listbox">
          {creating ? (
            <form className="project-menu-create" onSubmit={submitNew}>
              <input
                aria-label="New workspace name"
                disabled={submitting}
                onChange={(event) => setName(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Escape") {
                    event.preventDefault();
                    setCreating(false);
                    setName("");
                  }
                }}
                placeholder="Workspace name"
                ref={inputRef}
                value={name}
              />
              <button disabled={!name.trim() || submitting} type="submit">
                {submitting ? "Creating…" : "Create"}
              </button>
            </form>
          ) : (
            <button
              className="project-menu-item project-menu-item-new"
              disabled={disabled}
              onClick={() => setCreating(true)}
              type="button"
            >
              <Icon.Plus />
              <span className="project-menu-label">New workspace</span>
            </button>
          )}

          {projects.length ? <div className="project-menu-divider" role="separator" /> : null}

          {projects.length === 0 ? (
            <p className="project-menu-empty">No workspaces yet — create the first one above.</p>
          ) : (
            projects.map((project) => (
              <button
                aria-selected={project.id === activeProject?.id}
                className={project.id === activeProject?.id ? "project-menu-item active" : "project-menu-item"}
                key={project.id}
                onClick={() => {
                  onSelect(project);
                  setOpen(false);
                  setCreating(false);
                  setName("");
                }}
                role="option"
                type="button"
              >
                <span className="project-menu-thumb" aria-hidden="true" />
                <span className="project-menu-label">{project.name}</span>
              </button>
            ))
          )}
        </div>
      ) : null}
    </div>
  );
}

export function FirstRunProjectGate({ onCreate, disabled }) {
  const [name, setName] = useState("");
  const [submitting, setSubmitting] = useState(false);

  async function submit(event) {
    event.preventDefault();
    const trimmed = name.trim();
    if (!trimmed || submitting) {
      return;
    }
    setSubmitting(true);
    try {
      await onCreate(trimmed);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <section className="first-run-gate">
      <div className="first-run-card">
        <span className="first-run-mark" aria-hidden="true">
          <Logo size={52} />
        </span>
        <h2>Create your first workspace</h2>
        <p className="first-run-lede">
          SceneWorks keeps your images, videos, characters, and timelines inside a
          workspace. Create one to start generating.
        </p>
        <form className="first-run-form" onSubmit={submit}>
          <input
            aria-label="Workspace name"
            autoFocus
            disabled={disabled || submitting}
            onChange={(event) => setName(event.target.value)}
            placeholder="e.g. My First Project"
            value={name}
          />
          <button className="first-run-cta" disabled={disabled || submitting || !name.trim()} type="submit">
            {submitting ? "Creating…" : "Create workspace"}
          </button>
        </form>
      </div>
    </section>
  );
}
