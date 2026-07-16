import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { apiFetch, isAbortError } from "./api.js";
import { pollJobToCompletion } from "./pollJob.js";
import { AccentPicker } from "./components/AccentPicker.jsx";
import { Icon } from "./components/Icons.jsx";
import { Logo } from "./components/Logo.jsx";
import { StatusDot } from "./components/StatusDot.jsx";
import { FullscreenPreview, assetSeed } from "./components/assetPanels.jsx";
import { fallbackModels, terminalStatuses } from "./constants.js";
import { LibraryScreen } from "./screens/LibraryScreen.jsx";
import { PoseLibraryScreen } from "./screens/PoseLibraryScreen.jsx";
import { KeyPointLibraryScreen } from "./screens/KeyPointLibraryScreen.jsx";
import { ModelManagerScreen } from "./screens/ModelManagerScreen.jsx";
import { ImageStudio } from "./screens/ImageStudio.jsx";
import { DocumentStudio } from "./screens/DocumentStudio.jsx";
import { VideoStudio } from "./screens/VideoStudio.jsx";
import { TrainingDataSetsLibrary, TrainingStudio } from "./screens/TrainingStudio.jsx";
import { CharacterStudio } from "./screens/CharacterStudio.jsx";
import { EditorScreen } from "./screens/EditorScreen.jsx";
import { QueueScreen } from "./screens/QueueScreen.jsx";
import { PresetManagerScreen } from "./screens/PresetManagerScreen.jsx";
import { SettingsScreen } from "./screens/SettingsScreen.jsx";
import { LogsScreen } from "./screens/LogsScreen.jsx";
import { StatsScreen } from "./screens/StatsScreen.jsx";
import { LicensesScreen } from "./screens/LicensesScreen.jsx";
import { SetupWizard } from "./screens/SetupWizard.jsx";
import { editModelForAsset, workflowModelType } from "./presetUtils.js";
import { sortNewest, sortWorkers, upsertJobNewest } from "./sorters.js";
import { useCharacters } from "./hooks/useCharacters.js";
import { usePresets } from "./hooks/usePresets.js";
import { usePromptBatches } from "./hooks/usePromptBatches.js";
import { useTraining } from "./hooks/useTraining.js";
import { useModelsAndLoras } from "./hooks/useModelsAndLoras.js";
import { usePersonTracks } from "./hooks/usePersonTracks.js";
import { useTimelines } from "./hooks/useTimelines.js";
import { useAccessGate } from "./hooks/useAccessGate.js";
import { useDropNavigationGuard } from "./hooks/useDropNavigationGuard.js";
import { useJobEvents } from "./hooks/useJobEvents.js";
import { AppStaticContext, AppLiveContext } from "./context/AppContext.js";
import { ScreenActiveContext } from "./context/ScreenActiveContext.js";
import { DEFAULT_MAC_CAPABILITIES } from "./macGating.js";
import { isAccentId } from "./accents.js";
import { writeDefaultGenerationQuality } from "./generationQuality.js";
import {
  dropUpscaledVariants,
  findFoldedAssetById,
  foldUpscaledAssetVariants,
  restrictFoldedToScope,
} from "./assetVariants.js";
import { buildWorkersById } from "./workers.js";
import { createEditorScratchRegistry } from "./editorScratch.js";
import { appConfirm, ConfirmHost } from "./appConfirm.jsx";
import { isDesktop as isDesktopShell, tauriInvoke } from "./runtime.js";
import {
  buildLocalJobStack,
  isActiveWorker,
  isImageGenerationJob,
  isInterleaveJob,
  isPlaceholderOnlyGpuWorker,
  isSelectableGpuWorker,
  isVideoGenerationJob,
  localJobStackLimit,
  mergeFreshJobs,
  readStoredAccent,
  readStoredTheme,
} from "./appHelpers.js";

// Desktop (Tauri) shell detection (unified helper, epic 4484 story 6). The first-run
// setup wizard is desktop-only; web/Docker (and a remote LAN browser) keep the
// existing first-run project gate. Tauri commands persist the wizard state (the API
// binds a random port each launch, so localStorage — keyed to the origin — can't be
// relied on across launches).

// Lazy-load the canvas editor so Konva (canvas-based, heavy) stays out of the
// initial bundle and the jsdom test path — it only loads when the view is opened.
const ImageEditor = React.lazy(() =>
  import("./screens/ImageEditor.jsx").then((module) => ({ default: module.ImageEditor })),
);

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
    "Characters",
    "Document",
    "Train",
    "LibraryDataSets",
    "Presets",
    "Poses",
    "Keypoints",
    "Editor",
    "ImageEditor",
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
function KeepAlivePane({ active, children }) {
  return (
    <div className="keep-alive-pane" hidden={!active}>
      <ScreenActiveContext.Provider value={active}>{children}</ScreenActiveContext.Provider>
    </div>
  );
}

// Product version, injected at build time from package.json (see vite.config.js).
// Empty in unconfigured contexts (e.g. some test paths); the footer is hidden then.
const APP_VERSION = import.meta.env.VITE_APP_VERSION ?? "";

const navSections = [
  {
    label: "Workspace",
    items: [
      { id: "Image", icon: Icon.Image },
      { id: "Video", icon: Icon.Video },
      // Character Studio is a generative studio (sc-2300) — it sits with Image/Video,
      // below Video and above Training, not in the Library section.
      { id: "Characters", icon: Icon.Character },
      { id: "Document", icon: Icon.Wand },
      { id: "Train", icon: Icon.Train },
      { id: "ImageEditor", label: "Image Editor", icon: Icon.ImageEditor },
      { id: "Editor", label: "Video Editor", icon: Icon.Editor },
    ],
  },
  {
    label: "Library",
    items: [
      { id: "Library", label: "Assets", icon: Icon.Library },
      { id: "LibraryDataSets", label: "Data Sets", icon: Icon.Train },
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

const viewTitles = {
  Library: { title: "Assets", blurb: "Browse stills and clips across all your projects." },
  LibraryDataSets: { title: "Data Sets", blurb: "Create and caption training datasets." },
  Poses: { title: "Pose Library", blurb: "Manage whole-body pose skeletons and create new ones from photos." },
  Keypoints: {
    title: "Key Point Library",
    blurb: "Capture face-angle framing presets and compose angle-set collections for character turnarounds.",
  },
  Image: { title: "Image Studio", blurb: "Describe what you want — we'll render variations side by side." },
  Video: { title: "Video Studio", blurb: "Bring stills to life, or render new clips from scratch." },
  Document: { title: "Document Studio", blurb: "Generate interleaved text-image documents — guides, storyboards, tutorials." },
  Train: { title: "Training Studio", blurb: "Build datasets and prepare LoRA training plans." },
  Editor: { title: "Video Editor", blurb: "Cut, sequence and export your timeline." },
  ImageEditor: { title: "Image Editor", blurb: "Crop, upscale and refine a single image on a canvas." },
  Characters: { title: "Characters", blurb: "Keep the same face across every shot." },
  Presets: { title: "Presets", blurb: "Save and share recurring generation setups." },
  Models: { title: "Models", blurb: "Download, import and manage local checkpoints." },
  Queue: { title: "Queue", blurb: "All running and recent jobs across workers." },
  Stats: { title: "Generation Stats", blurb: "Compare runs by model, quant, settings, timing and memory." },
  Logs: { title: "Logs", blurb: "This session's activity — routing decisions, worker phases and errors." },
  Settings: { title: "Settings", blurb: "Paths, service tokens, and detected GPU." },
  Licenses: {
    title: "Licenses",
    blurb: "Third-party components bundled with SceneWorks and their license notices.",
  },
};

function ProjectSwitcher({ activeProject, projects, onSelect, onCreate, disabled }) {
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
        className="project-pill"
        disabled={disabled}
        onClick={() => setOpen((value) => !value)}
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

function FirstRunProjectGate({ onCreate, disabled }) {
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

export function App() {
  const [health, setHealth] = useState(null);
  const [projects, setProjects] = useState([]);
  const [projectsLoaded, setProjectsLoaded] = useState(false);
  // Desktop first-run wizard gate: null = unknown (still reading on desktop),
  // true = no wizard needed (web, or already completed), false = show the wizard.
  const [setupCompleted, setSetupCompleted] = useState(isDesktopShell ? null : true);
  const [activeProject, setActiveProject] = useState(null);
  const [activeView, setActiveView] = useState("Library");
  // Selective lazy keep-alive (sc-11959): the set of keep-alive views the user has
  // visited at least once. A view mounts on first visit (activeView === view) and,
  // once here, stays mounted (hidden) across navigation so its state survives.
  const [visitedKeepAliveViews, setVisitedKeepAliveViews] = useState(() => new Set());
  const [jobs, setJobs] = useState([]);
  const [localGenerationJobIds, setLocalGenerationJobIds] = useState({ image: [], video: [], document: [] });
  const [workers, setWorkers] = useState([]);
  const [queueSummary, setQueueSummary] = useState(null);
  // Mac UI gating (sc-3486): inert until the capabilities endpoint reports macGatingActive.
  const [macCapabilities, setMacCapabilities] = useState(DEFAULT_MAC_CAPABILITIES);
  const [trainingTargets, setTrainingTargets] = useState({ schemaVersion: 1, targets: [] });
  const [trainingPresets, setTrainingPresets] = useState({ schemaVersion: 1, presets: [] });
  const [trainingTargetsError, setTrainingTargetsError] = useState("");
  const [trainingPresetsError, setTrainingPresetsError] = useState("");
  const [assets, setAssets] = useState([]);
  const [selectedAssetId, setSelectedAssetId] = useState(null);
  const [projectFilter, setProjectFilter] = useState("all");
  const [requestedGpu, setRequestedGpu] = useState("auto");
  const [jobPrompt, setJobPrompt] = useState("Placeholder generation");
  const [latestGenerationSetId, setLatestGenerationSetId] = useState(null);
  const [previewAsset, setPreviewAsset] = useState(null);
  // The collection the fullscreen preview was launched from, as an ordered list
  // of asset ids. Navigation (next/previous and the discard-advance) stays bound
  // to this set so scrolling never escapes into the Library or another
  // character's assets. `null` falls back to "all assets" for any legacy caller
  // that opens the preview without a scope.
  const [previewScopeIds, setPreviewScopeIds] = useState(null);
  // Which way the user last scrolled in the fullscreen preview, so discarding an
  // asset advances in that same direction.
  const previewDirectionRef = useRef("next");
  // Open the fullscreen preview bound to the collection it was launched from.
  // `scopeAssets` is the exact list the calling gallery rendered (folded or not);
  // we snapshot its ids so navigation tracks the live asset state but never
  // wanders outside that collection. Passing no scope clears it (global nav).
  // sc-4194: useCallback so the context-exposed setPreviewAsset identity is stable.
  const openPreview = useCallback((asset, scopeAssets) => {
    if (!asset) {
      setPreviewScopeIds(null);
      setPreviewAsset(null);
      return;
    }
    setPreviewScopeIds(
      Array.isArray(scopeAssets) && scopeAssets.length
        ? scopeAssets.map((item) => item.id)
        : null,
    );
    setPreviewAsset(asset);
  }, []);
  const closePreview = () => {
    setPreviewScopeIds(null);
    setPreviewAsset(null);
  };
  const [studioLaunch, setStudioLaunch] = useState(null);
  // sc-8730: the launch channel INTO the Image Editor canvas (crop/upscale/refine
  // screen, activeView === "ImageEditor"). Mirrors studioLaunch but targets the
  // editor's openAsset(assetId) rather than Image Studio. `id` is a fresh UUID per
  // launch so relaunching the same asset still fires the editor's id-keyed effect.
  const [editorLaunch, setEditorLaunch] = useState(null);
  // sc-4198: a small notices store replaces the single `error` string that used to
  // double as a message bus — the fragile "lora import:"/"lora training:" startsWith
  // protocol. Each notice has a stable `kind`; pushing a kind replaces only that kind
  // and dismissing clears only that kind, so an unrelated success (or a background SSE
  // refresh) no longer wipes an unread, still-relevant notice of a different kind.
  const [notices, setNotices] = useState([]);
  const pushNotice = useCallback((kind, message) => {
    const text = String(message ?? "");
    setNotices((current) => {
      const others = current.filter((notice) => notice.kind !== kind);
      return text ? [...others, { kind, message: text }] : others;
    });
  }, []);
  const dismissNoticeKind = useCallback((kind) => {
    setNotices((current) => {
      // Bail to the same array when nothing matches — callers run this on hot
      // paths (e.g. every media-ticket refresh), and returning a fresh array
      // would force a no-op re-render of the whole app each time.
      const next = current.filter((notice) => notice.kind !== kind);
      return next.length === current.length ? current : next;
    });
  }, []);
  // Back-compat: the existing setError(msg)/setError("") call sites map onto the
  // "general" notice kind — a truthy message replaces it, "" dismisses only it.
  const setError = useCallback((message) => pushNotice("general", message), [pushNotice]);
  // Remote-access gate (epic 4484), extracted to a hook (sc-9750): owns access probe,
  // host password/token, login-gate draft + error, and the media-ticket mint that must
  // settle before protected data loads. `authenticated`/`ready` gate the data + SSE
  // effects below; `token` threads through the data hooks and every apiFetch call site.
  const {
    access,
    token,
    passwordDraft,
    setPasswordDraft,
    authError,
    authenticated,
    ready,
    saveToken,
    lockRemote,
  } = useAccessGate({ setError, pushNotice, dismissNoticeKind });
  // Stop a file dropped outside a real dropzone from navigating the webview to
  // the image and replacing the whole UI (issue #1308).
  useDropNavigationGuard();
  const [theme, setTheme] = useState(readStoredTheme);
  // Apply a theme and persist it through the API. localStorage gives an instant
  // initial paint, but on the desktop shell the UI runs at the API's per-launch
  // http://127.0.0.1:<port> origin, where both localStorage and Tauri IPC are
  // unreliable across launches — so the durable copy lives server-side.
  // Stable across renders (sc-10244): it flows through the static app context to the
  // Image Editor top-bar theme toggle, so a fresh identity each render would bust the
  // static-context memo every tick.
  const changeTheme = useCallback((next) => {
    setTheme(next);
    apiFetch("/api/v1/ui-preferences", "", {
      method: "PUT",
      body: JSON.stringify({ theme: next }),
    }).catch(() => {});
  }, []);
  const [accent, setAccent] = useState(readStoredAccent);
  // Same persistence contract as theme: instant localStorage cache + durable
  // server copy. The PUT sends only the changed field, so the endpoint must
  // MERGE partial updates (theme writes already rely on this).
  const changeAccent = (next) => {
    setAccent(next);
    apiFetch("/api/v1/ui-preferences", "", {
      method: "PUT",
      body: JSON.stringify({ accent: next }),
    }).catch(() => {});
  };
  const activeProjectRef = useRef(null);
  const activeViewRef = useRef(activeView);
  const localGenerationJobIdsRef = useRef(localGenerationJobIds);
  const generatedAssetRefreshesRef = useRef(new Map());
  const refreshDataRef = useRef(null);
  const refreshAssetsRef = useRef(null);
  // Latest purgeAsset, held in a ref so the App-level scratch-op survivor (sc-8850) can
  // purge orphaned scratch/result assets from the SSE handler without re-subscribing.
  const purgeAssetRef = useRef(null);
  const refreshCharactersRef = useRef(null);
  const refreshLorasRef = useRef(null);
  const refreshPresetsRef = useRef(null);
  const refreshPromptBatchesRef = useRef(null);
  const refreshTrainingDatasetsRef = useRef(null);
  const refreshPersonTracksRef = useRef(null);
  const refreshTimelinesRef = useRef(null);
  const refreshDataWithLoraOverlayRef = useRef(null);
  // A screen (the Image Editor, sc-2434) can register a guard that runs before a
  // user-initiated navigation leaves it — e.g. to confirm discarding unsaved edits.
  // Programmatic setActiveView calls (post-generation hops) deliberately bypass it.
  const leaveGuardRef = useRef(null);
  const registerLeaveGuard = useCallback((guard) => {
    leaveGuardRef.current = guard;
    return () => {
      if (leaveGuardRef.current === guard) leaveGuardRef.current = null;
    };
  }, []);
  // In-flight Image-Editor AI-op scratch registry (sc-8850). The editor stages an
  // ephemeral scratch asset per AI op and normally loads the result back + purges
  // everything itself. But navigating away mid-job unmounts the editor, so that
  // in-component purge never runs. This registry lives in App (survives the unmount) and
  // purges the tracked scratch/mask + result assets whenever such a job terminates and
  // the editor is no longer claiming it. It calls purgeAsset through a ref so it stays
  // stable across renders. See editorScratch.js for the full survivor behaviour.
  const editorScratchRegistryRef = useRef(null);
  if (!editorScratchRegistryRef.current) {
    editorScratchRegistryRef.current = createEditorScratchRegistry({
      purgeAsset: (asset) => purgeAssetRef.current?.(asset),
    });
  }
  const editorScratchRegistry = editorScratchRegistryRef.current;
  // Latest jobs list, so the claim-release sweep can read it without re-subscribing.
  const jobsRef = useRef([]);
  const trackEditorScratchOp = useCallback(
    (jobId, assets) => editorScratchRegistry.track(jobId, assets),
    [editorScratchRegistry],
  );
  const releaseEditorScratchOp = useCallback(
    (jobId, resultJob = null) => editorScratchRegistry.release(jobId, resultJob),
    [editorScratchRegistry],
  );
  const registerEditorScratchClaim = useCallback(
    (getClaimedIds) => editorScratchRegistry.registerClaim(getClaimedIds, () => jobsRef.current),
    [editorScratchRegistry],
  );
  const navTo = useCallback((viewId) => {
    if (viewId === activeViewRef.current) return;
    const guard = leaveGuardRef.current;
    if (!guard) {
      setActiveView(viewId);
      return;
    }
    // The guard may answer synchronously (a bare boolean, legacy) or asynchronously
    // (a Promise<boolean> — the Image Editor's desktop-safe appConfirm dialog, sc-11968).
    // Only a strict `false` / a promise resolving falsy cancels the leave; anything else
    // proceeds. A promise defers the view switch until the user answers the dialog.
    const decision = guard();
    if (decision && typeof decision.then === "function") {
      decision.then((ok) => {
        if (ok) setActiveView(viewId);
      });
      return;
    }
    if (decision === false) return; // guard returned false → user cancelled the leave
    setActiveView(viewId);
  }, []);

  // A screen holding an unsaved draft (Training Studio / Data Sets, sc-11970) can register
  // a guard consulted before the active PROJECT is switched — which would otherwise silently
  // reset the screen and discard the draft. Distinct from the nav leave-guard above: project
  // switch bypasses navTo entirely, and keep-alive screens must NOT prompt on plain nav, only
  // on a project change. Like the nav guard, a promise defers the switch until the user
  // answers; only a falsy answer cancels it. The guard receives the target project.
  const projectSwitchGuardRef = useRef(null);
  const registerProjectSwitchGuard = useCallback((guard) => {
    projectSwitchGuardRef.current = guard;
    return () => {
      if (projectSwitchGuardRef.current === guard) projectSwitchGuardRef.current = null;
    };
  }, []);
  // Consult the project-switch guard (if any) then switch. Resolves to whether the switch
  // actually happened, so imperative callers — creating a NEW workspace (createProject) — can
  // gate their follow-up (navigating into it) on the user confirming the discard (sc-11970).
  const requestProjectSwitch = useCallback(async (project) => {
    // No-op re-selection (same project) and clears bypass the guard — nothing is discarded.
    if (!project || project.id === activeProjectRef.current?.id) {
      setActiveProject(project);
      return true;
    }
    const guard = projectSwitchGuardRef.current;
    if (guard) {
      const decision = guard(project);
      const proceed =
        decision && typeof decision.then === "function" ? await decision : decision !== false;
      if (!proceed) return false; // guard cancelled → user kept editing, nothing discarded
    }
    setActiveProject(project);
    return true;
  }, []);
  // The ProjectSwitcher's onSelect fires the guarded switch and forgets the result.
  const selectProject = useCallback(
    (project) => {
      void requestProjectSwitch(project);
    },
    [requestProjectSwitch],
  );

  // sc-4194: defined here (above the data hooks) because useTimelines takes it as a
  // dependency; a stable identity keeps the timeline hook's queue action stable too.
  const createVideoJob = useCallback(
    async (payload, options = {}) => {
      const { navigateToQueue = false } = options;
      if (!activeProject) {
        setError("Create or open a project first.");
        return null;
      }
      try {
        const job = await apiFetch("/api/v1/video/jobs", token, {
          method: "POST",
          body: JSON.stringify({
            ...payload,
            projectId: activeProject.id,
            projectName: activeProject.name,
            requestedGpu,
          }),
        });
        if (navigateToQueue) {
          setActiveView("Queue");
        }
        setJobs((items) => upsertJobNewest(items, job));
        setError("");
        return job;
      } catch (err) {
        setError(err.message);
        return null;
      }
    },
    [token, activeProject, requestedGpu],
  );

  const {
    characters,
    setCharacters,
    refreshCharacters,
    createCharacter,
    updateCharacter,
    archiveCharacter,
    unarchiveCharacter,
    listArchivedCharacters,
    addCharacterReference,
    updateCharacterReference,
    removeCharacterReference,
    createCharacterLook,
    updateCharacterLook,
    deleteCharacterLook,
    attachCharacterLora,
    updateCharacterLora,
    detachCharacterLora,
    createCharacterTestJob,
  } = useCharacters({ token, activeProject, activeProjectRef, setError, requestedGpu, setActiveView });

  const {
    presets,
    setPresets,
    refreshPresets,
    createPreset,
    updatePreset,
    duplicatePreset,
    deletePreset,
  } = usePresets({ token, activeProject, setError });

  const {
    promptBatches,
    setPromptBatches,
    refreshPromptBatches,
    createPromptBatch,
    updatePromptBatch,
    duplicatePromptBatch,
    deletePromptBatch,
  } = usePromptBatches({ token, activeProject, setError });

  const {
    trainingDatasets,
    setTrainingDatasets,
    trainingDatasetsProjectId,
    setTrainingDatasetsProjectId,
    loadingTrainingDatasets,
    trainingDatasetsError,
    setTrainingDatasetsError,
    refreshTrainingDatasets,
    loadTrainingDataset,
    loadTrainingDatasetReadiness,
    setTrainingDatasetItemQualityAck,
    createTrainingDataset,
    uploadTrainingDatasetItem,
    updateTrainingDataset,
    batchRenameTrainingDataset,
    writeTrainingDatasetCaptionSidecars,
    createTrainingDatasetCaptionJob,
    createTrainingDatasetUpscaleJob,
    createTrainingDatasetAnalysisJob,
    createTrainingDatasetFaceAnalysisJob,
    smartCropTrainingDataset,
    stripExifTrainingDataset,
    createTrainingJob,
  } = useTraining({ token, activeProject, setError, setJobs });

  // sc-8811: useModelsAndLoras lists these two cross-cutting refresh orchestrators as
  // useCallback deps of deleteModel/deleteLora, which sit in appContextValue's
  // dependency array. The orchestrator bodies (refreshData / refreshDataWithLoraOverlay,
  // defined below) are plain per-render function declarations published into the
  // refs above, so passing them in directly would give deleteModel/deleteLora — and
  // therefore the whole ~130-key context value — a fresh identity on every App render,
  // silently defeating the sc-4194 memoization. These identity-stable wrappers delegate
  // through the refs instead: callers always invoke the latest body (fresh token /
  // activeProject), while the hook's actions stay referentially stable.
  const stableRefreshData = useCallback((...args) => refreshDataRef.current?.(...args), []);
  const stableRefreshDataWithLoraOverlay = useCallback(
    (...args) => refreshDataWithLoraOverlayRef.current?.(...args),
    [],
  );

  const {
    models,
    setModels,
    loras,
    setLoras,
    refreshLoras,
    deleteModel,
    deleteModelVariant,
    deleteLora,
    updateLora,
    fetchLoraEmbeddedTags,
    createModelImportJob,
    createLoraImportJob,
    createModelDownloadJob,
    createLoraDownloadJob,
    createModelConvertJob,
  } = useModelsAndLoras({
    token,
    activeProject,
    activeProjectRef,
    setError,
    setJobs,
    setActiveView,
    refreshData: stableRefreshData,
    refreshDataWithLoraOverlay: stableRefreshDataWithLoraOverlay,
  });

  const {
    personTracks,
    setPersonTracks,
    refreshPersonTracks,
    createPersonDetectionJob,
    createPersonTrackJob,
    saveTrackCorrections,
  } = usePersonTracks({ token, activeProject, activeProjectRef, setError, requestedGpu, setActiveView });

  const {
    timelines,
    setTimelines,
    setTimelinesProjectId,
    selectedTimelineId,
    setSelectedTimelineId,
    activeTimeline,
    setActiveTimeline,
    isActiveTimelineDirty,
    refreshTimelines,
    createTimeline,
    saveTimeline,
    exportTimeline,
    extractTimelineFrame,
    queueTimelineVideoJob,
    enqueueTimelineGenerationApply,
  } = useTimelines({
    token,
    activeProject,
    activeProjectRef,
    setError,
    pushNotice,
    requestedGpu,
    setActiveView,
    createVideoJob,
  });

  // `usable !== false` fails closed on external ComfyUI base models (sc-10667): they
  // are surfaced in the catalog with a reason but not yet runnable (the per-family
  // loaders are sc-10668+), so they must not be offered as a generation target.
  // Manifest models never set `usable`, so they are unaffected.
  const imageModels = useMemo(() => {
    const items = models.filter(
      (model) => model.type === "image" && model.installState !== "missing" && model.usable !== false,
    );
    return items.length || models.length ? items : fallbackModels.filter((model) => model.type === "image");
  }, [models]);
  const videoModels = useMemo(() => {
    const items = models.filter(
      (model) => model.type === "video" && model.installState !== "missing" && model.usable !== false,
    );
    return items.length || models.length ? items : fallbackModels.filter((model) => model.type === "video");
  }, [models]);
  const selectedAsset = useMemo(
    () => assets.find((asset) => asset.id === selectedAssetId) ?? assets[0] ?? null,
    [assets, selectedAssetId],
  );
  // Discarded (trashed) assets are excluded from the fullscreen navigation so
  // they don't show up while scrolling; purged assets are already dropped from
  // `assets` entirely.
  const foldedPreviewAssets = useMemo(
    () => foldUpscaledAssetVariants(assets.filter((asset) => !asset.status?.trashed)),
    [assets],
  );
  const previewedAsset = useMemo(
    () => (previewAsset ? findFoldedAssetById(foldedPreviewAssets, previewAsset.id) ?? previewAsset : null),
    [foldedPreviewAssets, previewAsset],
  );
  // The ordered, folded set the preview can navigate — restricted to the launch
  // collection so scrolling never escapes into the Library or another character.
  const previewScopeAssets = useMemo(
    () => restrictFoldedToScope(foldedPreviewAssets, previewScopeIds),
    [foldedPreviewAssets, previewScopeIds],
  );
  const previewNavigation = useMemo(() => {
    if (!previewedAsset || previewScopeAssets.length < 2) {
      return { previous: null, next: null };
    }
    const currentIndex = previewScopeAssets.findIndex((asset) => asset.id === previewedAsset.id);
    if (currentIndex < 0) {
      return { previous: null, next: null };
    }
    return {
      previous: currentIndex > 0 ? previewScopeAssets[currentIndex - 1] : null,
      next: currentIndex < previewScopeAssets.length - 1 ? previewScopeAssets[currentIndex + 1] : null,
    };
  }, [previewScopeAssets, previewedAsset]);
  const latestAssets = useMemo(
    () => assets.filter((asset) => asset.generationSetId === latestGenerationSetId),
    [assets, latestGenerationSetId],
  );
  const latestImageAssets = useMemo(() => latestAssets.filter((asset) => asset.type === "image"), [latestAssets]);
  const latestVideoAssets = useMemo(() => latestAssets.filter((asset) => asset.type === "video"), [latestAssets]);
  // Recent Assets (sc-2088 / sc-2089) — the 20 most recent image/video assets
  // generated in the active project. Replaces `latestImageAssets`/
  // `latestVideoAssets` (which only ever showed the latest single generation
  // set) as the studio "what just came out" list. Sorted newest-first.
  const recentImageAssets = useMemo(
    () =>
      dropUpscaledVariants(
        assets.filter((asset) => asset.type === "image" && (!activeProject?.id || asset.projectId === activeProject.id)),
      )
        .sort(sortNewest)
        .slice(0, 20),
    [assets, activeProject?.id],
  );
  const recentVideoAssets = useMemo(
    () =>
      assets
        .filter((asset) => asset.type === "video" && (!activeProject?.id || asset.projectId === activeProject.id))
        .slice()
        .sort(sortNewest)
        .slice(0, 20),
    [assets, activeProject?.id],
  );
  const imageLocalJobs = useMemo(
    () => buildLocalJobStack(localGenerationJobIds.image, jobs, activeProject?.id, isImageGenerationJob),
    [activeProject?.id, jobs, localGenerationJobIds.image],
  );
  const videoLocalJobs = useMemo(
    () => buildLocalJobStack(localGenerationJobIds.video, jobs, activeProject?.id, isVideoGenerationJob),
    [activeProject?.id, jobs, localGenerationJobIds.video],
  );
  const documentLocalJobs = useMemo(
    () => buildLocalJobStack(localGenerationJobIds.document, jobs, activeProject?.id, isInterleaveJob),
    [activeProject?.id, jobs, localGenerationJobIds.document],
  );
  const queueCounts = useMemo(() => {
    if (queueSummary?.counts) {
      return {
        ...queueSummary.counts,
        active: queueSummary.activeJobs?.length ?? jobs.filter((job) => !terminalStatuses.has(job.status)).length,
      };
    }
    return jobs.reduce(
      (counts, job) => {
        counts[job.status] = (counts[job.status] ?? 0) + 1;
        if (!terminalStatuses.has(job.status)) {
          counts.active += 1;
        }
        return counts;
      },
      { active: 0 },
    );
  }, [jobs, queueSummary]);
  const filteredJobs = useMemo(() => {
    if (projectFilter === "all") {
      return jobs;
    }
    return jobs.filter((job) => job.projectId === projectFilter);
  }, [jobs, projectFilter]);
  const visibleWorkers = useMemo(
    () => workers.filter((worker) => isActiveWorker(worker) && !isPlaceholderOnlyGpuWorker(worker)),
    [workers],
  );
  // O(1) lookup by worker.id so every WorkerProgressCard consumer reads live
  // worker state without rebuilding the map per screen (sc-2082).
  const workersById = useMemo(() => buildWorkersById(workers), [workers]);
  // Person-workflow readiness, derived from the live (non-offline) workers so it
  // tracks SSE worker registration/offline transitions instantly. Mirrors the
  // server's GET /api/v1/capabilities/person (person_readiness_from_workers); the
  // worker SSE handlers keep `workers` current, so this never goes stale.
  const personReadiness = useMemo(() => {
    const live = workers.filter((worker) => worker.status !== "offline");
    const ready = (capability) => live.some((worker) => (worker.capabilities ?? []).includes(capability));
    return {
      detect: { capability: "person_detect", ready: ready("person_detect") },
      track: { capability: "person_track", ready: ready("person_track") },
      segment: { capability: "person_segment", ready: ready("person_segment") },
      replace: { capability: "person_replace", ready: ready("person_replace") },
      detectPreview: { capability: "person_detect_preview", ready: ready("person_detect_preview") },
      trackPreview: { capability: "person_track_preview", ready: ready("person_track_preview") },
    };
  }, [workers]);
  const gpuOptions = useMemo(() => {
    const ids = visibleWorkers.filter(isSelectableGpuWorker).map((worker) => worker.gpuId);
    return ["auto", ...Array.from(new Set(ids))];
  }, [visibleWorkers]);
  const mediaAssets = useMemo(
    () => assets.filter((asset) => ["image", "video", "upload", "frame", "render", "document"].includes(asset.type)),
    [assets],
  );

  useEffect(() => {
    activeViewRef.current = activeView;
  }, [activeView]);

  // Record a keep-alive view the first time it becomes active so it stays mounted
  // thereafter (sc-11959). Never-visited keep-alive views are absent from the DOM.
  useEffect(() => {
    if (!KEEP_ALIVE_VIEWS.has(activeView)) {
      return;
    }
    setVisitedKeepAliveViews((prev) => (prev.has(activeView) ? prev : new Set(prev).add(activeView)));
  }, [activeView]);

  useEffect(() => {
    activeProjectRef.current = activeProject;
  }, [activeProject]);

  useEffect(() => {
    localGenerationJobIdsRef.current = localGenerationJobIds;
  }, [localGenerationJobIds]);

  useEffect(() => {
    if (typeof document === "undefined") {
      return;
    }
    document.documentElement.setAttribute("data-theme", theme);
    try {
      window.localStorage.setItem("sceneworks-theme", theme);
    } catch {
      // ignore (private mode etc.)
    }
  }, [theme]);

  useEffect(() => {
    if (typeof document === "undefined") {
      return;
    }
    document.documentElement.setAttribute("data-accent", accent);
    try {
      window.localStorage.setItem("sceneworks-accent", accent);
    } catch {
      // ignore (private mode etc.)
    }
  }, [accent]);

  // Seed the theme from the server on launch (the durable copy; localStorage is
  // only an instant-paint cache). Each toggle persists itself via changeTheme,
  // so there's no save effect to race with this read.
  useEffect(() => {
    let cancelled = false;
    apiFetch("/api/v1/ui-preferences", "")
      .then((prefs) => {
        if (cancelled) {
          return;
        }
        if (prefs?.theme === "dark" || prefs?.theme === "light") {
          setTheme(prefs.theme);
        }
        if (isAccentId(prefs?.accent)) {
          setAccent(prefs.accent);
        }
        // Re-prime the default-generation-quality cache from the durable server copy (sc-10728).
        // It isn't React state here — the studio reads it fresh from localStorage via
        // readDefaultGenerationQuality() — so seeding the cache is all that's needed for it to
        // survive a desktop relaunch (the GET always resolves a concrete tier). writeDefault…
        // normalizes, so an absent/legacy value lands on q8. No PUT: this is a read-only seed.
        if (prefs?.defaultGenerationQuality) {
          writeDefaultGenerationQuality(prefs.defaultGenerationQuality);
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    apiFetch("/api/v1/health", "")
      .then(setHealth)
      .catch((err) => setError(err.message));
    // The /api/v1/access probe (+ accessResolved release) lives in useAccessGate now (sc-9750).
  }, []);

  useEffect(() => {
    if (!isDesktopShell) {
      return;
    }
    tauriInvoke("get_storage_setup")
      .then((setup) => setSetupCompleted(Boolean(setup?.setupCompleted)))
      // Never block the app on a storage-state read failure; fall through to the studio.
      .catch(() => setSetupCompleted(true));
  }, []);

  // Desktop (WebView2 / Windows) only: opening a native file dialog from an
  // `<input type="file">` can strand the composited app shell painted blank.
  // When the modal dialog takes the foreground it disables the owner window and
  // Chromium stops presenting the promoted DirectComposition layers (`.sidebar`,
  // `.topbar`, `.workspace`); it does NOT restore them from ordinary window
  // messages (resize / minimize-restore) — only an internally-scheduled frame
  // recovers it, which is why disabling `CalculateNativeWinOcclusion` alone
  // doesn't help. So while a file picker is pending we pump a cheap, sub-pixel
  // transform on `.app` every frame to keep WebView2 presenting *during* the
  // dialog (not just after it closes, which was the visibly-flashing stopgap),
  // then do a final promote/demote once focus returns to guarantee recovery.
  // No-op off the desktop shell and on macOS WebKit, which never drops layers.
  useEffect(() => {
    if (!isDesktopShell) {
      return undefined;
    }

    let pumping = false;
    let rafId = 0;
    let timerId = 0;
    let deadlineId = 0;
    let phase = false;

    const shell = () => document.querySelector(".app");

    const tick = () => {
      const el = shell();
      if (el instanceof HTMLElement) {
        // The transform stays established the whole time (so `.app` never
        // toggles its containing-block status under any fixed child), but the
        // sub-pixel value changes each frame to force a fresh present. `.app`
        // fills the viewport at the origin, so the shift is imperceptible.
        phase = !phase;
        el.style.transform = phase ? "translate3d(0.02px, 0, 0)" : "translate3d(0, 0, 0)";
      }
      if (pumping) {
        rafId = requestAnimationFrame(tick);
      }
    };

    const stop = () => {
      if (!pumping) {
        return;
      }
      pumping = false;
      cancelAnimationFrame(rafId);
      window.clearInterval(timerId);
      window.clearTimeout(deadlineId);
      const el = shell();
      if (el instanceof HTMLElement) {
        // Promote-then-demote across two frames, then drop the inline transform
        // we borrowed, leaving the stylesheet in control again.
        requestAnimationFrame(() => {
          el.style.transform = "translateZ(0)";
          requestAnimationFrame(() => {
            el.style.transform = "";
          });
        });
      }
    };

    const start = () => {
      if (pumping) {
        return;
      }
      pumping = true;
      rafId = requestAnimationFrame(tick);
      // rAF is throttled while the window is occluded/deactivated — exactly our
      // failure window — so drive a timer fallback alongside it.
      timerId = window.setInterval(tick, 32);
      // Safety valve: never pump forever if a focus/change edge is missed.
      deadlineId = window.setTimeout(stop, 60000);
    };

    const onClick = (event) => {
      const target = event.target;
      if (target instanceof HTMLInputElement && target.type === "file") {
        start();
      }
    };

    // Only stop when visibility *returns*: opening the dialog can itself flip
    // the document to hidden, which must not tear down the pump we just armed.
    const onVisibility = () => {
      if (document.visibilityState === "visible") {
        stop();
      }
    };

    // The dialog closing restores focus / flips visibility back and (on select)
    // fires `change`; any of those edges ends the pump. Capture phase so the
    // click arms before the input's own handler opens the dialog.
    document.addEventListener("click", onClick, true);
    window.addEventListener("focus", stop);
    document.addEventListener("visibilitychange", onVisibility);
    document.addEventListener("change", stop, true);

    return () => {
      stop();
      document.removeEventListener("click", onClick, true);
      window.removeEventListener("focus", stop);
      document.removeEventListener("visibilitychange", onVisibility);
      document.removeEventListener("change", stop, true);
    };
  }, []);

  useEffect(() => {
    if (!ready) {
      return;
    }
    refreshDataRef.current?.();
  }, [ready, token]);

  useEffect(() => {
    if (!activeProject || !ready) {
      setAssets([]);
      setCharacters([]);
      setPersonTracks([]);
      setTimelines([]);
      setTimelinesProjectId(null);
      setPresets([]);
      setTrainingTargetsError("");
      setTrainingDatasets([]);
      setTrainingDatasetsProjectId(null);
      setTrainingDatasetsError("");
      setSelectedTimelineId(null);
      setActiveTimeline(null);
      return;
    }
    // Switching projects (or unmounting) aborts the previous project's in-flight
    // loads so a slow response can't overwrite the newly-selected project's data.
    const controller = new AbortController();
    const { signal } = controller;
    refreshAssetsRef.current?.(activeProject.id, { signal });
    refreshCharactersRef.current?.(activeProject.id, { signal });
    refreshLorasRef.current?.(activeProject.id, { signal });
    refreshPresetsRef.current?.(activeProject.id, { signal });
    refreshPromptBatchesRef.current?.(activeProject.id, { signal });
    refreshTrainingDatasetsRef.current?.(activeProject.id, { signal });
    refreshPersonTracksRef.current?.(activeProject.id, { signal });
    refreshTimelinesRef.current?.(activeProject.id, { signal });
    return () => controller.abort();
  }, [activeProject?.id, ready, token]);

  // sc-11231 (F-037): useJobEvents captures whichever callback existed at subscribe time
  // (its SSE effect deps are only [access.authRequired, ready, token]), so this MUST have a
  // stable identity — else the live stream keeps calling a stale closure. It reads no props
  // directly, only the two live refs (kept in sync by the effects above), so an empty-dep
  // useCallback is both stable and always current — the `stableRefreshData` ref-delegation
  // pattern used elsewhere in this file, expressed inline here since the whole body is refs.
  const hasVisibleLocalFailure = useCallback((job) => {
    const active = activeViewRef.current;
    const localIds = localGenerationJobIdsRef.current;
    if (active === "Image" && localIds.image.includes(job.id)) {
      return true;
    }
    if (active === "Video" && localIds.video.includes(job.id)) {
      return true;
    }
    if (active === "Document" && localIds.document.includes(job.id)) {
      return true;
    }
    return active === "Models" && job.type === "model_download";
  }, []);

  // Live job/worker/queue SSE stream, extracted to a hook (sc-9750). The handlers reach
  // back into App state through the identity-stable setters/refs/callbacks passed here;
  // the hook re-subscribes only on [access.authRequired, ready, token] (see useJobEvents).
  useJobEvents({
    access,
    ready,
    token,
    setJobs,
    setWorkers,
    setQueueSummary,
    setLatestGenerationSetId,
    setError,
    pushNotice,
    dismissNoticeKind,
    generatedAssetRefreshesRef,
    refreshAssetsRef,
    refreshDataRef,
    refreshDataWithLoraOverlayRef,
    refreshPersonTracksRef,
    activeProjectRef,
    enqueueTimelineGenerationApply,
    hasVisibleLocalFailure,
  });

  // Survivor sweep for orphaned Image-Editor scratch ops (sc-8850). Runs on every `jobs`
  // change (SSE ticks, initial load, etc.): purges the scratch + result assets of any
  // tracked editor AI-op whose job has terminated and whose editor is no longer claiming
  // it (unmounted mid-job). The editor's own in-component watcher handles the mounted
  // case and releases its claim; this catches everything the unmounted editor left behind.
  useEffect(() => {
    jobsRef.current = jobs;
    editorScratchRegistry.sweep(jobs);
  }, [jobs, editorScratchRegistry]);

  async function refreshData() {
    const fetchInitial = async (label, path, fallback, optional = false) => {
      try {
        return { label, value: await apiFetch(path, token), error: "" };
      } catch (err) {
        return { label, value: fallback, error: optional ? "" : `${label}: ${err.message}` };
      }
    };
    const [
      projectsResult,
      jobsResult,
      workersResult,
      modelsResult,
      lorasResult,
      presetsResult,
      trainingTargetsResult,
      trainingPresetsResult,
      promptBatchesResult,
    ] =
      await Promise.all([
        fetchInitial("Projects", "/api/v1/projects", []),
        fetchInitial("Jobs", "/api/v1/jobs", []),
        fetchInitial("Workers", "/api/v1/workers", []),
        fetchInitial("Models", "/api/v1/models", []),
        fetchInitial("LoRAs", "/api/v1/loras", []),
        fetchInitial("Presets", "/api/v1/recipe-presets", [], true),
        fetchInitial("Training targets", "/api/v1/training/targets", { schemaVersion: 1, targets: [] }),
        fetchInitial("Training presets", "/api/v1/training/presets", { schemaVersion: 1, presets: [] }),
        fetchInitial("Prompt batches", "/api/v1/prompt-batches", [], true),
      ]);
    // Mac UI gating (sc-3486): optional + non-fatal — a fetch failure leaves gating inert.
    fetchInitial("Mac capabilities", "/api/v1/capabilities/mac", DEFAULT_MAC_CAPABILITIES, true)
      .then((result) => setMacCapabilities(result.value ?? DEFAULT_MAC_CAPABILITIES))
      .catch(() => {});
    const projectItems = projectsResult.value;
    setProjects(projectItems);
    setProjectsLoaded(true);
    setActiveProject((current) => current ?? projectItems[0] ?? null);
    setJobs((current) => mergeFreshJobs(current, jobsResult.value));
    setWorkers(workersResult.value.sort(sortWorkers));
    setQueueSummary(null);
    setModels(modelsResult.value);
    setLoras(lorasResult.value);
    setPresets(presetsResult.value);
    setPromptBatches(promptBatchesResult.value);
    setTrainingTargets(trainingTargetsResult.value);
    setTrainingTargetsError(trainingTargetsResult.error);
    setTrainingPresets(trainingPresetsResult.value);
    setTrainingPresetsError(trainingPresetsResult.error);
    setError(
      [
        projectsResult,
        jobsResult,
        workersResult,
        modelsResult,
        lorasResult,
        presetsResult,
        trainingTargetsResult,
        trainingPresetsResult,
      ]
        .map((result) => result.error)
        .filter(Boolean)
        .join("; "),
    );
  }

  async function refreshAssets(projectId = activeProject?.id, { signal } = {}) {
    if (!projectId) {
      return;
    }
    try {
      const items = await apiFetch(`/api/v1/projects/${projectId}/assets?includeRejected=true&includeTrashed=true`, token, { signal });
      // sc-8858: an SSE-triggered refresh for the just-active project can resolve
      // after the user switches away; committing then would clobber the new
      // project's assets with the old one's. Drop the stale response — mirrors
      // refreshTimelines' guard (useTimelines.js).
      if (activeProjectRef.current?.id && activeProjectRef.current.id !== projectId) {
        return;
      }
      setAssets(items);
      const defaultAsset = items.find((asset) => !asset.status?.trashed && !asset.status?.rejected) ?? items[0] ?? null;
      setSelectedAssetId((current) => current ?? defaultAsset?.id ?? null);
      setError("");
    } catch (err) {
      if (isAbortError(err)) return;
      setError(err.message);
    }
  }

  function refreshDataWithLoraOverlay(projectId = activeProjectRef.current?.id) {
    refreshData()
      .then(() => {
        if (projectId) {
          refreshLoras(projectId);
        }
      })
      .catch(() => {});
  }

  // sc-8940: Publish the latest refresh closures into their refs from a post-commit
  // effect rather than the render body — React documents render-body ref mutation as
  // unsafe (a discarded concurrent/StrictMode render could leave a ref pointing at an
  // uncommitted closure). No dep array means this runs on every commit, so the refs
  // always hold the newest *committed* body (fresh token / activeProject). We use
  // useLayoutEffect (not useEffect) so the assignment flushes before every passive
  // effect on the same commit: the consumers of these refs (the [ready,token] load
  // effect, the project-switch effect, the SSE handler) are all plain useEffects/event
  // handlers, and React runs all layout effects before any passive effect — preserving
  // the original ordering where consumers saw the fresh closure.
  useLayoutEffect(() => {
    refreshDataRef.current = refreshData;
    refreshAssetsRef.current = refreshAssets;
    refreshCharactersRef.current = refreshCharacters;
    refreshLorasRef.current = refreshLoras;
    refreshPresetsRef.current = refreshPresets;
    refreshPromptBatchesRef.current = refreshPromptBatches;
    refreshTrainingDatasetsRef.current = refreshTrainingDatasets;
    refreshPersonTracksRef.current = refreshPersonTracks;
    refreshTimelinesRef.current = refreshTimelines;
    refreshDataWithLoraOverlayRef.current = refreshDataWithLoraOverlay;
  });

  // saveToken / lockRemote (the remote-browser login + lock affordances, epic 4484
  // story 7) live in useAccessGate now (sc-9750) and are destructured above.

  async function completeSetupWizard() {
    try {
      await tauriInvoke("complete_setup");
    } catch {
      // Persisting the marker failed; still dismiss the wizard so the user isn't
      // trapped. Worst case it re-appears next launch.
    }
    setSetupCompleted(true);
  }

  async function createProject(name) {
    const trimmed = String(name ?? "").trim();
    if (!trimmed) {
      return null;
    }
    try {
      const created = await apiFetch("/api/v1/projects", token, {
        method: "POST",
        body: JSON.stringify({ name: trimmed }),
      });
      setProjects((items) => [created, ...items.filter((item) => item.id !== created.id)]);
      // Switching to the new workspace resets keep-alive screens — including a Data Sets draft.
      // Route through the guard so a dirty draft prompts (consistent with picking an EXISTING
      // project) instead of silently wiping it (sc-11970). On cancel: stay on the current
      // project + view, draft intact; the created workspace remains in the list to open later.
      const switched = await requestProjectSwitch(created);
      if (switched) {
        setActiveView("Image");
      }
      setError("");
      return created;
    } catch (err) {
      setError(err.message);
      return null;
    }
  }

  const createPlaceholderJob = useCallback(
    async (event) => {
      event.preventDefault();
      try {
        await apiFetch("/api/v1/jobs", token, {
          method: "POST",
          body: JSON.stringify({
            type: "placeholder",
            projectId: activeProject?.id ?? null,
            projectName: activeProject?.name ?? null,
            requestedGpu,
            payload: {
              prompt: jobPrompt,
              createdFrom: activeView,
            },
          }),
        });
        setActiveView("Queue");
        setError("");
      } catch (err) {
        setError(err.message);
      }
    },
    [token, activeProject, requestedGpu, jobPrompt, activeView],
  );

  const createImageJob = useCallback(
    async (payload) => {
      if (!activeProject) {
        setError("Create or open a project first.");
        return null;
      }
      try {
        const job = await apiFetch("/api/v1/image/jobs", token, {
          method: "POST",
          body: JSON.stringify({
            ...payload,
            projectId: activeProject.id,
            projectName: activeProject.name,
            requestedGpu,
          }),
        });
        setJobs((items) => upsertJobNewest(items, job));
        setError("");
        return job;
      } catch (err) {
        setError(err.message);
        return null;
      }
    },
    [token, activeProject, requestedGpu],
  );

  // Standalone video upscale (epic 4811 / sc-4816): the net-new `video_upscale` job runs
  // on the generic /api/v1/jobs endpoint (like image_upscale), not the generation video
  // endpoint. `payload` carries { sourceAssetId, factor, engine, softness, model, displayName }.
  const createVideoUpscaleJob = useCallback(
    async (payload) => {
      if (!activeProject) {
        setError("Create or open a project first.");
        return null;
      }
      try {
        const job = await apiFetch("/api/v1/jobs", token, {
          method: "POST",
          body: JSON.stringify({
            type: "video_upscale",
            projectId: activeProject.id,
            projectName: activeProject.name,
            requestedGpu,
            payload: { ...payload, projectId: activeProject.id },
          }),
        });
        setJobs((items) => upsertJobNewest(items, job));
        setError("");
        return job;
      } catch (err) {
        setError(err.message);
        return null;
      }
    },
    [token, activeProject, requestedGpu],
  );

  // Refine a prompt via the prompt_refine worker job: POST creates the job, then
  // poll until it reaches a terminal state and return the rewritten prompt. Project-
  // independent (no activeProject gate); throws on failure so the studio can surface
  // the message inline without clobbering the original prompt.
  const refinePrompt = useCallback(
    ({ prompt, modelId, workflow, guide, signal }) =>
      pollJobToCompletion({
        createPath: "/api/v1/prompts/refine",
        body: { prompt, modelId, workflow, guide },
        deadlineMs: 120000,
        resolveResult: (job) => {
          const refined = job.result?.refinedPrompt;
          if (!refined) {
            throw new Error("Refinement returned an empty prompt.");
          }
          return refined;
        },
        signal,
        token,
        startError: "Could not start prompt refinement.",
        failureError: "Prompt refinement failed.",
        timeoutError: "Prompt refinement timed out. Is the refinement runtime running?",
      }),
    [token],
  );

  // Magic-prompt expansion (epic 4725, sc-5997): same `prompts/refine` endpoint + native
  // utility model, but `task: "magic_prompt"` swaps in Ideogram's caption system prompt and
  // returns a JSON caption string (the caller parses + validates it). Reuses the refine job's
  // poll-to-completion contract; captions can take longer, so the deadline is generous.
  const magicPrompt = useCallback(
    ({ prompt, modelId, aspectRatio, guide, signal }) =>
      pollJobToCompletion({
        createPath: "/api/v1/prompts/refine",
        body: { prompt, modelId, task: "magic_prompt", aspectRatio, guide },
        deadlineMs: 180000,
        resolveResult: (job) => {
          const caption = job.result?.refinedPrompt;
          if (!caption) {
            throw new Error("Magic-prompt returned an empty caption.");
          }
          return caption;
        },
        signal,
        token,
        startError: "Could not start magic-prompt.",
        failureError: "Magic-prompt failed.",
        timeoutError: "Magic-prompt timed out. Is the refinement runtime running?",
      }),
    [token],
  );

  // Reference-image → JSON caption (epic 8102, sc-8108): same `prompts/refine` endpoint + poll-to-
  // completion contract as magic-prompt, but `task: "image_caption"` drives the worker's `core_llm`
  // VISION path (sc-8105). The reference image is supplied as a project `sourceAssetId` (+ `projectId`),
  // which the API resolves to a confined on-disk `imagePath`; the vision model is named by its HF repo
  // string in `model` (the worker resolves it by repo, like the refiner). The caller parses + validates
  // the returned JSON with `parseVisionCaption` (aspect_ratio stripped, bboxes KEPT). C1: the image is
  // consumed only to produce the caption — it is NEVER passed to generation as img2img conditioning.
  const imageCaption = useCallback(
    ({ sourceAssetId, sourceAssetIds, projectId, model, signal }) =>
      pollJobToCompletion({
        createPath: "/api/v1/prompts/refine",
        // A mood board (epic 8588, sc-8595) sends `sourceAssetIds` (plural); the API synthesizes ONE
        // caption from the shared aesthetic. A single reference keeps the scalar `sourceAssetId`.
        body: { task: "image_caption", sourceAssetId, sourceAssetIds, projectId, model },
        deadlineMs: 180000,
        resolveResult: (job) => {
          const caption = job.result?.refinedPrompt;
          if (!caption) {
            throw new Error("Image captioning returned an empty caption.");
          }
          return caption;
        },
        signal,
        token,
        startError: "Could not start image captioning.",
        failureError: "Image captioning failed.",
        timeoutError: "Image captioning timed out. Is the captioning runtime running?",
      }),
    [token],
  );

  // Reference-image → plain-text description (epic 8203, sc-8208): the sibling of `imageCaption` for
  // NON-structured text-to-image models. Same `prompts/refine` endpoint + poll contract, but
  // `task: "image_describe"` drives the worker's prose/tags vision path (sc-8204/8205). `captionStyle`
  // (from the catalog, default prose) selects natural-language prose vs booru tags. Resolves to the raw
  // text the caller drops into the prompt box. C1: the image is consumed only to produce the prompt — it
  // is NEVER passed to generation as img2img conditioning.
  const imageDescribe = useCallback(
    ({ sourceAssetId, sourceAssetIds, projectId, model, captionStyle, signal }) =>
      pollJobToCompletion({
        createPath: "/api/v1/prompts/refine",
        // A mood board (epic 8588, sc-8595) sends `sourceAssetIds` (plural); the API synthesizes ONE
        // prompt from the aesthetic they share. A single reference keeps the scalar `sourceAssetId`.
        body: { task: "image_describe", sourceAssetId, sourceAssetIds, projectId, model, captionStyle },
        deadlineMs: 180000,
        resolveResult: (job) => {
          const description = job.result?.refinedPrompt;
          if (!description) {
            throw new Error("Image description returned empty text.");
          }
          return description;
        },
        signal,
        token,
        startError: "Could not start image description.",
        failureError: "Image description failed.",
        timeoutError: "Image description timed out. Is the captioning runtime running?",
      }),
    [token],
  );

  // On-demand "compare image to another" likeness (epic 4406, sc-4415): score a CANDIDATE asset
  // against a SOURCE identity reference asset through the shared SCRFD+ArcFace scorer in the worker.
  // Same poll-to-completion contract as the refine/describe runners, but `/face-likeness/compare`
  // enqueues the GPU-routed `face_likeness_compare` job and returns the full result object
  // (`{ score, detected, method, sourceRef, reason? }`) so the caller can render the band + N/A framing
  // via `classifyLikeness` / `LikenessBadge`. Non-fatal end to end: a no-face / non-frontal candidate
  // is an honest detected:false result (NOT an error); only a hard failure throws.
  const compareFaceLikeness = useCallback(
    ({ sourceAssetId, candidateAssetId, projectId, signal }) =>
      pollJobToCompletion({
        createPath: "/api/v1/face-likeness/compare",
        body: { sourceAssetId, candidateAssetId, projectId },
        deadlineMs: 180000,
        resolveResult: (job) => {
          // A completed compare always carries a result block (a detected:false N/A is a valid,
          // non-error outcome). Surface the whole block so the UI can band/N-A it.
          if (!job.result) {
            throw new Error("Likeness compare returned no result.");
          }
          return job.result;
        },
        signal,
        token,
        startError: "Could not start the likeness compare.",
        failureError: "Likeness compare failed.",
        timeoutError: "Likeness compare timed out. Is the worker running?",
      }),
    [token],
  );

  const createVqaJob = useCallback(
    async (asset, question, maxNewTokens) => {
      if (!activeProject) {
        setError("Create or open a project first.");
        return null;
      }
      try {
        const job = await apiFetch("/api/v1/image/vqa/jobs", token, {
          method: "POST",
          body: JSON.stringify({
            projectId: activeProject.id,
            projectName: activeProject.name,
            sourceAssetId: asset.id,
            question,
            maxNewTokens,
            requestedGpu,
          }),
        });
        setJobs((items) => upsertJobNewest(items, job));
        setError("");
        return job;
      } catch (err) {
        setError(err.message);
        return null;
      }
    },
    [token, activeProject, requestedGpu],
  );

  const createInterleaveJob = useCallback(
    async (payload) => {
      if (!activeProject) {
        setError("Create or open a project first.");
        return null;
      }
      try {
        const job = await apiFetch("/api/v1/image/interleave/jobs", token, {
          method: "POST",
          body: JSON.stringify({
            ...payload,
            projectId: activeProject.id,
            projectName: activeProject.name,
            requestedGpu,
          }),
        });
        setJobs((items) => upsertJobNewest(items, job));
        setError("");
        return job;
      } catch (err) {
        setError(err.message);
        return null;
      }
    },
    [token, activeProject, requestedGpu],
  );

  const rememberLocalGenerationJob = useCallback((kind, job) => {
    if (!job?.id) {
      return;
    }
    setLocalGenerationJobIds((current) => ({
      ...current,
      // Remember every submitted run (newest first, capped) so running and queued
      // runs stack in the studio instead of the latest run evicting the previous one.
      [kind]: [job.id, ...current[kind].filter((id) => id !== job.id)].slice(0, localJobStackLimit),
    }));
  }, []);

  const sendAssetToImage = useCallback((asset, mode = null) => {
    if (!asset) {
      return;
    }
    setSelectedAssetId(asset.id);
    if (mode) {
      setStudioLaunch({ id: crypto.randomUUID(), view: "Image", assetId: asset.id, mode });
    }
    setActiveView("Image");
  }, []);

  // Open Image Studio in edit mode with this image as the source, preselecting the
  // family-matched edit model when possible. sc-8730: this is the model-based path,
  // no longer wired to the preview Edit button — it stays on the context so S4
  // (sc-8729) can offer it as an "Edit in > Image Studio" context-menu item.
  const sendAssetToImageEdit = useCallback((asset) => {
    if (!asset) {
      return;
    }
    setSelectedAssetId(asset.id);
    setStudioLaunch({
      id: crypto.randomUUID(),
      view: "Image",
      assetId: asset.id,
      mode: "edit_image",
      model: editModelForAsset(asset, imageModels),
    });
    setActiveView("Image");
  }, [imageModels]);

  // sc-8730: "Edit" from the fullscreen preview now opens the Image Editor canvas
  // (crop/upscale/refine) with this asset loaded via the editor's openAsset. This is
  // the model-free path; the model-based Image Studio edit_image path lives in
  // sendAssetToImageEdit above (kept for the S4 "Edit in > Image Studio" menu item).
  const sendAssetToImageEditor = useCallback((asset) => {
    if (!asset) {
      return;
    }
    setSelectedAssetId(asset.id);
    setEditorLaunch({ id: crypto.randomUUID(), assetId: asset.id });
    setActiveView("ImageEditor");
  }, []);

  // The editor consumes editorLaunch and calls this to drop it, so navigating away
  // and back into the Image Editor without a fresh launch doesn't re-open a stale asset.
  const clearEditorLaunch = useCallback(() => setEditorLaunch(null), []);

  function recipeForAsset(asset) {
    return asset?.generationSet?.recipe ?? asset?.recipe ?? null;
  }

  function sendAssetRecipeToImage(asset, options = {}) {
    const recipe = recipeForAsset(asset);
    if (!asset || !recipe) {
      return;
    }
    // Keep-seed replays THIS image's own seed for a byte-for-byte rerun (e.g. to
    // reproduce and upscale with PiD). assetSeed prefers the per-asset recipe over the
    // set's shared base seed, so reuse honors the exact image the user is viewing.
    // Null → Image Studio leaves the seed random (a close variation), the default.
    const seed = assetSeed(asset);
    const replaySeed = options.keepSeed && seed != null && seed !== "" ? seed : null;
    setSelectedAssetId(asset.id);
    closePreview();
    setStudioLaunch({
      id: crypto.randomUUID(),
      view: "Image",
      assetId: asset.id,
      sourceAssetId: asset.lineage?.sourceAssetId ?? null,
      recipe,
      replaySeed,
    });
    setActiveView("Image");
  }

  // sc-10516: launch a saved preset straight into the studio that can run it.
  //
  // The id alone is not enough: a studio only resolves `selectedPresetId` against its
  // `availablePresets`, which is filtered by the current mode AND model
  // (generationStudio.jsx). So the launch carries the preset's model and sub-mode too,
  // and the studio sets all three together. `presetId` and `recipe` are mutually
  // exclusive — a recipe launch keeps clearing the preset.
  const sendPresetToStudio = useCallback((preset) => {
    if (!preset?.id) {
      return;
    }
    // General (model-agnostic) presets don't pin a model/mode — they layer onto whatever the
    // studio is on (epic 11949). Launching one just toggles it into the general stack; default
    // to Image Studio, and it stays active if the user switches to Video. Model presets keep
    // carrying model + sub-mode so they resolve in the target studio (sc-10516).
    if (preset.kind === "general") {
      setStudioLaunch({ id: crypto.randomUUID(), view: "Image", presetGeneralId: preset.id });
      setActiveView("Image");
      return;
    }
    const view = workflowModelType(preset.workflow) === "video" ? "Video" : "Image";
    setStudioLaunch({
      id: crypto.randomUUID(),
      view,
      presetId: preset.id,
      presetModel: preset.model ?? null,
      presetMode: preset.defaults?.mode ?? preset.workflow,
    });
    setActiveView(view);
  }, []);

  const sendAssetToVideo = useCallback((asset, mode = null) => {
    if (!asset) {
      return;
    }
    setSelectedAssetId(asset.id);
    if (mode) {
      setStudioLaunch({ id: crypto.randomUUID(), view: "Video", assetId: asset.id, mode });
    }
    setActiveView("Video");
  }, []);

  const sendCharacterToImage = useCallback((character, lookId = null, referenceAssetId = null) => {
    if (!character) {
      return;
    }
    setStudioLaunch({
      id: crypto.randomUUID(),
      view: "Image",
      characterId: character.id,
      lookId,
      referenceAssetId,
      mode: "character_image",
    });
    setActiveView("Image");
  }, []);

  const sendCharacterToVideo = useCallback((character, lookId = null) => {
    if (!character) {
      return;
    }
    setStudioLaunch({ id: crypto.randomUUID(), view: "Video", characterId: character.id, lookId, mode: "text_to_video" });
    setActiveView("Video");
  }, []);

  // sc-2022: open a specific dataset in the Dataset editor (Character Studio's
  // "Open" action on an associated dataset). The editor consumes studioLaunch.
  const openDatasetInLibrary = useCallback((datasetId) => {
    if (!datasetId) {
      return;
    }
    setStudioLaunch({ id: crypto.randomUUID(), view: "LibraryDataSets", datasetId });
    setActiveView("LibraryDataSets");
  }, []);

  const updateAssetStatus = useCallback(
    async (asset, changes) => {
      try {
        const updated = await apiFetch(`/api/v1/projects/${asset.projectId}/assets/${asset.id}/status`, token, {
          method: "PATCH",
          body: JSON.stringify(changes),
        });
        setAssets((items) => items.map((item) => (item.id === updated.id ? updated : item)));
        setError("");
      } catch (err) {
        setError(err.message);
      }
    },
    [token],
  );

  const updateAssetTags = useCallback(
    async (asset, tags) => {
      try {
        const updated = await apiFetch(`/api/v1/projects/${asset.projectId}/assets/${asset.id}/tags`, token, {
          method: "PATCH",
          body: JSON.stringify({ tags }),
        });
        setAssets((items) => items.map((item) => (item.id === updated.id ? updated : item)));
        setError("");
      } catch (err) {
        setError(err.message);
      }
    },
    [token],
  );

  // A move endpoint returns the whole moved upscale-fold group (sc-10205) as an
  // array, requested asset first — merge every member into state or the hidden
  // fold-mate lingers with a stale origin until the next full refresh.
  const mergeMovedAssets = useCallback((moved) => {
    const list = Array.isArray(moved) ? moved : [moved];
    const byId = new Map(list.map((item) => [item.id, item]));
    setAssets((items) => items.map((item) => byId.get(item.id) ?? item));
    return list[0];
  }, []);

  // Promote a character asset into the Main Asset Library (sc-8341): a true move —
  // the backend flips origin + detaches the character, so refresh characters too.
  const moveAssetToLibrary = useCallback(
    async (asset) => {
      try {
        const moved = await apiFetch(`/api/v1/projects/${asset.projectId}/assets/${asset.id}/move-to-library`, token, {
          method: "POST",
        });
        const updated = mergeMovedAssets(moved);
        refreshCharactersRef.current?.(asset.projectId);
        setError("");
        return updated;
      } catch (err) {
        setError(err.message);
        throw err;
      }
    },
    [token, mergeMovedAssets],
  );

  // Move an asset into a character's assets (sc-10200): the true-move twin of
  // moveAssetToLibrary, NOT a link — the backend flips origin to character_studio
  // (so the asset leaves the Library) and re-anchors the character association
  // without touching the curated references[] ("Approved set"). Refresh characters
  // so any curated reference the move detached drops out of their panels too.
  const moveAssetToCharacter = useCallback(
    async (asset, characterId) => {
      try {
        const moved = await apiFetch(`/api/v1/projects/${asset.projectId}/assets/${asset.id}/move-to-character`, token, {
          method: "POST",
          body: JSON.stringify({ characterId }),
        });
        const updated = mergeMovedAssets(moved);
        refreshCharactersRef.current?.(asset.projectId);
        setError("");
        return updated;
      } catch (err) {
        setError(err.message);
        throw err;
      }
    },
    [token, mergeMovedAssets],
  );

  const deleteAsset = useCallback(
    async (asset) => {
      try {
        await apiFetch(`/api/v1/projects/${asset.projectId}/assets/${asset.id}`, token, { method: "DELETE" });
        setAssets((items) =>
          items.map((item) =>
            item.id === asset.id ? { ...item, status: { ...item.status, trashed: true } } : item,
          ),
        );
        setError("");
      } catch (err) {
        setError(err.message);
      }
    },
    [token],
  );

  const purgeAsset = useCallback(
    async (asset) => {
      try {
        let result = await apiFetch(`/api/v1/projects/${asset.projectId}/assets/${asset.id}/purge`, token, { method: "DELETE" });
        // The purge first tries the OS trash (recoverable). If that fails nothing was
        // removed; confirm before falling back to a permanent delete. Routed through the
        // desktop-safe appConfirm (sc-12068) — window.confirm no-ops in the Tauri WebView.
        if (result?.status === "trash_unavailable") {
          const proceed = await appConfirm({
            title: "Move to trash failed",
            message: "Cannot move to trash. Continue to permanently delete.",
            confirmLabel: "Delete permanently",
            cancelLabel: "Cancel",
            tone: "danger",
          });
          if (!proceed) {
            setError("");
            return;
          }
          result = await apiFetch(`/api/v1/projects/${asset.projectId}/assets/${asset.id}/purge?permanent=true`, token, { method: "DELETE" });
        }
        setAssets((items) => items.filter((item) => item.id !== asset.id));
        setSelectedAssetId((current) => (current === asset.id ? null : current));
        setError("");
      } catch (err) {
        setError(err.message);
      }
    },
    [token],
  );
  // Keep the ref current for the App-level scratch-op survivor (sc-8850), which purges
  // from the SSE/sweep path without re-subscribing. Published from a post-commit effect
  // rather than the render body (sc-9641, following sc-8940): render-body ref mutation is
  // unsafe because a discarded concurrent/StrictMode render could leave the ref pointing
  // at an uncommitted closure. useLayoutEffect (no dep array) mirrors the sc-8940 refresh
  // block above so the ref always holds the newest *committed* purgeAsset, and flushes
  // before any passive effect on the same commit. `purgeAsset` is a useCallback([token]),
  // so this only rewrites when the token changes. It lives here (not in the sc-8940 block)
  // because purgeAsset is defined after it; the sole read site (the scratch registry's
  // purge callback) fires from the SSE/sweep path — post-commit — never during render.
  useLayoutEffect(() => {
    purgeAssetRef.current = purgeAsset;
  });

  const importAsset = useCallback(
    async (file, options = {}) => {
      if (!activeProject || !file) {
        const error = new Error("Create or open a project first.");
        if (options.throwOnError) {
          throw error;
        }
        setError(error.message);
        return;
      }
      const body = new FormData();
      body.append("file", file);
      // Optional lineage for derived imports (Image Editor Save, sc-2434): link the
      // new asset to the source it was opened from + record the edit-chain provenance.
      if (options.sourceAssetId) body.append("sourceAssetId", options.sourceAssetId);
      if (options.provenance) body.append("provenance", JSON.stringify(options.provenance));
      try {
        const imported = await apiFetch(`/api/v1/projects/${activeProject.id}/assets`, token, {
          method: "POST",
          body,
        });
        setAssets((items) => [imported, ...items.filter((item) => item.id !== imported.id)]);
        setSelectedAssetId(imported.id);
        setError("");
        return imported;
      } catch (err) {
        if (options.throwOnError) {
          throw err;
        }
        setError(err.message);
        return null;
      }
    },
    [token, activeProject],
  );

  const jobAction = useCallback(
    async (job, action, options = {}) => {
      try {
        const path = action === "duplicate" ? `/api/v1/jobs/${job.id}/duplicate` : `/api/v1/jobs/${job.id}/${action}`;
        const body =
          action === "duplicate"
            ? { payloadChanges: { duplicatedAt: new Date().toISOString() } }
            : (options.body ?? {});
        const updatedJob = await apiFetch(path, token, { method: "POST", body: JSON.stringify(body) });
        setJobs((items) => upsertJobNewest(items, updatedJob));
        setError("");
      } catch (err) {
        setError(err.message);
      }
    },
    [token],
  );

  // Clear completed items from the queue (issue #1556 / sc-12231). The server
  // soft-hides every terminal job (keeping Generation Stats + generated assets
  // intact) and returns the cleared ids; prune exactly those from the live jobs
  // list so the queue empties immediately without waiting for a refetch. Scoped
  // to the active project filter so it clears exactly what the operator sees
  // ("all" clears every project).
  const clearCompletedJobs = useCallback(
    async (projectId) => {
      try {
        const body = projectId && projectId !== "all" ? { projectId } : {};
        const response = await apiFetch("/api/v1/jobs/clear", token, {
          method: "POST",
          body: JSON.stringify(body),
        });
        const clearedIds = new Set(response?.clearedIds ?? []);
        if (clearedIds.size) {
          setJobs((items) => items.filter((job) => !clearedIds.has(job.id)));
        }
        setError("");
      } catch (err) {
        setError(err.message);
      }
    },
    [token],
  );

  // Clear a single completed item from the queue (issue #1556 / sc-12231) — the
  // per-card "×". The server soft-hides just this terminal job; drop it from the
  // live list on success so the card disappears immediately.
  const clearJob = useCallback(
    async (job) => {
      try {
        await apiFetch(`/api/v1/jobs/${job.id}/clear`, token, {
          method: "POST",
          body: JSON.stringify({}),
        });
        setJobs((items) => items.filter((item) => item.id !== job.id));
        setError("");
      } catch (err) {
        setError(err.message);
      }
    },
    [token],
  );

  const titleInfo = viewTitles[activeView] ?? { title: activeView, blurb: "" };
  // A keep-alive view is rendered once it is active OR has been visited before
  // (sc-11959): it mounts on first visit and then stays mounted (hidden) thereafter.
  const keepAliveMounted = (view) => activeView === view || visitedKeepAliveViews.has(view);
  // Activity dots only — counts live in the topbar so nav button textContent stays clean.
  const activeIndicators = {
    Editor: timelines.length > 0,
    Queue: queueCounts.active > 0,
  };
  // First-run gate: until at least one workspace exists, replace the studio area
  // with a create prompt so navigation never lands on dead, project-scoped controls.
  const needsFirstProject = authenticated && projectsLoaded && projects.length === 0;
  // Desktop first-run wizard (sc-1473): supersedes the project gate while the
  // completion marker is unset. `null` means we're still reading the marker on
  // desktop — hold the studio/gate back briefly to avoid a flash.
  const setupGateLoading = isDesktopShell && setupCompleted === null;
  const showSetupWizard = isDesktopShell && setupCompleted === false && authenticated;

  // sc-1651 Phase B: shared primitives screens read via useAppContext() instead of
  // drilled props. Screens build any screen-specific wrappers from these (e.g. a
  // send-to-studio action with a mode). Grown one screen at a time as screens convert.
  // sc-4194: memoized so the provider value only changes identity when one of its
  // entries actually changes, instead of being a fresh ~120-key literal on every App
  // render (SSE job/worker/queue ticks re-render App continuously). The actions above
  // and the data-hook actions are useCallback-stable, so this holds across renders
  // that don't change data.
  //
  // sc-8855 (F-053): split into TWO memoized values behind two providers.
  //   - appLiveValue: the high-churn fields derived from the SSE-updated jobs/workers
  //     state (jobs/filteredJobs/*LocalJobs/visibleWorkers/workersById/personReadiness/
  //     gpuOptions). These change identity on nearly every tick.
  //   - appStaticValue: everything else (actions/catalogs/project/presets/characters/
  //     training/timelines/models). Its dependency array MUST NOT reference jobs or
  //     workersById (or anything derived from them) — that exclusion is the whole point:
  //     cold-only consumers reading via useAppStatic() no longer re-render on job ticks.
  // NOTE: each dependency array must mirror the corresponding object below.
  const appLiveValue = useMemo(() => ({
    // Jobs / queue (high-churn — new identity per SSE tick)
    jobs,
    filteredJobs,
    imageLocalJobs,
    videoLocalJobs,
    documentLocalJobs,
    // Workers / GPU (high-churn — new identity per worker SSE tick)
    visibleWorkers,
    workersById,
    personReadiness,
    gpuOptions,
  }), [
    jobs, filteredJobs, imageLocalJobs, videoLocalJobs, documentLocalJobs,
    visibleWorkers, workersById, personReadiness, gpuOptions,
  ]);

  const appStaticValue = useMemo(() => ({
    activeProject,
    mediaAssets,
    setPreviewAsset: openPreview,
    sendAssetToImage,
    sendAssetToVideo,
    activeTimeline,
    timelines,
    selectedTimelineId,
    setSelectedTimelineId,
    setActiveTimeline,
    isActiveTimelineDirty,
    createTimeline,
    saveTimeline,
    exportTimeline,
    extractTimelineFrame,
    queueTimelineVideoJob,
    // Assets / library (sc-1651 Phase B batch 1)
    assets,
    selectedAsset,
    // The RAW selection id (null when nothing is explicitly selected). Studios need it
    // to tell an explicit user selection apart from `selectedAsset`'s assets[0] fallback
    // so a restored source isn't clobbered by the newest asset on a cold restart (sc-11964).
    selectedAssetId,
    setSelectedAssetId,
    deleteAsset,
    purgeAsset,
    moveAssetToLibrary,
    moveAssetToCharacter,
    importAsset,
    updateAssetStatus,
    updateAssetTags,
    latestImageAssets,
    // Job actions (creation/control — stable callbacks, NOT the churning jobs list)
    jobAction,
    clearCompletedJobs,
    clearJob,
    createVqaJob,
    createInterleaveJob,
    // Queue screen (sc-1651 Phase B batch 2)
    createPlaceholderJob,
    jobPrompt,
    setJobPrompt,
    projectFilter,
    setProjectFilter,
    projects,
    // Generation studios (sc-1651 Phase B batch 3)
    createVideoJob,
    createVideoUpscaleJob,
    createImageJob,
    refinePrompt,
    magicPrompt,
    imageCaption,
    imageDescribe,
    compareFaceLikeness,
    latestVideoAssets,
    recentImageAssets,
    recentVideoAssets,
    studioLaunch,
    // sc-8730: Image Editor launch channel + the two Edit paths. sendAssetToImageEditor
    // routes to the editor canvas (FullscreenPreview Edit button); sendAssetToImageEdit
    // routes to Image Studio edit_image (exposed for the S4 sc-8729 context menu).
    editorLaunch,
    clearEditorLaunch,
    sendAssetToImageEditor,
    sendAssetToImageEdit,
    rememberLocalGenerationJob,
    // Person tracks (Video Studio + Replace Person)
    personTracks,
    createPersonDetectionJob,
    createPersonTrackJob,
    saveTrackCorrections,
    // Models / GPU
    imageModels,
    videoModels,
    models,
    // Mac UI gating (sc-3486)
    macCapabilities,
    loras,
    deleteLora,
    updateLora,
    fetchLoraEmbeddedTags,
    deleteModel,
    deleteModelVariant,
    createModelDownloadJob,
    createLoraDownloadJob,
    createModelConvertJob,
    createLoraImportJob,
    createModelImportJob,
    requestedGpu,
    setRequestedGpu,
    // Presets
    presets,
    createPreset,
    updatePreset,
    deletePreset,
    duplicatePreset,
    // Prompt batches (sc-9954, epic 9952)
    promptBatches,
    createPromptBatch,
    updatePromptBatch,
    deletePromptBatch,
    duplicatePromptBatch,
    // Auth (sc-4168): pairing token for screens that call apiFetch directly
    // (Image Editor, Logs, Pose Library, useUserPoseLoader). Empty string when
    // the deployment doesn't require auth.
    token,
    // Training (sc-1651 Phase B batch 7)
    authenticated,
    trainingDatasets,
    trainingDatasetsProjectId,
    trainingDatasetsError,
    loadingTrainingDatasets,
    refreshTrainingDatasets,
    loadTrainingDataset,
    loadTrainingDatasetReadiness,
    setTrainingDatasetItemQualityAck,
    createTrainingDataset,
    uploadTrainingDatasetItem,
    updateTrainingDataset,
    batchRenameTrainingDataset,
    writeTrainingDatasetCaptionSidecars,
    createTrainingDatasetCaptionJob,
    createTrainingDatasetUpscaleJob,
    createTrainingDatasetAnalysisJob,
    createTrainingDatasetFaceAnalysisJob,
    smartCropTrainingDataset,
    stripExifTrainingDataset,
    createTrainingJob,
    trainingPresets,
    trainingPresetsError,
    trainingTargets,
    trainingTargetsError,
    // Navigation
    setActiveView,
    registerLeaveGuard,
    // Unsaved-draft guard consulted before a project switch (sc-11970)
    registerProjectSwitchGuard,
    // Image-Editor scratch-op survivor coordination (sc-8850)
    trackEditorScratchOp,
    releaseEditorScratchOp,
    registerEditorScratchClaim,
    // Characters
    characters,
    createCharacter,
    updateCharacter,
    archiveCharacter,
    unarchiveCharacter,
    listArchivedCharacters,
    addCharacterReference,
    updateCharacterReference,
    removeCharacterReference,
    createCharacterLook,
    updateCharacterLook,
    deleteCharacterLook,
    attachCharacterLora,
    updateCharacterLora,
    detachCharacterLora,
    createCharacterTestJob,
    sendCharacterToImage,
    sendCharacterToVideo,
    sendPresetToStudio,
    openDatasetInLibrary,
    // Global theme (sc-10244): exposed so the Image Editor's top-bar toggle
    // drives the app-wide data-theme rather than a screen-local override.
    theme,
    changeTheme,
  }), [
    activeProject, mediaAssets, openPreview, sendAssetToImage, sendAssetToVideo,
    activeTimeline, timelines, selectedTimelineId, setSelectedTimelineId, setActiveTimeline, isActiveTimelineDirty,
    createTimeline, saveTimeline, exportTimeline, extractTimelineFrame, queueTimelineVideoJob,
    assets, selectedAsset, selectedAssetId, setSelectedAssetId, deleteAsset, purgeAsset, moveAssetToLibrary, moveAssetToCharacter, importAsset,
    updateAssetStatus, updateAssetTags, latestImageAssets,
    jobAction, clearCompletedJobs, clearJob, createVqaJob, createInterleaveJob, createPlaceholderJob,
    jobPrompt, setJobPrompt, projectFilter, setProjectFilter, projects,
    createVideoJob, createVideoUpscaleJob, createImageJob, refinePrompt, magicPrompt, imageCaption, imageDescribe, compareFaceLikeness, latestVideoAssets, recentImageAssets,
    recentVideoAssets, studioLaunch,
    editorLaunch, clearEditorLaunch, sendAssetToImageEditor, sendAssetToImageEdit,
    rememberLocalGenerationJob, personTracks, createPersonDetectionJob,
    createPersonTrackJob, saveTrackCorrections, imageModels, videoModels, models, macCapabilities,
    loras, deleteLora, updateLora, fetchLoraEmbeddedTags, deleteModel, deleteModelVariant, createModelDownloadJob, createLoraDownloadJob, createModelConvertJob,
    createLoraImportJob, createModelImportJob, requestedGpu, setRequestedGpu,
    presets, createPreset, updatePreset, deletePreset, duplicatePreset, token, authenticated,
    promptBatches, createPromptBatch, updatePromptBatch, deletePromptBatch, duplicatePromptBatch,
    trainingDatasets, trainingDatasetsProjectId, trainingDatasetsError, loadingTrainingDatasets,
    refreshTrainingDatasets, loadTrainingDataset, loadTrainingDatasetReadiness, setTrainingDatasetItemQualityAck, createTrainingDataset, uploadTrainingDatasetItem,
    updateTrainingDataset, batchRenameTrainingDataset, writeTrainingDatasetCaptionSidecars,
    createTrainingDatasetCaptionJob, createTrainingDatasetUpscaleJob, createTrainingDatasetAnalysisJob, createTrainingDatasetFaceAnalysisJob, smartCropTrainingDataset, stripExifTrainingDataset, createTrainingJob, trainingPresets, trainingPresetsError,
    trainingTargets, trainingTargetsError, setActiveView, registerLeaveGuard, registerProjectSwitchGuard,
    trackEditorScratchOp, releaseEditorScratchOp, registerEditorScratchClaim, characters,
    createCharacter, updateCharacter, archiveCharacter, unarchiveCharacter, listArchivedCharacters,
    addCharacterReference, updateCharacterReference,
    removeCharacterReference, createCharacterLook, updateCharacterLook, deleteCharacterLook,
    attachCharacterLora, updateCharacterLora, detachCharacterLora, createCharacterTestJob,
    sendCharacterToImage, sendCharacterToVideo, sendPresetToStudio, openDatasetInLibrary, theme, changeTheme,
  ]);

  return (
    <AppStaticContext.Provider value={appStaticValue}>
    <AppLiveContext.Provider value={appLiveValue}>
    <main className="app">
      <aside className="sidebar" aria-label="Primary">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            <Logo size={32} />
          </span>
          <div>
            <h1>Scene<span className="light">Works</span></h1>
            <p>Local creative studio</p>
          </div>
        </div>

        <ProjectSwitcher
          activeProject={activeProject}
          disabled={!authenticated}
          onCreate={createProject}
          onSelect={selectProject}
          projects={projects}
        />

        {navSections.map((section) => (
          <div className="sidebar-section" key={section.label}>
            <div className="sidebar-section-title">{section.label}</div>
            <nav className="nav-list">
              {section.items.map((item) => {
                const IconComponent = item.icon;
                const active = activeIndicators[item.id];
                const label = item.label ?? item.id;
                return (
                  <button
                    className={activeView === item.id ? "nav-item active" : "nav-item"}
                    key={item.id}
                    onClick={() => navTo(item.id)}
                    title={label}
                    type="button"
                  >
                    <IconComponent />
                    <span className="nav-label">{label}</span>
                    {active ? <span aria-hidden="true" className="nav-pulse" /> : null}
                  </button>
                );
              })}
            </nav>
          </div>
        ))}

        {APP_VERSION ? (
          <div className="sidebar-footer">
            <span className="app-version" title={`SceneWorks ${APP_VERSION}`}>
              v{APP_VERSION}
            </span>
          </div>
        ) : null}
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div className="topbar-title">
            <h1>{titleInfo.title}</h1>
            <p>{titleInfo.blurb}</p>
          </div>
          <span className="topbar-spacer" />
          <div className="topbar-status">
            {/* Status collapses to one summary pill (UI-refinement 1d): API health · workers · GPU.
                Clicking through opens the Queue, where the per-worker/job detail lives. */}
            <button
              className={health?.status === "ok" ? "status-pill status-summary" : "status-pill status-summary warning"}
              onClick={() => setActiveView("Queue")}
              title="Workers and GPU activity — open the Queue for detail"
              type="button"
            >
              <StatusDot ok={health?.status === "ok"} />
              {health?.status === "ok" ? "Ready" : "API offline"}
              <span className="status-summary-sep">·</span>
              {visibleWorkers.length} worker{visibleWorkers.length === 1 ? "" : "s"}
              <span className="status-summary-sep">·</span>
              {gpuOptions.length > 1 ? `${gpuOptions.length - 1} GPU` : "GPU auto"}
              <Icon.ChevDown className="status-summary-caret" size={13} />
            </button>
            <button className="queue-chip" onClick={() => setActiveView("Queue")} type="button">
              Queue {queueCounts.active}
            </button>
          </div>
          <span className="topbar-divider" aria-hidden="true" />
          <button className="icon-btn" title="Notifications" type="button">
            <Icon.Bell />
          </button>
          <AccentPicker accent={accent} onChange={changeAccent} />
          <button
            className="icon-btn"
            onClick={() => changeTheme(theme === "light" ? "dark" : "light")}
            title={theme === "light" ? "Switch to dark mode" : "Switch to light mode"}
            type="button"
          >
            {theme === "light" ? <Icon.Moon /> : <Icon.Sun />}
          </button>
          {/* Lock/forget the saved password (epic 4484 story 7) — only meaningful in a
              remote browser where a password was entered to unlock the host. */}
          {access.authRequired && !isDesktopShell && token ? (
            <button
              className="icon-btn"
              onClick={lockRemote}
              title="Lock — forget the saved password"
              type="button"
            >
              Lock
            </button>
          ) : null}
        </header>

        {notices.map((notice) => (
          <p className="notice error" key={notice.kind}>{notice.message}</p>
        ))}

        {/* Gate visibility keys off the token STATE (not a render-time localStorage
            read), and the input edits a local draft — never the live token — so
            typing can't flip `authenticated` or fire API/SSE traffic (sc-8808). */}
        {access.authRequired && !isDesktopShell && !token ? (
          <section className="auth-band">
            <form onSubmit={saveToken}>
              <label htmlFor="token">Password</label>
              <div className="form-row">
                <input
                  id="token"
                  onChange={(event) => setPasswordDraft(event.target.value)}
                  placeholder="Enter the access password"
                  type="password"
                  value={passwordDraft}
                />
                <button type="submit">Unlock</button>
              </div>
              {authError ? <p className="notice error">{authError}</p> : null}
            </form>
          </section>
        ) : null}

        {showSetupWizard ? (
          <SetupWizard
            jobs={jobs}
            models={models}
            onComplete={completeSetupWizard}
            onCreateProject={createProject}
            onDownloadModel={createModelDownloadJob}
            onOpenQueue={() => setActiveView("Queue")}
          />
        ) : setupGateLoading ? null : needsFirstProject ? (
          <FirstRunProjectGate disabled={!authenticated} onCreate={createProject} />
        ) : (
          <>
        {/* OUT screens (Library/Assets, Queue, Models, Settings, Stats, Logs,
            Licenses): keep the conditional-unmount behavior — mounted only while
            active, unmounted on navigation (sc-11959). */}
        {activeView === "Library" ? <LibraryScreen /> : null}
        {activeView === "Queue" ? <QueueScreen /> : null}
        {activeView === "Models" ? <ModelManagerScreen /> : null}
        {activeView === "Settings" ? <SettingsScreen /> : null}
        {activeView === "Stats" ? <StatsScreen /> : null}
        {activeView === "Logs" ? <LogsScreen /> : null}
        {activeView === "Licenses" ? <LicensesScreen /> : null}

        {/* Keep-alive screens (sc-11959): mounted on first visit via keepAliveMounted,
            then kept mounted and toggled visible/hidden by KeepAlivePane so their
            state survives navigation. The key={activeProject?.id} on the studios +
            Image Editor is preserved so a PROJECT switch still remounts (resets) them
            even while kept alive. The studioLaunch apply effects inside each studio are
            keyed on the launch token id, not on mount, so a fresh "Use this Recipe" /
            "Use in Studio" injection still fires on an already-mounted studio. */}
        {keepAliveMounted("Image") ? (
          <KeepAlivePane active={activeView === "Image"}>
            <ImageStudio key={activeProject?.id ?? "default"} />
          </KeepAlivePane>
        ) : null}

        {keepAliveMounted("Video") ? (
          <KeepAlivePane active={activeView === "Video"}>
            <VideoStudio key={activeProject?.id ?? "default"} />
          </KeepAlivePane>
        ) : null}

        {keepAliveMounted("Characters") ? (
          <KeepAlivePane active={activeView === "Characters"}>
            <CharacterStudio key={activeProject?.id ?? "default"} />
          </KeepAlivePane>
        ) : null}

        {keepAliveMounted("Document") ? (
          <KeepAlivePane active={activeView === "Document"}>
            <DocumentStudio key={activeProject?.id ?? "default"} />
          </KeepAlivePane>
        ) : null}

        {keepAliveMounted("Train") ? (
          <KeepAlivePane active={activeView === "Train"}>
            <TrainingStudio />
          </KeepAlivePane>
        ) : null}

        {keepAliveMounted("LibraryDataSets") ? (
          <KeepAlivePane active={activeView === "LibraryDataSets"}>
            <TrainingDataSetsLibrary />
          </KeepAlivePane>
        ) : null}

        {keepAliveMounted("Presets") ? (
          <KeepAlivePane active={activeView === "Presets"}>
            <PresetManagerScreen />
          </KeepAlivePane>
        ) : null}

        {keepAliveMounted("Poses") ? (
          <KeepAlivePane active={activeView === "Poses"}>
            {/* Keyed on the project id (sc-11971): the Create tab's staged sources / phase
                are PROJECT-SCOPED, so a project switch must remount (reset) the screen —
                exactly like the studios — while keep-alive preserves in-progress review
                across a plain nav round trip. Otherwise project-A source picks could submit
                a pose_detect job under project B. */}
            <PoseLibraryScreen key={activeProject?.id ?? "default"} />
          </KeepAlivePane>
        ) : null}

        {keepAliveMounted("Keypoints") ? (
          <KeepAlivePane active={activeView === "Keypoints"}>
            {/* NOT keyed on the project id (sc-11971): the Key Point Library is GLOBAL
                (GLOBAL_KEYPOINTS_PROJECT_ID), so its capture / collection work must survive
                a project switch as well as a plain nav — keep-alive alone, no remount. */}
            <KeyPointLibraryScreen />
          </KeepAlivePane>
        ) : null}

        {keepAliveMounted("Editor") ? (
          <KeepAlivePane active={activeView === "Editor"}>
            <EditorScreen />
          </KeepAlivePane>
        ) : null}

        {keepAliveMounted("ImageEditor") ? (
          <KeepAlivePane active={activeView === "ImageEditor"}>
            <React.Suspense fallback={<section className="page-frame">Loading editor…</section>}>
              <ImageEditor key={activeProject?.id ?? "default"} />
            </React.Suspense>
          </KeepAlivePane>
        ) : null}
          </>
        )}
      </section>

      {previewedAsset ? (
        <FullscreenPreview
          asset={previewedAsset}
          deleteAsset={async (asset) => {
            // Stay in the preview and advance to the neighbour in the direction
            // the user was scrolling (falling back to the other side, then to
            // closing once nothing is left).
            const { previous, next } = previewNavigation;
            const target =
              previewDirectionRef.current === "previous" ? previous ?? next : next ?? previous;
            await deleteAsset(asset);
            // Advance within the launch collection; close (and drop the scope)
            // once it is exhausted.
            if (target) {
              setPreviewAsset(target);
            } else {
              closePreview();
            }
          }}
          nextAsset={previewNavigation.next}
          onClose={closePreview}
          onEditImage={sendAssetToImageEditor}
          onEditInStudio={sendAssetToImageEdit}
          onPreviewAsset={(asset, direction) => {
            if (direction) {
              previewDirectionRef.current = direction;
            }
            setPreviewAsset(asset);
          }}
          onUseRecipe={sendAssetRecipeToImage}
          previousAsset={previewNavigation.previous}
          purgeAsset={async (asset) => {
            await purgeAsset(asset);
            closePreview();
          }}
          updateAssetStatus={updateAssetStatus}
        />
      ) : null}

      {/* Desktop-safe confirm dialog host (sc-11968). Mounted once at the app root so
          appConfirm()/useConfirm() anywhere — including a leave-guard callback handed to
          navTo — resolve through a real React dialog instead of window.confirm (which
          silently no-ops in the Tauri WebView). Renders nothing until a confirm is asked. */}
      <ConfirmHost />
    </main>
    </AppLiveContext.Provider>
    </AppStaticContext.Provider>
  );
}
