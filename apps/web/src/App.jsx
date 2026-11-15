import { useAppDomainValue } from "./context/useAppDomainValue.js";
import { KEEP_ALIVE_VIEWS, navSections, viewTitles, KeepAlivePane, ProjectSwitcher, FirstRunProjectGate } from "./components/shell/ShellChrome.jsx";
import { useThemeAccent } from "./hooks/useThemeAccent.js";
import { useWebView2RepaintPump } from "./hooks/useWebView2RepaintPump.js";
import { useShellHydration } from "./hooks/useShellHydration.js";
import { useJobCreators } from "./hooks/useJobCreators.js";
import { useJobActions } from "./hooks/useJobActions.js";
import { useSendToStudio } from "./hooks/useSendToStudio.js";
import { useAssetActions } from "./hooks/useAssetActions.js";
import { usePromptRunners } from "./hooks/usePromptRunners.js";

import { systemApi } from "./api/systemResources.js";

import { projectsApi } from "./api/projectsResources.js";

import { AppVersion, useAppUpdate } from "./components/AppVersion.jsx";
import React, { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import { isAbortError } from "./api.js";
import { recordStartupMark } from "./startupTiming.js";

import { AccentPicker } from "./components/AccentPicker.jsx";
import { Icon } from "./components/Icons.jsx";
import { lazyInteraction, lazyScreen } from "./components/LazyScreen.jsx";
import { AccessGate } from "./components/AccessGate.jsx";
import { Logo } from "./components/Logo.jsx";
import { StatusDot } from "./components/StatusDot.jsx";
import { FullscreenPreview } from "./components/assetPanels.jsx";
import { isLibraryAsset, terminalStatuses } from "./constants.js";
import { isEditorJob } from "./jobTypes.js";
import { LibraryScreen } from "./screens/LibraryScreen.jsx";

import { sortNewest } from "./sorters.js";
import { useCharacters } from "./hooks/useCharacters.js";
import { usePresets } from "./hooks/usePresets.js";
import { useSavedVoices } from "./hooks/useSavedVoices.js";
import { usePromptBatches } from "./hooks/usePromptBatches.js";
import { useTraining } from "./hooks/useTraining.js";
import { useModelsAndLoras } from "./hooks/useModelsAndLoras.js";
import { usePersonTracks } from "./hooks/usePersonTracks.js";
import { useTimelines } from "./hooks/useTimelines.js";
import { useAccessGate } from "./hooks/useAccessGate.js";
import { useDropNavigationGuard } from "./hooks/useDropNavigationGuard.js";
import { useWorkflowDrop } from "./hooks/useWorkflowDrop.js";
import {
  persistWorkflowEmbedPreference,
  readEmbedWorkflowInImages,
  readWorkflowEmbedNoticeSeen,
  subscribeWorkflowEmbedFlags,
  writeEmbedWorkflowInImages,
  writeWorkflowEmbedNoticeSeen,
} from "./workflowEmbedPreferences.js";
import { useJobEvents } from "./hooks/useJobEvents.js";
import { AppStaticContext, AppLiveContext } from "./context/AppContext.js";

import { DEFAULT_MAC_CAPABILITIES } from "./modelEligibility.js";
import { generationModelsForType } from "./modelEligibility.js";
import { isAccentId } from "./accents.js";
import { writeDefaultGenerationQuality } from "./generationQuality.js";
import { persistNavigationPreferences, putUiPreferences } from "./uiPreferences.js";
import { seedLastTiersFromServer } from "./lastTierStore.js";
import { seedSimpleStudiosFromServer } from "./simpleStudioStore.js";
import { seedStudioSettingsFromServer } from "./hooks/useStudioSettings.js";
import { UpgradeError } from "./requestUpgrade.js";
import {
  dropUpscaledVariants,
  findFoldedAssetById,
  foldUpscaledAssetVariants,
  restrictFoldedToScope,
} from "./assetVariants.js";
import { buildWorkersById } from "./workers.js";
import { createEditorScratchRegistry } from "./editorScratch.js";
import { ConfirmHost } from "./appConfirm.jsx";
import { ModelLibraryRestartDialog } from "./components/ModelLibraryRestartDialog.jsx";
import {
  adoptModelLibrary,
  createModelLibraryGate,
  fetchModelLibraryStatus,
  relocateModelLibrary,
  setModelLibraryHandler,
  validateModelLibrary,
} from "./modelLibrary.js";
import { isDesktop as isDesktopShell, tauriInvoke } from "./runtime.js";
import { SimpleModeSwitch } from "./simple/SimpleModeSwitch.jsx";
import { useViewportWidth } from "./simple/useContainerWidth.js";
import {
  ADVANCED_MODE,
  SIMPLE_MODE,
  readStoredSimpleDefault,
  resolveUiMode,
  uiModeLockedToSimple,
  writeStoredSimpleDefault,
} from "./simple/uiMode.js";
import { buildLocalJobStack, canWorkerTakeWork, isActiveWorker, isAudioGenerationJob, isImageGenerationJob, isInterleaveJob, isPlaceholderOnlyGpuWorker, isSelectableGpuWorker, isVideoGenerationJob, localJobStackLimit } from "./appHelpers.js";
import { hasVisibleLocalFailureForView, selectEditorMediaAssets } from "./appStateHelpers.js";

// Simple UI (design handoff "Simple UI for creative studios") — an ALTERNATIVE shell that
// renders instead of this workspace when the sidebar switch is on Simple. The workspace
// below is unchanged apart from that switch in its footer.

// Desktop (Tauri) shell detection (unified helper, epic 4484 story 6). The first-run
// setup wizard is desktop-only; web/Docker (and a remote LAN browser) keep the
// existing first-run project gate. Tauri commands persist the wizard state (the API
// binds a random port each launch, so localStorage — keyed to the origin — can't be
// relied on across launches).

// Lazy-load the canvas editor so Konva (canvas-based, heavy) stays out of the
// initial bundle and the jsdom test path — it only loads when the view is opened.
const ImageStudio = lazyScreen(
  () => import("./screens/ImageStudio.jsx"),
  "ImageStudio",
  "Image Studio",
);
const VideoStudio = lazyScreen(
  () => import("./screens/VideoStudio.jsx"),
  "VideoStudio",
  "Video Studio",
);
const AudioStudio = lazyScreen(
  () => import("./screens/AudioStudio.jsx"),
  "AudioStudio",
  "Audio Studio",
);
const CharacterStudio = lazyScreen(
  () => import("./screens/CharacterStudio.jsx"),
  "CharacterStudio",
  "Characters",
);
const DocumentStudio = lazyScreen(
  () => import("./screens/DocumentStudio.jsx"),
  "DocumentStudio",
  "Document Studio",
);
const loadTrainingScreens = () => import("./screens/TrainingStudio.jsx");
const TrainingStudio = lazyScreen(loadTrainingScreens, "TrainingStudio", "Training Studio");
const TrainingDataSetsLibrary = lazyScreen(
  loadTrainingScreens,
  "TrainingDataSetsLibrary",
  "Data Sets",
);
const PresetManagerScreen = lazyScreen(
  () => import("./screens/PresetManagerScreen.jsx"),
  "PresetManagerScreen",
  "Presets",
);
const PoseLibraryScreen = lazyScreen(
  () => import("./screens/PoseLibraryScreen.jsx"),
  "PoseLibraryScreen",
  "Pose Library",
);
const KeyPointLibraryScreen = lazyScreen(
  () => import("./screens/KeyPointLibraryScreen.jsx"),
  "KeyPointLibraryScreen",
  "Key Point Library",
);
const EditorScreen = lazyScreen(
  () => import("./screens/EditorScreen.jsx"),
  "EditorScreen",
  "Video Editor",
);
const ImageEditor = lazyScreen(
  () => import("./screens/ImageEditor.jsx"),
  "ImageEditor",
  "Image Editor",
);
const VectorStudio = lazyScreen(
  () => import("./screens/VectorStudio.jsx"),
  "VectorStudio",
  "Vector Studio",
);
const QueueScreen = lazyScreen(
  () => import("./screens/QueueScreen.jsx"),
  "QueueScreen",
  "Queue",
);
const ModelManagerScreen = lazyScreen(
  () => import("./screens/ModelManagerScreen.jsx"),
  "ModelManagerScreen",
  "Models",
);
const DatasetCatalogsScreen = lazyScreen(
  () => import("./screens/DatasetCatalogsScreen.jsx"),
  "DatasetCatalogsScreen",
  "Dataset Catalogs",
);
const SettingsScreen = lazyScreen(
  () => import("./screens/SettingsScreen.jsx"),
  "SettingsScreen",
  "Settings",
);
const StatsScreen = lazyScreen(
  () => import("./screens/StatsScreen.jsx"),
  "StatsScreen",
  "Generation Stats",
);
const LogsScreen = lazyScreen(
  () => import("./screens/LogsScreen.jsx"),
  "LogsScreen",
  "Logs",
);
const LicensesScreen = lazyScreen(
  () => import("./screens/LicensesScreen.jsx"),
  "LicensesScreen",
  "Licenses",
);
const SetupWizard = lazyScreen(
  () => import("./screens/SetupWizard.jsx"),
  "SetupWizard",
  "Setup Wizard",
);
const SimpleShell = lazyScreen(
  () => import("./simple/SimpleShell.jsx"),
  "SimpleShell",
  "Simple UI",
);
const ModelLibraryDialog = lazyInteraction(
  () => import("./components/ModelLibraryDialog.jsx"),
  "ModelLibraryDialog",
  "Model library recovery",
);
const WorkflowEmbedDecisionModal = lazyInteraction(
  () => import("./components/WorkflowEmbedNotice.jsx"),
  "WorkflowEmbedDecisionModal",
  "Workflow sharing choice",
);
const WorkflowDropPanel = lazyInteraction(
  () => import("./components/WorkflowDropPanel.jsx"),
  "WorkflowDropPanel",
  "Workflow details",
);

export { KEEP_ALIVE_VIEWS, navSections, viewTitles } from "./components/shell/ShellChrome.jsx";

export function App() {
  const [health, setHealth] = useState(null);
  const [projects, setProjects] = useState([]);
  const [projectsLoaded, setProjectsLoaded] = useState(false);
  // Desktop first-run wizard gate: null = unknown (still reading on desktop),
  // true = no wizard needed (web, or already completed), false = show the wizard.
  const [setupCompleted, setSetupCompleted] = useState(isDesktopShell ? null : true);
  const [activeProject, setActiveProject] = useState(null);
  const [activeView, setActiveView] = useState("Library");
  const [navigationHydrated, setNavigationHydrated] = useState(false);
  // Bumped when the durable studio snapshot is seeded into the localStorage cache (sc-15425).
  // Part of the kept-alive studios' React key, so a studio that mounted before the seed landed
  // remounts and re-reads it. See the seed call in the ui-preferences effect.
  const [studioRestoreEpoch, setStudioRestoreEpoch] = useState(0);
  const [simpleActiveScreen, setSimpleActiveScreen] = useState("image");
  // Selective lazy keep-alive (sc-11959): the set of keep-alive views the user has
  // visited at least once. A view mounts on first visit (activeView === view) and,
  // once here, stays mounted (hidden) across navigation so its state survives.
  const [visitedKeepAliveViews, setVisitedKeepAliveViews] = useState(() => new Set());
  const [jobs, setJobs] = useState([]);
  const [localGenerationJobIds, setLocalGenerationJobIds] = useState({ image: [], video: [], audio: [], document: [] });
  const [workers, setWorkers] = useState([]);
  const [queueSummary, setQueueSummary] = useState(null);
  // Mac UI gating (sc-3486): the value is inert until the capabilities endpoint reports
  // macGatingActive, but keep a separate readiness bit so consumers do not mistake that
  // temporary fallback for authoritative Candle capability facts.
  const [macCapabilities, setMacCapabilities] = useState(DEFAULT_MAC_CAPABILITIES);
  const [macCapabilitiesAuthoritative, setMacCapabilitiesAuthoritative] = useState(false);
  const [macCapabilitiesError, setMacCapabilitiesError] = useState("");
  const [macCapabilitiesLoading, setMacCapabilitiesLoading] = useState(false);
  // The generation pickers must not treat the hand-authored fallback catalog as proof that a
  // model is present on this machine. Until GET /models completes its install/availability sweep,
  // keep the studios in an explicit initializing state instead (the request may legitimately take
  // tens of seconds on a cold start). A settled refresh keeps the last authoritative catalog live.
  const [trainingTargets, setTrainingTargets] = useState({ schemaVersion: 1, targets: [] });
  const [trainingPresets, setTrainingPresets] = useState({ schemaVersion: 1, presets: [] });
  const [trainingTargetsError, setTrainingTargetsError] = useState("");
  const [trainingPresetsError, setTrainingPresetsError] = useState("");
  const [loadedAssets, setAssets] = useState([]);
  const [loadedAssetsProjectId, setLoadedAssetsProjectId] = useState(null);
  const [assetLoadState, setAssetLoadState] = useState({
    projectId: null,
    status: "idle",
    error: "",
  });
  const [storedSelectedAssetId, setSelectedAssetId] = useState(null);
  // Scope project data during render, not in a passive effect. On A → B (or A
  // → no project), B's first render therefore cannot observe A's assets or raw
  // selection while B's catalog request is still pending.
  const assets = useMemo(
    () => (loadedAssetsProjectId === activeProject?.id ? loadedAssets : []),
    [activeProject?.id, loadedAssets, loadedAssetsProjectId],
  );
  const hasLibraryConsumerAsset = useMemo(
    () => assets.some(isLibraryAsset),
    [assets],
  );
  const selectedAssetId =
    loadedAssetsProjectId === activeProject?.id ? storedSelectedAssetId : null;
  // A project switch renders before its passive refresh effect runs. Treat that
  // first render as loading instead of briefly presenting the previous
  // project's settled state (or a misleading empty catalog).
  const activeAssetLoadState = useMemo(
    () =>
      activeProject && assetLoadState.projectId === activeProject.id
        ? assetLoadState
        : {
            projectId: activeProject?.id ?? null,
            status: activeProject ? "loading" : "idle",
            error: "",
          },
    [activeProject, assetLoadState],
  );
  const [projectFilter, setProjectFilter] = useState("all");
  const [requestedGpu, setRequestedGpu] = useState("auto");
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
  const closePreview = useCallback(() => {
    setPreviewScopeIds(null);
    setPreviewAsset(null);
  }, []);
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
  // Domain loaders and mutations own separate notice slots. A successful Models
  // request must never erase a Training or Characters failure (sc-14783).
  const setDomainError = useCallback(
    (domain, message) => pushNotice(`domain:${domain}`, message),
    [pushNotice],
  );
  const domainErrors = useMemo(
    () => ({
      assets: (message) => setDomainError("assets", message),
      characters: (message) => setDomainError("characters", message),
      loras: (message) => setDomainError("loras", message),
      models: (message) => setDomainError("models", message),
      personTracks: (message) => setDomainError("person-tracks", message),
      presets: (message) => setDomainError("presets", message),
      promptBatches: (message) => setDomainError("prompt-batches", message),
      savedVoices: (message) => setDomainError("saved-voices", message),
      timelines: (message) => setDomainError("timelines", message),
      trainingDatasets: (message) => setDomainError("training-datasets", message),
      trainingPresets: (message) => setDomainError("training-presets", message),
      trainingTargets: (message) => setDomainError("training-targets", message),
    }),
    [setDomainError],
  );
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
    gateStatus,
    ready,
    saveToken,
    lockRemote,
  } = useAccessGate({ setError, pushNotice, dismissNoticeKind });
  // The unavailable-model-library prompt (sc-19709). One gate for the whole app: it holds at most
  // one blocked action and resumes it at most once, so a reconnect (or an impatient second click)
  // can never turn one blocked generation into two jobs. Registered as the module-level handler so
  // every submission path — and the studios' selection check — reaches it without prop threading.
  // The relocation sequence's three seams, built once and shared by the blocked-action prompt
  // (below) and the Settings → Storage "Model library" control, which relocates with no blocked
  // action to resume but with exactly the same two-write ordering and undo.
  const modelLibraryRelocator = useMemo(
    () => ({
      validate: (path) => validateModelLibrary(token, path),
      adopt: (path) => relocateModelLibrary(token, path),
      // Durable persistence of a relocated library is desktop state: the API hands back the exact
      // HF_HOME the shell must store, in the same field, file and normalization as the first-run
      // storage step. Returns an undo so the gate can restore the previous location if the
      // server's re-bind then fails — the two copies must never disagree.
      //
      // Fails CLOSED when the previous location cannot be read, BEFORE the first durable write:
      // without a previous location there is no undo, so a re-bind failure afterwards would leave
      // the shell pointed at the new library while the gate reported the previous location still
      // in use — the disagreement this undo path exists to prevent, with the wrong message on top.
      persist: async (target) => {
        // Nothing to persist — the browser shell keeps no copy of the location. A NO-OP undo, not
        // `null`: the gate treats a missing undo as "changed and unrestorable", and here nothing
        // changed, so the previous location genuinely is still in use.
        if (!isDesktopShell || !target?.hfHome) return () => {};
        let before = null;
        try {
          before = await tauriInvoke("get_storage_setup");
        } catch (error) {
          throw new Error(
            `The current model library location could not be read, so nothing was changed. ${String(error)}`.trim(),
          );
        }
        const previous = before?.hfHome ?? before?.hfHomeDefault ?? null;
        if (!previous) {
          throw new Error(
            "The current model library location could not be read, so nothing was changed.",
          );
        }
        await tauriInvoke("set_model_library", { path: target.hfHome });
        return () => tauriInvoke("set_model_library", { path: previous });
      },
    }),
    [token],
  );
  const modelLibraryGate = useMemo(
    () =>
      createModelLibraryGate({
        probe: () => fetchModelLibraryStatus(token),
        ...modelLibraryRelocator,
      }),
    [token, modelLibraryRelocator],
  );
  const modelLibraryState = useSyncExternalStore(
    modelLibraryGate.subscribe,
    modelLibraryGate.getState,
  );
  useEffect(() => setModelLibraryHandler(modelLibraryGate.block), [modelLibraryGate]);
  // The relocated library, held until the user answers the restart disclosure. The relocation is
  // already durable at this point — this only decides WHEN the app picks it up.
  const [modelLibraryRelocation, setModelLibraryRelocation] = useState(null);
  // True while the NATIVE folder picker is up. The prompt's auto re-probe pauses on it: a drive
  // that comes back while the user is choosing a folder must not resume the submission behind a
  // modal OS dialog the user cannot see past.
  const [modelLibraryPickerOpen, setModelLibraryPickerOpen] = useState(false);
  const chooseModelLibraryLocation = useCallback(async () => {
    if (!isDesktopShell) return;
    setModelLibraryPickerOpen(true);
    let picked = null;
    try {
      picked = await tauriInvoke("choose_folder").catch(() => null);
    } finally {
      setModelLibraryPickerOpen(false);
    }
    if (!picked) return;
    const adopted = await modelLibraryGate.relocate(picked);
    if (adopted?.hfHome) {
      // `droppedSubmission`: the prompt path dropped whatever the user was submitting; the restart
      // disclosure says so. The Settings path below has no submission to drop.
      setModelLibraryRelocation({ ...adopted, droppedSubmission: true });
      // The notice outlives the dialog, so "Later" still leaves the reason visible.
      pushNotice(
        "general",
        `Model library set to ${adopted.hfHome} — restart SceneWorks to apply it. The generation you started was not queued; submit it again after the restart.`,
      );
    }
  }, [modelLibraryGate, pushNotice]);
  // Settings → Storage → Model library → Change…: the same relocation (validate, persist, re-bind,
  // undo on failure) and the same restart disclosure as the prompt, with no blocked action. Resolves
  // to the adopted target, `null` when the picker was dismissed; rejects with the user-facing
  // message when the folder was refused or a write failed, for the Settings screen to show inline.
  // Unlike the prompt path there is no `modelLibraryPickerOpen` bracket: that flag only pauses the
  // gate's auto re-probe, and with no blocked action pending there is nothing a re-probe could
  // resume behind the native dialog.
  const changeModelLibrary = useCallback(async () => {
    if (!isDesktopShell) return null;
    const picked = await tauriInvoke("choose_folder").catch(() => null);
    if (!picked) return null;
    const adopted = await adoptModelLibrary(modelLibraryRelocator, picked);
    setModelLibraryRelocation({ ...adopted, droppedSubmission: false });
    return adopted;
  }, [modelLibraryRelocator]);
  // The drop guard that stops a stray file from navigating the webview (issue #1308) is
  // installed further down, next to `useWorkflowDrop` — it now hands the unclaimed file to
  // the workflow inspector (sc-15951), which needs the active project and `importAsset`.
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const sidebarChanged = useRef(false);
  const changeSidebarCollapsed = (next) => {
    sidebarChanged.current = true;
    setSidebarCollapsed(next);
    persistNavigationPreferences({ sidebarCollapsed: next });
  };
  const { theme, setTheme, changeTheme, accent, setAccent, changeAccent, context: themeAccentContext } = useThemeAccent();

  // Simple UI (design handoff). `simpleUiDefault` is the persisted "Use Simple UI by
  // default" preference — same durable contract as theme/accent (server copy in
  // ui-preferences.json + a localStorage instant-paint cache), because the desktop
  // shell's per-launch 127.0.0.1:<port> origin wipes origin-keyed localStorage.
  // `uiModeOverride` is this session's sidebar pick, which wins over the preference;
  // a phone viewport wins over both (the full workspace is unusable at that width).
  const [simpleUiDefault, setSimpleUiDefault] = useState(readStoredSimpleDefault);
  const [uiModeOverride, setUiModeOverride] = useState(null);
  const viewportWidth = useViewportWidth();
  const uiModeLocked = uiModeLockedToSimple(viewportWidth);
  const uiMode = resolveUiMode({
    simpleDefault: simpleUiDefault,
    override: uiModeOverride,
    width: viewportWidth,
  });
  const changeSimpleUiDefault = useCallback((next) => {
    setSimpleUiDefault(next);
    writeStoredSimpleDefault(next);
    putUiPreferences({ simpleUi: next }).catch(() => {});
  }, []);
  // Embedded workflow (sc-15953, epic 15945). `embedWorkflow` is the durable
  // `embedWorkflowInImages` preference; the WORKER reads it off ui-preferences.json at the PNG
  // write seam on every job, so a flip here lands on the next generation with no restart. Same
  // durable contract as theme/accent, and for a sharper reason: the desktop shell's per-launch
  // origin wipes localStorage, so a preference stored only there would silently revert to the ON
  // default on the next launch — turning a deliberate privacy opt-out back on.
  const [embedWorkflow, setEmbedWorkflow] = useState(readEmbedWorkflowInImages);
  // `workflowEmbedNoticeSeen` is the legacy persisted key, now used as the durable "choice made"
  // marker. Keeping the key preserves every prior decision across upgrades.
  const [workflowEmbedNoticeSeen, setWorkflowEmbedNoticeSeen] = useState(
    readWorkflowEmbedNoticeSeen,
  );
  const [workflowEmbedNoticeOpen, setWorkflowEmbedNoticeOpen] = useState(false);
  const [workflowEmbedDecisionSaving, setWorkflowEmbedDecisionSaving] = useState(false);
  const [workflowEmbedDecisionError, setWorkflowEmbedDecisionError] = useState("");
  const [settingsSharingFocusRequest, setSettingsSharingFocusRequest] = useState(0);
  const pendingWorkflowGenerationRef = useRef(null);
  const workflowEmbedHydrationWaitersRef = useRef([]);
  const workflowEmbedDecisionSavingRef = useRef(false);
  // Read by the generation gate below, which is memoized and would otherwise close over stale
  // values. The gate waits for durable hydration before it decides whether the modal is needed.
  const workflowEmbedRef = useRef({ hydrated: false, embed: true, seen: false });
  workflowEmbedRef.current.embed = embedWorkflow;
  workflowEmbedRef.current.seen = workflowEmbedNoticeSeen;
  // Both writes go through `persistWorkflowEmbedPreference`, which retries and then REPORTS. A
  // swallowed `.catch(() => {})` is wrong for this pair specifically: localStorage is only an
  // instant-paint mirror, and on the desktop shell it is wiped by the per-launch origin — so a
  // dropped PUT is not "saved locally", it is a privacy opt-out that reverts to ON at the next
  // launch, or a disclosure that fires again as though it had never been dismissed. Both are
  // things the user should be told about rather than discover.
  const changeEmbedWorkflow = useCallback(
    (next) => {
      setEmbedWorkflow(next);
      writeEmbedWorkflowInImages(next);
      persistWorkflowEmbedPreference({ embedWorkflowInImages: next }).catch(() => {
        pushNotice(
          "workflow-embed",
          "Could not save whether generated images include their recipe. The setting is active now but may revert when SceneWorks restarts.",
        );
      });
    },
    [pushNotice],
  );
  const settlePendingWorkflowGeneration = useCallback((allow) => {
    const pending = pendingWorkflowGenerationRef.current;
    if (!pending) return;
    pendingWorkflowGenerationRef.current = null;
    pending(Boolean(allow));
  }, []);

  const chooseWorkflowEmbedding = useCallback(
    async (next) => {
      if (workflowEmbedDecisionSavingRef.current) return;
      workflowEmbedDecisionSavingRef.current = true;
      setWorkflowEmbedDecisionSaving(true);
      setWorkflowEmbedDecisionError("");
      try {
        await persistWorkflowEmbedPreference({
          embedWorkflowInImages: Boolean(next),
          workflowEmbedNoticeSeen: true,
        });
        workflowEmbedRef.current.embed = Boolean(next);
        workflowEmbedRef.current.seen = true;
        setEmbedWorkflow(Boolean(next));
        setWorkflowEmbedNoticeSeen(true);
        writeEmbedWorkflowInImages(Boolean(next));
        writeWorkflowEmbedNoticeSeen(true);
        setWorkflowEmbedNoticeOpen(false);
        settlePendingWorkflowGeneration(true);
      } catch {
        setWorkflowEmbedDecisionError(
          "SceneWorks could not save your choice. Check the connection and try again.",
        );
      } finally {
        workflowEmbedDecisionSavingRef.current = false;
        setWorkflowEmbedDecisionSaving(false);
      }
    },
    [settlePendingWorkflowGeneration],
  );

  const openWorkflowSharingSettings = useCallback(() => {
    setWorkflowEmbedNoticeOpen(false);
    setWorkflowEmbedDecisionError("");
    settlePendingWorkflowGeneration(false);
    setSettingsSharingFocusRequest((request) => request + 1);
    setSimpleActiveScreen("settings");
    setActiveView("Settings");
  }, [settlePendingWorkflowGeneration]);
  // A completed choice is mirrored to other mounted tabs so they cannot ask independently.
  useEffect(
    () =>
      subscribeWorkflowEmbedFlags(({ embed, seen }) => {
        setEmbedWorkflow(embed);
        // One-way: see `subscribeWorkflowEmbedFlags`. `seen` is a record that something was said,
        // and another tab clearing its storage is not evidence it never was.
        if (seen) {
          workflowEmbedRef.current.seen = true;
          setWorkflowEmbedNoticeSeen(true);
          setWorkflowEmbedNoticeOpen(false);
          settlePendingWorkflowGeneration(true);
        }
      }),
    [settlePendingWorkflowGeneration],
  );
  // The first undecided generation is held before its POST. A choice resumes that one request;
  // Settings cancels it so no image can be written with an unresolved preference.
  const confirmWorkflowEmbeddingBeforeGeneration = useCallback(async () => {
    if (!workflowEmbedRef.current.hydrated) {
      await new Promise((resolve) => {
        workflowEmbedHydrationWaitersRef.current.push(resolve);
      });
    }
    const state = workflowEmbedRef.current;
    if (state.seen) {
      return true;
    }
    if (pendingWorkflowGenerationRef.current) {
      return false;
    }
    setWorkflowEmbedDecisionError("");
    setWorkflowEmbedNoticeOpen(true);
    return new Promise((resolve) => {
      pendingWorkflowGenerationRef.current = resolve;
    });
  }, []);
  const activeProjectRef = useRef(null);

  const activeViewRef = useRef(activeView);
  const localGenerationJobIdsRef = useRef(localGenerationJobIds);
  const generatedAssetRefreshesRef = useRef(new Map());

  const healthRequestRef = useRef(null);

  // Latest purgeAsset, held in a ref so the App-level scratch-op survivor (sc-8850) can
  // purge orphaned scratch/result assets from the SSE handler without re-subscribing.
  const purgeAssetRef = useRef(null);

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
    (jobId, assets, options) => editorScratchRegistry.track(jobId, assets, options),
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

  const { createVideoJob, context: jobCreatorsContext } = useJobCreators({
    token, activeProject, requestedGpu, setJobs, setError, setActiveView, activeViewRef, confirmWorkflowEmbeddingBeforeGeneration
  });

  const { setCharacters, refreshCharacters, context: charactersContext } = useCharacters({
    token,
    activeProject,
    activeProjectRef,
    setError: domainErrors.characters,
    requestedGpu,
    setActiveView,
  });

  const { setPresets, refreshPresets, context: presetsContext } = usePresets({ token, activeProject, activeProjectRef, setError: domainErrors.presets });

  const { setSavedVoices, refreshSavedVoices, context: savedVoicesContext } = useSavedVoices({
    token,
    activeProject,
    activeProjectRef,
    setError: domainErrors.savedVoices,
  });

  const { setPromptBatches, refreshPromptBatches, context: promptBatchesContext } = usePromptBatches({
    token,
    activeProject,
    activeProjectRef,
    setError: domainErrors.promptBatches,
  });

  const { setTrainingDatasets, setTrainingDatasetsProjectId, setTrainingDatasetsError, refreshTrainingDatasets, context: trainingContext } = useTraining({
    token,
    activeProject,
    activeProjectRef,
    setError: domainErrors.trainingDatasets,
    setJobs,
  });

  const { refreshModels, refreshModelAndLoras, models, loras, setLoras, refreshLoras, createModelDownloadJob, createLoraDownloadJob, context: modelsAndLorasContext } = useModelsAndLoras({
    token,
    activeProject,
    activeProjectRef,
    setError: domainErrors.models,
    setLoraError: domainErrors.loras,
    setJobs,
    setActiveView,
  });

  const { setPersonTracks, refreshPersonTracks, context: personTracksContext } = usePersonTracks({
    token,
    activeProject,
    activeProjectRef,
    setError: domainErrors.personTracks,
    requestedGpu,
    setActiveView,
  });

  const { setTimelines, setTimelinesProjectId, setSelectedTimelineId, setActiveTimeline, refreshTimelines, enqueueTimelineGenerationApply, context: timelinesContext } = useTimelines({
    token,
    activeProject,
    activeProjectRef,
    setError: domainErrors.timelines,
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
    return generationModelsForType(models, "image");
  }, [models]);
  const videoModels = useMemo(() => generationModelsForType(models, "video"), [models]);
  // Audio models (epic 13400) are resolved from the authoritative live catalog and consumed by
  // Audio Studio (C0/C1). Per-mode eligibility comes from audioModelServesMode.
  const audioModels = useMemo(() => {
    return generationModelsForType(models, "audio");
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
  // The source image an edit was derived from (asset.lineage.sourceAssetId), resolved
  // from the live asset set so the Image Viewer can offer an Original↔Edited side-by-side
  // compare (the edit stays its own separate tile — this is a view link, not a fold).
  // Null when the previewed asset isn't a derived edit or its source is gone (purged).
  const previewSourceAsset = useMemo(() => {
    const sourceId = previewedAsset?.lineage?.sourceAssetId;
    if (!sourceId) {
      return null;
    }
    return assets.find((candidate) => candidate.id === sourceId) ?? null;
  }, [assets, previewedAsset]);
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
  // The audio twin of recentImage/recentVideoAssets (epic 14361): the Audio Studio's results
  // zone groups this session's runs AND the project's recent clips, so "Latest audio" is
  // populated on a fresh launch instead of empty until you generate something.
  const recentAudioAssets = useMemo(
    () =>
      assets
        .filter((asset) => asset.type === "audio" && (!activeProject?.id || asset.projectId === activeProject.id))
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
  const audioLocalJobs = useMemo(
    () => buildLocalJobStack(localGenerationJobIds.audio, jobs, activeProject?.id, isAudioGenerationJob),
    [activeProject?.id, jobs, localGenerationJobIds.audio],
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
  // Jobs actually executing on a worker. The restart disclosure (sc-19709) withholds "Restart now"
  // while any of these are in flight: the teardown is graceful, but interrupting a render still
  // throws away the work, so the user defers rather than discovers. Prefers the server's own count
  // when the queue summary is loaded, since `jobs` is a recent window rather than the whole queue.
  const runningJobCount = useMemo(
    () =>
      queueSummary?.counts?.running ??
      jobs.filter((job) => job.status === "running").length,
    [jobs, queueSummary],
  );
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
    // sc-16260: `canWorkerTakeWork`, not a bare non-offline check — an unhealthy worker is alive
    // but cannot run anything, and counting it here would ungate Replace Person on a host whose
    // GPU failed its startup probe. Keeps this in step with the server's
    // `person_readiness_from_workers`, which the comment above promises it mirrors.
    const live = workers.filter(canWorkerTakeWork);
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
  const mediaAssets = useMemo(() => selectEditorMediaAssets(assets), [assets]);

  useEffect(() => {
    activeViewRef.current = activeView;
  }, [activeView]);

  useEffect(() => {
    if (!navigationHydrated) return;
    persistNavigationPreferences({ activeView });
  }, [activeView, navigationHydrated]);

  // Record a keep-alive view the first time it becomes active so it stays mounted
  // thereafter (sc-11959). Never-visited keep-alive views are absent from the DOM.
  useEffect(() => {
    if (!KEEP_ALIVE_VIEWS.has(activeView)) {
      return;
    }
    setVisitedKeepAliveViews((prev) => (prev.has(activeView) ? prev : new Set(prev).add(activeView)));
  }, [activeView]);

  // Passive effects run after React commits the state batch from refreshData.
  // Keep this before the active-project effect so the marks reflect the commit
  // order that unlocks the project-scoped catalog request below.
  useEffect(() => {
    if (projectsLoaded) {
      recordStartupMark("projects-committed");
    }
  }, [projectsLoaded, projects]);

  useEffect(() => {
    activeProjectRef.current = activeProject;
    if (activeProject) {
      recordStartupMark("active-project-selected");
    }
  }, [activeProject]);

  useEffect(() => {
    localGenerationJobIdsRef.current = localGenerationJobIds;
  }, [localGenerationJobIds]);

  // Seed the theme from the server on launch (the durable copy; localStorage is
  // only an instant-paint cache). Each toggle persists itself via changeTheme,
  // so there's no save effect to race with this read.
  //
  // This READ is the one credential-less ui-preferences call, deliberately: it runs before
  // the access gate has a token, and the server keeps the GET public precisely so the theme
  // can paint pre-auth without a flash. Only the disk-writing PUT is gated (sc-8869,
  // F-067), which is why the writes go through `putUiPreferences` instead (sc-15136).
  useEffect(() => {
    let cancelled = false;
    systemApi.uiPreferences("")
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
        // A late launch read must not undo a sidebar toggle made in this session.
        if (typeof prefs?.sidebarCollapsed === "boolean" && !sidebarChanged.current) {
          setSidebarCollapsed(prefs.sidebarCollapsed);
        }
        // "Use Simple UI by default" is seeded from the durable copy; a session
        // override from the mode switch still wins over it.
        if (typeof prefs?.simpleUi === "boolean") {
          setSimpleUiDefault(prefs.simpleUi);
          writeStoredSimpleDefault(prefs.simpleUi);
        }
        // Re-prime the default-generation-quality cache from the durable server copy (sc-10728).
        // It isn't React state here — the studio reads it fresh from localStorage via
        // readDefaultGenerationQuality() — so seeding the cache is all that's needed for it to
        // survive a desktop relaunch (the GET always resolves a concrete tier). writeDefault…
        // normalizes, so an absent/legacy value lands on q8. No PUT: this is a read-only seed.
        if (prefs?.defaultGenerationQuality) {
          writeDefaultGenerationQuality(prefs.defaultGenerationQuality);
        }
        // Re-prime the per-(screen,model) tier sticky cache from the durable server copy (epic 10721
        // R1), so a tier a user picked for a model in a previous session is returned by readLastTier
        // even after the desktop origin — and its localStorage — changed on relaunch.
        if (prefs?.perModelTier) {
          seedLastTiersFromServer(prefs.perModelTier);
        }
        // Re-prime the Simple studios' settings cache from the durable server copy, for exactly
        // the same reason as the tier sticky above: the desktop shell's per-launch origin wipes
        // localStorage, so the model/prompt/resolution a user left in a Simple studio would be
        // gone every relaunch. SimpleShell waits for `navigationHydrated` before reading this,
        // so it can't race the seed and restore an empty snapshot.
        if (prefs?.simpleStudio) {
          seedSimpleStudiosFromServer(prefs.simpleStudio);
        }
        // The advanced studios' equivalent (sc-15425). Seeded BEFORE `setActiveView` below, on
        // purpose: restoring the view can mount a keep-alive studio in the same commit, and that
        // studio reads `loadStudioSettings` synchronously at mount — so the cache has to already
        // hold the durable copy or it would come up on defaults and never re-read.
        if (prefs?.advancedStudio && seedStudioSettingsFromServer(prefs.advancedStudio) > 0) {
          // Remount the kept-alive studios so a studio the user opened BEFORE this landed
          // re-reads the now-seeded cache — being keep-alive, it never would on its own. Only
          // when something was actually restored, so the common case (nothing stored yet)
          // doesn't throw away what they typed while the GET was in flight.
          setStudioRestoreEpoch((epoch) => epoch + 1);
        }
        // The embedded-workflow preference and its disclosure flag (sc-15953). Both ABSENT
        // states are meaningful and both are the permissive one: absent embedding means ON (the
        // worker's own reader agrees), and an absent notice flag means the user has never been
        // told — which is exactly the state an install upgrading into this build is in, so the
        // disclosure fires once for them rather than only for fresh installs.
        // A choice may finish while this launch GET is still in flight. Once `seen` has moved
        // true locally, an older response must not roll either half of that atomic choice back.
        if (
          !workflowEmbedRef.current.seen &&
          typeof prefs?.embedWorkflowInImages === "boolean"
        ) {
          setEmbedWorkflow(prefs.embedWorkflowInImages);
          writeEmbedWorkflowInImages(prefs.embedWorkflowInImages);
        }
        if (
          !workflowEmbedRef.current.seen &&
          typeof prefs?.workflowEmbedNoticeSeen === "boolean"
        ) {
          setWorkflowEmbedNoticeSeen(prefs.workflowEmbedNoticeSeen);
          writeWorkflowEmbedNoticeSeen(prefs.workflowEmbedNoticeSeen);
          if (prefs.workflowEmbedNoticeSeen) {
            workflowEmbedRef.current.seen = true;
            setWorkflowEmbedNoticeOpen(false);
            settlePendingWorkflowGeneration(true);
          }
        }
        const persistedView = prefs?.activeView;
        if (navSections.some((section) => section.items.some((item) => item.id === persistedView))) {
          setActiveView(persistedView);
        }
      })
      .catch((error) => {
        if (!cancelled && error instanceof UpgradeError) {
          setError(`Saved studio settings could not be restored: ${error.message}`);
        }
      })
      .finally(() => {
        if (!cancelled) {
          // Only now may the disclosure fire: before this, `seen` is whatever localStorage said,
          // which on the desktop shell is a fresh empty store every launch.
          workflowEmbedRef.current.hydrated = true;
          for (const resolve of workflowEmbedHydrationWaitersRef.current.splice(0)) {
            resolve();
          }
          setNavigationHydrated(true);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [setAccent, setTheme, setError, settlePendingWorkflowGeneration]);

  const refreshHealth = useCallback(() => {
    healthRequestRef.current?.abort();
    const controller = new AbortController();
    healthRequestRef.current = controller;
    return systemApi.health("", { signal: controller.signal })
      .then((nextHealth) => {
        if (healthRequestRef.current === controller) {
          setHealth(nextHealth);
        }
      })
      .catch((err) => {
        if (!isAbortError(err) && healthRequestRef.current === controller) {
          setHealth(null);
          setError(err.message);
        }
      });
  }, [setError]);

  useEffect(() => {
    refreshHealth();
    // The /api/v1/access probe (+ accessResolved release) lives in useAccessGate now (sc-9750).
    return () => healthRequestRef.current?.abort();
  }, [refreshHealth]);

  useEffect(() => {
    if (!isDesktopShell) {
      return;
    }
    tauriInvoke("get_storage_setup")
      .then((setup) => setSetupCompleted(Boolean(setup?.setupCompleted)))
      // Never block the app on a storage-state read failure; fall through to the studio.
      .catch(() => setSetupCompleted(true));
  }, []);

  useWebView2RepaintPump();

  const { refreshAssets, hydrateLaunchDomains, context: shellHydrationContext } = useShellHydration({
    ready, token, activeProject, setAssets, setLoadedAssetsProjectId, setAssetLoadState, setCharacters, setSavedVoices, setPersonTracks, setTimelines, setTimelinesProjectId, setPresets, setPromptBatches, setLoras, setTrainingDatasets, setTrainingDatasetsProjectId, setTrainingDatasetsError, setSelectedTimelineId, setActiveTimeline, setMacCapabilitiesAuthoritative, setMacCapabilitiesLoading, setMacCapabilities, setMacCapabilitiesError, setDomainError, setProjects, setActiveProject, setProjectsLoaded, setJobs, setQueueSummary, setWorkers, refreshModels, domainErrors, setTrainingTargets, setTrainingTargetsError, setTrainingPresets, setTrainingPresetsError, activeProjectRef, loadedAssetsProjectId, setSelectedAssetId, refreshLoras, setupCompleted, authenticated, uiMode, simpleActiveScreen, activeView, refreshCharacters, refreshSavedVoices, refreshPresets, refreshPromptBatches, refreshTrainingDatasets, refreshPersonTracks, refreshTimelines, projectsLoaded, activeAssetLoadState, hasLibraryConsumerAsset
  });
  // sc-11231 (F-037): useJobEvents captures whichever callback existed at subscribe time
  // (its SSE effect deps are only [access.authRequired, ready, token]), so this MUST have a
  // stable identity — else the live stream keeps calling a stale closure. It reads no props
  // directly, only the two live refs (kept in sync by the effects above), so an empty-dep
  // useCallback is both stable and always current — the `stableRefreshData` ref-delegation
  // pattern used elsewhere in this file, expressed inline here since the whole body is refs.
  const hasVisibleLocalFailure = useCallback((job) => {
    return hasVisibleLocalFailureForView(
      activeViewRef.current,
      localGenerationJobIdsRef.current,
      job,
    );
  }, []);

  // Live job/worker/queue SSE stream, extracted to a hook (sc-9750). The handlers reach
  // back into App state through the identity-stable setters/refs/callbacks passed here;
  // the hook re-subscribes only on [access.authRequired, ready, token] (see useJobEvents).
  useJobEvents({
    access,
    ready,
    token,
    jobsRef,
    setJobs,
    setWorkers,
    setQueueSummary,
    setLatestGenerationSetId,
    setError,
    pushNotice,
    dismissNoticeKind,
    generatedAssetRefreshesRef,
    refreshAssets,
    refreshModels,
    refreshModelAndLoras,
    refreshPersonTracks,
    activeProjectRef,
    enqueueTimelineGenerationApply,
    hasVisibleLocalFailure,
    refreshHealth,
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
      const created = await projectsApi.projects(token, {
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

  const { context: promptRunnersContext } = usePromptRunners({
    token
  });

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

  const { sendAssetToImageEdit, sendAssetToImageEditor, launchImageRecipe, sendAssetRecipeToStudio, sendAssetRecipeToQueue, context: sendToStudioContext } = useSendToStudio({
    setSelectedAssetId, setStudioLaunch, setActiveView, imageModels, setEditorLaunch, uiMode, uiModeLocked, setUiModeOverride, activeProjectRef, stableHydrateLaunchDomains: hydrateLaunchDomains, closePreview, token, requestedGpu, models, assets, confirmWorkflowEmbeddingBeforeGeneration, setJobs
  });

  const { updateAssetStatus, deletePreviewAsset, previewAssetInDirection, purgePreviewAsset, importAsset, context: assetActionsContext } = useAssetActions({
    token, setAssets, setError, refreshCharacters, setSelectedAssetId, previewNavigation, previewDirectionRef, setPreviewAsset, closePreview, purgeAssetRef, activeProject, activeProjectRef
  });

  // Drop a shared image anywhere → "Workflow found" → prefill Image Studio (sc-15951).
  //
  // The guard below is the unchanged issue-#1308 fallback: it swallows any drag no in-app
  // dropzone claimed so the webview cannot navigate to the file. The only addition is that an
  // unclaimed drop of exactly one file is now also offered to the inspector, on that SAME
  // "nothing else claimed it" branch — the five real dropzones (AssetPicker, the Image Editor
  // canvas, DatasetAddDialog, DocumentStudio, PoseLibraryScreen) each preventDefault() as the
  // event bubbles through them, so their drops never reach either behaviour.
  //
  // The envelope kind selects Image or Video; both use the studio's ordinary recipe injection.
  // `importAsset` is the ordinary upload path, which is what records `extra.importedWorkflow`
  // on the asset (sc-15949).
  // The images the panel's input pickers may offer (sc-15952). The same filter Image Studio's own
  // source/reference pickers use (`editImageAssets`), because a pick made here has to be selectable
  // there — offering a trashed asset would seed a control whose list does not contain it.
  const workflowInputAssets = useMemo(
    () =>
      assets.filter(
        (asset) =>
          (asset.type === "image" || asset.type === "frame" || asset.type === "tensor") &&
          asset.projectId === activeProject?.id &&
          !asset.status?.trashed &&
          !asset.status?.rejected,
      ),
    [assets, activeProject?.id],
  );
  // What this install has ON DISK, as one string (sc-15952).
  //
  // The signal behind "install a missing model, and the open panel re-resolves without a reload".
  // A completed `model_download` / `lora_download` makes `useJobEvents` refetch both catalogs, and
  // this changes only when an install STATE does — so a routine catalog refetch that changed
  // nothing does not send the panel back to the API, while a finished download always does.
  const catalogRevision = useMemo(
    () =>
      [
        ...models.map((model) => `m:${model.id}:${model.installState}`),
        ...models.flatMap((model) => Object.entries(model.conditionalPurposes ?? {}).map(([purpose, state]) => `c:${model.id}:${purpose}:${state.installState}:${Boolean(state.blocked)}`)),
        ...loras.map((lora) => `l:${lora.id}:${lora.scope ?? ""}:${lora.installState}`),
      ].join("|"),
    [models, loras],
  );
  // The downloads that FAILED (sc-15952). `catalogRevision` above deliberately cannot carry these:
  // a failed download changes no install state, so the string it produces is identical to the one
  // before the user pressed the button — which is exactly why the panel's "Queued" flag had no way
  // back and its retry button stayed disabled forever. This is the other half of the same job
  // stream, one hop away through the list `useJobEvents` already maintains.
  const failedInstallJobIds = useMemo(
    () =>
      jobs
        .filter(
          (job) =>
            (job.type === "model_download" || job.type === "lora_download") &&
            // "canceled" and "interrupted" are failures for this purpose too: no catalog change,
            // and the user's next move is the same. Only "completed" leaves the flag latched, and
            // that one re-resolves through `catalogRevision` instead.
            terminalStatuses.has(job.status) &&
            job.status !== "completed",
        )
        .map((job) => job.id),
    [jobs],
  );
  const workflowDrop = useWorkflowDrop({
    // Inspecting needs a live, authenticated API. Before the gate opens every request 401s, and
    // a drop on the login screen has nothing to prefill anyway.
    enabled: gateStatus === "open",
    projectId: activeProject?.id ?? null,
    token,
    importAsset,
    launchRecipe: launchImageRecipe,
    // The list the studio's own picker is built from, so a substitute chosen in the panel cannot be
    // a row the picker would drop on the next render (sc-15952).
    models: imageModels,
    assets,
    // The whole catalog, so an install requirement resolves to its real entry — including a VIDEO
    // model, which `imageModels` filters out — before the download starts (sc-17227).
    catalogModels: models,
    // The LoRA catalog, for the same reason: `createLoraDownloadJob` gates on the row's
    // server-stamped `licenseAcknowledgmentModelId`, which a `{ id }` stub does not carry
    // (sc-17227).
    catalogLoras: loras,
    macCapabilities,
    catalogRevision,
    failedInstallJobIds,
    installModel: createModelDownloadJob,
    installLora: createLoraDownloadJob,
  });
  // The viewer's "Use this recipe" for an IMPORTED asset. The preview closes first, the way the
  // recorded-recipe path does — the offer is a modal of its own, and stacking it over the viewer
  // would trap focus between two dialogs.
  const offerAssetWorkflow = workflowDrop.offerAssetWorkflow;
  const useImportedAssetRecipe = useCallback(
    (asset) => {
      closePreview();
      offerAssetWorkflow(asset);
    },
    // Depends on the CALLBACK, not the hook's return object, which is a fresh literal every render
    // — an unstable prop here would defeat `FullscreenPreview`'s `React.memo` on every App render.
    [closePreview, offerAssetWorkflow],
  );
  useDropNavigationGuard({ onUnclaimedFileDrop: workflowDrop.handleDroppedFile });
  // Rendered from a single place even though the app has two shells: `Modal` portals to
  // <body>, so this element does not have to sit inside whichever shell is on screen.
  const workflowDropPanel = workflowDrop.offer ? (
    <WorkflowDropPanel
      assets={workflowInputAssets}
      canImport={Boolean(activeProject)}
      importState={workflowDrop.importState}
      missing={workflowDrop.missing}
      offer={workflowDrop.offer}
      onDismiss={workflowDrop.dismiss}
      onImport={workflowDrop.importImage}
      onUse={workflowDrop.useWorkflow}
    />
  ) : null;

  // The unavailable-model-library prompt and its post-relocation restart disclosure (sc-19709).
  // Built once and rendered in BOTH shells for the same reason `workflowDropPanel` is: the handler
  // that opens them is registered at App level and therefore fires in both, so a Simple-UI user
  // whose library is disconnected would otherwise have their submission swallowed by a dialog that
  // was never mounted — a silently dead Generate button. Both portal to <body>, so neither
  // disturbs the shell it renders in.
  const modelLibraryOverlays = (
    <>
      {["blocked", "retrying", "relocating"].includes(modelLibraryState?.status) ? (
        <ModelLibraryDialog
          autoProbePaused={modelLibraryPickerOpen}
          canRelocate={isDesktopShell}
          onCancel={modelLibraryGate.cancel}
          onRelocate={chooseModelLibraryLocation}
          onRetry={modelLibraryGate.retry}
          state={modelLibraryState}
        />
      ) : null}
      <ModelLibraryRestartDialog
        canRestart={isDesktopShell}
        onLater={() => setModelLibraryRelocation(null)}
        // The rejection is RE-THROWN after the notice: the dialog needs it to leave its restarting
        // state, or a restart that never launched would leave the disclosure permanently stuck on
        // "Restarting SceneWorks…" with both buttons disabled, for this and every later relocation.
        onRestart={() =>
          tauriInvoke("restart_app").catch((error) => {
            pushNotice("general", `SceneWorks could not restart: ${String(error)}`);
            throw error;
          })
        }
        relocation={modelLibraryRelocation}
        runningJobCount={runningJobCount}
      />
    </>
  );

  const appUpdate = useAppUpdate(token, authenticated);

  const { context: jobActionsContext } = useJobActions({
    token, setJobs, setError
  });

  const titleInfo = viewTitles[activeView] ?? { title: activeView, blurb: "" };
  // A keep-alive view is rendered once it is active OR has been visited before
  // (sc-11959): it mounts on first visit and then stays mounted (hidden) thereafter.
  const keepAliveMounted = (view) => activeView === view || visitedKeepAliveViews.has(view);
  // Activity dots only — counts live in the topbar so nav button textContent stays clean.
  // The Editor dot means editor work is IN FLIGHT (export / frame extract / timeline-
  // initiated generation). It used to be `timelines.length > 0`, which lit permanently
  // once a project had a single saved timeline and so never signalled activity at all.
  const activeIndicators = {
    Editor: jobs.some((job) => !terminalStatuses.has(job.status) && isEditorJob(job)),
    Queue: queueCounts.active > 0,
  };
  // First-run gate: until at least one workspace exists, replace the studio area
  // with a create prompt so navigation never lands on dead, project-scoped controls.
  const needsFirstProject =
    authenticated &&
    projectsLoaded &&
    projects.length === 0 &&
    activeView !== "DatasetCatalogs";
  // Desktop first-run wizard (sc-1473): supersedes the project gate while the
  // completion marker is unset. `null` means we're still reading the marker on
  // desktop — hold the studio/gate back briefly to avoid a flash.
  const catalogBypassesProjectSetup = activeView === "DatasetCatalogs";
  const setupGateLoading =
    isDesktopShell && setupCompleted === null && !catalogBypassesProjectSetup;
  const showSetupWizard =
    isDesktopShell &&
    setupCompleted === false &&
    authenticated &&
    !catalogBypassesProjectSetup;
  // The Simple shell replaces the workspace only once the app is actually usable. The
  // first-run flows — the desktop setup wizard and the "create your first workspace"
  // gate — stay on the existing shell that owns them, so Simple never renders studios
  // whose Generate could only fail. Once past them the switch is live, and the Settings
  // toggle + the sidebar switch both reach it. (The remote-access gate used to be a
  // clause here too; it now short-circuits BOTH shells from the early return below.)
  const shellGated = showSetupWizard || setupGateLoading || needsFirstProject;
  const showSimpleShell = uiMode === SIMPLE_MODE && !shellGated;

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
  //     training/timelines/models). Its domains exclude jobs, workersById and
  //     other live values — that separation is the whole point:
  //     cold-only consumers reading via useAppStatic() no longer re-render on job ticks.
  // Owner-memoized domain values are composed without mirrored dependency lists.
  const appLiveValue = useAppDomainValue({
    // Jobs / queue (high-churn — new identity per SSE tick)
    jobs,
    filteredJobs,
    imageLocalJobs,
    videoLocalJobs,
    audioLocalJobs,
    documentLocalJobs,
    // Workers / GPU (high-churn — new identity per worker SSE tick)
    visibleWorkers,
    workersById,
    personReadiness,
    gpuOptions,
  });

  const shellContext = useAppDomainValue({
    activeProject,
    mediaAssets,
    setPreviewAsset: openPreview,
    assets,
    assetsReady: Boolean(
      activeProject &&
      loadedAssetsProjectId === activeProject.id &&
      activeAssetLoadState.status === "ready"
    ),
    assetsLoading: activeAssetLoadState.status === "loading",
    assetsError: activeAssetLoadState.error,
    selectedAsset,
    selectedAssetId,
    setSelectedAssetId,
    latestImageAssets,
    projectFilter,
    setProjectFilter,
    projects,
    confirmWorkflowEmbeddingBeforeGeneration,
    latestVideoAssets,
    recentImageAssets,
    recentVideoAssets,
    recentAudioAssets,
    studioLaunch,
    editorLaunch,
    rememberLocalGenerationJob,
    imageModels,
    videoModels,
    audioModels,
    macCapabilities,
    macCapabilitiesAuthoritative,
    macCapabilitiesError,
    macCapabilitiesLoading,
    requestedGpu,
    setRequestedGpu,
    token,
    authenticated,
    trainingPresets,
    trainingPresetsError,
    trainingTargets,
    trainingTargetsError,
    setActiveView,
    registerLeaveGuard,
    registerProjectSwitchGuard,
    trackEditorScratchOp,
    releaseEditorScratchOp,
    registerEditorScratchClaim,
    preferencesHydrated: navigationHydrated
  });
  const appStaticValue = useAppDomainValue({
    ...shellContext,
    ...charactersContext,
    ...presetsContext,
    ...savedVoicesContext,
    ...promptBatchesContext,
    ...trainingContext,
    ...modelsAndLorasContext,
    ...personTracksContext,
    ...timelinesContext,
    ...promptRunnersContext,
    ...assetActionsContext,
    ...sendToStudioContext,
    ...jobActionsContext,
    ...jobCreatorsContext,
    ...shellHydrationContext,
    ...themeAccentContext
  });

  // Remote-access blocker (sc-15102). Returned INSTEAD of either shell — no sidebar, no
  // topbar, no screens — so a remote browser that hasn't unlocked the host cannot walk
  // the nav tree through screens that are empty by design (every protected load is held
  // until `authenticated`). It replaces the `.auth-band` banner that used to render above
  // a fully live, fully empty shell. Placed after the providers' useMemos so the hook
  // order is identical whether or not the gate is up.
  if (gateStatus !== "open") {
    return (
      <AccessGate
        authError={authError}
        notices={notices}
        onForget={lockRemote}
        onSubmit={saveToken}
        passwordDraft={passwordDraft}
        setPasswordDraft={setPasswordDraft}
        status={gateStatus}
      />
    );
  }

  // Simple UI (design handoff): an ALTERNATIVE shell rendered in place of the workspace
  // below. It mounts INSIDE the same two providers, so its screens read the identical
  // catalogs, job feed and actions — there is no second data layer, and switching back
  // to Advanced lands on the same live state.
  const workspaceSwitcher = (
    <ProjectSwitcher
      activeProject={activeProject}
      collapsed={sidebarCollapsed && !uiModeLocked}
      disabled={!authenticated}
      onCreate={createProject}
      onOpen={() => { if (sidebarCollapsed && !uiModeLocked) changeSidebarCollapsed(false); }}
      onSelect={selectProject}
      projects={projects}
    />
  );

  if (showSimpleShell) {
    return (
      <AppStaticContext.Provider value={appStaticValue}>
        <AppLiveContext.Provider value={appLiveValue}>
          <SimpleShell
            // Flush the old workspace's draft and reset transient selections/overlays.
            key={activeProject?.id}
            appVersion={<AppVersion update={appUpdate} />}
            accent={accent}
            embedWorkflow={embedWorkflow}
            lockedToSimple={uiModeLocked}
            onAccentChange={changeAccent}
            onEmbedWorkflowChange={changeEmbedWorkflow}
            onModeChange={setUiModeOverride}
            onScreenChange={setSimpleActiveScreen}
            onSimpleDefaultChange={changeSimpleUiDefault}
            requestedScreen={simpleActiveScreen}
            settingsSharingFocusRequest={settingsSharingFocusRequest}
            /* The same flag that gates the durable `activeView` write: until the
               ui-preferences GET has landed, the Simple studios must neither restore from
               a possibly-empty cache nor write their catalog defaults over the stored copy. */
            preferencesHydrated={navigationHydrated}
            sidebarCollapsed={sidebarCollapsed}
            onSidebarCollapsedChange={changeSidebarCollapsed}
            simpleDefault={simpleUiDefault}
            workspaceSwitcher={workspaceSwitcher}
          />
          {workflowEmbedNoticeOpen ? (
            <WorkflowEmbedDecisionModal
              error={workflowEmbedDecisionError}
              onChoose={chooseWorkflowEmbedding}
              onOpenSettings={openWorkflowSharingSettings}
              saving={workflowEmbedDecisionSaving}
            />
          ) : null}
          {/* The dropped-workflow offer (sc-15951). The drop guard is installed at App level
              and therefore runs in BOTH shells, so the panel it opens has to render in both
              too — otherwise a Simple-UI user's drop would be inspected and then answered by
              nothing. It portals to <body>, so it does not disturb this shell's layout. */}
          {workflowDropPanel}
          {modelLibraryOverlays}
        </AppLiveContext.Provider>
      </AppStaticContext.Provider>
    );
  }

  return (
    <AppStaticContext.Provider value={appStaticValue}>
    <AppLiveContext.Provider value={appLiveValue}>
    <main className={sidebarCollapsed ? "app sidebar-collapsed" : "app"}>
      <aside className="sidebar" aria-label="Primary">
        <button
          aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          aria-expanded={!sidebarCollapsed}
          className="icon-btn sidebar-toggle"
          onClick={() => changeSidebarCollapsed(!sidebarCollapsed)}
          title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          type="button"
        >
          <Icon.Sidebar collapsed={sidebarCollapsed} />
        </button>
        <div className="sidebar-scroll">
          <div className="brand">
            <span className="brand-mark" aria-hidden="true">
              <Logo size={32} />
            </span>
            <div>
              <h1>Scene<span className="light">Works</span></h1>
              <p>Local creative studio</p>
            </div>
          </div>

          {workspaceSwitcher}

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
                      aria-label={label}
                      aria-current={activeView === item.id ? "page" : undefined}
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

        </div>

        {/* Simple ⇄ Advanced switch (design handoff). The ONE addition this shell makes for
            the Simple UI; the same component renders in the Simple sidebar footer, so the
            control can never present differently between the two. */}
        <div className="sidebar-footer">
          <div className="sidebar-footer-details"><AppVersion update={appUpdate} /></div>
          {sidebarCollapsed ? (
            <button
              aria-label="Switch to Simple UI"
              className="icon-btn"
              onClick={() => setUiModeOverride(SIMPLE_MODE)}
              title="Switch to Simple UI"
              type="button"
            >
              <Icon.Sparkle />
            </button>
          ) : (
            <SimpleModeSwitch
              locked={uiModeLocked}
              mode={ADVANCED_MODE}
              onChange={setUiModeOverride}
            />
          )}
        </div>
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

        {/* Every notice is dismissible. Most kinds clear themselves (a retry succeeds, the
            next run completes), but a terminal one — a failed job's panic message — is
            pushed once and never re-pushed, so without an X it sat above every screen for
            the rest of the session. Dismissing removes only this kind; the same kind
            failing again re-raises it. */}
        {notices.map((notice) => (
          <div className="notice error" key={notice.kind} role="alert">
            <span className="notice-message">{notice.message}</span>
            <button
              aria-label="Dismiss this message"
              className="notice-dismiss"
              onClick={() => dismissNoticeKind(notice.kind)}
              title="Dismiss"
              type="button"
            >
              <Icon.Close size={15} />
            </button>
          </div>
        ))}

        {workflowEmbedNoticeOpen ? (
          <WorkflowEmbedDecisionModal
            error={workflowEmbedDecisionError}
            onChoose={chooseWorkflowEmbedding}
            onOpenSettings={openWorkflowSharingSettings}
            saving={workflowEmbedDecisionSaving}
          />
        ) : null}

        {showSetupWizard ? (
          <SetupWizard
            token={token}
            jobs={jobs}
            macCapabilities={macCapabilities}
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
        {authenticated && activeView === "DatasetCatalogs" ? <DatasetCatalogsScreen /> : null}
        {/* Accent + the Simple-mode default live in App state alongside the sidebar's mode
            switch (they are not on the app context), so Settings receives them as props —
            the same pair SimpleShell gets, writing through to the same ui-preferences store. */}
        {activeView === "Settings" ? (
          <SettingsScreen
            accent={accent}
            appUpdate={appUpdate}
            embedWorkflow={embedWorkflow}
            lockedToSimple={uiModeLocked}
            onAccentChange={changeAccent}
            onChangeModelLibrary={changeModelLibrary}
            onEmbedWorkflowChange={changeEmbedWorkflow}
            onSimpleDefaultChange={changeSimpleUiDefault}
            sharingFocusRequest={settingsSharingFocusRequest}
            simpleDefault={simpleUiDefault}
          />
        ) : null}
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
            <ImageStudio key={`${activeProject?.id ?? "default"}:${studioRestoreEpoch}`} />
          </KeepAlivePane>
        ) : null}

        {keepAliveMounted("Video") ? (
          <KeepAlivePane active={activeView === "Video"}>
            <VideoStudio key={`${activeProject?.id ?? "default"}:${studioRestoreEpoch}`} />
          </KeepAlivePane>
        ) : null}

        {keepAliveMounted("Audio") ? (
          <KeepAlivePane active={activeView === "Audio"}>
            <AudioStudio key={`${activeProject?.id ?? "default"}:${studioRestoreEpoch}`} />
          </KeepAlivePane>
        ) : null}

        {keepAliveMounted("Characters") ? (
          <KeepAlivePane active={activeView === "Characters"}>
            <CharacterStudio key={`${activeProject?.id ?? "default"}:${studioRestoreEpoch}`} />
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
            {/* Deliberately NOT keyed on the studio restore epoch (sc-15425), unlike the four
                studios. The Editor holds timeline edit state and live scratch-op claims
                (registerEditorScratchClaim), so remounting it to recover its `editor-video`
                generation settings would risk discarding real editing work and leaking a claim —
                a far worse trade than the settings it would recover. It picks up a restored
                snapshot on its next mount instead. */}
            <EditorScreen />
          </KeepAlivePane>
        ) : null}

        {keepAliveMounted("ImageEditor") ? (
          <KeepAlivePane active={activeView === "ImageEditor"}>
            <ImageEditor key={activeProject?.id ?? "default"} />
          </KeepAlivePane>
        ) : null}
        {keepAliveMounted("VectorStudio") ? (
          <KeepAlivePane active={activeView === "VectorStudio"}>
            <VectorStudio key={activeProject?.id ?? "default"} />
          </KeepAlivePane>
        ) : null}
          </>
        )}
      </section>

      {previewedAsset ? (
        <FullscreenPreview
          asset={previewedAsset}
          deleteAsset={deletePreviewAsset}
          nextAsset={previewNavigation.next}
          onClose={closePreview}
          onEditImage={sendAssetToImageEditor}
          onEditInStudio={sendAssetToImageEdit}
          onPreviewAsset={previewAssetInDirection}
          onRegenerate={sendAssetRecipeToQueue}
          onUseImportedRecipe={useImportedAssetRecipe}
          onUseRecipe={sendAssetRecipeToStudio}
          previousAsset={previewNavigation.previous}
          sourceAsset={previewSourceAsset}
          purgeAsset={purgePreviewAsset}
          updateAssetStatus={updateAssetStatus}
        />
      ) : null}

      {/* Desktop-safe confirm dialog host (sc-11968). Mounted once at the app root so
          appConfirm()/useConfirm() anywhere — including a leave-guard callback handed to
          navTo — resolve through a real React dialog instead of window.confirm (which
          silently no-ops in the Tauri WebView). Renders nothing until a confirm is asked. */}
      <ConfirmHost />

      {/* The unavailable-model-library prompt and its post-relocation restart disclosure
          (sc-19709), built above and rendered in BOTH shells. Renders nothing while the gate is
          idle, which is every normal run. */}
      {modelLibraryOverlays}

      {/* "Workflow found" (sc-15951) — opened by a drop no in-app dropzone claimed, and only
          after the file turned out to carry a recipe. Renders nothing otherwise, which is the
          common case for every image that did not come from SceneWorks. */}
      {workflowDropPanel}
    </main>
    </AppLiveContext.Provider>
    </AppStaticContext.Provider>
  );
}
