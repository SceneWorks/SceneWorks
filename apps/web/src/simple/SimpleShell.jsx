import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Icon } from "../components/Icons.jsx";
import { Logo } from "../components/Logo.jsx";
import { ConfirmHost } from "../appConfirm.jsx";
import { useAppContext } from "../context/AppContext.js";
import { assetCanRenderAsAudio } from "../components/assetMedia.jsx";
import { terminalStatuses } from "../constants.js";
import { breakpointFor, contentMaxWidth } from "./breakpoint.js";
import { ADVANCED_MODE } from "./uiMode.js";
import { useContainerWidth } from "./useContainerWidth.js";
import { SimpleUiContext } from "./SimpleUiContext.js";
import { SimpleSheet, SimpleSheetOptions } from "./SimpleSheet.jsx";
import { PromptGuideSheet } from "./PromptGuideSheet.jsx";
import { SimplePreview } from "./SimplePreview.jsx";
import { SimpleModeSwitch } from "./SimpleModeSwitch.jsx";
import { SimpleImageStudio } from "./SimpleImageStudio.jsx";
import { SimpleVideoStudio } from "./SimpleVideoStudio.jsx";
import { SimpleAudioStudio } from "./SimpleAudioStudio.jsx";
import { SimpleAssets } from "./SimpleAssets.jsx";
import { SimpleModelManager } from "./SimpleModelManager.jsx";
import { SimpleQueue } from "./SimpleQueue.jsx";
import { SimpleSettings } from "./SimpleSettings.jsx";
import { SimpleLicenses } from "./SimpleLicenses.jsx";
import "./simple.css";

// The Simple UI shell (design handoff "Simple UI for creative studios").
//
// One responsive layout with three container-measured bands: a phone gets a top bar with
// a hamburger and an overlay drawer, tablet/desktop get a persistent 244px sidebar and a
// centered content column (680px / 900px). The shell owns the four overlays every screen
// shares — the picker sheet, the prompt guide, the asset preview and the toast — and
// publishes them on SimpleUiContext so a screen never re-implements one.
//
// It renders INSIDE App's providers, so every screen reads the same live catalogs, job
// feed and actions the full workspace does. There is no parallel data layer.

export const SIMPLE_SCREENS = [
  {
    group: "Studios",
    items: [
      { id: "image", label: "Image", icon: Icon.Image, title: "Image Studio", sub: "Text & Edit" },
      { id: "video", label: "Video", icon: Icon.Video, title: "Video Studio", sub: "Text & Image to Video" },
      { id: "audio", label: "Audio", icon: Icon.Audio, title: "Audio Studio", sub: "Music · Speech · SFX" },
      { id: "assets", label: "Assets", icon: Icon.Library, title: "Assets", sub: "Images, clips & audio" },
    ],
  },
  {
    group: "Manage",
    items: [
      { id: "models", label: "Model Manager", icon: Icon.Model, title: "Model Manager", sub: "Download & manage checkpoints" },
      { id: "queue", label: "Queue", icon: Icon.Queue, title: "Queue", sub: "Running & recent jobs" },
      { id: "settings", label: "Settings", icon: Icon.Sliders, title: "Settings", sub: "Appearance, generation & device" },
      { id: "licenses", label: "Licenses", icon: Icon.Book, title: "Licenses", sub: "Software & model terms" },
    ],
  },
];

const SCREEN_BY_ID = new Map(SIMPLE_SCREENS.flatMap((group) => group.items).map((item) => [item.id, item]));

const TOAST_MS = 1900;

export function SimpleShell({
  accent,
  onAccentChange,
  simpleDefault,
  onSimpleDefaultChange,
  onModeChange,
  lockedToSimple,
}) {
  const {
    jobs = [],
    assets = [],
    deleteAsset,
    updateAssetStatus,
    setSelectedAssetId,
    setActiveView,
    sendAssetToVideo,
  } = useAppContext();

  const [rootRef, width] = useContainerWidth();
  const breakpoint = breakpointFor(width);
  const phone = breakpoint === "phone";

  const [screen, setScreen] = useState("image");
  const [drawer, setDrawer] = useState(false);
  const [sheet, setSheet] = useState(null);
  const [guideOpen, setGuideOpen] = useState(false);
  const [preview, setPreview] = useState(null);
  // Which audio asset the Assets viewer holds (epic 14361 / sc-14366). It lives HERE, beside
  // `preview`, so the grid and the viewer agree on which clip is loaded — the grid rings the
  // loaded tile while the viewer IS that clip's play deck. Only the ID is shell state: the
  // transport itself lives in the viewer, so a `timeupdate` can't re-render every screen.
  const [loadedTakeId, setLoadedTakeId] = useState(null);
  const [audioAutoPlay, setAudioAutoPlay] = useState(false);
  const [toastText, setToastText] = useState(null);
  // The reference an asset preview hands to the Image Studio. Held here (not in the
  // studio) so "Use as reference" works from Assets, which is a different screen.
  const [referenceRequest, setReferenceRequest] = useState(null);
  // Failed runs the user has dismissed from a studio. Held HERE rather than in the studio
  // because the studios unmount on navigation — studio-local state would resurrect a
  // dismissed failure the moment the user came back to the tab. Session-scoped by design:
  // the Queue keeps the row permanently, so nothing is actually lost.
  const [dismissedJobIds, setDismissedJobIds] = useState(() => new Set());
  const dismissJob = useCallback((id) => {
    setDismissedJobIds((current) => new Set(current).add(id));
  }, []);
  const toastTimer = useRef(null);

  useEffect(() => () => clearTimeout(toastTimer.current), []);

  // Opening an asset. An audio asset also becomes the loaded take, so the grid rings it and
  // the viewer opens on its deck; `play` (the tile's play button) starts it immediately.
  const openPreview = useCallback((asset, options = {}) => {
    if (!asset) {
      setPreview(null);
      return;
    }
    if (assetCanRenderAsAudio(asset)) {
      setLoadedTakeId(asset.id);
      setAudioAutoPlay(Boolean(options.play));
    } else {
      setAudioAutoPlay(false);
    }
    setPreview(asset);
  }, []);

  const toast = useCallback((message) => {
    setToastText(message);
    clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToastText(null), TOAST_MS);
  }, []);

  // Navigating closes the drawer and every open overlay (design: "Screen switch clears
  // any open sheet/preview").
  const goTo = useCallback((next) => {
    setScreen(next);
    setDrawer(false);
    setSheet(null);
    setPreview(null);
    setGuideOpen(false);
  }, []);

  // Hand off to a full-workspace screen: point the workspace at `view` AND flip the shell,
  // because setting the view alone would do nothing visible from in here. Refuses (with a
  // reason) when the viewport has Simple locked on, so a control can never look inert.
  const openInAdvanced = useCallback(
    (view) => {
      if (lockedToSimple) {
        toast("That opens in the Advanced workspace — switch on a larger screen.");
        return false;
      }
      setActiveView?.(view);
      onModeChange(ADVANCED_MODE);
      return true;
    },
    [lockedToSimple, onModeChange, setActiveView, toast],
  );

  const ui = useMemo(
    () => ({
      breakpoint,
      goTo,
      openSheet: setSheet,
      closeSheet: () => setSheet(null),
      openGuide: () => setGuideOpen(true),
      openPreview,
      loadedTakeId,
      openInAdvanced,
      uiLocked: lockedToSimple,
      toast,
      referenceRequest,
      clearReferenceRequest: () => setReferenceRequest(null),
      dismissedJobIds,
      dismissJob,
    }),
    [
      breakpoint,
      goTo,
      toast,
      referenceRequest,
      openInAdvanced,
      lockedToSimple,
      dismissedJobIds,
      dismissJob,
      openPreview,
      loadedTakeId,
    ],
  );

  const active = SCREEN_BY_ID.get(screen) ?? SCREEN_BY_ID.get("image");
  const queueCount = useMemo(
    () => jobs.filter((job) => !terminalStatuses.has(job.status)).length,
    [jobs],
  );
  // The preview may be showing an asset that has since been deleted/refreshed away;
  // re-resolve it from the live catalog each render so its favorite state stays honest.
  const previewAsset = useMemo(
    () => (preview ? (assets.find((asset) => asset.id === preview.id) ?? preview) : null),
    [preview, assets],
  );

  return (
    <SimpleUiContext.Provider value={ui}>
      <div
        className="su-root"
        data-bp={breakpoint}
        ref={rootRef}
        style={{ "--su-content-max": contentMaxWidth(breakpoint) }}
      >
        {!phone || drawer ? (
          <nav aria-label="Primary" className="su-nav su-scroll">
            <div className="su-brand">
              <Logo size={30} />
              <div>
                <div className="su-brand-name">SceneWorks</div>
                <div className="su-brand-sub">Simple</div>
              </div>
            </div>
            {SIMPLE_SCREENS.map((group) => (
              <React.Fragment key={group.group}>
                <div className="su-nav-group-title">{group.group}</div>
                <div
                  className={
                    group.group === "Studios" ? "su-nav-group su-nav-group--studios" : "su-nav-group"
                  }
                >
                  {group.items.map((item) => {
                    const IconComponent = item.icon;
                    return (
                      <button
                        aria-current={screen === item.id ? "page" : undefined}
                        className={screen === item.id ? "su-nav-item active" : "su-nav-item"}
                        key={item.id}
                        onClick={() => goTo(item.id)}
                        type="button"
                      >
                        <IconComponent size={group.group === "Studios" ? 19 : 18} />
                        <span className="su-nav-label">{item.label}</span>
                        {item.id === "queue" && queueCount ? (
                          <span className="su-nav-badge">{queueCount}</span>
                        ) : null}
                      </button>
                    );
                  })}
                </div>
              </React.Fragment>
            ))}
            <div className="su-nav-footer">
              <SimpleModeSwitch locked={lockedToSimple} mode="simple" onChange={onModeChange} />
            </div>
          </nav>
        ) : null}

        {phone && drawer ? (
          <button aria-label="Close menu" className="su-scrim" onClick={() => setDrawer(false)} type="button" />
        ) : null}

        <div className="su-main">
          <header className="su-topbar">
            {phone ? (
              <button
                aria-label="Open menu"
                className="su-hamburger"
                onClick={() => setDrawer(true)}
                type="button"
              >
                <Icon.Menu size={21} />
              </button>
            ) : null}
            <div className="su-topbar-title">
              <strong>{active.title}</strong>
              <span>{active.sub}</span>
            </div>
          </header>

          <div className="su-body su-scroll">
            <div className="su-content">
              {screen === "image" ? <SimpleImageStudio /> : null}
              {screen === "video" ? <SimpleVideoStudio /> : null}
              {screen === "audio" ? <SimpleAudioStudio /> : null}
              {screen === "assets" ? <SimpleAssets /> : null}
              {screen === "models" ? <SimpleModelManager /> : null}
              {screen === "queue" ? <SimpleQueue /> : null}
              {screen === "settings" ? (
                <SimpleSettings
                  accent={accent}
                  lockedToSimple={lockedToSimple}
                  onAccentChange={onAccentChange}
                  onSimpleDefaultChange={onSimpleDefaultChange}
                  simpleDefault={simpleDefault}
                />
              ) : null}
              {screen === "licenses" ? <SimpleLicenses /> : null}
            </div>
          </div>
        </div>

        {sheet ? (
          <SimpleSheet onClose={() => setSheet(null)} phone={phone} title={sheet.title}>
            <SimpleSheetOptions
              kind={sheet.kind}
              onSelect={(value) => {
                sheet.onSelect(value);
                setSheet(null);
              }}
              options={sheet.options}
            />
          </SimpleSheet>
        ) : null}

        {guideOpen ? <PromptGuideSheet onClose={() => setGuideOpen(false)} phone={phone} /> : null}

        {previewAsset ? (
          <SimplePreview
            asset={previewAsset}
            audioAutoPlay={audioAutoPlay}
            breakpoint={breakpoint}
            onClose={() => setPreview(null)}
            onSendToVideo={
              sendAssetToVideo
                ? (asset) => {
                    if (openInAdvanced("Video")) {
                      sendAssetToVideo(asset);
                    }
                  }
                : null
            }
            onDelete={async (asset) => {
              setPreview(null);
              await deleteAsset(asset);
              toast("Moved to trash");
            }}
            onToggleFavorite={(asset) => {
              updateAssetStatus(asset, { favorite: !asset.status?.favorite });
              toast(asset.status?.favorite ? "Removed from favorites" : "Added to favorites");
            }}
            onUseAsReference={(asset) => {
              setSelectedAssetId?.(asset.id);
              setReferenceRequest({ id: asset.id, token: asset.id + Date.now() });
              setPreview(null);
              setScreen("image");
              toast("Added as reference in Image Studio");
            }}
          />
        ) : null}

        {toastText ? (
          <div aria-live="polite" className="su-toast" role="status">
            <span>{toastText}</span>
          </div>
        ) : null}

        {/* Desktop-safe confirm host (sc-11968): appConfirm() resolves through a real React
            dialog rather than window.confirm, which no-ops in the Tauri WebView. Mounted here
            so a Simple-side action that awaits a confirm still works. */}
        <ConfirmHost />
      </div>
    </SimpleUiContext.Provider>
  );
}
