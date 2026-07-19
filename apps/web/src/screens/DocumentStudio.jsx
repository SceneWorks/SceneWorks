import React, { useEffect, useMemo, useRef, useState } from "react";
import { AdvancedSection } from "../components/AdvancedSection.jsx";
import { AssetPickerModal } from "../components/AssetPicker.jsx";
import { AssetThumbnail } from "../components/assetMedia.jsx";
import { DocumentView } from "../components/DocumentView.jsx";
import { Icon } from "../components/Icons.jsx";
import { ModelAvailabilityGate } from "../components/ModelAvailabilityGate.jsx";
import { StudioUpdateBadge, StudioUpdateNotice, updateOptionLabel } from "../components/StudioUpdateNotice.jsx";
import { WorkerProgressCard } from "../components/WorkerProgressCard.jsx";
import { WorkPanel } from "../components/WorkPanel.jsx";
import {
  DEFAULT_INTERLEAVE_IMAGE_GUIDANCE,
  DEFAULT_INTERLEAVE_RESOLUTION,
  DEFAULT_INTERLEAVE_SYSTEM_MESSAGE,
  INTERLEAVE_IMAGE_GUIDANCE_MAX,
  INTERLEAVE_IMAGE_GUIDANCE_MIN,
  INTERLEAVE_IMAGE_GUIDANCE_STEP,
  INTERLEAVE_RESOLUTION_OPTIONS,
} from "../constants.js";
import { useAppContext } from "../context/AppContext.js";
import { DEFAULT_MAC_CAPABILITIES } from "../macGating.js";
import { documentModelUsable, downloadOffersFor } from "../modelEligibility.js";
import { selectStackedJobs } from "./generationStudio.jsx";

const MAX_IMAGES_DEFAULT = 6;
const MAX_IMAGES_LIMIT = 10;

// Quick-start scaffolds (design handoff 1a). Each chip is a *seeder*, not a limiter:
// selecting one prefills the brief with a starter template (and, where it helps, a
// sensible size / max-images default). It never constrains what the model generates —
// the user can rewrite the prompt into anything. "Blank" clears the scaffold. This list
// is illustrative, not exhaustive; keep it easy to extend.
const QUICK_STARTS = [
  { id: "blank", label: "Blank", icon: "Blank", blank: true },
  {
    id: "storyboard",
    label: "Storyboard",
    icon: "Storyboard",
    scaffold:
      "A storyboard for a short film: [describe the story]. Open on [first beat], build to [middle beat], resolve at [final beat]. One consistent look throughout — [style/mood].",
  },
  {
    id: "guide",
    label: "Illustrated guide",
    icon: "Guide",
    scaffold:
      "An illustrated step-by-step guide to [task]. For each step, write a short instruction and show an image of that step. Keep the same clean, consistent illustration style across every step.",
  },
  {
    id: "poster",
    label: "Poster",
    icon: "Poster",
    resolution: "1152x2048",
    maxImages: 1,
    scaffold:
      "A single striking poster for [event / product / idea]. Headline: [text]. One bold hero image plus a short supporting line. Style: [describe the look].",
  },
  {
    id: "ad",
    label: "Ad / landing page",
    icon: "AdPage",
    maxImages: 3,
    scaffold:
      "An ad / landing-page layout for [product]. Lead with a hero shot and a headline, then two or three short benefit blurbs, each with its own supporting image. Consistent brand style: [describe].",
  },
  {
    id: "infographic",
    label: "Infographic / chart",
    icon: "Bars",
    maxImages: 3,
    scaffold:
      "An infographic explaining [topic]. Break it into a few labelled sections, each with a clear chart or diagram image and a one-line takeaway. Clean, legible, consistent visual language.",
  },
  {
    id: "tutorial",
    label: "Tutorial",
    icon: "Preset",
    scaffold:
      "A tutorial that teaches [skill]. Introduce it, then walk through the steps in order — each step gets a short explanation and an image demonstrating it. Finish with a recap.",
  },
  {
    id: "comic",
    label: "Comic strip",
    icon: "Comic",
    maxImages: 6,
    scaffold:
      "A short comic strip about [premise]. Name the characters so they stay consistent. Panel 1: [beat]. Panel 2: [beat]. Panel 3: [beat]. Panel 4: [beat]. One consistent art style throughout.",
  },
  {
    id: "lookbook",
    label: "Lookbook",
    icon: "Lookbook",
    scaffold:
      "A lookbook for [collection / theme]. A short intro, then a series of looks — each with a full image and a one-line caption naming the pieces. Consistent styling and mood across every look.",
  },
];

// Starter chips (design handoff 1a) — one-tap snippets inserted at the caret to help
// the user structure a brief. UI aid only; they just edit the prompt text.
const STARTERS = [
  { label: "Set the scene", snippet: "Open on " },
  { label: "Add a beat", snippet: "Then, " },
  { label: "Name a character", snippet: "The main character is [name], a " },
  { label: "Describe the mood", snippet: "The mood is " },
  { label: "Add a caption line", snippet: 'Caption: "' },
];

const BRIEF_TIPS = [
  "Name each beat you want as an image — the model makes one per beat.",
  "Give an order: “open on…”, “then…”, “finally…”.",
  "State one consistent style so the shots match.",
  "Keep people/props named so they recur.",
];

// Plain-language examples of what a prompt can do *with* reference frames — the point
// being the model reads them for content, it doesn't imitate their style.
const REFERENCE_PROMPT_EXAMPLES = [
  "keep this character",
  "edit what's shown",
  "compose from these",
  "illustrate my text",
];

function modelSupportsInterleave(model) {
  return Array.isArray(model?.capabilities) && model.capabilities.includes("interleave");
}

function formatResolutionLabel(value) {
  const [width, height] = String(value).split("x");
  return height ? `${width} × ${height}` : value;
}

function documentSegments(job) {
  const segments = job.result?.segments;
  return Array.isArray(segments) && segments.length ? segments : null;
}

function clampMaxImages(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return MAX_IMAGES_DEFAULT;
  }
  return Math.min(MAX_IMAGES_LIMIT, Math.max(1, Math.floor(parsed)));
}

export function DocumentStudio() {
  const {
    activeProject,
    assets,
    createInterleaveJob,
    createModelDownloadJob,
    documentLocalJobs = [],
    gpuOptions,
    imageModels,
    importAsset,
    jobs = [],
    jobAction,
    macCapabilities = DEFAULT_MAC_CAPABILITIES,
    models = [],
    rememberLocalGenerationJob,
    setActiveView,
    requestedGpu,
    setRequestedGpu,
  } = useAppContext();
  const onCancelJob = (job) => jobAction(job, "cancel");
  const onOpenQueue = () => setActiveView("Queue");
  const interleaveModels = useMemo(
    () => (imageModels ?? []).filter(modelSupportsInterleave),
    [imageModels],
  );
  // Model-availability gate (sc-5947): when no interleave-capable model is present, show the
  // recommended downloads (SenseNova-U1) instead of the compose form. Offers are mac-aware.
  const modelReady = interleaveModels.length > 0;
  const modelOffers = useMemo(
    () => downloadOffersFor(models, documentModelUsable, macCapabilities),
    [models, macCapabilities],
  );
  const modelDownloadJobs = useMemo(
    () => (jobs ?? []).filter((job) => job.type === "model_download"),
    [jobs],
  );
  const [model, setModel] = useState("");
  const selectedModel = interleaveModels.find((item) => item.id === model) ?? interleaveModels[0] ?? null;
  const [prompt, setPrompt] = useState("");
  const [quickStart, setQuickStart] = useState(null);
  // Ordered, reorderable storyboard of reference frames. Replaces the flat multi-select:
  // order is significant and preserved on submit (sourceAssetIds = frames in order). Each
  // frame carries an optional reference image + a caption. Captions are a UI aid for
  // structuring the brief — the interleave worker has no per-frame caption field, so they
  // stay client-side (see submit()).
  const [storyboardFrames, setStoryboardFrames] = useState([]);
  const [referenceGuidance, setReferenceGuidance] = useState(DEFAULT_INTERLEAVE_IMAGE_GUIDANCE);
  const [maxImages, setMaxImages] = useState(MAX_IMAGES_DEFAULT);
  const [resolution, setResolution] = useState(DEFAULT_INTERLEAVE_RESOLUTION);
  const [systemMessage, setSystemMessage] = useState(DEFAULT_INTERLEAVE_SYSTEM_MESSAGE);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [pickerFrameId, setPickerFrameId] = useState(null);
  const [dragIndex, setDragIndex] = useState(null);
  const [dragOverIndex, setDragOverIndex] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  const promptRef = useRef(null);
  const frameSeq = useRef(0);
  const makeFrame = (caption = "") => ({ id: `frame-${frameSeq.current++}`, assetId: null, caption });

  useEffect(() => {
    if (interleaveModels.length && !interleaveModels.some((item) => item.id === model)) {
      setModel(interleaveModels[0].id);
    }
  }, [interleaveModels, model]);

  const sourceImageAssets = useMemo(
    () => (assets ?? []).filter((asset) => asset.type === "image" || asset.type === "frame" || asset.type === "upload"),
    [assets],
  );
  const assetById = useMemo(() => new Map((assets ?? []).map((asset) => [asset.id, asset])), [assets]);

  // Running and queued compose runs stack (oldest/active on top, queued below) and
  // each run streams its output beneath it, mirroring the Image and Video studios.
  const localJobs = useMemo(() => selectStackedJobs(documentLocalJobs), [documentLocalJobs]);

  const ready = Boolean(activeProject) && interleaveModels.length > 0;
  const canSubmit = ready && prompt.trim().length > 0 && !submitting;

  const referencedFrameCount = storyboardFrames.filter((frame) => frame.assetId).length;
  const pickerFrame = storyboardFrames.find((frame) => frame.id === pickerFrameId) ?? null;

  function applyQuickStart(entry) {
    setQuickStart(entry.id);
    if (entry.blank) {
      setPrompt("");
      return;
    }
    if (entry.scaffold != null) {
      setPrompt(entry.scaffold);
    }
    if (entry.resolution && INTERLEAVE_RESOLUTION_OPTIONS.includes(entry.resolution)) {
      setResolution(entry.resolution);
    }
    if (entry.maxImages != null) {
      setMaxImages(entry.maxImages);
    }
  }

  function insertStarter(snippet) {
    const field = promptRef.current;
    // Insert at the caret when we can reach the live textarea; otherwise append on a new
    // line. Selecting a chip no longer implies a quick-start scaffold is in force.
    setQuickStart(null);
    if (!field) {
      setPrompt((current) => (current ? `${current}\n${snippet}` : snippet));
      return;
    }
    const start = field.selectionStart ?? field.value.length;
    const end = field.selectionEnd ?? field.value.length;
    const before = prompt.slice(0, start);
    const after = prompt.slice(end);
    const separator = before && !before.endsWith("\n") && !before.endsWith(" ") ? "\n" : "";
    const next = `${before}${separator}${snippet}${after}`;
    setPrompt(next);
    const caret = before.length + separator.length + snippet.length;
    requestAnimationFrame(() => {
      field.focus();
      field.setSelectionRange(caret, caret);
    });
  }

  function addFrame() {
    setStoryboardFrames((frames) => [...frames, makeFrame()]);
  }

  function removeFrame(id) {
    setStoryboardFrames((frames) => frames.filter((frame) => frame.id !== id));
  }

  function updateFrameCaption(id, caption) {
    setStoryboardFrames((frames) => frames.map((frame) => (frame.id === id ? { ...frame, caption } : frame)));
  }

  function updateFrameAsset(id, assetId) {
    setStoryboardFrames((frames) => frames.map((frame) => (frame.id === id ? { ...frame, assetId: assetId || null } : frame)));
  }

  function reorderFrame(from, to) {
    setStoryboardFrames((frames) => {
      if (from === to || from == null || to == null || from < 0 || to < 0 || from >= frames.length || to >= frames.length) {
        return frames;
      }
      const next = [...frames];
      const [moved] = next.splice(from, 1);
      next.splice(to, 0, moved);
      return next;
    });
  }

  async function importIntoFrame(id, file) {
    if (!file || !importAsset) {
      return;
    }
    try {
      const imported = await importAsset(file, { throwOnError: true });
      if (imported?.id) {
        updateFrameAsset(id, imported.id);
      }
    } catch {
      // importAsset surfaces its own error into the app banner; nothing to add here.
    }
  }

  function handleFrameDrop(event, index) {
    event.preventDefault();
    setDragOverIndex(null);
    const file = event.dataTransfer?.files?.[0];
    if (file) {
      importIntoFrame(storyboardFrames[index].id, file);
      setDragIndex(null);
      return;
    }
    if (dragIndex != null) {
      reorderFrame(dragIndex, index);
    }
    setDragIndex(null);
  }

  async function submit(event) {
    event.preventDefault();
    if (!canSubmit) {
      return;
    }
    setSubmitting(true);
    const [width, height] = resolution.split("x").map((value) => Number(value));
    const trimmedSystem = systemMessage.trim();
    const advanced = {};
    // Only send the system prompt when edited; blank/default lets the worker use
    // its own _INTERLEAVE_SYSTEM_MESSAGE.
    if (trimmedSystem && trimmedSystem !== DEFAULT_INTERLEAVE_SYSTEM_MESSAGE) {
      advanced.systemMessage = trimmedSystem;
    }
    // Storyboard order IS the reference order the worker grounds on.
    const sourceAssetIds = storyboardFrames.map((frame) => frame.assetId).filter(Boolean);
    // Reference guidance only bites when the model is grounding on reference frames
    // (advanced.imageGuidanceScale → engine img_cfg_scale). Omit it otherwise so an
    // un-referenced run stays on the worker's plain-generation defaults.
    if (sourceAssetIds.length > 0) {
      advanced.imageGuidanceScale = referenceGuidance;
    }
    const job = await createInterleaveJob({
      prompt: prompt.trim(),
      model: model || undefined,
      maxImages: clampMaxImages(maxImages),
      width,
      height,
      sourceAssetIds,
      advanced,
    });
    setSubmitting(false);
    if (job) {
      // Stack the run in the studio instead of routing to the Queue, so its output
      // streams in below the prompt as it composes.
      rememberLocalGenerationJob?.("document", job);
    }
  }

  return (
    <ModelAvailabilityGate
      ready={modelReady}
      title="Document Studio needs an interleave-capable model"
      description="Interleaved text-image documents need a model like SenseNova-U1. Download one to get started."
      offers={modelOffers}
      downloadJobs={modelDownloadJobs}
      onDownload={createModelDownloadJob}
      onOpenModels={() => setActiveView("Models")}
      onOpenQueue={onOpenQueue}
      onCancelJob={onCancelJob}
    >
    <section className="page-frame document-studio">
      <WorkPanel
        eyebrow="Compose a document"
        hint="Describe the document you want — a guide, a poster, an ad page, an infographic, a comic, anything — then storyboard the shots. The model interleaves prose with the images it generates."
      >
      <form className="studio-form" onSubmit={submit}>
        {/* Quick-starts — broad, optional scaffolds. */}
        <div className="doc-quickstarts">
          <span className="doc-quickstarts-label">
            Start from an example{" "}
            <span className="doc-quickstarts-note">· optional — a starting scaffold, not a limit. Describe anything below.</span>
          </span>
          <div className="doc-quickstart-row">
            {QUICK_STARTS.map((entry) => {
              const Glyph = Icon[entry.icon] ?? Icon.Blank;
              const selected = quickStart === entry.id;
              const chipClass = [
                "doc-chip",
                entry.blank ? "doc-chip-blank" : "",
                selected ? "selected" : "",
              ]
                .filter(Boolean)
                .join(" ");
              return (
                <button className={chipClass} key={entry.id} onClick={() => applyQuickStart(entry)} type="button">
                  <Glyph size={13} />
                  {entry.label}
                </button>
              );
            })}
            <span className="doc-quickstart-more">…or anything you describe</span>
          </div>
        </div>

        {/* Prompt + brief tips. */}
        <div className="doc-compose-grid">
          <label className="field doc-prompt-field">
            <span>What should it cover?</span>
            <textarea
              onChange={(event) => setPrompt(event.target.value)}
              placeholder="Describe the document — the beats you want as images, their order, and one consistent style…"
              ref={promptRef}
              value={prompt}
            />
          </label>
          <aside className="doc-tips">
            <span className="doc-tips-title">
              <Icon.Info size={14} />
              Writing a good brief
            </span>
            <ol className="doc-tips-list">
              {BRIEF_TIPS.map((tip, index) => (
                <li key={index}>
                  <span className="doc-tips-num">{index + 1}.</span>
                  {tip}
                </li>
              ))}
            </ol>
          </aside>
        </div>

        {/* Starters — one-tap snippets inserted at the caret. */}
        <div className="doc-starters">
          <span className="doc-starters-label">Starters</span>
          {STARTERS.map((starter) => (
            <button className="doc-starter-chip" key={starter.label} onClick={() => insertStarter(starter.snippet)} type="button">
              <span className="doc-starter-plus">+</span> {starter.label}
            </button>
          ))}
        </div>

        {/* Storyboard — ordered, reorderable reference frames + reference-guidance slider. */}
        <div className="doc-storyboard">
          <div className="doc-storyboard-head">
            <span className="doc-storyboard-title">
              Storyboard <span className="doc-storyboard-note">· optional reference frames, drag to reorder</span>
            </span>
            <span className="doc-storyboard-count">
              {storyboardFrames.length} {storyboardFrames.length === 1 ? "frame" : "frames"}
            </span>
          </div>

          <div className="doc-storyboard-explainer">
            <Icon.Eye size={17} className="doc-storyboard-eye" />
            <div className="doc-storyboard-explainer-body">
              <span>
                <strong>The model looks at these frames — it doesn't copy their style.</strong> It reads them the way you would
                to describe a picture, then your prompt decides what to do with them.
              </span>
              <div className="doc-storyboard-examples">
                <span className="doc-storyboard-examples-lead">Your prompt might say:</span>
                {REFERENCE_PROMPT_EXAMPLES.map((example) => (
                  <span className="doc-storyboard-example" key={example}>
                    “{example}”
                  </span>
                ))}
              </div>
            </div>
          </div>

          <div className="doc-filmstrip">
            {storyboardFrames.map((frame, index) => {
              const frameAsset = frame.assetId ? assetById.get(frame.assetId) : null;
              const thumbClass = [
                "doc-frame-thumb",
                frameAsset ? "has-image" : "",
                dragOverIndex === index ? "drag-over" : "",
              ]
                .filter(Boolean)
                .join(" ");
              return (
                <div
                  className="doc-frame"
                  key={frame.id}
                  onDragOver={(event) => {
                    event.preventDefault();
                    setDragOverIndex(index);
                  }}
                  onDragLeave={() => setDragOverIndex((current) => (current === index ? null : current))}
                  onDrop={(event) => handleFrameDrop(event, index)}
                >
                  <button
                    aria-label={`Reference frame ${index + 1}${frameAsset ? " — change image" : " — pick an image"}`}
                    className={thumbClass}
                    draggable
                    onClick={() => setPickerFrameId(frame.id)}
                    onDragStart={() => setDragIndex(index)}
                    onDragEnd={() => {
                      setDragIndex(null);
                      setDragOverIndex(null);
                    }}
                    type="button"
                  >
                    <span className="doc-frame-badge">{index + 1}</span>
                    <span aria-hidden="true" className="doc-frame-handle">⠿</span>
                    {frameAsset ? (
                      <AssetThumbnail asset={frameAsset} className="doc-frame-image" />
                    ) : (
                      <span className="doc-frame-placeholder">
                        <Icon.Image size={18} />
                        Pick or drop
                      </span>
                    )}
                  </button>
                  <div className="doc-frame-footer">
                    <input
                      aria-label={`Caption for frame ${index + 1}`}
                      className="doc-frame-caption"
                      onChange={(event) => updateFrameCaption(frame.id, event.target.value)}
                      placeholder={`Beat ${index + 1}…`}
                      value={frame.caption}
                    />
                    <button
                      aria-label={`Remove frame ${index + 1}`}
                      className="doc-frame-remove"
                      onClick={() => removeFrame(frame.id)}
                      title="Remove frame"
                      type="button"
                    >
                      <Icon.Close size={13} />
                    </button>
                  </div>
                </div>
              );
            })}
            <button className="doc-frame-add" onClick={addFrame} type="button">
              <Icon.Plus size={20} />
              Add frame
            </button>
          </div>

          <div className="doc-storyboard-divider" />

          <div className="doc-refguidance">
            <div className="doc-refguidance-head">
              <span className="doc-refguidance-label">
                Reference guidance <span className="doc-refguidance-note">· how hard the model leans on the frames</span>
              </span>
              <span className="doc-refguidance-value">{referenceGuidance.toFixed(1)}</span>
            </div>
            <input
              aria-label="Reference guidance"
              className="doc-refguidance-slider"
              max={INTERLEAVE_IMAGE_GUIDANCE_MAX}
              min={INTERLEAVE_IMAGE_GUIDANCE_MIN}
              onChange={(event) => setReferenceGuidance(Number(event.target.value))}
              step={INTERLEAVE_IMAGE_GUIDANCE_STEP}
              type="range"
              value={referenceGuidance}
            />
            <div className="doc-refguidance-ends">
              <span>Follow the prompt</span>
              <span>Stay close to references</span>
            </div>
            <span className="doc-refguidance-hint">
              Sets the image guidance (CFG) for the references. Higher holds tighter to what the frames show; lower gives your
              prompt more room.
              {referencedFrameCount === 0 ? " Add a reference frame above for this to take effect." : ""}
            </span>
          </div>
        </div>

        {/* Settings strip — Model / Size / Max images (GPU lives in Advanced now). */}
        <div className="settings-bar">
          <div className="settings-bar-row">
            <label className="settings-field settings-field-model">
              Model
              <StudioUpdateBadge item={selectedModel} />
              <select onChange={(event) => setModel(event.target.value)} value={model}>
                {interleaveModels.map((item) => (
                  <option key={item.id} value={item.id}>
                    {updateOptionLabel(item)}
                  </option>
                ))}
              </select>
              <StudioUpdateNotice item={selectedModel} onUpdate={createModelDownloadJob} />
            </label>
            <label className="settings-field settings-field-aspect">
              Size
              <select onChange={(event) => setResolution(event.target.value)} value={resolution}>
                {INTERLEAVE_RESOLUTION_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    {formatResolutionLabel(option)}
                  </option>
                ))}
              </select>
            </label>
            <label className="settings-field settings-field-count">
              Max images
              <input
                max={MAX_IMAGES_LIMIT}
                min={1}
                onChange={(event) => setMaxImages(event.target.value)}
                type="number"
                value={maxImages}
              />
            </label>
          </div>
        </div>

        {/* Advanced — GPU + system prompt, collapsed by default. */}
        <AdvancedSection
          hint="GPU · system prompt"
          onToggle={() => setAdvancedOpen((value) => !value)}
          open={advancedOpen}
        >
          <div className="advanced-panel">
            {gpuOptions?.length ? (
              <label className="doc-advanced-gpu">
                GPU
                <select onChange={(event) => setRequestedGpu?.(event.target.value)} value={requestedGpu}>
                  {gpuOptions.map((option) => (
                    <option key={option} value={option}>
                      {option === "auto" ? "Auto" : option}
                    </option>
                  ))}
                </select>
              </label>
            ) : null}
            <label className="field document-system-prompt">
              <span>System prompt</span>
              <small>
                Steers the model's think / no-think composition. Prefilled with the default — edit to change behavior.
              </small>
              <textarea
                onChange={(event) => setSystemMessage(event.target.value)}
                rows={6}
                value={systemMessage}
              />
              {systemMessage !== DEFAULT_INTERLEAVE_SYSTEM_MESSAGE ? (
                <button
                  className="secondary-action"
                  onClick={() => setSystemMessage(DEFAULT_INTERLEAVE_SYSTEM_MESSAGE)}
                  type="button"
                >
                  Reset to default
                </button>
              ) : null}
            </label>
          </div>
        </AdvancedSection>

        <button className="primary-action" disabled={!canSubmit} type="submit">
          {submitting ? "Submitting…" : "Compose document"}
        </button>
      </form>
      </WorkPanel>

      <section className="studio-results">
        {localJobs.length ? (
          <div className="local-job-stack">
            {localJobs.map((job) => {
              const segments = job.status === "completed" ? documentSegments(job) : null;
              return (
                <article className="local-job-group" key={job.id}>
                  {segments ? (
                    <DocumentView assets={assets ?? []} projectId={activeProject?.id} segments={segments} />
                  ) : (
                    <WorkerProgressCard job={job} onCancel={onCancelJob} onOpenQueue={onOpenQueue} />
                  )}
                </article>
              );
            })}
          </div>
        ) : (
          <p className="empty-panel">Your generated document will appear here.</p>
        )}
      </section>

      {pickerFrame ? (
        <AssetPickerModal
          assets={sourceImageAssets}
          initialSelectedIds={pickerFrame.assetId ? [pickerFrame.assetId] : []}
          multiple={false}
          onCancel={() => setPickerFrameId(null)}
          onConfirm={(ids) => {
            updateFrameAsset(pickerFrame.id, ids[0] ?? "");
            setPickerFrameId(null);
          }}
          title="Reference frame"
        />
      ) : null}
    </section>
    </ModelAvailabilityGate>
  );
}
