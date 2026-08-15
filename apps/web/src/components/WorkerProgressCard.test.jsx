import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  WorkerProgressCard,
  deriveJobTitle,
  getJobTypeChip,
  useLiveJobElapsedSeconds,
} from "./WorkerProgressCard.jsx";
import { AppContext } from "../context/AppContext.js";
import { ScreenActiveContext } from "../context/ScreenActiveContext.js";
import { buildWorkersById } from "../workers.js";

const cudaWorker = {
  id: "worker-cuda-1",
  gpuId: "gpu-0",
  gpuName: "NVIDIA GeForce RTX 4090",
  capabilities: ["gpu", "image_generate"],
  utilization: { memoryUsedMb: 18000, memoryTotalMb: 24000, gpuLoadPercent: 67 },
};

const appleWorker = {
  id: "worker-mps-1",
  gpuId: "mps-0",
  gpuName: "Apple M2 Ultra",
  capabilities: ["gpu", "image_generate", "lora_train"],
  utilization: { memoryUsedMb: 30000, memoryTotalMb: 64000, gpuLoadPercent: 41 },
};

const cpuWorker = {
  id: "worker-cpu-1",
  gpuId: "cpu-0",
  gpuName: null,
  capabilities: ["cpu", "prompt_refine"],
  utilization: null,
};

function makeContext(workers) {
  return {
    visibleWorkers: workers,
    workersById: buildWorkersById(workers),
  };
}

function render(ui, contextValue) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => {
    root.render(<AppContext.Provider value={contextValue}>{ui}</AppContext.Provider>);
  });
  return {
    container,
    cleanup() {
      act(() => root.unmount());
      container.remove();
    },
  };
}

describe("deriveJobTitle", () => {
  it("prefers job.title when present", () => {
    expect(deriveJobTitle({ title: "Override", type: "image_generate" })).toBe("Override");
  });

  it("formats lora_train jobs with the LoRA name", () => {
    const job = { type: "lora_train", payload: { loraName: "kelsie-v3" } };
    expect(deriveJobTitle(job)).toBe("Training Run — kelsie-v3");
  });

  it("formats captioning jobs with the dataset name", () => {
    const job = { type: "training_caption", payload: { datasetName: "kelsie-set" } };
    expect(deriveJobTitle(job)).toBe("Dataset Captioning — kelsie-set");
  });

  it("formats image generation jobs and truncates long prompts", () => {
    const longPrompt = "a ".repeat(120);
    const job = { type: "image_generate", payload: { prompt: longPrompt } };
    const title = deriveJobTitle(job);
    expect(title.startsWith("Generate Image — ")).toBe(true);
    expect(title.endsWith("…")).toBe(true);
    expect(title.length).toBeLessThan(120);
  });

  it("formats character turnaround when characterId is set", () => {
    const job = {
      type: "image_generate",
      payload: { prompt: "anything", characterId: "char-1", characterName: "Aria" },
    };
    expect(deriveJobTitle(job)).toBe("Character Turnaround — Aria");
  });

  it("formats lora_import jobs with the filename when name absent", () => {
    const job = { type: "lora_import", payload: { filename: "kelsie.safetensors" } };
    expect(deriveJobTitle(job)).toBe("LoRA Import — kelsie.safetensors");
  });

  it("falls back to (unnamed) when no subject is present", () => {
    const job = { type: "lora_train", payload: {} };
    expect(deriveJobTitle(job)).toBe("Training Run — (unnamed LoRA)");
  });
});

describe("getJobTypeChip", () => {
  it("maps known types to display labels", () => {
    expect(getJobTypeChip("lora_train")).toBe("Training Run");
    expect(getJobTypeChip("training_caption")).toBe("Dataset Captioning");
    expect(getJobTypeChip("video_generate")).toBe("Generate Video");
    expect(getJobTypeChip("model_download")).toBe("Model Import");
  });

  it("falls back to a capitalized type for unknown values", () => {
    expect(getJobTypeChip("custom_job")).toBe("Custom job");
  });
});

describe("WorkerProgressCard layout", () => {
  let api;

  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-05-28T12:00:30Z"));
  });

  afterEach(() => {
    api?.cleanup();
    api = null;
    vi.useRealTimers();
  });

  it("renders the type chip, status badge, title, and id", () => {
    const job = {
      id: "job-abcdef123456",
      type: "image_generate",
      status: "running",
      stage: "denoising",
      progress: 0.4,
      attempts: 1,
      startedAt: "2026-05-28T12:00:00Z",
      workerId: cudaWorker.id,
      payload: { prompt: "a sunset over the mountains" },
    };
    api = render(<WorkerProgressCard job={job} />, makeContext([cudaWorker]));
    const card = api.container.querySelector(".worker-progress-card");
    expect(card.classList.contains("running")).toBe(true);
    expect(card.querySelector(".worker-progress-card__type").textContent).toBe("Generate Image");
    expect(card.querySelector(".status-badge").textContent).toBe("Running");
    expect(card.querySelector(".worker-progress-card__title").textContent).toBe(
      "Generate Image — a sunset over the mountains",
    );
    expect(card.querySelector(".worker-progress-card__id").getAttribute("title")).toBe("job-abcdef123456");
    expect(card.querySelector(".worker-progress-card__id").textContent).toBe("job-ab…3456");
  });

  it("renders live GPU meters for a running job assigned to a CUDA worker", () => {
    const job = {
      id: "job-1",
      type: "image_generate",
      status: "running",
      progress: 0.25,
      attempts: 1,
      startedAt: "2026-05-28T12:00:00Z",
      workerId: cudaWorker.id,
      payload: {},
    };
    api = render(<WorkerProgressCard job={job} />, makeContext([cudaWorker]));
    const arch = api.container.querySelector(".worker-progress-card__pill.arch");
    expect(arch.textContent).toBe("cuda");
    const meters = api.container.querySelectorAll(".worker-progress-card__meter-value");
    expect(meters).toHaveLength(2);
    expect(meters[0].textContent).toBe("75%"); // 18000/24000
    expect(meters[1].textContent).toBe("67%");
  });

  it("renders MPS arch for Apple Silicon workers", () => {
    const job = {
      id: "j",
      type: "lora_train",
      status: "running",
      progress: 0.1,
      attempts: 1,
      startedAt: "2026-05-28T12:00:00Z",
      workerId: appleWorker.id,
      payload: { loraName: "x" },
    };
    api = render(<WorkerProgressCard job={job} />, makeContext([appleWorker]));
    expect(api.container.querySelector(".worker-progress-card__pill.arch").textContent).toBe("mps");
  });

  it("uses peak meters and hides cancel for completed jobs", () => {
    const job = {
      id: "j",
      type: "image_generate",
      status: "completed",
      progress: 1,
      attempts: 1,
      startedAt: "2026-05-28T12:00:00Z",
      workerId: cudaWorker.id,
      payload: {},
      peakGpuMemoryPct: 88,
      peakGpuLoadPct: 95,
    };
    const onCancel = vi.fn();
    const onRetry = vi.fn();
    const onDuplicate = vi.fn();
    api = render(
      <WorkerProgressCard job={job} onCancel={onCancel} onRetry={onRetry} onDuplicate={onDuplicate} />,
      makeContext([cudaWorker]),
    );
    const metersHost = api.container.querySelector(".worker-progress-card__meters");
    expect(metersHost.getAttribute("data-meter-source")).toBe("peak");
    const values = api.container.querySelectorAll(".worker-progress-card__meter-value");
    expect(values[0].textContent).toBe("88%");
    expect(values[1].textContent).toBe("95%");
    const buttons = api.container.querySelectorAll(".worker-progress-card__actions button");
    const labels = Array.from(buttons).map((b) => b.textContent);
    expect(labels).not.toContain("Cancel");
    expect(labels).toContain("Retry");
    expect(labels).toContain("Duplicate");
  });

  it("shows only Cancel for queued jobs (no Retry/Duplicate)", () => {
    const job = {
      id: "j",
      type: "image_generate",
      status: "queued",
      progress: 0,
      attempts: 1,
      payload: { prompt: "" },
    };
    const handlers = { onCancel: vi.fn(), onRetry: vi.fn(), onDuplicate: vi.fn() };
    api = render(<WorkerProgressCard job={job} {...handlers} />, makeContext([]));
    const buttons = api.container.querySelectorAll(".worker-progress-card__actions button");
    const labels = Array.from(buttons).map((b) => b.textContent);
    expect(labels).toEqual(["Cancel"]);
  });

  it("renders a pending_caption job as an in-flight, cancelable job (sc-9120)", () => {
    // A pending_caption Ideogram job (awaiting the async prompt rewrite) is non-terminal: it gets a
    // readable badge and can be canceled before a worker starts, like a queued job.
    const job = {
      id: "j",
      type: "image_generate",
      status: "pending_caption",
      progress: 0,
      attempts: 1,
      payload: { prompt: "a red fox" },
    };
    const handlers = { onCancel: vi.fn(), onRetry: vi.fn(), onDuplicate: vi.fn() };
    api = render(<WorkerProgressCard job={job} {...handlers} />, makeContext([]));
    expect(api.container.querySelector(".status-badge").textContent).toBe("Preparing prompt");
    const labels = Array.from(
      api.container.querySelectorAll(".worker-progress-card__actions button"),
    ).map((b) => b.textContent);
    expect(labels).toEqual(["Cancel"]);
  });

  it("uses resume-first actions for failed model downloads", () => {
    const job = {
      id: "j",
      type: "model_download",
      status: "failed",
      progress: 0.7,
      attempts: 1,
      payload: { modelId: "realvisxl" },
      error: "download ended early",
    };
    const onRetry = vi.fn();
    const onFreshRetry = vi.fn();
    api = render(
      <WorkerProgressCard job={job} onRetry={onRetry} onFreshRetry={onFreshRetry} onDuplicate={vi.fn()} />,
      makeContext([]),
    );
    let buttons = Array.from(api.container.querySelectorAll(".worker-progress-card__actions button"));
    expect(buttons.map((button) => button.textContent)).toEqual(["Resume Download"]);
    act(() => {
      buttons[0].dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    });
    expect(onRetry).toHaveBeenCalledWith(job, { payloadChanges: { downloadAction: "resume" } });
    expect(onFreshRetry).not.toHaveBeenCalled();

    api.cleanup();
    const resumedJob = {
      ...job,
      id: "j2",
      attempts: 2,
      payload: { ...job.payload, downloadAction: "resume" },
    };
    api = render(
      <WorkerProgressCard job={resumedJob} onRetry={onRetry} onFreshRetry={onFreshRetry} />,
      makeContext([]),
    );
    buttons = Array.from(api.container.querySelectorAll(".worker-progress-card__actions button"));
    expect(buttons.map((button) => button.textContent)).toEqual(["Resume Download", "Retry Download"]);
    act(() => {
      buttons[1].dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    });
    expect(onFreshRetry).toHaveBeenCalledWith(resumedJob, { payloadChanges: { downloadAction: "fresh" } });
  });

  it("shows the dismiss × on a terminal job and calls onClear (issue #1556)", () => {
    const job = {
      id: "job-done",
      type: "image_generate",
      status: "completed",
      progress: 1,
      attempts: 1,
      payload: { prompt: "a fox" },
    };
    const onClear = vi.fn();
    api = render(<WorkerProgressCard job={job} onClear={onClear} />, makeContext([]));
    const dismiss = api.container.querySelector(".worker-progress-card__dismiss");
    expect(dismiss).not.toBeNull();
    act(() => {
      dismiss.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    });
    expect(onClear).toHaveBeenCalledWith(job);
  });

  it("does not show the dismiss × on an active job (would resurrect on SSE)", () => {
    const job = {
      id: "job-running",
      type: "image_generate",
      status: "running",
      progress: 0.4,
      attempts: 1,
      payload: { prompt: "a fox" },
    };
    api = render(<WorkerProgressCard job={job} onClear={vi.fn()} />, makeContext([]));
    expect(api.container.querySelector(".worker-progress-card__dismiss")).toBeNull();
  });

  it("omits the dismiss × when no onClear handler is provided", () => {
    const job = {
      id: "job-done",
      type: "image_generate",
      status: "completed",
      progress: 1,
      attempts: 1,
      payload: { prompt: "a fox" },
    };
    api = render(<WorkerProgressCard job={job} />, makeContext([]));
    expect(api.container.querySelector(".worker-progress-card__dismiss")).toBeNull();
  });

  it("hides View in Queue when hideOpenQueue is true", () => {
    const job = { id: "j", type: "image_generate", status: "running", progress: 0, attempts: 1, payload: {} };
    const onOpenQueue = vi.fn();
    api = render(
      <WorkerProgressCard job={job} onOpenQueue={onOpenQueue} hideOpenQueue />,
      makeContext([]),
    );
    const labels = Array.from(api.container.querySelectorAll(".worker-progress-card__actions button")).map(
      (b) => b.textContent,
    );
    expect(labels).not.toContain("View in Queue");
  });

  it("shows an indeterminate progress bar when running with no progress value", () => {
    const job = {
      id: "j",
      type: "image_generate",
      status: "running",
      progress: 0,
      attempts: 1,
      startedAt: "2026-05-28T12:00:00Z",
      payload: {},
    };
    api = render(<WorkerProgressCard job={job} />, makeContext([]));
    expect(api.container.querySelector(".worker-progress-card__progress.indeterminate")).not.toBeNull();
  });

  it("hides GPU pills and meters for CPU-only workers", () => {
    const job = {
      id: "j",
      type: "prompt_refine",
      status: "running",
      progress: 0,
      attempts: 1,
      startedAt: "2026-05-28T12:00:00Z",
      workerId: cpuWorker.id,
      payload: { prompt: "refine this" },
    };
    api = render(<WorkerProgressCard job={job} />, makeContext([cpuWorker]));
    const pills = api.container.querySelectorAll(".worker-progress-card__pill");
    expect(Array.from(pills).map((p) => p.textContent)).toEqual(["CPU"]);
    expect(api.container.querySelector(".worker-progress-card__meters")).toBeNull();
  });

  it("invokes onCancel when the Cancel button is clicked", () => {
    const job = { id: "j", type: "image_generate", status: "running", progress: 0.3, attempts: 1, payload: {} };
    const onCancel = vi.fn();
    api = render(<WorkerProgressCard job={job} onCancel={onCancel} />, makeContext([]));
    const button = api.container.querySelector(".worker-progress-card__actions button");
    act(() => {
      button.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    });
    expect(onCancel).toHaveBeenCalledTimes(1);
    expect(onCancel.mock.calls[0][0]).toBe(job);
  });
});

describe("WorkerProgressCard thumbnails", () => {
  let api;
  afterEach(() => {
    api?.cleanup();
    api = null;
  });

  const completedJob = {
    id: "job-1",
    type: "image_generate",
    status: "completed",
    progress: 1,
    attempts: 1,
    payload: {},
  };
  const runningJob = {
    id: "job-2",
    type: "image_generate",
    status: "running",
    progress: 0.4,
    attempts: 1,
    startedAt: "2026-05-28T12:00:00Z",
    payload: { count: 4 },
  };
  const videoJob = {
    id: "job-v",
    type: "video_generate",
    status: "completed",
    progress: 1,
    attempts: 1,
    payload: {},
  };

  const imageAssets = [
    { id: "a-1", type: "image", url: "/api/v1/files/a-1.png" },
    { id: "a-2", type: "image", url: "/api/v1/files/a-2.png" },
  ];
  const interimAsset = { id: "interim-1", type: "image", url: "/api/v1/files/i-1.jpg", __interim: true };
  const videoAsset = { id: "v-1", type: "video", url: "/api/v1/files/v-1.mp4" };

  it("hides the thumbnails region by default (variant=hidden)", () => {
    api = render(<WorkerProgressCard job={completedJob} thumbnailAssets={imageAssets} />, makeContext([]));
    expect(api.container.querySelector(".worker-progress-card__thumbnails")).toBeNull();
  });

  it("renders image-grid variant with one cell per final asset", () => {
    api = render(
      <WorkerProgressCard
        job={completedJob}
        thumbnailsVariant="image-grid"
        thumbnailAssets={imageAssets}
      />,
      makeContext([]),
    );
    const grid = api.container.querySelector(".worker-progress-card__thumbnails--image-grid");
    expect(grid).not.toBeNull();
    expect(grid.querySelectorAll(".worker-progress-card__thumb-cell")).toHaveLength(2);
  });

  it("merges interim and final assets, deduping by id", () => {
    const dup = { id: "a-1", type: "image", url: "/x", __interim: true };
    api = render(
      <WorkerProgressCard
        job={runningJob}
        thumbnailsVariant="image-grid"
        thumbnailAssets={imageAssets}
        interimThumbnailAssets={[dup, interimAsset]}
      />,
      makeContext([]),
    );
    const cells = api.container.querySelectorAll(".worker-progress-card__thumb-cell:not(.skeleton)");
    expect(cells).toHaveLength(3); // a-1, a-2, interim-1 (dup of a-1 dropped)
  });

  it("renders skeleton cells up to expectedThumbnailCount while running", () => {
    api = render(
      <WorkerProgressCard
        job={runningJob}
        thumbnailsVariant="image-grid"
        thumbnailAssets={[imageAssets[0]]}
        expectedThumbnailCount={4}
      />,
      makeContext([]),
    );
    const skeletons = api.container.querySelectorAll(".worker-progress-card__thumb-cell.skeleton");
    expect(skeletons).toHaveLength(3);
  });

  it("renders grouped thumbnail cycles with section labels", () => {
    api = render(
      <WorkerProgressCard
        job={runningJob}
        thumbnailsVariant="image-grid"
        thumbnailGroups={[
          { id: "step-500", label: "Sample #2 - Step 500", assets: [imageAssets[0]] },
          { id: "step-250", label: "Sample #1 - Step 250", assets: [imageAssets[1]] },
        ]}
        expectedThumbnailCount={4}
      />,
      makeContext([]),
    );
    const groups = api.container.querySelectorAll(".worker-progress-card__thumbnail-group");
    expect(groups).toHaveLength(2);
    expect(groups[0].textContent).toContain("Sample #2 - Step 500");
    expect(groups[1].textContent).toContain("Sample #1 - Step 250");
    expect(api.container.querySelectorAll(".worker-progress-card__thumb-cell.skeleton")).toHaveLength(0);
  });

  it("does not render skeletons after the job completes", () => {
    api = render(
      <WorkerProgressCard
        job={completedJob}
        thumbnailsVariant="image-grid"
        thumbnailAssets={[imageAssets[0]]}
        expectedThumbnailCount={4}
      />,
      makeContext([]),
    );
    expect(api.container.querySelectorAll(".worker-progress-card__thumb-cell.skeleton")).toHaveLength(0);
  });

  it("renders small-row variant with compact cells", () => {
    api = render(
      <WorkerProgressCard
        job={completedJob}
        thumbnailsVariant="small-row"
        thumbnailAssets={imageAssets}
      />,
      makeContext([]),
    );
    const grid = api.container.querySelector(".worker-progress-card__thumbnails--small-row");
    expect(grid).not.toBeNull();
    const cells = grid.querySelectorAll(".worker-progress-card__thumb-cell.small");
    expect(cells).toHaveLength(2);
  });

  it("invokes onThumbnailClick when a cell is activated", () => {
    const onThumbnailClick = vi.fn();
    api = render(
      <WorkerProgressCard
        job={completedJob}
        thumbnailsVariant="image-grid"
        thumbnailAssets={imageAssets}
        onThumbnailClick={onThumbnailClick}
      />,
      makeContext([]),
    );
    const cell = api.container.querySelector(".worker-progress-card__thumb-cell");
    act(() => {
      cell.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    });
    expect(onThumbnailClick).toHaveBeenCalledTimes(1);
    expect(onThumbnailClick.mock.calls[0][0].id).toBe("a-1");
  });

  it("renders video-player variant with a video element when an asset is present", () => {
    api = render(
      <WorkerProgressCard
        job={videoJob}
        thumbnailsVariant="video-player"
        thumbnailAssets={[videoAsset]}
      />,
      makeContext([]),
    );
    expect(api.container.querySelector(".worker-progress-card__thumbnails--video-player")).not.toBeNull();
    expect(api.container.querySelector("video")).not.toBeNull();
  });

  it("suppresses the native context menu on the video-player thumbnail (sc-8731)", () => {
    api = render(
      <WorkerProgressCard
        job={videoJob}
        thumbnailsVariant="video-player"
        thumbnailAssets={[videoAsset]}
      />,
      makeContext([]),
    );
    const wrapper = api.container.querySelector(".worker-progress-card__thumbnails--video-player");
    expect(wrapper).not.toBeNull();
    const event = new window.MouseEvent("contextmenu", { bubbles: true, cancelable: true });
    act(() => {
      wrapper.dispatchEvent(event);
    });
    expect(event.defaultPrevented).toBe(true);
  });

  it("renders a video placeholder when no asset is available yet", () => {
    api = render(
      <WorkerProgressCard
        job={{ ...videoJob, status: "running" }}
        thumbnailsVariant="video-player"
        thumbnailAssets={[]}
      />,
      makeContext([]),
    );
    expect(api.container.querySelector(".worker-progress-card__video-placeholder")).not.toBeNull();
  });

  it("renders nothing for image-grid when there are no assets and the job is terminal", () => {
    api = render(
      <WorkerProgressCard
        job={completedJob}
        thumbnailsVariant="image-grid"
        thumbnailAssets={[]}
      />,
      makeContext([]),
    );
    expect(api.container.querySelector(".worker-progress-card__thumbnails")).toBeNull();
  });

  it("dims discarded (trashed) thumbnails without hiding them", () => {
    const discarded = { id: "a-3", type: "image", url: "/api/v1/files/a-3.png", status: { trashed: true } };
    api = render(
      <WorkerProgressCard
        job={completedJob}
        thumbnailsVariant="small-row"
        thumbnailAssets={[imageAssets[0], discarded]}
        onThumbnailClick={() => {}}
      />,
      makeContext([]),
    );
    const cells = api.container.querySelectorAll(".worker-progress-card__thumb-cell");
    expect(cells).toHaveLength(2);
    const dimmed = api.container.querySelectorAll(".worker-progress-card__thumb-cell.discarded");
    expect(dimmed).toHaveLength(1);
    expect(dimmed[0].getAttribute("aria-label")).toContain("(discarded)");
  });

  it("swaps a broken (purged) thumbnail for the deleted-asset placeholder", () => {
    api = render(
      <WorkerProgressCard
        job={completedJob}
        thumbnailsVariant="small-row"
        thumbnailAssets={[imageAssets[0]]}
      />,
      makeContext([]),
    );
    const img = api.container.querySelector("img.worker-progress-card__thumb-media");
    expect(img).not.toBeNull();
    expect(api.container.querySelector(".asset-thumb-missing")).toBeNull();
    act(() => {
      img.dispatchEvent(new window.Event("error", { bubbles: false }));
    });
    expect(api.container.querySelector("img.worker-progress-card__thumb-media")).toBeNull();
    const placeholder = api.container.querySelector(".asset-thumb-missing");
    expect(placeholder).not.toBeNull();
    expect(placeholder.getAttribute("aria-label")).toBe("Deleted asset");
  });
});

// Audio-player variant (SceneWorks Audio Studio, epic 13400 A5 / sc-13405): a
// completed audio job renders a real transport in the shared results zone — an
// actual <audio> element driven by a custom play/pause control and a scrubber,
// with the total duration derived from the asset's audio file block.
describe("WorkerProgressCard audio-player variant (epic 13400 A5)", () => {
  let api;
  let playSpy;
  let pauseSpy;

  beforeEach(() => {
    // jsdom doesn't implement media playback; stub so the transport's play/pause
    // are observable and don't log "Not implemented".
    playSpy = vi
      .spyOn(window.HTMLMediaElement.prototype, "play")
      .mockImplementation(() => Promise.resolve());
    pauseSpy = vi.spyOn(window.HTMLMediaElement.prototype, "pause").mockImplementation(() => {});
  });

  afterEach(() => {
    api?.cleanup();
    api = null;
    vi.restoreAllMocks();
  });

  const audioJob = {
    id: "job-audio",
    type: "audio_generate",
    status: "completed",
    progress: 1,
    attempts: 1,
    payload: {},
  };

  // Completed audio result asset (origin audio_studio) with the audio file block:
  // duration (seconds) + sampleRate + channels. 65s → the read-out reads "1:05".
  const audioAsset = {
    id: "au-1",
    type: "audio",
    url: "/api/v1/files/au-1.wav",
    origin: "audio_studio",
    file: { path: "assets/au-1.wav", mimeType: "audio/wav", duration: 65, sampleRate: 24000, channels: 1 },
  };

  it("renders the transport (real <audio> + play control + scrubber) with the derived duration", () => {
    api = render(
      <WorkerProgressCard job={audioJob} thumbnailsVariant="audio-player" thumbnailAssets={[audioAsset]} />,
      makeContext([]),
    );
    expect(api.container.querySelector(".worker-progress-card__thumbnails--audio-player")).not.toBeNull();
    // A real <audio> element drives play/scrub.
    expect(api.container.querySelector("audio")).not.toBeNull();
    // Play/pause transport control.
    const play = api.container.querySelector(".worker-progress-card__audio-play");
    expect(play).not.toBeNull();
    expect(play.getAttribute("aria-label")).toBe("Play");
    // Scrubber (seek), its max seeded from the declared duration.
    const scrubber = api.container.querySelector("input[type='range'].worker-progress-card__audio-scrubber");
    expect(scrubber).not.toBeNull();
    expect(scrubber.getAttribute("max")).toBe("65");
    // Duration is shown/derived from the asset file block (65s → 1:05).
    expect(api.container.querySelector(".worker-progress-card__audio-time-total").textContent).toBe("1:05");
  });

  it("toggles play/pause when the transport control is activated", () => {
    api = render(
      <WorkerProgressCard job={audioJob} thumbnailsVariant="audio-player" thumbnailAssets={[audioAsset]} />,
      makeContext([]),
    );
    const play = api.container.querySelector(".worker-progress-card__audio-play");
    expect(play.getAttribute("aria-pressed")).toBe("false");
    act(() => {
      play.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    });
    expect(playSpy).toHaveBeenCalledTimes(1);
    expect(play.getAttribute("aria-label")).toBe("Pause");
    expect(play.getAttribute("aria-pressed")).toBe("true");
    act(() => {
      play.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    });
    expect(pauseSpy).toHaveBeenCalledTimes(1);
    expect(play.getAttribute("aria-label")).toBe("Play");
  });

  it("seeks the audio element and updates the read-out when the scrubber changes", () => {
    api = render(
      <WorkerProgressCard job={audioJob} thumbnailsVariant="audio-player" thumbnailAssets={[audioAsset]} />,
      makeContext([]),
    );
    const audioEl = api.container.querySelector("audio");
    // jsdom's currentTime setter is a no-op, so spy on it to prove the seek is
    // actually written through to the media element.
    const setCurrentTime = vi.fn();
    Object.defineProperty(audioEl, "currentTime", { configurable: true, get: () => 0, set: setCurrentTime });
    const scrubber = api.container.querySelector(".worker-progress-card__audio-scrubber");
    // Use the native value setter so React's controlled-input value tracker sees a
    // real change and fires onChange (the repo's slider-test convention).
    const valueSetter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(scrubber), "value").set;
    act(() => {
      valueSetter.call(scrubber, "30");
      scrubber.dispatchEvent(new window.Event("input", { bubbles: true }));
    });
    expect(setCurrentTime).toHaveBeenCalledWith(30);
    expect(api.container.querySelector(".worker-progress-card__audio-time-current").textContent).toBe("0:30");
  });

  it("shows a placeholder while the clip is still rendering (no asset yet)", () => {
    api = render(
      <WorkerProgressCard
        job={{ ...audioJob, status: "running" }}
        thumbnailsVariant="audio-player"
        thumbnailAssets={[]}
      />,
      makeContext([]),
    );
    expect(api.container.querySelector(".worker-progress-card__audio-placeholder")).not.toBeNull();
    expect(api.container.querySelector("audio")).toBeNull();
  });

  it("suppresses the native context menu on the audio-player transport (sc-8731)", () => {
    api = render(
      <WorkerProgressCard job={audioJob} thumbnailsVariant="audio-player" thumbnailAssets={[audioAsset]} />,
      makeContext([]),
    );
    const wrapper = api.container.querySelector(".worker-progress-card__thumbnails--audio-player");
    const event = new window.MouseEvent("contextmenu", { bubbles: true, cancelable: true });
    act(() => {
      wrapper.dispatchEvent(event);
    });
    expect(event.defaultPrevented).toBe(true);
  });

  it("invokes onThumbnailClick from the transport's Open control", () => {
    const onThumbnailClick = vi.fn();
    api = render(
      <WorkerProgressCard
        job={audioJob}
        thumbnailsVariant="audio-player"
        thumbnailAssets={[audioAsset]}
        onThumbnailClick={onThumbnailClick}
      />,
      makeContext([]),
    );
    const open = api.container.querySelector(".worker-progress-card__audio-open");
    expect(open).not.toBeNull();
    act(() => {
      open.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    });
    expect(onThumbnailClick).toHaveBeenCalledTimes(1);
    expect(onThumbnailClick.mock.calls[0][0].id).toBe("au-1");
  });
});

// sc-11961 (S2): the live elapsed-seconds timer is the one piece of WorkerProgressCard
// that runs continuously (a 1s setInterval) while an in-flight job's card is mounted.
// Under keep-alive a studio stays mounted while backgrounded, so without gating each
// visited studio's in-flight card would keep ticking — several concurrent timers doing
// hidden work. These tests assert the interval only runs when the card's screen is the
// foreground view (ScreenActiveContext=true) and that no timer is created for hidden
// (backgrounded) kept-alive screens.
describe("useLiveJobElapsedSeconds keep-alive gating (sc-11961)", () => {
  const runningJob = {
    id: "job-elapsed",
    type: "image_generate",
    status: "running",
    startedAt: "2026-05-28T12:00:00Z",
    elapsedSeconds: 0,
  };

  // Minimal probe that surfaces the hook's return value into the DOM so we can read it.
  function ElapsedProbe({ job }) {
    const seconds = useLiveJobElapsedSeconds(job);
    return <span data-testid="elapsed">{seconds}</span>;
  }

  let roots;
  let priorActEnv;

  beforeEach(() => {
    priorActEnv = global.IS_REACT_ACT_ENVIRONMENT;
    global.IS_REACT_ACT_ENVIRONMENT = true;
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-05-28T12:00:05Z"));
    roots = [];
  });

  afterEach(() => {
    act(() => roots.forEach(({ root }) => root.unmount()));
    roots.forEach(({ container }) => container.remove());
    vi.useRealTimers();
    vi.restoreAllMocks();
    global.IS_REACT_ACT_ENVIRONMENT = priorActEnv;
  });

  function mount(active) {
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);
    act(() => {
      root.render(
        <ScreenActiveContext.Provider value={active}>
          <ElapsedProbe job={runningJob} />
        </ScreenActiveContext.Provider>,
      );
    });
    const entry = { container, root };
    roots.push(entry);
    return entry;
  }

  function elapsed(entry) {
    return Number(entry.container.querySelector("[data-testid='elapsed']").textContent);
  }

  it("ticks the elapsed timer only for the ACTIVE (foreground) screen", () => {
    const activeEntry = mount(true);
    const hiddenEntry = mount(false);

    // Both start at 5s elapsed (system time is 5s past startedAt).
    expect(elapsed(activeEntry)).toBe(5);
    expect(elapsed(hiddenEntry)).toBe(5);

    act(() => {
      vi.advanceTimersByTime(3000);
    });

    // Active screen's interval advanced the clock; hidden screen stayed frozen —
    // no continuous work ran for the backgrounded card.
    expect(elapsed(activeEntry)).toBe(8);
    expect(elapsed(hiddenEntry)).toBe(5);
  });

  it("creates NO interval for a hidden kept-alive screen, and no duplicates across many mounted screens", () => {
    const setIntervalSpy = vi.spyOn(window, "setInterval");

    // One active + three hidden (as if four studios were all visited and kept alive).
    mount(true);
    mount(false);
    mount(false);
    mount(false);

    // Exactly one interval exists — only the foreground card runs its loop. The three
    // hidden kept-alive cards create none, so mounting many screens does not multiply
    // the background work.
    expect(setIntervalSpy).toHaveBeenCalledTimes(1);
  });

  it("starts an interval when a hidden screen becomes active again (resume on re-show)", () => {
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);
    roots.push({ container, root });

    act(() => {
      root.render(
        <ScreenActiveContext.Provider value={false}>
          <ElapsedProbe job={runningJob} />
        </ScreenActiveContext.Provider>,
      );
    });
    // Hidden: frozen at 5s even after time passes.
    act(() => vi.advanceTimersByTime(2000));
    expect(Number(container.querySelector("[data-testid='elapsed']").textContent)).toBe(5);

    // Re-show: the effect re-runs, snaps to the current time (now 7s past start), and
    // resumes ticking.
    act(() => {
      root.render(
        <ScreenActiveContext.Provider value={true}>
          <ElapsedProbe job={runningJob} />
        </ScreenActiveContext.Provider>,
      );
    });
    expect(Number(container.querySelector("[data-testid='elapsed']").textContent)).toBe(7);
    act(() => vi.advanceTimersByTime(1000));
    expect(Number(container.querySelector("[data-testid='elapsed']").textContent)).toBe(8);
  });
});

// Live denoise preview (epic 16624, sc-16905): the card derives an interim thumbnail from the
// worker-streamed result.previewFrame slot — no screen threads a prop for it.
describe("live denoise preview", () => {
  let api;

  afterEach(() => {
    api?.cleanup();
    api = undefined;
  });

  const DATA_URL = "data:image/jpeg;base64,QUJD";
  const previewJob = (status) => ({
    id: "job-preview",
    type: "image_generate",
    status,
    progress: 0.4,
    attempts: 1,
    startedAt: "2026-05-28T12:00:00Z",
    payload: {},
    result: { previewFrame: { imageIndex: 0, current: 8, total: 20, dataUrl: DATA_URL } },
  });

  it("shows the streamed frame as an interim cell while the job runs", () => {
    api = render(
      <WorkerProgressCard job={previewJob("running")} thumbnailsVariant="image-grid" />,
      makeContext([]),
    );
    const cell = api.container.querySelector(".worker-progress-card__thumb-cell.interim");
    expect(cell).not.toBeNull();
    const img = cell.querySelector("img");
    expect(img).not.toBeNull();
    // The data URL must reach the <img> byte-identical: no API prefix, no media
    // ticket, no thumbnail param (any of those would corrupt the base64 body).
    expect(img.getAttribute("src")).toBe(DATA_URL);
  });

  it("never shows a frame on a terminal job, even when one survived in the record", () => {
    api = render(
      <WorkerProgressCard job={previewJob("canceled")} thumbnailsVariant="image-grid" />,
      makeContext([]),
    );
    expect(api.container.querySelector(".worker-progress-card__thumb-cell.interim")).toBeNull();
  });

  it("an explicit interimThumbnailAssets prop keeps precedence over the derived frame", () => {
    const explicit = [{ id: "explicit-1", type: "image", url: "/api/v1/files/e-1.png", __interim: true }];
    api = render(
      <WorkerProgressCard
        job={previewJob("running")}
        thumbnailsVariant="image-grid"
        interimThumbnailAssets={explicit}
      />,
      makeContext([]),
    );
    const cells = api.container.querySelectorAll(".worker-progress-card__thumb-cell.interim");
    expect(cells).toHaveLength(1);
    expect(cells[0].querySelector("img")?.getAttribute("src") ?? "").not.toBe(DATA_URL);
  });

  it("final assets render alongside the live frame without colliding", () => {
    const job = previewJob("running");
    api = render(
      <WorkerProgressCard
        job={job}
        thumbnailsVariant="image-grid"
        thumbnailAssets={[{ id: "a-1", type: "image", url: "/api/v1/files/a-1.png" }]}
      />,
      makeContext([]),
    );
    const cells = api.container.querySelectorAll(".worker-progress-card__thumb-cell:not(.skeleton)");
    expect(cells).toHaveLength(2);
    expect(api.container.querySelectorAll(".worker-progress-card__thumb-cell.interim")).toHaveLength(1);
  });

  it("renders nothing extra for jobs without a streamed frame", () => {
    const job = { ...previewJob("running"), result: {} };
    api = render(
      <WorkerProgressCard job={job} thumbnailsVariant="image-grid" />,
      makeContext([]),
    );
    expect(api.container.querySelector(".worker-progress-card__thumb-cell.interim")).toBeNull();
  });
});

// Live-preview DISCOVERABILITY (sc-16965, epic 16948). Three situations used to render identically:
// the model cannot preview; it can and the first frame has not arrived; a frame is showing. The
// card now reads `preview.byBackend` off the catalog entry — a capability, never an id list — keyed
// by the backend of the worker that claimed the job, and exposes the resolved state as
// `data-preview-state`.
describe("live preview support states", () => {
  let api;

  afterEach(() => {
    api?.cleanup();
    api = undefined;
  });

  const DATA_URL = "data:image/jpeg;base64,QUJD";
  // The candle GPU worker advertises the "candle" lane marker (gpu.rs with_candle_capabilities).
  const candleWorker = {
    id: "worker-candle-1",
    gpuId: "gpu-0",
    gpuName: "NVIDIA RTX PRO 6000",
    capabilities: ["gpu", "image_generate", "candle"],
    utilization: null,
  };
  const mlxWorker = {
    id: "worker-mlx-1",
    gpuId: "mlx",
    gpuName: "Apple Silicon (MLX)",
    capabilities: ["gpu", "image_generate"],
    utilization: null,
  };
  // Same model id, two answers — the reason the flag can never be a single shipped boolean.
  const models = [
    { id: "krea_2_turbo", type: "image", preview: { byBackend: { candle: true, mlx: true } } },
    { id: "sd3_5_medium", type: "image", preview: { byBackend: { candle: false } } },
    {
      id: "split_preview_fixture",
      type: "image",
      preview: { byBackend: { candle: false, mlx: true } },
    },
    { id: "some_external_model", type: "image" },
  ];

  const contextWith = (workers) => ({ ...makeContext(workers), models });

  const job = (overrides) => ({
    id: "job-preview-support",
    type: "image_generate",
    status: "running",
    stage: "loading_model",
    progress: 0.1,
    attempts: 1,
    startedAt: "2026-05-28T12:00:00Z",
    workerId: "worker-candle-1",
    payload: { model: "krea_2_turbo" },
    result: {},
    ...overrides,
  });

  it("state 1 — a model that does NOT support preview renders exactly as before", () => {
    api = render(
      <WorkerProgressCard
        job={job({ payload: { model: "sd3_5_medium" } })}
        thumbnailsVariant="image-grid"
      />,
      contextWith([candleWorker]),
    );
    const card = api.container.querySelector(".worker-progress-card");
    expect(card.getAttribute("data-preview-state")).toBe("unsupported");
    // No placeholder, no explanatory chrome: a permanent "no live preview" label would be noise on
    // the majority of renders.
    expect(api.container.querySelector("[data-testid='live-preview-placeholder']")).toBeNull();
    expect(api.container.querySelector(".worker-progress-card__thumb-cell.interim")).toBeNull();
  });

  it("state 2 — supports preview but no frame yet shows the placeholder", () => {
    api = render(
      <WorkerProgressCard job={job()} thumbnailsVariant="image-grid" />,
      contextWith([candleWorker]),
    );
    const card = api.container.querySelector(".worker-progress-card");
    expect(card.getAttribute("data-preview-state")).toBe("pending");
    const placeholder = api.container.querySelector("[data-testid='live-preview-placeholder']");
    expect(placeholder).not.toBeNull();
    // It occupies the same interim grid slot the real frame will take, so the cell fills in rather
    // than appearing from nowhere.
    expect(placeholder.classList.contains("interim")).toBe(true);
    expect(placeholder.classList.contains("worker-progress-card__thumb-cell")).toBe(true);
    // No <img>: there is no media yet, so AssetThumbnail is never handed a urlless asset.
    expect(placeholder.querySelector("img")).toBeNull();
  });

  it("state 3 — a streamed frame supersedes the placeholder in the same slot", () => {
    api = render(
      <WorkerProgressCard
        job={job({
          stage: "generating",
          result: { previewFrame: { imageIndex: 0, current: 3, total: 8, dataUrl: DATA_URL } },
        })}
        thumbnailsVariant="image-grid"
      />,
      contextWith([candleWorker]),
    );
    const card = api.container.querySelector(".worker-progress-card");
    expect(card.getAttribute("data-preview-state")).toBe("live");
    expect(api.container.querySelector("[data-testid='live-preview-placeholder']")).toBeNull();
    const cells = api.container.querySelectorAll(".worker-progress-card__thumb-cell.interim");
    expect(cells).toHaveLength(1);
    expect(cells[0].querySelector("img")?.getAttribute("src")).toBe(DATA_URL);
  });

  it("the answer is ENGINE-KEYED: the same model resolves per worker backend", () => {
    // This synthetic split route answers differently per backend. Reading a single boolean would
    // be wrong on one of these two cards; shipped provider truth is covered by the catalog drift
    // test instead of being frozen into this UI-state unit test.
    api = render(
      <WorkerProgressCard
        job={job({ payload: { model: "split_preview_fixture" } })}
        thumbnailsVariant="image-grid"
      />,
      contextWith([candleWorker]),
    );
    expect(
      api.container.querySelector(".worker-progress-card").getAttribute("data-preview-state"),
    ).toBe("unsupported");
    api.cleanup();

    api = render(
      <WorkerProgressCard
        job={job({ payload: { model: "split_preview_fixture" }, workerId: "worker-mlx-1" })}
        thumbnailsVariant="image-grid"
      />,
      contextWith([mlxWorker]),
    );
    expect(
      api.container.querySelector(".worker-progress-card").getAttribute("data-preview-state"),
    ).toBe("pending");
  });

  it("a model with no catalog answer is UNKNOWN, not 'does not support'", () => {
    api = render(
      <WorkerProgressCard
        job={job({ payload: { model: "some_external_model" } })}
        thumbnailsVariant="image-grid"
      />,
      contextWith([candleWorker]),
    );
    expect(
      api.container.querySelector(".worker-progress-card").getAttribute("data-preview-state"),
    ).toBe("unknown");
    expect(api.container.querySelector("[data-testid='live-preview-placeholder']")).toBeNull();
  });

  it("an unresolvable worker backend is UNKNOWN rather than an invented answer", () => {
    api = render(
      <WorkerProgressCard job={job({ workerId: "worker-gone" })} thumbnailsVariant="image-grid" />,
      contextWith([]),
    );
    expect(
      api.container.querySelector(".worker-progress-card").getAttribute("data-preview-state"),
    ).toBe("unknown");
  });

  // The bound the story requires: a route that advertises support but never emits must degrade to
  // today's behaviour, not leave a placeholder up for the whole render. Denoise reporting (stage
  // `generating`) is the boundary — that same update is the one that would carry the first frame.
  it("the placeholder is bounded: it disappears once denoise reports", () => {
    api = render(
      <WorkerProgressCard
        job={job({ stage: "generating", progress: 0.5 })}
        thumbnailsVariant="image-grid"
      />,
      contextWith([candleWorker]),
    );
    expect(
      api.container.querySelector(".worker-progress-card").getAttribute("data-preview-state"),
    ).toBe("supported");
    expect(api.container.querySelector("[data-testid='live-preview-placeholder']")).toBeNull();
  });

  it("terminal jobs never show the placeholder", () => {
    for (const status of ["completed", "failed", "canceled"]) {
      api?.cleanup();
      api = render(
        <WorkerProgressCard
          job={job({ status, stage: "queued" })}
          thumbnailsVariant="image-grid"
        />,
        contextWith([candleWorker]),
      );
      expect(
        api.container.querySelector("[data-testid='live-preview-placeholder']"),
        status,
      ).toBeNull();
    }
  });

  it("an explicit interimThumbnailAssets prop still wins over the placeholder", () => {
    const explicit = [
      { id: "explicit-1", type: "image", url: "/api/v1/files/e-1.png", __interim: true },
    ];
    api = render(
      <WorkerProgressCard
        job={job()}
        thumbnailsVariant="image-grid"
        interimThumbnailAssets={explicit}
      />,
      contextWith([candleWorker]),
    );
    expect(api.container.querySelector("[data-testid='live-preview-placeholder']")).toBeNull();
    expect(api.container.querySelectorAll(".worker-progress-card__thumb-cell.interim")).toHaveLength(
      1,
    );
  });
});

// sc-19577 — the queue card is the FIRST place a just-finished render is seen, and for the joint
// audio+video family it is where "did this one come out with sound?" is asked. The badge's own
// suite covers the component and the review-grid/detail mount sites; this covers the queue card's
// `video-player` variant, which renders a different subtree (`VideoThumbnail`) and so is not
// reached by any of those.
describe("WorkerProgressCard audio track badge (sc-19577)", () => {
  let api = null;

  afterEach(() => {
    api?.cleanup();
    api = null;
  });

  // One fixture, `hasAudio` spliced in per case, so every assertion below discriminates on that one
  // key rather than on some unrelated absence.
  const clip = (hasAudio) => {
    const file = {
      path: "asset_1.mp4",
      mimeType: "video/mp4",
      width: 1344,
      height: 768,
      duration: 5.1667,
      fps: 24,
      frameCount: 124,
    };
    if (hasAudio !== undefined) {
      file.hasAudio = hasAudio;
    }
    return {
      id: "asset_1",
      type: "video",
      url: "/api/v1/files/asset_1.mp4",
      projectId: "project_1",
      generationSetId: "genset_1",
      file,
      status: {},
      recipe: { model: "minimax_h3", prompt: "a lighthouse keeper hums" },
    };
  };

  const videoJob = {
    id: "job-1",
    type: "video_generate",
    status: "succeeded",
    stage: "done",
    payload: { prompt: "a lighthouse keeper hums" },
  };

  const renderCard = (asset) =>
    render(
      <WorkerProgressCard
        job={videoJob}
        thumbnailsVariant="video-player"
        thumbnailAssets={[asset]}
      />,
      makeContext([appleWorker]),
    );

  it("badges a measured soundtrack on the queue card, and nothing else", () => {
    api = renderCard(clip(true));
    expect(api.container.querySelector(".audio-track-badge")).not.toBeNull();
    expect(api.container.querySelector(".audio-track-badge").getAttribute("aria-label")).toBe(
      "Has audio",
    );
    api.cleanup();

    // Measured and SILENT — the same MiniMax-H3 checkpoint produces both, so a family lookup would
    // wrongly badge this one.
    api = renderCard(clip(false));
    expect(api.container.querySelector(".audio-track-badge")).toBeNull();
    api.cleanup();

    // Never measured (every render predating sc-19577): no claim either way.
    api = renderCard(clip(undefined));
    expect(api.container.querySelector(".audio-track-badge")).toBeNull();
    api.cleanup();

    // …and back, so the two negatives above are discriminating on `hasAudio` rather than on the
    // queue card having quietly stopped rendering the badge at all.
    api = renderCard(clip(true));
    expect(api.container.querySelector(".audio-track-badge")).not.toBeNull();
  });
});
