import { MAX_TERMINAL_JOBS } from "./sorters.js";
import { isAudioGenerationJob, isImageGenerationJob, isInterleaveJob, isVideoGenerationJob, localJobStackLimit } from "./appHelpers.js";
import { isEditorJob, terminalStatuses } from "./jobTypes.js";

const EMPTY_DETAIL = Object.freeze({});
const DOWNLOAD_TYPES = new Set(["model_download", "model_import", "model_convert", "lora_download", "lora_import"]);

function shellPayload(job) {
  const action = job.payload?.advanced?.timelineAction;
  if (!action) return EMPTY_DETAIL;
  // The shell's Editor activity indicator uses this small routing field. Keep
  // it without retaining the timeline context, prompts, or other request data.
  if (Object.keys(job.payload).length === 1 && Object.keys(job.payload.advanced).length === 1) return job.payload;
  return { advanced: { timelineAction: action } };
}

function pageUsesJob(job, route, simple) {
  if (route === "Queue" || (simple && route === "queue")) return true;
  if (route === "Setup" || route === "Models" || (simple && route === "models")) return DOWNLOAD_TYPES.has(job.type);
  if (route === "Library" || (simple && route === "assets")) return job.type === "image_vqa";
  if (route === "Image" || (simple && route === "image")) return isImageGenerationJob(job) || job.type === "prompt_refine" || DOWNLOAD_TYPES.has(job.type);
  if (route === "Video" || (simple && route === "video")) return isVideoGenerationJob(job) || job.type?.startsWith("person_") || DOWNLOAD_TYPES.has(job.type);
  if (route === "Audio" || (simple && route === "audio")) return isAudioGenerationJob(job) || DOWNLOAD_TYPES.has(job.type);
  if (route === "Document") return isInterleaveJob(job) || DOWNLOAD_TYPES.has(job.type);
  if (route === "Editor") return isEditorJob(job) || DOWNLOAD_TYPES.has(job.type);
  if (route === "ImageEditor") return job.type?.startsWith("image_") || job.type?.startsWith("mask_");
  if (route === "Characters") return isImageGenerationJob(job) || job.type === "lora_train" || DOWNLOAD_TYPES.has(job.type);
  if (route === "Train" || route === "LibraryDataSets") return ["lora_train", "control_training"].includes(job.type) || job.type?.startsWith("training_") || job.type?.startsWith("dataset_") || DOWNLOAD_TYPES.has(job.type);
  if (route === "Poses") return job.type === "pose_detect";
  if (route === "Keypoints") return job.type === "kps_extract";
  if (route === "VectorStudio") return job.type?.startsWith("vector_") || DOWNLOAD_TYPES.has(job.type);
  return false;
}

// Keep live status/progress/revisions for shell counts and background work, but
// release payloads, previews and output records without a foreground consumer.
// Queue retains its complete existing history and every progress field.
export function retainPageJobs(jobs, { route, simple = false, projectId, ownsScratch = () => false }) {
  const queue = route === "Queue" || (simple && route === "queue");
  const history = new Map();
  let changed = false;
  const next = jobs.map((job) => {
    // Keypoint capture belongs to its global library project, independently of
    // the workspace currently selected in the shell.
    const projectMatches = route === "Keypoints" || !job.projectId || job.projectId === projectId;
    const relevant = queue || (projectMatches && pageUsesJob(job, route, simple));
    const bucket = isImageGenerationJob(job) ? "image" : isVideoGenerationJob(job) ? "video" : job.type;
    const count = history.get(bucket) ?? 0;
    if (relevant && terminalStatuses.has(job.status)) history.set(bucket, count + 1);
    const limit = route === "Library" || (simple && route === "assets") ? MAX_TERMINAL_JOBS : localJobStackLimit;
    const detail = relevant && (!terminalStatuses.has(job.status) || queue || count < limit);
    if (detail || ownsScratch(job.id)) return job;
    const payload = shellPayload(job);
    if (job.payload === payload && job.result === EMPTY_DETAIL) return job;
    changed = true;
    return { ...job, payload, result: EMPTY_DETAIL };
  });
  return changed ? next : jobs;
}

export function pageNeedsJobDetails(route, simple = false) {
  return !["Settings", "Stats", "Logs", "Licenses", "DatasetCatalogs", "Presets", "settings", "licenses"].includes(route)
    && (simple || Boolean(route));
}
