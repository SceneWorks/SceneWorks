// Bundled third-party license corpus for the About → Licenses screen (sc-3778).
//
// Single source of truth is apps/desktop/licenses/ — the same tracked files that
// build-sidecar.mjs stages next to the bundled binaries (ffmpeg GPLv3 §6 text,
// onnxruntime MIT notice). We import that corpus directly (manifest metadata as
// JSON, license text as ?raw) rather than keeping a second copy here, so the
// in-app notices can never drift from what actually ships. The embedded desktop
// UI is the same web build, so this works on every platform with no Tauri command
// or API round-trip.
import manifest from "../../../desktop/licenses/manifest.json";
import ffmpegNotice from "../../../desktop/licenses/ffmpeg/NOTICE.txt?raw";
import ffmpegGpl from "../../../desktop/licenses/ffmpeg/COPYING.GPLv3?raw";
import onnxruntimeNotice from "../../../desktop/licenses/onnxruntime/NOTICE.txt?raw";
import onnxruntimeLicense from "../../../desktop/licenses/onnxruntime/LICENSE?raw";
import cudaNotice from "../../../desktop/licenses/cuda/NOTICE.txt?raw";

// Maps a manifest document `key` to its imported text. New components: add the
// files under apps/desktop/licenses/<id>/, list them in manifest.json, and wire
// their keys here.
const DOCUMENT_TEXT = {
  "ffmpeg-notice": ffmpegNotice,
  "ffmpeg-gpl": ffmpegGpl,
  "onnxruntime-notice": onnxruntimeNotice,
  "onnxruntime-license": onnxruntimeLicense,
  "cuda-notice": cudaNotice,
};

// Resolve each component's document keys to its actual text once, at module load.
export const bundledLicenses = (manifest.components ?? []).map((component) => ({
  ...component,
  documents: (component.documents ?? [])
    .map((doc) => ({ label: doc.label, text: DOCUMENT_TEXT[doc.key] }))
    .filter((doc) => typeof doc.text === "string"),
}));

export const licensesIntro = manifest.description ?? "";
