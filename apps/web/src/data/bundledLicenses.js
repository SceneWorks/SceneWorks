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
// Re-hosted AI model weights (sc-5604). Upstream license text reproduced so the
// redistribution attribution travels with the app. The three Wan2.2 models are
// each redistributed under Apache-2.0; the LTX-2.3 bundle carries two restricted
// licenses (LTX-2 Community License + Google Gemma Terms).
import wanTi2v5bApache from "../../../desktop/licenses/wan2.2-ti2v-5b/Apache-2.0.txt?raw";
import wanI2vA14bApache from "../../../desktop/licenses/wan2.2-i2v-a14b/Apache-2.0.txt?raw";
import wanT2vA14bApache from "../../../desktop/licenses/wan2.2-t2v-a14b/Apache-2.0.txt?raw";
import ltxLicense from "../../../desktop/licenses/ltx-2.3/LTX-2-Community-License.txt?raw";
import ltxGemma from "../../../desktop/licenses/ltx-2.3/Gemma-Terms.txt?raw";
// Audio model weights (epic 13400, sc-13402). All permissive (Apache-2.0 / MIT) —
// downloaded on first use from the upstream Hugging Face repos and run natively
// (Candle) on every platform.
import kokoroApache from "../../../desktop/licenses/kokoro-82m/Apache-2.0.txt?raw";
import mossApache from "../../../desktop/licenses/moss-soundeffect-v2/Apache-2.0.txt?raw";
import acestepMit from "../../../desktop/licenses/acestep-v15-turbo/MIT.txt?raw";
// ACE-Step SFT Cover-restyle checkpoint (sc-13821, epic 13678): the Cover-only co-requisite of the
// Turbo Music model (transformer cover DiT + FSQ audio_tokenizer/detokenizer), MIT, same ACE-Step org.
import acestepSftCoverMit from "../../../desktop/licenses/acestep-v15-sft-cover/MIT.txt?raw";
import openvoiceMit from "../../../desktop/licenses/openvoice-v2/MIT.txt?raw";
// Chatterbox (Resemble AI, epic 13678): all three components are MIT (Copyright (c)
// 2025 Resemble AI). chatterboxTtsMit covers the PRIMARY T3/s3gen weights
// (t3_cfg.safetensors + s3gen.safetensors from ResembleAI/chatterbox); chatterboxMit
// covers the voice encoder (ve.safetensors, same repo); chatterboxPerthMit covers the
// Perth provenance watermarker staged on every clone render (SceneWorks/perth-implicit).
import chatterboxTtsMit from "../../../desktop/licenses/chatterbox-tts/MIT.txt?raw";
import chatterboxMit from "../../../desktop/licenses/chatterbox-ve/MIT.txt?raw";
import chatterboxPerthMit from "../../../desktop/licenses/chatterbox-perth/MIT.txt?raw";
// MOSS TTS speech models + their pinned codec co-requisites (epic 13678, sc-13681).
// Both the AR checkpoints and the codecs (XY_Tokenizer / MOSS-Audio-Tokenizer) are
// Apache-2.0, downloaded on first use from the upstream OpenMOSS-Team repos.
import mossTtsdApache from "../../../desktop/licenses/moss-ttsd-v05/Apache-2.0.txt?raw";
import xyTokenizerApache from "../../../desktop/licenses/xy-tokenizer-ttsd/Apache-2.0.txt?raw";
import mossTtsRealtimeApache from "../../../desktop/licenses/moss-tts-realtime/Apache-2.0.txt?raw";
import mossAudioTokenizerApache from "../../../desktop/licenses/moss-audio-tokenizer/Apache-2.0.txt?raw";
// MMAudio video→audio (Foley) — RESEARCH / NON-COMMERCIAL only (epic 13678, sc-13684). Three upstream
// licenses across three repos: CC-BY-NC-4.0 (hkchengrex/MMAudio weights), the Apple ML Research Model
// License (apple/DFN5B-CLIP conditioner, the research-only gate), and MIT (nvidia/bigvgan_v2 44k vocoder).
import mmaudioCcByNc from "../../../desktop/licenses/mmaudio/CC-BY-NC-4.0.txt?raw";
import mmaudioAppleAmlr from "../../../desktop/licenses/mmaudio/Apple-ML-Research-License.txt?raw";
import mmaudioBigvganMit from "../../../desktop/licenses/mmaudio/MIT.txt?raw";
// Whisper base (ASR) + LAION CLAP (audio embedder) — Apache-2.0 audio-validation utility models (sc-13684).
import whisperBaseApache from "../../../desktop/licenses/whisper-base/Apache-2.0.txt?raw";
import clapHtsatApache from "../../../desktop/licenses/clap-htsat-unfused/Apache-2.0.txt?raw";

// Maps a manifest document `key` to its imported text. New components: add the
// files under apps/desktop/licenses/<id>/, list them in manifest.json, and wire
// their keys here.
const DOCUMENT_TEXT = {
  "ffmpeg-notice": ffmpegNotice,
  "ffmpeg-gpl": ffmpegGpl,
  "onnxruntime-notice": onnxruntimeNotice,
  "onnxruntime-license": onnxruntimeLicense,
  "cuda-notice": cudaNotice,
  "wan2.2-ti2v-5b-apache": wanTi2v5bApache,
  "wan2.2-i2v-a14b-apache": wanI2vA14bApache,
  "wan2.2-t2v-a14b-apache": wanT2vA14bApache,
  "ltx-2.3-license": ltxLicense,
  "ltx-2.3-gemma": ltxGemma,
  "kokoro-82m-apache": kokoroApache,
  "moss-soundeffect-v2-apache": mossApache,
  "acestep-v15-turbo-mit": acestepMit,
  "acestep-v15-sft-cover-mit": acestepSftCoverMit,
  "openvoice-v2-mit": openvoiceMit,
  "chatterbox-tts-mit": chatterboxTtsMit,
  "chatterbox-ve-mit": chatterboxMit,
  "chatterbox-perth-mit": chatterboxPerthMit,
  "moss-ttsd-v05-apache": mossTtsdApache,
  "xy-tokenizer-ttsd-apache": xyTokenizerApache,
  "moss-tts-realtime-apache": mossTtsRealtimeApache,
  "moss-audio-tokenizer-apache": mossAudioTokenizerApache,
  "mmaudio-cc-by-nc": mmaudioCcByNc,
  "mmaudio-apple-amlr": mmaudioAppleAmlr,
  "mmaudio-bigvgan-mit": mmaudioBigvganMit,
  "whisper-base-apache": whisperBaseApache,
  "clap-htsat-unfused-apache": clapHtsatApache,
};

// Resolve each component's document keys to its actual text once, at module load.
export const bundledLicenses = (manifest.components ?? []).map((component) => ({
  ...component,
  documents: (component.documents ?? [])
    .map((doc) => ({ label: doc.label, text: DOCUMENT_TEXT[doc.key] }))
    .filter((doc) => typeof doc.text === "string"),
}));

export const licensesIntro = manifest.description ?? "";
