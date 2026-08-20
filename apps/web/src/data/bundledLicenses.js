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
import ltx23IcLorasNotice from "../../../desktop/licenses/ltx-2.3-ic-loras/NOTICE.txt?raw";
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

// Image / video / utility model weights (sc-13803). The About→Licenses page recorded only the
// bundled binaries, the Wan2.2/LTX video models and the audio models; every other shipped catalog
// primary was missing its upstream attribution. Each license below was verified against the
// upstream Hugging Face cardData, or against the LICENSE file shipped inside the SceneWorks
// re-host where cardData declared none. Several are RESTRICTED (non-commercial, RAIL use-based
// restrictions, or revenue-threshold community licenses) and are surfaced as such.
import animaCircleStoneLabsNonCommercialLicense from "../../../desktop/licenses/anima/CircleStone-Labs-Non-Commercial-License.txt?raw";
import berniniApache20 from "../../../desktop/licenses/bernini/Apache-2.0.txt?raw";
import booguImageApache20 from "../../../desktop/licenses/boogu-image/Apache-2.0.txt?raw";
import chroma1Apache20 from "../../../desktop/licenses/chroma1/Apache-2.0.txt?raw";
import flux1DevFLUX1DevNonCommercialLicense from "../../../desktop/licenses/flux1-dev/FLUX.1-dev-Non-Commercial-License.txt?raw";
import flux1SchnellApache20 from "../../../desktop/licenses/flux1-schnell/Apache-2.0.txt?raw";
import flux2DevFLUX2DevNonCommercialLicense from "../../../desktop/licenses/flux2-dev/FLUX.2-dev-Non-Commercial-License.txt?raw";
import flux2KleinFLUXNonCommercialLicense from "../../../desktop/licenses/flux2-klein/FLUX-Non-Commercial-License.txt?raw";
import ideogram4IdeogramNonCommercialModelAgreement from "../../../desktop/licenses/ideogram-4/Ideogram-Non-Commercial-Model-Agreement.txt?raw";
import illustriousXlV1CreativeMLOpenRAILM from "../../../desktop/licenses/illustrious-xl-v1/CreativeML-Open-RAIL++-M.txt?raw";
import illustriousXlV2CreativeMLOpenRAILM from "../../../desktop/licenses/illustrious-xl-v2/CreativeML-Open-RAIL-M.txt?raw";
import kolorsKolorsModelLicense from "../../../desktop/licenses/kolors/Kolors-Model-License.txt?raw";
import krea2NOTICE from "../../../desktop/licenses/krea-2/NOTICE.txt?raw";
import lensMIT from "../../../desktop/licenses/lens/MIT.txt?raw";
import qwenImageApache20 from "../../../desktop/licenses/qwen-image/Apache-2.0.txt?raw";
import realvisxlCreativeMLOpenRAILM from "../../../desktop/licenses/realvisxl/CreativeML-Open-RAIL++-M.txt?raw";
import sanaNVIDIAOpenModelLicense from "../../../desktop/licenses/sana/NVIDIA-Open-Model-License.txt?raw";
import sd35StabilityAICommunityLicense from "../../../desktop/licenses/sd3.5/Stability-AI-Community-License.txt?raw";
import sdxlCreativeMLOpenRAILM from "../../../desktop/licenses/sdxl/CreativeML-Open-RAIL++-M.txt?raw";
import sensenovaU1Apache20 from "../../../desktop/licenses/sensenova-u1/Apache-2.0.txt?raw";
import zImageApache20 from "../../../desktop/licenses/z-image/Apache-2.0.txt?raw";
import ltx23ErosLTX2CommunityLicense from "../../../desktop/licenses/ltx-2.3-eros/LTX-2-Community-License.txt?raw";
import kreaRealtimeApache20 from "../../../desktop/licenses/krea-realtime/Apache-2.0.txt?raw";
// MiniMax-H3 / Hailuo 3.0 (epic 17137, sc-17158). TWO notices, like the LTX-2.3 bundle: the DiT
// weights are under MiniMax's own restricted community licence, while the text encoder is
// byte-for-byte Qwen3-VL-32B-Instruct and stays Apache-2.0. Both entries — `minimax_h3` and
// `minimax_h3_ref` — download the same components, so one component covers both.
// The NOTICE is a THIRD document (sc-17227): §III.4 of the community licence requires
// distributions to be accompanied by a "NOTICE" text file carrying a specific attribution
// sentence, and About → Licenses is where SceneWorks serves it.
import minimaxH3Notice from "../../../desktop/licenses/minimax-h3/NOTICE.txt?raw";
import minimaxH3CommunityLicense from "../../../desktop/licenses/minimax-h3/MiniMax-H3-Community-License.txt?raw";
import minimaxH3QwenApache20 from "../../../desktop/licenses/minimax-h3/Apache-2.0.txt?raw";
import wan21VaeApache20 from "../../../desktop/licenses/wan2_1_t2v_14b_diffusers/Apache-2.0.txt?raw";
import scail2MIT from "../../../desktop/licenses/scail2/MIT.txt?raw";
import svdStableVideoDiffusionCommunityLicense from "../../../desktop/licenses/svd/Stable-Video-Diffusion-Community-License.txt?raw";
import wan22VaceFunApache20 from "../../../desktop/licenses/wan2.2-vace-fun/Apache-2.0.txt?raw";
import auraSrV2Apache20 from "../../../desktop/licenses/aura-sr-v2/Apache-2.0.txt?raw";
import sam3License from "../../../desktop/licenses/sam3/SAM-License.txt?raw";
import sam2Apache20 from "../../../desktop/licenses/sam2/Apache-2.0.txt?raw";
import yolo11Agpl30 from "../../../desktop/licenses/yolo11/AGPL-3.0.txt?raw";
import controlnetTileSdxlApache20 from "../../../desktop/licenses/controlnet-tile-sdxl/Apache-2.0.txt?raw";
import pidNVIDIALicense from "../../../desktop/licenses/pid/NVIDIA-License.txt?raw";
import realEsrganBSD3Clause from "../../../desktop/licenses/real-esrgan/BSD-3-Clause.txt?raw";
import qwen3VlCaptionApache20 from "../../../desktop/licenses/qwen3-vl-caption/Apache-2.0.txt?raw";
import seedvr2Apache20 from "../../../desktop/licenses/seedvr2/Apache-2.0.txt?raw";
// DWPose (sc-17634): mmpose's OWN LICENSE file, not the stock Apache text — it carries the
// "Copyright 2018-2020 Open-MMLab. All rights reserved." notice, which is the attribution that has
// to travel with the two ONNX graphs SceneWorks re-hosts at SceneWorks/dwpose-onnx.
import dwposeApache20 from "../../../desktop/licenses/dwpose/Apache-2.0.txt?raw";
// InsightFace SCRFD + ArcFace weights (sc-19708): upstream publishes NO standalone weights
// license — the governing terms live in the project README, so the notice reproduces those
// statements VERBATIM (code MIT; models trained on InsightFace's annotated data are for
// non-commercial research purposes only). Not a generic license template on purpose.
import insightfaceModelNotice from "../../../desktop/licenses/insightface/InsightFace-Model-License-Notice.txt?raw";
// Production third-party source/data compiled into the inference runtimes (sc-14403).
import cephesBsd3Clause from "../../../desktop/licenses/cephes/BSD-3-Clause.txt?raw";
import mageMit from "../../../desktop/licenses/mage/MIT.txt?raw";
import cmudictBsd2Clause from "../../../desktop/licenses/cmudict/BSD-2-Clause.txt?raw";
// Upstream CONTENT (not an algorithm port) reproduced in the native captioners: the JoyCaption
// prompt taxonomy. This is fpgaminer/joycaption's own LICENSE file, kept verbatim because it
// carries the upstream copyright line the generic Apache template does not (sc-15191 review).
import joycaptionApache20 from "../../../desktop/licenses/joycaption-source/Apache-2.0.txt?raw";

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
  "ltx-2.3-ic-loras-notice": ltx23IcLorasNotice,
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
  "anima-circlestone-labs-non-commercial-license": animaCircleStoneLabsNonCommercialLicense,
  "bernini-apache": berniniApache20,
  "boogu-image-apache": booguImageApache20,
  "chroma1-apache": chroma1Apache20,
  "flux1-dev-flux-1-dev-non-commercial-license": flux1DevFLUX1DevNonCommercialLicense,
  "flux1-schnell-apache": flux1SchnellApache20,
  "flux2-dev-flux-2-dev-non-commercial-license": flux2DevFLUX2DevNonCommercialLicense,
  "flux2-klein-flux-non-commercial-license": flux2KleinFLUXNonCommercialLicense,
  "ideogram-4-ideogram-non-commercial-model-agreement": ideogram4IdeogramNonCommercialModelAgreement,
  "illustrious-xl-v1-creativeml-open-rail-m": illustriousXlV1CreativeMLOpenRAILM,
  "illustrious-xl-v2-creativeml-open-rail-m": illustriousXlV2CreativeMLOpenRAILM,
  "kolors-kolors-model-license": kolorsKolorsModelLicense,
  "krea-2-notice": krea2NOTICE,
  "lens-mit": lensMIT,
  "qwen-image-apache": qwenImageApache20,
  "realvisxl-creativeml-open-rail-m": realvisxlCreativeMLOpenRAILM,
  "sana-nvidia-open-model-license": sanaNVIDIAOpenModelLicense,
  "sd3.5-stability-ai-community-license": sd35StabilityAICommunityLicense,
  "sdxl-creativeml-open-rail-m": sdxlCreativeMLOpenRAILM,
  "sensenova-u1-apache": sensenovaU1Apache20,
  "z-image-apache": zImageApache20,
  "ltx-2.3-eros-ltx-2-community-license": ltx23ErosLTX2CommunityLicense,
  "krea-realtime-apache": kreaRealtimeApache20,
  "minimax-h3-notice": minimaxH3Notice,
  "minimax-h3-community-license": minimaxH3CommunityLicense,
  "minimax-h3-qwen-apache": minimaxH3QwenApache20,
  "wan2.1-vae-alternate-apache": wan21VaeApache20,
  "scail2-mit": scail2MIT,
  "svd-stable-video-diffusion-community-license": svdStableVideoDiffusionCommunityLicense,
  "wan2.2-vace-fun-apache": wan22VaceFunApache20,
  "aura-sr-v2-apache": auraSrV2Apache20,
  "sam3-license": sam3License,
  "sam2-apache": sam2Apache20,
  "yolo11-agpl": yolo11Agpl30,
  "controlnet-tile-sdxl-apache": controlnetTileSdxlApache20,
  "pid-nvidia-license": pidNVIDIALicense,
  "real-esrgan-bsd-3-clause": realEsrganBSD3Clause,
  "qwen3-vl-caption-apache": qwen3VlCaptionApache20,
  "seedvr2-apache": seedvr2Apache20,
  "dwpose-apache": dwposeApache20,
  "insightface-model-notice": insightfaceModelNotice,
  "cephes-bsd-3-clause": cephesBsd3Clause,
  "mage-mit": mageMit,
  "cmudict-bsd-2-clause": cmudictBsd2Clause,
  "joycaption-source-apache": joycaptionApache20,
};

// Resolve each component's document keys to its actual text once, at module load.
export const bundledLicenses = (manifest.components ?? []).map((component) => ({
  ...component,
  documents: (component.documents ?? [])
    .map((doc) => ({ label: doc.label, text: DOCUMENT_TEXT[doc.key] }))
    .filter((doc) => typeof doc.text === "string"),
}));

export const licensesIntro = manifest.description ?? "";
