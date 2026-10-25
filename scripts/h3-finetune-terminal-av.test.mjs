// Focused CPU-only tests for terminal review finding 14: request-bound AV acceptance.
// No ffmpeg/ffprobe/model/server/browser/listener/Rust execution: the decode path is
// exercised with an injected command and a synthetic file, exactly as `decodeAV` allows.
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {
  MINIMAX_H3_LEGAL_FRAME_COUNTS, MINIMAX_H3_FPS, MINIMAX_H3_AUDIO_SAMPLE_RATE,
  rawFrameCount, minimaxH3FrameCount, ltxFrameCount, wanFrameCount, mochiFrameCount,
  videoFrameCount, isLegalFrameCount, audioFrameSamples,
  decodeAV, validateAVAgainstRequest, decodeAndValidateAV,
} from './h3-finetune-terminal.mjs';

const H3_PICTURE = 124 / 24;
const h3Request = (overrides = {}) => ({ model: 'minimax_h3_ref', fps: 24, width: 576, height: 320, duration: 5.1667, ...overrides });
const h3Contract = (overrides = {}) => ({
  model: 'minimax_h3_ref', fps: 24, width: 576, height: 320, frameCount: 124,
  audio: { required: true, sampleRate: 32_000, channels: 2, codec: 'aac' }, ...overrides,
});
function h3Decoded({ frames = 124, avg = '24/1', real = '24/1', width = 576, height = 320, videoDuration = frames / 24, videoStart = '0.000000', audioDuration = H3_PICTURE, audioStart = '0.000000', sampleRate = '32000', channels = 2, codec = 'aac', audio = true } = {}) {
  const streams = [{ codec_type: 'video', codec_name: 'h264', width, height, avg_frame_rate: avg, r_frame_rate: real, nb_read_frames: String(frames), nb_frames: String(frames), duration: String(videoDuration), start_time: videoStart }];
  if (audio) streams.push({ codec_type: 'audio', codec_name: codec, sample_rate: String(sampleRate), channels, nb_read_frames: '162', duration: String(audioDuration), start_time: audioStart });
  return { sha256: 'a'.repeat(64), streams };
}
const rejects = (decoded, request, contract, pattern) => assert.throws(() => validateAVAgainstRequest(decoded, request, contract), pattern);

test('frame ladders are the source-derived engine rules, not approximations', () => {
  // crates/sceneworks-core/src/video_request.rs: MINIMAX_H3_LEGAL_FRAME_COUNTS (523-525),
  // minimax_h3_frame_count aligns UP (556-563), ltx nearest 8k+1 (573-587),
  // wan floor 4k+1 (593-596), mochi nearest 6k+1 (610-620).
  assert.deepEqual([...MINIMAX_H3_LEGAL_FRAME_COUNTS], [124, 141, 158, 175, 192, 209, 226, 243, 260, 277, 294, 311, 328, 345]);
  assert.equal(MINIMAX_H3_FPS, 24);
  assert.equal(MINIMAX_H3_AUDIO_SAMPLE_RATE, 32_000);
  assert.equal(rawFrameCount(5.1667, 24), 124); // 124.0008 rounds to 124
  assert.equal(minimaxH3FrameCount(124), 124);
  assert.equal(minimaxH3FrameCount(125), 141, 'H3 aligns UP to the next 17n+5 rung');
  assert.equal(minimaxH3FrameCount(9999), 345, 'clamped to the released ceiling');
  assert.equal(ltxFrameCount(150), 153, 'LTX snaps to the nearest 8k+1');
  assert.equal(ltxFrameCount(1), 9);
  assert.equal(wanFrameCount(150), 149, 'Wan floors to the largest 4k+1');
  assert.equal(wanFrameCount(4), 5);
  assert.equal(mochiFrameCount(150), 151, 'Mochi snaps to the nearest 6k+1');
  assert.equal(videoFrameCount('minimax_h3_ref', 125), 141);
  assert.equal(videoFrameCount('ltx_2_5', 150), 153);
  assert.equal(videoFrameCount('wan_2_2', 150), 149);
  assert.equal(isLegalFrameCount('minimax_h3', 124), true);
  assert.equal(isLegalFrameCount('minimax_h3', 125), false);
  assert.equal(isLegalFrameCount('ltx_2_5', 153), true);
  assert.equal(isLegalFrameCount('wan_2_2', 149), true);
});

test('source-derived audio padding quanta per codec', () => {
  assert.equal(audioFrameSamples('aac', 32_000), 1024);
  assert.equal(audioFrameSamples('mp3', 44_100), 1152);
  assert.equal(audioFrameSamples('opus', 48_000), 960);
  assert.equal(audioFrameSamples('pcm_s16le', 32_000), 1);
  assert.equal(audioFrameSamples('some-future-codec', 32_000), null, 'unknown codecs are refused, not given arbitrary slack');
});

test('a decoded H3 job that matches its resolved request is accepted', () => {
  const report = validateAVAgainstRequest(h3Decoded(), h3Request(), h3Contract());
  assert.equal(report.frameCount, 124);
  assert.equal(report.fps, 24);
  assert.equal(report.durationSeconds, H3_PICTURE);
  assert.equal(report.audio.codec, 'aac');
  assert.equal(report.audio.sampleRate, 32_000);
  assert.equal(report.audio.expectedSamples, 165_333); // round(124/24*32000), the engine's fit target
  assert.equal(report.audio.frameSamples, 1024);
});

test('short video for a long request fails even though it decodes and has a positive duration', () => {
  rejects(h3Decoded({ frames: 100, videoDuration: 100 / 24 }), h3Request(), h3Contract(), /holds 100 frames/);
  // A 10 s (243-frame) request fulfilled by the 5 s clip.
  rejects(h3Decoded(), h3Request({ duration: 10.125 }), h3Contract({ frameCount: 243 }), /holds 124 frames .* resolved to 243/);
});

test('wrong fps, frame lattice, dimensions, canvas grid and start time all fail', () => {
  rejects(h3Decoded({ avg: '25/1' }), h3Request(), h3Contract(), /average fps 25 differs/);
  rejects(h3Decoded({ real: '25/1' }), h3Request(), h3Contract(), /real fps 25 differs/);
  rejects(h3Decoded(), h3Request(), h3Contract({ frameCount: 125 }), /not the request lattice snap/);
  rejects(h3Decoded({ width: 640 }), h3Request(), h3Contract(), /decoded width differs/);
  rejects(h3Decoded({ height: 322 }), h3Request(), h3Contract(), /decoded height differs/);
  rejects(h3Decoded({ width: 580, height: 320 }), h3Request({ width: 580 }), h3Contract({ width: 580 }), /off the model's 32px spatial grid/);
  rejects(h3Decoded({ videoStart: '0.500000' }), h3Request(), h3Contract(), /not the request origin/);
  rejects(h3Decoded({ audioStart: '0.500000' }), h3Request(), h3Contract(), /audio starts at 0.5/);
});

test('out-of-window requests are refused, never clamped onto an endpoint rung', () => {
  // 20 s * 24 fps snaps to the 345 ceiling, but the engine refuses >14.375 s.
  rejects(h3Decoded({ frames: 345, videoDuration: 345 / 24, audioDuration: 345 / 24 }), h3Request({ duration: 20 }), h3Contract({ frameCount: 345 }), /above the renderable maximum/);
  rejects(h3Decoded(), h3Request({ duration: 2 }), h3Contract(), /below the renderable minimum/);
});

test('request/contract drift is refused rather than silently trusted', () => {
  rejects(h3Decoded(), h3Request({ fps: 25 }), h3Contract(), /request fps differs/);
  rejects(h3Decoded(), h3Request({ width: 640 }), h3Contract(), /request width differs/);
  rejects(h3Decoded(), h3Request(), h3Contract({ fps: 25 }), /request fps differs/);
});

test('short, long, wrong-rate, wrong-channel and wrong-codec audio fail', () => {
  rejects(h3Decoded({ audioDuration: 2 }), h3Request(), h3Contract(), /decoded audio lasts 2s/);
  rejects(h3Decoded({ audioDuration: 10.333 }), h3Request(), h3Contract(), /decoded audio lasts 10.333s/);
  rejects(h3Decoded({ sampleRate: '44100' }), h3Request(), h3Contract(), /sample rate differs/);
  rejects(h3Decoded({ channels: 1 }), h3Request(), h3Contract(), /channels differ/);
  rejects(h3Decoded({ codec: 'opus' }), h3Request(), h3Contract(), /codec differs/);
  rejects(h3Decoded({ audio: false }), h3Request(), h3Contract(), /missing decoded audio/);
  rejects(h3Decoded({ codec: 'some-future-codec' }), h3Request(), h3Contract({ audio: { required: true, sampleRate: 32_000, channels: 2 } }), /no source-derived padding rule/);
});

test('legitimate AAC padding passes only within one codec frame', () => {
  // The mux trims at frames/fps with -t but AAC emits 1024-sample frames, so the
  // last packet may extend past the picture by up to one frame.
  const padded = validateAVAgainstRequest(h3Decoded({ audioDuration: H3_PICTURE + 1023 / 32_000 }), h3Request(), h3Contract());
  assert.equal(padded.audio.frameSamples, 1024);
  rejects(h3Decoded({ audioDuration: H3_PICTURE + 1026 / 32_000 }), h3Request(), h3Contract(), /decoded audio lasts/);
});

test('LTX uses its own fps/grid/audio rates from the resolved contract', () => {
  const request = { model: 'ltx_2_5', fps: 25, width: 768, height: 512, duration: 6 };
  const contract = { model: 'ltx_2_5', fps: 25, width: 768, height: 512, frameCount: 153, audio: { required: true, sampleRate: 48_000, channels: 2, codec: 'aac' } };
  const picture = 153 / 25;
  const streams = [
    { codec_type: 'video', codec_name: 'h264', width: 768, height: 512, avg_frame_rate: '25/1', r_frame_rate: '25/1', nb_read_frames: '153', nb_frames: '153', duration: String(picture), start_time: '0.000000' },
    { codec_type: 'audio', codec_name: 'aac', sample_rate: '48000', channels: 2, nb_read_frames: '287', duration: String(picture), start_time: '0.000000' },
  ];
  const report = validateAVAgainstRequest({ streams }, request, contract);
  assert.equal(report.frameCount, 153);
  assert.equal(report.audio.expectedSamples, 293_760);
  const wrongRate = { streams: [streams[0], { ...streams[1], sample_rate: '24000' }] };
  rejects(wrongRate, request, contract, /sample rate differs/);
});

test('unbound decodeAV still decodes both streams with an injected command; the bound form adds the request verdict', async t => {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'h3-av-test-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const file = path.join(dir, 'synthetic-av.bin');
  await writeFile(file, 'synthetic AV bytes, never passed to a real decoder');
  const calls = [];
  const command = async (bin, args) => {
    calls.push({ bin, args });
    if (bin === 'ffprobe') return { stdout: JSON.stringify({ streams: h3Decoded().streams }) };
    return { stdout: Buffer.alloc(0) };
  };
  const decoded = await decodeAV(file, command);
  assert.equal(decoded.streams.length, 2);
  assert.equal(calls.filter(c => c.bin === 'ffprobe').length, 1);
  assert.equal(calls.filter(c => c.bin === 'ffmpeg').length, 1);
  const ffprobe = calls.find(c => c.bin === 'ffprobe').args;
  assert.ok(ffprobe.includes('-count_frames'), 'ffprobe must count actual frames');
  const ffmpeg = calls.find(c => c.bin === 'ffmpeg').args;
  assert.ok(ffmpeg.includes('0:v:0') && ffmpeg.includes('0:a:0') && ffmpeg.includes('-xerror'), 'ffmpeg must fully decode both streams');
  const bound = await decodeAndValidateAV(file, h3Request(), h3Contract(), command);
  assert.equal(bound.verification.frameCount, 124);
  assert.equal(calls.filter(c => c.bin === 'ffmpeg').length, 2);
});
