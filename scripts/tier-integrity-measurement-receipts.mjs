/**
 * Reproducible sc-16015 safetensors-header receipts.
 *
 * Each key is `(model, component, backend lanes)`. Each tuple is
 * `[selectedElementCount, runtimeBytesPerElement]`; summing tuple products must reproduce every
 * `costBytesByTier` value in the ledger. The counts came from the immutable artifact and selector
 * named by the row's `evidence.source`. Keeping them separate from the reported byte totals makes a
 * casual or accidental byte edit fail the checker instead of silently rewriting the measurement.
 * Mixed-runtime-dtype components have more than one term (the Candle SenseNova head is BF16 embeds
 * plus an F32 flow-matching head).
 *
 * Every `::vae::` receipt has a structured scope below. `full-autoencoder` counts encoder + decoder;
 * `decoder-only` excludes encoder tensors that the entry's advertised routes never construct. The
 * scope map is keyed identically to these terms, so the checker binds the row's declaration to the
 * exact selected element count instead of trusting prose.
 */
export const HEADER_MEASUREMENT_TERMS = Object.freeze({
  "flux2_klein_9b::textEncoder::mlx": [[8190735360, 2]],
  "flux2_klein_9b::textEncoder::candle": [[8190735360, 4]],
  "flux2_klein_9b_kv::textEncoder::mlx": [[8190735360, 2]],
  "flux2_klein_9b_kv::textEncoder::candle": [[8190735360, 4]],
  "chroma1_base::textEncoder::mlx": [[4762310656, 2]],
  "chroma1_hd::textEncoder::mlx": [[4762310656, 2]],
  "chroma1_flash::textEncoder::mlx": [[4762310656, 2]],
  "qwen_image::textEncoder::mlx": [[8292166656, 2]],
  "qwen_image::textEncoder::candle": [[8292166656, 4]],
  "qwen_image_edit_2511::textEncoder::mlx": [[8292166656, 2]],
  "qwen_image_edit_2511::textEncoder::candle": [[8292166656, 4]],
  "qwen_image_edit_2511_lightning::textEncoder::mlx": [[8292166656, 2]],
  "qwen_image_edit_2511_lightning::textEncoder::candle": [[8292166656, 4]],
  "sd3_5_large::textEncoder::mlx": [[5580620800, 4]],
  "sd3_5_large::textEncoder::candle": [[5580620800, 2]],
  "sd3_5_large_turbo::textEncoder::mlx": [[5580620800, 4]],
  "sd3_5_large_turbo::textEncoder::candle": [[5580620800, 2]],
  "sd3_5_medium::textEncoder::mlx": [[5580620800, 4]],
  "sd3_5_medium::textEncoder::candle": [[5580620800, 2]],
  "anima_base::textEncoder::mlx+candle": [[596049920, 2]],
  "anima_aesthetic::textEncoder::mlx+candle": [[596049920, 2]],
  "anima_turbo::textEncoder::mlx+candle": [[596049920, 2]],
  "boogu_image_edit::visionTower::mlx+candle": [[576388336, 4]],
  "sdxl::vae::mlx": [[83653863, 4]],
  "sdxl::vae::candle": [[83653863, 2]],
  "realvisxl::vae::mlx": [[83653863, 4]],
  "realvisxl::vae::candle": [[83653863, 2]],
  "realvisxl_lightning::vae::mlx": [[83653863, 4]],
  "realvisxl_lightning::vae::candle": [[83653863, 2]],
  "illustrious_xl_v1::vae::mlx": [[83653863, 4]],
  "illustrious_xl_v1::vae::candle": [[83653863, 2]],
  "illustrious_xl_v2::vae::mlx": [[83653863, 4]],
  "illustrious_xl_v2::vae::candle": [[83653863, 2]],
  "kolors::vae::mlx+candle": [[83653863, 4]],
  "sana_1600m::vae::mlx+candle": [[312250275, 4]],
  "sana_sprint_1600m::vae::mlx+candle": [[312250275, 4]],
  "flux2_dev::vae::mlx+candle": [[84046371, 4]],
  "flux2_klein_9b::vae::mlx+candle": [[84046371, 4]],
  "flux2_klein_9b_kv::vae::mlx+candle": [[84046371, 4]],
  "qwen_image::vae::mlx": [[126892531, 2]],
  "qwen_image::vae::candle": [[73295603, 4]],
  "krea_2_turbo::vae::mlx+candle": [[126892531, 4]],
  "krea_2_raw::vae::mlx+candle": [[126892531, 4]],
  "ideogram_4::vae::mlx+candle": [[84046371, 4]],
  "ideogram_4_turbo::vae::mlx+candle": [[84046371, 4]],
  "boogu_image::vae::mlx+candle": [[83819683, 4]],
  "boogu_image_turbo::vae::mlx+candle": [[83819683, 4]],
  "boogu_image_edit::vae::mlx+candle": [[83819683, 4]],
  "mage_flow_base::vae::mlx+candle": [[70630211, 2]],
  "mage_flow::vae::mlx+candle": [[70630211, 2]],
  "mage_flow_turbo::vae::mlx+candle": [[70630211, 2]],
  "mage_flow_edit_base::vae::mlx+candle": [[138052035, 2]],
  "mage_flow_edit::vae::mlx+candle": [[138052035, 2]],
  "mage_flow_edit_turbo::vae::mlx+candle": [[138052035, 2]],
  "sd3_5_large::vae::mlx": [[83819683, 4]],
  "sd3_5_large::vae::candle": [[83819683, 2]],
  "sd3_5_large_turbo::vae::mlx": [[83819683, 4]],
  "sd3_5_large_turbo::vae::candle": [[83819683, 2]],
  "sd3_5_medium::vae::mlx": [[83819683, 4]],
  "sd3_5_medium::vae::candle": [[83819683, 2]],
  "chroma1_base::vae::mlx": [[83819683, 2]],
  "chroma1_hd::vae::mlx": [[83819683, 2]],
  "chroma1_flash::vae::mlx": [[83819683, 2]],
  "lens::vae::mlx": [[84046371, 4]],
  "lens::vae::candle": [[49620515, 4]],
  "lens_turbo::vae::mlx": [[84046371, 4]],
  "lens_turbo::vae::candle": [[49620515, 4]],
  "anima_base::vae::mlx": [[126892531, 4]],
  "anima_base::vae::candle": [[73295603, 4]],
  "anima_aesthetic::vae::mlx": [[126892531, 4]],
  "anima_aesthetic::vae::candle": [[73295603, 4]],
  "anima_turbo::vae::mlx": [[126892531, 4]],
  "anima_turbo::vae::candle": [[73295603, 4]],
  "sensenova_u1_8b::transformerHead::mlx": [[1274027008, 2]],
  "sensenova_u1_8b::transformerHead::candle": [[1244659712, 2], [29367296, 4]],
  "sensenova_u1_8b::visionTower::mlx": [[35137536, 2]],
  "sensenova_u1_8b::visionTower::candle": [[35137536, 4]],
  "sensenova_u1_8b_fast::transformerHead::mlx": [[1274027008, 2]],
  "sensenova_u1_8b_fast::transformerHead::candle": [[1244659712, 2], [29367296, 4]],
  "sensenova_u1_8b_fast::visionTower::mlx": [[35137536, 2]],
  "sensenova_u1_8b_fast::visionTower::candle": [[35137536, 4]],
  "sensenova_u1_8b_infographic_v2::transformerHead::mlx": [[1274027008, 2]],
  "sensenova_u1_8b_infographic_v2::transformerHead::candle": [[1244659712, 2], [29367296, 4]],
  "sensenova_u1_8b_infographic_v2::visionTower::mlx": [[35137536, 2]],
  "sensenova_u1_8b_infographic_v2::visionTower::candle": [[35137536, 4]],
  "sensenova_u1_8b_infographic_v2_fast::transformerHead::mlx": [[1274027008, 2]],
  "sensenova_u1_8b_infographic_v2_fast::transformerHead::candle": [[1244659712, 2], [29367296, 4]],
  "sensenova_u1_8b_infographic_v2_fast::visionTower::mlx": [[35137536, 2]],
  "sensenova_u1_8b_infographic_v2_fast::visionTower::candle": [[35137536, 4]],
  "sensenova_u1_8b_infographic_v3::transformerHead::mlx": [[1274027008, 2]],
  "sensenova_u1_8b_infographic_v3::transformerHead::candle": [[1244659712, 2], [29367296, 4]],
  "sensenova_u1_8b_infographic_v3::visionTower::mlx": [[35137536, 2]],
  "sensenova_u1_8b_infographic_v3::visionTower::candle": [[35137536, 4]],
  "sensenova_u1_8b_infographic_v3_fast::transformerHead::mlx": [[1274027008, 2]],
  "sensenova_u1_8b_infographic_v3_fast::transformerHead::candle": [[1244659712, 2], [29367296, 4]],
  "sensenova_u1_8b_infographic_v3_fast::visionTower::mlx": [[35137536, 2]],
  "sensenova_u1_8b_infographic_v3_fast::visionTower::candle": [[35137536, 4]],
  // ── MiniMax-H3 (epic 17137, sc-19506) — the first JOINT audio+video family in the ledger, and the
  // first entry needing two autoencoder receipts. Both counts are exact safetensors-header sums over
  // `MiniMaxAI/MiniMax-H3@939557dc319dd91227e30195a763f272ba7f8765`, the revision the catalog's own
  // co-requisite rows pin. The published component is F32 on disk in BOTH cases (703 and 1087
  // tensors, no bf16 anywhere in either), so the element counts below are on-disk bytes ÷ 4.
  //
  // The VIDEO VAE is resident at the MLX lane's `self.dtype` — bf16 — because every tensor is cast
  // on load (`mlx-gen-minimax-h3/src/vae.rs` `as_dtype(dtype)`), which is why the width is 2 rather
  // than 4. Loading an F32 checkpoint at f32 would be a NO-OP CAST, not a doubling; earlier copy in
  // this repository claimed a "~9.7 GB bf16 VAE loaded at f32 (~19.4 GB)" and that claim is
  // withdrawn — it was wrong in both halves.
  //
  // The AUDIO VAE is the opposite: `mlx-gen-minimax-h3/src/model.rs` hard-codes `Dtype::Float32` for
  // both halves regardless of tier or `Precision`, so its width is 4 on every tier.
  "minimax_h3::vae::mlx+candle": [[2603868984, 2]],
  "minimax_h3_ref::vae::mlx": [[2603868984, 2]],
  // t2va and fl2va decode audio and never encode it, so the base partition takes the 914-tensor
  // decode half (`decoder.*` + `dec_in_proj.*`). ref2va additionally constructs
  // `MiniMaxH3AudioVaeEncoder`, so the reference partition takes all 1087.
  "minimax_h3::audioVae::mlx": [[64939353, 4]],
  "minimax_h3_ref::audioVae::mlx": [[151326585, 4]],
  // The DiT's `DENSE_BY_POLICY` heads — the eight top-level bases the converter's packing predicate
  // excludes by name (`proj_in`, `proj_out`, `audio_proj_in`, `audio_proj_out`,
  // `time_embedder.linear_1`, `time_embedder.linear_2`, `context_embedder`, `norm_out.linear`),
  // 16 tensors with their biases. MIXED WIDTH, the `sensenova_u1_8b_infographic_v3_fast` shape: six
  // of the eight bases are published F32 (17,222,144 elements) and two are BF16 (56,442,624), and
  // `mlx-gen-minimax-h3` loads them through `dit::heads::LinearBias::from_weights`, which takes no
  // dtype parameter at all — a bare `require().clone()` — so they are resident at their PUBLISHED
  // dtype on every tier and are not widened by `Precision::Fp32` either. The offline converter
  // re-emits them verbatim (`quantize_map` casts only inside its packable branch), so a q4 shard
  // holds these 16 tensors byte-identical to upstream. Both partitions are the same architecture and
  // measure identically.
  "minimax_h3::transformerHead::mlx+candle": [[56442624, 2], [17222144, 4]],
  "minimax_h3_ref::transformerHead::mlx": [[56442624, 2], [17222144, 4]],
});

export const VAE_SCOPES = Object.freeze({
  FULL: "full-autoencoder",
  DECODER: "decoder-only",
});

const DECODER_ONLY_VAE_RECEIPT_KEYS = Object.freeze([
  // MiniMax-H3's BASE partition serves t2va + fl2va, neither of which encodes audio, so its audio
  // VAE is the decode half alone. The VIDEO VAE is deliberately NOT here for either partition:
  // MLX `MiniMaxH3VideoVae::load` refuses a snapshot whose encode half is missing, so MLX constructs
  // both halves on every advertised mode. Candle's FL2VA route also constructs both halves, while
  // its T2VA route uses the documented decode-only loader; the full row is the advertised-mode max.
  "minimax_h3::audioVae::mlx",
  "qwen_image::vae::candle",
  "lens::vae::candle",
  "lens_turbo::vae::candle",
  "anima_base::vae::candle",
  "anima_aesthetic::vae::candle",
  "anima_turbo::vae::candle",
  "mage_flow_base::vae::mlx+candle",
  "mage_flow::vae::mlx+candle",
  "mage_flow_turbo::vae::mlx+candle",
]);

const decoderOnlyKeys = new Set(DECODER_ONLY_VAE_RECEIPT_KEYS);
export const VAE_RESIDENCY_SCOPES = Object.freeze({
  ...Object.fromEntries(
    Object.keys(HEADER_MEASUREMENT_TERMS)
      // `::audioVae::` is scoped by the same convention (sc-19506): a joint audio+video family has
      // TWO autoencoders, the uniqueness key is (model, component, lane), and they are held at
      // different widths — so they are two components rather than one summed row, and each carries
      // its own worst-case-by-advertised-mode scope.
      .filter((key) => key.includes("::vae::") || key.includes("::audioVae::"))
      .map((key) => [key, decoderOnlyKeys.has(key) ? VAE_SCOPES.DECODER : VAE_SCOPES.FULL]),
  ),
});
