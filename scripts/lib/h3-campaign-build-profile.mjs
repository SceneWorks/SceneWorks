// sc-26691 — the ONE cargo profile every H3 terminal-campaign build that runs model compute uses,
// on both lanes (MLX and CUDA), so the two backends' timings are comparable and neither renders on
// an unoptimized binary.
//
// The campaign service (`sceneworks-rust-api`, the render path) and the inference-side native
// helpers (witnesses, numeric fixtures, the actual-checkpoint probe) used to build with the default
// `dev` profile: opt-level 0 host code for candle/MLX glue, offload, repacking and IO. Run
// 40779212245 measured that on GPU 1 — a 24 min debug build, 4.6 min DiT load, 4.7 min reference
// VAE/audio encode and ~10.5 min per denoise step.
//
// `release` is the profile the shipped desktop and server binaries already build with
// (`apps/desktop/scripts/build-sidecar.mjs` copies `target/release/sceneworks-rust-api`). The
// workspace does not customize it, so it is cargo's lean default: opt-level 3, no LTO (thin-local
// only), 16 codegen units, no debug info — no fat-LTO / codegen-units=1 link cost to justify a
// separate profile. The source-bound SceneWorks assertion tests (`h3-finetune-terminal-cases.mjs`)
// run no model compute and stay on the ordinary test profile.
export const CAMPAIGN_CARGO_PROFILE = 'release';
/** Directory under `CARGO_TARGET_DIR` holding the profile's binaries. */
export const CAMPAIGN_PROFILE_DIR = 'release';
/** Cargo arguments selecting the campaign profile (`cargo build`/`cargo test`). */
export function campaignProfileArgs() {
  return ['--profile', CAMPAIGN_CARGO_PROFILE];
}
