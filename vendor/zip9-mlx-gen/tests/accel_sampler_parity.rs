//! sc-2769: scheduler-math isolation parity for the few-step acceleration samplers
//! (`LcmSampler` / `LightningSampler` / `TcdSampler`) vs **diffusers**.
//!
//! `#[ignore]`d — needs the golden from `tools/dump_sdxl_accel_golden.py` (gitignored, regenerable):
//!   /Users/michael/Repos/mflux/.venv/bin/python3 tools/dump_sdxl_accel_golden.py
//!   cargo test -p mlx-gen --test integration accel_sampler_parity:: -- --ignored --nocapture
//!
//! Each scheduler is fed the SAME synthetic `(model_output, sample)` tensors diffusers saw at each
//! step; the Rust DETERMINISTIC output must match diffusers' to ~1e-5 (torch-f32 vs MLX-f32). This
//! validates the new scheduler math free of any U-Net-backend confound (the between-step re-noise
//! draws from a different RNG than torch, so only the deterministic core is compared — see the
//! `denoised`/`pred_noised` methods). The end-to-end render parity (which DOES exercise the U-Net) is
//! a separate, qualitative gate.

use mlx_gen::sampler::{AlphaSchedule, DiffusionSampler, LcmSampler, LightningSampler, TcdSampler};
use mlx_gen::weights::Weights;
use mlx_rs::{Array, Dtype};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tools/golden/sdxl_accel_sched_golden.safetensors"
);

// SDXL noise schedule.
const N_TRAIN: usize = 1000;
const BETA_START: f32 = 0.00085;
const BETA_END: f32 = 0.012;
const ORIGINAL_STEPS: usize = 50;

fn peak_rel(a: &Array, b: &Array) -> f32 {
    let n = b.shape().iter().product::<i32>();
    let a = a.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-6);
    a.iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()))
        / peak
}

fn sched() -> AlphaSchedule {
    AlphaSchedule::scaled_linear(N_TRAIN, BETA_START, BETA_END)
}

#[test]
#[ignore = "needs tools/golden/sdxl_accel_sched_golden.safetensors (dump_sdxl_accel_golden.py)"]
fn alphas_cumprod_matches_diffusers() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let s = sched();
    let ours = Array::from_slice(&s.alphas_cumprod, &[N_TRAIN as i32]);
    let pr = peak_rel(&ours, g.require("alphas_cumprod").unwrap());
    println!("alphas_cumprod peak_rel = {pr:.3e}");
    assert!(
        pr < 1e-5,
        "alphas_cumprod diverged from diffusers: {pr:.3e}"
    );
}

/// Drive every dumped config through its Rust sampler and compare the per-step deterministic output.
#[test]
#[ignore = "needs tools/golden/sdxl_accel_sched_golden.safetensors (dump_sdxl_accel_golden.py)"]
fn accel_schedulers_match_diffusers() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let configs: Vec<&str> = g.metadata("configs").unwrap().split(',').collect();
    let mut worst = 0f32;

    for cfg in configs {
        let num_steps: usize = cfg.rsplit('_').next().unwrap().parse().unwrap();
        // Compare the timestep schedule first (a wrong schedule shifts every step).
        let want_ts = g.require(&format!("{cfg}.timesteps")).unwrap();
        let want_ts = want_ts.as_slice::<f32>();

        // Build the matching sampler (f32 for the isolation comparison).
        enum S {
            Lcm(LcmSampler),
            Light(LightningSampler),
            Tcd(TcdSampler),
        }
        let s = if cfg.starts_with("lcm_") {
            S::Lcm(LcmSampler::new(
                sched(),
                N_TRAIN,
                ORIGINAL_STEPS,
                num_steps,
                Dtype::Float32,
                0,
            ))
        } else if cfg.starts_with("lightning_") {
            S::Light(LightningSampler::new(
                &sched(),
                N_TRAIN,
                num_steps,
                Dtype::Float32,
            ))
        } else if cfg.starts_with("tcd_eta0_") {
            S::Tcd(TcdSampler::new(
                sched(),
                N_TRAIN,
                ORIGINAL_STEPS,
                num_steps,
                0.0,
                Dtype::Float32,
                0,
            ))
        } else if cfg.starts_with("tcd_eta03_") {
            S::Tcd(TcdSampler::new(
                sched(),
                N_TRAIN,
                ORIGINAL_STEPS,
                num_steps,
                0.3,
                Dtype::Float32,
                0,
            ))
        } else {
            panic!("unknown config {cfg}");
        };

        let timestep_of = |i: usize| -> f32 {
            match &s {
                S::Lcm(x) => x.timestep(i),
                S::Light(x) => x.timestep(i),
                S::Tcd(x) => x.timestep(i),
            }
        };
        for (i, &want) in want_ts.iter().enumerate().take(num_steps) {
            assert!(
                (timestep_of(i) - want).abs() < 1e-3,
                "{cfg}: timestep[{i}] {} != diffusers {want}",
                timestep_of(i),
            );
        }

        for i in 0..num_steps {
            let eps = g.require(&format!("{cfg}.eps{i}")).unwrap().clone();
            let x = g.require(&format!("{cfg}.x{i}")).unwrap().clone();
            let want = g.require(&format!("{cfg}.det{i}")).unwrap();
            let got = match &s {
                S::Lcm(lcm) => lcm.denoised(&eps, &x, i).unwrap(),
                // Lightning's full step is deterministic → compare prev_sample directly. Also check
                // scale_model_input matches.
                S::Light(light) => {
                    let scaled_want = g.require(&format!("{cfg}.scaled{i}")).unwrap();
                    let scaled_got = light.scale_model_input(&x, i).unwrap();
                    let spr = peak_rel(&scaled_got, scaled_want);
                    assert!(
                        spr < 1e-4,
                        "{cfg}: scale_model_input[{i}] peak_rel {spr:.3e}"
                    );
                    worst = worst.max(spr);
                    light.step(&eps, &x, i).unwrap()
                }
                S::Tcd(tcd) => tcd.pred_noised(&eps, &x, i).unwrap(),
            };
            let pr = peak_rel(&got, want);
            worst = worst.max(pr);
            assert!(
                pr < 1e-4,
                "{cfg}: step[{i}] deterministic output peak_rel {pr:.3e}"
            );
        }
        println!("✓ {cfg}: {num_steps} steps match diffusers");
    }
    println!("✓ all acceleration schedulers match diffusers (worst peak_rel {worst:.3e})");
}
