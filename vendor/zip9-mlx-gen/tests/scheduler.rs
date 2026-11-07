//! sc-2350: flow-match Euler scheduler parity vs the Python mflux fork.
//!
//! Fixture `tests/fixtures/scheduler.safetensors` ← `tools/dump_scheduler.py`, whose goldens
//! come from the fork's own `_compute_empirical_mu` + `_time_shift_exponential_array`. The Rust
//! port recomputes the sigmas (and one Euler step) independently and must agree.

use mlx_gen::scheduler::{compute_mu, image_seq_len, FlowMatchEuler};
use mlx_gen::weights::Weights;
use mlx_rs::ops::all_close;
use mlx_rs::Array;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/scheduler.safetensors"
);

fn close(a: &Array, b: &Array) -> bool {
    all_close(a, b, 1e-4, 1e-4, false).unwrap().item::<bool>()
}

/// Parse a `cfg_i` / `step` metadata entry: "num_steps,width,height,seq_len,mu".
fn parse_cfg(s: &str) -> (usize, u32, u32, usize, f32) {
    let p: Vec<&str> = s.split(',').collect();
    (
        p[0].parse().unwrap(),
        p[1].parse().unwrap(),
        p[2].parse().unwrap(),
        p[3].parse().unwrap(),
        p[4].parse().unwrap(),
    )
}

#[test]
fn sigmas_match_fork() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let n: usize = w.metadata("num_cfgs").unwrap().parse().unwrap();

    for i in 0..n {
        let (steps, width, height, seq_len, mu) =
            parse_cfg(w.metadata(&format!("cfg_{i}")).unwrap());

        // image_seq_len + empirical mu reproduce the fork's values.
        assert_eq!(image_seq_len(width, height), seq_len, "cfg_{i}: seq_len");
        assert!(
            (compute_mu(seq_len, steps) - mu).abs() < 1e-4,
            "cfg_{i}: mu {} vs fork {mu}",
            compute_mu(seq_len, steps)
        );

        // The full sigma schedule matches.
        let mine = FlowMatchEuler::for_image(steps, width, height).sigmas;
        let mine = Array::from_slice(&mine, &[mine.len() as i32]);
        let golden = w.require(&format!("sigmas_{i}")).unwrap();
        assert!(close(&mine, golden), "cfg_{i}: sigma schedule diverged");
    }
}

#[test]
fn euler_step_matches_fork() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let (steps, width, height, _seq, _mu) = parse_cfg(w.metadata("step").unwrap());
    let t = 1;

    let sched = FlowMatchEuler::for_image(steps, width, height);
    let latents = w.require("step_latents").unwrap();
    let noise = w.require("step_noise").unwrap();
    let out = sched.step(latents, noise, t).unwrap();

    assert!(
        close(&out, w.require("step_out").unwrap()),
        "Euler step x + (sigma[t+1]-sigma[t])*v diverged from fork"
    );
}
