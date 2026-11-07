//! Intermediate-adapter checkpoint filenames + mid-schedule **resume** (sc-9560 / F-125).
//!
//! The family trainers share the intermediate-adapter filename convention (driven by
//! `config.save_every`); the trained adapter itself is written by the family trainer's `save_*`
//! (PEFT/LoKr safetensors). At each `save_every` a trainer additionally writes a **resume bundle**
//! ([`save_resume`]) — the raw trainable factors + optimizer state ([`TrainOptimizer::save_state`]) +
//! a [`ResumeMeta`] (step / update index / optimizer kind) — so a run interrupted mid-schedule can be
//! continued with [`find_latest_resume`] + [`load_resume`] rather than restarting at step 0. The
//! trainable factors are snapshotted **keyed exactly as the trainer holds them**, so reload is a
//! direct round-trip with no per-family key remapping.
//!
//! **Exactness bound.** The snapshot captures the factors + optimizer state + step/update index — but
//! NOT the in-flight gradient-accumulation buffer. Resume is therefore **bit-exact when the snapshot
//! lands on an optimizer-update boundary**: always for `gradient_accumulation = 1` (the default —
//! validated bit-exact on real SDXL), and for `> 1` when `save_every` is a multiple of it. A snapshot
//! taken mid-accumulation-window drops that window's partial gradients — a bounded drift affecting only
//! the first post-resume update (training still continues correctly, just not bit-identically).
//!
//! **Run identity (opt-in, sc-24163).** [`save_resume_with_identity`] additionally records the run's
//! [`ResumeIdentity`] — its training-config fingerprint and its dataset [`request_fingerprint`]
//! (resolution, item order, captions, image / control / ordered edit-reference paths **and
//! contents**) — and [`load_resume_with_identity`] refuses a bundle whose recorded identity differs
//! from (or is missing for) this run, exactly as candle-gen's resume does: both run gen-core's one
//! [`check_resume_fingerprints`]. [`save_resume`] / [`load_resume`] are unchanged, so trainers that
//! have not opted in keep writing and reading the bundles they always have.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use mlx_rs::Array;

pub use gen_core::train::resume::{
    check_resume_fingerprints, request_fingerprint, training_config_fingerprint,
    REQUEST_FINGERPRINT_KEY, TRAINING_CONFIG_KEY,
};

use crate::train::lora::LoraParams;
use crate::train::optim::TrainOptimizer;
use crate::train::TrainingConfig;
use crate::{Error, Result};

/// `{stem}-step{step:06}.safetensors` — the intermediate adapter checkpoint filename (matches the
/// Python kernel's `save_every` naming). Zero-padded so a lexical sort is a step-order sort.
pub fn checkpoint_filename(stem: &str, step: u32) -> String {
    format!("{stem}-step{step:06}.safetensors")
}

/// The sibling optimizer-state snapshot for the checkpoint at `step`.
pub fn optimizer_state_filename(stem: &str, step: u32) -> String {
    format!("{stem}-step{step:06}.optim.safetensors")
}

/// The resume snapshot (raw trainable factors + [`ResumeMeta`]) for `step`.
pub fn resume_snapshot_filename(stem: &str, step: u32) -> String {
    format!("{stem}-step{step:06}.resume.safetensors")
}

/// Where an interrupted run left off — read from the resume snapshot's safetensors metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResumeMeta {
    /// Micro-step the snapshot was taken at (resume continues at `step + 1`).
    pub step: u32,
    /// Optimizer update index at the snapshot (drives the LR-schedule position on resume).
    pub update_idx: u32,
    /// Optimizer kind tag ([`TrainOptimizer::kind_tag`]) — resume must restore into the same kind.
    pub optimizer: String,
}

/// Write the resume bundle for micro-step `step` into `dir`: the trainable factors `params` (keyed
/// exactly as the trainer holds them, so reload is a direct round-trip) with the [`ResumeMeta`] in the
/// safetensors metadata, plus the optimizer state in a sibling `.optim.safetensors`. Called alongside
/// the user-facing PEFT adapter checkpoint at each `config.save_every`.
pub fn save_resume(
    dir: &Path,
    stem: &str,
    step: u32,
    update_idx: u32,
    opt: &TrainOptimizer,
    params: &LoraParams,
) -> Result<()> {
    save_resume_inner(dir, stem, step, update_idx, opt, params, None)
}

/// What a fingerprinted resume bundle records about the run that wrote it, and what
/// [`load_resume_with_identity`] requires a resuming run to match.
#[derive(Clone, Copy, Debug)]
pub struct ResumeIdentity<'a> {
    /// The run's training config ([`training_config_fingerprint`] of it is recorded).
    pub config: &'a TrainingConfig,
    /// The run's dataset fingerprint ([`request_fingerprint`]).
    pub request_fingerprint: &'a str,
}

/// [`save_resume`] plus the run's [`ResumeIdentity`] in the snapshot metadata (under
/// [`TRAINING_CONFIG_KEY`] / [`REQUEST_FINGERPRINT_KEY`]), for [`load_resume_with_identity`].
pub fn save_resume_with_identity(
    dir: &Path,
    stem: &str,
    step: u32,
    update_idx: u32,
    opt: &TrainOptimizer,
    params: &LoraParams,
    identity: ResumeIdentity<'_>,
) -> Result<()> {
    save_resume_inner(dir, stem, step, update_idx, opt, params, Some(identity))
}

fn save_resume_inner(
    dir: &Path,
    stem: &str,
    step: u32,
    update_idx: u32,
    opt: &TrainOptimizer,
    params: &LoraParams,
    identity: Option<ResumeIdentity<'_>>,
) -> Result<()> {
    opt.save_state(dir.join(optimizer_state_filename(stem, step)))?;

    let entries: Vec<(String, &Array)> = params.iter().map(|(k, v)| (k.to_string(), v)).collect();
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("step".to_string(), step.to_string());
    meta.insert("update_idx".to_string(), update_idx.to_string());
    meta.insert("optimizer".to_string(), opt.kind_tag().to_string());
    if let Some(identity) = identity {
        meta.insert(
            TRAINING_CONFIG_KEY.to_string(),
            training_config_fingerprint(identity.config),
        );
        meta.insert(
            REQUEST_FINGERPRINT_KEY.to_string(),
            identity.request_fingerprint.to_string(),
        );
    }
    Array::save_safetensors(
        entries,
        Some(&meta),
        dir.join(resume_snapshot_filename(stem, step)),
    )?;
    Ok(())
}

/// Load a resume bundle written by [`save_resume`]. `snapshot` is the `.resume.safetensors` file; its
/// sibling `.optim.safetensors` is derived by name. Restores the optimizer state into `opt` (built
/// fresh from the same config; its [`kind_tag`](TrainOptimizer::kind_tag) must match the snapshot) and
/// returns the trainable factors + [`ResumeMeta`].
pub fn load_resume(snapshot: &Path, opt: &mut TrainOptimizer) -> Result<(LoraParams, ResumeMeta)> {
    load_resume_inner(snapshot, opt, None)
}

/// [`load_resume`] for a bundle written by [`save_resume_with_identity`]: refuses — before any
/// optimizer state is restored — a snapshot whose recorded training config or dataset fingerprint
/// differs from `identity`, or that records none (an unfingerprinted bundle's provenance is unknown).
pub fn load_resume_with_identity(
    snapshot: &Path,
    opt: &mut TrainOptimizer,
    identity: ResumeIdentity<'_>,
) -> Result<(LoraParams, ResumeMeta)> {
    load_resume_inner(snapshot, opt, Some(identity))
}

fn load_resume_inner(
    snapshot: &Path,
    opt: &mut TrainOptimizer,
    identity: Option<ResumeIdentity<'_>>,
) -> Result<(LoraParams, ResumeMeta)> {
    let (tensors, meta) = Array::load_safetensors_with_metadata(snapshot)?;
    if let Some(identity) = identity {
        check_resume_fingerprints(&meta, identity.config, identity.request_fingerprint)?;
    }
    let field = |k: &str| -> Result<String> {
        meta.get(k).cloned().ok_or_else(|| {
            Error::Msg(format!(
                "resume: snapshot {snapshot:?} missing metadata '{k}'"
            ))
        })
    };
    let parse = |k: &str| -> Result<u32> {
        field(k)?
            .parse::<u32>()
            .map_err(|e| Error::Msg(format!("resume: metadata '{k}' is not a u32: {e}")))
    };
    let rmeta = ResumeMeta {
        step: parse("step")?,
        update_idx: parse("update_idx")?,
        optimizer: field("optimizer")?,
    };
    if rmeta.optimizer != opt.kind_tag() {
        return Err(Error::Msg(format!(
            "resume: snapshot was saved with optimizer '{}' but this run uses '{}' — start a fresh \
             run or resume with the matching optimizer",
            rmeta.optimizer,
            opt.kind_tag()
        )));
    }
    opt.load_state(optim_sibling(snapshot)?)?;

    let params: LoraParams = tensors
        .into_iter()
        .map(|(k, v)| (Rc::from(k.as_str()), v))
        .collect();
    Ok((params, rmeta))
}

/// The `.optim.safetensors` sibling path of a `.resume.safetensors` snapshot.
fn optim_sibling(snapshot: &Path) -> Result<PathBuf> {
    let name = snapshot
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| Error::Msg(format!("resume: bad snapshot path {snapshot:?}")))?;
    Ok(snapshot.with_file_name(name.replace(".resume.safetensors", ".optim.safetensors")))
}

/// The highest-step resume snapshot for `stem` in `dir` (from a prior interrupted run), or `None`.
/// Scans `{stem}-step{NNNNNN}.resume.safetensors` and returns `(path, step)` for the largest step.
pub fn find_latest_resume(dir: &Path, stem: &str) -> Option<(PathBuf, u32)> {
    let prefix = format!("{stem}-step");
    let mut best: Option<(PathBuf, u32)> = None;
    for e in std::fs::read_dir(dir).ok()?.flatten() {
        let name = e.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(rest) = name.strip_prefix(&prefix) else {
            continue;
        };
        let Some(digits) = rest.strip_suffix(".resume.safetensors") else {
            continue;
        };
        let Ok(step) = digits.parse::<u32>() else {
            continue;
        };
        if best.as_ref().is_none_or(|(_, b)| step > *b) {
            best = Some((e.path(), step));
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_filename_is_zero_padded_and_sortable() {
        assert_eq!(
            checkpoint_filename("my_style", 250),
            "my_style-step000250.safetensors"
        );
        assert_eq!(
            checkpoint_filename("lora", 0),
            "lora-step000000.safetensors"
        );
        // Lexical order == step order.
        assert!(checkpoint_filename("a", 9) < checkpoint_filename("a", 10));
        assert!(checkpoint_filename("a", 999) < checkpoint_filename("a", 1000));
    }

    #[test]
    fn resume_bundle_round_trips_params_meta_and_optimizer_state() {
        // A tiny synthetic training state: a few steps of Prodigy over one factor.
        let mut params: LoraParams = LoraParams::new();
        params.insert(
            Rc::from("blk.lora_A"),
            Array::from_slice(&[0.1, -0.2, 0.3], &[3]),
        );
        let grads: LoraParams = std::iter::once((
            Rc::from("blk.lora_A"),
            Array::from_slice(&[0.4, 0.1, -0.3], &[3]),
        ))
        .collect();
        let mut opt = TrainOptimizer::from_config("prodigy", 1e-3, 0.0).unwrap();
        opt.set_lr_scaled(1.0);
        opt.step(&mut params, &grads).unwrap();
        opt.step(&mut params, &grads).unwrap();

        // Per-process scratch dir — a fixed `$TMPDIR` name races a second concurrent `cargo test`.
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        let stem = "swatch";
        save_resume(&dir, stem, 4, 2, &opt, &params).unwrap();

        // Discovery finds the step-4 snapshot.
        let (found, step) = find_latest_resume(&dir, stem).expect("latest resume");
        assert_eq!(step, 4);

        // Load into a fresh optimizer + compare.
        let mut opt2 = TrainOptimizer::from_config("prodigy", 1e-3, 0.0).unwrap();
        let (loaded, meta) = load_resume(&found, &mut opt2).unwrap();
        assert_eq!(meta.step, 4);
        assert_eq!(meta.update_idx, 2);
        assert_eq!(meta.optimizer, "prodigy");
        assert_eq!(loaded.len(), params.len());
        assert_eq!(
            loaded["blk.lora_A"].as_slice::<f32>(),
            params["blk.lora_A"].as_slice::<f32>()
        );

        // The restored optimizer produces the same next step as the original would.
        let mut a = params.clone();
        let mut b = loaded;
        opt.set_lr_scaled(1.0);
        opt2.set_lr_scaled(1.0);
        opt.step(&mut a, &grads).unwrap();
        opt2.step(&mut b, &grads).unwrap();
        for (x, y) in a["blk.lora_A"]
            .as_slice::<f32>()
            .iter()
            .zip(b["blk.lora_A"].as_slice::<f32>())
        {
            assert!((x - y).abs() <= 1e-6, "resumed step diverged: {x} vs {y}");
        }
    }

    #[test]
    fn load_resume_rejects_optimizer_mismatch() {
        let mut params: LoraParams =
            std::iter::once((Rc::from("blk.lora_A"), Array::from_slice(&[0.1, 0.2], &[2])))
                .collect();
        let grads = params.clone();
        let mut opt = TrainOptimizer::from_config("adamw", 1e-3, 0.0).unwrap();
        opt.set_lr_scaled(1.0);
        opt.step(&mut params, &grads).unwrap();
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        save_resume(&dir, "s", 1, 1, &opt, &params).unwrap();
        let (found, _) = find_latest_resume(&dir, "s").unwrap();

        let mut rose = TrainOptimizer::from_config("rose", 1e-3, 0.0).unwrap();
        let err = load_resume(&found, &mut rose).unwrap_err().to_string();
        assert!(err.contains("optimizer 'adamw'"), "got: {err}");
    }

    /// The wan dual-expert trainer snapshots each expert under a per-expert stem (`{stem}`,
    /// `{stem}-high_noise`, `{stem}-low_noise`) in ONE `output_dir`. `find_latest_resume` must isolate
    /// them: a lookup for the base stem must NOT match a suffixed expert's snapshot (F-125 / sc-9651).
    #[test]
    fn find_latest_resume_isolates_per_expert_stems() {
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        let one = |v: f32| -> LoraParams {
            std::iter::once((Rc::from("blk.lora_A"), Array::from_slice(&[v], &[1]))).collect()
        };
        let opt = TrainOptimizer::from_config("adamw", 1e-3, 0.0).unwrap();

        // A dense stem plus the two MoE expert stems, all at step 40, in the same directory.
        save_resume(&dir, "lora", 40, 20, &opt, &one(1.0)).unwrap();
        save_resume(&dir, "lora-high_noise", 40, 20, &opt, &one(2.0)).unwrap();
        save_resume(&dir, "lora-low_noise", 40, 20, &opt, &one(3.0)).unwrap();

        // Each lookup resolves ONLY its own stem's snapshot (no cross-match on the `-` boundary).
        for (stem, want) in [
            ("lora", 1.0),
            ("lora-high_noise", 2.0),
            ("lora-low_noise", 3.0),
        ] {
            let (found, step) = find_latest_resume(&dir, stem).expect("resume for stem");
            assert_eq!(step, 40);
            let mut o = TrainOptimizer::from_config("adamw", 1e-3, 0.0).unwrap();
            let (loaded, _) = load_resume(&found, &mut o).unwrap();
            assert_eq!(
                loaded["blk.lora_A"].as_slice::<f32>()[0],
                want,
                "stem {stem} loaded the wrong expert's factors"
            );
        }
    }

    /// sc-24163: a fingerprinted bundle resumes only into the same config + dataset; a changed
    /// dataset, a changed config and an unfingerprinted bundle are refused, while the plain
    /// [`load_resume`] still reads a fingerprinted bundle (other trainers are unaffected).
    ///
    /// *Mutation that reds this:* dropping the `check_resume_fingerprints` call in
    /// `load_resume_inner`, or not writing the identity in `save_resume_inner`.
    #[test]
    fn identity_bundles_refuse_a_changed_dataset_or_config() {
        let params: LoraParams =
            std::iter::once((Rc::from("blk.lora_A"), Array::from_slice(&[0.5f32], &[1]))).collect();
        let opt = TrainOptimizer::from_config("adamw", 1e-3, 0.0).unwrap();
        let cfg = TrainingConfig::default();
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path();
        let identity = ResumeIdentity {
            config: &cfg,
            request_fingerprint: "dataset-a",
        };
        save_resume_with_identity(dir, "id", 2, 2, &opt, &params, identity).unwrap();
        save_resume(dir, "plain", 2, 2, &opt, &params).unwrap();
        let (found, _) = find_latest_resume(dir, "id").unwrap();
        let fresh = || TrainOptimizer::from_config("adamw", 1e-3, 0.0).unwrap();

        let (loaded, meta) = load_resume_with_identity(&found, &mut fresh(), identity).unwrap();
        assert_eq!((meta.step, loaded.len()), (2, 1));
        load_resume(&found, &mut fresh()).expect("the plain loader ignores the identity");

        let other = ResumeIdentity {
            request_fingerprint: "dataset-b",
            ..identity
        };
        let err = load_resume_with_identity(&found, &mut fresh(), other)
            .unwrap_err()
            .to_string();
        assert!(err.contains("dataset/request fingerprint differs"), "{err}");

        let changed = TrainingConfig {
            rank: cfg.rank + 1,
            ..cfg.clone()
        };
        let err = load_resume_with_identity(
            &found,
            &mut fresh(),
            ResumeIdentity {
                config: &changed,
                ..identity
            },
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("training configuration differs"), "{err}");

        let (plain, _) = find_latest_resume(dir, "plain").unwrap();
        let err = load_resume_with_identity(&plain, &mut fresh(), identity)
            .unwrap_err()
            .to_string();
        assert!(err.contains("missing training_config"), "{err}");
    }
}
