//! sc-4881 signal attribution + sc-5567 job-kind-aware OOM remediation: a signal-9
//! (SIGKILL/OOM) death must give guidance that fits the dead job — count/resolution
//! for an image batch, frames for video, gradient checkpointing only for training —
//! and non-OOM uncatchable deaths must keep naming their real cause. sc-6320: a
//! non-signal non-zero exit (a self-terminated process / panic) names the exit code.
use super::{termination_failure_error, JobType};

#[test]
fn signal_9_image_batch_points_at_count_not_gradient_checkpointing() {
    let msg = termination_failure_error(Some(9), None, Some(&JobType::ImageGenerate));
    assert!(msg.contains("signal 9 (SIGKILL)"), "{msg}");
    assert!(msg.contains("out-of-memory"), "{msg}");
    assert!(msg.contains("image count or resolution"), "{msg}");
    // The old training-only hint must NOT leak onto an image batch (the sc-5567 bug).
    assert!(!msg.contains("Gradient Checkpointing"), "{msg}");
    assert!(!msg.contains("training step"), "{msg}");
}

#[test]
fn signal_9_training_keeps_gradient_checkpointing_hint() {
    let msg = termination_failure_error(Some(9), None, Some(&JobType::LoraTrain));
    assert!(msg.contains("Gradient Checkpointing"), "{msg}");
    assert!(msg.contains("training step"), "{msg}");
}

#[test]
fn signal_9_video_points_at_frame_count() {
    let msg = termination_failure_error(Some(9), None, Some(&JobType::VideoGenerate));
    assert!(msg.contains("out-of-memory"), "{msg}");
    assert!(msg.contains("frame count"), "{msg}");
    assert!(!msg.contains("Gradient Checkpointing"), "{msg}");
}

#[test]
fn signal_9_unknown_and_idle_fall_back_to_generic_oom() {
    // No active job (worker died idle) and an unmapped job kind both get the generic
    // OOM hint rather than a misleading training/image/video-specific one.
    for job_type in [None, Some(&JobType::Unknown("future".to_owned()))] {
        let msg = termination_failure_error(Some(9), None, job_type);
        assert!(msg.contains("out-of-memory"), "{msg}");
        assert!(!msg.contains("Gradient Checkpointing"), "{msg}");
        assert!(!msg.contains("image count"), "{msg}");
        assert!(!msg.contains("frame count"), "{msg}");
    }
}

#[test]
fn non_oom_signals_keep_their_own_cause_regardless_of_job_kind() {
    // SIGABRT / SIGSEGV are not OOM, so the job kind must not turn them into one.
    let abort = termination_failure_error(Some(6), None, Some(&JobType::ImageGenerate));
    assert!(abort.contains("signal 6 (SIGABRT)"), "{abort}");
    assert!(abort.contains("GPU/Metal command-buffer abort"), "{abort}");
    assert!(!abort.contains("out-of-memory"), "{abort}");

    let segv = termination_failure_error(Some(11), None, Some(&JobType::LoraTrain));
    assert!(segv.contains("signal 11 (SIGSEGV)"), "{segv}");
    assert!(segv.contains("segmentation fault"), "{segv}");
    assert!(!segv.contains("Gradient Checkpointing"), "{segv}");
}

#[test]
fn panic_exit_code_101_self_names_without_claiming_a_signal() {
    // sc-6320: a Rust panic unwinds to exit 101 (no signal). The attribution must
    // name the panic + code and must NOT fabricate a signal or an OOM hint.
    let msg = termination_failure_error(None, Some(101), Some(&JobType::ImageGenerate));
    assert!(msg.contains("panicked"), "{msg}");
    assert!(msg.contains("101"), "{msg}");
    assert!(!msg.contains("signal"), "{msg}");
    assert!(!msg.contains("out-of-memory"), "{msg}");
}

#[test]
fn other_non_zero_exit_reports_the_raw_code() {
    // A non-zero, non-101 self-exit reports the raw code so the cause is greppable.
    let msg = termination_failure_error(None, Some(2), None);
    assert!(msg.contains("exited unexpectedly (code 2)"), "{msg}");
    assert!(!msg.contains("signal"), "{msg}");
}

#[test]
fn signal_takes_precedence_when_both_are_present() {
    // Defensive: if both somehow arrive, the signal (the harder cause) wins.
    let msg = termination_failure_error(Some(11), Some(101), None);
    assert!(msg.contains("signal 11 (SIGSEGV)"), "{msg}");
    assert!(!msg.contains("101"), "{msg}");
}
