use super::{send_gen_progress, GenEvent};
use gen_core::{LoadPhase, Progress};

#[cfg(test)]
#[test]
fn conditioning_complete_stays_internal_and_preserves_first_positive_step() {
    let (tx, mut rx) = tokio::sync::mpsc::channel(4);
    send_gen_progress(&tx, 2, Progress::ConditioningComplete);
    assert!(matches!(
        rx.try_recv(),
        Err(tokio::sync::mpsc::error::TryRecvError::Empty)
    ));
    send_gen_progress(
        &tx,
        2,
        Progress::Step {
            current: 1,
            total: 4,
        },
    );
    assert!(matches!(
        rx.try_recv(),
        Ok(GenEvent::Step {
            index: 2,
            current: 1,
            total: 4
        })
    ));
    assert!(matches!(
        rx.try_recv(),
        Err(tokio::sync::mpsc::error::TryRecvError::Empty)
    ));
}

#[cfg(test)]
#[test]
fn prompt_encoding_complete_stays_internal_and_preserves_first_positive_step() {
    let (tx, mut rx) = tokio::sync::mpsc::channel(4);
    send_gen_progress(&tx, 2, Progress::PromptEncodingComplete);
    send_gen_progress(&tx, 2, Progress::PostProcessing);
    assert!(matches!(
        rx.try_recv(),
        Err(tokio::sync::mpsc::error::TryRecvError::Empty)
    ));
    send_gen_progress(
        &tx,
        2,
        Progress::Step {
            current: 1,
            total: 4,
        },
    );
    assert!(matches!(
        rx.try_recv(),
        Ok(GenEvent::Step {
            index: 2,
            current: 1,
            total: 4
        })
    ));
    assert!(matches!(
        rx.try_recv(),
        Err(tokio::sync::mpsc::error::TryRecvError::Empty)
    ));
}

/// Incremental prompt-upsampling token counts DO reach the FLUX 3 surface; only the completion
/// boundary stays internal (no event, no synthesized step).
#[cfg(test)]
#[test]
fn prompt_upsampling_tokens_are_emitted_but_completion_stays_internal() {
    let (tx, mut rx) = tokio::sync::mpsc::channel(4);
    send_gen_progress(
        &tx,
        0,
        Progress::Upsampling {
            tokens: 3,
            max_tokens: 8,
        },
    );
    assert!(matches!(
        rx.try_recv(),
        Ok(GenEvent::Upsampling {
            index: 0,
            tokens: 3,
            max_tokens: 8
        })
    ));
    send_gen_progress(&tx, 0, Progress::PromptUpsamplingComplete);
    send_gen_progress(&tx, 0, Progress::PromptEncodingComplete);
    send_gen_progress(&tx, 0, Progress::ConditioningComplete);
    assert!(matches!(
        rx.try_recv(),
        Err(tokio::sync::mpsc::error::TryRecvError::Empty)
    ));
    send_gen_progress(&tx, 0, Progress::Loading(LoadPhase::EncodingPrompt));
    assert!(matches!(
        rx.try_recv(),
        Ok(GenEvent::Loading {
            index: 0,
            phase: LoadPhase::EncodingPrompt
        })
    ));
}
