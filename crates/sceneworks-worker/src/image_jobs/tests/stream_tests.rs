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

/// Codex P2 on PR #3434: the Candle ControlNet / IP-Adapter / PuLID routes use the uncached
/// blocking producer. When a progress or heartbeat POST failure makes the consumer abandon it, a
/// later `ProcessPoisoned` reaches nobody — so the producer must latch the fault process-wide when
/// it classifies it, with no consumer and no cache boundary involved.
#[cfg(native_lane)]
#[tokio::test]
async fn an_abandoned_uncached_producer_latches_its_process_fault() {
    const DETAIL: &str = "uncached producer device lost (abandoned consumer)";
    let (cancel, rx, handle) = super::start_gen_stream(
        "job-abandoned".to_owned(),
        "fixture",
        0,
        || Ok(()),
        |(), _tx, _cancel| {
            Err(crate::generation_error(
                "fixture generation failed",
                gen_core::Error::ProcessPoisoned(DETAIL.to_owned()),
            ))
        },
    );
    // The consumer is gone: nobody reads events or inspects the producer's result.
    drop(rx);
    cancel.cancel();
    let _ = handle.await;
    assert!(
        crate::error::test_latched_process_faults()
            .lock()
            .unwrap()
            .iter()
            .any(|detail| detail == DETAIL),
        "the producer latched its typed fault without any consumer"
    );
}

/// SW-7171: once a typed fault is latched — here classified and then swallowed by the item closure,
/// as the old non-fatal scoring seam did — the multi-item driver starts no further pose or image and
/// fails the job with that fatal cause.
#[test]
fn a_latched_fault_stops_the_driver_before_the_next_item() {
    std::thread::spawn(|| {
        let (tx, _rx) = tokio::sync::mpsc::channel(8);
        let mut started = 0_usize;
        let result = super::drive_gen_items_scored(tx, [1_u8, 2, 3], |_, _, _, _| {
            started += 1;
            let _swallowed = crate::classify_engine_error(
                "face detect",
                gen_core::Error::ProcessPoisoned("driver device lost".into()),
            );
            Ok(Some((0, 1, 1, vec![0; 3], None)))
        });
        assert!(
            matches!(result, Err(crate::WorkerError::ProcessPoisoned(ref d)) if d == "driver device lost"),
            "{result:?}"
        );
        assert_eq!(started, 1, "no further item starts after the fault");
    })
    .join()
    .expect("driver thread");
}

/// SW-7171 AC1 end to end at the driver: a poisoned face score inside a pose fails the set with
/// the fatal cause and the next pose never starts.
#[cfg(native_lane)]
#[test]
fn a_poisoned_face_score_fails_the_set_and_starts_no_further_pose() {
    std::thread::spawn(|| {
        let image = |px: u8| gen_core::Image {
            width: 1,
            height: 1,
            pixels: vec![px; 3],
        };
        let (scorer, _calls) =
            crate::face_likeness::FaceLikenessScorer::with_stub_source(&image(200));
        let (tx, _rx) = tokio::sync::mpsc::channel(8);
        let mut started = 0_usize;
        let result = super::drive_gen_items_scored(tx, [255_u8, 200], |_, px, _, _| {
            started += 1;
            let face_likeness =
                crate::face_likeness::score_generated_image(Some(&scorer), &image(px), Some("s"))?;
            Ok(Some((0, 1, 1, vec![px; 3], face_likeness)))
        });
        assert!(
            matches!(result, Err(crate::WorkerError::ProcessPoisoned(_))),
            "{result:?}"
        );
        assert_eq!(started, 1);
    })
    .join()
    .expect("driver thread");
}
