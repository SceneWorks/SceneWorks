//! Shared task/stream supervision. Lane handlers retain their event and progress semantics.
use super::*;
use std::future::Future;
use std::pin::Pin;

pub(crate) enum StreamSignal<E> {
    Event(E),
    ProgressTick,
    CancelRequested,
    Finished,
}

/// Own the producer until it has actually stopped. Cancellation only trips the flag; the caller
/// writes its lane-specific terminal acknowledgment after `finish` joins. Errors also cancel and
/// join, so neither a processing lease nor a worker slot is released around a detached producer.
pub(crate) struct StreamSupervisor<C: CancelHandle, F: Future> {
    cancel: C,
    task: Pin<Box<F>>,
    result: Option<F::Output>,
    heartbeat: tokio::time::Interval,
    progress: Option<tokio::time::Interval>,
    stream_closed: bool,
    canceled: bool,
    finished: bool,
}

impl<C: CancelHandle, F: Future> StreamSupervisor<C, F> {
    pub(crate) fn new(cancel: C, task: F, heartbeat: Duration, progress: Option<Duration>) -> Self {
        let interval = |period| {
            let mut interval = tokio::time::interval(period);
            interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
            interval
        };
        Self {
            cancel,
            task: Box::pin(task),
            result: None,
            heartbeat: interval(heartbeat),
            progress: progress.map(interval),
            stream_closed: false,
            canceled: false,
            finished: false,
        }
    }

    pub(crate) fn request_cancel(&mut self) {
        self.cancel.cancel();
        self.canceled = true;
    }

    pub(crate) async fn next<E>(
        &mut self,
        api: &ApiClient,
        settings: &Settings,
        job_id: &str,
        mut rx: Option<&mut tokio::sync::mpsc::Receiver<E>>,
    ) -> WorkerResult<StreamSignal<E>> {
        loop {
            if self.result.is_some() && (rx.is_none() || self.stream_closed) {
                return Ok(StreamSignal::Finished);
            }
            tokio::select! {
                result = &mut self.task, if self.result.is_none() => {
                    self.result = Some(result);
                }
                event = async { rx.as_mut().expect("enabled receiver").recv().await },
                    if rx.is_some() && !self.stream_closed => {
                    match event {
                        Some(event) => return Ok(StreamSignal::Event(event)),
                        None => self.stream_closed = true,
                    }
                }
                _ = async { self.progress.as_mut().expect("enabled progress timer").tick().await },
                    if self.progress.is_some() => return Ok(StreamSignal::ProgressTick),
                _ = self.heartbeat.tick() => {
                    heartbeat(api, settings, WorkerStatus::Busy, Some(job_id)).await?;
                    if !self.canceled && self.result.is_none()
                        && (crate::progress::shutdown_requested()
                            || crate::shutdown_requested()
                            || cancel_requested_peek(api, job_id).await)
                    {
                        // The task may have finished while the heartbeat/cancel HTTP requests
                        // were in flight. Prefer that completed result over a late cancel.
                        if let std::task::Poll::Ready(result) = futures_util::poll!(&mut self.task) {
                            self.result = Some(result);
                            continue;
                        }
                        self.request_cancel();
                        return Ok(StreamSignal::CancelRequested);
                    }
                }
            }
        }
    }

    /// Supervise a task with no event stream or lane-specific progress hook.
    pub(crate) async fn wait(
        mut self,
        api: &ApiClient,
        settings: &Settings,
        job_id: &str,
    ) -> WorkerResult<(bool, F::Output)> {
        let handling = async {
            loop {
                if matches!(
                    self.next::<()>(api, settings, job_id, None).await?,
                    StreamSignal::Finished
                ) {
                    break;
                }
            }
            Ok(())
        }
        .await;
        self.finish(handling).await
    }

    /// Stop accepting stream events before joining after handler failure. Closing the receiver
    /// wakes a native producer blocked in `blocking_send`, so cancellation cannot deadlock teardown.
    #[cfg(any(test, target_os = "macos", feature = "backend-candle"))]
    pub(crate) async fn finish_stream<E>(
        self,
        handling: WorkerResult<()>,
        rx: &mut tokio::sync::mpsc::Receiver<E>,
    ) -> WorkerResult<(bool, F::Output)> {
        rx.close();
        self.finish(handling).await
    }

    /// Always await the producer, including after a failed event/progress/heartbeat handler. Unlike
    /// dropping or aborting a spawn_blocking handle this establishes that native work has stopped.
    pub(crate) async fn finish(
        mut self,
        handling: WorkerResult<()>,
    ) -> WorkerResult<(bool, F::Output)> {
        if handling.is_err() {
            self.request_cancel();
        }
        let result = match self.result.take() {
            Some(result) => result,
            None => self.task.as_mut().await,
        };
        self.finished = true;
        handling?;
        Ok((self.canceled, result))
    }
}

impl<C: CancelHandle, F: Future> Drop for StreamSupervisor<C, F> {
    fn drop(&mut self) {
        if !self.finished {
            self.cancel.cancel();
        }
    }
}
