use super::*;

#[derive(Debug, Clone)]
pub(crate) struct EventMessage {
    pub(crate) event: String,
    pub(crate) data: String,
    pub(crate) revision: u64,
}

#[derive(Debug, Default)]
pub(crate) struct EventHub {
    state: Mutex<EventHubState>,
}

#[derive(Debug, Default)]
struct EventHubState {
    next_subscriber_id: u64,
    revision: u64,
    subscribers: HashMap<u64, mpsc::Sender<EventMessage>>,
}

impl EventHub {
    pub(crate) fn subscribe(&self) -> ReceiverStream<EventMessage> {
        let (sender, receiver) = mpsc::channel(EVENT_BUFFER_SIZE);
        let mut state = self.state.lock();
        let subscriber_id = state.next_subscriber_id;
        state.next_subscriber_id = state.next_subscriber_id.wrapping_add(1);
        state.subscribers.insert(subscriber_id, sender);
        ReceiverStream::new(receiver)
    }

    pub(crate) fn publish(&self, mut message: EventMessage) {
        let mut state = self.state.lock();
        state.revision = state.revision.saturating_add(1);
        message.revision = state.revision;
        state.subscribers.retain(|_, sender| {
            sender
                .try_send(message.clone())
                .map(|_| true)
                .unwrap_or(false)
        });
    }

    pub(crate) fn revision(&self) -> u64 {
        self.state.lock().revision
    }
}

pub(crate) async fn create_event_ticket(
    State(state): State<AppState>,
    request: AxumRequest,
) -> Result<Json<TicketResponse>, ApiError> {
    // This reconnect context is intentionally far smaller than the router-wide
    // 10 MiB JSON allowance. Read with an explicit cap so a chunked body cannot
    // bypass the route-specific memory bound.
    let body = to_bytes(request.into_body(), MAX_EVENT_TICKET_BODY_BYTES)
        .await
        .map_err(|_| {
            ApiError::payload_too_large(format!(
                "event ticket request body exceeds {MAX_EVENT_TICKET_BODY_BYTES} bytes"
            ))
        })?;
    let request = if body.is_empty() {
        CreateEventTicketRequest::default()
    } else {
        serde_json::from_slice::<Option<CreateEventTicketRequest>>(&body)
            .map_err(|error| ApiError {
                status: StatusCode::UNPROCESSABLE_ENTITY,
                detail: format!("JSON decode error: {error}"),
                context: None,
                code: None,
            })?
            .unwrap_or_default()
    };
    validate_event_ticket_ids(
        "activeJobIds",
        &request.active_job_ids,
        MAX_EVENT_TICKET_ACTIVE_JOB_IDS,
    )?;
    validate_event_ticket_ids(
        "knownTerminalJobIds",
        &request.known_terminal_job_ids,
        MAX_EVENT_TICKET_TERMINAL_JOB_IDS,
    )?;
    let context_bytes = request
        .active_job_ids
        .iter()
        .chain(&request.known_terminal_job_ids)
        .try_fold(0usize, |total, id| total.checked_add(id.len()))
        .unwrap_or(usize::MAX);
    if context_bytes > MAX_EVENT_TICKET_CONTEXT_BYTES {
        return Err(ApiError::payload_too_large(format!(
            "event ticket job-id context exceeds {MAX_EVENT_TICKET_CONTEXT_BYTES} bytes"
        )));
    }
    let context = EventTicketContext {
        active_job_ids: dedupe_ids(request.active_job_ids),
        known_terminal_job_ids: dedupe_ids(request.known_terminal_job_ids),
    };
    state
        .event_tickets
        .try_issue_event(context)
        .map(Json)
        .ok_or_else(|| ApiError {
            status: StatusCode::TOO_MANY_REQUESTS,
            detail: format!(
                "Too many outstanding event tickets; retry after at most \
                 {EVENT_TICKET_TTL_SECONDS} seconds"
            ),
            context: None,
            code: None,
        })
}

fn validate_event_ticket_ids(
    field: &str,
    ids: &[String],
    max_count: usize,
) -> Result<(), ApiError> {
    if ids.len() > max_count {
        return Err(ApiError::payload_too_large(format!(
            "{field} may contain at most {max_count} job ids"
        )));
    }
    if ids.iter().any(|id| id.trim().is_empty()) {
        return Err(ApiError::bad_request(format!(
            "{field} must contain non-empty job ids"
        )));
    }
    if ids
        .iter()
        .any(|id| id.len() > MAX_EVENT_TICKET_JOB_ID_BYTES)
    {
        return Err(ApiError::payload_too_large(format!(
            "{field} job ids may contain at most {MAX_EVENT_TICKET_JOB_ID_BYTES} bytes"
        )));
    }
    Ok(())
}

fn dedupe_ids(ids: Vec<String>) -> Vec<String> {
    let mut seen = std::collections::HashSet::with_capacity(ids.len());
    ids.into_iter()
        .filter(|id| seen.insert(id.clone()))
        .collect()
}

// sc-8947 (F-146, Severity: Info — accepted control, recorded): the SSE stream is
// gated by a `?ticket=` query param instead of the normal `X-SceneWorks-Token`
// header because the browser `EventSource` API cannot set request headers. Putting
// the *access token* in a URL would be a real leak (URLs land in history, logs,
// Referer), so the token is NEVER used here — the client mints a throwaway ticket
// and passes that instead. The ticket's exposure is bounded by two controls, and
// this is the deliberate design; do NOT extend the query-string pattern to the
// long-lived token:
//   1. single-use — `consume()` removes the ticket on first redemption whether or
//      not it was valid, so a leaked URL can be replayed at most zero times after
//      the legitimate `EventSource` connects.
//   2. short TTL — `EVENT_TICKET_TTL_SECONDS` (30s), so an unredeemed leak dies fast.
//
// The suggested optional hardening — pinning the ticket to the issuing peer IP — is
// deliberately NOT implemented. Ticket issuance (`POST /jobs/events/ticket`, a
// `fetch`) and redemption (`GET /jobs/events`, the `EventSource`) are two separate
// connections, and the web client mints a fresh ticket on every (re)connect
// (App.jsx `connect()`), including each auto-reconnect. On a dual-stack loopback
// host `localhost` can resolve to `127.0.0.1` for one request and `::1` for the
// other, so pinning the ticket to the issuer's IP would reject the redemption and
// break legitimate SSE in exactly the loopback/LAN setup epic 4484 targets (and it
// adds nothing behind a reverse proxy, where every peer already shares one IP). The
// single-use + short-TTL pair is the accepted control.
pub(crate) async fn job_events(
    State(state): State<AppState>,
    Query(query): Query<EventsQuery>,
) -> Result<Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>>, ApiError> {
    let reconnect_context = match query.ticket.as_deref() {
        Some(ticket) => state
            .event_tickets
            .consume_event(ticket)
            .ok_or_else(|| ApiError::unauthorized("Invalid or expired event stream ticket"))?,
        None if state.settings.access_token.is_empty() => EventTicketContext::default(),
        None => {
            return Err(ApiError::unauthorized(
                "Invalid or expired event stream ticket",
            ))
        }
    };
    // Serialize the snapshot read and subscription with every queue publication.
    // A concurrent mutation is therefore either visible in these authoritative
    // snapshots or published after the subscription is installed. Subscribing
    // before an unlocked read is not sufficient: an event buffered during that
    // read could otherwise replay after (and overwrite) the newer snapshot.
    let _snapshot_guard = state.queue_snapshot_lock.lock().await;
    // Install the subscription while queue publishers are excluded. Job-only
    // progress ticks may still publish during the reads below; buffering them
    // prevents a missed job update, and the client rejects an older buffered
    // row by updatedAt after applying the snapshot.
    let messages = state.events.subscribe();
    let queue = queue_summary_snapshot(state.clone()).await?;
    // Events published before this barrier are represented by the authoritative
    // job snapshot read below. Clients discard those buffered revisions after
    // applying the snapshot, avoiding equal-second updatedAt ambiguity.
    let snapshot_revision = state.events.revision();
    let mut jobs = store_call(state.clone(), |store, _timeout| {
        store.list_jobs_recently_updated(500)
    })
    .await?;
    // The 500-row list is display history, not the correctness boundary. A
    // reconnecting client names its exact formerly-active set so each of those
    // rows is authoritative even after more than 500 later mutations.
    let mut known_job_ids = reconnect_context.active_job_ids.clone();
    known_job_ids.extend(reconnect_context.known_terminal_job_ids);
    let known_job_ids = dedupe_ids(known_job_ids);
    let requested_jobs = store_call(state.clone(), move |store, _timeout| {
        store.list_existing_jobs_by_ids(&reconnect_context.active_job_ids)
    })
    .await?;
    let cleared_job_ids = store_call(state.clone(), move |store, _timeout| {
        store.cleared_job_ids_by_ids(&known_job_ids)
    })
    .await?;
    let mut job_ids = jobs
        .iter()
        .map(|job| job.id.clone())
        .collect::<std::collections::HashSet<_>>();
    for job in &queue.active_jobs {
        if job_ids.insert(job.id.clone()) {
            jobs.push(job.clone());
        }
    }
    for job in requested_jobs {
        if job_ids.insert(job.id.clone()) {
            jobs.push(job);
        }
    }
    // Clear tombstones are authoritative even if a future store regression
    // accidentally returns a soft-hidden row through one of the composed lists.
    let cleared_job_id_set = cleared_job_ids
        .iter()
        .collect::<std::collections::HashSet<_>>();
    jobs.retain(|job| !cleared_job_id_set.contains(&job.id));
    jobs.sort_by(|left, right| right.created_at.cmp(&left.created_at));
    let jobs = public_job_snapshots(jobs);
    let initial_jobs = EventMessage {
        event: "jobs.snapshot".to_owned(),
        data: serde_json::to_string(&json!({
            "revision": snapshot_revision,
            "jobs": jobs,
            "clearedJobIds": cleared_job_ids,
        }))
        .map_err(|error| ApiError::internal(format!("Could not serialize jobs: {error}")))?,
        revision: snapshot_revision,
    };
    let initial_queue = EventMessage {
        event: "queue.updated".to_owned(),
        data: serde_json::to_string(&queue)
            .map_err(|error| ApiError::internal(format!("Could not serialize queue: {error}")))?,
        revision: snapshot_revision,
    };
    #[cfg(test)]
    let snapshot_barrier = state.sse_snapshot_before_subscribe_once.lock().take();
    #[cfg(test)]
    if let Some(barrier) = snapshot_barrier {
        barrier.wait().await;
        barrier.wait().await;
    }
    drop(_snapshot_guard);
    Ok(Sse::new(sse_event_stream(
        messages,
        vec![initial_jobs, initial_queue],
    )))
}

fn sse_event_stream(
    messages: ReceiverStream<EventMessage>,
    initial_messages: Vec<EventMessage>,
) -> impl futures_util::Stream<Item = Result<Event, Infallible>> {
    let mut heartbeat = tokio::time::interval_at(
        TokioInstant::now() + Duration::from_secs(15),
        Duration::from_secs(15),
    );
    heartbeat.set_missed_tick_behavior(MissedTickBehavior::Delay);
    futures_util::stream::unfold(
        (messages, heartbeat, true, initial_messages.into_iter()),
        |(mut messages, mut heartbeat, send_ready, mut initial_messages)| async move {
            if send_ready {
                return Some((
                    Ok(ready_event()),
                    (messages, heartbeat, false, initial_messages),
                ));
            }
            if let Some(initial_message) = initial_messages.next() {
                return Some((
                    Ok(sse_message_event(initial_message)),
                    (messages, heartbeat, false, initial_messages),
                ));
            }
            tokio::select! {
                message = messages.next() => {
                    message.map(|message| {
                        (
                            Ok(sse_message_event(message)),
                            (messages, heartbeat, false, initial_messages),
                        )
                    })
                }
                _ = heartbeat.tick() => {
                    Some((Ok(heartbeat_event()), (messages, heartbeat, false, initial_messages)))
                }
            }
        },
    )
}

fn ready_event() -> Event {
    Event::default()
        .event("ready")
        .data(json!({ "status": "connected" }).to_string())
}

fn sse_message_event(message: EventMessage) -> Event {
    Event::default()
        .event(message.event)
        .id(message.revision.to_string())
        .data(message.data)
}

fn heartbeat_event() -> Event {
    Event::default().event("heartbeat").data(HEARTBEAT_SSE_DATA)
}
