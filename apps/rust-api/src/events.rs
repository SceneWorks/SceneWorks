use super::*;

#[derive(Debug, Clone)]
pub(crate) struct EventMessage {
    pub(crate) event: String,
    pub(crate) data: String,
}

#[derive(Debug, Default)]
pub(crate) struct EventHub {
    state: Mutex<EventHubState>,
}

#[derive(Debug, Default)]
struct EventHubState {
    next_subscriber_id: u64,
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

    pub(crate) fn publish(&self, message: EventMessage) {
        let mut state = self.state.lock();
        state.subscribers.retain(|_, sender| {
            sender
                .try_send(message.clone())
                .map(|_| true)
                .unwrap_or(false)
        });
    }
}

pub(crate) async fn create_event_ticket(State(state): State<AppState>) -> Json<TicketResponse> {
    Json(state.event_tickets.issue())
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
    if !state.settings.access_token.is_empty()
        && !state
            .event_tickets
            .consume(query.ticket.as_deref().unwrap_or_default())
    {
        return Err(ApiError::unauthorized(
            "Invalid or expired event stream ticket",
        ));
    }
    Ok(Sse::new(sse_event_stream(state.events.subscribe())))
}

fn sse_event_stream(
    messages: ReceiverStream<EventMessage>,
) -> impl futures_util::Stream<Item = Result<Event, Infallible>> {
    let mut heartbeat = tokio::time::interval_at(
        TokioInstant::now() + Duration::from_secs(15),
        Duration::from_secs(15),
    );
    heartbeat.set_missed_tick_behavior(MissedTickBehavior::Delay);
    futures_util::stream::unfold(
        (messages, heartbeat, true),
        |(mut messages, mut heartbeat, send_ready)| async move {
            if send_ready {
                return Some((Ok(ready_event()), (messages, heartbeat, false)));
            }
            tokio::select! {
                message = messages.next() => {
                    message.map(|message| (Ok(sse_message_event(message)), (messages, heartbeat, false)))
                }
                _ = heartbeat.tick() => {
                    Some((Ok(heartbeat_event()), (messages, heartbeat, false)))
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
    Event::default().event(message.event).data(message.data)
}

fn heartbeat_event() -> Event {
    Event::default().event("heartbeat").data(HEARTBEAT_SSE_DATA)
}
