use super::*;

// Short-lived query-param tickets for endpoints the browser cannot reach with an
// auth header (epic 4484). Two flavors share this store:
//  - SSE (`/api/v1/jobs/events`): `EventSource` can't set headers → single-use
//    `issue()` + `consume()` tickets (sc-4484 story: events ticket).
//  - Media (`GET /api/v1/projects/:id/files/*`, `GET /api/v1/poses/preview/*`,
//    `GET /api/v1/catalogs/:id/records/:id/thumbnail`):
//    `<img src>`/`<video src>`/`<a download>` requests can't set headers either →
//    reusable `issue_sliding()` + `validate()` tickets (sc-8810). A page renders
//    dozens of media URLs (and <video> issues multiple Range requests), so these
//    tickets are multi-use within their TTL; the web client refreshes well before
//    expiry and `issue_sliding` keeps returning (and re-arming) the same ticket so
//    already-rendered <img>/<video> URLs stay valid without a re-render.
//
// Separate `TicketStore` instances give scope isolation for free: an event ticket
// is never accepted on the files route and vice versa.
#[derive(Debug)]
pub(crate) struct TicketStore {
    ttl: Duration,
    max_outstanding: Option<usize>,
    state: Mutex<TicketStoreState>,
}

#[derive(Debug, Default)]
struct TicketStoreState {
    tickets: HashMap<String, TicketEntry>,
    // Most recently issued ticket, so `issue_sliding` can hand the same value to
    // every caller while it stays alive (URL stability across React re-renders).
    latest: Option<String>,
}

#[derive(Debug)]
struct TicketEntry {
    expires_at: Instant,
    event_context: EventTicketContext,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct EventTicketContext {
    pub(crate) active_job_ids: Vec<String>,
    pub(crate) known_terminal_job_ids: Vec<String>,
}

impl TicketStore {
    pub(crate) fn new(ttl_seconds: u64) -> Self {
        Self {
            ttl: Duration::from_secs(ttl_seconds),
            max_outstanding: None,
            state: Mutex::new(TicketStoreState::default()),
        }
    }

    pub(crate) fn with_max_outstanding(ttl_seconds: u64, max_outstanding: usize) -> Self {
        assert!(max_outstanding > 0, "ticket capacity must be non-zero");
        Self {
            ttl: Duration::from_secs(ttl_seconds),
            max_outstanding: Some(max_outstanding),
            state: Mutex::new(TicketStoreState::default()),
        }
    }

    /// Issue a fresh single-use ticket (SSE flavor; pair with `consume`).
    pub(crate) fn issue(&self) -> TicketResponse {
        self.try_issue_event(EventTicketContext::default())
            .expect("unbounded ticket store issues")
    }

    /// Issue an SSE ticket carrying the reconnecting client's bounded job context.
    /// The opaque ticket keeps the EventSource GET URL short; its payload disappears
    /// with the same single-use redemption. A bounded event store returns `None`
    /// instead of allocating beyond its outstanding-ticket backpressure ceiling.
    pub(crate) fn try_issue_event(
        &self,
        event_context: EventTicketContext,
    ) -> Option<TicketResponse> {
        let now = Instant::now();
        let mut state = self.state.lock();
        prune_tickets(&mut state.tickets, now);
        if self
            .max_outstanding
            .is_some_and(|limit| state.tickets.len() >= limit)
        {
            return None;
        }
        let ticket = Uuid::new_v4().simple().to_string();
        state.tickets.insert(
            ticket.clone(),
            TicketEntry {
                expires_at: now + self.ttl,
                event_context,
            },
        );
        state.latest = Some(ticket.clone());
        Some(TicketResponse {
            ticket,
            expires_in_seconds: self.ttl.as_secs(),
        })
    }

    /// Issue a reusable ticket (media flavor; pair with `validate`). While the most
    /// recent ticket is still valid this returns it again and re-arms its expiry to
    /// a full TTL, so a client refreshing on an interval keeps one stable ticket
    /// alive for its whole session. A leaked URL therefore dies at most one TTL
    /// after the last authenticated refresh.
    pub(crate) fn issue_sliding(&self) -> TicketResponse {
        let now = Instant::now();
        let mut state = self.state.lock();
        prune_tickets(&mut state.tickets, now);
        if let Some(ticket) = state.latest.clone() {
            if state.tickets.contains_key(&ticket) {
                state.tickets.insert(
                    ticket.clone(),
                    TicketEntry {
                        expires_at: now + self.ttl,
                        event_context: EventTicketContext::default(),
                    },
                );
                return TicketResponse {
                    ticket,
                    expires_in_seconds: self.ttl.as_secs(),
                };
            }
        }
        drop(state);
        self.issue()
    }

    /// Redeem one SSE ticket and return the reconnect context minted with it.
    pub(crate) fn consume_event(&self, ticket: &str) -> Option<EventTicketContext> {
        let now = Instant::now();
        let mut state = self.state.lock();
        prune_tickets(&mut state.tickets, now);
        state
            .tickets
            .remove(ticket)
            .filter(|entry| entry.expires_at >= now)
            .map(|entry| entry.event_context)
    }

    /// Non-consuming check for multi-use (media) tickets.
    pub(crate) fn validate(&self, ticket: &str) -> bool {
        if ticket.is_empty() {
            return false;
        }
        let now = Instant::now();
        let mut state = self.state.lock();
        prune_tickets(&mut state.tickets, now);
        matches!(state.tickets.get(ticket), Some(entry) if entry.expires_at >= now)
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct TicketResponse {
    pub(crate) ticket: String,
    pub(crate) expires_in_seconds: u64,
}

/// POST /api/v1/files/ticket — auth-protected (header token or loopback trust via
/// the access-control middleware), so only an already-authenticated client can
/// mint a media ticket.
pub(crate) async fn create_media_ticket(State(state): State<AppState>) -> Json<TicketResponse> {
    Json(state.media_tickets.issue_sliding())
}

fn prune_tickets(tickets: &mut HashMap<String, TicketEntry>, now: Instant) {
    tickets.retain(|_, entry| entry.expires_at >= now);
}
