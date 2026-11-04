use super::*;

// Query credentials are isolated by route family and store: single-use SSE
// tickets cannot unlock media. Media credentials are private to each client and
// renewed only through the authenticated mint endpoint. Read access never extends
// expiry, and authenticated renewal cannot extend the original absolute lifetime.
#[derive(Debug)]
pub(crate) struct TicketStore {
    ttl: Duration,
    max_outstanding: Option<usize>,
    absolute_ttl: Duration,
    state: Mutex<TicketStoreState>,
}

#[derive(Debug, Default)]
struct TicketStoreState {
    tickets: HashMap<String, TicketEntry>,
}

#[derive(Debug)]
struct TicketEntry {
    expires_at: Instant,
    absolute_expires_at: Instant,
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
            absolute_ttl: Duration::from_secs(ttl_seconds),
            state: Mutex::new(TicketStoreState::default()),
        }
    }

    pub(crate) fn with_max_outstanding(ttl_seconds: u64, max_outstanding: usize) -> Self {
        assert!(max_outstanding > 0, "ticket capacity must be non-zero");
        Self {
            ttl: Duration::from_secs(ttl_seconds),
            max_outstanding: Some(max_outstanding),
            absolute_ttl: Duration::from_secs(ttl_seconds),
            state: Mutex::new(TicketStoreState::default()),
        }
    }

    pub(crate) fn media(idle_seconds: u64, absolute_seconds: u64) -> Self {
        Self {
            absolute_ttl: Duration::from_secs(absolute_seconds),
            ..Self::new(idle_seconds)
        }
    }

    /// Issue a fresh single-use ticket (SSE flavor; pair with `consume`).
    #[cfg(test)]
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
                expires_at: now + self.ttl.min(self.absolute_ttl),
                absolute_expires_at: now + self.absolute_ttl,
                event_context,
            },
        );
        Some(TicketResponse {
            ticket,
            expires_in_seconds: self.ttl.as_secs(),
        })
    }

    /// Renew only the client's supplied credential, never another client's latest
    /// ticket. Rotate before the absolute boundary while the previous credential
    /// still has idle overlap, so mounted media can switch URLs and finish ranges.
    pub(crate) fn issue_sliding(&self, previous: Option<&str>) -> TicketResponse {
        self.issue_sliding_at(previous, Instant::now())
    }

    fn issue_sliding_at(&self, previous: Option<&str>, now: Instant) -> TicketResponse {
        let mut state = self.state.lock();
        prune_tickets(&mut state.tickets, now);
        if let Some((ticket, entry)) =
            previous.and_then(|ticket| state.tickets.get_mut(ticket).map(|entry| (ticket, entry)))
        {
            if entry.absolute_expires_at.saturating_duration_since(now) > self.ttl {
                entry.expires_at = (now + self.ttl).min(entry.absolute_expires_at);
                return TicketResponse {
                    ticket: ticket.to_owned(),
                    expires_in_seconds: entry.expires_at.saturating_duration_since(now).as_secs(),
                };
            }
        }
        let ticket = Uuid::new_v4().simple().to_string();
        let expires_at = now + self.ttl.min(self.absolute_ttl);
        state.tickets.insert(
            ticket.clone(),
            TicketEntry {
                expires_at,
                absolute_expires_at: now + self.absolute_ttl,
                event_context: EventTicketContext::default(),
            },
        );
        TicketResponse {
            ticket,
            expires_in_seconds: expires_at.saturating_duration_since(now).as_secs(),
        }
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

    #[cfg(test)]
    pub(crate) fn expire_for_test(&self, ticket: &str) {
        let mut state = self.state.lock();
        let entry = state.tickets.get_mut(ticket).expect("test ticket exists");
        entry.expires_at = Instant::now() - Duration::from_secs(1);
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
pub(crate) async fn create_media_ticket(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Json<TicketResponse> {
    let previous = headers
        .get("x-sceneworks-media-ticket")
        .and_then(|value| value.to_str().ok());
    Json(state.media_tickets.issue_sliding(previous))
}

fn prune_tickets(tickets: &mut HashMap<String, TicketEntry>, now: Instant) {
    tickets.retain(|_, entry| entry.expires_at >= now && entry.absolute_expires_at >= now);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn media_credentials_have_independent_idle_and_absolute_expiry() {
        let store = TicketStore::media(30, 120);
        let now = Instant::now();
        let first = store.issue_sliding_at(None, now);
        let other = store.issue_sliding_at(None, now);
        assert_ne!(
            first.ticket, other.ticket,
            "different clients never share credentials"
        );
        for seconds in [20, 40, 60, 80] {
            assert_eq!(
                store
                    .issue_sliding_at(Some(&first.ticket), now + Duration::from_secs(seconds))
                    .ticket,
                first.ticket
            );
            assert_eq!(
                store.state.lock().tickets[&first.ticket].absolute_expires_at,
                now + Duration::from_secs(120),
                "renewal never extends absolute expiry"
            );
        }
        let rotated = store.issue_sliding_at(Some(&first.ticket), now + Duration::from_secs(91));
        assert_ne!(rotated.ticket, first.ticket);
        // Refreshing the replacement or another client cannot resurrect the original.
        let mut state = store.state.lock();
        prune_tickets(&mut state.tickets, now + Duration::from_secs(121));
        assert!(!state.tickets.contains_key(&first.ticket));
        assert!(
            !state.tickets.contains_key(&other.ticket),
            "unrenewed clients idle out"
        );
        assert!(state.tickets.contains_key(&rotated.ticket));
    }
}
