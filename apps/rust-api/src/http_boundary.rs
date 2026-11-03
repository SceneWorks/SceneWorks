//! Listener and request bounds. Deadlines cover headers and request-body idle
//! gaps, never whole handlers or streaming responses (SSE, MCP and Range media).
use super::*;
use axum::extract::ConnectInfo;
use futures_util::StreamExt;
use hyper::server::conn::http1::Builder;
use hyper_util::rt::{TokioIo, TokioTimer};
use hyper_util::service::TowerToHyperService;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::net::TcpListener;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

pub(crate) const MAX_CONCURRENT_REQUESTS: usize = 256;
const MAX_CONNECTIONS: usize = 256;
const HEADER_READ_TIMEOUT: Duration = Duration::from_secs(15);
const REQUEST_BODY_IDLE_TIMEOUT: Duration = Duration::from_secs(120);

/// Shared API outbound deadlines. Download controllers retain their distinct
/// streaming/read-progress and Range-resume policy without a total deadline.
fn outbound_builder() -> reqwest::ClientBuilder {
    reqwest::Client::builder()
        .connect_timeout(sceneworks_core::network_policy::HTTP_CONNECT_TIMEOUT)
        .timeout(sceneworks_core::network_policy::API_OUTBOUND_TIMEOUT)
        .read_timeout(sceneworks_core::network_policy::API_OUTBOUND_TIMEOUT)
}

/// Public metadata probes retain bounded redirect following (e.g. renamed HF
/// repositories); they carry no user-supplied planner credentials or script.
pub(crate) fn outbound_client() -> Result<reqwest::Client, reqwest::Error> {
    outbound_builder()
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
}

/// Planner endpoints receive credentials and private scripts. Never follow a
/// provider's redirect to a different recipient, including same-origin redirects.
pub(crate) fn planner_client() -> Result<reqwest::Client, reqwest::Error> {
    outbound_builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
}

pub(crate) async fn no_referrer(request: AxumRequest, next: Next) -> Response {
    let mut response = next.run(request).await;
    response.headers_mut().insert(
        header::REFERRER_POLICY,
        HeaderValue::from_static("no-referrer"),
    );
    response
}

/// Every body-consuming route, including streaming multipart uploads, gets the
/// same generous idle budget. It resets per received chunk and leaves each
/// route's existing JSON/multipart size limit intact. Waiting for compute or a
/// streaming response after the body has been read has no deadline here.
pub(crate) async fn request_body_idle_timeout(request: AxumRequest, next: Next) -> Response {
    body_idle_timeout(request, next, REQUEST_BODY_IDLE_TIMEOUT).await
}

async fn body_idle_timeout(request: AxumRequest, next: Next, idle: Duration) -> Response {
    let timed_out = Arc::new(AtomicBool::new(false));
    let (parts, body) = request.into_parts();
    let stream = stream::unfold(
        (body.into_data_stream(), timed_out.clone(), false),
        move |(mut body, timed_out, finished)| async move {
            if finished {
                return None;
            }
            match tokio::time::timeout(idle, body.next()).await {
                Ok(Some(chunk)) => Some((
                    chunk.map_err(std::io::Error::other),
                    (body, timed_out, false),
                )),
                Ok(None) => None,
                Err(_) => {
                    timed_out.store(true, Ordering::Relaxed);
                    Some((
                        Err(std::io::Error::new(
                            std::io::ErrorKind::TimedOut,
                            "Request body stalled",
                        )),
                        (body, timed_out, true),
                    ))
                }
            }
        },
    );
    // Let the handler observe the read failure and clean up its staged upload
    // before replacing the extractor's generic 400/422 with the stable 408 contract.
    let response = next
        .run(Request::from_parts(parts, Body::from_stream(stream)))
        .await;
    if timed_out.load(Ordering::Relaxed) {
        (
            StatusCode::REQUEST_TIMEOUT,
            Json(json!({"detail": "Request body stalled"})),
        )
            .into_response()
    } else {
        response
    }
}

pub(crate) async fn serve(
    listener: TcpListener,
    app: Router,
    shutdown: impl std::future::Future<Output = ()>,
) -> std::io::Result<()> {
    serve_with_limits(
        listener,
        app,
        shutdown,
        HEADER_READ_TIMEOUT,
        MAX_CONNECTIONS,
    )
    .await
}

async fn serve_with_limits(
    listener: TcpListener,
    app: Router,
    shutdown: impl std::future::Future<Output = ()>,
    header_timeout: Duration,
    max_connections: usize,
) -> std::io::Result<()> {
    let capacity = Arc::new(Semaphore::new(max_connections));
    let app = app.layer(middleware::from_fn(no_referrer));
    let stopping = CancellationToken::new();
    let mut connections = JoinSet::new();
    tokio::pin!(shutdown);
    loop {
        while connections.try_join_next().is_some() {}
        let permit = tokio::select! {
            biased;
            _ = &mut shutdown => break,
            permit = capacity.clone().acquire_owned() => permit.expect("listener semaphore stays open"),
        };
        let accepted = tokio::select! {
            biased;
            _ = &mut shutdown => break,
            accepted = listener.accept() => accepted,
        };
        let (stream, peer) = match accepted {
            Ok(accepted) => accepted,
            Err(error) => {
                // Match axum's tolerance of transient accept errors without a hot loop.
                tracing::warn!(event = "http_accept_failed", %error, "HTTP accept failed; retrying");
                tokio::select! {
                    _ = &mut shutdown => break,
                    _ = tokio::time::sleep(Duration::from_secs(1)) => continue,
                }
            }
        };
        // Wrap the router service once per connection; Router::layer here would
        // rebuild every registered route for each accepted socket.
        let service = tower::ServiceBuilder::new()
            .layer(axum::Extension(ConnectInfo(peer)))
            .service(app.clone());
        let stopping = stopping.clone();
        connections.spawn(async move {
            let _permit = permit;
            let mut builder = Builder::new();
            builder
                .timer(TokioTimer::new())
                .header_read_timeout(header_timeout);
            // axum's existing default listener is HTTP/1; retain that protocol
            // surface and do not pull in an unconfigured HTTP/2 stack.
            let connection = builder
                .serve_connection(TokioIo::new(stream), TowerToHyperService::new(service))
                .with_upgrades();
            tokio::pin!(connection);
            let result = tokio::select! {
                result = &mut connection => result,
                _ = stopping.cancelled() => {
                    connection.as_mut().graceful_shutdown();
                    connection.await
                }
            };
            if let Err(error) = result {
                tracing::debug!(event = "http_connection_closed", %error, "HTTP connection ended");
            }
        });
    }
    stopping.cancel();
    // The owning server's existing bounded_server_drain limits this graceful
    // drain and drops the JoinSet (aborting children) if a client never finishes.
    while connections.join_next().await.is_some() {}
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tower::ServiceExt;

    #[tokio::test]
    async fn listener_bounds_partial_headers_and_preserves_peer_and_streaming() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new()
            .route(
                "/",
                get(|ConnectInfo(peer): ConnectInfo<SocketAddr>| async move {
                    assert!(peer.ip().is_loopback());
                    "authorized peer"
                }),
            )
            .route(
                "/events",
                get(|| async {
                    (
                        [(header::CONTENT_TYPE, "text/event-stream")],
                        Body::from_stream(stream::once(async {
                            tokio::time::sleep(Duration::from_millis(150)).await;
                            Ok::<_, std::io::Error>("data: still streaming\n\n")
                        })),
                    )
                }),
            );
        let shutdown = CancellationToken::new();
        let stop = shutdown.clone();
        let server = tokio::spawn(serve_with_limits(
            listener,
            app,
            async move { stop.cancelled().await },
            Duration::from_millis(40),
            2,
        ));
        let mut slow = tokio::net::TcpStream::connect(address).await.unwrap();
        slow.write_all(b"GET / HTTP/1.1\r\nHost:").await.unwrap();
        let mut buf = [0; 256];
        let count = tokio::time::timeout(Duration::from_secs(2), slow.read(&mut buf))
            .await
            .unwrap()
            .unwrap();
        assert!(count == 0 || String::from_utf8_lossy(&buf[..count]).contains("408"));
        let response = reqwest::get(format!("http://{address}/")).await.unwrap();
        assert_eq!(response.text().await.unwrap(), "authorized peer");
        let response = reqwest::get(format!("http://{address}/events"))
            .await
            .unwrap();
        assert_eq!(
            response.text().await.unwrap(),
            "data: still streaming\n\n",
            "response streams have no header/request total deadline"
        );
        shutdown.cancel();
        server.await.unwrap().unwrap();
    }

    fn idle_app() -> Router {
        Router::new()
            .route(
                "/upload",
                post(|body: Body| async move {
                    match to_bytes(body, usize::MAX).await {
                        Ok(bytes) => (StatusCode::OK, bytes.len().to_string()).into_response(),
                        Err(_) => StatusCode::BAD_REQUEST.into_response(),
                    }
                }),
            )
            .layer(middleware::from_fn(|request, next| {
                body_idle_timeout(request, next, Duration::from_millis(100))
            }))
            .layer(middleware::from_fn(no_referrer))
    }

    #[tokio::test]
    async fn idle_body_returns_json_408_while_progressing_upload_exceeds_idle_window() {
        let stalled =
            Body::from_stream(stream::pending::<Result<axum::body::Bytes, std::io::Error>>());
        let response = idle_app()
            .oneshot(Request::post("/upload").body(stalled).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);
        assert_eq!(response.headers()[header::REFERRER_POLICY], "no-referrer");
        let body = to_bytes(response.into_body(), 1024).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&body).unwrap()["detail"],
            "Request body stalled"
        );
        let chunks = stream::unfold(0, |index| async move {
            if index == 8 {
                return None;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
            Some((Ok::<_, std::io::Error>(vec![b'x'; 1024 * 1024]), index + 1))
        });
        let response = idle_app()
            .oneshot(
                Request::post("/upload")
                    .body(Body::from_stream(chunks))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            to_bytes(response.into_body(), 64).await.unwrap(),
            (8 * 1024 * 1024).to_string()
        );
    }
    #[tokio::test]
    async fn global_request_limit_is_shared_across_routes_and_does_not_hold_sse_bodies() {
        let entered = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let (on_entry, wait) = (entered.clone(), release.clone());
        let app = Router::new()
            .route(
                "/busy",
                get(move || {
                    let entered = on_entry.clone();
                    let release = wait.clone();
                    async move {
                        entered.notify_one();
                        release.notified().await;
                        "finished"
                    }
                }),
            )
            .route("/other", get(|| async { "available" }))
            .route(
                "/events",
                get(|| async {
                    (
                        [(header::CONTENT_TYPE, "text/event-stream")],
                        Body::from_stream(stream::pending::<
                            Result<axum::body::Bytes, std::io::Error>,
                        >()),
                    )
                }),
            )
            .layer(tower::limit::GlobalConcurrencyLimitLayer::new(1));
        let first = tokio::spawn(
            app.clone()
                .oneshot(Request::get("/busy").body(Body::empty()).unwrap()),
        );
        entered.notified().await;
        let second = app
            .clone()
            .oneshot(Request::get("/other").body(Body::empty()).unwrap());
        tokio::pin!(second);
        assert!(tokio::time::timeout(Duration::from_millis(20), &mut second)
            .await
            .is_err());
        release.notify_one();
        assert_eq!(first.await.unwrap().unwrap().status(), StatusCode::OK);
        assert_eq!(second.await.unwrap().status(), StatusCode::OK);
        let streaming = app
            .clone()
            .oneshot(Request::get("/events").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(
            streaming.headers()[header::CONTENT_TYPE],
            "text/event-stream"
        );
        let ordinary = tokio::time::timeout(
            Duration::from_secs(1),
            app.oneshot(Request::get("/other").body(Body::empty()).unwrap()),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(ordinary.status(), StatusCode::OK);
        drop(streaming);
    }

    #[tokio::test]
    async fn outbound_api_client_refuses_redirects_and_keeps_request_deadline_override() {
        let reached = Arc::new(AtomicBool::new(false));
        let received = reached.clone();
        let app = Router::new()
            .route(
                "/redirect",
                get(|| async { axum::response::Redirect::temporary("/credential-target") }),
            )
            .route(
                "/credential-target",
                get(move || {
                    let received = received.clone();
                    async move {
                        received.store(true, Ordering::Relaxed);
                        "must not receive credential"
                    }
                }),
            )
            .route(
                "/stall",
                get(|| async {
                    std::future::pending::<()>().await;
                    "unreachable"
                }),
            );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let client = planner_client().unwrap();
        let response = client
            .get(format!("http://{address}/redirect"))
            .bearer_auth("fixture-credential")
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::TEMPORARY_REDIRECT);
        assert!(!reached.load(Ordering::Relaxed));
        let public_probe = outbound_client()
            .unwrap()
            .get(format!("http://{address}/redirect"))
            .send()
            .await
            .unwrap();
        assert_eq!(public_probe.status(), StatusCode::OK);
        assert!(
            reached.load(Ordering::Relaxed),
            "public metadata retains redirect compatibility"
        );
        let error = client
            .get(format!("http://{address}/stall"))
            .timeout(Duration::from_millis(30))
            .send()
            .await
            .unwrap_err();
        assert!(error.is_timeout());
        server.abort();
    }
    #[tokio::test]
    async fn stalled_real_multipart_route_cleans_staging_before_json_408() {
        use crate::tests::support::{request, test_settings};
        let temp = tempfile::tempdir().unwrap();
        let app = crate::create_app(test_settings(&temp)).unwrap();
        let (_, project) = request(
            app.clone(),
            "POST",
            "/api/v1/projects",
            json!({"name": "Stalled upload"}),
        )
        .await;
        let prefix = b"--idle-boundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"image.png\"\r\nContent-Type: image/png\r\n\r\npartial-upload-content";
        let body =
            stream::once(async { Ok::<_, std::io::Error>(axum::body::Bytes::from_static(prefix)) })
                .chain(stream::pending());
        let app = app.layer(middleware::from_fn(|request, next| {
            body_idle_timeout(request, next, Duration::from_millis(200))
        }));
        let response = app
            .oneshot(
                Request::post(format!(
                    "/api/v1/projects/{}/assets",
                    project["id"].as_str().unwrap()
                ))
                .header(
                    header::CONTENT_TYPE,
                    "multipart/form-data; boundary=idle-boundary",
                )
                .body(Body::from_stream(body))
                .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);
        let value: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 1024).await.unwrap()).unwrap();
        assert_eq!(value["detail"], "Request body stalled");
        let staging = temp.path().join("data/cache/uploads");
        assert!(
            staging.exists(),
            "the real multipart handler reached its staging writer"
        );
        assert_eq!(
            std::fs::read_dir(staging).unwrap().count(),
            0,
            "no abandoned upload file"
        );
    }
    #[tokio::test]
    async fn listener_connection_cap_queues_peers_until_header_timeout_releases_capacity() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new().route("/", get(|| async { "admitted" }));
        let shutdown = CancellationToken::new();
        let stop = shutdown.clone();
        let server = tokio::spawn(serve_with_limits(
            listener,
            app,
            async move { stop.cancelled().await },
            Duration::from_millis(150),
            1,
        ));
        let mut silent = tokio::net::TcpStream::connect(address).await.unwrap();
        // The first accepted peer sends no headers at all.
        let mut queued = tokio::net::TcpStream::connect(address).await.unwrap();
        queued
            .write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            .await
            .unwrap();
        let mut bytes = [0; 512];
        assert!(
            tokio::time::timeout(Duration::from_millis(20), queued.read(&mut bytes))
                .await
                .is_err(),
            "the second connection cannot enter the service while capacity is held"
        );
        tokio::time::timeout(Duration::from_secs(2), silent.read(&mut bytes))
            .await
            .unwrap()
            .unwrap();
        let count = tokio::time::timeout(Duration::from_secs(2), queued.read(&mut bytes))
            .await
            .unwrap()
            .unwrap();
        assert!(String::from_utf8_lossy(&bytes[..count]).contains("200 OK"));
        shutdown.cancel();
        server.await.unwrap().unwrap();
    }
}
