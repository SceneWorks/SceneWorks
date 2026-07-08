
#[test]
fn parse_credentials_env_normalizes_and_skips_blanks() {
    let credentials = parse_credentials_env(
        r#"{ "Civitai.com": { "token": " key ", "scheme": "query" },
            "huggingface.co": { "token": "hf" },
            "blank.example": { "token": "" } }"#,
    );
    assert_eq!(credentials.len(), 2);
    let civitai = credentials
        .iter()
        .find(|credential| credential.host == "civitai.com")
        .expect("civitai credential");
    assert_eq!(civitai.token, "key");
    assert_eq!(civitai.scheme, CredentialScheme::Query);
    let hugging_face = credentials
        .iter()
        .find(|credential| credential.host == "huggingface.co")
        .expect("hf credential");
    // An absent scheme defaults to bearer.
    assert_eq!(hugging_face.scheme, CredentialScheme::Bearer);
}

#[test]
fn parse_credentials_env_tolerates_invalid_json() {
    assert!(parse_credentials_env("not json").is_empty());
}

#[test]
fn credential_for_host_matches_case_insensitively() {
    let mut settings = test_settings("https://huggingface.co".to_owned(), None);
    settings.credentials = vec![WorkerCredential {
        host: "civitai.com".to_owned(),
        token: "key".to_owned(),
        scheme: CredentialScheme::Query,
    }];
    assert!(credential_for_host(&settings, "Civitai.com").is_some());
    assert!(credential_for_host(&settings, "example.com").is_none());
    assert!(credential_for_host(&settings, "").is_none());
}

#[test]
fn worker_credentials_env_overrides_file_per_host() {
    // Server reads the config-dir file store; an operator's SCENEWORKS_CREDENTIALS
    // env wins per host, and file-only hosts survive.
    let file = parse_credentials_env(
        r#"{ "civitai.com": { "token": "file-civitai", "scheme": "query" },
            "huggingface.co": { "token": "file-hf" } }"#,
    );
    let env = parse_credentials_env(
        r#"{ "civitai.com": { "token": "env-civitai", "scheme": "bearer" } }"#,
    );
    let merged = super::merge_credentials(file, env);
    assert_eq!(merged.len(), 2);
    let civitai = merged
        .iter()
        .find(|credential| credential.host == "civitai.com")
        .expect("civitai credential");
    assert_eq!(civitai.token, "env-civitai");
    assert_eq!(civitai.scheme, CredentialScheme::Bearer);
    let hugging_face = merged
        .iter()
        .find(|credential| credential.host == "huggingface.co")
        .expect("hf credential");
    assert_eq!(hugging_face.token, "file-hf");
}

#[tokio::test]
async fn source_url_follows_redirect_and_strips_auth_across_hosts() {
    let temp = tempdir().expect("tempdir creates");
    // The download host (127.0.0.1) requires a bearer token, then 302-redirects to
    // a different host (localhost) that rejects any Authorization header — so the
    // download only succeeds if the token is applied on hop 1 and dropped on hop 2.
    let download_base = spawn_cross_host_redirect_stub("testtoken").await;
    let api_base = spawn_binary_stub(b"ignored".to_vec()).await;
    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.api_url = api_base.clone();
    settings.credentials = vec![WorkerCredential {
        host: "127.0.0.1".to_owned(),
        token: "testtoken".to_owned(),
        scheme: CredentialScheme::Bearer,
    }];
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();
    let target_dir = temp.path().join("redirect-target");

    download_lora_source_url(
        &DownloadContext {
            api: &api,
            client: &client,
            settings: &settings,
            job_id: "job-1",
            cancel_message: "canceled",
            fresh_download: false,
        },
        &format!("{download_base}/download/style.safetensors"),
        &target_dir,
    )
    .await
    .expect("authenticated redirected download succeeds");

    assert_eq!(
        tokio::fs::read(target_dir.join("style.safetensors"))
            .await
            .unwrap(),
        b"civitai-lora"
    );
}

#[tokio::test]
async fn source_url_client_pins_dns_to_validated_address() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener binds");
    let address = listener.local_addr().expect("listener has address");
    let state = BinaryStubState {
        bytes: b"weights!!".to_vec(),
        status: AxumStatusCode::OK,
        cancel_requested: false,
    };
    let app = Router::new()
        .route("/file/style.safetensors", get(binary_stub))
        .with_state(state);
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("stub serves");
    });

    let url = reqwest::Url::parse(&format!(
        "http://rebind.test:{}/file/style.safetensors",
        address.port()
    ))
    .expect("test URL parses");
    let validated = [SocketAddr::new(
        "127.0.0.1".parse().unwrap(),
        address.port(),
    )];
    let client = build_source_url_client(&url, Some(&validated)).expect("client builds");

    let bytes = client
        .get(url)
        .send()
        .await
        .expect("request uses pinned address")
        .error_for_status()
        .expect("stub response is successful")
        .bytes()
        .await
        .expect("response body reads");

    assert_eq!(bytes.as_ref(), b"weights!!");
}

#[tokio::test]
async fn source_url_rejects_redirect_to_non_http_scheme() {
    let temp = tempdir().expect("tempdir creates");
    let download_base = spawn_location_redirect_stub("file:///etc/passwd").await;
    let api_base = spawn_binary_stub(b"ignored".to_vec()).await;
    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.api_url = api_base;
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();

    let error = download_lora_source_url(
        &DownloadContext {
            api: &api,
            client: &client,
            settings: &settings,
            job_id: "job-1",
            cancel_message: "canceled",
            fresh_download: false,
        },
        &format!("{download_base}/download/style.safetensors"),
        &temp.path().join("scheme-target"),
    )
    .await
    .expect_err("non-http redirect target is rejected");
    assert!(error.to_string().contains("http or https"));
}

#[tokio::test]
async fn source_url_rejects_excessive_redirects() {
    let temp = tempdir().expect("tempdir creates");
    // Always redirects to a sibling path on the same host — an unterminated loop.
    let download_base = spawn_location_redirect_stub("loop").await;
    let api_base = spawn_binary_stub(b"ignored".to_vec()).await;
    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.api_url = api_base;
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();

    let error = download_lora_source_url(
        &DownloadContext {
            api: &api,
            client: &client,
            settings: &settings,
            job_id: "job-1",
            cancel_message: "canceled",
            fresh_download: false,
        },
        &format!("{download_base}/download/style.safetensors"),
        &temp.path().join("loop-target"),
    )
    .await
    .expect_err("a redirect loop is bounded");
    assert!(error.to_string().contains("redirect limit"));
}

#[derive(Clone)]
struct CrossHostRedirectState {
    port: u16,
    token: String,
}

async fn spawn_cross_host_redirect_stub(token: &str) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener binds");
    let address = listener.local_addr().expect("listener has address");
    let state = CrossHostRedirectState {
        port: address.port(),
        token: token.to_owned(),
    };
    let app = Router::new()
        .route(
            "/download/*path",
            get(cross_host_download).head(cross_host_download),
        )
        .route("/file/*path", get(cross_host_file))
        .with_state(state);
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("stub serves");
    });
    format!("http://{address}")
}

async fn cross_host_download(
    State(state): State<CrossHostRedirectState>,
    headers: HeaderMap,
) -> Response {
    if !has_bearer(&headers, &state.token) {
        return AxumStatusCode::UNAUTHORIZED.into_response();
    }
    let mut response = Response::new(Body::empty());
    *response.status_mut() = AxumStatusCode::FOUND;
    response.headers_mut().insert(
        axum::http::header::LOCATION,
        axum::http::HeaderValue::from_str(&format!(
            "http://localhost:{}/file/style.safetensors",
            state.port
        ))
        .expect("location header"),
    );
    response
}

async fn cross_host_file(headers: HeaderMap) -> Response {
    // The bearer token must never be carried onto the cross-host CDN hop.
    if headers.contains_key(axum::http::header::AUTHORIZATION) {
        return AxumStatusCode::FORBIDDEN.into_response();
    }
    let bytes = b"civitai-lora".to_vec();
    let length = bytes.len();
    let mut response = bytes.into_response();
    response.headers_mut().insert(
        axum::http::header::CONTENT_LENGTH,
        axum::http::HeaderValue::from_str(&length.to_string()).expect("content length header"),
    );
    response
}

#[derive(Clone)]
struct LocationRedirectState {
    location: String,
}

async fn spawn_location_redirect_stub(location: &str) -> String {
    let state = LocationRedirectState {
        location: location.to_owned(),
    };
    let app = Router::new()
        .route(
            "/download/*path",
            get(location_redirect).head(location_redirect),
        )
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener binds");
    let address = listener.local_addr().expect("listener has address");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("stub serves");
    });
    format!("http://{address}")
}

async fn location_redirect(State(state): State<LocationRedirectState>) -> Response {
    let mut response = Response::new(Body::empty());
    *response.status_mut() = AxumStatusCode::FOUND;
    response.headers_mut().insert(
        axum::http::header::LOCATION,
        axum::http::HeaderValue::from_str(&state.location).expect("location header"),
    );
    response
}

fn has_bearer(headers: &HeaderMap, token: &str) -> bool {
    let expected = format!("Bearer {token}");
    headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        == Some(expected.as_str())
}
