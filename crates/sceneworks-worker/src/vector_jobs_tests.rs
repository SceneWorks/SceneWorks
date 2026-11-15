use super::*;

fn vector_request(mode: VectorMode, prompt: &str) -> VectorProviderRequest {
    VectorProviderRequest {
        mode,
        model: "starvector_1b".to_owned(),
        source_path: None,
        prompt: prompt.to_owned(),
        sampling: VectorSampling {
            temperature: 0.2,
            top_p: 0.9,
            top_k: 0,
            repetition_penalty: 1.0,
            repetition_context: 0,
            seed: Some(7),
        },
        detail_budget: VectorDetailBudget {
            max_new_tokens: 128,
            max_svg_bytes: 4_096,
            max_wall_time_ms: 1_000,
        },
    }
}

fn invalid_detail(result: WorkerResult<CanonicalSvg>) -> String {
    match result.expect_err("fixture must be rejected") {
        WorkerError::InvalidPayload(detail) => detail,
        other => panic!("expected invalid payload, got {other}"),
    }
}

fn terminal_fixture(
    finish_reason: StarVectorFinishReason,
    source: &str,
    svg: Option<&str>,
) -> (StarVectorOutput, Vec<StarVectorStreamEvent>) {
    let generated_tokens = u32::from(!source.is_empty());
    let generated_bytes = source.len();
    let mut events = Vec::new();
    if !source.is_empty() {
        events.push(StarVectorStreamEvent::Source {
            text: source.to_owned(),
            index: 0,
        });
    }
    events.push(StarVectorStreamEvent::Done {
        finish_reason,
        generated_tokens,
        generated_bytes,
    });
    (
        StarVectorOutput {
            svg: svg.map(str::to_owned),
            generated_tokens,
            generated_bytes,
            finish_reason,
            // Fused compressed KV report (inference epic sc-20669): the fakes ran no cache.
            kv_cache: None,
        },
        events,
    )
}

struct CancelingProvider;

impl MultimodalVectorProviderAdapter for CancelingProvider {
    fn provider_id(&self) -> &str {
        "starvector"
    }

    fn supports_mode(&self, mode: VectorMode) -> bool {
        mode == VectorMode::TextToSvg
    }

    fn generate_svg(
        &self,
        _request: &VectorProviderRequest,
        cancel: &gen_core::CancelFlag,
        _progress: tokio::sync::watch::Sender<u32>,
        on_source: &mut dyn FnMut(&str, u32) -> WorkerResult<()>,
    ) -> WorkerResult<()> {
        on_source("<svg xmlns=\"http://www.w3.org/2000/svg\">", 0)?;
        cancel.cancel();
        on_source("<rect width=\"1\" height=\"1\"/></svg>", 1)?;
        Ok(())
    }
}

struct NativeSourceSequence([u32; 3]);

impl MultimodalVectorProviderAdapter for NativeSourceSequence {
    fn provider_id(&self) -> &str {
        "starvector"
    }

    fn supports_mode(&self, mode: VectorMode) -> bool {
        mode == VectorMode::TextToSvg
    }

    fn generate_svg(
        &self,
        request: &VectorProviderRequest,
        _cancel: &gen_core::CancelFlag,
        progress: tokio::sync::watch::Sender<u32>,
        on_source: &mut dyn FnMut(&str, u32) -> WorkerResult<()>,
    ) -> WorkerResult<()> {
        // One hidden decoder token separates the static prefix and first visible delta.
        // Validate native output/events, then forward the original indices just as the
        // production adapter does. Source fragments and token counts stay independent.
        let svg = "<svg></svg>";
        let output = StarVectorOutput {
            svg: Some(svg.to_owned()),
            generated_tokens: 3,
            generated_bytes: svg.len(),
            finish_reason: StarVectorFinishReason::CompleteRoot,
            kv_cache: None,
        };
        let mut events = Vec::new();
        for (index, fragment) in self.0.into_iter().zip(["<svg", ">", "</svg>"]) {
            events.push(StarVectorStreamEvent::Source {
                text: fragment.to_owned(),
                index,
            });
        }
        for generated_tokens in 1..=3 {
            record_vector_token_progress(
                &progress,
                generated_tokens,
                request.detail_budget.max_new_tokens,
            );
        }
        events.push(StarVectorStreamEvent::Done {
            finish_reason: output.finish_reason,
            generated_tokens: output.generated_tokens,
            generated_bytes: output.generated_bytes,
        });
        for (fragment, index) in validate_native_starvector_generation(output, events)? {
            on_source(&fragment, index)?;
        }
        Ok(())
    }
}

/// A typed StarVector provider whose decode either completes with a dense KV report or is
/// refused by the engine's memory admission.
struct StubStarVector {
    refuse: bool,
}

impl gen_core::core_llm::TextLlm for StubStarVector {
    fn descriptor(&self) -> &gen_core::core_llm::TextLlmDescriptor {
        crate::llm_kv_cache::test_support::RefusingLlm.descriptor()
    }

    fn validate(&self, _req: &TextLlmRequest) -> gen_core::core_llm::Result<()> {
        Ok(())
    }

    fn generate(
        &self,
        _req: &TextLlmRequest,
        _on_event: &mut dyn FnMut(gen_core::core_llm::StreamEvent),
    ) -> gen_core::core_llm::Result<gen_core::core_llm::TextLlmOutput> {
        unreachable!("StarVector decodes through generate_svg")
    }
}

impl gen_core::core_llm::StarVectorProvider for StubStarVector {
    fn starvector_descriptor(&self) -> &gen_core::core_llm::StarVectorDescriptor {
        unreachable!("the decode never reads the descriptor")
    }

    fn generate_svg(
        &self,
        request: &StarVectorRequest,
        on_event: &mut dyn FnMut(StarVectorStreamEvent),
    ) -> gen_core::core_llm::Result<StarVectorOutput> {
        if self.refuse {
            return Err(crate::llm_kv_cache::test_support::refusal());
        }
        on_event(StarVectorStreamEvent::Progress {
            generated_tokens: 1,
        });
        Ok(StarVectorOutput {
            svg: Some("<svg/>".to_owned()),
            generated_tokens: 1,
            generated_bytes: 6,
            finish_reason: StarVectorFinishReason::CompleteRoot,
            kv_cache: Some(gen_core::core_llm::KvCacheReport::without_table_family(
                request.text_request.kv_compression,
            )),
        })
    }
}

fn starvector_request(policy: gen_core::core_llm::KvCompressionPolicy) -> StarVectorRequest {
    StarVectorRequest::new(
        TextLlmRequest {
            kv_compression: policy,
            max_new_tokens: 8,
            ..TextLlmRequest::default()
        },
        4_096,
        Duration::from_secs(1),
    )
}

/// sc-20682 / sc-20688: the native decode keeps its KV-cache record whichever way it ends —
/// the engine's report when it completes, the requested policy and the refusal when the
/// engine refuses it at memory admission.
#[test]
fn the_native_decode_records_its_kv_cache_on_success_and_refusal() {
    use gen_core::core_llm::KvCompressionPolicy;
    let slot = Mutex::new(None);
    let (progress, _) = tokio::sync::watch::channel(0);
    let request = starvector_request(KvCompressionPolicy::Qualified);
    decode_native_starvector(
        &StubStarVector { refuse: false },
        &request,
        &progress,
        "mlx",
        &slot,
    )
    .expect("completed decode");
    let completed = slot.lock().unwrap().clone().expect("completed record");
    assert_eq!(completed.outcome(), "completed");
    assert_eq!(completed.block()["fallbackReason"], "unqualified_model");
    assert_eq!(completed.block()["policy"], "qualified");
    assert_eq!(
        completed.event("job-9"),
        json!({ "jobId": "job-9", "engine": "mlx", "kvCache": completed.block() })
    );

    let error = decode_native_starvector(
        &StubStarVector { refuse: true },
        &request,
        &progress,
        "mlx",
        &slot,
    )
    .unwrap_err();
    assert!(
        matches!(&error, WorkerError::Engine(message) if message.contains("native StarVector generation failed")),
        "{error:?}"
    );
    let refused = slot.lock().unwrap().clone().expect("refused record");
    assert_eq!(refused.outcome(), "refused");
    assert_eq!(refused.block()["policy"], "qualified");
    assert_eq!(refused.block()["reason"], "request_resource_exhausted");
}

/// A vector adapter whose decode the engine refused: it fails, and reports the refusal's
/// KV-cache record.
struct RefusedVectorProvider;

impl MultimodalVectorProviderAdapter for RefusedVectorProvider {
    fn provider_id(&self) -> &str {
        "starvector"
    }

    fn supports_mode(&self, mode: VectorMode) -> bool {
        mode == VectorMode::TextToSvg
    }

    fn generate_svg(
        &self,
        _request: &VectorProviderRequest,
        _cancel: &gen_core::CancelFlag,
        _progress: tokio::sync::watch::Sender<u32>,
        _on_source: &mut dyn FnMut(&str, u32) -> WorkerResult<()>,
    ) -> WorkerResult<()> {
        Err(classify_starvector_error(
            crate::llm_kv_cache::test_support::refusal(),
        ))
    }

    fn kv_cache_record(&self) -> Option<crate::llm_kv_cache::KvCacheRecord> {
        Some(crate::llm_kv_cache::KvCacheRecord::of(
            "mlx",
            gen_core::core_llm::KvCompressionPolicy::Qualified,
            &Err::<StarVectorOutput, _>(crate::llm_kv_cache::test_support::refusal()),
            |output| output.kv_cache.as_ref(),
        ))
    }
}

/// sc-20688: a vector job whose decode the engine refused still emits its `llm_kv_cache`
/// event, with the requested policy, the outcome and the refusal.
#[test]
fn a_refused_vector_decode_records_its_kv_cache_event() {
    let request = vector_request(VectorMode::TextToSvg, "a compact mark");
    let job_id = "vector-refused";
    let result = collect_svg_source(
        job_id,
        &RefusedVectorProvider,
        &request,
        &gen_core::CancelFlag::new(),
        tokio::sync::watch::channel(0).0,
    );
    assert!(matches!(result, Err(WorkerError::Engine(_))));
    let events = crate::llm_kv_cache::test_support::events_for(job_id);
    assert_eq!(events.len(), 1, "{events:?}");
    assert_eq!(events[0]["engine"], "mlx");
    assert_eq!(events[0]["kvCache"]["policy"], "qualified");
    assert_eq!(events[0]["kvCache"]["outcome"], "refused");
    assert_eq!(events[0]["kvCache"]["reason"], "request_resource_exhausted");
}

#[test]
fn native_hidden_token_source_gaps_reach_product_collection() {
    let request = vector_request(VectorMode::TextToSvg, "a compact mark");
    let (progress, observed) = tokio::sync::watch::channel(0);
    let collected = collect_svg_source(
        "vector-hidden-gaps",
        &NativeSourceSequence([0, 2, 3]),
        &request,
        &gen_core::CancelFlag::new(),
        progress,
    )
    .expect("native hidden-token indices remain valid at the product boundary");
    assert_eq!(collected.source.as_deref(), Some("<svg></svg>"));
    assert_eq!(
        *observed.borrow(),
        3,
        "progress counts actual decoder tokens"
    );
}

#[test]
fn native_source_indices_still_reject_duplicates_regressions_and_missing_prefix() {
    let request = vector_request(VectorMode::TextToSvg, "a compact mark");
    for indices in [[0, 0, 3], [0, 3, 2], [1, 2, 3]] {
        let result = collect_svg_source(
            "vector-index-rejects",
            &NativeSourceSequence(indices),
            &request,
            &gen_core::CancelFlag::new(),
            tokio::sync::watch::channel(0).0,
        );
        assert!(
            matches!(result, Err(WorkerError::Engine(message)) if message.contains("source index")),
            "invalid sequence {indices:?} must remain rejected",
        );
    }
}

#[tokio::test]
async fn vector_progress_is_live_bounded_monotonic_and_private() {
    let (tx, rx) = tokio::sync::watch::channel(0);
    let (release, blocked) = tokio::sync::oneshot::channel();
    let finished = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let completed = finished.clone();
    let generation = async move {
        blocked.await.expect("test releases provider");
        completed.store(true, std::sync::atomic::Ordering::Release);
        Ok(())
    };
    // Exercise the same bounded sink used by the native provider callback. Older counts
    // cannot regress progress, and a bad provider cannot exceed the declared maximum.
    record_vector_token_progress(&tx, 1, 10);
    let mut release = Some(release);
    let mut observations = Vec::new();
    tokio::time::timeout(
        Duration::from_secs(3),
        with_vector_progress(
            rx,
            10,
            gen_core::CancelFlag::new(),
            generation,
            |count, maximum| {
                assert!(
                    !finished.load(std::sync::atomic::Ordering::Acquire),
                    "progress must precede decode completion"
                );
                observations.push(count);
                let payload = vector_token_progress(count, maximum);
                assert_eq!(payload.extra["generatedTokens"], count);
                assert_eq!(payload.extra["maxNewTokens"], 10);
                let encoded = serde_json::to_string(&payload).expect("serializes");
                assert!(!encoded.contains("<svg"));
                if count == 1 {
                    record_vector_token_progress(&tx, 0, 10);
                    assert_eq!(*tx.borrow(), 1);
                    record_vector_token_progress(&tx, 1000, 10);
                } else {
                    release
                        .take()
                        .expect("one completion")
                        .send(())
                        .expect("provider held");
                }
                std::future::ready(Ok(()))
            },
        ),
    )
    .await
    .expect("live progress completes without waiting for provider")
    .expect("progress succeeds");
    assert_eq!(observations, [1, 10]);
    assert!(finished.load(std::sync::atomic::Ordering::Acquire));
}

#[tokio::test]
async fn vector_progress_post_failure_cancels_and_joins_generation() {
    let (tx, rx) = tokio::sync::watch::channel(1);
    let cancel = gen_core::CancelFlag::new();
    let provider_cancel = cancel.clone();
    let joined = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let provider_joined = joined.clone();
    let generation = async move {
        while !provider_cancel.is_cancelled() {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        provider_joined.store(true, std::sync::atomic::Ordering::Release);
        Ok(())
    };
    let result = with_vector_progress(rx, 10, cancel.clone(), generation, |_, _| {
        std::future::ready(Err(WorkerError::Engine("progress unavailable".to_owned())))
    })
    .await;
    drop(tx);
    assert!(result.is_err());
    assert!(cancel.is_cancelled());
    assert!(joined.load(std::sync::atomic::Ordering::Acquire));
}

#[tokio::test]
async fn vector_progress_never_posts_after_cancellation() {
    let (_tx, rx) = tokio::sync::watch::channel(3);
    let cancel = gen_core::CancelFlag::new();
    cancel.cancel();
    let result: WorkerResult<()> = with_vector_progress(
        rx,
        10,
        cancel,
        async {
            tokio::time::sleep(Duration::from_millis(20)).await;
            Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned()))
        },
        |_, _| {
            panic!("cancelled generation must not post Running progress");
            #[allow(unreachable_code)]
            std::future::ready(Ok(()))
        },
    )
    .await;
    assert!(matches!(result, Err(WorkerError::Canceled(_))));
}

#[test]
fn native_request_is_image_only_and_rejects_text_guidance() {
    let image = ImageRef::new(1, 1, vec![1, 2, 3]).expect("valid RGB pixel");
    let mut provider_request = vector_request(VectorMode::ImageToSvg, "");
    provider_request.detail_budget.max_wall_time_ms = 300_000;
    let request = native_starvector_request(
        &provider_request,
        image.clone(),
        gen_core::core_llm::CancelFlag::new(),
    )
    .expect("image-only request");
    assert_eq!(request.text_request.messages.len(), 1);
    assert_eq!(request.text_request.messages[0].role, Role::User);
    assert!(matches!(
        request.text_request.messages[0].content.as_slice(),
        [Content::Image(actual)] if actual == &image
    ));
    assert!(!request.has_text());
    assert_eq!(request.max_wall_time, Duration::from_secs(300));

    let error = native_starvector_request(
        &vector_request(VectorMode::ImageToSvg, "undisclosed guidance"),
        image,
        gen_core::core_llm::CancelFlag::new(),
    )
    .expect_err("StarVector image-to-SVG must remain image-only");
    assert!(matches!(error, WorkerError::InvalidPayload(_)));
}

#[test]
fn exact_starvector_manifest_identity_is_fail_closed() {
    let identity = starvector_model_identity("starvector_1b").expect("known identity");
    let exact = serde_json::json!({
        "id": identity.model_id,
        "type": "vector",
        "adapter": "starvector",
        "downloads": [{
            "repo": identity.repository,
            "revision": identity.revision
        }]
    });
    manifest_binds_starvector_identity(&exact, identity).expect("exact immutable identity");

    for crossed in [
        serde_json::json!({
            "id": "starvector_8b",
            "type": "vector",
            "adapter": "starvector",
            "downloads": [{"repo": identity.repository, "revision": identity.revision}]
        }),
        serde_json::json!({
            "id": identity.model_id,
            "type": "vector",
            "adapter": "starvector",
            "downloads": [{"repo": identity.repository, "revision": "main"}]
        }),
        serde_json::json!({
            "id": identity.model_id,
            "type": "vector",
            "adapter": "starvector",
            "downloads": [{
                "repo": identity.repository,
                "revision": identity.revision,
                "coRequisite": true
            }]
        }),
    ] {
        assert!(manifest_binds_starvector_identity(&crossed, identity).is_err());
    }
    assert!(starvector_model_identity("unregistered-starvector").is_err());
}

#[cfg(native_lane)]
#[test]
fn cold_starvector_accepts_only_exact_guard_verified_publisher_source() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("models/portable-attempts/starvector");
    std::fs::create_dir_all(&path).unwrap();
    std::fs::write(path.join("model.safetensors"), b"synthetic path fixture").unwrap();
    let identity = starvector_model_identity("starvector_1b").unwrap();
    let mut settings = Settings::for_test(temp.path().to_path_buf());
    settings.gpu_id = if cfg!(target_os = "macos") {
        "mlx".to_owned()
    } else {
        "gpu-0".to_owned()
    };
    settings.backend_mlx_enabled = true;
    settings.backend_candle_enabled = true;
    active_starvector_backend(&settings).unwrap();
    let payload = VectorJobPayload {
        project_id: "project".to_owned(),
        mode: VectorMode::ImageToSvg,
        model: identity.model_id.to_owned(),
        source_asset_id: None,
        prompt: String::new(),
        sampling: vector_request(VectorMode::ImageToSvg, "").sampling,
        detail_budget: vector_request(VectorMode::ImageToSvg, "").detail_budget,
        model_manifest_entry: json!({
            "id": identity.model_id,
            "type": "vector",
            "adapter": "starvector",
            "downloads": [{"repo": identity.repository, "revision": identity.revision}],
            "vector": {"providers": {
                "mlx": {"available": true, "id": identity.mlx_provider_id},
                "candle": {"available": true, "id": identity.candle_provider_id}
            }}
        }),
        workflow: None,
    };
    assert!(NativeStarVectorProvider::resolve(&settings, &payload).is_err());
    settings.verified_revision_sources.push(
        crate::external_library_runtime::VerifiedRevisionSource {
            model_id: identity.model_id.to_owned(),
            repository: identity.repository.to_owned(),
            revision: identity.revision.to_owned(),
            path: path.clone(),
        },
    );
    assert_eq!(
        NativeStarVectorProvider::resolve(&settings, &payload)
            .unwrap()
            .weights_dir,
        path
    );
    settings.verified_revision_sources[0].repository = "other/repo".to_owned();
    assert!(NativeStarVectorProvider::resolve(&settings, &payload).is_err());
}

#[test]
fn native_terminal_events_publish_only_complete_root_or_eos() {
    let svg = "<svg/>";
    for reason in [
        StarVectorFinishReason::CompleteRoot,
        StarVectorFinishReason::Eos,
    ] {
        let (output, events) = terminal_fixture(reason, svg, Some(svg));
        assert_eq!(
            validate_native_starvector_generation(output, events)
                .expect("complete document is publishable"),
            vec![(svg.to_owned(), 0)]
        );
    }

    for reason in [
        StarVectorFinishReason::TokenLimit,
        StarVectorFinishReason::ByteLimit,
        StarVectorFinishReason::WallTimeLimit,
    ] {
        let (output, events) = terminal_fixture(reason, "<svg>", None);
        assert!(
            validate_native_starvector_generation(output, events)
                .expect("typed bounded outcome")
                .is_empty(),
            "bounded partial output must never reach publication"
        );
    }

    let (output, events) = terminal_fixture(StarVectorFinishReason::Cancelled, "<svg>", None);
    assert!(validate_native_starvector_generation(output, events)
        .expect("typed cancellation outcome")
        .is_empty());
    assert_eq!(
        terminal_finish_reason(StarVectorFinishReason::TokenLimit),
        "token_limit"
    );
}

#[test]
fn terminal_evidence_never_advertises_files_for_a_bounded_outcome() {
    let terminal = TerminalProviderOutcome {
        finish_reason: "token_limit",
        generated_tokens: 7,
        generated_bytes: 23,
        latency_seconds: 0.01,
        provider_id: "mlx-starvector-1b".to_owned(),
        model_id: "starvector_1b",
        model_repository: "starvector/starvector-1b-im2svg",
        model_revision: "380ab95d25a8e9ab1dc825debe238b4953ae13b9",
        backend: "mlx",
    };
    let result = terminal_generation_limit_result(
        &terminal,
        0.25,
        (Path::new("provider-terminal.json"), b"transcript"),
    )
    .expect("typed generation rejection");
    assert_eq!(result["accepted"], false);
    assert_eq!(result["outcome"], "rejected");
    assert_eq!(result["rejectionStage"], "generation_limit");
    assert_eq!(result["rejectionCode"], "token_limit");
    assert_eq!(result["finishReason"], "token_limit");
    assert_ne!(result["providerTranscriptSha256"], Value::Null);
    assert!(result["canonicalSvgPath"].is_null());
    assert!(result["previewPngPath"].is_null());
    assert!(result["comparisonPngPath"].is_null());
    assert_eq!(result["latencySeconds"], 0.01);
    assert_eq!(result["providerLatencySeconds"], 0.01);
    assert_eq!(result["productLatencySeconds"], 0.25);
    assert_eq!(result["providerTranscriptPath"], "provider-terminal.json");
    assert_eq!(result["resultContainsInlineSvg"], false);

    let cancelled = TerminalProviderOutcome {
        finish_reason: "cancelled",
        ..terminal.clone()
    };
    assert!(terminal_generation_limit_result(
        &cancelled,
        0.25,
        (Path::new("provider-terminal.json"), b"transcript"),
    )
    .is_err());
}

#[test]
fn accepted_terminal_evidence_binds_product_and_comparison_renders_separately() {
    let terminal = TerminalProviderOutcome {
        finish_reason: "complete_root",
        generated_tokens: 7,
        generated_bytes: 23,
        latency_seconds: 0.01,
        provider_id: "mlx-starvector-1b".to_owned(),
        model_id: "starvector_1b",
        model_repository: "starvector/starvector-1b-im2svg",
        model_revision: "380ab95d25a8e9ab1dc825debe238b4953ae13b9",
        backend: "mlx",
    };
    let result = terminal_result(
        &terminal,
        Some(0.25),
        Some((Path::new("vector.svg"), b"canonical")),
        Some((Path::new("preview.png"), b"intrinsic")),
        Some((Path::new("comparison-512.png"), b"fixed")),
        Some((Path::new("provider-terminal.json"), b"transcript")),
    );
    assert_eq!(result["previewPngPath"], "preview.png");
    assert_eq!(result["previewPngSha256"], sha256_hex(b"intrinsic"));
    assert_eq!(result["comparisonPngPath"], "comparison-512.png");
    assert_eq!(result["comparisonPngSha256"], sha256_hex(b"fixed"));
    assert_ne!(result["previewPngSha256"], result["comparisonPngSha256"]);
    assert_eq!(result["latencySeconds"], result["providerLatencySeconds"]);
    assert_eq!(result["productLatencySeconds"], 0.25);
}

#[test]
fn native_terminal_events_require_one_consistent_final_done() {
    let svg = "<svg/>";
    let (output, mut events) =
        terminal_fixture(StarVectorFinishReason::CompleteRoot, svg, Some(svg));
    events.pop();
    assert!(validate_native_starvector_generation(output.clone(), events).is_err());

    let (output, mut events) =
        terminal_fixture(StarVectorFinishReason::CompleteRoot, svg, Some(svg));
    events.push(StarVectorStreamEvent::Done {
        finish_reason: output.finish_reason,
        generated_tokens: output.generated_tokens,
        generated_bytes: output.generated_bytes,
    });
    assert!(validate_native_starvector_generation(output.clone(), events).is_err());

    let (output, mut events) =
        terminal_fixture(StarVectorFinishReason::CompleteRoot, svg, Some(svg));
    if let Some(StarVectorStreamEvent::Done {
        generated_bytes, ..
    }) = events.last_mut()
    {
        *generated_bytes += 1;
    }
    assert!(validate_native_starvector_generation(output, events).is_err());
}

#[test]
fn cancellation_during_streamed_svg_never_reaches_publication() {
    let request = vector_request(VectorMode::TextToSvg, "a compact mark");
    let cancel = gen_core::CancelFlag::new();
    let output = collect_svg_source(
        "vector-canceled",
        &CancelingProvider,
        &request,
        &cancel,
        tokio::sync::watch::channel(0).0,
    );
    assert!(matches!(output, Err(WorkerError::Canceled(_))));

    // This is the exact publication boundary used by `run_vector_job_with_provider`: staging
    // starts only after `collect_svg_source` returns Ok. A canceled stream has no source to
    // sanitize, render, or rename, so neither member of the asset pair can exist.
    let temp = tempfile::tempdir().expect("temp dir");
    let published = temp.path().join("asset");
    if let Ok(CollectedSvgSource {
        source: Some(source),
        ..
    }) = output
    {
        std::fs::create_dir_all(&published).expect("publication dir");
        std::fs::write(published.join("vector.svg"), source).expect("source writes");
    }
    assert!(!published.exists());
}

include!("svg_sanitize_tests.rs");

#[tokio::test]
async fn terminal_seam_delegates_to_sanitizer_and_publishes_only_the_pair() {
    assert!(terminal_sanitize_svg_bytes(&[0xff]).is_err());
    let value = terminal_sanitize_svg_bytes(
        b"<svg width=\"12\" height=\"8\" xmlns=\"http://www.w3.org/2000/svg\"><rect width=\"12\" height=\"8\" fill=\"#00ff00\"/></svg>",
    )
    .expect("valid inert SVG");
    let temp = tempfile::tempdir().expect("temp dir");
    let destination = temp.path().join("published");
    let (svg, preview) = terminal_write_sanitized_pair(&value, &destination)
        .await
        .expect("atomic terminal publication");
    assert_eq!(
        svg.file_name().and_then(|name| name.to_str()),
        Some("canonical.svg")
    );
    assert_eq!(
        preview.file_name().and_then(|name| name.to_str()),
        Some("preview.png")
    );
    assert!(svg.is_file() && preview.is_file());
    assert!(
        std::fs::read_dir(temp.path())
            .expect("read temp")
            .all(|entry| !entry
                .expect("entry")
                .file_name()
                .to_string_lossy()
                .contains(".tmp")),
        "terminal publication leaves no staging residue"
    );
}
