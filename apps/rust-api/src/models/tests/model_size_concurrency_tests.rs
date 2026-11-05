use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

#[tokio::test]
async fn bounded_collection_caps_concurrency_and_preserves_order() {
    let active = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let output = collect_bounded_ordered(0..12, 3, |value| {
        let active = active.clone();
        let peak = peak.clone();
        async move {
            let current = active.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(current, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(5)).await;
            active.fetch_sub(1, Ordering::SeqCst);
            value
        }
    })
    .await;

    assert_eq!(output, (0..12).collect::<Vec<_>>());
    assert_eq!(peak.load(Ordering::SeqCst), 3);
}

#[test]
fn builtin_download_contexts_fit_in_size_cache() {
    let manifest = sceneworks_core::builtin_manifests::builtin_manifest("builtin.models.jsonc")
        .expect("valid embedded model catalog");
    // The cache is keyed by `(repo, files)`, so shared model rows deliberately collapse to one
    // entry. Assert the invariant the production scan needs instead of sealing today's catalog
    // population: every platform's distinct contexts fit together and remain addressable.
    for os in ["macos", "windows", "linux"] {
        let mut keys = std::collections::HashSet::new();
        for mut model in manifest["models"]
            .as_array()
            .expect("models array")
            .iter()
            .cloned()
        {
            retain_downloads_for_os(&mut model, os);
            if let Some(context) = model_download_context(&model).expect("download context") {
                keys.insert((context.repo, context.files));
            }
        }
        assert!(!keys.is_empty(), "{os} has no builtin download contexts");
        assert!(
            keys.len() <= MODEL_SIZE_CACHE_LIMIT,
            "{os} builtin catalog has {} distinct download contexts but cache holds only {}",
            keys.len(),
            MODEL_SIZE_CACHE_LIMIT
        );

        let keys: Vec<_> = keys.into_iter().collect();
        let mut cache = ModelSizeCache::default();
        for (index, key) in keys.iter().cloned().enumerate() {
            cache.insert(key, index as u64 + 1);
        }
        assert_eq!(cache.entries.len(), keys.len());
        for (index, key) in keys.iter().enumerate() {
            assert_eq!(
                cache.get(key),
                Some(Some(index as u64 + 1)),
                "{os} context {key:?} was evicted from a cache that should fit the catalog"
            );
        }
    }
}

#[test]
fn successful_size_estimates_have_a_bounded_positive_ttl() {
    let mut cache = ModelSizeCache::default();
    let key = ("owner/model".to_owned(), vec!["*.safetensors".to_owned()]);
    let before = std::time::Instant::now();
    cache.insert(key.clone(), 1234);
    let expires_at = cache
        .entries
        .get(&key)
        .and_then(|entry| entry.expires_at)
        .expect("successful estimate expires");
    assert!(expires_at > before);
    assert!(expires_at <= before + MODEL_SIZE_POSITIVE_TTL + Duration::from_secs(1));
}

#[test]
fn in_flight_key_registry_is_bounded() {
    let mut cache = ModelSizeCache::default();
    for index in 0..MODEL_SIZE_CACHE_LIMIT {
        let key = (format!("owner/model-{index}"), Vec::new());
        assert!(matches!(
            cache.lookup_or_start(&key),
            ModelSizeLookup::Lead(_)
        ));
    }
    let overflow = ("owner/overflow".to_owned(), Vec::new());
    assert!(matches!(
        cache.lookup_or_start(&overflow),
        ModelSizeLookup::Unshared
    ));
    assert_eq!(cache.in_flight.len(), MODEL_SIZE_CACHE_LIMIT);
}
