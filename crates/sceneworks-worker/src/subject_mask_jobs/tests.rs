use super::*;

fn test_settings(data_dir: &Path) -> Settings {
    Settings {
        api_url: "http://127.0.0.1".to_owned(),
        access_token: None,
        data_dir: data_dir.to_path_buf(),
        config_dir: data_dir.join("config"),
        worker_id: "test-worker".to_owned(),
        gpu_id: "gpu-0".to_owned(),
        is_child_worker: false,
        poll_seconds: 1,
        heartbeat_seconds: 1,
        shutdown_timeout_seconds: 1,
        huggingface_base_url: DEFAULT_HUGGINGFACE_BASE_URL.to_owned(),
        huggingface_token: None,
        credentials: Vec::new(),
        max_lora_url_bytes: DEFAULT_MAX_LORA_URL_BYTES,
        max_model_url_bytes: DEFAULT_MAX_MODEL_URL_BYTES,
        allow_private_lora_urls: false,
        utility_workers: 1,
        backend_mlx_enabled: true,
        backend_candle_enabled: false,
        gpu_memory_limit_bytes: 0,
        external_model_roots: Vec::new(),
    }
}

/// `count` distinct images of `w`×`h` on disk, as job items.
fn items_on_disk(dir: &Path, count: usize, w: u32, h: u32) -> Vec<SubjectMaskItem> {
    (0..count)
        .map(|index| {
            let image_path = dir.join(format!("item_{index}.png"));
            image::RgbImage::from_pixel(w, h, image::Rgb([index as u8 * 50, 0, 0]))
                .save(&image_path)
                .expect("write image");
            SubjectMaskItem {
                image_path,
                content_hash: format!("hash_{index}"),
            }
        })
        .collect()
}

/// A person mask covering the left `cols` columns of a `w`×`h` image.
fn left_block(w: u32, h: u32, cols: u32) -> Vec<u8> {
    (0..h)
        .flat_map(|_| (0..w).map(move |x| if x < cols { 255 } else { 0 }))
        .collect()
}

fn decode(png: &[u8]) -> image::DynamicImage {
    image::load_from_memory(png).expect("mask PNG decodes")
}

#[test]
fn n_images_yield_n_single_channel_masks_at_image_size() {
    let dir = tempfile::tempdir().unwrap();
    let items = items_on_disk(dir.path(), 4, 6, 3);
    let (tx, mut rx) = tokio::sync::mpsc::channel(16);
    let mut calls = 0;
    let records = generate_subject_masks(items, CancelFlag::new(), tx, |image| {
        calls += 1;
        let (w, h) = image.dimensions();
        Ok(vec![left_block(w, h, 2)])
    })
    .expect("masks generate");
    assert_eq!(calls, 4, "the segmenter ran once per image");
    assert_eq!(records.len(), 4);
    for (index, record) in records.iter().enumerate() {
        assert_eq!(record.content_hash, format!("hash_{index}"));
        assert!(!record.empty);
        let mask = decode(&record.png);
        assert_eq!(mask.color(), image::ColorType::L8);
        assert_eq!((mask.width(), mask.height()), (6, 3));
        assert_eq!(
            mask.to_luma8().into_raw(),
            left_block(6, 3, 2),
            "white = subject"
        );
    }
    let mut reported = Vec::new();
    while let Ok(index) = rx.try_recv() {
        reported.push(index);
    }
    assert_eq!(reported, vec![0, 1, 2, 3], "per-item progress");
}

#[test]
fn an_image_with_no_person_gets_an_all_black_mask_flagged_empty() {
    let dir = tempfile::tempdir().unwrap();
    let items = items_on_disk(dir.path(), 2, 5, 4);
    let (tx, _rx) = tokio::sync::mpsc::channel(16);
    let mut index = 0;
    let records = generate_subject_masks(items, CancelFlag::new(), tx, |image| {
        index += 1;
        let (w, h) = image.dimensions();
        // Second image: SAM3 found nobody.
        Ok(if index == 2 {
            Vec::new()
        } else {
            vec![left_block(w, h, 1)]
        })
    })
    .unwrap();
    assert!(!records[0].empty);
    assert!(records[1].empty);
    let mask = decode(&records[1].png).to_luma8();
    assert_eq!(
        mask.dimensions(),
        (5, 4),
        "an empty mask still covers the image"
    );
    assert!(mask.as_raw().iter().all(|&value| value == 0));
}

#[test]
fn every_detected_person_is_unioned_into_the_subject() {
    let a = vec![255, 0, 0, 0];
    let b = vec![0, 0, 255, 0];
    assert_eq!(
        union_person_masks(&[a, b], 2, 2).unwrap(),
        vec![255, 0, 255, 0]
    );
    assert_eq!(union_person_masks(&[], 2, 2).unwrap(), vec![0; 4]);
    let error = union_person_masks(&[vec![255; 3]], 2, 2).expect_err("short mask refused");
    assert!(matches!(error, WorkerError::Engine(_)), "{error}");
}

#[test]
fn a_pre_tripped_cancel_stops_before_segmenting() {
    let dir = tempfile::tempdir().unwrap();
    let items = items_on_disk(dir.path(), 2, 2, 2);
    let (tx, _rx) = tokio::sync::mpsc::channel(16);
    let cancel = CancelFlag::new();
    cancel.cancel();
    let error = generate_subject_masks(items, cancel, tx, |_| {
        panic!("the segmenter must not run after cancel")
    })
    .expect_err("canceled");
    assert!(matches!(error, WorkerError::Canceled(_)), "{error}");
}

#[test]
fn records_payload_carries_base64_png_keyed_by_content_hash() {
    use base64::Engine as _;
    let png = encode_mask_png(vec![0, 255], 2, 1).unwrap();
    let payload = subject_mask_records_payload(&[SubjectMaskRecord {
        content_hash: "abc".to_owned(),
        png: png.clone(),
        empty: false,
    }]);
    assert_eq!(payload[0]["contentHash"], "abc");
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(payload[0]["maskPng"].as_str().unwrap())
        .unwrap();
    assert_eq!(decoded, png);
}

#[test]
fn items_require_hash_and_stay_inside_the_dataset_root() {
    let dir = tempfile::tempdir().unwrap();
    let settings = test_settings(dir.path());
    let dataset_root = dir.path().join("datasets").join("ds-1");
    let inside = dataset_root.join("images").join("a.png");
    let payload = |image: &Path, hash: Value| {
        serde_json::Map::from_iter([
            (
                "datasetRoot".to_owned(),
                json!(dataset_root.display().to_string()),
            ),
            (
                "items".to_owned(),
                json!([{ "itemId": "item_1", "imagePath": image.display().to_string(), "contentHash": hash }]),
            ),
        ])
    };
    let items = subject_mask_items(&settings, &payload(&inside, json!("h1"))).expect("parses");
    assert_eq!(items[0].content_hash, "h1");
    assert!(items[0].image_path.ends_with("datasets/ds-1/images/a.png"));

    let error = subject_mask_items(&settings, &payload(&inside, Value::Null)).expect_err("no hash");
    assert!(error.to_string().contains("missing contentHash"), "{error}");

    let error = subject_mask_items(
        &settings,
        &payload(&dir.path().join("outside.png"), json!("h1")),
    )
    .expect_err("outside the dataset root");
    assert!(
        error
            .to_string()
            .contains("Subject mask item item_1 imagePath"),
        "{error}"
    );
}
