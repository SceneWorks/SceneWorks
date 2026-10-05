use super::*;

use image::{GrayAlphaImage, LumaA, RgbImage};

use crate::time::utc_now;
use crate::training::{
    Caption, CaptionSource, TrainingDatasetStatus, TrainingModality,
    TRAINING_CONTRACT_SCHEMA_VERSION,
};
use crate::training_store::ensure_training_dataset_table;

const PROJECT: &str = "proj";
const DATASET: &str = "ds_masks";

/// A project holding one dataset of `count` distinct `w`×`h` PNG images, each with its real content
/// hash recorded — the exact state `create_dataset` leaves behind.
struct Fixture {
    _dir: tempfile::TempDir,
    project_path: PathBuf,
    store: TrainingDatasetStore,
}

impl Fixture {
    fn new(count: usize, w: u32, h: u32) -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        let project_path = dir.path().to_path_buf();
        ensure_training_dataset_table(&project_path).expect("dataset table");
        let root = dataset_root(&project_path, DATASET);
        fs::create_dir_all(root.join("images")).expect("images dir");
        let items = (0..count)
            .map(|index| {
                let id = format!("item_{index}");
                let path = format!("images/{id}.png");
                // Distinct pixels per item → distinct content hashes.
                RgbImage::from_pixel(w, h, image::Rgb([index as u8 * 40, 10, 20]))
                    .save(root.join(&path))
                    .expect("write image");
                let content_hash =
                    crate::media_convert::file_content_hash(&root.join(&path)).expect("hash");
                TrainingDatasetItem {
                    id: id.clone(),
                    asset_id: None,
                    path,
                    control_image_path: None,
                    display_name: id,
                    caption: Caption {
                        text: String::new(),
                        source: CaptionSource::Manual,
                        trigger_words: Vec::new(),
                        updated_at: None,
                        extra: Default::default(),
                    },
                    width: Some(w),
                    height: Some(h),
                    content_hash: Some(content_hash),
                    tier0_scalars: None,
                    quality_ack: None,
                    added_at: utc_now(),
                    extra: Default::default(),
                }
            })
            .collect();
        let dataset = TrainingDataset {
            schema_version: TRAINING_CONTRACT_SCHEMA_VERSION,
            id: DATASET.to_owned(),
            version: 1,
            project_id: Some(PROJECT.to_owned()),
            character_id: None,
            name: "Masks".to_owned(),
            modality: TrainingModality::Image,
            status: TrainingDatasetStatus::Draft,
            created_at: utc_now(),
            updated_at: utc_now(),
            items,
            extra: Default::default(),
        };
        let store = TrainingDatasetStore::new(project_path.clone());
        store.save_dataset(&dataset).expect("save dataset");
        Self {
            _dir: dir,
            project_path,
            store,
        }
    }

    fn dataset(&self) -> TrainingDataset {
        self.store.get_dataset(PROJECT, DATASET).expect("dataset")
    }

    fn root(&self) -> PathBuf {
        dataset_root(&self.project_path, DATASET)
    }

    fn hash(&self, index: usize) -> String {
        self.dataset().items[index]
            .content_hash
            .clone()
            .expect("hash")
    }
}

fn png(mask: &GrayImage) -> Vec<u8> {
    let mut out = Vec::new();
    mask.write_to(&mut Cursor::new(&mut out), ImageFormat::Png)
        .expect("encode");
    out
}

/// A mask with a white block in the top-left quadrant.
fn subject_mask(w: u32, h: u32) -> GrayImage {
    GrayImage::from_fn(w, h, |x, y| {
        image::Luma([if x < w / 2 && y < h / 2 { 255 } else { 0 }])
    })
}

fn field_code(error: ProjectStoreError) -> (&'static str, &'static str) {
    match error {
        ProjectStoreError::FieldInvalid { field, code, .. } => (field, code),
        other => panic!("expected a field-level error, got {other:?}"),
    }
}

#[test]
fn writing_n_masks_reports_full_coverage_and_resolves_every_item() {
    let fx = Fixture::new(3, 8, 6);
    let writes = (0..3)
        .map(|index| SubjectMaskWrite {
            content_hash: fx.hash(index),
            png: png(&subject_mask(8, 6)),
        })
        .collect();
    let report = fx
        .store
        .write_subject_masks(PROJECT, DATASET, writes, SubjectMaskSource::Auto)
        .expect("write masks");
    assert_eq!((report.masked, report.total), (3, 3));
    assert_eq!(report.empty, 0);
    let dataset = fx.dataset();
    for (item, status) in dataset.items.iter().zip(&report.items) {
        let path = resolve_subject_mask_path(&fx.root(), item).expect("mask resolves");
        assert_eq!(
            path,
            fx.root().join(format!(
                "masks/{}.png",
                item.content_hash.as_deref().unwrap()
            ))
        );
        let stored = image::open(&path).expect("decode stored mask");
        assert_eq!(stored.color(), image::ColorType::L8, "single-channel PNG");
        assert_eq!(
            (stored.width(), stored.height()),
            (8, 6),
            "same size as the image"
        );
        assert_eq!(
            status.mask_path.as_deref(),
            Some(
                format!(
                    "training/datasets/{DATASET}/masks/{}.png",
                    item.content_hash.as_deref().unwrap()
                )
                .as_str()
            )
        );
        assert_eq!(status.source, Some(SubjectMaskSource::Auto));
    }
    // A fresh read reports the same coverage (persisted, not just returned).
    let reread = fx
        .store
        .subject_mask_report(PROJECT, DATASET)
        .expect("report");
    assert_eq!((reread.masked, reread.total), (3, 3));
}

#[test]
fn an_all_black_mask_is_recorded_and_reported_empty() {
    let fx = Fixture::new(2, 4, 4);
    let writes = vec![
        SubjectMaskWrite {
            content_hash: fx.hash(0),
            png: png(&subject_mask(4, 4)),
        },
        SubjectMaskWrite {
            content_hash: fx.hash(1),
            png: png(&GrayImage::new(4, 4)),
        },
    ];
    let report = fx
        .store
        .write_subject_masks(PROJECT, DATASET, writes, SubjectMaskSource::Auto)
        .expect("write masks");
    assert_eq!((report.masked, report.empty), (2, 1));
    assert!(!report.items[0].empty);
    assert!(report.items[1].empty && report.items[1].has_mask);
}

/// sc-24828: the masked-loss lookup keys an image by its current bytes and reports a stored mask
/// as Present, an all-black one as Empty, and an unmasked / replaced / file-less one as Missing.
#[test]
fn lookup_by_image_bytes_reports_present_empty_and_missing() {
    let fx = Fixture::new(4, 4, 4);
    let image = |index: usize| fx.root().join(format!("images/item_{index}.png"));
    let writes = vec![
        SubjectMaskWrite {
            content_hash: fx.hash(0),
            png: png(&subject_mask(4, 4)),
        },
        SubjectMaskWrite {
            content_hash: fx.hash(1),
            png: png(&GrayImage::new(4, 4)),
        },
        SubjectMaskWrite {
            content_hash: fx.hash(3),
            png: png(&subject_mask(4, 4)),
        },
    ];
    fx.store
        .write_subject_masks(PROJECT, DATASET, writes, SubjectMaskSource::Auto)
        .unwrap();
    let index = read_subject_mask_index_at(&fx.root()).unwrap();
    let lookup = |index_ref: &DatasetSubjectMasks, i: usize| {
        lookup_subject_mask_for_image(&fx.root(), index_ref, &image(i)).unwrap()
    };
    assert_eq!(
        lookup(&index, 0),
        SubjectMaskLookup::Present(fx.root().join(subject_mask_relative_path(&fx.hash(0))))
    );
    assert_eq!(lookup(&index, 1), SubjectMaskLookup::Empty);
    assert_eq!(lookup(&index, 2), SubjectMaskLookup::Missing);
    // Item 3's image is repainted after its mask was written: the stale mask must not match.
    RgbImage::from_pixel(4, 4, image::Rgb([1, 2, 3]))
        .save(image(3))
        .unwrap();
    assert_eq!(lookup(&index, 3), SubjectMaskLookup::Missing);
    // A record whose file is gone is Missing, like the coverage report.
    fs::remove_file(fx.root().join(subject_mask_relative_path(&fx.hash(0)))).unwrap();
    assert_eq!(lookup(&index, 0), SubjectMaskLookup::Missing);
    // No index at all ⇒ an empty index, not an error.
    let bare = tempfile::tempdir().unwrap();
    assert!(read_subject_mask_index_at(bare.path())
        .unwrap()
        .masks
        .is_empty());
}

#[test]
fn a_record_whose_mask_file_is_gone_does_not_count_as_coverage() {
    let fx = Fixture::new(1, 4, 4);
    fx.store
        .write_subject_masks(
            PROJECT,
            DATASET,
            vec![SubjectMaskWrite {
                content_hash: fx.hash(0),
                png: png(&subject_mask(4, 4)),
            }],
            SubjectMaskSource::Auto,
        )
        .unwrap();
    fs::remove_file(fx.root().join(subject_mask_relative_path(&fx.hash(0)))).unwrap();
    let report = fx.store.subject_mask_report(PROJECT, DATASET).unwrap();
    assert_eq!(
        report.masked, 0,
        "coverage describes what the trainer can read"
    );
    assert!(!report.items[0].has_mask);
}

#[test]
fn a_report_without_masks_is_zero_coverage() {
    let fx = Fixture::new(2, 4, 4);
    let report = fx
        .store
        .subject_mask_report(PROJECT, DATASET)
        .expect("report");
    assert_eq!((report.masked, report.total), (0, 2));
    assert!(report
        .items
        .iter()
        .all(|item| !item.has_mask && item.mask_path.is_none()));
    assert!(fx
        .dataset()
        .items
        .iter()
        .all(|item| resolve_subject_mask_path(&fx.root(), item).is_none()));
}

#[test]
fn an_uploaded_replacement_mask_is_persisted_and_marked_upload() {
    let fx = Fixture::new(2, 8, 8);
    fx.store
        .write_subject_masks(
            PROJECT,
            DATASET,
            vec![SubjectMaskWrite {
                content_hash: fx.hash(0),
                png: png(&GrayImage::new(8, 8)),
            }],
            SubjectMaskSource::Auto,
        )
        .expect("auto mask");
    let upload = fx.project_path.join("upload.png");
    fs::write(
        &upload,
        png(&GrayImage::from_pixel(8, 8, image::Luma([255]))),
    )
    .expect("upload");
    let report = fx
        .store
        .upload_subject_mask(PROJECT, DATASET, "item_0", &upload)
        .expect("upload mask");
    assert_eq!(report.items[0].source, Some(SubjectMaskSource::Upload));
    assert!(
        !report.items[0].empty,
        "replacement replaced the empty auto mask"
    );
    assert_eq!(report.uploaded, 1);
    let stored =
        image::open(resolve_subject_mask_path(&fx.root(), &fx.dataset().items[0]).unwrap())
            .unwrap()
            .to_luma8();
    assert!(stored.as_raw().iter().all(|&value| value == 255));
}

#[test]
fn a_same_aspect_mask_is_resized_to_the_image_and_a_different_aspect_is_refused() {
    let fx = Fixture::new(1, 8, 4);
    let upload = fx.project_path.join("upload.png");
    // Half size, same 2:1 aspect → resized to 8x4.
    fs::write(
        &upload,
        png(&GrayImage::from_pixel(4, 2, image::Luma([255]))),
    )
    .unwrap();
    let report = fx
        .store
        .upload_subject_mask(PROJECT, DATASET, "item_0", &upload)
        .expect("same-aspect mask accepted");
    assert!(report.items[0].has_mask);
    let stored =
        image::open(resolve_subject_mask_path(&fx.root(), &fx.dataset().items[0]).unwrap())
            .unwrap();
    assert_eq!((stored.width(), stored.height()), (8, 4));

    // Square mask for a 2:1 image → refused on the upload field.
    fs::write(&upload, png(&GrayImage::new(4, 4))).unwrap();
    let error = fx
        .store
        .upload_subject_mask(PROJECT, DATASET, "item_0", &upload)
        .expect_err("aspect mismatch refused");
    assert_eq!(
        field_code(error),
        ("file", "subject_mask_dimensions_mismatch")
    );
}

#[test]
fn an_unsupported_or_oversized_upload_is_a_field_error() {
    let fx = Fixture::new(1, 4, 4);
    let upload = fx.project_path.join("upload.bin");
    fs::write(&upload, b"definitely not an image").unwrap();
    let error = fx
        .store
        .upload_subject_mask(PROJECT, DATASET, "item_0", &upload)
        .expect_err("garbage refused");
    assert_eq!(field_code(error), ("file", "subject_mask_unsupported_type"));

    // One byte over the limit (sparse file: no real allocation).
    let file = fs::File::create(&upload).unwrap();
    file.set_len(SUBJECT_MASK_MAX_UPLOAD_BYTES + 1).unwrap();
    let error = fx
        .store
        .upload_subject_mask(PROJECT, DATASET, "item_0", &upload)
        .expect_err("oversized refused");
    assert_eq!(field_code(error), ("file", "subject_mask_too_large"));
    // Nothing was stored by either rejected upload.
    assert_eq!(
        fx.store
            .subject_mask_report(PROJECT, DATASET)
            .unwrap()
            .masked,
        0
    );
}

#[test]
fn a_transparent_layer_mask_reads_alpha_as_background() {
    let mask = GrayAlphaImage::from_fn(2, 1, |x, _| {
        if x == 0 {
            LumaA([255, 255])
        } else {
            LumaA([255, 0])
        }
    });
    let mut bytes = Vec::new();
    mask.write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png)
        .unwrap();
    let (gray, empty) = normalize_subject_mask(&bytes, (2, 1)).expect("normalize");
    assert_eq!(gray.as_raw(), &vec![255, 0]);
    assert!(!empty);
}

fn mask_files(fx: &Fixture) -> Vec<String> {
    let mut names: Vec<String> = match fs::read_dir(fx.root().join(SUBJECT_MASK_DIR)) {
        Ok(entries) => entries
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(error) => panic!("read masks dir: {error}"),
    };
    names.sort();
    names
}

#[test]
fn an_uploaded_mask_for_an_unknown_image_is_refused_before_anything_is_written() {
    let fx = Fixture::new(1, 4, 4);
    let error = fx
        .store
        .write_subject_masks(
            PROJECT,
            DATASET,
            vec![
                SubjectMaskWrite {
                    content_hash: fx.hash(0),
                    png: png(&subject_mask(4, 4)),
                },
                SubjectMaskWrite {
                    content_hash: "0".repeat(64),
                    png: png(&GrayImage::new(4, 4)),
                },
            ],
            SubjectMaskSource::Upload,
        )
        .expect_err("unknown hash refused");
    assert!(matches!(error, ProjectStoreError::NotFound(_)), "{error:?}");
    // Validate-then-write: the valid first mask was not stored ahead of the failing second one.
    assert_eq!(mask_files(&fx), Vec::<String>::new());
}

#[test]
fn a_generated_mask_for_a_vanished_image_is_skipped_and_the_live_one_stored() {
    let fx = Fixture::new(2, 4, 4);
    let live = fx.hash(0);
    let removed = fx.hash(1);
    // The image leaves the dataset while the GPU job runs.
    let mut dataset = fx.dataset();
    dataset.items.truncate(1);
    fx.store.save_dataset(&dataset).unwrap();
    let report = fx
        .store
        .write_subject_masks(
            PROJECT,
            DATASET,
            vec![
                SubjectMaskWrite {
                    content_hash: live.clone(),
                    png: png(&subject_mask(4, 4)),
                },
                SubjectMaskWrite {
                    content_hash: removed.clone(),
                    png: png(&subject_mask(4, 4)),
                },
            ],
            SubjectMaskSource::Auto,
        )
        .expect("a vanished image does not fail the batch");
    assert_eq!(report.skipped_content_hashes, vec![removed]);
    assert_eq!((report.masked, report.total), (1, 1));
    assert_eq!(mask_files(&fx), vec![format!("{live}.png")]);
}

#[test]
fn an_orphan_mask_file_without_an_index_entry_is_pruned() {
    let fx = Fixture::new(2, 4, 4);
    // An interrupted write left a mask file for a live image with no index record, plus one for an
    // image that is not in the dataset at all.
    let masks = fx.root().join(SUBJECT_MASK_DIR);
    fs::create_dir_all(&masks).unwrap();
    fs::write(
        masks.join(format!("{}.png", fx.hash(1))),
        png(&subject_mask(4, 4)),
    )
    .unwrap();
    fs::write(
        masks.join(format!("{}.png", "c".repeat(64))),
        png(&subject_mask(4, 4)),
    )
    .unwrap();
    let report = fx
        .store
        .write_subject_masks(
            PROJECT,
            DATASET,
            vec![SubjectMaskWrite {
                content_hash: fx.hash(0),
                png: png(&subject_mask(4, 4)),
            }],
            SubjectMaskSource::Auto,
        )
        .unwrap();
    assert_eq!(mask_files(&fx), vec![format!("{}.png", fx.hash(0))]);
    assert!(resolve_subject_mask_path(&fx.root(), &fx.dataset().items[1]).is_none());
    assert_eq!(report.masked, 1);
}

#[test]
fn a_later_write_keeps_an_earlier_writes_masks() {
    // The worker POSTs a large dataset in chunks; chunk 2 must not prune chunk 1.
    let fx = Fixture::new(2, 4, 4);
    for index in 0..2 {
        fx.store
            .write_subject_masks(
                PROJECT,
                DATASET,
                vec![SubjectMaskWrite {
                    content_hash: fx.hash(index),
                    png: png(&subject_mask(4, 4)),
                }],
                SubjectMaskSource::Auto,
            )
            .unwrap();
    }
    let report = fx.store.subject_mask_report(PROJECT, DATASET).unwrap();
    assert_eq!((report.masked, report.total), (2, 2));
    assert_eq!(mask_files(&fx).len(), 2);
}

#[test]
fn the_mask_revision_follows_the_stored_bytes() {
    let fx = Fixture::new(1, 4, 4);
    let item = fx.dataset().items[0].id.clone();
    let upload = |mask: &GrayImage| {
        let path = fx.project_path.join("upload.png");
        fs::write(&path, png(mask)).unwrap();
        fx.store
            .upload_subject_mask(PROJECT, DATASET, &item, &path)
            .unwrap()
            .items[0]
            .revision
            .clone()
            .expect("revision")
    };
    // Two uploads in the same second with different bytes get different revisions (the editor's
    // `?v=`), and the revision is the stored PNG's SHA-256 prefix.
    let first = upload(&subject_mask(4, 4));
    let second = upload(&GrayImage::new(4, 4));
    assert_ne!(first, second);
    let stored = fs::read(fx.root().join(subject_mask_relative_path(&fx.hash(0)))).unwrap();
    assert_eq!(second, mask_revision(&stored));
    assert_eq!(second.len(), 16);
}

#[test]
fn masks_survive_an_item_rename_and_prune_when_the_image_leaves() {
    let fx = Fixture::new(2, 4, 4);
    let writes = (0..2)
        .map(|index| SubjectMaskWrite {
            content_hash: fx.hash(index),
            png: png(&subject_mask(4, 4)),
        })
        .collect();
    fx.store
        .write_subject_masks(PROJECT, DATASET, writes, SubjectMaskSource::Auto)
        .unwrap();
    let removed_hash = fx.hash(1);
    // Rename item_0 and drop item_1 (what an editor save does), then write any mask.
    let mut dataset = fx.dataset();
    dataset.items[0].id = "renamed".to_owned();
    dataset.items.truncate(1);
    fx.store.save_dataset(&dataset).unwrap();
    let report = fx
        .store
        .write_subject_masks(
            PROJECT,
            DATASET,
            vec![SubjectMaskWrite {
                content_hash: fx.hash(0),
                png: png(&subject_mask(4, 4)),
            }],
            SubjectMaskSource::Auto,
        )
        .unwrap();
    assert_eq!((report.masked, report.total), (1, 1));
    assert_eq!(report.items[0].item_id, "renamed");
    assert!(
        !fx.root()
            .join(subject_mask_relative_path(&removed_hash))
            .exists(),
        "the removed image's mask is pruned"
    );
}

#[test]
fn mask_targets_backfill_a_legacy_hash_and_refuse_an_unknown_item() {
    let fx = Fixture::new(2, 4, 4);
    let expected = fx.hash(1);
    let mut dataset = fx.dataset();
    dataset.items[1].content_hash = None;
    fx.store.save_dataset(&dataset).unwrap();

    let targets = fx
        .store
        .subject_mask_targets(PROJECT, DATASET, Some(&["item_1".to_owned()]))
        .expect("targets");
    assert_eq!(targets.len(), 1);
    assert_eq!(targets[0].content_hash, expected);
    assert_eq!(targets[0].image_path, fx.root().join("images/item_1.png"));
    assert_eq!(
        fx.dataset().items[1].content_hash.as_deref(),
        Some(expected.as_str()),
        "persisted"
    );
    assert_eq!(
        fx.store
            .subject_mask_targets(PROJECT, DATASET, None)
            .unwrap()
            .len(),
        2
    );

    let error = fx
        .store
        .subject_mask_targets(PROJECT, DATASET, Some(&["nope".to_owned()]))
        .expect_err("unknown item refused");
    assert_eq!(field_code(error), ("itemIds", "subject_mask_unknown_item"));
}
