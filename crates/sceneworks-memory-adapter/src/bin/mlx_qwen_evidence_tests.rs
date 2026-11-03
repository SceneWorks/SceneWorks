use super::*;

#[test]
fn negative_mutation_changes_pixels_and_is_measured_from_the_changed_image() {
    let baseline = Image {
        width: 2,
        height: 1,
        pixels: vec![0, 63, 127, 128, 191, 255],
    };
    let mutated = qwen_negative_mutation(&baseline);
    assert_ne!(mutated, baseline);
    let (maximum, mean) = image_max_mean_abs(&mutated, &baseline).unwrap();
    assert!(maximum >= 64.0 / 255.0);
    assert!(mean >= 64.0 / 255.0);
    assert!(!qwen_quality_passes(maximum, mean));
}

#[test]
fn physical_mlx_rgb_receipts_are_role_and_content_addressed_without_overwrite() {
    let fixture_owner = crate::adapter_test_support::Fixtures::default();
    let fixtures = &fixture_owner;
    let root =
        fixtures.path("physical-mlx-rgb-receipts-are-role-and-content-addressed-without-overwrite");
    std::fs::create_dir_all(&root).expect("create receipt fixture");
    let first = Image {
        width: 2,
        height: 1,
        pixels: vec![1, 2, 3, 4, 5, 6],
    };
    let changed = Image {
        width: 2,
        height: 1,
        pixels: vec![6, 5, 4, 3, 2, 1],
    };

    let selected = persist_physical_mlx_image(
        &root,
        "docs/calibration/sc-test",
        "imc-case",
        "selected_rgb",
        &first,
    )
    .expect("persist first selected receipt");
    let repeated = persist_physical_mlx_image(
        &root,
        "docs/calibration/sc-test",
        "imc-case",
        "selected_rgb",
        &first,
    )
    .expect("identical content-addressed receipt is reusable");
    assert_eq!(selected["path"], repeated["path"]);

    let changed_selected = persist_physical_mlx_image(
        &root,
        "docs/calibration/sc-test",
        "imc-case",
        "selected_rgb",
        &changed,
    )
    .expect("changed bytes receive a different content address");
    assert_ne!(selected["path"], changed_selected["path"]);

    let reference = persist_physical_mlx_image(
        &root,
        "docs/calibration/sc-test",
        "imc-case",
        "reference_rgb",
        &first,
    )
    .expect("the role is part of the receipt address");
    assert_ne!(selected["path"], reference["path"]);

    let selected_local = selected["localPath"]
        .as_str()
        .expect("receipt carries local path");
    std::fs::write(selected_local, b"tampered").expect("tamper fixture receipt");
    let error = persist_physical_mlx_image(
        &root,
        "docs/calibration/sc-test",
        "imc-case",
        "selected_rgb",
        &first,
    )
    .expect_err("exclusive receipt creation must not repair or overwrite tampering");
    assert!(error.contains("already exists with different bytes"));
}
