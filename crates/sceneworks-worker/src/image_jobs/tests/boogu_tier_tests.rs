use super::boogu_tier_subdir;
#[test]
fn every_variant_has_a_default_download_tier() {
    for variant in ["base", "turbo", "edit"] {
        assert_eq!(boogu_tier_subdir(variant, None), variant);
        assert_eq!(boogu_tier_subdir(variant, Some(8)), variant);
    }
}

#[test]
fn tier_subdir_selects_by_quant_bits() {
    assert_eq!(boogu_tier_subdir("base", Some(4)), "base-q4");
    assert_eq!(boogu_tier_subdir("turbo", Some(2)), "turbo-q4");
    assert_eq!(boogu_tier_subdir("edit", Some(0)), "edit-bf16");
    assert_eq!(boogu_tier_subdir("base", Some(-1)), "base-bf16");
}
