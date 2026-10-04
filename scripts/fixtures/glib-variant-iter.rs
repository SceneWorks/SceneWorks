// sc-17186: exercise the actual workspace-resolved glib with optimizations enabled.
use glib::variant::ToVariant;

#[test]
fn forward_and_reverse_strings() {
    let strings = ["", "alpha", "Grüße", "東京", "🦀"];
    let variant = strings.to_variant();
    assert_eq!(
        variant.array_iter_str().unwrap().collect::<Vec<_>>(),
        strings
    );
    assert_eq!(
        variant.array_iter_str().unwrap().rev().collect::<Vec<_>>(),
        strings.into_iter().rev().collect::<Vec<_>>()
    );
}

#[test]
fn mixed_ends_and_skips() {
    let variant = ["zero", "one", "two", "three", "four", "five"].to_variant();
    let mut iter = variant.array_iter_str().unwrap();
    assert_eq!(iter.next(), Some("zero"));
    assert_eq!(iter.next_back(), Some("five"));
    assert_eq!(iter.nth(1), Some("two"));
    assert_eq!(iter.nth_back(0), Some("four"));
    assert_eq!(iter.len(), 1);
    assert_eq!(iter.last(), Some("three"));
}

#[test]
fn empty_and_exhausted_iterators() {
    for strings in [vec![], vec!["only"]] {
        let variant = strings.to_variant();
        let mut iter = variant.array_iter_str().unwrap();
        assert_eq!(iter.next(), strings.first().copied());
        assert_eq!(iter.next_back(), None);
        assert_eq!(iter.next(), None);
        assert_eq!(iter.nth(usize::MAX), None);
        assert_eq!(iter.nth_back(usize::MAX), None);
        assert_eq!(iter.len(), 0);
    }
}
