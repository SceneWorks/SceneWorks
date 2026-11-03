use super::seal_recursive_receipt_load_spec;
use super::super::LoadSpec;
use super::super::WeightsSource;
#[test]
fn exact_inventory_is_carried_to_cache_load_and_mutation_fails_closed() {
    for provider in [
        "chroma1_base",
        "sd3_5_large",
        "sd3_5_large_turbo",
        "sd3_5_medium",
    ] {
        let temp = tempfile::tempdir().expect("temporary receipt artifact");
        let root = temp.path().join("q4");
        std::fs::create_dir_all(root.join("transformer")).expect("artifact directory");
        let config = root.join("transformer/config.json");
        std::fs::write(&config, b"{}").expect("artifact member");
        let mut spec = LoadSpec::new(WeightsSource::Dir(root)).with_resolved_route(provider);

        seal_recursive_receipt_load_spec(provider, true, &mut spec)
            .expect("seal recursive receipt load");
        assert!(spec.prepared_file_pins().is_prepared());
        assert_eq!(spec.prepared_file_pins().len(), 1);
        spec.validate_prepared_file_pins()
            .expect("stable prepared spec");

        std::fs::write(config, b"{\"changed\":true}").expect("mutate artifact member");
        assert!(spec.validate_prepared_file_pins().is_err(), "{provider}");
    }
}
