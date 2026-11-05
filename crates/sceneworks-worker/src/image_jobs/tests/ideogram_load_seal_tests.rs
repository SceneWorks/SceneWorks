use super::super::LoadSpec;
use super::super::WeightsSource;
use super::seal_ideogram_load_spec;
use crate::Path;
fn write(path: &Path, bytes: &[u8]) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn seals_only_loader_visible_inventory_and_detects_same_length_mutation() {
    for (provider, turbo) in [("ideogram_4", false), ("ideogram_4_turbo", true)] {
        let temp = tempfile::tempdir().expect("temporary Ideogram artifact");
        let root = temp.path().join("q4");
        for component in ["transformer", "text_encoder", "vae"] {
            write(&root.join(component).join("model.safetensors"), b"tensor-a");
        }
        write(&root.join("transformer/config.json"), b"{}");
        write(&root.join("tokenizer/tokenizer.json"), b"{}");
        if turbo {
            write(&root.join("turbo_lora.safetensors"), b"turbo---");
        } else {
            write(
                &root.join("unconditional_transformer/model.safetensors"),
                b"tensor-b",
            );
            write(&root.join("unconditional_transformer/config.json"), b"{}");
        }
        let nested = root.join("transformer/nested/ignored.safetensors");
        write(&nested, b"ignored-a");
        let direct = root.join("transformer/model.safetensors");
        let mut spec = LoadSpec::new(WeightsSource::Dir(root)).with_resolved_route(provider);

        seal_ideogram_load_spec(provider, true, &mut spec).expect("seal Ideogram load");
        assert!(spec.prepared_file_pins().is_prepared());
        spec.validate_prepared_file_pins()
            .expect("stable prepared spec");

        write(&nested, b"ignored-b");
        spec.validate_prepared_file_pins()
            .expect("provider-ignored nested member stays outside the receipt");
        write(&direct, b"tensor-z");
        assert!(spec.validate_prepared_file_pins().is_err());
    }
}
