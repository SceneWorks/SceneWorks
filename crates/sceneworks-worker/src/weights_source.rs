//! Path access for the backend-neutral provider weight source.
use std::path::Path;

pub(crate) trait WeightsSourceExt {
    fn path(&self) -> &Path;
}

impl WeightsSourceExt for gen_core::WeightsSource {
    fn path(&self) -> &Path {
        match self {
            Self::Dir(path) | Self::File(path) => path,
        }
    }
}

/// Filesystem evidence only; missing paths carry no evidence. Directories count tensor shards.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) fn path_weight_bytes(path: &Path) -> u64 {
    match std::fs::metadata(path) {
        Ok(metadata) if metadata.is_file() => metadata.len(),
        Ok(metadata) if metadata.is_dir() => crate::mlx_fit_gate::sum_safetensors_bytes(path),
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn both_weight_source_variants_expose_the_original_path() {
        for source in [
            gen_core::WeightsSource::Dir("dir".into()),
            gen_core::WeightsSource::File("file".into()),
        ] {
            let expected = match &source {
                gen_core::WeightsSource::Dir(p) | gen_core::WeightsSource::File(p) => p,
            };
            assert_eq!(source.path(), expected);
        }
    }
}
