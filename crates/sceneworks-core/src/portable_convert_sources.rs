//! Source files consumed by native model conversion before its output is registered.
//! Admission freezes this closure; the worker reads it from the verified snapshot.

pub const ANIMA_TE_FILE: &str = "split_files/text_encoders/qwen_3_06b_base.safetensors";
pub const ANIMA_VAE_FILE: &str = "split_files/vae/qwen_image_vae.safetensors";

fn source_file(value: Option<&str>) -> Result<&str, String> {
    let file = value.ok_or("conversion lacks sourceFile")?;
    if file.is_empty()
        || file.starts_with('/')
        || file.contains(['\\', '\0'])
        || file
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
    {
        return Err("conversion sourceFile is not a safe repository-relative file".to_owned());
    }
    Ok(file)
}

/// Exact primary selection for an uncataloged conversion source. Cataloged
/// sources retain their pinned manifest allow-list; this avoids fetching a
/// whole community repository when the converter reads only one checkpoint.
pub fn portable_convert_primary_file_selectors(
    converter: &str,
    selected_source_file: Option<&str>,
) -> Result<Vec<String>, String> {
    match converter {
        "flux2_klein_diffusers" | "ltx_video" | "supra2_img" => {
            Ok(vec![source_file(selected_source_file)?.to_owned()])
        }
        "anima_quant" => Ok(vec![
            source_file(selected_source_file)?.to_owned(),
            ANIMA_TE_FILE.to_owned(),
            ANIMA_VAE_FILE.to_owned(),
        ]),
        "flux2_dev_quant"
        | "sd3_5_large_quant"
        | "sd3_5_large_turbo_quant"
        | "sd3_5_medium_quant" => Ok(vec!["*".to_owned()]),
        _ => Err(format!(
            "no native source selection for converter {converter}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_selection_matches_actual_converter_inputs() {
        assert_eq!(
            portable_convert_primary_file_selectors("ltx_video", Some("weights/eros.safetensors"))
                .unwrap(),
            ["weights/eros.safetensors"]
        );
        assert_eq!(
            portable_convert_primary_file_selectors(
                "anima_quant",
                Some("split_files/diffusion_models/anima-base-v1.0.safetensors")
            )
            .unwrap(),
            [
                "split_files/diffusion_models/anima-base-v1.0.safetensors",
                ANIMA_TE_FILE,
                ANIMA_VAE_FILE,
            ]
        );
        assert_eq!(
            portable_convert_primary_file_selectors("flux2_dev_quant", None).unwrap(),
            ["*"]
        );
        assert!(portable_convert_primary_file_selectors("ltx_video", Some("../other")).is_err());
    }
}
