//! Canonical architecture identities and spelling aliases shared by model/adapter detectors.
//! Tensor signatures and compatibility remain in their owning detectors: sharing an identity does
//! not assert identical generation, conditioning, size, or loader support.

pub const FLUX: &str = "flux";
pub const FLUX2: &str = "flux2";
pub const FLUX3: &str = "flux3";
pub const CHROMA: &str = "chroma";
pub const KREA_2: &str = "krea_2";
pub const KREA_REALTIME: &str = "krea-realtime";
pub const MAGE_FLOW: &str = "mage-flow";
pub const MOSS_IMAGE: &str = "moss-image";
pub const MINIMAX_H3: &str = "minimax-h3";
/// MiniMax-H3-Image (SW-7100): its own family. Its DiT reuses H3's layer names but drops the audio
/// stream and adds six per-step `pdd_proj_out` heads, and it decodes through a 2-D VAE, so neither
/// H3's fine-tune import nor H3's LoRAs apply to it. Every needle that would match `minimax-h3`
/// is checked against this identity FIRST.
pub const MINIMAX_H3_IMAGE: &str = "minimax-h3-image";
pub const QWEN_IMAGE: &str = "qwen-image";
pub const QWEN_IMAGE_2_1: &str = "qwen-image-2-1";
pub const WAN_VIDEO: &str = "wan-video";
pub const LTX_VIDEO: &str = "ltx-video";
pub const MOCHI: &str = "mochi";
pub const SDXL: &str = "sdxl";
pub const SD3: &str = "sd3";
pub const SD1_5: &str = "sd1.5";
pub const Z_IMAGE: &str = "z-image";
pub const ANIMA: &str = "anima";
pub const IDEOGRAM: &str = "ideogram";
pub const IRIS: &str = "iris";
pub const LENS: &str = "lens";
pub const KOLORS: &str = "kolors";
pub const SVD: &str = "svd";
pub const BERNINI: &str = "bernini";
pub const SCAIL2: &str = "scail2";
pub const SENSENOVA_U1: &str = "sensenova-u1";

/// Exact spelling aliases shared by all family vocabularies. Model variants such as FLUX.2
/// klein/dev and Qwen 2.1 remain distinct identities; this is never an architecture-compatibility
/// table. Broader file/repository-name heuristics remain target-specific.
pub fn exact_identity(value: &str) -> Option<&'static str> {
    match value.trim().to_ascii_lowercase().replace('_', "-").as_str() {
        "flux" => Some(FLUX),
        "flux2" | "flux-2" | "flux.2" => Some(FLUX2),
        "flux3" => Some(FLUX3),
        "chroma" => Some(CHROMA),
        "krea2" | "krea-2" => Some(KREA_2),
        "krea-realtime" => Some(KREA_REALTIME),
        "mageflow" | "mage-flow" => Some(MAGE_FLOW),
        "mossimage" | "moss-image" => Some(MOSS_IMAGE),
        "minimaxh3" | "minimax-h3" => Some(MINIMAX_H3),
        "minimaxh3image" | "minimax-h3-image" => Some(MINIMAX_H3_IMAGE),
        "qwenimage" | "qwen-image" => Some(QWEN_IMAGE),
        "qwen-image-2-1" | "qwen-image-2.1" => Some(QWEN_IMAGE_2_1),
        "wan" | "wan-video" => Some(WAN_VIDEO),
        "ltx" | "ltx-video" => Some(LTX_VIDEO),
        "mochi" => Some(MOCHI),
        "sdxl" | "stable diffusion xl" | "stablediffusionxl" => Some(SDXL),
        "sd3" => Some(SD3),
        "sd1" | "sd1.5" => Some(SD1_5),
        "zimage" | "z-image" => Some(Z_IMAGE),
        "anima" => Some(ANIMA),
        "ideogram" => Some(IDEOGRAM),
        "iris" => Some(IRIS),
        "lens" => Some(LENS),
        "kolors" => Some(KOLORS),
        "svd" => Some(SVD),
        "bernini" => Some(BERNINI),
        "scail2" => Some(SCAIL2),
        "sensenova-u1" => Some(SENSENOVA_U1),
        _ => None,
    }
}

/// Canonical stored family identity, preserving unknown/custom families.
pub fn canonical(value: &str) -> String {
    exact_identity(value)
        .map(str::to_owned)
        .unwrap_or_else(|| normalized(value))
}

/// Normalized comparison spelling. The stored Krea identity remains `krea_2`.
pub(crate) fn normalized(value: &str) -> String {
    let normalized = value.trim().to_ascii_lowercase().replace('_', "-");
    match normalized.as_str() {
        "krea2" => "krea-2".to_owned(),
        _ => normalized,
    }
}

/// Conservative diffusers class-name aliases. Directory-specific config checks remain with callers.
pub fn diffusers_class_name_to_family(class_name: &str) -> Option<String> {
    let normalized = class_name.trim();
    let lower = normalized.to_ascii_lowercase();
    match lower.as_str() {
        "zimagepipeline"
        | "zimageimg2imgpipeline"
        | "zimageturbopipeline"
        | "zimageturboimg2imgpipeline" => Some(Z_IMAGE.to_owned()),
        "qwenimagepipeline" | "qwenimageimg2imgpipeline" | "qwenimageeditpipeline" => {
            Some(QWEN_IMAGE.to_owned())
        }
        "lenspipeline" => Some(LENS.to_owned()),
        // Directory detection additionally validates transformer/config.json before accepting this
        // family; this mapping is also used by callers that already hold a trusted class name.
        "mageflowpipeline" => Some(MAGE_FLOW.to_owned()),
        // Anima (epic 10512). Its diffusers modular pipeline (`AnimaModularPipeline`) is the
        // `model_index.json` `_class_name` a diffusers-format Anima export would carry.
        "animamodularpipeline" => Some(ANIMA.to_owned()),
        "fluxpipeline" | "fluximg2imgpipeline" | "fluxinpaintpipeline" => Some(FLUX.to_owned()),
        "chromapipeline" | "chromaimg2imgpipeline" => Some(CHROMA.to_owned()),
        "kolorspipeline" | "kolorsimg2imgpipeline" => Some(KOLORS.to_owned()),
        "wanpipeline" | "wani2vpipeline" | "wantext2videopipeline" => Some(WAN_VIDEO.to_owned()),
        "ltxpipeline" | "ltxvideopipeline" | "ltximagetovideopipeline" => {
            Some(LTX_VIDEO.to_owned())
        }
        "stablediffusion3pipeline"
        | "stablediffusion3img2imgpipeline"
        | "stablediffusion3inpaintpipeline" => Some(SD3.to_owned()),
        "stablediffusionpipeline" | "stablediffusionimg2imgpipeline" => Some(SD1_5.to_owned()),
        "stablediffusionxlpipeline" | "stablediffusionxlimg2imgpipeline" => Some(SDXL.to_owned()),
        _ => None,
    }
}

/// Checkpoint descriptors first preserve explicit canonical identities and generation aliases.
/// Broad class/path heuristics are fallback evidence only; loader admission remains separate.
pub(crate) fn checkpoint_descriptor_family(value: &str) -> Option<String> {
    if let Some(family) = exact_identity(value) {
        return Some(family.to_owned());
    }
    let lower = value.to_ascii_lowercase();
    for (needle, family) in [
        // MiniMax-H3-Image (SW-7100): `MiniMaxH3ImageTransformer2DModel` /
        // `AutoencoderKLMiniMaxH3Image` contain `minimaxh3`, so the image family's needles come
        // before H3's or every H3-Image descriptor would read as an H3 video checkpoint.
        ("minimaxh3image", MINIMAX_H3_IMAGE),
        ("minimax_h3_image", MINIMAX_H3_IMAGE),
        ("minimax-h3-image", MINIMAX_H3_IMAGE),
        // MiniMax-H3's diffusers `_class_name` (`MiniMaxH3Transformer3DModel`, sc-26680). First so
        // no broader needle below can claim it.
        ("minimaxh3", MINIMAX_H3),
        ("minimax_h3", MINIMAX_H3),
        ("minimax-h3", MINIMAX_H3),
        ("stable diffusion xl", SDXL),
        ("stablediffusionxl", SDXL),
        ("sdxl", SDXL),
        ("qwenimage", QWEN_IMAGE),
        ("qwen_image", QWEN_IMAGE),
        ("qwen-image", QWEN_IMAGE),
        ("flux2", FLUX2),
        ("flux.2", FLUX2),
        ("flux", FLUX),
        ("wan", WAN_VIDEO),
        ("z_image", Z_IMAGE),
        ("z-image", Z_IMAGE),
        ("mageflow", MAGE_FLOW),
        ("mage_flow", MAGE_FLOW),
        ("mage-flow", MAGE_FLOW),
        ("krea", KREA_2),
        ("ltx", LTX_VIDEO),
        ("ideogram", IDEOGRAM),
        ("anima", ANIMA),
        ("cosmos", ANIMA),
    ] {
        if lower.contains(needle) {
            return Some(family.to_owned());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_descriptor_identities_precede_ambiguous_substring_evidence() {
        for (label, expected) in [
            ("qwen-image-2-1", QWEN_IMAGE_2_1),
            ("QWEN_IMAGE_2_1", QWEN_IMAGE_2_1),
            ("krea-realtime", KREA_REALTIME),
            ("krea_2", KREA_2),
        ] {
            assert_eq!(
                checkpoint_descriptor_family(label).as_deref(),
                Some(expected)
            );
        }
        assert_eq!(
            checkpoint_descriptor_family("QwenImageTransformer2DModel").as_deref(),
            Some(QWEN_IMAGE)
        );
        assert_eq!(
            checkpoint_descriptor_family("MiniMaxH3Transformer3DModel").as_deref(),
            Some(MINIMAX_H3)
        );
        // SW-7100: the image family's class names contain `minimaxh3` and must not read as H3.
        for label in [
            "MiniMaxH3ImageTransformer2DModel",
            "AutoencoderKLMiniMaxH3Image",
            "minimax_h3_image",
            "minimax-h3-image",
        ] {
            assert_eq!(
                checkpoint_descriptor_family(label).as_deref(),
                Some(MINIMAX_H3_IMAGE),
                "{label}"
            );
        }
        assert_eq!(exact_identity("minimax_h3_image"), Some(MINIMAX_H3_IMAGE));
        assert_eq!(canonical("MiniMax_H3_Image"), MINIMAX_H3_IMAGE);
        assert_eq!(
            checkpoint_descriptor_family("unrecognized architecture"),
            None
        );
    }
}
