//! Typed production registry for load shapes that can reach deferred materialization.
//!
//! This is deliberately executable data shared by the worker route selectors and the engine-facts
//! dumper. Consumers must not recover it by parsing Rust source text: formatting and equivalent
//! control-flow refactors are not route facts.

use gen_core::{LoadShape, LoadSpec, Precision, Quant, WeightsSource};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryRouteBackend {
    Candle,
    Mlx,
}

impl MemoryRouteBackend {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Candle => "candle",
            Self::Mlx => "mlx",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryRouteTier {
    Bf16,
    Q4,
    Q8,
    Nvfp4,
}

impl MemoryRouteTier {
    pub const ALL: [Self; 4] = [Self::Bf16, Self::Q4, Self::Q8, Self::Nvfp4];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Q4 => "q4",
            Self::Q8 => "q8",
            Self::Nvfp4 => "nvfp4",
        }
    }

    fn from_spec(spec: &LoadSpec) -> Option<Self> {
        if spec.precision != Precision::Bf16 {
            return None;
        }
        match spec.quantize {
            None => Some(Self::Bf16),
            Some(Quant::Q4) => Some(Self::Q4),
            Some(Quant::Q8) => Some(Self::Q8),
            Some(Quant::Nvfp4) => Some(Self::Nvfp4),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryRouteMode {
    TextToImage,
    StyleVariations,
    EditImage,
    ImageToImage,
    ImageInpaint,
    ImageDetail,
    CharacterImage,
}

impl MemoryRouteMode {
    pub const ALL: [Self; 7] = [
        Self::TextToImage,
        Self::StyleVariations,
        Self::EditImage,
        Self::ImageToImage,
        Self::ImageInpaint,
        Self::ImageDetail,
        Self::CharacterImage,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TextToImage => "text_to_image",
            Self::StyleVariations => "style_variations",
            Self::EditImage => "edit_image",
            Self::ImageToImage => "image_to_image",
            Self::ImageInpaint => "image_inpaint",
            Self::ImageDetail => "image_detail",
            Self::CharacterImage => "character_image",
        }
    }

    pub fn from_request(mode: &str) -> Option<Self> {
        match mode {
            "image_generation" | "text_to_image" => Some(Self::TextToImage),
            "style_variations" => Some(Self::StyleVariations),
            "edit_image" => Some(Self::EditImage),
            "image_to_image" => Some(Self::ImageToImage),
            "image_inpaint" => Some(Self::ImageInpaint),
            "image_detail" => Some(Self::ImageDetail),
            "character_image" => Some(Self::CharacterImage),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryRouteOverlay {
    None,
    Lora,
    Control,
    Identity,
}

impl MemoryRouteOverlay {
    pub const ALL: [Self; 4] = [Self::None, Self::Lora, Self::Control, Self::Identity];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Lora => "lora",
            Self::Control => "control",
            Self::Identity => "identity",
        }
    }

    fn from_spec(spec: &LoadSpec) -> Option<Self> {
        let lora = !spec.adapters.is_empty();
        let control = spec.control.is_some() || !spec.extra_controls.is_empty();
        let identity = spec.ip_adapter.is_some() || spec.pid.is_some() || spec.identity.is_some();
        match (lora, control, identity) {
            (false, false, false) => Some(Self::None),
            (true, false, false) => Some(Self::Lora),
            (false, true, false) => Some(Self::Control),
            (false, false, true) => Some(Self::Identity),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct MemoryRouteSelector {
    pub backend: MemoryRouteBackend,
    pub provider: &'static str,
    pub tier: MemoryRouteTier,
    pub mode: MemoryRouteMode,
    pub overlay: MemoryRouteOverlay,
}

#[derive(Clone, Copy)]
struct MemoryRouteRule {
    backend: MemoryRouteBackend,
    provider: &'static str,
    tiers: &'static [MemoryRouteTier],
    modes: &'static [MemoryRouteMode],
    overlays: &'static [MemoryRouteOverlay],
    requires_sequential_selection: bool,
}

const ALL_TIERS: &[MemoryRouteTier] = &MemoryRouteTier::ALL;
const ALL_MODES: &[MemoryRouteMode] = &MemoryRouteMode::ALL;
const NONE: &[MemoryRouteOverlay] = &[MemoryRouteOverlay::None];
const NONE_LORA: &[MemoryRouteOverlay] = &[MemoryRouteOverlay::None, MemoryRouteOverlay::Lora];
const NONE_CONTROL: &[MemoryRouteOverlay] =
    &[MemoryRouteOverlay::None, MemoryRouteOverlay::Control];
const NONE_CONTROL_IDENTITY: &[MemoryRouteOverlay] = &[
    MemoryRouteOverlay::None,
    MemoryRouteOverlay::Control,
    MemoryRouteOverlay::Identity,
];
const TEXT_ONLY: &[MemoryRouteMode] = &[MemoryRouteMode::TextToImage];
const Q4_ONLY: &[MemoryRouteTier] = &[MemoryRouteTier::Q4];
const BF16_ONLY: &[MemoryRouteTier] = &[MemoryRouteTier::Bf16];

const RULES: &[MemoryRouteRule] = &[
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "lens",
        tiers: Q4_ONLY,
        modes: ALL_MODES,
        overlays: NONE,
        requires_sequential_selection: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "lens_turbo",
        tiers: BF16_ONLY,
        modes: ALL_MODES,
        overlays: NONE,
        requires_sequential_selection: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "qwen_image",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        overlays: NONE_LORA,
        requires_sequential_selection: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "qwen_image_edit",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        overlays: NONE_LORA,
        requires_sequential_selection: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "krea_2_turbo",
        tiers: ALL_TIERS,
        modes: TEXT_ONLY,
        overlays: NONE,
        requires_sequential_selection: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "sdxl",
        tiers: ALL_TIERS,
        modes: TEXT_ONLY,
        overlays: NONE,
        requires_sequential_selection: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "z_image_turbo",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        overlays: &MemoryRouteOverlay::ALL,
        requires_sequential_selection: true,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "qwen_image",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        overlays: NONE,
        requires_sequential_selection: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "qwen_image_edit",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        overlays: NONE,
        requires_sequential_selection: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "flux1_schnell",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        overlays: NONE_CONTROL_IDENTITY,
        requires_sequential_selection: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "flux1_dev",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        overlays: NONE_CONTROL_IDENTITY,
        requires_sequential_selection: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "flux2_dev",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        overlays: NONE_CONTROL,
        requires_sequential_selection: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "flux2_klein_9b",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        overlays: NONE_CONTROL,
        requires_sequential_selection: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "mage_flow_base",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        overlays: NONE,
        requires_sequential_selection: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "mage_flow",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        overlays: NONE,
        requires_sequential_selection: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "mage_flow_turbo",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        overlays: NONE,
        requires_sequential_selection: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "mage_flow_edit_base",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        overlays: NONE,
        requires_sequential_selection: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "mage_flow_edit",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        overlays: NONE,
        requires_sequential_selection: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "mage_flow_edit_turbo",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        overlays: NONE,
        requires_sequential_selection: false,
    },
];

fn rule_matches(selector: MemoryRouteSelector, sequential_selected: bool) -> bool {
    RULES.iter().any(|rule| {
        rule.backend == selector.backend
            && rule.provider == selector.provider
            && rule.tiers.contains(&selector.tier)
            && rule.modes.contains(&selector.mode)
            && rule.overlays.contains(&selector.overlay)
            && (!rule.requires_sequential_selection || sequential_selected)
    })
}

pub fn apply_registered_load_shape(
    backend: MemoryRouteBackend,
    provider: &str,
    mode: MemoryRouteMode,
    spec: LoadSpec,
    sequential_selected: bool,
) -> LoadSpec {
    let Some(rule) = RULES
        .iter()
        .find(|rule| rule.backend == backend && rule.provider == provider)
    else {
        return spec;
    };
    // Z-Image's production selector has always coupled its deferred shape directly to the
    // Sequential decision, independently of artifact layout. Its rule spans every normalized
    // tier/mode/overlay coordinate, while this early return also preserves the legacy behavior for
    // a load whose source cannot be normalized into those finite fact axes.
    if rule.requires_sequential_selection {
        return if sequential_selected {
            spec.with_load_shape(LoadShape::DeferredMaterialization)
        } else {
            spec
        };
    }
    if !matches!(spec.weights, WeightsSource::Dir(_)) {
        return spec;
    }
    let (Some(tier), Some(overlay)) = (
        MemoryRouteTier::from_spec(&spec),
        MemoryRouteOverlay::from_spec(&spec),
    ) else {
        return spec;
    };
    let selector = MemoryRouteSelector {
        backend,
        provider: rule.provider,
        tier,
        mode,
        overlay,
    };
    if rule_matches(selector, sequential_selected) {
        spec.with_load_shape(LoadShape::DeferredMaterialization)
    } else {
        spec
    }
}

/// Exact finite witness emitted into the pin-bound engine-capability facts.
pub fn deferred_route_witnesses() -> Vec<MemoryRouteSelector> {
    let mut out = Vec::new();
    for rule in RULES {
        for &tier in rule.tiers {
            for &mode in rule.modes {
                for &overlay in rule.overlays {
                    out.push(MemoryRouteSelector {
                        backend: rule.backend,
                        provider: rule.provider,
                        tier,
                        mode,
                        overlay,
                    });
                }
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(tier: MemoryRouteTier, overlay: MemoryRouteOverlay) -> LoadSpec {
        let base = LoadSpec::new(WeightsSource::Dir("fixture".into()));
        let base = match tier {
            MemoryRouteTier::Bf16 => base,
            MemoryRouteTier::Q4 => base.with_quant(Quant::Q4),
            MemoryRouteTier::Q8 => base.with_quant(Quant::Q8),
            MemoryRouteTier::Nvfp4 => base.with_quant(Quant::Nvfp4),
        };
        match overlay {
            MemoryRouteOverlay::None => base,
            MemoryRouteOverlay::Lora => base.with_adapters(vec![gen_core::AdapterSpec::new(
                "adapter.safetensors".into(),
                1.0,
                gen_core::AdapterKind::Lora,
            )]),
            MemoryRouteOverlay::Control => {
                base.with_control(WeightsSource::File("control.safetensors".into()))
            }
            MemoryRouteOverlay::Identity => {
                base.with_ip_adapter(WeightsSource::Dir("ip-adapter".into()))
            }
        }
    }

    #[test]
    fn qwen_tier_mode_and_overlay_axes_are_semantic() {
        let apply = |tier, mode, overlay| {
            apply_registered_load_shape(
                MemoryRouteBackend::Mlx,
                "qwen_image",
                mode,
                spec(tier, overlay),
                false,
            )
            .load_shape
        };
        assert_eq!(
            apply(
                MemoryRouteTier::Q4,
                MemoryRouteMode::TextToImage,
                MemoryRouteOverlay::None
            ),
            LoadShape::DeferredMaterialization
        );
        assert_eq!(
            apply(
                MemoryRouteTier::Q8,
                MemoryRouteMode::EditImage,
                MemoryRouteOverlay::Lora
            ),
            LoadShape::DeferredMaterialization
        );
        assert_eq!(
            apply(
                MemoryRouteTier::Q4,
                MemoryRouteMode::TextToImage,
                MemoryRouteOverlay::Control
            ),
            LoadShape::EagerMaterialization
        );
    }

    #[test]
    fn lens_q4_only_and_text_only_rules_fail_closed() {
        assert_eq!(
            apply_registered_load_shape(
                MemoryRouteBackend::Mlx,
                "lens",
                MemoryRouteMode::EditImage,
                spec(MemoryRouteTier::Q4, MemoryRouteOverlay::None),
                false
            )
            .load_shape,
            LoadShape::DeferredMaterialization
        );
        assert_eq!(
            apply_registered_load_shape(
                MemoryRouteBackend::Mlx,
                "lens",
                MemoryRouteMode::EditImage,
                spec(MemoryRouteTier::Q8, MemoryRouteOverlay::None),
                false
            )
            .load_shape,
            LoadShape::EagerMaterialization
        );
        assert_eq!(
            apply_registered_load_shape(
                MemoryRouteBackend::Mlx,
                "sdxl",
                MemoryRouteMode::EditImage,
                spec(MemoryRouteTier::Q4, MemoryRouteOverlay::None),
                false
            )
            .load_shape,
            LoadShape::EagerMaterialization
        );
    }

    #[test]
    fn witness_has_no_wildcard_coordinates() {
        let witnesses = deferred_route_witnesses();
        assert!(!witnesses.is_empty());
        assert!(witnesses.iter().any(|row| row.provider == "qwen_image"
            && row.tier == MemoryRouteTier::Q4
            && row.mode == MemoryRouteMode::TextToImage
            && row.overlay == MemoryRouteOverlay::None));
        assert!(witnesses.iter().any(|row| row.provider == "qwen_image"
            && row.tier == MemoryRouteTier::Q8
            && row.mode == MemoryRouteMode::EditImage
            && row.overlay == MemoryRouteOverlay::Lora));
    }

    #[test]
    fn finite_witness_is_the_executable_registry_cross_product() {
        let witnesses = deferred_route_witnesses();
        for rule in RULES {
            for tier in MemoryRouteTier::ALL {
                for mode in MemoryRouteMode::ALL {
                    for overlay in MemoryRouteOverlay::ALL {
                        let selector = MemoryRouteSelector {
                            backend: rule.backend,
                            provider: rule.provider,
                            tier,
                            mode,
                            overlay,
                        };
                        let shape = apply_registered_load_shape(
                            selector.backend,
                            selector.provider,
                            selector.mode,
                            spec(selector.tier, selector.overlay),
                            rule.requires_sequential_selection,
                        )
                        .load_shape;
                        assert_eq!(
                            shape == LoadShape::DeferredMaterialization,
                            witnesses.contains(&selector),
                            "typed witness disagrees with production evaluation for {selector:?}",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn z_image_deferred_shape_remains_bound_to_the_sequential_decision() {
        let file_spec = LoadSpec::new(WeightsSource::File("fixture.safetensors".into()));
        assert_eq!(
            apply_registered_load_shape(
                MemoryRouteBackend::Mlx,
                "z_image_turbo",
                MemoryRouteMode::TextToImage,
                file_spec.clone(),
                true,
            )
            .load_shape,
            LoadShape::DeferredMaterialization,
        );
        assert_eq!(
            apply_registered_load_shape(
                MemoryRouteBackend::Mlx,
                "z_image_turbo",
                MemoryRouteMode::TextToImage,
                file_spec,
                false,
            )
            .load_shape,
            LoadShape::EagerMaterialization,
        );
    }
}
