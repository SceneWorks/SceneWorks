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
}

/// Exact load-time component shape behind a public matrix overlay.
///
/// The matrix deliberately has four user-facing overlay coordinates, but those coordinates are not
/// interchangeable load specs. In particular, one ControlNet is not MultiControlNet, and an
/// IP-Adapter is not PiD or a provider-owned identity stack. Production matching and generated route
/// facts therefore key on this exact profile while [`MemoryRouteOverlay`] remains the manifest join.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryRouteLoadProfile {
    Plain,
    Lora,
    SingleControl,
    MultiControl,
    IpAdapter,
    Pid,
    Identity,
}

impl MemoryRouteLoadProfile {
    pub const ALL: [Self; 7] = [
        Self::Plain,
        Self::Lora,
        Self::SingleControl,
        Self::MultiControl,
        Self::IpAdapter,
        Self::Pid,
        Self::Identity,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::Lora => "lora",
            Self::SingleControl => "single_control",
            Self::MultiControl => "multi_control",
            Self::IpAdapter => "ip_adapter",
            Self::Pid => "pid",
            Self::Identity => "identity",
        }
    }

    pub const fn overlay(self) -> MemoryRouteOverlay {
        match self {
            Self::Plain | Self::Pid => MemoryRouteOverlay::None,
            Self::Lora => MemoryRouteOverlay::Lora,
            Self::SingleControl | Self::MultiControl => MemoryRouteOverlay::Control,
            Self::IpAdapter | Self::Identity => MemoryRouteOverlay::Identity,
        }
    }

    fn from_spec(spec: &LoadSpec) -> Option<Self> {
        let profiles = [
            (!spec.adapters.is_empty()).then_some(Self::Lora),
            (spec.control.is_some() && spec.extra_controls.is_empty())
                .then_some(Self::SingleControl),
            (!spec.extra_controls.is_empty()).then_some(Self::MultiControl),
            spec.ip_adapter.is_some().then_some(Self::IpAdapter),
            spec.pid.is_some().then_some(Self::Pid),
            spec.identity.is_some().then_some(Self::Identity),
        ];
        let mut present = profiles.into_iter().flatten();
        let first = present.next();
        if present.next().is_some() {
            return None;
        }
        Some(first.unwrap_or(Self::Plain))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct MemoryRouteSelector {
    pub backend: MemoryRouteBackend,
    pub provider: &'static str,
    pub tier: MemoryRouteTier,
    pub mode: MemoryRouteMode,
    pub overlay: MemoryRouteOverlay,
    pub load_profile: MemoryRouteLoadProfile,
}

#[derive(Clone, Copy)]
struct MemoryRouteRule {
    backend: MemoryRouteBackend,
    provider: &'static str,
    tiers: &'static [MemoryRouteTier],
    modes: &'static [MemoryRouteMode],
    load_profiles: &'static [MemoryRouteLoadProfile],
    requires_sequential_selection: bool,
}

const ALL_TIERS: &[MemoryRouteTier] = &MemoryRouteTier::ALL;
const ALL_MODES: &[MemoryRouteMode] = &MemoryRouteMode::ALL;
const PLAIN: &[MemoryRouteLoadProfile] = &[MemoryRouteLoadProfile::Plain];
const PLAIN_LORA: &[MemoryRouteLoadProfile] =
    &[MemoryRouteLoadProfile::Plain, MemoryRouteLoadProfile::Lora];
const PLAIN_SINGLE_CONTROL: &[MemoryRouteLoadProfile] = &[
    MemoryRouteLoadProfile::Plain,
    MemoryRouteLoadProfile::SingleControl,
];
const PLAIN_SINGLE_CONTROL_IP: &[MemoryRouteLoadProfile] = &[
    MemoryRouteLoadProfile::Plain,
    MemoryRouteLoadProfile::SingleControl,
    MemoryRouteLoadProfile::IpAdapter,
];
const ALL_LOAD_PROFILES: &[MemoryRouteLoadProfile] = &MemoryRouteLoadProfile::ALL;
const TEXT_ONLY: &[MemoryRouteMode] = &[MemoryRouteMode::TextToImage];
const Q4_ONLY: &[MemoryRouteTier] = &[MemoryRouteTier::Q4];
const BF16_ONLY: &[MemoryRouteTier] = &[MemoryRouteTier::Bf16];

const RULES: &[MemoryRouteRule] = &[
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "lens",
        tiers: Q4_ONLY,
        modes: ALL_MODES,
        load_profiles: PLAIN,
        requires_sequential_selection: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "lens_turbo",
        tiers: BF16_ONLY,
        modes: ALL_MODES,
        load_profiles: PLAIN,
        requires_sequential_selection: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "qwen_image",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        load_profiles: PLAIN_LORA,
        requires_sequential_selection: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "qwen_image_edit",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        load_profiles: PLAIN_LORA,
        requires_sequential_selection: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "krea_2_turbo",
        tiers: ALL_TIERS,
        modes: TEXT_ONLY,
        load_profiles: PLAIN,
        requires_sequential_selection: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "sdxl",
        tiers: ALL_TIERS,
        modes: TEXT_ONLY,
        load_profiles: PLAIN,
        requires_sequential_selection: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "z_image_turbo",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        load_profiles: ALL_LOAD_PROFILES,
        requires_sequential_selection: true,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "qwen_image",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        load_profiles: PLAIN,
        requires_sequential_selection: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "qwen_image_edit",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        load_profiles: PLAIN,
        requires_sequential_selection: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "flux1_schnell",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        load_profiles: PLAIN_SINGLE_CONTROL_IP,
        requires_sequential_selection: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "flux1_dev",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        load_profiles: PLAIN_SINGLE_CONTROL_IP,
        requires_sequential_selection: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "flux2_dev",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        load_profiles: PLAIN_SINGLE_CONTROL,
        requires_sequential_selection: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "flux2_klein_9b",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        load_profiles: PLAIN_SINGLE_CONTROL,
        requires_sequential_selection: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "mage_flow_base",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        load_profiles: PLAIN,
        requires_sequential_selection: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "mage_flow",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        load_profiles: PLAIN,
        requires_sequential_selection: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "mage_flow_turbo",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        load_profiles: PLAIN,
        requires_sequential_selection: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "mage_flow_edit_base",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        load_profiles: PLAIN,
        requires_sequential_selection: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "mage_flow_edit",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        load_profiles: PLAIN,
        requires_sequential_selection: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "mage_flow_edit_turbo",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        load_profiles: PLAIN,
        requires_sequential_selection: false,
    },
];

fn rule_matches(selector: MemoryRouteSelector, sequential_selected: bool) -> bool {
    RULES.iter().any(|rule| {
        rule.backend == selector.backend
            && rule.provider == selector.provider
            && rule.tiers.contains(&selector.tier)
            && rule.modes.contains(&selector.mode)
            && rule.load_profiles.contains(&selector.load_profile)
            && selector.overlay == selector.load_profile.overlay()
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
    let (Some(tier), Some(load_profile)) = (
        MemoryRouteTier::from_spec(&spec),
        MemoryRouteLoadProfile::from_spec(&spec),
    ) else {
        return spec;
    };
    let selector = MemoryRouteSelector {
        backend,
        provider: rule.provider,
        tier,
        mode,
        overlay: load_profile.overlay(),
        load_profile,
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
                for &load_profile in rule.load_profiles {
                    out.push(MemoryRouteSelector {
                        backend: rule.backend,
                        provider: rule.provider,
                        tier,
                        mode,
                        overlay: load_profile.overlay(),
                        load_profile,
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

    fn spec(tier: MemoryRouteTier, profile: MemoryRouteLoadProfile) -> LoadSpec {
        let base = LoadSpec::new(WeightsSource::Dir("fixture".into()));
        let base = match tier {
            MemoryRouteTier::Bf16 => base,
            MemoryRouteTier::Q4 => base.with_quant(Quant::Q4),
            MemoryRouteTier::Q8 => base.with_quant(Quant::Q8),
            MemoryRouteTier::Nvfp4 => base.with_quant(Quant::Nvfp4),
        };
        match profile {
            MemoryRouteLoadProfile::Plain => base,
            MemoryRouteLoadProfile::Lora => base.with_adapters(vec![gen_core::AdapterSpec::new(
                "adapter.safetensors".into(),
                1.0,
                gen_core::AdapterKind::Lora,
            )]),
            MemoryRouteLoadProfile::SingleControl => {
                base.with_control(WeightsSource::File("control.safetensors".into()))
            }
            MemoryRouteLoadProfile::MultiControl => base
                .with_control(WeightsSource::File("control.safetensors".into()))
                .with_extra_control(WeightsSource::File("control-2.safetensors".into())),
            MemoryRouteLoadProfile::IpAdapter => {
                base.with_ip_adapter(WeightsSource::Dir("ip-adapter".into()))
            }
            MemoryRouteLoadProfile::Pid => base.with_pid(
                WeightsSource::File("pid.safetensors".into()),
                WeightsSource::Dir("gemma".into()),
            ),
            MemoryRouteLoadProfile::Identity => {
                let mut base = base;
                base.identity = Some(gen_core::IdentityWeights {
                    encoder: Some(WeightsSource::File("identity.safetensors".into())),
                    eva: Some(WeightsSource::File("vision.safetensors".into())),
                    face_dir: Some(WeightsSource::Dir("face-analysis".into())),
                });
                base
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
                MemoryRouteLoadProfile::Plain
            ),
            LoadShape::DeferredMaterialization
        );
        assert_eq!(
            apply(
                MemoryRouteTier::Q8,
                MemoryRouteMode::EditImage,
                MemoryRouteLoadProfile::Lora
            ),
            LoadShape::DeferredMaterialization
        );
        assert_eq!(
            apply(
                MemoryRouteTier::Q4,
                MemoryRouteMode::TextToImage,
                MemoryRouteLoadProfile::SingleControl
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
                spec(MemoryRouteTier::Q4, MemoryRouteLoadProfile::Plain),
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
                spec(MemoryRouteTier::Q8, MemoryRouteLoadProfile::Plain),
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
                spec(MemoryRouteTier::Q4, MemoryRouteLoadProfile::Plain),
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
                    for load_profile in MemoryRouteLoadProfile::ALL {
                        let selector = MemoryRouteSelector {
                            backend: rule.backend,
                            provider: rule.provider,
                            tier,
                            mode,
                            overlay: load_profile.overlay(),
                            load_profile,
                        };
                        let shape = apply_registered_load_shape(
                            selector.backend,
                            selector.provider,
                            selector.mode,
                            spec(selector.tier, selector.load_profile),
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

    #[test]
    fn flux_load_profiles_preserve_the_pre_registry_fail_closed_predicates() {
        for provider in ["flux1_schnell", "flux1_dev"] {
            for profile in [
                MemoryRouteLoadProfile::Pid,
                MemoryRouteLoadProfile::Identity,
                MemoryRouteLoadProfile::MultiControl,
            ] {
                assert_eq!(
                    apply_registered_load_shape(
                        MemoryRouteBackend::Candle,
                        provider,
                        MemoryRouteMode::TextToImage,
                        spec(MemoryRouteTier::Bf16, profile),
                        false,
                    )
                    .load_shape,
                    LoadShape::EagerMaterialization,
                    "{provider} must reject {profile:?}",
                );
            }
            for profile in [
                MemoryRouteLoadProfile::SingleControl,
                MemoryRouteLoadProfile::IpAdapter,
            ] {
                assert_eq!(
                    apply_registered_load_shape(
                        MemoryRouteBackend::Candle,
                        provider,
                        MemoryRouteMode::TextToImage,
                        spec(MemoryRouteTier::Bf16, profile),
                        false,
                    )
                    .load_shape,
                    LoadShape::DeferredMaterialization,
                    "{provider} must retain {profile:?}",
                );
            }
        }

        for provider in ["flux2_dev", "flux2_klein_9b"] {
            assert_eq!(
                apply_registered_load_shape(
                    MemoryRouteBackend::Candle,
                    provider,
                    MemoryRouteMode::TextToImage,
                    spec(MemoryRouteTier::Bf16, MemoryRouteLoadProfile::MultiControl),
                    false,
                )
                .load_shape,
                LoadShape::EagerMaterialization,
                "{provider} must reject MultiControlNet",
            );
            assert_eq!(
                apply_registered_load_shape(
                    MemoryRouteBackend::Candle,
                    provider,
                    MemoryRouteMode::TextToImage,
                    spec(MemoryRouteTier::Bf16, MemoryRouteLoadProfile::SingleControl,),
                    false,
                )
                .load_shape,
                LoadShape::DeferredMaterialization,
                "{provider} must retain one ControlNet",
            );
        }
    }
}
