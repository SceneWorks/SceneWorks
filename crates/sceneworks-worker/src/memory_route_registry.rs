//! Typed production registry for load shapes that can reach deferred materialization.
//!
//! This is deliberately executable data shared by the worker route selectors and the engine-facts
//! dumper. Consumers must not recover it by parsing Rust source text: formatting and equivalent
//! control-flow refactors are not route facts.

use gen_core::{
    LoadShape, LoadShapeDeclarationResult, LoadSpec, MemoryStrategy, MemoryStrategySupport,
    Precision, Quant, WeightsSource,
};
use serde_json::{Map as JsonObject, Value};

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

    pub fn from_resolved_tier(tier: &str) -> Option<Self> {
        match tier {
            "bf16" => Some(Self::Bf16),
            "q4" => Some(Self::Q4),
            "q8" => Some(Self::Q8),
            "nvfp4" => Some(Self::Nvfp4),
            _ => None,
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
    /// Whether this coordinate existed in the pre-declaration legacy shaper. New declaration-owned
    /// routes must not become reachable merely because their manifest declaration is removed.
    legacy_shaping: bool,
}

const ALL_TIERS: &[MemoryRouteTier] = &MemoryRouteTier::ALL;
const ALL_MODES: &[MemoryRouteMode] = &MemoryRouteMode::ALL;
const PLAIN: &[MemoryRouteLoadProfile] = &[MemoryRouteLoadProfile::Plain];
const SINGLE_CONTROL: &[MemoryRouteLoadProfile] = &[MemoryRouteLoadProfile::SingleControl];
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
const TEXT_AND_STYLE: &[MemoryRouteMode] = &[
    MemoryRouteMode::TextToImage,
    MemoryRouteMode::StyleVariations,
];
const EDIT_MODES: &[MemoryRouteMode] = &[MemoryRouteMode::EditImage, MemoryRouteMode::ImageToImage];
const KOLORS_MODES: &[MemoryRouteMode] = &[
    MemoryRouteMode::TextToImage,
    MemoryRouteMode::StyleVariations,
    MemoryRouteMode::EditImage,
    MemoryRouteMode::CharacterImage,
];
const PLAIN_IP: &[MemoryRouteLoadProfile] = &[
    MemoryRouteLoadProfile::Plain,
    MemoryRouteLoadProfile::IpAdapter,
];
const PLAIN_LORA_IP: &[MemoryRouteLoadProfile] = &[
    MemoryRouteLoadProfile::Plain,
    MemoryRouteLoadProfile::Lora,
    MemoryRouteLoadProfile::IpAdapter,
];
const Q4_Q8: &[MemoryRouteTier] = &[MemoryRouteTier::Q4, MemoryRouteTier::Q8];
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
        legacy_shaping: true,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "lens_turbo",
        tiers: BF16_ONLY,
        modes: ALL_MODES,
        load_profiles: PLAIN,
        requires_sequential_selection: false,
        legacy_shaping: true,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "qwen_image",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        load_profiles: PLAIN_LORA,
        requires_sequential_selection: false,
        legacy_shaping: true,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "qwen_image_edit",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        load_profiles: PLAIN_LORA,
        requires_sequential_selection: false,
        legacy_shaping: true,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "krea_2_turbo",
        tiers: ALL_TIERS,
        modes: TEXT_ONLY,
        load_profiles: PLAIN,
        requires_sequential_selection: false,
        legacy_shaping: true,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "sdxl",
        tiers: ALL_TIERS,
        modes: TEXT_ONLY,
        load_profiles: PLAIN,
        requires_sequential_selection: false,
        legacy_shaping: true,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "z_image_turbo",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        load_profiles: ALL_LOAD_PROFILES,
        requires_sequential_selection: true,
        legacy_shaping: true,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "z_image_turbo",
        tiers: ALL_TIERS,
        modes: EDIT_MODES,
        load_profiles: PLAIN_LORA,
        requires_sequential_selection: false,
        legacy_shaping: true,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "z_image",
        tiers: ALL_TIERS,
        modes: TEXT_AND_STYLE,
        load_profiles: PLAIN_LORA,
        requires_sequential_selection: false,
        legacy_shaping: true,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "anima_base",
        tiers: ALL_TIERS,
        modes: TEXT_ONLY,
        load_profiles: PLAIN_LORA,
        requires_sequential_selection: false,
        legacy_shaping: true,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "anima_aesthetic",
        tiers: ALL_TIERS,
        modes: TEXT_ONLY,
        load_profiles: PLAIN_LORA,
        requires_sequential_selection: false,
        legacy_shaping: true,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "anima_turbo",
        tiers: ALL_TIERS,
        modes: TEXT_ONLY,
        load_profiles: PLAIN_LORA,
        requires_sequential_selection: false,
        legacy_shaping: true,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "chroma1_hd",
        tiers: ALL_TIERS,
        modes: TEXT_AND_STYLE,
        load_profiles: PLAIN,
        requires_sequential_selection: false,
        legacy_shaping: true,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "chroma1_base",
        tiers: ALL_TIERS,
        modes: TEXT_AND_STYLE,
        load_profiles: PLAIN,
        requires_sequential_selection: false,
        legacy_shaping: true,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "chroma1_flash",
        tiers: ALL_TIERS,
        modes: TEXT_AND_STYLE,
        load_profiles: PLAIN,
        requires_sequential_selection: false,
        legacy_shaping: true,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "kolors",
        tiers: BF16_ONLY,
        modes: KOLORS_MODES,
        load_profiles: PLAIN_IP,
        requires_sequential_selection: false,
        legacy_shaping: true,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "kolors",
        tiers: Q4_Q8,
        modes: KOLORS_MODES,
        load_profiles: PLAIN_LORA_IP,
        requires_sequential_selection: false,
        legacy_shaping: true,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "z_image",
        tiers: ALL_TIERS,
        modes: TEXT_AND_STYLE,
        load_profiles: PLAIN_LORA,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "z_image_turbo",
        tiers: ALL_TIERS,
        modes: TEXT_AND_STYLE,
        load_profiles: PLAIN_LORA,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "z_image_turbo",
        tiers: ALL_TIERS,
        modes: EDIT_MODES,
        load_profiles: PLAIN_LORA,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "z_image_control",
        tiers: ALL_TIERS,
        modes: TEXT_AND_STYLE,
        load_profiles: SINGLE_CONTROL,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "z_image_turbo_control",
        tiers: ALL_TIERS,
        modes: TEXT_AND_STYLE,
        load_profiles: SINGLE_CONTROL,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "lens",
        tiers: Q4_Q8,
        modes: TEXT_ONLY,
        load_profiles: PLAIN,
        requires_sequential_selection: true,
        legacy_shaping: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "lens_turbo",
        tiers: Q4_Q8,
        modes: TEXT_ONLY,
        load_profiles: PLAIN,
        requires_sequential_selection: true,
        legacy_shaping: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "qwen_image",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        load_profiles: PLAIN,
        requires_sequential_selection: false,
        legacy_shaping: true,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "qwen_image_edit",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        load_profiles: PLAIN,
        requires_sequential_selection: false,
        legacy_shaping: true,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "flux1_schnell",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        load_profiles: PLAIN_SINGLE_CONTROL_IP,
        requires_sequential_selection: false,
        legacy_shaping: true,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "flux1_dev",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        load_profiles: PLAIN_SINGLE_CONTROL_IP,
        requires_sequential_selection: false,
        legacy_shaping: true,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "flux2_dev",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        load_profiles: PLAIN_SINGLE_CONTROL,
        requires_sequential_selection: false,
        legacy_shaping: true,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "flux2_klein_9b",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        load_profiles: PLAIN_SINGLE_CONTROL,
        requires_sequential_selection: false,
        legacy_shaping: true,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "mage_flow_base",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        load_profiles: PLAIN,
        requires_sequential_selection: false,
        legacy_shaping: true,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "mage_flow",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        load_profiles: PLAIN,
        requires_sequential_selection: false,
        legacy_shaping: true,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "mage_flow_turbo",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        load_profiles: PLAIN,
        requires_sequential_selection: false,
        legacy_shaping: true,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "mage_flow_edit_base",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        load_profiles: PLAIN,
        requires_sequential_selection: false,
        legacy_shaping: true,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "mage_flow_edit",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        load_profiles: PLAIN,
        requires_sequential_selection: false,
        legacy_shaping: true,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "mage_flow_edit_turbo",
        tiers: ALL_TIERS,
        modes: ALL_MODES,
        load_profiles: PLAIN,
        requires_sequential_selection: false,
        legacy_shaping: true,
    },
];

fn matching_rules(selector: MemoryRouteSelector) -> impl Iterator<Item = &'static MemoryRouteRule> {
    RULES.iter().filter(move |rule| {
        rule.backend == selector.backend
            && rule.provider == selector.provider
            && rule.tiers.contains(&selector.tier)
            && rule.modes.contains(&selector.mode)
            && rule.load_profiles.contains(&selector.load_profile)
            && selector.overlay == selector.load_profile.overlay()
    })
}

#[cfg(test)]
fn rule_coordinates_match(selector: MemoryRouteSelector) -> bool {
    matching_rules(selector).next().is_some()
}

fn rule_matches(selector: MemoryRouteSelector, sequential_selected: bool) -> bool {
    matching_rules(selector).any(|rule| !rule.requires_sequential_selection || sequential_selected)
}

fn manifest_declares_selector(
    manifest: &JsonObject<String, Value>,
    selector: MemoryRouteSelector,
) -> bool {
    let Some(backend) = manifest
        .get(selector.backend.as_str())
        .and_then(Value::as_object)
    else {
        return false;
    };
    let Some(contract) = backend
        .get("memoryStrategyContract")
        .and_then(Value::as_object)
    else {
        return false;
    };
    let Some(contract_provider) = contract.get("provider").and_then(Value::as_str) else {
        return false;
    };
    contract
        .get("implementations")
        .and_then(Value::as_array)
        .is_some_and(|implementations| {
            implementations.iter().any(|implementation| {
                implementation
                    .get("runtimeProvider")
                    .and_then(Value::as_str)
                    .unwrap_or(contract_provider)
                    == selector.provider
                    && implementation.get("rung").and_then(Value::as_str)
                        == Some("bounded_transformer_residency")
                    && implementation
                        .get("tiers")
                        .and_then(Value::as_array)
                        .is_some_and(|tiers| {
                            tiers
                                .iter()
                                .any(|tier| tier.as_str() == Some(selector.tier.as_str()))
                        })
                    && implementation
                        .get("modes")
                        .and_then(Value::as_array)
                        .is_some_and(|modes| {
                            modes
                                .iter()
                                .any(|mode| mode.as_str() == Some(selector.mode.as_str()))
                        })
                    && implementation
                        .get("overlays")
                        .and_then(Value::as_array)
                        .is_some_and(|overlays| {
                            overlays
                                .iter()
                                .any(|overlay| overlay.as_str() == Some(selector.overlay.as_str()))
                        })
            })
        })
}

fn has_relevant_btr_declaration(
    manifest: &JsonObject<String, Value>,
    backend: MemoryRouteBackend,
) -> Result<bool, ()> {
    let Some(backend) = manifest.get(backend.as_str()).and_then(Value::as_object) else {
        return Ok(false);
    };
    let Some(contract_value) = backend.get("memoryStrategyContract") else {
        return Ok(false);
    };
    let Some(contract) = contract_value.as_object() else {
        return Err(());
    };
    let Some(implementations_value) = contract.get("implementations") else {
        return Err(());
    };
    let Some(implementations) = implementations_value.as_array() else {
        return Err(());
    };
    Ok(implementations.iter().any(|implementation| {
        implementation.get("rung").and_then(Value::as_str) == Some("bounded_transformer_residency")
    }))
}

/// Evaluate manifest-owned deferred shaping against the exact typed coordinate and linked provider.
/// The tri-state result distinguishes a route with no BTR declaration (the only case where the
/// caller may use legacy shaping) from a declared refusal. The resolved artifact tier is explicit:
/// `LoadSpec::quantize` describes a possible load-time transform and is not artifact-tier proof.
pub fn evaluate_declared_mlx_load_shape(
    provider: &str,
    resolved_tier: Option<&str>,
    mode: Option<MemoryRouteMode>,
    manifest: &JsonObject<String, Value>,
    spec: LoadSpec,
) -> LoadSpec {
    evaluate_declared_mlx_load_shape_with(
        provider,
        resolved_tier,
        mode,
        manifest,
        spec,
        |candidate| {
            crate::inference_runtime::media()
                .memory_strategy_contract(provider, candidate)
                .ok()
                .flatten()
                .is_some_and(|contract| {
                    contract
                        .capability(MemoryStrategy::BoundedTransformerResidency)
                        .is_some_and(|capability| {
                            capability.support == MemoryStrategySupport::Implemented
                        })
                })
        },
    )
}

fn evaluate_declared_mlx_load_shape_with(
    provider: &str,
    resolved_tier: Option<&str>,
    mode: Option<MemoryRouteMode>,
    manifest: &JsonObject<String, Value>,
    spec: LoadSpec,
    provider_implements: impl FnOnce(&LoadSpec) -> bool,
) -> LoadSpec {
    match has_relevant_btr_declaration(manifest, MemoryRouteBackend::Mlx) {
        Ok(false) => return spec,
        Err(()) => return spec.with_refused_load_shape_declaration(),
        Ok(true) => {}
    }
    let (Some(tier), Some(mode), Some(load_profile)) = (
        resolved_tier.and_then(MemoryRouteTier::from_resolved_tier),
        mode,
        MemoryRouteLoadProfile::from_spec(&spec),
    ) else {
        return spec.with_refused_load_shape_declaration();
    };
    let Some(registered_provider) = RULES
        .iter()
        .find(|rule| rule.backend == MemoryRouteBackend::Mlx && rule.provider == provider)
        .map(|rule| rule.provider)
    else {
        return spec.with_refused_load_shape_declaration();
    };
    let selector = MemoryRouteSelector {
        backend: MemoryRouteBackend::Mlx,
        provider: registered_provider,
        tier,
        mode,
        overlay: load_profile.overlay(),
        load_profile,
    };
    if !manifest_declares_selector(manifest, selector) {
        return spec.with_refused_load_shape_declaration();
    }
    let mut matching = matching_rules(selector).peekable();
    if matching.peek().is_none() {
        return spec.with_refused_load_shape_declaration();
    }
    if matching.all(|rule| rule.requires_sequential_selection) {
        return spec;
    }
    if !matches!(spec.weights, WeightsSource::Dir(_)) {
        return spec.with_refused_load_shape_declaration();
    }

    let candidate = spec.clone().with_applied_load_shape_declaration();
    if provider_implements(&candidate) {
        candidate
    } else {
        spec.with_refused_load_shape_declaration()
    }
}

/// Candle counterpart to [`evaluate_declared_mlx_load_shape`]. The provider predicate receives a
/// fully assembled candidate. Rules that require Sequential use it only for that predicate; the
/// returned request keeps its selected offload policy and carries declaration authority separately.
pub fn evaluate_declared_candle_load_shape(
    runtime_provider: &str,
    resolved_tier: Option<&str>,
    mode: Option<MemoryRouteMode>,
    manifest: &JsonObject<String, Value>,
    spec: LoadSpec,
    sequential_selected: bool,
) -> LoadSpec {
    evaluate_declared_candle_load_shape_with(
        runtime_provider,
        resolved_tier,
        mode,
        manifest,
        spec,
        sequential_selected,
        |candidate| {
            crate::inference_runtime::media()
                .memory_strategy_contract(runtime_provider, candidate)
                .ok()
                .flatten()
                .is_some_and(|contract| {
                    contract
                        .capability(MemoryStrategy::BoundedTransformerResidency)
                        .is_some_and(|capability| {
                            capability.support == MemoryStrategySupport::Implemented
                        })
                })
        },
    )
}

/// Rebuild the exact Sequential+Deferred provider contract used only by the shared selector while
/// the retained load spec remains `Eligible + Eager`. Eligibility is deliberately unbound in
/// gen-core, so this helper revalidates the original manifest route, runtime provider, explicit
/// resolved tier, mode, overlay/profile, source shape, and provider predicate on every call.
pub fn declared_candle_selector_contract(
    runtime_provider: &str,
    resolved_tier: Option<&str>,
    mode: Option<MemoryRouteMode>,
    manifest: &JsonObject<String, Value>,
    spec: &LoadSpec,
) -> Option<gen_core::MemoryProviderContract> {
    if spec.load_shape_declaration_result != LoadShapeDeclarationResult::Eligible {
        return None;
    }
    let mut contract = None;
    let revalidated = evaluate_declared_candle_load_shape_with(
        runtime_provider,
        resolved_tier,
        mode,
        manifest,
        spec.clone(),
        false,
        |candidate| {
            contract = crate::inference_runtime::media()
                .memory_strategy_contract(runtime_provider, candidate)
                .ok()
                .flatten();
            contract.as_ref().is_some_and(|contract| {
                contract
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .is_some_and(|capability| {
                        capability.support == MemoryStrategySupport::Implemented
                    })
            })
        },
    );
    (revalidated.load_shape_declaration_result == LoadShapeDeclarationResult::Eligible)
        .then_some(contract)
        .flatten()
}

fn evaluate_declared_candle_load_shape_with(
    runtime_provider: &str,
    resolved_tier: Option<&str>,
    mode: Option<MemoryRouteMode>,
    manifest: &JsonObject<String, Value>,
    spec: LoadSpec,
    sequential_selected: bool,
    provider_implements: impl FnOnce(&LoadSpec) -> bool,
) -> LoadSpec {
    match has_relevant_btr_declaration(manifest, MemoryRouteBackend::Candle) {
        Ok(false) => return spec,
        Err(()) => return spec.with_refused_load_shape_declaration(),
        Ok(true) => {}
    }
    let (Some(tier), Some(mode), Some(load_profile)) = (
        resolved_tier.and_then(MemoryRouteTier::from_resolved_tier),
        mode,
        MemoryRouteLoadProfile::from_spec(&spec),
    ) else {
        return spec.with_refused_load_shape_declaration();
    };
    let Some(manifest_route) = manifest.get("id").and_then(Value::as_str) else {
        return spec.with_refused_load_shape_declaration();
    };
    if spec.resolved_route.as_deref() != Some(manifest_route) {
        return spec.with_refused_load_shape_declaration();
    }
    let selector = MemoryRouteSelector {
        backend: MemoryRouteBackend::Candle,
        provider: RULES
            .iter()
            .find(|rule| {
                rule.backend == MemoryRouteBackend::Candle && rule.provider == runtime_provider
            })
            .map(|rule| rule.provider)
            .unwrap_or(""),
        tier,
        mode,
        overlay: load_profile.overlay(),
        load_profile,
    };
    if selector.provider.is_empty() || !matches!(spec.weights, WeightsSource::Dir(_)) {
        return spec.with_refused_load_shape_declaration();
    }
    let matching = matching_rules(selector)
        .filter(|_| manifest_declares_selector(manifest, selector))
        .collect::<Vec<_>>();
    if matching.is_empty() {
        return spec.with_refused_load_shape_declaration();
    }

    let requires_sequential = matching
        .iter()
        .all(|rule| rule.requires_sequential_selection);
    let mut candidate = spec.clone().with_applied_load_shape_declaration();
    if requires_sequential {
        candidate = candidate.with_offload_policy(gen_core::OffloadPolicy::Sequential);
    }
    if provider_implements(&candidate) {
        if requires_sequential && !sequential_selected {
            spec.with_eligible_load_shape_declaration()
        } else {
            spec.with_applied_load_shape_declaration()
        }
    } else {
        spec.with_refused_load_shape_declaration()
    }
}

/// Typed manifest-derived population used by correctness harnesses. This is the same declaration
/// parser production uses, so route additions/removals cannot be mirrored by source-text search.
pub fn declared_mlx_deferred_routes(models: &[Value]) -> std::collections::BTreeSet<String> {
    let mut routes = std::collections::BTreeSet::new();
    for model in models {
        let Some(model) = model.as_object() else {
            continue;
        };
        let Some(id) = model.get("id").and_then(Value::as_str) else {
            continue;
        };
        let Some(capabilities) = model.get("capabilities").and_then(Value::as_array) else {
            continue;
        };
        let Some(provider) = model
            .get("mlx")
            .and_then(Value::as_object)
            .and_then(|mlx| mlx.get("memoryStrategyContract"))
            .and_then(Value::as_object)
            .and_then(|contract| contract.get("provider"))
            .and_then(Value::as_str)
        else {
            continue;
        };
        let Some(provider) = RULES
            .iter()
            .find(|rule| rule.backend == MemoryRouteBackend::Mlx && rule.provider == provider)
            .map(|rule| rule.provider)
        else {
            continue;
        };
        if MemoryRouteTier::ALL.iter().any(|&tier| {
            MemoryRouteMode::ALL.iter().any(|&mode| {
                if !capabilities.iter().any(|capability| {
                    capability.as_str().and_then(MemoryRouteMode::from_request) == Some(mode)
                }) {
                    return false;
                }
                MemoryRouteLoadProfile::ALL.iter().any(|&load_profile| {
                    let selector = MemoryRouteSelector {
                        backend: MemoryRouteBackend::Mlx,
                        provider,
                        tier,
                        mode,
                        overlay: load_profile.overlay(),
                        load_profile,
                    };
                    manifest_declares_selector(model, selector) && rule_matches(selector, false)
                })
            })
        }) {
            routes.insert(id.to_owned());
        }
    }
    routes
}

/// Exact catalog route/runtime-provider population for Candle declaration-owned shaping. A single
/// catalog row may yield both its ordinary provider and a composed `_control` provider, while aliases
/// such as `z_image_edit` remain distinct by catalog id.
pub fn declared_candle_deferred_routes(
    models: &[Value],
) -> std::collections::BTreeSet<(String, String)> {
    let mut routes = std::collections::BTreeSet::new();
    for model in models {
        let Some(model) = model.as_object() else {
            continue;
        };
        let (Some(id), Some(capabilities), Some(_contract)) = (
            model.get("id").and_then(Value::as_str),
            model.get("capabilities").and_then(Value::as_array),
            model
                .get("candle")
                .and_then(Value::as_object)
                .and_then(|candle| candle.get("memoryStrategyContract"))
                .and_then(Value::as_object),
        ) else {
            continue;
        };
        for rule in RULES
            .iter()
            .filter(|rule| rule.backend == MemoryRouteBackend::Candle)
        {
            let declared = rule.tiers.iter().any(|&tier| {
                rule.modes.iter().any(|&mode| {
                    if !capabilities.iter().any(|capability| {
                        capability.as_str().and_then(MemoryRouteMode::from_request) == Some(mode)
                    }) {
                        return false;
                    }
                    rule.load_profiles.iter().any(|&load_profile| {
                        let selector = MemoryRouteSelector {
                            backend: MemoryRouteBackend::Candle,
                            provider: rule.provider,
                            tier,
                            mode,
                            overlay: load_profile.overlay(),
                            load_profile,
                        };
                        manifest_declares_selector(model, selector)
                    })
                })
            });
            if declared {
                routes.insert((id.to_owned(), rule.provider.to_owned()));
            }
        }
    }
    routes
}

/// Whether the typed Candle rule is declaration-owned rather than an unchanged legacy shaper.
/// This is derived from the same registry row production evaluates; callers must not maintain a
/// second provider list for declaration-only behavior.
pub fn candle_declaration_owns_load_shape(provider: &str) -> bool {
    RULES.iter().any(|rule| {
        rule.backend == MemoryRouteBackend::Candle
            && rule.provider == provider
            && !rule.legacy_shaping
    })
}

pub fn apply_registered_load_shape(
    backend: MemoryRouteBackend,
    provider: &str,
    mode: MemoryRouteMode,
    spec: LoadSpec,
    sequential_selected: bool,
) -> LoadSpec {
    if spec.load_shape_declaration_result != LoadShapeDeclarationResult::NotEvaluated {
        return spec;
    }
    let Some(rule) = RULES
        .iter()
        .find(|rule| rule.backend == backend && rule.provider == provider && rule.legacy_shaping)
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
    if matching_rules(selector).any(|rule| {
        rule.legacy_shaping && (!rule.requires_sequential_selection || sequential_selected)
    }) {
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

    fn declaration(
        provider: &str,
        tiers: &[&str],
        modes: &[&str],
        overlays: &[&str],
    ) -> JsonObject<String, Value> {
        serde_json::json!({
            "id": provider,
            "mlx": {
                "memoryStrategyContract": {
                    "abi": 1,
                    "provider": provider,
                    "implementations": [{
                        "rung": "bounded_transformer_residency",
                        "tiers": tiers,
                        "modes": modes,
                        "overlays": overlays
                    }]
                }
            }
        })
        .as_object()
        .unwrap()
        .clone()
    }

    fn candle_declaration(
        route: &str,
        contract_provider: &str,
        runtime_provider: &str,
        tiers: &[&str],
        modes: &[&str],
        overlays: &[&str],
    ) -> JsonObject<String, Value> {
        serde_json::json!({
            "id": route,
            "candle": {
                "memoryStrategyContract": {
                    "abi": 1,
                    "provider": contract_provider,
                    "implementations": [{
                        "rung": "bounded_transformer_residency",
                        "runtimeProvider": runtime_provider,
                        "tiers": tiers,
                        "modes": modes,
                        "overlays": overlays
                    }]
                }
            }
        })
        .as_object()
        .unwrap()
        .clone()
    }

    fn shipped_model(id: &str) -> JsonObject<String, Value> {
        let raw = include_str!("../../../config/manifests/builtin.models.jsonc");
        let manifest: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(raw))
                .expect("builtin model manifest parses");
        manifest["models"]
            .as_array()
            .expect("models array")
            .iter()
            .find(|model| model["id"].as_str() == Some(id))
            .and_then(Value::as_object)
            .unwrap_or_else(|| panic!("missing shipped model {id}"))
            .clone()
    }

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
    fn declaration_is_the_marker_and_provider_predicate_is_binding() {
        let manifest = declaration(
            "z_image",
            &["bf16", "q4", "q8"],
            &["text_to_image", "style_variations"],
            &["none", "lora"],
        );
        assert!(manifest["mlx"].get("supportsSequentialOffload").is_none());
        let candidate = spec(MemoryRouteTier::Q8, MemoryRouteLoadProfile::Plain);
        let shaped = evaluate_declared_mlx_load_shape_with(
            "z_image",
            Some("q4"),
            Some(MemoryRouteMode::TextToImage),
            &manifest,
            candidate.clone(),
            |candidate| candidate.load_shape == LoadShape::DeferredMaterialization,
        );
        assert_eq!(
            shaped.load_shape_declaration_result,
            LoadShapeDeclarationResult::Applied
        );
        assert_eq!(shaped.load_shape, LoadShape::DeferredMaterialization);
        assert_eq!(shaped.quantize, Some(Quant::Q8));

        let refused = evaluate_declared_mlx_load_shape_with(
            "z_image",
            Some("q4"),
            Some(MemoryRouteMode::TextToImage),
            &manifest,
            candidate,
            |_| false,
        );
        assert_eq!(
            refused.load_shape_declaration_result,
            LoadShapeDeclarationResult::Refused
        );
        assert_eq!(refused.load_shape, LoadShape::EagerMaterialization);
    }

    #[test]
    fn candle_sequential_declaration_requires_fresh_exact_second_pass() {
        let manifest = candle_declaration(
            "lens",
            "lens",
            "lens",
            &["q4"],
            &["text_to_image"],
            &["none"],
        );
        assert!(manifest["candle"]
            .get("supportsSequentialOffload")
            .is_none());
        let input =
            spec(MemoryRouteTier::Q4, MemoryRouteLoadProfile::Plain).with_resolved_route("lens");
        let first = evaluate_declared_candle_load_shape_with(
            "lens",
            Some("q4"),
            Some(MemoryRouteMode::TextToImage),
            &manifest,
            input,
            false,
            |candidate| {
                candidate.load_shape == LoadShape::DeferredMaterialization
                    && candidate.offload_policy == gen_core::OffloadPolicy::Sequential
            },
        );
        assert_eq!(
            first.load_shape_declaration_result,
            LoadShapeDeclarationResult::Eligible
        );
        assert_eq!(first.load_shape, LoadShape::EagerMaterialization);

        let generic = apply_registered_load_shape(
            MemoryRouteBackend::Candle,
            "lens",
            MemoryRouteMode::TextToImage,
            first.clone(),
            true,
        );
        assert_eq!(
            generic.load_shape_declaration_result,
            LoadShapeDeclarationResult::Eligible,
            "generic shaping must not consume unbound declaration eligibility",
        );
        assert_eq!(generic.load_shape, LoadShape::EagerMaterialization);

        let applied = evaluate_declared_candle_load_shape_with(
            "lens",
            Some("q4"),
            Some(MemoryRouteMode::TextToImage),
            &manifest,
            first.clone(),
            true,
            |_| true,
        );
        assert_eq!(
            applied.load_shape_declaration_result,
            LoadShapeDeclarationResult::Applied
        );
        assert_eq!(applied.load_shape, LoadShape::DeferredMaterialization);

        let resident = evaluate_declared_candle_load_shape_with(
            "lens",
            Some("q4"),
            Some(MemoryRouteMode::TextToImage),
            &manifest,
            first.clone(),
            false,
            |_| true,
        );
        assert_eq!(
            resident.load_shape_declaration_result,
            LoadShapeDeclarationResult::Eligible
        );
        assert_eq!(resident.load_shape, LoadShape::EagerMaterialization);

        for crossed in [
            evaluate_declared_candle_load_shape_with(
                "lens_turbo",
                Some("q4"),
                Some(MemoryRouteMode::TextToImage),
                &manifest,
                first.clone(),
                true,
                |_| true,
            ),
            evaluate_declared_candle_load_shape_with(
                "lens",
                Some("q8"),
                Some(MemoryRouteMode::TextToImage),
                &manifest,
                first.clone(),
                true,
                |_| true,
            ),
            evaluate_declared_candle_load_shape_with(
                "lens",
                Some("q4"),
                Some(MemoryRouteMode::StyleVariations),
                &manifest,
                first.clone(),
                true,
                |_| true,
            ),
            evaluate_declared_candle_load_shape_with(
                "lens",
                Some("q4"),
                Some(MemoryRouteMode::TextToImage),
                &manifest,
                first.clone().with_resolved_route("lens_turbo"),
                true,
                |_| true,
            ),
            evaluate_declared_candle_load_shape_with(
                "lens",
                Some("q4"),
                Some(MemoryRouteMode::TextToImage),
                &manifest,
                first.clone().with_adapters(vec![gen_core::AdapterSpec::new(
                    "adapter.safetensors".into(),
                    1.0,
                    gen_core::AdapterKind::Lora,
                )]),
                true,
                |_| true,
            ),
            evaluate_declared_candle_load_shape_with(
                "lens",
                Some("q4"),
                Some(MemoryRouteMode::TextToImage),
                &manifest,
                LoadSpec::new(WeightsSource::File("lens.safetensors".into()))
                    .with_quant(Quant::Q4)
                    .with_resolved_route("lens")
                    .with_eligible_load_shape_declaration(),
                true,
                |_| true,
            ),
        ] {
            assert_ne!(
                crossed.load_shape_declaration_result,
                LoadShapeDeclarationResult::Applied
            );
            assert_eq!(crossed.load_shape, LoadShape::EagerMaterialization);
        }

        let mut crossed_manifest = manifest;
        crossed_manifest["candle"]["memoryStrategyContract"]["implementations"][0]
            ["runtimeProvider"] = Value::String("lens_turbo".to_owned());
        let crossed = evaluate_declared_candle_load_shape_with(
            "lens",
            Some("q4"),
            Some(MemoryRouteMode::TextToImage),
            &crossed_manifest,
            first,
            true,
            |_| true,
        );
        assert_eq!(
            crossed.load_shape_declaration_result,
            LoadShapeDeclarationResult::Refused
        );
        assert_eq!(crossed.load_shape, LoadShape::EagerMaterialization);
    }

    #[test]
    fn candle_resolved_artifact_tier_is_not_recovered_from_quantize() {
        let manifest = candle_declaration(
            "z_image",
            "z_image",
            "z_image",
            &["q4"],
            &["text_to_image"],
            &["none"],
        );
        let prepacked_q4 =
            LoadSpec::new(WeightsSource::Dir("z-image-q4".into())).with_resolved_route("z_image");
        let shaped = evaluate_declared_candle_load_shape_with(
            "z_image",
            Some("q4"),
            Some(MemoryRouteMode::TextToImage),
            &manifest,
            prepacked_q4,
            false,
            |_| true,
        );
        assert_eq!(shaped.quantize, None);
        assert_eq!(
            shaped.load_shape_declaration_result,
            LoadShapeDeclarationResult::Applied
        );
        assert_eq!(shaped.load_shape, LoadShape::DeferredMaterialization);
    }

    #[test]
    fn candle_control_and_external_encoder_shapes_reach_the_provider_exactly() {
        let control_manifest = candle_declaration(
            "z_image",
            "z_image",
            "z_image_control",
            &["q4"],
            &["text_to_image"],
            &["control"],
        );
        let control = LoadSpec::new(WeightsSource::Dir("z-image-q4".into()))
            .with_resolved_route("z_image")
            .with_control(WeightsSource::File("control.safetensors".into()));
        let shaped = evaluate_declared_candle_load_shape_with(
            "z_image_control",
            Some("q4"),
            Some(MemoryRouteMode::TextToImage),
            &control_manifest,
            control,
            false,
            |candidate| {
                candidate.control.is_some()
                    && candidate.load_shape == LoadShape::DeferredMaterialization
            },
        );
        assert_eq!(
            shaped.load_shape_declaration_result,
            LoadShapeDeclarationResult::Applied
        );

        let control_with_pid = LoadSpec::new(WeightsSource::Dir("z-image-q4".into()))
            .with_resolved_route("z_image")
            .with_control(WeightsSource::File("control.safetensors".into()))
            .with_pid(
                WeightsSource::File("pid.safetensors".into()),
                WeightsSource::Dir("gemma".into()),
            );
        let refused = evaluate_declared_candle_load_shape_with(
            "z_image_control",
            Some("q4"),
            Some(MemoryRouteMode::TextToImage),
            &control_manifest,
            control_with_pid,
            false,
            |_| panic!("an undeclared combined profile cannot call the provider"),
        );
        assert_eq!(
            refused.load_shape_declaration_result,
            LoadShapeDeclarationResult::Refused
        );
        assert_eq!(refused.load_shape, LoadShape::EagerMaterialization);

        for refused in [
            evaluate_declared_candle_load_shape_with(
                "z_image_control",
                Some("q4"),
                Some(MemoryRouteMode::TextToImage),
                &control_manifest,
                LoadSpec::new(WeightsSource::Dir("z-image-q4".into()))
                    .with_resolved_route("z_image"),
                false,
                |_| true,
            ),
            evaluate_declared_candle_load_shape_with(
                "z_image_turbo_control",
                Some("q4"),
                Some(MemoryRouteMode::TextToImage),
                &control_manifest,
                LoadSpec::new(WeightsSource::Dir("z-image-q4".into()))
                    .with_resolved_route("z_image")
                    .with_control(WeightsSource::File("control.safetensors".into())),
                false,
                |_| true,
            ),
            evaluate_declared_candle_load_shape_with(
                "z_image_control",
                Some("q4"),
                Some(MemoryRouteMode::EditImage),
                &control_manifest,
                LoadSpec::new(WeightsSource::Dir("z-image-q4".into()))
                    .with_resolved_route("z_image")
                    .with_control(WeightsSource::File("control.safetensors".into())),
                false,
                |_| true,
            ),
        ] {
            assert_eq!(
                refused.load_shape_declaration_result,
                LoadShapeDeclarationResult::Refused
            );
            assert_eq!(refused.load_shape, LoadShape::EagerMaterialization);
        }

        let ordinary_manifest = candle_declaration(
            "z_image_turbo",
            "z_image_turbo",
            "z_image_turbo",
            &["q4"],
            &["text_to_image"],
            &["none"],
        );
        let mut external = LoadSpec::new(WeightsSource::Dir("z-image-turbo-q4".into()))
            .with_resolved_route("z_image_turbo");
        external.text_encoder = Some(WeightsSource::File("external-te.safetensors".into()));
        let shaped = evaluate_declared_candle_load_shape_with(
            "z_image_turbo",
            Some("q4"),
            Some(MemoryRouteMode::TextToImage),
            &ordinary_manifest,
            external,
            false,
            |candidate| candidate.text_encoder.is_none(),
        );
        assert_eq!(
            shaped.load_shape_declaration_result,
            LoadShapeDeclarationResult::Refused
        );
        assert!(shaped.text_encoder.is_some());
    }

    #[test]
    fn new_candle_routes_never_fall_through_when_the_declaration_is_absent() {
        for provider in [
            "z_image",
            "z_image_turbo",
            "z_image_control",
            "z_image_turbo_control",
            "lens",
            "lens_turbo",
        ] {
            let manifest = serde_json::json!({ "id": provider, "candle": {} })
                .as_object()
                .unwrap()
                .clone();
            let input =
                LoadSpec::new(WeightsSource::Dir("fixture".into())).with_resolved_route(provider);
            let evaluated = evaluate_declared_candle_load_shape_with(
                provider,
                Some("q4"),
                Some(MemoryRouteMode::TextToImage),
                &manifest,
                input,
                true,
                |_| panic!("an absent declaration cannot call the provider"),
            );
            assert_eq!(
                evaluated.load_shape_declaration_result,
                LoadShapeDeclarationResult::NotEvaluated
            );
            let shaped = apply_registered_load_shape(
                MemoryRouteBackend::Candle,
                provider,
                MemoryRouteMode::TextToImage,
                evaluated,
                true,
            );
            assert_eq!(shaped.load_shape, LoadShape::EagerMaterialization);
        }
    }

    #[test]
    fn shipped_candle_population_is_manifest_derived_and_mode_exact() {
        let raw = include_str!("../../../config/manifests/builtin.models.jsonc");
        let manifest: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(raw))
                .expect("builtin model manifest parses");
        let routes =
            declared_candle_deferred_routes(manifest["models"].as_array().expect("models array"));
        for expected in [
            ("z_image", "z_image"),
            ("z_image", "z_image_control"),
            ("z_image_turbo", "z_image_turbo"),
            ("z_image_turbo", "z_image_turbo_control"),
            ("z_image_edit", "z_image_turbo"),
            ("lens", "lens"),
            ("lens_turbo", "lens_turbo"),
        ] {
            assert!(
                routes.contains(&(expected.0.to_owned(), expected.1.to_owned())),
                "missing shipped Candle declaration {expected:?}",
            );
        }
        assert!(!routes.contains(&("z_image_turbo".to_owned(), "z_image_control".to_owned())));

        let lens = shipped_model("lens");
        let lens_style = MemoryRouteSelector {
            backend: MemoryRouteBackend::Candle,
            provider: "lens",
            tier: MemoryRouteTier::Q4,
            mode: MemoryRouteMode::StyleVariations,
            overlay: MemoryRouteOverlay::None,
            load_profile: MemoryRouteLoadProfile::Plain,
        };
        assert!(!manifest_declares_selector(&lens, lens_style));
        let ordinary_z_edit = MemoryRouteSelector {
            backend: MemoryRouteBackend::Candle,
            provider: "z_image",
            tier: MemoryRouteTier::Q4,
            mode: MemoryRouteMode::EditImage,
            overlay: MemoryRouteOverlay::None,
            load_profile: MemoryRouteLoadProfile::Plain,
        };
        assert!(!manifest_declares_selector(
            &shipped_model("z_image"),
            ordinary_z_edit,
        ));

        let native_turbo_edit = MemoryRouteSelector {
            provider: "z_image_turbo",
            ..ordinary_z_edit
        };
        assert!(!manifest_declares_selector(
            &shipped_model("z_image_turbo"),
            native_turbo_edit,
        ));
        let edit = shipped_model("z_image_edit");
        let edit_rows = edit["candle"]["memoryStrategyContract"]["implementations"]
            .as_array()
            .expect("Z-Image Edit Candle implementation rows");
        assert_eq!(
            edit_rows.len(),
            1,
            "the alias owns one composed top-rung row"
        );
        assert_eq!(
            edit_rows[0]["rung"].as_str(),
            Some("bounded_transformer_residency")
        );
        assert_eq!(
            edit_rows[0]["engagedRungs"],
            serde_json::json!([
                "resident",
                "staged_residency",
                "bounded_decode",
                "bounded_attention",
                "bounded_transformer_residency"
            ])
        );
        let edit_text = MemoryRouteSelector {
            backend: MemoryRouteBackend::Candle,
            provider: "z_image_turbo",
            tier: MemoryRouteTier::Q4,
            mode: MemoryRouteMode::TextToImage,
            overlay: MemoryRouteOverlay::None,
            load_profile: MemoryRouteLoadProfile::Plain,
        };
        assert!(!manifest_declares_selector(&edit, edit_text));
    }

    #[test]
    fn legacy_candle_qwen_flux_and_mage_shaping_is_unchanged() {
        for (provider, profile) in [
            ("qwen_image", MemoryRouteLoadProfile::Plain),
            ("qwen_image_edit", MemoryRouteLoadProfile::Plain),
            ("flux1_dev", MemoryRouteLoadProfile::SingleControl),
            ("flux2_dev", MemoryRouteLoadProfile::Plain),
            ("mage_flow_base", MemoryRouteLoadProfile::Plain),
        ] {
            let shaped = apply_registered_load_shape(
                MemoryRouteBackend::Candle,
                provider,
                MemoryRouteMode::TextToImage,
                spec(MemoryRouteTier::Q4, profile),
                false,
            );
            assert_eq!(
                shaped.load_shape,
                LoadShape::DeferredMaterialization,
                "legacy Candle route {provider} changed shape",
            );
            assert_eq!(
                shaped.load_shape_declaration_result,
                LoadShapeDeclarationResult::NotEvaluated
            );
        }
    }

    #[test]
    fn declaration_axes_source_and_alias_population_fail_closed() {
        let base = declaration(
            "z_image",
            &["q4"],
            &["text_to_image", "style_variations"],
            &["none", "lora"],
        );
        let edit = declaration(
            "z_image_turbo",
            &["q4"],
            &["edit_image", "image_to_image"],
            &["none", "lora"],
        );
        let mut edit = edit;
        edit["id"] = Value::String("z_image_edit".to_owned());
        let plain = spec(MemoryRouteTier::Q4, MemoryRouteLoadProfile::Plain);
        let apply = |provider: &str,
                     tier: Option<&str>,
                     mode: Option<MemoryRouteMode>,
                     manifest: &JsonObject<String, Value>,
                     spec: LoadSpec| {
            evaluate_declared_mlx_load_shape_with(provider, tier, mode, manifest, spec, |_| true)
        };
        assert_eq!(
            apply(
                "z_image",
                Some("q4"),
                Some(MemoryRouteMode::TextToImage),
                &base,
                plain.clone(),
            )
            .load_shape_declaration_result,
            LoadShapeDeclarationResult::Applied
        );
        assert_eq!(
            apply(
                "z_image_turbo",
                Some("q4"),
                Some(MemoryRouteMode::EditImage),
                &edit,
                plain.clone(),
            )
            .load_shape_declaration_result,
            LoadShapeDeclarationResult::Applied
        );
        for (provider, tier, mode, manifest) in [
            (
                "z_image",
                Some("q8"),
                Some(MemoryRouteMode::TextToImage),
                &base,
            ),
            (
                "z_image",
                Some("q4"),
                Some(MemoryRouteMode::EditImage),
                &base,
            ),
            (
                "z_image_turbo",
                Some("q4"),
                Some(MemoryRouteMode::TextToImage),
                &edit,
            ),
            (
                "z_image_turbo",
                None,
                Some(MemoryRouteMode::EditImage),
                &edit,
            ),
        ] {
            assert_eq!(
                apply(provider, tier, mode, manifest, plain.clone()).load_shape_declaration_result,
                LoadShapeDeclarationResult::Refused
            );
        }
        assert_eq!(
            apply(
                "z_image",
                Some("q4"),
                Some(MemoryRouteMode::TextToImage),
                &base,
                LoadSpec::new(WeightsSource::File("fixture.safetensors".into())),
            )
            .load_shape_declaration_result,
            LoadShapeDeclarationResult::Refused
        );

        let mut wrong_provider = base;
        wrong_provider["mlx"]["memoryStrategyContract"]["provider"] =
            Value::String("z_image_turbo".to_owned());
        assert_eq!(
            apply(
                "z_image",
                Some("q4"),
                Some(MemoryRouteMode::TextToImage),
                &wrong_provider,
                plain,
            )
            .load_shape_declaration_result,
            LoadShapeDeclarationResult::Refused
        );
    }

    #[test]
    fn chroma_external_text_encoder_is_visible_to_the_provider_predicate() {
        let manifest = declaration(
            "chroma1_base",
            &["bf16", "q4", "q8"],
            &["text_to_image", "style_variations"],
            &["none"],
        );
        let mut external = spec(MemoryRouteTier::Q4, MemoryRouteLoadProfile::Plain);
        external.text_encoder = Some(WeightsSource::File("external-te.safetensors".into()));

        let shaped = evaluate_declared_mlx_load_shape_with(
            "chroma1_base",
            Some("q4"),
            Some(MemoryRouteMode::TextToImage),
            &manifest,
            external,
            |candidate| candidate.text_encoder.is_none(),
        );
        assert_eq!(
            shaped.load_shape_declaration_result,
            LoadShapeDeclarationResult::Refused
        );
        assert_eq!(shaped.load_shape, LoadShape::EagerMaterialization);
        assert!(shaped.text_encoder.is_some());
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
    fn finite_witness_is_the_typed_rule_cross_product() {
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
                        assert_eq!(
                            rule_coordinates_match(selector),
                            witnesses.contains(&selector),
                            "typed witness disagrees with the finite rule cross-product for {selector:?}",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn no_btr_declaration_is_the_only_legacy_fallback() {
        let no_contract = serde_json::json!({ "id": "qwen_image", "mlx": {} })
            .as_object()
            .unwrap()
            .clone();
        let no_btr = serde_json::json!({
            "id": "qwen_image",
            "mlx": { "memoryStrategyContract": { "provider": "qwen_image", "implementations": [{
                "rung": "bounded_decode"
            }] } }
        })
        .as_object()
        .unwrap()
        .clone();
        let malformed = serde_json::json!({
            "id": "qwen_image",
            "mlx": { "memoryStrategyContract": { "provider": "qwen_image", "implementations": {} } }
        })
        .as_object()
        .unwrap()
        .clone();
        let evaluate = |manifest: &JsonObject<String, Value>| {
            evaluate_declared_mlx_load_shape_with(
                "qwen_image",
                Some("q4"),
                Some(MemoryRouteMode::TextToImage),
                manifest,
                spec(MemoryRouteTier::Q4, MemoryRouteLoadProfile::Plain),
                |_| true,
            )
        };
        let fallback = evaluate(&no_contract);
        assert_eq!(
            fallback.load_shape_declaration_result,
            LoadShapeDeclarationResult::NotEvaluated
        );
        let fallback = apply_registered_load_shape(
            MemoryRouteBackend::Mlx,
            "qwen_image",
            MemoryRouteMode::TextToImage,
            fallback,
            false,
        );
        assert_eq!(fallback.load_shape, LoadShape::DeferredMaterialization);
        assert_eq!(
            evaluate(&no_btr).load_shape_declaration_result,
            LoadShapeDeclarationResult::NotEvaluated
        );
        assert_eq!(
            evaluate(&malformed).load_shape_declaration_result,
            LoadShapeDeclarationResult::Refused
        );
    }

    #[test]
    fn sequential_owned_declaration_defers_source_judgment_to_the_late_shaper() {
        let spec = LoadSpec::new(WeightsSource::File("native-turbo.safetensors".into()));
        let deferred_to_legacy = evaluate_declared_mlx_load_shape_with(
            "z_image_turbo",
            Some("q4"),
            Some(MemoryRouteMode::TextToImage),
            &shipped_model("z_image_turbo"),
            spec,
            |_| panic!("sequential-owned declaration must not call the early provider predicate"),
        );
        assert_eq!(
            deferred_to_legacy.load_shape_declaration_result,
            LoadShapeDeclarationResult::NotEvaluated,
        );
        assert!(matches!(deferred_to_legacy.weights, WeightsSource::File(_)));
    }

    #[test]
    fn existing_declared_qwen_route_matches_legacy_shaping() {
        let manifest = declaration(
            "qwen_image",
            &["bf16", "q4", "q8"],
            &["text_to_image", "edit_image"],
            &["none", "lora"],
        );
        for (mode, profile) in [
            (MemoryRouteMode::TextToImage, MemoryRouteLoadProfile::Plain),
            (MemoryRouteMode::EditImage, MemoryRouteLoadProfile::Lora),
        ] {
            let input = spec(MemoryRouteTier::Q4, profile);
            let legacy = apply_registered_load_shape(
                MemoryRouteBackend::Mlx,
                "qwen_image",
                mode,
                input.clone(),
                false,
            );
            let declared = evaluate_declared_mlx_load_shape_with(
                "qwen_image",
                Some("q4"),
                Some(mode),
                &manifest,
                input,
                |_| true,
            );
            assert_eq!(
                declared.load_shape_declaration_result,
                LoadShapeDeclarationResult::Applied
            );
            assert_eq!(declared.load_shape, legacy.load_shape);
        }
    }

    #[test]
    fn shipped_preexisting_btr_routes_preserve_legacy_positive_surfaces() {
        for (model_id, provider, tier_name, tier, mode) in [
            (
                "z_image_turbo",
                "z_image_turbo",
                "q4",
                MemoryRouteTier::Q4,
                MemoryRouteMode::TextToImage,
            ),
            (
                "qwen_image",
                "qwen_image",
                "q4",
                MemoryRouteTier::Q4,
                MemoryRouteMode::TextToImage,
            ),
            (
                "qwen_image_edit_2511",
                "qwen_image_edit",
                "q4",
                MemoryRouteTier::Q4,
                MemoryRouteMode::EditImage,
            ),
            (
                "krea_2_turbo",
                "krea_2_turbo",
                "q4",
                MemoryRouteTier::Q4,
                MemoryRouteMode::TextToImage,
            ),
            (
                "sdxl",
                "sdxl",
                "bf16",
                MemoryRouteTier::Bf16,
                MemoryRouteMode::TextToImage,
            ),
            (
                "realvisxl",
                "sdxl",
                "bf16",
                MemoryRouteTier::Bf16,
                MemoryRouteMode::TextToImage,
            ),
        ] {
            let input = spec(tier, MemoryRouteLoadProfile::Plain).with_resolved_route(model_id);
            let legacy = apply_registered_load_shape(
                MemoryRouteBackend::Mlx,
                provider,
                mode,
                input.clone(),
                false,
            );
            let declared = evaluate_declared_mlx_load_shape_with(
                provider,
                Some(tier_name),
                Some(mode),
                &shipped_model(model_id),
                input,
                |_| true,
            );
            let selector = MemoryRouteSelector {
                backend: MemoryRouteBackend::Mlx,
                provider,
                tier,
                mode,
                overlay: MemoryRouteOverlay::None,
                load_profile: MemoryRouteLoadProfile::Plain,
            };
            let expected_authority =
                if matching_rules(selector).any(|rule| !rule.requires_sequential_selection) {
                    LoadShapeDeclarationResult::Applied
                } else {
                    LoadShapeDeclarationResult::NotEvaluated
                };
            assert_eq!(
                declared.load_shape_declaration_result, expected_authority,
                "{model_id} declaration ownership diverged from its exact typed rule",
            );
            assert_eq!(
                declared.load_shape, legacy.load_shape,
                "{model_id} declaration evaluation changed its preexisting positive surface",
            );
        }
    }

    #[test]
    fn shipped_sc18457_declarations_have_exact_nonoverlapping_axes() {
        let axes = |id: &str| {
            let model = shipped_model(id);
            let contract = model["mlx"]["memoryStrategyContract"]
                .as_object()
                .expect("memory strategy contract");
            let provider = contract["provider"].as_str().expect("provider").to_owned();
            let rows = contract["implementations"]
                .as_array()
                .expect("implementations")
                .iter()
                .filter(|row| row["rung"].as_str() == Some("bounded_transformer_residency"))
                .map(|row| {
                    serde_json::json!({
                        "tiers": row["tiers"],
                        "modes": row["modes"],
                        "overlays": row["overlays"],
                    })
                })
                .collect::<Vec<_>>();
            (provider, rows)
        };
        for id in ["anima_base", "anima_aesthetic", "anima_turbo"] {
            assert_eq!(
                axes(id),
                (
                    id.to_owned(),
                    vec![serde_json::json!({
                        "tiers": ["bf16", "q4", "q8"],
                        "modes": ["text_to_image"],
                        "overlays": ["none", "lora"],
                    })]
                )
            );
        }
        for id in ["chroma1_hd", "chroma1_base", "chroma1_flash"] {
            assert_eq!(
                axes(id),
                (
                    id.to_owned(),
                    vec![serde_json::json!({
                        "tiers": ["bf16", "q4", "q8"],
                        "modes": ["text_to_image", "style_variations"],
                        "overlays": ["none"],
                    })]
                )
            );
        }
        assert_eq!(
            axes("z_image"),
            (
                "z_image".to_owned(),
                vec![serde_json::json!({
                    "tiers": ["bf16", "q4", "q8"],
                    "modes": ["text_to_image", "style_variations"],
                    "overlays": ["none", "lora"],
                })]
            )
        );
        let base_z = shipped_model("z_image");
        assert!(base_z["mlx"]["memoryStrategyContract"]["implementations"]
            .as_array()
            .expect("base Z implementations")
            .iter()
            .all(|row| {
                row["modes"] == serde_json::json!(["text_to_image", "style_variations"])
                    && row["overlays"] == serde_json::json!(["none", "lora"])
            }));
        assert_eq!(
            axes("z_image_edit"),
            (
                "z_image_turbo".to_owned(),
                vec![serde_json::json!({
                    "tiers": ["bf16", "q4", "q8"],
                    "modes": ["edit_image", "image_to_image"],
                    "overlays": ["none", "lora"],
                })]
            )
        );
        assert_eq!(
            axes("kolors"),
            (
                "kolors".to_owned(),
                vec![
                    serde_json::json!({
                        "tiers": ["bf16"],
                        "modes": ["text_to_image", "edit_image", "character_image", "style_variations"],
                        "overlays": ["none", "identity"],
                    }),
                    serde_json::json!({
                        "tiers": ["q4", "q8"],
                        "modes": ["text_to_image", "edit_image", "character_image", "style_variations"],
                        "overlays": ["none", "lora", "identity"],
                    }),
                ]
            )
        );
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
