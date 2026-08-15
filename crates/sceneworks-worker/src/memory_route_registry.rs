//! Typed production registry for load shapes that can reach deferred materialization.
//!
//! This is deliberately executable data shared by the worker route selectors and the engine-facts
//! dumper. Consumers must not recover it by parsing Rust source text: formatting and equivalent
//! control-flow refactors are not route facts.

use gen_core::{
    LoadShape, LoadShapeDeclarationResult, LoadSpec, MemoryStrategy, MemoryStrategySupport,
    OffloadPolicy, Precision, Quant, WeightsSource,
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
    LoraPid,
    SingleControl,
    MultiControl,
    IpAdapter,
    Pid,
    Identity,
}

impl MemoryRouteLoadProfile {
    pub const ALL: [Self; 8] = [
        Self::Plain,
        Self::Lora,
        Self::LoraPid,
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
            Self::LoraPid => "lora_pid",
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
            Self::Lora | Self::LoraPid => MemoryRouteOverlay::Lora,
            Self::SingleControl | Self::MultiControl => MemoryRouteOverlay::Control,
            Self::IpAdapter | Self::Identity => MemoryRouteOverlay::Identity,
        }
    }

    fn from_spec(spec: &LoadSpec) -> Option<Self> {
        let has_lora = !spec.adapters.is_empty();
        let has_pid = spec.pid.is_some();
        let has_other_profile = spec.control.is_some()
            || !spec.extra_controls.is_empty()
            || spec.ip_adapter.is_some()
            || spec.identity.is_some();
        if has_lora && has_pid && !has_other_profile {
            return Some(Self::LoraPid);
        }
        let profiles = [
            has_lora.then_some(Self::Lora),
            (spec.control.is_some() && spec.extra_controls.is_empty())
                .then_some(Self::SingleControl),
            (!spec.extra_controls.is_empty()).then_some(Self::MultiControl),
            spec.ip_adapter.is_some().then_some(Self::IpAdapter),
            has_pid.then_some(Self::Pid),
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

/// Exact public request axes which are not part of a [`LoadSpec`]. Declaration-owned request
/// strategies must bind these before the provider contract is allowed into the shared selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryRouteRequestContext {
    pub mode: MemoryRouteMode,
    pub reference_count: u32,
    pub use_pid: bool,
    pub has_phases: bool,
}

/// Tri-state manifest authority for request-scoped strategy selection. This is deliberately
/// independent of `LoadShapeDeclarationResult`: staged component residency does not imply deferred
/// materialization.
pub enum DeclaredCandleStrategyContract {
    NoRelevantDeclaration,
    Applied {
        contract: Box<gen_core::MemoryProviderContract>,
        provider_overlay: Option<String>,
        provider_mode: String,
    },
    Refused,
}

/// Typed, non-authorizing diagnostic for a valid Krea dense-diff load which cannot use the
/// reopenable deferred DiT. The load remains `Refused + Eager`; request admission still decides
/// whether that conservative resident path fits the live host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MlxLoadShapeDeclarationWarning {
    DenseDiffRequiresEager,
}

pub fn mlx_load_shape_declaration_warning(
    spec: &LoadSpec,
) -> Option<MlxLoadShapeDeclarationWarning> {
    (spec.load_shape_declaration_result == LoadShapeDeclarationResult::Refused
        && spec.load_shape == LoadShape::EagerMaterialization
        && spec.adapters.iter().any(|adapter| {
            gen_core::safetensors_path_tensor_headers(&adapter.path)
                .ok()
                .is_some_and(|headers| {
                    headers.iter().any(|header| {
                        header.name.ends_with(".diff") || header.name.ends_with(".diff_b")
                    })
                })
        }))
    .then_some(MlxLoadShapeDeclarationWarning::DenseDiffRequiresEager)
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
const IDENTITY_ONLY: &[MemoryRouteLoadProfile] = &[MemoryRouteLoadProfile::Identity];
const SINGLE_CONTROL: &[MemoryRouteLoadProfile] = &[MemoryRouteLoadProfile::SingleControl];
const PLAIN_LORA: &[MemoryRouteLoadProfile] =
    &[MemoryRouteLoadProfile::Plain, MemoryRouteLoadProfile::Lora];
const PLAIN_LORA_PID: &[MemoryRouteLoadProfile] = &[
    MemoryRouteLoadProfile::Plain,
    MemoryRouteLoadProfile::Lora,
    MemoryRouteLoadProfile::LoraPid,
    MemoryRouteLoadProfile::Pid,
];
const PLAIN_SINGLE_CONTROL: &[MemoryRouteLoadProfile] = &[
    MemoryRouteLoadProfile::Plain,
    MemoryRouteLoadProfile::SingleControl,
];
const PLAIN_SINGLE_CONTROL_IP: &[MemoryRouteLoadProfile] = &[
    MemoryRouteLoadProfile::Plain,
    MemoryRouteLoadProfile::SingleControl,
    MemoryRouteLoadProfile::IpAdapter,
];
const LEGACY_Z_IMAGE_TURBO_PROFILES: &[MemoryRouteLoadProfile] = &[
    MemoryRouteLoadProfile::Plain,
    MemoryRouteLoadProfile::Lora,
    MemoryRouteLoadProfile::SingleControl,
    MemoryRouteLoadProfile::MultiControl,
    MemoryRouteLoadProfile::IpAdapter,
    MemoryRouteLoadProfile::Pid,
    MemoryRouteLoadProfile::Identity,
];
const TEXT_ONLY: &[MemoryRouteMode] = &[MemoryRouteMode::TextToImage];
const CHARACTER_ONLY: &[MemoryRouteMode] = &[MemoryRouteMode::CharacterImage];
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
        load_profiles: PLAIN_LORA_PID,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "krea_2_raw",
        tiers: ALL_TIERS,
        modes: TEXT_ONLY,
        load_profiles: PLAIN_LORA_PID,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "krea_2_edit",
        tiers: ALL_TIERS,
        modes: EDIT_MODES,
        load_profiles: PLAIN_LORA_PID,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "krea_2_turbo_edit",
        tiers: ALL_TIERS,
        modes: EDIT_MODES,
        load_profiles: PLAIN_LORA_PID,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "pulid_flux",
        tiers: ALL_TIERS,
        modes: CHARACTER_ONLY,
        load_profiles: IDENTITY_ONLY,
        requires_sequential_selection: false,
        legacy_shaping: false,
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
        load_profiles: LEGACY_Z_IMAGE_TURBO_PROFILES,
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

fn implementation_declares_selector(
    implementation: &Value,
    contract_provider: &str,
    selector: MemoryRouteSelector,
) -> bool {
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
}

fn mlx_request_implementation_matches(
    implementation: &Value,
    contract_provider: &str,
    selector: MemoryRouteSelector,
    spec: &LoadSpec,
    context: MemoryRouteRequestContext,
    requires_request_context: bool,
) -> Result<bool, ()> {
    if !implementation_declares_selector(implementation, contract_provider, selector) {
        return Ok(false);
    }
    let implementation = implementation.as_object().ok_or(())?;
    let Some(request_contexts) = implementation.get("requestContexts") else {
        if requires_request_context {
            return Err(());
        }
        // Existing declaration-owned routes predate the request-context schema. Their exact
        // provider/tier/mode/overlay/source predicate remains authoritative and unchanged.
        return Ok(true);
    };
    let runtime_provider = implementation
        .get("runtimeProvider")
        .and_then(Value::as_str)
        .unwrap_or(contract_provider);
    if runtime_provider != selector.provider {
        return Ok(false);
    }
    let includes = |field: &str, expected: &str| -> Result<bool, ()> {
        Ok(implementation
            .get(field)
            .and_then(Value::as_array)
            .ok_or(())?
            .iter()
            .any(|value| value.as_str() == Some(expected)))
    };
    let Some(load_profile) = MemoryRouteLoadProfile::from_spec(spec) else {
        return Ok(false);
    };
    let source_kind = match spec.weights {
        WeightsSource::Dir(_) => "dir",
        WeightsSource::File(_) => "file",
    };
    if !includes("loadProfiles", load_profile.as_str())? || !includes("sourceKinds", source_kind)? {
        return Ok(false);
    }
    let expected_provider_overlay = match load_profile {
        MemoryRouteLoadProfile::Identity | MemoryRouteLoadProfile::IpAdapter => "identity",
        MemoryRouteLoadProfile::SingleControl | MemoryRouteLoadProfile::MultiControl => "control",
        MemoryRouteLoadProfile::Plain
        | MemoryRouteLoadProfile::Lora
        | MemoryRouteLoadProfile::LoraPid
        | MemoryRouteLoadProfile::Pid => "none",
    };
    if implementation
        .get("providerOverlay")
        .and_then(Value::as_str)
        != Some(expected_provider_overlay)
    {
        return Err(());
    }
    let request_contexts = request_contexts.as_array().ok_or(())?;
    let matches = request_contexts
        .iter()
        .map(|request_context| request_strategy_context_matches(request_context, context))
        .collect::<Result<Vec<_>, _>>()?;
    let matching = request_contexts
        .iter()
        .zip(matches)
        .filter_map(|(request_context, matched)| matched.then_some(request_context))
        .collect::<Vec<_>>();
    let [matching] = matching.as_slice() else {
        return Ok(false);
    };
    let expected_provider_mode = match context.mode {
        MemoryRouteMode::EditImage => "edit_image",
        MemoryRouteMode::CharacterImage if context.reference_count == 1 => "image_to_image",
        MemoryRouteMode::TextToImage if context.reference_count == 1 => "image_to_image",
        MemoryRouteMode::TextToImage => "text_to_image",
        other => other.as_str(),
    };
    Ok(matching.get("providerMode").and_then(Value::as_str) == Some(expected_provider_mode))
}

fn manifest_contract(
    manifest: &JsonObject<String, Value>,
    backend: MemoryRouteBackend,
) -> Option<&JsonObject<String, Value>> {
    manifest
        .get(backend.as_str())
        .and_then(Value::as_object)?
        .get("memoryStrategyContract")
        .and_then(Value::as_object)
}

fn request_strategy_implementation_matches(
    implementation: &Value,
    contract_provider: &str,
    runtime_provider: &str,
    tier: MemoryRouteTier,
    load_profile: MemoryRouteLoadProfile,
    source_kind: &str,
    context: MemoryRouteRequestContext,
) -> Result<bool, ()> {
    let implementation = implementation.as_object().ok_or(())?;
    if implementation.get("rung").and_then(Value::as_str) != Some("staged_residency") {
        return Ok(false);
    }
    // `requestContexts` is the declaration-driven ownership marker. Older staged rows which only
    // describe calibration inventory remain on their unchanged legacy paths.
    let Some(request_contexts) = implementation.get("requestContexts") else {
        return Ok(false);
    };
    let request_contexts = request_contexts.as_array().ok_or(())?;
    let provider = implementation
        .get("runtimeProvider")
        .and_then(Value::as_str)
        .unwrap_or(contract_provider);
    if provider != runtime_provider {
        return Ok(false);
    }
    let includes = |field: &str, expected: &str| -> Result<bool, ()> {
        Ok(implementation
            .get(field)
            .and_then(Value::as_array)
            .ok_or(())?
            .iter()
            .any(|value| value.as_str() == Some(expected)))
    };
    if !includes("tiers", tier.as_str())?
        || !includes("modes", context.mode.as_str())?
        || !includes("overlays", load_profile.overlay().as_str())?
        || !includes("loadProfiles", load_profile.as_str())?
        || !includes("sourceKinds", source_kind)?
    {
        return Ok(false);
    }
    for request_context in request_contexts {
        if request_strategy_context_matches(request_context, context)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn request_strategy_context_matches(
    request_context: &Value,
    context: MemoryRouteRequestContext,
) -> Result<bool, ()> {
    let request_context = request_context.as_object().ok_or(())?;
    let mode = request_context
        .get("mode")
        .and_then(Value::as_str)
        .ok_or(())?;
    let provider_mode = request_context
        .get("providerMode")
        .and_then(Value::as_str)
        .ok_or(())?;
    if !matches!(
        provider_mode,
        "text_to_image" | "image_to_image" | "edit_image"
    ) {
        return Err(());
    }
    let reference_counts = request_context
        .get("referenceCounts")
        .and_then(Value::as_array)
        .ok_or(())?;
    let pid = request_context
        .get("pid")
        .and_then(Value::as_array)
        .ok_or(())?;
    let has_phases = request_context
        .get("hasPhases")
        .and_then(Value::as_bool)
        .ok_or(())?;
    Ok(mode == context.mode.as_str()
        && reference_counts
            .iter()
            .any(|value| value.as_u64() == Some(u64::from(context.reference_count)))
        && pid
            .iter()
            .any(|value| value.as_bool() == Some(context.use_pid))
        && has_phases == context.has_phases)
}

/// Intersect an exact manifest request-strategy declaration with the real pinned provider
/// predicate. A relevant but malformed, ambiguous, or unsupported declaration is terminally
/// refused; only complete absence preserves the pre-declaration selector path.
pub fn declared_candle_request_strategy_contract(
    runtime_provider: &str,
    resolved_tier: Option<&str>,
    manifest: &JsonObject<String, Value>,
    spec: &LoadSpec,
    context: MemoryRouteRequestContext,
) -> DeclaredCandleStrategyContract {
    declared_candle_request_strategy_contract_with(
        runtime_provider,
        resolved_tier,
        manifest,
        spec,
        context,
        |candidate| {
            crate::inference_runtime::media()
                .memory_strategy_contract(runtime_provider, candidate)
                .ok()
                .flatten()
        },
    )
}

fn declared_candle_request_strategy_contract_with(
    runtime_provider: &str,
    resolved_tier: Option<&str>,
    manifest: &JsonObject<String, Value>,
    spec: &LoadSpec,
    context: MemoryRouteRequestContext,
    provider_contract: impl FnOnce(&LoadSpec) -> Option<gen_core::MemoryProviderContract>,
) -> DeclaredCandleStrategyContract {
    let Some(contract) = manifest_contract(manifest, MemoryRouteBackend::Candle) else {
        return DeclaredCandleStrategyContract::NoRelevantDeclaration;
    };
    let Some(contract_provider) = contract.get("provider").and_then(Value::as_str) else {
        return DeclaredCandleStrategyContract::Refused;
    };
    let Some(implementations) = contract.get("implementations").and_then(Value::as_array) else {
        return DeclaredCandleStrategyContract::Refused;
    };
    let relevant = implementations.iter().any(|implementation| {
        implementation.get("rung").and_then(Value::as_str) == Some("staged_residency")
            && implementation.get("requestContexts").is_some()
    });
    if !relevant {
        return DeclaredCandleStrategyContract::NoRelevantDeclaration;
    }
    if spec.resolved_route.as_deref() != manifest.get("id").and_then(Value::as_str) {
        return DeclaredCandleStrategyContract::Refused;
    }
    let (Some(tier), Some(load_profile)) = (
        resolved_tier.and_then(MemoryRouteTier::from_resolved_tier),
        MemoryRouteLoadProfile::from_spec(spec),
    ) else {
        return DeclaredCandleStrategyContract::Refused;
    };
    let source_kind = match &spec.weights {
        WeightsSource::Dir(_) => "dir",
        WeightsSource::File(_) => "file",
    };
    let matches = implementations
        .iter()
        .map(|implementation| {
            request_strategy_implementation_matches(
                implementation,
                contract_provider,
                runtime_provider,
                tier,
                load_profile,
                source_kind,
                context,
            )
        })
        .collect::<Result<Vec<_>, _>>();
    let Ok(matches) = matches else {
        return DeclaredCandleStrategyContract::Refused;
    };
    if matches.into_iter().filter(|matched| *matched).count() != 1 {
        return DeclaredCandleStrategyContract::Refused;
    }
    let provider_contract = provider_contract(spec);
    let matching_implementation = implementations.iter().find(|implementation| {
        request_strategy_implementation_matches(
            implementation,
            contract_provider,
            runtime_provider,
            tier,
            load_profile,
            source_kind,
            context,
        ) == Ok(true)
    });
    let provider_overlay = match matching_implementation
        .and_then(|implementation| implementation.get("providerOverlay"))
        .and_then(Value::as_str)
    {
        Some("none") => None,
        Some(value @ ("lora" | "identity" | "control")) => Some(value.to_owned()),
        _ => return DeclaredCandleStrategyContract::Refused,
    };
    let Some(request_contexts) = matching_implementation
        .and_then(|implementation| implementation.get("requestContexts"))
        .and_then(Value::as_array)
    else {
        return DeclaredCandleStrategyContract::Refused;
    };
    let matching_request_contexts = request_contexts
        .iter()
        .filter(|request_context| {
            request_strategy_context_matches(request_context, context) == Ok(true)
        })
        .collect::<Vec<_>>();
    let [matching_request_context] = matching_request_contexts.as_slice() else {
        return DeclaredCandleStrategyContract::Refused;
    };
    let Some(provider_mode) = matching_request_context
        .get("providerMode")
        .and_then(Value::as_str)
        .map(str::to_owned)
    else {
        return DeclaredCandleStrategyContract::Refused;
    };
    match provider_contract {
        Some(contract)
            if contract
                .capability(MemoryStrategy::StagedResidency)
                .is_some_and(|capability| {
                    capability.support == MemoryStrategySupport::Implemented
                }) =>
        {
            DeclaredCandleStrategyContract::Applied {
                contract: Box::new(contract),
                provider_overlay,
                provider_mode,
            }
        }
        _ => DeclaredCandleStrategyContract::Refused,
    }
}

/// Whether this route-local manifest owns request-strategy reachability for `runtime_provider`.
/// Used by artifact certification to choose the same exact manifest-backed identity path as the
/// declaration evaluator, without a provider allow-list.
pub fn candle_manifest_declares_request_strategy_provider(
    manifest: &JsonObject<String, Value>,
    runtime_provider: &str,
) -> bool {
    let Some(contract) = manifest_contract(manifest, MemoryRouteBackend::Candle) else {
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
                implementation.get("rung").and_then(Value::as_str) == Some("staged_residency")
                    && implementation.get("requestContexts").is_some()
                    && implementation
                        .get("runtimeProvider")
                        .and_then(Value::as_str)
                        .unwrap_or(contract_provider)
                        == runtime_provider
            })
        })
}

fn manifest_declares_selector(
    manifest: &JsonObject<String, Value>,
    selector: MemoryRouteSelector,
) -> bool {
    let Some(contract) = manifest_contract(manifest, selector.backend) else {
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
                implementation_declares_selector(implementation, contract_provider, selector)
            })
        })
}

/// Whether the exact manifest row declares staged residency in the same request as rung 4.
/// This is a contract coordinate, not a provider allow-list: it tells the predicate which real
/// load policy must expose the declared BTR implementation while the returned request retains its
/// independently selected Resident/Sequential policy.
fn manifest_selector_requires_staged_residency(
    manifest: &JsonObject<String, Value>,
    selector: MemoryRouteSelector,
) -> Result<bool, ()> {
    let contract = manifest_contract(manifest, selector.backend).ok_or(())?;
    let contract_provider = contract.get("provider").and_then(Value::as_str).ok_or(())?;
    let implementations = contract
        .get("implementations")
        .and_then(Value::as_array)
        .ok_or(())?;
    let mut requirements = implementations
        .iter()
        .filter(|implementation| {
            implementation_declares_selector(implementation, contract_provider, selector)
        })
        .map(|implementation| match implementation.get("engagedRungs") {
            None => Ok(false),
            Some(Value::Array(rungs)) => Ok(rungs
                .iter()
                .any(|rung| rung.as_str() == Some("staged_residency"))),
            Some(_) => Err(()),
        });
    let required = requirements.next().transpose()?.ok_or(())?;
    requirements.try_fold(required, |expected, next| {
        let next = next?;
        (next == expected).then_some(expected).ok_or(())
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
    let context = MemoryRouteRequestContext {
        mode: mode.unwrap_or(MemoryRouteMode::TextToImage),
        reference_count: 0,
        use_pid: spec.pid.is_some(),
        has_phases: false,
    };
    evaluate_declared_mlx_load_shape_for_request(
        provider,
        resolved_tier,
        mode,
        manifest,
        spec,
        context,
    )
}

/// Request-exact MLX declaration intersection. Krea routes use the public request coordinate plus
/// the provider-owned reference/PiD/phase axes; older declarations without `requestContexts` keep
/// their established behavior through [`evaluate_declared_mlx_load_shape`].
pub fn evaluate_declared_mlx_load_shape_for_request(
    provider: &str,
    resolved_tier: Option<&str>,
    mode: Option<MemoryRouteMode>,
    manifest: &JsonObject<String, Value>,
    spec: LoadSpec,
    context: MemoryRouteRequestContext,
) -> LoadSpec {
    evaluate_declared_mlx_load_shape_for_request_with(
        provider,
        resolved_tier,
        mode,
        manifest,
        spec,
        context,
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

#[cfg(test)]
fn evaluate_declared_mlx_load_shape_with(
    provider: &str,
    resolved_tier: Option<&str>,
    mode: Option<MemoryRouteMode>,
    manifest: &JsonObject<String, Value>,
    spec: LoadSpec,
    provider_implements: impl FnOnce(&LoadSpec) -> bool,
) -> LoadSpec {
    let context = MemoryRouteRequestContext {
        mode: mode.unwrap_or(MemoryRouteMode::TextToImage),
        reference_count: 0,
        use_pid: spec.pid.is_some(),
        has_phases: false,
    };
    evaluate_declared_mlx_load_shape_for_request_with(
        provider,
        resolved_tier,
        mode,
        manifest,
        spec,
        context,
        provider_implements,
    )
}

fn evaluate_declared_mlx_load_shape_for_request_with(
    provider: &str,
    resolved_tier: Option<&str>,
    mode: Option<MemoryRouteMode>,
    manifest: &JsonObject<String, Value>,
    spec: LoadSpec,
    context: MemoryRouteRequestContext,
    provider_implements: impl FnOnce(&LoadSpec) -> bool,
) -> LoadSpec {
    match has_relevant_btr_declaration(manifest, MemoryRouteBackend::Mlx) {
        Ok(false) => return spec,
        Err(()) => return spec.with_refused_load_shape_declaration(),
        Ok(true) => {}
    }
    if let (Some(resolved_route), Some(manifest_route)) = (
        spec.resolved_route.as_deref(),
        manifest.get("id").and_then(Value::as_str),
    ) {
        if resolved_route != manifest_route {
            return spec.with_refused_load_shape_declaration();
        }
    }
    let (Some(tier), Some(mode), Some(load_profile)) = (
        resolved_tier.and_then(MemoryRouteTier::from_resolved_tier),
        mode,
        MemoryRouteLoadProfile::from_spec(&spec),
    ) else {
        return spec.with_refused_load_shape_declaration();
    };
    if context.mode != mode {
        return spec.with_refused_load_shape_declaration();
    }
    if context.use_pid != spec.pid.is_some() {
        return spec.with_refused_load_shape_declaration();
    }
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
    // A typed rule with no legacy shaper is request-context-owned. The manifest cannot turn that
    // exact authority back into an older context-free declaration by deleting requestContexts.
    let requires_request_context = matching_rules(selector).any(|rule| !rule.legacy_shaping);
    let Some(contract) = manifest_contract(manifest, MemoryRouteBackend::Mlx) else {
        return spec.with_refused_load_shape_declaration();
    };
    let Some(contract_provider) = contract.get("provider").and_then(Value::as_str) else {
        return spec.with_refused_load_shape_declaration();
    };
    let Some(implementations) = contract.get("implementations").and_then(Value::as_array) else {
        return spec.with_refused_load_shape_declaration();
    };
    if requires_request_context {
        let Some(manifest_route) = manifest.get("id").and_then(Value::as_str) else {
            return spec.with_refused_load_shape_declaration();
        };
        if spec.resolved_route.as_deref() != Some(manifest_route) {
            return spec.with_refused_load_shape_declaration();
        }
    }
    let matches = implementations
        .iter()
        .map(|implementation| {
            mlx_request_implementation_matches(
                implementation,
                contract_provider,
                selector,
                &spec,
                context,
                requires_request_context,
            )
        })
        .collect::<Result<Vec<_>, _>>();
    let Ok(matches) = matches else {
        return spec.with_refused_load_shape_declaration();
    };
    if matches.into_iter().filter(|matched| *matched).count() != 1 {
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

    let applied = spec.clone().with_applied_load_shape_declaration();
    let declaration_owned = matching_rules(selector).all(|rule| !rule.legacy_shaping);
    let provider_candidate = match manifest_selector_requires_staged_residency(manifest, selector) {
        Ok(true) if declaration_owned => applied
            .clone()
            .with_offload_policy(OffloadPolicy::Sequential),
        Ok(_) => applied.clone(),
        Err(()) => return spec.with_refused_load_shape_declaration(),
    };
    if provider_implements(&provider_candidate) {
        applied
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
        if !sequential_selected {
            return spec;
        }
        if let (Some(tier), Some(load_profile)) = (
            MemoryRouteTier::from_spec(&spec),
            MemoryRouteLoadProfile::from_spec(&spec),
        ) {
            let selector = MemoryRouteSelector {
                backend,
                provider: rule.provider,
                tier,
                mode,
                overlay: load_profile.overlay(),
                load_profile,
            };
            if !matching_rules(selector)
                .any(|rule| rule.legacy_shaping && rule.requires_sequential_selection)
            {
                return spec;
            }
        }
        return spec.with_load_shape(LoadShape::DeferredMaterialization);
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

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DeclaredRequestStrategyRouteWitness {
    pub provider: String,
    pub tier: MemoryRouteTier,
    pub mode: MemoryRouteMode,
    pub overlay: MemoryRouteOverlay,
    pub load_profile: MemoryRouteLoadProfile,
}

/// Typed manifest-derived Candle request-strategy population. Only rows carrying the exact
/// `requestContexts` ownership marker participate; malformed axes fail the facts dump instead of
/// silently broadening or disappearing.
pub fn declared_candle_request_strategy_route_witnesses(
    models: &[Value],
) -> Result<Vec<DeclaredRequestStrategyRouteWitness>, String> {
    let mut witnesses = Vec::new();
    for model in models {
        let Some(model) = model.as_object() else {
            return Err("builtin model entry is not an object".to_owned());
        };
        let Some(contract) = manifest_contract(model, MemoryRouteBackend::Candle) else {
            continue;
        };
        let contract_provider = contract
            .get("provider")
            .and_then(Value::as_str)
            .ok_or_else(|| "Candle memoryStrategyContract has no provider".to_owned())?;
        let implementations = contract
            .get("implementations")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("{contract_provider} has no implementations array"))?;
        for implementation in implementations {
            if implementation.get("rung").and_then(Value::as_str) != Some("staged_residency")
                || implementation.get("requestContexts").is_none()
            {
                continue;
            }
            let provider = implementation
                .get("runtimeProvider")
                .and_then(Value::as_str)
                .unwrap_or(contract_provider);
            let strings = |field: &str| -> Result<Vec<&str>, String> {
                implementation
                    .get(field)
                    .and_then(Value::as_array)
                    .ok_or_else(|| format!("{provider} staged declaration has no {field}"))?
                    .iter()
                    .map(|value| {
                        value.as_str().ok_or_else(|| {
                            format!("{provider} staged declaration has non-string {field}")
                        })
                    })
                    .collect()
            };
            let tiers = strings("tiers")?
                .into_iter()
                .map(|tier| {
                    MemoryRouteTier::from_resolved_tier(tier)
                        .ok_or_else(|| format!("{provider} has unknown staged tier {tier}"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let modes = strings("modes")?
                .into_iter()
                .map(|mode| {
                    MemoryRouteMode::from_request(mode)
                        .ok_or_else(|| format!("{provider} has unknown staged mode {mode}"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let overlays = strings("overlays")?
                .into_iter()
                .map(|overlay| match overlay {
                    "none" => Ok(MemoryRouteOverlay::None),
                    "lora" => Ok(MemoryRouteOverlay::Lora),
                    "control" => Ok(MemoryRouteOverlay::Control),
                    "identity" => Ok(MemoryRouteOverlay::Identity),
                    _ => Err(format!("{provider} has unknown staged overlay {overlay}")),
                })
                .collect::<Result<Vec<_>, _>>()?;
            let load_profiles = strings("loadProfiles")?
                .into_iter()
                .map(|profile| {
                    MemoryRouteLoadProfile::ALL
                        .into_iter()
                        .find(|candidate| candidate.as_str() == profile)
                        .ok_or_else(|| format!("{provider} has unknown load profile {profile}"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            for tier in &tiers {
                for mode in &modes {
                    for load_profile in &load_profiles {
                        if overlays.contains(&load_profile.overlay()) {
                            witnesses.push(DeclaredRequestStrategyRouteWitness {
                                provider: provider.to_owned(),
                                tier: *tier,
                                mode: *mode,
                                overlay: load_profile.overlay(),
                                load_profile: *load_profile,
                            });
                        }
                    }
                }
            }
        }
    }
    witnesses.sort_unstable();
    witnesses.dedup();
    Ok(witnesses)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn staged_contract(provider: &str) -> gen_core::MemoryProviderContract {
        let mut contract = gen_core::MemoryProviderContract::compatibility_default(
            provider,
            gen_core::MemoryBackendRealization::CandleCuda {
                device_residency: true,
                host_backed_weights: true,
                host_to_device_block_materialization: true,
                block_materialization: gen_core::MemoryWindowMaterialization::DeviceFormatTransfer,
            },
        );
        for capability in &mut contract.strategies {
            capability.support = if matches!(
                capability.strategy,
                MemoryStrategy::Resident | MemoryStrategy::StagedResidency
            ) {
                MemoryStrategySupport::Implemented
            } else {
                MemoryStrategySupport::Missing
            };
        }
        contract
    }

    fn request_strategy_declaration() -> JsonObject<String, Value> {
        serde_json::json!({
            "id": "krea_2_raw",
            "candle": { "memoryStrategyContract": {
                "abi": 1,
                "provider": "krea_2_raw",
                "implementations": [{
                    "rung": "staged_residency",
                    "fingerprint": "krea-candle-request-scoped-staged-residency-v1",
                    "tiers": ["bf16", "q4", "q8"],
                    "modes": ["text_to_image"],
                    "overlays": ["none", "lora"],
                    "loadProfiles": ["plain", "lora", "lora_pid", "pid"],
                    "sourceKinds": ["dir"],
                    "providerOverlay": "none",
                    "requestContexts": [
                        { "mode": "text_to_image", "providerMode": "text_to_image", "referenceCounts": [0], "pid": [false, true], "hasPhases": false },
                        { "mode": "text_to_image", "providerMode": "image_to_image", "referenceCounts": [1], "pid": [false, true], "hasPhases": false },
                        { "mode": "text_to_image", "providerMode": "text_to_image", "referenceCounts": [0], "pid": [false], "hasPhases": true }
                    ],
                    "engagedRungs": ["resident", "staged_residency"],
                    "parameters": {}, "parameterRanges": {}, "source": "fixture"
                }]
            }}
        })
        .as_object()
        .unwrap()
        .clone()
    }

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
            MemoryRouteLoadProfile::LoraPid => base
                .with_adapters(vec![gen_core::AdapterSpec::new(
                    "adapter.safetensors".into(),
                    1.0,
                    gen_core::AdapterKind::Lora,
                )])
                .with_pid(
                    WeightsSource::File("pid.safetensors".into()),
                    WeightsSource::Dir("gemma".into()),
                ),
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
    fn krea_raw_declaration_is_exact_true_cfg_generation_authority() {
        let manifest = shipped_model("krea_2_raw");
        let contract = manifest["mlx"]["memoryStrategyContract"]
            .as_object()
            .expect("Raw MLX memory contract");
        assert_eq!(contract["provider"], "krea_2_raw");
        assert!(manifest["mlx"].get("supportsSequentialOffload").is_none());
        assert_eq!(
            manifest["loraCompatibility"]["types"],
            serde_json::json!(["character", "style", "acceleration"]),
            "the product route exposes the supported low-rank family while adapter-role routing remains independent"
        );
        let implementations = contract["implementations"]
            .as_array()
            .expect("Raw implementation rows");
        assert_eq!(implementations.len(), 6);
        let (native, edit) = implementations.split_at(3);
        for implementation in native {
            assert_eq!(
                implementation["fingerprint"],
                "krea-2-mlx-full-ladder-native-pid-attn64m-window1-2026-08-03-v3"
            );
            assert_eq!(
                implementation["tiers"],
                serde_json::json!(["bf16", "q4", "q8"])
            );
            assert_eq!(
                implementation["modes"],
                serde_json::json!(["text_to_image"])
            );
            assert_eq!(
                implementation["overlays"],
                serde_json::json!(["none", "lora"])
            );
        }
        for implementation in edit {
            assert_eq!(implementation["runtimeProvider"], "krea_2_edit");
            assert_eq!(
                implementation["fingerprint"],
                "krea-2-mlx-full-ladder-native-pid-attn64m-window1-2026-08-03-v3"
            );
            assert_eq!(
                implementation["tiers"],
                serde_json::json!(["bf16", "q4", "q8"])
            );
            assert_eq!(implementation["modes"], serde_json::json!(["edit_image"]));
            assert_eq!(implementation["overlays"], serde_json::json!(["lora"]));
        }
        assert_eq!(
            implementations[2]["engagedRungs"],
            serde_json::json!([
                "resident",
                "staged_residency",
                "bounded_decode",
                "bounded_attention",
                "bounded_transformer_residency"
            ])
        );
        let raw = include_str!("../../../config/manifests/builtin.models.jsonc");
        let shipped: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(raw))
                .expect("builtin model manifest parses");
        let models = shipped["models"].as_array().expect("models array");
        assert!(
            declared_mlx_deferred_routes(models).contains("krea_2_raw"),
            "the typed manifest population, not a source-text mirror, must discover Raw"
        );
        let mut without_btr = manifest.clone();
        without_btr["mlx"]["memoryStrategyContract"]["implementations"]
            .as_array_mut()
            .expect("implementation rows")
            .retain(|row| row["rung"].as_str() != Some("bounded_transformer_residency"));
        assert!(
            !declared_mlx_deferred_routes(&[Value::Object(without_btr)]).contains("krea_2_raw"),
            "removing the exact BTR row must remove Raw from the derived population"
        );
        for implementation in implementations {
            assert_eq!(
                implementation["parameterRanges"]["decodeTileEdges"],
                serde_json::json!([2048, 512]),
                "Raw publishes its PiD and native decode domains without conflating their runtime selector"
            );
            assert_eq!(
                implementation["parameterRanges"]["decodeOverlaps"],
                serde_json::json!([256, 64])
            );
        }

        for tier in [
            MemoryRouteTier::Bf16,
            MemoryRouteTier::Q4,
            MemoryRouteTier::Q8,
        ] {
            for profile in [
                MemoryRouteLoadProfile::Plain,
                MemoryRouteLoadProfile::Lora,
                MemoryRouteLoadProfile::LoraPid,
                MemoryRouteLoadProfile::Pid,
            ] {
                let input = spec(tier, profile).with_resolved_route("krea_2_raw");
                let shaped = evaluate_declared_mlx_load_shape_with(
                    "krea_2_raw",
                    Some(tier.as_str()),
                    Some(MemoryRouteMode::TextToImage),
                    &manifest,
                    input,
                    |candidate| {
                        candidate.offload_policy == OffloadPolicy::Sequential
                            && candidate.load_shape == LoadShape::DeferredMaterialization
                            && candidate.load_shape_declaration_result
                                == LoadShapeDeclarationResult::Applied
                            && candidate.pid.is_some()
                                == matches!(
                                    profile,
                                    MemoryRouteLoadProfile::Pid | MemoryRouteLoadProfile::LoraPid
                                )
                            && !candidate.adapters.is_empty()
                                == matches!(
                                    profile,
                                    MemoryRouteLoadProfile::Lora | MemoryRouteLoadProfile::LoraPid
                                )
                    },
                );
                assert_eq!(
                    shaped.offload_policy,
                    OffloadPolicy::Resident,
                    "{tier:?} {profile:?}"
                );
                assert_eq!(shaped.load_shape, LoadShape::DeferredMaterialization);
                assert_eq!(
                    shaped.load_shape_declaration_result,
                    LoadShapeDeclarationResult::Applied
                );
                let sequential = shaped.with_offload_policy(OffloadPolicy::Sequential);
                assert_eq!(sequential.load_shape, LoadShape::DeferredMaterialization);
                assert_eq!(
                    sequential.load_shape_declaration_result,
                    LoadShapeDeclarationResult::Applied
                );
            }
        }

        let lokr = LoadSpec::new(WeightsSource::Dir("fixture".into()))
            .with_quant(Quant::Q4)
            .with_resolved_route("krea_2_raw")
            .with_adapters(vec![gen_core::AdapterSpec::new(
                "adapter.safetensors".into(),
                1.0,
                gen_core::AdapterKind::Lokr,
            )]);
        let lokr = evaluate_declared_mlx_load_shape_with(
            "krea_2_raw",
            Some("q4"),
            Some(MemoryRouteMode::TextToImage),
            &manifest,
            lokr,
            |_| true,
        );
        assert_eq!(
            lokr.load_shape_declaration_result,
            LoadShapeDeclarationResult::Applied,
            "LoRA and LoKr share the exact low-rank load profile"
        );

        let lokr_pid = LoadSpec::new(WeightsSource::Dir("fixture".into()))
            .with_quant(Quant::Q4)
            .with_resolved_route("krea_2_raw")
            .with_adapters(vec![gen_core::AdapterSpec::new(
                "adapter.safetensors".into(),
                1.0,
                gen_core::AdapterKind::Lokr,
            )])
            .with_pid(
                WeightsSource::File("pid.safetensors".into()),
                WeightsSource::Dir("gemma".into()),
            );
        let lokr_pid = evaluate_declared_mlx_load_shape_with(
            "krea_2_raw",
            Some("q4"),
            Some(MemoryRouteMode::TextToImage),
            &manifest,
            lokr_pid,
            |candidate| {
                !candidate.adapters.is_empty()
                    && candidate.pid.is_some()
                    && candidate.offload_policy == OffloadPolicy::Sequential
            },
        );
        assert_eq!(
            lokr_pid.load_shape_declaration_result,
            LoadShapeDeclarationResult::Applied,
            "the explicit low-rank+PiD profile preserves both provider-owned axes"
        );

        let composed = LoadSpec::new(WeightsSource::Dir("fixture".into()))
            .with_quant(Quant::Q4)
            .with_resolved_route("krea_2_raw")
            .with_text_encoder(WeightsSource::Dir("external-text-encoder".into()))
            .with_component("vae", WeightsSource::File("wan-vae.safetensors".into()));
        let composed = evaluate_declared_mlx_load_shape_with(
            "krea_2_raw",
            Some("q4"),
            Some(MemoryRouteMode::TextToImage),
            &manifest,
            composed,
            |candidate| {
                candidate.text_encoder.is_some()
                    && candidate.components.contains_key("vae")
                    && candidate.offload_policy == OffloadPolicy::Sequential
            },
        );
        assert_eq!(
            composed.load_shape_declaration_result,
            LoadShapeDeclarationResult::Applied,
            "compatible external TE and Wan decoder remain provider-owned Plain coordinates"
        );

        let provider_refused = evaluate_declared_mlx_load_shape_with(
            "krea_2_raw",
            Some("q4"),
            Some(MemoryRouteMode::TextToImage),
            &manifest,
            spec(MemoryRouteTier::Q4, MemoryRouteLoadProfile::Lora)
                .with_resolved_route("krea_2_raw"),
            |_| false,
        );
        assert_eq!(
            provider_refused.load_shape_declaration_result,
            LoadShapeDeclarationResult::Refused,
            "a low-rank-shaped route still fails closed when the provider identifies an unsupported dense diff patch"
        );

        let unknown_component = LoadSpec::new(WeightsSource::Dir("fixture".into()))
            .with_resolved_route("krea_2_raw")
            .with_component(
                "unknown_component",
                WeightsSource::File("unknown.safetensors".into()),
            );
        let unknown_component = evaluate_declared_mlx_load_shape_with(
            "krea_2_raw",
            Some("bf16"),
            Some(MemoryRouteMode::TextToImage),
            &manifest,
            unknown_component,
            |_| false,
        );
        assert_eq!(
            unknown_component.load_shape_declaration_result,
            LoadShapeDeclarationResult::Refused,
            "unknown components remain provider-owned refusals"
        );
    }

    #[test]
    fn krea_raw_declaration_refuses_crossed_coordinates_without_legacy_fallthrough() {
        let manifest = shipped_model("krea_2_raw");
        let plain = spec(MemoryRouteTier::Q4, MemoryRouteLoadProfile::Plain)
            .with_resolved_route("krea_2_raw");
        let apply = |provider: &str,
                     tier: Option<&str>,
                     mode: Option<MemoryRouteMode>,
                     manifest: &JsonObject<String, Value>,
                     input: LoadSpec| {
            evaluate_declared_mlx_load_shape_with(provider, tier, mode, manifest, input, |_| true)
        };

        for mode in MemoryRouteMode::ALL
            .into_iter()
            .filter(|mode| *mode != MemoryRouteMode::TextToImage)
        {
            assert_eq!(
                apply(
                    "krea_2_raw",
                    Some("q4"),
                    Some(mode),
                    &manifest,
                    plain.clone()
                )
                .load_shape_declaration_result,
                LoadShapeDeclarationResult::Refused,
                "mode {mode:?}"
            );
        }
        for profile in [
            MemoryRouteLoadProfile::SingleControl,
            MemoryRouteLoadProfile::MultiControl,
            MemoryRouteLoadProfile::IpAdapter,
            MemoryRouteLoadProfile::Identity,
        ] {
            assert_eq!(
                apply(
                    "krea_2_raw",
                    Some("q4"),
                    Some(MemoryRouteMode::TextToImage),
                    &manifest,
                    spec(MemoryRouteTier::Q4, profile).with_resolved_route("krea_2_raw"),
                )
                .load_shape_declaration_result,
                LoadShapeDeclarationResult::Refused,
                "profile {profile:?}"
            );
        }
        let low_rank_pid_control = spec(MemoryRouteTier::Q4, MemoryRouteLoadProfile::LoraPid)
            .with_control(WeightsSource::File("control.safetensors".into()))
            .with_resolved_route("krea_2_raw");
        assert_eq!(
            apply(
                "krea_2_raw",
                Some("q4"),
                Some(MemoryRouteMode::TextToImage),
                &manifest,
                low_rank_pid_control,
            )
            .load_shape_declaration_result,
            LoadShapeDeclarationResult::Refused,
            "the explicit low-rank+PiD profile must not admit a third component profile"
        );
        for tier in [None, Some("nvfp4"), Some("unknown")] {
            assert_eq!(
                apply(
                    "krea_2_raw",
                    tier,
                    Some(MemoryRouteMode::TextToImage),
                    &manifest,
                    plain.clone(),
                )
                .load_shape_declaration_result,
                LoadShapeDeclarationResult::Refused,
                "resolved tier {tier:?}"
            );
        }
        let tier_mismatch = evaluate_declared_mlx_load_shape_with(
            "krea_2_raw",
            Some("q4"),
            Some(MemoryRouteMode::TextToImage),
            &manifest,
            spec(MemoryRouteTier::Q8, MemoryRouteLoadProfile::Plain)
                .with_resolved_route("krea_2_raw"),
            |candidate| candidate.quantize == Some(Quant::Q4),
        );
        assert_eq!(
            tier_mismatch.load_shape_declaration_result,
            LoadShapeDeclarationResult::Refused,
            "the explicit resolved tier is not rewritten from or into LoadSpec.quantize; the provider owns physical tier validation"
        );
        assert_eq!(
            apply(
                "krea_2_turbo",
                Some("q4"),
                Some(MemoryRouteMode::TextToImage),
                &manifest,
                plain.clone(),
            )
            .load_shape_declaration_result,
            LoadShapeDeclarationResult::Refused
        );
        assert_eq!(
            apply(
                "krea_2_raw",
                Some("q4"),
                Some(MemoryRouteMode::TextToImage),
                &manifest,
                plain.clone().with_resolved_route("krea_2_turbo"),
            )
            .load_shape_declaration_result,
            LoadShapeDeclarationResult::Refused,
            "the Raw provider cannot consume a crossed catalog route"
        );
        assert_eq!(
            apply(
                "krea_2_raw",
                Some("q4"),
                Some(MemoryRouteMode::TextToImage),
                &manifest,
                LoadSpec::new(WeightsSource::File("raw.safetensors".into()))
                    .with_resolved_route("krea_2_raw"),
            )
            .load_shape_declaration_result,
            LoadShapeDeclarationResult::Refused
        );

        let mut wrong_provider = manifest.clone();
        wrong_provider["mlx"]["memoryStrategyContract"]["provider"] =
            Value::String("krea_2_turbo".to_owned());
        assert_eq!(
            apply(
                "krea_2_raw",
                Some("q4"),
                Some(MemoryRouteMode::TextToImage),
                &wrong_provider,
                plain.clone(),
            )
            .load_shape_declaration_result,
            LoadShapeDeclarationResult::Refused
        );

        let mut malformed = manifest.clone();
        malformed["mlx"]["memoryStrategyContract"]["implementations"][2]["engagedRungs"] =
            Value::String("staged_residency".to_owned());
        assert_eq!(
            apply(
                "krea_2_raw",
                Some("q4"),
                Some(MemoryRouteMode::TextToImage),
                &malformed,
                plain.clone(),
            )
            .load_shape_declaration_result,
            LoadShapeDeclarationResult::Refused
        );

        let mut missing = manifest;
        missing["mlx"]
            .as_object_mut()
            .expect("MLX object")
            .remove("memoryStrategyContract");
        let undeclared = apply(
            "krea_2_raw",
            Some("q4"),
            Some(MemoryRouteMode::TextToImage),
            &missing,
            plain,
        );
        assert_eq!(
            undeclared.load_shape_declaration_result,
            LoadShapeDeclarationResult::NotEvaluated
        );
        let late = apply_registered_load_shape(
            MemoryRouteBackend::Mlx,
            "krea_2_raw",
            MemoryRouteMode::TextToImage,
            undeclared,
            true,
        );
        assert_eq!(late.load_shape, LoadShape::EagerMaterialization);
        assert_eq!(
            late.load_shape_declaration_result,
            LoadShapeDeclarationResult::NotEvaluated,
            "a missing Raw declaration cannot be recreated by the legacy shaper"
        );
    }

    #[test]
    fn krea_turbo_and_edit_request_contexts_are_declaration_owned_and_fail_closed() {
        let manifest = shipped_model("krea_2_turbo");
        let evaluate = |provider: &str,
                        profile: MemoryRouteLoadProfile,
                        context: MemoryRouteRequestContext| {
            evaluate_declared_mlx_load_shape_for_request_with(
                provider,
                Some("q4"),
                Some(context.mode),
                &manifest,
                spec(MemoryRouteTier::Q4, profile).with_resolved_route("krea_2_turbo"),
                context,
                |candidate| {
                    candidate.offload_policy == OffloadPolicy::Sequential
                        && candidate.load_shape == LoadShape::DeferredMaterialization
                },
            )
        };
        for profile in PLAIN_LORA_PID {
            let applied = evaluate(
                "krea_2_turbo",
                *profile,
                MemoryRouteRequestContext {
                    mode: MemoryRouteMode::TextToImage,
                    reference_count: 0,
                    use_pid: matches!(
                        *profile,
                        MemoryRouteLoadProfile::Pid | MemoryRouteLoadProfile::LoraPid
                    ),
                    has_phases: false,
                },
            );
            assert_eq!(
                applied.load_shape_declaration_result,
                LoadShapeDeclarationResult::Applied,
                "Turbo {profile:?}"
            );
        }
        for (profile, crossed_use_pid) in [
            (MemoryRouteLoadProfile::Plain, true),
            (MemoryRouteLoadProfile::Pid, false),
            (MemoryRouteLoadProfile::Lora, true),
            (MemoryRouteLoadProfile::LoraPid, false),
        ] {
            let refused = evaluate(
                "krea_2_turbo",
                profile,
                MemoryRouteRequestContext {
                    mode: MemoryRouteMode::TextToImage,
                    reference_count: 0,
                    use_pid: crossed_use_pid,
                    has_phases: false,
                },
            );
            assert_eq!(
                refused.load_shape_declaration_result,
                LoadShapeDeclarationResult::Refused,
                "Turbo {profile:?} must bind request PiD identity to the loaded PiD components"
            );
            assert_eq!(refused.load_shape, LoadShape::EagerMaterialization);
        }
        let reference = evaluate(
            "krea_2_turbo",
            MemoryRouteLoadProfile::LoraPid,
            MemoryRouteRequestContext {
                mode: MemoryRouteMode::TextToImage,
                reference_count: 1,
                use_pid: true,
                has_phases: false,
            },
        );
        assert_eq!(
            reference.load_shape_declaration_result,
            LoadShapeDeclarationResult::Applied
        );
        for reference_count in [1, 2] {
            let edit = evaluate(
                "krea_2_turbo_edit",
                MemoryRouteLoadProfile::LoraPid,
                MemoryRouteRequestContext {
                    mode: MemoryRouteMode::EditImage,
                    reference_count,
                    use_pid: true,
                    has_phases: false,
                },
            );
            assert_eq!(
                edit.load_shape_declaration_result,
                LoadShapeDeclarationResult::Applied,
                "Turbo edit references={reference_count}"
            );
        }
        for crossed in [
            MemoryRouteRequestContext {
                mode: MemoryRouteMode::TextToImage,
                reference_count: 2,
                use_pid: false,
                has_phases: false,
            },
            MemoryRouteRequestContext {
                mode: MemoryRouteMode::TextToImage,
                reference_count: 0,
                use_pid: false,
                has_phases: true,
            },
            MemoryRouteRequestContext {
                mode: MemoryRouteMode::EditImage,
                reference_count: 0,
                use_pid: false,
                has_phases: false,
            },
        ] {
            assert_eq!(
                evaluate("krea_2_turbo", MemoryRouteLoadProfile::Plain, crossed)
                    .load_shape_declaration_result,
                LoadShapeDeclarationResult::Refused,
                "crossed context {crossed:?}"
            );
        }

        let plain_context = MemoryRouteRequestContext {
            mode: MemoryRouteMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            has_phases: false,
        };
        let route_unbound = evaluate_declared_mlx_load_shape_for_request_with(
            "krea_2_turbo",
            Some("q4"),
            Some(MemoryRouteMode::TextToImage),
            &manifest,
            spec(MemoryRouteTier::Q4, MemoryRouteLoadProfile::Plain),
            plain_context,
            |_| true,
        );
        assert_eq!(
            route_unbound.load_shape_declaration_result,
            LoadShapeDeclarationResult::Refused,
            "an exact request-context declaration cannot authorize a route-unbound load"
        );
        let evaluate_mutated = |mutated: &JsonObject<String, Value>| {
            evaluate_declared_mlx_load_shape_for_request_with(
                "krea_2_turbo",
                Some("q4"),
                Some(MemoryRouteMode::TextToImage),
                mutated,
                spec(MemoryRouteTier::Q4, MemoryRouteLoadProfile::Plain)
                    .with_resolved_route("krea_2_turbo"),
                plain_context,
                |_| true,
            )
        };
        let mut crossed_provider_mode = manifest.clone();
        crossed_provider_mode["mlx"]["memoryStrategyContract"]["implementations"][2]
            ["requestContexts"][0]["providerMode"] = Value::String("image_to_image".to_owned());
        assert_eq!(
            evaluate_mutated(&crossed_provider_mode).load_shape_declaration_result,
            LoadShapeDeclarationResult::Refused,
            "the public coordinate cannot authorize a crossed provider-facing mode"
        );

        let mut ambiguous = manifest.clone();
        let contexts = ambiguous["mlx"]["memoryStrategyContract"]["implementations"][2]
            ["requestContexts"]
            .as_array_mut()
            .expect("Turbo BTR request contexts");
        let duplicate = contexts[0].clone();
        contexts.push(duplicate);
        assert_eq!(
            evaluate_mutated(&ambiguous).load_shape_declaration_result,
            LoadShapeDeclarationResult::Refused,
            "duplicate matching request identities fail closed"
        );

        let mut missing_matching_contexts = manifest.clone();
        missing_matching_contexts["mlx"]["memoryStrategyContract"]["implementations"][2]
            .as_object_mut()
            .expect("Turbo BTR implementation")
            .remove("requestContexts");
        assert_eq!(
            evaluate_mutated(&missing_matching_contexts).load_shape_declaration_result,
            LoadShapeDeclarationResult::Refused,
            "a declaration-owned BTR row cannot become context-free"
        );

        let mut missing_all_contexts = manifest.clone();
        for implementation in missing_all_contexts["mlx"]["memoryStrategyContract"]
            ["implementations"]
            .as_array_mut()
            .expect("Turbo implementations")
        {
            implementation
                .as_object_mut()
                .expect("Turbo implementation")
                .remove("requestContexts");
        }
        assert_eq!(
            evaluate_mutated(&missing_all_contexts).load_shape_declaration_result,
            LoadShapeDeclarationResult::Refused,
            "removing every request context cannot reactivate legacy declaration semantics"
        );
        assert!(manifest["mlx"].get("supportsSequentialOffload").is_none());
    }

    #[test]
    fn krea_raw_multiphase_and_edit_rows_bind_exact_request_identity() {
        let manifest = shipped_model("krea_2_raw");
        let evaluate = |provider: &str,
                        profile: MemoryRouteLoadProfile,
                        context: MemoryRouteRequestContext| {
            evaluate_declared_mlx_load_shape_for_request_with(
                provider,
                Some("q8"),
                Some(context.mode),
                &manifest,
                spec(MemoryRouteTier::Q8, profile).with_resolved_route("krea_2_raw"),
                context,
                |_| true,
            )
        };
        let multiphase = evaluate(
            "krea_2_raw",
            MemoryRouteLoadProfile::Lora,
            MemoryRouteRequestContext {
                mode: MemoryRouteMode::TextToImage,
                reference_count: 0,
                use_pid: false,
                has_phases: true,
            },
        );
        assert_eq!(
            multiphase.load_shape_declaration_result,
            LoadShapeDeclarationResult::Applied
        );
        for crossed in [
            MemoryRouteRequestContext {
                use_pid: true,
                ..MemoryRouteRequestContext {
                    mode: MemoryRouteMode::TextToImage,
                    reference_count: 0,
                    use_pid: false,
                    has_phases: true,
                }
            },
            MemoryRouteRequestContext {
                mode: MemoryRouteMode::TextToImage,
                reference_count: 1,
                use_pid: false,
                has_phases: true,
            },
        ] {
            assert_eq!(
                evaluate("krea_2_raw", MemoryRouteLoadProfile::Lora, crossed)
                    .load_shape_declaration_result,
                LoadShapeDeclarationResult::Refused
            );
        }
        for reference_count in [1, 2] {
            assert_eq!(
                evaluate(
                    "krea_2_edit",
                    MemoryRouteLoadProfile::Lora,
                    MemoryRouteRequestContext {
                        mode: MemoryRouteMode::EditImage,
                        reference_count,
                        use_pid: false,
                        has_phases: false,
                    },
                )
                .load_shape_declaration_result,
                LoadShapeDeclarationResult::Applied
            );
        }
        assert_eq!(
            evaluate(
                "krea_2_edit",
                MemoryRouteLoadProfile::Lora,
                MemoryRouteRequestContext {
                    mode: MemoryRouteMode::EditImage,
                    reference_count: 3,
                    use_pid: false,
                    has_phases: false,
                },
            )
            .load_shape_declaration_result,
            LoadShapeDeclarationResult::Refused
        );
    }

    #[test]
    fn pulid_mlx_declaration_is_exhaustive_identity_only_and_fail_closed() {
        let manifest = shipped_model("pulid_flux_dev");
        let implementations = manifest["mlx"]["memoryStrategyContract"]["implementations"]
            .as_array()
            .expect("PuLID MLX implementations");
        let rungs = implementations
            .iter()
            .map(|implementation| implementation["rung"].as_str().expect("PuLID rung"))
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            rungs,
            [
                "resident",
                "staged_residency",
                "bounded_decode",
                "bounded_attention",
                "bounded_transformer_residency",
            ]
            .into_iter()
            .collect(),
            "the exhaustive declaration must publish every implemented PuLID strategy"
        );
        assert_eq!(implementations.len(), 5, "no duplicate PuLID rung rows");
        let btr_index = implementations
            .iter()
            .position(|implementation| {
                implementation["rung"].as_str() == Some("bounded_transformer_residency")
            })
            .expect("PuLID BTR implementation");
        for implementation in implementations {
            assert_eq!(
                implementation["tiers"],
                serde_json::json!(["bf16", "q4", "q8"])
            );
            assert_eq!(
                implementation["modes"],
                serde_json::json!(["character_image"])
            );
            assert_eq!(implementation["overlays"], serde_json::json!(["identity"]));
            assert_eq!(
                implementation["loadProfiles"],
                serde_json::json!(["identity"])
            );
            assert_eq!(implementation["sourceKinds"], serde_json::json!(["dir"]));
            assert_eq!(implementation["providerOverlay"], "identity");
            assert_eq!(
                implementation["requestContexts"],
                serde_json::json!([{
                    "mode": "character_image",
                    "providerMode": "image_to_image",
                    "referenceCounts": [1],
                    "pid": [false],
                    "hasPhases": false,
                }]),
                "every PuLID rung is bound to the same exact provider request"
            );
        }
        let context = MemoryRouteRequestContext {
            mode: MemoryRouteMode::CharacterImage,
            reference_count: 1,
            use_pid: false,
            has_phases: false,
        };
        let provider_accepts = |candidate: &LoadSpec| {
            candidate.offload_policy == OffloadPolicy::Sequential
                && candidate.load_shape == LoadShape::DeferredMaterialization
                && matches!(candidate.weights, WeightsSource::Dir(_))
                && candidate.resolved_route.as_deref() == Some("pulid_flux_dev")
                && candidate.identity.as_ref().is_some_and(|identity| {
                    identity.encoder.is_some()
                        && identity.eva.is_some()
                        && identity.face_dir.is_some()
                })
                && candidate.adapters.is_empty()
                && candidate.pid.is_none()
                && candidate.control.is_none()
                && candidate.extra_controls.is_empty()
                && candidate.ip_adapter.is_none()
                && candidate.text_encoder.is_none()
                && candidate.components.is_empty()
        };
        let evaluate = |tier: MemoryRouteTier,
                        candidate: LoadSpec,
                        request_context: MemoryRouteRequestContext,
                        candidate_manifest: &JsonObject<String, Value>| {
            evaluate_declared_mlx_load_shape_for_request_with(
                "pulid_flux",
                Some(tier.as_str()),
                Some(MemoryRouteMode::CharacterImage),
                candidate_manifest,
                candidate,
                request_context,
                provider_accepts,
            )
        };

        for tier in [
            MemoryRouteTier::Bf16,
            MemoryRouteTier::Q4,
            MemoryRouteTier::Q8,
        ] {
            let applied = evaluate(
                tier,
                spec(tier, MemoryRouteLoadProfile::Identity).with_resolved_route("pulid_flux_dev"),
                context,
                &manifest,
            );
            assert_eq!(
                applied.load_shape_declaration_result,
                LoadShapeDeclarationResult::Applied,
                "PuLID {tier:?}"
            );
            assert_eq!(applied.load_shape, LoadShape::DeferredMaterialization);
            assert_eq!(applied.offload_policy, OffloadPolicy::Resident);
        }

        let evaluate_refused = |candidate: LoadSpec, request_context: MemoryRouteRequestContext| {
            evaluate(MemoryRouteTier::Q4, candidate, request_context, &manifest)
        };
        let q4_identity = || {
            spec(MemoryRouteTier::Q4, MemoryRouteLoadProfile::Identity)
                .with_resolved_route("pulid_flux_dev")
        };
        for (label, candidate, request_context) in [
            (
                "missing identity",
                spec(MemoryRouteTier::Q4, MemoryRouteLoadProfile::Plain)
                    .with_resolved_route("pulid_flux_dev"),
                context,
            ),
            (
                "route missing",
                spec(MemoryRouteTier::Q4, MemoryRouteLoadProfile::Identity),
                context,
            ),
            (
                "route crossed",
                spec(MemoryRouteTier::Q4, MemoryRouteLoadProfile::Identity)
                    .with_resolved_route("flux_dev"),
                context,
            ),
            (
                "reference missing",
                q4_identity(),
                MemoryRouteRequestContext {
                    reference_count: 0,
                    ..context
                },
            ),
            (
                "reference duplicated",
                q4_identity(),
                MemoryRouteRequestContext {
                    reference_count: 2,
                    ..context
                },
            ),
            (
                "PiD crossed",
                q4_identity(),
                MemoryRouteRequestContext {
                    use_pid: true,
                    ..context
                },
            ),
            (
                "phases crossed",
                q4_identity(),
                MemoryRouteRequestContext {
                    has_phases: true,
                    ..context
                },
            ),
            (
                "mode crossed",
                q4_identity(),
                MemoryRouteRequestContext {
                    mode: MemoryRouteMode::TextToImage,
                    ..context
                },
            ),
        ] {
            let refused = evaluate_refused(candidate, request_context);
            assert_eq!(
                refused.load_shape_declaration_result,
                LoadShapeDeclarationResult::Refused,
                "{label}"
            );
            assert_eq!(refused.load_shape, LoadShape::EagerMaterialization);
        }

        let file = {
            let mut candidate = q4_identity();
            candidate.weights = WeightsSource::File("pulid.safetensors".into());
            candidate
        };
        let external_te =
            q4_identity().with_text_encoder(WeightsSource::Dir("external-text-encoder".into()));
        let mut unknown_component = q4_identity();
        unknown_component.components.insert(
            "unexpected".to_owned(),
            WeightsSource::File("unexpected.safetensors".into()),
        );
        let mut incomplete_identity = q4_identity();
        incomplete_identity
            .identity
            .as_mut()
            .expect("identity fixture")
            .eva = None;
        for (label, candidate) in [
            ("File source", file),
            ("external text encoder", external_te),
            ("unknown component", unknown_component),
            ("incomplete identity", incomplete_identity),
        ] {
            let refused = evaluate_refused(candidate, context);
            assert_eq!(
                refused.load_shape_declaration_result,
                LoadShapeDeclarationResult::Refused,
                "{label}"
            );
        }

        let mut crossed_overlay = manifest.clone();
        crossed_overlay["mlx"]["memoryStrategyContract"]["implementations"][btr_index]
            ["providerOverlay"] = Value::String("none".to_owned());
        assert_eq!(
            evaluate(
                MemoryRouteTier::Q4,
                q4_identity(),
                context,
                &crossed_overlay,
            )
            .load_shape_declaration_result,
            LoadShapeDeclarationResult::Refused,
            "identity profile cannot consume a none-overlay provider coordinate"
        );
        let mut crossed_provider_mode = manifest.clone();
        crossed_provider_mode["mlx"]["memoryStrategyContract"]["implementations"][btr_index]
            ["requestContexts"][0]["providerMode"] = Value::String("character_image".to_owned());
        assert_eq!(
            evaluate(
                MemoryRouteTier::Q4,
                q4_identity(),
                context,
                &crossed_provider_mode,
            )
            .load_shape_declaration_result,
            LoadShapeDeclarationResult::Refused,
            "public CharacterImage must map to provider ImageToImage"
        );
        let mut missing_context = manifest.clone();
        missing_context["mlx"]["memoryStrategyContract"]["implementations"][btr_index]
            .as_object_mut()
            .expect("PuLID BTR implementation")
            .remove("requestContexts");
        assert_eq!(
            evaluate(
                MemoryRouteTier::Q4,
                q4_identity(),
                context,
                &missing_context,
            )
            .load_shape_declaration_result,
            LoadShapeDeclarationResult::Refused,
            "a missing request identity cannot become a context-free declaration"
        );

        let mut no_btr = manifest.clone();
        no_btr["mlx"]["memoryStrategyContract"]["implementations"]
            .as_array_mut()
            .expect("PuLID implementations")
            .retain(|implementation| {
                implementation.get("rung").and_then(Value::as_str)
                    != Some("bounded_transformer_residency")
            });
        let undeclared = evaluate(MemoryRouteTier::Q4, q4_identity(), context, &no_btr);
        assert_eq!(
            undeclared.load_shape_declaration_result,
            LoadShapeDeclarationResult::NotEvaluated
        );
        assert_eq!(
            apply_registered_load_shape(
                MemoryRouteBackend::Mlx,
                "pulid_flux",
                MemoryRouteMode::CharacterImage,
                undeclared,
                true,
            )
            .load_shape,
            LoadShape::EagerMaterialization,
            "a missing PuLID declaration cannot fall through to legacy shaping"
        );
        assert!(manifest["mlx"].get("supportsSequentialOffload").is_none());
        assert!(manifest["candle"].get("memoryStrategyContract").is_none());
    }

    #[test]
    fn dense_diff_refusal_emits_typed_eager_warning_only() {
        let temp = tempfile::tempdir().unwrap();
        let adapter = temp.path().join("dense-diff.safetensors");
        let header = br#"{"transformer_blocks.0.attn.to_q.diff":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&0_f32.to_le_bytes());
        std::fs::write(&adapter, bytes).unwrap();
        let refused = LoadSpec::new(WeightsSource::Dir("fixture".into()))
            .with_adapters(vec![gen_core::AdapterSpec::new(
                adapter,
                1.0,
                gen_core::AdapterKind::Lora,
            )])
            .with_refused_load_shape_declaration();
        assert_eq!(
            mlx_load_shape_declaration_warning(&refused),
            Some(MlxLoadShapeDeclarationWarning::DenseDiffRequiresEager)
        );
        assert_eq!(
            mlx_load_shape_declaration_warning(
                &refused.clone().with_applied_load_shape_declaration()
            ),
            None
        );
        assert_eq!(
            mlx_load_shape_declaration_warning(&LoadSpec::new(WeightsSource::Dir(
                "fixture".into()
            ))),
            None
        );
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
        assert_eq!(
            apply_registered_load_shape(
                MemoryRouteBackend::Mlx,
                "z_image_turbo",
                MemoryRouteMode::TextToImage,
                spec(MemoryRouteTier::Q4, MemoryRouteLoadProfile::LoraPid),
                true,
            )
            .load_shape,
            LoadShape::EagerMaterialization,
            "the Raw-only combined low-rank+PiD profile must not expand legacy Turbo shaping",
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

    #[test]
    fn candle_request_strategy_declaration_is_exact_and_does_not_change_load_shape() {
        let manifest = request_strategy_declaration();
        let plain =
            LoadSpec::new(WeightsSource::Dir("q4".into())).with_resolved_route("krea_2_raw");
        let context = MemoryRouteRequestContext {
            mode: MemoryRouteMode::TextToImage,
            reference_count: 1,
            use_pid: false,
            has_phases: false,
        };
        let applied = declared_candle_request_strategy_contract_with(
            "krea_2_raw",
            Some("q4"),
            &manifest,
            &plain,
            context,
            |_| Some(staged_contract("krea_2_raw")),
        );
        let DeclaredCandleStrategyContract::Applied {
            provider_overlay,
            provider_mode,
            ..
        } = applied
        else {
            panic!("exact Raw declaration must apply");
        };
        assert_eq!(provider_overlay, None);
        assert_eq!(provider_mode, "image_to_image");
        assert_eq!(plain.load_shape, LoadShape::EagerMaterialization);
        assert_eq!(
            plain.load_shape_declaration_result,
            LoadShapeDeclarationResult::NotEvaluated
        );

        let lora_pid = plain
            .clone()
            .with_adapters(vec![gen_core::AdapterSpec::new(
                "adapter.safetensors".into(),
                1.0,
                gen_core::AdapterKind::Lora,
            )])
            .with_pid(
                WeightsSource::File("pid.safetensors".into()),
                WeightsSource::File("gemma".into()),
            );
        assert!(matches!(
            declared_candle_request_strategy_contract_with(
                "krea_2_raw",
                Some("q8"),
                &manifest,
                &lora_pid,
                MemoryRouteRequestContext {
                    use_pid: true,
                    ..context
                },
                |_| Some(staged_contract("krea_2_raw")),
            ),
            DeclaredCandleStrategyContract::Applied { .. }
        ));

        let multiphase = MemoryRouteRequestContext {
            reference_count: 0,
            use_pid: false,
            has_phases: true,
            ..context
        };
        let phases = declared_candle_request_strategy_contract_with(
            "krea_2_raw",
            Some("bf16"),
            &manifest,
            &plain,
            multiphase,
            |_| Some(staged_contract("krea_2_raw")),
        );
        let DeclaredCandleStrategyContract::Applied { provider_mode, .. } = phases else {
            panic!("actual GenerationRequest phases must apply");
        };
        assert_eq!(provider_mode, "text_to_image");

        // Hires.fix is a worker multipass, not GenerationRequest::phases. Its declaration context
        // therefore stays the ordinary reference-bearing Raw route; if both axes were collapsed,
        // this exact coordinate would incorrectly bind the multi-phase provider witness.
        let hires_only = declared_candle_request_strategy_contract_with(
            "krea_2_raw",
            Some("q4"),
            &manifest,
            &plain,
            context,
            |_| Some(staged_contract("krea_2_raw")),
        );
        assert!(matches!(
            hires_only,
            DeclaredCandleStrategyContract::Applied {
                provider_mode,
                ..
            } if provider_mode == "image_to_image"
        ));
        let both_present = MemoryRouteRequestContext {
            has_phases: true,
            ..context
        };
        assert!(matches!(
            declared_candle_request_strategy_contract_with(
                "krea_2_raw",
                Some("q4"),
                &manifest,
                &plain,
                both_present,
                |_| Some(staged_contract("krea_2_raw")),
            ),
            DeclaredCandleStrategyContract::Refused
        ));

        for refused in [
            declared_candle_request_strategy_contract_with(
                "krea_2_raw",
                Some("nvfp4"),
                &manifest,
                &plain,
                context,
                |_| Some(staged_contract("krea_2_raw")),
            ),
            declared_candle_request_strategy_contract_with(
                "krea_2_raw",
                Some("q4"),
                &manifest,
                &plain,
                MemoryRouteRequestContext {
                    mode: MemoryRouteMode::EditImage,
                    ..context
                },
                |_| Some(staged_contract("krea_2_raw")),
            ),
            declared_candle_request_strategy_contract_with(
                "krea_2_raw",
                Some("q4"),
                &manifest,
                &plain,
                MemoryRouteRequestContext {
                    reference_count: 2,
                    ..context
                },
                |_| Some(staged_contract("krea_2_raw")),
            ),
            declared_candle_request_strategy_contract_with(
                "krea_2_raw",
                Some("q4"),
                &manifest,
                &plain,
                MemoryRouteRequestContext {
                    use_pid: true,
                    has_phases: true,
                    reference_count: 0,
                    ..context
                },
                |_| Some(staged_contract("krea_2_raw")),
            ),
            declared_candle_request_strategy_contract_with(
                "krea_2_raw",
                Some("q4"),
                &manifest,
                &LoadSpec::new(WeightsSource::File("raw.safetensors".into()))
                    .with_resolved_route("krea_2_raw"),
                context,
                |_| Some(staged_contract("krea_2_raw")),
            ),
            declared_candle_request_strategy_contract_with(
                "krea_2_edit",
                Some("q4"),
                &manifest,
                &plain,
                context,
                |_| Some(staged_contract("krea_2_edit")),
            ),
        ] {
            assert!(matches!(refused, DeclaredCandleStrategyContract::Refused));
        }

        let mut missing_provider_mode = manifest.clone();
        missing_provider_mode["candle"]["memoryStrategyContract"]["implementations"][0]
            ["requestContexts"][1]
            .as_object_mut()
            .expect("reference context")
            .remove("providerMode");
        assert!(matches!(
            declared_candle_request_strategy_contract_with(
                "krea_2_raw",
                Some("q4"),
                &missing_provider_mode,
                &plain,
                context,
                |_| Some(staged_contract("krea_2_raw")),
            ),
            DeclaredCandleStrategyContract::Refused
        ));

        let mut duplicate_context = manifest.clone();
        let repeated = duplicate_context["candle"]["memoryStrategyContract"]["implementations"][0]
            ["requestContexts"][1]
            .clone();
        duplicate_context["candle"]["memoryStrategyContract"]["implementations"][0]
            ["requestContexts"]
            .as_array_mut()
            .expect("request contexts")
            .push(repeated);
        assert!(matches!(
            declared_candle_request_strategy_contract_with(
                "krea_2_raw",
                Some("q4"),
                &duplicate_context,
                &plain,
                context,
                |_| Some(staged_contract("krea_2_raw")),
            ),
            DeclaredCandleStrategyContract::Refused
        ));
    }

    #[test]
    fn shipped_candle_request_strategy_population_is_manifest_derived_and_complete() {
        let raw = include_str!("../../../config/manifests/builtin.models.jsonc");
        let manifest: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(raw))
                .expect("builtin manifest parses");
        let witnesses = declared_candle_request_strategy_route_witnesses(
            manifest["models"].as_array().expect("models array"),
        )
        .expect("request-strategy declarations are typed");
        let providers = witnesses
            .iter()
            .map(|witness| witness.provider.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            providers,
            ["krea_2_edit", "krea_2_raw", "krea_2_turbo_edit"].into()
        );
        assert_eq!(witnesses.len(), 24);
        assert!(witnesses.iter().all(|witness| {
            matches!(
                witness.tier,
                MemoryRouteTier::Bf16 | MemoryRouteTier::Q4 | MemoryRouteTier::Q8
            ) && matches!(
                witness.overlay,
                MemoryRouteOverlay::None | MemoryRouteOverlay::Lora
            ) && matches!(
                witness.load_profile,
                MemoryRouteLoadProfile::Plain
                    | MemoryRouteLoadProfile::Lora
                    | MemoryRouteLoadProfile::LoraPid
                    | MemoryRouteLoadProfile::Pid
            )
        }));
    }
}
