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
    /// The publisher FP8 rowwise package (sc-26241, FLUX 3 Image Klein), named by identity so a
    /// FLUX 3 FP8R load is never read back as another tier. Deliberately NOT in [`Self::ALL`]: no
    /// deferred route declares it, and `ALL` is the enumeration route witnesses are minted from,
    /// so listing it there would advertise an FP8R witness for every tier-agnostic declared route.
    Fp8Rowwise,
}

impl MemoryRouteTier {
    pub const ALL: [Self; 4] = [Self::Bf16, Self::Q4, Self::Q8, Self::Nvfp4];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Q4 => "q4",
            Self::Q8 => "q8",
            Self::Nvfp4 => "nvfp4",
            Self::Fp8Rowwise => sceneworks_core::image_request::FP8R_QUANT_TIER,
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
            Some(Quant::Fp8Rowwise) => Some(Self::Fp8Rowwise),
        }
    }

    pub fn from_resolved_tier(tier: &str) -> Option<Self> {
        match tier {
            "bf16" => Some(Self::Bf16),
            "q4" => Some(Self::Q4),
            "q8" => Some(Self::Q8),
            "nvfp4" => Some(Self::Nvfp4),
            sceneworks_core::image_request::FP8R_QUANT_TIER => Some(Self::Fp8Rowwise),
            _ => None,
        }
    }

    fn from_str(tier: &str) -> Option<Self> {
        Self::from_resolved_tier(tier)
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

    fn from_manifest(mode: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|candidate| candidate.as_str() == mode)
    }

    /// Preserve the public FLUX.2 `reference` compatibility spelling while keeping the
    /// normalization route-local. Other providers must opt into their own public mode aliases
    /// instead of silently inheriting this image-to-image coordinate.
    pub fn from_mlx_request(provider: &str, mode: &str) -> Option<Self> {
        if matches!(provider, "flux2_klein_9b" | "flux2_dev") && mode == "reference" {
            Some(Self::ImageToImage)
        } else {
            Self::from_request(mode)
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

    fn from_str(overlay: &str) -> Option<Self> {
        match overlay {
            "none" => Some(Self::None),
            "lora" => Some(Self::Lora),
            "control" => Some(Self::Control),
            "identity" => Some(Self::Identity),
            _ => None,
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
    LoraSingleControl,
    LoraSingleControlPid,
    LoraIpAdapter,
    IpAdapterPid,
    LoraIpAdapterPid,
    SingleControl,
    SingleControlPid,
    MultiControl,
    IpAdapter,
    Pid,
    Identity,
}

impl MemoryRouteLoadProfile {
    pub const ALL: [Self; 14] = [
        Self::Plain,
        Self::Lora,
        Self::LoraPid,
        Self::LoraSingleControl,
        Self::LoraSingleControlPid,
        Self::LoraIpAdapter,
        Self::IpAdapterPid,
        Self::LoraIpAdapterPid,
        Self::SingleControl,
        Self::SingleControlPid,
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
            Self::LoraSingleControl => "lora_single_control",
            Self::LoraSingleControlPid => "lora_single_control_pid",
            Self::LoraIpAdapter => "lora_ip_adapter",
            Self::IpAdapterPid => "ip_adapter_pid",
            Self::LoraIpAdapterPid => "lora_ip_adapter_pid",
            Self::SingleControl => "single_control",
            Self::SingleControlPid => "single_control_pid",
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
            Self::SingleControl
            | Self::SingleControlPid
            | Self::LoraSingleControl
            | Self::LoraSingleControlPid
            | Self::MultiControl => MemoryRouteOverlay::Control,
            Self::IpAdapter
            | Self::LoraIpAdapter
            | Self::IpAdapterPid
            | Self::LoraIpAdapterPid
            | Self::Identity => MemoryRouteOverlay::Identity,
        }
    }

    fn from_str(profile: &str) -> Option<Self> {
        match profile {
            "plain" => Some(Self::Plain),
            "lora" => Some(Self::Lora),
            "lora_pid" => Some(Self::LoraPid),
            "lora_single_control" => Some(Self::LoraSingleControl),
            "lora_single_control_pid" => Some(Self::LoraSingleControlPid),
            "lora_ip_adapter" => Some(Self::LoraIpAdapter),
            "ip_adapter_pid" => Some(Self::IpAdapterPid),
            "lora_ip_adapter_pid" => Some(Self::LoraIpAdapterPid),
            "single_control" => Some(Self::SingleControl),
            "single_control_pid" => Some(Self::SingleControlPid),
            "multi_control" => Some(Self::MultiControl),
            "ip_adapter" => Some(Self::IpAdapter),
            "pid" => Some(Self::Pid),
            "identity" => Some(Self::Identity),
            _ => None,
        }
    }

    fn from_spec(spec: &LoadSpec) -> Option<Self> {
        let has_lora = !spec.adapters.is_empty();
        let has_pid = spec.pid.is_some();
        let has_control = spec.control.is_some();
        let has_ip_adapter = spec.ip_adapter.is_some();
        let has_identity = spec.identity.is_some();
        if !spec.extra_controls.is_empty() {
            return (!has_lora && !has_pid && !has_ip_adapter && !has_identity)
                .then_some(Self::MultiControl);
        }
        if has_identity {
            return (!has_lora && !has_pid && !has_control && !has_ip_adapter)
                .then_some(Self::Identity);
        }
        if has_control {
            return match (has_lora, has_pid, has_ip_adapter) {
                (false, false, false) => Some(Self::SingleControl),
                (true, false, false) => Some(Self::LoraSingleControl),
                (false, true, false) => Some(Self::SingleControlPid),
                (true, true, false) => Some(Self::LoraSingleControlPid),
                _ => None,
            };
        }
        if has_ip_adapter {
            return match (has_lora, has_pid) {
                (false, false) => Some(Self::IpAdapter),
                (true, false) => Some(Self::LoraIpAdapter),
                (false, true) => Some(Self::IpAdapterPid),
                (true, true) => Some(Self::LoraIpAdapterPid),
            };
        }
        if has_lora && has_pid {
            return Some(Self::LoraPid);
        }
        let profiles = [has_lora.then_some(Self::Lora), has_pid.then_some(Self::Pid)];
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
const SDXL_CANDLE_PROFILES: &[MemoryRouteLoadProfile] = &[
    MemoryRouteLoadProfile::Plain,
    MemoryRouteLoadProfile::Lora,
    MemoryRouteLoadProfile::Pid,
    MemoryRouteLoadProfile::LoraPid,
    MemoryRouteLoadProfile::SingleControl,
    MemoryRouteLoadProfile::LoraSingleControl,
    MemoryRouteLoadProfile::IpAdapter,
    MemoryRouteLoadProfile::LoraIpAdapter,
    MemoryRouteLoadProfile::IpAdapterPid,
    MemoryRouteLoadProfile::LoraIpAdapterPid,
];
const FLUX1_IP_PROFILES: &[MemoryRouteLoadProfile] = &[
    MemoryRouteLoadProfile::IpAdapter,
    MemoryRouteLoadProfile::LoraIpAdapter,
    MemoryRouteLoadProfile::IpAdapterPid,
    MemoryRouteLoadProfile::LoraIpAdapterPid,
];
const FLUX1_CONTROL_PROFILES: &[MemoryRouteLoadProfile] = &[
    MemoryRouteLoadProfile::SingleControl,
    MemoryRouteLoadProfile::LoraSingleControl,
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
// sc-21609: SingleControl/MultiControl/IpAdapter/Identity removed. Control loads consult this
// registry under the `z_image_turbo_control` engine id (image_jobs/zimage.rs), never the base id,
// so those entries could not shape anything and their witnesses claimed routes production cannot
// take under this provider; the entry advertises no character mode, so no identity route exists.
// The control lane's real shaping lives on the dedicated `z_image_turbo_control` row below.
const LEGACY_Z_IMAGE_TURBO_PROFILES: &[MemoryRouteLoadProfile] = &[
    MemoryRouteLoadProfile::Plain,
    MemoryRouteLoadProfile::Lora,
    MemoryRouteLoadProfile::Pid,
];
// sc-21609: the MLX SDXL advanced lanes' real load shapes, widened from text_to_image+plain on
// Michael's coverage decision. Base/edit carry adapters (`image_jobs/sdxl.rs`, `base.rs`
// `with_adapters`); detail loads a tile ControlNet (`image_jobs/detail.rs`, single_control /
// lora_single_control); character sets an IP adapter (`SdxlSubMode::Ip`). `image_inpaint` is
// deliberately absent: production canonicalizes it to `edit_image` before routing, so an
// ImageInpaint witness would name a coordinate no request ever carries.
const SDXL_MLX_BASE_MODES: &[MemoryRouteMode] =
    &[MemoryRouteMode::TextToImage, MemoryRouteMode::EditImage];
const SDXL_MLX_DETAIL_PROFILES: &[MemoryRouteLoadProfile] = &[
    MemoryRouteLoadProfile::SingleControl,
    MemoryRouteLoadProfile::LoraSingleControl,
];
const SDXL_MLX_IP_PROFILES: &[MemoryRouteLoadProfile] = &[
    MemoryRouteLoadProfile::IpAdapter,
    MemoryRouteLoadProfile::LoraIpAdapter,
];
const TEXT_ONLY: &[MemoryRouteMode] = &[MemoryRouteMode::TextToImage];
const CHARACTER_ONLY: &[MemoryRouteMode] = &[MemoryRouteMode::CharacterImage];
const TEXT_AND_STYLE: &[MemoryRouteMode] = &[
    MemoryRouteMode::TextToImage,
    MemoryRouteMode::StyleVariations,
];
const EDIT_MODES: &[MemoryRouteMode] = &[MemoryRouteMode::EditImage, MemoryRouteMode::ImageToImage];
const SANA_MODES: &[MemoryRouteMode] =
    &[MemoryRouteMode::TextToImage, MemoryRouteMode::ImageToImage];
/// SC-20799: the two still modes candle-gen-bernini's `route_ok` admits — its still arm matches the
/// mode keys "text_to_image" (0 references) and "edit" (exactly 1), the latter reached through
/// `MemoryRouteMode::EditImage`. Deliberately NOT `MemoryRouteMode::ImageToImage`: that spelling maps
/// to `MemoryMode::ImageToImage`, whose key the still arm does not match, and it is not in the
/// `bernini_image` entry's `capabilities` either.
const BERNINI_STILL_MODES: &[MemoryRouteMode] =
    &[MemoryRouteMode::TextToImage, MemoryRouteMode::EditImage];
const IDEOGRAM_MODES: &[MemoryRouteMode] = &[
    MemoryRouteMode::TextToImage,
    MemoryRouteMode::ImageToImage,
    MemoryRouteMode::EditImage,
    MemoryRouteMode::ImageInpaint,
];
const QWEN_EDIT_MODES: &[MemoryRouteMode] =
    &[MemoryRouteMode::EditImage, MemoryRouteMode::CharacterImage];
/// SC-24114: Qwen-Image 2.1 serves ONE upstream call — an empty ordered reference list is
/// text-to-image, a non-empty one is the edit — under four public modes.
/// The provider's `MAX_REFERENCE_IMAGES` — this module compiles on every target, so it is restated
/// here and pinned against the linked provider by
/// `qwen_image_2_1_reference_bound_matches_the_linked_provider`.
pub(crate) const QWEN_IMAGE_2_1_MAX_REFERENCES: u32 = 10;
const QWEN_IMAGE_2_1_MODES: &[MemoryRouteMode] = &[
    MemoryRouteMode::TextToImage,
    MemoryRouteMode::EditImage,
    MemoryRouteMode::ImageToImage,
    MemoryRouteMode::CharacterImage,
];
const FLUX2_KLEIN_EDIT_MODES: &[MemoryRouteMode] = &[
    MemoryRouteMode::EditImage,
    MemoryRouteMode::CharacterImage,
    MemoryRouteMode::StyleVariations,
];
const FLUX2_KLEIN_BASE_MODES: &[MemoryRouteMode] =
    &[MemoryRouteMode::TextToImage, MemoryRouteMode::ImageToImage];
/// FLUX.2 Dev's three installed MLX tiers. NVFP4 belongs to the Candle family surface only.
const FLUX2_DEV_TIERS: &[MemoryRouteTier] = &[
    MemoryRouteTier::Bf16,
    MemoryRouteTier::Q4,
    MemoryRouteTier::Q8,
];
const FLUX2_DEV_EDIT_MODES: &[MemoryRouteMode] = &[
    MemoryRouteMode::EditImage,
    MemoryRouteMode::ImageToImage,
    MemoryRouteMode::CharacterImage,
    MemoryRouteMode::StyleVariations,
];
const KOLORS_MODES: &[MemoryRouteMode] = &[
    MemoryRouteMode::TextToImage,
    MemoryRouteMode::StyleVariations,
    MemoryRouteMode::EditImage,
    MemoryRouteMode::CharacterImage,
];
// SC-20790: Candle Kolors has three distinct provider identities.  Do not collapse the base
// registered generator, IP-Adapter character route, and pose-ControlNet route into an MLX-shaped
// "kolors" row: their artifacts and residency receipts are not interchangeable.
const KOLORS_CANDLE_BASE_MODES: &[MemoryRouteMode] =
    &[MemoryRouteMode::TextToImage, MemoryRouteMode::EditImage];
const KOLORS_CANDLE_IP_MODES: &[MemoryRouteMode] = &[MemoryRouteMode::CharacterImage];
const KOLORS_CANDLE_CONTROL_MODES: &[MemoryRouteMode] = &[
    MemoryRouteMode::TextToImage,
    MemoryRouteMode::StyleVariations,
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
const KOLORS_CANDLE_IP_PROFILES: &[MemoryRouteLoadProfile] = &[
    MemoryRouteLoadProfile::IpAdapter,
    MemoryRouteLoadProfile::LoraIpAdapter,
];
const KOLORS_CANDLE_CONTROL_PROFILES: &[MemoryRouteLoadProfile] = &[
    MemoryRouteLoadProfile::SingleControl,
    MemoryRouteLoadProfile::SingleControlPid,
    MemoryRouteLoadProfile::LoraSingleControl,
    MemoryRouteLoadProfile::LoraSingleControlPid,
];
const Q4_Q8: &[MemoryRouteTier] = &[MemoryRouteTier::Q4, MemoryRouteTier::Q8];
const Q4_ONLY: &[MemoryRouteTier] = &[MemoryRouteTier::Q4];
const BF16_ONLY: &[MemoryRouteTier] = &[MemoryRouteTier::Bf16];
const BF16_Q4_Q8: &[MemoryRouteTier] = &[
    MemoryRouteTier::Bf16,
    MemoryRouteTier::Q4,
    MemoryRouteTier::Q8,
];

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
    // sc-24112 — Qwen-Image 2.1, registered on BOTH lanes with the SAME coordinates. That symmetry
    // is the declaration, not a shortcut: after inference #1007 both providers load the same packed
    // artefacts, declare the same `supported_quants: [Q4, Q8]` (plus the dense bf16 snapshot), and
    // publish the same `Resident` / `StagedResidency` / `BoundedDecode` ladder off one derived
    // memory model. A per-lane divergence here would be a claim neither engine makes.
    //
    // `BF16_Q4_Q8` and not `ALL_TIERS`: the catalog ships exactly these three, and NVFP4 is not a
    // tier either provider can serve.
    //
    // `QWEN_IMAGE_2_1_MODES` — every PUBLIC mode the request route serves (sc-24114): text-to-image
    // and the three conditioned faces of the SAME upstream call (`edit_image`, `image_to_image`,
    // `character_image`). Before sc-24114 this was `TEXT_ONLY`, which left every reference request
    // with no declaration at all (candle: `NoRelevantDeclaration`), so the terminal campaign and
    // the admission path could only ever see text-to-image.
    //
    // The provider's `memory_strategy::safety_check` admits `MemoryMode::TextToImage` and PRICES
    // references there (each one an extra `REFERENCE_FIT_TOKENS` joint-sequence block, bounded by
    // `MAX_REFERENCE_IMAGES`) — there is no separate provider edit mode. So the manifest's
    // `requestContexts` map every public mode onto `providerMode: "text_to_image"`, and
    // [`expected_provider_mode`] states that mapping for this provider: the public coordinate
    // (what the matrix and the campaign key on) stays the request's own mode, while the provider
    // receives the one mode it implements.
    //
    // `PLAIN_LORA` (sc-24163): the pinned providers apply LoRA/LoKr (MLX sc-24156, Candle
    // sc-24157) and the pinned engine reports `supports_lora=true`, and since sc-24158 the routing
    // catalog sends a LoRA-carrying 2.1 job to Candle (`candle_quant_lora`). Until the terminal pin
    // bump these rows stayed `PLAIN` (sc-24158) because they are published as
    // `memoryRouteWitnesses` in the checked-in engine capability dumps and the macOS lane's
    // fresh-dump check reds on any witness the pinned dump lacks; the `lora` profile moved here
    // together with that re-dump and the `lora` overlay/profile on the manifest's hand-authored
    // contract rows.
    //
    // `requires_sequential_selection: false`: Resident is reachable with no sequential selection;
    // only the staged rung asks for one. `legacy_shaping: false`: this coordinate is
    // declaration-owned and was never in the pre-declaration shaper, so removing the declaration
    // must make it unreachable rather than fall back to legacy shaping.
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "qwen_image_2_1",
        tiers: BF16_Q4_Q8,
        modes: QWEN_IMAGE_2_1_MODES,
        load_profiles: PLAIN_LORA,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "qwen_image_2_1",
        tiers: BF16_Q4_Q8,
        modes: QWEN_IMAGE_2_1_MODES,
        load_profiles: PLAIN_LORA,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    // Qwen-Image 2.1 Turbo (sc-25907, epic 25899): the same provider crates register the distilled
    // checkpoint under its own id with the base contract's rungs, tiers, modes and load profiles —
    // under Turbo's own calibration fingerprint, so no base record can answer for it. Same rule
    // shape as the base rows above, for the same reasons.
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "qwen_image_2_1_turbo",
        tiers: BF16_Q4_Q8,
        modes: QWEN_IMAGE_2_1_MODES,
        load_profiles: PLAIN_LORA,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "qwen_image_2_1_turbo",
        tiers: BF16_Q4_Q8,
        modes: QWEN_IMAGE_2_1_MODES,
        load_profiles: PLAIN_LORA,
        requires_sequential_selection: false,
        legacy_shaping: false,
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
    // SC-20795: the six built-in Mage routes share one provider implementation, but their public
    // request identities do not. Generation admits plain or LoRA loads with no references; edit
    // admits only the manifest's plain, single-reference instruction-edit shape. These are new
    // declaration-owned coordinates so a missing or crossed manifest row cannot fall through to
    // the historical provider-id shaper.
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "mage_flow_base",
        tiers: BF16_Q4_Q8,
        modes: TEXT_ONLY,
        load_profiles: PLAIN_LORA,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "mage_flow",
        tiers: BF16_Q4_Q8,
        modes: TEXT_ONLY,
        load_profiles: PLAIN_LORA,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "mage_flow_turbo",
        tiers: BF16_Q4_Q8,
        modes: TEXT_ONLY,
        load_profiles: PLAIN_LORA,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "mage_flow_edit_base",
        tiers: BF16_Q4_Q8,
        modes: &[MemoryRouteMode::EditImage],
        load_profiles: PLAIN,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "mage_flow_edit",
        tiers: BF16_Q4_Q8,
        modes: &[MemoryRouteMode::EditImage],
        load_profiles: PLAIN,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "mage_flow_edit_turbo",
        tiers: BF16_Q4_Q8,
        modes: &[MemoryRouteMode::EditImage],
        load_profiles: PLAIN,
        requires_sequential_selection: false,
        legacy_shaping: false,
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
        provider: "flux1_schnell",
        tiers: ALL_TIERS,
        modes: TEXT_AND_STYLE,
        load_profiles: PLAIN_LORA_PID,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "flux1_dev",
        tiers: ALL_TIERS,
        modes: TEXT_AND_STYLE,
        load_profiles: PLAIN_LORA_PID,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "flux1_dev",
        tiers: ALL_TIERS,
        modes: CHARACTER_ONLY,
        load_profiles: FLUX1_IP_PROFILES,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "flux1_dev_control",
        tiers: ALL_TIERS,
        modes: &[
            MemoryRouteMode::TextToImage,
            MemoryRouteMode::StyleVariations,
            MemoryRouteMode::CharacterImage,
        ],
        load_profiles: FLUX1_CONTROL_PROFILES,
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
    // SC-20794: base, multi-reference edit, and Fun-Controlnet-Union are distinct MLX providers.
    // Their public mode/reference/control-load axes must remain separate; no rung transfers across
    // these sibling routes merely because they share the FLUX.2 Dev family.
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "flux2_dev",
        tiers: FLUX2_DEV_TIERS,
        modes: TEXT_ONLY,
        load_profiles: PLAIN,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "flux2_dev_edit",
        tiers: FLUX2_DEV_TIERS,
        modes: FLUX2_DEV_EDIT_MODES,
        load_profiles: PLAIN,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "flux2_dev_control",
        tiers: FLUX2_DEV_TIERS,
        modes: TEXT_ONLY,
        load_profiles: SINGLE_CONTROL,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "flux2_klein_9b",
        tiers: ALL_TIERS,
        modes: FLUX2_KLEIN_BASE_MODES,
        load_profiles: PLAIN_LORA_PID,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "flux2_klein_9b_edit",
        tiers: ALL_TIERS,
        modes: FLUX2_KLEIN_EDIT_MODES,
        load_profiles: PLAIN_LORA_PID,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "flux2_klein_9b_kv_edit",
        tiers: ALL_TIERS,
        modes: FLUX2_KLEIN_EDIT_MODES,
        load_profiles: PLAIN_LORA_PID,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    // sc-21609: `BF16_Q4_Q8`, not `ALL_TIERS` — Metal has no FP4 hardware and
    // `nvfp4_host_eligible()` is hard-false on macOS, so an MLX nvfp4 witness can never
    // correspond to a real load (the FLUX.2 rows record the same convention).
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "sdxl",
        tiers: BF16_Q4_Q8,
        modes: SDXL_MLX_BASE_MODES,
        load_profiles: PLAIN_LORA,
        requires_sequential_selection: false,
        legacy_shaping: true,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "sdxl",
        tiers: BF16_Q4_Q8,
        modes: &[MemoryRouteMode::ImageDetail],
        load_profiles: SDXL_MLX_DETAIL_PROFILES,
        requires_sequential_selection: false,
        legacy_shaping: true,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "sdxl",
        tiers: BF16_Q4_Q8,
        modes: CHARACTER_ONLY,
        load_profiles: SDXL_MLX_IP_PROFILES,
        requires_sequential_selection: false,
        legacy_shaping: true,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "sdxl",
        tiers: BF16_Q4_Q8,
        modes: ALL_MODES,
        load_profiles: SDXL_CANDLE_PROFILES,
        requires_sequential_selection: false,
        legacy_shaping: false,
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
    // sc-21609: the strict-pose control lanes consult the registry under these `_control` engine
    // ids (`image_jobs/zimage.rs` ZIMAGE_CONTROL_ENGINE_ID / ZIMAGE_BASE_CONTROL_ENGINE_ID), so
    // without rows here every control load stayed EagerMaterialization and the engines' fully
    // implemented rung-4 control surfaces were unreachable. `legacy_shaping: true` because these
    // rows must actually shape the spec — a witness-only row would claim a deferred route
    // production does not take. Single control only: the lane gates on pose inputs and the Candle
    // twins declare the same profile.
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "z_image_turbo_control",
        tiers: BF16_Q4_Q8,
        modes: TEXT_ONLY,
        load_profiles: SINGLE_CONTROL,
        requires_sequential_selection: false,
        legacy_shaping: true,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "z_image_control",
        tiers: BF16_Q4_Q8,
        modes: TEXT_ONLY,
        load_profiles: SINGLE_CONTROL,
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
    // sc-22731: the MLX twins of the two Candle SANA rows above. Until these existed the registry
    // named 37 MLX providers and NEITHER SANA route, so `mlx-gen-sana`'s windowed rung 4
    // (`bounded_transformer_residency` over `TRANSFORMER_WINDOW_COMPONENT::Dit`, inference PR 951)
    // was unroutable by construction: `evaluate_declared_mlx_load_shape_for_request_with_strategy`
    // refuses any provider with no `Mlx` rule, so no MLX SANA declaration could ever be consulted.
    //
    // `ALL_TIERS` like the three MLX Chroma1 rows above — the packed macOS turnkeys ship q4, q8 and
    // bf16 and `mlx-gen-sana` opens each. `SANA_MODES` and `PLAIN`, exactly as the Candle rows: the
    // engine's route gate admits text_to_image and image_to_image, and the MLX still lane builds its
    // spec with no adapters. `requires_sequential_selection: true` because PR 951's
    // `contract_with_asset_facts` declares the windowed rung ONLY under `OffloadPolicy::Sequential`
    // — a Resident load must therefore keep the shape it came in with, which is what the
    // `matching.all(|rule| rule.requires_sequential_selection)` short-circuit below guarantees, and
    // why the six `sana_*:*:mlx` RESIDENT anchors keep `eager_materialization`.
    //
    // `legacy_shaping: false` and NOT the `true` the MLX Chroma1 rows carry: these coordinates did
    // not exist in the pre-declaration shaper, so they must not become reachable through
    // `apply_registered_load_shape` merely because a manifest declaration is absent. The cost is
    // that the lane is request-context-owned (`requires_request_context` below), so it stays
    // refused-and-eager — exactly what it is today with no rule at all — until the
    // `sana_*.mlx` `memoryStrategyContract` blocks carry `requestContexts`.
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "sana_1600m",
        tiers: ALL_TIERS,
        modes: SANA_MODES,
        load_profiles: PLAIN,
        requires_sequential_selection: true,
        legacy_shaping: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "sana_sprint_1600m",
        tiers: ALL_TIERS,
        modes: SANA_MODES,
        load_profiles: PLAIN,
        requires_sequential_selection: true,
        legacy_shaping: false,
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
        provider: "sana_1600m",
        tiers: BF16_ONLY,
        modes: SANA_MODES,
        load_profiles: PLAIN,
        requires_sequential_selection: true,
        legacy_shaping: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "sana_sprint_1600m",
        tiers: BF16_ONLY,
        modes: SANA_MODES,
        load_profiles: PLAIN,
        requires_sequential_selection: true,
        legacy_shaping: false,
    },
    // SC-20799 gap 1: the three Anima Candle routes. One provider contract builder serves all three
    // (candle-gen-anima `IDS`), but each keeps its own calibration identity, so they are three rules
    // rather than one shared coordinate. `TEXT_ONLY` because both the engine route gate and the
    // entries' `capabilities` admit text_to_image alone; `BF16_ONLY` because `resolved_numeric_tier`
    // refuses any precision but bf16 AND the off-Mac download is the raw dense `split_files/` tree
    // (the `anima_quant` tier converter is macOS-only, so there is no other tier to resolve).
    // `PLAIN_LORA` because Resident admits a LoRA/LoKr overlay; staged residency refuses one, which
    // is recorded as a structural exemption on the manifest declaration rather than by narrowing the
    // load profiles here — the lora profile IS reachable, just not on the staged rung.
    // `requires_sequential_selection` stays false: Resident is reachable without a sequential
    // selection, and only the staged row carries `requiredOffloadPolicy: "sequential"`.
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "anima_base",
        tiers: BF16_ONLY,
        modes: TEXT_ONLY,
        load_profiles: PLAIN_LORA,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "anima_aesthetic",
        tiers: BF16_ONLY,
        modes: TEXT_ONLY,
        load_profiles: PLAIN_LORA,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "anima_turbo",
        tiers: BF16_ONLY,
        modes: TEXT_ONLY,
        load_profiles: PLAIN_LORA,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    // SC-20799 gap 1: the three Boogu Candle routes. `BF16_Q4_Q8` matches
    // candle-gen-boogu `validate_load_spec` (bf16 precision, quantize in {None, Q4, Q8}) and the
    // three turnkey subdirs. `PLAIN` only: the same function REFUSES adapters, control, extra
    // controls, IP-adapter, identity and external text encoders outright, so no overlay-bearing load
    // profile is admissible — the entries' `loraCompatibility` blocks are reserved family
    // placeholders with empty `types` lists. Base and Turbo are text-to-image; Edit is the
    // `Route::Edit` arm, which admits `MemoryMode::Edit` at 1..=5 references and is the only mode in
    // that entry's `capabilities`. The engine's base/turbo image_to_image arm is deliberately not
    // routed: neither entry lists `image_to_image` in `capabilities`, so it has no catalog axis.
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "boogu_image",
        tiers: BF16_Q4_Q8,
        modes: TEXT_ONLY,
        load_profiles: PLAIN,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "boogu_image_turbo",
        tiers: BF16_Q4_Q8,
        modes: TEXT_ONLY,
        load_profiles: PLAIN,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "boogu_image_edit",
        tiers: BF16_Q4_Q8,
        modes: &[MemoryRouteMode::EditImage],
        load_profiles: PLAIN,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    // SC-20799 gap 1: the Candle Bernini STILL-image ladder. The provider is the ENGINE id `bernini`
    // (`crate::bernini::MODEL_ID`, the FULL_MEMORY_REGISTRATION provider), which the MODEL_TABLE maps
    // the `bernini_image` sceneworks id onto — the same id the video entry uses, distinguished here
    // by the image modes. `route_ok`'s still arm admits mode key "text_to_image" (0 references) and
    // "edit" (exactly 1), the latter reached through `MemoryRouteMode::EditImage` because
    // `candle_memory_strategy::request_mode` maps the "edit_image" spelling to `MemoryMode::Edit`.
    // `BF16_Q4_Q8` matches `expected_packing` (None/Q4/Q8) and the published tier subdirs.
    // `PLAIN_LORA` because the still route accepts an absent overlay or exactly "lora", and the
    // off-Mac Candle descriptor genuinely serves LoRA/LoKr. Bernini declares no staged-residency
    // rung, so nothing here gates on a sequential selection.
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "bernini",
        tiers: BF16_Q4_Q8,
        modes: BERNINI_STILL_MODES,
        load_profiles: PLAIN_LORA,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    // sc-21609: the MLX twin, enabling the deferred shape Michael's coverage decision turned on.
    // mlx-gen-bernini implements rung 4 at every tier behind `structurally_streamable` (Dir +
    // DeferredMaterialization + no adapters); until this row existed no MLX bernini load was ever
    // shaped deferred, so the whole surface was unreachable. `PLAIN` and not `PLAIN_LORA`: the MLX
    // still lane builds its spec with no adapters at all (`image_jobs/bernini.rs`), so a Lora
    // witness here would be fiction — and the block stream refuses adapters regardless. The MLX
    // consult site passes `TextToImage` for every cold load, so plain VIDEO loads shape through
    // this row too (adapted video loads keep their Wan adapters and stay off the stream at the
    // engine gate); the registry's image-mode vocabulary cannot witness the video modes — that
    // residue is recorded on the rung-4 survey.
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "bernini",
        tiers: BF16_Q4_Q8,
        modes: BERNINI_STILL_MODES,
        load_profiles: PLAIN,
        requires_sequential_selection: false,
        legacy_shaping: true,
    },
    // SC-20788: all three Chroma turnkey routes expose the same exact request-scoped Candle
    // Resident/Staged surface. Their public identity remains route-local, and PiD is a typed load
    // profile rather than a mode or an untracked overlay.
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "chroma1_hd",
        tiers: BF16_Q4_Q8,
        modes: TEXT_ONLY,
        load_profiles: PLAIN_LORA_PID,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "chroma1_base",
        tiers: BF16_Q4_Q8,
        modes: TEXT_ONLY,
        load_profiles: PLAIN_LORA_PID,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "chroma1_flash",
        tiers: BF16_Q4_Q8,
        modes: TEXT_ONLY,
        load_profiles: PLAIN_LORA_PID,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    // SC-20790: the Candle base/edit, IP character, and pose-ControlNet routes remain separate
    // receipt/evidence domains.  In particular this leaves impossible IP+control and IP+PiD shapes
    // without a matching rule instead of granting them through the base profile.
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "kolors",
        tiers: BF16_Q4_Q8,
        modes: KOLORS_CANDLE_BASE_MODES,
        load_profiles: PLAIN_LORA_PID,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "candle_kolors_ipadapter",
        tiers: BF16_Q4_Q8,
        modes: KOLORS_CANDLE_IP_MODES,
        load_profiles: KOLORS_CANDLE_IP_PROFILES,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "candle_kolors_control",
        tiers: BF16_Q4_Q8,
        modes: KOLORS_CANDLE_CONTROL_MODES,
        load_profiles: KOLORS_CANDLE_CONTROL_PROFILES,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    // SC-20789: Ideogram Base and Turbo share the public request surface but retain distinct
    // provider receipts (Base owns two DiTs; Turbo owns one DiT plus its mandatory bundled
    // TurboTime adapter). Native/PiD and ordered user-adapter loads are exact typed profiles.
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "ideogram_4",
        tiers: BF16_Q4_Q8,
        modes: IDEOGRAM_MODES,
        load_profiles: PLAIN_LORA_PID,
        requires_sequential_selection: false,
        legacy_shaping: false,
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Candle,
        provider: "ideogram_4_turbo",
        tiers: BF16_Q4_Q8,
        modes: IDEOGRAM_MODES,
        load_profiles: PLAIN_LORA_PID,
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
        tiers: BF16_Q4_Q8,
        modes: QWEN_EDIT_MODES,
        load_profiles: PLAIN_LORA,
        requires_sequential_selection: false,
        legacy_shaping: false,
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

/// The Candle bespoke providers whose request admission `declared_candle_bespoke_request` is
/// authoritative for (SC-20790). Load-bearing in two places: the admission below refuses any
/// provider outside this census, and `scripts/generate-memory-matrix.mjs` reads this const AS TEXT
/// (`parseCandleBespokeRequestProviders`) so the published matrix can see receipt-backed bespoke
/// coverage that no manifest declaration row carries (sc-20799). Growing the census without
/// teaching the generator's lane map about the new provider fails the matrix build closed rather
/// than publishing a delivered lane as `Missing`.
pub(crate) const CANDLE_BESPOKE_REQUEST_PROVIDERS: &[&str] =
    &["candle_kolors_ipadapter", "candle_kolors_control"];

/// Exact typed authority for bespoke Candle providers whose physical contract is assembled from
/// request-local component paths instead of the registered catalog loader. This is intentionally
/// independent of legacy load shaping: absence or a crossed coordinate is a terminal pre-load
/// refusal, never permission to fall through to an older estimate.
#[cfg_attr(
    not(any(test, all(not(target_os = "macos"), feature = "backend-candle"))),
    allow(dead_code)
)]
pub(crate) fn declared_candle_bespoke_request(
    runtime_provider: &'static str,
    resolved_tier: Option<&str>,
    request_mode: &str,
    spec: &LoadSpec,
    context: MemoryRouteRequestContext,
) -> bool {
    let (Some(tier), Some(mode), Some(load_profile)) = (
        resolved_tier.and_then(MemoryRouteTier::from_resolved_tier),
        MemoryRouteMode::from_request(request_mode),
        MemoryRouteLoadProfile::from_spec(spec),
    ) else {
        return false;
    };
    if context.mode != mode || context.use_pid != spec.pid.is_some() || context.has_phases {
        return false;
    }
    // The census gate and the per-provider shape arms are one conjunction: a provider outside
    // `CANDLE_BESPOKE_REQUEST_PROVIDERS` can never admit, and a censused provider still has to
    // satisfy its exact reference shape.
    let reference_shape_matches = CANDLE_BESPOKE_REQUEST_PROVIDERS.contains(&runtime_provider)
        && match runtime_provider {
            "candle_kolors_ipadapter" => context.reference_count == 1 && !context.use_pid,
            "candle_kolors_control" => context.reference_count == 0,
            _ => false,
        };
    reference_shape_matches
        && rule_matches(
            MemoryRouteSelector {
                backend: MemoryRouteBackend::Candle,
                provider: runtime_provider,
                tier,
                mode,
                overlay: load_profile.overlay(),
                load_profile,
            },
            false,
        )
}

fn closed_array_contains<T: Copy + PartialEq>(
    implementation: &JsonObject<String, Value>,
    field: &str,
    expected: T,
    parse: impl Fn(&str) -> Option<T>,
) -> Result<bool, ()> {
    let values = implementation
        .get(field)
        .and_then(Value::as_array)
        .ok_or(())?;
    if values.is_empty() {
        return Err(());
    }
    let parsed = values
        .iter()
        .map(|value| value.as_str().and_then(&parse).ok_or(()))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(parsed.contains(&expected))
}

fn closed_source_kind_array_contains(
    implementation: &JsonObject<String, Value>,
    expected: &str,
) -> Result<bool, ()> {
    let values = implementation
        .get("sourceKinds")
        .and_then(Value::as_array)
        .ok_or(())?;
    if values.is_empty() {
        return Err(());
    }
    let parsed = values
        .iter()
        .map(|value| {
            value
                .as_str()
                .filter(|source| matches!(*source, "dir" | "file"))
                .ok_or(())
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(parsed.contains(&expected))
}

fn implementation_rung(implementation: &JsonObject<String, Value>) -> Result<&str, ()> {
    implementation
        .get("rung")
        .and_then(Value::as_str)
        .filter(|rung| {
            matches!(
                *rung,
                "resident"
                    | "staged_residency"
                    | "bounded_decode"
                    | "bounded_attention"
                    | "bounded_transformer_residency"
            )
        })
        .ok_or(())
}

fn implementation_runtime_provider<'a>(
    implementation: &'a JsonObject<String, Value>,
    contract_provider: &'a str,
) -> Result<&'a str, ()> {
    match implementation.get("runtimeProvider") {
        None => Ok(contract_provider),
        Some(Value::String(provider)) if !provider.is_empty() => Ok(provider),
        Some(_) => Err(()),
    }
}

fn implementation_declares_selector(
    implementation: &Value,
    contract_provider: &str,
    selector: MemoryRouteSelector,
) -> Result<bool, ()> {
    let implementation = implementation.as_object().ok_or(())?;
    let rung = implementation_rung(implementation)?;
    let runtime_provider = implementation_runtime_provider(implementation, contract_provider)?;
    if rung != "bounded_transformer_residency" {
        return Ok(false);
    }
    if runtime_provider != selector.provider {
        return Ok(false);
    }
    // Parse every closed coordinate before combining membership. A same-provider candidate with a
    // nonmatching tier must not hide a malformed later field behind boolean short-circuiting.
    let tier_matches = closed_array_contains(
        implementation,
        "tiers",
        selector.tier,
        MemoryRouteTier::from_str,
    )?;
    let mode_matches = closed_array_contains(
        implementation,
        "modes",
        selector.mode,
        MemoryRouteMode::from_manifest,
    )?;
    let overlay_matches = closed_array_contains(
        implementation,
        "overlays",
        selector.overlay,
        MemoryRouteOverlay::from_str,
    )?;
    Ok(tier_matches && mode_matches && overlay_matches)
}

fn mlx_request_implementation_matches(
    implementation: &Value,
    contract_provider: &str,
    selector: MemoryRouteSelector,
    spec: &LoadSpec,
    context: MemoryRouteRequestContext,
    requires_request_context: bool,
) -> Result<bool, ()> {
    let implementation_object = implementation.as_object().ok_or(())?;
    // This pass evaluates BTR declarations. Legacy Resident inventory rows need no request
    // contexts, and staged declarations are validated separately if no BTR row applies.
    if implementation_rung(implementation_object)? != "bounded_transformer_residency" {
        return Ok(false);
    }
    if implementation_object.get("requestContexts").is_none() {
        if requires_request_context {
            return Err(());
        }
        // Legacy BTR declarations predate load-profile/source/provider-overlay request ownership.
        // Preserve their established selector-only predicate exactly.
        return implementation_declares_selector(implementation, contract_provider, selector);
    }
    let static_axes_match = request_implementation_static_axes_match(
        implementation,
        contract_provider,
        selector,
        spec,
        Some("bounded_transformer_residency"),
    )?;
    let context_matches = request_implementation_context_matches(
        implementation,
        contract_provider,
        context,
        requires_request_context,
    )?;
    Ok(static_axes_match && context_matches)
}

fn request_implementation_context_matches(
    implementation: &Value,
    contract_provider: &str,
    context: MemoryRouteRequestContext,
    requires_request_context: bool,
) -> Result<bool, ()> {
    let implementation = implementation.as_object().ok_or(())?;
    let Some(request_contexts) = implementation.get("requestContexts") else {
        if requires_request_context {
            return Err(());
        }
        // Existing declaration-owned routes predate the request-context schema. Their exact
        // provider/tier/mode/overlay/source predicate remains authoritative and unchanged.
        return Ok(true);
    };
    let runtime_provider = implementation_runtime_provider(implementation, contract_provider)?;
    let request_contexts = request_contexts.as_array().ok_or(())?;
    let matches = request_contexts
        .iter()
        .map(|request_context| {
            request_strategy_provider_mode_is_exact(request_context, runtime_provider)?;
            request_strategy_context_matches(request_context, context)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let matching = request_contexts
        .iter()
        .zip(matches)
        .filter_map(|(request_context, matched)| matched.then_some(request_context))
        .collect::<Vec<_>>();
    let [matching] = matching.as_slice() else {
        return Ok(false);
    };
    let expected_provider_mode = expected_provider_mode(runtime_provider, context);
    Ok(matching.get("providerMode").and_then(Value::as_str) == Some(expected_provider_mode))
}

fn request_implementation_static_axes_match(
    implementation: &Value,
    contract_provider: &str,
    selector: MemoryRouteSelector,
    spec: &LoadSpec,
    expected_rung: Option<&str>,
) -> Result<bool, ()> {
    let implementation = implementation.as_object().ok_or(())?;
    let rung = implementation_rung(implementation)?;
    let runtime_provider = implementation_runtime_provider(implementation, contract_provider)?;
    if runtime_provider != selector.provider
        || expected_rung.is_some_and(|expected| rung != expected)
    {
        return Ok(false);
    }
    let tier_matches = closed_array_contains(
        implementation,
        "tiers",
        selector.tier,
        MemoryRouteTier::from_str,
    )?;
    let mode_matches = closed_array_contains(
        implementation,
        "modes",
        selector.mode,
        MemoryRouteMode::from_manifest,
    )?;
    let overlay_matches = closed_array_contains(
        implementation,
        "overlays",
        selector.overlay,
        MemoryRouteOverlay::from_str,
    )?;
    let Some(load_profile) = MemoryRouteLoadProfile::from_spec(spec) else {
        return Ok(false);
    };
    let source_kind = match spec.weights {
        WeightsSource::Dir(_) => "dir",
        WeightsSource::File(_) => "file",
    };
    let load_profile_matches = closed_array_contains(
        implementation,
        "loadProfiles",
        load_profile,
        MemoryRouteLoadProfile::from_str,
    )?;
    let source_kind_matches = closed_source_kind_array_contains(implementation, source_kind)?;
    let public_overlay = match load_profile.overlay() {
        MemoryRouteOverlay::Control => Some("control".to_owned()),
        MemoryRouteOverlay::Identity => Some("identity".to_owned()),
        MemoryRouteOverlay::None | MemoryRouteOverlay::Lora => None,
    };
    let expected_provider_overlay = crate::mlx_fit_gate::provider_overlay_for_load_spec(
        selector.provider,
        spec,
        public_overlay,
    )
    .unwrap_or_else(|| "none".to_owned());
    let declared_provider_overlay = implementation
        .get("providerOverlay")
        .and_then(Value::as_str)
        .filter(|overlay| {
            matches!(
                *overlay,
                "none"
                    | "lora"
                    | "identity"
                    | "control"
                    | "adapters"
                    | "pid"
                    | "adapters-pid"
                    | "ip-adapter"
                    | "adapters-ip-adapter"
                    | "ip-adapter-pid"
                    | "adapters-ip-adapter-pid"
                    | "control-adapters"
            )
        })
        .ok_or(())?;
    Ok(tier_matches
        && mode_matches
        && overlay_matches
        && load_profile_matches
        && source_kind_matches
        && declared_provider_overlay == expected_provider_overlay)
}

fn expected_provider_mode(
    runtime_provider: &str,
    context: MemoryRouteRequestContext,
) -> &'static str {
    match (runtime_provider, context.mode) {
        // SC-24114: Qwen-Image 2.1's provider implements ONE memory mode and prices the ordered
        // reference list inside it (0..=MAX_REFERENCE_IMAGES extra joint-sequence blocks).
        (
            "qwen_image_2_1" | "qwen_image_2_1_turbo",
            MemoryRouteMode::TextToImage
            | MemoryRouteMode::EditImage
            | MemoryRouteMode::ImageToImage
            | MemoryRouteMode::CharacterImage,
        ) if context.reference_count <= QWEN_IMAGE_2_1_MAX_REFERENCES => "text_to_image",
        ("qwen_image_edit", MemoryRouteMode::EditImage)
            if (1..=5).contains(&context.reference_count) =>
        {
            "edit_image"
        }
        ("qwen_image_edit", MemoryRouteMode::CharacterImage)
            if (1..=5).contains(&context.reference_count) =>
        {
            "character_image"
        }
        (
            "flux1_schnell" | "flux1_dev",
            MemoryRouteMode::TextToImage | MemoryRouteMode::StyleVariations,
        ) if context.reference_count == 0 => "text_to_image",
        ("flux1_dev", MemoryRouteMode::CharacterImage) if context.reference_count == 1 => {
            "image_to_image"
        }
        (
            "flux1_dev_control",
            MemoryRouteMode::TextToImage
            | MemoryRouteMode::StyleVariations
            | MemoryRouteMode::CharacterImage,
        ) if context.reference_count == 1 => "image_to_image",
        (
            "flux2_klein_9b_edit" | "flux2_klein_9b_kv_edit",
            MemoryRouteMode::StyleVariations
            | MemoryRouteMode::EditImage
            | MemoryRouteMode::CharacterImage,
        ) if (1..=8).contains(&context.reference_count) => "edit_image",
        (
            "flux2_dev_edit",
            MemoryRouteMode::StyleVariations
            | MemoryRouteMode::EditImage
            | MemoryRouteMode::ImageToImage
            | MemoryRouteMode::CharacterImage,
        ) if (2..=8).contains(&context.reference_count) => "edit_image",
        ("flux2_dev_control", MemoryRouteMode::TextToImage) if context.reference_count == 1 => {
            "text_to_image"
        }
        ("ideogram_4" | "ideogram_4_turbo", MemoryRouteMode::ImageInpaint)
            if context.reference_count == 2 =>
        {
            "edit_image"
        }
        (_, MemoryRouteMode::EditImage) => "edit_image",
        (_, MemoryRouteMode::CharacterImage) if context.reference_count == 1 => "image_to_image",
        (_, MemoryRouteMode::TextToImage) if context.reference_count == 1 => "image_to_image",
        (_, MemoryRouteMode::TextToImage) => "text_to_image",
        (_, other) => other.as_str(),
    }
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

#[allow(clippy::too_many_arguments)]
fn request_strategy_implementation_matches(
    implementation: &Value,
    contract_provider: &str,
    runtime_provider: &str,
    tier: MemoryRouteTier,
    load_profile: MemoryRouteLoadProfile,
    source_kind: &str,
    context: MemoryRouteRequestContext,
    expected_provider_overlay: Option<&str>,
) -> Result<bool, ()> {
    let implementation = implementation.as_object().ok_or(())?;
    let rung = implementation_rung(implementation)?;
    let provider = implementation_runtime_provider(implementation, contract_provider)?;
    if rung != "staged_residency" {
        return Ok(false);
    }
    // `requestContexts` is the declaration-driven ownership marker. Older staged rows which only
    // describe calibration inventory remain on their unchanged legacy paths.
    let Some(request_contexts) = implementation.get("requestContexts") else {
        return Ok(false);
    };
    let request_contexts = request_contexts.as_array().ok_or(())?;
    if provider != runtime_provider {
        return Ok(false);
    }
    let tier_matches =
        closed_array_contains(implementation, "tiers", tier, MemoryRouteTier::from_str)?;
    let mode_matches = closed_array_contains(
        implementation,
        "modes",
        context.mode,
        MemoryRouteMode::from_manifest,
    )?;
    let overlay_matches = closed_array_contains(
        implementation,
        "overlays",
        load_profile.overlay(),
        MemoryRouteOverlay::from_str,
    )?;
    let load_profile_matches = closed_array_contains(
        implementation,
        "loadProfiles",
        load_profile,
        MemoryRouteLoadProfile::from_str,
    )?;
    let source_kind_matches = closed_source_kind_array_contains(implementation, source_kind)?;
    let provider_overlay_matches = match expected_provider_overlay {
        Some(expected) => {
            implementation
                .get("providerOverlay")
                .and_then(Value::as_str)
                .ok_or(())?
                == expected
        }
        None => true,
    };
    let matches = request_contexts
        .iter()
        .map(|request_context| {
            request_strategy_provider_mode_is_exact(request_context, runtime_provider)?;
            request_strategy_context_matches(request_context, context)
        })
        .collect::<Result<Vec<_>, _>>()?;
    if !(tier_matches
        && mode_matches
        && overlay_matches
        && load_profile_matches
        && source_kind_matches
        && provider_overlay_matches)
    {
        return Ok(false);
    }
    let matching = request_contexts
        .iter()
        .zip(matches)
        .filter_map(|(request_context, matched)| matched.then_some(request_context))
        .collect::<Vec<_>>();
    let [matching] = matching.as_slice() else {
        return Ok(false);
    };
    Ok(matching.get("providerMode").and_then(Value::as_str)
        == Some(expected_provider_mode(runtime_provider, context)))
}

fn expected_provider_overlay(
    provider: &str,
    spec: &LoadSpec,
    load_profile: MemoryRouteLoadProfile,
) -> String {
    let public_overlay = match load_profile.overlay() {
        MemoryRouteOverlay::Control => Some("control".to_owned()),
        MemoryRouteOverlay::Identity => Some("identity".to_owned()),
        MemoryRouteOverlay::None | MemoryRouteOverlay::Lora => None,
    };
    crate::mlx_fit_gate::provider_overlay_for_load_spec(provider, spec, public_overlay)
        .unwrap_or_else(|| "none".to_owned())
}

fn request_strategy_provider_mode_is_exact(
    request_context: &Value,
    runtime_provider: &str,
) -> Result<(), ()> {
    let request_context = request_context.as_object().ok_or(())?;
    let mode = request_context
        .get("mode")
        .and_then(Value::as_str)
        .and_then(MemoryRouteMode::from_manifest)
        .ok_or(())?;
    let provider_mode = request_context
        .get("providerMode")
        .and_then(Value::as_str)
        .ok_or(())?;
    let reference_counts = request_context
        .get("referenceCounts")
        .and_then(Value::as_array)
        .ok_or(())?;
    if reference_counts.is_empty() {
        return Err(());
    }
    let pid = request_context
        .get("pid")
        .and_then(Value::as_array)
        .ok_or(())?;
    if pid.is_empty() {
        return Err(());
    }
    pid.iter()
        .map(Value::as_bool)
        .collect::<Option<Vec<_>>>()
        .ok_or(())?;
    request_context
        .get("hasPhases")
        .and_then(Value::as_bool)
        .ok_or(())?;
    reference_counts.iter().try_for_each(|reference_count| {
        let reference_count = reference_count
            .as_u64()
            .and_then(|count| u32::try_from(count).ok())
            .ok_or(())?;
        (provider_mode
            == expected_provider_mode(
                runtime_provider,
                MemoryRouteRequestContext {
                    mode,
                    reference_count,
                    use_pid: false,
                    has_phases: false,
                },
            ))
        .then_some(())
        .ok_or(())
    })
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
        "text_to_image" | "image_to_image" | "edit_image" | "character_image"
    ) {
        return Err(());
    }
    let reference_counts = request_context
        .get("referenceCounts")
        .and_then(Value::as_array)
        .ok_or(())?;
    if reference_counts.is_empty() {
        return Err(());
    }
    let reference_counts = reference_counts
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|count| u32::try_from(count).ok())
                .ok_or(())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let pid = request_context
        .get("pid")
        .and_then(Value::as_array)
        .ok_or(())?;
    if pid.is_empty() {
        return Err(());
    }
    let pid = pid
        .iter()
        .map(|value| value.as_bool().ok_or(()))
        .collect::<Result<Vec<_>, _>>()?;
    let has_phases = request_context
        .get("hasPhases")
        .and_then(Value::as_bool)
        .ok_or(())?;
    Ok(mode == context.mode.as_str()
        && reference_counts.contains(&context.reference_count)
        && pid.contains(&context.use_pid)
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
                None,
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
            None,
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
                    == Ok(true)
            })
        })
}

/// Require every declaration row at one exact static coordinate to carry the same singular typed
/// request identity. The selector may choose any rung after admission, so validating only the BTR or
/// staged row would let a lower rung borrow authority from a sibling whose request surface differs.
fn manifest_declares_selector_for_request(
    manifest: &JsonObject<String, Value>,
    selector: MemoryRouteSelector,
    spec: &LoadSpec,
    context: MemoryRouteRequestContext,
) -> Result<bool, ()> {
    if context.mode != selector.mode || context.use_pid != spec.pid.is_some() {
        return Ok(false);
    }
    let contract = manifest_contract(manifest, selector.backend).ok_or(())?;
    let contract_provider = contract.get("provider").and_then(Value::as_str).ok_or(())?;
    let implementations = contract
        .get("implementations")
        .and_then(Value::as_array)
        .ok_or(())?;
    let mut relevant = 0_usize;
    for implementation in implementations {
        if request_implementation_static_axes_match(
            implementation,
            contract_provider,
            selector,
            spec,
            None,
        )? {
            relevant += 1;
            if !request_implementation_context_matches(
                implementation,
                contract_provider,
                context,
                true,
            )? {
                return Ok(false);
            }
        }
    }
    Ok(relevant > 0)
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
    let mut matching = Vec::new();
    for implementation in implementations {
        if implementation_declares_selector(implementation, contract_provider, selector)? {
            matching.push(implementation);
        }
    }
    let mut requirements =
        matching
            .into_iter()
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

fn has_relevant_mlx_request_declaration(manifest: &JsonObject<String, Value>) -> Result<bool, ()> {
    let Some(backend) = manifest.get("mlx").and_then(Value::as_object) else {
        return Ok(false);
    };
    let Some(contract_value) = backend.get("memoryStrategyContract") else {
        return Ok(false);
    };
    let contract = contract_value.as_object().ok_or(())?;
    let implementations = contract
        .get("implementations")
        .and_then(Value::as_array)
        .ok_or(())?;
    Ok(implementations.iter().any(|implementation| {
        matches!(
            implementation.get("rung").and_then(Value::as_str),
            Some("bounded_transformer_residency")
        ) || (implementation.get("rung").and_then(Value::as_str) == Some("staged_residency")
            && implementation.get("requestContexts").is_some())
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

/// Does the pinned provider implement `strategy` for the candidate whose contract this is?
///
/// A contract-read ERROR is not the same fact as "the provider does not implement it", and the
/// declaration evaluator can only act on the latter. sc-22727 measured the cost of conflating them:
/// the shipped FLUX.2 Klein rehosts failed `KleinArtifactInventory::verify_for_provider` while the
/// contract was being read, the `Err` collapsed to `false` here, the BTR declaration was refused,
/// and the route silently downgraded to an eager resident load carrying NO calibration identity —
/// with nothing in the log to say an artifact had been rejected. The downgrade is still the right
/// fallback (a load that cannot prove its rung must not claim it), so this keeps returning `false`;
/// it just stops doing it silently.
fn provider_implements_declared_strategy(
    contract: gen_core::Result<Option<gen_core::MemoryProviderContract>>,
    provider: &str,
    strategy: MemoryStrategy,
) -> bool {
    match contract {
        Ok(contract) => contract.is_some_and(|contract| {
            contract
                .capability(strategy)
                .is_some_and(|capability| capability.support == MemoryStrategySupport::Implemented)
        }),
        Err(error) => {
            tracing::warn!(
                provider,
                strategy = ?strategy,
                %error,
                "memory-strategy contract read failed; treating the declared strategy as \
                 unimplemented and falling back to the eager resident load shape"
            );
            false
        }
    }
}

/// Request-exact MLX declaration intersection. Full BTR rows authorize `Applied + Deferred`;
/// exact lower-only staged rows authorize only `Eligible + Eager` under their declared load policy.
/// `Eligible` remains non-authorizing for BTR and is ignored by every legacy shaper. Krea and FLUX
/// routes use the public request coordinate plus provider-owned reference/PiD/phase axes; older
/// declarations without `requestContexts` keep their established behavior through
/// [`evaluate_declared_mlx_load_shape`].
pub fn evaluate_declared_mlx_load_shape_for_request(
    provider: &str,
    resolved_tier: Option<&str>,
    mode: Option<MemoryRouteMode>,
    manifest: &JsonObject<String, Value>,
    spec: LoadSpec,
    context: MemoryRouteRequestContext,
) -> LoadSpec {
    evaluate_declared_mlx_load_shape_for_request_with_strategy(
        provider,
        resolved_tier,
        mode,
        manifest,
        spec,
        context,
        |candidate, strategy| {
            provider_implements_declared_strategy(
                crate::inference_runtime::media().memory_strategy_contract(provider, candidate),
                provider,
                strategy,
            )
        },
    )
}

/// Bind the load policy required by the exact declaration row which just authorized an MLX
/// deferred load. This is intentionally manifest-derived: an Applied marker alone is not provider,
/// tier, mode, or profile authority, and some providers truthfully implement Deferred under a
/// Resident load policy. A single exact row may request Sequential through its staged-residency
/// prerequisite; malformed or crossed coordinates atomically become Refused + Eager rather than
/// being promoted by a provider-id allow-list or a later residency gate. The optional
/// `requiredOffloadPolicy` field is deliberately separate from `engagedRungs`: request-scoped
/// staging may be implemented under a Resident load default. When the declaration evaluator had
/// to prove BTR against a synthetic Sequential candidate, however, the exact BTR row must retain
/// that policy authority here; removing it is a terminal refusal rather than permission for a later
/// generic residency gate to rediscover Sequential independently.
pub fn apply_declared_mlx_load_policy_for_request(
    provider: &str,
    resolved_tier: Option<&str>,
    mode: Option<MemoryRouteMode>,
    manifest: &JsonObject<String, Value>,
    spec: LoadSpec,
    context: MemoryRouteRequestContext,
) -> LoadSpec {
    if spec.load_shape_declaration_result != LoadShapeDeclarationResult::Applied {
        return spec;
    }
    let (Some(tier), Some(mode), Some(load_profile)) = (
        resolved_tier.and_then(MemoryRouteTier::from_resolved_tier),
        mode,
        MemoryRouteLoadProfile::from_spec(&spec),
    ) else {
        return spec.with_refused_load_shape_declaration();
    };
    if context.mode != mode
        || context.use_pid != spec.pid.is_some()
        || spec.resolved_route.as_deref() != manifest.get("id").and_then(Value::as_str)
    {
        return spec.with_refused_load_shape_declaration();
    }
    let selector = MemoryRouteSelector {
        backend: MemoryRouteBackend::Mlx,
        provider: RULES
            .iter()
            .find(|rule| rule.backend == MemoryRouteBackend::Mlx && rule.provider == provider)
            .map(|rule| rule.provider)
            .unwrap_or(""),
        tier,
        mode,
        overlay: load_profile.overlay(),
        load_profile,
    };
    let Some(contract) = manifest_contract(manifest, MemoryRouteBackend::Mlx) else {
        return spec.with_refused_load_shape_declaration();
    };
    let exhaustive = match contract.get("exhaustive") {
        None => false,
        Some(Value::Bool(exhaustive)) => *exhaustive,
        Some(_) => return spec.with_refused_load_shape_declaration(),
    };
    let Some(contract_provider) = contract.get("provider").and_then(Value::as_str) else {
        return spec.with_refused_load_shape_declaration();
    };
    let Some(implementations) = contract.get("implementations").and_then(Value::as_array) else {
        return spec.with_refused_load_shape_declaration();
    };
    let requires_request_context = matching_rules(selector).any(|rule| !rule.legacy_shaping);
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
    let matching = implementations
        .iter()
        .zip(matches)
        .filter_map(|(implementation, matched)| matched.then_some(implementation))
        .collect::<Vec<_>>();
    let [implementation] = matching.as_slice() else {
        return spec.with_refused_load_shape_declaration();
    };
    let source_kind = match spec.weights {
        WeightsSource::Dir(_) => "dir",
        WeightsSource::File(_) => "file",
    };
    let expected_provider_overlay =
        expected_provider_overlay(selector.provider, &spec, load_profile);
    let staged_matches = implementations
        .iter()
        .map(|candidate| {
            request_strategy_implementation_matches(
                candidate,
                contract_provider,
                selector.provider,
                tier,
                load_profile,
                source_kind,
                context,
                Some(&expected_provider_overlay),
            )
        })
        .collect::<Result<Vec<_>, _>>();
    let Ok(staged_matches) = staged_matches else {
        return spec.with_refused_load_shape_declaration();
    };
    let staged = implementations
        .iter()
        .zip(staged_matches)
        .filter_map(|(candidate, matched)| matched.then_some(candidate))
        .collect::<Vec<_>>();
    let declared_policy = |implementation: &Value| match implementation.get("requiredOffloadPolicy")
    {
        None => Ok(None),
        Some(Value::String(policy)) if policy == "resident" => Ok(Some(OffloadPolicy::Resident)),
        Some(Value::String(policy)) if policy == "sequential" => {
            Ok(Some(OffloadPolicy::Sequential))
        }
        Some(_) => Err(()),
    };
    let staged_policy = match staged.as_slice() {
        [] => None,
        [staged] => match declared_policy(staged) {
            Ok(Some(policy)) => Some(policy),
            Ok(None) | Err(()) => return spec.with_refused_load_shape_declaration(),
        },
        _ => return spec.with_refused_load_shape_declaration(),
    };
    let btr_policy = match declared_policy(implementation) {
        Ok(policy) => policy,
        Err(()) => return spec.with_refused_load_shape_declaration(),
    };
    let declaration_owned = matching_rules(selector).all(|rule| !rule.legacy_shaping);
    let predicate_policy = match manifest_selector_requires_staged_residency(manifest, selector) {
        Ok(true) if declaration_owned => Some(OffloadPolicy::Sequential),
        Ok(_) => None,
        Err(()) => return spec.with_refused_load_shape_declaration(),
    };
    if btr_policy != predicate_policy
        || (staged_policy.is_some() && staged_policy != predicate_policy)
        || (exhaustive && predicate_policy.is_some() && staged_policy != predicate_policy)
    {
        return spec.with_refused_load_shape_declaration();
    }
    match btr_policy {
        Some(policy) => spec.with_offload_policy(policy),
        None => spec,
    }
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

#[cfg(test)]
pub(crate) fn evaluate_declared_mlx_load_shape_for_request_with(
    provider: &str,
    resolved_tier: Option<&str>,
    mode: Option<MemoryRouteMode>,
    manifest: &JsonObject<String, Value>,
    spec: LoadSpec,
    context: MemoryRouteRequestContext,
    provider_implements: impl FnOnce(&LoadSpec) -> bool,
) -> LoadSpec {
    evaluate_declared_mlx_load_shape_for_request_with_strategy(
        provider,
        resolved_tier,
        mode,
        manifest,
        spec,
        context,
        |candidate, _| provider_implements(candidate),
    )
}

fn evaluate_declared_mlx_load_shape_for_request_with_strategy(
    provider: &str,
    resolved_tier: Option<&str>,
    mode: Option<MemoryRouteMode>,
    manifest: &JsonObject<String, Value>,
    spec: LoadSpec,
    context: MemoryRouteRequestContext,
    provider_implements: impl FnOnce(&LoadSpec, MemoryStrategy) -> bool,
) -> LoadSpec {
    match has_relevant_mlx_request_declaration(manifest) {
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
    let btr_matches = implementations
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
    let Ok(btr_matches) = btr_matches else {
        return spec.with_refused_load_shape_declaration();
    };
    let btr_count = btr_matches.into_iter().filter(|matched| *matched).count();
    if btr_count > 1 {
        return spec.with_refused_load_shape_declaration();
    }
    let mut matching = matching_rules(selector).peekable();
    if matching.peek().is_none() {
        return spec.with_refused_load_shape_declaration();
    }
    if matching.all(|rule| rule.requires_sequential_selection) {
        return if btr_count == 1 {
            spec
        } else {
            spec.with_refused_load_shape_declaration()
        };
    }
    if !matches!(spec.weights, WeightsSource::Dir(_)) {
        return spec.with_refused_load_shape_declaration();
    }

    if btr_count == 0 {
        let source_kind = "dir";
        let expected_provider_overlay =
            expected_provider_overlay(selector.provider, &spec, load_profile);
        let staged_matches = implementations
            .iter()
            .map(|implementation| {
                request_strategy_implementation_matches(
                    implementation,
                    contract_provider,
                    selector.provider,
                    tier,
                    load_profile,
                    source_kind,
                    context,
                    Some(&expected_provider_overlay),
                )
            })
            .collect::<Result<Vec<_>, _>>();
        let Ok(staged_matches) = staged_matches else {
            return spec.with_refused_load_shape_declaration();
        };
        let matching_staged = implementations
            .iter()
            .zip(staged_matches)
            .filter_map(|(implementation, matched)| matched.then_some(implementation))
            .collect::<Vec<_>>();
        let [matching_staged] = matching_staged.as_slice() else {
            return spec.with_refused_load_shape_declaration();
        };
        if matching_staged
            .get("requiredOffloadPolicy")
            .and_then(Value::as_str)
            != Some("sequential")
        {
            return spec.with_refused_load_shape_declaration();
        }
        let staged = spec
            .clone()
            .with_eligible_load_shape_declaration()
            .with_offload_policy(OffloadPolicy::Sequential);
        return if provider_implements(&staged, MemoryStrategy::StagedResidency) {
            staged
        } else {
            spec.with_refused_load_shape_declaration()
        };
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
    if provider_implements(
        &provider_candidate,
        MemoryStrategy::BoundedTransformerResidency,
    ) {
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
    context: MemoryRouteRequestContext,
) -> Option<gen_core::MemoryProviderContract> {
    declared_candle_selector_contract_with(
        runtime_provider,
        resolved_tier,
        mode,
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

fn declared_candle_selector_contract_with(
    runtime_provider: &str,
    resolved_tier: Option<&str>,
    mode: Option<MemoryRouteMode>,
    manifest: &JsonObject<String, Value>,
    spec: &LoadSpec,
    context: MemoryRouteRequestContext,
    provider_contract: impl FnOnce(&LoadSpec) -> Option<gen_core::MemoryProviderContract>,
) -> Option<gen_core::MemoryProviderContract> {
    if spec.load_shape_declaration_result != LoadShapeDeclarationResult::Eligible {
        return None;
    }
    let mut contract = None;
    let revalidated = evaluate_declared_candle_load_shape_for_request_with(
        runtime_provider,
        resolved_tier,
        mode,
        manifest,
        spec.clone(),
        false,
        Some(context),
        |candidate| {
            contract = provider_contract(candidate);
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
    evaluate_declared_candle_load_shape_for_request_with(
        runtime_provider,
        resolved_tier,
        mode,
        manifest,
        spec,
        sequential_selected,
        None,
        provider_implements,
    )
}

#[allow(clippy::too_many_arguments)]
fn evaluate_declared_candle_load_shape_for_request_with(
    runtime_provider: &str,
    resolved_tier: Option<&str>,
    mode: Option<MemoryRouteMode>,
    manifest: &JsonObject<String, Value>,
    spec: LoadSpec,
    sequential_selected: bool,
    request_context: Option<MemoryRouteRequestContext>,
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
    if request_context.is_some_and(|context| {
        manifest_declares_selector_for_request(manifest, selector, &spec, context) != Ok(true)
    }) {
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
mod tests;

/// Provider families with worker gate semantics. This closed registry owns all family policy;
/// unknown ids have no optimized-policy fallback. Declaration-owned routes below must also have
/// an explicit production route rule before they can use the generic evidence revision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MemoryGateFamily {
    ZImage,
    Qwen,
    Flux1,
    Flux2Dev,
    Flux2Klein,
    Mage,
    Chroma,
    Ideogram,
    Kolors,
    KolorsBespoke,
    Sana,
    Sd35,
    Flux3,
    Krea,
    // The composed provider has its own overlay/load identity on every host; it is not a base DiT.
    KreaControl,
    Sdxl,
    Pulid,
}

pub(crate) fn gate_family(engine_id: &str) -> Option<MemoryGateFamily> {
    use MemoryGateFamily::*;
    Some(match engine_id {
        "z_image" | "z_image_turbo" | "z_image_control" | "z_image_turbo_control" => ZImage,
        "qwen_image" | "qwen_image_edit" => Qwen,
        "flux1_schnell" | "flux1_dev" | "flux1_dev_control" => Flux1,
        "flux2_dev" => Flux2Dev,
        "flux2_klein_9b" | "flux2_klein_9b_edit" | "flux2_klein_9b_kv_edit" => Flux2Klein,
        "mage_flow"
        | "mage_flow_base"
        | "mage_flow_turbo"
        | "mage_flow_edit"
        | "mage_flow_edit_base"
        | "mage_flow_edit_turbo" => Mage,
        "chroma1_hd" | "chroma1_base" | "chroma1_flash" => Chroma,
        "ideogram_4" | "ideogram_4_turbo" => Ideogram,
        "kolors" => Kolors,
        "candle_kolors_ipadapter" | "candle_kolors_control" => KolorsBespoke,
        "sana_1600m" | "sana_sprint_1600m" => Sana,
        "sd3_5_large" | "sd3_5_large_turbo" | "sd3_5_medium" => Sd35,
        "flux3_klein" | "flux3_klein_base" => Flux3,
        "krea_2_raw" | "krea_2_turbo" | "krea_2_edit" | "krea_2_turbo_edit" => Krea,
        "krea_2_turbo_control" => KreaControl,
        "sdxl" => Sdxl,
        "pulid_flux" => Pulid,
        _ => return None,
    })
}

pub(crate) const MLX_REQUEST_EVIDENCE_REVISION: &str = "sc-15507-request-scope-v1";

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) const SD35_REQUEST_EVIDENCE_REVISION: &str = "sd3.5-candle-request-contract-v1";

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) const SANA_REQUEST_EVIDENCE_REVISION: &str = "sana-candle-dense-request-contract-v1";

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) const KOLORS_REQUEST_EVIDENCE_REVISION: &str = "kolors-candle-request-contract-v1";

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) const SDXL_REQUEST_EVIDENCE_REVISION: &str = "sdxl-candle-request-contract-v3";

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) const DECLARATION_REQUEST_EVIDENCE_REVISION: &str =
    "sc-18456-candle-declaration-request-scope-v1";

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) const PULID_FLUX_REQUEST_EVIDENCE_REVISION: &str =
    "sc-15839-candle-pulid-flux-request-scope-v1";

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) const IDEOGRAM_REQUEST_EVIDENCE_REVISION: &str =
    "sc-20789-candle-ideogram-request-scoped-staged-residency-v1";

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) const CHROMA_REQUEST_EVIDENCE_REVISION: &str =
    "sc-20788-candle-chroma-staged-residency-v2";

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) const MAGE_FLOW_REQUEST_EVIDENCE_REVISION: &str =
    "sc-15813-candle-mage-flow-request-scope-v1";

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) const FLUX2_KLEIN_REQUEST_EVIDENCE_REVISION: &str =
    "sc-15831-candle-flux2-klein-request-scope-v1";

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) const FLUX2_DEV_REQUEST_EVIDENCE_REVISION: &str =
    "sc-15833-candle-flux2-dev-request-scope-v1";

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) const FLUX1_REQUEST_EVIDENCE_REVISION: &str = "sc-15823-candle-flux1-request-scope-v1";

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) const QWEN_IMAGE_REQUEST_EVIDENCE_REVISION: &str =
    "sc-15817-candle-qwen-image-request-scope-v1";

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) const Z_IMAGE_REQUEST_EVIDENCE_REVISION: &str =
    "sc-15815-candle-z-image-request-scope-v1";

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) fn candle_gate_evidence_revision(engine_id: &str) -> Option<&'static str> {
    use MemoryGateFamily::*;
    Some(match gate_family(engine_id) {
        Some(ZImage) => Z_IMAGE_REQUEST_EVIDENCE_REVISION,
        Some(Qwen) => QWEN_IMAGE_REQUEST_EVIDENCE_REVISION,
        Some(Flux1) if engine_id != "flux1_dev_control" => FLUX1_REQUEST_EVIDENCE_REVISION,
        Some(Flux2Dev) => FLUX2_DEV_REQUEST_EVIDENCE_REVISION,
        Some(Flux2Klein) if engine_id == "flux2_klein_9b" => FLUX2_KLEIN_REQUEST_EVIDENCE_REVISION,
        Some(Mage) => MAGE_FLOW_REQUEST_EVIDENCE_REVISION,
        Some(Chroma) => CHROMA_REQUEST_EVIDENCE_REVISION,
        Some(Ideogram) => IDEOGRAM_REQUEST_EVIDENCE_REVISION,
        Some(Kolors | KolorsBespoke) => KOLORS_REQUEST_EVIDENCE_REVISION,
        Some(Sana) => SANA_REQUEST_EVIDENCE_REVISION,
        Some(Sd35) => SD35_REQUEST_EVIDENCE_REVISION,
        Some(Sdxl) => SDXL_REQUEST_EVIDENCE_REVISION,
        Some(Pulid) => PULID_FLUX_REQUEST_EVIDENCE_REVISION,
        _ if has_gate_policy(MemoryRouteBackend::Candle, engine_id) => {
            DECLARATION_REQUEST_EVIDENCE_REVISION
        }
        _ => return None,
    })
}

#[cfg(test)]
mod gate_policy_tests;

/// Only explicit family contracts or production route rules authorize optimized gate semantics.
pub(crate) fn has_gate_policy(backend: MemoryRouteBackend, provider: &str) -> bool {
    // This explicitly injected contract is the unit selector's provider, never a production id.
    #[cfg(test)]
    if provider == "fixture_provider" {
        return true;
    }
    if gate_family(provider).is_some()
        || crate::engines::MODEL_TABLE
            .iter()
            .any(|row| row.engine_id == provider)
        || RULES
            .iter()
            .any(|rule| rule.backend == backend && rule.provider == provider)
    {
        return true;
    }
    // Composed/control providers share a registered base family's contract, while their exact
    // provider ids may have no ordinary MODEL_TABLE row or deferred RULES entry. Recognize these
    // through the pinned provider descriptors; a descriptor from an unknown family grants nothing.
    #[cfg(native_lane)]
    if let Some(descriptor) = crate::inference_runtime::media_descriptor(provider) {
        if descriptor.backend == backend.as_str() {
            return crate::engines::MODEL_TABLE
                .iter()
                .filter_map(|row| crate::inference_runtime::media_descriptor(row.engine_id))
                .any(|base| {
                    base.backend == descriptor.backend && base.family == descriptor.family
                });
        }
    }
    false
}
