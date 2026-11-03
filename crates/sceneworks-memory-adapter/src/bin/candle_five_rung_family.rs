//! CANDLE five rung capture owner.
use super::*;

/// The fixture must name the tier and geometry it measured, so a bf16 record can never be emitted
/// against a q4 capture that merely reused the fixture string.
///
/// Scoped to `krea_2_turbo` and the FLUX.1 family. Krea's plan spans several (tier, geometry) legs
/// through one adapter path — six of them, which is exactly how a mislabelled capture would arise.
/// The FLUX.1 members (sc-22726) get the stricter member/tier/edge/seed/step binding the MLX arm
/// applies ([`validate_flux_one_fixture`]), so a fixture can never name a seed the capture did not
/// render at. The Qwen legs declare a single tier and geometry each and their fixture names
/// (`qwen-image-candle-q4-seed15817-step2`) predate this convention: applying the geometry token
/// requirement to them would reject five plan rows that measure correctly today. Widen this when
/// those fixtures are renamed, not before.
/// Whether this Candle route's weights root descends into a `<tier>` sub-directory.
///
/// True for every packed SceneWorks turnkey. False for the two SANA routes (sc-22731): the
/// worker resolves the UPSTREAM dense diffusers snapshot root for them, which has no tier
/// component, and `candle-gen-sana` requires exactly that root. False, too, for Qwen-Image 2.1's
/// bf16 tier alone (sc-24112), which IS the released upstream snapshot at its root — the worker's
/// `qwen_image_2_1_declared_tier_dir` hands the engine that root, and its packed tiers are ordinary
/// `<snapshot>/<tier>` subdirs of the SceneWorks re-host.
pub(super) fn five_rung_root_is_tiered(provider_id: &str, tier: &str) -> bool {
    !matches!(provider_id, SANA_ID | SANA_SPRINT_ID)
        && !(matches!(provider_id, QWEN_IMAGE_2_1_ID | QWEN_IMAGE_2_1_TURBO_ID) && tier == "bf16")
}

/// Refuse a planned tier this LANE cannot open, by name, before any environment is read.
///
/// Only SANA constrains this today: `candle-gen-sana`'s `validate_load_spec` refuses any
/// `LoadSpec::quantize`, the worker pins the candle SANA tier to `bf16`, and no packed SANA
/// artifact ships off-Mac. A `sana_1600m:q4:candle` plan row would otherwise reach the engine and
/// come back as a quantization complaint, which reads as a spec bug rather than as a cell that
/// does not exist.
///
/// HAND-WRITTEN, and deliberately so: it is a REFUSAL that must fire before any environment read,
/// on a host with no manifest loaded, so it cannot be derived from the manifest at the point of
/// use. It is therefore a second spelling of a fact the manifest also states, and the binding that
/// keeps the two honest is on the JS side —
/// `scripts/generate-memory-matrix.test.mjs`'s "every published lane tier is one that lane's route
/// rules admit" and "no lane advertises a tier whose only downloads that lane's host would never
/// fetch". Between them, the manifest's `sana_*` off-Mac download (the untiered
/// `Efficient-Large-Model/Sana_1600M_1024px_diffusers` snapshot, `platforms: ["windows","linux"]`)
/// and `memory_route_registry.rs`'s `BF16_ONLY` candle SANA rows must keep agreeing with the
/// bf16-only rule below; publishing a packed candle SANA tier reds there, and the plan-row set
/// derived from this function in `sana_chroma_candle_tests` reds here.
pub(super) fn validate_five_rung_lane_tier(provider_id: &str, tier: &str) -> Result<(), String> {
    if matches!(provider_id, SANA_ID | SANA_SPRINT_ID) && tier != "bf16" {
        return Err(format!(
            "the Candle {provider_id} route loads the upstream dense snapshot only; there is no \
             {tier} artifact on this lane and candle-gen-sana refuses a LoadSpec quant"
        ));
    }
    Ok(())
}

/// The production calibration identity the loaded Candle generator publishes for one
/// `(provider, tier)` cell of the two families sc-22731 armed.
///
/// `candle-gen-sana` mints `sana-candle-dense-<route>-full-ladder-v1` per route (bf16 is its only
/// tier). `candle-gen-chroma` mints `<route>-<tier>-cuda-<ladder revision>` per (route, tier) from
/// the LOAD RECEIPT's tier — inference PR 951; before it, `build_contract` hard-coded
/// `calibration: None` and no Chroma anchor could be recorded on this lane at all.
///
/// `None` for every other provider: their arms predate this pre-load binding and are still checked
/// against the loaded contract in `run_five_rung_reference_loaded`.
pub(super) fn five_rung_calibration_fingerprint(provider_id: &str, tier: &str) -> Option<String> {
    match provider_id {
        SANA_ID => Some("sana-candle-dense-base-full-ladder-v1".to_owned()),
        SANA_SPRINT_ID => Some("sana-candle-dense-sprint-full-ladder-v1".to_owned()),
        CHROMA1_HD_ID | CHROMA1_BASE_ID | CHROMA1_FLASH_ID => Some(format!(
            "{}-{tier}-cuda-{CHROMA1_CANDLE_LADDER_REVISION}",
            provider_id.replace('_', "-")
        )),
        // sc-22737. `candle-gen-bernini` mints `bernini-{route}-{tier}-candle-dual-expert-ladder-v1`
        // per (provider, ARTIFACT-PROVEN tier) — the packing its `production_assets` has already
        // reconciled against the transformer's own `quantize_config.json` and tensor geometry, so a
        // dense root asked for as q4 never reaches the identity at all. Every SceneWorks load
        // resolves the `image` route (`bernini_renderer` is registered by the engine but named by
        // no load path in this repository), so the route token is constant and the tier is the only
        // axis. The capture still refuses a loaded contract whose identity differs from this, so a
        // drift is caught at the pin bump rather than inside a record.
        BERNINI_CANDLE_ID => Some(format!("bernini-image-{tier}-candle-dual-expert-ladder-v1")),
        _ => None,
    }
}

/// The five-rung fixture slug of each sc-22731 route: `fresh-five-rung-<slug>-<tier>-<edge>-seed…`.
pub(super) fn five_rung_family_slug(provider_id: &str) -> Option<&'static str> {
    match provider_id {
        SANA_ID => Some("sana-1600m"),
        SANA_SPRINT_ID => Some("sana-sprint"),
        CHROMA1_HD_ID => Some("chroma1-hd"),
        CHROMA1_BASE_ID => Some("chroma1-base"),
        CHROMA1_FLASH_ID => Some("chroma1-flash"),
        // sc-22737. `bernini-image`, not `bernini`: the slug names the CATALOG ENTRY this fixture
        // measures, and the bare `bernini` fixtures belong to the video member's own arm.
        BERNINI_CANDLE_ID => Some("bernini-image"),
        _ => None,
    }
}

/// The sc-22731 five-rung fixture binding: `fresh-five-rung-<slug>-<tier>-<edge>-seed16402-step2`.
///
/// Same claim `validate_flux_one_fixture` makes on the FLUX.1 base routes, on this family's
/// spellings — the record's fixture is the one claim about the render that nothing downstream can
/// re-derive, so it must name the route, the tier, the geometry edge, the seed and the step count.
pub(super) fn validate_five_rung_family_fixture(
    request: &Value,
    slug: &str,
    tier: &str,
) -> Result<(), String> {
    let fixture = protocol::planned(request)?
        .get("fixture")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.fixture must be a string".to_owned())?;
    let (width, _) = protocol::target_geometry(request)?;
    let prefix = format!("{FIVE_RUNG_FIXTURE_PREFIX}{slug}-{tier}-{width}-seed");
    let remainder = fixture
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("planned.fixture {fixture:?} must start with {prefix:?}"))?;
    let (seed, steps) = remainder
        .split_once("-step")
        .ok_or_else(|| format!("planned.fixture {fixture:?} must end with -step<count>"))?;
    let seed = seed
        .parse::<u64>()
        .map_err(|error| format!("parse {slug} fixture seed {seed:?}: {error}"))?;
    if seed != FIVE_RUNG_SEED {
        return Err(format!(
            "planned.fixture seed {seed} does not match the five-rung reference seed \
             {FIVE_RUNG_SEED}"
        ));
    }
    let steps = steps
        .parse::<u32>()
        .map_err(|error| format!("parse {slug} fixture step count {steps:?}: {error}"))?;
    if steps != 2 {
        return Err(format!(
            "planned.fixture {fixture:?} must use the two-step calibration request"
        ));
    }
    Ok(())
}

/// Everything one five-rung capture needs after the artifact identity is validated and the real
/// generator is resident: `(provider id, plain execution path, repository, resolved revision,
/// generator, VRAM probe already holding the load sample)`.
/// The `(env family, expected repository)` binding for one Ideogram member at one tier (sc-22732).
///
/// Ideogram is the only member of the turnkey family whose tiers do not all come from one
/// repository: `q4` and `q8` are the packed `SceneWorks/ideogram-4-mlx` turnkey, and `bf16` is the
/// separate `SceneWorks/ideogram-4` repo at a separate revision (`image_jobs/base.rs`
/// `IDEOGRAM_BF16_REPO`, and the manifest's own third `downloads[]` entry). Resolving bf16 through
/// the packed family would name the wrong repository AND the wrong revision in the record's
/// loadability fingerprint.
///
/// Written as a function returning the same six-tuple the match arms produce so the dispatch arm
/// stays an EXPRESSION with a bare `&str` const pattern: `stale-lane-report.mjs`'s
/// `adapterCapturableProviders` parses these arms to derive the capturable provider set, and it
/// accepts only a bare literal or a single `&str` const per pattern.
pub(super) fn ideogram_five_rung_family(
    provider_id: &'static str,
    execution_path: &'static str,
    tier: &str,
) -> (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
) {
    if tier == "bf16" {
        (
            provider_id,
            execution_path,
            "SCENEWORKS_IDEOGRAM_BF16_REPOSITORY",
            "SCENEWORKS_IDEOGRAM_BF16_REVISION",
            "SCENEWORKS_IDEOGRAM_BF16_ROOT",
            protocol::IDEOGRAM_BF16_REPOSITORY,
        )
    } else {
        (
            provider_id,
            execution_path,
            "SCENEWORKS_IDEOGRAM_REPOSITORY",
            "SCENEWORKS_IDEOGRAM_REVISION",
            "SCENEWORKS_IDEOGRAM_ROOT",
            protocol::IDEOGRAM_REPOSITORY,
        )
    }
}

/// The exact [`LoadSpec`] the five-rung Candle capture hands `catalog.media().load`.
///
/// Split out of [`load_five_rung_generator`] so the SHAPE of the spec is a pure function of
/// `(request, provider_id, tier, root)` and can be asserted directly (sc-22732 review). Everything
/// this binary decides about a load — offload policy, load shape, the per-family `resolved_route`,
/// and above all the per-member tier-quant fold — lives here, so a test that builds the spec is
/// testing the thing the loader actually receives rather than a table the loader happens to read.
///
/// Pure: it touches neither the filesystem nor the environment. `root` is already canonicalized and
/// validated by the caller.
pub(super) fn five_rung_load_spec(
    request: &Value,
    provider_id: &str,
    tier: &str,
    root: PathBuf,
) -> Result<LoadSpec, String> {
    let offload = if request.get("sdxlEndpoint").and_then(Value::as_str) == Some("resident") {
        OffloadPolicy::Resident
    } else {
        OffloadPolicy::Sequential
    };
    let load_shape = if request.get("sdxlEndpoint").is_some() {
        // Stock SDXL worker loads eagerly; its request-scoped phase staging does not require BTR.
        LoadShape::EagerMaterialization
    } else {
        LoadShape::DeferredMaterialization
    };
    let spec = LoadSpec::new(WeightsSource::Dir(root))
        .with_offload_policy(offload)
        .with_load_shape(load_shape);
    // sc-22731: the worker binds the exact resolved route on every Candle image load
    // (`image_jobs/base.rs`: `.with_resolved_route(request.model.clone())`), and
    // `candle-gen-chroma`'s `validate_load_shape` refuses a spec without it by name. Set for the
    // two families whose engines were ported against that shape; the older arms keep the spec they
    // have always been measured with.
    let spec = if matches!(
        provider_id,
        SANA_ID | SANA_SPRINT_ID | CHROMA1_HD_ID | CHROMA1_BASE_ID | CHROMA1_FLASH_ID
    ) {
        spec.with_resolved_route(provider_id.to_owned())
    } else {
        spec
    };
    let spec = match (provider_id, numeric_tier(tier)?.quant) {
        // Krea's loader takes the packed tier's quant explicitly; bf16 is the dense base and must
        // carry no quant at all (`Quant::None` — the same shape the worker's `tier_to_quant` uses).
        // sc-22735: Raw is the same loader on the same packed turnkeys, so it takes the same rule.
        // The worker forwards the resolved quant on this lane (`image_jobs/base.rs` keeps
        // `load_quant` off macOS), and `candle-gen-krea` treats it as recipe-only on a directory
        // load — it reads the real tier out of `transformer/config.json`. Passing it anyway is
        // what the worker does, so it is what the capture must do.
        (KREA_ID | KREA_RAW_ID, Some(quant)) => spec.with_quant(quant),
        (KREA_ID | KREA_RAW_ID, None) => spec,
        // sc-22729. SDXL is NOT a dense-TE tier: the worker's `candle_quant_for_resolved_tier`
        // falls through to `q4 -> Quant::Q4 / q8 -> Quant::Q8 / bf16 -> none` for it, and
        // `SdxlArtifactSeal::capture` hard-requires `spec.quantize == tier.quant`, so an omitted
        // quant on a packed tier refuses the seal rather than loading the packed artifact.
        (SDXL_ID, Some(quant)) => sdxl_candle_spec(request, spec.with_quant(quant))?,
        (SDXL_ID, None) => sdxl_candle_spec(request, spec)?,
        // FLUX.2 is per MEMBER (sc-22727 review): the dev route folds the planned tier the way the
        // worker's `candle_quant_for_resolved_tier` does, while both klein turnkeys are dense-TE
        // tiers the worker loads with `(None, resolved_bits)` — `candle-gen-flux2` quantizes the DiT
        // on-the-fly whenever `spec.quantize` is set, so folding it on a packed klein tier would
        // re-quantize the transformer and measure a load the app never performs. bf16 carries
        // `Quant::None` on every member, the worker's `tier_to_quant`.
        (FLUX2_DEV_ID | FLUX2_KLEIN_ID, Some(quant)) => {
            let arm =
                flux2_arm(request)?.expect("a FLUX.2 provider always resolves a member or errors");
            if arm.tier_quant_reaches_the_loader {
                spec.with_quant(quant)
            } else {
                spec
            }
        }
        (FLUX2_DEV_ID | FLUX2_KLEIN_ID, None) => spec,
        // sc-22732: the turnkey still family, PER MEMBER (`TURNKEY_CANDLE_MEMBERS`). Kolors and both
        // Lens routes take the packed tier's quant EXPLICITLY, because the worker does:
        // `candle_quant_for_resolved_tier` (`image_jobs/base.rs`) carves out sana, dense-TE tiers,
        // chroma, sd3.5 and the two FLUX.1 routes to `None`, none of these declares
        // `mlx.denseTextEncoderTier`, and every descriptor advertises `supported_quants: [Q4, Q8]`.
        // `candle-gen-kolors` treats it as an advisory no-op on the packed tier; `candle-gen-lens`
        // proves the artifact tier from disk and withholds its identity unless `spec.quantize`
        // EQUALS it, so binding `None` on a packed Lens tier would withhold the identity this cell
        // is measured under. The two Ideogram routes are the exception: `candle-gen-ideogram`'s
        // exact directory route (`validate_load_shape`, reached from the production loader through
        // `IdeogramLoadReceipt::capture`) REFUSES `quantize: Some(_)` outright and proves the tier
        // off the packed safetensors headers instead, so the quant never reaches that loader.
        // bf16 is the dense base and carries no quant on any member.
        (
            KOLORS_ID
            | IDEOGRAM_ID
            | IDEOGRAM_TURBO_ID
            | LENS_ID
            | LENS_TURBO_ID
            | QWEN_IMAGE_2_1_ID
            | QWEN_IMAGE_2_1_TURBO_ID,
            Some(quant),
        ) => {
            if turnkey_candle_member(provider_id)
                .expect("a turnkey provider always resolves a member")
                .tier_quant_reaches_the_loader
            {
                spec.with_quant(quant)
            } else {
                spec
            }
        }
        (
            KOLORS_ID
            | IDEOGRAM_ID
            | IDEOGRAM_TURBO_ID
            | LENS_ID
            | LENS_TURBO_ID
            | QWEN_IMAGE_2_1_ID
            | QWEN_IMAGE_2_1_TURBO_ID,
            None,
        ) => spec,
        // Qwen, Z-Image-Turbo and the Z-Image base packed Diffusers snapshots declare their
        // device-format quantization in transformer/config.json (`snapshot_quant_tier` in
        // candle-gen-z-image's memory_strategy.rs). Passing LoadSpec.quant would request a second,
        // unsupported runtime quantization pass — every one of those loaders rejects it by name —
        // instead of loading the packed artifact as authored.
        //
        // sc-22730: the SD3.5 turnkeys are packed the same way, and `candle-gen-sd3`'s
        // `validate_load_shape` refuses `LoadSpec::quantize` on them OUTRIGHT — a request knob can
        // never outrank the artifact. `Sd35LoadReceipt::capture` then reads the transformer's
        // packing off the safetensors headers and cross-checks it against the path tier, so the
        // tier in the published identity is the tier on disk. Falling through here is therefore
        // what the worker does, not an omission.
        //
        // sc-22731 puts SANA and Chroma1 in the same class, from the worker itself:
        // `candle_quant_for_resolved_tier` returns `(None, _)` for both families at EVERY tier, so
        // `LoadSpec::quantize` is `None` on every shipped Candle render of them. Both engines
        // refuse a quant by name (`"Candle supports only the dense physical tier"`,
        // `"turnkey q4/q8/bf16 all require precision=Bf16 and LoadSpec.quantize=None"`), and
        // Chroma's tier comes from the artifact path plus the transformer's own packed marker.
        //
        // sc-22738: the STILL Bernini entry takes the worker's exact still-lane spec — eager, the
        // packed tier's quant bound, `bernini_image` as the resolved route — see
        // `bernini_candle_load_spec`. The `Deferred` default above is what refused every
        // `bernini_image` cell its contract ("Bernini Candle memory contract requires
        // EagerMaterialization") before the quant was even consulted.
        (BERNINI_CANDLE_ID, _) => {
            bernini_candle_load_spec(spec, tier, BERNINI_CANDLE_IMAGE_MODEL_ID)?
        }
        _ => spec,
    };
    // The catalog model id reaches the engine as `resolved_route` — the same lever the worker
    // sets (`image_jobs/base.rs`, `spec.with_resolved_route(request.model)`), and the only thing
    // that distinguishes two catalog models sharing one registry id (sc-22727).
    let spec = match flux2_arm(request)? {
        Some(arm) => spec.with_resolved_route(arm.model_id),
        // sc-22732: the worker sets it on EVERY candle load, and `candle-gen-ideogram`'s exact
        // directory route refuses a spec whose `resolved_route` is not its own id
        // (`validate_load_shape`), so the turnkey members carry it too.
        None if turnkey_candle_member(provider_id).is_some() => {
            spec.with_resolved_route(provider_id)
        }
        None => spec,
    };
    // sc-24112: Qwen-Image 2.1 is declaration-owned with no load-shape declaration on either lane
    // (`memory_route_registry.rs` rule `legacy_shaping: false`, and the manifest entry carries no
    // `memoryStrategyContract`), so the worker's `apply_declared_candle_image_load_shape` leaves the
    // gen-core default — `EagerMaterialization` — in place, and `candle-gen-qwen-image-2-1` has no
    // block-streaming loader to defer into. The anchor is measured under the shape production loads
    // and the calibration identity it publishes, exactly as Bernini's still leg is.
    // sc-25907: the Turbo id is the same crate under the same worker rule.
    let spec = if matches!(provider_id, QWEN_IMAGE_2_1_ID | QWEN_IMAGE_2_1_TURBO_ID) {
        spec.with_load_shape(LoadShape::EagerMaterialization)
    } else {
        spec
    };
    Ok(spec)
}

pub(super) fn load_five_rung_generator(request: &Value) -> Result<LoadedFiveRungGenerator, String> {
    // sc-22733: Mage binds two artifact triples and stages two components; it has its own loader.
    if let Some(arm) = mage_arm(request)? {
        return load_mage_generator(request, arm);
    }
    // Resolved BEFORE the family match: Ideogram's bf16 tier is a different repository, so the tier
    // is an input to the binding rather than something checked after it.
    let tier = planned_tier(request)?;
    let (provider_id, execution_path, repository_env, revision_env, root_env, expected_repository) =
        match planned_provider(request)? {
            "qwen_image" => (
                QWEN_ID,
                QWEN_PLAIN_EXECUTION_PATH,
                "SCENEWORKS_QWEN_IMAGE_REPOSITORY",
                "SCENEWORKS_QWEN_IMAGE_REVISION",
                "SCENEWORKS_QWEN_IMAGE_ROOT",
                protocol::QWEN_REPOSITORY,
            ),
            "krea_2_turbo" => (
                KREA_ID,
                KREA_PLAIN_EXECUTION_PATH,
                "SCENEWORKS_KREA_REPOSITORY",
                "SCENEWORKS_KREA_REVISION",
                "SCENEWORKS_KREA_ROOT",
                protocol::KREA_REPOSITORY,
            ),
            // sc-22735. Raw's own tiered rehost, bound through its own env family so a Raw plan
            // can never be satisfied by Turbo weights and re-label the distilled model's peaks as
            // the true-CFG base's.
            "krea_2_raw" => (
                KREA_RAW_ID,
                KREA_RAW_PLAIN_EXECUTION_PATH,
                "SCENEWORKS_KREA_RAW_REPOSITORY",
                "SCENEWORKS_KREA_RAW_REVISION",
                "SCENEWORKS_KREA_RAW_ROOT",
                protocol::KREA_RAW_REPOSITORY,
            ),
            // sc-15859. The artifact family is `SceneWorks/z-image-turbo-mlx` (`Z_IMAGE_REPOSITORY`),
            // the same per-tier `q4/ q8/ bf16/` re-host the MLX arm measures, so the env family is
            // `SCENEWORKS_Z_IMAGE_*` on both adapters (docs/calibration-runbook.md, "Adapter
            // environment").
            "z_image_turbo" => (
                Z_IMAGE_TURBO_ID,
                // The edit route (`z_image_edit`) loads the same Turbo provider from the same
                // artifact; only the generation request and the admitted mode differ.
                plain_execution_path(request)?,
                "SCENEWORKS_Z_IMAGE_REPOSITORY",
                "SCENEWORKS_Z_IMAGE_REVISION",
                "SCENEWORKS_Z_IMAGE_ROOT",
                protocol::Z_IMAGE_REPOSITORY,
            ),
            // sc-22724. The base model's own rehost, bound through its own env family so a base
            // plan can never be satisfied by Turbo weights.
            "z_image" => (
                Z_IMAGE_ID,
                Z_IMAGE_PLAIN_EXECUTION_PATH,
                "SCENEWORKS_Z_IMAGE_BASE_REPOSITORY",
                "SCENEWORKS_Z_IMAGE_BASE_REVISION",
                "SCENEWORKS_Z_IMAGE_BASE_ROOT",
                protocol::Z_IMAGE_BASE_REPOSITORY,
            ),
            // sc-22727. Each FLUX.2 catalog model binds its OWN artifact family, so a KV plan can
            // never be satisfied by the base klein rehost even though both load through
            // `flux2_klein_9b`.
            FLUX2_DEV_ID | FLUX2_KLEIN_ID => {
                let arm = flux2_arm(request)?
                    .expect("a FLUX.2 provider always resolves a member or errors");
                (
                    arm.provider,
                    arm.execution_path,
                    arm.repository_env,
                    arm.revision_env,
                    arm.root_env,
                    arm.expected_repository,
                )
            }
            // sc-22726. PuLID is a BESPOKE route: `candle-gen-pulid` registers no `Generator`, so
            // there is nothing for `catalog.media().load` to return and reaching here at all means
            // the dispatch in `run` was bypassed. Named rather than left to the catch-all so the
            // refusal says which arm owns it — and so the derived capturability report
            // (`stale-lane-report.mjs#adapterCapturableProviders`, which intersects every dispatch
            // gate carrying the refusal phrase) does not read this lane as having no arm.
            // Expression-bodied ON PURPOSE: a block-bodied arm carries no trailing comma after
            // rustfmt, and the report's arm parser splits on depth-0 commas — a braced arm here
            // silently swallowed the NEXT arm's pattern and dropped `flux1_dev` from the derived
            // capturable set.
            "pulid_flux" => return Err(
                "pulid_flux is a bespoke Candle route and must not reach the provider registry; \
                 it is served by run_pulid_flux_capture"
                    .to_owned(),
            ),
            // Each FLUX.1 base provider binds its OWN tiered rehost, so a schnell plan can never be
            // satisfied by dev weights.
            "flux1_dev" => (
                FLUX1_DEV_ID,
                FLUX1_DEV_PLAIN_EXECUTION_PATH,
                "SCENEWORKS_FLUX1_DEV_REPOSITORY",
                "SCENEWORKS_FLUX1_DEV_REVISION",
                "SCENEWORKS_FLUX1_DEV_ROOT",
                protocol::FLUX1_DEV_REPOSITORY,
            ),
            "flux1_schnell" => (
                FLUX1_SCHNELL_ID,
                FLUX1_SCHNELL_PLAIN_EXECUTION_PATH,
                "SCENEWORKS_FLUX1_SCHNELL_REPOSITORY",
                "SCENEWORKS_FLUX1_SCHNELL_REVISION",
                "SCENEWORKS_FLUX1_SCHNELL_ROOT",
                protocol::FLUX1_SCHNELL_REPOSITORY,
            ),
            // sc-22729. Five catalog models share this registry id; the member — and therefore the
            // artifact family, the env triple and the fingerprint the engine mints — is named by
            // `planned.target.modelId`, and bound onto the spec as `resolved_route` below.
            SDXL_ID => {
                let arm = sdxl_candle_arm(request)?;
                (
                    SDXL_ID,
                    arm.execution_path,
                    arm.repository_env,
                    arm.revision_env,
                    arm.root_env,
                    arm.expected_repository,
                )
            }
            // sc-22733. Each Mage variant binds TWO artifact triples (its own tiered rehost plus
            // the shared text-encoder/VAE components snapshot) and stages two components, which
            // the single-root tuple above cannot express — so the early return at the top of this
            // function serves them and reaching here means that return was bypassed. Named rather
            // than left to the catch-all for the same two reasons `pulid_flux` is: the refusal must
            // say which loader owns the route, and `adapterCapturableProviders` INTERSECTS every
            // dispatch gate carrying the refusal phrase, so an absent arm here would erase all six
            // from the derived capturable set. Expression-bodied ON PURPOSE (see `pulid_flux`).
            "mage_flow" => return Err(mage_registry_bypass("mage_flow")),
            "mage_flow_base" => return Err(mage_registry_bypass("mage_flow_base")),
            "mage_flow_turbo" => return Err(mage_registry_bypass("mage_flow_turbo")),
            "mage_flow_edit" => return Err(mage_registry_bypass("mage_flow_edit")),
            "mage_flow_edit_base" => return Err(mage_registry_bypass("mage_flow_edit_base")),
            "mage_flow_edit_turbo" => return Err(mage_registry_bypass("mage_flow_edit_turbo")),
            // sc-22730. Each SD3.5 member binds its OWN tiered rehost, so a Medium plan can never
            // be satisfied by Large weights. The rehost is the SAME artifact family both lanes
            // load (`SceneWorks/sd3.5-<route>-mlx`, per-tier `q4/ q8/ bf16/` subdirs), so the env
            // family is `SCENEWORKS_SD3_5_<ROUTE>_*` on both adapters.
            SD3_5_LARGE_ID => (
                SD3_5_LARGE_ID,
                SD3_5_LARGE_PLAIN_EXECUTION_PATH,
                "SCENEWORKS_SD3_5_LARGE_REPOSITORY",
                "SCENEWORKS_SD3_5_LARGE_REVISION",
                "SCENEWORKS_SD3_5_LARGE_ROOT",
                protocol::SD3_5_LARGE_REPOSITORY,
            ),
            SD3_5_LARGE_TURBO_ID => (
                SD3_5_LARGE_TURBO_ID,
                SD3_5_LARGE_TURBO_PLAIN_EXECUTION_PATH,
                "SCENEWORKS_SD3_5_LARGE_TURBO_REPOSITORY",
                "SCENEWORKS_SD3_5_LARGE_TURBO_REVISION",
                "SCENEWORKS_SD3_5_LARGE_TURBO_ROOT",
                protocol::SD3_5_LARGE_TURBO_REPOSITORY,
            ),
            SD3_5_MEDIUM_ID => (
                SD3_5_MEDIUM_ID,
                SD3_5_MEDIUM_PLAIN_EXECUTION_PATH,
                "SCENEWORKS_SD3_5_MEDIUM_REPOSITORY",
                "SCENEWORKS_SD3_5_MEDIUM_REVISION",
                "SCENEWORKS_SD3_5_MEDIUM_ROOT",
                protocol::SD3_5_MEDIUM_REPOSITORY,
            ),
            // sc-22731. The SANA routes bind the UPSTREAM DENSE diffusers snapshot, not the
            // SceneWorks MLX turnkey: `resolve_weights_dir` returns
            // `huggingface_pinned_snapshot_dir(SANA_CANDLE_DIFFUSERS_REPO, …)` for this lane, whose
            // root has no tier component at all. Their own env families keep a base plan from
            // being satisfied by Sprint weights.
            "sana_1600m" => (
                SANA_ID,
                SANA_PLAIN_EXECUTION_PATH,
                "SCENEWORKS_SANA_DENSE_REPOSITORY",
                "SCENEWORKS_SANA_DENSE_REVISION",
                "SCENEWORKS_SANA_DENSE_ROOT",
                protocol::SANA_DENSE_REPOSITORY,
            ),
            "sana_sprint_1600m" => (
                SANA_SPRINT_ID,
                SANA_SPRINT_PLAIN_EXECUTION_PATH,
                "SCENEWORKS_SANA_SPRINT_DENSE_REPOSITORY",
                "SCENEWORKS_SANA_SPRINT_DENSE_REVISION",
                "SCENEWORKS_SANA_SPRINT_DENSE_ROOT",
                protocol::SANA_SPRINT_DENSE_REPOSITORY,
            ),
            // sc-22731. Each Chroma1 route binds its OWN rehost — `candle-gen-chroma`'s `ROUTES`
            // table pins one repository and revision per provider and refuses a root bound to
            // another, so an HD plan can never be satisfied by Flash weights.
            "chroma1_hd" => (
                CHROMA1_HD_ID,
                CHROMA1_HD_PLAIN_EXECUTION_PATH,
                "SCENEWORKS_CHROMA1_HD_REPOSITORY",
                "SCENEWORKS_CHROMA1_HD_REVISION",
                "SCENEWORKS_CHROMA1_HD_ROOT",
                protocol::CHROMA1_HD_REPOSITORY,
            ),
            "chroma1_base" => (
                CHROMA1_BASE_ID,
                CHROMA1_BASE_PLAIN_EXECUTION_PATH,
                "SCENEWORKS_CHROMA1_BASE_REPOSITORY",
                "SCENEWORKS_CHROMA1_BASE_REVISION",
                "SCENEWORKS_CHROMA1_BASE_ROOT",
                protocol::CHROMA1_BASE_REPOSITORY,
            ),
            "chroma1_flash" => (
                CHROMA1_FLASH_ID,
                CHROMA1_FLASH_PLAIN_EXECUTION_PATH,
                "SCENEWORKS_CHROMA1_FLASH_REPOSITORY",
                "SCENEWORKS_CHROMA1_FLASH_REVISION",
                "SCENEWORKS_CHROMA1_FLASH_ROOT",
                protocol::CHROMA1_FLASH_REPOSITORY,
            ),
            // sc-22737. The STILL Bernini entry. Its artifact is the CANDLE rehost
            // `SceneWorks/bernini` — a different repository from the macOS `SceneWorks/bernini-mlx`
            // one — which the manifest ships untiered at the download level while carrying `q4/`,
            // `q8/` and `bf16/` subtrees inside the snapshot, so the load root is still the tier
            // directory (`five_rung_root_is_tiered` is true here). The video member never reaches
            // this loader: it is dispatched above the still gate by `sc22737_video_arm`.
            BERNINI_CANDLE_ID => {
                bernini_candle_still_member(request)?;
                (
                    BERNINI_CANDLE_ID,
                    BERNINI_CANDLE_IMAGE_EXECUTION_PATH,
                    "SCENEWORKS_BERNINI_CANDLE_REPOSITORY",
                    "SCENEWORKS_BERNINI_CANDLE_REVISION",
                    "SCENEWORKS_BERNINI_CANDLE_ROOT",
                    protocol::BERNINI_CANDLE_REPOSITORY,
                )
            }
            // sc-22732: the turnkey still family. Each member binds its own artifact family, so a
            // Lens-Turbo plan can never be satisfied by base Lens weights; the two Ideogram members
            // share one packed repo AND one bf16 repo, and differ by provider.
            KOLORS_ID => (
                KOLORS_ID,
                KOLORS_PLAIN_EXECUTION_PATH,
                "SCENEWORKS_KOLORS_REPOSITORY",
                "SCENEWORKS_KOLORS_REVISION",
                "SCENEWORKS_KOLORS_ROOT",
                protocol::KOLORS_REPOSITORY,
            ),
            IDEOGRAM_ID => {
                ideogram_five_rung_family(IDEOGRAM_ID, IDEOGRAM_PLAIN_EXECUTION_PATH, tier)
            }
            IDEOGRAM_TURBO_ID => ideogram_five_rung_family(
                IDEOGRAM_TURBO_ID,
                IDEOGRAM_TURBO_PLAIN_EXECUTION_PATH,
                tier,
            ),
            LENS_ID => (
                LENS_ID,
                LENS_PLAIN_EXECUTION_PATH,
                "SCENEWORKS_LENS_REPOSITORY",
                "SCENEWORKS_LENS_REVISION",
                "SCENEWORKS_LENS_ROOT",
                protocol::LENS_REPOSITORY,
            ),
            LENS_TURBO_ID => (
                LENS_TURBO_ID,
                LENS_TURBO_PLAIN_EXECUTION_PATH,
                "SCENEWORKS_LENS_TURBO_REPOSITORY",
                "SCENEWORKS_LENS_TURBO_REVISION",
                "SCENEWORKS_LENS_TURBO_ROOT",
                protocol::LENS_TURBO_REPOSITORY,
            ),
            // sc-24112: Qwen-Image 2.1 binds the re-host for q4/q8 and the upstream snapshot for
            // bf16, each through its own env family.
            QWEN_IMAGE_2_1_ID => qwen_image_2_1_five_rung_family(tier),
            // sc-25907: Turbo binds its OWN re-host and upstream snapshot the same way.
            QWEN_IMAGE_2_1_TURBO_ID => qwen_image_2_1_turbo_five_rung_family(tier),
            provider => {
                return Err(format!(
                    "Candle five-rung calibration does not implement provider {provider:?}"
                ))
            }
        };
    validate_fixture_binds_tier_and_geometry(request)?;
    if turnkey_candle_member(provider_id).is_some() {
        validate_turnkey_identity(request, provider_id, tier)?;
    }
    validate_planned_fingerprint_is_producible(provider_id, tier, request)?;
    validate_five_rung_lane_tier(provider_id, tier)?;
    // The plan row must name the identity the LOADED contract publishes for this cell — checked
    // against the weights-free table BEFORE the load, so a stale row fails in milliseconds instead
    // of after a multi-gigabyte load (`run_five_rung_reference_loaded` still re-checks it against
    // the real contract afterwards, which is what keeps the two copies from drifting).
    if let Some(expected) = five_rung_calibration_fingerprint(provider_id, tier) {
        let planned = protocol::planned(request)?
            .get("calibrationFingerprint")
            .and_then(Value::as_str)
            .ok_or_else(|| "planned.calibrationFingerprint must be a string".to_owned())?;
        if planned != expected {
            return Err(format!(
                "plan/provider calibration mismatch: plan={planned}, the {provider_id} {tier} \
                 production identity is {expected}"
            ));
        }
    }
    let repository = protocol::required_env(repository_env)?;
    let revision = protocol::required_env(revision_env)?;
    protocol::validate_artifact_identity(&repository, &revision, expected_repository)?;
    let root = std::fs::canonicalize(PathBuf::from(protocol::required_env(root_env)?))
        .map_err(|error| format!("canonicalize {root_env}: {error}"))?;
    if five_rung_root_is_tiered(provider_id, tier) {
        // The root must end in the PLANNED tier's directory, so a stale `…/q4` export cannot satisfy a
        // q8 or bf16 plan and quietly re-label another tier's peaks.
        protocol::validate_huggingface_snapshot_root(
            &root,
            &repository,
            &revision,
            tier,
            expected_repository,
        )?;
    } else {
        // The SANA dense snapshot (and Qwen-Image 2.1's bf16 upstream snapshot) has no tier
        // sub-directory: the worker hands the engine the snapshot root itself, and
        // `candle-gen-sana`'s `validate_immutable_root` requires exactly that. Inventing a `bf16/`
        // component here would bind a path no production load opens.
        protocol::validate_huggingface_revision_root(
            &root,
            &repository,
            &revision,
            expected_repository,
        )?;
    }
    let spec = five_rung_load_spec(request, provider_id, tier, root)?;
    // sc-22726: the FLUX.1 base snapshots declare their packed tier the same way; the directory
    // name proved nothing about the weights, so read the tier off the transformer config before
    // paying for the load.
    if matches!(provider_id, FLUX1_DEV_ID | FLUX1_SCHNELL_ID) {
        validate_flux_one_snapshot_tier(&spec, provider_id, tier)?;
    }
    // sc-22730: `candle-gen-sd3`'s `validate_load_shape` REFUSES a spec whose `resolved_route` is
    // not exactly this provider id — it is the first thing every SD3.5 production load checks
    // (`Sd35LoadReceipt::capture`), and the worker sets it on every render
    // (`image_jobs/base.rs` `.with_resolved_route(request.model.clone())`). No arm on this lane set
    // it before, because no provider on this lane required it; without it all nine SD3.5 candle
    // cells would be refused at load rather than measured. Scoped to the family that demands it so
    // no other provider's spec shape moves.
    let spec = if matches!(
        provider_id,
        SD3_5_LARGE_ID | SD3_5_LARGE_TURBO_ID | SD3_5_MEDIUM_ID
    ) {
        spec.with_resolved_route(provider_id)
    } else {
        spec
    };
    let catalog =
        runtime_cuda::catalog().map_err(|error| format!("build CUDA catalog: {error}"))?;
    let mut vram = if request.get("sdxlEndpoint").is_some() {
        endpoint_vram_probe()?
    } else {
        certifying_vram_probe()
    };
    if request.get("sdxlEndpoint").is_some() {
        // Match the ordinary worker's advanced.flashAttn default before loading SDXL.
        runtime_cuda::providers::sdxl::set_flash_attn(true);
    }
    let load_sample = vram.phase();
    let generator = catalog
        .media()
        .load(provider_id, &spec)
        .map_err(|error| format!("load real {provider_id} {tier} generator: {error}"))?;
    vram.end_load(load_sample);
    Ok((
        provider_id,
        execution_path,
        repository,
        revision,
        generator,
        vram,
    ))
}

pub(super) fn run_five_rung_reference_loaded(
    request: &Value,
    provider_id: &str,
    execution_path: &str,
    generator: &dyn runtime_cuda::gen_core::Generator,
    vram: &mut VramProbe,
    repository: &str,
    revision: &str,
) -> Result<Value, String> {
    protocol::validate_plain_overlay_target(request, execution_path)?;
    protocol::validate_still_geometry(request, still_calibration_label(request)?)?;
    let contract = generator.memory_strategy_contract().ok_or_else(|| {
        format!(
            "loaded {provider_id} has {}",
            missing_contract_reason(generator, request)
        )
    })?;
    // sc-22738: the tier the WORKER selects for this loaded provider — planned quant plus the
    // provider's declared component floors — not a bare `(precision, quant)`.
    let selection = planned_selection_for(request, generator.descriptor())?;
    contract.validate_selection(&selection).map_err(|error| {
        format!("pinned {provider_id} provider rejected planned selection: {error}")
    })?;
    let strategy = measured_strategy(
        request,
        &selection,
        &contract.engaged_composition_for_selection(&selection),
    )?;
    let calibration = contract
        .calibration
        .as_ref()
        .ok_or_else(|| format!("pinned {provider_id} provider has no calibration identity"))?;
    let planned_fingerprint = protocol::planned(request)?
        .get("calibrationFingerprint")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.calibrationFingerprint must be a string".to_owned())?;
    if planned_fingerprint != calibration.fingerprint {
        return Err(format!(
            "plan/provider calibration mismatch: plan={planned_fingerprint}, pinned provider={}",
            calibration.fingerprint
        ));
    }
    let planned_load_shape = protocol::planned(request)?
        .get("loadShape")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.loadShape must be a string".to_owned())?;
    let actual_load_shape = match calibration.load_shape {
        LoadShape::EagerMaterialization => "eager_materialization",
        LoadShape::DeferredMaterialization => "deferred_materialization",
    };
    if planned_load_shape != actual_load_shape {
        return Err(format!(
            "plan/provider load-shape mismatch: plan={planned_load_shape}, pinned provider={actual_load_shape}"
        ));
    }
    let (width, height) = protocol::target_geometry(request)?;
    let hardware_bytes = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.memoryBytes must be an integer".to_owned())?;
    // sc-22724: the mode the plan declared, as the worker would admit it. The `z_image_edit`
    // route carries one reference and is admitted under `MemoryMode::Edit` (the contract the
    // `z_image_edit` manifest entry declares for the Turbo provider).
    // sc-22733: the three Mage instruction editors are admitted under `MemoryMode::Edit` too — the
    // engine's own route gate requires it (`memory_strategy.rs` `route_is_supported`).
    let mage = mage_arm(request)?;
    let edit = is_z_image_edit(request)? || mage.is_some_and(|arm| arm.edit);
    let context = MemoryRunContext {
        selection,
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
        calibration_abi: calibration.abi,
        calibration_fingerprint: calibration.fingerprint.clone(),
        load_shape: calibration.load_shape,
        mode: if edit {
            MemoryMode::Edit
        } else {
            MemoryMode::TextToImage
        },
        has_reference: edit,
        use_pid: false,
        has_phases: false,
        geometry: MemoryGeometry {
            width,
            height,
            batch: 1,
            frames: 1,
            reference_count: u32::from(edit),
        },
        reference_resources: None,
        overlay: None,
        budget: MemoryBudget {
            total_bytes: hardware_bytes,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes: 1,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: five_rung_evidence_revision(provider_id),
    };
    // sc-22738: the worker's admission sequence — safety check, then begin. `candle-gen-chroma`,
    // `candle-gen-flux2` and `candle-gen-mage` refuse a `begin` with no approval on file.
    let mut scope = open_memory_request_scope(
        generator,
        &context,
        &format!("{provider_id} fresh-reference"),
    )?;
    let parameters = context.selection.parameters;
    match (parameters.decode_tile_edge, parameters.decode_overlap) {
        (Some(edge), Some(overlap)) => scope
            .configure_decode(edge, overlap, context.geometry)
            .map_err(|error| format!("configure {provider_id} fresh-reference decode: {error}"))?,
        (None, None) => {}
        _ => {
            return Err(format!(
                "{provider_id} decode edge and overlap must be selected together"
            ))
        }
    }
    if let Some(attention) = parameters.attention_chunk_size {
        scope.configure_attention(attention).map_err(|error| {
            format!("configure {provider_id} fresh-reference attention: {error}")
        })?;
    }
    if let Some(window) = parameters.transformer_window_size {
        scope
            .materialize_transformer_window(0, window)
            .map_err(|error| {
                format!("configure {provider_id} fresh-reference transformer: {error}")
            })?;
    }
    let mut generation = match mage {
        Some(arm) => {
            mage_generation_request(arm, &generator.descriptor().capabilities, width, height)
        }
        None => five_rung_provider_generation_request(provider_id, width, height, edit),
    };
    if request.get("sdxlEndpoint").is_some() {
        // The issue's ordinary request uses the provider's default sampler and steps.
        generation.steps = Some(30);
        generation.guidance = Some(7.0);
    }
    scope
        .configure_request(&mut generation)
        .map_err(|error| format!("apply {provider_id} fresh-reference strategy: {error}"))?;
    scope
        .enter_phase(MemoryPhase::Conditioning)
        .map_err(|error| format!("enter {provider_id} fresh-reference conditioning: {error}"))?;
    let generation_sample = vram.phase();
    let mut phase_sample = Some(vram.phase());
    let mut phase = MemoryPhase::Conditioning;
    let mut conditioning_peak_gb = None;
    let mut denoise_peak_gb = None;
    let mut decode_peak_gb = None;
    let mut phase_error = None;
    let result = generator.generate(&generation, &mut |progress| {
        if phase_error.is_some() {
            return;
        }
        let boundary = match progress {
            Progress::Loading(runtime_cuda::gen_core::LoadPhase::Renderer) => {
                protocol::ReferenceBoundary::RendererLoad
            }
            Progress::Step { current: 1, .. } => protocol::ReferenceBoundary::FirstDenoiseStep,
            Progress::Decoding => protocol::ReferenceBoundary::Decoding,
            _ => return,
        };
        let Some(next) = protocol::next_reference_phase(reference_phase(phase), boundary) else {
            return;
        };
        let peak = phase_sample.take().map(|sample| vram.end_observed(sample));
        match phase {
            MemoryPhase::Conditioning => conditioning_peak_gb = peak,
            MemoryPhase::Denoise => denoise_peak_gb = peak,
            MemoryPhase::Decode => decode_peak_gb = peak,
        }
        if let Err(error) = scope.leave_phase(phase) {
            phase_error = Some(format!("leave {provider_id} {phase:?}: {error}"));
            return;
        }
        let next = memory_phase(next);
        if let Err(error) = scope.enter_phase(next) {
            phase_error = Some(format!("enter {provider_id} {next:?}: {error}"));
            return;
        }
        phase = next;
        phase_sample = Some(vram.phase());
    });
    if let Some(sample) = phase_sample.take() {
        let terminal_peak_gb = vram.end_observed(sample);
        match phase {
            MemoryPhase::Conditioning => conditioning_peak_gb = Some(terminal_peak_gb),
            MemoryPhase::Denoise => denoise_peak_gb = Some(terminal_peak_gb),
            MemoryPhase::Decode => decode_peak_gb = Some(terminal_peak_gb),
        }
    }
    vram.end_gen(generation_sample);
    if let Some(message) = phase_error {
        let _ = scope.finish(MemoryRunOutcome::Error {
            message: message.clone(),
        });
        return Err(message);
    }
    let mut endpoint_output = Value::Null;
    match result {
        Ok(runtime_cuda::gen_core::GenerationOutput::Images(images)) if images.len() == 1 => {
            if request.get("sdxlEndpoint").is_some() {
                endpoint_output = retain_sdxl_endpoint_output(request, &images[0])?;
            }
        }
        Ok(runtime_cuda::gen_core::GenerationOutput::Images(images)) => {
            return Err(format!(
                "{provider_id} fresh reference returned {} images",
                images.len()
            ));
        }
        Ok(_) => {
            return Err(format!(
                "{provider_id} fresh reference returned non-image output"
            ))
        }
        Err(error) => {
            let message = error.to_string();
            let _ = scope.finish(MemoryRunOutcome::Error {
                message: message.clone(),
            });
            return Err(format!(
                "{provider_id} fresh-reference generation failed: {message}"
            ));
        }
    }
    scope
        .leave_phase(phase)
        .map_err(|error| format!("leave {provider_id} fresh-reference terminal phase: {error}"))?;
    scope
        .finish(MemoryRunOutcome::Complete)
        .map_err(|error| format!("finish {provider_id} fresh-reference scope: {error}"))?;
    let conditioning_bytes = decimal_gb_to_bytes(conditioning_peak_gb.ok_or_else(|| {
        format!("{provider_id} fresh reference did not expose conditioning boundary")
    })?);
    let denoise_bytes =
        decimal_gb_to_bytes(denoise_peak_gb.ok_or_else(|| {
            format!("{provider_id} fresh reference did not expose denoise boundary")
        })?);
    let decode_bytes = decimal_gb_to_bytes(
        decode_peak_gb
            .ok_or_else(|| format!("{provider_id} fresh reference did not complete decode"))?,
    );
    let overall_bytes = if request.get("sdxlEndpoint").is_some() {
        decimal_gb_to_bytes(vram.report().peak_gb)
    } else {
        conditioning_bytes.max(denoise_bytes).max(decode_bytes)
    };
    let blocker = if provider_id == QWEN_ID {
        concat!(
            "SC-15817 five-rung conformance measures exact per-rung memory, strategy identity, ",
            "and loadability; it intentionally remains gated because this run does not repeat ",
            "each sibling story's promotion-quality, negative-mutation, and lifecycle suite"
        )
    } else if provider_id == Z_IMAGE_TURBO_ID {
        concat!(
            "SC-15859 anchor capture measures exact per-phase memory and strategy identity for ",
            "the Candle Z-Image-Turbo lane; it intentionally remains gated because this run does ",
            "not repeat the full promotion-quality, negative-mutation, and lifecycle scenario suite"
        )
    } else if provider_id == Z_IMAGE_ID {
        concat!(
            "sc-22724 anchor capture measures exact per-phase memory and strategy identity for ",
            "the Candle Z-Image base lane; it intentionally remains gated because this run does ",
            "not repeat the full promotion-quality, negative-mutation, and lifecycle scenario suite"
        )
    } else if provider_id == FLUX2_DEV_ID || provider_id == FLUX2_KLEIN_ID {
        concat!(
            "sc-22727 anchor capture measures exact per-phase memory and strategy identity for ",
            "the Candle FLUX.2 lanes; it intentionally remains gated because this run does not ",
            "repeat the full promotion-quality, negative-mutation, and lifecycle scenario suite"
        )
    } else if provider_id == FLUX1_DEV_ID {
        concat!(
            "sc-22726 anchor capture measures exact per-phase memory and strategy identity for ",
            "the Candle FLUX.1-dev lane; it intentionally remains gated because this run does ",
            "not repeat the full promotion-quality, negative-mutation, and lifecycle scenario suite"
        )
    } else if provider_id == FLUX1_SCHNELL_ID {
        concat!(
            "sc-22726 anchor capture measures exact per-phase memory and strategy identity for ",
            "the Candle FLUX.1-schnell lane; it intentionally remains gated because this run does ",
            "not repeat the full promotion-quality, negative-mutation, and lifecycle scenario suite"
        )
    } else if MAGE_ARMS.iter().any(|arm| arm.provider == provider_id) {
        concat!(
            "sc-22733 anchor capture measures exact per-phase memory and strategy identity for ",
            "the Candle Mage-Flow lane; it intentionally remains gated because this run does not ",
            "repeat the full promotion-quality, negative-mutation, and lifecycle scenario suite"
        )
    } else if matches!(
        provider_id,
        SD3_5_LARGE_ID | SD3_5_LARGE_TURBO_ID | SD3_5_MEDIUM_ID
    ) {
        concat!(
            "sc-22730 anchor capture measures exact per-phase memory and strategy identity for ",
            "the Candle SD3.5 lane; it intentionally remains gated because this run does not ",
            "repeat the full promotion-quality, negative-mutation, and lifecycle scenario suite"
        )
    } else {
        concat!(
            "five-rung oracle capture measures exact per-rung memory and strategy identity for ",
            "SC-16059; it intentionally remains gated because this run does not repeat the full ",
            "promotion-quality, negative-mutation, and lifecycle scenario suite"
        )
    };
    // stageResidency names a discrete composition, not a numeric sweep axis. Keep its boolean
    // in the executed case while the existing sweep schema describes numeric knobs only.
    let sweep = if request.get("sdxlEndpoint").is_some() {
        let mut numeric_request = request.clone();
        numeric_request["planned"]["strategy"]["parameters"]
            .as_object_mut()
            .ok_or("missing endpoint strategy parameters")?
            .remove("stageResidency");
        let mut sweep = protocol::reference_sweep(&numeric_request, "passed")?;
        sweep["cases"][0]["parameters"] = json!(protocol::strategy_parameters(request)?);
        sweep
    } else {
        protocol::reference_sweep(request, "passed")?
    };
    let mut fragment = protocol::plain_gated_fragment(
        request,
        execution_path,
        protocol::PlainGatedFragment {
            artifact: artifact(repository, revision, planned_tier(request)?),
            sweep,
            blocker,
            quality: json!({ "result": "not_run" }),
            negative_mutation: Value::Null,
            loadability: json!({
                "result": "passed",
                "resolvedPathFingerprint": loadability_fingerprint(
                    repository,
                    revision,
                    planned_tier(request)?,
                ),
            }),
            diagnostics: protocol::diagnostics(
                // sc-22724: `provider_id` is `z_image_turbo` for BOTH the text-to-image and the
                // edit capture, so the route has to be in the source or the two records are
                // indistinguishable by their own diagnostics. Mirrors `ZImageArm::slug` on MLX.
                // `provider_id` is `z_image_turbo` for BOTH the Turbo text-to-image and the edit
                // capture (sc-22724), and `flux2_klein_9b` for BOTH klein catalog models
                // (sc-22727), so the route has to be in the source or the records are
                // indistinguishable by their own diagnostics. Mirrors the MLX arms' slugs.
                &match flux2_arm(request)? {
                    Some(arm) => format!("memory-candle-adapter:{}-five-rung-reference", arm.slug),
                    None => format!(
                        "memory-candle-adapter:{provider_id}{}-five-rung-reference",
                        if edit { "-edit" } else { "" }
                    ),
                },
                "executed",
                [blocker.to_owned()],
                [
                    ("conditioningDevicePeakDelta", "bytes", conditioning_bytes),
                    ("denoiseDevicePeakDelta", "bytes", denoise_bytes),
                    ("decodeDevicePeakDelta", "bytes", decode_bytes),
                    ("overallDevicePeakDelta", "bytes", overall_bytes),
                    // The edit route conditions every request on one reference image.
                    ("referenceImages", "count", u64::from(edit)),
                ],
            ),
        },
    )?;
    fragment["strategy"] = strategy;
    fragment["loadShape"] = json!(load_shape_key(calibration.load_shape));
    fragment["observedMemory"] = json!({
        "conditioning": cuda_phase_metrics(conditioning_bytes),
        "denoise": cuda_phase_metrics(denoise_bytes),
        "decode": cuda_phase_metrics(decode_bytes),
        "overall": cuda_phase_metrics(overall_bytes),
    });
    if request.get("sdxlEndpoint").is_some() {
        // The runner hashes this exact staged tier inventory before invoking the adapter. Keep
        // that content identity in the immutable evidence record, not only its companion summary.
        let inventory_sha256 = protocol::required_env("SCENEWORKS_MEMORY_MODEL_INVENTORY_SHA256")?;
        if inventory_sha256.len() != 64
            || !inventory_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err("SDXL endpoint requires the captured artifact inventory SHA-256".into());
        }
        fragment["artifact"]["inventorySha256"] = json!(inventory_sha256);
        let report = vram.report();
        fragment["endpointMeasurement"] = json!({
            "inferencePin": protocol::INFERENCE_PIN,
            "engineBuild": {
                "adapterCandleFeature": cfg!(feature = "candle"),
                "cudaComputeCapabilityCompileTarget": option_env!("CUDA_COMPUTE_CAP"),
                // sc-9032 removed the no-op feature; the pinned provider components() selects
                // false unconditionally. The runner checks that source policy before capture.
                "sdxlFlashAttentionKernelCompiled": false,
            },
            "output": endpoint_output,
            "loadPeakBytes": decimal_gb_to_bytes(report.load_peak_gb),
            "baselineBytes": decimal_gb_to_bytes(report.baseline_gb),
            "wholeRunPeakBytes": decimal_gb_to_bytes(report.peak_gb),
            "wholeDevicePeakBytes": decimal_gb_to_bytes(report.baseline_gb + report.peak_gb),
            "offloadPolicy": if request.get("sdxlEndpoint").and_then(Value::as_str) == Some("resident") { "resident" } else { "sequential" },
            "settings": {
                "prompt": generation.prompt, "negativePrompt": generation.negative_prompt,
                "seed": generation.seed, "steps": generation.steps,
                "sampler": generation.sampler, "scheduler": generation.scheduler,
                "guidance": generation.guidance, "memory": format!("{:?}", generation.memory),
                "flashAttn": runtime_cuda::providers::sdxl::flash_attn_enabled(),
                "count": generation.count, "cfgInterval": generation.cfg_interval,
                "guidanceMethod": generation.guidance_method,
                "schedulerShift": generation.scheduler_shift,
            },
        });
    }
    Ok(fragment)
}

/// The one fresh planned request every five-rung reference renders. Two steps are intentional:
/// a resident image provider has no loading boundary between text encode and denoise, so the
/// first Step callback closes a conservative conditioning envelope and the second gives denoise
/// its own measured interval before Decoding. The edit route (sc-22724) is the worker's edit
/// request — one `Conditioning::Reference` fitted to the request geometry plus the strength
/// lever (`resolve_zimage_edit_init`) — with the step count raised so the engine-derived start
/// step (`floor(steps * strength)`) still leaves two executed denoise steps.
pub(super) fn five_rung_evidence_revision(provider: &str) -> String {
    if matches!(provider, SANA_ID | SANA_SPRINT_ID) {
        runtime_cuda::providers::sana::memory_strategy::REQUEST_EVIDENCE_REVISION.to_owned()
    } else if matches!(
        provider,
        SD3_5_LARGE_ID | SD3_5_LARGE_TURBO_ID | SD3_5_MEDIUM_ID
    ) {
        runtime_cuda::providers::sd3::memory_strategy::REQUEST_EVIDENCE_REVISION.to_owned()
    } else if provider == KOLORS_ID {
        runtime_cuda::providers::kolors::memory_strategy::REQUEST_EVIDENCE_REVISION.to_owned()
    } else {
        format!(
            "{}@{}",
            five_rung_evidence_story(provider),
            protocol::INFERENCE_PIN
        )
    }
}

pub(super) fn five_rung_provider_generation_request(
    provider: &str,
    width: u32,
    height: u32,
    edit: bool,
) -> GenerationRequest {
    let mut generation = five_rung_generation_request(width, height, edit);
    if provider == BERNINI_CANDLE_ID {
        // Bernini's shared video provider requires an explicit single-frame mode.
        generation.video_mode = Some(if edit { "i2i" } else { "t2i" }.to_owned());
        generation.frames = Some(1);
    }
    generation
}

pub(super) fn five_rung_generation_request(
    width: u32,
    height: u32,
    edit: bool,
) -> GenerationRequest {
    let mut generation = GenerationRequest {
        prompt: "a photorealistic red apple on a wooden table, studio lighting".to_owned(),
        width,
        height,
        count: 1,
        seed: Some(FIVE_RUNG_SEED),
        steps: Some(2),
        ..Default::default()
    };
    if edit {
        generation.steps = Some(Z_IMAGE_EDIT_STEPS);
        // `request.strength` stays None: the worker sets ONLY the per-reference strength
        // (`build_lane_conditioning`, image_jobs/base.rs:7136) and leaves the request-level lever —
        // gen-core's documented fallback for a single `Reference` with no strength of its own —
        // unset. This arm reproduces the worker's request shape, so it does the same (sc-22724).
        generation.conditioning = vec![Conditioning::Reference {
            image: Image {
                width,
                height,
                pixels: protocol::synthetic_reference_rgb(width, height),
            },
            strength: Some(Z_IMAGE_EDIT_STRENGTH),
        }];
    }
    generation
}

/// The story whose evidence a five-rung reference record cites in `evidence_revision`: the
/// story that gave the provider its arm on this lane. Krea/Qwen/Z-Image keep the SC-16402 tag
/// their packaged records already carry; the FLUX.1 members (sc-22726) and the SD3.5 members
/// (sc-22730) cite their own.
pub(super) fn five_rung_evidence_story(provider_id: &str) -> &'static str {
    if let Some(arm) = MAGE_ARMS.iter().find(|arm| arm.provider == provider_id) {
        return arm.evidence_story;
    }
    match provider_id {
        FLUX1_DEV_ID | FLUX1_SCHNELL_ID => "sc-22726",
        KOLORS_ID | IDEOGRAM_ID | IDEOGRAM_TURBO_ID | LENS_ID | LENS_TURBO_ID => "sc-22732",
        QWEN_IMAGE_2_1_ID => "sc-24112",
        QWEN_IMAGE_2_1_TURBO_ID => "sc-25907",
        SD3_5_LARGE_ID | SD3_5_LARGE_TURBO_ID | SD3_5_MEDIUM_ID => "sc-22730",
        BERNINI_CANDLE_ID => "sc-22737",
        _ => "sc-16402",
    }
}

pub(super) fn run_five_rung_reference(request: &Value) -> Result<Value, String> {
    let execution_path = plain_execution_path(request)?;
    protocol::validate_plain_overlay_target(request, execution_path)?;
    // Before `load_five_rung_generator`, for the same reason the overlay check is duplicated here:
    // a geometry this arm cannot honour must be refused before it costs a real weight load.
    protocol::validate_still_geometry(request, still_calibration_label(request)?)?;
    let (provider_id, execution_path, repository, revision, generator, mut vram) =
        load_five_rung_generator(request)?;
    let mut fragment = run_five_rung_reference_loaded(
        request,
        provider_id,
        execution_path,
        generator.as_ref(),
        &mut vram,
        &repository,
        &revision,
    )?;
    // sc-22733: a Mage load opens TWO artifact triples, and the shared loadability fingerprint
    // names only the variant's. A load that took the DiT from the planned variant but the text
    // encoder from another tier's components would otherwise leave no trace in the record.
    if mage_arm(request)?.is_some() {
        let tier = planned_tier(request)?;
        let components_repository =
            protocol::required_env("SCENEWORKS_MAGE_FLOW_COMPONENTS_REPOSITORY")?;
        let components_revision =
            protocol::required_env("SCENEWORKS_MAGE_FLOW_COMPONENTS_REVISION")?;
        let variant = fragment
            .pointer("/loadability/resolvedPathFingerprint")
            .and_then(Value::as_str)
            .ok_or_else(|| "the five-rung fragment carries no loadability fingerprint".to_owned())?
            .to_owned();
        fragment["loadability"]["resolvedPathFingerprint"] = json!(format!(
            "{variant}+{components_repository}@{components_revision}:{tier}"
        ));
    }
    Ok(fragment)
}

pub(super) fn run_five_rung_batch(request: &Value) -> Result<Value, String> {
    let planned = request
        .get("planned")
        .and_then(Value::as_array)
        .ok_or_else(|| "run_batch request.planned must be an array".to_owned())?;
    let expected_rungs = [
        "resident",
        "staged_residency",
        "bounded_decode",
        "bounded_attention",
        "bounded_transformer_residency",
    ];
    let actual_rungs = planned
        .iter()
        .map(|item| {
            item.pointer("/strategy/rung")
                .and_then(Value::as_str)
                .ok_or_else(|| "batched planned strategy.rung must be a string".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if actual_rungs != expected_rungs {
        return Err(format!(
            "run_batch requires one canonical five-rung target, got {actual_rungs:?}"
        ));
    }
    let first_target = planned[0]
        .get("target")
        .ok_or_else(|| "batched planned target is missing".to_owned())?;
    if planned
        .iter()
        .any(|item| item.get("target") != Some(first_target))
    {
        return Err("run_batch cannot mix calibration targets in one model load".to_owned());
    }
    for item in planned {
        let mut per_rung_request = request.clone();
        per_rung_request["action"] = json!("run");
        per_rung_request["planned"] = item.clone();
        let execution_path = plain_execution_path(&per_rung_request)?;
        protocol::validate_plain_overlay_target(&per_rung_request, execution_path)?;
        protocol::validate_still_geometry(
            &per_rung_request,
            still_calibration_label(&per_rung_request)?,
        )?;
    }

    let mut first_request = request.clone();
    first_request["action"] = json!("run");
    first_request["planned"] = planned[0].clone();
    let (provider_id, execution_path, repository, revision, generator, mut vram) =
        load_five_rung_generator(&first_request)?;
    let smi = NvidiaSmi::resolve()?;
    // Krea uses DeferredMaterialization, so loading the generator does not establish its
    // steady-state device residency. The canonical batch starts with `resident`; use the
    // memory retained after that first rung as the contamination baseline, then require every
    // later rung to release its transient allocations back to that warmed state.
    let mut settled_after_resident = None;
    let mut fragments = Vec::with_capacity(planned.len());
    for item in planned {
        let mut per_rung_request = request.clone();
        per_rung_request["action"] = json!("run");
        per_rung_request["planned"] = item.clone();
        fragments.push(run_five_rung_reference_loaded(
            &per_rung_request,
            provider_id,
            execution_path,
            generator.as_ref(),
            &mut vram,
            &repository,
            &revision,
        )?);
        let after = smi.used_bytes()?;
        update_warmed_retention_baseline(&mut settled_after_resident, after)?;
    }
    // sc-22667 (epic 22657 E6): the LOADED provider's component bytes, architecture facts and
    // published rung parameters ride along with the fragments, so the derived side of the
    // falsification prices the five rungs from exactly the contract this measurement ran under
    // rather than a hand-constructed twin. Additive: the harness reads `fragments` only.
    let provider_contract = generator
        .memory_strategy_contract()
        .map(loaded_contract_facts)
        .ok_or_else(|| format!("loaded {provider_id} has no memory-strategy contract"))?;
    Ok(json!({
        "modelLoads": 1,
        "fragments": fragments,
        "providerContract": provider_contract,
    }))
}

/// Which of [`run`]'s two branches a plan row takes: the five-rung reference path, or the inline
/// Krea arm.
///
/// Named rather than inlined so the decision is testable on its own (sc-18808 re-review). It is what
/// determines which arm [`run`]'s geometry guard is standing in front of, and every case in the
/// original regression table happened to answer `true` — so the inline arm, and with it `run`'s own
/// guard, went unexercised while the redundant copy at the head of [`run_five_rung_reference`]
/// produced the byte-identical message. Five shipped Candle plan rows answer `false`
/// (`krea-q4-1024-seed42` and its q8/bf16/768/v2 siblings).
pub(super) fn routes_to_five_rung_reference(request: &Value) -> Result<bool, String> {
    let is_five_rung_fixture = protocol::planned(request)?
        .get("fixture")
        .and_then(Value::as_str)
        .is_some_and(|fixture| fixture.starts_with(FIVE_RUNG_FIXTURE_PREFIX));
    // Qwen, Z-Image-Turbo and the Z-Image base have no inline arm at all, so every fixture on
    // them is a five-rung reference capture regardless of its spelling.
    let provider = planned_provider(request)?;
    // sc-22726: the FLUX.1 BASE providers have no inline arm either. PuLID is deliberately absent —
    // it is a bespoke route with its own arm, dispatched before this is ever consulted.
    // sc-22733: the six Mage routes have no inline arm either — they ride the shared five-rung
    // lifecycle behind their own loader (`load_mage_generator`) and their own request shape.
    // sc-22730: the three SD3.5 base providers have no inline arm either.
    Ok(is_five_rung_fixture
        || provider == QWEN_ID
        || provider == Z_IMAGE_TURBO_ID
        || provider == Z_IMAGE_ID
        // sc-22729: the whole SDXL family has no inline arm on this lane either — every member is
        // a five-rung reference capture through the registered `sdxl` generator.
        || provider == SDXL_ID
        // sc-22727: neither FLUX.2 provider has an inline arm on this adapter either.
        || provider == FLUX2_DEV_ID
        || provider == FLUX2_KLEIN_ID
        || provider == FLUX1_DEV_ID
        || provider == FLUX1_SCHNELL_ID
        // sc-22732: none of the five turnkey members has an inline arm either.
        || provider == KOLORS_ID
        || provider == IDEOGRAM_ID
        || provider == IDEOGRAM_TURBO_ID
        || provider == LENS_ID
        || provider == LENS_TURBO_ID
        // sc-24112: nor does Qwen-Image 2.1, and sc-25907: nor its Turbo sibling.
        || provider == QWEN_IMAGE_2_1_ID
        || provider == QWEN_IMAGE_2_1_TURBO_ID
        || MAGE_ARMS.iter().any(|arm| arm.provider == provider)
        || provider == SD3_5_LARGE_ID
        || provider == SD3_5_LARGE_TURBO_ID
        || provider == SD3_5_MEDIUM_ID
        // sc-22737: the STILL Bernini entry has no inline arm either. The VIDEO entry never
        // reaches this function — `sc22737_video_arm` dispatches it above the still gate — so
        // naming the shared engine provider id here claims only the still member.
        || provider == BERNINI_CANDLE_ID
        // sc-22735: the undistilled Krea base has no inline arm either — unlike Turbo, which keeps
        // its own certifying arm below. Named here as well as matched by fixture prefix, so a
        // future Raw fixture spelled differently still routes to the arm that implements it
        // instead of falling into Turbo's `provider != KREA_ID` refusal.
        || provider == KREA_RAW_ID)
}
