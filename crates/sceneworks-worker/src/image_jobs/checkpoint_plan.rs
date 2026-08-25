// Plan-driven image route (epic 20398, sc-20634): the universal-import walking skeleton.
//
// A user image model whose manifest entry carries `importPlan.checkpointId` is backed by a
// persisted `ImportPlanV1` in the app's checkpoint plan store (`<data>/checkpoints/`), compiled from
// a linked library root by `sceneworks_core::checkpoint_plan_store`. This route:
//
//   1. resolves and re-verifies the plan (record ↔ plan agreement, approved root present, every
//      layer's bytes still the bytes the plan was compiled from) — every refusal is a typed
//      `CheckpointPlanError` surfaced with its stable `[checkpoint-plan:<code>]` prefix, raised
//      BEFORE any loader is constructed;
//   2. resolves the provider by family + source shape + operation through the live provider
//      registry's imported-model authority (`imported_model_descriptor`), so an unknown family or
//      an unbound backend/operation refuses with a typed message rather than falling through;
//   3. builds an ordinary `LoadSpec` from the plan layers (primary `WeightsSource::File`, plus the
//      provider's declared required components), pins every file, and renders through the SAME
//      cached-generator seam every other lane uses. On the inference side the provider reads the
//      file through the mapped logical-weight reader and the registered codec table.
//
// Scope: `Generate` (text-to-image) for a single-file transformer plan. Edit / pose / multi-phase
// / reference shapes are not served here — they keep their family lanes until sc-20644 moves each
// family's full surface onto this route. The claim is therefore PER REQUEST, not per entry: this
// route takes the shapes it serves, and a managed entry's bespoke family lane keeps the rest, so
// importing a model under managed ownership never REMOVES a capability it had. Exactly one lane
// owns each REQUEST. A plan-backed request no lane claims at all is a typed refusal, never the
// procedural stub (see `CheckpointPlanSelection::into_unclaimed_refusal`).

use sceneworks_core::checkpoint_import::CheckpointContainerV1;
use sceneworks_core::checkpoint_plan_store::{
    CheckpointPlanError, CheckpointPlanStore, ResolvedCheckpointV1, ResolvedLayerV1,
};

/// The adapter/engine id recorded on assets rendered through the plan-driven route.
#[cfg(target_os = "macos")]
const CHECKPOINT_PLAN_ENGINE: &str = "mlx_checkpoint_plan";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CHECKPOINT_PLAN_ENGINE: &str = "candle_checkpoint_plan";

/// Plan layer roles the route maps to a provider source shape.
///
/// These are the inspector's role vocabulary (`checkpoint_inspector::infer_role_from_path` and
/// `base_weights::ComponentRole::as_str`), not a family list: which of them a given checkpoint
/// carries, and which source shape that implies, is decided by the family's registered
/// [`gen_core::CheckpointAdapterRegistration`], never by a branch on the family name.
const CHECKPOINT_PLAN_TRANSFORMER_ROLE: &str = "transformer";
const CHECKPOINT_PLAN_FUSED_ROLE: &str = "checkpoint";

/// The portable checkpoint-adapter authority for this plan's family, or a typed refusal.
///
/// This is the seam that makes a family's truth PLAN truth (E2/E5): eligible backends, the dialect
/// source shapes, the component topology and the per-operation capability policy are all read from
/// the adapter the provider crate registered, so adding a family is a registration rather than a
/// worker edit. A family with no registered adapter on this backend refuses here, before any loader
/// is constructed (E7).
fn checkpoint_plan_adapter(
    family: &str,
    checkpoint_id: &str,
) -> WorkerResult<&'static gen_core::CheckpointAdapterRegistration> {
    crate::inference_runtime::checkpoint_adapter(family).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "[checkpoint-plan:no-adapter-binding] checkpoint {checkpoint_id:?}: this runtime \
             registers no {family:?} checkpoint adapter, so its plan cannot be loaded on this \
             backend"
        ))
    })
}

/// Whether the family's own adapter declares this build's backend eligible.
///
/// Backend eligibility is ADAPTER truth, not an inference from "did a provider happen to register a
/// binding": Z-Image declares Candle only and Mage-Flow declares MLX only, and a request for the
/// other backend must say so with the adapter's own word rather than dying as a missing route.
fn checkpoint_plan_backend_eligible(
    adapter: &gen_core::CheckpointAdapterRegistration,
    checkpoint_id: &str,
) -> WorkerResult<()> {
    if adapter
        .eligible_backends
        .contains(&crate::inference_runtime::CHECKPOINT_BACKEND)
    {
        return Ok(());
    }
    Err(WorkerError::InvalidPayload(format!(
        "[checkpoint-plan:backend-ineligible] checkpoint {checkpoint_id:?} ({} family) declares \
         eligible backends {:?}; this build binds {:?}, so the family is not runnable here",
        adapter.family,
        adapter.eligible_backends,
        crate::inference_runtime::CHECKPOINT_BACKEND
    )))
}

/// The one on-disk source shape this family's registered dialects describe.
///
/// Read from [`gen_core::CheckpointDialectRegistration::source`] rather than inferred from which
/// roles the plan happens to carry: the dialect table is the adapter's declaration of what its
/// loader opens (a single transformer file, a fused checkpoint, a component directory, a ComfyUI
/// tree), and a plan whose roles could be read two ways must not be silently resolved one of them.
///
/// A family whose dialects disagree refuses rather than guessing. Nothing shipped today declares
/// two shapes; the refusal exists so that the day one does, it is a planning-time diagnostic and not
/// a load-time surprise.
fn checkpoint_plan_source_shape(
    adapter: &gen_core::CheckpointAdapterRegistration,
    checkpoint_id: &str,
) -> WorkerResult<gen_core::ImportedModelSource> {
    let mut shapes: Vec<gen_core::ImportedModelSource> =
        adapter.dialects.iter().map(|dialect| dialect.source).collect();
    shapes.sort();
    shapes.dedup();
    match shapes.as_slice() {
        [shape] => Ok(*shape),
        [] => Err(WorkerError::InvalidPayload(format!(
            "[checkpoint-plan:no-adapter-binding] checkpoint {checkpoint_id:?} ({} family): the \
             registered adapter declares no dialects, so no source shape can be resolved",
            adapter.family
        ))),
        many => Err(WorkerError::InvalidPayload(format!(
            "[checkpoint-plan:ambiguous-component] checkpoint {checkpoint_id:?} ({} family): the \
             registered adapter declares {} distinct dialect source shapes ({many:?}); a compiled \
             plan records component roles, not a dialect id, so the shape cannot be resolved",
            adapter.family,
            many.len()
        ))),
    }
}

/// The persisted checkpoint id a manifest entry is bound to, when it is plan-backed. The same
/// reader the scheduler's imported claim uses (`jobs_store::checkpoint_plan_checkpoint_id`), so
/// admission and the worker agree on what "plan-backed" means.
pub(crate) fn checkpoint_plan_checkpoint_id(entry: &JsonObject) -> Option<&str> {
    sceneworks_core::jobs_store::checkpoint_plan_checkpoint_id(entry)
}

/// The registry operation this request asks for.
///
/// The same four-way discrimination every imported lane makes, in the same precedence order: an
/// explicit phase list is the finest-grained control and wins over everything; a strict-pose set
/// outside edit mode is the control surface; `edit_image` is the edit surface; everything else is
/// generation (plain text-to-image and reference-guided img2img both live there, because a
/// reference-guided init is the Generate descriptor's `Reference` conditioning, not a second
/// operation).
fn checkpoint_plan_operation(request: &ImageRequest) -> gen_core::ImportedModelOperation {
    if request_has_multiphase(request) {
        gen_core::ImportedModelOperation::MultiPhase
    } else if !pose_entries(request).is_empty() && request.mode != "edit_image" {
        gen_core::ImportedModelOperation::Pose
    } else if request.mode == "edit_image" {
        gen_core::ImportedModelOperation::Edit
    } else {
        gen_core::ImportedModelOperation::Generate
    }
}

/// Whether THIS build binds a provider for the entry's family at the request's operation.
///
/// The per-request claim discriminator's registry half, and the reason the claim can widen family by
/// family without a family list here: a family whose row has landed binds every operation its
/// adapter declares, so the plan route takes its whole surface; a family still on its bespoke lane
/// binds only the operations that lane already delegated, so that lane keeps the rest.
///
/// Reads the entry's DECLARED family rather than the plan's, because this predicate must answer
/// without opening the plan store. The two disagreeing is a corrupt/edited entry, and the
/// conservative answer there is "the plan route owns it" — it then refuses with the plan's own
/// family, which is strictly better than handing a mislabelled entry to a family lane.
fn checkpoint_plan_binds_request_operation(request: &ImageRequest) -> bool {
    let Some(family) = request
        .model_manifest_entry
        .get("family")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|family| !family.is_empty())
    else {
        return true;
    };
    let Some(adapter) = crate::inference_runtime::checkpoint_adapter(family) else {
        return true;
    };
    let Ok(source) = checkpoint_plan_source_shape(adapter, "") else {
        return true;
    };
    crate::inference_runtime::imported_model_descriptor(
        family,
        source,
        checkpoint_plan_operation(request),
    )
    .is_some()
}

/// Every request axis the SELECTED provider cannot execute, as a typed refusal — or `None` when the
/// exact registered descriptor advertises everything this request asks for.
///
/// This is the generalization of what each bespoke imported lane used to answer with its own
/// hand-written predicate, and every gate below is read from the resolved
/// [`gen_core::ModelDescriptor`] rather than from a per-family constant:
///
/// * **adapters** — `supports_lora` / `supports_lokr`, which the registry already forces to `false`
///   for a binding declaring `inherit_adapters: false` (a full fine-tune whose moved base weights
///   cannot safely take an adapter fitted to the published checkpoint). So "may this family take a
///   LoRA on this source shape" is adapter truth, not a lane's opinion.
/// * **conditioning** — `Reference` / `MultiReference` / `Mask` / `Control` membership decides
///   img2img, the edit surface's reference count, masked edit, and strict pose.
/// * **operation** — the descriptor only exists because the binding for this exact
///   (family, source, operation) exists, so reaching here means the operation is bound.
///
/// The axes that are NOT descriptor-expressible stay explicit and fail closed: an identity
/// (`character_id` / `character_look_id`) needs asset-resolution paths this route does not own, and
/// Hires.fix on a pose set has no lane because the pose loop renders one image per pose and never
/// reads `hires_fix` — admitting it would return a successful image that silently ignored the
/// request (E8).
fn checkpoint_plan_request_shape_refusal(
    request: &ImageRequest,
    descriptor: &gen_core::ModelDescriptor,
    operation: gen_core::ImportedModelOperation,
    checkpoint_id: &str,
) -> Option<String> {
    let caps = &descriptor.capabilities;
    let refuse = |detail: &str| {
        Some(format!(
            "[checkpoint-plan:unsupported-operation] checkpoint {checkpoint_id:?}: the registered \
             {:?} provider {:?} for this checkpoint's source shape {detail}",
            operation, descriptor.id
        ))
    };
    if imported_model_quant(request, descriptor, "Checkpoint plan").is_err() {
        return refuse("does not accept the requested quantization tier");
    }
    if request.character_id.is_some() || request.character_look_id.is_some() {
        return refuse(
            "cannot render a character or look identity, which needs base-tier identity components \
             the plan-driven route does not stage",
        );
    }
    if !request.loras.is_empty() && !(caps.supports_lora || caps.supports_lokr) {
        return refuse("does not accept LoRA/LoKr adapters");
    }
    if non_empty(&request.mask_asset_id)
        && !caps.conditioning.contains(&gen_core::ConditioningKind::Mask)
    {
        return refuse("does not accept a mask");
    }
    // Material strict-control intent that did NOT resolve to the Pose operation (a pose set inside
    // edit mode, or a control payload the pose-mode reader rejects) has no lane anywhere.
    if sceneworks_core::jobs_store::imported_control_intent_is_material(&request.advanced)
        && operation != gen_core::ImportedModelOperation::Pose
    {
        return refuse("cannot combine strict control intent with this operation");
    }
    match operation {
        gen_core::ImportedModelOperation::MultiPhase => {
            // Multi-phase renders one trajectory from pure noise: every conditioning field would be
            // silently dropped by the phase driver, so each one is refused rather than ignored.
            if request.mode == "edit_image"
                || !pose_entries(request).is_empty()
                || !request.reference_asset_ids.is_empty()
                || request.reference_asset_id.is_some()
                || request.source_asset_id.is_some()
            {
                return refuse("renders a multi-phase trajectory from noise and takes no reference, \
                     source, pose, or edit conditioning");
            }
        }
        gen_core::ImportedModelOperation::Pose => {
            if !sceneworks_core::jobs_store::imported_pose_control_mode_is_supported(
                &request.advanced,
            ) {
                return refuse("does not support the requested pose control mode");
            }
            if !caps
                .conditioning
                .contains(&gen_core::ConditioningKind::Control)
            {
                return refuse("does not accept control conditioning");
            }
            if request.hires_fix.enabled {
                return refuse(
                    "renders one image per pose and never applies Hires.fix, so a Hires.fix pose \
                     request would silently drop the refinement",
                );
            }
            // The plural edit set and a bare `sourceAssetId` are read by neither the pose resolver
            // nor the pose render loop; a single `referenceAssetId` is the likeness source.
            if !request.reference_asset_ids.is_empty() || request.source_asset_id.is_some() {
                return refuse(
                    "reads a single reference as the pose likeness source and would drop a plural \
                     reference set or a bare source asset",
                );
            }
        }
        gen_core::ImportedModelOperation::Edit => {
            let has_edit_reference = !request.reference_asset_ids.is_empty()
                || non_empty(&request.reference_asset_id)
                || non_empty(&request.source_asset_id);
            if !has_edit_reference {
                return refuse("requires a source or reference image to edit");
            }
            if !caps
                .conditioning
                .contains(&gen_core::ConditioningKind::Reference)
            {
                return refuse("does not accept reference conditioning");
            }
            if request.reference_asset_ids.len() > 1
                && !caps
                    .conditioning
                    .contains(&gen_core::ConditioningKind::MultiReference)
            {
                return refuse("accepts a single edit reference, not a plural reference set");
            }
        }
        gen_core::ImportedModelOperation::Generate => {
            // img2img rides a single `referenceAssetId`. The plural edit set and a bare
            // `sourceAssetId` are not read by the generate conditioning resolver, so admitting
            // either would silently render plain text-to-image.
            if !request.reference_asset_ids.is_empty() || request.source_asset_id.is_some() {
                return refuse(
                    "reads a single reference as the img2img init and would drop a plural \
                     reference set or a bare source asset",
                );
            }
            if request.reference_asset_id.is_some()
                && !caps
                    .conditioning
                    .contains(&gen_core::ConditioningKind::Reference)
            {
                return refuse("does not accept a reference-guided init");
            }
        }
    }
    None
}

/// Plan families whose bespoke lane can load a PLAN-BACKED entry — the migration state of sc-20644,
/// one row per family whose parity row has landed.
///
/// Not family truth, and deliberately not derived from anything: it is the answer to "has this
/// family's lane been taught to take its bytes from the plan yet", which is a fact about THIS
/// repository's migration, true of a different set of families each week. A family in this list has
/// a bespoke lane for a LINKED checkpoint (it sources the plan's verified layer); a family not in it
/// has one only for a MANAGED install, which still has `paths.model` for the lane to scan.
///
/// sc-20651 **retains** this table, and the sc-20644 note that said it would be deleted "together
/// with the lanes it names" was wrong: the deletion step proved the table must OUTLIVE the lanes.
/// Its sole reader is [`checkpoint_plan_shape_has_other_lane`], so removing it collapses
/// [`checkpoint_plan_unservable_shape`] into [`checkpoint_plan_unservable`] — and every LINKED
/// checkpoint (no loadable path, so [`checkpoint_plan_entry_has_bespoke_lane`] is false) whose shape
/// this route does not serve, which is LoRA / reference-guided / Hires.fix per
/// [`checkpoint_plan_serves_request_shape`], flips from a DECLINE that its family lane picks up to a
/// hard refusal. That is a capability regression, not a cleanup. What the table names is precisely
/// the set of families whose plan-sourced lanes are must-outlive, so the table is must-outlive too.
const CHECKPOINT_PLAN_BESPOKE_PLAN_SOURCED_FAMILIES: &[&str] = &[
    "krea_2",
    "sdxl",
    "mage-flow",
    "z-image",
    "qwen-image",
    "flux2",
];

/// Whether SOME other lane could take this request if the plan route declines it.
///
/// Two ways a lane can exist, and both matter because the answer decides whether a route-capability
/// refusal is raised HERE or retained for the router's fall-through
/// ([`CheckpointPlanSelection::into_unclaimed_refusal`]):
///
/// * **installed bytes** — a MANAGED install (sc-20636) keeps `paths.model` alongside
///   `importPlan.checkpointId`, so its family's bespoke lane can scan for them. LOADABLE paths only
///   (`modelPath` / `paths.model` / `installedPath`), never `imported_entry_installed_path`'s
///   provenance-only `source.path` fallback: that one is a data-dir-relative breadcrumb an import
///   writes, and a linked entry carrying one would claim a bespoke lane it does not have and hand
///   that lane a path it must not read (sc-20636 review). The admission side keeps the wider reading
///   — "does this entry describe bytes anywhere" is a different question from "can a lane load it".
/// * **plan-sourced lane** — a family whose sc-20644 row has landed
///   ([`CHECKPOINT_PLAN_BESPOKE_PLAN_SOURCED_FAMILIES`]) has a lane for a LINKED checkpoint too: it
///   loads the plan's own verified layer. Without this, every shape the plan route does not serve
///   refused for a linked checkpoint, which is precisely the surface a family row exists to restore.
///
/// Declining rather than refusing never loses the refusal: the message is retained and re-raised if
/// no lane claims. Integrity refusals do not reach here at all — they are raised with `?`.
fn checkpoint_plan_entry_has_bespoke_lane(request: &ImageRequest) -> bool {
    sceneworks_core::jobs_store::imported_entry_loadable_path(&request.model_manifest_entry)
        .is_some()
}

/// Whether a lane exists that could serve a request SHAPE this route does not implement.
///
/// Strictly wider than [`checkpoint_plan_entry_has_bespoke_lane`], and used for exactly one class of
/// refusal: "this route's body does not do edit / pose / LoRA / Hires". That class is the one
/// another lane can cover, and for a family whose sc-20644 row has landed it covers it even for a
/// LINKED checkpoint, because the lane sources the plan's own verified layer.
///
/// Every OTHER refusal keeps the narrow reading. A missing base tier, an unresolvable component, an
/// unconsumed layer, an unbound adapter: no bespoke lane can fix any of those — it needs the same
/// component from the same place — so declining would only move an identical error out of PLANNING
/// and into a handler preamble, which is what AC2's "fail during planning, not in the loader" and E7
/// forbid.
fn checkpoint_plan_shape_has_other_lane(request: &ImageRequest) -> bool {
    if checkpoint_plan_entry_has_bespoke_lane(request) {
        return true;
    }
    request
        .model_manifest_entry
        .get("family")
        .and_then(Value::as_str)
        .map(str::trim)
        .is_some_and(|family| CHECKPOINT_PLAN_BESPOKE_PLAN_SOURCED_FAMILIES.contains(&family))
}

/// Whether the plan-driven route claims this request (the single-claim discriminator the bespoke
/// imported lanes consult so exactly one lane owns each request).
///
/// Plan-backed alone is not the discriminator while any family is still mid-migration: a managed
/// install reached that state by being imported — it had a full family lane a moment earlier — so
/// handing the route every request and refusing the ones its family's row has not reached would mean
/// importing a model through the managed path REMOVED a capability it had. The claim is therefore
/// per-request, and its width is REGISTRY-derived
/// ([`checkpoint_plan_binds_request_operation`]): the route takes every operation this build binds
/// for the entry's family, and that family's bespoke lane keeps only what is not yet bound. When a
/// family's last operation is bound, its lane claims nothing and can be deleted (E4).
pub(crate) fn request_is_checkpoint_plan_backed(request: &ImageRequest) -> bool {
    checkpoint_plan_checkpoint_id(&request.model_manifest_entry).is_some()
        && checkpoint_plan_binds_request_operation(request)
        && checkpoint_plan_serves_request_shape(request)
}

/// The request shapes the route's own generation body implements today.
///
/// Distinct from [`checkpoint_plan_binds_request_operation`], which asks the REGISTRY whether a
/// provider exists: this asks whether THIS route can drive that provider for the request's
/// conditioning. Both must hold for the route to claim, and the two are separate because they move
/// at different times — a family's binding is registered in the inference repo, while the body that
/// feeds it lives here.
///
/// Each family row of sc-20644 lifts one part of this gate and deletes the bespoke lane that was
/// holding the corresponding shape. Anything still listed here stays with its family lane, which is
/// what keeps importing a model from REMOVING a capability it had (sc-20636).
fn checkpoint_plan_serves_request_shape(request: &ImageRequest) -> bool {
    !imported_generate_request_has_unsupported_shape(request)
        && request.loras.is_empty()
        && request.reference_asset_id.is_none()
        && !request.hires_fix.enabled
}

/// How this route answers "I cannot serve this request": decline so the entry's own bespoke family
/// lane claims it, or — when there is no such lane — refuse.
///
/// The distinction is exactly whether the entry has SceneWorks-owned bytes another lane can load.
/// It applies ONLY to route-capability refusals (an unsupported shape, a layer this skeleton cannot
/// source, no adapter bound). Integrity refusals — drift, a missing source, a tampered plan — stay
/// fatal for every entry: falling back to a bespoke lane that would load the very bytes the plan
/// just rejected is the silent substitution E7/E8 forbid.
/// The outcome of offering a request to the plan-driven route.
///
/// A decline is not the same as "no opinion": the route declined because the entry LOOKED like it
/// had a bespoke family lane, and that lane may not exist for this shape. So the refusal it would
/// otherwise have raised is retained here and re-raised by the router if every lane declines —
/// without that, a plan-backed entry with no claiming lane reaches `generate_stub_stream` and the
/// job COMPLETES with procedural stub output instead of the typed refusal (sc-20636 review).
#[derive(Default)]
pub(crate) struct CheckpointPlanSelection {
    prepared: Option<PreparedCheckpointPlanSources>,
    declined: Option<String>,
}

impl CheckpointPlanSelection {
    fn served(sources: PreparedCheckpointPlanSources) -> Self {
        Self {
            prepared: Some(sources),
            declined: None,
        }
    }

    fn is_available(&self) -> bool {
        self.prepared.is_some()
    }

    fn into_sources(self) -> Option<PreparedCheckpointPlanSources> {
        self.prepared
    }

    /// The refusal to raise when NO lane claimed the request. `Ok(())` only for an entry that is
    /// not plan-backed at all; everything else is a typed refusal, never a stub render.
    fn into_unclaimed_refusal(self, request: &ImageRequest) -> WorkerResult<()> {
        if let Some(message) = self.declined {
            return Err(WorkerError::InvalidPayload(message));
        }
        match checkpoint_plan_checkpoint_id(&request.model_manifest_entry) {
            Some(checkpoint_id) => Err(WorkerError::InvalidPayload(format!(
                "[checkpoint-plan:no-adapter-binding] checkpoint {checkpoint_id:?}: no image lane \
                 on this backend serves this request, and a plan-backed entry never renders \
                 procedural stub output"
            ))),
            None => Ok(()),
        }
    }
}

fn checkpoint_plan_unservable(
    request: &ImageRequest,
    message: String,
) -> WorkerResult<CheckpointPlanSelection> {
    checkpoint_plan_unservable_when(request, message, checkpoint_plan_entry_has_bespoke_lane)
}

/// The SHAPE-refusal form of [`checkpoint_plan_unservable`]: declines to any lane that could serve
/// the shape, including a plan-sourced bespoke lane for a linked checkpoint.
fn checkpoint_plan_unservable_shape(
    request: &ImageRequest,
    message: String,
) -> WorkerResult<CheckpointPlanSelection> {
    checkpoint_plan_unservable_when(request, message, checkpoint_plan_shape_has_other_lane)
}

fn checkpoint_plan_unservable_when(
    request: &ImageRequest,
    message: String,
    has_other_lane: fn(&ImageRequest) -> bool,
) -> WorkerResult<CheckpointPlanSelection> {
    if has_other_lane(request) {
        return Ok(CheckpointPlanSelection {
            prepared: None,
            declined: Some(message),
        });
    }
    Err(WorkerError::InvalidPayload(message))
}

fn checkpoint_plan_refusal(error: CheckpointPlanError) -> WorkerError {
    WorkerError::InvalidPayload(error.to_string())
}

/// Every plan-derived, pinned input the handler needs. Selected once in `prepare_image_route` and
/// carried through dispatch so the async preamble can never retarget a source.
struct PreparedCheckpointPlanSources {
    checkpoint_id: String,
    resolved: ResolvedCheckpointV1,
    /// The family's portable checkpoint-adapter authority — eligible backends, dialect source
    /// shapes, component topology and per-operation capability policy.
    adapter: &'static gen_core::CheckpointAdapterRegistration,
    /// The registered provider that loads this family/source/operation.
    descriptor: gen_core::ModelDescriptor,
    source: gen_core::ImportedModelSource,
    /// The registry operation the request selected, and the one the descriptor was resolved for.
    operation: gen_core::ImportedModelOperation,
    /// The provider's primary weights, as the loader will receive them (a File for a single-file
    /// backbone, a Dir for a component-directory dialect).
    primary: WeightsSource,
    /// Components the provider declares as required, resolved to local sources.
    components: Vec<(&'static str, WeightsSource)>,
    /// Every payload-selected File token the route retains from selection through load: the
    /// primary's file(s) plus any component sourced from a plan layer.
    pins: Vec<gen_core::PinnedWeightsFile>,
}

/// The backbone role a given source shape's primary layer must carry.
///
/// A fused checkpoint is the one shape whose backbone is the inspector's `checkpoint` role (an
/// all-in-one LDM/A1111 container); every other shape's backbone is a `transformer`. Derived from
/// the shape rather than from the family, so a new family that declares an existing dialect shape
/// inherits the rule with no edit here.
fn checkpoint_plan_primary_role(source: gen_core::ImportedModelSource) -> &'static str {
    match source {
        gen_core::ImportedModelSource::FusedCheckpoint => CHECKPOINT_PLAN_FUSED_ROLE,
        gen_core::ImportedModelSource::TransformerFile
        | gen_core::ImportedModelSource::TransformerDirectory
        | gen_core::ImportedModelSource::ComfyUiTree => CHECKPOINT_PLAN_TRANSFORMER_ROLE,
    }
}

/// The provider's primary weights, resolved from the plan, plus every plan layer that resolution
/// consumed and every payload-selected File token that must stay pinned from selection to load.
struct CheckpointPlanPrimary {
    weights: WeightsSource,
    consumed: std::collections::BTreeSet<String>,
    pins: Vec<gen_core::PinnedWeightsFile>,
}

/// Resolve the plan's primary weights for `source`.
///
/// The shape decides the artifact the loader is handed:
///
/// * `TransformerFile` / `FusedCheckpoint` / `ComfyUiTree` — the single backbone layer's FILE. (A
///   ComfyUI tree's encoder and VAE are separate FILES too, but they arrive as declared provider
///   COMPONENTS, not as the primary; see [`checkpoint_plan_component_from_layers`].)
/// * `TransformerDirectory` — the DIRECTORY holding the single backbone layer, which is what a
///   diffusers component dir's loader opens. Every other plan layer inside that directory (its
///   `config.json` sidecar, sharded weights) is consumed by the directory itself and pinned, so a
///   torn artifact is caught at selection rather than mid-load.
///
/// Exactly one backbone layer, always: two would mean the plan describes two models and the route
/// would have to pick, which is the silent substitution E8 forbids.
/// The PURE half of [`checkpoint_plan_primary`]: which artifact the loader gets and which plan
/// layers that artifact covers, decided entirely from the plan and the source shape with no
/// filesystem access. Split out so the shape rules are unit-testable over a synthesized
/// [`ResolvedCheckpointV1`] without materializing multi-gigabyte files.
#[derive(Debug)]
struct CheckpointPlanPrimarySelection<'a> {
    /// `Some` only for a component-directory dialect, whose loader opens the directory itself.
    directory: Option<PathBuf>,
    /// Every layer the primary artifact covers, in plan order. The backbone is always first.
    layers: Vec<&'a ResolvedLayerV1>,
}

fn checkpoint_plan_primary_selection<'a>(
    resolved: &'a ResolvedCheckpointV1,
    source: gen_core::ImportedModelSource,
) -> WorkerResult<CheckpointPlanPrimarySelection<'a>> {
    let role = checkpoint_plan_primary_role(source);
    let backbones: Vec<_> = resolved.layers_with_role(role).collect();
    let backbone = match backbones.as_slice() {
        [backbone] => *backbone,
        [] => {
            return Err(WorkerError::InvalidPayload(format!(
                "[checkpoint-plan:missing-component] checkpoint {:?} ({} family) resolves to a \
                 {source:?} source, which needs exactly one {role:?} layer; its plan carries roles \
                 [{}]",
                resolved.checkpoint_id,
                resolved.family(),
                checkpoint_plan_layer_roles(resolved)
            )))
        }
        many => {
            return Err(WorkerError::InvalidPayload(format!(
                "[checkpoint-plan:ambiguous-component] checkpoint {:?} ({} family) carries {} \
                 {role:?} layers; a {source:?} source has exactly one primary",
                resolved.checkpoint_id,
                resolved.family(),
                many.len()
            )))
        }
    };
    if source != gen_core::ImportedModelSource::TransformerDirectory {
        return Ok(CheckpointPlanPrimarySelection {
            directory: None,
            layers: vec![backbone],
        });
    }
    let directory = backbone.path.parent().ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "[checkpoint-plan:missing-component] checkpoint {:?} ({} family) resolves to a \
             component directory, but its {role:?} layer {:?} has no parent directory",
            resolved.checkpoint_id,
            resolved.family(),
            backbone.layer.layer_id
        ))
    })?;
    let mut layers = vec![backbone];
    layers.extend(resolved.layers.iter().filter(|layer| {
        layer.layer.layer_id != backbone.layer.layer_id && layer.path.parent() == Some(directory)
    }));
    Ok(CheckpointPlanPrimarySelection {
        directory: Some(directory.to_path_buf()),
        layers,
    })
}

fn checkpoint_plan_primary(
    resolved: &ResolvedCheckpointV1,
    source: gen_core::ImportedModelSource,
) -> WorkerResult<CheckpointPlanPrimary> {
    let selection = checkpoint_plan_primary_selection(resolved, source)?;
    let mut consumed = std::collections::BTreeSet::new();
    let mut pins = Vec::with_capacity(selection.layers.len());
    for (index, layer) in selection.layers.iter().enumerate() {
        consumed.insert(layer.layer.layer_id.clone());
        // The backbone is always `layers[0]` (see `checkpoint_plan_primary_selection`), and it is
        // the artifact the weights loader itself opens, so it is held to the weights containers
        // alone. The rest exist only for the component-directory shape — the sharded siblings and
        // the `config.json` sidecar that live beside the backbone — so they may also be JSON.
        let allowed = if index == 0 {
            CHECKPOINT_PLAN_WEIGHTS_CONTAINERS
        } else {
            CHECKPOINT_PLAN_DIRECTORY_CONTAINERS
        };
        pins.push(checkpoint_plan_pin(resolved, layer, allowed)?);
    }
    let weights = match selection.directory {
        Some(directory) => WeightsSource::Dir(directory),
        None => WeightsSource::File(
            pins.first()
                .expect("a primary selection always carries its backbone layer")
                .loader_path()
                .to_path_buf(),
        ),
    };
    Ok(CheckpointPlanPrimary {
        weights,
        consumed,
        pins,
    })
}

/// The containers a layer a WEIGHTS loader opens may be packed in.
///
/// Safetensors and nothing else, and stated as an ALLOW-list rather than as "not GGUF" on purpose:
/// every dialect registered on either backend today — `ldm`, `diffusers`, the ComfyUI trees —
/// opens safetensors, so a container that is not on this list is one no registered loader can read.
/// A denylist would admit the next container the inspector learns to recognize by default, which is
/// exactly the failure this gate exists to prevent.
const CHECKPOINT_PLAN_WEIGHTS_CONTAINERS: &[CheckpointContainerV1] =
    &[CheckpointContainerV1::Safetensors];

/// The containers a NON-backbone layer of a component directory may be packed in.
///
/// The component-directory shape hands the loader the whole directory, so the plan's sharded weight
/// files and the `config.json` sidecar beside them are pinned too. The sidecar is legitimately a
/// JSON descriptor; nothing in the directory may be GGUF.
const CHECKPOINT_PLAN_DIRECTORY_CONTAINERS: &[CheckpointContainerV1] = &[
    CheckpointContainerV1::Safetensors,
    CheckpointContainerV1::JsonDescriptor,
];

/// Pin one resolved plan layer's bytes for the life of the request.
///
/// The plan store already proved these bytes are the bytes the plan compiled from; the pin is what
/// keeps the async job preamble from being able to retarget the path between selection and load
/// (sc-18306), exactly as every bespoke imported lane pins its payload-selected files.
///
/// This is also the ONE place a plan layer becomes a loadable artifact — `checkpoint_plan_primary`
/// (every source shape, and through it the plan route and all three bespoke helpers) and
/// `checkpoint_plan_component_from_layers` (declared components, and the roles-only candle seam)
/// are its only callers — so it is where the CONTAINER is checked, before `pin` so much as opens
/// the file (sc-20651). Components resolved from a resident tier rather than from a plan layer
/// never reach here and are not plan bytes.
fn checkpoint_plan_pin(
    resolved: &ResolvedCheckpointV1,
    layer: &ResolvedLayerV1,
    allowed: &[CheckpointContainerV1],
) -> WorkerResult<gen_core::PinnedWeightsFile> {
    if !allowed.contains(&layer.layer.container) {
        return Err(WorkerError::InvalidPayload(format!(
            "[checkpoint-plan:unsupported-container] checkpoint {:?} ({} family) layer {:?} is a \
             {:?} container ({}), and this route loads {allowed:?}; a plan is never handed to a \
             loader that cannot read its container",
            resolved.checkpoint_id,
            resolved.family(),
            layer.layer.role,
            layer.layer.container,
            layer.layer.target_path,
        )));
    }
    gen_core::PinnedWeightsFile::pin(&layer.path).map_err(|error| {
        crate::classify_engine_error("Checkpoint plan source preparation failed", error)
    })
}

/// Every role the plan carries, for diagnostics.
fn checkpoint_plan_layer_roles(resolved: &ResolvedCheckpointV1) -> String {
    resolved
        .plan
        .layers
        .iter()
        .map(|layer| layer.role.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Every plan layer this route does not source, as a typed refusal.
///
/// The inspector emits multi-layer plans — a linked directory Krea checkpoint compiles to
/// `transformer` + `vae` + `text_encoder`, and descriptor/artifact roles appear too. This skeleton
/// sources only the primary layer; the provider's remaining components come from the resident
/// family base tier. Loading the plan's transformer while quietly substituting the base tier's VAE
/// and encoder for the plan's OWN would be a silent substitution, and would mean the plan is not
/// consumed everywhere it is claimed to be. Refuse instead, naming the roles this route cannot
/// source, until sc-20644 maps each remaining role onto a provider component.
fn checkpoint_plan_unconsumed_layers(
    resolved: &ResolvedCheckpointV1,
    consumed: &std::collections::BTreeSet<String>,
) -> WorkerResult<()> {
    let unconsumed: Vec<&str> = resolved
        .layers
        .iter()
        .filter(|layer| !consumed.contains(&layer.layer.layer_id))
        .map(|layer| layer.layer.role.as_str())
        .collect();
    if unconsumed.is_empty() {
        return Ok(());
    }
    // Name BOTH halves: the roles that would go unloaded, and the roles that were sourced. Without
    // the second half the diagnostic says a checkpoint is unservable without saying what the route
    // did take, which is the difference between an actionable message and a dead end.
    let sourced: Vec<&str> = resolved
        .layers
        .iter()
        .filter(|layer| consumed.contains(&layer.layer.layer_id))
        .map(|layer| layer.layer.role.as_str())
        .collect();
    Err(WorkerError::InvalidPayload(format!(
        "[checkpoint-plan:unconsumed-layer] checkpoint {:?} ({} family) compiles to {} layers, but \
         the plan-driven route sources only its {:?} layer(s); layer role(s) [{}] would be \
         silently replaced by the resident base tier's own components, so this checkpoint is not \
         servable on this route",
        resolved.checkpoint_id,
        resolved.family(),
        resolved.plan.layers.len(),
        sourced.join(", "),
        unconsumed.join(", ")
    )))
}

/// One provider-declared component, resolved from the plan's own layers by inspector role.
///
/// This is what makes a multi-artifact checkpoint — a ComfyUI tree's DiT + text encoder + VAE — a
/// PLAN load rather than a catalog assembly: the encoder and VAE the provider declares as
/// components come from the same verified plan the backbone did, so every byte the route loads was
/// hashed by the inspector and re-checked by the store. Exactly one layer per role, for the
/// [`checkpoint_plan_primary`] reason.
fn checkpoint_plan_component_from_layers(
    resolved: &ResolvedCheckpointV1,
    component: &str,
    role: &'static str,
) -> WorkerResult<(WeightsSource, String, gen_core::PinnedWeightsFile)> {
    let layers: Vec<_> = resolved.layers_with_role(role).collect();
    match layers.as_slice() {
        [layer] => {
            // A declared component is bytes a loader opens, exactly like the backbone.
            let pin = checkpoint_plan_pin(resolved, layer, CHECKPOINT_PLAN_WEIGHTS_CONTAINERS)?;
            Ok((
                WeightsSource::File(pin.loader_path().to_path_buf()),
                layer.layer.layer_id.clone(),
                pin,
            ))
        }
        [] => Err(WorkerError::InvalidPayload(format!(
            "[checkpoint-plan:missing-component] checkpoint {:?} ({} family) requires component \
             {component:?}, which is sourced from this plan's {role:?} layer; the plan carries \
             roles [{}]",
            resolved.checkpoint_id,
            resolved.family(),
            checkpoint_plan_layer_roles(resolved)
        ))),
        many => Err(WorkerError::InvalidPayload(format!(
            "[checkpoint-plan:ambiguous-component] checkpoint {:?} ({} family) carries {} {role:?} \
             layers; component {component:?} is sourced from exactly one",
            resolved.checkpoint_id,
            resolved.family(),
            many.len()
        ))),
    }
}

/// One provider-declared component, resolved to bytes this backend can load.
struct CheckpointPlanComponent {
    id: &'static str,
    source: WeightsSource,
    /// The plan layer this component consumed, when it came from the plan rather than from a
    /// resident tier. Feeds [`checkpoint_plan_unconsumed_layers`].
    consumed: Option<String>,
    /// The payload-selected File token to keep pinned, when the component is a plan layer.
    pin: Option<gen_core::PinnedWeightsFile>,
}

/// Resolve one provider-declared required component to a local source.
///
/// Two kinds of component, and the difference is which authority owns the bytes:
///
/// * **From the plan** — a ComfyUI tree's `text_encoder` / `vae` are artifacts of the very
///   checkpoint being loaded, so they come from its verified plan layers and are marked consumed.
/// * **Resident** — `base_snapshot` is the family's installed base tier, which supplies the shared
///   tokenizer and architecture config a bare backbone omits. WHETHER a family needs one, and which
///   families may satisfy it, is adapter truth
///   ([`gen_core::CheckpointAdapterRegistration::component_topology`] / `base_compatibility`);
///   WHERE that tier's bytes live is SceneWorks catalog data, which is what
///   [`CHECKPOINT_PLAN_RESIDENT_BASE_TIERS`] records.
///
/// Any component id this route cannot source refuses by name during planning, never inside a loader
/// (E7).
fn resolve_checkpoint_plan_component(
    component: &'static str,
    resolved: &ResolvedCheckpointV1,
    settings: &Settings,
) -> WorkerResult<CheckpointPlanComponent> {
    let family = resolved.family();
    let from_layers = |role: &'static str| -> WorkerResult<CheckpointPlanComponent> {
        let (source, layer_id, pin) =
            checkpoint_plan_component_from_layers(resolved, component, role)?;
        Ok(CheckpointPlanComponent {
            id: component,
            source,
            consumed: Some(layer_id),
            pin: Some(pin),
        })
    };
    match component {
        gen_core::BASE_SNAPSHOT_COMPONENT => {
            Ok(CheckpointPlanComponent {
                id: component,
                source: WeightsSource::Dir(checkpoint_plan_resident_base_tier(
                    family,
                    &resolved.checkpoint_id,
                    settings,
                )?),
                consumed: None,
                pin: None,
            })
        }
        gen_core::COMFYUI_TEXT_ENCODER_COMPONENT => from_layers("text_encoder"),
        gen_core::COMFYUI_VAE_COMPONENT => from_layers("vae"),
        // A fused SDXL checkpoint's model-agnostic CLIP tokenizer vocabulary. The staging root and
        // the pinned CLIP-L revision are the SDXL lane's own, so the plan route and that lane hand
        // the loader ONE directory rather than two copies (sc-20644 SDXL row).
        CHECKPOINT_PLAN_LDM_TOKENIZER_COMPONENT => Ok(CheckpointPlanComponent {
            id: component,
            source: WeightsSource::Dir(resolve_sdxl_ldm_tokenizer_root_cache_only(
                settings,
                &resolved.checkpoint_id,
            )?),
            consumed: None,
            pin: None,
        }),
        // Anything else is a component only the FAMILY knows where to find — Mage-Flow's shared
        // `text_encoder` / `vae` live in an installed base tier addressed through the co-requisite
        // seam, not anywhere this route could guess. Ask that family's own resolver, so the plan
        // route and the family's other consumers refuse a missing component with one message.
        _ => {
            let resolved_components =
                checkpoint_plan_family_components(family, component, &resolved.checkpoint_id, settings)?;
            Ok(CheckpointPlanComponent {
                id: component,
                source: resolved_components,
                consumed: None,
                pin: None,
            })
        }
    }
}

/// The fused-LDM tokenizer component id (SDXL's `ldm_tokenizer`), spelled here because the constant
/// lives in the MLX-only `mlx-gen-sdxl` crate while this route serves both backends.
const CHECKPOINT_PLAN_LDM_TOKENIZER_COMPONENT: &str = "ldm_tokenizer";

/// Each plan family's SHARED-COMPONENT resolver: `(plan family, resolver)`.
///
/// The sibling of [`CHECKPOINT_PLAN_RESIDENT_BASE_TIERS`] for families whose provider declares named
/// components rather than one base snapshot. Catalog data, not family truth: the descriptor already
/// says WHICH components are required; this is only the app's answer to where that family installed
/// them. Each entry is the family's EXISTING resolver — the same function, not a copy — so a missing
/// component produces one message and one completeness probe no matter which route asked.
///
/// A family with no row refuses the component by name rather than loading a neighbour's (E8).
#[allow(clippy::type_complexity)]
const CHECKPOINT_PLAN_FAMILY_COMPONENT_RESOLVERS: &[(
    &str,
    fn(&Settings) -> WorkerResult<std::collections::BTreeMap<String, WeightsSource>>,
)] = &[("mage-flow", resolve_mage_finetuned_components)];

/// One provider-declared component resolved through its family's own component resolver.
fn checkpoint_plan_family_components(
    family: &str,
    component: &str,
    checkpoint_id: &str,
    settings: &Settings,
) -> WorkerResult<WeightsSource> {
    let unsupplyable = || {
        WorkerError::InvalidPayload(format!(
            "[checkpoint-plan:missing-component] checkpoint {checkpoint_id:?} ({family} family) \
             requires component '{component}', which this runtime cannot supply for that family"
        ))
    };
    let (_, resolve) = CHECKPOINT_PLAN_FAMILY_COMPONENT_RESOLVERS
        .iter()
        .find(|(candidate, _)| *candidate == family)
        .ok_or_else(unsupplyable)?;
    // The family's own resolver raises its own actionable error when the components are not
    // installed; that error is propagated verbatim rather than replaced with a generic one.
    resolve(settings)?.remove(component).ok_or_else(unsupplyable)
}

/// Each plan family's resident base tier resolver: `(plan family, resolver)`.
///
/// Catalog data, not family truth: the adapter already declares that the family HAS a
/// `base-snapshot` dependency and which families may satisfy it. This is only the app's answer to
/// "and where did we install it, and is it complete". Each entry is the family's EXISTING resolver
/// — the same function, not a copy — so the plan route and that family's other consumers refuse a
/// missing or torn base with one message and one completeness probe. Adding a family is a one-row
/// change; a family with no row refuses by name rather than loading a neighbour's tier (E8).
#[allow(clippy::type_complexity)]
const CHECKPOINT_PLAN_RESIDENT_BASE_TIERS: &[(&str, fn(&Settings) -> WorkerResult<PathBuf>)] =
    &[("krea_2", resolve_krea_imported_base_tier)];

/// The installed base tier directory for `family`, or that family's own typed refusal.
fn checkpoint_plan_resident_base_tier(
    family: &str,
    checkpoint_id: &str,
    settings: &Settings,
) -> WorkerResult<PathBuf> {
    let Some((_, resolve)) = CHECKPOINT_PLAN_RESIDENT_BASE_TIERS
        .iter()
        .find(|(candidate, _)| *candidate == family)
    else {
        return Err(WorkerError::InvalidPayload(format!(
            "[checkpoint-plan:missing-component] checkpoint {checkpoint_id:?} ({family} family) \
             requires a resident base tier, and this build records no installed base tier for that \
             family"
        )));
    };
    resolve(settings)
}

/// The companion directories the candle floor prices for this plan's declared components.
///
/// Mirrors [`resolve_checkpoint_plan_component`], which is the function that PRODUCED these
/// components: a `base_snapshot` is a resident family tier whose own `transformer/` is replaced by
/// the plan's primary file, so it is priced through the shared
/// [`imported_base_snapshot_companions`] construction (the legacy Krea imported lane's, exactly)
/// rather than recursively. Pricing the whole snapshot — which is what `admit_candle_load_spec_floor`
/// does to every `spec.components` value — would charge the plan's DiT twice and over-refuse.
///
/// Any component shape this floor cannot account for refuses rather than going unpriced.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn checkpoint_plan_candle_companion_dirs(
    sources: &PreparedCheckpointPlanSources,
    spec: &LoadSpec,
) -> WorkerResult<Vec<PathBuf>> {
    let mut companions = Vec::new();
    for (component, source) in &sources.components {
        match (*component, source) {
            (gen_core::BASE_SNAPSHOT_COMPONENT, WeightsSource::Dir(base_dir)) => companions.extend(
                imported_base_snapshot_companions(base_dir, spec.text_encoder.is_some()),
            ),
            _ => {
                return Err(WorkerError::InvalidPayload(format!(
                    "[checkpoint-plan:unpriceable-component] checkpoint {:?} ({} family) declares \
                     component {component:?} as {source:?}, which this route's candle floor cannot \
                     account for; admitting it would leave those bytes unpriced",
                    sources.checkpoint_id,
                    sources.resolved.family()
                )))
            }
        }
    }
    Ok(companions)
}

/// Select the plan-driven route for a plan-backed manifest entry. An empty selection with no
/// retained refusal happens only when the entry is not plan-backed; a plan-backed entry that cannot
/// be served either refuses here or retains its refusal for the router's fall-through, never a
/// silent fall-through to the stub.
fn prepare_checkpoint_plan_sources(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<CheckpointPlanSelection> {
    let Some(checkpoint_id) = checkpoint_plan_checkpoint_id(&request.model_manifest_entry) else {
        return Ok(CheckpointPlanSelection::default());
    };
    if !checkpoint_plan_serves_request_shape(request) {
        return checkpoint_plan_unservable_shape(
            request,
            format!(
                "[checkpoint-plan:unsupported-operation] checkpoint {checkpoint_id:?} is served \
                 through the plan-driven route for text-to-image generation only; edit, reference, \
                 pose, multi-phase, LoRA, and Hires.fix requests are not on this route yet"
            ),
        );
    }
    let store = CheckpointPlanStore::open(&settings.data_dir);
    // Integrity: never declined, always fatal. A drifted, missing, or tampered plan must not fall
    // through to a lane that would load the same bytes unverified.
    let resolved = store.resolve(checkpoint_id).map_err(checkpoint_plan_refusal)?;
    let family = resolved.family().to_owned();
    // Family truth, in the order a planner needs it: is there an adapter at all, is this backend
    // eligible for it, and what on-disk shape do its dialects describe. All three come from the
    // registered adapter, so a family is added by registering one (E2).
    let adapter = match checkpoint_plan_adapter(&family, checkpoint_id) {
        Ok(adapter) => adapter,
        Err(error) => return checkpoint_plan_unservable(request, error.to_string()),
    };
    if let Err(error) = checkpoint_plan_backend_eligible(adapter, checkpoint_id) {
        return checkpoint_plan_unservable(request, error.to_string());
    }
    let source = match checkpoint_plan_source_shape(adapter, checkpoint_id) {
        Ok(source) => source,
        Err(error) => return checkpoint_plan_unservable(request, error.to_string()),
    };
    let operation = checkpoint_plan_operation(request);
    let Some(descriptor) =
        crate::inference_runtime::imported_model_descriptor(&family, source, operation)
    else {
        return checkpoint_plan_unservable(
            request,
            format!(
                "[checkpoint-plan:no-adapter-binding] checkpoint {checkpoint_id:?}: this runtime's \
                 provider registry has no {family:?} adapter bound for {source:?} {operation:?} on \
                 this backend"
            ),
        );
    };
    if let Some(reason) =
        checkpoint_plan_request_shape_refusal(request, &descriptor, operation, checkpoint_id)
    {
        return checkpoint_plan_unservable_shape(request, reason);
    }
    let primary = match checkpoint_plan_primary(&resolved, source) {
        Ok(primary) => primary,
        Err(error) => return checkpoint_plan_unservable(request, error.to_string()),
    };
    let CheckpointPlanPrimary {
        weights: primary_weights,
        mut consumed,
        mut pins,
    } = primary;
    let mut components = Vec::with_capacity(descriptor.required_components.len());
    for component in descriptor.required_components {
        let resolved_component =
            match resolve_checkpoint_plan_component(component, &resolved, settings) {
                Ok(component) => component,
                Err(error) => return checkpoint_plan_unservable(request, error.to_string()),
            };
        if let Some(layer_id) = resolved_component.consumed {
            consumed.insert(layer_id);
        }
        if let Some(pin) = resolved_component.pin {
            pins.push(pin);
        }
        components.push((resolved_component.id, resolved_component.source));
    }
    // Last, because a component is one of the things that CAN consume a layer: only now is the
    // consumed set complete, so only now can "this plan carries bytes nobody loads" be decided.
    if let Err(error) = checkpoint_plan_unconsumed_layers(&resolved, &consumed) {
        return checkpoint_plan_unservable(request, error.to_string());
    }
    Ok(CheckpointPlanSelection::served(
        PreparedCheckpointPlanSources {
            checkpoint_id: checkpoint_id.to_owned(),
            resolved,
            adapter,
            descriptor,
            source,
            operation,
            primary: primary_weights,
            components,
            pins,
        },
    ))
}

/// Whether an entry's DECLARED family routes it to `family`'s lane at all.
///
/// The question every bespoke plan-source helper must ask FIRST, and the distinction that keeps
/// `family-mismatch` meaningful. A lane is offered every plan-backed request whose shape the plan
/// route declined, including requests belonging to other families — a plan-backed `krea_2` entry
/// carrying Hires.fix is offered to the Mage-Flow lane on its way past. That is not a corrupt plan
/// and not an error; it is simply not that lane's request, so the lane declines.
///
/// Without this the store was opened for every such request and the plan's own family compared
/// against the asking lane's, so a perfectly good Krea checkpoint raised a FATAL
/// `[checkpoint-plan:family-mismatch]` from the Mage-Flow lane — and `prepare_image_route`
/// propagates that with `?`, so the job died instead of being served by Krea's lane.
///
/// An entry with NO declared family is offered to every lane, exactly as it is today: the plan is
/// then the only authority, and [`checkpoint_plan_family_matches`] is what decides.
fn checkpoint_plan_entry_routes_to(entry: &JsonObject, family: &str) -> bool {
    entry
        .get("family")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|declared| !declared.is_empty())
        // `is_none_or` is stable only since 1.82; this workspace's MSRV is 1.80.
        .map_or(true, |declared| declared == family)
}

/// Refuse a plan whose family is not the one the asking lane loads.
///
/// Reached only for an entry whose DECLARED family already routes here
/// ([`checkpoint_plan_entry_routes_to`]), so a mismatch at this point means the entry claims one
/// family and its compiled plan says another — a corrupt or edited entry. That stays FATAL: handing
/// a mislabelled checkpoint to the wrong family's loader is the silent substitution E8 forbids.
fn checkpoint_plan_family_matches(
    resolved: &ResolvedCheckpointV1,
    checkpoint_id: &str,
    family: &str,
) -> WorkerResult<()> {
    if resolved.family() == family {
        return Ok(());
    }
    Err(WorkerError::InvalidPayload(format!(
        "[checkpoint-plan:family-mismatch] checkpoint {checkpoint_id:?} compiles to the {:?} \
         family, but this entry declares {family:?} and routes to that lane; a plan is never \
         loaded by another family's loader",
        resolved.family()
    )))
}

/// The verified plan layer a family's BESPOKE lane must load, for a plan-backed entry whose request
/// this route does not claim.
///
/// The two halves of sc-20644's per-family parity are separable: WHICH lane runs a request is the
/// single-claim discriminator ([`request_is_checkpoint_plan_backed`]), and WHICH BYTES that lane
/// opens is the plan. Before this existed the second half was tied to the first — a plan-backed
/// entry's bespoke lane declined outright — so importing a checkpoint through the managed or linked
/// path REMOVED every capability the plan route does not yet serve (LoRA, edit, pose, multi-phase,
/// img2img, Hires.fix), and a LINKED checkpoint, which has no installed path at all, could not reach
/// them by any route. The family's lane now serves those shapes, on the SAME verified layer the plan
/// route would have handed the provider, so the two routes are byte-identical inputs to the same
/// generation body and a fixed seed renders equal on either (E2/E5).
///
/// `Ok(None)` only when the entry is not plan-backed; every other outcome is decided here:
///
/// * integrity — a drifted, missing or tampered plan is FATAL, exactly as it is on the plan route.
///   Falling back to a directory scan of the same bytes the store just rejected is the silent
///   substitution E7/E8 forbid, and it is the reason this helper resolves the store rather than
///   letting the lane's own scan continue.
/// * family — a plan compiled for another family never feeds this lane, even if the manifest entry
///   claims otherwise.
/// * shape — a bespoke lane whose primary is one FILE cannot be handed a component directory; that
///   refuses by name rather than loading the directory's first shard.
/// * completeness — a plan carrying layers this lane does not source (a linked Krea directory's own
///   `vae` / `text_encoder`) refuses through the SAME [`checkpoint_plan_unconsumed_layers`] the plan
///   route uses, because loading the plan's transformer while quietly substituting the resident base
///   tier's encoder and VAE would consume the plan only partly while claiming it fully.
/// * routing — an entry whose DECLARED family is another lane's is DECLINED, not refused; see
///   [`checkpoint_plan_entry_routes_to`].
pub(crate) fn checkpoint_plan_bespoke_primary_pin(
    request: &ImageRequest,
    settings: &Settings,
    family: &str,
) -> WorkerResult<Option<gen_core::PinnedWeightsFile>> {
    let Some((checkpoint_id, source, primary)) =
        checkpoint_plan_bespoke_primary(request, settings, family)?
    else {
        return Ok(None);
    };
    if source == gen_core::ImportedModelSource::TransformerDirectory {
        return Err(WorkerError::InvalidPayload(format!(
            "[checkpoint-plan:unsupported-operation] checkpoint {checkpoint_id:?} ({family} family) \
             resolves to a component directory, which this family's single-file lane cannot open"
        )));
    }
    // A non-directory shape's selection is exactly its backbone layer, so its pin list is the one
    // pin the lane loads. Assert rather than index blindly: a future shape that widened the
    // selection would otherwise silently hand the lane the first of several files.
    match primary.pins.as_slice() {
        [pin] => Ok(Some(pin.clone())),
        many => Err(WorkerError::InvalidPayload(format!(
            "[checkpoint-plan:ambiguous-component] checkpoint {checkpoint_id:?} ({family} family) \
             resolves a {source:?} primary to {} files; this family's lane loads exactly one",
            many.len()
        ))),
    }
}

/// The DIRECTORY form of [`checkpoint_plan_bespoke_primary_pin`], for a family whose loader opens a
/// component directory rather than one file (Mage-Flow's `diffusers` dialect).
///
/// Same authority and the same refusals; the only difference is which artifact the lane is handed.
/// The directory's own layers — the transformer weight file and its `config.json` sidecar — are
/// consumed by the selection, so a torn artifact is caught here rather than mid-load, and the lane
/// re-pins the fixed filenames it loads from inside the verified directory.
pub(crate) fn checkpoint_plan_bespoke_primary_dir(
    request: &ImageRequest,
    settings: &Settings,
    family: &str,
) -> WorkerResult<Option<CheckpointPlanBespokeDir>> {
    let Some((checkpoint_id, source, primary)) =
        checkpoint_plan_bespoke_primary(request, settings, family)?
    else {
        return Ok(None);
    };
    match primary.weights {
        WeightsSource::Dir(directory) => Ok(Some(CheckpointPlanBespokeDir {
            directory,
            pins: primary.pins,
        })),
        WeightsSource::File(_) => Err(WorkerError::InvalidPayload(format!(
            "[checkpoint-plan:unsupported-operation] checkpoint {checkpoint_id:?} ({family} family) \
             resolves to a {source:?} source, but this family's loader opens a component directory"
        ))),
    }
}

/// A plan-sourced component directory and the pins covering every plan layer inside it.
///
/// The pins matter as much as the path. A LINKED library root is an APPROVED root of the checkpoint
/// plan store, not an app-managed root, so the lane must NOT re-pin the files it loads through
/// `paths::pin_app_managed_model_file` — that confinement rejects every linked checkpoint by
/// construction, which is the wrong answer rather than a safe one. The plan store is the authority
/// that already proved these bytes: it resolved the approved root, re-checked every layer's digest,
/// and pinned each one here.
pub(crate) struct CheckpointPlanBespokeDir {
    pub(crate) directory: PathBuf,
    pub(crate) pins: Vec<gen_core::PinnedWeightsFile>,
}

impl CheckpointPlanBespokeDir {
    /// The pin for the layer whose file name is `file_name`, or a typed refusal naming what the
    /// directory does carry. Never falls back to pinning the path itself: a file the plan did not
    /// record is a file the store never verified.
    pub(crate) fn pin_named(&self, file_name: &str) -> WorkerResult<gen_core::PinnedWeightsFile> {
        self.pins
            .iter()
            .find(|pin| {
                pin.loader_path()
                    .file_name()
                    .and_then(|name| name.to_str())
                    == Some(file_name)
            })
            .cloned()
            .ok_or_else(|| {
                let present: Vec<String> = self
                    .pins
                    .iter()
                    .filter_map(|pin| pin.loader_path().file_name())
                    .map(|name| name.to_string_lossy().into_owned())
                    .collect();
                WorkerError::InvalidPayload(format!(
                    "[checkpoint-plan:missing-component] the compiled checkpoint's component \
                     directory {} carries no verified {file_name:?} layer; it carries [{}]",
                    self.directory.display(),
                    present.join(", ")
                ))
            })
    }
}

/// A plan-sourced ComfyUI tree: the backbone plus each sidecar role the lane declared it consumes.
///
/// The tree families (Z-Image, Qwen-Image, FLUX.2) differ from the single-file ones in that their
/// checkpoint is SEVERAL files — a transformer, and depending on the family a text encoder and a
/// VAE — and the bespoke lanes assemble those from a live catalog scan of an `external_base_*` row.
/// A LINKED tree has no such row, so the plan's own verified layers are the only source, and they
/// are also the only source that was hashed by the inspector and re-checked by the store.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) struct CheckpointPlanBespokeTree {
    pub(crate) primary: gen_core::PinnedWeightsFile,
    /// `(role, pin)` for each role the caller declared, in the order it declared them.
    pub(crate) sidecars: Vec<(&'static str, gen_core::PinnedWeightsFile)>,
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
impl CheckpointPlanBespokeTree {
    /// The pin for one REQUIRED sidecar role. Panics only on a role the caller did not declare as
    /// required, which is a programming error rather than a runtime condition — a role declared and
    /// absent already refused inside [`checkpoint_plan_bespoke_tree`].
    pub(crate) fn sidecar(&self, role: &str) -> &gen_core::PinnedWeightsFile {
        self.optional_sidecar(role).unwrap_or_else(|| {
            panic!("role {role:?} was not declared required to checkpoint_plan_bespoke_tree")
        })
    }

    /// The pin for an OPTIONAL sidecar role: `None` when the plan does not carry it, which is the
    /// family's signal to fall back to its resident snapshot's own copy.
    pub(crate) fn optional_sidecar(&self, role: &str) -> Option<&gen_core::PinnedWeightsFile> {
        self.sidecars
            .iter()
            .find(|(candidate, _)| *candidate == role)
            .map(|(_, pin)| pin)
    }
}

/// The TREE form of [`checkpoint_plan_bespoke_primary_pin`], for a family whose checkpoint is a
/// backbone plus named sidecar artifacts.
///
/// `consumed_roles` is the lane's DECLARATION of which sidecars it loads, and it is what makes the
/// completeness check meaningful: the unconsumed-layer refusal runs over the backbone plus exactly
/// these roles, so a plan carrying an artifact the lane would silently replace with a resident
/// snapshot's own copy refuses instead (E8). Declaring a role the plan does not carry refuses too —
/// a lane never loads a sidecar the plan did not record.
///
/// `optional_roles` is for a family whose tree MAY carry an artifact (Qwen-Image's in-place VAE): a
/// plan that has it consumes it, a plan that does not falls back to the family's resident snapshot.
/// The distinction is the family's, not this route's, and it is spelled at the call site so the
/// completeness check still refuses anything neither list names.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) fn checkpoint_plan_bespoke_tree(
    request: &ImageRequest,
    settings: &Settings,
    family: &str,
    consumed_roles: &[&'static str],
    optional_roles: &[&'static str],
) -> WorkerResult<Option<CheckpointPlanBespokeTree>> {
    let Some(checkpoint_id) = checkpoint_plan_checkpoint_id(&request.model_manifest_entry) else {
        return Ok(None);
    };
    // Same first question as every other bespoke source helper: is this lane the one being asked?
    // (review blocker 1 — the ComfyUI-tree lanes are on the reviewer's list too.)
    if !checkpoint_plan_entry_routes_to(&request.model_manifest_entry, family) {
        return Ok(None);
    }
    let checkpoint_id = checkpoint_id.to_owned();
    let store = CheckpointPlanStore::open(&settings.data_dir);
    let resolved = store
        .resolve(&checkpoint_id)
        .map_err(checkpoint_plan_refusal)?;
    checkpoint_plan_family_matches(&resolved, &checkpoint_id, family)?;
    let adapter = checkpoint_plan_adapter(family, &checkpoint_id)?;
    checkpoint_plan_backend_eligible(adapter, &checkpoint_id)?;
    let source = checkpoint_plan_source_shape(adapter, &checkpoint_id)?;
    let primary = checkpoint_plan_primary(&resolved, source)?;
    let mut consumed = primary.consumed.clone();
    let mut sidecars = Vec::with_capacity(consumed_roles.len());
    for role in consumed_roles {
        let (_, layer_id, pin) = checkpoint_plan_component_from_layers(&resolved, role, role)?;
        consumed.insert(layer_id);
        sidecars.push((*role, pin));
    }
    for role in optional_roles {
        // Present-or-absent is the question; a role present MORE than once is still ambiguous and
        // still refuses, so this asks the layer table directly rather than swallowing the error.
        if resolved.layers_with_role(role).next().is_none() {
            continue;
        }
        let (_, layer_id, pin) = checkpoint_plan_component_from_layers(&resolved, role, role)?;
        consumed.insert(layer_id);
        sidecars.push((*role, pin));
    }
    checkpoint_plan_unconsumed_layers(&resolved, &consumed)?;
    match primary.pins.as_slice() {
        [pin] => Ok(Some(CheckpointPlanBespokeTree {
            primary: pin.clone(),
            sidecars,
        })),
        many => Err(WorkerError::InvalidPayload(format!(
            "[checkpoint-plan:ambiguous-component] checkpoint {checkpoint_id:?} ({family} family) \
             resolves a {source:?} backbone to {} files; this family's lane loads exactly one",
            many.len()
        ))),
    }
}

/// A plan-sourced checkpoint resolved entirely by NAMED ROLE, with no shape-derived primary.
///
/// The third and last shape a bespoke lane can want, and the one Wan 2.2 needs: its ComfyUI
/// checkpoint has no single backbone at all. It carries a high-noise and a low-noise expert, both
/// selected per denoise step, so there is nothing for [`checkpoint_plan_primary`] to pick and asking
/// it to pick would be the ambiguity the expert role vocabulary exists to remove.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) struct CheckpointPlanBespokeRoles {
    /// `(role, pin)` for every role the caller declared that the plan carries, in declaration order.
    pub(crate) layers: Vec<(&'static str, gen_core::PinnedWeightsFile)>,
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
impl CheckpointPlanBespokeRoles {
    /// The pin for one REQUIRED role. Panics only on a role the caller did not declare required,
    /// which is a programming error — a role declared required and absent already refused.
    pub(crate) fn required(&self, role: &str) -> &gen_core::PinnedWeightsFile {
        self.optional(role).unwrap_or_else(|| {
            panic!("role {role:?} was not declared required to checkpoint_plan_bespoke_roles")
        })
    }

    /// The pin for an OPTIONAL role: `None` when the plan does not carry it, which is the family's
    /// signal to fall back to its resident snapshot's own copy.
    pub(crate) fn optional(&self, role: &str) -> Option<&gen_core::PinnedWeightsFile> {
        self.layers
            .iter()
            .find(|(candidate, _)| *candidate == role)
            .map(|(_, pin)| pin)
    }
}

/// Resolve a plan-backed checkpoint to the exact set of layers a lane names, with no primary.
///
/// Takes the manifest ENTRY rather than a request, because the only thing it reads from a request is
/// `model_manifest_entry` and the caller may be a video job, whose request type is different. That is
/// also why it is `pub(crate)`: this seam is about checkpoint plans, not about image jobs.
///
/// It deliberately does NOT consult the checkpoint adapter, and the omission is the point rather
/// than an oversight. The adapter supplies two things — the dialect SOURCE SHAPE and the eligible
/// BACKENDS — and a roles-only resolution needs neither: there is no primary whose shape must be
/// decided, and the only caller is compiled behind `cfg(backend-candle)`, so "is this backend
/// eligible" is already a compile-time fact here. Requiring an adapter anyway would add a
/// cross-repository dependency for its own sake. The family truth this resolution DOES need — which
/// artifact is which expert — comes from the plan, which is the stronger authority: the inspector
/// hashed those bytes and the store re-checked them.
///
/// Every refusal the other bespoke helpers raise still applies: integrity is fatal, a plan compiled
/// for another family never feeds this lane, a declared-required role that is absent or ambiguous
/// refuses by name, and a layer neither list names trips
/// [`checkpoint_plan_unconsumed_layers`] rather than being silently left unloaded (E7/E8).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) fn checkpoint_plan_bespoke_roles(
    entry: &JsonObject,
    settings: &Settings,
    family: &str,
    required_roles: &[&'static str],
    optional_roles: &[&'static str],
) -> WorkerResult<Option<CheckpointPlanBespokeRoles>> {
    let Some(checkpoint_id) = checkpoint_plan_checkpoint_id(entry) else {
        return Ok(None);
    };
    // Same first question as [`checkpoint_plan_bespoke_primary`] and [`checkpoint_plan_bespoke_tree`],
    // and for the same reason (sc-20644 review blocker 1): this helper is offered every plan-backed
    // entry whose shape the plan route declined, other families' included. Without the gate an
    // entry DECLARING another family reached the store and then tripped the family-mismatch refusal
    // below — which is FATAL and propagated with `?`, so a perfectly good checkpoint killed the job
    // the moment the Wan lane was offered it. Declining is the right answer; the refusal below is
    // reserved for the case it was written for, an entry whose declared family routes HERE but
    // whose compiled plan says otherwise.
    if !checkpoint_plan_entry_routes_to(entry, family) {
        return Ok(None);
    }
    let checkpoint_id = checkpoint_id.to_owned();
    let store = CheckpointPlanStore::open(&settings.data_dir);
    let resolved = store
        .resolve(&checkpoint_id)
        .map_err(checkpoint_plan_refusal)?;
    if resolved.family() != family {
        return Err(WorkerError::InvalidPayload(format!(
            "[checkpoint-plan:family-mismatch] checkpoint {checkpoint_id:?} compiles to the {:?} \
             family, but this entry routes to the {family:?} lane; a plan is never loaded by \
             another family's loader",
            resolved.family()
        )));
    }
    let mut consumed = std::collections::BTreeSet::new();
    let mut layers = Vec::with_capacity(required_roles.len() + optional_roles.len());
    for role in required_roles {
        let (_, layer_id, pin) = checkpoint_plan_component_from_layers(&resolved, role, role)?;
        consumed.insert(layer_id);
        layers.push((*role, pin));
    }
    for role in optional_roles {
        // Present-or-absent is the question; a role present MORE than once is still ambiguous and
        // still refuses, so this asks the layer table directly rather than swallowing the error.
        if resolved.layers_with_role(role).next().is_none() {
            continue;
        }
        let (_, layer_id, pin) = checkpoint_plan_component_from_layers(&resolved, role, role)?;
        consumed.insert(layer_id);
        layers.push((*role, pin));
    }
    checkpoint_plan_unconsumed_layers(&resolved, &consumed)?;
    Ok(Some(CheckpointPlanBespokeRoles { layers }))
}

/// The shared body of both bespoke-source helpers: resolve and re-verify the plan, prove it belongs
/// to this family and this backend, read the shape from the adapter, select the primary, and refuse
/// a plan whose layers this lane would not consume.
#[allow(clippy::type_complexity)]
fn checkpoint_plan_bespoke_primary(
    request: &ImageRequest,
    settings: &Settings,
    family: &str,
) -> WorkerResult<Option<(String, gen_core::ImportedModelSource, CheckpointPlanPrimary)>> {
    let Some(checkpoint_id) = checkpoint_plan_checkpoint_id(&request.model_manifest_entry) else {
        return Ok(None);
    };
    // FIRST, before the store is even opened: is this lane the one being asked? A lane is offered
    // every plan-backed request whose shape the plan route declined, other families' included, and
    // one of those is not an error (review blocker 1).
    if !checkpoint_plan_entry_routes_to(&request.model_manifest_entry, family) {
        return Ok(None);
    }
    let checkpoint_id = checkpoint_id.to_owned();
    let store = CheckpointPlanStore::open(&settings.data_dir);
    let resolved = store
        .resolve(&checkpoint_id)
        .map_err(checkpoint_plan_refusal)?;
    checkpoint_plan_family_matches(&resolved, &checkpoint_id, family)?;
    let adapter = checkpoint_plan_adapter(family, &checkpoint_id)?;
    checkpoint_plan_backend_eligible(adapter, &checkpoint_id)?;
    let source = checkpoint_plan_source_shape(adapter, &checkpoint_id)?;
    let primary = checkpoint_plan_primary(&resolved, source)?;
    checkpoint_plan_unconsumed_layers(&resolved, &primary.consumed)?;
    Ok(Some((checkpoint_id, source, primary)))
}

/// Optional per-request override of a u32 knob: `advanced[key]`, else the manifest entry's
/// `[key]`. `None` means "the provider's own default", which the registry descriptor owns.
fn checkpoint_plan_u32_override(request: &ImageRequest, key: &str) -> Option<u32> {
    let parse = |value: &Value| {
        value
            .as_u64()
            .or_else(|| value.as_str()?.trim().parse().ok())
            .and_then(|value| u32::try_from(value).ok())
    };
    request
        .advanced
        .get(key)
        .and_then(parse)
        .or_else(|| request.model_manifest_entry.get(key).and_then(parse))
}

fn checkpoint_plan_f32_override(request: &ImageRequest, key: &str) -> Option<f32> {
    let parse = |value: &Value| {
        value
            .as_f64()
            .or_else(|| value.as_str()?.trim().parse().ok())
            .map(|value| value as f32)
    };
    request
        .advanced
        .get(key)
        .and_then(parse)
        .or_else(|| request.model_manifest_entry.get(key).and_then(parse))
}

/// The `LoadSpec` the plan implies: the primary file plus every declared component, with the
/// request's quant and every file pin finalized. Pure given the prepared sources, so the parity
/// test can compare it against the legacy lane's spec.
fn checkpoint_plan_load_spec(
    sources: &PreparedCheckpointPlanSources,
    quant: Option<Quant>,
) -> WorkerResult<LoadSpec> {
    let mut spec = sources.components.iter().cloned().fold(
        LoadSpec::new(sources.primary.clone()),
        |spec, (id, source)| spec.with_component(id, source),
    );
    if let Some(quant) = quant {
        spec = spec.with_quant(quant);
    }
    // Every File token the plan contributed — the primary's file(s) and each component sourced from
    // a plan layer — finalized in one atomic pass on the spec admission and load both use, so no
    // await between selection and load can retarget one of them (sc-18306).
    crate::paths::prepare_load_spec_with_file_pins(
        &mut spec,
        sources.pins.iter().cloned(),
        "Checkpoint plan source preparation failed",
    )?;
    Ok(spec)
}

fn checkpoint_plan_raw_settings(
    request: &ImageRequest,
    sources: &PreparedCheckpointPlanSources,
    steps: Option<u32>,
    guidance: Option<f32>,
    quant_bits: Option<i64>,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert(
        "mode".to_owned(),
        Value::String("text_to_image".to_owned()),
    );
    raw.insert(
        "engine".to_owned(),
        Value::String(CHECKPOINT_PLAN_ENGINE.to_owned()),
    );
    raw.insert(
        "checkpointId".to_owned(),
        Value::String(sources.checkpoint_id.clone()),
    );
    raw.insert(
        "importPlanId".to_owned(),
        Value::String(sources.resolved.plan.plan_id.clone()),
    );
    raw.insert(
        "importPlanSemanticDigest".to_owned(),
        Value::String(sources.resolved.record.plan.semantic_digest.clone()),
    );
    raw.insert(
        "importPlanFamily".to_owned(),
        Value::String(sources.resolved.plan.family.clone()),
    );
    raw.insert(
        "importPlanProvider".to_owned(),
        Value::String(sources.descriptor.id.to_owned()),
    );
    raw.insert(
        "importPlanSource".to_owned(),
        Value::String(format!("{:?}", sources.source)),
    );
    // The authority that decided this render's family truth, recorded beside the plan identity that
    // decided its bytes: an asset can be traced back to the exact adapter registration and the exact
    // registry operation the route resolved, not just to the provider that ran.
    raw.insert(
        "importPlanAdapter".to_owned(),
        Value::String(sources.adapter.adapter_id.to_owned()),
    );
    raw.insert(
        "importPlanOperation".to_owned(),
        Value::String(format!("{:?}", sources.operation)),
    );
    if let Some(steps) = steps {
        raw.insert("numInferenceSteps".to_owned(), json!(steps));
    }
    if let Some(guidance) = guidance {
        raw.insert("guidanceScale".to_owned(), json!(guidance));
    }
    raw.insert(
        "mlxQuantize".to_owned(),
        quant_bits.map_or(Value::Null, Value::from),
    );
    raw
}

/// The plan route's MLX request inputs. This skeleton serves plain text-to-image only — no
/// references, adapters, PiD, or phases reach it (`prepare_checkpoint_plan_sources` refuses those
/// shapes) — so this is the identity `krea_imported_memory_inputs(request, &[], None, 0)` produces
/// for the same request, and the two lanes price the same geometry.
#[cfg(target_os = "macos")]
fn checkpoint_plan_memory_inputs(request: &ImageRequest) -> crate::mlx_fit_gate::MlxRequestInputs {
    crate::mlx_fit_gate::MlxRequestInputs {
        width: request.width,
        height: request.height,
        count: request.count,
        mode: request.mode.clone(),
        overlay: None,
        adapter_count: 0,
        has_reference: false,
        reference_count: 0,
        use_pid: false,
        has_phases: false,
    }
}

/// Plan-driven text-to-image: `count` renders, each its own seed, through the provider the
/// registry bound for the plan's family and source shape.
#[allow(clippy::too_many_arguments)]
async fn generate_checkpoint_plan_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    sources: PreparedCheckpointPlanSources,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let (quant, quant_bits) =
        imported_model_quant(request, &sources.descriptor, "Checkpoint plan")?;
    let (width, height) = (request.width, request.height);
    let steps = checkpoint_plan_u32_override(request, "steps").map(|steps| steps.clamp(1, 100));
    let guidance = checkpoint_plan_f32_override(request, "guidanceScale");
    let raw_settings = checkpoint_plan_raw_settings(request, &sources, steps, guidance, quant_bits);
    let negative_prompt = (!request.negative_prompt.trim().is_empty())
        .then(|| request.negative_prompt.clone());
    let work: Vec<(i64, String)> = (0..request.count as usize)
        .map(|index| (resolve_seed(request, index), request.prompt.clone()))
        .collect();
    let total = work.len();

    let spec = checkpoint_plan_load_spec(&sources, quant)?;
    let engine_id = sources.descriptor.id;

    // Admission is the legacy Krea imported lane's, not a weaker twin (sc-20634 review).
    //
    // macOS: the manifest/geometry-aware request plan plus the request-state seam, so the same
    // weights get the same residency policy, the same warm/loaded-policy settlement, and the same
    // per-request refusal boundary they get on the bespoke lane. `apply_residency_policy` is NOT
    // called here: the generator cache runs it inside its own cold loader, and the bespoke Krea
    // lane leaves it to the cache for exactly that reason.
    //
    // candle: the base snapshot is a companion whose `transformer/` the plan's file replaces, so
    // the floor is the prepared-pin floor over the shared companion construction — never
    // `admit_candle_load_spec_floor`, which prices every component value recursively and would
    // charge the DiT twice.
    #[cfg(target_os = "macos")]
    let memory_plan = crate::mlx_fit_gate::MlxRequestPlan::try_for_spec_and_manifest(
        engine_id,
        &request.model,
        &spec,
        Some(&request.model_manifest_entry),
        None,
    )?;
    #[cfg(target_os = "macos")]
    let memory_inputs = checkpoint_plan_memory_inputs(request);
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let cold_admission = {
        let companions = checkpoint_plan_candle_companion_dirs(&sources, &spec)?;
        let companion_refs = companions.iter().map(PathBuf::as_path).collect::<Vec<_>>();
        prepare_cached_candle_base_floor(
            &request.model,
            "Checkpoint plan",
            settings,
            &spec,
            &companion_refs,
        )?
    };

    let checkpoint_id = sources.checkpoint_id.clone();
    let generate_one = move |model: &dyn gen_core::Generator,
                             seed: i64,
                             prompt: String,
                             memory: Option<gen_core::GenerationMemory>,
                             context: Option<&gen_core::MemoryRunContext>,
                             preview: gen_core::PreviewSink,
                             cancel: &CancelFlag,
                             on_progress: &mut dyn FnMut(Progress),
                             negative_prompt: Option<String>,
                             checkpoint_id: &str|
          -> WorkerResult<Option<GeneratedImage>> {
        if cancel.is_cancelled() {
            return Ok(None);
        }
        let mut generation = GenerationRequest {
            prompt,
            negative_prompt,
            width,
            height,
            count: 1,
            seed: Some(seed as u64),
            steps,
            guidance,
            preview,
            cancel: cancel.clone(),
            memory,
            ..Default::default()
        };
        // The same seam every request-scoped MLX lane generates through: the selected memory
        // strategy plus its run context. On candle both are `None` and this is the plain call.
        let output = match crate::memory_strategy::generate_with_scope(
            model,
            &mut generation,
            context,
            on_progress,
        ) {
            Ok(output) => output,
            Err(_) if cancel.is_cancelled() => return Ok(None),
            Err(error) => {
                return Err(WorkerError::Engine(format!(
                    "Checkpoint plan {checkpoint_id:?} generation failed: {error}"
                )));
            }
        };
        match output {
            GenerationOutput::Images(mut images) => {
                let image = images.pop().ok_or_else(|| {
                    WorkerError::Engine(format!(
                        "Checkpoint plan {checkpoint_id:?} produced no image"
                    ))
                })?;
                Ok(Some((seed, image.width, image.height, image.pixels)))
            }
            _ => Err(WorkerError::Engine(format!(
                "Checkpoint plan {checkpoint_id:?} returned non-image output"
            ))),
        }
    };

    #[cfg(target_os = "macos")]
    let (cancel, rx, blocking) = start_cached_gen_stream_with_request_state(
        job.id.clone(),
        engine_id,
        0,
        spec,
        format!("Checkpoint plan {checkpoint_id:?} load failed"),
        move |model,
              initial_cache_state,
              loaded_policy,
              warm_policy,
              external_committed_bytes,
              tx,
              cancel| {
            let mut request_cache_state = initial_cache_state;
            let mut warm_policy = crate::execution_planner::WarmPolicyOnce::new(warm_policy);
            drive_gen_items(tx, work, move |_index, (seed, prompt), preview, on_progress| {
                if cancel.is_cancelled() {
                    return Ok(None);
                }
                let evaluation = crate::mlx_fit_gate::evaluate_request(
                    model,
                    &memory_plan,
                    &memory_inputs,
                    request_cache_state,
                    loaded_policy.offload_policy,
                    warm_policy.take(),
                    external_committed_bytes,
                )?;
                request_cache_state = gen_core::MemoryCacheState::Warm;
                let _request_memory_limit = evaluation
                    .process_limit_bytes
                    .and_then(crate::generator_cache::apply_request_gpu_memory_limit);
                generate_one(
                    model,
                    seed,
                    prompt,
                    Some(evaluation.memory),
                    Some(&evaluation.context),
                    preview,
                    &cancel,
                    on_progress,
                    negative_prompt.clone(),
                    &checkpoint_id,
                )
            })
        },
    );

    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let incoming_reclaimable_weight_bytes = cold_admission.reclaimable_weight_bytes();
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let (cancel, rx, blocking) = start_cached_gen_stream_after_cold_admission(
        job.id.clone(),
        engine_id,
        0,
        spec,
        format!("Checkpoint plan {checkpoint_id:?} load failed"),
        ColdLoadAdmission::new(
            incoming_reclaimable_weight_bytes,
            move |resident_reclaimable_weight_bytes| {
                cold_admission.admit(resident_reclaimable_weight_bytes)
            },
        ),
        move |model, tx, cancel| {
            drive_gen_items(tx, work, move |_index, (seed, prompt), preview, on_progress| {
                generate_one(
                    model,
                    seed,
                    prompt,
                    None,
                    None,
                    preview,
                    &cancel,
                    on_progress,
                    negative_prompt.clone(),
                    &checkpoint_id,
                )
            })
        },
    );

    consume_gen_events(
        api,
        settings,
        job,
        plan,
        project_path,
        backend,
        CHECKPOINT_PLAN_ENGINE,
        &raw_settings,
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}
