//! `film-harness make-references` — generate a reference pack's plates locally (sc-23403).
//!
//! **TEST FIXTURES ONLY.** In the product the user supplies the reference images; a pack is a
//! human-approved document and nothing here changes that. This exists so the film harness can
//! produce its OWN courier/workshop fixtures on this machine instead of shipping flat placeholder
//! plates forever, and so that every plate it produces says where it came from.
//!
//! What it drives is the ordinary image-generation route — `POST /api/v1/image/jobs` with
//! `mode: "text_to_image"`, Krea 2 on the MLX lane by default — once per role the spec declares.
//! It adds no job type, no model and no route. The three properties that make it a fixture
//! generator rather than a second renderer:
//!
//! * **Everything it does is bounded by the spec's own declared limits.** Per-job wall clock,
//!   attempts per role, and the memory budget checked against the host before dispatch and against
//!   each job's `peakMemoryBytes` after it — the same metrics route the video run reads.
//! * **A pack is published or it does not exist.** Every file is written into a temporary
//!   directory beside `--out`, named after THIS run so two runs cannot share it, and the directory
//!   is renamed into place only once the last plate has landed and the document has been written. A
//!   refusal, a failed job, a timeout or an interrupt removes the temporary directory, so `--out` is
//!   never a half-written pack a later `validate` would have to reason about. A `--force`
//!   replacement renames the old pack aside, renames the new one in, and only then removes the old
//!   one — and `--force` is refused outright on a directory that is not a pack, so it can never be
//!   pointed at the plan directory and take `plan.jsonc` with it.
//! * **Provenance rides with the plate.** Every generated entry carries `generated: true` plus the
//!   model, tier, backend, prompt, negative prompt, seed, job id, asset id and sha256 it came from,
//!   and [`super::Session::import_reference`] carries the same block onto the imported asset. The
//!   flag changes nothing else: a generated reference is imported, tagged and conditioned on
//!   exactly like a supplied one, and `approved` remains the only gate on conditioning.
//!
//! Roles a plan needs but a spec does not render — the style reference, the keyframe plate — are
//! copied verbatim from an existing pack named by the spec's `inherit` block, along with its
//! approved sound, so the pack this writes validates against the same plan the source pack did.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use sceneworks_core::film_compile::mlx_quantize_for_tier;
use sceneworks_core::film_plan::{
    self, GeneratedReference, PlanDiagnostic, ReferenceEntry, ReferencePack, ReferenceSpec,
    ReferenceSpecEntry,
};
use sceneworks_core::time::utc_now;
use serde_json::{json, Map as JsonObject, Value};
use tokio::time::Instant;

use super::{
    api_detail, host_facts_for, job_peak_memory_bytes, live_worker_advertising,
    model_tier_installed, plan_catalog_for, sha256_hex, stale_workers_detail, ApiTransport, Client,
    HarnessError, PollBounds, PollStop, RunControl, BYTES_PER_GB, CANCEL_GRACE,
};

/// The pack document a run writes into `--out`.
pub const REFERENCE_PACK_FILE: &str = "references.jsonc";

/// The worker capability an image job needs somebody live to advertise.
pub const IMAGE_CAPABILITY: &str = "image_generate";

/// The route one plate is rendered through. Every dispatch in this module POSTs to it and to
/// nothing else — a generator that grew a second route would stop being a generator.
pub const IMAGE_ROUTE: &str = "/api/v1/image/jobs";

/// Infix of the directory a run writes into before it is renamed onto `--out`. The run id follows
/// it, so two runs against the same `--out` cannot land in the same pending directory.
const PENDING_INFIX: &str = "make-references-pending";

/// Infix of the directory the OLD pack is renamed aside to while a `--force` publish swaps the new
/// one in, so the published pack is replaced at a rename and never by a removal.
const REPLACED_INFIX: &str = "make-references-replaced";

/// Everything a `make-references` run needs beyond the transport.
#[derive(Debug, Clone)]
pub struct MakeReferencesOptions {
    /// The spec document to render.
    pub spec_path: PathBuf,
    /// Where the finished pack directory is published.
    pub out_dir: PathBuf,
    /// Override the spec's model id (`--model krea_2_raw`).
    pub model_id: Option<String>,
    /// Override the spec's tier (`--tier q4`).
    pub tier: Option<String>,
    /// Generate into an existing project instead of creating one.
    pub project_id: Option<String>,
    pub poll_interval: Duration,
    /// Refuse a model/tier the catalog reports as not installed.
    pub require_installed: bool,
    /// Replace an existing `--out` directory instead of refusing it.
    pub force: bool,
    pub control: RunControl,
}

impl MakeReferencesOptions {
    pub fn new(spec_path: PathBuf, out_dir: PathBuf) -> Self {
        Self {
            spec_path,
            out_dir,
            model_id: None,
            tier: None,
            project_id: None,
            poll_interval: Duration::from_secs(5),
            require_installed: true,
            force: false,
            control: RunControl::new(),
        }
    }
}

/// One plate a run rendered.
#[derive(Debug, Clone)]
pub struct GeneratedPlate {
    pub role: String,
    /// Path, relative to the pack directory, the plate was written to.
    pub file: String,
    pub job_id: String,
    pub asset_id: String,
    pub seed: Option<i64>,
    pub width: u32,
    pub height: u32,
    pub bytes: usize,
    pub sha256: String,
    pub backend: Option<String>,
    /// Peak the job's metrics block reported, in GiB, when it reported one.
    pub peak_memory_gb: Option<f64>,
    pub elapsed_seconds: f64,
}

/// What one `make-references` run produced.
#[derive(Debug, Clone)]
pub struct ReferencePackBuild {
    /// The written pack document.
    pub pack_path: PathBuf,
    pub pack: ReferencePack,
    pub generated: Vec<GeneratedPlate>,
    /// Roles copied verbatim from the spec's source pack, in pack order.
    pub inherited: Vec<String>,
    /// The project the plates were generated in; its assets are the generation's own record.
    pub project_id: String,
    pub model_id: String,
    pub tier: Option<String>,
    pub elapsed_seconds: f64,
}

/// Read, validate and render a reference spec into a pack directory.
///
/// Every refusal happens before the first job where it possibly can: the spec, the source pack it
/// inherits from, the catalog entry, the geometry, the host's memory and the live worker are all
/// checked up front, because each of them can otherwise be discovered after a plate has already
/// cost a GPU render.
pub async fn make_references(
    transport: &dyn ApiTransport,
    options: &MakeReferencesOptions,
) -> Result<ReferencePackBuild, HarnessError> {
    let started = Instant::now();
    let spec = film_plan::read_reference_spec_file(&options.spec_path)
        .map_err(|finding| HarnessError::Validation(vec![finding]))?;
    let spec_dir = options
        .spec_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let mut findings = film_plan::validate_reference_spec(&spec);
    let inherited = match findings.is_empty() {
        true => {
            let (entries, more) = resolve_inherited(&spec, &spec_dir);
            findings.extend(more);
            entries
        }
        false => Inherited::default(),
    };
    if !findings.is_empty() {
        return Err(HarnessError::Validation(findings));
    }
    let model_id = options
        .model_id
        .clone()
        .unwrap_or_else(|| spec.model.id.clone());
    let tier = options.tier.clone().or_else(|| spec.model.tier.clone());
    // `include_reference: false` (sc-23402): a generator dispatches `text_to_image` on this one
    // model and nothing else, so it never resolves — and must never gate on — a family's reference
    // partition. The plates it renders BECOME references; it does not condition on any.
    let catalog = plan_catalog_for(transport, &model_id, false).await?;
    let entry = catalog.base_entry();
    let facts = host_facts_for(transport).await?;
    let mut findings = model_findings(&spec, &model_id, tier.as_deref(), entry, options);
    let lane = facts.lane();
    // Geometry per role, which is also the last thing that can be answered without a render.
    let mut geometry: BTreeMap<String, (u32, u32)> = BTreeMap::new();
    if let Some(entry) = entry {
        for (index, role) in spec.references.iter().enumerate() {
            match resolve_geometry(&spec, role, entry) {
                Ok(size) => {
                    geometry.insert(role.role.clone(), size);
                }
                Err(finding) => findings.push(PlanDiagnostic::plan(
                    format!("referenceSpec.references[{index}].resolution"),
                    finding,
                )),
            }
        }
    }
    findings.extend(host_findings(&spec, &facts, entry, lane, transport).await?);
    if !findings.is_empty() {
        return Err(HarnessError::Validation(findings));
    }

    // The run id is minted here, not inside the render, because the PENDING DIRECTORY is named
    // after it: two runs against the same `--out` must not land in one directory and destroy each
    // other's in-flight work.
    let run_id = format!(
        "makerefs-{}-{}",
        utc_now()
            .chars()
            .filter(char::is_ascii_alphanumeric)
            .collect::<String>(),
        &uuid::Uuid::new_v4().simple().to_string()[..8]
    );
    let pending = pending_dir(&options.out_dir, &run_id, options.force)?;
    let result = generate_into(
        transport,
        options,
        &spec,
        &spec_dir,
        &inherited,
        &Resolved {
            model_id: model_id.clone(),
            tier: tier.clone(),
            geometry,
        },
        &pending,
        &run_id,
        started,
    )
    .await;
    match result {
        Ok(build) => Ok(build),
        Err(error) => {
            // A pack is published or it does not exist: nothing a failed run wrote survives, so
            // `--out` can never be a half-written pack.
            let _ = std::fs::remove_dir_all(&pending);
            Err(error)
        }
    }
}

/// The model id, tier, catalog entry and per-role geometry a run renders with, all resolved.
struct Resolved {
    model_id: String,
    tier: Option<String>,
    geometry: BTreeMap<String, (u32, u32)>,
}

/// The roles and sound a spec copies rather than renders, already read off the source pack.
#[derive(Debug, Clone, Default)]
struct Inherited {
    /// Pack directory the files are copied from.
    source_dir: PathBuf,
    references: Vec<ReferenceEntry>,
    sound: Vec<film_plan::SoundEntry>,
}

/// Read the spec's source pack and pick out the roles (and sound) it inherits. Findings name the
/// role rather than the file, because a role the source pack does not hold is a spec error.
fn resolve_inherited(spec: &ReferenceSpec, spec_dir: &Path) -> (Inherited, Vec<PlanDiagnostic>) {
    let Some(inherit) = spec.inherit.as_ref() else {
        return (Inherited::default(), Vec::new());
    };
    let pack_path = spec_dir.join(&inherit.pack);
    let pack = match film_plan::read_reference_pack_file(&pack_path) {
        Ok(pack) => pack,
        Err(finding) => {
            return (
                Inherited::default(),
                vec![PlanDiagnostic::plan(
                    "referenceSpec.inherit.pack",
                    finding.message,
                )],
            )
        }
    };
    let source_dir = pack_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let mut findings = film_plan::validate_reference_pack(&pack);
    let mut references = Vec::new();
    for role in &inherit.references {
        match pack
            .references
            .iter()
            .find(|entry| &entry.role == role)
            .cloned()
        {
            Some(entry) => references.push(entry),
            None => findings.push(PlanDiagnostic::plan(
                "referenceSpec.inherit.references",
                format!(
                    "role {role:?} is not in {} — an inherited role must exist in the pack it is \
                     copied from",
                    pack_path.display()
                ),
            )),
        }
    }
    let sound = if inherit.sound {
        pack.sound.clone()
    } else {
        Vec::new()
    };
    for (file, what) in references
        .iter()
        .map(|entry| (entry.file.clone(), format!("reference {:?}", entry.role)))
        .chain(sound.iter().filter_map(|entry| {
            // A synthesized line (sc-23404) has no clip on disk to inherit — `ensure_sound` speaks
            // it and overwrites `file` with what synthesis wrote. An entry that pins a filename
            // alongside its `text` is synthesized INTO that name, so there is still nothing here
            // to require on disk.
            entry
                .file
                .clone()
                .filter(|_| !entry.is_synthesized())
                .map(|file| (file, format!("sound {:?}", entry.role)))
        }))
    {
        let path = source_dir.join(&file);
        match std::fs::metadata(&path) {
            Ok(metadata) if metadata.is_file() && metadata.len() > 0 => {}
            _ => findings.push(PlanDiagnostic::plan(
                "referenceSpec.inherit",
                format!(
                    "inherited {what}: {} is missing or empty, so it cannot be copied",
                    path.display()
                ),
            )),
        }
    }
    (
        Inherited {
            source_dir,
            references,
            sound,
        },
        findings,
    )
}

/// Findings about the catalog ENTRY the spec names: present, an image model, able to generate from
/// text, installed, and — the one that would otherwise be discovered as a silently dropped field —
/// able to take the negative prompt the spec declares.
fn model_findings(
    spec: &ReferenceSpec,
    model_id: &str,
    tier: Option<&str>,
    entry: Option<&JsonObject<String, Value>>,
    options: &MakeReferencesOptions,
) -> Vec<PlanDiagnostic> {
    let Some(entry) = entry else {
        return vec![PlanDiagnostic::plan(
            "referenceSpec.model.id",
            format!("{model_id:?} is not in this API's model catalog"),
        )];
    };
    let mut findings = Vec::new();
    if entry.get("type").and_then(Value::as_str) != Some("image") {
        findings.push(PlanDiagnostic::plan(
            "referenceSpec.model.id",
            format!("{model_id:?} is not an image model"),
        ));
    }
    let capable = entry
        .get("capabilities")
        .and_then(Value::as_array)
        .is_some_and(|caps| caps.iter().any(|cap| cap == spec.model.mode.as_str()));
    if !capable {
        findings.push(PlanDiagnostic::plan(
            "referenceSpec.model.mode",
            format!(
                "{model_id} does not declare the {:?} capability",
                spec.model.mode
            ),
        ));
    }
    // The TIER, against the catalog entry's own variant list — independently of the install gate,
    // because `mlx_quantize_for_tier` maps anything it does not recognise to q4. A `--tier q6` that
    // reached dispatch would render q4 while the pack's provenance recorded "q6", which is a lie
    // that survives the run.
    if let Some(tier) = tier {
        if let Some(variants) = entry.get("variants").and_then(Value::as_array) {
            let declared: Vec<&str> = variants
                .iter()
                .filter_map(|variant| variant.get("variant").and_then(Value::as_str))
                .collect();
            if !declared.contains(&tier) {
                findings.push(PlanDiagnostic::plan(
                    "referenceSpec.model.tier",
                    format!(
                        "{model_id} does not declare tier {tier:?}; its tiers are {}",
                        match declared.is_empty() {
                            true => "none".to_owned(),
                            false => declared.join(", "),
                        }
                    ),
                ));
            }
        }
    }
    if options.require_installed && !model_tier_installed(entry, tier) {
        findings.push(PlanDiagnostic::plan(
            "referenceSpec.model.tier",
            format!(
                "{model_id}{} is not installed on this host (catalog installState is not \
                 \"installed\"); download it in the Model Manager first, or pass \
                 --skip-install-check",
                tier.map(|tier| format!(" tier {tier}")).unwrap_or_default()
            ),
        ));
    }
    // `image.supportsNegativePrompt` is ABSENT-means-TRUE, the same polarity the video block has.
    // Krea 2 Turbo is CFG-free and declares it false, and the engine drops the text — so a spec
    // that declares one for it is refused rather than rendered against a prompt the model never
    // saw.
    let supports_negative = entry
        .get("image")
        .and_then(|image| image.get("supportsNegativePrompt"))
        .and_then(Value::as_bool)
        .unwrap_or(true);
    if !supports_negative {
        for (index, role) in spec.references.iter().enumerate() {
            if role.negative_prompt_with(spec).is_some() {
                findings.push(PlanDiagnostic::plan(
                    format!("referenceSpec.references[{index}].negativePrompt"),
                    format!(
                        "role {:?} declares a negative prompt, but {model_id} declares \
                         image.supportsNegativePrompt: false and the engine never forwards one",
                        role.role
                    ),
                ));
            }
        }
    }
    findings
}

/// Findings about the HOST: somebody live to claim an image job, and a memory budget that is both
/// inside what the host reports AND at or above what the model declares it needs on this lane.
///
/// Both halves matter. A budget over the host's memory can never be met; a budget under the
/// model's own `<lane>.minMemoryGb` clears every pre-dispatch gate, pays a full GPU render, and
/// only then refuses on the observed peak — which is the expensive way to learn it.
async fn host_findings(
    spec: &ReferenceSpec,
    facts: &super::HostFacts,
    entry: Option<&JsonObject<String, Value>>,
    lane: film_plan::ModelLane,
    transport: &dyn ApiTransport,
) -> Result<Vec<PlanDiagnostic>, HarnessError> {
    let mut findings = Vec::new();
    let workers = super::expect_ok_on(transport, "GET", "/api/v1/workers", None).await?;
    let advert = live_worker_advertising(&workers, IMAGE_CAPABILITY);
    if advert.live.is_none() {
        findings.push(PlanDiagnostic::plan(
            "referenceSpec.model.id",
            format!(
                "no live registered worker advertises {IMAGE_CAPABILITY}{}; start the GPU worker \
                 (SCENEWORKS_WORKER_ONLY=1) and wait for it to register",
                stale_workers_detail(&advert.stale)
            ),
        ));
    }
    match facts.host_memory_gb() {
        Some(host) if spec.limits.max_memory_gb > host => findings.push(PlanDiagnostic::plan(
            "referenceSpec.limits.maxMemoryGb",
            format!(
                "budget {} GB exceeds the {host:.1} GB the registered worker reports for this host",
                spec.limits.max_memory_gb
            ),
        )),
        Some(_) => {}
        None => findings.push(PlanDiagnostic::plan(
            "referenceSpec.limits.maxMemoryGb",
            "no registered worker reports host memory, so the memory budget cannot be checked \
             before dispatch",
        )),
    }
    if let Some(minimum) = entry.and_then(|entry| film_plan::model_min_memory_gb(entry, lane)) {
        if spec.limits.max_memory_gb < minimum {
            findings.push(PlanDiagnostic::plan(
                "referenceSpec.limits.maxMemoryGb",
                format!(
                    "budget {} GB is below {}'s declared {}.minMemoryGb of {minimum} GB, so every \
                     plate would be rendered and then refused on its observed peak",
                    spec.limits.max_memory_gb,
                    entry
                        .and_then(|entry| entry.get("id"))
                        .and_then(Value::as_str)
                        .unwrap_or("the model"),
                    lane.manifest_key()
                ),
            ));
        }
    }
    Ok(findings)
}

/// The geometry one role renders at: its own resolution, else the spec's, else the model's declared
/// default — and, when the entry declares a resolution menu, one the menu admits.
fn resolve_geometry(
    spec: &ReferenceSpec,
    role: &ReferenceSpecEntry,
    entry: &JsonObject<String, Value>,
) -> Result<(u32, u32), String> {
    let declared = role.resolution_with(spec).map(str::to_owned).or_else(|| {
        entry
            .get("defaults")
            .and_then(|defaults| defaults.get("resolution"))
            .and_then(Value::as_str)
            .map(str::to_owned)
    });
    let Some(declared) = declared else {
        return Err(format!(
            "role {:?} declares no resolution and {} declares no default one",
            role.role,
            entry
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or("the model")
        ));
    };
    let size = film_plan::parse_resolution(&declared)
        .ok_or_else(|| format!("resolution {declared:?} must be \"WxH\""))?;
    if let Some(menu) = entry
        .get("limits")
        .and_then(|limits| limits.get("resolutions"))
        .and_then(Value::as_array)
    {
        let admitted = menu.iter().filter_map(Value::as_str).any(|candidate| {
            film_plan::parse_resolution(candidate).is_some_and(|value| value == size)
        });
        if !admitted {
            return Err(format!(
                "role {:?} renders at {declared}, which {} does not declare; its menu is {}",
                role.role,
                entry
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("the model"),
                menu.iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    }
    Ok(size)
}

/// The directory a run writes into before publishing. A sibling of `--out`, so the publish is a
/// rename on one filesystem rather than a copy, and named after THIS run, so two runs against the
/// same `--out` cannot write into one directory.
fn pending_dir(out_dir: &Path, run_id: &str, force: bool) -> Result<PathBuf, HarnessError> {
    if out_dir.exists() {
        let occupied = std::fs::read_dir(out_dir)
            .map(|mut entries| entries.next().is_some())
            .unwrap_or(true);
        if occupied {
            match force {
                false => {
                    return Err(HarnessError::Refused(format!(
                        "{} already exists and is not empty; pass --force to replace it, or pick \
                         another --out",
                        out_dir.display()
                    )))
                }
                // `--force` replaces a PACK. It is not a licence to remove whatever `--out` names:
                // pointed at the plan directory it would otherwise take plan.jsonc, brief.jsonc and
                // the spec itself with it.
                true => assert_replaceable_pack(out_dir)?,
            }
        }
    }
    let name = pack_dir_name(out_dir);
    let parent = out_dir
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    std::fs::create_dir_all(&parent)?;
    let pending = parent.join(format!(".{name}.{PENDING_INFIX}.{run_id}"));
    if pending.exists() {
        // The run id is unique to this run, so an existing one is another process's in-flight work
        // and not ours to remove.
        return Err(HarnessError::Refused(format!(
            "{} already exists; another run is writing it",
            pending.display()
        )));
    }
    std::fs::create_dir_all(&pending)?;
    Ok(pending)
}

/// The `--out` directory's own name, for the hidden siblings that are named after it.
fn pack_dir_name(out_dir: &Path) -> &str {
    out_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("references")
}

/// Refuse `--force` unless `out_dir` really is a reference pack: it holds a readable
/// [`REFERENCE_PACK_FILE`], and every other top-level entry is one the pack's own `file` paths
/// declare. The refusal names the stray entry, because the useful answer to "why not" is which file
/// would have been destroyed.
fn assert_replaceable_pack(out_dir: &Path) -> Result<(), HarnessError> {
    let pack_path = out_dir.join(REFERENCE_PACK_FILE);
    let refuse = |detail: String| {
        HarnessError::Refused(format!(
            "--force replaces a reference pack, and {} is not one: {detail}. Point --out at a pack \
             directory (or at a new one); nothing was removed",
            out_dir.display()
        ))
    };
    if !pack_path.is_file() {
        return Err(refuse(format!("it holds no {REFERENCE_PACK_FILE}")));
    }
    let pack = film_plan::read_reference_pack_file(&pack_path).map_err(|finding| {
        refuse(format!(
            "{REFERENCE_PACK_FILE} does not parse ({})",
            finding.message
        ))
    })?;
    // The top-level segment of every path the pack declares — `references/courier.png` declares
    // `references`. Anything else at the top level is not part of this pack.
    let declared: std::collections::BTreeSet<&str> = pack
        .references
        .iter()
        .map(|entry| entry.file.as_str())
        // A synthesized line may declare no `file` at all (sc-23404); one that pins a filename
        // still declares that path, so it is not swept as a stray.
        .chain(pack.sound.iter().filter_map(|entry| entry.file.as_deref()))
        .filter_map(|file| file.split('/').next())
        .collect();
    for entry in std::fs::read_dir(out_dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name == REFERENCE_PACK_FILE || declared.contains(name.as_str()) {
            continue;
        }
        return Err(refuse(format!(
            "it holds {name}, which the pack in it does not declare"
        )));
    }
    Ok(())
}

/// Render every role into `pending`, copy the inherited files, write the document, and rename the
/// directory onto `--out`. Every error path leaves `pending` to the caller to remove.
#[allow(clippy::too_many_arguments)]
async fn generate_into(
    transport: &dyn ApiTransport,
    options: &MakeReferencesOptions,
    spec: &ReferenceSpec,
    spec_dir: &Path,
    inherited: &Inherited,
    resolved: &Resolved,
    pending: &Path,
    run_id: &str,
    started: Instant,
) -> Result<ReferencePackBuild, HarnessError> {
    let client = Client {
        transport,
        control: &options.control,
    };
    let project_id = ensure_project(&client, options, spec, run_id).await?;
    let mut entries = Vec::new();
    let mut plates = Vec::new();
    for (index, role) in spec.references.iter().enumerate() {
        let (width, height) = *resolved
            .geometry
            .get(&role.role)
            .expect("every role's geometry was resolved before dispatch");
        let seed = role
            .seed
            .or_else(|| spec.seed_base.map(|base| base.saturating_add(index as i64)));
        let plate = render_role(
            &client,
            options,
            spec,
            resolved,
            &project_id,
            run_id,
            role,
            seed,
            (width, height),
            pending,
        )
        .await?;
        entries.push(ReferenceEntry {
            role: role.role.clone(),
            kind: role.kind.clone(),
            file: role.file.clone(),
            source_asset_id: None,
            description: role.description.clone(),
            approved: true,
            generated: true,
            generation: Some(GeneratedReference {
                model: resolved.model_id.clone(),
                tier: resolved.tier.clone(),
                backend: plate.backend.clone(),
                mode: spec.model.mode.clone(),
                prompt: role.prompt.clone(),
                negative_prompt: role.negative_prompt_with(spec).map(str::to_owned),
                seed: plate.seed,
                width: plate.width,
                height: plate.height,
                job_id: plate.job_id.clone(),
                asset_id: plate.asset_id.clone(),
                sha256: plate.sha256.clone(),
                created_at: utc_now(),
            }),
        });
        plates.push(plate);
    }
    // The inherited plates and sound, copied verbatim. They keep their own `generated` flag: a
    // supplied plate stays a supplied plate when a generated pack carries it forward.
    let mut inherited_roles = Vec::new();
    for entry in &inherited.references {
        copy_into(
            &inherited.source_dir.join(&entry.file),
            pending,
            &entry.file,
        )?;
        inherited_roles.push(entry.role.clone());
        entries.push(entry.clone());
    }
    for entry in &inherited.sound {
        // Same rule as the inherit-time existence check above: a synthesized line has no clip on
        // disk to carry forward, because the inheriting run speaks it itself (sc-23404).
        let Some(file) = entry.file.as_deref().filter(|_| !entry.is_synthesized()) else {
            continue;
        };
        copy_into(&inherited.source_dir.join(file), pending, file)?;
    }
    let pack = ReferencePack {
        schema_version: film_plan::REFERENCE_PACK_SCHEMA_VERSION,
        id: spec.id.clone(),
        version: spec.version,
        description: spec.description.clone(),
        references: entries,
        sound: inherited.sound.clone(),
    };
    let findings = film_plan::validate_reference_pack(&pack);
    if !findings.is_empty() {
        return Err(HarnessError::Validation(findings));
    }
    let document = pack_document(&pack, spec, spec_dir, options, resolved, &project_id)?;
    std::fs::write(pending.join(REFERENCE_PACK_FILE), document)?;
    let findings = film_plan::validate_reference_pack_files(&pack, pending);
    if !findings.is_empty() {
        return Err(HarnessError::Validation(findings));
    }
    // Publish. Only now does `--out` exist at all (or, with --force, change) — and a --force
    // replacement is a rename of the old pack ASIDE, a rename of the new one IN, and only then a
    // removal of the old one. The window in which `--out` does not name a complete pack is the one
    // rename between the two, not the whole removal of the pack that was there.
    let replaced = match options.out_dir.exists() {
        false => None,
        true => {
            let aside = options.out_dir.with_file_name(format!(
                ".{}.{REPLACED_INFIX}.{run_id}",
                pack_dir_name(&options.out_dir)
            ));
            std::fs::rename(&options.out_dir, &aside)?;
            Some(aside)
        }
    };
    if let Err(error) = std::fs::rename(pending, &options.out_dir) {
        // Put the pack that was there back: a failed publish must not have cost the user the pack
        // it was replacing.
        if let Some(aside) = replaced.as_ref() {
            let _ = std::fs::rename(aside, &options.out_dir);
        }
        return Err(error.into());
    }
    if let Some(aside) = replaced.as_ref() {
        // The new pack is ALREADY published at this point, so an error here is reported against a
        // complete `--out`: what it says is that the displaced pack is still on disk under the
        // path in the message and wants removing by hand. It is an error rather than a shrug
        // because a `--force` that silently leaves the old pack behind is how a directory fills up
        // with packs nobody knows the provenance of.
        std::fs::remove_dir_all(aside)?;
    }
    Ok(ReferencePackBuild {
        pack_path: options.out_dir.join(REFERENCE_PACK_FILE),
        pack,
        generated: plates,
        inherited: inherited_roles,
        project_id,
        model_id: resolved.model_id.clone(),
        tier: resolved.tier.clone(),
        elapsed_seconds: started.elapsed().as_secs_f64(),
    })
}

/// The project the plates are generated in. `--project-id` reuses one; otherwise a project named
/// after the spec and this run, so two runs never share one.
async fn ensure_project(
    client: &Client<'_>,
    options: &MakeReferencesOptions,
    spec: &ReferenceSpec,
    run_id: &str,
) -> Result<String, HarnessError> {
    if let Some(project_id) = options.project_id.clone() {
        client
            .expect_ok("GET", &format!("/api/v1/projects/{project_id}"), None)
            .await?;
        return Ok(project_id);
    }
    let project = client
        .expect_ok(
            "POST",
            "/api/v1/projects",
            Some(json!({ "name": format!("{} references ({run_id})", spec.id) })),
        )
        .await?;
    project
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| HarnessError::Transport(format!("project response has no id: {project}")))
}

/// Render ONE role, within the spec's declared attempts and per-job budget, and write the plate.
#[allow(clippy::too_many_arguments)]
async fn render_role(
    client: &Client<'_>,
    options: &MakeReferencesOptions,
    spec: &ReferenceSpec,
    resolved: &Resolved,
    project_id: &str,
    run_id: &str,
    role: &ReferenceSpecEntry,
    seed: Option<i64>,
    (width, height): (u32, u32),
    pending: &Path,
) -> Result<GeneratedPlate, HarnessError> {
    let started = Instant::now();
    let mut last_error = String::new();
    for attempt in 1..=spec.limits.max_attempts_per_role {
        if options.control.is_canceled() {
            return Err(HarnessError::Refused(format!(
                "canceled before role {:?} finished; nothing was published",
                role.role
            )));
        }
        let body = job_body(
            spec,
            resolved,
            project_id,
            run_id,
            role,
            seed,
            (width, height),
            attempt,
        );
        let response = client.expect_ok("POST", IMAGE_ROUTE, Some(body)).await?;
        let job_id = response
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                HarnessError::Transport(format!("image job response has no id: {response}"))
            })?
            .to_owned();
        let budget = Duration::from_secs(spec.limits.max_job_seconds);
        let deadline = Instant::now() + budget;
        let (view, stop) = client
            .wait_for_job(
                &job_id,
                PollBounds {
                    shot_deadline: deadline,
                    run_deadline: None,
                    poll_interval: options.poll_interval,
                    cancel_grace: CANCEL_GRACE.min(budget),
                    settle_grace: super::ASSET_SETTLE_GRACE.min(budget),
                },
            )
            .await?;
        // Whether the cancel `wait_for_job` posted was actually honoured. `wait_for_job` returns
        // the last view it observed whether or not the job went terminal, so a job still running
        // after the grace is a render in flight that nothing here can stop.
        let cancel_honoured = view.is_terminal();
        match stop {
            PollStop::Terminal if view.status == "completed" => {}
            PollStop::Operator => {
                return Err(HarnessError::Refused(format!(
                    "canceled while role {:?} was rendering (job {job_id}); nothing was published",
                    role.role
                )))
            }
            // The same halt the video run takes (`cancel_not_honoured`): the next attempt may NOT
            // go out beside a render still on the GPU. The spec declared ONE memory budget, and two
            // renders in flight is exactly what it is there to prevent.
            PollStop::ShotBudget | PollStop::RunBudget if !cancel_honoured => {
                return Err(HarnessError::Refused(format!(
                    "role {:?}'s job {job_id} was still {} after the cancel grace, so a render is \
                     in flight that nothing here can stop; no further dispatch and nothing was \
                     published",
                    role.role, view.status
                )))
            }
            PollStop::ShotBudget | PollStop::RunBudget => {
                last_error = format!(
                    "job {job_id} did not finish within the spec's maxJobSeconds of {}s; it was \
                     canceled",
                    spec.limits.max_job_seconds
                );
                continue;
            }
            PollStop::AssetsUnsettled => {
                last_error = format!(
                    "job {job_id} finished but the API never published its asset (the result still \
                     carries raw assetWrites): {}",
                    view.result
                );
                continue;
            }
            PollStop::Terminal => {
                last_error = format!(
                    "job {job_id} ended {}: {}",
                    view.status,
                    view.failure_text()
                );
                continue;
            }
        }
        // The memory budget, read off the metrics block the worker posts after the terminal
        // progress — the same signal the video run's per-attempt check reads. Over-budget is
        // terminal for the whole run: a retry would re-render at the same cost.
        let peak_memory_gb = job_peak_memory_bytes(client.transport, &job_id)
            .await
            .map(|bytes| bytes as f64 / BYTES_PER_GB);
        if peak_memory_gb.is_some_and(|observed| observed > spec.limits.max_memory_gb) {
            return Err(HarnessError::Refused(format!(
                "role {:?} peaked at {:.1} GB, over the spec's {} GB budget; raise \
                 limits.maxMemoryGb or pick a cheaper tier",
                role.role,
                peak_memory_gb.unwrap_or_default(),
                spec.limits.max_memory_gb
            )));
        }
        let asset = view
            .result
            .get("assets")
            .and_then(Value::as_array)
            .and_then(|assets| assets.first())
            .cloned()
            .ok_or_else(|| {
                HarnessError::Transport(format!(
                    "image job {job_id} completed without an asset: {}",
                    view.result
                ))
            })?;
        let asset_id = asset
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                HarnessError::Transport(format!("image job {job_id} asset has no id: {asset}"))
            })?
            .to_owned();
        let media_path = asset
            .pointer("/file/path")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let bytes = download_media(client.transport, project_id, &media_path).await?;
        let path = pending.join(&role.file);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, &bytes)?;
        return Ok(GeneratedPlate {
            role: role.role.clone(),
            file: role.file.clone(),
            job_id,
            asset_id,
            // The seed the RESULT reports, which is the one that rendered: with no seed named the
            // route picks one, and recording what we asked for would record nothing.
            seed: asset
                .pointer("/recipe/seed")
                .and_then(Value::as_i64)
                .or(seed),
            width: asset
                .pointer("/file/width")
                .and_then(Value::as_u64)
                .and_then(|value| u32::try_from(value).ok())
                .unwrap_or(width),
            height: asset
                .pointer("/file/height")
                .and_then(Value::as_u64)
                .and_then(|value| u32::try_from(value).ok())
                .unwrap_or(height),
            sha256: sha256_hex(&bytes),
            bytes: bytes.len(),
            backend: view.backend.clone(),
            peak_memory_gb,
            elapsed_seconds: started.elapsed().as_secs_f64(),
        });
    }
    Err(HarnessError::Refused(format!(
        "role {:?} did not render in {} attempt(s): {last_error}",
        role.role, spec.limits.max_attempts_per_role
    )))
}

/// The image job body one role is dispatched with.
#[allow(clippy::too_many_arguments)]
fn job_body(
    spec: &ReferenceSpec,
    resolved: &Resolved,
    project_id: &str,
    run_id: &str,
    role: &ReferenceSpecEntry,
    seed: Option<i64>,
    (width, height): (u32, u32),
    attempt: u32,
) -> Value {
    let mut advanced = JsonObject::new();
    advanced.insert(
        "filmHarness".to_owned(),
        json!({
            "kind": "reference",
            "role": role.role,
            "specId": spec.id,
            "specVersion": spec.version,
            "runId": run_id,
            "attempt": attempt,
        }),
    );
    if let Some(tier) = resolved.tier.as_deref() {
        // The same `advanced.mlxQuantize` convention the compiled video requests use for a tier.
        advanced.insert("mlxQuantize".to_owned(), mlx_quantize_for_tier(tier));
    }
    let mut body = json!({
        "projectId": project_id,
        "mode": spec.model.mode,
        "model": resolved.model_id,
        "prompt": role.prompt,
        "count": 1,
        "width": width,
        "height": height,
        "advanced": Value::Object(advanced),
    });
    let object = body.as_object_mut().expect("object literal");
    if let Some(seed) = seed {
        object.insert("seed".to_owned(), json!(seed));
    }
    if let Some(negative) = role.negative_prompt_with(spec) {
        object.insert("negativePrompt".to_owned(), json!(negative));
    }
    body
}

/// Download one rendered plate over the API.
///
/// Through the transport, not off the disk: `--api` may point at a SceneWorks API elsewhere on the
/// private network, whose project directory this process cannot read.
async fn download_media(
    transport: &dyn ApiTransport,
    project_id: &str,
    media_path: &str,
) -> Result<Vec<u8>, HarnessError> {
    if !is_safe_media_path(media_path) {
        return Err(HarnessError::Transport(format!(
            "the API reported the plate at {media_path:?}, which is not a plain relative media \
             path inside the project"
        )));
    }
    let path = format!("/api/v1/projects/{project_id}/files/{media_path}");
    let response = transport.get_bytes(path.clone()).await?;
    if !(200..300).contains(&response.status) {
        return Err(HarnessError::Api {
            method: "GET",
            path,
            status: response.status,
            detail: api_detail(&serde_json::from_slice(&response.bytes).unwrap_or(Value::Null)),
        });
    }
    if response.bytes.is_empty() {
        return Err(HarnessError::Transport(format!(
            "GET {path} returned an empty file"
        )));
    }
    Ok(response.bytes)
}

/// A project-relative media path that is safe to interpolate into the file route: no absolute
/// path, no `..`, no escaping or encoding needed.
fn is_safe_media_path(path: &str) -> bool {
    !path.is_empty()
        && !path.starts_with('/')
        && !path.split('/').any(|segment| {
            segment.is_empty()
                || segment == ".."
                || segment == "."
                || !segment
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
        })
}

/// Copy one inherited file into the pending pack at the same relative path.
fn copy_into(source: &Path, pending: &Path, relative: &str) -> Result<(), HarnessError> {
    let destination = pending.join(relative);
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::copy(source, &destination).map_err(|error| {
        HarnessError::Io(format!(
            "cannot copy {} to {}: {error}",
            source.display(),
            destination.display()
        ))
    })?;
    Ok(())
}

/// The pack document, with a header saying what produced it. A reader who opens a generated pack
/// should not have to infer that from the entries.
fn pack_document(
    pack: &ReferencePack,
    spec: &ReferenceSpec,
    spec_dir: &Path,
    options: &MakeReferencesOptions,
    resolved: &Resolved,
    project_id: &str,
) -> Result<String, HarnessError> {
    let body = serde_json::to_string_pretty(pack).map_err(|error| {
        HarnessError::Io(format!("cannot serialize the reference pack: {error}"))
    })?;
    let spec_name = options
        .spec_path
        .strip_prefix(spec_dir)
        .unwrap_or(&options.spec_path)
        .display()
        .to_string();
    Ok(format!(
        "// GENERATED TEST FIXTURES — written by `film-harness make-references` (sc-23403).\n\
         //\n\
         // Spec:    {spec_name} (id {:?} v{})\n\
         // Model:   {}{}\n\
         // Project: {project_id} (the generated assets and their jobs live there)\n\
         // Written: {}\n\
         //\n\
         // In the product a person supplies the reference images; these were generated so the film\n\
         // harness has its own fixtures. Every generated entry carries `generated: true` and the\n\
         // provenance it came from. Replace a file with a real approved plate, drop that entry's\n\
         // `generated`/`generation` keys and bump `version`; the roles and the plan do not change.\n\
         {body}\n",
        spec.id,
        spec.version,
        resolved.model_id,
        resolved
            .tier
            .as_deref()
            .map(|tier| format!(" (tier {tier})"))
            .unwrap_or_default(),
        utc_now(),
    ))
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn a_media_path_that_could_escape_the_project_is_refused() {
        assert!(is_safe_media_path("assets/images/genset_1/plate.png"));
        assert!(!is_safe_media_path("/etc/passwd"));
        assert!(!is_safe_media_path("assets/../../etc/passwd"));
        assert!(!is_safe_media_path("assets//plate.png"));
        assert!(!is_safe_media_path("assets/im ages/plate.png"));
        assert!(!is_safe_media_path("assets/plate.png?ticket=1"));
        assert!(!is_safe_media_path(""));
    }

    /// A minimal pack document that `read_reference_pack_file` accepts, declaring one plate under
    /// `references/` — so a directory holding it plus `references/` is a replaceable pack.
    fn write_pack(dir: &Path) {
        std::fs::create_dir_all(dir.join("references")).expect("references dir");
        std::fs::write(dir.join("references/plate.png"), b"\x89PNG").expect("plate");
        std::fs::write(
            dir.join(REFERENCE_PACK_FILE),
            serde_json::to_string_pretty(&json!({
                "schemaVersion": film_plan::REFERENCE_PACK_SCHEMA_VERSION,
                "id": "pack-under-test",
                "version": 1,
                "references": [{
                    "role": "courier",
                    "kind": "character",
                    "file": "references/plate.png",
                    "approved": true,
                }],
            }))
            .expect("pack serializes"),
        )
        .expect("pack document");
    }

    #[test]
    fn the_pending_directory_is_a_hidden_sibling_of_out_named_after_the_run() {
        let temp = tempfile::tempdir().expect("temp dir");
        let out = temp.path().join("pack");
        let pending = pending_dir(&out, "makerefs-run-a", false).expect("pending dir creates");
        assert_eq!(pending.parent(), Some(temp.path()));
        assert!(pending.is_dir());
        assert!(
            !out.exists(),
            "--out must not exist until the run publishes"
        );
        // A SECOND run against the same `--out` gets its OWN directory and leaves the first one's
        // in-flight work alone — two concurrent runs used to share one name and remove each other.
        std::fs::write(pending.join("plate.png"), b"in flight").expect("in-flight file");
        let second = pending_dir(&out, "makerefs-run-b", false).expect("second pending dir");
        assert_ne!(
            second, pending,
            "two runs must not share a pending directory"
        );
        assert!(
            pending.join("plate.png").is_file(),
            "the second run removed the first run's in-flight work"
        );
    }

    #[test]
    fn a_non_empty_out_directory_is_refused_rather_than_overwritten() {
        let temp = tempfile::tempdir().expect("temp dir");
        let out = temp.path().join("pack");
        write_pack(&out);
        let error =
            pending_dir(&out, "makerefs-run-a", false).expect_err("a populated --out is refused");
        assert!(
            matches!(&error, HarnessError::Refused(message) if message.contains("--force")),
            "{error}"
        );
        pending_dir(&out, "makerefs-run-b", true).expect("--force accepts a populated pack");
        assert!(
            out.join(REFERENCE_PACK_FILE).is_file(),
            "--force must not remove the published pack before the new one is ready"
        );
    }

    /// `--force` on a directory that is NOT a pack is refused by name. Pointed at the plan
    /// directory (`--out config/film-harness/courier-workshop`) the old code removed plan.jsonc,
    /// brief.jsonc, review.jsonc and the spec it was reading.
    #[test]
    fn force_is_refused_on_a_directory_that_is_not_a_reference_pack() {
        let temp = tempfile::tempdir().expect("temp dir");

        // (1) A plan directory: a pack document, but beside it a plan the pack does not declare.
        let plan_dir = temp.path().join("courier-workshop");
        write_pack(&plan_dir);
        std::fs::write(plan_dir.join("plan.jsonc"), "{ \"id\": \"courier\" }").expect("plan");
        let error = pending_dir(&plan_dir, "makerefs-run-a", true)
            .expect_err("--force on a directory holding a plan is refused");
        assert!(
            matches!(&error, HarnessError::Refused(message)
                if message.contains("plan.jsonc") && message.contains("does not declare")),
            "the refusal must name the stray entry: {error}"
        );
        assert!(
            plan_dir.join("plan.jsonc").is_file() && plan_dir.join(REFERENCE_PACK_FILE).is_file(),
            "a refused --force must remove nothing"
        );

        // (2) No pack document at all.
        let stranger = temp.path().join("not-a-pack");
        std::fs::create_dir_all(&stranger).expect("dir");
        std::fs::write(stranger.join("notes.txt"), "mine").expect("file");
        let error = pending_dir(&stranger, "makerefs-run-a", true)
            .expect_err("--force on a directory with no pack document is refused");
        assert!(
            matches!(&error, HarnessError::Refused(message)
                if message.contains(REFERENCE_PACK_FILE)),
            "{error}"
        );
        assert!(stranger.join("notes.txt").is_file(), "nothing was removed");
    }
}
