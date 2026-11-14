//! YuE2 Song Lab job and immutable score-version generation options.
//! FIELD_KINDS is the source authority for per-kind acceptance; source metadata is internal.
//! No serde aliases are declared: UI setting names normalize to these exact camelCase paths.

use crate::*;

pub(crate) const SPECS: &[OptionSpec] = &[
    OptionSpec::new(
        "yue2.kind",
        "yue2",
        "kind",
        Shape::OneOf("create | plan | fromPlan | cover | renderVersion | decode | transcribe"),
        DefaultRule::Required,
        IntentClass::WhatToMake,
        "Yue2JobSpec.kind",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: Yue2JobSpec; FIELD_KINDS rejects controls on kinds that do not consume them",
    ])
    .applies(Predicate::Family("yue2"))
    .effects(&[
        Effect { when: "plan", action: "publish score plan without synthesis", target: "SongParams.plan_only" },
        Effect { when: "renderVersion", action: "regenerate entire recording from immutable score version request", target: "GenerationRequest" },
        Effect { when: "decode", action: "decode cached latents without generating a new song", target: "SongParams.cached_latents" },
        Effect { when: "transcribe", action: "transcribe a recording for reviewed score versions", target: "SheetSage2 review" },
    ])
    .ui("Kind", "select"),
    OptionSpec::new(
        "yue2.style",
        "yue2",
        "style",
        Shape::Text,
        DefaultRule::Inherit("saved plan style on fromPlan; otherwise empty string"),
        IntentClass::WhatToMake,
        "GenerationRequest.prompt",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: <=4000 Unicode characters",
        "apps/rust-api/src/yue2_jobs.rs: fromPlan text must match saved plan; renderVersion uses version text",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds create, plan, fromPlan, cover",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [create, plan, fromPlan, cover]")]))
    .ui("Style", "text"),
    OptionSpec::new(
        "yue2.lyrics",
        "yue2",
        "lyrics",
        Shape::Text,
        DefaultRule::Inherit("saved plan lyrics on fromPlan; otherwise required for create/plan/cover"),
        IntentClass::WhatToMake,
        "AudioParams.lyrics",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: <=16000 Unicode characters; nonblank required for create/plan/cover",
        "apps/rust-api/src/yue2_jobs.rs: fromPlan text must match saved plan",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds create, plan, fromPlan, cover",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [create, plan, fromPlan, cover]")]))
    .ui("Lyrics", "text"),
    OptionSpec::new(
        "yue2.seed",
        "yue2",
        "seed",
        Shape::Integer,
        DefaultRule::Provider("fresh submit: random concrete 63-bit seed; each batch member uses (seed + index) modulo 2^63"),
        IntentClass::WhatToMake,
        "GenerationRequest.seed",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: integer 0 <= seed < 2^63",
        "apps/rust-api/src/yue2_jobs.rs: fresh create/plan/cover resolve omission before queueing",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds create, plan, cover",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [create, plan, cover]")]))
    .ui("Seed", "number"),
    OptionSpec::new(
        "yue2.cfgScale",
        "yue2",
        "cfgScale",
        Shape::Number,
        DefaultRule::Provider("YuE2 song guidance default; omitted remains absent"),
        IntentClass::WhatToMake,
        "GenerationRequest.guidance",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: finite 0..20",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds create, plan, cover",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [create, plan, cover]")]))
    .ui("Cfg scale", "number"),
    OptionSpec::new(
        "yue2.steps",
        "yue2",
        "steps",
        Shape::Integer,
        DefaultRule::Provider("YuE2 midpoint ODE step default"),
        IntentClass::WhatToMake,
        "GenerationRequest.steps",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: integer 1..10000",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds create, fromPlan, cover, renderVersion",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [create, fromPlan, cover, renderVersion]")]))
    .ui("Steps", "number"),
    OptionSpec::new(
        "yue2.planning",
        "yue2",
        "planning",
        Shape::OneOf("full | melody | off"),
        DefaultRule::Provider("YuE2 SongParams planning default; UI default full"),
        IntentClass::WhatToMake,
        "SongParams.planning",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: plan kind forbids off; external score requires full or melody; off forbids scoreSampling",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds create, plan",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [create, plan]")]))
    .ui("Planning", "select"),
    OptionSpec::new(
        "yue2.score",
        "yue2",
        "score",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "SongParams.score",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: <=262144 bytes; native bounded ABC dialect; incompatible with planning off and scoreSampling",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds create, plan",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [create, plan]")]))
    .ui("Score", "text"),
    OptionSpec::new(
        "yue2.scoreSampling",
        "yue2",
        "scoreSampling",
        Shape::Object("TokenSampling"),
        DefaultRule::Provider("phase token sampling defaults"),
        IntentClass::WhatToMake,
        "SongParams.score_sampling",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: check_sampling; scoreSampling only when score is sampled, never external score or planning off",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds create, plan",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [create, plan]")]))
    .ui("Score sampling", "object"),
    OptionSpec::new(
        "yue2.semanticSampling",
        "yue2",
        "semanticSampling",
        Shape::Object("TokenSampling"),
        DefaultRule::Provider("phase token sampling defaults"),
        IntentClass::WhatToMake,
        "SongParams.semantic_sampling",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: check_sampling; scoreSampling only when score is sampled, never external score or planning off",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds create, fromPlan, cover, renderVersion",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [create, fromPlan, cover, renderVersion]")]))
    .ui("Semantic sampling", "object"),
    OptionSpec::new(
        "yue2.decoder",
        "yue2",
        "decoder",
        Shape::OneOf("standard | legacy"),
        DefaultRule::Literal("\"standard\""),
        IntentClass::WhatToMake,
        "SongParams.decoder / LoadSpec staged decoder",
    )
    .limits(&[
        "crates/sceneworks-worker/src/yue2_jobs.rs: requested decoder must be installed",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds create, fromPlan, cover, renderVersion, decode",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [create, fromPlan, cover, renderVersion, decode]")]))
    .ui("Decoder", "select"),
    OptionSpec::new(
        "yue2.tier",
        "yue2",
        "tier",
        Shape::OneOf("bf16 | q8 | q4"),
        DefaultRule::Model("primary download variant marked default; fallback bf16"),
        IntentClass::MachineBudget,
        "LoadSpec.quantize / selected physical tier",
    )
    .limits(&[
        "crates/sceneworks-worker/src/yue2_jobs.rs: resolve_load selects matching installed primary download row",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds create, plan, fromPlan, cover, renderVersion, decode",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [create, plan, fromPlan, cover, renderVersion, decode]")]))
    .ui("Tier", "select"),
    OptionSpec::new(
        "yue2.precision",
        "yue2",
        "precision",
        Shape::OneOf("default | fp32"),
        DefaultRule::Absent,
        IntentClass::MachineBudget,
        "legacy LoadSpec.precision",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: default | fp32; mutually exclusive with computePolicy",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds create, plan, fromPlan, cover, renderVersion, decode",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [create, plan, fromPlan, cover, renderVersion, decode]")]))
    .compat("Historical queued-job compatibility only; new non-transcription submissions must select computePolicy, and precision cannot coexist with it")
    .ui("Precision", "select"),
    OptionSpec::new(
        "yue2.computePolicy",
        "yue2",
        "computePolicy",
        Shape::OneOf("auto | bf16 | fp32"),
        DefaultRule::Required,
        IntentClass::MachineBudget,
        "LoadSpec.yue2_compute_policy",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: explicitly required by fresh non-transcription submissions; excludes legacy precision; experimentalFp8 requires auto",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds create, plan, fromPlan, cover, renderVersion, decode",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [create, plan, fromPlan, cover, renderVersion, decode]")]))
    .ui("Compute policy", "select"),
    OptionSpec::new(
        "yue2.arMode",
        "yue2",
        "arMode",
        Shape::OneOf("native | experimentalFp8"),
        DefaultRule::Literal("\"native\""),
        IntentClass::MachineBudget,
        "LoadSpec.yue2_ar_mode",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: experimentalFp8 requires explicit bf16 weight tier, computePolicy auto, no legacy fp32",
        "crates/sceneworks-worker/src/yue2_jobs.rs: experimental FP8 additionally requires CUDA sm_89+ and compatible resolved compute/device",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds create, plan, fromPlan, cover, renderVersion",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [create, plan, fromPlan, cover, renderVersion]")]))
    .ui("Ar mode", "select"),
    OptionSpec::new(
        "yue2.offloadPolicy",
        "yue2",
        "offloadPolicy",
        Shape::OneOf("resident | sequential"),
        DefaultRule::Literal("\"resident\""),
        IntentClass::MachineBudget,
        "LoadSpec.offload_policy",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: Yue2JobSpec; FIELD_KINDS rejects controls on kinds that do not consume them",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds create, fromPlan, cover, renderVersion",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [create, fromPlan, cover, renderVersion]")]))
    .effects(&[
        Effect { when: "sequential and memory.stageResidency unset", action: "stage AR weights between phases", target: "GenerationMemory.stage_residency" },
    ])
    .ui("Offload policy", "select"),
    OptionSpec::new(
        "yue2.memory",
        "yue2",
        "memory",
        Shape::Object("MemoryControls"),
        DefaultRule::Absent,
        IntentClass::MachineBudget,
        "GenerationRequest.memory",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: acoustic controls restricted by memory.acoustic kinds, decode controls by memory.decode kinds",
    ])
    .applies(Predicate::Family("yue2"))
    .ui("Memory", "object"),
    OptionSpec::new(
        "yue2.planJobId",
        "yue2",
        "planJobId",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "server resolved SongParams.plan",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: safe id <=200 characters; required for fromPlan",
        "apps/rust-api/src/yue2_jobs.rs: same-project completed source/version identity is verified",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds fromPlan",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [fromPlan]")]))
    .ui("Plan job id", "text"),
    OptionSpec::new(
        "yue2.sourceJobId",
        "yue2",
        "sourceJobId",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "server resolved SongParams.cached_latents",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: safe id <=200 characters; required for decode",
        "apps/rust-api/src/yue2_jobs.rs: same-project completed source/version identity is verified",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds decode",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [decode]")]))
    .ui("Source job id", "text"),
    OptionSpec::new(
        "yue2.versionId",
        "yue2",
        "versionId",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "server resolved score-version generation request",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: safe id <=200 characters; required for renderVersion",
        "apps/rust-api/src/yue2_jobs.rs: same-project completed source/version identity is verified",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds renderVersion",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [renderVersion]")]))
    .ui("Version id", "text"),
    OptionSpec::new(
        "yue2.cover",
        "yue2",
        "cover",
        Shape::Object("CoverSpec"),
        DefaultRule::Required,
        IntentClass::WhatToMake,
        "SongParams.cover",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: reviewed score or version required; recording-only direct cover deliberately refused",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds cover",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [cover]")]))
    .ui("Cover", "object"),
    OptionSpec::new(
        "yue2.sourceAudioAssetId",
        "yue2",
        "sourceAudioAssetId",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::WhatToMake,
        "SheetSage2 SourceAudio",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: safe id <=200 characters required for transcribe",
        "apps/rust-api/src/yue2_jobs.rs: project audio asset resolved and SHA256 pinned; transcriber runs separately on CPU",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds transcribe",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [transcribe]")]))
    .ui("Source audio asset id", "text"),
    OptionSpec::new(
        "yue2.transcription",
        "yue2",
        "transcription",
        Shape::Object("TranscriptionSettings"),
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "Yue2JobSpec.transcription",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: Yue2JobSpec; FIELD_KINDS rejects controls on kinds that do not consume them",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds transcribe",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [transcribe]")]))
    .ui("Transcription", "object"),
    OptionSpec::new(
        "yue2.projectName",
        "yue2",
        "projectName",
        Shape::Text,
        DefaultRule::Inherit("project.name"),
        IntentClass::Context,
        "Job.project_name",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: Yue2JobSpec; FIELD_KINDS rejects controls on kinds that do not consume them",
    ])
    .applies(Predicate::Family("yue2"))
    .ui("Project name", "text"),
    OptionSpec::new(
        "yue2.count",
        "yue2",
        "count",
        Shape::Integer,
        DefaultRule::Literal("1"),
        IntentClass::WhatToMake,
        "API queued take fan-out",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: integer 1..8; only create/plan/cover may request multiple takes",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds create, plan, cover",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [create, plan, cover]")]))
    .effects(&[
        Effect { when: "count > 1", action: "queue serial independent jobs and increment concrete seed per member", target: "Yue2JobSpec.seed / server BatchMember" },
    ])
    .ui("Count", "number"),
    OptionSpec::new(
        "yue2.commercialUse",
        "yue2",
        "commercialUse",
        Shape::Bool,
        DefaultRule::Literal("false"),
        IntentClass::Context,
        "yue2_eligibility policy check",
    )
    .limits(&[
        "apps/rust-api/src/yue2_jobs.rs: commercial YuE2 requests are refused by model policy",
    ])
    .applies(Predicate::Family("yue2"))
    .ui("Commercial use", "toggle"),
    OptionSpec::new(
        "yue2.licenseAcknowledged",
        "yue2",
        "licenseAcknowledged",
        Shape::Bool,
        DefaultRule::Literal("false"),
        IntentClass::Context,
        "server license acknowledgment record",
    )
    .limits(&[
        "apps/rust-api/src/yue2_jobs.rs: false does not bypass required prior license acceptance; true records acceptance",
    ])
    .applies(Predicate::Family("yue2"))
    .ui("License acknowledged", "toggle"),
    OptionSpec::new(
        "yue2.requestedGpu",
        "yue2",
        "requestedGpu",
        Shape::Text,
        DefaultRule::Literal("\"auto\""),
        IntentClass::MachineBudget,
        "Job.requested_gpu -> worker routing",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: Yue2JobSpec; FIELD_KINDS rejects controls on kinds that do not consume them",
    ])
    .applies(Predicate::Family("yue2"))
    .ui("Requested gpu", "text"),
    OptionSpec::new(
        "yue2.runId",
        "yue2",
        "runId",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::Context,
        "Yue2JobSpec.run_id",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: server-assigned; validate_request refuses caller-supplied value; validate_for_execution checks resolved provenance",
    ])
    .applies(Predicate::Family("yue2"))
    .internal(),
    OptionSpec::new(
        "yue2.batch",
        "yue2",
        "batch",
        Shape::Object("BatchMember"),
        DefaultRule::Absent,
        IntentClass::Context,
        "Yue2JobSpec.batch",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: server-assigned; validate_request refuses caller-supplied value; validate_for_execution checks resolved provenance",
    ])
    .applies(Predicate::Family("yue2"))
    .internal(),
    OptionSpec::new(
        "yue2.sources",
        "yue2",
        "sources",
        Shape::Object("Sources"),
        DefaultRule::Absent,
        IntentClass::Context,
        "Yue2JobSpec.sources",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: server-assigned; validate_request refuses caller-supplied value; validate_for_execution checks resolved provenance",
    ])
    .applies(Predicate::Family("yue2"))
    .internal(),
    OptionSpec::new(
        "yue2.scoreSampling.temperature",
        "yue2.scoreSampling",
        "temperature",
        Shape::Number,
        DefaultRule::Provider("YuE2 scoreSampling protocol default; omitted remains absent"),
        IntentClass::WhatToMake,
        "SongParams.score_sampling.temperature",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: check_sampling: finite 0..5",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds create, plan",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [create, plan]")]))
    .ui("Temperature", "number"),
    OptionSpec::new(
        "yue2.scoreSampling.topP",
        "yue2.scoreSampling",
        "topP",
        Shape::Number,
        DefaultRule::Provider("YuE2 scoreSampling protocol default; omitted remains absent"),
        IntentClass::WhatToMake,
        "SongParams.score_sampling.top_p",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: check_sampling: finite 0 < value <=1",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds create, plan",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [create, plan]")]))
    .ui("Top p", "number"),
    OptionSpec::new(
        "yue2.scoreSampling.topK",
        "yue2.scoreSampling",
        "topK",
        Shape::Integer,
        DefaultRule::Provider("YuE2 scoreSampling protocol default; omitted remains absent"),
        IntentClass::WhatToMake,
        "SongParams.score_sampling.top_k",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: check_sampling: unsigned integer >=1",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds create, plan",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [create, plan]")]))
    .ui("Top k", "number"),
    OptionSpec::new(
        "yue2.scoreSampling.repetitionPenalty",
        "yue2.scoreSampling",
        "repetitionPenalty",
        Shape::Number,
        DefaultRule::Provider("YuE2 scoreSampling protocol default; omitted remains absent"),
        IntentClass::WhatToMake,
        "SongParams.score_sampling.repetition_penalty",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: check_sampling: finite positive",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds create, plan",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [create, plan]")]))
    .ui("Repetition penalty", "number"),
    OptionSpec::new(
        "yue2.scoreSampling.penaltyWindow",
        "yue2.scoreSampling",
        "penaltyWindow",
        Shape::Integer,
        DefaultRule::Provider("YuE2 scoreSampling protocol default; omitted remains absent"),
        IntentClass::WhatToMake,
        "SongParams.score_sampling.penalty_window",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: check_sampling: integer 1..100",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds create, plan",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [create, plan]")]))
    .ui("Penalty window", "number"),
    OptionSpec::new(
        "yue2.scoreSampling.minTokens",
        "yue2.scoreSampling",
        "minTokens",
        Shape::Integer,
        DefaultRule::Literal("32"),
        IntentClass::WhatToMake,
        "SongParams.score_sampling.min_tokens",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: check_sampling: unsigned integer <= resolved maxTokens",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds create, plan",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [create, plan]")]))
    .ui("Min tokens", "number"),
    OptionSpec::new(
        "yue2.scoreSampling.maxTokens",
        "yue2.scoreSampling",
        "maxTokens",
        Shape::Integer,
        DefaultRule::Literal("4096"),
        IntentClass::WhatToMake,
        "SongParams.score_sampling.max_tokens",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: check_sampling: integer 1..24576 and >= resolved minTokens",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds create, plan",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [create, plan]")]))
    .ui("Max tokens", "number"),
    OptionSpec::new(
        "yue2.semanticSampling.temperature",
        "yue2.semanticSampling",
        "temperature",
        Shape::Number,
        DefaultRule::Provider("YuE2 semanticSampling protocol default; omitted remains absent"),
        IntentClass::WhatToMake,
        "SongParams.semantic_sampling.temperature",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: check_sampling: finite 0..5",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds create, fromPlan, cover, renderVersion",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [create, fromPlan, cover, renderVersion]")]))
    .ui("Temperature", "number"),
    OptionSpec::new(
        "yue2.semanticSampling.topP",
        "yue2.semanticSampling",
        "topP",
        Shape::Number,
        DefaultRule::Provider("YuE2 semanticSampling protocol default; omitted remains absent"),
        IntentClass::WhatToMake,
        "SongParams.semantic_sampling.top_p",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: check_sampling: finite 0 < value <=1",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds create, fromPlan, cover, renderVersion",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [create, fromPlan, cover, renderVersion]")]))
    .ui("Top p", "number"),
    OptionSpec::new(
        "yue2.semanticSampling.topK",
        "yue2.semanticSampling",
        "topK",
        Shape::Integer,
        DefaultRule::Provider("YuE2 semanticSampling protocol default; omitted remains absent"),
        IntentClass::WhatToMake,
        "SongParams.semantic_sampling.top_k",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: check_sampling: unsigned integer >=1",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds create, fromPlan, cover, renderVersion",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [create, fromPlan, cover, renderVersion]")]))
    .ui("Top k", "number"),
    OptionSpec::new(
        "yue2.semanticSampling.repetitionPenalty",
        "yue2.semanticSampling",
        "repetitionPenalty",
        Shape::Number,
        DefaultRule::Provider("YuE2 semanticSampling protocol default; omitted remains absent"),
        IntentClass::WhatToMake,
        "SongParams.semantic_sampling.repetition_penalty",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: check_sampling: finite positive",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds create, fromPlan, cover, renderVersion",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [create, fromPlan, cover, renderVersion]")]))
    .ui("Repetition penalty", "number"),
    OptionSpec::new(
        "yue2.semanticSampling.penaltyWindow",
        "yue2.semanticSampling",
        "penaltyWindow",
        Shape::Integer,
        DefaultRule::Provider("YuE2 semanticSampling protocol default; omitted remains absent"),
        IntentClass::WhatToMake,
        "SongParams.semantic_sampling.penalty_window",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: check_sampling: integer 1..100",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds create, fromPlan, cover, renderVersion",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [create, fromPlan, cover, renderVersion]")]))
    .ui("Penalty window", "number"),
    OptionSpec::new(
        "yue2.semanticSampling.minTokens",
        "yue2.semanticSampling",
        "minTokens",
        Shape::Integer,
        DefaultRule::Literal("200"),
        IntentClass::WhatToMake,
        "SongParams.semantic_sampling.min_tokens",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: check_sampling: unsigned integer <= resolved maxTokens",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds create, fromPlan, cover, renderVersion",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [create, fromPlan, cover, renderVersion]")]))
    .ui("Min tokens", "number"),
    OptionSpec::new(
        "yue2.semanticSampling.maxTokens",
        "yue2.semanticSampling",
        "maxTokens",
        Shape::Integer,
        DefaultRule::Literal("9000"),
        IntentClass::WhatToMake,
        "SongParams.semantic_sampling.max_tokens",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: check_sampling: integer 1..24576 and >= resolved minTokens",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds create, fromPlan, cover, renderVersion",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [create, fromPlan, cover, renderVersion]")]))
    .ui("Max tokens", "number"),
    OptionSpec::new(
        "yue2.memory.stageResidency",
        "yue2.memory",
        "stageResidency",
        Shape::Bool,
        DefaultRule::Inherit("offloadPolicy == sequential when memory block present"),
        IntentClass::MachineBudget,
        "GenerationMemory.stage_residency",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: explicit value overrides offload policy",
        "crates/sceneworks-worker/src/yue2_jobs.rs: memory() lowering; absent memory block preserves engine/load policy",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds create, fromPlan, cover, renderVersion",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [create, fromPlan, cover, renderVersion]")]))
    .ui("Stage residency", "toggle"),
    OptionSpec::new(
        "yue2.memory.chunkAttention",
        "yue2.memory",
        "chunkAttention",
        Shape::Bool,
        DefaultRule::Literal("false"),
        IntentClass::MachineBudget,
        "GenerationMemory.chunk_attention",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: attentionChunkSize only read when true",
        "crates/sceneworks-worker/src/yue2_jobs.rs: memory() lowering; absent memory block preserves engine/load policy",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds create, fromPlan, cover, renderVersion",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [create, fromPlan, cover, renderVersion]")]))
    .ui("Chunk attention", "toggle"),
    OptionSpec::new(
        "yue2.memory.attentionChunkSize",
        "yue2.memory",
        "attentionChunkSize",
        Shape::Integer,
        DefaultRule::Provider("YuE2 attention chunk default"),
        IntentClass::MachineBudget,
        "GenerationMemory.attention_chunk_size",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: requires chunkAttention=true; at least 393216 score elements",
        "crates/sceneworks-worker/src/yue2_jobs.rs: memory() lowering; absent memory block preserves engine/load policy",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds create, fromPlan, cover, renderVersion",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [create, fromPlan, cover, renderVersion]")]))
    .ui("Attention chunk size", "number"),
    OptionSpec::new(
        "yue2.memory.tileVaeDecode",
        "yue2.memory",
        "tileVaeDecode",
        Shape::Bool,
        DefaultRule::Literal("false"),
        IntentClass::MachineBudget,
        "GenerationMemory.tile_vae_decode",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: decodeTileEdge only read when true",
        "crates/sceneworks-worker/src/yue2_jobs.rs: memory() lowering; absent memory block preserves engine/load policy",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds create, fromPlan, cover, renderVersion, decode",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [create, fromPlan, cover, renderVersion, decode]")]))
    .ui("Tile vae decode", "toggle"),
    OptionSpec::new(
        "yue2.memory.decodeTileEdge",
        "yue2.memory",
        "decodeTileEdge",
        Shape::Integer,
        DefaultRule::Provider("YuE2 decode tile default"),
        IntentClass::MachineBudget,
        "GenerationMemory.decode_tile_edge",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: requires tileVaeDecode=true; integer 1..1024 latent frames",
        "crates/sceneworks-worker/src/yue2_jobs.rs: memory() lowering; absent memory block preserves engine/load policy",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds create, fromPlan, cover, renderVersion, decode",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [create, fromPlan, cover, renderVersion, decode]")]))
    .ui("Decode tile edge", "number"),
    OptionSpec::new(
        "yue2.cover.mode",
        "yue2.cover",
        "mode",
        Shape::OneOf("melody | full"),
        DefaultRule::Required,
        IntentClass::WhatToMake,
        "SongCover.mode",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: explicitly required",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds cover",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [cover]")]))
    .ui("Mode", "select"),
    OptionSpec::new(
        "yue2.cover.score",
        "yue2.cover",
        "score",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "SongCover.score",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: exactly one of score/versionId; <=262144 bytes native ABC",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds cover",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [cover]")]))
    .ui("Score", "text"),
    OptionSpec::new(
        "yue2.cover.versionId",
        "yue2.cover",
        "versionId",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::Context,
        "SongCover.version_id",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: exactly one of score/versionId; safe id <=200 chars; server resolves immutable reviewed score",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds cover",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [cover]")]))
    .ui("Version id", "text"),
    OptionSpec::new(
        "yue2.cover.keep",
        "yue2.cover",
        "keep",
        Shape::OneOf("both | vocal | instrumental"),
        DefaultRule::Provider("YuE2 SongCover voice default"),
        IntentClass::WhatToMake,
        "SongCover.keep",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: accepted only with mode=melody",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds cover",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [cover]")]))
    .ui("Keep", "select"),
    OptionSpec::new(
        "yue2.cover.translatedFrom",
        "yue2.cover",
        "translatedFrom",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "SongCover.translated_from",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: nonblank source lyrics <=16000 chars; section-aligned translation semantics belong to provider",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds cover",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [cover]")]))
    .ui("Translated from", "text"),
    OptionSpec::new(
        "yue2.cover.sourceAudioAssetId",
        "yue2.cover",
        "sourceAudioAssetId",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "SongCover.source_audio_asset_id",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: always refused with yue2_transcription_review_required; use separate transcribe job then reviewed version",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds cover",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [cover]")]))
    .compat("Recognized refusal carrier; never submit direct recording cover")
    .ui("Source audio asset id", "text"),
    OptionSpec::new(
        "yue2.transcription.maxSeconds",
        "yue2.transcription",
        "maxSeconds",
        Shape::Number,
        DefaultRule::Provider("uncropped recording; engine default preset"),
        IntentClass::WhatToMake,
        "SheetSage2 TranscriptionSettings.max_seconds",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: check_transcription: finite positive seconds",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds transcribe",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [transcribe]")]))
    .ui("Max seconds", "number"),
    OptionSpec::new(
        "yue2.transcription.overlapSeconds",
        "yue2.transcription",
        "overlapSeconds",
        Shape::Number,
        DefaultRule::Literal("200"),
        IntentClass::WhatToMake,
        "SheetSage2 TranscriptionSettings.overlap_seconds",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: check_transcription: finite 0 <= overlap <300 seconds",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds transcribe",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [transcribe]")]))
    .ui("Overlap seconds", "number"),
    OptionSpec::new(
        "yue2.transcription.lookaheadSeconds",
        "yue2.transcription",
        "lookaheadSeconds",
        Shape::Number,
        DefaultRule::Literal("100"),
        IntentClass::WhatToMake,
        "SheetSage2 TranscriptionSettings.lookahead_seconds",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: check_transcription: finite 0 <= lookahead <= resolved overlap seconds",
        "crates/sceneworks-core/src/yue2_score/jobs.rs: FIELD_KINDS accepts job kinds transcribe",
    ])
    .applies(Predicate::All(&[Predicate::Family("yue2"), Predicate::Feature("yue2.kind in [transcribe]")]))
    .ui("Lookahead seconds", "number"),
    OptionSpec::new(
        "yue2Version.abc",
        "yue2Version",
        "abc",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::WhatToMake,
        "CreateVersionInput.abc -> rendered SongParams.score",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/mod.rs: <=262144 bytes; native bounded ABC dialect",
    ])
    .applies(Predicate::Family("yue2"))
    .ui("Abc", "text"),
    OptionSpec::new(
        "yue2Version.request",
        "yue2Version",
        "request",
        Shape::Object("SongRequest"),
        DefaultRule::Required,
        IntentClass::WhatToMake,
        "CreateVersionInput.request",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/mod.rs: validate_request; renderVersion uses this immutable request",
    ])
    .applies(Predicate::Family("yue2"))
    .ui("Request", "object"),
    OptionSpec::new(
        "yue2Version.request.style",
        "yue2Version.request",
        "style",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::WhatToMake,
        "GenerationRequest.prompt",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/mod.rs: SongRequest / validate_request: <=4000 Unicode characters",
    ])
    .applies(Predicate::Family("yue2"))
    .ui("Style", "text"),
    OptionSpec::new(
        "yue2Version.request.lyrics",
        "yue2Version.request",
        "lyrics",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::WhatToMake,
        "AudioParams.lyrics",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/mod.rs: SongRequest / validate_request: <=16000 Unicode characters",
    ])
    .applies(Predicate::Family("yue2"))
    .ui("Lyrics", "text"),
    OptionSpec::new(
        "yue2Version.request.cot",
        "yue2Version.request",
        "cot",
        Shape::OneOf("full | melody"),
        DefaultRule::Required,
        IntentClass::WhatToMake,
        "SongParams.planning",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/mod.rs: SongRequest / validate_request: melody requires chord-free score; no automatic chord stripping",
    ])
    .applies(Predicate::Family("yue2"))
    .ui("Cot", "select"),
    OptionSpec::new(
        "yue2Version.request.seed",
        "yue2Version.request",
        "seed",
        Shape::Integer,
        DefaultRule::Literal("831001"),
        IntentClass::WhatToMake,
        "GenerationRequest.seed",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/mod.rs: SongRequest / validate_request: integer 0 <= seed <2^63",
    ])
    .applies(Predicate::Family("yue2"))
    .ui("Seed", "number"),
    OptionSpec::new(
        "yue2Version.request.cfgScale",
        "yue2Version.request",
        "cfgScale",
        Shape::Number,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "GenerationRequest.guidance",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/mod.rs: SongRequest / validate_request: finite 0..20",
    ])
    .applies(Predicate::Family("yue2"))
    .ui("Cfg scale", "number"),
    OptionSpec::new(
        "yue2.batch.id",
        "yue2.batch",
        "id",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "BatchMember.id",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: server-resolved identity; caller-supplied sources/batch are refused",
    ])
    .applies(Predicate::Family("yue2"))
    .internal(),
    OptionSpec::new(
        "yue2.batch.index",
        "yue2.batch",
        "index",
        Shape::Integer,
        DefaultRule::Required,
        IntentClass::Context,
        "BatchMember.index",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: server-resolved identity; caller-supplied sources/batch are refused",
    ])
    .applies(Predicate::Family("yue2"))
    .internal(),
    OptionSpec::new(
        "yue2.batch.count",
        "yue2.batch",
        "count",
        Shape::Integer,
        DefaultRule::Required,
        IntentClass::Context,
        "BatchMember.count",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: server-resolved identity; caller-supplied sources/batch are refused",
    ])
    .applies(Predicate::Family("yue2"))
    .internal(),
    OptionSpec::new(
        "yue2.sources.plan",
        "yue2.sources",
        "plan",
        Shape::Object("RunSource"),
        DefaultRule::Absent,
        IntentClass::Context,
        "Sources.plan",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: server-resolved identity; caller-supplied sources/batch are refused",
    ])
    .applies(Predicate::Family("yue2"))
    .internal(),
    OptionSpec::new(
        "yue2.sources.sourceRun",
        "yue2.sources",
        "sourceRun",
        Shape::Object("RunSource"),
        DefaultRule::Absent,
        IntentClass::Context,
        "Sources.source_run",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: server-resolved identity; caller-supplied sources/batch are refused",
    ])
    .applies(Predicate::Family("yue2"))
    .internal(),
    OptionSpec::new(
        "yue2.sources.version",
        "yue2.sources",
        "version",
        Shape::Object("VersionSource"),
        DefaultRule::Absent,
        IntentClass::Context,
        "Sources.version",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: server-resolved identity; caller-supplied sources/batch are refused",
    ])
    .applies(Predicate::Family("yue2"))
    .internal(),
    OptionSpec::new(
        "yue2.sources.coverVersion",
        "yue2.sources",
        "coverVersion",
        Shape::Object("VersionSource"),
        DefaultRule::Absent,
        IntentClass::Context,
        "Sources.cover_version",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: server-resolved identity; caller-supplied sources/batch are refused",
    ])
    .applies(Predicate::Family("yue2"))
    .internal(),
    OptionSpec::new(
        "yue2.sources.recording",
        "yue2.sources",
        "recording",
        Shape::Object("RecordingSource"),
        DefaultRule::Absent,
        IntentClass::Context,
        "Sources.recording",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: server-resolved identity; caller-supplied sources/batch are refused",
    ])
    .applies(Predicate::Family("yue2"))
    .internal(),
    OptionSpec::new(
        "yue2.sources.transcription",
        "yue2.sources",
        "transcription",
        Shape::Object("TranscriptionSource"),
        DefaultRule::Absent,
        IntentClass::Context,
        "Sources.transcription",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: server-resolved identity; caller-supplied sources/batch are refused",
    ])
    .applies(Predicate::Family("yue2"))
    .internal(),
    OptionSpec::new(
        "yue2.sources.plan.jobId",
        "yue2.sources.plan",
        "jobId",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "RunSource.job_id",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: server-resolved identity; caller-supplied sources/batch are refused",
    ])
    .applies(Predicate::Family("yue2"))
    .internal(),
    OptionSpec::new(
        "yue2.sources.plan.runDir",
        "yue2.sources.plan",
        "runDir",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "RunSource.run_dir",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: server-resolved identity; caller-supplied sources/batch are refused",
    ])
    .applies(Predicate::Family("yue2"))
    .internal(),
    OptionSpec::new(
        "yue2.sources.plan.identity",
        "yue2.sources.plan",
        "identity",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "RunSource.identity",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: server-resolved identity; caller-supplied sources/batch are refused",
    ])
    .applies(Predicate::Family("yue2"))
    .internal(),
    OptionSpec::new(
        "yue2.sources.plan.planIdentity",
        "yue2.sources.plan",
        "planIdentity",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::Context,
        "RunSource.plan_identity",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: server-resolved identity; caller-supplied sources/batch are refused",
    ])
    .applies(Predicate::Family("yue2"))
    .internal(),
    OptionSpec::new(
        "yue2.sources.sourceRun.jobId",
        "yue2.sources.sourceRun",
        "jobId",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "RunSource.job_id",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: server-resolved identity; caller-supplied sources/batch are refused",
    ])
    .applies(Predicate::Family("yue2"))
    .internal(),
    OptionSpec::new(
        "yue2.sources.sourceRun.runDir",
        "yue2.sources.sourceRun",
        "runDir",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "RunSource.run_dir",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: server-resolved identity; caller-supplied sources/batch are refused",
    ])
    .applies(Predicate::Family("yue2"))
    .internal(),
    OptionSpec::new(
        "yue2.sources.sourceRun.identity",
        "yue2.sources.sourceRun",
        "identity",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "RunSource.identity",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: server-resolved identity; caller-supplied sources/batch are refused",
    ])
    .applies(Predicate::Family("yue2"))
    .internal(),
    OptionSpec::new(
        "yue2.sources.sourceRun.planIdentity",
        "yue2.sources.sourceRun",
        "planIdentity",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::Context,
        "RunSource.plan_identity",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: server-resolved identity; caller-supplied sources/batch are refused",
    ])
    .applies(Predicate::Family("yue2"))
    .internal(),
    OptionSpec::new(
        "yue2.sources.version.id",
        "yue2.sources.version",
        "id",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "VersionSource.id",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: server-resolved identity; caller-supplied sources/batch are refused",
    ])
    .applies(Predicate::Family("yue2"))
    .internal(),
    OptionSpec::new(
        "yue2.sources.version.scoreSha256",
        "yue2.sources.version",
        "scoreSha256",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "VersionSource.score_sha256",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: server-resolved identity; caller-supplied sources/batch are refused",
    ])
    .applies(Predicate::Family("yue2"))
    .internal(),
    OptionSpec::new(
        "yue2.sources.version.requestSha256",
        "yue2.sources.version",
        "requestSha256",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "VersionSource.request_sha256",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: server-resolved identity; caller-supplied sources/batch are refused",
    ])
    .applies(Predicate::Family("yue2"))
    .internal(),
    OptionSpec::new(
        "yue2.sources.coverVersion.id",
        "yue2.sources.coverVersion",
        "id",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "VersionSource.id",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: server-resolved identity; caller-supplied sources/batch are refused",
    ])
    .applies(Predicate::Family("yue2"))
    .internal(),
    OptionSpec::new(
        "yue2.sources.coverVersion.scoreSha256",
        "yue2.sources.coverVersion",
        "scoreSha256",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "VersionSource.score_sha256",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: server-resolved identity; caller-supplied sources/batch are refused",
    ])
    .applies(Predicate::Family("yue2"))
    .internal(),
    OptionSpec::new(
        "yue2.sources.coverVersion.requestSha256",
        "yue2.sources.coverVersion",
        "requestSha256",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "VersionSource.request_sha256",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: server-resolved identity; caller-supplied sources/batch are refused",
    ])
    .applies(Predicate::Family("yue2"))
    .internal(),
    OptionSpec::new(
        "yue2.sources.recording.assetId",
        "yue2.sources.recording",
        "assetId",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "RecordingSource.asset_id",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: server-resolved identity; caller-supplied sources/batch are refused",
    ])
    .applies(Predicate::Family("yue2"))
    .internal(),
    OptionSpec::new(
        "yue2.sources.recording.sha256",
        "yue2.sources.recording",
        "sha256",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "RecordingSource.sha256",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: server-resolved identity; caller-supplied sources/batch are refused",
    ])
    .applies(Predicate::Family("yue2"))
    .internal(),
    OptionSpec::new(
        "yue2.sources.recording.name",
        "yue2.sources.recording",
        "name",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::Context,
        "RecordingSource.name",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: server-resolved identity; caller-supplied sources/batch are refused",
    ])
    .applies(Predicate::Family("yue2"))
    .internal(),
    OptionSpec::new(
        "yue2.sources.transcription.id",
        "yue2.sources.transcription",
        "id",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "TranscriptionSource.id",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: server-resolved identity; caller-supplied sources/batch are refused",
    ])
    .applies(Predicate::Family("yue2"))
    .internal(),
    OptionSpec::new(
        "yue2.sources.transcription.jobId",
        "yue2.sources.transcription",
        "jobId",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "TranscriptionSource.job_id",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: server-resolved identity; caller-supplied sources/batch are refused",
    ])
    .applies(Predicate::Family("yue2"))
    .internal(),
    OptionSpec::new(
        "yue2.sources.transcription.sourceAudioAssetId",
        "yue2.sources.transcription",
        "sourceAudioAssetId",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "TranscriptionSource.source_audio_asset_id",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: server-resolved identity; caller-supplied sources/batch are refused",
    ])
    .applies(Predicate::Family("yue2"))
    .internal(),
    OptionSpec::new(
        "yue2.sources.transcription.manifestSha256",
        "yue2.sources.transcription",
        "manifestSha256",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "TranscriptionSource.manifest_sha256",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: server-resolved identity; caller-supplied sources/batch are refused",
    ])
    .applies(Predicate::Family("yue2"))
    .internal(),
    OptionSpec::new(
        "yue2.sources.transcription.mode",
        "yue2.sources.transcription",
        "mode",
        Shape::OneOf("melody | full"),
        DefaultRule::Required,
        IntentClass::Context,
        "TranscriptionSource.mode",
    )
    .limits(&[
        "crates/sceneworks-core/src/yue2_score/jobs.rs: server-resolved identity; caller-supplied sources/batch are refused",
    ])
    .applies(Predicate::Family("yue2"))
    .internal(),
];
