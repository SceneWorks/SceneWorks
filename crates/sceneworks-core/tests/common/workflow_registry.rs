#![allow(
    dead_code,
    reason = "Independent integration targets consume different parts of this shared audit metadata"
)]

//! Source-contract audit metadata, compiled only by the integration tests that use it.

use sceneworks_core::workflow_share::AdvancedKeySource;

/// The JS shape a registered builder writes its `advanced` map in, and therefore which extractor
/// in `crates/sceneworks-core/tests/workflow_share.rs` can read it.
///
/// Named in the registry rather than sniffed, because a wrong guess is the failure mode this whole
/// lint exists to prevent: an extractor that quietly understands nothing returns an empty key set
/// and passes. Each extractor panics with instructions when the file stops matching its shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdvancedBuilderShape {
    /// `return { key, ...(cond ? { key } : {}) }` — the conditional-spread object literal
    /// `buildImageJobAdvanced` is built from. Needs the brace-aware scanner.
    ReturnedObject,
    /// `advanced: { a, b }` — a flat object literal in a returned payload, no spreads, no
    /// nesting (`buildDetailJobBody`).
    FlatAdvancedLiteral,
    /// `const advanced = { … }` followed by `advanced.key = …` assignments. Both halves are
    /// emits, and a spread in the initializer must be declared in
    /// [`AdvancedBuilder::spread_of`].
    AssignedObject,
    /// `advancedExtras: { … }` — a controller's contribution, merged into a
    /// [`Self::AssignedObject`] builder's initializer by spread.
    ExtrasLiteral,
    /// The builder posts NO `advanced` map at all (`buildUpscaleJobBody`). Registered anyway,
    /// because it feeds an embedding lane: the lint asserts the function stays empty of
    /// `advanced`, so the day someone adds a knob there it has to be classified.
    NoAdvancedMap,
    /// `advanced: { key, ...(cond ? { key } : {}) }` — an `advanced:` literal WITH conditional
    /// spreads, in a payload passed to a call rather than returned (sc-15956).
    ///
    /// The video builders' shape, and the one the four existing arms between them could not read:
    /// [`Self::FlatAdvancedLiteral`] asserts the literal has no spread and no nesting, and
    /// [`Self::ReturnedObject`] looks for a `return {`. Uses the same brace-aware
    /// `scan_object_literal` [`Self::ReturnedObject`] does, and honours
    /// [`AdvancedBuilder::spread_of`] the way [`Self::AssignedObject`] does — so a spread of
    /// somebody else's map has to name the builder that accounts for it.
    SpreadAdvancedLiteral,
}

/// One JS builder whose `advanced` keys reach an embedding lane.
///
/// The registry the coverage lint enumerates. `path` + `function` are read out of the repo, so a
/// rename fails loudly rather than silently pointing the lint at nothing.
#[derive(Debug, Clone, Copy)]
pub struct AdvancedBuilder {
    /// The tag rules use to name this builder. Unique across the registry.
    pub source: AdvancedKeySource,
    /// Repo-relative path of the file that defines it.
    pub path: &'static str,
    /// The JS function whose body the extractor reads.
    pub function: &'static str,
    pub shape: AdvancedBuilderShape,
    /// The embedding lane this builder's payload ends up written by, so a reader can see why the
    /// keys matter. Prose, asserted only to be non-trivial.
    pub lane: &'static str,
    /// Keys the extractor MUST still find. A floor against an extractor that has quietly stopped
    /// understanding the file — the failure mode that makes a green lint worthless.
    pub anchors: &'static [&'static str],
    /// The smallest number of keys the extractor may report before the lint calls itself broken.
    /// Set well under the real count; it is a floor, not a census.
    pub minimum_keys: usize,
    /// Identifiers spread into an [`AdvancedBuilderShape::AssignedObject`] initializer, whose keys
    /// are accounted for by another registry entry, or by those producers before persistence for
    /// a recipe replay. Declared so a NEW spread fails the lint instead of vanishing into it.
    pub spread_of: &'static [&'static str],
}

/// Every builder whose `advanced` keys travel inside a shared image (sc-15948).
///
/// The lanes, as of sc-15948: `POST /api/v1/image/jobs` (`text_to_image` / `edit_image` /
/// `character_image`), `POST /api/v1/image/detail/jobs`, `POST /api/v1/image/interleave/jobs`, and
/// the standalone `image_upscale` job. All four write their PNG through a seam that embeds.
///
/// **Adding a builder here is what turns the lint on for it.** Adding a builder to the WEB and not
/// to this table fails `every_advanced_builder_in_the_web_app_is_accounted_for`, which walks
/// `apps/web/src` and refuses any `advanced`-map producer that is in neither this registry nor
/// [`DEFERRED_ADVANCED_BUILDERS`].
pub const ADVANCED_BUILDERS: &[AdvancedBuilder] = &[
    AdvancedBuilder {
        source: AdvancedKeySource::MossEditBuilder,
        path: "apps/web/src/mossImage.js",
        function: "buildMossEditJobAdvanced",
        shape: AdvancedBuilderShape::ReturnedObject,
        lane: "POST /api/v1/image/jobs — shared MOSS edit controls for Studio, Library and Image Editor, written by `write_image_asset`",
        anchors: &["mossImage", "mossImageEdit", "steps", "guidanceScale", "schedulerShift"],
        minimum_keys: 6,
        spread_of: &[],
    },
    AdvancedBuilder {
        source: AdvancedKeySource::StudioBuilder,
        path: "apps/web/src/imageJobAdvanced.js",
        function: "buildImageJobAdvanced",
        shape: AdvancedBuilderShape::ReturnedObject,
        lane:
            "POST /api/v1/image/jobs — text_to_image, edit_image and character_image, written by \
               `write_image_asset`",
        anchors: &["resolution", "sampler", "steps", "styleId", "poses"],
        minimum_keys: 25,
        spread_of: &[],
    },
    AdvancedBuilder {
        source: AdvancedKeySource::ImageRecipeReplayBuilder,
        path: "apps/web/src/regenerateImage.js",
        function: "imageRecipeRequest",
        shape: AdvancedBuilderShape::AssignedObject,
        lane: "POST /api/v1/image/jobs — replay persisted image recipe inputs when the original \
               job is unavailable, written by `write_image_asset`",
        // The persisted map's producers own its key classifications. This builder only removes
        // receipts; any new explicit key still has to pass the same classification check.
        anchors: &[],
        minimum_keys: 0,
        spread_of: &["raw"],
    },
    AdvancedBuilder {
        source: AdvancedKeySource::EditBuilder,
        path: "apps/web/src/imageJobs.js",
        function: "buildEditJobBody",
        shape: AdvancedBuilderShape::AssignedObject,
        lane: "POST /api/v1/image/jobs — edit_image from the Image Editor, written by \
               `write_image_asset`",
        anchors: &["guidanceScale"],
        minimum_keys: 1,
        spread_of: &[],
    },
    AdvancedBuilder {
        source: AdvancedKeySource::DetailBuilder,
        path: "apps/web/src/imageJobs.js",
        function: "buildDetailJobBody",
        shape: AdvancedBuilderShape::FlatAdvancedLiteral,
        lane: "POST /api/v1/image/detail/jobs, written by `image_jobs/detail.rs`",
        anchors: &["strength", "cnScale"],
        minimum_keys: 2,
        spread_of: &[],
    },
    AdvancedBuilder {
        source: AdvancedKeySource::CharacterBuilder,
        path: "apps/web/src/components/CharacterAdvancedOptions.jsx",
        function: "buildAdvanced",
        shape: AdvancedBuilderShape::AssignedObject,
        lane: "POST /api/v1/image/jobs — character_image, written by `write_image_asset`",
        anchors: &["ipAdapterScale", "steps", "usePid"],
        // The angle-set / pose-library controllers' `advancedExtras`, each registered below.
        minimum_keys: 9,
        spread_of: &["base"],
    },
    AdvancedBuilder {
        source: AdvancedKeySource::CharacterAngleExtras,
        path: "apps/web/src/screens/characterPanels.jsx",
        function: "useAngleController",
        shape: AdvancedBuilderShape::ExtrasLiteral,
        lane: "POST /api/v1/image/jobs — character_image (angle set), written by \
               `write_image_asset`",
        anchors: &["angleSet"],
        minimum_keys: 2,
        spread_of: &[],
    },
    AdvancedBuilder {
        source: AdvancedKeySource::CharacterPoseExtras,
        path: "apps/web/src/screens/characterPanels.jsx",
        function: "usePoseController",
        shape: AdvancedBuilderShape::ExtrasLiteral,
        lane: "POST /api/v1/image/jobs — character_image (pose library), written by \
               `write_image_asset`",
        anchors: &["poses", "faceRestore"],
        minimum_keys: 3,
        spread_of: &[],
    },
    AdvancedBuilder {
        source: AdvancedKeySource::InterleaveBuilder,
        path: "apps/web/src/screens/DocumentStudio.jsx",
        function: "submit",
        shape: AdvancedBuilderShape::AssignedObject,
        lane: "POST /api/v1/image/interleave/jobs, written by `sensenova_jobs.rs` through \
               `write_image_asset`",
        anchors: &["systemMessage", "imageGuidanceScale"],
        minimum_keys: 2,
        spread_of: &[],
    },
    AdvancedBuilder {
        source: AdvancedKeySource::UpscaleBuilder,
        path: "apps/web/src/imageJobs.js",
        function: "buildUpscaleJobBody",
        shape: AdvancedBuilderShape::NoAdvancedMap,
        lane: "POST /api/v1/jobs type=image_upscale, written by `single_child_asset.rs`",
        anchors: &[],
        minimum_keys: 0,
        spread_of: &[],
    },
    // -----------------------------------------------------------------------
    // The VIDEO builders (sc-15956), promoted out of `DEFERRED_ADVANCED_BUILDERS`
    // -----------------------------------------------------------------------
    //
    // All six moved in the SAME change as the video write seams below, because the back-check that
    // every entry here is named by an embedding seam makes the two halves inseparable — see
    // `WORKFLOW_WRITE_SEAMS`.
    AdvancedBuilder {
        source: AdvancedKeySource::VideoStudioBuilder,
        path: "apps/web/src/screens/videoStudio/useVideoSubmission.js",
        function: "submit",
        shape: AdvancedBuilderShape::SpreadAdvancedLiteral,
        lane: "POST /api/v1/jobs type=video_generate — the Video Studio, written by \
               `video_jobs/mod.rs::encode_media`",
        anchors: &[
            "motion",
            "selectedPersonTrack",
            "replacementModeLabel",
            "lightning",
            "videoConditioningStrength",
        ],
        minimum_keys: 20,
        spread_of: &[],
    },
    AdvancedBuilder {
        source: AdvancedKeySource::TimelineBaseBuilder,
        path: "apps/web/src/components/editor/useEditorGeneration.js",
        function: "buildBasePayload",
        shape: AdvancedBuilderShape::AssignedObject,
        lane: "POST /api/v1/jobs type=video_generate — the timeline editor's shared video payload \
               (queueTimelineVideoJob), written by `video_jobs/mod.rs::encode_media`",
        anchors: &["resolution", "motion", "lightning"],
        minimum_keys: 6,
        spread_of: &[],
    },
    AdvancedBuilder {
        source: AdvancedKeySource::TimelineExtendBuilder,
        path: "apps/web/src/screens/EditorScreen.jsx",
        function: "extendSelectedClip",
        shape: AdvancedBuilderShape::SpreadAdvancedLiteral,
        lane: "POST /api/v1/jobs type=video_generate — timeline extend, written by \
               `video_jobs/mod.rs::encode_media`",
        anchors: &["timelineAction", "timelineContext"],
        minimum_keys: 2,
        spread_of: &["base.advanced"],
    },
    AdvancedBuilder {
        source: AdvancedKeySource::TimelineReplaceBuilder,
        path: "apps/web/src/screens/EditorScreen.jsx",
        function: "replaceSelectedItem",
        shape: AdvancedBuilderShape::SpreadAdvancedLiteral,
        lane: "POST /api/v1/jobs type=video_generate — timeline replace, written by \
               `video_jobs/mod.rs::encode_media`",
        anchors: &["timelineAction", "timelineContext"],
        minimum_keys: 2,
        spread_of: &["base.advanced"],
    },
    AdvancedBuilder {
        source: AdvancedKeySource::TimelineBridgeBuilder,
        path: "apps/web/src/screens/EditorScreen.jsx",
        function: "bridgeGap",
        shape: AdvancedBuilderShape::SpreadAdvancedLiteral,
        lane: "POST /api/v1/jobs type=video_generate — timeline bridge, written by \
               `video_jobs/mod.rs::encode_media`",
        anchors: &["timelineAction", "timelineContext"],
        minimum_keys: 2,
        spread_of: &["base.advanced"],
    },
    AdvancedBuilder {
        source: AdvancedKeySource::SimpleVideoBuilder,
        path: "apps/web/src/simple/simpleJobs.js",
        function: "buildSimpleVideoRequest",
        shape: AdvancedBuilderShape::SpreadAdvancedLiteral,
        lane: "POST /api/v1/jobs type=video_generate — the Simple shell's video request, written \
               by `video_jobs/mod.rs::encode_media`",
        anchors: &["resolution"],
        minimum_keys: 1,
        spread_of: &[],
    },
    AdvancedBuilder {
        source: AdvancedKeySource::VideoUpscaleBuilder,
        path: "apps/web/src/screens/VideoUpscalePanel.jsx",
        function: "onUpscale",
        shape: AdvancedBuilderShape::NoAdvancedMap,
        lane: "POST /api/v1/jobs type=video_upscale, written by \
               `video_jobs/seedvr2.rs::encode_seedvr2_stream`",
        anchors: &[],
        minimum_keys: 0,
        spread_of: &[],
    },
];

/// A builder that produces an `advanced` map the lint has SEEN and deliberately does not
/// classify, because its lane does not embed yet.
///
/// This list exists so "unaccounted for" and "accounted for as out of scope" are different
/// states. Making one of these lanes embed means moving its entry into [`ADVANCED_BUILDERS`] —
/// at which point every key it emits must be classified, which is the point.
#[derive(Debug, Clone, Copy)]
pub struct DeferredAdvancedBuilder {
    pub path: &'static str,
    pub function: &'static str,
    /// Why it is not classified. Two categories, and the lint accepts nothing else, so a deferral
    /// cannot be a shrug:
    ///
    /// * **Awaiting classification** — names the `sc-` story that owns turning its lane on.
    /// * **Permanently exempt** — starts with [`PERMANENT_EXEMPTION`] and says why no story will
    ///   ever own it, plus what would have to change for that to stop being true.
    pub reason: &'static str,
}

/// The prefix that marks a [`DeferredAdvancedBuilder::reason`] as a PERMANENT exemption rather
/// than a story-owned deferral.
///
/// A deferral normally names the story that will classify its keys. Some `advanced` maps are not
/// waiting on anybody: they are a different namespace on a lane that embeds nothing, so there is no
/// story to name and inventing a fake id would be worse than saying so. The lint requires a reason
/// with this prefix to state what would have to change (an embedding write seam) and to name
/// [`ADVANCED_BUILDERS`] as where the entry moves if it ever does.
pub const PERMANENT_EXEMPTION: &str = "PERMANENT EXEMPTION:";

/// One permanent exemption for an `advanced` map that is not a generation payload at all.
///
/// **The six video builders that used to live here were promoted by sc-15956** — the story this
/// list was built for — and their 17 unclassified keys are now rules in [`ADVANCED_KEY_RULES`].
/// The mechanism worked exactly as designed: making a video seam embed was impossible until every
/// one of those keys had an `allow`/`deny` decision, and two of them (`selectedPersonTrack`,
/// `timelineContext`) turned out to be object-shaped id-and-free-text carriers that would have
/// travelled silently under the pre-sc-16113 arrangement.
///
/// Note the count. This list's own comment said "~15 keys"; the real number was 17, because two
/// (`timelineAction`, `timelineContext`) are emitted only by the three `EditorScreen.jsx` actions
/// and no one had read those builders. A deferral list records that a decision is owed, not what
/// the decision costs — and that gap is the argument for the lint being a discovery scan rather
/// than a hand-maintained tally.
///
/// # What enforces that, and what does not (sc-16113)
///
/// This list used to claim, in this comment, that "nothing about a video lane can start embedding
/// while its keys sit in this list". Nothing did. A reviewer added a `write_workflow_chunk` +
/// `embeddable_workflow_share` call to a real video-lane PNG write in
/// `crates/sceneworks-worker/src/video_jobs/seedvr2.rs` and every lint stayed green, because the
/// sweep over `apps/web/src` accepts membership in EITHER registry and the write-seam lint read
/// four hard-coded worker files.
///
/// What is enforced now, by `every_worker_write_seam_declares_the_lane_it_embeds_for` in
/// `crates/sceneworks-core/tests/workflow_share.rs`:
///
/// * every function in `crates/sceneworks-worker/src` that the scan can see an envelope reach —
///   it calls a name on the core write surface, or names a `WorkflowShare` (or a worker struct
///   holding one) in its own signature, or calls a worker function that does — must have a
///   [`WORKFLOW_WRITE_SEAMS`] entry, and an undeclared one fails the build. The seams are found by
///   walking the crate, not by a file list, and `use … as …` renames of those names are resolved
///   before the walk;
/// * a seam declared [`SeamDisposition::Embeds`] must name the web builders that feed it, and a
///   named builder that is in THIS list fails the build, naming the seam, the lane and the builder;
/// * a seam cannot dodge that by declaring [`SeamDisposition::Declines`]. A decline is checked
///   POSITIVELY against the source: it may not call `write_workflow_chunk` with anything but a
///   literal `None`, may not name `WorkflowShare` in its body at all, may not accept one through
///   its signature, and must fill every share-carrying struct field with `None` — by initializer
///   OR by later assignment. Where the envelope came from does not matter, so parsing one back out
///   of another file and writing it on is an embed like any other.
///
/// So making a video lane embed now requires EITHER moving its builder into [`ADVANCED_BUILDERS`],
/// which is what forces every key it emits to be classified, or writing a builder list into
/// [`WORKFLOW_WRITE_SEAMS`] that is not true. The lint cannot tell those two apart — it has no way
/// to know which web builder feeds a Rust seam — so what it guarantees is that the statement had
/// to be written at all, and that the honest version of it fails the build. Review is what catches
/// the dishonest one.
pub const DEFERRED_ADVANCED_BUILDERS: &[DeferredAdvancedBuilder] = &[DeferredAdvancedBuilder {
    path: "apps/web/src/training/trainingConfig.js",
    function: "trainingConfigSnapshot",
    reason: "PERMANENT EXEMPTION: the Training Studio's `advanced` is a DIFFERENT namespace \
                 from the image payload's — trainer hyperparameters (networkType, lrScheduler, \
                 sampleSteps, decomposeFactor) on a training job, not generation intent for an \
                 image. No training write seam EMBEDS a workflow, nothing sanitizes these keys \
                 through `ADVANCED_KEY_RULES`, and no story is waiting to classify them: this is \
                 not a deferral with an owner, it is deliberately outside the allow-list. What \
                 would have to change is a training write seam that embeds. Then, and only then, \
                 this entry moves into `ADVANCED_BUILDERS` and every key it emits needs an \
                 allow/deny decision.",
}];

/// A (file, function) reference to one entry in [`ADVANCED_BUILDERS`] or
/// [`DEFERRED_ADVANCED_BUILDERS`] (sc-16113).
///
/// The same key those two registries are already joined on by
/// `every_advanced_builder_in_the_web_app_is_accounted_for`, so a seam names a builder exactly the
/// way the sweep does. A reference that resolves to neither registry fails the lint; a reference
/// that resolves to the DEFERRED one fails it too, and that is the whole point of the type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WebBuilderRef {
    /// Repo-relative path of the JS/JSX file that defines the builder.
    pub path: &'static str,
    /// The builder function's name.
    pub function: &'static str,
}

/// What one worker seam does about the workflow chunk (sc-16113).
///
/// Four states, not two, because "writes no chunk", "writes whatever its caller decided" and
/// "an envelope reaches it but it writes no file" are three different claims and only one of them
/// ends the lane question. Each is checked against the seam's own source, so the disposition is a
/// declaration the lint verifies rather than one it trusts.
#[derive(Debug, Clone, Copy)]
pub enum SeamDisposition {
    /// An envelope leaves this function into a file or into a share-carrying field. Must name at
    /// least one builder, and every one of them must be in [`ADVANCED_BUILDERS`] — a reference to
    /// [`DEFERRED_ADVANCED_BUILDERS`] is exactly the state this registry exists to refuse.
    ///
    /// Where the envelope came from is not part of the test: building one, parsing one back out of
    /// another file, or cloning one held elsewhere are all embedding, because what reaches the
    /// written file is the same either way.
    Embeds(&'static [WebBuilderRef]),
    /// Writes an envelope its CALLER built, and obtains none of its own. The lane question belongs
    /// to the callers, and they are seams in their own right: a function that hands a
    /// `WorkflowShare` to this one names it, which is what the discovery scan reads. The reason
    /// says why the decision is not made here.
    ///
    /// Verified: it must accept a share through its signature, must reach the write surface (a
    /// function that reaches no writer is [`SeamDisposition::Inert`], not a conduit), and must not
    /// build, parse or name a `WorkflowShare` of its own — sourcing one is deciding a lane.
    Conduit(&'static str),
    /// Writes the file with no envelope at all, deliberately. The reason must say why that asset
    /// has no generation recipe to record.
    ///
    /// Verified positively, not by the absence of a builder call: it may not call
    /// `write_workflow_chunk` with anything but a literal `None`, may not name `WorkflowShare` in
    /// its body, may not accept one through its signature, and must fill every share-carrying
    /// struct field with `None` — by initializer or by later assignment. So declining is not a way
    /// to embed without classifying a lane, whatever the envelope's provenance.
    Declines(&'static str),
    /// A `WorkflowShare` reaches this function's signature and goes nowhere: it writes no file,
    /// builds no envelope and calls nothing on the write surface. A logging or validation helper
    /// that takes a spec is this, and calling it a [`SeamDisposition::Conduit`] would be a false
    /// statement — a conduit writes.
    ///
    /// Verified: it must carry a share, and it must call nothing the scan can see reach a write.
    /// The moment it calls a writer it stops being inert and owes a lane.
    Inert(&'static str),
}

/// One place in `sceneworks-worker` where a [`WorkflowShare`] can reach a written file (sc-16113).
///
/// The registry that ties "this lane has an embedding write seam" to "its builder is in
/// [`ADVANCED_BUILDERS`]". `path` + `function` are read out of the repo, so a rename or a move
/// fails loudly, and the seams themselves are DISCOVERED by scanning the worker crate rather than
/// listed here — an entry nobody can find is as much a failure as a seam nobody declared.
#[derive(Debug, Clone, Copy)]
pub struct WorkflowWriteSeam {
    /// Repo-relative path of the file that contains the seam.
    pub path: &'static str,
    /// The Rust function whose body touches the envelope.
    pub function: &'static str,
    /// Prose: which product lane's files this seam writes, so a reader can see why the builders
    /// below are the ones that feed it. Asserted only to be non-trivial.
    pub lane: &'static str,
    /// What it does about the chunk, checked against the source.
    pub disposition: SeamDisposition,
}

/// Every worker seam a [`WorkflowShare`] can pass through, and the web builders behind each one.
///
/// **The gate this whole epic's deferral list was supposed to be.** Adding an embedding call to a
/// lane whose builder sits in [`DEFERRED_ADVANCED_BUILDERS`] now fails the build twice over: once
/// because the new seam has no entry here, and again if the entry names the deferred builder.
///
/// Kept honest in both directions by `every_worker_write_seam_declares_the_lane_it_embeds_for`: a
/// discovered seam with no entry fails, and an entry whose function no longer touches an envelope
/// fails too.
///
/// # Ordering, for whoever wires a new lane (sc-15956)
///
/// The back-check that every [`ADVANCED_BUILDERS`] entry is named by some embedding seam means the
/// two halves of a new lane cannot be split across two PRs: classifying the ~15 video keys on its
/// own would leave `VideoStudio.jsx::submit` registered with no seam behind it, which fails. Move
/// the builder up and add the seam entry in the SAME change. The failure is loud either way, but
/// it is worth knowing before the split is attempted.
///
/// Likewise `a_seam_that_embeds_for_a_deferred_builder_fails_the_build` hard-codes
/// `VideoStudio.jsx::submit` as its deferred example, so promoting that builder turns that
/// mutation proof into a "did not panic" failure. Re-point it at whatever is still deferred then;
/// the proof is that SOME deferred builder is refused, not that one particular one is.
pub const WORKFLOW_WRITE_SEAMS: &[WorkflowWriteSeam] = &[
    WorkflowWriteSeam {
        path: "crates/sceneworks-worker/src/image_jobs/moss.rs",
        function: "write_moss_tensor_companion",
        lane: "MOSS edit latent output — bounded existing WorkflowShare JSON companion next to the exact NPY artifact",
        disposition: SeamDisposition::Embeds(IMAGE_JOB_BUILDERS),
    },
    WorkflowWriteSeam {
        path: "crates/sceneworks-worker/src/image_jobs.rs",
        function: "write_image_asset",
        lane: "POST /api/v1/image/jobs (text_to_image / edit_image / character_image) and \
               POST /api/v1/image/interleave/jobs — the one funnel every generated image is \
               written through",
        disposition: SeamDisposition::Embeds(IMAGE_JOB_BUILDERS),
    },
    WorkflowWriteSeam {
        path: "crates/sceneworks-worker/src/image_jobs.rs",
        function: "upscaled_workflow_share",
        lane: "POST /api/v1/image/jobs — the inline-upscale sub-step of a generate job, whose \
               envelope is the generation's payload with the APPLIED pass overlaid",
        disposition: SeamDisposition::Embeds(GENERATE_JOB_BUILDERS),
    },
    WorkflowWriteSeam {
        path: "crates/sceneworks-worker/src/image_jobs.rs",
        function: "write_upscaled_asset",
        lane: "POST /api/v1/image/jobs — the inline-upscaled variant's own PNG",
        disposition: SeamDisposition::Embeds(GENERATE_JOB_BUILDERS),
    },
    WorkflowWriteSeam {
        path: "crates/sceneworks-worker/src/image_jobs.rs",
        function: "detail_workflow_share",
        lane: "POST /api/v1/image/detail/jobs — the standalone detail pass's own envelope",
        disposition: SeamDisposition::Embeds(&[WebBuilderRef {
            path: "apps/web/src/imageJobs.js",
            function: "buildDetailJobBody",
        }]),
    },
    WorkflowWriteSeam {
        path: "crates/sceneworks-worker/src/image_jobs.rs",
        function: "standalone_upscale_workflow_share",
        lane: "POST /api/v1/jobs type=image_upscale — the standalone upscale job's own envelope",
        disposition: SeamDisposition::Embeds(&[WebBuilderRef {
            path: "apps/web/src/imageJobs.js",
            function: "buildUpscaleJobBody",
        }]),
    },
    WorkflowWriteSeam {
        path: "crates/sceneworks-worker/src/image_jobs.rs",
        function: "standalone_restore_workflow_share",
        lane: "POST /api/v1/jobs type=image_upscale engine=iris — the standalone Iris restoration's \
               own envelope (sc-25683)",
        disposition: SeamDisposition::Embeds(&[WebBuilderRef {
            path: "apps/web/src/imageJobs.js",
            function: "buildUpscaleJobBody",
        }]),
    },
    WorkflowWriteSeam {
        path: "crates/sceneworks-worker/src/image_jobs/detail.rs",
        function: "run_image_detail_job",
        lane: "POST /api/v1/image/detail/jobs — writes the refined PNG (macOS-gated; the scan is \
               textual, so this seam is read on every platform)",
        disposition: SeamDisposition::Embeds(&[WebBuilderRef {
            path: "apps/web/src/imageJobs.js",
            function: "buildDetailJobBody",
        }]),
    },
    WorkflowWriteSeam {
        path: "crates/sceneworks-worker/src/upscale_jobs.rs",
        function: "run_iris_restore_job_using",
        lane: "POST /api/v1/jobs type=image_upscale engine=iris — hands the standalone Iris \
               restoration's envelope to the shared single-child writer (sc-25683)",
        disposition: SeamDisposition::Embeds(&[WebBuilderRef {
            path: "apps/web/src/imageJobs.js",
            function: "buildUpscaleJobBody",
        }]),
    },
    WorkflowWriteSeam {
        path: "crates/sceneworks-worker/src/upscale_jobs.rs",
        function: "run_image_upscale_job",
        lane: "POST /api/v1/jobs type=image_upscale — hands the standalone upscale's envelope to \
               the shared single-child writer",
        disposition: SeamDisposition::Embeds(&[WebBuilderRef {
            path: "apps/web/src/imageJobs.js",
            function: "buildUpscaleJobBody",
        }]),
    },
    WorkflowWriteSeam {
        path: "crates/sceneworks-worker/src/single_child_asset.rs",
        function: "write_single_child_asset",
        lane: "The shared one-PNG-child writer: the standalone upscale and the smart-select mask \
               both come through here",
        disposition: SeamDisposition::Conduit(
            "It writes the `SingleChildAssetSpec::workflow` its caller decided on and builds no \
             envelope of its own, so the lane question belongs to the callers — and every caller \
             is a seam here in its own right, because handing a `WorkflowShare` in is what the \
             discovery scan reads.",
        ),
    },
    WorkflowWriteSeam {
        path: "crates/sceneworks-worker/src/video_jobs/mod.rs",
        function: "video_workflow_metadata",
        lane: "POST /api/v1/jobs type=video_generate — the funnel every GENERATED clip is encoded \
               through (`encode_media`), whatever engine, route or studio produced it",
        disposition: SeamDisposition::Embeds(VIDEO_JOB_BUILDERS),
    },
    WorkflowWriteSeam {
        path: "crates/sceneworks-worker/src/video_jobs/seedvr2.rs",
        function: "seedvr2_workflow_metadata",
        lane: "POST /api/v1/jobs type=video_upscale — the SeedVR2 upscale's own clip, encoded \
               through `encode_seedvr2_stream` (macOS + candle-gated; the scan is textual, so \
               this seam is read on every platform)",
        disposition: SeamDisposition::Embeds(VIDEO_UPSCALE_BUILDERS),
    },
    WorkflowWriteSeam {
        path: "crates/sceneworks-worker/src/segment_jobs.rs",
        function: "run_image_segment_job",
        lane: "POST /api/v1/jobs type=image_segment — the smart-select mask",
        disposition: SeamDisposition::Declines(
            "A smart-select mask is not a generated image: it is a binary selection derived from a \
             box prompt and a concept string, with no generation recipe to replay. It is also \
             grayscale, and the chunk writer encodes RGB8, so embedding would triple its size and \
             change its colour type.",
        ),
    },
];

/// The builder behind an upscaled clip (sc-15956 review).
///
/// Its own list rather than a seventh entry in [`VIDEO_JOB_BUILDERS`], because it is a different
/// job type on a different encoder: `type=video_upscale` through `encode_seedvr2_stream`, not
/// `type=video_generate` through `encode_media`. The image lane draws the same line —
/// `buildUpscaleJobBody` feeds `run_image_upscale_job` and nothing else.
///
/// The panel posts no `advanced` map at all, and is registered precisely so that stays true: it
/// feeds an embedding lane, so the day it grows a knob the coverage lint demands a classification
/// instead of dropping it silently.
const VIDEO_UPSCALE_BUILDERS: &[WebBuilderRef] = &[WebBuilderRef {
    path: "apps/web/src/screens/VideoUpscalePanel.jsx",
    function: "onUpscale",
}];

/// Every builder behind a generated clip (sc-15956).
///
/// All six feed ONE seam, because all six post `type=video_generate` and every video-GENERATE job
/// funnels through `encode_media` — the same property that let sc-12371 measure clip length in one
/// place. It is not, and never was, the only libx264 site in the worker: `video_upscale` has one of
/// its own, and [`VIDEO_UPSCALE_BUILDERS`] is the seam behind it.
const VIDEO_JOB_BUILDERS: &[WebBuilderRef] = &[
    WebBuilderRef {
        path: "apps/web/src/screens/videoStudio/useVideoSubmission.js",
        function: "submit",
    },
    WebBuilderRef {
        path: "apps/web/src/components/editor/useEditorGeneration.js",
        function: "buildBasePayload",
    },
    WebBuilderRef {
        path: "apps/web/src/screens/EditorScreen.jsx",
        function: "extendSelectedClip",
    },
    WebBuilderRef {
        path: "apps/web/src/screens/EditorScreen.jsx",
        function: "replaceSelectedItem",
    },
    WebBuilderRef {
        path: "apps/web/src/screens/EditorScreen.jsx",
        function: "bridgeGap",
    },
    WebBuilderRef {
        path: "apps/web/src/simple/simpleJobs.js",
        function: "buildSimpleVideoRequest",
    },
];

/// The builders behind every image the generate lane writes (`POST /api/v1/image/jobs`).
///
/// Named once because three seams share it: the base render, the inline-upscale envelope, and the
/// upscaled variant's own PNG.
const GENERATE_JOB_BUILDERS: &[WebBuilderRef] = &[
    WebBuilderRef {
        path: "apps/web/src/imageJobAdvanced.js",
        function: "buildImageJobAdvanced",
    },
    WebBuilderRef {
        path: "apps/web/src/regenerateImage.js",
        function: "imageRecipeRequest",
    },
    WebBuilderRef {
        path: "apps/web/src/imageJobs.js",
        function: "buildEditJobBody",
    },
    WebBuilderRef {
        path: "apps/web/src/components/CharacterAdvancedOptions.jsx",
        function: "buildAdvanced",
    },
    WebBuilderRef {
        path: "apps/web/src/screens/characterPanels.jsx",
        function: "useAngleController",
    },
    WebBuilderRef {
        path: "apps/web/src/screens/characterPanels.jsx",
        function: "usePoseController",
    },
];

/// [`GENERATE_JOB_BUILDERS`] plus the interleave lane, which reaches `write_image_asset` through
/// `sensenova_jobs.rs` threading its `workflow_source` into the same `ImagePlan`.
///
/// The interleave lane has no inline-upscale sub-step, so it is not in [`GENERATE_JOB_BUILDERS`].
const IMAGE_JOB_BUILDERS: &[WebBuilderRef] = &[
    WebBuilderRef {
        path: "apps/web/src/mossImage.js",
        function: "buildMossEditJobAdvanced",
    },
    WebBuilderRef {
        path: "apps/web/src/imageJobAdvanced.js",
        function: "buildImageJobAdvanced",
    },
    WebBuilderRef {
        path: "apps/web/src/regenerateImage.js",
        function: "imageRecipeRequest",
    },
    WebBuilderRef {
        path: "apps/web/src/imageJobs.js",
        function: "buildEditJobBody",
    },
    WebBuilderRef {
        path: "apps/web/src/components/CharacterAdvancedOptions.jsx",
        function: "buildAdvanced",
    },
    WebBuilderRef {
        path: "apps/web/src/screens/characterPanels.jsx",
        function: "useAngleController",
    },
    WebBuilderRef {
        path: "apps/web/src/screens/characterPanels.jsx",
        function: "usePoseController",
    },
    WebBuilderRef {
        path: "apps/web/src/screens/DocumentStudio.jsx",
        function: "submit",
    },
];
