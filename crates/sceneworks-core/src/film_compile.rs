//! Compiling a [`ProductionPlan`] into the model-specific requests the harness dispatches
//! (epic 22708, sc-22713).
//!
//! A plan says what the film is; a [`CompiledPlan`] says exactly what will be asked of the model.
//! It is a separate, versioned, diffable document for one reason: the prompt the engine sees is not
//! the prompt the plan holds — it has been through the model's own prompt refinement — and a POC
//! whose output cannot be explained is not evidence. With the compiled document beside the plan,
//! every dispatched request can be read, diffed against the previous version and corrected before
//! anything renders.
//!
//! The compiled request is also the ONLY place a video job body is built. [`CompiledRequest::
//! to_job_body`] is what the driver posts to `/api/v1/video/jobs`, so what a reviewer reads in
//! `compiled.json` and what the API receives cannot drift: the difference between them is only the
//! run-scoped ids and the reference roles resolved to imported asset ids.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{json, Map as JsonObject, Value};
use sha2::{Digest, Sha256};

use crate::film_plan::{
    is_reference_partition_id, plan_lora_payload_entries, plan_loras_for_partition,
    shot_resolution, ModelEntries, ModelLane, PlanDiagnostic, ProductionPlan, ReferenceEntry,
    ReferencePack, Shot, AUDIO_PROMPT_PREFIX,
};
use crate::minimax_h3_turbo::resolve_turbo_recipe;
use crate::video_request::effective_reference_image_short_edge;
use crate::MAX_PROMPT_CHARS;

/// Schema version of [`CompiledPlan`] documents this module reads and writes.
///
/// **2** (sc-23402): `CompiledRequest::model` is the RESOLVED partition id rather than the plan's
/// declared family model, and `partitionReason` says why. The document's semantics changed, so a v1
/// document is refused BY VERSION. Without the bump `partitionReason`'s `#[serde(default)]` would
/// let a v1 document parse with an empty reason and then fail [`request_differences`] as
/// hand-edited, which blames the operator for a schema migration. The remedy either way is
/// `film-harness compile`.
/// **3** (sc-23406): a request carries the LoRAs it dispatches with and the step count it renders
/// at. Both are DERIVED fields [`request_differences`] compares, so a v2 document read under this
/// build would default them to "none / unknown" and then be blamed as hand-edited — the same
/// migration trap the v2 bump above exists to avoid. The remedy is the same one line:
/// `film-harness compile`.
/// **4** (sc-24023): a reference request's prompt now LEADS with compiler-written binding sentences
/// naming each reference's `<Picture N>`, and the inserted text is recorded separately in
/// `insertedText`. A v3 document's prompt has no binding sentences and no `insertedText` key, so
/// reading one under this build would default the field to empty and then blame the operator for a
/// hand edit through [`request_differences`] — the same migration trap as the two bumps above. The
/// remedy is the same one line: `film-harness compile`.
/// **5** (sc-24026): EVERY request's prompt now TRAILS with the shot's `Audio:` sentence, and a
/// request whose shot places a dialogue line trails with [`NO_SPEECH_SENTENCE`] after it. Both are
/// recorded as further `insertedText` entries. A v4 document's prompt carries neither, so reading
/// one under this build would compare clean-but-different against a fresh compile and blame the
/// operator for a hand edit through [`request_differences`] — the same migration trap as the three
/// bumps above. The remedy is the same one line: `film-harness compile`.
/// **6** (sc-24025): every request may now carry the TEXT IDENTITY LOCK — the pack's description of
/// each `continuityRoles` entry the shot does not bind to an image, inserted word for word and
/// recorded as a further `insertedText` entry. It is not a reference feature: a plan whose shots
/// bind nothing at all gains it, which is exactly the case it exists for. A v5 document's prompt
/// carries none of it, so reading one under this build would compare clean-but-different against a
/// fresh compile and blame the operator for a hand edit through [`request_differences`] — the same
/// migration trap as the four bumps above. The remedy is the same one line: `film-harness compile`.
/// **7** (sc-24029): the document records the REFERENCE PACK it was compiled against
/// ([`CompiledPlan::reference_pack_sha256`]), and [`CompiledPlan::staleness_findings`] compares it.
/// The pack now decides the inserted text — a description edited in the pack changes every prompt
/// that repeats it — so a document keyed on the plan alone stayed "current" across a pack edit and
/// then failed [`request_differences`] at preflight, blaming the operator for a hand edit they did
/// not make. A v6 document has no such key, so it is refused BY VERSION like the five bumps above
/// rather than compared against a default. The remedy is the same one line: `film-harness compile`.
pub const COMPILED_PLAN_SCHEMA_VERSION: u32 = 7;

/// The field [`CompiledPlan::staleness_findings`] reports a changed reference pack under.
///
/// Named so the one caller that must tell a PACK finding from a PLAN one — the project store's
/// draft save, which carries a compile across its own revision bump — asks by constant rather than
/// by a string literal it would have to keep in step (sc-24029).
pub const COMPILED_PACK_STALENESS_FIELD: &str = "compiled.referencePackSha256";

/// The sentence appended when the harness itself puts a voice on this shot (sc-24026).
///
/// MiniMax-H3 scores a soundtrack from the prompt, speech included, and a shot whose dialogue bus
/// already carries a line would otherwise come back with a second voice over ours — two people
/// saying different things at once, which no downstream check can detect and no mix can undo.
/// FIXED wording, chosen once: it is compiler-owned text, so it must read the same on every shot of
/// every plan rather than varying with whoever authored the audio sentence.
///
/// It constrains the SOUNDTRACK and nothing else (sc-24026). These are exactly the shots that DO
/// have someone speaking on camera — our own line is about to play over them — so a sentence
/// phrased as a statement about the picture ("no one speaks on camera") would tell H3 to render
/// closed mouths under our dialogue track. The wording names the generated audio explicitly so the
/// model reads it as a constraint on what it scores, not on what it renders.
pub const NO_SPEECH_SENTENCE: &str =
    "No spoken dialogue in the generated audio; no voices on the soundtrack.";

/// Serialize a production plan exactly as the project store and harness persist it, then hash
/// those bytes. Keeping this beside the compiler prevents the editor preflight and CLI harness
/// from inventing competing definitions of a "current" compiled document.
pub fn production_plan_sha256(plan: &ProductionPlan) -> Result<String, serde_json::Error> {
    let mut bytes = serde_json::to_vec_pretty(plan)?;
    bytes.push(b'\n');
    Ok(format!("{:x}", Sha256::digest(&bytes)))
}

/// The identity of the reference PACK a compiled document was produced against (sc-24029).
///
/// # What is hashed, and why it is not the document's bytes
///
/// The PARSED pack, re-serialized canonically — exactly as [`production_plan_sha256`] treats a plan
/// — and NOT the bytes of the file on disk, although `resume` and `replace-take` do hash the pack's
/// bytes for their own pinning. The two paths that hold a pack hold it in different shapes: the CLI
/// reads a JSONC DOCUMENT whose comments and spacing are an author's, while the workspace holds a
/// typed pack inside a draft that was never a file. Hashing bytes would give the same pack two
/// identities and stale a compiled document on a comment-only edit; hashing the parsed value gives
/// one identity for one pack, and changes exactly when something the compiler reads changes.
///
/// Every field of [`ReferencePack`] is included rather than only the descriptions and locators the
/// inserted text repeats: `approved`, `kind`, `file` and `role` each decide whether a role is bound,
/// described or ignored, and a compiler-owned sentence follows from all of them.
pub fn reference_pack_sha256(pack: &ReferencePack) -> Result<String, serde_json::Error> {
    let mut bytes = serde_json::to_vec_pretty(pack)?;
    bytes.push(b'\n');
    Ok(format!("{:x}", Sha256::digest(&bytes)))
}

/// How far apart one shot's successive attempts are seeded (sc-22715).
///
/// Attempt `n` of a shot renders at `seed + (n - 1) * ATTEMPT_SEED_STRIDE`, so the seed a run
/// dispatches identifies a (shot, attempt) pair rather than only a shot. Anything smaller than the
/// gap a plan leaves between its own per-shot seeds makes two different renders share a seed: the
/// shipped courier fixture seeds its six shots 22710..22715, so at a stride of 1 SH020's second
/// attempt and SH030's first were the same number.
pub const ATTEMPT_SEED_STRIDE: i64 = 1000;

/// Where a compiled request's prompt text came from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptSource {
    /// The plan's prompt, verbatim. What a hand-authored plan compiles to, and what `--no-refine`
    /// produces.
    Authored,
    /// The plan's prompt after the model's own prompt refinement (the `prompt_refine` seam). The
    /// authored text is kept beside it so the rewrite is reviewable.
    Refined,
}

/// One role bound to a reference picture, with the pack entry it names.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundReferenceRole<'a> {
    pub role: String,
    /// The pack entry for `role`, when the pack declares one. `None` only for a plan that never
    /// passed [`crate::film_plan::validate_plan_against_pack`], which refuses an undeclared role.
    pub entry: Option<&'a ReferenceEntry>,
}

/// One reference picture a shot dispatches: the 1-based number the engine labels it with, and the
/// pack roles it carries.
#[derive(Debug, Clone, PartialEq)]
pub struct ReferencePicture<'a> {
    /// The `N` in `<Picture N>`, and the 1-based position of this picture's asset in the
    /// dispatched `referenceAssetIds`.
    pub number: u32,
    /// The roles bound to this picture, in the shot's own order. More than one when the shot binds
    /// several roles that name the same pack file — one photograph of two people (sc-24024).
    pub roles: Vec<BoundReferenceRole<'a>>,
}

impl ReferencePicture<'_> {
    /// The role whose imported asset this picture dispatches — the first one bound to it.
    pub fn dispatch_role(&self) -> Option<&str> {
        self.roles.first().map(|bound| bound.role.as_str())
    }
}

/// THE order a shot's reference images are supplied in, and therefore both the `<Picture N>` the
/// engine labels each one with and the position that image's asset takes in the dispatched
/// `referenceAssetIds` (sc-24023).
///
/// One function, called by the compiler that writes the binding sentences AND by the dispatcher
/// that builds `referenceAssetIds`, because the two numbers are the same number: a prompt that says
/// "the courier is the person shown in `<Picture 2>`" while the courier's asset is dispatched first
/// binds the model to the wrong image, and nothing downstream can detect it. The MiniMax-H3 text
/// encoder labels the supplied assets `<Picture 1>`, `<Picture 2>`, … in supply order, so the
/// numbering is positional and there is no id to check it against.
///
/// AND the de-duplication (sc-24024): roles whose pack entries name the SAME `file` share one
/// picture, numbered at the position of the first of them, because the engine labels each image it
/// is SUPPLIED and a file supplied once is one picture however many roles point at it. A shot
/// binding ten roles across nine files therefore sends nine images and names `<Picture 1>` …
/// `<Picture 9>`.
///
/// "The same file" is the pack's `file` string compared LITERALLY. Nothing here touches the
/// filesystem: the pack's own validator has already refused an absolute path, a `..` component and
/// an unsafe basename, and canonicalizing would make one document mean different things on two
/// machines. Two entries meaning one image must spell its path one way.
///
/// A role the pack does not declare (`entry: None`) can never share, since there is no file to
/// share: it gets a picture of its own. Only an unvalidated plan has one —
/// [`crate::film_plan::validate_plan_against_pack`] refuses an undeclared role.
///
/// A DESCRIBED-ONLY role — declared, but with no `file` (sc-24025) — produces NO picture at all,
/// which is a different answer from the undeclared role's "a picture of its own". There is no
/// image to supply, so numbering one would promise the engine a `<Picture N>` it is never sent and
/// shift every later number by one; and because the reference LIMIT, the route's pre-dispatch
/// payload gate and the dispatched `referenceAssetIds` are all built by walking this list, a role
/// that costs no image must not appear in it. Only an unvalidated plan can bind one —
/// `validate_plan_against_pack` refuses a described-only role in every `conditioning.*` slot — and
/// dropping it here is what makes that refusal the only way it is ever seen.
pub fn shot_reference_pictures<'a>(
    reference_roles: &[String],
    pack: &'a ReferencePack,
) -> Vec<ReferencePicture<'a>> {
    let mut pictures: Vec<ReferencePicture<'a>> = Vec::with_capacity(reference_roles.len());
    let mut by_file: BTreeMap<&'a str, usize> = BTreeMap::new();
    for role in reference_roles {
        let entry = pack.references.iter().find(|entry| &entry.role == role);
        if entry.is_some_and(crate::film_plan::ReferenceEntry::is_described_only) {
            continue;
        }
        let bound = BoundReferenceRole {
            role: role.clone(),
            entry,
        };
        let shared = entry
            .and_then(crate::film_plan::ReferenceEntry::file)
            .and_then(|file| by_file.get(file).copied());
        match shared {
            Some(index) => pictures[index].roles.push(bound),
            None => {
                if let Some(file) = entry.and_then(crate::film_plan::ReferenceEntry::file) {
                    by_file.insert(file, pictures.len());
                }
                pictures.push(ReferencePicture {
                    number: u32::try_from(pictures.len() + 1).unwrap_or(u32::MAX),
                    roles: vec![bound],
                });
            }
        }
    }
    pictures
}

/// A kind of text the COMPILER writes into a prompt, recorded so a reviewer can see exactly what
/// was added and why (sc-24023).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InsertedTextKind {
    /// One sentence per bound reference role, giving that reference a job in the prompt by naming
    /// the `<Picture N>` the engine will label its image with.
    ReferenceBinding,
    /// THE TEXT IDENTITY LOCK (sc-24025): the pack's own description of every `continuityRoles`
    /// entry this shot does NOT bind to an image, repeated word for word.
    ///
    /// Reference images are OPTIONAL in this harness, and a subject no picture conditions on is
    /// held together by nothing but the words used for it. Left to an author or a planner, the
    /// courier is re-worded in every shot and MiniMax-H3 duly renders a different courier; the
    /// compiler instead says the same thing every time, from the one place the pack states it.
    ContinuityDescription,
    /// `Audio: <the shot's own sentence>` — what this shot should sound like (sc-24026).
    Audio,
    /// [`NO_SPEECH_SENTENCE`], on a shot whose dialogue bus the harness already fills (sc-24026).
    NoSpeech,
}

/// Where in the prompt a kind of inserted text sits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertedTextPlacement {
    /// Ahead of the authored or refined prompt.
    Leading,
    /// After it.
    Trailing,
}

impl InsertedTextKind {
    /// Every kind, in prompt order. Exhaustively matched in [`Self::label`] below, so a kind added
    /// to the enum and left out of this list is a compile error rather than a kind the over-length
    /// refusal and the conformance difference silently stop counting (sc-24029).
    pub const ALL: &'static [Self] = &[
        Self::ReferenceBinding,
        Self::ContinuityDescription,
        Self::Audio,
        Self::NoSpeech,
    ];

    /// What this kind of inserted text is CALLED when a finding has to name it to an operator.
    ///
    /// Plain words rather than the variant name, because these appear in refusals a person acts
    /// on: the thing to go and shorten, or the thing that no longer matches.
    pub fn label(self) -> &'static str {
        match self {
            Self::ReferenceBinding => "reference binding sentences",
            Self::ContinuityDescription => {
                "identity text (the pack's description of each continuityRoles entry this shot \
                 does not bind to an image, written in word for word)"
            }
            Self::Audio => "the shot's audio sentence",
            Self::NoSpeech => "the no-spoken-dialogue sentence",
        }
    }

    /// Where this kind sits, and the ONE place that is decided.
    ///
    /// Reference bindings LEAD because the engine presents the reference media before the text: the
    /// binding a picture needs should be the first thing said about it. The audio sentences TRAIL
    /// because they describe a different track from everything before them — a prompt that opens on
    /// sound reads as a film about sound — and because the no-speech sentence only makes sense once
    /// the soundtrack has been described.
    /// The identity lock LEADS too, immediately after the bindings, for the same reason they do:
    /// both say what the shot is OF, and the two together are one block describing every subject
    /// in the film — the bound ones by picture, the rest by description — read before the prompt
    /// that puts those subjects in motion.
    pub fn placement(self) -> InsertedTextPlacement {
        match self {
            Self::ReferenceBinding | Self::ContinuityDescription => InsertedTextPlacement::Leading,
            Self::Audio | Self::NoSpeech => InsertedTextPlacement::Trailing,
        }
    }
}

/// Text the compiler wrote into [`CompiledRequest::prompt`], kept beside the authored prompt so the
/// document says exactly what was added rather than only that the prompt differs (sc-24023).
///
/// Each insertion sits where its kind's [`InsertedTextKind::placement`] puts it, in the order of
/// this list. Inserted text is written AFTER the model's own refine rewrite, so the refiner can
/// never paraphrase a `<Picture N>` into something the engine does not label, nor soften "No audio.
/// Silence." into a suggestion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct InsertedText {
    pub kind: InsertedTextKind,
    pub text: String,
}

/// The noun a binding sentence calls a bound reference, by its pack kind.
///
/// Only [`crate::film_plan::BINDABLE_REFERENCE_KINDS`] reach a `conditioning.referenceRoles` slot —
/// `validate_plan_against_pack` refuses the other two — so the fallback is for a role the pack does
/// not declare at all, which no validated plan has.
fn bound_reference_noun(kind: &str) -> &'static str {
    match kind {
        "character" => "person",
        "prop" => "object",
        "location" => "place",
        _ => "subject",
    }
}

/// `red_parcel` -> `red parcel`: the role as a prompt reads it.
fn role_phrase(role: &str) -> String {
    role.replace(['_', '-'], " ")
}

/// A pack-authored description as compiler-owned text may repeat it: every `\r`, `\n`, `\t` and run
/// of spaces collapsed to a single space, and the ends trimmed (sc-24023).
///
/// THE one place a description is normalized, because every kind of inserted text repeats one — the
/// reference binding sentences here and the shot's own `audio` sentence ([`audio_text`], sc-24026)
/// — and a description that carried a newline or a control run would otherwise land raw in the
/// dispatched prompt inside the one field [`CompiledPlan::conformance_findings`] treats as the
/// compiler's own authored text and therefore never re-reads.
///
/// WHITESPACE only. `<`, `>` and genuine control characters are refused at the document boundary:
/// a pack entry's `description` by [`crate::film_plan::reference_pack_findings`] and a shot's
/// `audio` by `validate_shot_structure`, both through the shared `inserted_prose_findings` helper
/// — because text that forges `<Picture 3>` is an authoring mistake to name rather than something
/// to silently rewrite.
pub fn normalized_description(description: &str) -> String {
    description.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// A pack description as ONE sentence of a prompt, or `None` when the pack says nothing.
///
/// THE one rule for repeating a description, shared by the two places that repeat one (sc-24025):
/// the binding sentence a bound reference gets ([`reference_binding_text`], sc-24023) and the
/// identity lock an unbound continuity role gets ([`continuity_description_text`]). One helper
/// because the two must read identically — the same courier described the same way whether this
/// shot happens to bind her plate — and two copies of "normalize, then add a period" would drift.
///
/// It adds NOTHING but a closing `.` to text that does not already end a sentence. In particular
/// it never prefixes the role name: the pack's descriptions are written as complete sentences
/// naming their own subject ("The courier: blue jacket, carries the parcel."), so prefixing would
/// produce "The courier: The courier: …". That contract is stated on
/// [`crate::film_plan::ReferenceEntry::description`], which is where an author reads it.
///
/// WHITESPACE is normalized first ([`normalized_description`]): the sentence is one line of a
/// prompt, and a description carrying a newline or a tab run would otherwise land raw in the
/// dispatched text (sc-24023).
///
/// "Already ends a sentence" is [`ends_sentence`] and nothing else (sc-24029). This function and
/// the trailing-insertion join are the only two places the compiler decides whether to supply a
/// missing `.`, and a second, narrower rule here put a stray period after every description that
/// ended in `…`, `."` or `.)` — all of which are finished sentences.
fn description_sentence(description: &str) -> Option<String> {
    let mut text = normalized_description(description);
    if text.is_empty() {
        return None;
    }
    if !ends_sentence(&text) {
        text.push('.');
    }
    Some(text)
}

/// THE SHARED-FILE RULE for continuity roles (sc-24025), and the ONE place it is applied: a
/// continuity role the shot does not list in `referenceRoles`, but whose `file` is the file of a
/// picture this shot IS binding, is added to that picture as a bound role — and therefore gets a
/// BINDING sentence with its locator instead of an identity sentence.
///
/// The case is a photograph of two people where the shot binds one of them. The other subject's
/// image is already supplied inside that `<Picture N>`, so an identity sentence would describe
/// somebody visibly present in a supplied picture without tying them to it — the second-subject
/// ambiguity the locator exists to remove, reintroduced. Tying it to the picture says the one
/// thing the prompt is missing: which of the two people in `<Picture 1>` the recipient is.
///
/// It never gets BOTH, because the binding sentence already repeats the pack's description and a
/// description stated twice is emphasis a prompt model acts on. Adding the role here is what makes
/// [`continuity_description_text`] skip it: that function asks the pictures it is given.
///
/// TEXT ONLY. The role is appended to an EXISTING picture, so nothing here changes
/// [`shot_reference_pictures`]' order or numbering, and nothing changes the dispatched
/// `referenceAssetIds`: those are built from each picture's [`ReferencePicture::dispatch_role`],
/// which is the FIRST role on it and therefore always one the shot listed. The image was already
/// being sent; this only names its second subject.
///
/// APPROVED roles only, the rule every other insertion follows, and appended AFTER the listed
/// roles so a picture's binding sentences read in the order the shot asked for them.
fn pictures_with_shared_continuity<'a>(
    shot: &Shot,
    pictures: &[ReferencePicture<'a>],
    pack: &'a ReferencePack,
) -> Vec<ReferencePicture<'a>> {
    let mut augmented = pictures.to_vec();
    for role in &shot.continuity_roles {
        if augmented
            .iter()
            .flat_map(|picture| picture.roles.iter())
            .any(|bound| &bound.role == role)
        {
            continue;
        }
        let Some(entry) = pack
            .references
            .iter()
            .find(|entry| &entry.role == role && entry.approved)
        else {
            continue;
        };
        // A DESCRIBED-ONLY role has no file and can share none: it falls through to the lock.
        let Some(file) = entry.file() else {
            continue;
        };
        let Some(picture) = augmented.iter_mut().find(|picture| {
            picture
                .roles
                .first()
                .and_then(|bound| bound.entry)
                .and_then(crate::film_plan::ReferenceEntry::file)
                == Some(file)
        }) else {
            continue;
        };
        picture.roles.push(BoundReferenceRole {
            role: role.clone(),
            entry: Some(entry),
        });
    }
    augmented
}

/// THE TEXT IDENTITY LOCK for one shot (sc-24025): the pack's description of every
/// `continuityRoles` entry this shot does not bind to an image, in the shot's own role order.
///
/// "Does not bind to an image" is asked of the shot's RESOLVED pictures and its keyframe slots,
/// not of the role's kind or of the plan's declared model, because that is the question that
/// matters: a role whose picture is actually being supplied already has its description in its
/// binding sentence, and saying it twice is the one thing this must not do. Everything else
/// qualifies — a described-only role on any shot, and an image-backed role on a shot that resolved
/// to the base checkpoint or simply did not bind it. Role KIND is irrelevant: a style and a plate
/// drift exactly as a character does when nobody repeats the words.
///
/// UNAPPROVED roles contribute nothing, which is the rule the binding sentences already follow —
/// `validate_plan_against_pack` refuses an unapproved role in a conditioning slot, so no binding
/// sentence has ever described one. `approved` defaults to TRUE, so an unapproved entry is an
/// explicit "do not use this", and text shapes the render exactly as conditioning does.
///
/// An empty description contributes nothing and is not an error here: an image-backed role that
/// says nothing still shows its picture. A DESCRIBED-only role cannot reach this state — the pack
/// validator refuses one with no description, because it would be a role that is nothing at all.
///
/// A role SHARING a bound picture's file gets no identity sentence either, and it is
/// [`pictures_with_shared_continuity`] that says so rather than a rule here: it has already added
/// the role to that picture, so the `pictures` this reads report it bound. Its image is being
/// supplied, so it is named by picture with its locator — see that function for why.
fn continuity_description_text(
    shot: &Shot,
    pictures: &[ReferencePicture<'_>],
    pack: &ReferencePack,
) -> Option<InsertedText> {
    let image_bound = |role: &str| {
        pictures
            .iter()
            .flat_map(|picture| picture.roles.iter())
            .any(|bound| bound.role == role)
            || [
                shot.conditioning.first_frame_role.as_deref(),
                shot.conditioning.last_frame_role.as_deref(),
            ]
            .into_iter()
            .flatten()
            .any(|frame| frame == role)
    };
    // ONE sentence per SUBJECT, at its first mention. `continuityRoles` is a free list and nothing
    // refuses a role written into it twice, but a subject described twice is the compiler
    // contradicting its own purpose: the lock exists to say one fixed thing about each subject, and
    // repetition is emphasis a prompt model acts on.
    let mut described: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    let sentences: Vec<String> = shot
        .continuity_roles
        .iter()
        .filter(|role| !image_bound(role))
        .filter_map(|role| {
            pack.references
                .iter()
                .find(|entry| entry.role == *role && entry.approved)
        })
        .filter(|entry| described.insert(entry.role.as_str()))
        .filter_map(|entry| description_sentence(&entry.description))
        .collect();
    (!sentences.is_empty()).then(|| InsertedText {
        kind: InsertedTextKind::ContinuityDescription,
        text: sentences.join(" "),
    })
}

/// The binding sentences for one shot: one per bound role, in picture order, each naming the
/// `<Picture N>` that role's image will be labelled with and repeating the pack's own description
/// of it verbatim (sc-24023).
///
/// Plain declarative sentences and nothing else — no emphasis, no imperatives, no restating of the
/// shot. The prompt guide's rule is that a reference needs a job ("the woman from `<Picture 1>`");
/// this is that job, stated once per reference.
///
/// "Bound role" is every role on the picture, which since sc-24025 includes a `continuityRoles`
/// entry whose file this shot is already supplying for some other role
/// ([`pictures_with_shared_continuity`]) — a second subject inside a picture that is being sent is
/// named here, by picture and locator, rather than described on its own by the identity lock.
/// It gets a binding sentence and no identity sentence; the picture's own number and its
/// dispatched asset are untouched.
fn reference_binding_text(pictures: &[ReferencePicture<'_>]) -> Option<InsertedText> {
    let mut sentences: Vec<String> = Vec::new();
    for picture in pictures {
        for bound in &picture.roles {
            // With a LOCATOR the sentence says WHICH subject in that picture this role is
            // (sc-24024) — "The courier is the woman on the left in <Picture 1>." — which is the
            // only thing that makes one photograph of two people bindable as two roles, since both
            // are bound to the same `<Picture N>`. Without one it keeps the wording a reference
            // with an image to itself has always had. The locator's whitespace is normalized like
            // every other pack-authored phrase the compiler repeats.
            let locator = bound
                .entry
                .and_then(crate::film_plan::ReferenceEntry::locator)
                .map(normalized_description)
                .filter(|locator| !locator.is_empty());
            let kind = bound.entry.map_or("", |entry| entry.kind.as_str());
            let mut sentence = match locator {
                Some(locator) => format!(
                    "The {} is {locator} in <Picture {}>.",
                    role_phrase(&bound.role),
                    picture.number
                ),
                None => format!(
                    "The {} is the {} shown in <Picture {}>.",
                    role_phrase(&bound.role),
                    bound_reference_noun(kind),
                    picture.number
                ),
            };
            // The pack's description is the author's own words about that image, so it is repeated
            // rather than paraphrased into the sentence above.
            if let Some(description) = bound
                .entry
                .and_then(|entry| description_sentence(&entry.description))
            {
                sentence.push(' ');
                sentence.push_str(&description);
            }
            sentences.push(sentence);
        }
    }
    (!sentences.is_empty()).then(|| InsertedText {
        kind: InsertedTextKind::ReferenceBinding,
        text: sentences.join(" "),
    })
}

/// The shot's audio sentence as the prompt states it (sc-24026): `Audio: ` and the author's own
/// words, whitespace-normalized exactly as a pack description is and otherwise verbatim.
///
/// The content is NEVER inspected. "No audio. Silence." is a complete answer, and a compiler that
/// tried to tell silence from sound would have to guess at prose — the plan already refused a shot
/// that said nothing, which is the only question worth asking here.
fn audio_text(shot: &Shot) -> Option<InsertedText> {
    let audio = normalized_description(&shot.audio);
    (!audio.is_empty()).then(|| InsertedText {
        kind: InsertedTextKind::Audio,
        text: format!("{AUDIO_PROMPT_PREFIX} {audio}"),
    })
}

/// [`NO_SPEECH_SENTENCE`], on a shot that PLACES a dialogue line (sc-24026).
///
/// Keyed on [`Shot::dialogue_clip`], which is the placement: `dialogue` beside it is intent prose
/// the run never plays, while a clip names a pack `sound` entry of kind `dialogue` that
/// `ensure_sound` either imports from disk or synthesizes and then imports. Pre-recorded or spoken,
/// both put OUR voice on the dialogue bus, so both are the reason not to let H3 add a second one. A
/// shot with no clip gets nothing: it has no voice to double.
fn no_speech_text(shot: &Shot) -> Option<InsertedText> {
    shot.dialogue_clip.as_ref().map(|_| InsertedText {
        kind: InsertedTextKind::NoSpeech,
        text: NO_SPEECH_SENTENCE.to_owned(),
    })
}

/// Everything the compiler writes into a shot's prompt, in the order the prompt reads it.
///
/// The one place an insertion kind is produced: a later kind of compiler-owned text is another
/// entry pushed here, and needs no change to the placement, the record or the conformance check.
fn inserted_text_for_shot<'a>(
    shot: &Shot,
    pictures: &[ReferencePicture<'a>],
    pack: &'a ReferencePack,
) -> Vec<InsertedText> {
    let pictures = pictures_with_shared_continuity(shot, pictures, pack);
    reference_binding_text(&pictures)
        .into_iter()
        .chain(continuity_description_text(shot, &pictures, pack))
        .chain(audio_text(shot))
        .chain(no_speech_text(shot))
        .collect()
}

/// Does `text` already end a sentence? (sc-24026)
///
/// A trailing insertion is appended after text nobody guaranteed was punctuated — an authored
/// prompt, a refiner rewrite, or the author's own `audio` sentence — and a bare space between them
/// produces a run-on the model reads as one clause: `...a courier enters Audio: Room tone`. The
/// answer is the last non-whitespace character, accepting a closing quote or bracket that itself
/// closes a punctuated sentence (`"...he said." )`).
fn ends_sentence(text: &str) -> bool {
    let mut chars = text.trim_end().chars().rev();
    let Some(last) = chars.next() else {
        // Nothing to join to: the caller writes no separator before the first piece anyway.
        return true;
    };
    let terminal = |value: char| matches!(value, '.' | '!' | '?' | '…');
    terminal(last)
        || (matches!(last, '"' | '\'' | ')' | ']' | '»' | '”' | '’')
            && chars.next().is_some_and(terminal))
}

/// Push the separator between the text composed so far and the next trailing insertion: a sentence
/// boundary, supplying the missing `.` when the composed text does not already end one (sc-24026).
fn push_sentence_break(composed: &mut String) {
    if composed.trim_end().is_empty() {
        // Nothing precedes this piece, so there is no boundary to draw and no leading space to add.
        return;
    }
    if !ends_sentence(composed) {
        composed.push('.');
    }
    composed.push(' ');
}

/// The prompt the engine receives: the leading insertions, in order, then the refined or authored
/// prompt, then the trailing ones.
///
/// Trailing pieces are joined by a SENTENCE boundary rather than a bare space, because neither the
/// authored prompt, the refiner's rewrite nor the author's `audio` text is guaranteed to end in
/// terminal punctuation and the compiler's own sentences must not be swallowed into whatever
/// precedes them (sc-24026).
///
/// The prompt is trimmed at BOTH ends (sc-24029). A leading insertion already ends in a single
/// space, so an authored prompt that opens with whitespace — which nothing refuses, and which a
/// text area produces readily — used to be joined to the bindings by two. Trimming both ends is
/// also what makes the composition reversible: `CompiledPlan::conformance_findings` recovers the
/// middle of a dispatched prompt by stripping the expected prefix and suffix, and it can only do
/// that if the compiler's own join added nothing it cannot predict.
fn apply_inserted_text(prompt: &str, inserted: &[InsertedText]) -> String {
    if inserted.is_empty() {
        return prompt.trim().to_owned();
    }
    let mut composed = String::new();
    for piece in inserted
        .iter()
        .filter(|piece| piece.kind.placement() == InsertedTextPlacement::Leading)
    {
        composed.push_str(piece.text.trim());
        composed.push(' ');
    }
    composed.push_str(prompt.trim());
    for piece in inserted
        .iter()
        .filter(|piece| piece.kind.placement() == InsertedTextPlacement::Trailing)
    {
        push_sentence_break(&mut composed);
        composed.push_str(piece.text.trim());
    }
    composed
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CompiledModel {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier: Option<String>,
    pub fps: u32,
    pub lane: String,
}

/// One shot, resolved to everything the video route needs except the run-scoped ids and the asset
/// ids the reference roles import to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CompiledRequest {
    pub shot_id: String,
    pub beat: String,
    pub mode: String,
    /// The catalog model id this request DISPATCHES as: the partition the shot resolved to, which
    /// on a split family (MiniMax-H3's `minimax_h3` / `minimax_h3_ref`) is not the plan's declared
    /// model (sc-23402). [`CompiledRequest::to_job_body_with`] writes exactly this into the job
    /// body's `model`, so `compiled.json` and the route agree by construction.
    pub model: String,
    /// The short edge this request's image references are encoded at, in pixels, when the plan asked
    /// for one (`model.advanced.referenceImageShortEdge`, sc-23402).
    ///
    /// Written ONLY for a request that resolved to the family's reference partition: the base
    /// partition encodes no reference, so carrying the knob there would dispatch a field the
    /// checkpoint has nothing to apply it to. Absent means the engine's own default, which
    /// [`CompiledRequest::effective_reference_image_short_edge`] resolves for the record.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_image_short_edge: Option<u32>,
    /// The catalog LoRA ids this request dispatches with, resolved PER PARTITION from the plan's
    /// one `model.loras` list (sc-23406).
    ///
    /// A step-distill adapter declares the partitions it was distilled for, so the same plan-level
    /// list produces the ref2v turbo on a `minimax_h3_ref` request and the fl2v turbo on a
    /// `minimax_h3` one — and neither on a request whose partition has no compatible entry, which
    /// is an empty list rather than a silent substitution.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub loras: Vec<String>,
    /// `model.advanced.steps`, when the plan set one — the value DISPATCHED as `advanced.steps`.
    /// `None` leaves the step count to the recipe or the model default, which is what
    /// [`CompiledRequest::effective_steps`] records.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<u32>,
    /// The model-evaluation count this request will actually render at: `steps` above, else the
    /// selected turbo recipe's own count, else the partition's declared `defaults.steps`.
    ///
    /// Resolved HERE rather than on read because only the compile holds all three inputs at once —
    /// the plan's override, the partition this shot resolved to, and that partition's catalog
    /// entry. The attempt record copies it, so the document and the record state one number.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_steps: Option<u32>,
    /// The video sigma shift the selected turbo recipe imposes, when one applies to this request's
    /// partition. Absent in the base regime, where the engine's own constant governs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turbo_scheduler_shift: Option<f64>,
    /// Why this request resolved to `model` and not the family's other partition
    /// ([`crate::film_plan::ShotPartition::reason`]). Derived, like every other field but the
    /// prompt: [`CompiledPlan::conformance_findings`] refuses a hand-edited one.
    #[serde(default)]
    pub partition_reason: String,
    /// The prompt the engine will receive: [`Self::inserted_text`], in order, around the authored
    /// or refined text — bindings leading it, the audio sentences trailing it.
    pub prompt: String,
    pub prompt_source: PromptSource,
    /// The plan's own prompt, kept when `prompt` was refined so the rewrite can be reviewed and
    /// reverted by editing the plan.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authored_prompt: Option<String>,
    /// What the COMPILER wrote into `prompt`, per kind, separately from the authored text
    /// (sc-24023). A reviewer reads this to see exactly what was added without diffing two
    /// paragraphs, and it is DERIVED like every other field but the prompt itself:
    /// [`CompiledPlan::conformance_findings`] refuses a hand-edited one.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inserted_text: Vec<InsertedText>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    pub duration_seconds: f64,
    pub fps: u32,
    pub width: u32,
    pub height: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_frame_role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_frame_role: Option<String>,
    #[serde(default)]
    pub reference_roles: Vec<String>,
    /// Declared continuity intent, carried through to the job so a take's provenance records it.
    /// It never becomes conditioning: the frames this request is conditioned on are the canonical
    /// roles above.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chain_from_shot_id: Option<String>,
    #[serde(default)]
    pub continuity_roles: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CompiledPlan {
    pub schema_version: u32,
    pub plan_id: String,
    pub plan_version: u32,
    /// SHA-256 of the plan document these requests were compiled from. A plan edited after the
    /// compile no longer matches, and the harness refuses to dispatch stale requests.
    pub plan_sha256: String,
    pub reference_pack_id: String,
    pub reference_pack_version: u32,
    /// [`reference_pack_sha256`] of the pack these requests were compiled against (sc-24029).
    ///
    /// The pack is an INPUT to the prompt, not only to the conditioning: every reference binding
    /// sentence and every identity-lock sentence is written out of a pack entry's own description
    /// and locator. Editing a pack description is a first-class workspace action, so without this
    /// key a compiled document survived the edit marked current and was then refused by
    /// [`request_differences`] — the wrong cause, naming a derived field instead of the pack.
    ///
    /// It hashes the PARSED pack, so the CLI's JSONC document and the workspace's typed pack agree
    /// on one value and a comment- or whitespace-only edit of the document does not stale a
    /// compile. See [`reference_pack_sha256`] for why that rather than the file's bytes.
    ///
    /// `#[serde(default)]` so a v6 document still DECODES and is then refused by
    /// [`COMPILED_PLAN_SCHEMA_VERSION`], which is the refusal that names the remedy — the pattern
    /// every earlier field addition follows.
    #[serde(default)]
    pub reference_pack_sha256: String,
    pub compiled_at: String,
    pub model: CompiledModel,
    pub requests: Vec<CompiledRequest>,
    /// What the planner's LLM decodes cost to produce this document (sc-22715): the jobs it
    /// created through the `prompt_refine` seam, their wall-clock, and the peak memory their
    /// metrics blocks reported. Absent on a plan compiled with no LLM at all (`--no-refine` over a
    /// hand-authored plan), present on every generated or refined one, so the planner's cost is
    /// persisted beside the requests it produced rather than only printed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub planner: Option<PlannerCostRecord>,
}

/// The cost of the planner's LLM work, persisted into `compiled.json` (sc-22715).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PlannerCostRecord {
    /// Every `prompt_refine` job this document's planning and refinement created, in order: the
    /// plan draft, each repair round, then one rewrite per shot.
    pub job_ids: Vec<String>,
    /// Wall-clock the LLM jobs took end to end, summed.
    pub elapsed_seconds: f64,
    /// Highest `peakMemoryBytes` any of those jobs' metrics blocks reported; `None` when no job
    /// reported one (a worker whose probe measured nothing posts no block).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_memory_bytes: Option<u64>,
    /// Repair rounds actually taken for the plan draft (0 when the first draft validated, and 0 on
    /// a `compile` of an existing plan).
    pub repair_rounds: u32,
    /// The budget the brief declared for those decodes (`limits.plannerMaxMemoryGb`), so the
    /// record carries the bound beside the measurement.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub planner_max_memory_gb: Option<f64>,
    /// One entry per LLM call, preserving the actual planner checkpoint separately from the target
    /// video model. Thinking is kept out of the accepted plan text and stored only in its own field.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub executions: Vec<PlannerExecutionRecord>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PlannerExecutionRecord {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
    pub provider: String,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    pub target_video_model_id: String,
    pub thinking_mode: String,
    /// Effective request bound advertised to an OpenAI-compatible planner. Native executions omit
    /// this because their token bound belongs to the worker job payload instead.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    /// Effective per-request wall-clock bound, retained even for failed dispatch attempts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_timeout_seconds: Option<u64>,
    /// Effective sampler temperature when explicitly controlled by the adapter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Whether approved reference pixels were included in the attempted request payload.
    /// This is not an acknowledgement of server receipt after a network failure. Kept separate
    /// from reference roles so the external-data boundary remains explicit in provenance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_pixels_sent: Option<bool>,
    /// Wall-clock time spent waiting for this individual planner response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_seconds: Option<f64>,
    /// Sanitized provider completion reason, when the compatible endpoint reports one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    /// Stable failure classification for a provider response that could not become plan text.
    /// Human-readable, actionable detail remains on the planning operation finding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    /// Provider-reported token counts. Optional because OpenAI-compatible servers are allowed to
    /// omit usage, but when present the sanitized counts travel with the plan provenance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<PlannerUsageRecord>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PlannerUsageRecord {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
}

/// The model-evaluation count a catalog entry declares as its default (`defaults.steps`) — what
/// the engine renders at when nothing names a count (sc-23406).
fn default_steps(entry: &JsonObject<String, Value>) -> Option<u32> {
    entry
        .get("defaults")
        .and_then(Value::as_object)
        .and_then(|defaults| defaults.get("steps"))
        .and_then(Value::as_u64)
        .and_then(|steps| u32::try_from(steps).ok())
        .filter(|steps| *steps > 0)
}

/// `advanced.mlxQuantize` for a tier — the shared convention the MLX lanes read.
pub fn mlx_quantize_for_tier(tier: &str) -> Value {
    match tier {
        "bf16" => json!(0),
        "q8" => json!(8),
        _ => json!(4),
    }
}

/// Inputs the compile needs beyond the plan itself.
pub struct CompileInputs<'a> {
    /// The catalog entries the plan's shots resolve against: the declared model, plus the family's
    /// reference partition when the catalog serves one. Each shot's geometry defaults come from the
    /// entry it will actually dispatch as (sc-23402).
    pub entries: &'a ModelEntries<'a>,
    pub lane: &'a str,
    pub plan_sha256: &'a str,
    pub compiled_at: &'a str,
    /// Refined prompt text per shot id. A shot with no entry compiles its authored prompt.
    pub refined_prompts: &'a BTreeMap<String, String>,
}

/// Compile every shot of `plan`. Returns findings instead of a document when a shot cannot be
/// expressed as a request — an unresolvable geometry, or a refined prompt that is empty or longer
/// than the route accepts. Nothing is truncated or substituted: an unusable rewrite is a finding
/// naming the shot, so the fix is to re-run the refinement or edit the plan, never to ship a
/// silently shortened prompt.
pub fn compile_plan(
    plan: &ProductionPlan,
    pack: &ReferencePack,
    inputs: &CompileInputs<'_>,
) -> Result<CompiledPlan, Vec<PlanDiagnostic>> {
    let mut findings = Vec::new();
    let Some(fps) = crate::film_plan::plan_fps(plan, inputs.entries.base_entry()) else {
        return Err(vec![PlanDiagnostic::plan(
            "model.fps",
            format!(
                "{} declares no default fps; set model.fps in the plan before compiling",
                plan.model.id
            ),
        )]);
    };
    let mut requests = Vec::with_capacity(plan.shots.len());
    for shot in &plan.shots {
        match compile_shot(plan, shot, pack, inputs, fps) {
            Ok(request) => requests.push(request),
            Err(mut shot_findings) => findings.append(&mut shot_findings),
        }
    }
    if !findings.is_empty() {
        return Err(findings);
    }
    let reference_pack_sha256 = reference_pack_sha256(pack).map_err(|error| {
        vec![PlanDiagnostic::plan(
            "referencePack",
            format!("this reference pack cannot be serialized to identify it: {error}"),
        )]
    })?;
    Ok(CompiledPlan {
        schema_version: COMPILED_PLAN_SCHEMA_VERSION,
        plan_id: plan.id.clone(),
        plan_version: plan.version,
        plan_sha256: inputs.plan_sha256.to_owned(),
        reference_pack_id: pack.id.clone(),
        reference_pack_version: pack.version,
        reference_pack_sha256,
        compiled_at: inputs.compiled_at.to_owned(),
        model: CompiledModel {
            id: plan.model.id.clone(),
            tier: plan.model.tier.clone(),
            fps,
            lane: inputs.lane.to_owned(),
        },
        requests,
        planner: None,
    })
}

fn compile_shot(
    plan: &ProductionPlan,
    shot: &Shot,
    pack: &ReferencePack,
    inputs: &CompileInputs<'_>,
    fps: u32,
) -> Result<CompiledRequest, Vec<PlanDiagnostic>> {
    // Which of the family's checkpoints this shot dispatches as, decided ONCE here and carried into
    // the request, the job body and the attempt record (sc-23402). A shot that binds no reference
    // roles stays on the plan's declared model: references are optional input, never a requirement.
    let (partition, partition_entry) = inputs.entries.resolve_shot(shot);
    let Some(partition_entry) = partition_entry else {
        return Err(vec![PlanDiagnostic::shot(
            &shot.id,
            "conditioning.referenceRoles",
            format!(
                "{} is not in this API's model catalog, so this shot cannot be compiled ({})",
                partition.model_id, partition.reason
            ),
        )]);
    };
    let Some((width, height)) = shot_resolution(plan, shot, partition_entry) else {
        return Err(vec![PlanDiagnostic::shot(
            &shot.id,
            "resolution",
            format!(
                "{} declares no default resolution; set one on the plan or the shot",
                partition.model_id
            ),
        )]);
    };
    let (prompt, prompt_source, authored) = match inputs.refined_prompts.get(&shot.id) {
        Some(refined) => {
            let trimmed = refined.trim();
            let length = trimmed.chars().count();
            if trimmed.is_empty() || length > MAX_PROMPT_CHARS {
                return Err(vec![PlanDiagnostic::shot(
                    &shot.id,
                    "prompt",
                    format!(
                        "the refined prompt is {length} characters, outside the 1-{MAX_PROMPT_CHARS} \
                         the video route accepts; re-run the refinement or compile with --no-refine \
                         (the authored prompt is not silently substituted)"
                    ),
                )]);
            }
            // THE REFINER MAY NOT WRITE AN ENGINE LABEL (sc-24029). The rewrite is produced by a
            // language model that is handed the model's own prompt guide, and that guide teaches
            // `<Picture N>` as the way to give a reference a job — so the text most likely to come
            // back carrying a label is exactly this text. The worker's marker filter deliberately
            // KEEPS `<Picture N>`, and nothing downstream reads the dispatched prompt back against
            // the pictures the shot actually supplies, so a label survives here into a binding to
            // an image this request never sends.
            //
            // Refused rather than stripped: a rewrite that named a picture is a rewrite built
            // around one, and deleting the label would leave the sentence that depends on it. The
            // AUTHORED branch below is deliberately NOT scanned — a person who types `<Picture 1>`
            // into a plan means it, which is the same line `film_planner::anchoring_findings`
            // draws between a draft and a hand-authored document.
            if crate::film_plan::engine_label_at(trimmed).is_some() {
                return Err(vec![PlanDiagnostic::shot(
                    &shot.id,
                    "prompt",
                    format!(
                        "the refined prompt for {} contains {}, a label this film's renderer \
                         assigns itself: the compiler writes every such label after the rewrite, \
                         numbered from this shot's referenceRoles, so one written here names a \
                         picture the request never supplies. Re-run the refinement or use \
                         --no-refine (the refined prompt is not silently edited)",
                        shot.id,
                        crate::film_plan::quoted_engine_label(trimmed),
                    ),
                )]);
            }
            (
                trimmed.to_owned(),
                PromptSource::Refined,
                Some(shot.prompt.clone()),
            )
        }
        None => (shot.prompt.clone(), PromptSource::Authored, None),
    };
    // THE INSERTION STEP (sc-24023). It runs HERE — after the refine rewrite has already been
    // chosen above — because the refiner is a language model: text handed to it comes back
    // paraphrased, and a paraphrased `<Picture 2>` is a binding to an image the engine never
    // labelled that way. Writing it afterwards makes the compiler, not the model, the author of
    // every word the engine reads that the plan did not write.
    let inserted_text = inserted_text_for_shot(
        shot,
        &shot_reference_pictures(&shot.conditioning.reference_roles, pack),
        pack,
    );
    let prompt = apply_inserted_text(&prompt, &inserted_text);
    let length = prompt.chars().count();
    if length > MAX_PROMPT_CHARS {
        // What the COMPILER contributed, per kind, so the refusal points at the text the author
        // has to go and shorten rather than at the total. EVERY kind is counted (sc-24029): the
        // audio and no-speech sentences trail every request since sc-24026, so a message that
        // reported only the two leading kinds could account for a few hundred characters of an
        // over-length prompt and leave the rest of the compiler's own text unexplained. The
        // identity text is named at length because it is the one an author is least likely to
        // suspect: it appears on shots that bind nothing at all, and it is written from a pack
        // description that shot never mentions (sc-24025).
        let inserted_chars = |kind: InsertedTextKind| -> usize {
            inserted_text
                .iter()
                .filter(|piece| piece.kind == kind)
                .map(|piece| piece.text.chars().count())
                .sum()
        };
        let contributions = InsertedTextKind::ALL
            .iter()
            .map(|kind| format!("{} as {}", inserted_chars(*kind), kind.label()))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(vec![PlanDiagnostic::shot(
            &shot.id,
            "prompt",
            format!(
                "this shot's prompt is {length} characters once the compiler's own sentences lead \
                 and trail it, outside the 1-{MAX_PROMPT_CHARS} the video route accepts; shorten \
                 the shot's prompt, its audio sentence, or the pack descriptions the compiler \
                 repeats. Of those characters the compiler wrote {contributions} (nothing is \
                 silently truncated)"
            ),
        )]);
    }
    // A reference-only knob reaches a reference-only request (sc-23402). The plan declares it once
    // on the family; the shots that resolve to the base partition encode no reference, so the field
    // is not written onto them and never reaches their job body or their attempt record.
    let reference_image_short_edge = plan
        .model
        .advanced
        .as_ref()
        .and_then(|advanced| advanced.reference_image_short_edge)
        .filter(|_| is_reference_partition_id(&partition.model_id));
    // The plan's one LoRA list, resolved against the partition this shot ACTUALLY dispatches as
    // (sc-23406). A shot whose partition has no compatible entry gets none — the empty list is the
    // record of that, and `effective_steps` below then resolves to the model's own default.
    let loras: Vec<String> = plan_loras_for_partition(&plan.model.loras, &partition.model_id)
        .into_iter()
        .map(|lora| lora.id.clone())
        .collect();
    // Resolved through the SAME resolver the worker calls, on the same payload shape, so the
    // schedule this document promises is the schedule the engine runs. A conflict is refused by
    // `validate_plan_structure` before a compile is attempted; reaching it here means the plan was
    // compiled anyway, and a finding beats compiling a request with two schedules in it.
    let payload_loras = plan_lora_payload_entries(&plan.model.loras, &partition.model_id);
    let recipe = match resolve_turbo_recipe(&partition.model_id, &payload_loras) {
        Ok(recipe) => recipe,
        Err(error) => {
            return Err(vec![PlanDiagnostic::shot(&shot.id, "model.loras", error)]);
        }
    };
    let steps = plan
        .model
        .advanced
        .as_ref()
        .and_then(|advanced| advanced.steps)
        .and_then(|steps| u32::try_from(steps).ok())
        .filter(|steps| *steps > 0);
    let effective_steps = steps
        .or_else(|| recipe.map(|recipe| recipe.steps))
        .or_else(|| default_steps(partition_entry));
    let turbo_scheduler_shift = recipe.map(|recipe| f64::from(recipe.video_shift));
    Ok(CompiledRequest {
        shot_id: shot.id.clone(),
        beat: shot.beat.clone(),
        mode: shot.conditioning.mode.clone(),
        model: partition.model_id,
        reference_image_short_edge,
        loras,
        steps,
        effective_steps,
        turbo_scheduler_shift,
        partition_reason: partition.reason,
        prompt,
        prompt_source,
        authored_prompt: authored,
        inserted_text,
        negative_prompt: shot.negative_prompt.clone(),
        duration_seconds: shot.target_duration_seconds,
        fps,
        width,
        height,
        seed: shot.seed,
        first_frame_role: shot.conditioning.first_frame_role.clone(),
        last_frame_role: shot.conditioning.last_frame_role.clone(),
        reference_roles: shot.conditioning.reference_roles.clone(),
        chain_from_shot_id: shot.conditioning.chain_from_shot_id.clone(),
        continuity_roles: shot.continuity_roles.clone(),
    })
}

/// The run-scoped facts a compiled request needs to become a job body.
pub struct DispatchContext<'a> {
    pub project_id: &'a str,
    pub run_id: &'a str,
    pub plan_id: &'a str,
    pub plan_version: u32,
    pub attempt: u32,
    pub tier: Option<&'a str>,
    /// The key this attempt dispatches under, stamped into the job's `filmHarness` provenance so a
    /// controller that died between the POST and the record write finds its OWN job instead of
    /// enqueuing a second render for the same attempt (sc-22711). `None` leaves it out.
    pub idempotency_key: Option<&'a str>,
    /// The approved pack the roles below were imported from. Dispatch reads it for ONE thing: the
    /// reference order ([`shot_reference_pictures`]), so the position an asset takes in
    /// `referenceAssetIds` is decided by the same function that numbered the `<Picture N>` in the
    /// prompt (sc-24023).
    pub pack: &'a ReferencePack,
    /// Reference role -> imported asset id.
    pub role_assets: &'a BTreeMap<String, String>,
}

/// The reference assets one request is conditioned on, as the run record stores them.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResolvedConditioning {
    pub first_frame_asset_id: Option<String>,
    pub last_frame_asset_id: Option<String>,
    pub reference_asset_ids: Vec<String>,
}

impl CompiledRequest {
    /// The reference-image short edge this request will actually render at, for the attempt record
    /// (sc-23402) — or `None` for a request that encodes no reference at all.
    ///
    /// `Some(requested)`, `Some(default)` and `None` are three different facts: a reference request
    /// that named no value still renders at the engine's default, and recording that number is what
    /// makes a run comparable against one that lowered it. A base-partition request has no reference
    /// to size, so it records nothing rather than a number that never applied.
    ///
    /// The default comes from [`effective_reference_image_short_edge`], the local twin of gen-core's
    /// `effective_reference_image_short_edge` (this crate has no gen-core dependency), so the
    /// recorded value cannot drift from the value the engine resolved.
    pub fn effective_reference_image_short_edge(&self) -> Option<u32> {
        is_reference_partition_id(&self.model)
            .then(|| effective_reference_image_short_edge(self.reference_image_short_edge))
    }

    /// Resolve this request's reference roles against the imported assets. A role with no asset is
    /// a finding — the run never dispatches a keyframe shot with its keyframe quietly missing.
    ///
    /// The reference list is built by walking [`shot_reference_pictures`] — the same function the
    /// compiler numbered this request's `<Picture N>` with — so the position of an asset here and
    /// the number in the prompt are one decision rather than two that agree by coincidence
    /// (sc-24023).
    pub fn resolve_conditioning(
        &self,
        pack: &ReferencePack,
        role_assets: &BTreeMap<String, String>,
    ) -> Result<ResolvedConditioning, Vec<PlanDiagnostic>> {
        let mut findings = Vec::new();
        let mut resolve = |field: &str, role: &str| -> Option<String> {
            match role_assets.get(role) {
                Some(asset) => Some(asset.clone()),
                None => {
                    findings.push(PlanDiagnostic::shot(
                        &self.shot_id,
                        field,
                        format!("reference role {role:?} was never imported as an asset"),
                    ));
                    None
                }
            }
        };
        let first = self
            .first_frame_role
            .as_deref()
            .and_then(|role| resolve("conditioning.firstFrameRole", role));
        let last = self
            .last_frame_role
            .as_deref()
            .and_then(|role| resolve("conditioning.lastFrameRole", role));
        let pictures = shot_reference_pictures(&self.reference_roles, pack);
        let references: Vec<String> = pictures
            .iter()
            .filter_map(|picture| picture.dispatch_role())
            .filter_map(|role| resolve("conditioning.referenceRoles", role))
            .collect();
        if findings.is_empty() {
            Ok(ResolvedConditioning {
                first_frame_asset_id: first,
                last_frame_asset_id: last,
                reference_asset_ids: references,
            })
        } else {
            Err(findings)
        }
    }

    /// The `POST /api/v1/video/jobs` body for one attempt. The single place a video job body is
    /// built, for the generated and the hand-authored path alike.
    pub fn to_job_body(&self, context: &DispatchContext<'_>) -> Result<Value, Vec<PlanDiagnostic>> {
        let assets = self.resolve_conditioning(context.pack, context.role_assets)?;
        Ok(self.to_job_body_with(context, &assets))
    }

    /// [`to_job_body`](Self::to_job_body) with the conditioning already resolved.
    pub fn to_job_body_with(
        &self,
        context: &DispatchContext<'_>,
        assets: &ResolvedConditioning,
    ) -> Value {
        let mut advanced = JsonObject::new();
        if let Some(tier) = context.tier {
            advanced.insert("mlxQuantize".to_owned(), mlx_quantize_for_tier(tier));
        }
        if let Some(steps) = self.steps {
            // The plan-level override, dispatched on the same `advanced` convention the Video
            // Studio uses and read by the same worker branch (`minimax_h3_sampling`), where it
            // wins over a selected recipe's own count.
            advanced.insert("steps".to_owned(), json!(steps));
        }
        if let Some(edge) = self.reference_image_short_edge {
            // The same `advanced` convention as the tier (sc-23402): a request axis the engine reads
            // off the job, not a document axis. Only ever present on a reference-partition request,
            // because `compile_shot` is the only thing that writes the field.
            advanced.insert("referenceImageShortEdge".to_owned(), json!(edge));
        }
        let mut provenance = json!({
            "runId": context.run_id,
            "planId": context.plan_id,
            "planVersion": context.plan_version,
            "shotId": self.shot_id,
            "attempt": context.attempt,
        });
        if let Some(key) = context.idempotency_key {
            provenance["idempotencyKey"] = json!(key);
        }
        if !self.partition_reason.is_empty() {
            // The dispatched body says which of the family's checkpoints it asked for AND why
            // (sc-23402): `model` above is the resolved id, and this is the sentence that explains
            // it, so a job read back on its own carries the same two facts as the attempt record.
            provenance["partitionReason"] = json!(self.partition_reason);
        }
        if self.prompt_source == PromptSource::Refined {
            provenance["promptSource"] = json!("refined");
        }
        if let Some(chain) = &self.chain_from_shot_id {
            // Recorded so a take's provenance says which shot it was meant to continue. It is not
            // conditioning: the keyframe/reference slots above are the only anchors.
            provenance["chainFromShotId"] = json!(chain);
        }
        if !self.continuity_roles.is_empty() {
            // The canonical roles this shot was written to depict. Recorded for the same reason as
            // the chain and with the same status — traceability, never conditioning — so a take's
            // provenance can say which approved references it was meant to carry even on a model
            // whose declared `maxReferenceAssets` is 0 and whose shots are all `text_to_video`.
            provenance["continuityRoles"] = json!(self.continuity_roles);
        }
        advanced.insert("filmHarness".to_owned(), provenance);
        let mut body = json!({
            "projectId": context.project_id,
            "mode": self.mode,
            "model": self.model,
            "prompt": self.prompt,
            "duration": self.duration_seconds,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "fitMode": "crop",
            "requestedGpu": "auto",
            "advanced": advanced,
        });
        if !self.loras.is_empty() {
            // `{ id, weight }` per entry — the exact shape `generationStudio.jsx` posts for a
            // studio selection, so the route's `hydrate_lora_spec` hydrates it from the catalog the
            // same way and the worker's `resolve_turbo_recipe` reads the same ids. The weight is
            // the catalog's own `defaultWeight`, which is what the route would have filled in for
            // an id alone; sending it explicitly keeps the body readable beside a studio job.
            body["loras"] = json!(plan_lora_payload_entries(&self.loras, &self.model));
        }
        if let Some(negative) = self.negative_prompt.as_deref() {
            body["negativePrompt"] = json!(negative);
        }
        // Attempt `n` renders at `seed + (n - 1) * ATTEMPT_SEED_STRIDE` (sc-22715). The MLX render
        // is deterministic for a seed — two runs of the fixture's SH010 at seed 22710 were
        // pixel-identical — so a replacement that kept the plan's seed would re-render the very
        // take it rejects. The plan's seed is still attempt 1, the derivation is recorded
        // (`filmHarness.seed`, and the take's recipe), and it is the only thing about the
        // dispatched request that varies by attempt: the prompt, the geometry and the conditioning
        // are the compiled request's.
        //
        // The STRIDE is what keeps a seed unique per (shot, attempt) within a run. A stride of 1
        // collides with the plan's own per-shot seed spacing — the shipped fixture numbers its
        // shots 22710, 22711, … 22715, so SH020's second attempt and SH030's first were both
        // seed 22712, and the 2026-09-14 evaluation dispatched 22712, 22714 and 22715 twice each
        // in one run. A shot's attempts are therefore spaced far enough apart that no plausible
        // plan puts two shots inside one shot's attempt range.
        if let Some(seed) = self.seed {
            let attempt_seed = seed.wrapping_add(
                i64::from(context.attempt.saturating_sub(1)).wrapping_mul(ATTEMPT_SEED_STRIDE),
            );
            body["seed"] = json!(attempt_seed);
            body["advanced"]["filmHarness"]["seed"] = json!(attempt_seed);
        }
        if let Some(first) = &assets.first_frame_asset_id {
            body["sourceAssetId"] = json!(first);
        }
        if let Some(last) = &assets.last_frame_asset_id {
            body["lastFrameAssetId"] = json!(last);
        }
        if !assets.reference_asset_ids.is_empty() {
            body["referenceAssetIds"] = json!(assets.reference_asset_ids);
        }
        body
    }
}

impl CompiledPlan {
    /// The compiled request for `shot_id`, if the plan compiled one.
    pub fn request(&self, shot_id: &str) -> Option<&CompiledRequest> {
        self.requests
            .iter()
            .find(|request| request.shot_id == shot_id)
    }

    /// Findings that make these requests unusable for `plan` and `pack`: a different plan, a
    /// different version, an edit to either document since the compile, or a shot the compile does
    /// not cover. This is what keeps a hand-edited plan from being dispatched with stale prompts.
    ///
    /// The PACK is judged here as well as the plan (sc-24029) because it is an input to the prompt:
    /// the binding sentences and the identity lock are written out of pack descriptions and
    /// locators, so editing a pack description changes what every request that repeats it would
    /// say. Editing one in the workspace is a first-class action, and before this the draft's
    /// compiled document stayed marked current across it — the edit then surfaced at preflight as
    /// a difference in a derived field, naming `insertedText` rather than the pack the operator
    /// had just changed.
    pub fn staleness_findings(
        &self,
        plan: &ProductionPlan,
        plan_sha256: &str,
        pack: &ReferencePack,
    ) -> Vec<PlanDiagnostic> {
        let mut findings = Vec::new();
        if self.schema_version != COMPILED_PLAN_SCHEMA_VERSION {
            findings.push(PlanDiagnostic::plan(
                "compiled.schemaVersion",
                format!(
                    "unsupported compiled plan schema version {} (this build reads \
                     {COMPILED_PLAN_SCHEMA_VERSION}); re-run `film-harness compile` to rewrite \
                     these requests",
                    self.schema_version
                ),
            ));
            return findings;
        }
        if self.plan_id != plan.id {
            findings.push(PlanDiagnostic::plan(
                "compiled.planId",
                format!(
                    "these requests were compiled from plan {:?}, not {:?}",
                    self.plan_id, plan.id
                ),
            ));
        }
        if self.plan_sha256 != plan_sha256 {
            findings.push(PlanDiagnostic::plan(
                "compiled.planSha256",
                format!(
                    "the plan has changed since it was compiled (plan v{} is {}, the requests were \
                     compiled from {}); re-run `film-harness compile`",
                    plan.version,
                    &plan_sha256[..plan_sha256.len().min(12)],
                    &self.plan_sha256[..self.plan_sha256.len().min(12)]
                ),
            ));
        }
        match reference_pack_sha256(pack) {
            Ok(pack_sha256) if pack_sha256 != self.reference_pack_sha256 => {
                findings.push(PlanDiagnostic::plan(
                    COMPILED_PACK_STALENESS_FIELD,
                    "the reference pack changed since these requests were compiled; recompile, or \
                     use authored prompts",
                ));
            }
            Ok(_) => {}
            Err(error) => findings.push(PlanDiagnostic::plan(
                "referencePack",
                format!("this reference pack cannot be serialized to identify it: {error}"),
            )),
        }
        for shot in &plan.shots {
            if self.request(&shot.id).is_none() {
                findings.push(PlanDiagnostic::shot(
                    &shot.id,
                    "compiled",
                    "no compiled request for this shot; re-run `film-harness compile`",
                ));
            }
        }
        findings
    }

    /// Findings that make these requests a different ASK than the plan describes.
    ///
    /// [`staleness_findings`](Self::staleness_findings) proves the requests were compiled from
    /// THIS plan; this proves they still say what compiling it now would say. It matters because
    /// the compiled document — not the plan — is what becomes the job body: `--compiled FILE`
    /// accepts one from any path, and every dispatched field but the prompt is taken from it. A
    /// hand-edited `durationSeconds` the engine would silently snap onto its frame lattice, a
    /// swapped `model`, a `mode` the checkpoint does not declare or a `negativePrompt` on a model
    /// that has none would otherwise reach the route unjudged, because the document validators only
    /// ever read the plan.
    ///
    /// Only `promptSource` and `authoredPrompt`, and the REFINED MIDDLE of `prompt`, may differ
    /// from a fresh compile: those are the compile's output (the model's own rewrite), and
    /// everything else is a transcription of the plan. The expected request is produced by the
    /// compiler itself rather than by a second list of rules, so the two cannot drift.
    ///
    /// `insertedText` is NOT in that exemption (sc-24023): the compiler's own sentences are derived
    /// from the plan and the pack, a fresh compile reproduces them exactly, and a hand-edited
    /// `<Picture N>` would bind the model to the wrong image with nothing downstream able to tell.
    ///
    /// Neither is the prompt those sentences were composed INTO (sc-24029). `insertedText` is a
    /// record of what was written, not of where it ended up, and `prompt` is the field that becomes
    /// the job body: a document with pristine `insertedText` and a `prompt` with its bindings
    /// deleted or its picture labels swapped used to pass. [`prompt_differences`] asks the question
    /// that closes it — see there for how a refined prompt is checked without a second rewrite.
    pub fn conformance_findings(
        &self,
        plan: &ProductionPlan,
        pack: &ReferencePack,
        entries: &ModelEntries<'_>,
        lane: ModelLane,
    ) -> Vec<PlanDiagnostic> {
        let entry = entries.base_entry();
        let mut findings = Vec::new();
        if self.model.id != plan.model.id {
            findings.push(PlanDiagnostic::plan(
                "compiled.model.id",
                format!(
                    "the compiled requests target model {:?}, but the plan renders through {:?}",
                    self.model.id, plan.model.id
                ),
            ));
        }
        if self.model.tier != plan.model.tier {
            findings.push(PlanDiagnostic::plan(
                "compiled.model.tier",
                format!(
                    "the compiled requests declare tier {:?}, but the plan asks for {:?}",
                    self.model.tier, plan.model.tier
                ),
            ));
        }
        if self.model.lane != lane.manifest_key() {
            findings.push(PlanDiagnostic::plan(
                "compiled.model.lane",
                format!(
                    "these requests were compiled for the {:?} lane, but this run's host renders \
                     on {:?}; re-run `film-harness compile` against this host",
                    self.model.lane,
                    lane.manifest_key()
                ),
            ));
        }
        let Some(fps) = crate::film_plan::plan_fps(plan, entry) else {
            findings.push(PlanDiagnostic::plan(
                "model.fps",
                format!(
                    "{} declares no default fps; set model.fps in the plan before dispatching",
                    plan.model.id
                ),
            ));
            return findings;
        };
        if self.model.fps != fps {
            findings.push(PlanDiagnostic::plan(
                "compiled.model.fps",
                format!(
                    "the compiled requests declare {} fps, but the plan renders at {fps}",
                    self.model.fps
                ),
            ));
        }
        let empty = BTreeMap::new();
        let inputs = CompileInputs {
            entries,
            lane: lane.manifest_key(),
            plan_sha256: &self.plan_sha256,
            compiled_at: &self.compiled_at,
            refined_prompts: &empty,
        };
        for shot in &plan.shots {
            // A shot with no request at all is `staleness_findings`' finding, not a second one.
            let Some(request) = self.request(&shot.id) else {
                continue;
            };
            match compile_shot(plan, shot, pack, &inputs, fps) {
                Ok(expected) => {
                    findings.extend(request_differences(request, &expected));
                    findings.extend(prompt_differences(request, &expected));
                }
                Err(mut shot_findings) => findings.append(&mut shot_findings),
            }
        }
        findings
    }
}

/// The DISPATCHED prompt, judged against what the compiler would compose (sc-24029).
///
/// [`request_differences`] exempts `prompt` because the refine rewrite genuinely is the compile's
/// own output and no second compile reproduces it. That exemption was total, so it also exempted
/// everything the compiler wrote AROUND the rewrite: a `compiled.json` whose `insertedText` was
/// pristine but whose `prompt` had a binding sentence deleted, two `<Picture N>` labels swapped or
/// the trailing `Audio:` sentence removed passed conformance and was dispatched, while the doc
/// comment on [`CompiledPlan::conformance_findings`] promised the opposite.
///
/// What is checkable differs by source, so the two are asked different questions:
///
/// * AUTHORED — the whole prompt is derived from the plan and the pack, so a fresh compile
///   reproduces it exactly and equality is the whole check.
/// * REFINED — only the middle is the model's. The compiler's own contribution is recovered by
///   stripping the expected leading prefix and trailing suffix from the dispatched text and then
///   RECOMPOSING: a middle is accepted only if putting the expected insertions back around it
///   reproduces the dispatched prompt character for character. That is what makes the recovery
///   exact rather than approximate — in particular it settles the one thing the composition loses,
///   the sentence-boundary `.` [`apply_inserted_text`] supplies before the first trailing piece
///   when the middle does not end one, without having to guess whose period it is.
///
/// The recovered middle is then held to the rule the refine branch of [`compile_shot`] applies at
/// compile time: it may not contain an engine label. A document read back from disk never went
/// through that branch, so this is where a label hand-written into the middle of a refined prompt
/// is caught.
fn prompt_differences(actual: &CompiledRequest, expected: &CompiledRequest) -> Vec<PlanDiagnostic> {
    let tampered = |detail: &str| {
        vec![PlanDiagnostic::shot(
            &expected.shot_id,
            "compiled.prompt",
            format!(
                "the prompt {} would dispatch is not the prompt the compiler composed ({detail}); \
                 re-run `film-harness compile`",
                expected.shot_id
            ),
        )]
    };
    match actual.prompt_source {
        PromptSource::Authored => {
            if actual.prompt == expected.prompt {
                Vec::new()
            } else {
                tampered(
                    "this request states its prompt is the plan's own, so compiling the plan \
                          now must reproduce it exactly, and it does not",
                )
            }
        }
        PromptSource::Refined => match recovered_middle(&actual.prompt, &expected.inserted_text) {
            None => tampered(
                "the compiler's own leading and trailing sentences are not around the refined \
                 text where it wrote them",
            ),
            Some(middle) => {
                if let Some(label) = crate::film_plan::engine_label_at(&middle)
                    .is_some()
                    .then(|| crate::film_plan::quoted_engine_label(&middle))
                {
                    tampered(&format!(
                        "the refined text inside it contains {label}, a label this film's renderer \
                         assigns itself and the compiler writes"
                    ))
                } else {
                    Vec::new()
                }
            }
        },
    }
}

/// The text [`apply_inserted_text`] was given, recovered from what it produced, or `None` when
/// `prompt` is not something it could have produced from `inserted` (sc-24029).
///
/// Every part of the composition is fixed by `inserted` except ONE character: the `.` supplied
/// before the first trailing piece when the middle does not already end a sentence. That `.` is
/// deliberately left ON the recovered middle rather than guessed at — whether it was the author's
/// or the compiler's is exactly what the composition loses, and both readings compose back to the
/// same prompt — so a refined text WITH terminal punctuation and one WITHOUT recover alike.
///
/// The split is then accepted only if recomposing reproduces `prompt` character for character. It
/// is that check, not the stripping, that makes this exact: a prompt whose bindings were deleted,
/// whose labels were swapped or whose trailing sentence was removed fails it.
fn recovered_middle(prompt: &str, inserted: &[InsertedText]) -> Option<String> {
    let leading: String = inserted
        .iter()
        .filter(|piece| piece.kind.placement() == InsertedTextPlacement::Leading)
        .map(|piece| format!("{} ", piece.text.trim()))
        .collect();
    let body = prompt.strip_prefix(&leading)?;
    // The trailing composition, minus the one separator period that depends on the middle. Every
    // later separator is decided by the piece before it, which is the compiler's own text.
    let mut fixed = String::new();
    for (index, piece) in inserted
        .iter()
        .filter(|piece| piece.kind.placement() == InsertedTextPlacement::Trailing)
        .enumerate()
    {
        if index > 0 && !ends_sentence(&fixed) {
            fixed.push('.');
        }
        fixed.push(' ');
        fixed.push_str(piece.text.trim());
    }
    let middle = body.strip_suffix(&fixed)?;
    (apply_inserted_text(middle, inserted) == prompt).then(|| middle.to_owned())
}

/// Every field of `actual` that a fresh compile would have written differently, except the three
/// the compile itself produces (`prompt`, `promptSource`, `authoredPrompt`).
///
/// `expected` is destructured exhaustively on purpose: a field added to [`CompiledRequest`] fails
/// to compile here until it is either compared or deliberately exempted, so the guarantee this
/// function states cannot quietly narrow as the request grows.
fn request_differences(
    actual: &CompiledRequest,
    expected: &CompiledRequest,
) -> Vec<PlanDiagnostic> {
    let CompiledRequest {
        shot_id,
        beat,
        mode,
        model,
        reference_image_short_edge,
        loras,
        steps,
        effective_steps,
        turbo_scheduler_shift,
        partition_reason,
        prompt: _,
        prompt_source: _,
        authored_prompt: _,
        inserted_text,
        negative_prompt,
        duration_seconds,
        fps,
        width,
        height,
        seed,
        first_frame_role,
        last_frame_role,
        reference_roles,
        chain_from_shot_id,
        continuity_roles,
    } = expected;
    let mut findings = Vec::new();
    let mut differ = |field: &str, found: String, planned: String| {
        if found != planned {
            findings.push(PlanDiagnostic::shot(
                shot_id,
                field,
                format!(
                    "the compiled request asks for {found}, but the plan says {planned}; re-run \
                     `film-harness compile` (only the prompt may differ from the plan)"
                ),
            ));
        }
    };
    differ("compiled.model", quoted(&actual.model), quoted(model));
    differ(
        "compiled.partitionReason",
        quoted(&actual.partition_reason),
        quoted(partition_reason),
    );
    differ(
        "compiled.referenceImageShortEdge",
        format!("{:?}", actual.reference_image_short_edge),
        format!("{reference_image_short_edge:?}"),
    );
    differ(
        "compiled.loras",
        format!("{:?}", actual.loras),
        format!("{loras:?}"),
    );
    differ(
        "compiled.steps",
        format!("{:?}", actual.steps),
        format!("{steps:?}"),
    );
    differ(
        "compiled.effectiveSteps",
        format!("{:?}", actual.effective_steps),
        format!("{effective_steps:?}"),
    );
    differ(
        "compiled.turboSchedulerShift",
        format!("{:?}", actual.turbo_scheduler_shift),
        format!("{turbo_scheduler_shift:?}"),
    );
    differ(
        "compiled.insertedText",
        rendered_inserted_text(&actual.inserted_text),
        rendered_inserted_text(inserted_text),
    );
    differ("compiled.beat", quoted(&actual.beat), quoted(beat));
    differ("compiled.mode", quoted(&actual.mode), quoted(mode));
    differ(
        "compiled.durationSeconds",
        format!("{}s", actual.duration_seconds),
        format!("{duration_seconds}s"),
    );
    differ(
        "compiled.fps",
        format!("{} fps", actual.fps),
        format!("{fps} fps"),
    );
    differ(
        "compiled.resolution",
        format!("{}x{}", actual.width, actual.height),
        format!("{width}x{height}"),
    );
    differ(
        "compiled.seed",
        format!("{:?}", actual.seed),
        format!("{seed:?}"),
    );
    differ(
        "compiled.negativePrompt",
        format!("{:?}", actual.negative_prompt),
        format!("{negative_prompt:?}"),
    );
    differ(
        "compiled.firstFrameRole",
        format!("{:?}", actual.first_frame_role),
        format!("{first_frame_role:?}"),
    );
    differ(
        "compiled.lastFrameRole",
        format!("{:?}", actual.last_frame_role),
        format!("{last_frame_role:?}"),
    );
    differ(
        "compiled.referenceRoles",
        format!("{:?}", actual.reference_roles),
        format!("{reference_roles:?}"),
    );
    differ(
        "compiled.chainFromShotId",
        format!("{:?}", actual.chain_from_shot_id),
        format!("{chain_from_shot_id:?}"),
    );
    differ(
        "compiled.continuityRoles",
        format!("{:?}", actual.continuity_roles),
        format!("{continuity_roles:?}"),
    );
    findings
}

fn quoted(value: &str) -> String {
    format!("{value:?}")
}

/// The compiler's inserted text as a finding may state it: each piece named by its kind, with the
/// text it holds (sc-24029).
///
/// A plain reading rather than `{:?}` on the `Vec<InsertedText>`. The difference reaches an
/// operator — the film workspace renders it when a preflight refuses — and a Rust struct dump
/// (`[InsertedText { kind: ReferenceBinding, text: "…" }]`) is the compiler's private notation, not
/// a description of what changed in a prompt.
fn rendered_inserted_text(inserted: &[InsertedText]) -> String {
    if inserted.is_empty() {
        return "no inserted text".to_owned();
    }
    inserted
        .iter()
        .map(|piece| format!("{}: {}", piece.kind.label(), piece.text))
        .collect::<Vec<_>>()
        .join(" | ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::film_plan::{parse_plan, parse_reference_pack, PLAN_SCHEMA_VERSION};

    fn plan_text() -> String {
        serde_json::to_string(&json!({
            "schemaVersion": PLAN_SCHEMA_VERSION,
            "id": "courier-workshop",
            "version": 2,
            "title": "Courier",
            "model": { "id": "minimax_h3", "tier": "q4", "fps": 24, "resolution": "576x320" },
            "limits": { "maxRunSeconds": 3600, "maxShotSeconds": 1800, "maxAttemptsPerShot": 1, "maxMemoryGb": 96 },
            "shots": [
                {
                    "id": "SH010", "beat": "enter", "framing": "wide", "prompt": "a courier enters",
                    "targetDurationSeconds": 5.1667, "startState": "empty", "endState": "courier inside", "audio": "Room tone, no music.",
                    "seed": 7, "conditioning": { "mode": "text_to_video" },
                    "continuityRoles": ["courier"]
                },
                {
                    "id": "SH020", "beat": "place", "framing": "medium", "prompt": "places the parcel",
                    "targetDurationSeconds": 5.875, "startState": "courier inside", "endState": "parcel on table", "audio": "Room tone, no music.",
                    "conditioning": {
                        "mode": "image_to_video",
                        "firstFrameRole": "workshop_plate",
                        "chainFromShotId": "SH010"
                    },
                    "continuityRoles": ["courier", "red_parcel"]
                }
            ]
        }))
        .unwrap()
    }

    fn pack() -> ReferencePack {
        parse_reference_pack(
            &json!({
                "schemaVersion": crate::film_plan::REFERENCE_PACK_SCHEMA_VERSION,
                "id": "courier-refs",
                "version": 3,
                "references": [
                    { "role": "courier", "kind": "character", "file": "references/courier.png" },
                    { "role": "red_parcel", "kind": "prop", "file": "references/red_parcel.png" },
                    { "role": "workshop_plate", "kind": "plate", "file": "references/plate.png" }
                ]
            })
            .to_string(),
        )
        .unwrap()
    }

    fn entry() -> JsonObject<String, Value> {
        json!({
            "id": "minimax_h3",
            // `steps` mirrors the shipped catalog: it is what the engine renders at when nothing
            // names a count, and therefore what `effective_steps` records in the base regime.
            "defaults": { "fps": 24, "resolution": "1344x768", "steps": 50 },
            "limits": { "resolutions": ["1344x768", "576x320"] }
        })
        .as_object()
        .cloned()
        .unwrap()
    }

    fn reference_entry() -> JsonObject<String, Value> {
        json!({
            "id": "minimax_h3_ref",
            "defaults": { "fps": 24, "resolution": "1344x768", "steps": 50 },
            "limits": { "resolutions": ["1344x768", "576x320"], "maxReferenceAssets": 9 }
        })
        .as_object()
        .cloned()
        .unwrap()
    }

    /// The fixture pack, borrowed for a [`DispatchContext`]'s lifetime.
    fn pack_ref() -> &'static ReferencePack {
        static PACK: std::sync::OnceLock<ReferencePack> = std::sync::OnceLock::new();
        PACK.get_or_init(pack)
    }

    fn role_assets() -> BTreeMap<String, String> {
        [
            ("courier", "asset_courier"),
            ("red_parcel", "asset_parcel"),
            ("workshop_plate", "asset_plate"),
        ]
        .into_iter()
        .map(|(role, asset)| (role.to_owned(), asset.to_owned()))
        .collect()
    }

    fn compiled(refined: BTreeMap<String, String>) -> CompiledPlan {
        let plan = parse_plan(&plan_text()).unwrap();
        let entry = entry();
        compile_plan(
            &plan,
            &pack(),
            &CompileInputs {
                entries: &ModelEntries::single("minimax_h3", &entry),
                lane: "mlx",
                plan_sha256: "abc123def456",
                compiled_at: "2026-09-13T00:00:00Z",
                refined_prompts: &refined,
            },
        )
        .expect("compiles")
    }

    #[test]
    fn a_plan_compiles_to_one_request_per_shot_inside_the_declared_geometry() {
        let compiled = compiled(BTreeMap::new());
        assert_eq!(compiled.requests.len(), 2);
        assert_eq!(compiled.plan_version, 2);
        assert_eq!(compiled.reference_pack_version, 3);
        assert_eq!(compiled.model.fps, 24);
        let first = compiled.request("SH010").unwrap();
        assert_eq!(
            first.prompt,
            "a courier enters. Audio: Room tone, no music."
        );
        assert_eq!(first.prompt_source, PromptSource::Authored);
        assert_eq!(first.authored_prompt, None);
        assert_eq!((first.width, first.height), (576, 320));
        let second = compiled.request("SH020").unwrap();
        assert_eq!(second.first_frame_role.as_deref(), Some("workshop_plate"));
        assert_eq!(second.chain_from_shot_id.as_deref(), Some("SH010"));
        // Round-trips through JSON with no unknown fields.
        let text = serde_json::to_string(&compiled).unwrap();
        let back: CompiledPlan = serde_json::from_str(&text).unwrap();
        assert_eq!(back, compiled);
    }

    /// The attempt offset is a STRIDE, not `+1` (sc-22715). The shipped courier fixture seeds its
    /// shots one apart (22710..22715), so a stride of 1 made SH020's second attempt and SH030's
    /// first the same seed — the 2026-09-14 evaluation run dispatched 22712, 22714 and 22715 twice
    /// each. The seed a run dispatches has to identify the (shot, attempt) pair it came from.
    #[test]
    fn a_later_attempt_renders_at_the_plans_seed_offset_by_its_attempt_number() {
        let compiled = compiled(BTreeMap::new());
        let assets = role_assets();
        // Every attempt of a shot whose plan seed is `base`, dispatched through the real body
        // builder. Taking it from `to_job_body` rather than computing it here is the point: the
        // derivation is what is under test, not the constant.
        let seeds_from = |base: i64| -> Vec<i64> {
            let mut request = compiled.request("SH010").unwrap().clone();
            request.seed = Some(base);
            (1..=8u32)
                .map(|attempt| {
                    let context = DispatchContext {
                        project_id: "proj_1",
                        run_id: "run_abc",
                        plan_id: "courier-workshop",
                        plan_version: 2,
                        attempt,
                        tier: Some("q4"),
                        idempotency_key: None,
                        pack: pack_ref(),
                        role_assets: &assets,
                    };
                    request.to_job_body(&context).unwrap()["seed"]
                        .as_i64()
                        .expect("a seeded request dispatches a seed")
                })
                .collect()
        };
        let body_for = |attempt: u32| {
            let context = DispatchContext {
                project_id: "proj_1",
                run_id: "run_abc",
                plan_id: "courier-workshop",
                plan_version: 2,
                attempt,
                tier: Some("q4"),
                idempotency_key: None,
                pack: pack_ref(),
                role_assets: &assets,
            };
            compiled
                .request("SH010")
                .unwrap()
                .to_job_body(&context)
                .unwrap()
        };
        // Attempt 1 is the plan's own seed; the replacement (attempt 2) must not be the same
        // render, because the MLX pipeline is deterministic for a seed.
        assert_eq!(body_for(1)["seed"], 7);
        assert_eq!(body_for(1)["advanced"]["filmHarness"]["seed"], 7);
        assert_eq!(body_for(2)["seed"], 1007);
        assert_eq!(body_for(2)["advanced"]["filmHarness"]["seed"], 1007);
        assert_eq!(body_for(5)["seed"], 4007);
        // A plan that numbers its shots one apart — which every fixture and the evaluation plan do
        // (22710…22715) — must not have one shot's later attempts land on the next shot's renders.
        // At an offset of `n − 1` they did: SH020-a2, SH030-a1 and three more pairs were the same
        // seed in one run, so a dispatched seed no longer said which render it belonged to.
        let (first, next) = (seeds_from(22710), seeds_from(22711));
        for seed in &first {
            assert!(
                !next.contains(seed),
                "seed {seed} is dispatched for two different shots of the same run: {first:?} vs \
                 {next:?}"
            );
        }
        assert_eq!(
            first
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            first.len(),
            "no shot repeats a seed across its own attempts: {first:?}"
        );
        // Nothing else about the request varies by attempt.
        let (one, two) = (body_for(1), body_for(2));
        for key in ["prompt", "mode", "duration", "fps", "width", "height"] {
            assert_eq!(one[key], two[key], "{key} must not vary by attempt");
        }
        // A request with no seed leaves the field out at every attempt.
        let mut unseeded = compiled.clone();
        unseeded.requests[0].seed = None;
        let context = DispatchContext {
            project_id: "proj_1",
            run_id: "run_abc",
            plan_id: "courier-workshop",
            plan_version: 2,
            attempt: 3,
            tier: Some("q4"),
            idempotency_key: None,
            pack: pack_ref(),
            role_assets: &assets,
        };
        let body = unseeded
            .request("SH010")
            .unwrap()
            .to_job_body(&context)
            .unwrap();
        assert!(body.get("seed").is_none());
        assert!(body["advanced"]["filmHarness"].get("seed").is_none());
    }

    #[test]
    fn the_job_body_is_the_compiled_request_with_roles_resolved() {
        let compiled = compiled(BTreeMap::new());
        let assets = role_assets();
        let context = DispatchContext {
            project_id: "proj_1",
            run_id: "run_abc",
            plan_id: "courier-workshop",
            plan_version: 2,
            attempt: 1,
            tier: Some("q4"),
            idempotency_key: Some("run_abc:SH010:a1"),
            pack: pack_ref(),
            role_assets: &assets,
        };
        let body = compiled
            .request("SH010")
            .unwrap()
            .to_job_body(&context)
            .unwrap();
        assert_eq!(body["projectId"], "proj_1");
        assert_eq!(body["mode"], "text_to_video");
        assert_eq!(
            body["prompt"],
            "a courier enters. Audio: Room tone, no music."
        );
        assert_eq!(body["duration"], 5.1667);
        assert_eq!(body["fps"], 24);
        assert_eq!(body["width"], 576);
        assert_eq!(body["height"], 320);
        assert_eq!(body["fitMode"], "crop");
        assert_eq!(body["seed"], 7);
        assert_eq!(body["advanced"]["mlxQuantize"], 4);
        assert_eq!(body["advanced"]["filmHarness"]["shotId"], "SH010");
        assert_eq!(body["advanced"]["filmHarness"]["planVersion"], 2);
        // The replay key rides the dispatched payload: it is how a controller that died between the
        // POST and the record write finds its own job instead of enqueuing a second render.
        assert_eq!(
            body["advanced"]["filmHarness"]["idempotencyKey"],
            "run_abc:SH010:a1"
        );
        assert!(body.get("sourceAssetId").is_none());
        assert!(body["advanced"]["filmHarness"]
            .get("promptSource")
            .is_none());

        // The declared continuity roles ride the provenance too, so a take can say which approved
        // references it was meant to depict even when the model takes no reference conditioning.
        assert_eq!(
            body["advanced"]["filmHarness"]["continuityRoles"],
            json!(["courier"])
        );

        let body = compiled
            .request("SH020")
            .unwrap()
            .to_job_body(&context)
            .unwrap();
        assert_eq!(body["sourceAssetId"], "asset_plate");
        assert_eq!(body["advanced"]["filmHarness"]["chainFromShotId"], "SH010");
        assert_eq!(
            body["advanced"]["filmHarness"]["continuityRoles"],
            json!(["courier", "red_parcel"])
        );
        assert!(body.get("referenceAssetIds").is_none());

        // A shot that declares no continuity roles claims none in its provenance.
        let mut bare = compiled.request("SH010").unwrap().clone();
        bare.continuity_roles.clear();
        let body = bare.to_job_body(&context).unwrap();
        assert!(body["advanced"]["filmHarness"]
            .get("continuityRoles")
            .is_none());

        // A role that was never imported refuses the dispatch instead of sending a body with a
        // missing keyframe.
        let empty = BTreeMap::new();
        let context = DispatchContext {
            pack: pack_ref(),
            role_assets: &empty,
            ..context
        };
        let findings = compiled
            .request("SH020")
            .unwrap()
            .to_job_body(&context)
            .expect_err("missing asset refuses");
        assert_eq!(findings.len(), 1);
        assert!(
            findings[0].message.contains("never imported"),
            "{findings:?}"
        );
    }

    #[test]
    fn a_refined_prompt_replaces_the_text_and_keeps_the_authored_one() {
        let refined = [(
            "SH010".to_owned(),
            "  integrated_multimodal_description: a courier enters a warm workshop  ".to_owned(),
        )]
        .into_iter()
        .collect();
        let compiled = compiled(refined);
        let first = compiled.request("SH010").unwrap();
        assert_eq!(first.prompt_source, PromptSource::Refined);
        assert_eq!(
            first.prompt,
            "integrated_multimodal_description: a courier enters a warm workshop. Audio: Room tone, no music."
        );
        assert_eq!(first.authored_prompt.as_deref(), Some("a courier enters"));
        // The untouched shot still compiles its authored prompt.
        assert_eq!(
            compiled.request("SH020").unwrap().prompt_source,
            PromptSource::Authored
        );
        let assets = role_assets();
        let body = first
            .to_job_body(&DispatchContext {
                project_id: "p",
                run_id: "r",
                plan_id: "courier-workshop",
                plan_version: 2,
                attempt: 1,
                tier: None,
                idempotency_key: None,
                pack: pack_ref(),
                role_assets: &assets,
            })
            .unwrap();
        assert_eq!(body["advanced"]["filmHarness"]["promptSource"], "refined");
        assert!(body["advanced"].get("mlxQuantize").is_none());
    }

    #[test]
    fn an_unusable_refined_prompt_is_a_finding_not_a_silent_fallback() {
        let plan = parse_plan(&plan_text()).unwrap();
        for bad in ["   ", &"x".repeat(MAX_PROMPT_CHARS + 1)] {
            let refined: BTreeMap<String, String> =
                [("SH010".to_owned(), bad.to_owned())].into_iter().collect();
            let findings = compile_plan(
                &plan,
                &pack(),
                &CompileInputs {
                    entries: &ModelEntries::single("minimax_h3", &entry()),
                    lane: "mlx",
                    plan_sha256: "abc",
                    compiled_at: "now",
                    refined_prompts: &refined,
                },
            )
            .expect_err("refuses");
            assert_eq!(findings.len(), 1, "{findings:?}");
            assert_eq!(findings[0].shot_id.as_deref(), Some("SH010"));
            assert!(findings[0].message.contains("the video route accepts"));
        }
    }

    #[test]
    fn stale_compiled_requests_are_refused_against_the_plan_they_claim() {
        let plan = parse_plan(&plan_text()).unwrap();
        let compiled = compiled(BTreeMap::new());
        assert!(compiled
            .staleness_findings(&plan, "abc123def456", &pack())
            .is_empty());
        let findings = compiled.staleness_findings(&plan, "999999999999", &pack());
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(
            findings[0]
                .message
                .contains("has changed since it was compiled"),
            "{findings:?}"
        );

        let mut compiled = compiled;
        compiled
            .requests
            .retain(|request| request.shot_id != "SH020");
        let findings = compiled.staleness_findings(&plan, "abc123def456", &pack());
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert_eq!(findings[0].shot_id.as_deref(), Some("SH020"));

        compiled.plan_id = "other".to_owned();
        assert!(compiled
            .staleness_findings(&plan, "abc123def456", &pack())
            .iter()
            .any(|finding| finding.field == "compiled.planId"));
    }

    /// sc-23402 review. A `compiled.json` written by a PRE-STORY build is refused by SCHEMA
    /// VERSION, not blamed on the operator as a hand edit.
    ///
    /// Such a document has no `partitionReason` key at all. `#[serde(default)]` reads it back as
    /// `""`, which `request_differences` would report as `compiled.partitionReason` — "the
    /// compiled request asks for …, but the plan says …", a tampering message — so a phase-1 run
    /// directory could no longer be resumed and the refusal named the wrong cause. The version
    /// bump to 2 is what makes the first finding the true one, and it names the remedy.
    #[test]
    fn a_pre_story_compiled_document_is_refused_by_schema_version_not_as_tampered() {
        let plan = parse_plan(&plan_text()).unwrap();
        let entry = entry();
        let entries = ModelEntries::single("minimax_h3", &entry);

        // Exactly what a v1 document on disk deserializes to: version 1, and the key absent.
        let mut v1 = compiled(BTreeMap::new());
        v1.schema_version = 1;
        for request in &mut v1.requests {
            request.partition_reason = String::new();
        }
        let round_tripped: CompiledPlan =
            serde_json::from_value(serde_json::to_value(&v1).unwrap()).unwrap();
        assert_eq!(round_tripped.schema_version, 1);
        assert!(round_tripped.requests[0].partition_reason.is_empty());

        let findings = round_tripped.staleness_findings(&plan, "abc123def456", &pack());
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert_eq!(findings[0].field, "compiled.schemaVersion", "{findings:?}");
        assert!(
            findings[0].message.contains("schema version 1")
                && findings[0].message.contains("film-harness compile"),
            "the refusal must name the version AND the remedy: {findings:?}"
        );
        assert!(
            !findings
                .iter()
                .any(|finding| finding.field == "compiled.partitionReason"),
            "a schema migration must never be reported as a hand edit: {findings:?}"
        );

        // And this is the finding the bump replaced: at the CURRENT version the same empty
        // `partitionReason` is (correctly) a tampering report, which is why v1 had to be refused
        // by version rather than left to fall through to conformance.
        let mut current = round_tripped;
        current.schema_version = COMPILED_PLAN_SCHEMA_VERSION;
        assert!(current
            .staleness_findings(&plan, "abc123def456", &pack())
            .is_empty());
        assert!(
            current
                .conformance_findings(&plan, &pack(), &entries, ModelLane::Mlx)
                .iter()
                .any(|finding| finding.field == "compiled.partitionReason"),
            "without the bump a v1 document lands here instead"
        );
    }

    #[test]
    fn a_hand_edited_request_is_refused_even_when_it_pins_the_right_plan() {
        let plan = parse_plan(&plan_text()).unwrap();
        let entry = entry();
        let entries = ModelEntries::single("minimax_h3", &entry);
        let clean = compiled(BTreeMap::new());
        assert!(clean
            .conformance_findings(&plan, &pack(), &entries, ModelLane::Mlx)
            .is_empty());
        // The compile's own output is exempt: a refined prompt is why the document exists.
        let refined = compiled(
            [("SH010".to_owned(), "a rewritten courier".to_owned())]
                .into_iter()
                .collect(),
        );
        assert!(refined
            .conformance_findings(&plan, &pack(), &entries, ModelLane::Mlx)
            .is_empty());

        // Every other field is a transcription of the plan, and an edit to one is named.
        // (field the finding must name, the hand edit, a fragment of the message it must carry)
        type TamperCase = (&'static str, fn(&mut CompiledRequest), &'static str);
        let cases: Vec<TamperCase> = vec![
            (
                "compiled.durationSeconds",
                |request| request.duration_seconds = 9.0,
                "9s",
            ),
            (
                "compiled.mode",
                |request| request.mode = "image_to_video".to_owned(),
                "image_to_video",
            ),
            (
                "compiled.model",
                |request| request.model = "ltx_2_5".to_owned(),
                "ltx_2_5",
            ),
            (
                "compiled.partitionReason",
                |request| request.partition_reason = "because I said so".to_owned(),
                "because I said so",
            ),
            ("compiled.fps", |request| request.fps = 30, "30 fps"),
            (
                "compiled.resolution",
                |request| request.width = 1344,
                "1344x320",
            ),
            ("compiled.seed", |request| request.seed = Some(8), "Some(8)"),
            (
                "compiled.negativePrompt",
                |request| request.negative_prompt = Some("blurry".to_owned()),
                "blurry",
            ),
            (
                "compiled.firstFrameRole",
                |request| request.first_frame_role = Some("workshop_plate".to_owned()),
                "workshop_plate",
            ),
            (
                "compiled.lastFrameRole",
                |request| request.last_frame_role = Some("workshop_plate".to_owned()),
                "workshop_plate",
            ),
            (
                "compiled.referenceRoles",
                |request| request.reference_roles = vec!["courier".to_owned()],
                "courier",
            ),
            (
                "compiled.chainFromShotId",
                |request| request.chain_from_shot_id = Some("SH020".to_owned()),
                "SH020",
            ),
            (
                "compiled.continuityRoles",
                |request| request.continuity_roles.clear(),
                "[]",
            ),
            (
                "compiled.beat",
                |request| request.beat = "exit".to_owned(),
                "exit",
            ),
        ];
        for (field, edit, expected) in cases {
            let mut tampered = clean.clone();
            edit(&mut tampered.requests[0]);
            let findings = tampered.conformance_findings(&plan, &pack(), &entries, ModelLane::Mlx);
            assert_eq!(findings.len(), 1, "{field}: {findings:?}");
            assert_eq!(findings[0].shot_id.as_deref(), Some("SH010"), "{field}");
            assert_eq!(findings[0].field, field, "{findings:?}");
            assert!(
                findings[0].message.contains(expected),
                "{field}: {findings:?}"
            );
        }

        // The document's own model block is checked too: a swapped model, tier, fps or lane.
        let mut tampered = clean.clone();
        tampered.model.id = "ltx_2_5".to_owned();
        tampered.model.tier = Some("q8".to_owned());
        tampered.model.fps = 30;
        let fields: Vec<String> = tampered
            .conformance_findings(&plan, &pack(), &entries, ModelLane::Candle)
            .into_iter()
            .map(|finding| finding.field)
            .collect();
        assert!(
            fields.contains(&"compiled.model.id".to_owned())
                && fields.contains(&"compiled.model.tier".to_owned())
                && fields.contains(&"compiled.model.fps".to_owned())
                && fields.contains(&"compiled.model.lane".to_owned()),
            "{fields:?}"
        );

        // A shot the compile does not cover is `staleness_findings`' report, not a second one here.
        let mut short = clean;
        short.requests.retain(|request| request.shot_id != "SH020");
        assert!(short
            .conformance_findings(&plan, &pack(), &entries, ModelLane::Mlx)
            .is_empty());
    }

    /// A mixed plan: SH010 binds two reference roles, SH020 binds none (sc-23402, AC1).
    fn mixed_plan_text() -> String {
        serde_json::to_string(&json!({
            "schemaVersion": PLAN_SCHEMA_VERSION,
            "id": "courier-workshop",
            "version": 2,
            "title": "Courier",
            "model": { "id": "minimax_h3", "tier": "q4", "fps": 24, "resolution": "576x320" },
            "limits": { "maxRunSeconds": 3600, "maxShotSeconds": 1800, "maxAttemptsPerShot": 1, "maxMemoryGb": 96 },
            "shots": [
                {
                    "id": "SH010", "beat": "enter", "framing": "wide", "prompt": "a courier enters",
                    "targetDurationSeconds": 5.1667, "startState": "empty", "endState": "courier inside", "audio": "Room tone, no music.",
                    "conditioning": {
                        "mode": "reference_to_video",
                        // Both BINDABLE kinds (character, prop): a `plate` is refused in this
                        // slot by `validate_plan_against_pack`.
                        "referenceRoles": ["courier", "red_parcel"]
                    },
                    "continuityRoles": ["courier"]
                },
                {
                    "id": "SH020", "beat": "place", "framing": "medium", "prompt": "places the parcel",
                    "targetDurationSeconds": 5.875, "startState": "courier inside", "endState": "parcel on table", "audio": "Room tone, no music.",
                    "conditioning": { "mode": "text_to_video" },
                    "continuityRoles": ["courier", "red_parcel"]
                }
            ]
        }))
        .unwrap()
    }

    /// The mixed plan with `model.advanced.referenceImageShortEdge` set, compiled against both
    /// partitions.
    fn mixed_compiled_with_short_edge(edge: Option<u32>) -> CompiledPlan {
        let mut document: Value = serde_json::from_str(&mixed_plan_text()).unwrap();
        if let Some(edge) = edge {
            document["model"]["advanced"] = json!({ "referenceImageShortEdge": edge });
        }
        let plan = parse_plan(&document.to_string()).unwrap();
        let base = entry();
        let reference = reference_entry();
        compile_plan(
            &plan,
            &pack(),
            &CompileInputs {
                entries: &ModelEntries::with_reference_partition(
                    "minimax_h3",
                    &base,
                    Some(("minimax_h3_ref", &reference)),
                ),
                lane: "mlx",
                plan_sha256: "abc",
                compiled_at: "now",
                refined_prompts: &BTreeMap::new(),
            },
        )
        .expect("the mixed plan compiles")
    }

    fn short_edge_context<'a>(assets: &'a BTreeMap<String, String>) -> DispatchContext<'a> {
        DispatchContext {
            project_id: "proj_1",
            run_id: "run_abc",
            plan_id: "courier-workshop",
            plan_version: 2,
            attempt: 1,
            tier: Some("q4"),
            idempotency_key: Some("run_abc:SH010:a1"),
            pack: pack_ref(),
            role_assets: assets,
        }
    }

    /// The mixed plan with a turbo selection and, optionally, a `model.advanced.steps` override.
    fn mixed_compiled_with_turbo(loras: Value, steps: Option<i64>) -> CompiledPlan {
        let mut document: Value = serde_json::from_str(&mixed_plan_text()).unwrap();
        document["model"]["loras"] = loras;
        if let Some(steps) = steps {
            document["model"]["advanced"] = json!({ "steps": steps });
        }
        let plan = parse_plan(&document.to_string()).unwrap();
        let base = entry();
        let reference = reference_entry();
        compile_plan(
            &plan,
            &pack(),
            &CompileInputs {
                entries: &ModelEntries::with_reference_partition(
                    "minimax_h3",
                    &base,
                    Some(("minimax_h3_ref", &reference)),
                ),
                lane: "mlx",
                plan_sha256: "abc",
                compiled_at: "now",
                refined_prompts: &BTreeMap::new(),
            },
        )
        .expect("the turbo plan compiles")
    }

    /// 🔴 sc-23406. ONE plan-level LoRA list, resolved PER PARTITION — into the compiled request,
    /// into the dispatched body, and into the schedule each request records.
    ///
    /// Every assertion here is a silent failure if it goes the other way: the ref2v adapter on the
    /// base checkpoint folds cleanly at the wrong quality, and an fl2v adapter on the reference one
    /// does the same in the other direction (sc-19563). The step count and the shift are asserted
    /// as VALUES rather than as "not the default", because 4/12.0 is what the catalog declares for
    /// both of these files and 50/absent is what the base regime is.
    #[test]
    fn the_plans_loras_resolve_per_partition_into_the_request_and_the_body() {
        let compiled = mixed_compiled_with_turbo(
            json!(["minimax_h3_ref2v_turbo_4step", "minimax_h3_turbo_4step_v01"]),
            None,
        );
        let referenced = compiled.request("SH010").unwrap();
        let plain = compiled.request("SH020").unwrap();
        assert_eq!(referenced.model, "minimax_h3_ref");
        assert_eq!(referenced.loras, vec!["minimax_h3_ref2v_turbo_4step"]);
        assert_eq!(plain.model, "minimax_h3");
        assert_eq!(plain.loras, vec!["minimax_h3_turbo_4step_v01"]);
        // The schedule each one will actually run, from the catalog's own declaration.
        assert_eq!(referenced.effective_steps, Some(4));
        assert_eq!(referenced.turbo_scheduler_shift, Some(12.0));
        assert_eq!(plain.effective_steps, Some(4));
        assert_eq!(plain.turbo_scheduler_shift, Some(12.0));
        assert_eq!(referenced.steps, None, "no plan-level override was set");

        // The body: `{ id, weight }`, the shape `generationStudio.jsx` posts, on the partition it
        // belongs to and NOWHERE else.
        let assets = role_assets();
        let context = short_edge_context(&assets);
        let referenced_body = referenced.to_job_body(&context).expect("SH010 body");
        let plain_body = plain.to_job_body(&context).expect("SH020 body");
        assert_eq!(
            referenced_body["loras"],
            json!([{ "id": "minimax_h3_ref2v_turbo_4step", "weight": 1.0 }])
        );
        assert_eq!(
            plain_body["loras"],
            json!([{ "id": "minimax_h3_turbo_4step_v01", "weight": 1.0 }])
        );
        assert!(
            referenced_body["advanced"].get("steps").is_none(),
            "no override ⇒ no advanced.steps; the recipe governs: {}",
            referenced_body["advanced"]
        );

        // A plan that names ONLY the ref2v adapter leaves the base shot with none — an empty list,
        // recorded as such, rather than the ref2v file quietly attaching to the wrong checkpoint.
        let compiled = mixed_compiled_with_turbo(json!(["minimax_h3_ref2v_turbo_4step"]), None);
        let plain = compiled.request("SH020").unwrap();
        assert!(plain.loras.is_empty());
        assert_eq!(
            plain.effective_steps,
            Some(50),
            "no recipe applies, so the model's declared default governs"
        );
        assert_eq!(plain.turbo_scheduler_shift, None);
        let body = plain.to_job_body(&context).expect("SH020 body");
        assert!(
            body.get("loras").is_none(),
            "an empty list writes no field: {body}"
        );
    }

    /// `model.advanced.steps` overrides the recipe's own count — the plan-level twin of the knob
    /// `minimax_h3_sampling` already honours — and rides `advanced.steps` on every partition,
    /// including the one no accelerator reached.
    #[test]
    fn a_plan_level_steps_override_wins_over_the_recipe_and_rides_the_body() {
        let compiled = mixed_compiled_with_turbo(json!(["minimax_h3_ref2v_turbo_4step"]), Some(6));
        let referenced = compiled.request("SH010").unwrap();
        let plain = compiled.request("SH020").unwrap();
        assert_eq!(referenced.steps, Some(6));
        assert_eq!(
            referenced.effective_steps,
            Some(6),
            "the override wins over the recipe's 4"
        );
        assert_eq!(
            referenced.turbo_scheduler_shift,
            Some(12.0),
            "the SHIFT is not overridable: a distilled checkpoint keeps its trained shift"
        );
        assert_eq!(
            plain.effective_steps,
            Some(6),
            "the override wins over the model's 50 as well"
        );
        let assets = role_assets();
        let context = short_edge_context(&assets);
        for request in [referenced, plain] {
            let body = request.to_job_body(&context).expect("body");
            assert_eq!(body["advanced"]["steps"], json!(6), "{}", request.shot_id);
        }
    }

    /// A plan that declares no LoRAs compiles exactly as it did before the field existed, and a
    /// hand edit to any of the four new derived fields is caught as a hand edit.
    #[test]
    fn a_plan_without_loras_compiles_unchanged_and_the_derived_fields_are_conformance_checked() {
        let compiled = mixed_compiled_with_turbo(json!([]), None);
        for request in &compiled.requests {
            assert!(request.loras.is_empty(), "{}", request.shot_id);
            assert_eq!(request.steps, None, "{}", request.shot_id);
            assert_eq!(request.effective_steps, Some(50), "{}", request.shot_id);
            assert_eq!(request.turbo_scheduler_shift, None, "{}", request.shot_id);
        }
        let document = serde_json::to_value(&compiled).expect("serializes");
        assert!(
            document["requests"][0].get("loras").is_none()
                && document["requests"][0].get("turboSchedulerShift").is_none(),
            "absent values write no fields: {}",
            document["requests"][0]
        );

        // Conformance: each derived field is compared, so a hand-edited compiled document is
        // refused by the field that was edited rather than dispatched.
        let mut plan_document: Value = serde_json::from_str(&mixed_plan_text()).unwrap();
        plan_document["model"]["loras"] = json!(["minimax_h3_ref2v_turbo_4step"]);
        let plan = parse_plan(&plan_document.to_string()).unwrap();
        let base = entry();
        let reference = reference_entry();
        let entries = ModelEntries::with_reference_partition(
            "minimax_h3",
            &base,
            Some(("minimax_h3_ref", &reference)),
        );
        let clean = mixed_compiled_with_turbo(json!(["minimax_h3_ref2v_turbo_4step"]), None);
        type Tamper = fn(&mut CompiledRequest);
        let cases: Vec<(&str, Tamper)> = vec![
            ("compiled.loras", |request| request.loras.clear()),
            ("compiled.steps", |request| request.steps = Some(9)),
            ("compiled.effectiveSteps", |request| {
                request.effective_steps = Some(50)
            }),
            ("compiled.turboSchedulerShift", |request| {
                request.turbo_scheduler_shift = None
            }),
        ];
        for (field, edit) in cases {
            let mut tampered = clean.clone();
            edit(&mut tampered.requests[0]);
            let fields: Vec<String> = tampered
                .conformance_findings(&plan, &pack(), &entries, ModelLane::Mlx)
                .into_iter()
                .map(|finding| finding.field)
                .collect();
            assert!(fields.contains(&field.to_owned()), "{field}: {fields:?}");
        }
    }

    /// sc-23402. The plan declares the short edge ONCE on the family; it reaches only the request
    /// that resolved to the reference partition, and only that request's job body.
    #[test]
    fn the_reference_short_edge_reaches_only_the_reference_partitions_request() {
        let compiled = mixed_compiled_with_short_edge(Some(1536));
        let referenced = compiled.request("SH010").unwrap();
        let plain = compiled.request("SH020").unwrap();
        assert_eq!(referenced.model, "minimax_h3_ref");
        assert_eq!(referenced.reference_image_short_edge, Some(1536));
        assert_eq!(plain.model, "minimax_h3");
        assert_eq!(
            plain.reference_image_short_edge, None,
            "the base partition encodes no reference, so it carries no short edge"
        );

        let assets = role_assets();
        let context = short_edge_context(&assets);
        let body = referenced.to_job_body(&context).unwrap();
        assert_eq!(body["advanced"]["referenceImageShortEdge"], json!(1536));
        let body = plain.to_job_body(&context).unwrap();
        assert!(
            body["advanced"].get("referenceImageShortEdge").is_none(),
            "{}",
            body["advanced"]
        );

        // The document survives a conformance check, and a hand-edited value is caught by name —
        // the compiled document, not the plan, is what becomes the job body.
        let mut document: Value = serde_json::from_str(&mixed_plan_text()).unwrap();
        document["model"]["advanced"] = json!({ "referenceImageShortEdge": 1536 });
        let plan = parse_plan(&document.to_string()).unwrap();
        let base = entry();
        let reference = reference_entry();
        let entries = ModelEntries::with_reference_partition(
            "minimax_h3",
            &base,
            Some(("minimax_h3_ref", &reference)),
        );
        assert!(compiled
            .conformance_findings(&plan, &pack(), &entries, ModelLane::Mlx)
            .is_empty());
        let mut tampered = compiled.clone();
        tampered.requests[0].reference_image_short_edge = Some(1024);
        let fields: Vec<String> = tampered
            .conformance_findings(&plan, &pack(), &entries, ModelLane::Mlx)
            .into_iter()
            .map(|finding| finding.field)
            .collect();
        assert_eq!(
            fields,
            vec!["compiled.referenceImageShortEdge".to_owned()],
            "{fields:?}"
        );
    }

    /// A plan that names no short edge compiles and dispatches exactly what it did before sc-23402,
    /// and the EFFECTIVE value a reference attempt records is the engine's own default — 2048, the
    /// same number `sceneworks_gen_core::effective_reference_image_short_edge` resolves (this crate
    /// has no gen-core dependency, so the rule is applied locally).
    #[test]
    fn an_absent_short_edge_dispatches_nothing_and_records_the_default_2048() {
        let compiled = mixed_compiled_with_short_edge(None);
        let referenced = compiled.request("SH010").unwrap();
        let plain = compiled.request("SH020").unwrap();
        assert_eq!(referenced.reference_image_short_edge, None);
        assert_eq!(plain.reference_image_short_edge, None);

        let assets = role_assets();
        let context = short_edge_context(&assets);
        for request in [referenced, plain] {
            let body = request.to_job_body(&context).unwrap();
            assert!(
                body["advanced"].get("referenceImageShortEdge").is_none(),
                "an absent knob dispatches no key: {}",
                body["advanced"]
            );
        }

        assert_eq!(
            referenced.effective_reference_image_short_edge(),
            Some(2048),
            "a reference request with no value still renders at the engine's default"
        );
        assert_eq!(
            plain.effective_reference_image_short_edge(),
            None,
            "a base-partition request records no short edge at all"
        );
        let asked = mixed_compiled_with_short_edge(Some(1024));
        assert_eq!(
            asked
                .request("SH010")
                .unwrap()
                .effective_reference_image_short_edge(),
            Some(1024),
            "a requested value is recorded verbatim"
        );
        assert_eq!(
            asked
                .request("SH020")
                .unwrap()
                .effective_reference_image_short_edge(),
            None
        );
    }

    #[test]
    fn each_shot_compiles_to_the_partition_its_own_conditioning_needs() {
        let plan = parse_plan(&mixed_plan_text()).unwrap();
        let base = entry();
        let reference = reference_entry();
        let entries = ModelEntries::with_reference_partition(
            "minimax_h3",
            &base,
            Some(("minimax_h3_ref", &reference)),
        );
        let compiled = compile_plan(
            &plan,
            &pack(),
            &CompileInputs {
                entries: &entries,
                lane: "mlx",
                plan_sha256: "abc123def456",
                compiled_at: "2026-09-14T00:00:00Z",
                refined_prompts: &BTreeMap::new(),
            },
        )
        .expect("compiles");
        // The plan still declares the FAMILY once; only the requests differ.
        assert_eq!(compiled.model.id, "minimax_h3");

        let referenced = compiled.request("SH010").unwrap();
        assert_eq!(referenced.model, "minimax_h3_ref");
        assert_eq!(referenced.mode, "reference_to_video");
        assert_eq!(
            referenced.reference_roles,
            vec!["courier".to_owned(), "red_parcel".to_owned()]
        );
        assert!(
            referenced.partition_reason.contains("minimax_h3_ref")
                && referenced.partition_reason.contains("2 reference role"),
            "{}",
            referenced.partition_reason
        );

        let plain = compiled.request("SH020").unwrap();
        assert_eq!(plain.model, "minimax_h3");
        assert_eq!(plain.mode, "text_to_video");
        assert!(plain.reference_roles.is_empty());
        assert!(
            plain.partition_reason.contains("no reference roles"),
            "{}",
            plain.partition_reason
        );

        // Both partitions reach the route under their own id, in role ORDER, and the reason rides
        // the payload beside it.
        let assets = role_assets();
        let context = DispatchContext {
            project_id: "proj_1",
            run_id: "run_abc",
            plan_id: "courier-workshop",
            plan_version: 2,
            attempt: 1,
            tier: Some("q4"),
            idempotency_key: Some("run_abc:SH010:a1"),
            pack: pack_ref(),
            role_assets: &assets,
        };
        let body = referenced.to_job_body(&context).unwrap();
        assert_eq!(body["model"], "minimax_h3_ref");
        assert_eq!(body["mode"], "reference_to_video");
        assert_eq!(
            body["referenceAssetIds"],
            json!(["asset_courier", "asset_parcel"])
        );
        assert_eq!(
            body["advanced"]["filmHarness"]["partitionReason"],
            json!(referenced.partition_reason)
        );
        let body = plain.to_job_body(&context).unwrap();
        assert_eq!(body["model"], "minimax_h3");
        assert!(body.get("referenceAssetIds").is_none());

        // Re-compiling the same plan says the same thing, so a conformance check on this document
        // is clean — and a document whose reference shot was re-pointed at the base checkpoint is
        // not.
        assert!(compiled
            .conformance_findings(&plan, &pack(), &entries, ModelLane::Mlx)
            .is_empty());
        let mut tampered = compiled.clone();
        tampered.requests[0].model = "minimax_h3".to_owned();
        let fields: Vec<String> = tampered
            .conformance_findings(&plan, &pack(), &entries, ModelLane::Mlx)
            .into_iter()
            .map(|finding| finding.field)
            .collect();
        assert_eq!(fields, vec!["compiled.model".to_owned()], "{fields:?}");
    }

    #[test]
    fn a_reference_shot_refuses_to_compile_when_the_partition_is_not_in_the_catalog() {
        let plan = parse_plan(&mixed_plan_text()).unwrap();
        let base = entry();
        let findings = compile_plan(
            &plan,
            &pack(),
            &CompileInputs {
                entries: &ModelEntries::single("minimax_h3", &base),
                lane: "mlx",
                plan_sha256: "abc",
                compiled_at: "now",
                refined_prompts: &BTreeMap::new(),
            },
        )
        .expect_err("refuses");
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert_eq!(findings[0].shot_id.as_deref(), Some("SH010"));
        assert!(
            findings[0].message.contains("minimax_h3_ref")
                && findings[0]
                    .message
                    .contains("not in this API's model catalog"),
            "{findings:?}"
        );
    }

    /// The mixed plan's `role_assets`, borrowed for a [`DispatchContext`]'s lifetime.
    fn mixed_entries<'a>(
        base: &'a JsonObject<String, Value>,
        reference: &'a JsonObject<String, Value>,
    ) -> ModelEntries<'a> {
        ModelEntries::with_reference_partition(
            "minimax_h3",
            base,
            Some(("minimax_h3_ref", reference)),
        )
    }

    /// sc-24023, E2. A reference shot's prompt LEADS with one binding sentence per bound role, and
    /// the `<Picture N>` each sentence names is the 1-based position that role's asset takes in the
    /// dispatched `referenceAssetIds` — both read off [`shot_reference_pictures`], which is why
    /// they cannot disagree. A shot that resolved to the base checkpoint says nothing about
    /// pictures, because it sends none.
    ///
    /// The numbering is positional and unverifiable downstream: the MiniMax-H3 text encoder labels
    /// the assets it is handed `<Picture 1>`, `<Picture 2>`, … in supply order, so a prompt that
    /// binds the courier to `<Picture 2>` while the courier's asset is dispatched first renders a
    /// confidently wrong film with nothing to flag.
    #[test]
    fn a_reference_shots_prompt_leads_with_a_binding_sentence_numbered_as_the_asset_is_dispatched()
    {
        let compiled = mixed_compiled_with_short_edge(None);
        let referenced = compiled.request("SH010").unwrap();
        let plain = compiled.request("SH020").unwrap();

        // The FIRST insertion is the binding kind, holding one sentence per bound role in picture
        // order. The audio sentence sc-24026 appends trails it and is asserted in its own tests.
        assert_eq!(referenced.inserted_text.len(), 2, "{referenced:?}");
        assert_eq!(
            referenced.inserted_text[0].kind,
            InsertedTextKind::ReferenceBinding
        );
        assert_eq!(
            referenced.inserted_text[0].text,
            "The courier is the person shown in <Picture 1>. The red parcel is the object shown in \
             <Picture 2>."
        );
        // It LEADS: the engine presents the pictures before the text, so the binding is the first
        // thing the prompt says, and the authored text survives verbatim behind it.
        assert_eq!(
            referenced.prompt,
            "The courier is the person shown in <Picture 1>. The red parcel is the object shown in \
             <Picture 2>. a courier enters. Audio: Room tone, no music."
        );
        assert!(
            referenced
                .prompt
                .starts_with(&referenced.inserted_text[0].text),
            "{}",
            referenced.prompt
        );

        // THE criterion: N == the 1-based position in the dispatched list. Read out of the job body
        // this compiled request builds rather than recomputed here, so the assertion is against the
        // shipped list builder. This is the COMPILED REQUEST's body, not the harness's own dispatch:
        // the harness resolves through the same `resolve_conditioning` and posts `to_job_body_with`,
        // and the end-to-end assertion in `apps/rust-api/src/tests/film_harness.rs` is what pins the
        // agreement on the body a real run actually sent.
        let assets = role_assets();
        let context = short_edge_context(&assets);
        let body = referenced.to_job_body(&context).expect("SH010 body");
        let dispatched: Vec<String> = body["referenceAssetIds"]
            .as_array()
            .expect("a reference request dispatches its assets")
            .iter()
            .map(|id| id.as_str().expect("asset ids are strings").to_owned())
            .collect();
        assert_eq!(dispatched, vec!["asset_courier", "asset_parcel"]);
        for (index, role) in referenced.reference_roles.iter().enumerate() {
            let number = index + 1;
            assert!(
                referenced.inserted_text[0]
                    .text
                    .contains(&format!("{} is the", role_phrase(role))),
                "{role} is never named: {}",
                referenced.inserted_text[0].text
            );
            let sentence_at = referenced.inserted_text[0]
                .text
                .find(&format!("{} is the", role_phrase(role)))
                .expect("the role is named");
            let picture_at = referenced.inserted_text[0].text[sentence_at..]
                .find(&format!("<Picture {number}>"))
                .map(|at| at + sentence_at);
            assert!(
                picture_at.is_some(),
                "{role} must be bound to <Picture {number}>, the position its asset takes in \
                 {dispatched:?}: {}",
                referenced.inserted_text[0].text
            );
            assert_eq!(
                dispatched[index],
                *assets.get(role).expect("every role imported"),
                "{role} is dispatched at position {number}"
            );
        }

        // The base partition sends no pictures, so it claims no BINDING — its only insertion is the
        // audio sentence every shot carries (sc-24026), and the plan's own words are untouched.
        assert!(
            plain
                .inserted_text
                .iter()
                .all(|piece| piece.kind != InsertedTextKind::ReferenceBinding),
            "{plain:?}"
        );
        assert_eq!(
            plain.prompt,
            "places the parcel. Audio: Room tone, no music."
        );
        let body = plain.to_job_body(&context).expect("SH020 body");
        assert!(body.get("referenceAssetIds").is_none(), "{body}");
        assert!(
            !body["prompt"]
                .as_str()
                .expect("a prompt is dispatched")
                .contains("<Picture"),
            "{body}"
        );

        // The document round-trips with the new field and nothing unknown.
        let text = serde_json::to_string(&compiled).unwrap();
        let back: CompiledPlan = serde_json::from_str(&text).unwrap();
        assert_eq!(back, compiled);
        let document: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(
            document["requests"][0]["insertedText"][0]["kind"],
            json!("reference_binding")
        );
        // A shot that binds nothing records no binding — only the audio sentence every shot
        // carries since sc-24026, so the two kinds stay distinguishable in the document.
        assert_eq!(
            document["requests"][1]["insertedText"]
                .as_array()
                .unwrap_or_else(|| panic!("{}", document["requests"][1]))
                .iter()
                .map(|piece| piece["kind"].clone())
                .collect::<Vec<_>>(),
            vec![json!("audio")]
        );
    }

    /// sc-24023, E5. The binding sentences are written AFTER the refine rewrite, so the refiner
    /// cannot paraphrase a `<Picture N>` into a label the engine never applies — and the record
    /// keeps the three texts apart: the plan's, the model's rewrite, and the compiler's own.
    #[test]
    fn the_binding_sentences_are_written_after_the_refine_rewrite_and_recorded_apart_from_it() {
        let mut document: Value = serde_json::from_str(&mixed_plan_text()).unwrap();
        document["model"]["advanced"] = json!({ "referenceImageShortEdge": 1536 });
        let plan = parse_plan(&document.to_string()).unwrap();
        let base = entry();
        let reference = reference_entry();
        let refined: BTreeMap<String, String> = [(
            "SH010".to_owned(),
            "integrated_multimodal_description: a courier steps into a warm workshop".to_owned(),
        )]
        .into_iter()
        .collect();
        let compiled = compile_plan(
            &plan,
            &pack(),
            &CompileInputs {
                entries: &mixed_entries(&base, &reference),
                lane: "mlx",
                plan_sha256: "abc",
                compiled_at: "now",
                refined_prompts: &refined,
            },
        )
        .expect("the refined mixed plan compiles");
        let referenced = compiled.request("SH010").unwrap();
        assert_eq!(referenced.prompt_source, PromptSource::Refined);
        // The rewrite is carried through UNTOUCHED, behind the bindings and ahead of the audio.
        assert_eq!(
            referenced.prompt,
            "The courier is the person shown in <Picture 1>. The red parcel is the object shown in \
             <Picture 2>. integrated_multimodal_description: a courier steps into a warm workshop. \
             Audio: Room tone, no music."
        );
        // The authored prompt is the PLAN's, with no compiler text in it: the insertion happened
        // after the rewrite, not before it, so neither recorded text has been polluted.
        assert_eq!(
            referenced.authored_prompt.as_deref(),
            Some("a courier enters")
        );
        assert!(
            !referenced
                .authored_prompt
                .as_deref()
                .unwrap()
                .contains("<Picture"),
            "the authored prompt must stay the plan's own words"
        );
        assert_eq!(
            referenced
                .inserted_text
                .iter()
                .map(|piece| piece.kind)
                .collect::<Vec<_>>(),
            vec![InsertedTextKind::ReferenceBinding, InsertedTextKind::Audio]
        );
        for piece in &referenced.inserted_text {
            assert!(
                !piece.text.contains("courier steps into"),
                "the inserted text is the compiler's alone: {}",
                piece.text
            );
        }
    }

    /// sc-24026. EVERY shot's prompt ends with its own `Audio:` sentence — base partition and
    /// reference partition alike — written after the refine rewrite and recorded apart from it.
    #[test]
    fn every_shots_prompt_ends_with_its_audio_sentence_on_both_partitions() {
        let mut document: Value = serde_json::from_str(&mixed_plan_text()).unwrap();
        document["shots"][0]["audio"] = json!("A door latch clicking.   Room tone.\nNo music.");
        document["shots"][1]["audio"] = json!("No audio. Silence.");
        let plan = parse_plan(&document.to_string()).unwrap();
        let base = entry();
        let reference = reference_entry();
        // Refined, so the insertion is provably AFTER the rewrite on both shots rather than
        // something the authored text could have carried.
        let refined: BTreeMap<String, String> = plan
            .shots
            .iter()
            .map(|shot| (shot.id.clone(), format!("rewritten {}", shot.id)))
            .collect();
        let compiled = compile_plan(
            &plan,
            &pack(),
            &CompileInputs {
                entries: &mixed_entries(&base, &reference),
                lane: "mlx",
                plan_sha256: "abc",
                compiled_at: "now",
                refined_prompts: &refined,
            },
        )
        .expect("the mixed plan compiles");

        // Two DIFFERENT partitions, so this is not one lane asserted twice.
        let models: Vec<&str> = compiled
            .requests
            .iter()
            .map(|request| request.model.as_str())
            .collect();
        assert!(
            models.contains(&"minimax_h3") && models.contains(&"minimax_h3_ref"),
            "{models:?}"
        );

        for request in &compiled.requests {
            let shot = plan
                .shots
                .iter()
                .find(|shot| shot.id == request.shot_id)
                .unwrap();
            let audio = request
                .inserted_text
                .iter()
                .find(|piece| piece.kind == InsertedTextKind::Audio)
                .unwrap_or_else(|| panic!("{} records its audio text", request.shot_id));
            // Whitespace-normalized exactly as a pack description is, and otherwise the author's
            // own words: the runs of spaces and the newline above are collapsed, nothing else.
            assert_eq!(
                audio.text,
                format!("Audio: {}", normalized_description(&shot.audio)),
                "{}",
                request.shot_id
            );
            assert!(
                request.prompt.ends_with(&audio.text),
                "{}: {}",
                request.shot_id,
                request.prompt
            );
            // The rewrite is still in there, ahead of it — the insertion ran after the refiner.
            assert!(
                request.prompt.contains(&format!("rewritten {}", shot.id)),
                "{}: {}",
                request.shot_id,
                request.prompt
            );
            assert!(
                !request
                    .authored_prompt
                    .as_deref()
                    .unwrap()
                    .contains("Audio:"),
                "{}: the authored prompt stays the plan's own words",
                request.shot_id
            );
        }
        assert_eq!(compiled.requests.len(), plan.shots.len());
    }

    /// sc-24026. A trailing insertion follows text nobody guaranteed was punctuated — an authored
    /// prompt, a refiner rewrite, or the author's own `audio` sentence. Joined by a bare space it
    /// reads as one clause (`...a courier enters Audio: Room tone`), so the compiler supplies the
    /// missing `.` — exactly one, and only when one is missing.
    #[test]
    fn an_unpunctuated_prompt_is_joined_to_its_audio_sentence_by_exactly_one_sentence_break() {
        // Each case: the prompt, the audio sentence, and the exact tail the dispatched text must
        // have. Shot 0 of the mixed plan takes no reference bindings' worth of rewriting here —
        // only the trailing join is under test.
        for (prompt, audio, expected_tail) in [
            // No terminal punctuation at all: the compiler writes the boundary.
            (
                "a courier enters",
                "Room tone, no music.",
                "a courier enters. Audio: Room tone, no music.",
            ),
            // Already a sentence: nothing is added, and there is no ".." anywhere.
            (
                "a courier enters.",
                "Room tone, no music.",
                "a courier enters. Audio: Room tone, no music.",
            ),
            // The other terminals, and trailing whitespace, count as ended too.
            (
                "does he knock?",
                "Room tone.",
                "does he knock? Audio: Room tone.",
            ),
            ("he knocks!  ", "Room tone.", "he knocks! Audio: Room tone."),
            (
                "he trails off…",
                "Room tone.",
                "he trails off… Audio: Room tone.",
            ),
            // A closing quote or bracket that itself closes a punctuated sentence.
            (
                "she says \"open it.\"",
                "Room tone.",
                "she says \"open it.\" Audio: Room tone.",
            ),
            ("(he waits.)", "Room tone.", "(he waits.) Audio: Room tone."),
            // A closing quote with NO punctuation inside it is still unfinished.
            (
                "she says \"open it\"",
                "Room tone.",
                "she says \"open it\". Audio: Room tone.",
            ),
        ] {
            let mut document: Value = serde_json::from_str(&mixed_plan_text()).unwrap();
            document["shots"][0]["prompt"] = json!(prompt);
            document["shots"][0]["audio"] = json!(audio);
            let plan = parse_plan(&document.to_string()).unwrap();
            let base = entry();
            let reference = reference_entry();
            let compiled = compile_plan(
                &plan,
                &pack(),
                &CompileInputs {
                    entries: &mixed_entries(&base, &reference),
                    lane: "mlx",
                    plan_sha256: "abc",
                    compiled_at: "now",
                    refined_prompts: &BTreeMap::new(),
                },
            )
            .expect("the mixed plan compiles");
            let request = compiled.request(&plan.shots[0].id).unwrap();
            assert!(
                request.prompt.ends_with(expected_tail),
                "{prompt:?} + {audio:?}\n  wanted tail: {expected_tail:?}\n  got:         {:?}",
                request.prompt
            );
            // Exactly ONE `. ` joins the prompt to the label — never a doubled period and never a
            // bare space.
            assert!(
                !request.prompt.contains(".. Audio:") && !request.prompt.contains("  Audio:"),
                "{:?}",
                request.prompt
            );
            let before_label = &request.prompt[..request.prompt.find("Audio:").unwrap()];
            assert!(
                before_label.ends_with(". ")
                    || before_label.ends_with("? ")
                    || before_label.ends_with("! ")
                    || before_label.ends_with("… ")
                    || before_label.ends_with("\" ")
                    || before_label.ends_with(") "),
                "the label is preceded by a finished sentence and one space: {before_label:?}"
            );
        }
    }

    /// The same boundary rule applies to the AUTHOR's audio text before the fixed no-speech
    /// sentence, which is the other place compiler-owned text follows prose it did not write
    /// (sc-24026).
    #[test]
    fn an_unpunctuated_audio_sentence_is_joined_to_the_no_speech_sentence_by_a_sentence_break() {
        let mut document: Value = serde_json::from_str(&mixed_plan_text()).unwrap();
        document["shots"][0]["prompt"] = json!("a courier enters");
        document["shots"][0]["audio"] = json!("room tone and a low hum");
        document["shots"][0]["dialogueClip"] =
            json!({ "role": "courier_line", "offsetSeconds": 0.5 });
        let plan = parse_plan(&document.to_string()).unwrap();
        let base = entry();
        let reference = reference_entry();
        let compiled = compile_plan(
            &plan,
            &pack(),
            &CompileInputs {
                entries: &mixed_entries(&base, &reference),
                lane: "mlx",
                plan_sha256: "abc",
                compiled_at: "now",
                refined_prompts: &BTreeMap::new(),
            },
        )
        .expect("the mixed plan compiles");
        let request = compiled.request(&plan.shots[0].id).unwrap();
        assert!(
            request.prompt.ends_with(&format!(
                "a courier enters. Audio: room tone and a low hum. {NO_SPEECH_SENTENCE}"
            )),
            "{:?}",
            request.prompt
        );
        // The recorded insertion is still the author's own words: the boundary is composition, not
        // a rewrite of what `insertedText` says was added.
        assert_eq!(
            request
                .inserted_text
                .iter()
                .find(|piece| piece.kind == InsertedTextKind::Audio)
                .unwrap()
                .text,
            "Audio: room tone and a low hum"
        );
    }

    /// sc-24026. The no-speech sentence constrains the SOUNDTRACK. These are exactly the shots with
    /// someone speaking on camera, so wording it as a statement about the picture would tell H3 to
    /// render closed mouths under our own dialogue track.
    #[test]
    fn the_no_speech_sentence_constrains_the_soundtrack_and_never_the_picture() {
        assert_eq!(
            NO_SPEECH_SENTENCE,
            "No spoken dialogue in the generated audio; no voices on the soundtrack."
        );
        let lowered = NO_SPEECH_SENTENCE.to_ascii_lowercase();
        for picture_word in ["on camera", "mouth", "lips", "silently", "no one speaks"] {
            assert!(
                !lowered.contains(picture_word),
                "the sentence must not direct the picture: {picture_word:?}"
            );
        }
        assert!(lowered.contains("generated audio") && lowered.contains("soundtrack"));
    }

    /// sc-24026. The no-speech sentence is keyed on a PLACED dialogue line — `dialogueClip`, which
    /// is what puts our own voice on the dialogue bus — and on nothing else. Intent prose in
    /// `dialogue` is not a placement and gets no sentence.
    #[test]
    fn only_a_shot_that_places_a_dialogue_line_gets_the_no_speech_sentence() {
        let mut document: Value = serde_json::from_str(&mixed_plan_text()).unwrap();
        // Shot 0 SPEAKS a placed line; shot 1 only declares the intent prose beside it.
        document["shots"][0]["dialogueClip"] =
            json!({ "role": "courier_line", "offsetSeconds": 0.5 });
        document["shots"][1]["dialogue"] = json!("Recipient: \"Huh.\"");
        let plan = parse_plan(&document.to_string()).unwrap();
        let base = entry();
        let reference = reference_entry();
        let compiled = compile_plan(
            &plan,
            &pack(),
            &CompileInputs {
                entries: &mixed_entries(&base, &reference),
                lane: "mlx",
                plan_sha256: "abc",
                compiled_at: "now",
                refined_prompts: &BTreeMap::new(),
            },
        )
        .expect("the mixed plan compiles");

        let placed = compiled.request(&plan.shots[0].id).unwrap();
        assert!(
            placed.prompt.ends_with(NO_SPEECH_SENTENCE),
            "{}: {}",
            placed.shot_id,
            placed.prompt
        );
        // After the audio sentence, not instead of it, and recorded as its own kind.
        assert_eq!(
            placed
                .inserted_text
                .iter()
                .map(|piece| piece.kind)
                .filter(|kind| *kind != InsertedTextKind::ReferenceBinding)
                .collect::<Vec<_>>(),
            vec![InsertedTextKind::Audio, InsertedTextKind::NoSpeech]
        );
        assert!(
            placed
                .prompt
                .contains(&format!("Audio: {} ", plan.shots[0].audio.trim()))
                || placed.prompt.contains(&format!(
                    "Audio: {} {NO_SPEECH_SENTENCE}",
                    normalized_description(&plan.shots[0].audio)
                )),
            "{}",
            placed.prompt
        );

        let unplaced = compiled.request(&plan.shots[1].id).unwrap();
        assert!(
            !unplaced.prompt.contains(NO_SPEECH_SENTENCE),
            "a shot with only dialogue INTENT places no voice, so it silences none: {}",
            unplaced.prompt
        );
        assert!(unplaced
            .inserted_text
            .iter()
            .all(|piece| piece.kind != InsertedTextKind::NoSpeech));
    }

    /// The sentence a role gets is built from its pack entry: the KIND chooses the noun, and the
    /// author's own DESCRIPTION is repeated verbatim rather than paraphrased.
    #[test]
    fn the_pack_kind_chooses_the_noun_and_the_description_is_repeated_verbatim() {
        let described = parse_reference_pack(
            &json!({
                "schemaVersion": 1,
                "id": "described",
                "version": 1,
                "references": [
                    { "role": "courier", "kind": "character", "file": "references/a.png",
                      "description": "Blue jacket, carries the parcel." },
                    { "role": "red_parcel", "kind": "prop", "file": "references/b.png",
                      "description": "Small bright red cardboard parcel" },
                    { "role": "workshop_location", "kind": "location", "file": "references/c.png",
                      "description": "Cluttered woodworking workshop, door camera-left." }
                ]
            })
            .to_string(),
        )
        .unwrap();
        let mut document: Value = serde_json::from_str(&mixed_plan_text()).unwrap();
        document["shots"][0]["conditioning"]["referenceRoles"] =
            json!(["courier", "red_parcel", "workshop_location"]);
        let plan = parse_plan(&document.to_string()).unwrap();
        let base = entry();
        let reference = reference_entry();
        let compiled = compile_plan(
            &plan,
            &described,
            &CompileInputs {
                entries: &mixed_entries(&base, &reference),
                lane: "mlx",
                plan_sha256: "abc",
                compiled_at: "now",
                refined_prompts: &BTreeMap::new(),
            },
        )
        .expect("compiles");
        assert_eq!(
            compiled.request("SH010").unwrap().inserted_text[0].text,
            "The courier is the person shown in <Picture 1>. Blue jacket, carries the parcel. \
             The red parcel is the object shown in <Picture 2>. Small bright red cardboard parcel. \
             The workshop location is the place shown in <Picture 3>. Cluttered woodworking \
             workshop, door camera-left."
        );
    }

    /// sc-24023. A description's WHITESPACE is normalized before it is appended: the binding
    /// sentence is one line of a dispatched prompt, and a pack description that wrapped over several
    /// lines would otherwise put a raw newline and tab run into the prompt — inside `insertedText`,
    /// the one field `conformance_findings` reads as the compiler's own authored text and therefore
    /// never checks. Only the joined whole was ever trimmed, never each description.
    #[test]
    fn a_multi_line_description_is_collapsed_to_one_line_before_it_reaches_the_prompt() {
        let wrapped = parse_reference_pack(
            &json!({
                "schemaVersion": 1,
                "id": "wrapped",
                "version": 1,
                "references": [
                    { "role": "courier", "kind": "character", "file": "references/a.png",
                      "description": "  Blue jacket,\r\n\tcarries   the parcel.  " },
                    { "role": "red_parcel", "kind": "prop", "file": "references/b.png",
                      "description": "Small\nred parcel" }
                ]
            })
            .to_string(),
        )
        .unwrap();
        let mut document: Value = serde_json::from_str(&mixed_plan_text()).unwrap();
        document["shots"][0]["conditioning"]["referenceRoles"] = json!(["courier", "red_parcel"]);
        let plan = parse_plan(&document.to_string()).unwrap();
        let base = entry();
        let reference = reference_entry();
        let compiled = compile_plan(
            &plan,
            &wrapped,
            &CompileInputs {
                entries: &mixed_entries(&base, &reference),
                lane: "mlx",
                plan_sha256: "abc",
                compiled_at: "now",
                refined_prompts: &BTreeMap::new(),
            },
        )
        .expect("compiles");
        let request = compiled.request("SH010").unwrap();
        assert_eq!(
            request.inserted_text[0].text,
            "The courier is the person shown in <Picture 1>. Blue jacket, carries the parcel. \
             The red parcel is the object shown in <Picture 2>. Small red parcel."
        );
        // And the prompt the engine receives carries no control run either.
        assert!(
            !request.prompt.contains(['\n', '\r', '\t']),
            "{:?}",
            request.prompt
        );
    }

    /// The compiler's own sentences are a DERIVED field: a hand-edited `insertedText` — the one
    /// edit that could repoint a binding at the wrong picture without changing anything else the
    /// document declares — is refused by name, exactly like a swapped model or duration.
    #[test]
    fn a_hand_edited_inserted_text_is_refused_like_every_other_derived_field() {
        let plan = parse_plan(&mixed_plan_text()).unwrap();
        let base = entry();
        let reference = reference_entry();
        let entries = mixed_entries(&base, &reference);
        let clean = mixed_compiled_with_short_edge(None);
        assert!(clean
            .conformance_findings(&plan, &pack(), &entries, ModelLane::Mlx)
            .is_empty());

        let mut tampered = clean.clone();
        tampered.requests[0].inserted_text[0].text =
            "The courier is the person shown in <Picture 2>.".to_owned();
        let findings = tampered.conformance_findings(&plan, &pack(), &entries, ModelLane::Mlx);
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert_eq!(findings[0].field, "compiled.insertedText", "{findings:?}");
        assert_eq!(findings[0].shot_id.as_deref(), Some("SH010"));

        // Deleting the insertion entirely is the same refusal: an unbound reference prompt is not
        // what compiling this plan produces.
        let mut emptied = clean;
        emptied.requests[0].inserted_text.clear();
        let fields: Vec<String> = emptied
            .conformance_findings(&plan, &pack(), &entries, ModelLane::Mlx)
            .into_iter()
            .map(|finding| finding.field)
            .collect();
        assert_eq!(
            fields,
            vec!["compiled.insertedText".to_owned()],
            "{fields:?}"
        );
    }

    /// sc-24023, E6. A `compiled.json` written by the PREVIOUS schema version is refused BY
    /// VERSION, not blamed on the operator as a hand edit — the same rule the v2 and v3 bumps
    /// established. Such a document has no `insertedText` key and a prompt with no binding
    /// sentences, which conformance would otherwise report as tampering.
    #[test]
    fn a_previous_version_compiled_document_is_refused_by_schema_version_not_as_tampered() {
        let plan = parse_plan(&mixed_plan_text()).unwrap();
        let base = entry();
        let reference = reference_entry();
        let entries = mixed_entries(&base, &reference);

        // The BUMP itself. 3 is the version a `compiled.json` written before this story carries,
        // and its requests have no `insertedText` and no binding sentences; leaving the constant
        // at 3 would let such a document pass `staleness_findings` and then be blamed at
        // conformance for a hand edit it never made. The number is named, not derived, because it
        // is the specific document on disk that has to be refused — leave the constant at 3 and
        // this document stops producing a finding at all.
        let mut v3 = mixed_compiled_with_short_edge(None);
        v3.schema_version = 3;
        for request in &mut v3.requests {
            request.inserted_text.clear();
        }
        let findings = v3.staleness_findings(&plan, "abc", &pack());
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert_eq!(findings[0].field, "compiled.schemaVersion", "{findings:?}");

        // Exactly what the previous version on disk deserializes to: the old number, and the key
        // absent.
        let mut previous = mixed_compiled_with_short_edge(None);
        previous.schema_version = COMPILED_PLAN_SCHEMA_VERSION - 1;
        for request in &mut previous.requests {
            request.inserted_text.clear();
        }
        let round_tripped: CompiledPlan =
            serde_json::from_value(serde_json::to_value(&previous).unwrap()).unwrap();
        assert!(round_tripped.requests[0].inserted_text.is_empty());

        let findings = round_tripped.staleness_findings(&plan, "abc", &pack());
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert_eq!(findings[0].field, "compiled.schemaVersion", "{findings:?}");
        assert!(
            findings[0].message.contains(&format!(
                "schema version {}",
                COMPILED_PLAN_SCHEMA_VERSION - 1
            )) && findings[0].message.contains("film-harness compile"),
            "the refusal must name the version AND the remedy: {findings:?}"
        );

        // And this is the finding the bump replaced: at the CURRENT version the same missing
        // insertion is (correctly) a tampering report.
        let mut current = round_tripped;
        current.schema_version = COMPILED_PLAN_SCHEMA_VERSION;
        assert!(current.staleness_findings(&plan, "abc", &pack()).is_empty());
        assert!(
            current
                .conformance_findings(&plan, &pack(), &entries, ModelLane::Mlx)
                .iter()
                .any(|finding| finding.field == "compiled.insertedText"),
            "without the bump a previous-version document lands here instead"
        );
    }

    /// A prompt that no longer fits once the bindings lead it is a FINDING naming the shot, not a
    /// silently truncated prompt — the same rule an unusable refinement already obeys.
    #[test]
    fn a_prompt_that_no_longer_fits_once_the_bindings_lead_it_is_refused() {
        let long = parse_reference_pack(
            &json!({
                "schemaVersion": 1,
                "id": "long",
                "version": 1,
                "references": [
                    { "role": "courier", "kind": "character", "file": "references/a.png",
                      "description": "x".repeat(MAX_PROMPT_CHARS) },
                    { "role": "red_parcel", "kind": "prop", "file": "references/b.png" }
                ]
            })
            .to_string(),
        )
        .unwrap();
        let plan = parse_plan(&mixed_plan_text()).unwrap();
        let base = entry();
        let reference = reference_entry();
        let findings = compile_plan(
            &plan,
            &long,
            &CompileInputs {
                entries: &mixed_entries(&base, &reference),
                lane: "mlx",
                plan_sha256: "abc",
                compiled_at: "now",
                refined_prompts: &BTreeMap::new(),
            },
        )
        .expect_err("refuses");
        // BOTH shots, for two different reasons, and the message says which is which (sc-24025):
        // SH010 BINDS the courier, so the over-long description arrives in its binding sentence;
        // SH020 binds nothing at all and merely lists the courier in `continuityRoles`, so the
        // same description arrives as identity text. The second is the case an author cannot
        // otherwise account for — a reference-free shot refused over a reference's description —
        // so the refusal names the identity text and attributes the characters to it.
        let by_shot: BTreeMap<&str, &PlanDiagnostic> = findings
            .iter()
            .map(|finding| (finding.shot_id.as_deref().unwrap_or_default(), finding))
            .collect();
        assert_eq!(by_shot.len(), findings.len(), "one per shot: {findings:?}");
        for shot_id in ["SH010", "SH020"] {
            let finding = by_shot
                .get(shot_id)
                .unwrap_or_else(|| panic!("{shot_id} is refused: {findings:?}"));
            assert_eq!(finding.field, "prompt");
            assert!(
                finding.message.contains("the compiler's own sentences")
                    && finding.message.contains("the video route accepts")
                    && finding.message.contains("identity text"),
                "{finding:?}"
            );
        }
        // The attribution is per shot and truthful, asserted as SHAPE rather than as a character
        // count this test would have to recompute: the bound shot's long text is binding sentence
        // and none of it is identity text; the unbound shot's is the exact reverse.
        assert!(
            by_shot["SH010"].message.contains("0 as identity text")
                && !by_shot["SH010"]
                    .message
                    .contains("wrote 0 as reference binding"),
            "SH010 binds the courier, so its characters are binding sentence: {:?}",
            by_shot["SH010"].message
        );
        assert!(
            by_shot["SH020"]
                .message
                .contains("wrote 0 as reference binding")
                && !by_shot["SH020"].message.contains("0 as identity text"),
            "SH020 binds nothing, so all of its inserted characters are identity text: {:?}",
            by_shot["SH020"].message
        );
        // EVERY kind is accounted for, not only the two leading ones (sc-24029). Both fixture
        // shots trail with an `Audio:` sentence, so a message that stopped at the bindings and the
        // identity text left real inserted characters out of its own breakdown — the author is
        // told the prompt is too long and then handed an incomplete account of who made it so.
        // Asserted as SHAPE: every kind's label appears with a count beside it, and the audio
        // sentence's count is positive because these shots have one.
        for finding in by_shot.values() {
            for kind in InsertedTextKind::ALL {
                assert!(
                    finding.message.contains(kind.label()),
                    "{} is unaccounted for in {:?}",
                    kind.label(),
                    finding.message
                );
            }
            assert!(
                !finding
                    .message
                    .contains(&format!("0 as {}", InsertedTextKind::Audio.label())),
                "both fixture shots carry an audio sentence, so its count cannot be zero: {:?}",
                finding.message
            );
            // And the one kind neither shot has reads as zero rather than being omitted.
            assert!(
                finding
                    .message
                    .contains(&format!("0 as {}", InsertedTextKind::NoSpeech.label())),
                "neither fixture shot places a dialogue clip: {:?}",
                finding.message
            );
        }
    }

    #[test]
    fn tier_maps_to_the_shared_mlx_quantize_convention() {
        assert_eq!(mlx_quantize_for_tier("q4"), json!(4));
        assert_eq!(mlx_quantize_for_tier("q8"), json!(8));
        assert_eq!(mlx_quantize_for_tier("bf16"), json!(0));
    }

    #[test]
    fn production_plan_hash_matches_the_persisted_pretty_document() {
        let plan = parse_plan(&mixed_plan_text()).unwrap();
        let mut persisted = serde_json::to_vec_pretty(&plan).unwrap();
        persisted.push(b'\n');
        let expected = format!("{:x}", Sha256::digest(&persisted));
        assert_eq!(production_plan_sha256(&plan).unwrap(), expected);
    }

    /// A shot off the fixture plan, for the tests whose subject is the insertions rather than the
    /// shot. It has an ordinary `audio` sentence and no placed dialogue clip, so
    /// [`inserted_text_for_shot`] returns the bindings and one audio piece (sc-24026).
    fn fixture_shot() -> Shot {
        parse_plan(&mixed_plan_text())
            .expect("the fixture plan parses")
            .shots[0]
            .clone()
    }

    /// The fixture pack with the courier and the recipient on ONE plate, each with a locator — a
    /// photograph of two people (sc-24024). `third` is a role with a plate of its own, so the
    /// numbering below is exercised against a picture that follows a shared one.
    fn shared_plate_pack() -> ReferencePack {
        parse_reference_pack(
            &json!({
                "schemaVersion": crate::film_plan::REFERENCE_PACK_SCHEMA_VERSION,
                "id": "pair-refs",
                "version": 1,
                "references": [
                    {
                        "role": "courier", "kind": "character", "file": "references/pair.png",
                        "locator": "the woman on the left",
                        "description": "Blue jacket."
                    },
                    {
                        "role": "recipient", "kind": "character", "file": "references/pair.png",
                        "locator": "  the man\non the  right  "
                    },
                    { "role": "red_parcel", "kind": "prop", "file": "references/red_parcel.png" }
                ]
            })
            .to_string(),
        )
        .unwrap()
    }

    /// Roles naming the same pack file are ONE picture, numbered at the first of them, and the
    /// next role's own file is the NEXT number — not the next role's index (sc-24024).
    #[test]
    fn roles_that_share_a_file_share_one_picture_and_do_not_consume_its_number() {
        let pack = shared_plate_pack();
        let roles = [
            "courier".to_owned(),
            "recipient".to_owned(),
            "red_parcel".to_owned(),
        ];
        let pictures = shot_reference_pictures(&roles, &pack);

        assert_eq!(pictures.len(), 2, "two files, two pictures: {pictures:?}");
        assert_eq!(pictures[0].number, 1);
        assert_eq!(
            pictures[0]
                .roles
                .iter()
                .map(|bound| bound.role.as_str())
                .collect::<Vec<_>>(),
            vec!["courier", "recipient"],
            "both roles on the shared plate ride one picture, in the shot's order"
        );
        assert_eq!(
            pictures[1].number, 2,
            "the third role is the SECOND picture"
        );
        assert_eq!(pictures[1].dispatch_role(), Some("red_parcel"));
        assert_eq!(
            pictures[0].dispatch_role(),
            Some("courier"),
            "one file is dispatched once, under the first role that named it"
        );

        // A role bound TWICE to the same file through the same entry cannot appear twice either:
        // the reversed order proves the number follows the first occurrence, not the pack's order.
        let reversed = [
            "red_parcel".to_owned(),
            "recipient".to_owned(),
            "courier".to_owned(),
        ];
        let pictures = shot_reference_pictures(&reversed, &pack);
        assert_eq!(pictures.len(), 2);
        assert_eq!(pictures[0].dispatch_role(), Some("red_parcel"));
        assert_eq!(pictures[1].number, 2);
        assert_eq!(pictures[1].dispatch_role(), Some("recipient"));
    }

    /// Each sharing role's binding sentence carries ITS OWN locator against the shared
    /// `<Picture N>`, whitespace-normalized like every other pack phrase the compiler repeats; a
    /// role with a plate to itself keeps the wording it had before locators existed (sc-24024).
    #[test]
    fn a_locator_replaces_the_bare_noun_in_that_roles_binding_sentence() {
        let pack = shared_plate_pack();
        let roles = [
            "courier".to_owned(),
            "recipient".to_owned(),
            "red_parcel".to_owned(),
        ];
        // A real shot, because since sc-24026 the insertions for a shot are the bindings AND its
        // audio sentence. The bindings are what this test is about, so they are selected by kind
        // rather than by position.
        let shot = fixture_shot();
        let inserted =
            inserted_text_for_shot(&shot, &shot_reference_pictures(&roles, &pack), &pack);
        assert_eq!(
            inserted.iter().map(|piece| piece.kind).collect::<Vec<_>>(),
            vec![InsertedTextKind::ReferenceBinding, InsertedTextKind::Audio],
            "{inserted:?}"
        );
        let text = inserted[0].text.as_str();

        assert!(
            text.contains("The courier is the woman on the left in <Picture 1>. Blue jacket."),
            "{text:?}"
        );
        assert!(
            text.contains("The recipient is the man on the right in <Picture 1>."),
            "a locator's whitespace is normalized before it reaches the prompt: {text:?}"
        );
        assert!(
            text.contains("The red parcel is the object shown in <Picture 2>."),
            "a role with its own file keeps the unlocated wording: {text:?}"
        );
        assert!(!text.contains("<Picture 3>"), "{text:?}");

        // A shot may bind only ONE of the two roles on the shared plate. The image still shows two
        // people, so the locator is still what says which one the courier is — it is read off the
        // bound ENTRY, never off the group of roles this shot happens to bind (sc-24024).
        let alone = inserted_text_for_shot(
            &shot,
            &shot_reference_pictures(&["courier".to_owned()], &shared_plate_pack()),
            &shared_plate_pack(),
        );
        assert_eq!(
            alone[0].kind,
            InsertedTextKind::ReferenceBinding,
            "{alone:?}"
        );
        assert!(
            alone[0]
                .text
                .contains("The courier is the woman on the left in <Picture 1>."),
            "the locator survives a shot that binds neither of its co-subjects: {:?}",
            alone[0].text
        );
    }

    /// THE SHARED-FILE RULE (sc-24025). A continuity role the shot does not list, but whose file
    /// is the file of a picture it IS binding, is named BY PICTURE — its image is already inside
    /// that `<Picture N>`, and describing it on its own would put a second subject in the prompt
    /// with nothing tying it to the picture it is visibly in.
    ///
    /// Everything about the dispatch stays as it was: one file is one picture, numbered once, and
    /// one asset is sent.
    #[test]
    fn a_continuity_role_sharing_a_bound_picture_is_bound_to_it_and_never_described_twice() {
        let mut pack = shared_plate_pack();
        // The shipped fixture leaves the recipient silent; give it words, so "described twice"
        // is a state this test could actually observe.
        pack.references[1].description = "Grey work apron.".to_owned();

        // Through the real compiler, so the prompt asserted below is the prompt a run dispatches.
        let mut document: Value = serde_json::from_str(&mixed_plan_text()).unwrap();
        document["shots"][0]["conditioning"] =
            json!({ "mode": "reference_to_video", "referenceRoles": ["courier"] });
        document["shots"][0]["continuityRoles"] = json!(["recipient"]);
        let plan = parse_plan(&document.to_string()).expect("the edited plan parses");
        let base = entry();
        let reference = reference_entry();
        let compiled = compile_plan(
            &plan,
            &pack,
            &CompileInputs {
                entries: &mixed_entries(&base, &reference),
                lane: "mlx",
                plan_sha256: "abc",
                compiled_at: "now",
                refined_prompts: &BTreeMap::new(),
            },
        )
        .expect("the plan compiles against the shared plate pack");
        let request = compiled.request("SH010").expect("SH010 compiled");

        assert_eq!(
            request
                .inserted_text
                .iter()
                .map(|piece| piece.kind)
                .collect::<Vec<_>>(),
            vec![InsertedTextKind::ReferenceBinding, InsertedTextKind::Audio],
            "the sharer is BOUND, so there is no identity text at all: {request:?}"
        );
        assert_eq!(
            request.inserted_text[0].text,
            "The courier is the woman on the left in <Picture 1>. Blue jacket. The recipient is \
             the man on the right in <Picture 1>. Grey work apron.",
            "two binding sentences on ONE picture number, the listed role's first"
        );
        assert!(
            !request.inserted_text[0].text.contains("<Picture 2>"),
            "the shared file is one picture: {:?}",
            request.inserted_text[0].text
        );
        assert_eq!(
            request.prompt.matches("Grey work apron.").count(),
            1,
            "a description is never stated twice: {:?}",
            request.prompt
        );

        // The dispatch is untouched: one picture, numbered 1, one asset, sent under the role the
        // shot actually listed.
        let pictures = shot_reference_pictures(&request.reference_roles, &pack);
        assert_eq!(pictures.len(), 1, "{pictures:?}");
        assert_eq!(pictures[0].number, 1);
        assert_eq!(pictures[0].dispatch_role(), Some("courier"));
        let role_assets = BTreeMap::from([
            ("courier".to_owned(), "asset_courier".to_owned()),
            ("recipient".to_owned(), "asset_recipient".to_owned()),
        ]);
        let resolved = request
            .resolve_conditioning(&pack, &role_assets)
            .expect("the bound role was imported");
        assert_eq!(
            resolved.reference_asset_ids,
            vec!["asset_courier".to_owned()],
            "the image was already being sent; the rule adds no asset"
        );
    }

    /// The other half of the rule: with NEITHER sharer bound there is no picture to tie the role
    /// to, so it falls back to the identity lock — a description and no `<Picture N>` at all.
    #[test]
    fn a_sharer_on_a_shot_that_binds_neither_is_described_with_no_picture_label() {
        let mut pack = shared_plate_pack();
        pack.references[1].description = "Grey work apron.".to_owned();

        let mut shot = fixture_shot();
        shot.conditioning.mode = "text_to_video".to_owned();
        shot.conditioning.reference_roles = Vec::new();
        shot.conditioning.first_frame_role = None;
        shot.conditioning.last_frame_role = None;
        shot.continuity_roles = vec!["recipient".to_owned()];

        let inserted = inserted_text_for_shot(&shot, &[], &pack);
        assert_eq!(
            inserted.iter().map(|piece| piece.kind).collect::<Vec<_>>(),
            vec![
                InsertedTextKind::ContinuityDescription,
                InsertedTextKind::Audio
            ],
            "nothing is bound, so the lock is the only thing holding this subject: {inserted:?}"
        );
        assert_eq!(inserted[0].text, "Grey work apron.");
        assert!(
            !inserted[0].text.contains("<Picture"),
            "no image is supplied, so no picture may be named: {:?}",
            inserted[0].text
        );
    }

    /// A pack mixing an image-backed role, a DESCRIBED-ONLY one, an unapproved described-only one
    /// and a described-only role with a keyframe plate — everything the identity lock has to tell
    /// apart (sc-24025).
    fn described_pack() -> ReferencePack {
        parse_reference_pack(
            &json!({
                "schemaVersion": crate::film_plan::REFERENCE_PACK_SCHEMA_VERSION,
                "id": "described-refs",
                "version": 1,
                "references": [
                    {
                        "role": "courier", "kind": "character",
                        "file": "references/courier.png",
                        "description": "The courier: blue jacket."
                    },
                    // No file: words alone.
                    {
                        "role": "red_parcel", "kind": "prop",
                        "description": "  The parcel: small,\n  bright red.  "
                    },
                    // A kind that may never be BOUND, to prove the lock does not care about kind.
                    { "role": "house_style", "kind": "style", "description": "Warm light." },
                    // Approved-false: the author's explicit "do not use this".
                    {
                        "role": "draft_look", "kind": "style", "approved": false,
                        "description": "Not approved."
                    },
                    // Declared, but says nothing: an image-backed role may.
                    { "role": "workshop_plate", "kind": "plate", "file": "references/plate.png" }
                ]
            })
            .to_string(),
        )
        .unwrap()
    }

    /// The identity text of one shot, or `None` when the compiler wrote none.
    fn identity_text(shot: &Shot, pack: &ReferencePack) -> Option<String> {
        inserted_text_for_shot(
            shot,
            &shot_reference_pictures(&shot.conditioning.reference_roles, pack),
            pack,
        )
        .into_iter()
        .find(|piece| piece.kind == InsertedTextKind::ContinuityDescription)
        .map(|piece| piece.text)
    }

    /// THE LOCK (sc-24025). A continuity role the shot does NOT bind is described in the prompt,
    /// word for word from the pack; one it DOES bind is not, because its binding sentence already
    /// carries the same description and saying it twice is the one thing this must not do.
    ///
    /// Driven through the mixed fixture, whose two shots are exactly the two cases: SH010 binds
    /// the courier it lists in `continuityRoles`, SH020 binds nothing at all.
    #[test]
    fn an_unbound_continuity_role_is_described_and_a_bound_one_is_not_described_twice() {
        let plan = parse_plan(&mixed_plan_text()).expect("the fixture plan parses");
        let pack = described_pack();
        let courier = "The courier: blue jacket.";

        // SH010 BINDS the courier and lists it in continuityRoles: the binding sentence describes
        // it, and the lock adds nothing.
        let bound = &plan.shots[0];
        assert_eq!(bound.continuity_roles, ["courier"]);
        assert!(
            bound
                .conditioning
                .reference_roles
                .contains(&"courier".to_owned()),
            "this shot must BIND the role it lists, or the test proves nothing"
        );
        assert_eq!(
            identity_text(bound, &pack),
            None,
            "a bound role already has its description in its binding sentence"
        );
        let pictures = shot_reference_pictures(&bound.conditioning.reference_roles, &pack);
        let bindings = reference_binding_text(&pictures).expect("the shot binds a role");
        assert_eq!(
            bindings.text.matches(courier).count(),
            1,
            "exactly once, in the binding sentence: {:?}",
            bindings.text
        );

        // SH020 binds NOTHING and lists two roles: both are described, in the shot's own order.
        let unbound = &plan.shots[1];
        assert_eq!(unbound.continuity_roles, ["courier", "red_parcel"]);
        assert!(unbound.conditioning.reference_roles.is_empty());
        let text = identity_text(unbound, &pack).expect("an unbound continuity role is described");
        assert_eq!(
            text, "The courier: blue jacket. The parcel: small, bright red.",
            "the pack's own words, whitespace normalized, in continuityRoles order"
        );
    }

    /// What the lock leaves out, and why — each of these is a role that would otherwise be
    /// described (sc-24025).
    #[test]
    fn a_keyframe_an_unapproved_and_a_silent_role_contribute_no_identity_text() {
        let plan = parse_plan(&mixed_plan_text()).expect("the fixture plan parses");
        let pack = described_pack();
        let mut shot = plan.shots[1].clone();

        // A role placed as a KEYFRAME is image-bound just as a reference is: its frame is supplied.
        shot.continuity_roles = vec!["workshop_plate".to_owned(), "house_style".to_owned()];
        shot.conditioning.mode = "image_to_video".to_owned();
        shot.conditioning.first_frame_role = Some("workshop_plate".to_owned());
        assert_eq!(
            identity_text(&shot, &pack).as_deref(),
            Some("Warm light."),
            "the keyframe role is supplied as a picture; the style role is not and is a `style`, \
             which the lock describes like any other kind"
        );

        // UNAPPROVED: `approved` defaults to true, so false is the author's explicit "do not use
        // this" — and prompt text shapes a render exactly as conditioning does.
        shot.conditioning.first_frame_role = None;
        shot.conditioning.mode = "text_to_video".to_owned();
        shot.continuity_roles = vec!["draft_look".to_owned()];
        assert_eq!(identity_text(&shot, &pack), None);

        // SILENT: an image-backed role that describes itself with nothing contributes nothing, and
        // is not an error — it still shows its picture wherever it is bound.
        shot.continuity_roles = vec!["workshop_plate".to_owned()];
        assert_eq!(identity_text(&shot, &pack), None);
    }

    /// The lock is BYTE-IDENTICAL across shots: identical input, identical text, which is the
    /// entire point of moving the wording from the author to the compiler (sc-24025).
    #[test]
    fn the_same_role_yields_byte_identical_text_in_every_shot_that_names_it() {
        let plan = parse_plan(&mixed_plan_text()).expect("the fixture plan parses");
        let pack = described_pack();
        let mut first = plan.shots[1].clone();
        first.continuity_roles = vec!["red_parcel".to_owned()];
        let mut second = plan.shots[1].clone();
        second.id = "SH030".to_owned();
        // A different prompt, a different beat, a different audio sentence: everything about the
        // shot differs except the role it locks.
        second.prompt = "a wholly different shot of the same parcel".to_owned();
        second.beat = "reveal".to_owned();
        second.audio = "Distant traffic.".to_owned();
        second.continuity_roles = vec!["red_parcel".to_owned()];

        let text = identity_text(&first, &pack).expect("the role is described");
        assert_eq!(
            identity_text(&second, &pack).as_deref(),
            Some(text.as_str())
        );
        assert_eq!(text, "The parcel: small, bright red.");

        // And a role written into `continuityRoles` TWICE is still described once. Nothing refuses
        // the duplicate, and the lock exists to say one fixed thing about each subject — saying it
        // twice is emphasis a prompt model acts on.
        let mut doubled = first.clone();
        doubled.continuity_roles = vec!["red_parcel".to_owned(), "red_parcel".to_owned()];
        assert_eq!(
            identity_text(&doubled, &pack).as_deref(),
            Some(text.as_str())
        );
    }

    /// A described-only role supplies no image, so it is numbered as none: it must never create a
    /// `<Picture N>`, because the engine is only ever sent the pictures that exist and a phantom
    /// one shifts every later number (sc-24025).
    ///
    /// Only an unvalidated plan can reach this — `validate_plan_against_pack` refuses a
    /// described-only role in every conditioning slot — which is why the compiler is asked
    /// directly here rather than through a plan.
    #[test]
    fn a_described_only_role_is_never_numbered_as_a_picture() {
        let pack = described_pack();
        let roles = [
            "red_parcel".to_owned(),
            "courier".to_owned(),
            "house_style".to_owned(),
        ];
        let pictures = shot_reference_pictures(&roles, &pack);
        assert_eq!(
            pictures.len(),
            1,
            "only the image-backed role is a picture: {pictures:?}"
        );
        assert_eq!(pictures[0].number, 1);
        assert_eq!(pictures[0].dispatch_role(), Some("courier"));
    }

    // -----------------------------------------------------------------------------------------
    // sc-24029 — the feature-end review's findings
    // -----------------------------------------------------------------------------------------

    /// The mixed fixture compiled with `refined` standing in for the refiner's answers.
    fn compile_mixed_with(
        refined: &BTreeMap<String, String>,
    ) -> Result<CompiledPlan, Vec<PlanDiagnostic>> {
        let plan = parse_plan(&mixed_plan_text()).expect("the fixture plan parses");
        let base = entry();
        let reference = reference_entry();
        compile_plan(
            &plan,
            &pack(),
            &CompileInputs {
                entries: &mixed_entries(&base, &reference),
                lane: "mlx",
                plan_sha256: "abc",
                compiled_at: "now",
                refined_prompts: refined,
            },
        )
    }

    fn refined_for(shot_id: &str, text: &str) -> BTreeMap<String, String> {
        [(shot_id.to_owned(), text.to_owned())]
            .into_iter()
            .collect()
    }

    /// sc-24029, E5/E2. The REFINER may not write an engine label into the text the compiler then
    /// builds a prompt around.
    ///
    /// The refiner is handed the model's own prompt guide, which teaches `<Picture N>` as the way
    /// to give a reference a job, and the worker's marker filter deliberately KEEPS such a label —
    /// so a rewrite carrying one reaches the engine as a binding to an image numbered by nothing.
    /// The spelling here is the awkward one on purpose: `< picture 2>` is what the scan exists to
    /// catch and what a `contains("<Picture")` test would miss.
    #[test]
    fn a_refined_prompt_carrying_an_engine_label_is_refused_and_names_the_shot() {
        let findings = compile_mixed_with(&refined_for(
            "SH010",
            "the courier from < picture 2> crosses the room",
        ))
        .expect_err("a refined prompt naming a picture the compiler numbers is refused");
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert_eq!(
            findings[0].shot_id.as_deref(),
            Some("SH010"),
            "{findings:?}"
        );
        assert_eq!(findings[0].field, "prompt", "{findings:?}");
        assert!(
            findings[0].message.contains("< picture 2>")
                && findings[0].message.contains("SH010")
                && findings[0].message.contains("--no-refine"),
            "the refusal names the shot, the label and the remedy: {findings:?}"
        );
    }

    /// sc-24029, E5. The AUTHORED branch stays unscanned: a person who types `<Picture 1>` into a
    /// plan's prompt is writing against an engine whose grammar that is, with the references in
    /// front of them, and `compile` must not refuse them for it. This is the same line
    /// `film_planner::anchoring_findings` draws between a planner DRAFT and a hand-authored
    /// document — pinned here because the refusal above sits in the very next branch.
    #[test]
    fn a_hand_authored_prompt_may_name_a_picture_itself() {
        let mut document: Value = serde_json::from_str(&mixed_plan_text()).unwrap();
        document["shots"][0]["prompt"] =
            json!("the courier from <Picture 1> crosses toward <Picture 2>");
        let plan = parse_plan(&document.to_string()).expect("the plan parses");
        let base = entry();
        let reference = reference_entry();
        let compiled = compile_plan(
            &plan,
            &pack(),
            &CompileInputs {
                entries: &mixed_entries(&base, &reference),
                lane: "mlx",
                plan_sha256: "abc",
                compiled_at: "now",
                refined_prompts: &BTreeMap::new(),
            },
        )
        .expect("a hand-authored label is the author's own and compiles");
        assert!(
            compiled
                .request("SH010")
                .unwrap()
                .prompt
                .contains("<Picture 1>"),
            "the authored label reaches the prompt as written"
        );
    }

    /// sc-24029, E1/E4. `description_sentence` has ONE rule for "already ends a sentence", shared
    /// with the trailing-insertion join: a description closing on `.)`, `."` or `…` is finished,
    /// and adding a second period to it puts a stray `.` into every prompt that repeats it.
    #[test]
    fn a_description_that_already_closes_a_sentence_gains_no_second_period() {
        for finished in [
            "The bench sits under the window (door camera-left.)",
            "The courier trails off\u{2026}",
            "She said \"put it on the bench.\"",
            "Small bright red parcel!",
        ] {
            assert_eq!(
                description_sentence(finished).as_deref(),
                Some(finished),
                "{finished:?} already ends a sentence"
            );
        }
        // And the rule still SUPPLIES one where it is missing.
        assert_eq!(
            description_sentence("Small bright red parcel").as_deref(),
            Some("Small bright red parcel."),
        );
    }

    /// sc-24029, E1/E4. A leading insertion already ends in one space, so an authored prompt that
    /// opens with whitespace — which nothing refuses — used to be joined to the bindings by two.
    #[test]
    fn a_prompt_with_leading_whitespace_is_joined_by_one_space() {
        let shot = fixture_shot();
        let inserted = inserted_text_for_shot(
            &shot,
            &shot_reference_pictures(&shot.conditioning.reference_roles, &pack()),
            &pack(),
        );
        assert!(
            inserted
                .iter()
                .any(|piece| piece.kind == InsertedTextKind::ReferenceBinding),
            "the fixture shot binds references, so something leads its prompt"
        );
        let composed = apply_inserted_text("   a courier enters   ", &inserted);
        assert!(
            !composed.contains("  "),
            "no run of two spaces survives the join: {composed:?}"
        );
        assert_eq!(
            composed,
            apply_inserted_text("a courier enters", &inserted),
            "the surrounding whitespace changes nothing about the dispatched prompt"
        );
        // With nothing inserted at all the prompt is still trimmed, which is what makes the
        // composition reversible for `prompt_differences`.
        assert_eq!(apply_inserted_text("  hello  ", &[]), "hello");
    }

    /// sc-24029, E2/E5. A hand-edited `prompt` is refused even when `insertedText` is pristine.
    ///
    /// `insertedText` records what the compiler WROTE, not where it ended up, and `prompt` is the
    /// field that becomes the job body. Each mutation below leaves `insertedText` untouched and
    /// changes only the dispatched text, in both prompt-source modes.
    #[test]
    fn a_hand_edited_dispatched_prompt_is_refused_in_both_prompt_source_modes() {
        let plan = parse_plan(&mixed_plan_text()).unwrap();
        let base = entry();
        let reference = reference_entry();
        let entries = mixed_entries(&base, &reference);

        // The refined document is produced by a STUB refiner rather than assembled by hand, so
        // what is under test is what a real compile writes.
        let refined_source = compile_mixed_with(&refined_for(
            "SH010",
            "a courier crosses the cluttered room",
        ))
        .expect("a clean rewrite compiles");

        for (label, mut document) in [
            ("authored", compile_mixed_with(&BTreeMap::new()).unwrap()),
            ("refined", refined_source.clone()),
        ] {
            let original = document.request("SH010").unwrap().prompt.clone();
            assert!(
                original.contains("<Picture 1>") && original.contains("<Picture 2>"),
                "{label}: the fixture shot binds two references: {original:?}"
            );

            let mutations = [
                // The bindings deleted: the labels the engine applies are still supplied, but
                // nothing in the text says what they are for.
                (
                    "bindings removed",
                    original
                        .split_once("<Picture 2>")
                        .map(|(_, rest)| rest.trim_start().to_owned())
                        .expect("the binding block leads the prompt"),
                ),
                // The two labels SWAPPED: every sentence is the compiler's own, and the request
                // now binds the courier to the parcel's image and the parcel to the courier's.
                (
                    "picture labels swapped",
                    original
                        .replace("<Picture 1>", "\u{0}")
                        .replace("<Picture 2>", "<Picture 1>")
                        .replace('\u{0}', "<Picture 2>"),
                ),
                // The trailing audio sentence removed: H3 scores a soundtrack from the prompt, so
                // this is a different ask with an identical `insertedText`.
                (
                    "trailing audio removed",
                    original
                        .split_once(AUDIO_PROMPT_PREFIX)
                        .map(|(head, _)| head.trim_end().to_owned())
                        .expect("the audio sentence trails the prompt"),
                ),
            ];

            for (mutation, tampered) in mutations {
                assert_ne!(tampered, original, "{label}/{mutation} changes the prompt");
                let request = document
                    .requests
                    .iter_mut()
                    .find(|request| request.shot_id == "SH010")
                    .unwrap();
                let pristine = request.inserted_text.clone();
                request.prompt = tampered;
                let findings =
                    document.conformance_findings(&plan, &pack(), &entries, ModelLane::Mlx);
                assert!(
                    findings
                        .iter()
                        .any(|finding| finding.field == "compiled.prompt"
                            && finding.shot_id.as_deref() == Some("SH010")
                            && finding
                                .message
                                .contains("is not the prompt the compiler composed")
                            && finding.message.contains("film-harness compile")),
                    "{label}/{mutation} must be refused with the shot, the cause and the remedy: \
                     {findings:?}"
                );
                assert!(
                    !findings
                        .iter()
                        .any(|finding| finding.field == "compiled.insertedText"),
                    "{label}/{mutation}: `insertedText` is untouched, so it must not be blamed: \
                     {findings:?}"
                );
                // Restore for the next mutation.
                let request = document
                    .requests
                    .iter_mut()
                    .find(|request| request.shot_id == "SH010")
                    .unwrap();
                request.prompt = original.clone();
                assert_eq!(request.inserted_text, pristine);
            }

            assert!(
                document
                    .conformance_findings(&plan, &pack(), &entries, ModelLane::Mlx)
                    .is_empty(),
                "{label}: the restored document conforms"
            );
        }
    }

    /// sc-24029, E2. An untouched REFINED document survives serialization and conforms — the
    /// round trip the check above would silently pass by refusing everything.
    ///
    /// Both punctuation cases are exercised because the recovery has to settle the one character
    /// the composition loses: `apply_inserted_text` supplies a `.` before the first trailing piece
    /// when the middle does not already end a sentence, so a rewrite WITHOUT terminal punctuation
    /// and one WITH it produce prompts that differ by a period the compiler wrote.
    #[test]
    fn an_untouched_refined_document_round_trips_and_conforms() {
        let plan = parse_plan(&mixed_plan_text()).unwrap();
        let base = entry();
        let reference = reference_entry();
        let entries = mixed_entries(&base, &reference);
        for rewrite in [
            "a courier crosses the cluttered room", // no terminal punctuation
            "a courier crosses the cluttered room.", // terminal punctuation
            "a courier crosses the cluttered room!",
            "she says \"leave it on the bench.\"",
        ] {
            let compiled = compile_mixed_with(&refined_for("SH010", rewrite))
                .unwrap_or_else(|findings| panic!("{rewrite:?} compiles: {findings:?}"));
            let round_tripped: CompiledPlan =
                serde_json::from_str(&serde_json::to_string(&compiled).unwrap()).unwrap();
            assert_eq!(
                round_tripped.request("SH010").unwrap().prompt_source,
                PromptSource::Refined
            );
            let findings =
                round_tripped.conformance_findings(&plan, &pack(), &entries, ModelLane::Mlx);
            assert!(
                findings.is_empty(),
                "{rewrite:?} is the compiler's own output and must conform: {findings:?}"
            );
            // The recovery is EXACT, not merely accepting: it hands back the rewrite the compiler
            // was given (up to the terminal `.` the composition itself supplies).
            let request = round_tripped.request("SH010").unwrap();
            let middle = recovered_middle(&request.prompt, &request.inserted_text)
                .unwrap_or_else(|| panic!("{rewrite:?} is recoverable"));
            assert!(
                middle == rewrite || middle == format!("{rewrite}."),
                "{rewrite:?} recovered as {middle:?}"
            );
        }
    }

    /// sc-24029, E2. A label hand-written into the REFINED middle of a dispatched prompt is
    /// refused. The compile-time guard never saw this document — it was read back from disk — so
    /// this is the only place it is caught.
    #[test]
    fn an_engine_label_edited_into_a_refined_middle_is_refused() {
        let plan = parse_plan(&mixed_plan_text()).unwrap();
        let base = entry();
        let reference = reference_entry();
        let entries = mixed_entries(&base, &reference);
        let mut compiled =
            compile_mixed_with(&refined_for("SH010", "a courier crosses the room")).unwrap();
        let request = compiled
            .requests
            .iter_mut()
            .find(|request| request.shot_id == "SH010")
            .unwrap();
        request.prompt = request.prompt.replace(
            "a courier crosses the room",
            "the courier from <Picture 2> crosses",
        );
        let findings = compiled.conformance_findings(&plan, &pack(), &entries, ModelLane::Mlx);
        assert!(
            findings
                .iter()
                .any(|finding| finding.field == "compiled.prompt"
                    && finding.message.contains("<Picture 2>")
                    && finding.message.contains("assigns itself")),
            "{findings:?}"
        );
    }

    /// sc-24029, E5/E7. A compiled document is stale once the PACK changes, and the refusal names
    /// the pack rather than a derived field.
    #[test]
    fn editing_a_pack_description_stales_a_compiled_document() {
        let plan = parse_plan(&mixed_plan_text()).unwrap();
        let compiled = compile_mixed_with(&BTreeMap::new()).unwrap();
        assert!(
            compiled
                .staleness_findings(&plan, "abc", &pack())
                .is_empty(),
            "the pack it was compiled from is current"
        );

        let mut edited = pack();
        edited
            .references
            .iter_mut()
            .find(|entry| entry.role == "courier")
            .unwrap()
            .description = "The courier: blue quilted jacket.".to_owned();
        let findings = compiled.staleness_findings(&plan, "abc", &edited);
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert_eq!(
            findings[0].field, "compiled.referencePackSha256",
            "{findings:?}"
        );
        assert_eq!(
            findings[0].message,
            "the reference pack changed since these requests were compiled; recompile, or use \
             authored prompts",
            "{findings:?}"
        );
    }

    /// sc-24029, E5/E7. The identity is the PARSED pack, so the CLI's JSONC document and the
    /// workspace's typed draft agree — and a comment- or whitespace-only edit of the document is
    /// not a pack change.
    #[test]
    fn the_pack_identity_ignores_comments_and_document_whitespace() {
        let commented = parse_reference_pack(
            r#"{
                // The same pack, with an author's notes in it.
                "schemaVersion": 2,
                "id": "courier-refs",
                "version": 3,
                "references": [
                    /* the courier */
                    { "role": "courier",        "kind": "character", "file": "references/courier.png" },
                    { "role": "red_parcel",     "kind": "prop",      "file": "references/red_parcel.png" },
                    { "role": "workshop_plate", "kind": "plate",     "file": "references/plate.png" }
                ]
            }"#,
        )
        .expect("the commented document parses");
        assert_eq!(
            reference_pack_sha256(&commented).unwrap(),
            reference_pack_sha256(&pack()).unwrap(),
            "comments and spacing are the document's, not the pack's"
        );
    }

    /// sc-24029, E5/E7. The `insertedText` difference reaches an operator, so it is rendered as
    /// per-kind plain text rather than a `Vec<InsertedText>` struct dump.
    #[test]
    fn an_inserted_text_difference_reads_as_plain_text_not_a_debug_dump() {
        let plan = parse_plan(&mixed_plan_text()).unwrap();
        let base = entry();
        let reference = reference_entry();
        let entries = mixed_entries(&base, &reference);
        let mut compiled = compile_mixed_with(&BTreeMap::new()).unwrap();
        let request = compiled
            .requests
            .iter_mut()
            .find(|request| request.shot_id == "SH010")
            .unwrap();
        request
            .inserted_text
            .retain(|piece| piece.kind != InsertedTextKind::Audio);
        let findings = compiled.conformance_findings(&plan, &pack(), &entries, ModelLane::Mlx);
        let finding = findings
            .iter()
            .find(|finding| finding.field == "compiled.insertedText")
            .unwrap_or_else(|| panic!("{findings:?}"));
        assert!(
            !finding.message.contains("InsertedText {") && !finding.message.contains("kind:"),
            "no Rust struct notation reaches an operator: {:?}",
            finding.message
        );
        assert!(
            finding.message.contains(InsertedTextKind::Audio.label()),
            "the missing piece is named by what it IS: {:?}",
            finding.message
        );
    }
}
