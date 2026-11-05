//! Durable run/action provenance and deterministic fixture media.

use super::*;
/// The latest controller action is separate from the film verdict: a refused resume or
/// replacement must not rewrite the prior outcome, spend an attempt, or lose an owned job.
pub(super) const ACTION_OPERATION_FILE: &str = "action-operation.json";

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ActionOperation {
    pub action: String,
    pub shot_id: Option<String>,
    pub status: String,
    pub detail: Option<String>,
    pub started_at: String,
    pub finished_at: Option<String>,
}

pub fn read_action_operation(directory: &Path) -> Result<Option<ActionOperation>, HarnessError> {
    match std::fs::read(directory.join(ACTION_OPERATION_FILE)) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .map(Some)
            .map_err(|error| HarnessError::Io(error.to_string())),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

/// The caller retains its controller lease until this receipt is durable, including early
/// document/transport/admission failures. An interrupted receipt never authorizes redispatch;
/// only the run's existing resume and job reconciliation rules do that.
pub(crate) async fn record_action<T>(
    directory: &Path,
    action: &str,
    shot_id: Option<&str>,
    operation: impl Future<Output = Result<T, HarnessError>>,
) -> Result<T, HarnessError> {
    let mut receipt = ActionOperation {
        action: action.to_owned(),
        shot_id: shot_id.map(str::to_owned),
        status: "running".to_owned(),
        detail: None,
        started_at: utc_now(),
        finished_at: None,
    };
    let persist = |receipt: &ActionOperation| {
        let bytes = serde_json::to_vec_pretty(receipt)
            .map_err(|error| HarnessError::Io(error.to_string()))?;
        write_atomically(&directory.join(ACTION_OPERATION_FILE), &bytes)
    };
    persist(&receipt)?;
    let result = operation.await;
    receipt.status = if matches!(&result, Err(HarnessError::Canceled(_))) {
        "canceled"
    } else if result.is_ok() {
        "completed"
    } else {
        "failed"
    }
    .to_owned();
    receipt.detail = result.as_ref().err().map(ToString::to_string);
    receipt.finished_at = Some(utc_now());
    persist(&receipt)?;
    result
}

/// Tag every harness-imported APPROVED reference carries beside its role tag.
pub(super) const REFERENCE_TAG: &str = "film-harness-reference";

/// Tag an imported reference the pack has NOT approved carries instead, so a query for the
/// conditioning-eligible references cannot pick it up.
pub(super) const UNAPPROVED_REFERENCE_TAG: &str = "film-harness-reference-unapproved";

/// A harness-owned asset facet that fits the project store's public 40-byte tag contract.
///
/// Short role and pack tags keep their established spelling for CLI/API interoperability. Plan
/// ids and roles are valid up to 64 ASCII characters, though, so their prefixed form does not
/// always fit in an asset tag. In that case the tag carries a namespaced 128-bit digest while the
/// exact value remains in `extra.filmHarness`, which is also the authoritative adoption identity.
pub(super) fn harness_asset_tag(namespace: &str, value: &str) -> String {
    let literal = format!("{namespace}:{value}");
    if literal.len() <= 40 {
        return literal;
    }
    let digest = sha256_hex(literal.as_bytes());
    format!("{namespace}:h:{}", &digest[..32])
}

/// File name of the run record inside the run directory.
pub const RUN_RECORD_FILE: &str = "run.json";

/// Read the run record in `run_dir` without touching the API. What `film-harness status` prints and
/// what `resume` / `replace-take` start from.
pub fn read_run_record(run_dir: &Path) -> Result<RunRecord, HarnessError> {
    let path = run_dir.join(RUN_RECORD_FILE);
    let text = std::fs::read_to_string(&path).map_err(|error| {
        HarnessError::Refused(format!("cannot read {}: {error}", path.display()))
    })?;
    serde_json::from_str(&text).map_err(|error| {
        HarnessError::Refused(format!("{} is not a run record: {error}", path.display()))
    })
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    sceneworks_core::hashing::sha256_bytes(bytes)
}

/// The key one attempt of one shot dispatches under. Stable across restarts because every part of
/// it is: the run id is in the record, the shot id is in the plan, and attempt numbers never repeat
/// within a shot (sc-22711).
///
/// It is stamped into the dispatched body's `advanced.filmHarness` block (sc-22713's
/// [`CompiledRequest::to_job_body_with`] carries it through [`DispatchContext`]), which is what lets
/// a controller that died between the POST and the record write find its OWN job instead of
/// enqueuing a second render for the same attempt.
pub fn idempotency_key(run_id: &str, shot_id: &str, attempt: u32) -> String {
    format!("{run_id}:{shot_id}:a{attempt}")
}

/// The key one dialogue synthesis dispatches under (sc-23404).
///
/// Keyed on the CONTENT as well as the role — model, voice and the trimmed line — rather than on
/// the role alone. A role-only key would be stable across restarts too, but it would also make
/// re-casting a line (a new voice, a rewritten line) adopt the job that spoke the OLD one, and the
/// film would quietly keep saying the wrong thing.
///
/// `attempt` is the retry axis, the render key's `a{n}` under a different name: a synthesis that
/// FAILED must be re-dispatched by the next resume, and a key without it would find the failed job
/// and re-read the same failure forever. Every part is stable across restarts — the run id is in
/// the record, the content is in the pack, and the attempt is in the record.
pub fn dialogue_idempotency_key(
    run_id: &str,
    role: &str,
    model: &str,
    voice: Option<&str>,
    text: &str,
    attempt: u32,
) -> String {
    let digest =
        sha256_hex(format!("{model}\n{}\n{}", voice.unwrap_or(""), text.trim()).as_bytes());
    format!("{run_id}:sound:{role}:{}:a{attempt}", &digest[..12])
}

/// Where a synthesized line's WAV lands in the pack directory when the entry pins no `file`.
///
/// Deterministic in the role and the line, so the same pack run twice writes the same name and a
/// resume finds the clip it wrote — and so a changed line is a different file rather than a silent
/// overwrite of the one the last export used.
pub fn synthesized_sound_file(role: &str, text_sha256: &str) -> String {
    let digest: String = text_sha256.chars().take(12).collect();
    format!("sound/{role}.tts-{digest}.wav")
}

/// The `type: audio` asset one completed `audio_generate` job produced: its id and its
/// project-relative media path.
///
/// Read off the job result the API rewrote from the worker's `assetWrites` (the same `assets` block
/// [`take_from_result`] reads a render out of), not off the worker's own fact — the rewrite is what
/// says the asset is actually persisted.
pub(super) fn audio_asset_from_result(result: &Value) -> Option<(String, String)> {
    let asset = result.get("assets")?.as_array()?.first()?;
    let id = asset.get("id")?.as_str()?.to_owned();
    let path = asset.pointer("/file/path")?.as_str()?.to_owned();
    (!path.is_empty()).then_some((id, path))
}

pub(super) fn take_from_result(
    result: &Value,
    model: &str,
    backend: Option<&str>,
) -> Option<TakeRecord> {
    let asset = result.get("assets")?.as_array()?.first()?;
    let file = asset.get("file").unwrap_or(&Value::Null);
    let recipe = asset.get("recipe").unwrap_or(&Value::Null);
    Some(TakeRecord {
        asset_id: asset.get("id")?.as_str()?.to_owned(),
        media_path: file
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        encoded_duration_seconds: file.get("duration").and_then(Value::as_f64),
        encoded_fps: file.get("fps").and_then(Value::as_f64),
        encoded_frame_count: file.get("frameCount").and_then(Value::as_u64),
        has_audio: file.get("hasAudio").and_then(Value::as_bool),
        seed: recipe.get("seed").and_then(Value::as_i64),
        adapter: recipe
            .get("adapter")
            .and_then(Value::as_str)
            .map(str::to_owned),
        backend: backend.map(str::to_owned),
        model: recipe
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or(model)
            .to_owned(),
        raw_adapter_settings: recipe
            .get("rawAdapterSettings")
            .cloned()
            .unwrap_or(Value::Null),
    })
}

pub(super) fn seconds_since(start: Instant) -> f64 {
    start.elapsed().as_secs_f64()
}

/// The audio layers a `timeline_export` result says it mixed WITHOUT (`droppedAudioLayers`, one
/// object per placed clip whose asset, file or audio stream was missing — sc-22715), copied into
/// the run record's export entry verbatim. Empty for a result that reports none.
pub(super) fn dropped_audio_layers(result: &Value) -> Vec<Value> {
    result
        .get("droppedAudioLayers")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

/// Write `record` as `run.json` under `out_dir` and, when the project directory is reachable on
/// this filesystem, as `<project>/film-harness/<run_id>/run.json` too. Source documents are copied
/// beside the out-dir record so the run is self-describing.
pub(super) fn persist_record(
    record: &RunRecord,
    out_dir: &Path,
    plan_path: &Path,
    pack_path: &Path,
) -> Result<PathBuf, HarnessError> {
    std::fs::create_dir_all(out_dir)?;
    let json = serde_json::to_string_pretty(&record.to_json())
        .map_err(|error| HarnessError::Io(error.to_string()))?;
    let record_path = out_dir.join(RUN_RECORD_FILE);
    write_atomically(&record_path, json.as_bytes())?;
    // Written once and then left alone, so an edit to the source plan mid-run cannot rewrite the
    // copy the run was actually started from — which is what a resume re-hashes (sc-22711).
    for (source, copy) in [
        (plan_path, out_dir.join("plan.json")),
        (pack_path, out_dir.join("references.json")),
    ] {
        if !copy.exists() {
            if let Ok(text) = std::fs::read(source) {
                write_atomically(&copy, &text)?;
            }
        }
    }
    // The compiled requests travel with the record too: without them the run record says which
    // prompts were dispatched only by reference.
    if let Some(compiled) = record.compiled.as_ref() {
        if let Ok(text) = std::fs::read(&compiled.path) {
            std::fs::write(out_dir.join("compiled.json"), text)?;
        }
    }
    if let Some(project_path) = record.project_path.as_deref().map(Path::new) {
        // A Film workspace run already lives under its project. In that case `record_path` is the
        // canonical run record the locator names, so do not create a second mutable mirror.
        if project_path.is_dir() && !record_path.starts_with(project_path) {
            let project_record_dir = project_path.join("film-harness").join(&record.run_id);
            std::fs::create_dir_all(&project_record_dir)?;
            write_atomically(&project_record_dir.join(RUN_RECORD_FILE), json.as_bytes())?;
        }
    }
    Ok(record_path)
}

/// Write `bytes` to `path` through a sibling temp file and a rename.
///
/// sc-22711 rewrites the record at every state transition rather than once at the end, so a
/// controller killed during a write must leave the PREVIOUS record intact rather than a truncated
/// file no resume can parse.
///
/// The shared writer syncs both staged bytes and the published directory entry.
pub(super) fn write_atomically(path: &Path, bytes: &[u8]) -> Result<(), HarnessError> {
    Ok(sceneworks_core::fsutil::write_atomic(path, bytes)?)
}

/// The source document at `original`, falling back to the copy the run kept beside its record.
pub(super) fn read_source(
    original: &Path,
    fallback: &Path,
    label: &str,
) -> Result<(PathBuf, Vec<u8>), HarnessError> {
    if let Ok(bytes) = std::fs::read(original) {
        return Ok((original.to_path_buf(), bytes));
    }
    let bytes = std::fs::read(fallback).map_err(|error| {
        HarnessError::Refused(format!(
            "cannot read the {label} from {} or {}: {error}",
            original.display(),
            fallback.display()
        ))
    })?;
    Ok((fallback.to_path_buf(), bytes))
}

/// The timeline items as the SAVE persisted them, in the order the run assembled them, or `None`
/// when the saved document does not hold exactly the items that were sent (which would mean the
/// record and the project disagree about what the sequence is).
/// Every picture item the run assembled, as the SAVED document holds it — or `None` if the save
/// does not hold exactly them.
///
/// Scoped to the picture track since sc-22712: the sequence now also carries dialogue and bed
/// tracks, so "the saved item count equals the intended count" is only a true statement about the
/// track the shots live on. The guarantee is unchanged — the run record cannot describe a sequence
/// the project does not hold.
#[cfg(test)]
pub(super) fn persisted_picture_items<'a>(
    saved: &'a Value,
    track_id: &str,
    intended_item_ids: &[String],
) -> Option<Vec<&'a Value>> {
    let items = saved
        .get("tracks")
        .and_then(Value::as_array)?
        .iter()
        .find(|track| track.get("id").and_then(Value::as_str) == Some(track_id))
        .and_then(|track| track.get("items").and_then(Value::as_array))?;
    let mut persisted = Vec::with_capacity(intended_item_ids.len());
    for item_id in intended_item_ids {
        persisted.push(
            items
                .iter()
                .find(|item| item.get("id").and_then(Value::as_str) == Some(item_id.as_str()))?,
        );
    }
    (persisted.len() == items.len()).then_some(persisted)
}

pub(super) fn base_record(
    run_id: &str,
    plan: &ProductionPlan,
    pack: &ReferencePack,
    options: &RunOptions,
    plan_bytes: &[u8],
    pack_bytes: &[u8],
) -> RunRecord {
    let selected_shot_ids = match &options.shot_ids {
        Some(ids) => plan
            .shots
            .iter()
            .filter(|shot| ids.contains(&shot.id))
            .map(|shot| shot.id.clone())
            .collect(),
        None => plan.shots.iter().map(|shot| shot.id.clone()).collect(),
    };
    RunRecord {
        schema_version: RUN_RECORD_SCHEMA_VERSION,
        run_id: run_id.to_owned(),
        created_at: utc_now(),
        finished_at: None,
        // A record is born `running`: it exists before the first API write, and only `finish`
        // decides what it became (sc-22711).
        state: RunState::Running,
        outcome: RunOutcome::Failed,
        stop: None,
        plan: SourceDocument {
            id: plan.id.clone(),
            version: plan.version,
            path: options.plan_path.display().to_string(),
            sha256: sha256_hex(plan_bytes),
        },
        reference_pack: SourceDocument {
            id: pack.id.clone(),
            version: pack.version,
            path: options.reference_pack_path.display().to_string(),
            sha256: sha256_hex(pack_bytes),
        },
        compiled: None,
        project_id: options.project_id.clone(),
        project_path: None,
        model: None,
        limits: plan.limits.clone(),
        selected_shot_ids,
        references: Vec::new(),
        sound: Vec::new(),
        synthesized_sound: Vec::new(),
        shots: Vec::new(),
        timeline: None,
        export: None,
        export_pending: None,
        superseded_export_job_ids: Vec::new(),
        diagnostics: Vec::new(),
        decisions: Vec::new(),
        elapsed_seconds: 0.0,
        human_requested_elapsed_seconds: 0.0,
        active_take_operation: None,
    }
}

/// A `rejected` record for a plan that failed document-level validation — the documents may not
/// even have parsed, so ids and limits are read leniently from the raw JSON.
pub(super) fn rejected_record(
    run_id: &str,
    options: &RunOptions,
    plan_bytes: &[u8],
    pack_bytes: &[u8],
    findings: Vec<PlanDiagnostic>,
    started: Instant,
) -> RunRecord {
    let lenient = |bytes: &[u8]| -> Value {
        std::str::from_utf8(bytes)
            .ok()
            .and_then(|text| {
                serde_json::from_str::<Value>(&sceneworks_core::jsonc::strip_jsonc_comments(text))
                    .ok()
            })
            .unwrap_or(Value::Null)
    };
    let plan_value = lenient(plan_bytes);
    let pack_value = lenient(pack_bytes);
    let string_at = |value: &Value, key: &str| {
        value
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned()
    };
    let version_at = |value: &Value| {
        value
            .get("version")
            .and_then(Value::as_u64)
            .and_then(|version| u32::try_from(version).ok())
            .unwrap_or(0)
    };
    let limits = plan_value
        .get("limits")
        .cloned()
        .and_then(|limits| serde_json::from_value(limits).ok())
        .unwrap_or(film_plan::PlanLimits {
            max_run_seconds: 0,
            max_shot_seconds: 0,
            max_attempts_per_shot: 0,
            max_memory_gb: 0.0,
            planner_max_memory_gb: None,
        });
    RunRecord {
        schema_version: RUN_RECORD_SCHEMA_VERSION,
        run_id: run_id.to_owned(),
        created_at: utc_now(),
        finished_at: Some(utc_now()),
        state: RunState::Finished,
        outcome: RunOutcome::Rejected,
        stop: Some(RunStop {
            reason: "rejected".to_owned(),
            detail: "the plan was refused before anything was created; fix the findings and start \
                     a new run"
                .to_owned(),
            resumable: false,
        }),
        plan: SourceDocument {
            id: string_at(&plan_value, "id"),
            version: version_at(&plan_value),
            path: options.plan_path.display().to_string(),
            sha256: sha256_hex(plan_bytes),
        },
        reference_pack: SourceDocument {
            id: string_at(&pack_value, "id"),
            version: version_at(&pack_value),
            path: options.reference_pack_path.display().to_string(),
            sha256: sha256_hex(pack_bytes),
        },
        compiled: None,
        project_id: options.project_id.clone(),
        project_path: None,
        model: None,
        limits,
        selected_shot_ids: options.shot_ids.clone().unwrap_or_default(),
        references: Vec::new(),
        sound: Vec::new(),
        synthesized_sound: Vec::new(),
        shots: Vec::new(),
        timeline: None,
        export: None,
        export_pending: None,
        superseded_export_job_ids: Vec::new(),
        diagnostics: findings,
        decisions: Vec::new(),
        elapsed_seconds: seconds_since(started),
        human_requested_elapsed_seconds: 0.0,
        active_take_operation: None,
    }
}

// ---------------------------------------------------------------------------------------------
// Fixture images
// ---------------------------------------------------------------------------------------------

/// The courier/workshop/red-parcel reference roles and the flat colour each placeholder plate is
/// painted with. Deterministic, so the checked-in PNGs under
/// `config/film-harness/courier-workshop/references/` are reproducible byte for byte and a test can
/// prove it.
pub const FIXTURE_REFERENCES: &[(&str, [u8; 3])] = &[
    ("courier", [64, 80, 120]),
    ("recipient", [120, 96, 64]),
    ("red_parcel", [200, 32, 32]),
    ("workshop_location", [96, 88, 72]),
    ("workbench_table", [140, 110, 70]),
    ("house_style", [40, 40, 48]),
    ("workshop_plate", [84, 76, 64]),
];

/// Width and height of every fixture plate: MiniMax-H3's cheapest declared canvas, so a keyframe
/// plate needs no resampling.
pub const FIXTURE_PLATE_SIZE: (u32, u32) = (576, 320);

/// Paint one deterministic placeholder plate: a flat field with a lighter horizontal band (a
/// "table line") and a darker margin, so the plates are distinguishable at a glance and give an
/// image-conditioned model something other than a single colour.
pub fn fixture_plate_png(role: &str, rgb: [u8; 3]) -> Result<Vec<u8>, HarnessError> {
    let (width, height) = FIXTURE_PLATE_SIZE;
    fixture_plate_png_sized(role, rgb, width, height)
}

/// The same deterministic plate at an arbitrary canvas. The fixture plates are
/// [`FIXTURE_PLATE_SIZE`]; a caller that must honour a REQUESTED geometry — the scripted image
/// worker the sc-23403 tests drive, which answers `make-references` at the size the job asked for
/// — names its own.
pub fn fixture_plate_png_sized(
    role: &str,
    rgb: [u8; 3],
    width: u32,
    height: u32,
) -> Result<Vec<u8>, HarnessError> {
    let (width, height) = (width.max(16), height.max(16));
    let mut image = image::RgbImage::new(width, height);
    let seed = role.bytes().fold(7_u32, |acc, byte| {
        acc.wrapping_mul(31).wrapping_add(u32::from(byte))
    });
    let band_top = height * 3 / 5;
    let band_bottom = band_top + height / 12;
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        let mut color = rgb;
        if x < 16 || x >= width - 16 || y < 16 || y >= height - 16 {
            color = color.map(|channel| channel.saturating_sub(24));
        } else if (band_top..band_bottom).contains(&y) {
            color = color.map(|channel| channel.saturating_add(40));
        } else if (x.wrapping_mul(7).wrapping_add(y.wrapping_mul(13)) ^ seed) % 97 == 0 {
            color = color.map(|channel| channel.saturating_add(12));
        }
        *pixel = image::Rgb(color);
    }
    let mut bytes = Vec::new();
    // Fast selects png's FdeflateUltraFast path, not flate2. A fixed Sub filter keeps
    // flat plates compact (NoFilter stores nearly raw pixels and exceeds the planner
    // fixture's upload budget), while remaining byte-identical across backend unions.
    let encoder = image::codecs::png::PngEncoder::new_with_quality(
        &mut bytes,
        image::codecs::png::CompressionType::Fast,
        image::codecs::png::FilterType::Sub,
    );
    image::ImageEncoder::write_image(
        encoder,
        image.as_raw(),
        width,
        height,
        image::ExtendedColorType::Rgb8,
    )
    .map_err(|error| HarnessError::Io(error.to_string()))?;
    Ok(bytes)
}

/// Write every fixture plate as `<out_dir>/<role>.png`; returns the written paths.
pub fn write_fixture_images(out_dir: &Path) -> Result<Vec<PathBuf>, HarnessError> {
    std::fs::create_dir_all(out_dir)?;
    let mut written = Vec::new();
    for (role, rgb) in FIXTURE_REFERENCES {
        let path = out_dir.join(format!("{role}.png"));
        std::fs::write(&path, fixture_plate_png(role, *rgb)?)?;
        written.push(path);
    }
    Ok(written)
}

/// Placeholder BED sound for the fixture pack (sc-22712): `(role, seconds, hz, amplitude)`.
///
/// The two beds are long enough to play under the WHOLE six-shot sequence (6 x 5.1667s ~= 31s)
/// without running out, because a bed that stops partway would make the one thing this fixture is
/// meant to demonstrate — continuous sound across intentional cuts — unobservable.
///
/// The fixture's three DIALOGUE roles are NOT here (sc-23404): they carry `text` and are spoken by
/// the run through the audio route, so there is no placeholder tone left to write for them. The
/// tones were placeholders for exactly this, and a "with-dialogue" export that carried a 400 Hz
/// beep instead of a line was the thing they were standing in for.
pub const FIXTURE_SOUNDS: &[(&str, f64, u32, i16)] = &[
    ("workshop_room_tone", 32.0, 100, 2600),
    ("main_theme", 32.0, 250, 3600),
];

/// Sample rate of every fixture clip. Low on purpose: import transcodes each one to 48 kHz PCM-16
/// anyway, and a placeholder tone gains nothing from being stored at the higher rate.
pub const FIXTURE_SOUND_RATE: u32 = 8_000;

/// Write one deterministic placeholder clip: a mono PCM-16 triangle wave.
///
/// **Every sample is integer arithmetic, with no floating point anywhere.** A sine would be the
/// obvious choice and would be wrong: `f64::sin` may differ by an ULP between platforms, which is
/// enough to change a rounded `i16` and break the byte-for-byte check on the checked-in fixture.
/// A triangle is exactly reproducible on every host, and is far gentler to listen to than the
/// square wave that would be the other integer option.
pub fn fixture_sound_wav(seconds: f64, hz: u32, amplitude: i16) -> Vec<u8> {
    let rate = FIXTURE_SOUND_RATE;
    let frames = ((seconds.max(0.0) * f64::from(rate)) as u32).max(1);
    let period = (rate / hz.max(1)).max(2) as i32;
    let half = period / 2;
    let amplitude = i32::from(amplitude);
    let mut samples = Vec::with_capacity(frames as usize * 2);
    for index in 0..frames as i32 {
        let phase = index % period;
        let ramp = if phase < half { phase } else { period - phase };
        let value = (amplitude * (2 * ramp - half)) / half.max(1);
        samples.extend_from_slice(&(value as i16).to_le_bytes());
    }
    let data_len = samples.len() as u32;
    let mut wav = Vec::with_capacity(samples.len() + 44);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16_u32.to_le_bytes());
    wav.extend_from_slice(&1_u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1_u16.to_le_bytes()); // mono
    wav.extend_from_slice(&rate.to_le_bytes());
    wav.extend_from_slice(&(rate * 2).to_le_bytes()); // byte rate
    wav.extend_from_slice(&2_u16.to_le_bytes()); // block align
    wav.extend_from_slice(&16_u16.to_le_bytes()); // bits per sample
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    wav.extend_from_slice(&samples);
    wav
}

/// Write every fixture clip into `out_dir` as `<role>.wav`.
pub fn write_fixture_sound(out_dir: &Path) -> Result<Vec<PathBuf>, HarnessError> {
    std::fs::create_dir_all(out_dir)?;
    let mut written = Vec::new();
    for (role, seconds, hz, amplitude) in FIXTURE_SOUNDS {
        let path = out_dir.join(format!("{role}.wav"));
        std::fs::write(&path, fixture_sound_wav(*seconds, *hz, *amplitude))?;
        written.push(path);
    }
    Ok(written)
}
