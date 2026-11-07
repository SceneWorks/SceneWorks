//! Neutral declared sampling-step constraints shared by request lanes.

use crate::contracts::JsonObject;
use serde_json::Value;

/// `limits.hardMinSteps` from a resolved manifest entry, or `None` for **no floor**.
///
/// The sampling-axis sibling of [`crate::video_request::hard_min_duration`], added by sc-19426 because the catalog had no
/// way to say "this engine REFUSES below N denoise steps" at all — the constraint could only be
/// written as a manifest comment, which nothing reads.
///
/// MiniMax-H3 is the case that forced it, and its floor is a **product judgement, not an engine
/// limit** (the corrected sc-18726 account, carried by the manifest's `hardMinSteps` comment).
/// The unit everywhere in SceneWorks is MODEL EVALUATIONS (NFE): the engine builds the sigma grid
/// as `linspace(1, 0, steps + 1)`, appending the terminal `0` itself
/// (`JointSchedule::with_shifts(evaluations + 1, ..)`), so a `steps: 1` request is a legal
/// 2-point grid that renders — as noise, because one evaluation is a single Euler jump from pure
/// randomness; 2 is the smallest schedule with an intermediate sigma. This doc previously carried
/// the retracted "`steps` points drive `steps - 1` evaluations, so `steps: 1` is unrenderable"
/// account — that describes diffusers' `num_inference_steps` grid-point convention, which is NOT
/// what this key counts. Pinned by
/// `minimax_h3_turbo_steps_are_model_evaluations_not_grid_points` (this crate). Either way the
/// floor is REFUSED rather than quietly raised, the same reject-never-clamp posture the duration
/// bounds take.
///
/// Absent ⇒ no floor, so every model that declares nothing is byte-identical to before this key had
/// a reader — the same never-block-without-evidence posture [`crate::video_request::hard_min_duration`] takes, and the
/// same typo'd-manifest tolerance: a zero, negative, fractional or non-numeric floor is not a
/// constraint anyone could satisfy, so it falls back to no floor rather than bricking the model.
///
/// ⚠️ A **declared** floor is not automatically a **read** one: this reads `as_u64`, so an authored
/// `2.0` parses as a float and lands back on `None` even though JSON Schema's `integer` accepts it.
/// `shipped_manifest_step_floors_admit_everything_they_advertise` compares the reader's answer
/// against a transcription of the manifest bytes for exactly that reason.
pub fn hard_min_steps(model_manifest_entry: &JsonObject) -> Option<u32> {
    model_manifest_entry
        .get("limits")
        .and_then(Value::as_object)
        .and_then(|limits| limits.get("hardMinSteps"))
        .and_then(Value::as_u64)
        .filter(|floor| *floor > 0)
        .and_then(|floor| u32::try_from(floor).ok())
}

/// `limits.hardMaxSteps` — the model's hard CEILING on `advanced.steps`, or `None` for **no
/// declared ceiling** (sc-24215). The floor's sibling: a request above it is REFUSED (API enqueue
/// and, through each model's own boundary, the worker) rather than clamped. Absent ⇒ every model
/// keeps exactly its old behaviour (the image lane's historical clamp). Same typo tolerance as
/// [`hard_min_steps`]: a zero or non-integer ceiling is not a constraint and reads as `None`.
pub fn hard_max_steps(model_manifest_entry: &JsonObject) -> Option<u32> {
    model_manifest_entry
        .get("limits")
        .and_then(Value::as_object)
        .and_then(|limits| limits.get("hardMaxSteps"))
        .and_then(Value::as_u64)
        .filter(|ceiling| *ceiling > 0)
        .and_then(|ceiling| u32::try_from(ceiling).ok())
}

/// `limits.steps` — the discrete set of denoise step counts a model advertises — or `None` for
/// **no menu**, meaning any count the model's other bounds allow.
///
/// The **exact-value** sibling of [`hard_min_steps`], added by sc-19502 because a floor could only
/// express half of LTX-2.3's constraint. LTX is distilled: its σ waypoints are baked into training,
/// so `candle-gen-ltx` runs a fixed 8-step stage-1 schedule and can honor *no other count*.
/// `hardMinSteps: 8` would have said "at least 8" and wrongly admitted `steps: 30`, which the engine
/// refuses — so the shape had to be a SET, membership-tested like [`crate::video_request::allowed_fps`], not a `<`
/// comparison. `limits.steps: [8]` says exactly 8 and generalizes to a family with two baked
/// schedules without becoming a range.
///
/// # Why one lane-agnostic key is legitimate here
///
/// sc-19502 filed this as possibly needing a **per-lane** block, because the two backends disagreed:
/// candle rejected `steps: 30` while `mlx-gen-ltx` never read `req.steps` at all and silently
/// rendered its baked schedule anyway. Reading both engines settled it — the divergence was a lane
/// **defect**, not a real difference. Both lanes bake a byte-identical `STAGE1_SIGMAS` table (the
/// same 9 σ waypoints ⇒ the same 8 steps); mlx had simply never enforced it. The inference-side
/// half of sc-19502 moved the constraint onto the shared `Capabilities::supported_steps` seam that
/// both lanes' `validate` already calls, so they now refuse identically. A per-lane key would have
/// encoded the bug in the schema and made it permanent.
///
/// Absent ⇒ no menu, so a model that declares nothing is byte-identical to before this key had a
/// reader — the same never-block-without-evidence posture [`crate::video_request::allowed_fps`] and [`hard_min_steps`]
/// take. An array with no usable entry (empty, or every value non-integer / zero) is likewise **no
/// menu**: a menu nothing can satisfy would reject every request including the model's own
/// `defaults.steps`, which is a typo'd manifest bricking a model rather than a constraint. Zero is
/// filtered because every engine shares gen-core's `steps >= 1` rejection, so a declared `0` is not
/// a count anyone could render.
///
/// ⚠️ Reads `as_u64`, so an authored `8.0` parses as a float and lands back on `None` even though
/// JSON Schema's `integer` accepts it — a declaration with no reader.
/// `shipped_manifest_step_floors_admit_everything_they_advertise` compares this against a
/// transcription of the manifest bytes for exactly that reason — it covers both step axes.
pub fn allowed_steps(model_manifest_entry: &JsonObject) -> Option<Vec<u32>> {
    let menu: Vec<u32> = model_manifest_entry
        .get("limits")
        .and_then(Value::as_object)
        .and_then(|limits| limits.get("steps"))
        .and_then(Value::as_array)?
        .iter()
        .filter_map(Value::as_u64)
        .filter(|steps| *steps > 0)
        .filter_map(|steps| u32::try_from(steps).ok())
        .collect();
    (!menu.is_empty()).then_some(menu)
}

/// The denoise step count a video payload actually renders at — `advanced.steps` as the caller named
/// it — or `None` when they named none, in which case the engine picks and there is nothing to
/// judge.
///
/// Deliberately reads the SAME two shapes, with the same width arithmetic, as the worker's own
/// `video_jobs::wan::advanced_opt_u32` — a JSON number or a numeric string, narrowed with `as u32`.
/// A gate that reads fewer shapes than the consumer is a gate with a hole: `"steps": "1"` would slip
/// past a number-only gate and still reach the engine as 1, and `steps: 4294967297` would slip past
/// a checked-conversion gate and reach it as 1 as well, which is the exact value the floor exists to
/// refuse. Pinned by `requested_steps_reads_every_shape_the_worker_reads`.
///
/// Unlike [`resolve_duration`] / [`resolve_fps`] this does **not** fall back to `defaults.steps`.
/// Those two resolve because the API serializes a value for them whether or not the caller named
/// one, so the gate had to judge the resolved value or it would reject payloads the route built
/// itself (sc-12400). `advanced` is a verbatim passthrough map with no blanket step count, so an
/// omitted `steps` really is "the engine's own default" and inventing one here would gate a value
/// nobody sends. That the shipped `defaults.steps` clears the shipped floor is a MANIFEST invariant,
/// asserted against the real bytes rather than papered over at runtime.
pub fn requested_steps(advanced: &JsonObject) -> Option<u32> {
    advanced.get("steps").and_then(|value| {
        value
            .as_u64()
            .or_else(|| value.as_str()?.trim().parse().ok())
            .map(|steps: u64| steps as u32)
    })
}

/// The actionable rejection for a `steps` count under the model's declared [`hard_min_steps`], or
/// `None` to admit. The pure decision, so both enqueue gates assert on one message (sc-19426).
///
/// **Reject, never clamp**, for [`crate::video_request::duration_limit_error`]'s reason: quietly raising a 1-step request
/// to 2 doubles the compute the caller asked for with no error and no signal, which is the same
/// silent-coercion class as the 848→832 dimension rewrite (sc-11993 / sc-12294) and as delivering
/// 14.375 s for a 15 s request (sc-17147). It also matches the engine's own posture — the reference
/// raises rather than rounding up.
///
/// Message follows the house convention (`mlx_fit_gate::too_big_error`): name the model, state what
/// was asked and what the floor is, and give the lever. Pinned by
/// `steps_limit_error_names_the_model_the_request_and_the_floor`.
///
/// # Two independent constraints, one gate
///
/// sc-19502 added the [`allowed_steps`] menu here rather than as a second function so that the new
/// key needed **no new wiring**: the API (`generation.rs`) and the worker
/// (`video_jobs/mod.rs`) already call this on the video enqueue path, so both immediately became
/// readers of `limits.steps` and the rejection is reachable by a real request the day the key is
/// declared. A parallel `steps_menu_error` would have been a second thing to remember to call at
/// two seams — and forgetting one is precisely how a declared key becomes an inert one.
///
/// The menu is checked **first**: it is the stricter, exact statement, so when a model somehow
/// declares both, the caller gets told the legal value rather than a floor that would still leave
/// them off-menu.
pub fn steps_limit_error(
    model: &str,
    steps: u32,
    model_manifest_entry: &JsonObject,
) -> Option<String> {
    // `limits.steps` — the exact-value menu (sc-19502). Membership, not `<`: an over-menu count is
    // as unrenderable as an under-menu one on a distilled schedule.
    if let Some(menu) = allowed_steps(model_manifest_entry) {
        if !menu.contains(&steps) {
            let allowed = humanized_number_menu(&menu);
            // Singular gets the natural "a fixed 8-step schedule"; a multi-entry menu cannot take
            // the `-step` suffix ("a fixed 4, 8, or 12-step schedule" reads as one number), so it
            // states the set instead.
            let schedule = match menu.as_slice() {
                [only] => format!("runs a fixed {only}-step schedule"),
                _ => format!("renders at {allowed} steps only"),
            };
            return Some(format!(
                "{model} {schedule}, but this request asks for {steps} steps. Set steps to \
                 {allowed}, or choose a model whose step count you can change."
            ));
        }
        return None;
    }
    if let Some(floor) = hard_min_steps(model_manifest_entry) {
        if steps < floor {
            return Some(format!(
                "{model} renders with at least {floor} sampling steps, but this request asks for \
                 {steps}. Raise the step count to {floor} or more, or choose a model that renders \
                 in fewer steps."
            ));
        }
    }
    let ceiling = hard_max_steps(model_manifest_entry)?;
    (steps > ceiling).then(|| {
        format!(
            "{model} renders with at most {ceiling} sampling steps, but this request asks for \
             {steps}. Lower the step count to {ceiling} or fewer."
        )
    })
}

/// An advertised numeric menu as a human list: `30`, `16 or 24`, `6, 7, 8, 10, 12, or 25`.
///
/// Shared by the fps and step menus (sc-19502) — the formatting is axis-agnostic, and a second copy
/// would be a second place for the list style to drift.
pub(crate) fn humanized_number_menu(menu: &[u32]) -> String {
    match menu {
        [only] => only.to_string(),
        [first, second] => format!("{first} or {second}"),
        [rest @ .., last] => format!(
            "{}, or {last}",
            rest.iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        [] => String::new(),
    }
}

#[cfg(test)]
mod tests;
