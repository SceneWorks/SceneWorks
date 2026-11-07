//! Declared image-lane reference, geometry and sampling-menu admission.

use crate::contracts::JsonObject;
use serde_json::Value;

/// `limits.maxReferenceAssets` on an IMAGE model — the ordered reference list's ceiling — or
/// `None` for **no declared cap** (sc-24113, epic 24107).
///
/// The image sibling of [`crate::video_request::reference_limit_error`], and deliberately NOT that function: the video
/// one judges three lists with video field names and, crucially, DEFAULTS `images` to 8. A default
/// is right there (every video model really does take references) and wrong here — most image
/// models take none at all, and an absent key on the image lane must mean "this gate has no
/// opinion", so every already-shipped image model is byte-for-byte unchanged. The per-family
/// refusals that exist today (the worker's `MAX_EDIT_REFERENCES`, Krea's 2, FLUX.2's 5) keep
/// working exactly as they do; this only adds an enqueue-time 400 for a model that declares a
/// number, so the user hears about it before a job is created rather than after one fails.
///
/// Qwen-Image 2.1 is the first image model to declare one: its edit path takes 1–10 ORDERED
/// references (S3 edit contract) and the engine refuses an 11th by name at `validate`.
pub fn image_max_reference_assets(model_manifest_entry: &JsonObject) -> Option<usize> {
    model_manifest_entry
        .get("limits")
        .and_then(Value::as_object)
        .and_then(|limits| limits.get("maxReferenceAssets"))
        .and_then(Value::as_u64)
        .and_then(|cap| usize::try_from(cap).ok())
}

/// Reject an image request that supplies more references than the model declares, or `None`.
///
/// Reject, never truncate. Dropping the 11th reference would change the render — and for this
/// family it would change it twice over, because the ordered list is semantic: the template numbers
/// the images (`<image1>` …) and block-causal attention makes each one visible only to what follows,
/// so silently shortening the list re-numbers every reference after the cut.
///
/// `references` is the FLATTENED ordered conditioning list, not `referenceAssetIds.len()` — the
/// engine receives ONE list and `sourceAssetId` / `maskAssetId` are entries in it (sc-24110). The
/// wording below says so, because naming a single carrier would send the caller to trim a field
/// that may not even be the one over budget.
pub fn image_reference_limit_error(
    model: &str,
    references: usize,
    model_manifest_entry: &JsonObject,
) -> Option<String> {
    let cap = image_max_reference_assets(model_manifest_entry)?;
    (references > cap).then(|| {
        if cap == 0 {
            format!(
                "{model} takes no reference images, but this request supplies {references}. \
                 Remove the reference carriers (sourceAssetId, maskAssetId and referenceAssetIds), \
                 or choose a model that conditions on references."
            )
        } else {
            format!(
                "{model} takes up to {cap} reference images, but this request supplies \
                 {references}. Reduce the ordered conditioning list (source, mask and references \
                 together) to {cap} or fewer."
            )
        }
    })
}

/// A model's declared free-size envelope for IMAGE generation (sc-24113): the inclusive
/// `[limits.minDimension, limits.maxDimension]` range and the `limits.requiresDimensionsMultipleOf`
/// grid, each `None` when the model declares nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ImageDimensionEnvelope {
    pub min: Option<u32>,
    pub max: Option<u32>,
    pub multiple: Option<u32>,
}

/// Read the declared envelope off a manifest entry.
///
/// Every field is independently optional, and an absent one means the global API bounds still
/// decide that edge. That is what keeps this safe to run on EVERY image enqueue: a model declaring
/// nothing gets exactly the 256..=4096 it always had.
///
/// Values that could not be satisfied are dropped rather than enforced, the same typo'd-manifest
/// tolerance [`crate::limits::hard_min_steps`] takes: a zero/negative/fractional bound or a `multiple` of 0 or 1
/// is not a constraint, so it falls back to "undeclared" instead of bricking the model.
pub fn image_dimension_envelope(model_manifest_entry: &JsonObject) -> ImageDimensionEnvelope {
    let Some(limits) = model_manifest_entry
        .get("limits")
        .and_then(Value::as_object)
    else {
        return ImageDimensionEnvelope::default();
    };
    let read = |key: &str, floor: u64| {
        limits
            .get(key)
            .and_then(Value::as_u64)
            .filter(|value| *value >= floor)
            .and_then(|value| u32::try_from(value).ok())
    };
    ImageDimensionEnvelope {
        min: read("minDimension", 1),
        max: read("maxDimension", 1),
        multiple: read("requiresDimensionsMultipleOf", 2),
    }
}

/// Reject an image request whose geometry falls outside the model's declared envelope, or `None`.
///
/// Reject, never coerce. Image dimensions have always COERCED silently in this codebase (the video
/// path floors to a grid, the worker refits), and that is precisely why a wrong size reached the
/// engine unnoticed for so long — sc-12400 named the same failure for `defaults.resolution`. A user
/// who typed 2800 into the free Width box gets told 2752 is the ceiling; they do not get a 2752
/// render they did not ask for.
///
/// `fallback_min` / `fallback_max` are the caller's own blanket bounds, applied to any axis the
/// model does not declare. They are parameters rather than constants here because this crate has no
/// business knowing the API's envelope — and because passing them explicitly is what lets a
/// DECLARING model legitimately go BELOW the blanket floor (2.1 renders from 32 px, where the
/// API-wide floor is 256) without that floor being lowered for every model that declares nothing.
pub fn image_dimension_error(
    model: &str,
    width: u32,
    height: u32,
    model_manifest_entry: &JsonObject,
    fallback_min: u32,
    fallback_max: u32,
) -> Option<String> {
    let envelope = image_dimension_envelope(model_manifest_entry);
    let min = envelope.min.unwrap_or(fallback_min);
    let max = envelope.max.unwrap_or(fallback_max);
    // The GRID is enforced only for a model that ALSO declares its own `minDimension`/`maxDimension`
    // — i.e. one that has opted into a complete size envelope.
    //
    // `requiresDimensionsMultipleOf` is NOT new: six shipped `mage_flow*` image models have
    // declared `16` for a long time, and on the image lane it had exactly one reader — the Studio's
    // free Width/Height control, which SNAPS client-side. Enforcing it at enqueue for every
    // declaring model turned a silent snap into a 400 and would have rejected an off-grid size
    // those models have always accepted, including stored recipes and workflow replays (a 1000x1000
    // Mage-Flow render is not a multiple of 16).
    //
    // Gating on the envelope keeps the new refusal to models that declared one — today the
    // Qwen-Image 2.1 family (`qwen_image_2_1`, `qwen_image_2_1_turbo`) — while leaving the Mage
    // family exactly as it was. It is also the honest
    // reading: a bare stride was authored as advice to a snapping control, whereas a declared
    // min/max/stride triple is a model stating its whole request contract.
    let grid = envelope
        .multiple
        .filter(|_| envelope.min.is_some() && envelope.max.is_some());
    for (value, axis) in [(width, "width"), (height, "height")] {
        if value < min {
            return Some(format!(
                "{model} renders no smaller than {min} px per side; this request asks for a \
                 {axis} of {value}."
            ));
        }
        if value > max {
            return Some(format!(
                "{model} renders no larger than {max} px per side; this request asks for a \
                 {axis} of {value}."
            ));
        }
        if let Some(multiple) = grid {
            if !value.is_multiple_of(multiple) {
                return Some(format!(
                    "{model} renders on a {multiple}-pixel grid; this request's {axis} of {value} \
                     is not a multiple of {multiple}. The nearest are {} and {}.",
                    value - value % multiple,
                    value - value % multiple + multiple
                ));
            }
        }
    }
    None
}

/// Reject an image request whose `advanced.sampler` / `advanced.scheduler` is not on the model's
/// declared menu, or `None` (sc-24114).
///
/// The menu is `limits.samplers` / `limits.schedulers` UNIONED with every per-backend override
/// (`mlx.limits.*`, `candle.limits.*`): the API does not know which lane will claim the job, and a
/// name that ANY lane advertises is honoured there (the worker's manifest ⊆ engine drift guard
/// enforces that). A name NO lane advertises is one the worker would silently fall back from
/// (`normalize_sampling_knob`, sc-7127) — the render the caller asked for never happens and the job
/// reports success — so it is a 400 naming the menu instead.
///
/// Inert when the knob is absent, blank or the `"default"` sentinel (the same reading as the
/// worker's `read_advanced_sampling_knobs`), and when the model declares no menu at all.
pub fn image_sampling_knob_error(
    model: &str,
    advanced: &JsonObject,
    model_manifest_entry: &JsonObject,
) -> Option<String> {
    for (knob, menu_key) in [("sampler", "samplers"), ("scheduler", "schedulers")] {
        let Some(requested) = advanced
            .get(knob)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty() && *value != "default")
        else {
            continue;
        };
        let mut menu: Vec<&str> = Vec::new();
        let mut declared = false;
        let blocks = std::iter::once(Some(model_manifest_entry)).chain(
            ["mlx", "candle"]
                .into_iter()
                .map(|backend| model_manifest_entry.get(backend).and_then(Value::as_object)),
        );
        for block in blocks.flatten() {
            if let Some(names) = block
                .get("limits")
                .and_then(Value::as_object)
                .and_then(|limits| limits.get(menu_key))
                .and_then(Value::as_array)
            {
                declared = true;
                menu.extend(names.iter().filter_map(Value::as_str));
            }
        }
        if declared && !menu.contains(&requested) {
            let mut offered: Vec<&str> =
                menu.into_iter().filter(|name| *name != "default").collect();
            offered.sort_unstable();
            offered.dedup();
            let offered = if offered.is_empty() {
                "only its engine default".to_owned()
            } else {
                offered.join(", ")
            };
            return Some(format!(
                "{model} does not offer the {knob} \"{requested}\" (it offers {offered}). Pick one \
                 of those, or send \"default\"."
            ));
        }
    }
    None
}

#[cfg(test)]
mod tests;
