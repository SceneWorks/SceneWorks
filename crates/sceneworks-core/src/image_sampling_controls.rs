//! Typed reading and enqueue validation of the image-lane sampling controls that ride in a job's
//! `advanced` block beyond sampler / scheduler / steps / guidance (epic sc-25678, sc-25681):
//!
//! | `advanced` key | engine field | declared by (manifest `image` block) |
//! | --- | --- | --- |
//! | `schedulerShift` | `scheduler_shift` | `supportsSchedulerShift` (an axis without a scheduler menu) |
//! | `cfgInterval` (`[lo, hi]`) | `cfg_interval` | `supportsCfgInterval` |
//! | `captionOverflow` (`warn` / `error` / `silent`) | `caption_overflow` | `captionOverflowPolicies` |
//!
//! Plus the CFG-bound negative prompt (`image.negativePromptRequiresGuidance`, sc-25679): the
//! negative prompt is the CFG unconditional, so a non-empty one at guidance exactly 1.0 does
//! nothing and its engine refuses it.
//!
//! One function per question, shared by the API's enqueue gate ([`image_sampling_control_error`])
//! and the worker's pre-load request shaping, so the two can never disagree about what a payload
//! means. An undeclared control is REFUSED by name rather than accepted and dropped (E3), and a
//! control that could only be accepted-then-ignored at the resolved guidance (the interval and the
//! negative prompt at CFG 1.0) is refused too — the engine refuses both, so a 400 at enqueue is the
//! honest place to say so.

use serde_json::Value;

use crate::contracts::JsonObject;

/// `advanced` key of the CFG interval `[lo, hi]`.
pub const CFG_INTERVAL_KEY: &str = "cfgInterval";
/// `advanced` key of the caption-overflow policy.
pub const CAPTION_OVERFLOW_KEY: &str = "captionOverflow";
/// `advanced` key of the schedule shift (shared with the scheduler-menu models).
pub const SCHEDULER_SHIFT_KEY: &str = "schedulerShift";

/// What a text encoder with a fixed caption budget does with a prompt that tokenizes past it. The
/// spellings are upstream Iris-3B's `text_encoder.on_caption_overflow` vocabulary (and gen-core's
/// `CaptionOverflowPolicy::name`), verbatim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptionOverflow {
    /// Truncate and report a `caption_truncated` warning.
    Warn,
    /// Refuse the render.
    Error,
    /// Truncate without reporting.
    Silent,
}

impl CaptionOverflow {
    /// Every policy, in menu order.
    pub const ALL: [Self; 3] = [Self::Warn, Self::Error, Self::Silent];

    /// Parse the wire spelling; anything else is `None`.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|policy| policy.name() == name)
    }

    /// The wire spelling.
    pub fn name(self) -> &'static str {
        match self {
            Self::Warn => "warn",
            Self::Error => "error",
            Self::Silent => "silent",
        }
    }
}

/// The `image` sub-block of a manifest entry, if any.
fn image_block(model_manifest_entry: &JsonObject) -> Option<&JsonObject> {
    model_manifest_entry.get("image").and_then(Value::as_object)
}

fn declares(model_manifest_entry: &JsonObject, key: &str) -> bool {
    image_block(model_manifest_entry)
        .and_then(|image| image.get(key))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// `image.supportsSchedulerShift`: the model has a schedule-shift axis WITHOUT a scheduler menu
/// (Iris-3B's FlowDPM-Solver++ has one schedule, the shifted linear grid). ABSENT MEANS FALSE —
/// the scheduler-menu models keep their historical shift-with-a-named-scheduler behaviour.
pub fn supports_scheduler_shift(model_manifest_entry: &JsonObject) -> bool {
    declares(model_manifest_entry, "supportsSchedulerShift")
}

/// `image.supportsCfgInterval`. ABSENT MEANS FALSE.
pub fn supports_cfg_interval(model_manifest_entry: &JsonObject) -> bool {
    declares(model_manifest_entry, "supportsCfgInterval")
}

/// `image.negativePromptRequiresGuidance` (sc-25679). ABSENT MEANS FALSE.
pub fn negative_prompt_requires_guidance(model_manifest_entry: &JsonObject) -> bool {
    declares(model_manifest_entry, "negativePromptRequiresGuidance")
}

/// `image.captionOverflowPolicies`: the policies the model offers, or empty when it declares none.
pub fn caption_overflow_policies(model_manifest_entry: &JsonObject) -> Vec<CaptionOverflow> {
    image_block(model_manifest_entry)
        .and_then(|image| image.get("captionOverflowPolicies"))
        .and_then(Value::as_array)
        .map(|names| {
            names
                .iter()
                .filter_map(Value::as_str)
                .filter_map(CaptionOverflow::from_name)
                .collect()
        })
        .unwrap_or_default()
}

/// `advanced.cfgInterval` as `(lo, hi)`. Absent or `null` ⇒ `Ok(None)` (the model's own interval);
/// anything else must be a two-number array with `0 <= lo < hi <= 1` (model time runs from 1 =
/// noise to 0 = clean), or it is a typed error naming the field.
pub fn requested_cfg_interval(advanced: &JsonObject) -> Result<Option<(f32, f32)>, String> {
    let Some(value) = advanced
        .get(CFG_INTERVAL_KEY)
        .filter(|value| !value.is_null())
    else {
        return Ok(None);
    };
    let bounds = value
        .as_array()
        .filter(|pair| pair.len() == 2)
        .and_then(|pair| Some((pair[0].as_f64()?, pair[1].as_f64()?)))
        .ok_or_else(|| {
            format!(
                "advanced.{CFG_INTERVAL_KEY} must be a [start, end] pair of numbers, got {value}"
            )
        })?;
    let (lo, hi) = bounds;
    if !(lo.is_finite() && hi.is_finite() && 0.0 <= lo && lo < hi && hi <= 1.0) {
        return Err(format!(
            "advanced.{CFG_INTERVAL_KEY} [{lo}, {hi}] must satisfy 0 <= start < end <= 1 (model \
             time runs from 1 = noise to 0 = clean)"
        ));
    }
    Ok(Some((lo as f32, hi as f32)))
}

/// `advanced.captionOverflow`. Absent, `null`, empty or `"default"` ⇒ `Ok(None)` (the checkpoint's
/// own policy); an unknown name is a typed error naming the vocabulary.
pub fn requested_caption_overflow(
    advanced: &JsonObject,
) -> Result<Option<CaptionOverflow>, String> {
    let Some(value) = advanced
        .get(CAPTION_OVERFLOW_KEY)
        .filter(|value| !value.is_null())
    else {
        return Ok(None);
    };
    let name = value
        .as_str()
        .map(str::trim)
        .ok_or_else(|| format!("advanced.{CAPTION_OVERFLOW_KEY} must be a string, got {value}"))?;
    if name.is_empty() || name == "default" {
        return Ok(None);
    }
    CaptionOverflow::from_name(name).map(Some).ok_or_else(|| {
        format!(
            "advanced.{CAPTION_OVERFLOW_KEY} \"{name}\" is not a caption-overflow policy (use warn, \
             error or silent)"
        )
    })
}

/// `advanced.schedulerShift`, typed: absent / `null` ⇒ `Ok(None)`; otherwise a finite number
/// `> 0` (a numeric string is accepted, as the worker's historical reader does).
pub fn requested_scheduler_shift(advanced: &JsonObject) -> Result<Option<f32>, String> {
    let Some(value) = advanced
        .get(SCHEDULER_SHIFT_KEY)
        .filter(|value| !value.is_null())
    else {
        return Ok(None);
    };
    let shift = value
        .as_f64()
        .or_else(|| value.as_str()?.trim().parse().ok())
        .ok_or_else(|| format!("advanced.{SCHEDULER_SHIFT_KEY} must be a number, got {value}"))?;
    if !(shift.is_finite() && shift > 0.0) {
        return Err(format!(
            "advanced.{SCHEDULER_SHIFT_KEY} {shift} must be a finite value > 0"
        ));
    }
    Ok(Some(shift as f32))
}

/// The guidance a job resolves to: `advanced.guidanceScale` when sent (number or numeric string,
/// the worker's `resolve_guidance` reading), else the model's `defaults.guidanceScale`.
pub fn effective_guidance(advanced: &JsonObject, model_manifest_entry: &JsonObject) -> Option<f64> {
    advanced
        .get("guidanceScale")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .or_else(|| {
            model_manifest_entry
                .get("defaults")
                .and_then(|defaults| defaults.get("guidanceScale"))
                .and_then(Value::as_f64)
        })
}

/// Does classifier-free guidance run at this guidance? Exactly 1.0 is CFG off (the engines'
/// `uses_cfg` test).
fn cfg_off(guidance: Option<f64>) -> bool {
    guidance == Some(1.0)
}

/// The enqueue-time refusal for the sampling controls above, or `None` when the request is
/// honourable. `model` only labels the message. Pure; the API calls it on the post-preset payload
/// and the worker re-applies the same rules before load.
pub fn image_sampling_control_error(
    model: &str,
    advanced: &JsonObject,
    negative_prompt: &str,
    model_manifest_entry: &JsonObject,
) -> Option<String> {
    let interval = match requested_cfg_interval(advanced) {
        Ok(interval) => interval,
        Err(message) => return Some(format!("{model}: {message}")),
    };
    let overflow = match requested_caption_overflow(advanced) {
        Ok(overflow) => overflow,
        Err(message) => return Some(format!("{model}: {message}")),
    };
    if interval.is_some() && !supports_cfg_interval(model_manifest_entry) {
        return Some(format!(
            "{model} does not offer a guidance interval (advanced.{CFG_INTERVAL_KEY}); drop it"
        ));
    }
    if let Some(policy) = overflow {
        let offered = caption_overflow_policies(model_manifest_entry);
        if !offered.contains(&policy) {
            return Some(if offered.is_empty() {
                format!(
                    "{model} does not offer a caption-overflow policy \
                     (advanced.{CAPTION_OVERFLOW_KEY}); drop it"
                )
            } else {
                format!(
                    "{model} does not offer the caption-overflow policy \"{}\" (it offers {})",
                    policy.name(),
                    offered
                        .iter()
                        .map(|policy| policy.name())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            });
        }
    }
    if supports_scheduler_shift(model_manifest_entry) {
        if let Err(message) = requested_scheduler_shift(advanced) {
            return Some(format!("{model}: {message}"));
        }
    }
    let guidance = effective_guidance(advanced, model_manifest_entry);
    if cfg_off(guidance) {
        if interval.is_some() {
            return Some(format!(
                "{model}: a guidance interval has no effect at guidance 1.0 (classifier-free \
                 guidance is off); raise guidance or drop advanced.{CFG_INTERVAL_KEY}"
            ));
        }
        if negative_prompt_requires_guidance(model_manifest_entry)
            && !negative_prompt.trim().is_empty()
        {
            return Some(format!(
                "{model}: the negative prompt has no effect at guidance 1.0 (classifier-free \
                 guidance is off); raise guidance or clear the negative prompt"
            ));
        }
    }
    None
}

/// Top-level payload key of a NATIVE prompt batch (sc-25681): the prompts one engine call renders
/// together (upstream Iris-3B `generate(prompts=[…])`). `prompt` then carries `prompts[0]`, so every
/// single-prompt consumer (titles, recipes, the prompt-length gate) keeps working, and the worker
/// hands the engine the list alone.
pub const PROMPTS_KEY: &str = "prompts";

/// `image.maxPromptBatchImages`: the model renders a list of prompts as ONE batched pass, at most
/// this many images (`prompts × count`) per request — the engine's own `max_count` bound on the
/// batch. ABSENT MEANS NONE: a multi-prompt run stays one job per prompt for every other model.
pub fn max_prompt_batch_images(model_manifest_entry: &JsonObject) -> Option<u32> {
    image_block(model_manifest_entry)
        .and_then(|image| image.get("maxPromptBatchImages"))
        .and_then(Value::as_u64)
        .and_then(|max| u32::try_from(max).ok())
        .filter(|max| *max > 0)
}

/// The enqueue-time refusal of a native prompt batch, or `None` when `prompts` is empty (an
/// ordinary request) or honourable. `prompt` is the request's single prompt AFTER every server-side
/// fold (preset / style); it must equal `prompts[0]`, so a fold that rewrote it — and would not
/// have rewritten the rest — is refused rather than half-applied. `count` repeats the whole batch:
/// image `k` of the `prompts.len() × count` renders `prompts[k % prompts.len()]` on seed `seed + k`.
pub fn prompt_batch_error(
    model: &str,
    prompt: &str,
    prompts: &[String],
    count: u32,
    model_manifest_entry: &JsonObject,
) -> Option<String> {
    if prompts.is_empty() {
        return None;
    }
    let Some(max) = max_prompt_batch_images(model_manifest_entry) else {
        return Some(format!(
            "{model} does not render a prompt batch in one pass ({PROMPTS_KEY}); send one job \
             per prompt"
        ));
    };
    if prompts.len() < 2 {
        return Some(format!(
            "{model}: a prompt batch needs at least two {PROMPTS_KEY}; send a single prompt as \
             `prompt`"
        ));
    }
    if let Some(index) = prompts.iter().position(|entry| entry.trim().is_empty()) {
        return Some(format!("{model}: {PROMPTS_KEY}[{index}] is empty"));
    }
    if prompt != prompts[0] {
        return Some(format!(
            "{model}: `prompt` must equal {PROMPTS_KEY}[0] for a prompt batch (a server-side \
             preset or style fold rewrites only `prompt`; compose every batch prompt client-side)"
        ));
    }
    let images = prompts.len() as u64 * u64::from(count.max(1));
    (images > u64::from(max)).then(|| {
        format!(
            "{model}: {} prompts × count {count} = {images} images exceeds the {max} images it \
             renders in one batch",
            prompts.len()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn object(value: Value) -> JsonObject {
        value.as_object().cloned().expect("object literal")
    }

    fn iris_like() -> JsonObject {
        object(json!({
            "id": "iris_like",
            "defaults": { "guidanceScale": 3.0 },
            "image": {
                "negativePromptRequiresGuidance": true,
                "supportsSchedulerShift": true,
                "supportsCfgInterval": true,
                "captionOverflowPolicies": ["warn", "error", "silent"]
            }
        }))
    }

    fn plain() -> JsonObject {
        object(json!({ "id": "plain", "defaults": { "guidanceScale": 3.0 } }))
    }

    #[test]
    fn caption_overflow_names_round_trip() {
        for policy in CaptionOverflow::ALL {
            assert_eq!(CaptionOverflow::from_name(policy.name()), Some(policy));
        }
        assert_eq!(CaptionOverflow::from_name("truncate"), None);
    }

    #[test]
    fn requested_controls_parse_table() {
        // (advanced, interval, overflow, shift) — Err strings are matched by a fragment.
        type Expect = (
            Result<Option<(f32, f32)>, &'static str>,
            Result<Option<CaptionOverflow>, &'static str>,
            Result<Option<f32>, &'static str>,
        );
        let cases: Vec<(Value, Expect)> = vec![
            (json!({}), (Ok(None), Ok(None), Ok(None))),
            (
                json!({ "cfgInterval": null, "captionOverflow": null, "schedulerShift": null }),
                (Ok(None), Ok(None), Ok(None)),
            ),
            (
                json!({ "cfgInterval": [0.1, 0.9], "captionOverflow": "error", "schedulerShift": 2.5 }),
                (
                    Ok(Some((0.1, 0.9))),
                    Ok(Some(CaptionOverflow::Error)),
                    Ok(Some(2.5)),
                ),
            ),
            (
                json!({ "cfgInterval": [0, 1], "captionOverflow": "default", "schedulerShift": "3" }),
                (Ok(Some((0.0, 1.0))), Ok(None), Ok(Some(3.0))),
            ),
            (
                json!({ "cfgInterval": [0.9, 0.1], "captionOverflow": "loud", "schedulerShift": 0 }),
                (
                    Err("0 <= start < end"),
                    Err("not a caption-overflow"),
                    Err("> 0"),
                ),
            ),
            (
                json!({ "cfgInterval": [0.5], "captionOverflow": 3, "schedulerShift": "x" }),
                (
                    Err("pair of numbers"),
                    Err("must be a string"),
                    Err("must be a number"),
                ),
            ),
            (
                json!({ "cfgInterval": [-0.1, 0.5] }),
                (Err("0 <= start < end"), Ok(None), Ok(None)),
            ),
            (
                json!({ "cfgInterval": [0.2, 1.5] }),
                (Err("0 <= start < end"), Ok(None), Ok(None)),
            ),
        ];
        for (advanced, (interval, overflow, shift)) in cases {
            let advanced = object(advanced);
            let check =
                |label: &str, got: Result<String, String>, want: Result<String, &str>| match (
                    &got, &want,
                ) {
                    (Ok(got), Ok(want)) => assert_eq!(got, want, "{label} {advanced:?}"),
                    (Err(got), Err(fragment)) => {
                        assert!(got.contains(fragment), "{label} {advanced:?}: {got}")
                    }
                    _ => panic!("{label} {advanced:?}: got {got:?}, want {want:?}"),
                };
            check(
                "interval",
                requested_cfg_interval(&advanced).map(|v| format!("{v:?}")),
                interval.map(|v| format!("{v:?}")),
            );
            check(
                "overflow",
                requested_caption_overflow(&advanced).map(|v| format!("{v:?}")),
                overflow.map(|v| format!("{v:?}")),
            );
            check(
                "shift",
                requested_scheduler_shift(&advanced).map(|v| format!("{v:?}")),
                shift.map(|v| format!("{v:?}")),
            );
        }
    }

    #[test]
    fn a_declaring_model_accepts_every_control_with_cfg_on() {
        let advanced = object(json!({
            "guidanceScale": 4.5,
            "cfgInterval": [0.2, 0.8],
            "captionOverflow": "silent",
            "schedulerShift": 3.0,
            "sampler": "euler",
        }));
        assert_eq!(
            image_sampling_control_error("iris_like", &advanced, "blurry", &iris_like()),
            None
        );
    }

    #[test]
    fn an_undeclared_control_is_refused_by_name_not_dropped() {
        let interval = object(json!({ "cfgInterval": [0.2, 0.8] }));
        let message = image_sampling_control_error("plain", &interval, "", &plain()).unwrap();
        assert!(message.contains("guidance interval"), "{message}");
        let overflow = object(json!({ "captionOverflow": "warn" }));
        let message = image_sampling_control_error("plain", &overflow, "", &plain()).unwrap();
        assert!(message.contains("caption-overflow"), "{message}");
        // A non-declaring model keeps its historical shift handling (no validation here).
        let shift = object(json!({ "schedulerShift": -1 }));
        assert_eq!(
            image_sampling_control_error("plain", &shift, "", &plain()),
            None
        );
        // A declaring model validates it.
        let message = image_sampling_control_error("iris_like", &shift, "", &iris_like()).unwrap();
        assert!(message.contains("> 0"), "{message}");
    }

    #[test]
    fn an_offered_subset_of_policies_names_what_is_offered() {
        let mut entry = iris_like();
        entry["image"]["captionOverflowPolicies"] = json!(["warn"]);
        let advanced = object(json!({ "captionOverflow": "error" }));
        let message = image_sampling_control_error("iris_like", &advanced, "", &entry).unwrap();
        assert!(message.contains("it offers warn"), "{message}");
    }

    #[test]
    fn cfg_off_refuses_the_interval_and_a_bound_negative_prompt() {
        let entry = iris_like();
        let off = object(json!({ "guidanceScale": 1.0, "cfgInterval": [0.2, 0.8] }));
        let message = image_sampling_control_error("iris_like", &off, "", &entry).unwrap();
        assert!(message.contains("guidance 1.0"), "{message}");
        let off = object(json!({ "guidanceScale": "1" }));
        let message = image_sampling_control_error("iris_like", &off, "blurry", &entry).unwrap();
        assert!(message.contains("negative prompt"), "{message}");
        // Blank negative prompt at CFG off is fine; so is the default guidance (3.0).
        assert_eq!(
            image_sampling_control_error("iris_like", &off, "  ", &entry),
            None
        );
        assert_eq!(
            image_sampling_control_error("iris_like", &object(json!({})), "blurry", &entry),
            None
        );
        // A model whose default IS 1.0 is CFG-off with no override.
        let mut default_off = entry.clone();
        default_off["defaults"]["guidanceScale"] = json!(1.0);
        assert!(image_sampling_control_error(
            "iris_like",
            &object(json!({})),
            "blurry",
            &default_off
        )
        .is_some());
        // A model that does not bind its negative prompt to CFG is untouched.
        assert_eq!(
            image_sampling_control_error(
                "plain",
                &object(json!({ "guidanceScale": 1.0 })),
                "blurry",
                &plain()
            ),
            None
        );
    }

    fn batching() -> JsonObject {
        object(json!({ "id": "iris_like", "image": { "maxPromptBatchImages": 8 } }))
    }

    fn prompts(list: &[&str]) -> Vec<String> {
        list.iter().map(|prompt| (*prompt).to_owned()).collect()
    }

    /// sc-25681: the native prompt batch — declared cap, at least two prompts, `prompt` is
    /// `prompts[0]`, and `prompts × count` within the cap.
    ///
    /// *Mutation that reds this:* compare `prompts.len()` (not `× count`) against the cap.
    #[test]
    fn a_prompt_batch_is_bounded_by_the_declared_image_cap() {
        let entry = batching();
        let two = prompts(&["a cat", "a dog"]);
        assert_eq!(prompt_batch_error("iris_like", "", &[], 8, &entry), None);
        assert_eq!(
            prompt_batch_error("iris_like", "a cat", &two, 4, &entry),
            None
        );
        let over = prompt_batch_error("iris_like", "a cat", &two, 5, &entry).expect("2 × 5 > 8");
        assert!(over.contains("10 images") && over.contains('8'), "{over}");
        let nine = prompts(&["a", "b", "c", "d", "e", "f", "g", "h", "i"]);
        assert!(prompt_batch_error("iris_like", "a", &nine, 1, &entry).is_some());
        assert!(
            prompt_batch_error("iris_like", "a cat", &prompts(&["a cat"]), 1, &entry)
                .expect("one prompt is not a batch")
                .contains("at least two")
        );
        assert!(
            prompt_batch_error("iris_like", "a cat", &prompts(&["a cat", " "]), 1, &entry)
                .expect("blank prompt")
                .contains("prompts[1]")
        );
        assert!(
            prompt_batch_error("iris_like", "Subject: a cat", &two, 1, &entry)
                .expect("folded prompt")
                .contains("prompts[0]")
        );
        let undeclared = prompt_batch_error("plain", "a cat", &two, 1, &plain())
            .expect("an undeclared batch is refused by name");
        assert!(undeclared.contains("one job per prompt"), "{undeclared}");
        assert_eq!(max_prompt_batch_images(&plain()), None);
        assert_eq!(max_prompt_batch_images(&entry), Some(8));
    }
}
