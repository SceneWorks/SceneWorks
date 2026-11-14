//! Portable declarations shared by the native application and browser. Loading the catalog
//! requires neither a database nor an installed model or inference runtime.
use serde::Serialize;
use serde_json::Value;

mod advanced;
mod requests;
mod requests_aux;
mod requests_yue2;
mod training;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Shape {
    Bool,
    Integer,
    Number,
    Text,
    Strings,
    Numbers,
    Object(&'static str),
    List(&'static str),
    OneOf(&'static str),
}

impl Shape {
    const fn renderer(self) -> &'static str {
        match self {
            Self::Bool => "checkbox",
            Self::Integer | Self::Number => "number",
            Self::Text => "text",
            Self::OneOf(_) => "select",
            Self::Strings | Self::Numbers => "list",
            Self::Object(schema) | Self::List(schema) => schema,
        }
    }
}

/// Literal uses the JSON spelling (including quotes on strings). Other variants identify
/// the authority that supplies an omitted value. Explicit submission remains distinguishable
/// from omission even when the payload and the resolved default compare equal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum DefaultRule {
    Absent,
    Required,
    Literal(&'static str),
    Model(&'static str),
    Target(&'static str),
    Provider(&'static str),
    Inherit(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum IntentClass {
    WhatToMake,
    MachineBudget,
    Context,
}

/// Evaluated against the model and the selected backend/target's exported capabilities.
/// Positive facts require a declaration. ModelNotFalse preserves existing opt-out controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Predicate {
    Always,
    Operation(&'static str),
    Modality(&'static str),
    Family(&'static str),
    Capability(&'static str),
    Adapter(&'static str),
    Generator(&'static str),
    ModelField(&'static str),
    ModelNotFalse(&'static str),
    ModelContains(&'static str, &'static str),
    Not(&'static Predicate),
    Feature(&'static str),
    TrainingTarget(&'static str),
    TrainingKernel(&'static str),
    TrainingId(&'static str),
    Any(&'static [Predicate]),
    All(&'static [Predicate]),
}

pub struct OptionContext<'a> {
    /// Current request mode or job kind, supplied by the request owner.
    pub operation: &'a str,
    /// Product projection of ModelSpec. Includes authored limits and the selected defaults.
    pub model: &'a Value,
    /// Features of the selected native generator/provider, not the union of both backends.
    pub features: &'a Value,
    /// The effective training target, including per-platform defaults/limits.
    pub target: &'a Value,
}

fn field<'a>(root: &'a Value, path: &str) -> Option<&'a Value> {
    path.split('.').try_fold(root, |value, key| value.get(key))
}

fn enabled(value: &Value) -> bool {
    match value {
        Value::Null | Value::Bool(false) => false,
        Value::Array(list) => !list.is_empty(),
        Value::Object(map) => !map.is_empty(),
        Value::String(text) => !text.is_empty(),
        _ => true,
    }
}

impl Predicate {
    pub fn matches(self, ctx: &OptionContext<'_>) -> bool {
        match self {
            Self::Always => true,
            Self::Operation(kind) => ctx.operation == kind,
            Self::Modality(kind) => ctx.model.get("type").and_then(Value::as_str) == Some(kind),
            Self::Family(family) => ctx.model.get("family").and_then(Value::as_str) == Some(family),
            Self::Capability(key) => ctx
                .model
                .get("capabilities")
                .and_then(Value::as_array)
                .is_some_and(|v| v.iter().any(|value| value.as_str() == Some(key))),
            Self::Adapter(key) => ctx.model.get("adapter").and_then(Value::as_str) == Some(key),
            Self::Generator(key) => field(ctx.features, "id").and_then(Value::as_str) == Some(key),
            Self::ModelField(key) => field(ctx.model, key).is_some_and(enabled),
            Self::ModelNotFalse(key) => field(ctx.model, key) != Some(&Value::Bool(false)),
            Self::ModelContains(path, expected) => field(ctx.model, path)
                .and_then(Value::as_array)
                .is_some_and(|values| values.iter().any(|value| value.as_str() == Some(expected))),
            Self::Not(predicate) => !predicate.matches(ctx),
            Self::Feature(key) => {
                field(ctx.features, key).is_some_and(enabled)
                    || ctx
                        .features
                        .as_array()
                        .is_some_and(|list| list.iter().any(|v| v.as_str() == Some(key)))
            }
            Self::TrainingTarget(key) => field(ctx.target, key).is_some_and(enabled),
            Self::TrainingKernel(key) => {
                ctx.target.get("kernel").and_then(Value::as_str) == Some(key)
            }
            Self::TrainingId(key) => ctx.target.get("id").and_then(Value::as_str) == Some(key),
            Self::Any(items) => items.iter().any(|it| it.matches(ctx)),
            Self::All(items) => items.iter().all(|it| it.matches(ctx)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Effect {
    pub when: &'static str,
    pub action: &'static str,
    pub target: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct UiMeta {
    pub label: &'static str,
    pub renderer: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OptionSpec {
    pub id: &'static str,
    /// Qualified parent of the wire field. Same-spelled fields on different request types
    /// may have different meanings (e.g. audio versus image guidance).
    pub location: &'static str,
    pub key: &'static str,
    pub aliases: &'static [&'static str],
    pub shape: Shape,
    pub default: DefaultRule,
    /// Product/target limit paths and the owning rule for structured and cross-field constraints.
    pub limits: &'static [&'static str],
    pub applies_when: Predicate,
    pub effects: &'static [Effect],
    pub intent: IntentClass,
    pub ui: UiMeta,
    /// Engine field, host operation or receipt which consumes the value.
    pub binding: &'static str,
    pub user_settable: bool,
    /// Existing fallback/coercion behavior of this wire carrier.
    pub compatibility: &'static str,
}

impl OptionSpec {
    pub const fn new(
        id: &'static str,
        location: &'static str,
        key: &'static str,
        shape: Shape,
        default: DefaultRule,
        intent: IntentClass,
        binding: &'static str,
    ) -> Self {
        Self {
            id,
            location,
            key,
            aliases: &[],
            shape,
            default,
            limits: &[],
            applies_when: Predicate::Always,
            effects: &[],
            intent,
            ui: UiMeta {
                label: key,
                renderer: shape.renderer(),
            },
            binding,
            user_settable: true,
            compatibility: "typed wire value",
        }
    }

    pub const fn aliases(mut self, aliases: &'static [&'static str]) -> Self {
        self.aliases = aliases;
        self
    }
    pub const fn limits(mut self, limits: &'static [&'static str]) -> Self {
        self.limits = limits;
        self
    }
    pub const fn applies(mut self, predicate: Predicate) -> Self {
        self.applies_when = predicate;
        self
    }
    pub const fn effects(mut self, effects: &'static [Effect]) -> Self {
        self.effects = effects;
        self
    }
    pub const fn ui(mut self, label: &'static str, renderer: &'static str) -> Self {
        self.ui = UiMeta { label, renderer };
        self
    }
    pub const fn internal(mut self) -> Self {
        self.user_settable = false;
        self.ui.renderer = "hidden";
        self
    }
    pub const fn compat(mut self, rule: &'static str) -> Self {
        self.compatibility = rule;
        self
    }

    pub fn read<'a>(&self, parent: &'a Value) -> Option<(&'static str, &'a Value)> {
        std::iter::once(self.key)
            .chain(self.aliases.iter().copied())
            .find_map(|name| parent.get(name).map(|value| (name, value)))
    }
}

pub fn registry() -> impl Iterator<Item = &'static OptionSpec> {
    advanced::SPECS
        .iter()
        .chain(requests::SPECS)
        .chain(requests_aux::SPECS)
        .chain(requests_yue2::SPECS)
        .chain(training::SPECS)
}

pub fn by_id(id: &str) -> Option<&'static OptionSpec> {
    registry().find(|option| option.id == id)
}

pub fn by_wire(location: &str, key: &str) -> Option<&'static OptionSpec> {
    let location = match location {
        "film.model.advanced" | "image.advanced" | "video.advanced" => "advanced",
        location if location.starts_with("advanced.mossImageAdvanced") => {
            return by_wire(&location.replacen("advanced.", "image.", 1), key)
        }
        location => location,
    };
    registry().find(|option| {
        option.location == location && (option.key == key || option.aliases.contains(&key))
    })
}
