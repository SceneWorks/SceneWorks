//! Static model registry. Every shipped builtin projects from the validated product manifest,
//! native descriptor export, and evidence store. Host-specific bindings are added by consumers.
use std::collections::BTreeMap;
use std::sync::LazyLock;

use serde_json::{Map, Value};

mod spec;
#[cfg(test)]
pub(crate) use spec::GeneratorRoute;
pub use spec::{supports_hires_fix, ModelSpec, RoutePolicy};
#[allow(dead_code)]
// Build-time parser validates the serialized policy; runtime reads that projection.
mod memory_route_policy;
pub use memory_route_policy::DeclaredMemoryRoute;
mod exceptions;
pub use exceptions::{approved_backend_exceptions, ApprovedBackendException};
mod trainer_routes;
pub(crate) use trainer_routes::{expected_backend_local_trainer_id, native_training_kernels};
#[cfg(test)]
pub(crate) use trainer_routes::{native_trainer_serves, offered_network_types};
#[allow(dead_code)] // Shared with build.rs, where every shipped evidence row is validated.
mod evidence;
#[cfg(test)]
mod policy;
pub use evidence::{builtin_memory_evidence, project_builtin_memory_evidence};

static BUILTINS: LazyLock<BTreeMap<String, ModelSpec>> = LazyLock::new(|| {
    serde_json::from_str(include_str!(concat!(
        env!("OUT_DIR"),
        "/model-registry.json"
    )))
    .expect("validated build-time ModelSpec registry")
});
static MEMORY_ROUTE_POLICY: LazyLock<memory_route_policy::MemoryRoutePolicyDocument> =
    LazyLock::new(|| {
        serde_json::from_str(include_str!(concat!(
            env!("OUT_DIR"),
            "/memory-route-policy.json"
        )))
        .expect("validated build-time memory route policy")
    });
static LEGACY_PREVIEW: LazyLock<BTreeMap<String, BTreeMap<String, bool>>> = LazyLock::new(|| {
    serde_json::from_str(include_str!(concat!(
        env!("OUT_DIR"),
        "/legacy-model-preview.json"
    )))
    .expect("validated legacy preview feature export")
});

pub fn builtin_model_spec(id: &str) -> Option<&'static ModelSpec> {
    BUILTINS.get(id)
}

/// All build-validated builtin specs in stable model-id order.
pub fn builtin_model_specs() -> impl Iterator<Item = &'static ModelSpec> {
    BUILTINS.values()
}

/// Declaration-owned load selectors, checked against native routes and memory evidence at build.
pub fn declared_memory_routes() -> &'static [DeclaredMemoryRoute] {
    &MEMORY_ROUTE_POLICY.declared_routes
}

/// The preview feature export includes registered builtins and historical job aliases that have
/// no public catalog row. An absent lane is unknown rather than unsupported.
pub fn preview_feature(id: &str) -> Option<&'static BTreeMap<String, bool>> {
    if let Some(spec) = builtin_model_spec(id) {
        return (!spec.preview_by_backend.is_empty()).then_some(&spec.preview_by_backend);
    }
    LEGACY_PREVIEW.get(id)
}

impl ModelSpec {
    /// Author choices may change display text, visibility, product menus and valid defaults. Native
    /// routes, immutable download identities, restrictions, and evidence always come from the
    /// static spec. Credentials, install observations and physical bindings are added by their host.
    pub fn project(&self, user: Option<&Value>) -> Value {
        let mut model = serde_json::to_value(&self.product).expect("serializable product model");
        if let Some(user) = user.and_then(Value::as_object) {
            self.apply_author_choices(&mut model, user);
        }
        let mut facts = Map::new();
        for (backend, evidence) in &self.evidence {
            facts.insert(
                backend.clone(),
                serde_json::to_value(evidence).expect("serializable backend evidence"),
            );
        }
        if !self.model_evidence.is_empty() {
            facts.insert(
                "model".to_owned(),
                Value::Object(self.model_evidence.clone()),
            );
        }
        if !self.depth_evidence.is_empty() {
            facts.insert(
                "depth".to_owned(),
                Value::Object(self.depth_evidence.clone()),
            );
        }
        if !self.download_evidence.is_empty() {
            facts.insert(
                "downloads".to_owned(),
                serde_json::to_value(&self.download_evidence)
                    .expect("serializable download evidence"),
            );
        }
        evidence::project_memory_evidence(&mut model, &Value::Object(facts))
            .expect("build-validated static model evidence projection");
        model
    }

    fn apply_author_choices(&self, model: &mut Value, user: &Map<String, Value>) {
        if let Some(name) = user.get("name").and_then(Value::as_str) {
            model["name"] = Value::from(name);
        }
        if let Some(hidden) = user.get("hidden").and_then(Value::as_bool) {
            model["hidden"] = Value::from(hidden);
        }
        if let Some(ui) = user.get("ui").and_then(Value::as_object) {
            for key in [
                "label",
                "description",
                "durationHint",
                "promptHint",
                "defaultPrompt",
                "defaultNegativePrompt",
            ] {
                if let Some(text) = ui.get(key).and_then(Value::as_str) {
                    model["ui"][key] = Value::from(text);
                }
            }
        }
        // Author menus can narrow shipped choices, never supply another native capability. Keep
        // defaults inside the effective menu even when an existing user's override narrows it.
        let limits = user.get("limits");
        let defaults = user.get("defaults");
        for (menu, default) in [
            ("resolutions", "resolution"),
            ("count", "count"),
            ("samplers", "sampler"),
            ("schedulers", "scheduler"),
        ] {
            // Not every builtin declares every menu: video/audio/utility products may
            // have no `count` (or no `limits` at all). A presentation-only override for
            // such a model must not panic — there is simply no menu to narrow.
            let Some(offered) = model["limits"][menu].as_array() else {
                continue;
            };
            if let Some(requested) = limits
                .and_then(|limits| limits.get(menu))
                .and_then(Value::as_array)
            {
                if !requested.is_empty()
                    && requested.iter().enumerate().all(|(i, value)| {
                        offered.contains(value) && !requested[..i].contains(value)
                    })
                {
                    let current = &model["defaults"][default];
                    if (!current.is_null() && !requested.contains(current))
                        || (current.is_null() && !requested.iter().any(|value| value == "default"))
                    {
                        model["defaults"][default] = requested[0].clone();
                    }
                    model["limits"][menu] = Value::Array(requested.clone());
                }
            }
            if let Some(choice) = defaults.and_then(|defaults| defaults.get(default)) {
                if model["limits"][menu]
                    .as_array()
                    .is_some_and(|offered| offered.contains(choice))
                {
                    model["defaults"][default] = choice.clone();
                }
            }
        }
        let Some(defaults) = defaults.and_then(Value::as_object) else {
            return;
        };
        if let Some(steps) = defaults.get("steps").and_then(Value::as_u64) {
            if (1..=u64::from(u32::MAX)).contains(&steps) {
                model["defaults"]["steps"] = steps.into();
            }
        }
        if self
            .product
            .fields
            .get("image")
            .and_then(|image| image.get("supportsGuidance"))
            .and_then(Value::as_bool)
            == Some(true)
        {
            if let Some(guidance) = defaults.get("guidanceScale").and_then(Value::as_f64) {
                if guidance >= 0.0 {
                    model["defaults"]["guidanceScale"] = guidance.into();
                }
            }
        }
    }
}

#[cfg(test)]
mod assemble;
#[cfg(test)]
mod parity_contract_tests;
#[cfg(test)]
mod tests;
