//! Static model registry. The compatibility projection gives the product catalog and job
//! resolver one authority for converted builtins while legacy manifest consumers migrate.
use std::collections::BTreeMap;
use std::sync::LazyLock;

use serde_json::{Map, Value};

mod spec;
pub use spec::ModelSpec;

static BUILTINS: LazyLock<BTreeMap<String, ModelSpec>> = LazyLock::new(|| {
    serde_json::from_str(include_str!(concat!(
        env!("OUT_DIR"),
        "/model-registry.json"
    )))
    .expect("validated build-time ModelSpec registry")
});

pub fn builtin_model_spec(id: &str) -> Option<&'static ModelSpec> {
    BUILTINS.get(id)
}

impl ModelSpec {
    /// Author choices may change display text, visibility and valid product defaults. Native
    /// routes, immutable download identities, restrictions, and evidence always come from the
    /// static spec. Credentials, install observations and physical bindings are added by their host.
    pub fn project(&self, user: Option<&Value>) -> Value {
        let mut model = serde_json::to_value(&self.product).expect("serializable product model");
        for (backend, evidence) in &self.evidence {
            let fields = serde_json::to_value(evidence).expect("serializable backend evidence");
            let mut lane = model[backend].as_object().cloned().unwrap_or_default();
            lane.extend(fields.as_object().expect("evidence object").clone());
            model[backend] = Value::Object(lane);
        }
        if let Some(user) = user.and_then(Value::as_object) {
            self.apply_author_choices(&mut model, user);
        }
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
        let Some(defaults) = user.get("defaults").and_then(Value::as_object) else {
            return;
        };
        let menus = &self.product.limits;
        if let Some(resolution) = defaults.get("resolution").and_then(Value::as_str) {
            if menus.resolutions.iter().any(|r| r == resolution) {
                model["defaults"]["resolution"] = resolution.into();
            }
        }
        if let Some(count) = defaults.get("count").and_then(Value::as_u64) {
            if menus.count.iter().any(|&n| u64::from(n) == count) {
                model["defaults"]["count"] = count.into();
            }
        }
        if let Some(steps) = defaults.get("steps").and_then(Value::as_u64) {
            if (1..=u64::from(u32::MAX)).contains(&steps) {
                model["defaults"]["steps"] = steps.into();
            }
        }
        for (key, menu) in [
            ("sampler", &menus.samplers),
            ("scheduler", &menus.schedulers),
        ] {
            if let Some(choice) = defaults.get(key).and_then(Value::as_str) {
                if menu.iter().any(|item| item == choice) {
                    model["defaults"][key] = choice.into();
                }
            }
        }
        if self.product.image.supports_guidance {
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
mod tests;
