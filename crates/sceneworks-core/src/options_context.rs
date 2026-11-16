//! Join portable option predicates to the selected ModelSpec lane. This projection contains
//! catalog/native declaration facts only; local installation policy remains with the worker.
use crate::registry::ModelSpec;
use serde_json::{json, Value};

pub struct OptionFacts {
    pub operation: String,
    pub model: Value,
    pub features: Value,
    pub target: Value,
}

impl OptionFacts {
    pub fn context(&self) -> crate::options::OptionContext<'_> {
        crate::options::OptionContext {
            operation: &self.operation,
            model: &self.model,
            features: &self.features,
            target: &self.target,
        }
    }
}

pub fn for_model(
    spec: &ModelSpec,
    backend: &str,
    mode: &str,
    generator: Option<&str>,
) -> OptionFacts {
    let route = spec
        .routes
        .get(backend)
        .filter(|route| generator.is_none_or(|id| id == route.id))
        .or_else(|| {
            spec.video_routes
                .get(backend)?
                .iter()
                .filter(|route| route.mode == mode)
                .flat_map(|route| &route.engines)
                .find(|route| generator.is_none_or(|id| id == route.id))
        })
        .or_else(|| {
            spec.provider_routes
                .get(backend)?
                .iter()
                .filter_map(|route| route.generator.as_ref())
                .find(|route| generator.is_none_or(|id| id == route.id))
        });
    let mut model = spec.project(None);
    crate::decoder_support::apply_to_model_entry(model.as_object_mut().expect("model projection"));
    let mut features = route
        .map(|route| serde_json::to_value(route).expect("serializable generator facts"))
        .unwrap_or_else(|| json!({}));
    if let Some(decoders) = model
        .get("decoders")
        .and_then(|value| value.get("byBackend"))
        .and_then(|value| value.get(backend))
    {
        features["decoders"] = decoders.clone();
    }
    OptionFacts {
        operation: mode.to_owned(),
        model,
        features,
        target: Value::Null,
    }
}
