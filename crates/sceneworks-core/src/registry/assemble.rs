//! Consume the current descriptor dumps. ENG-7 replaces this input adapter at its producer cutover.
use super::spec::{BackendEvidence, GeneratorRoute, ModelSpec, ProductModel};
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct DescriptorDump {
    schema_version: u32,
    model_mappings: BTreeMap<String, String>,
    snapshot: Snapshot,
}

#[derive(Deserialize)]
struct Snapshot {
    backend: String,
    generator_capabilities: Vec<GeneratorRoute>,
}

pub fn assemble(
    products: &str,
    evidence: &str,
    dumps: &[(&str, &str)],
) -> Result<BTreeMap<String, ModelSpec>, String> {
    let products: Vec<ProductModel> = serde_json::from_str(products).map_err(|e| e.to_string())?;
    let mut evidence: BTreeMap<String, BTreeMap<String, BackendEvidence>> =
        serde_json::from_str(evidence).map_err(|e| e.to_string())?;
    let mut exports = BTreeMap::new();
    for &(backend, contents) in dumps {
        let dump: DescriptorDump = serde_json::from_str(contents).map_err(|e| e.to_string())?;
        if dump.schema_version != 2 || dump.snapshot.backend != backend {
            return Err(format!(
                "unsupported {backend} descriptor schema or backend"
            ));
        }
        if exports.insert(backend, dump).is_some() {
            return Err(format!("duplicate {backend} descriptor"));
        }
    }
    let mut specs = BTreeMap::new();
    for mut product in products {
        let id = product.id.clone();
        if product.downloads.is_empty()
            || product.downloads.iter().any(|download| {
                download.provider != "huggingface"
                    || crate::hf_repo_id::namespaced(&download.repo) != Ok(download.repo.as_str())
                    || download.revision.len() != 40
                    || !download
                        .revision
                        .bytes()
                        .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
            })
        {
            return Err(format!(
                "builtin {id} needs an immutable download repository and revision"
            ));
        }
        let evidence = evidence
            .remove(&id)
            .ok_or_else(|| format!("missing evidence for {id}"))?;
        let mut routes = BTreeMap::new();
        for (&backend, dump) in &exports {
            let Some(provider) = dump.model_mappings.get(&id) else {
                continue;
            };
            let mut matched = dump
                .snapshot
                .generator_capabilities
                .iter()
                .filter(|r| &r.id == provider);
            let route = matched
                .next()
                .ok_or_else(|| format!("missing {backend} descriptor for {id}/{provider}"))?;
            if matched.next().is_some()
                || route.backend != backend
                || route.modality != product.model_type
                || route.family != product.family
            {
                return Err(format!(
                    "inconsistent {backend} descriptor for {id}/{provider}"
                ));
            }
            let lane = evidence
                .get(backend)
                .ok_or_else(|| format!("missing {backend} evidence for {id}"))?;
            if lane
                .memory_strategy_contract
                .get("provider")
                .and_then(|v| v.as_str())
                != Some(provider)
                || lane
                    .calibrations
                    .iter()
                    .flatten()
                    .any(|record| record.get("provider").and_then(|v| v.as_str()) != Some(provider))
            {
                return Err(format!("mismatched evidence provider for {id}/{backend}"));
            }
            routes.insert(backend.to_owned(), route.clone());
        }
        if routes.is_empty() || routes.len() != evidence.len() {
            return Err(format!("descriptor/evidence routes disagree for {id}"));
        }
        // These menus are shared across platforms. 'default' is a product choice resolved
        // by the worker; named methods must be offered on every advertised route.
        product.limits.samplers.retain(|value| {
            value == "default" || routes.values().all(|r| r.samplers.contains(value))
        });
        product.limits.schedulers.retain(|value| {
            value == "default" || routes.values().all(|r| r.schedulers.contains(value))
        });
        if specs
            .insert(
                id.clone(),
                ModelSpec {
                    product,
                    routes,
                    evidence,
                },
            )
            .is_some()
        {
            return Err(format!("duplicate product {id}"));
        }
    }
    if !evidence.is_empty() {
        return Err(format!("evidence without a product: {:?}", evidence.keys()));
    }
    Ok(specs)
}
