//! The SceneWorks MCP tool surface (epic 10231, sc-10233).
//!
//! `SceneWorksMcp` is the rmcp server/service struct: a `#[tool_router]` impl
//! holds one method per MCP tool, and `#[tool_handler]` wires that router into
//! the `ServerHandler` the streamable-HTTP transport serves. Every tool is a
//! thin wrapper over an existing `/api/v1/*` route via [`ApiClient`] — later
//! stories (generate_image, video tools) add methods to the `#[tool_router]`
//! block, nothing else.
//!
//! The catalog endpoints return large manifest-derived objects (multi-KB per
//! model: downloads, footprints, platform notes …). Tools re-shape them into
//! compact JSON an LLM can actually use — ids/names plus the values a job
//! request needs — via the pure `compact_*` mappers below (unit-tested).

use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, ContentBlock, ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router, ErrorData, ServerHandler,
};
use serde_json::{Map, Value};

use crate::api_client::{ApiClient, ApiClientError};

#[derive(Clone)]
pub struct SceneWorksMcp {
    api: ApiClient,
    tool_router: ToolRouter<Self>,
}

impl SceneWorksMcp {
    pub fn new(api: ApiClient) -> Self {
        Self {
            api,
            tool_router: Self::tool_router(),
        }
    }
}

/// Optional filters for `list_loras`, forwarded verbatim to the
/// `GET /api/v1/loras` query params (`LorasQuery` in the API is camelCase).
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ListLorasArgs {
    #[schemars(
        description = "Only return LoRAs compatible with this model family (e.g. \"sdxl\", \"z-image\", \"flux\")."
    )]
    pub model_family: Option<String>,
    #[schemars(
        description = "Also include LoRAs trained/imported in this project (by project id)."
    )]
    pub project_id: Option<String>,
}

#[tool_router]
impl SceneWorksMcp {
    #[tool(
        description = "List SceneWorks projects. Returns [{id, name, createdAt}]; use the id as projectId in other calls."
    )]
    async fn list_projects(&self) -> Result<CallToolResult, ErrorData> {
        let projects = self
            .api
            .get_json("/api/v1/projects", &[])
            .await
            .map_err(api_error)?;
        json_result(compact_projects(&projects))
    }

    #[tool(
        description = "List the generation model catalog. Returns compact entries: id (use as the model for a job), name, family, type (image/video), capabilities, installState, defaults (resolution/steps/guidanceScale/count) and supported resolutions."
    )]
    async fn list_models(&self) -> Result<CallToolResult, ErrorData> {
        let models = self
            .api
            .get_json("/api/v1/models", &[])
            .await
            .map_err(api_error)?;
        json_result(compact_models(&models))
    }

    #[tool(
        description = "List the LoRA adapter catalog (built-in, imported and trained). Returns compact entries: id, name, family, compatibleFamilies, triggerWords, defaultWeight, installState. Optionally filter by model family and/or project."
    )]
    async fn list_loras(
        &self,
        Parameters(args): Parameters<ListLorasArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let loras = self
            .api
            .get_json(
                "/api/v1/loras",
                &[
                    ("modelFamily", args.model_family.as_deref()),
                    ("projectId", args.project_id.as_deref()),
                ],
            )
            .await
            .map_err(api_error)?;
        json_result(compact_loras(&loras))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for SceneWorksMcp {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build()).with_instructions(
            "SceneWorks local generation studio. Use list_projects for project ids, \
             list_models for the generation model catalog (model ids + job defaults), and \
             list_loras for LoRA adapters compatible with a model family.",
        )
    }
}

/// A tool result whose single content block is the compact JSON payload. Plain
/// text-JSON (not `structured_content`) for the widest MCP-client compatibility.
fn json_result(value: Value) -> Result<CallToolResult, ErrorData> {
    Ok(CallToolResult::success(vec![ContentBlock::json(&value)?]))
}

/// Surface an API failure as a JSON-RPC internal error; the Display impl already
/// includes the upstream status + detail, and never the token.
fn api_error(error: ApiClientError) -> ErrorData {
    ErrorData::internal_error(error.to_string(), None)
}

/// Map an API array response item-by-item; anything non-array (defensive — the
/// routes today always return arrays) passes through unchanged so a future shape
/// change degrades to "verbose" rather than "wrong".
fn compact_array(value: &Value, compact_one: impl Fn(&Value) -> Value) -> Value {
    match value.as_array() {
        Some(items) => Value::Array(items.iter().map(compact_one).collect()),
        None => value.clone(),
    }
}

/// Copy the given top-level keys, skipping absent/null ones.
fn copy_keys(item: &Value, keys: &[&str], out: &mut Map<String, Value>) {
    for key in keys {
        if let Some(value) = item.get(*key).filter(|value| !value.is_null()) {
            out.insert((*key).to_owned(), value.clone());
        }
    }
}

pub(crate) fn compact_projects(projects: &Value) -> Value {
    compact_array(projects, |project| {
        let mut out = Map::new();
        copy_keys(project, &["id", "name", "createdAt"], &mut out);
        Value::Object(out)
    })
}

pub(crate) fn compact_models(models: &Value) -> Value {
    compact_array(models, |model| {
        let mut out = Map::new();
        copy_keys(
            model,
            &[
                "id",
                "name",
                "family",
                "type",
                "capabilities",
                "installState",
                "gated",
                "defaults",
            ],
            &mut out,
        );
        // The resolution menu is the one `limits` field a job request needs; the
        // rest (sampler/scheduler menus, counts) stays on the full API response.
        if let Some(resolutions) = model.pointer("/limits/resolutions") {
            out.insert("resolutions".to_owned(), resolutions.clone());
        }
        // Which LoRA families this model accepts — pairs with list_loras.
        if let Some(families) = model.pointer("/loraCompatibility/families") {
            out.insert("loraFamilies".to_owned(), families.clone());
        }
        Value::Object(out)
    })
}

pub(crate) fn compact_loras(loras: &Value) -> Value {
    compact_array(loras, |lora| {
        let mut out = Map::new();
        copy_keys(
            lora,
            &[
                "id",
                "name",
                "family",
                "triggerWords",
                "defaultWeight",
                "installState",
            ],
            &mut out,
        );
        if let Some(families) = lora.pointer("/compatibility/families") {
            out.insert("compatibleFamilies".to_owned(), families.clone());
        }
        Value::Object(out)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn compact_projects_keeps_only_id_name_created_at() {
        let full = json!([{
            "id": "p1",
            "name": "My Film",
            "path": "/data/projects/p1",
            "createdAt": "2026-07-07T00:00:00Z"
        }]);
        assert_eq!(
            compact_projects(&full),
            json!([{ "id": "p1", "name": "My Film", "createdAt": "2026-07-07T00:00:00Z" }])
        );
    }

    #[test]
    fn compact_models_keeps_job_request_fields_and_flattens_menus() {
        let full = json!([{
            "id": "z_image_turbo",
            "name": "Z-Image-Turbo",
            "family": "z-image",
            "type": "image",
            "capabilities": ["text_to_image"],
            "installState": "installed",
            "gated": false,
            "defaults": { "resolution": "1024x1024", "steps": 8, "guidanceScale": 0, "count": 4 },
            "limits": {
                "resolutions": ["768x768", "1024x1024"],
                "samplers": ["default", "euler"]
            },
            "loraCompatibility": { "families": ["z-image"], "types": ["style"] },
            // Verbose catalog fields that must be dropped:
            "downloads": [{ "repo": "SceneWorks/z-image-turbo-mlx", "files": ["q4/*"] }],
            "mlx": { "minMemoryGb": 40 },
            "candle": { "minMemoryGb": 40 }
        }]);
        assert_eq!(
            compact_models(&full),
            json!([{
                "id": "z_image_turbo",
                "name": "Z-Image-Turbo",
                "family": "z-image",
                "type": "image",
                "capabilities": ["text_to_image"],
                "installState": "installed",
                "gated": false,
                "defaults": { "resolution": "1024x1024", "steps": 8, "guidanceScale": 0, "count": 4 },
                "resolutions": ["768x768", "1024x1024"],
                "loraFamilies": ["z-image"]
            }])
        );
    }

    #[test]
    fn compact_loras_keeps_trigger_and_compatibility_fields() {
        let full = json!([{
            "id": "ltx_2_3_ic_hdr",
            "name": "LTX-2.3 IC-LoRA HDR",
            "family": "ltx-video",
            "triggerWords": [],
            "compatibility": { "families": ["ltx-video"] },
            "icLora": true,
            "defaultWeight": 0.8,
            "installState": "missing",
            "source": { "provider": "huggingface", "repo": "Lightricks/x", "file": "y.safetensors" }
        }]);
        assert_eq!(
            compact_loras(&full),
            json!([{
                "id": "ltx_2_3_ic_hdr",
                "name": "LTX-2.3 IC-LoRA HDR",
                "family": "ltx-video",
                "triggerWords": [],
                "defaultWeight": 0.8,
                "installState": "missing",
                "compatibleFamilies": ["ltx-video"]
            }])
        );
    }

    #[test]
    fn compact_mappers_skip_absent_and_null_fields() {
        let sparse = json!([{ "id": "m1", "name": null }]);
        assert_eq!(compact_models(&sparse), json!([{ "id": "m1" }]));
        assert_eq!(compact_loras(&sparse), json!([{ "id": "m1" }]));
    }

    #[test]
    fn compact_mappers_pass_non_arrays_through() {
        // Defensive: an unexpected shape must degrade to verbose, not panic/lie.
        let detail = json!({ "detail": "unexpected" });
        assert_eq!(compact_projects(&detail), detail);
        assert_eq!(compact_models(&detail), detail);
        assert_eq!(compact_loras(&detail), detail);
    }
}
