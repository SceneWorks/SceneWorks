//! YuE2 score tools (sc-22997): bounded, agent-accessible score inspection, editing and listening
//! comparisons, as thin wrappers over the `/api/v1/.../yue2/...` routes.
//!
//! "Bounded" is enforced server-side: the only way to change a score is one operation from the
//! closed `ScoreEditOperation` set (`sceneworks_core::yue2_score::ops`), and every result is
//! re-parsed in the native dialect and invariant-checked against its source before it is stored as
//! a new version. These helpers add only what an MCP caller cannot be trusted to set itself —
//! the agent provenance — plus id validation before any id is spliced into a route.

use rmcp::{model::CallToolResult, schemars, ErrorData};
use serde_json::{json, Value};

use crate::api_client::{ApiClient, ApiClientError};
use crate::server::valid_project_id;

/// Arguments for `yue2_inspect_score`.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Yue2InspectArgs {
    #[schemars(description = "Project id (with versionId) to inspect a stored score version.")]
    pub project_id: Option<String>,
    #[schemars(description = "Score version id from yue2_list_score_versions.")]
    pub version_id: Option<String>,
    #[schemars(
        description = "Raw native YuE2 ABC text to inspect instead of a stored version (max 256 KiB)."
    )]
    pub abc: Option<String>,
}

/// Arguments for the project-scoped listing tools.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Yue2ProjectArgs {
    #[schemars(description = "Project id (from list_projects).")]
    pub project_id: String,
}

/// Arguments for `yue2_get_score_version`.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Yue2VersionArgs {
    #[schemars(description = "Project id (from list_projects).")]
    pub project_id: String,
    #[schemars(description = "Score version id from yue2_list_score_versions.")]
    pub version_id: String,
}

/// Arguments for `yue2_create_score_version`.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Yue2CreateArgs {
    #[schemars(description = "Project id (from list_projects).")]
    pub project_id: String,
    #[schemars(description = "Complete native YuE2 two-voice ABC score (max 256 KiB).")]
    pub abc: String,
    #[schemars(
        description = "Style prompt: genre, instruments, vocal character, language, tempo."
    )]
    pub style: String,
    #[schemars(description = "Lyrics with section tags such as [Verse] and [Chorus].")]
    pub lyrics: String,
    #[schemars(
        description = "\"full\" (melody + chord symbols) or \"melody\" (chord-free score). Default \"full\"."
    )]
    pub cot: Option<String>,
    #[schemars(description = "Generation seed (default 831001, the YuE2 protocol default).")]
    pub seed: Option<u64>,
    #[schemars(description = "Classifier-free guidance scale in [0, 20]; omit for the default.")]
    pub cfg_scale: Option<f64>,
    #[schemars(
        description = "Where the score came from: \"import\" (default), \"plan\" or \"transcription\"."
    )]
    pub origin: Option<String>,
    #[schemars(description = "Your agent name, recorded in the version's provenance.")]
    pub agent_name: Option<String>,
}

/// Arguments for `yue2_edit_score`.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Yue2EditArgs {
    #[schemars(description = "Project id (from list_projects).")]
    pub project_id: String,
    #[schemars(description = "The SOURCE score version to edit; it is never modified.")]
    pub version_id: String,
    #[schemars(
        description = "Exactly one operation object, keyed by \"op\": {\"op\":\"reharmonize\",\"changes\":[{\"bar\":1,\"onsetQuarters\":\"0\",\"chord\":\"Cmaj7\"}]} (chord null removes; harmony-only, splits held notes with ties) | {\"op\":\"strip_chords\",\"keepVoice\":\"both\"|\"Vocal\"|\"Ins\"} (chord-free cover score, cot becomes melody) | {\"op\":\"set_tempo\",\"bpm\":96,\"style\":optional} | {\"op\":\"arrange_sections\",\"sectionOrder\":[0,1,1],\"lyrics\":\"restated lyrics\",\"style\":optional} (0-based source sections; reorder/repeat/drop) | {\"op\":\"set_lyrics\",\"lyrics\":\"...\"} | {\"op\":\"set_style\",\"style\":\"...\"} | {\"op\":\"replace_score\",\"abc\":\"complete edited ABC\",\"allow\":{\"harmony\":bool,\"tempo\":bool,\"melody\":{\"voices\":[\"Vocal\"],\"fromBar\":3,\"toBar\":4}},\"lyrics\":optional,\"style\":optional,\"cot\":optional}. Chords use the native vocabulary only: root + (\"\", m, dim, aug, 7, maj7, m7, dim7, m7b5, sus4, sus2, 6, m6, 7sus4, m(maj7)) + optional /bass."
    )]
    pub operation: Value,
    #[schemars(
        description = "The change brief: what is changed, why, and what must be preserved (required, max 4000 characters). Stored with the version and every render of it."
    )]
    pub brief: String,
    #[schemars(description = "Check the edit and return the would-be version without storing it.")]
    pub dry_run: Option<bool>,
    #[schemars(description = "Your agent name, recorded in the version's provenance.")]
    pub agent_name: Option<String>,
}

/// Arguments for `yue2_compare_versions`.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Yue2CompareArgs {
    #[schemars(description = "Project id (from list_projects).")]
    pub project_id: String,
    #[schemars(description = "Version A (usually the source).")]
    pub version_a: String,
    #[schemars(description = "Version B (usually the edit).")]
    pub version_b: String,
    #[schemars(description = "A completed render of version A to listen to (optional).")]
    pub render_a: Option<String>,
    #[schemars(description = "A completed render of version B to listen to (optional).")]
    pub render_b: Option<String>,
    #[schemars(
        description = "Listening notes: what you actually heard. Do not claim listening you did not do."
    )]
    pub notes: Option<String>,
    #[schemars(description = "Your agent name, recorded in the comparison's provenance.")]
    pub agent_name: Option<String>,
}

fn invalid(message: impl Into<String>) -> ErrorData {
    ErrorData::invalid_params(message.into(), None)
}

fn record_id<'a>(value: &'a str, field: &str) -> Result<&'a str, ErrorData> {
    let value = value.trim();
    if value.is_empty()
        || value.len() > 64
        || !value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
    {
        return Err(invalid(format!(
            "{field} \"{value}\" is not a valid id (letters, digits, '-' or '_')"
        )));
    }
    Ok(value)
}

fn project_id(value: &str) -> Result<&str, ErrorData> {
    let value = value.trim();
    if !valid_project_id(value) {
        return Err(invalid(format!("\"{value}\" is not a valid project id")));
    }
    Ok(value)
}

/// The provenance an MCP call records: always an agent, always via MCP.
pub(crate) fn agent_provenance(agent_name: Option<&str>) -> Value {
    let mut provenance = json!({ "actor": "agent", "channel": "mcp" });
    if let Some(name) = agent_name.map(str::trim).filter(|name| !name.is_empty()) {
        provenance["agentName"] = json!(name);
    }
    provenance
}

/// Route + body for `yue2_inspect_score`: exactly one of `versionId` (+ `projectId`) or `abc`.
pub(crate) fn inspect_target(args: &Yue2InspectArgs) -> Result<(String, Option<Value>), ErrorData> {
    match (&args.project_id, &args.version_id, &args.abc) {
        (Some(project), Some(version), None) => Ok((
            format!(
                "/api/v1/projects/{}/yue2/score-versions/{}/inspection",
                project_id(project)?,
                record_id(version, "versionId")?
            ),
            None,
        )),
        (None, None, Some(abc)) => Ok((
            "/api/v1/yue2/score/inspect".to_owned(),
            Some(json!({ "abc": abc })),
        )),
        _ => Err(invalid(
            "pass either projectId + versionId (a stored version) or abc (raw text), not both",
        )),
    }
}

pub(crate) fn create_body(args: &Yue2CreateArgs) -> Value {
    let mut request = json!({
        "style": args.style,
        "lyrics": args.lyrics,
        "cot": args.cot.as_deref().unwrap_or("full"),
    });
    if let Some(seed) = args.seed {
        request["seed"] = json!(seed);
    }
    if let Some(cfg) = args.cfg_scale {
        request["cfgScale"] = json!(cfg);
    }
    json!({
        "abc": args.abc,
        "request": request,
        "origin": args.origin.as_deref().unwrap_or("import"),
        "provenance": agent_provenance(args.agent_name.as_deref()),
    })
}

pub(crate) fn edit_body(args: &Yue2EditArgs) -> Result<Value, ErrorData> {
    if !args.operation.is_object() || args.operation.get("op").is_none() {
        return Err(invalid(
            "operation must be one object with an \"op\" field naming a supported edit",
        ));
    }
    Ok(json!({
        "operation": args.operation,
        "brief": args.brief,
        "dryRun": args.dry_run.unwrap_or(false),
        "provenance": agent_provenance(args.agent_name.as_deref()),
    }))
}

pub(crate) fn compare_body(args: &Yue2CompareArgs) -> Result<Value, ErrorData> {
    let mut body = json!({
        "versionA": record_id(&args.version_a, "versionA")?,
        "versionB": record_id(&args.version_b, "versionB")?,
        "provenance": agent_provenance(args.agent_name.as_deref()),
    });
    if let Some(render) = &args.render_a {
        body["renderA"] = json!(record_id(render, "renderA")?);
    }
    if let Some(render) = &args.render_b {
        body["renderB"] = json!(record_id(render, "renderB")?);
    }
    if let Some(notes) = &args.notes {
        body["notes"] = json!(notes);
    }
    Ok(body)
}

fn result(value: Value) -> Result<CallToolResult, ErrorData> {
    Ok(CallToolResult::success(vec![
        rmcp::model::ContentBlock::json(&value)?,
    ]))
}

/// A rejected score operation (unsupported notation, an invariant violation, a bad request) is a
/// domain answer the agent must read and act on — an `isError` tool result carrying the API's
/// detail and typed context — not a protocol error. Transport and server faults stay errors.
fn outcome(response: Result<Value, ApiClientError>) -> Result<CallToolResult, ErrorData> {
    match response {
        Ok(value) => result(value),
        Err(ApiClientError::Api { status, detail }) if status.is_client_error() => {
            Ok(CallToolResult::error(vec![
                rmcp::model::ContentBlock::text(format!(
                    "SceneWorks refused the YuE2 score request ({status}): {detail}"
                )),
            ]))
        }
        Err(error) => Err(ErrorData::internal_error(error.to_string(), None)),
    }
}

pub(crate) async fn inspect(
    api: &ApiClient,
    args: Yue2InspectArgs,
) -> Result<CallToolResult, ErrorData> {
    let (path, body) = inspect_target(&args)?;
    outcome(match body {
        Some(body) => api.post_json(&path, &body).await,
        None => api.get_json(&path, &[]).await,
    })
}

pub(crate) async fn list_versions(
    api: &ApiClient,
    args: Yue2ProjectArgs,
) -> Result<CallToolResult, ErrorData> {
    let path = format!(
        "/api/v1/projects/{}/yue2/score-versions",
        project_id(&args.project_id)?
    );
    outcome(api.get_json(&path, &[]).await)
}

pub(crate) async fn get_version(
    api: &ApiClient,
    args: Yue2VersionArgs,
) -> Result<CallToolResult, ErrorData> {
    let path = format!(
        "/api/v1/projects/{}/yue2/score-versions/{}",
        project_id(&args.project_id)?,
        record_id(&args.version_id, "versionId")?
    );
    outcome(api.get_json(&path, &[]).await)
}

pub(crate) async fn create_version(
    api: &ApiClient,
    args: Yue2CreateArgs,
) -> Result<CallToolResult, ErrorData> {
    let path = format!(
        "/api/v1/projects/{}/yue2/score-versions",
        project_id(&args.project_id)?
    );
    outcome(api.post_json(&path, &create_body(&args)).await)
}

pub(crate) async fn edit(api: &ApiClient, args: Yue2EditArgs) -> Result<CallToolResult, ErrorData> {
    let path = format!(
        "/api/v1/projects/{}/yue2/score-versions/{}/edits",
        project_id(&args.project_id)?,
        record_id(&args.version_id, "versionId")?
    );
    outcome(api.post_json(&path, &edit_body(&args)?).await)
}

pub(crate) async fn compare(
    api: &ApiClient,
    args: Yue2CompareArgs,
) -> Result<CallToolResult, ErrorData> {
    let path = format!(
        "/api/v1/projects/{}/yue2/comparisons",
        project_id(&args.project_id)?
    );
    outcome(api.post_json(&path, &compare_body(&args)?).await)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provenance_is_always_agent_via_mcp() {
        assert_eq!(
            agent_provenance(Some(" claude ")),
            json!({"actor": "agent", "channel": "mcp", "agentName": "claude"})
        );
        assert_eq!(
            agent_provenance(Some("  ")),
            json!({"actor": "agent", "channel": "mcp"})
        );
    }

    #[test]
    fn inspect_takes_exactly_one_target() {
        let args =
            |project: Option<&str>, version: Option<&str>, abc: Option<&str>| Yue2InspectArgs {
                project_id: project.map(str::to_owned),
                version_id: version.map(str::to_owned),
                abc: abc.map(str::to_owned),
            };
        let (path, body) = inspect_target(&args(Some("p1"), Some("yue2v_ab"), None)).unwrap();
        assert_eq!(
            path,
            "/api/v1/projects/p1/yue2/score-versions/yue2v_ab/inspection"
        );
        assert!(body.is_none());
        let (path, body) = inspect_target(&args(None, None, Some("X:1"))).unwrap();
        assert_eq!(path, "/api/v1/yue2/score/inspect");
        assert_eq!(body.unwrap()["abc"], "X:1");
        assert!(inspect_target(&args(Some("p1"), Some("v"), Some("X:1"))).is_err());
        assert!(inspect_target(&args(None, None, None)).is_err());
        assert!(inspect_target(&args(Some("../x"), Some("v"), None)).is_err());
        assert!(inspect_target(&args(Some("p1"), Some("v/../../x"), None)).is_err());
    }

    #[test]
    fn edit_body_requires_one_operation_object() {
        let args = |operation: Value| Yue2EditArgs {
            project_id: "p1".into(),
            version_id: "yue2v_1".into(),
            operation,
            brief: "b".into(),
            dry_run: None,
            agent_name: Some("claude".into()),
        };
        assert!(edit_body(&args(json!("reharmonize"))).is_err());
        assert!(edit_body(&args(json!({"changes": []}))).is_err());
        let body = edit_body(&args(json!({"op": "set_tempo", "bpm": 96}))).unwrap();
        assert_eq!(body["dryRun"], false);
        assert_eq!(body["provenance"]["actor"], "agent");
        assert_eq!(body["operation"]["op"], "set_tempo");
    }

    #[test]
    fn create_body_defaults_to_an_imported_full_score() {
        let body = create_body(&Yue2CreateArgs {
            project_id: "p1".into(),
            abc: "X:1".into(),
            style: "pop".into(),
            lyrics: "[Verse]".into(),
            cot: None,
            seed: None,
            cfg_scale: None,
            origin: None,
            agent_name: None,
        });
        assert_eq!(body["origin"], "import");
        assert_eq!(body["request"]["cot"], "full");
        assert!(
            body["request"].get("seed").is_none(),
            "the API owns the default seed"
        );
    }

    #[test]
    fn compare_body_validates_every_id() {
        let args = Yue2CompareArgs {
            project_id: "p1".into(),
            version_a: "yue2v_a".into(),
            version_b: "yue2v_b".into(),
            render_a: Some("yue2r_a?x=1".into()),
            render_b: None,
            notes: None,
            agent_name: None,
        };
        assert!(compare_body(&args).is_err());
    }
}
