//! The HTTP error type for the rust-api handlers.
//!
//! `ApiError` carries an HTTP status plus a client-facing `detail` string and
//! renders itself as a `{ "detail": ... }` JSON body. It is the single error
//! type every handler returns, with `From` conversions for the two store error
//! families so `?` propagates cleanly. Extracted from `lib.rs` (sc-8890, F-088)
//! so the crate root no longer owns the error type inline.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

use sceneworks_core::catalog_store::CatalogError;
use sceneworks_core::jobs_store::JobsStoreError;
use sceneworks_core::model_artifacts::external_library::ExternalLibraryUnavailableContext;
use sceneworks_core::project_store::ProjectStoreError;

#[derive(Debug)]
pub(crate) struct ApiError {
    pub(crate) status: StatusCode,
    pub(crate) detail: String,
    pub(crate) code: Option<&'static str>,
    /// Typed machine-readable context for the codes whose client must ACT on the rejection rather
    /// than print it (sc-19709). Rendered as a `context` object beside `detail`/`code`, so a client
    /// drives its recovery UI from named fields instead of parsing the detail sentence.
    pub(crate) context: Option<serde_json::Value>,
}

impl ApiError {
    pub(crate) fn bad_request(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            detail: detail.into(),
            context: None,
            code: None,
        }
    }

    pub(crate) fn unauthorized(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            detail: detail.into(),
            context: None,
            code: None,
        }
    }

    pub(crate) fn forbidden(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            detail: detail.into(),
            context: None,
            code: None,
        }
    }

    pub(crate) fn payload_too_large(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::PAYLOAD_TOO_LARGE,
            detail: detail.into(),
            context: None,
            code: None,
        }
    }

    pub(crate) fn conflict(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::CONFLICT,
            detail: detail.into(),
            context: None,
            code: None,
        }
    }

    pub(crate) fn service_unavailable(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            detail: detail.into(),
            context: None,
            code: Some("catalog_preflight_unavailable"),
        }
    }

    /// The typed "Installed — external library unavailable" submission-preflight rejection
    /// (sc-19708). 503 because the installation is intact and the request is well-formed: the
    /// operator reconnects the library and retries.
    ///
    /// The `context` payload (sc-19709) is what makes the rejection actionable: the desktop prompt
    /// names the model and the expected library location from those fields, so no client ever
    /// parses `detail` and no raw filesystem error reaches a user.
    pub(crate) fn external_model_library_unavailable(
        detail: impl Into<String>,
        context: &ExternalLibraryUnavailableContext,
    ) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            detail: detail.into(),
            context: serde_json::to_value(context).ok(),
            code: Some(
                sceneworks_core::model_artifacts::external_library::EXTERNAL_LIBRARY_UNAVAILABLE_CODE,
            ),
        }
    }

    /// A typed rejection whose client must act on named fields (the model-library relocation
    /// rejections, sc-19709). Status is the caller's: relocation rejections are 409 because the
    /// request is well-formed but the named library cannot be adopted.
    pub(crate) fn typed(
        status: StatusCode,
        detail: impl Into<String>,
        code: &'static str,
        context: serde_json::Value,
    ) -> Self {
        Self {
            status,
            detail: detail.into(),
            context: Some(context),
            code: Some(code),
        }
    }

    pub(crate) fn model_artifact_conflict(detail: impl Into<String>, code: &'static str) -> Self {
        Self {
            status: StatusCode::CONFLICT,
            detail: detail.into(),
            context: None,
            code: Some(code),
        }
    }

    pub(crate) fn internal(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            detail: detail.into(),
            context: None,
            code: None,
        }
    }

    fn database_locked(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            detail: detail.into(),
            context: None,
            code: Some("database_locked"),
        }
    }
}

impl From<JobsStoreError> for ApiError {
    fn from(error: JobsStoreError) -> Self {
        if error.is_database_locked() {
            return Self::database_locked(error.to_string());
        }
        match error {
            JobsStoreError::NotFound(_) => Self {
                status: StatusCode::NOT_FOUND,
                detail: "Record not found".to_owned(),
                context: None,
                code: None,
            },
            JobsStoreError::InvalidStatus(status) => Self {
                status: StatusCode::BAD_REQUEST,
                detail: format!("Unsupported job status: {status}"),
                context: None,
                code: None,
            },
            JobsStoreError::InvalidNumber(field) => {
                Self::bad_request(format!("Invalid numeric value for {field}"))
            }
            JobsStoreError::InvalidRequestedGpu(detail) => Self::bad_request(detail),
            JobsStoreError::RetryLimit { max_attempts } => Self {
                status: StatusCode::BAD_REQUEST,
                detail: format!("Job retry limit reached after {max_attempts} attempts."),
                context: None,
                code: None,
            },
            // 409 tells the worker its report lost a race with cancel/sweep/
            // reclaim: abandon the job instead of retrying (sc-4172).
            JobsStoreError::TerminalJobImmutable { job_id, status } => Self {
                status: StatusCode::CONFLICT,
                detail: format!(
                    "Job {job_id} is already {status}; terminal jobs cannot be updated."
                ),
                context: None,
                code: None,
            },
            JobsStoreError::NotJobOwner { job_id } => Self {
                status: StatusCode::CONFLICT,
                detail: format!(
                    "Progress rejected: the reporting worker no longer owns job {job_id}."
                ),
                context: None,
                code: None,
            },
            other => Self::internal(other.to_string()),
        }
    }
}

impl From<ProjectStoreError> for ApiError {
    fn from(error: ProjectStoreError) -> Self {
        match error {
            ProjectStoreError::BadRequest(detail) => Self::bad_request(detail),
            ProjectStoreError::NotFound(detail) => Self {
                status: StatusCode::NOT_FOUND,
                detail,
                context: None,
                code: None,
            },
            // A non-writable workspace folder is an environment problem, not a bad
            // request — 507 keeps the actionable, path-naming detail intact and out
            // of the 4xx validation bucket, while still logging server-side for
            // diagnosis (issue #1435 / sc-11855).
            ProjectStoreError::StorageNotWritable(detail) => Self {
                status: StatusCode::INSUFFICIENT_STORAGE,
                detail,
                context: None,
                code: None,
            },
            other => Self::internal(other.to_string()),
        }
    }
}

impl From<CatalogError> for ApiError {
    fn from(error: CatalogError) -> Self {
        match error {
            CatalogError::InvalidCatalog(_) => Self {
                status: StatusCode::BAD_REQUEST,
                detail: "Invalid catalog request".to_owned(),
                context: None,
                code: Some("catalog_invalid"),
            },
            CatalogError::NotFound(_) => Self {
                status: StatusCode::NOT_FOUND,
                detail: "Attached catalog not found".to_owned(),
                context: None,
                code: Some("catalog_not_found"),
            },
            CatalogError::AlreadyExists(_) => Self {
                status: StatusCode::CONFLICT,
                detail: "Catalog location is not empty or is already in use".to_owned(),
                context: None,
                code: Some("catalog_already_exists"),
            },
            CatalogError::Conflict(_) => Self {
                status: StatusCode::CONFLICT,
                detail: "Catalog processing state changed or is active".to_owned(),
                context: None,
                code: Some("catalog_processing_conflict"),
            },
            CatalogError::Incompatible { .. } => Self {
                status: StatusCode::CONFLICT,
                detail: "Catalog format is newer than this SceneWorks version supports".to_owned(),
                context: None,
                code: Some("catalog_incompatible"),
            },
            CatalogError::Corrupt { .. } => Self {
                status: StatusCode::UNPROCESSABLE_ENTITY,
                detail: "Catalog files are invalid or corrupt".to_owned(),
                context: None,
                code: Some("catalog_corrupt"),
            },
            CatalogError::Io(_) | CatalogError::Sqlite(_) | CatalogError::Json(_) => Self {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                detail: "Catalog operation failed".to_owned(),
                context: None,
                code: Some("catalog_operation_failed"),
            },
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        // Make every 5xx leave a server-side trace (it previously returned `{detail}`
        // to the client and logged nothing). Expected/normal typed 4xx domain errors
        // stay at debug to avoid drowning the error level in routine validation noise.
        if self.status.is_server_error() {
            tracing::error!(
                event = "api_error",
                status = self.status.as_u16(),
                detail = %self.detail,
                "API request failed"
            );
        } else if self.status.is_client_error() {
            tracing::debug!(
                event = "api_error",
                status = self.status.as_u16(),
                detail = %self.detail,
            );
        }
        let mut body = match self.code {
            Some(code) => json!({ "detail": self.detail, "code": code }),
            None => json!({ "detail": self.detail }),
        };
        if let (Some(object), Some(context)) = (body.as_object_mut(), self.context) {
            object.insert("context".to_owned(), context);
        }
        (self.status, Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sqlite_lock_maps_to_machine_readable_api_code() {
        let sqlite = rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_BUSY),
            Some("wording is deliberately irrelevant".to_owned()),
        );
        let error = ApiError::from(JobsStoreError::Sqlite(sqlite));
        assert_eq!(error.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(error.code, Some("database_locked"));
    }
}
