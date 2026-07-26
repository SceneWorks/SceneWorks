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

use sceneworks_core::jobs_store::JobsStoreError;
use sceneworks_core::project_store::ProjectStoreError;

#[derive(Debug)]
pub(crate) struct ApiError {
    pub(crate) status: StatusCode,
    pub(crate) detail: String,
    pub(crate) code: Option<&'static str>,
}

impl ApiError {
    pub(crate) fn bad_request(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            detail: detail.into(),
            code: None,
        }
    }

    pub(crate) fn unauthorized(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            detail: detail.into(),
            code: None,
        }
    }

    pub(crate) fn forbidden(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            detail: detail.into(),
            code: None,
        }
    }

    pub(crate) fn payload_too_large(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::PAYLOAD_TOO_LARGE,
            detail: detail.into(),
            code: None,
        }
    }

    pub(crate) fn conflict(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::CONFLICT,
            detail: detail.into(),
            code: None,
        }
    }

    pub(crate) fn internal(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            detail: detail.into(),
            code: None,
        }
    }

    fn database_locked(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            detail: detail.into(),
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
                code: None,
            },
            JobsStoreError::InvalidStatus(status) => Self {
                status: StatusCode::BAD_REQUEST,
                detail: format!("Unsupported job status: {status}"),
                code: None,
            },
            JobsStoreError::InvalidNumber(field) => {
                Self::bad_request(format!("Invalid numeric value for {field}"))
            }
            JobsStoreError::InvalidRequestedGpu(detail) => Self::bad_request(detail),
            JobsStoreError::RetryLimit { max_attempts } => Self {
                status: StatusCode::BAD_REQUEST,
                detail: format!("Job retry limit reached after {max_attempts} attempts."),
                code: None,
            },
            // 409 tells the worker its report lost a race with cancel/sweep/
            // reclaim: abandon the job instead of retrying (sc-4172).
            JobsStoreError::TerminalJobImmutable { job_id, status } => Self {
                status: StatusCode::CONFLICT,
                detail: format!(
                    "Job {job_id} is already {status}; terminal jobs cannot be updated."
                ),
                code: None,
            },
            JobsStoreError::NotJobOwner { job_id } => Self {
                status: StatusCode::CONFLICT,
                detail: format!(
                    "Progress rejected: the reporting worker no longer owns job {job_id}."
                ),
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
                code: None,
            },
            // A non-writable workspace folder is an environment problem, not a bad
            // request — 507 keeps the actionable, path-naming detail intact and out
            // of the 4xx validation bucket, while still logging server-side for
            // diagnosis (issue #1435 / sc-11855).
            ProjectStoreError::StorageNotWritable(detail) => Self {
                status: StatusCode::INSUFFICIENT_STORAGE,
                detail,
                code: None,
            },
            other => Self::internal(other.to_string()),
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
        let body = match self.code {
            Some(code) => json!({ "detail": self.detail, "code": code }),
            None => json!({ "detail": self.detail }),
        };
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
