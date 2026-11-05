//! ProjectStore errors aggregate.

use super::*;

impl std::fmt::Display for ProjectStoreError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "{error}"),
            Self::Sqlite(error) => write!(formatter, "{error}"),
            Self::Json(error) => write!(formatter, "{error}"),
            Self::BadRequest(detail) => write!(formatter, "{detail}"),
            Self::Conflict(detail) => write!(formatter, "{detail}"),
            Self::NotFound(detail) => write!(formatter, "{detail}"),
            Self::TimelineConflict { code, context } => write!(formatter, "{code}: {context}"),
            Self::StorageNotWritable(detail) => write!(formatter, "{detail}"),
            Self::FieldInvalid { detail, .. } => write!(formatter, "{detail}"),
        }
    }
}

impl std::error::Error for ProjectStoreError {}

impl From<crate::media_convert::IngestImageError> for ProjectStoreError {
    fn from(error: crate::media_convert::IngestImageError) -> Self {
        match error {
            crate::media_convert::IngestImageError::Io(error) => Self::Io(error),
            error => Self::BadRequest(error.to_string()),
        }
    }
}

impl From<std::io::Error> for ProjectStoreError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<rusqlite::Error> for ProjectStoreError {
    fn from(error: rusqlite::Error) -> Self {
        Self::Sqlite(error)
    }
}

impl From<serde_json::Error> for ProjectStoreError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}
