use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("capability denied: {0:?}")]
    CapabilityDenied(crate::capability::Capability),

    #[error("capability denied: {0}")]
    CapabilityDeniedByName(String),

    /// Capability was denied by policy but proceeded because BypassPermissions
    /// is active. Returned as an informational result, not a hard failure.
    #[error("capability bypassed (audit): {0:?}")]
    CapabilityBypassed(crate::capability::Capability),

    #[error("tool not found: {0}")]
    ToolNotFound(String),

    #[error("tool execution failed: {0}")]
    ToolExecution(String),

    #[error("provider error: {0}")]
    Provider(String),

    #[error("state backend error: {0}")]
    State(String),

    #[error("sandbox error: {0}")]
    Sandbox(String),

    #[error("migration error: {0}")]
    Migration(String),

    #[error("fleet error: {0}")]
    Fleet(String),

    #[error("parse error: {0}")]
    Parse(String),

    #[error("transport error: {0}")]
    Transport(String),

    #[error("i/o error: {0}")]
    Io(String),

    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self { Self::Io(e.to_string()) }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
