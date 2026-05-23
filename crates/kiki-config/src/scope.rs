//! Scopes (the three tiers) and revisioned values.

use serde::{Deserialize, Serialize};

/// Monotonic revision number scoped to (user_id, scope, key).
///
/// Wraps `u64` so the type system prevents accidentally treating an unrelated
/// integer as a revision in CAS calls.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct Revision(pub u64);

impl Revision {
    pub const ZERO: Revision = Revision(0);

    pub fn next(self) -> Revision {
        Revision(self.0.saturating_add(1))
    }
}

/// The three storage tiers. Determines crypto + endpoint shape.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    /// E2E-encrypted on the client. API keys, OAuth tokens, billing tokens.
    Secrets,
    /// Server-side at-rest encryption. Default model, control mode, theme.
    Preferences,
    /// Server-side at-rest encryption + append-only audit. Which apps have which permissions.
    Capabilities,
}

impl Scope {
    pub fn as_path(self) -> &'static str {
        match self {
            Scope::Secrets => "secrets",
            Scope::Preferences => "preferences",
            Scope::Capabilities => "capabilities",
        }
    }
}

/// A value-at-revision, regardless of scope. `payload` carries the
/// scope-specific shape: ciphertext for `Secrets`, JSON for the others.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopedValue {
    pub key:      String,
    pub revision: Revision,
    pub payload:  serde_json::Value,
}
