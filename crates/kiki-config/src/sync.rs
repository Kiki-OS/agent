//! Pull-on-boot + push-on-change sync engine against the kiki-cloud config API.
//!
//! Endpoints (mirrored in `kiki-cloud/workers/kiki-beta-api/src/routes/config.ts`):
//!
//! ```text
//! GET    /v1/config/{scope}?since=<rev>     → list updates since rev
//! GET    /v1/config/{scope}/{key}           → single value at current rev
//! POST   /v1/config/{scope}/{key}           → CAS write (if_match: <rev>)
//! DELETE /v1/config/{scope}/{key}           → CAS delete (if_match: <rev>)
//! GET    /v1/config/{scope}/_kdf            → KDF params for current user (Secrets only)
//! ```
//!
//! - `Secrets` carry `Sealed` payloads — the server never sees plaintext.
//! - `Preferences` and `Capabilities` are JSON values, server-side at-rest encrypted.
//! - All writes are CAS by `if_match: revision`. On conflict the server
//!   returns 409 with the current `revision` and the caller refetches.
//!
//! Conflict resolution:
//! - `Preferences`: last-writer-wins. Caller may still re-fetch + merge before retry.
//! - `Capabilities`: server-side reconciliation via audit log; client only retries.
//! - `Secrets`: caller MUST re-fetch, decrypt, merge in app-layer, re-encrypt, retry.

use std::time::Duration;

use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::crypto::{KdfParams, Sealed};
use crate::scope::{Revision, Scope, ScopedValue};

const USER_AGENT: &str = concat!("kiki-config/", env!("CARGO_PKG_VERSION"));
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Where to reach the config API and how to authenticate.
#[derive(Clone)]
pub struct ConfigEndpoint {
    pub base_url:     String,
    pub bearer_token: String,
}

pub struct SyncClient {
    http:     Client,
    endpoint: ConfigEndpoint,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListResponse {
    pub values:        Vec<ScopedValue>,
    /// Highest revision in this page. Caller persists and uses as `since`
    /// on next pull.
    pub head_revision: Revision,
}

#[derive(Debug, Clone, Serialize)]
struct WriteRequest<'a> {
    payload: &'a serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WriteResponse {
    pub revision: Revision,
}

#[derive(Debug, Error)]
pub enum SyncError {
    #[error("http: {0}")]
    Http(#[from] reqwest::Error),
    #[error("server returned {0}")]
    Status(StatusCode),
    #[error("CAS conflict: current revision is {current:?}")]
    Conflict { current: Revision },
    #[error("not found: {0}")]
    NotFound(String),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

impl SyncClient {
    pub fn new(endpoint: ConfigEndpoint) -> Result<Self, SyncError> {
        let http = Client::builder()
            .user_agent(USER_AGENT)
            .timeout(DEFAULT_TIMEOUT)
            .build()?;
        Ok(Self { http, endpoint })
    }

    /// Fetch the user's Argon2id parameters. Required to derive the master key.
    pub async fn fetch_kdf_params(&self) -> Result<KdfParams, SyncError> {
        let url = format!("{}/v1/config/secrets/_kdf", self.endpoint.base_url);
        let resp = self.bearer(self.http.get(&url)).send().await?;
        if resp.status() == StatusCode::NOT_FOUND {
            return Err(SyncError::NotFound("_kdf".into()));
        }
        if !resp.status().is_success() {
            return Err(SyncError::Status(resp.status()));
        }
        Ok(resp.json().await?)
    }

    /// Pull all values changed since `since` for the given scope.
    pub async fn pull_since(
        &self,
        scope: Scope,
        since: Revision,
    ) -> Result<ListResponse, SyncError> {
        let url = format!(
            "{}/v1/config/{}?since={}",
            self.endpoint.base_url,
            scope.as_path(),
            since.0
        );
        let resp = self.bearer(self.http.get(&url)).send().await?;
        if !resp.status().is_success() {
            return Err(SyncError::Status(resp.status()));
        }
        Ok(resp.json().await?)
    }

    /// CAS write a Secret. `payload` must be a [`Sealed`] blob produced by
    /// [`crate::crypto::seal_secret`].
    pub async fn put_secret(
        &self,
        key: &str,
        sealed: &Sealed,
        if_match: Revision,
    ) -> Result<Revision, SyncError> {
        self.put_raw(Scope::Secrets, key, serde_json::to_value(sealed)?, if_match)
            .await
    }

    /// CAS write a Preference (plaintext JSON).
    pub async fn put_preference(
        &self,
        key: &str,
        value: serde_json::Value,
        if_match: Revision,
    ) -> Result<Revision, SyncError> {
        self.put_raw(Scope::Preferences, key, value, if_match).await
    }

    /// CAS write a Capability grant. The server adds an audit entry; this call
    /// requires the device session to be authorized (enforced server-side).
    pub async fn put_capability(
        &self,
        key: &str,
        value: serde_json::Value,
        if_match: Revision,
    ) -> Result<Revision, SyncError> {
        self.put_raw(Scope::Capabilities, key, value, if_match).await
    }

    async fn put_raw(
        &self,
        scope: Scope,
        key: &str,
        payload: serde_json::Value,
        if_match: Revision,
    ) -> Result<Revision, SyncError> {
        let url = format!(
            "{}/v1/config/{}/{}",
            self.endpoint.base_url,
            scope.as_path(),
            url_escape(key)
        );
        let resp = self
            .bearer(self.http.post(&url))
            .header("if-match", if_match.0.to_string())
            .json(&WriteRequest { payload: &payload })
            .send()
            .await?;
        match resp.status() {
            s if s.is_success() => {
                let body: WriteResponse = resp.json().await?;
                Ok(body.revision)
            }
            StatusCode::PRECONDITION_FAILED | StatusCode::CONFLICT => {
                // The server SHOULD return the current revision in the body.
                let body: WriteResponse = resp.json().await.unwrap_or(WriteResponse {
                    revision: Revision::ZERO,
                });
                Err(SyncError::Conflict { current: body.revision })
            }
            s => Err(SyncError::Status(s)),
        }
    }

    /// CAS delete a value.
    pub async fn delete(
        &self,
        scope: Scope,
        key: &str,
        if_match: Revision,
    ) -> Result<(), SyncError> {
        let url = format!(
            "{}/v1/config/{}/{}",
            self.endpoint.base_url,
            scope.as_path(),
            url_escape(key)
        );
        let resp = self
            .bearer(self.http.delete(&url))
            .header("if-match", if_match.0.to_string())
            .send()
            .await?;
        match resp.status() {
            s if s.is_success() => Ok(()),
            StatusCode::PRECONDITION_FAILED | StatusCode::CONFLICT => {
                let body: WriteResponse = resp.json().await.unwrap_or(WriteResponse {
                    revision: Revision::ZERO,
                });
                Err(SyncError::Conflict { current: body.revision })
            }
            StatusCode::NOT_FOUND => Err(SyncError::NotFound(key.to_string())),
            s => Err(SyncError::Status(s)),
        }
    }

    fn bearer(&self, b: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        b.bearer_auth(&self.endpoint.bearer_token)
    }
}

fn url_escape(s: &str) -> String {
    // Simple percent-encode for path segments. We don't pull `urlencoding`
    // just for this; keys are well-bounded dotted identifiers in practice.
    s.bytes()
        .flat_map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                vec![b]
            }
            _ => format!("%{b:02X}").into_bytes(),
        })
        .map(|b| b as char)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_escape_passes_safe_chars() {
        assert_eq!(url_escape("openai.api_key"), "openai.api_key");
        assert_eq!(url_escape("a/b"), "a%2Fb");
        assert_eq!(url_escape("hello world"), "hello%20world");
    }
}
