//! Device-flow authentication for fleet enrollment.
//!
//! On first boot a node has no credential, so the fleet worker would register
//! it under the "system" placeholder owner and it would never surface in a
//! user's authenticated fleet view. To bind the node to a real account we run
//! the OAuth2 Device Authorization Grant (RFC 8628) against the auth worker:
//! the node prints a short `user_code` + verification URL to the console, the
//! user approves it in a browser, and the node receives a Bearer token which it
//! persists and uses for all subsequent fleet calls.
//!
//! Endpoints (auth worker): `POST /api/device/authorize`, `POST /api/device/token`.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AuthError {
    #[error("network error: {0}")]
    Network(#[from] reqwest::Error),
    #[error("device code expired before approval")]
    ExpiredToken,
    #[error("access denied by user")]
    AccessDenied,
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("auth server error: {0}")]
    Server(String),
}

/// Response from `POST /api/device/authorize`.
#[derive(Debug, Clone, Deserialize)]
pub struct DeviceCode {
    pub device_code:      String,
    pub user_code:        String,
    pub verification_uri: String,
    pub expires_in:       u64,
    pub interval:         u64,
}

enum PollOutcome {
    Pending,
    Authorized { access_token: String },
}

/// Device-flow client against the auth worker.
#[derive(Clone)]
pub struct DeviceFlow {
    auth_url: String,
    http:     reqwest::Client,
}

impl DeviceFlow {
    pub fn new(auth_url: impl Into<String>) -> Self {
        Self { auth_url: auth_url.into(), http: reqwest::Client::new() }
    }

    pub async fn start(&self, node_label: Option<&str>) -> Result<DeviceCode, AuthError> {
        #[derive(Serialize)]
        struct Req<'a> {
            #[serde(skip_serializing_if = "Option::is_none")]
            node_label: Option<&'a str>,
        }
        let url  = format!("{}/api/device/authorize", self.auth_url);
        let resp = self.http.post(&url).json(&Req { node_label })
            .send().await?.error_for_status()?;
        Ok(resp.json().await?)
    }

    async fn poll_once(&self, device_code: &str) -> Result<PollOutcome, AuthError> {
        #[derive(Serialize)]
        struct Req<'a> { device_code: &'a str }
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Resp {
            Ok  { access_token: String },
            Err { error: String },
        }
        let url    = format!("{}/api/device/token", self.auth_url);
        let parsed: Resp = self.http.post(&url).json(&Req { device_code })
            .send().await?.json().await?;
        match parsed {
            Resp::Ok { access_token }  => Ok(PollOutcome::Authorized { access_token }),
            Resp::Err { error } => match error.as_str() {
                "authorization_pending" | "slow_down" => Ok(PollOutcome::Pending),
                "access_denied"  => Err(AuthError::AccessDenied),
                "expired_token"  => Err(AuthError::ExpiredToken),
                other            => Err(AuthError::Server(other.to_owned())),
            },
        }
    }

    /// Run the full flow to completion: start, print the code, poll until the
    /// user approves (or the code expires). Returns the Bearer access token.
    pub async fn authorize(&self, node_label: Option<&str>) -> Result<String, AuthError> {
        let code = self.start(node_label).await?;
        // The verification prompt is operator-facing — print it unconditionally.
        println!(
            "\n┌─ Kiki OS enrollment ─────────────────────────────\n\
             │  Open: {}\n\
             │  Enter code: {}\n\
             └──────────────────────────────────────────────────\n",
            code.verification_uri, code.user_code,
        );
        tracing::info!(user_code = %code.user_code, uri = %code.verification_uri,
            "awaiting device approval");

        let interval = Duration::from_secs(code.interval.max(1));
        loop {
            tokio::time::sleep(interval).await;
            match self.poll_once(&code.device_code).await? {
                PollOutcome::Pending => continue,
                PollOutcome::Authorized { access_token } => {
                    tracing::info!("device authorized — node enrolled");
                    return Ok(access_token);
                }
            }
        }
    }
}

/// Persisted credential store: a single Bearer token at `path` (mode 0600).
pub struct TokenStore {
    path: PathBuf,
}

impl TokenStore {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// Load a previously persisted token, if any.
    pub fn load(&self) -> Option<String> {
        let raw = std::fs::read_to_string(&self.path).ok()?;
        let t = raw.trim();
        if t.is_empty() { None } else { Some(t.to_string()) }
    }

    /// Persist a token, creating parent dirs and restricting permissions.
    pub fn save(&self, token: &str) -> Result<(), AuthError> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&self.path, token)?;
        restrict_permissions(&self.path)?;
        Ok(())
    }
}

#[cfg(unix)]
fn restrict_permissions(path: &Path) -> Result<(), AuthError> {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = std::fs::metadata(path)?.permissions();
    perms.set_mode(0o600);
    std::fs::set_permissions(path, perms)?;
    Ok(())
}

#[cfg(not(unix))]
fn restrict_permissions(_path: &Path) -> Result<(), AuthError> { Ok(()) }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_store_roundtrip() {
        let dir = std::env::temp_dir().join(format!("kiki-tok-{}", std::process::id()));
        let store = TokenStore::new(dir.join("token"));
        assert!(store.load().is_none());
        store.save("abc.def.ghi").unwrap();
        assert_eq!(store.load().as_deref(), Some("abc.def.ghi"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
