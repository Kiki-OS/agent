//! HTTP + WebSocket client for the kiki-cloud vault worker.
//!
//! Endpoints (mirrored in `kiki-cloud/workers/kiki-beta-vault`):
//!
//! ```text
//! GET    /v1/vault/{scope}/{owner}/{path}                 → blob bytes + headers
//! HEAD   /v1/vault/{scope}/{owner}/{path}                 → BlobMeta only
//! PUT    /v1/vault/{scope}/{owner}/{path}  if-match: <rev>→ CAS write
//! DELETE /v1/vault/{scope}/{owner}/{path}  if-match: <rev>→ CAS delete
//! GET    /v1/vault/{scope}/{owner}?prefix=...&since=<rev> → list
//! POST   /v1/vault/{scope}/{owner}/_token                 → capability token
//! GET    /v1/vault/{scope}/{owner}/_acl/{path}            → ACL inspect
//! WS     /v1/vault/{scope}/{owner}/_watch?glob=...        → live events
//! ```

use std::sync::Arc;
use std::time::Duration;

use futures::{SinkExt, StreamExt};
use reqwest::header::{HeaderMap, HeaderValue, IF_MATCH};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::mpsc;

use crate::cache::{CacheError, LocalCache};
use crate::uri::{glob_match, VaultUri};

const USER_AGENT: &str = concat!("kiki-vault/", env!("CARGO_PKG_VERSION"));
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct Revision(pub u64);

impl Revision {
    pub const ZERO: Revision = Revision(0);
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlobMeta {
    pub sha256:    String,
    pub size:      u64,
    pub revision:  Revision,
    pub mime_type: Option<String>,
    pub modified_at_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListEntry {
    pub path: String,
    pub meta: BlobMeta,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Acl {
    pub readers: Vec<String>,
    pub writers: Vec<String>,
    pub watch:   Vec<String>,
}

/// Capability token returned by the server. JWT-encoded; the client treats it
/// as opaque and forwards in `authorization`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityToken {
    pub jwt:           String,
    pub paths_allowed: Vec<String>,
    pub expires_at_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum VaultEvent {
    Put { path: String, meta: BlobMeta },
    Delete { path: String, revision: Revision },
}

#[derive(Debug, Error)]
pub enum VaultError {
    #[error("http: {0}")]
    Http(#[from] reqwest::Error),
    #[error("server returned {0}")]
    Status(StatusCode),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("CAS conflict: current revision is {current:?}")]
    Conflict { current: Revision },
    #[error("permission denied for path `{0}`")]
    Forbidden(String),
    #[error("capability token does not allow `{0}`")]
    CapabilityScope(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("cache: {0}")]
    Cache(#[from] CacheError),
    #[error("ws: {0}")]
    Ws(String),
    #[error("missing meta header `{0}`")]
    MissingHeader(&'static str),
}

#[derive(Clone)]
pub struct VaultClient {
    http:     Client,
    base_url: Arc<str>,
    token:    Arc<CapabilityToken>,
    cache:    Option<Arc<LocalCache>>,
}

impl VaultClient {
    pub fn new(
        base_url: impl Into<String>,
        token: CapabilityToken,
        cache: Option<Arc<LocalCache>>,
    ) -> Result<Self, VaultError> {
        let http = Client::builder()
            .user_agent(USER_AGENT)
            .timeout(DEFAULT_TIMEOUT)
            .build()?;
        Ok(Self {
            http,
            base_url: base_url.into().into(),
            token: Arc::new(token),
            cache,
        })
    }

    fn ensure_allowed(&self, uri: &VaultUri) -> Result<(), VaultError> {
        if self
            .token
            .paths_allowed
            .iter()
            .any(|pat| glob_match(pat, &uri.path))
        {
            Ok(())
        } else {
            Err(VaultError::CapabilityScope(uri.to_string()))
        }
    }

    fn auth_headers(&self) -> HeaderMap {
        let mut h = HeaderMap::new();
        if let Ok(v) = HeaderValue::from_str(&format!("Bearer {}", self.token.jwt)) {
            h.insert(reqwest::header::AUTHORIZATION, v);
        }
        h
    }

    fn url(&self, uri: &VaultUri) -> String {
        format!(
            "{}/v1/vault/{}/{}/{}",
            self.base_url,
            uri.scope.as_str(),
            uri.owner_id,
            uri.path
        )
    }

    /// Fetch the blob bytes for `uri`. Cache is consulted first by HEAD-checking
    /// the server; if cached revision matches, returns local bytes.
    pub async fn read(&self, uri: &VaultUri) -> Result<(BlobMeta, Vec<u8>), VaultError> {
        self.ensure_allowed(uri)?;
        let meta = self.head(uri).await?;
        if let Some(cache) = &self.cache {
            if let Some(cached) = cache.lookup(uri).await {
                if cached.sha256 == meta.sha256 && cached.revision == meta.revision.0 {
                    let bytes = cache.read_blob(&meta.sha256).await?;
                    return Ok((meta, bytes));
                }
            }
        }
        let resp = self
            .http
            .get(self.url(uri))
            .headers(self.auth_headers())
            .send()
            .await?;
        let status = resp.status();
        match status {
            StatusCode::OK => {}
            StatusCode::NOT_FOUND => return Err(VaultError::NotFound(uri.path.clone())),
            StatusCode::FORBIDDEN => return Err(VaultError::Forbidden(uri.path.clone())),
            s => return Err(VaultError::Status(s)),
        }
        let bytes = resp.bytes().await?.to_vec();
        if let Some(cache) = &self.cache {
            let hash = cache.insert_blob(&bytes, Some(&meta.sha256)).await?;
            cache
                .record(uri, &hash, meta.revision.0, meta.size)
                .await?;
        }
        Ok((meta, bytes))
    }

    /// HEAD — returns metadata without downloading the blob.
    pub async fn head(&self, uri: &VaultUri) -> Result<BlobMeta, VaultError> {
        self.ensure_allowed(uri)?;
        let resp = self
            .http
            .head(self.url(uri))
            .headers(self.auth_headers())
            .send()
            .await?;
        match resp.status() {
            StatusCode::OK => {}
            StatusCode::NOT_FOUND => return Err(VaultError::NotFound(uri.path.clone())),
            StatusCode::FORBIDDEN => return Err(VaultError::Forbidden(uri.path.clone())),
            s => return Err(VaultError::Status(s)),
        }
        meta_from_headers(resp.headers())
    }

    /// CAS write. `if_match = Revision::ZERO` for create-only.
    pub async fn write(
        &self,
        uri: &VaultUri,
        bytes: Vec<u8>,
        if_match: Revision,
        mime_type: Option<&str>,
    ) -> Result<BlobMeta, VaultError> {
        self.ensure_allowed(uri)?;
        let mut req = self
            .http
            .put(self.url(uri))
            .headers(self.auth_headers())
            .header(IF_MATCH, if_match.0.to_string())
            .body(bytes.clone());
        if let Some(mt) = mime_type {
            req = req.header("content-type", mt);
        }
        let resp = req.send().await?;
        match resp.status() {
            s if s.is_success() => {
                let meta = meta_from_headers(resp.headers())?;
                if let Some(cache) = &self.cache {
                    let hash = cache.insert_blob(&bytes, Some(&meta.sha256)).await?;
                    cache
                        .record(uri, &hash, meta.revision.0, meta.size)
                        .await?;
                }
                Ok(meta)
            }
            StatusCode::PRECONDITION_FAILED | StatusCode::CONFLICT => {
                let current = parse_revision_header(resp.headers(), "x-kiki-revision")
                    .unwrap_or(Revision::ZERO);
                Err(VaultError::Conflict { current })
            }
            StatusCode::FORBIDDEN => Err(VaultError::Forbidden(uri.path.clone())),
            s => Err(VaultError::Status(s)),
        }
    }

    pub async fn delete(&self, uri: &VaultUri, if_match: Revision) -> Result<(), VaultError> {
        self.ensure_allowed(uri)?;
        let resp = self
            .http
            .delete(self.url(uri))
            .headers(self.auth_headers())
            .header(IF_MATCH, if_match.0.to_string())
            .send()
            .await?;
        match resp.status() {
            s if s.is_success() => Ok(()),
            StatusCode::PRECONDITION_FAILED | StatusCode::CONFLICT => {
                let current = parse_revision_header(resp.headers(), "x-kiki-revision")
                    .unwrap_or(Revision::ZERO);
                Err(VaultError::Conflict { current })
            }
            StatusCode::NOT_FOUND => Err(VaultError::NotFound(uri.path.clone())),
            StatusCode::FORBIDDEN => Err(VaultError::Forbidden(uri.path.clone())),
            s => Err(VaultError::Status(s)),
        }
    }

    /// List entries under a path prefix. `since` returns only entries whose
    /// revision > since (incremental sync).
    pub async fn list(
        &self,
        scope: crate::uri::VaultScope,
        owner_id: &str,
        prefix: &str,
        since: Revision,
    ) -> Result<Vec<ListEntry>, VaultError> {
        let url = format!(
            "{}/v1/vault/{}/{}?prefix={}&since={}",
            self.base_url,
            scope.as_str(),
            owner_id,
            urlencode(prefix),
            since.0
        );
        let resp = self.http.get(url).headers(self.auth_headers()).send().await?;
        if !resp.status().is_success() {
            return Err(VaultError::Status(resp.status()));
        }
        Ok(resp.json().await?)
    }

    /// Open a WebSocket subscription for `globs`. Events are emitted on the
    /// returned receiver; dropping it terminates the connection.
    pub async fn watch(
        &self,
        scope: crate::uri::VaultScope,
        owner_id: &str,
        globs: &[&str],
    ) -> Result<mpsc::Receiver<VaultEvent>, VaultError> {
        let base = self
            .base_url
            .replace("https://", "wss://")
            .replace("http://", "ws://");
        let glob_qs = globs
            .iter()
            .map(|g| format!("glob={}", urlencode(g)))
            .collect::<Vec<_>>()
            .join("&");
        let url = format!(
            "{}/v1/vault/{}/{}/_watch?{}",
            base,
            scope.as_str(),
            owner_id,
            glob_qs
        );
        let request = tokio_tungstenite::tungstenite::http::Request::builder()
            .uri(&url)
            .header("authorization", format!("Bearer {}", self.token.jwt))
            .header("sec-websocket-key", tokio_tungstenite::tungstenite::handshake::client::generate_key())
            .header("sec-websocket-version", "13")
            .header("connection", "Upgrade")
            .header("upgrade", "websocket")
            .header("host", host_from(&url))
            .body(())
            .map_err(|e| VaultError::Ws(e.to_string()))?;
        let (ws, _) = tokio_tungstenite::connect_async(request)
            .await
            .map_err(|e| VaultError::Ws(e.to_string()))?;
        let (mut sink, mut stream) = ws.split();
        let (tx, rx) = mpsc::channel::<VaultEvent>(64);
        tokio::spawn(async move {
            while let Some(msg) = stream.next().await {
                let msg = match msg {
                    Ok(m) => m,
                    Err(e) => {
                        tracing::warn!(error = %e, "vault watch ws stream error");
                        break;
                    }
                };
                use tokio_tungstenite::tungstenite::Message;
                match msg {
                    Message::Text(t) => {
                        if let Ok(ev) = serde_json::from_str::<VaultEvent>(&t) {
                            if tx.send(ev).await.is_err() {
                                break;
                            }
                        }
                    }
                    Message::Binary(b) => {
                        if let Ok(ev) = serde_json::from_slice::<VaultEvent>(&b) {
                            if tx.send(ev).await.is_err() {
                                break;
                            }
                        }
                    }
                    Message::Ping(p) => {
                        let _ = sink.send(Message::Pong(p)).await;
                    }
                    Message::Close(_) => break,
                    _ => {}
                }
            }
        });
        Ok(rx)
    }

    pub fn token(&self) -> &CapabilityToken {
        &self.token
    }
}

fn host_from(url: &str) -> String {
    url.split("://")
        .nth(1)
        .and_then(|rest| rest.split('/').next())
        .unwrap_or("")
        .to_string()
}

fn meta_from_headers(h: &HeaderMap) -> Result<BlobMeta, VaultError> {
    let sha256 = h
        .get("x-kiki-sha256")
        .and_then(|v| v.to_str().ok())
        .ok_or(VaultError::MissingHeader("x-kiki-sha256"))?
        .to_string();
    let size = h
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let revision = parse_revision_header(h, "x-kiki-revision")
        .ok_or(VaultError::MissingHeader("x-kiki-revision"))?;
    let mime_type = h
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .map(String::from);
    let modified_at_ms = h
        .get("x-kiki-modified-ms")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    Ok(BlobMeta {
        sha256,
        size,
        revision,
        mime_type,
        modified_at_ms,
    })
}

fn parse_revision_header(h: &HeaderMap, name: &str) -> Option<Revision> {
    h.get(name)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok())
        .map(Revision)
}

fn urlencode(s: &str) -> String {
    s.bytes()
        .flat_map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                vec![b as char]
            }
            _ => format!("%{b:02X}").chars().collect(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uri::VaultScope;

    fn token(allowed: Vec<&str>) -> CapabilityToken {
        CapabilityToken {
            jwt:           "fake".into(),
            paths_allowed: allowed.into_iter().map(String::from).collect(),
            expires_at_ms: u64::MAX,
        }
    }

    #[test]
    fn capability_scope_blocks_unlisted_paths() {
        let c = VaultClient::new("https://x", token(vec!["notes/**"]), None).unwrap();
        let allowed = VaultUri {
            scope:    VaultScope::Personal,
            owner_id: "u".into(),
            path:     "notes/a.md".into(),
        };
        let blocked = VaultUri {
            scope:    VaultScope::Personal,
            owner_id: "u".into(),
            path:     "secrets/key".into(),
        };
        assert!(c.ensure_allowed(&allowed).is_ok());
        assert!(matches!(c.ensure_allowed(&blocked), Err(VaultError::CapabilityScope(_))));
    }
}
