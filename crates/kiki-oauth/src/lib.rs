//! kiki-oauth — on-device OAuth → Secrets (unlock #3; see spec/APPS.md §3.3).
//!
//! L2 apps that talk to user accounts (mail, calendar, contacts, paid maps) need
//! per-account credentials. This crate runs the OAuth flow on the device
//! (distinct from the fleet-enrollment DeviceFlow in kiki-sdk — that enrolls the
//! *node*; this authorizes a *service account*), stores the resulting tokens
//! sealed in a [`SecretStore`], and exposes them ONLY to the egress broker via a
//! [`kiki_net::CredentialInjector`] — the app never reads the token in clear.
//!
//! Headless nodes use the OAuth 2.0 Device Authorization Grant (RFC 8628); the
//! token endpoint and refresh logic work against any RFC 6749 provider.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, warn};

mod store;
pub use store::{MemorySecretStore, SealedFileSecretStore, SecretStore};

#[derive(Debug, Error)]
pub enum OAuthError {
    #[error("provider not configured: {0:?}")]
    Provider(Provider),
    #[error("authorization failed: {0}")]
    Authorize(String),
    #[error("authorization pending")]
    AuthorizationPending,
    #[error("token refresh failed: {0}")]
    Refresh(String),
    #[error("no credential stored for handle: {0}")]
    NotFound(String),
    #[error("invalid handle: {0}")]
    BadHandle(String),
    #[error("secret store error: {0}")]
    Secret(String),
    #[error("http error: {0}")]
    Http(String),
}

pub type Result<T> = std::result::Result<T, OAuthError>;

/// Supported credential providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Provider {
    Google,
    Microsoft,
    GenericPkce,
    /// Generic IMAP/CalDAV/CardDAV app-password (no OAuth dance — stored as-is).
    AppPassword,
}

impl Provider {
    fn as_str(&self) -> &'static str {
        match self {
            Provider::Google => "google",
            Provider::Microsoft => "microsoft",
            Provider::GenericPkce => "generic_pkce",
            Provider::AppPassword => "app_password",
        }
    }
    fn parse(s: &str) -> Option<Provider> {
        Some(match s {
            "google" => Provider::Google,
            "microsoft" => Provider::Microsoft,
            "generic_pkce" => Provider::GenericPkce,
            "app_password" => Provider::AppPassword,
            _ => return None,
        })
    }
}

/// Endpoints + client identity for a provider.
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub token_url:                String,
    /// RFC 8628 device authorization endpoint (headless consent). Optional.
    pub device_authorization_url: Option<String>,
    pub client_id:                String,
    pub client_secret:            Option<String>,
    pub scopes:                   Vec<String>,
}

/// Tokens held sealed in the store (never returned to apps in clear).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenSet {
    pub access:        String,
    pub refresh:       Option<String>,
    /// Unix ms when `access` expires.
    pub expires_at_ms: u64,
}

/// Opaque reference to a stored credential:
/// `secrets://app/<app_id>/<provider>/<account>`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SecretHandle(pub String);

impl SecretHandle {
    pub fn new(app_id: &str, provider: Provider, account: &str) -> Self {
        SecretHandle(format!("secrets://app/{app_id}/{}/{account}", provider.as_str()))
    }
    /// Parse the provider out of the handle.
    fn provider(&self) -> Result<Provider> {
        // secrets://app/<app_id>/<provider>/<account>
        let rest = self
            .0
            .strip_prefix("secrets://app/")
            .ok_or_else(|| OAuthError::BadHandle(self.0.clone()))?;
        let parts: Vec<&str> = rest.split('/').collect();
        if parts.len() < 3 {
            return Err(OAuthError::BadHandle(self.0.clone()));
        }
        Provider::parse(parts[parts.len() - 2]).ok_or_else(|| OAuthError::BadHandle(self.0.clone()))
    }
}

/// Returned to a caller that started a device-authorization flow.
#[derive(Debug, Clone, Deserialize)]
pub struct DeviceAuth {
    pub device_code:      String,
    pub user_code:        String,
    pub verification_uri: String,
    #[serde(default)]
    pub verification_uri_complete: Option<String>,
    #[serde(default = "default_interval")]
    pub interval:         u64,
    #[serde(default)]
    pub expires_in:       u64,
}

fn default_interval() -> u64 {
    5
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token:  String,
    #[serde(default)]
    expires_in:    u64,
    #[serde(default)]
    refresh_token: Option<String>,
    #[serde(default)]
    error:         Option<String>,
}

/// Drives the on-device OAuth flow and brokers token access.
pub struct OAuthFlow<S: SecretStore> {
    store:     S,
    http:      reqwest::Client,
    providers: HashMap<Provider, ProviderConfig>,
    /// Refresh when the access token is within this many ms of expiry.
    skew_ms:   u64,
}

impl<S: SecretStore> OAuthFlow<S> {
    pub fn new(store: S) -> Self {
        Self {
            store,
            http: reqwest::Client::new(),
            providers: HashMap::new(),
            skew_ms: 60_000,
        }
    }

    pub fn with_provider(mut self, provider: Provider, cfg: ProviderConfig) -> Self {
        self.providers.insert(provider, cfg);
        self
    }

    fn now_ms() -> u64 {
        SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
    }

    fn cfg(&self, p: Provider) -> Result<&ProviderConfig> {
        self.providers.get(&p).ok_or(OAuthError::Provider(p))
    }

    // ── Storage ──────────────────────────────────────────────────────────────

    /// Seal + persist a token set under the handle (after a completed auth).
    pub fn store_tokens(&self, handle: &SecretHandle, tokens: &TokenSet) -> Result<()> {
        let bytes = serde_json::to_vec(tokens).map_err(|e| OAuthError::Secret(e.to_string()))?;
        self.store.put(&handle.0, &bytes)
    }

    fn load_tokens(&self, handle: &SecretHandle) -> Result<TokenSet> {
        let bytes = self
            .store
            .get(&handle.0)?
            .ok_or_else(|| OAuthError::NotFound(handle.0.clone()))?;
        serde_json::from_slice(&bytes).map_err(|e| OAuthError::Secret(e.to_string()))
    }

    // ── Token access (called by the broker) ───────────────────────────────────

    /// Resolve a handle to a valid access token, refreshing (and persisting the
    /// rotated token) if it is at/near expiry. Never called by the app.
    pub async fn access_token(&self, handle: &SecretHandle) -> Result<String> {
        let tokens = self.load_tokens(handle)?;
        if tokens.expires_at_ms > Self::now_ms() + self.skew_ms {
            return Ok(tokens.access);
        }
        // Near/at expiry → refresh.
        let provider = handle.provider()?;
        let cfg = self.cfg(provider)?;
        let Some(refresh_token) = tokens.refresh.clone() else {
            // No refresh token and expired — surface so the app re-authorizes.
            return Err(OAuthError::Refresh("access token expired, no refresh token".into()));
        };
        debug!(handle = %handle.0, "refreshing access token");
        let refreshed = self.refresh(cfg, &refresh_token).await?;
        self.store_tokens(handle, &refreshed)?;
        Ok(refreshed.access)
    }

    async fn refresh(&self, cfg: &ProviderConfig, refresh_token: &str) -> Result<TokenSet> {
        let mut form: Vec<(&str, String)> = vec![
            ("grant_type", "refresh_token".into()),
            ("refresh_token", refresh_token.to_string()),
            ("client_id", cfg.client_id.clone()),
        ];
        if let Some(secret) = &cfg.client_secret {
            form.push(("client_secret", secret.clone()));
        }
        let resp = self
            .http
            .post(&cfg.token_url)
            .form(&form)
            .send()
            .await
            .map_err(|e| OAuthError::Http(e.to_string()))?;
        let body: TokenResponse =
            resp.json().await.map_err(|e| OAuthError::Refresh(e.to_string()))?;
        if let Some(err) = body.error {
            return Err(OAuthError::Refresh(err));
        }
        Ok(TokenSet {
            access:        body.access_token,
            // Providers often omit refresh_token on refresh — keep the old one.
            refresh:       body.refresh_token.or_else(|| Some(refresh_token.to_string())),
            expires_at_ms: Self::now_ms() + body.expires_in.saturating_mul(1000),
        })
    }

    // ── Device authorization grant (RFC 8628) ──────────────────────────────────

    /// Begin a device-authorization flow. Show the returned `user_code` +
    /// `verification_uri` to the operator, then poll [`Self::poll_device_token`].
    pub async fn start_device_flow(&self, provider: Provider) -> Result<DeviceAuth> {
        let cfg = self.cfg(provider)?;
        let url = cfg
            .device_authorization_url
            .as_ref()
            .ok_or_else(|| OAuthError::Authorize("provider has no device endpoint".into()))?;
        let resp = self
            .http
            .post(url)
            .form(&[
                ("client_id", cfg.client_id.clone()),
                ("scope", cfg.scopes.join(" ")),
            ])
            .send()
            .await
            .map_err(|e| OAuthError::Http(e.to_string()))?;
        resp.json::<DeviceAuth>().await.map_err(|e| OAuthError::Authorize(e.to_string()))
    }

    /// Exchange a `device_code` for tokens. Returns
    /// [`OAuthError::AuthorizationPending`] while the user hasn't consented yet —
    /// the caller polls at `DeviceAuth.interval` until success or expiry.
    pub async fn poll_device_token(
        &self,
        provider: Provider,
        device_code: &str,
    ) -> Result<TokenSet> {
        let cfg = self.cfg(provider)?;
        let mut form: Vec<(&str, String)> = vec![
            ("grant_type", "urn:ietf:params:oauth:grant-type:device_code".into()),
            ("device_code", device_code.to_string()),
            ("client_id", cfg.client_id.clone()),
        ];
        if let Some(secret) = &cfg.client_secret {
            form.push(("client_secret", secret.clone()));
        }
        let resp = self
            .http
            .post(&cfg.token_url)
            .form(&form)
            .send()
            .await
            .map_err(|e| OAuthError::Http(e.to_string()))?;
        let body: TokenResponse =
            resp.json().await.map_err(|e| OAuthError::Authorize(e.to_string()))?;
        if let Some(err) = body.error {
            return match err.as_str() {
                "authorization_pending" | "slow_down" => Err(OAuthError::AuthorizationPending),
                other => Err(OAuthError::Authorize(other.to_string())),
            };
        }
        Ok(TokenSet {
            access:        body.access_token,
            refresh:       body.refresh_token,
            expires_at_ms: Self::now_ms() + body.expires_in.saturating_mul(1000),
        })
    }

    /// Run a device flow to completion: start, then poll until consent or expiry.
    /// Persists the tokens and returns the handle the app stores.
    pub async fn authorize(
        &self,
        provider: Provider,
        app_id: &str,
        account: &str,
    ) -> Result<SecretHandle> {
        let device = self.start_device_flow(provider).await?;
        let handle = SecretHandle::new(app_id, provider, account);
        let deadline = Self::now_ms()
            + if device.expires_in > 0 { device.expires_in * 1000 } else { 600_000 };
        let interval = device.interval.max(1);
        loop {
            if Self::now_ms() > deadline {
                return Err(OAuthError::Authorize("device flow expired".into()));
            }
            tokio::time::sleep(std::time::Duration::from_secs(interval)).await;
            match self.poll_device_token(provider, &device.device_code).await {
                Ok(tokens) => {
                    self.store_tokens(&handle, &tokens)?;
                    return Ok(handle);
                }
                Err(OAuthError::AuthorizationPending) => continue,
                Err(e) => return Err(e),
            }
        }
    }

    /// Revoke + delete a stored credential.
    pub async fn revoke(&self, handle: &SecretHandle) -> Result<()> {
        // Provider-side revocation is best-effort and provider-specific; the
        // authoritative action is removing the local secret.
        self.store.delete(&handle.0)
    }
}

/// Maps an (`app_id`, `host`) pair to a stored credential and injects the
/// resulting bearer token into brokered requests — the egress broker's
/// [`kiki_net::CredentialInjector`]. The app never sees the token.
pub struct SecretsCredentialInjector<S: SecretStore> {
    flow:     Arc<OAuthFlow<S>>,
    /// (app_id, host) → handle. Populated when an account is configured.
    mappings: HashMap<(String, String), SecretHandle>,
}

impl<S: SecretStore> SecretsCredentialInjector<S> {
    pub fn new(flow: Arc<OAuthFlow<S>>) -> Self {
        Self { flow, mappings: HashMap::new() }
    }

    /// Route requests from `app_id` to `host` through the credential `handle`.
    pub fn map(&mut self, app_id: &str, host: &str, handle: SecretHandle) {
        self.mappings.insert((app_id.to_string(), host.to_string()), handle);
    }
}

#[async_trait::async_trait]
impl<S: SecretStore + 'static> kiki_net::CredentialInjector for SecretsCredentialInjector<S> {
    async fn inject(&self, app_id: &str, host: &str) -> Vec<(String, String)> {
        let Some(handle) = self.mappings.get(&(app_id.to_string(), host.to_string())) else {
            return vec![];
        };
        match self.flow.access_token(handle).await {
            Ok(token) => vec![("authorization".into(), format!("Bearer {token}"))],
            Err(e) => {
                warn!(app = app_id, host, error = %e, "credential injection failed");
                vec![]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn flow_with_token_url(url: String) -> OAuthFlow<MemorySecretStore> {
        OAuthFlow::new(MemorySecretStore::new()).with_provider(
            Provider::Google,
            ProviderConfig {
                token_url:                url,
                device_authorization_url: None,
                client_id:                "client-1".into(),
                client_secret:            None,
                scopes:                   vec!["mail".into()],
            },
        )
    }

    #[test]
    fn handle_roundtrips_provider() {
        let h = SecretHandle::new("io.kiki.mail", Provider::Google, "me@example.com");
        assert_eq!(h.0, "secrets://app/io.kiki.mail/google/me@example.com");
        assert_eq!(h.provider().unwrap(), Provider::Google);
    }

    #[tokio::test]
    async fn access_token_returns_valid_without_refresh() {
        let flow = flow_with_token_url("http://unused".into());
        let h = SecretHandle::new("app", Provider::Google, "acct");
        flow.store_tokens(
            &h,
            &TokenSet { access: "live".into(), refresh: Some("r".into()), expires_at_ms: u64::MAX },
        )
        .unwrap();
        assert_eq!(flow.access_token(&h).await.unwrap(), "live");
    }

    #[tokio::test]
    async fn access_token_refreshes_when_expired() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "access_token": "fresh", "expires_in": 3600, "refresh_token": "r2"
            })))
            .mount(&server)
            .await;
        let flow = flow_with_token_url(format!("{}/token", server.uri()));
        let h = SecretHandle::new("app", Provider::Google, "acct");
        // Expired token with a refresh token.
        flow.store_tokens(
            &h,
            &TokenSet { access: "stale".into(), refresh: Some("r1".into()), expires_at_ms: 0 },
        )
        .unwrap();
        assert_eq!(flow.access_token(&h).await.unwrap(), "fresh");
        // The rotated token was persisted.
        assert_eq!(flow.access_token(&h).await.unwrap(), "fresh");
    }

    #[tokio::test]
    async fn injector_adds_bearer_only_for_mapped_app_host() {
        let flow = Arc::new(flow_with_token_url("http://unused".into()));
        let h = SecretHandle::new("io.kiki.mail", Provider::Google, "me");
        flow.store_tokens(
            &h,
            &TokenSet { access: "tok".into(), refresh: None, expires_at_ms: u64::MAX },
        )
        .unwrap();
        let mut inj = SecretsCredentialInjector::new(flow);
        inj.map("io.kiki.mail", "imap.gmail.com", h);

        use kiki_net::CredentialInjector;
        let hdrs = inj.inject("io.kiki.mail", "imap.gmail.com").await;
        assert_eq!(hdrs, vec![("authorization".to_string(), "Bearer tok".to_string())]);
        // Unmapped host → nothing.
        assert!(inj.inject("io.kiki.mail", "evil.example").await.is_empty());
        // Unmapped app → nothing.
        assert!(inj.inject("io.kiki.other", "imap.gmail.com").await.is_empty());
    }
}
