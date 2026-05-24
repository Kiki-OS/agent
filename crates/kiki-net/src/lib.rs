//! kiki-net — egress broker (unlock #2 of the L2 plumbing; see spec/APPS.md §3.2).
//!
//! Apps never open sockets directly. They call the `net.fetch` MCP tool; agentd
//! routes that to this broker, which: validates the host against the calling
//! app's allowlist (from `kiki.toml [network] egress`), applies the router
//! policy, injects credentials from Secrets (without exposing them to the app),
//! performs the request, and writes an audit record. Brokered model = the agent
//! is the single auditable egress point (Safety > Privacy > Security).
//!
//! SCAFFOLD: the allowlist check (`check`) is implemented for real (deny-by-
//! default); `fetch` returns NotImplemented with a note on what it must cover.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EgressError {
    #[error("egress denied for {app}: {host}:{port} not in allowlist")]
    Denied { app: String, host: String, port: u16 },
    #[error("blocked by router policy: {0}")]
    Policy(String),
    #[error("invalid request url: {0}")]
    BadUrl(String),
    #[error("response exceeded max size of {max} bytes")]
    TooLarge { max: usize },
    #[error("upstream error: {0}")]
    Upstream(String),
}

pub type Result<T> = std::result::Result<T, EgressError>;

/// A host:port the broker may reach on an app's behalf. `host == "*"` marks a
/// dynamic-egress app (web-fetch/browser): the host is not fixed, so the broker
/// authorizes per-call, ControlMode-gated, instead of matching the allowlist.
#[derive(Debug, Clone)]
pub struct HostPort {
    pub host: String,
    pub port: u16,
}

impl HostPort {
    /// Parse a `"host:port"` entry (the format used in `[network] egress`).
    pub fn parse(s: &str) -> Option<Self> {
        let (host, port) = s.rsplit_once(':')?;
        Some(Self { host: host.to_string(), port: port.parse().ok()? })
    }
}

/// A brokered request issued by an app via the `net.fetch` tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FetchRequest {
    pub method: String,
    pub url:    String,
    #[serde(default)]
    pub headers: Vec<(String, String)>,
    #[serde(default)]
    pub body:    Option<String>,
}

/// The brokered response handed back to the app.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FetchResponse {
    pub status:  u16,
    pub headers: Vec<(String, String)>,
    pub body:    String,
}

/// Injects credentials for an authenticated host without exposing them to the
/// calling app. Returns headers to add to the outbound request (e.g.
/// `Authorization`). agentd wires an implementation backed by Secrets/vault;
/// kiki-net stays decoupled so it can be unit-tested in isolation.
#[async_trait::async_trait]
pub trait CredentialInjector: Send + Sync {
    async fn inject(&self, app_id: &str, host: &str) -> Vec<(String, String)>;
}

/// One audited egress event — the broker is the single auditable egress point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EgressAudit {
    pub app:    String,
    pub method: String,
    pub host:   String,
    pub port:   u16,
    pub status: u16,
    pub bytes:  usize,
    pub ts_ms:  u64,
}

/// Sink for egress audit records (agentd persists these / ships to the fleet).
#[async_trait::async_trait]
pub trait AuditSink: Send + Sync {
    async fn record(&self, entry: &EgressAudit);
}

/// Per-app egress allowlists + the brokered fetch entrypoint.
pub struct EgressBroker {
    allowlists:     HashMap<String, Vec<HostPort>>,
    http:           reqwest::Client,
    injector:       Option<Arc<dyn CredentialInjector>>,
    audit:          Option<Arc<dyn AuditSink>>,
    max_body_bytes: usize,
}

impl Default for EgressBroker {
    fn default() -> Self {
        Self::new()
    }
}

impl EgressBroker {
    pub fn new() -> Self {
        Self {
            allowlists:     HashMap::new(),
            http:           reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .unwrap_or_default(),
            injector:       None,
            audit:          None,
            max_body_bytes: 8 * 1024 * 1024, // 8 MiB default ceiling
        }
    }

    /// Wire a credential injector (agentd backs this with Secrets/vault).
    pub fn with_injector(mut self, injector: Arc<dyn CredentialInjector>) -> Self {
        self.injector = Some(injector);
        self
    }

    /// Wire an audit sink for egress records.
    pub fn with_audit(mut self, audit: Arc<dyn AuditSink>) -> Self {
        self.audit = Some(audit);
        self
    }

    /// Cap on the response body the broker will buffer (bounds app memory use).
    pub fn with_max_body_bytes(mut self, max: usize) -> Self {
        self.max_body_bytes = max;
        self
    }

    /// Register an app's egress allowlist (resolved from its `[network] egress`
    /// at install/grant time).
    pub fn allow(&mut self, app_id: impl Into<String>, hosts: Vec<HostPort>) {
        self.allowlists.insert(app_id.into(), hosts);
    }

    /// Deny-by-default host check. Returns Ok if the app declared this host:port
    /// or is a dynamic-egress app (`*`).
    pub fn check(&self, app_id: &str, host: &str, port: u16) -> Result<()> {
        let deny = || EgressError::Denied {
            app: app_id.to_string(), host: host.to_string(), port,
        };
        let list = self.allowlists.get(app_id).ok_or_else(deny)?;
        let dynamic = list.iter().any(|h| h.host == "*");
        let matched = list.iter().any(|h| h.host == host && h.port == port);
        if dynamic || matched { Ok(()) } else { Err(deny()) }
    }

    /// Perform a brokered request on the app's behalf: authorize the host against
    /// the app's allowlist, inject credentials (never exposed to the app), issue
    /// the request, enforce the body-size ceiling, and write an audit record.
    pub async fn fetch(&self, app_id: &str, req: FetchRequest) -> Result<FetchResponse> {
        let url = url::Url::parse(&req.url).map_err(|e| EgressError::BadUrl(e.to_string()))?;
        let host = url
            .host_str()
            .ok_or_else(|| EgressError::BadUrl("missing host".into()))?
            .to_string();
        let port = url
            .port_or_known_default()
            .ok_or_else(|| EgressError::BadUrl("unknown scheme/port".into()))?;

        self.check(app_id, &host, port)?;

        let method = reqwest::Method::from_bytes(req.method.as_bytes())
            .map_err(|e| EgressError::BadUrl(format!("bad method: {e}")))?;
        let mut builder = self.http.request(method, url.clone());
        for (k, v) in &req.headers {
            builder = builder.header(k, v);
        }
        // Inject credentials for this host — the app never sees them.
        if let Some(injector) = &self.injector {
            for (k, v) in injector.inject(app_id, &host).await {
                builder = builder.header(k, v);
            }
        }
        if let Some(body) = req.body {
            builder = builder.body(body);
        }

        let resp = builder.send().await.map_err(|e| EgressError::Upstream(e.to_string()))?;
        let status = resp.status().as_u16();
        let headers: Vec<(String, String)> = resp
            .headers()
            .iter()
            .map(|(k, v)| (k.as_str().to_string(), v.to_str().unwrap_or("").to_string()))
            .collect();

        // Read the body with a hard ceiling so a hostile/huge response can't
        // exhaust device memory. `chunk()` streams without pulling in a Stream
        // extension trait.
        let mut resp = resp;
        let mut buf: Vec<u8> = Vec::new();
        while let Some(chunk) =
            resp.chunk().await.map_err(|e| EgressError::Upstream(e.to_string()))?
        {
            if buf.len() + chunk.len() > self.max_body_bytes {
                return Err(EgressError::TooLarge { max: self.max_body_bytes });
            }
            buf.extend_from_slice(&chunk);
        }
        let bytes = buf.len();
        let body = String::from_utf8_lossy(&buf).into_owned();

        if let Some(audit) = &self.audit {
            audit
                .record(&EgressAudit {
                    app: app_id.to_string(),
                    method: req.method.clone(),
                    host,
                    port,
                    status,
                    bytes,
                    ts_ms: now_ms(),
                })
                .await;
        }

        Ok(FetchResponse { status, headers, body })
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deny_by_default_and_allowlist() {
        let mut b = EgressBroker::new();
        b.allow("io.kiki.weather", vec![HostPort { host: "api.open-meteo.com".into(), port: 443 }]);

        assert!(b.check("io.kiki.weather", "api.open-meteo.com", 443).is_ok());
        assert!(b.check("io.kiki.weather", "evil.example", 443).is_err());
        // Unknown app: denied.
        assert!(b.check("io.kiki.unknown", "api.open-meteo.com", 443).is_err());
    }

    #[test]
    fn dynamic_egress_allows_any_host() {
        let mut b = EgressBroker::new();
        b.allow("io.kiki.web-fetch", vec![HostPort { host: "*".into(), port: 0 }]);
        assert!(b.check("io.kiki.web-fetch", "anything.example", 443).is_ok());
    }

    #[test]
    fn hostport_parse() {
        let hp = HostPort::parse("api.open-meteo.com:443").unwrap();
        assert_eq!(hp.host, "api.open-meteo.com");
        assert_eq!(hp.port, 443);
        assert!(HostPort::parse("no-port").is_none());
    }

    // ── fetch() integration (wiremock) ──────────────────────────────────────────
    use std::sync::Mutex;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn req(url: &str) -> FetchRequest {
        FetchRequest { method: "GET".into(), url: url.into(), headers: vec![], body: None }
    }

    /// Records injected creds + audit entries for assertions.
    struct TestCreds {
        header: (String, String),
        seen:   Mutex<Vec<String>>,
    }
    #[async_trait::async_trait]
    impl CredentialInjector for TestCreds {
        async fn inject(&self, _app: &str, host: &str) -> Vec<(String, String)> {
            self.seen.lock().unwrap().push(host.to_string());
            vec![self.header.clone()]
        }
    }

    struct TestAudit {
        entries: Mutex<Vec<EgressAudit>>,
    }
    #[async_trait::async_trait]
    impl AuditSink for TestAudit {
        async fn record(&self, e: &EgressAudit) {
            self.entries.lock().unwrap().push(e.clone());
        }
    }

    #[tokio::test]
    async fn fetch_denied_when_host_not_allowed() {
        let broker = EgressBroker::new(); // no allowlist for the app
        let r = broker.fetch("io.kiki.weather", req("https://evil.example/x")).await;
        assert!(matches!(r, Err(EgressError::Denied { .. })));
    }

    #[tokio::test]
    async fn fetch_allowed_host_injects_creds_and_audits() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/weather"))
            .respond_with(ResponseTemplate::new(200).set_body_string("{\"temp\":21}"))
            .mount(&server)
            .await;

        let host = server.address().ip().to_string();
        let port = server.address().port();

        let creds = Arc::new(TestCreds {
            header: ("authorization".into(), "Bearer secret-not-seen-by-app".into()),
            seen:   Mutex::new(vec![]),
        });
        let audit = Arc::new(TestAudit { entries: Mutex::new(vec![]) });

        let mut broker = EgressBroker::new()
            .with_injector(creds.clone())
            .with_audit(audit.clone());
        broker.allow("io.kiki.weather", vec![HostPort { host: host.clone(), port }]);

        let url = format!("{}/weather", server.uri());
        let resp = broker.fetch("io.kiki.weather", req(&url)).await.expect("fetch ok");
        assert_eq!(resp.status, 200);
        assert!(resp.body.contains("temp"));

        // Credentials were injected for the right host.
        assert_eq!(creds.seen.lock().unwrap().as_slice(), &[host.clone()]);
        // An audit record was written.
        let entries = audit.entries.lock().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].app, "io.kiki.weather");
        assert_eq!(entries[0].status, 200);
        assert!(entries[0].bytes > 0);
    }

    #[tokio::test]
    async fn fetch_enforces_body_size_ceiling() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/big"))
            .respond_with(ResponseTemplate::new(200).set_body_string("x".repeat(10_000)))
            .mount(&server)
            .await;
        let host = server.address().ip().to_string();
        let port = server.address().port();

        let mut broker = EgressBroker::new().with_max_body_bytes(1024);
        broker.allow("app", vec![HostPort { host: host.clone(), port }]);
        let url = format!("{}/big", server.uri());
        let r = broker.fetch("app", req(&url)).await;
        assert!(matches!(r, Err(EgressError::TooLarge { .. })));
    }
}
