use kiki_core::{
    error::{Error, Result},
    state::MigrationBundle,
};
use serde::Deserialize;
use serde_json::Value;

/// HTTP client for the kiki-cloud fleet worker.
/// Wraps all fleet API calls; used by heartbeat, sync, and migration modules.
pub struct FleetClient {
    base_url:   String,
    node_id:    String,
    http:       reqwest::Client,
    /// Bearer token bound to the owning user (from device-flow auth). When set,
    /// `register` binds the node to that user/org so it surfaces in the
    /// authenticated fleet view; unauthenticated registration falls back to the
    /// "system" placeholder owner.
    token:      Option<String>,
    /// Node identity reported on (re-)registration.
    flavor:     String,
    os_version: String,
    name:       Option<String>,
}

impl FleetClient {
    pub fn new(base_url: impl Into<String>, node_id: impl Into<String>) -> Self {
        Self {
            base_url:   base_url.into(),
            node_id:    node_id.into(),
            http:       reqwest::Client::new(),
            token:      None,
            flavor:     "desktop".into(),
            os_version: "0.1.0".into(),
            name:       None,
        }
    }

    /// Attach a Bearer token (from device-flow auth) for ownership binding.
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Set the identity reported on registration/heartbeat.
    pub fn with_identity(
        mut self,
        flavor:     impl Into<String>,
        os_version: impl Into<String>,
        name:       Option<String>,
    ) -> Self {
        self.flavor     = flavor.into();
        self.os_version = os_version.into();
        self.name       = name;
        self
    }

    pub fn node_id(&self) -> &str { &self.node_id }
    pub fn flavor(&self) -> &str { &self.flavor }
    pub fn os_version(&self) -> &str { &self.os_version }

    /// Register (or refresh) this node using its stored identity.
    pub async fn register_self(&self) -> Result<()> {
        let mut body = serde_json::json!({
            "flavor":     self.flavor,
            "os_version": self.os_version,
        });
        if let Some(name) = &self.name {
            body["name"] = serde_json::Value::String(name.clone());
        }
        self.post(&format!("/v1/fleet/nodes/{}/register", self.node_id), body).await
    }

    pub async fn register(&self, flavor: &str, os_version: &str) -> Result<()> {
        self.post(
            &format!("/v1/fleet/nodes/{}/register", self.node_id),
            serde_json::json!({ "flavor": flavor, "os_version": os_version }),
        ).await
    }

    /// POST a MigrationBundle to the fleet relay for a session.
    /// The DO stores it and the target node picks it up via `poll_migrations()`.
    pub async fn send_migration(
        &self,
        session_id:     &str,
        bundle:         &MigrationBundle,
        target_node_id: &str,
    ) -> Result<()> {
        let body = serde_json::json!({
            "bundle":         serde_json::to_value(bundle).map_err(Error::Json)?,
            "target_node_id": target_node_id,
        });
        self.post(&format!("/v1/fleet/sessions/{}/migrate", session_id), body).await
    }

    /// Poll for MigrationBundles addressed to this node.
    /// Returns `(session_id, bundle)` pairs.
    pub async fn poll_migrations(&self) -> Result<Vec<(String, MigrationBundle)>> {
        let url = format!("{}/v1/fleet/nodes/{}/migrations", self.base_url, self.node_id);
        let items = self.http.get(&url)
            .send().await.map_err(fleet_err)?
            .error_for_status().map_err(fleet_err)?
            .json::<Vec<PendingItem>>().await.map_err(fleet_err)?;

        items.into_iter()
            .map(|item| {
                let bundle = serde_json::from_value::<MigrationBundle>(item.bundle)
                    .map_err(Error::Json)?;
                Ok((item.session_id, bundle))
            })
            .collect()
    }

    /// Acknowledge a completed migration: removes the pending entry from the cloud
    /// and notifies the session DO that the target node has taken ownership.
    pub async fn complete_migration(
        &self,
        session_id:   &str,
        new_node_id:  &str,
    ) -> Result<()> {
        self.post(
            &format!("/v1/fleet/sessions/{}/migrate/complete", session_id),
            serde_json::json!({ "new_node_id": new_node_id }),
        ).await?;
        // Remove from the node's pending queue.
        let url = format!(
            "{}/v1/fleet/nodes/{}/migrations/{}",
            self.base_url, self.node_id, session_id,
        );
        self.http.delete(&url).send().await.map_err(fleet_err)?
            .error_for_status().map_err(fleet_err)?;
        Ok(())
    }

    // ── internal ───────────────────────────────────────────────────────────────

    async fn post(&self, path: &str, body: Value) -> Result<()> {
        let url = format!("{}{}", self.base_url, path);
        let mut req = self.http.post(&url).json(&body);
        if let Some(token) = &self.token {
            req = req.bearer_auth(token);
        }
        req.send().await.map_err(fleet_err)?
            .error_for_status().map_err(fleet_err)?;
        Ok(())
    }
}

#[derive(Deserialize)]
struct PendingItem {
    session_id: String,
    bundle:     Value,
}

fn fleet_err(e: reqwest::Error) -> Error {
    Error::Fleet(e.to_string())
}
