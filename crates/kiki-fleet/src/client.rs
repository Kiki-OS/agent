use kiki_core::{
    error::{Error, Result},
    state::MigrationBundle,
};
use serde::Deserialize;
use serde_json::Value;

/// HTTP client for the kiki-cloud fleet worker.
/// Wraps all fleet API calls; used by heartbeat, sync, and migration modules.
pub struct FleetClient {
    base_url: String,
    node_id:  String,
    http:     reqwest::Client,
}

impl FleetClient {
    pub fn new(base_url: impl Into<String>, node_id: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            node_id:  node_id.into(),
            http:     reqwest::Client::new(),
        }
    }

    pub fn node_id(&self) -> &str { &self.node_id }

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
        self.http.post(&url)
            .json(&body)
            .send().await.map_err(fleet_err)?
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
