use std::sync::Arc;
use std::time::Duration;
use kiki_core::error::Result;
use crate::client::FleetClient;

/// Sends periodic heartbeats to the fleet worker.
/// Keeps the node entry alive in KV (TTL 120s) and signals liveness to the DO.
pub struct Heartbeat {
    fleet:    Arc<FleetClient>,
    interval: Duration,
}

impl Heartbeat {
    pub fn new(fleet: Arc<FleetClient>, interval: Duration) -> Self {
        Self { fleet, interval }
    }

    pub fn with_default_interval(fleet: Arc<FleetClient>) -> Self {
        Self::new(fleet, Duration::from_secs(60))
    }

    /// Spawn a background task that sends heartbeats forever.
    /// The task stops when the returned `JoinHandle` is dropped or aborted.
    pub fn spawn(self) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(self.interval);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                interval.tick().await;
                if let Err(e) = self.tick().await {
                    tracing::warn!(error = %e, "fleet heartbeat failed");
                }
            }
        })
    }

    async fn tick(&self) -> Result<()> {
        // Re-register refreshes the KV TTL (120s), keeping the node online and
        // reporting its real identity (flavor/os_version) and ownership token.
        self.fleet.register_self().await?;
        tracing::debug!(node_id = self.fleet.node_id(), "heartbeat sent");
        Ok(())
    }
}
