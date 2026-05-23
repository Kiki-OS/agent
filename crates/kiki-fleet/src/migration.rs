use std::sync::Arc;
use kiki_core::{
    capability::CapabilitySet,
    context::Context,
    error::{Error, Result},
    state::{MigrationBundle, StateBackend},
};
use kiki_orchestrator::session::AgentSession;
use crate::client::FleetClient;

// ─── Source side ─────────────────────────────────────────────────────────────

/// Orchestrates the source side of a live session migration.
///
/// Full send protocol:
///   1. Request freeze on the PRA loop (waits for current step to complete).
///   2. Build a MigrationBundle from the frozen snapshot.
///   3. Push OSTree durable state delta to the registry remote.
///   4. POST the bundle + target node ID to the fleet relay (stored in DO + KV).
///   5. Transition session phase to Migrating.
pub struct MigrationSender {
    fleet:           Arc<FleetClient>,
    registry_remote: String,
}

impl MigrationSender {
    pub fn new(fleet: Arc<FleetClient>, registry_remote: impl Into<String>) -> Self {
        Self { fleet, registry_remote: registry_remote.into() }
    }

    /// Execute the full send flow. `ctx` must be the live Context driving the session.
    /// Returns the bundle that was sent (useful for audit logs / testing).
    pub async fn send(
        &self,
        session:        &AgentSession,
        ctx:            &Context,
        target_node_id: &str,
    ) -> Result<MigrationBundle> {
        // 1. Freeze PRA loop; wait for confirmation.
        tracing::info!(
            session_id = %session.id,
            target = target_node_id,
            "requesting session freeze for migration",
        );
        let rx = session.request_freeze();
        rx.await.map_err(|_| Error::Migration("freeze channel dropped".into()))?;

        // 2. Build bundle from frozen state.
        let bundle = session.build_bundle(ctx).await?;
        tracing::info!(
            bundle_id  = %bundle.bundle_id,
            step       = bundle.runtime.step,
            "migration bundle built",
        );

        // 3. Push OSTree delta to registry (only the diff since last shared commit).
        let ref_hash = session.state.push(&self.registry_remote).await?;
        tracing::info!(ref_hash = %ref_hash, "ostree delta pushed to registry");

        // 4. Relay bundle to fleet worker (→ SessionDO + KV pending queue).
        self.fleet.send_migration(&session.id, &bundle, target_node_id).await?;
        tracing::info!(target = target_node_id, "migration bundle relayed to fleet");

        // 5. Mark session as migrating so the local PRA loop knows not to resume.
        session.begin_migration(target_node_id);

        Ok(bundle)
    }
}

// ─── Target side ─────────────────────────────────────────────────────────────

/// Orchestrates the target side of a live session migration.
///
/// Full receive protocol:
///   1. Poll fleet for bundles addressed to this node.
///   2. Pull the OSTree delta from the registry into the local store.
///   3. Restore the state backend from the bundle.
///   4. Reconstruct Context from RuntimeSnapshot (history, interrupts, mode).
///   5. Ack to fleet worker (removes from pending queue, notifies DO of new node).
pub struct MigrationReceiver {
    fleet:           Arc<FleetClient>,
    registry_remote: String,
}

impl MigrationReceiver {
    pub fn new(fleet: Arc<FleetClient>, registry_remote: impl Into<String>) -> Self {
        Self { fleet, registry_remote: registry_remote.into() }
    }

    /// Poll for pending migrations. Non-blocking — returns immediately.
    /// Returns `(session_id, bundle)` pairs. May return an empty vec.
    pub async fn poll(&self) -> Result<Vec<(String, MigrationBundle)>> {
        self.fleet.poll_migrations().await
    }

    /// Restore a received bundle into a live Context ready to run the PRA loop.
    ///
    /// Capabilities are re-declared per the agent's kiki.toml after receipt;
    /// pass `CapabilitySet::default()` if you want to re-grant them separately.
    pub async fn restore(
        &self,
        bundle:       MigrationBundle,
        state:        Arc<dyn StateBackend>,
        capabilities: CapabilitySet,
    ) -> Result<RestoredSession> {
        let session_id = bundle.session_id.clone();
        let ref_hash   = bundle.checkpoint.ref_hash.as_deref().unwrap_or_default();

        // 2. Pull OSTree delta (no-op for memory/test backends).
        if !ref_hash.is_empty() {
            state.pull(&self.registry_remote, ref_hash).await?;
            tracing::info!(ref_hash, "ostree delta pulled from registry");
        }

        // 3. Restore state backend (e.g. checks out OSTree working tree).
        state.restore(bundle.clone()).await?;

        // 4. Reconstruct Context.
        let ctx = Context::from_snapshot(&bundle.runtime, state, capabilities);

        // 5. Notify fleet: remove from pending queue + update DO ownership.
        self.fleet.complete_migration(&session_id, self.fleet.node_id()).await?;
        tracing::info!(session_id = %session_id, "migration complete, session resumed");

        Ok(RestoredSession { bundle, ctx })
    }
}

/// Output of a successful `MigrationReceiver::restore` call.
/// The caller creates a new `AgentSession` from `bundle.runtime` and feeds
/// `ctx` into the PRA loop.
pub struct RestoredSession {
    pub bundle: MigrationBundle,
    pub ctx:    Context,
}
