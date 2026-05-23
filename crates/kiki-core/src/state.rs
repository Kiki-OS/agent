//! State backend trait and migration types.
//!
//! StateBackend is the durable persistence contract. Implementations live in kiki-state:
//! - OstreeBackend  — production (OSTree commit/push/pull, delta-transferable)
//! - MemoryBackend  — ephemeral/session, or for tests
//! - HybridBackend  — memory cache + async write-through to OSTree

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use crate::{
    context::ControlMode,
    error::Result,
    interrupt::Interrupt,
    surface::SessionLayout,
    types::ConversationMessage,
};

// ─── Persistence contract ─────────────────────────────────────────────────────

/// Persistence tier declared per artifact in kiki.toml [state].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Persistence {
    Ephemeral,  // process lifetime only
    Session,    // desktop session lifetime (parked to OSTree on park)
    Durable,    // survives restarts; OSTree-backed; migratable to cloud
    Eternal,    // durable + synced across all the user's devices
}

// ─── OSTree checkpoint ────────────────────────────────────────────────────────

/// A committed durable-state snapshot in the OSTree store.
/// Created after every Durable/Eternal step. `ref_hash` is an OSTree commit hash —
/// content-addressed and delta-transferable over R2 to CF fleet worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OstreeCheckpoint {
    pub agent_id:   String,
    pub session_id: String,
    pub step:       u64,
    /// OSTree commit hash (None = checkpoint requested but not yet committed).
    pub ref_hash:   Option<String>,
    pub message:    String,
}

// ─── Runtime snapshot ─────────────────────────────────────────────────────────

/// Serializable snapshot of the harness loop's in-memory state.
///
/// This is the "hot" layer — not in OSTree, transferred directly in the
/// MigrationBundle. It gives the resumed harness full conversation context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeSnapshot {
    pub agent_id:        String,
    pub session_id:      String,
    /// Step index the agent was on when frozen. Resume begins at step + 1.
    pub step:            u64,

    /// Full conversation history — this is what the LLM gets on resume.
    pub messages:        Vec<ConversationMessage>,

    /// Interrupts queued but not resolved at the time of freeze.
    pub interrupt_queue: Vec<Interrupt>,

    /// ControlMode at the moment of freeze.
    pub control_mode:    ControlMode,

    /// Human-readable goal for this session (shown in HUD title bar).
    pub session_label:   String,

    /// Active scenario key (determines remote surface layout on the target node).
    pub scenario:        Option<String>,

    /// Desktop layout at the time of freeze.
    pub layout:          SessionLayout,

    /// App IDs that had open surfaces in this session.
    pub active_apps:     Vec<String>,
}

// ─── Migration bundle ─────────────────────────────────────────────────────────

/// Everything needed to resume a live session on a different host.
///
/// Transfer path:
///   source: agentd → MigrationBundle → POST /v1/fleet/sessions/:id/migrate/bundle
///   fleet DO: stores bundle in R2, signals target node via KV pointer
///   target: pull OSTree delta → restore RuntimeSnapshot → resume harness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationBundle {
    pub bundle_id:    String,
    pub session_id:   String,
    pub runtime:      RuntimeSnapshot,
    pub checkpoint:   OstreeCheckpoint,
    /// MCP tool artifacts that were registered on the source.
    /// Target must reinstall/reattach them before resuming the harness.
    pub artifact_refs: Vec<ArtifactRef>,
    pub created_at_ms: u64,
}

/// Reference to an installed artifact (app/component/durable) active in the session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub artifact_id: String,
    pub version:     String,
    /// OSTree ref for the artifact's installed files on the source node.
    /// The target pulls this ref to reinstall the artifact.
    pub ostree_ref:  String,
}

impl MigrationBundle {
    pub fn bundle_id(session_id: &str, step: u64) -> String {
        format!("{session_id}-step{step}")
    }
}

// ─── StateBackend trait ───────────────────────────────────────────────────────

#[async_trait]
pub trait StateBackend: Send + Sync {
    async fn get(&self, key: &str) -> Result<Option<Value>>;
    async fn set(&self, key: &str, value: Value) -> Result<()>;

    /// Atomically commit current durable state to OSTree. Returns the commit hash.
    /// Called after each Durable/Eternal step, and on session park/freeze.
    async fn commit(&self, message: &str) -> Result<String>;

    /// Capture runtime + durable state into a MigrationBundle.
    async fn snapshot(&self, runtime: RuntimeSnapshot) -> Result<MigrationBundle>;

    /// Restore a MigrationBundle on this host (after receiving from fleet DO).
    async fn restore(&self, bundle: MigrationBundle) -> Result<()>;

    /// Push OSTree delta to a remote registry (R2 bucket via HTTP).
    /// Returns the OSTree ref hash of the pushed commit.
    async fn push(&self, remote: &str) -> Result<String>;

    /// Pull OSTree delta from a remote registry into the local store.
    async fn pull(&self, remote: &str, ref_hash: &str) -> Result<()>;
}
