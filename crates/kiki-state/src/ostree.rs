use async_trait::async_trait;
use kiki_core::{
    error::Result,
    state::{MigrationBundle, OstreeCheckpoint, RuntimeSnapshot, StateBackend},
};
use serde_json::Value;

/// Production OSTree-backed state.
///
/// Every Durable/Eternal PRA step calls commit() → creates an OSTree commit.
/// The commit hash is content-addressed; only deltas transfer on push/pull.
///
/// Migration path:
///   source: commit() → push(registry_remote) → build MigrationBundle
///   target: pull(registry_remote, ref_hash) → restore(bundle) → resume PRA
pub struct OstreeBackend {
    pub repo_path:  std::path::PathBuf,
    pub session_id: String,
    pub agent_id:   String,
}

impl OstreeBackend {
    pub fn new(session_id: impl Into<String>, agent_id: impl Into<String>) -> Self {
        Self {
            repo_path:  std::path::PathBuf::from("/var/kiki/state"),
            session_id: session_id.into(),
            agent_id:   agent_id.into(),
        }
    }
}

#[async_trait]
impl StateBackend for OstreeBackend {
    async fn get(&self, _key: &str) -> Result<Option<Value>> {
        // TODO: read from OSTree working tree checkout
        Ok(None)
    }

    async fn set(&self, _key: &str, _value: Value) -> Result<()> {
        // TODO: write to staged OSTree tree (committed on next commit() call)
        Ok(())
    }

    async fn commit(&self, message: &str) -> Result<String> {
        // TODO: ostree commit --repo=/var/kiki/state \
        //         --branch=session/{session_id} --subject={message}
        tracing::debug!(
            session = %self.session_id,
            msg = message,
            "ostree commit (stub)"
        );
        Ok(format!("stub-ref-{}", self.session_id))
    }

    async fn snapshot(&self, runtime: RuntimeSnapshot) -> Result<MigrationBundle> {
        let ref_hash = self.commit("pre-migration snapshot").await?;
        Ok(MigrationBundle {
            bundle_id:    MigrationBundle::bundle_id(&runtime.session_id, runtime.step),
            session_id:   runtime.session_id.clone(),
            checkpoint:   OstreeCheckpoint {
                agent_id:   runtime.agent_id.clone(),
                session_id: runtime.session_id.clone(),
                step:       runtime.step,
                ref_hash:   Some(ref_hash),
                message:    "migration snapshot".into(),
            },
            runtime,
            artifact_refs: Vec::new(), // TODO: collect from MCP hub
            created_at_ms: now_ms(),
        })
    }

    async fn restore(&self, bundle: MigrationBundle) -> Result<()> {
        // TODO:
        //   1. ostree pull <registry> <bundle.checkpoint.ref_hash>
        //   2. ostree checkout into /var/kiki/state/sessions/<id>/
        //   3. for each artifact_ref: ostree pull + symlink
        tracing::info!(bundle_id = %bundle.bundle_id, "ostree restore (stub)");
        Ok(())
    }

    async fn push(&self, remote: &str) -> Result<String> {
        // TODO: ostree push --repo=/var/kiki/state <remote> session/{session_id}
        tracing::info!(remote, session = %self.session_id, "ostree push (stub)");
        Ok(format!("stub-ref-{}", self.session_id))
    }

    async fn pull(&self, remote: &str, ref_hash: &str) -> Result<()> {
        // TODO: ostree pull --repo=/var/kiki/state <remote> <ref_hash>
        tracing::info!(remote, ref_hash, "ostree pull (stub)");
        Ok(())
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
