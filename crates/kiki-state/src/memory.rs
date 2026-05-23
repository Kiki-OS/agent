use async_trait::async_trait;
use kiki_core::{
    error::Result,
    state::{MigrationBundle, OstreeCheckpoint, RuntimeSnapshot, StateBackend},
};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::RwLock;

/// In-memory state backend for ephemeral/session persistence and tests.
/// Does not persist across restarts. Not migratable via OSTree (no ref_hash).
#[derive(Default)]
pub struct MemoryBackend(RwLock<HashMap<String, Value>>);

#[async_trait]
impl StateBackend for MemoryBackend {
    async fn get(&self, key: &str) -> Result<Option<Value>> {
        Ok(self.0.read().unwrap().get(key).cloned())
    }

    async fn set(&self, key: &str, value: Value) -> Result<()> {
        self.0.write().unwrap().insert(key.to_string(), value);
        Ok(())
    }

    async fn commit(&self, _message: &str) -> Result<String> {
        Ok(String::new()) // no-op for ephemeral
    }

    async fn snapshot(&self, runtime: RuntimeSnapshot) -> Result<MigrationBundle> {
        Ok(MigrationBundle {
            bundle_id:    MigrationBundle::bundle_id(&runtime.session_id, runtime.step),
            session_id:   runtime.session_id.clone(),
            checkpoint:   OstreeCheckpoint {
                agent_id:   runtime.agent_id.clone(),
                session_id: runtime.session_id.clone(),
                step:       runtime.step,
                ref_hash:   None, // memory backend has no OSTree ref
                message:    "memory snapshot".into(),
            },
            runtime,
            artifact_refs: Vec::new(),
            created_at_ms: 0,
        })
    }

    async fn restore(&self, _bundle: MigrationBundle) -> Result<()> {
        Ok(()) // caller restores RuntimeSnapshot into Context directly
    }

    async fn push(&self, _remote: &str) -> Result<String> {
        Ok(String::new()) // no-op
    }

    async fn pull(&self, _remote: &str, _ref_hash: &str) -> Result<()> {
        Ok(()) // no-op
    }
}
