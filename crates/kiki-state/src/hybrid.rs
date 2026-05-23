use async_trait::async_trait;
use kiki_core::{
    error::Result,
    state::{MigrationBundle, RuntimeSnapshot, StateBackend},
};
use serde_json::Value;
use std::sync::Arc;

use crate::{memory::MemoryBackend, ostree::OstreeBackend};

/// Write-through cache: reads from memory, writes sync to memory + async to OSTree.
///
/// Useful for sessions where low-latency reads are critical (e.g., high-frequency
/// tool calls) but durable writes are still required.
pub struct HybridBackend {
    memory: MemoryBackend,
    ostree: Arc<OstreeBackend>,
}

impl HybridBackend {
    pub fn new(session_id: impl Into<String>, agent_id: impl Into<String>) -> Self {
        let s = session_id.into();
        let a = agent_id.into();
        Self {
            memory: MemoryBackend::default(),
            ostree: Arc::new(OstreeBackend::new(s, a)),
        }
    }
}

#[async_trait]
impl StateBackend for HybridBackend {
    async fn get(&self, key: &str) -> Result<Option<Value>> {
        // Read from memory cache first.
        if let Some(v) = self.memory.get(key).await? {
            return Ok(Some(v));
        }
        // Fall through to OSTree (cold path — only on cache miss).
        self.ostree.get(key).await
    }

    async fn set(&self, key: &str, value: Value) -> Result<()> {
        self.memory.set(key, value.clone()).await?;
        // Fire-and-forget write to OSTree (will be committed on next commit()).
        self.ostree.set(key, value).await
    }

    async fn commit(&self, message: &str) -> Result<String> {
        self.ostree.commit(message).await
    }

    async fn snapshot(&self, runtime: RuntimeSnapshot) -> Result<MigrationBundle> {
        // Always snapshot via OSTree to get a real ref_hash.
        self.ostree.snapshot(runtime).await
    }

    async fn restore(&self, bundle: MigrationBundle) -> Result<()> {
        self.ostree.restore(bundle).await
    }

    async fn push(&self, remote: &str) -> Result<String> {
        self.ostree.push(remote).await
    }

    async fn pull(&self, remote: &str, ref_hash: &str) -> Result<()> {
        self.ostree.pull(remote, ref_hash).await
    }
}
