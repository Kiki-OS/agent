//! Durable file-backed state backend.
//!
//! Persists the agent's key/value durable state to a JSON file under a state
//! directory, so it **survives process restart and reboot** (unlike
//! [`MemoryBackend`](crate::MemoryBackend)). Writes are atomic (temp + rename).
//! `commit` returns the sha256 of the serialized state as a content ref.
//!
//! This is the local-durability backend for runtime/session state living under
//! the mutable `/var/kiki/state`. OSTree (the immutable base) and cloud sync are
//! separate concerns — `push`/`pull` (remote replication) are no-ops here.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::RwLock;

use async_trait::async_trait;
use kiki_core::{
    error::{Error, Result},
    state::{MigrationBundle, OstreeCheckpoint, RuntimeSnapshot, StateBackend},
};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Durable key/value state persisted under `dir`.
pub struct FileBackend {
    dir:   PathBuf,
    state: RwLock<BTreeMap<String, Value>>,
}

const STATE_FILE: &str = "state.json";

impl FileBackend {
    /// Open (or create) a state directory, loading any existing state.
    pub fn open(dir: impl Into<PathBuf>) -> Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        let state = match std::fs::read(dir.join(STATE_FILE)) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map_err(|e| Error::Parse(format!("state.json: {e}")))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => BTreeMap::new(),
            Err(e) => return Err(Error::Io(e.to_string())),
        };
        Ok(Self { dir, state: RwLock::new(state) })
    }

    /// Serialize the current state deterministically (sorted keys).
    fn serialize(&self) -> Result<Vec<u8>> {
        let guard = self.state.read().unwrap();
        serde_json::to_vec(&*guard).map_err(|e| Error::Parse(e.to_string()))
    }

    /// Atomically persist current state to `dir/state.json`.
    fn persist(&self) -> Result<()> {
        let bytes = self.serialize()?;
        let tmp = self.dir.join(format!("{STATE_FILE}.tmp"));
        std::fs::write(&tmp, &bytes)?;
        std::fs::rename(&tmp, self.dir.join(STATE_FILE))?;
        Ok(())
    }

    fn content_ref(&self) -> Result<String> {
        let bytes = self.serialize()?;
        let digest = Sha256::digest(&bytes);
        Ok(digest.iter().map(|b| format!("{b:02x}")).collect())
    }

    fn now_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }

    fn snapshots_dir(&self) -> PathBuf {
        self.dir.join("snapshots")
    }
}

#[async_trait]
impl StateBackend for FileBackend {
    async fn get(&self, key: &str) -> Result<Option<Value>> {
        Ok(self.state.read().unwrap().get(key).cloned())
    }

    async fn set(&self, key: &str, value: Value) -> Result<()> {
        self.state.write().unwrap().insert(key.to_string(), value);
        self.persist()
    }

    async fn commit(&self, _message: &str) -> Result<String> {
        self.persist()?;
        self.content_ref()
    }

    async fn snapshot(&self, runtime: RuntimeSnapshot) -> Result<MigrationBundle> {
        let ref_hash = self.content_ref()?;
        let bundle = MigrationBundle {
            bundle_id:     MigrationBundle::bundle_id(&runtime.session_id, runtime.step),
            session_id:    runtime.session_id.clone(),
            checkpoint:    OstreeCheckpoint {
                agent_id:   runtime.agent_id.clone(),
                session_id: runtime.session_id.clone(),
                step:       runtime.step,
                ref_hash:   Some(ref_hash),
                message:    "file snapshot".into(),
            },
            runtime,
            artifact_refs: Vec::new(),
            created_at_ms: Self::now_ms(),
        };
        // Persist the bundle so a restart can resume from it.
        let dir = self.snapshots_dir();
        std::fs::create_dir_all(&dir)?;
        let json =
            serde_json::to_vec_pretty(&bundle).map_err(|e| Error::Parse(e.to_string()))?;
        std::fs::write(dir.join(format!("{}.json", bundle.bundle_id)), json)?;
        Ok(bundle)
    }

    async fn restore(&self, bundle: MigrationBundle) -> Result<()> {
        // The harness restores the RuntimeSnapshot into its Context directly; the
        // backend persists the bundle so it is recoverable across restarts.
        let dir = self.snapshots_dir();
        std::fs::create_dir_all(&dir)?;
        let json =
            serde_json::to_vec_pretty(&bundle).map_err(|e| Error::Parse(e.to_string()))?;
        std::fs::write(dir.join(format!("{}.json", bundle.bundle_id)), json)?;
        Ok(())
    }

    async fn push(&self, _remote: &str) -> Result<String> {
        // Local durable backend; remote replication is OSTree/hybrid's job.
        Ok(String::new())
    }

    async fn pull(&self, _remote: &str, _ref_hash: &str) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn state_survives_reopen() {
        let dir = tempfile::tempdir().unwrap();
        {
            let backend = FileBackend::open(dir.path()).unwrap();
            backend.set("session/1", json!({ "step": 3 })).await.unwrap();
            backend.set("mode", json!("agent")).await.unwrap();
        }
        // New instance from the same dir = simulated restart.
        let reopened = FileBackend::open(dir.path()).unwrap();
        assert_eq!(reopened.get("session/1").await.unwrap(), Some(json!({ "step": 3 })));
        assert_eq!(reopened.get("mode").await.unwrap(), Some(json!("agent")));
        assert_eq!(reopened.get("missing").await.unwrap(), None);
    }

    #[tokio::test]
    async fn commit_ref_changes_with_state() {
        let dir = tempfile::tempdir().unwrap();
        let backend = FileBackend::open(dir.path()).unwrap();
        backend.set("a", json!(1)).await.unwrap();
        let r1 = backend.commit("one").await.unwrap();
        assert_eq!(r1.len(), 64); // sha256 hex
        backend.set("a", json!(2)).await.unwrap();
        let r2 = backend.commit("two").await.unwrap();
        assert_ne!(r1, r2, "ref must change when state changes");
    }

    #[tokio::test]
    async fn snapshot_has_ref_hash_and_persists_bundle() {
        let dir = tempfile::tempdir().unwrap();
        let backend = FileBackend::open(dir.path()).unwrap();
        backend.set("k", json!("v")).await.unwrap();
        let runtime = RuntimeSnapshot {
            agent_id:        "a1".into(),
            session_id:      "s1".into(),
            step:            2,
            messages:        vec![],
            interrupt_queue: vec![],
            control_mode:    kiki_core::context::ControlMode::AgentMode,
            session_label:   "test".into(),
            scenario:        None,
            layout:          Default::default(),
            active_apps:     vec![],
            app_states:      Default::default(),
        };
        let bundle = backend.snapshot(runtime).await.unwrap();
        assert!(bundle.checkpoint.ref_hash.is_some());
        assert!(dir.path().join("snapshots/s1-step2.json").exists());
    }
}
