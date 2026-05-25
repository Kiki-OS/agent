//! Production OSTree-backed state.
//!
//! Session durable state is stored in a content-addressed OSTree repo, so it is
//! delta-transferable: only changed objects move on push/pull (cross-host
//! migration, cloud sync). Each session is an OSTree branch `session/<id>`; its
//! key/value state lives as a `state.json` blob in the committed tree.
//!
//! Migration path:
//!   source: set()… → commit() → push(remote) → build MigrationBundle
//!   target: pull(remote, ref_hash) → restore(bundle) → resume harness
//!
//! Layout under `repo_path` (default `/var/kiki/state`, override `KIKI_OSTREE_REPO`):
//!   repo/                      bare-user OSTree repo (the object store)
//!   worktree/<session_id>/     checked-out working tree (holds state.json)
//!   snapshots/<bundle>.json    persisted MigrationBundles (resume-on-restart)
//!
//! Requires the `ostree` CLI on PATH (present on the Fedora bootc image). The
//! constructor never fails on a host without ostree — operations return
//! `Error::State` instead, so [`crate::HybridBackend`] stays constructible
//! everywhere and tests skip cleanly off-target.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use async_trait::async_trait;
use kiki_core::{
    error::{Error, Result},
    state::{MigrationBundle, OstreeCheckpoint, RuntimeSnapshot, StateBackend},
};
use reqwest::Client;
use serde_json::Value;
use tokio::process::Command;

const STATE_FILE: &str = "state.json";

pub struct OstreeBackend {
    pub repo_path:  PathBuf,
    pub session_id: String,
    pub agent_id:   String,
    cache:          RwLock<BTreeMap<String, Value>>,
}

impl OstreeBackend {
    pub fn new(session_id: impl Into<String>, agent_id: impl Into<String>) -> Self {
        let repo_path =
            std::env::var("KIKI_OSTREE_REPO").unwrap_or_else(|_| "/var/kiki/state".into());
        Self::at(PathBuf::from(repo_path), session_id, agent_id)
    }

    /// Construct with an explicit base path (tests, multi-repo setups).
    pub fn at(
        repo_path:  impl Into<PathBuf>,
        session_id: impl Into<String>,
        agent_id:   impl Into<String>,
    ) -> Self {
        let backend = Self {
            repo_path:  repo_path.into(),
            session_id: session_id.into(),
            agent_id:   agent_id.into(),
            cache:      RwLock::new(BTreeMap::new()),
        };
        backend.load_cache();
        backend
    }

    fn repo(&self)     -> PathBuf { self.repo_path.join("repo") }
    fn worktree(&self) -> PathBuf { self.repo_path.join("worktree").join(&self.session_id) }
    fn branch(&self)   -> String  { format!("session/{}", self.session_id) }
    fn snapshots_dir(&self) -> PathBuf { self.repo_path.join("snapshots") }

    /// Load the worktree's `state.json` into the in-memory cache (best-effort).
    fn load_cache(&self) {
        let path = self.worktree().join(STATE_FILE);
        if let Ok(bytes) = std::fs::read(&path) {
            if let Ok(map) = serde_json::from_slice::<BTreeMap<String, Value>>(&bytes) {
                *self.cache.write().unwrap() = map;
            }
        }
    }

    /// Write the in-memory cache to the worktree atomically (temp + rename).
    fn persist_worktree(&self) -> Result<()> {
        let dir = self.worktree();
        std::fs::create_dir_all(&dir).map_err(|e| Error::State(format!("worktree mkdir: {e}")))?;
        let bytes = {
            let guard = self.cache.read().unwrap();
            serde_json::to_vec(&*guard).map_err(|e| Error::Parse(e.to_string()))?
        };
        let tmp = dir.join(format!("{STATE_FILE}.tmp"));
        std::fs::write(&tmp, &bytes).map_err(|e| Error::State(format!("worktree write: {e}")))?;
        std::fs::rename(&tmp, dir.join(STATE_FILE))
            .map_err(|e| Error::State(format!("worktree rename: {e}")))?;
        Ok(())
    }

    /// Idempotently initialise the bare-user OSTree repo.
    async fn ensure_repo(&self) -> Result<()> {
        let repo = self.repo();
        if repo.join("config").exists() {
            return Ok(());
        }
        std::fs::create_dir_all(&repo).map_err(|e| Error::State(format!("repo mkdir: {e}")))?;
        run_ostree(&["init".as_ref(), "--repo".as_ref(), repo.as_os_str(),
                     "--mode".as_ref(), "bare-user".as_ref()]).await?;
        Ok(())
    }

    /// Initialise a remote (archive-mode) repo if it doesn't exist yet.
    async fn ensure_archive_repo(repo: &Path) -> Result<()> {
        if repo.join("config").exists() {
            return Ok(());
        }
        std::fs::create_dir_all(repo).map_err(|e| Error::State(format!("remote mkdir: {e}")))?;
        run_ostree(&["init".as_ref(), "--repo".as_ref(), repo.as_os_str(),
                     "--mode".as_ref(), "archive".as_ref()]).await?;
        Ok(())
    }

    /// The current commit hash of this session's branch, if any.
    async fn rev_parse(&self) -> Result<Option<String>> {
        let repo = self.repo();
        let out = Command::new("ostree")
            .args(["rev-parse".as_ref(), "--repo".as_ref(), repo.as_os_str(), self.branch().as_ref()])
            .output().await
            .map_err(|e| Error::State(format!("ostree rev-parse spawn: {e}")))?;
        if out.status.success() {
            Ok(parse_checksum(&String::from_utf8_lossy(&out.stdout)))
        } else {
            Ok(None) // branch doesn't exist yet
        }
    }

    /// Check out `commit` into the worktree (overwriting) and reload the cache.
    async fn checkout(&self, commit: &str) -> Result<()> {
        let dir = self.worktree();
        std::fs::create_dir_all(&dir).map_err(|e| Error::State(format!("worktree mkdir: {e}")))?;
        // `--user-mode` checks out without restoring ownership/SELinux xattrs
        // (matches the bare-user repo + lets a non-root agent restore state on any
        // filesystem, incl. tmpfs/overlay that reject security.selinux).
        run_ostree(&["checkout".as_ref(), "--repo".as_ref(), self.repo().as_os_str(),
                     "--user-mode".as_ref(), "--union".as_ref(),
                     commit.as_ref(), dir.as_os_str()]).await?;
        self.load_cache();
        Ok(())
    }

    /// Push the current session ref to an HTTP registry remote.
    ///
    /// `remote_url` is the base URL, e.g. `"https://registry.kiki-os.com"`.
    /// `token` is a Bearer token for authenticated PUT requests.
    ///
    /// Steps:
    ///   1. Commit current state so the branch exists.
    ///   2. Walk `<repo>/objects/` and upload every object file via
    ///      `PUT <remote_url>/objects/<2-char-prefix>/<rest>`.
    ///   3. Upload the resolved commit hash to
    ///      `PUT <remote_url>/refs/heads/session/<session_id>`.
    pub async fn push_to_remote(&self, remote_url: &str, token: &str) -> Result<()> {
        // 1. Ensure the branch exists and state is committed.
        self.commit("pre-push snapshot").await?;

        // 2. Resolve the current commit hash.
        let commit_hash = self.rev_parse().await?
            .ok_or_else(|| Error::State("push_to_remote: branch has no commits".into()))?;

        let client = Client::new();
        let objects_dir = self.repo().join("objects");

        // 3. Walk all objects in <repo>/objects/<xx>/<rest> and PUT each one.
        if objects_dir.is_dir() {
            let prefix_entries = std::fs::read_dir(&objects_dir)
                .map_err(|e| Error::State(format!("objects dir read: {e}")))?;

            for prefix_entry in prefix_entries {
                let prefix_entry = prefix_entry
                    .map_err(|e| Error::State(format!("objects dir entry: {e}")))?;
                let prefix_path = prefix_entry.path();
                if !prefix_path.is_dir() {
                    continue;
                }
                let prefix = prefix_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .ok_or_else(|| Error::State("objects prefix: non-UTF8 path".into()))?
                    .to_string();

                let object_entries = std::fs::read_dir(&prefix_path)
                    .map_err(|e| Error::State(format!("objects subdir read: {e}")))?;

                for object_entry in object_entries {
                    let object_entry = object_entry
                        .map_err(|e| Error::State(format!("objects subdir entry: {e}")))?;
                    let object_path = object_entry.path();
                    if !object_path.is_file() {
                        continue;
                    }
                    let rest = object_path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .ok_or_else(|| Error::State("object filename: non-UTF8 path".into()))?
                        .to_string();

                    let bytes = std::fs::read(&object_path)
                        .map_err(|e| Error::State(format!("object read {rest}: {e}")))?;

                    let url = format!("{remote_url}/objects/{prefix}/{rest}");
                    let resp = client
                        .put(&url)
                        .bearer_auth(token)
                        .body(bytes)
                        .send()
                        .await
                        .map_err(|e| Error::State(format!("PUT {url}: {e}")))?;

                    if !resp.status().is_success() {
                        let status = resp.status();
                        let body = resp.text().await.unwrap_or_default();
                        return Err(Error::State(format!(
                            "PUT {url} failed with {status}: {body}"
                        )));
                    }
                }
            }
        }

        // 4. Upload the ref.
        let ref_url = format!("{remote_url}/refs/heads/session/{}", self.session_id);
        let resp = client
            .put(&ref_url)
            .bearer_auth(token)
            .body(commit_hash.clone())
            .send()
            .await
            .map_err(|e| Error::State(format!("PUT ref {ref_url}: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(Error::State(format!(
                "PUT ref {ref_url} failed with {status}: {body}"
            )));
        }

        Ok(())
    }

    /// Pull a ref from an HTTP registry remote and check it out.
    ///
    /// `remote_url` is the base URL, e.g. `"https://registry.kiki-os.com"`.
    /// `ref_` is the branch name or commit hash to pull (e.g. `"session/abc123"`).
    ///
    /// Steps:
    ///   1. `ostree remote add --no-gpg-verify kiki-cloud <remote_url>` (idempotent).
    ///   2. `ostree pull kiki-cloud <ref_>`.
    ///   3. Check out the pulled commit into the worktree.
    pub async fn pull_from_remote(&self, remote_url: &str, ref_: &str) -> Result<()> {
        self.ensure_repo().await?;
        let repo = self.repo();

        // 1. Add the remote (ignore "already exists" error by removing and re-adding).
        let add_result = run_ostree(&[
            "remote".as_ref(),
            "add".as_ref(),
            "--no-gpg-verify".as_ref(),
            "--repo".as_ref(), repo.as_os_str(),
            "kiki-cloud".as_ref(),
            remote_url.as_ref(),
        ]).await;

        if let Err(e) = add_result {
            let msg = e.to_string();
            if msg.contains("already exists") || msg.contains("exists") {
                // Remove and re-add so we always use the latest URL.
                run_ostree(&[
                    "remote".as_ref(),
                    "delete".as_ref(),
                    "--repo".as_ref(), repo.as_os_str(),
                    "kiki-cloud".as_ref(),
                ]).await?;
                run_ostree(&[
                    "remote".as_ref(),
                    "add".as_ref(),
                    "--no-gpg-verify".as_ref(),
                    "--repo".as_ref(), repo.as_os_str(),
                    "kiki-cloud".as_ref(),
                    remote_url.as_ref(),
                ]).await?;
            } else {
                return Err(e);
            }
        }

        // 2. Pull the ref.
        run_ostree(&[
            "pull".as_ref(),
            "--repo".as_ref(), repo.as_os_str(),
            "kiki-cloud".as_ref(),
            ref_.as_ref(),
        ]).await?;

        // 3. Resolve the pulled commit hash and point the session branch at it.
        let remote_ref = format!("kiki-cloud:{ref_}");
        let out = Command::new("ostree")
            .args(["rev-parse".as_ref(), "--repo".as_ref(), repo.as_os_str(), remote_ref.as_ref()])
            .output().await
            .map_err(|e| Error::State(format!("rev-parse after pull: {e}")))?;
        let target = parse_checksum(&String::from_utf8_lossy(&out.stdout))
            .ok_or_else(|| Error::State(format!(
                "pull_from_remote: cannot resolve pulled ref {ref_}: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            )))?;

        // Point local session branch at the pulled commit.
        run_ostree(&[
            "refs".as_ref(), "--repo".as_ref(), repo.as_os_str(),
            "--create".as_ref(), self.branch().as_ref(), target.as_ref(),
        ]).await?;

        // 4. Check out into the worktree.
        self.checkout(&target).await
    }
}

#[async_trait]
impl StateBackend for OstreeBackend {
    async fn get(&self, key: &str) -> Result<Option<Value>> {
        Ok(self.cache.read().unwrap().get(key).cloned())
    }

    async fn set(&self, key: &str, value: Value) -> Result<()> {
        self.cache.write().unwrap().insert(key.to_string(), value);
        self.persist_worktree()
    }

    async fn commit(&self, message: &str) -> Result<String> {
        self.ensure_repo().await?;
        self.persist_worktree()?;
        let repo = self.repo();
        let stdout = run_ostree(&[
            "commit".as_ref(),
            "--repo".as_ref(), repo.as_os_str(),
            "--branch".as_ref(), self.branch().as_ref(),
            "--subject".as_ref(), message.as_ref(),
            "--tree".as_ref(), format!("dir={}", self.worktree().display()).as_ref(),
        ]).await?;
        parse_checksum(&stdout)
            .ok_or_else(|| Error::State(format!("ostree commit: no checksum in output: {stdout}")))
    }

    async fn snapshot(&self, runtime: RuntimeSnapshot) -> Result<MigrationBundle> {
        let ref_hash = self.commit("pre-migration snapshot").await?;
        let bundle = MigrationBundle {
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
            artifact_refs: Vec::new(),
            created_at_ms: now_ms(),
        };
        let dir = self.snapshots_dir();
        std::fs::create_dir_all(&dir).map_err(|e| Error::State(format!("snapshots mkdir: {e}")))?;
        let json = serde_json::to_vec_pretty(&bundle).map_err(|e| Error::Parse(e.to_string()))?;
        std::fs::write(dir.join(format!("{}.json", bundle.bundle_id)), json)
            .map_err(|e| Error::State(format!("bundle write: {e}")))?;
        Ok(bundle)
    }

    async fn restore(&self, bundle: MigrationBundle) -> Result<()> {
        self.ensure_repo().await?;
        // Check out the committed durable tree (if this host has the objects).
        if let Some(commit) = &bundle.checkpoint.ref_hash {
            if self.rev_parse().await?.is_some() || self.checkout(commit).await.is_ok() {
                // Either the branch already points here, or we checked out the commit.
            }
        }
        // Persist the bundle so a restart can resume from it.
        let dir = self.snapshots_dir();
        std::fs::create_dir_all(&dir).map_err(|e| Error::State(format!("snapshots mkdir: {e}")))?;
        let json = serde_json::to_vec_pretty(&bundle).map_err(|e| Error::Parse(e.to_string()))?;
        std::fs::write(dir.join(format!("{}.json", bundle.bundle_id)), json)
            .map_err(|e| Error::State(format!("bundle write: {e}")))?;
        Ok(())
    }

    async fn push(&self, remote: &str) -> Result<String> {
        // Capture the latest worktree state so push always ships current durable
        // state (and guarantees the branch exists before we mirror it).
        let hash = self.commit("pre-push snapshot").await?;
        let remote_repo = PathBuf::from(remote);
        Self::ensure_archive_repo(&remote_repo).await?;
        // Mirror our branch into the remote object store (only deltas transfer).
        run_ostree(&[
            "pull-local".as_ref(),
            "--repo".as_ref(), remote_repo.as_os_str(),
            self.repo().as_os_str(),
            self.branch().as_ref(),
        ]).await?;
        Ok(hash)
    }

    async fn pull(&self, remote: &str, ref_hash: &str) -> Result<()> {
        self.ensure_repo().await?;
        let remote_repo = PathBuf::from(remote);
        // Pull the requested ref (commit hash or branch) from the remote store.
        run_ostree(&[
            "pull-local".as_ref(),
            "--repo".as_ref(), self.repo().as_os_str(),
            remote_repo.as_os_str(),
            ref_hash.as_ref(),
        ]).await?;
        // Make the pulled commit the session branch head + check it out locally.
        let target = if looks_like_checksum(ref_hash) { ref_hash.to_string() } else {
            // ref_hash was a branch name: resolve it in the remote.
            let out = Command::new("ostree")
                .args(["rev-parse".as_ref(), "--repo".as_ref(), remote_repo.as_os_str(), ref_hash.as_ref()])
                .output().await
                .map_err(|e| Error::State(format!("remote rev-parse: {e}")))?;
            parse_checksum(&String::from_utf8_lossy(&out.stdout))
                .ok_or_else(|| Error::State("pull: cannot resolve ref".into()))?
        };
        run_ostree(&[
            "refs".as_ref(), "--repo".as_ref(), self.repo().as_os_str(),
            "--create".as_ref(), self.branch().as_ref(), target.as_ref(),
        ]).await?;
        self.checkout(&target).await
    }
}

/// Run `ostree <args>`, returning stdout on success or `Error::State` with stderr.
async fn run_ostree(args: &[&std::ffi::OsStr]) -> Result<String> {
    let out = Command::new("ostree").args(args).output().await
        .map_err(|e| Error::State(format!("ostree spawn failed (is it installed?): {e}")))?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    } else {
        Err(Error::State(format!(
            "ostree {:?} failed: {}",
            args.iter().map(|a| a.to_string_lossy()).collect::<Vec<_>>().join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        )))
    }
}

/// The last non-empty whitespace-trimmed token that is a 64-char hex checksum.
fn parse_checksum(s: &str) -> Option<String> {
    s.split_whitespace().rev().find(|t| looks_like_checksum(t)).map(str::to_string)
}

fn looks_like_checksum(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    /// Skip a test cleanly when the `ostree` CLI isn't available (e.g. macOS dev
    /// box). The real coverage runs on the Fedora bootc target.
    fn ostree_available() -> bool {
        std::process::Command::new("ostree").arg("--version").output()
            .map(|o| o.status.success()).unwrap_or(false)
    }

    #[tokio::test]
    async fn commit_and_get_round_trip() {
        if !ostree_available() { eprintln!("skip: ostree not installed"); return; }
        let dir = tempfile::tempdir().unwrap();
        let be = OstreeBackend::at(dir.path(), "s1", "a1");
        be.set("k", json!({ "n": 1 })).await.unwrap();
        let c1 = be.commit("first").await.unwrap();
        assert!(looks_like_checksum(&c1), "commit hash: {c1}");
        assert_eq!(be.get("k").await.unwrap(), Some(json!({ "n": 1 })));

        be.set("k", json!({ "n": 2 })).await.unwrap();
        let c2 = be.commit("second").await.unwrap();
        assert_ne!(c1, c2, "commit hash must change with state");

        // A fresh backend over the same repo path loads the persisted worktree.
        let reopened = OstreeBackend::at(dir.path(), "s1", "a1");
        assert_eq!(reopened.get("k").await.unwrap(), Some(json!({ "n": 2 })));
    }

    #[tokio::test]
    async fn push_pull_migrates_state_cross_repo() {
        if !ostree_available() { eprintln!("skip: ostree not installed"); return; }
        let src_dir    = tempfile::tempdir().unwrap();
        let remote_dir = tempfile::tempdir().unwrap();
        let dst_dir    = tempfile::tempdir().unwrap();
        let remote = remote_dir.path().join("registry"); // archive repo path

        // Source node writes + commits + pushes to the shared remote.
        let src = OstreeBackend::at(src_dir.path(), "sess", "agent");
        src.set("doc", json!("hello cross-host")).await.unwrap();
        let pushed = src.push(remote.to_str().unwrap()).await.unwrap();
        assert!(looks_like_checksum(&pushed));

        // Target node (separate repo) pulls the commit and resumes the state.
        let dst = OstreeBackend::at(dst_dir.path(), "sess", "agent");
        assert_eq!(dst.get("doc").await.unwrap(), None, "dst starts empty");
        dst.pull(remote.to_str().unwrap(), &pushed).await.unwrap();
        assert_eq!(dst.get("doc").await.unwrap(), Some(json!("hello cross-host")));
    }

    #[tokio::test]
    async fn snapshot_carries_real_commit_hash() {
        if !ostree_available() { eprintln!("skip: ostree not installed"); return; }
        let dir = tempfile::tempdir().unwrap();
        let be = OstreeBackend::at(dir.path(), "s2", "a2");
        be.set("x", json!(42)).await.unwrap();
        let runtime = RuntimeSnapshot {
            agent_id: "a2".into(), session_id: "s2".into(), step: 1,
            messages: vec![], interrupt_queue: vec![],
            control_mode: kiki_core::context::ControlMode::AgentMode,
            session_label: "t".into(), scenario: None,
            layout: Default::default(), active_apps: vec![], app_states: Default::default(),
        };
        let bundle = be.snapshot(runtime).await.unwrap();
        let hash = bundle.checkpoint.ref_hash.unwrap();
        assert!(looks_like_checksum(&hash), "snapshot ref_hash: {hash}");
        assert!(dir.path().join("snapshots/s2-step1.json").exists());
    }

    // -------------------------------------------------------------------------
    // HTTP remote tests
    // -------------------------------------------------------------------------

    /// Minimal async HTTP mock server: records (method, path, body) for each
    /// request and always responds 200 OK.  Returns the bound address and the
    /// shared request log.
    async fn spawn_mock_server() -> (std::net::SocketAddr, Arc<Mutex<Vec<(String, String, Vec<u8>)>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let log: Arc<Mutex<Vec<(String, String, Vec<u8>)>>> = Arc::new(Mutex::new(Vec::new()));
        let log_clone = log.clone();

        tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else { break };
                let log_ref = log_clone.clone();
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 65536];
                    let n = stream.read(&mut buf).await.unwrap_or(0);
                    if n == 0 { return; }
                    let raw = String::from_utf8_lossy(&buf[..n]);

                    // Parse the request line.
                    let first_line = raw.lines().next().unwrap_or("");
                    let mut parts = first_line.splitn(3, ' ');
                    let method = parts.next().unwrap_or("").to_string();
                    let path   = parts.next().unwrap_or("").to_string();

                    // Extract body (after \r\n\r\n).
                    let body = if let Some(idx) = raw.find("\r\n\r\n") {
                        buf[idx + 4..n].to_vec()
                    } else {
                        Vec::new()
                    };

                    log_ref.lock().unwrap().push((method, path, body));

                    let resp = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n";
                    let _ = stream.write_all(resp).await;
                });
            }
        });

        (addr, log)
    }

    #[tokio::test]
    async fn push_to_remote_uploads_objects_and_ref() {
        if !ostree_available() { eprintln!("skip: ostree not installed"); return; }

        let dir = tempfile::tempdir().unwrap();
        let be = OstreeBackend::at(dir.path(), "sess-push", "agent");

        // Write some state so there are real objects in the repo.
        be.set("hello", json!("world")).await.unwrap();
        be.commit("initial").await.unwrap();

        let (addr, log) = spawn_mock_server().await;
        let remote_url = format!("http://{addr}");

        be.push_to_remote(&remote_url, "test-token").await.unwrap();

        // Give the spawned tasks a moment to flush.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let requests = log.lock().unwrap();

        // At least one object was PUT.
        let obj_puts: Vec<_> = requests.iter()
            .filter(|(m, p, _)| m == "PUT" && p.starts_with("/objects/"))
            .collect();
        assert!(!obj_puts.is_empty(), "expected object PUTs, got: {requests:?}");

        // The ref was PUT.
        let ref_puts: Vec<_> = requests.iter()
            .filter(|(m, p, _)| m == "PUT" && p.contains("/refs/heads/session/sess-push"))
            .collect();
        assert_eq!(ref_puts.len(), 1, "expected exactly one ref PUT, got: {ref_puts:?}");

        // The ref body is a valid 64-char hex checksum.
        let ref_body = std::str::from_utf8(&ref_puts[0].2).unwrap_or("").trim().to_string();
        assert!(
            looks_like_checksum(&ref_body),
            "ref body should be a checksum, got: '{ref_body}'"
        );
    }

    #[tokio::test]
    async fn push_to_remote_returns_error_on_server_failure() {
        if !ostree_available() { eprintln!("skip: ostree not installed"); return; }

        let dir = tempfile::tempdir().unwrap();
        let be = OstreeBackend::at(dir.path(), "sess-err", "agent");
        be.set("k", json!(1)).await.unwrap();
        be.commit("c").await.unwrap();

        // Spawn a server that always returns 401.
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else { break };
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 4096];
                    let _ = stream.read(&mut buf).await;
                    let resp = b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\n\r\n";
                    let _ = stream.write_all(resp).await;
                });
            }
        });

        let remote_url = format!("http://{addr}");
        let result = be.push_to_remote(&remote_url, "bad-token").await;
        assert!(result.is_err(), "expected error on 401, got Ok");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("401") || msg.contains("failed"), "unexpected error: {msg}");
    }

    #[tokio::test]
    async fn pull_from_remote_returns_error_on_missing_ostree() {
        // This test verifies that pull_from_remote propagates errors properly
        // when the remote server returns an error (or ostree is absent).
        // We use a URL that points at a non-existent server to force an error.
        if !ostree_available() { eprintln!("skip: ostree not installed"); return; }

        let dir = tempfile::tempdir().unwrap();
        let be = OstreeBackend::at(dir.path(), "sess-pull-err", "agent");

        // Point at a port that has nothing listening — ostree pull will fail.
        let result = be
            .pull_from_remote("http://127.0.0.1:1", "session/sess-pull-err")
            .await;
        assert!(result.is_err(), "expected error pulling from dead server");
    }
}
