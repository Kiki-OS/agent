//! On-disk LRU cache for vault blobs.
//!
//! Layout:
//! ```text
//! <root>/blobs/<sha256-prefix-2>/<sha256>     # content-addressed blob files
//! <root>/index/<scope>/<owner>/<path>.json    # path → {sha256, revision}
//! <root>/pins                                 # newline-delimited pinned paths
//! ```
//!
//! Eviction: when capacity is exceeded, LRU among unpinned blobs is dropped.
//! The cache is process-local; concurrent `agentd` instances are not expected
//! (one daemon per node). A simple lockfile prevents accidental dual-mount.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;
use tokio::fs;
use tokio::io::AsyncWriteExt;

use crate::uri::VaultUri;

#[derive(Debug, Error)]
pub enum CacheError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("blob hash mismatch: expected {expected}, got {actual}")]
    HashMismatch { expected: String, actual: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexEntry {
    pub sha256:   String,
    pub revision: u64,
    pub size:     u64,
    pub accessed: u64, // unix millis
}

pub struct LocalCache {
    root:      PathBuf,
    capacity:  u64,
    pins:      Mutex<Vec<String>>, // pinned URI strings
}

impl LocalCache {
    pub fn new(root: PathBuf, capacity_bytes: u64) -> Self {
        Self {
            root,
            capacity: capacity_bytes,
            pins: Mutex::new(Vec::new()),
        }
    }

    pub async fn init(&self) -> Result<(), CacheError> {
        fs::create_dir_all(self.root.join("blobs")).await?;
        fs::create_dir_all(self.root.join("index")).await?;
        let pin_path = self.root.join("pins");
        if let Ok(s) = fs::read_to_string(&pin_path).await {
            let mut p = self.pins.lock().expect("poisoned");
            *p = s.lines().map(String::from).collect();
        }
        Ok(())
    }

    pub fn pin(&self, uri: &VaultUri) {
        let mut p = self.pins.lock().expect("poisoned");
        let s = uri.to_string();
        if !p.contains(&s) {
            p.push(s);
        }
    }

    pub fn unpin(&self, uri: &VaultUri) {
        let mut p = self.pins.lock().expect("poisoned");
        let s = uri.to_string();
        p.retain(|x| x != &s);
    }

    pub fn is_pinned(&self, uri: &VaultUri) -> bool {
        let p = self.pins.lock().expect("poisoned");
        p.iter().any(|x| x == &uri.to_string())
    }

    /// Persist pinned-paths file. Call after batch pin/unpin operations.
    pub async fn flush_pins(&self) -> Result<(), CacheError> {
        let snapshot = self.pins.lock().expect("poisoned").join("\n");
        let path = self.root.join("pins");
        fs::write(&path, snapshot).await?;
        Ok(())
    }

    /// Insert a blob into the cache. Computes sha256, verifies if `expected_hash`
    /// is provided, then writes. Returns the canonical hash.
    pub async fn insert_blob(
        &self,
        bytes: &[u8],
        expected_hash: Option<&str>,
    ) -> Result<String, CacheError> {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        let hash = hex::encode(hasher.finalize());
        if let Some(exp) = expected_hash {
            if !exp.eq_ignore_ascii_case(&hash) {
                return Err(CacheError::HashMismatch {
                    expected: exp.to_string(),
                    actual: hash,
                });
            }
        }
        let path = self.blob_path(&hash);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await?;
        }
        if !path.exists() {
            let mut f = fs::File::create(&path).await?;
            f.write_all(bytes).await?;
            f.flush().await?;
        }
        Ok(hash)
    }

    pub async fn read_blob(&self, hash: &str) -> Result<Vec<u8>, CacheError> {
        let path = self.blob_path(hash);
        let bytes = fs::read(&path).await?;
        Ok(bytes)
    }

    pub fn blob_path(&self, hash: &str) -> PathBuf {
        let prefix = &hash[..2.min(hash.len())];
        self.root.join("blobs").join(prefix).join(hash)
    }

    /// Record path→blob mapping. Updates "accessed" timestamp for LRU.
    pub async fn record(
        &self,
        uri: &VaultUri,
        sha256: &str,
        revision: u64,
        size: u64,
    ) -> Result<(), CacheError> {
        let path = self.index_path(uri);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await?;
        }
        let entry = IndexEntry {
            sha256: sha256.to_string(),
            revision,
            size,
            accessed: now_millis(),
        };
        fs::write(&path, serde_json::to_vec(&entry)?).await?;
        Ok(())
    }

    pub async fn lookup(&self, uri: &VaultUri) -> Option<IndexEntry> {
        let path = self.index_path(uri);
        let bytes = fs::read(&path).await.ok()?;
        let entry: IndexEntry = serde_json::from_slice(&bytes).ok()?;
        Some(entry)
    }

    pub fn index_path(&self, uri: &VaultUri) -> PathBuf {
        self.root
            .join("index")
            .join(uri.scope.as_str())
            .join(&uri.owner_id)
            .join(format!("{}.json", uri.path))
    }

    /// Walk index, compute total cached size, evict LRU among unpinned until
    /// total ≤ capacity. Returns bytes evicted.
    pub async fn evict_to_fit(&self) -> Result<u64, CacheError> {
        let mut entries: Vec<(PathBuf, IndexEntry, String)> = Vec::new();
        let index_root = self.root.join("index");
        if !index_root.exists() {
            return Ok(0);
        }

        // Async dir walk — collect every index entry.
        let mut stack = vec![index_root];
        while let Some(dir) = stack.pop() {
            let mut rd = fs::read_dir(&dir).await?;
            while let Some(entry) = rd.next_entry().await? {
                let p = entry.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().and_then(|e| e.to_str()) == Some("json") {
                    if let Ok(bytes) = fs::read(&p).await {
                        if let Ok(ie) = serde_json::from_slice::<IndexEntry>(&bytes) {
                            let uri_str = self.index_to_uri_string(&p);
                            entries.push((p, ie, uri_str));
                        }
                    }
                }
            }
        }

        // Total occupied (per *unique* sha256 — paths can share blobs).
        let mut by_blob: HashMap<String, u64> = HashMap::new();
        for (_, ie, _) in &entries {
            by_blob.insert(ie.sha256.clone(), ie.size);
        }
        let total: u64 = by_blob.values().sum();
        if total <= self.capacity {
            return Ok(0);
        }

        // Sort entries oldest first, skip pinned.
        let pins = self.pins.lock().expect("poisoned").clone();
        entries.sort_by_key(|(_, ie, _)| ie.accessed);

        let mut evicted: u64 = 0;
        for (idx_path, ie, uri) in entries {
            if pins.iter().any(|p| p == &uri) {
                continue;
            }
            // Remove index entry; remove blob iff no other index entry still references it.
            let _ = fs::remove_file(&idx_path).await;
            let still_referenced = self.is_blob_referenced(&ie.sha256).await.unwrap_or(true);
            if !still_referenced {
                let _ = fs::remove_file(self.blob_path(&ie.sha256)).await;
                evicted = evicted.saturating_add(ie.size);
            }
            // Recompute total occupied to know when to stop.
            let new_total = total.saturating_sub(evicted);
            if new_total <= self.capacity {
                break;
            }
        }
        Ok(evicted)
    }

    async fn is_blob_referenced(&self, sha256: &str) -> Result<bool, CacheError> {
        let index_root = self.root.join("index");
        let mut stack = vec![index_root];
        while let Some(dir) = stack.pop() {
            if !dir.exists() {
                continue;
            }
            let mut rd = fs::read_dir(&dir).await?;
            while let Some(entry) = rd.next_entry().await? {
                let p = entry.path();
                if p.is_dir() {
                    stack.push(p);
                } else if let Ok(bytes) = fs::read(&p).await {
                    if let Ok(ie) = serde_json::from_slice::<IndexEntry>(&bytes) {
                        if ie.sha256 == sha256 {
                            return Ok(true);
                        }
                    }
                }
            }
        }
        Ok(false)
    }

    fn index_to_uri_string(&self, p: &Path) -> String {
        let rel = p.strip_prefix(self.root.join("index")).unwrap_or(p);
        let mut comps: Vec<String> = rel
            .components()
            .map(|c| c.as_os_str().to_string_lossy().to_string())
            .collect();
        // First component = scope, second = owner, rest = path (last with .json stripped)
        if let Some(last) = comps.last_mut() {
            if let Some(stripped) = last.strip_suffix(".json") {
                *last = stripped.to_string();
            }
        }
        if comps.len() < 3 {
            return String::new();
        }
        let scope = &comps[0];
        let owner = &comps[1];
        let path = comps[2..].join("/");
        format!("vault://{scope}/{owner}/{path}")
    }
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uri::{VaultScope, VaultUri};

    fn uri(path: &str) -> VaultUri {
        VaultUri {
            scope:    VaultScope::Personal,
            owner_id: "u".into(),
            path:     path.into(),
        }
    }

    #[tokio::test]
    async fn blob_insert_then_read() {
        let dir = tempfile::tempdir().unwrap();
        let c = LocalCache::new(dir.path().to_path_buf(), 1024);
        c.init().await.unwrap();
        let hash = c.insert_blob(b"hello", None).await.unwrap();
        let back = c.read_blob(&hash).await.unwrap();
        assert_eq!(back, b"hello");
    }

    #[tokio::test]
    async fn hash_mismatch_rejects() {
        let dir = tempfile::tempdir().unwrap();
        let c = LocalCache::new(dir.path().to_path_buf(), 1024);
        c.init().await.unwrap();
        let err = c.insert_blob(b"hello", Some("ffff")).await.unwrap_err();
        assert!(matches!(err, CacheError::HashMismatch { .. }));
    }

    #[tokio::test]
    async fn pinning_persists() {
        let dir = tempfile::tempdir().unwrap();
        let c = LocalCache::new(dir.path().to_path_buf(), 1024);
        c.init().await.unwrap();
        let u = uri("note.md");
        c.pin(&u);
        assert!(c.is_pinned(&u));
        c.flush_pins().await.unwrap();
        // Re-open and confirm pin survived.
        let c2 = LocalCache::new(dir.path().to_path_buf(), 1024);
        c2.init().await.unwrap();
        assert!(c2.is_pinned(&u));
    }

    #[tokio::test]
    async fn eviction_drops_unpinned_first() {
        let dir = tempfile::tempdir().unwrap();
        // 64 bytes total capacity, three 30-byte blobs, two unpinned.
        let c = LocalCache::new(dir.path().to_path_buf(), 64);
        c.init().await.unwrap();
        let big_a = vec![1u8; 30];
        let big_b = vec![2u8; 30];
        let big_c = vec![3u8; 30];
        let h_a = c.insert_blob(&big_a, None).await.unwrap();
        let h_b = c.insert_blob(&big_b, None).await.unwrap();
        let h_c = c.insert_blob(&big_c, None).await.unwrap();
        let u_a = uri("a");
        let u_b = uri("b");
        let u_c = uri("c");
        c.record(&u_a, &h_a, 1, 30).await.unwrap();
        // Force u_b to be the most recently accessed.
        std::thread::sleep(std::time::Duration::from_millis(2));
        c.record(&u_b, &h_b, 1, 30).await.unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2));
        c.record(&u_c, &h_c, 1, 30).await.unwrap();
        c.pin(&u_b);
        let evicted = c.evict_to_fit().await.unwrap();
        assert!(evicted > 0);
        // Pinned blob must survive.
        assert!(c.blob_path(&h_b).exists());
    }
}
