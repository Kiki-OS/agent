//! Secret storage backends for the OAuth credential store.
//!
//! Tokens are sealed at rest. The production [`SealedFileSecretStore`] uses the
//! kiki-config AEAD primitives (XChaCha20-Poly1305 under the node master key);
//! [`MemorySecretStore`] is for tests. Both are keyed by the credential handle.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

use kiki_config::crypto::{open_secret, seal_secret, MasterKey, Sealed};

use crate::OAuthError;

pub type Result<T> = std::result::Result<T, OAuthError>;

/// Stores opaque secret bytes keyed by a credential handle. Implementations seal
/// at rest; the plaintext handed in/out is the serialized [`crate::TokenSet`].
pub trait SecretStore: Send + Sync {
    fn put(&self, key: &str, plaintext: &[u8]) -> Result<()>;
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>>;
    fn delete(&self, key: &str) -> Result<()>;
}

/// In-memory store for tests (no sealing).
#[derive(Default)]
pub struct MemorySecretStore {
    map: Mutex<HashMap<String, Vec<u8>>>,
}

impl MemorySecretStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl SecretStore for MemorySecretStore {
    fn put(&self, key: &str, plaintext: &[u8]) -> Result<()> {
        self.map.lock().unwrap().insert(key.to_string(), plaintext.to_vec());
        Ok(())
    }
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        Ok(self.map.lock().unwrap().get(key).cloned())
    }
    fn delete(&self, key: &str) -> Result<()> {
        self.map.lock().unwrap().remove(key);
        Ok(())
    }
}

/// Production store: each secret is sealed (XChaCha20-Poly1305 under the node
/// master key, with the handle bound into the AEAD AAD) and written as a JSON
/// `Sealed` blob under `dir/<sanitized-handle>.json`.
pub struct SealedFileSecretStore {
    key: MasterKey,
    dir: PathBuf,
}

const SCOPE: &str = "oauth";

impl SealedFileSecretStore {
    pub fn new(key: MasterKey, dir: impl Into<PathBuf>) -> Self {
        Self { key, dir: dir.into() }
    }

    fn path(&self, key: &str) -> PathBuf {
        let safe: String = key
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') { c } else { '_' })
            .collect();
        self.dir.join(format!("{safe}.json"))
    }
}

impl SecretStore for SealedFileSecretStore {
    fn put(&self, key: &str, plaintext: &[u8]) -> Result<()> {
        std::fs::create_dir_all(&self.dir).map_err(|e| OAuthError::Secret(e.to_string()))?;
        let sealed = seal_secret(&self.key, SCOPE, key, plaintext)
            .map_err(|e| OAuthError::Secret(e.to_string()))?;
        let json = serde_json::to_vec(&sealed).map_err(|e| OAuthError::Secret(e.to_string()))?;
        std::fs::write(self.path(key), json).map_err(|e| OAuthError::Secret(e.to_string()))
    }

    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let raw = match std::fs::read(self.path(key)) {
            Ok(r) => r,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(OAuthError::Secret(e.to_string())),
        };
        let sealed: Sealed =
            serde_json::from_slice(&raw).map_err(|e| OAuthError::Secret(e.to_string()))?;
        let pt = open_secret(&self.key, SCOPE, key, &sealed)
            .map_err(|e| OAuthError::Secret(e.to_string()))?;
        Ok(Some(pt))
    }

    fn delete(&self, key: &str) -> Result<()> {
        match std::fs::remove_file(self.path(key)) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(OAuthError::Secret(e.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sealed_file_store_roundtrips_and_rejects_tamper() {
        let dir = tempfile::tempdir().unwrap();
        let store = SealedFileSecretStore::new(MasterKey::from_bytes([7u8; 32]), dir.path());
        store.put("secrets://app/io.kiki.mail/google/me", b"token-bytes").unwrap();
        let got = store.get("secrets://app/io.kiki.mail/google/me").unwrap();
        assert_eq!(got.as_deref(), Some(&b"token-bytes"[..]));

        // A different key cannot open it.
        let other = SealedFileSecretStore::new(MasterKey::from_bytes([9u8; 32]), dir.path());
        assert!(other.get("secrets://app/io.kiki.mail/google/me").is_err());

        store.delete("secrets://app/io.kiki.mail/google/me").unwrap();
        assert_eq!(store.get("secrets://app/io.kiki.mail/google/me").unwrap(), None);
    }
}
