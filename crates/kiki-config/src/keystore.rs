//! OS keystore adapter for caching the unlocked [`MasterKey`].
//!
//! After the user has unlocked the master key once (via password or recovery
//! code), the client caches the 32 raw bytes in the platform secure store so
//! subsequent boots can re-derive without a password prompt — as long as the
//! OS unlock state (login session, biometric, etc.) trusts the request.
//!
//! - **Linux:** libsecret via the `secret-service` D-Bus interface
//!   (gnome-keyring, KeePassXC, KWallet bridge). Schema = `com.kiki-os.config`,
//!   `account = <user_id>`.
//! - **macOS:** Keychain Services via `security` framework. Service =
//!   `com.kiki-os.config`, account = `<user_id>`.
//! - **Other:** [`Keystore::Memory`] only — caller MUST require password each boot.
//!
//! Mobile (kiki-app) uses `expo-secure-store` in JS; that surface is outside
//! this crate but shares the same scheme.
//!
//! ## Safety
//!
//! Loss of the keystore (re-imaged disk, lost device) is recoverable via
//! password OR recovery code — the master key is never the only copy.

use std::collections::HashMap;
use std::sync::Mutex;

use thiserror::Error;
use zeroize::Zeroize;

use crate::crypto::{MasterKey, KEY_LEN};

/// Pluggable keystore backend. The default constructor selects the platform
/// implementation; tests use [`Keystore::memory`].
pub enum Keystore {
    Memory(Mutex<HashMap<String, [u8; KEY_LEN]>>),
    #[cfg(target_os = "linux")]
    Secret(linux::SecretService),
    #[cfg(target_os = "macos")]
    Keychain(macos::Keychain),
}

#[derive(Debug, Error)]
pub enum KeystoreError {
    #[error("keystore not available on this platform; use Keystore::memory()")]
    Unavailable,
    #[error("entry not found for account `{0}`")]
    NotFound(String),
    #[error("backend error: {0}")]
    Backend(String),
}

impl Keystore {
    /// In-memory backend. Loses contents on process exit. Use for tests and
    /// for headless server roles that re-prompt on every boot.
    pub fn memory() -> Self {
        Keystore::Memory(Mutex::new(HashMap::new()))
    }

    /// Best-effort platform constructor. Falls back to in-memory on platforms
    /// where no system keystore is wired up.
    pub fn platform() -> Self {
        #[cfg(target_os = "linux")]
        {
            return Keystore::Secret(linux::SecretService::new());
        }
        #[cfg(target_os = "macos")]
        {
            return Keystore::Keychain(macos::Keychain::new());
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            Keystore::Memory(Mutex::new(HashMap::new()))
        }
    }

    pub fn put(&self, account: &str, key: &MasterKey) -> Result<(), KeystoreError> {
        match self {
            Keystore::Memory(m) => {
                let mut g = m.lock().expect("poisoned");
                g.insert(account.to_string(), *key.as_bytes());
                Ok(())
            }
            #[cfg(target_os = "linux")]
            Keystore::Secret(s) => s.put(account, key.as_bytes()),
            #[cfg(target_os = "macos")]
            Keystore::Keychain(k) => k.put(account, key.as_bytes()),
        }
    }

    pub fn get(&self, account: &str) -> Result<MasterKey, KeystoreError> {
        match self {
            Keystore::Memory(m) => {
                let g = m.lock().expect("poisoned");
                let bytes = g
                    .get(account)
                    .copied()
                    .ok_or_else(|| KeystoreError::NotFound(account.to_string()))?;
                Ok(MasterKey::from_bytes(bytes))
            }
            #[cfg(target_os = "linux")]
            Keystore::Secret(s) => {
                let mut buf = s.get(account)?;
                if buf.len() != KEY_LEN {
                    buf.zeroize();
                    return Err(KeystoreError::Backend("bad key length".into()));
                }
                let mut bytes = [0u8; KEY_LEN];
                bytes.copy_from_slice(&buf);
                buf.zeroize();
                Ok(MasterKey::from_bytes(bytes))
            }
            #[cfg(target_os = "macos")]
            Keystore::Keychain(k) => {
                let mut buf = k.get(account)?;
                if buf.len() != KEY_LEN {
                    buf.zeroize();
                    return Err(KeystoreError::Backend("bad key length".into()));
                }
                let mut bytes = [0u8; KEY_LEN];
                bytes.copy_from_slice(&buf);
                buf.zeroize();
                Ok(MasterKey::from_bytes(bytes))
            }
        }
    }

    pub fn forget(&self, account: &str) -> Result<(), KeystoreError> {
        match self {
            Keystore::Memory(m) => {
                m.lock().expect("poisoned").remove(account);
                Ok(())
            }
            #[cfg(target_os = "linux")]
            Keystore::Secret(s) => s.forget(account),
            #[cfg(target_os = "macos")]
            Keystore::Keychain(k) => k.forget(account),
        }
    }
}

// ---------------------------------------------------------------------------
// Platform backends
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
mod linux {
    //! libsecret bridge via the `secret-service` D-Bus protocol. The actual
    //! D-Bus client lives in a separate library not pulled into this crate by
    //! default to keep the cross-platform build slim. On a Linux host, agentd
    //! brings the `secret-service` crate in via its own dependency and wires
    //! the methods here through a feature flag.
    //!
    //! This stub returns [`KeystoreError::Unavailable`] until the feature is
    //! enabled — wiring it up requires running on Linux and isn't useful for
    //! cross-compilation checks. See `kiki-agent` build profile.

    use super::KeystoreError;

    pub struct SecretService;

    impl SecretService {
        pub fn new() -> Self { Self }
        pub fn put(&self, _account: &str, _bytes: &[u8]) -> Result<(), KeystoreError> {
            Err(KeystoreError::Unavailable)
        }
        pub fn get(&self, _account: &str) -> Result<Vec<u8>, KeystoreError> {
            Err(KeystoreError::Unavailable)
        }
        pub fn forget(&self, _account: &str) -> Result<(), KeystoreError> {
            Err(KeystoreError::Unavailable)
        }
    }
}

#[cfg(target_os = "macos")]
mod macos {
    //! macOS Keychain Services backend via `security-framework`. The master key
    //! is stored as a generic password under service `com.kiki-os.config`,
    //! account `<user_id>`. Keychain ACLs gate access to the login session, so
    //! a subsequent boot re-reads without a Kiki password prompt.

    use super::KeystoreError;
    use security_framework::passwords::{
        delete_generic_password, get_generic_password, set_generic_password,
    };

    const SERVICE: &str = "com.kiki-os.config";

    pub struct Keychain;

    impl Keychain {
        pub fn new() -> Self { Self }

        pub fn put(&self, account: &str, bytes: &[u8]) -> Result<(), KeystoreError> {
            set_generic_password(SERVICE, account, bytes)
                .map_err(|e| KeystoreError::Backend(e.to_string()))
        }

        pub fn get(&self, account: &str) -> Result<Vec<u8>, KeystoreError> {
            match get_generic_password(SERVICE, account) {
                Ok(bytes) => Ok(bytes),
                // errSecItemNotFound (-25300) → NotFound; anything else → Backend.
                Err(e) if e.code() == -25300 => Err(KeystoreError::NotFound(account.to_string())),
                Err(e) => Err(KeystoreError::Backend(e.to_string())),
            }
        }

        pub fn forget(&self, account: &str) -> Result<(), KeystoreError> {
            match delete_generic_password(SERVICE, account) {
                Ok(()) => Ok(()),
                Err(e) if e.code() == -25300 => Ok(()), // already gone
                Err(e) => Err(KeystoreError::Backend(e.to_string())),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::{derive_master_key, KdfParams};
    use base64::Engine as _;

    fn params() -> KdfParams {
        KdfParams {
            salt_b64:    base64::engine::general_purpose::STANDARD.encode(b"0123456789ABCDEF"),
            m_cost_kib:  8,
            t_cost:      1,
            parallelism: 1,
        }
    }

    #[test]
    fn memory_roundtrip() {
        let ks = Keystore::memory();
        let k = derive_master_key("pw", &params()).unwrap();
        ks.put("user-1", &k).unwrap();
        let k2 = ks.get("user-1").unwrap();
        assert_eq!(k.as_bytes(), k2.as_bytes());
        ks.forget("user-1").unwrap();
        assert!(ks.get("user-1").is_err());
    }

    /// Exercises the real platform keystore (macOS Keychain / Linux secret-service).
    /// Gated behind `KIKI_KEYSTORE_TEST=1` because it touches the OS keychain
    /// (may require an unlocked login keychain and isn't appropriate for CI).
    #[test]
    fn platform_roundtrip() {
        if std::env::var("KIKI_KEYSTORE_TEST").as_deref() != Ok("1") {
            eprintln!("skipped: set KIKI_KEYSTORE_TEST=1 to exercise the OS keychain");
            return;
        }
        let ks = Keystore::platform();
        let account = format!("kiki-test-{}", std::process::id());
        let k = derive_master_key("pw", &params()).unwrap();

        ks.forget(&account).ok(); // clean slate
        ks.put(&account, &k).expect("put into OS keychain");
        let got = ks.get(&account).expect("get from OS keychain");
        assert_eq!(k.as_bytes(), got.as_bytes(), "keychain must round-trip the master key bytes");

        // NotFound after forget.
        ks.forget(&account).expect("forget");
        match ks.get(&account) {
            Err(KeystoreError::NotFound(_)) => {}
            other => panic!("expected NotFound after forget, got {other:?}"),
        }
    }
}
