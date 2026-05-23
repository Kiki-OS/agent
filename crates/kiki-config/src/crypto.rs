//! Client-side crypto for `Scope::Secrets`.
//!
//! ## Construction
//!
//! - **KDF:** Argon2id (RFC 9106). Parameters stored alongside each user
//!   (salt + m_cost + t_cost + parallelism). Default tuning: 64 MiB, t=3,
//!   p=4 — strong on a modern device; tune down for sensor-class hardware
//!   via [`KdfParams::sensor_class`] when on RAM-constrained nodes.
//! - **AEAD:** XChaCha20-Poly1305 (RFC 8439 stream + RFC 7539 Poly1305).
//!   24-byte nonces let us safely use random nonces per encrypt without a
//!   reuse risk over realistic operational lifetimes.
//! - **Master key:** 32 bytes. Held in a [`MasterKey`] wrapper that wipes on
//!   drop via `zeroize`. Never serialized or logged.
//!
//! ## Domain separation
//!
//! Each `Sealed` blob's AAD is `b"kiki-config/v1|" || scope || "|" || key_id`,
//! preventing a ciphertext for one secret from being decrypted as another.
//!
//! ## Roundtrip property (see `tests::roundtrip`)
//!
//! For any `(password, salt, plaintext)`, `open_secret(seal_secret(p, k)) == p`.

use std::fmt;

use argon2::{Algorithm, Argon2, Params, Version};
use chacha20poly1305::aead::{Aead, KeyInit, Payload};
use chacha20poly1305::{XChaCha20Poly1305, XNonce};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use zeroize::ZeroizeOnDrop;
#[cfg(test)]
use zeroize::Zeroize;

pub const KEY_LEN: usize = 32;
pub const SALT_LEN: usize = 16;
pub const NONCE_LEN: usize = 24;
pub const AAD_PREFIX: &[u8] = b"kiki-config/v1";

/// Argon2id parameters serialized alongside the user's encrypted blob
/// metadata. Stored server-side (the server holds these, the password it
/// derives is never sent).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KdfParams {
    /// Random 16-byte salt, base64-encoded for transport.
    pub salt_b64:    String,
    /// Argon2id memory cost in KiB.
    pub m_cost_kib:  u32,
    /// Argon2id time cost (iterations).
    pub t_cost:      u32,
    /// Argon2id parallelism (lanes).
    pub parallelism: u32,
}

impl KdfParams {
    /// Default tuning for desktop / mobile class. ~250ms on M2-class hardware.
    pub fn default_desktop() -> Self {
        let mut salt = [0u8; SALT_LEN];
        rand::thread_rng().fill_bytes(&mut salt);
        Self {
            salt_b64:    b64(&salt),
            m_cost_kib:  64 * 1024,
            t_cost:      3,
            parallelism: 4,
        }
    }

    /// Tuning for sensor-class devices (≤512 MiB RAM). Trades cost for
    /// feasibility; still well above the minimum recommended in RFC 9106.
    pub fn sensor_class() -> Self {
        let mut salt = [0u8; SALT_LEN];
        rand::thread_rng().fill_bytes(&mut salt);
        Self {
            salt_b64:    b64(&salt),
            m_cost_kib:  19 * 1024,
            t_cost:      2,
            parallelism: 1,
        }
    }

    fn salt(&self) -> Result<Vec<u8>, CryptoError> {
        unb64(&self.salt_b64)
    }
}

/// 32-byte master key derived from a password. Zeroized on drop; never logged.
#[derive(ZeroizeOnDrop)]
pub struct MasterKey {
    bytes: [u8; KEY_LEN],
}

impl MasterKey {
    pub fn from_bytes(bytes: [u8; KEY_LEN]) -> Self {
        Self { bytes }
    }

    pub fn as_bytes(&self) -> &[u8; KEY_LEN] {
        &self.bytes
    }
}

impl fmt::Debug for MasterKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("MasterKey(<redacted>)")
    }
}

/// Sealed (ciphertext + nonce) blob suitable for transport/storage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sealed {
    pub ciphertext_b64: String,
    pub nonce_b64:      String,
}

/// Derive a master key from a UTF-8 password + KDF params. Synchronous and
/// CPU-bound; callers in async contexts should wrap in [`tokio::task::spawn_blocking`].
pub fn derive_master_key(password: &str, params: &KdfParams) -> Result<MasterKey, CryptoError> {
    let salt = params.salt()?;
    let argon_params = Params::new(params.m_cost_kib, params.t_cost, params.parallelism, Some(KEY_LEN))
        .map_err(|e| CryptoError::Kdf(e.to_string()))?;
    let argon = Argon2::new(Algorithm::Argon2id, Version::V0x13, argon_params);
    let mut out = [0u8; KEY_LEN];
    argon
        .hash_password_into(password.as_bytes(), &salt, &mut out)
        .map_err(|e| CryptoError::Kdf(e.to_string()))?;
    Ok(MasterKey::from_bytes(out))
}

/// Encrypt `plaintext` under `key` with random nonce. The `scope` and `key_id`
/// are bound into the AAD so a swap across keys/scopes is rejected.
pub fn seal_secret(
    key: &MasterKey,
    scope: &str,
    key_id: &str,
    plaintext: &[u8],
) -> Result<Sealed, CryptoError> {
    let cipher = XChaCha20Poly1305::new(key.as_bytes().into());
    let mut nonce_bytes = [0u8; NONCE_LEN];
    rand::thread_rng().fill_bytes(&mut nonce_bytes);
    let nonce = XNonce::from_slice(&nonce_bytes);
    let aad = build_aad(scope, key_id);
    let ct = cipher
        .encrypt(nonce, Payload { msg: plaintext, aad: &aad })
        .map_err(|_| CryptoError::Aead("encryption failed".into()))?;
    Ok(Sealed {
        ciphertext_b64: b64(&ct),
        nonce_b64:      b64(&nonce_bytes),
    })
}

/// Decrypt a sealed blob and return plaintext. Returns `Err` on tag failure
/// (wrong key, tampered ciphertext, or wrong scope/key_id binding).
pub fn open_secret(
    key: &MasterKey,
    scope: &str,
    key_id: &str,
    sealed: &Sealed,
) -> Result<Vec<u8>, CryptoError> {
    let cipher = XChaCha20Poly1305::new(key.as_bytes().into());
    let nonce_bytes = unb64(&sealed.nonce_b64)?;
    if nonce_bytes.len() != NONCE_LEN {
        return Err(CryptoError::BadNonce);
    }
    let nonce = XNonce::from_slice(&nonce_bytes);
    let ct = unb64(&sealed.ciphertext_b64)?;
    let aad = build_aad(scope, key_id);
    cipher
        .decrypt(nonce, Payload { msg: &ct, aad: &aad })
        .map_err(|_| CryptoError::Aead("decryption failed (wrong key, tampered, or wrong scope/key_id)".into()))
}

fn build_aad(scope: &str, key_id: &str) -> Vec<u8> {
    let mut v = Vec::with_capacity(AAD_PREFIX.len() + scope.len() + key_id.len() + 2);
    v.extend_from_slice(AAD_PREFIX);
    v.push(b'|');
    v.extend_from_slice(scope.as_bytes());
    v.push(b'|');
    v.extend_from_slice(key_id.as_bytes());
    v
}

fn b64(bytes: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn unb64(s: &str) -> Result<Vec<u8>, CryptoError> {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD
        .decode(s.as_bytes())
        .map_err(|e| CryptoError::Base64(e.to_string()))
}

/// Constant-time equality for [`MasterKey`]. Use this for "are these two
/// derivations equal" rather than `==`.
pub fn keys_equal(a: &MasterKey, b: &MasterKey) -> bool {
    use subtle::ConstantTimeEq;
    a.as_bytes().ct_eq(b.as_bytes()).into()
}

#[derive(Debug, Error)]
pub enum CryptoError {
    #[error("kdf error: {0}")]
    Kdf(String),
    #[error("aead error: {0}")]
    Aead(String),
    #[error("nonce wrong length")]
    BadNonce,
    #[error("base64 decode error: {0}")]
    Base64(String),
}

// Manually impl Zeroize for the bytes field, since the derive macro on the
// wrapper struct already does the right thing via ZeroizeOnDrop. We expose
// `wipe` for tests that want to assert the bytes are gone.
impl MasterKey {
    #[cfg(test)]
    fn wipe(&mut self) {
        self.bytes.zeroize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_params() -> KdfParams {
        // Tiny params just for fast tests; production uses default_desktop().
        KdfParams {
            salt_b64:    b64(b"0123456789ABCDEF"),
            m_cost_kib:  8,
            t_cost:      1,
            parallelism: 1,
        }
    }

    #[test]
    fn roundtrip() {
        let key = derive_master_key("correct horse battery staple", &test_params()).unwrap();
        let pt = b"the password is hunter2";
        let sealed = seal_secret(&key, "secrets", "openai.api_key", pt).unwrap();
        let out = open_secret(&key, "secrets", "openai.api_key", &sealed).unwrap();
        assert_eq!(out, pt);
    }

    #[test]
    fn wrong_key_fails() {
        let k1 = derive_master_key("alpha", &test_params()).unwrap();
        let k2 = derive_master_key("beta", &test_params()).unwrap();
        let sealed = seal_secret(&k1, "secrets", "key", b"data").unwrap();
        assert!(open_secret(&k2, "secrets", "key", &sealed).is_err());
    }

    #[test]
    fn tampered_ciphertext_fails() {
        let k = derive_master_key("alpha", &test_params()).unwrap();
        let mut sealed = seal_secret(&k, "secrets", "key", b"data").unwrap();
        // flip a bit in ciphertext
        let mut ct = unb64(&sealed.ciphertext_b64).unwrap();
        ct[0] ^= 0x01;
        sealed.ciphertext_b64 = b64(&ct);
        assert!(open_secret(&k, "secrets", "key", &sealed).is_err());
    }

    #[test]
    fn wrong_scope_fails() {
        let k = derive_master_key("alpha", &test_params()).unwrap();
        let sealed = seal_secret(&k, "secrets", "k", b"data").unwrap();
        assert!(open_secret(&k, "preferences", "k", &sealed).is_err());
    }

    #[test]
    fn wrong_key_id_fails() {
        let k = derive_master_key("alpha", &test_params()).unwrap();
        let sealed = seal_secret(&k, "secrets", "openai.api_key", b"data").unwrap();
        assert!(open_secret(&k, "secrets", "anthropic.api_key", &sealed).is_err());
    }

    #[test]
    fn nonces_are_unique_per_seal() {
        let k = derive_master_key("alpha", &test_params()).unwrap();
        let a = seal_secret(&k, "secrets", "k", b"x").unwrap();
        let b = seal_secret(&k, "secrets", "k", b"x").unwrap();
        assert_ne!(a.nonce_b64, b.nonce_b64);
        assert_ne!(a.ciphertext_b64, b.ciphertext_b64);
    }

    #[test]
    fn keys_equal_constant_time() {
        let k1 = derive_master_key("p", &test_params()).unwrap();
        let k2 = derive_master_key("p", &test_params()).unwrap();
        let k3 = derive_master_key("q", &test_params()).unwrap();
        assert!(keys_equal(&k1, &k2));
        assert!(!keys_equal(&k1, &k3));
    }

    #[test]
    fn master_key_wipes_on_drop() {
        // Sanity: after explicit wipe, bytes are zero. (We don't directly observe
        // post-drop state in safe Rust; this checks the Zeroize impl works.)
        let mut k = derive_master_key("p", &test_params()).unwrap();
        k.wipe();
        assert_eq!(k.as_bytes(), &[0u8; KEY_LEN]);
    }
}
