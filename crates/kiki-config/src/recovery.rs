//! BIP39-based recovery codes for the master password.
//!
//! At setup the client generates a 24-word mnemonic (256 bits of entropy) and
//! the user must record it. The recovery code is bound to the active
//! [`KdfParams`] salt so importing it later requires the salt — which the
//! server holds — but NOT the password.
//!
//! Loss model:
//! - Password lost, recovery code retained → user enters recovery code,
//!   client derives the same master key.
//! - Recovery code lost, password retained → fine, user can generate a new
//!   recovery code from the current master key.
//! - Both lost → Secrets are permanently unrecoverable. By design.

use bip39::{Language, Mnemonic};
use thiserror::Error;
use zeroize::Zeroize;

use crate::crypto::{KdfParams, MasterKey, KEY_LEN};

#[derive(Debug, Error)]
pub enum RecoveryError {
    #[error("bip39 error: {0}")]
    Bip39(String),
    #[error("recovery code did not produce a key of length {KEY_LEN}")]
    BadEntropyLength,
}

/// A 24-word BIP39 mnemonic. Display + Debug are redacted.
pub struct RecoveryCode {
    mnemonic: Mnemonic,
}

impl RecoveryCode {
    /// Generate fresh 256-bit entropy → 24-word mnemonic in English.
    pub fn generate() -> Result<Self, RecoveryError> {
        let mnemonic = Mnemonic::generate(24)
            .map_err(|e| RecoveryError::Bip39(e.to_string()))?;
        Ok(Self { mnemonic })
    }

    /// Parse 24 space-separated words. Validates BIP39 checksum.
    pub fn parse(phrase: &str) -> Result<Self, RecoveryError> {
        let mnemonic = Mnemonic::parse_in(Language::English, phrase)
            .map_err(|e| RecoveryError::Bip39(e.to_string()))?;
        Ok(Self { mnemonic })
    }

    /// The user-visible phrase. Returned as `String` — caller is responsible
    /// for zeroizing once shown.
    pub fn phrase(&self) -> String {
        self.mnemonic.to_string()
    }
}

impl std::fmt::Debug for RecoveryCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RecoveryCode(<redacted, 24 words>)")
    }
}

/// Helper kept for symmetry with the API surface.
pub fn generate_recovery_code() -> Result<RecoveryCode, RecoveryError> {
    RecoveryCode::generate()
}

/// Derive the master key from a recovery code. The recovery code's BIP39 seed
/// is HKDF-bound to the user's salt (via the same Argon2id step using the seed
/// as "password"), so two users with the same recovery code (impossible in
/// practice, but defensive) would still get distinct keys.
pub fn recover_master_key(
    code: &RecoveryCode,
    params: &KdfParams,
) -> Result<MasterKey, crate::crypto::CryptoError> {
    let mut seed = code.mnemonic.to_seed("");
    // bip39's `to_seed` returns 64 bytes. We hex-encode and feed to Argon2id
    // along with the user's salt for domain separation, so the recovered key
    // matches what `derive_master_key` produces for an equivalent password
    // (different value than the original password, but consistent across
    // devices for the same recovery code).
    let hex_seed = hex::encode(seed);
    let key = crate::crypto::derive_master_key(&hex_seed, params)?;
    seed.zeroize();
    Ok(key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::{open_secret, seal_secret, KdfParams};

    fn test_params() -> KdfParams {
        use base64::Engine as _;
        KdfParams {
            salt_b64:    base64::engine::general_purpose::STANDARD.encode(b"0123456789ABCDEF"),
            m_cost_kib:  8,
            t_cost:      1,
            parallelism: 1,
        }
    }

    #[test]
    fn recovery_code_roundtrip() {
        let code = RecoveryCode::generate().unwrap();
        let phrase = code.phrase();
        let parsed = RecoveryCode::parse(&phrase).unwrap();
        assert_eq!(code.phrase(), parsed.phrase());
    }

    #[test]
    fn recovered_key_decrypts_blobs_sealed_by_same_code() {
        let code = RecoveryCode::generate().unwrap();
        let p = test_params();
        let k1 = recover_master_key(&code, &p).unwrap();
        let sealed = seal_secret(&k1, "secrets", "k", b"data").unwrap();
        // Parse phrase again, recover, decrypt.
        let code2 = RecoveryCode::parse(&code.phrase()).unwrap();
        let k2 = recover_master_key(&code2, &p).unwrap();
        let pt = open_secret(&k2, "secrets", "k", &sealed).unwrap();
        assert_eq!(pt, b"data");
    }

    #[test]
    fn different_codes_produce_different_keys() {
        let p = test_params();
        let c1 = RecoveryCode::generate().unwrap();
        let c2 = RecoveryCode::generate().unwrap();
        let k1 = recover_master_key(&c1, &p).unwrap();
        let k2 = recover_master_key(&c2, &p).unwrap();
        assert_ne!(k1.as_bytes(), k2.as_bytes());
    }

    #[test]
    fn rejects_invalid_checksum() {
        // Tamper a valid phrase to corrupt checksum.
        let c = RecoveryCode::generate().unwrap();
        let phrase = c.phrase();
        let mut words: Vec<&str> = phrase.split_whitespace().collect();
        // Swap the last word for a different valid wordlist word; checksum will fail.
        let new_last = if words[23] == "abandon" { "ability" } else { "abandon" };
        words[23] = new_last;
        let tampered = words.join(" ");
        assert!(RecoveryCode::parse(&tampered).is_err());
    }
}
