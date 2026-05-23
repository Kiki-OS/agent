//! `kiki-config` — hybrid end-to-end config and secrets sync engine.
//!
//! Three tiers per the architecture decision (see memory `project-arch-decisions`):
//!
//! | Tier          | Storage                                      | Crypto                              |
//! |---------------|----------------------------------------------|-------------------------------------|
//! | `Secrets`     | Server holds ciphertext only, never plaintext| Client-side: Argon2id → XChaCha20-Poly1305 |
//! | `Preferences` | D1 (transparent at-rest encryption)          | Plaintext to worker                 |
//! | `Capabilities`| D1 + immutable audit log                     | Plaintext to worker                 |
//!
//! Each scope carries a monotonic `revision`. The client does CAS pushes via
//! the [`SyncClient`]; on conflict it refetches and lets the caller resolve.
//!
//! Bootstrap on a new device:
//! 1. Device flow authenticates the device session token.
//! 2. Client pulls ciphertext blobs for `Secrets`, plaintext for the other two tiers.
//! 3. User supplies the master password (or unlocks via passkey).
//! 4. [`crypto::derive_master_key`] runs Argon2id with the stored `kdf_params`.
//! 5. [`crypto::open_secret`] decrypts each blob locally.
//!
//! Recovery: a BIP39 recovery code generated at setup time can be used to
//! reset the master password if forgotten. Without password OR recovery code,
//! Secrets are unrecoverable by design.

pub mod crypto;
pub mod keystore;
pub mod recovery;
pub mod scope;
pub mod sync;

pub use crypto::{
    derive_master_key, open_secret, seal_secret, KdfParams, MasterKey, Sealed, CryptoError,
};
pub use recovery::{RecoveryCode, generate_recovery_code, recover_master_key};
pub use scope::{Revision, Scope, ScopedValue};
pub use sync::{ConfigEndpoint, SyncClient, SyncError};
