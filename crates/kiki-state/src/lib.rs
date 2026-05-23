//! OSTree-backed state backend.
//!
//! Every durable agent step creates an OSTree commit. The commit hash becomes
//! the step ID. Migration (local → cloud) = ostree push + ostree pull on the
//! target host. Only the delta transfers (content-addressed).
//!
//! Backends:
//!   OstreeBackend   — production, durable / eternal persistence
//!   MemoryBackend   — ephemeral / session, or for tests
//!   HybridBackend   — local cache + async sync to OSTree remote

pub mod ostree;
pub mod memory;
pub mod hybrid;

pub use ostree::OstreeBackend;
pub use memory::MemoryBackend;
pub use hybrid::HybridBackend;
