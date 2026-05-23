//! kiki-vault — client for the Kiki shared blob store.
//!
//! ## Surface
//!
//! Three scopes, each addressable by URI:
//!
//! - `vault://personal/{user_id}/...`     — user's own cross-device store
//! - `vault://fleet/{org_id}/...`         — org-shared (admins write, members read)
//! - `vault://share/{share_id}/...`       — point-to-point share between users
//!
//! ## Consistency
//!
//! Every `put` is CAS: caller passes `if_match: Revision`. Conflicts are
//! surfaced — clients merge in-app and retry. Blobs are content-addressed by
//! sha256; the same content stored at two paths shares one underlying object.
//!
//! ## Caching
//!
//! [`LocalCache`] is an LRU on disk (default `/var/kiki/vault/cache`, capped
//! at 1 GiB). Pinned paths bypass eviction. The cache stores the *blob*
//! (sha256-named) plus a path → sha256 index for fast resolve.
//!
//! ## Watch
//!
//! [`VaultClient::watch`] opens a WebSocket subscription to one or more path
//! globs and emits [`VaultEvent`]s when revisions change. The server is the
//! [DurableObject coordinator](https://developers.cloudflare.com/durable-objects/)
//! per scope.
//!
//! ## MCP tools
//!
//! [`mcp_tools`] exposes the surface as builtin MCP tools (`vault_read`,
//! `vault_write`, `vault_list`, `vault_head`, `vault_watch`) for the agent.
//! Each call is gated by [`kiki_schema::VaultCapabilities`] declared in
//! the calling artifact's `kiki.toml`.

pub mod cache;
pub mod client;
pub mod mcp_tools;
pub mod uri;

pub use cache::{CacheError, LocalCache};
pub use client::{
    Acl, BlobMeta, CapabilityToken, ListEntry, Revision, VaultClient, VaultError, VaultEvent,
};
pub use uri::{VaultScope, VaultUri};
