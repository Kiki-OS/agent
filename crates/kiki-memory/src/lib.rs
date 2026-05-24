//! kiki-memory — the OS-level memory subsystem (the library `memoryd` serves).
//!
//! Memory in Kiki is OS infrastructure, not an agent feature: it persists across
//! sessions, reboots, and devices. Four layers (slow → fast):
//!
//! - **Identity**  — who the user is ([`UserProfile`]); slowest, one document.
//! - **Semantic**  — what the system knows ([`SemanticFact`]s).
//! - **Procedural**— how to do things well here ([`ProceduralEntry`], incl. the
//!                   user's corrections); the only layer the agent writes directly.
//! - **Episodic**  — what happened and when ([`EpisodeEvent`]); fast, high volume.
//!
//! This crate is the pure storage + query core ([`MemoryStore`]) plus the wire
//! protocol ([`MemoryQuery`] / [`MemoryWrite`] / [`MemoryResult`]) that `memoryd`
//! exposes over `/run/kiki/memory.sock`. It is on-device only — nothing here
//! talks to the network. Search is lexical for now (a vector index over the
//! semantic layer is future work).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

pub mod store;
pub use store::MemoryStore;

#[cfg(feature = "client")]
pub mod client;
#[cfg(feature = "client")]
pub use client::MemoryClient;

/// The four memory layers. `snake_case` on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryLayer {
    Episodic,
    Semantic,
    Procedural,
    Identity,
}

impl MemoryLayer {
    /// All layers, for queries that don't restrict (an empty `layers` vec means
    /// "all" — this is the canonical expansion).
    pub fn all() -> Vec<MemoryLayer> {
        vec![
            MemoryLayer::Episodic,
            MemoryLayer::Semantic,
            MemoryLayer::Procedural,
            MemoryLayer::Identity,
        ]
    }
}

/// A significant system event (session done, action, error + resolution, …).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpisodeEvent {
    pub id:         String,
    /// Coarse type: "session_done", "tool_error", "user_pref", …
    pub kind:       String,
    #[serde(default)]
    pub session_id: String,
    pub summary:    String,
    #[serde(default)]
    pub outcome:    String,
    pub ts_ms:      u64,
    /// Milestones flagged important are exempt from retention expiry.
    #[serde(default)]
    pub important:  bool,
}

/// A learned "how to do X here" recipe/heuristic, or a user correction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProceduralEntry {
    pub key:        String,
    pub content:    String,
    /// 0.0–1.0. Corrections are written at 1.0; failed approaches get lowered.
    pub confidence: f32,
    pub updated_ms: u64,
    /// True when this entry came from an explicit user correction (high priority
    /// for context injection).
    #[serde(default)]
    pub correction: bool,
}

/// A general fact the system knows about the user or the world.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticFact {
    pub id:         String,
    pub topic:      String,
    pub content:    String,
    pub updated_ms: u64,
}

/// The user model. Mirrors `identity/profile.json` in the spec.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UserProfile {
    #[serde(default)]
    pub display_name: String,
    #[serde(default = "default_language")]
    pub language:     String,
    #[serde(default)]
    pub timezone:     String,
    #[serde(default)]
    pub expertise:    Vec<String>,
    #[serde(default)]
    pub preferences:  BTreeMap<String, String>,
    #[serde(default)]
    pub privacy:      Privacy,
    /// When the profile was last updated (ms). The agent suggests a review when
    /// this is stale (>30 days).
    #[serde(default)]
    pub updated_ms:   u64,
}

fn default_language() -> String { "en".to_string() }

impl Default for UserProfile {
    fn default() -> Self {
        Self {
            display_name: String::new(),
            language:     default_language(),
            timezone:     String::new(),
            expertise:    Vec::new(),
            preferences:  BTreeMap::new(),
            privacy:      Privacy::default(),
            updated_ms:   0,
        }
    }
}

/// Privacy defaults are the safe ones: nothing leaves the device, no transcripts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Privacy {
    #[serde(default)]
    pub sync_to_cloud:          bool,
    #[serde(default)]
    pub store_voice_transcripts: bool,
    #[serde(default)]
    pub log_screen_content:     bool,
    #[serde(default = "default_retention_days")]
    pub memory_retention_days:  u32,
}

fn default_retention_days() -> u32 { 90 }

impl Default for Privacy {
    fn default() -> Self {
        Self {
            sync_to_cloud:           false,
            store_voice_transcripts: false,
            log_screen_content:      false,
            memory_retention_days:   default_retention_days(),
        }
    }
}

// ── Wire protocol (agentd ↔ memoryd over /run/kiki/memory.sock) ────────────────

/// A read query from agentd to memoryd.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum MemoryQuery {
    /// Lexical search across the given layers (empty = all), ranked, capped.
    Search { query: String, #[serde(default)] layers: Vec<MemoryLayer>, limit: usize },
    /// Everything in the given layers updated at/after `since_ms`.
    Recent { since_ms: u64, #[serde(default)] layers: Vec<MemoryLayer> },
    /// The user profile (identity layer).
    UserProfile,
    /// The most recent user corrections (high-priority procedural entries).
    Corrections { limit: usize },
}

/// A write from agentd to memoryd.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum MemoryWrite {
    Episode { event: EpisodeEvent },
    Procedural { key: String, content: String, confidence: f32 },
    UserCorrection { correction: String, context: String, ts_ms: u64 },
    Semantic { id: String, topic: String, content: String, ts_ms: u64 },
    /// Replace the user profile (identity layer).
    Profile { profile: UserProfile },
}

/// A single search/recent hit, layer-tagged with a relevance score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryHit {
    pub layer:   MemoryLayer,
    pub id:      String,
    pub score:   f32,
    pub content: String,
    pub ts_ms:   u64,
}

/// A single request frame on the memoryd socket — either a read or a write.
/// Externally tagged so the inner `op` enums stay unambiguous:
/// `{"query":{"op":"search",…}}` or `{"write":{"op":"episode",…}}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryRequest {
    Query(MemoryQuery),
    Write(MemoryWrite),
}

/// The reply memoryd returns for a [`MemoryQuery`] or [`MemoryWrite`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MemoryResult {
    Hits { hits: Vec<MemoryHit> },
    Profile { profile: UserProfile },
    Ok,
    Error { message: String },
}

/// A full export of every layer — for `kpkg memory export/import` and
/// device-to-device migration. Self-contained JSON.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MemorySnapshot {
    pub profile:    UserProfile,
    pub facts:      Vec<SemanticFact>,
    pub procedural: Vec<ProceduralEntry>,
    pub episodes:   Vec<EpisodeEvent>,
}

#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("io: {0}")]
    Io(String),
    #[error("serialize: {0}")]
    Serde(String),
}

pub(crate) type Result<T> = std::result::Result<T, MemoryError>;

/// Day bucket for episodic JSONL files — `ts_ms / 86_400_000`. Avoids a date
/// dependency while keeping per-day files (and deterministic tests).
pub(crate) fn day_index(ts_ms: u64) -> u64 {
    ts_ms / 86_400_000
}

pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| MemoryError::Io(e.to_string()))?;
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes).map_err(|e| MemoryError::Io(e.to_string()))?;
    std::fs::rename(&tmp, path).map_err(|e| MemoryError::Io(e.to_string()))
}

pub(crate) fn read_json<T: for<'de> Deserialize<'de> + Default>(path: &PathBuf) -> Result<T> {
    match std::fs::read(path) {
        Ok(b) => serde_json::from_slice(&b).map_err(|e| MemoryError::Serde(e.to_string())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(T::default()),
        Err(e) => Err(MemoryError::Io(e.to_string())),
    }
}
