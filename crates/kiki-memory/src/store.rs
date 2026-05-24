//! [`MemoryStore`] — the durable four-layer backend behind `memoryd`.
//!
//! On-disk layout (mirrors `spec/MEMORY.md`):
//! ```text
//! <root>/
//! ├── episodic/day-<n>.jsonl   one append-only line per event, bucketed by day
//! ├── semantic/facts.json      { id -> SemanticFact }
//! ├── procedural/entries.json  { key -> ProceduralEntry }  (incl. corrections)
//! └── identity/profile.json    UserProfile
//! ```
//! All non-append writes are atomic (tmp + rename). Search is lexical: a hit
//! scores by how many query terms appear in the entry's text.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;

use crate::{
    day_index, read_json, write_atomic, EpisodeEvent, MemoryError, MemoryHit, MemoryLayer,
    MemoryQuery, MemoryResult, MemoryWrite, ProceduralEntry, Result, SemanticFact, UserProfile,
};

pub struct MemoryStore {
    root: PathBuf,
}

impl MemoryStore {
    /// Open (creating the layout if absent) a store rooted at `dir`
    /// (e.g. `/var/kiki/memory`).
    pub fn open(dir: impl Into<PathBuf>) -> Result<Self> {
        let root = dir.into();
        for sub in ["episodic", "semantic", "procedural", "identity"] {
            std::fs::create_dir_all(root.join(sub)).map_err(|e| MemoryError::Io(e.to_string()))?;
        }
        Ok(Self { root })
    }

    // ── paths ──────────────────────────────────────────────────────────────────
    fn episodic_day(&self, ts_ms: u64) -> PathBuf {
        self.root.join("episodic").join(format!("day-{}.jsonl", day_index(ts_ms)))
    }
    fn facts_path(&self) -> PathBuf { self.root.join("semantic").join("facts.json") }
    fn procedural_path(&self) -> PathBuf { self.root.join("procedural").join("entries.json") }
    fn profile_path(&self) -> PathBuf { self.root.join("identity").join("profile.json") }

    // ── dispatch ─────────────────────────────────────────────────────────────────

    /// Apply a write. Returns [`MemoryResult::Ok`] or an `Error`.
    pub fn write(&self, w: MemoryWrite) -> MemoryResult {
        let r = match w {
            MemoryWrite::Episode { event } => self.append_episode(&event),
            MemoryWrite::Procedural { key, content, confidence } => {
                self.put_procedural(&key, &content, confidence, false, now_from(&content))
            }
            MemoryWrite::UserCorrection { correction, context, ts_ms } => {
                let key = format!("correction:{ts_ms}");
                let content = if context.is_empty() {
                    correction
                } else {
                    format!("{correction}\n(context: {context})")
                };
                self.put_procedural(&key, &content, 1.0, true, ts_ms)
            }
            MemoryWrite::Semantic { id, topic, content, ts_ms } => {
                self.put_fact(&id, &topic, &content, ts_ms)
            }
            MemoryWrite::Profile { profile } => self.put_profile(&profile),
        };
        match r {
            Ok(()) => MemoryResult::Ok,
            Err(e) => MemoryResult::Error { message: e.to_string() },
        }
    }

    /// Answer a query.
    pub fn query(&self, q: MemoryQuery) -> MemoryResult {
        match q {
            MemoryQuery::UserProfile => match self.load_profile() {
                Ok(profile) => MemoryResult::Profile { profile },
                Err(e) => MemoryResult::Error { message: e.to_string() },
            },
            MemoryQuery::Search { query, layers, limit } => match self.search(&query, &layers, limit) {
                Ok(hits) => MemoryResult::Hits { hits },
                Err(e) => MemoryResult::Error { message: e.to_string() },
            },
            MemoryQuery::Recent { since_ms, layers } => match self.recent(since_ms, &layers) {
                Ok(hits) => MemoryResult::Hits { hits },
                Err(e) => MemoryResult::Error { message: e.to_string() },
            },
            MemoryQuery::Corrections { limit } => match self.corrections(limit) {
                Ok(hits) => MemoryResult::Hits { hits },
                Err(e) => MemoryResult::Error { message: e.to_string() },
            },
        }
    }

    // ── writes ─────────────────────────────────────────────────────────────────

    pub fn append_episode(&self, event: &EpisodeEvent) -> Result<()> {
        let path = self.episodic_day(event.ts_ms);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| MemoryError::Io(e.to_string()))?;
        }
        let mut line = serde_json::to_string(event).map_err(|e| MemoryError::Serde(e.to_string()))?;
        line.push('\n');
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| MemoryError::Io(e.to_string()))?;
        f.write_all(line.as_bytes()).map_err(|e| MemoryError::Io(e.to_string()))
    }

    pub fn put_procedural(
        &self,
        key: &str,
        content: &str,
        confidence: f32,
        correction: bool,
        updated_ms: u64,
    ) -> Result<()> {
        let mut map: BTreeMap<String, ProceduralEntry> = read_json(&self.procedural_path())?;
        map.insert(
            key.to_string(),
            ProceduralEntry {
                key: key.to_string(),
                content: content.to_string(),
                confidence: confidence.clamp(0.0, 1.0),
                updated_ms,
                correction,
            },
        );
        let bytes = serde_json::to_vec_pretty(&map).map_err(|e| MemoryError::Serde(e.to_string()))?;
        write_atomic(&self.procedural_path(), &bytes)
    }

    pub fn put_fact(&self, id: &str, topic: &str, content: &str, ts_ms: u64) -> Result<()> {
        let mut map: BTreeMap<String, SemanticFact> = read_json(&self.facts_path())?;
        map.insert(
            id.to_string(),
            SemanticFact { id: id.to_string(), topic: topic.to_string(), content: content.to_string(), updated_ms: ts_ms },
        );
        let bytes = serde_json::to_vec_pretty(&map).map_err(|e| MemoryError::Serde(e.to_string()))?;
        write_atomic(&self.facts_path(), &bytes)
    }

    pub fn put_profile(&self, profile: &UserProfile) -> Result<()> {
        let bytes = serde_json::to_vec_pretty(profile).map_err(|e| MemoryError::Serde(e.to_string()))?;
        write_atomic(&self.profile_path(), &bytes)
    }

    // ── reads ──────────────────────────────────────────────────────────────────

    pub fn load_profile(&self) -> Result<UserProfile> {
        read_json(&self.profile_path())
    }

    fn load_procedural(&self) -> Result<Vec<ProceduralEntry>> {
        let map: BTreeMap<String, ProceduralEntry> = read_json(&self.procedural_path())?;
        Ok(map.into_values().collect())
    }

    fn load_facts(&self) -> Result<Vec<SemanticFact>> {
        let map: BTreeMap<String, SemanticFact> = read_json(&self.facts_path())?;
        Ok(map.into_values().collect())
    }

    /// All episodes across all day files (used by search/recent; high-volume
    /// callers should prefer `recent` with a `since_ms` bound).
    fn load_episodes(&self) -> Result<Vec<EpisodeEvent>> {
        let dir = self.root.join("episodic");
        let mut out = Vec::new();
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
            Err(e) => return Err(MemoryError::Io(e.to_string())),
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("jsonl") {
                continue;
            }
            let content = std::fs::read_to_string(&path).map_err(|e| MemoryError::Io(e.to_string()))?;
            for line in content.lines().filter(|l| !l.trim().is_empty()) {
                if let Ok(ev) = serde_json::from_str::<EpisodeEvent>(line) {
                    out.push(ev);
                }
            }
        }
        Ok(out)
    }

    /// Lexical search. Score = number of query terms found in the entry text
    /// (case-insensitive), normalized to 0.0–1.0. Entries with score 0 dropped.
    fn search(&self, query: &str, layers: &[MemoryLayer], limit: usize) -> Result<Vec<MemoryHit>> {
        let terms: Vec<String> = query.to_lowercase().split_whitespace().map(str::to_string).collect();
        if terms.is_empty() {
            return Ok(Vec::new());
        }
        let layers = if layers.is_empty() { MemoryLayer::all() } else { layers.to_vec() };
        let mut hits = Vec::new();
        let score = |text: &str| -> f32 {
            let t = text.to_lowercase();
            let matched = terms.iter().filter(|term| t.contains(term.as_str())).count();
            matched as f32 / terms.len() as f32
        };

        if layers.contains(&MemoryLayer::Procedural) {
            for e in self.load_procedural()? {
                // Matched corrections get a relevance boost — they're high
                // priority. (Only boost actual matches, never a zero-score entry.)
                let mut s = score(&e.content);
                if s > 0.0 && e.correction { s = (s + 0.25).min(1.0); }
                if s > 0.0 {
                    hits.push(MemoryHit { layer: MemoryLayer::Procedural, id: e.key, score: s, content: e.content, ts_ms: e.updated_ms });
                }
            }
        }
        if layers.contains(&MemoryLayer::Semantic) {
            for f in self.load_facts()? {
                let s = score(&format!("{} {}", f.topic, f.content));
                if s > 0.0 {
                    hits.push(MemoryHit { layer: MemoryLayer::Semantic, id: f.id, score: s, content: f.content, ts_ms: f.updated_ms });
                }
            }
        }
        if layers.contains(&MemoryLayer::Episodic) {
            for ev in self.load_episodes()? {
                let s = score(&format!("{} {} {}", ev.kind, ev.summary, ev.outcome));
                if s > 0.0 {
                    hits.push(MemoryHit { layer: MemoryLayer::Episodic, id: ev.id, score: s, content: ev.summary, ts_ms: ev.ts_ms });
                }
            }
        }

        // Highest score first; break ties by most recent.
        hits.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
                .then(b.ts_ms.cmp(&a.ts_ms))
        });
        hits.truncate(limit);
        Ok(hits)
    }

    fn recent(&self, since_ms: u64, layers: &[MemoryLayer]) -> Result<Vec<MemoryHit>> {
        let layers = if layers.is_empty() { MemoryLayer::all() } else { layers.to_vec() };
        let mut hits = Vec::new();
        if layers.contains(&MemoryLayer::Episodic) {
            for ev in self.load_episodes()? {
                if ev.ts_ms >= since_ms {
                    hits.push(MemoryHit { layer: MemoryLayer::Episodic, id: ev.id, score: 1.0, content: ev.summary, ts_ms: ev.ts_ms });
                }
            }
        }
        if layers.contains(&MemoryLayer::Procedural) {
            for e in self.load_procedural()? {
                if e.updated_ms >= since_ms {
                    hits.push(MemoryHit { layer: MemoryLayer::Procedural, id: e.key, score: 1.0, content: e.content, ts_ms: e.updated_ms });
                }
            }
        }
        if layers.contains(&MemoryLayer::Semantic) {
            for f in self.load_facts()? {
                if f.updated_ms >= since_ms {
                    hits.push(MemoryHit { layer: MemoryLayer::Semantic, id: f.id, score: 1.0, content: f.content, ts_ms: f.updated_ms });
                }
            }
        }
        hits.sort_by(|a, b| b.ts_ms.cmp(&a.ts_ms));
        Ok(hits)
    }

    fn corrections(&self, limit: usize) -> Result<Vec<MemoryHit>> {
        let mut hits: Vec<MemoryHit> = self
            .load_procedural()?
            .into_iter()
            .filter(|e| e.correction)
            .map(|e| MemoryHit { layer: MemoryLayer::Procedural, id: e.key, score: e.confidence, content: e.content, ts_ms: e.updated_ms })
            .collect();
        hits.sort_by(|a, b| b.ts_ms.cmp(&a.ts_ms));
        hits.truncate(limit);
        Ok(hits)
    }

    // ── retention ────────────────────────────────────────────────────────────────

    /// Drop episodic events older than `retention_days` (unless `important`).
    /// Rewrites each day file in place; removes day files left empty. Returns the
    /// number of events expired.
    pub fn expire(&self, retention_days: u32, now_ms: u64) -> Result<usize> {
        let cutoff = now_ms.saturating_sub(retention_days as u64 * 86_400_000);
        let dir = self.root.join("episodic");
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(e) => return Err(MemoryError::Io(e.to_string())),
        };
        let mut expired = 0usize;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("jsonl") {
                continue;
            }
            let content = std::fs::read_to_string(&path).map_err(|e| MemoryError::Io(e.to_string()))?;
            let mut kept = String::new();
            for line in content.lines().filter(|l| !l.trim().is_empty()) {
                match serde_json::from_str::<EpisodeEvent>(line) {
                    Ok(ev) if ev.ts_ms < cutoff && !ev.important => { expired += 1; }
                    _ => { kept.push_str(line); kept.push('\n'); }
                }
            }
            if kept.trim().is_empty() {
                let _ = std::fs::remove_file(&path);
            } else {
                write_atomic(&path, kept.as_bytes())?;
            }
        }
        Ok(expired)
    }

    // ── user control (kpkg memory) ───────────────────────────────────────────────

    /// Delete a single entry by id/key across all layers. Returns true if
    /// something was removed.
    pub fn delete(&self, id: &str) -> Result<bool> {
        let mut removed = false;

        // Procedural (keyed) + semantic (by id).
        let mut proc: BTreeMap<String, ProceduralEntry> = read_json(&self.procedural_path())?;
        if proc.remove(id).is_some() {
            let bytes = serde_json::to_vec_pretty(&proc).map_err(|e| MemoryError::Serde(e.to_string()))?;
            write_atomic(&self.procedural_path(), &bytes)?;
            removed = true;
        }
        let mut facts: BTreeMap<String, SemanticFact> = read_json(&self.facts_path())?;
        if facts.remove(id).is_some() {
            let bytes = serde_json::to_vec_pretty(&facts).map_err(|e| MemoryError::Serde(e.to_string()))?;
            write_atomic(&self.facts_path(), &bytes)?;
            removed = true;
        }

        // Episodic: rewrite each day file dropping the matching event.
        let dir = self.root.join("episodic");
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) != Some("jsonl") {
                    continue;
                }
                let content = std::fs::read_to_string(&path).map_err(|e| MemoryError::Io(e.to_string()))?;
                let mut kept = String::new();
                let mut changed = false;
                for line in content.lines().filter(|l| !l.trim().is_empty()) {
                    match serde_json::from_str::<EpisodeEvent>(line) {
                        Ok(ev) if ev.id == id => { changed = true; }
                        _ => { kept.push_str(line); kept.push('\n'); }
                    }
                }
                if changed {
                    removed = true;
                    if kept.trim().is_empty() {
                        let _ = std::fs::remove_file(&path);
                    } else {
                        write_atomic(&path, kept.as_bytes())?;
                    }
                }
            }
        }
        Ok(removed)
    }

    /// Wipe all memory (every layer). The caller is responsible for confirmation.
    pub fn clear(&self) -> Result<()> {
        for sub in ["episodic", "semantic", "procedural", "identity"] {
            let p = self.root.join(sub);
            if p.exists() {
                std::fs::remove_dir_all(&p).map_err(|e| MemoryError::Io(e.to_string()))?;
            }
            std::fs::create_dir_all(&p).map_err(|e| MemoryError::Io(e.to_string()))?;
        }
        Ok(())
    }

    /// Export every layer into a self-contained [`MemorySnapshot`].
    pub fn export_snapshot(&self) -> Result<crate::MemorySnapshot> {
        Ok(crate::MemorySnapshot {
            profile:    self.load_profile()?,
            facts:      self.load_facts()?,
            procedural: self.load_procedural()?,
            episodes:   self.load_episodes()?,
        })
    }

    /// Import a snapshot. Replaces the profile; upserts facts + procedural by
    /// id/key; appends episodes. (Re-importing the same snapshot is idempotent
    /// for facts/procedural/profile; episodes append — dedupe by id if needed.)
    pub fn import_snapshot(&self, snap: &crate::MemorySnapshot) -> Result<()> {
        self.put_profile(&snap.profile)?;
        for f in &snap.facts {
            self.put_fact(&f.id, &f.topic, &f.content, f.updated_ms)?;
        }
        for p in &snap.procedural {
            self.put_procedural(&p.key, &p.content, p.confidence, p.correction, p.updated_ms)?;
        }
        // Append episodes that aren't already present (dedupe by id).
        let existing: std::collections::HashSet<String> =
            self.load_episodes()?.into_iter().map(|e| e.id).collect();
        for ev in &snap.episodes {
            if !existing.contains(&ev.id) {
                self.append_episode(ev)?;
            }
        }
        Ok(())
    }
}

/// Procedural writes via the plain `Procedural` op carry no timestamp; derive a
/// stable-ish updated_ms of "now" is not available without a clock dep, so we use
/// 0 unless the content embeds one. memoryd (which has a clock) sets ts on the
/// richer ops; this keeps the pure core clock-free + deterministic in tests.
fn now_from(_content: &str) -> u64 {
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (MemoryStore, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let s = MemoryStore::open(dir.path()).unwrap();
        (s, dir)
    }

    fn ep(id: &str, kind: &str, summary: &str, ts_ms: u64, important: bool) -> EpisodeEvent {
        EpisodeEvent {
            id: id.into(), kind: kind.into(), session_id: "s1".into(),
            summary: summary.into(), outcome: "ok".into(), ts_ms, important,
        }
    }

    #[test]
    fn episodes_persist_and_recent_filters_by_time() {
        let (s, _d) = store();
        s.append_episode(&ep("e1", "session_done", "built the OS", 1_000, false)).unwrap();
        s.append_episode(&ep("e2", "tool_error", "network timeout", 90_000_000, false)).unwrap();

        let MemoryResult::Hits { hits } = s.query(MemoryQuery::Recent { since_ms: 50_000_000, layers: vec![] }) else {
            panic!("expected hits");
        };
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "e2");
    }

    #[test]
    fn search_ranks_and_scopes_layers() {
        let (s, _d) = store();
        s.append_episode(&ep("e1", "tool_error", "network timeout on upload", 1_000, false)).unwrap();
        s.put_fact("f1", "network", "the user is on a flaky network", 2_000).unwrap();
        s.put_procedural("p1", "retry uploads on network failure", 0.8, false, 3_000).unwrap();

        // search all layers for "network"
        let MemoryResult::Hits { hits } = s.query(MemoryQuery::Search { query: "network".into(), layers: vec![], limit: 10 }) else {
            panic!("hits");
        };
        assert_eq!(hits.len(), 3, "all three layers match 'network'");

        // scope to episodic only
        let MemoryResult::Hits { hits } = s.query(MemoryQuery::Search { query: "network".into(), layers: vec![MemoryLayer::Episodic], limit: 10 }) else {
            panic!("hits");
        };
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].layer, MemoryLayer::Episodic);
    }

    #[test]
    fn corrections_are_high_priority_and_listed() {
        let (s, _d) = store();
        s.write(MemoryWrite::UserCorrection {
            correction: "don't force push to main".into(),
            context: "git".into(),
            ts_ms: 5_000,
        });
        s.write(MemoryWrite::UserCorrection {
            correction: "prefer concise replies".into(),
            context: String::new(),
            ts_ms: 9_000,
        });

        let MemoryResult::Hits { hits } = s.query(MemoryQuery::Corrections { limit: 10 }) else { panic!("hits"); };
        assert_eq!(hits.len(), 2);
        // most recent first
        assert!(hits[0].content.contains("concise"));
        assert_eq!(hits[0].score, 1.0, "corrections stored at full confidence");

        // a correction also surfaces in search with a relevance boost
        let MemoryResult::Hits { hits } = s.query(MemoryQuery::Search { query: "force push".into(), layers: vec![MemoryLayer::Procedural], limit: 5 }) else { panic!("hits"); };
        assert_eq!(hits.len(), 1);
        assert!(hits[0].content.contains("force push"));
    }

    #[test]
    fn profile_roundtrips_with_safe_privacy_defaults() {
        let (s, _d) = store();
        // default profile (never written) has safe privacy defaults
        let MemoryResult::Profile { profile } = s.query(MemoryQuery::UserProfile) else { panic!("profile"); };
        assert!(!profile.privacy.sync_to_cloud);
        assert_eq!(profile.privacy.memory_retention_days, 90);

        let mut p = UserProfile { display_name: "Diana".into(), ..Default::default() };
        p.expertise.push("rust".into());
        s.write(MemoryWrite::Profile { profile: p.clone() });

        let MemoryResult::Profile { profile } = s.query(MemoryQuery::UserProfile) else { panic!("profile"); };
        assert_eq!(profile.display_name, "Diana");
        assert_eq!(profile.expertise, vec!["rust".to_string()]);
    }

    #[test]
    fn expire_drops_old_unimportant_keeps_important() {
        let (s, _d) = store();
        let now = 100 * 86_400_000; // day 100
        s.append_episode(&ep("old", "x", "old event", 1 * 86_400_000, false)).unwrap();      // day 1
        s.append_episode(&ep("milestone", "x", "kept", 1 * 86_400_000, true)).unwrap();        // day 1, important
        s.append_episode(&ep("fresh", "x", "recent", 99 * 86_400_000, false)).unwrap();        // day 99

        let expired = s.expire(90, now).unwrap();
        assert_eq!(expired, 1, "only the old non-important event expires");

        let MemoryResult::Hits { hits } = s.query(MemoryQuery::Recent { since_ms: 0, layers: vec![MemoryLayer::Episodic] }) else { panic!("hits"); };
        let ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
        assert!(ids.contains(&"milestone"));
        assert!(ids.contains(&"fresh"));
        assert!(!ids.contains(&"old"));
    }

    #[test]
    fn delete_removes_across_layers() {
        let (s, _d) = store();
        s.append_episode(&ep("e1", "x", "an episode", 1_000, false)).unwrap();
        s.put_fact("f1", "topic", "a fact", 2_000).unwrap();
        s.put_procedural("p1", "a recipe", 0.5, false, 3_000).unwrap();

        assert!(s.delete("e1").unwrap());
        assert!(s.delete("f1").unwrap());
        assert!(s.delete("p1").unwrap());
        assert!(!s.delete("nope").unwrap());

        let MemoryResult::Hits { hits } = s.query(MemoryQuery::Recent { since_ms: 0, layers: vec![] }) else { panic!() };
        assert!(hits.is_empty(), "all entries deleted");
    }

    #[test]
    fn clear_wipes_everything() {
        let (s, _d) = store();
        s.append_episode(&ep("e1", "x", "y", 1_000, false)).unwrap();
        s.put_fact("f1", "t", "c", 2_000).unwrap();
        s.clear().unwrap();
        let MemoryResult::Hits { hits } = s.query(MemoryQuery::Recent { since_ms: 0, layers: vec![] }) else { panic!() };
        assert!(hits.is_empty());
    }

    #[test]
    fn export_import_roundtrips() {
        let dir_a = tempfile::tempdir().unwrap();
        let a = MemoryStore::open(dir_a.path()).unwrap();
        a.append_episode(&ep("e1", "x", "remember this", 1_000, true)).unwrap();
        a.put_fact("f1", "user", "likes rust", 2_000).unwrap();
        a.write(MemoryWrite::UserCorrection { correction: "be concise".into(), context: String::new(), ts_ms: 3_000 });
        let mut p = UserProfile::default();
        p.display_name = "Diana".into();
        a.put_profile(&p).unwrap();

        let snap = a.export_snapshot().unwrap();

        // Import into a fresh store on another "device".
        let dir_b = tempfile::tempdir().unwrap();
        let b = MemoryStore::open(dir_b.path()).unwrap();
        b.import_snapshot(&snap).unwrap();

        assert_eq!(b.export_snapshot().unwrap(), snap, "snapshot round-trips exactly");
        // Re-import is idempotent (episodes deduped by id).
        b.import_snapshot(&snap).unwrap();
        assert_eq!(b.export_snapshot().unwrap().episodes.len(), 1);
    }

    #[test]
    fn protocol_json_is_stable() {
        // The wire protocol is tagged on `op`/`kind`, snake_case — memoryd + agentd
        // (and any cross-repo client) depend on this shape.
        let q = MemoryQuery::Search { query: "x".into(), layers: vec![MemoryLayer::Procedural], limit: 3 };
        let v = serde_json::to_value(&q).unwrap();
        assert_eq!(v["op"], "search");
        assert_eq!(v["layers"][0], "procedural");

        let w = MemoryWrite::UserCorrection { correction: "c".into(), context: "ctx".into(), ts_ms: 1 };
        assert_eq!(serde_json::to_value(&w).unwrap()["op"], "user_correction");

        let r = MemoryResult::Ok;
        assert_eq!(serde_json::to_value(&r).unwrap()["kind"], "ok");
    }
}
