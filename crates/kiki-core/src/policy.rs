//! Node capability policy — the on-disk grant ledger agentd loads at boot.
//!
//! The package manager (`kpkg`) writes grants here when an artifact is installed
//! and the operator (or fleet policy) approves its declared capabilities; agentd
//! reads it and seeds the [`CapabilityGate`](crate::gate). Grants are keyed by
//! artifact id so an uninstall can revoke exactly what it granted, without
//! touching other artifacts' capabilities.
//!
//! A legacy flat `{ "granted": [...] }` shape is still honored (merged in) so
//! existing policy files keep working.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::capability::{Capability, CapabilitySet};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodePolicy {
    /// Legacy flat grant list (operator-managed, not attributed to an artifact).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub granted: Vec<Capability>,
    /// Per-artifact grants, keyed by artifact id (`<ns>/<name>@<version>`).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub grants: BTreeMap<String, Vec<Capability>>,
}

impl NodePolicy {
    /// Load the policy from `path`. A missing or unreadable file yields an empty
    /// policy (deny-by-default); a malformed file is an error so callers can warn.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, PolicyError> {
        match std::fs::read_to_string(path.as_ref()) {
            Ok(raw) => serde_json::from_str(&raw).map_err(PolicyError::Parse),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(PolicyError::Io(e)),
        }
    }

    /// Persist the policy to `path`, creating parent directories as needed.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), PolicyError> {
        let path = path.as_ref();
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(PolicyError::Io)?;
        }
        let json = serde_json::to_string_pretty(self).map_err(PolicyError::Parse)?;
        std::fs::write(path, json).map_err(PolicyError::Io)
    }

    /// Flatten every grant (legacy + per-artifact) into a single [`CapabilitySet`].
    pub fn to_capability_set(&self) -> CapabilitySet {
        let mut set = CapabilitySet::new();
        for c in &self.granted {
            set.insert(c.clone());
        }
        for caps in self.grants.values() {
            for c in caps {
                set.insert(c.clone());
            }
        }
        set
    }

    /// Total number of distinct capability grants (for logging).
    pub fn grant_count(&self) -> usize {
        self.granted.len() + self.grants.values().map(Vec::len).sum::<usize>()
    }

    /// Record (or replace) the grants attributed to `artifact_id`.
    pub fn set_artifact(&mut self, artifact_id: impl Into<String>, caps: Vec<Capability>) {
        self.grants.insert(artifact_id.into(), caps);
    }

    /// Revoke all grants attributed to `artifact_id`. Returns whether anything
    /// was removed.
    pub fn remove_artifact(&mut self, artifact_id: &str) -> bool {
        self.grants.remove(artifact_id).is_some()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PolicyError {
    #[error("policy i/o: {0}")]
    Io(std::io::Error),
    #[error("policy parse: {0}")]
    Parse(serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flattens_legacy_and_per_artifact_grants() {
        let mut p = NodePolicy::default();
        p.granted.push(Capability::NetworkOutbound);
        p.set_artifact(
            "u1/notes@1.0.0",
            vec![Capability::FsRead("/var/kiki".into()), Capability::AudioInput],
        );
        let set = p.to_capability_set();
        assert!(set.contains(&Capability::NetworkOutbound));
        assert!(set.contains(&Capability::AudioInput));
        assert!(set.contains(&Capability::FsRead("/var/kiki".into())));
        assert_eq!(p.grant_count(), 3);
    }

    #[test]
    fn remove_artifact_revokes_only_its_grants() {
        let mut p = NodePolicy::default();
        p.set_artifact("a@1", vec![Capability::AudioInput]);
        p.set_artifact("b@1", vec![Capability::NetworkOutbound]);
        assert!(p.remove_artifact("a@1"));
        assert!(!p.remove_artifact("a@1"));
        let set = p.to_capability_set();
        assert!(!set.contains(&Capability::AudioInput));
        assert!(set.contains(&Capability::NetworkOutbound));
    }

    #[test]
    fn roundtrips_through_json() {
        let mut p = NodePolicy::default();
        p.set_artifact("x/y@2.0.0", vec![Capability::WaylandSurface]);
        let json = serde_json::to_string(&p).unwrap();
        let back: NodePolicy = serde_json::from_str(&json).unwrap();
        assert!(back.to_capability_set().contains(&Capability::WaylandSurface));
    }

    #[test]
    fn parses_legacy_flat_shape() {
        let raw = r#"{ "granted": ["NetworkOutbound", { "FsRead": "/x" }] }"#;
        let p: NodePolicy = serde_json::from_str(raw).unwrap();
        let set = p.to_capability_set();
        assert!(set.contains(&Capability::NetworkOutbound));
        assert!(set.contains(&Capability::FsRead("/x".into())));
    }
}
