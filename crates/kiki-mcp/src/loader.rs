//! Plugin loader — OSTree artifact checkout → capability validate → MCP register.
//!
//! When agentd starts (or an artifact is installed at runtime), the loader:
//! 1. Reads kiki.toml from the artifact's OSTree checkout path
//! 2. Validates declared capabilities against the node's granted capability set
//! 3. Spawns the artifact's process (or connects to its running socket)
//! 4. Registers the artifact with the McpHub

use std::{path::PathBuf, sync::Arc};
use tracing::{error, info, warn};
use kiki_core::{
    capability::CapabilitySet,
    error::{Error, Result},
};
use crate::hub::McpHub;

// ─── Artifact manifest (kiki.toml) ───────────────────────────────────────────

#[derive(Debug, serde::Deserialize)]
pub struct ArtifactManifest {
    pub artifact: ArtifactMeta,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub tools: Vec<ToolDecl>,
}

#[derive(Debug, serde::Deserialize)]
pub struct ArtifactMeta {
    pub id:      String,
    pub version: String,
    pub kind:    String,   // app | component | agent | durable
}

#[derive(Debug, serde::Deserialize)]
pub struct ToolDecl {
    pub name:        String,
    pub description: String,
}

// ─── PluginLoader ─────────────────────────────────────────────────────────────

pub struct PluginLoader {
    hub:        Arc<McpHub>,
    granted:    CapabilitySet,
    socket_dir: String,
}

impl PluginLoader {
    pub fn new(
        hub:        Arc<McpHub>,
        granted:    CapabilitySet,
        socket_dir: impl Into<String>,
    ) -> Self {
        Self { hub, granted, socket_dir: socket_dir.into() }
    }

    /// Load all artifacts from an OSTree checkout root (e.g. /var/kiki/apps).
    pub async fn load_directory(&self, root: &str) -> usize {
        let mut loaded = 0;
        let Ok(entries) = std::fs::read_dir(root) else { return 0; };

        for entry in entries.flatten() {
            let manifest_path = entry.path().join("kiki.toml");
            if !manifest_path.exists() { continue; }
            match self.load_artifact(manifest_path).await {
                Ok(id) => { info!(artifact = %id, "artifact loaded"); loaded += 1; }
                Err(e) => { error!(path = ?entry.path(), error = %e, "artifact load failed"); }
            }
        }
        loaded
    }

    /// Load a single artifact from its kiki.toml path.
    pub async fn load_artifact(&self, manifest_path: PathBuf) -> Result<String> {
        let raw = std::fs::read_to_string(&manifest_path)
            .map_err(|e| Error::Io(e.to_string()))?;

        let manifest: ArtifactManifest = toml::from_str(&raw)
            .map_err(|e| Error::Parse(e.to_string()))?;

        // Validate capabilities
        for cap_str in &manifest.capabilities {
            if !self.granted.has_by_name(cap_str) {
                warn!(
                    artifact = %manifest.artifact.id,
                    capability = %cap_str,
                    "artifact requires capability not granted — skipping"
                );
                return Err(Error::CapabilityDeniedByName(cap_str.to_string()));
            }
        }

        let artifact_id = manifest.artifact.id.clone();

        // The artifact process is expected to connect to the MCP socket on its own.
        // The loader just validates and logs; the McpServer handles the connection.
        // For headless/embedded artifacts, we'd spawn a process here.
        info!(
            artifact = %artifact_id,
            version  = %manifest.artifact.version,
            kind     = %manifest.artifact.kind,
            tools    = manifest.tools.len(),
            "artifact manifest validated"
        );

        Ok(artifact_id)
    }
}
