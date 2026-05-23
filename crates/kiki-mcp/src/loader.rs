//! Plugin loader — OSTree artifact checkout → capability validate → MCP register.
//!
//! When agentd starts (or an artifact is installed at runtime), the loader:
//! 1. Reads kiki.toml from the artifact's OSTree checkout path
//! 2. Validates declared capabilities against the node's granted capability set
//! 3. Spawns the artifact's process (or connects to its running socket)
//! 4. Registers the artifact with the McpHub

use std::{path::PathBuf, sync::Arc};
use tokio::process::{Child, Command};
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
    /// How to launch the artifact process. Optional: an artifact may instead be
    /// started by an external supervisor (systemd) and just connect to the MCP
    /// socket on its own.
    pub exec: Option<ExecSpec>,
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

#[derive(Debug, serde::Deserialize)]
pub struct ExecSpec {
    /// Binary to run. Relative paths resolve against the artifact directory.
    pub command: String,
    #[serde(default)]
    pub args:    Vec<String>,
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
    /// Returns the child processes the loader spawned (those with an `[exec]`
    /// block); the caller must keep them alive for the artifacts to stay up.
    pub async fn load_directory(&self, root: &str) -> Vec<Child> {
        let mut children = Vec::new();
        let Ok(entries) = std::fs::read_dir(root) else { return children; };

        for entry in entries.flatten() {
            let manifest_path = entry.path().join("kiki.toml");
            if !manifest_path.exists() { continue; }
            match self.load_artifact(manifest_path).await {
                Ok((id, child)) => {
                    info!(artifact = %id, spawned = child.is_some(), "artifact loaded");
                    if let Some(c) = child { children.push(c); }
                }
                Err(e) => { error!(path = ?entry.path(), error = %e, "artifact load failed"); }
            }
        }
        children
    }

    /// Load a single artifact from its kiki.toml path. Validates capabilities,
    /// and if the manifest has an `[exec]` block, spawns the artifact process
    /// (with `KIKI_MCP_SOCKET` set so it connects back to this hub).
    pub async fn load_artifact(&self, manifest_path: PathBuf) -> Result<(String, Option<Child>)> {
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
        info!(
            artifact = %artifact_id,
            version  = %manifest.artifact.version,
            kind     = %manifest.artifact.kind,
            tools    = manifest.tools.len(),
            "artifact manifest validated"
        );

        // Spawn the artifact process if it declares how to launch. Otherwise an
        // external supervisor (systemd) starts it and it connects on its own.
        let child = match &manifest.exec {
            Some(spec) => {
                let dir = manifest_path.parent().map(PathBuf::from).unwrap_or_default();
                let command = {
                    let p = PathBuf::from(&spec.command);
                    if p.is_absolute() { p } else { dir.join(&spec.command) }
                };
                let child = Command::new(&command)
                    .args(&spec.args)
                    .current_dir(&dir)
                    .env("KIKI_MCP_SOCKET", &self.socket_dir)
                    .spawn()
                    .map_err(|e| Error::Io(format!("spawn {}: {e}", command.display())))?;
                info!(artifact = %artifact_id, command = %command.display(), pid = child.id(), "artifact process spawned");
                Some(child)
            }
            None => None,
        };

        Ok((artifact_id, child))
    }
}
