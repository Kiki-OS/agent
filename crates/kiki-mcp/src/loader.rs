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
    capability::{Capability, CapabilitySet},
    error::{Error, Result},
};
use kiki_schema::ArtifactManifest;
use kiki_sandbox::SandboxProfile;
use crate::hub::McpHub;

// The artifact manifest is `kiki_schema::ArtifactManifest` — the single manifest
// format (also what `kpkg` writes to disk and the registry signs). It carries
// the structured `[capabilities]`, optional `[exec]` launch block, and `[[tools]]`.

/// Scan an apps directory and build each artifact's egress allowlist from its
/// manifest's `[capabilities].network`. The `"*"` sentinel (dynamic egress) maps
/// to `HostPort{host:"*", port:0}`, which the broker treats as per-call dynamic.
/// Used by agentd to seed the [`kiki_net::EgressBroker`] before serving.
pub fn scan_egress_allowlists(root: &str) -> Vec<(String, Vec<kiki_net::HostPort>)> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(root) else { return out; };
    for entry in entries.flatten() {
        let manifest_path = entry.path().join("kiki.toml");
        if !manifest_path.exists() {
            continue;
        }
        let Ok(raw) = std::fs::read_to_string(&manifest_path) else { continue; };
        let Ok(manifest) = toml::from_str::<ArtifactManifest>(&raw) else { continue; };
        let hosts: Vec<kiki_net::HostPort> = manifest
            .capabilities
            .network
            .iter()
            .filter_map(|s| {
                if s == "*" {
                    Some(kiki_net::HostPort { host: "*".into(), port: 0 })
                } else {
                    kiki_net::HostPort::parse(s)
                }
            })
            .collect();
        if !hosts.is_empty() {
            out.push((manifest.artifact.id.clone(), hosts));
        }
    }
    out
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
    pub async fn load_directory(&self, root: &str, trusted: bool) -> Vec<Child> {
        let mut children = Vec::new();
        let Ok(entries) = std::fs::read_dir(root) else { return children; };

        for entry in entries.flatten() {
            let manifest_path = entry.path().join("kiki.toml");
            if !manifest_path.exists() { continue; }
            match self.load_artifact(manifest_path, trusted).await {
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
    pub async fn load_artifact(
        &self,
        manifest_path: PathBuf,
        trusted: bool,
    ) -> Result<(String, Option<Child>)> {
        let raw = std::fs::read_to_string(&manifest_path)
            .map_err(|e| Error::Io(e.to_string()))?;

        let manifest: ArtifactManifest = toml::from_str(&raw)
            .map_err(|e| Error::Parse(e.to_string()))?;

        // Built-in apps (baked into the immutable image) are trusted — their
        // declared capabilities are part of the OS. User-installed (L2) apps are
        // validated against the node's granted set (deny-by-default); the grant
        // is written by kpkg on install. The cap→token mapping is shared with
        // kpkg (`Capability::from_manifest`) so grant and validate can't drift.
        if !trusted {
            for cap in Capability::from_manifest(&manifest.capabilities) {
                if !self.granted.contains(&cap) {
                    warn!(
                        artifact = %manifest.artifact.id,
                        capability = ?cap,
                        "artifact requires capability not granted — skipping"
                    );
                    return Err(Error::CapabilityDeniedByName(format!("{cap:?}")));
                }
            }
        }

        let artifact_id = manifest.artifact.id.clone();
        info!(
            artifact = %artifact_id,
            version  = %manifest.artifact.version,
            kind     = ?manifest.artifact.kind,
            tools    = manifest.tools.len(),
            "artifact manifest validated"
        );

        // Spawn the artifact process if it declares how to launch. Otherwise an
        // external supervisor (systemd) starts it and it connects on its own.
        // The process is sandboxed per its derived profile — an app is put in an
        // empty network namespace so it can reach the network ONLY via the
        // brokered net.fetch (set `KIKI_DISABLE_SANDBOX=1` to opt out in dev).
        let child = match &manifest.exec {
            Some(spec) => {
                let dir = manifest_path.parent().map(PathBuf::from).unwrap_or_default();
                let command = {
                    let p = PathBuf::from(&spec.command);
                    if p.is_absolute() { p } else { dir.join(&spec.command) }
                };
                let mut std_cmd = std::process::Command::new(&command);
                std_cmd
                    .args(&spec.args)
                    .current_dir(&dir)
                    .env("KIKI_MCP_SOCKET", &self.socket_dir);
                if std::env::var("KIKI_DISABLE_SANDBOX").is_err() {
                    // Runtime paths the app legitimately needs on top of its
                    // declared caps: its own checkout dir (binary + manifest) and
                    // the dir holding the MCP socket it connects back on.
                    let mcp_socket_dir = std::path::Path::new(&self.socket_dir)
                        .parent()
                        .map(|p| p.to_string_lossy().into_owned())
                        .unwrap_or_else(|| "/run/kiki".to_string());
                    let runtime = kiki_sandbox::RuntimePaths {
                        binary_dir: dir.to_string_lossy().into_owned(),
                        mcp_socket_dir,
                    };
                    let profile = SandboxProfile::for_artifact(
                        manifest.artifact.kind,
                        &manifest.capabilities,
                        &manifest.permissions,
                    )
                    .with_runtime_paths(&runtime);
                    profile
                        .apply(&mut std_cmd)
                        .map_err(|e| Error::Io(format!("sandbox {artifact_id}: {e}")))?;
                }
                let child = Command::from(std_cmd)
                    .spawn()
                    .map_err(|e| Error::Io(format!("spawn {}: {e}", command.display())))?;
                info!(artifact = %artifact_id, command = %command.display(), pid = child.id(), "artifact process spawned (sandboxed)");
                Some(child)
            }
            None => None,
        };

        Ok((artifact_id, child))
    }
}
