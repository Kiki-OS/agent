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
use kiki_sandbox::{IsolationLevel, MicroVmConfig, SandboxProfile};
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

/// Host-level Firecracker settings, shared by every microVM-isolated artifact.
/// The guest kernel is OS-provided (one image for all microVMs); each artifact
/// supplies its own root filesystem image in its bundle. Absent ⇒ artifacts that
/// require microVM isolation fail to load (fail-closed — never a host fallback).
#[derive(Debug, Clone)]
pub struct FirecrackerConfig {
    /// Path to the `firecracker` binary.
    pub firecracker_bin: String,
    /// Path to the shared guest kernel image (vmlinux).
    pub kernel_image:    String,
}

pub struct PluginLoader {
    hub:         Arc<McpHub>,
    granted:     CapabilitySet,
    socket_dir:  String,
    firecracker: Option<FirecrackerConfig>,
}

impl PluginLoader {
    pub fn new(
        hub:        Arc<McpHub>,
        granted:    CapabilitySet,
        socket_dir: impl Into<String>,
    ) -> Self {
        Self { hub, granted, socket_dir: socket_dir.into(), firecracker: None }
    }

    /// Enable Firecracker-backed isolation for untrusted (`ArtifactKind::Agent`)
    /// artifacts. Without it, such artifacts fail to load rather than run on the
    /// host.
    pub fn with_firecracker(mut self, config: FirecrackerConfig) -> Self {
        self.firecracker = Some(config);
        self
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
                // Runtime paths the app legitimately needs on top of its declared
                // caps: its own checkout dir (binary + manifest) and the dir
                // holding the MCP socket it connects back on.
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

                if profile.isolation == IsolationLevel::Firecracker {
                    // Untrusted (LLM-executing) artifact → hardware-isolated
                    // microVM. Fail-closed at every gap: if Firecracker isn't
                    // configured, the host lacks KVM, or the artifact ships no
                    // rootfs, we refuse to load it — NEVER fall back to a host
                    // process (that would defeat the isolation this kind demands).
                    let fc = self.firecracker.as_ref().ok_or_else(|| {
                        Error::Io(format!(
                            "artifact {artifact_id} requires Firecracker isolation but it is not configured"
                        ))
                    })?;
                    let rootfs = dir.join("rootfs.ext4");
                    if !rootfs.exists() {
                        return Err(Error::Io(format!(
                            "artifact {artifact_id}: Firecracker rootfs missing at {}",
                            rootfs.display()
                        )));
                    }
                    let vm_cfg = MicroVmConfig::for_agent(
                        &artifact_id,
                        &fc.kernel_image,
                        &rootfs.to_string_lossy(),
                    );
                    let vm = vm_cfg
                        .launch(&fc.firecracker_bin)
                        .await
                        .map_err(|e| Error::Io(format!("microvm {artifact_id}: {e}")))?;
                    info!(
                        artifact = %artifact_id, pid = vm.pid, vsock = %vm.vsock_uds,
                        "artifact booted in Firecracker microVM (hardware-isolated)"
                    );
                    // Track the firecracker VMM process like any other child:
                    // killing it tears the guest down.
                    Some(vm.into_child())
                } else {
                    let mut std_cmd = std::process::Command::new(&command);
                    std_cmd
                        .args(&spec.args)
                        .current_dir(&dir)
                        .env("KIKI_MCP_SOCKET", &self.socket_dir);
                    if std::env::var("KIKI_DISABLE_SANDBOX").is_err() {
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
            }
            None => None,
        };

        Ok((artifact_id, child))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_agent_manifest(dir: &std::path::Path) {
        // An `agent` artifact runs model output → SandboxProfile routes it to
        // Firecracker isolation. It declares an [exec] so the loader tries to
        // launch it.
        std::fs::write(
            dir.join("kiki.toml"),
            r#"
[artifact]
id      = "io.kiki.test.agent"
name    = "test-agent"
version = "1.0.0"
kind    = "agent"
license = "MIT"

[capabilities]

[exec]
command = "agent-bin"
"#,
        )
        .unwrap();
    }

    /// An `agent` (untrusted) artifact must NOT load as a host process when
    /// Firecracker isn't configured — it fails closed instead of escaping the
    /// microVM isolation its kind demands.
    #[tokio::test]
    async fn agent_artifact_fails_closed_without_firecracker() {
        let tmp = std::env::temp_dir().join(format!("kiki-loader-fc-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        write_agent_manifest(&tmp);

        let hub = Arc::new(McpHub::new());
        // Trusted=true so capability validation can't be the thing that fails —
        // we're asserting the isolation routing fails closed.
        let loader = PluginLoader::new(hub, CapabilitySet::default(), "/run/kiki/mcp.sock");
        let res = loader.load_artifact(tmp.join("kiki.toml"), true).await;

        assert!(res.is_err(), "agent artifact must not load without Firecracker");
        let msg = format!("{:?}", res.unwrap_err());
        assert!(msg.contains("Firecracker"), "error should name Firecracker: {msg}");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// With Firecracker configured but the artifact shipping no rootfs image, the
    /// loader still fails closed (won't boot a VM with no guest fs).
    #[tokio::test]
    async fn agent_artifact_fails_closed_without_rootfs() {
        let tmp = std::env::temp_dir().join(format!("kiki-loader-fc2-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        write_agent_manifest(&tmp);

        let hub = Arc::new(McpHub::new());
        let loader = PluginLoader::new(hub, CapabilitySet::default(), "/run/kiki/mcp.sock")
            .with_firecracker(FirecrackerConfig {
                firecracker_bin: "/usr/bin/firecracker".into(),
                kernel_image:    "/var/kiki/vm/vmlinux".into(),
            });
        let res = loader.load_artifact(tmp.join("kiki.toml"), true).await;

        assert!(res.is_err(), "agent artifact must not load without a rootfs");
        let msg = format!("{:?}", res.unwrap_err());
        assert!(msg.contains("rootfs"), "error should name the missing rootfs: {msg}");

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
