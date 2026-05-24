//! Namespace isolation for native apps.
//!
//! The security property that backs the egress broker: an L2 app is spawned into
//! an empty **network namespace** (only loopback, no routes), so it physically
//! cannot open external sockets. Its only path to the network is the brokered
//! `net.fetch` over the MCP Unix socket — which is filesystem-scoped and works
//! across network namespaces — routed through agentd's egress broker (allowlist
//! + credential injection + audit). Forces "all egress through the broker."
//!
//! Profile derivation is pure + cross-platform (tested everywhere). Application
//! is Linux-only (unshare in a pre-exec hook); on other platforms it is a no-op
//! so the workspace builds + tests on dev machines.

use kiki_schema::{ArtifactKind, CapabilitySet, PermissionSet};

/// How strongly an artifact is isolated, chosen from its declared kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsolationLevel {
    /// No isolation — trusted system services (declared `system_service`).
    None,
    /// Linux namespaces incl. an empty network namespace (default for apps).
    NamespaceNet,
    /// Firecracker MicroVM — for artifacts that execute untrusted/LLM code.
    Firecracker,
}

/// The concrete sandbox to apply when spawning an artifact.
#[derive(Debug, Clone)]
pub struct SandboxProfile {
    pub isolation:    IsolationLevel,
    /// Spawn into an empty network namespace (no external sockets).
    pub deny_network: bool,
    /// Filesystem paths the artifact may read / write (raw declared caps; the
    /// enforced Landlock allowlist is derived via [`SandboxProfile::landlock_rules`]).
    pub fs_read:      Vec<String>,
    pub fs_write:     Vec<String>,
    /// Whether the artifact may spawn child processes.
    pub allow_exec:   bool,
    /// Resolved Landlock allowlist to enforce at spawn. Empty until
    /// [`SandboxProfile::with_runtime_paths`] is called (and always empty for a
    /// `system_service`), in which case no filesystem restriction is applied.
    pub landlock:     Vec<crate::landlock::LandlockRule>,
    /// Install the seccomp-bpf syscall denylist at spawn. True for sandboxed
    /// (namespaced) artifacts; false for trusted `system_service`s.
    pub seccomp:      bool,
}

impl SandboxProfile {
    /// Derive the profile from the artifact's declared kind + capabilities.
    ///
    /// - `system_service` permission -> [`IsolationLevel::None`] (trusted).
    /// - [`ArtifactKind::Agent`] (runs model output) -> [`IsolationLevel::Firecracker`].
    /// - everything else (app/component/durable/model) -> [`IsolationLevel::NamespaceNet`]
    ///   with the network denied: even apps that declare egress hosts reach them
    ///   only via the broker, never directly.
    pub fn for_artifact(kind: ArtifactKind, caps: &CapabilitySet, perms: &PermissionSet) -> Self {
        if perms.system_service {
            return Self {
                isolation:    IsolationLevel::None,
                deny_network: false,
                fs_read:      caps.fs_read.clone(),
                fs_write:     caps.fs_write.clone(),
                allow_exec:   true,
                landlock:     Vec::new(),
                seccomp:      false,
            };
        }
        let isolation = match kind {
            ArtifactKind::Agent => IsolationLevel::Firecracker,
            _ => IsolationLevel::NamespaceNet,
        };
        Self {
            isolation,
            deny_network: true,
            fs_read:      caps.fs_read.clone(),
            fs_write:     caps.fs_write.clone(),
            allow_exec:   !caps.exec.is_empty(),
            landlock:     Vec::new(),
            seccomp:      true,
        }
    }

    /// Resolve + attach the Landlock allowlist for this profile, given the
    /// runtime paths the app legitimately needs (its binary dir + the MCP socket
    /// dir). After this, [`apply`](Self::apply) enforces the filesystem allowlist
    /// in addition to the network namespace. Without it, no fs restriction is
    /// applied (back-compat for callers/tests that only want the netns).
    pub fn with_runtime_paths(mut self, runtime: &crate::landlock::RuntimePaths) -> Self {
        self.landlock = self.landlock_rules(runtime);
        self
    }

    /// Apply the profile to a process about to be spawned.
    ///
    /// Linux: installs a pre-exec hook that unshares the requested namespaces in
    /// the child before launch. Returns an error for [`IsolationLevel::Firecracker`]
    /// (the MicroVM backend isn't wired yet — fail closed rather than run
    /// untrusted code unsandboxed). Non-Linux: no-op (dev builds).
    pub fn apply(&self, cmd: &mut std::process::Command) -> Result<(), SandboxError> {
        if self.isolation == IsolationLevel::Firecracker {
            return Err(SandboxError::FirecrackerUnavailable);
        }
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::process::CommandExt;
            let mut flags = nix::sched::CloneFlags::empty();
            if self.deny_network {
                flags |= nix::sched::CloneFlags::CLONE_NEWNET;
            }
            let rules = self.landlock.clone();
            let seccomp = self.seccomp;
            if !flags.is_empty() || !rules.is_empty() || seccomp {
                // SAFETY: the closure runs in the forked child before execve, in
                // order: (1) unshare the requested namespaces, (2) install the
                // Landlock filesystem allowlist, (3) install the seccomp-bpf
                // syscall denylist. seccomp is last so it can't block the landlock
                // syscalls. All three fail closed (error → spawn aborts) rather
                // than launching the app under-sandboxed.
                //
                // Landlock + seccomp are validated on a real kernel (Fedora VM),
                // not on this darwin dev host where both are no-ops.
                unsafe {
                    cmd.pre_exec(move || {
                        if !flags.is_empty() {
                            nix::sched::unshare(flags)
                                .map_err(|e| std::io::Error::from_raw_os_error(e as i32))?;
                        }
                        if !rules.is_empty() {
                            crate::landlock::enforce(&rules).map_err(|e| {
                                std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
                            })?;
                        }
                        if seccomp {
                            crate::seccomp::apply().map_err(|e| {
                                std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
                            })?;
                        }
                        Ok(())
                    });
                }
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = cmd; // namespaces are Linux-only; nothing to do on dev hosts.
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SandboxError {
    #[error("firecracker backend unavailable — refusing to run untrusted artifact unsandboxed")]
    FirecrackerUnavailable,
    #[error("landlock: {0}")]
    Landlock(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps_with_network() -> CapabilitySet {
        let mut c = CapabilitySet::default();
        c.network.push("api.example.com:443".into());
        c
    }

    #[test]
    fn app_is_network_denied_even_when_it_declares_egress() {
        let p = SandboxProfile::for_artifact(
            ArtifactKind::App,
            &caps_with_network(),
            &PermissionSet::default(),
        );
        assert_eq!(p.isolation, IsolationLevel::NamespaceNet);
        assert!(p.deny_network, "apps must reach the network only via the broker");
    }

    #[test]
    fn system_service_is_not_isolated() {
        let perms = PermissionSet { system_service: true, ..Default::default() };
        let p = SandboxProfile::for_artifact(ArtifactKind::App, &CapabilitySet::default(), &perms);
        assert_eq!(p.isolation, IsolationLevel::None);
        assert!(!p.deny_network);
    }

    #[test]
    fn agent_kind_requires_firecracker_and_fails_closed() {
        let p = SandboxProfile::for_artifact(
            ArtifactKind::Agent,
            &CapabilitySet::default(),
            &PermissionSet::default(),
        );
        assert_eq!(p.isolation, IsolationLevel::Firecracker);
        let mut cmd = std::process::Command::new("/bin/true");
        assert!(matches!(p.apply(&mut cmd), Err(SandboxError::FirecrackerUnavailable)));
    }

    #[test]
    fn applying_namespace_profile_succeeds() {
        // On non-Linux this is a no-op; on Linux it installs the pre-exec hook.
        let p = SandboxProfile::for_artifact(
            ArtifactKind::App,
            &CapabilitySet::default(),
            &PermissionSet::default(),
        );
        let mut cmd = std::process::Command::new("/bin/true");
        assert!(p.apply(&mut cmd).is_ok());
    }
}
