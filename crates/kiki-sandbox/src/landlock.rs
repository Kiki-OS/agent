//! Landlock filesystem policy — turns an app's *advisory* `fs_read`/`fs_write`
//! capability declarations into an enforced allowlist.
//!
//! Two layers, deliberately split so the security decision is testable anywhere:
//!
//! 1. **Policy** ([`SandboxProfile::landlock_rules`]) — pure, cross-platform:
//!    given the declared caps + the runtime paths an app legitimately needs
//!    (its own binary dir, the MCP socket), compute the complete set of
//!    [`LandlockRule`]s. Everything not covered is denied. Fully unit-tested.
//!
//! 2. **Enforcement** ([`enforce`]) — Linux-only (Landlock LSM, kernel ≥ 5.13):
//!    install the ruleset on the calling process. Best-effort: on a kernel
//!    without Landlock it logs and continues (the network namespace + the
//!    advisory caps still apply). Compiled but pending validation on a real
//!    Linux kernel during the OS image build — it cannot run on this dev host.

use crate::namespace::SandboxProfile;

/// Filesystem access a rule grants on a path subtree (handled hierarchically by
/// Landlock — a rule on `/a` covers `/a/b/c`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FsAccess {
    pub read:    bool,
    pub write:   bool,
    pub execute: bool,
}

impl FsAccess {
    /// Read + execute (binaries, shared libraries, read-only assets).
    pub const RX: Self = Self { read: true, write: false, execute: true };
    /// Read only.
    pub const R: Self = Self { read: true, write: false, execute: false };
    /// Read + write (data dirs, sockets, scratch).
    pub const RW: Self = Self { read: true, write: true, execute: false };

    /// Merge two grants on the same path (union of permissions).
    fn union(self, other: Self) -> Self {
        Self {
            read:    self.read || other.read,
            write:   self.write || other.write,
            execute: self.execute || other.execute,
        }
    }
}

/// One Landlock path rule: grant `access` on the `path` subtree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LandlockRule {
    pub path:   String,
    pub access: FsAccess,
}

/// Runtime paths an app legitimately needs beyond its declared caps. These are
/// added to every namespaced app's ruleset so it can actually start + talk to
/// agentd, without widening access to anything else.
#[derive(Debug, Clone)]
pub struct RuntimePaths {
    /// The app's checkout dir (binary + manifest + bundled assets) — RX.
    pub binary_dir: String,
    /// Dir containing the MCP socket the app connects back on — RW.
    pub mcp_socket_dir: String,
}

/// Read-only system paths every app needs for dynamic linking + locale/config.
/// (In the empty network namespace there is no DNS, so `/etc` is for ld.so and
/// timezone data, not resolv.conf.)
const SYSTEM_RX_PATHS: &[&str] = &["/usr", "/lib", "/lib64", "/bin"];
const SYSTEM_R_PATHS:  &[&str] = &["/etc"];
/// Scratch dir (also the fallback app-data location when the real dir is
/// unwritable). RW but isolated to /tmp.
const SCRATCH_RW_PATH: &str = "/tmp";

impl SandboxProfile {
    /// Compute the complete Landlock allowlist for this profile. The returned
    /// rules are de-duplicated by path (permissions unioned) and sorted by path
    /// for determinism. Anything not represented here is denied once enforced.
    ///
    /// A `system_service` (isolation `None`) gets no filesystem restriction:
    /// returns an empty rule set, meaning "do not enforce Landlock".
    pub fn landlock_rules(&self, runtime: &RuntimePaths) -> Vec<LandlockRule> {
        use crate::namespace::IsolationLevel;
        if self.isolation == IsolationLevel::None {
            return Vec::new();
        }

        use std::collections::BTreeMap;
        let mut by_path: BTreeMap<String, FsAccess> = BTreeMap::new();
        let mut add = |path: &str, access: FsAccess| {
            if path.is_empty() {
                return;
            }
            by_path
                .entry(path.to_string())
                .and_modify(|a| *a = a.union(access))
                .or_insert(access);
        };

        // Mandatory runtime paths.
        add(&runtime.binary_dir, FsAccess::RX);
        add(&runtime.mcp_socket_dir, FsAccess::RW);
        for p in SYSTEM_RX_PATHS {
            add(p, FsAccess::RX);
        }
        for p in SYSTEM_R_PATHS {
            add(p, FsAccess::R);
        }
        add(SCRATCH_RW_PATH, FsAccess::RW);

        // Declared capabilities.
        for p in &self.fs_read {
            add(p, FsAccess::R);
        }
        for p in &self.fs_write {
            add(p, FsAccess::RW);
        }

        by_path
            .into_iter()
            .map(|(path, access)| LandlockRule { path, access })
            .collect()
    }
}

/// Enforce a Landlock ruleset on the calling process (Linux only).
///
/// Best-effort: returns `Ok(false)` when the kernel has no Landlock support (so
/// the caller can proceed — the netns + advisory caps still hold), `Ok(true)`
/// when the ruleset was applied, and `Err` only on an unexpected failure.
///
/// NOTE: compiled but UNVALIDATED on this dev host (darwin). Must be exercised
/// against a real kernel during the OS image build.
#[cfg(target_os = "linux")]
pub fn enforce(rules: &[LandlockRule]) -> Result<bool, crate::namespace::SandboxError> {
    use landlock::{
        Access, AccessFs, PathBeneath, PathFd, Ruleset, RulesetAttr, RulesetCreatedAttr,
        RulesetStatus, ABI,
    };

    if rules.is_empty() {
        return Ok(false); // nothing to enforce (e.g. system_service)
    }

    let map_err = |e: landlock::RulesetError| crate::namespace::SandboxError::Landlock(e.to_string());

    let abi = ABI::V2;
    // Handle every filesystem access right this ABI knows about, so anything not
    // explicitly granted by a rule below is denied.
    let mut ruleset = Ruleset::default()
        .handle_access(AccessFs::from_all(abi))
        .map_err(map_err)?
        .create()
        .map_err(map_err)?;

    for rule in rules {
        // Skip paths that don't exist on this node rather than failing the whole
        // ruleset — a missing optional dir shouldn't unsandbox the app.
        let Ok(fd) = PathFd::new(&rule.path) else { continue; };
        // Read/exec subtrees get the read-only right set; writable subtrees get
        // the full set (these are the app's own data dirs / sockets). Using the
        // ABI helpers keeps the right bits correct across kernel versions.
        let access = if rule.access.write {
            AccessFs::from_all(abi)
        } else {
            AccessFs::from_read(abi)
        };
        ruleset = ruleset
            .add_rule(PathBeneath::new(fd, access))
            .map_err(map_err)?;
    }

    let status = ruleset.restrict_self().map_err(map_err)?;
    Ok(status.ruleset != RulesetStatus::NotEnforced)
}

/// Non-Linux stub: Landlock is a Linux LSM. No-op so the workspace builds + the
/// policy layer can be tested on dev machines.
#[cfg(not(target_os = "linux"))]
pub fn enforce(_rules: &[LandlockRule]) -> Result<bool, crate::namespace::SandboxError> {
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiki_schema::{ArtifactKind, CapabilitySet, PermissionSet};

    fn runtime() -> RuntimePaths {
        RuntimePaths {
            binary_dir:     "/usr/libexec/kiki/apps/io.kiki.settings".into(),
            mcp_socket_dir: "/run/kiki".into(),
        }
    }

    fn settings_profile() -> SandboxProfile {
        let mut caps = CapabilitySet::default();
        caps.fs_read.push("/var/kiki/apps-data/io.kiki.settings".into());
        caps.fs_write.push("/var/kiki/apps-data/io.kiki.settings".into());
        SandboxProfile::for_artifact(ArtifactKind::App, &caps, &PermissionSet::default())
    }

    fn access_for<'a>(rules: &'a [LandlockRule], path: &str) -> Option<&'a FsAccess> {
        rules.iter().find(|r| r.path == path).map(|r| &r.access)
    }

    #[test]
    fn app_gets_its_data_dir_rw_and_binary_rx() {
        let rules = settings_profile().landlock_rules(&runtime());

        let data = access_for(&rules, "/var/kiki/apps-data/io.kiki.settings").unwrap();
        assert_eq!(*data, FsAccess::RW, "app must read+write its own data dir");

        let bin = access_for(&rules, "/usr/libexec/kiki/apps/io.kiki.settings").unwrap();
        assert_eq!(*bin, FsAccess::RX, "app must read+exec its own binary dir");

        let sock = access_for(&rules, "/run/kiki").unwrap();
        assert_eq!(*sock, FsAccess::RW, "app must reach the MCP socket");
    }

    #[test]
    fn app_cannot_touch_other_apps_data_or_secrets() {
        let rules = settings_profile().landlock_rules(&runtime());
        // No rule grants another app's data dir or the node secrets dir.
        assert!(access_for(&rules, "/var/kiki/apps-data/io.kiki.notes").is_none());
        assert!(access_for(&rules, "/var/kiki/secrets").is_none());
        // And /var itself is not blanket-granted — only the specific data subtree.
        assert!(access_for(&rules, "/var").is_none());
        assert!(access_for(&rules, "/var/kiki").is_none());
    }

    #[test]
    fn system_paths_are_read_or_exec_never_write() {
        let rules = settings_profile().landlock_rules(&runtime());
        assert_eq!(*access_for(&rules, "/usr").unwrap(), FsAccess::RX);
        assert_eq!(*access_for(&rules, "/etc").unwrap(), FsAccess::R);
        // /etc must never be writable by an app.
        assert!(!access_for(&rules, "/etc").unwrap().write);
    }

    #[test]
    fn read_and_write_on_same_path_unions_to_rw() {
        let mut caps = CapabilitySet::default();
        caps.fs_read.push("/data/shared".into());
        caps.fs_write.push("/data/shared".into());
        let p = SandboxProfile::for_artifact(ArtifactKind::App, &caps, &PermissionSet::default());
        let rules = p.landlock_rules(&runtime());
        assert_eq!(*access_for(&rules, "/data/shared").unwrap(), FsAccess::RW);
    }

    #[test]
    fn system_service_is_not_landlocked() {
        let perms = PermissionSet { system_service: true, ..Default::default() };
        let p = SandboxProfile::for_artifact(ArtifactKind::App, &CapabilitySet::default(), &perms);
        assert!(
            p.landlock_rules(&runtime()).is_empty(),
            "system services run unrestricted — empty rule set means 'do not enforce'"
        );
    }

    #[test]
    fn rules_are_sorted_and_deduped() {
        let rules = settings_profile().landlock_rules(&runtime());
        let paths: Vec<&str> = rules.iter().map(|r| r.path.as_str()).collect();
        let mut sorted = paths.clone();
        sorted.sort();
        assert_eq!(paths, sorted, "rules must be deterministic (sorted by path)");
        // No duplicate paths.
        let mut uniq = paths.clone();
        uniq.dedup();
        assert_eq!(paths.len(), uniq.len(), "no duplicate path rules");
    }
}
