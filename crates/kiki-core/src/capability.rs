use std::collections::HashSet;

/// Fine-grained OS capability, declared in kiki.toml [capabilities].
/// The capability broker (kiki-sandbox) enforces these at runtime.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Capability {
    // Filesystem
    FsRead(String),
    FsWrite(String),

    // Network
    NetworkOutbound,
    NetworkInbound(u16),
    NetworkStream,

    // Hardware
    AudioOutput,
    AudioInput,
    GpuCompute,
    GpuDisplay,
    UsbDevice(String),

    // System
    ProcessSpawn,
    ProcessKill,
    SystemdUnit(String),
    KernelModule(String),

    // Wayland / GUI
    WaylandSurface,
    WaylandInput,
    AccessibilityTree,

    // Agent-specific
    AgentSpawn,
    AgentKill,
    FleetControl,
    SecretsRead(String),
}

impl Capability {
    /// Map an artifact's declared manifest capabilities (`kiki-schema`) onto the
    /// OS capability tokens the gate + egress broker enforce. The single source
    /// of this mapping — used by `kpkg` (to write grants) and the plugin loader
    /// (to validate against the node's granted set), so they can't drift.
    ///
    /// Coarse where the OS token is coarse: any declared `network` entry maps to
    /// `NetworkOutbound` (the per-host allowlist is enforced separately by the
    /// egress broker from the same `network` list).
    pub fn from_manifest(cs: &kiki_schema::CapabilitySet) -> Vec<Capability> {
        let mut caps = Vec::new();
        for p in &cs.fs_read {
            caps.push(Capability::FsRead(p.clone()));
        }
        for p in &cs.fs_write {
            caps.push(Capability::FsWrite(p.clone()));
        }
        if !cs.network.is_empty() {
            caps.push(Capability::NetworkOutbound);
        }
        if !cs.exec.is_empty() {
            caps.push(Capability::ProcessSpawn);
        }
        if cs.mcp_spawn {
            caps.push(Capability::AgentSpawn);
        }
        if cs.display {
            caps.push(Capability::WaylandSurface);
        }
        if cs.audio_in {
            caps.push(Capability::AudioInput);
        }
        for p in &cs.vault.read {
            caps.push(Capability::SecretsRead(p.clone()));
        }
        caps
    }
}

#[derive(Debug, Clone, Default)]
pub struct CapabilitySet(HashSet<Capability>);

impl CapabilitySet {
    pub fn new() -> Self { Self::default() }
    pub fn insert(&mut self, cap: Capability) { self.0.insert(cap); }
    pub fn contains(&self, cap: &Capability) -> bool { self.0.contains(cap) }
    pub fn is_subset_of(&self, other: &CapabilitySet) -> bool {
        self.0.is_subset(&other.0)
    }

    /// Checks a capability, respecting the current control mode.
    /// In BypassPermissions mode all capability checks pass unconditionally.
    pub fn check(&self, cap: &Capability, bypass: bool) -> CapabilityResult {
        if bypass { return CapabilityResult::Bypassed; }
        if self.contains(cap) { CapabilityResult::Allowed } else { CapabilityResult::Denied }
    }

    /// Check by capability variant name (e.g. "FsRead", "NetworkOutbound").
    /// Used by the plugin loader when validating kiki.toml capability declarations.
    /// Matching is by variant prefix — "FsRead" matches any FsRead(_) grant.
    pub fn has_by_name(&self, name: &str) -> bool {
        self.0.iter().any(|c| {
            let debug_str = format!("{c:?}");
            debug_str == name || debug_str.starts_with(&format!("{name}("))
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapabilityResult {
    Allowed,
    /// Bypassed because ControlMode::BypassPermissions is active.
    /// The action proceeds but is written to the audit log.
    Bypassed,
    Denied,
}
