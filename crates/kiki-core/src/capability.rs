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
