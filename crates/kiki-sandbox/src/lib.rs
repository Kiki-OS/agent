//! Capability broker and sandboxing layer.
//!
//! Tiered isolation — level chosen based on artifact type declared in kiki.toml:
//!
//!   cli-tool / mcp-tool    → Namespace (Linux namespaces + seccomp)   ~10ms
//!   desktop-app            → Namespace + Wayland protocol filter
//!   headless-app           → Namespace + network namespace
//!   system-service         → capabilities explicitly listed, no FS sandbox
//!   agent (external code)  → Firecracker MicroVM  (LLM output = hostile)
//!
//! Security principle from research: shared-kernel containers are insufficient
//! for untrusted LLM-generated code. Firecracker for anything that executes
//! model output.

pub mod broker;      // capability broker: checks CapabilitySet before every action
pub mod namespace;   // Linux namespaces + seccomp profiles
pub mod landlock;    // Landlock filesystem allowlist (policy + Linux enforcement)
pub mod seccomp;     // seccomp-bpf syscall denylist (policy + Linux enforcement)
pub mod firecracker; // Firecracker MicroVM backend for untrusted code
pub mod wayland;     // Wayland protocol filter (per-surface isolation)

pub use firecracker::{kvm_available, FirecrackerError, MicroVm, MicroVmConfig};
pub use landlock::{FsAccess, LandlockRule, RuntimePaths};
pub use namespace::{IsolationLevel, SandboxError, SandboxProfile};

use async_trait::async_trait;
use kiki_core::{capability::CapabilitySet, error::Result};

#[async_trait]
pub trait SandboxBackend: Send + Sync {
    async fn spawn(&self, artifact_id: &str, caps: &CapabilitySet) -> Result<SandboxHandle>;
}

pub struct SandboxHandle {
    pub pid:         u32,
    pub socket_path: std::path::PathBuf,
}
