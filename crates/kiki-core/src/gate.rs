//! Capability gate — intercepts every tool call before execution.
//!
//! The gate enforces two orthogonal policies:
//!
//! 1. **Static capability grant** — declared in kiki.toml [capabilities].
//!    If the tool requires a capability the session doesn't have, it's denied
//!    in all modes (no exceptions).
//!
//! 2. **ControlMode policy** — how the gate responds to granted capabilities:
//!    - BypassPermissions: pass immediately, write to audit log.
//!    - AgentMode: pass immediately, write to session log.
//!    - AssistedMode: gate destructive tools only (approval dialog).
//!    - HumanMode: gate every tool call (approval dialog).
//!
//! The gate communicates with the compositor via `surface_tx` (sends approval
//! dialogs) and `pending_approvals` (matches responses to waiting callers).

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};
use tokio::sync::{mpsc, oneshot};
use crate::{
    capability::CapabilitySet,
    context::ControlMode,
    error::Result,
    interrupt::{Interrupt, InterruptKind},
    surface::{ChoiceStyle, InterruptChoice, SurfaceSignal},
    types::{ApprovalDecision, ToolCall},
};

// ─── Gate config ──────────────────────────────────────────────────────────────

/// Tools that require explicit approval in AssistedMode (even if capability is granted).
/// These are actions that are expensive, irreversible, or have broad impact.
const DESTRUCTIVE_TOOLS: &[&str] = &[
    "fs_delete", "fs_move", "fs_write",
    "process_kill", "systemd_unit_stop", "systemd_unit_restart",
    "agent_spawn", "agent_kill",
    "network_outbound_post", "secrets_write",
    "session_migrate", "session_close",
];

fn is_destructive(tool_name: &str) -> bool {
    DESTRUCTIVE_TOOLS.contains(&tool_name)
}

/// Maps a tool name to the capability variant it requires, if any.
///
/// Returned values are capability *variant* names (matched via
/// [`CapabilitySet::has_by_name`], which treats `"FsRead"` as matching any
/// `FsRead(_)` grant). Tools absent from this table require no special
/// capability — pure-compute or read-only informational tools (`get_weather`,
/// `echo`, calculators) pass the static check unconditionally.
fn required_capability(tool_name: &str) -> Option<&'static str> {
    let cap = match tool_name {
        "fs_read" | "fs_list" | "fs_stat"                       => "FsRead",
        "fs_write" | "fs_delete" | "fs_move" | "fs_mkdir"       => "FsWrite",
        "process_spawn" | "shell_exec"                          => "ProcessSpawn",
        "process_kill"                                          => "ProcessKill",
        "network_outbound_get" | "network_outbound_post"
            | "http_fetch"                                      => "NetworkOutbound",
        "audio_play"                                            => "AudioOutput",
        "audio_record"                                          => "AudioInput",
        "agent_spawn"                                           => "AgentSpawn",
        "agent_kill"                                            => "AgentKill",
        "secrets_read"                                          => "SecretsRead",
        "wayland_input_inject"                                  => "WaylandInput",
        "fleet_control"                                         => "FleetControl",
        s if s.starts_with("systemd_unit_")                    => "SystemdUnit",
        _ => return None,
    };
    Some(cap)
}

// ─── Capability gate ──────────────────────────────────────────────────────────

pub type GateHandle = Arc<CapabilityGate>;

pub struct CapabilityGate {
    /// Capability set granted to this session via kiki.toml.
    pub capabilities: CapabilitySet,

    /// Channel to push approval request dialogs to the compositor.
    surface_tx: mpsc::Sender<SurfaceSignal>,

    /// Pending one-shot channels waiting for user approval responses.
    /// Keyed by `request_id`.
    pending: Mutex<HashMap<String, oneshot::Sender<ApprovalDecision>>>,

    /// Audit log — all bypassed/approved/rejected gate decisions.
    audit: Mutex<Vec<GateEvent>>,
}

impl CapabilityGate {
    pub fn new(capabilities: CapabilitySet, surface_tx: mpsc::Sender<SurfaceSignal>) -> Arc<Self> {
        Arc::new(Self {
            capabilities,
            surface_tx,
            pending: Mutex::new(HashMap::new()),
            audit:   Mutex::new(Vec::new()),
        })
    }

    /// Main gate check — called by the harness before every tool execution.
    ///
    /// Returns `Ok(true)` = proceed, `Ok(false)` = skip (rejected or denied),
    /// `Err(_)` = gate error (channel closed, timeout).
    pub async fn check(
        &self,
        call:    &ToolCall,
        mode:    ControlMode,
        bypass:  bool,
    ) -> Result<GateDecision> {
        // 1. Static capability check (enforced in every mode except bypass).
        //    Deny-by-default: a tool that maps to a capability the session was
        //    not granted is skipped regardless of ControlMode.
        if !bypass {
            if let Some(required) = required_capability(&call.name) {
                if !self.capabilities.has_by_name(required) {
                    self.audit_decision(call, AuditAction::Denied);
                    return Ok(GateDecision::Skip {
                        reason: format!("missing capability: {required}"),
                    });
                }
            }
        }

        // 2. ControlMode policy
        let decision = match mode {
            ControlMode::BypassPermissions => {
                self.audit_decision(call, AuditAction::Bypassed);
                GateDecision::Proceed
            }

            ControlMode::AgentMode => {
                self.audit_decision(call, AuditAction::Allowed);
                GateDecision::Proceed
            }

            ControlMode::AssistedMode => {
                if is_destructive(&call.name) {
                    self.request_approval(call, mode).await?
                } else {
                    self.audit_decision(call, AuditAction::Allowed);
                    GateDecision::Proceed
                }
            }

            ControlMode::HumanMode => {
                self.request_approval(call, mode).await?
            }
        };

        Ok(decision)
    }

    /// Called by the harness control loop when an ApprovalResponse arrives
    /// from the compositor (user tapped Approve/Reject in the HUD).
    pub fn resolve_approval(&self, request_id: &str, decision: ApprovalDecision) {
        if let Some(tx) = self.pending.lock().unwrap().remove(request_id) {
            let _ = tx.send(decision);
        }
    }

    /// Full audit log — used for BypassPermissions review and compliance.
    pub fn audit_log(&self) -> Vec<GateEvent> {
        self.audit.lock().unwrap().clone()
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    async fn request_approval(&self, call: &ToolCall, mode: ControlMode) -> Result<GateDecision> {
        let request_id = format!("gate-{}-{}", call.name, uuid_v4_simple());
        let (tx, rx) = oneshot::channel::<ApprovalDecision>();
        self.pending.lock().unwrap().insert(request_id.clone(), tx);

        // Show approval dialog in the HUD
        let _ = self.surface_tx.send(SurfaceSignal::ApprovalRequired {
            request_id:   request_id.clone(),
            tool_name:    call.name.clone(),
            description:  format!("Allow `{}` to run?", call.name),
            input_preview: call.input.clone(),
        }).await;

        // Also emit a blocking interrupt to pause the status bar
        let interrupt = Interrupt::confirmation(
            &request_id,
            format!("Waiting for approval: {}", call.name),
        );
        let _ = self.surface_tx.send(SurfaceSignal::Interrupt {
            interrupt_id: request_id.clone(),
            kind:         InterruptKind::Confirmation,
            message:      interrupt.message,
            choices:      vec![
                InterruptChoice { id: "approve".into(), label: "Approve".into(), style: ChoiceStyle::Primary },
                InterruptChoice { id: "reject".into(),  label: "Reject".into(),  style: ChoiceStyle::Destructive },
            ],
        }).await;

        // Set a timeout proportional to mode urgency
        let timeout = if mode == ControlMode::HumanMode {
            std::time::Duration::from_secs(300)  // 5 min for explicit human mode
        } else {
            std::time::Duration::from_secs(60)   // 1 min for assisted
        };

        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(ApprovalDecision::Approved)) => {
                self.audit_decision(call, AuditAction::Approved);
                Ok(GateDecision::Proceed)
            }
            Ok(Ok(ApprovalDecision::Rejected)) => {
                self.audit_decision(call, AuditAction::Rejected);
                Ok(GateDecision::Skip { reason: "Rejected by user".into() })
            }
            Ok(Ok(ApprovalDecision::Redirected { new_intent })) => {
                Ok(GateDecision::Redirect { new_intent })
            }
            Ok(Err(_)) | Err(_) => {
                // Channel closed or timed out
                self.audit_decision(call, AuditAction::TimedOut);
                Ok(GateDecision::Skip { reason: "Approval timed out".into() })
            }
        }
    }

    fn audit_decision(&self, call: &ToolCall, action: AuditAction) {
        self.audit.lock().unwrap().push(GateEvent {
            tool_name: call.name.clone(),
            action,
            timestamp_ms: now_ms(),
        });
    }
}

// ─── Gate outcome ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum GateDecision {
    /// Execute the tool call normally.
    Proceed,
    /// Skip this tool call — insert a synthetic rejected result into history.
    Skip { reason: String },
    /// User redirected — discard current plan, restart with new_intent as context.
    Redirect { new_intent: String },
}

// ─── Audit ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GateEvent {
    pub tool_name:    String,
    pub action:       AuditAction,
    pub timestamp_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuditAction {
    Allowed,
    Bypassed,
    Approved,
    Rejected,
    TimedOut,
    Denied,
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn uuid_v4_simple() -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    now_ms().hash(&mut h);
    format!("{:x}", h.finish())
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::Capability;
    use serde_json::json;

    fn gate(caps: CapabilitySet) -> Arc<CapabilityGate> {
        // Receiver is dropped; approval paths aren't exercised in these tests.
        let (tx, _rx) = mpsc::channel::<SurfaceSignal>(8);
        CapabilityGate::new(caps, tx)
    }

    fn call(name: &str) -> ToolCall {
        ToolCall { id: "c1".into(), name: name.into(), input: json!({}) }
    }

    #[tokio::test]
    async fn unmapped_tool_proceeds_without_capability() {
        let g = gate(CapabilitySet::new());
        let d = g.check(&call("get_weather"), ControlMode::AgentMode, false).await.unwrap();
        assert!(matches!(d, GateDecision::Proceed));
    }

    #[tokio::test]
    async fn sensitive_tool_denied_without_grant() {
        let g = gate(CapabilitySet::new());
        let d = g.check(&call("fs_write"), ControlMode::AgentMode, false).await.unwrap();
        match d {
            GateDecision::Skip { reason } => assert!(reason.contains("FsWrite"), "got: {reason}"),
            other => panic!("expected Skip, got {other:?}"),
        }
        assert_eq!(g.audit_log().last().unwrap().action, AuditAction::Denied);
    }

    #[tokio::test]
    async fn sensitive_tool_allowed_with_grant() {
        let mut caps = CapabilitySet::new();
        caps.insert(Capability::FsWrite("/var/kiki".into()));
        let g = gate(caps);
        // Granted FsWrite(_) matches the "FsWrite" requirement by variant name.
        let d = g.check(&call("fs_write"), ControlMode::AgentMode, false).await.unwrap();
        assert!(matches!(d, GateDecision::Proceed));
    }

    #[tokio::test]
    async fn bypass_skips_capability_enforcement() {
        let g = gate(CapabilitySet::new());
        // Even with no grant, BypassPermissions (bypass=true) proceeds.
        let d = g.check(&call("fs_write"), ControlMode::BypassPermissions, true).await.unwrap();
        assert!(matches!(d, GateDecision::Proceed));
        assert_eq!(g.audit_log().last().unwrap().action, AuditAction::Bypassed);
    }

    #[tokio::test]
    async fn systemd_prefix_requires_capability() {
        let g = gate(CapabilitySet::new());
        let d = g.check(&call("systemd_unit_restart"), ControlMode::AgentMode, false).await.unwrap();
        assert!(matches!(d, GateDecision::Skip { .. }));
    }
}
