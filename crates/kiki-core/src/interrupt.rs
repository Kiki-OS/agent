use serde::{Deserialize, Serialize};

/// An interrupt from the agent to the human.
/// Behavior depends on the active ControlMode (gated at the compositor / agentd layer).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Interrupt {
    pub id:      String,
    pub kind:    InterruptKind,
    pub message: String,
    #[serde(default)]
    pub context: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterruptKind {
    /// Agent found ambiguity it cannot resolve autonomously.
    ///
    /// BypassPermissions → enqueued to audit log; agent picks best option and continues.
    /// AgentMode         → non-blocking overlay shown to human; agent waits up to timeout.
    /// AssistedMode      → blocks until human responds (Approve/Reject/Redirect).
    /// HumanMode         → blocks until human responds.
    DecisionRequired,

    /// Agent is about to execute a destructive or irreversible action.
    ///
    /// BypassPermissions → suppressed; written to audit log; agent proceeds.
    /// AgentMode         → written to session log; agent proceeds.
    /// AssistedMode      → non-blocking confirmation banner; agent waits for Enter/Esc.
    /// HumanMode         → modal confirmation; agent waits.
    Confirmation,

    /// Anomaly or error the human should be aware of.
    /// Never blocks. Written to audit log + StatusBar overlay in all modes.
    Attention,

    /// Progress update. Written only to StatusBar and session log.
    /// Invisible in BypassPermissions (audit log only).
    Info,
}

impl InterruptKind {
    /// Whether this interrupt can be silently resolved by the agent without
    /// human input. True only for BypassPermissions mode on blocking types.
    pub fn can_be_autonomous(&self) -> bool {
        matches!(self, Self::DecisionRequired | Self::Confirmation)
    }

    /// Whether this interrupt produces a visible UI element (overlay, banner).
    pub fn is_visual(&self, bypass: bool) -> bool {
        if bypass { return false; }
        matches!(self, Self::DecisionRequired | Self::Confirmation | Self::Attention)
    }
}

/// Human response to a blocking interrupt.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterruptResolution {
    Approved,
    Rejected,
    /// Human redirects with new intent; agent resets plan with this context.
    Redirected { new_intent: String },
    /// Timed out without response — agent proceeds with best-effort decision.
    TimedOut,
}

impl Interrupt {
    pub fn decision(id: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            id:      id.into(),
            kind:    InterruptKind::DecisionRequired,
            message: message.into(),
            context: serde_json::Value::Null,
        }
    }

    pub fn confirmation(id: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            id:      id.into(),
            kind:    InterruptKind::Confirmation,
            message: message.into(),
            context: serde_json::Value::Null,
        }
    }

    pub fn info(message: impl Into<String>) -> Self {
        Self {
            id:      String::new(),
            kind:    InterruptKind::Info,
            message: message.into(),
            context: serde_json::Value::Null,
        }
    }

    pub fn with_context(mut self, ctx: serde_json::Value) -> Self {
        self.context = ctx;
        self
    }
}
