//! Session context — the working memory of a running agent session.
//!
//! Context travels through the harness turn loop and persists to the state
//! backend after each Durable step. It is the unit of migration: freezing
//! a session means serializing this struct to a RuntimeSnapshot.

use std::sync::Arc;
use crate::{
    capability::CapabilitySet,
    interrupt::Interrupt,
    state::StateBackend,
    surface::SessionLayout,
    types::ConversationMessage,
};

// ─── ControlMode ──────────────────────────────────────────────────────────────

/// Governs how the harness gate and interrupt system behave.
///
/// Serialized identically to kiki-hud's ControlMode (snake_case) — the
/// compositor and agentd exchange this value over the control socket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlMode {
    /// Fully autonomous — gate passes all tool calls immediately.
    /// All events written to the audit log; checkpoint every N steps.
    BypassPermissions,

    /// Default — gate passes all capability-granted tool calls.
    /// Interrupts shown non-blocking in the HUD.
    #[default]
    AgentMode,

    /// Gate pauses before destructive tools for explicit user approval.
    /// Non-destructive tools pass without interruption.
    AssistedMode,

    /// Gate pauses before every tool call. Full human-in-the-loop.
    HumanMode,
}

impl ControlMode {
    pub fn is_bypass(self) -> bool { matches!(self, Self::BypassPermissions) }
    pub fn is_autonomous(self) -> bool {
        matches!(self, Self::BypassPermissions | Self::AgentMode)
    }
    pub fn requires_approval_for_all(self) -> bool {
        matches!(self, Self::HumanMode)
    }
}

// ─── Context ──────────────────────────────────────────────────────────────────

/// Runtime context passed through every harness turn.
///
/// `messages` is the live conversation history that gets passed to the LLM.
/// `state` is the durable backend (OSTree in production, memory in tests).
/// Everything else is session metadata.
pub struct Context {
    // Identity
    pub agent_id:   String,
    pub session_id: String,
    pub label:      String,         // human-readable goal, shown in the HUD title
    pub scenario:   Option<String>, // layout scenario key → remote surface template

    // Conversation history (the LLM's working memory)
    pub messages: Vec<ConversationMessage>,

    // Capability + mode
    pub capabilities: CapabilitySet,
    pub control_mode: ControlMode,

    // Interrupt audit log
    pub interrupt_log: Vec<Interrupt>,

    // State backend (durable/eternal persistence)
    pub state: Arc<dyn StateBackend>,

    // Session limits
    pub max_steps: Option<u32>,
    steps_taken:   u32,

    // Desktop integration
    pub layout:      SessionLayout,
    pub active_apps: Vec<String>,   // app IDs with open surfaces in this session
}

impl Context {
    pub fn new(
        agent_id:   impl Into<String>,
        session_id: impl Into<String>,
        state:      Arc<dyn StateBackend>,
    ) -> Self {
        Self {
            agent_id:     agent_id.into(),
            session_id:   session_id.into(),
            label:        String::new(),
            scenario:     None,
            messages:     Vec::new(),
            capabilities: CapabilitySet::new(),
            control_mode: ControlMode::default(),
            interrupt_log: Vec::new(),
            state,
            max_steps:    None,
            steps_taken:  0,
            layout:       SessionLayout::default(),
            active_apps:  Vec::new(),
        }
    }

    /// Reconstruct a Context from a migrated RuntimeSnapshot.
    ///
    /// Capabilities are not snapshotted (they're tied to installed artifacts
    /// on the *target* node) — the caller resolves them after receiving the bundle.
    pub fn from_snapshot(
        runtime:      &crate::state::RuntimeSnapshot,
        state:        Arc<dyn StateBackend>,
        capabilities: CapabilitySet,
    ) -> Self {
        Self {
            agent_id:     runtime.agent_id.clone(),
            session_id:   runtime.session_id.clone(),
            label:        runtime.session_label.clone(),
            scenario:     runtime.scenario.clone(),
            messages:     runtime.messages.clone(),
            capabilities,
            control_mode: runtime.control_mode,
            interrupt_log: runtime.interrupt_queue.clone(),
            state,
            max_steps:    None,
            steps_taken:  runtime.step as u32,
            layout:       runtime.layout,
            active_apps:  runtime.active_apps.clone(),
        }
    }

    // ── Message accessors ─────────────────────────────────────────────────────

    pub fn push_message(&mut self, msg: ConversationMessage) {
        let is_assistant = matches!(msg, ConversationMessage::Assistant(_));
        self.messages.push(msg);
        if is_assistant {
            self.steps_taken += 1;
        }
    }

    // Convenience — expose push_message correctly for assistant/tool results
    pub fn push_assistant(&mut self, turn: crate::types::AssistantTurn) {
        self.steps_taken += 1;
        self.messages.push(ConversationMessage::Assistant(turn));
    }

    pub fn push_tool_results(&mut self, results: Vec<crate::types::ToolResult>) {
        self.messages.push(ConversationMessage::ToolResults { results });
    }

    pub fn push_user_text(&mut self, text: impl Into<String>) {
        self.messages.push(ConversationMessage::user_text(text));
    }

    pub fn push_perception(&mut self, blocks: Vec<crate::types::ContentBlock>) {
        if !blocks.is_empty() {
            self.messages.push(ConversationMessage::user(blocks));
        }
    }

    /// The system prompt is always the first message. Returns it if present.
    pub fn system_prompt(&self) -> Option<&str> {
        self.messages.first().and_then(|m| {
            if let ConversationMessage::System { content } = m {
                Some(content.as_str())
            } else {
                None
            }
        })
    }

    pub fn set_system_prompt(&mut self, prompt: String) {
        let msg = ConversationMessage::system(prompt);
        if self.messages.is_empty() {
            self.messages.push(msg);
        } else if matches!(self.messages[0], ConversationMessage::System { .. }) {
            self.messages[0] = msg;
        } else {
            self.messages.insert(0, msg);
        }
    }

    // ── Interrupt log ─────────────────────────────────────────────────────────

    pub fn log_interrupt(&mut self, interrupt: Interrupt) {
        self.interrupt_log.push(interrupt);
    }

    // ── Step tracking ─────────────────────────────────────────────────────────

    pub fn steps_taken(&self) -> u32 { self.steps_taken }

    pub fn step_limit_reached(&self) -> bool {
        self.max_steps.map_or(false, |max| self.steps_taken >= max)
    }

    // ── Mode helpers ──────────────────────────────────────────────────────────

    pub fn is_bypass(&self) -> bool { self.control_mode.is_bypass() }

    pub fn set_mode(&mut self, mode: ControlMode) {
        self.control_mode = mode;
    }
}
