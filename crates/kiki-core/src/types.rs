//! Conversation-based message types for the agentic harness.
//!
//! Models the Anthropic API message structure natively — the harness speaks
//! this format internally; providers convert to/from their wire format.
//! Replaces the old Observation/Plan/PlanStep PRA types.

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ─── Content blocks ───────────────────────────────────────────────────────────

/// A single content block within a user or assistant message.
/// Multi-modal: text, OS perception, or image (last resort).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text { text: String },

    /// Native Wayland accessibility tree — first-class perception.
    /// No screenshots needed; full semantic structure available.
    WaylandTree { session_id: String, tree: Value },

    /// Direct state snapshot from a Kiki-native app (IPC push).
    AppState { app_id: String, state: Value },

    /// Kernel event: inotify, netlink, procfs, cgroup notification.
    KernelEvent { kind: String, data: Value },

    /// AT-SPI tree for legacy (non-Kiki) apps.
    AtSpi { window_id: u64, tree: Value },

    /// Base64-encoded image. Used only when no structural API is available.
    Image { media_type: String, data_base64: String },
}

impl ContentBlock {
    pub fn text(s: impl Into<String>) -> Self {
        ContentBlock::Text { text: s.into() }
    }
    pub fn is_perception(&self) -> bool {
        !matches!(self, ContentBlock::Text { .. } | ContentBlock::Image { .. })
    }
}

// ─── Tool calling ─────────────────────────────────────────────────────────────

/// A tool invocation emitted by the LLM in an assistant turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Provider-assigned id — used to match tool_use with tool_result.
    pub id:    String,
    pub name:  String,
    pub input: Value,
}

/// Result of executing a ToolCall.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub content:      String,
    pub is_error:     bool,
}

impl ToolResult {
    pub fn ok(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self { tool_call_id: id.into(), content: content.into(), is_error: false }
    }
    pub fn err(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self { tool_call_id: id.into(), content: content.into(), is_error: true }
    }
    pub fn rejected(id: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::err(id, format!("[rejected by user] {}", reason.into()))
    }
}

// ─── Conversation messages ────────────────────────────────────────────────────

/// The content of an assistant turn — mirrors Anthropic's assistant message structure.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AssistantTurn {
    /// Extended thinking block (Claude 3.7+ thinking budget).
    pub thinking:   Option<String>,
    /// Prose output from the model (may be empty if only tool calls were generated).
    pub text:       Option<String>,
    /// Tool calls emitted in this turn (zero or more, executed in parallel or sequence).
    pub tool_calls: Vec<ToolCall>,
}

impl AssistantTurn {
    pub fn has_tool_calls(&self) -> bool { !self.tool_calls.is_empty() }
    pub fn is_terminal(&self) -> bool { self.tool_calls.is_empty() }
}

/// A message in the conversation history.
///
/// The harness builds the LLM context from these messages each turn.
/// Providers convert this representation to their wire format.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum ConversationMessage {
    /// Injected once at the start of each session. Not included in token counting.
    System { content: String },

    /// User input + any OS perceptions gathered this turn.
    User { content: Vec<ContentBlock>, timestamp_ms: u64 },

    /// LLM response: thinking + text + tool calls.
    Assistant(AssistantTurn),

    /// Results of the tool calls from the previous assistant turn.
    /// Always follows an Assistant message that had tool_calls.
    ToolResults { results: Vec<ToolResult> },
}

impl ConversationMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self::System { content: content.into() }
    }

    pub fn user(content: Vec<ContentBlock>) -> Self {
        Self::User {
            content,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        }
    }

    pub fn user_text(text: impl Into<String>) -> Self {
        Self::user(vec![ContentBlock::text(text)])
    }
}

// ─── Control messages (compositor → agentd) ───────────────────────────────────

/// Messages received from the compositor or remote client over the control socket.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ControlMessage {
    /// User typed text input (prompt bar, voice-to-text).
    UserInput { text: String },

    /// User changed ControlMode via HUD toggle.
    ModeChange { mode: crate::context::ControlMode },

    /// User responded to an approval request (AssistedMode / HumanMode).
    ApprovalResponse { request_id: String, decision: ApprovalDecision },

    /// Hard stop: user pressed the stop button.
    StopSession { session_id: String },

    /// Park session (freeze without migration).
    ParkSession { session_id: String },

    /// Migrate session to target node.
    MigrateSession { session_id: String, target_node: String },

    /// Capture a point-in-time snapshot of the current session and upload it as
    /// fleet snapshot `snapshot_id` (for multiply / clone). Does not freeze.
    CaptureSnapshot { snapshot_id: String },
}

/// User's response to an in-context approval dialog.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalDecision {
    Approved,
    Rejected,
    /// User provides a new direction; agent replans with this as context.
    Redirected { new_intent: String },
}

// ─── Legacy compatibility re-exports ─────────────────────────────────────────

pub use crate::state::{MigrationBundle, OstreeCheckpoint, RuntimeSnapshot};
