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

    /// The compositor's structured view of what is on screen, pushed whenever the
    /// surface set changes. The agent-first perception channel: agentd caches the
    /// latest inventory and exposes it via the built-in `screen.inventory` tool so
    /// the agent perceives the screen as DATA (not by scraping a pixel a11y tree).
    SurfaceInventory { surfaces: Vec<SurfaceInfo> },

    /// Resume a previously-parked session by id into the running daemon. agentd
    /// reconstructs it from its local snapshot bundle (NOT a command for the
    /// current harness — agentd intercepts it before the command channel).
    ResumeSession { session_id: String },

    /// Create a brand-new desktop session in the running daemon, with its own
    /// isolated app instances. The DE owns the `session_id` (its desktop join
    /// key). agentd intercepts this before the command channel.
    CreateSession { session_id: String, label: Option<String> },

    /// OOBE step input from the shell: the user submitted a value for `step`.
    /// agentd intercepts this before the command channel; the OOBE state
    /// machine drives the conversation until it emits `OobeComplete`.
    OobeInput { step: String, value: serde_json::Value },

    /// Park the named session and hold it in a "locked" suspended state
    /// (distinct from a simple park — the session is suspended in memory
    /// awaiting an unlock rather than discarded to disk).
    LockSession { session_id: String },

    /// Unlock a previously locked session, resuming it. `pin` is validated
    /// when present; for now any pin (or `None`) is accepted — real hardware
    /// auth is a future extension.
    UnlockSession { session_id: String, pin: Option<String> },
}

// ─── OOBE ────────────────────────────────────────────────────────────────────

/// Coarse phases of the out-of-box experience flow.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OobeStepKind {
    /// Show the welcome screen and prompt for a device name.
    Welcome,
    /// Link the device to a cloud account (device flow).
    AccountSetup,
    /// Trigger `kpkg install` of the default model (fire-and-forget download).
    ModelDownload,
    /// OOBE complete — initial desktop session has been created.
    Done,
}

/// One on-screen surface as the compositor sees it. Byte-compatible with
/// `kiki_shell_core::SurfaceInfo` in the DE repo (the two repos share no code,
/// only this JSON shape).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceInfo {
    pub app_id:  String,
    pub title:   String,
    pub x:       i32,
    pub y:       i32,
    pub w:       i32,
    pub h:       i32,
    pub focused: bool,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The SurfaceInventory wire shape MUST match `kiki_shell_core::ControlMessage`
    /// in the DE repo (the two repos share no code). This fixes the JSON so cross-
    /// repo drift is caught.
    #[test]
    fn surface_inventory_wire_matches_de() {
        let m = ControlMessage::SurfaceInventory {
            surfaces: vec![SurfaceInfo {
                app_id: "org.gnome.TextEditor".into(),
                title: "Untitled".into(),
                x: 0,
                y: 0,
                w: 800,
                h: 600,
                focused: true,
            }],
        };
        assert_eq!(
            serde_json::to_value(&m).unwrap(),
            serde_json::json!({
                "type": "surface_inventory",
                "surfaces": [{
                    "app_id": "org.gnome.TextEditor",
                    "title": "Untitled",
                    "x": 0, "y": 0, "w": 800, "h": 600,
                    "focused": true
                }]
            })
        );

        // And it round-trips from the DE's serialized form.
        let line = r#"{"type":"surface_inventory","surfaces":[{"app_id":"a","title":"t","x":1,"y":2,"w":3,"h":4,"focused":false}]}"#;
        match serde_json::from_str::<ControlMessage>(line).unwrap() {
            ControlMessage::SurfaceInventory { surfaces } => {
                assert_eq!(surfaces.len(), 1);
                assert_eq!(surfaces[0].app_id, "a");
                assert_eq!(surfaces[0].w, 3);
            }
            other => panic!("expected SurfaceInventory, got {other:?}"),
        }
    }

    #[test]
    fn resume_session_wire_matches_de() {
        let m = ControlMessage::ResumeSession { session_id: "s1".into() };
        assert_eq!(
            serde_json::to_value(&m).unwrap(),
            serde_json::json!({ "type": "resume_session", "session_id": "s1" })
        );
        // Round-trips from the DE's form.
        let parsed: ControlMessage =
            serde_json::from_str(r#"{"type":"resume_session","session_id":"x"}"#).unwrap();
        assert!(matches!(parsed, ControlMessage::ResumeSession { session_id } if session_id == "x"));
    }

    #[test]
    fn create_session_wire_matches_de() {
        let m = ControlMessage::CreateSession { session_id: "d2".into(), label: Some("Work".into()) };
        assert_eq!(
            serde_json::to_value(&m).unwrap(),
            serde_json::json!({ "type": "create_session", "session_id": "d2", "label": "Work" })
        );
        // label is optional (null/omitted) and round-trips from the DE's form.
        let parsed: ControlMessage =
            serde_json::from_str(r#"{"type":"create_session","session_id":"d3","label":null}"#).unwrap();
        assert!(matches!(parsed, ControlMessage::CreateSession { session_id, label }
            if session_id == "d3" && label.is_none()));
    }

    #[test]
    fn oobe_and_lock_wire_matches_de() {
        // OobeInput
        let m = ControlMessage::OobeInput {
            step:  "welcome".into(),
            value: serde_json::json!({ "device_name": "kiki-home" }),
        };
        let v = serde_json::to_value(&m).unwrap();
        assert_eq!(v["type"], "oobe_input");
        assert_eq!(v["step"], "welcome");
        assert_eq!(v["value"]["device_name"], "kiki-home");
        let back: ControlMessage = serde_json::from_value(v).unwrap();
        assert!(matches!(back, ControlMessage::OobeInput { ref step, .. } if step == "welcome"));

        // LockSession
        let m = ControlMessage::LockSession { session_id: "s1".into() };
        assert_eq!(
            serde_json::to_value(&m).unwrap(),
            serde_json::json!({ "type": "lock_session", "session_id": "s1" })
        );

        // UnlockSession with pin
        let m = ControlMessage::UnlockSession { session_id: "s1".into(), pin: Some("1234".into()) };
        let v = serde_json::to_value(&m).unwrap();
        assert_eq!(v["type"], "unlock_session");
        assert_eq!(v["pin"], "1234");

        // UnlockSession without pin
        let m = ControlMessage::UnlockSession { session_id: "s2".into(), pin: None };
        let v = serde_json::to_value(&m).unwrap();
        assert_eq!(v["type"], "unlock_session");
        assert_eq!(v["pin"], serde_json::Value::Null);
    }
}
