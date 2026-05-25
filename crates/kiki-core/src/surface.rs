//! Surface signals: the harness → compositor communication channel.
//!
//! The harness emits these signals over a tokio channel; the compositor (kiki-ded)
//! listens on the IPC socket and translates them into Wayland surface operations.
//!
//! Design principle: the agent drives UI intent, the compositor owns layout policy.
//! The agent says "I need a card here" — the compositor decides where to put it.

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ─── Session layout ───────────────────────────────────────────────────────────

/// The agent's desired screen layout for this session.
/// The compositor translates this to concrete Wayland surface geometry.
///
/// Also defined in kiki-de/crates/kiki-session — serialized identically for IPC.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionLayout {
    /// Agent occupies the full screen. No other app surfaces visible.
    Fullscreen,

    /// Agent sidebar + one app surface. `ratio` is the agent sidebar width (0.0–1.0).
    SplitTwo { ratio_agent: u8 },  // percent 0-100, e.g. 30 = 30% sidebar

    /// One app is fullscreen; agent is a narrow context panel (e.g. 20% width).
    FocusContext,

    /// Four surfaces in a 2×2 grid (multi-task).
    GridFour,

    /// Agent is ambient / invisible. User has full desktop. Agent listens passively.
    Ambient,
}

impl Default for SessionLayout {
    fn default() -> Self { Self::SplitTwo { ratio_agent: 30 } }
}

// ─── Surface signal ───────────────────────────────────────────────────────────

/// A signal emitted by the harness and consumed by the compositor.
///
/// Signals are fire-and-forget over a bounded channel; the compositor buffers
/// and applies them in order. For acknowledgement-required operations (layout
/// changes), the compositor sends a ControlMessage back over the control socket.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SurfaceSignal {
    // ── Agent state feedback ─────────────────────────────────────────────────

    /// Agent is thinking — show streaming text in the status strip or card.
    Thinking { text: String },

    /// Agent completed its turn — dismiss thinking indicator, show result.
    Done { summary: String },

    /// Agent is running a tool — show progress in status bar.
    ToolRunning { tool_name: String },

    /// Tool completed — optionally show a brief result notification.
    ToolDone { tool_name: String, success: bool },

    // ── Human approval (AssistedMode / HumanMode) ────────────────────────────

    /// Show an approval dialog before executing a gated tool.
    /// The user's response arrives as ControlMessage::ApprovalResponse.
    ApprovalRequired {
        request_id:  String,
        tool_name:   String,
        description: String,
        /// Serialized tool input for display (masked if contains secrets).
        input_preview: Value,
    },

    // ── Interrupt overlays ───────────────────────────────────────────────────

    /// Show an interrupt overlay (Decision / Confirmation / Attention / Info).
    Interrupt {
        interrupt_id: String,
        kind:         crate::interrupt::InterruptKind,
        message:      String,
        /// Available response options shown as buttons in the HUD.
        choices:      Vec<InterruptChoice>,
    },

    // ── Widget / card output ─────────────────────────────────────────────────

    /// Show or update a widget card on the surface grid.
    /// The compositor matches by `widget_id` — same id = update in place.
    ShowWidget {
        widget_id:    String,
        widget_type:  String,  // matches kiki-widgets SCENARIO_TEMPLATES key
        data:         Value,
        surface_kind: SurfaceKind,
    },

    /// Remove a widget card from the surface.
    DismissWidget { widget_id: String },

    // ── Layout control ───────────────────────────────────────────────────────

    /// Request a layout change for this session.
    /// The compositor may ignore or negotiate (sends ControlMessage back).
    RequestLayout { layout: SessionLayout },

    /// Request a new surface of a given kind for content output.
    RequestSurface { kind: SurfaceKind, label: String },

    // ── Status bar ───────────────────────────────────────────────────────────

    /// Update the status bar message (session-scoped, bottom strip).
    Status { text: String, level: StatusLevel },
}

/// The visual hierarchy level of a surface request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SurfaceKind {
    /// Background ambient content (always visible, non-intrusive).
    Ambient,
    /// Quick-glance notification strip.
    Glanceable,
    /// Structured data card (most common for tool outputs).
    Card,
    /// Sidebar panel (persistent reference info).
    Panel,
    /// Take over the screen (for immersive tasks).
    Fullscreen,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterruptChoice {
    pub id:    String,
    pub label: String,
    pub style: ChoiceStyle,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChoiceStyle {
    Primary,
    Destructive,
    Ghost,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StatusLevel {
    Info,
    Warning,
    Error,
    Success,
}
