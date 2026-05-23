//! Event bus — tokio broadcast channel for inter-artifact pub/sub.
//!
//! Any component (session harness, MCP server, fleet client) can publish an
//! `AgentEvent` or `BusEvent`. Subscribers receive all events from the moment
//! they subscribe. Slow subscribers see their channel lag (lagged error) and
//! are automatically re-subscribed at the latest message.

use std::sync::Arc;
use tokio::sync::broadcast;
use kiki_core::harness::AgentEvent;
use serde::{Deserialize, Serialize};

// ─── BusEvent ─────────────────────────────────────────────────────────────────

/// Events published on the global bus (superset of AgentEvent).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BusEvent {
    /// Agent turn event from a specific session.
    Agent { session_id: String, event: WrappedAgentEvent },
    /// A session started, ended, paused, or migrated.
    Session { session_id: String, transition: String },
    /// An artifact was installed or removed.
    Artifact { artifact_id: String, installed: bool },
    /// Control mode changed for a session.
    ModeChange { session_id: String, mode: String },
    /// Fleet connectivity event.
    Fleet { status: String, message: String },
}

/// JSON-serializable wrapper for AgentEvent (which has non-serializable fields in tests).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WrappedAgentEvent {
    pub kind:    String,
    pub payload: serde_json::Value,
}

impl From<AgentEvent> for WrappedAgentEvent {
    fn from(e: AgentEvent) -> Self {
        let (kind, payload) = match &e {
            AgentEvent::Thinking      { text }          => ("thinking",       serde_json::json!({ "text": text })),
            AgentEvent::Content       { text }          => ("content",        serde_json::json!({ "text": text })),
            AgentEvent::ToolStart     { name, input }   => ("tool_start",     serde_json::json!({ "name": name, "input": input })),
            AgentEvent::ToolComplete  { name, success }  => ("tool_complete",  serde_json::json!({ "name": name, "success": success })),
            AgentEvent::ModeChange    { mode }           => ("mode_change",    serde_json::json!({ "mode": format!("{mode:?}") })),
            AgentEvent::Checkpoint    { step, reason }  => ("checkpoint",     serde_json::json!({ "step": step, "reason": reason })),
            AgentEvent::Compacting    { dropped_turns } => ("compacting",     serde_json::json!({ "dropped_turns": dropped_turns })),
            AgentEvent::Healing       { attempt, error } => ("healing",       serde_json::json!({ "attempt": attempt, "error": error })),
            AgentEvent::Done          { session_id, steps } => ("done",       serde_json::json!({ "session_id": session_id, "steps": steps })),
            AgentEvent::Error         { error }         => ("error",          serde_json::json!({ "error": error })),
        };
        Self { kind: kind.into(), payload }
    }
}

// ─── EventBus ─────────────────────────────────────────────────────────────────

const BUS_CAPACITY: usize = 256;

#[derive(Clone)]
pub struct EventBus {
    tx: Arc<broadcast::Sender<BusEvent>>,
}

impl EventBus {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(BUS_CAPACITY);
        Self { tx: Arc::new(tx) }
    }

    /// Publish an event to all active subscribers.
    /// If there are no subscribers, this is a no-op (not an error).
    pub fn publish(&self, event: BusEvent) {
        let _ = self.tx.send(event);
    }

    /// Shorthand: publish an AgentEvent scoped to a session.
    pub fn publish_agent(&self, session_id: impl Into<String>, event: AgentEvent) {
        self.publish(BusEvent::Agent {
            session_id: session_id.into(),
            event:      event.into(),
        });
    }

    /// Subscribe to all future events. Returns a `broadcast::Receiver`.
    /// Lagged receivers are automatically reset to the latest message.
    pub fn subscribe(&self) -> broadcast::Receiver<BusEvent> {
        self.tx.subscribe()
    }

    /// Number of active subscribers.
    pub fn subscriber_count(&self) -> usize {
        self.tx.receiver_count()
    }
}

impl Default for EventBus {
    fn default() -> Self { Self::new() }
}
