//! Session scheduler — priority-based harness execution.
//!
//! Desktop sessions are first-class citizens. At most one session is
//! "foreground" (the user is interacting with it); the rest run in "background"
//! mode with a configurable token-budget throttle so the device stays responsive.
//!
//! Scheduling is simple: no DAG needed at this layer. The scheduler just tracks
//! priorities and provides admission control for new sessions.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};
use tokio::task::JoinHandle;
use crate::session::{AgentSession, SessionId, SessionPhase};

// ─── Priority ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SessionPriority {
    /// Background: non-interactive, throttled (e.g. ambient agent, sync task).
    Background = 0,
    /// Normal: interactive but not currently in focus.
    Normal     = 1,
    /// Foreground: the session the user is actively driving.
    Foreground = 2,
}

impl Default for SessionPriority {
    fn default() -> Self { Self::Normal }
}

// ─── ScheduledSession ─────────────────────────────────────────────────────────

struct ScheduledSession {
    session:  Arc<AgentSession>,
    priority: SessionPriority,
    _handle:  Option<JoinHandle<()>>,
}

// ─── Scheduler ────────────────────────────────────────────────────────────────

pub struct SessionScheduler {
    sessions: Mutex<HashMap<SessionId, ScheduledSession>>,
    /// Max concurrent foreground sessions (usually 1 on desktop).
    max_foreground: usize,
    /// Max concurrent total sessions.
    max_sessions:   usize,
}

impl SessionScheduler {
    pub fn new() -> Self {
        Self {
            sessions:       Mutex::new(HashMap::new()),
            max_foreground: 1,
            max_sessions:   8,
        }
    }

    /// Register a session. Returns `false` if the session cap is reached.
    pub fn add(
        &self,
        session:  Arc<AgentSession>,
        priority: SessionPriority,
        handle:   Option<JoinHandle<()>>,
    ) -> bool {
        let mut map = self.sessions.lock().unwrap();
        if map.len() >= self.max_sessions {
            return false;
        }
        map.insert(session.id.clone(), ScheduledSession { session, priority, _handle: handle });
        true
    }

    /// Remove a session when it completes or is killed.
    pub fn remove(&self, id: &SessionId) -> Option<Arc<AgentSession>> {
        self.sessions.lock().unwrap()
            .remove(id)
            .map(|s| s.session)
    }

    /// Promote a session to foreground. Demotes the previous foreground to Normal.
    pub fn set_foreground(&self, id: &SessionId) {
        let mut map = self.sessions.lock().unwrap();
        // Demote current foreground(s)
        for s in map.values_mut() {
            if s.priority == SessionPriority::Foreground {
                s.priority = SessionPriority::Normal;
            }
        }
        if let Some(s) = map.get_mut(id) {
            s.priority = SessionPriority::Foreground;
        }
    }

    /// Demote a session to background.
    pub fn set_background(&self, id: &SessionId) {
        if let Some(s) = self.sessions.lock().unwrap().get_mut(id) {
            s.priority = SessionPriority::Background;
        }
    }

    pub fn foreground(&self) -> Option<Arc<AgentSession>> {
        self.sessions.lock().unwrap()
            .values()
            .filter(|s| s.priority == SessionPriority::Foreground)
            .max_by_key(|s| s.priority)
            .map(|s| s.session.clone())
    }

    /// All running sessions sorted by priority descending.
    pub fn running_by_priority(&self) -> Vec<(Arc<AgentSession>, SessionPriority)> {
        let mut v: Vec<_> = self.sessions.lock().unwrap()
            .values()
            .filter(|s| s.session.phase() == SessionPhase::Running)
            .map(|s| (s.session.clone(), s.priority))
            .collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v
    }

    pub fn session_count(&self) -> usize {
        self.sessions.lock().unwrap().len()
    }

    pub fn at_capacity(&self) -> bool {
        self.sessions.lock().unwrap().len() >= self.max_sessions
    }
}

impl Default for SessionScheduler {
    fn default() -> Self { Self::new() }
}
