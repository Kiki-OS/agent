use std::sync::{Arc, Mutex};
use tokio::sync::oneshot;
use kiki_core::{
    context::{Context, ControlMode},
    state::{MigrationBundle, RuntimeSnapshot, StateBackend},
    error::{Error, Result},
};

pub type SessionId = String;

/// Phase of the session's PRA loop lifecycle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionPhase {
    /// PRA loop is running normally.
    Running,
    /// Freeze requested; waiting for the current step to complete.
    Freezing,
    /// PRA loop is paused. State is serializable. Can migrate or resume.
    Frozen,
    /// OSTree push in progress; bundle sent to target.
    Migrating,
    /// Migration complete. Session no longer owned by this node.
    Migrated { target_node: String },
    /// Session ended normally.
    Completed,
    /// Session ended with an error.
    Failed { reason: String },
}

/// Agent-side session: owns the PRA loop lifecycle and its migration contract.
///
/// This is the agent's view of a session — distinct from kiki-wm's Session
/// (which owns the Wayland surface). They share the same `id` as the join key.
pub struct AgentSession {
    pub id:           SessionId,
    pub label:        String,
    pub agent_id:     String,
    phase:            Arc<Mutex<SessionPhase>>,
    freeze_tx:        Arc<Mutex<Option<oneshot::Sender<()>>>>,
    pub state:        Arc<dyn StateBackend>,
    pub control_mode: ControlMode,
    pub scenario:     Option<String>,
}

impl AgentSession {
    pub fn new(
        id:    impl Into<SessionId>,
        label: impl Into<String>,
        agent_id: impl Into<String>,
        state: Arc<dyn StateBackend>,
    ) -> Self {
        Self {
            id:           id.into(),
            label:        label.into(),
            agent_id:     agent_id.into(),
            phase:        Arc::new(Mutex::new(SessionPhase::Running)),
            freeze_tx:    Arc::new(Mutex::new(None)),
            state,
            control_mode: ControlMode::AgentMode,
            scenario:     None,
        }
    }

    pub fn phase(&self) -> SessionPhase {
        self.phase.lock().unwrap().clone()
    }

    pub fn is_running(&self) -> bool {
        self.phase() == SessionPhase::Running
    }

    /// Request the PRA loop to freeze after its current step.
    /// Returns a receiver that fires when the freeze is complete.
    pub fn request_freeze(&self) -> oneshot::Receiver<()> {
        let (tx, rx) = oneshot::channel();
        *self.phase.lock().unwrap() = SessionPhase::Freezing;
        *self.freeze_tx.lock().unwrap() = Some(tx);
        rx
    }

    /// Called by the PRA loop itself at the end of a step when it detects
    /// phase == Freezing. Confirms the freeze and notifies any waiter.
    pub fn confirm_freeze(&self) {
        *self.phase.lock().unwrap() = SessionPhase::Frozen;
        if let Some(tx) = self.freeze_tx.lock().unwrap().take() {
            let _ = tx.send(());
        }
    }

    pub fn should_freeze(&self) -> bool {
        self.phase() == SessionPhase::Freezing
    }

    /// Build a MigrationBundle from the current frozen state.
    /// Must only be called when phase == Frozen.
    pub async fn build_bundle(&self, ctx: &Context) -> Result<MigrationBundle> {
        if self.phase() != SessionPhase::Frozen {
            return Err(Error::Migration(
                "build_bundle called on non-frozen session".into(),
            ));
        }
        let runtime = RuntimeSnapshot {
            agent_id:        self.agent_id.clone(),
            session_id:      self.id.clone(),
            step:            ctx.steps_taken() as u64,
            messages:        ctx.messages.clone(),
            interrupt_queue: ctx.interrupt_log.clone(),
            control_mode:    ctx.control_mode,
            session_label:   self.label.clone(),
            scenario:        self.scenario.clone(),
            layout:          ctx.layout,
            active_apps:     ctx.active_apps.clone(),
        };
        self.state.snapshot(runtime).await
    }

    /// Transition to Migrating and record the target node.
    pub fn begin_migration(&self, target_node: impl Into<String>) {
        let _ = target_node; // stored in Migrating variant when we set it
        *self.phase.lock().unwrap() = SessionPhase::Migrating;
    }

    pub fn complete_migration(&self, target_node: impl Into<String>) {
        *self.phase.lock().unwrap() = SessionPhase::Migrated {
            target_node: target_node.into(),
        };
    }

    pub fn resume(&self) {
        *self.phase.lock().unwrap() = SessionPhase::Running;
    }

    pub fn complete(&self) {
        *self.phase.lock().unwrap() = SessionPhase::Completed;
    }

    pub fn fail(&self, reason: impl Into<String>) {
        *self.phase.lock().unwrap() = SessionPhase::Failed { reason: reason.into() };
    }
}

/// Manages all AgentSessions on this node.
pub struct SessionManager {
    sessions: Arc<Mutex<std::collections::HashMap<SessionId, Arc<AgentSession>>>>,
}

impl SessionManager {
    pub fn new() -> Self {
        Self { sessions: Arc::new(Mutex::new(std::collections::HashMap::new())) }
    }

    pub fn add(&self, session: Arc<AgentSession>) {
        self.sessions.lock().unwrap().insert(session.id.clone(), session);
    }

    pub fn get(&self, id: &SessionId) -> Option<Arc<AgentSession>> {
        self.sessions.lock().unwrap().get(id).cloned()
    }

    pub fn remove(&self, id: &SessionId) -> Option<Arc<AgentSession>> {
        self.sessions.lock().unwrap().remove(id)
    }

    pub fn running(&self) -> Vec<Arc<AgentSession>> {
        self.sessions.lock().unwrap()
            .values()
            .filter(|s| s.is_running())
            .cloned()
            .collect()
    }

    pub fn all(&self) -> Vec<Arc<AgentSession>> {
        self.sessions.lock().unwrap().values().cloned().collect()
    }
}

impl Default for SessionManager {
    fn default() -> Self { Self::new() }
}
