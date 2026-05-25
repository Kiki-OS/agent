//! Lock-screen state machine for `agentd`.
//!
//! The lock handler operates at the agentd orchestration layer, between the
//! control socket and the individual harness(es). It maintains a per-session
//! lock state and emits `ShellEvent`-compatible JSON lines so the shell can
//! render the lock screen overlay.
//!
//! ## Protocol
//! - `LockSession   { session_id }` → park the session, emit `session_locked`.
//! - `UnlockSession { session_id, pin? }` → validate pin (accept any / None for
//!   now), resume the session, emit `session_unlocked`.
//!
//! ## Lock timeout
//! Reading `KIKI_LOCK_TIMEOUT_SECS` (default 0 = disabled). When > 0 a timeout
//! task is spawned that fires `LockSession` after N seconds without a
//! `UserInput` from any harness. Inactivity is tracked via the
//! [`InactivityTracker`] that the control socket calls on every `UserInput`.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::sync::{broadcast, mpsc};
use tracing::{info, warn};

use kiki_core::types::ControlMessage;

/// Sender used to emit lock-screen shell events (JSON lines) to connected shells.
pub type ShellEventSender = broadcast::Sender<String>;

// ─── Lock state ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LockStatus {
    Unlocked,
    Locked,
}

/// Per-session lock state maintained by the agentd orchestrator.
#[derive(Debug)]
pub struct SessionLockState {
    pub status: LockStatus,
}

impl SessionLockState {
    pub fn new() -> Self {
        Self { status: LockStatus::Unlocked }
    }

    pub fn is_locked(&self) -> bool {
        self.status == LockStatus::Locked
    }
}

// ─── LockManager ─────────────────────────────────────────────────────────────

/// Tracks lock state for all active sessions and emits shell events.
pub struct LockManager {
    sessions:  Arc<Mutex<HashMap<String, SessionLockState>>>,
    shell_tx:  ShellEventSender,
}

impl LockManager {
    pub fn new(shell_tx: ShellEventSender) -> Self {
        Self {
            sessions: Arc::new(Mutex::new(HashMap::new())),
            shell_tx,
        }
    }

    /// Lock a session: mark it locked and emit `session_locked`.
    ///
    /// The caller (agentd main) is responsible for parking the harness —
    /// this method only updates lock state and notifies the shell.
    pub fn lock(&self, session_id: &str) {
        {
            let mut map = self.sessions.lock().unwrap();
            let entry = map.entry(session_id.to_string()).or_insert_with(SessionLockState::new);
            entry.status = LockStatus::Locked;
        }
        info!(session = %session_id, "session locked");
        emit_session_locked(&self.shell_tx, session_id);
    }

    /// Unlock a session: validate `pin` (accept any / None for now), mark it
    /// unlocked, and emit `session_unlocked`.
    ///
    /// Returns `true` if the unlock was accepted (pin valid or absent), `false`
    /// if the pin was rejected (future: hardware-backed check). Currently all
    /// pins and `None` are accepted.
    pub fn unlock(&self, session_id: &str, pin: Option<&str>) -> bool {
        // Future: validate pin against the stored hash / hardware keystore.
        // For now, accept any input.
        let _ = pin;

        {
            let mut map = self.sessions.lock().unwrap();
            let entry = map.entry(session_id.to_string()).or_insert_with(SessionLockState::new);
            entry.status = LockStatus::Unlocked;
        }
        info!(session = %session_id, "session unlocked");
        emit_session_unlocked(&self.shell_tx, session_id);
        true
    }

    /// Whether a session is currently locked.
    pub fn is_locked(&self, session_id: &str) -> bool {
        self.sessions
            .lock()
            .unwrap()
            .get(session_id)
            .map(|s| s.is_locked())
            .unwrap_or(false)
    }
}

// ─── Inactivity tracker ───────────────────────────────────────────────────────

/// Tracks the last `UserInput` activity timestamp so the lock timeout task can
/// detect idle periods. Call [`InactivityTracker::touch`] on every `UserInput`.
#[derive(Clone)]
pub struct InactivityTracker {
    last_activity: Arc<Mutex<Instant>>,
}

impl InactivityTracker {
    pub fn new() -> Self {
        Self { last_activity: Arc::new(Mutex::new(Instant::now())) }
    }

    /// Record that the user was active now.
    pub fn touch(&self) {
        *self.last_activity.lock().unwrap() = Instant::now();
    }

    /// Seconds since the last user activity.
    pub fn idle_secs(&self) -> u64 {
        self.last_activity.lock().unwrap().elapsed().as_secs()
    }
}

// ─── Lock timeout task ────────────────────────────────────────────────────────

/// Spawn a background task that sends `ControlMessage::LockSession` for
/// `session_id` after `timeout_secs` of inactivity. The task re-arms itself on
/// each unlock so it fires again after the next idle period.
///
/// `KIKI_LOCK_TIMEOUT_SECS=0` (or unset) disables the timer entirely.
pub fn spawn_lock_timeout(
    session_id:   String,
    timeout_secs: u64,
    tracker:      InactivityTracker,
    ctrl_tx:      mpsc::Sender<ControlMessage>,
) {
    if timeout_secs == 0 {
        return;
    }
    tokio::spawn(async move {
        let period = Duration::from_secs(1);
        loop {
            tokio::time::sleep(period).await;
            if tracker.idle_secs() >= timeout_secs {
                info!(session = %session_id, timeout = timeout_secs, "lock timeout: locking session");
                let _ = ctrl_tx
                    .send(ControlMessage::LockSession { session_id: session_id.clone() })
                    .await;
                // Back off until the session is presumably unlocked.
                tokio::time::sleep(Duration::from_secs(timeout_secs)).await;
                // Re-arm: reset the tracker so the next timeout starts fresh.
                tracker.touch();
            }
        }
    });
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Read `KIKI_LOCK_TIMEOUT_SECS` (default 0 = disabled).
pub fn lock_timeout_secs() -> u64 {
    std::env::var("KIKI_LOCK_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

fn emit_session_locked(shell_tx: &ShellEventSender, session_id: &str) {
    let line = serde_json::json!({
        "type":       "session_locked",
        "session_id": session_id,
    })
    .to_string();
    let _ = shell_tx.send(line);
}

fn emit_session_unlocked(shell_tx: &ShellEventSender, session_id: &str) {
    let line = serde_json::json!({
        "type":       "session_unlocked",
        "session_id": session_id,
    })
    .to_string();
    let _ = shell_tx.send(line);
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::broadcast;

    fn make_manager() -> (LockManager, broadcast::Receiver<String>) {
        let (tx, rx) = broadcast::channel::<String>(16);
        (LockManager::new(tx), rx)
    }

    #[test]
    fn lock_and_unlock_emits_events() {
        let (mgr, mut rx) = make_manager();

        assert!(!mgr.is_locked("s1"));

        mgr.lock("s1");
        assert!(mgr.is_locked("s1"));

        let line = rx.try_recv().expect("session_locked event");
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["type"], "session_locked");
        assert_eq!(v["session_id"], "s1");

        let accepted = mgr.unlock("s1", None);
        assert!(accepted, "unlock with no pin should be accepted");
        assert!(!mgr.is_locked("s1"));

        let line = rx.try_recv().expect("session_unlocked event");
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["type"], "session_unlocked");
        assert_eq!(v["session_id"], "s1");
    }

    #[test]
    fn unlock_with_pin_is_accepted() {
        let (mgr, _rx) = make_manager();
        mgr.lock("s2");
        assert!(mgr.unlock("s2", Some("1234")));
        assert!(!mgr.is_locked("s2"));
    }

    #[test]
    fn unlock_unlocked_session_is_noop_and_accepted() {
        let (mgr, _rx) = make_manager();
        // Unlocking a session that was never locked should succeed gracefully.
        assert!(mgr.unlock("unknown", None));
    }

    #[test]
    fn lock_multiple_sessions_independently() {
        let (mgr, _rx) = make_manager();
        mgr.lock("sA");
        mgr.lock("sB");
        assert!(mgr.is_locked("sA"));
        assert!(mgr.is_locked("sB"));

        mgr.unlock("sA", None);
        assert!(!mgr.is_locked("sA"));
        assert!(mgr.is_locked("sB"), "sB should still be locked after sA is unlocked");
    }

    #[test]
    fn inactivity_tracker_touch_resets_idle() {
        let tracker = InactivityTracker::new();
        // Right after creation, idle time should be ~0.
        assert!(tracker.idle_secs() < 2, "should not be idle immediately after creation");
        tracker.touch();
        assert!(tracker.idle_secs() < 2, "touching should reset the idle clock");
    }

    #[test]
    fn lock_timeout_secs_env_behavior() {
        // Run both assertions in one test so the env-var writes are atomic from
        // the perspective of the parallel test runner (no cross-test race).
        std::env::remove_var("KIKI_LOCK_TIMEOUT_SECS");
        assert_eq!(lock_timeout_secs(), 0, "unset → default 0");
        std::env::set_var("KIKI_LOCK_TIMEOUT_SECS", "300");
        assert_eq!(lock_timeout_secs(), 300, "set → reads the value");
        std::env::remove_var("KIKI_LOCK_TIMEOUT_SECS");
    }

    #[tokio::test]
    async fn lock_roundtrip_wire_shape() {
        // Verify the shell-event JSON wire shape for session_locked/unlocked
        // matches what the DE protocol expects (cross-repo wire contract).
        let (tx, mut rx) = broadcast::channel::<String>(8);
        let mgr = LockManager::new(tx);

        mgr.lock("sess-x");
        let locked_line = rx.try_recv().unwrap();
        let v: serde_json::Value = serde_json::from_str(&locked_line).unwrap();
        assert_eq!(
            v,
            serde_json::json!({ "type": "session_locked", "session_id": "sess-x" }),
            "session_locked wire shape mismatch"
        );

        mgr.unlock("sess-x", Some("0000"));
        let unlocked_line = rx.try_recv().unwrap();
        let v: serde_json::Value = serde_json::from_str(&unlocked_line).unwrap();
        assert_eq!(
            v,
            serde_json::json!({ "type": "session_unlocked", "session_id": "sess-x" }),
            "session_unlocked wire shape mismatch"
        );
    }
}
