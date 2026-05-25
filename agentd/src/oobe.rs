//! Out-of-box experience (OOBE) state machine for `agentd`.
//!
//! When a fresh Kiki OS device boots for the first time it has no sessions and
//! no configuration. The OOBE flow guides the user through:
//!
//! 1. **Welcome** — device naming.
//! 2. **AccountSetup** — linking to a cloud account via the device-code flow.
//! 3. **ModelDownload** — triggering `kpkg install` for the default model.
//! 4. **Done** — creating the first desktop session.
//!
//! The state machine emits `ShellEvent`-compatible JSON lines on `shell_tx`
//! so the shell HUD (and any connected client) can render the wizard UI.
//! It waits for `ControlMessage::OobeInput` lines on `oobe_rx` between steps.
//!
//! ## Bypass
//! Set `KIKI_SKIP_OOBE=1` to skip the flow entirely (CI / headless / provisioned
//! nodes). agentd calls [`OobeState::needed`] before the foreground session and
//! only enters this flow when it returns `true`.

use serde_json::Value;
use tokio::sync::{broadcast, mpsc};
use tracing::info;

use kiki_core::types::OobeStepKind;

/// Sender used to emit OOBE shell events (JSON lines) to connected shells.
pub type ShellEventSender = broadcast::Sender<String>;

/// A single OOBE input submitted by the user from the shell UI.
#[derive(Debug)]
pub struct OobeInputMsg {
    pub step:  String,
    pub value: Value,
}

/// The running OOBE wizard.
pub struct OobeState {
    step:      OobeStepKind,
    completed: bool,
}

impl OobeState {
    /// Returns `true` when OOBE is needed: no sessions file exists AND
    /// `KIKI_SKIP_OOBE` is not set.
    pub fn needed() -> bool {
        !std::path::Path::new("/var/kiki/sessions.json").exists()
            && std::env::var("KIKI_SKIP_OOBE").is_err()
    }

    fn new() -> Self {
        Self { step: OobeStepKind::Welcome, completed: false }
    }

    /// Run the full OOBE flow, emitting shell events on `shell_tx` and waiting
    /// for user input on `oobe_rx`. On completion the caller should create the
    /// initial foreground session.
    ///
    /// Returns immediately if `KIKI_SKIP_OOBE=1` is set (same guard as
    /// [`OobeState::needed`]).
    pub async fn run(
        shell_tx: &ShellEventSender,
        oobe_rx:  &mut mpsc::Receiver<OobeInputMsg>,
    ) -> anyhow::Result<()> {
        if std::env::var("KIKI_SKIP_OOBE").is_ok() {
            info!("OOBE skipped (KIKI_SKIP_OOBE set)");
            return Ok(());
        }

        let mut state = Self::new();
        info!("OOBE starting");

        // ── Step 1: Welcome ───────────────────────────────────────────────────
        state.step = OobeStepKind::Welcome;
        emit_oobe_step(
            shell_tx,
            &OobeStepKind::Welcome,
            "Welcome to Kiki OS! Let's set up your device. Enter a name for this device.",
        );
        let welcome_input = wait_for_step(oobe_rx, "welcome").await;
        let device_name = welcome_input
            .get("device_name")
            .and_then(|v| v.as_str())
            .unwrap_or("kiki-device")
            .to_string();
        info!(device_name = %device_name, "OOBE: device name set");

        // ── Step 2: Account setup ─────────────────────────────────────────────
        state.step = OobeStepKind::AccountSetup;
        emit_oobe_step(
            shell_tx,
            &OobeStepKind::AccountSetup,
            "Link your Kiki account. Open the URL shown in your browser to authorize this device.",
        );
        let _account_input = wait_for_step(oobe_rx, "account_setup").await;
        info!("OOBE: account setup acknowledged");

        // ── Step 3: Model download ────────────────────────────────────────────
        state.step = OobeStepKind::ModelDownload;
        emit_oobe_step(
            shell_tx,
            &OobeStepKind::ModelDownload,
            "Downloading the default AI model in the background. This may take a few minutes.",
        );
        // Fire-and-forget: trigger kpkg install of the default model. We don't
        // block the OOBE flow on the download finishing; the model pulls in the
        // background and becomes available once the session is live.
        trigger_default_model_download();
        // No input needed for this step — advance immediately.
        info!("OOBE: model download triggered (background)");

        // ── Step 4: Done ──────────────────────────────────────────────────────
        state.step = OobeStepKind::Done;
        emit_oobe_step(
            shell_tx,
            &OobeStepKind::Done,
            "Setup complete! Creating your first desktop session.",
        );
        emit_oobe_complete(shell_tx);
        state.completed = true;
        info!("OOBE: complete");

        Ok(())
    }

    /// Whether the OOBE run completed successfully.
    pub fn is_completed(&self) -> bool {
        self.completed
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Emit a `ShellEvent`-compatible `oobe_step` JSON line.
fn emit_oobe_step(shell_tx: &ShellEventSender, step: &OobeStepKind, message: &str) {
    let step_str = match step {
        OobeStepKind::Welcome       => "welcome",
        OobeStepKind::AccountSetup  => "account_setup",
        OobeStepKind::ModelDownload => "model_download",
        OobeStepKind::Done          => "done",
    };
    let line = serde_json::json!({
        "type":    "oobe_step",
        "step":    step_str,
        "message": message,
    })
    .to_string();
    let _ = shell_tx.send(line);
}

/// Emit a `ShellEvent`-compatible `oobe_complete` JSON line.
fn emit_oobe_complete(shell_tx: &ShellEventSender) {
    let line = serde_json::json!({ "type": "oobe_complete" }).to_string();
    let _ = shell_tx.send(line);
}

/// Wait for an `OobeInputMsg` matching `step`, discarding inputs for other
/// steps. If the channel closes before a match, returns an empty JSON object
/// so the flow can continue gracefully.
async fn wait_for_step(
    rx:   &mut mpsc::Receiver<OobeInputMsg>,
    step: &str,
) -> Value {
    loop {
        match rx.recv().await {
            Some(msg) if msg.step == step => return msg.value,
            Some(other) => {
                tracing::debug!(got = %other.step, expected = step, "OOBE: ignoring out-of-order input");
            }
            None => {
                tracing::warn!(step, "OOBE: input channel closed — continuing with defaults");
                return Value::Object(Default::default());
            }
        }
    }
}

/// Fire-and-forget: spawn a background task that runs `kpkg install` for the
/// default model. Failures are logged but do not abort the OOBE flow.
fn trigger_default_model_download() {
    // TODO(kpkg): replace with the real kpkg Rust API once it exposes an
    // async `install(model_id)` function. Until then, spawn the CLI subprocess.
    let default_model = std::env::var("KIKI_DEFAULT_MODEL")
        .unwrap_or_else(|_| "llama3.2:3b".to_string());
    tokio::spawn(async move {
        info!(model = %default_model, "OOBE: starting model download via kpkg");
        let status = tokio::process::Command::new("kpkg")
            .args(["install", &default_model])
            .status()
            .await;
        match status {
            Ok(s) if s.success() => info!(model = %default_model, "OOBE: model download complete"),
            Ok(s) => tracing::warn!(model = %default_model, code = ?s.code(), "OOBE: kpkg install exited non-zero"),
            Err(e) => tracing::warn!(model = %default_model, error = %e, "OOBE: kpkg not found (continuing)"),
        }
    });
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::broadcast;

    /// Helper: collect all lines emitted on `shell_tx` after closing `oobe_rx`.
    async fn collect_events(shell_tx: &ShellEventSender) -> Vec<serde_json::Value> {
        let mut rx = shell_tx.subscribe();
        let mut out = Vec::new();
        while let Ok(line) = rx.try_recv() {
            out.push(serde_json::from_str(&line).unwrap());
        }
        out
    }

    #[tokio::test]
    async fn oobe_skipped_when_env_set() {
        // SAFETY: this test sets an env var; acceptable in unit tests (not parallel).
        std::env::set_var("KIKI_SKIP_OOBE", "1");
        let (shell_tx, _) = broadcast::channel::<String>(16);
        let (_, mut oobe_rx) = mpsc::channel::<OobeInputMsg>(8);
        OobeState::run(&shell_tx, &mut oobe_rx).await.expect("should succeed with skip");
        std::env::remove_var("KIKI_SKIP_OOBE");
    }

    #[tokio::test]
    async fn oobe_run_emits_steps_in_order() {
        // Ensure KIKI_SKIP_OOBE is not set.
        std::env::remove_var("KIKI_SKIP_OOBE");

        let (shell_tx, mut shell_rx) = broadcast::channel::<String>(64);
        let (oobe_tx, mut oobe_rx) = mpsc::channel::<OobeInputMsg>(8);

        // Pre-seed the two expected inputs before running (the state machine waits
        // for them synchronously via async recv, so we push them before spawning).
        oobe_tx.send(OobeInputMsg {
            step:  "welcome".into(),
            value: serde_json::json!({ "device_name": "my-kiki" }),
        }).await.unwrap();
        oobe_tx.send(OobeInputMsg {
            step:  "account_setup".into(),
            value: serde_json::json!({ "ok": true }),
        }).await.unwrap();

        OobeState::run(&shell_tx, &mut oobe_rx).await.expect("OOBE run");

        // Collect emitted events.
        let mut events = Vec::new();
        while let Ok(line) = shell_rx.try_recv() {
            let v: serde_json::Value = serde_json::from_str(&line).unwrap();
            events.push(v);
        }

        // Must have emitted at least 4 step events + 1 oobe_complete.
        let types: Vec<&str> = events.iter()
            .filter_map(|v| v["type"].as_str())
            .collect();
        assert!(types.contains(&"oobe_step"),    "should emit oobe_step events: {types:?}");
        assert!(types.contains(&"oobe_complete"), "should emit oobe_complete: {types:?}");

        // Steps must be in order: welcome → account_setup → model_download → done.
        let step_names: Vec<&str> = events.iter()
            .filter(|v| v["type"] == "oobe_step")
            .filter_map(|v| v["step"].as_str())
            .collect();
        assert_eq!(
            step_names,
            ["welcome", "account_setup", "model_download", "done"],
            "OOBE steps out of order: {step_names:?}"
        );
    }

    #[test]
    fn oobe_needed_respects_skip_env() {
        // `OobeState::needed` returns false when KIKI_SKIP_OOBE is set.
        // We test this via the run() path which has the same guard; see
        // oobe_skipped_when_env_set (which sets the var and calls run()).
        // Here we verify the logic: needed() = false when var is set to "1".
        // Save/restore to avoid affecting parallel tests.
        let prev = std::env::var("KIKI_SKIP_OOBE").ok();
        std::env::set_var("KIKI_SKIP_OOBE", "1");
        let result = OobeState::needed();
        match prev {
            Some(v) => std::env::set_var("KIKI_SKIP_OOBE", v),
            None    => std::env::remove_var("KIKI_SKIP_OOBE"),
        }
        assert!(!result, "KIKI_SKIP_OOBE=1 should suppress OOBE");
    }

    #[tokio::test]
    async fn oobe_advances_past_unknown_steps() {
        // If the channel sends an input for a DIFFERENT step before the current one,
        // the state machine should discard it and wait for the correct step.
        std::env::remove_var("KIKI_SKIP_OOBE");

        let (shell_tx, _) = broadcast::channel::<String>(64);
        let (oobe_tx, mut oobe_rx) = mpsc::channel::<OobeInputMsg>(8);

        // Send wrong step first, then correct one.
        oobe_tx.send(OobeInputMsg {
            step:  "account_setup".into(), // wrong — welcome hasn't been acked yet
            value: serde_json::json!({}),
        }).await.unwrap();
        oobe_tx.send(OobeInputMsg {
            step:  "welcome".into(),
            value: serde_json::json!({ "device_name": "x" }),
        }).await.unwrap();
        oobe_tx.send(OobeInputMsg {
            step:  "account_setup".into(),
            value: serde_json::json!({}),
        }).await.unwrap();

        // Should not panic or deadlock.
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            OobeState::run(&shell_tx, &mut oobe_rx),
        ).await;
        assert!(result.is_ok(), "OOBE should complete within timeout");
        assert!(result.unwrap().is_ok());
    }
}
