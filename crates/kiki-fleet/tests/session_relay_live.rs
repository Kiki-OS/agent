//! Live integration: the device → SessionDO state relay against the deployed
//! fleet worker.
//!
//! This closes the audit-identified gap "agentd never connects to the SessionDO
//! as role=device, so agent state never reaches the cloud". It drives the real
//! [`connect_device`] publisher (the device leg, intentionally unauthenticated)
//! and asserts a published `state_patch` + heartbeat are accepted.
//!
//! The cloud-persistence read (`GET /state`) and the app-client relay leg now
//! require an authenticated user session — and the client leg additionally
//! requires ownership of a real registered node — so they can't be exercised
//! with an ad-hoc session id here. Those legs are covered by the cloud dataflow
//! gate (`scripts/load/dataflow.mjs`) and the web E2E instead.
//!
//! Gated behind `KIKI_CLOUD_TEST=1`. Run with:
//! ```sh
//! KIKI_CLOUD_TEST=1 KIKI_FLEET_URL=https://fleet-preview.kiki-os.com \
//!   cargo test -p kiki-fleet --test session_relay_live -- --nocapture
//! ```

use kiki_fleet::{connect_device, StatePatch};
use serde_json::json;

fn enabled() -> bool {
    std::env::var("KIKI_CLOUD_TEST").as_deref() == Ok("1")
}
fn fleet_url() -> String {
    std::env::var("KIKI_FLEET_URL").unwrap_or_else(|_| "https://fleet-preview.kiki-os.com".into())
}

#[tokio::test]
async fn device_state_reaches_cloud() {
    if !enabled() {
        eprintln!("skipped: set KIKI_CLOUD_TEST=1 to run");
        return;
    }
    let base = fleet_url();
    let nonce = format!(
        "{}-{}",
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis(),
        std::process::id(),
    );
    let session_id = format!("relay-{nonce}");

    // Device connects as role=device (unauthenticated by contract) and publishes
    // its live state; the DO accepts and persists it. A failure here means the
    // device leg of the relay is broken.
    let (device, _inbound) = connect_device(&base, &session_id, None).await.expect("device connect");
    device.publish_patch(&StatePatch {
        phase:        Some("active".into()),
        agent_status: Some(json!("thinking")),
        context:      Some(json!({ "progress": 42, "task": "Editing video" })),
        ..Default::default()
    }).await.expect("publish patch");
    device.heartbeat().await.expect("heartbeat");
    device.publish_patch(&StatePatch::context(json!({ "progress": 99 }))).await.expect("publish 2");

    // Give the DO a moment to persist before the connection drops.
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    println!("[relay] device leg published to session {session_id}");
}
