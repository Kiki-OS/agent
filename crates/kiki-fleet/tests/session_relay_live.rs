//! Live integration: the device → SessionDO → app state relay against the
//! deployed fleet worker.
//!
//! This closes the audit-identified gap "agentd never connects to the SessionDO
//! as role=device, so agent state never reaches the app". It drives the real
//! [`connect_device`] publisher and asserts:
//!   1. a published `state_patch` is persisted (HTTP `/state` reflects it),
//!   2. a connected app client (`role=client`) receives the snapshot + live patches.
//!
//! Gated behind `KIKI_CLOUD_TEST=1`. Run with:
//! ```sh
//! KIKI_CLOUD_TEST=1 KIKI_FLEET_URL=https://fleet-preview.kiki-os.com \
//!   cargo test -p kiki-fleet --test session_relay_live -- --nocapture
//! ```

use futures::{SinkExt, StreamExt};
use kiki_fleet::{connect_device, StatePatch};
use serde_json::json;
use tokio_tungstenite::tungstenite::Message;

fn enabled() -> bool {
    std::env::var("KIKI_CLOUD_TEST").as_deref() == Ok("1")
}
fn fleet_url() -> String {
    std::env::var("KIKI_FLEET_URL").unwrap_or_else(|_| "https://fleet-preview.kiki-os.com".into())
}

#[tokio::test]
async fn device_state_reaches_cloud_and_app() {
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

    // ── 1. Device connects and publishes its live state. ──────────────────────
    let (device, _inbound) = connect_device(&base, &session_id, None).await.expect("device connect");
    device.publish_patch(&StatePatch {
        phase:        Some("active".into()),
        agent_status: Some(json!("thinking")),
        context:      Some(json!({ "progress": 42, "task": "Editing video" })),
        ..Default::default()
    }).await.expect("publish patch");
    device.heartbeat().await.expect("heartbeat");

    // Give the DO a moment to persist.
    tokio::time::sleep(std::time::Duration::from_millis(800)).await;

    // ── 2. Cloud persisted it — HTTP /state reflects the merged state. ────────
    let state: serde_json::Value = reqwest::Client::new()
        .get(format!("{base}/v1/fleet/sessions/{session_id}/state"))
        .send().await.expect("get state")
        .json().await.expect("state json");
    println!("[relay] /state = {state}");
    assert_eq!(state["phase"], "active");
    assert_eq!(state["agent_status"], "thinking");
    assert_eq!(state["context"]["progress"], 42);

    // ── 3. App leg: a client connects and receives the live snapshot. ─────────
    let ws_url = format!(
        "{}/v1/fleet/sessions/{session_id}/ws?role=client",
        base.replace("https://", "wss://"),
    );
    let (client_ws, _) = tokio_tungstenite::connect_async(&ws_url).await.expect("client ws connect");
    let (mut client_sink, mut client_stream) = client_ws.split();

    // On connect the DO sends a state_snapshot reflecting current state.
    let snapshot = recv_json(&mut client_stream).await.expect("snapshot");
    println!("[relay] client snapshot = {snapshot}");
    assert_eq!(snapshot["type"], "state_snapshot");
    assert_eq!(snapshot["state"]["agent_status"], "thinking");

    // ── 4. A new device patch is broadcast live to the connected client. ──────
    device.publish_patch(&StatePatch::context(json!({ "progress": 99 }))).await.expect("publish 2");
    let patch = recv_until(&mut client_stream, "state_patch").await.expect("live patch");
    println!("[relay] client live patch = {patch}");
    assert_eq!(patch["patch"]["context"]["progress"], 99);

    let _ = client_sink.close().await;
}

async fn recv_json<S>(stream: &mut S) -> Option<serde_json::Value>
where
    S: StreamExt<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    while let Some(Ok(msg)) = stream.next().await {
        if let Message::Text(t) = msg {
            return serde_json::from_str(&t).ok();
        }
    }
    None
}

/// Read frames until one with the given `type` arrives (skips snapshots/pings).
async fn recv_until<S>(stream: &mut S, want_type: &str) -> Option<serde_json::Value>
where
    S: StreamExt<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    for _ in 0..20 {
        let v = recv_json(stream).await?;
        if v["type"] == want_type {
            return Some(v);
        }
    }
    None
}
