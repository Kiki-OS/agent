//! Device → SessionDO state publisher.
//!
//! This is the device leg of the agent ↔ cloud ↔ app relay. `agentd` connects
//! to the fleet worker's per-session Durable Object as `role=device` over a
//! WebSocket and pushes `state_patch` messages so connected app/web clients
//! render the agent's live state. It also receives `tool_call` requests routed
//! from clients (remote control) and replies with `tool_result`.
//!
//! Wire protocol (mirrors `kiki-cloud/packages/types` and `workers/fleet/src/session-do.ts`):
//!
//! ```text
//! device → DO : {"type":"state_patch","patch":{...}}
//!               {"type":"heartbeat","ts":<ms>}
//!               {"type":"interrupt","kind":..,"id":..,"message":..}
//!               {"type":"tool_result","request_id":..,"result":..,"error":..}
//! DO → device: {"type":"tool_call","request_id":..,"tool":..,"input":{..}}
//!               {"type":"interrupt_response","interrupt_id":..,"resolution":..}
//! ```

use std::time::{SystemTime, UNIX_EPOCH};

use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;

use kiki_core::error::{Error, Result};

/// A subset of `SessionDOState` the device can patch. All fields optional —
/// only present fields are sent (the DO merges them).
#[derive(Debug, Clone, Default, Serialize)]
pub struct StatePatch {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase:         Option<String>, // "active" | "parked" | "migrating"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_status:  Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scenario:      Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub control_mode:  Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub component_ref: Option<String>,
    /// Mutable context map UI components observe (track_info, progress, …).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context:       Option<Value>,
}

impl StatePatch {
    pub fn agent_status(status: impl Into<Value>) -> Self {
        Self { agent_status: Some(status.into()), ..Default::default() }
    }
    pub fn context(ctx: impl Into<Value>) -> Self {
        Self { context: Some(ctx.into()), ..Default::default() }
    }
}

/// A message routed from an app/web client down to this device.
#[derive(Debug, Clone)]
pub enum DeviceInbound {
    /// A client wants the device to invoke a tool. Reply with
    /// [`SessionPublisher::tool_result`] using the same `request_id`.
    ToolCall { request_id: String, tool: String, input: Value },
    /// A client answered an interrupt the device raised.
    InterruptResponse { interrupt_id: String, resolution: Value },
    /// A client asked the device to move this session to the cloud. The device
    /// freezes it and sends its MigrationBundle to node `cloud-<session_id>`,
    /// where a CloudSessionDO-launched agentd resumes it.
    MigrateToCloud { session_id: String },
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum InboundWire {
    #[serde(rename = "tool_call")]
    ToolCall { request_id: String, tool: String, #[serde(default)] input: Value },
    #[serde(rename = "interrupt_response")]
    InterruptResponse { interrupt_id: String, #[serde(default)] resolution: Value },
    #[serde(rename = "migrate_to_cloud")]
    MigrateToCloud { session_id: String },
    #[serde(other)]
    Other,
}

/// Handle for publishing to the SessionDO. Cloneable; sends are queued to a
/// single writer task, so concurrent callers are safe.
#[derive(Clone)]
pub struct SessionPublisher {
    out_tx: mpsc::Sender<Message>,
}

impl SessionPublisher {
    async fn send(&self, v: Value) -> Result<()> {
        self.out_tx
            .send(Message::Text(v.to_string()))
            .await
            .map_err(|_| Error::Fleet("session ws writer closed".into()))
    }

    /// Push a partial state update; the DO merges it and broadcasts to clients.
    pub async fn publish_patch(&self, patch: &StatePatch) -> Result<()> {
        self.send(json!({ "type": "state_patch", "patch": patch })).await
    }

    /// Liveness ping; the DO records `last_heartbeat`.
    pub async fn heartbeat(&self) -> Result<()> {
        self.send(json!({ "type": "heartbeat", "ts": now_ms() })).await
    }

    /// Raise an interrupt (decision/confirmation/attention/info) to clients.
    pub async fn interrupt(&self, kind: &str, id: &str, message: &str, context: Value) -> Result<()> {
        self.send(json!({
            "type": "interrupt", "kind": kind, "id": id, "message": message, "context": context,
        })).await
    }

    /// Reply to a [`DeviceInbound::ToolCall`] with its result.
    pub async fn tool_result(&self, request_id: &str, result: Value, error: Option<String>) -> Result<()> {
        self.send(json!({
            "type": "tool_result", "request_id": request_id, "result": result, "error": error,
        })).await
    }
}

/// Connect to the fleet SessionDO as `role=device`.
///
/// `base_url` is the fleet worker origin (`https://fleet.kiki-os.com`); it is
/// rewritten to `wss://` for the upgrade. Returns the publisher plus a receiver
/// of inbound client→device messages.
pub async fn connect_device(
    base_url:   &str,
    session_id: &str,
) -> Result<(SessionPublisher, mpsc::Receiver<DeviceInbound>)> {
    let ws_url = format!(
        "{}/v1/fleet/sessions/{}/ws?role=device",
        base_url.replace("https://", "wss://").replace("http://", "ws://"),
        session_id,
    );

    let (ws, _resp) = tokio_tungstenite::connect_async(&ws_url)
        .await
        .map_err(|e| Error::Fleet(format!("session ws connect: {e}")))?;
    let (mut sink, mut stream) = ws.split();

    let (out_tx, mut out_rx) = mpsc::channel::<Message>(64);
    let (in_tx,  in_rx)      = mpsc::channel::<DeviceInbound>(64);

    // Writer task: drain outgoing queue → WS sink.
    tokio::spawn(async move {
        while let Some(msg) = out_rx.recv().await {
            if sink.send(msg).await.is_err() { break; }
        }
        let _ = sink.close().await;
    });

    // Reader task: WS stream → typed inbound channel; auto-pong.
    let pong_tx = out_tx.clone();
    tokio::spawn(async move {
        while let Some(frame) = stream.next().await {
            let msg = match frame {
                Ok(m) => m,
                Err(e) => { tracing::warn!(error = %e, "session ws stream error"); break; }
            };
            match msg {
                Message::Text(t) => {
                    match serde_json::from_str::<InboundWire>(&t) {
                        Ok(InboundWire::ToolCall { request_id, tool, input }) => {
                            if in_tx.send(DeviceInbound::ToolCall { request_id, tool, input }).await.is_err() { break; }
                        }
                        Ok(InboundWire::InterruptResponse { interrupt_id, resolution }) => {
                            if in_tx.send(DeviceInbound::InterruptResponse { interrupt_id, resolution }).await.is_err() { break; }
                        }
                        Ok(InboundWire::MigrateToCloud { session_id }) => {
                            if in_tx.send(DeviceInbound::MigrateToCloud { session_id }).await.is_err() { break; }
                        }
                        Ok(InboundWire::Other) => {}
                        Err(e) => tracing::warn!(error = %e, "session ws parse"),
                    }
                }
                Message::Ping(p) => { let _ = pong_tx.send(Message::Pong(p)).await; }
                Message::Close(_) => break,
                _ => {}
            }
        }
    });

    Ok((SessionPublisher { out_tx }, in_rx))
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64
}
