//! Unix socket MCP server — artifacts connect here to register their tools.
//!
//! Protocol: newline-delimited JSON-RPC 2.0 over a Unix domain socket.
//! Each connected client is a Kiki artifact (app/component/durable) that
//! may declare tools, subscribe to events, and receive tool call requests.
//!
//! Handshake:
//!   client → {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"artifactId":"...","version":"...","tools":[...]}}
//!   server → {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"0.1.0"}}
//!   then: server routes calls, client responds with results

use std::{path::Path, sync::Arc};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::{UnixListener, UnixStream},
    sync::mpsc,
};
use tracing::{error, info, warn};
use crate::hub::{McpHub, McpToolSpec, RegisteredServer, ToolCallRequest};
use kiki_core::error::Result;

const PROTOCOL_VERSION: &str = "0.1.0";
const CHANNEL_CAP: usize = 32;

pub struct McpServer {
    hub:         Arc<McpHub>,
    socket_path: String,
}

impl McpServer {
    pub fn new(hub: Arc<McpHub>, socket_path: impl Into<String>) -> Self {
        Self { hub, socket_path: socket_path.into() }
    }

    /// Bind the socket and start accepting connections.
    /// Returns a `JoinHandle` — the server runs until the handle is dropped.
    pub async fn serve(self) -> Result<tokio::task::JoinHandle<()>> {
        let path = self.socket_path.clone();

        // Remove stale socket if present
        if Path::new(&path).exists() {
            let _ = std::fs::remove_file(&path);
        }

        let listener = UnixListener::bind(&path)
            .map_err(|e| kiki_core::error::Error::Io(e.to_string()))?;

        info!(socket = %path, "MCP server listening");

        let hub = self.hub.clone();
        let handle = tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((stream, _)) => {
                        let hub2 = hub.clone();
                        tokio::spawn(handle_connection(stream, hub2));
                    }
                    Err(e) => {
                        error!(error = %e, "MCP accept error");
                        break;
                    }
                }
            }
        });

        Ok(handle)
    }
}

async fn handle_connection(stream: UnixStream, hub: Arc<McpHub>) {
    let (read_half, mut write_half) = tokio::io::split(stream);
    let mut lines = BufReader::new(read_half).lines();

    // ── Handshake ─────────────────────────────────────────────────────────────
    let first_line = match lines.next_line().await {
        Ok(Some(l)) => l,
        _ => return,
    };

    let msg: serde_json::Value = match serde_json::from_str(&first_line) {
        Ok(v) => v,
        Err(e) => {
            warn!(error = %e, "invalid MCP init message");
            return;
        }
    };

    let method = msg["method"].as_str().unwrap_or("");
    if method != "initialize" {
        warn!(method, "expected initialize as first message");
        return;
    }

    let params  = &msg["params"];
    let art_id  = params["artifactId"].as_str().unwrap_or("unknown").to_string();
    let version = params["version"].as_str().unwrap_or("0.0.0").to_string();
    let tools: Vec<McpToolSpec> = serde_json::from_value(
        params["tools"].clone()
    ).unwrap_or_default();

    info!(artifact = %art_id, tools = tools.len(), "artifact connected");

    // Reply to initialize
    let id = msg["id"].clone();
    let reply = serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": { "protocolVersion": PROTOCOL_VERSION }
    });
    if let Err(e) = send_line(&mut write_half, &reply).await {
        error!(error = %e, "failed to send initialize reply");
        return;
    }

    // ── Register server ───────────────────────────────────────────────────────
    let (call_tx, mut call_rx) = mpsc::channel::<ToolCallRequest>(CHANNEL_CAP);
    hub.register(RegisteredServer {
        artifact_id: art_id.clone(),
        version,
        tools,
        call_tx,
    });

    // ── Dispatch loop — two concurrent directions ─────────────────────────────
    // Direction A: harness → artifact (forward tool call requests over socket)
    // Direction B: artifact → agentd (receive results)

    let mut pending: std::collections::HashMap<
        String,
        tokio::sync::oneshot::Sender<kiki_core::error::Result<serde_json::Value>>,
    > = std::collections::HashMap::new();

    let mut req_id: u64 = 0;

    loop {
        tokio::select! {
            // Incoming call request from harness
            req = call_rx.recv() => {
                let Some(req) = req else { break; };
                req_id += 1;
                let id_str = req_id.to_string();
                pending.insert(id_str.clone(), req.reply_tx);

                let call_msg = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id_str,
                    "method": "tools/call",
                    "params": { "name": req.tool_name, "arguments": req.input }
                });
                if let Err(e) = send_line(&mut write_half, &call_msg).await {
                    error!(error = %e, "failed to forward tool call");
                    break;
                }
            }

            // Incoming result/notification from artifact
            line = lines.next_line() => {
                match line {
                    Ok(Some(l)) => {
                        let Ok(msg): std::result::Result<serde_json::Value, _> = serde_json::from_str(&l) else { continue; };
                        let id_str = msg["id"].as_str().map(str::to_owned)
                            .or_else(|| msg["id"].as_u64().map(|n| n.to_string()));

                        if let Some(id_key) = id_str {
                            if let Some(tx) = pending.remove(&id_key) {
                                if msg["error"].is_null() {
                                    let _ = tx.send(Ok(msg["result"].clone()));
                                } else {
                                    let err_msg = msg["error"]["message"].as_str().unwrap_or("tool error").to_string();
                                    let _ = tx.send(Err(kiki_core::error::Error::ToolExecution(err_msg)));
                                }
                            }
                        }
                    }
                    Ok(None) | Err(_) => break,
                }
            }
        }
    }

    hub.unregister(&art_id);
}

async fn send_line<W: AsyncWriteExt + Unpin>(
    w:    &mut W,
    msg:  &serde_json::Value,
) -> std::io::Result<()> {
    let mut line = serde_json::to_string(msg).unwrap_or_default();
    line.push('\n');
    w.write_all(line.as_bytes()).await
}
