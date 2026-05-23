//! MCP client — used by artifact processes to connect to agentd's MCP hub.
//! Also used by kiki-sdk so developers can write artifacts in Rust.
//!
//! Connection: connects to /run/kiki/mcp.sock (or the path in kiki.toml).
//! After connecting, sends initialize, then handles incoming tool call
//! requests by dispatching to registered handler closures.

use std::{collections::HashMap, sync::Arc};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::UnixStream,
};
use tracing::{debug, error, info};
use kiki_core::error::{Error, Result};
use crate::hub::McpToolSpec;

// ─── Tool handler ─────────────────────────────────────────────────────────────

/// A boxed async function that handles a single tool call.
pub type ToolHandler = Arc<
    dyn Fn(serde_json::Value) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<serde_json::Value>> + Send>>
    + Send + Sync
>;

// ─── McpClient ────────────────────────────────────────────────────────────────

pub struct McpClient {
    socket_path:  String,
    artifact_id:  String,
    version:      String,
    tools:        Vec<McpToolSpec>,
    handlers:     HashMap<String, ToolHandler>,
}

impl McpClient {
    pub fn new(
        socket_path: impl Into<String>,
        artifact_id: impl Into<String>,
        version:     impl Into<String>,
    ) -> Self {
        Self {
            socket_path:  socket_path.into(),
            artifact_id:  artifact_id.into(),
            version:      version.into(),
            tools:        Vec::new(),
            handlers:     HashMap::new(),
        }
    }

    /// Register a tool with a handler closure.
    pub fn register_tool<F, Fut>(
        &mut self,
        spec:    McpToolSpec,
        handler: F,
    ) where
        F:   Fn(serde_json::Value) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<serde_json::Value>> + Send + 'static,
    {
        let name = spec.name.clone();
        self.tools.push(spec);
        self.handlers.insert(name, Arc::new(move |v| Box::pin(handler(v))));
    }

    /// Connect to the MCP hub, send initialize, and start handling tool calls.
    /// This runs until the connection is closed.
    pub async fn run(self) -> Result<()> {
        let stream = UnixStream::connect(&self.socket_path).await
            .map_err(|e| Error::Io(format!("MCP connect failed: {e}")))?;

        info!(artifact = %self.artifact_id, socket = %self.socket_path, "MCP client connected");

        let (read_half, mut write_half) = tokio::io::split(stream);
        let mut lines = BufReader::new(read_half).lines();

        // ── Send initialize ───────────────────────────────────────────────────
        let init = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "artifactId": self.artifact_id,
                "version":    self.version,
                "tools":      self.tools,
            }
        });
        send_line(&mut write_half, &init).await?;

        // ── Wait for initialize response ──────────────────────────────────────
        let resp_line = lines.next_line().await
            .map_err(|e| Error::Io(e.to_string()))?
            .ok_or_else(|| Error::Transport("connection closed during handshake".into()))?;
        let _resp: serde_json::Value = serde_json::from_str(&resp_line)
            .map_err(|e| Error::Parse(e.to_string()))?;

        let handlers = Arc::new(self.handlers);

        // ── Tool call dispatch loop ───────────────────────────────────────────
        loop {
            let line = match lines.next_line().await {
                Ok(Some(l)) => l,
                Ok(None) | Err(_) => {
                    info!(artifact = %self.artifact_id, "MCP connection closed");
                    break;
                }
            };

            let msg: serde_json::Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(e) => { error!(%e, "invalid MCP message"); continue; }
            };

            if msg["method"].as_str() != Some("tools/call") { continue; }

            let req_id    = msg["id"].clone();
            let tool_name = msg["params"]["name"].as_str().unwrap_or("").to_string();
            let arguments = msg["params"]["arguments"].clone();

            let handlers2    = handlers.clone();
            let _reply_result = {
                // We need to send replies back — clone write_half access.
                // For simplicity, use a channel to serialize writes.
                // In practice, artifact handlers are usually quick.
                //
                // Since we can't clone the write half directly, we'll await
                // the handler synchronously per message (fine for most artifacts).
                let reply = if let Some(handler) = handlers2.get(&tool_name) {
                    match handler(arguments).await {
                        Ok(result) => serde_json::json!({
                            "jsonrpc": "2.0", "id": req_id, "result": result
                        }),
                        Err(e) => serde_json::json!({
                            "jsonrpc": "2.0", "id": req_id,
                            "error": { "code": -32000, "message": e.to_string() }
                        }),
                    }
                } else {
                    serde_json::json!({
                        "jsonrpc": "2.0", "id": req_id,
                        "error": { "code": -32601, "message": format!("unknown tool: {tool_name}") }
                    })
                };

                debug!(tool = %tool_name, "tool call handled");
                if let Err(e) = send_line(&mut write_half, &reply).await {
                    error!(%e, "failed to send tool reply");
                    break;
                }
            };
        }

        Ok(())
    }
}

async fn send_line<W: AsyncWriteExt + Unpin>(
    w:   &mut W,
    msg: &serde_json::Value,
) -> Result<()> {
    let mut line = serde_json::to_string(msg).map_err(|e| Error::Parse(e.to_string()))?;
    line.push('\n');
    w.write_all(line.as_bytes()).await.map_err(|e| Error::Io(e.to_string()))
}
