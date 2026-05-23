//! McpHub — the nerve center for all tool communication.
//!
//! Every installed Kiki artifact that declares tools connects here as an MCP server
//! over a Unix domain socket at /run/kiki/mcp.sock. agentd is the MCP host.
//!
//! The hub:
//! 1. Accepts incoming MCP server connections (artifacts register their tools)
//! 2. Routes tool calls from the harness to the correct registered server
//! 3. Aggregates tool specs so the harness can pass them to the LLM
//! 4. Handles reconnection when artifact servers restart

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{debug, info, warn};
use kiki_core::{
    error::{Error, Result},
    tool::{Tool, ToolOutput, ToolRegistry},
};

// ─── MCP message types ────────────────────────────────────────────────────────

/// JSON-RPC 2.0 message (minimal, only what we need).
#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRpc {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id:      Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub method:  Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params:  Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result:  Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error:   Option<JsonRpcError>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code:    i32,
    pub message: String,
}

// ─── Registered server ────────────────────────────────────────────────────────

/// A registered MCP server (one per artifact that has tools).
#[derive(Clone)]
pub struct RegisteredServer {
    pub artifact_id: String,
    pub version:     String,
    /// Tools this server exports.
    pub tools:       Vec<McpToolSpec>,
    /// Channel to send tool-call requests to the server's handler task.
    pub call_tx:     tokio::sync::mpsc::Sender<ToolCallRequest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolSpec {
    pub name:         String,
    pub description:  String,
    pub input_schema: Value,
}

/// A tool call routed to a server.
pub struct ToolCallRequest {
    pub tool_name: String,
    pub input:     Value,
    pub reply_tx:  tokio::sync::oneshot::Sender<Result<Value>>,
}

// ─── McpHub ───────────────────────────────────────────────────────────────────

pub struct McpHub {
    servers: Arc<Mutex<HashMap<String, RegisteredServer>>>,
}

impl McpHub {
    pub fn new() -> Self {
        Self { servers: Arc::new(Mutex::new(HashMap::new())) }
    }

    /// Register a server (called when an artifact connects and declares its tools).
    pub fn register(&self, server: RegisteredServer) {
        info!(artifact = %server.artifact_id, tools = server.tools.len(), "MCP server registered");
        self.servers.lock().unwrap().insert(server.artifact_id.clone(), server);
    }

    /// Unregister a server (called when an artifact disconnects).
    pub fn unregister(&self, artifact_id: &str) {
        warn!(artifact = %artifact_id, "MCP server unregistered");
        self.servers.lock().unwrap().remove(artifact_id);
    }

    /// All tool specs from all registered servers.
    pub fn all_tools(&self) -> Vec<McpToolSpec> {
        self.servers.lock().unwrap()
            .values()
            .flat_map(|s| s.tools.clone())
            .collect()
    }

    /// Route a tool call to the correct server. Returns the raw JSON result.
    pub async fn call(&self, tool_name: &str, input: Value) -> Result<Value> {
        let server = self.servers.lock().unwrap()
            .values()
            .find(|s| s.tools.iter().any(|t| t.name == tool_name))
            .cloned();

        let server = server.ok_or_else(|| Error::ToolNotFound(tool_name.into()))?;

        debug!(tool = %tool_name, artifact = %server.artifact_id, "routing tool call");

        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        server.call_tx.send(ToolCallRequest {
            tool_name: tool_name.to_string(),
            input,
            reply_tx,
        }).await.map_err(|_| Error::Transport("MCP server channel closed".into()))?;

        reply_rx.await
            .map_err(|_| Error::Transport("MCP server dropped reply channel".into()))?
    }

    /// Build a `ToolRegistry` snapshot from all currently registered tools.
    /// The registry is static — rebuild when a server registers/unregisters.
    pub fn build_registry(&self) -> ToolRegistry {
        let mut registry = ToolRegistry::new();
        let servers = self.servers.lock().unwrap();

        for (_, server) in servers.iter() {
            for spec in &server.tools {
                let call_tx   = server.call_tx.clone();
                let tool_name = spec.name.clone();
                let desc      = spec.description.clone();
                let schema    = spec.input_schema.clone();

                registry.register(McpProxyTool {
                    name:        tool_name,
                    description: desc,
                    schema,
                    call_tx,
                });
            }
        }
        registry
    }

    pub fn server_count(&self) -> usize {
        self.servers.lock().unwrap().len()
    }
}

impl Default for McpHub {
    fn default() -> Self { Self::new() }
}

// ─── McpProxyTool ─────────────────────────────────────────────────────────────

/// A `Tool` impl that proxies calls through the McpHub channel.
struct McpProxyTool {
    name:        String,
    description: String,
    schema:      Value,
    call_tx:     tokio::sync::mpsc::Sender<ToolCallRequest>,
}

#[async_trait::async_trait]
impl Tool for McpProxyTool {
    fn name(&self) -> &str       { &self.name }
    fn description(&self) -> &str { &self.description }
    fn input_schema(&self) -> &serde_json::Value { &self.schema }

    async fn call(&self, input: Value) -> Result<ToolOutput> {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        self.call_tx.send(ToolCallRequest {
            tool_name: self.name.clone(),
            input,
            reply_tx,
        }).await.map_err(|_| Error::Transport("MCP proxy channel closed".into()))?;

        let result = reply_rx.await
            .map_err(|_| Error::Transport("MCP proxy reply channel dropped".into()))??;

        Ok(ToolOutput::ok(result))
    }
}
