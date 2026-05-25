//! MCP hub — Unix socket server, tool registry, streaming parser, plugin loader.
//!
//! Every installed Kiki artifact that declares tools connects here as an MCP server.
//! agentd is the MCP host. Communication is JSON-RPC 2.0 over a Unix domain socket
//! at /run/kiki/mcp.sock.
//!
//! Streaming parser adapted from eikarna/hermes-rs: detects tool_call closing tags
//! in partial LLM output and dispatches immediately, before the full response is
//! complete. Reduces perceived latency for multi-step plans on local models.

pub mod client;
pub mod hub;
pub mod loader;
pub mod parser;
pub mod server;

pub use client::McpClient;
pub use hub::{InstalledApp, McpHub, McpToolSpec, RegisteredServer, ToolCallRequest, ToolKind};
pub use kiki_schema::ArtifactManifest;
pub use loader::{scan_egress_allowlists, PluginLoader};
pub use parser::{ParsedChunk, ToolCallParser};
pub use server::McpServer;
