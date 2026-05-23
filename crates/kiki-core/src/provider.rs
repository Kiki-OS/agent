//! LLM provider abstraction — lives in kiki-core so Harness can reference it
//! without creating a circular dependency with kiki-provider.
//!
//! Design follows eikarna/hermes-rs: streaming-first, tool calls detected from
//! partial output as each closing delimiter arrives, not after full completion.
//! kiki-provider implements the concrete providers; this module owns the contract.

use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::pin::Pin;
use crate::error::Result;

// ─── Wire types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role { System, User, Assistant, Tool }

/// Multi-modal provider content block — the wire representation sent to the API.
/// Harness converts ConversationMessage → Vec<ProviderMessage> before each call.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProviderBlock {
    Text    { text: String },
    Image   { media_type: String, data: String },  // base64
    ToolUse { id: String, name: String, input: Value },
    ToolResult { tool_use_id: String, content: String, is_error: bool },
    Thinking   { thinking: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderMessage {
    pub role:    Role,
    pub content: Vec<ProviderBlock>,
}

impl ProviderMessage {
    pub fn system(text: impl Into<String>) -> Self {
        Self { role: Role::System, content: vec![ProviderBlock::Text { text: text.into() }] }
    }
    pub fn user_text(text: impl Into<String>) -> Self {
        Self { role: Role::User, content: vec![ProviderBlock::Text { text: text.into() }] }
    }
}

/// Schema for a tool exposed to the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name:         String,
    pub description:  String,
    pub input_schema: Value,
}

/// Parameters for an LLM completion call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model:          String,
    pub messages:       Vec<ProviderMessage>,
    pub tools:          Vec<ToolSpec>,
    pub max_tokens:     Option<u32>,
    pub temperature:    Option<f32>,
    /// Extended thinking token budget (Claude 3.7+). None = disabled.
    pub thinking_tokens: Option<u32>,
    /// System prompt injected as a top-level system field (Anthropic convention).
    pub system:          Option<String>,
}

// ─── Streaming chunks ─────────────────────────────────────────────────────────

/// A chunk emitted from a streaming LLM response.
///
/// hermes-rs adaptation: ToolCallStart/ToolCallDelta/ToolCallEnd are emitted
/// incrementally as the JSON input accumulates, enabling early dispatch.
#[derive(Debug, Clone)]
pub enum StreamChunk {
    /// Incremental text token.
    Text(String),

    /// Extended thinking token (Claude extended thinking).
    Thinking(String),

    /// A complete tool call, accumulated from streaming deltas.
    /// Emitted when the closing delimiter is detected in the stream.
    ToolCall {
        id:    String,
        name:  String,
        input: Value,
    },

    /// Stop signal — stream is complete.
    Done,
}

pub type CompletionStream = Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>;

// ─── Provider trait ───────────────────────────────────────────────────────────

/// Implemented by each LLM backend (Ollama, Anthropic, OpenAI, Modal).
/// The harness holds an `Arc<dyn LlmProvider>` and routes requests based on
/// the router policy in agentd.toml.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    fn name(&self) -> &str;
    fn supports_model(&self, model: &str) -> bool;

    /// Initiate a streaming completion. Returns a stream of chunks.
    /// The caller is responsible for accumulating chunks into AssistantTurn.
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionStream>;

    /// Count tokens in a request (used for context window management).
    /// Returns None if the provider doesn't support count APIs.
    async fn count_tokens(&self, request: &CompletionRequest) -> Option<u32> {
        let _ = request;
        None
    }
}
