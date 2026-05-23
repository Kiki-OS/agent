//! Anthropic Messages API provider (claude-* models).
//!
//! Implements the Messages API with SSE streaming, tool use, extended thinking,
//! and vision. Wire format verified against the published streaming spec:
//! events carry both an `event:` name and a matching `type` field in their data;
//! we key on the `type` field.
//!
//! Stream shape:
//! - `message_start`                       → ignored (metadata)
//! - `content_block_start` (text|tool_use|thinking)
//!     - tool_use ⇒ record `{index → (id, name)}`, open an args buffer
//! - `content_block_delta`
//!     - `text_delta`        → [`StreamChunk::Text`]
//!     - `thinking_delta`    → [`StreamChunk::Thinking`]
//!     - `input_json_delta`  → append `partial_json` to the block's args buffer
//!     - `signature_delta`   → ignored (thinking-integrity signature)
//! - `content_block_stop`                   → if tool_use, parse args & emit [`StreamChunk::ToolCall`]
//! - `message_delta`                        → ignored (stop_reason / usage)
//! - `message_stop`                         → [`StreamChunk::Done`]
//! - `ping`                                 → ignored
//! - `error`                                → stream error

use std::collections::HashMap;

use async_trait::async_trait;
use futures::StreamExt;
use kiki_core::{
    error::{Error, Result},
    provider::{
        CompletionRequest, CompletionStream, LlmProvider,
        ProviderBlock, Role, StreamChunk,
    },
};
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::warn;

const DEFAULT_MAX_TOKENS: u32 = 4096;
const ANTHROPIC_VERSION:  &str = "2023-06-01";

#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    /// Base URL without trailing slash. Default: `https://api.anthropic.com/v1`.
    pub base_url: String,
    pub api_key:  String,
    /// Value for the `anthropic-version` header.
    pub version:  String,
}

impl Default for AnthropicConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.anthropic.com/v1".into(),
            api_key:  String::new(),
            version:  ANTHROPIC_VERSION.into(),
        }
    }
}

pub struct AnthropicProvider {
    cfg:    AnthropicConfig,
    client: Client,
}

impl AnthropicProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_config(AnthropicConfig { api_key: api_key.into(), ..Default::default() })
    }

    pub fn with_config(cfg: AnthropicConfig) -> Self {
        Self { cfg, client: Client::new() }
    }
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    fn name(&self) -> &str { "anthropic" }

    fn supports_model(&self, model: &str) -> bool {
        model.starts_with("claude-")
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionStream> {
        if request.model.is_empty() {
            return Err(Error::Provider("model id required".into()));
        }

        let body = build_request_body(&request);
        let url  = format!("{}/messages", self.cfg.base_url);

        let resp = self.client.post(&url)
            .header("x-api-key", &self.cfg.api_key)
            .header("anthropic-version", &self.cfg.version)
            .header("content-type", "application/json")
            .json(&body)
            .send().await
            .map_err(|e| Error::Provider(format!("anthropic request: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text   = resp.text().await.unwrap_or_default();
            return Err(Error::Provider(format!("anthropic {status}: {text}")));
        }

        let (tx, rx) = mpsc::channel::<Result<StreamChunk>>(64);
        tokio::spawn(stream_to_chunks(resp, tx));
        Ok(Box::pin(ReceiverStream::new(rx)))
    }
}

// ── Request building ─────────────────────────────────────────────────────────

fn build_request_body(request: &CompletionRequest) -> Value {
    // Anthropic has only user/assistant roles; system is a top-level field and
    // tool results live inside a *user* turn as tool_result blocks.
    let mut system = request.system.clone().unwrap_or_default();
    let mut messages: Vec<Value> = Vec::new();

    for m in &request.messages {
        match m.role {
            Role::System => {
                let text = collect_text(&m.content);
                if !system.is_empty() { system.push_str("\n\n"); }
                system.push_str(&text);
            }
            Role::User => messages.push(json!({
                "role": "user",
                "content": user_blocks(&m.content),
            })),
            Role::Assistant => messages.push(json!({
                "role": "assistant",
                "content": assistant_blocks(&m.content),
            })),
            Role::Tool => messages.push(json!({
                // tool results are delivered to the model in a user turn
                "role": "user",
                "content": tool_result_blocks(&m.content),
            })),
        }
    }

    let mut body = json!({
        "model":      request.model,
        "max_tokens": request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        "messages":   messages,
        "stream":     true,
    });
    let obj = body.as_object_mut().unwrap();

    if !system.is_empty() {
        obj.insert("system".into(), json!(system));
    }
    if !request.tools.is_empty() {
        let tools: Vec<Value> = request.tools.iter().map(|t| json!({
            "name":         t.name,
            "description":  t.description,
            "input_schema": t.input_schema,
        })).collect();
        obj.insert("tools".into(), json!(tools));
    }

    // Extended thinking: when enabled, temperature must be left at default (1.0),
    // and max_tokens must exceed the thinking budget.
    if let Some(budget) = request.thinking_tokens {
        let max = request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
        if budget > 0 && max > budget {
            obj.insert("thinking".into(), json!({
                "type": "enabled",
                "budget_tokens": budget,
            }));
        } else if let Some(temp) = request.temperature {
            obj.insert("temperature".into(), json!(temp));
        }
    } else if let Some(temp) = request.temperature {
        obj.insert("temperature".into(), json!(temp));
    }

    body
}

fn user_blocks(content: &[ProviderBlock]) -> Vec<Value> {
    content.iter().filter_map(|b| match b {
        ProviderBlock::Text { text } => Some(json!({ "type": "text", "text": text })),
        ProviderBlock::Image { media_type, data } => Some(json!({
            "type": "image",
            "source": { "type": "base64", "media_type": media_type, "data": data },
        })),
        // A user turn may also carry tool_result blocks (mixed turns).
        ProviderBlock::ToolResult { tool_use_id, content, is_error } => Some(json!({
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
            "is_error": is_error,
        })),
        _ => None,
    }).collect()
}

fn assistant_blocks(content: &[ProviderBlock]) -> Vec<Value> {
    // Thinking blocks are intentionally omitted: replaying them requires the
    // original `signature`, which the harness does not retain.
    content.iter().filter_map(|b| match b {
        ProviderBlock::Text { text } => Some(json!({ "type": "text", "text": text })),
        ProviderBlock::ToolUse { id, name, input } => Some(json!({
            "type": "tool_use", "id": id, "name": name, "input": input,
        })),
        _ => None,
    }).collect()
}

fn tool_result_blocks(content: &[ProviderBlock]) -> Vec<Value> {
    content.iter().filter_map(|b| match b {
        ProviderBlock::ToolResult { tool_use_id, content, is_error } => Some(json!({
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
            "is_error": is_error,
        })),
        _ => None,
    }).collect()
}

fn collect_text(blocks: &[ProviderBlock]) -> String {
    blocks.iter().filter_map(|b| match b {
        ProviderBlock::Text { text } => Some(text.as_str()),
        _ => None,
    }).collect::<Vec<_>>().join("")
}

// ── Streaming parse ──────────────────────────────────────────────────────────

#[derive(Deserialize)]
#[serde(tag = "type")]
enum Event {
    #[serde(rename = "message_start")]
    MessageStart {},
    #[serde(rename = "content_block_start")]
    ContentBlockStart { index: usize, content_block: ContentBlock },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { index: usize, delta: BlockDelta },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: usize },
    #[serde(rename = "message_delta")]
    MessageDelta {},
    #[serde(rename = "message_stop")]
    MessageStop {},
    #[serde(rename = "ping")]
    Ping {},
    #[serde(rename = "error")]
    Error { error: ApiError },
}

#[derive(Deserialize)]
struct ApiError { message: String, #[serde(rename = "type")] _kind: String }

#[derive(Deserialize)]
#[serde(tag = "type")]
enum ContentBlock {
    #[serde(rename = "text")]
    Text {},
    #[serde(rename = "thinking")]
    Thinking {},
    #[serde(rename = "tool_use")]
    ToolUse { id: String, name: String },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum BlockDelta {
    #[serde(rename = "text_delta")]
    Text { text: String },
    #[serde(rename = "thinking_delta")]
    Thinking { thinking: String },
    #[serde(rename = "input_json_delta")]
    InputJson { partial_json: String },
    #[serde(rename = "signature_delta")]
    Signature {},
    #[serde(other)]
    Other,
}

/// Per-stream state: which block indices are tool_use, plus their accumulators.
#[derive(Default)]
struct ToolBlocks {
    meta: HashMap<usize, (String, String)>, // index → (id, name)
    args: HashMap<usize, String>,           // index → accumulated partial_json
}

async fn stream_to_chunks(
    resp: reqwest::Response,
    tx:   mpsc::Sender<Result<StreamChunk>>,
) {
    let mut byte_stream = resp.bytes_stream();
    let mut buf    = String::new();
    let mut tools  = ToolBlocks::default();

    while let Some(chunk) = byte_stream.next().await {
        let bytes = match chunk {
            Ok(b)  => b,
            Err(e) => { let _ = tx.send(Err(Error::Provider(e.to_string()))).await; return; }
        };
        buf.push_str(&String::from_utf8_lossy(&bytes));

        // SSE lines; we only care about `data:` lines (the `type` field inside
        // tells us the event kind, so `event:` lines are redundant).
        while let Some(pos) = buf.find('\n') {
            let line = buf[..pos].trim().to_string();
            buf = buf[pos + 1..].to_string();
            if !line.starts_with("data:") { continue; }
            let data = line["data:".len()..].trim();
            if data.is_empty() { continue; }

            let event: Event = match serde_json::from_str(data) {
                Ok(e)  => e,
                Err(e) => { warn!("anthropic SSE parse: {e}: {data}"); continue; }
            };

            match event {
                Event::ContentBlockStart { index, content_block } => {
                    if let ContentBlock::ToolUse { id, name } = content_block {
                        tools.meta.insert(index, (id, name));
                        tools.args.entry(index).or_default();
                    }
                }
                Event::ContentBlockDelta { index, delta } => match delta {
                    BlockDelta::Text { text } => {
                        if !text.is_empty() {
                            let _ = tx.send(Ok(StreamChunk::Text(text))).await;
                        }
                    }
                    BlockDelta::Thinking { thinking } => {
                        if !thinking.is_empty() {
                            let _ = tx.send(Ok(StreamChunk::Thinking(thinking))).await;
                        }
                    }
                    BlockDelta::InputJson { partial_json } => {
                        tools.args.entry(index).or_default().push_str(&partial_json);
                    }
                    BlockDelta::Signature {} | BlockDelta::Other => {}
                },
                Event::ContentBlockStop { index } => {
                    if let Some((id, name)) = tools.meta.remove(&index) {
                        let raw = tools.args.remove(&index).unwrap_or_default();
                        let input = if raw.trim().is_empty() {
                            Value::Object(Default::default())
                        } else {
                            serde_json::from_str(&raw)
                                .unwrap_or(Value::Object(Default::default()))
                        };
                        let _ = tx.send(Ok(StreamChunk::ToolCall { id, name, input })).await;
                    }
                }
                Event::MessageStop {} => {
                    let _ = tx.send(Ok(StreamChunk::Done)).await;
                    return;
                }
                Event::Error { error } => {
                    let _ = tx.send(Err(Error::Provider(format!("anthropic stream error: {}", error.message)))).await;
                    return;
                }
                Event::MessageStart {} | Event::MessageDelta {} | Event::Ping {} => {}
            }
        }
    }
    // Stream closed without an explicit message_stop.
    let _ = tx.send(Ok(StreamChunk::Done)).await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiki_core::provider::{ProviderMessage, ToolSpec};

    fn req(messages: Vec<ProviderMessage>, tools: Vec<ToolSpec>) -> CompletionRequest {
        CompletionRequest {
            model: "claude-sonnet-4-6".into(),
            messages,
            tools,
            max_tokens: Some(1024),
            temperature: Some(0.0),
            thinking_tokens: None,
            system: Some("You are helpful.".into()),
        }
    }

    #[test]
    fn supports_only_claude() {
        let p = AnthropicProvider::new("sk-ant-test");
        assert!(p.supports_model("claude-sonnet-4-6"));
        assert!(!p.supports_model("gpt-4o"));
    }

    #[test]
    fn body_has_required_fields_and_system() {
        let body = build_request_body(&req(
            vec![ProviderMessage::user_text("Hi")],
            vec![],
        ));
        assert_eq!(body["model"], "claude-sonnet-4-6");
        assert_eq!(body["max_tokens"], 1024);
        assert_eq!(body["stream"], true);
        assert_eq!(body["system"], "You are helpful.");
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"][0]["type"], "text");
        assert_eq!(body["messages"][0]["content"][0]["text"], "Hi");
        // temperature present (no thinking)
        assert_eq!(body["temperature"], 0.0);
    }

    #[test]
    fn tool_result_goes_in_user_turn() {
        let assistant = ProviderMessage {
            role: Role::Assistant,
            content: vec![ProviderBlock::ToolUse {
                id: "toolu_1".into(),
                name: "get_weather".into(),
                input: json!({ "city": "Paris" }),
            }],
        };
        let tool = ProviderMessage {
            role: Role::Tool,
            content: vec![ProviderBlock::ToolResult {
                tool_use_id: "toolu_1".into(),
                content: "18C".into(),
                is_error: false,
            }],
        };
        let body = build_request_body(&req(vec![assistant, tool], vec![]));
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "assistant");
        assert_eq!(msgs[0]["content"][0]["type"], "tool_use");
        assert_eq!(msgs[0]["content"][0]["id"], "toolu_1");
        // tool result delivered as a USER turn with a tool_result block
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[1]["content"][0]["type"], "tool_result");
        assert_eq!(msgs[1]["content"][0]["tool_use_id"], "toolu_1");
        assert_eq!(msgs[1]["content"][0]["is_error"], false);
    }

    #[test]
    fn thinking_enabled_omits_temperature() {
        let mut r = req(vec![ProviderMessage::user_text("Think hard")], vec![]);
        r.thinking_tokens = Some(2048);
        r.max_tokens = Some(8192);
        let body = build_request_body(&r);
        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["thinking"]["budget_tokens"], 2048);
        assert!(body.get("temperature").is_none(), "temperature must be omitted when thinking is enabled");
    }

    #[test]
    fn tools_serialized_with_input_schema() {
        let tools = vec![ToolSpec {
            name: "search".into(),
            description: "Search the web".into(),
            input_schema: json!({ "type": "object", "properties": { "q": { "type": "string" } } }),
        }];
        let body = build_request_body(&req(vec![ProviderMessage::user_text("hi")], tools));
        assert_eq!(body["tools"][0]["name"], "search");
        assert_eq!(body["tools"][0]["input_schema"]["type"], "object");
    }
}
