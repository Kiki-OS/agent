//! OpenAI-compatible chat-completions provider.
//!
//! Targets the OpenAI `/v1/chat/completions` API and any server that speaks the
//! same wire format: Together, Fireworks, Groq, vLLM, llama.cpp's `llama-server`,
//! and **Ollama** (`http://localhost:11434/v1`). The `base_url` and `api_key` are
//! configurable so a single implementation covers cloud and on-device endpoints.
//!
//! Streaming: Server-Sent Events. Text deltas arrive on `choices[].delta.content`;
//! tool calls arrive incrementally on `choices[].delta.tool_calls[]` keyed by
//! `index`, with `id` and `function.name` on the first delta and `function.arguments`
//! accumulated across subsequent deltas. We flush accumulated tool calls when a
//! `finish_reason` of `tool_calls` (or end-of-stream) is observed.
//!
//! Message conversion is structured (not flattened): assistant turns carry their
//! `tool_calls` array and tool results become discrete `role: "tool"` messages
//! keyed by `tool_call_id`, so multi-turn tool-calling loops round-trip correctly.

use std::collections::HashMap;

use async_trait::async_trait;
use futures::StreamExt;
use kiki_core::{
    error::{Error, Result},
    provider::{
        CompletionRequest, CompletionStream, LlmProvider,
        ProviderBlock, ProviderMessage, Role, StreamChunk,
    },
};
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::warn;

/// Configuration for an OpenAI-compatible endpoint.
#[derive(Debug, Clone)]
pub struct OpenAiConfig {
    /// API base URL **without** a trailing slash, e.g. `https://api.openai.com/v1`
    /// or `http://localhost:11434/v1` for Ollama.
    pub base_url: String,
    /// Bearer token. May be empty for local servers that don't require auth.
    pub api_key:  String,
    /// Models this provider should claim via [`LlmProvider::supports_model`].
    /// `None` ⇒ accept the canonical OpenAI prefixes (`gpt-`, `o1`, `o3`).
    pub model_prefixes: Option<Vec<String>>,
}

impl Default for OpenAiConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.openai.com/v1".into(),
            api_key:  String::new(),
            model_prefixes: None,
        }
    }
}

pub struct OpenAiProvider {
    cfg:    OpenAiConfig,
    client: Client,
}

impl OpenAiProvider {
    pub fn new(cfg: OpenAiConfig) -> Self {
        Self { cfg, client: Client::new() }
    }

    /// Convenience constructor preserved for the original API key signature.
    pub fn with_api_key(api_key: impl Into<String>) -> Self {
        Self::new(OpenAiConfig { api_key: api_key.into(), ..Default::default() })
    }

    /// Point the provider at a local Ollama server.
    pub fn ollama() -> Self {
        Self::new(OpenAiConfig {
            base_url: "http://localhost:11434/v1".into(),
            api_key:  String::new(),
            model_prefixes: Some(vec![]), // supports any model id ollama has pulled
        })
    }
}

#[async_trait]
impl LlmProvider for OpenAiProvider {
    fn name(&self) -> &str { "openai" }

    /// Remote unless the endpoint is loopback (e.g. a local Ollama / llama-server).
    fn is_remote(&self) -> bool {
        !(self.cfg.base_url.contains("://localhost") || self.cfg.base_url.contains("://127.0.0.1"))
    }

    fn supports_model(&self, model: &str) -> bool {
        match &self.cfg.model_prefixes {
            // Empty list ⇒ accept anything (used for local/Ollama endpoints).
            Some(prefixes) if prefixes.is_empty() => true,
            Some(prefixes) => prefixes.iter().any(|p| model.starts_with(p)),
            None => model.starts_with("gpt-") || model.starts_with("o1") || model.starts_with("o3"),
        }
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionStream> {
        if request.model.is_empty() {
            return Err(Error::Provider("model id required".into()));
        }

        let body = build_request_body(&request);
        let url  = format!("{}/chat/completions", self.cfg.base_url);

        let mut req = self.client.post(&url).json(&body);
        if !self.cfg.api_key.is_empty() {
            req = req.bearer_auth(&self.cfg.api_key);
        }

        let resp = req.send().await
            .map_err(|e| Error::Provider(format!("openai request: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text   = resp.text().await.unwrap_or_default();
            return Err(Error::Provider(format!("openai {status}: {text}")));
        }

        let (tx, rx) = mpsc::channel::<Result<StreamChunk>>(64);
        tokio::spawn(stream_to_chunks(resp, tx));
        Ok(Box::pin(ReceiverStream::new(rx)))
    }
}

// ── Request building ─────────────────────────────────────────────────────────

fn build_request_body(request: &CompletionRequest) -> Value {
    let mut messages: Vec<Value> = Vec::new();

    if let Some(sys) = &request.system {
        messages.push(json!({ "role": "system", "content": sys }));
    }
    for m in &request.messages {
        append_message(&mut messages, m);
    }

    let mut body = json!({
        "model":    request.model,
        "messages": messages,
        "stream":   true,
    });
    let obj = body.as_object_mut().unwrap();

    if !request.tools.is_empty() {
        let tools: Vec<Value> = request.tools.iter().map(|t| json!({
            "type": "function",
            "function": {
                "name":        t.name,
                "description": t.description,
                "parameters":  t.input_schema,
            }
        })).collect();
        obj.insert("tools".into(), json!(tools));
    }
    if let Some(mt) = request.max_tokens {
        obj.insert("max_tokens".into(), json!(mt));
    }
    if let Some(temp) = request.temperature {
        obj.insert("temperature".into(), json!(temp));
    }
    body
}

/// Convert one [`ProviderMessage`] into one-or-more OpenAI wire messages.
///
/// Tool results expand into discrete `role: "tool"` messages (OpenAI requires one
/// message per `tool_call_id`); everything else maps to a single message.
fn append_message(out: &mut Vec<Value>, m: &ProviderMessage) {
    match m.role {
        Role::System => {
            out.push(json!({ "role": "system", "content": collect_text(&m.content) }));
        }
        Role::User => {
            // Vision: if any image blocks are present, emit the array content form.
            if m.content.iter().any(|b| matches!(b, ProviderBlock::Image { .. })) {
                let parts: Vec<Value> = m.content.iter().filter_map(|b| match b {
                    ProviderBlock::Text { text } =>
                        Some(json!({ "type": "text", "text": text })),
                    ProviderBlock::Image { media_type, data } =>
                        Some(json!({
                            "type": "image_url",
                            "image_url": { "url": format!("data:{media_type};base64,{data}") }
                        })),
                    _ => None,
                }).collect();
                out.push(json!({ "role": "user", "content": parts }));
            } else {
                out.push(json!({ "role": "user", "content": collect_text(&m.content) }));
            }
        }
        Role::Assistant => {
            let text = collect_text(&m.content);
            let tool_calls: Vec<Value> = m.content.iter().filter_map(|b| match b {
                ProviderBlock::ToolUse { id, name, input } => Some(json!({
                    "id":   id,
                    "type": "function",
                    "function": {
                        "name":      name,
                        "arguments": serde_json::to_string(input).unwrap_or_else(|_| "{}".into()),
                    }
                })),
                _ => None,
            }).collect();

            let mut msg = json!({ "role": "assistant" });
            let obj = msg.as_object_mut().unwrap();
            // content must be present; null is allowed when only tool calls exist.
            if text.is_empty() {
                obj.insert("content".into(), Value::Null);
            } else {
                obj.insert("content".into(), json!(text));
            }
            if !tool_calls.is_empty() {
                obj.insert("tool_calls".into(), json!(tool_calls));
            }
            out.push(msg);
        }
        Role::Tool => {
            for b in &m.content {
                if let ProviderBlock::ToolResult { tool_use_id, content, .. } = b {
                    out.push(json!({
                        "role":         "tool",
                        "tool_call_id": tool_use_id,
                        "content":      content,
                    }));
                }
            }
        }
    }
}

fn collect_text(blocks: &[ProviderBlock]) -> String {
    blocks.iter().filter_map(|b| match b {
        ProviderBlock::Text { text } => Some(text.as_str()),
        _ => None,
    }).collect::<Vec<_>>().join("")
}

// ── Streaming parse ──────────────────────────────────────────────────────────

#[derive(Deserialize, Default)]
struct Delta {
    content:    Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Deserialize)]
struct ToolCallDelta {
    #[serde(default)]
    index:    usize,
    id:       Option<String>,
    function: Option<FunctionDelta>,
}

#[derive(Deserialize, Default)]
struct FunctionDelta {
    name:      Option<String>,
    arguments: Option<String>,
}

#[derive(Deserialize)]
struct Choice {
    #[serde(default)]
    delta:         Delta,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct ChatChunk {
    #[serde(default)]
    choices: Vec<Choice>,
}

/// Accumulator for in-flight tool calls, keyed by stream `index`.
#[derive(Default)]
struct ToolAccum {
    ids:   HashMap<usize, String>,
    names: HashMap<usize, String>,
    args:  HashMap<usize, String>,
}

impl ToolAccum {
    fn apply(&mut self, deltas: Vec<ToolCallDelta>) {
        for tc in deltas {
            if let Some(id) = tc.id { self.ids.insert(tc.index, id); }
            if let Some(func) = tc.function {
                if let Some(name) = func.name { self.names.insert(tc.index, name); }
                if let Some(a) = func.arguments {
                    self.args.entry(tc.index).or_default().push_str(&a);
                }
            }
        }
    }

    fn is_empty(&self) -> bool { self.names.is_empty() }

    async fn flush(&mut self, tx: &mpsc::Sender<Result<StreamChunk>>) {
        let mut indices: Vec<usize> = self.names.keys().copied().collect();
        indices.sort_unstable();
        for idx in indices {
            let id   = self.ids.get(&idx).cloned().unwrap_or_else(|| format!("call_{idx}"));
            let name = self.names[&idx].clone();
            let raw  = self.args.get(&idx).map(String::as_str).unwrap_or("{}");
            let input = serde_json::from_str(raw)
                .unwrap_or(Value::Object(Default::default()));
            let _ = tx.send(Ok(StreamChunk::ToolCall { id, name, input })).await;
        }
        self.ids.clear();
        self.names.clear();
        self.args.clear();
    }
}

async fn stream_to_chunks(
    resp: reqwest::Response,
    tx:   mpsc::Sender<Result<StreamChunk>>,
) {
    let mut byte_stream = resp.bytes_stream();
    let mut buf   = String::new();
    let mut accum = ToolAccum::default();

    while let Some(chunk) = byte_stream.next().await {
        let bytes = match chunk {
            Ok(b) => b,
            Err(e) => { let _ = tx.send(Err(Error::Provider(e.to_string()))).await; return; }
        };
        buf.push_str(&String::from_utf8_lossy(&bytes));

        // SSE frames are newline-delimited; events end on a blank line but
        // OpenAI emits one `data:` per line so per-line parsing is sufficient.
        while let Some(pos) = buf.find('\n') {
            let line = buf[..pos].trim().to_string();
            buf = buf[pos + 1..].to_string();
            if line.is_empty() || !line.starts_with("data:") { continue; }
            let data = line["data:".len()..].trim();
            if data == "[DONE]" {
                if !accum.is_empty() { accum.flush(&tx).await; }
                let _ = tx.send(Ok(StreamChunk::Done)).await;
                return;
            }
            let parsed: ChatChunk = match serde_json::from_str(data) {
                Ok(c)  => c,
                Err(e) => { warn!("openai SSE parse: {e}: {data}"); continue; }
            };
            for choice in parsed.choices {
                if let Some(text) = choice.delta.content {
                    if !text.is_empty() {
                        let _ = tx.send(Ok(StreamChunk::Text(text))).await;
                    }
                }
                if let Some(tc) = choice.delta.tool_calls {
                    accum.apply(tc);
                }
                if choice.finish_reason.as_deref() == Some("tool_calls") {
                    accum.flush(&tx).await;
                }
            }
        }
    }
    // Stream ended without an explicit [DONE] (some servers omit it).
    if !accum.is_empty() { accum.flush(&tx).await; }
    let _ = tx.send(Ok(StreamChunk::Done)).await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiki_core::provider::{ProviderMessage, ToolSpec};

    fn req(messages: Vec<ProviderMessage>, tools: Vec<ToolSpec>) -> CompletionRequest {
        CompletionRequest {
            model: "gpt-4o".into(),
            messages,
            tools,
            max_tokens: Some(256),
            temperature: Some(0.0),
            thinking_tokens: None,
            system: Some("You are helpful.".into()),
        }
    }

    #[test]
    fn supports_model_default_prefixes() {
        let p = OpenAiProvider::with_api_key("sk-test");
        assert!(p.supports_model("gpt-4o"));
        assert!(p.supports_model("o1-preview"));
        assert!(!p.supports_model("claude-3-5-sonnet"));
    }

    #[test]
    fn ollama_supports_any_model() {
        let p = OpenAiProvider::ollama();
        assert!(p.supports_model("qwen2.5:1.5b"));
        assert!(p.supports_model("granite4.1:3b"));
    }

    #[test]
    fn body_includes_system_tools_and_params() {
        let tools = vec![ToolSpec {
            name: "get_weather".into(),
            description: "Get weather for a city".into(),
            input_schema: json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"],
            }),
        }];
        let body = build_request_body(&req(
            vec![ProviderMessage::user_text("Weather in Paris?")],
            tools,
        ));
        assert_eq!(body["model"], "gpt-4o");
        assert_eq!(body["stream"], true);
        assert_eq!(body["max_tokens"], 256);
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][1]["role"], "user");
        assert_eq!(body["messages"][1]["content"], "Weather in Paris?");
        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["function"]["name"], "get_weather");
    }

    #[test]
    fn assistant_tool_call_roundtrips_to_wire() {
        let assistant = ProviderMessage {
            role: Role::Assistant,
            content: vec![ProviderBlock::ToolUse {
                id: "call_abc".into(),
                name: "get_weather".into(),
                input: json!({ "city": "Paris" }),
            }],
        };
        let tool_result = ProviderMessage {
            role: Role::Tool,
            content: vec![ProviderBlock::ToolResult {
                tool_use_id: "call_abc".into(),
                content: "18C, clear".into(),
                is_error: false,
            }],
        };
        let body = build_request_body(&req(vec![assistant, tool_result], vec![]));
        let msgs = body["messages"].as_array().unwrap();
        // [system, assistant(tool_calls), tool]
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["content"], Value::Null);
        assert_eq!(msgs[1]["tool_calls"][0]["id"], "call_abc");
        assert_eq!(msgs[1]["tool_calls"][0]["function"]["name"], "get_weather");
        // arguments must be a JSON-encoded string, not an object
        assert_eq!(msgs[1]["tool_calls"][0]["function"]["arguments"], "{\"city\":\"Paris\"}");
        assert_eq!(msgs[2]["role"], "tool");
        assert_eq!(msgs[2]["tool_call_id"], "call_abc");
        assert_eq!(msgs[2]["content"], "18C, clear");
    }

    #[test]
    fn vision_user_message_uses_array_content() {
        let user = ProviderMessage {
            role: Role::User,
            content: vec![
                ProviderBlock::Text { text: "What is this?".into() },
                ProviderBlock::Image { media_type: "image/png".into(), data: "AAAA".into() },
            ],
        };
        let body = build_request_body(&req(vec![user], vec![]));
        let content = &body["messages"][1]["content"];
        assert!(content.is_array());
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "image_url");
        assert!(content[1]["image_url"]["url"].as_str().unwrap().starts_with("data:image/png;base64,"));
    }
}
