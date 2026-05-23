//! Provider streaming-parse tests against a local mock SSE server.
//!
//! These need no API keys: a raw TCP listener replays byte-exact Server-Sent
//! Event payloads (taken from the published OpenAI and Anthropic streaming
//! formats) and we assert the providers decode them into the right
//! [`StreamChunk`] sequence. The body is written in several slices with the
//! split landing mid-line, which exercises the incremental line buffer.

use futures::StreamExt;
use kiki_provider::anthropic::{AnthropicConfig, AnthropicProvider};
use kiki_provider::openai::{OpenAiConfig, OpenAiProvider};
use kiki_provider::{CompletionRequest, LlmProvider, ProviderMessage, StreamChunk};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Start a one-shot HTTP/1.1 server that returns `body` as an event stream,
/// writing it in `slices` pieces to simulate chunked network arrival.
/// Returns the base URL (without a trailing path segment).
async fn spawn_sse(body: &'static str, slices: usize) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (mut sock, _) = listener.accept().await.unwrap();
        // Drain the request head (best-effort; we don't need the body).
        let mut tmp = [0u8; 8192];
        let _ = sock.read(&mut tmp).await;

        let head = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
            body.len()
        );
        sock.write_all(head.as_bytes()).await.unwrap();

        let bytes = body.as_bytes();
        let step = bytes.len().div_ceil(slices.max(1));
        let mut i = 0;
        while i < bytes.len() {
            let end = (i + step).min(bytes.len());
            sock.write_all(&bytes[i..end]).await.unwrap();
            sock.flush().await.unwrap();
            tokio::time::sleep(std::time::Duration::from_millis(3)).await;
            i = end;
        }
        let _ = sock.shutdown().await;
    });
    format!("http://{addr}")
}

#[derive(Default, Debug)]
struct Collected {
    text:       String,
    thinking:   String,
    tool_calls: Vec<(String, String, serde_json::Value)>, // (id, name, input)
    saw_done:   bool,
    error:      Option<String>,
}

async fn drain(provider: &dyn LlmProvider, req: CompletionRequest) -> Collected {
    let mut stream = provider.complete(req).await.expect("complete failed");
    let mut out = Collected::default();
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(StreamChunk::Text(t))     => out.text.push_str(&t),
            Ok(StreamChunk::Thinking(t)) => out.thinking.push_str(&t),
            Ok(StreamChunk::ToolCall { id, name, input }) => out.tool_calls.push((id, name, input)),
            Ok(StreamChunk::Done)        => { out.saw_done = true; }
            Err(e)                       => { out.error = Some(e.to_string()); break; }
        }
    }
    out
}

fn simple_req(model: &str) -> CompletionRequest {
    CompletionRequest {
        model: model.into(),
        messages: vec![ProviderMessage::user_text("hi")],
        tools: vec![],
        max_tokens: Some(256),
        temperature: Some(0.0),
        thinking_tokens: None,
        system: None,
    }
}

// ── OpenAI ────────────────────────────────────────────────────────────────────

const OPENAI_TEXT_AND_TOOL: &str = concat!(
    "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"content\":\"Sure, \"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"content\":\"checking now.\"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\":\"}}]}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\" \\\"Paris\\\"}\"}}]}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
    "data: [DONE]\n\n",
);

#[tokio::test]
async fn openai_parses_text_and_tool_call() {
    let base = spawn_sse(OPENAI_TEXT_AND_TOOL, 7).await;
    let provider = OpenAiProvider::new(OpenAiConfig {
        base_url: format!("{base}/v1"),
        api_key: String::new(),
        model_prefixes: Some(vec![]),
    });
    let out = drain(&provider, simple_req("gpt-4o")).await;

    assert!(out.error.is_none(), "unexpected error: {:?}", out.error);
    assert!(out.saw_done);
    assert_eq!(out.text, "Sure, checking now.");
    assert_eq!(out.tool_calls.len(), 1);
    let (id, name, input) = &out.tool_calls[0];
    assert_eq!(id, "call_1");
    assert_eq!(name, "get_weather");
    assert_eq!(input["city"], "Paris");
}

// ── Anthropic ─────────────────────────────────────────────────────────────────

const ANTHROPIC_TEXT_AND_TOOL: &str = concat!(
    "event: message_start\n",
    "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-sonnet-4-6\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":25,\"output_tokens\":1}}}\n\n",
    "event: content_block_start\n",
    "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
    "event: ping\n",
    "data: {\"type\": \"ping\"}\n\n",
    "event: content_block_delta\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Let me check\"}}\n\n",
    "event: content_block_delta\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" the weather.\"}}\n\n",
    "event: content_block_stop\n",
    "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
    "event: content_block_start\n",
    "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_01\",\"name\":\"get_weather\",\"input\":{}}}\n\n",
    "event: content_block_delta\n",
    "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\"}}\n\n",
    "event: content_block_delta\n",
    "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\" \\\"Paris\\\"}\"}}\n\n",
    "event: content_block_stop\n",
    "data: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
    "event: message_delta\n",
    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":40}}\n\n",
    "event: message_stop\n",
    "data: {\"type\":\"message_stop\"}\n\n",
);

#[tokio::test]
async fn anthropic_parses_text_and_tool_call() {
    let base = spawn_sse(ANTHROPIC_TEXT_AND_TOOL, 9).await;
    let provider = AnthropicProvider::with_config(AnthropicConfig {
        base_url: format!("{base}/v1"),
        api_key: "sk-ant-test".into(),
        version: "2023-06-01".into(),
    });
    let out = drain(&provider, simple_req("claude-sonnet-4-6")).await;

    assert!(out.error.is_none(), "unexpected error: {:?}", out.error);
    assert!(out.saw_done);
    assert_eq!(out.text, "Let me check the weather.");
    assert_eq!(out.tool_calls.len(), 1);
    let (id, name, input) = &out.tool_calls[0];
    assert_eq!(id, "toolu_01");
    assert_eq!(name, "get_weather");
    assert_eq!(input["city"], "Paris");
}

const ANTHROPIC_THINKING: &str = concat!(
    "event: content_block_start\n",
    "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n",
    "event: content_block_delta\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"Let me reason step by step.\"}}\n\n",
    "event: content_block_delta\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"abc123==\"}}\n\n",
    "event: content_block_stop\n",
    "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
    "event: content_block_start\n",
    "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
    "event: content_block_delta\n",
    "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"The answer is 42.\"}}\n\n",
    "event: content_block_stop\n",
    "data: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
    "event: message_stop\n",
    "data: {\"type\":\"message_stop\"}\n\n",
);

#[tokio::test]
async fn anthropic_parses_thinking_then_text() {
    let base = spawn_sse(ANTHROPIC_THINKING, 5).await;
    let provider = AnthropicProvider::with_config(AnthropicConfig {
        base_url: format!("{base}/v1"),
        api_key: "sk-ant-test".into(),
        version: "2023-06-01".into(),
    });
    let out = drain(&provider, simple_req("claude-sonnet-4-6")).await;

    assert!(out.error.is_none());
    assert!(out.saw_done);
    assert_eq!(out.thinking, "Let me reason step by step.");
    assert_eq!(out.text, "The answer is 42.");
    assert!(out.tool_calls.is_empty(), "signature_delta must not become a tool call");
}

const ANTHROPIC_ERROR: &str = concat!(
    "event: error\n",
    "data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n",
);

#[tokio::test]
async fn anthropic_surfaces_stream_error() {
    let base = spawn_sse(ANTHROPIC_ERROR, 1).await;
    let provider = AnthropicProvider::with_config(AnthropicConfig {
        base_url: format!("{base}/v1"),
        api_key: "sk-ant-test".into(),
        version: "2023-06-01".into(),
    });
    let out = drain(&provider, simple_req("claude-sonnet-4-6")).await;

    assert!(out.error.is_some(), "stream error event must surface as Err");
    assert!(out.error.unwrap().contains("Overloaded"));
    assert!(!out.saw_done, "error stream must not also report Done");
}
