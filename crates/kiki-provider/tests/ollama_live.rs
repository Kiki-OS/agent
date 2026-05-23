//! Live integration tests against a local Ollama server.
//!
//! These exercise [`OpenAiProvider`] end-to-end against real model inference —
//! real HTTP, real SSE streaming, real tool-call emission. They are gated behind
//! `KIKI_OLLAMA_TEST=1` so the normal `cargo test` run (and CI without a GPU)
//! skips them.
//!
//! Run with:
//! ```sh
//! KIKI_OLLAMA_TEST=1 KIKI_OLLAMA_MODEL=qwen2.5:1.5b \
//!   cargo test -p kiki-provider --test ollama_live -- --nocapture --test-threads=1
//! ```

use futures::StreamExt;
use kiki_provider::openai::OpenAiProvider;
use kiki_provider::{CompletionRequest, LlmProvider, ProviderMessage, StreamChunk, ToolSpec};
use serde_json::json;

fn enabled() -> bool {
    std::env::var("KIKI_OLLAMA_TEST").as_deref() == Ok("1")
}

fn model() -> String {
    std::env::var("KIKI_OLLAMA_MODEL").unwrap_or_else(|_| "qwen2.5:1.5b".into())
}

/// Collected results of draining a completion stream.
#[derive(Default, Debug)]
struct Collected {
    text:       String,
    tool_calls: Vec<(String, serde_json::Value)>, // (name, input)
    saw_done:   bool,
}

async fn drain(provider: &OpenAiProvider, req: CompletionRequest) -> Collected {
    let mut stream = provider.complete(req).await.expect("provider.complete failed");
    let mut out = Collected::default();
    while let Some(chunk) = stream.next().await {
        match chunk.expect("stream chunk error") {
            StreamChunk::Text(t)               => out.text.push_str(&t),
            StreamChunk::Thinking(_)           => {}
            StreamChunk::ToolCall { name, input, .. } => out.tool_calls.push((name, input)),
            StreamChunk::Done                  => { out.saw_done = true; }
        }
    }
    out
}

#[tokio::test(flavor = "multi_thread")]
async fn ollama_text_streaming() {
    if !enabled() {
        eprintln!("skipped: set KIKI_OLLAMA_TEST=1 to run");
        return;
    }
    let provider = OpenAiProvider::ollama();
    let req = CompletionRequest {
        model:       model(),
        messages:    vec![ProviderMessage::user_text(
            "Reply with exactly the word: pong",
        )],
        tools:       vec![],
        max_tokens:  Some(32),
        temperature: Some(0.0),
        thinking_tokens: None,
        system:      Some("You are a terse test fixture. Follow instructions exactly.".into()),
    };

    let out = drain(&provider, req).await;
    println!("[text] model={} reply={:?}", model(), out.text);
    assert!(out.saw_done, "stream must terminate with Done");
    assert!(!out.text.trim().is_empty(), "model must produce text");
    assert!(
        out.text.to_lowercase().contains("pong"),
        "expected 'pong' in reply, got: {:?}", out.text
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn ollama_tool_calling() {
    if !enabled() {
        eprintln!("skipped: set KIKI_OLLAMA_TEST=1 to run");
        return;
    }
    let provider = OpenAiProvider::ollama();
    let tool = ToolSpec {
        name:        "get_weather".into(),
        description: "Get the current weather for a given city.".into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "city": { "type": "string", "description": "City name, e.g. Tokyo" }
            },
            "required": ["city"]
        }),
    };
    let req = CompletionRequest {
        model:       model(),
        messages:    vec![ProviderMessage::user_text(
            "What is the weather in Tokyo right now? Use the available tool.",
        )],
        tools:       vec![tool],
        max_tokens:  Some(256),
        temperature: Some(0.0),
        thinking_tokens: None,
        system:      Some(
            "You are a helpful assistant. When asked about weather, you MUST call the get_weather tool.".into(),
        ),
    };

    let out = drain(&provider, req).await;
    println!("[tool] model={} text={:?} calls={:?}", model(), out.text, out.tool_calls);
    assert!(out.saw_done, "stream must terminate with Done");
    assert!(!out.tool_calls.is_empty(), "model must emit at least one tool call");
    let (name, input) = &out.tool_calls[0];
    assert_eq!(name, "get_weather", "tool name must match the spec");
    let city = input.get("city").and_then(|v| v.as_str()).unwrap_or("");
    assert!(
        city.to_lowercase().contains("tokyo"),
        "tool input should reference Tokyo, got: {input}"
    );
}

/// Two-turn loop: model calls the tool, we feed a result back, model answers.
/// Proves the structured assistant/tool message round-trip works against a real model.
#[tokio::test(flavor = "multi_thread")]
async fn ollama_tool_result_roundtrip() {
    if !enabled() {
        eprintln!("skipped: set KIKI_OLLAMA_TEST=1 to run");
        return;
    }
    use kiki_provider::{ProviderBlock, Role};

    let provider = OpenAiProvider::ollama();
    let tool = ToolSpec {
        name:        "get_weather".into(),
        description: "Get the current weather for a given city.".into(),
        input_schema: json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        }),
    };

    // Turn 1: get the tool call.
    let first = drain(&provider, CompletionRequest {
        model:       model(),
        messages:    vec![ProviderMessage::user_text("Weather in Tokyo? Use the tool.")],
        tools:       vec![tool.clone()],
        max_tokens:  Some(256),
        temperature: Some(0.0),
        thinking_tokens: None,
        system:      Some("You must call get_weather for weather questions.".into()),
    }).await;

    if first.tool_calls.is_empty() {
        eprintln!("model did not call a tool on turn 1; skipping round-trip assertion");
        return;
    }
    let (call_name, call_input) = first.tool_calls[0].clone();
    let call_id = "call_0".to_string();

    // Turn 2: feed the tool result back and ask for a final answer.
    let messages = vec![
        ProviderMessage::user_text("Weather in Tokyo? Use the tool."),
        ProviderMessage {
            role: Role::Assistant,
            content: vec![ProviderBlock::ToolUse {
                id:    call_id.clone(),
                name:  call_name,
                input: call_input,
            }],
        },
        ProviderMessage {
            role: Role::Tool,
            content: vec![ProviderBlock::ToolResult {
                tool_use_id: call_id,
                content:     "22°C and sunny".into(),
                is_error:    false,
            }],
        },
    ];

    let second = drain(&provider, CompletionRequest {
        model:       model(),
        messages,
        tools:       vec![tool],
        max_tokens:  Some(256),
        temperature: Some(0.0),
        thinking_tokens: None,
        system:      Some("Answer the user using the tool result.".into()),
    }).await;

    println!("[roundtrip] final={:?}", second.text);
    assert!(second.saw_done);
    assert!(
        second.text.contains("22") || second.text.to_lowercase().contains("sunny"),
        "final answer should incorporate the tool result, got: {:?}", second.text
    );
}
