//! Full-stack capstone: real LLM → real Harness → MCP hub → a tool served by a
//! SEPARATE app process over a Unix socket → result back into the agent.
//!
//! This is the strictest integration we run locally: unlike `harness_e2e` (which
//! registers an in-process `Tool`), the agent's tool here is an `McpProxyTool`
//! built from [`McpHub::build_registry`], so every tool call round-trips over the
//! real MCP server socket to an external app and back — the same path `agentd`
//! drives in production. Proves the LLM ↔ harness ↔ hub ↔ app chain is wired
//! end-to-end.
//!
//! Gated behind `KIKI_OLLAMA_TEST=1`. Run with:
//! ```sh
//! KIKI_OLLAMA_TEST=1 KIKI_OLLAMA_MODEL=qwen2.5:1.5b \
//!   cargo test -p kiki-provider --test capstone_e2e -- --nocapture --test-threads=1
//! ```

use std::sync::Arc;
use std::time::Duration;

use kiki_core::{
    capability::CapabilitySet,
    context::{Context, ControlMode},
    gate::CapabilityGate,
    harness::{AgentConfig, AgentEvent, Harness, HarnessConfig, HarnessOutcome},
    surface::SurfaceSignal,
    types::{ControlMessage, ConversationMessage},
};
use kiki_mcp::{McpHub, McpServer};
use kiki_provider::openai::OpenAiProvider;
use kiki_state::MemoryBackend;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::sync::mpsc;

fn enabled() -> bool {
    std::env::var("KIKI_OLLAMA_TEST").as_deref() == Ok("1")
}
fn model() -> String {
    std::env::var("KIKI_OLLAMA_MODEL").unwrap_or_else(|_| "qwen2.5:1.5b".into())
}

/// A real external app: handshake declaring `get_weather`, then serve
/// `tools/call` by returning structured weather JSON until the socket closes.
async fn run_weather_app(socket: String) {
    let stream = UnixStream::connect(&socket).await.expect("app connect");
    let (r, mut w) = stream.into_split();
    let mut lines = BufReader::new(r).lines();

    let init = json!({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "artifactId": "io.kiki.weather",
            "version": "1.0.0",
            "tools": [{
                "name": "get_weather",
                "description": "Get the current weather for a given city.",
                "input_schema": {
                    "type": "object",
                    "properties": { "city": { "type": "string", "description": "City name" } },
                    "required": ["city"]
                }
            }]
        }
    });
    w.write_all(format!("{init}\n").as_bytes()).await.unwrap();
    let _ = lines.next_line().await.unwrap(); // initialize reply

    while let Ok(Some(l)) = lines.next_line().await {
        let Ok(msg): Result<Value, _> = serde_json::from_str(&l) else { continue; };
        if msg["method"] == "tools/call" {
            let id = msg["id"].clone();
            let city = msg["params"]["arguments"]["city"].as_str().unwrap_or("unknown");
            let reply = json!({
                "jsonrpc": "2.0", "id": id,
                "result": { "city": city, "temp_c": 22, "condition": "sunny", "humidity_pct": 48 }
            });
            w.write_all(format!("{reply}\n").as_bytes()).await.unwrap();
        }
    }
}

struct TestAgent;
impl AgentConfig for TestAgent {
    fn id(&self) -> &str { "kiki-capstone-assistant" }
    fn system_prompt(&self, _ctx: &Context) -> String {
        "You are Kiki, an OS assistant. When the user asks about the weather you \
         MUST call the get_weather tool. After receiving the tool result, reply with \
         a short sentence stating the temperature and condition."
            .into()
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn llm_drives_tool_served_by_external_app() {
    if !enabled() {
        eprintln!("skipped: set KIKI_OLLAMA_TEST=1 to run");
        return;
    }

    // ── Real MCP server + external weather app ────────────────────────────────
    let socket = "/tmp/kiki-capstone-weather.sock";
    let _ = std::fs::remove_file(socket);
    let (hub_inner, mut catalog_rx) = McpHub::new().with_catalog_notifier();
    let hub = Arc::new(hub_inner);
    let _server = McpServer::new(hub.clone(), socket.to_string())
        .serve()
        .await
        .expect("serve");

    tokio::spawn(run_weather_app(socket.into()));

    // Wait until the app has registered its tools with the hub.
    tokio::time::timeout(Duration::from_secs(2), catalog_rx.recv())
        .await
        .expect("app registration timed out")
        .expect("notifier closed");

    // Tools the harness sees are MCP proxies routing over the socket to the app.
    let tools = Arc::new(hub.build_registry());

    // ── Real harness + real ollama provider ───────────────────────────────────
    let state = Arc::new(MemoryBackend::default());
    let provider = Arc::new(OpenAiProvider::ollama());

    let (surface_tx, mut surface_rx) = mpsc::channel::<SurfaceSignal>(256);
    tokio::spawn(async move { while surface_rx.recv().await.is_some() {} });

    let (cap_surface_tx, _cap_surface_rx) = mpsc::channel::<SurfaceSignal>(64);
    let gate = CapabilityGate::new(CapabilitySet::new(), cap_surface_tx);

    let (control_tx, control_rx) = mpsc::channel::<ControlMessage>(16);
    let (event_tx, mut event_rx) = mpsc::channel::<AgentEvent>(256);

    let mut ctx = Context::new(
        "kiki-capstone-assistant".to_string(),
        "session-capstone".to_string(),
        state.clone(),
    );
    ctx.set_mode(ControlMode::AgentMode);
    ctx.max_steps = Some(8);

    let config = HarnessConfig {
        model: model(),
        thinking_tokens: None,
        temperature: Some(0.0),
        ..Default::default()
    };

    let mut harness = Harness::new(
        Arc::new(TestAgent),
        ctx,
        config,
        provider,
        tools,
        gate,
        surface_tx,
        control_rx,
    ).with_event_channel(event_tx);

    control_tx
        .send(ControlMessage::UserInput {
            text: "What is the weather in Tokyo right now?".into(),
        })
        .await
        .unwrap();

    // ── Run + assert the chain end-to-end ─────────────────────────────────────
    let outcome = harness.run().await.expect("harness run failed");
    println!("[capstone] outcome={outcome:?}");
    assert_eq!(outcome, HarnessOutcome::Complete);

    let mut tool_started = false;
    let mut tool_succeeded = false;
    while let Ok(ev) = event_rx.try_recv() {
        match ev {
            AgentEvent::ToolStart { name, input } => {
                if name == "get_weather" {
                    tool_started = true;
                    assert_eq!(
                        input.get("city").and_then(|v| v.as_str()).map(str::to_lowercase),
                        Some("tokyo".into()),
                    );
                }
            }
            AgentEvent::ToolComplete { name, success } => {
                if name == "get_weather" && success { tool_succeeded = true; }
            }
            _ => {}
        }
    }
    assert!(tool_started, "agent must call get_weather (served by the external app)");
    assert!(tool_succeeded, "the externally-served get_weather must succeed");

    let final_text = harness.ctx.messages.iter().rev().find_map(|m| {
        if let ConversationMessage::Assistant(t) = m { t.text.clone() } else { None }
    }).unwrap_or_default();
    println!("[capstone] final answer: {final_text:?}");
    assert!(
        final_text.contains("22") || final_text.to_lowercase().contains("sunny"),
        "final answer should incorporate the app's tool result, got: {final_text:?}"
    );

    let _ = std::fs::remove_file(socket);
}
