//! End-to-end agentic loop against real Ollama inference.
//!
//! This drives the *real* [`Harness`] from `kiki-core` — the same loop `agentd`
//! runs — wired to the Ollama-backed [`OpenAiProvider`], a real [`CapabilityGate`],
//! a real [`ToolRegistry`] with a working tool, and a [`MemoryBackend`]. It feeds
//! a user prompt over the control channel and asserts the agent:
//!   1. calls the `get_weather` tool with the right argument,
//!   2. receives the tool result,
//!   3. produces a final answer incorporating it,
//!   4. terminates with [`HarnessOutcome::Complete`].
//!
//! Gated behind `KIKI_OLLAMA_TEST=1`. Run with:
//! ```sh
//! KIKI_OLLAMA_TEST=1 KIKI_OLLAMA_MODEL=qwen2.5:1.5b \
//!   cargo test -p kiki-provider --test harness_e2e -- --nocapture --test-threads=1
//! ```

use std::sync::Arc;

use async_trait::async_trait;
use kiki_core::{
    capability::CapabilitySet,
    context::{Context, ControlMode},
    gate::CapabilityGate,
    harness::{AgentConfig, AgentEvent, Harness, HarnessConfig, HarnessOutcome},
    surface::SurfaceSignal,
    tool::{Tool, ToolOutput, ToolRegistry},
    types::{ControlMessage, ConversationMessage},
    error::Result,
};
use kiki_provider::openai::OpenAiProvider;
use kiki_state::MemoryBackend;
use serde_json::{json, Value};
use tokio::sync::mpsc;

fn enabled() -> bool {
    std::env::var("KIKI_OLLAMA_TEST").as_deref() == Ok("1")
}
fn model() -> String {
    std::env::var("KIKI_OLLAMA_MODEL").unwrap_or_else(|_| "qwen2.5:1.5b".into())
}

/// A real, deterministic weather tool. Returns structured JSON the model can
/// summarize. (`get_weather` is not in the gate's sensitive-capability table,
/// so it passes the static check in AgentMode.)
struct WeatherTool {
    schema: Value,
}
impl WeatherTool {
    fn new() -> Self {
        Self {
            schema: json!({
                "type": "object",
                "properties": { "city": { "type": "string", "description": "City name" } },
                "required": ["city"],
            }),
        }
    }
}
#[async_trait]
impl Tool for WeatherTool {
    fn name(&self) -> &str { "get_weather" }
    fn description(&self) -> &str { "Get the current weather for a given city." }
    fn input_schema(&self) -> &Value { &self.schema }
    async fn call(&self, input: Value) -> Result<ToolOutput> {
        let city = input.get("city").and_then(|v| v.as_str()).unwrap_or("unknown");
        Ok(ToolOutput::ok(json!({
            "city": city,
            "temp_c": 22,
            "condition": "sunny",
            "humidity_pct": 48,
        })))
    }
}

struct TestAgent;
impl AgentConfig for TestAgent {
    fn id(&self) -> &str { "kiki-test-assistant" }
    fn system_prompt(&self, _ctx: &Context) -> String {
        "You are Kiki, an OS assistant. When the user asks about the weather you \
         MUST call the get_weather tool. After receiving the tool result, reply with \
         a short sentence stating the temperature and condition."
            .into()
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn full_agentic_loop_with_real_tool() {
    if !enabled() {
        eprintln!("skipped: set KIKI_OLLAMA_TEST=1 to run");
        return;
    }

    // ── Wire the real components ──────────────────────────────────────────────
    let state = Arc::new(MemoryBackend::default());

    let mut registry = ToolRegistry::new();
    registry.register(WeatherTool::new());
    let tools = Arc::new(registry);

    let provider = Arc::new(OpenAiProvider::ollama());

    let (surface_tx, mut surface_rx) = mpsc::channel::<SurfaceSignal>(256);
    tokio::spawn(async move { while surface_rx.recv().await.is_some() {} });

    let (cap_surface_tx, _cap_surface_rx) = mpsc::channel::<SurfaceSignal>(64);
    let gate = CapabilityGate::new(CapabilitySet::new(), cap_surface_tx);

    let (control_tx, control_rx) = mpsc::channel::<ControlMessage>(16);
    let (event_tx, mut event_rx) = mpsc::channel::<AgentEvent>(256);

    let mut ctx = Context::new(
        "kiki-test-assistant".to_string(),
        "session-e2e".to_string(),
        state.clone(),
    );
    ctx.set_mode(ControlMode::AgentMode);
    ctx.max_steps = Some(8); // safety bound

    let config = HarnessConfig {
        model: model(),
        thinking_tokens: None,    // ollama models here don't do extended thinking
        temperature: Some(0.0),   // deterministic tool selection
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

    // Pre-load the user prompt; the harness drains pending control messages at
    // the top of its loop, so it starts work immediately.
    control_tx
        .send(ControlMessage::UserInput {
            text: "What is the weather in Tokyo right now?".into(),
        })
        .await
        .unwrap();

    // ── Run the loop ──────────────────────────────────────────────────────────
    let outcome = harness.run().await.expect("harness run failed");
    println!("[e2e] outcome={outcome:?}");
    assert_eq!(outcome, HarnessOutcome::Complete, "agent should finish cleanly");

    // ── Inspect emitted events ────────────────────────────────────────────────
    let mut tool_started = false;
    let mut tool_succeeded = false;
    while let Ok(ev) = event_rx.try_recv() {
        match ev {
            AgentEvent::ToolStart { name, input } => {
                println!("[e2e] tool_start {name} {input}");
                if name == "get_weather" {
                    tool_started = true;
                    assert_eq!(
                        input.get("city").and_then(|v| v.as_str()).map(str::to_lowercase),
                        Some("tokyo".into()),
                        "tool must be called with city=Tokyo",
                    );
                }
            }
            AgentEvent::ToolComplete { name, success } => {
                println!("[e2e] tool_complete {name} success={success}");
                if name == "get_weather" && success { tool_succeeded = true; }
            }
            _ => {}
        }
    }
    assert!(tool_started, "agent must have started the get_weather tool");
    assert!(tool_succeeded, "the get_weather tool must have succeeded");

    // ── Final assistant message must reflect the tool result ──────────────────
    let final_text = harness.ctx.messages.iter().rev().find_map(|m| {
        if let ConversationMessage::Assistant(t) = m { t.text.clone() } else { None }
    }).unwrap_or_default();
    println!("[e2e] final answer: {final_text:?}");
    assert!(
        final_text.contains("22") || final_text.to_lowercase().contains("sunny"),
        "final answer should incorporate the tool result, got: {final_text:?}"
    );
}
