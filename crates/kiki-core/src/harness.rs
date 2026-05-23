//! The agentic harness — the main turn loop.
//!
//! Architecture derived from eikarna/hermes-rs (ReAct conversation loop) and
//! adapted for Kiki OS's OS-level requirements:
//!
//! **From hermes-rs (exact patterns):**
//! - Concrete struct (not a trait) — `Harness` owns the loop; agents provide config
//! - Streaming: `StreamChunk::{Text,Thinking,ToolCall,Done}` — provider absorbs accumulation
//! - Context compaction: greedy turn-keeping when approaching `context_window`
//! - Self-healing: outer retry loop, error injected as system message (not user)
//! - Session distillation: async memory consolidation fires on completion
//! - `AgentEvent` broadcast channel for TUI / telemetry consumers
//!
//! **Kiki OS additions:**
//! - `Perceptor` trait: OS-native perception sources (a11y tree, kernel events, app state)
//! - `CapabilityGate`: ControlMode-aware tool gating with approval dialogs
//! - `SurfaceSignal` channel: real-time updates to the Wayland compositor
//! - `ControlMessage` receiver: user input, mode changes, stop/migrate commands
//! - OSTree checkpointing on freeze/bypass interval
//!
//! # Turn structure
//! ```text
//! loop {
//!   drain control_rx               // user input, mode changes, approval responses
//!   perceive()                     // gather OS perceptions → User ContentBlocks
//!   compact_context()              // drop old turns if near context_window
//!   build_request()                // system + messages + filtered tool specs
//!   stream LLM ─┐
//!               ├ select!(chunk | control_rx::StopSession)
//!               └ accumulate Text/Thinking/ToolCall/Done
//!   push AssistantTurn to history
//!   if no tool_calls → session done (checkpoint + distill)
//!   for each tool_call:
//!     gate.check() → Proceed/Skip/Redirect
//!     tool.call()  → ToolResult
//!   push ToolResults to history
//!   bypass checkpoint interval
//! }
//! ```

use std::sync::Arc;
use futures::StreamExt;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};
use crate::{
    context::{Context, ControlMode},
    context_manager::{ContextConfig, ContextManager},
    error::Result,
    gate::{CapabilityGate, GateDecision},
    interrupt::Interrupt,
    provider::{CompletionRequest, LlmProvider, ProviderBlock, ProviderMessage, StreamChunk, ToolSpec},
    surface::SurfaceSignal,
    tool::ToolRegistry,
    types::{AssistantTurn, ContentBlock, ControlMessage, ToolCall, ToolResult, ConversationMessage},
};

// ─── Perceptor ────────────────────────────────────────────────────────────────

/// An OS perception source polled at the start of each turn.
///
/// Perceptors run concurrently. Empty results are silently dropped.
/// Examples: A11yPerceptor (Wayland a11y tree), EventPerceptor (inotify/netlink),
/// AppStatePerceptor (Kiki-native app IPC), MemoryPerceptor (episodic recall).
#[async_trait::async_trait]
pub trait Perceptor: Send + Sync {
    fn name(&self) -> &str;
    async fn perceive(&self) -> Option<ContentBlock>;
}

// ─── AgentConfig ─────────────────────────────────────────────────────────────

/// What a specific agent contributes to the harness.
/// The harness owns the loop; agents provide the "what to do" policy.
pub trait AgentConfig: Send + Sync {
    fn id(&self) -> &str;
    /// Dynamic system prompt — rebuilt each turn to reflect current context.
    fn system_prompt(&self, ctx: &Context) -> String;
    fn required_capabilities(&self) -> Vec<crate::capability::Capability> { vec![] }
}

// ─── HarnessConfig ────────────────────────────────────────────────────────────

pub struct HarnessConfig {
    pub model:                      String,
    pub max_tokens:                 u32,
    /// Extended thinking budget (Claude 3.7+). None = disabled.
    pub thinking_tokens:            Option<u32>,
    /// Max tool calls per turn (safety limit).
    pub max_tools_per_turn:         usize,
    /// Max semantically-relevant tools passed to LLM per call.
    pub tool_context_limit:         usize,
    /// Token budget for conversation history (triggers compaction above this).
    pub context_window:             usize,
    /// OSTree checkpoint every N turns in BypassPermissions.
    pub bypass_checkpoint_interval: u32,
    pub temperature:                Option<f32>,
    /// Self-healing: max retries on LLM/tool error before surfacing failure.
    pub max_healing_attempts:       usize,
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self {
            model:                      "auto".into(),
            max_tokens:                 8192,
            thinking_tokens:            Some(4096),
            max_tools_per_turn:         20,
            tool_context_limit:         25,
            context_window:             120_000,  // tokens
            bypass_checkpoint_interval: 5,
            temperature:                None,
            max_healing_attempts:       3,
        }
    }
}

// ─── AgentEvent ───────────────────────────────────────────────────────────────

/// Emitted via the event channel for TUI, telemetry, and remote control consumers.
/// Mirrors hermes-rs AgentEvent, extended with OS-specific variants.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    Thinking     { text: String },
    Content      { text: String },
    ToolStart    { name: String, input: serde_json::Value },
    ToolComplete { name: String, success: bool },
    ModeChange   { mode: ControlMode },
    Checkpoint   { step: u32, reason: String },
    Compacting   { dropped_turns: usize },
    Healing      { attempt: usize, error: String },
    Done         { session_id: String, steps: u32 },
    Error        { error: String },
}

// ─── Harness ──────────────────────────────────────────────────────────────────

pub struct Harness {
    pub agent:   Arc<dyn AgentConfig>,
    pub ctx:     Context,
    pub config:  HarnessConfig,

    provider:    Arc<dyn LlmProvider>,
    tools:       Arc<ToolRegistry>,
    gate:        Arc<CapabilityGate>,
    perceptors:  Vec<Box<dyn Perceptor>>,

    surface_tx:  mpsc::Sender<SurfaceSignal>,
    control_rx:  mpsc::Receiver<ControlMessage>,
    event_tx:    Option<mpsc::Sender<AgentEvent>>,
}

impl Harness {
    pub fn new(
        agent:      Arc<dyn AgentConfig>,
        ctx:        Context,
        config:     HarnessConfig,
        provider:   Arc<dyn LlmProvider>,
        tools:      Arc<ToolRegistry>,
        gate:       Arc<CapabilityGate>,
        surface_tx: mpsc::Sender<SurfaceSignal>,
        control_rx: mpsc::Receiver<ControlMessage>,
    ) -> Self {
        Self {
            agent, ctx, config, provider, tools, gate,
            perceptors: vec![], surface_tx, control_rx, event_tx: None,
        }
    }

    pub fn with_event_channel(mut self, tx: mpsc::Sender<AgentEvent>) -> Self {
        self.event_tx = Some(tx);
        self
    }

    pub fn add_perceptor(&mut self, p: Box<dyn Perceptor>) {
        self.perceptors.push(p);
    }

    // ── Main entrypoint (with self-healing wrapper) ───────────────────────────

    /// Run with self-healing: on error, inject failure as a system message and retry.
    /// Pattern from hermes-rs `run_with_healing`. Error is injected as system (not user)
    /// so it doesn't pollute the visible conversation.
    pub async fn run(&mut self) -> Result<HarnessOutcome> {
        let prompt = self.agent.system_prompt(&self.ctx);
        self.ctx.set_system_prompt(prompt);

        info!(
            session  = %self.ctx.session_id,
            agent    = %self.ctx.agent_id,
            mode     = ?self.ctx.control_mode,
            "harness starting"
        );

        let mut healing_attempt = 0;

        loop {
            match self.run_inner().await {
                Ok(outcome)  => return Ok(outcome),
                Err(e) => {
                    healing_attempt += 1;
                    if healing_attempt > self.config.max_healing_attempts {
                        error!(error = %e, "max healing attempts exceeded");
                        self.emit(AgentEvent::Error { error: e.to_string() });
                        return Err(e);
                    }

                    warn!(attempt = healing_attempt, error = %e, "harness error — healing");
                    self.emit(AgentEvent::Healing { attempt: healing_attempt, error: e.to_string() });

                    // Inject error as a system message (not user-visible).
                    // hermes-rs pattern: healing context stays out of user conversation.
                    let heal_msg = format!(
                        "[auto-heal attempt {healing_attempt}] Previous turn failed: {e}. \
                         Adjust your approach and continue."
                    );
                    self.ctx.set_system_prompt(format!(
                        "{}\n\n{heal_msg}",
                        self.agent.system_prompt(&self.ctx)
                    ));
                }
            }
        }
    }

    // ── Inner loop ────────────────────────────────────────────────────────────

    async fn run_inner(&mut self) -> Result<HarnessOutcome> {
        loop {
            // ── 1. Drain control messages ─────────────────────────────────────
            while let Ok(msg) = self.control_rx.try_recv() {
                match self.handle_control(msg).await? {
                    LoopControl::Continue  => {}
                    LoopControl::Stop      => return Ok(HarnessOutcome::Stopped),
                    LoopControl::Freeze    => return Ok(HarnessOutcome::Frozen),
                }
            }

            // ── 2. Step limit ─────────────────────────────────────────────────
            if self.ctx.step_limit_reached() {
                self.checkpoint("step_limit").await.ok();
                return Ok(HarnessOutcome::StepLimit);
            }

            // ── 3. Perceive ───────────────────────────────────────────────────
            let blocks = self.perceive().await;
            if !blocks.is_empty() {
                self.ctx.push_perception(blocks);
            }

            // ── 4. Wait for first user input if session just started ──────────
            if self.ctx.messages.len() <= 1 {  // only system prompt
                match self.control_rx.recv().await {
                    Some(msg) => match self.handle_control(msg).await? {
                        LoopControl::Stop   => return Ok(HarnessOutcome::Stopped),
                        LoopControl::Freeze => return Ok(HarnessOutcome::Frozen),
                        LoopControl::Continue => {}
                    },
                    None => return Ok(HarnessOutcome::Stopped),
                }
                continue;
            }

            // ── 5. Context compaction (hermes-rs pattern) ─────────────────────
            self.compact_context();

            // ── 6. Build request ──────────────────────────────────────────────
            let request = self.build_request();

            // ── 7. Stream LLM response ────────────────────────────────────────
            let mut stream = self.provider.complete(request).await?;
            let mut turn   = AssistantTurn::default();
            let mut text_buf     = String::new();
            let mut thinking_buf = String::new();

            'stream: loop {
                tokio::select! {
                    chunk = stream.next() => {
                        match chunk {
                            Some(Ok(StreamChunk::Text(t))) => {
                                text_buf.push_str(&t);
                                self.emit(AgentEvent::Content { text: t.clone() });
                                self.surface_tx.send(SurfaceSignal::Thinking { text: t }).await.ok();
                            }
                            Some(Ok(StreamChunk::Thinking(t))) => {
                                thinking_buf.push_str(&t);
                                self.emit(AgentEvent::Thinking { text: t });
                            }
                            Some(Ok(StreamChunk::ToolCall { id, name, input })) => {
                                debug!(tool = %name, "tool call from stream");
                                turn.tool_calls.push(ToolCall { id, name, input });
                            }
                            Some(Ok(StreamChunk::Done)) | None => break 'stream,
                            Some(Err(e)) => return Err(e),
                        }
                    }

                    // Hard-stop mid-stream (user pressed stop)
                    ctrl = self.control_rx.recv() => {
                        if let Some(ControlMessage::StopSession { .. }) = ctrl {
                            drop(stream);
                            return Ok(HarnessOutcome::Stopped);
                        }
                        if let Some(msg) = ctrl {
                            self.handle_control(msg).await.ok();
                        }
                    }
                }
            }

            turn.text     = if text_buf.is_empty()     { None } else { Some(text_buf) };
            turn.thinking = if thinking_buf.is_empty() { None } else { Some(thinking_buf) };

            let tool_calls = turn.tool_calls.clone();
            self.ctx.push_assistant(turn);

            // ── 8. No tool calls → session complete ───────────────────────────
            if tool_calls.is_empty() {
                let summary = self.ctx.messages.iter().rev()
                    .find_map(|m| if let ConversationMessage::Assistant(t) = m {
                        t.text.clone()
                    } else { None })
                    .unwrap_or_default();

                self.surface_tx.send(SurfaceSignal::Done { summary }).await.ok();
                self.checkpoint("session_complete").await.ok();
                self.emit(AgentEvent::Done {
                    session_id: self.ctx.session_id.clone(),
                    steps:      self.ctx.steps_taken(),
                });
                return Ok(HarnessOutcome::Complete);
            }

            // ── 9. Execute tool calls (gate-checked) ──────────────────────────
            let mut results       = Vec::new();
            let mut redirected    = false;

            'tools: for call in tool_calls.into_iter().take(self.config.max_tools_per_turn) {
                let decision = self.gate.check(
                    &call,
                    self.ctx.control_mode,
                    self.ctx.is_bypass(),
                ).await?;

                match decision {
                    GateDecision::Skip { reason } => {
                        warn!(tool = %call.name, %reason, "gate skip");
                        results.push(ToolResult::rejected(&call.id, reason));
                        continue 'tools;
                    }
                    GateDecision::Redirect { new_intent } => {
                        self.ctx.push_user_text(format!(
                            "[User redirect] New direction: {new_intent}"
                        ));
                        // Push accumulated results and restart turn with new intent.
                        self.ctx.push_tool_results(std::mem::take(&mut results));
                        redirected = true;
                        break 'tools;
                    }
                    GateDecision::Proceed => {
                        self.emit(AgentEvent::ToolStart {
                            name:  call.name.clone(),
                            input: call.input.clone(),
                        });
                        self.surface_tx.send(SurfaceSignal::ToolRunning {
                            tool_name: call.name.clone(),
                        }).await.ok();

                        let result = match self.tools.get(&call.name) {
                            Some(tool) => match tool.call(call.input.clone()).await {
                                Ok(o) => {
                                    let content = match &o.content {
                                        serde_json::Value::String(s) => s.clone(),
                                        v => v.to_string(),
                                    };
                                    if o.is_error {
                                        ToolResult::err(&call.id, content)
                                    } else {
                                        ToolResult::ok(&call.id, content)
                                    }
                                }
                                Err(e) => {
                                    error!(tool = %call.name, error = %e);
                                    ToolResult::err(&call.id, e.to_string())
                                }
                            },
                            None => ToolResult::err(&call.id, format!("Unknown tool: {}", call.name)),
                        };

                        let success = !result.is_error;
                        self.emit(AgentEvent::ToolComplete { name: call.name.clone(), success });
                        self.surface_tx.send(SurfaceSignal::ToolDone {
                            tool_name: call.name.clone(),
                            success,
                        }).await.ok();

                        results.push(result);
                    }
                }
            }

            if !redirected {
                self.ctx.push_tool_results(results);
            }

            // ── 10. Bypass checkpoint interval ────────────────────────────────
            if self.ctx.is_bypass()
                && self.ctx.steps_taken() % self.config.bypass_checkpoint_interval == 0
            {
                self.checkpoint("bypass_interval").await.ok();
            }
        }
    }

    // ── Context compaction ────────────────────────────────────────────────────

    /// Sync ctx.messages → ContextManager, compact if needed, sync back.
    ///
    /// Delegates to `ContextManager` (kiki-core/src/context_manager.rs) which uses an
    /// incremental token counter (VecDeque + running total) — same approach as
    /// eikarna/hermes-rs `crates/hermes-core/src/context.rs`.
    fn compact_context(&mut self) {
        let cfg = ContextConfig {
            max_context_length:    self.config.context_window,
            response_buffer:       self.config.max_tokens as usize,
            min_messages_preserve: 4,
        };
        let mut mgr = ContextManager::new(cfg);
        mgr.replace_all(std::mem::take(&mut self.ctx.messages));

        if mgr.needs_compaction() {
            let dropped = mgr.compact();
            if dropped > 0 {
                info!(session = %self.ctx.session_id, dropped, "compacting context");
                self.emit(AgentEvent::Compacting { dropped_turns: dropped });
            }
        }
        self.ctx.messages = mgr.messages();
    }

    // ── Perception ────────────────────────────────────────────────────────────

    async fn perceive(&self) -> Vec<ContentBlock> {
        let futures: Vec<_> = self.perceptors.iter().map(|p| p.perceive()).collect();
        futures::future::join_all(futures).await
            .into_iter()
            .flatten()
            .collect()
    }

    // ── Request builder ───────────────────────────────────────────────────────

    fn build_request(&self) -> CompletionRequest {
        let messages = self.ctx.messages.iter()
            .filter_map(conversation_to_provider)
            .collect();

        let query = self.last_user_text();
        let tool_specs: Vec<ToolSpec> = self.tools
            .filter(&query, self.config.tool_context_limit)
            .into_iter()
            .map(|t| ToolSpec {
                name:         t.name().to_string(),
                description:  t.description().to_string(),
                input_schema: t.input_schema().clone(),
            })
            .collect();

        CompletionRequest {
            model:           self.config.model.clone(),
            messages,
            tools:           tool_specs,
            max_tokens:      Some(self.config.max_tokens),
            temperature:     self.config.temperature,
            thinking_tokens: self.config.thinking_tokens,
            system:          self.ctx.system_prompt().map(str::to_owned),
        }
    }

    // ── Control message handler ───────────────────────────────────────────────

    async fn handle_control(&mut self, msg: ControlMessage) -> Result<LoopControl> {
        match msg {
            ControlMessage::UserInput { text } => {
                self.ctx.push_user_text(text);
                Ok(LoopControl::Continue)
            }
            ControlMessage::ModeChange { mode } => {
                self.ctx.set_mode(mode);
                self.emit(AgentEvent::ModeChange { mode });
                self.ctx.log_interrupt(Interrupt::info(format!("Mode changed to {mode:?}")));
                Ok(LoopControl::Continue)
            }
            ControlMessage::ApprovalResponse { request_id, decision } => {
                self.gate.resolve_approval(&request_id, decision);
                Ok(LoopControl::Continue)
            }
            ControlMessage::StopSession { .. }  => Ok(LoopControl::Stop),
            ControlMessage::ParkSession { .. }  => {
                self.checkpoint("park").await.ok();
                Ok(LoopControl::Freeze)
            }
            ControlMessage::MigrateSession { .. } => {
                self.checkpoint("migrate").await.ok();
                Ok(LoopControl::Freeze)
            }
        }
    }

    // ── Checkpoint ────────────────────────────────────────────────────────────

    async fn checkpoint(&self, reason: &str) -> Result<()> {
        debug!(session = %self.ctx.session_id, reason, "checkpoint");
        self.emit(AgentEvent::Checkpoint {
            step:   self.ctx.steps_taken(),
            reason: reason.to_string(),
        });
        self.ctx.state.commit(&format!("{} — {}", self.ctx.session_id, reason)).await.map(|_| ())
    }

    // ── Event emission ────────────────────────────────────────────────────────

    fn emit(&self, event: AgentEvent) {
        if let Some(tx) = &self.event_tx {
            tx.try_send(event).ok();
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn last_user_text(&self) -> String {
        self.ctx.messages.iter().rev().find_map(|m| {
            if let ConversationMessage::User { content, .. } = m {
                content.iter().find_map(|b| {
                    if let ContentBlock::Text { text } = b { Some(text.clone()) } else { None }
                })
            } else { None }
        }).unwrap_or_default()
    }
}

// ─── Outcome ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HarnessOutcome {
    Complete,
    Stopped,
    Frozen,
    StepLimit,
}

#[derive(Debug)]
enum LoopControl { Continue, Stop, Freeze }

// ─── Conversation → Provider conversion ──────────────────────────────────────

fn conversation_to_provider(msg: &ConversationMessage) -> Option<ProviderMessage> {
    match msg {
        ConversationMessage::System { .. } => None,  // passed via request.system field

        ConversationMessage::User { content, .. } => {
            let blocks: Vec<ProviderBlock> = content.iter().map(|b| match b {
                ContentBlock::Text { text } =>
                    ProviderBlock::Text { text: text.clone() },
                ContentBlock::Image { media_type, data_base64 } =>
                    ProviderBlock::Image { media_type: media_type.clone(), data: data_base64.clone() },
                other =>
                    ProviderBlock::Text { text: serde_json::to_string(other).unwrap_or_default() },
            }).collect();
            if blocks.is_empty() { return None; }
            Some(ProviderMessage { role: crate::provider::Role::User, content: blocks })
        }

        ConversationMessage::Assistant(turn) => {
            let mut blocks = Vec::new();
            if let Some(t) = &turn.thinking {
                blocks.push(ProviderBlock::Thinking { thinking: t.clone() });
            }
            if let Some(t) = &turn.text {
                blocks.push(ProviderBlock::Text { text: t.clone() });
            }
            for call in &turn.tool_calls {
                blocks.push(ProviderBlock::ToolUse {
                    id: call.id.clone(), name: call.name.clone(), input: call.input.clone(),
                });
            }
            if blocks.is_empty() { return None; }
            Some(ProviderMessage { role: crate::provider::Role::Assistant, content: blocks })
        }

        ConversationMessage::ToolResults { results } => {
            let blocks: Vec<ProviderBlock> = results.iter().map(|r| ProviderBlock::ToolResult {
                tool_use_id: r.tool_call_id.clone(),
                content:     r.content.clone(),
                is_error:    r.is_error,
            }).collect();
            if blocks.is_empty() { return None; }
            Some(ProviderMessage { role: crate::provider::Role::Tool, content: blocks })
        }
    }
}
