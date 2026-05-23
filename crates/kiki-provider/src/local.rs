//! Local inference via `llama.cpp` (`llama-server` subprocess pool).
//!
//! Why not Ollama? It adds an externally-managed daemon and limits access to
//! advanced features (KV cache reuse across requests, GPU offload tuning,
//! mid-flight batching). See `project-arch-decisions` (2026-05-22).
//!
//! Architecture:
//!
//! - One `llama-server` subprocess per model artifact. The provider holds a
//!   [`ServerPool`] keyed by model id; a slot is spawned on first request,
//!   reused on subsequent requests, and reaped after `idle_ttl` of inactivity.
//! - Servers expose the OpenAI-compatible `/v1/chat/completions` endpoint
//!   over a loopback port assigned by us. We talk SSE to it.
//! - Weights paths come from the [`ArtifactManager`] (resolved from a
//!   [`kiki_schema::ArtifactKind::Model`] manifest). The provider never touches
//!   the registry or R2 directly.
//!
//! Failure model:
//! - Crashed server: detected via probe failure; reaped + respawned on next call.
//! - Spawn failure: surfaced as `Error::Provider` to the caller.
//! - OOM: caller's responsibility to gate by `ModelRequirements::satisfied_by`
//!   before requesting the model.

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};

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
use serde::{Deserialize, Serialize};
use tokio::process::Child;
use tokio::sync::{mpsc, Mutex};
use tokio_stream::wrappers::ReceiverStream;
use tracing::{info, warn};

/// One slot in the server pool. Each slot owns one `llama-server` process.
struct Slot {
    child:        Child,
    port:         u16,
    last_used:    Instant,
}

impl Drop for Slot {
    fn drop(&mut self) {
        // Best-effort shutdown when the slot is evicted or the pool drops.
        let _ = self.child.start_kill();
    }
}

/// Static resolver from model id → weights path on disk.
///
/// In production this is wired to `ArtifactManager::resolve_model_weights`,
/// which checks installation status, hardware requirements, and returns the
/// path that `llama-server` should mmap.
#[async_trait]
pub trait WeightsResolver: Send + Sync + 'static {
    async fn resolve(&self, model_id: &str) -> Result<WeightsRef>;
}

#[derive(Debug, Clone)]
pub struct WeightsRef {
    pub gguf_path:    std::path::PathBuf,
    /// Context window the model was compiled for. Passed to `-c`.
    pub context_size: u32,
    /// Optional GPU layer count. `Some(0)` forces CPU; `None` lets llama.cpp
    /// auto-detect.
    pub gpu_layers:   Option<u32>,
}

/// Subprocess-pool config.
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    /// Path to the `llama-server` binary. Default: `llama-server` on PATH.
    pub binary:    std::path::PathBuf,
    /// Idle TTL before a slot is reaped.
    pub idle_ttl:  Duration,
    /// Loopback host to bind. Always `127.0.0.1` in production.
    pub host:      String,
    /// First port to try; the pool picks an unused port starting here.
    pub port_base: u16,
    /// Max concurrent servers (cap on RAM/VRAM use). Hard-error past this.
    pub max_slots: usize,
}

impl Default for LlamaConfig {
    fn default() -> Self {
        Self {
            binary:    std::path::PathBuf::from("llama-server"),
            idle_ttl:  Duration::from_secs(300),
            host:      "127.0.0.1".into(),
            port_base: 18000,
            max_slots: 2,
        }
    }
}

pub struct LlamaCppProvider {
    cfg:       LlamaConfig,
    client:    Client,
    resolver:  Arc<dyn WeightsResolver>,
    pool:      Mutex<HashMap<String, Slot>>,
}

impl LlamaCppProvider {
    pub fn new(cfg: LlamaConfig, resolver: Arc<dyn WeightsResolver>) -> Self {
        Self {
            cfg,
            client: Client::new(),
            resolver,
            pool: Mutex::new(HashMap::new()),
        }
    }

    /// Get-or-spawn the slot for `model_id`. Side-effect: increments last_used.
    async fn slot_for(&self, model_id: &str) -> Result<u16> {
        let mut pool = self.pool.lock().await;
        self.evict_expired(&mut pool);
        if let Some(slot) = pool.get_mut(model_id) {
            // Health probe; if server died (try_wait returns Some), respawn.
            if slot.child.try_wait().map_err(io_err)?.is_some() {
                pool.remove(model_id);
            } else {
                slot.last_used = Instant::now();
                return Ok(slot.port);
            }
        }
        if pool.len() >= self.cfg.max_slots {
            // Evict LRU.
            if let Some(victim) = lru_key(&pool) {
                pool.remove(&victim);
            }
        }
        let weights = self.resolver.resolve(model_id).await?;
        let port = pick_port(&pool, self.cfg.port_base);
        let child = spawn_llama_server(&self.cfg, &weights, port).await?;
        let slot = Slot { child, port, last_used: Instant::now() };
        pool.insert(model_id.to_string(), slot);
        // Wait for the health endpoint to come up before returning the port.
        drop(pool);
        wait_ready(&self.client, &self.cfg.host, port, Duration::from_secs(30)).await?;
        Ok(port)
    }

    fn evict_expired(&self, pool: &mut HashMap<String, Slot>) {
        let now = Instant::now();
        let ttl = self.cfg.idle_ttl;
        pool.retain(|model, slot| {
            let keep = now.duration_since(slot.last_used) < ttl;
            if !keep {
                info!(model, "reaping idle llama-server slot");
            }
            keep
        });
    }
}

fn lru_key(pool: &HashMap<String, Slot>) -> Option<String> {
    pool.iter()
        .min_by_key(|(_, s)| s.last_used)
        .map(|(k, _)| k.clone())
}

fn pick_port(pool: &HashMap<String, Slot>, base: u16) -> u16 {
    let used: std::collections::HashSet<u16> = pool.values().map(|s| s.port).collect();
    (base..base.saturating_add(256))
        .find(|p| !used.contains(p))
        .unwrap_or(base)
}

async fn spawn_llama_server(
    cfg:     &LlamaConfig,
    weights: &WeightsRef,
    port:    u16,
) -> Result<Child> {
    let mut cmd = tokio::process::Command::new(&cfg.binary);
    cmd.arg("-m").arg(&weights.gguf_path)
        .arg("--host").arg(&cfg.host)
        .arg("--port").arg(port.to_string())
        .arg("-c").arg(weights.context_size.to_string())
        .arg("--api-key-required").arg("false")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    if let Some(n) = weights.gpu_layers {
        cmd.arg("-ngl").arg(n.to_string());
    }
    info!(
        binary = %cfg.binary.display(),
        gguf   = %weights.gguf_path.display(),
        port,
        "spawning llama-server",
    );
    cmd.spawn().map_err(|e| Error::Provider(format!("failed to spawn llama-server: {e}")))
}

async fn wait_ready(client: &Client, host: &str, port: u16, max: Duration) -> Result<()> {
    let deadline = Instant::now() + max;
    let url = format!("http://{host}:{port}/health");
    loop {
        if let Ok(resp) = client.get(&url).send().await {
            if resp.status().is_success() {
                return Ok(());
            }
        }
        if Instant::now() >= deadline {
            return Err(Error::Provider(format!(
                "llama-server on port {port} did not become ready within {max:?}"
            )));
        }
        tokio::time::sleep(Duration::from_millis(150)).await;
    }
}

fn io_err(e: std::io::Error) -> Error {
    Error::Provider(format!("io: {e}"))
}

// ── Wire types: subset of llama-server OAI compat ────────────────────────────

#[derive(Serialize)]
struct ChatRequest {
    model:    String,
    messages: Vec<WireMessage>,
    stream:   bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools:    Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Serialize)]
struct WireMessage {
    role:    &'static str,
    content: String,
}

#[derive(Deserialize, Default)]
struct Delta {
    content:    Option<String>,
    tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Deserialize)]
struct ToolCallDelta {
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
    delta:         Delta,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct ChatChunk {
    choices: Vec<Choice>,
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn blocks_to_string(blocks: &[ProviderBlock]) -> String {
    blocks.iter().map(|b| match b {
        ProviderBlock::Text { text } => text.clone(),
        ProviderBlock::Thinking { thinking } => format!("[thinking: {thinking}]"),
        ProviderBlock::Image { .. } => "[image]".into(),
        ProviderBlock::ToolUse { id, name, input } =>
            format!("<tool_call>{{\"name\":\"{name}\",\"id\":\"{id}\",\"arguments\":{input}}}</tool_call>"),
        ProviderBlock::ToolResult { tool_use_id, content, is_error } => {
            let kind = if *is_error { "error" } else { "result" };
            format!("[tool_{kind} id={tool_use_id}]: {content}")
        }
    }).collect::<Vec<_>>().join("")
}

// ── LlmProvider impl ─────────────────────────────────────────────────────────

#[async_trait]
impl LlmProvider for LlamaCppProvider {
    fn name(&self) -> &str { "llama.cpp" }

    fn supports_model(&self, model: &str) -> bool {
        // The resolver is the source of truth — anything it can resolve is supported.
        // For supports_model we don't want an async dispatch, so we approve any
        // model id that isn't an obviously-cloud identifier.
        !is_cloud_prefix(model)
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionStream> {
        if request.model.is_empty() {
            return Err(Error::Provider("model id required for local provider".into()));
        }
        let port = self.slot_for(&request.model).await?;
        let url = format!("http://{}:{}/v1/chat/completions", self.cfg.host, port);

        let mut messages: Vec<WireMessage> = Vec::new();
        if let Some(sys) = &request.system {
            messages.push(WireMessage { role: "system", content: sys.clone() });
        }
        for m in &request.messages {
            let role = match m.role {
                Role::System    => "system",
                Role::User      => "user",
                Role::Assistant => "assistant",
                Role::Tool      => "tool",
            };
            messages.push(WireMessage { role, content: blocks_to_string(&m.content) });
        }
        let tools = if request.tools.is_empty() {
            None
        } else {
            Some(request.tools.iter().map(|t| serde_json::json!({
                "type": "function",
                "function": {
                    "name":        t.name,
                    "description": t.description,
                    "parameters":  t.input_schema,
                }
            })).collect())
        };
        let body = ChatRequest {
            model:       request.model.clone(),
            messages,
            stream:      true,
            tools,
            temperature: request.temperature,
        };

        let resp = self.client.post(&url).json(&body).send().await
            .map_err(|e| Error::Provider(format!("llama-server request: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text   = resp.text().await.unwrap_or_default();
            return Err(Error::Provider(format!("llama-server {status}: {text}")));
        }

        let (tx, rx) = mpsc::channel::<Result<StreamChunk>>(64);
        tokio::spawn(stream_to_chunks(resp, tx));
        Ok(Box::pin(ReceiverStream::new(rx)))
    }
}

async fn stream_to_chunks(
    resp: reqwest::Response,
    tx:   mpsc::Sender<Result<StreamChunk>>,
) {
    let mut byte_stream = resp.bytes_stream();
    let mut buf = String::new();
    let mut ids:   HashMap<usize, String> = HashMap::new();
    let mut names: HashMap<usize, String> = HashMap::new();
    let mut args:  HashMap<usize, String> = HashMap::new();
    while let Some(chunk) = byte_stream.next().await {
        let bytes = match chunk {
            Ok(b) => b,
            Err(e) => {
                let _ = tx.send(Err(Error::Provider(e.to_string()))).await;
                return;
            }
        };
        buf.push_str(&String::from_utf8_lossy(&bytes));
        while let Some(pos) = buf.find('\n') {
            let line = buf[..pos].trim().to_string();
            buf = buf[pos + 1..].to_string();
            if !line.starts_with("data: ") { continue; }
            let data = &line["data: ".len()..];
            if data == "[DONE]" {
                flush_tool_calls(&tx, &ids, &names, &args).await;
                let _ = tx.send(Ok(StreamChunk::Done)).await;
                return;
            }
            let chunk: ChatChunk = match serde_json::from_str(data) {
                Ok(c) => c,
                Err(e) => { warn!("llama-server SSE parse: {e}"); continue; }
            };
            for choice in chunk.choices {
                if let Some(text) = choice.delta.content {
                    if !text.is_empty() {
                        let _ = tx.send(Ok(StreamChunk::Text(text))).await;
                    }
                }
                if let Some(tc_deltas) = choice.delta.tool_calls {
                    for tc in tc_deltas {
                        if let Some(id) = tc.id { ids.insert(tc.index, id); }
                        if let Some(func) = tc.function {
                            if let Some(name) = func.name { names.insert(tc.index, name); }
                            if let Some(a) = func.arguments {
                                args.entry(tc.index).or_default().push_str(&a);
                            }
                        }
                    }
                }
                if choice.finish_reason.as_deref() == Some("tool_calls") {
                    flush_tool_calls(&tx, &ids, &names, &args).await;
                    ids.clear(); names.clear(); args.clear();
                }
            }
        }
    }
    flush_tool_calls(&tx, &ids, &names, &args).await;
    let _ = tx.send(Ok(StreamChunk::Done)).await;
}

async fn flush_tool_calls(
    tx:    &mpsc::Sender<Result<StreamChunk>>,
    ids:   &HashMap<usize, String>,
    names: &HashMap<usize, String>,
    args:  &HashMap<usize, String>,
) {
    let mut indices: Vec<usize> = names.keys().copied().collect();
    indices.sort_unstable();
    for idx in indices {
        let id    = ids.get(&idx).cloned().unwrap_or_else(|| format!("tool-{idx}"));
        let name  = names[&idx].clone();
        let raw   = args.get(&idx).map(|s| s.as_str()).unwrap_or("{}");
        let input = serde_json::from_str(raw).unwrap_or(serde_json::Value::Object(Default::default()));
        let _ = tx.send(Ok(StreamChunk::ToolCall { id, name, input })).await;
    }
}

fn is_cloud_prefix(model: &str) -> bool {
    const CLOUD: &[&str] = &[
        "claude-", "gpt-4", "gpt-3", "gemini-", "command-r",
        "anthropic/", "openai/", "google/", "mistral-large",
    ];
    CLOUD.iter().any(|p| model.starts_with(p))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StaticResolver(WeightsRef);

    #[async_trait::async_trait]
    impl WeightsResolver for StaticResolver {
        async fn resolve(&self, _model_id: &str) -> Result<WeightsRef> {
            Ok(self.0.clone())
        }
    }

    #[test]
    fn cloud_models_not_supported() {
        let resolver = Arc::new(StaticResolver(WeightsRef {
            gguf_path: "/nonexistent.gguf".into(),
            context_size: 4096,
            gpu_layers: None,
        }));
        let p = LlamaCppProvider::new(LlamaConfig::default(), resolver);
        assert!(!p.supports_model("claude-3-5-sonnet"));
        assert!(!p.supports_model("gpt-4o"));
        assert!(p.supports_model("llama-3.1-8b-instruct"));
    }
}
