use std::sync::Arc;
use clap::Parser;
use kiki_telemetry::init as init_telemetry;
use serde::Deserialize;
use tokio::sync::mpsc;
use tracing::{error, info, warn};

use kiki_core::{
    capability::CapabilitySet,
    context::{Context, ControlMode},
    gate::CapabilityGate,
    harness::{AgentConfig, Harness, HarnessConfig},
    surface::SurfaceSignal,
    types::ControlMessage,
};
use kiki_mcp::{McpHub, McpServer, PluginLoader};
use kiki_orchestrator::{
    bus::EventBus,
    dreamer::Dreamer,
    scheduler::{SessionPriority, SessionScheduler},
    session::{AgentSession, SessionManager},
};
use kiki_state::MemoryBackend;

// ── CLI args ──────────────────���───────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "agentd", about = "Kiki OS agentic daemon")]
struct Args {
    #[arg(long, default_value = "/etc/kiki/agentd.toml")]
    config: std::path::PathBuf,

    #[arg(long)]
    no_fleet: bool,

    #[arg(long)]
    no_wm: bool,
}

// ── Config schema ─────────────��───────────────────────────────────────────────

#[derive(Deserialize, Debug)]
struct Config {
    inference:     InferenceConfig,
    router_policy: RouterPolicy,
    control_mode:  ControlModeConfig,
    sockets:       SocketsConfig,
    fleet:         FleetConfig,
    #[serde(default)]
    apps:          AppsConfig,
}

#[derive(Deserialize, Debug)]
struct InferenceConfig {
    default_model:    String,
    distill_model:    Option<String>,
    /// Path to the `llama-server` binary. Defaults to `llama-server` (on PATH).
    #[serde(default = "default_llama_server")]
    llama_server_bin: String,
    /// Directory where GGUF model weights are installed by ArtifactManager.
    /// Resolver maps `model_id` → `<local_model_dir>/<model_id>.gguf`.
    local_model_dir:  String,
    /// Idle TTL in seconds before an in-process `llama-server` slot is reaped.
    #[serde(default = "default_idle_ttl")]
    idle_ttl_secs:    u64,
    /// Maximum concurrent `llama-server` slots (RAM/VRAM cap).
    #[serde(default = "default_max_slots")]
    max_slots:        usize,
    /// Default `-c` context size when a model manifest hasn't been loaded yet.
    #[serde(default = "default_context_size")]
    context_size:     u32,
}

fn default_llama_server() -> String { "llama-server".into() }
fn default_idle_ttl()     -> u64    { 300 }
fn default_max_slots()    -> usize  { 2 }
fn default_context_size() -> u32    { 8192 }

#[derive(Deserialize, Debug)]
struct RouterPolicy {
    allow_remote:                     bool,
    #[allow(dead_code)]
    allow_third_party_remote:         bool,
    #[allow(dead_code)]
    default_privacy_level:            String,
    preferred_model:                  String,
    #[allow(dead_code)]
    disable_remote_below_battery_pct: u8,
    #[allow(dead_code)]
    disable_third_party_for_voice:    bool,
    #[allow(dead_code)]
    trace_decisions:                  bool,
}

#[derive(Deserialize, Debug)]
struct ControlModeConfig {
    /// One of: bypass_permissions | agent_mode | assisted_mode | human_mode
    default:                         String,
    #[allow(dead_code)]
    allow_remote_bypass:             bool,
    bypass_checkpoint_interval_secs: u64,
}

#[derive(Deserialize, Debug)]
struct SocketsConfig {
    mcp:     String,
    #[allow(dead_code)]
    a11y:    String,
    #[allow(dead_code)]
    memory:  String,
    control: String,
}

#[derive(Deserialize, Debug)]
struct FleetConfig {
    enabled:   bool,
    cloud_url: String,
    #[allow(dead_code)]
    heartbeat_interval: u64,
}

#[derive(Deserialize, Debug, Default)]
struct AppsConfig {
    #[serde(default = "default_apps_dir")]
    dir: String,
}

fn default_apps_dir() -> String { "/var/kiki/apps".into() }

// ── Weights resolver ─────────────────────────────────────────────────────────
//
// Maps `model_id` → `<root>/<model_id>.gguf`. Used until ArtifactManager is
// wired (then it provides a resolver that also checks installation state and
// hardware requirements).

struct StaticDirResolver {
    root:         std::path::PathBuf,
    context_size: u32,
}

#[async_trait::async_trait]
impl kiki_provider::local::WeightsResolver for StaticDirResolver {
    async fn resolve(
        &self,
        model_id: &str,
    ) -> kiki_core::error::Result<kiki_provider::local::WeightsRef> {
        let safe = sanitize_model_id(model_id);
        let path = self.root.join(format!("{safe}.gguf"));
        if !tokio::fs::try_exists(&path).await.unwrap_or(false) {
            return Err(kiki_core::error::Error::Provider(format!(
                "model weights not found at {}", path.display()
            )));
        }
        Ok(kiki_provider::local::WeightsRef {
            gguf_path:    path,
            context_size: self.context_size,
            gpu_layers:   None,
        })
    }
}

fn sanitize_model_id(id: &str) -> String {
    id.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' { c } else { '_' })
        .collect()
}

// ── Entry point ───────────────��─────────────────────────────���─────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_telemetry();
    let args = Args::parse();

    let raw = std::fs::read_to_string(&args.config)
        .unwrap_or_else(|_| {
            warn!(path = ?args.config, "config not found — using defaults");
            include_str!("../default_config.toml").into()
        });
    let cfg: Config = toml::from_str(&raw)?;

    let control_mode = parse_control_mode(&cfg.control_mode.default);
    let model = if cfg.router_policy.allow_remote {
        cfg.router_policy.preferred_model.clone()
    } else {
        cfg.inference.default_model.clone()
    };

    info!(
        model   = %model,
        mode    = %cfg.control_mode.default,
        fleet   = cfg.fleet.enabled,
        "agentd starting"
    );

    // ── 1. Durable state backend ────────────────────────────────���─────────────
    // Production: OstreeBackend. Dev/test: MemoryBackend.
    let state = Arc::new(MemoryBackend::default());
    info!("state backend: memory (dev mode)");

    // ── 2. MCP hub + Unix socket server ────────────────────────────��─────────
    let hub = Arc::new(McpHub::new());
    let mcp_server = McpServer::new(hub.clone(), cfg.sockets.mcp.clone());
    let _mcp_handle = mcp_server.serve().await
        .map_err(|e| anyhow::anyhow!("MCP server failed to start: {e}"))?;
    info!(socket = %cfg.sockets.mcp, "MCP server started");

    // ── 3. Plugin loader — scan /var/kiki/apps for installed artifacts ────────
    let granted = CapabilitySet::new(); // TODO: load from node policy file
    let loader  = PluginLoader::new(hub.clone(), granted.clone(), &cfg.sockets.mcp);
    let loaded  = loader.load_directory(&cfg.apps.dir).await;
    info!(artifacts = loaded, dir = %cfg.apps.dir, "artifacts loaded");

    // ── 4. Event bus ─────────���───────────────────────────────────────────────
    let bus       = EventBus::new();
    let scheduler = SessionScheduler::new();
    let sessions  = SessionManager::new();

    // ── 5. Provider ──────────────────────────────────────��────────────────────
    // Wire provider based on routing policy. All providers implement LlmProvider.
    // Local provider drives a llama.cpp `llama-server` subprocess pool. Cloud
    // providers (Anthropic/OpenAI/Modal) are layered above via the router (out
    // of scope here — they require API keys synced via kiki-config Secrets).
    let provider: Arc<dyn kiki_core::provider::LlmProvider> = {
        use kiki_provider::local::{LlamaCppProvider, LlamaConfig};
        let llama_cfg = LlamaConfig {
            binary:    std::path::PathBuf::from(&cfg.inference.llama_server_bin),
            idle_ttl:  std::time::Duration::from_secs(cfg.inference.idle_ttl_secs),
            host:      "127.0.0.1".into(),
            port_base: 18000,
            max_slots: cfg.inference.max_slots,
        };
        let resolver = Arc::new(StaticDirResolver {
            root:         std::path::PathBuf::from(&cfg.inference.local_model_dir),
            context_size: cfg.inference.context_size,
        });
        Arc::new(LlamaCppProvider::new(llama_cfg, resolver))
            as Arc<dyn kiki_core::provider::LlmProvider>
    };

    // ── 6. Dreamer (post-session memory consolidation) ─────────────────────────
    let distill_model = cfg.inference.distill_model
        .clone()
        .unwrap_or_else(|| "llama3.2:1b".into()); // fast local model for distillation
    let dreamer = Arc::new(Dreamer::new(distill_model, provider.clone()));

    // ── 7. Fleet client (optional) ────────────────────────────────────────────
    if !args.no_fleet && cfg.fleet.enabled {
        info!(cloud_url = %cfg.fleet.cloud_url, "fleet client enabled (TODO: init)");
        // TODO: kiki_fleet::FleetClient::connect(cfg.fleet.cloud_url, state.clone()).await?
    }

    // ── 8. Control socket — listen for compositor + remote control messages ───
    let (control_tx, control_rx) = mpsc::channel::<ControlMessage>(64);
    let (surface_tx, mut surface_rx) = mpsc::channel::<SurfaceSignal>(256);

    // Surface signal drain (log them for now; wm will read over IPC)
    tokio::spawn(async move {
        while let Some(sig) = surface_rx.recv().await {
            tracing::debug!(?sig, "surface signal");
        }
    });

    // Start control socket listener
    if !args.no_wm {
        let socket_path = cfg.sockets.control.clone();
        let ctrl_tx     = control_tx.clone();
        tokio::spawn(async move {
            run_control_socket(socket_path, ctrl_tx).await;
        });
        info!(socket = %cfg.sockets.control, "control socket listener started");
    }

    // ── 9. Spin up the default agent session ──────────────────────────────────
    let session_id = format!("session-{}", now_ms());
    let agent_id   = "kiki-assistant".to_string();

    let session = Arc::new(AgentSession::new(
        session_id.clone(),
        "Kiki OS Assistant",
        agent_id.clone(),
        state.clone(),
    ));
    sessions.add(session.clone());

    let ctx = {
        let mut c = Context::new(agent_id.clone(), session_id.clone(), state.clone());
        c.set_mode(control_mode);
        c.max_steps = None; // no step limit
        c
    };

    let (cap_surface_tx, _cap_surface_rx) = mpsc::channel::<SurfaceSignal>(64);
    let gate = CapabilityGate::new(granted.clone(), cap_surface_tx);

    let harness_cfg = HarnessConfig {
        model:                      model.clone(),
        bypass_checkpoint_interval: cfg.control_mode.bypass_checkpoint_interval_secs as u32,
        ..Default::default()
    };

    let tools    = Arc::new(hub.build_registry());
    let (ev_tx, mut ev_rx) = mpsc::channel(128);

    let mut harness = Harness::new(
        Arc::new(KikiAssistantAgent { agent_id: agent_id.clone() }),
        ctx,
        harness_cfg,
        provider.clone(),
        tools,
        gate,
        surface_tx.clone(),
        control_rx,
    ).with_event_channel(ev_tx);

    // Relay AgentEvents to the event bus
    let bus2 = bus.clone();
    let sid2  = session_id.clone();
    tokio::spawn(async move {
        while let Some(event) = ev_rx.recv().await {
            bus2.publish_agent(sid2.clone(), event);
        }
    });

    // ── 10. Run the harness ──────────────────────────���─────────────────────────
    info!(session = %session_id, "harness starting");
    scheduler.add(session.clone(), SessionPriority::Foreground, None);
    scheduler.set_foreground(&session_id);

    match harness.run().await {
        Ok(outcome) => {
            let messages = harness.ctx.messages.clone();
            info!(session = %session_id, ?outcome, "session complete");
            session.complete();
            dreamer.spawn(session_id.clone(), agent_id, messages, state);
        }
        Err(e) => {
            error!(session = %session_id, error = %e, "session failed");
            session.fail(e.to_string());
        }
    }

    Ok(())
}

// ── Default agent config ──────────────────────────────────────────────────────

struct KikiAssistantAgent { agent_id: String }

impl AgentConfig for KikiAssistantAgent {
    fn id(&self) -> &str { &self.agent_id }

    fn system_prompt(&self, ctx: &kiki_core::context::Context) -> String {
        format!(
            "You are Kiki, an intelligent OS assistant running on a Kiki OS device. \
             You have access to OS tools via MCP. Be concise and helpful.\n\
             Session: {}. Mode: {:?}.",
            ctx.session_id, ctx.control_mode
        )
    }
}

// ── Control socket listener ─────────────────��─────────────────────────────────

async fn run_control_socket(path: String, tx: mpsc::Sender<ControlMessage>) {
    use tokio::{
        io::{AsyncBufReadExt, BufReader},
        net::UnixListener,
    };

    if std::path::Path::new(&path).exists() {
        let _ = std::fs::remove_file(&path);
    }

    let listener = match UnixListener::bind(&path) {
        Ok(l)  => l,
        Err(e) => { error!(socket = %path, error = %e, "control socket bind failed"); return; }
    };

    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                let tx2 = tx.clone();
                tokio::spawn(async move {
                    let (read, _write) = tokio::io::split(stream);
                    let mut lines = BufReader::new(read).lines();
                    while let Ok(Some(line)) = lines.next_line().await {
                        match serde_json::from_str::<ControlMessage>(&line) {
                            Ok(msg) => { let _ = tx2.send(msg).await; }
                            Err(e)  => { warn!(error = %e, "invalid control message"); }
                        }
                    }
                });
            }
            Err(e) => { error!(error = %e, "control socket accept error"); break; }
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn parse_control_mode(s: &str) -> ControlMode {
    match s {
        "bypass_permissions" => ControlMode::BypassPermissions,
        "assisted_mode"      => ControlMode::AssistedMode,
        "human_mode"         => ControlMode::HumanMode,
        _                    => ControlMode::AgentMode,
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
