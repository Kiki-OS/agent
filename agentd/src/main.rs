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
    /// Fleet worker origin (node register + session relay), e.g.
    /// `https://fleet.kiki-os.com`. Empty disables cloud connectivity.
    cloud_url: String,
    /// Auth worker origin for device-flow enrollment, e.g.
    /// `https://auth.kiki-os.com`. Defaults to `cloud_url` with the leading
    /// `fleet.` host swapped for `auth.` when left empty.
    #[serde(default)]
    auth_url:  String,
    /// Node flavor reported to the fleet registry (base/server/lite/desktop).
    #[serde(default = "default_flavor")]
    flavor:    String,
    #[serde(default = "default_heartbeat")]
    heartbeat_interval: u64,
}

fn default_flavor()    -> String { "headless".into() }
fn default_heartbeat() -> u64    { 30 }

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
    // Keep the spawned artifact processes alive for the lifetime of agentd.
    let _app_children = loader.load_directory(&cfg.apps.dir).await;
    info!(artifacts = _app_children.len(), dir = %cfg.apps.dir, "artifacts loaded (with exec)");

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
        use kiki_provider::{ProviderRouter, anthropic::AnthropicProvider, openai::OpenAiProvider};

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
        let local = Arc::new(LlamaCppProvider::new(llama_cfg, resolver));

        // Router: cloud providers (specific model prefixes) take priority, the
        // local llama.cpp runtime is the catch-all fallback. The router enforces
        // the privacy policy — when allow_remote is false, remote providers are
        // never selected regardless of model id. Cloud API keys are read from the
        // environment here as a bootstrap; production syncs them via kiki-config
        // Secrets.
        let mut router = ProviderRouter::new(cfg.router_policy.allow_remote);
        if cfg.router_policy.allow_remote {
            if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
                if !key.is_empty() {
                    info!("router: Anthropic cloud provider enabled");
                    router.add(Arc::new(AnthropicProvider::new(key)));
                }
            }
            if let Ok(key) = std::env::var("OPENAI_API_KEY") {
                if !key.is_empty() {
                    info!("router: OpenAI cloud provider enabled");
                    router.add(Arc::new(OpenAiProvider::with_api_key(key)));
                }
            }
        }
        router.add(local);
        info!(providers = router.provider_count(), allow_remote = cfg.router_policy.allow_remote, "provider router built");
        Arc::new(router) as Arc<dyn kiki_core::provider::LlmProvider>
    };

    // ── 6. Dreamer (post-session memory consolidation) ─────────────────────────
    let distill_model = cfg.inference.distill_model
        .clone()
        .unwrap_or_else(|| "llama3.2:1b".into()); // fast local model for distillation
    let dreamer = Arc::new(Dreamer::new(distill_model, provider.clone()));

    // ── 7. Fleet client (optional) ────────────────────────────────────────────
    // Wired below (§9.5) once the control channel + event stream exist, so the
    // cloud relay can drive sessions (web → device) and mirror agent state
    // (device → web).
    let fleet_enabled = !args.no_fleet && cfg.fleet.enabled && !cfg.fleet.cloud_url.is_empty();

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

    // Relay AgentEvents to the event bus, and tee a compact state mirror to the
    // fleet relay (device → cloud → web) when fleet is enabled.
    let bus2 = bus.clone();
    let sid2  = session_id.clone();
    let (state_tx, state_rx) = mpsc::channel::<kiki_fleet::StatePatch>(128);
    tokio::spawn(async move {
        while let Some(event) = ev_rx.recv().await {
            if fleet_enabled {
                // Lossy on purpose: if the relay isn't draining we drop mirror
                // updates rather than stalling the agent loop.
                let _ = state_tx.try_send(event_to_patch(&event));
            }
            bus2.publish_agent(sid2.clone(), event);
        }
    });

    // ── 9.5. Fleet enrollment + cloud relay ─────────────────────────────────────
    if fleet_enabled {
        let fleet_url = cfg.fleet.cloud_url.clone();
        let auth_url  = resolve_auth_url(&cfg.fleet.auth_url, &fleet_url);
        let flavor    = cfg.fleet.flavor.clone();
        let os_version = env!("CARGO_PKG_VERSION").to_string();
        let hb_secs   = cfg.fleet.heartbeat_interval;
        let node_id   = derive_node_id();
        let relay_session = node_id.clone(); // one session per node (web connects here)
        let token_store = kiki_fleet::TokenStore::new("/var/kiki/state/fleet-token");
        let ctrl_tx   = control_tx.clone();

        info!(node_id = %node_id, fleet = %fleet_url, "fleet enabled — enrolling");
        tokio::spawn(spawn_fleet(FleetSetup {
            fleet_url, auth_url, flavor, os_version, hb_secs,
            node_id, relay_session, token_store, ctrl_tx,
            state_rx, hub: hub.clone(),
        }));
    } else {
        info!("fleet disabled (no_fleet flag, disabled, or empty cloud_url)");
        drop(state_rx);
    }

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

// ── Fleet wiring ────────────────────────────────────────────────────────────

/// Derive a stable node id: prefer the host machine-id, else a persisted random.
fn derive_node_id() -> String {
    for p in ["/etc/machine-id", "/var/lib/dbus/machine-id"] {
        if let Ok(s) = std::fs::read_to_string(p) {
            let id = s.trim();
            if !id.is_empty() {
                return format!("node-{}", &id[..id.len().min(16)]);
            }
        }
    }
    // Fallback: persisted random id under the state dir.
    let path = std::path::Path::new("/var/kiki/state/node-id");
    if let Ok(s) = std::fs::read_to_string(path) {
        let id = s.trim();
        if !id.is_empty() { return id.to_string(); }
    }
    let id = format!("node-{}", random_hex(8));
    if let Some(parent) = path.parent() { let _ = std::fs::create_dir_all(parent); }
    let _ = std::fs::write(path, &id);
    id
}

fn random_hex(bytes: usize) -> String {
    // /dev/urandom exists on Linux and macOS; fall back to time-based entropy.
    let mut buf = vec![0u8; bytes];
    if std::fs::File::open("/dev/urandom")
        .and_then(|mut f| std::io::Read::read_exact(&mut f, &mut buf))
        .is_err()
    {
        let n = now_ms();
        for (i, b) in buf.iter_mut().enumerate() {
            *b = ((n >> (i % 8 * 8)) as u8) ^ (std::process::id() as u8);
        }
    }
    buf.iter().map(|b| format!("{b:02x}")).collect()
}

/// Derive the auth-worker origin from an explicit value or the fleet origin
/// (swap the leading `fleet.` host label for `auth.`).
fn resolve_auth_url(explicit: &str, fleet_url: &str) -> String {
    if !explicit.is_empty() {
        return explicit.to_string();
    }
    if let Some(rest) = fleet_url.strip_prefix("https://fleet.") {
        return format!("https://auth.{rest}");
    }
    if let Some(rest) = fleet_url.strip_prefix("http://fleet.") {
        return format!("http://auth.{rest}");
    }
    fleet_url.to_string()
}

/// Map an [`AgentEvent`] to a compact state mirror the web renders.
fn event_to_patch(ev: &kiki_core::harness::AgentEvent) -> kiki_fleet::StatePatch {
    use kiki_core::harness::AgentEvent as E;
    let status = match ev {
        E::Thinking { .. }          => serde_json::json!({ "kind": "thinking" }),
        E::Content { text }         => serde_json::json!({ "kind": "content", "text": text }),
        E::ToolStart { name, .. }   => serde_json::json!({ "kind": "tool_start", "tool": name }),
        E::ToolComplete { name, success } =>
            serde_json::json!({ "kind": "tool_complete", "tool": name, "success": success }),
        E::ModeChange { mode }      => serde_json::json!({ "kind": "mode_change", "mode": format!("{mode:?}") }),
        E::Checkpoint { step, .. }  => serde_json::json!({ "kind": "checkpoint", "step": step }),
        E::Compacting { .. }        => serde_json::json!({ "kind": "compacting" }),
        E::Healing { attempt, .. }  => serde_json::json!({ "kind": "healing", "attempt": attempt }),
        E::Done { steps, .. }       => serde_json::json!({ "kind": "done", "steps": steps }),
        E::Error { error }          => serde_json::json!({ "kind": "error", "error": error }),
    };
    kiki_fleet::StatePatch::agent_status(status)
}

struct FleetSetup {
    fleet_url:     String,
    auth_url:      String,
    flavor:        String,
    os_version:    String,
    hb_secs:       u64,
    node_id:       String,
    relay_session: String,
    token_store:   kiki_fleet::TokenStore,
    ctrl_tx:       mpsc::Sender<ControlMessage>,
    state_rx:      mpsc::Receiver<kiki_fleet::StatePatch>,
    hub:           Arc<kiki_mcp::McpHub>,
}

/// Enroll the node (device flow, persisted token), register + heartbeat, then
/// hold the SessionDO relay: mirror agent state up (device → cloud → web) and
/// execute remote `tool_call`s coming down (web → cloud → device).
async fn spawn_fleet(setup: FleetSetup) {
    let FleetSetup {
        fleet_url, auth_url, flavor, os_version, hb_secs,
        node_id, relay_session, token_store, ctrl_tx, mut state_rx, hub,
    } = setup;

    // ── Enrollment: token from env override → persisted store → device flow. ───
    // `KIKI_FLEET_TOKEN` injects a Bearer token directly (provisioning / CI).
    // `KIKI_FLEET_SKIP_ENROLL=1` registers unauthenticated (dev / headless nodes
    // that bind to an account later).
    let env_token = std::env::var("KIKI_FLEET_TOKEN").ok().filter(|t| !t.is_empty());
    let skip_enroll = std::env::var("KIKI_FLEET_SKIP_ENROLL").as_deref() == Ok("1");
    let token = if let Some(t) = env_token {
        info!("fleet: using KIKI_FLEET_TOKEN");
        let _ = token_store.save(&t);
        Some(t)
    } else if let Some(t) = token_store.load() {
        info!("fleet: reusing persisted enrollment token");
        Some(t)
    } else if skip_enroll {
        info!("fleet: KIKI_FLEET_SKIP_ENROLL set — registering unauthenticated");
        None
    } else {
        let label = format!("{flavor} ({node_id})");
        match kiki_fleet::DeviceFlow::new(&auth_url).authorize(Some(&label)).await {
            Ok(t) => {
                if let Err(e) = token_store.save(&t) {
                    warn!(error = %e, "fleet: could not persist token");
                }
                Some(t)
            }
            Err(e) => {
                warn!(error = %e, "fleet: device enrollment failed — \
                      registering unauthenticated (node won't bind to a user)");
                None
            }
        }
    };

    // ── Register + heartbeat. ──────────────────────────────────────────────────
    let mut client = kiki_fleet::FleetClient::new(&fleet_url, &node_id)
        .with_identity(&flavor, &os_version, None);
    if let Some(t) = &token {
        client = client.with_token(t.clone());
    }
    let client = Arc::new(client);

    match client.register_self().await {
        Ok(())  => info!(node_id = %node_id, "fleet: node registered"),
        Err(e)  => warn!(error = %e, "fleet: initial registration failed (heartbeat will retry)"),
    }
    let _hb = kiki_fleet::Heartbeat::new(client.clone(), std::time::Duration::from_secs(hb_secs.max(5)))
        .spawn();

    // ── SessionDO relay with reconnect. ────────────────────────────────────────
    loop {
        let (publisher, mut inbound) = match kiki_fleet::connect_device(&fleet_url, &relay_session).await {
            Ok(pair) => { info!(session = %relay_session, "fleet: session relay connected"); pair }
            Err(e) => {
                warn!(error = %e, "fleet: session relay connect failed — retrying in 5s");
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                continue;
            }
        };

        // Announce we're online.
        let _ = publisher.publish_patch(&kiki_fleet::StatePatch {
            phase: Some("active".into()),
            ..Default::default()
        }).await;

        // Drain state mirror → relay until the socket drops.
        loop {
            tokio::select! {
                patch = state_rx.recv() => match patch {
                    Some(p) => { if publisher.publish_patch(&p).await.is_err() { break; } }
                    None    => return, // agent shut down
                },
                msg = inbound.recv() => match msg {
                    Some(kiki_fleet::DeviceInbound::ToolCall { request_id, tool, input }) => {
                        info!(request_id = %request_id, tool = %tool, "fleet: remote tool_call");
                        let hub2 = hub.clone();
                        let pub2 = publisher.clone();
                        // Run the tool off the relay loop so a slow tool doesn't
                        // stall state mirroring.
                        tokio::spawn(async move {
                            match hub2.call(&tool, input).await {
                                Ok(result) => { let _ = pub2.tool_result(&request_id, result, None).await; }
                                Err(e)     => { let _ = pub2.tool_result(&request_id, serde_json::Value::Null, Some(e.to_string())).await; }
                            }
                        });
                    }
                    Some(kiki_fleet::DeviceInbound::InterruptResponse { interrupt_id, resolution }) => {
                        // Forward the human's decision into the harness control loop.
                        let _ = ctrl_tx.send(ControlMessage::ApprovalResponse {
                            request_id: interrupt_id,
                            decision:   approval_from_resolution(&resolution),
                        }).await;
                    }
                    None => break, // relay dropped → reconnect
                },
            }
        }
        warn!("fleet: session relay dropped — reconnecting");
    }
}

fn approval_from_resolution(v: &serde_json::Value) -> kiki_core::types::ApprovalDecision {
    use kiki_core::types::ApprovalDecision;
    let s = v.get("decision").and_then(|d| d.as_str())
        .or_else(|| v.as_str())
        .unwrap_or("");
    match s {
        "approve" | "approved" | "allow" | "yes" => ApprovalDecision::Approved,
        _ => ApprovalDecision::Rejected,
    }
}
