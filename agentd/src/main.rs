use std::sync::Arc;
use clap::Parser;
use kiki_telemetry::init as init_telemetry;

mod memory_ctx;
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

// ── Egress audit ───────────────────────────────────────────────────────────────
// The broker is the single auditable egress point; for now we log each brokered
// request. A durable sink (fleet/vault) can replace this later.
struct TracingEgressAudit;

#[async_trait::async_trait]
impl kiki_net::AuditSink for TracingEgressAudit {
    async fn record(&self, e: &kiki_net::EgressAudit) {
        info!(
            app = %e.app, method = %e.method, host = %e.host, port = e.port,
            status = e.status, bytes = e.bytes, "egress"
        );
    }
}

// ── Credential injection (OAuth → sealed Secrets) ────────────────────────────
// Build the egress broker's credential injector backed by an on-device sealed
// secret store. Provider client ids/secrets come from the environment (set per
// deployment); device endpoints are the well-known Google/Microsoft ones.
fn build_credential_injector() -> Option<Arc<dyn kiki_net::CredentialInjector>> {
    use kiki_oauth::{OAuthFlow, Provider, ProviderConfig, SealedFileSecretStore, SecretsCredentialInjector};

    let key = node_master_key()?;
    let store = SealedFileSecretStore::new(key, "/var/kiki/secrets");
    let flow = OAuthFlow::new(store)
        .with_provider(Provider::Google, ProviderConfig {
            token_url:                "https://oauth2.googleapis.com/token".into(),
            device_authorization_url: Some("https://oauth2.googleapis.com/device/code".into()),
            client_id:                std::env::var("KIKI_GOOGLE_CLIENT_ID").unwrap_or_default(),
            client_secret:            std::env::var("KIKI_GOOGLE_CLIENT_SECRET").ok(),
            scopes:                   vec!["https://mail.google.com/".into()],
        })
        .with_provider(Provider::Microsoft, ProviderConfig {
            token_url:                "https://login.microsoftonline.com/common/oauth2/v2.0/token".into(),
            device_authorization_url: Some("https://login.microsoftonline.com/common/oauth2/v2.0/devicecode".into()),
            client_id:                std::env::var("KIKI_MS_CLIENT_ID").unwrap_or_default(),
            client_secret:            None,
            scopes:                   vec!["offline_access".into(), "https://outlook.office.com/IMAP.AccessAsUser.All".into()],
        });
    // Account→host mappings are registered by the settings/OAuth flow as users
    // connect accounts; none are wired at boot.
    Some(Arc::new(SecretsCredentialInjector::new(Arc::new(flow))))
}

/// The node's master key for sealing on-device secrets. Interim: a 0600 key file
/// under /var/kiki/secrets, generated on first use. TODO(hardening): back this
/// with the platform keystore / TPM instead of a file.
fn node_master_key() -> Option<kiki_config::crypto::MasterKey> {
    use rand::RngCore;
    let path = std::path::Path::new("/var/kiki/secrets/.node-key");
    if let Ok(b) = std::fs::read(path) {
        if let Ok(arr) = <[u8; 32]>::try_from(b.as_slice()) {
            return Some(kiki_config::crypto::MasterKey::from_bytes(arr));
        }
    }
    let mut k = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut k);
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if std::fs::write(path, k).is_ok() {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }
    Some(kiki_config::crypto::MasterKey::from_bytes(k))
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
    let mut cfg: Config = toml::from_str(&raw)?;
    // Provisioning overrides (cloud-init / fleet): an env var wins over the baked
    // toml so a server node can be pointed at its control plane without rewriting
    // the config file. KIKI_FLEET_TOKEN is consumed later in fleet enrollment.
    if let Ok(url) = std::env::var("KIKI_FLEET_URL") {
        if !url.is_empty() {
            info!(cloud_url = %url, "fleet cloud_url overridden by KIKI_FLEET_URL");
            cfg.fleet.cloud_url = url;
        }
    }

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

    // ── 1. Durable state backend ──────────────────────────────────────────────
    // Durable file-backed state under /var/kiki/state survives restart + reboot.
    // Falls back to in-memory only if the state dir is unusable (dev/CI).
    let state_dir = std::env::var("KIKI_STATE_DIR").unwrap_or_else(|_| "/var/kiki/state".into());
    let state: Arc<dyn kiki_core::state::StateBackend> =
        match kiki_state::FileBackend::open(&state_dir) {
            Ok(b) => {
                info!(dir = %state_dir, "state backend: file (durable)");
                Arc::new(b)
            }
            Err(e) => {
                warn!(dir = %state_dir, error = %e, "durable state unavailable — using memory");
                Arc::new(MemoryBackend::default())
            }
        };

    // ── 2. MCP hub + Unix socket server ────────────────────────────��─────────
    // Ensure the runtime socket dir exists. systemd creates /run/kiki via
    // RuntimeDirectory=, but agentd must also work when launched directly
    // (containers, dev, provisioning tools).
    for sock in [&cfg.sockets.mcp, &cfg.sockets.control] {
        if let Some(dir) = std::path::Path::new(sock).parent() {
            let _ = std::fs::create_dir_all(dir);
        }
    }

    // The catalog notifier fires on every artifact register/unregister so the
    // shell's command palette + launcher refresh when apps load after boot.
    let (hub_inner, mut catalog_rx) = McpHub::new().with_catalog_notifier();
    let hub = Arc::new(hub_inner);

    // Egress broker: the single audited network-egress point. Seed each installed
    // app's allowlist from its manifest's [capabilities].network so `net.fetch`
    // is deny-by-default per app. (Credential injection from Secrets lands with
    // the OAuth unlock; for now no injector is wired.)
    let broker = {
        let mut b = kiki_net::EgressBroker::new().with_audit(Arc::new(TracingEgressAudit));
        for (app_id, hosts) in kiki_mcp::scan_egress_allowlists(&cfg.apps.dir) {
            info!(app = %app_id, hosts = hosts.len(), "egress allowlist registered");
            b.allow(app_id, hosts);
        }
        // Wire per-account credential injection (OAuth → sealed Secrets). The
        // broker injects bearer tokens into authenticated requests without ever
        // exposing them to the app. Mappings are added when accounts are
        // authorized; with none configured the injector is a no-op.
        if let Some(injector) = build_credential_injector() {
            b = b.with_injector(injector);
            info!("credential injector wired (sealed secret store)");
        }
        Arc::new(b)
    };

    let mcp_server = McpServer::new(hub.clone(), cfg.sockets.mcp.clone())
        .with_broker(broker.clone());
    let _mcp_handle = mcp_server.serve().await
        .map_err(|e| anyhow::anyhow!("MCP server failed to start: {e}"))?;
    info!(socket = %cfg.sockets.mcp, "MCP server started");

    // ── 3. Plugin loader — scan /var/kiki/apps for installed artifacts ────────
    // Capabilities the operator granted to installed artifacts (written by kpkg
    // on approval). Deny-by-default when the policy file is absent.
    let policy_path = std::env::var("KIKI_POLICY_FILE")
        .unwrap_or_else(|_| "/etc/kiki/policy.json".to_string());
    let granted = load_granted_caps(&policy_path);
    let loader  = PluginLoader::new(hub.clone(), granted.clone(), &cfg.sockets.mcp);
    // Built-in apps (baked, trusted) load from cfg.apps.dir. User-installed L2
    // apps load from /var/kiki/apps, validated against the granted policy.
    // Keep the spawned artifact processes alive for the lifetime of agentd.
    let mut _app_children = loader.load_directory(&cfg.apps.dir, true).await;
    let builtin_count = _app_children.len();
    let l2_dir = "/var/kiki/apps";
    if l2_dir != cfg.apps.dir {
        _app_children.extend(loader.load_directory(l2_dir, false).await);
    }
    info!(
        builtin = builtin_count,
        l2 = _app_children.len() - builtin_count,
        "artifacts loaded (with exec)"
    );

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

    // Cloud session container: when KIKI_RESUME_SESSION is set we resume a
    // session migrated from a device instead of spinning up a fresh one. The
    // node id is derived from the session so the source can address the
    // migration bundle to us and our poll finds it.
    let resume_session = std::env::var("KIKI_RESUME_SESSION").ok().filter(|s| !s.is_empty());
    // Node identity precedence:
    //   1. KIKI_NODE_ID — set by the cloud orchestrator (fleet instance / migrated
    //      session); the migration pointer + group membership are keyed by it.
    //   2. cloud-<session> — legacy migrated-session fallback.
    //   3. a stable host-derived id for an on-device node.
    let node_id = match (std::env::var("KIKI_NODE_ID").ok().filter(|s| !s.is_empty()), &resume_session) {
        (Some(n), _)    => n,
        (None, Some(s)) => format!("cloud-{s}"),
        (None, None)    => derive_node_id(),
    };

    // ── 8. Control socket — bidirectional: inbound ControlMessages from the
    // shell, outbound ShellEvents (HUD activity stream) to connected clients. ──
    let (control_tx, control_rx) = mpsc::channel::<ControlMessage>(64);
    let (surface_tx, mut surface_rx) = mpsc::channel::<SurfaceSignal>(256);
    // Pre-serialized ShellEvent JSON lines fan out to every connected shell.
    let (shell_tx, _) = tokio::sync::broadcast::channel::<String>(256);

    // Surface signal drain (log them for now; wm will read over IPC)
    tokio::spawn(async move {
        while let Some(sig) = surface_rx.recv().await {
            tracing::debug!(?sig, "surface signal");
        }
    });

    // Catalog refresh: when an artifact registers/unregisters after a shell is
    // already connected, re-broadcast the command palette + launcher app grid so
    // late-loading apps appear without a reconnect.
    {
        let events      = shell_tx.clone();
        let hub_refresh = hub.clone();
        tokio::spawn(async move {
            while catalog_rx.recv().await.is_ok() {
                let _ = events.send(commands_available_line(&hub_refresh));
                let _ = events.send(apps_available_line(&hub_refresh));
            }
        });
    }

    // Start control socket listener
    if !args.no_wm {
        let socket_path = cfg.sockets.control.clone();
        let ctrl_tx     = control_tx.clone();
        let events      = shell_tx.clone();
        let hub_ctl     = hub.clone();
        tokio::spawn(async move {
            run_control_socket(socket_path, ctrl_tx, events, hub_ctl).await;
        });
        info!(socket = %cfg.sockets.control, "control socket listener started");
    }

    // ── 9. Session: resume a migrated one (cloud), or spin up a fresh default ──
    let (session, ctx) = match &resume_session {
        Some(id) if fleet_enabled => {
            match resume_from_cloud(id, &node_id, &cfg, state.clone()).await {
                Ok(Some(pair)) => {
                    info!(session = %id, "resumed migrated session from bundle");
                    pair
                }
                Ok(None) => {
                    warn!(session = %id, "no migration bundle arrived — starting fresh");
                    fresh_session(id, control_mode, state.clone())
                }
                Err(e) => {
                    warn!(session = %id, error = %e, "resume failed — starting fresh");
                    fresh_session(id, control_mode, state.clone())
                }
            }
        }
        _ => fresh_session(&format!("session-{}", now_ms()), control_mode, state.clone()),
    };
    sessions.add(session.clone());
    let session_id = session.id.clone();
    let agent_id   = session.agent_id.clone();

    let (cap_surface_tx, mut cap_surface_rx) = mpsc::channel::<SurfaceSignal>(64);
    let gate = CapabilityGate::new(granted.clone(), cap_surface_tx);

    // Forward agent surface signals (approvals/interrupts) into the shell event
    // stream so the UI's approval banners + interrupt modals receive them. The
    // HUD activity stream (thinking/tools/mode) comes from AgentEvents below;
    // here we forward only the human-action signals AgentEvent lacks.
    {
        let events = shell_tx.clone();
        tokio::spawn(async move {
            while let Some(sig) = cap_surface_rx.recv().await {
                if let Some(line) = surface_signal_line(&sig) {
                    let _ = events.send(line);
                }
            }
        });
    }

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
    )
    .with_event_channel(ev_tx)
    .with_memory(Arc::new(memory_ctx::MemorydContext::new()));

    // Relay AgentEvents to the event bus, and tee a compact state mirror to the
    // fleet relay (device → cloud → web) when fleet is enabled.
    let bus2 = bus.clone();
    let sid2  = session_id.clone();
    let shell_events = shell_tx.clone();
    let snap_fleet_url = cfg.fleet.cloud_url.clone();
    let snap_flavor    = cfg.fleet.flavor.clone();
    let snap_node_id   = node_id.clone();
    let (state_tx, state_rx) = mpsc::channel::<kiki_fleet::StatePatch>(128);
    tokio::spawn(async move {
        while let Some(event) = ev_rx.recv().await {
            // A captured snapshot is uploaded to the snapshot store, not mirrored
            // to the relay or the local bus (it's an internal fleet operation).
            if let kiki_core::harness::AgentEvent::SnapshotCaptured { snapshot_id, bundle } = &event {
                if fleet_enabled {
                    let (url, flavor, nid) = (snap_fleet_url.clone(), snap_flavor.clone(), snap_node_id.clone());
                    let (sid, b) = (snapshot_id.clone(), (**bundle).clone());
                    tokio::spawn(async move {
                        let mut client = kiki_fleet::FleetClient::new(&url, &nid)
                            .with_identity(&flavor, env!("CARGO_PKG_VERSION"), None);
                        let token = std::env::var("KIKI_FLEET_TOKEN").ok().filter(|t| !t.is_empty())
                            .or_else(|| kiki_fleet::TokenStore::new("/var/kiki/state/fleet-token").load());
                        if let Some(t) = token { client = client.with_token(t); }
                        match client.upload_snapshot(&sid, &b).await {
                            Ok(())  => info!(snapshot = %sid, "fleet: snapshot uploaded"),
                            Err(e)  => warn!(snapshot = %sid, error = %e, "fleet: snapshot upload failed"),
                        }
                    });
                }
                continue;
            }
            // Post-task reflection: record session completion, errors, and healed
            // failures as episodic memory. Best-effort — if memoryd isn't running,
            // skip silently. (The event still flows to the shell + bus below.)
            if let Some(write) = memory_ctx::reflection_write(&event, &sid2, now_ms()) {
                tokio::spawn(async move {
                    let client = kiki_memory::MemoryClient::default_socket();
                    if let Err(e) = client.write(write).await {
                        tracing::debug!(error = %e, "memoryd reflection write skipped");
                    }
                });
            }
            // Fan the event out to connected shell clients (HUD activity stream).
            if let Some(line) = shell_event_line(&event) {
                let _ = shell_events.send(line);
            }
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
        let node_id   = node_id.clone();
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
    // agentd is a long-lived daemon: it keeps serving the MCP hub, control
    // socket, and fleet relay for the whole machine session, independent of any
    // single agent session. The foreground session therefore runs in its own
    // task — when it completes, fails, or has no model yet, the daemon stays up
    // ready for the next one (started on demand via the control socket / fleet).
    info!(session = %session_id, "foreground session starting");
    scheduler.add(session.clone(), SessionPriority::Foreground, None);
    scheduler.set_foreground(&session_id);

    {
        let node_id = node_id.clone();
        tokio::spawn(async move {
            let outcome = harness.run().await;

            // Frozen for cloud migration? Build the bundle from the (now-frozen)
            // context and relay it to the target node, where a cloud agentd
            // resumes it. (The harness can't send it itself — kiki-core doesn't
            // depend on kiki-fleet.)
            if let Some(target) = harness.pending_migration.clone() {
                info!(session = %session_id, target = %target, "migrating session to cloud");
                match send_migration_to(&target, &harness.ctx, &cfg, &node_id).await {
                    Ok(())  => {
                        session.complete_migration(&target);
                        info!(session = %session_id, target = %target, "session migrated");
                    }
                    Err(e) => {
                        error!(session = %session_id, error = %e, "migration send failed");
                        session.fail(format!("migration failed: {e}"));
                    }
                }
                return;
            }

            match outcome {
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
        });
    }

    // ── 11. Stay alive until the OS stops us ──────────────────────────────────
    // Block on SIGTERM (systemd stop) / SIGINT so the daemon keeps serving the
    // MCP hub, control socket, and fleet relay until the machine shuts it down.
    wait_for_shutdown().await;
    info!("agentd received shutdown signal — exiting");
    Ok(())
}

// ── Default agent config ──────────────────────────────────────────────────────

/// Load the node's granted capability set from the policy file written by the
/// package manager (kpkg) when an artifact's capabilities are approved. Supports
/// per-artifact grants and the legacy flat `{ "granted": [...] }` shape (see
/// [`kiki_core::NodePolicy`]). Absent → empty set (deny-by-default); malformed →
/// empty set with a warning.
fn load_granted_caps(path: &str) -> CapabilitySet {
    match kiki_core::NodePolicy::load(path) {
        Ok(policy) => {
            info!(path, granted = policy.grant_count(), "loaded node capability policy");
            policy.to_capability_set()
        }
        Err(e) => {
            warn!(path, error = %e, "invalid policy file — using empty capability set");
            CapabilitySet::new()
        }
    }
}

/// Block until the process receives a termination signal (SIGTERM from a systemd
/// `stop`, or SIGINT from a terminal). Keeps agentd running as a daemon.
async fn wait_for_shutdown() {
    use tokio::signal::unix::{signal, SignalKind};
    let mut term = match signal(SignalKind::terminate()) {
        Ok(s)  => s,
        Err(e) => { error!(error = %e, "failed to install SIGTERM handler"); return; }
    };
    let mut intr = match signal(SignalKind::interrupt()) {
        Ok(s)  => s,
        Err(e) => { error!(error = %e, "failed to install SIGINT handler"); return; }
    };
    tokio::select! {
        _ = term.recv() => {}
        _ = intr.recv() => {}
    }
}

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

async fn run_control_socket(
    path:   String,
    tx:     mpsc::Sender<ControlMessage>,
    events: tokio::sync::broadcast::Sender<String>,
    hub:    Arc<McpHub>,
) {
    use tokio::{
        io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
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
                let mut ev_rx = events.subscribe();
                // Snapshot the current command catalog + installed apps for this shell.
                let commands_line = commands_available_line(&hub);
                let apps_line     = apps_available_line(&hub);
                tokio::spawn(async move {
                    let (read, mut write) = tokio::io::split(stream);
                    // Outbound: send the command catalog + app grid, then stream events.
                    tokio::spawn(async move {
                        let _ = write.write_all(commands_line.as_bytes()).await;
                        let _ = write.write_all(b"\n").await;
                        let _ = write.write_all(apps_line.as_bytes()).await;
                        let _ = write.write_all(b"\n").await;
                        while let Ok(line) = ev_rx.recv().await {
                            if write.write_all(line.as_bytes()).await.is_err()
                                || write.write_all(b"\n").await.is_err()
                            {
                                break;
                            }
                        }
                    });
                    // Inbound: parse ControlMessages from the client.
                    let mut lines = BufReader::new(read).lines();
                    while let Ok(Some(line)) = lines.next_line().await {
                        match serde_json::from_str::<ControlMessage>(&line) {
                            Ok(msg) => {
                                // Persist user corrections to memory immediately
                                // (best-effort), before forwarding to the harness.
                                if let ControlMessage::UserInput { text } = &msg {
                                    if let Some(correction) = memory_ctx::detect_correction(text) {
                                        tokio::spawn(async move {
                                            let client = kiki_memory::MemoryClient::default_socket();
                                            let _ = client
                                                .write(kiki_memory::MemoryWrite::UserCorrection {
                                                    correction,
                                                    context: String::new(),
                                                    ts_ms: now_ms(),
                                                })
                                                .await;
                                        });
                                    }
                                }
                                let _ = tx2.send(msg).await;
                            }
                            Err(e)  => { warn!(error = %e, "invalid control message"); }
                        }
                    }
                });
            }
            Err(e) => { error!(error = %e, "control socket accept error"); break; }
        }
    }
}

/// Map an [`AgentEvent`] to a ShellEvent JSON line for the shell HUD stream.
/// The wire format must match `kiki_shell_core::ShellEvent` (the WM repo mirrors
/// this protocol; the two repos share no code). Returns None for events the
/// shell doesn't render.
fn shell_event_line(event: &kiki_core::harness::AgentEvent) -> Option<String> {
    use kiki_core::harness::AgentEvent;
    let v = match event {
        AgentEvent::Thinking { text } => serde_json::json!({"type":"thinking","text":text}),
        AgentEvent::Content { text } => serde_json::json!({"type":"content","text":text}),
        // AgentEvent has no per-call id; the tool name correlates start/complete.
        AgentEvent::ToolStart { name, .. } => {
            serde_json::json!({"type":"tool_start","id":name,"name":name})
        }
        AgentEvent::ToolComplete { name, success } => {
            serde_json::json!({"type":"tool_complete","id":name,"name":name,"success":success})
        }
        AgentEvent::ModeChange { mode } => {
            serde_json::json!({"type":"mode_changed","mode":control_mode_wire(mode)})
        }
        AgentEvent::Done { session_id, .. } => {
            serde_json::json!({"type":"session_done","id":session_id})
        }
        AgentEvent::TokenUsage { used, limit } => {
            serde_json::json!({"type":"token_usage","used":used,"limit":limit})
        }
        AgentEvent::Error { error } => serde_json::json!({"type":"error","message":error}),
        _ => return None,
    };
    Some(v.to_string())
}

/// Map an agent [`SurfaceSignal`] to a ShellEvent JSON line. Only the
/// human-action signals (approvals/interrupts) are forwarded here; the HUD
/// activity stream comes from AgentEvents. Wire must match `ShellEvent`.
fn surface_signal_line(sig: &SurfaceSignal) -> Option<String> {
    let v = match sig {
        SurfaceSignal::ApprovalRequired { request_id, tool_name, description, .. } => {
            serde_json::json!({
                "type": "interrupt",
                "id": request_id,
                "kind": "confirmation",
                "message": format!("{tool_name}: {description}"),
            })
        }
        SurfaceSignal::Interrupt { interrupt_id, kind, message, .. } => {
            serde_json::json!({
                "type": "interrupt",
                "id": interrupt_id,
                "kind": interrupt_kind_wire(kind),
                "message": message,
            })
        }
        _ => return None,
    };
    Some(v.to_string())
}

/// kiki-core InterruptKind → the snake_case wire string the shell expects.
fn interrupt_kind_wire(kind: &kiki_core::interrupt::InterruptKind) -> &'static str {
    use kiki_core::interrupt::InterruptKind;
    match kind {
        InterruptKind::DecisionRequired => "decision_required",
        InterruptKind::Confirmation => "confirmation",
        InterruptKind::Attention => "attention",
        InterruptKind::Info => "info",
    }
}

/// Build a `commands_available` ShellEvent line from the hub's registered tools.
fn commands_available_line(hub: &McpHub) -> String {
    let commands: Vec<serde_json::Value> = hub
        .all_tools()
        .into_iter()
        .map(|t| {
            serde_json::json!({
                "id": t.name,
                "title": t.name,
                "description": t.description,
                "category": "tool",
            })
        })
        .collect();
    serde_json::json!({ "type": "commands_available", "commands": commands }).to_string()
}

/// Build an `apps_available` ShellEvent line from the hub's registered artifacts.
/// Drives the launcher app grid. Wire must match `kiki_shell_core::ShellEvent`.
fn apps_available_line(hub: &McpHub) -> String {
    let apps: Vec<serde_json::Value> = hub
        .installed_apps()
        .into_iter()
        .map(|a| {
            serde_json::json!({
                "id": a.artifact_id,
                "name": a.artifact_id,
                "version": a.version,
                "tools": a.tools,
            })
        })
        .collect();
    serde_json::json!({ "type": "apps_available", "apps": apps }).to_string()
}

/// ControlMode → the snake_case wire string the shell expects.
fn control_mode_wire(mode: &ControlMode) -> &'static str {
    match mode {
        ControlMode::BypassPermissions => "bypass_permissions",
        ControlMode::AgentMode => "agent_mode",
        ControlMode::AssistedMode => "assisted_mode",
        ControlMode::HumanMode => "human_mode",
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

// ── Session helpers ─────────────────────────────────────────────────────────

/// Build a fresh default session + context.
fn fresh_session(
    session_id: &str,
    mode:       ControlMode,
    state:      Arc<dyn kiki_core::state::StateBackend>,
) -> (Arc<AgentSession>, Context) {
    let agent_id = "kiki-assistant".to_string();
    let session  = Arc::new(AgentSession::new(
        session_id, "Kiki OS Assistant", agent_id.clone(), state.clone(),
    ));
    let mut ctx = Context::new(agent_id, session_id, state);
    ctx.set_mode(mode);
    ctx.max_steps = None; // no step limit
    (session, ctx)
}

/// Resume a session migrated to the cloud: poll the fleet relay for the bundle
/// addressed to this node, pull + restore it, and reconstruct the session +
/// context. Returns `Ok(None)` if no bundle arrives within the poll window (the
/// caller then falls back to a fresh session).
async fn resume_from_cloud(
    resume_id: &str,
    node_id:   &str,
    cfg:       &Config,
    state:     Arc<dyn kiki_core::state::StateBackend>,
) -> anyhow::Result<Option<(Arc<AgentSession>, Context)>> {
    let registry = std::env::var("KIKI_REGISTRY_URL")
        .unwrap_or_else(|_| "https://registry.kiki-os.com".into());

    let mut client = kiki_fleet::FleetClient::new(&cfg.fleet.cloud_url, node_id)
        .with_identity(&cfg.fleet.flavor, env!("CARGO_PKG_VERSION"), None);
    if let Ok(t) = std::env::var("KIKI_FLEET_TOKEN") {
        if !t.is_empty() { client = client.with_token(t); }
    }
    let client = Arc::new(client);
    // Best-effort: announce this cloud node so the relay/dashboard can find it.
    if let Err(e) = client.register_self().await {
        warn!(error = %e, "cloud node registration failed (continuing)");
    }

    let receiver = kiki_fleet::MigrationReceiver::new(client, registry);

    // The source pushes the bundle around the time we boot; poll briefly for it.
    let mut bundle = None;
    for _ in 0..30 {
        match receiver.poll().await {
            Ok(items) => {
                if let Some((_, b)) = items.into_iter().find(|(sid, _)| sid == resume_id) {
                    bundle = Some(b);
                    break;
                }
            }
            Err(e) => warn!(error = %e, "poll migrations failed; retrying"),
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }
    let Some(bundle) = bundle else { return Ok(None) };

    let restored = receiver
        .restore(bundle, state.clone(), CapabilitySet::new())
        .await
        .map_err(|e| anyhow::anyhow!("restore failed: {e}"))?;

    let session = Arc::new(AgentSession::new(
        restored.bundle.session_id.clone(),
        restored.bundle.runtime.session_label.clone(),
        restored.bundle.runtime.agent_id.clone(),
        state,
    ));
    Ok(Some((session, restored.ctx)))
}

/// Build a MigrationBundle from a frozen session's context and relay it to the
/// target node (e.g. `cloud-<session>`). Lives here because kiki-core (the
/// harness) can't depend on kiki-fleet — it signals intent via
/// `Harness::pending_migration` and main performs the transport.
async fn send_migration_to(
    target:  &str,
    ctx:     &Context,
    cfg:     &Config,
    node_id: &str,
) -> anyhow::Result<()> {
    use kiki_core::state::RuntimeSnapshot;

    let registry = std::env::var("KIKI_REGISTRY_URL")
        .unwrap_or_else(|_| "https://registry.kiki-os.com".into());

    let runtime = RuntimeSnapshot {
        agent_id:        ctx.agent_id.clone(),
        session_id:      ctx.session_id.clone(),
        step:            ctx.steps_taken() as u64,
        messages:        ctx.messages.clone(),
        interrupt_queue: ctx.interrupt_log.clone(),
        control_mode:    ctx.control_mode,
        session_label:   ctx.label.clone(),
        scenario:        ctx.scenario.clone(),
        layout:          ctx.layout,
        active_apps:     ctx.active_apps.clone(),
    };

    let bundle = ctx.state.snapshot(runtime).await
        .map_err(|e| anyhow::anyhow!("snapshot: {e}"))?;
    ctx.state.push(&registry).await
        .map_err(|e| anyhow::anyhow!("push: {e}"))?;

    let mut client = kiki_fleet::FleetClient::new(&cfg.fleet.cloud_url, node_id)
        .with_identity(&cfg.fleet.flavor, env!("CARGO_PKG_VERSION"), None);
    let token = std::env::var("KIKI_FLEET_TOKEN").ok().filter(|t| !t.is_empty())
        .or_else(|| kiki_fleet::TokenStore::new("/var/kiki/state/fleet-token").load());
    if let Some(t) = token { client = client.with_token(t); }

    client.send_migration(&ctx.session_id, &bundle, target).await
        .map_err(|e| anyhow::anyhow!("send_migration: {e}"))?;
    Ok(())
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
        E::TokenUsage { used, limit } => serde_json::json!({ "kind": "token_usage", "used": used, "limit": limit }),
        E::Error { error }          => serde_json::json!({ "kind": "error", "error": error }),
        E::SnapshotCaptured { .. }  => serde_json::json!({ "kind": "snapshot_captured" }),
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
        let (publisher, mut inbound) = match kiki_fleet::connect_device(&fleet_url, &relay_session, token.as_deref()).await {
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
                    Some(kiki_fleet::DeviceInbound::MigrateToCloud { session_id }) => {
                        // Freeze + ship this session to a cloud node; the harness
                        // freezes and main's post-run step sends the bundle.
                        info!(session = %session_id, "fleet: migrate-to-cloud requested");
                        let _ = ctrl_tx.send(ControlMessage::MigrateSession {
                            session_id:  session_id.clone(),
                            target_node: format!("cloud-{session_id}"),
                        }).await;
                    }
                    Some(kiki_fleet::DeviceInbound::CaptureSnapshot { snapshot_id }) => {
                        // Capture a point-in-time bundle (no freeze); the harness
                        // builds it and main's event drain uploads it.
                        info!(snapshot = %snapshot_id, "fleet: capture-snapshot requested");
                        let _ = ctrl_tx.send(ControlMessage::CaptureSnapshot { snapshot_id }).await;
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
