//! `kpkg` — CLI front-end for the device-side artifact manager (kiki-pkg).
//!
//! Resolves, verifies, and installs registry artifacts into `/var/kiki/apps`,
//! persisting capability grants to the node policy file agentd loads.

use clap::{Parser, Subcommand};
use kiki_memory::{MemoryLayer, MemoryStore};
use kiki_pkg::{ArtifactManager, InstallRequest};
use kiki_registry_client::TrustRoot;

#[derive(Parser)]
#[command(name = "kpkg", about = "Kiki package manager (device-side artifact manager)")]
struct Cli {
    /// Directory where installed artifacts live.
    #[arg(long, default_value = "/var/kiki/apps")]
    apps_dir: std::path::PathBuf,
    /// Registry origin to resolve artifacts from.
    #[arg(long, default_value = "https://registry.kiki-os.com")]
    registry: String,
    /// Directory of trusted signing keys (`*.ed25519`, hex public keys).
    #[arg(long, default_value = "/etc/kiki/trust")]
    trust_dir: std::path::PathBuf,
    /// Node policy file where capability grants are persisted for agentd.
    #[arg(long, default_value = "/etc/kiki/policy.json")]
    policy_file: std::path::PathBuf,
    /// Fetch the trust root from the registry instead of `--trust-dir`. INSECURE
    /// (trust-on-first-use over TLS) — for dev/bootstrap only.
    #[arg(long)]
    fetch_trust: bool,
    /// Node id to report installs under (also reads `KIKI_NODE_ID`).
    #[arg(long, env = "KIKI_NODE_ID")]
    node_id: Option<String>,
    /// Bearer token for install reporting (also reads `KIKI_REGISTRY_TOKEN`).
    #[arg(long, env = "KIKI_REGISTRY_TOKEN")]
    token: Option<String>,
    /// Durable memory store root (for `kpkg memory`).
    #[arg(long, default_value = "/var/kiki/memory", env = "KIKI_MEMORY_DIR")]
    memory_dir: std::path::PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Search the registry catalog (browse before installing).
    Search {
        /// Free-text query matched against name/description.
        query: Option<String>,
        /// Scope to an artifact type: app | model | provider | component | …
        #[arg(long = "type")]
        type_filter: Option<String>,
    },
    /// Show a single artifact's catalog entry by full id.
    Info { id: String },
    /// Install an artifact by id (`<ns>/<name>`, optionally pinned to a version).
    Install { id: String, #[arg(long)] version: Option<String> },
    /// Remove an installed artifact.
    Remove { id: String },
    /// Update an installed artifact to the latest version.
    Update { id: String },
    /// List installed artifacts.
    List,
    /// Inspect / control on-device memory (all local; nothing leaves the device).
    Memory {
        #[command(subcommand)]
        cmd: MemoryCmd,
    },
}

#[derive(Subcommand)]
enum MemoryCmd {
    /// Show stored memories (optionally scoped to a layer / time window).
    Inspect {
        /// Restrict to one layer: episodic | semantic | procedural | identity.
        #[arg(long)]
        layer: Option<String>,
        /// Only entries from the last N days.
        #[arg(long)]
        since_days: Option<u64>,
    },
    /// Delete a single memory entry by id/key.
    Delete { id: String },
    /// Erase ALL memory (requires --yes to confirm).
    Clear {
        #[arg(long)]
        yes: bool,
    },
    /// Export all memory to a JSON snapshot file.
    Export { #[arg(long)] path: std::path::PathBuf },
    /// Import a JSON snapshot (merges into existing memory).
    Import { #[arg(long)] path: std::path::PathBuf },
}

fn parse_layer(s: &str) -> anyhow::Result<MemoryLayer> {
    Ok(match s {
        "episodic"   => MemoryLayer::Episodic,
        "semantic"   => MemoryLayer::Semantic,
        "procedural" => MemoryLayer::Procedural,
        "identity"   => MemoryLayer::Identity,
        other => anyhow::bail!("unknown layer '{other}' (episodic|semantic|procedural|identity)"),
    })
}

fn run_memory(memory_dir: std::path::PathBuf, cmd: MemoryCmd) -> anyhow::Result<()> {
    use kiki_memory::{MemoryQuery, MemoryResult, MemorySnapshot};
    let store = MemoryStore::open(&memory_dir)?;
    match cmd {
        MemoryCmd::Inspect { layer, since_days } => {
            let layers = match layer {
                Some(l) => vec![parse_layer(&l)?],
                None => vec![],
            };
            // identity has no "recent" list — show the profile separately.
            let show_identity = layers.is_empty() || layers.contains(&MemoryLayer::Identity);
            if show_identity {
                if let MemoryResult::Profile { profile } = store.query(MemoryQuery::UserProfile) {
                    println!("[identity] {}", serde_json::to_string(&profile)?);
                }
            }
            let data_layers: Vec<MemoryLayer> =
                layers.into_iter().filter(|l| *l != MemoryLayer::Identity).collect();
            if data_layers.is_empty() && !show_identity {
                return Ok(());
            }
            let since_ms = since_days
                .map(|d| now_ms().saturating_sub(d * 86_400_000))
                .unwrap_or(0);
            if let MemoryResult::Hits { hits } =
                store.query(MemoryQuery::Recent { since_ms, layers: data_layers })
            {
                for h in hits {
                    println!("[{:?}] {}\t{}", h.layer, h.id, h.content.replace('\n', " "));
                }
            }
        }
        MemoryCmd::Delete { id } => {
            if store.delete(&id)? {
                println!("deleted {id}");
            } else {
                println!("no entry with id {id}");
            }
        }
        MemoryCmd::Clear { yes } => {
            if !yes {
                anyhow::bail!("refusing to erase all memory without --yes");
            }
            store.clear()?;
            println!("memory cleared");
        }
        MemoryCmd::Export { path } => {
            let snap = store.export_snapshot()?;
            std::fs::write(&path, serde_json::to_vec_pretty(&snap)?)?;
            println!("exported memory to {}", path.display());
        }
        MemoryCmd::Import { path } => {
            let bytes = std::fs::read(&path)?;
            let snap: MemorySnapshot = serde_json::from_slice(&bytes)?;
            store.import_snapshot(&snap)?;
            println!("imported memory from {}", path.display());
        }
    }
    Ok(())
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Memory commands are purely local — no registry/trust setup needed.
    let artifact_cmd = match cli.cmd {
        Cmd::Memory { cmd } => return run_memory(cli.memory_dir, cmd),
        other => other,
    };

    // Search/info are read-only catalog browses — they verify no signatures, so a
    // missing trust dir shouldn't block them. Install/update/remove still
    // fail-closed on an empty trust root (enforced inside the manager).
    let read_only = matches!(artifact_cmd, Cmd::Search { .. } | Cmd::Info { .. });
    let trust = if cli.fetch_trust {
        eprintln!("warning: fetching trust root over the wire (insecure; bootstrap only)");
        kiki_registry_client::RegistryClient::fetch_trust(&cli.registry).await?
    } else {
        match TrustRoot::from_dir(&cli.trust_dir) {
            Ok(t) => t,
            Err(_) if read_only => TrustRoot::new(),
            Err(e) => return Err(e.into()),
        }
    };

    let mut mgr = ArtifactManager::new(cli.apps_dir, &cli.registry)
        .with_trust(trust)
        .with_policy_path(cli.policy_file);
    if let (Some(node), Some(token)) = (cli.node_id.clone(), cli.token.clone()) {
        mgr = mgr.with_identity(node, token);
    }

    match artifact_cmd {
        Cmd::Search { query, type_filter } => {
            let items = mgr.search(query.as_deref(), type_filter.as_deref()).await?;
            if items.is_empty() {
                println!("no artifacts found");
            }
            for a in items {
                let desc = a.description.unwrap_or_default();
                println!("{}\t{}\t{}\t{}", a.id, a.version, a.artifact_type, desc);
            }
        }
        Cmd::Info { id } => {
            let a = mgr.info(&id).await?;
            println!("id:          {}", a.id);
            println!("name:        {}", a.name);
            println!("version:     {}", a.version);
            println!("type:        {}", a.artifact_type);
            println!("license:     {}", a.license);
            if let Some(d) = a.description {
                println!("description: {d}");
            }
        }
        Cmd::Install { id, version } => {
            let a = mgr.install(InstallRequest { id, version }).await?;
            println!("installed {} {}", a.id, a.path.display());
        }
        Cmd::Remove { id } => {
            mgr.remove(&id).await?;
            println!("removed {id}");
        }
        Cmd::Update { id } => {
            let a = mgr.update(&id).await?;
            println!("updated {} {}", a.id, a.version);
        }
        Cmd::List => {
            for a in mgr.list()? {
                println!("{}\t{}\t{}", a.id, a.version, a.path.display());
            }
        }
        Cmd::Memory { .. } => unreachable!("handled above"),
    }
    Ok(())
}
