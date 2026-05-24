//! memoryd — the Kiki OS memory daemon.
//!
//! Serves the four-layer [`MemoryStore`](kiki_memory::MemoryStore) over a Unix
//! socket (`/run/kiki/memory.sock`). Newline-delimited JSON: each line is one
//! [`MemoryRequest`](kiki_memory::MemoryRequest) (a query or a write), answered
//! with one [`MemoryResult`](kiki_memory::MemoryResult) line. On-device only —
//! the daemon never opens a network socket.
//!
//! Config via env:
//!   KIKI_MEMORY_DIR     durable store root   (default /var/kiki/memory)
//!   KIKI_MEMORY_SOCKET  control socket path  (default /run/kiki/memory.sock)

use std::sync::Arc;

use kiki_memory::{MemoryRequest, MemoryResult, MemoryStore};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixListener;
use tracing::{error, info, warn};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().init();

    let dir = std::env::var("KIKI_MEMORY_DIR").unwrap_or_else(|_| "/var/kiki/memory".into());
    let socket = std::env::var("KIKI_MEMORY_SOCKET").unwrap_or_else(|_| "/run/kiki/memory.sock".into());

    let store = match MemoryStore::open(&dir) {
        Ok(s) => Arc::new(s),
        Err(e) => {
            error!(dir = %dir, error = %e, "failed to open memory store");
            std::process::exit(1);
        }
    };

    if let Some(parent) = std::path::Path::new(&socket).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if std::path::Path::new(&socket).exists() {
        let _ = std::fs::remove_file(&socket);
    }

    let listener = match UnixListener::bind(&socket) {
        Ok(l) => l,
        Err(e) => {
            error!(socket = %socket, error = %e, "failed to bind memory socket");
            std::process::exit(1);
        }
    };
    info!(socket = %socket, dir = %dir, "memoryd listening");

    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                let store = store.clone();
                tokio::spawn(handle_connection(stream, store));
            }
            Err(e) => {
                error!(error = %e, "accept error");
                break;
            }
        }
    }
}

async fn handle_connection(stream: tokio::net::UnixStream, store: Arc<MemoryStore>) {
    let (read, mut write) = stream.into_split();
    let mut lines = BufReader::new(read).lines();

    while let Ok(Some(line)) = lines.next_line().await {
        if line.trim().is_empty() {
            continue;
        }
        let result = match serde_json::from_str::<MemoryRequest>(&line) {
            Ok(MemoryRequest::Query(q)) => store.query(q),
            Ok(MemoryRequest::Write(w)) => store.write(w),
            Err(e) => {
                warn!(error = %e, "invalid memory request");
                MemoryResult::Error { message: format!("invalid request: {e}") }
            }
        };
        let mut out = match serde_json::to_string(&result) {
            Ok(s) => s,
            Err(e) => {
                error!(error = %e, "failed to serialize result");
                continue;
            }
        };
        out.push('\n');
        if write.write_all(out.as_bytes()).await.is_err() || write.flush().await.is_err() {
            break;
        }
    }
}
