//! Async client for the memoryd socket (`/run/kiki/memory.sock`).
//!
//! Newline-delimited JSON: one [`MemoryRequest`] per line → one [`MemoryResult`]
//! line back. agentd uses this to read/write memory without depending on the
//! daemon binary. Behind the `client` feature so the pure core stays tokio-free.

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

use crate::{MemoryQuery, MemoryRequest, MemoryResult, MemoryWrite};

/// A connection to memoryd. One request/response per call (the daemon handles
/// each line independently), so this is cheap to hold and reconnect.
pub struct MemoryClient {
    path: String,
}

impl MemoryClient {
    pub fn new(socket_path: impl Into<String>) -> Self {
        Self { path: socket_path.into() }
    }

    /// Default socket location.
    pub fn default_socket() -> Self {
        Self::new(
            std::env::var("KIKI_MEMORY_SOCKET").unwrap_or_else(|_| "/run/kiki/memory.sock".into()),
        )
    }

    async fn round_trip(&self, req: &MemoryRequest) -> std::io::Result<MemoryResult> {
        let stream = UnixStream::connect(&self.path).await?;
        let (r, mut w) = stream.into_split();
        let mut line = serde_json::to_string(req)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        line.push('\n');
        w.write_all(line.as_bytes()).await?;
        w.flush().await?;

        let mut reader = BufReader::new(r).lines();
        match reader.next_line().await? {
            Some(resp) => serde_json::from_str(&resp)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
            None => Ok(MemoryResult::Error { message: "memoryd closed without reply".into() }),
        }
    }

    pub async fn query(&self, q: MemoryQuery) -> std::io::Result<MemoryResult> {
        self.round_trip(&MemoryRequest::Query(q)).await
    }

    pub async fn write(&self, w: MemoryWrite) -> std::io::Result<MemoryResult> {
        self.round_trip(&MemoryRequest::Write(w)).await
    }
}
