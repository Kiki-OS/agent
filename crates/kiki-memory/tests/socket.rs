//! Integration: the memoryd socket protocol round-trips over a real Unix socket
//! using [`MemoryClient`]. The server loop here mirrors `memoryd`'s
//! `handle_connection` (same store dispatch + newline-JSON framing), so this
//! proves the client ↔ daemon contract end-to-end.
#![cfg(feature = "client")]

use std::sync::Arc;

use kiki_memory::{
    EpisodeEvent, MemoryClient, MemoryLayer, MemoryQuery, MemoryRequest, MemoryResult, MemoryStore,
    MemoryWrite,
};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixListener;

async fn spawn_memoryd(name: &str, store: Arc<MemoryStore>) -> String {
    // Unique per test (the pid is shared across tests in one binary).
    let socket = format!("/tmp/kiki-memoryd-test-{}-{name}.sock", std::process::id());
    let _ = std::fs::remove_file(&socket);
    let listener = UnixListener::bind(&socket).unwrap();
    tokio::spawn(async move {
        while let Ok((stream, _)) = listener.accept().await {
            let store = store.clone();
            tokio::spawn(async move {
                let (r, mut w) = stream.into_split();
                let mut lines = BufReader::new(r).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    if line.trim().is_empty() {
                        continue;
                    }
                    let result = match serde_json::from_str::<MemoryRequest>(&line) {
                        Ok(MemoryRequest::Query(q)) => store.query(q),
                        Ok(MemoryRequest::Write(wr)) => store.write(wr),
                        Err(e) => MemoryResult::Error { message: e.to_string() },
                    };
                    let mut out = serde_json::to_string(&result).unwrap();
                    out.push('\n');
                    if w.write_all(out.as_bytes()).await.is_err() {
                        break;
                    }
                }
            });
        }
    });
    socket
}

#[tokio::test]
async fn write_then_search_over_socket() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(MemoryStore::open(dir.path()).unwrap());
    let socket = spawn_memoryd("search", store).await;

    let client = MemoryClient::new(socket.clone());

    // Write an episode through the socket.
    let ack = client
        .write(MemoryWrite::Episode {
            event: EpisodeEvent {
                id: "e1".into(),
                kind: "session_done".into(),
                session_id: "s1".into(),
                summary: "compiled the kernel successfully".into(),
                outcome: "ok".into(),
                ts_ms: 1_000,
                important: false,
            },
        })
        .await
        .unwrap();
    assert_eq!(ack, MemoryResult::Ok);

    // Search it back through the socket.
    let res = client
        .query(MemoryQuery::Search { query: "kernel".into(), layers: vec![MemoryLayer::Episodic], limit: 5 })
        .await
        .unwrap();
    let MemoryResult::Hits { hits } = res else { panic!("expected hits, got {res:?}") };
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "e1");
    assert!(hits[0].content.contains("kernel"));

    let _ = std::fs::remove_file(&socket);
}

#[tokio::test]
async fn correction_write_and_profile_query_over_socket() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(MemoryStore::open(dir.path()).unwrap());
    let socket = spawn_memoryd("correction", store).await;
    let client = MemoryClient::new(socket.clone());

    client
        .write(MemoryWrite::UserCorrection {
            correction: "never force push to main".into(),
            context: "git".into(),
            ts_ms: 42,
        })
        .await
        .unwrap();

    let res = client.query(MemoryQuery::Corrections { limit: 10 }).await.unwrap();
    let MemoryResult::Hits { hits } = res else { panic!("expected hits") };
    assert_eq!(hits.len(), 1);
    assert!(hits[0].content.contains("force push"));

    // Default profile comes back with safe privacy defaults.
    let res = client.query(MemoryQuery::UserProfile).await.unwrap();
    let MemoryResult::Profile { profile } = res else { panic!("expected profile") };
    assert!(!profile.privacy.sync_to_cloud);

    let _ = std::fs::remove_file(&socket);
}
