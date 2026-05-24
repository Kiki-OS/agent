//! Integration: an app connects, handshakes, registers a tool, and a host-side
//! `hub.call` round-trips through the real MCP server socket to the app and back.
//! Also exercises the launcher catalog: `installed_apps()` + the change notifier.

use std::sync::Arc;
use std::time::Duration;

use kiki_mcp::{McpHub, McpServer};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

/// A minimal fake app: handshake declaring one tool, then serve `tools/call`
/// requests by uppercasing the `text` argument until the socket closes.
async fn run_fake_app(socket: String, artifact_id: String) {
    let stream = UnixStream::connect(&socket).await.expect("app connect");
    let (r, mut w) = stream.into_split();
    let mut lines = BufReader::new(r).lines();

    let init = json!({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "artifactId": artifact_id,
            "version": "1.0.0",
            "tools": [{
                "name": "echo.upper",
                "description": "Uppercase the given text.",
                "input_schema": { "type": "object", "properties": { "text": { "type": "string" } } }
            }]
        }
    });
    w.write_all(format!("{init}\n").as_bytes()).await.unwrap();
    let _ = lines.next_line().await.unwrap(); // initialize reply

    while let Ok(Some(l)) = lines.next_line().await {
        let Ok(msg): Result<Value, _> = serde_json::from_str(&l) else { continue; };
        if msg["method"] == "tools/call" {
            let id = msg["id"].clone();
            let text = msg["params"]["arguments"]["text"].as_str().unwrap_or("");
            let reply = json!({
                "jsonrpc": "2.0", "id": id,
                "result": { "upper": text.to_uppercase() }
            });
            w.write_all(format!("{reply}\n").as_bytes()).await.unwrap();
        }
    }
}

#[tokio::test]
async fn app_registers_and_tool_call_roundtrips() {
    let socket = "/tmp/kiki-mcp-rt-roundtrip.sock";
    let _ = std::fs::remove_file(socket);

    let (hub_inner, mut catalog_rx) = McpHub::new().with_catalog_notifier();
    let hub = Arc::new(hub_inner);
    let _handle = McpServer::new(hub.clone(), socket.to_string())
        .serve()
        .await
        .expect("serve");

    tokio::spawn(run_fake_app(socket.into(), "io.kiki.echo".into()));

    // The catalog notifier fires on register — wait for it instead of sleeping.
    tokio::time::timeout(Duration::from_secs(2), catalog_rx.recv())
        .await
        .expect("registration notifier timed out")
        .expect("notifier closed");

    // Launcher view: the app shows up with its declared tool.
    let apps = hub.installed_apps();
    assert_eq!(apps.len(), 1);
    assert_eq!(apps[0].artifact_id, "io.kiki.echo");
    assert_eq!(apps[0].tools, vec!["echo.upper".to_string()]);

    // Host → app tool call routes over the socket and returns the app's result.
    let out = tokio::time::timeout(Duration::from_secs(2), hub.call("echo.upper", json!({ "text": "hi" })))
        .await
        .expect("tool call timed out")
        .expect("tool call failed");
    assert_eq!(out["upper"], "HI");

    let _ = std::fs::remove_file(socket);
}
