//! Integration: an app calls `net.fetch` over its MCP connection; the broker
//! enforces the calling app's egress allowlist using the connection identity.

use std::sync::Arc;

use kiki_mcp::{McpHub, McpServer};
use kiki_net::{EgressBroker, HostPort};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Connect, handshake as `artifact_id`, send one net.fetch, return the reply.
async fn fetch_via_mcp(socket: &str, artifact_id: &str, url: &str) -> serde_json::Value {
    let stream = UnixStream::connect(socket).await.expect("connect");
    let (r, mut w) = stream.into_split();
    let mut lines = BufReader::new(r).lines();

    let init = serde_json::json!({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": { "artifactId": artifact_id, "version": "0.1.0", "tools": [] }
    });
    w.write_all(format!("{init}\n").as_bytes()).await.unwrap();
    let _init_reply = lines.next_line().await.unwrap().expect("init reply");

    let req = serde_json::json!({
        "jsonrpc": "2.0", "id": 2, "method": "net.fetch",
        "params": { "method": "GET", "url": url }
    });
    w.write_all(format!("{req}\n").as_bytes()).await.unwrap();
    let line = lines.next_line().await.unwrap().expect("fetch reply");
    serde_json::from_str(&line).unwrap()
}

async fn start_server(name: &str, broker: Arc<EgressBroker>) -> String {
    // Short path under /tmp + a per-test name keeps it unique and within the
    // macOS Unix-socket path length limit.
    let socket = format!("/tmp/kiki-mcp-test-{name}.sock");
    let _ = std::fs::remove_file(&socket);
    let hub = Arc::new(McpHub::new());
    McpServer::new(hub, socket.clone())
        .with_broker(broker)
        .serve()
        .await
        .expect("serve");
    socket
}

#[tokio::test]
async fn app_fetches_allowed_host_through_broker() {
    let upstream = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/data"))
        .respond_with(ResponseTemplate::new(200).set_body_string("{\"ok\":true}"))
        .mount(&upstream)
        .await;
    let host = upstream.address().ip().to_string();
    let port = upstream.address().port();

    let mut broker = EgressBroker::new();
    broker.allow("io.kiki.weather", vec![HostPort { host: host.clone(), port }]);
    let socket = start_server("allowed", Arc::new(broker)).await;

    let reply = fetch_via_mcp(&socket, "io.kiki.weather", &format!("{}/data", upstream.uri())).await;
    assert!(reply["error"].is_null(), "unexpected error: {reply}");
    assert_eq!(reply["result"]["status"], 200);
    assert!(reply["result"]["body"].as_str().unwrap().contains("ok"));
}

#[tokio::test]
async fn app_denied_when_host_not_in_its_allowlist() {
    let upstream = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/x"))
        .respond_with(ResponseTemplate::new(200).set_body_string("nope"))
        .mount(&upstream)
        .await;

    // Broker knows a different app; the calling app has no allowlist → denied.
    let mut broker = EgressBroker::new();
    broker.allow("someone.else", vec![HostPort { host: "x".into(), port: 1 }]);
    let socket = start_server("denied", Arc::new(broker)).await;

    let reply = fetch_via_mcp(&socket, "io.kiki.intruder", &format!("{}/x", upstream.uri())).await;
    assert!(!reply["error"].is_null(), "expected a denied error, got: {reply}");
}
