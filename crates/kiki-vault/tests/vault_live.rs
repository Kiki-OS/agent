//! Live integration: the `kiki-agent` vault client against the deployed
//! `kiki-cloud` vault worker.
//!
//! Real cross-repo, real-backend test. It signs up a user on the live auth
//! worker, mints a capability token on the live vault worker, then drives the
//! actual [`VaultClient`] (`write`, `read`, `head`) over HTTPS:
//!   - CAS create (`if_match = 0`) returns revision 1,
//!   - read returns byte-identical content,
//!   - head reports the same revision and sha256.
//!
//! Gated behind `KIKI_CLOUD_TEST=1`. Run with:
//! ```sh
//! KIKI_CLOUD_TEST=1 \
//! KIKI_AUTH_URL=https://auth-preview.kiki-os.com \
//! KIKI_VAULT_URL=https://vault-preview.kiki-os.com \
//!   cargo test -p kiki-vault --test vault_live -- --nocapture
//! ```

use kiki_vault::{CapabilityToken, Revision, VaultClient, VaultScope, VaultUri};

fn enabled() -> bool {
    std::env::var("KIKI_CLOUD_TEST").as_deref() == Ok("1")
}
fn auth_url() -> String {
    std::env::var("KIKI_AUTH_URL").unwrap_or_else(|_| "https://auth-preview.kiki-os.com".into())
}
fn vault_url() -> String {
    std::env::var("KIKI_VAULT_URL").unwrap_or_else(|_| "https://vault-preview.kiki-os.com".into())
}

#[tokio::test]
async fn vault_write_read_against_live_worker() {
    if !enabled() {
        eprintln!("skipped: set KIKI_CLOUD_TEST=1 to run");
        return;
    }
    let http = reqwest::Client::new();
    let auth = auth_url();
    let vault = vault_url();

    // ── 1. Sign up → session bearer + user id. ────────────────────────────────
    let email = format!(
        "rust-vault-{}-{}@kiki-test.local",
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis(),
        std::process::id(),
    );
    let signup: serde_json::Value = http
        .post(format!("{auth}/api/auth/sign-up/email"))
        .header("origin", &auth)
        .json(&serde_json::json!({ "email": email, "password": "TestPassword!1", "name": "Rust Vault IT" }))
        .send().await.expect("signup")
        .json().await.expect("signup json");
    let user_id = signup["user"]["id"].as_str()
        .unwrap_or_else(|| panic!("no user id: {signup}")).to_string();
    let session_token = signup["token"].as_str()
        .unwrap_or_else(|| panic!("no session token: {signup}")).to_string();
    println!("[vault] user_id={user_id}");

    // ── 2. Mint a capability token (session-bearer auth). ─────────────────────
    // The worker's _token response shape is exactly CapabilityToken.
    let cap: CapabilityToken = http
        .post(format!("{vault}/v1/vault/personal/{user_id}/_token"))
        .bearer_auth(&session_token)
        .json(&serde_json::json!({
            "paths": ["**"],
            "rights": { "read": true, "write": true, "watch": true },
            "ttl_seconds": 600,
        }))
        .send().await.expect("mint token")
        .json().await.expect("token json");
    println!("[vault] cap token paths_allowed={:?}", cap.paths_allowed);
    assert!(!cap.jwt.is_empty(), "capability jwt must be non-empty");

    // ── 3. Real VaultClient: write → read → head. ─────────────────────────────
    let client = VaultClient::new(&vault, cap, None).expect("client");
    let uri = VaultUri {
        scope:    VaultScope::Personal,
        owner_id: user_id.clone(),
        path:     "notes/rust-integration.md".into(),
    };
    let payload = b"hello from the kiki-agent vault client".to_vec();

    let meta = client
        .write(&uri, payload.clone(), Revision::ZERO, Some("text/markdown"))
        .await
        .expect("write");
    println!("[vault] wrote revision={:?} sha256={}", meta.revision, meta.sha256);
    assert_eq!(meta.revision, Revision(1), "create-only write yields revision 1");
    assert!(!meta.sha256.is_empty());

    let (read_meta, bytes) = client.read(&uri).await.expect("read");
    println!("[vault] read {} bytes, revision={:?}", bytes.len(), read_meta.revision);
    assert_eq!(bytes, payload, "read bytes must match what we wrote");
    assert_eq!(read_meta.revision, Revision(1));
    assert_eq!(read_meta.sha256, meta.sha256, "sha256 must be stable across write/read");

    let head = client.head(&uri).await.expect("head");
    assert_eq!(head.revision, Revision(1));
    assert_eq!(head.size, payload.len() as u64, "head size must match payload length");
}
