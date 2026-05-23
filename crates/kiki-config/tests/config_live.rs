//! Live integration: the `kiki-config` sync client against the deployed
//! `kiki-cloud` config worker (`kiki-beta-config`).
//!
//! Real cross-repo, real-backend test exercising the actual [`SyncClient`]:
//!   - `fetch_kdf_params` returns NotFound before init, then the stored params,
//!   - `put_secret` performs CAS create (rev 1) and update (rev 2),
//!   - a stale `if_match` is rejected as a CAS conflict carrying the current rev,
//!   - `pull_since` returns the written values with the correct head revision,
//!   - `put_preference` writes a plaintext-JSON preference.
//!
//! The KDF-params *set* step uses raw HTTP (the Rust client only fetches them).
//!
//! Gated behind `KIKI_CLOUD_TEST=1`. Run with:
//! ```sh
//! KIKI_CLOUD_TEST=1 \
//! KIKI_AUTH_URL=https://auth-preview.kiki-os.com \
//! KIKI_CONFIG_URL=https://config-preview.kiki-os.com \
//!   cargo test -p kiki-config --test config_live -- --nocapture
//! ```

use kiki_config::crypto::Sealed;
use kiki_config::{ConfigEndpoint, Revision, Scope, SyncClient, SyncError};

fn enabled() -> bool {
    std::env::var("KIKI_CLOUD_TEST").as_deref() == Ok("1")
}
fn auth_url() -> String {
    std::env::var("KIKI_AUTH_URL").unwrap_or_else(|_| "https://auth-preview.kiki-os.com".into())
}
fn config_url() -> String {
    std::env::var("KIKI_CONFIG_URL").unwrap_or_else(|_| "https://config-preview.kiki-os.com".into())
}

#[tokio::test]
async fn config_sync_against_live_worker() {
    if !enabled() {
        eprintln!("skipped: set KIKI_CLOUD_TEST=1 to run");
        return;
    }
    let http = reqwest::Client::new();
    let auth = auth_url();
    let config = config_url();

    // ── 1. Sign up → session bearer. ──────────────────────────────────────────
    let email = format!(
        "rust-config-{}-{}@kiki-test.local",
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis(),
        std::process::id(),
    );
    let signup: serde_json::Value = http
        .post(format!("{auth}/api/auth/sign-up/email"))
        .header("origin", &auth)
        .json(&serde_json::json!({ "email": email, "password": "TestPassword!1", "name": "Rust Config IT" }))
        .send().await.expect("signup")
        .json().await.expect("signup json");
    let session_token = signup["token"].as_str()
        .unwrap_or_else(|| panic!("no session token: {signup}")).to_string();

    let client = SyncClient::new(ConfigEndpoint {
        base_url:     config.clone(),
        bearer_token: session_token.clone(),
    }).expect("client");

    // ── 2. KDF params: NotFound before init. ──────────────────────────────────
    match client.fetch_kdf_params().await {
        Err(SyncError::NotFound(_)) => println!("[config] kdf correctly uninitialized"),
        other => panic!("expected NotFound before kdf init, got {other:?}"),
    }

    // ── 3. Initialize KDF params (raw HTTP — client only fetches). ────────────
    let set = http
        .post(format!("{config}/v1/config/secrets/_kdf"))
        .bearer_auth(&session_token)
        .json(&serde_json::json!({ "salt_b64": "c2FsdHktc2FsdC0xMjM=", "m_cost_kib": 65536, "t_cost": 3, "parallelism": 4 }))
        .send().await.expect("set kdf");
    assert_eq!(set.status().as_u16(), 200, "kdf set failed: {}", set.text().await.unwrap_or_default());

    // ── 4. Now fetch returns the stored params. ───────────────────────────────
    let kdf = client.fetch_kdf_params().await.expect("fetch kdf");
    println!("[config] kdf m_cost_kib={} t_cost={} parallelism={}", kdf.m_cost_kib, kdf.t_cost, kdf.parallelism);
    assert_eq!(kdf.m_cost_kib, 65536);
    assert_eq!(kdf.t_cost, 3);
    assert_eq!(kdf.parallelism, 4);

    // ── 5. CAS create a secret, then update it. ───────────────────────────────
    let sealed_v1 = Sealed { ciphertext_b64: "Y2lwaGVydGV4dC12MQ==".into(), nonce_b64: "bm9uY2UtdjE=".into() };
    let r1 = client.put_secret("openai.api_key", &sealed_v1, Revision::ZERO).await.expect("create secret");
    println!("[config] created secret at revision {:?}", r1);
    assert_eq!(r1, Revision(1));

    let sealed_v2 = Sealed { ciphertext_b64: "Y2lwaGVydGV4dC12Mg==".into(), nonce_b64: "bm9uY2UtdjI=".into() };
    let r2 = client.put_secret("openai.api_key", &sealed_v2, Revision(1)).await.expect("update secret");
    assert_eq!(r2, Revision(2));

    // ── 6. Stale if_match → CAS conflict carrying the current revision. ───────
    match client.put_secret("openai.api_key", &sealed_v1, Revision::ZERO).await {
        Err(SyncError::Conflict { current }) => {
            println!("[config] stale write correctly rejected; current={current:?}");
            assert_eq!(current, Revision(2));
        }
        other => panic!("expected CAS conflict, got {other:?}"),
    }

    // ── 7. pull_since returns the written secret at head revision 2. ──────────
    let page = client.pull_since(Scope::Secrets, Revision::ZERO).await.expect("pull");
    println!("[config] pulled {} value(s), head={:?}", page.values.len(), page.head_revision);
    assert_eq!(page.head_revision, Revision(2));
    let v = page.values.iter().find(|v| v.key == "openai.api_key")
        .expect("secret must appear in pull");
    assert_eq!(v.revision, Revision(2));
    // Payload round-trips the sealed blob.
    assert_eq!(v.payload["ciphertext_b64"], "Y2lwaGVydGV4dC12Mg==");

    // ── 8. Plaintext preference write. ────────────────────────────────────────
    let pref_rev = client
        .put_preference("ui.theme", serde_json::json!({ "mode": "dark" }), Revision::ZERO)
        .await.expect("put preference");
    assert_eq!(pref_rev, Revision(1));
    println!("[config] preference written at revision {:?}", pref_rev);
}
