//! Live integration: the `kiki-fleet` client against the deployed `kiki-cloud`
//! fleet worker (`kiki-beta-fleet`).
//!
//! Real cross-repo, real-backend test of the migration relay. It drives the
//! actual [`FleetClient`] through the full session-handoff protocol:
//!   1. source + target nodes register,
//!   2. the target's pending queue starts empty,
//!   3. the source posts a `MigrationBundle` (staged in R2 via the worker),
//!   4. the target polls and gets the bundle back — proving the bundle's serde
//!      round-trips Rust → JSON → R2 → JSON → Rust intact,
//!   5. the target completes the migration and the queue drains.
//!
//! Gated behind `KIKI_CLOUD_TEST=1`. Run with:
//! ```sh
//! KIKI_CLOUD_TEST=1 KIKI_FLEET_URL=https://fleet-preview.kiki-os.com \
//!   cargo test -p kiki-fleet --test fleet_live -- --nocapture
//! ```

use kiki_core::state::{MigrationBundle, OstreeCheckpoint, RuntimeSnapshot};
use kiki_core::{context::ControlMode, surface::SessionLayout};
use kiki_fleet::FleetClient;

fn enabled() -> bool {
    std::env::var("KIKI_CLOUD_TEST").as_deref() == Ok("1")
}
fn fleet_url() -> String {
    std::env::var("KIKI_FLEET_URL").unwrap_or_else(|_| "https://fleet-preview.kiki-os.com".into())
}

fn sample_bundle(session_id: &str) -> MigrationBundle {
    let runtime = RuntimeSnapshot {
        agent_id:        "kiki-assistant".into(),
        session_id:      session_id.into(),
        step:            7,
        messages:        Vec::new(),
        interrupt_queue: Vec::new(),
        control_mode:    ControlMode::default(),
        session_label:   "Editing Monday's video".into(),
        scenario:        None,
        layout:          SessionLayout::default(),
        active_apps:     vec!["notes".into(), "media-player".into()],
    };
    MigrationBundle {
        bundle_id:     MigrationBundle::bundle_id(session_id, runtime.step),
        session_id:    session_id.into(),
        checkpoint:    OstreeCheckpoint {
            agent_id:   "kiki-assistant".into(),
            session_id: session_id.into(),
            step:       7,
            ref_hash:   None,
            message:    "pre-migration checkpoint".into(),
        },
        runtime,
        artifact_refs: Vec::new(),
        created_at_ms: 1_716_400_000_000,
    }
}

#[tokio::test]
async fn fleet_migration_relay_round_trip() {
    if !enabled() {
        eprintln!("skipped: set KIKI_CLOUD_TEST=1 to run");
        return;
    }
    let base = fleet_url();
    let nonce = format!(
        "{}-{}",
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis(),
        std::process::id(),
    );
    let source_id  = format!("rust-it-source-{nonce}");
    let target_id  = format!("rust-it-target-{nonce}");
    let session_id = format!("session-{nonce}");

    let source = FleetClient::new(base.clone(), source_id.clone());
    let target = FleetClient::new(base.clone(), target_id.clone());

    // ── 1. Both nodes register. ───────────────────────────────────────────────
    source.register("desktop", "kiki-os 0.1.0").await.expect("source register");
    target.register("server", "kiki-os 0.1.0").await.expect("target register");
    println!("[fleet] registered source={source_id} target={target_id}");

    // ── 2. Target's pending queue is empty. ───────────────────────────────────
    let before = target.poll_migrations().await.expect("poll empty");
    assert!(before.iter().all(|(s, _)| s != &session_id), "queue should not yet contain our session");

    // ── 3. Source sends a migration bundle aimed at the target. ───────────────
    let bundle = sample_bundle(&session_id);
    source.send_migration(&session_id, &bundle, &target_id).await.expect("send_migration");
    println!("[fleet] sent migration bundle for {session_id}");

    // ── 4. Target polls and receives the bundle — full serde round trip. ──────
    // Cloudflare KV *list* is eventually consistent — a freshly-written pointer
    // can take up to ~60s to surface in `KV.list` results. A real target node
    // polls periodically over minutes; we mirror that with a wide retry window.
    let mut found = None;
    for attempt in 0..45 {
        let pending = target.poll_migrations().await.expect("poll after send");
        if let Some(hit) = pending.into_iter().find(|(s, _)| s == &session_id) {
            println!("[fleet] bundle surfaced on attempt {attempt}");
            found = Some(hit);
            break;
        }
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
    }
    let (got_session, got_bundle) = found.expect("target must receive the migrated session within the KV-consistency window");
    assert_eq!(got_session, session_id);
    assert_eq!(got_bundle.bundle_id, bundle.bundle_id);
    assert_eq!(got_bundle.runtime.step, 7);
    assert_eq!(got_bundle.runtime.session_label, "Editing Monday's video");
    assert_eq!(got_bundle.runtime.active_apps, vec!["notes".to_string(), "media-player".into()]);
    assert_eq!(got_bundle.checkpoint.message, "pre-migration checkpoint");
    println!("[fleet] target received bundle id={} step={}", got_bundle.bundle_id, got_bundle.runtime.step);

    // ── 5. Target completes the migration; the queue drains. ──────────────────
    target.complete_migration(&session_id, &target_id).await.expect("complete_migration");
    let after = target.poll_migrations().await.expect("poll after complete");
    assert!(after.iter().all(|(s, _)| s != &session_id), "queue must drain after completion");
    println!("[fleet] migration completed and queue drained");
}
