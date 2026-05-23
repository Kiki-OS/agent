//! Structured tracing and metrics. Every PRA step emits a span.
//! Fleet telemetry aggregates to the cloud control plane.

use tracing_subscriber::{fmt, prelude::*, EnvFilter};

/// Initialize the process-wide tracing subscriber.
///
/// Honors `RUST_LOG` (defaulting to `info`). Idempotent: a second call is a
/// no-op rather than a panic, so tests and embedders can call it freely.
/// Set `KIKI_LOG_JSON=1` for line-delimited JSON output (production / log
/// shipping); otherwise a human-readable format is used.
pub fn init() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    let json = std::env::var("KIKI_LOG_JSON").as_deref() == Ok("1");

    let registry = tracing_subscriber::registry().with(filter);
    if json {
        let _ = registry.with(fmt::layer().json().with_target(true)).try_init();
    } else {
        let _ = registry.with(fmt::layer().with_target(true)).try_init();
    }
}
