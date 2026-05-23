//! Fleet management client (on-device side).
//!
//! Connects to kiki-cloud/workers/fleet. Responsibilities:
//!   - Register node and keep it alive in the fleet registry (heartbeat)
//!   - Pull machine config from cloud (OSTree ref → apply atomically via bootc)
//!   - Send and receive live session migrations (MigrationBundle over HTTP relay)
//!
//! Usage:
//!   ```rust,no_run
//!   use std::sync::Arc;
//!   use kiki_fleet::client::FleetClient;
//!   use kiki_fleet::heartbeat::Heartbeat;
//!   use kiki_fleet::migration::{MigrationSender, MigrationReceiver};
//!
//!   let fleet = Arc::new(FleetClient::new("https://fleet.kiki-os.com", "node-abc"));
//!   let _hb = Heartbeat::with_default_interval(fleet.clone()).spawn();
//!   let sender   = MigrationSender::new(fleet.clone(), "https://registry.kiki-os.com");
//!   let receiver = MigrationReceiver::new(fleet.clone(), "https://registry.kiki-os.com");
//!   ```

pub mod client;
pub mod sync;
pub mod heartbeat;
pub mod migration;

pub use client::FleetClient;
pub use heartbeat::Heartbeat;
pub use migration::{MigrationReceiver, MigrationSender, RestoredSession};
pub use sync::{connect_device, DeviceInbound, SessionPublisher, StatePatch};
