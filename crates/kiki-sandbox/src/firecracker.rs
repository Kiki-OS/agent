//! Firecracker microVM backend — hardware-isolated execution for artifacts that
//! run untrusted / LLM-generated code (`ArtifactKind::Agent`).
//!
//! Shared-kernel sandboxes (namespaces + Landlock + seccomp) are sufficient for
//! ordinary apps, but model-generated code is treated as hostile: it gets its
//! own kernel inside a Firecracker microVM. Firecracker exposes a small HTTP/1.1
//! REST API over a Unix socket; we drive it directly (no extra HTTP dependency).
//!
//! Layering mirrors the other sandbox modules:
//! - [`MicroVmConfig`] — the pure, serializable machine description + the exact
//!   API request bodies Firecracker expects. Fully unit-tested anywhere.
//! - [`configure_over`] — send the boot sequence over a connected socket. Tested
//!   against a mock API server (no real VMM needed).
//! - [`MicroVm::launch`] — spawn the `firecracker` binary + KVM and boot. Needs
//!   `/dev/kvm` + kernel/rootfs images, so it is validated on bare-metal Linux,
//!   not on this dev host nor inside a nested VM (Apple Virtualization exposes no
//!   nested KVM).

use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;

#[derive(Debug, thiserror::Error)]
pub enum FirecrackerError {
    #[error("io: {0}")]
    Io(String),
    #[error("firecracker API {endpoint} returned status {status}")]
    Api { endpoint: String, status: u16 },
    #[error("kvm unavailable: /dev/kvm not present — cannot run untrusted code in a microVM")]
    KvmUnavailable,
}

/// Description of a microVM to boot. Paths reference host files (a guest kernel
/// image and a root filesystem image baked into the OS image).
#[derive(Debug, Clone)]
pub struct MicroVmConfig {
    pub kernel_image_path: String,
    pub rootfs_path:       String,
    pub vcpus:             u8,
    pub mem_mib:           u32,
    /// Kernel command line.
    pub boot_args:         String,
    /// Host-side Unix socket for the guest↔host vsock (app control channel).
    pub vsock_uds_path:    String,
    /// Guest context id for vsock.
    pub guest_cid:         u32,
    pub read_only_rootfs:  bool,
}

impl MicroVmConfig {
    /// A locked-down default for running an Agent artifact: small, read-only
    /// root, no network device (egress still flows through the host broker over
    /// vsock), serial console off.
    pub fn for_agent(artifact_id: &str, kernel_image_path: &str, rootfs_path: &str) -> Self {
        Self {
            kernel_image_path: kernel_image_path.to_string(),
            rootfs_path:       rootfs_path.to_string(),
            vcpus:             1,
            mem_mib:           256,
            boot_args:         "console=ttyS0 reboot=k panic=1 pci=off".to_string(),
            vsock_uds_path:    format!("/run/kiki/fc-{artifact_id}.vsock"),
            guest_cid:         3,
            read_only_rootfs:  true,
        }
    }

    // ── API request bodies (Firecracker REST schema) ─────────────────────────────

    pub fn machine_config_body(&self) -> Value {
        json!({ "vcpu_count": self.vcpus, "mem_size_mib": self.mem_mib, "smt": false })
    }
    pub fn boot_source_body(&self) -> Value {
        json!({ "kernel_image_path": self.kernel_image_path, "boot_args": self.boot_args })
    }
    pub fn rootfs_drive_body(&self) -> Value {
        json!({
            "drive_id": "rootfs",
            "path_on_host": self.rootfs_path,
            "is_root_device": true,
            "is_read_only": self.read_only_rootfs,
        })
    }
    pub fn vsock_body(&self) -> Value {
        json!({ "vsock_id": "vsock0", "guest_cid": self.guest_cid, "uds_path": self.vsock_uds_path })
    }

    /// The ordered (endpoint, body) sequence to fully configure + start the VM.
    pub fn boot_sequence(&self) -> Vec<(&'static str, Value)> {
        vec![
            ("/machine-config", self.machine_config_body()),
            ("/boot-source",    self.boot_source_body()),
            ("/drives/rootfs",  self.rootfs_drive_body()),
            ("/vsock",          self.vsock_body()),
            ("/actions",        json!({ "action_type": "InstanceStart" })),
        ]
    }
}

/// Send a single Firecracker API `PUT` over an already-connected socket and
/// check the status. Firecracker replies `204 No Content` on success.
pub async fn put_over(stream: &mut UnixStream, endpoint: &str, body: &Value) -> Result<(), FirecrackerError> {
    let payload = serde_json::to_vec(body).map_err(|e| FirecrackerError::Io(e.to_string()))?;
    let req = format!(
        "PUT {endpoint} HTTP/1.1\r\nHost: localhost\r\nAccept: application/json\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: keep-alive\r\n\r\n",
        payload.len()
    );
    stream.write_all(req.as_bytes()).await.map_err(|e| FirecrackerError::Io(e.to_string()))?;
    stream.write_all(&payload).await.map_err(|e| FirecrackerError::Io(e.to_string()))?;
    stream.flush().await.map_err(|e| FirecrackerError::Io(e.to_string()))?;

    // Read just enough to parse the status line.
    let mut buf = [0u8; 256];
    let n = stream.read(&mut buf).await.map_err(|e| FirecrackerError::Io(e.to_string()))?;
    let head = String::from_utf8_lossy(&buf[..n]);
    let status = parse_status(&head)
        .ok_or_else(|| FirecrackerError::Api { endpoint: endpoint.to_string(), status: 0 })?;
    if (200..300).contains(&status) {
        Ok(())
    } else {
        Err(FirecrackerError::Api { endpoint: endpoint.to_string(), status })
    }
}

/// Drive the full boot sequence over a connected API socket.
pub async fn configure_over(stream: &mut UnixStream, cfg: &MicroVmConfig) -> Result<(), FirecrackerError> {
    for (endpoint, body) in cfg.boot_sequence() {
        put_over(stream, endpoint, &body).await?;
    }
    Ok(())
}

/// Parse the HTTP status code from a response head (`HTTP/1.1 204 No Content`).
fn parse_status(head: &str) -> Option<u16> {
    head.lines().next()?.split_whitespace().nth(1)?.parse().ok()
}

/// `true` if this host can run microVMs (KVM present).
pub fn kvm_available() -> bool {
    std::path::Path::new("/dev/kvm").exists()
}

/// A running microVM handle.
#[derive(Debug)]
pub struct MicroVm {
    pub pid:        u32,
    pub api_socket: String,
    /// Host-side vsock UDS — the app control channel into the guest.
    pub vsock_uds:  String,
}

impl MicroVmConfig {
    /// Spawn `firecracker`, configure it over its API socket, and boot the guest.
    /// Fail-closed on a host without KVM (refuses to run untrusted code without
    /// hardware isolation). Requires the `firecracker` binary + the kernel/rootfs
    /// images to exist; validated on bare-metal Linux with KVM (not on this dev
    /// host nor a nested VM).
    pub async fn launch(&self, firecracker_bin: &str) -> Result<MicroVm, FirecrackerError> {
        if !kvm_available() {
            return Err(FirecrackerError::KvmUnavailable);
        }
        let api_socket = format!("/run/kiki/fc-api-{}.sock", self.guest_cid);
        let _ = std::fs::remove_file(&api_socket);

        let child = tokio::process::Command::new(firecracker_bin)
            .arg("--api-sock")
            .arg(&api_socket)
            .spawn()
            .map_err(|e| FirecrackerError::Io(format!("spawn {firecracker_bin}: {e}")))?;
        let pid = child.id().unwrap_or(0);

        // Wait for the VMM to create its API socket (≤1s).
        for _ in 0..50 {
            if std::path::Path::new(&api_socket).exists() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }

        let mut stream = UnixStream::connect(&api_socket)
            .await
            .map_err(|e| FirecrackerError::Io(format!("connect api socket: {e}")))?;
        configure_over(&mut stream, self).await?;

        Ok(MicroVm { pid, api_socket, vsock_uds: self.vsock_uds_path.clone() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::UnixListener;

    fn cfg() -> MicroVmConfig {
        MicroVmConfig::for_agent("io.kiki.agent", "/var/kiki/vm/vmlinux", "/var/kiki/vm/rootfs.ext4")
    }

    #[test]
    fn boot_sequence_is_ordered_and_complete() {
        let seq = cfg().boot_sequence();
        let endpoints: Vec<&str> = seq.iter().map(|(e, _)| *e).collect();
        assert_eq!(endpoints, ["/machine-config", "/boot-source", "/drives/rootfs", "/vsock", "/actions"]);
        // InstanceStart is last.
        assert_eq!(seq.last().unwrap().1["action_type"], "InstanceStart");
    }

    #[test]
    fn agent_default_is_locked_down() {
        let c = cfg();
        assert!(c.read_only_rootfs, "agent rootfs must be read-only");
        assert_eq!(c.rootfs_drive_body()["is_read_only"], true);
        assert!(c.boot_args.contains("pci=off"));
        // No network interface in the boot sequence — egress only via host broker.
        assert!(cfg().boot_sequence().iter().all(|(e, _)| !e.contains("network")));
    }

    #[test]
    fn parses_http_status() {
        assert_eq!(parse_status("HTTP/1.1 204 No Content\r\n"), Some(204));
        assert_eq!(parse_status("HTTP/1.1 400 Bad Request\r\n"), Some(400));
        assert_eq!(parse_status("garbage"), None);
    }

    #[tokio::test]
    async fn configure_over_drives_the_full_sequence() {
        // Mock Firecracker API: accept the connection, reply 204 to each PUT,
        // and record the endpoints it received.
        let sock = std::env::temp_dir().join(format!("kiki-fc-test-{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&sock);
        let listener = UnixListener::bind(&sock).unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut seen = Vec::new();
            let mut buf = [0u8; 1024];
            // 5 requests in the boot sequence.
            for _ in 0..5 {
                let n = stream.read(&mut buf).await.unwrap();
                let req = String::from_utf8_lossy(&buf[..n]);
                let endpoint = req.lines().next().unwrap().split_whitespace().nth(1).unwrap().to_string();
                seen.push(endpoint);
                stream.write_all(b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n\r\n").await.unwrap();
            }
            seen
        });

        let mut client = UnixStream::connect(&sock).await.unwrap();
        configure_over(&mut client, &cfg()).await.expect("configure");

        let seen = server.await.unwrap();
        assert_eq!(seen, ["/machine-config", "/boot-source", "/drives/rootfs", "/vsock", "/actions"]);
        let _ = std::fs::remove_file(&sock);
    }

    #[tokio::test]
    async fn put_surfaces_api_errors() {
        let sock = std::env::temp_dir().join(format!("kiki-fc-err-{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&sock);
        let listener = UnixListener::bind(&sock).unwrap();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 1024];
            let _ = stream.read(&mut buf).await.unwrap();
            stream.write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n").await.unwrap();
        });
        let mut client = UnixStream::connect(&sock).await.unwrap();
        let err = put_over(&mut client, "/machine-config", &json!({})).await.unwrap_err();
        assert!(matches!(err, FirecrackerError::Api { status: 400, .. }));
        let _ = std::fs::remove_file(&sock);
    }
}
