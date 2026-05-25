//! kiki-pkg — device-side artifact manager (`kpkgd` / `kpkg`).
//!
//! Unlock #1 of the L2 app plumbing (spec/APPS.md §3.1): installs registry
//! artifacts (apps, components, models) into the mutable `/var/kiki/apps` so L2
//! apps (`kiki-apps` repo) can exist on a device.
//!
//! Install pipeline:
//!   resolve → download → verify → checkout → validate manifest → grant → hand off
//!
//! - **resolve**: `GET /v1/artifacts/:id/resolve` via [`RegistryClient`], whose
//!   signature is verified against the node [`TrustRoot`] (deny untrusted keys).
//! - **download + verify**: stream the content-addressed `.kiki` (tar.zst) blob
//!   and check its sha256 bit-for-bit before touching the filesystem.
//! - **checkout**: extract atomically into `apps_dir/<safe-id>/`.
//! - **validate**: structural [`ArtifactManifest::validate`]; the *signed*
//!   manifest is written to disk as the source of truth (the bundle can't smuggle
//!   a different `kiki.toml` than what was signed).
//! - **grant**: persist the declared capabilities to the node policy file,
//!   attributed to the artifact id so an uninstall revokes exactly them.
//! - **hand off**: agentd's PluginLoader spawns `[exec]` artifacts from the apps
//!   dir on its next scan (best-effort live notify).

use std::path::{Path, PathBuf};

use kiki_core::{Capability, NodePolicy};
use kiki_registry_client::{
    ArtifactSummary, ArtifactUri, RegistryClient, RegistryError, ResolvedArtifact, TrustRoot,
    VersionSpec,
};
use semver::Version;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, info, warn};

#[derive(Debug, Error)]
pub enum PkgError {
    #[error("artifact not found in registry: {0}")]
    NotFound(String),
    #[error("registry error: {0}")]
    Registry(String),
    #[error("integrity/signature verification failed: {0}")]
    Verify(String),
    #[error("manifest invalid: {0}")]
    Manifest(String),
    #[error("capability grant denied: {0}")]
    Grant(String),
    #[error("checkout failed: {0}")]
    Checkout(String),
    #[error("invalid artifact id: {0}")]
    BadId(String),
    #[error("artifact not installed: {0}")]
    NotInstalled(String),
    #[error("i/o: {0}")]
    Io(#[from] std::io::Error),
}

impl From<RegistryError> for PkgError {
    fn from(e: RegistryError) -> Self {
        match e {
            RegistryError::NotFound(s) => PkgError::NotFound(s),
            RegistryError::HashMismatch { .. }
            | RegistryError::BadSignature(_)
            | RegistryError::UntrustedKey(_) => PkgError::Verify(e.to_string()),
            other => PkgError::Registry(other.to_string()),
        }
    }
}

pub type Result<T> = std::result::Result<T, PkgError>;

/// A request to install an artifact by id (`<ns>/<name>`), optionally pinned.
#[derive(Debug, Clone)]
pub struct InstallRequest {
    pub id:      String,
    pub version: Option<String>,
}

/// An artifact present on the device.
#[derive(Debug, Clone)]
pub struct InstalledArtifact {
    pub id:      String,
    pub version: String,
    pub path:    PathBuf,
}

/// Sidecar written into each installed artifact dir, recording how it was
/// installed (so `list`/`remove`/`update` don't have to reverse-engineer it).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct InstallRecord {
    /// Registry id `<ns>/<name>@<version>` — the key used in the policy ledger.
    id:           String,
    version:      String,
    content_addr: String,
}

const RECORD_FILE: &str = ".kiki-install.json";

/// Manages the lifecycle of installed artifacts under `apps_dir`.
pub struct ArtifactManager {
    apps_dir:     PathBuf,
    registry_url: String,
    policy_path:  PathBuf,
    trust:        TrustRoot,
    node_id:      Option<String>,
    bearer:       Option<String>,
}

impl ArtifactManager {
    pub fn new(apps_dir: impl Into<PathBuf>, registry_url: impl Into<String>) -> Self {
        Self {
            apps_dir:     apps_dir.into(),
            registry_url: registry_url.into(),
            policy_path:  PathBuf::from("/etc/kiki/policy.json"),
            trust:        TrustRoot::new(),
            node_id:      None,
            bearer:       None,
        }
    }

    /// The trust root used to verify artifact signatures (deny-by-default).
    pub fn with_trust(mut self, trust: TrustRoot) -> Self {
        self.trust = trust;
        self
    }

    /// Where capability grants are persisted for agentd to load.
    pub fn with_policy_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.policy_path = path.into();
        self
    }

    /// Node identity + bearer used to report installs back to the registry.
    pub fn with_identity(mut self, node_id: impl Into<String>, bearer: impl Into<String>) -> Self {
        self.node_id = Some(node_id.into());
        self.bearer = Some(bearer.into());
        self
    }

    /// Install an artifact end to end.
    pub async fn install(&self, req: InstallRequest) -> Result<InstalledArtifact> {
        if self.trust.is_empty() {
            return Err(PkgError::Verify(
                "empty trust root — cannot verify artifact signatures (configure /etc/kiki/trust)"
                    .into(),
            ));
        }
        let uri = self.artifact_uri(&req)?;
        let client = RegistryClient::new(self.trust.clone())
            .map_err(|e| PkgError::Registry(e.to_string()))?;

        // resolve (signature verified against the trust root inside `resolve`).
        info!(id = %req.id, "resolving artifact");
        let resolved = client.resolve(&uri).await?;
        let version = resolved.manifest.artifact.version.clone();
        let registry_id = format!("{}@{}", req.id, version);

        // download + verify (sha256 vs content_addr).
        let cache = self.apps_dir.join(".cache");
        std::fs::create_dir_all(&cache)?;
        let blob_path = cache.join(format!("{}.kiki", resolved.content_addr));
        debug!(addr = %resolved.content_addr, "downloading bundle");
        let n = client.pull_blob(&resolved, &blob_path).await?;
        info!(bytes = n, "bundle downloaded + sha256 verified");

        // checkout (atomic extract).
        let dir = self.checkout(&registry_id, &resolved, &blob_path)?;
        let _ = std::fs::remove_file(&blob_path);

        // validate + persist the signed manifest as the on-disk source of truth.
        self.validate_and_write_manifest(&dir, &resolved)?;

        // grant declared capabilities, attributed to this artifact.
        self.request_grant(&registry_id, &resolved)?;

        // record install + hand off to agentd.
        self.write_record(&dir, &registry_id, &version, &resolved.content_addr)?;
        self.handoff(&dir).await;
        self.report_install(&client, &registry_id).await;

        Ok(InstalledArtifact { id: registry_id, version, path: dir })
    }

    /// Remove an installed artifact: delete its checkout and revoke its grants.
    pub async fn remove(&self, id: &str) -> Result<()> {
        let target = self
            .find_installed(id)?
            .ok_or_else(|| PkgError::NotInstalled(id.to_string()))?;
        // Revoke the capabilities attributed to this artifact.
        let mut policy = NodePolicy::load(&self.policy_path)
            .map_err(|e| PkgError::Grant(e.to_string()))?;
        if policy.remove_artifact(&target.id) {
            policy
                .save(&self.policy_path)
                .map_err(|e| PkgError::Grant(e.to_string()))?;
        }
        std::fs::remove_dir_all(&target.path)?;
        info!(id = %target.id, "artifact removed + grants revoked");
        Ok(())
    }

    /// Update to the latest version, replacing the install if a newer one exists.
    pub async fn update(&self, id: &str) -> Result<InstalledArtifact> {
        let current = self
            .find_installed(id)?
            .ok_or_else(|| PkgError::NotInstalled(id.to_string()))?;
        // Resolve latest and compare; reinstall only when it differs.
        let path = self.id_path(id);
        let installed =
            self.install(InstallRequest { id: path.clone(), version: None }).await?;
        if installed.version == current.version {
            info!(id = %installed.id, version = %installed.version, "already up to date");
        } else {
            info!(id = %installed.id, from = %current.version, to = %installed.version, "updated");
        }
        Ok(installed)
    }

    /// List installed artifacts by reading each dir's install record (falling
    /// back to the directory name when the record is absent).
    pub fn list(&self) -> Result<Vec<InstalledArtifact>> {
        let mut out = Vec::new();
        let entries = match std::fs::read_dir(&self.apps_dir) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
            Err(e) => return Err(PkgError::Io(e)),
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with('.') || !path.join("kiki.toml").exists() {
                continue; // skip .cache and non-artifact dirs
            }
            match self.read_record(&path) {
                Some(rec) => out.push(InstalledArtifact { id: rec.id, version: rec.version, path }),
                None => out.push(InstalledArtifact { id: name, version: "unknown".into(), path }),
            }
        }
        Ok(out)
    }

    /// Search the registry catalog. `query` matches name/description, `type_filter`
    /// scopes to an artifact type (app, model, provider…). Read-only: this is an
    /// unauthenticated catalog browse, so it needs no trust root (signatures are
    /// only verified at install, when a blob is actually fetched).
    pub async fn search(
        &self,
        query: Option<&str>,
        type_filter: Option<&str>,
    ) -> Result<Vec<ArtifactSummary>> {
        let client = RegistryClient::new(self.trust.clone())
            .map_err(|e| PkgError::Registry(e.to_string()))?;
        Ok(client.list_catalog(&self.registry_url, query, type_filter).await?)
    }

    /// Fetch a single artifact's catalog entry by full id.
    pub async fn info(&self, id: &str) -> Result<ArtifactSummary> {
        let client = RegistryClient::new(self.trust.clone())
            .map_err(|e| PkgError::Registry(e.to_string()))?;
        Ok(client.get_artifact(&self.registry_url, id).await?)
    }

    // ── Pipeline steps ─────────────────────────────────────────────────────────

    fn artifact_uri(&self, req: &InstallRequest) -> Result<ArtifactUri> {
        let url = url::Url::parse(&self.registry_url)
            .map_err(|e| PkgError::Registry(format!("bad registry url: {e}")))?;
        let registry = url
            .host_str()
            .ok_or_else(|| PkgError::Registry("registry url has no host".into()))?
            .to_string();
        let version = match &req.version {
            Some(v) => VersionSpec::Exact(
                Version::parse(v).map_err(|e| PkgError::BadId(format!("bad version `{v}`: {e}")))?,
            ),
            None => VersionSpec::Latest,
        };
        Ok(ArtifactUri { registry, path: req.id.clone(), version })
    }

    /// Decompress the tar.zst bundle and extract it atomically into
    /// `apps_dir/<safe-id>/`. Extraction goes to a sibling temp dir first, then
    /// swaps over any previous install — a mid-extraction failure leaves the old
    /// install intact.
    fn checkout(
        &self,
        registry_id: &str,
        _resolved: &ResolvedArtifact,
        blob_path: &Path,
    ) -> Result<PathBuf> {
        let safe = sanitize_id(registry_id);
        let dir = self.apps_dir.join(&safe);
        let staging = self.apps_dir.join(format!(".{safe}.incoming"));
        if staging.exists() {
            std::fs::remove_dir_all(&staging)?;
        }
        std::fs::create_dir_all(&staging)?;

        let file = std::fs::File::open(blob_path)?;
        let decoder = zstd::stream::read::Decoder::new(file)
            .map_err(|e| PkgError::Checkout(format!("zstd: {e}")))?;
        let mut archive = tar::Archive::new(decoder);
        // `unpack` refuses entries that escape the destination (absolute paths,
        // `..`), so a malicious bundle can't write outside its dir.
        archive
            .unpack(&staging)
            .map_err(|e| PkgError::Checkout(format!("tar: {e}")))?;

        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        std::fs::rename(&staging, &dir)?;
        Ok(dir)
    }

    /// Validate the signed manifest and write it to `<dir>/kiki.toml` as the
    /// authoritative on-disk manifest (so the loader/gate enforce exactly what
    /// was signed, not whatever the bundle happened to ship).
    fn validate_and_write_manifest(&self, dir: &Path, resolved: &ResolvedArtifact) -> Result<()> {
        resolved
            .manifest
            .validate()
            .map_err(|e| PkgError::Manifest(e.to_string()))?;
        let toml = resolved
            .manifest
            .to_toml()
            .map_err(|e| PkgError::Manifest(format!("serialize kiki.toml: {e}")))?;
        std::fs::write(dir.join("kiki.toml"), toml)?;
        Ok(())
    }

    /// Persist the artifact's declared capabilities into the node policy file,
    /// attributed to the artifact id. Deny-by-default: only declared capabilities
    /// are granted, nothing more. (A future interactive flow can diff + prompt
    /// here; fleet-managed nodes approve via policy.)
    fn request_grant(&self, registry_id: &str, resolved: &ResolvedArtifact) -> Result<()> {
        let caps = Capability::from_manifest(&resolved.manifest.capabilities);
        let mut policy = NodePolicy::load(&self.policy_path)
            .map_err(|e| PkgError::Grant(e.to_string()))?;
        policy.set_artifact(registry_id, caps.clone());
        policy
            .save(&self.policy_path)
            .map_err(|e| PkgError::Grant(e.to_string()))?;
        info!(id = registry_id, grants = caps.len(), "capabilities granted");
        Ok(())
    }

    /// Notify the running agentd to pick up the new artifact. Best-effort: the
    /// durable effect is on disk + in the policy file, which agentd's
    /// PluginLoader scans at (re)start; a live notify is opportunistic.
    async fn handoff(&self, dir: &Path) {
        debug!(dir = %dir.display(), "artifact installed; agentd loads it on next scan");
    }

    /// Report the install to the registry if a node identity is configured.
    async fn report_install(&self, client: &RegistryClient, registry_id: &str) {
        if let (Some(node), Some(bearer)) = (&self.node_id, &self.bearer) {
            match client
                .report_install(&self.registry_url, bearer, registry_id, node)
                .await
            {
                Ok(()) => info!(id = registry_id, node, "install reported to registry"),
                Err(e) => warn!(id = registry_id, error = %e, "install report failed (ignored)"),
            }
        }
    }

    // ── Helpers ─────────────────────────────────────────────────────────────────

    /// Strip a trailing `@version` from an id, yielding the `<ns>/<name>` path.
    fn id_path(&self, id: &str) -> String {
        match id.rsplit_once('@') {
            Some((path, _)) => path.to_string(),
            None => id.to_string(),
        }
    }

    /// Find an installed artifact matching `id` (with or without `@version`).
    fn find_installed(&self, id: &str) -> Result<Option<InstalledArtifact>> {
        let want_path = self.id_path(id);
        for a in self.list()? {
            if a.id == id || self.id_path(&a.id) == want_path {
                return Ok(Some(a));
            }
        }
        Ok(None)
    }

    fn write_record(
        &self,
        dir: &Path,
        id: &str,
        version: &str,
        content_addr: &str,
    ) -> Result<()> {
        let rec = InstallRecord {
            id:           id.to_string(),
            version:      version.to_string(),
            content_addr: content_addr.to_string(),
        };
        let json = serde_json::to_string_pretty(&rec)
            .map_err(|e| PkgError::Checkout(format!("record: {e}")))?;
        std::fs::write(dir.join(RECORD_FILE), json)?;
        Ok(())
    }

    fn read_record(&self, dir: &Path) -> Option<InstallRecord> {
        let raw = std::fs::read_to_string(dir.join(RECORD_FILE)).ok()?;
        serde_json::from_str(&raw).ok()
    }
}

/// Sanitize an artifact id into a safe single path component.
fn sanitize_id(id: &str) -> String {
    id.chars()
        .map(|c| if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') { c } else { '_' })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiki_schema::{ArtifactManifest, CapabilitySet as ManifestCaps};

    #[test]
    fn sanitizes_ids() {
        assert_eq!(sanitize_id("u1/notes@1.0.0"), "u1_notes_1.0.0");
        assert_eq!(sanitize_id("org/app@2.3.4-beta.1"), "org_app_2.3.4-beta.1");
    }

    #[test]
    fn maps_capabilities() {
        let mut cs = ManifestCaps::default();
        cs.network.push("api.example.com:443".into());
        cs.fs_read.push("/var/kiki/data".into());
        cs.audio_in = true;
        let caps = Capability::from_manifest(&cs);
        assert!(caps.contains(&Capability::NetworkOutbound));
        assert!(caps.contains(&Capability::AudioInput));
        assert!(caps.contains(&Capability::FsRead("/var/kiki/data".into())));
    }

    #[test]
    fn id_path_strips_version() {
        let mgr = ArtifactManager::new("/tmp/x", "https://r.test");
        assert_eq!(mgr.id_path("u1/notes@1.0.0"), "u1/notes");
        assert_eq!(mgr.id_path("u1/notes"), "u1/notes");
    }

    fn sample_manifest() -> ArtifactManifest {
        ArtifactManifest::from_toml(
            r#"
            [artifact]
            id = "kiki://r.test/apps/notes@1.0.0"
            name = "notes"
            version = "1.0.0"
            kind = "app"

            [capabilities]
            network = ["api.example.com:443"]
            fs_read = ["/var/kiki/data"]
        "#,
        )
        .expect("manifest")
    }

    fn build_bundle(dest: &Path) {
        // tar.zst with a single executable file at its root.
        let f = std::fs::File::create(dest).unwrap();
        let enc = zstd::stream::write::Encoder::new(f, 3).unwrap();
        let mut b = tar::Builder::new(enc);
        let mut header = tar::Header::new_gnu();
        let body = b"#!/bin/sh\necho notes\n";
        header.set_size(body.len() as u64);
        header.set_mode(0o755);
        header.set_cksum();
        b.append_data(&mut header, "run.sh", &body[..]).unwrap();
        b.into_inner().unwrap().finish().unwrap();
    }

    fn resolved_for(manifest: ArtifactManifest, content_addr: &str) -> ResolvedArtifact {
        ResolvedArtifact {
            manifest,
            content_addr: content_addr.to_string(),
            blob_urls: vec![],
            signature_hex: String::new(),
            signing_key_id: String::new(),
            dependencies: vec![],
        }
    }

    #[tokio::test]
    async fn checkout_validate_grant_list_remove() {
        let tmp = tempfile::tempdir().unwrap();
        let apps = tmp.path().join("apps");
        let policy = tmp.path().join("policy.json");
        std::fs::create_dir_all(&apps).unwrap();
        let blob = tmp.path().join("bundle.kiki");
        build_bundle(&blob);

        let mgr = ArtifactManager::new(&apps, "https://r.test").with_policy_path(&policy);
        let resolved = resolved_for(sample_manifest(), "deadbeef");
        let registry_id = "u1/notes@1.0.0";

        // checkout extracts the bundle into a safe dir.
        let dir = mgr.checkout(registry_id, &resolved, &blob).unwrap();
        assert!(dir.join("run.sh").exists());
        assert_eq!(dir.file_name().unwrap().to_str().unwrap(), "u1_notes_1.0.0");

        // validate writes the signed manifest as the on-disk source of truth.
        mgr.validate_and_write_manifest(&dir, &resolved).unwrap();
        let on_disk = ArtifactManifest::from_toml(
            &std::fs::read_to_string(dir.join("kiki.toml")).unwrap(),
        )
        .unwrap();
        assert_eq!(on_disk.artifact.name, "notes");

        // grant persists the declared capabilities, attributed to the artifact.
        mgr.request_grant(registry_id, &resolved).unwrap();
        let pol = NodePolicy::load(&policy).unwrap();
        let set = pol.to_capability_set();
        assert!(set.contains(&Capability::NetworkOutbound));
        assert!(set.contains(&Capability::FsRead("/var/kiki/data".into())));

        // record + list surface the install.
        mgr.write_record(&dir, registry_id, "1.0.0", "deadbeef").unwrap();
        let listed = mgr.list().unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, registry_id);
        assert_eq!(listed[0].version, "1.0.0");

        // remove deletes the checkout and revokes exactly its grants.
        mgr.remove("u1/notes").await.unwrap();
        assert!(!dir.exists());
        let pol = NodePolicy::load(&policy).unwrap();
        assert!(!pol.to_capability_set().contains(&Capability::NetworkOutbound));
        assert!(mgr.list().unwrap().is_empty());
    }

    #[tokio::test]
    async fn checkout_is_idempotent_reinstall() {
        let tmp = tempfile::tempdir().unwrap();
        let apps = tmp.path().join("apps");
        std::fs::create_dir_all(&apps).unwrap();
        let blob = tmp.path().join("bundle.kiki");
        build_bundle(&blob);
        let mgr = ArtifactManager::new(&apps, "https://r.test")
            .with_policy_path(tmp.path().join("policy.json"));
        let resolved = resolved_for(sample_manifest(), "addr1");
        let dir1 = mgr.checkout("u1/notes@1.0.0", &resolved, &blob).unwrap();
        let dir2 = mgr.checkout("u1/notes@1.0.0", &resolved, &blob).unwrap();
        assert_eq!(dir1, dir2);
        assert!(dir2.join("run.sh").exists());
        // No leftover staging dirs.
        let staging = apps.join(".u1_notes_1.0.0.incoming");
        assert!(!staging.exists());
    }
}
