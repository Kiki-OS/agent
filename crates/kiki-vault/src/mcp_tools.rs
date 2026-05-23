//! Built-in MCP tool schemas for the agent.
//!
//! Each `T*Args` struct is the JSON schema arguments the agent supplies; the
//! corresponding `T*Output` is what the tool returns. `agentd` wires these to
//! [`VaultClient`] methods inside `kiki-mcp/src/builtin.rs`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaultReadArgs {
    /// Full `vault://` URI.
    pub uri: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaultReadOutput {
    pub sha256:    String,
    pub revision:  u64,
    pub mime_type: Option<String>,
    /// Base64-encoded bytes.
    pub bytes_b64: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaultWriteArgs {
    pub uri:       String,
    /// CAS guard. Use 0 for create-only.
    pub if_match:  u64,
    /// Base64-encoded bytes.
    pub bytes_b64: String,
    #[serde(default)]
    pub mime_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaultWriteOutput {
    pub sha256:   String,
    pub revision: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaultListArgs {
    pub scope:    String,
    pub owner_id: String,
    pub prefix:   String,
    #[serde(default)]
    pub since:    u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaultListOutput {
    pub entries: Vec<VaultListEntryOut>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaultListEntryOut {
    pub path:     String,
    pub sha256:   String,
    pub size:     u64,
    pub revision: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaultHeadArgs {
    pub uri: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaultHeadOutput {
    pub sha256:        String,
    pub size:          u64,
    pub revision:      u64,
    pub mime_type:     Option<String>,
    pub modified_at_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaultWatchArgs {
    pub scope:    String,
    pub owner_id: String,
    pub globs:    Vec<String>,
}

/// Catalogue used by `kiki-mcp` to register the builtin tools at startup.
pub fn tool_descriptors() -> Vec<ToolDescriptor> {
    vec![
        ToolDescriptor {
            name:        "vault_read",
            description: "Read a blob from the Kiki vault.",
        },
        ToolDescriptor {
            name:        "vault_write",
            description: "CAS-write a blob to the Kiki vault. if_match guards against lost-update.",
        },
        ToolDescriptor {
            name:        "vault_list",
            description: "List blobs under a prefix, optionally since a revision.",
        },
        ToolDescriptor {
            name:        "vault_head",
            description: "Read metadata for a vault path without fetching bytes.",
        },
        ToolDescriptor {
            name:        "vault_watch",
            description: "Subscribe to live change events for one or more glob patterns.",
        },
    ]
}

#[derive(Debug, Clone, Copy)]
pub struct ToolDescriptor {
    pub name:        &'static str,
    pub description: &'static str,
}
