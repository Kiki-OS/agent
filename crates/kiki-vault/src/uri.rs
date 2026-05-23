//! `vault://` URI parsing.

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VaultUri {
    pub scope:    VaultScope,
    pub owner_id: String,
    pub path:     String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VaultScope {
    Personal,
    Fleet,
    Share,
}

impl VaultScope {
    pub fn as_str(self) -> &'static str {
        match self {
            VaultScope::Personal => "personal",
            VaultScope::Fleet => "fleet",
            VaultScope::Share => "share",
        }
    }
}

#[derive(Debug, Error)]
pub enum UriError {
    #[error("expected scheme `vault://`, got `{0}`")]
    BadScheme(String),
    #[error("unknown scope `{0}`")]
    UnknownScope(String),
    #[error("missing owner id (e.g. `vault://personal/<user_id>/...`)")]
    MissingOwner,
    #[error("missing path (e.g. `vault://personal/u1/notes/draft.md`)")]
    MissingPath,
}

impl VaultUri {
    pub fn parse(s: &str) -> Result<Self, UriError> {
        let rest = s
            .strip_prefix("vault://")
            .ok_or_else(|| UriError::BadScheme(s.split("://").next().unwrap_or("").to_string()))?;
        let (scope_str, after_scope) = rest
            .split_once('/')
            .ok_or_else(|| UriError::MissingOwner)?;
        let scope = match scope_str {
            "personal" => VaultScope::Personal,
            "fleet" => VaultScope::Fleet,
            "share" => VaultScope::Share,
            other => return Err(UriError::UnknownScope(other.to_string())),
        };
        let (owner, path) = after_scope
            .split_once('/')
            .ok_or(UriError::MissingPath)?;
        if owner.is_empty() {
            return Err(UriError::MissingOwner);
        }
        if path.is_empty() {
            return Err(UriError::MissingPath);
        }
        Ok(Self {
            scope,
            owner_id: owner.to_string(),
            path:     path.to_string(),
        })
    }

    pub fn to_string(&self) -> String {
        format!("vault://{}/{}/{}", self.scope.as_str(), self.owner_id, self.path)
    }
}

/// Check whether a concrete path matches a glob pattern with `*` (single
/// segment) and `**` (multi-segment). Used by capability-token enforcement
/// and watch subscriptions.
pub fn glob_match(pattern: &str, path: &str) -> bool {
    let pat_segs: Vec<&str> = pattern.split('/').collect();
    let path_segs: Vec<&str> = path.split('/').collect();
    glob_segs(&pat_segs, &path_segs)
}

fn glob_segs(pat: &[&str], path: &[&str]) -> bool {
    match (pat, path) {
        ([], []) => true,
        ([], _) => false,
        (["**"], _) => true,
        (["**", rest_pat @ ..], path) => {
            (0..=path.len()).any(|i| glob_segs(rest_pat, &path[i..]))
        }
        ([first, rest_pat @ ..], [head, rest_path @ ..]) if seg_match(first, head) => {
            glob_segs(rest_pat, rest_path)
        }
        _ => false,
    }
}

fn seg_match(pat: &str, seg: &str) -> bool {
    if pat == "*" {
        return true;
    }
    // Allow `prefix*` and `*suffix` and `*middle*` patterns within a segment.
    if let Some((before, after)) = pat.split_once('*') {
        return seg.starts_with(before) && seg.ends_with(after) && seg.len() >= before.len() + after.len();
    }
    pat == seg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_personal() {
        let u = VaultUri::parse("vault://personal/u-123/notes/draft.md").unwrap();
        assert_eq!(u.scope, VaultScope::Personal);
        assert_eq!(u.owner_id, "u-123");
        assert_eq!(u.path, "notes/draft.md");
    }

    #[test]
    fn rejects_bad_scheme() {
        assert!(matches!(
            VaultUri::parse("https://x"),
            Err(UriError::BadScheme(_))
        ));
    }

    #[test]
    fn rejects_unknown_scope() {
        assert!(matches!(
            VaultUri::parse("vault://other/u/x"),
            Err(UriError::UnknownScope(_))
        ));
    }

    #[test]
    fn glob_exact() {
        assert!(glob_match("notes/draft.md", "notes/draft.md"));
        assert!(!glob_match("notes/draft.md", "notes/other.md"));
    }

    #[test]
    fn glob_single_segment() {
        assert!(glob_match("notes/*", "notes/draft.md"));
        assert!(!glob_match("notes/*", "notes/sub/draft.md"));
    }

    #[test]
    fn glob_recursive() {
        assert!(glob_match("notes/**", "notes/sub/deep/file.md"));
        assert!(glob_match("**", "anything/at/all"));
    }

    #[test]
    fn glob_prefix() {
        assert!(glob_match("img-*", "img-001"));
        assert!(!glob_match("img-*", "vid-001"));
    }
}
