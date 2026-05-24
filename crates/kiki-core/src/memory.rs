//! Memory context hook for the harness.
//!
//! The harness is decoupled from the memory subsystem (`kiki-memory` / `memoryd`):
//! it depends only on this trait. agentd provides the concrete implementation
//! backed by the memoryd socket, so kiki-core never learns the wire format or
//! the socket path.
//!
//! At the start of a session (first user input) the harness asks the provider to
//! [`recall`](MemoryContext::recall) memory relevant to the user's request. The
//! returned lines (identity summary, recalled procedural/episodic memories, and
//! the user's standing corrections) are folded into the system prompt so the
//! model sees them within its context budget — provider-agnostic, since every
//! backend honors the system prompt.

use async_trait::async_trait;

/// Source of session-relevant memory for the harness.
#[async_trait]
pub trait MemoryContext: Send + Sync {
    /// Return already-formatted memory lines relevant to `hint` (the user's
    /// input). An empty vec means "no memory to inject". Implementations should
    /// keep the result bounded (the harness injects it into a fixed context slot).
    async fn recall(&self, hint: &str) -> Vec<String>;
}

/// Format recalled memory lines into a system-prompt addendum, or `None` when
/// there's nothing to add. Pure + testable; the harness appends the result to
/// the agent's base system prompt.
pub fn memory_preamble(lines: &[String]) -> Option<String> {
    let non_empty: Vec<&String> = lines.iter().filter(|l| !l.trim().is_empty()).collect();
    if non_empty.is_empty() {
        return None;
    }
    let body = non_empty.iter().map(|l| format!("- {l}")).collect::<Vec<_>>().join("\n");
    Some(format!(
        "# Relevant memory (on-device; from prior sessions)\n{body}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preamble_skips_when_empty() {
        assert!(memory_preamble(&[]).is_none());
        assert!(memory_preamble(&["".into(), "   ".into()]).is_none());
    }

    #[test]
    fn preamble_bullets_non_empty_lines() {
        let p = memory_preamble(&["likes concise replies".into(), "on flaky network".into()]).unwrap();
        assert!(p.contains("# Relevant memory"));
        assert!(p.contains("- likes concise replies"));
        assert!(p.contains("- on flaky network"));
    }
}
