//! agentd's [`MemoryContext`] implementation, backed by the memoryd socket.
//!
//! Bridges the harness's `kiki_core::MemoryContext` hook to `memoryd` via
//! [`MemoryClient`]. On the first user input the harness calls [`recall`], which
//! gathers the user's standing corrections, memory relevant to the request
//! (procedural + episodic), and an identity summary — all on-device, best-effort
//! (if memoryd isn't running the queries fail and recall returns nothing).

use async_trait::async_trait;
use kiki_core::harness::AgentEvent;
use kiki_core::memory::MemoryContext;
use kiki_memory::{EpisodeEvent, MemoryClient, MemoryLayer, MemoryQuery, MemoryResult, MemoryWrite};

/// Heuristic: does this user input look like a correction of the agent's
/// behavior? If so, return the correction text to persist immediately (the spec
/// writes `UserCorrection` the moment the user corrects something). Conservative
/// — only clear correction openers trigger, to avoid polluting memory with
/// ordinary requests.
pub fn detect_correction(text: &str) -> Option<String> {
    let t = text.trim();
    if t.is_empty() {
        return None;
    }
    let lower = t.to_lowercase();
    // Opening phrases that signal "you did/are doing this wrong".
    const OPENERS: &[&str] = &[
        "no,", "no ", "don't ", "do not ", "stop ", "actually,", "actually ",
        "instead ", "i prefer ", "prefer ", "never ", "always ", "not like that",
        "that's wrong", "thats wrong", "that is wrong", "incorrect", "you should",
    ];
    let matched = OPENERS.iter().any(|p| lower.starts_with(p) || lower.contains(p));
    if matched {
        Some(t.to_string())
    } else {
        None
    }
}

/// Post-task reflection: map an agent event to a memory write worth keeping.
/// Successes (session done) and failures (errors, healed turns) both become
/// episodes so future recall can surface "this happened before". Returns `None`
/// for events not worth recording. Pure + testable; the agentd event loop spawns
/// the actual best-effort write.
pub fn reflection_write(event: &AgentEvent, session_id: &str, ts_ms: u64) -> Option<MemoryWrite> {
    let ep = match event {
        AgentEvent::Done { session_id, steps } => EpisodeEvent {
            id:         format!("session-{session_id}-{ts_ms}"),
            kind:       "session_done".into(),
            session_id: session_id.clone(),
            summary:    format!("session {session_id} completed in {steps} steps"),
            outcome:    "ok".into(),
            ts_ms,
            important:  false,
        },
        AgentEvent::Error { error } => EpisodeEvent {
            id:         format!("error-{ts_ms}"),
            kind:       "agent_error".into(),
            session_id: session_id.to_string(),
            summary:    format!("agent error: {error}"),
            outcome:    "error".into(),
            ts_ms,
            important:  false,
        },
        AgentEvent::Healing { attempt, error } => EpisodeEvent {
            id:         format!("healing-{ts_ms}-{attempt}"),
            kind:       "healing".into(),
            session_id: session_id.to_string(),
            summary:    format!("recovered from failure (attempt {attempt}): {error}"),
            outcome:    "healed".into(),
            ts_ms,
            important:  false,
        },
        _ => return None,
    };
    Some(MemoryWrite::Episode { event: ep })
}

pub struct MemorydContext {
    client: MemoryClient,
}

impl MemorydContext {
    pub fn new() -> Self {
        Self { client: MemoryClient::default_socket() }
    }
}

#[async_trait]
impl MemoryContext for MemorydContext {
    async fn recall(&self, hint: &str) -> Vec<String> {
        let mut lines = Vec::new();

        // Identity summary (only the parts that are set).
        if let Ok(MemoryResult::Profile { profile }) = self.client.query(MemoryQuery::UserProfile).await {
            let mut who = Vec::new();
            if !profile.display_name.is_empty() {
                who.push(format!("user is {}", profile.display_name));
            }
            if !profile.expertise.is_empty() {
                who.push(format!("expertise: {}", profile.expertise.join(", ")));
            }
            if !who.is_empty() {
                lines.push(format!("identity — {}", who.join("; ")));
            }
        }

        // Standing corrections always come first in priority (high value).
        if let Ok(MemoryResult::Hits { hits }) =
            self.client.query(MemoryQuery::Corrections { limit: 5 }).await
        {
            for h in hits {
                lines.push(format!("correction: {}", h.content.replace('\n', " ")));
            }
        }

        // Memory relevant to this specific request (procedural recipes + past episodes).
        if let Ok(MemoryResult::Hits { hits }) = self
            .client
            .query(MemoryQuery::Search {
                query:  hint.to_string(),
                layers: vec![MemoryLayer::Procedural, MemoryLayer::Episodic, MemoryLayer::Semantic],
                limit:  5,
            })
            .await
        {
            for h in hits {
                lines.push(format!("{:?}: {}", h.layer, h.content.replace('\n', " ")));
            }
        }

        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn done_event_becomes_session_episode() {
        let ev = AgentEvent::Done { session_id: "s1".into(), steps: 7 };
        let MemoryWrite::Episode { event } = reflection_write(&ev, "s1", 1000).unwrap() else {
            panic!("expected episode");
        };
        assert_eq!(event.kind, "session_done");
        assert_eq!(event.outcome, "ok");
        assert!(event.summary.contains("7 steps"));
    }

    #[test]
    fn error_and_healing_events_are_recorded() {
        let err = AgentEvent::Error { error: "boom".into() };
        let MemoryWrite::Episode { event } = reflection_write(&err, "s1", 5).unwrap() else {
            panic!("expected episode");
        };
        assert_eq!(event.kind, "agent_error");
        assert_eq!(event.outcome, "error");
        assert!(event.summary.contains("boom"));

        let heal = AgentEvent::Healing { attempt: 2, error: "timeout".into() };
        let MemoryWrite::Episode { event } = reflection_write(&heal, "s1", 6).unwrap() else {
            panic!("expected episode");
        };
        assert_eq!(event.kind, "healing");
        assert!(event.summary.contains("attempt 2"));
        assert!(event.summary.contains("timeout"));
    }

    #[test]
    fn non_reflection_events_are_skipped() {
        assert!(reflection_write(&AgentEvent::Thinking { text: "x".into() }, "s1", 1).is_none());
        assert!(reflection_write(&AgentEvent::Content { text: "y".into() }, "s1", 1).is_none());
    }

    #[test]
    fn detects_clear_corrections() {
        assert!(detect_correction("No, don't deploy to prod").is_some());
        assert!(detect_correction("stop using force push").is_some());
        assert!(detect_correction("actually, use the staging bucket").is_some());
        assert!(detect_correction("I prefer concise replies").is_some());
        assert!(detect_correction("that's wrong, the port is 8080").is_some());
        // preserves the full text
        assert_eq!(detect_correction("prefer tabs").unwrap(), "prefer tabs");
    }

    #[test]
    fn ignores_ordinary_requests() {
        assert!(detect_correction("deploy the app to staging").is_none());
        assert!(detect_correction("what's the weather in Tokyo?").is_none());
        assert!(detect_correction("create a new note").is_none());
        assert!(detect_correction("").is_none());
        assert!(detect_correction("   ").is_none());
    }
}
