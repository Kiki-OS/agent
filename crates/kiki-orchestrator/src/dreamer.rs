//! Dreamer — post-session memory consolidation (a.k.a. distillation).
//!
//! Mirrors eikarna/hermes-rs `distill_session_to_memory`:
//! - After a session ends, ask a small/fast model to extract **durable facts** from
//!   the conversation as a JSON array of strings.
//! - Each fact is stored as a `MemoryFact` in the durable state backend under
//!   `memory/{agent_id}/facts/{id}`. Future sessions can load these as preface
//!   context for continuity.
//!
//! Why strings (not a rich structured schema):
//!   The schema-free representation is robust to LLM output drift — any JSON array
//!   of strings round-trips. Structured fields (goal/outcome/decisions) break the
//!   instant the model returns extra prose or omits a field. We keep this layer
//!   permissive; richer aggregation can happen above this.

use std::sync::Arc;
use tracing::{debug, error, info, warn};
use kiki_core::{
    error::Result,
    provider::{CompletionRequest, LlmProvider, ProviderBlock, ProviderMessage, Role, StreamChunk},
    state::StateBackend,
    types::ConversationMessage,
};

// ─── Prompt ───────────────────────────────────────────────────────────────────

const DISTILL_SYSTEM: &str = "\
Analyze the conversation history. Extract ONLY permanent, durable knowledge, rules, \
and user preferences that should persist across future sessions. \
Ignore ephemeral bugs, narrative, or code snippets. \
Output ONLY a JSON array of strings — concise facts, no prose, no markdown fences.";

// ─── Fact record ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemoryFact {
    pub id:           String,
    pub agent_id:     String,
    pub session_id:   String,
    pub text:         String,
    pub created_at_ms: u64,
    /// Importance score (0-100). Distilled facts default to 90.
    pub importance:   u8,
    pub tags:         Vec<String>,
}

// ─── Dreamer ──────────────────────────────────────────────────────────────────

/// Coordinator for post-session distillation. Holds the provider + model used
/// for distillation calls so the caller doesn't have to thread them through.
pub struct Dreamer {
    model:    String,
    provider: Arc<dyn LlmProvider>,
}

impl Dreamer {
    pub fn new(model: impl Into<String>, provider: Arc<dyn LlmProvider>) -> Self {
        Self { model: model.into(), provider }
    }

    /// Fire-and-forget: spawn distillation for a completed session.
    /// Errors are logged, not propagated — distillation is a best-effort task.
    pub fn spawn(
        &self,
        session_id: String,
        agent_id:   String,
        messages:   Vec<ConversationMessage>,
        state:      Arc<dyn StateBackend>,
    ) {
        let model    = self.model.clone();
        let provider = self.provider.clone();

        tokio::spawn(async move {
            match distill(session_id.clone(), agent_id, messages, model, provider, state).await {
                Ok(count) => info!(session = %session_id, facts = count, "distillation complete"),
                Err(e)    => error!(session = %session_id, error = %e, "distillation failed"),
            }
        });
    }
}

// ─── Distillation logic ───────────────────────────────────────────────────────

async fn distill(
    session_id: String,
    agent_id:   String,
    messages:   Vec<ConversationMessage>,
    model:      String,
    provider:   Arc<dyn LlmProvider>,
    state:      Arc<dyn StateBackend>,
) -> Result<usize> {
    let transcript = format_transcript(&messages);
    if transcript.trim().is_empty() {
        debug!(session = %session_id, "empty transcript — nothing to distill");
        return Ok(0);
    }

    let request = CompletionRequest {
        model,
        messages: vec![ProviderMessage {
            role:    Role::User,
            content: vec![ProviderBlock::Text {
                text: format!("Conversation history:\n{transcript}"),
            }],
        }],
        tools:           vec![],
        max_tokens:      Some(1024),
        temperature:     Some(0.0),
        thinking_tokens: None,
        system:          Some(DISTILL_SYSTEM.into()),
    };

    // Drain stream to a single string.
    use futures::StreamExt;
    let mut stream = provider.complete(request).await?;
    let mut raw    = String::new();
    while let Some(chunk) = stream.next().await {
        if let Ok(StreamChunk::Text(t)) = chunk { raw.push_str(&t); }
    }

    let facts = parse_facts(&raw);
    if facts.is_empty() {
        warn!(session = %session_id, raw = %truncate(&raw, 200), "no facts parsed");
        return Ok(0);
    }

    // Dedup + persist.
    let mut seen   = std::collections::HashSet::new();
    let mut stored = 0;
    for (i, text) in facts.into_iter().enumerate() {
        let trimmed = text.trim().to_string();
        if trimmed.is_empty() || !seen.insert(trimmed.clone()) { continue; }

        let fact = MemoryFact {
            id:            fact_id(&trimmed, i),
            agent_id:      agent_id.clone(),
            session_id:    session_id.clone(),
            text:          trimmed,
            created_at_ms: now_ms(),
            importance:    90,
            tags:          vec!["distilled".into(), "long_term".into()],
        };
        let key = format!("memory/{}/facts/{}", agent_id, fact.id);
        state.set(&key, serde_json::to_value(&fact)?).await?;
        stored += 1;
    }
    Ok(stored)
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn format_transcript(messages: &[ConversationMessage]) -> String {
    messages.iter().filter_map(|m| match m {
        ConversationMessage::System { .. } => None,
        ConversationMessage::User { content, .. } => {
            let text: String = content.iter().filter_map(|b| {
                if let kiki_core::types::ContentBlock::Text { text } = b {
                    Some(text.as_str())
                } else { None }
            }).collect::<Vec<_>>().join(" ");
            if text.is_empty() { None } else { Some(format!("user: {text}")) }
        }
        ConversationMessage::Assistant(turn) => {
            turn.text.as_deref().map(|t| format!("assistant: {t}"))
        }
        ConversationMessage::ToolResults { .. } => None,
    }).collect::<Vec<_>>().join("\n")
}

/// Tolerant fact parser: tries strict JSON array first, falls back to extracting
/// the largest `[...]` substring (handles models that wrap output in prose).
fn parse_facts(raw: &str) -> Vec<String> {
    let trimmed = raw.trim();

    // Strict path.
    if let Ok(v) = serde_json::from_str::<Vec<String>>(trimmed) { return v; }

    // Fenced markdown code block ```json ... ```
    if let Some(stripped) = strip_code_fence(trimmed) {
        if let Ok(v) = serde_json::from_str::<Vec<String>>(stripped) { return v; }
    }

    // Find the largest [...] substring and try.
    if let (Some(start), Some(end)) = (trimmed.find('['), trimmed.rfind(']')) {
        if end > start {
            let slice = &trimmed[start..=end];
            if let Ok(v) = serde_json::from_str::<Vec<String>>(slice) { return v; }
        }
    }

    vec![]
}

fn strip_code_fence(s: &str) -> Option<&str> {
    let s = s.strip_prefix("```json").or_else(|| s.strip_prefix("```"))?;
    s.strip_suffix("```").map(str::trim)
}

fn fact_id(text: &str, index: usize) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut h);
    format!("distilled_{}_{}_{}", now_ms() / 1000, index, h.finish())
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n { s.into() } else { format!("{}…", &s[..n]) }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_strict_array() {
        let v = parse_facts("[\"user prefers Rust\",\"deploys via OSTree\"]");
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn parses_array_in_prose() {
        let v = parse_facts("Here are the facts:\n[\"a\",\"b\"]\nThat's it.");
        assert_eq!(v, vec!["a", "b"]);
    }

    #[test]
    fn parses_fenced_array() {
        let v = parse_facts("```json\n[\"a\"]\n```");
        assert_eq!(v, vec!["a"]);
    }

    #[test]
    fn empty_on_garbage() {
        assert!(parse_facts("not json at all").is_empty());
    }
}
