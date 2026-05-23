//! Context window management — incremental token tracking + compaction.
//!
//! Adapted from eikarna/hermes-rs `crates/hermes-core/src/context.rs`:
//! - `VecDeque<MessageWithTokens>` so `pop_front()` is O(1)
//! - `total_tokens` running counter (no recomputation on every check)
//! - `pop_front()` until budget fits, preserving the last N messages
//!
//! Kiki adaptations:
//! - Operates on `ConversationMessage` (multi-block + tool results), not flat strings
//! - Token estimation aware of tool-call blocks (extra overhead per call)
//! - Always preserves the System message (slot 0) regardless of budget pressure

use std::collections::VecDeque;
use crate::types::{ConversationMessage, ContentBlock};

// ─── Config ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct ContextConfig {
    /// Maximum context length in tokens (model's window).
    pub max_context_length:    usize,
    /// Reserved tokens for the next response.
    pub response_buffer:       usize,
    /// Minimum non-system messages to keep, even under budget pressure.
    pub min_messages_preserve: usize,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_context_length:    128_000,
            response_buffer:       4_096,
            min_messages_preserve: 4,
        }
    }
}

// ─── Bookkeeping ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Tracked {
    message: ConversationMessage,
    tokens:  usize,
}

// ─── Manager ──────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct ContextManager {
    config:       ContextConfig,
    /// System prompt slot — always preserved, never compacted.
    system:       Option<Tracked>,
    /// All non-system messages in order.
    history:      VecDeque<Tracked>,
    /// Running token total (system + history).
    total_tokens: usize,
}

impl ContextManager {
    pub fn new(config: ContextConfig) -> Self {
        Self { config, system: None, history: VecDeque::new(), total_tokens: 0 }
    }

    /// Append a message and track its token cost.
    pub fn push(&mut self, msg: ConversationMessage) {
        let tokens = estimate_message_tokens(&msg);
        self.total_tokens += tokens;

        match &msg {
            ConversationMessage::System { .. } => {
                if let Some(old) = self.system.take() {
                    self.total_tokens = self.total_tokens.saturating_sub(old.tokens);
                }
                self.system = Some(Tracked { message: msg, tokens });
            }
            _ => self.history.push_back(Tracked { message: msg, tokens }),
        }
    }

    /// Replace the manager's full history at once (used when loading a snapshot).
    pub fn replace_all(&mut self, messages: Vec<ConversationMessage>) {
        self.system       = None;
        self.history.clear();
        self.total_tokens = 0;
        for msg in messages { self.push(msg); }
    }

    /// Current ordered view: `[system?, ...history]`.
    pub fn messages(&self) -> Vec<ConversationMessage> {
        let mut out = Vec::with_capacity(self.history.len() + 1);
        if let Some(s) = &self.system { out.push(s.message.clone()); }
        out.extend(self.history.iter().map(|t| t.message.clone()));
        out
    }

    pub fn len(&self)      -> usize { self.history.len() + self.system.is_some() as usize }
    pub fn is_empty(&self) -> bool  { self.history.is_empty() && self.system.is_none() }
    pub fn token_count(&self) -> usize { self.total_tokens }

    /// Available budget for the next prompt (excludes response buffer).
    pub fn budget(&self) -> usize {
        self.config.max_context_length.saturating_sub(self.config.response_buffer)
    }

    pub fn needs_compaction(&self) -> bool {
        self.total_tokens > self.budget()
    }

    /// Drop oldest non-system messages until the budget fits, keeping at least
    /// `min_messages_preserve` and never touching the system prompt.
    ///
    /// Returns the number of messages dropped.
    pub fn compact(&mut self) -> usize {
        let budget = self.budget();
        let mut dropped = 0;
        while self.total_tokens > budget && self.history.len() > self.config.min_messages_preserve {
            if let Some(front) = self.history.pop_front() {
                self.total_tokens = self.total_tokens.saturating_sub(front.tokens);
                dropped += 1;
            } else { break; }
        }
        dropped
    }
}

// ─── Token estimation ─────────────────────────────────────────────────────────

/// ~4 chars per token is a rough estimate; sufficient for budget-based compaction.
/// Providers with a `count_tokens` API can refine this asynchronously.
pub fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(4)
}

pub fn estimate_message_tokens(msg: &ConversationMessage) -> usize {
    const ROLE_OVERHEAD: usize = 4;
    const TOOL_OVERHEAD: usize = 10;

    let body = match msg {
        ConversationMessage::System { content } => estimate_tokens(content),

        ConversationMessage::User { content, .. } => content.iter().map(|b| match b {
            ContentBlock::Text { text }            => estimate_tokens(text),
            ContentBlock::Image { data_base64, .. } => estimate_tokens(data_base64) / 2, // image tokens ≈ half raw size
            other => estimate_tokens(&serde_json::to_string(other).unwrap_or_default()),
        }).sum(),

        ConversationMessage::Assistant(turn) => {
            let thinking = turn.thinking.as_deref().map_or(0, estimate_tokens);
            let text     = turn.text.as_deref().map_or(0, estimate_tokens);
            let tool_calls: usize = turn.tool_calls.iter().map(|c| {
                TOOL_OVERHEAD + estimate_tokens(&c.name) +
                estimate_tokens(&serde_json::to_string(&c.input).unwrap_or_default())
            }).sum();
            thinking + text + tool_calls
        }

        ConversationMessage::ToolResults { results } => results.iter()
            .map(|r| TOOL_OVERHEAD + estimate_tokens(&r.content))
            .sum(),
    };

    body + ROLE_OVERHEAD
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AssistantTurn;

    fn user(text: &str) -> ConversationMessage { ConversationMessage::user_text(text) }
    fn assistant(text: &str) -> ConversationMessage {
        ConversationMessage::Assistant(AssistantTurn {
            text: Some(text.into()), ..Default::default()
        })
    }

    #[test]
    fn token_count_is_incremental() {
        let mut m = ContextManager::new(ContextConfig::default());
        let n0 = m.token_count();
        m.push(user("hello world"));
        let n1 = m.token_count();
        assert!(n1 > n0);
        m.push(assistant("hi"));
        assert!(m.token_count() > n1);
    }

    #[test]
    fn compaction_preserves_system_and_min_messages() {
        let mut m = ContextManager::new(ContextConfig {
            max_context_length:    40,   // tiny budget to force compaction
            response_buffer:       10,
            min_messages_preserve: 2,
        });
        m.push(ConversationMessage::system("you are kiki"));
        for i in 0..10 {
            m.push(user(&format!("turn {i} with a fair amount of text padding here")));
        }
        let dropped = m.compact();
        assert!(dropped > 0);
        let msgs = m.messages();
        // System always first.
        assert!(matches!(msgs.first(), Some(ConversationMessage::System { .. })));
        // At least min_messages_preserve non-system messages.
        let non_sys = msgs.iter().filter(|m| !matches!(m, ConversationMessage::System { .. })).count();
        assert!(non_sys >= 2);
    }
}
