//! Anthropic API provider (claude-* models).
//!
//! Uses the Anthropic Messages API with streaming.
//! Supports extended thinking (claude-3-7-*), tool use, and vision.
//! TODO: implement streaming; stub returns error for now.

use kiki_core::{
    error::{Error, Result},
    provider::{CompletionRequest, CompletionStream, LlmProvider},
};
use async_trait::async_trait;

pub struct AnthropicProvider {
    api_key: String,
}

impl AnthropicProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self { api_key: api_key.into() }
    }
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    fn name(&self) -> &str { "anthropic" }

    fn supports_model(&self, model: &str) -> bool {
        model.starts_with("claude-")
    }

    async fn complete(&self, _request: CompletionRequest) -> Result<CompletionStream> {
        Err(Error::Provider("anthropic provider not yet implemented".into()))
    }
}
