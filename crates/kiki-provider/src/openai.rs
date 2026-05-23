//! OpenAI API provider (gpt-* models). Not yet implemented.

use super::{CompletionRequest, CompletionStream, LlmProvider};
use async_trait::async_trait;
use kiki_core::error::{Error, Result};

pub struct OpenAiProvider {
    api_key: String,
}

impl OpenAiProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self { api_key: api_key.into() }
    }
}

#[async_trait]
impl LlmProvider for OpenAiProvider {
    fn name(&self) -> &str { "openai" }

    fn supports_model(&self, model: &str) -> bool {
        model.starts_with("gpt-") || model.starts_with("o1") || model.starts_with("o3")
    }

    async fn complete(&self, _request: CompletionRequest) -> Result<CompletionStream> {
        Err(Error::Provider("openai provider not yet implemented".into()))
    }
}
