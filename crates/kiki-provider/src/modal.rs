//! Modal.com cloud inference provider (open models via vLLM). Not yet implemented.

use super::{CompletionRequest, CompletionStream, LlmProvider};
use async_trait::async_trait;
use kiki_core::error::{Error, Result};

pub struct ModalProvider {
    endpoint: String,
    token: String,
}

impl ModalProvider {
    pub fn new(endpoint: impl Into<String>, token: impl Into<String>) -> Self {
        Self { endpoint: endpoint.into(), token: token.into() }
    }
}

#[async_trait]
impl LlmProvider for ModalProvider {
    fn name(&self) -> &str { "modal" }

    fn supports_model(&self, model: &str) -> bool {
        model.starts_with("modal/")
    }

    async fn complete(&self, _request: CompletionRequest) -> Result<CompletionStream> {
        Err(Error::Provider("modal provider not yet implemented".into()))
    }
}
