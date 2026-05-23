//! Provider router — selects an [`LlmProvider`] per request based on the node's
//! routing policy and the requested model.
//!
//! `agentd` builds a router from the providers it has available (always the local
//! llama.cpp runtime; optionally cloud providers when API keys are present) and
//! hands the harness a single `Arc<dyn LlmProvider>`. The harness is unaware of
//! routing — it just calls `complete`.
//!
//! Selection rules:
//! - If `allow_remote == false`, remote providers ([`LlmProvider::is_remote`]) are
//!   never selected — the device stays local-only regardless of the model id.
//! - Among eligible providers, the first one (in registration order) whose
//!   `supports_model` matches wins. Register more specific/preferred providers first.

use std::sync::Arc;

use async_trait::async_trait;
use kiki_core::{
    error::{Error, Result},
    provider::{CompletionRequest, CompletionStream, LlmProvider},
};

pub struct ProviderRouter {
    providers:    Vec<Arc<dyn LlmProvider>>,
    allow_remote: bool,
}

impl ProviderRouter {
    pub fn new(allow_remote: bool) -> Self {
        Self { providers: Vec::new(), allow_remote }
    }

    /// Register a provider. Order matters: earlier = higher priority.
    pub fn with(mut self, provider: Arc<dyn LlmProvider>) -> Self {
        self.providers.push(provider);
        self
    }

    pub fn add(&mut self, provider: Arc<dyn LlmProvider>) {
        self.providers.push(provider);
    }

    pub fn provider_count(&self) -> usize { self.providers.len() }

    /// Pick the provider that should serve `model`, honoring the remote policy.
    pub fn select(&self, model: &str) -> Option<Arc<dyn LlmProvider>> {
        self.providers.iter()
            .filter(|p| self.allow_remote || !p.is_remote())
            .find(|p| p.supports_model(model))
            .cloned()
    }
}

#[async_trait]
impl LlmProvider for ProviderRouter {
    fn name(&self) -> &str { "router" }

    fn supports_model(&self, model: &str) -> bool {
        self.select(model).is_some()
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionStream> {
        match self.select(&request.model) {
            Some(p) => p.complete(request).await,
            None => Err(Error::Provider(format!(
                "no provider for model `{}` (allow_remote={})",
                request.model, self.allow_remote
            ))),
        }
    }

    async fn count_tokens(&self, request: &CompletionRequest) -> Option<u32> {
        self.select(&request.model)?.count_tokens(request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiki_core::provider::StreamChunk;
    use futures::stream;

    struct FakeProvider { name: &'static str, prefix: &'static str, remote: bool }
    #[async_trait]
    impl LlmProvider for FakeProvider {
        fn name(&self) -> &str { self.name }
        fn is_remote(&self) -> bool { self.remote }
        fn supports_model(&self, model: &str) -> bool { model.starts_with(self.prefix) }
        async fn complete(&self, _r: CompletionRequest) -> Result<CompletionStream> {
            let n = self.name;
            Ok(Box::pin(stream::iter(vec![Ok(StreamChunk::Text(n.to_string())), Ok(StreamChunk::Done)])))
        }
    }

    fn local() -> Arc<dyn LlmProvider> {
        Arc::new(FakeProvider { name: "local", prefix: "llama", remote: false })
    }
    fn cloud() -> Arc<dyn LlmProvider> {
        Arc::new(FakeProvider { name: "cloud", prefix: "claude-", remote: true })
    }

    #[test]
    fn selects_local_for_local_model() {
        let r = ProviderRouter::new(true).with(local()).with(cloud());
        assert_eq!(r.select("llama-3.1-8b").unwrap().name(), "local");
        assert_eq!(r.select("claude-sonnet-4-6").unwrap().name(), "cloud");
    }

    #[test]
    fn remote_suppressed_when_disallowed() {
        let r = ProviderRouter::new(false).with(local()).with(cloud());
        // local still works
        assert!(r.select("llama-3.1-8b").is_some());
        // cloud model has no eligible provider → None (stays local-only)
        assert!(r.select("claude-sonnet-4-6").is_none());
        assert!(!r.supports_model("claude-sonnet-4-6"));
    }

    #[test]
    fn priority_order_first_match_wins() {
        // two providers both matching "gpt-"; first registered wins
        let a: Arc<dyn LlmProvider> = Arc::new(FakeProvider { name: "a", prefix: "gpt-", remote: true });
        let b: Arc<dyn LlmProvider> = Arc::new(FakeProvider { name: "b", prefix: "gpt-", remote: true });
        let r = ProviderRouter::new(true).with(a).with(b);
        assert_eq!(r.select("gpt-4o").unwrap().name(), "a");
    }
}
