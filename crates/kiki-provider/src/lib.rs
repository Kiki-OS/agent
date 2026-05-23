//! LLM provider implementations.
//!
//! The provider abstraction (`LlmProvider`, `CompletionRequest`, `StreamChunk`, etc.)
//! lives in `kiki_core::provider` to avoid circular dependencies — the harness in
//! kiki-core needs `Arc<dyn LlmProvider>`, so the trait must be defined there.
//!
//! This crate provides concrete implementations:
//! - `AnthropicProvider` — Claude via Anthropic API
//! - `OpenAIProvider`    — GPT-4o and compatible (Together, Fireworks, etc.)
//! - `LlamaCppProvider`  — `llama.cpp` `llama-server` subprocess pool (on-device)
//! - `ModalProvider`     — Modal.com hosted inference (GPU burst for heavy tasks)

pub mod anthropic;
pub mod openai;
pub mod local;
pub mod modal;
pub mod router;

pub use router::ProviderRouter;

// Re-export provider trait and types so callers only need `kiki_provider`.
pub use kiki_core::provider::{
    CompletionRequest, CompletionStream, LlmProvider,
    ProviderBlock, ProviderMessage, Role, StreamChunk, ToolSpec,
};
