use async_trait::async_trait;
use serde_json::Value;
use std::{collections::HashMap, sync::Arc};
use crate::error::Result;

/// A callable tool exposed to the agent.
///
/// Tools are registered in the MCP hub and filtered to ~20-30 per task to avoid
/// LLM degradation. `input_schema` is a JSON Schema object (draft-07 subset
/// compatible with Anthropic, OpenAI, and Ollama tool APIs).
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn input_schema(&self) -> &Value;
    async fn call(&self, input: Value) -> Result<ToolOutput>;
}

#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub content:  Value,
    pub is_error: bool,
}

impl ToolOutput {
    pub fn ok(content: impl Into<Value>) -> Self {
        Self { content: content.into(), is_error: false }
    }
    pub fn err(content: impl Into<String>) -> Self {
        Self { content: Value::String(content.into()), is_error: true }
    }
}

/// Central registry — the MCP hub feeds tools into this.
/// Semantic filtering happens here before passing ~20-30 tools to the LLM.
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: HashMap::new() }
    }

    pub fn register(&mut self, tool: impl Tool + 'static) {
        let arc: Arc<dyn Tool> = Arc::new(tool);
        self.tools.insert(arc.name().to_string(), arc);
    }

    pub fn register_arc(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.get(name)
    }

    /// Returns the N most semantically relevant tools for a given query.
    /// Prevents LLM performance degradation from large tool contexts.
    pub fn filter(&self, _query: &str, limit: usize) -> Vec<&Arc<dyn Tool>> {
        // TODO: embed query + tool descriptions, cosine similarity filter
        self.tools.values().take(limit).collect()
    }

    pub fn len(&self) -> usize { self.tools.len() }
    pub fn is_empty(&self) -> bool { self.tools.is_empty() }
}

impl Default for ToolRegistry {
    fn default() -> Self { Self::new() }
}
