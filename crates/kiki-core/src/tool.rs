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

    /// Returns the `limit` most relevant tools for `query`, highest-scoring first.
    ///
    /// Prevents LLM degradation from oversized tool contexts. Relevance is a
    /// lexical overlap score between the query tokens and each tool's name and
    /// description (no embedding model required, fully deterministic). When the
    /// registry already fits within `limit`, all tools are returned sorted by
    /// name for stable ordering.
    pub fn filter(&self, query: &str, limit: usize) -> Vec<&Arc<dyn Tool>> {
        if self.tools.len() <= limit {
            let mut all: Vec<&Arc<dyn Tool>> = self.tools.values().collect();
            all.sort_by(|a, b| a.name().cmp(b.name()));
            return all;
        }

        let q_tokens = tokenize(query);
        let mut scored: Vec<(i32, &Arc<dyn Tool>)> = self.tools.values()
            .map(|t| (relevance_score(&q_tokens, t.name(), t.description()), t))
            .collect();
        // Highest score first; ties broken by name for determinism.
        scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.name().cmp(b.1.name())));
        scored.into_iter().take(limit).map(|(_, t)| t).collect()
    }

    pub fn len(&self) -> usize { self.tools.len() }
    pub fn is_empty(&self) -> bool { self.tools.is_empty() }
}

impl Default for ToolRegistry {
    fn default() -> Self { Self::new() }
}

// ─── Lexical relevance scoring ──────────────────────────────────────────────

/// Common words that carry no tool-selection signal.
const STOPWORDS: &[&str] = &[
    "the", "a", "an", "to", "of", "in", "on", "for", "and", "or", "is", "are",
    "what", "how", "do", "does", "can", "you", "i", "me", "my", "please", "use",
    "with", "this", "that", "it", "be", "get", "want", "need", "from", "at",
];

fn tokenize(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(|t| t.to_ascii_lowercase())
        .filter(|t| !STOPWORDS.contains(&t.as_str()))
        .collect()
}

/// Score a tool against pre-tokenized query terms.
/// Name matches weigh more than description matches; exact token matches weigh
/// more than substring matches.
fn relevance_score(q_tokens: &[String], name: &str, desc: &str) -> i32 {
    if q_tokens.is_empty() { return 0; }
    let name_lc = name.to_ascii_lowercase();
    let desc_lc = desc.to_ascii_lowercase();
    let name_tokens = tokenize(name);
    let desc_tokens = tokenize(desc);

    let mut score = 0;
    for qt in q_tokens {
        if name_tokens.iter().any(|t| t == qt)       { score += 5; }
        else if name_lc.contains(qt.as_str())        { score += 3; }
        if desc_tokens.iter().any(|t| t == qt)       { score += 2; }
        else if desc_lc.contains(qt.as_str())        { score += 1; }
    }
    score
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    struct FakeTool { name: String, desc: String, schema: Value }
    #[async_trait]
    impl Tool for FakeTool {
        fn name(&self) -> &str { &self.name }
        fn description(&self) -> &str { &self.desc }
        fn input_schema(&self) -> &Value { &self.schema }
        async fn call(&self, _input: Value) -> Result<ToolOutput> {
            Ok(ToolOutput::ok("ok"))
        }
    }

    fn tool(name: &str, desc: &str) -> FakeTool {
        FakeTool { name: name.into(), desc: desc.into(), schema: json!({"type":"object"}) }
    }

    fn registry() -> ToolRegistry {
        let mut r = ToolRegistry::new();
        r.register(tool("get_weather",   "Get the current weather forecast for a city"));
        r.register(tool("read_file",     "Read the contents of a file from disk"));
        r.register(tool("write_file",    "Write contents to a file on disk"));
        r.register(tool("send_email",    "Send an email message to a recipient"));
        r.register(tool("list_process",  "List the running processes on the system"));
        r.register(tool("play_audio",    "Play an audio track through the speakers"));
        r
    }

    #[test]
    fn returns_all_sorted_when_within_limit() {
        let r = registry();
        let got: Vec<&str> = r.filter("anything", 100).iter().map(|t| t.name()).collect();
        assert_eq!(got, vec!["get_weather","list_process","play_audio","read_file","send_email","write_file"]);
    }

    #[test]
    fn ranks_weather_query_first() {
        let r = registry();
        let got: Vec<&str> = r.filter("what is the weather in Tokyo?", 2).iter().map(|t| t.name()).collect();
        assert_eq!(got[0], "get_weather", "weather tool must rank first, got {got:?}");
    }

    #[test]
    fn ranks_file_query() {
        let r = registry();
        let got: Vec<&str> = r.filter("read the config file from disk", 3).iter().map(|t| t.name()).collect();
        assert!(got.contains(&"read_file"), "file tools should rank, got {got:?}");
        assert_eq!(got.len(), 3);
    }

    #[test]
    fn respects_limit() {
        let r = registry();
        assert_eq!(r.filter("email", 1).len(), 1);
        assert_eq!(r.filter("email", 1)[0].name(), "send_email");
    }
}
