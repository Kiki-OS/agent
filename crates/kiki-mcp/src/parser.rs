//! Tolerant streaming tool-call parser.
//!
//! Adapted from eikarna/hermes-rs `crates/hermes-core/src/parser.rs` — character-by-character
//! state machine that detects `<tool_call>{...}</tool_call>` blocks in streaming LLM output
//! and tolerates malformed XML/JSON (common with local models).
//!
//! Dual-path detection:
//! - **Native path:** providers that return `StreamChunk::ToolCall` natively (Anthropic, OpenAI).
//!   The parser is not invoked.
//! - **XML path:** providers that emit tool calls inline as text (Ollama with Hermes-format
//!   models). The parser scans streamed text and emits `ParsedChunk::ToolCall` as soon as the
//!   closing `</tool_call>` arrives — before the full LLM response completes.
//!
//! Tolerant strategies (in order):
//! 1. Direct `serde_json::from_str` of the inner content.
//! 2. Regex scan for the largest JSON object substring (handles surrounding noise).
//! 3. Aggressive regex extraction of `"name"` / `"arguments"` fields (handles malformed JSON).

use kiki_core::types::ToolCall;
use lazy_static::lazy_static;
use regex::Regex;
use serde_json::Value;
use tracing::{debug, warn};

// ─── Regexes for tolerant extraction ─────────────────────────────────────────

lazy_static! {
    /// Match a JSON object: one level of nesting tolerated.
    static ref JSON_RE: Regex =
        Regex::new(r#"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"#).expect("JSON_RE compile");
    /// Match a "name" or "function" JSON field with a string value.
    static ref NAME_RE: Regex =
        Regex::new(r#""(?:name|function)":\s*"([^"]+)""#).expect("NAME_RE compile");
    /// Match an "arguments" field (object or string).
    static ref ARGS_RE: Regex =
        Regex::new(r#""arguments":\s*("\{[^}]*\}"|\{[^}]*\}|"[^"]*")"#).expect("ARGS_RE compile");
    /// Match an "input" field — same shape as arguments (Anthropic naming).
    static ref INPUT_RE: Regex =
        Regex::new(r#""input":\s*("\{[^}]*\}"|\{[^}]*\}|"[^"]*")"#).expect("INPUT_RE compile");
}

// ─── Parser events ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum ParsedChunk {
    /// Printable text that should be shown to the user / stored in the turn.
    Text(String),
    /// A fully parsed tool call extracted from a `<tool_call>` block.
    ToolCall(ToolCall),
    /// Parser-level failure (kept as a chunk so the harness can log/surface it).
    Error(String),
}

// ─── State machine ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Outside any tag — accumulate text.
    Outside,
    /// Read `<` — accumulating tag name until `>`.
    InsideOpenTag,
    /// Inside the body of a `<tool_call>` — accumulating JSON content.
    InsideContent,
    /// Read `<` inside content — could be a nested tag or the closing tag.
    InsideNestedTag,
}

pub struct ToolCallParser {
    state:         State,
    /// Accumulator for text (when Outside) or tool-call body (when InsideContent).
    buffer:        String,
    /// Accumulator for the current tag name.
    tag_buffer:    String,
    /// Nesting depth tracker for nested `<...>` inside content.
    nested_depth:  usize,
    /// Synthetic id counter for tool calls that don't include their own id.
    id_counter:    u64,
}

impl ToolCallParser {
    pub fn new() -> Self {
        Self {
            state:        State::Outside,
            buffer:       String::new(),
            tag_buffer:   String::new(),
            nested_depth: 0,
            id_counter:   0,
        }
    }

    /// Feed a chunk of streamed text. Returns events ready for emission.
    pub fn feed(&mut self, data: &str) -> Vec<ParsedChunk> {
        let mut out = Vec::new();
        for ch in data.chars() {
            self.step(ch, &mut out);
        }
        // Emit accumulated outside-text as a Text chunk if we're back to Outside.
        // Keep it accumulating across calls only if we might still be in a partial tag.
        if self.state == State::Outside && !self.buffer.is_empty() {
            out.push(ParsedChunk::Text(std::mem::take(&mut self.buffer)));
        }
        out
    }

    /// Flush any pending buffer (call at end of stream).
    pub fn finish(&mut self) -> Vec<ParsedChunk> {
        let mut out = Vec::new();
        // If we were mid-tool_call, surface as error rather than losing it.
        match self.state {
            State::Outside => {
                if !self.buffer.is_empty() {
                    out.push(ParsedChunk::Text(std::mem::take(&mut self.buffer)));
                }
            }
            State::InsideContent | State::InsideNestedTag => {
                let leftover = std::mem::take(&mut self.buffer);
                warn!(buf = %leftover, "tool_call block not closed at end of stream");
                out.push(ParsedChunk::Error(format!(
                    "unterminated <tool_call> block (got {} chars)", leftover.len()
                )));
            }
            State::InsideOpenTag => {
                let leftover = format!("<{}", std::mem::take(&mut self.tag_buffer));
                out.push(ParsedChunk::Text(leftover));
            }
        }
        self.reset();
        out
    }

    /// Reset the parser. Discards any buffered state.
    pub fn reset(&mut self) {
        self.state        = State::Outside;
        self.buffer.clear();
        self.tag_buffer.clear();
        self.nested_depth = 0;
    }

    // ── Per-character step ────────────────────────────────────────────────────

    fn step(&mut self, ch: char, out: &mut Vec<ParsedChunk>) {
        match self.state {
            State::Outside => {
                if ch == '<' {
                    self.state = State::InsideOpenTag;
                    self.tag_buffer.clear();
                } else {
                    self.buffer.push(ch);
                }
            }

            State::InsideOpenTag => {
                if ch == '>' {
                    let tag = self.tag_buffer.trim().to_lowercase();
                    self.tag_buffer.clear();
                    if tag.starts_with("tool_call") && !tag.starts_with('/') {
                        // Flush any outside-text before entering content.
                        if !self.buffer.is_empty() {
                            out.push(ParsedChunk::Text(std::mem::take(&mut self.buffer)));
                        }
                        self.state = State::InsideContent;
                    } else {
                        // Not a tool_call tag — push back as raw text and resume Outside.
                        self.buffer.push('<');
                        self.buffer.push_str(&tag);
                        self.buffer.push('>');
                        self.state = State::Outside;
                    }
                } else if ch == '<' {
                    // Spurious '<' — treat preceding accumulation as text.
                    self.buffer.push('<');
                    self.buffer.push_str(&self.tag_buffer);
                    self.tag_buffer.clear();
                } else {
                    self.tag_buffer.push(ch);
                }
            }

            State::InsideContent => {
                if ch == '<' {
                    self.state        = State::InsideNestedTag;
                    self.tag_buffer.clear();
                    self.nested_depth = 1;
                } else {
                    self.buffer.push(ch);
                }
            }

            State::InsideNestedTag => {
                if ch == '<' {
                    self.nested_depth += 1;
                    self.tag_buffer.push(ch);
                } else if ch == '>' {
                    self.nested_depth -= 1;
                    if self.nested_depth == 0 {
                        let nested = self.tag_buffer.trim().to_lowercase();
                        self.tag_buffer.clear();

                        if nested.starts_with("/tool_call") {
                            // Closing tag — parse the accumulated content.
                            self.emit_tool_call(out);
                            self.state = State::Outside;
                        } else if nested == "tool_call" {
                            // Nested <tool_call> (malformed) — push back as content.
                            warn!("malformed XML: nested <tool_call>");
                            self.buffer.push('<');
                            self.buffer.push_str(&nested);
                            self.buffer.push('>');
                            self.state = State::InsideContent;
                        } else {
                            // Some other tag inside content — keep it verbatim.
                            self.buffer.push('<');
                            self.buffer.push_str(&nested);
                            self.buffer.push('>');
                            self.state = State::InsideContent;
                        }
                    } else {
                        self.tag_buffer.push(ch);
                    }
                } else {
                    self.tag_buffer.push(ch);
                }
            }
        }
    }

    // ── Tool call extraction (tolerant) ───────────────────────────────────────

    fn emit_tool_call(&mut self, out: &mut Vec<ParsedChunk>) {
        let content = std::mem::take(&mut self.buffer);
        let trimmed = content.trim();
        if trimmed.is_empty() {
            debug!("empty tool_call block — ignoring");
            return;
        }

        // Strategy 1: parse the whole block as JSON.
        if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
            if let Some(call) = self.from_json(&v) {
                out.push(ParsedChunk::ToolCall(call));
                return;
            }
        }

        // Strategy 2: scan for the largest valid JSON object substring.
        for m in JSON_RE.find_iter(trimmed) {
            if let Ok(v) = serde_json::from_str::<Value>(m.as_str()) {
                if let Some(call) = self.from_json(&v) {
                    out.push(ParsedChunk::ToolCall(call));
                    return;
                }
            }
        }

        // Strategy 3: aggressive regex extraction.
        if let Some(call) = self.from_regex(trimmed) {
            out.push(ParsedChunk::ToolCall(call));
            return;
        }

        warn!(content = %truncate(trimmed, 120), "failed to parse tool_call content");
        out.push(ParsedChunk::Error(format!(
            "failed to parse tool_call: {}",
            truncate(trimmed, 120)
        )));
    }

    fn from_json(&mut self, v: &Value) -> Option<ToolCall> {
        let name = v.get("name").and_then(Value::as_str)?;
        // "input" (Anthropic) | "arguments" (OpenAI/Hermes) | "parameters"
        let input = v.get("input")
            .or_else(|| v.get("arguments"))
            .or_else(|| v.get("parameters"))
            .cloned()
            .map(|raw| match raw {
                // Some local models emit arguments as a JSON-encoded string.
                Value::String(s) => serde_json::from_str(&s).unwrap_or(Value::String(s)),
                other => other,
            })
            .unwrap_or(Value::Object(Default::default()));

        let id = v.get("id").and_then(Value::as_str)
            .map(String::from)
            .unwrap_or_else(|| self.next_id());

        Some(ToolCall { id, name: name.to_string(), input })
    }

    fn from_regex(&mut self, content: &str) -> Option<ToolCall> {
        let name = NAME_RE.captures(content)
            .and_then(|c| c.get(1))
            .map(|m| m.as_str().to_string())?;

        let args_str = ARGS_RE.captures(content)
            .or_else(|| INPUT_RE.captures(content))
            .and_then(|c| c.get(1).map(|m| m.as_str().to_string()))
            .unwrap_or_else(|| "{}".to_string());

        let input: Value = serde_json::from_str(&args_str)
            .or_else(|_| {
                // The captured group may itself be a JSON-encoded string.
                if args_str.starts_with('"') && args_str.ends_with('"') {
                    let unquoted = &args_str[1..args_str.len() - 1];
                    serde_json::from_str(unquoted)
                } else {
                    Ok(Value::Object(Default::default()))
                }
            })
            .unwrap_or(Value::Object(Default::default()));

        Some(ToolCall { id: self.next_id(), name, input })
    }

    fn next_id(&mut self) -> String {
        self.id_counter += 1;
        format!("xml-{}", self.id_counter)
    }
}

impl Default for ToolCallParser {
    fn default() -> Self { Self::new() }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max { s.to_string() } else { format!("{}…", &s[..max]) }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_full(s: &str) -> Vec<ParsedChunk> {
        let mut p = ToolCallParser::new();
        let mut out = p.feed(s);
        out.extend(p.finish());
        out
    }

    #[test]
    fn happy_path() {
        let out = parse_full(
            r#"Let me check.<tool_call>{"name":"fs_read","arguments":{"path":"/etc/hosts"}}</tool_call> Done."#
        );
        assert!(out.iter().any(|c| matches!(c, ParsedChunk::ToolCall(t) if t.name == "fs_read")));
        let texts: Vec<_> = out.iter().filter_map(|c|
            if let ParsedChunk::Text(t) = c { Some(t.as_str()) } else { None }
        ).collect();
        assert!(texts.iter().any(|t| t.contains("Let me check")));
        assert!(texts.iter().any(|t| t.contains("Done")));
    }

    #[test]
    fn split_across_chunks() {
        let mut p = ToolCallParser::new();
        let mut all = p.feed(r#"<tool_call>{"name":"fs_"#);
        all.extend(p.feed(r#"read","arguments":{}}</tool_call>"#));
        all.extend(p.finish());
        assert!(all.iter().any(|c| matches!(c, ParsedChunk::ToolCall(t) if t.name == "fs_read")));
    }

    #[test]
    fn anthropic_input_naming() {
        let out = parse_full(
            r#"<tool_call>{"name":"fs_read","input":{"path":"/a"}}</tool_call>"#
        );
        assert!(out.iter().any(|c| matches!(c, ParsedChunk::ToolCall(t)
            if t.name == "fs_read" && t.input["path"] == "/a"
        )));
    }

    #[test]
    fn arguments_as_string() {
        // Some local models stringify the arguments object.
        let out = parse_full(
            r#"<tool_call>{"name":"fs_read","arguments":"{\"path\":\"/a\"}"}</tool_call>"#
        );
        assert!(out.iter().any(|c| matches!(c, ParsedChunk::ToolCall(t)
            if t.name == "fs_read" && t.input["path"] == "/a"
        )));
    }

    #[test]
    fn malformed_falls_back_to_regex() {
        // Missing closing brace — JSON parse fails, regex still extracts name.
        let out = parse_full(
            r#"<tool_call>{"name":"fs_read", "arguments": {"path": "/a"</tool_call>"#
        );
        assert!(out.iter().any(|c| matches!(c, ParsedChunk::ToolCall(t) if t.name == "fs_read")));
    }

    #[test]
    fn unrelated_tag_passes_through() {
        let out = parse_full("hello <thinking>foo</thinking> world");
        let text: String = out.iter().filter_map(|c|
            if let ParsedChunk::Text(t) = c { Some(t.as_str()) } else { None }
        ).collect::<Vec<_>>().join("");
        assert!(text.contains("hello"));
        assert!(text.contains("world"));
    }

    #[test]
    fn unterminated_block_surfaces_error() {
        let out = parse_full(r#"<tool_call>{"name":"fs_read""#);
        assert!(out.iter().any(|c| matches!(c, ParsedChunk::Error(_))));
    }
}
