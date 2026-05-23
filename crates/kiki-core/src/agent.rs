//! AgentConfig — what a specific agent contributes to the harness.
//!
//! The harness owns the conversation loop; agents provide the "what" policy:
//! system prompt, required capabilities, and session metadata. There is no
//! perceive/reason/act split here — the harness handles the full turn structure.
//!
//! For the full harness loop and Perceptor trait, see harness.rs.

pub use crate::harness::AgentConfig;
