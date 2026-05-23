//! Desktop session manager, DAG task scheduler, event bus, multi-agent coordination.
//!
//! Desktop sessions are first-class: each is an independent Wayland surface with
//! its own agent, tool context, and OSTree state. Sessions run in parallel.
//! Park = commit state to OSTree. Migrate = push to remote + resume on new host.

pub mod bus;
pub mod dreamer;
pub mod multi;
pub mod scheduler;
pub mod session;

pub use bus::{BusEvent, EventBus, WrappedAgentEvent};
pub use dreamer::{Dreamer, MemoryFact};
pub use scheduler::{SessionPriority, SessionScheduler};
pub use session::{AgentSession, SessionId, SessionPhase, SessionManager};
