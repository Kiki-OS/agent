# agent

`agentd` — the central privileged daemon for Kiki OS.

The agent is not an application running on the OS; it *is* the OS's primary userspace process. Every interaction, tool call, and decision flows through `agentd`.

---

## Architecture

The agent follows a **Perceive → Reason → Act** (PRA) loop as its core runtime:

```
┌─────────────────────────────────────────────────────┐
│                    agentd (PRA loop)                 │
│                                                      │
│  Perceive ──► Reason ──► Act ──► Observe            │
│      ▲                              │                │
│      └──────────── Context ◄────────┘                │
└─────────────────────────────────────────────────────┘
         │              │              │
      Sensors         Tools          Memory
    (input events)  (MCP / shell)   (episodic/
                                     semantic)
```

---

## Crates

| Crate | Role |
|---|---|
| `kiki-core` | Fundamental types and traits — `Agent`, `Context`, `CapabilitySet`, `ControlMode`, state backend |
| `kiki-mcp` | MCP server/client — JSON-RPC 2.0 over `/run/kiki/mcp.sock` |
| `kiki-provider` | LLM provider abstraction — local (Ollama / llama.cpp) and remote (AI Gateway) |
| `kiki-sandbox` | Process isolation — Landlock, seccomp, namespaces, cgroups |
| `kiki-state` | Agent state persistence — OSTree-backed snapshots per reasoning step |
| `kiki-orchestrator` | Session lifecycle — `SessionPhase` state machine, freeze/migrate protocol |
| `kiki-fleet` | Fleet node management — migration sender/receiver, heartbeat, sync |
| `kiki-telemetry` | Structured telemetry and tracing |
| `agentd` | Binary entry point — starts all subsystems, owns the PRA loop |

---

## Key concepts

**ControlMode** — the agent operates in one of three modes, which the compositor and all subsystems observe:

```
Active     → agent is actively processing, tools in use
Ambient    → agent is listening, low-resource footprint
Suspended  → agent is frozen (migration in progress or device sleeping)
```

**Capability gate** — every sensitive action has a named capability; the gate is deny-by-default. Apps request capabilities at install time; the user grants them.

**Memory layers** — the agent's memory is structured across five layers:

```
Sensory    → ring buffer of recent perceptions (ephemeral)
Working    → current context window
Episodic   → SQLite + vectors (durable, searchable)
Semantic   → bitemporal knowledge graph
Procedural → learned skills and schemas
Identity   → SOUL.md — the agent's voice and values
```

**agentic sessions** — each desktop session is an independent unit of work with its own context, goal, and state. Sessions run in parallel; background sessions continue executing without a UI surface.

---

## IPC

```
/run/kiki/mcp.sock    ← MCP / JSON-RPC 2.0 (tool calls from apps)
/run/kiki/a11y.sock   ← FlatBuffers accessibility tree
                         (shared memory + notification socket for sensor data)
DBus                  ← Linux ecosystem integration (MPRIS, portals, notifications)
```

---

## Building

Requires Rust 1.86+.

```sh
cargo build --release
```

---

## Related repos

| Repo | Description |
|---|---|
| [os](https://github.com/Kiki-OS/os) | OS images (embeds the `agentd` binary) |
| [sdk](https://github.com/Kiki-OS/sdk) | App developer SDK |
| [de](https://github.com/Kiki-OS/de) | The desktop environment (reads `ControlMode` from agentd; graphics optional) |

---

## License

MIT License. See [LICENSE](LICENSE).
