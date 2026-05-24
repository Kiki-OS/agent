//! Seccomp-bpf syscall filtering for native apps — the second sandbox layer
//! (after the network namespace + Landlock filesystem allowlist).
//!
//! Policy is a **denylist**: by default every syscall is allowed (apps need a
//! broad surface to run), but a fixed set of clearly-dangerous syscalls is
//! blocked with `EPERM` — kernel module loading, `ptrace`, mounting, `kexec`,
//! `bpf`, namespace/`chroot` escapes, raw kernel log/perf, cross-process memory.
//! This meaningfully shrinks the attack surface without the fragility of a tight
//! allowlist (which tends to break ordinary apps).
//!
//! Same split as [`crate::landlock`]: the denied-syscall *list* is pure and
//! testable anywhere; the BPF compilation + `apply` is Linux-only (validated on
//! a real kernel, not this dev host).

/// The dangerous syscalls an app is forbidden to make. Names are the canonical
/// Linux syscall names; the Linux backend maps them to numbers via `libc::SYS_*`.
/// Pure data so the policy is testable on any platform.
pub fn denied_syscall_names() -> &'static [&'static str] {
    &[
        "ptrace",            // debugger attach / code injection into other procs
        "mount",             // mounting filesystems
        "umount2",           // unmounting
        "pivot_root",        // root filesystem swap
        "chroot",            // filesystem root change
        "setns",             // join another namespace (sandbox escape)
        "unshare",           // create new namespaces (already done pre-filter)
        "reboot",            // halt / reboot the machine
        "kexec_load",        // load a new kernel
        "kexec_file_load",
        "init_module",       // load kernel modules
        "finit_module",
        "delete_module",
        "bpf",               // load BPF programs
        "perf_event_open",   // perf / tracing subsystem
        "swapon",            // swap management
        "swapoff",
        "acct",              // process accounting
        "syslog",            // kernel ring buffer
        "process_vm_readv",  // read another process's memory
        "process_vm_writev", // write another process's memory
    ]
}

#[derive(Debug, thiserror::Error)]
pub enum SeccompError {
    #[error("seccomp: {0}")]
    Apply(String),
}

/// Compile + install the seccomp denylist on the calling process (Linux only).
/// The filter is inherited across `execve`, so installing it in a pre-exec hook
/// sandboxes the launched app. Fail-closed: returns `Err` if the filter can't be
/// applied (the caller aborts the spawn rather than run unfiltered).
///
/// NOTE: compiled but exercised only on a real kernel (this dev host is darwin —
/// here it's a no-op). Validated via `examples/seccomp_check.rs`.
#[cfg(target_os = "linux")]
pub fn apply() -> Result<(), SeccompError> {
    use seccompiler::{BpfProgram, SeccompAction, SeccompFilter};
    use std::collections::BTreeMap;
    use std::convert::TryInto;

    // Deny-listed syscalls → matched → EPERM; everything else → Allow.
    let rules: BTreeMap<i64, Vec<seccompiler::SeccompRule>> =
        denied_syscall_numbers().into_iter().map(|nr| (nr, vec![])).collect();

    let filter = SeccompFilter::new(
        rules,
        SeccompAction::Allow,                       // mismatch (everything else)
        SeccompAction::Errno(libc::EPERM as u32),   // match (the denylist)
        std::env::consts::ARCH.try_into().map_err(|e| SeccompError::Apply(format!("{e:?}")))?,
    )
    .map_err(|e| SeccompError::Apply(format!("{e:?}")))?;

    let program: BpfProgram = filter.try_into().map_err(|e| SeccompError::Apply(format!("{e:?}")))?;
    seccompiler::apply_filter(&program).map_err(|e| SeccompError::Apply(format!("{e:?}")))
}

/// Non-Linux stub: seccomp-bpf is Linux-only. No-op so the workspace builds + the
/// policy layer can be tested on dev machines.
#[cfg(not(target_os = "linux"))]
pub fn apply() -> Result<(), SeccompError> {
    Ok(())
}

/// Map the denied names to their `libc::SYS_*` numbers on this architecture.
/// Kept adjacent to [`denied_syscall_names`] — a Linux test asserts they line up.
#[cfg(target_os = "linux")]
pub fn denied_syscall_numbers() -> Vec<i64> {
    vec![
        libc::SYS_ptrace,
        libc::SYS_mount,
        libc::SYS_umount2,
        libc::SYS_pivot_root,
        libc::SYS_chroot,
        libc::SYS_setns,
        libc::SYS_unshare,
        libc::SYS_reboot,
        libc::SYS_kexec_load,
        libc::SYS_kexec_file_load,
        libc::SYS_init_module,
        libc::SYS_finit_module,
        libc::SYS_delete_module,
        libc::SYS_bpf,
        libc::SYS_perf_event_open,
        libc::SYS_swapon,
        libc::SYS_swapoff,
        libc::SYS_acct,
        libc::SYS_syslog,
        libc::SYS_process_vm_readv,
        libc::SYS_process_vm_writev,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn denylist_covers_the_dangerous_classes() {
        let names = denied_syscall_names();
        for must in ["ptrace", "mount", "bpf", "init_module", "kexec_load", "setns", "reboot"] {
            assert!(names.contains(&must), "denylist must block {must}");
        }
    }

    #[test]
    fn denylist_has_no_duplicates() {
        let names = denied_syscall_names();
        let mut seen = std::collections::HashSet::new();
        for n in names {
            assert!(seen.insert(*n), "duplicate syscall in denylist: {n}");
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn names_and_numbers_line_up() {
        assert_eq!(
            denied_syscall_names().len(),
            denied_syscall_numbers().len(),
            "every denied name must map to a syscall number",
        );
    }
}
