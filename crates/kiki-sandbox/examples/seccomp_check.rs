//! Standalone seccomp validator — run ON a real Linux kernel (e.g. the Fedora
//! UTM VM) to verify the syscall denylist actually blocks dangerous syscalls.
//!
//! Usage (inside the Linux VM, from the kiki-agent checkout):
//!   cargo run -p kiki-sandbox --example seccomp_check
//!
//! Applies the seccomp filter, then checks that an allowed syscall (`getpid`)
//! still works and a denied one (`ptrace(PTRACE_TRACEME)`, which normally
//! succeeds for an untraced process) is rejected with EPERM. Exits 0 on success.

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("seccomp_check: Linux-only (seccomp-bpf). No-op on this host.");
}

#[cfg(target_os = "linux")]
fn main() {
    use kiki_sandbox::seccomp;

    let pid_before = unsafe { libc::getpid() };

    if let Err(e) = seccomp::apply() {
        eprintln!("seccomp: apply failed: {e}");
        std::process::exit(3);
    }
    eprintln!("seccomp: filter installed");

    // Allowed syscall — must still work after the filter is installed.
    let pid_after = unsafe { libc::getpid() };

    // Denied syscall — ptrace(PTRACE_TRACEME) returns 0 normally, but the filter
    // makes the kernel return -1/EPERM without executing it.
    let r = unsafe {
        libc::syscall(libc::SYS_ptrace, libc::PTRACE_TRACEME as libc::c_long, 0, 0, 0)
    };
    let errno = std::io::Error::last_os_error().raw_os_error();

    println!("getpid before/after filter: {pid_before}/{pid_after}");
    println!("ptrace(TRACEME): ret={r} errno={errno:?}");

    if pid_after == pid_before && r == -1 && errno == Some(libc::EPERM) {
        println!("PASS: seccomp allowed getpid and denied ptrace with EPERM");
        std::process::exit(0);
    } else {
        println!("FAIL: seccomp did not behave as expected");
        std::process::exit(1);
    }
}
