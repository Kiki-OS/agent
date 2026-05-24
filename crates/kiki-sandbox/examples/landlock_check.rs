//! Standalone Landlock enforcement validator — run this ON a real Linux kernel
//! (>= 5.13, e.g. the Fedora/Debian UTM VM) to verify `enforce()` actually
//! restricts filesystem access, not just that the policy computes correctly.
//!
//! Usage (inside the Linux VM, from the kiki-agent checkout):
//!   cargo run -p kiki-sandbox --example landlock_check
//!
//! It creates a temp dir with an "allowed" subtree and a "forbidden" file,
//! enforces a ruleset granting only the allowed subtree, then checks that:
//!   - reading a file under the allowed subtree SUCCEEDS, and
//!   - reading the forbidden file is DENIED (EACCES).
//! Exits 0 on success (enforcement works), non-zero otherwise.

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("landlock_check: Linux-only (Landlock LSM). No-op on this host.");
}

#[cfg(target_os = "linux")]
fn main() {
    use kiki_sandbox::landlock::{enforce, FsAccess, LandlockRule};
    use std::io::Write;

    let base = std::env::temp_dir().join(format!("kiki-landlock-check-{}", std::process::id()));
    let allowed = base.join("allowed");
    std::fs::create_dir_all(&allowed).expect("mkdir allowed");
    let allowed_file = allowed.join("ok.txt");
    std::fs::File::create(&allowed_file)
        .and_then(|mut f| f.write_all(b"hello"))
        .expect("write allowed file");
    let forbidden_file = base.join("secret.txt");
    std::fs::File::create(&forbidden_file)
        .and_then(|mut f| f.write_all(b"top secret"))
        .expect("write forbidden file");

    // Grant only the allowed subtree (RW). Everything else is denied once enforced.
    let rules = vec![LandlockRule { path: allowed.to_string_lossy().into_owned(), access: FsAccess::RW }];

    match enforce(&rules) {
        Ok(true) => eprintln!("landlock: ruleset enforced"),
        Ok(false) => {
            eprintln!("landlock: NOT enforced (kernel lacks Landlock?) — cannot validate here");
            std::process::exit(2);
        }
        Err(e) => {
            eprintln!("landlock: enforce failed: {e}");
            std::process::exit(3);
        }
    }

    let allowed_read = std::fs::read(&allowed_file).is_ok();
    let forbidden_read = std::fs::read(&forbidden_file);

    println!("allowed read  : {}", if allowed_read { "OK" } else { "DENIED" });
    println!(
        "forbidden read: {}",
        match &forbidden_read {
            Ok(_) => "OK (LEAK!)".to_string(),
            Err(e) => format!("DENIED ({})", e.kind()),
        }
    );

    if allowed_read && forbidden_read.is_err() {
        println!("PASS: Landlock enforcement is working");
        std::process::exit(0);
    } else {
        println!("FAIL: enforcement did not behave as expected");
        std::process::exit(1);
    }
}
