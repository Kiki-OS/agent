//! Dev helper: parse + validate a kiki.toml against kiki-schema::ArtifactManifest.
//! Usage: cargo run -p kiki-pkg --example parse_manifest -- <path>...
use kiki_schema::ArtifactManifest;

fn main() {
    let mut bad = 0;
    for path in std::env::args().skip(1) {
        let raw = std::fs::read_to_string(&path).expect("read");
        match toml::from_str::<ArtifactManifest>(&raw) {
            Ok(m) => match m.validate() {
                Ok(()) => println!(
                    "OK   {path}  id={} kind={:?} exec={} net={:?}",
                    m.artifact.id, m.artifact.kind, m.exec.is_some(), m.capabilities.network
                ),
                Err(e) => { println!("INVALID {path}: {e}"); bad += 1; }
            },
            Err(e) => { println!("PARSE-ERR {path}: {e}"); bad += 1; }
        }
    }
    if bad > 0 { std::process::exit(1); }
}
