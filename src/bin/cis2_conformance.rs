//! `cis2-conformance <your-binary>` — E21 third-party conformance CLI.
//!
//! Runs every `tests/conformance/vectors/*.txt` / `*.expected` pair
//! against an arbitrary candidate binary via the stdin/stdout protocol
//! documented in `tests/conformance/PROTOCOL.md` (that document is the
//! normative contract; this file is the reference checker, not the other
//! way around). No knowledge of this repo's `src/math.rs` is required by
//! the candidate binary.
//!
//! Usage: `cis2-conformance /path/to/your-binary`

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const TIMEOUT: Duration = Duration::from_secs(10);

fn parse_field<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix(&format!("{key}=")) {
            return Some(rest.trim());
        }
    }
    None
}

/// Run `binary op` with `input` on stdin, collecting stdout with a
/// wall-clock timeout. Returns `Err(reason)` on spawn failure, nonzero
/// exit, or timeout.
fn run_candidate(binary: &Path, op: &str, input: &str) -> Result<String, String> {
    let mut child = Command::new(binary)
        .arg(op)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to spawn {}: {e}", binary.display()))?;

    {
        let stdin = child.stdin.as_mut().expect("piped stdin");
        stdin
            .write_all(input.as_bytes())
            .map_err(|e| format!("failed to write stdin: {e}"))?;
    }
    // Close stdin (EOF) by dropping the handle before waiting, so a
    // well-behaved candidate that reads to EOF can proceed.
    drop(child.stdin.take());

    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait().map_err(|e| format!("wait failed: {e}"))? {
            let mut stdout = String::new();
            child
                .stdout
                .take()
                .expect("piped stdout")
                .read_to_string(&mut stdout)
                .map_err(|e| format!("failed to read stdout: {e}"))?;
            if !status.success() {
                return Err(format!("candidate exited with {status}"));
            }
            return Ok(stdout);
        }
        if start.elapsed() > TIMEOUT {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!("timed out after {TIMEOUT:?}"));
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn check_vector(binary: &Path, txt_path: &Path) -> (String, bool, String) {
    let name = txt_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("<unknown>")
        .to_string();
    let expected_path = txt_path.with_extension("expected");

    let vector_text = match std::fs::read_to_string(txt_path) {
        Ok(t) => t,
        Err(e) => return (name, false, format!("failed to read {}: {e}", txt_path.display())),
    };
    let expected_text = match std::fs::read_to_string(&expected_path) {
        Ok(t) => t,
        Err(e) => return (name, false, format!("failed to read {}: {e}", expected_path.display())),
    };

    let op = match parse_field(&vector_text, "op") {
        Some(op) => op.to_string(),
        None => return (name, false, "vector file has no op= field".to_string()),
    };
    let expected_digest = match parse_field(&expected_text, "out_sha256") {
        Some(d) => d.to_ascii_lowercase(),
        None => return (name, false, "expected file has no out_sha256= field".to_string()),
    };

    let stdout = match run_candidate(binary, &op, &vector_text) {
        Ok(out) => out,
        Err(reason) => return (name, false, reason),
    };

    let got_digest = match parse_field(&stdout, "out_sha256") {
        Some(d) => d.to_ascii_lowercase(),
        None => {
            return (
                name,
                false,
                format!("candidate stdout had no out_sha256= line; got: {stdout:?}"),
            )
        }
    };

    if got_digest == expected_digest {
        (name, true, "ok".to_string())
    } else {
        (
            name,
            false,
            format!("digest mismatch: got {got_digest}, want {expected_digest}"),
        )
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: cis2-conformance <path-to-your-binary>");
        eprintln!("see tests/conformance/PROTOCOL.md for the stdin/stdout contract your-binary must implement");
        std::process::exit(2);
    }
    let binary = PathBuf::from(&args[1]);

    let vectors_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/conformance/vectors");
    let mut txt_files: Vec<PathBuf> = std::fs::read_dir(&vectors_dir)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", vectors_dir.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("txt"))
        .collect();
    txt_files.sort();

    if txt_files.is_empty() {
        eprintln!("no vector files found under {}", vectors_dir.display());
        std::process::exit(2);
    }

    let mut all_pass = true;
    for txt_path in &txt_files {
        let (name, pass, detail) = check_vector(&binary, txt_path);
        if pass {
            println!("PASS  {name}");
        } else {
            println!("FAIL  {name}  ({detail})");
            all_pass = false;
        }
    }

    if all_pass {
        println!("\nall {} vector(s) passed", txt_files.len());
        std::process::exit(0);
    } else {
        println!("\nsome vectors FAILED");
        std::process::exit(1);
    }
}
