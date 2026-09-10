//! The `cis2-verify` command.
//!
//!   cis2-verify selftest
//!       Pin the floating-point environment and run the checks that need no
//!       artifacts: spec 6.6's table digest and spec 5.1's cancellation
//!       regression.
//!
//!   cis2-verify run <artifact-dir> [--prompt TEXT] [--gen-toks N] [-o FILE]
//!       Run the decode and print (or write) a receipt.
//!
//!   cis2-verify verify <artifact-dir> <receipt-file>
//!       Re-derive the run from the artifacts and check every field of the
//!       receipt against it.
//!
//!   cis2-verify check <receipt-file> [--config FILE] [--tokenizer FILE]
//!       Audit the receipt without the weights: canonical form, and every
//!       field recomputable from nothing, from config.json, or from the
//!       values spec 13.1 pins. Reports what it could not establish.
//!
//! The exit status is 0 only when everything asked for passed.

use cis2_verify::{check, fpenv, hex, mathpin, receipt::Receipt, spec, verify};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(|s| s.as_str()).unwrap_or("");
    let rest = &args[args.len().min(1)..];

    // Spec 1.2: pin the FP environment before any floating-point work, and
    // fail loudly rather than silently producing different bits.
    if let Err(e) = fpenv::pin_and_selftest() {
        eprintln!("FAIL fp-env: {e:?} ({})", fpenv::control_name());
        return ExitCode::FAILURE;
    }
    println!("CIS2-VERIFY fp-env=pinned control={}", fpenv::control_name());

    let ok = match cmd {
        "selftest" => selftest(),
        "run" => cmd_run(rest),
        "verify" => cmd_verify(rest),
        "check" => cmd_check(rest),
        _ => {
            eprintln!(
                "usage:\n  cis2-verify selftest\n  cis2-verify run <artifact-dir> \
                 [--prompt TEXT] [--gen-toks N] [-o FILE]\n  cis2-verify verify \
                 <artifact-dir> <receipt-file>\n  cis2-verify check <receipt-file> \
                 [--config FILE] [--tokenizer FILE]"
            );
            false
        }
    };
    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn selftest() -> bool {
    let mut ok = true;
    let table = hex::encode(&mathpin::table_digest());
    if table == spec::TABLE_DIGEST {
        println!("CIS2-VERIFY table_digest={table} PASS");
    } else {
        println!("CIS2-VERIFY table_digest={table} FAIL (spec 6.6 pins {})", spec::TABLE_DIGEST);
        ok = false;
    }
    // Spec 5.1's own regression check.
    let cancel = cis2_verify::ops::dot_seq(&[1e8, 1.0, -1e8], &[1.0, 1.0, 1.0]);
    if cancel == 0.0 {
        println!("CIS2-VERIFY spec_5_1_cancellation=PASS");
    } else {
        println!("CIS2-VERIFY spec_5_1_cancellation=FAIL got {cancel}");
        ok = false;
    }
    ok
}

struct Artifacts {
    weights: Vec<u8>,
    config: Vec<u8>,
    tokenizer: Vec<u8>,
}

fn load_artifacts(dir: &Path) -> Result<Artifacts, String> {
    let read = |name: &str| -> Result<Vec<u8>, String> {
        let p: PathBuf = dir.join(name);
        std::fs::read(&p).map_err(|e| format!("cannot read {}: {e}", p.display()))
    };
    Ok(Artifacts {
        weights: read("model.safetensors")?,
        config: read("config.json")?,
        tokenizer: read("tokenizer.json")?,
    })
}

fn cmd_run(args: &[String]) -> bool {
    let mut dir: Option<String> = None;
    let mut prompt = spec::PROMPT.to_string();
    let mut gen_toks = spec::GEN_TOKS;
    let mut out: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt" => {
                i += 1;
                match args.get(i) {
                    Some(v) => prompt = v.clone(),
                    None => return fail("--prompt needs a value"),
                }
            }
            "--gen-toks" => {
                i += 1;
                match args.get(i).and_then(|v| v.parse::<usize>().ok()) {
                    Some(v) => gen_toks = v,
                    None => return fail("--gen-toks needs a non-negative integer"),
                }
            }
            "-o" => {
                i += 1;
                match args.get(i) {
                    Some(v) => out = Some(v.clone()),
                    None => return fail("-o needs a filename"),
                }
            }
            other if dir.is_none() => dir = Some(other.to_string()),
            other => return fail(&format!("unexpected argument {other:?}")),
        }
        i += 1;
    }
    let dir = match dir {
        Some(d) => d,
        None => return fail("run needs an artifact directory"),
    };
    let art = match load_artifacts(Path::new(&dir)) {
        Ok(a) => a,
        Err(e) => return fail(&e),
    };
    let art = verify::Artifacts {
        weights: &art.weights,
        config: &art.config,
        tokenizer: &art.tokenizer,
    };
    let r = match verify::run(&art, &prompt, gen_toks) {
        Ok(r) => r,
        Err(e) => return fail(&e),
    };
    report(&r);
    let text = r.render();
    match out {
        Some(path) => {
            if let Err(e) = std::fs::write(&path, text.as_bytes()) {
                return fail(&format!("cannot write {path}: {e}"));
            }
            println!("CIS2-VERIFY receipt written to {path}");
        }
        None => print!("{text}"),
    }
    // The pinned test vector is only claimed for the pinned prompt and
    // length; on anything else the receipt still stands on its own.
    if prompt == spec::PROMPT && gen_toks == spec::GEN_TOKS {
        let bad = verify::check_against_pinned(&r);
        if bad.is_empty() {
            println!("CIS2-VERIFY conformance=PASS (spec 13.1 test vector reproduced)");
            true
        } else {
            for line in &bad {
                println!("CIS2-VERIFY conformance FAIL: {line}");
            }
            false
        }
    } else {
        println!("CIS2-VERIFY conformance=N/A (not the spec 13.1 pinned prompt/length)");
        true
    }
}

fn cmd_verify(args: &[String]) -> bool {
    if args.len() != 2 {
        return fail("verify needs an artifact directory and a receipt file");
    }
    let art = match load_artifacts(Path::new(&args[0])) {
        Ok(a) => a,
        Err(e) => return fail(&e),
    };
    let text = match std::fs::read_to_string(&args[1]) {
        Ok(t) => t,
        Err(e) => return fail(&format!("cannot read {}: {e}", args[1])),
    };
    let claimed = match Receipt::parse(&text) {
        Ok(r) => r,
        Err(e) => return fail(&e),
    };
    let art = verify::Artifacts {
        weights: &art.weights,
        config: &art.config,
        tokenizer: &art.tokenizer,
    };
    let bad = verify::verify(&art, &claimed);
    if bad.is_empty() {
        report(&claimed);
        println!("CIS2-VERIFY VERIFY PASS");
        true
    } else {
        for line in &bad {
            println!("CIS2-VERIFY VERIFY FAIL: {line}");
        }
        false
    }
}

/// The weights-free tier. Exit status is 0 when nothing failed; a skip is
/// not a failure, which is exactly why the residual is printed too.
fn cmd_check(args: &[String]) -> bool {
    let mut receipt_path: Option<String> = None;
    let mut config_path: Option<String> = None;
    let mut tokenizer_path: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--config" => {
                i += 1;
                match args.get(i) {
                    Some(v) => config_path = Some(v.clone()),
                    None => return fail("--config needs a filename"),
                }
            }
            "--tokenizer" => {
                i += 1;
                match args.get(i) {
                    Some(v) => tokenizer_path = Some(v.clone()),
                    None => return fail("--tokenizer needs a filename"),
                }
            }
            other if receipt_path.is_none() => receipt_path = Some(other.to_string()),
            other => return fail(&format!("unexpected argument {other:?}")),
        }
        i += 1;
    }
    let receipt_path = match receipt_path {
        Some(p) => p,
        None => return fail("check needs a receipt file"),
    };
    let text = match std::fs::read_to_string(&receipt_path) {
        Ok(t) => t,
        Err(e) => return fail(&format!("cannot read {receipt_path}: {e}")),
    };
    let read_opt = |p: &Option<String>| -> Result<Option<Vec<u8>>, String> {
        match p {
            None => Ok(None),
            Some(p) => std::fs::read(p)
                .map(Some)
                .map_err(|e| format!("cannot read {p}: {e}")),
        }
    };
    let config = match read_opt(&config_path) {
        Ok(v) => v,
        Err(e) => return fail(&e),
    };
    let tokenizer = match read_opt(&tokenizer_path) {
        Ok(v) => v,
        Err(e) => return fail(&e),
    };
    let ev = check::Evidence {
        config: config.as_deref(),
        tokenizer: tokenizer.as_deref(),
    };

    let checks = check::check(&text, &ev);
    for k in &checks {
        println!("CIS2-VERIFY CHECK {} {}: {}", k.status.label(), k.name, k.detail);
    }
    let residual = check::residual(&checks);
    if residual.is_empty() {
        println!("CIS2-VERIFY CHECK residual=none (every field established without the weights)");
    } else {
        println!(
            "CIS2-VERIFY CHECK residual={} field(s) not established here: {}",
            residual.len(),
            residual.join(",")
        );
        println!(
            "CIS2-VERIFY CHECK to establish them: cis2-verify verify <artifact-dir> {receipt_path}"
        );
    }
    let ok = check::passed(&checks);
    println!("CIS2-VERIFY CHECK {}", if ok { "PASS" } else { "FAIL" });
    ok
}

fn report(r: &Receipt) {
    println!(
        "CIS2-VERIFY digest={} prompt_toks={} gen_toks={} dtype={}",
        hex::encode(&r.witness_digest),
        r.prompt_token_ids.len(),
        r.gen_toks,
        r.dtype
    );
    println!("CIS2-VERIFY argmax_digest={}", hex::encode(&r.argmax_digest));
    println!("CIS2-VERIFY table_digest={}", hex::encode(&r.table_digest));
    println!(
        "CIS2-VERIFY inv_freq_table_digest={}",
        hex::encode(&r.inv_freq_table_digest)
    );
    let ids: Vec<String> = r.generated_token_ids.iter().map(|t| t.to_string()).collect();
    println!("CIS2-VERIFY generated_token_ids={}", ids.join(","));
}

fn fail(msg: &str) -> bool {
    eprintln!("CIS2-VERIFY FAIL: {msg}");
    false
}
