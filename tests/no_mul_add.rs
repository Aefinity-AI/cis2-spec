//! CIS-2 §3.2 mechanical gate: fail the build if any `.mul_add(` call
//! (which lowers to LLVM's `llvm.fma` intrinsic — see src/math.rs module
//! docs) appears anywhere in src/. This is a grep gate, not just a prose
//! rule, per the memo's "under-specified constraints are where
//! implementations diverge" lesson.

use std::fs;
use std::path::Path;

#[test]
fn no_mul_add_calls_in_src() {
    let src_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut offenders = Vec::new();
    visit(&src_dir, &mut offenders);
    assert!(
        offenders.is_empty(),
        "found `.mul_add(` calls (forbidden by CIS-2 §3.2): {offenders:?}"
    );
}

fn visit(dir: &Path, offenders: &mut Vec<String>) {
    for entry in fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_dir() {
            visit(&path, offenders);
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let text = fs::read_to_string(&path).unwrap();
        for (lineno, line) in text.lines().enumerate() {
            // Skip comment-only lines (module docs discuss the forbidden
            // pattern by name; only flag actual call sites `.mul_add(`).
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") {
                continue;
            }
            if line.contains(".mul_add(") {
                offenders.push(format!(
                    "{}:{}: {}",
                    path.display(),
                    lineno + 1,
                    line.trim()
                ));
            }
        }
    }
}
