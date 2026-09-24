#!/usr/bin/env python3
"""Stdio MCP server exposing the CIS-2 verify3 clean-room verifier.

Two tools:
  - verify_receipt(dir): run the already-built (or auto-built) verify3
    binary against a directory containing model.safetensors, config.json,
    tokenizer.json, and compare every digest field against the pinned
    v0.3b vector in EXPECTED_DIGESTS.md.
  - selfcheck(): equivalent of scripts/self_check.sh — fetch the pinned
    SmolLM2-135M weights if missing, build verify3/, run the FMA gate,
    run cis2_verify3, and compare.

No timing numbers are ever produced or printed anywhere in this server,
matching repo policy (see scripts/self_check.sh).
"""
import re
import subprocess
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

REPO_ROOT = Path(__file__).resolve().parents[2]
VERIFY3_DIR = REPO_ROOT / "verify3"
VERIFY3_BIN = VERIFY3_DIR / "cis2_verify3"
WEIGHTS_DIR = REPO_ROOT / "weights"
FETCH_SCRIPT = REPO_ROOT / "scripts" / "fetch_weights.sh"

# Pinned normative digests for the §13.1 SmolLM2-135M vector, spec v0.3b.
# Mirrors EXPECTED_DIGESTS.md and scripts/self_check.sh exactly — do not
# invent new comparison semantics here, reuse those pinned values.
EXPECTED = {
    "weights_sha256": "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1",
    "config_sha256": "1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843",
    "tokenizer_sha256": "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
    "table_digest": "23c7bfaf5cef0095fd021af2eb1808abb4928bae4219756d86bdac670a06b35d",
    "inv_freq_table_digest": "da9f6dcfde0425588815509e874515cdcd3d6b8818b6d0136590052e7bbf6f12",
    "argmax_digest": "0b9c8f3ac90d0b9cd5f1719ac327dca1fc639fd87468305fccebbe3d56f67aff",
    "generated_token_ids": "28,665,436,253,1838,8180,3365,14176,30,2306,4161,281,253,2066,2291,351",
    "CIS2_REF": "d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df",
}

mcp = FastMCP("cis2-verify")


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_verify3() -> dict[str, Any]:
    """Build verify3/ if the binary is missing. Returns a status dict."""
    if VERIFY3_BIN.exists():
        return {"built": False, "note": "binary already present"}
    proc = subprocess.run(
        ["make", "-C", str(VERIFY3_DIR)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not VERIFY3_BIN.exists():
        raise RuntimeError(
            f"verify3 build failed (rc={proc.returncode}):\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return {"built": True, "stdout": proc.stdout, "stderr": proc.stderr}


def _fma_gate() -> dict[str, Any]:
    """Disassemble the built binary and check for FMA-family instructions
    (spec §1.4). Mirrors the objdump check in scripts/self_check.sh."""
    import shutil

    if shutil.which("objdump") is None:
        return {"ran": False, "note": "objdump not found, skipping FMA gate"}
    proc = subprocess.run(
        ["objdump", "-d", str(VERIFY3_BIN)], capture_output=True, text=True
    )
    disasm = proc.stdout
    pattern = re.compile(
        r"vfmadd|vfnmadd|vfmsub|vfnmsub|fmadd|fmsub|fmla|fmls", re.IGNORECASE
    )
    matches = pattern.findall(disasm)
    return {
        "ran": True,
        "pass": len(matches) == 0,
        "match_count": len(matches),
    }


def _parse_verify3_output(stdout: str) -> dict[str, str]:
    """Parse the CIS2_VERIFY3 stdout lines into a field->value dict.
    Mirrors the extract_field logic in scripts/self_check.sh, extended to
    every field checked against EXPECTED_DIGESTS.md."""
    fields: dict[str, str] = {}

    m = re.search(r"^CIS2_VERIFY3 selftest=(\S+)", stdout, re.MULTILINE)
    if m:
        fields["selftest"] = m.group(1)

    m = re.search(
        r"^CIS2_VERIFY3 digest=([0-9a-f]+)", stdout, re.MULTILINE
    )
    if m:
        fields["CIS2_REF"] = m.group(1)

    for name in ("argmax_digest", "table_digest", "inv_freq_table_digest"):
        m = re.search(rf"^CIS2_VERIFY3 {name}=([0-9a-f]+)", stdout, re.MULTILINE)
        if m:
            fields[name] = m.group(1)

    m = re.search(
        r"^CIS2_VERIFY3 generated_token_ids=([0-9,]+)", stdout, re.MULTILINE
    )
    if m:
        fields["generated_token_ids"] = m.group(1)

    m = re.search(r"^CIS2_VERIFY3 conformance=(\S+)", stdout, re.MULTILINE)
    if m:
        fields["conformance"] = m.group(1)

    return fields


def _compare_to_pinned(
    fields: dict[str, str],
    weights_sha256: str | None,
    config_sha256: str | None,
    tokenizer_sha256: str | None,
) -> dict[str, Any]:
    """Compare every relevant field against EXPECTED, per-field PASS/FAIL,
    plus an overall verdict. Reuses the same comparison semantics as
    scripts/self_check.sh (exact string equality per field)."""
    results: dict[str, Any] = {}
    computed = dict(fields)
    if weights_sha256 is not None:
        computed["weights_sha256"] = weights_sha256
    if config_sha256 is not None:
        computed["config_sha256"] = config_sha256
    if tokenizer_sha256 is not None:
        computed["tokenizer_sha256"] = tokenizer_sha256

    overall_pass = True
    for field, expected_val in EXPECTED.items():
        got = computed.get(field)
        ok = got == expected_val
        if not ok:
            overall_pass = False
        results[field] = {"expected": expected_val, "got": got, "pass": ok}

    return {
        "verdict": "PASS" if overall_pass else "FAIL",
        "fields": results,
        "raw_selftest": fields.get("selftest"),
        "raw_conformance": fields.get("conformance"),
    }


def _run_verify3(
    weights_path: Path, config_path: Path, tokenizer_path: Path
) -> dict[str, Any]:
    for p in (weights_path, config_path, tokenizer_path):
        if not p.exists():
            raise FileNotFoundError(f"missing input file: {p}")

    build_info = _build_verify3()
    fma_info = _fma_gate()

    proc = subprocess.run(
        [
            str(VERIFY3_BIN),
            str(weights_path),
            str(config_path),
            str(tokenizer_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(VERIFY3_DIR),
    )
    fields = _parse_verify3_output(proc.stdout)

    weights_sha256 = _sha256_file(weights_path)
    config_sha256 = _sha256_file(config_path)
    tokenizer_sha256 = _sha256_file(tokenizer_path)

    comparison = _compare_to_pinned(
        fields, weights_sha256, config_sha256, tokenizer_sha256
    )

    return {
        "build": build_info,
        "fma_gate": fma_info,
        "verify3_returncode": proc.returncode,
        "verify3_stdout": proc.stdout,
        "verify3_stderr": proc.stderr,
        **comparison,
    }


@mcp.tool()
def verify_receipt(path: str) -> dict[str, Any]:
    """Verify a CIS-2 receipt for a model directory.

    `path` must be a directory containing model.safetensors, config.json,
    and tokenizer.json. Builds verify3/cis2_verify3 if not already built,
    runs it, and compares every digest field (weights_sha256, config_sha256,
    tokenizer_sha256, table_digest, inv_freq_table_digest, argmax_digest,
    generated_token_ids, CIS2_REF) against the pinned normative §13.1
    SmolLM2-135M v0.3b vector from EXPECTED_DIGESTS.md.

    Only this exact pinned model/prompt/gen_toks combination will PASS;
    any other model directory will correctly FAIL (this tool checks
    conformance to the pinned test vector, not general model validity —
    see EXPECTED_DIGESTS.md for the vector's scope).

    Returns a dict with an overall `verdict` ("PASS"/"FAIL") and a
    per-field breakdown. No timing numbers are included, by repo policy.
    """
    d = Path(path).expanduser().resolve()
    if not d.is_dir():
        return {"error": f"not a directory: {d}"}
    result = _run_verify3(
        d / "model.safetensors", d / "config.json", d / "tokenizer.json"
    )
    return result


@mcp.tool()
def selfcheck() -> dict[str, Any]:
    """Run the full CIS-2 v0.3b self-check: fetch the pinned SmolLM2-135M
    weights if missing (into <repo>/weights/, sha256-verified), build
    verify3/, run the FMA-family instruction gate (spec §1.4), run
    cis2_verify3, and compare all digests against the pinned §13.1 vector.

    This is the MCP-tool equivalent of scripts/self_check.sh. No timing
    numbers are included anywhere in the output, by repo policy.
    """
    fetch_proc = subprocess.run(
        [str(FETCH_SCRIPT), str(WEIGHTS_DIR)], capture_output=True, text=True
    )
    if fetch_proc.returncode != 0:
        return {
            "verdict": "FAIL",
            "error": "fetch_weights.sh failed",
            "fetch_stdout": fetch_proc.stdout,
            "fetch_stderr": fetch_proc.stderr,
        }

    result = _run_verify3(
        WEIGHTS_DIR / "model.safetensors",
        WEIGHTS_DIR / "config.json",
        WEIGHTS_DIR / "tokenizer.json",
    )
    result["fetch_stdout"] = fetch_proc.stdout
    return result


if __name__ == "__main__":
    mcp.run(transport="stdio")
