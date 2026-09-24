#!/usr/bin/env python3
"""evidence_pack.py — build a CIS-2 evidence pack for one receipt.

Takes a receipt (file or directory containing one) plus a verifier
output (a log file produced by running the project's verifier — e.g.
alice-aegis's cis-verify, or cis2-spec's verify3/cis2_verify3_int8 — or
a command to invoke that verifier), and produces:

  <out>/evidence-pack.json   — machine-readable pack, validated against
                                tools/evidence_pack.schema.json
  <out>/evidence-pack.{pdf,html,md} — human-readable rendering, honest
                                about which format it fell back to

No timing/latency/performance numbers are ever written into the pack,
by hard policy (this project does not publish timing numbers from
certain hosts; the safe rule applied here is "never include timing in
an evidence pack, period").

Nothing in this script fabricates a digest, hash, or verdict: any field
it cannot honestly determine is left absent (schema treats most non-core
fields as optional) or is explicitly flagged as a fallback/gap.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "cis-evidence-pack/0.2"

SCOPE_DISCLAIMER = (
    "This evidence pack documents ONE CIS-2 verifiable-inference receipt: "
    "that a specific, pinned set of model weights, config, and tokenizer, "
    "run through a specific reference verifier, reproduces a specific "
    "witness/decode digest bit-for-bit. It is Processing Integrity evidence "
    "for the inference computation step only. It contains no timing/latency "
    "numbers, no access-control, availability, or confidentiality evidence. "
    "An auditor should independently re-run the referenced verifier against "
    "the referenced inputs and confirm the digest match themselves; this "
    "pack is a transport format for what to feed the verifier, not a "
    "substitute for that re-verification."
)

VERDICT_PATTERNS = [
    re.compile(r"CIS2_VERIFY3 conformance=(PASS|FAIL)"),
    re.compile(r"\b(VERIFY PASS|VERIFY FAIL)\b"),
    re.compile(r"\b(MATCH|MISMATCH)\b"),
    re.compile(r"conformance=(PASS|FAIL)"),
    re.compile(r"determinism_in_process=(PASS|FAIL)"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def find_receipt_file(receipt_arg: Path) -> Path:
    if receipt_arg.is_file():
        return receipt_arg
    if receipt_arg.is_dir():
        candidates = sorted(receipt_arg.glob("*.receipt"))
        if not candidates:
            candidates = sorted(receipt_arg.rglob("*.receipt"))
        if not candidates:
            raise SystemExit(f"no *.receipt file found under {receipt_arg}")
        return candidates[0]
    raise SystemExit(f"receipt path does not exist: {receipt_arg}")


def parse_receipt_fields(receipt_path: Path) -> dict:
    """Best-effort key/value parse of a receipt file. Generic: splits
    'key value...' lines. Does not assume a specific receipt format
    beyond that, so it works for AEGIS-WITNESS v1-CIS style receipts and
    degrades gracefully (empty fields dict) for anything else, rather
    than fabricating structure."""
    fields = {}
    try:
        text = receipt_path.read_text(errors="replace")
    except Exception:
        return fields
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # style A: "key value..." (e.g. AEGIS-WITNESS receipts)
        if " " in line:
            key, _, rest = line.partition(" ")
            if re.match(r"^[A-Za-z0-9_-]{2,32}$", key):
                fields[key] = rest.strip()
        # style B: inline "key=value" tokens (e.g. CIS2_VERIFY3_INT8 stdout
        # logs used as a receipt when no formal *.receipt file exists).
        # Timing/latency/seconds/tok_per_s fields are deliberately dropped —
        # this pack never carries timing numbers, by policy.
        for m in re.finditer(r"\b([A-Za-z0-9_]{2,40})=([^\s]+)", line):
            k, v = m.group(1), m.group(2)
            if re.search(r"tim(e|ing)|second|tok_per_s|_s$|wall|latency|duration", k, re.IGNORECASE):
                continue
            if k not in fields:
                fields[k] = v
    return fields


def find_canonical_weights_script(search_roots) -> Path | None:
    for root in search_roots:
        if root is None:
            continue
        cand = Path(root) / "tools" / "canonical_weights_id.py"
        if cand.is_file():
            return cand
    return None


def compute_canonical_weights_id(args, receipt_fields):
    search_roots = [
        Path(__file__).resolve().parent.parent,  # this repo (cis2-spec)
        args.alt_repo,
    ]
    script = find_canonical_weights_script(search_roots)
    if script is not None and args.weights_file:
        try:
            out = subprocess.run(
                [sys.executable, str(script), str(args.weights_file)],
                capture_output=True, text=True, timeout=120, check=True,
            )
            return {
                "value": out.stdout.strip(),
                "method": "tools/canonical_weights_id.py",
                "source": f"{script} {args.weights_file}",
            }
        except Exception as e:
            print(f"WARNING: canonical_weights_id.py failed ({e}); "
                  f"falling back", file=sys.stderr)
    if args.weights_file:
        wf = Path(args.weights_file)
        return {
            "value": sha256_file(wf),
            "method": "direct-sha256-of-weights-file",
            "source": str(wf),
        }
    # fall back to whatever digest the receipt itself embeds
    for key in ("model", "weights_sha256", "weights"):
        if key in receipt_fields:
            return {
                "value": receipt_fields[key],
                "method": "receipt-embedded-digest-not-independently-recomputed",
                "source": f"receipt field '{key}'",
            }
    return {
        "value": "UNAVAILABLE",
        "method": "receipt-embedded-digest-not-independently-recomputed",
        "source": "no weights file provided and no recognizable weights digest field in receipt",
    }


def get_verifier_info(args):
    verifier_path = Path(args.verifier_bin)
    if not verifier_path.is_file():
        raise SystemExit(f"--verifier-bin does not exist: {verifier_path}")
    verifier_sha = sha256_file(verifier_path)

    version_string = None
    if args.verify_output:
        output_text = Path(args.verify_output).read_text(errors="replace")
        output_source = "provided-log-file"
        invocation = args.verify_invocation or f"(pre-captured log: {args.verify_output})"
    elif args.verify_cmd:
        proc = subprocess.run(args.verify_cmd, shell=True, capture_output=True, text=True, timeout=600)
        output_text = proc.stdout + "\n" + proc.stderr
        output_source = "invoked-by-evidence-pack"
        invocation = args.verify_cmd
    else:
        raise SystemExit("must supply --verify-output or --verify-cmd")

    verdict = "UNKNOWN"
    excerpt = None
    for pat in VERDICT_PATTERNS:
        m = pat.search(output_text)
        if m:
            verdict = m.group(1)
            start = max(0, m.start() - 40)
            end = min(len(output_text), m.end() + 40)
            excerpt = output_text[start:end].strip()
            break

    return {
        "path": str(verifier_path),
        "sha256": verifier_sha,
        "version_string": version_string,
        "invocation": invocation,
        "output_source": output_source,
    }, verdict, excerpt


def get_host_fingerprint():
    isa = platform.machine()
    plat = platform.platform()
    cpu_model = None
    flags = []
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
        m = re.search(r"^model name\s*:\s*(.+)$", cpuinfo, re.MULTILINE)
        if m:
            cpu_model = m.group(1).strip()
        m = re.search(r"^flags\s*:\s*(.+)$", cpuinfo, re.MULTILINE)
        if m:
            flags = m.group(1).split()
    except Exception:
        pass

    pinned_env = {
        "RUSTFLAGS": os.environ.get("RUSTFLAGS"),
        "taskset_available": shutil.which("taskset") is not None,
        "nproc": os.cpu_count(),
        "in_container_cgroup_hint": Path("/.dockerenv").exists() or Path("/run/.containerenv").exists(),
    }

    return {
        "isa": isa,
        "platform": plat,
        "cpu_model": cpu_model,
        "cpu_flags": flags,
        "pinned_env": pinned_env,
    }


def load_ack_block(ack_path: Path) -> str:
    if not ack_path.is_file():
        raise SystemExit(f"--ack-file not found: {ack_path}")
    text = ack_path.read_text()
    # extract from "## Acknowledgments" heading to end of file, verbatim
    idx = text.find("## Acknowledgments")
    if idx == -1:
        return text  # whole file, verbatim, as instructed
    return text[idx:].rstrip("\n") + "\n"


# ---- minimal schema validator (no third-party jsonschema available) ----

def validate_against_schema(instance: dict, schema: dict, path="$"):
    errors = []
    t = schema.get("type")
    if t == "object":
        if not isinstance(instance, dict):
            return [f"{path}: expected object"]
        for req in schema.get("required", []):
            if req not in instance:
                errors.append(f"{path}: missing required property '{req}'")
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for k in instance:
                if k not in props:
                    errors.append(f"{path}: unexpected property '{k}'")
        for k, v in instance.items():
            if k in props:
                errors.extend(validate_against_schema(v, props[k], f"{path}.{k}"))
    elif t == "array":
        if not isinstance(instance, list):
            return [f"{path}: expected array"]
        item_schema = schema.get("items")
        if item_schema:
            for i, item in enumerate(instance):
                errors.extend(validate_against_schema(item, item_schema, f"{path}[{i}]"))
    elif t == "string":
        if not isinstance(instance, str):
            return [f"{path}: expected string"]
        if "const" in schema and instance != schema["const"]:
            errors.append(f"{path}: expected const '{schema['const']}', got '{instance}'")
        if "enum" in schema and instance not in schema["enum"]:
            errors.append(f"{path}: '{instance}' not in enum {schema['enum']}")
        if "pattern" in schema and not re.match(schema["pattern"], instance):
            errors.append(f"{path}: '{instance}' does not match pattern {schema['pattern']}")
    elif isinstance(t, list):
        # union type e.g. ["string", "null"]
        ok = False
        for sub_t in t:
            sub_schema = dict(schema)
            sub_schema["type"] = sub_t
            if not validate_against_schema(instance, sub_schema, path):
                ok = True
                break
        if not ok and instance is not None:
            errors.append(f"{path}: value does not match any of {t}")
        elif not ok and "null" not in t:
            errors.append(f"{path}: null not permitted")
    return errors


def render_pack(pack: dict, out_dir: Path) -> dict:
    """Try reportlab -> try chromium (html->pdf) -> markdown fallback.
    Always returns which format was actually used and writes it honestly;
    never silently mislabels a fallback as the primary format."""
    md_text = render_markdown(pack)

    # 1) reportlab
    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import inch

        pdf_path = out_dir / "evidence-pack.pdf"
        c = canvas.Canvas(str(pdf_path), pagesize=LETTER)
        text_obj = c.beginText(0.75 * inch, 10.5 * inch)
        text_obj.setFont("Helvetica", 9)
        for line in md_text.splitlines():
            text_obj.textLine(line[:110])
        c.drawText(text_obj)
        c.save()
        return {"format": "pdf-reportlab", "fallback_disclosed": False, "path": str(pdf_path)}
    except ImportError:
        pass

    # 2) chromium headless html->pdf
    chromium_bin = None
    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        p = shutil.which(name)
        if p:
            chromium_bin = p
            break
    if chromium_bin:
        html_path = out_dir / "evidence-pack.html"
        html_path.write_text(render_html(pack))
        pdf_path = out_dir / "evidence-pack.pdf"
        try:
            subprocess.run(
                [chromium_bin, "--headless", "--disable-gpu",
                 f"--print-to-pdf={pdf_path}", str(html_path)],
                capture_output=True, timeout=60, check=True,
            )
            if pdf_path.is_file():
                return {"format": "pdf-chromium", "fallback_disclosed": False, "path": str(pdf_path)}
        except Exception as e:
            print(f"WARNING: chromium PDF render failed ({e})", file=sys.stderr)

    # 3) markdown fallback — explicit, not silent
    md_path = out_dir / "evidence-pack.md"
    banner = (
        "> **RENDER FALLBACK NOTICE**: neither `reportlab` nor a `chromium`/"
        "`google-chrome` binary was available on this host, so this evidence "
        "pack was rendered as Markdown instead of PDF. This is disclosed "
        "here explicitly rather than silently substituting formats.\n\n"
    )
    md_path.write_text(banner + md_text)
    print(
        "WARNING: PDF rendering unavailable (no reportlab, no chromium); "
        "wrote evidence-pack.md fallback instead.",
        file=sys.stderr,
    )
    return {"format": "markdown-fallback", "fallback_disclosed": True, "path": str(md_path)}


def render_markdown(pack: dict) -> str:
    lines = []
    lines.append("# CIS-2 Evidence Pack")
    lines.append("")
    lines.append(f"schema_version: `{pack['schema_version']}`  ")
    lines.append(f"generated_at_utc: `{pack['generated_at_utc']}`  ")
    lines.append(f"verdict: **{pack['verdict']}**")
    lines.append("")
    lines.append("## Scope")
    lines.append(pack["scope_disclaimer"])
    lines.append("")
    lines.append("## Receipt")
    lines.append(f"- source_path: `{pack['receipt']['source_path']}`")
    lines.append(f"- receipt_sha256: `{pack['receipt']['receipt_sha256']}`")
    for k, v in pack["receipt"]["fields"].items():
        lines.append(f"  - {k}: `{v}`")
    lines.append("")
    lines.append("## Canonical weights ID")
    cw = pack["canonical_weights_id"]
    lines.append(f"- value: `{cw['value']}`")
    lines.append(f"- method: `{cw['method']}`")
    lines.append(f"- source: {cw['source']}")
    lines.append("")
    lines.append("## Verifier")
    v = pack["verifier"]
    lines.append(f"- path: `{v['path']}`")
    lines.append(f"- sha256: `{v['sha256']}`")
    lines.append(f"- invocation: `{v['invocation']}`")
    lines.append(f"- output_source: {v['output_source']}")
    if pack.get("verdict_raw_excerpt"):
        lines.append(f"- verdict excerpt: `{pack['verdict_raw_excerpt']}`")
    lines.append("")
    lines.append("## Host fingerprint")
    hf = pack["host_fingerprint"]
    lines.append(f"- isa: `{hf['isa']}`")
    lines.append(f"- platform: `{hf['platform']}`")
    lines.append(f"- cpu_model: `{hf.get('cpu_model')}`")
    lines.append(f"- cpu_flags: {' '.join(hf.get('cpu_flags', []))}")
    lines.append(f"- pinned_env: `{json.dumps(hf['pinned_env'])}`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(pack["acknowledgments"])
    return "\n".join(lines)


def render_html(pack: dict) -> str:
    import html as html_mod
    body = render_markdown(pack)
    return "<html><body><pre>" + html_mod.escape(body) + "</pre></body></html>"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", required=True, type=Path, help="receipt file or directory containing one *.receipt file")
    ap.add_argument("--verifier-bin", required=True, help="path to the verifier binary/script actually used (its sha256 is pinned into the pack)")
    ap.add_argument("--verify-output", type=Path, default=None, help="path to a pre-captured verifier stdout/stderr log")
    ap.add_argument("--verify-cmd", default=None, help="shell command to invoke the verifier (alternative to --verify-output)")
    ap.add_argument("--verify-invocation", default=None, help="human-readable description of how --verify-output was produced")
    ap.add_argument("--weights-file", default=None, help="path to the model weights file, for canonical-weights-id computation")
    ap.add_argument("--alt-repo", default=None, help="additional repo root to search for tools/canonical_weights_id.py")
    ap.add_argument("--ack-file", required=True, type=Path, help="path to ACKNOWLEDGMENTS.md (block appended verbatim)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--schema", default=None, type=Path, help="path to evidence_pack.schema.json (default: alongside this script)")
    args = ap.parse_args()

    schema_path = args.schema or (Path(__file__).resolve().parent / "evidence_pack.schema.json")
    schema = json.loads(schema_path.read_text())

    args.out_dir.mkdir(parents=True, exist_ok=True)

    receipt_path = find_receipt_file(args.receipt)
    receipt_fields = parse_receipt_fields(receipt_path)
    receipt_sha256 = sha256_file(receipt_path)

    canonical_weights_id = compute_canonical_weights_id(args, receipt_fields)
    verifier_info, verdict, excerpt = get_verifier_info(args)
    host_fp = get_host_fingerprint()
    ack_text = load_ack_block(args.ack_file)

    pack = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "receipt": {
            "source_path": str(receipt_path),
            "receipt_sha256": receipt_sha256,
            "fields": receipt_fields,
            "receipt_timestamp_utc": None,
        },
        "canonical_weights_id": canonical_weights_id,
        "verifier": verifier_info,
        "host_fingerprint": host_fp,
        "verdict": verdict,
        "verdict_raw_excerpt": excerpt,
        "scope_disclaimer": SCOPE_DISCLAIMER,
        "acknowledgments": ack_text,
    }

    errors = validate_against_schema(pack, schema)
    if errors:
        print("SCHEMA VALIDATION FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        raise SystemExit(1)

    json_path = args.out_dir / "evidence-pack.json"
    json_path.write_text(json.dumps(pack, indent=2, sort_keys=False) + "\n")

    render_info = render_pack(pack, args.out_dir)
    pack["render"] = {"format": render_info["format"], "fallback_disclosed": render_info["fallback_disclosed"]}
    # rewrite json now that render info is known, re-validate
    errors = validate_against_schema(pack, schema)
    if errors:
        print("SCHEMA VALIDATION FAILED (post-render):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        raise SystemExit(1)
    json_path.write_text(json.dumps(pack, indent=2, sort_keys=False) + "\n")

    json_sha = sha256_file(json_path)
    render_sha = sha256_file(Path(render_info["path"]))

    print(f"schema: OK ({schema_path})")
    print(f"verdict: {verdict}")
    print(f"render_format: {render_info['format']}")
    print(f"evidence-pack.json sha256: {json_sha}  ({json_path})")
    print(f"{Path(render_info['path']).name} sha256: {render_sha}  ({render_info['path']})")


if __name__ == "__main__":
    main()
