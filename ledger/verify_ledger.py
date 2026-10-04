#!/usr/bin/env python3
"""Check an experiment fingerprint ledger. Python 3 with just the standard library, no network.

  python3 verify_ledger.py [LEDGER.tsv]
      Checks the chain: every line points at the line before it, the line numbers run 1, 2, 3 ... without a
      gap, and the experiment numbers run E-0001, E-0002 ... without a gap. Also checks the order inside each
      experiment: a bar line comes before anything else, no amended bar after a start, no second start, and a
      verdict comes after a start and no more than once.

  python3 verify_ledger.py [LEDGER.tsv] --record record.json --salt <64 hex characters>
      Also checks one record that its owner has shared with you: sha256(salt, canonical record) must equal the
      fingerprint on one line of the ledger.

Exit code 0 means every check passed, 1 means a check failed, 2 means a usage error.
"""
import hashlib
import json
import os
import re
import sys

ZERO = "0" * 64
HEX64 = re.compile(r"^[0-9a-f]{64}$")
EXP_RE = re.compile(r"^E-([0-9]{4,})$")
BASE_KINDS = ("bar", "bar-amended", "start", "verdict")
KINDS = set(BASE_KINDS) | set(k + "-backfill" for k in BASE_KINDS) | {"start-unanchored"}


def canonical(record):
    """The exact bytes that are committed to: sorted keys, no spaces, UTF-8."""
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def commitment(salt_hex, record):
    return hashlib.sha256(bytes.fromhex(salt_hex) + canonical(record)).hexdigest()


def line_hash(line_bytes):
    return hashlib.sha256(line_bytes).hexdigest()


def parse_ledger(data):
    """Return (rows, errors). rows are dicts with seq, exp, kind, fingerprint (key "commitment"), previous-line
    fingerprint (key "prev") and the line bytes."""
    rows, errors = [], []
    if not data:
        return rows, ["the file is empty"]
    if not data.endswith(b"\n"):
        errors.append("the file does not end with a newline, so the last line may be cut short")
    if b"\r" in data:
        errors.append("the file contains carriage returns, so it was changed after it was published")
    raw = data.split(b"\n")
    if raw and raw[-1] == b"":
        raw = raw[:-1]
    prev_expected = ZERO
    seen = set()
    started = set()
    judged = set()
    next_exp = 1
    for i, lb in enumerate(raw, 1):
        try:
            text = lb.decode("utf-8")
        except UnicodeDecodeError:
            errors.append("line %d: not valid text" % i)
            prev_expected = line_hash(lb)
            continue
        f = text.split("\t")
        if len(f) != 5:
            errors.append("line %d: expected 5 tab-separated fields, found %d" % (i, len(f)))
            prev_expected = line_hash(lb)
            continue
        seq, exp, kind, com, prev = f
        row = {"n": i, "seq": seq, "exp": exp, "kind": kind, "commitment": com, "prev": prev, "bytes": lb}
        rows.append(row)
        if seq != str(i):
            errors.append("line %d: line number says %s, expected %d. A line was removed, added or moved "
                          "here." % (i, seq, i))
        m = EXP_RE.match(exp)
        if not m:
            errors.append("line %d: experiment number %r is not of the form E-0001" % (i, exp))
        else:
            n = int(m.group(1))
            if exp not in seen:
                if n != next_exp:
                    errors.append("line %d: experiment %s appears but E-%04d was expected next. An experiment "
                                  "is missing or out of order." % (i, exp, next_exp))
                seen.add(exp)
                next_exp = n + 1
                if kind not in ("bar", "bar-backfill"):
                    errors.append("line %d: the opening line of %s must be a bar line, found kind %r"
                                  % (i, exp, kind))
        if kind not in KINDS:
            errors.append("line %d: unknown kind %r" % (i, kind))
        else:
            base = kind[:-9] if kind.endswith("-backfill") else kind
            if base == "bar-amended" and exp in started:
                errors.append("line %d: %s has a pass mark amended after its run began" % (i, exp))
            elif base in ("start", "start-unanchored"):
                if exp in started:
                    errors.append("line %d: %s has a second start line" % (i, exp))
                started.add(exp)
            elif base == "verdict":
                if exp not in started:
                    errors.append("line %d: %s has a verdict line with no start line before it" % (i, exp))
                if exp in judged:
                    errors.append("line %d: %s has a second verdict line" % (i, exp))
                judged.add(exp)
        if not HEX64.match(com):
            errors.append("line %d: the fingerprint is not 64 lowercase hex characters" % i)
        if not HEX64.match(prev):
            errors.append("line %d: the previous-line fingerprint is not 64 lowercase hex characters" % i)
        elif prev != prev_expected:
            if i == 1:
                errors.append("line 1: the previous-line fingerprint must be 64 zeros")
            else:
                errors.append("line %d: the previous-line fingerprint does not match line %d as it stands now. Line %d was changed, or "
                              "a line was removed or moved before this point." % (i, i - 1, i - 1))
        prev_expected = line_hash(lb)
    return rows, errors


def summary(rows):
    exps = []
    verdicts = set()
    for r in rows:
        if r["exp"] not in exps:
            exps.append(r["exp"])
        if r["kind"] in ("verdict", "verdict-backfill"):
            verdicts.add(r["exp"])
    head = line_hash(rows[-1]["bytes"]) if rows else ZERO
    nb = sum(1 for r in rows if r["kind"].endswith("-backfill"))
    return {"lines": len(rows), "experiments": len(exps), "with_verdict": len(verdicts), "backfill": nb, "head": head}


def check_record(rows, record, salt_hex):
    """Return (matching rows, error or None)."""
    if not re.match(r"^[0-9a-fA-F]{64}$", salt_hex or ""):
        return [], "the salt must be 64 hex characters (32 bytes)"
    c = commitment(salt_hex.lower(), record)
    return [r for r in rows if r["commitment"] == c], None


def main(argv):
    args = list(argv)
    rec_path = salt = None
    path = None
    while args:
        a = args.pop(0)
        if a == "--record" and args:
            rec_path = args.pop(0)
        elif a == "--salt" and args:
            salt = args.pop(0)
        elif a.startswith("-"):
            print("usage: verify_ledger.py [LEDGER.tsv] [--record record.json --salt HEX]", file=sys.stderr)
            return 2
        elif path is None:
            path = a
        else:
            print("usage: verify_ledger.py [LEDGER.tsv] [--record record.json --salt HEX]", file=sys.stderr)
            return 2
    if (rec_path is None) != (salt is None):
        print("--record and --salt go together", file=sys.stderr)
        return 2
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LEDGER.tsv")
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError as e:
        print("FAIL: cannot read %s (%s)" % (path, e))
        return 1
    rows, errors = parse_ledger(data)
    if errors:
        for e in errors:
            print("FAIL: " + e)
        print("The chain check failed with %d problem(s). This file is not the ledger as it was published." %
              len(errors))
        return 1
    s = summary(rows)
    print("OK: chain intact. %d lines, %d experiments, %d with a verdict line." %
          (s["lines"], s["experiments"], s["with_verdict"]))
    print("Head hash (compare with a copy you saved earlier): %s" % s["head"])
    if rec_path is None:
        return 0
    try:
        with open(rec_path, "rb") as fh:
            record = json.loads(fh.read().decode("utf-8"))
    except (OSError, ValueError) as e:
        print("FAIL: cannot read the record file (%s)" % e)
        return 1
    hits, err = check_record(rows, record, salt)
    if err:
        print("FAIL: " + err)
        return 1
    if not hits:
        print("FAIL: no line in this ledger matches that record and salt. Either the salt is wrong, the record "
              "was altered, or the record is not in this ledger.")
        return 1
    for r in hits:
        print("OK: fingerprint matches line %s, experiment %s, kind %s." % (r["seq"], r["exp"], r["kind"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
