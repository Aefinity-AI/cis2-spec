#!/usr/bin/env python3
"""Spec 14.6 / E29 evidence: structural diff of two layer dumps.

Both files are written by `cis2-verify --features layerdump` (see
`examples/layer_dump.rs`). Record format, little-endian, repeated to EOF:

    u32 name_len | u8 name[name_len] | u32 n | f32 data[n]

Two uses:

  1. *Cross-machine / cross-build identity.* Two dumps of the same decode
     produced by different binaries or on different CPUs must agree bit for
     bit. This reports the first record that does not, if any.

  2. *Sensitivity control.* A dump compared against one taken with a single
     perturbed weight bit shows how far that perturbation reaches: which
     tensors move, from which layer onward, and by how much. Without this
     control, an identity result says nothing -- a dump that never moved
     would also be "identical".

The comparison is on raw bits, not on values: NaN payloads and the sign of
zero are differences here, as they are for a digest.

Usage: diff_dumps.py <a.bin> <b.bin> [--max-report N]
"""
import re
import struct
import sys
from collections import OrderedDict


def load(path):
    """Return an OrderedDict name -> bytes (the raw f32 payload).

    First occurrence wins; a repeated name is the second decode of spec
    12.4's determinism check and must carry identical bytes. A repeat that
    differs is reported rather than silently dropped.
    """
    recs, dup_ok, dup_bad = OrderedDict(), 0, []
    with open(path, "rb") as f:
        blob = f.read()
    i, n = 0, len(blob)
    while i < n:
        (ln,) = struct.unpack_from("<I", blob, i)
        i += 4
        name = blob[i:i + ln].decode()
        i += ln
        (cnt,) = struct.unpack_from("<I", blob, i)
        i += 4
        payload = blob[i:i + 4 * cnt]
        i += 4 * cnt
        if name in recs:
            if recs[name] == payload:
                dup_ok += 1
            else:
                dup_bad.append(name)
        else:
            recs[name] = payload
    return recs, dup_ok, dup_bad


def kind_of(name):
    return name.split(".")[-1]


def layer_of(name):
    m = re.search(r"\.L(\d+)\.", name)
    return int(m.group(1)) if m else -1


def pos_of(name):
    m = re.match(r"p(\d+)\.", name)
    return int(m.group(1)) if m else -1


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    max_report = 20
    if "--max-report" in sys.argv:
        max_report = int(sys.argv[sys.argv.index("--max-report") + 1])

    A, a_dup, a_bad = load(a_path)
    B, b_dup, b_bad = load(b_path)

    print(f"A {a_path}")
    print(f"B {b_path}")
    print(f"records(unique)  A={len(A)} B={len(B)}")
    print(f"duplicate records identical  A={a_dup} B={b_dup}")
    if a_bad or b_bad:
        print(f"!! duplicate records that DIFFER  A={len(a_bad)} B={len(b_bad)}")

    only_a = [k for k in A if k not in B]
    only_b = [k for k in B if k not in A]
    if only_a or only_b:
        print(f"!! name mismatch: only-in-A={len(only_a)} only-in-B={len(only_b)}")
        for k in (only_a + only_b)[:max_report]:
            print(f"   {k}")

    common = [k for k in A if k in B]
    same = [k for k in common if A[k] == B[k]]
    diff = [k for k in common if A[k] != B[k]]
    shape = [k for k in common if len(A[k]) != len(B[k])]

    print(f"common records   {len(common)}")
    print(f"bit-identical    {len(same)}")
    print(f"differing        {len(diff)}")
    print(f"length mismatch  {len(shape)}")

    if not diff:
        print("VERDICT identical -- every recorded intermediate agrees bit for bit")
        return 0

    # Where does the difference start, and how far does it spread?
    order = list(A.keys())
    first = next(k for k in order if k in B and A[k] != B[k])
    print(f"first differing record (emission order)  {first}")

    def stats(k):
        u = struct.unpack(f"<{len(A[k]) // 4}f", A[k])
        v = struct.unpack(f"<{len(B[k]) // 4}f", B[k])
        m = max(abs(x - y) for x, y in zip(u, v))
        nz = sum(1 for x, y in zip(u, v) if x != y)
        return m, nz, len(u)

    by_kind, by_layer = {}, {}
    for k in common:
        kk, lk = kind_of(k), layer_of(k)
        by_kind.setdefault(kk, [0, 0])
        by_layer.setdefault(lk, [0, 0])
        by_kind[kk][1] += 1
        by_layer[lk][1] += 1
        if A[k] != B[k]:
            by_kind[kk][0] += 1
            by_layer[lk][0] += 1

    print("\nby tensor kind: differing / total")
    for kk in sorted(by_kind, key=lambda x: -by_kind[x][0]):
        d, t = by_kind[kk]
        print(f"  {kk:12s} {d:5d} / {t:5d}")

    print("\nby layer: differing / total  (-1 = not inside a layer)")
    for lk in sorted(by_layer):
        d, t = by_layer[lk]
        if d:
            print(f"  L{lk:<3d} {d:5d} / {t:5d}")

    print(f"\nworst {max_report} records by max |a-b|")
    scored = sorted(((stats(k), k) for k in diff), key=lambda s: -s[0][0])
    for (m, nz, tot), k in scored[:max_report]:
        print(f"  {k:34s} max|d|={m:.6e}  elems differing {nz}/{tot}")

    print("\nVERDICT differs")
    return 1


if __name__ == "__main__":
    sys.exit(main())
