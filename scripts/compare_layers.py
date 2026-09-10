#!/usr/bin/env python3
"""Spec 14.6 evidence, comparison half.

Reads the two dumps produced by `cis2-verify --features layerdump` and by
`scripts/oracle_layers.py` and reports, for every named intermediate tensor
at every position, how far the pinned verifier is from the `transformers`
fp32 oracle.

Why this closes 14.6. 13.3 compares the *output* of the stack: greedy token
ids, plus one step's full logit vector. Two errors inside the stack that
cancelled at the output would pass that check. Comparing all 393 named
intermediates per position, at every position, leaves no interior place for
such a pair to hide: any layer that computed something different from the
oracle shows up as a jump in that layer's own row, whatever the output does.

Metrics, per tensor:
  rel_l2   = ||v - o||_2 / ||o||_2        -- the headline; scale-free, stable
  max_abs  = max |v - o|
  rel_rms  = max_abs / rms(o)             -- worst element, as a fraction of
                                             the tensor's own scale

Usage: compare_layers.py <verifier.bin> <oracle.bin>
"""
import os
import struct
import sys
from collections import defaultdict

import numpy as np


def load(path):
    """name -> np.float32 array, first occurrence wins.

    The verifier runs the decode twice (spec 12.4's determinism check), so
    every tensor appears twice in its dump; the duplicate is checked to be
    bit-identical rather than discarded silently.
    """
    out, dups, dup_mismatch = {}, 0, 0
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        while f.tell() < size:
            (nl,) = struct.unpack("<I", f.read(4))
            name = f.read(nl).decode("ascii")
            (n,) = struct.unpack("<I", f.read(4))
            buf = np.frombuffer(f.read(4 * n), dtype="<f4")
            if name in out:
                dups += 1
                if not np.array_equal(out[name].view("<u4"), buf.view("<u4")):
                    dup_mismatch += 1
            else:
                out[name] = buf
    return out, dups, dup_mismatch


def main():
    vpath, opath = sys.argv[1], sys.argv[2]
    V, vd, vdm = load(vpath)
    O, od, odm = load(opath)

    print(f"E28 verifier tensors={len(V)} duplicate-records={vd} "
          f"duplicates-that-differ={vdm}")
    print(f"E28 oracle   tensors={len(O)} duplicate-records={od} "
          f"duplicates-that-differ={odm}")

    only_v = sorted(set(V) - set(O))
    only_o = sorted(set(O) - set(V))
    if only_v or only_o:
        print(f"E28 WARNING unmatched: verifier-only={len(only_v)} oracle-only={len(only_o)}")
        for n in (only_v[:5] + only_o[:5]):
            print(f"E28   unmatched {n}")

    common = sorted(set(V) & set(O))
    print(f"E28 compared tensors={len(common)}")

    rows = []
    shape_mismatch = 0
    for name in common:
        a, b = V[name], O[name]
        if a.shape != b.shape:
            shape_mismatch += 1
            continue
        a64, b64 = a.astype(np.float64), b.astype(np.float64)
        d = np.abs(a64 - b64)
        nb = np.linalg.norm(b64)
        rms = float(np.sqrt(np.mean(b64 * b64)))
        rel_l2 = float(np.linalg.norm(a64 - b64) / nb) if nb > 0 else 0.0
        max_abs = float(d.max())
        rel_rms = max_abs / rms if rms > 0 else 0.0
        rows.append((name, rel_l2, max_abs, rel_rms, int(d.argmax())))
    print(f"E28 shape-mismatches={shape_mismatch}")

    def parse(name):
        parts = name.split(".")
        pos = int(parts[0][1:])
        if parts[1].startswith("L"):
            return pos, int(parts[1][1:]), parts[2]
        return pos, -1, parts[1]

    # ---- headline
    worst = max(rows, key=lambda r: r[1])
    print(f"E28 WORST rel_l2 over all {len(rows)} tensors: {worst[1]:.3e} at {worst[0]} "
          f"(max_abs={worst[2]:.3e} rel_rms={worst[3]:.3e})")
    worst_rr = max(rows, key=lambda r: r[3])
    print(f"E28 WORST rel_rms: {worst_rr[3]:.3e} at {worst_rr[0]} "
          f"(rel_l2={worst_rr[1]:.3e} max_abs={worst_rr[2]:.3e})")

    # ---- by tensor kind
    bykind = defaultdict(list)
    for name, rl2, ma, rr, _ in rows:
        bykind[parse(name)[2]].append((rl2, rr))
    print("\nE28 --- by tensor kind (max over all layers and positions) ---")
    print(f"{'kind':<12} {'n':>6} {'max rel_l2':>12} {'max rel_rms':>12}")
    for k in sorted(bykind, key=lambda k: -max(x[0] for x in bykind[k])):
        vs = bykind[k]
        print(f"{k:<12} {len(vs):>6} {max(x[0] for x in vs):>12.3e} "
              f"{max(x[1] for x in vs):>12.3e}")

    # ---- by depth. Two questions, and they need two different columns.
    #
    # The residual stream is the wrong thing to measure relative error on:
    # a Llama residual add can cancel heavily (at layer 9 position 0 the
    # oracle has ||resid_attn|| = 486.5 and ||down_proj|| = 414.1 summing to
    # ||resid_mlp|| = 105.6), and a 4.6x cancellation inflates relative error
    # by 4.6x while nothing has been computed differently. So the column that
    # answers 14.6 is the worst relative error over the layer's OWN computed
    # tensors -- if a layer computed something the oracle did not, that is
    # where it appears, whatever the residual does.
    OWN = ("ln1", "q_proj", "k_proj", "v_proj", "attn_out", "o_proj",
           "ln2", "gate_proj", "up_proj", "mlp_act", "down_proj")
    own, resid, cancel = defaultdict(list), defaultdict(list), defaultdict(list)
    for name, rl2, ma, rr, _ in rows:
        pos, li, kind = parse(name)
        if li < 0:
            continue
        if kind in OWN:
            own[li].append(rl2)
        elif kind == "resid_mlp":
            resid[li].append(rl2)
    for li in sorted(own):
        for pos in range(19):
            ra = O.get(f"p{pos}.L{li}.resid_attn")
            dp = O.get(f"p{pos}.L{li}.down_proj")
            rm = O.get(f"p{pos}.L{li}.resid_mlp")
            if ra is None or dp is None or rm is None:
                continue
            num = np.linalg.norm(ra.astype(np.float64)) + np.linalg.norm(dp.astype(np.float64))
            den = np.linalg.norm(rm.astype(np.float64))
            if den > 0:
                cancel[li].append(num / den)

    print("\nE28 --- by depth (max over all 19 positions) ---")
    print(f"{'layer':>5} {'own tensors':>13} {'resid_mlp':>12} {'cancellation':>13}")
    print(f"{'':>5} {'worst rel_l2':>13} {'rel_l2':>12} {'x at worst':>13}")
    own_worst = []
    for li in sorted(own):
        mo, mr = max(own[li]), max(resid[li])
        mc = max(cancel[li]) if cancel[li] else float("nan")
        own_worst.append((li, mo))
        print(f"{li:>5} {mo:>13.3e} {mr:>12.3e} {mc:>13.2f}")

    lo, hi = min(x[1] for x in own_worst), max(x[1] for x in own_worst)
    print(f"E28 per-layer worst own-tensor rel_l2: min={lo:.3e} max={hi:.3e} "
          f"spread={hi / lo:.2f}x over {len(own_worst)} layers")
    jumps = [(li, own_worst[i][1] / own_worst[i - 1][1])
             for i, (li, _) in enumerate(own_worst)
             if i > 0 and own_worst[i - 1][1] > 0
             and own_worst[i][1] / own_worst[i - 1][1] > 3.0]
    print(f"E28 layers whose own-tensor error more than tripled over the "
          f"previous layer: {len(jumps)} {jumps}")

    # ---- the question 14.6 actually asks: does any layer CREATE error?
    #
    # A compensating pair of errors inside the stack requires one layer to
    # diverge from the oracle and a later one to bring the output back. Both
    # halves are visible here. A layer that diverges shows its own tensors
    # far above the error it was handed; a layer that compensates shows the
    # opposite. So compare each layer's error against its input's.
    print("\nE28 --- does a layer create error, or carry in what it was handed? ---")
    print(f"{'layer':>5} {'carried-in':>12} {'own ln1':>12} {'ratio':>7} "
          f"{'worst own':>12} {'/carried':>9}")
    amps = []
    for li in range(1, len(own)):
        ci = max(r[1] for r in rows if parse(r[0]) == (parse(r[0])[0], li - 1, "resid_mlp"))
        l1 = max(r[1] for r in rows if parse(r[0]) == (parse(r[0])[0], li, "ln1"))
        mo = max(r[1] for r in rows if parse(r[0])[1] == li and parse(r[0])[2] in OWN)
        amps.append(mo / ci if ci else float("nan"))
        print(f"{li:>5} {ci:>12.3e} {l1:>12.3e} {l1 / ci:>7.2f} {mo:>12.3e} "
              f"{mo / ci:>9.2f}")
    print(f"E28 worst amplification of the carried-in error by any layer's own "
          f"tensors: {max(amps):.2f}x")
    print("E28 (a divergent layer would show a large ratio at that layer alone; "
          "a compensating layer, a ratio far below 1)")

    # ---- worst 12 tensors overall
    print("\nE28 --- worst 12 tensors by rel_l2 ---")
    print(f"{'tensor':<24} {'rel_l2':>12} {'max_abs':>12} {'rel_rms':>12}")
    for name, rl2, ma, rr, _ in sorted(rows, key=lambda r: -r[1])[:12]:
        print(f"{name:<24} {rl2:>12.3e} {ma:>12.3e} {rr:>12.3e}")

    # ---- logits, the tensor 13.3 already checks, for continuity with it
    lg = [r for r in rows if r[0].endswith(".logits")]
    print(f"\nE28 logits: n={len(lg)} max rel_l2={max(r[1] for r in lg):.3e} "
          f"max rel_rms={max(r[3] for r in lg):.3e} "
          f"max abs={max(r[2] for r in lg):.3e}")


if __name__ == "__main__":
    main()
