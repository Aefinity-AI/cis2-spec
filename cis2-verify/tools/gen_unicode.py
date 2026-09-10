#!/usr/bin/env python3
"""Regenerate src/unicode.rs from Python's Unicode Character Database.

src/unicode.rs has said "Regenerate with tools/gen_unicode.py" since it was
written, and this script did not exist. That is a provenance claim a reviewer
could not check, in a repository whose whole argument is that its claims are
checkable. This closes it.

Usage:
    python3 tools/gen_unicode.py            # check: is src/unicode.rs current?
    python3 tools/gen_unicode.py --write    # rewrite src/unicode.rs

The check mode is the useful one: it exits non-zero if the committed file is
not byte-for-byte what this script produces, so a hand-edit or a Unicode
version drift is caught rather than assumed away.

What the three tables are, and why they are these and not something else --
spec 3.1.4.b pins the ByteLevel pre-tokenizer with use_regex = true, whose
pattern is GPT-2's:

    's|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+

so the crate needs exactly three predicates: \\p{L}, \\p{N}, and \\s.

  LETTER     = General_Category in {Lu, Ll, Lt, Lm, Lo}  -- what \\p{L} means.
  NUMBER     = General_Category in {Nd, Nl, No}          -- what \\p{N} means.
  WHITESPACE = the White_Space property.

The first two come straight from `unicodedata.category`. The third needs care
and is the one place this script cannot just ask the UCD: `unicodedata`
exposes no White_Space predicate, and `str.isspace()` is NOT the same set --
it additionally returns True for U+001C..U+001F (the C0 file/group/record/unit
separators), which do not carry White_Space. Rust's `regex` crate matches \\s
against White_Space, so those four must be excluded or the pre-tokenizer would
split on bytes the reference implementation does not split on. The exclusion is
asserted below rather than trusted, by checking the result against the 25
codepoints PropList.txt lists for White_Space.
"""
import sys
import os

# The UCD version is part of the answer, not an incidental detail: \p{L} is
# version-dependent. Unicode 15.1 added U+2EBF0..U+2EE5D (CJK Extension I, 622
# codepoints) to General_Category Lo; 15.0 does not have them. A host Python
# carries whatever UCD its build shipped with (GitHub's ubuntu-24.04 images
# carry 15.0.0; Debian trixie carries 15.1.0), so this script pins the version
# explicitly and prefers the standalone `unicodedata2` package, which ships a
# chosen UCD independent of the interpreter.
REQUIRED_UCD = "15.1.0"
try:
    import unicodedata2 as unicodedata
except ImportError:
    import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "..", "src", "unicode.rs")
MAX_CP = 0x110000

# White_Space, from PropList.txt. Written out so the assert below is a real
# comparison against the UCD's own list and not a restatement of the code.
WHITE_SPACE = (
    list(range(0x09, 0x0E)) + [0x20, 0x85, 0xA0, 0x1680]
    + list(range(0x2000, 0x200B)) + [0x2028, 0x2029, 0x202F, 0x205F, 0x3000]
)


def ranges(pred):
    """Ascending, maximal, non-adjacent ranges of codepoints satisfying pred."""
    out, start, prev = [], None, None
    for cp in range(MAX_CP):
        if pred(cp):
            if start is None:
                start = cp
            prev = cp
        elif start is not None:
            out.append((start, prev))
            start = None
    if start is not None:
        out.append((start, prev))
    return out


def cat_pred(prefix):
    def p(cp):
        return unicodedata.category(chr(cp)).startswith(prefix)
    return p


def space_pred(cp):
    # isspace() minus the four C0 separators; see the module docstring.
    return chr(cp).isspace() and cp not in (0x1C, 0x1D, 0x1E, 0x1F)


def render_table(name, rs):
    lines = [
        f"/// Unicode ranges for {name}, generated from the Unicode Character",
        "/// Database via `unicodedata` and embedded so this crate needs no",
        f"/// regex or unicode dependency. {len(rs)} ranges.",
        f"pub static {name}: [(u32, u32); {len(rs)}] = [",
    ]
    for i in range(0, len(rs), 4):
        chunk = rs[i:i + 4]
        lines.append("    " + " ".join(f"(0x{a:X},0x{b:X})," for a, b in chunk))
    lines.append("];")
    return "\n".join(lines)


def build():
    letter = ranges(cat_pred("L"))
    number = ranges(cat_pred("N"))
    space = ranges(space_pred)

    flat = [cp for a, b in space for cp in range(a, b + 1)]
    assert flat == WHITE_SPACE, (
        "the whitespace set does not match PropList.txt's White_Space; "
        f"got {len(flat)} codepoints, expected {len(WHITE_SPACE)}"
    )

    head = [
        "//! Unicode range tables used by the pre-tokenizer (spec 3.1.4).",
        "//!",
        "//! GENERATED FILE - do not hand-edit. Regenerate with tools/gen_unicode.py.",
        f"//! Unicode version: {unicodedata.unidata_version}",
        "",
    ]
    body = "\n\n".join(
        render_table(n, r)
        for n, r in (("LETTER", letter), ("NUMBER", number), ("WHITESPACE", space))
    )
    return "\n".join(head) + "\n" + body + "\n\n"


def main():
    if unicodedata.unidata_version != REQUIRED_UCD:
        print(
            f"WRONG UCD: this interpreter carries Unicode "
            f"{unicodedata.unidata_version}, but src/unicode.rs is generated "
            f"from {REQUIRED_UCD}. Comparing the tables here would test the "
            f"host's Python, not this repository. Install the pinned database "
            f"with:  pip install unicodedata2=={REQUIRED_UCD}",
            file=sys.stderr,
        )
        return 2
    text = build()
    write = "--write" in sys.argv[1:]
    with open(TARGET, encoding="utf-8") as f:
        current = f.read()
    if write:
        if current == text:
            print(f"src/unicode.rs already current (Unicode {unicodedata.unidata_version})")
            return 0
        with open(TARGET, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"src/unicode.rs rewritten (Unicode {unicodedata.unidata_version})")
        return 0
    if current == text:
        print(f"OK: src/unicode.rs is byte-for-byte what this script generates "
              f"(Unicode {unicodedata.unidata_version})")
        return 0
    print(f"MISMATCH: src/unicode.rs differs from this script's output "
          f"(this Python carries Unicode {unicodedata.unidata_version}).", file=sys.stderr)
    print("Run with --write to regenerate, then re-run the test suite.", file=sys.stderr)
    import difflib
    for line in list(difflib.unified_diff(
            current.splitlines(), text.splitlines(),
            "committed", "generated", lineterm=""))[:40]:
        print(line, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
