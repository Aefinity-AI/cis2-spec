# E15j clean-room (C11) — spec ambiguities / gaps encountered

**Note (added v0.2.1, 2026-08-28):** spec v0.2.1 added a normative §3.1
(H4) that pins the tokenizer JSON schema and BPE algorithm this clean-room
had to infer from general knowledge (item 1 below). This closes that
ambiguity **at the spec level**, but `verify3/tokenizer.c` itself has
**not** been re-derived from §3.1 — it was written before §3.1 existed,
from general pretrained knowledge of the public HuggingFace `tokenizers`
byte-level BPE format, not from the spec text. It happens to produce the
pinned §13.1 tokenization and is runtime-checked against that pinned
value (see item 2 below and `CLEANROOM_LOG.md`'s "Notes on non-spec
knowledge required"), but a reader should not treat `verify3/tokenizer.c`
as a from-§3.1-text-alone clean-room artifact until it is re-audited or
rewritten against §3.1 directly. This is an honest gap, not a claim of
compliance.

1. **Tokenizer JSON schema and BPE algorithm are not specified by
   `CIS2_SPEC_v0.2.md`** (§3.1 only says "loaded as a complete,
   self-contained HuggingFace `tokenizers`-format tokenizer (BPE-family)").
   Resolved using public, pretrained knowledge of the HF `tokenizers`
   library's on-disk schema (`model.vocab`, `model.merges`, GPT-2
   byte<->unicode mapping, greedy lowest-rank-pair BPE merge). Not a
   from-spec-text-only derivation, but this knowledge predates and is
   external to this repo — see CLEANROOM_LOG.md.

2. **Pre-tokenizer regex not fully implemented.** The full GPT-2/ByteLevel
   pre-tokenizer regex (`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+|
   ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`) requires Unicode property classes;
   `verify3/tokenizer.c`'s `pretokenize()` implements an ASCII-sufficient
   approximation (single optional leading space + letter run / digit run /
   other-non-space run / whitespace run). This is adequate for the pinned
   §13.1 prompt `"Once upon a time"` (verified: tokenizes to the pinned
   `[6403, 1980, 253, 655]`, checked at runtime in `main.c` against the
   spec's pinned value, hard-fails if it ever mismatches) but would not be
   correct for arbitrary Unicode input. Out of scope for this spec's
   normative test vector.

3. **§9.1 QKV bias / §11.1 untied lm_head branches are implemented but
   untested by this clean-room build** (SmolLM2-135M has neither — no
   bias tensors, tied embeddings) — matches the spec's own statement that
   these branches are "informative for SmolLM2-135M" (spec text, §9.1/§11.1).

4. **KV-cache vs. full-recompute-per-step**: spec §3.5 states both are
   conformant iff bit-identical. This implementation recomputes the full
   forward pass from scratch at every decode step (no incremental KV
   cache) — simpler to get right, explicitly declared conformant by the
   spec, and no timing requirement applies (task rules: no local model
   runs, no timing numbers; CI-only correctness).

No ambiguity was found in the numeric/transcendental core (§1, §5, §6, §7,
§8) — the pinned `table_digest` and `inv_freq_table_digest` reproduced
bit-for-bit against the spec's §13.1 values on the first local unit-test
run (`verify3/selftest.c`, no model weights required), which is strong
evidence the polynomial/reduction-order transcription is correct.
