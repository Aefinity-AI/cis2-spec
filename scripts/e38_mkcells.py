# E38: build per-model cell lists for the 14.1(a) reach census.
# Every cell supplies its prompt token ids from the model's own tokenizer, so a
# non-ASCII prompt does not depend on 3.1.4's still-open Unicode question and a
# Qwen cell does not depend on 3.1.3/3.1.4 refusing its tokenizer.json (E-3).
from transformers import AutoTokenizer

CELLS = [
    # (gen_toks, prompt, why this cell is here)
    (16, "Once upon a time",                    "13.1 reference decode, ASCII control"),
    (16, "こんにちは世界", "Japanese, 3-byte UTF-8"),
    (16, "Привет мир", "Cyrillic, 2-byte UTF-8"),
    (16, "مرحبا بالعالم", "Arabic, RTL"),
    (16, "\U0001f642\U0001f680✨ emoji test", "4-byte UTF-8 astral plane"),
    (16, "café naïve résumé", "accented Latin, NFC-sensitive"),
    (16, "∑∫∂∇ ℵ₀ ⊗", "mathematical symbols"),
    (16, "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", "32x repetition, degenerate attention"),
    (16, "!!!!!!!!!!!!!!!!", "punctuation repetition"),
    (256, "Once upon a time", "4x the previous generated-token ceiling"),
]

for tag, path in (("smollm2", "/home/justinbrianthompson/projects/cis2-spec/weights"),
                  ("qwen", "/home/justinbrianthompson/qwen05b")):
    tok = AutoTokenizer.from_pretrained(path)
    with open(f"/home/justinbrianthompson/e38-cells-{tag}.txt", "w") as f:
        for g, p, why in CELLS:
            ids = tok(p, add_special_tokens=False)["input_ids"]
            f.write(f"{g}|{p}|{','.join(str(i) for i in ids)}|{why}\n")
    print(tag, "written")
