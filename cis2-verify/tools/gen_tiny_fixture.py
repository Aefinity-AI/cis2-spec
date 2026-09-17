#!/usr/bin/env python3
"""Generate the tiny end-to-end fixture in tests/fixtures/tiny/.

Why this exists: every other test in this crate is a unit test over one
function, or a golden that only a 500 MB checkpoint can exercise. Nothing
ran the whole pipeline -- safetensors -> config -> tokenizer -> forward pass
-> witness chain -> receipt render -> receipt parse -> verify -- in CI. This
builds a model small enough to commit (a few kB) that still exercises every
stage: BF16 widening, GQA (2 query heads over 1 KV head), RoPE, RMSNorm,
SwiGLU, an untied lm_head, and a multi-token prompt.

It is NOT a conformance vector and proves nothing about the spec's pinned
digests. It is a regression harness: if a refactor changes any stage's
arithmetic, tests/end_to_end.rs fails.

Determinism: no RNG. Every weight is drawn from a fixed table by index, so
running this script on any machine, in any Python, rewrites the identical
bytes. Regenerate with:  python3 tools/gen_tiny_fixture.py
"""
import json, os, struct

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "tests", "fixtures", "tiny")

H, HEADS, KV_HEADS, INTER, LAYERS, VOCAB = 8, 2, 1, 16, 1, 16
HEAD_DIM = H // HEADS          # 4
KV = KV_HEADS * HEAD_DIM       # 4

# Exactly representable BF16 patterns, so no rounding happens on the way in
# and the file is byte-reproducible without depending on a float formatter.
WEIGHT_BITS = [0x3E80, 0xBE80, 0x3E00, 0x3F00, 0xBE00, 0x3EC0, 0xBF00, 0x3F40]
#              0.25    -0.25   0.125   0.5    -0.125   0.375   -0.5    0.75
NORM_BITS = [0x3F80, 0x3F40, 0x3F00, 0x3FA0]   # 1.0, 0.75, 0.5, 1.25


def salt(name):
    """A stable per-tensor offset so two same-shaped tensors differ."""
    s = 0
    for ch in name.encode():
        s = (s * 131 + ch) & 0xFFFF
    return s


def tensor(name, n, row, table):
    """`row` is the trailing dimension. The per-row rotation matters: without
    it, index*7 mod 8 repeats with period 8 and every 8-wide row comes out
    byte-identical, which makes every lm_head logit equal and every argmax a
    16-way tie broken by 11.2's first-maximal rule. That is a real case, but
    it is covered by the argmax tie golden in mathpin.rs; here we want rows
    that differ, so decoded tokens actually vary."""
    off = salt(name)
    return b"".join(
        struct.pack("<H", table[(i * 7 + off + 3 * (i // row)) % len(table)])
        for i in range(n)
    )


def main():
    os.makedirs(OUT, exist_ok=True)

    tensors = [
        ("model.embed_tokens.weight", [VOCAB, H], VOCAB * H, H, WEIGHT_BITS),
        ("model.norm.weight", [H], H, H, NORM_BITS),
        ("lm_head.weight", [VOCAB, H], VOCAB * H, H, WEIGHT_BITS),
    ]
    for i in range(LAYERS):
        p = f"model.layers.{i}"
        tensors += [
            (f"{p}.input_layernorm.weight", [H], H, H, NORM_BITS),
            (f"{p}.post_attention_layernorm.weight", [H], H, H, NORM_BITS),
            (f"{p}.self_attn.q_proj.weight", [H, H], H * H, H, WEIGHT_BITS),
            (f"{p}.self_attn.k_proj.weight", [KV, H], KV * H, H, WEIGHT_BITS),
            (f"{p}.self_attn.v_proj.weight", [KV, H], KV * H, H, WEIGHT_BITS),
            (f"{p}.self_attn.o_proj.weight", [H, H], H * H, H, WEIGHT_BITS),
            (f"{p}.mlp.gate_proj.weight", [INTER, H], INTER * H, H, WEIGHT_BITS),
            (f"{p}.mlp.up_proj.weight", [INTER, H], INTER * H, H, WEIGHT_BITS),
            (f"{p}.mlp.down_proj.weight", [H, INTER], H * INTER, INTER, WEIGHT_BITS),
        ]

    # Sorted by name so the header is canonical regardless of the order above.
    tensors.sort(key=lambda t: t[0])

    header, payload, off = {}, bytearray(), 0
    for name, shape, n, row, table in tensors:
        blob = tensor(name, n, row, table)
        header[name] = {"dtype": "BF16", "shape": shape,
                        "data_offsets": [off, off + len(blob)]}
        payload += blob
        off += len(blob)

    # separators= with no spaces keeps the header byte-stable across versions.
    hjson = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    with open(os.path.join(OUT, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(payload)

    config = {
        "hidden_size": H, "intermediate_size": INTER,
        "num_hidden_layers": LAYERS, "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS, "vocab_size": VOCAB,
        "rms_norm_eps": 1e-05, "rope_theta": 10000.0,
        # Untied: the tied path is what the pinned SmolLM2 vector already
        # covers, so the fixture takes the branch nothing else exercises.
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16", "hidden_act": "silu",
    }
    with open(os.path.join(OUT, "config.json"), "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")

    # 16 single-char tokens plus three merges, so the prompt "abcd" encodes
    # to two ids ("abc", "d") and the multi-position attention path runs.
    chars = "abcdefghijklm"
    vocab = {c: i for i, c in enumerate(chars)}
    vocab["ab"] = 13
    vocab["bc"] = 14
    vocab["abc"] = 15
    assert len(vocab) == VOCAB and sorted(vocab.values()) == list(range(VOCAB))
    tok = {
        "version": "1.0",
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Digits", "individual_digits": True},
                {"type": "ByteLevel", "add_prefix_space": False,
                 "trim_offsets": True, "use_regex": True},
            ],
        },
        "post_processor": None,
        "decoder": {"type": "ByteLevel"},
        "model": {
            "type": "BPE", "dropout": None, "unk_token": None,
            "continuing_subword_prefix": None, "end_of_word_suffix": None,
            "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
            "vocab": vocab,
            "merges": ["a b", "b c", "ab c"],
        },
    }
    with open(os.path.join(OUT, "tokenizer.json"), "w") as f:
        json.dump(tok, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"wrote {OUT}: {len(tensors)} tensors, {8 + len(hjson) + len(payload)} bytes of weights")


if __name__ == "__main__":
    main()
