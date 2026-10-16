#!/usr/bin/env python3
"""Constructed CIS-2 "coverage vector" model (numpy only). Writes weights/{model.safetensors,config.json,tokenizer.json}
and ids/coverage-4.txt. Deterministic given SEED.
Features: (a) exact argmax tie TIE_A<TIE_B (identical lm_head rows, distinct embeddings, max logit at decode step 0);
(b) subnormal logit SUB (lm_head row of bf16 subnormals 0x0001/0x8001 -> flushed to +0.0 under the 1.3 FTZ/DAZ pin,
non-zero without it); (c) 4-token prompt."""
import json, struct, sys, os
import numpy as np

SEED = 20261016
H, NH, NKV, L, I, V = 64, 2, 2, 2, 128, 32
EPS, THETA = 1e-5, 10000.0
TIE_A, TIE_B, SUB = 5, 17, 29
PROMPT = [3, 11, 22, 8]
TIE_SCALE = 2.0

def f32_to_bf16_bits(x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    u = x.view(np.uint32).astype(np.uint64)
    rnd = ((u >> 16) & 1) + 0x7FFF            # round-to-nearest-even
    return ((u + rnd) >> 16).astype(np.uint16)

def bf16_to_f32(b):
    return (b.astype(np.uint32) << 16).view(np.float32)

def forward_f64(T, ids):
    """Plain float64 forward of the last prompt position; used only to pick the sign pattern of the tie rows."""
    W = {k: bf16_to_f32(v).astype(np.float64) for k, v in T.items()}
    hd = H // NH
    inv = THETA ** (-(2.0 * np.arange(hd // 2)) / hd)
    def rms(x, w): return x / np.sqrt((x * x).mean() + EPS) * w
    def rope(x, pos):  # x [heads, hd], rotate-half
        a = pos * inv; c, s = np.cos(a), np.sin(a)
        x1, x2 = x[:, :hd // 2], x[:, hd // 2:]
        return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=1)
    kc = [[] for _ in range(L)]; vc = [[] for _ in range(L)]
    for pos, t in enumerate(ids):
        h = W["model.embed_tokens.weight"][t].copy()
        for l in range(L):
            p = f"model.layers.{l}."
            ln = rms(h, W[p + "input_layernorm.weight"])
            q = (W[p + "self_attn.q_proj.weight"] @ ln).reshape(NH, hd)
            k = (W[p + "self_attn.k_proj.weight"] @ ln).reshape(NKV, hd)
            v = (W[p + "self_attn.v_proj.weight"] @ ln).reshape(NKV, hd)
            q, k = rope(q, pos), rope(k, pos)
            kc[l].append(k); vc[l].append(v)
            K = np.stack(kc[l]); Vv = np.stack(vc[l])
            out = np.zeros((NH, hd))
            for hh in range(NH):
                sc = K[:, hh] @ q[hh] / np.sqrt(hd); e = np.exp(sc - sc.max()); e /= e.sum()
                out[hh] = e @ Vv[:, hh]
            h = h + W[p + "self_attn.o_proj.weight"] @ out.reshape(-1)
            ln = rms(h, W[p + "post_attention_layernorm.weight"])
            g = W[p + "mlp.gate_proj.weight"] @ ln; u = W[p + "mlp.up_proj.weight"] @ ln
            h = h + W[p + "mlp.down_proj.weight"] @ ((g / (1 + np.exp(-g))) * u)
    return rms(h, W["model.norm.weight"])

def build():
    rng = np.random.default_rng(SEED)
    n = lambda *s: rng.normal(0.0, 0.15, size=s).astype(np.float32)
    nw = lambda *s: (1.0 + 0.1 * rng.normal(size=s)).astype(np.float32)
    T = {}
    T["model.embed_tokens.weight"] = f32_to_bf16_bits(n(V, H))
    for l in range(L):
        p = f"model.layers.{l}."
        T[p + "input_layernorm.weight"] = f32_to_bf16_bits(nw(H))
        T[p + "post_attention_layernorm.weight"] = f32_to_bf16_bits(nw(H))
        T[p + "self_attn.q_proj.weight"] = f32_to_bf16_bits(n(H, H))
        T[p + "self_attn.k_proj.weight"] = f32_to_bf16_bits(n(NKV * (H // NH), H))
        T[p + "self_attn.v_proj.weight"] = f32_to_bf16_bits(n(NKV * (H // NH), H))
        T[p + "self_attn.o_proj.weight"] = f32_to_bf16_bits(n(H, H))
        T[p + "mlp.gate_proj.weight"] = f32_to_bf16_bits(n(I, H))
        T[p + "mlp.up_proj.weight"] = f32_to_bf16_bits(n(I, H))
        T[p + "mlp.down_proj.weight"] = f32_to_bf16_bits(n(H, I))
    T["model.norm.weight"] = f32_to_bf16_bits(nw(H))
    T["lm_head.weight"] = f32_to_bf16_bits(n(V, H))
    # embeddings of the tie pair must differ (random draws already do; assert)
    E = T["model.embed_tokens.weight"]; assert not np.array_equal(E[TIE_A], E[TIE_B])
    # (a) tie rows: +-TIE_SCALE following the sign of the final hidden state -> large positive, identical logit
    hfin = forward_f64(T, PROMPT)
    row = np.where(hfin >= 0, TIE_SCALE, -TIE_SCALE).astype(np.float32)
    T["lm_head.weight"][TIE_A] = f32_to_bf16_bits(row)
    T["lm_head.weight"][TIE_B] = f32_to_bf16_bits(row)
    # (b) subnormal row: bf16 0x0001 (f32 2^-133) with the sign of the hidden state, so no cancellation without FTZ
    T["lm_head.weight"][SUB] = np.where(hfin >= 0, 0x0001, 0x8001).astype(np.uint16)
    return T

def write_safetensors(path, T):
    hdr, off = {}, 0
    names = sorted(T)
    for k in names:
        nb = T[k].size * 2
        hdr[k] = {"dtype": "BF16", "shape": list(T[k].shape), "data_offsets": [off, off + nb]}
        off += nb
    hdr["__metadata__"] = {"format": "pt"}
    hj = json.dumps(hdr, separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj))); f.write(hj)
        for k in names: f.write(T[k].astype("<u2").tobytes())

def main(out):
    os.makedirs(out + "/weights", exist_ok=True); os.makedirs(out + "/ids", exist_ok=True)
    T = build()
    write_safetensors(out + "/weights/model.safetensors", T)
    cfg = {"architectures": ["LlamaForCausalLM"], "hidden_size": H, "intermediate_size": I, "num_hidden_layers": L,
           "num_attention_heads": NH, "num_key_value_heads": NKV, "vocab_size": V, "rms_norm_eps": EPS,
           "rope_theta": THETA, "tie_word_embeddings": False, "hidden_act": "silu", "torch_dtype": "bfloat16",
           "model_type": "llama"}
    open(out + "/weights/config.json", "w").write(json.dumps(cfg, indent=2) + "\n")
    open(out + "/weights/tokenizer.json", "w").write('{"version":"coverage-stub","model":{"type":"WordLevel","vocab":{}}}\n')
    open(out + "/ids/coverage-4.txt", "w").write("".join(f"{t}\n" for t in PROMPT))
    print(f"seed={SEED} TIE_A={TIE_A} TIE_B={TIE_B} SUB={SUB}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
