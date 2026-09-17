#!/usr/bin/env python3
"""Spec 14.6 evidence, oracle half: dump every intermediate activation of a
`transformers` fp32 forward pass, in the same record format and under the
same tensor names as `cis2-verify --features layerdump`.

Section 14.6 records that section 13.3's oracle checks are spot-checks on the
*output* of the layer stack, and that a compensating pair of errors inside
the stack that preserved step-0's logits and every argmax decision could not
be ruled out by them. Comparing the intermediates rules it out. This produces
the oracle side of that comparison; `scripts/compare_layers.py` does the
comparison.

This is not part of the CIS-2 normative reference. It is an external
correctness check, like `scripts/oracle_compare.py`.

Method. The token sequence is supplied on the command line -- the prompt ids
and the ids the verifier generated -- and stepped one token at a time with a
KV cache, so that each oracle forward corresponds to exactly one verifier
forward at the same position. That is teacher forcing: it isolates the
numerics from any divergence in token choice. The oracle's own argmax at each
step is recorded separately, so token agreement is still checked rather than
assumed.

Usage:
    oracle_layers.py <weights-dir> <out.bin> <comma-separated-token-ids>
"""
import struct
import sys

import torch
from transformers import AutoModelForCausalLM

CAPTURE_OUT = [
    # (module path template, tensor name template)
    ("model.embed_tokens", "p{p}.embed"),
    ("model.norm", "p{p}.final_norm"),
    ("lm_head", "p{p}.logits"),
]
PER_LAYER_OUT = [
    ("input_layernorm", "ln1"),
    ("self_attn.q_proj", "q_proj"),
    ("self_attn.k_proj", "k_proj"),
    ("self_attn.v_proj", "v_proj"),
    ("self_attn.o_proj", "o_proj"),
    ("post_attention_layernorm", "ln2"),
    ("mlp.gate_proj", "gate_proj"),
    ("mlp.up_proj", "up_proj"),
    ("mlp.down_proj", "down_proj"),
]
PER_LAYER_IN = [
    # the input to a module is the value the verifier names on the left
    ("self_attn.o_proj", "attn_out"),
    ("post_attention_layernorm", "resid_attn"),
    ("mlp.down_proj", "mlp_act"),
]


def getmod(root, path):
    m = root
    for part in path.split("."):
        m = getattr(m, part)
    return m


def main():
    weights, out_path, ids_csv = sys.argv[1], sys.argv[2], sys.argv[3]
    ids = [int(x) for x in ids_csv.split(",")]

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False

    model = AutoModelForCausalLM.from_pretrained(weights, dtype=torch.float32)
    model.eval()
    print(f"ORACLE torch={torch.__version__} class={type(model).__name__}", file=sys.stderr)
    print(f"ORACLE layers={len(model.model.layers)} tokens={len(ids)}", file=sys.stderr)

    sink = open(out_path, "wb")
    state = {"pos": 0}

    def emit(name, t):
        v = t.detach().reshape(-1).contiguous().to(torch.float32)
        nb = name.encode("ascii")
        sink.write(struct.pack("<I", len(nb)))
        sink.write(nb)
        sink.write(struct.pack("<I", v.numel()))
        sink.write(v.numpy().tobytes())

    handles = []

    def hook_out(name_t):
        def fn(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            emit(name_t.format(p=state["pos"]), o)
        return fn

    def hook_in(name_t):
        def fn(mod, inp):
            emit(name_t.format(p=state["pos"]), inp[0])
        return fn

    for path, name_t in CAPTURE_OUT:
        handles.append(getmod(model, path).register_forward_hook(hook_out(name_t)))
    for li, layer in enumerate(model.model.layers):
        for path, short in PER_LAYER_OUT:
            handles.append(
                getmod(layer, path).register_forward_hook(hook_out("p{p}.L%d.%s" % (li, short)))
            )
        for path, short in PER_LAYER_IN:
            handles.append(
                getmod(layer, path).register_forward_pre_hook(hook_in("p{p}.L%d.%s" % (li, short)))
            )
        # resid_mlp is the decoder layer's own output
        handles.append(layer.register_forward_hook(hook_out("p{p}.L%d.resid_mlp" % li)))

    argmaxes = []
    past = None
    with torch.no_grad():
        for p in range(len(ids)):
            state["pos"] = p
            out = model(
                input_ids=torch.tensor([[ids[p]]]),
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            argmaxes.append(int(out.logits[0, -1].argmax()))

    for h in handles:
        h.remove()
    sink.close()

    print(f"ORACLE argmax-per-step={argmaxes}", file=sys.stderr)
    print(f"ORACLE bytes={sum(1 for _ in [0]) and __import__('os').path.getsize(out_path)}",
          file=sys.stderr)
    # the token the oracle would have emitted after each step
    print(",".join(str(a) for a in argmaxes))


if __name__ == "__main__":
    main()
