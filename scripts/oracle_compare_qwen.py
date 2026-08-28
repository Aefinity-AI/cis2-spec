#!/usr/bin/env python3
"""E15d(c) correctness oracle: torch/transformers fp32 greedy decode of
Qwen/Qwen2.5-0.5B on the SAME prompt as the Rust reference, for comparison
against src/main.rs's CIS2_REF output. Same pattern/tolerance bar as
scripts/oracle_compare.py (E15b m1.5) for SmolLM2-135M.

Not part of the CIS-2 normative reference itself -- this is an external
correctness check only. See docs/E15d_bc_RESULT.md.
"""
import sys
import struct

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

WEIGHTS_DIR = sys.argv[2] if len(sys.argv) > 2 else \
    "weights_qwen"
PROMPT = "Once upon a time"
N_GEN = int(__import__("os").environ.get("CIS2_ORACLE_N_GEN", "16"))


def main():
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)

    print(f"torch version: {torch.__version__}", file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(WEIGHTS_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS_DIR, torch_dtype=torch.float32
    )
    model.eval()

    enc = tok(PROMPT, return_tensors="pt")
    prompt_ids = enc["input_ids"][0].tolist()
    print(f"prompt token ids: {prompt_ids}", file=sys.stderr)

    input_ids = enc["input_ids"]
    generated = list(prompt_ids)
    step0_logits = None

    # KV-cache incremental decode: without this, decoding N_GEN=512 steps by
    # re-running the FULL growing sequence from scratch every step is
    # O(N_GEN^2) attention work and is impractically slow in CI for a 1.5B
    # model (would be fine at N_GEN=16, as in the original E15d(c) script,
    # but not at 512). `use_cache=True` + `past_key_values` makes each step
    # O(1) in the already-processed prefix, matching what the Rust reference
    # does with its own KV cache.
    past_key_values = None
    with torch.no_grad():
        for step in range(N_GEN):
            if past_key_values is None:
                out = model(input_ids=input_ids, use_cache=True)
            else:
                out = model(
                    input_ids=input_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = out.past_key_values
            logits = out.logits[0, -1, :]
            if step == 0:
                step0_logits = logits.clone()
            next_id = int(torch.argmax(logits).item())
            generated.append(next_id)
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_id]], dtype=input_ids.dtype)],
                dim=1,
            )
            print(f"[oracle] step {step}: token_id={next_id}", file=sys.stderr)

    gen_only = generated[len(prompt_ids):]
    print(f"prompt_tokens={prompt_ids}")
    print(f"oracle_tokens={gen_only}")
    print(f"oracle_all_tokens={generated}")

    out_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/oracle_qwen_step0_logits.f32le"
    arr = step0_logits.to(torch.float32).numpy()
    with open(out_path, "wb") as f:
        for v in arr:
            f.write(struct.pack("<f", float(v)))
    print(f"wrote step-0 logits ({len(arr)} f32 values) to {out_path}", file=sys.stderr)

    # H1 512-tok oracle check: mirror the Rust reference's
    # CIS2_DUMP_LASTSTEP_LOGITS -- one more forward pass over the full
    # prompt+generated sequence to get the logits at the final position
    # (the same logits the Rust loop holds after its last iteration,
    # which predicts the NEXT, not-yet-generated token). Only run if
    # requested via env var, to avoid the extra full-sequence forward
    # pass on the default 16-tok path.
    laststep_path = __import__("os").environ.get("CIS2_ORACLE_DUMP_LASTSTEP")
    if laststep_path:
        with torch.no_grad():
            out = model(input_ids=input_ids)
            last_logits = out.logits[0, -1, :].to(torch.float32).numpy()
        with open(laststep_path, "wb") as f:
            for v in last_logits:
                f.write(struct.pack("<f", float(v)))
        print(
            f"wrote final-step logits ({len(last_logits)} f32 values) to {laststep_path}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
