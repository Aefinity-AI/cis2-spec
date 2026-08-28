#!/usr/bin/env python3
"""E15b milestone-1.5 correctness oracle: torch/transformers fp32 greedy
decode of HuggingFaceTB/SmolLM2-135M on the SAME prompt as the Rust
reference, for comparison against src/main.rs's CIS2_REF output.

Not part of the CIS-2 normative reference itself -- this is an external
correctness check only. See docs/E15b_m1p5_CORRECTNESS.md.
"""
import sys
import json
import struct

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

WEIGHTS_DIR = "weights"
PROMPT = "Once upon a time"
N_GEN = 16

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

    with torch.no_grad():
        for step in range(N_GEN):
            out = model(input_ids=input_ids)
            logits = out.logits[0, -1, :]  # last position, full vocab
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

    # Dump step-0 full fp32 logit vector as raw LE bytes, matching the
    # Rust reference's CIS2_DUMP_STEP0_LOGITS format exactly.
    out_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/oracle_step0_logits.f32le"
    arr = step0_logits.to(torch.float32).numpy()
    with open(out_path, "wb") as f:
        for v in arr:
            f.write(struct.pack("<f", float(v)))
    print(f"wrote step-0 logits ({len(arr)} f32 values) to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
