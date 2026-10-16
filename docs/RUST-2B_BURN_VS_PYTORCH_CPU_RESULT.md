# rust-2b result: Burn (pure Rust) vs PyTorch, CPU training step

Date: 2026-10-16. Verdict: **FRAMEWORK: BURN BEHIND** (framework, not language).

Scale: 1 to 5 million parameters, CPU, 200 steps. GPU training and 135M to
360M models are untested and do not follow.

## What was measured

Training speed, final loss and peak memory of the same small model in two
frameworks on one old laptop CPU:

- Model: byte-level decoder transformer, 1.8M parameters, vocabulary 256,
  context 64, d_model 192, 4 layers, 6 heads, GELU feed-forward 768, tied head.
- Arm B: Burn 0.21.0, CPU "Flex" backend with autodiff, pure Rust (no -sys
  crates, no C compiler driven by a build script).
- Arm T: PyTorch 2.14.0+cpu, fp32, `torch.nn` primitives, no `torch.compile`.
- Same initial weights and same batches in both arms (shared init file and
  batch-order file); AdamW, lr 3e-4, batch 16 sequences, 5 untimed warm-up
  steps then 200 timed steps. A one-step pre-timing check confirmed the two
  arms agree on loss and parameters before any timing.
- Host: box1, Intel i5-5200U (2.20 GHz), threads N=1. N=1 host, one machine.
  Median of 5 repetitions per arm, arm order rotated each repetition.

## Registered bar

Written 2026-10-03, before any run. It could come out against Rust, for Rust,
or as a tie. With r = Burn median steps/s divided by PyTorch median steps/s:

- Loss condition, checked before the speed labels: |loss_B(200) - loss_T(200)| <= 0.05 nats, otherwise
  the speed labels are invalid.
- BURN AHEAD: r > 1.10 and every Burn repetition above every PyTorch one.
- TIE: 0.90 <= r <= 1.10.
- BURN BEHIND: r < 0.90 and every PyTorch repetition above every Burn one.

Only box1 N=1 carries a label.

## Numbers

| Quantity | Burn | PyTorch |
|---|---|---|
| Steps/s, 5 repetitions | 1.4962, 1.4570, 1.5111, 1.5034, 1.5054 | 3.3585, 3.2588, 3.2442, 3.3791, 3.3774 |
| Steps/s, median | 1.5034 | 3.3585 |
| Final loss at step 200 (mean of last 10) | 2.4778 | 2.4778 |
| Peak PSS, median | 151603 kB | 424351 kB |

- Steps/s ratio r = 0.4476463 (Burn is behind).
- Final loss: 2.4778 against 2.4778; the loss condition holds.
- Burn peak PSS ratio: 0.357 (Burn lower).
- Every PyTorch repetition (minimum 3.2442) is above every Burn repetition
  (maximum 1.5111), so the BEHIND clause is met.

Reported sentence: FRAMEWORK, not language (CPU, 1.8M-parameter byte
transformer, 200 steps, N=1): Burn Flex, pure Rust is behind PyTorch on box1,
steps/s ratio r=0.4476463, final loss 2.4778 against 2.4778. FRAMEWORK
(memory): Burn peak PSS is lower than PyTorch on box1, ratio 0.357; this is a
no-Python effect, not a Rust-language effect.

The result answers FRAMEWORK (a Rust training framework against PyTorch). It
does not answer LANGUAGE (there is no C arm) or KERNELS (each framework brings
its own kernels).

## Scope limits

- 1 to 5 million parameters only; 135M to 360M models are untested.
- CPU only; GPU training is untested.
- N=1 host (box1). A second host (box2) was run but is unlabelled: its N=1
  timing was invalid (clock drop of 17.7 percent on one Burn repetition) and
  its N=2 run had too few valid repetitions (valid reps T=1, B=2, need 5; seven
  rows cost-capped). N=2 on box1 (r=0.350) is reported without a label.
- 200 steps, one model shape, one dataset.

## Check of the figures and two minor notes

A second automated pass recomputed the figures from the row logs (steps/s,
final loss, PSS, repetition ranges, thread counts, thermal and busy checks) and
reproduced them. It recorded two minor notes:

1. `cpu_max` is null on both hosts: the runner records null when the cgroup
   `cpu.max` file is absent, so the job did not record that no quota applied.
   CPU seconds over wall seconds was 1.00 for both arms at N=1, which is
   consistent with no cap. It is not a bar condition.
2. On box2, the run wrapper that was used does not match the box2 wrapper hash
   listed in the final wrapper hash file (the hash recorded for the run itself
   matches), and box2 is the unlabelled host, so the labelled result is not
   affected.

## Reproduce

The run scripts and frozen build record are not published in this repository;
scripts available on request. The registered bar, model specification and the
numbers above are enough to rebuild the comparison: Burn 0.21.0 with features
`flex` and `autodiff`, PyTorch 2.14.0+cpu, the model and optimizer settings
listed above, 5 repetitions per arm at N=1.
