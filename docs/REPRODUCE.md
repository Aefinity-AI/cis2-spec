# REPRODUCE: how to check any result of ours

Version 0.2. Written for a stranger with a laptop. Every statement about what you can check today is a fact about this repository as it stands.

## 1. What you can check today, and how long it takes

| What | How | Time and cost | What a match shows | What it does not show |
|---|---|---|---|---|
| The CIS-2 digests (bit-identical fp32 inference of SmolLM2-135M) | `git clone https://github.com/Aefinity-AI/cis2-spec && cd cis2-spec && ./selfcheck.sh` | about 2 to 9 minutes on the maintainer's machines (README table: 1m58s to 8m32s); no account, no GPU; needs `git`, `curl`, `gcc`, `make`, `objdump`, a Rust toolchain. Without Rust: `scripts/self_check.sh` | your machine produced the same bits as ours under the specification; you can re-verify offline | that the model is correct, safe or honest; anything about sampling, batching, quantization, bf16 or GPU serving beyond `docs/GPU_RESULT.md` |
| The ledger chain | `python3 ledger/verify_ledger.py ledger/LEDGER.tsv` | seconds; Python 3, no network | the published fingerprints form an unbroken chain and the experiment numbers have no gaps | when any experiment ran, or how it turned out (section 2) |
| One ledger record you were given | `python3 ledger/verify_ledger.py ledger/LEDGER.tsv --record record.json --salt <64 hex>` | seconds | the record you hold is the one fingerprinted on that line, unchanged to the character | that the bar was sensible or the run honest |

Honest state: no unaffiliated person has reported a reproduction of any result of ours (README; `HALL-OF-DIVERGENCE.md` holds only our own internal rows). Replay instructions for individual labs are not published yet; this page will name them as they are. The full per-machine logs behind the README's machine table are not published ("log not published" in every row).

## 2. The ledger: what it is and how to read it

The ledger (`ledger/LEDGER.tsv`, `ledger/README.md`) is a public list of fingerprints of the records in our private experiment register. Before an experiment runs we write down its pass mark (the bar); after the run we write a verdict. Each record is fingerprinted as sha256(salt || canonical JSON) with its own random salt, and the lines are chained. The file shows the order of publication of a bar line and a verdict line, that no published line was rewritten, and that no experiment was removed from the middle. Some lines are marked backfill: they were published after their records existed.

What it cannot show, in the ledger README's own words: when an experiment ran; any order for backfill lines; whether every run got a verdict line; whether every experiment was registered; whether the run was honest or the bar sensible; how an experiment turned out. Dates come from GitHub's push and merge record, not from us. An OpenTimestamps proof file, `ledger/LEDGER.tsv.ots`, dates the file as a whole (check it with the free `ots verify`). Save the head hash the verifier prints and compare it next time; without a saved head hash, a rewrite from some line onward would pass the chain check.

## 3. Commit-reveal: how a ledger line ties to a record

The mechanism, and nothing about any particular record:

1. A ledger line holds an experiment number, a kind (for example bar or verdict) and fingerprints. It never holds a lab name or the record text.
2. The fingerprint is sha256(salt || canonical JSON of the record). The salt is random and secret until the record is revealed; one revealed salt tells you nothing about any other line, because each record has its own.
3. To reveal a record, the record JSON and its salt are handed over. Today that happens under a confidentiality agreement (`ledger/README.md`, "How a client under a confidentiality agreement checks one record").
4. The holder runs `verify_ledger.py --record --salt`. If the record or salt is wrong by one character the check prints FAIL; if it matches, the record is exactly what was fingerprinted on that line, and the line's position and GitHub date show it was public before anything published after it.
5. Without the record you can read only a line's kind, its position, and the GitHub date of the pull request that added it.

Public reports are meant to name their experiment number and ledger lines so you can start from the report and find the line. Reports written before that practice have no such block.

## 4. Running the offline verifier

For CIS-2 results:

```
git clone https://github.com/Aefinity-AI/cis2-spec && cd cis2-spec
./selfcheck.sh                 # full: fetch pinned weights (sha256-checked), build src/ + verify3/, rerun every pinned digest, FMA gate
scripts/self_check.sh          # shorter: digests only, no Rust needed
```

Expected: `== SUMMARY: 9 passed, 0 failed ==` (10 when the Hugging Face mirror cross-check also runs) and `PASS: all reproducible pinned digests match`. The pinned values are in `EXPECTED_DIGESTS.md` (SmolLM2-135M, prompt "Once upon a time", 16 greedy tokens; `CIS2_REF = d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df`). A non-zero `failed` count is a finding: `CHALLENGE.md` says what counts as a divergence (digest differs on a conforming machine), what is an environment failure (an FMA instruction in the binary, FTZ/DAZ not pinned), and what is out of scope (download errors, wrong model, non-fp32 paths, timing).

To verify a receipt directly: `verify3/cis2_verify3 <model.safetensors> <config.json> <tokenizer.json>` prints the digests and refuses a weight file whose sha256 does not match the pin. `cis2-verify/` is a further from-spec verifier that is not a clean room (`cis2-verify/CLEANROOM.md`).

For quantized local runs (GGUF, llama.cpp) there is no CIS-2 receipt. For hosted models there is no bit replay; the evidence is per-request hashes and batch ids, and the yardstick is a lab's own noise arm.

## 5. Replaying a lab

When a lab publishes a replay file it will list: inputs and their sha256; environment pins; the exact commands in order; expected outputs with hashes; the receipt step; time and hardware; what you cannot replay without a credential, a licence or a private file; the public/private split of the items. Expect bit-identity for local greedy runs on the same build, label-level agreement within the stated noise floor for hosted models, and statistics equal to a tight numerical tolerance from the same raw files with the stated seed. Raw completions of safety labs stay private (they may contain harmful text); per-item labels and hashes are public.

## 6. Filing a dispute

Open an issue in `Aefinity-AI/cis2-spec`. For a CIS-2 digest, use the existing `divergence-report.yml` or `reproduction-report.yml` templates. For any other published number, paste this:

```
Title: [result-dispute] <document>: <the number or statement disputed>

1. Which result: path or URL of the report, its commit or sha256, the experiment number and ledger line if given.
2. As published: quote the number or sentence exactly, with its interval and n as printed.
3. What you got: your number with its interval and n, from which script, at which commit.
4. How you ran it: the steps you followed; your environment (host, CPU, OS, Python, numpy, llama.cpp commit or API version); the sha256 of your inputs and outputs.
5. What you think the cause is (optional).
6. Naming: "credit me as <name or handle>" or "do not name me".
```

Reports missing items 1 to 4 get a request for the missing piece before anything can be reproduced.

## 7. What we commit to

- We acknowledge the issue within 7 calendar days.
- We try to reproduce your result on our own machines (older CPUs and a small GPU; hosted-model replays cost money and we say if we cannot afford one) and answer in the issue, in public, within 30 calendar days, or write in the issue why not and when.
- If you are right, the correction is a dated entry in [ERRATA.md](ERRATA.md) with the original hash, the corrected statement, the cause and your name or handle if you wish; the original file is not edited.
- If your run did not reproduce ours, that is recorded too: a verified divergence goes in `HALL-OF-DIVERGENCE.md` with your name or handle if you wish, whether or not it makes us look bad (`CHALLENGE.md`, "Honesty note"). That page is scoped to CIS-2 digests today.
- Credit only: no cash, no prize, no payment. The credit line in `HALL-OF-DIVERGENCE.md` is the credit.
- We are a very small team, and every reviewer of our own verdicts so far has been an AI agent run by the same operator; no human second-reviewer signature is in place yet. A reader deciding how much to trust a result should know that.

## 8. Where things are

| Item | Place |
|---|---|
| Specification and verifiers | `docs/CIS2_SPEC_v0.3b.md`, `verify2/`, `verify3/`, `cis2-verify/`, `src/` |
| Pinned digests | `EXPECTED_DIGESTS.md`, `docs/GPU_RESULT.md` |
| Specification errata | `CHANGELOG.md`, "Errata against v0.3b" (E-1 to E-14) |
| Result and page errata | [ERRATA.md](ERRATA.md) |
| Ledger | `ledger/` |
| Divergences and reproductions | `HALL-OF-DIVERGENCE.md`, `CHALLENGE.md`, `.github/ISSUE_TEMPLATE/` |
| Result write-ups | `docs/results/` |
| Public plain-language materials | `docs/public/` |
