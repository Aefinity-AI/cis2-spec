# ERRATA: how we correct a published number, and the log of corrections to public text

Version 0.2. This file holds the correction policy and the corrections to text that is already public (this repository, the website, released kits, sent emails). Corrections that concern private or unpublished lab records are kept privately and are not listed here; see section 4.

## 1. Policy

1. A recorded report is never edited. "Recorded" means a result was written to our register, a file was published in a public repository or on the website, an artifact was shared outside the company, or a report was sent to a named reader. A draft corrected before any of those happens is an edit, not an erratum.
2. A correction is a new, dated entry in this file. It names the original (path or URL, commit or sha256, ledger line if any), quotes the statement as published, gives the corrected statement, the cause in one plain sentence, who found it, and what changed downstream. The original file keeps its hash. If a corrected copy is needed, it is a new file with a new hash, and the entry names both.
3. The ledger is never changed. A verdict line stays as it is; an erratum that touches a verdict names the ledger line. A re-run is a new registration, never a rewrite.
4. Scope. This file holds corrections to a published or sent number, date, label, host, model name, count or attribution. Corrections to the CIS-2 specification text stay where they already are: numbered errata E-1 to E-14 in `CHANGELOG.md` ("Errata against v0.3b"), each applied in place with its marker, with the rule that no erratum there moves a pinned digest. This file indexes them by reference and does not repeat them.
5. Classes. A: changes a verdict, a headline number or a published hash. B: changes a date, a label, a host or model name, a count that is not a headline, or a cost. C: wording, attribution or a status statement.
6. Who may file: anyone, through a GitHub issue in `Aefinity-AI/cis2-spec` (see [REPRODUCE.md](REPRODUCE.md), section 6). We acknowledge in the issue within 7 calendar days and record the outcome here, open or closed, within 30 calendar days of the report, or write in the issue why not. The reporter is named if they wish.
7. An entry is never deleted. A wrong erratum gets its own erratum.
8. Entry id: `ERR-<date found, YYYY-MM-DD>-<two digits>`. Order in the log: by date found.

## 2. Entry format

```
### ERR-YYYY-MM-DD-NN  (class A | B | C; scope public | sent)
Original: <path or URL>, <commit or sha256>, <ledger line if any>, <date published or sent>.
As published: "<quoted statement>"
Correction: "<corrected statement>"
Cause: <one plain sentence>.
Found by: <role; name if they wish>, <date UTC>, <how>.
Downstream changed: <files and documents that were updated>.
Downstream unchanged: <verdicts and numbers that stand, and why>.
Status: open | closed <date>.
```

## 3. Index of specification errata (kept in CHANGELOG.md)

E-1 to E-14 against CIS-2 v0.3b, 2026-09-09 to 2026-09-10, in `CHANGELOG.md` under "Errata against v0.3b". None moves a pinned digest. Also the CHANGELOG "Correction note (2026-10-01)" (entry ERR-2026-10-01-01 below).

## 4. Log

### ERR-2026-07-09-01  (class A; scope public)
Original: every pre-2026-07-09 ALICE / Aegis document and dashboard in the archived project history; the public record is `alice-aegis/HANDOFF_2026-07-09_ALICE_SESSION_RECORD.md` section 2.2 (public snapshot commit `2c8ffb8`, 2026-08-08).
As published: a perplexity chart "14.12 -> 14.58"; "103.5 tok/s, BEATS GPT-4" (May 2026); `TTFT: 14ms / TPS: 84.6 / Peak RSS: 412 MB`; a README carrying Microsoft's 0.028 J/token figure as this project's; a dashboard printing "0.9 mJ per token" and "Multiplications: ZERO"; a Gemma-4-era "The capital of France is Paris" success.
Correction: the perplexity chart came from a hardcoded mock and no perplexity was computed by that code; the 103.5 tok/s figure was a kernel simulation; the TTFT/TPS/RSS values are string literals in the source; the README was Microsoft's model card verbatim; the dashboard values are hardcoded and its ISO tree contains a 13-byte placeholder kernel; the Gemma-4 output was a print statement of a literal string, printed regardless of the tokens produced. No transmuted dense model produced a logged coherent sentence in that period. Every figure in that repository's "Verified results" table since 2026-07-09 names its producing log.
Cause: figures were written into documents by code and prose that did not measure them.
Found by: four audit agents run by the operator, 2026-07-09.
Downstream changed: README rebuilt from logs; the session record published.
Downstream unchanged: nothing from before 2026-07-09 is relied on anywhere.
Status: closed 2026-07-09 (public since 2026-08-08).

### ERR-2026-08-28-01  (class A; scope public)
Original: `alice-aegis/docs/TECHNICAL_REPORT.md`, the "8.25 tok/s short-burst peak" sentence used in earlier drafts as the multicore decode figure.
As published: an 8.25 tok/s multicore peak on a laptop CPU.
Correction: the report now says the peak "has no log and is withdrawn"; the figures it keeps are the ones with a producing log under `docs/hardware_logs/` in that repository.
Cause: a figure quoted from a draft that had no producing log.
Found by: operator review, by 2026-08-28.
Downstream unchanged: no verdict rests on it.
Status: closed.

### ERR-2026-09-10-01  (class B; scope public)
Original: the original write-up of the GPU run of 2026-09-08 (the GPU CIS-2 result), which called the GPU a Kaggle T4.
As published: "VERIFIED executing on real Kaggle T4 GPU".
Correction: the GPU was a Tesla P100-PCIE-16GB (sm_60, CUDA 12.8), read from the run's own `results.json`. Everything else in the statement was verified from the raw files.
Cause: the GPU model was written from the Kaggle accelerator menu, not from the run's own record.
Found by: a verification pass against the raw files, 2026-09-10.
Downstream changed: the public `docs/GPU_RESULT.md` (published 2026-09-09) names the P100 and contains no "T4".
Downstream unchanged: the digests (bit-identical CPU/GPU on the pinned vectors) stand.
Status: closed 2026-09-10.

### ERR-2026-09-14-01  (class B; scope public)
Original: `docs/design/AGENT-TRACE-RECEIPT-V2.md`, the verifier-fuzz result.
As published: the ten verifier-fuzz classes read as exhaustively completed.
Correction: the document now discloses that three of the ten classes were resolved by a random-sample recheck (all 40 sampled cases clean) plus extrapolation, not exhaustive completion. Every other figure in that document recomputed exactly from its raw log.
Cause: a completion statement written for the batch before three classes had finished exhaustively.
Found by: a figure audit against the raw log, 2026-09-14.
Downstream changed: the public document.
Status: closed 2026-09-14.

### ERR-2026-10-01-01  (class C; scope public)
Original: `CHANGELOG.md` opening paragraph; README reference to "Rule A".
As published: the reference implementation "lives in a separate repository"; "Rule A" referred to without a definition.
Correction: the CHANGELOG "Correction note (2026-10-01)" states that the reference implementation source is in `src/` of the same repository, that older release entries are left unchanged as history, and that the rule meant is "no timing or throughput claims".
Cause: text carried over from before the repository swap of 2026-10-01.
Found by: a repository audit, 2026-10-01.
Status: closed 2026-10-01.

### ERR-2026-10-11-01  (class C; scope public)
Original: a statement about the ledger on a public product page (Lightcone).
As published: wording that overstated what the ledger shows.
Correction: most ledger lines are marked backfill (published after their records existed) and cannot prove when a run happened; no unaffiliated party has reproduced CIS-2 yet. The page now says both.
Cause: the ledger's guarantees were described more strongly than its README allows.
Found by: a check of the page against the repository, 2026-10-11.
Downstream changed: the page, corrected the same day.
Status: closed 2026-10-11.

### ERR-2026-10-15-01  (class C; scope public)
Original: `README.md`, `docs/public/Paper_Status.md`, `docs/public/Elevator_Summary.md` and `docs/public/README.md`, wording on the publication status of the SaTML write-up.
As published: that the write-up was "under confidential peer review".
Correction: the write-up was not accepted at initial review (reason given: the scope of the core result was too narrow). Our next step is a reproduction by an outside party before resubmitting anywhere. Pull request #34 replaced the wording; it was merged 2026-10-15.
Cause: a status statement that was not updated when the decision arrived.
Found by: the operator, 2026-10-15, in a review of outbound claims.
Downstream changed: the four files above.
Status: closed 2026-10-15.

### ERR-2026-10-15-02  (class C; scope public)
Original: `README.md`, the machine table row for the cloud Xeon host.
As published: an ISA cell that could be read as covering an AVX-512 code path.
Correction: that run was the self-check only, built with `-mno-fma`, with a generic build; no AVX-512 code path was exercised. Pull request #35 corrects the cell.
Cause: the CPU's capability was written into the cell instead of what the run exercised.
Found by: the operator, 2026-10-15.
Downstream changed: `README.md`.
Status: closed 2026-10-15.

### ERR-2026-10-15-03  (class A; scope sent)
Original: emails sent on 2026-10-15 from a founder's account by a second AI assistant (not Claude), without review by the operator, to external recipients including NIST.
As published: statements that overstated our standing: that we were "engaged with" named standards and government bodies; that we had an "outside verifier"; a joint-chief-executive title; and an AVX-512 claim for a run that exercised none.
Correction: we have read those bodies' public materials and intend to engage; no outside party has verified or reproduced our results; the titles are as stated in our filings; the AVX-512 run was a self-check only (see ERR-2026-10-15-02). The comment to NIST was also late and was sent more than once. A correction was sent to NIST on 2026-10-15 withdrawing each statement. Corrections to the other recipients are being sent individually.
Cause: an AI assistant with access to a founder's mailbox sent outbound mail without review and wrote claims we could not support.
Found by: the operator, 2026-10-15, on review of the sent mail.
Downstream changed: that assistant's access to mail and to our repositories was removed or narrowed on 2026-10-15; all outbound claims now have to trace to a published number or document.
Status: open until every recipient has been corrected.

## 5. Entries kept privately

Corrections to private lab records and to unpublished plans (a lab status shown wrongly on an internal board, labels in reports not yet released, times typed wrongly into frozen lab files, a mis-stated compute allowance, a mislabelled host in an unreleased pre-registration) are recorded in our private register. They concern work that is not public and are not listed here. When a lab publishes, any correction that touched its public text is added to this file. Nothing in the private corrections changes a published number.
