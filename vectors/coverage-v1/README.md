# Coverage vector v1 (informative)

A tiny constructed model plus a 4-token prompt, built so that two clauses the SmolLM2-135M vector never reaches are reached on purpose: the section 1.3 FTZ/DAZ pin and the section 11.2 lowest-index tie rule. It is informative, not normative; the pinned section 13.1 vector is unchanged. Write-up: [`docs/results/2026-10-16-COVERAGE-VECTOR.md`](../../docs/results/2026-10-16-COVERAGE-VECTOR.md). Expected digest: `EXPECTED_DIGESTS.md`, section "Informative: coverage vector".

The weights are our own seeded random numbers (seed 20261016), not a trained model and not derived from any third-party checkpoint.

| file | sha256 |
|---|---|
| `weights/model.safetensors` | `677a5a25208b08594b0cb9375cb28ed427bdfeec72ddcd2833b0513e7031f4ca` |
| `weights/config.json` | `af88fb256b150dad9050f760c6390ef45a9c539c4d14c0bcea4fa31b125e9695` |
| `weights/tokenizer.json` (stub, content unused) | `e39fc11fc1c2b004fcf5f6b5a817f79562037c05a256439a8d4cdef8ecb5dd23` |
| `ids/coverage-4.txt` (prompt ids 3,11,22,8) | `aa1bb87d585ab545985af17145a4e71a83bc1c0e563aec8ca9c2dd1c43f14853` |

Regenerate (numpy only): `python3 -I gen_coverage_model.py <outdir>` writes `weights/` and `ids/` with the hashes above.

Run: greedy decode of `gen_toks=8` from `ids/coverage-4.txt` with the sequential-reduction `CIS2_REF` witness. Expected `CIS2_REF = 4660d6c33b6c0d51529b4be43687c004c8b7c0d8804439999b5af32683528734`, generated ids `5,5,5,5,5,5,5,5`.

`mutants/m01.diff` and `mutants/m15.diff` are the two single-clause edits used in the write-up (diffs against the C++ host source, which is not part of this tree; see `docs/REPRODUCTIONS.md` note 2).
