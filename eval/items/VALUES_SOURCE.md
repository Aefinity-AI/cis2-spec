# values.jsonl provenance

`eval/items/values.jsonl` is a byte-identical copy of
`alice-aegis-cm-values` repo, branch `cm/values-charter`,
`values/cases.jsonl`.

- sha256: `0c58c075da9cf854d9acb615bea787e4310242082221b14fbe736b498d822a56`
- 48 cases, schema: `id`, `value`, `expect`, `prompt`, `pass_if`, `fail_if`, `source`.

This file must not be edited here; any changes to the case set happen
upstream in alice-aegis-cm-values and get re-vendored (with an updated,
re-verified sha256 recorded above).

Run via this repo's harness:

```
python3 eval/run_eval.py --mode run --categories values
python3 eval/run_eval.py --mode verify --categories values
```

`values` is an UNGRADED category (see `score_values` in `eval/run_eval.py`):
it records raw generations + CIS-2 receipt digests only, no pass/fail
verdict. `n_pass`/`score_frac` are `None` for this category by design.
Grading against each case's `pass_if`/`fail_if` is a separate, later task.
