# Hall of Divergence

This file records, by name, everyone who has independently run a CIS-2
conforming implementation and reported the result — whether it matched or
not.

There is no cash offer attached to this. The reward is the record: your
name, your machine, and your finding, published here and cited wherever
the result is used.

## What to send

Open an issue on this repository titled `DIVERGENCE:` or `REPRODUCTION:`
with:

- the four digests your run produced (witness / `CIS2_REF`, argmax,
  table, inv_freq_table), or the `selfcheck.sh` summary line;
- the machine: CPU model, ISA flags, OS and kernel, compiler and version;
- the full log, not a trimmed excerpt;
- which implementation you ran — `verify2/` (Rust), `verify3/` (C), or
  one you wrote yourself from `docs/CIS2_SPEC_v0.3b.md`.

A divergence caused by a spec sentence that permits two honest readings is
the most useful thing anyone can send. It is a defect in the specification,
not in you, and it is what this file exists to surface.

## Reproductions

| Date | Who | Machine | ISA | Implementation | Result |
| --- | --- | --- | --- | --- | --- |
| _(none reported by outside parties yet)_ | | | | | |

Runs by the author and by this project's own CI are not listed here; they
are in the repository README and in the CI logs. This table is for
reproductions by people unaffiliated with the project.

## Divergences

| Date | Who | Machine | Root cause | Spec change |
| --- | --- | --- | --- | --- |
| _(none reported yet)_ | | | | |
