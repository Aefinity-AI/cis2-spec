# Malformed safetensors regression inputs

Files here are named by a 16-hex-digit prefix of their SHA-256. Most are
minimised inputs from an ASan fuzzing run of this loader (lab rust-3); a few
are hand-built (non-string JSON key, non-object header, negative / reversed /
fractional `data_offsets`, wrong dtype, header length of 2^63).

They are data only and are never executed. `make fuzzcheck` (in `verify3/`)
passes each one to `cis2_model_load()` in a child process and requires a
clean rejection (exit status 95) with no signal and no sanitizer report.
