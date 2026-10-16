# Model catalog record schema, draft v0.1

Status: draft v0.1, 2026-10-16

## 1. Purpose

A model catalog record is one JSON document per model artifact. It binds the
artifact's content hashes to its training manifest, its CIS-2 conformance
digests, its evaluation results and its licence, so that a registry can store
it and a reader can see exactly what was checked, on which machine, from which
log, and what was not checked. The format is built around two rules. Every
number, digest or hash carries the host that produced it and a pointer to the
log, report or commit it was read from. Absence is stated explicitly
(`"status": "not_run"`), never filled with a placeholder. The record has no
single "verified" flag: checking is recorded per section and per host.

## 2. Design rules

1. **No bare numbers.** An eval result without `host` and `source` is invalid
   under the schema.
2. **Determinism claims are scoped.** Matching loss values do not show
   matching weights; a training run can print identical losses while its
   checkpoint bytes differ across CPU instruction sets. Loss therefore lives
   under `training.checkpoints[].loss`, and the claim level lives under
   `training.determinism_claim`. The level `cross-machine-bit-identical`
   requires at least two hosts and `cross_machine_status` equal to
   `confirmed-match`; the schema rejects it otherwise.
3. **Identity is a content hash.** The artifact is identified by
   `artifact_ref.weights_sha256`, not by a name or version string.
4. **A partial digest set is not a result.** The four CIS-2 digests
   (`witness`, `argmax`, `table`, `inv_freq`) are required together.
5. **Third-party weights are allowed.** `training.status = "third_party"`
   records that the weights were not trained by the record's author and that
   no training claim is made.

## 3. JSON Schema

JSON Schema draft 2020-12. The same document is published as
[`schema/model-catalog.schema.json`](../schema/model-catalog.schema.json).

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://github.com/Aefinity-AI/cis2-spec/schema/model-catalog.schema.json",
  "title": "Model catalog record",
  "description": "One record per model artifact: identity, training manifest, CIS-2 conformance digests, eval results, licence. Draft v0.1.",
  "type": "object",
  "required": ["schema_version", "record_id", "model_name", "licence", "artifact_ref", "training", "inference_conformance", "eval"],
  "additionalProperties": false,
  "$defs": {
    "sha256": { "type": "string", "pattern": "^[0-9a-f]{64}$" },
    "source": { "type": "string", "minLength": 1, "description": "Path, URL or commit identifying the log or report the value was read from." },
    "host": {
      "type": "object",
      "required": ["host", "cpu_or_gpu"],
      "additionalProperties": false,
      "properties": {
        "host": { "type": "string", "minLength": 1 },
        "cpu_or_gpu": { "type": "string", "minLength": 1 },
        "framework_version": { "type": "string" },
        "kernel_pin": { "type": "string", "description": "Forced-kernel environment settings, or the literal string none." }
      }
    }
  },
  "properties": {
    "schema_version": { "const": "model-catalog/0.1" },
    "record_id": { "type": "string", "minLength": 1 },
    "model_name": { "type": "string", "minLength": 1 },
    "licence": { "type": "string", "minLength": 1, "description": "SPDX identifier, UNLICENSED, PROPRIETARY, or unknown-needs-decision." },
    "artifact_ref": {
      "type": "object",
      "required": ["weights_sha256"],
      "additionalProperties": false,
      "properties": {
        "model_id": { "type": "string" },
        "weights_sha256": { "$ref": "#/$defs/sha256" },
        "config_sha256": { "$ref": "#/$defs/sha256" },
        "tokenizer_sha256": { "$ref": "#/$defs/sha256" },
        "repo": { "type": "string" },
        "commit": { "type": "string" }
      }
    },
    "training": {
      "type": "object",
      "required": ["status"],
      "additionalProperties": false,
      "properties": {
        "status": { "enum": ["run", "not_run", "third_party"] },
        "note": { "type": "string" },
        "seed": { "type": "integer" },
        "total_steps": { "type": "integer", "minimum": 1 },
        "model_config": { "type": "object" },
        "param_counts": {
          "type": "object",
          "required": ["total"],
          "properties": { "total": { "type": "integer" }, "trainable": { "type": "integer" } }
        },
        "data_sha256": { "$ref": "#/$defs/sha256" },
        "checkpoints": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["step", "checkpoint_sha256", "host"],
            "additionalProperties": false,
            "properties": {
              "step": { "type": "integer", "minimum": 0 },
              "loss": { "type": "number" },
              "checkpoint_sha256": { "$ref": "#/$defs/sha256" },
              "host": { "type": "string" }
            }
          }
        },
        "hosts": { "type": "array", "items": { "$ref": "#/$defs/host" } },
        "determinism_claim": {
          "type": "object",
          "required": ["level", "source"],
          "additionalProperties": false,
          "properties": {
            "level": { "enum": ["unverified", "same-machine-reproducible", "cross-machine-bit-identical"] },
            "cross_machine_status": { "enum": ["not_attempted", "unresolved", "confirmed-match", "confirmed-mismatch"] },
            "note": { "type": "string" },
            "source": { "type": "array", "items": { "$ref": "#/$defs/source" }, "minItems": 1 }
          }
        }
      },
      "allOf": [
        {
          "if": { "properties": { "status": { "const": "run" } }, "required": ["status"] },
          "then": { "required": ["seed", "total_steps", "checkpoints", "hosts", "determinism_claim"] }
        },
        {
          "if": {
            "properties": { "determinism_claim": { "properties": { "level": { "const": "cross-machine-bit-identical" } }, "required": ["level"] } },
            "required": ["determinism_claim"]
          },
          "then": { "properties": { "hosts": { "minItems": 2 }, "determinism_claim": { "properties": { "cross_machine_status": { "const": "confirmed-match" } }, "required": ["cross_machine_status"] } } }
        }
      ]
    },
    "inference_conformance": {
      "type": "object",
      "required": ["status"],
      "additionalProperties": false,
      "properties": {
        "status": { "enum": ["run", "not_run"] },
        "spec": { "type": "string" },
        "spec_version": { "type": "string" },
        "test_vector": { "type": "string" },
        "digests": {
          "type": "object",
          "required": ["witness", "argmax", "table", "inv_freq"],
          "additionalProperties": false,
          "properties": {
            "witness": { "$ref": "#/$defs/sha256" },
            "argmax": { "$ref": "#/$defs/sha256" },
            "table": { "$ref": "#/$defs/sha256" },
            "inv_freq": { "$ref": "#/$defs/sha256" }
          }
        },
        "hosts_replayed": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["host", "cpu_or_gpu", "result"],
            "additionalProperties": false,
            "properties": {
              "host": { "type": "string" },
              "cpu_or_gpu": { "type": "string" },
              "result": { "enum": ["PASS", "FAIL"] }
            }
          }
        },
        "overall_result": { "enum": ["MATCH", "MISMATCH", "PARTIAL"] },
        "source": { "$ref": "#/$defs/source" }
      },
      "if": { "properties": { "status": { "const": "run" } }, "required": ["status"] },
      "then": { "required": ["spec", "spec_version", "digests", "hosts_replayed", "overall_result", "source"] }
    },
    "eval": {
      "type": "object",
      "required": ["status"],
      "additionalProperties": false,
      "properties": {
        "status": { "enum": ["run", "not_run"] },
        "results": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["metric", "value", "host", "source"],
            "additionalProperties": false,
            "properties": {
              "metric": { "type": "string", "minLength": 1 },
              "value": { "type": "number" },
              "host": { "type": "string", "minLength": 1 },
              "source": { "$ref": "#/$defs/source" },
              "commit": { "type": "string" },
              "eval_data_sha256": { "$ref": "#/$defs/sha256" }
            }
          }
        }
      },
      "if": { "properties": { "status": { "const": "run" } }, "required": ["status"] },
      "then": { "required": ["results"], "properties": { "results": { "minItems": 1 } } }
    },
    "publish_status": { "type": "string" }
  }
}
```

## 4. Field reference

Required means required by the schema at that level; some fields become
required when a section's `status` is `run`.

| Field | Type | Required | Meaning | Where the value comes from |
|---|---|---|---|---|
| `schema_version` | const `"model-catalog/0.1"` | yes | Schema version; bumped on any breaking change. | This document. |
| `record_id` | string | yes | Unique id of this record. | Assigned by the record author. |
| `model_name` | string | yes | Human label; not authoritative, the hash is. | Model card or release name. |
| `licence` | string | yes | SPDX identifier, `UNLICENSED`, `PROPRIETARY`, or `unknown-needs-decision`. Never blank. | The model's licence file or card. |
| `artifact_ref.weights_sha256` | sha256 hex | yes | SHA-256 of the weights file; canonical identity. | `sha256sum` of the weights file. |
| `artifact_ref.config_sha256`, `tokenizer_sha256` | sha256 hex | no | SHA-256 of the config and tokenizer files. | `sha256sum`; for the pinned CIS-2 artifact, `weights/MANIFEST.sha256`. |
| `artifact_ref.model_id`, `repo`, `commit` | string | no | Upstream model id, or the repository and commit that produced the artifact. | Upstream hub id; `git rev-parse HEAD`. |
| `training.status` | `run` / `not_run` / `third_party` | yes | Whether a training manifest is supplied. | Record author. |
| `training.note` | string | no | Free text. | Record author. |
| `training.seed`, `total_steps` | integer | when `run` | Seed and step count. | Training manifest. |
| `training.model_config`, `param_counts`, `data_sha256` | object / object / sha256 | no | Architecture settings, parameter counts, SHA-256 of the training data file. | Training manifest; `sha256sum` of the data. |
| `training.checkpoints[]` | array of `{step, loss, checkpoint_sha256, host}` | when `run` | One entry per saved checkpoint. `host` is required because the same step and loss can map to different hashes on different hosts. | Training log and `sha256sum` of each checkpoint. |
| `training.hosts[]` | array of `{host, cpu_or_gpu, framework_version, kernel_pin}` | when `run` | Machines the training ran on. `kernel_pin` lists forced-kernel environment settings, or the string `none`. | Run environment. |
| `training.determinism_claim` | object `{level, cross_machine_status, note, source[]}` | when `run` | Scoped claim. `level` is `unverified`, `same-machine-reproducible` or `cross-machine-bit-identical`. `source` lists the reports the claim rests on and must be non-empty. | Repeat-run and cross-host hash comparisons. |
| `inference_conformance.status` | `run` / `not_run` | yes | Whether CIS-2 digests are supplied. | Record author. |
| `inference_conformance.spec`, `spec_version`, `test_vector` | string | when `run` (`test_vector` optional) | Spec document, its version, and the prompt and decode settings used. | `docs/CIS2_SPEC_v0.3b.md`; `EXPECTED_DIGESTS.md`. |
| `inference_conformance.digests` | object `{witness, argmax, table, inv_freq}` | when `run` | The four CIS-2 digests, all required together. `witness` is `CIS2_REF`. | Output of `scripts/self_check.sh` or a conforming implementation. |
| `inference_conformance.hosts_replayed[]` | array of `{host, cpu_or_gpu, result}` | when `run` | Each machine or implementation the digests were reproduced on; `result` is `PASS` or `FAIL`. | Replay logs. |
| `inference_conformance.overall_result` | `MATCH` / `MISMATCH` / `PARTIAL` | when `run` | `MATCH` only if every listed host matched. | Derived from `hosts_replayed`. |
| `inference_conformance.source` | string | when `run` | Report, log or commit. | Replay report. |
| `eval.status` | `run` / `not_run` | yes | Whether eval results are supplied. | Record author. |
| `eval.results[]` | array of `{metric, value, host, source, commit, eval_data_sha256}` | when `run` | `metric`, `value`, `host` and `source` are required; `commit` and `eval_data_sha256` pin the code and corpus. | Eval log; `sha256sum` of the corpus. |
| `publish_status` | string | no | Free text pointer if the model is under a release hold. A record's existence does not imply clearance to publish the model. | Record author. |

## 5. Worked example

The record below describes the pinned CIS-2 test artifact. Every value is
taken from files in this repository: the three artifact hashes from
`weights/MANIFEST.sha256`, the four digests from `EXPECTED_DIGESTS.md`
(spec v0.3b primary vector), and the replay hosts and PASS results from
`docs/GPU_RESULT.md`. Two values are deliberate choices rather than
repository facts: `licence` is `unknown-needs-decision` because this
repository does not state the upstream model's licence, and `eval.status` is
`not_run` because no evaluation result for this model is recorded here.
No value in the example is a synthetic placeholder.

```json
{
  "schema_version": "model-catalog/0.1",
  "record_id": "smollm2-135m-cis2-v0.3b",
  "model_name": "SmolLM2-135M",
  "licence": "unknown-needs-decision",
  "artifact_ref": {
    "model_id": "HuggingFaceTB/SmolLM2-135M",
    "weights_sha256": "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1",
    "config_sha256": "1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843",
    "tokenizer_sha256": "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"
  },
  "training": {
    "status": "third_party",
    "note": "Weights are an upstream release; this record makes no claim about how they were trained."
  },
  "inference_conformance": {
    "status": "run",
    "spec": "docs/CIS2_SPEC_v0.3b.md",
    "spec_version": "v0.3b",
    "test_vector": "prompt \"Once upon a time\", gen_toks=16, fp32 (EXPECTED_DIGESTS.md, primary normative vector)",
    "digests": {
      "witness": "d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df",
      "argmax": "0b9c8f3ac90d0b9cd5f1719ac327dca1fc639fd87468305fccebbe3d56f67aff",
      "table": "23c7bfaf5cef0095fd021af2eb1808abb4928bae4219756d86bdac670a06b35d",
      "inv_freq": "da9f6dcfde0425588815509e874515cdcd3d6b8818b6d0136590052e7bbf6f12"
    },
    "hosts_replayed": [
      {"host": "CPU reference (Rust)", "cpu_or_gpu": "x86_64", "result": "PASS"},
      {"host": "CPU reference (Rust)", "cpu_or_gpu": "aarch64", "result": "PASS"},
      {"host": "CUDA implementation", "cpu_or_gpu": "NVIDIA Tesla P100-PCIE-16GB (sm_60)", "result": "PASS"}
    ],
    "overall_result": "MATCH",
    "source": "docs/GPU_RESULT.md"
  },
  "eval": {
    "status": "not_run"
  }
}
```

### 5.1 Fragment: a trained model (synthetic placeholders)

The example above has no trained-model section. This fragment shows the shape
of `training` and `eval` for a model the author trained. **Every value in it
is synthetic** (all-zero digests, invented names and numbers) and describes no
real model or run.

```json
{
  "training": {
    "status": "run",
    "seed": 0,
    "total_steps": 100,
    "checkpoints": [
      {"step": 50,  "loss": 0.0, "checkpoint_sha256": "0000000000000000000000000000000000000000000000000000000000000000", "host": "example-host-a"},
      {"step": 100, "loss": 0.0, "checkpoint_sha256": "0000000000000000000000000000000000000000000000000000000000000000", "host": "example-host-a"}
    ],
    "hosts": [
      {"host": "example-host-a", "cpu_or_gpu": "example-cpu", "framework_version": "0.0.0", "kernel_pin": "none"}
    ],
    "determinism_claim": {
      "level": "same-machine-reproducible",
      "cross_machine_status": "not_attempted",
      "note": "Repeat runs on example-host-a gave identical checkpoint hashes; no other host was tried.",
      "source": ["example-report.md"]
    }
  },
  "eval": {
    "status": "run",
    "results": [
      {"metric": "example_metric", "value": 0.0, "host": "example-host-a", "source": "example-report.md"}
    ]
  }
}
```

## 6. What the schema does not let a record claim

- No top-level "verified" boolean. Checking is per section and per host.
- No cross-machine training determinism without a recorded multi-host
  comparison (see rule 2).
- No metric without host and source.
- No partial CIS-2 digest set.

## 7. Versioning

`schema_version` is `model-catalog/0.1`. This is a draft: any breaking change
to a field name, type or requirement increments the version, and a record
states the version it was written against. Additive optional fields may be
added within a version. The CIS-2 digests in a record are tied to the spec
version named in `inference_conformance.spec_version`; a record does not
carry over to a later spec version without a new replay.

## 8. Licence

This document and `schema/model-catalog.schema.json` are released under the
repository's licence, Apache-2.0 (see `LICENSE` and `NOTICE`). The licence of
a model described by a record is stated by that record's `licence` field and is
separate from this schema's licence.
