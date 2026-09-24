# cis2-mcp

A tiny stdio [MCP](https://modelcontextprotocol.io) server that exposes CIS-2's
existing `verify3/` clean-room verifier as MCP tools, so any AI agent (or
agent framework) can check a model/receipt before trusting it — "no receipt,
no action."

## Tools

- **`verify_receipt(path: str) -> dict`** — `path` is a directory containing
  `model.safetensors`, `config.json`, and `tokenizer.json`. Builds
  `verify3/cis2_verify3` if the binary isn't already built, runs it, and
  compares every digest field (`weights_sha256`, `config_sha256`,
  `tokenizer_sha256`, `table_digest`, `inv_freq_table_digest`,
  `argmax_digest`, `generated_token_ids`, `CIS2_REF`) against the pinned
  normative §13.1 SmolLM2-135M v0.3b vector in `EXPECTED_DIGESTS.md`. Only
  that exact pinned model/prompt/`gen_toks` combination will PASS — any other
  model directory correctly FAILs (this checks conformance to the pinned
  test vector, not general model validity).

- **`selfcheck() -> dict`** — the MCP-tool equivalent of
  `scripts/self_check.sh`: fetches the pinned SmolLM2-135M weights into
  `<repo>/weights/` if missing (sha256-verified), builds `verify3/`, runs the
  FMA-family-instruction gate (spec §1.4, via `objdump`), runs
  `cis2_verify3`, and returns the same structured PASS/FAIL comparison.

Both tools return a dict with an overall `"verdict"` (`"PASS"`/`"FAIL"`) and
a per-field breakdown (`expected`, `got`, `pass`). **No timing numbers are
ever produced or printed**, matching repo policy — verified by grepping
`scripts/self_check.sh`, which never prints timings either.

The comparison logic (field extraction regexes, pinned expected values,
exact-string-equality semantics) mirrors `scripts/self_check.sh` exactly;
it does not invent new comparison semantics.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and network access (to install
the `mcp` SDK and, for `selfcheck`, to fetch the pinned weights from
Hugging Face on first run).

```sh
cd tools/mcp
uv sync
```

This installs the `mcp` Python SDK (pinned to `mcp<2`; see "SDK version"
below) into a local `.venv/`.

## Running

As a stdio server (for an MCP client to launch directly):

```sh
uv run server.py
```

### Inspecting interactively

With the [MCP Inspector](https://github.com/modelcontextprotocol/inspector):

```sh
npx @modelcontextprotocol/inspector uv run server.py
```

(Requires Node/npx and network access to fetch the inspector package.)

### Scripted test (no Node required)

`test_client.py` launches `server.py` over stdio, calls `selfcheck()`, and
asserts an overall `PASS` verdict:

```sh
uv run test_client.py
```

## SDK version note

At the time this was built, `uv add mcp` installed `mcp==2.2.0` by default,
which renamed `mcp.server.fastmcp.FastMCP` to `mcp.server.mcpserver.MCPServer`
(breaking `from mcp.server.fastmcp import FastMCP`). This server targets the
more widely-documented v1 `FastMCP` API, so `pyproject.toml` pins
`mcp<2` (currently resolves to `1.30.0`). If upgrading to `mcp>=2`, port
`server.py`'s `from mcp.server.fastmcp import FastMCP` /
`@mcp.tool()` / `mcp.run(transport="stdio")` calls to the v2 `MCPServer` API
first.

## Files

- `server.py` — the MCP server (two tools, stdio transport).
- `test_client.py` — scripted stdio client exercising both tools.
- `pyproject.toml` / `uv.lock` — `uv` project files (`mcp<2` dependency).
