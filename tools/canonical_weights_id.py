#!/usr/bin/env python3
"""canonical_weights_id.py — canonical identity digest for a safetensors file.

Computes a SHA-256 digest of a safetensors weights file that is INDEPENDENT of:
  - the order in which tensors are stored in the file's JSON header
    (safetensors headers are JSON objects; key order is not semantically
    meaningful and two files holding the exact same tensors can differ only
    in header key order)
  - single-file vs. sharded storage, as long as you compute this digest per
    logical tensor set (this tool operates on a single .safetensors file;
    for a sharded checkpoint, feed each shard's tensors through the same
    canonicalization and combine, or concatenate shards' tensor dicts before
    hashing — see NOTE below).

It is NOT independent of: dtype, shape, or raw byte content (bit-for-bit).
Any change to weight values, dtype, or shape will change the digest.

Format assumed (safetensors on-disk layout, no external deps required):
    bytes[0:8]   little-endian uint64 N = header length in bytes
    bytes[8:8+N] UTF-8 JSON header:
        {
          "<tensor_name>": {"dtype": "...", "shape": [...], "data_offsets": [start, end]},
          ...,
          "__metadata__": {...}   # optional, ignored by this tool
        }
    bytes[8+N:]  raw tensor data section; each tensor's bytes are the slice
                 data_offsets[0]:data_offsets[1] of this section.

Canonicalization algorithm:
    1. Parse the header, drop any "__metadata__" key (file-level metadata is
       not part of tensor identity).
    2. Sort tensor names lexicographically (byte-wise, i.e. Python's default
       str sort on the UTF-8 decoded name).
    3. For each tensor, in sorted order, feed the following bytes into a
       running SHA-256 hash, each field length-prefixed with a 4-byte
       little-endian uint32 count to avoid ambiguity from separator
       collisions:
           len(name_utf8)        (uint32 LE) + name_utf8
           len(dtype_utf8)       (uint32 LE) + dtype_utf8
           len(shape)            (uint32 LE) + shape, each dim as uint64 LE
           len(raw_bytes)        (uint64 LE) + raw_bytes
    4. The final hex digest is the canonical weights id.

This exact byte encoding is deterministic and reproducible in any language;
no dependency on Python dict ordering, JSON library, or hash seed is used.

NOTE on sharding: to make a multi-shard checkpoint's id match a single-file
repackaging of the same tensors, canonicalize by tensor identity (name,
dtype, shape, raw bytes) and sort across the UNION of tensors from all
shards before hashing — i.e. treat all shards as one flat namespace. This
tool implements the single-file case; extending to multiple files is a
matter of merging their (name -> record) maps before running step 2 above.

Usage:
    python3 tools/canonical_weights_id.py <path-to-safetensors-file>

Prints the hex digest to stdout, followed by a newline. No other deps.
"""
import hashlib
import json
import struct
import sys


def parse_safetensors_header(path):
    with open(path, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) != 8:
            raise ValueError(f"{path}: truncated file, could not read 8-byte header length")
        (header_len,) = struct.unpack("<Q", header_len_bytes)
        header_json = f.read(header_len)
        if len(header_json) != header_len:
            raise ValueError(f"{path}: truncated file, header shorter than declared length")
        header = json.loads(header_json.decode("utf-8"))
        data_start = 8 + header_len
        return header, data_start


def canonical_weights_id(path):
    header, data_start = parse_safetensors_header(path)
    header.pop("__metadata__", None)

    names = sorted(header.keys())

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for name in names:
            info = header[name]
            dtype = info["dtype"]
            shape = info["shape"]
            start, end = info["data_offsets"]

            f.seek(data_start + start)
            raw = f.read(end - start)
            if len(raw) != end - start:
                raise ValueError(f"{path}: truncated tensor data for {name!r}")

            name_b = name.encode("utf-8")
            dtype_b = dtype.encode("utf-8")

            h.update(struct.pack("<I", len(name_b)))
            h.update(name_b)

            h.update(struct.pack("<I", len(dtype_b)))
            h.update(dtype_b)

            h.update(struct.pack("<I", len(shape)))
            for dim in shape:
                h.update(struct.pack("<Q", dim))

            h.update(struct.pack("<Q", len(raw)))
            h.update(raw)

    return h.hexdigest()


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        print("error: expected exactly one argument, a path to a .safetensors file", file=sys.stderr)
        return 2
    path = argv[1]
    try:
        digest = canonical_weights_id(path)
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
