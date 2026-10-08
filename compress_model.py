#!/usr/bin/env python3
"""Repack a browser model, with lossless compression and optional SH quantization.

No checkpoint, model rebuild or Gaussian filtering is involved. The default
preserves the SH dtype and bits; --sh-dtype float16 explicitly quantizes SH.
Geometry and BVH bytes remain unchanged. Camera/identity metadata and
alignment.json are retained. Expanding float16 to float32 cannot recover detail
lost by the previous half-precision conversion.
Existing output paths are never overwritten.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.parse import urlsplit

from export_model import (COMPRESSION_MODES, MAX_PART_BYTES, SCHEMA, SH_DTYPES,
                          convert_sh_bytes, decode_binary_part, set_sh_precision_metadata,
                          write_binary_part)


def _asset_path(root, name):
    if not isinstance(name, str) or not name or any(char in name for char in "\\\x00%?#"):
        raise ValueError("Part paths must be relative local filenames")
    url = urlsplit(name)
    path = Path(name)
    if url.scheme or url.netloc or path.is_absolute() or any(part in ("", ".", "..") for part in name.split("/")):
        raise ValueError("Part paths must remain inside the input model directory")
    resolved = (root / path).resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("Part paths must resolve to files inside the input model directory")
    return resolved


def _decoded_chunks(root, spec, chunk_bytes):
    total = spec.get("byteLength")
    stride = spec.get("stride")
    parts = spec.get("parts")
    if type(total) is not int or total < 1 or type(stride) is not int or stride < 1 or stride % 4 or total % stride:
        raise ValueError("Invalid buffer byteLength/stride")
    if not isinstance(parts, list) or not parts:
        raise ValueError("Buffer must contain at least one part")
    if not stride <= chunk_bytes < MAX_PART_BYTES:
        raise ValueError("Chunk size must hold one record and remain below 100 MiB")
    limit = chunk_bytes // stride * stride
    pending = bytearray()
    consumed = 0
    for part in parts:
        if not isinstance(part, dict) or type(part.get("byteLength")) is not int or part["byteLength"] % stride:
            raise ValueError("Each input part must contain whole records")
        path = _asset_path(root, part.get("path"))
        declared_download = part.get("downloadByteLength", part["byteLength"])
        if type(declared_download) is not int or not 0 < declared_download < MAX_PART_BYTES or path.stat().st_size != declared_download:
            raise ValueError(f"Encoded file size does not match its manifest: {path.name}")
        decoded = decode_binary_part(path.read_bytes(), part)
        consumed += len(decoded)
        if consumed > total:
            raise ValueError("Input parts exceed the declared buffer byteLength")
        pending.extend(decoded)
        while len(pending) >= limit:
            yield bytes(pending[:limit])
            del pending[:limit]
    if consumed != total:
        raise ValueError("Input parts do not fill the declared buffer byteLength")
    if pending:
        yield bytes(pending)


def compress_model(source, output, *, compression="gzip-shuffle", chunk_bytes=64 * 1024 * 1024,
                   sh_dtype="preserve"):
    source = Path(source).expanduser().resolve(strict=True)
    output = Path(output).expanduser()
    if not source.is_dir():
        raise ValueError("Input must be an exported model directory")
    if os.path.lexists(output):
        raise FileExistsError(f"Output already exists: {output}; use a new directory")
    output = output.resolve()
    if output.is_relative_to(source):
        raise ValueError("Output must be outside the original model directory")
    if compression not in COMPRESSION_MODES:
        raise ValueError(f"Unsupported compression: {compression}")
    if sh_dtype not in ("preserve", *SH_DTYPES):
        raise ValueError("SH dtype must be preserve, float16 or float32")
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or set(manifest.get("files", {})) != {"geometry", "sh", "bvh"}:
        raise ValueError("Input must be a FullCircle reference WebGPU model manifest")
    source_sh = manifest["files"]["sh"]
    if (manifest.get("byte_order", "little-endian") != "little-endian"
            or source_sh.get("byte_order", "little-endian") != "little-endian"):
        raise ValueError("Model and SH buffers must use little-endian byte order")
    source_dtype = source_sh.get("dtype", "float32")
    if source_dtype not in SH_DTYPES:
        raise ValueError("Input SH dtype must be float16 or float32 (missing means legacy float32)")
    count = manifest.get("gaussian_count")
    source_stride = 48 * SH_DTYPES[source_dtype].itemsize
    if (type(count) is not int or count < 1 or source_sh.get("stride") != source_stride
            or source_sh.get("byteLength") != count * source_stride):
        raise ValueError("SH dtype, record stride, count and decoded byteLength do not agree")
    target_dtype = source_dtype if sh_dtype == "preserve" else sh_dtype
    target_stride = 48 * SH_DTYPES[target_dtype].itemsize
    if chunk_bytes < target_stride:
        raise ValueError("Chunk size must hold one output SH record")
    result = copy.deepcopy(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".model-compression-", dir=output.parent) as temporary:
        stage = Path(temporary)
        for name, spec in manifest["files"].items():
            if name == "sh":
                # Half->float32 doubles decoded bytes. Bound input chunk size
                # so the expanded output still respects the upload budget.
                input_budget = max(source_stride, min(chunk_bytes, (chunk_bytes // target_stride) * source_stride))
                encoded_parts = [write_binary_part(stage, name, index,
                                                  convert_sh_bytes(raw, source_dtype, target_dtype), compression,
                                                  shuffle_bytes=SH_DTYPES[target_dtype].itemsize)
                                 for index, raw in enumerate(_decoded_chunks(source, spec, input_budget))]
            else:
                encoded_parts = [write_binary_part(stage, name, index, raw, compression)
                                 for index, raw in enumerate(_decoded_chunks(source, spec, chunk_bytes))]
            # Preserve buffer-level extension metadata while updating only the
            # transport description. Render/camera/alignment identities stay put.
            result["files"][name] = {**copy.deepcopy(spec), "parts": encoded_parts,
                                     "byteLength": sum(part["byteLength"] for part in encoded_parts),
                                     "stride": target_stride if name == "sh" else spec["stride"],
                                     "downloadByteLength": sum(part["downloadByteLength"] for part in encoded_parts)}
        set_sh_precision_metadata(result, target_dtype, source_sh)
        for sidecar in ("alignment.json", "cameras.json"):
            path = source / sidecar
            if os.path.lexists(path):
                validated = _asset_path(source, sidecar)
                (stage / sidecar).write_bytes(validated.read_bytes())
        (stage / "manifest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        # Claim the output name exclusively, including against a directory
        # created concurrently after preflight. Publish its manifest last.
        output.mkdir()
        for path in sorted(stage.iterdir(), key=lambda path: path.name):
            if path.name != "manifest.json":
                shutil.move(str(path), str(output / path.name))
        shutil.move(str(stage / "manifest.json"), str(output / "manifest.json"))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Existing exported model directory")
    parser.add_argument("--output", type=Path, required=True, help="New model directory; never overwrite the original")
    parser.add_argument("--compression", choices=COMPRESSION_MODES, default="gzip-shuffle")
    parser.add_argument("--sh-dtype", choices=("preserve", *SH_DTYPES), default="preserve",
                        help="Default preserves SH storage; float16 opts into quantization; float32 cannot undo prior rounding")
    parser.add_argument("--chunk-mib", type=float, default=64)
    args = parser.parse_args(argv)
    if not 0 < args.chunk_mib < 100:
        parser.error("chunk-mib must be greater than 0 and below 100")
    manifest = compress_model(args.input, args.output, compression=args.compression,
                              chunk_bytes=int(args.chunk_mib * 1024 * 1024), sh_dtype=args.sh_dtype)
    source_manifest = json.loads((args.input / "manifest.json").read_text())
    source_decoded = sum(spec["byteLength"] for spec in source_manifest["files"].values())
    decoded = sum(spec["byteLength"] for spec in manifest["files"].values())
    downloaded = sum(spec["downloadByteLength"] for spec in manifest["files"].values())
    print(json.dumps({"manifest": str(args.output / "manifest.json"), "compression_requested": args.compression,
                      "sh_dtype": manifest["files"]["sh"]["dtype"],
                      "decoded_bytes": decoded, "stored_bytes": downloaded, "compression_ratio": decoded / downloaded,
                      "saved_percent": 100 * (1 - downloaded / decoded),
                      "compression_saved_percent": 100 * (1 - downloaded / decoded),
                      "source_decoded_bytes": source_decoded,
                      "total_saved_percent": 100 * (1 - downloaded / source_decoded)}, indent=2))


if __name__ == "__main__":
    main()
