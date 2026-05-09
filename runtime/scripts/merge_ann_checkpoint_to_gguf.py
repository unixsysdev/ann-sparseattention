#!/usr/bin/env python3
"""Merge trained ANN search projections into a Qwen3 GGUF.

The output GGUF keeps all base tensors unchanged and appends optional
llama.cpp indexer tensors:

  blk.{layer}.indexer.proj.weight   = W_Qs, shape [d_search, d_model]
  blk.{layer}.indexer.attn_k.weight = W_Ks, shape [d_search, d_model]

GGUF stores dimensions reversed relative to numpy arrays. Passing the PyTorch
Linear weight directly as [d_search, d_model] loads in llama.cpp as the desired
ggml matrix shape {d_model, d_search}.

The patched llama.cpp Qwen3 loader uses these tensors for decode-time learned
exact top-K sparse attention.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


def _add_llama_cpp_gguf_to_path(repo: Path) -> None:
    gguf_py = repo / "gguf-py"
    if not gguf_py.exists():
        raise SystemExit(f"missing llama.cpp gguf-py directory: {gguf_py}")
    sys.path.insert(0, str(gguf_py))


def _load_checkpoint_tensors(path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "search_module" not in ckpt:
        raise SystemExit(f"{path} does not look like a search-projection checkpoint")
    return ckpt["search_module"], ckpt.get("config", {})


def _projection_layers(state: dict[str, torch.Tensor]) -> list[int]:
    layers: set[int] = set()
    for name in state:
        parts = name.split(".")
        if len(parts) == 4 and parts[0] == "projections" and parts[2] in {"W_Qs", "W_Ks"} and parts[3] == "weight":
            layers.add(int(parts[1]))
    return sorted(layers)


def merge(base_gguf: Path, checkpoint: Path, output: Path, llama_cpp_repo: Path, top_k: int, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise SystemExit(f"refusing to overwrite existing output: {output}")

    _add_llama_cpp_gguf_to_path(llama_cpp_repo)
    import gguf  # type: ignore
    from gguf import GGUFReader, GGUFValueType, GGUFWriter  # type: ignore

    state, cfg = _load_checkpoint_tensors(checkpoint)
    layers = _projection_layers(state)
    if not layers:
        raise SystemExit(f"no projection tensors found in {checkpoint}")

    sample = state[f"projections.{layers[0]}.W_Qs.weight"]
    d_search, d_model = sample.shape

    reader = GGUFReader(str(base_gguf))
    arch_field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    if arch_field is None:
        raise SystemExit(f"{base_gguf} has no general.architecture field")
    arch = arch_field.contents()

    writer = GGUFWriter(str(output), arch=arch, endianess=reader.endianess)

    alignment_field = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if alignment_field is not None and alignment_field.contents() is not None:
        writer.data_alignment = alignment_field.contents()

    # Copy metadata, excluding fields written by the writer itself. Existing
    # indexer metadata, if any, is replaced by the checkpoint dimensions below.
    skip_keys = {
        gguf.Keys.General.ARCHITECTURE,
        f"{arch}.attention.indexer.head_count",
        f"{arch}.attention.indexer.key_length",
        f"{arch}.attention.indexer.top_k",
    }
    for field in reader.fields.values():
        if field.name in skip_keys or field.name.startswith("GGUF."):
            continue
        value = field.contents()
        if value is None:
            continue
        value_type = field.types[0]
        sub_type = field.types[-1] if value_type == GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, value, value_type, sub_type=sub_type)

    writer.add_indexer_head_count(1)
    writer.add_indexer_key_length(int(d_search))
    writer.add_indexer_top_k(int(top_k))
    writer.add_array("ann_sparse.layers", layers)
    writer.add_string("ann_sparse.checkpoint", str(checkpoint))
    writer.add_string("ann_sparse.config", repr(cfg))

    for tensor in reader.tensors:
        writer.add_tensor(
            tensor.name,
            tensor.data,
            raw_shape=tensor.data.shape,
            raw_dtype=tensor.tensor_type,
            tensor_endianess=reader.endianess,
        )

    for layer in layers:
        wq = state[f"projections.{layer}.W_Qs.weight"].float().numpy().astype(np.float16, copy=False)
        wk = state[f"projections.{layer}.W_Ks.weight"].float().numpy().astype(np.float16, copy=False)
        if wq.shape != (d_search, d_model) or wk.shape != (d_search, d_model):
            raise SystemExit(f"unexpected projection shape at layer {layer}: {wq.shape}, {wk.shape}")
        writer.add_tensor(f"blk.{layer}.indexer.proj.weight", wq)
        writer.add_tensor(f"blk.{layer}.indexer.attn_k.weight", wk)

    output.parent.mkdir(parents=True, exist_ok=True)
    writer.open_output_file(output)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    print(f"wrote {output}")
    print(f"layers={layers}")
    print(f"d_model={d_model} d_search={d_search} top_k={top_k}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-gguf", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llama-cpp-repo", type=Path, default=Path("runtime/llama.cpp-ann"))
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    merge(args.base_gguf, args.checkpoint, args.output, args.llama_cpp_repo, args.top_k, args.overwrite)


if __name__ == "__main__":
    main()
