# Runtime Model Files

The llama.cpp runtime was tested with these local GGUF files:

- `Qwen3-4B-Instruct-2507-F16.gguf`
- `Qwen3-4B-Instruct-2507-F16-ann-6layer-k128-v2.gguf`
- `Qwen3-4B-Instruct-2507-F16-ann-all32-k128-v2.gguf`
- `Qwen3-4B-Instruct-2507-F16-ann-all36-k128-v2.gguf`

The merged ANN runtime GGUFs are hosted on Hugging Face:

| variant | hosted file |
|---|---|
| 6-layer pilot | [`Qwen3-4B-Instruct-2507-F16-ann-6layer-k128-v2.gguf`](https://huggingface.co/datasysdev/ann-sparseattention/blob/main/gguf/Qwen3-4B-Instruct-2507-F16-ann-6layer-k128-v2.gguf) |
| all32 reserved-edge | [`Qwen3-4B-Instruct-2507-F16-ann-all32-k128-v2.gguf`](https://huggingface.co/datasysdev/ann-sparseattention/blob/main/gguf/Qwen3-4B-Instruct-2507-F16-ann-all32-k128-v2.gguf) |
| all36 full substitution | [`Qwen3-4B-Instruct-2507-F16-ann-all36-k128-v2.gguf`](https://huggingface.co/datasysdev/ann-sparseattention/blob/main/gguf/Qwen3-4B-Instruct-2507-F16-ann-all36-k128-v2.gguf) |

The base `Qwen3-4B-Instruct-2507-F16.gguf` is the original full-attention model
used for comparison and is not mirrored here. Use a local upstream/base GGUF for
base runs, and use the hosted ANN GGUFs above for the sparse-runtime variants.

Do not use the old `Qwen3-4B-Instruct-2507-F16-ann-all32-k128-bad-shape.gguf`; it was a first merge with transposed projection tensor dimensions and is kept only locally for traceability.

The checkpoint-to-GGUF merge script is:

```bash
runtime/scripts/merge_ann_checkpoint_to_gguf.py
```

Example:

```bash
python3 runtime/scripts/merge_ann_checkpoint_to_gguf.py \
  --base-gguf runtime/models/Qwen3-4B-Instruct-2507-F16.gguf \
  --checkpoint runtime/checkpoints/checkpoints_all32_d128_block_reserve_0_1_2_35/search_step_1000.pt \
  --output runtime/models/Qwen3-4B-Instruct-2507-F16-ann-all32-k128-v2.gguf \
  --top-k 128
```
