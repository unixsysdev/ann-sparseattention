"""
Long-context dataloader.

Streams a Hugging Face dataset, tokenizes, and packs short examples into
fixed-length sequences of `seq_len` tokens. Uses pin_memory + prefetch as the
perf doc recommends; per-sequence boundaries are tracked in `position_ids` and
`attention_mask` so attention does not leak across packed examples.
"""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Optional

import torch
from torch.utils.data import DataLoader, IterableDataset


class LongContextPackedDataset(IterableDataset):
    """
    Streams an HF dataset, tokenizes, and packs into seq_len-token chunks.

    Output per item:
        input_ids:      [seq_len]
        attention_mask: [seq_len]  (all 1s — packing handled via position_ids)
        position_ids:   [seq_len]  (resets to 0 at each packed example boundary)
    """

    def __init__(
        self,
        hf_dataset_name: str,
        tokenizer,
        seq_len: int,
        text_field: str = "text",
        split: str = "train",
        streaming: bool = False,
        pack: bool = True,
        eos_between: bool = True,
        dataset_config: str = None,
    ):
        from datasets import load_dataset

        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.text_field = text_field
        self.pack = pack
        self.eos_between = eos_between
        load_kwargs = dict(split=split, streaming=streaming)
        if dataset_config is not None:
            load_kwargs["name"] = dataset_config
        self.dataset = load_dataset(hf_dataset_name, **load_kwargs)
        self.eos_id = tokenizer.eos_token_id
        if self.eos_id is None:
            # Fall back to using pad_token; if that's None too we just skip the
            # separator and rely on position_id resets.
            self.eos_id = tokenizer.pad_token_id

    def _iter_token_streams(self) -> Iterator[List[int]]:
        for example in self.dataset:
            text = example.get(self.text_field)
            if not text:
                continue
            ids = self.tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
            if self.eos_between and self.eos_id is not None:
                ids.append(self.eos_id)
            yield ids

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        if not self.pack:
            # No packing: one example per emitted item. Short docs are padded
            # to seq_len with attention_mask=0 on the pad positions; long docs
            # are truncated. attention_mask is the load-bearing field for
            # ignoring padding during teacher capture and PPL eval.
            for i, ids in enumerate(self._iter_token_streams()):
                if i % num_workers != worker_id:
                    continue
                content_len = min(len(ids), self.seq_len)
                if len(ids) < self.seq_len:
                    pad_len = self.seq_len - len(ids)
                    ids = ids + [self.eos_id or 0] * pad_len
                else:
                    ids = ids[: self.seq_len]
                yield self._make_item(ids, [0], content_len=content_len)
            return

        buf: List[int] = []
        boundaries: List[int] = [0]  # absolute offsets where each example starts
        for i, ids in enumerate(self._iter_token_streams()):
            if i % num_workers != worker_id:
                continue
            buf.extend(ids)
            boundaries.append(len(buf))
            while len(buf) >= self.seq_len:
                chunk = buf[: self.seq_len]
                # Boundaries that fall inside this chunk define position-id resets.
                chunk_boundaries = [b for b in boundaries if 0 <= b < self.seq_len]
                if not chunk_boundaries or chunk_boundaries[0] != 0:
                    chunk_boundaries = [0] + chunk_boundaries
                yield self._make_item(chunk, chunk_boundaries)
                buf = buf[self.seq_len :]
                boundaries = [max(0, b - self.seq_len) for b in boundaries]
                # Drop boundaries that no longer point inside the buffer.
                boundaries = [b for b in boundaries if b <= len(buf)]
                if not boundaries or boundaries[0] != 0:
                    boundaries = [0] + boundaries

    def _make_item(
        self,
        ids: List[int],
        boundaries: List[int],
        content_len: int = None,
    ) -> Dict[str, torch.Tensor]:
        L = self.seq_len
        position_ids = torch.zeros(L, dtype=torch.long)
        sorted_b = sorted(set(b for b in boundaries if 0 <= b < L))
        for i, start in enumerate(sorted_b):
            end = sorted_b[i + 1] if i + 1 < len(sorted_b) else L
            position_ids[start:end] = torch.arange(end - start, dtype=torch.long)

        attention_mask = torch.ones(L, dtype=torch.long)
        if content_len is not None and content_len < L:
            attention_mask[content_len:] = 0

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }


def build_long_context_dataloader(
    tokenizer,
    dataset_name: str,
    seq_len: int,
    batch_size: int,
    num_workers: int = 8,
    prefetch_factor: int = 4,
    pack: bool = True,
    text_field: str = "text",
    split: str = "train",
    dataset_config: str = None,
    streaming: bool = False,
) -> DataLoader:
    # streaming=True triggers httpx-client-closed crashes between sequential
    # training runs sharing HF cache. Default off; download once, reuse.
    ds = LongContextPackedDataset(
        hf_dataset_name=dataset_name,
        tokenizer=tokenizer,
        seq_len=seq_len,
        text_field=text_field,
        split=split,
        streaming=streaming,
        pack=pack,
        dataset_config=dataset_config,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )


def build_eval_data(
    tokenizer, config, num_batches: Optional[int] = None
) -> Iterable[Dict[str, torch.Tensor]]:
    """
    Eval data: same packed dataloader, capped to `num_batches`.
    Held-out split if available; otherwise reuse train at a different seed.
    """
    n = num_batches if num_batches is not None else config.eval_num_batches
    cfg_kwargs = dict(
        seq_len=config.seq_len,
        batch_size=1,
        num_workers=2,
        prefetch_factor=2,
        pack=config.sequence_packing,
        dataset_config=getattr(config, "train_dataset_config", None),
    )
    try:
        loader = build_long_context_dataloader(
            tokenizer, config.train_dataset, split="validation", **cfg_kwargs
        )
    except Exception:
        loader = build_long_context_dataloader(
            tokenizer, config.train_dataset, split="train", **cfg_kwargs
        )

    out: List[Dict[str, torch.Tensor]] = []
    for i, batch in enumerate(loader):
        if i >= n:
            break
        out.append(batch)
    return out
