from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compare_retrieval import config_from_checkpoint  # noqa: E402
from config import (  # noqa: E402
    make_pilot_d64_packed_config,
    make_pilot_d128_packed_config,
    make_pilot_d256_packed_config,
)
from data import LongContextPackedDataset  # noqa: E402
from k_sweep import config_from_checkpoint as k_sweep_config_from_checkpoint  # noqa: E402


def test_packed_ablation_configs_are_dense_and_comparable():
    configs = [
        make_pilot_d64_packed_config(),
        make_pilot_d128_packed_config(),
        make_pilot_d256_packed_config(),
    ]
    assert [cfg.d_search for cfg in configs] == [64, 128, 256]
    assert all(cfg.sequence_packing for cfg in configs)
    assert {cfg.total_steps for cfg in configs} == {1000}
    assert {cfg.eval_every for cfg in configs} == {250}
    assert {cfg.eval_num_batches for cfg in configs} == {16}


def test_packed_dataset_emits_all_real_tokens_without_padding():
    class ToyTokenizer:
        eos_token_id = 0
        pad_token_id = 0

        def __call__(self, text, add_special_tokens=False, return_attention_mask=False):
            return {"input_ids": [int(x) for x in text.split()]}

    ds = LongContextPackedDataset.__new__(LongContextPackedDataset)
    ds.tokenizer = ToyTokenizer()
    ds.seq_len = 8
    ds.text_field = "text"
    ds.pack = True
    ds.eos_between = False
    ds.eos_id = 0
    ds.dataset = [{"text": "1 2 3"}, {"text": "4 5 6 7 8"}, {"text": "9 10 11"}]

    item = next(iter(ds))
    assert item["input_ids"].shape == (8,)
    assert item["attention_mask"].sum().item() == 8
    assert torch.equal(item["attention_mask"], torch.ones(8, dtype=torch.long))


def test_compare_retrieval_uses_checkpoint_d_search():
    ckpt = {
        "config": {
            "d_search": 256,
            "sequence_packing": True,
            "checkpoint_dir": "/tmp/checkpoints_packed_d256",
        }
    }
    cfg = config_from_checkpoint(ckpt)
    assert cfg.d_search == 256
    assert cfg.sequence_packing is True
    assert cfg.checkpoint_dir == "/tmp/checkpoints_packed_d256"

    k_cfg = k_sweep_config_from_checkpoint(ckpt)
    assert k_cfg.d_search == 256
    assert k_cfg.sequence_packing is True
    assert k_cfg.checkpoint_dir == "/tmp/checkpoints_packed_d256"
