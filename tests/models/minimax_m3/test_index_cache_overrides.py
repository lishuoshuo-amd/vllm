# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 reads IndexCache settings from the text config only.

``--hf-overrides`` applies top-level keys to the outer VL config, so IndexCache
keys there are rejected when the model config is built.
"""

import re

import pytest

from vllm.config import ModelConfig
from vllm.models.minimax_m3.amd.model import _should_skip_index_topk
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.minimax_m3 import MiniMaxM3Config

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="MiniMax-M3 AMD model"
)

# 3 dense layers, then 9 sparse-attention layers. With freq=4 the sparse layers
# 3, 7 and 11 recompute the indexer top-k and the other six reuse it.
NUM_LAYERS = 12
SPARSE_ATTENTION_FREQ = [0] * 3 + [1] * (NUM_LAYERS - 3)
SKIPPED_LAYERS = [4, 5, 6, 8, 9, 10]
INDEX_CACHE = {"use_index_cache": True, "index_topk_freq": 4}
ERROR_PREFIX = "MiniMax-M3 reads IndexCache settings from text_config, so top-level "

TEXT_CONFIG = dict(
    num_hidden_layers=NUM_LAYERS,
    vocab_size=4096,
    moe_layer_freq=SPARSE_ATTENTION_FREQ,
    sparse_attention_config=dict(
        use_sparse_attention=True,
        sparse_index_dim=128,
        sparse_num_index_heads=4,
        sparse_topk_blocks=16,
        sparse_block_size=128,
        sparse_disable_index_value=SPARSE_ATTENTION_FREQ,
        sparse_score_type="max",
        sparse_init_block=0,
        sparse_local_block=1,
        sparse_attention_freq=SPARSE_ATTENTION_FREQ,
    ),
)


@pytest.fixture(scope="module")
def vl_checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("minimax_m3_vl")
    MiniMaxM3Config(
        text_config=TEXT_CONFIG,
        architectures=["MiniMaxM3SparseForConditionalGeneration"],
        torch_dtype="bfloat16",
    ).save_pretrained(path)
    return str(path)


def _model_config(path, hf_overrides):
    return ModelConfig(
        path, hf_overrides=hf_overrides, skip_tokenizer_init=True, dtype="bfloat16"
    )


@pytest.mark.parametrize(
    ("hf_overrides", "expected_error"),
    [
        (
            INDEX_CACHE,
            ERROR_PREFIX + "use_index_cache, index_topk_freq would be ignored. "
            "Pass them as --hf-overrides "
            '\'{"text_config": {"use_index_cache": true, "index_topk_freq": 4}}\'.',
        ),
        (
            {"use_index_cache": True, "index_topk_freq": 2},
            ERROR_PREFIX + "use_index_cache, index_topk_freq would be ignored. "
            "Pass them as --hf-overrides "
            '\'{"text_config": {"use_index_cache": true, "index_topk_freq": 2}}\'.',
        ),
        (
            {"index_skip_topk_offset": 1},
            ERROR_PREFIX + "index_skip_topk_offset would be ignored. "
            "Pass them as --hf-overrides "
            '\'{"text_config": {"index_skip_topk_offset": 1}}\'.',
        ),
        (
            {"use_index_cache": True, "text_config": {"index_topk_freq": 2}},
            ERROR_PREFIX + "use_index_cache would be ignored. "
            "Pass them as --hf-overrides "
            '\'{"text_config": {"use_index_cache": true}}\'.',
        ),
    ],
    ids=["freq4", "freq2", "offset_only", "mixed"],
)
def test_top_level_overrides_are_rejected(vl_checkpoint, hf_overrides, expected_error):
    with pytest.raises(ValueError, match=re.escape(expected_error)):
        _model_config(vl_checkpoint, hf_overrides)


def test_text_config_overrides_take_effect(vl_checkpoint):
    model_config = _model_config(vl_checkpoint, {"text_config": INDEX_CACHE})

    assert [
        layer_id
        for layer_id in range(NUM_LAYERS)
        if _should_skip_index_topk(model_config.hf_text_config, layer_id)
    ] == SKIPPED_LAYERS
