# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 reads IndexCache settings from the text config only.

``--hf-overrides`` is applied to the outer VL config, so the IndexCache keys
must be nested under ``text_config`` to take effect.
"""

import pytest

from vllm.config import ModelConfig
from vllm.models.minimax_m3.amd.model import (
    _should_skip_index_topk,
    _warn_index_cache_keys_outside_text_config,
)
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
WARNING_PREFIX = (
    "MiniMax-M3 reads IndexCache settings from text_config; ignoring top-level "
)
MODEL_LOGGER = "vllm.models.minimax_m3.amd.model"

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


def _skipped_layers(model_config):
    return [
        layer_id
        for layer_id in range(NUM_LAYERS)
        if _should_skip_index_topk(model_config.hf_text_config, layer_id)
    ]


def _warnings(caplog_vllm):
    return [
        record.getMessage()
        for record in caplog_vllm.records
        if record.name == MODEL_LOGGER and record.levelname == "WARNING"
    ]


@pytest.mark.parametrize(
    ("hf_overrides", "expected_warning"),
    [
        (
            INDEX_CACHE,
            WARNING_PREFIX + "use_index_cache, index_topk_freq. Pass them as "
            "--hf-overrides "
            '\'{"text_config": {"use_index_cache": true, "index_topk_freq": 4}}\'.',
        ),
        (
            {"use_index_cache": True, "index_topk_freq": 2},
            WARNING_PREFIX + "use_index_cache, index_topk_freq. Pass them as "
            "--hf-overrides "
            '\'{"text_config": {"use_index_cache": true, "index_topk_freq": 2}}\'.',
        ),
        (
            {"index_skip_topk_offset": 1},
            WARNING_PREFIX + "index_skip_topk_offset. Pass them as --hf-overrides "
            '\'{"text_config": {"index_skip_topk_offset": 1}}\'.',
        ),
        (
            {"use_index_cache": True, "text_config": {"index_topk_freq": 2}},
            WARNING_PREFIX + "use_index_cache. Pass them as --hf-overrides "
            '\'{"text_config": {"use_index_cache": true}}\'.',
        ),
    ],
    ids=["freq4", "freq2", "offset_only", "mixed"],
)
def test_top_level_overrides_are_ignored_and_warned(
    vl_checkpoint, caplog_vllm, hf_overrides, expected_warning
):
    model_config = _model_config(vl_checkpoint, hf_overrides)

    _warn_index_cache_keys_outside_text_config(model_config)

    assert _skipped_layers(model_config) == []
    assert _warnings(caplog_vllm) == [expected_warning]


def test_text_config_overrides_take_effect(vl_checkpoint, caplog_vllm):
    model_config = _model_config(vl_checkpoint, {"text_config": INDEX_CACHE})

    _warn_index_cache_keys_outside_text_config(model_config)

    assert _skipped_layers(model_config) == SKIPPED_LAYERS
    assert _warnings(caplog_vllm) == []
