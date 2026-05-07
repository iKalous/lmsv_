#!/usr/bin/env python3
"""
TransformerConfig adapter utilities.

Provides:
- field set discovery
- filter-only conversion
- optional key mapping for known renames

This module is framework-agnostic and can run without MindSpeed/MindFormers
installed by accepting field sets explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set, Tuple


@dataclass
class AdaptResult:
    config: Dict
    dropped_keys: Set[str]
    mapped_keys: Set[str]


# Static field snapshots (fill from test/mf_tsc_fields.txt and test/ms_tsc_fields.txt).
MS_FIELDS_SNAPSHOT: Set[str] = {
    '_cpu_offloading_context', 'account_for_embedding_in_pipeline_split', 'account_for_loss_in_pipeline_split',
    'activation_func_fp8_input_store', 'async_tensor_model_parallel_allreduce', 'attention_backend',
    'autocast_dtype', 'barrier_with_L1_time', 'batch_p2p_comm', 'batch_p2p_sync',
    'clone_scatter_output_in_embedding', 'config_logger_dir', 'cpu_offloading_activations',
    'cpu_offloading_weights', 'cross_entropy_fusion_impl', 'cross_entropy_loss_fusion',
    'cuda_graph_retain_backward_graph', 'cuda_graph_scope', 'cuda_graph_use_single_mempool',
    'cuda_graph_warmup_steps', 'deallocate_pipeline_outputs', 'defer_embedding_wgrad_compute',
    'deterministic_mode', 'disable_parameter_transpose_cache', 'distribute_saved_activations',
    'enable_autocast', 'enable_cuda_graph', 'external_cuda_graph', 'first_last_layers_bf16',
    'flash_decode', 'fp8', 'fp8_amax_compute_algo', 'fp8_amax_history_len', 'fp8_dot_product_attention',
    'fp8_interval', 'fp8_margin', 'fp8_multi_head_attention', 'fp8_param', 'fp8_recipe', 'fp8_wgrad',
    'gradient_accumulation_fusion', 'heterogeneous_block_specs', 'inference_rng_tracker',
    'is_hybrid_model', 'mamba_head_dim', 'mamba_num_groups', 'mamba_state_dim',
    'microbatch_group_size_per_vp_stage', 'moe_extended_tp', 'moe_layer_recompute',
    'moe_router_topk_limited_devices', 'moe_token_dropping', 'moe_use_legacy_grouped_gemm',
    'mrope_section', 'no_sync_func', 'num_layers_at_end_in_bf16', 'num_layers_at_start_in_bf16',
    'num_layers_in_first_pipeline_stage', 'num_layers_in_last_pipeline_stage', 'overlap_p2p_comm',
    'overlap_p2p_comm_warmup_flush', 'perform_initialization', 'pipeline_dtype',
    'pipeline_model_parallel_comm_backend', 'pipeline_model_parallel_split_rank',
    'recompute_granularity', 'recompute_method', 'recompute_modules', 'recompute_num_layers',
    'test_mode', 'timers', 'tp_comm_atomic_ag', 'tp_comm_atomic_rs', 'tp_comm_bootstrap_backend',
    'tp_comm_bulk_dgrad', 'tp_comm_bulk_wgrad', 'tp_comm_overlap', 'tp_comm_overlap_ag',
    'tp_comm_overlap_disable_fc1', 'tp_comm_overlap_disable_qkv', 'tp_comm_overlap_rs',
    'tp_comm_overlap_rs_dgrad', 'tp_comm_split_ag', 'tp_comm_split_rs', 'tp_only_amax_red',
    'use_cpu_initialization', 'use_ring_exchange_p2p', 'use_te_rng_tracker', 'variable_seq_lengths',
    'wgrad_deferral_limit', 'window_size',
}

MF_FIELDS_SNAPSHOT: Set[str] = {
    'add_mlp_fc1_bias_linear', 'add_mlp_fc2_bias_linear', 'attention_next_tokens', 'attention_pre_tokens',
    'attn_allgather', 'attn_allreduce', 'attn_reduce_scatter', 'batch_size', 'bias_swiglu_fusion',
    'block_size', 'callback_moe_droprate', 'comp_comm_parallel', 'comp_comm_parallel_degree',
    'compute_dtype', 'context_parallel_algo', 'data_parallel_size', 'default_prefetch',
    'disable_lazy_inline', 'dispatch_global_max_bs', 'ffn_allgather', 'ffn_allreduce',
    'ffn_reduce_scatter', 'first_k_dense_replace', 'fp16_lm_cross_entropy', 'fused_norm',
    'gradient_aggregation_group', 'group_wise_a2a', 'hidden_act', 'ignore_token_id', 'input_layout',
    'is_dynamic', 'layernorm_compute_dtype', 'mask_func_type', 'max_position_embeddings',
    'micro_batch_num', 'mla_qkv_concat', 'moe_init_method_std', 'moe_router_force_expert_balance',
    'moe_router_fusion', 'mp_comm_recompute', 'norm_topk_prob', 'npu_nums_per_device', 'num_blocks',
    'offset', 'op_swap', 'pad_token_id', 'parallel_config', 'parallel_decoding_params',
    'parallel_optimizer_comm_recompute', 'partial_rotary_factor', 'pet_config', 'position_embedding_type',
    'post_process', 'pre_process', 'print_separate_loss', 'quantization_config', 'recompute',
    'recompute_slice_activation', 'rope_scaling', 'rotary_base', 'rotary_cos_format', 'rotary_dtype',
    'rotary_seq_len_interpolation_factor', 'sandwich_norm', 'select_comm_recompute',
    'select_comm_recompute_exclude', 'select_recompute', 'select_recompute_exclude', 'seq_length',
    'seq_split_num', 'shared_expert_num', 'softmax_compute_dtype', 'sparse_mode', 'tie_word_embeddings',
    'topk_method', 'ulysses_degree_in_cp', 'untie_embeddings_and_output_weights', 'use_alibi_mask',
    'use_alltoall', 'use_attention_mask', 'use_attn_mask_compression',
    'use_contiguous_weight_layout_attention', 'use_eod_attn_mask_compression', 'use_eod_reset',
    'use_flash_attention', 'use_fused_mla', 'use_fused_ops_topkrouter', 'use_interleaved_weight_layout_mlp',
    'use_pad_tokens', 'use_ring_attention', 'use_rope_scaling', 'use_shared_expert_gating',
    'vocab_emb_dp', 'vocab_size',
}

# Extend this when you confirm a semantic rename between frameworks.
DEFAULT_KEY_MAP_MS_TO_MF = {
    # Example:
    # "seq_length": "max_position_embeddings",
}

DEFAULT_KEY_MAP_MF_TO_MS = {
    # Reverse mappings if needed.
}


def _normalize_field_set(fields: Iterable[str]) -> Set[str]:
    return {str(f) for f in fields}


def build_field_sets(
    ms_fields: Optional[Iterable[str]] = None,
    mf_fields: Optional[Iterable[str]] = None,
) -> Tuple[Set[str], Set[str]]:
    """
    Build field sets. If not provided, try to import and introspect.
    """
    if ms_fields is None and MS_FIELDS_SNAPSHOT:
        ms_fields = MS_FIELDS_SNAPSHOT
    if mf_fields is None and MF_FIELDS_SNAPSHOT:
        mf_fields = MF_FIELDS_SNAPSHOT
    if ms_fields is None:
        try:
            from megatron.core.transformer.transformer_config import TransformerConfig as MSConfig
            ms_fields = MSConfig.__dataclass_fields__.keys()
        except Exception:
            ms_fields = []
    if mf_fields is None:
        try:
            from mindformers.parallel_core.transformer_config import TransformerConfig as MFConfig
            mf_fields = MFConfig.__dataclass_fields__.keys()
        except Exception:
            mf_fields = []
    return _normalize_field_set(ms_fields), _normalize_field_set(mf_fields)


def diff_fields(ms_fields: Iterable[str], mf_fields: Iterable[str]) -> Dict[str, Set[str]]:
    ms_set = _normalize_field_set(ms_fields)
    mf_set = _normalize_field_set(mf_fields)
    return {
        "only_ms": ms_set - mf_set,
        "only_mf": mf_set - ms_set,
        "common": ms_set & mf_set,
    }


def adapt_config(
    config: Dict,
    target_fields: Iterable[str],
    key_map: Optional[Dict[str, str]] = None,
    drop_unknown: bool = True,
) -> AdaptResult:
    """
    Filter and map keys to target fields.

    Args:
        config: source config dict
        target_fields: allowed keys in target config
        key_map: optional mapping from source -> target keys
        drop_unknown: if True, drop keys not in target_fields

    Returns:
        AdaptResult with adapted config and metadata.
    """
    key_map = key_map or {}
    target_set = _normalize_field_set(target_fields)

    out = {}
    dropped = set()
    mapped = set()

    for key, value in config.items():
        mapped_key = key_map.get(key, key)
        if mapped_key in target_set:
            out[mapped_key] = value
            if mapped_key != key:
                mapped.add(key)
        else:
            if drop_unknown:
                dropped.add(key)
            else:
                out[mapped_key] = value
                if mapped_key != key:
                    mapped.add(key)

    return AdaptResult(config=out, dropped_keys=dropped, mapped_keys=mapped)


def adapt_ms_to_mf(
    ms_config: Dict,
    ms_fields: Optional[Iterable[str]] = None,
    mf_fields: Optional[Iterable[str]] = None,
    key_map: Optional[Dict[str, str]] = None,
) -> AdaptResult:
    """
    Convert a MindSpeed config dict to MindFormers-compatible config.
    """
    _, mf_set = build_field_sets(ms_fields=ms_fields, mf_fields=mf_fields)
    return adapt_config(
        ms_config,
        target_fields=mf_set,
        key_map=key_map or DEFAULT_KEY_MAP_MS_TO_MF,
        drop_unknown=True,
    )


def adapt_mf_to_ms(
    mf_config: Dict,
    ms_fields: Optional[Iterable[str]] = None,
    mf_fields: Optional[Iterable[str]] = None,
    key_map: Optional[Dict[str, str]] = None,
) -> AdaptResult:
    """
    Convert a MindFormers config dict to MindSpeed-compatible config.
    """
    ms_set, _ = build_field_sets(ms_fields=ms_fields, mf_fields=mf_fields)
    return adapt_config(
        mf_config,
        target_fields=ms_set,
        key_map=key_map or DEFAULT_KEY_MAP_MF_TO_MS,
        drop_unknown=True,
    )


if __name__ == "__main__":
    ms_set, mf_set = build_field_sets()
    diff = diff_fields(ms_set, mf_set)
    print("only_ms", len(diff["only_ms"]))
    print("only_mf", len(diff["only_mf"]))
    print("common", len(diff["common"]))
