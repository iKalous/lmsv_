# Copyright (c) 2023, HUAWEI CORPORATION.  All rights reserved.
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
from functools import partial
from typing import Union

import torch
from mindspeed_llm import megatron_adaptor
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.rerun_state_machine import get_rerun_state_machine
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from msm_replace.new_training import pretrain
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    average_losses_across_data_parallel_group
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
try:
    from mindspeed_llm.training.utils import generate_actual_seq_len, set_mtp_batch_list, get_mtp_batch_list
except ImportError:
    # Compatibility fallback for older/newer MindSpeed-LLM versions.
    def generate_actual_seq_len(batch, actual_seq_len):
        if isinstance(batch, dict):
            batch["actual_seq_len"] = actual_seq_len
        return batch

    def set_mtp_batch_list(_mtp_batch_list):
        return None

    def get_mtp_batch_list():
        return None

try:
    from mindspeed_llm.core.transformer.multi_token_prediction import generate_mtp_batch_list_on_this_tp_rank
except ImportError:
    def generate_mtp_batch_list_on_this_tp_rank(_batch):
        return None

#工具依赖
import csv

from utils.runtime.constants import LOG_DIR, get_log_round
from utils.runtime.debug_utils import (
    begin_debug_step,
    debug_message,
    debug_parameter_summary,
    debug_scalar,
    debug_tensor_summary,
    is_rank0,
    mark_weights_logged,
    should_log_full,
    should_log_heavy,
    should_log_weights_once,
)

# 使用外部传入的变异轮次作为日志编号
next_num = get_log_round()

csv_path = f"{LOG_DIR}/training_log-{next_num}.csv"

# SwanLab (optional)
SWANLAB_ENABLE = os.getenv("SWANLAB_ENABLE", "0").lower() in ("1", "true", "yes")
SWANLAB_API_KEY = os.getenv("SWANLAB_API_KEY", "") or os.getenv("SWANLAB_TOKEN", "")
SWANLAB_PROJECT = os.getenv("SWANLAB_PROJECT", "lmsv")
SWANLAB_RUN_PREFIX = os.getenv("SWANLAB_RUN_PREFIX", "lmsv")
_mode = "msa" if "training_log_msa" in LOG_DIR else "pta"
SWANLAB_RUN_NAME = f"{SWANLAB_RUN_PREFIX}_{_mode}_{next_num}"
os.environ["SWANLAB_RUN_NAME"] = SWANLAB_RUN_NAME

_swanlab_run = None

def init_swanlab():
    global _swanlab_run
    if not SWANLAB_ENABLE:
        return None
    try:
        import swanlab
    except Exception:
        return None

    try:
        if SWANLAB_API_KEY:
            swanlab.login(api_key=SWANLAB_API_KEY, save=False)
    except Exception:
        pass

    config = {
        "run_name": SWANLAB_RUN_NAME,
        "log_dir": LOG_DIR,
    }

    # common training hyperparameters
    try:
        args = get_args()
        common_keys = [
            # data/tokens
            "data_path",
            "train_samples",
            "consumed_train_samples",
            # batch/parallel
            "micro_batch_size",
            "global_batch_size",
            "data_parallel_size",
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
            # training
            "train_iters",
            "lr_warmup_iters",
            "lr_decay_iters",
            "learning_rate",
            "min_lr",
            "lr_decay_style",
            "weight_decay",
            "clip_grad",
            "adam_beta1",
            "adam_beta2",
            "adam_eps",
            "seed",
            # model
            "seq_length",
            "max_position_embeddings",
            "num_layers",
            "hidden_size",
            "ffn_hidden_size",
            "num_attention_heads",
            "rotary_percent",
            "vocab_size",
        ]
        for k in common_keys:
            if hasattr(args, k):
                v = getattr(args, k)
                if v is not None:
                    config[k] = v
    except Exception:
        pass

    kwargs = {
        "project": SWANLAB_PROJECT,
        "experiment_name": SWANLAB_RUN_NAME,
        "description": f"auto from lmsv, mode={_mode}",
        "tags": ["lmsv", "mindspeed", _mode],
        "config": config,
    }

    _swanlab_run = swanlab.init(**kwargs)
    return _swanlab_run

import random
import os
import numpy as np
import torch
import torch_npu

def seed_all(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch_npu.npu.manual_seed_all(seed)
    torch_npu.npu.manual_seed(seed)


def _find_first_param(module, preferred_keywords):
    if module is None or not hasattr(module, "named_parameters"):
        return None, None
    lowered = [keyword.lower() for keyword in preferred_keywords]
    fallback = None
    for name, param in module.named_parameters():
        lower_name = name.lower()
        if fallback is None and "weight" in lower_name:
            fallback = (name, param)
        if "weight" in lower_name and any(keyword in lower_name for keyword in lowered):
            return name, param
    if fallback is not None:
        return fallback
    return None, None


_STRUCTURE_KEYWORDS = [
    "embed",
    "embedding",
    "word",
    "tok",
    "token",
    "attention",
    "attn",
    "qkv",
    "mlp",
    "norm",
    "output",
    "lm_head",
]


def _format_debug_items(items, limit=200):
    preview = list(items[:limit])
    return "[" + ",".join(str(item) for item in preview) + "]"


def _keyword_hits(items):
    result = {}
    for keyword in _STRUCTURE_KEYWORDS:
        hits = [item for item in items if keyword in item.lower()]
        if hits:
            result[keyword] = hits
    return result


def _log_model_structure_once(model):
    if getattr(model, "_lmsv_debug_structure_logged", False) or not is_rank0():
        return
    try:
        param_names = [name for name, _ in model.named_parameters()]
        module_names = [name for name, _ in model.named_modules()]
        print(
            f"[LMSV_DEBUG] step=-1 full_model.type={type(model).__module__}.{type(model).__name__}"
        )
        print(
            f"[LMSV_DEBUG] step=-1 full_model.named_parameters_count={len(param_names)} "
            f"full_model.named_parameters={_format_debug_items(param_names, limit=200)}"
        )
        print(
            f"[LMSV_DEBUG] step=-1 full_model.named_modules_count={len(module_names)} "
            f"full_model.named_modules={_format_debug_items(module_names, limit=200)}"
        )
        for keyword, hits in _keyword_hits(param_names).items():
            print(
                f"[LMSV_DEBUG] step=-1 full_model.named_parameters_keyword={keyword} "
                f"hits={_format_debug_items(hits, limit=200)}"
            )
        for keyword, hits in _keyword_hits(module_names).items():
            print(
                f"[LMSV_DEBUG] step=-1 full_model.named_modules_keyword={keyword} "
                f"hits={_format_debug_items(hits, limit=200)}"
            )
    except Exception as exc:
        print(f"[LMSV_DEBUG] step=-1 full_model.structure_log_error={exc}")
    model._lmsv_debug_structure_logged = True


def _log_model_weight_summaries_once(model):
    if not should_log_weights_once():
        return

    embedding_name, embedding_weight = _find_first_param(model, ["word_embeddings", "embedding"])
    debug_parameter_summary("weight.embedding", embedding_weight, max_items=8)
    if embedding_name is not None:
        debug_scalar("weight.embedding_name", embedding_name)

    attn_name, attn_weight = _find_first_param(
        model,
        ["self_attention", "attention", "attn", "query_key_value", "qkv"],
    )
    debug_parameter_summary("weight.first_attention", attn_weight, max_items=8)
    if attn_name is not None:
        debug_scalar("weight.first_attention_name", attn_name)

    mlp_name, mlp_weight = _find_first_param(
        model,
        ["mlp", "dense_h_to_4h", "up_proj", "gate_proj", "fc1", "w1", "w3"],
    )
    debug_parameter_summary("weight.first_mlp", mlp_weight, max_items=8)
    if mlp_name is not None:
        debug_scalar("weight.first_mlp_name", mlp_name)

    norm_name, norm_weight = _find_first_param(model, ["final_norm", "final_layernorm", "norm"])
    debug_parameter_summary("weight.final_norm", norm_weight, max_items=8)
    if norm_name is not None:
        debug_scalar("weight.final_norm_name", norm_name)

    lm_head_source = getattr(model, "output_layer", None) or getattr(model, "lm_head", None) or model
    lm_head_name, lm_head_weight = _find_first_param(lm_head_source, ["output_layer", "lm_head"])
    debug_parameter_summary("weight.lm_head", lm_head_weight, max_items=8)
    if lm_head_name is not None:
        debug_scalar("weight.lm_head_name", lm_head_name)

    mark_weights_logged()


def _install_model_debug_hooks(model):
    if getattr(model, "_lmsv_debug_hooks_installed", False):
        return model
    _log_model_structure_once(model)

    output_layer = getattr(model, "output_layer", None) or getattr(model, "lm_head", None)
    if output_layer is not None and hasattr(output_layer, "forward"):
        original_output_forward = output_layer.forward

        def wrapped_output_forward(*args, **kwargs):
            result = original_output_forward(*args, **kwargs)
            _log_model_weight_summaries_once(model)
            logits = result[0] if isinstance(result, tuple) else result
            debug_tensor_summary("logits", logits, max_items=16, include_stats=True)
            if isinstance(logits, torch.Tensor) and logits.dim() >= 3:
                debug_tensor_summary("logits.raw_0_0_8", logits[0, 0, :8], max_items=8, include_stats=True)
                if logits.shape[1] > 1:
                    debug_tensor_summary("logits.raw_0_1_8", logits[0, 1, :8], max_items=8, include_stats=True)
            return result

        output_layer.forward = wrapped_output_forward

    if hasattr(model, "compute_language_model_loss"):
        original_compute_loss = model.compute_language_model_loss

        def wrapped_compute_language_model_loss(labels, logits, *args, **kwargs):
            debug_message("loss.compute_language_model_loss")
            if should_log_full() or should_log_heavy():
                debug_tensor_summary("loss.labels", labels, max_items=16, include_stats=False)
            loss = original_compute_loss(labels, logits, *args, **kwargs)
            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                debug_tensor_summary("loss.model_output", loss, max_items=16, include_stats=True, include_sum=True)
            else:
                debug_scalar("loss.model_output", loss.item() if hasattr(loss, "item") else loss)
            return loss

        model.compute_language_model_loss = wrapped_compute_language_model_loss

    model._lmsv_debug_hooks_installed = True
    return model

def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"


    global csv_path
    if args.rank == 0:
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Iteration','Execution Time (s)','NPU Memory (MB)','loss'])
    
    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if not args.use_legacy_models:
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm)
        mtp_block_spec = None
        if args.mtp_num_layers is not None:
            mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec, use_transformer_engine=use_te)

        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling,
            mtp_block_spec=mtp_block_spec,
        )
    else:
        if not args.context_parallel_size == 1:
            raise ValueError("Context parallelism is only supported with Megatron Core!")

        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process
        )

    return _install_model_debug_hooks(model)


def get_batch(data_iterator):
    """Generate a batch."""

    args = get_args()

    is_middle_stage = not (mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage())
    pretrain_not_tnd_flags = not args.is_instruction_dataset and not args.reset_position_ids
    if pretrain_not_tnd_flags and is_middle_stage:
        return (None,) * 5

    # get batches based on the TP rank you are on.
    # Some PTA/Megatron variants return only the batch, while others return
    # (batch, actual_seq_len). Support both to keep the runtime portable.
    batch_result = get_batch_on_this_tp_rank(data_iterator)
    if isinstance(batch_result, tuple) and len(batch_result) == 2:
        batch, actual_seq_len = batch_result
    else:
        batch = batch_result
        actual_seq_len = None

    if args.return_document_ids and mpu.get_context_parallel_rank() == 0 and mpu.get_tensor_model_parallel_rank() == 0 and mpu.get_pipeline_model_parallel_rank() == 0:
        print("current idx: {}, current rank: {}, data_parallel_rank: {}, document_ids: {}".format(batch['idx'], torch.distributed.get_rank(), mpu.get_data_parallel_rank(), batch['document_ids']))
        batch.pop('document_ids', None)
        batch.pop('idx', None)

    # get batch_list for mtp_block
    if args.mtp_num_layers:
        mtp_batch_list = generate_mtp_batch_list_on_this_tp_rank(batch)
        set_mtp_batch_list(mtp_batch_list)

    if args.reset_position_ids and not args.reset_attention_mask:
        generate_actual_seq_len(batch, actual_seq_len)
        batch = get_batch_on_this_cp_rank(batch)
    else:
        # slice batch along sequence dimension for context parallelism
        batch = get_batch_on_this_cp_rank(batch)
    return batch.values()


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])

    if should_log_full():
        debug_tensor_summary("loss_mask", loss_mask, max_items=16, include_stats=False, include_sum=True)
        debug_tensor_summary("token_loss_pre_reduction", losses, max_items=16, include_stats=True, include_sum=True)
    elif should_log_heavy():
        debug_tensor_summary("token_loss_pre_reduction", losses, max_items=16, include_stats=True, include_sum=True)
    debug_scalar("effective_tokens", total_tokens.item() if hasattr(total_tokens, "item") else total_tokens)
    debug_scalar("loss_weighted_sum", loss[0].item() if hasattr(loss[0], "item") else loss[0])

    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=False,
        )
    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    try:
        from taskd.python.adaptor.elastic_training import common
        if not args.enable_elastic_training or not common.zit_scale_in_running_state():
            torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())
    except ImportError:
        torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

    # loss[0] is a view of loss, so it has ._base not None, which triggers assert error
    # in core/pipeline_parallel/schedule.py::deallocate_output_tensor, calling .clone()
    # on loss[0] fixes this
    local_num_tokens = loss[1].clone().detach().to(torch.int)
    debug_scalar("loss_return", loss[0].item() if hasattr(loss[0], "item") else loss[0])
    return (
        loss[0].clone(),
        local_num_tokens,
        {'lm loss': (reporting_loss[0], reporting_loss[1])},
    )


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()
    step = begin_debug_step()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    if should_log_full(step):
        debug_tensor_summary("input_ids", tokens, max_items=16, include_stats=False)
        debug_tensor_summary("labels", labels, max_items=16, include_stats=False)
        debug_tensor_summary("attention_mask", attention_mask, max_items=16, include_stats=False, include_sum=True)
        debug_tensor_summary("position_ids", position_ids, max_items=16, include_stats=False)
        debug_tensor_summary("loss_mask", loss_mask, max_items=16, include_stats=False, include_sum=True)
        if labels is not None:
            ignore_index = -100
            valid_label_count = (labels != ignore_index).sum()
            debug_scalar("ignore_index_assumed", ignore_index)
            debug_scalar(
                "labels_ne_assumed_ignore_index",
                valid_label_count.item() if hasattr(valid_label_count, "item") else valid_label_count,
            )

    if args.use_legacy_models:
        output_tensor = model(tokens, position_ids, attention_mask,
                              labels=labels)
    else:
        output_tensor = model(tokens, position_ids, attention_mask,
                              labels=labels, loss_mask=loss_mask)

    
    return output_tensor, partial(loss_func, loss_mask)


def is_dataset_built_on_rank():
    return mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=get_blend_from_list(args.data_path),
        blend_per_split=[
            get_blend_from_list(args.train_data_path),
            get_blend_from_list(args.valid_data_path),
            get_blend_from_list(args.test_data_path)
        ],
        split=args.split,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if config.mock:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset
    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")
    # if train_ds is not None and is_dataset_built_on_rank():
    #     temp_path = "test/dataset_samples.csv"
    #     with open(temp_path, mode='w', newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerow(['Dataset Samples'])
    #         writer.writerow(['Train Samples', len(train_ds)])
    #         for i in range(min(3, len(train_ds))):
    #             writer.writerow([f'Train Sample {i+1}', train_ds[i]])

    return train_ds, valid_ds, test_ds


def main():
    seed_all()
    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step
             ,csv_path)
    print_rank_0(f"pretrain finished, log in {csv_path}")


if __name__ == "__main__":
    main()
