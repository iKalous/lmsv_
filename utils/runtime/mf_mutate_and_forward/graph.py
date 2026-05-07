import os
import copy
import sys
import mindspore as ms
from mindspore import nn, ops, Tensor
from ruamel.yaml import YAML

from mindformers.parallel_core.inference.tensor_parallel.layers import ColumnParallelLinear
from mindformers.parallel_core.transformer_config import TransformerConfig
try:
    from mindformers.parallel_core.training_graph.transformer.utils import get_attn_mask_func
except ImportError:
    from mindformers.parallel_core.inference.utils import get_attn_mask_func
from mindformers.parallel_core.training_graph.base_models.common.embeddings.language_model_embedding import (
    LanguageModelEmbedding,
)
from mindformers.parallel_core.training_graph.device_matrix import layout
from mindformers.parallel_core.inference.base_models.common.embeddings.rotary_pos_embedding import (
    RotaryEmbedding,
)

sys.path.append(".")
sys.path.append("..")

import json
from types import SimpleNamespace

from utils.runtime.debug_utils import debug_tensor_summary


ATTENTION_ALIGN_JSON_KEY = "__pta_attention_align__"
DECODER_ALIGN_JSON_KEY = "__pta_decoder_align__"
ATTENTION_ALIGN_FIELDS = (
    "num_attention_heads",
    "num_query_groups",
    "kv_channels",
    "attention_dropout",
    "hidden_dropout",
    "normalization",
    "layernorm_epsilon",
    "masked_softmax_fusion",
    "attention_softmax_in_fp32",
    "apply_query_key_layer_scaling",
)

DECODER_CRITICAL_ALIGN_FIELDS = (
    "bias_dropout_fusion",
    "apply_rope_fusion",
    "context_parallel_algo",
    "use_flash_attn",
)


def _env_flag(name, default="1"):
    # 注意：MindSpore JIT严格模式不支持链式字符串方法，因此使用直接比较
    # 环境变量由我们控制，格式统一为小写
    value = os.getenv(name, default)
    return value == "1" or value == "true" or value == "yes" or value == "on"


def _should_emit_layer_summary():
    return _env_flag("LMSV_DEBUG_COMPARE", "0") or _env_flag("LMSV_LAYER_SUMMARY", "0")


def _emit_layer_summary(name, tensor):
    if _should_emit_layer_summary():
        debug_tensor_summary(name, tensor, max_items=8, include_stats=True)


def _decoder_align_policy():
    return {
        "add_qkv_bias": _env_flag("LMSV_ALIGN_ADD_QKV_BIAS", "0"),
        "bias_dropout_fusion": _env_flag("LMSV_ALIGN_BIAS_DROPOUT_FUSION", "1"),
        "apply_rope_fusion": _env_flag("LMSV_ALIGN_APPLY_ROPE_FUSION", "1"),
        "context_parallel_algo": _env_flag("LMSV_ALIGN_CONTEXT_PARALLEL_ALGO", "1"),
        "use_flash_attn": _env_flag("LMSV_ALIGN_USE_FLASH_ATTN", "1"),
        # activation_func is only aligned when PTA has explicit non-null value.
        "activation_func": _env_flag("LMSV_ALIGN_ACTIVATION_FUNC", "1"),
    }


def _torch_size_str(shape):
    return f"torch.Size({list(shape)})"


def _resolve_attention_mask_func(config):
    """Resolve attention mask function across MindFormers versions."""
    try:
        mask_func = get_attn_mask_func("attn_mask_fill")
        if isinstance(mask_func, type):
            return mask_func(config)
        return mask_func
    except Exception:
        def _mask_fill(attention_scores: Tensor, attention_mask, fill_value=-10000.0):
            return ops.masked_fill(attention_scores, attention_mask,
                                   Tensor(fill_value, attention_scores.dtype))
        return _mask_fill


def _ensure_parallel_config(config):
    parallel_config = getattr(config, "parallel_config", None)
    if parallel_config is None:
        parallel_config = SimpleNamespace()
        config.parallel_config = parallel_config
    elif isinstance(parallel_config, dict):
        parallel_config = SimpleNamespace(**parallel_config)
        config.parallel_config = parallel_config


def _ensure_layout_initialized(config):
    """Ensure DP/CP/TP layout is initialized for training-graph modules."""
    try:
        if not hasattr(config, "data_parallel_size"):
            config.data_parallel_size = 1
        if not hasattr(config, "tensor_model_parallel_size"):
            config.tensor_model_parallel_size = 1
        if not hasattr(config, "context_parallel_size"):
            config.context_parallel_size = 1
        layout.init_dp_cp_tp_layout(config)
    except Exception:
        return


def _normalize_init_method(config_dict):
    if not config_dict or "init_method" not in config_dict:
        return
    init_method = config_dict.get("init_method")
    xavier_aliases = {
        "torch.nn.init.xavier_uniform_",
        "xavieruniform",
        "xavier_uniform",
        "xavier_uniform_",
    }
    if isinstance(init_method, str) and init_method in xavier_aliases:
        initializer = ms.common.initializer.XavierUniform()
    elif init_method == ms.common.initializer.XavierUniform:
        initializer = ms.common.initializer.XavierUniform()
    else:
        initializer = None

    if initializer is None:
        return

    def _init_wrapper(shape):
        shape_tuple = tuple(shape)
        return ms.common.initializer.initializer(initializer, shape_tuple, ms.float32)

    config_dict["init_method"] = _init_wrapper


def _load_pta_attention_alignment(layer_configs, debug=False):
    if not isinstance(layer_configs, dict):
        return {}

    entry = layer_configs.get(ATTENTION_ALIGN_JSON_KEY)
    if not isinstance(entry, dict):
        return {}

    node_cfgs = entry.get("nodes")
    if not isinstance(node_cfgs, dict):
        return {}

    allow_fields = entry.get("attention_fields")
    if not isinstance(allow_fields, list) or not allow_fields:
        allow_fields = list(ATTENTION_ALIGN_FIELDS)
    allow_set = set(allow_fields)

    normalized = {}
    for node_id, cfg in node_cfgs.items():
        if not isinstance(cfg, dict):
            continue
        filtered = {
            k: v for k, v in cfg.items()
            if k in allow_set
        }
        if filtered:
            normalized[str(node_id)] = filtered

    if debug and normalized:
        print(
            f"  检测到PTA attention对齐元数据: nodes={sorted(normalized.keys())} "
            f"fields={sorted(list(allow_set))}"
        )

    return normalized


def _load_pta_decoder_alignment(layer_configs, debug=False):
    if not isinstance(layer_configs, dict):
        return {}

    entry = layer_configs.get(DECODER_ALIGN_JSON_KEY)
    if not isinstance(entry, dict):
        return {}

    node_cfgs = entry.get("nodes")
    if not isinstance(node_cfgs, dict):
        return {}

    policy = _decoder_align_policy()

    normalized = {}
    for node_id, cfg in node_cfgs.items():
        if not isinstance(cfg, dict):
            continue
        filtered = {}
        for key in DECODER_CRITICAL_ALIGN_FIELDS:
            if not policy.get(key, True):
                continue
            value = cfg.get(key)
            if value is not None:
                filtered[key] = value

        if policy.get("add_qkv_bias", False) and cfg.get("add_qkv_bias") is not None:
            filtered["add_qkv_bias"] = cfg.get("add_qkv_bias")

        # activation_func is high impact, but only when PTA has explicit value.
        if policy.get("activation_func", True) and "activation_func" in cfg and cfg.get("activation_func") is not None:
            filtered["activation_func"] = cfg.get("activation_func")

        if filtered:
            normalized[str(node_id)] = filtered

    if debug and normalized:
        aligned_fields = sorted(list(set().union(*[set(v.keys()) for v in normalized.values()]))) if normalized else []
        print(
            f"  检测到PTA decoder关键对齐元数据: nodes={sorted(normalized.keys())} "
            f"fields={aligned_fields}"
        )
        if not policy.get("add_qkv_bias", False):
            print("  decoder字段 add_qkv_bias 默认不对齐（可通过 LMSV_ALIGN_ADD_QKV_BIAS=1 开启）")

        disabled = sorted([k for k, v in policy.items() if not v])
        if disabled:
            print(f"  decoder字段对齐开关已关闭: {disabled}")

    return normalized


 


def reshape_tensor_nd(
    input_tensor: ms.Tensor,
    target_shape: tuple,
    fill_value: float = 0
) -> ms.Tensor:
    """Resize tensor by flattening and padding/slicing to target shape."""
    input_shape = input_tensor.shape
    input_numel = input_tensor.size

    if len(input_shape) == len(target_shape) and all(
        input_shape[i] == target_shape[i] for i in range(len(target_shape) - 1)
    ):
        last_in = input_shape[-1]
        last_tgt = target_shape[-1]
        if last_tgt == last_in:
            return input_tensor.copy()
        if last_tgt < last_in:
            slices = [slice(None)] * (len(target_shape) - 1) + [slice(0, last_tgt)]
            return input_tensor[tuple(slices)].copy()
        pad_len = last_tgt - last_in
        pad_shape = input_shape[:-1] + (pad_len,)
        padding = ops.full(pad_shape, fill_value, dtype=input_tensor.dtype)
        return ops.cat((input_tensor, padding), axis=-1).copy()

    target_numel = 1
    for s in target_shape:
        target_numel *= s

    flattened = input_tensor.flatten()
    if target_numel > input_numel:
        pad_shape = (target_numel - input_numel,)
        padding = ops.full(pad_shape, fill_value, dtype=input_tensor.dtype)
        output = ops.cat((flattened, padding), axis=0)
    elif target_numel < input_numel:
        output = flattened[:target_numel]
    else:
        output = flattened

    return ops.identity(output.reshape(target_shape))


def transfor_shape(output):
    """Match legacy embedding->decoder shape expectations."""
    output = output.swapaxes(0, 1)
    if output.dtype != ms.float32:
        output = ops.cast(output, ms.float32)
    if output.shape[-1] != 896:
        seq_len, batch_size, _ = output.shape
        output = reshape_tensor_nd(output, (seq_len, batch_size, 896))
    output = ops.tile(output, (4, 2, 1))
    return output


class Node:
    def __init__(self, config, index=-1):
        super().__init__()
        self.from_nodes = []
        self.to_nodes = []
        self.layer_limits = []
        self.op = None
        self.id = index
        self.origin_id = -1
        self.state = 'none'
        self.is_des = False
        self.is_src = False
        self.visit_count = 0
        self.succ_count = 0
        self.str_op = 'empty'
        self.params = {}
        self.input_shape = []
        self.output_shape = []
        self.in_degree = len(self.from_nodes)
        self.out_degree = len(self.to_nodes)
        self.config = config
        self.block = None

    def add_from(self, node):
        self.from_nodes.append(node.id)

    def add_to(self, node):
        self.to_nodes.append(node.id)


class Graph(nn.Cell):

    def __init__(
            self,
            config_path: str = None,
            config_dict: dict = None,
            nums: list = None,
            mutated_nodes: dict = None,
    ):
        if config_dict is not None:
            model_config = config_dict
            if 'config' in model_config:
                _normalize_init_method(model_config['config'])
        elif config_path is not None:
            yaml = YAML()
            with open(config_path, 'r', encoding='utf-8') as file:
                model_config = yaml.load(file)
                _normalize_init_method(model_config.get('config', {}))
        else:
            raise ValueError("必须提供 config_path 或 config_dict 中的一个")

        valid_fields = set(TransformerConfig.__dataclass_fields__.keys())
        filtered_cfg_dict = {
            k: v for k, v in model_config["config"].items() if k in valid_fields
        }
        if filtered_cfg_dict.get("num_attention_heads", 0) in (0, None):
            filtered_cfg_dict["num_attention_heads"] = 1
        if filtered_cfg_dict.get("num_query_groups") in (0, None):
            filtered_cfg_dict["num_query_groups"] = filtered_cfg_dict.get("num_attention_heads", 1)
        if filtered_cfg_dict.get("tensor_model_parallel_size", 0) in (0, None):
            filtered_cfg_dict["tensor_model_parallel_size"] = 1
        if filtered_cfg_dict.get("pipeline_model_parallel_size", 0) in (0, None):
            filtered_cfg_dict["pipeline_model_parallel_size"] = 1
        if filtered_cfg_dict.get("num_layers", 0) in (0, None):
            filtered_cfg_dict["num_layers"] = 1
        if filtered_cfg_dict.get("kv_channels") in (0, None):
            hidden_size = filtered_cfg_dict.get("hidden_size", 0)
            num_heads = filtered_cfg_dict.get("num_attention_heads", 1)
            filtered_cfg_dict["kv_channels"] = hidden_size // num_heads if num_heads else hidden_size

        transformerblock_config = TransformerConfig(**filtered_cfg_dict)
        _ensure_parallel_config(transformerblock_config)
        _ensure_layout_initialized(transformerblock_config)
        model_config["config"] = transformerblock_config
        self.total_config = model_config

        super().__init__()
        self.transformer_config = transformerblock_config
        self.attn_mask_func = _resolve_attention_mask_func(self.transformer_config)
        self.is_ascend_available = ms.hal.is_available("Ascend")

        init_config = TransformerConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            num_layers=24,
            hidden_size=896,
            ffn_hidden_size=4864,
            num_attention_heads=14,
            num_query_groups=2,
            attention_dropout=0.0,
            init_method_std=0.01,
            hidden_dropout=0.0,
            normalization="RMSNorm",
            layernorm_epsilon=1e-6
        )
        self.nodes = dict(zip([id for id in nums], [Node(config=init_config, index=id) for id in nums]))

        self._block_prefix = "node_block_"

        self.mutated_nodes = mutated_nodes if mutated_nodes is not None else {}

        self.embedding = None
        position_embedding_type = self.total_config.get(
            'position_embedding_type', self.transformer_config.position_embedding_type
        )
        if position_embedding_type == 'rope' and not self.transformer_config.multi_latent_attention:
            rotary_kwargs = dict(
                kv_channels=self.transformer_config.kv_channels,
                rotary_percent=self.total_config.get('rotary_percent', 1.0),
                rotary_interleaved=self.transformer_config.rotary_interleaved,
                seq_len_interpolation_factor=self.total_config.get('seq_len_interpolation_factor', 1.0),
                rotary_base=self.total_config.get('rotary_base', 10000),
                rope_scaling=self.total_config.get('rope_scaling', None),
                rope_scaling_factor=self.total_config.get('rope_scaling_factor', 1.0),
            )
            try:
                self.rotary_pos_emb = RotaryEmbedding(**rotary_kwargs)
            except TypeError:
                rotary_kwargs.pop('rope_scaling', None)
                rotary_kwargs.pop('rope_scaling_factor', None)
                rotary_kwargs.pop('seq_len_interpolation_factor', None)
                self.rotary_pos_emb = RotaryEmbedding(**rotary_kwargs)

        self.pre_process = False
        self.post_process = False
        if self.post_process:
            if getattr(self.transformer_config, "defer_embedding_wgrad_compute", False):
                self.embedding_activation_buffer = []
                self.grad_output_buffer = []
            else:
                self.embedding_activation_buffer = None
                self.grad_output_buffer = None

            self.share_embeddings_and_output_weights = False

            if getattr(self.transformer_config, "_cpu_offloading_context", None) == 'None':
                self.transformer_config._cpu_offloading_context = None

            output_layer_kwargs = dict(
                config=self.transformer_config,
                init_method=self.transformer_config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=False,
                skip_weight_param_allocation=self.pre_process
                                             and self.share_embeddings_and_output_weights,
                embedding_activation_buffer=self.embedding_activation_buffer,
                grad_output_buffer=self.grad_output_buffer,
            )
            try:
                self.output_layer = ColumnParallelLinear(
                    self.transformer_config.hidden_size,
                    self.total_config.get('vocab_size', self.transformer_config.vocab_size),
                    **output_layer_kwargs,
                )
            except TypeError:
                output_layer_kwargs.pop('grad_output_buffer', None)
                output_layer_kwargs.pop('embedding_activation_buffer', None)
                self.output_layer = ColumnParallelLinear(
                    self.transformer_config.hidden_size,
                    self.total_config.get('vocab_size', self.transformer_config.vocab_size),
                    **output_layer_kwargs,
                )

    def construct(self, input_ids=None, position_ids=None, attention_mask=None, labels=None, debug=False):
        compare_eval_only = _env_flag("LMSV_COMPARE_EVAL_ONLY", "0")
        if compare_eval_only:
            self.set_train(False)

        mutated_nodes = self.mutated_nodes
        start_node = 0 if 0 in self.nodes else 1
        cur_node = self.nodes[start_node]

        if input_ids is None:
            values = [[5032], [39706], [24761], [14473], [35428], [1358], [20794], [6819]]
            input_ids = ms.Tensor(values, dtype=ms.int32)
        if position_ids is None:
            batch_size = input_ids.shape[0]
            seq_length = input_ids.shape[1]
            position_ids = ops.broadcast_to(
                ops.arange(seq_length, dtype=ms.int32),
                (batch_size, seq_length),
            )
        # 改为（下三角 causal mask）：
        if attention_mask is None:
            seq_length = position_ids.shape[1]
            # 创建 causal mask: [1, 1, seq, seq]
            mask = ops.ones((seq_length, seq_length), dtype=ms.bool_)
            mask = ops.triu(mask, diagonal=1)  # 上三角为 1（需要 mask 的位置）
            attention_mask = mask[None, None, :, :]  # [1, 1, seq, seq]
        output = None
        input_data = None
        decoder_index = 0
        while True:
            if debug:
                print(f"\n处理节点 {cur_node.id}: {cur_node.str_op}")
            if cur_node.block is not None:
                cur_block = getattr(self, f"{self._block_prefix}{cur_node.id}", cur_node.block)
                if self.is_ascend_available and hasattr(cur_block, "npu"):
                    cur_block = cur_block.npu()

                if cur_node.str_op == "embedding":
                    output = cur_block(input_ids=input_ids, position_ids=position_ids)
                    input_data = transfor_shape(output)
                    _emit_layer_summary("embedding.output", input_data)
                elif "decoder" in cur_node.str_op:
                    if input_data is None:
                        input_data = output
                    expected_hidden = cur_node.config.hidden_size
                    if input_data is not None and input_data.shape[-1] != expected_hidden:
                        seq_len, batch_size, _ = input_data.shape
                        input_data = reshape_tensor_nd(input_data, (seq_len, batch_size, expected_hidden))
                    if decoder_index == 0:
                        _emit_layer_summary("block0.input", input_data)
                    if input_data is not None:
                        seq_len = input_data.shape[0]
                        if attention_mask is None or attention_mask.shape[-1] != seq_len:
                            attention_mask = ops.zeros((1, 1, seq_len, seq_len), dtype=ms.bool_)
                    res = cur_block(input_data, attention_mask)
                    output = res[0] if isinstance(res, tuple) else res
                    input_data = output
                    if decoder_index == 0:
                        _emit_layer_summary("block0.output", output)
                    elif decoder_index == 1:
                        _emit_layer_summary("block1.output", output)
                    if len(cur_node.to_nodes) == 0:
                        _emit_layer_summary("last_block.output", output)
                    decoder_index += 1
                    if debug and input_data is not None:
                        print(f"  Decoder输出形状: {_torch_size_str(input_data.shape)}")
                else:
                    res = cur_block(input_data)
                    output = res[0] if isinstance(res, tuple) else res
                    input_data = output

                if debug and output is not None:
                    out_for_shape = output[0] if isinstance(output, tuple) else output
                    print(f"  模块输出形状: {out_for_shape.shape}")

            if len(cur_node.to_nodes) == 0:
                break
            cur_node = self.nodes[cur_node.to_nodes[0]]

        if debug and input_data is not None:
            print(f"\n--- 最终输出处理 ---")
            print(f"最终输出形状: {_torch_size_str(input_data.shape)}")

        if getattr(self, 'output_layer', None) is not None and input_data is not None:
            _emit_layer_summary("lm_head.input", input_data)
            res = self.output_layer(input_data)
            logits = res[0] if isinstance(res, tuple) else res
            _emit_layer_summary("lm_head.output", logits)
            return logits
        return input_data

    def forward(self, *args, **kwargs):
        return self.construct(*args, **kwargs)

    def set_mutated_nodes(self, mutated_nodes: dict):
        self.mutated_nodes = mutated_nodes if mutated_nodes is not None else {}

    def get_mutated_nodes(self):
        return self.mutated_nodes

    def load(self, config_yaml_path: str, config_json_path: str, debug: bool = True):
        try:
            if debug:
                print(f"=== 从配置文件加载图: {config_yaml_path} ===")

            yaml = YAML()
            with open(config_yaml_path, 'r', encoding='utf-8') as file:
                loaded_config = yaml.load(file)

            if debug:
                print(f"成功读取配置文件")
                if 'metadata' in loaded_config:
                    metadata = loaded_config['metadata']
                    print(f"  迭代轮数: {metadata.get('iteration', 'N/A')}")
                    print(f"  变异率: {metadata.get('mutation_rate', 'N/A')}")
                    print(f"  创建时间: {metadata.get('timestamp', 'N/A')}")

            if 'base_config' not in loaded_config or 'graph_structure' not in loaded_config:
                raise ValueError("配置文件格式错误：缺少base_config或graph_structure")

            base_config = loaded_config['base_config']

            _normalize_init_method(base_config.get('config', {}))

            if 'autocast_dtype' in base_config['config'] and isinstance(base_config['config']['autocast_dtype'], str):
                if base_config['config']['autocast_dtype'] in ['torch.float16', 'ms.float16']:
                    base_config['config']['autocast_dtype'] = ms.float16
                elif base_config['config']['autocast_dtype'] in ['torch.float32', 'ms.float32']:
                    base_config['config']['autocast_dtype'] = ms.float32
                elif base_config['config']['autocast_dtype'] in ['torch.half', 'ms.half']:
                    base_config['config']['autocast_dtype'] = ms.float16

            valid_fields = set(TransformerConfig.__dataclass_fields__.keys())
            filtered_cfg_dict = {
                k: v for k, v in base_config["config"].items() if k in valid_fields
            }
            unknown_keys = set(base_config["config"].keys()) - valid_fields
            if debug and unknown_keys:
                print(
                    f"  检测到未识别的 TransformerConfig 字段，已忽略: {sorted(list(unknown_keys))}"
                )

            transformerblock_config = TransformerConfig(**filtered_cfg_dict)
            _ensure_parallel_config(transformerblock_config)
            _ensure_layout_initialized(transformerblock_config)

            base_config["config"] = transformerblock_config
            self.total_config = base_config

            self.transformer_config = transformerblock_config
            self.attn_mask_func = _resolve_attention_mask_func(self.transformer_config)
            self.is_ascend_available = ms.hal.is_available("Ascend")

            with open(config_json_path, 'r', encoding='utf-8') as file:
                layer_configs = json.load(file)

            pta_attention_align = _load_pta_attention_alignment(layer_configs, debug=debug)
            pta_decoder_align = _load_pta_decoder_alignment(layer_configs, debug=debug)
            numeric_node_ids = sorted(
                int(k) for k in layer_configs.keys()
                if str(k).isdigit()
            )

            # Keep embedding/base positional behavior consistent with PTA node config.
            first_decoder_id = next((nid for nid in numeric_node_ids if int(nid) > 0), None)
            if first_decoder_id is not None:
                raw_node_cfg = layer_configs.get(str(first_decoder_id), {}).get('after', {}).get('TransformerConfig', {})
                if isinstance(raw_node_cfg, dict):
                    pta_pos_type = raw_node_cfg.get('position_embedding_type')
                    pta_add_pos = raw_node_cfg.get('add_position_embedding')

                    if pta_pos_type is not None and self.total_config.get('position_embedding_type') != pta_pos_type:
                        self.total_config['position_embedding_type'] = pta_pos_type
                        if debug:
                            print(f"  位置编码类型已按PTA对齐: position_embedding_type={pta_pos_type}")

                    if pta_add_pos is not None and self.total_config.get('add_position_embedding') != pta_add_pos:
                        self.total_config['add_position_embedding'] = pta_add_pos
                        if debug:
                            print(f"  位置编码开关已按PTA对齐: add_position_embedding={pta_add_pos}")

            if debug:
                print(f"  节点数量: {max(0, len(numeric_node_ids) - 1)}")

            self.nodes.clear()
            self.mutated_nodes.clear()

            embedd_config = TransformerConfig(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                num_layers=24,
                hidden_size=896,
                ffn_hidden_size=4864,
                num_attention_heads=14,
                num_query_groups=2,
                attention_dropout=0.0,
                init_method_std=0.01,
                hidden_dropout=0.0,
                normalization="RMSNorm",
                layernorm_epsilon=1e-6
            )

            max_node_id = max(numeric_node_ids) if numeric_node_ids else 0
            for node_id in numeric_node_ids:
                layer_config = layer_configs.get(str(node_id), layer_configs.get(node_id, {}))
                if not isinstance(layer_config, dict):
                    continue
                if int(node_id) > 0:
                    raw_cfg = layer_config['after']['TransformerConfig']
                    valid_fields = set(TransformerConfig.__dataclass_fields__.keys())
                    filtered_cfg = {k: v for k, v in raw_cfg.items() if k in valid_fields}

                    align_cfg = pta_attention_align.get(str(node_id), {})
                    if align_cfg:
                        for key, value in align_cfg.items():
                            if key in valid_fields:
                                filtered_cfg[key] = value
                        if debug:
                            print(
                                f"  节点 {node_id} 应用PTA attention对齐字段: "
                                f"{sorted(list(align_cfg.keys()))}"
                            )

                    decoder_align_cfg = pta_decoder_align.get(str(node_id), {})
                    if decoder_align_cfg:
                        for key, value in decoder_align_cfg.items():
                            if key in valid_fields:
                                filtered_cfg[key] = value
                        if debug:
                            print(
                                f"  节点 {node_id} 应用PTA decoder关键对齐字段: "
                                f"{sorted(list(decoder_align_cfg.keys()))}"
                            )

                    init_config = TransformerConfig(**filtered_cfg)
                    _ensure_parallel_config(init_config)
                    _ensure_layout_initialized(init_config)
                    unknown_fields = set(raw_cfg.keys()) - valid_fields
                    if unknown_fields:
                        if debug:
                            print(f"  节点 {node_id} 检测到未识别的 TransformerConfig 字段，已忽略: {sorted(list(unknown_fields))}")
                        self.mutated_nodes[int(node_id)] = {
                            'invalid_fields': sorted(list(unknown_fields)),
                            'raw_config': raw_cfg,
                        }
                else:
                    init_config = embedd_config
                node = Node(config=init_config, index=node_id)
                node.str_op = "mutated_decoder" if node_id > 0 else "embedding"
                node.from_nodes = [node_id - 1] if node_id > 0 else []
                node.to_nodes = [node_id + 1] if node_id < max_node_id else []
                node.params = layer_config.get('params', {})
                node.state = layer_config.get('state', 'none')

                node.in_degree = len(node.from_nodes)
                node.out_degree = len(node.to_nodes)
                if node.state == 'des':
                    node.out_degree = 1

                if node.str_op == "embedding":
                    node.block = LanguageModelEmbedding(
                        config=self.transformer_config,
                        vocab_size=self.total_config['vocab_size'],
                        max_sequence_length=self.total_config['max_sequence_length'],
                        position_embedding_type=self.total_config['position_embedding_type'],
                    )
                else:
                    from mindformers.parallel_core.training_graph.transformer.transformer_block import TransformerBlock
                    from mindformers.parallel_core.training_graph.base_models.gpt.gpt_layer_specs import (
                        get_gpt_layer_local_spec,
                    )

                    _ensure_parallel_config(node.config)
                    _ensure_layout_initialized(node.config)
                    node.block = TransformerBlock(
                        config=node.config,
                        spec=get_gpt_layer_local_spec(
                            None,
                            False,
                            False,
                        ),
                        pre_process=False,
                        post_process=False,
                    )

                setattr(self, f"{self._block_prefix}{node_id}", node.block)

                self.nodes[node_id] = node

            self._reinitialize_components()

            if debug:
                print(f"  图加载完成!")
                print(f"  总节点数: {len(self.nodes)-1}")

            return True

        except Exception as e:
            if debug:
                print(f"? 加载配置文件失败: {e}")
                import traceback
                traceback.print_exc()
            return False

    def _reinitialize_components(self):
        self.embedding = None
        if self.total_config['position_embedding_type'] == 'rope' and not self.transformer_config.multi_latent_attention:
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.transformer_config.kv_channels,
                rotary_percent=self.total_config['rotary_percent'],
                rotary_interleaved=self.transformer_config.rotary_interleaved,
                seq_len_interpolation_factor=self.total_config['seq_len_interpolation_factor'],
                rotary_base=self.total_config['rotary_base'],
                rope_scaling=self.total_config['rope_scaling'],
                rope_scaling_factor=self.total_config['rope_scaling_factor'],
                use_cpu_initialization=self.transformer_config.use_cpu_initialization,
            )

    def __len__(self):
        return len(self.nodes)
