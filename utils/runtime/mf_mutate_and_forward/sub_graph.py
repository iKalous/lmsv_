import os
import copy
import sys
import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter
from mindspore import numpy as mnp
from ruamel.yaml import YAML

# MindFormers imports
from mindformers.parallel_core.inference.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
from mindformers.parallel_core.inference.parallel_state import initialize_model_parallel, get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
from mindformers.parallel_core.transformer_config import TransformerConfig
try:
    from mindformers.parallel_core.training_graph.transformer.utils import get_attn_mask_func
except ImportError:
    from mindformers.parallel_core.inference.utils import get_attn_mask_func
from mindformers.parallel_core.training_graph.base_models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from mindformers.parallel_core.training_graph.device_matrix import layout

from mindformers.parallel_core.inference.base_models.common.embeddings.rotary_pos_embedding import (
    RotaryEmbedding,
)

# distributed settings
from argparse import ArgumentParser


sys.path.append(".")
sys.path.append("..")

from utils.runtime import common_utils, model_helpers
from utils.runtime.OperatorSet import insert_operators
import random
import json
from types import SimpleNamespace


class _NoNoneSecondOutput(nn.Cell):
    def __init__(self, base_layer):
        super().__init__()
        self.base_layer = base_layer

    def construct(self, x):
        out = self.base_layer(x)
        if isinstance(out, tuple):
            first = out[0]
        else:
            first = out
        fake_second = ops.zeros((1,), first.dtype)
        return first, fake_second


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


def _resolve_dropout_block(core_attn):
    """Resolve attention dropout to a callable module."""
    dropout = getattr(core_attn, "attention_dropout", None)
    if callable(dropout):
        return dropout
    if isinstance(dropout, (float, int)):
        keep_prob = 1.0 - float(dropout)
        keep_prob = max(0.0, min(1.0, keep_prob))
        return nn.Dropout(keep_prob=keep_prob)

    return nn.Dropout(keep_prob=1.0)



def _sync_hidden_size_with_input(base_config, input_sample=None):
    if input_sample is None or getattr(input_sample, "ndim", 0) < 3:
        return base_config
    runtime_hidden = int(input_sample.shape[-1])
    yaml_hidden = base_config["config"].get("hidden_size")
    if yaml_hidden != runtime_hidden:
        print(f"[FIX] Sync hidden_size: yaml={yaml_hidden} -> runtime={runtime_hidden}")
        base_config["config"]["hidden_size"] = runtime_hidden
        num_heads = base_config["config"].get("num_attention_heads", 1)
        if num_heads:
            kv_channels = runtime_hidden // num_heads
            base_config["config"]["kv_channels"] = kv_channels
            base_config["config"]["head_dim"] = kv_channels
    return base_config


def _normalize_init_method(config_dict):
    if not config_dict or "init_method" not in config_dict:
        return

    init_method = config_dict.get("init_method")
    xavier_aliases = {
        "torch.nn.init.xavier_uniform_", "xavieruniform",
        "xavier_uniform", "xavier_uniform_"
    }

    initializer = None
    if isinstance(init_method, str) and init_method.lower() in xavier_aliases:
        initializer = ms.common.initializer.XavierUniform()
    elif init_method == ms.common.initializer.XavierUniform:
        initializer = ms.common.initializer.XavierUniform()

    if initializer is None:
        return

    def _init_wrapper(shape):
        shape_tuple = tuple(shape)
        return ms.common.initializer.initializer(initializer, shape_tuple, ms.float32)

    config_dict["init_method"] = _init_wrapper


def _force_fp32_for_cpu(config, debug=False, prefix=""):
    """CPU backend cannot run some training-graph kernels in bfloat16."""
    if config is None:
        return
    if ms.hal.is_available("Ascend"):
        return

    changed = []
    float32_attrs = (
        "pipeline_dtype",
        "params_dtype",
        "compute_dtype",
        "layernorm_compute_dtype",
        "softmax_compute_dtype",
        "rotary_dtype",
    )
    for attr in float32_attrs:
        if hasattr(config, attr):
            old = getattr(config, attr)
            if old != ms.float32:
                setattr(config, attr, ms.float32)
                changed.append(f"{attr}:{old}->ms.float32")

    if hasattr(config, "bf16") and getattr(config, "bf16"):
        setattr(config, "bf16", False)
        changed.append("bf16:True->False")
    if hasattr(config, "fp16") and getattr(config, "fp16"):
        setattr(config, "fp16", False)
        changed.append("fp16:True->False")

    if debug and changed:
        scope = f"{prefix} " if prefix else ""
        print(f"[CPU-FP32-FALLBACK] {scope}{', '.join(changed)}")


def _build_attention_mask(seq_len, flash_layout=False, causal=False):
    if flash_layout:
        return ops.zeros((seq_len, 1, 1, 1), dtype=ms.bool_)
    if causal:
        mask = ops.ones((seq_len, seq_len), dtype=ms.bool_)
        mask = ops.triu(mask, diagonal=1)
        return mask[None, None, :, :]
    return ops.zeros((1, 1, seq_len, seq_len), dtype=ms.bool_)


def _to_plain_data(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_plain_data(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_plain_data(v) for v in obj]
    if isinstance(obj, SimpleNamespace):
        return _to_plain_data(vars(obj))
    if hasattr(obj, "__dict__"):
        return _to_plain_data(vars(obj))
    return str(obj)


def _dump_node_config(node_id, config, output_dir="res/node_config_dumps"):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"node_{node_id}_config.json")
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(_to_plain_data(config), file, ensure_ascii=False, indent=2, default=str)
    return file_path

def fill_zeros_with_nonzero(tensor):
    """简化版：用均值填充零值，纯 MindSpore 实现。"""
    nonzero_mask = tensor != 0
    if not nonzero_mask.any():
        return ops.zeros_like(tensor)
    nonzero_values = ops.masked_select(tensor, nonzero_mask)
    mean_val = ops.reduce_mean(nonzero_values)
    return ops.where(nonzero_mask, tensor, ops.full_like(tensor, mean_val))

def reshape_tensor_nd(
    input_tensor: ms.Tensor,
    target_shape: tuple,
    fill_value: float = 0
) -> ms.Tensor:
    """
    将任意维度的输入张量调整为任意目标形状，支持填充/裁剪/维度增减。
    """
    input_shape = input_tensor.shape
    input_numel = 1
    for dim in input_shape:
        input_numel *= int(dim)

    # 优先处理：维度一致且除最后一维外形状一致时，按最后一维进行裁剪/填充
    if len(input_shape) == len(target_shape) and all(
        input_shape[i] == target_shape[i] for i in range(len(target_shape) - 1)
    ):
        last_in = input_shape[-1]
        last_tgt = target_shape[-1]
        if last_tgt == last_in:
            return ops.identity(input_tensor)
        if last_tgt < last_in:
            slices = [slice(None)] * (len(target_shape) - 1) + [slice(0, last_tgt)]
            return ops.identity(input_tensor[tuple(slices)])
        pad_len = last_tgt - last_in
        pad_shape = input_shape[:-1] + (pad_len,)
        padding = ms.ops.full(pad_shape, fill_value, dtype=input_tensor.dtype)
        return ops.identity(ms.ops.cat((input_tensor, padding), axis=-1))
    
    # target_numel 计算
    target_numel = 1
    for s in target_shape:
        target_numel *= s

    # Step 1: 将输入展平为1D向量
    flattened = input_tensor.flatten()

    # Step 2: 处理元素数量差异
    if target_numel > input_numel:
        # 填充不足部分
        pad_shape = (target_numel - input_numel,)
        padding = ms.ops.full(pad_shape, fill_value, dtype=input_tensor.dtype)
        padded = ms.ops.cat((flattened, padding), axis=0)
        output = padded
    elif target_numel < input_numel:
        # 裁剪多余部分
        output = flattened[:target_numel]
    else:
        output = flattened

    # Step 3: 调整为目标形状
    return ops.identity(output.reshape(target_shape))


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
        self.block_num = 0
        self.block = None

    def add_from(self, node):
        self.from_nodes.append(node.id)

    def get_from(self, index=-1):
        if index == -1:
            return self.from_nodes
        return self.from_nodes[index]

    def del_from(self, index):
        if index in self.from_nodes:
            self.from_nodes.remove(index)

    def add_to(self, node):
        self.to_nodes.append(node.id)

    def get_to(self, index=-1):
        if index == -1:
            return self.to_nodes
        return self.to_nodes[index]

    def del_to(self, index):
        if index in self.to_nodes:
            self.to_nodes.remove(index)

    def set_op(self, operator, state='none', str_op='empty'):
        self.op = operator
        self.str_op = str_op
        self.state = state
        if self.op is not None and str_op == 'empty':
            op_name = str(type(self.op))
            op_name = op_name[:-2]
            dot_ip = op_name.rindex('.')
            op_name = op_name[dot_ip + 1:]
            self.str_op = op_name.lower()

    def get_op(self):
        return self.op

    def run(self, input_data):
        return self.op(input_data)

    def get_id(self):
        return self.id

    def set_id(self, index):
        self.id = index

    def set_state(self, state):
        self.state = state
        if state == 'des':
            self.is_des = True
        if state == 'src':
            self.is_src = True

    def set_input_shape(self, input_shape=None):
        if input_shape is None:
            input_shape = [None, None]
        self.input_shape = input_shape

    def get_input_shape(self):
        return self.input_shape

    def set_output_shape(self, output_shape):
        self.output_shape = output_shape

    def get_output_shape(self):
        return self.output_shape

    def __str__(self) -> str:
        return f'id:{self.id},op:{self.str_op},to:{self.to_nodes},from:{self.from_nodes},' \
               f'state:{self.state},in_degree:{self.in_degree},out_degree:{self.out_degree},' \
               f'input_shape:{self.input_shape},output_shape:{self.output_shape},' \
               f'params:{self.params},({self.origin_id})'

    def __hash__(self) -> int:
        return hash(self.str_op)

    def __eq__(self, o) -> bool:
        return hash(self.str_op) == hash(o.str_op)


def transfor_shape(output, config):  # (1,8,1024) -> (32,3,hidden)
    # PyTorch transpose(0,1) 交换维度0和1
    output = output.swapaxes(0, 1)  # (1,8,1024) -> (8,1,1024)
    if output.dtype != ms.float32:
        output = ops.cast(output, ms.float32)
    target_hidden = getattr(config, 'hidden_size', 896)
    if output.shape[-1] != target_hidden:
        seq_len, batch_size, _ = output.shape
        output = reshape_tensor_nd(output, (seq_len, batch_size, target_hidden))
    tile_factor_0 = getattr(config, 'tile_factor_0', 4)
    tile_factor_1 = getattr(config, 'tile_factor_1', 2)
    output = ops.tile(output, (tile_factor_0, tile_factor_1, 1))
    return output


class Graph(nn.Cell): # MindSpore 模型通常继承 nn.Cell

    def __init__(
            self,
            config_path: str = None,
            config_dict: dict = None,
            nums: list = None,
            mutated_nodes: dict = None,
    ):
        # 支持两种初始化方式：配置文件路径或配置字典
        if config_dict is not None:
            # 使用配置字典初始化
            model_config = config_dict
        elif config_path is not None:
            # 使用配置文件路径初始化
            config_path = model_helpers.resolve_repo_path(config_path)
            yaml = YAML()
            with open(config_path, 'r', encoding='utf-8') as file:
                model_config = yaml.load(file)
        else:
            raise ValueError("必须提供 config_path 或 config_dict 中的一个")

        _normalize_init_method(model_config.get('config', {}))

        # del model_config["config"]["multi_latent_attention"]
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
        if "attention_dropout" in valid_fields:
            filtered_cfg_dict["attention_dropout"] = 0.0
        transformerblock_config = TransformerConfig(**filtered_cfg_dict)
        _ensure_parallel_config(transformerblock_config)
        _ensure_layout_initialized(transformerblock_config)
        _force_fp32_for_cpu(transformerblock_config)
        model_config["config"] = transformerblock_config
        self.total_config = model_config
        config = dict()
        for key, value in model_config.items():
            if key != "config":
                config[key] = value
        # Graph 继承 nn.Cell，需要调用 super
        super().__init__()
        self._block_prefix = "node_block_"
        # 如果需要 LanguageModule 的特性，请确保 LanguageModule 兼容 MindSpore 或修改继承关系
        # 这里假设 Graph 是顶层调用
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
        # nums.append(nums[-1]+1)
        self.nodes = dict(zip([id for id in nums], [Node(config=init_config, index=id) for id in nums]))  

        # 修改：保存变异节点信息作为实例属性
        self.mutated_nodes = mutated_nodes if mutated_nodes is not None else {}

        self.embedding = None
        self.rotary_pos_emb = None
        self._rotary_requires_position_ids = False
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

            if self.rotary_pos_emb is not None:
                try:
                    _ = self.rotary_pos_emb(1)
                    self._rotary_requires_position_ids = False
                except TypeError:
                    self._rotary_requires_position_ids = True

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

            # print("++++++++++++++++++++++++++++++++++++=",self.config)

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

    def construct(self, input_ids=None, input_data=None, position_ids=None, debug=True):
        """
        前向传播方法 
        """
        # 修改：使用实例属性中的变异节点信息
        mutated_nodes = self.mutated_nodes
        # MindSpore 自动管理 Device，不需要显式 .to(device)
        
        cur_node = self.nodes[1]
        # 修改：使用传入的张量或者使用默认值
        if input_ids is None:
            # 使用默认值作为后备方案
            values = [
                [5032],
                [39706],
                [24761],
                [14473],
                [35428],
                [1358],
                [20794],
                [6819],
                    ]
            print(f"using default tensors!!!!!!!!!!!!!!!!!!!!!!")
            input_ids = ms.Tensor(values, dtype=ms.int32)
        else:
            # MindSpore stop_gradient 对应 detach
            input_ids = ms.ops.stop_gradient(input_ids)
            
        if input_data is None:
            # 如果没有提供input_data，使用input_ids
            print(f"========input_data is None, using input_ids as input_data")
            input_data = ops.identity(input_ids)
            # MindSpore 不需要显式 requires_grad 除非是 Parameter
        else:
            print(f"using provided input_data!!!!!!!!!!!!!!!!!!!!!!")
            input_data = ms.ops.stop_gradient(input_data)

        # print(f"输入tensor值如下：input_ids为{input_ids}input tensor为{input_data}") 
        
        seq_len = input_data.shape[0]
        attention_mask = ms.ops.zeros((1, 1, seq_len, seq_len), dtype=ms.bool_)

        output = None
        if mutated_nodes:
            print(f"检测到 {len(mutated_nodes)} 个变异节点: {list(mutated_nodes.keys())}")

        max_steps = len(self.nodes) + 2
        for _ in range(max_steps):
            if debug:
                print(f"\n处理节点 {cur_node.id}: {type(cur_node.block)}")
            if input_data is None:
                input_data = input_ids
            if cur_node.block is not None:
                if not callable(cur_node.block):
                    raise TypeError(
                        f"Invalid block for node {cur_node.id}: type={type(cur_node.block)}; "
                        f"block_num={cur_node.block_num}"
                    )
                cur_block = getattr(self, f"{self._block_prefix}{cur_node.id}", cur_node.block)
                if self.is_ascend_available and hasattr(cur_block, "npu"):
                    cur_block = cur_block.npu()
                # 新增：维度检查和调整
                submodule_num = cur_node.block_num
                if submodule_num == 0:
                    input_data = reshape_tensor_nd(input_data,(32,1,cur_node.config.hidden_size))
                    output = cur_block(
                        input_data,
                    )
                elif submodule_num == 1:
                    input_data = reshape_tensor_nd(input_data, (32, 1, cur_node.config.hidden_size))
                    # SelfAttention 期望输入布局为 [seq, batch, hidden]

                    if hasattr(cur_block, "linear_qkv") and not hasattr(cur_block, "_lmsv_qkv_non_none_wrapped"):
                        cur_block.linear_qkv = _NoNoneSecondOutput(cur_block.linear_qkv)
                        cur_block._lmsv_qkv_non_none_wrapped = True

                    def _ensure_non_none_bias(linear_layer):
                        if linear_layer is None:
                            return
                        if getattr(linear_layer, "bias", None) is not None:
                            return
                        output_size = int(getattr(linear_layer, "output_size", cur_node.config.hidden_size))
                        params_dtype = getattr(linear_layer, "params_dtype", ms.float32)
                        zero_bias = Parameter(ops.zeros((output_size,), params_dtype), name="debug_bias")
                        linear_layer.bias = zero_bias

                    _ensure_non_none_bias(getattr(cur_block, "linear_qkv", None))
                    _ensure_non_none_bias(getattr(cur_block, "linear_proj", None))

                    if os.getenv("LMSV_DISABLE_FLASH_ATTN", "0") == "1" and hasattr(cur_block, "use_flash_attention"):
                        cur_block.use_flash_attention = False

                    seq_len = int(input_data.shape[0])
                    attention_mask = _build_attention_mask(seq_len, flash_layout=False, causal=False)
                    print("attention_mask shape:", attention_mask.shape)
                    print("input_data shape before attention:", input_data.shape)

                    rotary_pos_emb = None
                    if getattr(self, "rotary_pos_emb", None) is not None:
                        if self._rotary_requires_position_ids:
                            position_ids = ops.arange(seq_len, dtype=ms.int32).reshape((1, seq_len))
                            rotary_pos_emb = self.rotary_pos_emb(seq_len, position_ids=position_ids)
                        else:
                            rotary_pos_emb = self.rotary_pos_emb(seq_len)

                    actual_seq_len = Tensor([seq_len], dtype=ms.int32)

                    if rotary_pos_emb is not None:
                        res = cur_block(
                            input_data,
                            attention_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            actual_seq_len=actual_seq_len
                        )
                    else:
                        res = cur_block(input_data, attention_mask, actual_seq_len=actual_seq_len)
                    output = res[0] if isinstance(res, tuple) else res
                elif submodule_num == 2:
                    input_data = reshape_tensor_nd(input_data, (32, 1, cur_node.config.hidden_size))
                    num_heads = cur_node.config.num_attention_heads
                    num_q = cur_node.config.num_query_groups
                    kv_channels = cur_node.config.hidden_size // num_heads
                    seq_len, batch_size = input_data.shape[0], input_data.shape[1]

                    # Keep MF/PTA alignment: build q/k/v from the same input tensor without
                    # introducing extra randomly initialized projection layers.
                    q = reshape_tensor_nd(input_data, (seq_len, batch_size, num_heads, kv_channels))
                    k = reshape_tensor_nd(input_data, (seq_len, batch_size, num_q, kv_channels))
                    v = k
                    attention_mask = _build_attention_mask(seq_len, flash_layout=False, causal=False)
                    output = cur_block(q, k, v, attention_mask)
                elif submodule_num == 3:
                    seq_len = input_data.shape[0]
                    input_data = reshape_tensor_nd(input_data, (input_data.shape[0], input_data.shape[1], seq_len,seq_len))
                    attention_mask = _build_attention_mask(seq_len, flash_layout=False, causal=False)
                    cur_block.scale = 1.0
                    cur_block.mask_func = self.attn_mask_func
                    output = cur_block(
                        input_data,
                        attention_mask,
                    )
                elif submodule_num == 4:
                    output = cur_block(
                        input_data,
                    )
                    print("output4:", output)
                    if isinstance(output, tuple):
                        output = output[0]
                elif submodule_num == 5:
                    num_heads = cur_node.config.num_attention_heads
                    kv_channels = cur_node.config.hidden_size // num_heads
                    input_data = reshape_tensor_nd(input_data, (input_data.shape[0], input_data.shape[1], kv_channels * num_heads))
                    res = cur_block(input_data)
                    output = res[0] if isinstance(res, tuple) else res
                elif submodule_num == 6:
                    hidden_size = cur_node.config.hidden_size
                    input_data = reshape_tensor_nd(input_data, (input_data.shape[0], input_data.shape[1], hidden_size))
                    res = cur_block(input_data)
                    output = res[0] if isinstance(res, tuple) else res
                elif submodule_num == 7:
                    hidden_size = cur_node.config.hidden_size
                    input_data = reshape_tensor_nd(input_data, (input_data.shape[0], input_data.shape[1], hidden_size))
                    output = cur_block(
                        input_data,
                    )
                elif submodule_num == 8 or submodule_num == 9:
                    hidden_size = cur_node.config.hidden_size
                    input_data = reshape_tensor_nd(input_data, (input_data.shape[0], input_data.shape[1], hidden_size))
                    res = cur_block(input_data)
                    output = res[0] if isinstance(res, tuple) else res
                elif submodule_num == 10:
                    ffn_hidden = cur_node.config.ffn_hidden_size
                    input_data = reshape_tensor_nd(input_data,(input_data.shape[0],input_data.shape[1],ffn_hidden))
                    res = cur_block(input_data)
                    output = res[0] if isinstance(res, tuple) else res

                input_data = output
                if debug:
                    out_for_shape = output[0] if isinstance(output, tuple) else output
                    print(f"  子模块输出形状: {_torch_size_str(out_for_shape.shape)}")
                   

            if len(cur_node.to_nodes) == 0:
                break
            cur_node = self.nodes[cur_node.to_nodes[0]]

        if debug:
            print(f"\n--- 最终输出处理 ---")
            print(f"最终输出形状: {_torch_size_str(input_data.shape)}")

        return input_data

    # 兼容 forward 调用
    def forward(self, *args, **kwargs):
        return self.construct(*args, **kwargs)

    def set_mutated_nodes(self, mutated_nodes: dict):
        """
        设置或更新变异节点信息
        """
        self.mutated_nodes = mutated_nodes if mutated_nodes is not None else {}

    def get_mutated_nodes(self):
        """
        获取变异节点信息
        """
        return self.mutated_nodes

    def load(self, config_yaml_path: str, config_json_path:str, debug: bool = True):
        """
        从之前生成的yaml配置文件加载图配置
        """
        try:
            if debug:
                print(f"=== 从配置文件加载图: {config_yaml_path} ===")

            # 读取yaml配置文件
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

            # 从配置中提取基础配置和图结构
            if 'base_config' not in loaded_config or 'graph_structure' not in loaded_config:
                raise ValueError("配置文件格式错误：缺少base_config或graph_structure")

            base_config = loaded_config['base_config']
            graph_structure = loaded_config['graph_structure']
            base_config = _sync_hidden_size_with_input(base_config, input_sample=None)
            _normalize_init_method(base_config.get('config', {}))

            # 处理torch数据类型的字符串表示
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
            if "attention_dropout" in valid_fields:
                filtered_cfg_dict["attention_dropout"] = 0.0
            unknown_keys = set(base_config["config"].keys()) - valid_fields
            if debug and unknown_keys:
                print(
                    f"  检测到未识别的 TransformerConfig 字段，已忽略: {sorted(list(unknown_keys))}"
                )

            transformerblock_config = TransformerConfig(**filtered_cfg_dict)
            _ensure_parallel_config(transformerblock_config)
            _ensure_layout_initialized(transformerblock_config)

            # 将过滤后的配置对象写回，保持 total_config 的完整性
            base_config["config"] = transformerblock_config
            self.total_config = base_config

            # 更新父类配置
            self.transformer_config = transformerblock_config
            self.attn_mask_func = _resolve_attention_mask_func(self.transformer_config)
            self.is_ascend_available = ms.hal.is_available("Ascend")
            with open(config_json_path, 'r', encoding='utf-8') as file:
                layer_configs = json.load(file)  # 返回字典或列表
            # 重新创建nodes

            node_ids = list(layer_configs.keys())
            if "block_num_list" in node_ids:
                node_ids.remove("block_num_list")

            if debug:
                print(f"  节点数量: {len(node_ids)-1}")

            # 清空现有的节点
            self.nodes.clear()
            self.mutated_nodes.clear()

            # 节点初始化
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
            _ensure_parallel_config(embedd_config)
            _ensure_layout_initialized(embedd_config)
            # 重建节点
            for node_id, layer_config in layer_configs.items():
                if node_id == "block_num_list" or node_id == "success":
                    continue
                # 确保node_id是整数
                if int(node_id) > 0:
                    print("==========使用变异的decoder配置==========")
                    raw_cfg = layer_configs[node_id]['after']['TransformerConfig']
                    valid_fields = set(TransformerConfig.__dataclass_fields__.keys())
                    filtered_cfg = {k: v for k, v in raw_cfg.items() if k in valid_fields}
                    _normalize_init_method(filtered_cfg)
                    if "attention_dropout" in valid_fields:
                        filtered_cfg["attention_dropout"] = 0.0
                    init_config = TransformerConfig(**filtered_cfg)
                    _ensure_parallel_config(init_config)
                    _ensure_layout_initialized(init_config)
                    _force_fp32_for_cpu(init_config, debug=debug, prefix=f"node {node_id}")
                    #存储失效的cfg字段信息
                    unknown_fields = set(raw_cfg.keys()) - valid_fields
                    if unknown_fields:
                        if debug:
                            print(f"  节点 {node_id} 检测到未识别的 TransformerConfig 字段，已忽略: {sorted(list(unknown_fields))}")
                        self.mutated_nodes[int(node_id)] = {
                            'invalid_fields': sorted(list(unknown_fields)),
                            'raw_config': raw_cfg,
                        }
                else:
                    print("==========使用默认的embedding配置==========")
                    init_config = embedd_config
                _force_fp32_for_cpu(init_config, debug=debug, prefix=f"node {node_id}")
                if isinstance(node_id, str):
                    node_id = int(node_id)
    
                node = Node(config=init_config, index=node_id)
                node.str_op = "mutated_decoder" if node_id > 0 else "embedding"
                node.from_nodes = [node_id-1] if node_id > 0 else []
                node.to_nodes = [node_id+1] if node_id < len(node_ids)-2 else []
                node.params = layer_config.get('params', {})
                node.state = layer_config.get('state', 'none')
                block_num_list = layer_configs.get("block_num_list")
                if isinstance(block_num_list, dict):
                    node.block_num = block_num_list.get(str(node_id), block_num_list.get(node_id, 0))
                elif isinstance(block_num_list, list):
                    node.block_num = block_num_list[node_id] if node_id < len(block_num_list) else 0
                else:
                    node.block_num = 0

                # 设置node的度数
                node.in_degree = len(node.from_nodes)
                node.out_degree = len(node.to_nodes)
                if node.state == 'des':
                    node.out_degree = 1

                # 创建对应的block
                if node.str_op == "embedding":
                    from mindformers.parallel_core.training_graph.base_models.common.embeddings.language_model_embedding import LanguageModelEmbedding
                    node.block = LanguageModelEmbedding(
                        config=self.transformer_config,
                        vocab_size=self.total_config['vocab_size'],
                        max_sequence_length=self.total_config['max_sequence_length'],
                        position_embedding_type=self.total_config['position_embedding_type'],
                    )

                elif "mutated_decoder" in node.str_op.lower() or "decoderlayer" in node.str_op.lower():
                    from mindformers.parallel_core.training_graph.transformer.transformer_block import TransformerBlock
                    from mindformers.parallel_core.training_graph.base_models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

                    _ensure_parallel_config(node.config)
                    _ensure_layout_initialized(node.config)
                    dumped_config_path = _dump_node_config(node_id, node.config)
                    if debug:
                        print(f"  节点 {node_id} 配置已保存: {dumped_config_path}")
                    transformer_block = TransformerBlock(
                        config=node.config,
                        spec=get_gpt_layer_local_spec(
                            None,
                            False,
                            False,
                        ),
                        pre_process=False,
                        post_process=False,
                    )
                    
                    submodule_num = node.block_num
                    if submodule_num == 0:
                        node.block = transformer_block.layers[0].input_layernorm
                    elif submodule_num == 1:
                        node.block = transformer_block.layers[0].self_attention 
                    elif submodule_num == 2:
                        node.block = transformer_block.layers[0].self_attention.core_attention
                    elif submodule_num == 3:
                        core_attn = transformer_block.layers[0].self_attention.core_attention
                        print(f"core_attn下的全部属性:{core_attn.cells_and_names()}")
                        scale_mask = getattr(core_attn, "scale_mask_softmax", None)
                        if callable(scale_mask):
                            node.block = scale_mask
                        else:
                            node.block = core_attn
                            node.block_num = 2
                    elif submodule_num == 4:
                        core_attn = transformer_block.layers[0].self_attention.core_attention
                        node.block = _resolve_dropout_block(core_attn)
                    elif submodule_num == 5:
                        node.block = transformer_block.layers[0].self_attention.linear_proj
                    elif submodule_num == 6:
                        node.block = transformer_block.layers[0].self_attention.linear_qkv
                    elif submodule_num == 7:
                        node.block = transformer_block.layers[0].pre_mlp_layernorm
                    elif submodule_num == 8:
                        node.block = transformer_block.layers[0].mlp
                    elif submodule_num == 9:
                        node.block = transformer_block.layers[0].mlp.linear_fc1
                    elif submodule_num == 10:
                        node.block = transformer_block.layers[0].mlp.linear_fc2

                if node.block is not None:
                    setattr(self, f"{self._block_prefix}{node_id}", node.block)

                self.nodes[node_id] = node

            # 重新初始化其他必要的组件
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
        """重新初始化必要的组件"""
        # 重新设置embedding
        self.embedding = None
        self.rotary_pos_emb = None
        self._rotary_requires_position_ids = False

        # 重新初始化位置编码
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
            try:
                _ = self.rotary_pos_emb(1)
                self._rotary_requires_position_ids = False
            except TypeError:
                self._rotary_requires_position_ids = True


    def get_node(self, index):
        return self.nodes[index]

    def set_op(self, index, operator, state='none'):
        self.nodes[index].set_op(operator, state)

    def set_ops(self, indexes, operator, state='none'):
        for index in indexes:
            self.set_op(index, operator, state)

    def add_edge(self, src: int, des: int):
        r"""add edge from node[src] to node[des] without fix the graph"""
        self.nodes[src].add_to(self.nodes[des])
        self.nodes[des].add_from(self.nodes[src])

    def insert_edge(self, src, des):
        r"""
        insert an edge from node[src] to node[des], and fix the graph
        :return:
        """
        op_add = Node()
        op_add.str_op = 'add'
        op_add.in_degree = 2
        op_add.out_degree = 1
        op_add.id = first_missing_positive([index for index in self.nodes.keys()])
        pre_id_ = random.randint(0, len(self.nodes[des].from_nodes) - 1)
        pre = self.nodes[des].from_nodes[pre_id_]
        op_add.output_shape = self.nodes[pre].output_shape
        op_add.input_shape.append(op_add.output_shape)
        self.insert_node_between(op_add, pre, des)
        self.add_edge(src, op_add.id)
        # self.del_edge(pre, des)
        self.fix_shape(op_add.id)

    def add_edges(self, src, des_nodes):
        for des in des_nodes:
            self.add_edge(src, des)

    def add_node(self, src: Node, des: Node):
        r"""add node des after node src, des doesn't need to link src.to_nodes"""
        self.nodes[des.id] = des
        self.add_edge(src.id, des.id)

    def append_node(self, src: Node, des: Node):
        r"""
        append node des after node src, src link des and des link src.to_nodes
        illustration:
        before append, src--> src_link1, src_link2
        after append, src --> des--> src_link1, src_link2
        """
        des.in_degree = 1
        des.out_degree = src.out_degree
        src.out_degree = 1

        for to_node in src.to_nodes:
            self.add_edge(des.id, to_node)
        for to_node in src.to_nodes:
            self.del_edge(src.id, to_node)
        self.add_node(src, des)

    def insert_node(self, src: int, node: Node):
        r"""
        insert node between node[src] and des(random selected from node[src].to_nodes
        :param src: insert node after src
        :param node: node to be inserted after src
        :return:
        """
        print(f'insert_node: insert {node} after {src}')

        self.nodes[node.id] = node

        to_nodes_id_list = self.nodes[src].to_nodes
        if len(to_nodes_id_list) == 0:
            id = random.randint(0, len(to_nodes_id_list) - 1)
        else:
            id = random.randint(0, len(to_nodes_id_list) - 1)
        des = self.nodes[src].to_nodes[id]
        self.add_edge(src, node.id)
        self.add_edge(node.id, des)
        self.del_edge(src, des)

    def insert_node_between(self, node: Node, id1, id2):
        r"""insert node between node[id1] and node[id2] without fix the graph"""
        self.nodes[node.id] = node
        self.del_edge(id1, id2)
        self.add_edge(id1, node.id)
        self.add_edge(node.id, id2)

    def del_edge(self, src: int, des: int):
        r"""delete edge from node[src] to node[des]"""
        self.nodes[src].del_to(des)
        self.nodes[des].del_from(src)

    def remove_edge(self, src: int, des: int):
        r"""
        remove edge from node[src] to node[des] in logically way
        """
        id = first_missing_positive(list(self.nodes.keys()))
        op_remove_edge = Node(id)
        op_remove_edge.str_op = 'remove_edge_operator'
        op_remove_edge.in_degree = op_remove_edge.out_degree = 1
        op_remove_edge.input_shape.append(self.nodes[src].output_shape)
        op_remove_edge.output_shape = self.nodes[src].output_shape
        self.insert_node_between(op_remove_edge, src, des)

    def del_node(self, index):

        print(f'debug del_node, delete node {index}')

        if index not in self.nodes.keys():
            print(f'graph does not have node:{index}, delete node abolish ')
            return

        self.nodes[index].params = {}

        if self.nodes[index].in_degree > 1:
            self.nodes[index].str_op = 'empty_merge_operator'
        else:
            self.nodes[index].str_op = 'empty_seq_operator'
        self.nodes[index].output_shape = self.nodes[index].input_shape[0]
        for nxt in self.nodes[index].to_nodes:
            self.fix_shape(nxt)
        self.fix_shape(index)

    def fix_shape(self, id):
        r"""
        fix input_shape and output_shape for node[id] and it's pre_nodes
        """
        node = self.nodes[id]
        # output_shape = node.output_shape
        input_shapes = node.input_shape

        for i in range(node.in_degree):
            ''''''
            pre = node.from_nodes[i]
            pre_node = self.nodes[pre]
            # print(f'debug fix pre_node:{pre_node}')
            pre_output_shape = pre_node.output_shape
            batch_size = pre_output_shape[0]
            input_shape = input_shapes[0]  # a little problem here
            if pre_output_shape == input_shape:
                continue
            pre_output_nums = get_element_nums(pre_output_shape)
            input_nums = get_element_nums(input_shape)
            det = pre_output_nums - input_nums

            op_flatten = Node()
            op_flatten.str_op = 'flatten'
            op_flatten.in_degree = op_flatten.out_degree = 1
            op_flatten.input_shape.append(pre_node.output_shape)
            op_flatten.output_shape = [batch_size, pre_output_nums // batch_size]
            op_reshape = Node()
            op_reshape.str_op = 'reshape'
            op_reshape.in_degree = op_reshape.out_degree = 1
            op_reshape.output_shape = input_shape
            op_reshape.params['size'] = op_reshape.output_shape

            if det == 0:
                op_reshape.input_shape.append(pre_node.output_shape)
                op_reshape.id = first_missing_positive([index for index in self.nodes.keys()])
                self.insert_node_between(op_reshape, pre, id)
            elif det < 0:
                op_flatten.id = first_missing_positive([index for index in self.nodes.keys()])
                self.insert_node_between(op_flatten, pre, id)

                op_pad = Node()
                op_pad.str_op = 'pad'
                op_pad.in_degree = 1
                op_pad.out_degree = 1
                op_pad.input_shape.append(op_flatten.output_shape)
                op_pad.output_shape = [batch_size, input_nums // batch_size]
                print(f"op_pad output_shape[-1]:{op_pad.output_shape[-1]}")
                print(f"op_flatten output_shape[-1]:{op_flatten.output_shape[-1]}")
                zeros_len = op_pad.output_shape[-1] - op_flatten.output_shape[-1]
                op_pad.params['pad'] = (0, zeros_len)
                print(f"op_pad.params:{op_pad.params['pad']}")
                op_pad.id = first_missing_positive([index for index in self.nodes.keys()])
                self.insert_node_between(op_pad, op_flatten.id, id)

                op_reshape.input_shape.append(op_pad.output_shape)
                op_reshape.id = first_missing_positive([index for index in self.nodes.keys()])
                self.insert_node_between(op_reshape, op_pad.id, id)
            else:
                '''det>0'''
                op_flatten.id = first_missing_positive([index for index in self.nodes.keys()])
                self.insert_node_between(op_flatten, pre, id)

                op_slice = Node()
                op_slice.str_op = 'slice'
                op_slice.in_degree = op_slice.out_degree = 1
                op_slice.input_shape.append(op_flatten.output_shape)
                op_slice.output_shape = [batch_size, input_nums // batch_size]
                op_slice.params['size'] = [-1, op_slice.output_shape[-1]]
                op_slice.id = first_missing_positive([index for index in self.nodes.keys()])
                self.insert_node_between(op_slice, op_flatten.id, id)

                op_reshape.input_shape.append(op_slice.output_shape)
                op_reshape.id = first_missing_positive([index for index in self.nodes.keys()])
                self.insert_node_between(op_reshape, op_slice.id, id)

    def dup_node(self, index):
        r"""
        dup a node between node[index] and des (des will be random selected from node[index].to_nodes
        :param index:
        :return:
        """
        op_set = [*insert_operators]
        random_id = random.randint(0, len(op_set) - 1)
        random_op = op_set[random_id]
        node_ = Node()
        node_.in_degree = node_.out_degree = 1
        node_.str_op = random_op
        nums = [node.id for node in self.nodes.values()]
        id = first_missing_positive(nums)
        print(f'debug dub_node, id_nums:{nums}, get new id:{id}, name:{node_.str_op}')
        node_.id = id
        node_.input_shape.append(self.nodes[index].output_shape)

        first_input_shape = node_.input_shape[0]
        batch_size = first_input_shape[0]
        if node_.str_op.lower() == 'conv2d':
            node_.params['in_channels'] = first_input_shape[1]
            ch_set = [1, 3, 4, 8, 16, 32, 64, 128, 256]
            in_ch = get_random_num(ch_set)
            out_ch = get_random_num(ch_set)
            height_set = [28, 32, 112]
            height = get_random_num(height_set)

            node_.input_shape[0] = [batch_size, in_ch, height, height]
            node_.output_shape = [batch_size, out_ch, height, height]
            node_.params['in_channels'] = in_ch
            node_.params['out_channels'] = out_ch
        elif node_.str_op.lower() == 'softmax':
            node_.params['dim'] = -1
            node_.params['axis'] = -1
            node_.output_shape = self.nodes[index].output_shape
        elif node_.str_op.lower() == 'sum' or node_.str_op == 'mean':
            node_.output_shape = self.nodes[index].output_shape[:-1]
        elif node_.str_op.lower() == 'flatten':
            node_.output_shape = [batch_size, get_element_nums(first_input_shape) // batch_size]
        elif node_.str_op == 'Qwen2DecoderLayer':
            from pool.Decode import Qwen2Config, generate_random_data_ms
            node_.in_degree = 1
            config = Qwen2Config()
            config.use_cache = False  
            inputs = generate_random_data_ms(config)
            hidden_states = inputs["hidden_states"]  
            attention_mask = inputs["attention_mask"]  
            position_ids = inputs["position_ids"]  
            # node_.input_shape=[hidden_states.shape,attention_mask.shape,position_ids.shape]
            node_.input_shape = [hidden_states.shape]
            node_.output_shape = inputs["hidden_states"].shape
            node_.params['attention_mask'] = attention_mask
            node_.params['position_ids'] = position_ids
            node_.params['config'] = config
        elif node_.str_op == 'QWenBlockDecoderLayer':
            from pool.qwen_decode import generate_random_data, QWenConfig
            node_.in_degree = 1
            config = QWenConfig()
            config.use_cache = False  
            inputs = generate_random_data(config)
            hidden_states = inputs["hidden_states"]  
            attention_mask = inputs["attention_mask"]  
            # node_.input_shape=[hidden_states.shape,attention_mask.shape,position_ids.shape]
            node_.input_shape = [hidden_states.shape]
            node_.output_shape = inputs["hidden_states"].shape
            node_.params['attention_mask'] = attention_mask
            node_.params['config'] = config
        elif node_.str_op == 'CodeLlamaDecoderLayer':
            from pool.codellama_pt import CodeLlamaConfig, generate_random_data_pt
            node_.in_degree = 1
            config = CodeLlamaConfig()
            config.use_cache = False  
            inputs = generate_random_data_pt(config)
            hidden_states = inputs["hidden_states"]  
            attention_mask = inputs["attention_mask"]  
            position_ids = inputs["position_ids"]  
            # node_.input_shape=[hidden_states.shape,attention_mask.shape,position_ids.shape]
            node_.input_shape = [hidden_states.shape]
            node_.output_shape = inputs["hidden_states"].shape
            node_.params['attention_mask'] = attention_mask
            node_.params['position_ids'] = position_ids
            node_.params['config'] = config
        elif node_.str_op == 'BaiChuanDecoderLayer':
            from pool.baichuan import BaiChuanConfig, DecoderLayer
            node_.in_degree = 1
            config = BaiChuanConfig()
            batch_size = 1
            seq_length = 1  
            hidden_size = config.hidden_size
            input_shape = (batch_size, seq_length, hidden_size)
            node_.input_shape = [input_shape]
            node_.params['config'] = config
        elif node_.str_op == 'ChatGLMDecoderLayer':
            from pool.GLMBlock_pt import ChatGLMConfig, generate_random_data
            node_.in_degree = 1
            config = ChatGLMConfig()
            config.use_cache = False  
            inputs = generate_random_data(config)
            hidden_states = inputs["hidden_states"]  
            attention_mask = inputs["attention_mask"]  
            rotary_pos_emb = inputs["rotary_pos_emb"]  
            # node_.input_shape=[hidden_states.shape,attention_mask.shape,position_ids.shape]
            node_.input_shape = [hidden_states.shape]
            node_.output_shape = inputs["hidden_states"].shape
            node_.params['attention_mask'] = attention_mask
            node_.params['rotary_pos_emb'] = rotary_pos_emb
            node_.params['config'] = config
        elif node_.str_op == 'Qwen2DecoderLayer':
            from pool.Qwen_Decode_pt import Qwen2Config, generate_random_data
            node_.in_degree = 1
            config = Qwen2Config()
            inputs = generate_random_data(config)
            hidden_states = inputs["hidden_states"]  
            attention_mask = inputs["attention_mask"]  
            position_ids = inputs["position_ids"]  
            position_embeddings = inputs["position_embeddings"]  

            # node_.input_shape=[hidden_states.shape,attention_mask.shape,position_ids.shape]
            node_.input_shape = [hidden_states.shape]
            node_.output_shape = inputs["hidden_states"].shape
            node_.params['attention_mask'] = attention_mask
            node_.params['position_ids'] = position_ids
            node_.params['position_embeddings'] = position_embeddings
            node_.params['config'] = config
        elif node_.str_op == 'CogVLMDecoderLayer':
            from pool.CogVLM_Decode import CogVLMConfig, generate_dummy_input
            node_.in_degree = 1
            config = CogVLMConfig()
            inputs = generate_dummy_input(config)
            # 假设 generate_dummy_input 返回 ms tensor，如果不是则需要转换
            hidden_states = inputs["hidden_states"]
            attention_mask = inputs["attention_mask"]  
            token_type_ids = inputs["token_type_ids"]
            output_attentions = inputs["output_attentions"]
            position_ids = inputs["position_ids"]  

            node_.input_shape = [hidden_states.shape]
            node_.output_shape = inputs["hidden_states"].shape
            node_.params = {
                "token_type_ids": token_type_ids,
                'position_ids': position_ids,
                'attention_mask': attention_mask,
                'output_attentions': output_attentions,
                'use_cache': False,
                'config': config
            }
        elif node_.str_op == "ChatGLM3DecoderLayer":
            from pool.chatglm_modeling_torch_modules import GlmConfig
            node_.in_degree = 1
            config = GlmConfig(
                hidden_size=128,
                num_attention_heads=4,
                num_key_value_heads=2,
                intermediate_size=256,
                hidden_act="silu",
                attention_dropout=0.1,
                rms_norm_eps=1e-6
            )
            attention_mask = None
            batch_size = 2
            seq_length = 12
            
            hidden_states = ms.ops.StandardNormal()((batch_size, seq_length, config.hidden_size))
            node_.input_shape = [hidden_states.shape]
            node_.output_shape = hidden_states.shape
            node_.params = {
                'attention_mask': attention_mask,
                'config': config
            }
        elif node_.str_op == "MixtralDecoderlayer":
            from pool.mixtral_deconstruction import MixtralConfig
            node_.in_degree = 1
            config = MixtralConfig(
                vocab_size=32000,
                hidden_size=1024,
                intermediate_size=14336,
                num_hidden_layers=32,
                num_attention_heads=32,
                num_key_value_heads=8,
                max_position_embeddings=4096 * 32,
                use_cache=False  
            )
            batch_size = 1
            seq_length = 3
            hidden_size = config.hidden_size
            
            dtype = ms.float32

            # 模拟位置编码生成
            position_embeddings = (
                ms.ops.StandardNormal()((batch_size, seq_length, hidden_size // config.num_attention_heads)).astype(dtype),
                ms.ops.StandardNormal()((batch_size, seq_length, hidden_size // config.num_attention_heads)).astype(dtype)
            )

            attention_mask = ms.ops.ones((batch_size, seq_length), dtype=ms.float32)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  
            hidden_states = ms.ops.StandardNormal()((batch_size, seq_length, config.hidden_size))
            node_.input_shape = [hidden_states.shape]
            node_.output_shape = hidden_states.shape
            node_.params = {
                'attention_mask': attention_mask,
                'position_embeddings': position_embeddings,
                'output_attentions': False,
                'output_router_logits': False,
                'use_cache': False,
                'config': config
            }
        elif node_.str_op == "Grok1DecoderLayer":
            from pool.grok1_Decode import Grok1Config, generate_dummy_input
            config = Grok1Config()
            node_.in_degree = 1
            inputs = generate_dummy_input(config)
            hidden_states = inputs["hidden_states"]
            attention_mask = inputs["attention_mask"]  
            position_ids = inputs["position_ids"]  

            node_.input_shape = [hidden_states.shape]
            node_.output_shape = inputs["hidden_states"].shape

            node_.params = {
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "output_attentions": False,
                "use_cache": False,
            }
        elif node_.str_op == "llavaDecoderLayer":
            from pool.llava_pt import LlavaConfig, generate_random_data
            node_.in_degree = 1
            vision_config = type('VisionConfig', (object,), {'hidden_size': 1024})()  
            text_config = type('TextConfig', (object,), {'hidden_size': 1024})()  
            config = LlavaConfig(vision_config=vision_config, text_config=text_config)
            
            hidden_states = ms.ops.StandardNormal()((1, 3, 1024))
            node_.input_shape = [hidden_states.shape]
            node_.output_shape = hidden_states.shape
            node_.params = {
                "config": config
            }
        elif node_.str_op == "YiDecoderLayer":
            from pool.yi_modeling_torch_modules import LlamaConfig
            node_.in_degree = 1
            config = LlamaConfig(
                vocab_size=32000,
                hidden_size=512,
                intermediate_size=1024,
                num_hidden_layers=4,
                num_attention_heads=8,
                max_position_embeddings=512,
            )
            position_ids = ms.ops.arange(3).unsqueeze(0).expand(2, -1)
            hidden_states = ms.ops.StandardNormal()((2, 3, config.hidden_size))
            node_.input_shape = [hidden_states.shape]
            node_.output_shape = hidden_states.shape
            node_.params = {
                'attention_mask': None,
                'position_ids': position_ids,
                'past_key_value': None,
                'output_attentions': False,
                'use_cache': False,
                'cache_position': None,
                'config': config
            }
        else:
            node_.output_shape = self.nodes[index].output_shape

        # node_.output_shape = self.nodes[index].output_shape
        to_nodes = self.nodes[index].to_nodes
        id_ = random.randint(0, len(to_nodes) - 1)
        to_id = to_nodes[id_]
        self.insert_node_between(node_, index, to_id)
        # self.display()
        self.fix_shape(node_.id)
        self.fix_shape(to_id)

    def mutate_shape(self, index):
        r"""
        exchange the last two dimension of input data,
        for example, when mutate_shape node a in a-->b,
        the graph becomes a-->transpose-->b
        :param index: the id of node
        :return:
        """
        print(f"debug mutate_shape, node id:{index}")
        output_shape = self.nodes[index].output_shape
        if len(output_shape) != 4:
            print('the length of output_shape < 4, mutate_shape abolish!')
            return
        if output_shape[-1] != output_shape[-2]:
            print(f'the last two dimensions are not equal, mutate_shape abolish!')
            return
        for nxt in self.nodes[index].to_nodes:
            new_id = first_missing_positive([node.id for node in self.nodes.values()])
            node_trans = Node(new_id)
            node_trans.in_degree = node_trans.out_degree = 1
            node_trans.output_shape = output_shape
            node_trans.input_shape.append(output_shape)
            node_trans.str_op = 'transpose'
            node_trans.params['dim0'] = 3
            node_trans.params['dim1'] = 2
            self.insert_node_between(node_trans, index, nxt)

    def mutate_params(self, index):
        r"""
        mutate the params of node index
        if the output shape of the node changes, fix the shape
        a-->b ==> a-->empty_node-->b, then fix shape of empty
        :param index:
        :return:
        """
        if len(self.nodes[index].params) == 0:
            print(f'there is no params in node')
            return
        tmp_output_shape = self.nodes[index].output_shape
        name = self.nodes[index].str_op.lower()
        if name == 'conv2d':
            self.nodes[index].params['kernel_size'] = 1
            self.nodes[index].params['stride'] = 1
            self.nodes[index].params['padding'] = 0
            self.nodes[index].output_shape = self.nodes[index].input_shape[0]

            for nxt in self.nodes[index].to_nodes:
                new_id = first_missing_positive([node.id for node in self.nodes.values()])
                node_empty = Node(new_id)
                node_empty.str_op = 'empty_seq_operator'
                node_empty.in_degree = node_empty.out_degree = 1
                node_empty.output_shape = tmp_output_shape
                node_empty.input_shape.append(node_empty.output_shape)
                self.insert_node_between(node_empty, index, nxt)
                self.fix_shape(nxt)

    def get_src(self):
        for i in self.nodes.keys():
            if self.nodes[i].state == 'src':
                return self.nodes[i]

    def get_des(self):
        for i in self.nodes.keys():
            if self.nodes[i].state == 'des':
                return self.nodes[i]

    def display(self):
        print('display graph:')
        for i in self.nodes.keys():
            print("id:" + str(self.nodes[i].id) + ", layer:" + str(self.nodes[i].str_op) + ", from:" + str(
                self.nodes[i].from_nodes) +
                  ", to:" + str(self.nodes[i].to_nodes))
        for i in self.nodes.keys():
            print("param" + str(self.nodes[i].id) + ":", self.nodes[i].params)

    def get_graph(self):
        g = []
        for i in self.nodes.keys():
            g.append(str(self.nodes[i]))
        return g

    def __len__(self):
        return len(self.nodes)
