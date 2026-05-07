import os

os.environ["PYTORCH_NPU_FORCE_FALLBACK"] = "1"

import mindspeed.megatron_adaptor as ma

import copy
import sys
import torch
import torch_npu
from ruamel.yaml import YAML
from megatron.core import tensor_parallel
from megatron.core import parallel_state
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_layer import TransformerConfig
from megatron.core.transformer.utils import attention_mask_func
from megatron.core.models.common.language_module.language_module import LanguageModule

from megatron.core.models.common.embeddings.rotary_pos_embedding import (
    # MultimodalRotaryEmbedding,
    RotaryEmbedding,
)
# from megatron.core.inference.contexts import StaticInferenceContext

# distributed settings
import sys
from argparse import ArgumentParser
from mindspeed.arguments import process_args
import megatron.training.global_vars as global_vars


sys.path.append(".")
sys.path.append("..")

from utils.runtime import common_utils, model_helpers
from utils.runtime.OperatorSet import insert_operators
from utils.runtime.debug_utils import (
    debug_parameter_summary,
    debug_scalar,
    debug_tensor_summary,
    mark_weights_logged,
    should_log_full,
    should_log_heavy,
    should_log_weights_once,
)
import random
import torch
import json
def fill_zeros_with_nonzero(tensor):
    if tensor.dim() == 0:
        return tensor
    
    # 创建一个掩码，标记非零元素的位置
    nonzero_mask = tensor != 0
    
    # 如果没有非零元素，直接返回原张量（无法填充）
    if not nonzero_mask.any():
        return tensor
    
    # 创建一个与原始张量形状相同的坐标网格
    coords = torch.meshgrid(*[torch.arange(dim, device=tensor.device) for dim in tensor.shape], indexing='ij')
    coords = torch.stack(coords, dim=-1).float()
    
    # 获取非零元素的坐标
    nonzero_coords = coords[nonzero_mask]
    nonzero_values = tensor[nonzero_mask]
    
    # 获取零元素的坐标
    zero_mask = ~nonzero_mask
    zero_coords = coords[zero_mask]
    
    if zero_coords.numel() == 0:
        return tensor  # 没有零元素需要填充
    
    # 对于每个零元素坐标，找到最近的非零元素坐标
    # 使用广播计算所有零元素到所有非零元素的距离
    distances = torch.cdist(zero_coords, nonzero_coords, p=2)
    
    # 找到每个零元素最近的邻居索引
    nearest_indices = torch.argmin(distances, dim=1)
    
    # 用最近的非零元素值填充零元素
    result = tensor.clone()
    result[zero_mask] = nonzero_values[nearest_indices]
    
    return result

def reshape_tensor_nd(
        input_tensor: torch.Tensor,
        target_shape: tuple,
        fill_value: float = 0
) -> torch.Tensor:
    """
    将任意维度的输入张量调整为任意目标形状，支持填充/裁剪/维度增减。

    Args:
        input_tensor: 输入张量（如 3D 的 (x,y,z)）
        target_shape: 目标形状（如 4D 的 (a,b,c,d)）
        fill_value: 填充时使用的值（默认为0）

    Returns:
        输出张量，形状严格等于 target_shape
    """
    input_shape = input_tensor.shape
    input_numel = input_tensor.numel()
    target_numel = torch.prod(torch.tensor(target_shape)).item()

    # Step 1: 将输入展平为1D向量
    flattened = input_tensor.flatten()

    # Step 2: 处理元素数量差异
    if target_numel > input_numel:
        # 填充不足部分
        padded = torch.cat([
            flattened,
            torch.full((target_numel - input_numel,), fill_value, dtype=input_tensor.dtype, device=input_tensor.device)
        ])
        output = padded
    elif target_numel < input_numel:
        # 裁剪多余部分
        output = flattened[:target_numel]
    else:
        output = flattened

    # Step 3: 调整为目标形状
    output = output.reshape(target_shape)
    return output


def _reshape_parallel_input(input_tensor: torch.Tensor, module, target_shape: tuple) -> torch.Tensor:
    """Adapt isolated submodule inputs to local TP shard expectations when needed."""
    output = reshape_tensor_nd(input_tensor, target_shape)
    weight = getattr(module, "weight", None)
    if weight is None or not hasattr(weight, "shape") or len(weight.shape) < 2:
        return output

    expected_last_dim = int(weight.shape[-1])
    current_last_dim = int(output.shape[-1])
    if expected_last_dim <= 0 or current_last_dim == expected_last_dim:
        return output

    module_name = type(module).__name__
    is_row_parallel = module_name == "RowParallelLinear" or bool(getattr(module, "input_is_parallel", False))
    if is_row_parallel and current_last_dim > expected_last_dim:
        tp_world_size = 1
        tp_rank = 0
        try:
            tp_world_size = max(1, int(parallel_state.get_tensor_model_parallel_world_size()))
            tp_rank = max(0, int(parallel_state.get_tensor_model_parallel_rank()))
        except Exception:
            tp_world_size = 1
            tp_rank = 0

        if current_last_dim == expected_last_dim * tp_world_size:
            start = tp_rank * expected_last_dim
            end = start + expected_last_dim
            return output[..., start:end].contiguous()

    fixed_shape = tuple(list(output.shape[:-1]) + [expected_last_dim])
    return reshape_tensor_nd(output, fixed_shape)


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
        self.block_num = None
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


def transfor_shape(output):  # (1,8,1024) -> (32,3,896)
    output = output.transpose(0, 1)  # ??(1,8,1024)???(8,1,1024)
    # ????????1024??????896?
    linear = torch.nn.Linear(1024, 896, device="npu")
    output = linear(output)  # ???(8,1,896)

    # ??????δ?С?????г???
    output = output.repeat(4, 2, 1)  # ??(8,1,896)???(32,2,896)
    return output


class Graph(LanguageModule):

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
            if 'config' in model_config and model_config['config']['init_method'] == "torch.nn.init.xavier_uniform_":
                model_config['config']['init_method'] = torch.nn.init.xavier_uniform_
        elif config_path is not None:
            # 使用配置文件路径初始化
            config_path = model_helpers.resolve_repo_path(config_path)
            yaml = YAML()
            with open(config_path, 'r', encoding='utf-8') as file:
                model_config = yaml.load(file)
                if model_config['config']['init_method'] == "torch.nn.init.xavier_uniform_":
                    model_config['config']['init_method'] = torch.nn.init.xavier_uniform_
        else:
            raise ValueError("必须提供 config_path 或 config_dict 中的一个")

        # del model_config["config"]["multi_latent_attention"]
        transformerblock_config = TransformerConfig(**model_config["config"])
        model_config["config"] = transformerblock_config
        self.total_config = model_config
        config = dict()
        for key, value in model_config.items():
            if key != "config":
                config[key] = value
        super().__init__(config=transformerblock_config)
        self.node_blocks = torch.nn.ModuleDict()
        # 从model_config提取配置，避免硬编码默认值导致与tp=8不兼容
        base_cfg = model_config.get("config", {})

        def _cfg_get(cfg, key, default):
            if isinstance(cfg, dict):
                return cfg.get(key, default)
            return getattr(cfg, key, default)

        init_config = TransformerConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            num_layers=_cfg_get(base_cfg, "num_layers", 24),
            hidden_size=_cfg_get(base_cfg, "hidden_size", 896),
            ffn_hidden_size=_cfg_get(base_cfg, "ffn_hidden_size", 4864),
            num_attention_heads=_cfg_get(base_cfg, "num_attention_heads", 16),  # 默认改为16以兼容tp=8
            num_query_groups=_cfg_get(base_cfg, "num_query_groups", 2),
            attention_dropout=_cfg_get(base_cfg, "attention_dropout", 0.0),
            init_method_std=_cfg_get(base_cfg, "init_method_std", 0.01),
            hidden_dropout=_cfg_get(base_cfg, "hidden_dropout", 0.0),
            normalization=_cfg_get(base_cfg, "normalization", "RMSNorm"),
            layernorm_epsilon=_cfg_get(base_cfg, "layernorm_epsilon", 1e-6)
        )
        # nums.append(nums[-1]+1)
        self.nodes = dict(zip([id for id in nums], [Node(config=init_config, index=id) for id in nums]))  # ?????????

        # 修改：保存变异节点信息作为实例属性
        self.mutated_nodes = mutated_nodes if mutated_nodes is not None else {}

        self.embedding = None
        if self.total_config['position_embedding_type'] == 'rope' and not self.config.multi_latent_attention:
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=self.total_config['rotary_percent'],
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=self.total_config['seq_len_interpolation_factor'],
                rotary_base=self.total_config['rotary_base'],
                rope_scaling=self.total_config['rope_scaling'],
                rope_scaling_factor=self.total_config['rope_scaling_factor'],
                use_cpu_initialization=self.config.use_cpu_initialization,
            )
        # elif self.total_config['position_embedding_type'] == 'mrope' and not self.config.multi_latent_attention:
        #     self.rotary_pos_emb = MultimodalRotaryEmbedding(
        #         kv_channels=self.config.kv_channels,
        #         rotary_percent=self.total_config['rotary_percent'],
        #         rotary_interleaved=self.config.rotary_interleaved,
        #         seq_len_interpolation_factor=self.total_config['seq_len_interpolation_factor'],
        #         rotary_base=self.total_config['rotary_base'],
        #     )
        #     self.mrope_section = self.config.mrope_section
        #     assert (
        #         self.mrope_section is not None
        #     ), "mrope require mrope_section setting, but we got None from TransformerConfig"
        #     self.rotary_pos_emb_cache = {}
        self.pre_process = True
        self.post_process = True
        if self.post_process:

            if self.config.defer_embedding_wgrad_compute:
                # The embedding activation buffer preserves a reference to the input activations
                # of the final embedding projection layer GEMM. It will hold the activations for
                # all the micro-batches of a global batch for the last pipeline stage. Once we are
                # done with all the back props for all the microbatches for the last pipeline stage,
                # it will be in the pipeline flush stage. During this pipeline flush we use the
                # input activations stored in embedding activation buffer and gradient outputs
                # stored in gradient buffer to calculate the weight gradients for the embedding
                # final linear layer.
                self.embedding_activation_buffer = []
                self.grad_output_buffer = []
            else:
                self.embedding_activation_buffer = None
                self.grad_output_buffer = None

            self.share_embeddings_and_output_weights = False

            if self.config._cpu_offloading_context == 'None':
                self.config._cpu_offloading_context = None

            # print("++++++++++++++++++++++++++++++++++++=",self.config)

            self.output_layer = tensor_parallel.ColumnParallelLinear(
                self.config.hidden_size,
                self.total_config['vocab_size'],
                config=self.config,
                init_method=self.config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=False,
                skip_weight_param_allocation=self.pre_process
                                             and self.share_embeddings_and_output_weights,
                embedding_activation_buffer=self.embedding_activation_buffer,
                grad_output_buffer=self.grad_output_buffer,
            )

    def forward(self, input_ids=None, input_data=None, position_ids=None, debug=True):
        """
        前向传播方法

        Args:
            input_ids: 输入token IDs张量，如果为None则使用默认值
            input_data: 输入数据张量，如果为None则使用默认值  
            position_ids: 位置ID张量，如果为None则使用默认值
            debug: 是否打印调试信息

        Returns:
            最终的logits输出
        """
        # 修改：使用实例属性中的变异节点信息
        mutated_nodes = self.mutated_nodes
        device = torch.device("npu" if torch.npu.is_available() else "cpu")
        cur_node = self.nodes[1]
        layer_count = 0

        def emit_layer_summary(name, tensor):
            if should_log_full() or should_log_heavy():
                debug_tensor_summary(name, tensor, max_items=8, include_stats=True)

        def find_first_param(module, preferred_keywords):
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
            return fallback if fallback is not None else (None, None)

        def emit_weight_debug_once():
            if not should_log_weights_once():
                return
            embedding_name, embedding_weight = None, None
            attn_name, attn_weight = None, None
            mlp_name, mlp_weight = None, None
            lm_head_name, lm_head_weight = find_first_param(getattr(self, "output_layer", None), ["output_layer", "lm_head"])
            for node_id in sorted(self.nodes.keys()):
                block = getattr(self.nodes.get(node_id), "block", None)
                if block is None:
                    continue
                if embedding_weight is None:
                    embedding_name, embedding_weight = find_first_param(block, ["embedding", "word"])
                if attn_weight is None:
                    attn_name, attn_weight = find_first_param(block, ["attention", "attn", "qkv"])
                if mlp_weight is None:
                    mlp_name, mlp_weight = find_first_param(block, ["mlp", "fc1", "gate", "up_proj"])
            debug_parameter_summary("weight.embedding", embedding_weight, max_items=8)
            if embedding_name is not None:
                debug_scalar("weight.embedding_name", embedding_name)
            debug_parameter_summary("weight.first_attention", attn_weight, max_items=8)
            if attn_name is not None:
                debug_scalar("weight.first_attention_name", attn_name)
            debug_parameter_summary("weight.first_mlp", mlp_weight, max_items=8)
            if mlp_name is not None:
                debug_scalar("weight.first_mlp_name", mlp_name)
            debug_tensor_summary("weight.final_norm", None, max_items=8, include_stats=True)
            debug_parameter_summary("weight.lm_head", lm_head_weight, max_items=8)
            if lm_head_name is not None:
                debug_scalar("weight.lm_head_name", lm_head_name)
            mark_weights_logged()
        
        # 修改：使用传入的张量或者使用默认值
        if input_ids is None:
            # 使用默认值作为后备方案
            values = [[0.5032], [3.9706], [2.4761], [1.4473], 
              [0.35428], [0.1358], [0.20794], [0.6819]]
            print(f"using default tensors!!!!!!!!!!!!!!!!!!!!!!\n")
            input_ids = torch.tensor(values, dtype=torch.float32, device=device)
        else:
            input_ids = input_ids.clone().detach().to(device)
            
        if input_data is None:
            # 如果没有提供input_data，使用input_ids
            input_data = input_ids.clone().requires_grad_(True)
        else:
            input_data = input_data.clone().detach().to(device).requires_grad_(True)
        print(f"输入tensor值如下：input_ids为{input_ids}input tensor为{input_data}") 
        emit_weight_debug_once()
        emit_layer_summary("embedding.output", input_data)
        # input_ids.device = device 
        seq_len = input_data.shape[0]
        attention_mask = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.bool, device=device)

        output = None
        if mutated_nodes:
            print(f"检测到 {len(mutated_nodes)} 个变异节点: {list(mutated_nodes.keys())}")

        while True:
            if debug:
                print(f"\n处理节点 {cur_node.id}: {type(cur_node.block)}")
            if input_data == None:
                input_data = input_ids
            if cur_node.block is not None:
                cur_block = cur_node.block.npu() if torch.npu.is_available() else cur_node.block
                # node_info = mutated_nodes[cur_node.id]
                print(f"  执行子模块 {type(cur_node.block)}):")
                if layer_count == 0:
                    emit_layer_summary("block0.input", input_data)

                # 新增：维度检查和调整
                submodule_num = cur_node.block_num
                if submodule_num == 0:
                    input_data = reshape_tensor_nd(input_data,(32,1,cur_node.config.hidden_size))
                    output = cur_block(
                        input_data,
                    )
                elif submodule_num == 1:
                    input_data = reshape_tensor_nd(input_data, (32, 1, cur_node.config.hidden_size))
                    seq_len = input_data.shape[0]
                    attention_mask = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.bool, device=device)
                    output = cur_block(
                        input_data,
                        attention_mask
                    )[0]
                elif submodule_num == 2:
                    input_data = reshape_tensor_nd(input_data, (32, 1, cur_node.config.hidden_size))
                    seq_len = input_data.shape[0]
                    attention_mask = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.bool, device=device)

                    # DotProductAttention expects per-TP-partition q/k/v shapes.
                    num_heads = int(getattr(cur_block, "num_attention_heads_per_partition", cur_node.config.num_attention_heads))
                    num_q = int(getattr(cur_block, "num_query_groups_per_partition", cur_node.config.num_query_groups))
                    kv_channels = int(
                        getattr(
                            cur_block,
                            "hidden_size_per_attention_head",
                            max(1, cur_node.config.hidden_size // max(1, cur_node.config.num_attention_heads)),
                        )
                    )
                    if num_q <= 0:
                        num_q = 1
                    if num_heads <= 0:
                        num_heads = 1
                    if num_heads % num_q != 0:
                        num_q = 1

                    local_hidden = int(getattr(cur_block, "hidden_size_per_partition", num_heads * kv_channels))
                    input_data = reshape_tensor_nd(input_data, (input_data.shape[0], input_data.shape[1], local_hidden))
                    q = reshape_tensor_nd(input_data, (input_data.shape[0], input_data.shape[1], num_heads, kv_channels))
                    k = reshape_tensor_nd(input_data, (input_data.shape[0], input_data.shape[1], num_q, kv_channels))
                    v = k
                    output = cur_block(
                        q, k, v,
                        attention_mask,
                        None,
                        None,
                    )
                elif submodule_num == 3:
                    seq_len = input_data.shape[0]
                    input_data = reshape_tensor_nd(input_data, (input_data.shape[0], input_data.shape[1], seq_len,seq_len))
                    attention_mask = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.bool, device=device)
                    cur_block.scale = 1.0
                    cur_block.mask_func = attention_mask_func
                    output = cur_block(
                        input_data,
                        attention_mask,
                    )
                elif submodule_num == 4:
                    output = cur_block(
                        input_data,
                    )
                    print("output4:",output)
                elif submodule_num == 5:
                    num_heads = cur_node.config.num_attention_heads
                    kv_channels = cur_node.config.hidden_size // num_heads
                    input_data = _reshape_parallel_input(
                        input_data,
                        cur_block,
                        (input_data.shape[0], input_data.shape[1], kv_channels * num_heads),
                    )
                    output = cur_block(
                        input_data,
                    )[0]
                elif submodule_num == 6:
                    hidden_size = cur_node.config.hidden_size
                    input_data = _reshape_parallel_input(
                        input_data,
                        cur_block,
                        (input_data.shape[0], input_data.shape[1], hidden_size),
                    )
                    output = cur_block(
                        input_data,
                    )[0]
                elif submodule_num == 7:
                    hidden_size = cur_node.config.hidden_size
                    input_data = reshape_tensor_nd(input_data, (input_data.shape[0], input_data.shape[1], hidden_size))
                    output = cur_block(
                        input_data,
                    )
                elif submodule_num == 8 or submodule_num == 9:
                    hidden_size = cur_node.config.hidden_size
                    input_data = _reshape_parallel_input(
                        input_data,
                        cur_block,
                        (input_data.shape[0], input_data.shape[1], hidden_size),
                    )
                    output = cur_block(
                        input_data,
                    )[0]
                elif submodule_num == 10:
                    ffn_hidden = cur_node.config.ffn_hidden_size
                    input_data = _reshape_parallel_input(
                        input_data,
                        cur_block,
                        (input_data.shape[0], input_data.shape[1], ffn_hidden),
                    )
                    output = cur_block(
                        input_data,
                    )[0]

                if debug:
                    print(f"  模块输出形状: {output.shape}")
                if layer_count == 0:
                    emit_layer_summary("block0.output", output)
                elif layer_count == 1:
                    emit_layer_summary("block1.output", output)
                if len(cur_node.to_nodes) == 0:
                    emit_layer_summary("last_block.output", output)
                layer_count += 1
                   

            if len(cur_node.to_nodes) == 0:
                break
            cur_node = self.nodes[cur_node.to_nodes[0]]

        if debug:
            print(f"\n--- 最终输出处理 ---")
            print(f"最终输出形状: {output.shape}")
        debug_tensor_summary("final_norm.output", None, max_items=8, include_stats=True)
        emit_layer_summary("lm_head.input", output)
        emit_layer_summary("logits", output)

        return output

    def set_mutated_nodes(self, mutated_nodes: dict):
        """
        设置或更新变异节点信息

        Args:
            mutated_nodes: 变异节点信息字典
        """
        self.mutated_nodes = mutated_nodes if mutated_nodes is not None else {}

    def get_mutated_nodes(self):
        """
        获取变异节点信息

        Returns:
            dict: 变异节点信息字典
        """
        return self.mutated_nodes

    def load(self, config_yaml_path: str, config_json_path:str, debug: bool = True):
        """
        从之前生成的yaml配置文件加载图配置

        Args:
            config_yaml_path: 配置文件路径（例如：demo_graph_forward_n_nodes_configs/mutated_config_iter_001.yaml）
            debug: 是否打印调试信息

        Returns:
            bool: 加载是否成功
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

            # 重新初始化Graph的配置
            if base_config['config']['init_method'] == "torch.nn.init.xavier_uniform_":
                base_config['config']['init_method'] = torch.nn.init.xavier_uniform_

            # 处理torch数据类型的字符串表示
            if 'autocast_dtype' in base_config['config'] and isinstance(base_config['config']['autocast_dtype'], str):
                if base_config['config']['autocast_dtype'] == 'torch.float16':
                    base_config['config']['autocast_dtype'] = torch.float16
                elif base_config['config']['autocast_dtype'] == 'torch.float32':
                    base_config['config']['autocast_dtype'] = torch.float32
                elif base_config['config']['autocast_dtype'] == 'torch.half':
                    base_config['config']['autocast_dtype'] = torch.half

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

            # 将过滤后的配置对象写回，保持 total_config 的完整性
            base_config["config"] = transformerblock_config
            self.total_config = base_config

            # 更新父类配置
            self.config = transformerblock_config
            with open(config_json_path, 'r', encoding='utf-8') as file:
                layer_configs = json.load(file)  # 返回字典或列表
            # 重新创建nodes

            node_ids = list(layer_configs.keys())
            node_ids.remove("block_num_list")

            if debug:
                print(f"  节点数量: {len(node_ids)-1}")

            # 清空现有的节点
            self.nodes.clear()
            self.mutated_nodes.clear()
            self.node_blocks = torch.nn.ModuleDict()

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
            # 重建节点
            for node_id, layer_config in layer_configs.items():
                if node_id == "block_num_list" or node_id == "success":
                    continue
                # 确保node_id是整数
                init_config = TransformerConfig(**layer_configs[node_id]['after']['TransformerConfig']) if int(node_id) > 0 else embedd_config
                if int(node_id) > 0:
                    tp_size = int(getattr(init_config, "tensor_model_parallel_size", 1) or 1)
                    try:
                        from megatron.core import parallel_state as _parallel_state

                        if _parallel_state.model_parallel_is_initialized():
                            runtime_tp = int(_parallel_state.get_tensor_model_parallel_world_size() or 1)
                            tp_size = max(tp_size, runtime_tp)
                    except Exception:
                        pass

                    num_q_groups = int(getattr(init_config, "num_query_groups", 0) or 0)
                    num_heads = int(getattr(init_config, "num_attention_heads", 0) or 0)

                    # Guard against mutated configs that break SelfAttention construction at runtime TP.
                    if tp_size > 0:
                        if num_heads > 0 and num_heads % tp_size != 0:
                            init_config.num_attention_heads = tp_size
                            num_heads = tp_size
                        if num_q_groups <= 0:
                            num_q_groups = num_heads if num_heads > 0 else tp_size
                        if num_q_groups % tp_size != 0:
                            init_config.num_query_groups = num_heads if num_heads > 0 and num_heads % tp_size == 0 else tp_size
                if isinstance(node_id, str):
                    node_id = int(node_id)
    
                node = Node(config=init_config, index=node_id)
                node.str_op = "mutated_decoder" if node_id > 0 else "embedding"
                node.from_nodes = [node_id-1] if node_id > 0 else []
                node.to_nodes = [node_id+1] if node_id < len(node_ids)-2 else []
                node.params = layer_config.get('params', {})
                node.state = layer_config.get('state', 'none')
                node.block_num = layer_configs["block_num_list"][node_id]

                # 设置node的度数
                node.in_degree = len(node.from_nodes)
                node.out_degree = len(node.to_nodes)
                if node.state == 'des':
                    node.out_degree = 1

                # 创建对应的block
                if node.str_op == "embedding":
                    from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
                    node.block = LanguageModelEmbedding(
                        config=self.config,
                        vocab_size=self.total_config['vocab_size'],
                        max_sequence_length=self.total_config['max_sequence_length'],
                        position_embedding_type=self.total_config['position_embedding_type'],
                    )

                elif "mutated_decoder" in node.str_op.lower() or "decoderlayer" in node.str_op.lower():
                    from megatron.core.transformer.transformer_block import TransformerBlock
                    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

                    transformer_block = TransformerBlock(
                        config=node.config,
                        spec=get_gpt_layer_local_spec(
                            None,
                            False,
                            False,
                        ),
                        pre_process=True,
                        post_process=True,
                    )
                    submodule_num = node.block_num
                    if submodule_num == 0:
                        node.block = transformer_block.layers[0].input_layernorm
                        # input:(x,y,z)
                        # input_tensor,
                        # output=input
                    elif submodule_num == 1:
                        node.block = transformer_block.layers[0].self_attention  #
                        # input:[seq_length, batch_size, hidden_size]
                        # hidden_states = input_tensor,
                        # attention_mask = attention_mask
                        # output[0],[seq_length, batch_size, hidden_size]
                    elif submodule_num == 2:
                        node.block = transformer_block.layers[0].self_attention.core_attention
                        # input: q(seq_length, batch_size, num_heads, kv_channels),k,v        attn_mask_type = None,PackedSeqParams = None,
                        #        k=v(seq_length, batch_size, num_query_groups, kv_channels)
                        # num_attention_heads == num_query_groups
                        # output,(x,y,h)
                    elif submodule_num == 3:
                        node.block = transformer_block.layers[0].self_attention.core_attention.scale_mask_softmax
                        # input: (x,y,sq,sq)
                        # output,(x,y,sq,sq)
                    elif submodule_num == 4:
                        node.block = transformer_block.layers[0].self_attention.core_attention.attention_dropout
                        # input: (x,y,z,...)
                        # output = input
                    elif submodule_num == 5:
                        node.block = transformer_block.layers[0].self_attention.linear_proj
                        # input: (seq_length, batch_size,kv_channels*num_attention_heads)
                        # output[0],[seq_length, batch_size, hidden_size]
                    elif submodule_num == 6:
                        node.block = transformer_block.layers[0].self_attention.linear_qkv
                        # input: (seq_length, batch_size,hidden_size)
                        # output[0],[seq_length, batch_size, ?]
                    elif submodule_num == 7:
                        node.block = transformer_block.layers[0].pre_mlp_layernorm
                        # input:(x,y,z)
                        # input_tensor,
                        # output[0],(y,z)
                    elif submodule_num == 8:
                        node.block = transformer_block.layers[0].mlp
                        # input:(x,y,z)
                        # output[0],(x,y,z) =input
                    elif submodule_num == 9:
                        node.block = transformer_block.layers[0].mlp.linear_fc1
                        # input: (seq_length, batch_size,hidden_size)
                        # output[0],[seq_length, batch_size, ?]
                    elif submodule_num == 10:
                        node.block = transformer_block.layers[0].mlp.linear_fc2
                        # input: (seq_length, batch_size,ffn_hidden_size)
                        # output[0],[seq_length, batch_size, hidden_size]

                if node.block is not None and isinstance(node.block, torch.nn.Module):
                    self.node_blocks[str(node_id)] = node.block
                    node.block = self.node_blocks[str(node_id)]

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

        # 重新初始化位置编码
        if self.total_config['position_embedding_type'] == 'rope' and not self.config.multi_latent_attention:
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=self.total_config['rotary_percent'],
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=self.total_config['seq_len_interpolation_factor'],
                rotary_base=self.total_config['rotary_base'],
                rope_scaling=self.total_config['rope_scaling'],
                rope_scaling_factor=self.total_config['rope_scaling_factor'],
                use_cpu_initialization=self.config.use_cpu_initialization,
            )
        # elif self.total_config['position_embedding_type'] == 'mrope' and not self.config.multi_latent_attention:
        #     self.rotary_pos_emb = MultimodalRotaryEmbedding(
        #         kv_channels=self.config.kv_channels,
        #         rotary_percent=self.total_config['rotary_percent'],
        #         rotary_interleaved=self.config.rotary_interleaved,
        #         seq_len_interpolation_factor=self.total_config['seq_len_interpolation_factor'],
        #         rotary_base=self.total_config['rotary_base'],
        #     )
        #     self.mrope_section = self.config.mrope_section
        #     self.rotary_pos_emb_cache = {}

        # 重新设置处理标志
        self.pre_process = True
        self.post_process = True

        # 重新初始化输出层
        if self.post_process:
            if self.config.defer_embedding_wgrad_compute:
                self.embedding_activation_buffer = []
                self.grad_output_buffer = []
            else:
                self.embedding_activation_buffer = None
                self.grad_output_buffer = None

            self.share_embeddings_and_output_weights = False

            if self.config._cpu_offloading_context == 'None':
                self.config._cpu_offloading_context = None

            self.output_layer = tensor_parallel.ColumnParallelLinear(
                self.config.hidden_size,
                self.total_config['vocab_size'],
                config=self.config,
                init_method=self.config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=False,
                skip_weight_param_allocation=self.pre_process
                                             and self.share_embeddings_and_output_weights,
                embedding_activation_buffer=self.embedding_activation_buffer,
                grad_output_buffer=self.grad_output_buffer,
            )

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
            # print(f'debug to_nodes_id_list,graph:')
            # self.display()
            # print(f'debug to_nodes_id_list,node:{self.nodes[src]}')
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
        :param src:
        :param des:
        :return:
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
        take a-->b as example, if the output_shape of a doesn't match input_shape
        of b, it's likely to fix the graph as a-->flatten-->concat/slice-->b by
        fix_shape(b)
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
            config.use_cache = False  # ????????????????
            inputs = generate_random_data_ms(config)
            hidden_states = inputs["hidden_states"]  # ?????????????
            attention_mask = inputs["attention_mask"]  # ???????
            position_ids = inputs["position_ids"]  # ???????
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
            config.use_cache = False  # ????????????????
            inputs = generate_random_data(config)
            hidden_states = inputs["hidden_states"]  # ?????????????
            attention_mask = inputs["attention_mask"]  # ???????
            # node_.input_shape=[hidden_states.shape,attention_mask.shape,position_ids.shape]
            node_.input_shape = [hidden_states.shape]
            node_.output_shape = inputs["hidden_states"].shape
            node_.params['attention_mask'] = attention_mask
            node_.params['config'] = config
        elif node_.str_op == 'CodeLlamaDecoderLayer':
            from pool.codellama_pt import CodeLlamaConfig, generate_random_data_pt
            node_.in_degree = 1
            config = CodeLlamaConfig()
            config.use_cache = False  # ????????????????
            inputs = generate_random_data_pt(config)
            hidden_states = inputs["hidden_states"]  # ?????????????
            attention_mask = inputs["attention_mask"]  # ???????
            position_ids = inputs["position_ids"]  # ???????
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
            seq_length = 1  # ????????????
            hidden_size = config.hidden_size
            input_shape = (batch_size, seq_length, hidden_size)
            node_.input_shape = [input_shape]
            node_.params['config'] = config
        elif node_.str_op == 'ChatGLMDecoderLayer':
            from pool.GLMBlock_pt import ChatGLMConfig, generate_random_data
            node_.in_degree = 1
            config = ChatGLMConfig()
            config.use_cache = False  # ????????????????
            inputs = generate_random_data(config)
            hidden_states = inputs["hidden_states"]  # ?????????????
            attention_mask = inputs["attention_mask"]  # ???????
            rotary_pos_emb = inputs["rotary_pos_emb"]  # ???????
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
            hidden_states = inputs["hidden_states"]  # ?????????????
            attention_mask = inputs["attention_mask"]  # ???????
            position_ids = inputs["position_ids"]  # ???????
            position_embeddings = inputs["position_embeddings"]  # ???????

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
            # import torch
            # device = "cuda" if torch.cuda.is_available() else "cpu"
            # inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            hidden_states = inputs["hidden_states"]
            attention_mask = inputs["attention_mask"]  # ???????
            token_type_ids = inputs["token_type_ids"]
            output_attentions = inputs["output_attentions"]
            position_ids = inputs["position_ids"]  # ???????

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
            import torch
            hidden_states = torch.randn(batch_size, seq_length, config.hidden_size)
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
                use_cache=False  # ????????????????
            )
            batch_size = 1
            seq_length = 3
            hidden_size = config.hidden_size
            import torch
            device = "cpu"
            dtype = torch.float32

            # ????λ????????
            position_embeddings = (
                torch.randn(batch_size, seq_length, hidden_size // config.num_attention_heads, device=device,
                            dtype=dtype),
                torch.randn(batch_size, seq_length, hidden_size // config.num_attention_heads, device=device,
                            dtype=dtype)
            )

            # ????????????????????????
            attention_mask = torch.ones(batch_size, seq_length, device=device, dtype=torch.float32)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # ???????
            hidden_states = torch.randn(batch_size, seq_length, config.hidden_size)
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
            attention_mask = inputs["attention_mask"]  # ???????
            position_ids = inputs["position_ids"]  # ???????

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
            vision_config = type('VisionConfig', (object,), {'hidden_size': 1024})()  # ?????????????
            text_config = type('TextConfig', (object,), {'hidden_size': 1024})()  # ?????????????
            config = LlavaConfig(vision_config=vision_config, text_config=text_config)
            import torch
            hidden_states = torch.randn(1, 3, 1024)
            node_.input_shape = [hidden_states.shape]
            node_.output_shape = hidden_states.shape
            node_.params = {
                "config": config
            }
        elif node_.str_op == "YiDecoderLayer":
            from pool.yi_modeling_torch_modules import LlamaConfig
            node_.in_degree = 1
            import torch
            config = LlamaConfig(
                vocab_size=32000,
                hidden_size=512,
                intermediate_size=1024,
                num_hidden_layers=4,
                num_attention_heads=8,
                max_position_embeddings=512,
            )
            position_ids = torch.arange(3).unsqueeze(0).expand(2, -1)
            hidden_states = torch.randn(2, 3, config.hidden_size)
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

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Include unregistered node.block parameters used by subgraph execution."""
        if destination is None:
            destination = {}

        super().state_dict(destination, prefix, keep_vars)

        for node_id, node in self.nodes.items():
            if hasattr(node, 'block') and node.block is not None:
                node_state_dict = node.block.state_dict(
                    prefix=f'{prefix}nodes.{node_id}.',
                    keep_vars=keep_vars,
                )
                destination.update(node_state_dict)

        return destination

    def load_state_dict(self, state_dict, strict=True):
        """Load both registered graph parameters and unregistered node.block parameters."""
        result = super().load_state_dict(state_dict, strict=False)

        for node_id, node in self.nodes.items():
            if hasattr(node, 'block') and node.block is not None:
                node_prefix = f'nodes.{node_id}.'
                node_state_dict = {
                    key.replace(node_prefix, ''): value
                    for key, value in state_dict.items()
                    if key.startswith(node_prefix)
                }
                if node_state_dict:
                    node.block.load_state_dict(node_state_dict, strict=False)

        return result

    def __len__(self):
        return len(self.nodes)
