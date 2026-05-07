import mindspeed.megatron_adaptor as ma
import sys
from argparse import ArgumentParser
from typing import Dict, Any
import numpy as np
import torch_npu
from mindspeed.arguments import process_args
from megatron.training.arguments import parse_args, validate_args
from megatron.training.global_vars import set_global_variables
from megatron.training.arguments import core_transformer_config_from_args
import torch.distributed as dist
from megatron.training import get_args
from megatron.training.initialize import _initialize_distributed,_init_autoresume,_set_random_seed,_compile_dependencies,_initialize_tp_communicators


def add_extra_args(parser):
    """Add custom arguments for mutation system"""
    # Add arguments that this script uses
    parser.add_argument("-c", "--configs", type=str, help="The path to the configs dir")
    parser.add_argument("-n", "--node-num", type=int, default = 1, help="nodes num")
    parser.add_argument("-r", "--rounds", type=int, default = 10, help="mutating rounds")
    parser.add_argument("--mutnm", type=int, default = 2, help="mutating num")
    parser.add_argument("-m", "--module", type=str, help="The targeted single module")
    parser.add_argument("--sub", type=str, help="The list of submodule num")
    parser.add_argument("--path", type=str, help="The mutation results path")
    parser.add_argument("--load-path", type=str, help="The path of the graph config to load")
    parser.add_argument("--args_path", type=str, help="The path of the mutation arguments yaml")
    return parser



def finish_mpu_init():
    args = get_args()
    _initialize_distributed(None,None)
    if args.rank == 0:
        print("> setting random seeds to {} ...".format(args.seed))
    _set_random_seed(args.seed, args.data_parallel_random_init)
    
args = parse_args(extra_args_provider=add_extra_args)
validate_args(args, {})


set_global_variables(args,False)
args = get_args()
finish_mpu_init()
_init_autoresume()
_compile_dependencies()

if args.tp_comm_overlap:
    _initialize_tp_communicators()


def _normalize_graph_node_indices(graph):
    """Convert legacy 1-based graph indices to 0-based only when needed."""
    node_keys = sorted(graph.nodes.keys())
    if not node_keys:
        return
    if node_keys[0] == 0:
        return
    if node_keys[0] != 1 or node_keys != list(range(1, len(node_keys) + 1)):
        raise ValueError(f"unexpected graph node keys: {node_keys}")

    remapped_nodes = {}
    for old_key in node_keys:
        node = graph.nodes[old_key]
        node.id = old_key - 1
        node.from_nodes = [src - 1 for src in node.from_nodes]
        node.to_nodes = [dst - 1 for dst in node.to_nodes]
        remapped_nodes[node.id] = node
    graph.nodes = remapped_nodes
    
def seed_all(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch_npu.npu.manual_seed_all(seed)
    torch_npu.npu.manual_seed(seed)

from megatron.training import get_args

import torch
import json
import random
import os
from ruamel.yaml import YAML
import copy
import time
import sys

from utils.runtime import common_utils, model_helpers
from utils.runtime.core.withnum_mutation_system import ConfigMutator
from utils.runtime.core.graph import Graph, Node
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding


yaml = YAML()
TEMPLATE_CONFIG_PATH = model_helpers.resolve_repo_path("assets/runtime/configs/template_config.yaml")
STRUCTURE_CONFIG_PATH = model_helpers.resolve_repo_path("assets/runtime/configs/structure_config.yaml")

config_dir = ""
res_dir = ""
mutating_rate = 0.3

# params in one iteration
iteration = 0
# store this mutation iteration info
mutating_record = {}
record_file_name = ""
no_mutating = False
module = args.module
node_num = args.node_num
config_dir = args.configs
# rounds = 100
rounds = args.rounds
# mutating_num = 2
mutating_num = args.mutnm
mutating_res_dir = args.path
def load_random_model_configs(count: int = 2):
    """
    随机选择模型配置文件
    
    Args:
        count: 选择的配置文件数量
        
    Returns:
        List[Dict]: 加载的配置列表
    """
    configs = []
    file_names = []
    
    if module:
        # 分割路径（兼容逗号、空格、分号分隔）
        module_paths = [p.strip() for p in module.split(',') if p.strip()]
        
        for path in module_paths:
            try:
                file_name = os.path.basename(path)
                print(f"正在加载模块文件：{path}")
                
                with open(path, 'r', encoding='utf-8') as f:
                    config = yaml.load(f)  # 使用安全加载
                    config['_source_file'] = file_name
                    configs.append(config)
                    file_names.append(file_name)
                    
            except FileNotFoundError:
                print(f"警告：文件不存在，跳过 {path}")
            except Exception as e:
                print(f"YAML解析错误（{path}）：{str(e)}")
        return configs, file_names
        
    # 获取所有可用的配置文件
    available_configs = []
    if os.path.exists(config_dir):
        for file in os.listdir(config_dir):
            if file.endswith('.yaml') and file != 'note.txt':
                available_configs.append(file)
    
    if len(available_configs) < count:
        raise ValueError(f"需要至少{count}个配置文件，但只找到 {len(available_configs)} 个")
    
    # 随机选择配置文件
    selected_files = random.sample(available_configs, count)
    print(f"随机选择的配置文件: {selected_files}")
    
    # 加载配置
    for config_file in selected_files:
        config_path = os.path.join(config_dir, config_file)
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.load(f)
            config['_source_file'] = config_file  # 记录来源文件
            configs.append(config)
    
    return configs, selected_files


def extract_transformer_config_from_yaml(yaml_config: dict) -> dict:
    """
    从YAML配置中提取TransformerConfig参数
    
    Args:
        yaml_config: YAML配置字典
        
    Returns:
        dict: TransformerConfig参数字典
    """
    if 'TransformerConfig' in yaml_config:
        base_config = yaml_config['TransformerConfig'].copy()
    else:
        # 使用默认配置
        base_config = {
            'tensor_model_parallel_size': 1,
            'pipeline_model_parallel_size': 1,
            'num_layers': 24,
            'hidden_size': 896,
            'ffn_hidden_size': 4864,
            'num_attention_heads': 14,
            'num_query_groups': 2,
            'attention_dropout': 0.0,
            'init_method_std': 0.01,
            'hidden_dropout': 0.0,
            'normalization': "RMSNorm",
            'layernorm_epsilon': 1e-6
        }
    
    # 确保包含必需的字段
    required_fields = {
        'tensor_model_parallel_size': 1,
        'pipeline_model_parallel_size': 1,
        'init_method_std': 0.01,
        'normalization': "RMSNorm",
        'layernorm_epsilon': 1e-6,
        'attention_dropout': 0.0,
        'hidden_dropout': 0.0,
    }
    
    for field, default_value in required_fields.items():
        if field not in base_config:
            base_config[field] = default_value
    
    # 修复数据类型问题
    # 移除可能导致问题的字段
    problematic_fields = [
        'params_dtype', 'bf16', 'fp16', 'attention_softmax_in_fp32', 
        'masked_softmax_fusion', 'sequence_parallel', 'gated_linear_unit',
        'multi_latent_attention'
    ]
    
    for field in problematic_fields:
        if field in base_config:
            # print(f"  移除可能有问题的字段: {field} = {base_config[field]}")
            del base_config[field]
    
    # 检查MoE配置并处理add_bias_linear冲突
    # 根据TransformerConfig.__post_init__()中的断言：assert not self.add_bias_linear, "Bias is not supported for MoE"
    moe_indicators = ['num_moe_experts', 'moe_router_topk', 'moe_grouped_gemm', 'moe_ffn_hidden_size']
    has_moe = any(field in base_config and base_config[field] is not None and base_config[field] != 0 for field in moe_indicators)
    
    if has_moe:
        print(f"  检测到MoE配置，强制设置 add_bias_linear = False")
        base_config['add_bias_linear'] = False
    elif 'add_bias_linear' not in base_config:
        # 如果没有MoE且没有明确设置，默认为False以避免其他问题
        base_config['add_bias_linear'] = False
    
    # 确保数值类型正确
    numeric_fields = {
        'tensor_model_parallel_size': int,
        'pipeline_model_parallel_size': int,
        'num_layers': int,
        'hidden_size': int,
        'ffn_hidden_size': int,
        'num_attention_heads': int,
        'num_query_groups': int,
        'attention_dropout': float,
        'hidden_dropout': float,
        'init_method_std': float,
        'layernorm_epsilon': float,
    }
    
    for field, expected_type in numeric_fields.items():
        if field in base_config:
            try:
                base_config[field] = expected_type(base_config[field])
            except (ValueError, TypeError) as e:
                print(f"  警告: 无法转换字段 {field} = {base_config[field]} 到 {expected_type.__name__}")
                # 使用默认值
                if field in required_fields:
                    base_config[field] = required_fields[field]
    
    # 确保num_query_groups存在且合理
    if 'num_query_groups' not in base_config:
        # 如果没有指定，使用num_attention_heads
        base_config['num_query_groups'] = base_config.get('num_attention_heads', 2)
    
    # 确保num_query_groups不超过num_attention_heads
    if base_config['num_query_groups'] > base_config['num_attention_heads']:
        base_config['num_query_groups'] = base_config['num_attention_heads']
    
    return base_config


def create_mutated_graph_with_random_configs(configs, selected_files, num_decoder_nodes: int = 2,
                                           mutated_node_ids: list = None,
                                           mutation_num: int = 3):
    """
    使用随机选择的模型配置创建包含变异decoder的图
    
    Args:
        num_decoder_nodes: decoder节点的数量
        mutated_node_ids: 要变异的节点ID列表，如果为None则变异所有decoder节点
        mutation_rate: 变异概率
        
    Returns:
        Tuple[Graph, Dict, List]: 图对象、变异节点配置、选择的配置文件
    """    
    # args = get_args()
    # config = core_transformer_config_from_args(args)
    
    
    # new configs after this iteration
    new_configs = []
    
    # store this mutation iteration info
    global mutating_record
    mutating_record = {}
    
    # 动态创建图配置
    network_nodes = []
    
    # add the first embedding node
    network_nodes.append({
        "id": 0,
        "name": "embedding",
        "from": [],
        "to": [1] if num_decoder_nodes > 0 else [],
        "params": {},
        "state": "src",
        "layer_limits": ["embedding"]
    })
    
    # add decoder nodes
    for i in range(num_decoder_nodes):
        node_id = i + 1
        from_nodes = [node_id - 1]  # 从前一个节点接收输入
        to_nodes = [node_id + 1] if i < num_decoder_nodes - 1 else []  # 连接到下一个节点
        state = "des" if i == num_decoder_nodes - 1 else "none"  # 最后一个节点标记为destination
        
        network_nodes.append({
            "id": node_id,
            "name": "Qwen2DecoderLayer",
            "from": from_nodes,
            "to": to_nodes,
            "params": {},
            "state": state,
            "layer_limits": ["Qwen2DecoderLayer"]
        })
    
    # 创建完整的图配置
    graph_config = {"network": network_nodes}
    
    print(f"\n创建包含 {num_decoder_nodes} 个decoder节点的图")
    print(f"节点连接: embedding(0) -> " + " -> ".join([f"decoder{i}({i+1})" for i in range(num_decoder_nodes)]))
    
    # 保存临时JSON文件
    with open("temp_graph.json", "w") as f:
        json.dump(graph_config, f)
    
    # 读取JSON配置
    with open("temp_graph.json", "r") as f:
        output_data = json.load(f)
        g = output_data['network']
    
    # 创建基础图
    graph = Graph(
        config_path=TEMPLATE_CONFIG_PATH,
        nums=[i for i in range(len(g))]
    )
    # print(graph.total_config)
    
    # 创建变异器 - 使用新的合并配置系统
    mutator = ConfigMutator(
        structure_config_path=STRUCTURE_CONFIG_PATH,
        template_config_path=TEMPLATE_CONFIG_PATH,
        output_dir=res_dir,
        config_dir=config_dir
    )
    mutated_nodes = {}
    
    # 为每次forward迭代保存变异配置
    iteration_id = getattr(create_mutated_graph_with_random_configs, '_iteration_counter', 0) + 1
    setattr(create_mutated_graph_with_random_configs, '_iteration_counter', iteration_id)
    
    # 如果没有指定要变异的节点，默认变异所有decoder节点
    if mutated_node_ids is None:
        mutated_node_ids = list(range(1, num_decoder_nodes + 1))
    
    print(f"将对节点 {mutated_node_ids} 进行变异")
    
    # 处理每个节点
    for node_ in g:
        node_id = node_['id']
        mutating_record[node_id] = {"mutated": False, "before": {}, "after": {}}
        config_index = (node_id - 1) % len(configs)
        mutating_record[node_id]["before"] = configs[config_index]
        config_after = configs[config_index]
        
        # extract TransformerConfig
        if node_id > 0:  # decoder节点
            # 循环使用配置文件
            base_config = extract_transformer_config_from_yaml(configs[config_index])
            source_file = selected_files[config_index]
        else:
            # embedding节点使用默认配置
            base_config = {
                'tensor_model_parallel_size': 1,
                'pipeline_model_parallel_size': 1,
                'num_layers': 24,
                'hidden_size': 896,
                'ffn_hidden_size': 4864,
                'num_attention_heads': 14,
                'num_query_groups': 2,
                'attention_dropout': 0.0,
                'init_method_std': 0.01,
                'hidden_dropout': 0.0,
                'normalization': "RMSNorm",
                'layernorm_epsilon': 1e-6
            }
            source_file = "default"
        
        # create TransformerConfig object
        init_config = TransformerConfig(**base_config)
        
        # create Node
        node = Node(config=init_config, index=node_id)
        node.from_nodes = node_['from']
        node.to_nodes = node_['to']
        node.str_op = node_['name']
        node.params = node_['params']
        node.state = node_['state']
        
        # 处理layer_limits
        if 'layer_limits' in node_:
            node.layer_limits = node_['layer_limits']
            if node.layer_limits[0] != "none":
                import random
                random_num = random.randrange(len(node.layer_limits))
                node.str_op = node.layer_limits[random_num]
                
                # 检查是否需要变异此节点
                should_mutate = (node_id in mutated_node_ids) and (not no_mutating)
                is_last_decoder = (node_id == num_decoder_nodes)  # 最后一个decoder节点
                
                if node.str_op == "Qwen2DecoderLayer" and should_mutate:
                    mutating_record[node_id]["mutated"] = True
                    print(f"\n对节点 {node_id} 应用变异 (基于 {source_file})...")
                    
                    # 保存当前迭代的变异配置到文件（仅在第一个变异节点时保存）
                    if node_id == min(mutated_node_ids):
                        print(f"\n--- 保存第 {iteration_id} 轮变异配置 ---")
                        try:
                            saved_config, config_filepath = mutator.create_and_save_mutated_config(
                                iteration=iteration_id, 
                                mutation_num=mutation_num
                            )
                            print(f"已保存变异配置到: {config_filepath}")
                        except Exception as e:
                            print(f"保存变异配置失败: {e}")
                    
                    # mutate config
                    mutated_config_dict = mutator.mutate_config_dict(
                        base_config=base_config,
                        mutation_num=mutation_num,
                        is_last_decoder=is_last_decoder,
                        graph_hidden_size=graph.config.hidden_size
                    )
                    # record info: diff, after
                    diff = common_utils.compare_dicts(before=base_config, after=mutated_config_dict)
                    mutating_record[node_id]["diff"] = diff
                    common_utils.print_diff(diff)
                    mutated_config = TransformerConfig(**mutated_config_dict)
                    all_config_copy = copy.deepcopy(configs[config_index])
                    all_config_copy["TransformerConfig"] = mutated_config_dict
                    mutating_record[node_id]["after"] = all_config_copy
                    config_after = all_config_copy
                    
                    node.config = mutated_config
                    node.str_op = 'mutated_decoder'
                    mutated_nodes[node_id] = {
                        'config': mutated_config,
                        'source_file': source_file,
                        'original_config': base_config
                    }
                    
                elif node.str_op == "Qwen2DecoderLayer":
                    # 使用原始配置
                    print(f"\n节点 {node_id} 使用原始配置 (基于 {source_file})")
                    node.config = init_config
                
                # 创建TransformerBlock（参考generate_graph.py）
                if ('decoderlayer' in node.str_op.lower() or 'mutated_decoder' in node.str_op.lower()):
                    node.block = TransformerBlock(
                        config=node.config,
                        spec=get_gpt_layer_local_spec(
                            None,
                            False,
                            False,
                            # False,
                            # False,
                            # normalization="RMSNorm",
                        ),
                        pre_process=True,
                        post_process=True,
                        # vp_stage=None,
                    )
                
                elif  "embedding" in node.str_op.lower():
                    # 创建embedding层（参考generate_graph.py）
                    node.config = graph.total_config
                    node.block = LanguageModelEmbedding(
                        config=node.config['config'],
                        vocab_size=node.config['vocab_size'],
                        max_sequence_length=node.config['max_sequence_length'],
                        position_embedding_type=node.config['position_embedding_type'],
                    )
        
        # 设置节点属性
        node.in_degree = len(node.from_nodes)
        node.out_degree = len(node.to_nodes)
        if node.state == 'des':
            node.out_degree = 1
        
        graph.nodes[node.id] = node
        
        if node_id > 0:
            new_configs.append(config_after)
    
    # 修改：在所有节点处理完成后，设置完整的mutated_nodes到graph中
    graph.set_mutated_nodes(mutated_nodes)

    return graph, mutated_nodes, selected_files, new_configs


def demo_forward_with_random_configs(configs, selected_files, num_decoder_nodes: int = 2):
    global mutating_record
    """演示使用随机配置和变异的图的forward过程"""
    try:
        # 创建包含随机配置变异decoder的图
        all_decoder_ids = list(range(1, num_decoder_nodes + 1))
        graph, mutated_nodes, selected_files, new_configs = create_mutated_graph_with_random_configs(
            configs=configs,
            selected_files=selected_files,
            num_decoder_nodes=num_decoder_nodes,
            mutated_node_ids=all_decoder_ids,  # 变异所有decoder节点
            mutation_num=mutating_num
        )
        
        print(f"\n图创建完成，节点数: {len(graph.nodes)}")
        print(f"变异节点: {list(mutated_nodes.keys())}")
        print(f"使用的配置文件: {selected_files}")
#         from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
#         from megatron.core.distributed import DistributedDataParallelConfig
#         from megatron.core.distributed import DistributedDataParallel as DDP
        
#         ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
#         model_graph = [graph]
#         model = [DDP(graph.total_config["config"], ddp_config, model_chunk, disable_bucketing=(model_chunk_idx > 0))
#          for (model_chunk_idx, model_chunk) in enumerate(model_graph)]
#         optimizer_config = OptimizerConfig(
#             optimizer='adam',
#             lr=1e-4,
#             weight_decay=0.01,
#         )
#         optimizer = get_megatron_optimizer(
#             config=optimizer_config,
#             model_chunks=model
#         )           
        # 执行forward过程 - 修改：直接调用graph.forward()方法
        print("\n--- 执行Forward过程 ---")
        
        # 直接调用graph的forward方法（mutated_nodes已在初始化时设置）
        final_output = graph.forward(debug=True)
        print(f"\nForward过程完成！最终输出形状: {final_output.shape}")
        mutating_record["success"]=True
        loss = final_output.norm()
        
#         import mindspore as ms
#         from mindspore import grad, ops
#         def forward_fn(x):
#             return ops.norm(x)
        
#         gradient = grad(forward_fn)(loss)
        # loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()  

        print("loss计算结果：",loss)
        
        # output_file = "output1.pt"
        # torch.save(final_output, output_file)
        # print(f"最终输出已保存到文件: {output_file}")
        
        # clear temp files
        if os.path.exists("temp_graph.json"):
            os.remove("temp_graph.json")
        
        return selected_files, new_configs
        
    except Exception as e:
        print(f"✗ Forward演示失败: {e}")
        
        # record err info to file and console
        import traceback
        traceback.print_exc()
        error_stack = traceback.format_exc()
        mutating_record["success"]=False
        mutating_record["err_stack"]=error_stack
        
        return None

def print_npu_memory(tag=""):
    try:
        # 获取NPU当前分配的显存
        allocated = torch.npu.memory_allocated() / 1024 / 1024  # 单位 MB

        print(f"[NPU显存][{tag}] 当前分配: {allocated:.2f} MB")
    except Exception as e:
        print(f"[NPU显存][{tag}] 记录失败: {e}")

class Logger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)  # 输出到控制台
        self.log.write(message)       # 写入日志文件

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def generate_layer_config(node_count: int) -> Dict[str, Any]:
    """
    生成指定节点数量的层配置

    Args:
        node_count: 总节点数量

    Returns:
        包含所有节点配置的字典
    """
    if node_count < 2:
        raise ValueError("节点数量必须至少为2")

    config = {"LayerConfig": {}}

    # 节点1: Embedding (起始节点)
    config["LayerConfig"][1] = {
        "name": "Embedding",
        "params": {},
        "state": "src",
        "layer_limits": ["embedding"],
        "to": [2],
        "from": [],
        "layer_nums": 1
    }

    # 节点2: Decoderlayer1
    config["LayerConfig"][2] = {
        "name": "Decoderlayer1",
        "params": {},
        "state": "none",
        "layer_limits": ["Qwen2DecoderLayer"],
        "to": [],
        "from": [1],
        "layer_nums": 1
    }

    # 生成节点3到n的配置
    for i in range(3, node_count + 1):
        config["LayerConfig"][i] = {
            "name": f"Decoderlayer{i - 1}",
            "params": {},
            "state": "none",
            "layer_limits": ["Qwen2DecoderLayer"],
            "to": [],
            "from": [],
            "layer_nums": 1
        }

    # 修正所有节点的to和from
    _fix_connections(config, node_count)

    return config


def _fix_connections(config: Dict[str, Any], node_count: int) -> None:
    """
    修正所有节点的连接关系

    Args:
        config: 配置字典
        node_count: 总节点数量
    """
    # 修正to字段
    for i in range(1, node_count):
        current_node = i
        next_node = i + 1
        config["LayerConfig"][current_node]["to"] = [next_node]

    # 最后一个节点的to为空列表
    config["LayerConfig"][node_count]["to"] = []

    # 修正from字段
    for i in range(2, node_count + 1):
        current_node = i
        prev_node = i - 1
        config["LayerConfig"][current_node]["from"] = [prev_node]


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    保存配置到YAML文件

    Args:
        config: 配置字典
        output_path: 输出文件路径
    """
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.preserve_quotes = True

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f)
        
        
        
if __name__ == "__main__":
    # parse arguments
    # module = "assets/runtime/model_config/baichuan.yaml" 
    # node_num = 1
    # config_dir = "model_config"
    # rounds = 2
    # mutating_num = 2
    
#     pipeline_dtype = torch.bfloat16
#     tensor_model_parallel_size = args.tensor_model_parallel_size
#     pipeline_model_parallel_size = args.pipeline_model_parallel_size

#     with open('assets/runtime/configs/template_config.yaml', 'r') as file:
#         data = yaml.load(file)  # 解析YAML内容为字典
#         data['config']['tensor_model_parallel_size'] = tensor_model_parallel_size
#         data['config']['pipeline_model_parallel_size'] = pipeline_model_parallel_size
#         data['config']['pipeline_dtype'] = "torch.bfloat16"
#     print(data)
        
#     with open('assets/runtime/configs/template_config.yaml', 'w') as file:
#         yaml.dump(data, file)
    
    # if args.module:
    #     node_num = 1
    # else:
    #     node_num = args.node_num
    # rounds = args.round
    # mutating_num = args.mutnm
    # config_dir = args.configs
    seed_all()
    common_utils.print_seperate_line()
    print(f"开始{rounds}次随机模块组合变异（模块数={node_num}）")
    common_utils.print_seperate_line()
    
    # prepare dir
    
    # if args.module:
    #     filename = os.path.basename(args.module).split('.')[0]
    #     res_dir = os.path.join("./res", f"{common_utils.gen_timestamp_h()}_{filename}")
    # else:
    #     res_dir = os.path.join("./res", f"{common_utils.gen_timestamp_h()}_random{node_num}nodes")
    if module:
        filename = os.path.basename(module).split('.')[0]
        res_dir = os.path.join("./res", f"{filename}")
    else:
        res_dir = os.path.join("./res", f"random{node_num}nodes")
    output_path = STRUCTURE_CONFIG_PATH
    if args.rank == 0:
        node_count = node_num + 1
        print(f"正在生成 {node_count} 个节点的配置文件...")
        config = generate_layer_config(node_count)
        save_config(config, output_path)
        print(f"配置文件已成功生成: {output_path}")
    
    rounds = 100
    os.makedirs(res_dir, exist_ok=True)
    log_file_path = os.path.join(res_dir, "log.txt")
    sys.stdout = Logger(log_file_path)
    # select configs to be mutated, max = 12
    # configs, selected_files = load_random_model_configs(count=min(node_num, 12))
    # mutate and forward
    successes = 0
    graph = Graph(
        config_path=TEMPLATE_CONFIG_PATH,
        nums=[int(i) for i in range(3)]
    )
    import csv
    csv_path = "execution_msa.csv"
    
    if args.rank == 0:
        with open(csv_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Iteration", "Execution Time (s)", "NPU Memory (MB)","loss"])
            
    for k in range(1,rounds+1):
        loss = torch.tensor(1e9)
        start_time = time.time()
        torch.npu.reset_peak_memory_stats()
        iteration = iteration + 1
        print(f"\n当前第 {iteration}/{rounds} 次迭代")
        file_name = f"mutated_config_iter_{k:03d}.yaml"  # 格式化文件名，例如 "mutated_config_iter_001.yaml"
        file_path = os.path.join(mutating_res_dir, file_name)
        success = 0
        json_file_path = os.path.join(mutating_res_dir, f"mutating-{k}.json")
        
        if args.rank == 0:
            if not os.path.exists(json_file_path):
                with open(csv_path, mode='a+', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow([iteration, '-','-','-']) 
                continue

        try:
            load_success = graph.load(file_path,json_file_path)
            _normalize_graph_node_indices(graph)
            for i in range(len(graph.nodes)):
                print(i,graph.nodes[i].id,graph.nodes[i].str_op)
            print("开始forward")
            final_output = graph.forward(debug=True)
            print(f"\nForward过程完成！最终输出形状: {final_output.shape}")
            mutating_record["success"]=True
            loss = final_output.norm()
            print("loss计算结果：",loss)
            successes+=1
            import mindspore as ms
            from mindspore import grad, ops
            def forward_fn(x):
                return ops.norm(x)
        
            gradient = grad(forward_fn)(loss)
        except Exception as e:
            print(e)
            import traceback
            error_traceback = traceback.format_exc()
            print("错误信息:", error_traceback)
            print("pass")   

        # loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()  

            
        # run forward
        # forward_res = demo_forward_with_random_configs(configs, selected_files, num_decoder_nodes=node_num)
        
        # analyse forward result
        # if forward_res:
        #     selected_files, configs = forward_res
        #     successes += 1
        #     record_file_path = os.path.join(res_dir, f"mutating-{iteration}.json")
        # else:
        #     record_file_path = os.path.join(res_dir, f"mutating-{iteration}-err.json")
        
        # store mutating info into a json file
        # common_utils.save_dict_into_json(mutating_record, record_file_path)
        # print(f"\n变异信息存储已存储到{record_file_path}")
        end_time = time.time()
        end_mem = torch.npu.max_memory_allocated() / 1024 / 1024
        execution_time = end_time - start_time
        mem_usage = end_mem  # 直接记录最终显存占用
        torch.npu.reset_peak_memory_stats()
        if args.rank == 0:
            with open(csv_path, mode='a+', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([iteration,  round(execution_time, 4), round(mem_usage, 4), round(loss.item(), 4)])   
    
    success_rate = (successes / rounds) * 100
    print("变异成功率：",success_rate)

    # print(f"\n{node_num} 个节点配置变异结果: {successes}/{rounds} 次成功 ({success_rate:.2f}%)")


    # file output summary
    # common_utils.print_seperate_line()
    # if os.path.exists(res_dir):
    #     config_files = [f for f in os.listdir(res_dir) if f.endswith('.json')]
    #     print(f"\n保存的变异配置文件总结:")
    #     print(f"   目录: {res_dir}， 总数: {len(config_files)}")
    # common_utils.print_seperate_line()
    
