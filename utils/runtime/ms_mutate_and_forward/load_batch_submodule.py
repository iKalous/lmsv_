import mindspeed.megatron_adaptor as ma
import sys
import numpy as np
from argparse import ArgumentParser
from mindspeed.arguments import process_args
from megatron.training.arguments import parse_args, validate_args
from megatron.training.global_vars import set_global_variables
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training import get_args
import torch_npu
from megatron.training.initialize import _initialize_distributed,_init_autoresume,_set_random_seed,_compile_dependencies,_initialize_tp_communicators
from megatron.training import get_args

import torch
import json
import random
import os
from ruamel.yaml import YAML
import copy
import time
import sys


from utils.runtime.core.submutation import ConfigMutator
from utils.runtime.core.subgraph import Graph, Node
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed import DistributedDataParallel as DDP
import torch.distributed as dist
from utils.runtime import common_utils, model_helpers
from utils.runtime.logger import Logger
from utils.runtime.tensor_manager import TensorManager




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

def seed_all(seed=42):
    model_helpers.seed_all(seed, np_module=np, torch_module=torch, torch_npu_module=torch_npu)
    
    


yaml = YAML()
# # choice_list = [0,1,7,8]  # 二级模块
# # choice_list = [2,5,6,9,10] # 三级模块
# choice_list = [3,4] # 四级模块
sub_list_set = [[0,1,7,8],[2,5,6,9,10],[3,4]]
# # block_num_list = random.choices(choice_list, k=7)

submodule_list = []
config_dir = ""
res_dir = ""
mutating_rate = 0.3
iteration = 0
mutating_record = {}
record_file_name = ""
no_mutating = False
module = args.module
# sub_list = sub_list_set[int(args.sub)-2]
# block_num_list = random.choices(sub_list, k=7)

node_num = args.node_num

def parse_numbers_simple(input_str):
    return model_helpers.parse_numbers_simple(input_str)
    
block_num_list = parse_numbers_simple(args.sub)
block_num_list.insert(0,0)


config_dir = args.configs
rounds = args.rounds
mutating_num = args.mutnm
mutating_res_dir = args.path

# import sys
# import torch
# import json
# import random
# import os
# import yaml

# os.environ["PYTORCH_NPU_FORCE_FALLBACK"] = "1"
# sys.path.append(".")

# import mindspeed.megatron_adaptor as ma
# import common_utils
# from subgraph import Graph, Node
# from submutation import ConfigMutator
# from megatron.core.transformer.transformer_config import TransformerConfig
# from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
# from megatron.core.transformer.transformer_block import TransformerBlock
# from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
# import copy
# import torch
# import torch_npu
# import torch.nn.functional as F
# from megatron.core import parallel_state
# from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
# from megatron.core.transformer.transformer_config import TransformerConfig
# from megatron.core.transformer.transformer_block import TransformerBlock
# from ruamel.yaml import YAML
# import sys
# from argparse import ArgumentParser
# from mindspeed.arguments import process_args
# import megatron.training.global_vars as global_vars
# from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
# from megatron.core.distributed import DistributedDataParallelConfig
# from megatron.core.distributed import DistributedDataParallel as DDP

# if not hasattr(sys, "argv"):
#     sys.argv = ["submodule_demo.py"]
# parser = ArgumentParser()
# parser.add_argument('--overlap-param-gather', action='store_true', default=False, help='Enable overlap param gather')
# parser.add_argument('--num-experts', type=int, default=0, help='Number of experts for MoE/MLP')
# parser.add_argument('--use-flash-attn', action='store_true', default=False, help='Use flash attn')
# parser.add_argument('--recompute-num-layers', type=int, default=0, help='Number of recompute layers')
# process_args(parser)
# args = parser.parse_args()
# global_vars.set_args(args)

# torch.npu.set_device(0)
# yaml = YAML()
# mutating_record = {}
# record_file_name = ""
# no_mutating = False
# # module = "assets/runtime/model_config/chatglm_decoder.yaml"
# module = None
# config_dir = "model_config"
# rounds = 100
# mutating_num = 2
# node_num = 2

# # choice_list = [0,1,7,8]  # 二级模块
# # choice_list = [2,5,6,9,10] # 三级模块
# choice_list = [3,4] # 四级模块
# # block_num_list = random.choices(choice_list, k=7)
# block_num_list = [0,0,0,1,1]
# try:
#     torch.distributed.init_process_group(
#         backend="hccl",
#         init_method="tcp://127.0.0.1:12349",
#         rank=0,
#         world_size=1,
#     )
# except:
#     print("Distributed already initialized or failed")

# # 3. 初始化模型并行（必须在分布式环境和全局args之后）
# parallel_state.initialize_model_parallel(
#     tensor_model_parallel_size=1,
#     pipeline_model_parallel_size=1,
#     expert_model_parallel_size=1
# )

# random_seed = 42
# model_parallel_cuda_manual_seed(random_seed)

# submodule_list = []


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
    return model_helpers.extract_transformer_config_from_yaml(yaml_config)


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

    print(f"\n创建包含 {num_decoder_nodes} 个节点的图")
    print(f"节点连接:".join([f"模块{i}({i + 1})" for i in range(num_decoder_nodes)]))

    # 保存临时JSON文件
    with open("temp_graph.json", "w") as f:
        json.dump(graph_config, f)

    # 读取JSON配置
    with open("temp_graph.json", "r") as f:
        output_data = json.load(f)
        g = output_data['network']

    # 创建基础图
    graph = Graph(
        config_path="assets/runtime/configs/template_config.yaml",
        nums=[i for i in range(len(g))]
    )
    # print(graph.total_config)

    # 创建变异器 - 使用新的合并配置系统
    mutator = ConfigMutator(
        structure_config_path="assets/runtime/configs/structure_config.yaml",
        template_config_path="assets/runtime/configs/template_config.yaml",
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
    global block_num_list
    mutating_record["block_num_list"] = block_num_list
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

        node.block_num = block_num_list[node_id]

        # 处理layer_limits
        if 'layer_limits' in node_:
            node.layer_limits = node_['layer_limits']
            if node.layer_limits[0] != "none":
                import random
                random_num = random.randrange(len(node.layer_limits))
                node.str_op = node.layer_limits[random_num]

                # 检查是否需要变异此节点
                should_mutate = True
                is_last_decoder = False  # 最后一个decoder节点

                if node.str_op == "Qwen2DecoderLayer" and should_mutate:
                    mutating_record[node_id]["mutated"] = True
                    print(f"\n对节点 {node_id} 应用变异")

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

                # 创建TransformerBlock（参考generate_graph.py）
                if ('decoderlayer' in node.str_op.lower() or 'mutated_decoder' in node.str_op.lower()):
                    transformer_block = TransformerBlock(
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
                    else:
                        node.block = transformer_block.layers[0].mlp.linear_fc2
                        # input: (seq_length, batch_size,ffn_hidden_size)
                        # output[0],[seq_length, batch_size, hidden_size]
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

        for i in range(1, len(graph.nodes)):
            graph.nodes[i].block_num = block_num_list[graph.nodes[i].id]

        print(f"\n图创建完成，节点数: {len(graph.nodes) - 1}")
        print(f"变异节点: {list(mutated_nodes.keys())}")
        # print(f"使用的配置文件: {selected_files}")

        # 执行forward过程 - 修改：直接调用graph.forward()方法
        print("\n--- 执行Forward过程 ---")

        # 直接调用graph的forward方法（mutated_nodes已在初始化时设置）
        # final_output = graph.forward(debug=True)
        # print(f"\nForward过程完成！最终输出形状: {final_output.shape}")
        # mutating_record["success"] = True

#         ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
#         graph_model = [graph]
#         model = [DDP(graph.total_config["config"], ddp_config, model_chunk, disable_bucketing=(model_chunk_idx > 0))
#                  for (model_chunk_idx, model_chunk) in enumerate(graph_model)]

#         optimizer_config = OptimizerConfig(
#             optimizer='adam',
#             lr=1e-4,
#             weight_decay=0.01,
#         )
#         optimizer = get_megatron_optimizer(
#             config=optimizer_config,
#             model_chunks=model
#         )

        output = graph.forward(debug=True)
        # print(f"\nForward过程完成！最终输出形状: {output.shape}")
        loss = output.norm()
        
        import mindspore as ms
        from mindspore import grad, ops
        def forward_fn(x):
            return ops.norm(x)
        
        gradient = grad(forward_fn)(loss)
        # optimizer.step()
        # optimizer.zero_grad()
        # loss = final_output.sum()
        # loss.backward()
        print("Backward过程完成，loss计算结果:", loss)
        # output_file = "output1.pt"
        # torch.save(final_output, output_file)
        # print(f"最终输出已保存到文件: {output_file}")

        # clear temp files
        if os.path.exists("temp_graph.json"):
            os.remove("temp_graph.json")

        return selected_files, new_configs,loss

    except Exception as e:
        print(f"Forward演示失败: {e}")

        # record err info to file and console
        import traceback
        traceback.print_exc()
        error_stack = traceback.format_exc()
        mutating_record["success"] = False
        mutating_record["err_stack"] = error_stack

        return None


def print_npu_memory(tag=""):
    try:
        # 获取NPU当前分配的显存
        allocated = torch.npu.memory_allocated() / 1024 / 1024  # 单位 MB

        print(f"[NPU显存][{tag}] 当前分配: {allocated:.2f} MB")
    except Exception as e:
        print(f"[NPU显存][{tag}] 记录失败: {e}")


if __name__ == "__main__":
    
    seed_all()
    if module:
        filename = os.path.basename(module).split('.')[0]
        res_dir = os.path.join("./res", f"submodule_{filename}")
    else:
        res_dir = os.path.join("./res", f"submodule_random{node_num}nodes")

    print(res_dir)
    os.makedirs(res_dir, exist_ok=True)
    log_file_path = os.path.join(res_dir, "msa_verify_batch_submodule_log.txt")
    sys.stdout = Logger(log_file_path)
    # select configs to be mutated, max = 12
    # configs, selected_files = load_random_model_configs(count=min(node_num, 12))  # 随机选择node_num个模型配置文件
    # mutate and forward

    # 初始化TensorManager
    tensor_dir = os.path.join(mutating_res_dir, "tensors")
    tensor_manager = TensorManager(base_dir=tensor_dir, seed=42)
    print(f"\n=== TensorManager初始化完成 ===")
    print(f"Tensor存储目录: {tensor_dir}")
    
    # 打印现有张量情况
    summary = tensor_manager.get_tensor_summary()
    print(f"已存在的迭代: {summary['available_iterations']}")
    print(f"支持的张量类型: {summary['tensor_types']}")
    
    
    import csv
    import time
    submodule_list = [(i, 0) for i in range(node_num + 1)]
    csv_path = "res/submodule_execution_msa.csv"
    
    if args.rank == 0:
        with open(csv_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Iteration", "Execution Time (s)", "NPU Memory (MB)","loss"])
    successes = 0
    iteration = 0
    rounds = 100
    
    graph = Graph(
        config_path="assets/runtime/configs/template_config.yaml",
        nums=[int(i) for i in range(5)]
    )
    for k in range(1,rounds+1):
        output = torch.tensor(1e9)
        start_time = time.time()
        torch.npu.reset_peak_memory_stats()
        iteration = iteration + 1
        print(f"\n当前第 {iteration}/{rounds} 次迭代")
        # forward_res = demo_forward_with_random_configs(configs, selected_files, num_decoder_nodes=node_num)
        yaml_file_name = f"mutated_config_iter_{k:03d}.yaml"  # 格式化文件名，例如 "mutated_config_iter_001.yaml"
        yaml_file_path = os.path.join(mutating_res_dir, yaml_file_name)
        json_file_name = f"mutating-{k}.json"  
        json_file_path = os.path.join(mutating_res_dir, json_file_name)
        
        if args.rank == 0:
            if not os.path.exists(json_file_path):
                with open(csv_path, mode='a+', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow([iteration, '-','-','-']) 
                continue
                
        # dist.barrier()
        try:
            graph.load(yaml_file_path,json_file_path)

            if tensor_manager is None:
                raise ValueError("TensorManager未初始化，请先调用初始化函数")
            print(f"使用第 {iteration} 次迭代的输入张量")
            device = torch.device("npu" if torch.npu.is_available() else "cpu")
            iteration_tensors = tensor_manager.get_iteration_tensors(iteration, device)
            if iteration_tensors is None:
                panic("Fault generating tensors")
            print(f"使用的张量类型: {list(iteration_tensors.keys())}")
            for tensor_type, tensor in iteration_tensors.items():
                print(f"  {tensor_type}: shape={tensor.shape}, dtype={tensor.dtype}")  
            
            output = graph.forward(
                input_ids=iteration_tensors.get('input_ids'),
                input_data=iteration_tensors.get('input_data'), 
                debug=True
            )
            output = output.norm()
            successes += 1
        
        except Exception as e:
            print("pass")
            print(e)
        print(output)
        
        end_time = time.time()
        end_mem = torch.npu.max_memory_allocated() / 1024 / 1024
        execution_time = end_time - start_time
        mem_usage = end_mem  # 直接记录最终显存占用
        torch.npu.reset_peak_memory_stats()
        if args.rank == 0:
            with open(csv_path, mode='a+', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([iteration,  round(execution_time, 4), round(mem_usage, 4), round(output.item(), 4)])
        # store mutating info into a json file
        # common_utils.save_dict_into_json(mutating_record, record_file_path)
        # print(f"\n变异信息存储已存储到{record_file_path}")

    success_rate = (successes / rounds) * 100
    print("变异成功率：",success_rate)
    # config = TransformerConfig(
    #     tensor_model_parallel_size=1,
    #     pipeline_model_parallel_size=1,
    #     num_layers=32,
    #     hidden_size=896,
    #     ffn_hidden_size=4864,
    #     num_attention_heads=14,
    #     num_query_groups=2,
    #     attention_dropout=0.0,
    #     init_method_std=0.01,
    #     hidden_dropout=0.0,
    #     normalization="RMSNorm",
    #     layernorm_epsilon=1e-6
    # )
    # # 实例化 attention_block
    # qwen_decoder_model = TransformerBlock(
    #     config=config,
    #     spec=get_gpt_layer_local_spec(
    #         None,
    #         False,
    #         False,
    #     ),
    #     pre_process=True,
    #     post_process=True,
    # )

    # print(vars(qwen_decoder_model.layers[2].self_attention))
    # for i in range(len(qwen_decoder_model.layers)):
    #     print(qwen_decoder_model.layers[i])

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     qwen_decoder_model = qwen_decoder_model.to(device)
#     batch_size = 2
#     seq_length = 32
#     hidden_size = config.hidden_size

#     # 生成输入张量 [seq_length, batch_size, hidden_size]
#     input_tensor = torch.randn(seq_length, batch_size, hidden_size, device=device)

#     # 生成attention mask [1, 1, seq_length, seq_length]
#     attention_mask = torch.ones(1, 1, seq_length, seq_length, dtype=torch.bool, device=device)

#     # 前向计算
#     with torch.no_grad():
#         output = qwen_decoder_model(
#             hidden_states=input_tensor,
#             attention_mask=attention_mask
#         )
#     print("Input shape:", input_tensor.shape)
#     print("Output shape:", output.shape)
