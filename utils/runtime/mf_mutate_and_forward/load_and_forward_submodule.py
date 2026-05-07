import os
import sys
import json
import time
import numpy as np
from argparse import ArgumentParser
from ruamel.yaml import YAML

import mindspore as ms
import mindspore.ops as ops
import mindspore.runtime as rt

from mindformers.core.context import build_context
from mindformers.tools.register import MindFormerConfig


from utils.runtime import model_helpers
from utils.runtime.logger import Logger
from utils.runtime.step_logger import append_step_metrics, resolve_step_log_csv, resolve_train_iters
from utils.runtime.tensor_manager import TensorManager
from utils.runtime.mf_mutate_and_forward.sub_graph import Graph


def add_extra_args(parser):
    """Add custom arguments for mutation system"""
    parser.add_argument("-c", "--configs", type=str, help="The path to the configs dir")
    parser.add_argument("-n", "--node-num", type=int, default=1, help="nodes num")
    parser.add_argument("-r", "--rounds", type=int, default=10, help="mutating rounds")
    parser.add_argument("--mutnm", type=int, default=2, help="mutating num")
    parser.add_argument("-m", "--module", type=str, help="The targeted single module")
    parser.add_argument("--sub", type=str, help="The list of submodule num")
    parser.add_argument("--load-path", type=str, help="The path of the graph config to load")
    parser.add_argument("--args_path", type=str, help="The path of the mutation arguments yaml")
    parser.add_argument("--train-iters", type=int, default=1, help="Number of verification steps")
    return parser


def seed_all(seed=42):
    model_helpers.seed_all(seed, np_module=np, ms_module=ms)


def parse_numbers_simple(input_str):
    return model_helpers.parse_numbers_simple(input_str)


def extract_number_split(text):
    if "-" in text and "." in text:
        parts = text.split("-", 1)
        if len(parts) > 1:
            subparts = parts[1].split(".", 1)
            if len(subparts) > 0:
                try:
                    return float(subparts[0])
                except ValueError:
                    return None
    return None


def print_graph(graph: Graph):
    print("打印模型结构：")
    for i in range(1, len(graph.nodes)):
        print("节点编号:", i, "子模块类型:", graph.nodes[i].block.__class__.__name__)


def _to_ms_tensor(cpu_tensor, device_target="Ascend"):
    if cpu_tensor is None:
        raise ValueError("输入张量不能为空")
    if isinstance(cpu_tensor, ms.Tensor):
        return cpu_tensor
    if hasattr(cpu_tensor, "detach"):
        np_array = cpu_tensor.detach().cpu().numpy()
    elif hasattr(cpu_tensor, "asnumpy"):
        np_array = cpu_tensor.asnumpy()
    else:
        np_array = np.asarray(cpu_tensor)
    return ms.Tensor(np_array)


def _ensure_min_seq_len(tensor, min_seq_len=4):
    if tensor is None:
        return None
    if len(tensor.shape) == 1:
        tensor = ops.expand_dims(tensor, -1)
    if len(tensor.shape) < 2:
        return tensor
    seq_len = int(tensor.shape[1])
    if seq_len >= min_seq_len:
        return tensor
    repeat_times = (min_seq_len + seq_len - 1) // seq_len
    tensor = ops.tile(tensor, (1, repeat_times))
    return tensor[:, :min_seq_len]


def _get_npu_memory_mb():
    if rt is None:
        return 0.0
    try:
        return float(rt.max_memory_allocated()) / (1024 * 1024)
    except Exception:
        return 0.0


def _reset_npu_memory():
    if rt is None:
        return
    try:
        rt.reset_peak_memory_stats()
    except Exception:
        return


def _sync_pending_config_from_runtime_yaml(base_cfg_dict, runtime_yaml_path):
    if base_cfg_dict is None or not runtime_yaml_path or not os.path.exists(runtime_yaml_path):
        return base_cfg_dict

    yaml = YAML(typ="safe")
    with open(runtime_yaml_path, "r", encoding="utf-8") as f:
        runtime_data = yaml.load(f) or {}

    base_config = runtime_data.get("base_config", {})
    base_inner_cfg = base_config.get("config", {})
    model_cfg = base_cfg_dict.setdefault("model", {}).setdefault("model_config", {})
    parallel_cfg = base_cfg_dict.setdefault("parallel_config", {})

    updated_keys = []

    field_map = {
        "hidden_size": ["hidden_size"],
        "num_attention_heads": ["num_attention_heads"],
        "num_layers": ["num_layers", "num_hidden_layers"],
        "hidden_dropout": ["hidden_dropout"],
        "vocab_size": ["vocab_size"],
        "max_sequence_length": ["max_position_embeddings", "seq_length"],
        "position_embedding_type": ["position_embedding_type"],
        "rotary_base": ["rotary_base"],
    }

    merged_source = dict(base_config)
    merged_source.update(base_inner_cfg)

    for src_key, dst_keys in field_map.items():
        if src_key not in merged_source:
            continue
        src_val = merged_source[src_key]
        for dst_key in dst_keys:
            model_cfg[dst_key] = src_val
            updated_keys.append(f"model.model_config.{dst_key}")

    hidden = model_cfg.get("hidden_size")
    heads = model_cfg.get("num_attention_heads")
    if hidden and heads and int(heads) > 0:
        model_cfg["head_dim"] = int(hidden) // int(heads)
        updated_keys.append("model.model_config.head_dim")

    tp = base_inner_cfg.get("tensor_model_parallel_size")
    pp = base_inner_cfg.get("pipeline_model_parallel_size")
    if tp is not None:
        parallel_cfg["model_parallel"] = int(tp)
        updated_keys.append("parallel_config.model_parallel")
    if pp is not None:
        parallel_cfg["pipeline_stage"] = int(pp)
        updated_keys.append("parallel_config.pipeline_stage")

    if updated_keys:
        print(f"同步 pending_config(dict): {', '.join(sorted(set(updated_keys)))}")

    return base_cfg_dict


def _get_rank_info():
    try:
        rank_id = int(os.getenv("RANK_ID", os.getenv("RANK", "0")) or 0)
    except (TypeError, ValueError):
        rank_id = 0
    try:
        world_size = int(os.getenv("RANK_SIZE", os.getenv("WORLD_SIZE", "1")) or 1)
    except (TypeError, ValueError):
        world_size = 1
    return rank_id, max(1, world_size)




if __name__ == "__main__":
    parser = ArgumentParser()
    add_extra_args(parser)
    args = parser.parse_args()

    seed_all()
    pending_config_dict = None
    if args.args_path:
        yaml = YAML(typ="safe")
        with open(args.args_path, "r", encoding="utf-8") as f:
            pending_config_dict = yaml.load(f) or {}

    module = args.module
    node_num = args.node_num
    rounds = args.rounds
    load_path = args.load_path

    block_num_list = parse_numbers_simple(args.sub)
    block_num_list.insert(0, 0)
    train_iters = resolve_train_iters(args)
    step_csv_path = resolve_step_log_csv()

    csv_path = "res/submodule_execution_mf.csv"

    if module:
        filename = os.path.basename(module).split(".")[0]
        res_dir = os.path.join("./res", f"submodule_{filename}")
    else:
        res_dir = os.path.join("./res", f"submodule_random{node_num}nodes")

    os.makedirs(res_dir, exist_ok=True)

    tensor_dir = os.path.join(res_dir, "tensors")
    tensor_manager = TensorManager(base_dir=tensor_dir, seed=42)
    print("\n=== TensorManager初始化完成 ===")
    print(f"Tensor存储目录: {tensor_dir}")

    rank_id, world_size = _get_rank_info()
    is_main_rank = rank_id == 0

    summary = tensor_manager.get_tensor_summary()
    print(f"分布式信息: rank={rank_id}, world_size={world_size}")
    print(f"已存在的迭代: {summary['available_iterations']}")
    print(f"支持的张量类型: {summary['tensor_types']}")

    log_suffix = "" if is_main_rank else f".rank{rank_id}"
    log_file_path = os.path.join(res_dir, f"verify_submodule_log{log_suffix}.txt")
    sys.stdout = Logger(log_file_path)

    successes = 0

    graph = None
    
    if is_main_rank and not os.path.exists(csv_path):
        import csv
        with open(csv_path, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Iteration", "Execution Time (s)", "NPU Memory (MB)", "loss"])

    forward_res = 1
    start_time = time.time()

    mutate_round = os.getenv("MUTATE_ROUND", "").strip()
    if mutate_round:
        iteration = int(mutate_round)
    else:
        iteration = int(extract_number_split(load_path))

    loss_value = 1e9
    

    yaml_file_name = f"mutated_config_iter_{iteration:03d}.yaml"
    yaml_file_path = os.path.join(res_dir, yaml_file_name)
    json_file_path = os.path.join(res_dir, f"mutating-{iteration}.json")

    succ_path = os.path.join(res_dir, f"mutating-{iteration}.json")
    err_path = os.path.join(res_dir, f"mutating-{iteration}-err.json")
    if os.path.exists(succ_path):
        forward_res = 1
    elif os.path.exists(err_path):
        forward_res = 0

    if not forward_res:
        import csv
        if is_main_rank:
            with open(csv_path, mode="a+", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([iteration, "-", "-", "-"])
        sys.exit(0)

    try:
        if pending_config_dict is not None:
            pending_config_dict = _sync_pending_config_from_runtime_yaml(pending_config_dict, yaml_file_path)
            context_cfg_path = os.path.join(res_dir, f"context_config_iter_{iteration:03d}.yaml")
            dump_yaml = YAML()
            dump_yaml.default_flow_style = False
            with open(context_cfg_path, "w", encoding="utf-8") as f:
                dump_yaml.dump(pending_config_dict, f)
            try:
                build_context(MindFormerConfig(context_cfg_path))
            except TypeError as err:
                print(f"构建上下文失败: {err}")
                # 打印 pending_config_dict 以帮助调试
                print("当前 pending_config_dict 内容:")
                print(json.dumps(pending_config_dict, indent=2))
                raise RuntimeError("构建上下文失败") from err
        graph = Graph(
            config_path="assets/runtime/configs/template_config.yaml",
            nums=[int(i) for i in range(5)]
        )
        load_success = graph.load(yaml_file_path, json_file_path)
        if not load_success:
            print(f"加载变异配置失败: {yaml_file_path} 或 {json_file_path} 不存在或格式错误")
            raise RuntimeError("加载变异配置失败")
        _reset_npu_memory()
        print_graph(graph)
        print("开始forward")
        print(f"使用第 {iteration} 次迭代的输入张量")

        iteration_tensors = tensor_manager.get_iteration_tensors(iteration, device="cpu")
        if iteration_tensors is None:
            raise RuntimeError("Fault generating tensors")

        print(f"使用的张量类型: {list(iteration_tensors.keys())}")
        for tensor_type, tensor in iteration_tensors.items():
            print(f"  {tensor_type}: shape={tensor.shape}, dtype={tensor.dtype}")

        input_ids = _to_ms_tensor(iteration_tensors.get("input_ids"))
        raw_input_data = iteration_tensors.get("input_data")
        input_data = _to_ms_tensor(raw_input_data) if raw_input_data is not None else None
        
        if input_data is None and input_ids is not None:
            input_data = input_ids.copy()

        graph.set_train(True)

        def forward_loss(ids, data):
            output = graph.forward(
                input_ids=ids,
                input_data=data,
                debug=False
            )
            if output.dtype != ms.float32:
                output = ops.cast(output, ms.float32)
            return ops.norm(output)
        grad_fn = ms.grad(forward_loss, weights=graph.trainable_params())

        for step_idx in range(train_iters):
            _reset_npu_memory()
            step_start_time = time.time()
            loss = forward_loss(input_ids, input_data)
            loss_value = float(loss.asnumpy())
            print("norm计算结果", loss)
            print(f"step {step_idx + 1} loss计算结果：", loss_value)

            grads = grad_fn(input_ids, input_data)
            if grads:
                def _iter_tensors(value):
                    if value is None:
                        return
                    if isinstance(value, (tuple, list)):
                        for item in value:
                            yield from _iter_tensors(item)
                        return
                    yield value

                grad_norms = [ops.norm(g).asnumpy() for g in _iter_tensors(grads)]
                total_grad_norm = float(np.sum(grad_norms)) if grad_norms else 0.0
                print(f"参数梯度数量: {len(grad_norms)}, 总梯度范数: {total_grad_norm}")

            step_mem_usage = _get_npu_memory_mb()
            step_elapsed = time.time() - step_start_time
            if step_csv_path and is_main_rank:
                append_step_metrics(step_csv_path, step_idx + 1, step_elapsed, step_mem_usage, loss_value)

        successes += 1

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print("错误信息:", error_traceback)
        print("pass")

    end_time = time.time()
    end_mem = _get_npu_memory_mb()
    execution_time = end_time - start_time
    mem_usage = end_mem
    _reset_npu_memory()

    if is_main_rank:
        import csv
        with open(csv_path, mode="a+", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([iteration, round(execution_time, 4), round(mem_usage, 4), round(loss_value, 4)])

    final_summary = tensor_manager.get_tensor_summary()
    print(f"\n=== Tensor管理器最终状态 ===")
    print(f"存储目录: {final_summary['base_dir']}")
    print(f"总迭代数: {final_summary['total_iterations']}")
    print(f"可用迭代: {final_summary['available_iterations']}")

    if successes:
        print("变异结果验证成功")
    else:
        print("变异结果验证失败")
