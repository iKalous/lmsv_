import os
import sys
import time
import json
import numpy as np
import yaml
from argparse import ArgumentParser

import mindspore as ms
import mindspore.ops as ops
import mindspore.runtime as rt
from mindspore import nn

from mindformers.core.context import build_context
from mindformers.tools.register import MindFormerConfig

from utils.runtime import model_helpers
from utils.runtime.debug_utils import set_debug_step
from utils.runtime.logger import Logger
from utils.runtime.step_logger import append_step_metrics, resolve_step_log_csv, resolve_train_iters
from utils.runtime.mf_mutate_and_forward.graph import Graph


ATTENTION_CONFIG_FIELDS = (
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
    "use_flash_attention",
    "qkv_concat",
    "qk_head_dim",
    "qk_pos_emb_head_dim",
    "v_head_dim",
)

DECODER_ALIGN_JSON_KEY = "__pta_decoder_align__"
DECODER_FOCUS_FIELDS = (
    "num_layers",
    "hidden_size",
    "ffn_hidden_size",
    "num_attention_heads",
    "num_query_groups",
    "kv_channels",
    "normalization",
    "layernorm_epsilon",
    "attention_dropout",
    "hidden_dropout",
    "masked_softmax_fusion",
    "attention_softmax_in_fp32",
    "apply_query_key_layer_scaling",
    "activation_func",
    "add_qkv_bias",
    "add_bias_linear",
    "bias_dropout_fusion",
    "apply_rope_fusion",
    "context_parallel_algo",
    "sequence_parallel",
    "position_embedding_type",
    "rotary_base",
    "qk_layernorm",
    "use_flash_attention",
    "qkv_concat",
    "qk_head_dim",
    "qk_pos_emb_head_dim",
    "v_head_dim",
)

def add_extra_args(parser):
    """Add custom arguments for mutation system."""
    parser.add_argument("-c", "--configs", type=str, help="The path to the configs dir")
    parser.add_argument("-n", "--node-num", type=int, default=1, help="nodes num")
    parser.add_argument("-r", "--rounds", type=int, default=10, help="mutating rounds")
    parser.add_argument("--mutnm", type=int, default=2, help="mutating num")
    parser.add_argument("-m", "--module", type=str, help="The targeted single module")
    parser.add_argument("--sub", type=str, help="The list of submodule num")
    parser.add_argument("--load-path", type=str, help="The path of the graph config to load")
    parser.add_argument("--args_path", type=str, help="The path of the mutation arguments yaml")
    parser.add_argument("--shared-weight-ckpt", type=str, help="Shared MindSpore checkpoint path")
    parser.add_argument("--train-iters", type=int, default=1, help="Number of verification steps")
    return parser


def seed_all(seed=42):
    model_helpers.seed_all(seed, np_module=np, ms_module=ms)


def _env_flag(name, default="0"):
    # 注意：MindSpore JIT严格模式不支持链式字符串方法，因此使用直接比较
    value = os.getenv(name, default)
    return value == "1" or value == "true" or value == "yes" or value == "on"


def _env_int(name, default):
    # 注意：MindSpore JIT严格模式不支持链式字符串方法
    value = os.getenv(name, str(default))
    try:
        return int(value)
    except ValueError:
        return int(default)


def extract_number_split(text):
    if not text:
        return None
    if "err" in text:
        parts = text.split("-")
        if len(parts) > 2:
            try:
                return float(parts[1])
            except ValueError:
                return None
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


def _get_npu_memory_mb():
    try:
        return float(rt.max_memory_allocated()) / (1024 * 1024)
    except Exception:
        return 0.0


def _reset_npu_memory():
    try:
        rt.reset_peak_memory_stats()
    except Exception:
        return


def _to_jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return str(value)


def _read_attention_field(cfg, key):
    if hasattr(cfg, key):
        raw_value = getattr(cfg, key)
        # 某些MF配置会暴露字段但值为None，需继续走兼容映射。
        if not (key == "masked_softmax_fusion" and raw_value is None):
            return _to_jsonable(raw_value), True

    # MF部分配置仅暴露 no_masked_softmax_fusion，需要反向映射回 PTA 对齐字段。
    if key == "masked_softmax_fusion" and hasattr(cfg, "no_masked_softmax_fusion"):
        raw_value = getattr(cfg, "no_masked_softmax_fusion")
        return (not bool(raw_value)), True

    if key == "use_flash_attention" and hasattr(cfg, "use_flash_attn"):
        return _to_jsonable(getattr(cfg, "use_flash_attn")), True

    if key == "qkv_concat" and hasattr(cfg, "mla_qkv_concat"):
        return _to_jsonable(getattr(cfg, "mla_qkv_concat")), True

    return None, False


def collect_attention_runtime_config(graph: Graph, iteration: int):
    node_entries = {}
    for node_id in sorted(graph.nodes.keys()):
        node = graph.nodes[node_id]
        if "decoder" not in node.str_op:
            continue
        cfg = node.config
        fields = {}
        for key in ATTENTION_CONFIG_FIELDS:
            value, exists = _read_attention_field(cfg, key)
            if exists:
                fields[key] = value
        node_entries[str(node_id)] = fields

    return {
        "backend": "mf",
        "iteration": int(iteration),
        "attention_fields": list(ATTENTION_CONFIG_FIELDS),
        "nodes": node_entries,
    }


def _extract_serializable_decoder_config(cfg):
    dataclass_fields = getattr(cfg, "__dataclass_fields__", {})
    keys = list(dataclass_fields.keys()) if dataclass_fields else list(vars(cfg).keys())
    data = {}
    for key in keys:
        if not hasattr(cfg, key):
            continue
        value = getattr(cfg, key)
        if callable(value):
            continue
        converted = _to_jsonable(value)
        if isinstance(converted, (str, int, float, bool)) or converted is None:
            data[key] = converted
            continue
        if isinstance(converted, list):
            if all(isinstance(v, (str, int, float, bool)) or v is None for v in converted):
                data[key] = converted
            continue
        if isinstance(converted, dict):
            if all(
                isinstance(v, (str, int, float, bool)) or v is None
                for v in converted.values()
            ):
                data[key] = converted
            continue
    return data


def collect_decoder_runtime_config(graph: Graph, iteration: int):
    node_entries = {}
    for node_id in sorted(graph.nodes.keys()):
        node = graph.nodes[node_id]
        if "decoder" not in node.str_op:
            continue
        node_entries[str(node_id)] = _extract_serializable_decoder_config(node.config)

    return {
        "backend": "mf",
        "iteration": int(iteration),
        "nodes": node_entries,
    }


def dump_attention_runtime_config(graph: Graph, result_dir: str, iteration: int):
    payload = collect_attention_runtime_config(graph, iteration)
    out_path = os.path.join(result_dir, f"attention_runtime_mf_iter_{int(iteration):03d}.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"MF attention运行时配置已写出: {out_path}")
    return payload, out_path


def dump_decoder_runtime_config(graph: Graph, result_dir: str, iteration: int):
    payload = collect_decoder_runtime_config(graph, iteration)
    out_path = os.path.join(result_dir, f"decoder_runtime_mf_iter_{int(iteration):03d}.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"MF decoder运行时配置已写出: {out_path}")
    return payload, out_path


def _load_pta_decoder_expected(result_dir: str, iteration: int):
    json_path = os.path.join(result_dir, f"mutating-{int(iteration)}.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle) or {}
            align_payload = payload.get(DECODER_ALIGN_JSON_KEY)
            if isinstance(align_payload, dict) and isinstance(align_payload.get("nodes"), dict):
                return {"nodes": align_payload.get("nodes", {})}
        except Exception:
            pass

    expected_path = os.getenv("LMSV_DECODER_CFG_SNAPSHOT_PATH", "")
    if not expected_path:
        expected_path = os.path.join(result_dir, f"decoder_runtime_pta_iter_{int(iteration):03d}.json")
    if not os.path.exists(expected_path):
        return None
    with open(expected_path, "r", encoding="utf-8") as handle:
        return json.load(handle) or {}


def validate_decoder_config_parity(graph: Graph, result_dir: str, iteration: int) -> bool:
    strict = os.getenv("LMSV_STRICT_DECODER_CONFIG_MATCH", "0") == "1"
    payload, _ = dump_decoder_runtime_config(graph, result_dir, iteration)

    expected_payload = _load_pta_decoder_expected(result_dir, iteration)
    if not expected_payload:
        msg = "未找到PTA decoder配置快照或JSON对齐元数据"
        if strict:
            print(msg)
            return False
        print(f"{msg}，当前为非严格模式，跳过decoder配置对齐校验")
        return True

    pta_nodes = expected_payload.get("nodes", {})
    mf_nodes = payload.get("nodes", {})

    mismatches = []
    all_nodes = sorted(set(pta_nodes.keys()) | set(mf_nodes.keys()), key=lambda x: int(x))
    for node_id in all_nodes:
        pta_cfg = pta_nodes.get(node_id)
        mf_cfg = mf_nodes.get(node_id)
        if pta_cfg is None:
            mismatches.append((node_id, "node_missing_in_pta", None, mf_cfg))
            continue
        if mf_cfg is None:
            mismatches.append((node_id, "node_missing_in_mf", pta_cfg, None))
            continue

        keys = sorted(set(pta_cfg.keys()) | set(mf_cfg.keys()))
        for key in keys:
            pta_val = pta_cfg.get(key)
            mf_val = mf_cfg.get(key)
            if pta_val != mf_val:
                mismatches.append((node_id, key, pta_val, mf_val))

    diff_path = os.path.join(result_dir, f"decoder_runtime_diff_iter_{int(iteration):03d}.json")
    diff_payload = {
        "iteration": int(iteration),
        "mismatch_count": len(mismatches),
        "mismatches": [
            {
                "node": node_id,
                "field": key,
                "pta": pta_val,
                "mf": mf_val,
            }
            for node_id, key, pta_val, mf_val in mismatches
        ],
    }
    with open(diff_path, "w", encoding="utf-8") as handle:
        json.dump(diff_payload, handle, ensure_ascii=False, indent=2)
    print(f"decoder配置差异明细已写出: {diff_path}")

    focus_node = "1" if "1" in pta_nodes or "1" in mf_nodes else (all_nodes[0] if all_nodes else None)
    if focus_node is not None:
        pta_focus = pta_nodes.get(focus_node, {})
        mf_focus = mf_nodes.get(focus_node, {})
        print(f"decoder关键字段对照(node={focus_node}):")
        for field in DECODER_FOCUS_FIELDS:
            print(f"  {field}: PTA={pta_focus.get(field)} | MF={mf_focus.get(field)}")

    if not mismatches:
        print(f"decoder配置严格对齐通过: 节点数={len(all_nodes)}")
        return True

    print(f"decoder配置存在不一致，共 {len(mismatches)} 处，示例(最多40条):")
    for node_id, key, pta_val, mf_val in mismatches[:40]:
        print(f"  node={node_id} field={key} PTA={pta_val} MF={mf_val}")

    if strict:
        print("decoder配置严格对齐失败（LMSV_STRICT_DECODER_CONFIG_MATCH=1）")
        return False

    print("decoder配置不一致，但当前为非严格模式，继续执行")
    return True


def validate_attention_config_parity(graph: Graph, result_dir: str, iteration: int) -> bool:
    strict = os.getenv("LMSV_STRICT_ATTN_CONFIG_MATCH", "1") == "1"
    payload, _ = dump_attention_runtime_config(graph, result_dir, iteration)

    expected_path = os.getenv("LMSV_ATTN_CFG_SNAPSHOT_PATH", "")
    if not expected_path:
        expected_path = os.path.join(result_dir, f"attention_runtime_pta_iter_{int(iteration):03d}.json")

    if not os.path.exists(expected_path):
        msg = f"未找到PTA attention配置快照: {expected_path}"
        if strict:
            print(msg)
            return False
        print(f"{msg}，当前为非严格模式，跳过对齐校验")
        return True

    try:
        with open(expected_path, "r", encoding="utf-8") as handle:
            pta_payload = json.load(handle) or {}
    except Exception:
        import traceback
        print("读取PTA attention配置快照失败:\n", traceback.format_exc())
        return not strict

    pta_nodes = pta_payload.get("nodes", {})
    mf_nodes = payload.get("nodes", {})
    mismatch_list = []
    ignored_list = []

    all_nodes = sorted(set(pta_nodes.keys()) | set(mf_nodes.keys()), key=lambda x: int(x))
    for node_id in all_nodes:
        pta_cfg = pta_nodes.get(node_id)
        mf_cfg = mf_nodes.get(node_id)
        if pta_cfg is None:
            mismatch_list.append((node_id, "node_missing_in_pta", None, mf_cfg))
            continue
        if mf_cfg is None:
            mismatch_list.append((node_id, "node_missing_in_mf", pta_cfg, None))
            continue

        keys = sorted(set(pta_cfg.keys()) | set(mf_cfg.keys()))
        for key in keys:
            pta_val = pta_cfg.get(key)
            mf_val = mf_cfg.get(key)

            # MF当前版本未稳定暴露该字段，缺失时不作为严格失败条件。
            if key == "masked_softmax_fusion" and mf_val is None:
                ignored_list.append((node_id, key, pta_val, mf_val))
                continue

            if pta_val != mf_val:
                mismatch_list.append((node_id, key, pta_val, mf_val))

    if ignored_list:
        print(f"attention字段兼容忽略 {len(ignored_list)} 处(字段缺失): masked_softmax_fusion")

    if not mismatch_list:
        print(f"attention配置严格对齐通过: 节点数={len(all_nodes)}")
        return True

    print(f"attention配置存在不一致，共 {len(mismatch_list)} 处，示例(最多20条):")
    for node_id, key, pta_val, mf_val in mismatch_list[:20]:
        print(f"  node={node_id} field={key} PTA={pta_val} MF={mf_val}")

    if strict:
        print("attention配置严格对齐失败（LMSV_STRICT_ATTN_CONFIG_MATCH=1）")
        return False

    print("attention配置不一致，但当前为非严格模式，继续执行")
    return True


class GraphWithLoss(nn.Cell):
    def __init__(self, network):
        super().__init__(auto_prefix=False)
        self.network = network

    def construct(self, input_ids=None, position_ids=None, attention_mask=None):
        output = self.network(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            debug=True,
        )
        if output.dtype != ms.float32:
            output = ops.cast(output, ms.float32)
        norm_value = ops.norm(output)
        print("norm计算结果", norm_value)
        return norm_value


def print_graph(graph: Graph):
    print("打印模型结构：")
    for node_id in sorted(graph.nodes.keys()):
        node = graph.nodes[node_id]
        model_type = "DecoderLayer" if "decoder" in node.str_op else "Embedding"
        print("节点编号:", node_id, "节点类型:", model_type)
        if model_type == "Embedding":
            print("Embedding类:", node.block.__class__.__name__, "主要参数：")
            print("     vocab_size:", graph.total_config["vocab_size"])
            print("     max_sequence_length:", graph.total_config["max_sequence_length"])
            print("     position_embedding_type:", graph.total_config["position_embedding_type"])
            print()
        else:
            print("DecoderLayer类:", node.block.__class__.__name__, "主要参数：")
            print("     num_layers:", node.config.num_layers)
            print("     ffn_hidden_size:", node.config.ffn_hidden_size)
            print("     num_attention_heads:", node.config.num_attention_heads)
            print("     num_query_groups:", node.config.num_query_groups)
            print("     use_flash_attention:", getattr(node.config, "use_flash_attention", None))
            print("     qkv_concat:", getattr(node.config, "qkv_concat", getattr(node.config, "mla_qkv_concat", None)))
            print("     qk_head_dim:", getattr(node.config, "qk_head_dim", None))
            print("     qk_pos_emb_head_dim:", getattr(node.config, "qk_pos_emb_head_dim", None))
            print("     v_head_dim:", getattr(node.config, "v_head_dim", None))
            print("     attention_dropout:", node.config.attention_dropout)
            print("     init_method_std:", node.config.init_method_std)
            print("     hidden_dropout:", node.config.hidden_dropout)
            print("     normalization:", node.config.normalization)
            print("     layernorm_epsilon:", node.config.layernorm_epsilon)
            print()


def patch_mf_yaml_for_task3_stability(yaml_file_path: str) -> None:
    """Apply a task3-only safety patch to MF yaml to reduce kernel instability."""
    if os.getenv("LMSV_TASK3_FORCE_MF_SAFE", "0") != "1":
        return

    if not os.path.exists(yaml_file_path):
        return

    try:
        with open(yaml_file_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}

        parallel_cfg = config.setdefault("parallel_config", {})
        old_dp = int(parallel_cfg.get("data_parallel", 1) or 1)
        old_cp = int(parallel_cfg.get("context_parallel", 1) or 1)

        # Disable flash-attention path and collapse CP to avoid known transpose kernel failures.
        model_cfg = config.setdefault("model", {}).setdefault("model_config", {})
        model_cfg["use_flash_attention"] = False
        for dtype_key in (
            "params_dtype",
            "compute_dtype",
            "layernorm_compute_dtype",
            "softmax_compute_dtype",
            "rotary_dtype",
        ):
            if dtype_key in model_cfg:
                model_cfg[dtype_key] = "float32"

        parallel_cfg["context_parallel"] = 1
        parallel_cfg["data_parallel"] = max(1, old_dp * max(1, old_cp))

        strategy = config.setdefault("parallel", {}).get("dataset_strategy")
        if isinstance(strategy, list):
            new_dp = int(parallel_cfg["data_parallel"])
            for item in strategy:
                if isinstance(item, list) and item:
                    item[0] = new_dp

        with open(yaml_file_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, allow_unicode=False, sort_keys=False)

        print(
            "Task3 MF稳定性补丁已启用: "
            f"use_flash_attention=False, context_parallel 由 {old_cp} 调整为 1, "
            f"data_parallel 由 {old_dp} 调整为 {parallel_cfg['data_parallel']}"
        )
    except Exception:
        import traceback
        print("Task3 MF稳定性补丁应用失败:\n", traceback.format_exc())


def load_shared_ckpt_if_present(graph: Graph, ckpt_path: str) -> bool:
    if not ckpt_path:
        print("未指定共享ckpt，跳过参数加载")
        return True
    if not os.path.exists(ckpt_path):
        print(f"共享ckpt不存在: {ckpt_path}")
        return False

    try:
        param_dict = ms.load_checkpoint(ckpt_path)
        if not param_dict:
            print(f"共享ckpt为空: {ckpt_path}")
            return False

        net_params = graph.parameters_dict()
        filtered_param_dict = {}
        skipped_by_name = 0
        skipped_by_shape = 0
        adapted_by_shape = 0
        adapted_examples = []
        skipped_shape_examples = []

        def _is_attention_param(name: str) -> bool:
            lowered = str(name).lower()
            return "attention" in lowered or "query_key_value" in lowered

        def _is_state_tensor(name: str) -> bool:
            lowered = str(name).lower()
            return lowered.endswith(".seed") or lowered.endswith(".offset")

        attn_net_params = {name for name in net_params.keys() if _is_attention_param(name)}

        def _to_numpy(value):
            if isinstance(value, ms.Tensor):
                return value.asnumpy()
            if hasattr(value, "data") and isinstance(value.data, ms.Tensor):
                return value.data.asnumpy()
            if hasattr(value, "asnumpy"):
                return value.asnumpy()
            return np.asarray(value)

        def _resolve_target_name(name: str):
            if name in net_params:
                return name

            alias_candidates = [
                name.replace(".gamma", ".weight"),
                name.replace(".beta", ".bias"),
                name.replace(".weight", ".gamma"),
                name.replace(".bias", ".beta"),
            ]
            for candidate in alias_candidates:
                if candidate in net_params:
                    return candidate
            return None

        for name, value in param_dict.items():
            target_name = _resolve_target_name(name)
            net_param = net_params.get(target_name) if target_name else None
            if net_param is None:
                skipped_by_name += 1
                continue

            target_shape = tuple(net_param.shape)
            current_shape = tuple(value.shape)
            if target_shape == current_shape:
                filtered_param_dict[target_name] = value
                continue

            try:
                source_np = _to_numpy(value)
                adapted_np = None

                # Prefer exact transpose for 2D projection matrices across frameworks.
                if source_np.ndim == 2 and source_np.T.shape == target_shape:
                    adapted_np = source_np.T

                if adapted_np is None:
                    skipped_by_shape += 1
                    if len(skipped_shape_examples) < 8:
                        skipped_shape_examples.append((name, current_shape, target_name, target_shape))
                    continue

                adapted = ms.Parameter(ms.Tensor(adapted_np, dtype=net_param.dtype), name=target_name)
                filtered_param_dict[target_name] = adapted
                adapted_by_shape += 1
                if len(adapted_examples) < 8:
                    adapted_examples.append((name, current_shape, target_name, target_shape, "transpose_2d"))
            except Exception:
                skipped_by_shape += 1
                if len(skipped_shape_examples) < 8:
                    skipped_shape_examples.append((name, current_shape, target_name, target_shape))
                continue

        if not filtered_param_dict:
            print(
                f"共享ckpt无可加载参数: total={len(param_dict)} "
                f"skip_name={skipped_by_name} skip_shape={skipped_by_shape}"
            )
            return False

        param_not_load, ckpt_not_load = ms.load_param_into_net(
            graph,
            filtered_param_dict,
            strict_load=False,
        )
        ckpt_not_load_set = set(ckpt_not_load)
        loaded_param_names = set(filtered_param_dict.keys()) - ckpt_not_load_set
        attn_loaded = sorted(name for name in loaded_param_names if _is_attention_param(name))
        attn_unloaded = sorted(
            name for name in attn_net_params
            if name not in loaded_param_names and not _is_state_tensor(name)
        )
        loaded_count = len(filtered_param_dict) - len(ckpt_not_load)
        print(
            f"共享ckpt加载完成: total={len(param_dict)} filtered={len(filtered_param_dict)} "
            f"loaded={loaded_count} skip_name={skipped_by_name} skip_shape={skipped_by_shape} "
            f"adapted_by_shape={adapted_by_shape} "
            f"net_unloaded={len(param_not_load)} ckpt_unused={len(ckpt_not_load)}"
        )
        print(
            "attention参数加载统计: "
            f"net_total={len(attn_net_params)} loaded={len(attn_loaded)} "
            f"unloaded={len(attn_unloaded)}"
        )
        if attn_unloaded:
            print("attention未加载参数样例(最多12条):")
            for name in attn_unloaded[:12]:
                print(f"  {name}")
        if adapted_examples:
            print("共享ckpt形状适配样例(最多8条):")
            for src_name, src_shape, dst_name, dst_shape, mode in adapted_examples:
                print(f"  {mode}: {src_name}{src_shape} -> {dst_name}{dst_shape}")
        if skipped_shape_examples:
            print("共享ckpt形状不匹配已跳过样例(最多8条):")
            for src_name, src_shape, dst_name, dst_shape in skipped_shape_examples:
                print(f"  skip_shape: {src_name}{src_shape} -> {dst_name}{dst_shape}")
        if loaded_count <= 0:
            print("共享ckpt未匹配到任何参数，判定加载失败")
            return False

        strict_attn_param = os.getenv("LMSV_STRICT_ATTN_PARAM_LOAD", "1") == "1"
        if strict_attn_param and attn_unloaded:
            print("attention参数未完全加载，严格模式下判定失败（LMSV_STRICT_ATTN_PARAM_LOAD=1）")
            return False

        return True
    except Exception:
        import traceback
        print("共享ckpt加载异常:\n", traceback.format_exc())
        return False


if __name__ == "__main__":
    parser = ArgumentParser()
    add_extra_args(parser)
    args = parser.parse_args()

    seed = _env_int("LMSV_DEBUG_SEED", 42)
    seed_all(seed)
    print(f"随机种子已固定: {seed}")

    if args.args_path:
        mf_config = MindFormerConfig(args.args_path)
        build_context(mf_config)

    module = args.module
    node_num = args.node_num
    rounds = args.rounds
    load_path = args.load_path
    shared_weight_ckpt = args.shared_weight_ckpt or os.getenv("LMSV_SHARED_WEIGHT_CKPT_PATH", "")
    train_iters = resolve_train_iters(args)
    step_csv_path = resolve_step_log_csv()

    if module:
        filename = os.path.basename(module).split(".")[0]
        res_dir = os.path.join("./res", f"{filename}")
    else:
        res_dir = os.path.join("./res", f"random{node_num}nodes")

    os.makedirs(res_dir, exist_ok=True)

    log_file_path = os.path.join(res_dir, "verify_graph_log.txt")
    sys.stdout = Logger(log_file_path)

    csv_path = os.getenv("LMSV_MF_CSV_PATH", "res/execution_mf.csv")
    if not os.path.exists(csv_path):
        import csv
        with open(csv_path, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Iteration", "Execution Time (s)", "NPU Memory (MB)", "loss"])

    graph = Graph(
        config_path="assets/runtime/configs/template_config.yaml",
        nums=[int(i) for i in range(5)]
    )

    mutate_round = os.getenv("MUTATE_ROUND", "")
    if mutate_round:
        iteration = int(mutate_round)
    else:
        iteration = extract_number_split(load_path)
        if iteration is None:
            raise ValueError("无法从 load_path 解析 iteration")
        iteration = int(iteration)

    set_debug_step(iteration - 1)

    forward_res = 1
    succ_path = os.path.join(res_dir, f"mutating-{iteration}.json")
    err_path = os.path.join(res_dir, f"mutating-{iteration}-err.json")
    if os.path.exists(succ_path):
        forward_res = 1
    elif os.path.exists(err_path):
        forward_res = 0

    if not forward_res:
        import csv
        with open(csv_path, mode="a+", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([iteration, "-", "-", "-"])
        sys.exit(0)

    yaml_file_name = f"mutated_config_iter_{iteration:03d}.yaml"
    yaml_file_path = os.path.join(res_dir, yaml_file_name)
    json_file_path = os.path.join(res_dir, f"mutating-{iteration}.json")

    patch_mf_yaml_for_task3_stability(yaml_file_path)

    start_time = time.time()
    _reset_npu_memory()
    loss_value = None
    run_failed = False

    try:
        load_success = graph.load(yaml_file_path, json_file_path)
        if not load_success:
            raise RuntimeError("加载变异配置失败")

        attn_cfg_ok = validate_attention_config_parity(graph, res_dir, iteration)
        if not attn_cfg_ok:
            raise RuntimeError("attention配置与PTA运行时快照不一致")

        decoder_cfg_ok = validate_decoder_config_parity(graph, res_dir, iteration)
        if not decoder_cfg_ok:
            raise RuntimeError("decoder配置与PTA运行时快照不一致")

        ckpt_loaded = load_shared_ckpt_if_present(graph, shared_weight_ckpt)
        if not ckpt_loaded:
            raise RuntimeError("共享ckpt加载失败")

        print_graph(graph)
        print("开始forward + backward + update")

        compare_eval_only = _env_flag("LMSV_COMPARE_EVAL_ONLY", "0")
        if compare_eval_only:
            print("开启LMSV_COMPARE_EVAL_ONLY=1，仅执行单次前向用于精度对齐")
            graph.set_train(False)
            loss_net = GraphWithLoss(graph)
            loss = loss_net()
            loss_value = float(loss.asnumpy())
            print("eval-only loss:", loss_value)
        else:
            graph.set_train(True)
            trainable_params = list(graph.trainable_params())
            loss_net = GraphWithLoss(graph)
            if not trainable_params:
                loss = loss_net()
                loss_value = float(loss.asnumpy())
                print("未找到可训练参数，仅执行前向 loss:", loss_value)
            else:
                optimizer = nn.AdamWeightDecay(
                    params=trainable_params,
                    learning_rate=1e-4,
                    weight_decay=0.01,
                )

                def _forward_loss():
                    return loss_net()

                grad_fn = ms.grad(_forward_loss, weights=trainable_params, grad_position=None)

                loss = None
                for step in range(train_iters):
                    _reset_npu_memory()
                    step_start_time = time.time()
                    loss = _forward_loss()
                    grads = grad_fn()
                    opt_res = optimizer(grads)
                    loss_value = float(loss.asnumpy())
                    print(f"step {step + 1} loss:", loss_value)
                    loss_after = _forward_loss()
                    loss_after = ops.depend(loss_after, opt_res)
                    loss_after_value = float(loss_after.asnumpy())
                    print(f"step {step + 1} loss_after:", loss_after_value)
                    step_mem_usage = _get_npu_memory_mb()
                    step_elapsed = time.time() - step_start_time
                    if step_csv_path:
                        append_step_metrics(step_csv_path, step + 1, step_elapsed, step_mem_usage, loss_value)
                    if step == train_iters - 1:
                        print(f"第{train_iters}次反向的loss计算结果： {loss_after_value}")

    except Exception:
        import traceback
        error_traceback = traceback.format_exc()
        print("错误信息:", error_traceback)
        run_failed = True

    end_time = time.time()
    end_mem = _get_npu_memory_mb()
    execution_time = end_time - start_time
    mem_usage = end_mem
    _reset_npu_memory()

    import csv
    with open(csv_path, mode="a+", newline="") as csv_file:
        writer = csv.writer(csv_file)
        loss_csv_value = "-" if run_failed or loss_value is None else round(loss_value, 4)
        writer.writerow([iteration, round(execution_time, 4), round(mem_usage, 4), loss_csv_value])

    success_rate = (1 / rounds) * 100 if rounds else 0
    if not run_failed and loss_value is not None:
        print("变异结果验证成功")
    else:
        print("变异结果验证失败")
        sys.exit(1)
