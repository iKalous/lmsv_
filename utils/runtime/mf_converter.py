#!/usr/bin/env python3
"""Convert Megatron-style pretrain shell scripts into MindFormers YAML configs."""

from __future__ import annotations

import argparse
import copy
import os
import random
import re
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from utils.runtime.paths import CONFIG_DIR, MF_TEMPLATE_DIR

DEFAULT_MAX_DEVICE_MEMORY_GB = 58

ARG_MAPPING_CONFIG_PATH = CONFIG_DIR / "mf_converter_mapping.yaml"
MF_MUTATION_CONFIG_PATH = CONFIG_DIR / "mf_mutation_optimization.yaml"


def _load_arg_mapping(config_path: Path = ARG_MAPPING_CONFIG_PATH) -> Dict[str, tuple]:
    if not config_path.exists():
        raise FileNotFoundError(f"参数映射配置不存在: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    mapping_section = raw.get("arg_mapping")
    if not isinstance(mapping_section, dict):
        raise ValueError(f"参数映射配置格式错误: 缺少 arg_mapping 字典 ({config_path})")

    mapping: Dict[str, tuple] = {}
    for arg_key, cfg in mapping_section.items():
        if not isinstance(arg_key, str) or not arg_key:
            raise ValueError(f"参数映射项键非法: {arg_key!r} ({config_path})")
        if not isinstance(cfg, dict):
            raise ValueError(f"参数映射项必须是字典: {arg_key} ({config_path})")

        target = cfg.get("target")
        if target in (None, ""):
            continue
        if not isinstance(target, list) or not target:
            raise ValueError(f"参数映射 target 必须是非空列表: {arg_key} ({config_path})")

        normalized_target: List[Any] = []
        for item in target:
            if isinstance(item, (str, int)):
                normalized_target.append(item)
            else:
                raise ValueError(
                    f"参数映射 target 仅支持 str/int: {arg_key} -> {target!r} ({config_path})"
                )
        mapping[arg_key] = tuple(normalized_target)

    return mapping


ARG_MAPPING = _load_arg_mapping()


def _load_mf_mutation_config(config_path: Path = MF_MUTATION_CONFIG_PATH) -> Dict[str, Any]:
    if not config_path.exists():
        return {}

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"MF 变异优化配置格式错误: {config_path}")

    parameters = raw.get("parameters")
    if not isinstance(parameters, list):
        raise ValueError(f"MF 变异优化配置缺少 parameters 列表: {config_path}")

    normalized_params: List[Dict[str, Any]] = []
    for item in parameters:
        if not isinstance(item, dict):
            raise ValueError(f"MF 变异优化配置项必须是字典: {item!r} ({config_path})")

        target = item.get("target")
        values = item.get("values")
        if not isinstance(target, list) or not target:
            raise ValueError(f"MF 变异优化配置 target 必须是非空列表: {item!r} ({config_path})")
        if not isinstance(values, list) or not values:
            raise ValueError(f"MF 变异优化配置 values 必须是非空列表: {item!r} ({config_path})")

        normalized_target: List[Any] = []
        for key in target:
            if isinstance(key, (str, int)):
                normalized_target.append(key)
            else:
                raise ValueError(f"MF 变异优化配置 target 仅支持 str/int: {item!r} ({config_path})")

        normalized_values: List[Any] = []
        for value in values:
            if isinstance(value, (str, int, float, bool)) or value is None:
                normalized_values.append(value)
            else:
                raise ValueError(f"MF 变异优化配置 values 仅支持标量: {item!r} ({config_path})")

        normalized_params.append(
            {
                "name": item.get("name") or ".".join(str(part) for part in normalized_target),
                "target": tuple(normalized_target),
                "values": tuple(normalized_values),
            }
        )

    supported_models = raw.get("supported_models", [])
    if supported_models not in (None, "") and not isinstance(supported_models, list):
        raise ValueError(f"MF 变异优化配置 supported_models 必须是列表: {config_path}")

    mutation_count = raw.get("mutation_count", {})
    if mutation_count in (None, ""):
        mutation_count = {}
    if not isinstance(mutation_count, dict):
        raise ValueError(f"MF 变异优化配置 mutation_count 必须是字典: {config_path}")

    return {
        "supported_models": [str(item).strip().lower() for item in (supported_models or []) if str(item).strip()],
        "mutation_count": {
            "min": _to_int_safe(mutation_count.get("min"), 1) or 1,
            "max": _to_int_safe(mutation_count.get("max"), 1) or 1,
        },
        "parameters": normalized_params,
    }


def _get_nested_value(root: Dict[str, Any], keys: Iterable[Any]) -> Any:
    current: Any = root
    for key in keys:
        if isinstance(key, int):
            if not isinstance(current, list) or key < 0 or key >= len(current):
                return None
            current = current[key]
        else:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
    return current


def _apply_mf_seed_mutations(
    config: Dict[str, Any],
    all_args: Dict[str, Any],
    model_name: Optional[str],
    schema: Optional[Dict[str, Any]] = None,
) -> None:
    seed = _to_int_safe(all_args.get("seed"), _to_int_safe(os.getenv("BASE_SEED"), None))
    if seed is None:
        return

    mutation_schema = schema if schema is not None else _load_mf_mutation_config()
    if not mutation_schema:
        return

    normalized_model = (model_name or all_args.get("model_name") or "").strip().lower()
    supported_models = mutation_schema.get("supported_models") or []
    if supported_models and normalized_model not in supported_models:
        return

    parameters = mutation_schema.get("parameters") or []
    if not parameters:
        return

    applicable_params = [
        item
        for item in parameters
        if _get_nested_value(config, item.get("target", ())) is not None
    ]
    if not applicable_params:
        return

    count_rule = mutation_schema.get("mutation_count") or {}
    min_count = max(1, _to_int_safe(count_rule.get("min"), 1) or 1)
    max_count = max(min_count, _to_int_safe(count_rule.get("max"), min_count) or min_count)
    mutation_count = min(len(applicable_params), max_count)
    if mutation_count < min_count:
        mutation_count = len(applicable_params)
    if mutation_count <= 0:
        return

    rng = random.Random(f"{seed}:{normalized_model}")
    selected_params = rng.sample(applicable_params, k=mutation_count)

    for param in selected_params:
        target = param.get("target")
        values = list(param.get("values") or [])
        if not target or not values:
            continue

        current_value = _get_nested_value(config, target)
        candidates = [value for value in values if value != current_value]
        if not candidates:
            candidates = values
        selected_value = candidates[rng.randrange(len(candidates))]
        set_nested_value(config, target, selected_value)


def _resolve_runtime_global_batch_size(
    micro_bs: int,
    data_parallel: int,
    micro_batch_num: int,
) -> int:
    micro_bs = max(1, int(micro_bs))
    data_parallel = max(1, int(data_parallel))
    micro_batch_num = max(1, int(micro_batch_num))
    return micro_bs * data_parallel * micro_batch_num


def _resolve_mf_train_dataset_sample_size(
    train_iters: int,
    resume_iteration: int,
    global_batch_size: int,
    runtime_global_batch_size: int,
) -> int:
    effective_train_iters = max(1, int(train_iters) - max(0, int(resume_iteration)))
    effective_batch_size = max(1, int(global_batch_size), int(runtime_global_batch_size))
    return max(1, effective_train_iters * effective_batch_size)


SECTION_NAMES = (
    "GPT_ARGS",
    "DATA_ARGS",
    "OUTPUT_ARGS",
    "MLA_ARGS",
    "MOE_ARGS",
    "ROPE_ARGS",
    "PATH_ARGS",
    "TRAIN_ARGS",
    "MODEL_PARALLEL_ARGS",
    "OPTIMIZE_ARGS",
    "DISTRIBUTED_ARGS",
)


def try_convert_value(text: str) -> Any:
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        if any(ch in text.lower() for ch in (".", "e")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _to_int_safe(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None:
            return default
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        text = str(value).strip()
        if text == "":
            return default
        return int(float(text))
    except Exception:
        return default


def _parse_memory_gb(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip().upper()
    match = re.fullmatch(r"([0-9]+)\s*GB", text)
    if match:
        return int(match.group(1))
    if text.isdigit():
        return int(text)
    return None


def _resolve_max_device_memory(existing: Any) -> str:
    env_gb = _to_int_safe(os.getenv("LMSV_MF_MAX_DEVICE_MEMORY_GB"), None)
    if env_gb is None:
        env_gb = _parse_memory_gb(os.getenv("LMSV_MF_MAX_DEVICE_MEMORY"))
    target_gb = env_gb if env_gb and env_gb > 0 else DEFAULT_MAX_DEVICE_MEMORY_GB

    current_gb = _parse_memory_gb(existing)
    if current_gb is None or current_gb <= 0:
        resolved_gb = target_gb
    else:
        # Raise conservative template defaults (for example 30GB) to runtime target.
        resolved_gb = max(current_gb, target_gb)

    return f"{resolved_gb}GB"


def _evaluate_shell_arithmetic(value: str) -> str:
    text = value.strip()
    if not (text.startswith("$((") and text.endswith("))")):
        return text
    expr = text[3:-2].strip()
    if not re.fullmatch(r"[0-9+\-*/%() \t]+", expr):
        return text
    try:
        return str(int(eval(expr, {"__builtins__": {}}, {})))
    except Exception:
        return text


def extract_variable_assignments(content: str) -> Dict[str, str]:
    variables: Dict[str, str] = {}
    pattern = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            line = line.split("#", 1)[0].rstrip()
        match = pattern.match(line)
        if not match:
            continue
        name = match.group(1)
        value = match.group(2).strip()
        if value.startswith(('"', "'")) and value.endswith(('"', "'")):
            value = value[1:-1]
        value = expand_variables_in_string(value, variables)
        value = _evaluate_shell_arithmetic(value)
        variables[name] = value
    return variables


def expand_variables_in_string(text: str, variables: Dict[str, str]) -> str:
    def replace_var(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        return variables.get(name, match.group(0))

    pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")
    return pattern.sub(replace_var, text)


def extract_args_section(content: str, args_name: str, variables: Dict[str, str]) -> str:
    pattern = re.compile(rf"{re.escape(args_name)}\s*=\s*[\"\']((?:\\\s*\n)?[\s\S]*?)(?=[\"\']\s*(?:\n|$))", re.DOTALL)
    match = pattern.search(content)
    if not match:
        return ""
    args_str = match.group(1).strip()
    if args_str.startswith("\\"):
        args_str = args_str[1:].lstrip()
    return expand_variables_in_string(args_str, variables)


def parse_args_string(args_str: str) -> Dict[str, Any]:
    flattened = re.sub(r"\\\s*\n\s*", " ", args_str)
    try:
        parts = shlex.split(flattened, posix=True)
    except ValueError:
        parts = flattened.split()

    # Shell scripts in this repo often spell boolean flags as ``--flag \``
    # to keep line continuations uniform. A standalone backslash is not a
    # real argument value, so drop it before pairing flags with values.
    parts = [part for part in parts if part != "\\"]

    parsed: Dict[str, Any] = {}
    index = 0
    while index < len(parts):
        part = parts[index]
        if not part.startswith("--"):
            index += 1
            continue
        key = part[2:].replace("-", "_")
        if index + 1 < len(parts) and not parts[index + 1].startswith("--"):
            parsed[key] = try_convert_value(parts[index + 1])
            index += 2
        else:
            parsed[key] = True
            index += 1
    return parsed


def _read_resume_iteration(load_path: Any) -> int:
    try:
        if not isinstance(load_path, str) or not load_path.strip() or load_path in {"None", "none"}:
            return 0
        tracker = Path(load_path) / "latest_checkpointed_iteration.txt"
        if not tracker.exists():
            return 0
        return max(0, _to_int_safe(tracker.read_text(encoding="utf-8").strip(), 0) or 0)
    except Exception:
        return 0


def _resolve_world_size(all_args: Dict[str, Any], default_world_size: int = 8) -> int:
    world_size = _to_int_safe(all_args.get("world_size"), None)
    if world_size is None or world_size < 1:
        npus = _to_int_safe(all_args.get("npus_per_node"), None)
        nnodes = _to_int_safe(all_args.get("nnodes"), None)
        if npus is not None and npus > 0 and nnodes is not None and nnodes > 0:
            world_size = int(npus) * int(nnodes)
    if world_size is None or world_size < 1:
        world_size = default_world_size
    return max(1, int(world_size))


def _validate_qwen3_head_dim(model_cfg: Dict[str, Any], all_args: Dict[str, Any]) -> None:
    """Align and validate qwen3 attention dimensions at conversion time.

    Priority:
    1. Explicit ``kv_channels`` from source script.
    2. Derived value from ``hidden_size // num_attention_heads``.

    This prevents template default ``head_dim`` from silently drifting away from
    mutated PTA arguments.
    """
    hidden_size = _to_int_safe(model_cfg.get("hidden_size"), None)
    num_heads = _to_int_safe(model_cfg.get("num_attention_heads"), None)
    kv_channels = _to_int_safe(all_args.get("kv_channels"), None)

    if hidden_size is None or num_heads is None or num_heads <= 0:
        return

    if hidden_size % num_heads != 0 and kv_channels is None:
        raise ValueError(
            "Qwen3 参数不一致: hidden_size 不能被 num_attention_heads 整除，"
            f"且未提供 kv_channels。hidden_size={hidden_size}, num_attention_heads={num_heads}"
        )

    expected_head_dim = kv_channels if kv_channels is not None else hidden_size // num_heads
    model_cfg["head_dim"] = int(expected_head_dim)


def _validate_qwen3_config_consistency(model_cfg: Dict[str, Any], all_args: Dict[str, Any]) -> None:
    """Fail early if converted MF config diverges from parsed PTA arguments."""
    mismatch_msgs: List[str] = []

    def _check(name: str, mf_key: str, src_key: str) -> None:
        src_v = _to_int_safe(all_args.get(src_key), None)
        mf_v = _to_int_safe(model_cfg.get(mf_key), None)
        if src_v is not None and mf_v is not None and src_v != mf_v:
            mismatch_msgs.append(
                f"{name} 不一致: source({src_key})={src_v}, mf(model_config.{mf_key})={mf_v}"
            )

    _check("hidden_size", "hidden_size", "hidden_size")
    _check("num_attention_heads", "num_attention_heads", "num_attention_heads")
    _check("ffn_hidden_size/intermediate_size", "intermediate_size", "ffn_hidden_size")

    src_kv = _to_int_safe(all_args.get("kv_channels"), None)
    mf_head_dim = _to_int_safe(model_cfg.get("head_dim"), None)
    if src_kv is not None and mf_head_dim is not None and src_kv != mf_head_dim:
        mismatch_msgs.append(
            f"kv_channels/head_dim 不一致: source(kv_channels)={src_kv}, mf(model_config.head_dim)={mf_head_dim}"
        )

    if mismatch_msgs:
        raise ValueError("Qwen3 转换参数校验失败: " + "; ".join(mismatch_msgs))


def _build_nonnegative_pipeline_offset(num_layers: int, pp: int) -> List[int]:
    if pp <= 0:
        return []
    if num_layers < pp:
        raise ValueError(
            f"Pipeline stage 数量不能大于总层数: num_hidden_layers={num_layers}, pipeline_stage={pp}"
        )

    base_layers = num_layers // pp
    remainder = num_layers % pp
    if base_layers < 1:
        raise ValueError(
            f"Some middle stage has fewer than 1 layer. num_hidden_layers={num_layers}, pipeline_stage={pp}"
        )

    # Use non-negative per-stage adjustments so every stage keeps at least one layer.
    # For evenly divisible models (e.g. 28 layers / 4 stages), this becomes [0, 0, 0, 0].
    return [1 if index < remainder else 0 for index in range(pp)]


def set_nested_value(root: Dict[str, Any], keys: Iterable[Any], value: Any) -> None:
    current: Any = root
    keys_list = list(keys)
    for idx, key in enumerate(keys_list):
        last = idx == len(keys_list) - 1
        if isinstance(key, int):
            if not isinstance(current, list):
                raise TypeError(f"Expected list for key {key}, got {type(current)}")
            while len(current) <= key:
                current.append({})
            if last:
                current[key] = value
            else:
                current = current[key]
        else:
            if not isinstance(current, dict):
                raise TypeError(f"Expected dict for key {key}, got {type(current)}")
            if last:
                current[key] = value
            else:
                current = current.setdefault(key, {})


def _build_dataset_strategy(config: Dict[str, Any], dp: int) -> List[List[int]]:
    """Build dataset strategy entries aligned with the real input columns.

    MindSpore requires `len(dataset_strategy)` to equal the number of dataset
    input tensors. The previous hardcoded 6-item strategy breaks when the
    generated yaml uses 5 input columns.
    """
    train_dataset = config.get("train_dataset") or {}
    input_columns = train_dataset.get("input_columns")
    if not isinstance(input_columns, list) or not input_columns:
        construct_args_key = train_dataset.get("construct_args_key")
        if isinstance(construct_args_key, list) and construct_args_key:
            input_columns = construct_args_key

    if not isinstance(input_columns, list) or not input_columns:
        # Fallback for old templates with unspecified columns.
        return [[dp, 1], [dp, 1], [dp, 1], [dp, 1], [dp, 1], [dp, 1, 1, 1]]

    strategy: List[List[int]] = []
    for column in input_columns:
        if str(column) == "attention_mask":
            strategy.append([dp, 1, 1, 1])
        else:
            strategy.append([dp, 1])
    return strategy


def load_template_config(template_path: str) -> Dict[str, Any]:
    with open(template_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _apply_recompute_args(config: Dict[str, Any], all_args: Dict[str, Any]) -> None:
    """Normalize recompute args from internal extra config and bridge to MF runtime config.

    Mapping writes recompute knobs into ``__lmsv_extra__.recompute`` first,
    then this helper translates them into MindFormers-recognized fields.
    By default, recompute knobs are not mixed into ``model.model_config``.
    Set ``LMSV_MF_RECOMPUTE_MIRROR_MODEL_CONFIG=1`` to mirror them for
    compatibility experiments.
    """
    recompute_cfg = config.setdefault("recompute_config", {})
    extra_root = config.pop("__lmsv_extra__", None)
    recompute_extra: Dict[str, Any] = {}
    if isinstance(extra_root, dict):
        raw = extra_root.get("recompute")
        if isinstance(raw, dict):
            recompute_extra = raw

    def _pick(arg_key: str, extra_key: str) -> Any:
        if arg_key in all_args:
            return all_args.get(arg_key)
        return recompute_extra.get(extra_key)

    granularity_raw = _pick("recompute_granularity", "granularity")
    granularity = str(granularity_raw).strip().lower() if granularity_raw is not None else ""

    method_raw = _pick("recompute_method", "method")
    method = str(method_raw).strip().lower() if method_raw is not None else ""

    num_layers_raw = _pick("recompute_num_layers", "num_layers")
    recompute_num_layers = _to_int_safe(num_layers_raw, None)

    activation_raw = _pick("recompute_activation_function", "activation_function")
    reuse_raw = _pick("reuse_fp32_param", "reuse_fp32_param")

    has_recompute_detail = any(
        item is not None and item != ""
        for item in (granularity_raw, method_raw, num_layers_raw, activation_raw)
    )
    if has_recompute_detail:
        recompute_cfg["recompute"] = True

    if granularity == "selective":
        recompute_cfg["select_recompute"] = True
    elif granularity == "full":
        recompute_cfg["select_recompute"] = False

    # Optional compatibility mirror for runtimes that still expect these
    # knobs under model_config.
    mirror_to_model_cfg = os.getenv("LMSV_MF_RECOMPUTE_MIRROR_MODEL_CONFIG", "0") == "1"
    if mirror_to_model_cfg:
        model_cfg = config.setdefault("model", {}).setdefault("model_config", {})
        if granularity:
            model_cfg["recompute_granularity"] = granularity
        if method:
            model_cfg["recompute_method"] = method
        if recompute_num_layers is not None:
            model_cfg["recompute_num_layers"] = recompute_num_layers
        if activation_raw is not None:
            model_cfg["recompute_activation_function"] = bool(activation_raw)
        if reuse_raw is not None:
            model_cfg["reuse_fp32_param"] = bool(reuse_raw)


def _normalize_qwen3_config(config: Dict[str, Any], all_args: Dict[str, Any], dp: int, tp: int, pp: int, cp: int) -> None:
    model_cfg = config.setdefault("model", {}).setdefault("model_config", {})
    model_cfg.pop("load_checkpoint", None)
    model_cfg.pop("use_fused_rotary_pos_emb", None)

    explicit_moe_keys = {
        "num_experts",
        "moe_intermediate_size",
        "moe_router_topk",
        "moe_router_pre_softmax",
        "moe_layer_freq",
        "first_k_dense_replace",
    }
    if not any(key in all_args for key in explicit_moe_keys):
        for key in (
            "n_routed_experts",
            "moe_router_topk",
            "moe_router_pre_softmax",
            "moe_layer_freq",
            "first_k_dense_replace",
            "moe_intermediate_size",
        ):
            model_cfg.pop(key, None)

    # Normalize legacy key name to the current MF field.
    if "fused_rms_norm" not in model_cfg and "use_fused_rmsnorm" in model_cfg:
        model_cfg["fused_rms_norm"] = bool(model_cfg.pop("use_fused_rmsnorm"))

    if "qkv_concat" not in all_args:
        model_cfg["qkv_concat"] = True
    # MindFormers requires flash attention when context_parallel > 1.
    if cp > 1:
        model_cfg["use_flash_attention"] = True
    elif model_cfg.get("use_flash_attention") is None:
        model_cfg["use_flash_attention"] = True
    # Fix: qk_layernorm from ARG_MAPPING may be empty string/None when PTA passes --qk-layernorm as flag
    qk_ln_val = all_args.get("qk_layernorm")
    if qk_ln_val is None or qk_ln_val == "" or qk_ln_val == "\\":
        model_cfg["qk_layernorm"] = True
    else:
        model_cfg["qk_layernorm"] = bool(qk_ln_val)
    if "untie_embeddings_and_output_weights" not in all_args:
        model_cfg["untie_embeddings_and_output_weights"] = False
    untie = bool(model_cfg.get("untie_embeddings_and_output_weights", False))
    model_cfg["tie_word_embeddings"] = not untie

    if "disable_bias_linear" in all_args:
        model_cfg["add_bias_linear"] = not bool(all_args.get("disable_bias_linear"))
    if "swiglu" in all_args:
        model_cfg["hidden_act"] = "swiglu"

    num_heads = _to_int_safe(model_cfg.get("num_attention_heads"), None)
    num_query_groups = _to_int_safe(all_args.get("num_query_groups"), None)
    if bool(all_args.get("group_query_attention", False)) and num_query_groups:
        model_cfg["num_key_value_heads"] = num_query_groups
    elif num_heads:
        model_cfg["num_key_value_heads"] = num_heads

    kv_channels = _to_int_safe(all_args.get("kv_channels"), None)
    if kv_channels:
        model_cfg["head_dim"] = kv_channels

    _validate_qwen3_head_dim(model_cfg, all_args)

    if "init_method_std" in all_args:
        model_cfg["initializer_range"] = float(all_args["init_method_std"])
    if "moe_router_topk" in all_args:
        model_cfg["moe_router_topk"] = _to_int_safe(all_args["moe_router_topk"], all_args["moe_router_topk"])
    if "moe_router_pre_softmax" in all_args:
        model_cfg["moe_router_pre_softmax"] = bool(all_args["moe_router_pre_softmax"])

    if tp == 1:
        config.setdefault("parallel_config", {})["use_seq_parallel"] = False

    num_layers = _to_int_safe(model_cfg.get("num_hidden_layers"), None)
    if isinstance(model_cfg.get("offset"), list) and num_layers and pp > 0:
        model_cfg["offset"] = _build_nonnegative_pipeline_offset(num_layers, pp)

    parallel = config.setdefault("parallel", {})
    parallel.setdefault("enable_alltoall", True)
    parallel["dataset_strategy"] = _build_dataset_strategy(config, dp)

    context = config.setdefault("context", {})
    context["max_device_memory"] = _resolve_max_device_memory(context.get("max_device_memory"))
    ascend = context.setdefault("ascend_config", {})
    ascend.setdefault("parallel_speed_up_json_path", "assets/runtime/configs/qwen3/parallel_speed_up.json")


def update_template_with_args(template_config: Dict[str, Any], all_args: Dict[str, Any]) -> Dict[str, Any]:
    config = copy.deepcopy(template_config)

    if "seed" in all_args:
        config["seed"] = _to_int_safe(all_args["seed"], all_args["seed"])
    if "save" in all_args and all_args["save"] not in (None, "", "None"):
        config["output_dir"] = all_args["save"]
    if "load" in all_args and all_args["load"] not in (None, "", "None"):
        config["load_checkpoint"] = all_args["load"]
    if "data_path" in all_args:
        train_dataset = config.setdefault("train_dataset", {})
        data_loader = train_dataset.setdefault("data_loader", {})
        data_cfg = data_loader.setdefault("config", {})
        data_cfg["data_path"] = ["1", all_args["data_path"]]
    if "seq_length" in all_args:
        train_dataset = config.setdefault("train_dataset", {})
        data_loader = train_dataset.setdefault("data_loader", {})
        data_cfg = data_loader.setdefault("config", {})
        data_cfg["seq_length"] = all_args["seq_length"]
    config["compute_dtype"] = "bfloat16" if all_args.get("bf16") else "float32"

    for arg_key, mapping in ARG_MAPPING.items():
        if arg_key not in all_args:
            continue
        if not mapping or mapping[0] is None:
            continue
        set_nested_value(config, mapping, all_args[arg_key])

    _apply_recompute_args(config, all_args)

    model_cfg = config.setdefault("model", {}).setdefault("model_config", {})
    context = config.setdefault("context", {})
    context["max_device_memory"] = _resolve_max_device_memory(context.get("max_device_memory"))
    if "model_name" in all_args:
        trainer = config.setdefault("trainer", {})
        trainer["model_name"] = all_args["model_name"]
    if "clip_grad" in all_args:
        config.setdefault("runner_wrapper", {})["use_clip_grad"] = True

    # Keep dataset and model sequence length consistent.
    # Mismatch here can trigger compile-time shape errors in MF graph mode.
    seq_length = _to_int_safe(
        all_args.get("seq_length", model_cfg.get("seq_length")),
        None,
    )
    if seq_length is not None and seq_length > 0:
        model_cfg["seq_length"] = int(seq_length)
        train_dataset = config.setdefault("train_dataset", {})
        data_loader = train_dataset.setdefault("data_loader", {})
        data_cfg = data_loader.setdefault("config", {})
        data_cfg["seq_length"] = int(seq_length)

    world_size = _resolve_world_size(all_args)
    tp = _to_int_safe(
        all_args.get("tensor_model_parallel_size", all_args.get("tensor_parallel_size")),
        1,
    ) or 1
    pp = _to_int_safe(
        all_args.get("pipeline_model_parallel_size", all_args.get("pipeline_parallel_size")),
        1,
    ) or 1
    cp = _to_int_safe(all_args.get("context_parallel_size"), 1) or 1
    ep = _to_int_safe(
        all_args.get("expert_model_parallel_size", all_args.get("expert_parallel_size")),
        1,
    ) or 1
    denom = max(1, tp * pp * cp)
    dp = max(1, world_size // denom)

    parallel_config = config.setdefault("parallel_config", {})
    parallel_config["data_parallel"] = dp
    parallel_config["model_parallel"] = tp
    parallel_config["pipeline_stage"] = pp
    parallel_config["context_parallel"] = cp
    parallel_config["expert_parallel"] = ep
    parallel_config["use_seq_parallel"] = bool(all_args.get("sequence_parallel", False)) and tp > 1

    parallel = config.setdefault("parallel", {})
    parallel["enable_parallel_optimizer"] = bool(all_args.get("use_distributed_optimizer", False))
    parallel["dataset_strategy"] = _build_dataset_strategy(config, dp)

    # MindSpore/MindFormers requires flash attention when context parallel > 1.
    # Apply this constraint for all model types, not only qwen3.
    if cp > 1:
        model_cfg["use_flash_attention"] = True

    if "model_type" in all_args and str(all_args["model_type"]).lower() == "qwen3":
        _normalize_qwen3_config(config, all_args, dp, tp, pp, cp)
        _validate_qwen3_config_consistency(model_cfg, all_args)

    disable_deepseek_hf_load = False

    # DeepSeek config may carry alias or unsupported keys from template/checkpoint
    # internals, which can trigger MindFormers strict key-conversion failures.
    model_type_lower = str(model_cfg.get("model_type", "") or "").lower()
    if "deepseek" in model_type_lower:
        # Explicitly carry over MoE scale knobs from PTA args.
        # Use DeepSeek alias/source keys to avoid MF "Multiple Config" collisions.
        routed_experts = _to_int_safe(all_args.get("num_experts"), None)
        if routed_experts is not None and routed_experts > 0:
            model_cfg["n_routed_experts"] = int(routed_experts)
            model_cfg.pop("num_moe_experts", None)

        moe_ffn = _to_int_safe(all_args.get("moe_intermediate_size"), None)
        if moe_ffn is not None and moe_ffn > 0:
            model_cfg["moe_intermediate_size"] = int(moe_ffn)
            model_cfg.pop("moe_ffn_hidden_size", None)

        shared_experts = _to_int_safe(all_args.get("n_shared_experts"), None)
        if shared_experts is not None and shared_experts >= 0:
            model_cfg["n_shared_experts"] = int(shared_experts)
            model_cfg.pop("shared_expert_num", None)

        router_topk = _to_int_safe(all_args.get("moe_router_topk"), None)
        if router_topk is not None and router_topk > 0:
            model_cfg["moe_router_topk"] = int(router_topk)

        for key in (
            "num_experts",
            "num_experts_per_tok",
            "num_experts_chosen",
            "moe_router_topk",
            "aux_loss_alpha",
            "seq_aux",
            "num_shared_experts",
        ):
            model_cfg.pop(key, None)

        # MindFormers deepseek3 currently requires grouped GEMM enabled for MoE.
        model_cfg["moe_grouped_gemm"] = True

        # Keep offset aligned with the current pipeline parallel degree.
        # DeepSeek/MindFormers now rejects negative offsets, so rebuild it as
        # a non-negative per-stage adjustment list.
        if pp > 0 and isinstance(model_cfg.get("offset"), list):
            deepseek_layers = _to_int_safe(model_cfg.get("num_hidden_layers"), 0) or 0
            model_cfg["offset"] = _build_nonnegative_pipeline_offset(deepseek_layers, pp)

        if model_cfg.get("mtp_loss_scaling_factor") in (None, "None"):
            model_cfg["mtp_loss_scaling_factor"] = 1.0

        # Current MF training class for deepseek3 does not implement
        # safetensors conversion hook (convert_weight_dict), so disable direct
        # HF checkpoint loading to keep training flow runnable.
        disable_deepseek_hf_load = True

        # Some templates may still place these keys under model root.
        model_root = config.get("model", {})
        if isinstance(model_root, dict):
            for key in ("aux_loss_alpha", "seq_aux", "num_shared_experts"):
                model_root.pop(key, None)

    seed = _to_int_safe(all_args.get("seed"), None)
    if seed is not None:
        config["seed"] = seed
        train_dataset = config.setdefault("train_dataset", {})
        train_dataset["seed"] = seed
        data_loader = train_dataset.setdefault("data_loader", {})
        data_loader.setdefault("config", {})["seed"] = seed

    _apply_mf_seed_mutations(config, all_args, all_args.get("model_name"))
    config.pop("micro_batch_interleave_num", None)

    max_global_bs_exclusive = _to_int_safe(
        os.getenv("LMSV_MAX_GLOBAL_BATCH_SIZE_EXCLUSIVE", "64"),
        64,
    ) or 64
    if max_global_bs_exclusive < 2:
        max_global_bs_exclusive = 64
    max_global_bs = max_global_bs_exclusive - 1

    micro_bs = _to_int_safe(all_args.get("micro_batch_size"), 1) or 1
    raw_global_bs = _to_int_safe(all_args.get("global_batch_size"), None)
    if raw_global_bs is None or raw_global_bs < 1:
        raw_global_bs = micro_bs * max(1, dp)
    divisor = max(1, micro_bs * max(1, dp))

    # Keep effective global batch under cap; when divisor itself is too large,
    # shrink micro-batch first to obtain a feasible capped global batch.
    if divisor > max_global_bs:
        micro_bs = max(1, max_global_bs // max(1, dp))
        divisor = max(1, micro_bs * max(1, dp))

    global_bs = raw_global_bs
    if global_bs > max_global_bs:
        global_bs = max(divisor, (max_global_bs // divisor) * divisor)

    grad_acc = max(1, global_bs // max(1, micro_bs * dp))

    # For pipeline parallel training, MindSpore requires micro_batch_num > 1.
    # Keep effective global batch by moving accumulation into micro_batch_num.
    effective_micro_batch_num = 1
    effective_grad_acc = int(grad_acc)
    if pp > 1:
        effective_micro_batch_num = max(2, int(grad_acc))
        effective_grad_acc = 1

    runtime_batch_factor = max(int(effective_grad_acc), int(effective_micro_batch_num))
    runtime_global_bs = _resolve_runtime_global_batch_size(micro_bs, dp, runtime_batch_factor)
    global_bs = max(global_bs, runtime_global_bs)

    parallel_config["micro_batch_num"] = int(effective_micro_batch_num)

    runner_config = config.setdefault("runner_config", {})
    runner_config["batch_size"] = int(micro_bs)
    runner_config["gradient_accumulation_steps"] = int(effective_grad_acc)
    runner_config["epochs"] = 1

    callbacks = config.get("callbacks")
    if isinstance(callbacks, dict):
        callbacks = [callbacks]
    if not isinstance(callbacks, list) or not callbacks:
        callbacks = [{"type": "MFLossMonitor"}]
    found = False
    for callback in callbacks:
        if isinstance(callback, dict) and callback.get("type") == "MFLossMonitor":
            callback["global_batch_size"] = global_bs
            callback["gradient_accumulation_steps"] = int(effective_grad_acc)
            found = True
    if not found:
        callbacks.append({
            "type": "MFLossMonitor",
            "global_batch_size": global_bs,
            "gradient_accumulation_steps": int(effective_grad_acc),
        })
    config["callbacks"] = callbacks

    resume_iteration = _read_resume_iteration(all_args.get("load"))
    train_iters = _to_int_safe(all_args.get("train_iters"), None)
    if train_iters is not None:
        effective_train_iters = max(1, int(train_iters) - max(0, int(resume_iteration)))
        target_samples = _resolve_mf_train_dataset_sample_size(
            train_iters,
            resume_iteration,
            global_bs,
            runtime_global_bs,
        )
        train_dataset = config.setdefault("train_dataset", {})
        data_loader = train_dataset.setdefault("data_loader", {})
        sizes = data_loader.get("sizes")
        if isinstance(sizes, list) and sizes:
            sizes[0] = target_samples
        else:
            data_loader["sizes"] = [target_samples, 0, 0]
        data_loader.setdefault("config", {})["split"] = "1, 0, 0"
        # Keep MF LR progression consistent with PTA when lr-decay-iters is provided.
        lr_decay_iters = _to_int_safe(all_args.get("lr_decay_iters"), None)
        schedule_total_steps = int(effective_train_iters)
        if lr_decay_iters is not None and lr_decay_iters > 0:
            schedule_total_steps = int(lr_decay_iters)
        config.setdefault("lr_schedule", {})["total_steps"] = schedule_total_steps

    config["use_parallel"] = bool(world_size > 1 or tp > 1 or pp > 1 or cp > 1)
    if disable_deepseek_hf_load:
        config["load_checkpoint"] = ""
        config["load_ckpt_format"] = "ckpt"

    has_load_checkpoint = config.get("load_checkpoint") not in (None, "", "None")
    config["auto_trans_ckpt"] = bool(has_load_checkpoint and config.get("use_parallel", False))

    return config


def build_all_args(content: str, model_name: Optional[str], train_iters: Optional[int]) -> Dict[str, Any]:
    variables = extract_variable_assignments(content)
    all_args: Dict[str, Any] = {}

    for section in SECTION_NAMES:
        args_str = extract_args_section(content, section, variables)
        if args_str:
            all_args.update(parse_args_string(args_str))

    save_match = re.search(r"--save\s+([^\s\\]+)", content)
    if save_match:
        all_args["save"] = expand_variables_in_string(save_match.group(1), variables)
    load_match = re.search(r"--load\s+([^\s\\]+)", content)
    if load_match:
        all_args["load"] = expand_variables_in_string(load_match.group(1), variables)

    variable_fallbacks = {
        "DATA_PATH": "data_path",
        "TOKENIZER_PATH": "tokenizer_name_or_path",
        "SEQ_LENGTH": "seq_length",
        "TRAIN_ITERS": "train_iters",
        "NPUS_PER_NODE": "npus_per_node",
        "NNODES": "nnodes",
        "WORLD_SIZE": "world_size",
        "TP": "tensor_model_parallel_size",
        "PP": "pipeline_model_parallel_size",
        "MBS": "micro_batch_size",
        "GBS": "global_batch_size",
        "CKPT_SAVE_DIR": "save",
        "CKPT_LOAD_DIR": "load",
    }
    for var_name, arg_name in variable_fallbacks.items():
        if arg_name in all_args or var_name not in variables:
            continue
        all_args[arg_name] = try_convert_value(variables[var_name])

    if model_name:
        all_args["model_name"] = model_name
    if train_iters is not None:
        all_args["train_iters"] = train_iters

    normalized_model = (model_name or "").strip().lower()
    input_lower = content.lower()
    if normalized_model == "qwen3" or "qwen3" in input_lower:
        all_args.setdefault("model_type", "qwen3")
        all_args.setdefault("architectures", ["Qwen3ForCausalLM"])
        all_args.setdefault("parallel_speed_up_json_path", "assets/runtime/configs/qwen3/parallel_speed_up.json")

    return all_args


def convert_sh_to_yaml(
    input_file: str,
    output_file: Optional[str] = None,
    model_name: Optional[str] = None,
    template: Optional[str] = None,
    train_iters: Optional[int] = None,
) -> str:
    input_path = Path(input_file)
    content = input_path.read_text(encoding="utf-8")
    all_args = build_all_args(content, model_name, train_iters)

    template_path = Path(template) if template else (MF_TEMPLATE_DIR / "basic.yaml")
    if not template_path.exists():
        raise FileNotFoundError(f"模板文件不存在: {template_path}")
    template_config = load_template_config(str(template_path))
    final_config = update_template_with_args(template_config, all_args)

    if output_file is None:
        base_name = model_name or input_path.stem
        output_file = f"{base_name}.yaml"
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.dump(
            final_config,
            handle,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            indent=2,
            width=float("inf"),
        )
    return str(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Megatron pretrain shell to MindFormers yaml")
    parser.add_argument("-i", "--input", required=True, help="Input shell script path")
    parser.add_argument("-o", "--output", help="Output yaml path")
    parser.add_argument("-m", "--model-name", help="Model name for template specialization")
    parser.add_argument(
        "-t",
        "--template",
        default=str(MF_TEMPLATE_DIR / "basic.yaml"),
        help="MindFormers yaml template path",
    )
    parser.add_argument("--train-iters", type=int, help="Override train iters for dataset sizing")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file does not exist: {args.input}", file=sys.stderr)
        return 1

    output_path = convert_sh_to_yaml(
        args.input,
        output_file=args.output,
        model_name=args.model_name,
        template=args.template,
        train_iters=args.train_iters,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
