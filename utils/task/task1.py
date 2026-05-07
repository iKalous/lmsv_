#!/usr/bin/env python3
"""
自动化变异+PTA/MSA训练全流程任务
从 exe_script.sh 重构为 Python 版本
"""

import os
import json
import re
import signal
import subprocess
import threading
import time
import shutil
import shlex
from datetime import datetime
from pathlib import Path

import utils
from ruamel.yaml import YAML
from utils.analyze.precision import find_series_loss_mismatch
from utils.runtime.cluster_runtime import ClusterMaster, parse_task123_cluster_config
from utils.runtime.common_utils import get_card_num
from utils.runtime.model_support import ensure_task1_model_supported_for_mode
from utils.runtime.model_support import resolve_task1_weight_convert_model_alias
from utils.runtime.paths import CONFIG_DIR, MF_TEMPLATE_DIR, MODEL_CONFIG_DIR, MUTATION_SCRIPT_DIR, RUNTIME_SCRIPT_DIR, SCRIPT_TEMPLATE_DIR, repo_rel
from utils.runtime.profiler_tools import generate_profile_report
from utils.task import data_helpers, runtime_helpers

LMSV_ROOT = Path(__file__).resolve().parents[2]
PROJECT_TMP_ROOT = LMSV_ROOT / "tmp"
TASK1_TMP_ROOT = PROJECT_TMP_ROOT / "task1"
RUNTIME_HOOK_DIR = LMSV_ROOT / "utils" / "replace"
MODEL_CONFIG_REL = repo_rel(MODEL_CONFIG_DIR)
MUTATION_SCHEMA_REL = repo_rel(CONFIG_DIR / "mutation_schema.yaml")
MUTATION_SCRIPT_REL = repo_rel(MUTATION_SCRIPT_DIR)
SCRIPT_TEMPLATE_REL = repo_rel(SCRIPT_TEMPLATE_DIR)


# ====================== 配置区 ======================
class Config:
    # 基础配置
    MODEL_NAME = "qwen3"
    DATA_PATH = str(PROJECT_TMP_ROOT / "data" / "wiki_4096_text_document")
    TOTAL_ITER = 10
    BASE_SEED = 43
    MUTNM = 2
    MUTATE_NODE_NUM = 1
    TEST_ITERATIONS = 10
    COMPARE_MODE = "pta_msa"
    
    # 模式配置
    MODE = "DEVELOP"  # DEVELOP 或 TEST
    MAX_ITERATIONS = TOTAL_ITER
    
    # 训练配置
    SUPPORT_MF = True
    ENABLE_WEIGHT_CONVERT = True
    ENABLE_MF_WEIGHT_LOAD = True
    SAVE_TRAIN_ITERS = 1  # SAVE模式训练轮数
    LOAD_TRAIN_ITERS = 30  # LOAD模式训练轮数
    PTA_MAX_RUNTIME = 3000
    TARGET_TENSOR_PARALLEL_SIZE = 2
    TARGET_PIPELINE_PARALLEL_SIZE = 1
    TARGET_EXPERT_PARALLEL_SIZE = 1
    TARGET_NPUS_PER_NODE = 0
    TARGET_WORLD_SIZE = 0
    TARGET_NNODES = 1
    TARGET_NODE_RANK = 0
    TARGET_MASTER_ADDR = "127.0.0.1"
    TARGET_REMOTE_MASTER_ADDR = "127.0.0.1"
    TARGET_MASTER_PORT = 6000
    MF_LOSS_TOLERANCE = 1e-6
    
    # 路径配置
    LOG_PATH = "res/execution.log"
    DUMP_DIR = "res/dump_logs"
    CKPT_ROOT_DIR = str(TASK1_TMP_ROOT / "ckpt")
    MF_CKPT_ROOT_DIR = str(TASK1_TMP_ROOT / "ckpt_mf")
    ACC_LOG_ROOT = "res/accuracy_log"
    TMP_ROOT_DIR = str(TASK1_TMP_ROOT / "test")
    PARALLEL_MUTATE_TMP_DIR = str(TASK1_TMP_ROOT / "parallel_mutate")
    PERSISTENT_LOG_DIR = "res/logs"
    PTA_LOG_SRC_DIR = "res/training_log_pta"
    MSA_LOG_SRC_DIR = "res/training_log_msa"
    MF_LOG_SRC_DIR = "res/training_log_mf"
    MY_PERSIST_ROOT = ""
    
    # MSA监控配置
    MSA_LOG_DIR = "msrun_log"
    MSA_MONITOR_LOGS = ["msrun_log/worker_0.log"]
    MSA_MAX_RUNTIME = 3000
    LOG_INIT_WAIT = 240
    LOG_STABLE_THRESHOLD = 150
    
    # MF分析配置
    MF_ANALYSIS_ENABLE = 1
    MF_ANALYSIS_PLOT = 0
    
    # 环境名称
    MS_ENV = "msadapter"
    PTA_ENV = "mindspeed"
    MF_ENV = "mindf_py311"


# ====================== 日志函数 ======================
LOG_SCOPE = "Task1"
YAML_SAFE = YAML(typ="safe")
SCRIPT_CONSTRAINT_RULES = {
    "position_embedding_type": {"type": "value_arg", "flags": ("--position-embedding-type",)},
    "overlap_grad_reduce": {"type": "bool_flag", "flags": ("--overlap-grad-reduce",)},
    "use_fused_rotary_pos_emb": {"type": "bool_flag", "flags": ("--use-fused-rotary-pos-emb",)},
    "recompute_granularity": {"type": "value_arg", "flags": ("--recompute-granularity",)},
    "recompute_method": {"type": "value_arg", "flags": ("--recompute-method",)},
    "recompute_num_layers": {"type": "int_arg", "flags": ("--recompute-num-layers",)},
    "recompute_activation_function": {"type": "bool_flag", "flags": ("--recompute-activation-function",)},
    "use_distributed_optimizer": {"type": "bool_flag", "flags": ("--use-distributed-optimizer",)},
    "reuse_fp32_param": {"type": "bool_flag", "flags": ("--reuse-fp32-param",)},
    "sequence_parallel": {"type": "bool_flag", "flags": ("--sequence-parallel",)},
    "disable_bias_linear": {"type": "bool_flag", "flags": ("--disable-bias-linear",)},
    "untie_embeddings_and_output_weights": {"type": "bool_flag", "flags": ("--untie-embeddings-and-output-weights",)},
    "use_flash_attn": {"type": "bool_flag", "flags": ("--use-flash-attn",)},
    "use_fused_rmsnorm": {"type": "bool_flag", "flags": ("--use-fused-rmsnorm",)},
    "use_fused_swiglu": {"type": "bool_flag", "flags": ("--use-fused-swiglu",)},
    "use_rotary_position_embeddings": {"type": "bool_flag", "flags": ("--use-rotary-position-embeddings",)},
    "swiglu": {"type": "bool_flag", "flags": ("--swiglu",)},
    "group_query_attention": {"type": "bool_flag", "flags": ("--group-query-attention",)},
    "no_masked_softmax_fusion": {"type": "bool_flag", "flags": ("--no-masked-softmax-fusion",)},
    "no_gradient_accumulation_fusion": {"type": "bool_flag", "flags": ("--no-gradient-accumulation-fusion",)},
    "attention_softmax_in_fp32": {"type": "bool_flag", "flags": ("--attention-softmax-in-fp32",)},
    "overlap_param_gather": {"type": "bool_flag", "flags": ("--overlap-param-gather",)},
    "distributed_backend": {"type": "value_arg", "flags": ("--distributed-backend",)},
    "use_mcore_models": {"type": "bool_flag", "flags": ("--use-mcore-models",)},
    "normalization": {"type": "value_arg", "flags": ("--normalization",)},
    "make_vocab_size_divisible_by": {"type": "int_arg", "flags": ("--make-vocab-size-divisible-by",)},
    "add_bias_linear": {"type": "bool_flag", "flags": ("--add-bias-linear",)},
    "num_query_groups": {"type": "int_arg", "flags": ("--num-query-groups",)},
    "init_method_std": {"type": "float_arg", "flags": ("--init-method-std",)},
    "layernorm_epsilon": {"type": "float_arg", "flags": ("--layernorm-epsilon",)},
    "rotary_base": {"type": "int_arg", "flags": ("--rotary-base",)},
    "add_qkv_bias": {"type": "bool_flag", "flags": ("--add-qkv-bias",)},
    "cuda_device_max_connections": {"type": "env_int", "envs": ("CUDA_DEVICE_MAX_CONNECTIONS",)},
    "pytorch_npu_alloc_conf": {"type": "env_value", "envs": ("PYTORCH_NPU_ALLOC_CONF",)},
    "tensor_parallel_size": {"type": "int_arg", "flags": ("--tensor-model-parallel-size", "--tensor-parallel-size")},
    "pipeline_parallel_size": {"type": "int_arg", "flags": ("--pipeline-model-parallel-size", "--pipeline-parallel-size")},
    "expert_parallel_size": {"type": "int_arg", "flags": ("--expert-model-parallel-size", "--expert-parallel-size")},
    "data_parallel_size": {"type": "int_arg", "flags": ("--data-parallel-size",)},
    "context_parallel_size": {"type": "int_arg", "flags": ("--context-parallel-size",)},
    "first_k_dense_replace": {"type": "int_arg", "flags": ("--first-k-dense-replace",)},
}


def _format_log(tag, msg):
    text = str(msg)
    if tag:
        return f"[{LOG_SCOPE}][{tag}] {text}"
    return f"[{LOG_SCOPE}] {text}"


def log_info(msg):
    utils.log.write.info(_format_log(None, msg))

def log_warn(msg):
    utils.log.write.warn(_format_log(None, msg))

def log_error(msg):
    utils.log.write.error(_format_log(None, msg))

def log_step(msg):
    utils.log.write.info(_format_log("阶段", msg))

def log_acc(msg):
    utils.log.write.info(_format_log("统计", msg))

def log_backup(msg):
    utils.log.write.info(_format_log("归档", msg))


def log_kv(group, key, value):
    utils.log.write.info(_format_log(group, f"{key}: {value}"))


# ====================== 核心工具函数 ======================
def _parse_optional_positive_int(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def _load_script_constraints():
    cache = getattr(_load_script_constraints, "_cache", None)
    if cache is not None:
        return ok




def _pick_allowed_parallel_size(current_value, allowed_values):
    allowed = sorted({_parse_optional_positive_int(value) for value in allowed_values if _parse_optional_positive_int(value) > 0})
    if not allowed:
        return 0
    if current_value in allowed:
        return current_value

    lower_or_equal = [value for value in allowed if value <= current_value]
    if lower_or_equal:
        return max(lower_or_equal)
    return min(allowed)


def _replace_script_int_flag_if_present(script_path, flag_key, new_value):
    path = Path(script_path)
    if not path.exists():
        return False, False

    content = path.read_text(encoding="utf-8")
    pattern = rf"({re.escape(flag_key)}\s+)(\d+)"
    updated, count = re.subn(pattern, rf"\g<1>{int(new_value)}", content, count=1)
    if count <= 0:
        return False, False
    if updated == content:
        return True, False
    path.write_text(updated, encoding="utf-8")
    return True, True


def _replace_script_float_flag_if_present(script_path, flag_key, new_value):
    path = Path(script_path)
    if not path.exists():
        return False, False

    content = path.read_text(encoding="utf-8")
    pattern = rf"({re.escape(flag_key)}\s+)([0-9eE+\-.]+)"
    replacement = f"\\g<1>{new_value}"
    updated, count = re.subn(pattern, replacement, content, count=1)
    if count <= 0:
        return False, False
    if updated == content:
        return True, False
    path.write_text(updated, encoding="utf-8")
    return True, True


def _normalize_allowed_bool_values(allowed_values):
    normalized = set()
    for value in allowed_values:
        if isinstance(value, bool):
            normalized.add(value)
            continue
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "on"}:
            normalized.add(True)
        elif text in {"false", "0", "no", "off"}:
            normalized.add(False)
    return normalized


def _pick_allowed_scalar_value(current_value, allowed_values):
    allowed = [str(value) for value in allowed_values]
    if not allowed:
        return ""
    current_text = str(current_value)
    if current_text in allowed:
        return current_text
    return allowed[0]


def _pick_allowed_float_value(current_value, allowed_values):
    parsed = []
    for value in allowed_values:
        try:
            parsed.append(float(value))
        except (TypeError, ValueError):
            continue
    if not parsed:
        return ""
    if any(abs(current_value - value) <= 1e-12 for value in parsed):
        if float(current_value).is_integer():
            return str(int(current_value))
        return format(current_value, "g")
    lower_or_equal = [value for value in parsed if value <= current_value]
    picked = max(lower_or_equal) if lower_or_equal else min(parsed)
    if float(picked).is_integer():
        return str(int(picked))
    return format(picked, "g")


def _replace_script_value_flag_if_present(script_path, flag_key, new_value):
    path = Path(script_path)
    if not path.exists():
        return False, False

    content = path.read_text(encoding="utf-8")
    pattern = rf"({re.escape(flag_key)}\s+)([^\s\\]+)"
    updated, count = re.subn(pattern, rf"\g<1>{new_value}", content, count=1)
    if count <= 0:
        return False, False
    if updated == content:
        return True, False
    path.write_text(updated, encoding="utf-8")
    return True, True


def _remove_script_flag_if_present(script_path, flag_key):
    path = Path(script_path)
    if not path.exists():
        return False, False

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    updated_lines = []
    removed = False
    for line in lines:
        if re.search(rf"(^|\s){re.escape(flag_key)}(\s|\\|$)", line):
            removed = True
            continue
        updated_lines.append(line)
    if not removed:
        return False, False
    new_content = "".join(updated_lines)
    old_content = "".join(lines)
    if new_content == old_content:
        return True, False
    path.write_text(new_content, encoding="utf-8")
    return True, True


def _replace_script_env_if_present(script_path, env_name, new_value):
    path = Path(script_path)
    if not path.exists():
        return False, False

    content = path.read_text(encoding="utf-8")
    pattern = rf"(export\s+{re.escape(env_name)}=)([^\n]+)"
    updated, count = re.subn(pattern, rf"\g<1>{new_value}", content, count=1)
    if count <= 0:
        return False, False
    if updated == content:
        return True, False
    path.write_text(updated, encoding="utf-8")
    return True, True


def ensure_gqa_tensor_parallel_compatible(script_path):
    """Ensure GQA query groups are divisible by tensor parallel size."""
    path = Path(script_path)
    if not path.exists():
        return False

    content = path.read_text(encoding="utf-8")
    groups_match = re.search(r"--num-query-groups\s+([0-9]+)", content)
    tp_match = re.search(r"--tensor-model-parallel-size\s+([0-9]+)", content)
    if not groups_match or not tp_match:
        return True

    num_query_groups = max(1, int(groups_match.group(1)))
    tensor_parallel = max(1, int(tp_match.group(1)))
    if num_query_groups % tensor_parallel == 0:
        return True

    candidate_tps = [
        value
        for value in range(min(tensor_parallel, num_query_groups), 0, -1)
        if num_query_groups % value == 0
    ]
    adjusted_tp = candidate_tps[0] if candidate_tps else 1
    found, changed = _replace_script_int_flag_if_present(path, "--tensor-model-parallel-size", adjusted_tp)
    if not found:
        return False
    if changed:
        log_info(
            f"GQA并行约束生效 | 脚本: {path.name} | "
            f"--tensor-model-parallel-size {tensor_parallel} -> {adjusted_tp} | "
            f"num_query_groups={num_query_groups}"
        )
    return True


def apply_script_constraints(script_path):
    script_constraints = _load_script_constraints()
    if not script_constraints:
        return ensure_gqa_tensor_parallel_compatible(script_path)

    path = Path(script_path)
    if not path.exists():
        log_error(f"脚本不存在，无法应用 script_constraints 约束: {script_path}")
        return False

    for rule_key, rule_meta in SCRIPT_CONSTRAINT_RULES.items():
        rule = script_constraints.get(rule_key)
        if not isinstance(rule, dict):
            continue

        allowed_values = rule.get("enums")
        if not isinstance(allowed_values, list):
            continue

        content = path.read_text(encoding="utf-8")
        rule_type = rule_meta.get("type")

        if rule_type == "int_arg":
            matched_flag = None
            current_value = 0
            for flag_key in rule_meta.get("flags", ()):
                match = re.search(rf"{re.escape(flag_key)}\s+(\d+)", content)
                if match:
                    matched_flag = flag_key
                    current_value = _parse_optional_positive_int(match.group(1))
                    break
            if not matched_flag or current_value <= 0:
                continue

            if rule_key == "first_k_dense_replace":
                constrained_value = min(current_value, _get_first_k_dense_replace_limit(path))
                if constrained_value == current_value:
                    continue
                found, changed = _replace_script_int_flag_if_present(path, matched_flag, constrained_value)
                if found and changed:
                    log_info(
                        f"脚本约束生效 | 脚本: {path.name} | 参数: {matched_flag} | "
                        f"{current_value} -> {constrained_value}"
                    )
                continue

            constrained_value = _pick_allowed_parallel_size(current_value, allowed_values)
            if constrained_value <= 0 or constrained_value == current_value:
                continue
            found, changed = _replace_script_int_flag_if_present(path, matched_flag, constrained_value)
            if found and changed:
                log_info(f"脚本约束生效 | 脚本: {path.name} | 参数: {matched_flag} | {current_value} -> {constrained_value}")
            continue

        if rule_type == "float_arg":
            matched_flag = None
            current_value = None
            for flag_key in rule_meta.get("flags", ()):
                match = re.search(rf"{re.escape(flag_key)}\s+([0-9eE+\-.]+)", content)
                if match:
                    matched_flag = flag_key
                    try:
                        current_value = float(match.group(1))
                    except ValueError:
                        current_value = None

                if rule_key == "first_k_dense_replace":
                    constrained_value = min(current_value, _get_first_k_dense_replace_limit(path))
                    if constrained_value == current_value:
                        continue
                    found, changed = _replace_script_int_flag_if_present(path, matched_flag, constrained_value)
                    if found and changed:
                        log_info(
                            f"脚本约束生效 | 脚本: {path.name} | 参数: {matched_flag} | "
                            f"{current_value} -> {constrained_value}"
                        )
                    continue

                    break
            if not matched_flag or current_value is None:
                continue
            constrained_value = _pick_allowed_float_value(current_value, allowed_values)
            if not constrained_value or constrained_value == format(current_value, "g"):
                continue
            found, changed = _replace_script_value_flag_if_present(path, matched_flag, constrained_value)
            if found and changed:
                log_info(f"脚本约束生效 | 脚本: {path.name} | 参数: {matched_flag} | {format(current_value, 'g')} -> {constrained_value}")
            continue

        if rule_type == "value_arg":
            matched_flag = None
            current_value = None
            for flag_key in rule_meta.get("flags", ()):
                match = re.search(rf"{re.escape(flag_key)}\s+([^\s\\]+)", content)
                if match:
                    matched_flag = flag_key
                    current_value = match.group(1)
                    break
            if not matched_flag or current_value is None:
                continue
            constrained_value = _pick_allowed_scalar_value(current_value, allowed_values)
            if not constrained_value or constrained_value == str(current_value):
                continue
            found, changed = _replace_script_value_flag_if_present(path, matched_flag, constrained_value)
            if found and changed:
                log_info(f"脚本约束生效 | 脚本: {path.name} | 参数: {matched_flag} | {current_value} -> {constrained_value}")
            continue

        if rule_type == "bool_flag":
            allowed_bools = _normalize_allowed_bool_values(allowed_values)
            if not allowed_bools:
                continue
            for flag_key in rule_meta.get("flags", ()):
                present = re.search(rf"(^|\s){re.escape(flag_key)}(\s|\\|$)", content, flags=re.MULTILINE)
                if not present:
                    continue
                if True in allowed_bools:
                    break
                if False in allowed_bools:
                    found, changed = _remove_script_flag_if_present(path, flag_key)
                    if found and changed:
                        log_info(f"脚本约束生效 | 脚本: {path.name} | 参数: {flag_key} | 已移除")
                    break
            continue

        if rule_type == "env_int":
            matched_env = None
            current_value = 0
            for env_name in rule_meta.get("envs", ()):
                match = re.search(rf"export\s+{re.escape(env_name)}=(.+)", content)
                if match:
                    matched_env = env_name
                    current_value = _parse_optional_positive_int(match.group(1).strip().strip('"').strip("'"))
                    break
            if not matched_env or current_value <= 0:
                continue
            constrained_value = _pick_allowed_parallel_size(current_value, allowed_values)
            if constrained_value <= 0 or constrained_value == current_value:
                continue
            found, changed = _replace_script_env_if_present(path, matched_env, constrained_value)
            if found and changed:
                log_info(f"脚本约束生效 | 脚本: {path.name} | 环境变量: {matched_env} | {current_value} -> {constrained_value}")
            continue

        if rule_type == "env_value":
            matched_env = None
            current_value = None
            for env_name in rule_meta.get("envs", ()):
                match = re.search(rf"export\s+{re.escape(env_name)}=(.+)", content)
                if match:
                    matched_env = env_name
                    current_value = match.group(1).strip().strip('"').strip("'")
                    break
            if not matched_env or current_value is None:
                continue
            constrained_value = _pick_allowed_scalar_value(current_value, allowed_values)
            if not constrained_value or constrained_value == str(current_value):
                continue
            found, changed = _replace_script_env_if_present(path, matched_env, constrained_value)
            if found and changed:
                log_info(f"脚本约束生效 | 脚本: {path.name} | 环境变量: {matched_env} | {current_value} -> {constrained_value}")
    return ensure_gqa_tensor_parallel_compatible(script_path)


def configure_project_tmp_env():
    return runtime_helpers.configure_project_tmp_env(PROJECT_TMP_ROOT)


def run_shell(cmd, check=True, capture=True, timeout=None, timeout_label=None):
    """执行shell命令（强制使用bash并开启set -e与pipefail）"""
    shell_cmd = f"set -e -o pipefail\n{cmd}"
    if capture:
        process = subprocess.Popen(
            ["bash", "-lc", shell_cmd],
            cwd=str(LMSV_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            label = timeout_label or "命令执行"
            log_error(f"{label}超时（>{timeout}s），已终止当前进程，按失败处理")
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                stdout, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = process.communicate()
        result = subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)
        result.timed_out = timed_out
        if check and result.returncode != 0:
            log_error(f"命令执行失败: {cmd}")
            log_error(f"错误输出: {result.stderr}")
            return None
        return result
    else:
        return subprocess.run(["bash", "-lc", shell_cmd], check=check, cwd=str(LMSV_ROOT))


def run_shell_to_file(cmd, log_file, check=True, timeout=None, timeout_label=None):
    result = runtime_helpers.run_shell_to_file(
        cmd,
        log_file,
        LMSV_ROOT,
        log_error,
        check=False,
        timeout=timeout,
        timeout_label=timeout_label,
    )
    if check and result and result.returncode != 0:
        log_error(f"命令执行失败: {cmd}")
        log_error(f"执行日志已保存: {log_file}")
        return None
    return result


def create_backup_dir(stage, iter_num, module, type_):
    """创建备份目录"""
    if stage in ("save", "load"):
        backup_dir = f"{Config.TMP_ROOT_DIR}/{stage}/{iter_num}/{module}/{type_}"
    else:
        backup_dir = f"{Config.TMP_ROOT_DIR}/default/{iter_num}/{module}/{type_}"
    os.makedirs(backup_dir, exist_ok=True)
    return backup_dir


def backup_artifact_to_output(src_path, run_persist_dir, iter_num, category, dst_name=None, missing_log_level="warn"):
    return runtime_helpers.backup_artifact_to_output(
        src_path,
        run_persist_dir,
        iter_num,
        category,
        LMSV_ROOT,
        log_backup,
        log_warn,
        dst_name=dst_name,
        missing_log_level=missing_log_level,
    )


def backup_weight_on_pta_msa_failure(weight_path, run_persist_dir, iter_num, reason):
    log_warn(f"[iter{iter_num}] 检测到PTA/MSA异常，尝试备份权重: {reason}")
    return backup_artifact_to_output(weight_path, run_persist_dir, iter_num, "weights/pta-save")


def backup_weight_on_precision_issue(weight_path, run_persist_dir, iter_num, reason):
    log_warn(f"[iter{iter_num}] 检测到精度问题，尝试备份权重: {reason}")
    return backup_artifact_to_output(weight_path, run_persist_dir, iter_num, "weights/pta-save")


def backup_runtime_log_to_output(log_path, run_persist_dir, iter_num, dst_name=None):
    return backup_artifact_to_output(log_path, run_persist_dir, iter_num, "runtime_logs", dst_name)


def is_weight_artifact_missing(weight_path):
    """检查权重产物是否不存在或为空。"""
    path = Path(weight_path)
    if not path.is_absolute():
        path = LMSV_ROOT / path

    if not path.exists():
        return True
    if path.is_file():
        return path.stat().st_size <= 0
    if path.is_dir():
        try:
            next(path.iterdir())
            return False
        except StopIteration:
            return True
    return False


def generate_persistent_log_name(log_type, iter_num):
    """生成持久化日志名"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{log_type.lower()}_{Config.MODEL_NAME}_{timestamp}_iter{iter_num}.log"


def build_conda_activate_block(env_name, load_ascend=False):
    return runtime_helpers.build_conda_activate_block(env_name, load_ascend=load_ascend)

def check_param_exists(script_path, param):
    """检查脚本中是否存在指定参数"""
    result = run_shell(
        f'if grep -qE -- "{param}[[:space:]]+" "{script_path}" 2>/dev/null; then echo 0; else echo 1; fi'
    )
    return result.stdout.strip() == "0"


def replace_save_to_load(script_path, load_path):
    """将脚本中的--save替换为--load"""
    cmd = f'sed -i -- "s|--save[[:space:]]\\+[^[:space:]]\\+|--load {load_path}|g" "{script_path}"'
    result = run_shell(cmd)
    if result and result.returncode == 0:
        log_info(f"已将脚本中--save参数替换为--load {load_path}")
        return True
    log_error("替换--save为--load失败")
    return False


def modify_script_param(script_path, old_pattern, new_param):
    """修改脚本参数"""
    if not os.path.exists(script_path):
        log_error(f"脚本不存在: {script_path}")
        return False
    
    param_key = new_param.split()[0]
    param_value = new_param.split()[1] if len(new_param.split()) > 1 else ""
    
    # 检查参数是否存在
    param_exists = check_param_exists(script_path, param_key)
    
    if param_exists:
        # 参数存在，执行修改
        cmd = f'sed -i -- "s|{old_pattern}|{new_param}|g" "{script_path}"'
        result = run_shell(cmd)
        if result and result.returncode == 0:
            return True
        log_error(f"参数修改失败 | 脚本: {script_path}")
        return False
    else:
        # 参数不存在，插入
        log_info(f"参数插入警告 | 脚本: {os.path.basename(script_path)} | 未找到参数{param_key}，将自动插入")
        if check_param_exists(script_path, "--train-iters"):
            cmd = f'sed -i -- "/--train-iters/a\\\\    {new_param} \\\\" "{script_path}"'
        else:
            cmd = f'echo "{new_param}" >> "{script_path}"'
        result = run_shell(cmd)
        if result and result.returncode == 0:
            log_info(f"参数插入成功 | 脚本: {os.path.basename(script_path)} | 插入: {new_param}")
            return True
        log_error(f"参数插入失败 | 脚本: {script_path}")
        return False


def delete_script_param_line(script_path, param_key):
    """删除脚本中的参数整行"""
    if not os.path.exists(script_path):
        log_error(f"脚本不存在: {script_path}")
        return False
    
    # 检查是否存在（grep 输出计数，永远 exit 0）
    result = run_shell(
        f'grep -cE -- "^[[:space:]]*{param_key}([[:space:]]+|$)" '
        f'"{script_path}" 2>/dev/null || true'
    )
    count = int(result.stdout.strip() or 0)

    if count == 0:
        log_info(
            f"参数行删除警告 | 脚本: {os.path.basename(script_path)} | "
            f"未找到独立的{param_key}参数行，无需删除"
        )
        return True
    
    # 删除
    cmd = f'sed -i -E "/^[[:space:]]*{param_key}([[:space:]]+|$).*/d" "{script_path}"'
    result = run_shell(cmd)
    if result and result.returncode == 0:
        log_info(f"参数行删除成功 | 脚本: {os.path.basename(script_path)} | 已删除{count}行")
        return True

    log_error(f"参数行删除失败 | 脚本: {script_path}")
    return False


def ensure_script_flag(script_path, flag_key):
    """确保脚本中存在某个无值开关参数（如 --foo-bar）。"""
    if not os.path.exists(script_path):
        log_error(f"脚本不存在: {script_path}")
        return False

    if check_param_exists(script_path, flag_key):
        return True

    if check_param_exists(script_path, "--train-iters"):
        cmd = f'sed -i -- "/--train-iters/a\\    {flag_key} \\\\" "{script_path}"'
    else:
        cmd = f'echo "{flag_key}" >> "{script_path}"'

    result = run_shell(cmd)
    if result and result.returncode == 0:
        log_info(f"参数插入成功 | 脚本: {os.path.basename(script_path)} | 插入: {flag_key}")
        return True

    log_error(f"参数插入失败 | 脚本: {script_path} | 参数: {flag_key}")
    return False


def _extract_int_script_param(script_path, param_key, default):
    """从脚本中提取整型参数值，提取失败时返回默认值。"""
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()
        match = re.search(rf"{re.escape(param_key)}\s+([0-9]+)", content)
        if match:
            return int(match.group(1))
    except Exception:
        pass
    return int(default)


def _get_first_k_dense_replace_limit(script_path):
    """根据 num_layers 和 pipeline parallel 计算 first_k_dense_replace 的安全上限。"""
    num_layers = max(1, _extract_int_script_param(script_path, "--num-layers", 8))
    pipeline_parallel_size = max(1, _extract_int_script_param(script_path, "--pipeline-model-parallel-size", 1))
    layers_per_stage = max(1, (num_layers + pipeline_parallel_size - 1) // pipeline_parallel_size)
    return max(0, layers_per_stage - 1)


def ensure_global_batch_divisible(script_path, world_size=8):
    """确保 global-batch-size 能被 micro-batch-size * dp 整除。"""
    if not os.path.exists(script_path):
        return False

    micro_bs = max(1, _extract_int_script_param(script_path, "--micro-batch-size", 1))
    global_bs = max(1, _extract_int_script_param(script_path, "--global-batch-size", 1))

    try:
        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return False

    tp_match = re.search(r"--tensor-model-parallel-size\s+([0-9]+)", content)
    pp_match = re.search(r"--pipeline-model-parallel-size\s+([0-9]+)", content)
    cp_match = re.search(r"--context-parallel-size\s+([0-9]+)", content)

    if not (tp_match and pp_match and cp_match):
        log_info(
            f"批大小修正跳过 | 脚本: {os.path.basename(script_path)} | "
            "并行参数未显式落盘，保持原始global-batch-size"
        )
        return True

    tp = max(1, int(tp_match.group(1)))
    pp = max(1, int(pp_match.group(1)))
    cp = max(1, int(cp_match.group(1)))

    denom = max(1, tp * pp * cp)
    dp = max(1, int(world_size) // denom)
    divisor = max(1, micro_bs * dp)

    if global_bs % divisor == 0:
        log_info(
            f"批大小校验通过 | 脚本: {os.path.basename(script_path)} | "
            f"global={global_bs}, micro={micro_bs}, dp={dp}"
        )
        return True

    adjusted_global = ((global_bs + divisor - 1) // divisor) * divisor
    modify_script_param(
        script_path,
        r"--global-batch-size[[:space:]]\+[0-9]\+",
        f"--global-batch-size {adjusted_global}",
    )
    log_info(
        f"批大小自适应修正 | 脚本: {os.path.basename(script_path)} | "
        f"global {global_bs} -> {adjusted_global}, micro={micro_bs}, dp={dp}"
    )
    return True


def apply_deepseekv3_unified_low_memory_profile(script_path):
    """Apply unified low-memory controls for DeepSeekV3 on PTA side."""
    ok = True

    ok = delete_script_param_line(script_path, "--group-query-attention") and ok
    ok = ensure_script_flag(script_path, "--moe-router-enable-expert-bias") and ok
    ok = ensure_script_flag(script_path, "--no-check-for-nan-in-loss-and-grad") and ok
    ok = modify_script_param(
        script_path,
        r"--moe-router-score-function[[:space:]]\+[^\ ]\+",
        "--moe-router-score-function sigmoid",
    ) and ok
    ok = modify_script_param(
        script_path,
        r"--micro-batch-size[[:space:]]\+[0-9]\+",
        "--micro-batch-size 1",
    ) and ok
    ok = modify_script_param(
        script_path,
        r"--global-batch-size[[:space:]]\+[0-9]\+",
        "--global-batch-size 8",
    ) and ok
    ok = modify_script_param(
        script_path,
        r"--num-layers[[:space:]]\+[0-9]\+",
        "--num-layers 8",
    ) and ok
    ok = modify_script_param(
        script_path,
        r"--hidden-size[[:space:]]\+[0-9]\+",
        "--hidden-size 1024",
    ) and ok
    ok = modify_script_param(
        script_path,
        r"--ffn-hidden-size[[:space:]]\+[0-9]\+",
        "--ffn-hidden-size 2048",
    ) and ok
    ok = modify_script_param(
        script_path,
        r"--num-attention-heads[[:space:]]\+[0-9]\+",
        "--num-attention-heads 16",
    ) and ok
    ok = modify_script_param(
        script_path,
        r"--q-lora-rank[[:space:]]\+[0-9]\+",
        "--q-lora-rank 192",
    ) and ok
    ok = modify_script_param(
        script_path,
        r"--kv-lora-rank[[:space:]]\+[0-9]\+",
        "--kv-lora-rank 64",
    ) and ok
    ok = modify_script_param(
        script_path,
        r"--moe-intermediate-size[[:space:]]\+[0-9]\+",
        "--moe-intermediate-size 768",
    ) and ok
    ok = modify_script_param(
        script_path,
        r"--num-experts[[:space:]]\+[0-9]\+",
        "--num-experts 16",
    ) and ok
    ok = modify_script_param(
        script_path,
        r"--n-shared-experts[[:space:]]\+[0-9]\+",
        "--n-shared-experts 1",
    ) and ok
    ok = modify_script_param(
        script_path,
        r"--moe-router-topk[[:space:]]\+[0-9]\+",
        "--moe-router-topk 2",
    ) and ok
    ok = modify_script_param(
        script_path,
        r"--moe-aux-loss-coeff[[:space:]]\+[0-9]*\.?[0-9]+",
        "--moe-aux-loss-coeff 0.01",
    ) and ok
    ok = modify_script_param(
        script_path,
        r"--seq-length[[:space:]]\+[0-9]\+",
        "--seq-length 1024",
    ) and ok
    ok = modify_script_param(
        script_path,
        r"--max-position-embeddings[[:space:]]\+[0-9]\+",
        "--max-position-embeddings 1024",
    ) and ok
    ok = modify_script_param(
        script_path,
        r"--moe-layer-freq[[:space:]]\+[0-9]\+",
        "--moe-layer-freq 1",
    ) and ok
    first_k_dense_replace = _get_first_k_dense_replace_limit(script_path)
    ok = modify_script_param(
        script_path,
        r"--first-k-dense-replace[[:space:]]\+[0-9]\+",
        f"--first-k-dense-replace {first_k_dense_replace}",
    ) and ok
    ok = ensure_global_batch_divisible(script_path) and ok
    ok = delete_script_param_line(script_path, "--num-layers-per-virtual-pipeline-stage") and ok

    if ok:
        log_info(
            "DeepSeekV3统一减配生效 | "
            "--num-layers=8 | --hidden-size=1024 | --ffn-hidden-size=2048 | "
            "--num-attention-heads=16 | --q-lora-rank=192 | --kv-lora-rank=64 | "
            "--moe-intermediate-size=768 | --num-experts=16 | --n-shared-experts=1 | --moe-router-topk=2 | "
            "--moe-aux-loss-coeff=0.01 | "
            "--seq-length=1024 | --max-position-embeddings=1024 | --global-batch-size=8 | "
            f"--moe-layer-freq=1 | --first-k-dense-replace={first_k_dense_replace} | --micro-batch-size=1"
        )
    else:
        log_error("DeepSeekV3统一减配失败")

    return ok


def check_script_valid(script_path):
    """检查脚本有效性"""
    if not os.path.exists(script_path):
        log_error(f"脚本不存在: {script_path}")
        return False
    if os.path.getsize(script_path) == 0:
        log_error(f"脚本为空: {script_path}")
        return False
    log_info(f"脚本有效性检查通过: {os.path.basename(script_path)}")
    return True

def _script_uses_position_embedding(script_path, embedding_type):
    path = Path(script_path)
    if not path.exists():
        return False
    pattern = rf"--position-embedding-type[[:space:]]+{re.escape(str(embedding_type))}([[:space:]]|$)"
    result = run_shell(
        f'grep -cE -- "{pattern}" "{path}" 2>/dev/null || true'
    )
    if not result:
        return False
    try:
        return int(result.stdout.strip() or "0") > 0
    except ValueError:
        return False


def sanitize_alibi_script(script_path):
    """Remove ALiBi-incompatible flags from the final generated training script."""
    if not _script_uses_position_embedding(script_path, "alibi"):
        return True

    log_info(f"检测到 ALiBi 脚本，清理不兼容参数: {os.path.basename(script_path)}")
    flags_to_remove = (
        "--context-parallel-size",
        "--context-parallel-algo",
        "--use-rotary-position-embeddings",
        "--rotary-base",
        "--use-flash-attn",
    )
    for flag in flags_to_remove:
        if not delete_script_param_line(script_path, flag):
            return False
    return True


def sanitize_pangu_script(script_path):
    """Remove known Pangu-incompatible RoPE/Gated-MLP flags from the generated training script."""
    path = Path(script_path)
    if not path.exists():
        return False

    content = path.read_text(encoding="utf-8")
    if "--use-top-query-embedding" not in content and "pangu" not in path.name.lower():
        return True

    log_info(f"检测到 Pangu 脚本，清理不兼容参数: {os.path.basename(script_path)}")
    flags_to_remove = (
        "--use-top-query-embedding",
        "--use-rotary-position-embeddings",
        "--rotary-base",
        "--swiglu",
        "--disable-bias-linear",
        "--untie-embeddings-and-output-weights",
    )
    for flag in flags_to_remove:
        if not delete_script_param_line(script_path, flag):
            return False
    return True


def sanitize_swiglu_fusion_script(script_path):
    """Disable unsupported activation fusion combinations in generated training scripts."""
    path = Path(script_path)
    if not path.exists():
        return False

    content = path.read_text(encoding="utf-8")
    has_moe = (
        ("--num-experts" in content)
        or ("--num-moe-experts" in content)
        or ("--expert-num" in content)
        or ("--moe-" in content)
    )
    has_swiglu = "--swiglu" in content

    if not has_moe and not has_swiglu:
        return True

    if has_moe:
        log_info(f"检测到 MoE 脚本，清理 activation fusion 参数: {os.path.basename(script_path)}")
        if not ensure_script_flag(script_path, "--no-bias-gelu-fusion"):
            return False
        if not ensure_script_flag(script_path, "--no-bias-swiglu-fusion"):
            return False
        if not delete_script_param_line(script_path, "--add-bias-linear"):
            return False
        if not ensure_script_flag(script_path, "--disable-bias-linear"):
            return False

    if has_swiglu:
        log_info(f"检测到 SwiGLU 脚本，清理不兼容 fusion 参数: {os.path.basename(script_path)}")
        if not delete_script_param_line(script_path, "--use-fused-swiglu"):
            return False
        if not ensure_script_flag(script_path, "--no-bias-swiglu-fusion"):
            return False
    return True


def sanitize_moe_expert_bias_aux_loss(script_path, fallback_value=0.01):
    """Ensure expert-bias MoE scripts keep a positive aux loss coefficient."""
    path = Path(script_path)
    if not path.exists():
        return False

    content = path.read_text(encoding="utf-8")
    if "--moe-router-enable-expert-bias" not in content:
        return True

    pattern = r"(--moe-aux-loss-coeff\s+)0+(?:\.0+)?(?=([ \t\\]|$))"
    updated, count = re.subn(pattern, rf"\g<1>{fallback_value}", content, count=1)
    if count <= 0:
        return True
    if updated == content:
        return True

    path.write_text(updated, encoding="utf-8")
    log_info(
        f"MoE专家偏置兼容化 | 脚本: {os.path.basename(script_path)} | "
        f"--moe-aux-loss-coeff -> {fallback_value}"
    )
    return True


def align_bias_linear_flags(reference_script_path, target_script_path):
    """Mirror bias-linear flags from a reference script to a converted target script."""
    reference = Path(reference_script_path)
    target = Path(target_script_path)
    if not reference.exists() or not target.exists():
        log_error(
            "bias-linear参数对齐失败，脚本不存在 | "
            f"reference={reference_script_path} | target={target_script_path}"
        )
        return False

    reference_content = reference.read_text(encoding="utf-8", errors="ignore")
    disable_in_reference = "--disable-bias-linear" in reference_content
    add_in_reference = "--add-bias-linear" in reference_content

    ok = True
    if disable_in_reference:
        ok = delete_script_param_line(target_script_path, "--add-bias-linear") and ok
        ok = ensure_script_flag(target_script_path, "--disable-bias-linear") and ok
    else:
        ok = delete_script_param_line(target_script_path, "--disable-bias-linear") and ok

    if add_in_reference:
        ok = delete_script_param_line(target_script_path, "--disable-bias-linear") and ok
        ok = ensure_script_flag(target_script_path, "--add-bias-linear") and ok
    elif not disable_in_reference:
        ok = delete_script_param_line(target_script_path, "--add-bias-linear") and ok

    if ok:
        log_info(
            "bias-linear参数已按PTA脚本对齐到MSA脚本 | "
            f"disable={disable_in_reference} | add={add_in_reference}"
        )
    return ok


def sanitize_rotary_base_script(script_path):
    """Normalize integer-like rotary-base values to plain integers for argparse compatibility."""
    path = Path(script_path)
    if not path.exists():
        return False

    content = path.read_text(encoding="utf-8")
    pattern = r"(--rotary-base\s+)([0-9]+)\.0(?=([ \t\\]|$))"
    updated, count = re.subn(pattern, r"\g<1>\2", content)
    if count <= 0:
        return True

    path.write_text(updated, encoding="utf-8")
    log_info(f"检测到 rotary-base 浮点整数字面量，已归一化: {os.path.basename(script_path)}")
    return True


def sanitize_task1_mutation_runtime_flags(script_path):
    """Task1 mutation scripts should not enable unstable flash-attn/overlap runtime flags."""
    path = Path(script_path)
    if not path.exists():
        return False

    flags_to_remove = (
        "--use-flash-attn",
        "--overlap-grad-reduce",
        "--overlap-param-gather",
    )
    log_info(f"执行 Task1 变异脚本运行时参数清理: {os.path.basename(script_path)}")
    for flag in flags_to_remove:
        if not delete_script_param_line(script_path, flag):
            return False
    return True


def ensure_external_pretrain_entry(script_path):
    """
    将训练脚本中的 pretrain_gpt.py 改写为可由环境变量控制的入口。
    不改模板源码，只改每轮动态生成脚本。
    """
    if not os.path.exists(script_path):
        log_error(f"脚本不存在: {script_path}")
        return False

    with open(script_path, "r", encoding="utf-8") as handle:
        content = handle.read()

    replacement = "${LMSV_PRETRAIN_GPT}"
    default_line = ': "${LMSV_PRETRAIN_GPT:=pretrain_gpt.py}"'
    if replacement in content:
        if default_line not in content:
            lines = content.splitlines()
            insert_at = 1 if lines and lines[0].startswith("#!") else 0
            lines.insert(insert_at, default_line)
            content = "\n".join(lines) + ("\n" if content.endswith("\n") else "")
            with open(script_path, "w", encoding="utf-8") as handle:
                handle.write(content)
        return True

    new_content, replaced = re.subn(r"\bpretrain_gpt\.py\b", replacement, content)
    if replaced == 0:
        log_error(f"脚本中未找到 pretrain_gpt.py 入口: {script_path}")
        return False

    lines = new_content.splitlines()
    insert_at = 1 if lines and lines[0].startswith("#!") else 0
    lines.insert(insert_at, default_line)
    new_content = "\n".join(lines) + ("\n" if new_content.endswith("\n") else "")

    with open(script_path, "w", encoding="utf-8") as handle:
        handle.write(new_content)
    log_info(f"已切换为外部 pretrain 入口: {os.path.basename(script_path)}")
    return True


def _to_abs_path(path_str):
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = LMSV_ROOT / path
    return str(path.resolve())


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def get_mutation_artifact_paths(iter_num):
    mutation_dir = LMSV_ROOT / "res" / Config.MODEL_NAME
    succ_path = mutation_dir / f"mutating-{iter_num}.json"
    err_path = mutation_dir / f"mutating-{iter_num}-err.json"
    return succ_path, err_path


def _is_loadable_mutation_json(json_path):
    path = Path(json_path)
    if not path.exists():
        return False, "文件不存在"
    if path.stat().st_size <= 0:
        return False, "文件为空"

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        return False, f"JSON解析失败: {exc}"

    if not isinstance(payload, dict):
        return False, "JSON根节点不是dict"

    has_numeric_config = any(str(key).isdigit() for key in payload)
    if not has_numeric_config:
        return False, "JSON中缺少可加载的变异节点配置"

    return True, ""


def validate_mutation_artifacts(iter_num, run_persist_dir, mutate_exec_log):
    succ_path, err_path = get_mutation_artifact_paths(iter_num)
    succ_ok, succ_reason = _is_loadable_mutation_json(succ_path)

    if succ_ok:
        log_info(f"变异产物检查通过: {succ_path}")
        backup_artifact_to_output(succ_path, run_persist_dir, iter_num, "mutation_inputs")
        if err_path.exists() and err_path.stat().st_size > 0:
            backup_artifact_to_output(err_path, run_persist_dir, iter_num, "mutation_inputs")
        return True

    if err_path.exists() and err_path.stat().st_size > 0:
        log_error(
            f"第{iter_num}轮 mutate 未生成可加载产物: {succ_path} | "
            f"检测到失败记录 {err_path}"
        )
        backup_artifact_to_output(err_path, run_persist_dir, iter_num, "mutation_inputs")
    else:
        log_error(f"第{iter_num}轮 mutate 未生成变异产物: {succ_path}")

    if succ_reason:
        log_error(f"第{iter_num}轮 mutate 产物不可用原因: {succ_reason}")
    log_error(f"请检查变异执行日志: {mutate_exec_log}")
    return False


def should_treat_mutate_as_success(run_ok, iter_num):
    """mutate 即使退出码非0，只要当前轮产物已生成且可加载，也按成功处理。"""
    if run_ok:
        return True

    succ_path, _ = get_mutation_artifact_paths(iter_num)
    succ_ok, succ_reason = _is_loadable_mutation_json(succ_path)
    if succ_ok:
        log_warn(
            "mutate 进程退出码非0，但检测到当前轮变异产物已成功生成，"
            f"按成功处理 | iter={iter_num} | json={succ_path}"
        )
        return True

    if succ_reason:
        log_warn(
            "mutate 进程退出码非0，且当前轮变异产物仍不可加载，"
            f"iter={iter_num} | 原因: {succ_reason}"
        )
    return False


def extract_numeric_param_from_script(script_path, *keys):
    if not os.path.exists(script_path):
        return None
    content = Path(script_path).read_text(encoding="utf-8")
    variables = {}
    for line in content.splitlines():
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([0-9]+)\s*$", line.strip())
        if match:
            variables[match.group(1)] = int(match.group(2))
    for key in keys:
        pattern = re.compile(rf"(^|[ \t]){re.escape(key)}[ \t]+([^ \t\\]+)")
        matches = pattern.findall(content)
        if not matches:
            continue
        raw_value = matches[-1][1]
        if raw_value.isdigit():
            return int(raw_value)
        var_match = re.fullmatch(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?", raw_value)
        if var_match:
            resolved = variables.get(var_match.group(1))
            if resolved is not None:
                return int(resolved)
    return None


def _repo_rel_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        try:
            return path.relative_to(LMSV_ROOT).as_posix()
        except ValueError:
            return str(path)
    return path.as_posix()


def _path_is_relative_to(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def _cluster_dataset_upload_items(data_path):
    source = Path(data_path).expanduser()
    original_text = str(source)
    if not source.is_absolute():
        source = (LMSV_ROOT / source).resolve()
    else:
        source = source.resolve()

    if _path_is_relative_to(source, LMSV_ROOT):
        remote_prefix = source.relative_to(LMSV_ROOT.resolve()).as_posix()
    else:
        remote_prefix = f"dataset/_cluster_bundle/{source.name}"

    if source.exists():
        return [(source, remote_prefix)], remote_prefix, original_text, source

    parent = source.parent
    prefix = source.name
    parts = sorted(item for item in parent.glob(f"{prefix}*") if item.is_file())
    if not parts:
        raise FileNotFoundError(f"多机 dataset 下发失败，未找到数据集或前缀文件: {data_path}")

    upload_items = []
    for item in parts:
        suffix = item.name[len(prefix):] if item.name.startswith(prefix) else f"_{item.name}"
        upload_items.append((item, f"{remote_prefix}{suffix}"))
    return upload_items, remote_prefix, original_text, source


def _replace_dataset_path_in_script(script_path, original_dataset_path, resolved_dataset_path, remote_dataset_path):
    path = Path(script_path)
    content = path.read_text(encoding="utf-8")
    replacements = {
        str(original_dataset_path),
        str(resolved_dataset_path),
    }
    if _path_is_relative_to(resolved_dataset_path, LMSV_ROOT):
        repo_rel = resolved_dataset_path.relative_to(LMSV_ROOT.resolve()).as_posix()
        replacements.update(
            {
                repo_rel,
                f"./{repo_rel}",
                f"{LMSV_ROOT.resolve().as_posix()}/{repo_rel}",
            }
        )
    for old in sorted((item for item in replacements if item), key=len, reverse=True):
        content = content.replace(old, remote_dataset_path)
    path.write_text(content, encoding="utf-8")


def _build_cluster_session_id():
    output_name = Path(Config.MY_PERSIST_ROOT).name or f"task1-{os.getpid()}"
    return f"{output_name}-task1"


def _wait_remote_job(cluster, handle, timeout_seconds):
    deadline = time.time() + max(30, int(timeout_seconds))
    last_status = ""
    while time.time() < deadline:
        state = cluster.job_state(handle)
        status = str(state.get("status", ""))
        if status != last_status:
            log_info(
                f"[多机][{handle.stage_name}] 远端状态更新 | "
                f"rank={handle.node.node_rank} | status={status}"
            )
            last_status = status
        if status in {"success", "failed", "timeout", "cancelled"}:
            return status == "success", state
        time.sleep(2)
    cluster.cancel_job(handle)
    return False, {"status": "timeout", "error": "remote wait timeout"}


def _run_cluster_stage(
    *,
    cluster,
    session_id,
    stage_name,
    runtime_log_dir,
    local_runner,
    payload_builder,
    upload_builder=None,
    collect_builder=None,
    timeout_seconds=3600,
):
    handles = []
    stop_events = []
    log_threads = []
    remote_success = True
    remote_states = {}
    local_ok = False

    try:
        for node in cluster.config.slaves:
            node_workers = cluster.slave_worker_count(node)
            if upload_builder is not None:
                upload_items = upload_builder(node, node_workers) or []
                if upload_items:
                    cluster.upload_paths(node, session_id, upload_items)
            payload = payload_builder(node, node_workers)
            handle = cluster.start_job(
                node,
                session_id,
                payload.pop("job_type"),
                payload,
                stage_name=stage_name,
            )
            handles.append(handle)
            stop_event = threading.Event()
            stop_events.append(stop_event)
            log_threads.append(
                cluster.stream_job_log(
                    handle,
                    Path(runtime_log_dir) / f"{stage_name}_node{node.node_rank}.log",
                    stop_event,
                )
            )

        local_ok = bool(local_runner())
        if not local_ok:
            for handle in handles:
                cluster.cancel_job(handle)

        for handle in handles:
            ok, state = _wait_remote_job(cluster, handle, timeout_seconds)
            state = dict(state)
            state["log_text"] = cluster.job_log_text(handle)
            state["log_tail"] = cluster.job_log_tail(handle)
            remote_success = remote_success and ok
            remote_states[handle.node.node_rank] = state
    finally:
        for stop_event in stop_events:
            stop_event.set()
        for thread in log_threads:
            thread.join(timeout=5)

    if collect_builder is not None:
        for handle in handles:
            items, target_dir = collect_builder(handle.node, cluster.slave_worker_count(handle.node))
            if items:
                cluster.download_items(handle.node, session_id, items, target_dir)

    return local_ok, remote_success, remote_states


def _replace_or_insert_shell_var(script_path, key, value):
    path = Path(script_path)
    if not path.exists():
        return False
    lines = path.read_text(encoding="utf-8").splitlines()
    pattern = re.compile(rf"^\s*{re.escape(key)}=.*$")
    replacement = f"{key}={value}"
    replaced = False
    for index, line in enumerate(lines):
        if pattern.match(line):
            lines[index] = replacement
            replaced = True
            break
    if not replaced:
        insert_at = 1 if lines and lines[0].startswith("#!") else 0
        lines.insert(insert_at, replacement)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def _ensure_script_line(script_path, line_text, *, after_key=None):
    path = Path(script_path)
    if not path.exists():
        return False
    content = path.read_text(encoding="utf-8")
    if line_text in content:
        return True
    lines = content.splitlines()
    insert_at = len(lines)
    if after_key:
        for index, line in enumerate(lines):
            if line.startswith(f"{after_key}="):
                insert_at = index + 1
    lines.insert(insert_at, line_text)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def _ensure_script_block(script_path, block_lines, *, after_key=None):
    path = Path(script_path)
    if not path.exists():
        return False
    content = path.read_text(encoding="utf-8")
    block_text = "\n".join(block_lines)
    if block_text in content:
        return True
    lines = content.splitlines()
    insert_at = len(lines)
    if after_key:
        for index, line in enumerate(lines):
            if line.startswith(f"{after_key}="):
                insert_at = index + 1
    for offset, line_text in enumerate(block_lines):
        lines.insert(insert_at + offset, line_text)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def _pta_socket_ifname_block():
    return [
        '# LMSV clear stale distributed env for torchrun multinode PTA',
        'unset RANK_TABLE_FILE',
        'unset RANK_SIZE',
        'unset RANK_ID',
        'unset LOCAL_RANK',
        'unset RANK',
        'unset GROUP_RANK',
        'unset ROLE_RANK',
        'unset ROLE_WORLD_SIZE',
        'unset LOCAL_WORLD_SIZE',
        'unset TORCHELASTIC_RUN_ID',
        'unset TORCHELASTIC_RESTART_COUNT',
        'unset TORCHELASTIC_MAX_RESTARTS',
        'unset TORCHELASTIC_ERROR_FILE',
        '# LMSV auto-detect socket interface for multinode PTA',
        'if [[ -z "${GLOO_SOCKET_IFNAME:-}" || -z "${TP_SOCKET_IFNAME:-}" || -z "${HCCL_SOCKET_IFNAME:-}" ]]; then',
        '    LMSV_SOCKET_IFNAME_TARGET="$(getent hosts "$MASTER_ADDR" 2>/dev/null | awk \'NR==1 {print $1}\')"',
        '    LMSV_SOCKET_IFNAME_TARGET="${LMSV_SOCKET_IFNAME_TARGET:-$MASTER_ADDR}"',
        '    LMSV_SOCKET_IFNAME="$(ip route get "$LMSV_SOCKET_IFNAME_TARGET" 2>/dev/null | awk \'/ dev / {for (i=1;i<=NF;i++) if ($i=="dev") {print $(i+1); exit}}\')"',
        '    if [[ -z "${LMSV_SOCKET_IFNAME:-}" ]]; then',
        '        LMSV_SOCKET_IFNAME="$(ip route show default 2>/dev/null | awk \'/default/ {for (i=1;i<=NF;i++) if ($i=="dev") {print $(i+1); exit}}\')"',
        '    fi',
        '    if [[ -n "${LMSV_SOCKET_IFNAME:-}" ]]; then',
        '        if [[ -z "${GLOO_SOCKET_IFNAME:-}" ]]; then export GLOO_SOCKET_IFNAME="$LMSV_SOCKET_IFNAME"; fi',
        '        if [[ -z "${TP_SOCKET_IFNAME:-}" ]]; then export TP_SOCKET_IFNAME="$LMSV_SOCKET_IFNAME"; fi',
        '        if [[ -z "${HCCL_SOCKET_IFNAME:-}" ]]; then export HCCL_SOCKET_IFNAME="$LMSV_SOCKET_IFNAME"; fi',
        '        echo "Auto-detected PTA socket interfaces: GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME TP_SOCKET_IFNAME=$TP_SOCKET_IFNAME HCCL_SOCKET_IFNAME=$HCCL_SOCKET_IFNAME"',
        '    fi',
        'fi',
    ]


def apply_multinode_script_settings(
    script_path,
    *,
    local_workers,
    total_workers,
    nnodes,
    node_rank,
    master_addr,
    master_port,
    enable_pta_env=False,
):
    updates = {
        "NPUS_PER_NODE": int(local_workers),
        "MASTER_ADDR": shlex.quote(str(master_addr)),
        "MASTER_PORT": int(master_port),
        "NNODES": int(nnodes),
        "NODE_RANK": int(node_rank),
        "WORLD_SIZE": int(total_workers),
    }
    for key, value in updates.items():
        if not _replace_or_insert_shell_var(script_path, key, value):
            return False
    if enable_pta_env:
        if not _ensure_script_line(script_path, 'export MA_NUM_HOSTS="$NNODES"', after_key="NODE_RANK"):
            return False
        if not _ensure_script_line(script_path, 'export VC_TASK_INDEX="$NODE_RANK"', after_key="NODE_RANK"):
            return False
        if not _ensure_script_line(script_path, 'export MASTER_ADDR="$MASTER_ADDR"', after_key="NODE_RANK"):
            return False
        if int(nnodes) > 1 and not _ensure_script_block(script_path, _pta_socket_ifname_block(), after_key="NODE_RANK"):
            return False
    return True


def rewrite_repo_local_paths_for_remote(path_value):
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        return False
    content = path.read_text(encoding="utf-8")
    repo_prefix = str(LMSV_ROOT)
    if repo_prefix not in content:
        return True
    replacement = "."
    updated = content.replace(repo_prefix, replacement)
    if updated != content:
        path.write_text(updated, encoding="utf-8")
    return True


def prepare_node_specific_script_copy(
    source_script,
    target_dir,
    *,
    local_workers,
    total_workers,
    nnodes,
    node_rank,
    master_addr,
    master_port,
    enable_pta_env=False,
):
    source_path = Path(source_script)
    target_path = Path(target_dir) / f"node_{node_rank}_{source_path.name}"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    ok = apply_multinode_script_settings(
        target_path,
        local_workers=local_workers,
        total_workers=total_workers,
        nnodes=nnodes,
        node_rank=node_rank,
        master_addr=master_addr,
        master_port=master_port,
        enable_pta_env=enable_pta_env,
    )
    if ok:
        ok = rewrite_repo_local_paths_for_remote(target_path)
    return ok, target_path


def build_remote_script_upload_items(
    source_script,
    dest_rel,
    target_dir,
    *,
    local_workers,
    total_workers,
    nnodes,
    node_rank,
    master_addr,
    master_port,
    enable_pta_env=False,
    dataset_path=None,
):
    dataset_items = []
    remote_dataset_path = ""
    original_dataset_path = ""
    resolved_dataset_path = None
    if dataset_path:
        dataset_items, remote_dataset_path, original_dataset_path, resolved_dataset_path = _cluster_dataset_upload_items(dataset_path)
    ok, node_script = prepare_node_specific_script_copy(
        source_script,
        target_dir,
        local_workers=local_workers,
        total_workers=total_workers,
        nnodes=nnodes,
        node_rank=node_rank,
        master_addr=master_addr,
        master_port=master_port,
        enable_pta_env=enable_pta_env,
    )
    if not ok:
        raise RuntimeError(f"多机脚本改写失败: {source_script} -> rank{node_rank}")
    if remote_dataset_path:
        _replace_dataset_path_in_script(
            node_script,
            original_dataset_path,
            resolved_dataset_path,
            remote_dataset_path,
        )
    return [(node_script, dest_rel), *dataset_items]


def build_remote_portable_upload_items(source_path, dest_rel, target_dir):
    source = Path(source_path)
    target = Path(target_dir) / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if not rewrite_repo_local_paths_for_remote(target):
        raise RuntimeError(f"远端产物路径改写失败: {source}")
    return [(target, dest_rel)]


def read_checkpoint_iteration(load_dir):
    """读取checkpoint目录记录的最近迭代步数，用于修正resume后的实际执行步数。"""
    try:
        tracker = Path(load_dir) / "latest_checkpointed_iteration.txt"
        if not tracker.exists():
            return 0
        return max(0, int(tracker.read_text(encoding="utf-8").strip() or "0"))
    except Exception as exc:
        log_warn(f"读取checkpoint迭代元数据失败，按0处理 | 路径: {load_dir} | 原因: {exc}")
        return 0


def update_mf_yaml_load_checkpoint(yaml_path, ckpt_dir):
    yaml_file = Path(yaml_path)
    ckpt_path = Path(ckpt_dir)
    if not yaml_file.exists():
        log_error(f"MF配置更新失败 | yaml不存在: {yaml_path}")
        return False
    if not ckpt_path.exists() or not any(ckpt_path.iterdir()):
        log_error(f"MF配置更新失败 | 转换后的ckpt目录不存在或为空: {ckpt_dir}")
        return False

    content = yaml_file.read_text(encoding="utf-8")

    ckpt_resolved = str(ckpt_path.resolve())
    replacement = f"load_checkpoint: '{ckpt_resolved}'"
    is_hf_checkpoint = ckpt_resolved.endswith("-hf")
    ckpt_format = "safetensors" if is_hf_checkpoint else "ckpt"
    auto_trans = "true" if is_hf_checkpoint else "false"

    updates = {
        "load_checkpoint": replacement,
        "load_ckpt_format": f"load_ckpt_format: {ckpt_format}",
        "auto_trans_ckpt": f"auto_trans_ckpt: {auto_trans}",
    }

    for field, line in updates.items():
        pattern = rf"^[ \t]*{re.escape(field)}:.*$"
        if re.search(pattern, content, flags=re.MULTILINE):
            content = re.sub(pattern, line, content, flags=re.MULTILINE)
        else:
            content = line + "\n" + content

    yaml_file.write_text(content, encoding="utf-8")
    final_content = yaml_file.read_text(encoding="utf-8")
    return (
        replacement in final_content
        and f"load_ckpt_format: {ckpt_format}" in final_content
        and f"auto_trans_ckpt: {auto_trans}" in final_content
    )


def disable_mf_yaml_load_checkpoint(yaml_path):
    """Disable MF checkpoint loading so MF can run without converted PTA weights."""
    yaml_file = Path(yaml_path)
    if not yaml_file.exists():
        log_error(f"MF配置更新失败 | yaml不存在: {yaml_path}")
        return False

    content = yaml_file.read_text(encoding="utf-8")
    updates = {
        "load_checkpoint": "load_checkpoint: ''",
        "auto_trans_ckpt": "auto_trans_ckpt: false",
    }
    for field, line in updates.items():
        pattern = rf"^[ \t]*{re.escape(field)}:.*$"
        if re.search(pattern, content, flags=re.MULTILINE):
            content = re.sub(pattern, line, content, flags=re.MULTILINE)
        else:
            content = line + "\n" + content

    yaml_file.write_text(content, encoding="utf-8")
    final_content = yaml_file.read_text(encoding="utf-8")
    return "load_checkpoint: ''" in final_content and "auto_trans_ckpt: false" in final_content


def resolve_weight_convert_assets():
    local_ckpt = LMSV_ROOT / "utils" / "runtime" / "convert_ckpt.py"
    local_shell = LMSV_ROOT / "scripts" / "runtime" / "convert.sh"

    if local_ckpt.exists() and local_shell.exists():
        return str(local_ckpt.resolve()), str(local_shell.resolve())

    pta_root = os.environ.get("PTA_PATH") or os.environ.get("PTAPATH")
    if not pta_root:
        return None, None
    pta_root_path = Path(pta_root).expanduser()
    fallback_ckpt_candidates = [
        pta_root_path / "test" / "weight" / "convert_ckpt.py",
        pta_root_path / "MindSpeed-LLM" / "test" / "weight" / "convert_ckpt.py",
    ]
    fallback_sh_candidates = [
        pta_root_path / "test" / "weight" / "convert.sh",
        pta_root_path / "MindSpeed-LLM" / "test" / "weight" / "convert.sh",
    ]

    convert_ckpt = None
    convert_sh = None
    for candidate in fallback_ckpt_candidates:
        if candidate.exists():
            convert_ckpt = str(candidate.resolve())
            break
    for candidate in fallback_sh_candidates:
        if candidate.exists():
            convert_sh = str(candidate.resolve())
            break
    return convert_ckpt, convert_sh


def generate_mf_script(pta_script, mf_script, model_name, train_iters, exec_log_file=None):
    template_path = repo_rel(MF_TEMPLATE_DIR / f"{model_name}.yaml")
    cmd = f'''
    {build_conda_activate_block(Config.MF_ENV, load_ascend=True)}
    python -m utils.runtime.mf_converter \
      -i {shlex.quote(pta_script)} \
      -o {shlex.quote(mf_script)} \
      -m {shlex.quote(model_name)} \
      --template {shlex.quote(template_path)} \
      --train-iters {int(train_iters)}
    '''
    if exec_log_file:
        result = run_shell_to_file(cmd, exec_log_file, check=False)
        return bool(result and result.returncode == 0)
    result = run_shell(cmd, check=False)
    return bool(result and result.returncode == 0)


def convert_pta_checkpoint_for_mf(load_dir, save_dir, model_name, tp, pp, ep, exec_log_file=None):
    convert_entry, convert_shell = resolve_weight_convert_assets()
    if not convert_entry or not convert_shell:
        log_error("未找到PTA权重转换入口 convert_ckpt.py，请检查 PTA_PATH")
        return False

    model_name_for_convert = resolve_task1_weight_convert_model_alias(model_name)
    if model_name_for_convert != str(model_name).strip().lower():
        log_info(f"权重转换模型名映射: {model_name} -> {model_name_for_convert}")

    log_info(f"权重转换将使用: {convert_shell} + {convert_entry}")

    pta_path = shlex.quote(os.environ["PTA_PATH"])
    cmd = f'''
    {build_conda_activate_block(Config.PTA_ENV, load_ascend=True)}
    export PTAPATH={pta_path}
    source scripts/envset/pta.sh
        export LMSV_CONVERT_CKPT_ENTRY={shlex.quote(convert_entry)}
    rm -rf {shlex.quote(save_dir)}
    mkdir -p {shlex.quote(Config.MF_CKPT_ROOT_DIR)}
        bash {shlex.quote(convert_shell)} \
      --load-model-type mg \
      --save-model-type hf \
      --load-dir {shlex.quote(load_dir)} \
      --save-dir {shlex.quote(save_dir)} \
            --model-type-hf {shlex.quote(model_name_for_convert)} \
      --target-tensor-parallel-size {int(tp)} \
      --target-pipeline-parallel-size {int(pp)} \
      --target-expert-parallel-size {int(ep)}
    '''
    if exec_log_file:
        result = run_shell_to_file(cmd, exec_log_file, check=False)
        return bool(result and result.returncode == 0)
    result = run_shell(cmd, check=False)
    return bool(result and result.returncode == 0)


def run_mf_training(mf_yaml_path, card_num, exec_log_file=None, csv_path=None):
    mf_start_script = repo_rel(RUNTIME_SCRIPT_DIR / "mf_start.sh")
    csv_export = "unset LMSV_MF_TRAINING_LOG_CSV"
    if csv_path:
        csv_export = f"export LMSV_MF_TRAINING_LOG_CSV={shlex.quote(_to_abs_path(csv_path))}"
    master_port_export = (
        "if [ -z \"${LMSV_MSRUN_MASTER_PORT:-}\" ]; then "
        "export LMSV_MSRUN_MASTER_PORT=$((12000 + RANDOM % 20000)); "
        "fi"
    )
    # 插桩环境变量传递
    align_dump_vars = ""
    if os.environ.get('LMSV_ALIGN_DUMP_NPY'):
        align_dump_vars = f"""
    export LMSV_ALIGN_DUMP_NPY={os.environ.get('LMSV_ALIGN_DUMP_NPY', '')}
    export LMSV_ALIGN_DUMP_DIR={os.environ.get('LMSV_ALIGN_DUMP_DIR', '/app/lm-sv/lmsv_rec/output/alignment_npy')}
    export LMSV_ALIGN_DUMP_LAYER={os.environ.get('LMSV_ALIGN_DUMP_LAYER', '1')}
    export LMSV_ALIGN_DUMP_TAGS={os.environ.get('LMSV_ALIGN_DUMP_TAGS', 'layer_input,layer_output')}
    export LMSV_ALIGN_DUMP_ONCE={os.environ.get('LMSV_ALIGN_DUMP_ONCE', '1')}
    export LMSV_ALIGN_DUMP_INPUT_LN_WEIGHT={os.environ.get('LMSV_ALIGN_DUMP_INPUT_LN_WEIGHT', '0')}"""
    cluster_env = ""
    target_nnodes = max(1, int(getattr(Config, "TARGET_NNODES", 1) or 1))
    target_world_size = max(1, int(getattr(Config, "TARGET_WORLD_SIZE", card_num) or card_num))
    target_local_workers = max(1, int(getattr(Config, "TARGET_NPUS_PER_NODE", card_num) or card_num))
    if target_nnodes > 1 or target_world_size > int(card_num):
        cluster_env = f"""
    export LMSV_MF_WORKER_NUM={target_world_size}
    export LMSV_MF_LOCAL_WORKER={target_local_workers}
    export LMSV_MF_MASTER_ADDR={shlex.quote(str(Config.TARGET_MASTER_ADDR))}
    export LMSV_MF_MASTER_PORT={int(Config.TARGET_MASTER_PORT)}
    export LMSV_MF_NODE_RANK={int(Config.TARGET_NODE_RANK)}"""
    cmd = f'''
    {build_conda_activate_block(Config.MF_ENV, load_ascend=True)}
    export PYTHONPATH={shlex.quote(str(LMSV_ROOT))}:${{PYTHONPATH:-}}
    {master_port_export}
    {csv_export}{align_dump_vars}{cluster_env}
    bash {shlex.quote(mf_start_script)} {shlex.quote(_to_abs_path(mf_yaml_path))} {int(target_local_workers)}
    '''
    if exec_log_file:
        result = run_shell_to_file(cmd, exec_log_file, check=False)
        return bool(result and result.returncode == 0)
    result = run_shell(cmd, check=False)
    return bool(result and result.returncode == 0)


def _contains_any_text(path_obj, patterns):
    if not path_obj or not path_obj.exists():
        return None
    try:
        text = path_obj.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    for pattern in patterns:
        if pattern in text:
            return pattern
    return None


def _strip_benign_mf_tracebacks(text):
    """Remove known benign MF worker tracebacks from system temp cleanup."""
    if not text:
        return text

    benign_cleanup = re.compile(
        r"Traceback \(most recent call last\):"
        r"(?:(?!\n\d{4}-\d{2}-\d{2}|\n\[).)*?"
        r"OSError: \[Errno 16\] Device or resource busy: '\.nfs[^']*'",
        flags=re.DOTALL,
    )
    return benign_cleanup.sub("", text)


def _contains_mf_fatal_text(path_obj, patterns):
    if not path_obj or not path_obj.exists():
        return None
    try:
        text = path_obj.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    filtered_text = _strip_benign_mf_tracebacks(text)
    for pattern in patterns:
        if pattern in filtered_text:
            return pattern
    return None


def _safe_file_size(path_obj):
    if not path_obj or not path_obj.exists():
        return 0
    try:
        return int(path_obj.stat().st_size)
    except OSError:
        return 0


def _resolve_worker_count_from_script(script_path, default_workers=8):
    path = Path(script_path)
    if not path.exists():
        return max(1, int(default_workers))

    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return max(1, int(default_workers))

    var_values = {}
    for match in re.finditer(r"^\s*([A-Za-z_][A-Za-z0-9_]*)=(\d+)\s*$", content, re.MULTILINE):
        var_values[match.group(1)] = int(match.group(2))

    for key in ("NPUS_PER_NODE", "WORKER_NUM", "LOCAL_WORKER", "WORLD_SIZE"):
        value = var_values.get(key)
        if isinstance(value, int) and value > 0:
            return value

    patterns = (
        r"--nproc_per_node(?:=|\s+)(\d+)\b",
        r"--worker_num(?:=|\s+)(\d+)\b",
        r"msrun_launcher\.sh\s+\S+\s+(\d+)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            return max(1, int(match.group(1)))

    for pattern in (
        r"--nproc_per_node(?:=|\s+)\$([A-Za-z_][A-Za-z0-9_]*)",
        r"--worker_num(?:=|\s+)\$([A-Za-z_][A-Za-z0-9_]*)",
    ):
        match = re.search(pattern, content)
        if match:
            value = var_values.get(match.group(1))
            if isinstance(value, int) and value > 0:
                return value

    return max(1, int(default_workers))


def _resolve_last_worker_log_relpath(script_path, log_dir=None):
    worker_count = _resolve_worker_count_from_script(script_path)
    worker_index = max(0, worker_count - 1)
    log_root = log_dir or Config.MSA_LOG_DIR
    return f"{log_root}/worker_{worker_index}.log"


def _mf_process_running():
    try:
        result = subprocess.run(
            ["bash", "-lc", "pgrep -fa 'run_mindformer.py|msrun --bind_core=True'"],
            capture_output=True,
            text=True,
        )
    except OSError:
        return True
    return result.returncode == 0


def wait_mf_finish(
    expected_csv_path,
    max_wait=1800,
    poll_interval=10,
    exec_log_file=None,
    startup_grace=45,
    progress_log_interval=60,
):
    finish_pattern = "MindFormer 任务运行成功"
    worker_log = LMSV_ROOT /"msrun_log" / "worker_0.log"
    csv_file = LMSV_ROOT / expected_csv_path
    exec_log_path = Path(exec_log_file) if exec_log_file else None
    deadline = time.time() + max_wait
    started_at = time.time()
    last_exec_size = _safe_file_size(exec_log_path)
    last_worker_size = _safe_file_size(worker_log)
    last_growth_at = time.time()
    last_progress_log_at = 0.0
    last_settle_log_at = 0.0
    stagnant_grace = max(20, int(poll_interval) * 3)
    fatal_patterns = [
        "Traceback (most recent call last)",
        "RuntimeError:",
        "ModuleNotFoundError:",
        "FileNotFoundError:",
        "ImportError:",
        "AssertionError:",
        "Segmentation fault",
        "Aborted",
        "killed",
    ]

    while time.time() < deadline:
        log_ready = worker_log.exists() and finish_pattern in worker_log.read_text(encoding="utf-8", errors="ignore")
        csv_ready = csv_file.exists() and csv_file.stat().st_size > 0
        if log_ready and csv_ready:
            log_info(f"MF验证完成，训练日志就绪: {csv_file}")
            return True

        worker_size = _safe_file_size(worker_log)
        exec_size = _safe_file_size(exec_log_path)
        if worker_size > last_worker_size or exec_size > last_exec_size:
            last_growth_at = time.time()
            last_worker_size = max(last_worker_size, worker_size)
            last_exec_size = max(last_exec_size, exec_size)

        worker_fatal = _contains_mf_fatal_text(worker_log, fatal_patterns)
        if worker_fatal:
            log_error(f"MF验证失败：在 worker 日志中检测到异常标记: {worker_fatal}")
            return False

        exec_fatal = _contains_any_text(exec_log_path, fatal_patterns)
        if exec_fatal:
            log_error(f"MF验证失败：在执行日志中检测到异常标记: {exec_fatal}")
            return False

        elapsed = time.time() - started_at
        now = time.time()
        if elapsed >= startup_grace and not _mf_process_running() and not csv_ready:
            stagnant_for = time.time() - last_growth_at
            if stagnant_for >= stagnant_grace:
                log_error(
                    "MF验证失败：进程已结束且日志大小持续未变化，未生成训练日志；"
                    f"worker_size={worker_size}, exec_size={exec_size}, stagnant_for={int(stagnant_for)}s"
                )
                return False
            if now - last_settle_log_at >= max(10, int(progress_log_interval)):
                log_info(
                    "MF进程已结束但日志仍可能在落盘，继续等待；"
                    f"worker_size={worker_size}, exec_size={exec_size}"
                )
                last_settle_log_at = now

        if now - last_progress_log_at >= max(10, int(progress_log_interval)):
            log_info("MF验证仍在进行，等待完成标记/日志文件...")
            last_progress_log_at = now
        time.sleep(max(1, int(poll_interval)))

    log_error("MF验证超时")
    return False


def _build_runtime_hook_env(csv_path=None):
    hook_prefix = f"{RUNTIME_HOOK_DIR}"
    lines = [
        f"export PYTHONPATH={shlex.quote(hook_prefix)}:${{PYTHONPATH:-}}",
        "export LMSV_ENABLE_TRAINING_LOG_PATCH=1",
        "export LMSV_DISABLE_TORCH_COMPILE=${LMSV_DISABLE_TORCH_COMPILE:-0}",
        'if [ "${LMSV_DISABLE_TORCH_COMPILE}" = "1" ]; then',
        "  export TORCH_COMPILE_DISABLE=1",
        "  export TORCHDYNAMO_DISABLE=1",
        "  export PYTORCH_DISABLE_TRITON=1",
        "  export TORCHINDUCTOR_USE_TRITON=0",
        "fi",
    ]
    if csv_path:
        lines.append(f"export LMSV_TRAINING_LOG_CSV={shlex.quote(_to_abs_path(csv_path))}")
    else:
        lines.append("unset LMSV_TRAINING_LOG_CSV")
    lines.append('echo "[LMSV] runtime patch enabled: LMSV_ENABLE_TRAINING_LOG_PATCH=$LMSV_ENABLE_TRAINING_LOG_PATCH"')
    lines.append('echo "[LMSV] runtime patch csv: ${LMSV_TRAINING_LOG_CSV:-<unset>}"')
    lines.append('echo "[LMSV] runtime patch verbose: LMSV_PATCH_LOG=${LMSV_PATCH_LOG:-1}"')
    return "\n".join(lines)


def _build_external_pretrain_resolver(env_type):
    if env_type == 1:
        candidates = [
            "$PTAPATH/MindSpeed-LLM/pretrain_gpt.py",
            "$PTAPATH/pretrain_gpt.py",
        ]
        error = "PTA"
    elif env_type == 2:
        candidates = [
            "$MSAPATH/MSAdapter/pretrain_gpt.py",
            "$MSAPATH/MindSpeed-LLM/pretrain_gpt.py",
            "$MSAPATH/Megatron-LM/pretrain_gpt.py",
            "$MSAPATH/pretrain_gpt.py",
        ]
        error = "MSA"
    else:
        return "unset LMSV_PRETRAIN_GPT"

    quoted_candidates = " ".join(f'"{item}"' for item in candidates)
    return f"""
unset LMSV_PRETRAIN_GPT
for candidate in {quoted_candidates}; do
    if [ -f "$candidate" ]; then
        export LMSV_PRETRAIN_GPT="$candidate"
        break
    fi
done
if [ -z "${{LMSV_PRETRAIN_GPT:-}}" ]; then
    echo "ERROR: 未找到外部 {error} pretrain_gpt.py，请检查 *_PATH 配置" >&2
    exit 1
fi
echo "[LMSV] using pretrain entry: $LMSV_PRETRAIN_GPT"
"""


# ====================== 进程清理 ======================
def cleanup_training_processes(stage_name, iter_num):
    utils.control.clean.kill_pretraingpt()


# ====================== 备份与清理函数 ======================
def init_workspace_global():
    """全局初始化工作目录"""
    log_step("初始化工作目录（全局）| 彻底删除res目录后重建预期结构")
    
    clean_dirs = ["ms/", "pta/", "mf/", "msrun_log/", Config.TMP_ROOT_DIR, Config.DUMP_DIR, Config.CKPT_ROOT_DIR, Config.MF_CKPT_ROOT_DIR, Config.ACC_LOG_ROOT, "res/"]
    for d in clean_dirs:
        path = Path(d)
        if path.exists() or path.is_symlink():
            runtime_helpers.clear_path(path)
            log_info(f"全局清理: 已删除历史残留 {d}")
    
    # 重建res目录
    rebuild_dirs = [
        "ms/", "pta/", "mf/", "msrun_log/", Config.TMP_ROOT_DIR, Config.DUMP_DIR,
        Config.CKPT_ROOT_DIR, Config.MF_CKPT_ROOT_DIR, Config.ACC_LOG_ROOT, Config.MY_PERSIST_ROOT,
        Config.PERSISTENT_LOG_DIR, "res/", f"res/{Config.MODEL_NAME}/",
        "res/training_log_pta/", "res/training_log_msa/", "res/training_log_mf/",
        "res/analyse_report/"
    ]
    for d in rebuild_dirs:
        os.makedirs(d, exist_ok=True)
    
    log_info("全局目录清理完成")


def init_workspace_iter(iter_num):
    """迭代初始化"""
    log_step(f"初始化工作目录（迭代{iter_num}）| 仅清理当前迭代的临时文件")
    
    for d in ["ms/", "pta/", "mf/", "msrun_log/"]:
        runtime_helpers.clear_path(Path(d))
        os.makedirs(d, exist_ok=True)
        log_info(f"迭代{iter_num}清理: 已重建空目录 {d}")
    
    cleanup_training_processes("ITER-INIT", iter_num)
    log_info(f"迭代{iter_num}清理完成")


def backup_and_clean_log_dir(log_dir, stage, iter_num, module):
    """备份并清空日志目录"""
    if not os.path.exists(log_dir):
        log_info(f"[{module}-{stage}] 日志目录不存在，跳过备份清空: {log_dir}")
        return True
    
    files = list(Path(log_dir).iterdir())
    if not files:
        log_info(f"[{module}-{stage}] 日志目录为空，无需备份: {log_dir}")
        return True
    
    # 备份当前迭代的日志
    target_log = f"{log_dir}/training_log-{iter_num}.csv"
    if os.path.exists(target_log):
        backup_dir = create_backup_dir(stage, iter_num, module, "save_logs")
        shutil.copy2(target_log, backup_dir)
        log_backup(f"[{module}-{stage}] 精准备份当前迭代日志: {target_log} -> {backup_dir}")
        os.remove(target_log)
        log_info(f"[{module}-{stage}] 已删除当前迭代日志: {target_log}")
    else:
        log_warn(f"[{module}-{stage}] 未找到当前迭代的日志文件: {target_log}")
    
    return True


def handle_log(log_type, iter_num, log_dst, stage):
    """处理日志迁移"""
    log_src_dir = getattr(Config, f"{log_type}_LOG_SRC_DIR".upper())
    log_name_rule = getattr(Config, f"{log_type}_LOG_NAME_RULE".upper(), "training_log-{i}.csv")
    log_src = log_src_dir + "/" + log_name_rule.replace("{i}", str(iter_num))
    
    log_info(f"[{log_type}-{stage}] 迁移日志到精度目录...")
    
    if not os.path.exists(log_src):
        log_error(f"[{log_type}-{stage}] 日志源文件不存在: {log_src}")
        return False
    
    shutil.copy2(log_src, log_dst)
    
    if not os.path.exists(log_dst) or os.path.getsize(log_dst) == 0:
        log_error(f"[{log_type}-{stage}] 迁移后日志为空!")
        return False
    
    # 备份
    log_backup_dir = create_backup_dir(stage, iter_num, log_type.lower(), "logs")
    shutil.copy2(log_src, f"{log_backup_dir}/original_{os.path.basename(log_src)}")
    shutil.copy2(log_dst, f"{log_backup_dir}/migrated_{os.path.basename(log_dst)}")
    log_backup(f"[{log_type}-{stage}] 原始日志精准备份")
    
    log_info(f"[{log_type}-{stage}] 日志处理完成")
    return True


def dump_logs(iter_num, run_persist_dir):
    """备份核心状态信息"""
    dump_iter_dir = f"{run_persist_dir}/iter_{iter_num}"
    os.makedirs(dump_iter_dir, exist_ok=True)
    log_info(f"【开始备份】迭代{iter_num} -> 最终目录: {dump_iter_dir}")

    # 同时备份框架日志目录和真实输出目录日志（output/msrun_log）。
    log_dir_candidates = [
        ("msrun_log", Path(Config.MSA_LOG_DIR)),
        ("output_msrun_log", LMSV_ROOT / "output" / "msrun_log"),
    ]
    seen_real_paths = set()
    for alias, log_dir in log_dir_candidates:
        real_path = log_dir.resolve()
        if real_path in seen_real_paths:
            continue
        seen_real_paths.add(real_path)

        if log_dir.exists():
            shutil.copytree(
                str(log_dir),
                f"{dump_iter_dir}/{alias}",
                dirs_exist_ok=True,
            )
            log_info(f"【备份成功】{alias}目录: {log_dir}")
        else:
            log_warn(f"【备份缺失】{alias}目录不存在: {log_dir}")
    
    log_info(f"【备份完成】迭代{iter_num}所有文件已备份")


def finalize_iter(iter_num, run_persist_dir, pta_result="OK", msa_result="OK", mf_result="OK"):
    """迭代收尾"""
    # 备份
    dump_logs(iter_num, run_persist_dir)
    
    # 失败标记
    dump_iter_dir = f"{run_persist_dir}/iter_{iter_num}"
    if pta_result == "ERROR" or msa_result == "ERROR" or mf_result == "ERROR":
        os.makedirs(dump_iter_dir, exist_ok=True)
        with open(f"{dump_iter_dir}/FAILED_FLAG", "w") as f:
            f.write(f"PTA={pta_result} MSA={msa_result} MF={mf_result}\n")
        with open(f"{dump_iter_dir}/failure_info.txt", "w") as f:
            f.write(f"FAILED_COMPONENTS: PTA={pta_result} MSA={msa_result} MF={mf_result}\n")
        log_warn(f"【迭代失败】已在{dump_iter_dir}创建失败标记文件")
    
    pta_csv = Path(Config.PTA_LOG_SRC_DIR) / f"training_log-{iter_num}.csv"
    msa_csv = Path(Config.MSA_LOG_SRC_DIR) / f"training_log-{iter_num}.csv"
    mf_csv = Path(Config.MF_LOG_SRC_DIR) / f"training_log-{iter_num}.csv"
    backup_artifact_to_output(pta_csv, run_persist_dir, iter_num, "", f"training_log_pta-{iter_num}.csv")
    backup_artifact_to_output(msa_csv, run_persist_dir, iter_num, "", f"training_log_msa-{iter_num}.csv", missing_log_level="info")
    backup_artifact_to_output(mf_csv, run_persist_dir, iter_num, "", f"training_log_mf-{iter_num}.csv", missing_log_level="info")


def write_iteration_status(
    iter_num,
    run_persist_dir,
    overall_status,
    reason="",
    *,
    mutate_result="SKIP",
    pta_save_result="SKIP",
    pta_load_result="SKIP",
    msa_load_result="SKIP",
    mf_result="SKIP",
    analyse_result="SKIP",
):
    iter_dir = Path(run_persist_dir) / f"iter_{iter_num}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "task_type": 1,
        "iteration": iter_num,
        "overall_status": overall_status,
        "reason": reason,
        "components": {
            "MUTATE": mutate_result,
            "PTA_SAVE": pta_save_result,
            "PTA_LOAD": pta_load_result,
            "MSA_LOAD": msa_load_result,
            "MF": mf_result,
            "ANALYSE": analyse_result,
        },
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(iter_dir / "status.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    if overall_status != "PASS":
        with open(iter_dir / "FAILED_FLAG", "w", encoding="utf-8") as handle:
            handle.write(
                "MUTATE={MUTATE} PTA_SAVE={PTA_SAVE} PTA_LOAD={PTA_LOAD} "
                "MSA_LOAD={MSA_LOAD} MF={MF} ANALYSE={ANALYSE}\n".format(**payload["components"])
            )
        with open(iter_dir / "failure_info.txt", "w", encoding="utf-8") as handle:
            handle.write(
                "FAILED_COMPONENTS: "
                "MUTATE={MUTATE} PTA_SAVE={PTA_SAVE} PTA_LOAD={PTA_LOAD} "
                "MSA_LOAD={MSA_LOAD} MF={MF} ANALYSE={ANALYSE}\n".format(**payload["components"])
            )
            if reason:
                handle.write(f"REASON: {reason}\n")


# ====================== 主流程 ======================
def run_mutate(exec_log_file=None):
    """执行模型变异"""
    mutate_args = (
        f"-c {MODEL_CONFIG_REL} -r {Config.TOTAL_ITER} --mutnm {Config.MUTNM} "
        f"-n {Config.MUTATE_NODE_NUM} -m {MODEL_CONFIG_REL}/{Config.MODEL_NAME}.yaml "
        f"--args_path {MUTATION_SCHEMA_REL}"
    )
    pta_path = shlex.quote(os.environ["PTA_PATH"])
    cmd = f'''
    {build_conda_activate_block(Config.PTA_ENV, load_ascend=True)}
    export PTAPATH={pta_path}
    source scripts/envset/pta.sh
    export MUTATE_ROUND=${{MUTATE_ROUND:-1}}
    export MUTATE_ARGS={shlex.quote(mutate_args)}
    bash {shlex.quote(f"{MUTATION_SCRIPT_REL}/mutate-auto.sh")}
    '''
    if exec_log_file:
        result = run_shell_to_file(cmd, exec_log_file, check=False)
        return bool(result and result.returncode == 0)
    result = run_shell(cmd)
    return bool(result and result.returncode == 0)


def generate_pta_script(input_file, pta_script, model_name=None, enable_deepseek_profile=False, exec_log_file=None):
    """生成PTA训练脚本"""
    input_shell = f"{SCRIPT_TEMPLATE_REL}/pretrain_mutated_{Config.MODEL_NAME}.sh"
    pta_path = shlex.quote(os.environ["PTA_PATH"])
    data_path = shlex.quote(Config.DATA_PATH)
    parallel_mutate_output_dir = shlex.quote(Config.PARALLEL_MUTATE_TMP_DIR)
    deepseek_profile_line = f"      --enable_deepseek_profile \\\n" if enable_deepseek_profile else ""
    cmd = f'''
    {build_conda_activate_block(Config.PTA_ENV, load_ascend=True)}
    export PTAPATH={pta_path}
    source scripts/envset/pta.sh
    python -m utils.runtime.mutate_and_forward.parallel_mutate \
      -i {input_file} \
      -o {parallel_mutate_output_dir} \
      -isc {input_shell} \
      -osc {pta_script} \
      --model_name {shlex.quote(str(model_name or Config.MODEL_NAME))} \
{deepseek_profile_line}      --data_path {data_path}
    '''
    if exec_log_file:
        result = run_shell_to_file(cmd, exec_log_file, check=False)
        return bool(result and result.returncode == 0)
    result = run_shell(cmd)
    return bool(result and result.returncode == 0)


def convert_msa_script(pta_script, msa_script, exec_log_file=None):
    """转换MSA脚本"""
    pta_path = shlex.quote(os.environ["PTA_PATH"])
    cmd = f'''
    {build_conda_activate_block(Config.PTA_ENV, load_ascend=True)}
    export PTAPATH={pta_path}
    source scripts/envset/pta.sh
    python -m utils.runtime.convert_pretrain_script {pta_script} {msa_script}
    '''
    if exec_log_file:
        result = run_shell_to_file(cmd, exec_log_file, check=False)
        return bool(result and result.returncode == 0)
    result = run_shell(cmd)
    return bool(result and result.returncode == 0)


def run_training(
    script,
    env_name,
    env_type,
    exec_log_file=None,
    csv_path=None,
): # env_type: 1.PTA 2.MSA 3.MF或其它
    """运行训练脚本"""
    if not RUNTIME_HOOK_DIR.exists():
        log_error(f"运行时hook目录不存在: {RUNTIME_HOOK_DIR}")
        return False

    active_env = env_name or ""
    script_quoted = shlex.quote(script)
    runtime_hook_env = _build_runtime_hook_env(csv_path)
    pretrain_resolver = _build_external_pretrain_resolver(env_type)
    sigterm_shield = runtime_helpers.build_sigterm_shield_block()
    if env_type == 1:
        pta_path = shlex.quote(os.environ["PTA_PATH"])
        cmd = f'''
        {build_conda_activate_block(active_env, load_ascend=True)}
        export PTAPATH={pta_path}
        source scripts/envset/pta.sh
        {sigterm_shield}
        {runtime_hook_env}
        {pretrain_resolver}
        bash -e -o pipefail {script_quoted}
        '''
    elif env_type == 2:
        msa_path = shlex.quote(os.environ["MSA_PATH"])
        cmd = f'''
        {build_conda_activate_block(active_env, load_ascend=True)}
        export MSAPATH={msa_path}
        source scripts/envset/msa.sh
        {runtime_hook_env}
        {pretrain_resolver}
        bash -e -o pipefail {script_quoted}
        '''
    elif env_type == 3:
        cmd = f'''
        {runtime_hook_env}
        {pretrain_resolver}
        bash -e -o pipefail {script_quoted}
        '''
    else:
        log_error(f"不支持的训练环境类型: {env_type}")
        return False
    timeout = Config.PTA_MAX_RUNTIME if env_type == 1 else None
    timeout_label = "PTA执行" if env_type == 1 else None
    if exec_log_file:
        result = run_shell_to_file(cmd, exec_log_file, check=False, timeout=timeout, timeout_label=timeout_label)
    else:
        result = run_shell(cmd, check=False, timeout=timeout, timeout_label=timeout_label)
    return result and result.returncode == 0


def csv_has_iteration(csv_path, iteration):
    return data_helpers.csv_has_iteration(csv_path, iteration)


def csv_iteration_is_valid(csv_path, iteration):
    return data_helpers.csv_iteration_is_valid(csv_path, iteration)


def csv_has_valid_metrics_row(csv_path):
    return data_helpers.csv_has_valid_metrics_row(csv_path)


def should_treat_pta_load_as_success(run_ok, csv_path, iteration):
    """PTA-LOAD 即使 exit!=0，只要当前轮 step 日志已产出有效指标，也视作成功。"""
    if run_ok:
        return True

    if csv_has_valid_metrics_row(csv_path):
        log_warn(
            "PTA-LOAD 进程退出码非0，但检测到当前轮 step 日志已包含有效结果，"
            f"按成功处理 | iter={iteration} | csv={csv_path}"
        )
        return True

    return False


def wait_msa_finish(iter_num):
    """等待 MSA 校验完成。成功以日志稳定且当前轮 step 日志存在有效指标为准。"""
    log_step(f"等待MSA验证完成 | 迭代{iter_num}")
    log_path = LMSV_ROOT / Config.MSA_MONITOR_LOGS[-1]
    csv_path = LMSV_ROOT / Config.MSA_LOG_SRC_DIR / f"training_log-{iter_num}.csv"
    return runtime_helpers.wait_msa_finish(
        iter_num=iter_num,
        log_path=log_path,
        total_timeout=Config.MSA_MAX_RUNTIME,
        init_wait=Config.LOG_INIT_WAIT,
        stable_threshold=Config.LOG_STABLE_THRESHOLD,
        poll_interval=20,
        log_info=log_info,
        log_error=log_error,
        success_checker=lambda: csv_has_valid_metrics_row(csv_path),
        result_exists_checker=lambda: Path(csv_path).exists() and Path(csv_path).stat().st_size > 0,
    )


# ====================== 主函数 ======================
def main(params):
    """主流程"""
    # 加载配置
    project_tmp_root = configure_project_tmp_env()
    utils.control.clean.kill_pretraingpt()
    
    MODEL_NAME = params.get('MODEL_NAME', Config.MODEL_NAME)
    TOTAL_ITER = params.get('TOTAL_ITER', Config.TOTAL_ITER)
    SUPPORT_MF = params.get('SUPPORT_MF', Config.SUPPORT_MF)
    compare_mode_raw = str(params.get('COMPARE_MODE', '') or '').strip().lower()
    if compare_mode_raw in {'pta_msa', 'pta_mf'}:
        COMPARE_MODE = compare_mode_raw
    else:
        # Backward compatibility for old configs using SUPPORT_MF only.
        COMPARE_MODE = 'pta_mf' if _to_bool(SUPPORT_MF) else 'pta_msa'
    run_msa = COMPARE_MODE == 'pta_msa'
    run_mf = COMPARE_MODE == 'pta_mf'
    ENABLE_WEIGHT_CONVERT = _to_bool(params.get('ENABLE_WEIGHT_CONVERT', Config.ENABLE_WEIGHT_CONVERT))
    ENABLE_MF_WEIGHT_LOAD = _to_bool(params.get('ENABLE_MF_WEIGHT_LOAD', Config.ENABLE_MF_WEIGHT_LOAD))
    BASE_SEED = params.get('BASE_SEED', Config.BASE_SEED)
    MUTNM = params.get('MUTNM', 2)
    SAVE_STEPS = params.get('SAVE_STEPS', Config.SAVE_TRAIN_ITERS)
    LOAD_STEPS = params.get('LOAD_STEPS', Config.LOAD_TRAIN_ITERS)
    MF_LOSS_TOLERANCE = float(params.get('MF_LOSS_TOLERANCE', Config.MF_LOSS_TOLERANCE))
    PTA_MAX_RUNTIME = params.get('PTA_MAX_RUNTIME', Config.PTA_MAX_RUNTIME)
    MSA_MAX_RUNTIME = params.get('MSA_MAX_RUNTIME', params.get('MAX_VALIDATE_TIME', Config.MSA_MAX_RUNTIME))
    LOG_INIT_WAIT = params.get('LOG_INIT_WAIT', Config.LOG_INIT_WAIT)
    LOG_STABLE_THRESHOLD = params.get('LOG_STABLE_THRESHOLD', Config.LOG_STABLE_THRESHOLD)
    TARGET_TENSOR_PARALLEL_SIZE = _parse_optional_positive_int(
        params.get('TARGET_TENSOR_PARALLEL_SIZE', Config.TARGET_TENSOR_PARALLEL_SIZE)
    ) or Config.TARGET_TENSOR_PARALLEL_SIZE
    TARGET_PIPELINE_PARALLEL_SIZE = _parse_optional_positive_int(
        params.get('TARGET_PIPELINE_PARALLEL_SIZE', Config.TARGET_PIPELINE_PARALLEL_SIZE)
    ) or Config.TARGET_PIPELINE_PARALLEL_SIZE
    TARGET_EXPERT_PARALLEL_SIZE = _parse_optional_positive_int(
        params.get('TARGET_EXPERT_PARALLEL_SIZE', Config.TARGET_EXPERT_PARALLEL_SIZE)
    ) or Config.TARGET_EXPERT_PARALLEL_SIZE
    TARGET_NPUS_PER_NODE = int(params.get('TARGET_NPUS_PER_NODE', Config.TARGET_NPUS_PER_NODE) or 0)
    TARGET_WORLD_SIZE = int(params.get('TARGET_WORLD_SIZE', Config.TARGET_WORLD_SIZE) or 0)
    TARGET_NNODES = max(1, int(params.get('TARGET_NNODES', Config.TARGET_NNODES) or Config.TARGET_NNODES))
    TARGET_NODE_RANK = int(params.get('TARGET_NODE_RANK', Config.TARGET_NODE_RANK))
    TARGET_MASTER_ADDR = str(params.get('TARGET_MASTER_ADDR', Config.TARGET_MASTER_ADDR))
    TARGET_REMOTE_MASTER_ADDR = str(params.get('TARGET_MASTER_ADDR', Config.TARGET_REMOTE_MASTER_ADDR))
    TARGET_MASTER_PORT = int(params.get('TARGET_MASTER_PORT', Config.TARGET_MASTER_PORT))
    cluster_cfg = parse_task123_cluster_config(params)
    cluster = None
    cluster_session_id = ""
    DATA_PATH = params.get('DATA_PATH') or os.environ.get('DATA_PATH') or os.environ.get('DATAPATH') or Config.DATA_PATH
    DATA_PATH = _to_abs_path(DATA_PATH)

    # DeepSeekV3 在 Task1 pta_mf 场景下强制不加载权重：
    # 无论配置如何都跳过 PTA->MF 转换与 load_checkpoint 回填。
    model_name_norm = str(MODEL_NAME).strip().lower()
    if run_mf and model_name_norm == "deepseekv3":
        if ENABLE_MF_WEIGHT_LOAD or ENABLE_WEIGHT_CONVERT:
            log_warn(
                "deepseekv3 在 Task1 pta_mf 模式下强制关闭 MF 权重转换与加载，"
                "忽略 ENABLE_MF_WEIGHT_LOAD/ENABLE_WEIGHT_CONVERT 配置"
            )
        ENABLE_MF_WEIGHT_LOAD = False
        ENABLE_WEIGHT_CONVERT = False
    
    # 更新配置
    Config.MODEL_NAME = MODEL_NAME
    Config.TOTAL_ITER = TOTAL_ITER
    Config.COMPARE_MODE = COMPARE_MODE
    Config.BASE_SEED = BASE_SEED
    Config.MUTNM = MUTNM
    Config.SUPPORT_MF = run_mf
    Config.ENABLE_WEIGHT_CONVERT = ENABLE_WEIGHT_CONVERT
    Config.ENABLE_MF_WEIGHT_LOAD = ENABLE_MF_WEIGHT_LOAD
    Config.SAVE_TRAIN_ITERS = SAVE_STEPS
    Config.LOAD_TRAIN_ITERS = LOAD_STEPS
    Config.MF_LOSS_TOLERANCE = MF_LOSS_TOLERANCE
    Config.PTA_MAX_RUNTIME = PTA_MAX_RUNTIME
    Config.MSA_MAX_RUNTIME = MSA_MAX_RUNTIME
    Config.LOG_INIT_WAIT = LOG_INIT_WAIT
    Config.LOG_STABLE_THRESHOLD = LOG_STABLE_THRESHOLD
    Config.TARGET_TENSOR_PARALLEL_SIZE = TARGET_TENSOR_PARALLEL_SIZE
    Config.TARGET_PIPELINE_PARALLEL_SIZE = TARGET_PIPELINE_PARALLEL_SIZE
    Config.TARGET_EXPERT_PARALLEL_SIZE = TARGET_EXPERT_PARALLEL_SIZE
    Config.TARGET_NPUS_PER_NODE = TARGET_NPUS_PER_NODE
    Config.TARGET_WORLD_SIZE = TARGET_WORLD_SIZE
    Config.TARGET_NNODES = TARGET_NNODES
    Config.TARGET_NODE_RANK = TARGET_NODE_RANK
    Config.TARGET_MASTER_ADDR = TARGET_MASTER_ADDR
    Config.TARGET_REMOTE_MASTER_ADDR = TARGET_REMOTE_MASTER_ADDR
    Config.TARGET_MASTER_PORT = TARGET_MASTER_PORT
    Config.DATA_PATH = DATA_PATH
    task_tmp_root = Path(project_tmp_root) / "task1"
    Config.CKPT_ROOT_DIR = str(task_tmp_root / "ckpt")
    Config.MF_CKPT_ROOT_DIR = str(task_tmp_root / "ckpt_mf")
    Config.TMP_ROOT_DIR = str(task_tmp_root / "test")
    Config.PARALLEL_MUTATE_TMP_DIR = str(task_tmp_root / "parallel_mutate")
    Config.CKPT_ROOT_DIR = _to_abs_path(Config.CKPT_ROOT_DIR)
    Config.MF_CKPT_ROOT_DIR = _to_abs_path(Config.MF_CKPT_ROOT_DIR)
    Config.TMP_ROOT_DIR = _to_abs_path(Config.TMP_ROOT_DIR)
    Config.MY_PERSIST_ROOT = _to_abs_path(os.environ['LMSV_OUTPATH'])
    Config.PTA_ENV = os.environ['PTA_NAME']
    Config.MS_ENV = os.environ.get('MSA_NAME', Config.MS_ENV)
    Config.MF_ENV = os.environ.get('MF_NAME', Config.MF_ENV)

    if run_msa:
        if not os.environ.get('MSA_NAME'):
            log_error('当前为 pta_msa 模式，缺少 MSA_NAME 配置')
            return 1
        if not os.environ.get('MSA_PATH'):
            log_error('当前为 pta_msa 模式，缺少 MSA_PATH 配置')
            return 1
    if run_mf:
        if not os.environ.get('MF_NAME'):
            log_error('当前为 pta_mf 模式，缺少 MF_NAME 配置')
            return 1
    if cluster_cfg.enabled:
        cluster = ClusterMaster(cluster_cfg, log_info, log_warn, log_error)
        cluster.preflight()
        Config.TARGET_NPUS_PER_NODE = cluster.local_worker_count()
        Config.TARGET_WORLD_SIZE = cluster.total_workers()
        Config.TARGET_NNODES = cluster.config.nnodes
        Config.TARGET_NODE_RANK = cluster.config.node_rank
        Config.TARGET_MASTER_ADDR = cluster.config.local_master_addr
        Config.TARGET_REMOTE_MASTER_ADDR = cluster.config.broadcast_master_addr
        Config.TARGET_MASTER_PORT = cluster.config.master_port
    os.environ['DATA_PATH'] = DATA_PATH
    os.environ['DATAPATH'] = DATA_PATH
    os.environ['BASE_SEED'] = str(Config.BASE_SEED)
    
    msa_or_mf = "MSA" if run_msa else "MF"
    log_step("任务配置加载完成")
    log_kv("配置", "模型名称", MODEL_NAME)
    log_kv("配置", "总迭代数", TOTAL_ITER)
    log_kv("配置", "COMPARE_MODE", COMPARE_MODE)
    log_kv("配置", "当前对比目标", msa_or_mf)
    log_kv("配置", "是否运行 MSA", run_msa)
    log_kv("配置", "是否运行 MF", run_mf)
    log_kv("配置", "是否启用 MF 权重转换", ENABLE_WEIGHT_CONVERT)
    log_kv("配置", "是否启用 MF 权重加载", ENABLE_MF_WEIGHT_LOAD)
    log_kv("配置", "基础随机种子", BASE_SEED)
    log_kv("配置", "每轮变异参数数量", MUTNM)
    log_kv("配置", "SAVE 模式训练轮数", SAVE_STEPS)
    log_kv("配置", "LOAD 模式训练轮数", LOAD_STEPS)
    log_kv("配置", "MF loss对齐阈值", MF_LOSS_TOLERANCE)
    log_kv("配置", "PTA 最大运行时间", PTA_MAX_RUNTIME)
    log_kv("配置", f"{msa_or_mf} 最大运行时间", MSA_MAX_RUNTIME)
    log_kv("配置", "日志初始化等待", LOG_INIT_WAIT)
    log_kv("配置", "日志稳定阈值", LOG_STABLE_THRESHOLD)
    log_kv(
        "配置",
        "分布式设置",
        (
            f"TP={Config.TARGET_TENSOR_PARALLEL_SIZE} | PP={Config.TARGET_PIPELINE_PARALLEL_SIZE} | "
            f"EP={Config.TARGET_EXPERT_PARALLEL_SIZE} | NPUS_PER_NODE={Config.TARGET_NPUS_PER_NODE or '-'} | "
            f"NNODES={Config.TARGET_NNODES} | WORLD_SIZE={Config.TARGET_WORLD_SIZE or '-'}"
        ),
    )
    if cluster is not None:
        slave_summary = ", ".join(
            f"rank{node.node_rank}@{node.endpoint}:{cluster.slave_worker_count(node)}卡"
            for node in cluster.config.slaves
        )
        log_kv("配置", "多机模式", f"启用 | MASTER={cluster.config.master_addr}:{cluster.config.master_port}")
        log_kv("配置", "多机节点", f"本机rank{cluster.config.node_rank}:{cluster.local_worker_count()}卡 | {slave_summary}")
    log_kv("配置", "项目临时目录", project_tmp_root)
    log_kv("配置", "数据集路径", DATA_PATH)
    
    ensure_task1_model_supported_for_mode(MODEL_NAME, COMPARE_MODE, SCRIPT_TEMPLATE_DIR)
    
    # 初始化
    log_step(f"=============== 自动化变异+PTA/{msa_or_mf}训练流程 启动 ===============")
    log_kv("概览", "模型名称", MODEL_NAME)
    log_kv("概览", "迭代数", TOTAL_ITER)
    log_kv("概览", "训练配置", f"SAVE({SAVE_STEPS}轮) | LOAD({LOAD_STEPS}轮)")
    log_kv("概览", "开始时间", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # 初始化工作目录
    init_workspace_global()
    
    # 创建持久化目录
    run_persist_dir = (Path(Config.MY_PERSIST_ROOT) / "iters").resolve()
    run_persist_dir.mkdir(parents=True, exist_ok=True)
    if cluster is not None:
        cluster_session_id = _build_cluster_session_id()
        cluster.prepare_session(cluster_session_id)

    def refresh_iteration_analysis():
        try:
            from utils.analyze.task1_result import analyze_task1_run

            analyze_task1_run(
                output_root=Path(Config.MY_PERSIST_ROOT).resolve(),
                run_dir=run_persist_dir,
                model_name=MODEL_NAME,
                planned_iterations=TOTAL_ITER,
            )
        except Exception as exc:
            log_warn(f"迭代分析刷新失败，已跳过: {exc}")
    
    # 统计
    pta_success_count = 0
    msa_success_count = 0
    mf_success_count = 0
    
    # 主循环
    for i in range(1, TOTAL_ITER + 1):
        pta_result = "OK"
        msa_result = "OK"
        mf_result = "OK"
        mutate_stage_result = "SKIP"
        pta_save_stage_result = "SKIP"
        pta_load_stage_result = "SKIP"
        msa_load_stage_result = "SKIP" if run_msa else "DISABLED"
        mf_stage_result = "SKIP" if run_mf else "DISABLED"
        analyse_stage_result = "SKIP"
        iter_reason = ""
        
        log_step(f"开始迭代 {i}/{TOTAL_ITER}")
        
        # 初始化迭代
        init_workspace_iter(i)
        
        # 路径配置
        iter_ckpt_artifact_path = (Path(Config.CKPT_ROOT_DIR) / f"{MODEL_NAME}-{i}").resolve()
        iter_ckpt_path = str(iter_ckpt_artifact_path)
        input_file_rel = f"res/{MODEL_NAME}/mutating-{i}.json"
        input_file_path = input_file_rel
        input_file_abs = (LMSV_ROOT / input_file_rel).resolve()
        pta_dir_rel = f"pta/{MODEL_NAME}"
        pta_script_rel = f"{pta_dir_rel}/pretrain_mutated_{MODEL_NAME}-{i}.sh"
        pta_script_path = pta_script_rel
        msa_dir_rel = f"ms/{MODEL_NAME}"
        msa_script_rel = f"{msa_dir_rel}/pretrain_mutated_{MODEL_NAME}-{i}.sh"
        msa_script_path = msa_script_rel
        mf_dir_rel = f"mf/{MODEL_NAME}"
        mf_script_rel = f"{mf_dir_rel}/pretrain_mutated_{MODEL_NAME}-{i}.yaml"
        mf_script_path = mf_script_rel
        pta_acc_log = f"{Config.ACC_LOG_ROOT}/pta_{MODEL_NAME}-{i}.log"
        msa_acc_log = f"{Config.ACC_LOG_ROOT}/msa_{MODEL_NAME}-{i}.log"
        mf_acc_log = f"{Config.ACC_LOG_ROOT}/mf_{MODEL_NAME}-{i}.log"
        pta_csv_log = f"{Config.PTA_LOG_SRC_DIR}/training_log-{i}.csv"
        msa_csv_log = f"{Config.MSA_LOG_SRC_DIR}/training_log-{i}.csv"
        mf_csv_log = f"{Config.MF_LOG_SRC_DIR}/training_log-{i}.csv"
        mf_ckpt_dir = str((Path(Config.MF_CKPT_ROOT_DIR) / f"{MODEL_NAME}-{i}-hf").resolve())
        iter_runtime_log_dir = f"{run_persist_dir}/iter_{i}/runtime_logs"
        os.makedirs(iter_runtime_log_dir, exist_ok=True)
        mutate_exec_log = f"{iter_runtime_log_dir}/pta_mutate_iter{i}.log"
        pta_script_gen_exec_log = f"{iter_runtime_log_dir}/pta_script_gen_iter{i}.log"
        msa_script_convert_exec_log = f"{iter_runtime_log_dir}/msa_script_convert_iter{i}.log"
        pta_save_exec_log = f"{iter_runtime_log_dir}/pta_save_iter{i}.log"
        pta_load_exec_log = f"{iter_runtime_log_dir}/pta_load_iter{i}.log"
        msa_load_exec_log = f"{iter_runtime_log_dir}/msa_load_iter{i}.log"
        mf_script_gen_exec_log = f"{iter_runtime_log_dir}/mf_script_gen_iter{i}.log"
        mf_convert_exec_log = f"{iter_runtime_log_dir}/mf_convert_iter{i}.log"
        mf_run_exec_log = f"{iter_runtime_log_dir}/mf_run_iter{i}.log"
        cluster_stage_tmp_dir = TASK1_TMP_ROOT / "cluster_bundle" / (cluster_session_id or "local") / f"iter_{i}"
        cluster_stage_tmp_dir.mkdir(parents=True, exist_ok=True)
        
        os.makedirs(pta_dir_rel, exist_ok=True)
        if run_msa:
            os.makedirs(msa_dir_rel, exist_ok=True)
        if run_mf:
            os.makedirs(mf_dir_rel, exist_ok=True)

        def finalize_current_iter():
            overall_status = "PASS"
            if mutate_stage_result == "ERROR":
                overall_status = "MUTATION_FAILED"
            elif any(
                stage == "ERROR"
                for stage in (
                    pta_save_stage_result,
                    pta_load_stage_result,
                    msa_load_stage_result,
                    mf_stage_result,
                )
            ):
                overall_status = "EXECUTION_FAILED"

            finalize_iter(i, run_persist_dir, pta_result, msa_result, mf_result)
            write_iteration_status(
                i,
                run_persist_dir,
                overall_status,
                iter_reason,
                mutate_result=mutate_stage_result,
                pta_save_result=pta_save_stage_result,
                pta_load_result=pta_load_stage_result,
                msa_load_result=msa_load_stage_result,
                mf_result=mf_stage_result,
                analyse_result=analyse_stage_result,
            )
        
        try:
            # 1. 执行模型变异
            log_step("1.1 执行模型变异")
            os.environ["MUTATE_ROUND"] = str(i)
            log_info(f"变异执行日志路径: {mutate_exec_log}")
            mutate_ok = run_mutate(exec_log_file=mutate_exec_log)
            backup_runtime_log_to_output(mutate_exec_log, run_persist_dir, i)
            mutate_ok = should_treat_mutate_as_success(mutate_ok, i)
            if not mutate_ok:
                log_error("变异脚本执行失败！跳过本轮")
                mutate_stage_result = "ERROR"
                iter_reason = "变异脚本执行失败"
                pta_result = "ERROR"
                finalize_current_iter()
                continue
            mutate_stage_result = "OK"

            if not validate_mutation_artifacts(i, run_persist_dir, mutate_exec_log):
                mutate_stage_result = "ERROR"
                iter_reason = "变异未生成可加载产物"
                pta_result = "ERROR"
                finalize_current_iter()
                continue
            
            # 备份JSON
            json_backup_dir = create_backup_dir("save", i, "common", "json")
            if os.path.exists(input_file_path):
                shutil.copy2(input_file_path, json_backup_dir)
            
            # 2. 生成PTA训练脚本
            log_step("1.2 生成PTA训练脚本")
            log_info(f"PTA脚本生成日志路径: {pta_script_gen_exec_log}")
            pta_script_generated = generate_pta_script(
                input_file_rel,
                pta_script_rel,
                model_name=MODEL_NAME,
                enable_deepseek_profile=str(MODEL_NAME).lower() == "deepseekv3",
                exec_log_file=pta_script_gen_exec_log,
            )
            backup_runtime_log_to_output(pta_script_gen_exec_log, run_persist_dir, i)
            if not pta_script_generated:
                log_error("PTA脚本生成失败！跳过本轮")
                iter_reason = "PTA脚本生成失败"
                pta_result = "ERROR"
                finalize_current_iter()
                continue
            
            if not check_script_valid(pta_script_path):
                iter_reason = "PTA脚本校验失败"
                pta_result = "ERROR"
                finalize_current_iter()
                continue

            if not ensure_external_pretrain_entry(pta_script_path):
                iter_reason = "PTA外部入口校验失败"
                pta_result = "ERROR"
                finalize_current_iter()
                continue

            if not sanitize_task1_mutation_runtime_flags(pta_script_path):
                iter_reason = "PTA脚本运行时参数清理失败"
                pta_result = "ERROR"
                finalize_current_iter()
                continue

            if not sanitize_moe_expert_bias_aux_loss(pta_script_path):
                iter_reason = "PTA脚本MoE专家偏置兼容化失败"
                pta_result = "ERROR"
                finalize_current_iter()
                continue

            if not sanitize_alibi_script(pta_script_path):
                iter_reason = "PTA脚本ALiBi兼容化失败"
                pta_result = "ERROR"
                finalize_current_iter()
                continue

            if not sanitize_pangu_script(pta_script_path):
                iter_reason = "PTA脚本Pangu兼容化失败"
                pta_result = "ERROR"
                finalize_current_iter()
                continue

            if not sanitize_swiglu_fusion_script(pta_script_path):
                iter_reason = "PTA脚本SwiGLU融合兼容化失败"
                pta_result = "ERROR"
                finalize_current_iter()
                continue

            if not sanitize_rotary_base_script(pta_script_path):
                iter_reason = "PTA脚本rotary-base兼容化失败"
                pta_result = "ERROR"
                finalize_current_iter()
                continue

            if not apply_script_constraints(pta_script_path):
                iter_reason = "PTA脚本约束失败"
                pta_result = "ERROR"
                finalize_current_iter()
                continue
            
            # 备份原始脚本
            pta_save_ori_dir = create_backup_dir("save", i, "pta", "original")
            shutil.copy2(pta_script_path, pta_save_ori_dir)
            
            # 3. 预配置PTA SAVE参数
            log_step("1.3 预配置PTA SAVE参数")
            reg_train_iters = r"--train-iters[[:space:]]\+[0-9]\+"
            reg_save = r"--save[[:space:]]\+[^\ ]\+"
            modify_script_param(pta_script_path, reg_train_iters, f"--train-iters {SAVE_STEPS}")
            modify_script_param(pta_script_path, reg_save, f"--save {iter_ckpt_path}")
            delete_script_param_line(pta_script_path, "--load")
            if str(MODEL_NAME).lower() == "deepseekv3":
                if not apply_deepseekv3_unified_low_memory_profile(pta_script_path):
                    iter_reason = "PTA SAVE统一减配失败"
                    pta_result = "ERROR"
                    finalize_current_iter()
                    continue
            
            # 备份修改后脚本
            pta_save_mod_dir = create_backup_dir("save", i, "pta", "modified")
            shutil.copy2(pta_script_path, pta_save_mod_dir)
            if cluster is not None:
                local_save_script = Path(pta_save_mod_dir) / os.path.basename(pta_script_rel)
                if not apply_multinode_script_settings(
                    local_save_script,
                    local_workers=Config.TARGET_NPUS_PER_NODE,
                    total_workers=Config.TARGET_WORLD_SIZE,
                    nnodes=Config.TARGET_NNODES,
                    node_rank=Config.TARGET_NODE_RANK,
                    master_addr=Config.TARGET_MASTER_ADDR,
                    master_port=Config.TARGET_MASTER_PORT,
                    enable_pta_env=True,
                ):
                    iter_reason = "PTA-SAVE 多机脚本改写失败"
                    pta_result = "ERROR"
                    pta_save_stage_result = "ERROR"
                    finalize_current_iter()
                    continue
            backup_artifact_to_output(
                Path(pta_save_mod_dir) / os.path.basename(pta_script_rel),
                run_persist_dir,
                i,
                "scripts",
                f"pta-save_{os.path.basename(pta_script_rel)}",
            )
            
            # 4. 预配置PTA LOAD参数
            log_step("1.4 预配置PTA LOAD参数")
            reg_load = r"--load[[:space:]]\+[^\ ]\+"
            resume_iteration = read_checkpoint_iteration(iter_ckpt_path)
            if resume_iteration <= 0:
                resume_iteration = int(SAVE_STEPS)
            effective_load_train_iters = int(LOAD_STEPS) + int(resume_iteration)
            msa_load_train_iters = effective_load_train_iters
            log_info(
                "LOAD步数修正 | "
                f"目标执行步数={LOAD_STEPS} | checkpoint_iteration={resume_iteration} | "
                f"脚本写入train-iters={effective_load_train_iters}"
            )
            log_info(
                "MSA LOAD步数修正 | "
                f"目标执行步数={LOAD_STEPS} | checkpoint_iteration={resume_iteration} | "
                f"脚本写入train-iters={msa_load_train_iters}"
            )
            # 恢复原始脚本并修改
            shutil.copy2(f"{pta_save_ori_dir}/{os.path.basename(pta_script_rel)}", pta_script_path)
            modify_script_param(pta_script_path, reg_train_iters, f"--train-iters {effective_load_train_iters}")
            modify_script_param(pta_script_path, reg_load, f"--load {iter_ckpt_path}")
            delete_script_param_line(pta_script_path, "--save")
            if str(MODEL_NAME).lower() == "deepseekv3":
                if not apply_deepseekv3_unified_low_memory_profile(pta_script_path):
                    iter_reason = "PTA LOAD统一减配失败"
                    pta_result = "ERROR"
                    finalize_current_iter()
                    continue
            
            pta_load_mod_dir = create_backup_dir("load", i, "pta", "modified")
            shutil.copy2(pta_script_path, pta_load_mod_dir)
            if cluster is not None:
                local_load_script = Path(pta_load_mod_dir) / os.path.basename(pta_script_rel)
                if not apply_multinode_script_settings(
                    local_load_script,
                    local_workers=Config.TARGET_NPUS_PER_NODE,
                    total_workers=Config.TARGET_WORLD_SIZE,
                    nnodes=Config.TARGET_NNODES,
                    node_rank=Config.TARGET_NODE_RANK,
                    master_addr=Config.TARGET_MASTER_ADDR,
                    master_port=Config.TARGET_MASTER_PORT,
                    enable_pta_env=True,
                ):
                    iter_reason = "PTA-LOAD 多机脚本改写失败"
                    pta_result = "ERROR"
                    pta_load_stage_result = "ERROR"
                    finalize_current_iter()
                    continue
            backup_artifact_to_output(
                Path(pta_load_mod_dir) / os.path.basename(pta_script_rel),
                run_persist_dir,
                i,
                "scripts",
                f"pta-load_{os.path.basename(pta_script_rel)}",
            )
            
            if run_msa:
                # 5. 生成MSA脚本
                log_step("1.5 生成MSA训练脚本")
                log_info(f"MSA脚本转换日志路径: {msa_script_convert_exec_log}")
                msa_script_converted = convert_msa_script(
                    pta_script_rel,
                    msa_script_rel,
                    exec_log_file=msa_script_convert_exec_log,
                )
                backup_runtime_log_to_output(msa_script_convert_exec_log, run_persist_dir, i)
                if not msa_script_converted:
                    log_error("MSA脚本生成失败！跳过本轮")
                    iter_reason = "MSA脚本生成失败"
                    msa_result = "ERROR"
                    msa_load_stage_result = "ERROR"
                    finalize_current_iter()
                    continue

                if not sanitize_swiglu_fusion_script(msa_script_path):
                    iter_reason = "MSA脚本SwiGLU融合兼容化失败"
                    msa_result = "ERROR"
                    msa_load_stage_result = "ERROR"
                    finalize_current_iter()
                    continue
                
                if not check_script_valid(msa_script_path):
                    iter_reason = "MSA脚本校验失败"
                    msa_result = "ERROR"
                    msa_load_stage_result = "ERROR"
                    finalize_current_iter()
                    continue

                if not ensure_external_pretrain_entry(msa_script_path):
                    iter_reason = "MSA外部入口校验失败"
                    msa_result = "ERROR"
                    msa_load_stage_result = "ERROR"
                    finalize_current_iter()
                    continue

                if not sanitize_task1_mutation_runtime_flags(msa_script_path):
                    iter_reason = "MSA脚本运行时参数清理失败"
                    msa_result = "ERROR"
                    msa_load_stage_result = "ERROR"
                    finalize_current_iter()
                    continue

                if not sanitize_moe_expert_bias_aux_loss(msa_script_path):
                    iter_reason = "MSA脚本MoE专家偏置兼容化失败"
                    msa_result = "ERROR"
                    msa_load_stage_result = "ERROR"
                    finalize_current_iter()
                    continue

                if not sanitize_alibi_script(msa_script_path):
                    iter_reason = "MSA脚本ALiBi兼容化失败"
                    msa_result = "ERROR"
                    msa_load_stage_result = "ERROR"
                    finalize_current_iter()
                    continue

                if not sanitize_pangu_script(msa_script_path):
                    iter_reason = "MSA脚本Pangu兼容化失败"
                    msa_result = "ERROR"
                    msa_load_stage_result = "ERROR"
                    finalize_current_iter()
                    continue

                if not sanitize_rotary_base_script(msa_script_path):
                    iter_reason = "MSA脚本rotary-base兼容化失败"
                    msa_result = "ERROR"
                    msa_load_stage_result = "ERROR"
                    finalize_current_iter()
                    continue

                if not apply_script_constraints(msa_script_path):
                    iter_reason = "MSA脚本约束失败"
                    msa_result = "ERROR"
                    msa_load_stage_result = "ERROR"
                    finalize_current_iter()
                    continue

                if not align_bias_linear_flags(pta_script_path, msa_script_path):
                    iter_reason = "MSA脚本bias-linear参数对齐失败"
                    msa_result = "ERROR"
                    msa_load_stage_result = "ERROR"
                    finalize_current_iter()
                    continue
                
                # 预配置MSA LOAD参数
                modify_script_param(msa_script_path, reg_train_iters, f"--train-iters {msa_load_train_iters}")
                modify_script_param(msa_script_path, reg_load, f"--load {iter_ckpt_path}")
                delete_script_param_line(msa_script_path, "--save")
                
                msa_load_mod_dir = create_backup_dir("load", i, "msa", "modified")
                shutil.copy2(msa_script_path, msa_load_mod_dir)
                if cluster is not None:
                    local_msa_script = Path(msa_load_mod_dir) / os.path.basename(msa_script_rel)
                    if not apply_multinode_script_settings(
                        local_msa_script,
                        local_workers=Config.TARGET_NPUS_PER_NODE,
                        total_workers=Config.TARGET_WORLD_SIZE,
                        nnodes=Config.TARGET_NNODES,
                        node_rank=Config.TARGET_NODE_RANK,
                        master_addr=Config.TARGET_MASTER_ADDR,
                        master_port=Config.TARGET_MASTER_PORT,
                    ):
                        iter_reason = "MSA-LOAD 多机脚本改写失败"
                        msa_result = "ERROR"
                        msa_load_stage_result = "ERROR"
                        finalize_current_iter()
                        continue
                backup_artifact_to_output(
                    Path(msa_load_mod_dir) / os.path.basename(msa_script_rel),
                    run_persist_dir,
                    i,
                    "scripts",
                    f"msa-load_{os.path.basename(msa_script_rel)}",
                )

            if run_mf:
                log_step("1.6 生成MF训练脚本")
                log_info(f"MF脚本生成日志路径: {mf_script_gen_exec_log}")
                mf_script_generated = generate_mf_script(
                    pta_script_rel,
                    mf_script_rel,
                    MODEL_NAME,
                    LOAD_STEPS,
                    exec_log_file=mf_script_gen_exec_log,
                )
                backup_runtime_log_to_output(mf_script_gen_exec_log, run_persist_dir, i)
                if not mf_script_generated or not check_script_valid(mf_script_path):
                    log_error("MF脚本生成失败！跳过本轮")
                    iter_reason = "MF脚本生成失败"
                    mf_result = "ERROR"
                    mf_stage_result = "ERROR"
                    finalize_current_iter()
                    continue
                backup_artifact_to_output(mf_script_path, run_persist_dir, i, "scripts")
            
            log_step("完成阶段 1/2: 脚本生成与归档")
            
            # 6. 执行PTA-SAVE训练
            log_step("2.1 执行PTA-SAVE训练")
            pta_save_script_src = Path(pta_save_mod_dir) / os.path.basename(pta_script_rel)
            shutil.copy2(pta_save_script_src, pta_script_path)
            log_info(f"PTA-SAVE执行日志路径: {pta_save_exec_log}")
            def _local_pta_save():
                return run_training(
                    pta_script_rel,
                    Config.PTA_ENV,
                    1,
                    exec_log_file=pta_save_exec_log,
                    csv_path=pta_csv_log,
                )

            if cluster is not None:
                local_ok, remote_ok, remote_states = _run_cluster_stage(
                    cluster=cluster,
                    session_id=cluster_session_id,
                    stage_name=f"pta_save_iter{i}",
                    runtime_log_dir=iter_runtime_log_dir,
                    local_runner=_local_pta_save,
                    upload_builder=lambda node, node_workers: build_remote_script_upload_items(
                        pta_save_script_src,
                        pta_script_rel,
                        cluster_stage_tmp_dir / "pta_save",
                        local_workers=node_workers,
                        total_workers=Config.TARGET_WORLD_SIZE,
                        nnodes=Config.TARGET_NNODES,
                        node_rank=node.node_rank,
                        master_addr=Config.TARGET_REMOTE_MASTER_ADDR,
                        master_port=Config.TARGET_MASTER_PORT,
                        enable_pta_env=True,
                        dataset_path=Config.DATA_PATH,
                    ),
                    payload_builder=lambda node, node_workers: {
                        "job_type": "task1_run_script",
                        "script_rel": pta_script_rel,
                        "env_type": 1,
                        "csv_rel": pta_csv_log,
                        "cleanup_paths": [
                            _repo_rel_path(iter_ckpt_artifact_path),
                            pta_csv_log,
                        ],
                        "timeout": Config.PTA_MAX_RUNTIME,
                    },
                    timeout_seconds=Config.PTA_MAX_RUNTIME + 300,
                )
                if not remote_ok:
                    log_error(f"PTA-SAVE 存在从机失败: {remote_states}")
                pta_save_ok = local_ok and remote_ok
            else:
                pta_save_ok = _local_pta_save()
            backup_runtime_log_to_output(pta_save_exec_log, run_persist_dir, i)
            if not pta_save_ok:
                log_error("PTA-SAVE执行失败！跳过本轮")
                pta_save_stage_result = "ERROR"
                iter_reason = "PTA-SAVE执行失败"
                pta_result = "ERROR"
                backup_weight_on_pta_msa_failure(
                    iter_ckpt_artifact_path,
                    run_persist_dir,
                    i,
                    "PTA-SAVE执行失败",
                )
                finalize_current_iter()
                continue
            if is_weight_artifact_missing(iter_ckpt_artifact_path):
                log_error(f"PTA-SAVE权重产物不存在或为空: {iter_ckpt_path}")
                log_error("本轮迭代判定为ERROR，直接进入下一轮")
                pta_save_stage_result = "ERROR"
                iter_reason = "PTA-SAVE权重产物不存在或为空"
                pta_result = "ERROR"
                backup_weight_on_pta_msa_failure(
                    iter_ckpt_artifact_path,
                    run_persist_dir,
                    i,
                    "PTA-SAVE权重产物不存在或为空",
                )
                finalize_current_iter()
                continue
            pta_save_stage_result = "OK"
            pta_result = "OK"
            log_info(f"PTA-SAVE训练成功，权重已保存至: {iter_ckpt_path}")
            
            # 7. 执行PTA-LOAD训练
            log_step("2.2 执行PTA-LOAD训练")
            backup_and_clean_log_dir(Config.PTA_LOG_SRC_DIR, "load", i, "pta")
            pta_load_script_src = Path(pta_load_mod_dir) / os.path.basename(pta_script_rel)
            shutil.copy2(pta_load_script_src, pta_script_path)
            time.sleep(10)
            log_info(f"PTA-LOAD执行日志路径: {pta_load_exec_log}")
            def _local_pta_load():
                return run_training(
                    pta_script_rel,
                    Config.PTA_ENV,
                    1,
                    exec_log_file=pta_load_exec_log,
                    csv_path=pta_csv_log,
                )

            if cluster is not None:
                local_ok, remote_ok, remote_states = _run_cluster_stage(
                    cluster=cluster,
                    session_id=cluster_session_id,
                    stage_name=f"pta_load_iter{i}",
                    runtime_log_dir=iter_runtime_log_dir,
                    local_runner=_local_pta_load,
                    upload_builder=lambda node, node_workers: build_remote_script_upload_items(
                        pta_load_script_src,
                        pta_script_rel,
                        cluster_stage_tmp_dir / "pta_load",
                        local_workers=node_workers,
                        total_workers=Config.TARGET_WORLD_SIZE,
                        nnodes=Config.TARGET_NNODES,
                        node_rank=node.node_rank,
                        master_addr=Config.TARGET_REMOTE_MASTER_ADDR,
                        master_port=Config.TARGET_MASTER_PORT,
                        enable_pta_env=True,
                        dataset_path=Config.DATA_PATH,
                    ),
                    payload_builder=lambda node, node_workers: {
                        "job_type": "task1_run_script",
                        "script_rel": pta_script_rel,
                        "env_type": 1,
                        "csv_rel": pta_csv_log,
                        "cleanup_paths": [pta_csv_log],
                        "timeout": Config.PTA_MAX_RUNTIME,
                    },
                    timeout_seconds=Config.PTA_MAX_RUNTIME + 300,
                )
                if not remote_ok:
                    log_error(f"PTA-LOAD 存在从机失败: {remote_states}")
                pta_load_ok = local_ok and remote_ok
            else:
                pta_load_ok = _local_pta_load()
            backup_runtime_log_to_output(pta_load_exec_log, run_persist_dir, i)
            pta_load_ok = should_treat_pta_load_as_success(pta_load_ok, pta_csv_log, i)
            if not pta_load_ok:
                log_error("PTA-LOAD训练失败，跳过本轮")
                pta_load_stage_result = "ERROR"
                iter_reason = "PTA-LOAD执行失败"
                pta_result = "ERROR"
                backup_weight_on_pta_msa_failure(
                    iter_ckpt_artifact_path,
                    run_persist_dir,
                    i,
                    "PTA-LOAD执行失败",
                )
                finalize_current_iter()
                continue
            
            # 处理PTA日志
            log_step("2.3 处理PTA精度日志")
            if not handle_log("PTA", i, pta_acc_log, "load"):
                log_error("PTA日志处理失败，跳过本轮")
                pta_load_stage_result = "ERROR"
                iter_reason = "PTA日志处理失败"
                pta_result = "ERROR"
                backup_weight_on_pta_msa_failure(
                    iter_ckpt_artifact_path,
                    run_persist_dir,
                    i,
                    "PTA日志处理失败",
                )
                finalize_current_iter()
                continue
            pta_load_stage_result = "OK"
            backup_artifact_to_output(pta_acc_log, run_persist_dir, i, "runtime_logs")
            pta_success_count += 1
            log_acc(f"PTA训练成功计数+1 | 当前累计: {pta_success_count}/{TOTAL_ITER}")
            
            if run_msa:
                # 8. 执行MSA-LOAD训练
                log_step("2.4 执行MSA-LOAD训练")
                msa_load_script_src = Path(msa_load_mod_dir) / os.path.basename(msa_script_rel)
                shutil.copy2(msa_load_script_src, msa_script_path)
                Config.MSA_MONITOR_LOGS = [_resolve_last_worker_log_relpath(msa_script_path)]
                log_info(f"MSA结束判定日志切换为: {Config.MSA_MONITOR_LOGS[-1]}")
                msa_profile_dir = run_persist_dir / f"iter_{i}" / "profiler" / "msa-load"
                msa_profile_report_dir = run_persist_dir / f"iter_{i}" / "analysis" / "msa-profiler"
                log_info(f"MSA-LOAD执行日志路径: {msa_load_exec_log}")
                def _local_msa_load():
                    return run_training(
                        msa_script_rel,
                        Config.MS_ENV,
                        2,
                        exec_log_file=msa_load_exec_log,
                        csv_path=msa_csv_log,
                    )

                if cluster is not None:
                    local_ok, remote_ok, remote_states = _run_cluster_stage(
                        cluster=cluster,
                        session_id=cluster_session_id,
                        stage_name=f"msa_load_iter{i}",
                        runtime_log_dir=iter_runtime_log_dir,
                        local_runner=_local_msa_load,
                        upload_builder=lambda node, node_workers: build_remote_script_upload_items(
                            msa_load_script_src,
                            msa_script_rel,
                            cluster_stage_tmp_dir / "msa_load",
                            local_workers=node_workers,
                            total_workers=Config.TARGET_WORLD_SIZE,
                            nnodes=Config.TARGET_NNODES,
                            node_rank=node.node_rank,
                            master_addr=Config.TARGET_REMOTE_MASTER_ADDR,
                            master_port=Config.TARGET_MASTER_PORT,
                            dataset_path=Config.DATA_PATH,
                        ),
                        payload_builder=lambda node, node_workers: {
                            "job_type": "task1_run_script",
                            "script_rel": msa_script_rel,
                            "env_type": 2,
                            "csv_rel": msa_csv_log,
                            "cleanup_paths": [Config.MSA_LOG_DIR, msa_csv_log],
                            "timeout": Config.MSA_MAX_RUNTIME + Config.LOG_INIT_WAIT + Config.LOG_STABLE_THRESHOLD + 300,
                        },
                        collect_builder=lambda node, node_workers: (
                            [{"path": Config.MSA_LOG_DIR, "flatten": True}],
                            Path(run_persist_dir) / f"iter_{i}" / "msrun_log" / f"node_{node.node_rank}",
                        ),
                        timeout_seconds=Config.MSA_MAX_RUNTIME + Config.LOG_INIT_WAIT + Config.LOG_STABLE_THRESHOLD + 600,
                    )
                    if not remote_ok:
                        log_error(f"MSA-LOAD 存在从机失败: {remote_states}")
                    msa_load_ok = local_ok and remote_ok
                else:
                    msa_load_ok = _local_msa_load()
                backup_runtime_log_to_output(msa_load_exec_log, run_persist_dir, i)
                if not msa_load_ok:
                    log_error("MSA启动失败! 跳过本轮")
                    msa_load_stage_result = "ERROR"
                    iter_reason = "MSA-LOAD执行失败"
                    msa_result = "ERROR"
                    backup_weight_on_pta_msa_failure(
                        iter_ckpt_artifact_path,
                        run_persist_dir,
                        i,
                        "MSA-LOAD执行失败",
                    )
                    finalize_current_iter()
                    continue
                msa_result = "OK"

                # 等待MSA完成
                if not wait_msa_finish(i):
                    msa_result = "ERROR"
                    msa_load_stage_result = "ERROR"
                    iter_reason = "MSA日志校验失败或超时"
                    backup_weight_on_pta_msa_failure(
                        iter_ckpt_artifact_path,
                        run_persist_dir,
                        i,
                        "MSA日志校验失败或超时",
                    )
                    finalize_current_iter()
                    continue

                # 处理MSA日志
                log_step("2.5 处理MSA精度日志")
                if not handle_log("MSA", i, msa_acc_log, "load"):
                    log_error("MSA日志处理失败，跳过本轮")
                    msa_load_stage_result = "ERROR"
                    iter_reason = "MSA日志处理失败"
                    msa_result = "ERROR"
                    backup_weight_on_pta_msa_failure(
                        iter_ckpt_artifact_path,
                        run_persist_dir,
                        i,
                        "MSA日志处理失败",
                    )
                    finalize_current_iter()
                    continue
                msa_load_stage_result = "OK"
                backup_artifact_to_output(msa_acc_log, run_persist_dir, i, "runtime_logs")
                if msa_profile_dir.exists() and any(msa_profile_dir.rglob("*")):
                    generate_profile_report(
                        profile_dir=msa_profile_dir,
                        report_dir=msa_profile_report_dir,
                        step_csv_path=msa_csv_log,
                        exec_log_path=msa_load_exec_log,
                        task_label="Task1-MSA",
                        iter_num=i,
                    )
                msa_success_count += 1
                log_acc(f"MSA训练成功计数+1 | 当前累计: {msa_success_count}/{TOTAL_ITER}")

                precision_issue = find_series_loss_mismatch(
                    LMSV_ROOT / pta_csv_log,
                    LMSV_ROOT / msa_csv_log,
                )
                if precision_issue:
                    backup_weight_on_precision_issue(
                        iter_ckpt_artifact_path,
                        run_persist_dir,
                        i,
                        precision_issue,
                    )

            if run_mf:
                log_step("2.6 执行MF权重转换与训练")
                pta_parse_source = f"{pta_load_mod_dir}/{os.path.basename(pta_script_rel)}"
                convert_tp = extract_numeric_param_from_script(
                    pta_parse_source,
                    "--tensor-model-parallel-size",
                    "--tensor-parallel-size",
                ) or Config.TARGET_TENSOR_PARALLEL_SIZE
                convert_pp = extract_numeric_param_from_script(
                    pta_parse_source,
                    "--pipeline-model-parallel-size",
                    "--pipeline-parallel-size",
                ) or Config.TARGET_PIPELINE_PARALLEL_SIZE
                convert_ep = extract_numeric_param_from_script(
                    pta_parse_source,
                    "--expert-model-parallel-size",
                    "--expert-parallel-size",
                ) or Config.TARGET_EXPERT_PARALLEL_SIZE

                def _local_mf_prepare():
                    if Config.ENABLE_MF_WEIGHT_LOAD:
                        log_info(f"MF权重转换日志路径: {mf_convert_exec_log}")
                        if Config.ENABLE_WEIGHT_CONVERT:
                            mf_convert_ok = convert_pta_checkpoint_for_mf(
                                iter_ckpt_path,
                                mf_ckpt_dir,
                                MODEL_NAME,
                                convert_tp,
                                convert_pp,
                                convert_ep,
                                exec_log_file=mf_convert_exec_log,
                            )
                            if not mf_convert_ok:
                                return False
                        else:
                            log_warn("已关闭MF权重转换，默认使用当前mf_ckpt_dir作为load_checkpoint")
                        return update_mf_yaml_load_checkpoint(mf_script_path, mf_ckpt_dir)

                    log_step("2.6.1 跳过MF权重转换与加载（ENABLE_MF_WEIGHT_LOAD=False）")
                    return disable_mf_yaml_load_checkpoint(mf_script_path)

                mf_prepare_ok = True
                if cluster is not None:
                    local_ok, remote_ok, remote_states = _run_cluster_stage(
                        cluster=cluster,
                        session_id=cluster_session_id,
                        stage_name=f"mf_prepare_iter{i}",
                        runtime_log_dir=iter_runtime_log_dir,
                        local_runner=_local_mf_prepare,
                        upload_builder=lambda node, node_workers: build_remote_portable_upload_items(
                            mf_script_path,
                            mf_script_rel,
                            cluster_stage_tmp_dir / "mf_prepare",
                        ),
                        payload_builder=lambda node, node_workers: {
                            "job_type": "task1_mf_prepare",
                            "yaml_rel": mf_script_rel,
                            "ckpt_load_dir": _repo_rel_path(iter_ckpt_artifact_path),
                            "ckpt_save_dir": _repo_rel_path(mf_ckpt_dir),
                            "model_name": MODEL_NAME,
                            "tp": convert_tp,
                            "pp": convert_pp,
                            "ep": convert_ep,
                            "enable_weight_load": Config.ENABLE_MF_WEIGHT_LOAD,
                            "enable_weight_convert": Config.ENABLE_WEIGHT_CONVERT,
                            "timeout": Config.PTA_MAX_RUNTIME + 600,
                        },
                        timeout_seconds=Config.PTA_MAX_RUNTIME + 900,
                    )
                    if not remote_ok:
                        log_error(f"MF预处理存在从机失败: {remote_states}")
                    mf_prepare_ok = local_ok and remote_ok
                else:
                    mf_prepare_ok = _local_mf_prepare()
                if Config.ENABLE_MF_WEIGHT_LOAD and Config.ENABLE_WEIGHT_CONVERT:
                    backup_runtime_log_to_output(mf_convert_exec_log, run_persist_dir, i)
                if not mf_prepare_ok:
                    log_error("MF权重转换或配置更新失败，跳过本轮")
                    mf_stage_result = "ERROR"
                    iter_reason = "MF权重转换或配置更新失败"
                    mf_result = "ERROR"
                    finalize_current_iter()
                    continue

                backup_artifact_to_output(
                    mf_script_path,
                    run_persist_dir,
                    i,
                    "scripts",
                    f"mf-load_{os.path.basename(mf_script_rel)}",
                )

                backup_and_clean_log_dir(Config.MF_LOG_SRC_DIR, "load", i, "mf")
                cleanup_training_processes("MF-PRELAUNCH", i)
                card_num = int(Config.TARGET_NPUS_PER_NODE or get_card_num(mf_script_path))
                log_info(f"MF执行日志路径: {mf_run_exec_log}")
                def _local_mf_train():
                    return run_mf_training(
                        mf_script_path,
                        card_num,
                        exec_log_file=mf_run_exec_log,
                        csv_path=mf_csv_log,
                    )

                if cluster is not None:
                    local_ok, remote_ok, remote_states = _run_cluster_stage(
                        cluster=cluster,
                        session_id=cluster_session_id,
                        stage_name=f"mf_train_iter{i}",
                        runtime_log_dir=iter_runtime_log_dir,
                        local_runner=_local_mf_train,
                        payload_builder=lambda node, node_workers: {
                            "job_type": "task1_mf_train",
                            "yaml_rel": mf_script_rel,
                            "csv_rel": mf_csv_log,
                            "local_workers": node_workers,
                            "total_workers": Config.TARGET_WORLD_SIZE,
                            "master_addr": Config.TARGET_REMOTE_MASTER_ADDR,
                            "master_port": Config.TARGET_MASTER_PORT,
                            "node_rank": node.node_rank,
                            "timeout": Config.MSA_MAX_RUNTIME + Config.LOG_INIT_WAIT + Config.LOG_STABLE_THRESHOLD + 300,
                        },
                        collect_builder=lambda node, node_workers: (
                            [{"path": "msrun_log", "flatten": True}],
                            Path(run_persist_dir) / f"iter_{i}" / "msrun_log" / f"node_{node.node_rank}",
                        ) if max(1, int(node_workers)) > 1 else ([], Path(run_persist_dir) / f"iter_{i}" / "msrun_log" / f"node_{node.node_rank}"),
                        timeout_seconds=Config.MSA_MAX_RUNTIME + Config.LOG_INIT_WAIT + Config.LOG_STABLE_THRESHOLD + 600,
                    )
                    if not remote_ok:
                        log_error(f"MF训练存在从机失败: {remote_states}")
                    mf_run_ok = local_ok and remote_ok
                else:
                    mf_run_ok = _local_mf_train()
                backup_runtime_log_to_output(mf_run_exec_log, run_persist_dir, i)
                if not mf_run_ok:
                    log_error("MF启动失败，跳过本轮")
                    mf_stage_result = "ERROR"
                    iter_reason = "MF启动失败"
                    mf_result = "ERROR"
                    finalize_current_iter()
                    continue
                if not wait_mf_finish(
                    mf_csv_log,
                    max_wait=Config.MSA_MAX_RUNTIME,
                    poll_interval=10,
                    exec_log_file=mf_run_exec_log,
                ):
                    mf_stage_result = "ERROR"
                    iter_reason = "MF日志校验失败或超时"
                    mf_result = "ERROR"
                    finalize_current_iter()
                    continue
                if not handle_log("MF", i, mf_acc_log, "load"):
                    log_error("MF日志处理失败，跳过本轮")
                    mf_stage_result = "ERROR"
                    iter_reason = "MF日志处理失败"
                    mf_result = "ERROR"
                    finalize_current_iter()
                    continue
                backup_artifact_to_output(mf_acc_log, run_persist_dir, i, "runtime_logs")
                mf_precision_issue = find_series_loss_mismatch(
                    LMSV_ROOT / pta_csv_log,
                    LMSV_ROOT / mf_csv_log,
                    tolerance=Config.MF_LOSS_TOLERANCE,
                )
                if mf_precision_issue:
                    backup_weight_on_precision_issue(
                        iter_ckpt_artifact_path,
                        run_persist_dir,
                        i,
                        f"PTA-MF精度异常(>{Config.MF_LOSS_TOLERANCE}): {mf_precision_issue}",
                    )
                mf_result = "OK"
                mf_stage_result = "OK"
                mf_success_count += 1
            
            # 9. 迭代收尾
            analyse_stage_result = "OK"
            finalize_current_iter()
            log_step(f"完成迭代 {i}/{TOTAL_ITER}")
            
        except Exception as e:
            log_error(f"迭代{i}执行异常: {str(e)}")
            pta_result = "ERROR"
            msa_result = "ERROR"
            iter_reason = f"执行异常: {str(e)}"
            backup_weight_on_pta_msa_failure(
                iter_ckpt_artifact_path,
                run_persist_dir,
                i,
                f"执行异常: {str(e)}",
            )
            finalize_current_iter()
            continue
        finally:
            refresh_iteration_analysis()
    
    # 统计结果
    log_step("任务执行完成，开始汇总结果")
    pta_rate = (pta_success_count * 100 // TOTAL_ITER) if TOTAL_ITER > 0 else 0
    msa_rate = (msa_success_count * 100 // TOTAL_ITER) if TOTAL_ITER > 0 else 0
    mf_rate = (mf_success_count * 100 // TOTAL_ITER) if TOTAL_ITER > 0 else 0
    log_acc(f"PTA训练成功次数: {pta_success_count}/{TOTAL_ITER} | 成功率: {pta_rate}%")
    if run_msa:
        log_acc(f"MSA训练成功次数: {msa_success_count}/{TOTAL_ITER} | 成功率: {msa_rate}%")
    if run_mf:
        log_acc(f"MF训练成功次数: {mf_success_count}/{TOTAL_ITER} | 成功率: {mf_rate}%")

    log_step("开始自动分析实验结果")
    try:
        from utils.analyze.task1_result import analyze_task1_run

        analysis = analyze_task1_run(
            output_root=Path(Config.MY_PERSIST_ROOT).resolve(),
            run_dir=run_persist_dir,
            model_name=MODEL_NAME,
            planned_iterations=TOTAL_ITER,
        )
        log_info(
            "实验结果分析完成 | 执行轮次: "
            f"{analysis.executed_iterations} | 变异成功: "
            f"{analysis.mutation_success_count}/{analysis.executed_iterations} "
            f"({analysis.mutation_success_rate * 100:.2f}%)"
        )
        log_info(
            "问题统计 | 功能: "
            f"{analysis.functional_failures} | 精度: {analysis.precision_failures} | "
            f"性能: {analysis.performance_failures} | 显存: {analysis.memory_failures}"
        )
        log_info(f"分析目录: {analysis.analysis_dir}")
        log_info(f"HTML报告: {analysis.report_html}")
        log_info(f"JSON汇总: {analysis.summary_json}")
        log_info(f"失败复现目录: {analysis.repro_root}")
    except Exception as exc:
        log_warn(f"自动分析失败，已跳过: {exc}")

    log_step(f"=============== 自动化变异+PTA/{msa_or_mf}训练流程 结束 ===============")
    log_kv("概览", "结束时间", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    log_step("任务结束")
    log_step(f"=============== 自动化变异+PTA/{msa_or_mf}训练流程 结束 ===============")
    if cluster is not None and cluster_session_id:
        cluster.cleanup_session(cluster_session_id)
    
    return 0


# ====================== 兼容旧接口 ======================
def dump_logs_wrapper(iter_num, run_persist_dir):
    """兼容旧接口"""
    dump_logs(iter_num, run_persist_dir)
