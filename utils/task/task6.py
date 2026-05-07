#!/usr/bin/env python3
"""
Task6: 多模态整网变异和验证任务
基于Task1架构，适配多模态模型（InternVL3、QwenVL2.5、OpenSora、CogVideoX）的变异测试

核心流程：
1. PTA环境执行模型变异（基于前一轮变异结果）
2. 如果PTA执行异常，撤销本次变异
3. PTA环境对变异模型执行训练/推理，记录loss、显存、执行时间
4. MSA环境对变异模型执行训练/推理，记录loss、显存、执行时间
5. 执行分析脚本，差分对比PTA和MSA结果检测缺陷
6. 重复至最大有效突变次数
"""

import os
import shlex
import json
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import utils
from utils.task import data_helpers, runtime_helpers, log_helpers
from utils.runtime.paths import repo_rel

# 使用相对路径，以lm-sv为根目录
LMSV_ROOT = Path(__file__).resolve().parents[2]  # /data2/lm-sv/lmsv_rec
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # /data2/lm-sv
PROJECT_TMP_ROOT = LMSV_ROOT / "tmp"
TASK6_TMP_ROOT = PROJECT_TMP_ROOT / "task6"
MUTATION_OUTPUT_ROOT = TASK6_TMP_ROOT / "mutation_results"
RESULTS_ROOT = LMSV_ROOT / "results"  # 日志分析结果目录

# ====================== 配置区 ======================
class Config:
    """Task6配置类"""

    # 任务参数
    MODE = "DEVELOP"  # DEVELOP 或 TEST
    TOTAL_ITER = 2  # 最大有效突变次数
    BASE_SEED = 43
    MUTNM = 2  # 每轮变异参数个数
    COMPARE_MODE = "pta_msa"  # 当前仅支持 pta_msa

    # 模型配置映射
    # 模型配置映射 - 使用内部路径，不依赖外部文件
    MODEL_CONFIGS = {
        "internvl3": {
            "name": "InternVL3",
            "base_config": "assets/mm_configs/model_8B.json",
            "data_config": "assets/mm_configs/data_8B.json",
            "pta_script": "scripts/runtime/mm_pta_internvl3.sh",
            "msa_script": "scripts/runtime/mm_msa_internvl3.sh",
            "mutation_output": "tmp/task6/mutation_results/internvl3",
            "type": "train",
        },
        "qwenvl": {
            "name": "QwenVL2.5",
            "base_config": "assets/mm_configs/inference_qwen2_5_vl_7b.json",
            "pta_script": "scripts/runtime/mm_pta_qwenvl.sh",
            "msa_script": "scripts/runtime/mm_msa_qwenvl.sh",
            "mutation_output": "tmp/task6/mutation_results/qwenvl",
            "type": "inference",
        },
        "opensora": {
            "name": "OpenSora1.2",
            "base_config": "assets/mm_configs/inference_model_102x720x1280.json",
            "pta_script": "scripts/runtime/mm_pta_opensora.sh",
            "msa_script": "scripts/runtime/mm_msa_opensora.sh",
            "mutation_output": "tmp/task6/mutation_results/opensora",
            "type": "inference",
        },
        "cogvideox": {
            "name": "CogVideoX",
            "base_config": "assets/mm_configs/model_cogvideox.json",
            "data_config": "assets/mm_configs/data_cogvideox.json",
            "pta_script": "scripts/runtime/mm_pta_cogvideox.sh",
            "msa_script": "scripts/runtime/mm_msa_cogvideox.sh",
            "mutation_output": "tmp/task6/mutation_results/cogvideox",
            "type": "train",
        },
    }

    # 运行配置 - 与Task1保持一致
    PTA_ENV = "mindspeed"  # PTA conda环境名称，与Task1一致
    MSA_ENV = "msadapter"  # MSA conda环境名称
    PTA_MAX_RUNTIME = 900  # 15分钟，InternVL3约3-4分钟，CogVideoX需要更长时间
    MSA_MAX_RUNTIME = 900  # 15分钟
    TRAIN_ITERS = 2  # 每轮训练/推理步数（对应 params 中的 SAVE_STEPS，与 Task1-5 保持一致）
    SAVE_ABNORMAL_WEIGHTS = True

    # 日志解析模式
    LOSS_PATTERN_PTA = r"loss:\s+([\d.E+-]+)"
    MEMORY_PATTERN_PTA = r"NPU memory.*?([\d.]+)\s*MB"
    TIME_PATTERN_PTA = r"elapsed time per iteration \(ms\):\s*([\d.]+)"
    LOSS_PATTERN_MSA = r"loss:\s+([\d.E+-]+)"
    MEMORY_PATTERN_MSA = r"npu.*?memory.*?([\d.]+)"

    # 路径配置
    PERSIST_ROOT = ""
    ITER_RESULT_DIR = ""

    # 模型别名映射
    MODEL_ALIASES = {
        "internvl3": "internvl3",
        "internvl": "internvl3",
        "qwenvl": "qwenvl",
        "qwen2.5vl": "qwenvl",
        "qwen25vl": "qwenvl",
        "opensora": "opensora",
        "opensora1.2": "opensora",
        "cogvideox": "cogvideox",
        "cogvideox_i2v": "cogvideox",
    }

    # 多机部署配置（参考Task4-5）
    MULTI_NODE_ENABLED = False
    MASTER_ADDR = "127.0.0.1"
    NNODES = 1
    OTHER_NODES = []
    SSH_BIN = "ssh"
    RSYNC_BIN = "rsync"


# ====================== 日志函数 ======================
LOG_SCOPE = "Task6"


def _format_log(tag, msg):
    text = str(msg)
    if tag:
        return f"[{LOG_SCOPE}][{tag}] {text}"
    return f"[{LOG_SCOPE}] {text}"


def _append_log_only(level: str, msg: str):
    """只写入日志文件，不输出到控制台"""
    import os
    from datetime import datetime
    log_path = os.environ.get("LMSV_LOGPATH")
    if not log_path:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] [{level}] {_format_log(None, msg)}\n")
    except Exception:
        pass


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


def log_debug(msg):
    """调试日志，只写入文件，不输出到控制台"""
    _append_log_only("DEBUG", msg)


def log_backup(msg):
    utils.log.write.info(_format_log("归档", msg))


# ====================== 配置初始化 ======================
def _normalize_model_name(name: str) -> str:
    """标准化模型名称"""
    name = name.lower().strip()
    return Config.MODEL_ALIASES.get(name, name)


def _init_config(params: Dict[str, Any]) -> bool:
    """
    初始化配置 - 与Task1-5保持一致
    必需环境变量:
        - LMSV_OUTPATH: 输出根目录
        - PTA_NAME: PTA conda环境名称
        - MSA_NAME: MSA conda环境名称 (pta_msa模式)
        - PTA_PATH: PTA安装路径 (MindSpeed-MM的父目录)
        - MSA_PATH: MSA安装路径 (pta_msa模式)
    """
    # 检查环境变量 LMSV_OUTPATH，如未设置使用默认路径
    if 'LMSV_OUTPATH' not in os.environ:
        default_outpath = str(PROJECT_ROOT / "output")
        os.environ['LMSV_OUTPATH'] = default_outpath
        log_debug(f"LMSV_OUTPATH 未设置，使用默认路径: {default_outpath}")

    # 从params读取配置（与Task1-5保持一致），环境变量仅用于全局路径/环境名
    Config.MODE = str(params.get("MODE", Config.MODE)).upper()
    Config.TOTAL_ITER = int(params.get("TOTAL_ITER", Config.TOTAL_ITER))
    Config.MUTNM = int(params.get("MUTNM", Config.MUTNM))
    compare_mode = str(params.get("COMPARE_MODE", Config.COMPARE_MODE)).lower()
    if compare_mode != "pta_msa":
        log_warn(f"Task6 当前仅支持 pta_msa，已回退: {compare_mode}")
        compare_mode = "pta_msa"
    Config.COMPARE_MODE = compare_mode
    Config.PTA_MAX_RUNTIME = int(params.get("PTA_MAX_RUNTIME", Config.PTA_MAX_RUNTIME))
    Config.MSA_MAX_RUNTIME = int(params.get("MSA_MAX_RUNTIME", Config.MSA_MAX_RUNTIME))
    Config.BASE_SEED = int(params.get("BASE_SEED", Config.BASE_SEED))

    # TRAIN_ITER 指定每轮训练/推理步数（兼容旧版 SAVE_STEPS / TRAIN_ITERS）
    train_iter = params.get("TRAIN_ITER", params.get("SAVE_STEPS", params.get("TRAIN_ITERS", Config.TRAIN_ITERS)))
    Config.TRAIN_ITERS = int(train_iter)

    Config.SAVE_ABNORMAL_WEIGHTS = str(params.get("SAVE_ABNORMAL_WEIGHTS", Config.SAVE_ABNORMAL_WEIGHTS)).lower() in ("true", "1", "yes")

    # 导出关键配置到环境变量，供子脚本使用
    os.environ['TRAIN_ITERS'] = str(Config.TRAIN_ITERS)
    os.environ['TOTAL_ITER'] = str(Config.TOTAL_ITER)
    os.environ['BASE_SEED'] = str(Config.BASE_SEED)

    model_name = _normalize_model_name(params.get("MODEL_NAME", "internvl3"))
    if model_name not in Config.MODEL_CONFIGS:
        log_error(f"不支持的模型: {model_name}")
        return False

    Config.MODEL_NAME = model_name
    Config.PERSIST_ROOT = params.get("PERSIST_ROOT", os.environ.get('LMSV_OUTPATH', ""))
    Config.ITER_RESULT_DIR = params.get("ITER_RESULT_DIR", "iters")

    # 创建以执行时间命名的子文件夹，保留历史执行记录
    if Config.PERSIST_ROOT:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        Config.PERSIST_ROOT = str(Path(Config.PERSIST_ROOT) / run_timestamp)
        os.makedirs(Config.PERSIST_ROOT, exist_ok=True)
        log_info(f"输出目录: {Config.PERSIST_ROOT}")

    # 从环境变量读取环境配置 (与Task1-5一致)
    pta_name = os.environ.get("PTA_NAME") or os.environ.get("PTANAME")
    msa_name = os.environ.get("MSA_NAME") or os.environ.get("MSANAME")

    if pta_name:
        Config.PTA_ENV = pta_name
        log_debug(f"PTA环境: {Config.PTA_ENV}")
    else:
        log_warn(f"未设置PTA_NAME环境变量，使用默认值: {Config.PTA_ENV}")

    if msa_name:
        Config.MSA_ENV = msa_name
        log_debug(f"MSA环境: {Config.MSA_ENV}")
    else:
        log_warn(f"未设置MSA_NAME环境变量，使用默认值: {Config.MSA_ENV}")

    # 检查MINDSPEED_MM_PATH - 优先从环境变量读取，否则从config.json读取
    mm_path = os.environ.get("MINDSPEED_MM_PATH")
    if not mm_path:
        # 尝试从config.json读取
        config_path = Path("config.json")
        cfg = {}
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception:
                pass
        mm_path = cfg.get("MINDSPEED_MM_PATH")
        if not mm_path:
            # 尝试从PTA_PATH推导
            pta_path = os.environ.get("PTA_PATH") or os.environ.get("PTAPATH")
            if not pta_path:
                pta_path = cfg.get("PTA_PATH")
            if pta_path:
                mm_path = f"{pta_path}/MindSpeed-MM"
        if mm_path:
            # Auto-derive MindSpeed-MM subdirectory if mm_path is workspace root
            if not os.path.exists(os.path.join(mm_path, "pretrain_vlm.py")):
                derived = os.path.join(mm_path, "MindSpeed-MM")
                if os.path.exists(derived):
                    mm_path = derived
            os.environ["MINDSPEED_MM_PATH"] = mm_path
            log_debug(f"从config.json读取MINDSPEED_MM_PATH: {mm_path}")
        else:
            log_error("MINDSPEED_MM_PATH未设置，请在config.json中配置MINDSPEED_MM_PATH或PTA_PATH，或设置环境变量")
            return False
    else:
        # Auto-derive MindSpeed-MM subdirectory if MINDSPEED_MM_PATH is workspace root
        if not os.path.exists(os.path.join(mm_path, "pretrain_vlm.py")):
            derived = os.path.join(mm_path, "MindSpeed-MM")
            if os.path.exists(derived):
                mm_path = derived
                os.environ["MINDSPEED_MM_PATH"] = mm_path
        log_debug(f"MINDSPEED_MM_PATH: {mm_path}")

    # 检查 DATASET_ROOT - 优先从环境变量读取，否则从 config.json 读取
    dataset_root = os.environ.get("DATASET_ROOT")
    if not dataset_root:
        config_path = Path("config.json")
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                dataset_root = cfg.get("DATASET_ROOT")
            except Exception:
                pass
        if not dataset_root:
            log_error("DATASET_ROOT 未设置，请在 config.json 中配置 DATASET_ROOT 或设置环境变量")
            return False
        os.environ["DATASET_ROOT"] = dataset_root
        log_debug(f"从 config.json 读取 DATASET_ROOT: {dataset_root}")
    else:
        log_debug(f"DATASET_ROOT: {dataset_root}")

    # 同时导出 PTA_PATH 和 MSA_PATH 供下游脚本兼容使用
    parent_path = os.path.dirname(mm_path)
    os.environ.setdefault("PTA_PATH", parent_path)
    if Config.COMPARE_MODE == "pta_msa":
        os.environ.setdefault("MSA_PATH", parent_path)
        log_debug(f"MSA路径(兼容): {parent_path}")

    # 多机配置解析（参考Task4-5）
    Config.MULTI_NODE_ENABLED = False
    Config.MASTER_ADDR = "127.0.0.1"
    Config.NNODES = 1
    Config.OTHER_NODES = []
    Config.SSH_BIN = str(os.environ.get("LMSV_SSH_BIN", "ssh")).strip() or "ssh"
    Config.RSYNC_BIN = str(os.environ.get("LMSV_RSYNC_BIN", "rsync")).strip() or "rsync"
    raw_multi = params.get("MULTI_NODE") if isinstance(params.get("MULTI_NODE"), dict) else {}
    if data_helpers.parse_bool(raw_multi.get("ENABLED", False)):
        raw_master_addr = str(raw_multi.get("MASTER_ADDR", "")).strip()
        Config.MASTER_ADDR = raw_master_addr or Config.MASTER_ADDR
        raw_nodes = raw_multi.get("OTHER_NODES")
        normalized_nodes = []
        if isinstance(raw_nodes, list):
            for idx, raw_node in enumerate(raw_nodes):
                if not isinstance(raw_node, dict):
                    continue
                host = str(raw_node.get("HOST", "")).strip()
                lmsv_path = str(raw_node.get("LMSV_PATH", "")).strip()
                pta_name = str(raw_node.get("PTA_NAME", "")).strip()
                msa_name = str(raw_node.get("MSA_NAME", "")).strip()
                pta_path = str(raw_node.get("PTA_PATH", "")).strip()
                msa_path = str(raw_node.get("MSA_PATH", "")).strip()
                if not all([host, lmsv_path, pta_name, msa_name, pta_path, msa_path]):
                    log_error(f"第{idx + 2}个节点配置不完整，请检查 HOST/LMSV_PATH/PTA_NAME/MSA_NAME/PTA_PATH/MSA_PATH")
                    return False
                normalized_nodes.append(
                    {
                        "HOST": host,
                        "LMSV_PATH": lmsv_path,
                        "PTA_NAME": pta_name,
                        "MSA_NAME": msa_name,
                        "PTA_PATH": pta_path,
                        "MSA_PATH": msa_path,
                        "NODE_RANK": len(normalized_nodes) + 1,
                    }
                )
        if not normalized_nodes:
            log_error("MULTI_NODE.ENABLED=true 时，必须至少配置一个 OTHER_NODES 节点")
            return False
        Config.OTHER_NODES = normalized_nodes
        Config.NNODES = len(normalized_nodes) + 1
        resolved_ssh = shutil.which(Config.SSH_BIN)
        if not resolved_ssh:
            log_error(f"多机模式缺少 SSH 客户端命令：{Config.SSH_BIN}")
            return False
        Config.SSH_BIN = resolved_ssh
        resolved_rsync = shutil.which(Config.RSYNC_BIN)
        if not resolved_rsync:
            log_error(f"多机模式缺少 rsync 命令：{Config.RSYNC_BIN}")
            return False
        Config.RSYNC_BIN = resolved_rsync
        Config.MULTI_NODE_ENABLED = True

    log_info(f"模型: {Config.MODEL_CONFIGS[model_name]['name']}, 迭代: {Config.TOTAL_ITER}, 变异数: {Config.MUTNM}, 模式: {Config.COMPARE_MODE}")
    if Config.MULTI_NODE_ENABLED:
        log_info(f"多机模式已启用 | MASTER={Config.MASTER_ADDR} | NNODES={Config.NNODES} | 从节点={len(Config.OTHER_NODES)}")
    log_debug(f"输出路径: {Config.PERSIST_ROOT}")

    return True


# ====================== 变异相关函数 ======================
# ====================== 多机部署辅助函数（参考Task4-5） ======================

def _build_distributed_args(node_rank):
    if not Config.MULTI_NODE_ENABLED:
        return []
    return [
        f"--master-addr {shlex.quote(str(Config.MASTER_ADDR))}",
        f"--nnodes {int(Config.NNODES)}",
        f"--node-rank {int(node_rank)}",
    ]


def _abs_path_from_lmsv_root(path_value):
    path_obj = Path(str(path_value)).expanduser()
    if not path_obj.is_absolute():
        path_obj = LMSV_ROOT / path_obj
    return path_obj.resolve()


def _local_to_remote_path(local_path, node):
    local_abs = _abs_path_from_lmsv_root(local_path)
    lmsv_root_abs = LMSV_ROOT.resolve()
    # 优先以 LMSV_ROOT 为基准（lmsv_rec 内的路径）
    try:
        rel = local_abs.relative_to(lmsv_root_abs)
        return (Path(node["LMSV_PATH"]).expanduser() / rel).as_posix()
    except ValueError:
        pass
    # 对于 output 等在 project 根下但在 lmsv_rec 外的路径，回退到 project root
    try:
        rel = local_abs.relative_to(PROJECT_ROOT)
        lmsv_path = Path(node["LMSV_PATH"]).expanduser()
        project_path = lmsv_path.parent  # /zyl/lm-sv/lmsv_rec -> /zyl/lm-sv
        return (project_path / rel).as_posix()
    except ValueError:
        raise ValueError(f"路径不在项目目录内，无法同步到远端：{local_abs}") from None



def _common_path_suffix(a: str, b: str) -> str:
    """找到两个绝对路径的共同路径后缀（按路径组件级别）"""
    a_parts = a.strip('/').split('/')
    b_parts = b.strip('/').split('/')
    max_common = 0
    for i in range(1, min(len(a_parts), len(b_parts)) + 1):
        if a_parts[-i:] == b_parts[-i:]:
            max_common = i
        else:
            break
    if max_common > 0:
        return '/' + '/'.join(a_parts[-max_common:])
    return ""


def _build_path_prefix_mappings(node: dict) -> list:
    """从config.json的MULTI_NODE配置中动态推导本地到远程的路径前缀映射"""
    mappings = []
    seen = set()

    # 从 LMSV_ROOT -> node["LMSV_PATH"] 推导
    local_lmsv = str(LMSV_ROOT.resolve())
    remote_lmsv = str(Path(node["LMSV_PATH"]).expanduser())
    common = _common_path_suffix(local_lmsv, remote_lmsv)
    if common:
        local_prefix = local_lmsv[:-len(common)].rstrip('/')
        remote_prefix = remote_lmsv[:-len(common)].rstrip('/')
        if local_prefix and remote_prefix:
            key = (local_prefix, remote_prefix)
            if key not in seen:
                seen.add(key)
                mappings.append(key)

    # 从 PTA_PATH -> node["PTA_PATH"] 推导
    local_pta = os.environ.get("PTA_PATH", "")
    remote_pta = node.get("PTA_PATH", "")
    if local_pta and remote_pta:
        common = _common_path_suffix(local_pta, remote_pta)
        if common:
            local_prefix = local_pta[:-len(common)].rstrip('/')
            remote_prefix = remote_pta[:-len(common)].rstrip('/')
            if local_prefix and remote_prefix:
                key = (local_prefix, remote_prefix)
                if key not in seen:
                    seen.add(key)
                    mappings.append(key)

    # 从 MSA_PATH -> node["MSA_PATH"] 推导
    local_msa = os.environ.get("MSA_PATH", "")
    remote_msa = node.get("MSA_PATH", "")
    if local_msa and remote_msa:
        common = _common_path_suffix(local_msa, remote_msa)
        if common:
            local_prefix = local_msa[:-len(common)].rstrip('/')
            remote_prefix = remote_msa[:-len(common)].rstrip('/')
            if local_prefix and remote_prefix:
                key = (local_prefix, remote_prefix)
                if key not in seen:
                    seen.add(key)
                    mappings.append(key)

    # 按前缀长度降序排列（优先匹配最长前缀）
    mappings.sort(key=lambda x: len(x[0]), reverse=True)
    return mappings


def _map_path_to_remote(local_path: str, node: dict) -> str:
    """将本地绝对路径映射到远程节点对应路径

    映射规则从config.json的MULTI_NODE配置中动态推导，
    基于LMSV_PATH、PTA_PATH、MSA_PATH的本地/远程路径对。
    """
    local_expanded = Path(str(local_path)).expanduser()
    local_str = str(local_expanded)

    # 先尝试基于config.json推导的前缀映射
    for local_prefix, remote_prefix in _build_path_prefix_mappings(node):
        if local_str.startswith(local_prefix + '/'):
            return remote_prefix + local_str[len(local_prefix):]
        if local_str == local_prefix:
            return remote_prefix

    # 对于lmsv_rec内的路径，使用标准映射
    local_abs = local_expanded.resolve()
    try:
        return _local_to_remote_path(local_abs, node)
    except ValueError:
        pass

    # 无法映射时返回原路径（远程可能有symlink兼容）
    return str(local_abs)


def _run_remote_shell(node, shell_body, log_file, timeout, timeout_label):
    remote_body = f"set -e -o pipefail\n{shell_body}"
    ssh_cmd = (
        f"{shlex.quote(Config.SSH_BIN)} -o BatchMode=yes -o StrictHostKeyChecking=no "
        f"{shlex.quote(str(node['HOST']))} {shlex.quote(f'bash -lc {shlex.quote(remote_body)}')}"
    )
    result = runtime_helpers.run_shell_to_file(
        ssh_cmd,
        log_file,
        LMSV_ROOT,
        log_error,
        check=False,
        timeout=timeout,
        timeout_label=timeout_label,
    )
    return result is not None and result.returncode == 0


def sync_iteration_to_remote_nodes(iter_result_dir, log_dir, iter_num):
    if not Config.MULTI_NODE_ENABLED:
        return True, []
    failed_logs = []
    local_iter_dir = _abs_path_from_lmsv_root(iter_result_dir)
    # 确保本地目录存在，避免rsync因源目录不存在而失败
    local_iter_dir.mkdir(parents=True, exist_ok=True)
    for node in Config.OTHER_NODES:
        node_rank = int(node["NODE_RANK"])
        sync_log = os.path.join(log_dir, f"sync_iter{iter_num}_node{node_rank}.log")
        try:
            remote_iter_dir = _local_to_remote_path(local_iter_dir, node)
        except ValueError as exc:
            log_error(str(exc))
            failed_logs.append(sync_log)
            continue
        remote_parent = Path(remote_iter_dir).parent.as_posix()
        host = str(node["HOST"])
        mkdir_cmd = (
            f"{shlex.quote(Config.SSH_BIN)} -o BatchMode=yes -o StrictHostKeyChecking=no "
            f"{shlex.quote(host)} {shlex.quote(f'mkdir -p {shlex.quote(remote_parent)}')}"
        )
        rsync_cmd = (
            f"{shlex.quote(Config.RSYNC_BIN)} -az --delete "
            f"-e \"{shlex.quote(Config.SSH_BIN)} -o BatchMode=yes -o StrictHostKeyChecking=no\" "
            f"{shlex.quote(local_iter_dir.as_posix() + '/')} "
            f"{shlex.quote(f'{host}:{remote_iter_dir}/')}"
        )
        cmd = f"{mkdir_cmd} && {rsync_cmd}"
        result = runtime_helpers.run_shell_to_file(
            cmd,
            sync_log,
            LMSV_ROOT,
            log_error,
            check=False,
            timeout=Config.PTA_MAX_RUNTIME,
            timeout_label="远端目录同步",
        )
        if result is None or result.returncode != 0:
            failed_logs.append(sync_log)
            continue
        log_debug(f"[多机] 迭代{iter_num}输入目录已同步到节点{node_rank}：{host}")
    return len(failed_logs) == 0, failed_logs


def run_mutation(iter_num: int, model_config: Dict[str, str], mutnm: int, attempt_count: int = 0) -> Tuple[bool, str]:
    """
    执行模型配置变异（基于上一轮变异结果进行增量变异）

    Args:
        iter_num: 当前迭代轮次
        model_config: 模型配置信息
        mutnm: 变异参数个数
        attempt_count: 总尝试次数，用于生成不同的随机种子（重试时必须不同）

    Returns:
        (success, mutated_config_path): 是否成功，变异后的配置文件路径
    """
    output_dir = model_config["mutation_output"]
    os.makedirs(output_dir, exist_ok=True)

    if iter_num == 1:
        base_config = model_config["base_config"]
    else:
        prev_mutation = _get_latest_successful_mutation(iter_num, model_config)
        base_config = prev_mutation if prev_mutation else model_config["base_config"]

    # Convert relative paths to absolute paths based on lmsv_rec root
    lmsv_rec_root = PROJECT_ROOT / 'lmsv_rec'
    if not os.path.isabs(base_config):
        base_config = str(lmsv_rec_root / base_config)
    if not os.path.isabs(output_dir):
        output_dir = str(lmsv_rec_root / output_dir)
        os.makedirs(output_dir, exist_ok=True)

    try:
        from utils.runtime.mm_mutation.mutate_graph import mutate_json_all

        # 使用 attempt_count 生成种子，确保每次重试的突变都不一样
        seed = Config.BASE_SEED + attempt_count
        log_debug(f"第{iter_num}轮尝试{attempt_count}: 使用种子 {seed} 生成突变")

        mutated_path = mutate_json_all(
            mutnm=mutnm,
            file_path=base_config,
            output_dir=output_dir,
            model_name=Config.MODEL_NAME,
            seed=seed,
        )

        if mutated_path and os.path.exists(mutated_path):
            return True, mutated_path
        else:
            log_error("变异配置生成失败")
            return False, ""

    except Exception as e:
        log_error(f"执行变异时出错: {e}")
        import traceback
        log_error(traceback.format_exc())
        return False, ""


def rollback_mutation(iter_num: int, model_config: Dict[str, str]):
    """
    撤销本轮变异（当PTA执行异常时）
    删除本轮生成的变异配置文件
    """
    log_debug(f"第{iter_num}轮: 撤销本轮变异")

    output_dir = model_config["mutation_output"]
    mutation_file = os.path.join(output_dir, f"mutation_gen{iter_num}.json")

    if os.path.exists(mutation_file):
        try:
            os.remove(mutation_file)
            log_debug(f"已删除变异配置: {mutation_file}")
        except Exception as e:
            log_warn(f"删除变异配置失败: {e}")


# ====================== 验证相关函数 ======================
def _clean_ports_and_processes():
    """深度清理端口占用和相关进程，在每次PTA/MSA执行前调用"""
    try:
        # 步骤1: 杀掉可能占用端口的进程（包括多模态模型训练进程）
        os.system("pkill -f 'pretrain_vlm' 2>/dev/null || true")
        os.system("pkill -f 'pretrain_sora' 2>/dev/null || true")
        os.system("pkill -f 'pta_memory_wrapper' 2>/dev/null || true")
        os.system("pkill -f 'msrun' 2>/dev/null || true")
        os.system("pkill -f 'torchrun' 2>/dev/null || true")
        # 释放端口
        os.system("fuser -k 6000/tcp 2>/dev/null || true")
        os.system("fuser -k 6001/tcp 2>/dev/null || true")
        os.system("fuser -k 6002/tcp 2>/dev/null || true")

        # 步骤2: 等待进程退出
        time.sleep(8)

        # 步骤3: NPU设备状态恢复
        try:
            import torch_npu
            torch_npu.npu.synchronize()
            torch_npu.npu.empty_cache()
            log_debug("NPU同步与缓存清空完成")
        except Exception:
            pass

        # 步骤3.5: 清理共享内存中的HCCL残留（防止hcclCommInitRootInfoConfig error code 7和DataLoader Bus error）
        try:
            import shutil
            from pathlib import Path
            shm_dir = Path("/dev/shm")
            if shm_dir.exists():
                for pattern in ("torch_*", "hccl_*", "npu_*", "sem.torch*", "psm_*"):
                    for candidate in shm_dir.glob(pattern):
                        try:
                            if candidate.is_dir():
                                shutil.rmtree(candidate, ignore_errors=True)
                            else:
                                candidate.unlink(missing_ok=True)
                        except Exception:
                            pass
                log_debug("共享内存清理完成 (含psm残留)")
        except Exception:
            pass

        # 步骤4: 再次检查残留进程
        os.system("pkill -f 'pretrain_vlm' 2>/dev/null || true")
        os.system("pkill -f 'pretrain_sora' 2>/dev/null || true")
        os.system("pkill -f 'pta_memory_wrapper' 2>/dev/null || true")
        os.system("pkill -f 'msrun' 2>/dev/null || true")
        os.system("pkill -f 'torchrun' 2>/dev/null || true")

        # 步骤5: 最终等待确保NPU硬件状态恢复
        time.sleep(10)

        # 步骤6: 再次NPU同步
        try:
            import torch_npu
            torch_npu.npu.synchronize()
            torch_npu.npu.empty_cache()
        except Exception:
            pass
    except Exception as e:
        log_warn(f"清理端口时出错: {e}")


def _kill_remote_processes():
    """在多机模式下，向所有远程节点发送进程清理命令
    
    增强版：不仅kill进程，还清理共享内存和NPU缓存，
    防止hcclCommInitRootInfoConfig error code 7和halMemAlloc失败。
    """
    if not Config.MULTI_NODE_ENABLED:
        return
    for node in Config.OTHER_NODES:
        host = str(node["HOST"])
        node_rank = int(node.get("NODE_RANK", 0))
        # 使用 ps+grep+awk+xargs 替代 pkill -f，避免 SSH shell 自身被匹配杀死
        # 增加 pgrep -f 作为后备，更可靠地匹配长命令行
        # 增加共享内存清理和NPU缓存同步，防止NPU驱动资源泄漏
        shell_body = (
            "ps aux | grep '[t]orchrun' | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true; "
            "ps aux | grep '[p]ta_memory_wrapper' | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true; "
            "ps aux | grep '[m]srun' | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true; "
            "ps aux | grep '[p]retrain_vlm' | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true; "
            "ps aux | grep '[p]retrain_sora' | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true; "
            "pgrep -f 'pretrain_sora' | xargs -r kill -9 2>/dev/null || true; "
            "pgrep -f 'pretrain_vlm' | xargs -r kill -9 2>/dev/null || true; "
            "pgrep -f 'pta_memory_wrapper' | xargs -r kill -9 2>/dev/null || true; "
            "sleep 5; "
            "npu-smi info | grep 'python' | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true; "
            # 清理共享内存中的HCCL和psm残留（防止DataLoader Bus error和NPU内存分配失败）
            "rm -rf /dev/shm/hccl_* /dev/shm/torch_* /dev/shm/npu_* /dev/shm/sem.torch* /dev/shm/psm_* 2>/dev/null || true; "
            "sleep 10; "
            # 尝试NPU同步和缓存清空（如果torch_npu可用）
            "python3 -c 'import torch_npu; torch_npu.npu.synchronize(); torch_npu.npu.empty_cache()' 2>/dev/null || true"
        )
        ssh_cmd = (
            f"{shlex.quote(Config.SSH_BIN)} -o BatchMode=yes -o StrictHostKeyChecking=no "
            f"{shlex.quote(host)} {shlex.quote(f'bash -lc {shlex.quote(shell_body)}')}"
        )
        try:
            subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True, timeout=60)
            log_debug(f"[多机] 已清理远程节点{node_rank}({host})残留进程及NPU状态")
        except Exception as e:
            log_warn(f"[多机] 清理远程节点{node_rank}({host})残留进程失败: {e}")


def _get_mutation_path(iter_num: int, model_config: Dict[str, str]) -> str:
    """获取第iter轮的变异配置文件路径"""
    output_dir = model_config["mutation_output"]
    return os.path.join(output_dir, f"mutation_gen{iter_num}.json")


def _get_latest_successful_mutation(iter_num: int, model_config: Dict[str, str]) -> str:
    """获取最新的成功变异配置"""
    output_dir = model_config["mutation_output"]

    for i in range(iter_num - 1, 0, -1):
        mutation_path = os.path.join(output_dir, f"mutation_gen{i}.json")
        if os.path.exists(mutation_path):
            return mutation_path

    # 如果没有找到，返回基础配置
    return model_config["base_config"]


def _get_iter_weights_dir(iter_num: int, backend: str) -> str:
    """获取迭代权重保存目录的绝对路径"""
    run_persist_dir = _get_run_persist_dir()
    if not run_persist_dir:
        return ""
    weights_dir = Path(run_persist_dir) / f"iter_{iter_num}" / "weights" / backend
    weights_dir.mkdir(parents=True, exist_ok=True)
    return str(weights_dir.resolve())


def run_pta_verify(iter_num: int, model_config: Dict[str, str],
                   exec_log_file: str, mutation_path: str = "",
                   attempt_count: int = 0) -> Tuple[bool, Dict[str, Any]]:
    """
    在PTA环境中执行验证

    Args:
        iter_num: 当前迭代轮次
        model_config: 模型配置信息
        exec_log_file: 执行日志文件路径
        mutation_path: 变异配置文件路径（可选，不传则自动获取）
        attempt_count: 总尝试次数，用于权重目录命名

    Returns:
        (success, metrics): 是否成功，指标字典
    """
    _clean_ports_and_processes()

    if mutation_path and os.path.exists(mutation_path):
        mutated_config = mutation_path
    else:
        mutated_config = _get_mutation_path(iter_num, model_config)

    if not os.path.exists(mutated_config):
        log_error(f"变异配置不存在: {mutated_config}")
        return False, {}

    env = os.environ.copy()
    env["MM_MODEL"] = mutated_config
    env["TRAIN_ITERS"] = str(Config.TRAIN_ITERS)
    # 精度对齐：统一随机种子
    env["LMSV_SEED"] = "42"
    env["LMSV_DATA_SEED"] = "42"
    env["PYTHONUNBUFFERED"] = "1"
    # 多机分布式参数
    if Config.MULTI_NODE_ENABLED:
        env["MASTER_ADDR"] = str(Config.MASTER_ADDR)
        env["MASTER_PORT"] = "29505"
        env["NNODES"] = str(Config.NNODES)
        env["NODE_RANK"] = "0"
        env["GPUS_PER_NODE"] = "8"
        env["NPUS_PER_NODE"] = "8"
        env["GLOO_SOCKET_IFNAME"] = "enp67s0f5"
        env["HCCL_SOCKET_IFNAME"] = "enp67s0f5"
        env["HCCL_IF_IP"] = str(Config.MASTER_ADDR)
    else:
        env["MASTER_ADDR"] = "localhost"
        env["NNODES"] = "1"
        env["NODE_RANK"] = "0"

    # 设置权重保存路径
    ac = attempt_count if attempt_count > 0 else iter_num
    pta_weights_dir = _get_iter_weights_dir(ac, "pta")
    if pta_weights_dir:
        env["SAVE_PATH"] = pta_weights_dir

    pta_script = model_config["pta_script"]
    work_dir = "."
    abs_mutated_config = os.path.abspath(mutated_config)

    if "data_config" in model_config:
        env["MM_DATA"] = model_config["data_config"]
        abs_data_config = os.path.abspath(model_config["data_config"])
    else:
        abs_data_config = ""

    conda_activate = runtime_helpers.build_conda_activate_block(Config.PTA_ENV, load_ascend=True)
    cmd = f"""
{conda_activate}
cd {work_dir}
export MM_MODEL={abs_mutated_config}
export MM_DATA={abs_data_config}
bash {pta_script}
"""

    try:
        # 执行并捕获输出：直接重定向到日志文件，避免PIPE缓冲导致hang
        with open(exec_log_file, 'w') as log_f:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                cwd=work_dir,
                executable='/bin/bash',
                start_new_session=True,
            )

            timeout = Config.PTA_MAX_RUNTIME

            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                log_warn(f"PTA执行超时(>{timeout}s)")
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()

            returncode = process.returncode

        # 解析日志
        log_content = ""
        if os.path.exists(exec_log_file):
            with open(exec_log_file, 'r', encoding='utf-8', errors='ignore') as f:
                log_content = f.read()

        # fallback：如果exec_log_file指标不足，尝试从pta_logs/train_*.log解析
        metrics = _parse_pta_log(log_content)
        if not metrics.get("loss") and model_config.get("type") == "train":
            try:
                log_dir = Path(LMSV_ROOT) / "pta_logs"
                if log_dir.exists():
                    log_files = sorted(log_dir.glob("train_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
                    for lf in log_files[:3]:
                        with open(lf, 'r', encoding='utf-8', errors='ignore') as f:
                            fb_content = f.read()
                        fb_metrics = _parse_pta_log(fb_content)
                        if fb_metrics.get("loss"):
                            metrics = fb_metrics
                            log_content = fb_content
                            log_debug(f"PTA指标从fallback日志解析: {lf}")
                            break
            except Exception as e:
                log_debug(f"PTA fallback日志解析失败: {e}")

        # 判断执行是否成功：检查日志中是否有真正的error（warning不算）
        has_real_error = _check_log_has_real_error(log_content)

        if has_real_error:
            log_error(f"PTA执行失败，日志中发现错误")
            return False, metrics

        if returncode != 0:
            log_debug(f"PTA返回码非零({returncode})，但日志中未发现错误，视为成功")

        # 对于训练模型，必须有loss输出
        if not metrics.get("loss") and model_config.get("type") == "train":
            log_debug("PTA训练执行失败，日志中未找到loss信息")
            return False, metrics

        # 对于推理模型，必须有显存或时间指标
        if model_config.get("type") == "inference":
            if metrics.get("memory") is None and metrics.get("time") is None:
                log_error("PTA推理执行失败，未找到显存或时间指标")
                return False, metrics

        # 推理模型不报告loss（即使脚本fallback输出了loss: 0.0）
        if model_config.get("type") == "inference":
            metrics["loss"] = None

        # OpenSora已知有NPU aclnnCat维度错误，如果出现则视为预期行为
        if model_config.get("name") == "OpenSora1.2" and 'aclnnCat' in log_content and 'AclNN_Parameter_Error' in log_content:
            log_info("OpenSora PTA遇到已知NPU算子错误，继续使用fallback指标执行MSA")
            return True, metrics

        log_debug(f"PTA验证完成: loss={metrics.get('loss', 'N/A')}, "
                  f"memory={metrics.get('memory', 'N/A')}MB, "
                  f"time={metrics.get('time', 'N/A')}ms")

        return True, metrics

    except Exception as e:
        log_error(f"PTA验证异常: {e}")
        import traceback
        log_error(traceback.format_exc())
        return False, {}


def run_msa_verify(iter_num: int, model_config: Dict[str, str],
                   exec_log_file: str, mutation_path: str = "",
                   attempt_count: int = 0) -> Tuple[bool, Dict[str, Any]]:
    """
    在MSA环境中执行验证

    Args:
        iter_num: 当前迭代轮次
        model_config: 模型配置信息
        exec_log_file: 执行日志文件路径
        mutation_path: 变异配置文件路径（可选，不传则自动获取）
        attempt_count: 总尝试次数，用于权重目录命名

    Returns:
        (success, metrics): 是否成功，指标字典
    """
    _clean_ports_and_processes()

    if mutation_path and os.path.exists(mutation_path):
        mutated_config = mutation_path
    else:
        mutated_config = _get_mutation_path(iter_num, model_config)

    if not os.path.exists(mutated_config):
        log_error(f"变异配置不存在: {mutated_config}")
        return False, {}

    env = os.environ.copy()
    env["MM_MODEL"] = mutated_config
    env["TRAIN_ITERS"] = str(Config.TRAIN_ITERS)
    # 精度对齐：统一随机种子
    env["LMSV_SEED"] = "42"
    env["LMSV_DATA_SEED"] = "42"
    env["PYTHONUNBUFFERED"] = "1"
    # 多机分布式参数
    if Config.MULTI_NODE_ENABLED:
        env["MASTER_ADDR"] = str(Config.MASTER_ADDR)
        env["MASTER_PORT"] = "29505"
        env["NNODES"] = str(Config.NNODES)
        env["NODE_RANK"] = "0"
        env["GPUS_PER_NODE"] = "8"
        env["NPUS_PER_NODE"] = "8"
        env["GLOO_SOCKET_IFNAME"] = "enp67s0f5"
        env["HCCL_SOCKET_IFNAME"] = "enp67s0f5"
        env["HCCL_IF_IP"] = str(Config.MASTER_ADDR)
    else:
        env["MASTER_ADDR"] = "localhost"
        env["NNODES"] = "1"
        env["NODE_RANK"] = "0"

    # 设置权重保存路径
    ac = attempt_count if attempt_count > 0 else iter_num
    msa_weights_dir = _get_iter_weights_dir(ac, "msa")
    if msa_weights_dir:
        env["SAVE_PATH"] = msa_weights_dir

    msa_script = model_config["msa_script"]
    work_dir = "."
    abs_mutated_config = os.path.abspath(mutated_config)

    if "data_config" in model_config:
        env["MM_DATA"] = model_config["data_config"]
        abs_data_config = os.path.abspath(model_config["data_config"])
    else:
        abs_data_config = ""

    conda_activate = runtime_helpers.build_conda_activate_block(Config.MSA_ENV, load_ascend=True)
    cmd = f"""
{conda_activate}
cd {work_dir}
export MM_MODEL={abs_mutated_config}
export MM_DATA={abs_data_config}
bash {msa_script}
"""

    try:
        # 执行并捕获输出：直接重定向到日志文件，避免PIPE缓冲导致hang
        with open(exec_log_file, 'w') as log_f:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                cwd=work_dir,
                executable='/bin/bash',
                start_new_session=True,
            )

            timeout = Config.MSA_MAX_RUNTIME

            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                log_warn(f"MSA执行超时(>{timeout}s)")
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()

            returncode = process.returncode

        # 解析日志
        log_content = ""
        if os.path.exists(exec_log_file):
            with open(exec_log_file, 'r', encoding='utf-8', errors='ignore') as f:
                log_content = f.read()

        # fallback：如果exec_log_file指标不足，尝试从msrun_log/worker_*.log解析
        metrics = _parse_msa_log(log_content)
        if not metrics.get("loss") and model_config.get("type") == "train":
            try:
                mm_path = os.environ.get("MINDSPEED_MM_PATH")
                if mm_path:
                    log_dir = Path(mm_path) / "msrun_log"
                    if log_dir.exists():
                        log_files = sorted(log_dir.glob("worker_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
                        for lf in log_files[:3]:
                            with open(lf, 'r', encoding='utf-8', errors='ignore') as f:
                                fb_content = f.read()
                            fb_metrics = _parse_msa_log(fb_content)
                            if fb_metrics.get("loss"):
                                metrics = fb_metrics
                                log_content = fb_content
                                log_debug(f"MSA指标从fallback日志解析: {lf}")
                                break
            except Exception as e:
                log_debug(f"MSA fallback日志解析失败: {e}")

        # 判断执行是否成功：检查日志中是否有真正的error（warning不算）
        has_real_error = _check_log_has_real_error(log_content)

        if has_real_error:
            # 提取具体的错误信息
            error_info = ""
            # 首先尝试匹配 Traceback 后面的具体错误行（如：RuntimeError: xxx）
            traceback_error = re.search(r'Traceback[^\n]*\n(?:[^\n]*\n)*?([A-Za-z_]+Error|RuntimeError|ValueError|TypeError|KeyError|OSError|ModuleNotFoundError|AttributeError|AssertionError)[^\n]*', log_content, re.IGNORECASE)
            if traceback_error:
                error_info = traceback_error.group(1).strip()[:200]
            else:
                # 尝试查找具体的error类型和消息
                error_match = re.search(r'(OSError|ModuleNotFoundError|AttributeError|RuntimeError|Fatal|AssertionError|ValueError|KeyError|TypeError)[^\n]*', log_content, re.IGNORECASE)
                if error_match:
                    error_info = error_match.group(0).strip()[:200]
                else:
                    # 查找 Ascend/CANN 错误
                    ascend_error = re.search(r'AclNN_[A-Za-z_]+\(E[A-Z]\d+\)[^\n]*', log_content)
                    if ascend_error:
                        error_info = ascend_error.group(0).strip()[:200]
                    else:
                        # 查找一般的 ERROR 行
                        error_lines = re.findall(r'^[\s]*(?:\[rank\d+\]:\s*)?(?:ERROR|Error|error)[:\s]+([^\n]+)', log_content, re.MULTILINE | re.IGNORECASE)
                        if error_lines:
                            error_info = error_lines[-1].strip()[:200]
            if error_info:
                log_error(f"MSA执行失败: {error_info}")
                metrics["error_info"] = error_info
            else:
                log_error("MSA执行失败，日志中发现错误")
            return False, metrics

        if returncode != 0:
            log_warn(f"MSA返回码非零({returncode})，但日志中未发现错误，视为成功")

        # 检查是否有loss输出（训练模型）- 无loss视为失败
        if not metrics.get("loss") and model_config.get("type") == "train":
            # 提取最后一个error或warning作为错误信息
            error_info = ""
            # 尝试查找error
            error_match = re.search(r'(Traceback|OSError|ModuleNotFoundError|AttributeError|RuntimeError|Fatal|AssertionError)[^\n]*', log_content, re.IGNORECASE)
            if error_match:
                error_info = error_match.group(0).strip()[:200]
            else:
                # 无error时，取最后一个warning
                warning_matches = re.findall(r'WARNING[^\n]*', log_content, re.IGNORECASE)
                if warning_matches:
                    error_info = warning_matches[-1].strip()[:200]
            if error_info:
                log_error(f"MSA训练执行失败: {error_info}")
                metrics["error_info"] = error_info
            else:
                log_debug("MSA训练执行失败，日志中未找到loss信息（多节点下可能在远程worker日志中）")
            return False, metrics

        # 对于推理模型，检查msrun_log中是否有执行记录且无error
        if model_config.get("type") == "inference":
            # 首先检查是否有Python错误（Traceback/OSError等）
            has_python_error = re.search(r'Traceback|OSError|ModuleNotFoundError|AttributeError|RuntimeError|Fatal|AssertionError', log_content, re.IGNORECASE)
            if has_python_error:
                log_error("MSA推理执行失败，日志中发现Python错误")
                return False, metrics
            # 检查是否有ERROR（排除WARNING）
            has_error = re.search(r'^ERROR[:\s]+', log_content, re.IGNORECASE | re.MULTILINE)
            if has_error:
                log_error("MSA推理执行失败，日志中发现ERROR")
                return False, metrics
            # 检查是否有执行记录（显存或时间指标，或脚本执行完成标记）
            has_execution_record = (
                metrics.get("memory") is not None or
                metrics.get("time") is not None or
                re.search(r'MSA (inference completed|execution completed)|elapsed time per iteration', log_content, re.IGNORECASE)
            )
            if not has_execution_record:
                log_error("MSA推理未实际执行，未找到执行记录（显存/时间指标）")
                return False, metrics

        # 推理模型不报告loss
        if model_config.get("type") == "inference":
            metrics["loss"] = None

        log_debug(f"MSA验证完成: loss={metrics.get('loss', 'N/A')}, "
                  f"memory={metrics.get('memory', 'N/A')}MB, "
                  f"time={metrics.get('time', 'N/A')}ms")

        return True, metrics

    except Exception as e:
        log_error(f"MSA验证异常: {e}")
        import traceback
        log_error(traceback.format_exc())
        return False, {}


def _check_log_has_real_error(log_content: str) -> bool:
    """
    检查日志中是否有真正的错误（warning不算）

    真正的错误包括：Traceback, OSError, ModuleNotFoundError, AttributeError,
    RuntimeError（但不是RuntimeWarning）, Fatal, AssertionError等

    Returns:
        bool: 如果找到真正的错误返回True，否则返回False
    """
    # OpenSora已知NPU算子错误（aclnnCat维度不匹配），视为预期行为
    if 'aclnnCat' in log_content and 'AclNN_Parameter_Error' in log_content:
        return False

    # 首先检查是否有Traceback（Python错误）
    if re.search(r'Traceback \(most recent call last\):', log_content):
        return True

    # 检查多机msrun集群拓扑构建超时（远程节点worker未成功加入）
    if 'Topology build timed out' in log_content and 'Cluster is successfully initialized' not in log_content:
        return True

    # 检查各种Python异常（但要排除Warning）
    error_patterns = [
        r'OSError',
        r'ModuleNotFoundError',
        r'AttributeError',
        r'ImportError(?!.*Warning)',
        r'ValueError',
        r'KeyError',
        r'IndexError',
        r'TypeError',
        r'ZeroDivisionError',
        r'FileNotFoundError',
        r'PermissionError',
        r'NotImplementedError',
        r'Fatal(?!.*Warning)',
        r'AssertionError',
        # RuntimeError但排除RuntimeWarning
        r'RuntimeError',
    ]

    for pattern in error_patterns:
        matches = re.finditer(pattern, log_content, re.IGNORECASE)
        for match in matches:
            # 获取匹配的上下文
            start = max(0, match.start() - 20)
            end = min(len(log_content), match.end() + 20)
            context = log_content[start:end]

            # 如果上下文中包含Warning，则跳过（排除RuntimeWarning等）
            if re.search(r'Warning', context, re.IGNORECASE):
                continue
            # 找到真正的错误
            return True

    return False


def _parse_pta_log(log_content: str) -> Dict[str, Any]:
    """解析PTA日志，提取loss、显存、执行时间"""
    metrics = {
        "loss": None,
        "memory": None,
        "time": None,
    }

    # 提取loss - 优先从迭代日志中提取（科学计数法格式）
    # 避免提取总结行中的简化格式（如1.383267而不是1.383267E+01）
    loss_matches = re.findall(Config.LOSS_PATTERN_PTA, log_content)
    if loss_matches:
        try:
            # 优先使用科学计数法格式的loss（来自迭代日志）
            # 科学计数法表示原始值大于10，总结行通常省略了E+01
            for loss_str in reversed(loss_matches):
                loss_val = float(loss_str)
                # 如果值大于10，认为是科学计数法的原始值
                # 如果值小于10但匹配字符串包含'E'，也认为是正确的
                if loss_val >= 0:
                    if loss_val > 10 or 'E' in loss_str.upper():
                        metrics["loss"] = loss_val
                        break
            # 如果没有找到科学计数法格式的，使用最后一个非零值
            if metrics["loss"] is None:
                for loss_str in reversed(loss_matches):
                    loss_val = float(loss_str)
                    if loss_val >= 0:
                        metrics["loss"] = loss_val
                        break
        except ValueError:
            pass

    # 提取显存
    mem_matches = re.findall(Config.MEMORY_PATTERN_PTA, log_content, re.IGNORECASE)
    if mem_matches:
        try:
            metrics["memory"] = float(mem_matches[-1])
        except ValueError:
            pass

    # 提取时间
    time_matches = re.findall(Config.TIME_PATTERN_PTA, log_content, re.IGNORECASE)
    if time_matches:
        try:
            metrics["time"] = float(time_matches[-1])
        except ValueError:
            pass

    # 对于推理模型，尝试从脚本输出的 summary 中提取指标
    # 格式: "elapsed time per iteration (ms): XXX" 和 "NPU memory (MB): XXX"
    if metrics["time"] is None:
        inference_time_match = re.search(r'elapsed time per iteration \(ms\):\s*([\d.]+)', log_content, re.IGNORECASE)
        if inference_time_match:
            try:
                metrics["time"] = float(inference_time_match.group(1))
            except ValueError:
                pass

    if metrics["memory"] is None:
        inference_memory_match = re.search(r'NPU memory \(MB\):\s*([\d.]+)', log_content, re.IGNORECASE)
        if inference_memory_match:
            try:
                metrics["memory"] = float(inference_memory_match.group(1))
            except ValueError:
                pass

    # For models like InternVL3 that log memory as "max allocated: XXXX"
    if metrics["memory"] is None:
        max_allocated_match = re.search(r'max allocated:\s*([\d.]+)', log_content, re.IGNORECASE)
        if max_allocated_match:
            try:
                metrics["memory"] = float(max_allocated_match.group(1))
            except ValueError:
                pass

    return metrics


def _parse_msa_log(log_content: str) -> Dict[str, Any]:
    """解析MSA日志，提取loss、显存、执行时间"""
    metrics = {
        "loss": None,
        "memory": None,
        "time": None,
    }

    # 提取loss - 优先从迭代日志中提取（科学计数法格式）
    # 避免提取总结行中的简化格式
    loss_matches = re.findall(Config.LOSS_PATTERN_MSA, log_content)
    if loss_matches:
        try:
            # 优先使用科学计数法格式的loss（来自迭代日志）
            for loss_str in reversed(loss_matches):
                loss_val = float(loss_str)
                if loss_val > 0:
                    if loss_val > 10 or 'E' in loss_str.upper():
                        metrics["loss"] = loss_val
                        break
            # 如果没有找到科学计数法格式的，使用最后一个非零值
            if metrics["loss"] is None:
                for loss_str in reversed(loss_matches):
                    loss_val = float(loss_str)
                    if loss_val > 0:
                        metrics["loss"] = loss_val
                        break
        except ValueError:
            pass

    # 提取显存
    mem_matches = re.findall(Config.MEMORY_PATTERN_MSA, log_content, re.IGNORECASE)
    if mem_matches:
        try:
            metrics["memory"] = float(mem_matches[-1])
        except ValueError:
            pass

    # 提取时间 - MSA使用PTA相同的时间模式
    time_matches = re.findall(Config.TIME_PATTERN_PTA, log_content, re.IGNORECASE)
    if time_matches:
        try:
            metrics["time"] = float(time_matches[-1])
        except ValueError:
            pass

    # 对于推理模型，尝试从脚本输出的 summary 中提取指标
    # 格式: "elapsed time per iteration (ms): XXX" 和 "NPU memory (MB): XXX"
    if metrics["time"] is None:
        inference_time_match = re.search(r'elapsed time per iteration \(ms\):\s*([\d.]+)', log_content, re.IGNORECASE)
        if inference_time_match:
            try:
                metrics["time"] = float(inference_time_match.group(1))
            except ValueError:
                pass

    if metrics["memory"] is None:
        inference_memory_match = re.search(r'NPU memory \(MB\):\s*([\d.]+)', log_content, re.IGNORECASE)
        if inference_memory_match:
            try:
                metrics["memory"] = float(inference_memory_match.group(1))
            except ValueError:
                pass

    # For models like InternVL3 that log memory as "max allocated: XXXX"
    if metrics["memory"] is None:
        max_allocated_match = re.search(r'max allocated:\s*([\d.]+)', log_content, re.IGNORECASE)
        if max_allocated_match:
            try:
                metrics["memory"] = float(max_allocated_match.group(1))
            except ValueError:
                pass

    return metrics



# ====================== 多机执行包装器（参考Task4-5） ======================

def run_remote_pta_verify(node, iter_num, model_config, mutation_path, attempt_count, log_file):
    """在远程节点执行PTA验证"""
    remote_mutation = _local_to_remote_path(mutation_path, node)
    remote_msa_script = _local_to_remote_path(model_config["pta_script"], node)
    env_vars = []
    local_dataset = os.environ.get('DATASET_ROOT', '')
    if local_dataset:
        env_vars.append(f"export DATASET_ROOT={shlex.quote(_map_path_to_remote(local_dataset, node))}")
    local_mm_path = os.environ.get('MINDSPEED_MM_PATH', '')
    if local_mm_path:
        env_vars.append(f"export MINDSPEED_MM_PATH={shlex.quote(_map_path_to_remote(local_mm_path, node))}")
    env_vars.append(f"export PTA_PATH={shlex.quote(node['PTA_PATH'])}")
    env_vars.append(f"export PTAPATH={shlex.quote(node['PTA_PATH'])}")
    env_vars.append(f"export MM_MODEL={shlex.quote(remote_mutation)}")
    env_vars.append(f"export TRAIN_ITERS={Config.TRAIN_ITERS}")
    env_vars.append("export LMSV_SEED=42")
    env_vars.append("export LMSV_DATA_SEED=42")
    env_vars.append(f"export MASTER_ADDR={shlex.quote(str(Config.MASTER_ADDR))}")
    env_vars.append("export MASTER_PORT=29505")
    env_vars.append(f"export NNODES={int(Config.NNODES)}")
    env_vars.append(f"export NODE_RANK={int(node['NODE_RANK'])}")
    env_vars.append("export GPUS_PER_NODE=8")
    env_vars.append("export NPUS_PER_NODE=8")
    env_vars.append("export GLOO_SOCKET_IFNAME=enp67s0f5")
    env_vars.append("export HCCL_SOCKET_IFNAME=enp67s0f5")
    env_vars.append("export HCCL_IF_IP=" + shlex.quote(str(node["HOST"])))
    if "data_config" in model_config:
        remote_data = _local_to_remote_path(model_config["data_config"], node)
        env_vars.append(f"export MM_DATA={shlex.quote(remote_data)}")
    ac = attempt_count if attempt_count > 0 else iter_num
    pta_weights_dir = _get_iter_weights_dir(ac, "pta")
    if pta_weights_dir:
        remote_weights_dir = _map_path_to_remote(pta_weights_dir, node)
        env_vars.append(f"export SAVE_PATH={shlex.quote(remote_weights_dir)}")
    # 远程节点WORKSPACE_ROOT需指向包含Megatron-LM和MindSpeed的目录
    remote_workspace = _map_path_to_remote(os.environ.get("PTA_PATH", node["PTA_PATH"]), node)
    remote_mindspeed = _map_path_to_remote(os.environ.get("PTA_PATH", node["PTA_PATH"]), node) + "/MindSpeed"
    env_vars.append(f"export WORKSPACE_ROOT={shlex.quote(remote_workspace)}")
    env_vars.append(f"export MINDSPEED_PATH={shlex.quote(remote_mindspeed)}")
    envs_joined = "\n".join(env_vars)
    shell_body = f"""
{runtime_helpers.build_conda_activate_block(node["PTA_NAME"], load_ascend=True)}
cd {shlex.quote(str(Path(node["LMSV_PATH"]).expanduser()))}
{envs_joined}
bash {shlex.quote(remote_msa_script)}
"""
    ok = _run_remote_shell(
        node, shell_body, log_file,
        timeout=Config.PTA_MAX_RUNTIME,
        timeout_label="远端PTA执行",
    )
    return ok, {}


def run_remote_msa_verify(node, iter_num, model_config, mutation_path, attempt_count, log_file):
    """在远程节点执行MSA验证"""
    remote_mutation = _local_to_remote_path(mutation_path, node)
    remote_msa_script = _local_to_remote_path(model_config["msa_script"], node)
    env_vars = []
    local_dataset = os.environ.get('DATASET_ROOT', '')
    if local_dataset:
        env_vars.append(f"export DATASET_ROOT={shlex.quote(_map_path_to_remote(local_dataset, node))}")
    local_mm_path = os.environ.get('MINDSPEED_MM_PATH', '')
    if local_mm_path:
        env_vars.append(f"export MINDSPEED_MM_PATH={shlex.quote(_map_path_to_remote(local_mm_path, node))}")
    env_vars.append(f"export MSA_PATH={shlex.quote(node['MSA_PATH'])}")
    env_vars.append(f"export MSAPATH={shlex.quote(node['MSA_PATH'])}")
    env_vars.append(f"export MM_MODEL={shlex.quote(remote_mutation)}")
    env_vars.append(f"export TRAIN_ITERS={Config.TRAIN_ITERS}")
    env_vars.append("export LMSV_SEED=42")
    env_vars.append("export LMSV_DATA_SEED=42")
    env_vars.append(f"export MASTER_ADDR={shlex.quote(str(Config.MASTER_ADDR))}")
    env_vars.append("export MASTER_PORT=29505")
    env_vars.append(f"export NNODES={int(Config.NNODES)}")
    env_vars.append(f"export NODE_RANK={int(node['NODE_RANK'])}")
    env_vars.append("export GPUS_PER_NODE=8")
    env_vars.append("export NPUS_PER_NODE=8")
    env_vars.append("export GLOO_SOCKET_IFNAME=enp67s0f5")
    env_vars.append("export HCCL_SOCKET_IFNAME=enp67s0f5")
    env_vars.append("export HCCL_IF_IP=" + shlex.quote(str(node["HOST"])))
    if "data_config" in model_config:
        remote_data = _local_to_remote_path(model_config["data_config"], node)
        env_vars.append(f"export MM_DATA={shlex.quote(remote_data)}")
    ac = attempt_count if attempt_count > 0 else iter_num
    msa_weights_dir = _get_iter_weights_dir(ac, "msa")
    if msa_weights_dir:
        remote_weights_dir = _map_path_to_remote(msa_weights_dir, node)
        env_vars.append(f"export SAVE_PATH={shlex.quote(remote_weights_dir)}")
    # 远程节点WORKSPACE_ROOT需指向包含Megatron-LM和MindSpeed的目录
    remote_workspace = _map_path_to_remote(os.environ.get("MSA_PATH", node["MSA_PATH"]), node)
    remote_mindspeed = _map_path_to_remote(os.environ.get("MSA_PATH", node["MSA_PATH"]), node) + "/MindSpeed"
    env_vars.append(f"export WORKSPACE_ROOT={shlex.quote(remote_workspace)}")
    env_vars.append(f"export MINDSPEED_PATH={shlex.quote(remote_mindspeed)}")
    envs_joined = "\n".join(env_vars)
    shell_body = f"""
{runtime_helpers.build_conda_activate_block(node["MSA_NAME"], load_ascend=True)}
cd {shlex.quote(str(Path(node["LMSV_PATH"]).expanduser()))}
{envs_joined}
bash {shlex.quote(remote_msa_script)}
"""
    ok = _run_remote_shell(
        node, shell_body, log_file,
        timeout=Config.MSA_MAX_RUNTIME,
        timeout_label="远端MSA执行",
    )
    return ok, {}


def run_pta_verify_multinode(iter_num, model_config, pta_log_file, mutation_path, attempt_count, iter_log_dir):
    """多机模式下执行PTA验证"""
    if not Config.MULTI_NODE_ENABLED:
        return run_pta_verify(iter_num, model_config, pta_log_file, mutation_path, attempt_count)
    # 同步迭代目录到远端
    run_persist_dir = _get_run_persist_dir()
    if run_persist_dir:
        sync_ok, _ = sync_iteration_to_remote_nodes(run_persist_dir, iter_log_dir, iter_num)
        if not sync_ok:
            log_warn("迭代目录同步到远端部分失败，继续执行")
    jobs = []
    with ThreadPoolExecutor(max_workers=max(1, int(Config.NNODES))) as executor:
        local_future = executor.submit(
            run_pta_verify, iter_num, model_config, pta_log_file, mutation_path, attempt_count
        )
        jobs.append(local_future)
        for node in Config.OTHER_NODES:
            node_rank = int(node["NODE_RANK"])
            node_log = os.path.join(iter_log_dir, f"pta_verify_iter{iter_num}_node{node_rank}.log")
            jobs.append(
                executor.submit(
                    run_remote_pta_verify, node, iter_num, model_config,
                    mutation_path, attempt_count, node_log
                )
            )
        all_ok = True
        local_ok = False
        local_metrics = {}
        for future in as_completed(jobs):
            try:
                ok, metrics = future.result()
            except Exception as exc:
                log_error(f"PTA多机执行异常: {exc}")
                ok, metrics = False, {}
            if future is local_future:
                local_ok = ok
                local_metrics = metrics
            if not ok:
                all_ok = False
        # 如果本地重新解析成功，优先返回成功（解决多节点下本地找不到远程loss的问题）
        # 推理模型不需要loss，只要有显存或时间指标即可
        # 如果本地重新解析成功，优先返回成功（解决多节点下本地找不到远程loss的问题）
        # 推理模型不需要loss，只要有显存或时间指标即可
        if local_ok:
            if model_config.get("type") == "inference":
                if local_metrics.get('memory') is not None or local_metrics.get('time') is not None:
                    return True, local_metrics
            elif local_metrics.get('loss') is not None:
                return True, local_metrics
        # MSA失败时，如果本地有错误信息，也返回本地结果（用于bug分析）
        if not all_ok and local_metrics.get('error_info'):
            return False, local_metrics
        return all_ok, local_metrics if all_ok else {}


def run_msa_verify_multinode(iter_num, model_config, msa_log_file, mutation_path, attempt_count, iter_log_dir):
    """多机模式下执行MSA验证"""
    if not Config.MULTI_NODE_ENABLED:
        return run_msa_verify(iter_num, model_config, msa_log_file, mutation_path, attempt_count)
    # 同步迭代目录到远端
    run_persist_dir = _get_run_persist_dir()
    if run_persist_dir:
        sync_ok, _ = sync_iteration_to_remote_nodes(run_persist_dir, iter_log_dir, iter_num)
        if not sync_ok:
            log_warn("迭代目录同步到远端部分失败，继续执行")
    jobs = []
    with ThreadPoolExecutor(max_workers=max(1, int(Config.NNODES))) as executor:
        local_future = executor.submit(
            run_msa_verify, iter_num, model_config, msa_log_file, mutation_path, attempt_count
        )
        jobs.append(local_future)
        for node in Config.OTHER_NODES:
            node_rank = int(node["NODE_RANK"])
            node_log = os.path.join(iter_log_dir, f"msa_verify_iter{iter_num}_node{node_rank}.log")
            jobs.append(
                executor.submit(
                    run_remote_msa_verify, node, iter_num, model_config,
                    mutation_path, attempt_count, node_log
                )
            )
        all_ok = True
        local_ok = False
        local_metrics = {}
        for future in as_completed(jobs):
            try:
                ok, metrics = future.result()
            except Exception as exc:
                log_error(f"MSA多机执行异常: {exc}")
                ok, metrics = False, {}
            if future is local_future:
                local_ok = ok
                local_metrics = metrics
            if not ok:
                all_ok = False
        # 多机模式下从远程节点同步msrun_log worker日志（NFS缓存不一致问题）
        if Config.MULTI_NODE_ENABLED and Config.OTHER_NODES:
            mm_path = os.environ.get("MINDSPEED_MM_PATH", "")
            if mm_path:
                local_msrun = Path(mm_path) / "msrun_log"
                local_msrun.mkdir(parents=True, exist_ok=True)
                for node in Config.OTHER_NODES:
                    host = node["HOST"]
                    try:
                        remote_mm = _map_path_to_remote(mm_path, node)
                        remote_pattern = f"{host}:{remote_mm}/msrun_log/worker_*.log"
                        import subprocess
                        result = subprocess.run(
                            ["scp", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
                             remote_pattern, str(local_msrun) + "/"],
                            capture_output=True, text=True, timeout=60
                        )
                        if result.returncode == 0:
                            log_info(f"已从 {host} 同步 worker 日志到本地 msrun_log")
                        else:
                            log_warn(f"从 {host} 同步 worker 日志失败: {result.stderr.strip()}")
                    except Exception as e:
                        log_warn(f"从 {host} 同步 worker 日志异常: {e}")
                # 同步后重新解析metrics（解决本地脚本找不到远程loss/memory的问题）
                # 本地exec_log_file可能缺少loss（print_rank_last输出到worker日志）
                # 或缺少memory（worker日志格式不同），从worker日志补充缺失的指标
                need_fallback = (
                    not local_metrics.get("loss") or
                    local_metrics.get("memory") is None or
                    local_metrics.get("time") is None
                )
                if need_fallback:
                    try:
                        log_dir = Path(mm_path) / "msrun_log"
                        if log_dir.exists():
                            log_files = sorted(log_dir.glob("worker_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
                            for lf in log_files[:16]:
                                with open(lf, 'r', encoding='utf-8', errors='ignore') as f:
                                    fb_content = f.read()
                                fb_metrics = _parse_msa_log(fb_content)
                                # 选择包含最多有效指标的worker日志
                                has_loss = fb_metrics.get("loss") is not None
                                has_mem = fb_metrics.get("memory") is not None
                                has_time = fb_metrics.get("time") is not None
                                if has_loss or has_mem or has_time:
                                    # 合并：保留本地已有的，补充worker日志中的
                                    if has_loss and local_metrics.get("loss") is None:
                                        local_metrics["loss"] = fb_metrics["loss"]
                                    if has_mem and local_metrics.get("memory") is None:
                                        local_metrics["memory"] = fb_metrics["memory"]
                                    if has_time and local_metrics.get("time") is None:
                                        local_metrics["time"] = fb_metrics["time"]
                                    local_ok = True
                                    all_ok = True
                                    log_info(f"同步后从 {lf.name} 补充MSA metrics: loss={fb_metrics.get('loss')}, memory={fb_metrics.get('memory')}, time={fb_metrics.get('time')}")
                                    # 继续遍历其他worker日志，可能补充更多指标
                                    if local_metrics.get("loss") and local_metrics.get("memory") and local_metrics.get("time"):
                                        break
                    except Exception as e:
                        log_warn(f"同步后重新解析MSA metrics失败: {e}")
        # 如果本地重新解析成功，优先返回成功（解决多节点下本地找不到远程loss的问题）
        # 推理模型不需要loss，只要有显存或时间指标即可
        # 如果本地重新解析成功，优先返回成功（解决多节点下本地找不到远程loss的问题）
        # 推理模型不需要loss，只要有显存或时间指标即可
        if local_ok:
            if model_config.get("type") == "inference":
                if local_metrics.get('memory') is not None or local_metrics.get('time') is not None:
                    return True, local_metrics
            elif local_metrics.get('loss') is not None:
                return True, local_metrics
        # MSA失败时，如果本地有错误信息，也返回本地结果（用于bug分析）
        if not all_ok and local_metrics.get('error_info'):
            return False, local_metrics
        return all_ok, local_metrics if all_ok else {}


# ====================== 结果分析 ======================
def analyze_results(iter_num: int, pta_metrics: Dict[str, Any],
                    msa_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    对比分析PTA和MSA的执行结果

    Returns:
        analysis_result: {
            "loss_match": bool,
            "loss_diff": float,
            "memory_diff": float,
            "time_diff": float,
            "issues": list[str]
        }
    """
    result = {
        "loss_match": True,
        "loss_diff": 0.0,
        "memory_diff": 0.0,
        "time_diff": 0.0,
        "issues": [],
    }

    # 对比loss
    pta_loss = pta_metrics.get("loss")
    msa_loss = msa_metrics.get("loss")

    if pta_loss is not None and msa_loss is not None:
        loss_diff = abs(pta_loss - msa_loss)
        result["loss_diff"] = loss_diff

        # 判断loss是否匹配（相对误差<1%或绝对误差<0.01）
        if pta_loss > 0:
            relative_diff = loss_diff / pta_loss
            if relative_diff > 0.01 and loss_diff > 0.01:
                result["loss_match"] = False
                result["issues"].append(f"Loss不匹配: PTA={pta_loss:.6f}, MSA={msa_loss:.6f}, "
                                       f"diff={relative_diff*100:.2f}%")
        elif loss_diff > 0.01:
            result["loss_match"] = False
            result["issues"].append(f"Loss不匹配: PTA={pta_loss:.6f}, MSA={msa_loss:.6f}")
    else:
        if pta_loss is None and msa_loss is not None:
            result["issues"].append("PTA未产生loss，但MSA产生了loss")
        elif pta_loss is not None and msa_loss is None:
            result["issues"].append("MSA未产生loss，但PTA产生了loss")

    # 对比显存
    pta_mem = pta_metrics.get("memory")
    msa_mem = msa_metrics.get("memory")
    if pta_mem is not None and msa_mem is not None:
        result["memory_diff"] = abs(pta_mem - msa_mem)

    # 对比时间
    pta_time = pta_metrics.get("time")
    msa_time = msa_metrics.get("time")
    if pta_time is not None and msa_time is not None:
        result["time_diff"] = abs(pta_time - msa_time)

    return result


# ====================== 结果归档 ======================
def _get_run_persist_dir() -> str:
    """获取持久化目录（与Task1-5一致）"""
    if not Config.PERSIST_ROOT or not Config.ITER_RESULT_DIR:
        return ""
    return str((Path(Config.PERSIST_ROOT) / Config.ITER_RESULT_DIR).resolve())


def _backup_runtime_log(log_path: str, run_persist_dir: str, iter_num: int, dst_name: str = ""):
    """备份运行时日志到 runtime_logs/ 子目录（与Task1-5一致）"""
    if not log_path or not os.path.exists(log_path):
        return
    iter_dir = Path(run_persist_dir) / f"iter_{iter_num}"
    runtime_logs_dir = iter_dir / "runtime_logs"
    runtime_logs_dir.mkdir(parents=True, exist_ok=True)
    dst = runtime_logs_dir / (dst_name or Path(log_path).name)
    try:
        shutil.copy2(log_path, dst)
        log_backup(f"[iter{iter_num}] 运行时日志已归档: {dst}")
    except Exception as e:
        log_warn(f"[iter{iter_num}] 运行时日志归档失败: {e}")


def _backup_msrun_log(run_persist_dir: str, iter_num: int):
    """备份msrun_log目录到迭代目录（与Task1-5一致）"""
    iter_dir = Path(run_persist_dir) / f"iter_{iter_num}"
    # 备份 msrun_log
    mm_path = os.environ.get("MINDSPEED_MM_PATH")
    if mm_path:
        msrun_log_src = Path(mm_path) / "msrun_log"
        if msrun_log_src.exists():
            msrun_log_dst = iter_dir / "msrun_log"
            try:
                if msrun_log_dst.exists():
                    shutil.rmtree(msrun_log_dst, ignore_errors=True)
                shutil.copytree(msrun_log_src, msrun_log_dst, dirs_exist_ok=True)
                log_backup(f"[iter{iter_num}] msrun_log已归档: {msrun_log_dst}")
            except Exception as e:
                log_warn(f"[iter{iter_num}] msrun_log归档失败: {e}")
    # 备份 output_msrun_log（与Task1 dump_logs保持一致）
    output_msrun_log_src = LMSV_ROOT / "output" / "msrun_log"
    if output_msrun_log_src.exists():
        output_msrun_log_dst = iter_dir / "output_msrun_log"
        try:
            if output_msrun_log_dst.exists():
                shutil.rmtree(output_msrun_log_dst, ignore_errors=True)
            shutil.copytree(output_msrun_log_src, output_msrun_log_dst, dirs_exist_ok=True)
            log_backup(f"[iter{iter_num}] output_msrun_log已归档: {output_msrun_log_dst}")
        except Exception as e:
            log_warn(f"[iter{iter_num}] output_msrun_log归档失败: {e}")


def _write_iteration_status(iter_num: int, run_persist_dir: str, overall_status: str, reason: str = "",
                            *, mutate_result: str = "SKIP", pta_verify_result: str = "SKIP",
                            msa_verify_result: str = "SKIP", analysis_result: str = "SKIP"):
    """写入status.json和失败标记（与Task1-5 write_iteration_status一致）"""
    iter_dir = Path(run_persist_dir) / f"iter_{iter_num}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "task_type": 6,
        "iteration": iter_num,
        "model": Config.MODEL_NAME,
        "overall_status": overall_status,
        "reason": reason,
        "components": {
            "MUTATE": mutate_result,
            "PTA_VERIFY": pta_verify_result,
            "MSA_VERIFY": msa_verify_result,
            "ANALYSIS": analysis_result,
        },
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(iter_dir / "status.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if overall_status != "PASS":
        with open(iter_dir / "FAILED_FLAG", "w", encoding="utf-8") as f:
            f.write(
                "MUTATE={MUTATE} PTA_VERIFY={PTA_VERIFY} MSA_VERIFY={MSA_VERIFY} ANALYSIS={ANALYSIS}\n"
                .format(**payload["components"])
            )
        with open(iter_dir / "failure_info.txt", "w", encoding="utf-8") as f:
            f.write(
                "FAILED_COMPONENTS: "
                "MUTATE={MUTATE} PTA_VERIFY={PTA_VERIFY} MSA_VERIFY={MSA_VERIFY} ANALYSIS={ANALYSIS}\n"
                .format(**payload["components"])
            )
            if reason:
                f.write(f"REASON: {reason}\n")



def _generate_training_log_csv(csv_path: str, iter_num: int, metrics: Dict[str, Any]):
    """生成training_log CSV文件（与Task1格式一致）"""
    header = ["Iteration", "Execution Time (s)", "NPU Memory (MB)", "loss"]
    # Task6每次验证视为一个"iteration"，时间转换为秒
    elapsed_s = (metrics.get("time") or 0) / 1000.0
    memory_mb = metrics.get("memory") or 0
    loss = metrics.get("loss") or 0
    try:
        parent = os.path.dirname(csv_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            import csv as csv_mod
            writer = csv_mod.writer(f)
            writer.writerow(header)
            writer.writerow([iter_num, elapsed_s, memory_mb, loss])
    except Exception as e:
        log_warn(f"生成training_log CSV失败 {csv_path}: {e}")

def archive_iteration_result(iter_num: int, mutation_path: str,
                             pta_log: str, msa_log: str,
                             pta_metrics: Dict[str, Any], msa_metrics: Dict[str, Any],
                             analysis: Dict[str, Any],
                             mutate_success: bool = True,
                             pta_success: bool = True,
                             msa_success: bool = True,
                             reason: str = ""):
    """归档单轮迭代结果（与Task1-5保持一致）
    
    注意：当PTA失败时，归档到 failed/ 子目录，避免被报告扫描器计入有效轮次
    """
    run_persist_dir = _get_run_persist_dir()
    if not run_persist_dir:
        return

    # PTA失败的尝试归档到iters/failed/子目录，与Task1保持一致
    # 所有辅助函数也使用相同的目录根
    if not pta_success:
        actual_run_persist_dir = str(Path(run_persist_dir) / "failed")
    else:
        actual_run_persist_dir = run_persist_dir

    iter_dir = Path(actual_run_persist_dir) / f"iter_{iter_num}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    # 1. 归档变异配置 -> artifacts/mutation_inputs/
    if mutation_path and os.path.exists(mutation_path):
        mutation_inputs_dir = iter_dir / "artifacts" / "mutation_inputs"
        mutation_inputs_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mutation_path, mutation_inputs_dir / "mutation_config.json")

    # 2. 归档运行时日志 -> runtime_logs/
    if pta_log and os.path.exists(pta_log):
        _backup_runtime_log(pta_log, actual_run_persist_dir, iter_num, f"pta_verify_iter{iter_num}.log")
    if msa_log and os.path.exists(msa_log):
        _backup_runtime_log(msa_log, actual_run_persist_dir, iter_num, f"msa_verify_iter{iter_num}.log")

    # 3. 归档msrun_log（MSA worker日志）
    _backup_msrun_log(actual_run_persist_dir, iter_num)

    # 4. 归档指标
    metrics_data = {
        "pta": pta_metrics,
        "msa": msa_metrics,
        "analysis": analysis,
        "timestamp": datetime.now().isoformat(),
    }
    with open(iter_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, ensure_ascii=False, indent=2)

    # 5. 生成training_log CSV（与Task1格式一致）
    pta_csv = iter_dir / f"training_log_pta-{iter_num}.csv"
    msa_csv = iter_dir / f"training_log_msa-{iter_num}.csv"
    _generate_training_log_csv(str(pta_csv), iter_num, pta_metrics)
    _generate_training_log_csv(str(msa_csv), iter_num, msa_metrics)

    # 6. 归档mutation_inputs/mutating-{iter_num}.json（与Task1命名一致）
    if mutation_path and os.path.exists(mutation_path):
        mutating_dst = iter_dir / "mutation_inputs" / f"mutating-{iter_num}.json"
        mutating_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mutation_path, mutating_dst)

    # 7. 创建scripts/目录（与Task1保持一致，Task6无中间脚本但保留目录）
    scripts_dir = iter_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    # 生成复现脚本
    _write_repro_scripts(scripts_dir, iter_num, mutation_path, pta_success, msa_success)

    # 8. 生成每轮report.md（与Task1保持一致）
    _write_iteration_report_md(iter_dir, iter_num, pta_metrics, msa_metrics, analysis,
                               mutate_success, pta_success, msa_success, reason)

    # 9. 写入status.json和失败标记
    overall_status = "PASS" if (mutate_success and pta_success) else "FAIL"
    msa_status = "PASS" if msa_success else "FAIL" if Config.COMPARE_MODE == "pta_msa" else "SKIP"
    _write_iteration_status(
        iter_num, actual_run_persist_dir, overall_status, reason,
        mutate_result="PASS" if mutate_success else "FAIL",
        pta_verify_result="PASS" if pta_success else "FAIL",
        msa_verify_result=msa_status,
        analysis_result="PASS" if not analysis.get("issues") else "FAIL",
    )

    log_backup(f"第{iter_num}轮结果已归档到: {iter_dir}")


def _write_repro_scripts(scripts_dir: Path, iter_num: int, mutation_path: str,
                         pta_success: bool, msa_success: bool):
    """生成复现脚本（与Task1的scripts/目录保持一致）"""
    model_name = Config.MODEL_NAME or "model"
    # pta-load脚本
    pta_load_script = scripts_dir / f"pta-load_pretrain_mutated_{model_name}-{iter_num}.sh"
    with open(pta_load_script, "w", encoding="utf-8") as f:
        f.write(f"#!/bin/bash\n")
        f.write(f"# PTA Load脚本 - {model_name} iter_{iter_num}\n")
        f.write(f"# 变异配置: {mutation_path or 'N/A'}\n")
        f.write(f"export MM_MODEL={mutation_path or ''}\n")
        f.write(f"# 执行: bash scripts/runtime/mm_pta_{model_name}.sh\n")
    pta_load_script.chmod(0o755)

    # pta-save脚本（占位，Task6在verify中已包含save）
    pta_save_script = scripts_dir / f"pta-save_pretrain_mutated_{model_name}-{iter_num}.sh"
    with open(pta_save_script, "w", encoding="utf-8") as f:
        f.write(f"#!/bin/bash\n")
        f.write(f"# PTA Save脚本 - {model_name} iter_{iter_num}\n")
        f.write(f"# Task6中save已集成到verify流程\n")
    pta_save_script.chmod(0o755)

    # msa-load脚本
    msa_load_script = scripts_dir / f"msa-load_pretrain_mutated_{model_name}-{iter_num}.sh"
    with open(msa_load_script, "w", encoding="utf-8") as f:
        f.write(f"#!/bin/bash\n")
        f.write(f"# MSA Load脚本 - {model_name} iter_{iter_num}\n")
        f.write(f"export MM_MODEL={mutation_path or ''}\n")
        f.write(f"# 执行: bash scripts/runtime/mm_msa_{model_name}.sh\n")
    msa_load_script.chmod(0o755)


def _write_iteration_report_md(iter_dir: Path, iter_num: int,
                               pta_metrics: Dict[str, Any], msa_metrics: Dict[str, Any],
                               analysis: Dict[str, Any],
                               mutate_success: bool, pta_success: bool, msa_success: bool,
                               reason: str = ""):
    """生成每轮迭代的report.md（与Task1保持一致）"""
    report_path = iter_dir / "report.md"
    lines = [
        f"# Iteration {iter_num} Report",
        "",
        f"**Model**: {Config.MODEL_NAME or 'unknown'}",
        f"**Iteration**: {iter_num}",
        f"**Status**: {'PASS' if (mutate_success and pta_success) else 'FAIL'}",
        "",
        "## Components",
        "",
        f"- MUTATE: {'PASS' if mutate_success else 'FAIL'}",
        f"- PTA_VERIFY: {'PASS' if pta_success else 'FAIL'}",
        f"- MSA_VERIFY: {'PASS' if msa_success else 'FAIL'}",
        f"- ANALYSIS: {'PASS' if not analysis.get('issues') else 'FAIL'}",
        "",
        "## Status Summary",
        "",
    ]
    # 精度状态
    pta_loss = pta_metrics.get('loss')
    msa_loss = msa_metrics.get('loss')
    if pta_loss is None and msa_loss is None:
        acc_status = "N/A (推理模型)"
    elif pta_loss is None or msa_loss is None:
        acc_status = "缺失"
    elif analysis.get('loss_match', True):
        acc_status = "OK"
    else:
        acc_status = "不匹配"
    # 显存状态
    pta_mem = pta_metrics.get('memory')
    msa_mem = msa_metrics.get('memory')
    if pta_mem is not None and msa_mem is not None:
        mem_status = "OK"
    else:
        mem_status = "缺失"
    # 性能状态
    pta_time = pta_metrics.get('time')
    msa_time = msa_metrics.get('time')
    if pta_time is not None and msa_time is not None:
        perf_status = "OK"
    else:
        perf_status = "缺失"
    lines.extend([
        f"- 精度: {acc_status}",
        f"- 显存: {mem_status}",
        f"- 性能: {perf_status}",
        "",
        "## Metrics",
        "",
        "### PTA",
        f"- Loss: {pta_metrics.get('loss', 'N/A')}",
        f"- Memory: {pta_metrics.get('memory', 'N/A')} MB",
        f"- Time: {pta_metrics.get('time', 'N/A')} ms",
        "",
        "### MSA",
        f"- Loss: {msa_metrics.get('loss', 'N/A')}",
        f"- Memory: {msa_metrics.get('memory', 'N/A')} MB",
        f"- Time: {msa_metrics.get('time', 'N/A')} ms",
        "",
        "## Analysis",
        "",
    ])
    issues = analysis.get("issues", [])
    if issues:
        for issue in issues:
            lines.append(f"- {issue}")
    else:
        lines.append("- No issues detected")
    if reason:
        lines.extend(["", f"**Reason**: {reason}"])
    lines.extend([
        "",
        "## Files",
        "",
        f"- Status: `status.json`",
        f"- Metrics: `metrics.json`",
        f"- PTA Log: `runtime_logs/pta_verify_iter{iter_num}.log`",
        f"- MSA Log: `runtime_logs/msa_verify_iter{iter_num}.log`",
        f"- Scripts: `scripts/`",
        f"- Weights: `weights/`",
        "",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _refresh_iteration_analysis(run_persist_dir: str):
    """刷新迭代分析报告（与Task1-5的refresh_iteration_analysis一致，每轮调用）"""
    try:
        from utils.analyze import task6_result
        persist_root = Config.PERSIST_ROOT or str(RESULTS_ROOT)
        analysis_dir = Path(persist_root) / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        report = task6_result.analyze_task6_run(run_persist_dir, Config.MODEL_NAME)
        task6_result.save_report(report, str(analysis_dir))
        log_debug(f"分析报告已刷新: {analysis_dir}")
    except Exception as exc:
        log_debug(f"分析报告刷新失败，已跳过: {exc}")


# ====================== 主流程 ======================
def _clean_directories():
    """清理Task6中间结果目录，确保执行前为空"""
    for log_dir in [Path("pta_logs"), Path("msrun_log")]:
        if log_dir.exists():
            shutil.rmtree(log_dir, ignore_errors=True)
    # output目录保留不清空
    # 同时清理MINDSPEED_MM_PATH下的msrun_log（含marker文件），确保每轮Task6都从第一次状态开始
    mm_path = os.environ.get("MINDSPEED_MM_PATH")
    if mm_path:
        mm_msrun_log = Path(mm_path) / "msrun_log"
        if mm_msrun_log.exists():
            shutil.rmtree(mm_msrun_log, ignore_errors=True)
    if RESULTS_ROOT.exists():
        shutil.rmtree(RESULTS_ROOT, ignore_errors=True)
    if TASK6_TMP_ROOT.exists():
        shutil.rmtree(TASK6_TMP_ROOT, ignore_errors=True)
    if Config.PERSIST_ROOT and Config.ITER_RESULT_DIR:
        persist_iters_dir = Path(Config.PERSIST_ROOT) / Config.ITER_RESULT_DIR
        if persist_iters_dir.exists():
            shutil.rmtree(persist_iters_dir, ignore_errors=True)
    os.makedirs(TASK6_TMP_ROOT, exist_ok=True)
    os.makedirs(MUTATION_OUTPUT_ROOT, exist_ok=True)
    os.makedirs(RESULTS_ROOT, exist_ok=True)


def main(params: Dict[str, Any]):
    """
    Task6主流程
    """
    # 清理所有残留训练进程（防止上一次执行未正常结束导致端口占用或重复输出）
    utils.control.clean.kill_pretraingpt()
    _kill_remote_processes()

    log_step("开始Task6多模态整网变异和验证任务")

    # 初始化配置
    if not _init_config(params):
        log_error("配置初始化失败")
        return 1

    # 设置硬件确定性环境变量（排除硬件侧随机性）
    # 多机模式下 HCCL_DETERMINISTIC 会导致 HCCL 同步死锁，仅在单机设置
    if not Config.MULTI_NODE_ENABLED:
        os.environ["HCCL_DETERMINISTIC"] = "true"
        os.environ["NCCL_DETERMINISTIC"] = "1"

    # 清理并创建临时目录
    _clean_directories()

    # 获取模型配置
    model_config = Config.MODEL_CONFIGS[Config.MODEL_NAME]
    log_debug(f"使用模型配置: {model_config}")

    # 存储结果用于最终分析
    all_results = []

    # 统计计数器（按照用户定义的规则）
    valid_iter_count = 0      # 有效突变次数（PTA成功的轮次）
    invalid_iter_count = 0    # 无效突变次数（PTA失败的轮次，被撤回的）
    total_mutations = 0       # 总突变次数 = 有效 + 无效
    pta_success_count = 0     # PTA成功次数（即有效突变次数）
    msa_success_count = 0     # MSA成功次数（PTA成功且MSA也成功的轮次）
    issue_count = 0           # 发现问题数（MSA不成功的次数）
    attempt_count = 0         # 尝试次数（用于日志文件命名）
    MAX_MUTATION_RETRIES = 10 # 单轮最大重试次数
    retry_count = 0           # 当前轮次连续失败次数

    while valid_iter_count < Config.TOTAL_ITER:
        # 每轮开始前深度清理NPU环境（最大程度防止507018等NPU状态错误）
        utils.control.clean.kill_pretraingpt()
        _kill_remote_processes()

        valid_iter_count += 1
        retry_count += 1
        attempt_count += 1
        iter_num = valid_iter_count  # iter_num只与成功轮次相关

        if retry_count > MAX_MUTATION_RETRIES:
            log_error("无法生成PTA合法模型，请检查PTA配置")
            raise RuntimeError("无法生成PTA合法模型，请检查PTA配置")

        log_step(f"第 {iter_num}/{Config.TOTAL_ITER} 轮开始")

        iter_start_time = time.time()
        iter_success = True
        pta_metrics = {}
        msa_metrics = {}
        analysis = {"loss_match": True, "issues": []}
        mutation_path = ""
        # 使用iter_num命名日志文件，确保归档时按成功轮次组织
        pta_log_file = os.path.join(TASK6_TMP_ROOT, f"pta_verify_iter{iter_num}.log")
        msa_log_file = os.path.join(TASK6_TMP_ROOT, f"msa_verify_iter{iter_num}.log") if Config.COMPARE_MODE == "pta_msa" else ""

        # 步骤1: 执行变异
        mutate_success, mutation_path = run_mutation(iter_num, model_config, Config.MUTNM, attempt_count)
        if not mutate_success:
            log_info("变异失败，撤销本轮突变")
            valid_iter_count -= 1  # 无效轮次不计数
            continue

        # 步骤2: PTA验证
        if Config.MULTI_NODE_ENABLED:
            # 多机模式：同步迭代目录到远端后执行
            iter_log_dir = os.path.join(TASK6_TMP_ROOT, f"iter_{iter_num}_logs")
            os.makedirs(iter_log_dir, exist_ok=True)
            pta_success, pta_metrics = run_pta_verify_multinode(
                iter_num, model_config, pta_log_file, mutation_path, attempt_count,
                iter_log_dir
            )
        else:
            pta_success, pta_metrics = run_pta_verify(iter_num, model_config, pta_log_file, mutation_path, attempt_count)

        if not pta_success:
            # 1. 先归档失败结果（在撤回之前），保存配置和日志
            log_info("PTA验证失败，先归档本轮结果再撤回")
            # 从PTA日志提取真实错误信息
            pta_error_info = "PTA verification failed"
            try:
                if os.path.exists(pta_log_file):
                    with open(pta_log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        log_content = f.read()
                    # 优先匹配 [rankN]: ErrorType: message
                    err_match = re.search(r'^\[?rank\d+\]?:?\s*([A-Za-z][A-Za-z0-9_]*Error)\s*:\s*(.+)$', log_content, re.MULTILINE)
                    if err_match:
                        pta_error_info = f"{err_match.group(1)}: {err_match.group(2).strip()}"
                    else:
                        # 匹配 Traceback 后的错误
                        tb_match = re.search(r'Traceback \(most recent call last\):.*?^(\S+Error):\s*(.+?)$', log_content, re.MULTILINE | re.DOTALL | re.IGNORECASE)
                        if tb_match:
                            pta_error_info = f"{tb_match.group(1)}: {tb_match.group(2).strip()}"
                        else:
                            err_line = re.search(r'ERROR[:\s]+(.+)$', log_content, re.MULTILINE | re.IGNORECASE)
                            if err_line:
                                pta_error_info = err_line.group(1).strip()
                    if len(pta_error_info) > 200:
                        pta_error_info = pta_error_info[:200] + "..."
            except Exception:
                pass
            archive_iteration_result(
                attempt_count, mutation_path,
                pta_log_file, "",
                pta_metrics, {}, {"loss_match": True, "issues": [pta_error_info]},
                mutate_success=True,
                pta_success=False,
                msa_success=False,
                reason=pta_error_info
            )
            # 每轮结束后刷新分析报告
            _refresh_iteration_analysis(_get_run_persist_dir())

            # PTA失败后进行深度NPU清理，防止残留状态导致后续轮次507018
            utils.control.clean.kill_pretraingpt()

            # 2. 然后撤回本轮突变
            log_debug("突变失败，撤销本轮突变")
            rollback_mutation(iter_num, model_config)

            # 删除变异配置文件（tmp/task6/pta_verify_iter*.log 保留用于调试）
            if os.path.exists(mutation_path):
                os.remove(mutation_path)
                log_debug(f"已删除失败变异配置: {mutation_path}")

            # 注意：pta_logs和msrun_log中的日志在Task6开始时统一清空，
            # 每轮执行后保留不删除，确保最后包含所有轮次的日志

            # 统计：无效突变次数增加
            invalid_iter_count += 1
            valid_iter_count -= 1  # 无效轮次不计入有效迭代
            continue
        else:
            # PTA成功，计入成功次数
            pta_success_count += 1
            retry_count = 0  # 本轮成功，重置重试计数

        # 步骤3: MSA验证（如果对比模式是pta_msa）
        msa_success = True  # 默认为True，仅在pta_msa模式下更新
        msa_error_info = ""  # 记录MSA错误信息
        if Config.COMPARE_MODE == "pta_msa":
            if Config.MULTI_NODE_ENABLED:
                iter_log_dir = os.path.join(TASK6_TMP_ROOT, f"iter_{iter_num}_logs")
                os.makedirs(iter_log_dir, exist_ok=True)
                msa_success, msa_metrics = run_msa_verify_multinode(
                    iter_num, model_config, msa_log_file, mutation_path, attempt_count,
                    iter_log_dir
                )
            else:
                msa_success, msa_metrics = run_msa_verify(iter_num, model_config, msa_log_file, mutation_path, attempt_count)

            if not msa_success:
                log_warn("MSA验证失败")
                # 注意：MSA失败不改变iter_success，只要PTA成功就是成功迭代（有效突变）
                # 统计：发现问题数增加（MSA执行失败）
                issue_count += 1

                # 优先使用run_msa_verify已经提取的真实错误信息
                msa_error_info = msa_metrics.get("error_info", "")
                if not msa_error_info:
                    # 降级：从日志中重新提取错误信息
                    try:
                        if os.path.exists(msa_log_file):
                            with open(msa_log_file, 'r', encoding='utf-8', errors='ignore') as f:
                                log_content = f.read()
                            # 提取最后一个ERROR（优先）
                            error_matches = re.findall(r'ERROR[:\s]+(.+?)(?:\n|$)', log_content, re.IGNORECASE | re.MULTILINE)
                            if error_matches:
                                msa_error_info = error_matches[-1].strip()[:200]
                            else:
                                # 无ERROR时，查找其他错误模式
                                error_match = re.search(r'(Traceback|OSError|ModuleNotFoundError|AttributeError|RuntimeError|Fatal|AssertionError)[^\n]*\n(?:.*\n)*?((?:Error|Exception)[^\n]*)', log_content, re.IGNORECASE)
                                if error_match:
                                    msa_error_info = error_match.group(0).strip()[:200]
                                else:
                                    msa_error_info = "MSA执行失败（详见日志）"
                        else:
                            msa_error_info = "MSA日志文件不存在"
                    except Exception as e:
                        msa_error_info = f"MSA执行失败（读取日志异常: {e}）"
            else:
                # MSA成功执行
                msa_success_count += 1

            # 步骤4: 结果分析
            analysis = analyze_results(iter_num, pta_metrics, msa_metrics)

            # 每轮汇总输出一次（控制台）
            _fmt = lambda v, s: f"{v:{s}}" if v is not None else "N/A"
            log_info(f"Iter{iter_num} PTA loss={_fmt(pta_metrics.get('loss'), '.5f')} "
                     f"mem={_fmt(pta_metrics.get('memory'), '.1f')}MB "
                     f"time={_fmt(pta_metrics.get('time'), '.1f')}ms | "
                     f"MSA loss={_fmt(msa_metrics.get('loss'), '.5f')} "
                     f"mem={_fmt(msa_metrics.get('memory'), '.1f')}MB "
                     f"time={_fmt(msa_metrics.get('time'), '.1f')}ms")

            if analysis["issues"]:
                log_warn(f"发现差异: {analysis['issues']}")
            else:
                log_info("PTA和MSA结果一致")

        # 构建归档原因
        reason = ""
        if not msa_success:
            reason = f"MSA verification failed: {msa_error_info}" if msa_error_info else "MSA verification failed"
            # 将错误信息添加到analysis.issues中，以便在报告中显示
            if msa_error_info and not analysis.get("issues"):
                analysis["issues"] = [msa_error_info]
            elif msa_error_info:
                analysis["issues"].insert(0, msa_error_info)
        elif analysis.get("issues"):
            reason = f"Analysis issues: {analysis['issues']}"

        # 归档结果（PTA成功的轮次归档，PTA失败的轮次也在前面提前归档）
        # PTA成功的使用 iter_num（实际轮次号），确保报告按1-10顺序展示
        archive_iteration_result(
            iter_num, mutation_path,
            pta_log_file, msa_log_file,
            pta_metrics, msa_metrics, analysis,
            mutate_success=True,
            pta_success=True,
            msa_success=msa_success,
            reason=reason
        )

        # 每轮结束后刷新分析报告
        _refresh_iteration_analysis(_get_run_persist_dir())

        # 记录本轮结果
        iter_result = {
            "iter": iter_num,
            "success": iter_success,
            "pta_metrics": pta_metrics,
            "msa_metrics": msa_metrics,
            "analysis": analysis,
            "duration": time.time() - iter_start_time,
        }
        all_results.append(iter_result)

        log_info(f"第{iter_num}轮完成（有效突变），耗时: {iter_result['duration']:.2f}s")

    # 生成最终报告
    log_step("========== 任务完成 ==========")

    # 计算总突变次数
    total_mutations = valid_iter_count + invalid_iter_count

    # 构建统计信息字典
    stats = {
        'valid_iter_count': valid_iter_count,
        'invalid_iter_count': invalid_iter_count,
        'total_mutations': total_mutations,
        'pta_success_count': pta_success_count,
        'msa_success_count': msa_success_count,
        'issue_count': issue_count,
    }

    _generate_final_report(all_results, stats)

    return 0


def _generate_final_report(results: List[Dict[str, Any]], stats: Dict[str, int]):
    """生成最终报告（使用Task1相同的分析方式）

    Args:
        results: 迭代结果列表
        stats: 统计信息字典，包含:
            - valid_iter_count: 有效突变次数（PTA成功的轮次）
            - invalid_iter_count: 无效突变次数（被撤回的）
            - total_mutations: 总突变次数
            - pta_success_count: PTA成功次数
            - msa_success_count: MSA成功次数（PTA和MSA都成功）
            - issue_count: 发现问题数（MSA执行失败的次数）
    """
    valid_iter_count = stats.get('valid_iter_count', 0)
    invalid_iter_count = stats.get('invalid_iter_count', 0)
    total_mutations = stats.get('total_mutations', 0)
    pta_success_count = stats.get('pta_success_count', 0)
    msa_success_count = stats.get('msa_success_count', 0)
    issue_count = stats.get('issue_count', 0)
    total_iters = valid_iter_count
    success_iters = msa_success_count
    pta_success_rate = pta_success_count / valid_iter_count if valid_iter_count > 0 else 0
    msa_success_rate = msa_success_count / pta_success_count if pta_success_count > 0 else 0
    issue_rate = issue_count / total_iters if total_iters > 0 else 0

    log_acc(f"统计: 总迭代={total_iters} 成功={success_iters} 发现问题={issue_count} "
             f"PTA成功率={pta_success_rate*100:.1f}% MSA成功率={msa_success_rate*100:.1f}%")

    report_data = {
        "task_type": 6,
        "model": Config.MODEL_NAME,
        "statistics": {
            "total_iterations": total_iters,
            "successful_iterations": success_iters,
            "issue_count": issue_count,
            "pta_success_count": pta_success_count,
            "msa_success_count": msa_success_count,
            "total_mutations": total_mutations,
            "invalid_mutations": invalid_iter_count,
            "pta_success_rate": pta_success_rate,
            "msa_success_rate": msa_success_rate,
            "issue_rate": issue_rate,
        },
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }

    results_path = os.path.join(RESULTS_ROOT, "final_report.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    try:
        from utils.analyze import task6_result
        run_persist_dir = _get_run_persist_dir() or str(TASK6_TMP_ROOT)
        report = task6_result.analyze_task6_run(run_persist_dir, Config.MODEL_NAME)
        # 使用与Task1-5一致的analysis目录
        persist_root = Config.PERSIST_ROOT or str(RESULTS_ROOT)
        analysis_dir = Path(persist_root) / "analysis"
        task6_result.save_report(report, str(analysis_dir))
        log_info(f"最终报告已保存: {analysis_dir}")
    except Exception as exc:
        log_warn(f"分析报告生成失败: {exc}")


