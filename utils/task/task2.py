#!/usr/bin/env python3
"""
模块内组件泛化测试任务（Task2）
重构自旧版 sub_exe_script.sh
"""

import csv
import json
import math
import os
import shutil
import subprocess
import threading
import time
import shutil
import shlex
from datetime import datetime
from pathlib import Path
import yaml

import utils
from utils.runtime.cluster_runtime import ClusterMaster, parse_task123_cluster_config
from utils.analyze.precision import find_iteration_loss_mismatch
from utils.runtime.paths import MODEL_CONFIG_DIR, MUTATION_SCRIPT_DIR, RUNTIME_SCRIPT_DIR, TOKENIZER_DIR, repo_rel
from utils.runtime.profiler_tools import generate_profile_report
from utils.task import data_helpers, runtime_helpers

LMSV_ROOT = Path(__file__).resolve().parents[2]
PROJECT_TMP_ROOT = LMSV_ROOT / "tmp"
TASK2_TMP_ROOT = PROJECT_TMP_ROOT / "task2"
MODEL_CONFIG_REL = repo_rel(MODEL_CONFIG_DIR)
MUTATION_SCRIPT_REL = repo_rel(MUTATION_SCRIPT_DIR)
RUNTIME_SCRIPT_REL = repo_rel(RUNTIME_SCRIPT_DIR)
TOKENIZER_BAICHUAN_REL = repo_rel(TOKENIZER_DIR / "baichuan2")


class Config:
    # 任务参数
    MODE = "DEVELOP"
    TOTAL_ITER = 100
    TEST_ITERATIONS = 1
    MUTATION_ROUNDS = 10
    BASE_SEED = 43
    MUTNM = 2
    MODELS = ["qwen2", "qwen2"]
    SUBMODULES = [4, 3, 5]
    COMPARE_MODE = "pta_msa"
    MF_ARGS_PATH = "assets/runtime/mf_templates/basic.yaml"
    SAVE_STEPS = 1
    LOAD_STEPS = 15

    # 运行配置
    PTA_ENV = "mindspeed"
    MSA_ENV = "msadapter"
    MF_ENV = "mindf_py311"
    PTA_MAX_RUNTIME = 3000
    MSA_MAX_RUNTIME = 3000
    LOG_INIT_WAIT = 240
    LOG_STABLE_THRESHOLD = 150
    ENABLE_MF_WEIGHT_LOAD = False
    SAVE_ABNORMAL_WEIGHTS = True
    TARGET_TENSOR_PARALLEL_SIZE = 0
    TARGET_PIPELINE_PARALLEL_SIZE = 0
    TARGET_EXPERT_PARALLEL_SIZE = 0
    TARGET_NPUS_PER_NODE = 0
    TARGET_WORLD_SIZE = 0
    TARGET_NNODES = 1
    TARGET_NODE_RANK = 0
    TARGET_MASTER_ADDR = "localhost"
    TARGET_REMOTE_MASTER_ADDR = "localhost"
    TARGET_MASTER_PORT = 6000

    # 路径配置
    LOG_PATH = "res/sub_execution.log"
    MSA_MONITOR_LOG = "msrun_log/worker_0.log"
    PTA_CSV_PATH = "res/submodule_execution_pta.csv"
    MSA_CSV_PATH = "res/submodule_execution_msa.csv"
    MF_CSV_PATH = "res/submodule_execution_mf.csv"
    PERSIST_ROOT = ""
    SHARED_WEIGHT_TMP_ROOT = str(TASK2_TMP_ROOT / "shared_weight")


LOG_SCOPE = "Task2"


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


def log_kv(group, key, value):
    utils.log.write.info(_format_log(group, f"{key}: {value}"))


def configure_project_tmp_env():
    return runtime_helpers.configure_project_tmp_env(PROJECT_TMP_ROOT)


def _parse_positive_int(value, default):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    return parsed if parsed > 0 else int(default)


def _parse_optional_positive_int(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def _infer_visible_device_count():
    for env_name in (
        "ASCEND_RT_VISIBLE_DEVICES",
        "ASCEND_VISIBLE_DEVICES",
        "NPU_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
    ):
        raw = str(os.environ.get(env_name, "")).strip()
        if not raw:
            continue
        parts = [item.strip() for item in raw.split(",") if item.strip()]
        if parts:
            return len(parts)

    try:
        import torch

        if hasattr(torch, "npu") and callable(getattr(torch.npu, "device_count", None)):
            count = int(torch.npu.device_count())
            if count > 0:
                return count
        if callable(getattr(torch.cuda, "device_count", None)):
            count = int(torch.cuda.device_count())
            if count > 0:
                return count
    except Exception:
        pass

    for cmd in (
        ["bash", "-lc", "npu-smi info -l | grep -c '^\\s*NPU ID'"],
        ["bash", "-lc", "npu-smi info | grep -c '| NPU '"],
    ):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except OSError:
            continue
        if result.returncode != 0:
            continue
        count = _parse_optional_positive_int((result.stdout or "").strip())
        if count > 0:
            return count

    return 1


def _load_transformer_config(model_path):
    config_path = Path(model_path)
    if not config_path.is_absolute():
        config_path = LMSV_ROOT / config_path
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config = (
        data.get("TransformerConfig")
        or data.get("MLATransformerConfig")
        or data.get("config")
        or data
    )
    return config if isinstance(config, dict) else {}


def _infer_model_aware_tensor_parallel_size(model_paths, visible_cards):
    max_tp = max(1, int(visible_cards))
    if not model_paths:
        return max_tp

    constraints = []
    for model_path in model_paths:
        cfg = _load_transformer_config(model_path)
        for key in ("num_attention_heads", "num_query_groups", "hidden_size", "ffn_hidden_size"):
            value = _parse_optional_positive_int(cfg.get(key))
            if value > 0:
                constraints.append(value)

    if not constraints:
        return max_tp

    for candidate in range(max_tp, 0, -1):
        if all(value % candidate == 0 for value in constraints):
            return candidate
    return 1


def resolve_distributed_config():
    inferred_cards = _infer_visible_device_count()
    tp = _parse_optional_positive_int(Config.TARGET_TENSOR_PARALLEL_SIZE)
    pp = _parse_optional_positive_int(Config.TARGET_PIPELINE_PARALLEL_SIZE)
    ep = _parse_optional_positive_int(Config.TARGET_EXPERT_PARALLEL_SIZE)

    if tp <= 0 and pp <= 0 and ep <= 0:
        tp = inferred_cards
        pp = 1
        ep = 1
    else:
        tp = max(1, tp)
        pp = max(1, pp)
        ep = max(1, ep)

    nnodes = _parse_positive_int(Config.TARGET_NNODES, 1)
    parallel_cards = max(1, tp * pp * ep)
    configured_world = int(Config.TARGET_WORLD_SIZE or 0)

    configured_npus = int(Config.TARGET_NPUS_PER_NODE or 0)
    if configured_npus <= 0:
        configured_npus = int(math.ceil(parallel_cards / nnodes))
    npus_per_node = max(1, configured_npus)
    if configured_world > 0:
        world_size = max(parallel_cards, configured_world)
    else:
        world_size = max(parallel_cards, npus_per_node * nnodes)

    return {
        "tp": tp,
        "pp": pp,
        "ep": ep,
        "inferred_cards": inferred_cards,
        "nnodes": nnodes,
        "node_rank": int(Config.TARGET_NODE_RANK),
        "master_addr": str(Config.TARGET_MASTER_ADDR),
        "broadcast_master_addr": str(Config.TARGET_REMOTE_MASTER_ADDR or Config.TARGET_MASTER_ADDR),
        "master_port": int(Config.TARGET_MASTER_PORT),
        "npus_per_node": npus_per_node,
        "world_size": world_size,
    }


def configure_auto_parallel_from_models(model_paths):
    if any(
        _parse_optional_positive_int(value) > 0
        for value in (
            Config.TARGET_TENSOR_PARALLEL_SIZE,
            Config.TARGET_PIPELINE_PARALLEL_SIZE,
            Config.TARGET_EXPERT_PARALLEL_SIZE,
        )
    ):
        return

    visible_cards = _infer_visible_device_count()
    Config.TARGET_TENSOR_PARALLEL_SIZE = _infer_model_aware_tensor_parallel_size(model_paths, visible_cards)
    Config.TARGET_PIPELINE_PARALLEL_SIZE = 1
    Config.TARGET_EXPERT_PARALLEL_SIZE = 1


def resolve_msa_monitor_log():
    worker_index = max(0, resolve_distributed_config()["npus_per_node"] - 1)
    return f"msrun_log/worker_{worker_index}.log"


def _build_msa_profile_dir_env_block(profile_dir=None):
    if profile_dir:
        return f"export LMSV_MSA_PROFILE_DIR={shlex.quote(str(Path(profile_dir).resolve()))}"
    return "unset LMSV_MSA_PROFILE_DIR"


def _build_pta_socket_ifname_env_block() -> str:
    return """
    unset RANK_TABLE_FILE
    unset RANK_SIZE
    unset RANK_ID
    unset LOCAL_RANK
    unset RANK
    unset GROUP_RANK
    unset ROLE_RANK
    unset ROLE_WORLD_SIZE
    unset LOCAL_WORLD_SIZE
    unset TORCHELASTIC_RUN_ID
    unset TORCHELASTIC_RESTART_COUNT
    unset TORCHELASTIC_MAX_RESTARTS
    unset TORCHELASTIC_ERROR_FILE
    if [[ -z "${GLOO_SOCKET_IFNAME:-}" || -z "${TP_SOCKET_IFNAME:-}" || -z "${HCCL_SOCKET_IFNAME:-}" ]]; then
        LMSV_SOCKET_IFNAME_TARGET="$(getent hosts "$MASTER_ADDR" 2>/dev/null | awk 'NR==1 {print $1}')"
        LMSV_SOCKET_IFNAME_TARGET="${LMSV_SOCKET_IFNAME_TARGET:-$MASTER_ADDR}"
        LMSV_SOCKET_IFNAME="$(ip route get "$LMSV_SOCKET_IFNAME_TARGET" 2>/dev/null | awk '/ dev / {for (i=1;i<=NF;i++) if ($i==\"dev\") {print $(i+1); exit}}')"
        if [[ -z "${LMSV_SOCKET_IFNAME:-}" ]]; then
            LMSV_SOCKET_IFNAME="$(ip route show default 2>/dev/null | awk '/default/ {for (i=1;i<=NF;i++) if ($i==\"dev\") {print $(i+1); exit}}')"
        fi
        if [[ -n "${LMSV_SOCKET_IFNAME:-}" ]]; then
            if [[ -z "${GLOO_SOCKET_IFNAME:-}" ]]; then export GLOO_SOCKET_IFNAME="$LMSV_SOCKET_IFNAME"; fi
            if [[ -z "${TP_SOCKET_IFNAME:-}" ]]; then export TP_SOCKET_IFNAME="$LMSV_SOCKET_IFNAME"; fi
            if [[ -z "${HCCL_SOCKET_IFNAME:-}" ]]; then export HCCL_SOCKET_IFNAME="$LMSV_SOCKET_IFNAME"; fi
            echo "Auto-detected PTA socket interfaces: GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME TP_SOCKET_IFNAME=$TP_SOCKET_IFNAME HCCL_SOCKET_IFNAME=$HCCL_SOCKET_IFNAME"
        fi
    fi
    """.strip()


def _build_multinode_pta_socket_ifname_env_block(dist_cfg) -> str:
    if int(dist_cfg.get("nnodes", 1)) <= 1:
        return ""
    return _build_pta_socket_ifname_env_block()


def run_shell_to_file(cmd, log_file, check=False, timeout=None, timeout_label=None):
    return runtime_helpers.run_shell_to_file(
        cmd,
        log_file,
        LMSV_ROOT,
        log_error,
        check=check,
        timeout=timeout,
        timeout_label=timeout_label,
    )


def backup_artifact_to_output(src_path, run_dir, iter_num, category, dst_name=None, missing_log_level="warn"):
    return runtime_helpers.backup_artifact_to_output(
        src_path,
        run_dir,
        iter_num,
        category,
        LMSV_ROOT,
        log_info,
        log_warn,
        dst_name=dst_name,
        missing_log_level=missing_log_level,
    )


def backup_weight_on_pta_msa_failure(weight_path, run_dir, iter_num, reason):
    if not Config.SAVE_ABNORMAL_WEIGHTS:
        log_info(f"[iter{iter_num}] 已关闭异常迭代权重备份，跳过保存: {reason}")
        return False
    return runtime_helpers.backup_weight_on_pta_msa_failure(
        weight_path,
        run_dir,
        iter_num,
        reason,
        LMSV_ROOT,
        log_info,
        log_warn,
    )


def backup_weight_on_precision_issue(weight_path, run_dir, iter_num, reason):
    if not Config.SAVE_ABNORMAL_WEIGHTS:
        log_info(f"[iter{iter_num}] 已关闭异常迭代权重备份，跳过保存: {reason}")
        return False
    return runtime_helpers.backup_weight_on_precision_issue(
        weight_path,
        run_dir,
        iter_num,
        reason,
        LMSV_ROOT,
        log_info,
        log_warn,
    )


def backup_runtime_log_to_output(log_path, run_dir, iter_num, dst_name=None):
    return runtime_helpers.backup_runtime_log_to_output(
        log_path,
        run_dir,
        iter_num,
        LMSV_ROOT,
        log_info,
        log_warn,
        dst_name=dst_name,
    )


def write_runtime_script(script_path, cmd):
    return runtime_helpers.write_runtime_script(script_path, cmd)


def build_conda_activate_block(env_name, load_ascend=False):
    return runtime_helpers.build_conda_activate_block(env_name, load_ascend=load_ascend)


def normalize_models(raw_models):
    return runtime_helpers.normalize_models(raw_models, MODEL_CONFIG_DIR)


def normalize_submodules(raw_submodules):
    return data_helpers.normalize_int_list(raw_submodules)


def build_mutate_args(model_paths, submodules, mutnm, rounds):
    if len(model_paths) != len(submodules):
        raise ValueError("MODELS 与 SUBMODULES 数量不一致，无法生成变异参数")
    model_arg = ",".join(model_paths)
    sub_arg = ",".join(str(i) for i in submodules)
    return (
        f"-c {MODEL_CONFIG_REL} -r {rounds} --mutnm {mutnm} "
        f"-n {len(submodules)} -m {model_arg} --sub {sub_arg}"
    )


def remove_iteration_rows(csv_path, iteration):
    return data_helpers.remove_iteration_rows(csv_path, iteration, log_warn, log_info)


def cleanup_shared_weight_file(weight_path):
    data_helpers.cleanup_shared_weight_file(weight_path)


def find_mutation_artifacts(iteration):
    return data_helpers.find_mutation_artifacts(LMSV_ROOT / "res", iteration)


def recover_err_mutation_json(err_path, succ_path):
    return data_helpers.recover_err_mutation_json(err_path, succ_path, log_warn)


def csv_has_iteration(csv_path, iteration):
    return data_helpers.csv_has_iteration(csv_path, iteration)


def csv_iteration_is_valid(csv_path, iteration):
    return data_helpers.csv_iteration_is_valid(csv_path, iteration)


def wait_msa_finish(iter_num):
    """等待 MSA 校验完成。成功以日志稳定且结果 CSV 出现当前轮有效指标为准。"""
    log_step(f"等待MSA验证完成 | 迭代{iter_num}")
    log_path = LMSV_ROOT / Config.MSA_MONITOR_LOG
    csv_path = LMSV_ROOT / Config.MSA_CSV_PATH
    return runtime_helpers.wait_msa_finish(
        iter_num=iter_num,
        log_path=log_path,
        total_timeout=Config.MSA_MAX_RUNTIME,
        init_wait=Config.LOG_INIT_WAIT,
        stable_threshold=Config.LOG_STABLE_THRESHOLD,
        poll_interval=20,
        log_info=log_info,
        log_error=log_error,
        success_checker=lambda: csv_iteration_is_valid(csv_path, iter_num),
        result_exists_checker=lambda: csv_has_iteration(csv_path, iter_num),
    )


def init_workspace():
    """清理本任务相关历史产物。"""
    log_step("初始化模块内测试工作目录")
    targets = [
        LMSV_ROOT / "msrun_log",
        LMSV_ROOT / "res" / "submodule_execution_pta.csv",
        LMSV_ROOT / "res" / "submodule_execution_msa.csv",
        LMSV_ROOT / "res" / "training_log_pta",
        LMSV_ROOT / "res" / "training_log_msa",
        LMSV_ROOT / "res" / "training_log_mf",
    ]

    submodule_dir_pattern = LMSV_ROOT / "res"
    if submodule_dir_pattern.exists():
        for item in submodule_dir_pattern.iterdir():
            if item.is_dir() and item.name.startswith("submodule_"):
                targets.append(item)

    for target in targets:
        if target.exists() or target.is_symlink():
            runtime_helpers.clear_path(target)

    (LMSV_ROOT / "res").mkdir(parents=True, exist_ok=True)
    (LMSV_ROOT / "msrun_log").mkdir(parents=True, exist_ok=True)
    (LMSV_ROOT / "res" / "training_log_pta").mkdir(parents=True, exist_ok=True)
    (LMSV_ROOT / "res" / "training_log_msa").mkdir(parents=True, exist_ok=True)
    (LMSV_ROOT / "res" / "training_log_mf").mkdir(parents=True, exist_ok=True)


def snapshot_iter_artifacts(iter_num, run_dir):
    """收集每轮关键产物，便于追溯。"""
    iter_dir = Path(run_dir) / f"iter_{iter_num}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    msrun_src = LMSV_ROOT / "msrun_log"
    if msrun_src.exists():
        shutil.copytree(msrun_src, iter_dir / "msrun_log", dirs_exist_ok=True)

    mutation_dir = iter_dir / "mutation_inputs"
    mutation_dir.mkdir(parents=True, exist_ok=True)
    succ_json, err_json, yaml_cfg = find_mutation_artifacts(iter_num)
    for src in [*succ_json, *err_json, *yaml_cfg]:
        if src.exists():
            shutil.copy2(src, mutation_dir / src.name)


def _repo_rel_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        try:
            return path.relative_to(LMSV_ROOT).as_posix()
        except ValueError:
            return str(path)
    return path.as_posix()


def _build_cluster_session_id():
    output_name = Path(Config.PERSIST_ROOT).name or f"task2-{os.getpid()}"
    return f"{output_name}-task2"


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
    iter_num,
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


def write_iteration_status(
    iter_num,
    run_dir,
    overall_status,
    reason="",
    *,
    mutate_result="SKIP",
    pta_save_result="SKIP",
    pta_load_result="SKIP",
    msa_load_result="SKIP",
    mf_result="SKIP",
):
    iter_dir = Path(run_dir) / f"iter_{iter_num}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "task_type": 2,
        "iteration": iter_num,
        "overall_status": overall_status,
        "reason": reason,
        "components": {
            "MUTATE": mutate_result,
            "PTA_SAVE": pta_save_result,
            "PTA_LOAD": pta_load_result,
            "MSA_LOAD": msa_load_result,
            "MF": mf_result,
        },
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(iter_dir / "status.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    if overall_status != "PASS":
        with open(iter_dir / "FAILED_FLAG", "w", encoding="utf-8") as handle:
            handle.write(
                "MUTATE={MUTATE} PTA_SAVE={PTA_SAVE} PTA_LOAD={PTA_LOAD} "
                "MSA_LOAD={MSA_LOAD} MF={MF}\n".format(**payload["components"])
            )
        with open(iter_dir / "failure_info.txt", "w", encoding="utf-8") as handle:
            handle.write(
                "FAILED_COMPONENTS: "
                "MUTATE={MUTATE} PTA_SAVE={PTA_SAVE} PTA_LOAD={PTA_LOAD} "
                "MSA_LOAD={MSA_LOAD} MF={MF}\n".format(**payload["components"])
            )
            if reason:
                handle.write(f"REASON: {reason}\n")


def run_pta_mutate(iter_num, mutate_args, exec_log_file, pta_env, pta_path):
    cmd = f"""
    {build_conda_activate_block(pta_env, load_ascend=True)}
    export PTAPATH={shlex.quote(pta_path)}
    source scripts/envset/pta.sh
    export MUTATE_ROUND={iter_num}
    export MUTATE_ARGS={shlex.quote(mutate_args)}
    bash {shlex.quote(f"{MUTATION_SCRIPT_REL}/mutate_submodule-auto.sh")}
    """
    result = run_shell_to_file(
        cmd,
        exec_log_file,
        check=False,
        timeout=Config.PTA_MAX_RUNTIME,
        timeout_label="PTA执行",
    )
    return result is not None and result.returncode == 0


def build_pta_verify_stage_cmd(
    iter_num,
    mutate_args,
    pta_env,
    pta_path,
    shared_weight_path,
    shared_mode,
    train_iters,
    step_log_csv_path=None,
):
    train_iters = int(train_iters)
    dist_cfg = resolve_distributed_config()
    if step_log_csv_path:
        step_log_block = f"export LMSV_TRAINING_LOG_CSV={shlex.quote(str(Path(step_log_csv_path).resolve()))}"
    else:
        step_log_block = "unset LMSV_TRAINING_LOG_CSV"
    return f"""
    {build_conda_activate_block(pta_env, load_ascend=True)}
    export PTAPATH={shlex.quote(pta_path)}
    source scripts/envset/pta.sh
    export LMSV_ENABLE_SUBMODULE_SHARED_WEIGHT_PATCH=1
    export LMSV_PATCH_LOG=1
    export LMSV_SUBMODULE_TARGET_SCRIPT=mutate_and_forward/load_and_forward_submodule-auto.py
    export LMSV_SHARED_WEIGHT_TARGET_MODULES=core.subgraph,utils.runtime.core.subgraph

    export HCCL_DETERMINISTIC=true
    export ASCEND_LAUNCH_BLOCKING=1
    export NCCL_DETERMINISTIC=1
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    NPUS_PER_NODE={dist_cfg["npus_per_node"]}
    MASTER_ADDR={shlex.quote(dist_cfg["master_addr"])}
    MASTER_PORT={dist_cfg["master_port"]}
    NNODES={dist_cfg["nnodes"]}
    NODE_RANK={dist_cfg["node_rank"]}
    export MA_NUM_HOSTS="$NNODES"
    export VC_TASK_INDEX="$NODE_RANK"
    export MASTER_ADDR="$MASTER_ADDR"
    {_build_multinode_pta_socket_ifname_env_block(dist_cfg)}

    DISTRIBUTED_ARGS="
        --nproc_per_node $NPUS_PER_NODE \
        --nnodes $NNODES \
        --node_rank $NODE_RANK \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT
    "

    GPT_ARGS="
        --tensor-model-parallel-size {dist_cfg["tp"]} \
        --pipeline-model-parallel-size {dist_cfg["pp"]} \
        --expert-model-parallel-size {dist_cfg["ep"]} \
        --num-layers 16 \
        --hidden-size 928 \
        --ffn-hidden-size 1712 \
        --num-attention-heads 8 \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path {TOKENIZER_BAICHUAN_REL} \
        --seq-length 1024 \
        --max-position-embeddings 1024 \
        --micro-batch-size 1 \
        --global-batch-size 8 \
        --make-vocab-size-divisible-by 1 \
        --seed 114514 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --position-embedding-type rope \
    "

    export MUTATE_ROUND={iter_num}
    export MUTATE_ARGS={shlex.quote(mutate_args)}
    export LMSV_SHARED_WEIGHT_PATH={shlex.quote(shared_weight_path)}
    export LMSV_SHARED_WEIGHT_MODE={shlex.quote(shared_mode)}
    export LMSV_TRAIN_ITERS={train_iters}
    {step_log_block}
    torchrun $DISTRIBUTED_ARGS {shlex.quote(f"{RUNTIME_SCRIPT_REL}/submodule_entry.py")} \
        $GPT_ARGS \
        $MUTATE_ARGS \
        --train-iters {train_iters}
    """


def run_pta_verify_stage(
    iter_num,
    mutate_args,
    exec_log_file,
    pta_env,
    pta_path,
    shared_weight_path,
    shared_mode,
    train_iters,
    step_log_csv_path=None,
    script_output_path=None,
):
    cmd = build_pta_verify_stage_cmd(
        iter_num,
        mutate_args,
        pta_env,
        pta_path,
        shared_weight_path,
        shared_mode,
        train_iters,
        step_log_csv_path=step_log_csv_path,
    )
    if script_output_path:
        write_runtime_script(script_output_path, cmd)
    result = run_shell_to_file(
        cmd,
        exec_log_file,
        check=False,
        timeout=Config.PTA_MAX_RUNTIME,
        timeout_label="PTA执行",
    )
    return result is not None and result.returncode == 0


def build_msa_verify_load_cmd(
    iter_num,
    mutate_args,
    msa_env,
    msa_path,
    shared_weight_path,
    train_iters,
    step_log_csv_path=None,
    profile_output_dir=None,
):
    train_iters = int(train_iters)
    dist_cfg = resolve_distributed_config()
    if step_log_csv_path:
        step_log_block = f"export LMSV_TRAINING_LOG_CSV={shlex.quote(str(Path(step_log_csv_path).resolve()))}"
    else:
        step_log_block = "unset LMSV_TRAINING_LOG_CSV"
    return f"""
    {build_conda_activate_block(msa_env, load_ascend=True)}
    export MSAPATH={shlex.quote(msa_path)}
    source scripts/envset/msa.sh
    {_build_msa_profile_dir_env_block(profile_output_dir)}
    export LMSV_ENABLE_SUBMODULE_SHARED_WEIGHT_PATCH=1
    export LMSV_PATCH_LOG=1
    export LMSV_SUBMODULE_TARGET_SCRIPT=mutate_and_forward/load_and_forward_submodule-auto.py
    export LMSV_SHARED_WEIGHT_TARGET_MODULES=core.subgraph,utils.runtime.core.subgraph

    export HCCL_DETERMINISTIC=true
    export ASCEND_LAUNCH_BLOCKING=1
    export NCCL_DETERMINISTIC=1
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    NPUS_PER_NODE={dist_cfg["npus_per_node"]}
    MASTER_ADDR={shlex.quote(dist_cfg["master_addr"])}
    MASTER_PORT={dist_cfg["master_port"]}
    NNODES={dist_cfg["nnodes"]}
    NODE_RANK={dist_cfg["node_rank"]}
    WORLD_SIZE={dist_cfg["world_size"]}

    DISTRIBUTED_ARGS="
        --master_addr $MASTER_ADDR \
        --node_rank $NODE_RANK \
        --worker_num $WORLD_SIZE \
        --local_worker_num $NPUS_PER_NODE \
        --master_port $MASTER_PORT \
        --log_dir=msrun_log \
        --join=False \
        --cluster_time_out=300 \
        --bind_core=True
    "

    GPT_ARGS="
        --tensor-model-parallel-size {dist_cfg["tp"]} \
        --pipeline-model-parallel-size {dist_cfg["pp"]} \
        --expert-model-parallel-size {dist_cfg["ep"]} \
        --num-layers 16 \
        --hidden-size 928 \
        --ffn-hidden-size 1712 \
        --num-attention-heads 8 \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path {TOKENIZER_BAICHUAN_REL} \
        --seq-length 1024 \
        --max-position-embeddings 1024 \
        --micro-batch-size 1 \
        --global-batch-size 8 \
        --make-vocab-size-divisible-by 1 \
        --seed 114514 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --position-embedding-type rope \
    "

    export MUTATE_ROUND={iter_num}
    export MUTATE_ARGS={shlex.quote(mutate_args)}
    export LMSV_SHARED_WEIGHT_PATH={shlex.quote(shared_weight_path)}
    export LMSV_SHARED_WEIGHT_MODE=load
    export LMSV_MSA_CSV_PATH={shlex.quote(str((LMSV_ROOT / Config.MSA_CSV_PATH).resolve()))}
    export LMSV_TRAIN_ITERS={train_iters}
    {step_log_block}
    msrun $DISTRIBUTED_ARGS {shlex.quote(f"{RUNTIME_SCRIPT_REL}/submodule_entry.py")} \
        $GPT_ARGS \
        $MUTATE_ARGS \
        --train-iters {train_iters} \
        --msa
    """


def run_msa_verify_load(
    iter_num,
    mutate_args,
    exec_log_file,
    msa_env,
    msa_path,
    shared_weight_path,
    train_iters,
    step_log_csv_path=None,
    profile_output_dir=None,
    script_output_path=None,
):
    cmd = build_msa_verify_load_cmd(
        iter_num,
        mutate_args,
        msa_env,
        msa_path,
        shared_weight_path,
        train_iters,
        step_log_csv_path=step_log_csv_path,
        profile_output_dir=profile_output_dir,
    )
    if script_output_path:
        write_runtime_script(script_output_path, cmd)
    result = run_shell_to_file(cmd, exec_log_file, check=False)
    return result is not None and result.returncode == 0


def convert_shared_weight_for_mf(
    shared_weight_pth_path,
    exec_log_file,
    pta_env,
    mf_env,
    script_output_path=None,
):
    pth = Path(shared_weight_pth_path).resolve()
    npz = pth.with_suffix(".npz")
    ckpt = pth.with_suffix(".ckpt")

    cmd = f"""
    conda run -n {shlex.quote(pta_env)} python utils/runtime/export_pth_to_npz.py \
        --pth {shlex.quote(str(pth))} \
        --npz {shlex.quote(str(npz))}
    conda run -n {shlex.quote(mf_env)} python utils/runtime/convert_npz_to_ckpt.py \
        --npz {shlex.quote(str(npz))} \
        --ckpt {shlex.quote(str(ckpt))}
    """
    if script_output_path:
        write_runtime_script(script_output_path, cmd)
    result = run_shell_to_file(cmd, exec_log_file, check=False)
    ok = (
        result is not None
        and result.returncode == 0
        and ckpt.exists()
        and ckpt.stat().st_size > 0
    )
    return ok, str(ckpt)


def run_mf_verify(
    iter_num,
    mutate_args,
    load_path,
    exec_log_file,
    mf_env,
    mf_args_path,
    train_iters,
    shared_weight_ckpt_path=None,
    step_log_csv_path=None,
    script_output_path=None,
):
    dist_cfg = resolve_distributed_config()
    use_msrun = int(dist_cfg.get("world_size", 1)) > 1
    train_iters = int(train_iters)
    if step_log_csv_path:
        step_log_block = f"export LMSV_TRAINING_LOG_CSV={shlex.quote(str(Path(step_log_csv_path).resolve()))}"
    else:
        step_log_block = "unset LMSV_TRAINING_LOG_CSV"
    if use_msrun:
        launch_cmd = f"""
    NPUS_PER_NODE={dist_cfg["npus_per_node"]}
    MASTER_ADDR={shlex.quote(dist_cfg["master_addr"])}
    MASTER_PORT={dist_cfg["master_port"]}
    NNODES={dist_cfg["nnodes"]}
    NODE_RANK={dist_cfg["node_rank"]}
    WORLD_SIZE={dist_cfg["world_size"]}

    DISTRIBUTED_ARGS="
        --master_addr $MASTER_ADDR \
        --node_rank $NODE_RANK \
        --worker_num $WORLD_SIZE \
        --local_worker_num $NPUS_PER_NODE \
        --master_port $MASTER_PORT \
        --log_dir=msrun_log \
        --join=True \
        --cluster_time_out=300 \
        --bind_core=True
    "

    msrun $DISTRIBUTED_ARGS python utils/runtime/mf_mutate_and_forward/load_and_forward_submodule.py \
        $MUTATE_ARGS \
        --load-path {shlex.quote(load_path)} \
        --train-iters {train_iters} \
        --args_path {shlex.quote(mf_args_path)}
    """
    else:
        launch_cmd = f"""
    python utils/runtime/mf_mutate_and_forward/load_and_forward_submodule.py \
        $MUTATE_ARGS \
        --load-path {shlex.quote(load_path)} \
        --train-iters {train_iters} \
        --args_path {shlex.quote(mf_args_path)}
    """
    cmd = f"""
    {build_conda_activate_block(mf_env, load_ascend=True)}
    export MUTATE_ROUND={iter_num}
    export MUTATE_ARGS={shlex.quote(mutate_args)}
    export LMSV_SHARED_WEIGHT_CKPT_PATH={shlex.quote(str(shared_weight_ckpt_path or ''))}
    export LMSV_ALIGN_ADD_QKV_BIAS=${{LMSV_ALIGN_ADD_QKV_BIAS:-1}}
    export LMSV_TASK3_FORCE_MF_SAFE=${{LMSV_TASK3_FORCE_MF_SAFE:-0}}
    export LMSV_STRICT_ATTN_CONFIG_MATCH=${{LMSV_STRICT_ATTN_CONFIG_MATCH:-0}}
    export LMSV_STRICT_ATTN_PARAM_LOAD=${{LMSV_STRICT_ATTN_PARAM_LOAD:-1}}
    export LMSV_STRICT_DECODER_CONFIG_MATCH=${{LMSV_STRICT_DECODER_CONFIG_MATCH:-0}}
    export LMSV_TRAIN_ITERS={train_iters}
    {step_log_block}
    {launch_cmd}
    """
    if script_output_path:
        write_runtime_script(script_output_path, cmd)
    result = run_shell_to_file(cmd, exec_log_file, check=False)
    return result is not None and result.returncode == 0


def verify_mf(iter_num):
    csv_path = LMSV_ROOT / Config.MF_CSV_PATH
    if not csv_has_iteration(csv_path, iter_num):
        log_error(f"MF第{iter_num}轮结果缺失: {Config.MF_CSV_PATH}")
        return False
    if not data_helpers.csv_iteration_is_valid(csv_path, iter_num):
        log_error(f"MF第{iter_num}轮结果为空（'-'）: {Config.MF_CSV_PATH}")
        return False
    loss_value = data_helpers.csv_iteration_loss(csv_path, iter_num)
    if loss_value is not None and loss_value >= 1e8:
        log_error(
            f"MF第{iter_num}轮loss异常（疑似失败哨兵值）: {loss_value} | {Config.MF_CSV_PATH}"
        )
        return False
    return True


def main(params):
    project_tmp_root = configure_project_tmp_env()
    utils.control.clean.kill_pretraingpt()

    # 读取参数
    Config.MODE = str(params.get("MODE", Config.MODE)).upper()
    Config.TOTAL_ITER = int(params.get("TOTAL_ITER", Config.TOTAL_ITER))
    Config.TEST_ITERATIONS = int(params.get("TEST_ITERATIONS", Config.TEST_ITERATIONS))
    Config.MUTATION_ROUNDS = int(params.get("MUTATION_ROUNDS", Config.MUTATION_ROUNDS))
    Config.BASE_SEED = int(params.get("BASE_SEED", Config.BASE_SEED))
    Config.MUTNM = int(params.get("MUTNM", Config.MUTNM))
    Config.SAVE_STEPS = int(params.get("SAVE_STEPS", Config.SAVE_STEPS))
    Config.LOAD_STEPS = int(params.get("LOAD_STEPS", Config.LOAD_STEPS))
    Config.PTA_MAX_RUNTIME = int(params.get("PTA_MAX_RUNTIME", Config.PTA_MAX_RUNTIME))
    Config.MSA_MAX_RUNTIME = int(params.get("MSA_MAX_RUNTIME", params.get("MAX_VALIDATE_TIME", Config.MSA_MAX_RUNTIME)))
    Config.LOG_INIT_WAIT = int(params.get("LOG_INIT_WAIT", Config.LOG_INIT_WAIT))
    Config.LOG_STABLE_THRESHOLD = int(params.get("LOG_STABLE_THRESHOLD", Config.LOG_STABLE_THRESHOLD))
    Config.TARGET_TENSOR_PARALLEL_SIZE = _parse_optional_positive_int(
        params.get("TARGET_TENSOR_PARALLEL_SIZE", Config.TARGET_TENSOR_PARALLEL_SIZE),
    )
    Config.TARGET_PIPELINE_PARALLEL_SIZE = _parse_optional_positive_int(
        params.get("TARGET_PIPELINE_PARALLEL_SIZE", Config.TARGET_PIPELINE_PARALLEL_SIZE),
    )
    Config.TARGET_EXPERT_PARALLEL_SIZE = _parse_optional_positive_int(
        params.get("TARGET_EXPERT_PARALLEL_SIZE", Config.TARGET_EXPERT_PARALLEL_SIZE),
    )
    Config.TARGET_NPUS_PER_NODE = int(params.get("TARGET_NPUS_PER_NODE", Config.TARGET_NPUS_PER_NODE) or 0)
    Config.TARGET_WORLD_SIZE = int(params.get("TARGET_WORLD_SIZE", Config.TARGET_WORLD_SIZE) or 0)
    Config.TARGET_NNODES = _parse_positive_int(params.get("TARGET_NNODES", Config.TARGET_NNODES), Config.TARGET_NNODES)
    Config.TARGET_NODE_RANK = int(params.get("TARGET_NODE_RANK", Config.TARGET_NODE_RANK))
    Config.TARGET_MASTER_ADDR = str(params.get("TARGET_MASTER_ADDR", Config.TARGET_MASTER_ADDR))
    Config.TARGET_REMOTE_MASTER_ADDR = str(params.get("TARGET_MASTER_ADDR", Config.TARGET_REMOTE_MASTER_ADDR))
    Config.TARGET_MASTER_PORT = int(params.get("TARGET_MASTER_PORT", Config.TARGET_MASTER_PORT))
    cluster_cfg = parse_task123_cluster_config(params)
    cluster = None
    cluster_session_id = ""

    model_paths = normalize_models(params.get("MODELS", Config.MODELS))
    submodules = normalize_submodules(params.get("SUBMODULES", Config.SUBMODULES))
    if not model_paths:
        log_error("任务2参数错误：MODELS 为空或格式非法")
        return 1
    if not submodules:
        log_error("任务2参数错误：SUBMODULES 为空或格式非法")
        return 1
    if len(model_paths) != len(submodules):
        log_error("任务2参数错误：MODELS 与 SUBMODULES 必须一一对应")
        return 1
    configure_auto_parallel_from_models(model_paths)
    Config.MSA_MONITOR_LOG = resolve_msa_monitor_log()

    pta_path = os.environ.get("PTA_PATH") or os.environ.get("PTAPATH")
    msa_path = os.environ.get("MSA_PATH") or os.environ.get("MSAPATH")

    # COMPARE_MODE 解析（需要先于环境变量检查，因为不同模式需要不同的环境变量）
    compare_mode_raw = str(params.get("COMPARE_MODE", Config.COMPARE_MODE)).strip().lower()
    aliases = {"pta_msa": "pta_msa", "pta-msa": "pta_msa", "pta_ms": "pta_msa", "pta-ms": "pta_msa", "msa": "pta_msa",
               "pta_mf": "pta_mf", "pta-mf": "pta_mf", "mf": "pta_mf"}
    Config.COMPARE_MODE = aliases.get(compare_mode_raw, "pta_msa")
    run_msa = Config.COMPARE_MODE == "pta_msa"
    run_mf = Config.COMPARE_MODE == "pta_mf"

    if not pta_path:
        log_error("环境变量缺失：请先配置 PTA_PATH")
        return 1
    if run_msa and not msa_path:
        log_error("当前为 pta_msa 模式，环境变量缺失：请先配置 MSA_PATH")
        return 1

    Config.PTA_ENV = str(params.get("PTA_ENV", os.environ.get("PTA_NAME", Config.PTA_ENV)))
    Config.MSA_ENV = str(params.get("MSA_ENV", os.environ.get("MSA_NAME", Config.MSA_ENV)))
    # Task2 的 MF 环境统一走全局 MF_NAME；保留旧配置里的 MF_ENV 作为兼容兜底。
    Config.MF_ENV = str(
        os.environ.get("MF_NAME")
        or params.get("MF_NAME")
        or params.get("MF_ENV")
        or Config.MF_ENV
    )
    Config.MF_ARGS_PATH = str(params.get("MF_ARGS_PATH", Config.MF_ARGS_PATH))
    Config.ENABLE_MF_WEIGHT_LOAD = data_helpers.parse_bool(
        params.get("ENABLE_MF_WEIGHT_LOAD", Config.ENABLE_MF_WEIGHT_LOAD)
    )
    Config.SAVE_ABNORMAL_WEIGHTS = data_helpers.parse_bool(
        params.get("SAVE_ABNORMAL_WEIGHTS", os.environ.get("SAVE_ABNORMAL_WEIGHTS", Config.SAVE_ABNORMAL_WEIGHTS))
    )

    os.environ["BASE_SEED"] = str(Config.BASE_SEED)
    raw_persist_root = str(
        params.get(
            "PERSIST_ROOT",
            f"{os.environ.get('LMSV_OUTPATH', str(LMSV_ROOT / 'output'))}",
        )
    )
    persist_root_path = Path(raw_persist_root).expanduser()
    if not persist_root_path.is_absolute():
        persist_root_path = LMSV_ROOT / persist_root_path
    Config.PERSIST_ROOT = str(persist_root_path.resolve())
    raw_tmp_root = str(params.get("SHARED_WEIGHT_TMP_ROOT", Config.SHARED_WEIGHT_TMP_ROOT))
    tmp_root_path = Path(raw_tmp_root).expanduser()
    if not tmp_root_path.is_absolute():
        tmp_root_path = LMSV_ROOT / tmp_root_path
    Config.SHARED_WEIGHT_TMP_ROOT = str(tmp_root_path.resolve())

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
        Config.MSA_MONITOR_LOG = resolve_msa_monitor_log()

    max_iterations = Config.TEST_ITERATIONS if Config.MODE == "TEST" else Config.TOTAL_ITER
    mutate_args = build_mutate_args(model_paths, submodules, Config.MUTNM, Config.MUTATION_ROUNDS)

    log_step("任务启动")
    log_kv("配置", "模式", Config.MODE)
    log_kv("配置", "迭代次数", max_iterations)
    log_kv("配置", "基础随机种子", Config.BASE_SEED)
    log_kv("配置", "模型配置", model_paths)
    log_kv("配置", "子模块序列", submodules)
    log_kv("配置", "MUTATE_ARGS", mutate_args)
    log_kv("配置", "对比模式", Config.COMPARE_MODE)
    log_kv("配置", "训练步数", f"SAVE({Config.SAVE_STEPS}) | LOAD({Config.LOAD_STEPS})")
    dist_cfg = resolve_distributed_config()
    log_kv(
        "配置",
        "并行设置",
        (
            f"TP={dist_cfg['tp']} | PP={dist_cfg['pp']} | EP={dist_cfg['ep']} | "
            f"NPUS_PER_NODE={dist_cfg['npus_per_node']} | NNODES={dist_cfg['nnodes']} | WORLD_SIZE={dist_cfg['world_size']}"
        ),
    )
    log_kv("配置", "可见卡数(自动探测)", dist_cfg["inferred_cards"])
    log_kv("配置", "MSA监控日志", Config.MSA_MONITOR_LOG)
    if cluster is not None:
        slave_summary = ", ".join(
            f"rank{node.node_rank}@{node.endpoint}:{cluster.slave_worker_count(node)}卡"
            for node in cluster.config.slaves
        )
        log_kv("配置", "多机模式", f"启用 | MASTER={cluster.config.master_addr}:{cluster.config.master_port}")
        log_kv("配置", "多机节点", f"本机rank{cluster.config.node_rank}:{cluster.local_worker_count()}卡 | {slave_summary}")
    peer_env = Config.MSA_ENV if run_msa else Config.MF_ENV
    log_kv("配置", "当前执行对", f"PTA + {'MSA' if run_msa else 'MF'}")
    log_kv("配置", "激活环境", f"PTA={Config.PTA_ENV} | {'MSA' if run_msa else 'MF'}={peer_env}")
    log_kv("配置", "异常迭代权重备份", "启用" if Config.SAVE_ABNORMAL_WEIGHTS else "关闭")
    if run_mf:
        log_kv("配置", "MF 权重加载", "启用" if Config.ENABLE_MF_WEIGHT_LOAD else "跳过（仅跑流程）")
    if not run_msa:
        log_info("MSA 链路未启用（COMPARE_MODE=pta_mf）")
    if not run_mf:
        log_info("MF 链路未启用（COMPARE_MODE=pta_msa）")
    log_kv("配置", "PTA 最大运行时间", f"{Config.PTA_MAX_RUNTIME}s")
    log_kv("配置", "MSA 最大运行时间", f"{Config.MSA_MAX_RUNTIME}s")
    log_kv("配置", "项目临时目录", project_tmp_root)
    log_kv("配置", "共享权重临时目录", Config.SHARED_WEIGHT_TMP_ROOT)
    log_kv("概览", "开始时间", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    init_workspace()

    run_dir = (Path(Config.PERSIST_ROOT) / "iters").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    shared_weight_tmp_run_dir = (
        Path(Config.SHARED_WEIGHT_TMP_ROOT) / f"task2_{run_dir.parent.name}_{os.getpid()}"
    ).resolve()
    shared_weight_tmp_run_dir.mkdir(parents=True, exist_ok=True)
    if cluster is not None:
        cluster_session_id = _build_cluster_session_id()
        cluster.prepare_session(cluster_session_id)

    def refresh_iteration_analysis() -> None:
        try:
            from utils.analyze.task2_result import analyze_task2_run

            analyze_task2_run(
                output_root=Path(Config.PERSIST_ROOT).resolve(),
                run_dir=run_dir,
                model_name=",".join(Path(path).stem for path in model_paths),
                planned_iterations=max_iterations,
            )
        except Exception as exc:
            log_warn(f"迭代分析刷新失败，已跳过: {exc}")

    pta_success_count = 0
    msa_success_count = 0
    mf_success_count = 0

    for i in range(1, max_iterations + 1):
        try:
            log_step(f"开始迭代 {i}/{max_iterations}")
            utils.control.clean.kill_pretraingpt()

            mutate_result = "SKIP"
            pta_save_result = "SKIP"
            pta_load_result = "SKIP"
            msa_load_result = "SKIP" if run_msa else "DISABLED"
            mf_result = "SKIP" if run_mf else "DISABLED"

            runtime_log_dir = run_dir / f"iter_{i}" / "runtime_logs"
            runtime_log_dir.mkdir(parents=True, exist_ok=True)
            mutate_log = runtime_log_dir / f"pta_mutate_iter{i}.log"
            pta_save_log = runtime_log_dir / f"pta_save_iter{i}.log"
            pta_load_log = runtime_log_dir / f"pta_load_iter{i}.log"
            msa_load_log = runtime_log_dir / f"msa_load_iter{i}.log"
            msa_profile_dir = run_dir / f"iter_{i}" / "profiler" / "msa-load"
            msa_profile_report_dir = run_dir / f"iter_{i}" / "analysis" / "msa-profiler"
            mf_log = runtime_log_dir / f"mf_iter{i}.log"
            convert_log = runtime_log_dir / f"convert_iter{i}.log"
            script_artifact_dir = run_dir / f"iter_{i}" / "scripts"
            pta_step_csv = LMSV_ROOT / "res" / "training_log_pta" / f"training_log-{i}.csv"
            msa_step_csv = LMSV_ROOT / "res" / "training_log_msa" / f"training_log-{i}.csv"
            mf_step_csv = LMSV_ROOT / "res" / "training_log_mf" / f"training_log-{i}.csv"
            pta_step_csv.unlink(missing_ok=True)
            msa_step_csv.unlink(missing_ok=True)
            mf_step_csv.unlink(missing_ok=True)
            shared_weight_file = (
                shared_weight_tmp_run_dir / f"iter{i}.pth"
            ).resolve()
            shared_weight_ckpt_file = shared_weight_file.with_suffix(".ckpt")
            shared_weight_file.parent.mkdir(parents=True, exist_ok=True)
            cleanup_shared_weight_file(shared_weight_file)
            shared_weight_path = str(shared_weight_file)
            remote_shared_weight_path = _repo_rel_path(shared_weight_file)
            remote_shared_weight_ckpt_path = _repo_rel_path(shared_weight_ckpt_file)

            log_step("1. 执行PTA侧子模块变异")
            mutate_ok = run_pta_mutate(i, mutate_args, mutate_log, Config.PTA_ENV, pta_path)
            backup_runtime_log_to_output(mutate_log, run_dir, i)
            if not mutate_ok:
                log_error(f"第{i}轮 mutate_submodule-auto 执行失败，跳过本轮")
                mutate_result = "ERROR"
                write_iteration_status(
                    i,
                    run_dir,
                    "FAILED",
                    "mutate_submodule-auto 执行失败",
                    mutate_result=mutate_result,
                    pta_save_result=pta_save_result,
                    pta_load_result=pta_load_result,
                    msa_load_result=msa_load_result,
                    mf_result=mf_result,
                )
                snapshot_iter_artifacts(i, run_dir)
                cleanup_shared_weight_file(shared_weight_file)
                continue
            mutate_result = "OK"

            succ_json, err_json, yaml_cfg = find_mutation_artifacts(i)
            if not succ_json and err_json:
                recovered = False
                for err_file in err_json:
                    succ_file = err_file.with_name(err_file.name.replace("-err", ""))
                    if recover_err_mutation_json(err_file, succ_file):
                        recovered = True
                if recovered:
                    succ_json, err_json, yaml_cfg = find_mutation_artifacts(i)

            if not succ_json:
                log_error(f"第{i}轮 mutate 未生成可加载JSON（mutating-{i}.json），跳过本轮")
                write_iteration_status(
                    i,
                    run_dir,
                    "FAILED",
                    f"mutate 未生成可加载JSON（mutating-{i}.json）",
                    mutate_result=mutate_result,
                    pta_save_result=pta_save_result,
                    pta_load_result=pta_load_result,
                    msa_load_result=msa_load_result,
                    mf_result=mf_result,
                )
                snapshot_iter_artifacts(i, run_dir)
                cleanup_shared_weight_file(shared_weight_file)
                continue
            if not yaml_cfg:
                log_error(f"第{i}轮 mutate 未生成YAML（mutated_config_iter_{i:03d}.yaml），跳过本轮")
                write_iteration_status(
                    i,
                    run_dir,
                    "FAILED",
                    f"mutate 未生成YAML（mutated_config_iter_{i:03d}.yaml）",
                    mutate_result=mutate_result,
                    pta_save_result=pta_save_result,
                    pta_load_result=pta_load_result,
                    msa_load_result=msa_load_result,
                    mf_result=mf_result,
                )
                snapshot_iter_artifacts(i, run_dir)
                cleanup_shared_weight_file(shared_weight_file)
                continue

            # 保存load_path供后续使用（MF需要）
            load_path = str(succ_json[0]) if succ_json else ""

            log_step("2. PTA-SAVE：生成共享权重")
            if cluster is None:
                pta_save_ok = run_pta_verify_stage(
                    i,
                    mutate_args,
                    pta_save_log,
                    Config.PTA_ENV,
                    pta_path,
                    shared_weight_path,
                    "save",
                    Config.SAVE_STEPS,
                    script_output_path=script_artifact_dir / f"pta-save_iter{i}.sh",
                )
            else:
                stage_dist_cfg = resolve_distributed_config()

                def _local_pta_save():
                    return run_pta_verify_stage(
                        i,
                        mutate_args,
                        pta_save_log,
                        Config.PTA_ENV,
                        pta_path,
                        shared_weight_path,
                        "save",
                        Config.SAVE_STEPS,
                        script_output_path=script_artifact_dir / f"pta-save_iter{i}.sh",
                    )

                local_ok, remote_ok, remote_states = _run_cluster_stage(
                    cluster=cluster,
                    session_id=cluster_session_id,
                    stage_name=f"pta_save_iter{i}",
                    iter_num=i,
                    runtime_log_dir=runtime_log_dir,
                    local_runner=_local_pta_save,
                    payload_builder=lambda node, node_workers: {
                        "job_type": "task2_pta_verify",
                        "iter_num": i,
                        "mutate_args": mutate_args,
                        "shared_weight_path": remote_shared_weight_path,
                        "shared_mode": "save",
                        "train_iters": Config.SAVE_STEPS,
                        "pta_max_runtime": Config.PTA_MAX_RUNTIME,
                        "msa_max_runtime": Config.MSA_MAX_RUNTIME,
                        "log_init_wait": Config.LOG_INIT_WAIT,
                        "log_stable_threshold": Config.LOG_STABLE_THRESHOLD,
                        "tp": stage_dist_cfg["tp"],
                        "pp": stage_dist_cfg["pp"],
                        "ep": stage_dist_cfg["ep"],
                        "local_workers": node_workers,
                        "total_workers": stage_dist_cfg["world_size"],
                        "nnodes": stage_dist_cfg["nnodes"],
                        "node_rank": node.node_rank,
                        "master_addr": stage_dist_cfg["broadcast_master_addr"],
                        "master_port": stage_dist_cfg["master_port"],
                        "timeout": Config.PTA_MAX_RUNTIME,
                    },
                    timeout_seconds=Config.PTA_MAX_RUNTIME + 300,
                )
                if not remote_ok:
                    log_error(f"第{i}轮 PTA-SAVE 存在从机失败: {remote_states}")
                pta_save_ok = local_ok and remote_ok
            backup_runtime_log_to_output(pta_save_log, run_dir, i)
            if not pta_save_ok:
                log_error(f"第{i}轮 PTA-SAVE 执行失败，跳过本轮")
                pta_save_result = "ERROR"
                backup_weight_on_pta_msa_failure(shared_weight_file, run_dir, i, "PTA-SAVE执行失败")
                write_iteration_status(
                    i,
                    run_dir,
                    "FAILED",
                    "PTA-SAVE执行失败",
                    mutate_result=mutate_result,
                    pta_save_result=pta_save_result,
                    pta_load_result=pta_load_result,
                    msa_load_result=msa_load_result,
                    mf_result=mf_result,
                )
                snapshot_iter_artifacts(i, run_dir)
                cleanup_shared_weight_file(shared_weight_file)
                continue
            pta_save_result = "OK"
            if not shared_weight_file.exists() or shared_weight_file.stat().st_size <= 0:
                log_error(f"第{i}轮 PTA-SAVE 未产出共享权重: {shared_weight_path}")
                log_error(f"请检查PTA-SAVE日志: {pta_save_log}")
                log_error(f"第{i}轮判定为ERROR，直接进入下一轮")
                pta_save_result = "ERROR"
                backup_weight_on_pta_msa_failure(shared_weight_file, run_dir, i, "PTA-SAVE未产出共享权重")
                write_iteration_status(
                    i,
                    run_dir,
                    "FAILED",
                    "PTA-SAVE未产出共享权重",
                    mutate_result=mutate_result,
                    pta_save_result=pta_save_result,
                    pta_load_result=pta_load_result,
                    msa_load_result=msa_load_result,
                    mf_result=mf_result,
                )
                snapshot_iter_artifacts(i, run_dir)
                cleanup_shared_weight_file(shared_weight_file)
                continue
            remove_iteration_rows(Config.PTA_CSV_PATH, i)
            log_info(f"第{i}轮共享权重已生成: {shared_weight_path}")

            shared_weight_ckpt_path = ""
            if run_mf:
                if Config.ENABLE_MF_WEIGHT_LOAD:
                    log_step("2.1 PTA->MF 权重格式转换（pth->ckpt）")
                    convert_ok, shared_weight_ckpt_path = convert_shared_weight_for_mf(
                        shared_weight_path,
                        convert_log,
                        Config.PTA_ENV,
                        Config.MF_ENV,
                        script_output_path=script_artifact_dir / f"convert_iter{i}.sh",
                    )
                    backup_runtime_log_to_output(convert_log, run_dir, i)
                    if not convert_ok:
                        log_error(f"第{i}轮 权重转换失败: {shared_weight_path} -> {shared_weight_ckpt_path}")
                        mf_result = "ERROR"
                        write_iteration_status(
                            i,
                            run_dir,
                            "FAILED",
                            "PTA->MF权重转换失败",
                            mutate_result=mutate_result,
                            pta_save_result=pta_save_result,
                            pta_load_result=pta_load_result,
                            msa_load_result=msa_load_result,
                            mf_result=mf_result,
                        )
                        snapshot_iter_artifacts(i, run_dir)
                        cleanup_shared_weight_file(shared_weight_file)
                        continue
                    log_info(f"第{i}轮共享权重转换完成: {shared_weight_ckpt_path}")
                else:
                    log_step("2.1 跳过MF权重转换（ENABLE_MF_WEIGHT_LOAD=False）")

            utils.control.clean.kill_pretraingpt()

            log_step("3. PTA-LOAD：基于共享权重验证")
            if cluster is None:
                pta_load_ok = run_pta_verify_stage(
                    i,
                    mutate_args,
                    pta_load_log,
                    Config.PTA_ENV,
                    pta_path,
                    shared_weight_path,
                    "load",
                    Config.LOAD_STEPS,
                    step_log_csv_path=pta_step_csv,
                    script_output_path=script_artifact_dir / f"pta-load_iter{i}.sh",
                )
            else:
                stage_dist_cfg = resolve_distributed_config()

                def _local_pta_load():
                    return run_pta_verify_stage(
                        i,
                        mutate_args,
                        pta_load_log,
                        Config.PTA_ENV,
                        pta_path,
                        shared_weight_path,
                        "load",
                        Config.LOAD_STEPS,
                        step_log_csv_path=pta_step_csv,
                        script_output_path=script_artifact_dir / f"pta-load_iter{i}.sh",
                    )

                local_ok, remote_ok, remote_states = _run_cluster_stage(
                    cluster=cluster,
                    session_id=cluster_session_id,
                    stage_name=f"pta_load_iter{i}",
                    iter_num=i,
                    runtime_log_dir=runtime_log_dir,
                    local_runner=_local_pta_load,
                    upload_builder=lambda node, node_workers: [
                        (shared_weight_file, remote_shared_weight_path),
                    ],
                    payload_builder=lambda node, node_workers: {
                        "job_type": "task2_pta_verify",
                        "iter_num": i,
                        "mutate_args": mutate_args,
                        "shared_weight_path": remote_shared_weight_path,
                        "shared_mode": "load",
                        "train_iters": Config.LOAD_STEPS,
                        "step_log_csv_path": _repo_rel_path(pta_step_csv),
                        "pta_max_runtime": Config.PTA_MAX_RUNTIME,
                        "msa_max_runtime": Config.MSA_MAX_RUNTIME,
                        "log_init_wait": Config.LOG_INIT_WAIT,
                        "log_stable_threshold": Config.LOG_STABLE_THRESHOLD,
                        "tp": stage_dist_cfg["tp"],
                        "pp": stage_dist_cfg["pp"],
                        "ep": stage_dist_cfg["ep"],
                        "local_workers": node_workers,
                        "total_workers": stage_dist_cfg["world_size"],
                        "nnodes": stage_dist_cfg["nnodes"],
                        "node_rank": node.node_rank,
                        "master_addr": stage_dist_cfg["broadcast_master_addr"],
                        "master_port": stage_dist_cfg["master_port"],
                        "timeout": Config.PTA_MAX_RUNTIME,
                    },
                    timeout_seconds=Config.PTA_MAX_RUNTIME + 300,
                )
                if not remote_ok:
                    log_error(f"第{i}轮 PTA-LOAD 存在从机失败: {remote_states}")
                pta_load_ok = local_ok and remote_ok
            backup_runtime_log_to_output(pta_load_log, run_dir, i)
            backup_artifact_to_output(pta_step_csv, run_dir, i, "", f"training_log_pta-{i}.csv")
            if not pta_load_ok:
                log_error(f"第{i}轮 PTA-LOAD 执行失败，跳过本轮")
                pta_load_result = "ERROR"
                backup_weight_on_pta_msa_failure(shared_weight_file, run_dir, i, "PTA-LOAD执行失败")
                write_iteration_status(
                    i,
                    run_dir,
                    "FAILED",
                    "PTA-LOAD执行失败",
                    mutate_result=mutate_result,
                    pta_save_result=pta_save_result,
                    pta_load_result=pta_load_result,
                    msa_load_result=msa_load_result,
                    mf_result=mf_result,
                )
                snapshot_iter_artifacts(i, run_dir)
                cleanup_shared_weight_file(shared_weight_file)
                continue
            pta_load_result = "OK"
            pta_success_count += 1
            log_info(f"PTA第{i}轮（LOAD）执行成功")

            utils.control.clean.kill_pretraingpt()

            if run_msa:
                remove_iteration_rows(Config.MSA_CSV_PATH, i)
            elif run_mf:
                remove_iteration_rows(Config.MF_CSV_PATH, i)

            if run_msa:
                log_step("4. MSA-LOAD：基于共享权重验证")
                if cluster is None:
                    msa_load_ok = run_msa_verify_load(
                        i,
                        mutate_args,
                        msa_load_log,
                        Config.MSA_ENV,
                        msa_path,
                        shared_weight_path,
                        Config.LOAD_STEPS,
                        step_log_csv_path=msa_step_csv,
                        profile_output_dir=msa_profile_dir,
                        script_output_path=script_artifact_dir / f"msa-load_iter{i}.sh",
                    )
                else:
                    stage_dist_cfg = resolve_distributed_config()

                    def _local_msa_load():
                        return run_msa_verify_load(
                            i,
                            mutate_args,
                            msa_load_log,
                            Config.MSA_ENV,
                            msa_path,
                            shared_weight_path,
                            Config.LOAD_STEPS,
                            step_log_csv_path=msa_step_csv,
                            profile_output_dir=msa_profile_dir,
                            script_output_path=script_artifact_dir / f"msa-load_iter{i}.sh",
                        )

                    local_ok, remote_ok, remote_states = _run_cluster_stage(
                        cluster=cluster,
                        session_id=cluster_session_id,
                        stage_name=f"msa_load_iter{i}",
                        iter_num=i,
                        runtime_log_dir=runtime_log_dir,
                        local_runner=_local_msa_load,
                        upload_builder=lambda node, node_workers: [
                            (shared_weight_file, remote_shared_weight_path),
                        ],
                        payload_builder=lambda node, node_workers: {
                            "job_type": "task2_msa_verify",
                            "iter_num": i,
                            "mutate_args": mutate_args,
                            "shared_weight_path": remote_shared_weight_path,
                            "train_iters": Config.LOAD_STEPS,
                            "step_log_csv_path": _repo_rel_path(msa_step_csv),
                            "pta_max_runtime": Config.PTA_MAX_RUNTIME,
                            "msa_max_runtime": Config.MSA_MAX_RUNTIME,
                            "log_init_wait": Config.LOG_INIT_WAIT,
                            "log_stable_threshold": Config.LOG_STABLE_THRESHOLD,
                            "tp": stage_dist_cfg["tp"],
                            "pp": stage_dist_cfg["pp"],
                            "ep": stage_dist_cfg["ep"],
                            "local_workers": node_workers,
                            "total_workers": stage_dist_cfg["world_size"],
                            "nnodes": stage_dist_cfg["nnodes"],
                            "node_rank": node.node_rank,
                            "master_addr": stage_dist_cfg["broadcast_master_addr"],
                            "master_port": stage_dist_cfg["master_port"],
                            "timeout": Config.MSA_MAX_RUNTIME,
                        },
                        collect_builder=lambda node, node_workers: (
                            [{"path": "msrun_log", "flatten": True}],
                            run_dir / f"iter_{i}" / "msrun_log" / f"node_{node.node_rank}",
                        ),
                        timeout_seconds=Config.MSA_MAX_RUNTIME + 600,
                    )
                    if not remote_ok:
                        log_error(f"第{i}轮 MSA-LOAD 存在从机失败: {remote_states}")
                    msa_load_ok = local_ok and remote_ok
                backup_runtime_log_to_output(msa_load_log, run_dir, i)
                if not msa_load_ok:
                    log_error(f"第{i}轮 MSA-LOAD 启动失败，跳过本轮")
                    msa_load_result = "ERROR"
                    backup_weight_on_pta_msa_failure(shared_weight_file, run_dir, i, "MSA-LOAD执行失败")
                    write_iteration_status(
                        i,
                        run_dir,
                        "FAILED",
                        "MSA-LOAD执行失败",
                        mutate_result=mutate_result,
                        pta_save_result=pta_save_result,
                        pta_load_result=pta_load_result,
                        msa_load_result=msa_load_result,
                        mf_result=mf_result,
                    )
                    snapshot_iter_artifacts(i, run_dir)
                    cleanup_shared_weight_file(shared_weight_file)
                    continue

                if not wait_msa_finish(i):
                    log_error(f"第{i}轮 MSA-LOAD 校验等待超时或失败，跳过分析")
                    msa_load_result = "ERROR"
                    backup_weight_on_pta_msa_failure(shared_weight_file, run_dir, i, "MSA日志校验失败或超时")
                    write_iteration_status(
                        i,
                        run_dir,
                        "FAILED",
                        "MSA日志校验失败或超时",
                        mutate_result=mutate_result,
                        pta_save_result=pta_save_result,
                        pta_load_result=pta_load_result,
                        msa_load_result=msa_load_result,
                        mf_result=mf_result,
                    )
                    snapshot_iter_artifacts(i, run_dir)
                    cleanup_shared_weight_file(shared_weight_file)
                    continue
                backup_artifact_to_output(msa_step_csv, run_dir, i, "", f"training_log_msa-{i}.csv", missing_log_level="info")
                if msa_profile_dir.exists() and any(msa_profile_dir.rglob("*")):
                    generate_profile_report(
                        profile_dir=msa_profile_dir,
                        report_dir=msa_profile_report_dir,
                        step_csv_path=msa_step_csv,
                        exec_log_path=msa_load_log,
                        task_label="Task2-MSA",
                        iter_num=i,
                    )
                msa_load_result = "OK"
                msa_success_count += 1
                log_info(f"MSA第{i}轮（LOAD）执行成功")
            elif run_mf:
                weight_load_info = "加载共享权重" if Config.ENABLE_MF_WEIGHT_LOAD and shared_weight_ckpt_path else "不加载权重（仅跑流程）"
                log_step(f"4. MF验证（{weight_load_info}）")
                if cluster is None:
                    mf_ok = run_mf_verify(
                        i,
                        mutate_args,
                        load_path,
                        mf_log,
                        Config.MF_ENV,
                        Config.MF_ARGS_PATH,
                        Config.LOAD_STEPS,
                        shared_weight_ckpt_path=shared_weight_ckpt_path,
                        step_log_csv_path=mf_step_csv,
                        script_output_path=script_artifact_dir / f"mf_iter{i}.sh",
                    )
                else:
                    stage_dist_cfg = resolve_distributed_config()

                    def _local_mf():
                        return run_mf_verify(
                            i,
                            mutate_args,
                            load_path,
                            mf_log,
                            Config.MF_ENV,
                            Config.MF_ARGS_PATH,
                            Config.LOAD_STEPS,
                            shared_weight_ckpt_path=shared_weight_ckpt_path,
                            step_log_csv_path=mf_step_csv,
                            script_output_path=script_artifact_dir / f"mf_iter{i}.sh",
                        )

                    upload_items = []
                    if Config.ENABLE_MF_WEIGHT_LOAD and shared_weight_ckpt_path:
                        upload_items.append((Path(shared_weight_ckpt_path), remote_shared_weight_ckpt_path))
                    local_ok, remote_ok, remote_states = _run_cluster_stage(
                        cluster=cluster,
                        session_id=cluster_session_id,
                        stage_name=f"mf_iter{i}",
                        iter_num=i,
                        runtime_log_dir=runtime_log_dir,
                        local_runner=_local_mf,
                        upload_builder=lambda node, node_workers: list(upload_items),
                        payload_builder=lambda node, node_workers: {
                            "job_type": "task2_mf_verify",
                            "iter_num": i,
                            "mutate_args": mutate_args,
                            "load_path": _repo_rel_path(load_path),
                            "mf_args_path": Config.MF_ARGS_PATH,
                            "train_iters": Config.LOAD_STEPS,
                            "shared_weight_ckpt_path": remote_shared_weight_ckpt_path if upload_items else "",
                            "step_log_csv_path": _repo_rel_path(mf_step_csv),
                            "pta_max_runtime": Config.PTA_MAX_RUNTIME,
                            "msa_max_runtime": Config.MSA_MAX_RUNTIME,
                            "log_init_wait": Config.LOG_INIT_WAIT,
                            "log_stable_threshold": Config.LOG_STABLE_THRESHOLD,
                            "tp": stage_dist_cfg["tp"],
                            "pp": stage_dist_cfg["pp"],
                            "ep": stage_dist_cfg["ep"],
                            "local_workers": node_workers,
                            "total_workers": stage_dist_cfg["world_size"],
                            "nnodes": stage_dist_cfg["nnodes"],
                            "node_rank": node.node_rank,
                            "master_addr": stage_dist_cfg["broadcast_master_addr"],
                            "master_port": stage_dist_cfg["master_port"],
                            "timeout": Config.MSA_MAX_RUNTIME,
                        },
                        collect_builder=lambda node, node_workers: (
                            ([{"path": "msrun_log", "flatten": True}] if node_workers > 1 else []),
                            run_dir / f"iter_{i}" / "msrun_log" / f"node_{node.node_rank}",
                        ),
                        timeout_seconds=Config.MSA_MAX_RUNTIME + 600,
                    )
                    if not remote_ok:
                        log_error(f"第{i}轮 MF 存在从机失败: {remote_states}")
                    mf_ok = local_ok and remote_ok
                backup_runtime_log_to_output(mf_log, run_dir, i)
                backup_artifact_to_output(mf_step_csv, run_dir, i, "", f"training_log_mf-{i}.csv", missing_log_level="info")
                if mf_ok and verify_mf(i):
                    mf_result = "OK"
                    mf_success_count += 1
                    log_info(f"MF第{i}轮执行成功")
                else:
                    mf_result = "ERROR"
                    log_warn(f"第{i}轮 MF执行失败")
                utils.control.clean.kill_pretraingpt()

                precision_issue = find_iteration_loss_mismatch(
                    LMSV_ROOT / Config.PTA_CSV_PATH,
                    LMSV_ROOT / Config.MF_CSV_PATH,
                    i,
                )
                if precision_issue:
                    backup_weight_on_precision_issue(
                        shared_weight_file,
                        run_dir,
                        i,
                        precision_issue.replace("MSA=", "MF="),
                    )
            peer_label = "MSA" if run_msa else "MF"
            log_step(f"5. 校验PTA/{peer_label}结果")
            pta_has_iter = csv_has_iteration(LMSV_ROOT / Config.PTA_CSV_PATH, i)
            peer_csv_path = Config.MSA_CSV_PATH if run_msa else Config.MF_CSV_PATH
            peer_has_iter = csv_has_iteration(LMSV_ROOT / peer_csv_path, i)
            if not pta_has_iter or not peer_has_iter:
                log_warn(f"第{i}轮结果文件缺失迭代记录（PTA={pta_has_iter}, {peer_label}={peer_has_iter}）")
                backup_weight_on_pta_msa_failure(
                    shared_weight_file,
                    run_dir,
                    i,
                    f"结果文件缺失迭代记录（PTA={pta_has_iter}, {peer_label}={peer_has_iter}）",
                )
                write_iteration_status(
                    i,
                    run_dir,
                    "FAILED",
                    f"结果文件缺失迭代记录（PTA={pta_has_iter}, {peer_label}={peer_has_iter}）",
                    mutate_result=mutate_result,
                    pta_save_result=pta_save_result,
                    pta_load_result=pta_load_result,
                    msa_load_result=msa_load_result,
                    mf_result=mf_result,
                )
                snapshot_iter_artifacts(i, run_dir)
                cleanup_shared_weight_file(shared_weight_file)
                continue

            precision_issue = find_iteration_loss_mismatch(
                LMSV_ROOT / Config.PTA_CSV_PATH,
                LMSV_ROOT / peer_csv_path,
                i,
            )
            if precision_issue:
                precision_issue = precision_issue.replace("MSA=", f"{peer_label}=")
                backup_weight_on_precision_issue(
                    shared_weight_file,
                    run_dir,
                    i,
                    precision_issue,
                )

            write_iteration_status(
                i,
                run_dir,
                "PASS",
                "迭代执行完成",
                mutate_result=mutate_result,
                pta_save_result=pta_save_result,
                pta_load_result=pta_load_result,
                msa_load_result=msa_load_result,
                mf_result=mf_result,
            )
            snapshot_iter_artifacts(i, run_dir)
            cleanup_shared_weight_file(shared_weight_file)
        finally:
            refresh_iteration_analysis()

    log_step("任务结束")
    if max_iterations > 0:
        pta_rate = pta_success_count * 100 // max_iterations
        msa_rate = msa_success_count * 100 // max_iterations
        mf_rate = mf_success_count * 100 // max_iterations if run_mf else 0
    else:
        pta_rate = 0
        msa_rate = 0
        mf_rate = 0

    log_kv("统计", "PTA 成功", f"{pta_success_count}/{max_iterations} ({pta_rate}%)")
    if run_msa:
        log_kv("统计", "MSA 成功", f"{msa_success_count}/{max_iterations} ({msa_rate}%)")
    if run_mf:
        log_kv("统计", "MF 成功", f"{mf_success_count}/{max_iterations} ({mf_rate}%)")
    log_kv("统计", "结果归档目录", run_dir)
    log_step("开始自动分析实验结果")
    try:
        from utils.analyze.task2_result import analyze_task2_run

        analysis = analyze_task2_run(
            output_root=Path(Config.PERSIST_ROOT).resolve(),
            run_dir=run_dir,
            model_name=",".join(Path(path).stem for path in model_paths),
            planned_iterations=max_iterations,
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
    log_kv("概览", "结束时间", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    if cluster is not None and cluster_session_id:
        cluster.cleanup_session(cluster_session_id)
    shutil.rmtree(shared_weight_tmp_run_dir, ignore_errors=True)
    return 0
