import os, shlex, json, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing_extensions import runtime

import utils
from utils.task import log_helpers
from utils.task import runtime_helpers
from utils.task import data_helpers

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LMSV_ROOT = Path(__file__).resolve().parents[2]
MM_MUTATE_ROOT = Path(__file__).resolve().parents[3] / "module_combination_mutation"
PROJECT_TMP_ROOT = LMSV_ROOT / "tmp"
TASK4_TMP_ROOT = PROJECT_TMP_ROOT / "task4"

class Config:
    MODE = "DEVELOP"
    TOTAL_ITER = 5
    COMPARE_MODE = "pta_msa"
    SAVE_STEPS = 1
    RUN_STEPS = 20

    RUN_MSA = False
    RUN_MF = False
    PTA_ENV = "mindspeed"
    MSA_ENV = "msadapter"
    MF_ENV = "mindf_py311"
    PTA_PATH = ""
    MSA_PATH = ""
    ENABLE_MF_WEIGHT_LOAD = False
    PTA_MAX_RUNTIME = 3000
    MSA_MAX_RUNTIME = 3000
    LOG_INIT_WAIT = 240
    LOG_STABLE_THRESHOLD = 150

    LOG_PATH = "res/execution.log"
    MSA_MONITOR_LOG = "msrun_log/worker_0.log"
    PTA_CSV_PATH = "res/execution_pta.csv"
    MSA_CSV_PATH = "res/execution_msa.csv"
    PERSIST_ROOT = ""
    SHARED_WEIGHT_TMP_ROOT = str(TASK4_TMP_ROOT / "shared_weight")

    ITER_RESULT_DIR = ""
    MULTI_NODE_ENABLED = False
    MASTER_ADDR = "127.0.0.1"
    NNODES = 1
    OTHER_NODES = []
    SSH_BIN = "ssh"
    RSYNC_BIN = "rsync"




def configure_project_tmp_env():
    return runtime_helpers.configure_project_tmp_env(PROJECT_TMP_ROOT)

def _normalize_compare_mode(value):
    text = str(value or "").strip().lower()
    aliases = {
        "pta_msa": "pta_msa",
        "pta-msa": "pta_msa",
        "msa": "pta_msa",
    }
    return aliases.get(text, "")

def _resolve_compare_mode(params):
    raw_mode = params.get("COMPARE_MODE")
    resolved = _normalize_compare_mode(raw_mode)
    if resolved:
        return resolved

    fallback = _normalize_compare_mode(Config.COMPARE_MODE)
    return fallback or "pta_msa"

def _init_config(params):
    Config.MODE = str(params.get("MODE", Config.MODE)).upper()
    Config.TOTAL_ITER = int(params.get("TOTAL_ITER", params.get("TITAL_ITER", Config.TOTAL_ITER)))
    Config.SAVE_STEPS = int(params.get("SAVE_STEPS", Config.SAVE_STEPS))
    Config.RUN_STEPS = int(params.get("RUN_STEPS", Config.RUN_STEPS))
    Config.PTA_MAX_RUNTIME = int(params.get("PTA_MAX_RUNTIME", Config.PTA_MAX_RUNTIME))
    Config.MSA_MAX_RUNTIME = int(params.get("MSA_MAX_RUNTIME", params.get("MAX_VALIDATE_TIME", Config.MSA_MAX_RUNTIME)))
    Config.LOG_INIT_WAIT = int(params.get("LOG_INIT_WAIT", Config.LOG_INIT_WAIT))
    Config.LOG_STABLE_THRESHOLD = int(params.get("LOG_STABLE_THRESHOLD", Config.LOG_STABLE_THRESHOLD))

    compare_mode = _resolve_compare_mode(params)
    run_msa = compare_mode == "pta_msa"
    run_mf = False
    if compare_mode != "pta_msa":
        log_helpers.log_warn(f"COMPARE_MODE 非法，已回退到 pta_msa: {params.get('COMPARE_MODE')}")
        compare_mode = "pta_msa"
        run_msa = True
        run_mf = False
    Config.COMPARE_MODE = compare_mode
    Config.RUN_MSA = run_msa
    Config.RUN_MF = run_mf

    pta_path = os.environ.get("PTA_PATH") or os.environ.get("PTAPATH")
    msa_path = os.environ.get("MSA_PATH") or os.environ.get("MSAPATH")
    if not pta_path:
        log_helpers.log_error("环境变量缺失：请先配置 PTA_PATH")
        return 1
    if run_msa and not msa_path:
        log_helpers.log_error("当前为 pta_msa 模式，环境变量缺失：请先配置 MSA_PATH")
        return 1
    Config.PTA_PATH = str(pta_path)
    Config.MSA_PATH = str(msa_path)

    Config.PTA_ENV = str(params.get("PTA_ENV", os.environ.get("PTA_NAME", Config.PTA_ENV)))
    Config.MSA_ENV = str(params.get("MSA_ENV", os.environ.get("MSA_NAME", Config.MSA_ENV)))
    Config.MF_ENV = str(params.get("MF_ENV", os.environ.get("MF_NAME", Config.MF_ENV)))
    Config.ENABLE_MF_WEIGHT_LOAD = data_helpers.parse_bool(
        params.get("ENABLE_MF_WEIGHT_LOAD", Config.ENABLE_MF_WEIGHT_LOAD)
    )

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
    Config.ITER_RESULT_DIR = str(persist_root_path / "iters")

    raw_tmp_root = str(params.get("SHARED_WEIGHT_TMP_ROOT", Config.SHARED_WEIGHT_TMP_ROOT))
    tmp_root_path = Path(raw_tmp_root).expanduser()
    if not tmp_root_path.is_absolute():
        tmp_root_path = LMSV_ROOT / tmp_root_path
    Config.SHARED_WEIGHT_TMP_ROOT = str(tmp_root_path.resolve())

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
                try:
                    ssh_port = int(raw_node.get("SSH_PORT", 22))
                except (TypeError, ValueError):
                    log_helpers.log_error(f"第{idx + 2}个节点 SSH_PORT 非法")
                    return 1
                if ssh_port <= 0:
                    log_helpers.log_error(f"第{idx + 2}个节点 SSH_PORT 必须大于0")
                    return 1
                lmsv_path = str(raw_node.get("LMSV_PATH", "")).strip()
                pta_name = str(raw_node.get("PTA_NAME", "")).strip()
                msa_name = str(raw_node.get("MSA_NAME", "")).strip()
                pta_path = str(raw_node.get("PTA_PATH", "")).strip()
                msa_path = str(raw_node.get("MSA_PATH", "")).strip()
                has_container = data_helpers.parse_bool(raw_node.get("HAS_CONTAINER", False))
                container_name = str(raw_node.get("CONTAINER_NAME", "")).strip()
                if has_container and not container_name:
                    log_helpers.log_error(f"第{idx + 2}个节点启用了容器，但缺少 CONTAINER_NAME")
                    return 1
                if not all([host, lmsv_path, pta_name, msa_name, pta_path, msa_path]):
                    log_helpers.log_error(f"第{idx + 2}个节点配置不完整，请检查 HOST/LMSV_PATH/PTA_NAME/MSA_NAME/PTA_PATH/MSA_PATH")
                    return 1
                normalized_nodes.append(
                    {
                        "HOST": host,
                        "SSH_PORT": ssh_port,
                        "LMSV_PATH": lmsv_path,
                        "PTA_NAME": pta_name,
                        "MSA_NAME": msa_name,
                        "PTA_PATH": pta_path,
                        "MSA_PATH": msa_path,
                        "HAS_CONTAINER": has_container,
                        "CONTAINER_NAME": container_name,
                        "NODE_RANK": len(normalized_nodes) + 1,
                    }
                )

        if not normalized_nodes:
            log_helpers.log_error("MULTI_NODE.ENABLED=true 时，必须至少配置一个 OTHER_NODES 节点")
            return 1

        Config.OTHER_NODES = normalized_nodes
        Config.NNODES = len(normalized_nodes) + 1
        raw_nnodes = raw_multi.get("NNODES")
        try:
            parsed_nnodes = int(raw_nnodes)
            if parsed_nnodes != Config.NNODES:
                log_helpers.log_warn(f"MULTI_NODE.NNODES={parsed_nnodes} 与 OTHER_NODES 数量不一致，已自动修正为 {Config.NNODES}")
        except (TypeError, ValueError):
            pass
        resolved_ssh = shutil.which(Config.SSH_BIN)
        if not resolved_ssh:
            log_helpers.log_error(
                f"多机模式缺少 SSH 客户端命令：{Config.SSH_BIN}。"
                "请安装 ssh（如 openssh-client），或通过环境变量 LMSV_SSH_BIN 指定可执行路径。"
            )
            return 1
        Config.SSH_BIN = resolved_ssh

        resolved_rsync = shutil.which(Config.RSYNC_BIN)
        if not resolved_rsync:
            log_helpers.log_error(
                f"多机模式缺少 rsync 命令：{Config.RSYNC_BIN}。"
                "请安装 rsync，或通过环境变量 LMSV_RSYNC_BIN 指定可执行路径。"
            )
            return 1
        Config.RSYNC_BIN = resolved_rsync
        Config.MULTI_NODE_ENABLED = True

    log_helpers.log_step("配置初始化完成")
    log_helpers.log_kv("配置", "模式", Config.MODE)
    log_helpers.log_kv("配置", "迭代次数", Config.TOTAL_ITER)
    log_helpers.log_kv("配置", "对比模式", Config.COMPARE_MODE)
    log_helpers.log_kv("配置", "保存步数", Config.SAVE_STEPS)
    log_helpers.log_kv("配置", "PTA环境", Config.PTA_ENV)
    log_helpers.log_kv("配置", "MSA环境", Config.MSA_ENV)
    log_helpers.log_kv("配置", "MF环境", Config.MF_ENV)
    log_helpers.log_kv("配置", "启用MF权重加载", Config.ENABLE_MF_WEIGHT_LOAD)
    log_helpers.log_kv("配置", "持久化根目录", Config.PERSIST_ROOT)
    log_helpers.log_kv("配置", "临时权重根目录", Config.SHARED_WEIGHT_TMP_ROOT)
    log_helpers.log_kv("配置", "多机启动", Config.MULTI_NODE_ENABLED)
    if Config.MULTI_NODE_ENABLED:
        log_helpers.log_kv("配置", "主节点地址", Config.MASTER_ADDR)
        log_helpers.log_kv("配置", "节点总数", Config.NNODES)
        log_helpers.log_kv("配置", "从节点数量", len(Config.OTHER_NODES))
        log_helpers.log_kv("配置", "SSH命令", Config.SSH_BIN)
        log_helpers.log_kv("配置", "RSYNC命令", Config.RSYNC_BIN)


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
    try:
        rel = local_abs.relative_to(PROJECT_ROOT)
    except ValueError:
        raise ValueError(f"路径不在项目目录内，无法同步到远端：{local_abs}") from None
    return (Path(node["LMSV_PATH"]).expanduser() / rel).as_posix()


def _remote_lmsv_root(node):
    rel = LMSV_ROOT.relative_to(PROJECT_ROOT)
    return (Path(node["LMSV_PATH"]).expanduser() / rel).as_posix()


def _remote_mm_mutate_root(node):
    rel = MM_MUTATE_ROOT.relative_to(PROJECT_ROOT)
    return (Path(node["LMSV_PATH"]).expanduser() / rel).as_posix()


def _run_remote_shell(node, shell_body, log_file, timeout, timeout_label):
    remote_body = f"set -e -o pipefail\n{shell_body}"
    if node.get("HAS_CONTAINER"):
        container_name = str(node.get("CONTAINER_NAME", "")).strip()
        if not container_name:
            log_helpers.log_error(f"[{node.get('HOST')}] 缺少容器名，无法执行远程命令")
            return False
        remote_cmd = f"docker exec {shlex.quote(container_name)} bash -lc {shlex.quote(remote_body)}"
    else:
        remote_cmd = f"bash -lc {shlex.quote(remote_body)}"

    ssh_port = int(node.get("SSH_PORT", 22))
    ssh_cmd = (
        f"{shlex.quote(Config.SSH_BIN)} -p {ssh_port} -o BatchMode=yes -o StrictHostKeyChecking=no "
        f"{shlex.quote(str(node['HOST']))} {shlex.quote(remote_cmd)}"
    )
    result = runtime_helpers.run_shell_to_file(
        ssh_cmd,
        log_file,
        LMSV_ROOT,
        log_helpers.log_error,
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
    for node in Config.OTHER_NODES:
        node_rank = int(node["NODE_RANK"])
        sync_log = os.path.join(log_dir, f"sync_iter{iter_num}_node{node_rank}.log")
        try:
            remote_iter_dir = _local_to_remote_path(local_iter_dir, node)
        except ValueError as exc:
            log_helpers.log_error(str(exc))
            failed_logs.append(sync_log)
            continue
        remote_parent = Path(remote_iter_dir).parent.as_posix()
        host = str(node["HOST"])
        ssh_port = int(node.get("SSH_PORT", 22))
        mkdir_cmd = (
            f"{shlex.quote(Config.SSH_BIN)} -p {ssh_port} -o BatchMode=yes -o StrictHostKeyChecking=no "
            f"{shlex.quote(host)} {shlex.quote(f'mkdir -p {shlex.quote(remote_parent)}')}"
        )
        rsync_cmd = (
            f"{shlex.quote(Config.RSYNC_BIN)} -az --delete "
            f"-e \"{shlex.quote(Config.SSH_BIN)} -p {ssh_port} -o BatchMode=yes -o StrictHostKeyChecking=no\" "
            f"{shlex.quote(local_iter_dir.as_posix() + '/')} "
            f"{shlex.quote(f'{host}:{remote_iter_dir}/')}"
        )
        cmd = f"{mkdir_cmd} && {rsync_cmd}"
        result = runtime_helpers.run_shell_to_file(
            cmd,
            sync_log,
            LMSV_ROOT,
            log_helpers.log_error,
            check=False,
            timeout=Config.PTA_MAX_RUNTIME,
            timeout_label="远端目录同步",
        )
        if result is None or result.returncode != 0:
            failed_logs.append(sync_log)
            continue
        log_helpers.log_info(f"[多机] 迭代{iter_num}输入目录已同步到节点{node_rank}：{host}")
    return len(failed_logs) == 0, failed_logs

def run_pta_mutate(res_dir, log_file):
    args = [
        "--rounds 1",
        f"--results-dir {shlex.quote(res_dir)}",
        f"--dir-name 1-pta-mutate"
    ]
    cmd = f"""
    {runtime_helpers.build_conda_activate_block(Config.PTA_ENV, load_ascend=True)}
    export PTAPATH={shlex.quote(Config.PTA_PATH)}
    source scripts/envset/mm-pta.sh
    bash {shlex.quote(f"{MM_MUTATE_ROOT}/mm_mutate.sh")} {' '.join(args)}
    """
    result = runtime_helpers.run_shell_to_file(
        cmd,
        log_file,
        LMSV_ROOT,
        log_helpers.log_error,
        check=False,
        timeout=Config.PTA_MAX_RUNTIME,
        timeout_label="PTA执行",
    )
    return result is not None and result.returncode == 0

def post_pta_mutate(pta_mutate_dir):
    """删除 pta-mutate 目录下可再生的中间产物：configs、mutate_* 目录、mm_mutate.log。"""
    root = Path(pta_mutate_dir)
    if not root.is_dir():
        return
    profile = root / "profile"
    if profile.is_dir():
        shutil.rmtree(profile, ignore_errors=True)
    
    mm_log = root / "train.log"
    if mm_log.is_file():
        try:
            mm_log.unlink()
        except OSError:
            pass

def _init_workspace():
    """初始化Task4结果目录。"""
    log_helpers.log_step("初始化Task4结果目录")

    if Config.ITER_RESULT_DIR:
        os.makedirs(Config.ITER_RESULT_DIR, exist_ok=True)
    else:
        log_helpers.log_error("ITER_RESULT_DIR 为空，请先配置")
        return 1

def write_iteration_status(
    iter_num,
    iter_dir,
    overall_status,
    reason="",
    *,
    mutate_result="SKIP",
    pta_save_result="SKIP",
    pta_load_result="SKIP",
    msa_load_result="SKIP",
    mf_result="SKIP",
):
    iter_dir = Path(iter_dir)
    iter_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "task_type": 4,
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
    
def run_pta_save(config_dir,res_dir, log_file):
    args = [
        f"--config {shlex.quote(str(config_dir))}",
        f"--iterations {int(Config.SAVE_STEPS)}",
        f"--results-dir {shlex.quote(str(res_dir))}",
        "--save-ckpt",
    ]
    cmd = f"""
    {runtime_helpers.build_conda_activate_block(Config.PTA_ENV, load_ascend=True)}
    export PTAPATH={shlex.quote(Config.PTA_PATH)}
    source scripts/envset/mm-pta.sh
    bash {shlex.quote(f"{MM_MUTATE_ROOT}/mm_test.sh")} {' '.join(args)}
    """
    result = runtime_helpers.run_shell_to_file(
        cmd,
        log_file,
        LMSV_ROOT,
        log_helpers.log_error,
        check=False,
        timeout=Config.PTA_MAX_RUNTIME,
        timeout_label="PTA执行",
    )
    return result is not None and result.returncode == 0

def post_pta_save(pta_save_dir):
    """删除 pta-save 目录下可再生的中间产物：configs、mutate_* 目录、mm_test.log。"""
    root = Path(pta_save_dir)
    if not root.is_dir():
        return
    configs = root / "configs"
    if configs.is_dir():
        shutil.rmtree(configs, ignore_errors=True)
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith("mutate_"):
            shutil.rmtree(child, ignore_errors=True)
    mm_log = root / "mm_test.log"
    if mm_log.is_file():
        try:
            mm_log.unlink()
        except OSError:
            pass

def run_pta_run(config_dir, res_dir, ckpt_path, log_file, distributed_args=None):
    distributed_args = distributed_args or []
    common_args = [
        f"--config {shlex.quote(str(config_dir))}",
        f"--iterations {int(Config.RUN_STEPS)}",
        f"--results-dir {shlex.quote(str(res_dir))}",
        "--load-ckpt",
        f"--ckpt {shlex.quote(str(ckpt_path))}",
    ]
    cmd = f"""
    {runtime_helpers.build_conda_activate_block(Config.PTA_ENV, load_ascend=True)}
    export PTAPATH={shlex.quote(Config.PTA_PATH)}
    source scripts/envset/mm-pta.sh
    bash {shlex.quote(f"{MM_MUTATE_ROOT}/mm_test.sh")} {' '.join(distributed_args)} {' '.join(common_args)}
    """
    result = runtime_helpers.run_shell_to_file(
        cmd,
        log_file,
        LMSV_ROOT,
        log_helpers.log_error,
        check=False,
        timeout=Config.PTA_MAX_RUNTIME,
        timeout_label="PTA执行",
    )
    return result is not None and result.returncode == 0

def post_pta_run(pta_run_dir):
    """将 mm_test 在 mutate_* 子目录下生成的 runtime_info.csv 提升到 3-pta-run 根目录，删除其余内容。"""
    root = Path(pta_run_dir)
    if not root.is_dir():
        return
    nested = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and child.name.startswith("mutate_"):
            csv_path = child / "runtime_info.csv"
            if csv_path.is_file():
                nested.append((child, csv_path))
    kept = set()
    if len(nested) == 1:
        _, src = nested[0]
        shutil.move(str(src), str(root / "runtime_info.csv"))
        kept.add("runtime_info.csv")
    elif len(nested) > 1:
        for mutate_dir, src in nested:
            dst = root / f"runtime_info_{mutate_dir.name}.csv"
            shutil.move(str(src), str(dst))
            kept.add(dst.name)
    elif (root / "runtime_info.csv").is_file():
        kept.add("runtime_info.csv")
    for child in list(root.iterdir()):
        if child.name == "repro.log":
            continue
        if child.name in kept:
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            try:
                child.unlink()
            except OSError:
                pass

def run_msa_run(config_file, res_dir, ckpt_path, log_file, distributed_args=None):
    distributed_args = distributed_args or []
    msrun_log_dir = os.path.join(res_dir, "msrun_log")
    config_dir = os.path.abspath(os.path.dirname(os.path.expanduser(str(config_file))))
    common_args = [
        f"--config-dir {shlex.quote(config_dir)}",
        f"--config {shlex.quote(str(config_file))}",
        f"--iterations {int(Config.RUN_STEPS)}",
        f"--results-dir {shlex.quote(str(res_dir))}",
        "--load-ckpt",
        f"--ckpt {shlex.quote(str(ckpt_path))}",
        f"--msrun-log-dir {shlex.quote(str(msrun_log_dir))}",
    ]
    cmd = f"""
    {runtime_helpers.build_conda_activate_block(Config.MSA_ENV, load_ascend=True)}
    export MSAPATH={shlex.quote(Config.MSA_PATH)}
    source scripts/envset/mm-msa.sh
    bash {shlex.quote(f"{MM_MUTATE_ROOT}/ms_mm_test.sh")} {' '.join(distributed_args)} {' '.join(common_args)}
    """
    result = runtime_helpers.run_shell_to_file(
        cmd,
        log_file,
        LMSV_ROOT,
        log_helpers.log_error,
        check=False,
        timeout=Config.MSA_MAX_RUNTIME,
        timeout_label="MSA执行",
    )
    return result is not None and result.returncode == 0


def run_remote_pta_run(node, config_file, res_dir, ckpt_path, log_file):
    remote_config = _local_to_remote_path(config_file, node)
    remote_res_dir = _local_to_remote_path(res_dir, node)
    remote_ckpt = _local_to_remote_path(ckpt_path, node)
    distributed_args = _build_distributed_args(int(node["NODE_RANK"]))
    common_args = [
        f"--config {shlex.quote(remote_config)}",
        f"--iterations {int(Config.RUN_STEPS)}",
        f"--results-dir {shlex.quote(remote_res_dir)}",
        "--load-ckpt",
        f"--ckpt {shlex.quote(remote_ckpt)}",
    ]
    shell_body = f"""
    {runtime_helpers.build_conda_activate_block(node["PTA_NAME"], load_ascend=True)}
    export PTAPATH={shlex.quote(node["PTA_PATH"])}
    cd {shlex.quote(_remote_lmsv_root(node))}
    source scripts/envset/mm-pta.sh
    bash {shlex.quote(f"{_remote_mm_mutate_root(node)}/mm_test.sh")} {' '.join(distributed_args)} {' '.join(common_args)}
    """
    return _run_remote_shell(
        node,
        shell_body,
        log_file,
        timeout=Config.PTA_MAX_RUNTIME,
        timeout_label="远端PTA执行",
    )


def run_remote_msa_run(node, config_file, res_dir, ckpt_path, log_file):
    remote_config = _local_to_remote_path(config_file, node)
    remote_res_dir = _local_to_remote_path(res_dir, node)
    remote_ckpt = _local_to_remote_path(ckpt_path, node)
    remote_msrun_log = f"{remote_res_dir}/msrun_log"
    remote_config_dir = os.path.dirname(remote_config)
    distributed_args = _build_distributed_args(int(node["NODE_RANK"]))
    common_args = [
        f"--config-dir {shlex.quote(remote_config_dir)}",
        f"--config {shlex.quote(remote_config)}",
        f"--iterations {int(Config.RUN_STEPS)}",
        f"--results-dir {shlex.quote(remote_res_dir)}",
        "--load-ckpt",
        f"--ckpt {shlex.quote(remote_ckpt)}",
        f"--msrun-log-dir {shlex.quote(remote_msrun_log)}",
    ]
    shell_body = f"""
    {runtime_helpers.build_conda_activate_block(node["MSA_NAME"], load_ascend=True)}
    export MSAPATH={shlex.quote(node["MSA_PATH"])}
    cd {shlex.quote(_remote_lmsv_root(node))}
    source scripts/envset/mm-msa.sh
    bash {shlex.quote(f"{_remote_mm_mutate_root(node)}/ms_mm_test.sh")} {' '.join(distributed_args)} {' '.join(common_args)}
    """
    return _run_remote_shell(
        node,
        shell_body,
        log_file,
        timeout=Config.MSA_MAX_RUNTIME,
        timeout_label="远端MSA执行",
    )


def run_pta_run_multinode(config_file, res_dir, ckpt_path, local_log_file, iter_log_result_dir, iter_idx):
    if not Config.MULTI_NODE_ENABLED:
        return run_pta_run(config_file, res_dir, ckpt_path, local_log_file, distributed_args=[])

    jobs = []
    with ThreadPoolExecutor(max_workers=max(1, int(Config.NNODES))) as executor:
        local_dist_args = _build_distributed_args(0)
        jobs.append(
            executor.submit(
                run_pta_run,
                config_file,
                res_dir,
                ckpt_path,
                local_log_file,
                local_dist_args,
            )
        )
        for node in Config.OTHER_NODES:
            node_rank = int(node["NODE_RANK"])
            node_log = os.path.join(iter_log_result_dir, f"pta_load_iter{iter_idx}_node{node_rank}.log")
            jobs.append(
                executor.submit(
                    run_remote_pta_run,
                    node,
                    config_file,
                    res_dir,
                    ckpt_path,
                    node_log,
                )
            )
        all_ok = True
        for future in as_completed(jobs):
            try:
                ok = bool(future.result())
            except Exception as exc:
                log_helpers.log_error(f"PTA多机执行异常: {exc}")
                ok = False
            if not ok:
                all_ok = False
        return all_ok


def run_msa_run_multinode(config_file, res_dir, ckpt_path, local_log_file, iter_log_result_dir, iter_idx):
    if not Config.MULTI_NODE_ENABLED:
        return run_msa_run(config_file, res_dir, ckpt_path, local_log_file, distributed_args=[])

    jobs = []
    with ThreadPoolExecutor(max_workers=max(1, int(Config.NNODES))) as executor:
        local_dist_args = _build_distributed_args(0)
        jobs.append(
            executor.submit(
                run_msa_run,
                config_file,
                res_dir,
                ckpt_path,
                local_log_file,
                local_dist_args,
            )
        )
        for node in Config.OTHER_NODES:
            node_rank = int(node["NODE_RANK"])
            node_log = os.path.join(iter_log_result_dir, f"msa_load_iter{iter_idx}_node{node_rank}.log")
            jobs.append(
                executor.submit(
                    run_remote_msa_run,
                    node,
                    config_file,
                    res_dir,
                    ckpt_path,
                    node_log,
                )
            )
        all_ok = True
        for future in as_completed(jobs):
            try:
                ok = bool(future.result())
            except Exception as exc:
                log_helpers.log_error(f"MSA多机执行异常: {exc}")
                ok = False
            if not ok:
                all_ok = False
        return all_ok

def _print_task4_test_summary(mutate_failures, pta_failures, pta_ok_msa_failures):
    """全部迭代结束后打印：mutate 失败、PTA 失败、PTA 成功但 MSA 失败及对应日志路径。"""
    log_helpers.log_step("========== 全部迭代结束 · 测试信息汇总 ==========")

    log_helpers.log_info("  [1] Mutate 失败的迭代及日志路径")
    if not mutate_failures:
        log_helpers.log_info("      （无）")
    else:
        for it in mutate_failures:
            log_helpers.log_info(f"      迭代 {it['iter']}: {', '.join(it['paths'])}")

    log_helpers.log_info("  [2] PTA 失败的迭代及日志路径（Mutate 已成功）")
    if not pta_failures:
        log_helpers.log_info("      （无）")
    else:
        for it in pta_failures:
            log_helpers.log_info(f"      迭代 {it['iter']} ({it['stage']}): {', '.join(it['paths'])}")

    log_helpers.log_info("  [3] PTA 成功但 MSA 失败的迭代及日志路径")
    if not pta_ok_msa_failures:
        log_helpers.log_info("      （无）")
    else:
        for it in pta_ok_msa_failures:
            log_helpers.log_info(f"      迭代 {it['iter']}: {', '.join(it['paths'])}")


def post_msa_run(msa_run_dir, success=True):
    """成功时：将 mutate_*/runtime_info.csv 提升到 4-msa-run 根目录；保留 msrun_log/；删除其余内容。
    失败时：不处理 runtime_info，仅保留 msrun_log/，删除其余一切。"""
    root = Path(msa_run_dir)
    if not root.is_dir():
        return
    if not success:
        for child in list(root.iterdir()):
            if child.name == "msrun_log" and child.is_dir():
                continue
            if child.name == "repro.log":
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                try:
                    child.unlink()
                except OSError:
                    pass
        return
    nested = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and child.name.startswith("mutate_"):
            csv_path = child / "runtime_info.csv"
            if csv_path.is_file():
                nested.append((child, csv_path))
    kept = set()
    if len(nested) == 1:
        _, src = nested[0]
        shutil.move(str(src), str(root / "runtime_info.csv"))
        kept.add("runtime_info.csv")
    elif len(nested) > 1:
        for mutate_dir, src in nested:
            dst = root / f"runtime_info_{mutate_dir.name}.csv"
            shutil.move(str(src), str(dst))
            kept.add(dst.name)
    elif (root / "runtime_info.csv").is_file():
        kept.add("runtime_info.csv")
    for child in list(root.iterdir()):
        if child.name in kept:
            continue
        if child.name == "msrun_log" and child.is_dir():
            continue
        if child.name == "repro.log":
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            try:
                child.unlink()
            except OSError:
                pass

def main(params):
    log_helpers.LOG_SCOPE = "Task4"
    project_tmp_root = configure_project_tmp_env()
    utils.control.clean.kill_pretraingpt()

    init_ret = _init_config(params)
    if init_ret:
        return int(init_ret)
    _init_workspace()

    def refresh_iteration_analysis():
        try:
            from utils.analyze.task45_result import analyze_task_run

            analyze_task_run(
                output_root=Path(Config.PERSIST_ROOT).resolve(),
                run_dir=Path(Config.ITER_RESULT_DIR).resolve(),
                model_name=None,
                planned_iterations=Config.TOTAL_ITER,
                task_type=4
            )
        except Exception as exc:
            log_helpers.log_warn(f"迭代分析刷新失败，已跳过: {exc}")

    mutate_failed_num = 0
    pta_save_failed_num = 0
    pta_run_failed_num = 0
    msa_run_failed_num = 0
    mutate_failures = []
    pta_failures = []
    pta_ok_msa_failures = []
    for i in range(Config.TOTAL_ITER):
        log_helpers.log_step(f"【 🔁 第{i+1}轮迭代 】")

        # status init
        mutate_result = "SKIP"
        pta_save_result = "SKIP"
        pta_load_result = "SKIP"
        msa_load_result = "SKIP" if Config.RUN_MSA else "DISABLED"
        mf_result = "SKIP" if Config.RUN_MF else "DISABLED"

        # init iter result dir
        iter_result_dir = os.path.join(Config.ITER_RESULT_DIR, f"iter_{i+1}")
        os.makedirs(iter_result_dir, exist_ok=True)
        iter_core_backup_dir = os.path.join(iter_result_dir, "core_backup")
        os.makedirs(iter_core_backup_dir, exist_ok=True)
        iter_log_result_dir = os.path.join(iter_result_dir, "runtime_logs")
        os.makedirs(iter_log_result_dir, exist_ok=True)
        mutate_log = os.path.join(iter_log_result_dir, f"pta_mutate_iter{i+1}.log")
        pta_save_log = os.path.join(iter_log_result_dir, f"pta_save_iter{i+1}.log")
        pta_run_log = os.path.join(iter_log_result_dir, f"pta_load_iter{i+1}.log")
        msa_run_log = os.path.join(iter_log_result_dir, f"msa_load_iter{i+1}.log")

        def finalize_current_iter():
            overall_status = "PASS"
            if mutate_result == "ERROR":
                overall_status = "MUTATION_FAILED"
            elif any(
                stage == "ERROR"
                for stage in (
                    pta_save_result,
                    pta_load_result,
                    msa_load_result,
                    mf_result,
                )
            ):
                overall_status = "EXECUTION_FAILED"

            write_iteration_status(
                i+1,
                iter_result_dir,
                overall_status,
                "成功",
                mutate_result=mutate_result,
                pta_save_result=pta_save_result,
                pta_load_result=pta_load_result,
                msa_load_result=msa_load_result,
                mf_result=mf_result,
            )


        try:
            # 1 pta mutate
            log_helpers.log_step("1. 执行PTA侧模块间变异")
            mutate_ok = run_pta_mutate(iter_core_backup_dir, mutate_log)
            if not mutate_ok:
                log_helpers.log_error(f"第{i+1}轮 PTA侧模块间变异执行失败，跳过本轮")
                mutate_result = "ERROR"
                write_iteration_status(
                    i+1,
                    iter_result_dir,
                    "FAILED",
                    "PTA侧模块间变异执行失败",
                    mutate_result=mutate_result,
                    pta_save_result=pta_save_result,
                    pta_load_result=pta_load_result,
                    msa_load_result=msa_load_result,
                    mf_result=mf_result,
                )
                mutate_failed_num += 1
                mutate_failures.append({"iter": i + 1, "paths": [mutate_log]})
                continue
            mutate_result = "OK"
            mutated_config_file = os.path.join(iter_core_backup_dir, "1-pta-mutate", "configs", "round_0.json")
            mutated_config_dot_file = os.path.join(iter_core_backup_dir, "1-pta-mutate", "dots", "graph_round0.dot")
            if not os.path.exists(mutated_config_file):
                log_helpers.log_error(f"第{i+1}轮 PTA侧模块间变异未生成可加载JSON: {mutated_config_file}")
                mutate_result = "ERROR"
                write_iteration_status(
                    i+1,
                    iter_result_dir,
                    "FAILED",
                    "PTA侧模块间变异未生成可加载JSON",
                    mutate_result=mutate_result,
                    pta_save_result=pta_save_result,
                    pta_load_result=pta_load_result,
                    msa_load_result=msa_load_result,
                    mf_result=mf_result,
                )
                mutate_failed_num += 1
                mutate_failures.append({"iter": i + 1, "paths": [mutate_log]})
                continue
            post_pta_mutate(os.path.join(iter_core_backup_dir, "1-pta-mutate"))
            log_helpers.log_kv("产物", "多模态模块间变异生成配置", mutated_config_file)
            log_helpers.log_kv("产物", "多模态模块间变异生成DOT图", mutated_config_dot_file)

            # 2 pta save
            log_helpers.log_step("2. 执行PTA侧模型保存")
            pta_save_dir = os.path.join(iter_core_backup_dir, "2-pta-save")
            os.makedirs(pta_save_dir, exist_ok=True)

            pta_save_ok = run_pta_save(mutated_config_file, pta_save_dir, pta_save_log)
            if not pta_save_ok:
                log_helpers.log_error(f"第{i+1}轮 PTA侧模型保存执行失败，跳过本轮")
                pta_save_result = "ERROR"
                write_iteration_status(
                    i+1,
                    iter_result_dir,
                    "FAILED",
                    "PTA侧模型保存执行失败",
                    mutate_result=mutate_result,
                    pta_save_result=pta_save_result,
                    pta_load_result=pta_load_result,
                    msa_load_result=msa_load_result,
                    mf_result=mf_result,
                )
                pta_save_failed_num += 1
                pta_failures.append(
                    {"iter": i + 1, "stage": "pta_save", "paths": [pta_save_log]}
                )
                continue
            pta_save_result = "OK"
            post_pta_save(pta_save_dir)
            ckpt_path = os.path.join(pta_save_dir, "ckpts", "round_0.pt")
            log_helpers.log_kv("产物", "PTA侧模型保存CKPT", ckpt_path)

            if Config.MULTI_NODE_ENABLED:
                log_helpers.log_step("2.4 同步迭代输入目录到非主节点")
                sync_ok, _sync_failed_logs = sync_iteration_to_remote_nodes(iter_result_dir, iter_log_result_dir, i + 1)
                if not sync_ok:
                    log_helpers.log_error(f"第{i+1}轮 迭代输入目录同步失败，跳过本轮")
                    pta_load_result = "ERROR"
                    write_iteration_status(
                        i+1,
                        iter_result_dir,
                        "FAILED",
                        "迭代输入目录同步到非主节点失败",
                        mutate_result=mutate_result,
                        pta_save_result=pta_save_result,
                        pta_load_result=pta_load_result,
                        msa_load_result=msa_load_result,
                        mf_result=mf_result,
                    )
                    pta_run_failed_num += 1
                    pta_failures.append(
                        {
                            "iter": i + 1,
                            "stage": "sync_inputs",
                            "paths": [pta_save_log],
                        }
                    )
                    continue

            # 3 pta run
            log_helpers.log_step("3. 执行PTA侧模型训练")
            pta_run_dir = os.path.join(iter_core_backup_dir, "3-pta-run")
            os.makedirs(pta_run_dir, exist_ok=True)

            pta_run_ok = run_pta_run_multinode(
                mutated_config_file,
                pta_run_dir,
                ckpt_path,
                pta_run_log,
                iter_log_result_dir,
                i + 1,
            )
            if not pta_run_ok:
                log_helpers.log_error(f"第{i+1}轮 PTA侧模型训练执行失败，跳过本轮")
                pta_load_result = "ERROR"
                write_iteration_status(
                    i+1,
                    iter_result_dir,
                    "FAILED",
                    "PTA侧模型训练执行失败",
                    mutate_result=mutate_result,
                    pta_save_result=pta_save_result,
                    pta_load_result=pta_load_result,
                    msa_load_result=msa_load_result,
                    mf_result=mf_result,
                )
                pta_run_failed_num += 1
                pta_failures.append(
                    {
                        "iter": i + 1,
                        "stage": "pta_run",
                        "paths": [pta_save_log, pta_run_log],
                    }
                )
                continue
            pta_load_result = "OK"
            post_pta_run(pta_run_dir)
            runtime_info_csv = os.path.join(pta_run_dir, "runtime_info.csv")
            log_helpers.log_kv("产物", "PTA侧模型训练运行信息", runtime_info_csv)

            # 4 msa run
            log_helpers.log_step("4. 执行MSA侧模型训练")
            msa_run_dir = os.path.join(iter_core_backup_dir, "4-msa-run")
            os.makedirs(msa_run_dir, exist_ok=True)

            msa_run_ok = run_msa_run_multinode(
                mutated_config_file,
                msa_run_dir,
                ckpt_path,
                msa_run_log,
                iter_log_result_dir,
                i + 1,
            )
            if not msa_run_ok:
                log_helpers.log_error(f"第{i+1}轮 MSA侧模型训练执行失败，跳过本轮")
                msa_load_result = "ERROR"
                write_iteration_status(
                    i+1,
                    iter_result_dir,
                    "FAILED",
                    "MSA侧模型训练执行失败",
                    mutate_result=mutate_result,
                    pta_save_result=pta_save_result,
                    pta_load_result=pta_load_result,
                    msa_load_result=msa_load_result,
                    mf_result=mf_result,
                )
                msa_run_failed_num += 1
                _msa_paths = [msa_run_log]
                _worker_log = os.path.join(msa_run_dir, "msrun_log", "worker_0.log")
                if os.path.isfile(_worker_log):
                    _msa_paths.append(_worker_log)
                pta_ok_msa_failures.append({"iter": i + 1, "paths": _msa_paths})
                post_msa_run(msa_run_dir, success=False)
                continue
            msa_load_result = "OK"
            post_msa_run(msa_run_dir)
            runtime_info_csv = os.path.join(msa_run_dir, "runtime_info.csv")
            log_helpers.log_kv("产物", "MSA侧模型训练运行信息", runtime_info_csv)

            finalize_current_iter()
        except Exception as e:
            log_helpers.log_error(f"迭代{i}执行异常: {str(e)}")
            pta_result = "ERROR"
            msa_result = "ERROR"
            iter_reason = f"执行异常: {str(e)}"
            finalize_current_iter()
            continue
        finally:
            refresh_iteration_analysis()

    _print_task4_test_summary(mutate_failures, pta_failures, pta_ok_msa_failures)

    log_helpers.log_step("开始自动分析实验结果")
    try:
        from utils.analyze.task45_result import analyze_task_run

        analysis = analyze_task_run(
            output_root=Path(Config.PERSIST_ROOT).resolve(),
            run_dir=Path(Config.ITER_RESULT_DIR).resolve(),
            model_name=None,
            planned_iterations=Config.TOTAL_ITER,
            task_type=4
        )
        log_helpers.log_info(
            "实验结果分析完成 | 执行轮次: "
            f"{analysis.executed_iterations} | 变异成功: "
            f"{analysis.mutation_success_count}/{analysis.executed_iterations} "
            f"({analysis.mutation_success_rate * 100:.2f}%)"
        )
        log_helpers.log_info(
            "问题统计 | 功能: "
            f"{analysis.functional_failures} | 精度: {analysis.precision_failures} | "
            f"性能: {analysis.performance_failures} | 显存: {analysis.memory_failures}"
        )
        log_helpers.log_info(f"分析目录: {analysis.analysis_dir}")
        log_helpers.log_info(f"HTML报告: {analysis.report_html}")
        log_helpers.log_info(f"JSON汇总: {analysis.summary_json}")
        log_helpers.log_info(f"失败复现目录: {analysis.repro_root}")
    except Exception as exc:
        log_helpers.log_warn(f"自动分析失败，已跳过: {exc}")

    log_helpers.log_step(f"=============== 自动化变异+PTA/MSA训练流程 结束 ===============")
    log_helpers.log_kv("概览", "结束时间", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    log_helpers.log_step("任务结束")
    log_helpers.log_step(f"=============== 自动化变异+PTA/MSA训练流程 结束 ===============")
    
    return 0

