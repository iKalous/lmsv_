#!/usr/bin/env python3

from __future__ import annotations

import datetime
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
import tempfile
import utils
from utils.runtime.auto_dataset import ensure_task1_data_path


REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "config.json"
CONFIG_EXAMPLE_PATH = REPO_ROOT / "config.json.example"


def _normalize_output_root(value: str | Path) -> Path:
    output_root = Path(value).expanduser()
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()
    return output_root


def _can_create_directory(path: Path) -> bool:
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return os.access(probe, os.W_OK | os.X_OK)


NETWORK_FSTYPES = {"nfs", "nfs4", "cifs", "smbfs", "fuse.sshfs", "sshfs", "glusterfs"}


def _path_fstype(path: Path) -> str:
    try:
        result = subprocess.run(
            ["findmnt", "-T", str(path), "-n", "-o", "FSTYPE"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip().splitlines()[0].strip().lower() if result.stdout.strip() else ""


def _is_network_path(path: Path) -> bool:
    return _path_fstype(path) in NETWORK_FSTYPES


def _local_scratch_root() -> Path:
    raw_value = os.environ.get("LMSV_LOCAL_SCRATCH_ROOT", "").strip()
    if raw_value:
        return Path(raw_value).expanduser()
    return Path(tempfile.gettempdir()) / "lmsv_rec"


def _resolve_output_root() -> Path:
    raw_value = os.environ.get("LMSV_OUTPUT_ROOT", "").strip()
    candidates = []
    if raw_value:
        candidates.append(_normalize_output_root(raw_value))

    local_output = _local_scratch_root() / "output"
    repo_output = REPO_ROOT / "output"
    if _is_network_path(REPO_ROOT):
        candidates.extend([local_output, repo_output])
    else:
        candidates.extend([repo_output, local_output])

    for candidate in candidates:
        if _can_create_directory(candidate):
            return candidate

    return candidates[-1]


TASK_LABELS = {
    1: "整网泛化变异测试",
    2: "模块内组件泛化测试",
    3: "模块间泛化组合变异测试",
    4: "【多模态模型】模块间泛化组合变异测试",
    5: "【多模态模型】模块内组件泛化测试",
    6: "【多模态模型】整网泛化变异测试"
}


def _main_log(tag, message):
    text = str(message)
    if tag:
        return f"[主控][{tag}] {text}"
    return f"[主控] {text}"


def _handle_sigint(_signum, _frame):
    print("\n[do] 已中断。", flush=True)
    raise SystemExit(130)


signal.signal(signal.SIGINT, _handle_sigint)


def ensure_config() -> None:
    if CONFIG_PATH.exists():
        return
    print("config.json不存在，正在使用示例创建...")
    shutil.copy(CONFIG_EXAMPLE_PATH, CONFIG_PATH)


def create_output_dir() -> Path:
    output_root = _resolve_output_root()
    for attempt in range(5):
        suffix = "" if attempt == 0 else f"-{attempt}"
        output_dir = output_root / f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}{suffix}"
        try:
            output_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            if attempt == 4:
                utils.log.write.error(_main_log(None, "创建输出目录失败"))
                raise SystemExit(1)
    else:
        raise SystemExit(1)

    shutil.copy(CONFIG_PATH, output_dir / "config.json")
    (output_dir / "log.txt").write_text("", encoding="utf-8")

    os.environ["LMSV_LOGPATH"] = str(output_dir / "log.txt")
    os.environ["LMSV_OUTPATH"] = str(output_dir)
    return output_dir


def configure_local_scratch_defaults() -> None:
    if not _is_network_path(REPO_ROOT):
        return

    scratch_root = _local_scratch_root()
    if not os.environ.get("LMSV_PROJECT_TMP_ROOT"):
        os.environ["LMSV_PROJECT_TMP_ROOT"] = str((scratch_root / "tmp").resolve())
    if not os.environ.get("TMPDIR"):
        os.environ["TMPDIR"] = str((scratch_root / "python_tmp").resolve())
    if not os.environ.get("TMP"):
        os.environ["TMP"] = os.environ["TMPDIR"]
    if not os.environ.get("TEMP"):
        os.environ["TEMP"] = os.environ["TMPDIR"]

    for env_name in ("LMSV_PROJECT_TMP_ROOT", "TMPDIR"):
        Path(os.environ[env_name]).mkdir(parents=True, exist_ok=True)


def check_other_instances() -> None:
    current_pid = os.getpid()
    result = subprocess.run(
        ["ps", "-eo", "pid=,comm=,args="],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        utils.log.write.error(_main_log(None, "检查进程失败"))
        raise SystemExit(1)

    other_pids = []
    current_cwd = str(Path(__file__).resolve().parent)
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid_text, comm, args = parts
        if not pid_text.isdigit():
            continue
        pid = int(pid_text)
        if pid == current_pid:
            continue

        # 只识别真正执行 do.py 的 Python 进程，避免误伤 conda run 包装进程。
        if not comm.startswith("python"):
            continue
        if "do.py" not in args:
            continue
        # 排除不同路径下的 do.py 实例（如 /data/yd/lm-sv/lmsv_rec/do.py）
        try:
            pid_cwd = str(Path(f"/proc/{pid}/cwd").resolve())
            if current_cwd not in pid_cwd:
                continue
        except (OSError, ValueError):
            continue
        other_pids.append(pid)

    if other_pids:
        utils.log.write.error(_main_log("进程", f"发现其他 do.py 进程正在运行: {other_pids}"))
        raise SystemExit(1)


def _resolve_mindspeed_mm_path(value: str) -> str:
    """If value points to workspace root, append /MindSpeed-MM."""
    from pathlib import Path

    p = Path(value).expanduser()
    mm_sub = p / "MindSpeed-MM"
    if mm_sub.is_dir():
        return str(mm_sub)
    return value


def export_runtime_env(config: dict) -> None:
    env_map = {
        "MINDSPEED_MM_PATH": ("MINDSPEED_MM_PATH", "MindSpeed-MM路径"),
        "MSA_PATH": ("MSAPATH", "MSA路径"),
        "PTA_PATH": ("PTAPATH", "PTA路径"),
        "MSA_NAME": ("MSANAME", "MSA的conda环境名称"),
        "PTA_NAME": ("PTANAME", "PTA的conda环境名称"),
        "MF_NAME": ("MFNAME", "MF的conda环境名称"),
        "SAVE_ABNORMAL_WEIGHTS": ("SAVE_ABNORMAL_WEIGHTS", "异常迭代权重备份"),
        "DATASET_ROOT": ("DATASET_ROOT", "数据集根目录"),
    }
    for config_key, (legacy_env_key, label) in env_map.items():
        value = config.get(config_key)
        if value is None:
            continue
        value = str(value)
        if config_key == "MINDSPEED_MM_PATH":
            value = _resolve_mindspeed_mm_path(value)
        os.environ[legacy_env_key] = value
        os.environ[config_key] = value


def run_task(task_type: int, config: dict) -> None:
    task_params = (config.get("tasks") or {}).get(str(task_type))
    if task_params is None:
        raise ValueError(f"配置文件存在异常：task_type {task_type} 缺少 tasks.{task_type} 配置")
    task_params = dict(task_params)
    cluster_config = config.get("CLUSTER")
    if isinstance(cluster_config, dict) and task_type in (1, 2, 3):
        task_params["CLUSTER"] = dict(cluster_config)
    multi_node_config = config.get("MULTI_NODE")
    if isinstance(multi_node_config, dict) and (
        task_type == 6 or (task_type in (1, 2, 3, 4, 5) and "MULTI_NODE" not in task_params)
    ):
        task_params["MULTI_NODE"] = dict(multi_node_config)
    if "SAVE_ABNORMAL_WEIGHTS" in config:
        task_params["SAVE_ABNORMAL_WEIGHTS"] = config["SAVE_ABNORMAL_WEIGHTS"]
    if "SHARED_WEIGHT_TMP_ROOT" not in task_params and os.environ.get("LMSV_PROJECT_TMP_ROOT"):
        task_params["SHARED_WEIGHT_TMP_ROOT"] = str(
            Path(os.environ["LMSV_PROJECT_TMP_ROOT"]) / f"task{task_type}" / "shared_weight"
        )
    if task_type == 1:
        task_params["DATA_PATH"] = os.environ["DATA_PATH"]

    utils.log.write.info(_main_log("任务", f"开始执行 {TASK_LABELS[task_type]}"))
    utils.control.protect.task(task_type, task_params)


def main() -> int:
    ensure_config()
    check_other_instances()
    configure_local_scratch_defaults()
    create_output_dir()

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    task_type = config.get("task_type")
    if task_type not in TASK_LABELS:
        raise ValueError(f"配置文件存在异常：Unknown task_type: {task_type}")

    # 先导出运行时环境，再执行 task1 的自动数据准备，避免预处理阶段读不到 PTA/MSA 配置。
    export_runtime_env(config)

    if task_type == 1:
        prepared_data_path = ensure_task1_data_path(
            config,
            REPO_ROOT,
            lambda message: utils.log.write.info(_main_log("数据", message)),
            lambda message: utils.log.write.error(_main_log("数据", message)),
        )
        os.environ["DATA_PATH"] = prepared_data_path
        os.environ["DATAPATH"] = prepared_data_path

    run_task(task_type, config)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        utils.log.write.exception(_main_log("异常", "任务执行失败"), exc, default_component="主控调度")
        sys.exit(1)
