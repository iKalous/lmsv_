#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = REPO_ROOT / "output"
SUPPORTED_TASK_TYPES = {1, 2, 3, 4, 5}
DEFAULT_MF_ENV = "mindf_py311"
DEFAULT_MF_ARGS_PATH = "assets/runtime/mf_templates/basic.yaml"
DEFAULT_MSA_FOLLOW_TIMEOUT = 300
DEFAULT_MSA_STABLE_SECONDS = 150
MSA_FINISH_PATTERNS = (
    "pretrain finished",
    "training completed",
    "epoch done",
    "train end",
    "exit successfully",
)


def _handle_sigint(_signum, _frame) -> None:
    print("\n[repro] 已中断。", flush=True)
    raise SystemExit(130)


signal.signal(signal.SIGINT, _handle_sigint)


@dataclass(frozen=True)
class OutputEntry:
    path: Path
    task_type: int | None
    model_name: str
    iters: tuple[int, ...]


@dataclass(frozen=True)
class RunEntry:
    key: str
    label: str
    env_kind: str
    mode: str
    task_type: int
    script_path: Path | None = None
    synthetic: bool = False


@dataclass(frozen=True)
class StageInfo:
    res_dir: Path | None = None
    load_path_arg: str | None = None


@dataclass(frozen=True)
class ExecutionResult:
    return_code: int
    note: str
    session_dir: Path | None = None


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def path_has_payload(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return path.stat().st_size > 0
    try:
        next(path.iterdir())
        return True
    except StopIteration:
        return False


def reset_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def sanitize_pythonpath(env: dict[str, str]) -> None:
    current = env.get("PYTHONPATH", "")
    if not current:
        return
    replace_dir = str((REPO_ROOT / "utils" / "replace").resolve())
    kept = [item for item in current.split(":") if item and item != replace_dir]
    if kept:
        env["PYTHONPATH"] = ":".join(kept)
    else:
        env.pop("PYTHONPATH", None)


def build_conda_activate_block(env_name: str, load_ascend: bool = True) -> str:
    lines = [
        "CONDA_PATH=$(conda info --base 2>/dev/null)",
        'if [ -z "$CONDA_PATH" ]; then',
        '  echo "ERROR: conda base path not found" >&2',
        "  exit 1",
        "fi",
        'source "$CONDA_PATH/etc/profile.d/conda.sh"',
    ]
    if load_ascend:
        lines.extend(
            [
                'if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then',
                "  source /usr/local/Ascend/ascend-toolkit/set_env.sh",
                "fi",
            ]
        )
    if env_name:
        lines.append(f"conda activate {shlex.quote(env_name)}")
    return "\n".join(lines)


def extract_iteration(path: Path) -> int:
    match = re.search(r"(\d+)$", path.name)
    if not match:
        raise ValueError(f"无法解析轮次目录: {path}")
    return int(match.group(1))


def choose_one(items: list, title: str, render) -> object:
    while True:
        print()
        print(title)
        for index, item in enumerate(items, start=1):
            print(f"{index}. {render(item)}")
        choice = input("请输入编号，或输入 q 退出: ").strip().lower()
        if choice in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        if not choice.isdigit():
            print("输入无效，请重新输入。")
            continue
        index = int(choice)
        if 1 <= index <= len(items):
            return items[index - 1]
        print("编号超出范围，请重新输入。")


def normalize_models(raw_models) -> list[str]:
    if isinstance(raw_models, str):
        items = [part.strip() for part in raw_models.split(",") if part.strip()]
    elif isinstance(raw_models, (list, tuple)):
        items = [str(item).strip() for item in raw_models if str(item).strip()]
    else:
        return []

    paths = []
    for item in items:
        if item.endswith(".yaml"):
            model_path = item if "/" in item else f"assets/runtime/model_config/{item}"
        else:
            model_path = f"assets/runtime/model_config/{item}.yaml"
        paths.append(model_path)
    return paths


def infer_model_name(config: dict, task_type: int | None) -> str:
    task_conf = ((config.get("tasks") or {}).get(str(task_type)) or {}) if task_type else {}
    if task_type == 1:
        model_name = task_conf.get("MODEL_NAME")
        return str(model_name) if model_name else "unknown"
    if task_type in {2, 3}:
        models = normalize_models(task_conf.get("MODELS"))
        if models:
            return ",".join(Path(path).stem for path in models)
    return "unknown"


def discover_output_entries() -> list[OutputEntry]:
    if not OUTPUT_ROOT.exists():
        return []

    entries: list[OutputEntry] = []
    for output_dir in sorted(
        (path for path in OUTPUT_ROOT.iterdir() if path.is_dir()),
        key=lambda item: item.name,
        reverse=True,
    ):
        config_path = output_dir / "config.json"
        config = load_json(config_path) if config_path.exists() else {}
        summary_path = output_dir / "analysis" / "data" / "summary.json"
        summary = load_json(summary_path) if summary_path.exists() else {}

        raw_task_type = summary.get("task_type", config.get("task_type"))
        try:
            task_type = int(raw_task_type) if raw_task_type is not None else None
        except (TypeError, ValueError):
            task_type = None

        if task_type not in SUPPORTED_TASK_TYPES:
            continue

        iter_root = output_dir / "iters"
        if not iter_root.exists():
            iter_root = output_dir
        if not any(path.is_dir() and path.name.startswith("iter_") for path in iter_root.iterdir()):
            legacy_repro_root = output_dir / "repro" / "failed_iters"
            if legacy_repro_root.exists():
                iter_root = legacy_repro_root

        available_iters = tuple(
            sorted(
                extract_iteration(path)
                for path in iter_root.iterdir()
                if path.is_dir() and path.name.startswith("iter_")
            )
        )
        if not available_iters:
            continue

        model_name = str(summary.get("model_name") or infer_model_name(config, task_type))
        entries.append(
            OutputEntry(
                path=output_dir,
                task_type=task_type,
                model_name=model_name,
                iters=available_iters,
            )
        )
    return entries


def find_archived_weight(iter_dir: Path) -> Path | None:
    roots = [iter_dir / "weights", iter_dir / "artifacts" / "weights"]
    weights_root = next((root for root in roots if root.exists()), None)
    if weights_root is None:
        return None

    preferred_root = weights_root / "pta-save"
    roots = [preferred_root, weights_root] if preferred_root.exists() else [weights_root]
    for root in roots:
        direct_children = sorted(root.iterdir())
        for child in direct_children:
            if path_has_payload(child):
                return child.resolve()
        if path_has_payload(root):
            return root.resolve()
    return None


def runtime_root_candidates(iter_dir: Path) -> list[Path]:
    return [iter_dir / "repro_runtime", iter_dir / ".repro_runtime"]


def runtime_root_for(iter_dir: Path) -> Path:
    root = runtime_root_candidates(iter_dir)[0]
    root.mkdir(parents=True, exist_ok=True)
    return root


def runtime_weight_path(iter_dir: Path, task_type: int) -> Path:
    if task_type == 1:
        return runtime_root_for(iter_dir) / "weights" / "pta-save-generated"
    return runtime_root_for(iter_dir) / "weights" / "pta-save-generated.pth"


def runtime_weight_ready(iter_dir: Path, task_type: int) -> Path | None:
    relative = Path("weights") / ("pta-save-generated" if task_type == 1 else "pta-save-generated.pth")
    for root in runtime_root_candidates(iter_dir):
        candidate = root / relative
        if path_has_payload(candidate):
            return candidate.resolve()
    return None


def material_dir(iter_dir: Path, name: str) -> Path | None:
    return next((root for root in (iter_dir / name, iter_dir / "artifacts" / name) if root.exists()), None)


def parse_mutate_args_value(iter_dir: Path) -> str | None:
    scripts_dir = material_dir(iter_dir, "scripts")
    if scripts_dir is None:
        return None
    pattern = re.compile(r"export\s+MUTATE_ARGS=([\"'])(.*?)\1", re.DOTALL)
    for script_path in sorted(scripts_dir.glob("*.sh")):
        content = script_path.read_text(encoding="utf-8")
        match = pattern.search(content)
        if match:
            return " ".join(match.group(2).split())
    return None


def parse_mutate_flag(iter_dir: Path, flag: str) -> str | None:
    mutate_args = parse_mutate_args_value(iter_dir)
    if not mutate_args:
        return None
    match = re.search(rf"(?:^|\s){re.escape(flag)}\s+([^\s]+)", mutate_args)
    if match:
        return match.group(1).strip()
    return None


def parse_module_arg(iter_dir: Path) -> str | None:
    return parse_mutate_flag(iter_dir, "-m")


def parse_int_flag(iter_dir: Path, flag: str) -> int | None:
    raw = parse_mutate_flag(iter_dir, flag)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def task3_support_mf(config: dict) -> bool:
    task_conf = ((config.get("tasks") or {}).get("3")) or {}
    compare_mode = str(task_conf.get("COMPARE_MODE", "pta_msa") or "").strip().lower()
    return compare_mode == "pta_mf"


def infer_runs(iter_dir: Path, task_type: int, config: dict) -> list[RunEntry]:
    if task_type in {4, 5}:
        return infer_runs_task45(iter_dir, task_type)

    runs: list[RunEntry] = []
    scripts_dir = material_dir(iter_dir, "scripts")
    bucket: dict[str, RunEntry] = {}

    if scripts_dir is not None and scripts_dir.exists():
        if task_type == 1:
            mapping = {
                "pta-save": ("PTA-SAVE", "pta", "save"),
                "pta-load": ("PTA-LOAD", "pta", "load"),
                "msa-load": ("MSA-LOAD", "msa", "load"),
                "mf": ("MF", "mf", "load"),
            }
            ordered_keys = ("pta-save", "pta-load", "msa-load", "mf")
        else:
            mapping = {
                "pta-save": ("PTA-SAVE", "pta", "save"),
                "pta-load": ("PTA-LOAD", "pta", "load"),
                "msa-load": ("MSA-LOAD", "msa", "load"),
                "mf": ("MF", "mf", "load"),
            }
            ordered_keys = ("pta-save", "pta-load", "msa-load", "mf")

        for script_path in sorted(scripts_dir.glob("*.sh")):
            stem = script_path.stem
            for key, (label, env_kind, mode) in mapping.items():
                if stem.startswith(key):
                    bucket[key] = RunEntry(
                        key=key,
                        label=label,
                        env_kind=env_kind,
                        mode=mode,
                        task_type=task_type,
                        script_path=script_path.resolve(),
                    )
                    break

        if task_type == 1 and "mf" not in bucket:
            for yaml_path in sorted(scripts_dir.glob("*.yaml")):
                if not yaml_path.stem.startswith("mf-load"):
                    continue
                bucket["mf"] = RunEntry(
                    key="mf",
                    label="MF",
                    env_kind="mf",
                    mode="load",
                    task_type=task_type,
                    script_path=yaml_path.resolve(),
                )
                break

        runs.extend(bucket[key] for key in ordered_keys if key in bucket)

    if task_type == 3 and task3_support_mf(config) and "mf" not in bucket:
        runs.append(
            RunEntry(
                key="mf",
                label="MF",
                env_kind="mf",
                mode="load",
                task_type=task_type,
                script_path=None,
                synthetic=True,
            )
        )

    return runs

def infer_runs_task45(iter_dir: Path, task_type: int) -> list[RunEntry]:
    runtime_logs_dir = material_dir(iter_dir, "runtime_logs")
    if runtime_logs_dir is None or not runtime_logs_dir.exists():
        return []

    mapping = {
        "pta_save": ("pta-save", "PTA-SAVE", "pta", "save"),
        "pta_load": ("pta-load", "PTA-LOAD", "pta", "load"),
        "msa_load": ("msa-load", "MSA-LOAD", "msa", "load"),
    }
    ordered_prefixes = ("pta_save", "pta_load", "msa_load")

    detected: set[str] = set()
    for log_path in runtime_logs_dir.glob("*.log"):
        stem = log_path.stem.lower()
        for prefix in ordered_prefixes:
            if stem.startswith(prefix):
                detected.add(prefix)
                break

    runs: list[RunEntry] = []
    for prefix in ordered_prefixes:
        if prefix not in detected:
            continue
        key, label, env_kind, mode = mapping[prefix]
        runs.append(
            RunEntry(
                key=key,
                label=label,
                env_kind=env_kind,
                mode=mode,
                task_type=task_type,
                script_path=None,
            )
        )
    return runs


def task_config(config: dict, task_type: int) -> dict:
    return ((config.get("tasks") or {}).get(str(task_type))) or {}


def coerce_positive_int(value, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def result_dir_name_from_task(task_type: int, config: dict, iter_dir: Path) -> str:
    conf = task_config(config, task_type)
    module_arg = parse_module_arg(iter_dir)
    parsed_models = normalize_models(conf.get("MODELS"))

    if task_type == 2:
        if parsed_models:
            return f"submodule_{Path(parsed_models[-1]).stem}"
        if module_arg:
            return f"submodule_{Path(module_arg.split(',')[-1]).stem}"
        node_num = len(conf.get("SUBMODULES") or []) or parse_int_flag(iter_dir, "-n") or 1
        return f"submodule_random{node_num}nodes"

    if task_type == 3:
        explicit = conf.get("RESULT_DIR_NAME")
        if explicit:
            return Path(str(explicit)).stem
        if parsed_models:
            return Path(parsed_models[-1]).stem
        if module_arg:
            return Path(module_arg.split(",")[-1]).stem
        node_num = int(conf.get("NODE_NUM", parse_int_flag(iter_dir, "-n") or 1))
        return f"random{node_num}nodes"

    raise ValueError(f"不支持的 task_type: {task_type}")


def stage_mutation_inputs(iter_dir: Path, task_type: int, config: dict) -> StageInfo:
    mutation_root = material_dir(iter_dir, "mutation_inputs")
    if mutation_root is None:
        raise RuntimeError(f"缺少 mutation_inputs 目录: {mutation_root}")

    result_dir_name = result_dir_name_from_task(task_type, config, iter_dir)
    stage_dir = (REPO_ROOT / "res" / result_dir_name).resolve()
    stage_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for source in sorted(mutation_root.iterdir()):
        if not source.is_file():
            continue
        shutil.copy2(source, stage_dir / source.name)
        copied += 1

    if copied == 0:
        raise RuntimeError(f"mutation_inputs 目录为空: {mutation_root}")

    iteration = extract_iteration(iter_dir)
    return StageInfo(
        res_dir=stage_dir,
        load_path_arg=f"res/{stage_dir.name}/mutating-{iteration}.json",
    )


def patch_task1_script(content: str, run: RunEntry, *, load_path: Path | None = None, save_path: Path | None = None) -> str:
    updated = content
    if run.mode == "load":
        if load_path is None:
            raise ValueError("load run 需要提供 load_path")
        updated, replaced = re.subn(
            r"--load[ \t]+[^ \t\\]+",
            f"--load {load_path}",
            updated,
            count=1,
        )
        if replaced == 0:
            raise RuntimeError(f"脚本中未找到 --load 参数: {run.script_path}")
    elif run.mode == "save":
        if save_path is None:
            raise ValueError("save run 需要提供 save_path")
        updated, replaced = re.subn(
            r"--save[ \t]+[^ \t\\]+",
            f"--save {save_path}",
            updated,
            count=1,
        )
        if replaced == 0:
            raise RuntimeError(f"脚本中未找到 --save 参数: {run.script_path}")
    return updated


def patch_task23_script(content: str, run: RunEntry, shared_weight_path: Path, stage: StageInfo) -> str:
    updated = content
    if run.env_kind in {"pta", "msa"}:
        updated, replaced = re.subn(
            r"(?m)^(\s*export\s+LMSV_SHARED_WEIGHT_PATH=).*$",
            rf"\1{shlex.quote(str(shared_weight_path))}",
            updated,
            count=1,
        )
        if replaced == 0:
            raise RuntimeError(f"脚本中未找到 LMSV_SHARED_WEIGHT_PATH: {run.script_path}")

    if run.task_type == 3 and stage.load_path_arg and "--load-path" in updated:
        updated, replaced = re.subn(
            r"--load-path[ \t]+[^ \t\\]+",
            f"--load-path {stage.load_path_arg}",
            updated,
            count=1,
        )
        if replaced == 0:
            raise RuntimeError(f"脚本中未找到 --load-path 参数: {run.script_path}")

    return updated


def normalize_runtime_script_content(content: str) -> str:
    updated = content
    updated = re.sub(r"(?m)^\s*cd legacy\s*$\n?", "", updated)
    updated = updated.replace("../scripts/submodule_entry.py", "scripts/runtime/submodule_entry.py")
    updated = updated.replace("scripts/submodule_entry.py", "scripts/runtime/submodule_entry.py")
    updated = updated.replace("../scripts/msrun_launcher.sh", "scripts/runtime/msrun_launcher.sh")
    updated = updated.replace("scripts/msrun_launcher.sh", "scripts/runtime/msrun_launcher.sh")
    updated = updated.replace("bash scripts/mutate-auto.sh", "bash scripts/mutation/mutate-auto.sh")
    updated = updated.replace("bash scripts/mutate_submodule-auto.sh", "bash scripts/mutation/mutate_submodule-auto.sh")
    updated = updated.replace("python run_mindformer.py", "python utils/runtime/run_mindformer.py")
    updated = updated.replace('"run_mindformer.py', '"python utils/runtime/run_mindformer.py')
    updated = updated.replace(
        "python mf_mutate_and_forward/load_and_forward_graph.py",
        "python utils/runtime/mf_mutate_and_forward/load_and_forward_graph.py",
    )
    updated = updated.replace(
        "python mf_mutate_and_forward/load_and_forward_submodule.py",
        "python utils/runtime/mf_mutate_and_forward/load_and_forward_submodule.py",
    )
    updated = updated.replace("-c model_config", "-c assets/runtime/model_config")
    updated = updated.replace("../model_config", "assets/runtime/model_config")
    updated = updated.replace("configs/mutation_schema.yaml", "assets/runtime/configs/mutation_schema.yaml")
    updated = updated.replace("./baichuan2tokenizer/", "./assets/runtime/tokenizers/baichuan2/")
    updated = updated.replace("./llama2tokenizer/", "./assets/runtime/tokenizers/llama2/")
    updated = updated.replace("./qwen2tokenizer/", "./assets/runtime/tokenizers/qwen2/")
    updated = re.sub(
        r"(--position-embedding-type[ \t]+)learned\b",
        r"\1learned_absolute",
        updated,
    )
    return updated


def prepare_script(iter_dir: Path, run: RunEntry, *, load_path: Path | None = None, save_path: Path | None = None, stage: StageInfo | None = None) -> Path:
    if run.script_path is None:
        raise RuntimeError(f"{run.label} 没有可用脚本")

    runtime_scripts_dir = runtime_root_for(iter_dir) / "scripts"
    runtime_scripts_dir.mkdir(parents=True, exist_ok=True)
    dst = runtime_scripts_dir / run.script_path.name
    content = run.script_path.read_text(encoding="utf-8")

    if run.task_type == 1 and run.env_kind in {"pta", "msa"}:
        content = patch_task1_script(content, run, load_path=load_path, save_path=save_path)
    elif run.task_type in {2, 3}:
        if save_path is None and load_path is None:
            raise RuntimeError("task2/task3 脚本执行前需要共享权重路径")
        shared_weight_path = save_path or load_path
        if shared_weight_path is None or stage is None:
            raise RuntimeError("task2/task3 脚本缺少 stage 或共享权重路径")
        content = patch_task23_script(content, run, shared_weight_path, stage)

    content = normalize_runtime_script_content(content)

    dst.write_text(content, encoding="utf-8")
    dst.chmod(0o755)
    return dst.resolve()


def resolve_pretrain_path(config: dict, env_kind: str) -> str:
    if env_kind == "pta":
        base = config.get("PTA_PATH")
        candidates = [
            "MindSpeed-LLM/pretrain_gpt.py",
            "pretrain_gpt.py",
        ]
    elif env_kind == "msa":
        base = config.get("MSA_PATH")
        candidates = [
            "MSAdapter/pretrain_gpt.py",
            "MindSpeed-LLM/pretrain_gpt.py",
            "Megatron-LM/pretrain_gpt.py",
            "pretrain_gpt.py",
        ]
    else:
        raise ValueError(f"不支持的环境类型: {env_kind}")

    if not base:
        raise RuntimeError(f"配置里缺少 {env_kind.upper()}_PATH")

    base_path = Path(str(base)).expanduser()
    for relative in candidates:
        candidate = (base_path / relative).resolve()
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError(f"未找到 {env_kind.upper()} 环境里的 pretrain_gpt.py，请检查复现目录内的 config.json")


def build_task1_command(config: dict, run: RunEntry, prepared_script: Path, pretrain_path: str) -> str:
    if run.env_kind == "pta":
        env_name = str(config.get("PTA_NAME") or "")
        work_path = str(config.get("PTA_PATH") or "")
        path_var = "PTAPATH"
        envset = "scripts/envset/pta.sh"
    elif run.env_kind == "msa":
        env_name = str(config.get("MSA_NAME") or "")
        work_path = str(config.get("MSA_PATH") or "")
        path_var = "MSAPATH"
        envset = "scripts/envset/msa.sh"
    else:
        raise ValueError(f"不支持的环境类型: {run.env_kind}")

    return f"""
{build_conda_activate_block(env_name, load_ascend=True)}
export {path_var}={shlex.quote(work_path)}
source {envset}
export LMSV_PRETRAIN_GPT={shlex.quote(pretrain_path)}
echo "[repro] using pretrain entry: $LMSV_PRETRAIN_GPT"
bash {shlex.quote(str(prepared_script))}
"""


def make_session_dir(iter_dir: Path, run: RunEntry) -> Path:
    logs_root = runtime_root_for(iter_dir) / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = logs_root / f"{stamp}_{run.key}"
    session_dir = base
    index = 1
    while session_dir.exists():
        session_dir = logs_root / f"{base.name}_{index}"
        index += 1
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def candidate_label(path: Path) -> str:
    try:
        return "_".join(path.resolve().relative_to(REPO_ROOT.resolve()).parts)
    except ValueError:
        return path.name


def msrun_log_dir_candidates() -> list[Path]:
    return [
        REPO_ROOT / "msrun_log",
        REPO_ROOT / "output" / "msrun_log",
    ]


def resolve_worker_count_from_script(script_path: Path | None, default_workers: int = 8) -> int:
    if script_path is None or not script_path.exists():
        return max(1, int(default_workers))

    try:
        content = script_path.read_text(encoding="utf-8")
    except OSError:
        return max(1, int(default_workers))

    var_values: dict[str, int] = {}
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


def resolve_msa_worker_log(run: RunEntry) -> Path | None:
    worker_index = max(0, resolve_worker_count_from_script(run.script_path) - 1)
    worker_name = f"worker_{worker_index}.log"
    return next((candidate / worker_name for candidate in msrun_log_dir_candidates() if (candidate / worker_name).exists()), None)


def prepare_msa_log_dirs(session_dir: Path) -> Path | None:
    archive_root = session_dir / "preexisting_msrun_logs"
    archived_any = False
    for candidate in msrun_log_dir_candidates():
        if not candidate.exists():
            continue
        if path_has_payload(candidate):
            shutil.copytree(candidate, archive_root / candidate_label(candidate), dirs_exist_ok=True)
            archived_any = True
        reset_path(candidate)
        candidate.mkdir(parents=True, exist_ok=True)
    return archive_root if archived_any else None


def archive_msrun_logs(session_dir: Path) -> list[Path]:
    archive_root = session_dir / "msrun_logs"
    archived: list[Path] = []
    for candidate in msrun_log_dir_candidates():
        if not candidate.exists() or not path_has_payload(candidate):
            continue
        dst = archive_root / candidate_label(candidate)
        shutil.copytree(candidate, dst, dirs_exist_ok=True)
        archived.append(dst.resolve())
    return archived


def follow_msa_worker_log(worker_log: Path, *, max_wait: int, stable_seconds: int) -> None:
    deadline = time.time() + max_wait
    last_size = 0
    stable_since: float | None = None
    finish_seen = False
    printed_any = False
    waiting_reported = False
    ended_with_newline = True

    while time.time() < deadline:
        if not worker_log.exists():
            if not waiting_reported:
                print(f"[repro] 启动命令已返回，等待日志生成: {worker_log}", flush=True)
                waiting_reported = True
            time.sleep(1)
            continue

        if not printed_any:
            print(f"[repro] 开始打印 MSA 日志: {worker_log}", flush=True)
            printed_any = True

        try:
            current_size = worker_log.stat().st_size
        except OSError:
            time.sleep(1)
            continue

        if current_size < last_size:
            last_size = 0

        if current_size > last_size:
            with worker_log.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(last_size)
                chunk = handle.read()
            if chunk:
                sys.stdout.write(chunk)
                sys.stdout.flush()
                ended_with_newline = chunk.endswith("\n")
                lowered = chunk.lower()
                if any(pattern in lowered for pattern in MSA_FINISH_PATTERNS):
                    finish_seen = True
            last_size = current_size
            stable_since = None
            time.sleep(1)
            continue

        if stable_since is None:
            stable_since = time.time()
        stable_for = int(time.time() - stable_since)
        if finish_seen and stable_for >= 3:
            if not ended_with_newline:
                print()
            print("[repro] 检测到 MSA 结束标记，停止打印日志。", flush=True)
            return
        if stable_for >= stable_seconds:
            if not ended_with_newline:
                print()
            print(f"[repro] MSA 日志已连续 {stable_seconds}s 无更新，停止打印。", flush=True)
            return
        time.sleep(1)

    if printed_any and not ended_with_newline:
        print()
    print(f"[repro] 等待 MSA 日志打印超时（>{max_wait}s），停止打印。", flush=True)


def follow_and_archive_msa_logs(config: dict, run: RunEntry, session_dir: Path | None) -> list[Path]:
    if session_dir is None:
        return []

    conf = task_config(config, run.task_type)
    max_wait = coerce_positive_int(conf.get("MSA_MAX_RUNTIME", conf.get("MAX_VALIDATE_TIME")), DEFAULT_MSA_FOLLOW_TIMEOUT)
    stable_seconds = coerce_positive_int(conf.get("LOG_STABLE_THRESHOLD"), DEFAULT_MSA_STABLE_SECONDS)

    worker_index = max(0, resolve_worker_count_from_script(run.script_path) - 1)
    worker_name = f"worker_{worker_index}.log"
    worker_log = resolve_msa_worker_log(run)
    if worker_log is None:
        deadline = time.time() + max_wait
        while time.time() < deadline:
            worker_log = resolve_msa_worker_log(run)
            if worker_log is not None:
                break
            time.sleep(1)

    if worker_log is not None:
        follow_msa_worker_log(worker_log, max_wait=max_wait, stable_seconds=stable_seconds)
    else:
        print(f"[repro] 未检测到 msrun_log/{worker_name}，跳过日志打印。", flush=True)

    archived = archive_msrun_logs(session_dir)
    if archived:
        if len(archived) == 1:
            print(f"[repro] msrun 日志已保留到: {archived[0]}", flush=True)
        else:
            joined = " | ".join(str(path) for path in archived)
            print(f"[repro] msrun 日志已保留到: {joined}", flush=True)
    return archived


def archive_msa_logs_only(session_dir: Path | None) -> list[Path]:
    if session_dir is None:
        return []
    archived = archive_msrun_logs(session_dir)
    if archived:
        if len(archived) == 1:
            print(f"[repro] msrun 日志已保留到: {archived[0]}", flush=True)
        else:
            joined = " | ".join(str(path) for path in archived)
            print(f"[repro] msrun 日志已保留到: {joined}", flush=True)
    return archived


def build_task3_mf_command(config: dict, iter_dir: Path, stage: StageInfo) -> str:
    conf = task_config(config, 3)
    models = normalize_models(conf.get("MODELS"))
    module_arg = ",".join(models) if models else parse_module_arg(iter_dir)
    if not module_arg:
        raise RuntimeError("无法推导 task3 的 MODELS 参数，不能生成 MF 复现命令")

    mutnm = int(conf.get("MUTNM", parse_int_flag(iter_dir, "--mutnm") or 2))
    node_num = int(conf.get("NODE_NUM", parse_int_flag(iter_dir, "-n") or len(models) or 1))
    mutation_rounds = int(conf.get("MUTATION_ROUNDS", conf.get("TOTAL_ITER", 100)))
    mf_env = resolve_mf_env_name(config, 3)
    mf_args_path = str(conf.get("MF_ARGS_PATH") or DEFAULT_MF_ARGS_PATH)
    iteration = extract_iteration(iter_dir)

    mutate_args = (
        f"-c assets/runtime/model_config -r {mutation_rounds} --mutnm {mutnm} "
        f"-n {node_num} -m {module_arg}"
    )

    if not stage.load_path_arg:
        raise RuntimeError("task3 MF 复现缺少 load-path")

    return f"""
{build_conda_activate_block(mf_env, load_ascend=True)}
export MUTATE_ROUND={iteration}
export MUTATE_ARGS={shlex.quote(mutate_args)}
python utils/runtime/mf_mutate_and_forward/load_and_forward_graph.py \
    $MUTATE_ARGS \
    --load-path {shlex.quote(stage.load_path_arg)} \
    --args_path {shlex.quote(mf_args_path)}
"""


def infer_mf_card_num_from_yaml(yaml_path: Path) -> int:
    text = yaml_path.read_text(encoding="utf-8")
    values = {}
    for key in ("data_parallel", "model_parallel", "pipeline_stage"):
        match = re.search(rf"(?m)^\s*{key}\s*:\s*(\d+)\s*$", text)
        values[key] = int(match.group(1)) if match else 1
    card_num = values["data_parallel"] * values["model_parallel"] * values["pipeline_stage"]
    return max(1, card_num)


def resolve_mf_env_name(config: dict, task_type: int) -> str:
    conf = task_config(config, task_type)
    return str(
        conf.get("MF_ENV")
        or config.get("MF_NAME")
        or os.environ.get("MF_NAME")
        or DEFAULT_MF_ENV
    )


def build_task1_mf_yaml_command(config: dict, mf_yaml_path: Path) -> str:
    mf_env = resolve_mf_env_name(config, 1)
    card_num = infer_mf_card_num_from_yaml(mf_yaml_path)
    return f"""
{build_conda_activate_block(mf_env, load_ascend=True)}
export PYTHONPATH={shlex.quote(str(REPO_ROOT))}:${{PYTHONPATH:-}}
bash scripts/runtime/mf_start.sh {shlex.quote(str(mf_yaml_path))} {card_num}
"""


def run_shell_streaming(command: str, log_path: Path | None = None) -> int:
    env = os.environ.copy()
    env.pop("LMSV_ENABLE_TRAINING_LOG_PATCH", None)
    env.pop("LMSV_TRAINING_LOG_CSV", None)
    env.pop("LMSV_PATCH_LOG", None)
    env.pop("LMSV_PRETRAIN_GPT", None)
    sanitize_pythonpath(env)
    stream = nullcontext()
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stream = log_path.open("w", encoding="utf-8")

    with stream as handle:
        if handle is not None:
            handle.write(f"[START] {datetime.utcnow().isoformat()}Z\n")
            handle.write("[COMMAND]\n")
            handle.write(command)
            if not command.endswith("\n"):
                handle.write("\n")
            handle.write("\n")
            handle.flush()

        process = subprocess.Popen(
            ["bash", "-lc", command],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                if handle is not None:
                    handle.write(line)
                    handle.flush()
            return_code = process.wait()
        except BaseException:
            with suppress(ProcessLookupError):
                process.terminate()
            with suppress(subprocess.TimeoutExpired, ProcessLookupError):
                process.wait(timeout=5)
            with suppress(ProcessLookupError):
                process.kill()
            raise

        if handle is not None:
            handle.write(f"\n[END] {datetime.utcnow().isoformat()}Z\n")
            handle.write(f"[RETURNCODE] {return_code}\n")
            handle.flush()
        return return_code


def execute_run(iter_dir: Path, config: dict, run: RunEntry, *, load_path: Path | None = None, save_path: Path | None = None) -> ExecutionResult:
    session_dir = make_session_dir(iter_dir, run)
    command_log_path = session_dir / "command.log"
    previous_msrun_log_archive: Path | None = None
    if run.env_kind == "msa":
        previous_msrun_log_archive = prepare_msa_log_dirs(session_dir)

    if run.task_type == 1:
        if run.env_kind == "mf":
            if run.script_path is None:
                raise RuntimeError("MF 复现缺少脚本或 YAML")

            path = run.script_path
            if path.suffix == ".yaml":
                command = build_task1_mf_yaml_command(config, path)
                note = f"YAML: {path}\n[repro] 执行日志: {command_log_path}"
                return ExecutionResult(
                    return_code=run_shell_streaming(command, command_log_path),
                    note=note,
                    session_dir=session_dir,
                )

            prepared_script = prepare_script(iter_dir, run)
            note = f"脚本: {prepared_script}\n[repro] 执行日志: {command_log_path}"
            return ExecutionResult(
                return_code=run_shell_streaming(f"bash {shlex.quote(str(prepared_script))}", command_log_path),
                note=note,
                session_dir=session_dir,
            )

        prepared_script = prepare_script(iter_dir, run, load_path=load_path, save_path=save_path)
        pretrain_path = resolve_pretrain_path(config, run.env_kind)
        command = build_task1_command(config, run, prepared_script, pretrain_path)
        note = f"脚本: {prepared_script}"
        if load_path is not None:
            note += f"\n[repro] load 权重: {load_path}"
        if save_path is not None:
            note += f"\n[repro] save 权重输出: {save_path}"
        note += f"\n[repro] 执行日志: {command_log_path}"
        if previous_msrun_log_archive is not None:
            note += f"\n[repro] 启动前旧 msrun 日志已备份到: {previous_msrun_log_archive}"
        return ExecutionResult(
            return_code=run_shell_streaming(command, command_log_path),
            note=note,
            session_dir=session_dir,
        )

    if run.task_type in {2, 3}:
        stage = stage_mutation_inputs(iter_dir, run.task_type, config)
        if run.synthetic:
            command = build_task3_mf_command(config, iter_dir, stage)
            note = f"mutation 输入已回填到: {stage.res_dir}\n[repro] load-path: {stage.load_path_arg}"
            note += f"\n[repro] 执行日志: {command_log_path}"
            return ExecutionResult(
                return_code=run_shell_streaming(command, command_log_path),
                note=note,
                session_dir=session_dir,
            )

        prepared_script = prepare_script(
            iter_dir,
            run,
            load_path=load_path,
            save_path=save_path,
            stage=stage,
        )
        note = f"脚本: {prepared_script}\n[repro] mutation 输入已回填到: {stage.res_dir}"
        if stage.load_path_arg:
            note += f"\n[repro] load-path: {stage.load_path_arg}"
        if load_path is not None:
            note += f"\n[repro] 共享权重: {load_path}"
        if save_path is not None:
            note += f"\n[repro] 共享权重输出: {save_path}"
        note += f"\n[repro] 执行日志: {command_log_path}"
        if previous_msrun_log_archive is not None:
            note += f"\n[repro] 启动前旧 msrun 日志已备份到: {previous_msrun_log_archive}"
        return ExecutionResult(
            return_code=run_shell_streaming(f"bash {shlex.quote(str(prepared_script))}", command_log_path),
            note=note,
            session_dir=session_dir,
        )

    raise RuntimeError(f"不支持的 task_type: {run.task_type}")


def ensure_weight_for_load(iter_dir: Path, config: dict, run: RunEntry, runs: list[RunEntry]) -> Path:
    archived = find_archived_weight(iter_dir)
    if archived is not None:
        return archived

    runtime_weight = runtime_weight_ready(iter_dir, run.task_type)
    if runtime_weight is not None:
        return runtime_weight

    save_run = next((item for item in runs if item.key == "pta-save"), None)
    if save_run is None:
        raise RuntimeError("当前轮次缺少 pta-save 脚本，无法为 load 复现补权重")

    answer = input("当前 repro 里没有可用权重，是否先执行同轮 PTA-SAVE 生成复现权重？[y/N]: ").strip().lower()
    if answer not in {"y", "yes"}:
        raise RuntimeError("已取消：load 复现需要可用权重")

    target = runtime_weight_path(iter_dir, run.task_type)
    target.parent.mkdir(parents=True, exist_ok=True)
    reset_path(target)

    print()
    print(f"[repro] 即将执行前置 {save_run.label}")
    result = execute_run(iter_dir, config, save_run, save_path=target)
    print(result.note)
    print()
    if result.return_code != 0:
        raise RuntimeError(f"前置 PTA-SAVE 执行失败，退出码: {result.return_code}")
    if not path_has_payload(target):
        raise RuntimeError(f"前置 PTA-SAVE 未生成权重: {target}")
    return target.resolve()


def read_task_config(output_dir: Path) -> dict:
    config_path = output_dir / "config.json"
    if not config_path.exists():
        raise RuntimeError(f"缺少配置文件: {config_path}")
    return load_json(config_path)


def render_output_entry(entry: OutputEntry) -> str:
    task_label = f"task{entry.task_type}" if entry.task_type is not None else "task?"
    iter_text = ",".join(f"iter{item}" for item in entry.iters)
    return f"{entry.path.name} | {task_label} | model={entry.model_name} | 可复现轮次={iter_text}"


def render_run_entry(run: RunEntry) -> str:
    if run.key == "mf":
        return "MF"
    extra = "（需要权重）" if run.mode == "load" else "（会生成权重）"
    return f"{run.label} {extra}"

def ensure_ckpt_path_task45(iter_dir: Path, task_type: int) -> Path:
    ckpt_path = None
    if task_type == 4:
        ckpt_path = iter_dir / "core_backup" / "2-pta-save" / "ckpts" / "round_0.pt"
    elif task_type == 5:
        ckpt_path = iter_dir / "core_backup" / "2-pta-save" / "ckpts" / "mutated_config.pt"
    else:
        raise RuntimeError(f"不支持的 task_type: {task_type}")

    if not ckpt_path.exists():
        raise RuntimeError(f"未找到权重: {ckpt_path}")
    return ckpt_path.resolve()

def kill_all_processses_on_npu():
    from utils.control.clean import _collect_npu_smi_processes, kill_pretraingpt

    npu_processes = _collect_npu_smi_processes()
    pids = {pid for pid in npu_processes if isinstance(pid, int) and pid > 0}
    print(f"[repro] npu-smi 解析到 {len(pids)} 个 NPU 进程。")

    current_pid = os.getpid()
    killed: list[int] = []
    not_found: list[int] = []
    permission_denied: list[int] = []
    for pid in sorted(pids):
        if pid == current_pid:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
        except ProcessLookupError:
            not_found.append(pid)
        except PermissionError as exc:
            permission_denied.append(pid)
            print(f"[repro] 无权限终止 NPU 进程 {pid}: {exc}")

    if killed:
        print(f"[repro] 已终止 NPU 进程: {killed}")
    else:
        print("[repro] 未发现可终止的 NPU 进程。")
    print(
        "[repro] NPU 进程清理统计: "
        f"killed={len(killed)}, not_found={len(not_found)}, permission_denied={len(permission_denied)}"
    )
    if not killed and not_found:
        print("[repro] 提示：PID 可能来自宿主机命名空间，容器内无法直接 kill（ProcessLookupError）。")
        print("[repro] 回退到容器内运行时进程清理（kill_pretraingpt）。")
        kill_pretraingpt()

def execute_run_task45(iter_dir: Path, config: dict, run: RunEntry, ckpt_path: Path | None = None) -> ExecutionResult:
    # create the repro_runtime directory under the iter_dir
    repro_runtime_dir = iter_dir / "repro_runtime"
    repro_runtime_dir.mkdir(parents=True, exist_ok=True)
    # create the sub directory under the repro_runtime directory
    run_dir = repro_runtime_dir / run.key
    # if the run_dir already exists, delete it
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    repro_log_path = run_dir / "repro.log"

    # find the config path
    config_path = None
    if run.task_type == 4:
        config_path = iter_dir / "core_backup" / "1-pta-mutate" / "configs" / "round_0.json"
    elif run.task_type == 5:
        config_path = iter_dir / "core_backup" / "1-pta-mutate" / "mutated_config.json"
    else:
        raise RuntimeError(f"不支持的 task_type: {run.task_type}")
    if not config_path.exists():
        raise RuntimeError(f"未找到配置: {config_path}")

    kill_all_processses_on_npu()
    time.sleep(1)
    if run.task_type == 4:
        if run.key == "pta-save":
            from utils.task.task4 import Config,run_pta_save,post_pta_save
            Config.PTA_PATH = config.get("PTA_PATH")
            Config.PTA_ENV = config.get("PTA_NAME")
            Config.SAVE_STEPS = config.get("tasks").get(str(run.task_type)).get("SAVE_STEPS")
            Config.PTA_MAX_RUNTIME = config.get("tasks").get(str(run.task_type)).get("PTA_MAX_RUNTIME")
            ok = run_pta_save(config_path, run_dir, repro_log_path)
            post_pta_save(run_dir)
            return ExecutionResult(
                return_code=0 if ok else 1,
                note=f"复现日志: {repro_log_path}",
                session_dir=run_dir,
            )
        elif run.key == "pta-load":
            from utils.task.task4 import Config,run_pta_run,post_pta_run
            Config.PTA_PATH = config.get("PTA_PATH")
            Config.PTA_ENV = config.get("PTA_NAME")
            Config.RUN_STEPS = config.get("tasks").get(str(run.task_type)).get("RUN_STEPS")
            Config.PTA_MAX_RUNTIME = config.get("tasks").get(str(run.task_type)).get("PTA_MAX_RUNTIME")
            ok = run_pta_run(config_path, run_dir, ckpt_path, repro_log_path)
            post_pta_run(run_dir)
            return ExecutionResult(
                return_code=0 if ok else 1,
                note=f"复现日志: {repro_log_path}",
                session_dir=run_dir,
            )
        elif run.key == "msa-load":
            from utils.task.task4 import Config,run_msa_run,post_msa_run
            Config.MSA_PATH = config.get("MSA_PATH")
            Config.MSA_ENV = config.get("MSA_NAME")
            Config.RUN_STEPS = config.get("tasks").get(str(run.task_type)).get("RUN_STEPS")
            Config.MSA_MAX_RUNTIME = config.get("tasks").get(str(run.task_type)).get("MSA_MAX_RUNTIME")
            ok = run_msa_run(config_path, run_dir, ckpt_path, repro_log_path)
            post_msa_run(run_dir, success=ok)
            return ExecutionResult(
                return_code=0 if ok else 1,
                note=f"复现日志: {repro_log_path}",
                session_dir=run_dir,
            )
        else:
            raise RuntimeError(f"不支持的 run_key: {run.key}")
    if run.task_type == 5:
        if run.key == "pta-save":
            from utils.task.task5 import Config,run_pta_save,post_pta_save
            Config.PTA_PATH = config.get("PTA_PATH")
            Config.PTA_ENV = config.get("PTA_NAME")
            Config.SAVE_STEPS = config.get("tasks").get(str(run.task_type)).get("SAVE_STEPS")
            Config.PTA_MAX_RUNTIME = config.get("tasks").get(str(run.task_type)).get("PTA_MAX_RUNTIME")
            ok = run_pta_save(config_path, run_dir, repro_log_path)
            post_pta_save(run_dir)
            return ExecutionResult(
                return_code=0 if ok else 1,
                note=f"复现日志: {repro_log_path}",
                session_dir=run_dir,
            )
        elif run.key == "pta-load":
            from utils.task.task5 import Config,run_pta_run,post_pta_run
            Config.PTA_PATH = config.get("PTA_PATH")
            Config.PTA_ENV = config.get("PTA_NAME")
            Config.RUN_STEPS = config.get("tasks").get(str(run.task_type)).get("RUN_STEPS")
            Config.PTA_MAX_RUNTIME = config.get("tasks").get(str(run.task_type)).get("PTA_MAX_RUNTIME")
            ok = run_pta_run(config_path, run_dir, ckpt_path, repro_log_path)
            post_pta_run(run_dir)
            return ExecutionResult(
                return_code=0 if ok else 1,
                note=f"复现日志: {repro_log_path}",
                session_dir=run_dir,
            )
        elif run.key == "msa-load":
            from utils.task.task5 import Config,run_msa_run,post_msa_run
            Config.MSA_PATH = config.get("MSA_PATH")
            Config.MSA_ENV = config.get("MSA_NAME")
            Config.RUN_STEPS = config.get("tasks").get(str(run.task_type)).get("RUN_STEPS")
            Config.MSA_MAX_RUNTIME = config.get("tasks").get(str(run.task_type)).get("MSA_MAX_RUNTIME")
            ok = run_msa_run(config_path, run_dir, ckpt_path, repro_log_path)
            post_msa_run(run_dir, success=ok)
            return ExecutionResult(
                return_code=0 if ok else 1,
                note=f"复现日志: {repro_log_path}",
                session_dir=run_dir,
            )
        else:
            raise RuntimeError(f"不支持的 run_key: {run.key}")
    

def main() -> int:
    print("LMSV repro helper")
    print("当前版本支持 task1 / task2 / task3 / task4 / task5 的单次 run 复现。")

    outputs = discover_output_entries()
    if not outputs:
        print("未找到可复现的输出目录。")
        return 1

    try:
        output_entry = choose_one(outputs, "请选择 output 任务：", render_output_entry)
    except KeyboardInterrupt:
        print("已退出。")
        return 1

    config = read_task_config(output_entry.path)
    repro_root = output_entry.path / "iters"
    if not repro_root.exists():
        repro_root = output_entry.path
    if not any(candidate.is_dir() and candidate.name.startswith("iter_") for candidate in repro_root.iterdir()):
        repro_root = output_entry.path / "repro" / "failed_iters"
    iter_dirs = [
        path
        for path in sorted(
            (
                candidate
                for candidate in repro_root.iterdir()
                if candidate.is_dir() and candidate.name.startswith("iter_")
            ),
            key=extract_iteration,
        )
    ]
    if not iter_dirs:
        print(f"未找到复现轮次目录: {repro_root}")
        return 1

    task_type = int(output_entry.task_type) if output_entry.task_type is not None else None
    if task_type is None:
        print("无法识别 task_type。")
        return 1

    try:
        if task_type in {1, 2, 3}:
            iter_dir = choose_one(
                iter_dirs,
                "请选择要复现的轮次：",
                lambda path: f"{path.name} | {path / 'scripts'}",
            )
        elif task_type in {4, 5}:
            iter_dir = choose_one(
                iter_dirs,
                "请选择要复现的轮次：",
                lambda path: f"{path.name} | {path}",
            )
    except KeyboardInterrupt:
        print("已退出。")
        return 1

    runs = infer_runs(iter_dir, task_type, config)
    if not runs:
        print(f"当前轮次未找到可复现脚本或可生成命令: {iter_dir}")
        return 1

    try:
        run = choose_one(runs, "请选择要复现的具体 run：", render_run_entry)
    except KeyboardInterrupt:
        print("已退出。")
        return 1

    try:
        if task_type in {1, 2, 3}:
            if run.mode == "load" and run.key != "mf":
                weight_path = ensure_weight_for_load(iter_dir, config, run, runs)
                print()
                print(f"[repro] 即将执行: {run.label}")
                result = execute_run(iter_dir, config, run, load_path=weight_path)
            elif run.mode == "save":
                save_path = runtime_weight_path(iter_dir, task_type)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                reset_path(save_path)
                print()
                print(f"[repro] 即将执行: {run.label}")
                result = execute_run(iter_dir, config, run, save_path=save_path)
            else:
                print()
                print(f"[repro] 即将执行: {run.label}")
                result = execute_run(iter_dir, config, run)

            print(result.note)
            print()

            if run.env_kind == "msa":
                if result.return_code == 0:
                    follow_and_archive_msa_logs(config, run, result.session_dir)
                else:
                    archive_msa_logs_only(result.session_dir)
                print()

            if result.return_code != 0:
                print(f"[repro] 执行失败，退出码: {result.return_code}")
                return result.return_code

            if run.mode == "save":
                final_weight = runtime_weight_path(iter_dir, task_type)
                if not path_has_payload(final_weight):
                    print(f"[repro] PTA-SAVE 已退出，但未检测到权重产物: {final_weight}")
                    return 1
                print(f"[repro] PTA-SAVE 已完成，权重已保存到: {final_weight}")
            else:
                print(f"[repro] {run.label} 执行完成。")
            return 0
        elif task_type in {4, 5}:
            if run.mode == "load":
                ckpt_path = ensure_ckpt_path_task45(iter_dir, task_type)
                print()
                print(f"[repro] 正在执行: {run.label}")
                result = execute_run_task45(iter_dir, config, run, ckpt_path)
                print(result.note)
                print()
            else:
                print()
                print(f"[repro] 正在执行: {run.label}")
                result = execute_run_task45(iter_dir, config, run)
                print(result.note)
                print()
    except KeyboardInterrupt:
        print("已中断。")
        return 130
    except Exception as exc:
        print(f"[repro] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
