#!/usr/bin/env python3
"""
Task 实验结果自动分析。

支持：
1. 统计执行轮次、变异成功率、有效对比轮次。
2. 对有效轮次做精度 / 性能 / 显存定量分析并分级。
3. 对异常轮次做问题归类，并导出 repro 材料。
4. 生成 JSON / Markdown / HTML / SVG 报告。
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import shutil
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from . import rules


TASK_PROFILES = {
    1: {
        "task_name": "task1",
        "task_title": "Task1 训练一致性自动分析报告",
        "hero_title": "PTA / MSA 训练一致性实验报告",
        "comparison_mode": "series",
        "csv_relpaths": {
            "pta": "training_log_pta-{iteration}.csv",
            "msa": "training_log_msa-{iteration}.csv",
            "mf": "training_log_mf-{iteration}.csv",
        },
        "supports_mf": False,
        "performance_rule": "去掉第一个 step，比较其余 step 的平均耗时。",
        "performance_chart_subtitle": "已忽略第一个 step；正值表示 MSA 更慢，负值按 0 展示。",
        "precision_rule": "按同一步数对齐后，loss 必须与 PTA 零误差。",
        "memory_rule": "比较整轮 CSV 中记录到的最大显存。",
        "memory_chart_subtitle": "取各自 CSV 中的最大显存；正值表示 MSA 更高，负值按 0 展示。",
    },
    2: {
        "task_name": "task2",
        "task_title": "Task2 模块内组件泛化自动分析报告",
        "hero_title": "PTA / MSA 模块内组件泛化报告",
        "comparison_mode": "single_row",
        "csv_relpaths": {
            "pta": "submodule_execution_pta.csv",
            "msa": "submodule_execution_msa.csv",
            "mf": "submodule_execution_mf.csv",
        },
        "step_csv_relpaths": {
            "pta": "training_log_pta-{iteration}.csv",
            "msa": "training_log_msa-{iteration}.csv",
            "mf": "training_log_mf-{iteration}.csv",
        },
        "supports_mf": False,
        "performance_rule": "优先使用逐 step 日志，去掉第一个 step 后比较其余 step 的平均耗时；缺失时回退为比较当前迭代记录的 Execution Time (s)。",
        "performance_chart_subtitle": "优先忽略首步后按逐 step 日志比较；缺失时回退到当前迭代行。正值表示 MSA 更慢，负值按 0 展示。",
        "precision_rule": "优先按逐 step 日志对齐公共 step，比较最大 loss 绝对差；缺失时回退到当前迭代记录。",
        "memory_rule": "优先比较逐 step 日志中的最大显存；缺失时回退到当前迭代记录。",
        "memory_chart_subtitle": "优先取逐 step 日志最大显存；缺失时回退到当前迭代行。正值表示 MSA 更高，负值按 0 展示。",
    },
    3: {
        "task_name": "task3",
        "task_title": "Task3 模块间泛化组合自动分析报告",
        "hero_title": "PTA / MSA 模块间泛化组合报告",
        "comparison_mode": "single_row",
        "csv_relpaths": {
            "pta": "execution_pta.csv",
            "msa": "execution_msa.csv",
            "mf": "execution_mf.csv",
        },
        "step_csv_relpaths": {
            "pta": "training_log_pta-{iteration}.csv",
            "msa": "training_log_msa-{iteration}.csv",
            "mf": "training_log_mf-{iteration}.csv",
        },
        "supports_mf": True,
        "performance_rule": "优先使用逐 step 日志，去掉第一个 step 后比较其余 step 的平均耗时；缺失时回退为比较当前迭代记录的 Execution Time (s)。",
        "performance_chart_subtitle": "优先忽略首步后按逐 step 日志比较；缺失时回退到当前迭代行。正值表示 MSA 更慢，负值按 0 展示。",
        "precision_rule": "优先按逐 step 日志对齐公共 step，比较最大 loss 绝对差；缺失时回退到当前迭代记录。",
        "memory_rule": "优先比较逐 step 日志中的最大显存；缺失时回退到当前迭代记录。",
        "memory_chart_subtitle": "优先取逐 step 日志最大显存；缺失时回退到当前迭代行。正值表示 MSA 更高，负值按 0 展示。",
    },
    4: {
        "task_name": "task4",
        "task_title": "Task4 多模态模块间泛化组合自动分析报告",
        "hero_title": "PTA / MSA 多模态模块间泛化组合报告",
        "comparison_mode": "series",
        "csv_relpaths": {
            "pta": "core_backup/3-pta-run/runtime_info.csv",
            "msa": "core_backup/4-msa-run/runtime_info.csv",
            "mf": "runtime_info.csv",
        },
        "supports_mf": False,
        "performance_rule": "去掉第一个 step，比较其余 step 的平均耗时。",
        "performance_chart_subtitle": "已忽略第一个 step；正值表示 MSA 更慢，负值按 0 展示。",
        "precision_rule": "按同一步数对齐后，loss 必须与 PTA 零误差。",
        "memory_rule": "比较整轮 CSV 中记录到的最大显存。",
        "memory_chart_subtitle": "取各自 CSV 中的最大显存；正值表示 MSA 更高，负值按 0 展示。",
    },
    5: {
        "task_name": "task5",
        "task_title": "Task5 多模态模块内组件泛化自动分析报告",
        "hero_title": "PTA / MSA 多模态模块内组件泛化报告",
        "comparison_mode": "series",
        "csv_relpaths": {
            "pta": "core_backup/3-pta-run/runtime_info.csv",
            "msa": "core_backup/4-msa-run/runtime_info.csv",
            "mf": "runtime_info.csv",
        },
        "supports_mf": False,
        "performance_rule": "去掉第一个 step，比较其余 step 的平均耗时。",
        "performance_chart_subtitle": "已忽略第一个 step；正值表示 MSA 更慢，负值按 0 展示。",
        "precision_rule": "按同一步数对齐后，loss 必须与 PTA 零误差。",
        "memory_rule": "比较整轮 CSV 中记录到的最大显存。",
        "memory_chart_subtitle": "取各自 CSV 中的最大显存；正值表示 MSA 更高，负值按 0 展示。",
    },
}

STATUS_BADGE_COLORS = {
    "PASS": "#16a34a",
    "COMPLETED_WITH_ISSUES": "#f97316",
    "EXECUTION_FAILED": "#dc2626",
    "MUTATION_FAILED": "#7c3aed",
    "PASS_WITH_WARNINGS": "#ea580c",
}

GENERIC_COMPLETION_REASONS = {
    "",
    "分析阶段执行完成",
    "迭代执行完成",
}

SIGNAL_CATEGORY_ORDER = {
    "功能问题": 0,
    "精度问题": 1,
    "性能问题": 2,
    "显存问题": 3,
}


@dataclass
class IssueSignal:
    category: str
    message: str
    log_path: str
    line_number: int


@dataclass
class FunctionalReason:
    issue_subtype: str
    message: str


def _functional_owner_from_text(*values: object) -> str:
    text = " ".join(str(value or "") for value in values).lower()
    pta_tokens = ("pta", "torch", "torchrun", "pta-load", "pta-save")
    ms_tokens = ("msa", "ms", "mindspore", "msrun", "msa-load")
    mf_markers = (
        "mf执行",
        "mf未",
        "mf ",
        " mf",
        "mf_",
        "mf-",
        "(mf",
        "/mf",
        "training_log_mf",
    )

    has_pta = any(token in text for token in pta_tokens)
    has_ms = any(token in text for token in ms_tokens) or any(marker in text for marker in mf_markers) or text.startswith("mf")

    if has_pta:
        return "PTA问题"
    if has_ms and not has_pta:
        return "MS问题"
    return "待定"


def _functional_owner_from_record(record: "IterationAnalysis") -> str:
    candidates: list[str] = []
    for reason in record.functional_reasons:
        candidates.extend([reason.issue_subtype, reason.message])
    for signal in record.issue_signals:
        if signal.category != "功能问题":
            continue
        candidates.extend([signal.message, signal.log_path])
    return _functional_owner_from_text(*candidates)


def _extract_stack_keyword(message: str) -> str:
    if not message:
        return "-"
    for line in message.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        matched = re.search(r"([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))", stripped)
        if matched:
            return matched.group(1)
        if stripped.startswith("Traceback"):
            return "Traceback"
        if "[ERROR]" in stripped.upper() or stripped.lower().startswith("error:"):
            return stripped[:120]
    first = next((line.strip() for line in message.splitlines() if line.strip()), "")
    return first[:120] if first else "-"


def _normalize_functional_issue_message(message: str) -> str:
    if not message:
        return ""

    normalized = message
    normalized = re.sub(r"\[rank\d+\]", "[rank*]", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bnpu:\d+\b", "npu:*", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b(rank)\d+\b", r"\1*", normalized, flags=re.IGNORECASE)
    return normalized


def _functional_issue_group_key(message: str) -> str:
    normalized = _normalize_functional_issue_message(message)
    return re.sub(r"\s+", "", normalized).lower()


@dataclass
class IterationAnalysis:
    task_type: int
    iteration: int
    iteration_tag: str
    iteration_dir: str
    failed_flag: bool
    mutation_success: bool
    comparison_available: bool
    overall_status: str
    pta_execution_success: bool = False
    msa_execution_success: bool = False
    mf_execution_success: Optional[bool] = None
    categories: list[str] = field(default_factory=list)
    functional_reasons: list[FunctionalReason] = field(default_factory=list)
    issue_signals: list[IssueSignal] = field(default_factory=list)
    pta_csv: Optional[str] = None
    msa_csv: Optional[str] = None
    pta_step_csv: Optional[str] = None
    msa_step_csv: Optional[str] = None
    mf_step_csv: Optional[str] = None
    pta_avg_step_time_skip1: Optional[float] = None
    msa_avg_step_time_skip1: Optional[float] = None
    performance_delta_seconds: Optional[float] = None
    performance_delta_ratio: Optional[float] = None
    performance_severity: str = "UNAVAILABLE"
    performance_source: str = "single_row"
    max_loss_diff: Optional[float] = None
    avg_loss_diff: Optional[float] = None
    precision_severity: str = "UNAVAILABLE"
    pta_max_memory_mb: Optional[float] = None
    msa_max_memory_mb: Optional[float] = None
    memory_delta_mb: Optional[float] = None
    memory_delta_ratio: Optional[float] = None
    memory_severity: str = "UNAVAILABLE"
    runtime_log_paths: list[str] = field(default_factory=list)
    msrun_log_paths: list[str] = field(default_factory=list)
    reproduction_dir: Optional[str] = None
    iteration_report_path: Optional[str] = None
    mutation_input_paths: list[str] = field(default_factory=list)
    mf_csv: Optional[str] = None
    mf_valid: Optional[bool] = None
    pta_loss: Optional[float] = None
    msa_loss: Optional[float] = None
    mf_loss: Optional[float] = None
    status_file: Optional[str] = None
    status_components: dict[str, str] = field(default_factory=dict)
    step_results: dict[str, str] = field(default_factory=dict)
    compare_mode: Optional[str] = None


@dataclass
class AnalysisArtifacts:
    task_type: int
    output_root: str
    run_dir: str
    analysis_dir: str
    summary_json: str
    summary_markdown: str
    report_html: str
    iteration_csv: str
    issue_json: str
    svg_assets: list[str]
    repro_root: str
    iteration_report_root: str
    executed_iterations: int
    planned_iterations: Optional[int]
    mutation_success_count: int
    mutation_success_rate: float
    pta_execution_success_count: int
    pta_execution_success_rate: float
    msa_execution_success_count: int
    msa_execution_success_rate: Optional[float]
    valid_comparisons: int
    functional_failures: int
    precision_failures: int
    performance_failures: int
    memory_failures: int
    mf_failures: int


def _path_text(path: Path | str | None) -> Optional[str]:
    if path is None:
        return None
    return str(Path(path).resolve())


def _task_profile(task_type: int) -> dict:
    return TASK_PROFILES[task_type]


def _load_output_config(output_root: Path) -> dict:
    config_path = output_root / "config.json"
    if not config_path.exists():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _resolve_run_dir(output_root: Path, run_dir: Optional[Path]) -> Path:
    if run_dir is not None:
        return run_dir.resolve()

    nested_iter_root = output_root / "iters"
    direct_iter_dirs = _iter_dirs(nested_iter_root)
    if direct_iter_dirs:
        return nested_iter_root.resolve()

    direct_iter_dirs = _iter_dirs(output_root)
    if direct_iter_dirs:
        return output_root.resolve()

    legacy_log_root = output_root / "legacy_log"
    if not legacy_log_root.exists():
        raise FileNotFoundError(f"未找到可分析的迭代目录: {output_root}")

    candidates = [item for item in legacy_log_root.iterdir() if item.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"未找到可分析的 run 目录: {legacy_log_root}")

    candidates.sort(key=lambda item: (item.stat().st_mtime, item.name), reverse=True)
    return candidates[0].resolve()


def _detect_task_type(run_dir: Path) -> int:
    name = run_dir.name.lower()
    if name.startswith("task2_"):
        return 2
    if name.startswith("task3_"):
        return 3
    if name.startswith("task1_"):
        return 1

    iter_dirs = _iter_dirs(run_dir)
    for iter_dir in iter_dirs[:3]:
        status_payload = _read_status_payload(iter_dir)
        task_type = status_payload.get("task_type")
        if isinstance(task_type, int) and task_type in TASK_PROFILES:
            return task_type
    return 1


def _normalize_model_name_value(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        if "," in value:
            items = [part.strip() for part in value.split(",") if part.strip()]
            if items:
                return ",".join(Path(item).stem for item in items)
        return value.strip() or None
    if isinstance(value, (list, tuple)):
        items = [str(item).strip() for item in value if str(item).strip()]
        if items:
            return ",".join(Path(item).stem for item in items)
    return str(value).strip() or None


def _resolve_run_metadata(
    output_root: Path,
    run_dir: Path,
    task_type: int,
    model_name: Optional[str],
    planned_iterations: Optional[int],
) -> tuple[str, Optional[int]]:
    config = _load_output_config(output_root)
    task_config = config.get("tasks", {}).get(str(task_type), {})

    resolved_model_name = model_name
    if resolved_model_name is None:
        resolved_model_name = _normalize_model_name_value(
            task_config.get("MODEL_NAME") or task_config.get("MODELS")
        )
    if resolved_model_name is None:
        if task_type == 1 and "_" in run_dir.name:
            resolved_model_name = run_dir.name.split("_", 1)[0]
        else:
            resolved_model_name = run_dir.name

    resolved_planned_iters = planned_iterations
    if resolved_planned_iters is None:
        raw_iters = task_config.get("TOTAL_ITER")
        try:
            resolved_planned_iters = int(raw_iters) if raw_iters is not None else None
        except (TypeError, ValueError):
            resolved_planned_iters = None

    return resolved_model_name, resolved_planned_iters


def _load_task_config(output_root: Path, task_type: int) -> dict:
    config = _load_output_config(output_root)
    task_config = config.get("tasks", {}).get(str(task_type), {})
    return task_config if isinstance(task_config, dict) else {}


def _extract_iteration_number(path: Path) -> int:
    match = re.search(r"iter_(\d+)$", path.name)
    if not match:
        raise ValueError(f"非法迭代目录名: {path}")
    return int(match.group(1))


def _iter_dirs(run_dir: Path) -> list[Path]:
    if not run_dir.exists():
        return []
    dirs = [item for item in run_dir.iterdir() if item.is_dir() and re.match(r"iter_\d+$", item.name)]
    return sorted(dirs, key=_extract_iteration_number)


def _find_current_csv(iter_dir: Path, task_type: int, kind: str, iteration: int) -> Optional[Path]:
    relpath = _task_profile(task_type)["csv_relpaths"].get(kind)
    if not relpath:
        return None
    csv_path = iter_dir / relpath.format(iteration=iteration)
    if csv_path.exists():
        return csv_path.resolve()
    runtime_logs_dir = _iter_material_dir(iter_dir, "runtime_logs")
    if runtime_logs_dir is not None:
        runtime_csv = runtime_logs_dir / Path(relpath.format(iteration=iteration)).name
        if runtime_csv.exists():
            return runtime_csv.resolve()
    return None


def _find_step_csv(iter_dir: Path, task_type: int, kind: str, iteration: int) -> Optional[Path]:
    relpath = (_task_profile(task_type).get("step_csv_relpaths") or {}).get(kind)
    if not relpath:
        return None
    csv_path = iter_dir / relpath.format(iteration=iteration)
    if csv_path.exists():
        return csv_path.resolve()
    runtime_logs_dir = _iter_material_dir(iter_dir, "runtime_logs")
    if runtime_logs_dir is not None:
        runtime_csv = runtime_logs_dir / Path(relpath.format(iteration=iteration)).name
        if runtime_csv.exists():
            return runtime_csv.resolve()
    return None


def _iter_material_dir(iter_dir: Path, name: str) -> Optional[Path]:
    for candidate in (iter_dir / name, iter_dir / "core_backup" / name, iter_dir / "artifacts" / name):
        if candidate.exists():
            return candidate.resolve()
    return None


def _iter_aux_file(iter_dir: Path, name: str) -> Path:
    direct = iter_dir / name
    if direct.exists():
        return direct
    return iter_dir / "core_backup" / name


def _find_mutation_inputs(iter_dir: Path, iteration: int) -> list[Path]:
    mutation_dir = _iter_material_dir(iter_dir, "mutation_inputs")
    if mutation_dir is not None:
        files = [item.resolve() for item in sorted(mutation_dir.iterdir()) if item.is_file()]
        if files:
            return files

    res_dir = iter_dir / "res"
    if not res_dir.exists():
        return []

    collected: list[Path] = []
    model_dirs = [item for item in res_dir.iterdir() if item.is_dir()]
    for model_dir in model_dirs:
        if model_dir.name.startswith("training_log_") or model_dir.name == "accuracy_log":
            continue
        candidates = [
            model_dir / f"mutating-{iteration}.json",
            model_dir / f"mutated_config_iter_{iteration:03d}.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                collected.append(candidate.resolve())
    return sorted(collected)


def _pick_column(fieldnames: Iterable[str], candidates: Iterable[str]) -> str:
    names = list(fieldnames)
    lowered = {name.lower(): name for name in names}
    for candidate in candidates:
        if candidate in names:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    raise KeyError(f"未找到目标列，候选={list(candidates)}，实际={names}")


def _pick_optional_column(fieldnames: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    try:
        return _pick_column(fieldnames, candidates)
    except KeyError:
        return None


def _parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == "-":
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _read_training_csv(csv_path: Path) -> list[dict[str, float]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        step_key = _pick_column(reader.fieldnames, rules.ITERATION_COLUMN_NAMES)
        time_key = _pick_column(reader.fieldnames, rules.TIME_COLUMN_NAMES)
        loss_key = _pick_column(reader.fieldnames, rules.LOSS_COLUMN_NAMES)
        memory_key = _pick_column(reader.fieldnames, rules.MEMORY_COLUMN_NAMES)

        rows: list[dict[str, float]] = []
        for row in reader:
            try:
                rows.append(
                    {
                        "iteration": float(row[step_key]),
                        "time": float(row[time_key]),
                        "loss": float(row[loss_key]),
                        "memory": float(row[memory_key]),
                    }
                )
            except (TypeError, ValueError, KeyError):
                continue
        return rows


def _training_csv_has_valid_rows(csv_path: Path | None) -> bool:
    if csv_path is None:
        return False
    try:
        return bool(_read_training_csv(csv_path))
    except Exception:
        return False


def _read_single_iteration_metrics(csv_path: Path, iteration: int) -> Optional[dict[str, Optional[float]]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return None

        iteration_key = _pick_column(reader.fieldnames, rules.ITERATION_COLUMN_NAMES)
        time_key = _pick_optional_column(reader.fieldnames, rules.TIME_COLUMN_NAMES)
        loss_key = _pick_optional_column(reader.fieldnames, rules.LOSS_COLUMN_NAMES)
        memory_key = _pick_optional_column(reader.fieldnames, rules.MEMORY_COLUMN_NAMES)

        for row in reader:
            row_iteration = _parse_float(row.get(iteration_key))
            if row_iteration is None or int(row_iteration) != int(iteration):
                continue

            metrics = {
                "iteration": float(iteration),
                "time": _parse_float(row.get(time_key)) if time_key else None,
                "loss": _parse_float(row.get(loss_key)) if loss_key else None,
                "memory": _parse_float(row.get(memory_key)) if memory_key else None,
            }
            if all(metrics[key] is None for key in ("time", "loss", "memory")):
                return None
            return metrics
    return None


def _index_rows(rows: list[dict[str, float]]) -> dict[int, dict[str, float]]:
    indexed: dict[int, dict[str, float]] = {}
    for row in rows:
        indexed[int(row["iteration"])] = row
    return indexed


def _safe_mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _relative_delta(reference: Optional[float], current: Optional[float]) -> Optional[float]:
    if reference in (None, 0.0) or current is None:
        return None
    return (current - reference) / reference


def _regression_value(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return max(value, 0.0)


def _severity(metric_name: str, value: Optional[float]) -> str:
    if value is None:
        return "UNAVAILABLE"
    if isinstance(value, float) and not math.isfinite(value):
        return "CRITICAL"
    for threshold, label in rules.SEVERITY_LEVELS[metric_name]:
        if value <= threshold:
            return label
    return "CRITICAL"


def _is_problem_severity(value: object) -> bool:
    return str(value or "UNAVAILABLE").strip().upper() != "PASS"


def _is_metric_issue_severity(value: object) -> bool:
    normalized = str(value or "UNAVAILABLE").strip().upper()
    return normalized not in {"PASS", "UNAVAILABLE"}


def _empty_comparison_metrics() -> dict[str, Optional[float] | str]:
    return {
        "pta_avg_step_time_skip1": None,
        "msa_avg_step_time_skip1": None,
        "performance_delta_seconds": None,
        "performance_delta_ratio": None,
        "performance_severity": "UNAVAILABLE",
        "max_loss_diff": None,
        "avg_loss_diff": None,
        "precision_severity": "UNAVAILABLE",
        "pta_max_memory_mb": None,
        "msa_max_memory_mb": None,
        "memory_delta_mb": None,
        "memory_delta_ratio": None,
        "memory_severity": "UNAVAILABLE",
    }


def _override_performance_metrics(
    base_metrics: dict[str, Optional[float] | str],
    perf_metrics: dict[str, Optional[float] | str],
) -> dict[str, Optional[float] | str]:
    merged = dict(base_metrics)
    for key in (
        "pta_avg_step_time_skip1",
        "msa_avg_step_time_skip1",
        "performance_delta_seconds",
        "performance_delta_ratio",
        "performance_severity",
    ):
        merged[key] = perf_metrics.get(key)
    return merged


def _compare_series_csvs(pta_csv: Path, msa_csv: Path) -> dict[str, Optional[float] | str]:
    pta_rows = _read_training_csv(pta_csv)
    msa_rows = _read_training_csv(msa_csv)
    pta_index = _index_rows(pta_rows)
    msa_index = _index_rows(msa_rows)
    common_steps = sorted(set(pta_index) & set(msa_index))
    common_steps_skip1 = [step for step in common_steps if step > 1]

    pta_step_values = [pta_index[step]["time"] for step in common_steps_skip1]
    msa_step_values = [msa_index[step]["time"] for step in common_steps_skip1]
    pta_avg = _safe_mean(pta_step_values)
    msa_avg = _safe_mean(msa_step_values)
    perf_delta_ratio = _relative_delta(pta_avg, msa_avg)

    loss_diffs = [abs(pta_index[step]["loss"] - msa_index[step]["loss"]) for step in common_steps]
    max_loss_diff = max(loss_diffs) if loss_diffs else None
    avg_loss_diff = _safe_mean(loss_diffs)

    pta_max_memory = max((row["memory"] for row in pta_rows), default=None)
    msa_max_memory = max((row["memory"] for row in msa_rows), default=None)
    memory_delta_ratio = _relative_delta(pta_max_memory, msa_max_memory)

    performance_level_value = _regression_value(perf_delta_ratio)
    precision_level_value = max_loss_diff
    memory_level_value = _regression_value(memory_delta_ratio)

    return {
        "pta_avg_step_time_skip1": pta_avg,
        "msa_avg_step_time_skip1": msa_avg,
        "performance_delta_seconds": (
            msa_avg - pta_avg if pta_avg is not None and msa_avg is not None else None
        ),
        "performance_delta_ratio": perf_delta_ratio,
        "performance_severity": _severity("performance", performance_level_value),
        "max_loss_diff": max_loss_diff,
        "avg_loss_diff": avg_loss_diff,
        "precision_severity": _severity("precision", precision_level_value),
        "pta_max_memory_mb": pta_max_memory,
        "msa_max_memory_mb": msa_max_memory,
        "memory_delta_mb": (
            msa_max_memory - pta_max_memory
            if pta_max_memory is not None and msa_max_memory is not None
            else None
        ),
        "memory_delta_ratio": memory_delta_ratio,
        "memory_severity": _severity("memory", memory_level_value),
    }


def _compare_single_iteration(
    pta_metrics: dict[str, Optional[float]],
    msa_metrics: dict[str, Optional[float]],
) -> dict[str, Optional[float] | str]:
    pta_time = pta_metrics.get("time")
    msa_time = msa_metrics.get("time")
    perf_delta_ratio = _relative_delta(pta_time, msa_time)

    pta_loss = pta_metrics.get("loss")
    msa_loss = msa_metrics.get("loss")
    max_loss_diff = (
        abs(pta_loss - msa_loss)
        if pta_loss is not None and msa_loss is not None
        else None
    )

    pta_memory = pta_metrics.get("memory")
    msa_memory = msa_metrics.get("memory")
    memory_delta_ratio = _relative_delta(pta_memory, msa_memory)

    performance_level_value = _regression_value(perf_delta_ratio)
    precision_level_value = max_loss_diff
    memory_level_value = _regression_value(memory_delta_ratio)

    return {
        "pta_avg_step_time_skip1": pta_time,
        "msa_avg_step_time_skip1": msa_time,
        "performance_delta_seconds": (
            msa_time - pta_time if pta_time is not None and msa_time is not None else None
        ),
        "performance_delta_ratio": perf_delta_ratio,
        "performance_severity": _severity("performance", performance_level_value),
        "max_loss_diff": max_loss_diff,
        "avg_loss_diff": max_loss_diff,
        "precision_severity": _severity("precision", precision_level_value),
        "pta_max_memory_mb": pta_memory,
        "msa_max_memory_mb": msa_memory,
        "memory_delta_mb": (
            msa_memory - pta_memory if pta_memory is not None and msa_memory is not None else None
        ),
        "memory_delta_ratio": memory_delta_ratio,
        "memory_severity": _severity("memory", memory_level_value),
    }


def _read_last_training_metrics(csv_path: Path) -> Optional[dict[str, Optional[float]]]:
    rows = _read_training_csv(csv_path)
    if not rows:
        return None
    last = rows[-1]
    return {
        "iteration": last.get("iteration"),
        "time": last.get("time"),
        "loss": last.get("loss"),
        "memory": last.get("memory"),
    }


def _load_metrics_from_paths(
    task_type: int,
    iteration: int,
    primary_csv: Path | None,
    step_csv: Path | None,
) -> Optional[dict[str, Optional[float]]]:
    profile = _task_profile(task_type)
    if profile["comparison_mode"] == "series":
        source = step_csv or primary_csv
        return _read_last_training_metrics(source) if source is not None else None

    metrics = _read_single_iteration_metrics(primary_csv, iteration) if primary_csv else None
    if metrics is None and step_csv is not None:
        metrics = _read_last_training_metrics(step_csv)
    return metrics


def _normalize_message(line: str) -> str:
    normalized_lines: list[str] = []
    for raw_line in line.splitlines():
        text = raw_line.strip()
        if re.match(r"^\[(?:INFO|DEBUG)\]\s", text, flags=re.IGNORECASE):
            continue
        if "Event base dispatch success" in text:
            continue
        text = re.sub(r"\b\d{4}-\d{2}-\d{2}[- :]\d{2}:\d{2}:\d{2}(?:\.\d+)?\b", "", text)
        text = re.sub(r"\[RANK=\d+\]", "[RANK=?]", text)
        text = re.sub(r"\b[Rr]ank\s+\d+\b", "Rank ?", text)
        text = re.sub(r"\bworker_\d+\.log\b", "worker_?.log", text)
        text = re.sub(r"0x[0-9a-fA-F]+", "0x?", text)
        text = re.sub(r"[ \t]+", " ", text).strip()
        if text:
            normalized_lines.append(text)
    return "\n".join(normalized_lines)


def _should_skip_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    for token in rules.IGNORE_LINE_KEYWORDS:
        if token in stripped:
            return True
    if stripped.startswith(("export ", "source ", "conda activate ", "if [ ", "fi", "bash ")):
        return True
    return False


def _scan_log_for_signals(log_path: Path) -> list[IssueSignal]:
    signals: list[IssueSignal] = []
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
            lines = handle.readlines()
    except FileNotFoundError:
        return []

    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        if _should_skip_line(line):
            line_index += 1
            continue
        category = _match_issue_category(line)
        if category is None:
            line_index += 1
            continue

        block_message, end_index = _extract_error_block(lines, line_index)
        signals.append(
            IssueSignal(
                category=category,
                message=_normalize_message(block_message),
                log_path=_path_text(log_path) or str(log_path),
                line_number=line_index + 1,
            )
        )
        if len(signals) >= rules.MAX_SIGNAL_PER_FILE:
            break
        line_index = end_index + 1
    return signals


def _read_status_payload(iter_dir: Path) -> dict:
    status_path = iter_dir / "status.json"
    if not status_path.exists():
        return {}
    try:
        with status_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _collect_log_paths(iter_dir: Path) -> tuple[list[Path], list[Path]]:
    runtime_logs: list[Path] = []
    seen_names: set[str] = set()
    for candidate_root in filter(None, (_iter_material_dir(iter_dir, "runtime_logs"),)):
        for path in sorted(candidate_root.glob("*.log")):
            if path.name.startswith("analyse_"):
                continue
            if path.name in seen_names:
                continue
            seen_names.add(path.name)
            runtime_logs.append(path.resolve())
    runtime_root_logs = sorted(iter_dir.glob("*.log"))
    runtime_all = sorted({path.resolve() for path in runtime_logs + runtime_root_logs}, key=str)
    msrun_logs: list[Path] = []
    seen_msrun: set[Path] = set()
    for root_name in ("msrun_log", "output_msrun_log"):
        msrun_root = _iter_material_dir(iter_dir, root_name)
        if msrun_root is None:
            continue
        for path in sorted(msrun_root.rglob("*.log")):
            resolved = path.resolve()
            if resolved in seen_msrun:
                continue
            seen_msrun.add(resolved)
            msrun_logs.append(resolved)
    return runtime_all, msrun_logs


def _read_failure_info(iter_dir: Path) -> list[str]:
    failure_info_path = _iter_aux_file(iter_dir, "failure_info.txt")
    if not failure_info_path.exists():
        return []
    try:
        lines = failure_info_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []

    reasons: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("FAILED_COMPONENTS:"):
            continue
        if stripped.startswith("REASON:"):
            stripped = stripped.split("REASON:", 1)[1].strip()
        if stripped:
            reasons.append(stripped)
    return reasons


def _component_is_failure(status: object) -> bool:
    if status is None:
        return False
    text = str(status).strip().upper()
    return text not in {"", "NONE", "NULL", "OK", "PASS", "SKIP", "DISABLED"}


def _component_is_success(status: object) -> bool:
    if status is None:
        return False
    return str(status).strip().upper() in {"OK", "PASS"}


def _component_is_skipped(status: object) -> bool:
    if status is None:
        return True
    return str(status).strip().upper() in {"", "NONE", "NULL", "SKIP", "DISABLED"}


def _step_result(success: bool, skipped: bool = False) -> str:
    if success:
        return "成功"
    if skipped:
        return "未执行"
    return "失败"


def _detect_script_generation_success(
    iter_dir: Path,
    task_type: int,
    iteration: int,
    status_components: dict[str, str],
    msa_expected: bool,
    mf_expected: bool,
) -> tuple[bool, bool]:
    script_dir = _iter_material_dir(iter_dir, "scripts") or (iter_dir / "scripts")
    script_dir_exists = script_dir.exists()
    expected_scripts = [
        f"pta-save_iter{iteration}.sh",
        f"pta-load_iter{iteration}.sh",
    ]
    if msa_expected:
        expected_scripts.append(f"msa-load_iter{iteration}.sh")
    if mf_expected:
        expected_scripts.append(f"mf_iter{iteration}.sh")
        if not _component_is_skipped(status_components.get("MF")) and (script_dir / f"convert_iter{iteration}.sh").exists():
            expected_scripts.append(f"convert_iter{iteration}.sh")

    if task_type not in {2, 3}:
        return False, True
    if not script_dir_exists:
        return False, False
    if not expected_scripts:
        return False, True
    return True, all((script_dir / name).exists() for name in expected_scripts)


def _build_step_results(
    iter_dir: Path,
    task_type: int,
    iteration: int,
    compare_mode: Optional[str],
    status_components: dict[str, str],
    mutation_success: bool,
    pta_execution_success: bool,
    msa_execution_success: bool,
    mf_execution_success: Optional[bool],
) -> dict[str, str]:
    normalized_compare_mode = (compare_mode or "").strip().lower()
    compare_to_mf = task_type in {1, 2, 3} and normalized_compare_mode == "pta_mf"
    msa_expected = not _component_is_skipped(status_components.get("MSA_LOAD"))
    mf_expected = not _component_is_skipped(status_components.get("MF"))
    scripts_applicable, scripts_ok = _detect_script_generation_success(
        iter_dir,
        task_type,
        iteration,
        status_components,
        msa_expected,
        mf_expected,
    )

    steps = {
        "变异": _step_result(
            mutation_success,
            skipped=not status_components and not _iter_aux_file(iter_dir, "FAILED_FLAG").exists(),
        ),
        "生成脚本": _step_result(
            scripts_ok,
            skipped=not scripts_applicable,
        ),
        "执行PTA-SAVE训练": _step_result(
            _component_is_success(status_components.get("PTA_SAVE")),
            skipped=_component_is_skipped(status_components.get("PTA_SAVE")),
        ),
        "执行PTA-LOAD训练": _step_result(
            pta_execution_success,
            skipped=_component_is_skipped(status_components.get("PTA_LOAD")) and not pta_execution_success,
        ),
    }
    if not compare_to_mf and (msa_expected or "MSA_LOAD" in status_components or task_type in {2, 3}):
        steps["执行MSA-LOAD训练"] = _step_result(
            msa_execution_success,
            skipped=not msa_expected and not msa_execution_success,
        )
    if compare_to_mf or mf_expected or mf_execution_success is not None:
        steps["执行MF权重转换与训练"] = _step_result(
            bool(mf_execution_success),
            skipped=mf_execution_success is None,
        )
    return steps


def _dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        text = str(item).strip()
        canonical = re.sub(r"\s+", "", text).lower()
        if not text or canonical in seen:
            continue
        seen.add(canonical)
        output.append(text)
    return output


def _dedupe_functional_reasons(items: Iterable[FunctionalReason]) -> list[FunctionalReason]:
    seen: set[tuple[str, str]] = set()
    output: list[FunctionalReason] = []
    for item in items:
        subtype = item.issue_subtype.strip()
        message = item.message.strip()
        canonical = (subtype, re.sub(r"\s+", "", message).lower())
        if not subtype or not message or canonical in seen:
            continue
        seen.add(canonical)
        output.append(FunctionalReason(issue_subtype=subtype, message=message))
    return output


def _functional_reason_text(reason: FunctionalReason) -> str:
    return f"{reason.issue_subtype}: {reason.message}"


def _match_issue_category(text: str) -> Optional[str]:
    lowered = text.lower()
    for category, patterns in rules.ISSUE_KEYWORDS.items():
        if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in patterns):
            return category
    return None


def _is_noise_followup_line(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if re.match(r"^\[(?:INFO|DEBUG)\]\s", stripped, flags=re.IGNORECASE):
        return True
    if "Event base dispatch success" in stripped:
        return True
    return False


def _extract_error_block(lines: list[str], start_index: int) -> tuple[str, int]:
    block: list[str] = []
    index = start_index

    while index < len(lines) and len(block) < rules.MAX_ERROR_BLOCK_LINES:
        raw_line = lines[index].rstrip("\n")
        stripped = raw_line.strip()

        if not block:
            block.append(raw_line)
            index += 1
            continue

        if not stripped:
            break
        if _is_noise_followup_line(stripped):
            break

        block.append(raw_line)
        index += 1

    while block and not block[-1].strip():
        block.pop()
    return "\n".join(block).strip(), max(index - 1, start_index)


def _should_suppress_keyword_signals(mutation_success: bool, comparison_available: bool) -> bool:
    """训练成功且存在正常对比数据时，不再使用日志关键词做异常告警。"""
    return mutation_success and comparison_available


def _filter_suppressed_signals(
    signals: list[IssueSignal],
    mutation_success: bool,
    comparison_available: bool,
) -> list[IssueSignal]:
    if not _should_suppress_keyword_signals(mutation_success, comparison_available):
        return signals
    # 成功轮次仍保留明确的功能异常，避免 Bad-CRC32 这类报错被吞掉。
    return [signal for signal in signals if signal.category == "功能问题"]


def _should_ignore_keyword_only_functional_signals(
    task_type: int,
    comparison_available: bool,
    comparison_metrics: dict[str, Optional[float] | str],
    functional_reasons: list[FunctionalReason],
) -> bool:
    if task_type not in {1, 2, 3}:
        return False
    if not comparison_available or functional_reasons:
        return False
    return (
        str(comparison_metrics.get("precision_severity") or "UNAVAILABLE") == "PASS"
        and str(comparison_metrics.get("performance_severity") or "UNAVAILABLE") == "PASS"
        and str(comparison_metrics.get("memory_severity") or "UNAVAILABLE") == "PASS"
    )


def _suppress_keyword_only_functional_signals(
    task_type: int,
    signals: list[IssueSignal],
    comparison_available: bool,
    comparison_metrics: dict[str, Optional[float] | str],
    functional_reasons: list[FunctionalReason],
) -> list[IssueSignal]:
    if not _should_ignore_keyword_only_functional_signals(
        task_type=task_type,
        comparison_available=comparison_available,
        comparison_metrics=comparison_metrics,
        functional_reasons=functional_reasons,
    ):
        return signals
    return [signal for signal in signals if signal.category != "功能问题"]


def _is_generic_functional_reason(reason: FunctionalReason) -> bool:
    message = reason.message.strip()
    return message.endswith("执行失败") or message.endswith("执行失败。") or message.endswith("未生成当前轮有效结果。")


def _select_issue_messages_for_record(
    record: IterationAnalysis,
    repro_hints: list[str],
) -> list[tuple[str, str, str, list[str]]]:
    messages: list[tuple[str, str, str, list[str]]] = []
    functional_signals = [signal for signal in record.issue_signals if signal.category == "功能问题"]

    if functional_signals:
        for signal in functional_signals:
            messages.append((signal.category, "", signal.message, repro_hints))
        for reason in record.functional_reasons:
            if _is_generic_functional_reason(reason):
                continue
            messages.append(("功能问题", reason.issue_subtype, reason.message, repro_hints))
    else:
        for reason in record.functional_reasons:
            messages.append(("功能问题", reason.issue_subtype, reason.message, repro_hints))

    for signal in record.issue_signals:
        if signal.category == "功能问题":
            continue
        messages.append((signal.category, "", signal.message, repro_hints))

    return messages


def _derive_functional_reasons(
    iter_dir: Path,
    task_type: int,
    compare_mode: Optional[str],
    mutation_success: bool,
    pta_valid: bool,
    msa_valid: bool,
    mf_required: bool,
    mf_valid: Optional[bool],
    status_payload: dict,
) -> list[FunctionalReason]:
    reasons: list[FunctionalReason] = []
    normalized_compare_mode = (compare_mode or "").strip().lower()
    compare_to_mf = task_type in {1, 2, 3} and normalized_compare_mode == "pta_mf"
    status_reason = str(status_payload.get("reason") or "").strip()
    status_overall = str(status_payload.get("overall_status") or "").strip()
    components = {
        str(key): str(value)
        for key, value in (status_payload.get("components") or {}).items()
    }

    if status_overall in {"FAILED", "ERROR"} and status_reason not in GENERIC_COMPLETION_REASONS:
        reasons.append(FunctionalReason(issue_subtype="执行失败", message=status_reason))

    component_details = {
        "MUTATE": ("工具变异失败", "MUTATE 执行失败"),
        "PTA_SAVE": ("PTA保存权重失败", "PTA-SAVE 执行失败"),
        "PTA_LOAD": ("PTA加载执行失败", "PTA-LOAD 执行失败"),
        "MSA_LOAD": ("MSA加载执行失败", "MSA-LOAD 执行失败"),
        "MF": ("MF执行失败", "MF 执行失败"),
    }
    component_sequence = ["MUTATE", "PTA_SAVE", "PTA_LOAD"]
    if not compare_to_mf:
        component_sequence.append("MSA_LOAD")
    for component in component_sequence:
        if _component_is_failure(components.get(component)):
            subtype, message = component_details[component]
            reasons.append(FunctionalReason(issue_subtype=subtype, message=message))

    if (task_type == 3 and mf_required) or compare_to_mf:
        if _component_is_failure(components.get("MF")):
            subtype, message = component_details["MF"]
            reasons.append(FunctionalReason(issue_subtype=subtype, message=message))

    if _component_is_failure(components.get("MUTATE")):
        reasons.append(
            FunctionalReason(
                issue_subtype="工具变异失败",
                message="MUTATE 步骤执行失败。",
            )
        )
    elif not pta_valid:
        reasons.append(FunctionalReason(issue_subtype="PTA加载执行失败", message="PTA-LOAD 未生成当前轮有效结果。"))
    elif compare_to_mf and mf_valid is False and not _component_is_failure(components.get("MF")):
        subtype, message = component_details["MF"]
        reasons.append(FunctionalReason(issue_subtype=subtype, message=message))
    elif not compare_to_mf and not msa_valid and not _component_is_failure(components.get("MSA_LOAD")):
        reasons.append(FunctionalReason(issue_subtype="MSA加载执行失败", message="MSA-LOAD 未生成当前轮有效结果。"))

    if task_type == 3 and mf_required and mf_valid is False and not _component_is_failure(components.get("MF")):
        reasons.append(FunctionalReason(issue_subtype="MF执行失败", message="MF 未生成当前轮有效结果。"))

    reasons.extend(
        FunctionalReason(issue_subtype="执行失败", message=reason)
        for reason in _read_failure_info(iter_dir)
    )

    if not reasons and _iter_aux_file(iter_dir, "FAILED_FLAG").exists():
        reasons.append(
            FunctionalReason(issue_subtype="执行失败", message="存在 FAILED_FLAG，但未解析到更具体的失败信息。")
        )

    return _dedupe_functional_reasons(reasons)


def _has_pta_functional_failure(functional_reasons: list[FunctionalReason]) -> bool:
    text = " ".join(
        f"{reason.issue_subtype} {reason.message}"
        for reason in functional_reasons
    ).lower()
    return any(token in text for token in ("pta", "torchrun", "pta-load", "pta-save"))


def _categorize_iteration(
    mutation_success: bool,
    comparison_available: bool,
    failed_flag: bool,
    comparison_metrics: dict[str, Optional[float] | str],
    issue_signals: list[IssueSignal],
    functional_reasons: list[FunctionalReason],
) -> tuple[list[str], str]:
    categories: list[str] = []
    execution_failed = failed_flag and not comparison_available
    pta_functional_failure = _has_pta_functional_failure(functional_reasons)

    if _is_metric_issue_severity(comparison_metrics.get("precision_severity")):
        categories.append("精度问题")

    if _is_metric_issue_severity(comparison_metrics.get("performance_severity")):
        categories.append("性能问题")

    if _is_metric_issue_severity(comparison_metrics.get("memory_severity")):
        categories.append("显存问题")

    signal_categories = {signal.category for signal in issue_signals}
    if (
        "显存问题" in signal_categories
        and "显存问题" not in categories
        and not (pta_functional_failure and execution_failed)
    ):
        categories.append("显存问题")
    if (
        "性能问题" in signal_categories
        and "性能问题" not in categories
        and not execution_failed
        and comparison_metrics.get("msa_avg_step_time_skip1") is None
    ):
        categories.append("性能问题")

    if functional_reasons or "功能问题" in signal_categories or (failed_flag and not comparison_available):
        categories.append("功能问题")

    categories = sorted(set(categories), key=lambda item: SIGNAL_CATEGORY_ORDER.get(item, 99))

    if not mutation_success:
        overall_status = "MUTATION_FAILED"
    elif not comparison_available:
        overall_status = "EXECUTION_FAILED"
    elif categories:
        overall_status = "COMPLETED_WITH_ISSUES"
    else:
        overall_status = "PASS"

    return categories, overall_status


def _copytree_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _copy_file_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_iteration_snapshots(iter_dir: Path, dst_root: Path) -> None:
    for candidate in sorted(iter_dir.iterdir()):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in {".csv", ".txt", ".log", ".json"}:
            continue
        _copy_file_if_exists(candidate, dst_root / "snapshots" / candidate.name)


def _normalize_repro_data_name(src: Path, data_key: str, iteration: int) -> str:
    if data_key == "pta_csv":
        return f"training_log_pta-{iteration}{src.suffix}"
    if data_key == "msa_csv":
        return f"training_log_msa-{iteration}{src.suffix}"
    if data_key == "mf_csv":
        return f"training_log_mf-{iteration}{src.suffix}"
    return src.name


def _export_failed_repro(
    iter_dir: Path,
    iteration: int,
    repro_root: Path,
    mutation_inputs: list[Path],
    extra_files: list[tuple[str, Path]],
) -> Optional[Path]:
    repro_root.mkdir(parents=True, exist_ok=True)
    dst_root = repro_root / f"iter_{iteration}"
    dst_root.mkdir(parents=True, exist_ok=True)

    scripts_src = _iter_material_dir(iter_dir, "scripts")
    if scripts_src is not None:
        _copytree_if_exists(scripts_src, dst_root / "scripts")

    weights_src = _iter_material_dir(iter_dir, "weights")
    weights_dst = dst_root / "weights"
    if weights_src is not None:
        _copytree_if_exists(weights_src, weights_dst)

    for mutation_input in mutation_inputs:
        _copy_file_if_exists(mutation_input, dst_root / "mutation_inputs" / mutation_input.name)

    runtime_logs, msrun_logs = _collect_log_paths(iter_dir)
    for log_path in runtime_logs:
        _copy_file_if_exists(log_path, dst_root / "logs" / "runtime_logs" / log_path.name)
    for log_path in msrun_logs:
        _copy_file_if_exists(log_path, dst_root / "logs" / "msrun_log" / log_path.name)

    for data_key, file_path in extra_files:
        target_name = _normalize_repro_data_name(file_path, data_key, iteration)
        _copy_file_if_exists(file_path, dst_root / "data" / target_name)

    _copy_file_if_exists(_iter_aux_file(iter_dir, "failure_info.txt"), dst_root / "logs" / "failure_info.txt")
    _copy_file_if_exists(_iter_aux_file(iter_dir, "FAILED_FLAG"), dst_root / "logs" / "FAILED_FLAG")
    _copy_iteration_snapshots(iter_dir, dst_root)

    manifest = {
        "iteration": iteration,
        "source_iter_dir": _path_text(iter_dir),
        "scripts_dir": _path_text(dst_root / "scripts"),
        "weights_dir": _path_text(weights_dst) if weights_dst.exists() else None,
        "mutation_inputs": [_path_text(dst_root / "mutation_inputs" / path.name) for path in mutation_inputs],
        "logs_dir": _path_text(dst_root / "logs"),
        "data_dir": _path_text(dst_root / "data") if (dst_root / "data").exists() else None,
        "snapshots_dir": _path_text(dst_root / "snapshots") if (dst_root / "snapshots").exists() else None,
    }
    with (dst_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    return dst_root.resolve()


def _build_issue_groups(records: list[IterationAnalysis]) -> list[dict]:
    grouped: dict[tuple[str, str, str], dict] = {}
    for record in records:
        repro_hints = [record.reproduction_dir] if record.reproduction_dir else []
        messages = _select_issue_messages_for_record(record, repro_hints)
        owner = _functional_owner_from_record(record)

        for category, issue_subtype, message, repro_entries in messages:
            normalized_message = (
                _functional_issue_group_key(message)
                if category == "功能问题"
                else re.sub(r"\s+", "", message).lower()
            )
            issue_owner = owner if category == "功能问题" else "-"
            display_message = _normalize_functional_issue_message(message) if category == "功能问题" else message
            stack_keyword = _extract_stack_keyword(display_message) if category == "功能问题" else "-"
            key = (category, issue_owner, issue_subtype, normalized_message)
            entry = grouped.setdefault(
                key,
                {
                    "问题类型": category,
                    "归属": issue_owner,
                    "功能问题类型": issue_subtype,
                    "堆栈关键字": stack_keyword,
                    "报错信息": display_message,
                    "迭代列表": [],
                    "复现入口": [],
                },
            )
            entry["迭代列表"].append(record.iteration_tag)
            entry["复现入口"].extend(repro_entries)

    normalized: list[dict] = []
    for entry in grouped.values():
        entry["迭代列表"] = sorted(set(entry["迭代列表"]), key=lambda item: int(item.replace("iter", "")))
        entry["复现入口"] = sorted(set(entry["复现入口"]))
        normalized.append(entry)
    normalized.sort(
        key=lambda item: (
            item["问题类型"],
            item.get("归属", ""),
            item.get("功能问题类型", ""),
            item["报错信息"],
        )
    )
    return normalized


def _build_functional_issue_stack(function_groups: list[dict]) -> list[dict]:
    stack: list[dict] = []
    for index, item in enumerate(function_groups, start=1):
        stack.append(
            {
                "问题归属": item.get("归属", "待定"),
                "问题类别": item.get("功能问题类型", ""),
                "堆栈关键字": item.get("堆栈关键字", "-"),
                f"报错信息{index}": item.get("报错信息", ""),
                "迭代列表": item.get("迭代列表", []),
            }
        )
    return stack


def _sorted_iter_tags(tags: list[str]) -> list[str]:
    return sorted(set(tags), key=lambda item: int(item.replace("iter", "")))


def _functional_owner_iters(function_group_details: list[dict], owner: str) -> list[str]:
    tags: list[str] = []
    for item in function_group_details:
        if item.get("归属") == owner:
            tags.extend(item.get("迭代列表") or [])
    return _sorted_iter_tags(tags)


def _series_color(value: Optional[float], positive_bad: bool = True) -> str:
    if value is None:
        return "#cbd5e1"
    if positive_bad:
        if value <= 0:
            return "#16a34a"
        if value <= 0.05:
            return "#eab308"
        if value <= 0.20:
            return "#f97316"
        return "#dc2626"
    if value == 0:
        return "#16a34a"
    if value <= 1e-6:
        return "#eab308"
    if value <= 1e-5:
        return "#f97316"
    return "#dc2626"


def _format_percent(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.2f}%"


def _format_float(value: Optional[float], digits: int = 6) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _write_svg_bar_chart(
    path: Path,
    title: str,
    subtitle: str,
    rows: list[tuple[str, Optional[float], str]],
    percent_mode: bool = False,
    negative_as_zero: bool = False,
) -> None:
    width = 1040
    height = 460
    left = 86
    right = 48
    top = 92
    chart_height = 236
    row_count = len(rows)
    min_plot_width = width - left - right

    if row_count <= 8:
        bar_width = 80
        gap = 28
    elif row_count <= 16:
        bar_width = 56
        gap = 20
    elif row_count <= 32:
        bar_width = 36
        gap = 14
    else:
        bar_width = 24
        gap = 18

    plot_width = min_plot_width
    if row_count:
        plot_width = max(min_plot_width, row_count * bar_width + max(row_count - 1, 0) * gap + 56)
        width = left + right + plot_width

    def _chart_value(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        if negative_as_zero:
            return max(value, 0.0)
        return value

    values = [chart_value for _, value, _ in rows if (chart_value := _chart_value(value)) is not None]
    max_value = max(values) if values else 1.0
    if max_value == 0:
        max_value = 1.0
    baseline = top + chart_height
    plot_right = left + plot_width
    start_x = left + 28
    rotate_labels = row_count > 10
    label_font_size = 11 if row_count > 32 else 12 if row_count > 16 else 13
    value_font_size = 11 if row_count > 24 else 12
    show_value_labels = row_count <= 24

    def _format_axis_value(value: float) -> str:
        if percent_mode:
            return f"{value * 100:.0f}%"
        if max_value >= 1:
            return f"{value:.3g}"
        return f"{value:.2g}"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<rect x="{left - 18}" y="{top - 28}" width="{plot_width + 36}" height="{chart_height + 52}" rx="24" fill="#ffffff" stroke="#e2e8f0"/>',
        f'<text x="{left}" y="34" font-size="28" font-family="Arial, sans-serif" fill="#0f172a">{html.escape(title)}</text>',
        f'<text x="{left}" y="60" font-size="15" font-family="Arial, sans-serif" fill="#475569">{html.escape(subtitle)}</text>',
    ]

    for tick_index in range(5):
        ratio = tick_index / 4
        y = baseline - ratio * chart_height
        tick_value = max_value * ratio
        parts.append(
            f'<line x1="{left}" y1="{y}" x2="{plot_right}" y2="{y}" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="4 6"/>'
        )
        parts.append(
            f'<text x="{left - 12}" y="{y + 4}" text-anchor="end" font-size="11" font-family="Arial, sans-serif" fill="#64748b">'
            f"{html.escape(_format_axis_value(tick_value))}</text>"
        )
    parts.append(
        f'<line x1="{left}" y1="{baseline}" x2="{plot_right}" y2="{baseline}" stroke="#94a3b8" stroke-width="2"/>'
    )

    if not rows:
        parts.append(
            f'<text x="{left + plot_width / 2}" y="{top + chart_height / 2}" text-anchor="middle" '
            'font-size="16" font-family="Arial, sans-serif" fill="#64748b">暂无可视化数据</text>'
        )
        parts.append("</svg>")
        path.write_text("\n".join(parts), encoding="utf-8")
        return

    for index, (label, value, color) in enumerate(rows):
        x = start_x + index * (bar_width + gap)
        bar_radius = max(4, min(10, bar_width // 3))
        label_y = baseline + 26
        chart_value = _chart_value(value)
        label_value = chart_value if negative_as_zero else value
        label_text = "N/A" if label_value is None else (f"{label_value * 100:.2f}%" if percent_mode else f"{label_value:.6g}")
        title_text = html.escape(f"{label}: {label_text}")
        parts.append("<g>")
        parts.append(f"<title>{title_text}</title>")
        parts.append(f'<line x1="{x + bar_width / 2}" y1="{baseline}" x2="{x + bar_width / 2}" y2="{baseline + 6}" stroke="#cbd5e1" stroke-width="1"/>')

        if chart_value is None:
            parts.append(f'<rect x="{x}" y="{baseline - 4}" width="{bar_width}" height="4" rx="2" fill="#cbd5e1"/>')
            if show_value_labels:
                parts.append(
                    f'<text x="{x + bar_width / 2}" y="{baseline - 14}" text-anchor="middle" '
                    f'font-size="{value_font_size}" font-family="Arial, sans-serif" fill="#64748b">N/A</text>'
                )
        elif chart_value <= 0:
            if show_value_labels:
                parts.append(
                    f'<text x="{x + bar_width / 2}" y="{baseline - 10}" text-anchor="middle" '
                    f'font-size="{value_font_size}" font-family="Arial, sans-serif" fill="#0f172a">'
                    f"{html.escape(label_text)}</text>"
                )
        else:
            bar_height = max(8, chart_value / max_value * (chart_height - 20))
            y = baseline - bar_height
            parts.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" rx="{bar_radius}" fill="{color}"/>')
            if show_value_labels:
                parts.append(
                    f'<text x="{x + bar_width / 2}" y="{y - 10}" text-anchor="middle" '
                    f'font-size="{value_font_size}" font-family="Arial, sans-serif" fill="#0f172a">'
                    f"{html.escape(label_text)}</text>"
                )

        if rotate_labels:
            parts.append(
                f'<text x="{x + bar_width / 2}" y="{label_y}" text-anchor="end" '
                f'transform="rotate(-40 {x + bar_width / 2} {label_y})" '
                f'font-size="{label_font_size}" font-family="Arial, sans-serif" fill="#475569">'
                f"{html.escape(label)}</text>"
            )
        else:
            parts.append(
                f'<text x="{x + bar_width / 2}" y="{baseline + 28}" text-anchor="middle" '
                f'font-size="{label_font_size}" font-family="Arial, sans-serif" fill="#475569">'
                f"{html.escape(label)}</text>"
            )
        parts.append("</g>")

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _format_signed_percent(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value * 100:+.2f}%"


def _format_signed_float(value: Optional[float], digits: int = 6) -> str:
    if value is None:
        return "-"
    return f"{value:+.{digits}f}"


def _format_bool(value: Optional[bool]) -> str:
    if value is None:
        return "-"
    return "PASS" if value else "FAIL"


def _markdown_table_row(values: Iterable[object]) -> str:
    escaped = [
        str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br/>")
        for value in values
    ]
    return "| " + " | ".join(escaped) + " |"


def _markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = [
        _markdown_table_row(headers),
        _markdown_table_row(["---"] * len(headers)),
    ]
    lines.extend(_markdown_table_row(row) for row in rows)
    return lines


def _load_series_step_rows(record: IterationAnalysis) -> dict[str, object]:
    pta_csv = record.pta_step_csv or record.pta_csv
    msa_csv = record.msa_step_csv or record.msa_csv
    if not pta_csv or not msa_csv:
        return {
            "pta_total_steps": 0,
            "msa_total_steps": 0,
            "common_steps": [],
            "pta_only_steps": [],
            "msa_only_steps": [],
            "rows": [],
        }

    pta_rows = _read_training_csv(Path(pta_csv))
    msa_rows = _read_training_csv(Path(msa_csv))
    pta_index = _index_rows(pta_rows)
    msa_index = _index_rows(msa_rows)

    common_steps = sorted(set(pta_index) & set(msa_index))
    rows: list[list[object]] = []
    for step in common_steps:
        pta = pta_index[step]
        msa = msa_index[step]
        rows.append(
            [
                step,
                _format_float(pta["time"]),
                _format_float(msa["time"]),
                _format_signed_float(msa["time"] - pta["time"]),
                _format_signed_percent(_relative_delta(pta["time"], msa["time"])),
                _format_float(pta["loss"], digits=8),
                _format_float(msa["loss"], digits=8),
                _format_float(abs(msa["loss"] - pta["loss"]), digits=8),
                _format_float(pta["memory"], digits=2),
                _format_float(msa["memory"], digits=2),
                _format_signed_float(msa["memory"] - pta["memory"], digits=2),
                _format_signed_percent(_relative_delta(pta["memory"], msa["memory"])),
            ]
        )

    return {
        "pta_total_steps": len(pta_rows),
        "msa_total_steps": len(msa_rows),
        "common_steps": common_steps,
        "pta_only_steps": sorted(set(pta_index) - set(msa_index)),
        "msa_only_steps": sorted(set(msa_index) - set(pta_index)),
        "rows": rows,
    }


def _load_single_iteration_metrics(record: IterationAnalysis) -> dict[str, Optional[dict[str, Optional[float]]]]:
    pta_metrics = _load_metrics_from_paths(
        record.task_type,
        record.iteration,
        Path(record.pta_csv) if record.pta_csv else None,
        Path(record.pta_step_csv) if record.pta_step_csv else None,
    )
    msa_metrics = _load_metrics_from_paths(
        record.task_type,
        record.iteration,
        Path(record.msa_csv) if record.msa_csv else None,
        Path(record.msa_step_csv) if record.msa_step_csv else None,
    )
    mf_metrics = _load_metrics_from_paths(
        record.task_type,
        record.iteration,
        Path(record.mf_csv) if record.mf_csv else None,
        Path(record.mf_step_csv) if record.mf_step_csv else None,
    )
    return {
        "pta": pta_metrics,
        "msa": msa_metrics,
        "mf": mf_metrics,
    }


def _active_compare_metrics(
    single_metrics: dict[str, Optional[dict[str, Optional[float]]]],
    compare_label: str,
) -> dict[str, Optional[float]]:
    return (single_metrics.get("mf" if compare_label == "MF" else "msa") or {})


def _load_series_side_snapshot(csv_path: str | None) -> dict[str, Optional[float]]:
    if not csv_path:
        return {
            "avg_step_time_skip1": None,
            "last_loss": None,
            "max_memory_mb": None,
        }

    try:
        rows = _read_training_csv(Path(csv_path))
    except FileNotFoundError:
        rows = []
    if not rows:
        return {
            "avg_step_time_skip1": None,
            "last_loss": None,
            "max_memory_mb": None,
        }

    step_times_skip1 = [row["time"] for row in rows if int(row["iteration"]) > 1]
    return {
        "avg_step_time_skip1": _safe_mean(step_times_skip1),
        "last_loss": rows[-1]["loss"],
        "max_memory_mb": max((row["memory"] for row in rows), default=None),
    }


def _series_metric_snapshot(
    record: IterationAnalysis,
    single_metrics: dict[str, Optional[dict[str, Optional[float]]]],
    compare_label: str,
) -> dict[str, Optional[float]]:
    pta_metrics = single_metrics.get("pta") or {}
    compare_metrics = _active_compare_metrics(single_metrics, compare_label)
    compare_csv = record.mf_step_csv or record.mf_csv if compare_label == "MF" else record.msa_step_csv or record.msa_csv

    pta_series = _load_series_side_snapshot(record.pta_step_csv or record.pta_csv)
    compare_series = _load_series_side_snapshot(compare_csv)

    return {
        "pta_perf": (
            record.pta_avg_step_time_skip1
            if record.pta_avg_step_time_skip1 is not None
            else pta_series["avg_step_time_skip1"]
            if pta_series["avg_step_time_skip1"] is not None
            else pta_metrics.get("time")
        ),
        "compare_perf": (
            record.msa_avg_step_time_skip1
            if record.msa_avg_step_time_skip1 is not None
            else compare_series["avg_step_time_skip1"]
            if compare_series["avg_step_time_skip1"] is not None
            else compare_metrics.get("time")
        ),
        "pta_loss": (
            pta_metrics.get("loss")
            if pta_metrics.get("loss") is not None
            else pta_series["last_loss"]
        ),
        "compare_loss": (
            compare_metrics.get("loss")
            if compare_metrics.get("loss") is not None
            else compare_series["last_loss"]
        ),
        "pta_mem": (
            record.pta_max_memory_mb
            if record.pta_max_memory_mb is not None
            else pta_series["max_memory_mb"]
            if pta_series["max_memory_mb"] is not None
            else pta_metrics.get("memory")
        ),
        "compare_mem": (
            record.msa_max_memory_mb
            if record.msa_max_memory_mb is not None
            else compare_series["max_memory_mb"]
            if compare_series["max_memory_mb"] is not None
            else compare_metrics.get("memory")
        ),
    }


def _build_metric_overview_rows(
    record: IterationAnalysis,
    profile: dict,
    single_metrics: dict[str, Optional[dict[str, Optional[float]]]],
) -> list[list[object]]:
    pta_metrics = single_metrics.get("pta") or {}
    compare_label = _compare_label_for_mode(record.task_type, record.compare_mode)
    compare_metrics = _active_compare_metrics(single_metrics, compare_label)

    if profile["comparison_mode"] == "series":
        series_snapshot = _series_metric_snapshot(record, single_metrics, compare_label)
        return [
            [
                "平均 step 耗时(去首步, s)",
                _format_float(series_snapshot["pta_perf"]),
                _format_float(series_snapshot["compare_perf"]),
                _format_signed_float(record.performance_delta_seconds),
                _format_signed_percent(record.performance_delta_ratio),
                record.performance_severity,
            ],
            [
                "精度对比(末步 loss / 公共 step 最大 loss 绝对差)",
                _format_float(series_snapshot["pta_loss"], digits=8),
                _format_float(series_snapshot["compare_loss"], digits=8),
                _format_float(record.max_loss_diff, digits=8),
                "-",
                record.precision_severity,
            ],
            [
                "最大显存(MB)",
                _format_float(series_snapshot["pta_mem"], digits=2),
                _format_float(series_snapshot["compare_mem"], digits=2),
                _format_signed_float(record.memory_delta_mb, digits=2),
                _format_signed_percent(record.memory_delta_ratio),
                record.memory_severity,
            ],
        ]

    time_label = "当前迭代耗时(s)"
    use_series_metrics = record.performance_source.startswith("series")
    if use_series_metrics:
        time_label = "平均 step 耗时(去首步, s)"
    loss_label = "精度对比"
    memory_label = "当前迭代显存(MB)"
    if use_series_metrics:
        loss_label = "精度对比(当前 loss / 公共 step 最大 loss 绝对差)"
        memory_label = "逐 step 最大显存(MB)"
    return [
        [
            time_label,
            _format_float(
                record.pta_avg_step_time_skip1
                if use_series_metrics
                else pta_metrics.get("time")
            ),
            _format_float(
                record.msa_avg_step_time_skip1
                if use_series_metrics
                else compare_metrics.get("time")
            ),
            _format_signed_float(record.performance_delta_seconds),
            _format_signed_percent(record.performance_delta_ratio),
            record.performance_severity,
        ],
        [
            loss_label,
            _format_float(pta_metrics.get("loss"), digits=8),
            _format_float(compare_metrics.get("loss"), digits=8),
            _format_float(record.max_loss_diff, digits=8),
            "-",
            record.precision_severity,
        ],
        [
            memory_label,
            _format_float(record.pta_max_memory_mb, digits=2) if use_series_metrics else _format_float(pta_metrics.get("memory"), digits=2),
            _format_float(record.msa_max_memory_mb, digits=2) if use_series_metrics else _format_float(compare_metrics.get("memory"), digits=2),
            _format_signed_float(record.memory_delta_mb, digits=2),
            _format_signed_percent(record.memory_delta_ratio),
            record.memory_severity,
        ],
    ]


def _summary_metric_snapshot(
    record: IterationAnalysis,
    profile: dict,
) -> dict[str, Optional[float]]:
    single_metrics = _load_single_iteration_metrics(record)
    pta_metrics = single_metrics.get("pta") or {}
    compare_label = _compare_label_for_mode(record.task_type, record.compare_mode)
    compare_metrics = _active_compare_metrics(single_metrics, compare_label)
    if profile["comparison_mode"] == "series" or record.performance_source.startswith("series"):
        series_snapshot = _series_metric_snapshot(record, single_metrics, compare_label)
        pta_perf = series_snapshot["pta_perf"]
        msa_perf = series_snapshot["compare_perf"]
    else:
        pta_perf = pta_metrics.get("time")
        msa_perf = compare_metrics.get("time")

    if profile["comparison_mode"] == "series" or record.performance_source.startswith("series"):
        pta_mem = series_snapshot["pta_mem"]
        msa_mem = series_snapshot["compare_mem"]
    else:
        pta_mem = pta_metrics.get("memory")
        msa_mem = compare_metrics.get("memory")

    return {
        "pta_perf": pta_perf,
        "msa_perf": msa_perf,
        "pta_mem": pta_mem,
        "msa_mem": msa_mem,
    }


def _partial_comparison_summary_lines(
    record: IterationAnalysis,
    profile: dict,
    compare_label: str,
    single_metrics: dict[str, Optional[dict[str, Optional[float]]]],
) -> list[str]:
    metric_snapshot = _summary_metric_snapshot(record, profile)
    pta_metrics = single_metrics.get("pta") or {}
    compare_metrics = _active_compare_metrics(single_metrics, compare_label)
    lines = [
        f"- 当前轮 PTA/{compare_label} 对比数据不完整，本报告优先展示 PTA 已产出的指标。",
    ]

    if metric_snapshot["pta_perf"] is not None:
        perf_label = "当前迭代耗时"
        if profile["comparison_mode"] == "series" or record.performance_source.startswith("series"):
            perf_label = "平均 step 耗时(去首步)"
        lines.append(
            f"- PTA {perf_label}: `{_format_float(metric_snapshot['pta_perf'])}` s。"
        )

    pta_loss = pta_metrics.get("loss")
    if pta_loss is not None:
        loss_label = "当前迭代 loss"
        if profile["comparison_mode"] == "series" or record.performance_source.startswith("series"):
            loss_label = "末步 loss"
        lines.append(
            f"- PTA {loss_label}: `{_format_float(pta_loss, digits=8)}`。"
        )

    if metric_snapshot["pta_mem"] is not None:
        memory_label = "当前迭代显存"
        if profile["comparison_mode"] == "series" or record.performance_source.startswith("series"):
            memory_label = "最大显存"
        lines.append(
            f"- PTA {memory_label}: `{_format_float(metric_snapshot['pta_mem'], digits=2)}` MB。"
        )

    available_compare_metrics = [
        name
        for name, value in (
            ("耗时", compare_metrics.get("time")),
            ("loss", compare_metrics.get("loss")),
            ("显存", compare_metrics.get("memory")),
        )
        if value is not None
    ]
    if available_compare_metrics:
        lines.append(
            f"- {compare_label} 已产出 `{', '.join(available_compare_metrics)}` 数据，但仍缺少完整对比结果。"
        )
    else:
        lines.append(
            f"- {compare_label} 未生成可用指标，暂无法计算差值与等级明细。"
        )
    return lines


def _render_iteration_report(
    record: IterationAnalysis,
    profile: dict,
    analysis_report_path: Optional[Path],
    iteration_report_dir: Path,
) -> str:
    single_metrics = _load_single_iteration_metrics(record)
    use_series_metrics = record.performance_source.startswith("series")
    normalized_compare_mode = (record.compare_mode or "").strip().lower()
    compare_label = _compare_label_for_mode(record.task_type, normalized_compare_mode)
    compare_metrics = _active_compare_metrics(single_metrics, compare_label)
    compare_target_success = record.mf_execution_success if compare_label == "MF" else record.msa_execution_success
    compare_pair_label = f"PTA/{compare_label}"
    compare_delta_label = f"{compare_label}-PTA"
    lines = [
        f"# {record.iteration_tag} 分析报告",
        "",
        "## 概览",
        "",
        f"- 任务: `{profile['task_name']}`",
        f"- 状态: `{record.overall_status}`",
        f"- 问题分类: `{', '.join(record.categories) if record.categories else '无'}`",
        f"- 变异成功: `{record.mutation_success}`",
        f"- PTA执行成功: `{record.pta_execution_success}`",
        f"- {compare_label}执行成功: `{compare_target_success}`",
        f"- {compare_pair_label} 对比可用: `{record.comparison_available}`",
        f"- 原始迭代目录: `{record.iteration_dir}`",
        f"- 统一报告目录: `{_path_text(iteration_report_dir)}`",
    ]

    if record.reproduction_dir:
        lines.append(f"- 异常轮复现目录: `{record.reproduction_dir}`")
    if analysis_report_path is not None:
        lines.append(f"- 原始 analyse 文本: `{_path_text(analysis_report_path)}`")

    lines.extend(
        [
            "",
            "## 简要结论",
            "",
        ]
    )

    if record.comparison_available:
        if profile["comparison_mode"] == "series":
            lines.extend(
                [
                    (
                        "- 忽略第一个 step 后，PTA 平均 step 耗时 "
                        f"`{_format_float(record.pta_avg_step_time_skip1)}` s，{compare_label} 为 "
                        f"`{_format_float(record.msa_avg_step_time_skip1)}` s，差值 "
                        f"`{_format_signed_float(record.performance_delta_seconds)}` s "
                        f"(`{_format_signed_percent(record.performance_delta_ratio)}`)，"
                        f"等级 `{record.performance_severity}`。"
                    ),
                    (
                        "- 对齐公共 step 后，最大 loss 绝对差为 "
                        f"`{_format_float(record.max_loss_diff, digits=8)}`，"
                        f"等级 `{record.precision_severity}`。"
                    ),
                    (
                        f"- 最大显存 PTA=`{_format_float(record.pta_max_memory_mb, digits=2)}` MB，"
                        f"{compare_label}=`{_format_float(record.msa_max_memory_mb, digits=2)}` MB，差值 "
                        f"`{_format_signed_float(record.memory_delta_mb, digits=2)}` MB "
                        f"(`{_format_signed_percent(record.memory_delta_ratio)}`)，"
                        f"等级 `{record.memory_severity}`。"
                    ),
                ]
            )
        else:
            performance_line = (
                "- 当前迭代耗时 PTA=`{pta}` s，{cmp}=`{msa}` s，差值 `{delta}` s "
                "(`{ratio}`)，等级 `{level}`。"
            )
            if use_series_metrics:
                performance_line = (
                    "- 忽略第一个 step 后，PTA 平均 step 耗时 `{pta}` s，{cmp}=`{msa}` s，"
                    "差值 `{delta}` s (`{ratio}`)，等级 `{level}`。"
                )
            lines.extend(
                [
                    performance_line.format(
                        cmp=compare_label,
                        pta=_format_float(
                            record.pta_avg_step_time_skip1
                            if use_series_metrics
                            else (single_metrics.get("pta") or {}).get("time")
                        ),
                        msa=_format_float(
                            record.msa_avg_step_time_skip1
                            if use_series_metrics
                            else compare_metrics.get("time")
                        ),
                        delta=_format_signed_float(record.performance_delta_seconds),
                        ratio=_format_signed_percent(record.performance_delta_ratio),
                        level=record.performance_severity,
                    ),
                ]
            )
            if use_series_metrics:
                lines.extend(
                    [
                        (
                            "- 对齐公共 step 后，最大 loss 绝对差为 `{delta}`，等级 `{level}`。"
                        ).format(
                            delta=_format_float(record.max_loss_diff, digits=8),
                            level=record.precision_severity,
                        ),
                        (
                            "- 逐 step 最大显存 PTA=`{pta}` MB，{cmp}=`{msa}` MB，差值 `{delta}` MB "
                            "(`{ratio}`)，等级 `{level}`。"
                        ).format(
                            cmp=compare_label,
                            pta=_format_float(record.pta_max_memory_mb, digits=2),
                            msa=_format_float(record.msa_max_memory_mb, digits=2),
                            delta=_format_signed_float(record.memory_delta_mb, digits=2),
                            ratio=_format_signed_percent(record.memory_delta_ratio),
                            level=record.memory_severity,
                        ),
                    ]
                )
            else:
                lines.extend(
                    [
                        (
                            "- 当前迭代 loss PTA=`{pta}`，{cmp}=`{msa}`，绝对差 `{delta}`，"
                            "等级 `{level}`。"
                        ).format(
                            cmp=compare_label,
                            pta=_format_float((single_metrics.get("pta") or {}).get("loss"), digits=8),
                            msa=_format_float(compare_metrics.get("loss"), digits=8),
                            delta=_format_float(record.max_loss_diff, digits=8),
                            level=record.precision_severity,
                        ),
                        (
                            "- 当前迭代显存 PTA=`{pta}` MB，{cmp}=`{msa}` MB，差值 `{delta}` MB "
                            "(`{ratio}`)，等级 `{level}`。"
                        ).format(
                            cmp=compare_label,
                            pta=_format_float((single_metrics.get("pta") or {}).get("memory"), digits=2),
                            msa=_format_float(compare_metrics.get("memory"), digits=2),
                            delta=_format_signed_float(record.memory_delta_mb, digits=2),
                            ratio=_format_signed_percent(record.memory_delta_ratio),
                            level=record.memory_severity,
                        ),
                    ]
                )
    else:
        lines.extend(
            _partial_comparison_summary_lines(
                record,
                profile,
                compare_label,
                single_metrics,
            )
        )

    if record.functional_reasons:
        lines.extend(
            f"- 失败原因: `{_functional_reason_text(reason)}`" for reason in record.functional_reasons
        )
    elif record.issue_signals:
        lines.append("- 未归纳出显式失败原因，但日志中存在异常信号，见下文。")
    else:
        lines.append("- 未发现额外失败信号。")

    lines.extend(
        [
            "",
            "## 指标总表",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            ["指标", "PTA", compare_label, compare_delta_label, "相对差值", "等级"],
            _build_metric_overview_rows(record, profile, single_metrics),
        )
    )

    if profile["supports_mf"] and compare_label != "MF":
        mf_metrics = single_metrics.get("mf") or {}
        lines.extend(
            [
                "",
                "## MF 补充",
                "",
                f"- MF 有效: `{_format_bool(record.mf_valid)}`",
                f"- MF CSV: `{record.mf_csv or '-'}`",
            ]
        )
        if mf_metrics:
            lines.extend(
                _markdown_table(
                    ["指标", "MF"],
                    [
                        ["当前迭代耗时(s)", _format_float(mf_metrics.get("time"))],
                        ["当前迭代 loss", _format_float(mf_metrics.get("loss"), digits=8)],
                        ["当前迭代显存(MB)", _format_float(mf_metrics.get("memory"), digits=2)],
                    ],
                )
            )

    if (profile["comparison_mode"] == "series" or record.performance_source.startswith("series")) and record.comparison_available:
        step_payload = _load_series_step_rows(record)
        lines.extend(
            [
                "",
                f"## PTA / {compare_label} 每步对比",
                "",
                f"- PTA step 数: `{step_payload['pta_total_steps']}`",
                f"- {compare_label} step 数: `{step_payload['msa_total_steps']}`",
                f"- 公共 step 数: `{len(step_payload['common_steps'])}`",
            ]
        )
        if step_payload["pta_only_steps"]:
            lines.append(f"- 仅 PTA 存在的 step: `{step_payload['pta_only_steps']}`")
        if step_payload["msa_only_steps"]:
            lines.append(f"- 仅 {compare_label} 存在的 step: `{step_payload['msa_only_steps']}`")
        lines.append("")
        if step_payload["rows"]:
            lines.extend(
                _markdown_table(
                    [
                        "step",
                        "PTA耗时(s)",
                        f"{compare_label}耗时(s)",
                        "耗时差(s)",
                        "耗时差值",
                        "PTA loss",
                        f"{compare_label} loss",
                        "loss绝对差",
                        "PTA显存(MB)",
                        f"{compare_label}显存(MB)",
                        "显存差(MB)",
                        "显存差值",
                    ],
                    step_payload["rows"],
                )
            )
        else:
            lines.append("当前轮没有可对齐的公共 step。")

    if record.status_components:
        component_order = ["MUTATE", "PTA_SAVE", "PTA_LOAD", "MSA_LOAD", "MF", "ANALYSE"]
        ordered_keys = [key for key in component_order if key in record.status_components]
        ordered_keys.extend(
            key for key in sorted(record.status_components) if key not in component_order
        )
        lines.extend(
            [
                "",
                "## 组件状态",
                "",
            ]
        )
        lines.extend(
            _markdown_table(
                ["组件", "状态"],
                [[key, record.status_components[key]] for key in ordered_keys],
            )
        )

    if record.step_results:
        lines.extend(
            [
                "",
                "## 步骤执行结果",
                "",
            ]
        )
        lines.extend(
            _markdown_table(
                ["步骤", "结果"],
                [[key, value] for key, value in record.step_results.items()],
            )
        )

    lines.extend(
        [
            "",
            "## 日志信号",
            "",
        ]
    )
    if record.issue_signals:
        for signal in record.issue_signals:
            lines.append(
                f"- `{signal.category}`: `{signal.message}` "
                f"(`{signal.log_path}:{signal.line_number}`)"
            )
    else:
        lines.append("- 未扫描到额外错误信号。")

    lines.extend(
        [
            "",
            "## 数据入口",
            "",
            f"- PTA CSV: `{record.pta_csv or '-'}`",
            f"- {compare_label} CSV: `{record.msa_csv or '-'}`",
            f"- PTA step CSV: `{record.pta_step_csv or '-'}`",
            f"- {compare_label} step CSV: `{record.msa_step_csv or '-'}`",
        ]
    )
    if compare_label != "MF":
        lines.extend(
            [
                f"- MF CSV: `{record.mf_csv or '-'}`",
                f"- MF step CSV: `{record.mf_step_csv or '-'}`",
            ]
        )
    lines.append(f"- status.json: `{record.status_file or '-'}`")
    if analysis_report_path is not None:
        lines.append(f"- 原始 analyse 文本: `{_path_text(analysis_report_path)}`")

    if record.mutation_input_paths:
        lines.append("- mutation 输入:")
        lines.extend(f"  - `{path}`" for path in record.mutation_input_paths)
    if record.runtime_log_paths:
        lines.append("- runtime 日志:")
        lines.extend(f"  - `{path}`" for path in record.runtime_log_paths)
    if record.msrun_log_paths:
        lines.append("- msrun 日志:")
        lines.extend(f"  - `{path}`" for path in record.msrun_log_paths)

    return "\n".join(lines) + "\n"


def _write_iteration_csv(path: Path, records: list[IterationAnalysis]) -> None:
    fieldnames = [
        "task_type",
        "iteration",
        "iteration_tag",
        "overall_status",
        "failed_flag",
        "mutation_success",
        "pta_execution_success",
        "msa_execution_success",
        "mf_execution_success",
        "comparison_available",
        "categories",
        "pta_csv",
        "msa_csv",
        "mf_csv",
        "mf_valid",
        "pta_loss",
        "msa_loss",
        "mf_loss",
        "pta_avg_step_time_skip1",
        "msa_avg_step_time_skip1",
        "performance_delta_seconds",
        "performance_delta_ratio",
        "performance_severity",
        "max_loss_diff",
        "avg_loss_diff",
        "precision_severity",
        "pta_max_memory_mb",
        "msa_max_memory_mb",
        "memory_delta_mb",
        "memory_delta_ratio",
        "memory_severity",
        "status_file",
        "status_components",
        "step_results",
        "functional_reasons",
        "reproduction_dir",
        "iteration_report_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "task_type": record.task_type,
                    "iteration": record.iteration,
                    "iteration_tag": record.iteration_tag,
                    "overall_status": record.overall_status,
                    "failed_flag": record.failed_flag,
                    "mutation_success": record.mutation_success,
                    "pta_execution_success": record.pta_execution_success,
                    "msa_execution_success": record.msa_execution_success,
                    "mf_execution_success": record.mf_execution_success,
                    "comparison_available": record.comparison_available,
                    "categories": "|".join(record.categories),
                    "pta_csv": record.pta_csv or "",
                    "msa_csv": record.msa_csv or "",
                    "mf_csv": record.mf_csv or "",
                    "mf_valid": record.mf_valid,
                    "pta_loss": record.pta_loss,
                    "msa_loss": record.msa_loss,
                    "mf_loss": record.mf_loss,
                    "pta_avg_step_time_skip1": record.pta_avg_step_time_skip1,
                    "msa_avg_step_time_skip1": record.msa_avg_step_time_skip1,
                    "performance_delta_seconds": record.performance_delta_seconds,
                    "performance_delta_ratio": record.performance_delta_ratio,
                    "performance_severity": record.performance_severity,
                    "max_loss_diff": record.max_loss_diff,
                    "avg_loss_diff": record.avg_loss_diff,
                    "precision_severity": record.precision_severity,
                    "pta_max_memory_mb": record.pta_max_memory_mb,
                    "msa_max_memory_mb": record.msa_max_memory_mb,
                    "memory_delta_mb": record.memory_delta_mb,
                    "memory_delta_ratio": record.memory_delta_ratio,
                    "memory_severity": record.memory_severity,
                    "status_file": record.status_file or "",
                    "status_components": json.dumps(record.status_components, ensure_ascii=False),
                    "step_results": json.dumps(record.step_results, ensure_ascii=False),
                    "functional_reasons": " | ".join(
                        _functional_reason_text(reason) for reason in record.functional_reasons
                    ),
                    "reproduction_dir": record.reproduction_dir or "",
                    "iteration_report_path": record.iteration_report_path or "",
                }
            )


def _metric_stats(values: list[float]) -> dict:
    if not values:
        return {"平均值": None, "中位数": None, "最大值": None, "最小值": None}
    return {
        "平均值": sum(values) / len(values),
        "中位数": statistics.median(values),
        "最大值": max(values),
        "最小值": min(values),
    }


def _compare_label_for_mode(task_type: int, compare_mode: Optional[str]) -> str:
    normalized = (compare_mode or "").strip().lower()
    if task_type in {1, 2, 3} and normalized == "pta_mf":
        return "MF"
    return "MSA"


def _summary_payload(
    task_type: int,
    output_root: Path,
    run_dir: Path,
    model_name: str,
    planned_iterations: Optional[int],
    records: list[IterationAnalysis],
    issue_groups: list[dict],
) -> dict:
    profile = _task_profile(task_type)
    task_config = _load_task_config(output_root, task_type)
    compare_mode = str(task_config.get("COMPARE_MODE") or "").strip().lower() or None
    compare_label = _compare_label_for_mode(task_type, compare_mode)
    target_component = "MF" if compare_label == "MF" else "MSA_LOAD"
    executed = len(records)
    mutation_success = sum(1 for record in records if record.mutation_success)
    pta_success = sum(1 for record in records if record.pta_execution_success)
    compare_enabled_records = [
        record for record in records
        if (
            not _component_is_skipped(record.status_components.get(target_component))
            or record.msa_execution_success
            or record.msa_csv
        )
    ]
    compare_success = sum(1 for record in compare_enabled_records if record.msa_execution_success)
    valid_comparisons = sum(1 for record in records if record.comparison_available)
    precision_iters = [record.iteration_tag for record in records if "精度问题" in record.categories]
    performance_iters = [record.iteration_tag for record in records if "性能问题" in record.categories]
    memory_iters = [record.iteration_tag for record in records if "显存问题" in record.categories]
    function_groups = [group for group in issue_groups if group["问题类型"] == "功能问题"]
    functional_issue_stack = _build_functional_issue_stack(function_groups)
    mf_iters = [record.iteration_tag for record in records if record.mf_valid is False]

    valid_records = [record for record in records if record.comparison_available]
    performance_ratios = [
        record.performance_delta_ratio
        for record in valid_records
        if record.performance_delta_ratio is not None
    ]
    loss_diffs = [record.max_loss_diff for record in valid_records if record.max_loss_diff is not None]
    memory_ratios = [
        record.memory_delta_ratio
        for record in valid_records
        if record.memory_delta_ratio is not None
    ]

    payload = {
        "task_type": task_type,
        "task_name": profile["task_name"],
        "compare_mode": compare_mode,
        "对比对象": compare_label,
        "model_name": model_name,
        "nutnm": task_config.get("MUTNM"),
        "iters": planned_iterations if planned_iterations is not None else executed,
        "日志目录": _path_text(run_dir),
        "执行总数": executed,
        "变异成功数": mutation_success,
        "变异成功率": f"{(mutation_success / executed * 100):.2f}%" if executed else "0.00%",
        "PTA执行成功数": pta_success,
        "PTA执行成功率": f"{(pta_success / executed * 100):.2f}%" if executed else "0.00%",
        "MS执行成功数": compare_success if compare_enabled_records else None,
        "MS执行成功率": (
            f"{(compare_success / len(compare_enabled_records) * 100):.2f}%"
            if compare_enabled_records
            else None
        ),
        "有效对比轮次": valid_comparisons,
        "复现目录": _path_text(output_root),
        "迭代报告目录": _path_text(output_root),
        "output目录": _path_text(output_root),
        "issue_groups": issue_groups,
        "issue-groups": issue_groups,
        "功能问题": functional_issue_stack,
        "功能问题详情": function_groups,
        "精度问题": {
            "数量": len(precision_iters),
            "迭代列表": precision_iters,
        },
        "性能问题": {
            "数量": len(performance_iters),
            "迭代列表": performance_iters,
        },
        "显存问题": {
            "数量": len(memory_iters),
            "迭代列表": memory_iters,
        },
        "验证标准": {
            "性能": profile["performance_rule"],
            "精度": profile["precision_rule"],
            "显存": profile["memory_rule"],
        },
        "定量统计": {
            "性能相对差值": _metric_stats(performance_ratios),
            "最大loss绝对差": _metric_stats(loss_diffs),
            "显存相对差值": _metric_stats(memory_ratios),
        },
        "迭代详情": [
            {
                "迭代": record.iteration_tag,
                "状态": record.overall_status,
                "问题分类": record.categories,
                "步骤执行结果": record.step_results,
                "PTA平均step耗时(去首步)": record.pta_avg_step_time_skip1,
                f"{compare_label}平均step耗时(去首步)": record.msa_avg_step_time_skip1,
                "性能数据来源": record.performance_source,
                "性能相对差值": record.performance_delta_ratio,
                "性能等级": record.performance_severity,
                "PTA loss": record.pta_loss,
                f"{compare_label} loss": record.msa_loss,
                "最大loss绝对差": record.max_loss_diff,
                "精度等级": record.precision_severity,
                "PTA最大显存(MB)": record.pta_max_memory_mb,
                f"{compare_label}最大显存(MB)": record.msa_max_memory_mb,
                "显存相对差值": record.memory_delta_ratio,
                "显存等级": record.memory_severity,
                "MF loss": record.mf_loss,
                "MF有效": record.mf_valid,
                "复现目录": record.reproduction_dir,
                "迭代报告": record.iteration_report_path,
            }
            for record in records
        ],
    }

    if task_type == 2:
        payload["submodules"] = task_config.get("SUBMODULES")

    if profile["supports_mf"]:
        payload["MF问题"] = {
            "数量": len(mf_iters),
            "迭代列表": mf_iters,
        }

    return payload


def _strict_summary_payload(
    model_name: str,
    planned_iterations: Optional[int],
    run_dir: Path,
    records: list[IterationAnalysis],
    issue_groups: list[dict],
) -> dict:
    precision_iters = [record.iteration_tag for record in records if "精度问题" in record.categories]
    performance_iters = [record.iteration_tag for record in records if "性能问题" in record.categories]
    memory_iters = [record.iteration_tag for record in records if "显存问题" in record.categories]
    function_groups = [group for group in issue_groups if group["问题类型"] == "功能问题"]

    return {
        "model_name": model_name,
        "iters": planned_iterations if planned_iterations is not None else len(records),
        "日志目录": _path_text(run_dir),
        "精度问题": {
            "迭代列表": precision_iters,
        },
        "性能问题": {
            "迭代列表": performance_iters,
        },
        "显存问题": {
            "迭代列表": memory_iters,
        },
        "功能问题": _build_functional_issue_stack(function_groups),
    }


def _write_markdown(path: Path, payload: dict, records: list[IterationAnalysis], assets_dir: Path) -> None:
    profile = _task_profile(int(payload["task_type"]))
    compare_label = str(payload.get("对比对象") or "MSA")
    function_group_details = payload.get("功能问题详情") or []
    pta_function_iters = _functional_owner_iters(function_group_details, "PTA问题")
    ms_function_iters = _functional_owner_iters(function_group_details, "MS问题")
    lines = [
        f"# {profile['task_title']}",
        "",
        "## 总览",
        "",
        f"- 模型: `{payload['model_name']}`",
        f"- MUTNM: `{payload.get('nutnm', '-')}`",
        f"- 计划轮次: `{payload['iters']}`",
        f"- 实际执行轮次: `{payload['执行总数']}`",
        f"- 变异成功数: `{payload['变异成功数']}`",
        f"- 变异成功率: `{payload['变异成功率']}`",
        f"- PTA执行成功数: `{payload['PTA执行成功数']}`",
        f"- PTA执行成功率: `{payload['PTA执行成功率']}`",
        f"- {compare_label}执行成功数: `{payload.get('MS执行成功数', '-') if payload.get('MS执行成功数', '-') is not None else '-'}`",
        f"- {compare_label}执行成功率: `{payload.get('MS执行成功率') or '-'}`",
        f"- 有效对比轮次: `{payload['有效对比轮次']}`",
        f"- 迭代目录根: `{payload['迭代报告目录']}`",
        f"- 复现目录根: `{payload['复现目录']}`",
        "",
        "## 问题归类",
        "",
        f"- 功能问题-PTA: `{len(pta_function_iters)}` 轮 -> `{pta_function_iters}`",
        f"- 功能问题-MS: `{len(ms_function_iters)}` 轮 -> `{ms_function_iters}`",
        f"- 精度问题: `{payload['精度问题']['数量']}` 轮 -> `{payload['精度问题']['迭代列表']}`",
        f"- 性能问题: `{payload['性能问题']['数量']}` 轮 -> `{payload['性能问题']['迭代列表']}`",
        f"- 显存问题: `{payload['显存问题']['数量']}` 轮 -> `{payload['显存问题']['迭代列表']}`",
    ]

    if int(payload["task_type"]) == 2:
        lines.insert(10, f"- SUBMODULES: `{payload.get('submodules', '-')}`")

    if "MF问题" in payload:
        lines.append(f"- MF问题: `{payload['MF问题']['数量']}` 轮 -> `{payload['MF问题']['迭代列表']}`")

    lines.extend(
        [
            "",
            "## 验证标准",
            "",
            f"- 性能: `{payload['验证标准']['性能']}`",
            f"- 精度: `{payload['验证标准']['精度']}`",
            f"- 显存: `{payload['验证标准']['显存']}`",
            f"- 性能/显存负值表示 {compare_label} 优于 PTA，按正常结果处理；SVG 图表中按 `0` 展示。",
            "",
            "## 定量明细",
            "",
            f"| 迭代 | 状态 | 总体结果 | PTA性能(s) | {compare_label}性能(s) | 性能差值 | 性能情况 | PTA loss | {compare_label} loss | max loss diff | loss情况 | PTA显存(MB) | {compare_label}显存(MB) | 显存差值 | 显存情况 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for record in records:
        metric_snapshot = _summary_metric_snapshot(record, profile)
        lines.append(
            "| {tag} | {status} | {overall} | {pta_perf} | {msa_perf} | {perf} | {perf_status} | {pta_loss} | {msa_loss} | {loss} | {loss_status} | {pta_mem} | {msa_mem} | {mem} | {mem_status} |".format(
                tag=record.iteration_tag,
                status=_markdown_status(record.overall_status),
                overall=_overall_result_text(record),
                pta_perf=_format_float(metric_snapshot["pta_perf"]),
                msa_perf=_format_float(metric_snapshot["msa_perf"]),
                pta_loss=_format_float(record.pta_loss, digits=8),
                msa_loss=_format_float(record.msa_loss, digits=8),
                perf=_format_percent(record.performance_delta_ratio),
                perf_status=record.performance_severity,
                loss=_format_float(record.max_loss_diff, digits=8),
                loss_status=record.precision_severity,
                pta_mem=_format_float(metric_snapshot["pta_mem"], digits=2),
                msa_mem=_format_float(metric_snapshot["msa_mem"], digits=2),
                mem=_format_percent(record.memory_delta_ratio),
                mem_status=record.memory_severity,
            )
        )

    lines.extend(
        [
            "",
            "## 步骤执行结果",
            "",
            "| 迭代 | 步骤结果 |",
            "| --- | --- |",
        ]
    )

    for record in records:
        step_text = " / ".join(f"{key}:{value}" for key, value in record.step_results.items())
        lines.append(
            f"| {record.iteration_tag} | {step_text or '-'} |"
        )

    lines.extend(
        [
            "",
            "## 如何排查异常轮",
            "",
            "- 每轮报告直接写在对应的 `iters/iter_x/report.md`。",
            "- 复现时直接进入 `iters/iter_x/` 查看 `status.json`、`scripts/`、`weights/`、`mutation_inputs/` 和日志。",
            "- 所有轮次都会保留；是否有权重仍按运行时原规则决定。",
            "- 报告不再依赖 `legacy_log` 或 `failed_iters`。",
            "",
            "## 图表产物",
            "",
            f"- `{_path_text(assets_dir / 'performance_delta.svg')}`",
            f"- `{_path_text(assets_dir / 'loss_delta.svg')}`",
            f"- `{_path_text(assets_dir / 'memory_delta.svg')}`",
        ]
    )

    lines.extend(
        [
            "",
            "## 功能问题明细",
            "",
        ]
    )

    if function_group_details:
        lines.extend(
            [
                "| 问题类别 | 归属 | 堆栈关键字 | 报错信息 | 迭代列表 |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for item in function_group_details:
            lines.append(
                "| {subtype} | {owner} | {stack} | {message} | {iters} |".format(
                    subtype=str(item.get("功能问题类型") or "-").replace("|", "\\|"),
                    owner=str(item.get("归属") or "待定").replace("|", "\\|"),
                    stack=str(item.get("堆栈关键字") or "-").replace("|", "\\|"),
                    message=str(item.get("报错信息") or "-").replace("|", "\\|").replace("\n", "<br>"),
                    iters=", ".join(item.get("迭代列表") or []).replace("|", "\\|"),
                )
            )
    else:
        lines.append("- 未发现功能问题。")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _status_badge(status: str) -> str:
    color = STATUS_BADGE_COLORS.get(status, "#475569")
    return (
        f'<span style="display:inline-block;padding:4px 10px;border-radius:999px;'
        f'background:{color};color:#fff;font-size:12px">{html.escape(status)}</span>'
    )


def _markdown_status(status: str) -> str:
    return "pass" if status == "PASS" else "fail"


def _compare_aware_text(text: str, compare_label: str) -> str:
    return text if compare_label == "MSA" else text.replace("MSA", compare_label)


def _overall_result_text(record: IterationAnalysis) -> str:
    labels: list[str] = []
    if "功能问题" in record.categories:
        owner = _functional_owner_from_record(record)
        if owner == "PTA问题":
            labels.append("PTA功能异常")
        elif owner == "MS问题":
            labels.append("MS功能异常")
        else:
            labels.append("功能异常")
    if "性能问题" in record.categories or _is_metric_issue_severity(record.performance_severity):
        labels.append("性能异常")
    if "精度问题" in record.categories or _is_metric_issue_severity(record.precision_severity):
        labels.append("精度异常")
    if "显存问题" in record.categories or _is_metric_issue_severity(record.memory_severity):
        labels.append("显存异常")
    return "/".join(dict.fromkeys(labels)) if labels else "正常"


def _write_html_report(path: Path, payload: dict, records: list[IterationAnalysis], analysis_dir: Path) -> None:
    profile = _task_profile(int(payload["task_type"]))
    compare_label = str(payload.get("对比对象") or "MSA")
    hero_title = _compare_aware_text(profile["hero_title"], compare_label)
    card_style = (
        "background:#fff;border:1px solid #e2e8f0;border-radius:16px;padding:18px 20px;"
        "box-shadow:0 10px 30px rgba(15,23,42,0.05);"
    )
    rows_html = []
    step_rows_html = []
    for record in records:
        metric_snapshot = _summary_metric_snapshot(record, profile)
        step_text = " / ".join(f"{key}:{value}" for key, value in record.step_results.items()) or "-"
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(record.iteration_tag)}</td>"
            f"<td>{_status_badge(record.overall_status)}</td>"
            f"<td>{html.escape(', '.join(record.categories) or '通过')}</td>"
            f"<td>{html.escape(_format_float(metric_snapshot['pta_perf']))}</td>"
            f"<td>{html.escape(_format_float(metric_snapshot['msa_perf']))}</td>"
            f"<td>{html.escape(_format_percent(record.performance_delta_ratio))}</td>"
            f"<td>{html.escape(record.performance_severity)}</td>"
            f"<td>{html.escape(_format_float(record.max_loss_diff, 8))}</td>"
            f"<td>{html.escape(record.precision_severity)}</td>"
            f"<td>{html.escape(_format_float(metric_snapshot['pta_mem'], 2))}</td>"
            f"<td>{html.escape(_format_float(metric_snapshot['msa_mem'], 2))}</td>"
            f"<td>{html.escape(_format_percent(record.memory_delta_ratio))}</td>"
            f"<td>{html.escape(record.memory_severity)}</td>"
            "</tr>"
        )
        step_rows_html.append(
            "<tr>"
            f"<td>{html.escape(record.iteration_tag)}</td>"
            f"<td>{html.escape(step_text)}</td>"
            "</tr>"
        )

    function_group_details = payload.get("功能问题详情") or payload["功能问题"]
    function_items = []
    for item in function_group_details:
        repro_entries = item.get("复现入口") or []
        repro_entry = html.escape(repro_entries[0]) if repro_entries else "-"
        subtype = html.escape(item.get("功能问题类型") or "-")
        message = html.escape(item.get("报错信息") or next(
            (value for key, value in item.items() if str(key).startswith("报错信息")),
            "",
        ))
        function_items.append(
            "<div style='padding:14px 0;border-bottom:1px solid #e2e8f0'>"
            f"<div style='font-weight:700;color:#0f172a'>{message}</div>"
            f"<div style='margin-top:8px;color:#334155'>类型: {subtype}</div>"
            f"<div style='margin-top:8px;color:#475569'>迭代: {html.escape(', '.join(item['迭代列表']))}</div>"
            f"<div style='margin-top:8px;color:#64748b;font-size:13px;word-break:break-all'>复现入口: {repro_entry}</div>"
            "</div>"
        )
    if not function_items:
        function_items.append("<div style='color:#475569'>未发现功能问题分组。</div>")

    mf_card = ""
    if "MF问题" in payload:
        mf_card = (
            f"<div style=\"{card_style}\"><div style=\"color:#64748b\">MF问题轮次</div>"
            f"<div style=\"font-size:32px;font-weight:800\">{payload['MF问题']['数量']}</div></div>"
        )

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(profile['task_title'])}</title>
  <style>
    :root {{
      --bg: #f8fafc;
      --panel: #ffffff;
      --line: #e2e8f0;
      --text: #0f172a;
      --sub: #475569;
      --accent: #0f766e;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.10), transparent 28%),
        linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
      color: var(--text);
    }}
    .wrap {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 36px 24px 64px;
    }}
    .hero {{
      padding: 28px;
      border-radius: 28px;
      background: linear-gradient(135deg, #ffffff 0%, #ecfeff 100%);
      border: 1px solid rgba(15,118,110,0.12);
      box-shadow: 0 20px 60px rgba(15,23,42,0.08);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-top: 24px;
    }}
    .section {{
      margin-top: 24px;
      {card_style}
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      padding: 12px 10px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}
    th {{
      color: var(--sub);
      font-weight: 600;
    }}
    img {{
      width: 100%;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: #fff;
    }}
    .chart-frame {{
      overflow-x: auto;
      overflow-y: hidden;
      padding-bottom: 6px;
      scrollbar-width: thin;
      scrollbar-color: #94a3b8 transparent;
    }}
    .chart-frame img {{
      display: block;
      width: auto;
      min-width: 100%;
      max-width: none;
    }}
    code {{
      background: #f1f5f9;
      padding: 2px 6px;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <div style="font-size:14px;color:#0f766e;font-weight:700;letter-spacing:0.08em">{html.escape(profile['task_name'].upper())} AUTO ANALYSIS</div>
      <h1 style="margin:10px 0 8px;font-size:34px">{html.escape(hero_title)}</h1>
      <div style="color:#475569;line-height:1.7">
        模型 <code>{html.escape(str(payload['model_name']))}</code>，
        迭代目录根 <code>{html.escape(str(payload['迭代报告目录']))}</code>，
        复现目录根 <code>{html.escape(str(payload['复现目录']))}</code>，
        生成时间 <code>{html.escape(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}</code>。
      </div>
      <div class="grid">
        <div style="{card_style}"><div style="color:#64748b">实际执行轮次</div><div style="font-size:32px;font-weight:800">{payload['执行总数']}</div></div>
        <div style="{card_style}"><div style="color:#64748b">变异成功率</div><div style="font-size:32px;font-weight:800">{payload['变异成功率']}</div></div>
        <div style="{card_style}"><div style="color:#64748b">PTA执行成功率</div><div style="font-size:32px;font-weight:800">{payload['PTA执行成功率']}</div></div>
        <div style="{card_style}"><div style="color:#64748b">{compare_label}执行成功率</div><div style="font-size:32px;font-weight:800">{payload.get('MS执行成功率') or '-'}</div></div>
        <div style="{card_style}"><div style="color:#64748b">有效对比轮次</div><div style="font-size:32px;font-weight:800">{payload['有效对比轮次']}</div></div>
        <div style="{card_style}"><div style="color:#64748b">问题总轮次</div><div style="font-size:32px;font-weight:800">{sum(1 for record in records if record.categories)}</div></div>
        {mf_card}
      </div>
    </div>

    <div class="grid">
      <div class="section">
        <div style="font-size:20px;font-weight:700">问题归类</div>
        <div style="margin-top:12px;color:#475569;line-height:1.9">
          功能问题组数: <b>{len(function_group_details)}</b><br/>
          精度问题轮次: <b>{payload['精度问题']['数量']}</b><br/>
          性能问题轮次: <b>{payload['性能问题']['数量']}</b><br/>
          显存问题轮次: <b>{payload['显存问题']['数量']}</b>
        </div>
      </div>
      <div class="section">
        <div style="font-size:20px;font-weight:700">验证标准</div>
        <div style="margin-top:12px;color:#475569;line-height:1.9">
          性能: {html.escape(payload['验证标准']['性能'])}<br/>
          精度: {html.escape(payload['验证标准']['精度'])}<br/>
          显存: {html.escape(payload['验证标准']['显存'])}
        </div>
      </div>
    </div>

    <div class="section">
      <div style="font-size:20px;font-weight:700;margin-bottom:12px">迭代总表</div>
      <table>
        <thead>
          <tr>
            <th>迭代</th>
            <th>状态</th>
            <th>问题分类</th>
            <th>PTA性能(s)</th>
            <th>{compare_label}性能(s)</th>
            <th>性能差值</th>
            <th>性能情况</th>
            <th>max loss diff</th>
            <th>loss情况</th>
            <th>PTA显存(MB)</th>
            <th>{compare_label}显存(MB)</th>
            <th>显存差值</th>
            <th>显存情况</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>
    </div>

        <div class="section">
            <div style="font-size:20px;font-weight:700;margin-bottom:12px">步骤执行结果</div>
            <table>
                <thead>
                    <tr>
                        <th>迭代</th>
                        <th>步骤结果</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(step_rows_html)}
                </tbody>
            </table>
        </div>

    <div class="grid">
      <div class="section"><div class="chart-frame"><img src="assets/performance_delta.svg" alt="performance_delta" /></div></div>
      <div class="section"><div class="chart-frame"><img src="assets/loss_delta.svg" alt="loss_delta" /></div></div>
    </div>
    <div class="section"><div class="chart-frame"><img src="assets/memory_delta.svg" alt="memory_delta" /></div></div>

    <div class="section">
      <div style="font-size:20px;font-weight:700;margin-bottom:12px">功能问题分组</div>
      {''.join(function_items)}
    </div>

    <div class="section">
      <div style="font-size:20px;font-weight:700;margin-bottom:12px">目录说明</div>
      <div style="color:#475569;line-height:1.9">
        每个 <code>iters/iter_x</code> 目录都会直接保留运行材料，并生成本轮 <code>report.md</code>。<br/>
        复现时直接进入对应 <code>iters/iter_x</code>，查看 <code>status.json</code>、<code>scripts/</code>、
        <code>weights/</code>、<code>mutation_inputs/</code>、<code>runtime_logs/</code> 和相关 CSV / 分析结果即可。<br/>
        报告不再依赖 <code>legacy_log</code>、<code>failed_iters</code> 或单独的 <code>iter_reports</code>。
      </div>
    </div>
  </div>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def _write_repro_readme(path: Path, payload: dict) -> None:
    compare_label = str(payload.get("对比对象") or "MSA")
    lines = [
        "# output 目录说明",
        "",
        "本目录本身就是运行材料与复现入口，不再依赖 `legacy_log`、`failed_iters` 或 `iter_reports`。",
        "",
        "## 目录结构",
        "",
        "- `iters/iter_x/report.md`: 当前轮次的简版分析报告。",
        "- `iters/iter_x/status.json`: 当前轮次组件状态。",
        "- `iters/iter_x/scripts/`: 本轮相关脚本（如 PTA-SAVE / PTA-LOAD / MSA-LOAD / MF 等）。",
        "- `iters/iter_x/weights/`: PTA-SAVE 产出的权重。",
        "- `iters/iter_x/mutation_inputs/`: 本轮变异输入 JSON / YAML。",
        "- `iters/iter_x/runtime_logs/`: 本轮运行日志。",
        "- `iters/iter_x/msrun_log/`: 本轮 msrun 日志快照。",
        "- `iters/iter_x/FAILED_FLAG` / `failure_info.txt`: 失败标记与失败原因。",
        "- `iters/iter_x/*.csv|*.txt|*.json`: 本轮快照结果。",
        "",
        "## 使用方式",
        "",
        f"- 先看 `iters/iter_x/report.md`，快速确认本轮状态、指标和 PTA/{compare_label} 对比结果。",
        "- 再结合 `status.json`、`scripts/`、`weights/` 和 `mutation_inputs/` 做手工复现。",
        "- 也可以直接在项目根目录执行 `python repro.py`，按提示选择 output / iter / run 进行单次复现（task1/2/3 均支持）。",
        "- 如果需要日志细节，直接看对应 `iter_x` 目录里的日志与快照文件即可。",
        "",
        f"当前输出目录: `{payload['output目录']}`",
        f"当前迭代报告目录: `{payload['迭代报告目录']}`",
        f"当前复现目录根: `{payload['复现目录']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_task_run(
    output_root: str | Path,
    run_dir: str | Path | None = None,
    model_name: Optional[str] = None,
    planned_iterations: Optional[int] = None,
    task_type: Optional[int] = None,
) -> AnalysisArtifacts:
    output_root_path = Path(output_root).resolve()
    run_dir_path = _resolve_run_dir(output_root_path, Path(run_dir) if run_dir else None)
    resolved_task_type = task_type or _detect_task_type(run_dir_path)
    profile = _task_profile(resolved_task_type)
    task_config = _load_task_config(output_root_path, resolved_task_type)
    compare_mode = str(task_config.get("COMPARE_MODE") or "").strip().lower() or None
    compare_label = _compare_label_for_mode(resolved_task_type, compare_mode)
    compare_to_mf = resolved_task_type in {1, 2, 3} and compare_mode == "pta_mf"
    model_name, planned_iterations = _resolve_run_metadata(
        output_root_path,
        run_dir_path,
        resolved_task_type,
        model_name,
        planned_iterations,
    )

    analysis_dir = output_root_path / "analysis"
    data_dir = analysis_dir / "data"
    assets_dir = analysis_dir / "assets"

    data_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    records: list[IterationAnalysis] = []
    for iter_dir in _iter_dirs(run_dir_path):
        iteration = _extract_iteration_number(iter_dir)
        status_payload = _read_status_payload(iter_dir)
        status_components = {
            str(key): str(value)
            for key, value in (status_payload.get("components") or {}).items()
        }

        pta_csv = _find_current_csv(iter_dir, resolved_task_type, "pta", iteration)
        msa_csv = _find_current_csv(iter_dir, resolved_task_type, "msa", iteration)
        mf_csv = _find_current_csv(iter_dir, resolved_task_type, "mf", iteration)
        pta_step_csv = _find_step_csv(iter_dir, resolved_task_type, "pta", iteration)
        msa_step_csv = _find_step_csv(iter_dir, resolved_task_type, "msa", iteration)
        mf_step_csv = _find_step_csv(iter_dir, resolved_task_type, "mf", iteration)
        if compare_to_mf:
            msa_csv = mf_csv
            msa_step_csv = mf_step_csv or mf_csv
        status_file = (iter_dir / "status.json") if (iter_dir / "status.json").exists() else None

        mutation_inputs = _find_mutation_inputs(iter_dir, iteration)
        runtime_logs, msrun_logs = _collect_log_paths(iter_dir)
        failed_flag = (
            _iter_aux_file(iter_dir, "FAILED_FLAG").exists()
            or str(status_payload.get("overall_status") or "").strip() not in {"", "PASS"}
        )

        signals: list[IssueSignal] = []
        scan_targets = runtime_logs + msrun_logs + [_iter_aux_file(iter_dir, "failure_info.txt")]
        for log_path in scan_targets:
            signals.extend(_scan_log_for_signals(log_path))
            if len(signals) >= rules.MAX_SIGNALS_PER_ITER:
                break
        signals = signals[: rules.MAX_SIGNALS_PER_ITER]

        comparison_metrics = _empty_comparison_metrics()
        pta_valid = False
        msa_valid = False
        mf_valid: Optional[bool] = None
        performance_source = "single_row"

        if profile["comparison_mode"] == "series":
            pta_valid = _training_csv_has_valid_rows(pta_csv)
            msa_valid = _training_csv_has_valid_rows(msa_csv)
            if compare_to_mf:
                mf_valid = msa_valid
            if pta_valid and msa_valid:
                comparison_metrics = _compare_series_csvs(pta_csv, msa_csv)
                performance_source = "series_skip1"
        else:
            pta_metrics = _read_single_iteration_metrics(pta_csv, iteration) if pta_csv else None
            msa_metrics = _read_single_iteration_metrics(msa_csv, iteration) if msa_csv else None
            mf_metrics = _read_single_iteration_metrics(mf_csv, iteration) if mf_csv else None
            if pta_metrics is None and pta_step_csv is not None:
                pta_metrics = _read_last_training_metrics(pta_step_csv)
            if msa_metrics is None and msa_step_csv is not None:
                msa_metrics = _read_last_training_metrics(msa_step_csv)
            if mf_metrics is None and mf_step_csv is not None:
                mf_metrics = _read_last_training_metrics(mf_step_csv)
            pta_step_valid = _training_csv_has_valid_rows(pta_step_csv)
            msa_step_valid = _training_csv_has_valid_rows(msa_step_csv)
            pta_valid = pta_metrics is not None
            msa_valid = msa_metrics is not None
            if compare_to_mf:
                mf_valid = msa_valid
            if pta_step_valid and msa_step_valid:
                comparison_metrics = _compare_series_csvs(pta_step_csv, msa_step_csv)
                performance_source = "series_step_csv"
            elif pta_valid and msa_valid and pta_metrics and msa_metrics:
                comparison_metrics = _compare_single_iteration(pta_metrics, msa_metrics)

            if profile["supports_mf"]:
                if status_components.get("MF") == "DISABLED":
                    mf_valid = None
                elif mf_csv is not None or "MF" in status_components:
                    mf_valid = mf_metrics is not None

        mutation_success = (
            _component_is_success(status_components.get("MUTATE"))
            if "MUTATE" in status_components
            else bool(mutation_inputs)
        )
        pta_execution_success = (
            _component_is_success(status_components.get("PTA_LOAD"))
            if "PTA_LOAD" in status_components
            else pta_valid
        )
        msa_execution_success = (
            _component_is_success(status_components.get("MF"))
            if compare_to_mf and "MF" in status_components
            else _component_is_success(status_components.get("MSA_LOAD"))
            if "MSA_LOAD" in status_components
            else msa_valid
        )
        comparison_available = pta_valid and msa_valid
        signals = _filter_suppressed_signals(signals, mutation_success, comparison_available)
        mf_required = profile["supports_mf"] and status_components.get("MF") != "DISABLED" and (
            mf_csv is not None or "MF" in status_components
        )
        mf_execution_success = (
            _component_is_success(status_components.get("MF"))
            if "MF" in status_components and status_components.get("MF") != "DISABLED"
            else mf_valid
        )
        single_metrics = {
            "pta": _load_metrics_from_paths(resolved_task_type, iteration, pta_csv, pta_step_csv),
            "msa": _load_metrics_from_paths(resolved_task_type, iteration, msa_csv, msa_step_csv),
            "mf": _load_metrics_from_paths(resolved_task_type, iteration, mf_csv, mf_step_csv),
        }
        step_results = _build_step_results(
            iter_dir=iter_dir,
            task_type=resolved_task_type,
            iteration=iteration,
            compare_mode=compare_mode,
            status_components=status_components,
            mutation_success=mutation_success,
            pta_execution_success=pta_execution_success,
            msa_execution_success=msa_execution_success,
            mf_execution_success=mf_execution_success,
        )

        functional_reasons = _derive_functional_reasons(
            iter_dir=iter_dir,
            task_type=resolved_task_type,
            compare_mode=compare_mode,
            mutation_success=mutation_success,
            pta_valid=pta_valid,
            msa_valid=msa_valid,
            mf_required=mf_required,
            mf_valid=mf_valid,
            status_payload=status_payload,
        )
        signals = _suppress_keyword_only_functional_signals(
            task_type=resolved_task_type,
            signals=signals,
            comparison_available=comparison_available,
            comparison_metrics=comparison_metrics,
            functional_reasons=functional_reasons,
        )

        categories, overall_status = _categorize_iteration(
            mutation_success=mutation_success,
            comparison_available=comparison_available,
            failed_flag=failed_flag,
            comparison_metrics=comparison_metrics,
            issue_signals=signals,
            functional_reasons=functional_reasons,
        )

        record = IterationAnalysis(
            task_type=resolved_task_type,
            iteration=iteration,
            iteration_tag=f"iter{iteration}",
            iteration_dir=_path_text(iter_dir) or str(iter_dir),
            failed_flag=failed_flag,
            mutation_success=mutation_success,
            pta_execution_success=pta_execution_success,
            msa_execution_success=msa_execution_success,
            mf_execution_success=mf_execution_success,
            comparison_available=comparison_available,
            overall_status=overall_status,
            categories=categories,
            functional_reasons=functional_reasons,
            issue_signals=signals,
            pta_csv=_path_text(pta_csv),
            msa_csv=_path_text(msa_csv),
            pta_step_csv=_path_text(pta_step_csv),
            msa_step_csv=_path_text(msa_step_csv),
            mf_step_csv=_path_text(mf_step_csv),
            pta_avg_step_time_skip1=comparison_metrics["pta_avg_step_time_skip1"],
            msa_avg_step_time_skip1=comparison_metrics["msa_avg_step_time_skip1"],
            performance_delta_seconds=comparison_metrics["performance_delta_seconds"],
            performance_delta_ratio=comparison_metrics["performance_delta_ratio"],
            performance_severity=str(comparison_metrics["performance_severity"]),
            performance_source=performance_source,
            max_loss_diff=comparison_metrics["max_loss_diff"],
            avg_loss_diff=comparison_metrics["avg_loss_diff"],
            precision_severity=str(comparison_metrics["precision_severity"]),
            pta_max_memory_mb=comparison_metrics["pta_max_memory_mb"],
            msa_max_memory_mb=comparison_metrics["msa_max_memory_mb"],
            memory_delta_mb=comparison_metrics["memory_delta_mb"],
            memory_delta_ratio=comparison_metrics["memory_delta_ratio"],
            memory_severity=str(comparison_metrics["memory_severity"]),
            runtime_log_paths=[_path_text(path) or str(path) for path in runtime_logs],
            msrun_log_paths=[_path_text(path) or str(path) for path in msrun_logs],
            mutation_input_paths=[_path_text(path) or str(path) for path in mutation_inputs],
            mf_csv=_path_text(mf_csv),
            mf_valid=mf_valid,
            pta_loss=(single_metrics.get("pta") or {}).get("loss"),
            msa_loss=(single_metrics.get("msa") or {}).get("loss"),
            mf_loss=(single_metrics.get("mf") or {}).get("loss"),
            status_file=_path_text(status_file),
            status_components=status_components,
            step_results=step_results,
            compare_mode=compare_mode,
        )

        record.reproduction_dir = _path_text(iter_dir) or str(iter_dir)
        iteration_report_path = iter_dir / "report.md"
        report_text = _render_iteration_report(
            record,
            profile,
            None,
            iter_dir,
        )
        iteration_report_path.write_text(report_text, encoding="utf-8")
        record.iteration_report_path = _path_text(iteration_report_path)

        records.append(record)

    issue_groups = _build_issue_groups(records)
    payload = _summary_payload(
        resolved_task_type,
        output_root_path,
        run_dir_path,
        model_name,
        planned_iterations,
        records,
        issue_groups,
    )
    strict_summary_payload = _strict_summary_payload(
        model_name=model_name,
        planned_iterations=planned_iterations,
        run_dir=run_dir_path,
        records=records,
        issue_groups=issue_groups,
    )

    summary_json_path = data_dir / "summary.json"
    iteration_csv_path = data_dir / "iteration_metrics.csv"
    issue_json_path = data_dir / "issue_groups.json"
    summary_md_path = analysis_dir / "summary.md"
    report_html_path = analysis_dir / "report.html"

    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(strict_summary_payload, handle, ensure_ascii=False, indent=2)
    with issue_json_path.open("w", encoding="utf-8") as handle:
        json.dump(issue_groups, handle, ensure_ascii=False, indent=2)

    _write_iteration_csv(iteration_csv_path, records)

    performance_rows = [
        (
            record.iteration_tag,
            _regression_value(record.performance_delta_ratio),
            _series_color(_regression_value(record.performance_delta_ratio), positive_bad=True),
        )
        for record in records
    ]
    loss_rows = [
        (record.iteration_tag, record.max_loss_diff, _series_color(record.max_loss_diff, positive_bad=False))
        for record in records
    ]
    memory_rows = [
        (
            record.iteration_tag,
            _regression_value(record.memory_delta_ratio),
            _series_color(_regression_value(record.memory_delta_ratio), positive_bad=True),
        )
        for record in records
    ]

    performance_svg = assets_dir / "performance_delta.svg"
    loss_svg = assets_dir / "loss_delta.svg"
    memory_svg = assets_dir / "memory_delta.svg"
    performance_chart_title = f"{compare_label} vs PTA 性能差值"
    loss_chart_title = f"{compare_label} vs PTA max loss 绝对差"
    memory_chart_title = f"{compare_label} vs PTA 显存差值"
    performance_chart_subtitle = _compare_aware_text(profile["performance_chart_subtitle"], compare_label)
    memory_chart_subtitle = _compare_aware_text(profile["memory_chart_subtitle"], compare_label)

    _write_svg_bar_chart(
        performance_svg,
        performance_chart_title,
        performance_chart_subtitle,
        performance_rows,
        percent_mode=True,
        negative_as_zero=True,
    )
    _write_svg_bar_chart(
        loss_svg,
        loss_chart_title,
        "标准为严格零误差；正值越大表示偏差越明显。",
        loss_rows,
        percent_mode=False,
    )
    _write_svg_bar_chart(
        memory_svg,
        memory_chart_title,
        memory_chart_subtitle,
        memory_rows,
        percent_mode=True,
        negative_as_zero=True,
    )

    _write_markdown(summary_md_path, payload, records, assets_dir)
    _write_html_report(report_html_path, payload, records, analysis_dir)
    _write_repro_readme(output_root_path / "README.md", payload)

    executed = len(records)
    mutation_success_count = sum(1 for record in records if record.mutation_success)
    pta_execution_success_count = sum(1 for record in records if record.pta_execution_success)
    msa_enabled_records = [
        record for record in records
        if (
            not _component_is_skipped(record.status_components.get("MSA_LOAD"))
            or record.msa_execution_success
            or record.msa_csv
        )
    ]
    msa_execution_success_count = sum(1 for record in msa_enabled_records if record.msa_execution_success)
    valid_comparisons = sum(1 for record in records if record.comparison_available)
    functional_failures = sum(1 for record in records if "功能问题" in record.categories)
    precision_failures = sum(1 for record in records if "精度问题" in record.categories)
    performance_failures = sum(1 for record in records if "性能问题" in record.categories)
    memory_failures = sum(1 for record in records if "显存问题" in record.categories)
    mf_failures = sum(1 for record in records if record.mf_valid is False)

    return AnalysisArtifacts(
        task_type=resolved_task_type,
        output_root=_path_text(output_root_path) or str(output_root_path),
        run_dir=_path_text(run_dir_path) or str(run_dir_path),
        analysis_dir=_path_text(analysis_dir) or str(analysis_dir),
        summary_json=_path_text(summary_json_path) or str(summary_json_path),
        summary_markdown=_path_text(summary_md_path) or str(summary_md_path),
        report_html=_path_text(report_html_path) or str(report_html_path),
        iteration_csv=_path_text(iteration_csv_path) or str(iteration_csv_path),
        issue_json=_path_text(issue_json_path) or str(issue_json_path),
        svg_assets=[
            _path_text(performance_svg) or str(performance_svg),
            _path_text(loss_svg) or str(loss_svg),
            _path_text(memory_svg) or str(memory_svg),
        ],
        repro_root=_path_text(run_dir_path) or str(run_dir_path),
        iteration_report_root=_path_text(run_dir_path) or str(run_dir_path),
        executed_iterations=executed,
        planned_iterations=planned_iterations,
        mutation_success_count=mutation_success_count,
        mutation_success_rate=(mutation_success_count / executed) if executed else 0.0,
        pta_execution_success_count=pta_execution_success_count,
        pta_execution_success_rate=(pta_execution_success_count / executed) if executed else 0.0,
        msa_execution_success_count=msa_execution_success_count,
        msa_execution_success_rate=(
            (msa_execution_success_count / len(msa_enabled_records))
            if msa_enabled_records
            else None
        ),
        valid_comparisons=valid_comparisons,
        functional_failures=functional_failures,
        precision_failures=precision_failures,
        performance_failures=performance_failures,
        memory_failures=memory_failures,
        mf_failures=mf_failures,
    )


def analyze_task1_run(
    output_root: str | Path,
    run_dir: str | Path | None = None,
    model_name: Optional[str] = None,
    planned_iterations: Optional[int] = None,
) -> AnalysisArtifacts:
    return analyze_task_run(
        output_root=output_root,
        run_dir=run_dir,
        model_name=model_name,
        planned_iterations=planned_iterations,
        task_type=1,
    )


def analyze_task2_run(
    output_root: str | Path,
    run_dir: str | Path | None = None,
    model_name: Optional[str] = None,
    planned_iterations: Optional[int] = None,
) -> AnalysisArtifacts:
    return analyze_task_run(
        output_root=output_root,
        run_dir=run_dir,
        model_name=model_name,
        planned_iterations=planned_iterations,
        task_type=2,
    )


def analyze_task3_run(
    output_root: str | Path,
    run_dir: str | Path | None = None,
    model_name: Optional[str] = None,
    planned_iterations: Optional[int] = None,
) -> AnalysisArtifacts:
    return analyze_task_run(
        output_root=output_root,
        run_dir=run_dir,
        model_name=model_name,
        planned_iterations=planned_iterations,
        task_type=3,
    )


def main_cli_for_task(task_type: Optional[int], argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Task output 自动分析")
    parser.add_argument("output_root", help="output 根目录，例如 output/2026-03-06-17-30-29")
    parser.add_argument("--run-dir", help="指定 run 目录；兼容 legacy_log 下旧结构")
    parser.add_argument("--model-name", help="覆盖模型名")
    parser.add_argument("--planned-iters", type=int, help="覆盖计划轮次")
    parser.add_argument("--task-type", type=int, choices=[1, 2, 3], help="显式指定任务类型")
    args = parser.parse_args(argv)

    resolved_task_type = task_type or args.task_type
    artifacts = analyze_task_run(
        output_root=args.output_root,
        run_dir=args.run_dir,
        model_name=args.model_name,
        planned_iterations=args.planned_iters,
        task_type=resolved_task_type,
    )
    print(json.dumps(asdict(artifacts), ensure_ascii=False, indent=2))
    return 0


def main_cli(argv: Optional[list[str]] = None) -> int:
    return main_cli_for_task(None, argv)


if __name__ == "__main__":
    raise SystemExit(main_cli())
