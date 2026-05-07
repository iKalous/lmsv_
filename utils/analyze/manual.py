#!/usr/bin/env python3

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "output"
OUTPUT_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _has_iter_dirs(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(item.is_dir() and re.match(r"iter_\d+$", item.name) for item in path.iterdir())


def _has_iters_container(path: Path) -> bool:
    return _has_iter_dirs(path / "iters")


def load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def resolve_output_dir(value: str | Path, *, output_root: Path = OUTPUT_ROOT) -> Path:
    text = str(value).strip()
    if not text:
        raise ValueError("output 目录不能为空")

    requested = Path(text)
    output_root_resolved = output_root.resolve()
    candidates: list[Path] = []

    if OUTPUT_NAME_RE.fullmatch(text):
        candidates.append((output_root_resolved / text).resolve())

    if requested.is_absolute():
        candidates.append(requested.resolve())
    else:
        candidates.append((REPO_ROOT / requested).resolve())

    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if candidate.parent != output_root_resolved:
            raise ValueError(f"非法 output 目录: {candidate}")
        return candidate

    raise FileNotFoundError(f"output 目录不存在: {text}")


def resolve_run_dir(output_dir: Path, run_dir: str | Path | None) -> Path | None:
    if run_dir is None:
        return None

    raw = Path(str(run_dir).strip())
    if not str(raw):
        return None

    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw.resolve())
    else:
        candidates.append((output_dir / raw).resolve())
        candidates.append((output_dir / "legacy_log" / raw).resolve())
        candidates.append((REPO_ROOT / raw).resolve())

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(f"run_dir 不存在: {run_dir}")


def output_has_replayable_analysis_data(path: Path) -> bool:
    if _has_iters_container(path):
        return True
    if _has_iter_dirs(path):
        return True
    legacy_log_root = path / "legacy_log"
    if not legacy_log_root.is_dir():
        return False
    return any(_has_iter_dirs(item) or item.is_dir() for item in legacy_log_root.iterdir())


def infer_output_task_type(path: Path) -> int | None:
    summary = load_json(path / "analysis" / "data" / "summary.json")
    config = load_json(path / "config.json")
    raw_task_type = summary.get("task_type", config.get("task_type"))
    try:
        task_type = int(raw_task_type) if raw_task_type is not None else None
    except (TypeError, ValueError):
        return None
    return task_type if task_type in {1, 2, 3} else None


def discover_replayable_outputs(output_root: Path = OUTPUT_ROOT) -> list[Path]:
    if not output_root.exists():
        return []
    return sorted(
        (
            path
            for path in output_root.iterdir()
            if path.is_dir() and output_has_replayable_analysis_data(path)
        ),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )


def regenerate_output_analysis(
    value: str | Path,
    *,
    output_root: Path = OUTPUT_ROOT,
    run_dir: str | Path | None = None,
    model_name: Optional[str] = None,
    planned_iterations: Optional[int] = None,
    task_type: Optional[int] = None,
) -> dict:
    output_dir = resolve_output_dir(value, output_root=output_root)
    if not output_has_replayable_analysis_data(output_dir):
        raise FileNotFoundError(f"{output_dir} 缺少可分析的迭代数据")

    from .task1_result import _resolve_run_dir, _detect_task_type
    resolved_run_dir = _resolve_run_dir(output_dir, run_dir)
    task_type = _detect_task_type(resolved_run_dir)
    if task_type is None:
        raise ValueError(f"未找到可分析的 task_type: {resolved_run_dir}")
    if task_type in {1, 2, 3}:

        from .task1_result import analyze_task_run

        artifacts = analyze_task_run(
            output_root=output_dir,
            run_dir=resolved_run_dir,
            model_name=model_name,
            planned_iterations=planned_iterations,
            task_type=task_type or infer_output_task_type(output_dir),
        )
        payload = asdict(artifacts)
        payload.update(
            {
                "output_id": output_dir.name,
                "output_dir": str(output_dir),
                "report_url": f"results/{output_dir.name}/analysis/report.html",
                "message": f"已基于 output/{output_dir.name} 的历史数据重新生成 analysis/report.html",
            }
        )
        return payload

    elif task_type in {4, 5}:
        from .task45_result import analyze_task_run
        artifacts = analyze_task_run(
            output_root=output_dir,
            run_dir=resolved_run_dir,
            model_name=model_name,
            planned_iterations=planned_iterations,
            task_type=task_type or infer_output_task_type(output_dir),
        )
        payload = asdict(artifacts)
        payload.update(
            {
                "output_id": output_dir.name,
                "output_dir": str(output_dir),
                "report_url": f"results/{output_dir.name}/analysis/report.html",
                "message": f"已基于 output/{output_dir.name} 的历史数据重新生成 analysis/report.html",
            }
        )
        return payload

    else:
        raise ValueError(f"未找到可分析的 task_type: {task_type}")