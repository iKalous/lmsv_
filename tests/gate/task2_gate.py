#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.json"
CASES_PATH = Path(__file__).resolve().parent / "task2_cases.json"
CASES_PATH_BY_TASK = {
    1: Path(__file__).resolve().parent / "task1_cases.json",
    2: CASES_PATH,
    3: Path(__file__).resolve().parent / "task3_cases.json",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _deep_merge_dict(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _normalize_models(models: Any) -> list[str]:
    if not isinstance(models, list):
        return []
    return [str(x).strip() for x in models if str(x).strip()]


def _normalize_ints(values: Any) -> list[int]:
    if not isinstance(values, list):
        return []
    return [v for v in values if isinstance(v, int)]


def _extract_case_meta(config_patch: dict[str, Any]) -> tuple[int, str, str]:
    task_type = config_patch.get("task_type")
    if not isinstance(task_type, int):
        task_type = -1

    tasks = config_patch.get("tasks")
    if not isinstance(tasks, dict):
        return task_type, "unknown", ""

    task_cfg = tasks.get(str(task_type)) if task_type > 0 else None
    if not isinstance(task_cfg, dict):
        return task_type, "unknown", ""

    if task_type == 1:
        model_name = str(task_cfg.get("MODEL_NAME", "")).strip()
        if not model_name:
            return task_type, "task1", ""
        return task_type, model_name, ""

    if task_type == 2:
        models = _normalize_models(task_cfg.get("MODELS"))
        submodules = _normalize_ints(task_cfg.get("SUBMODULES"))
        if not models or not submodules:
            return task_type, "task2", ""

        unique_models = sorted(set(models))
        model_label = unique_models[0] if len(unique_models) == 1 else "+".join(unique_models)
        submodule_label = ",".join(str(x) for x in submodules)
        return task_type, model_label, submodule_label

    if task_type == 3:
        models = _normalize_models(task_cfg.get("MODELS"))
        if not models:
            return task_type, "task3", ""
        model_label = "+".join(models)
        return task_type, model_label, ""

    return task_type, f"task{task_type}", ""


def _extract_runtime_envs(config_patch: dict[str, Any]) -> tuple[str, str, str]:
    pta = str(config_patch.get("PTA_NAME", "")).strip() or "<unset>"
    msa = str(config_patch.get("MSA_NAME", "")).strip() or "<unset>"
    mf = str(config_patch.get("MF_NAME", "")).strip() or "<unset>"
    return pta, msa, mf


def _load_cases(path: Path) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    payload = _read_json(path)
    base_patch = payload.get("base_config_patch")
    common = payload.get("common", {})
    cases = payload.get("cases")
    if not isinstance(base_patch, dict):
        raise ValueError(f"Invalid base_config_patch in {path}")
    if not isinstance(common, dict):
        raise ValueError(f"Invalid common section in {path}")
    if not isinstance(cases, list):
        raise ValueError(f"Invalid cases in {path}")

    default_env_name = str(common.get("env_name", "")).strip()

    normalized = []
    seen_ids = set()
    for raw in cases:
        if not isinstance(raw, dict):
            raise ValueError("Each case must be an object")
        case_id = str(raw.get("id", "")).strip()
        enabled = bool(raw.get("enabled", True))
        env_name = str(raw.get("env_name", "")).strip() or default_env_name
        config_patch = raw.get("config_patch")

        if not case_id:
            raise ValueError(f"Case missing id: {raw}")
        if case_id in seen_ids:
            raise ValueError(f"Duplicated case id: {case_id}")
        if not isinstance(config_patch, dict):
            raise ValueError(f"Case {case_id} missing config_patch")

        merged_preview = _deep_merge_dict(copy.deepcopy(base_patch), config_patch)
        task_type, model, submodule_text = _extract_case_meta(merged_preview)
        if task_type not in {1, 2, 3}:
            raise ValueError(f"Case {case_id} must resolve task_type to 1/2/3")
        if not model:
            raise ValueError(
                f"Case {case_id} missing model metadata for task{task_type}"
            )

        seen_ids.add(case_id)
        normalized.append(
            {
                "id": case_id,
                "enabled": enabled,
                "env_name": env_name,
                "config_patch": config_patch,
                "task_type": task_type,
                "model": model,
                "submodule": submodule_text,
            }
        )
    return base_patch, default_env_name, normalized


def _build_task2_config(
    base_config: dict[str, Any],
    base_patch: dict[str, Any],
    case_patch: dict[str, Any],
) -> dict[str, Any]:
    config = _deep_merge_dict(base_config, base_patch)
    config = _deep_merge_dict(config, case_patch)
    return config


def _pick_cases(
    all_cases: list[dict[str, Any]],
    selected_ids: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    enabled_cases = [c for c in all_cases if c["enabled"]]
    if selected_ids:
        by_id = {c["id"]: c for c in enabled_cases}
        missing = [cid for cid in selected_ids if cid not in by_id]
        if missing:
            raise ValueError(f"Unknown/disabled case ids: {', '.join(missing)}")
        chosen = [by_id[cid] for cid in selected_ids]
    else:
        chosen = enabled_cases

    if limit > 0:
        chosen = chosen[:limit]
    return chosen


def _run_one_case(case: dict[str, Any], env_name_override: str) -> tuple[bool, float]:
    start = time.time()
    env_name = env_name_override.strip()
    cmd = ["python", "do.py"]
    if env_name:
        cmd = ["conda", "run", "-n", env_name, "python", "do.py"]

    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    return proc.returncode == 0, time.time() - start


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task1/2/3 pre-merge gate runner")
    parser.add_argument(
        "--task",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Task type for case file auto-selection. Default: 2",
    )
    parser.add_argument(
        "--cases-file",
        default="",
        help="Path to gate case json file. Empty means auto by --task",
    )
    parser.add_argument(
        "--cases",
        default="",
        help="Comma separated case IDs to run. Empty means all enabled cases.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Run only first N selected cases. 0 means no limit.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List cases and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show selected cases and exit without running",
    )
    parser.add_argument(
        "--env-name",
        default="",
        help="Override conda env name for all selected cases",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    default_cases_file = CASES_PATH_BY_TASK[args.task]
    raw_cases_file = args.cases_file.strip() or str(default_cases_file)
    cases_file = Path(raw_cases_file).resolve()
    if not cases_file.exists():
        print(f"[gate] cases file not found: {cases_file}", file=sys.stderr)
        return 2

    if not CONFIG_PATH.exists():
        print(f"[gate] config not found: {CONFIG_PATH}", file=sys.stderr)
        return 2

    base_patch, default_env_name, all_cases = _load_cases(cases_file)
    env_name_override = str(args.env_name).strip()
    base_config = _read_json(CONFIG_PATH)
    runtime_envs_by_case: dict[str, tuple[str, str, str]] = {}
    for case in all_cases:
        preview_config = _build_task2_config(
            base_config=base_config,
            base_patch=base_patch,
            case_patch=case["config_patch"],
        )
        runtime_envs_by_case[case["id"]] = _extract_runtime_envs(preview_config)

    if args.list:
        for c in all_cases:
            state = "enabled" if c["enabled"] else "disabled"
            runner_env = env_name_override or "<current>"
            pta_env, msa_env, mf_env = runtime_envs_by_case.get(c["id"], ("<unset>", "<unset>", "<unset>"))
            task_label = f"task{c['task_type']}"
            submodule_suffix = f" submodule={c['submodule']}" if c["submodule"] else ""
            print(
                f"{c['id']}: {task_label} model={c['model']}{submodule_suffix} "
                f"runner_env={runner_env} runtime_envs=(pta={pta_env}, msa={msa_env}, mf={mf_env}) [{state}]"
            )
        return 0

    selected_ids = [x.strip() for x in args.cases.split(",") if x.strip()]
    selected = _pick_cases(all_cases, selected_ids, args.limit)
    if not selected:
        print("[gate] no case selected")
        return 2

    print(f"[gate] selected {len(selected)} case(s)")
    for idx, c in enumerate(selected, 1):
        runner_env = env_name_override or "<current>"
        pta_env, msa_env, mf_env = runtime_envs_by_case.get(c["id"], ("<unset>", "<unset>", "<unset>"))
        submodule_suffix = f" / submodule {c['submodule']}" if c["submodule"] else ""
        print(
            f"  {idx}. {c['id']} (task{c['task_type']} / {c['model']}{submodule_suffix} / "
            f"runner {runner_env} / runtime pta={pta_env}, msa={msa_env}, mf={mf_env})"
        )

    if args.dry_run:
        return 0

    backup_dir = Path(tempfile.mkdtemp(prefix=f"lmsv-task{args.task}-gate-"))
    backup_config = backup_dir / "config.json.bak"
    shutil.copy2(CONFIG_PATH, backup_config)

    results: list[dict[str, Any]] = []
    try:
        for idx, case in enumerate(selected, 1):
            case_config = _build_task2_config(
                base_config=base_config,
                base_patch=base_patch,
                case_patch=case["config_patch"],
            )
            _write_json(CONFIG_PATH, case_config)

            print(f"\n[gate] ({idx}/{len(selected)}) running {case['id']}")
            ok, elapsed = _run_one_case(case, env_name_override)
            results.append({"case": case, "ok": ok, "elapsed": elapsed})
            state = "PASS" if ok else "FAIL"
            print(f"[gate] {case['id']} -> {state} ({elapsed:.1f}s)")
            if not ok:
                print("[gate] stop on first failure")
                break
    finally:
        shutil.copy2(backup_config, CONFIG_PATH)
        shutil.rmtree(backup_dir, ignore_errors=True)

    passed = sum(1 for item in results if item["ok"])
    total = len(results)
    print("\n[gate] summary")
    print(f"[gate] passed: {passed}/{total}")
    for item in results:
        c = item["case"]
        state = "PASS" if item["ok"] else "FAIL"
        submodule_suffix = f", submodule={c['submodule']}" if c["submodule"] else ""
        runner_env = env_name_override or "<current>"
        pta_env, msa_env, mf_env = runtime_envs_by_case.get(c["id"], ("<unset>", "<unset>", "<unset>"))
        print(
            f"[gate] - {c['id']}: {state}, model={c['model']}, "
            f"task=task{c['task_type']}{submodule_suffix}, runner_env={runner_env}, "
            f"runtime_envs=(pta={pta_env}, msa={msa_env}, mf={mf_env}), "
            f"elapsed={item['elapsed']:.1f}s"
        )

    if total < len(selected):
        return 1
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
