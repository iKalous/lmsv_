#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import signal
import sys
from datetime import datetime
from pathlib import Path

from utils.analyze.manual import (
    discover_replayable_outputs,
    infer_output_task_type,
    load_json,
    output_has_replayable_analysis_data,
    regenerate_output_analysis,
)


TASK_LABELS = {
    1: "task1 整网泛化变异测试",
    2: "task2 模块内组件泛化测试",
    3: "task3 模块间泛化组合变异测试",
}


def _handle_sigint(_signum, _frame) -> None:
    print("\n[analyze] 已中断。", flush=True)
    raise SystemExit(130)


signal.signal(signal.SIGINT, _handle_sigint)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LMSV 历史数据 analysis 重生成工具")
    parser.add_argument("output", nargs="?", help="output 目录名，或 output/<目录> 路径")
    parser.add_argument("--latest", action="store_true", help="直接选择最近一次可分析的 output")
    parser.add_argument("--list", action="store_true", help="列出当前可用于重生成 analysis 的 output")
    parser.add_argument("--run-dir", help="指定 run 目录名或路径；兼容 legacy_log 下旧结构")
    parser.add_argument("--model-name", help="覆盖模型名")
    parser.add_argument("--planned-iters", type=int, help="覆盖计划轮次")
    parser.add_argument("--task-type", type=int, choices=[1, 2, 3], help="显式指定任务类型")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    return parser.parse_args(argv)


def task_summary(path: Path) -> str:
    summary = load_json(path / "analysis" / "data" / "summary.json")
    config = load_json(path / "config.json")
    task_type = infer_output_task_type(path)
    task_label = TASK_LABELS.get(task_type, "task? 未识别任务类型")
    model_name = summary.get("model_name")
    if not model_name:
        tasks = config.get("tasks") or {}
        task_config = tasks.get(str(task_type)) if task_type else {}
        model_name = task_config.get("MODEL_NAME") or task_config.get("MODELS") or "-"
    report_exists = (path / "analysis" / "report.html").exists()
    updated_at = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return f"{path.name} | {task_label} | model={model_name} | report={'yes' if report_exists else 'no'} | updated={updated_at}"


def choose_one(items: list[Path]) -> Path:
    while True:
        print()
        print("请选择要重新生成 analysis 的 output：")
        for index, item in enumerate(items, start=1):
            print(f"{index}. {task_summary(item)}")
        choice = input("请输入编号，或输入 q 退出: ").strip().lower()
        if choice in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        if not choice.isdigit():
            print("输入无效，请重新输入。")
            continue
        selected = int(choice)
        if 1 <= selected <= len(items):
            return items[selected - 1]
        print("编号超出范围，请重新输入。")


def print_output_list(outputs: list[Path]) -> int:
    if not outputs:
        print("未找到可用于重生成 analysis 的 output 目录。")
        return 0
    print("可用于重生成 analysis 的 output：")
    for item in outputs:
        print(f"- {task_summary(item)}")
    return 0


def pick_output(args: argparse.Namespace, outputs: list[Path]) -> Path:
    if args.output:
        from utils.analyze.manual import resolve_output_dir

        return resolve_output_dir(args.output)
    if args.latest:
        if not outputs:
            raise FileNotFoundError("未找到可用于重生成 analysis 的 output 目录")
        return outputs[0]
    if not sys.stdin.isatty():
        raise ValueError("未提供 output 参数；非交互终端下请显式传入 output 或 --latest")
    try:
        return choose_one(outputs)
    except KeyboardInterrupt:
        raise RuntimeError("已退出") from None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = discover_replayable_outputs()

    if args.list:
        return print_output_list(outputs)

    if not outputs:
        print("未找到可用于重生成 analysis 的 output 目录。")
        return 1

    try:
        target = pick_output(args, outputs)
    except Exception as exc:  # noqa: BLE001
        print(f"[analyze] {exc}")
        return 1

    if not output_has_replayable_analysis_data(target):
        print(f"[analyze] {target} 缺少可分析的迭代数据")
        return 1

    try:
        payload = regenerate_output_analysis(
            target,
            run_dir=args.run_dir,
            model_name=args.model_name,
            planned_iterations=args.planned_iters,
            task_type=args.task_type,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[analyze] {exc}")
        return 1

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print("[analyze] analysis 重生成完成")
    print(f"[analyze] output: {payload['output_id']}")
    print(f"[analyze] task_type: {payload['task_type']}")
    print(f"[analyze] analysis_dir: {payload['analysis_dir']}")
    print(f"[analyze] report_html: {payload['report_html']}")
    print(f"[analyze] summary_json: {payload['summary_json']}")
    print(f"[analyze] repro_root: {payload['repro_root']}")
    if payload.get("iteration_report_root"):
        print(f"[analyze] iteration_report_root: {payload['iteration_report_root']}")
    print(f"[analyze] executed_iterations: {payload['executed_iterations']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
