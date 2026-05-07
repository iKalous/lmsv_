#!/usr/bin/env python3
from __future__ import annotations
import json, csv, html, re, shutil, statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from .task1_result import AnalysisArtifacts, IterationAnalysis, IssueSignal, _task_profile, _load_task_config, _resolve_run_dir, _detect_task_type, _resolve_run_metadata, _iter_dirs, _extract_iteration_number, _read_status_payload, _find_current_csv, _find_step_csv, _find_mutation_inputs, _collect_log_paths, _scan_log_for_signals, _empty_comparison_metrics, _compare_series_csvs, _read_single_iteration_metrics, _read_last_training_metrics, _compare_single_iteration, _component_is_success, _component_is_skipped, _filter_suppressed_signals, _build_issue_groups, _summary_payload, _strict_summary_payload, _write_iteration_csv, _write_svg_bar_chart, _write_markdown, _write_html_report, _write_repro_readme, _regression_value, _series_color, _categorize_iteration, _derive_functional_reasons, _build_functional_issue_stack, _path_text, _iter_aux_file, _render_iteration_report
from .task1_result import _load_metrics_from_paths, _build_step_results, _training_csv_has_valid_rows
from . import rules

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

    _write_svg_bar_chart(
        performance_svg,
        "MSA vs PTA 性能差值",
        profile["performance_chart_subtitle"],
        performance_rows,
        percent_mode=True,
        negative_as_zero=True,
    )
    _write_svg_bar_chart(
        loss_svg,
        "MSA vs PTA max loss 绝对差",
        "标准为严格零误差；正值越大表示偏差越明显。",
        loss_rows,
        percent_mode=False,
    )
    _write_svg_bar_chart(
        memory_svg,
        "MSA vs PTA 显存差值",
        profile["memory_chart_subtitle"],
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
