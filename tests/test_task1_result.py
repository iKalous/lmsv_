import math
import tempfile
import unittest
from pathlib import Path

from utils.analyze.task1_result import (
    IterationAnalysis,
    FunctionalReason,
    IssueSignal,
    _build_metric_overview_rows,
    _categorize_iteration,
    _compare_aware_text,
    _functional_owner_from_text,
    _markdown_status,
    _overall_result_text,
    _partial_comparison_summary_lines,
    _severity,
    _suppress_keyword_only_functional_signals,
    _summary_metric_snapshot,
    _task_profile,
    _training_csv_has_valid_rows,
    _write_markdown,
)


class CategorizeIterationTests(unittest.TestCase):
    def test_mf_failure_is_classified_as_ms_side_issue(self) -> None:
        self.assertEqual(_functional_owner_from_text("MF 执行失败"), "MS问题")
        self.assertEqual(_functional_owner_from_text("training_log_mf-1.csv"), "MS问题")

    def test_pta_failure_owner_does_not_become_pta_ms_public_issue(self) -> None:
        self.assertEqual(
            _functional_owner_from_text("PTA/MSA异常: PTA-LOAD 执行失败，msrun_log 中有错误"),
            "PTA问题",
        )

    def test_compare_aware_text_switches_msa_to_mf(self) -> None:
        self.assertEqual(
            _compare_aware_text("PTA / MSA 模块间泛化组合报告", "MF"),
            "PTA / MF 模块间泛化组合报告",
        )
        self.assertEqual(
            _compare_aware_text("正值表示 MSA 更慢，负值按 0 展示。", "MF"),
            "正值表示 MF 更慢，负值按 0 展示。",
        )

    def test_overall_result_only_lists_abnormal_dimensions(self) -> None:
        record = IterationAnalysis(
            task_type=1,
            iteration=33,
            iteration_tag="iter33",
            iteration_dir="/tmp/iter33",
            failed_flag=False,
            mutation_success=True,
            comparison_available=True,
            overall_status="COMPLETED_WITH_ISSUES",
            categories=["精度问题"],
            performance_severity="PASS",
            precision_severity="CRITICAL",
            memory_severity="PASS",
        )
        self.assertEqual(_overall_result_text(record), "精度异常")

    def test_overall_result_uses_functional_owner_label(self) -> None:
        record = IterationAnalysis(
            task_type=1,
            iteration=2,
            iteration_tag="iter2",
            iteration_dir="/tmp/iter2",
            failed_flag=True,
            mutation_success=True,
            comparison_available=False,
            overall_status="EXECUTION_FAILED",
            categories=["功能问题"],
            functional_reasons=[
                FunctionalReason(issue_subtype="执行失败", message="mindspore runtime failed")
            ],
        )
        self.assertEqual(_overall_result_text(record), "MS功能异常")

    def test_markdown_status_is_simplified_to_pass_fail(self) -> None:
        self.assertEqual(_markdown_status("PASS"), "pass")
        self.assertEqual(_markdown_status("COMPLETED_WITH_ISSUES"), "fail")
        self.assertEqual(_markdown_status("EXECUTION_FAILED"), "fail")

    def test_memory_info_is_treated_as_issue(self) -> None:
        categories, status = _categorize_iteration(
            mutation_success=True,
            comparison_available=True,
            failed_flag=False,
            comparison_metrics={
                "precision_severity": "PASS",
                "performance_severity": "PASS",
                "memory_severity": "INFO",
                "msa_avg_step_time_skip1": 1.0,
            },
            issue_signals=[],
            functional_reasons=[],
        )

        self.assertIn("显存问题", categories)
        self.assertEqual(status, "COMPLETED_WITH_ISSUES")

    def test_precision_nan_is_treated_as_critical_issue(self) -> None:
        self.assertEqual(_severity("precision", math.nan), "CRITICAL")

        categories, status = _categorize_iteration(
            mutation_success=True,
            comparison_available=True,
            failed_flag=False,
            comparison_metrics={
                "precision_severity": "CRITICAL",
                "performance_severity": "PASS",
                "memory_severity": "PASS",
                "msa_avg_step_time_skip1": 1.0,
            },
            issue_signals=[],
            functional_reasons=[],
        )

        self.assertIn("精度问题", categories)
        self.assertEqual(status, "COMPLETED_WITH_ISSUES")

    def test_unavailable_metrics_from_execution_failure_are_not_reclassified(self) -> None:
        categories, status = _categorize_iteration(
            mutation_success=True,
            comparison_available=False,
            failed_flag=True,
            comparison_metrics={
                "precision_severity": "UNAVAILABLE",
                "performance_severity": "UNAVAILABLE",
                "memory_severity": "UNAVAILABLE",
                "msa_avg_step_time_skip1": None,
            },
            issue_signals=[],
            functional_reasons=[FunctionalReason(issue_subtype="执行失败", message="pta load failed")],
        )

        self.assertEqual(categories, ["功能问题"])
        self.assertEqual(status, "EXECUTION_FAILED")

    def test_pta_functional_failure_ignores_keyword_memory_label(self) -> None:
        categories, status = _categorize_iteration(
            mutation_success=True,
            comparison_available=False,
            failed_flag=True,
            comparison_metrics={
                "precision_severity": "UNAVAILABLE",
                "performance_severity": "UNAVAILABLE",
                "memory_severity": "UNAVAILABLE",
                "msa_avg_step_time_skip1": None,
            },
            issue_signals=[
                IssueSignal(
                    category="显存问题",
                    message="RuntimeError: NPU out of memory while PTA-LOAD failed",
                    log_path="/tmp/pta.log",
                    line_number=1,
                )
            ],
            functional_reasons=[FunctionalReason(issue_subtype="PTA加载执行失败", message="PTA-LOAD 执行失败")],
        )

        self.assertEqual(categories, ["功能问题"])
        self.assertEqual(status, "EXECUTION_FAILED")

    def test_empty_step_csv_is_not_valid_metrics_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "empty.csv"
            csv_path.write_text("Iteration,Execution Time (s),NPU Memory (MB),loss\n", encoding="utf-8")
            self.assertFalse(_training_csv_has_valid_rows(csv_path))

    def test_task123_keyword_only_functional_signal_is_ignored_when_all_metrics_pass(self) -> None:
        signals = _suppress_keyword_only_functional_signals(
            task_type=1,
            signals=[
                IssueSignal(
                    category="功能问题",
                    message="Traceback: runtime error in log",
                    log_path="/tmp/runtime.log",
                    line_number=12,
                )
            ],
            comparison_available=True,
            comparison_metrics={
                "precision_severity": "PASS",
                "performance_severity": "PASS",
                "memory_severity": "PASS",
            },
            functional_reasons=[],
        )

        self.assertEqual(signals, [])

        categories, status = _categorize_iteration(
            mutation_success=True,
            comparison_available=True,
            failed_flag=False,
            comparison_metrics={
                "precision_severity": "PASS",
                "performance_severity": "PASS",
                "memory_severity": "PASS",
                "msa_avg_step_time_skip1": 1.0,
            },
            issue_signals=signals,
            functional_reasons=[],
        )

        self.assertEqual(categories, [])
        self.assertEqual(status, "PASS")

    def test_series_metric_table_keeps_pta_loss_when_compare_side_missing(self) -> None:
        record = IterationAnalysis(
            task_type=1,
            iteration=1,
            iteration_tag="iter1",
            iteration_dir="/tmp/iter1",
            failed_flag=True,
            mutation_success=True,
            comparison_available=False,
            overall_status="EXECUTION_FAILED",
            precision_severity="UNAVAILABLE",
        )
        rows = _build_metric_overview_rows(
            record,
            _task_profile(1),
            {
                "pta": {
                    "time": 2.248689889907837,
                    "loss": 11.463295936584473,
                    "memory": 8207.3330078125,
                },
                "msa": {},
            },
        )

        self.assertEqual(rows[1][0], "精度对比(末步 loss / 公共 step 最大 loss 绝对差)")
        self.assertEqual(rows[1][1], "11.46329594")
        self.assertEqual(rows[1][2], "-")
        self.assertEqual(rows[1][3], "-")
        self.assertEqual(rows[1][5], "UNAVAILABLE")

    def test_series_partial_report_uses_pta_step_csv_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "pta.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "Iteration,Execution Time (s),NPU Memory (MB),loss",
                        "1,0.0,100.0,1.0",
                        "2,4.0,150.0,1.1",
                        "3,2.0,130.0,1.2",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            record = IterationAnalysis(
                task_type=1,
                iteration=1,
                iteration_tag="iter1",
                iteration_dir=str(Path(tmpdir) / "iter1"),
                failed_flag=True,
                mutation_success=True,
                comparison_available=False,
                overall_status="EXECUTION_FAILED",
                pta_csv=str(csv_path),
            )
            profile = _task_profile(1)

            metric_snapshot = _summary_metric_snapshot(record, profile)
            self.assertEqual(metric_snapshot["pta_perf"], 3.0)
            self.assertEqual(metric_snapshot["pta_mem"], 150.0)

            lines = _partial_comparison_summary_lines(
                record,
                profile,
                "MF",
                {
                    "pta": {"loss": 1.2},
                    "msa": {},
                    "mf": {},
                },
            )
            self.assertIn("- PTA 平均 step 耗时(去首步): `3.000000` s。", lines)
            self.assertIn("- PTA 末步 loss: `1.20000000`。", lines)
            self.assertIn("- PTA 最大显存: `150.00` MB。", lines)

    def test_summary_markdown_detail_table_excludes_mf_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            assets = tmp / "assets"
            assets.mkdir()
            summary_path = tmp / "summary.md"
            record = IterationAnalysis(
                task_type=1,
                iteration=1,
                iteration_tag="iter1",
                iteration_dir=str(tmp / "iter1"),
                failed_flag=False,
                mutation_success=True,
                comparison_available=True,
                overall_status="PASS",
                performance_severity="PASS",
                precision_severity="PASS",
                memory_severity="PASS",
                compare_mode="pta_msa",
            )
            payload = {
                "task_type": 1,
                "对比对象": "MSA",
                "model_name": "demo",
                "nutnm": "-",
                "iters": 1,
                "执行总数": 1,
                "变异成功数": 1,
                "变异成功率": "100.00%",
                "PTA执行成功数": 1,
                "PTA执行成功率": "100.00%",
                "MS执行成功数": 1,
                "MS执行成功率": "100.00%",
                "有效对比轮次": 1,
                "迭代报告目录": str(tmp),
                "复现目录": str(tmp),
                "功能问题详情": [],
                "精度问题": {"数量": 0, "迭代列表": []},
                "性能问题": {"数量": 0, "迭代列表": []},
                "显存问题": {"数量": 0, "迭代列表": []},
                "验证标准": {
                    "性能": _task_profile(1)["performance_rule"],
                    "精度": _task_profile(1)["precision_rule"],
                    "显存": _task_profile(1)["memory_rule"],
                },
            }

            _write_markdown(summary_path, payload, [record], assets)
            text = summary_path.read_text(encoding="utf-8")

            self.assertIn("| 迭代 | 状态 | 总体结果 |", text)
            self.assertNotIn("| MF |", text)


if __name__ == "__main__":
    unittest.main()
