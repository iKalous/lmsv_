#!/usr/bin/env python3
"""
Task6 多模态整网变异分析模块
用于分析Task6的执行结果，包括精度、性能、显存对比
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class IterationMetrics:
    """单轮迭代的指标"""
    iter_num: int
    pta_loss: Optional[float] = None
    pta_memory: Optional[float] = None
    pta_time: Optional[float] = None
    msa_loss: Optional[float] = None
    msa_memory: Optional[float] = None
    msa_time: Optional[float] = None
    loss_match: bool = True
    loss_diff: float = 0.0
    memory_diff: float = 0.0
    time_diff: float = 0.0
    issues: List[str] = field(default_factory=list)
    pta_success: bool = False
    msa_success: bool = False
    pta_error: Optional[str] = None  # PTA错误信息
    msa_error: Optional[str] = None  # MSA实际错误信息


@dataclass
class AnalysisReport:
    """分析报告"""
    task_type: int = 6
    model_name: str = ""
    total_iterations: int = 0
    successful_iterations: int = 0
    pta_success_count: int = 0
    msa_success_count: int = 0
    issue_count: int = 0
    iterations: List[IterationMetrics] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


def extract_error_from_log(log_file: str) -> Optional[str]:
    """
    从日志文件中提取错误信息

    Args:
        log_file: 日志文件路径

    Returns:
        Optional[str]: 提取的错误信息，如果没有找到则返回None
    """
    if not os.path.exists(log_file):
        return None

    import re

    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception:
        return None

    # 1. 优先提取显式错误行（如 [rank2]: AssertionError: ...）
    # 匹配 [rankN]: ErrorType: message  或  ErrorType: message
    explicit_error_pattern = re.compile(
        r"^\[?rank\d+\]?:?\s*([A-Za-z][A-Za-z0-9_]*Error)\s*:\s*(.+)$",
        re.MULTILINE,
    )
    matches = explicit_error_pattern.findall(content)
    if matches:
        # 去重并保持顺序
        seen = set()
        for err_type, msg in matches:
            combined = f"{err_type}: {msg.strip()}"
            if combined not in seen:
                seen.add(combined)
                if len(combined) > 200:
                    combined = combined[:200] + "..."
                return combined

    # 2. 从 Traceback 块中提取真正的错误（取最后一个 Traceback 后的错误行）
    traceback_pattern = re.compile(
        r"Traceback \(most recent call last\):.*?^(\S+Error):\s*(.+?)$",
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    tb_matches = traceback_pattern.findall(content)
    if tb_matches:
        err_type, msg = tb_matches[-1]
        combined = f"{err_type}: {msg.strip()}"
        if len(combined) > 200:
            combined = combined[:200] + "..."
        return combined

    # 3. 匹配各种常见错误模式（整行）
    error_line_patterns = [
        re.compile(r"^.*?(ERROR:\s*.+)$", re.MULTILINE | re.IGNORECASE),
        re.compile(r"^.*?(Error:\s*.+)$", re.MULTILINE),
        re.compile(r"^.*?(Fatal.*)$", re.MULTILINE | re.IGNORECASE),
        re.compile(r"^.*?(ChildFailedError.*)$", re.MULTILINE | re.IGNORECASE),
    ]
    for pat in error_line_patterns:
        m = pat.search(content)
        if m:
            msg = m.group(1).strip()
            if len(msg) > 200:
                msg = msg[:200] + "..."
            return msg

    # 4. 查找最后一个 WARNING
    warning_pattern = re.compile(r"WARNING[:\s]+(.+)$", re.MULTILINE | re.IGNORECASE)
    warn_matches = warning_pattern.findall(content)
    if warn_matches:
        msg = warn_matches[-1].strip()
        if len(msg) > 200:
            msg = msg[:200] + "..."
        return f"Warning: {msg}"

    return None


def extract_error_from_worker_logs(msrun_log_dir: str) <-> Optional[str]:
    """
    从msrun_log/worker_*.log中提取真实的错误信息

    Args:
        msrun_log_dir: msrun_log目录路径

    Returns:
        Optional[str]: 提取的错误信息，如果没有找到则返回None
    """
    if not os.path.isdir(msrun_log_dir):
        return None

    import glob
    import re

    worker_logs = glob.glob(os.path.join(msrun_log_dir, "worker_*.log"))
    if not worker_logs:
        return None

    # 按文件大小排序，优先查看较大的日志（通常包含更多错误信息）
    worker_logs.sort(key=lambda x: os.path.getsize(x), reverse=True)

    best_error = None
    best_priority = 999

    # 优先级：RuntimeError > 其他Error > Traceback > ERROR > WARNING
    priority_map = {
        "RuntimeError": 1,
        "ValueError": 2,
        "TypeError": 2,
        "AssertionError": 2,
        "Exception": 3,
    }

    for log_file in worker_logs:
        err = extract_error_from_log(log_file)
        if err and not err.startswith("Warning:") and "Failed to load image Python extension" not in err and "error_injection_rate" not in err:
            # 获取优先级
            p = 10
            for key, val in priority_map.items():
                if key in err:
                    p = val
                    break
            if p < best_priority:
                best_priority = p
                best_error = err
                if p == 1:
                    break  # 找到RuntimeError，直接返回

    return best_error


def analyze_task6_run(output_dir: str, model_name: str = "") -> AnalysisReport:
    """
    分析Task6运行结果

    Args:
        output_dir: 输出目录路径
        model_name: 模型名称

    Returns:
        AnalysisReport: 分析报告
    """
    report = AnalysisReport(
        task_type=6,
        model_name=model_name,
    )

    if not os.path.exists(output_dir):
        return report

    # 支持传入 persist_root 或 iters/ 目录：优先在 output_dir/iters/ 下查找，
    # 找不到时回退到 output_dir 本身
    iters_dir = os.path.join(output_dir, "iters")
    search_dir = iters_dir if os.path.isdir(iters_dir) else output_dir

    # 遍历迭代目录
    iter_dirs = sorted(
        [d for d in os.listdir(search_dir) if d.startswith("iter_")],
        key=lambda x: int(x.split("_")[1])
    )

    for iter_dir in iter_dirs:
        iter_path = os.path.join(search_dir, iter_dir)
        if not os.path.isdir(iter_path):
            continue

        # 解析迭代号
        try:
            iter_num = int(iter_dir.split("_")[1])
        except (IndexError, ValueError):
            continue

        metrics = IterationMetrics(iter_num=iter_num)

        # 读取status.json（作为成功状态的主要依据）
        status_file = os.path.join(iter_path, "status.json")
        if not os.path.exists(status_file):
            # 跳过只有权重目录的占位目录（PTA失败的已归档到failed/子目录）
            continue
        with open(status_file, 'r', encoding='utf-8') as f:
            status_data = json.load(f)

        # 从status.json获取PTA/MSA成功状态（这是最准确的）
        components = status_data.get("components", {})
        pta_success_from_status = components.get("PTA_VERIFY") == "PASS"
        msa_success_from_status = components.get("MSA_VERIFY") == "PASS"

        # PTA失败的轮次不计入报告统计，确保PTA成功率为100%
        if not pta_success_from_status:
            continue

        # 读取metrics.json
        metrics_file = os.path.join(iter_path, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            pta_metrics = data.get("pta", {})
            msa_metrics = data.get("msa", {})
            analysis = data.get("analysis", {})

            metrics.pta_loss = pta_metrics.get("loss")
            metrics.pta_memory = pta_metrics.get("memory")
            metrics.pta_time = pta_metrics.get("time")
            metrics.msa_loss = msa_metrics.get("loss")
            metrics.msa_memory = msa_metrics.get("memory")
            metrics.msa_time = msa_metrics.get("time")
            metrics.loss_match = analysis.get("loss_match", True)
            # 只有当MSA实际执行了（有msa_metrics数据）时才使用diff值，否则设为None
            has_msa_data = bool(msa_metrics)
            metrics.loss_diff = analysis.get("loss_diff") if has_msa_data else None
            metrics.memory_diff = analysis.get("memory_diff") if has_msa_data else None
            metrics.time_diff = analysis.get("time_diff") if has_msa_data else None
            metrics.issues = analysis.get("issues", [])

            # 优先使用status.json中的状态
            metrics.pta_success = pta_success_from_status
            metrics.msa_success = msa_success_from_status

            # 如果MSA失败，提取MSA错误日志中的实际错误信息
            if not metrics.msa_success:
                msa_log_file = os.path.join(iter_path, "runtime_logs", f"msa_verify_iter{iter_num}.log")
                msa_error = extract_error_from_log(msa_log_file)
                # 如果主日志只返回Warning/通用信息，尝试从worker日志中提取真实错误
                if not msa_error or msa_error.startswith("Warning:") or "No loss found" in msa_error:
                    msrun_log_dir = os.path.join(iter_path, "msrun_log")
                    worker_error = extract_error_from_worker_logs(msrun_log_dir)
                    if worker_error:
                        msa_error = worker_error
                if msa_error:
                    metrics.msa_error = msa_error
                    # 替换issues中的通用信息为实际错误
                    if metrics.issues and ("MSA未产生loss" in str(metrics.issues) or "MSA verification failed" in str(metrics.issues)):
                        metrics.issues = [f"MSA执行失败: {msa_error}"]
                    elif not metrics.issues:
                        metrics.issues = [f"MSA执行失败: {msa_error}"]

            # 如果PTA失败，也提取PTA错误
            if not metrics.pta_success:
                pta_log_file = os.path.join(iter_path, "runtime_logs", f"pta_verify_iter{iter_num}.log")
                pta_error = extract_error_from_log(pta_log_file)
                if pta_error:
                    metrics.pta_error = pta_error
                    if metrics.issues and ("PTA未产生loss" in str(metrics.issues) or "PTA verification failed" in str(metrics.issues)):
                        metrics.issues = [f"PTA执行失败: {pta_error}"]
                    elif not metrics.issues:
                        metrics.issues = [f"PTA执行失败: {pta_error}"]

        report.iterations.append(metrics)

    # 计算统计信息（按照用户定义的规则）
    # 总迭代次数 = 有效突变轮次数（PTA成功的轮次）
    report.total_iterations = len(report.iterations)

    # PTA成功次数 = 有效突变次数
    report.pta_success_count = sum(1 for m in report.iterations if m.pta_success)

    # MSA成功次数 = PTA成功且MSA也成功的轮次
    report.msa_success_count = sum(1 for m in report.iterations if m.pta_success and m.msa_success)

    # 成功迭代次数 = PTA成功的轮次
    report.successful_iterations = report.pta_success_count

    # 发现问题数 = MSA不成功执行的次数（有issues的轮次）
    report.issue_count = sum(1 for m in report.iterations if m.issues)

    # 从final_report.json读取总突变次数（如果存在）
    final_report_path = os.path.join(output_dir, "final_report.json")
    total_mutations = report.total_iterations  # 默认为有效突变次数
    if os.path.exists(final_report_path):
        try:
            with open(final_report_path, 'r', encoding='utf-8') as f:
                final_data = json.load(f)
            stats = final_data.get("statistics", {})
            total_mutations = stats.get("total_mutations", report.total_iterations)
        except Exception:
            pass

    # 计算成功率（按照用户定义）
    # PTA成功率 = 有效突变次数 / 总突变次数
    pta_success_rate = report.pta_success_count / total_mutations if total_mutations > 0 else 0

    # MSA成功率 = PTA成功且MSA也成功的轮次 / PTA成功的轮次
    msa_success_rate = report.msa_success_count / report.pta_success_count if report.pta_success_count > 0 else 0

    # 问题发现率 = 发现问题数 / 总迭代次数
    issue_rate = report.issue_count / report.total_iterations if report.total_iterations > 0 else 0

    # 生成摘要
    report.summary = {
        "total_iterations": report.total_iterations,  # 总迭代次数 = 有效突变轮次数
        "successful_iterations": report.successful_iterations,  # 成功迭代次数 = PTA成功的轮次
        "issue_count": report.issue_count,  # 发现问题数 = MSA执行失败次数
        "pta_success_count": report.pta_success_count,  # PTA成功次数
        "msa_success_count": report.msa_success_count,  # MSA成功次数
        "total_mutations": total_mutations,  # 总突变次数
        "pta_success_rate": pta_success_rate,  # PTA成功率
        "msa_success_rate": msa_success_rate,  # MSA成功率
        "issue_rate": issue_rate,  # 问题发现率
        "avg_loss_diff": sum(m.loss_diff for m in report.iterations if m.pta_success and m.msa_success) / sum(1 for m in report.iterations if m.pta_success and m.msa_success) if any(m.pta_success and m.msa_success for m in report.iterations) else 0,
        "avg_memory_diff": sum(m.memory_diff for m in report.iterations if m.pta_success and m.msa_success) / sum(1 for m in report.iterations if m.pta_success and m.msa_success) if any(m.pta_success and m.msa_success for m in report.iterations) else 0,
        "avg_time_diff": sum(m.time_diff for m in report.iterations if m.pta_success and m.msa_success) / sum(1 for m in report.iterations if m.pta_success and m.msa_success) if any(m.pta_success and m.msa_success for m in report.iterations) else 0,
    }

    return report




def _get_accuracy_status(m: IterationMetrics) -> str:
    """精度状态: 返回具体的Loss差异值"""
    if m.loss_diff is not None:
        return f"{m.loss_diff:.6f}"
    if m.pta_loss is None or m.msa_loss is None:
        return "N/A"
    return "N/A"


def _get_memory_status(m: IterationMetrics) -> str:
    """显存状态: 返回具体的显存差异值(MB)"""
    if m.memory_diff is not None:
        return f"{m.memory_diff:.2f}"
    if m.pta_memory is None or m.msa_memory is None:
        return "N/A"
    return "N/A"


def _get_perf_status(m: IterationMetrics) -> str:
    """性能状态: 返回具体的时间差异值(ms)"""
    if m.time_diff is not None:
        return f"{m.time_diff:.2f}"
    if m.pta_time is None or m.msa_time is None:
        return "N/A"
    return "N/A"

def generate_markdown_report(report: AnalysisReport) -> str:
    """生成Markdown格式的报告"""
    summary = report.summary
    lines = [
        "# Task6 多模态整网变异分析报告",
        "",
        f"**模型**: {report.model_name}",
        f"**总迭代次数**: {report.total_iterations}",
        f"**成功迭代次数**: {report.successful_iterations}",
        f"**发现问题数**: {report.issue_count}",
        "",
        "## 统计摘要",
        "",
        f"- **PTA成功率**: {summary.get('pta_success_rate', 0)*100:.1f}% ({report.pta_success_count}/{summary.get('total_mutations', report.total_iterations)})",
        f"- **MSA成功率**: {summary.get('msa_success_rate', 0)*100:.1f}% ({report.msa_success_count}/{report.pta_success_count})",
        f"- **问题发现率**: {summary.get('issue_rate', 0)*100:.1f}% ({report.issue_count}/{report.total_iterations})",
        f"- **平均Loss差异**: {summary.get('avg_loss_diff', 0):.6f}",
        f"- **平均显存差异**: {summary.get('avg_memory_diff', 0):.2f} MB",
        f"- **平均时间差异**: {summary.get('avg_time_diff', 0):.2f} ms",
        "",
        "## 迭代详情",
        "",
        "| 轮次 | PTA状态 | PTA Loss | PTA显存(MB) | PTA时间(ms) | MSA状态 | MSA Loss | MSA显存(MB) | MSA时间(ms) | Loss差异 | 显存差异(MB) | 时间差异(ms) | 问题 |",
        "|------|---------|----------|-------------|-------------|---------|----------|-------------|-------------|----------|--------------|--------------|------|",
    ]

    for m in report.iterations:
        pta_status = "pass" if m.pta_success else "fail"
        msa_status = "pass" if m.msa_success else "fail"
        pta_loss = f"{m.pta_loss:.6f}" if m.pta_loss is not None else "N/A"
        pta_mem = f"{m.pta_memory:.2f}" if m.pta_memory is not None else "N/A"
        pta_time = f"{m.pta_time:.2f}" if m.pta_time is not None else "N/A"
        msa_loss = f"{m.msa_loss:.6f}" if m.msa_loss is not None else "N/A"
        msa_mem = f"{m.msa_memory:.2f}" if m.msa_memory is not None else "N/A"
        msa_time = f"{m.msa_time:.2f}" if m.msa_time is not None else "N/A"
        match = "pass" if m.loss_match else "fail"
        mem_diff = f"{m.memory_diff:.2f}" if m.memory_diff is not None else "N/A"
        time_diff = f"{m.time_diff:.2f}" if m.time_diff is not None else "N/A"
        issues = "; ".join(m.issues) if m.issues else "无"
        acc_status = _get_accuracy_status(m)
        mem_status = _get_memory_status(m)
        perf_status = _get_perf_status(m)
        lines.append(
            f"| {m.iter_num} | {pta_status} | {pta_loss} | {pta_mem} | {pta_time} | "
            f"{msa_status} | {msa_loss} | {msa_mem} | {msa_time} | {acc_status} | {mem_status} | {perf_status} | {issues} |"
        )

    lines.extend([
        "",
        "## 问题汇总",
        "",
    ])

    # 汇总问题（去重，每个轮次只显示一次）
    all_issues = []
    for m in report.iterations:
        for issue in m.issues:
            all_issues.append(f"轮次{m.iter_num}: {issue}")

    if all_issues:
        for issue in all_issues:
            lines.append(f"- {issue}")
    else:
        lines.append("未发现明显问题。")

    return "\n".join(lines)


def generate_json_report(report: AnalysisReport) -> Dict[str, Any]:
    """生成JSON格式的报告"""
    return {
        "task_type": report.task_type,
        "model_name": report.model_name,
        "total_iterations": report.total_iterations,
        "successful_iterations": report.successful_iterations,
        "pta_success_count": report.pta_success_count,
        "msa_success_count": report.msa_success_count,
        "issue_count": report.issue_count,
        "summary": report.summary,
        "iterations": [
            {
                "iter_num": m.iter_num,
                "pta_loss": m.pta_loss,
                "pta_memory": m.pta_memory,
                "pta_time": m.pta_time,
                "msa_loss": m.msa_loss,
                "msa_memory": m.msa_memory,
                "msa_time": m.msa_time,
                "loss_match": m.loss_match,
                "loss_diff": m.loss_diff,
                "memory_diff": m.memory_diff,
                "time_diff": m.time_diff,
                "issues": m.issues,
                "pta_success": m.pta_success,
                "msa_success": m.msa_success,
                "pta_error": m.pta_error,
                "msa_error": m.msa_error,
            }
            for m in report.iterations
        ]
    }


def generate_html_report(report: AnalysisReport) -> str:
    """生成HTML格式的报告"""
    from datetime import datetime

    # 计算统计数据
    summary = report.summary
    pta_success_rate = summary.get('pta_success_rate', 0) * 100
    msa_success_rate = summary.get('msa_success_rate', 0) * 100
    issue_rate = summary.get('issue_rate', 0) * 100
    avg_loss_diff = summary.get('avg_loss_diff', 0)
    avg_memory_diff = summary.get('avg_memory_diff', 0)
    avg_time_diff = summary.get('avg_time_diff', 0)
    total_mutations = summary.get('total_mutations', report.total_iterations)

    # 生成表格行
    rows = []
    for m in report.iterations:
        pta_loss = f"{m.pta_loss:.6f}" if m.pta_loss is not None else "N/A"
        pta_mem = f"{m.pta_memory:.2f}" if m.pta_memory is not None else "N/A"
        pta_time = f"{m.pta_time:.2f}" if m.pta_time is not None else "N/A"
        msa_loss = f"{m.msa_loss:.6f}" if m.msa_loss is not None else "N/A"
        msa_mem = f"{m.msa_memory:.2f}" if m.msa_memory is not None else "N/A"
        msa_time = f"{m.msa_time:.2f}" if m.msa_time is not None else "N/A"
        loss_diff = f"{m.loss_diff:.6f}" if m.loss_diff is not None else "N/A"
        mem_diff = f"{m.memory_diff:.2f}" if m.memory_diff is not None else "N/A"
        time_diff = f"{m.time_diff:.2f}" if m.time_diff is not None else "N/A"
        status = "pass" if m.pta_success and m.msa_success else "fail"
        status_color = "#16a34a" if m.pta_success and m.msa_success else "#dc2626"
        issues = "; ".join(m.issues) if m.issues else "无"

        acc_status_html = _get_accuracy_status(m)
        mem_status_html = _get_memory_status(m)
        perf_status_html = _get_perf_status(m)
        rows.append(f"""
            <tr>
                <td>{m.iter_num}</td>
                <td style="color: {status_color}; font-weight: bold;">{status}</td>
                <td>{pta_loss}</td>
                <td>{pta_mem}</td>
                <td>{pta_time}</td>
                <td>{msa_loss}</td>
                <td>{msa_mem}</td>
                <td>{msa_time}</td>
                <td>{acc_status_html}</td>
                <td>{mem_status_html}</td>
                <td>{perf_status_html}</td>
                <td style="max-width: 400px; word-break: break-all;">{issues}</td>
            </tr>
        """)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Task6 多模态整网变异分析报告 - {report.model_name}</title>
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
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
            background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
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
        h1 {{
            margin: 10px 0 8px;
            font-size: 32px;
            color: var(--text);
        }}
        .subtitle {{
            color: var(--sub);
            line-height: 1.7;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-top: 24px;
        }}
        .card {{
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 18px 20px;
            box-shadow: 0 10px 30px rgba(15,23,42,0.05);
        }}
        .card-label {{
            color: var(--sub);
            font-size: 14px;
        }}
        .card-value {{
            font-size: 32px;
            font-weight: 800;
            color: var(--accent);
            margin-top: 8px;
        }}
        .section {{
            margin-top: 24px;
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 10px 30px rgba(15,23,42,0.05);
        }}
        .section-title {{
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 16px;
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
        }}
        th {{
            color: var(--sub);
            font-weight: 600;
            background: #f1f5f9;
        }}
        tr:hover {{
            background: #f8fafc;
        }}
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 12px;
        }}
        .stat-item {{
            display: flex;
            justify-content: space-between;
            padding: 12px;
            background: #f8fafc;
            border-radius: 8px;
        }}
        .stat-label {{
            color: var(--sub);
        }}
        .stat-value {{
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="wrap">
        <div class="hero">
            <div style="font-size:14px;color:#0f766e;font-weight:700;letter-spacing:0.08em">TASK6 MULTIMODAL ANALYSIS</div>
            <h1>Task6 多模态整网变异分析报告</h1>
            <div class="subtitle">
                模型: <strong>{report.model_name}</strong> |
                生成时间: <strong>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</strong>
            </div>

            <div class="grid">
                <div class="card">
                    <div class="card-label">总迭代次数</div>
                    <div class="card-value">{report.total_iterations}</div>
                </div>
                <div class="card">
                    <div class="card-label">成功迭代数</div>
                    <div class="card-value">{report.successful_iterations}</div>
                </div>
                <div class="card">
                    <div class="card-label">发现问题数</div>
                    <div class="card-value">{report.issue_count}</div>
                </div>
                <div class="card">
                    <div class="card-label">PTA成功率</div>
                    <div class="card-value">{pta_success_rate:.1f}%</div>
                </div>
                <div class="card">
                    <div class="card-label">MSA成功率</div>
                    <div class="card-value">{msa_success_rate:.1f}%</div>
                </div>
                <div class="card">
                    <div class="card-label">问题发现率</div>
                    <div class="card-value">{issue_rate:.1f}%</div>
                </div>
            </div>
        </div>

        <div class="section">
            <div class="section-title">统计摘要</div>
            <div class="summary-stats">
                <div class="stat-item">
                    <span class="stat-label">总突变次数</span>
                    <span class="stat-value">{total_mutations}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">有效突变次数</span>
                    <span class="stat-value">{report.total_iterations}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">PTA成功次数</span>
                    <span class="stat-value">{report.pta_success_count}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">MSA成功次数</span>
                    <span class="stat-value">{report.msa_success_count}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">平均Loss差异</span>
                    <span class="stat-value">{avg_loss_diff:.6f}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">平均显存差异</span>
                    <span class="stat-value">{avg_memory_diff:.2f} MB</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">平均时间差异</span>
                    <span class="stat-value">{avg_time_diff:.2f} ms</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">PTA成功率</span>
                    <span class="stat-value">{pta_success_rate:.1f}%</span>
                </div>
            </div>
        </div>

        <div class="section">
            <div class="section-title">迭代详情</div>
            <table>
                <thead>
                    <tr>
                        <th>轮次</th>
                        <th>状态</th>
                        <th>PTA Loss</th>
                        <th>PTA显存(MB)</th>
                        <th>PTA时间(ms)</th>
                        <th>MSA Loss</th>
                        <th>MSA显存(MB)</th>
                        <th>MSA时间(ms)</th>
                        <th>Loss差异</th>
                        <th>显存差异(MB)</th>
                        <th>时间差异(ms)</th>
                        <th>问题</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>"""
    return html


def save_report(report: AnalysisReport, output_dir: str):
    """保存报告到文件（目录结构与Task1-5保持一致）

    output_dir: analysis目录路径（如 persist_root/analysis）
    生成的文件:
      - analysis/summary.md
      - analysis/report.html
      - data/summary.json
      - data/iteration_metrics.csv
      - assets/ (目录，预留)
    """
    analysis_dir = Path(output_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    persist_root = analysis_dir.parent
    data_dir = persist_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = persist_root / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # 保存Markdown报告 -> analysis/summary.md
    md_content = generate_markdown_report(report)
    md_path = analysis_dir / "summary.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

    # 保存HTML报告 -> analysis/report.html
    html_content = generate_html_report(report)
    html_path = analysis_dir / "report.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    # 保存JSON报告 -> data/summary.json
    json_content = generate_json_report(report)
    json_path = data_dir / "summary.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_content, f, ensure_ascii=False, indent=2)

    # 保存CSV -> data/iteration_metrics.csv
    _write_iteration_csv(data_dir / "iteration_metrics.csv", report)

    # 生成issue_groups.json（与Task1保持一致）
    issue_groups = _build_issue_groups(report)
    issue_json_path = data_dir / "issue_groups.json"
    with open(issue_json_path, 'w', encoding='utf-8') as f:
        json.dump(issue_groups, f, ensure_ascii=False, indent=2)

    # 生成README.md（与Task1保持一致）
    readme_path = persist_root / "README.md"
    _write_readme(readme_path, persist_root, report)

    # 生成每轮迭代的report.md（与Task1保持一致，作为task6.py的fallback）
    _write_iteration_reports(persist_root, report)

    # 创建assets占位SVG文件（与Task1保持一致）
    _create_placeholder_assets(assets_dir)

    print(f"报告已保存:")
    print(f"  Markdown: {md_path}")
    print(f"  HTML: {html_path}")
    print(f"  JSON: {json_path}")
    print(f"  CSV: {data_dir / 'iteration_metrics.csv'}")
    print(f"  Issue Groups: {issue_json_path}")
    print(f"  README: {readme_path}")


def _write_iteration_csv(path: Path, report: AnalysisReport):
    """生成迭代指标CSV（与Task1-5保持一致）"""
    import csv
    fieldnames = [
        "iteration", "pta_success", "msa_success",
        "pta_loss", "pta_memory_mb", "pta_time_ms",
        "msa_loss", "msa_memory_mb", "msa_time_ms",
        "loss_match", "loss_diff", "memory_diff", "time_diff",
        "issues",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for m in report.iterations:
            writer.writerow({
                "iteration": m.iter_num,
                "pta_success": "PASS" if m.pta_success else "FAIL",
                "msa_success": "PASS" if m.msa_success else "FAIL",
                "pta_loss": m.pta_loss if m.pta_loss is not None else "",
                "pta_memory_mb": m.pta_memory if m.pta_memory is not None else "",
                "pta_time_ms": m.pta_time if m.pta_time is not None else "",
                "msa_loss": m.msa_loss if m.msa_loss is not None else "",
                "msa_memory_mb": m.msa_memory if m.msa_memory is not None else "",
                "msa_time_ms": m.msa_time if m.msa_time is not None else "",
                "loss_match": "PASS" if m.loss_match else "FAIL",
                "loss_diff": m.loss_diff,
                "memory_diff": m.memory_diff,
                "time_diff": m.time_diff,
                "issues": "; ".join(m.issues) if m.issues else "",
            })




def _build_issue_groups(report: AnalysisReport) -> dict:
    """构建问题分组（与Task1的issue_groups.json格式一致）"""
    groups = {}
    for m in report.iterations:
        for issue in m.issues:
            if issue not in groups:
                groups[issue] = {
                    "count": 0,
                    "iterations": [],
                    "severity": "warning",
                }
            groups[issue]["count"] += 1
            groups[issue]["iterations"].append(m.iter_num)
    return groups


def _write_readme(readme_path: Path, persist_root: Path, report: AnalysisReport):
    """生成README.md（与Task1保持一致）"""
    lines = [
        "# output 目录说明",
        "",
        "本目录本身就是运行材料与复现入口。",
        "",
        "## 目录结构",
        "",
        "- `iters/iter_x/report.md`: 当前轮次的简版分析报告。",
        "- `iters/iter_x/status.json`: 当前轮次组件状态。",
        "- `iters/iter_x/scripts/`: 本轮相关脚本。",
        "- `iters/iter_x/weights/`: PTA/MSA产出的权重。",
        "- `iters/iter_x/mutation_inputs/`: 本轮变异输入 JSON。",
        "- `iters/iter_x/runtime_logs/`: 本轮运行日志。",
        "- `iters/iter_x/msrun_log/`: 本轮 msrun 日志快照。",
        "",
        "## 使用方式",
        "",
        "- 先看 `iters/iter_x/report.md`，快速确认本轮状态、指标和 PTA/MSA 对比结果。",
        "- 再结合 `status.json`、`scripts/`、`weights/` 和 `mutation_inputs/` 做手工复现。",
        "",
        f"当前输出目录: `{persist_root}`",
        f"当前模型: `{report.model_name}`",
        "",
    ]
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_iteration_reports(persist_root: Path, report: AnalysisReport):
    """为每轮迭代生成/更新report.md（作为task6.py的fallback）"""
    iters_dir = persist_root / "iters"
    if not iters_dir.exists():
        return
    for m in report.iterations:
        iter_dir = iters_dir / f"iter_{m.iter_num}"
        if not iter_dir.exists():
            continue
        report_path = iter_dir / "report.md"
        if report_path.exists():
            continue
        lines = [
            f"# Iteration {m.iter_num} Report",
            "",
            f"**Model**: {report.model_name}",
            f"**Iteration**: {m.iter_num}",
            f"**Status**: {'PASS' if m.pta_success else 'FAIL'}",
            "",
            "## Components",
            "",
            f"- PTA_VERIFY: {'PASS' if m.pta_success else 'FAIL'}",
            f"- MSA_VERIFY: {'PASS' if m.msa_success else 'FAIL'}",
            "",
            "## Metrics",
            "",
            "### PTA",
            f"- Loss: {m.pta_loss if m.pta_loss is not None else 'N/A'}",
            f"- Memory: {m.pta_memory if m.pta_memory is not None else 'N/A'} MB",
            f"- Time: {m.pta_time if m.pta_time is not None else 'N/A'} ms",
            "",
            "### MSA",
            f"- Loss: {m.msa_loss if m.msa_loss is not None else 'N/A'}",
            f"- Memory: {m.msa_memory if m.msa_memory is not None else 'N/A'} MB",
            f"- Time: {m.msa_time if m.msa_time is not None else 'N/A'} ms",
            "",
            "## Analysis",
            "",
        ]
        if m.issues:
            for issue in m.issues:
                lines.append(f"- {issue}")
        else:
            lines.append("- No issues detected")
        lines.append("")
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _create_placeholder_assets(assets_dir: Path):
    """创建占位SVG文件（与Task1保持一致）"""
    assets_dir.mkdir(parents=True, exist_ok=True)
    svg_t = """\u003c?xml version="1.0" encoding="UTF-8"?\u003e
\u003csvg xmlns="http://www.w3.org/2000/svg" width="400" height="200"\u003e
  \u003ctext x="50%" y="50%" text-anchor="middle" dominant-baseline="middle"\u003e
    Placeholder for {name}
  \u003c/text\u003e
\u003c/svg\u003e"""
    for name in ["loss_delta.svg", "memory_delta.svg", "performance_delta.svg"]:
        path = assets_dir / name
        if not path.exists():
            path.write_text(svg_t.format(name=name.replace('.svg', '')), encoding="utf-8")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python task6_result.py <persist_root> [model_name]")
        print("  persist_root: 实验输出根目录（包含 iters/、analysis/、data/ 等子目录）")
        sys.exit(1)

    persist_root = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else ""

    report = analyze_task6_run(persist_root, model_name)
    analysis_dir = os.path.join(persist_root, "analysis")
    save_report(report, analysis_dir)
