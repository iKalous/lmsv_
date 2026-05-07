#!/usr/bin/env python3
"""
Task1 自动分析规则。

如需调整报错分类关键词、性能/显存阈值、精度零误差规则，直接改本文件即可。
"""

from __future__ import annotations

TIME_COLUMN_NAMES = (
    "Execution Time (s)",
    "execution_time",
    "step_time",
)

LOSS_COLUMN_NAMES = (
    "loss",
    "Loss",
)

MEMORY_COLUMN_NAMES = (
    "NPU Memory (MB)",
    "memory_mb",
    "max_memory_mb",
)

ITERATION_COLUMN_NAMES = (
    "Iteration",
    "iteration",
    "step",
)

COMPARISON_THRESHOLDS = {
    # 甲方要求：与 PTA 对比必须零误差才算通过；默认不放宽。
    "precision_abs_tolerance": 0.0,
    # 默认阈值可改。相对 PTA 的平均 step 耗时/最大显存超出该比例即记为问题。
    "performance_relative_fail": 0.5,
    "memory_relative_fail": 0.05,
}

SEVERITY_LEVELS = {
    "performance": (
        (0.0, "PASS"),
        (0.5, "INFO"),
        (0.8, "LOW"),
        (1.0, "MEDIUM"),
        (3.0, "HIGH"),
        (float("inf"), "CRITICAL"),
    ),
    "precision": (
        (0.0, "PASS"),
        (1e-8, "TRACE"),
        (1e-6, "LOW"),
        (1e-5, "MEDIUM"),
        (1e-4, "HIGH"),
        (float("inf"), "CRITICAL"),
    ),
    "memory": (
        (0.0, "PASS"),
        (0.02, "INFO"),
        (0.05, "LOW"),
        (0.10, "MEDIUM"),
        (0.20, "HIGH"),
        (float("inf"), "CRITICAL"),
    ),
}

ISSUE_KEYWORDS = {
    "显存问题": (
        r"out of memory",
        r"\boom\b",
        r"npu out of memory",
        r"cuda out of memory",
        r"memory exhausted",
        r"memory allocation failed",
        r"malloc failed",
        r"显存不足",
        r"内存不足",
    ),
    "性能问题": (
        r"hang detected",
        r"stuck",
        r"slow step",
        r"throughput dropped",
    ),
    "功能问题": (
        r"traceback",
        r"runtimeerror",
        r"valueerror",
        r"typeerror",
        r"assertionerror",
        r"importerror",
        r"modulenotfounderror",
        r"filenotfounderror",
        r"permissionerror",
        r"keyerror",
        r"attributeerror",
        r"notimplementederror",
        r"segmentation fault",
        r"core dumped",
        r"processexitedexception",
        r"Bad CRC-32",
        r"checksum error",
        r"checksum mismatch",
        r"corrupt(?:ed)? file",
        r"\[error\]",
        r"^error:",
    ),
}

IGNORE_LINE_KEYWORDS = (
    "[COMMAND]",
    'echo "ERROR:',
    "memory_snapshot_path",
    "profile_with_memory",
    "log_memory_to_tensorboard",
    "record_memory_history",
    "Theoretical memory footprints",
)

MAX_SIGNAL_PER_FILE = 5
MAX_SIGNALS_PER_ITER = 8
MAX_ERROR_BLOCK_LINES = 40
