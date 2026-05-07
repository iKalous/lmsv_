#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from copy import deepcopy
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

import utils
from utils.analyze.manual import (
    output_has_replayable_analysis_data,
    regenerate_output_analysis,
)


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
EXAMPLE_CONFIG_PATH = ROOT / "config.json.example"
OUTPUT_ROOT = ROOT / "output"
DO_SCRIPT = ROOT / "do.py"
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
LOG_BUFFER_LIMIT = 5000
OUTPUT_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
CPU_TOTAL_RE = re.compile(r"^cpu\s+(.+)$", re.MULTILINE)
STOP_TERM_TIMEOUT = 5.0
STOP_KILL_TIMEOUT = 2.0


def _handle_sigint(_signum, _frame) -> None:
    print("\n[webui] 正在停止...", flush=True)
    raise SystemExit(130)


signal.signal(signal.SIGINT, _handle_sigint)

TASK_META = {
    1: {
        "label": "整网泛化变异测试",
        "tagline": "针对单模型整网训练一致性与变异验证。",
        "accent": "#d97706",
    },
    2: {
        "label": "模块内组件泛化测试",
        "tagline": "按子模块维度组合模型配置并执行模块内验证。",
        "accent": "#0f766e",
    },
    3: {
        "label": "模块间泛化组合变异测试",
        "tagline": "跨模型组合变异，验证 PTA / MSA / MF 结果一致性。",
        "accent": "#2563eb",
    },
    4: {
      "label": "【多模态模型】模块间泛化组合变异测试",
      "tagline": "跨模型组合变异，验证 PTA / MSA 多模态模型结果一致性。",
      "accent": "#f97316",
    },
    5: {
      "label": "【多模态模型】模块内组件变异测试",
      "tagline": "跨模型组合变异，验证 PTA / MSA 多模态模型结果一致性。",
      "accent": "#8b5cf6",
    },
    6: {
      "label": "【多模态模型】整网泛化变异测试",
      "tagline": "针对多模态整网训练/推理链路执行 PTA / MSA 一致性验证。",
      "accent": "#dc2626",
    },
}

FORM_SCHEMA = {
    "global": [
        {"key": "PTA_NAME", "label": "PTA conda 环境", "type": "text", "placeholder": "mindspeed"},
        {"key": "MSA_NAME", "label": "MSA conda 环境", "type": "text", "placeholder": "msadapter"},
      {"key": "MF_NAME", "label": "MF conda 环境", "type": "text", "placeholder": "mindf_py311"},
        {"key": "PTA_PATH", "label": "PTA 代码路径", "type": "text", "placeholder": "<YOUR_PTA_PATH>"},
        {"key": "MSA_PATH", "label": "MSA 代码路径", "type": "text", "placeholder": "<YOUR_MSA_PATH>"},
        {"key": "SAVE_ABNORMAL_WEIGHTS", "label": "保存异常迭代权重", "type": "checkbox"},
        {
            "key": "CLUSTER",
            "label": "旧版 Task1/2/3 CLUSTER 配置",
            "type": "cluster",
            "help": "仅保留旧版 slave 模式兼容；新配置请在任务参数里使用 MULTI_NODE。",
        },
    ],
    "tasks": {
        "1": [
            {"key": "MODEL_NAME", "label": "模型名称", "type": "text", "placeholder": "qwen2"},
            {"key": "TOTAL_ITER", "label": "总迭代数", "type": "number", "min": 1},
            {"key": "PTA_MAX_RUNTIME", "label": "PTA 最大运行时间(秒)", "type": "number", "min": 1, "advanced": True},
            {"key": "MSA_MAX_RUNTIME", "label": "MSA 最大运行时间(秒)", "type": "number", "min": 1, "advanced": True},
            {"key": "LOG_INIT_WAIT", "label": "MSA 日志初始化等待(秒)", "type": "number", "min": 1, "advanced": True},
            {"key": "LOG_STABLE_THRESHOLD", "label": "MSA 日志稳定阈值(秒)", "type": "number", "min": 1, "advanced": True},
            {"key": "COMPARE_MODE", "label": "对比模式", "type": "select", "options": ["pta_msa", "pta_mf"], "placeholder": "pta_msa"},
          {"key": "ENABLE_MF_WEIGHT_LOAD", "label": "Task1 启用 MF 权重加载", "type": "checkbox", "advanced": True},
            {"key": "BASE_SEED", "label": "基础随机种子", "type": "number", "min": 0},
            {"key": "MUTNM", "label": "每轮变异参数数量", "type": "number", "min": 1},
            {"key": "SAVE_STEPS", "label": "SAVE 模式训练轮数", "type": "number", "min": 1, "advanced": True},
            {"key": "LOAD_STEPS", "label": "LOAD 模式训练轮数", "type": "number", "min": 1},
            {
                "key": "MULTI_NODE",
                "label": "Task1 多机配置",
                "type": "task45_multinode",
                "help": "启用后将按主从多机执行 pta_load/msa_load；从节点通过 ssh（可选 docker exec）运行。",
            },
        ],
        "2": [
            {
                "key": "MODELS",
                "label": "模型列表",
                "type": "list_text",
                "placeholder": "qwen2,qwen2,qwen2",
                "help": "逗号分隔，必须与 SUBMODULES 一一对应。",
            },
            {"key": "TOTAL_ITER", "label": "总迭代数", "type": "number", "min": 1},
            {"key": "PTA_MAX_RUNTIME", "label": "PTA 最大运行时间(秒)", "type": "number", "min": 1, "advanced": True},
            {"key": "MSA_MAX_RUNTIME", "label": "MSA 最大运行时间(秒)", "type": "number", "min": 1, "advanced": True},
            {"key": "LOG_INIT_WAIT", "label": "MSA 日志初始化等待(秒)", "type": "number", "min": 1, "advanced": True},
            {"key": "LOG_STABLE_THRESHOLD", "label": "MSA 日志稳定阈值(秒)", "type": "number", "min": 1, "advanced": True},
            {"key": "BASE_SEED", "label": "基础随机种子", "type": "number", "min": 0},
            {
                "key": "SUBMODULES",
                "label": "子模块列表",
                "type": "list_int",
                "placeholder": "3,4,5",
                "help": "逗号分隔，取值范围 0~10，顺序与 MODELS 对齐。",
            },
            {"key": "MUTNM", "label": "每轮变异参数数量", "type": "number", "min": 1},
            {"key": "SAVE_STEPS", "label": "SAVE 模式训练步数", "type": "number", "min": 1, "advanced": True},
            {"key": "LOAD_STEPS", "label": "LOAD 模式训练步数", "type": "number", "min": 1},
            {"key": "COMPARE_MODE", "label": "对比模式", "type": "select", "options": ["pta_msa", "pta_mf"], "placeholder": "pta_msa"},
            {"key": "MF_ARGS_PATH", "label": "MF 参数模板路径", "type": "text", "placeholder": "assets/runtime/mf_templates/basic.yaml"},
            {"key": "ENABLE_MF_WEIGHT_LOAD", "label": "Task2 启用 MF 权重加载", "type": "checkbox"},
            {
                "key": "MULTI_NODE",
                "label": "Task2 多机配置",
                "type": "task45_multinode",
                "help": "启用后将按主从多机执行 pta_load/msa_load；从节点通过 ssh（可选 docker exec）运行。",
            },
        ],
        "3": [
            {
                "key": "MODELS",
                "label": "模型列表",
                "type": "list_text",
                "placeholder": "qwen2,glm4",
            },
            {"key": "TOTAL_ITER", "label": "变异轮次", "type": "number", "min": 1},
            {"key": "PTA_MAX_RUNTIME", "label": "PTA 最大运行时间(秒)", "type": "number", "min": 1, "advanced": True},
            {"key": "MSA_MAX_RUNTIME", "label": "MSA 最大运行时间(秒)", "type": "number", "min": 1, "advanced": True},
            {"key": "LOG_INIT_WAIT", "label": "MSA 日志初始化等待(秒)", "type": "number", "min": 1, "advanced": True},
            {"key": "LOG_STABLE_THRESHOLD", "label": "MSA 日志稳定阈值(秒)", "type": "number", "min": 1, "advanced": True},
            {"key": "MAX_MUTATION_WAIT", "label": "变异产物等待(秒)", "type": "number", "min": 1, "advanced": True},
            {"key": "BASE_SEED", "label": "基础随机种子", "type": "number", "min": 0},
            {"key": "MUTNM", "label": "每轮变异参数数量", "type": "number", "min": 1},
            {"key": "SAVE_STEPS", "label": "SAVE 模式训练步数", "type": "number", "min": 1, "advanced": True},
            {"key": "LOAD_STEPS", "label": "LOAD 模式训练步数", "type": "number", "min": 1},
            {"key": "COMPARE_MODE", "label": "对比模式", "type": "select", "options": ["pta_msa", "pta_mf"], "placeholder": "pta_msa"},
            {
                "key": "MULTI_NODE",
                "label": "Task3 多机配置",
                "type": "task45_multinode",
                "help": "启用后将按主从多机执行 pta_load/msa_load；从节点通过 ssh（可选 docker exec）运行。",
            },
        ],
        "4": [
            {"key": "TOTAL_ITER", "label": "变异轮次", "type": "number", "min": 1},
            {"key": "COMPARE_MODE", "label": "对比模式", "type": "select", "options": ["pta_msa"], "placeholder": "pta_msa"},
            {"key": "SAVE_STEPS", "label": "SAVE 模式训练步数", "type": "number", "min": 1},
            {"key": "RUN_STEPS", "label": "RUN 模式训练步数", "type": "number", "min": 1},
            {
                "key": "MULTI_NODE",
                "label": "Task4 多机配置",
                "type": "task45_multinode",
                "help": "启用后将按主从多机执行 pta_load/msa_load；从节点通过 ssh（可选 docker exec）运行。",
            },
            {"key": "PTA_MAX_RUNTIME", "label": "PTA 最大运行时间(秒)", "type": "number", "min": 1, "advanced": True},
            {"key": "MSA_MAX_RUNTIME", "label": "MSA 最大运行时间(秒)", "type": "number", "min": 1, "advanced": True},
            {"key": "LOG_INIT_WAIT", "label": "MSA 日志初始化等待(秒)", "type": "number", "min": 1, "advanced": True},
            {"key": "LOG_STABLE_THRESHOLD", "label": "MSA 日志稳定阈值(秒)", "type": "number", "min": 1, "advanced": True},
            {"key": "ENABLE_MF_WEIGHT_LOAD", "label": "Task4 启用 MF 权重加载", "type": "checkbox", "advanced": True},
        ],
        "5": [
            {"key": "TOTAL_ITER", "label": "变异轮次", "type": "number", "min": 1},
            {"key": "COMPARE_MODE", "label": "对比模式", "type": "select", "options": ["pta_msa"], "placeholder": "pta_msa"},
            {"key": "SAVE_STEPS", "label": "SAVE 模式训练步数", "type": "number", "min": 1},
            {"key": "RUN_STEPS", "label": "RUN 模式训练步数", "type": "number", "min": 1},
            {"key": "MUTATE_STEPS", "label": "变异步数", "type": "number", "min": 1},
            {
                "key": "MODULE_TYPE",
                "label": "模块类型",
                "type": "select",
                "options": ["all", "text_decoder", "image_encoder"],
                "placeholder": "all",
                "help": "all 为不过滤；其余按多模态模块类型筛选。",
            },
            {
                "key": "MULTI_NODE",
                "label": "Task5 多机配置",
                "type": "task45_multinode",
                "help": "启用后将按主从多机执行 pta_load/msa_load；从节点通过 ssh（可选 docker exec）运行。",
            },
            {"key": "PTA_MAX_RUNTIME", "label": "PTA 最大运行时间(秒)", "type": "number", "min": 1, "advanced": True},
            {"key": "MSA_MAX_RUNTIME", "label": "MSA 最大运行时间(秒)", "type": "number", "min": 1, "advanced": True},
            {"key": "LOG_INIT_WAIT", "label": "MSA 日志初始化等待(秒)", "type": "number", "min": 1, "advanced": True},
            {"key": "LOG_STABLE_THRESHOLD", "label": "MSA 日志稳定阈值(秒)", "type": "number", "min": 1, "advanced": True},
            {"key": "ENABLE_MF_WEIGHT_LOAD", "label": "Task5 启用 MF 权重加载", "type": "checkbox", "advanced": True},
        ],
        "6": [
            {
                "key": "MODEL_NAME",
                "label": "多模态模型",
                "type": "select",
                "options": ["internvl3", "qwenvl", "opensora", "cogvideox"],
                "placeholder": "internvl3",
            },
            {"key": "TOTAL_ITER", "label": "总迭代数", "type": "number", "min": 1},
            {"key": "MUTNM", "label": "每轮变异参数数量", "type": "number", "min": 1},
            {"key": "COMPARE_MODE", "label": "对比模式", "type": "select", "options": ["pta_msa"], "placeholder": "pta_msa"},
            {"key": "TRAIN_ITERS", "label": "训练步数", "type": "number", "min": 1},
            {"key": "PTA_MAX_RUNTIME", "label": "PTA 最大运行时间(秒)", "type": "number", "min": 1, "advanced": True},
            {"key": "MSA_MAX_RUNTIME", "label": "MSA 最大运行时间(秒)", "type": "number", "min": 1, "advanced": True},
        ],
    },
}

BASE_CONFIG = {
    "task_type": 1,
    "PTA_NAME": "mindspeed",
    "MSA_NAME": "msadapter",
    "MF_NAME": "mindf_py311",
    "PTA_PATH": "",
    "MSA_PATH": "",
    "SAVE_ABNORMAL_WEIGHTS": True,
    "CLUSTER": {
        "ENABLED": False,
        "MASTER_ADDR": "192.168.0.170",
        "MASTER_PORT": 8118,
        "LISTEN_HOST": "0.0.0.0",
        "LISTEN_PORT": 19001,
        "REQUEST_TIMEOUT": 30,
        "SESSION_TIMEOUT": 7200,
        "LOCAL_NPUS_PER_NODE": 0,
        "SLAVES": ["192.168.0.203:19001"],
    },
    "tasks": {
        "1": {
            "MODEL_NAME": "qwen2",
            "TOTAL_ITER": 10,
            "PTA_MAX_RUNTIME": 3000,
            "MSA_MAX_RUNTIME": 3000,
            "LOG_INIT_WAIT": 240,
            "LOG_STABLE_THRESHOLD": 150,
            "COMPARE_MODE": "pta_msa",
          "ENABLE_MF_WEIGHT_LOAD": True,
            "BASE_SEED": 43,
            "MUTNM": 2,
            "SAVE_STEPS": 1,
            "LOAD_STEPS": 30,
            "MULTI_NODE": {
                "ENABLED": False,
            },
        },
        "2": {
            "MODELS": ["qwen2", "qwen2", "qwen2"],
            "TOTAL_ITER": 100,
            "PTA_MAX_RUNTIME": 3000,
            "MSA_MAX_RUNTIME": 3000,
            "LOG_INIT_WAIT": 240,
            "LOG_STABLE_THRESHOLD": 150,
            "BASE_SEED": 43,
            "SUBMODULES": [3, 4, 5],
            "MUTNM": 2,
            "SAVE_STEPS": 1,
            "LOAD_STEPS": 15,
            "COMPARE_MODE": "pta_msa",
            "MF_ARGS_PATH": "assets/runtime/mf_templates/basic.yaml",
            "ENABLE_MF_WEIGHT_LOAD": False,
            "MULTI_NODE": {
                "ENABLED": False,
            },
        },
        "3": {
            "MODELS": ["qwen2", "glm4"],
            "TOTAL_ITER": 100,
            "PTA_MAX_RUNTIME": 3000,
            "MSA_MAX_RUNTIME": 3000,
            "LOG_INIT_WAIT": 240,
            "LOG_STABLE_THRESHOLD": 150,
            "MAX_MUTATION_WAIT": 600,
            "BASE_SEED": 43,
            "MUTNM": 2,
            "SAVE_STEPS": 1,
            "LOAD_STEPS": 15,
            "COMPARE_MODE": "pta_msa",
            "MULTI_NODE": {
                "ENABLED": False,
            },
        },
        "4": {
            "TOTAL_ITER": 5,
            "COMPARE_MODE": "pta_msa",
            "SAVE_STEPS": 1,
            "RUN_STEPS": 10,
            "MULTI_NODE": {
                "ENABLED": False,
            },
            "PTA_MAX_RUNTIME": 3000,
            "MSA_MAX_RUNTIME": 3000,
            "LOG_INIT_WAIT": 240,
            "LOG_STABLE_THRESHOLD": 150,
            "ENABLE_MF_WEIGHT_LOAD": False,
        },
        "5": {
            "TOTAL_ITER": 5,
            "COMPARE_MODE": "pta_msa",
            "SAVE_STEPS": 1,
            "RUN_STEPS": 10,
            "MUTATE_STEPS": 10,
            "MODULE_TYPE": "all",
            "MULTI_NODE": {
                "ENABLED": False,
            },
            "PTA_MAX_RUNTIME": 3000,
            "MSA_MAX_RUNTIME": 3000,
            "LOG_INIT_WAIT": 240,
            "LOG_STABLE_THRESHOLD": 150,
            "ENABLE_MF_WEIGHT_LOAD": False,
        },
        "6": {
            "MODEL_NAME": "internvl3",
            "TOTAL_ITER": 10,
            "MUTNM": 2,
            "COMPARE_MODE": "pta_msa",
            "TRAIN_ITERS": 5,
            "PTA_MAX_RUNTIME": 900,
            "MSA_MAX_RUNTIME": 900,
        },
    },
}

HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LMSV WebUI</title>
  <style>
    :root {
      --bg: #f5efe5;
      --bg-soft: #fbf7f2;
      --panel: rgba(255, 252, 247, 0.88);
      --panel-strong: rgba(255, 255, 255, 0.96);
      --line: rgba(148, 163, 184, 0.26);
      --line-strong: rgba(120, 135, 156, 0.36);
      --text: #17212f;
      --sub: #5f6f82;
      --accent: #d97706;
      --accent-2: #0f766e;
      --accent-3: #2563eb;
      --danger: #b42318;
      --success: #16794f;
      --warn: #b45309;
      --shadow: 0 24px 70px rgba(29, 41, 57, 0.10);
      --radius-xl: 30px;
      --radius-lg: 22px;
      --radius-md: 16px;
      --radius-sm: 12px;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      color: var(--text);
      font-family: "Avenir Next", "Trebuchet MS", "PingFang SC", "Microsoft YaHei", sans-serif;
      background:
        radial-gradient(circle at 0% 0%, rgba(217, 119, 6, 0.16), transparent 28%),
        radial-gradient(circle at 100% 0%, rgba(15, 118, 110, 0.16), transparent 22%),
        linear-gradient(180deg, #f8f3eb 0%, #efe8dc 46%, #f5efe5 100%);
      min-height: 100vh;
    }

    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background-image:
        linear-gradient(rgba(255,255,255,0.14) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.14) 1px, transparent 1px);
      background-size: 28px 28px;
      mask-image: linear-gradient(180deg, rgba(0,0,0,0.28), transparent 80%);
    }

    .shell {
      max-width: 1600px;
      margin: 0 auto;
      padding: 28px 22px 34px;
    }

    .hero {
      position: relative;
      overflow: hidden;
      display: grid;
      grid-template-columns: minmax(0, 1.25fr) minmax(280px, 0.9fr);
      gap: 20px;
      padding: 28px;
      border-radius: var(--radius-xl);
      background:
        linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(251,247,242,0.84) 44%, rgba(236,253,245,0.74) 100%);
      border: 1px solid rgba(255,255,255,0.82);
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
    }

    .hero::after {
      content: "";
      position: absolute;
      width: 320px;
      height: 320px;
      right: -110px;
      top: -120px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(217, 119, 6, 0.18), transparent 68%);
      pointer-events: none;
    }

    .hero-kicker {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      padding: 8px 14px;
      border-radius: 999px;
      background: rgba(23, 33, 47, 0.06);
      color: var(--sub);
      letter-spacing: 0.14em;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }

    .hero-title {
      margin: 16px 0 12px;
      font-size: clamp(34px, 4vw, 54px);
      line-height: 1.02;
      letter-spacing: -0.03em;
    }

    .hero-copy {
      max-width: 780px;
      color: var(--sub);
      font-size: 15px;
      line-height: 1.8;
    }

    .hero-side {
      display: grid;
      gap: 14px;
      align-content: start;
    }

    .hero-stat {
      padding: 18px 18px 16px;
      border-radius: 22px;
      background: rgba(255,255,255,0.84);
      border: 1px solid rgba(255,255,255,0.74);
      box-shadow: 0 14px 40px rgba(23, 33, 47, 0.08);
    }

    .hero-stat-label {
      color: var(--sub);
      font-size: 13px;
      margin-bottom: 8px;
    }

    .hero-stat strong {
      font-size: 22px;
      letter-spacing: -0.02em;
    }

    .hero-stat code {
      display: block;
      margin-top: 8px;
      color: var(--sub);
      word-break: break-all;
    }

    .layout {
      display: grid;
      grid-template-columns: minmax(460px, 0.96fr) minmax(420px, 1.04fr);
      gap: 20px;
      margin-top: 20px;
      align-items: start;
    }

    .panel {
      background: var(--panel);
      border: 1px solid rgba(255,255,255,0.78);
      border-radius: var(--radius-lg);
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
    }

    .panel-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      padding: 22px 22px 0;
    }

    .panel-head h2 {
      margin: 0;
      font-size: 24px;
      letter-spacing: -0.02em;
    }

    .panel-sub {
      padding: 8px 22px 0;
      color: var(--sub);
      font-size: 14px;
      line-height: 1.7;
    }

    .actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }

    button {
      border: 0;
      border-radius: 999px;
      padding: 11px 16px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      transition: transform 0.16s ease, box-shadow 0.16s ease, background 0.16s ease, opacity 0.16s ease;
    }

    button:hover {
      transform: translateY(-1px);
      box-shadow: 0 10px 24px rgba(23, 33, 47, 0.12);
    }

    button:disabled {
      opacity: 0.55;
      cursor: not-allowed;
      transform: none;
      box-shadow: none;
    }

    .btn-primary {
      background: linear-gradient(135deg, #d97706, #f59e0b);
      color: white;
    }

    .btn-secondary {
      background: rgba(23, 33, 47, 0.08);
      color: var(--text);
    }

    .btn-danger {
      background: rgba(180, 35, 24, 0.10);
      color: var(--danger);
    }

    .status-banner {
      margin: 18px 22px 0;
      padding: 13px 16px;
      border-radius: 16px;
      background: rgba(37, 99, 235, 0.08);
      color: #1d4ed8;
      display: none;
      line-height: 1.7;
    }

    .status-banner.success {
      display: block;
      background: rgba(22, 121, 79, 0.10);
      color: var(--success);
    }

    .status-banner.info {
      display: block;
      background: rgba(37, 99, 235, 0.08);
      color: #1d4ed8;
    }

    .status-banner.error {
      display: block;
      background: rgba(180, 35, 24, 0.10);
      color: var(--danger);
    }

    .status-banner.warn {
      display: block;
      background: rgba(180, 83, 9, 0.10);
      color: var(--warn);
    }

    .task-switch {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      padding: 20px 22px 0;
    }

    .task-pill {
      min-width: 150px;
      padding: 14px 16px;
      border-radius: 18px;
      background: rgba(255,255,255,0.54);
      border: 1px solid transparent;
      cursor: pointer;
      transition: transform 0.16s ease, border-color 0.16s ease, background 0.16s ease, box-shadow 0.16s ease;
    }

    .task-pill:hover {
      transform: translateY(-1px);
      box-shadow: 0 12px 24px rgba(23, 33, 47, 0.08);
    }

    .task-pill.active {
      background: rgba(255,255,255,0.94);
      border-color: rgba(255,255,255,0.88);
      box-shadow: 0 14px 28px rgba(23, 33, 47, 0.10);
    }

    .task-pill-title {
      font-size: 15px;
      font-weight: 800;
      margin-bottom: 5px;
    }

    .task-pill-copy {
      font-size: 12px;
      color: var(--sub);
      line-height: 1.55;
    }

    .section-block {
      padding: 18px 22px 22px;
    }

    .section-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 14px;
    }

    .section-title h3 {
      margin: 0;
      font-size: 18px;
      letter-spacing: -0.02em;
    }

    .section-title span {
      color: var(--sub);
      font-size: 13px;
    }

    .field-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }

    .field-card {
      padding: 16px;
      border-radius: 18px;
      border: 1px solid rgba(255,255,255,0.76);
      background: var(--panel-strong);
      box-shadow: 0 12px 28px rgba(23, 33, 47, 0.06);
    }

    .field-card.wide {
      grid-column: 1 / -1;
    }

    .field-card label {
      display: block;
      font-size: 14px;
      font-weight: 800;
      margin-bottom: 10px;
    }

    .field-help {
      margin-top: 8px;
      color: var(--sub);
      font-size: 12px;
      line-height: 1.6;
    }

    .field-card input[type="text"],
    .field-card input[type="number"],
    .field-card textarea {
      width: 100%;
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fff;
      color: var(--text);
      font: inherit;
      transition: border-color 0.16s ease, box-shadow 0.16s ease;
    }

    .field-card input[type="text"]:focus,
    .field-card input[type="number"]:focus,
    .field-card textarea:focus {
      outline: none;
      border-color: rgba(217, 119, 6, 0.56);
      box-shadow: 0 0 0 4px rgba(217, 119, 6, 0.12);
    }

    .field-card textarea {
      min-height: 168px;
      resize: vertical;
      font-family: "SFMono-Regular", "Menlo", "Consolas", monospace;
      line-height: 1.6;
    }

    .switch-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 14px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: #fff;
    }

    .switch-label {
      color: var(--sub);
      font-size: 13px;
      line-height: 1.6;
    }

    .toggle {
      appearance: none;
      width: 56px;
      height: 32px;
      border-radius: 999px;
      background: rgba(148, 163, 184, 0.42);
      position: relative;
      cursor: pointer;
      transition: background 0.16s ease;
    }

    .toggle::after {
      content: "";
      position: absolute;
      top: 4px;
      left: 4px;
      width: 24px;
      height: 24px;
      border-radius: 50%;
      background: #fff;
      box-shadow: 0 4px 12px rgba(15, 23, 42, 0.18);
      transition: transform 0.16s ease;
    }

    .toggle:checked {
      background: linear-gradient(135deg, #d97706, #f59e0b);
    }

    .toggle:checked::after {
      transform: translateX(24px);
    }

    .task-panel {
      display: none;
    }

    .task-panel.active {
      display: block;
    }

    .stats-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      padding: 20px 22px 12px;
    }

    .stat-card {
      padding: 16px;
      border-radius: 18px;
      background: var(--panel-strong);
      border: 1px solid rgba(255,255,255,0.76);
      box-shadow: 0 12px 28px rgba(23, 33, 47, 0.06);
    }

    .stat-card-label {
      color: var(--sub);
      font-size: 13px;
      margin-bottom: 8px;
    }

    .stat-card-value {
      font-size: 20px;
      font-weight: 800;
      letter-spacing: -0.02em;
      word-break: break-all;
    }

    .stat-card-foot {
      margin-top: 10px;
      color: var(--sub);
      font-size: 12px;
      line-height: 1.6;
    }

    .tool-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 0 22px 12px;
      color: var(--sub);
      font-size: 13px;
    }

    .tool-row label {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      cursor: pointer;
    }

    .hardware-panel {
      margin: 0 22px 16px;
      padding: 16px;
      border-radius: 20px;
      background: rgba(255,255,255,0.72);
      border: 1px solid rgba(255,255,255,0.78);
      box-shadow: 0 12px 28px rgba(23, 33, 47, 0.06);
    }

    .hardware-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 14px;
    }

    .hardware-head strong {
      font-size: 16px;
    }

    .hardware-head span {
      color: var(--sub);
      font-size: 12px;
    }

    .hardware-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }

    .hardware-card {
      padding: 14px;
      border-radius: 16px;
      background: var(--panel-strong);
      border: 1px solid rgba(255,255,255,0.78);
    }

    .hardware-card label {
      display: block;
      margin-bottom: 8px;
      color: var(--sub);
      font-size: 12px;
    }

    .hardware-card strong {
      display: block;
      font-size: 18px;
      letter-spacing: -0.02em;
    }

    .hardware-card span {
      display: block;
      margin-top: 6px;
      color: var(--sub);
      font-size: 12px;
      line-height: 1.6;
    }

    .device-list {
      display: grid;
      gap: 12px;
      margin-bottom: 14px;
    }

    .device-list[hidden] {
      display: none;
    }

    .device-card {
      padding: 14px;
      border-radius: 16px;
      background: rgba(255,255,255,0.9);
      border: 1px solid rgba(255,255,255,0.8);
    }

    .device-title {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
      font-weight: 800;
    }

    .device-metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }

    .device-metric {
      padding: 10px 12px;
      border-radius: 14px;
      background: rgba(23, 33, 47, 0.04);
    }

    .device-metric label {
      display: block;
      margin-bottom: 6px;
      color: var(--sub);
      font-size: 11px;
    }

    .device-metric strong {
      font-size: 13px;
      word-break: break-word;
    }

    .collapse-toggle {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 0;
      background: transparent;
      color: var(--accent-3);
      font-size: 13px;
      font-weight: 700;
      border-radius: 0;
    }

    .collapse-toggle:hover {
      transform: none;
      box-shadow: none;
      color: var(--accent);
    }

    .log-viewer {
      margin: 0 22px 22px;
      min-height: 520px;
      max-height: 740px;
      overflow: auto;
      padding: 18px;
      border-radius: 20px;
      background:
        linear-gradient(180deg, rgba(17, 24, 39, 0.98) 0%, rgba(15, 23, 42, 0.96) 100%);
      color: #d7e3f5;
      border: 1px solid rgba(15, 23, 42, 0.34);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
      font-family: "SFMono-Regular", "Consolas", "Liberation Mono", monospace;
      font-size: 13px;
      line-height: 1.7;
      white-space: pre-wrap;
      word-break: break-word;
    }

    .results-panel {
      margin-top: 20px;
      overflow: hidden;
    }

    .results-layout {
      display: grid;
      grid-template-columns: minmax(280px, 380px) minmax(0, 1fr);
      gap: 18px;
      padding: 20px 22px 22px;
    }

    .result-list {
      display: grid;
      gap: 12px;
      max-height: 860px;
      overflow: auto;
      padding-right: 4px;
    }

    .result-card {
      padding: 16px;
      border-radius: 18px;
      border: 1px solid rgba(255,255,255,0.78);
      background: var(--panel-strong);
      cursor: pointer;
      transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
      box-shadow: 0 10px 24px rgba(23, 33, 47, 0.06);
    }

    .result-card:hover {
      transform: translateY(-1px);
      box-shadow: 0 16px 30px rgba(23, 33, 47, 0.10);
    }

    .result-card.active {
      border-color: rgba(217, 119, 6, 0.40);
      box-shadow: 0 18px 34px rgba(217, 119, 6, 0.18);
    }

    .result-head {
      display: flex;
      justify-content: space-between;
      align-items: start;
      gap: 12px;
      margin-bottom: 12px;
    }

    .result-title {
      font-weight: 800;
      font-size: 15px;
      line-height: 1.5;
      word-break: break-all;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 800;
      white-space: nowrap;
    }

    .badge.report {
      background: rgba(22, 121, 79, 0.10);
      color: var(--success);
    }

    .badge.pending {
      background: rgba(37, 99, 235, 0.10);
      color: #1d4ed8;
    }

    .badge.warn {
      background: rgba(180, 83, 9, 0.10);
      color: var(--warn);
    }

    .badge.task {
      background: rgba(23, 33, 47, 0.07);
      color: var(--text);
      margin-top: 8px;
    }

    .result-copy {
      color: var(--sub);
      font-size: 13px;
      line-height: 1.75;
    }

    .result-foot {
      margin-top: 12px;
      color: var(--sub);
      font-size: 12px;
    }

    .report-shell {
      display: grid;
      gap: 14px;
      min-width: 0;
    }

    .report-meta {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }

    .report-meta-card {
      padding: 14px 16px;
      border-radius: 16px;
      border: 1px solid rgba(255,255,255,0.78);
      background: var(--panel-strong);
      box-shadow: 0 10px 24px rgba(23, 33, 47, 0.06);
    }

    .report-meta-card label {
      display: block;
      color: var(--sub);
      font-size: 12px;
      margin-bottom: 8px;
    }

    .report-meta-card strong,
    .report-meta-card a {
      color: var(--text);
      font-size: 14px;
      word-break: break-all;
      text-decoration: none;
    }

    .report-meta-card a:hover {
      color: var(--accent);
    }

    .report-empty {
      min-height: 640px;
      display: grid;
      place-items: center;
      border-radius: 22px;
      background:
        linear-gradient(135deg, rgba(255,255,255,0.88), rgba(251,247,242,0.86));
      border: 1px dashed var(--line-strong);
      color: var(--sub);
      text-align: center;
      padding: 26px;
      line-height: 1.8;
    }

    .viewer-toolbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
      padding: 14px 16px;
      border-radius: 18px;
      background: rgba(255,255,255,0.74);
      border: 1px solid rgba(255,255,255,0.8);
    }

    .viewer-tabs {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }

    .viewer-tab {
      padding: 9px 14px;
      border-radius: 999px;
      background: rgba(23, 33, 47, 0.07);
      color: var(--text);
      font-size: 13px;
      font-weight: 800;
    }

    .viewer-tab.active {
      background: linear-gradient(135deg, #d97706, #f59e0b);
      color: #fff;
    }

    .viewer-tab:disabled {
      opacity: 0.42;
    }

    .viewer-actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }

    .preview-head-actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }

    .report-frame,
    .file-preview-frame {
      width: 100%;
      min-height: 980px;
      border: 0;
      border-radius: 22px;
      background: white;
      box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.06);
      display: none;
    }

    .browser-shell {
      display: none;
      grid-template-columns: minmax(280px, 360px) minmax(0, 1fr);
      gap: 14px;
      min-width: 0;
    }

    .browser-pane,
    .preview-pane {
      min-width: 0;
      padding: 16px;
      border-radius: 22px;
      background: rgba(255,255,255,0.92);
      border: 1px solid rgba(255,255,255,0.8);
      box-shadow: 0 10px 24px rgba(23, 33, 47, 0.06);
    }

    .browser-head,
    .preview-head {
      display: flex;
      justify-content: space-between;
      align-items: start;
      gap: 12px;
      margin-bottom: 14px;
    }

    .browser-title,
    .preview-title {
      font-size: 16px;
      font-weight: 800;
      word-break: break-word;
    }

    .browser-copy,
    .preview-copy {
      color: var(--sub);
      font-size: 12px;
      line-height: 1.7;
    }

    .breadcrumb-row {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }

    .crumb {
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(23, 33, 47, 0.06);
      color: var(--text);
      font-size: 12px;
      text-decoration: none;
      cursor: pointer;
    }

    .crumb:hover {
      color: var(--accent);
    }

    .browser-list {
      display: grid;
      gap: 10px;
      max-height: 840px;
      overflow: auto;
      padding-right: 4px;
    }

    .browser-entry-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 8px;
      align-items: stretch;
    }

    .browser-entry {
      display: flex;
      justify-content: space-between;
      align-items: start;
      gap: 12px;
      width: 100%;
      padding: 14px;
      border-radius: 16px;
      background: rgba(23, 33, 47, 0.04);
      border: 1px solid transparent;
      text-align: left;
      box-shadow: none;
    }

    .browser-entry:hover {
      background: rgba(217, 119, 6, 0.08);
    }

    .browser-entry.active {
      border-color: rgba(217, 119, 6, 0.38);
      background: rgba(217, 119, 6, 0.10);
    }

    .browser-entry-name {
      font-weight: 800;
      line-height: 1.6;
      word-break: break-all;
    }

    .browser-entry-meta {
      margin-top: 6px;
      color: var(--sub);
      font-size: 12px;
      line-height: 1.6;
    }

    .browser-entry-kind {
      color: var(--sub);
      font-size: 12px;
      white-space: nowrap;
    }

    .browser-head-actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }

    .preview-meta {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }

    .preview-meta-card {
      padding: 12px 14px;
      border-radius: 16px;
      background: rgba(23, 33, 47, 0.04);
    }

    .preview-meta-card label {
      display: block;
      color: var(--sub);
      font-size: 11px;
      margin-bottom: 6px;
    }

    .preview-meta-card strong,
    .preview-meta-card a {
      color: var(--text);
      font-size: 13px;
      word-break: break-word;
      text-decoration: none;
    }

    .preview-meta-card a:hover {
      color: var(--accent);
    }

    .text-preview {
      margin: 0;
      max-height: 760px;
      overflow: auto;
      padding: 18px;
      border-radius: 18px;
      background: linear-gradient(180deg, rgba(17, 24, 39, 0.98) 0%, rgba(15, 23, 42, 0.96) 100%);
      color: #d7e3f5;
      font-family: "SFMono-Regular", "Consolas", "Liberation Mono", monospace;
      font-size: 13px;
      line-height: 1.7;
      white-space: pre-wrap;
      word-break: break-word;
    }

    .markdown-preview,
    .csv-preview-shell {
      max-height: 760px;
      overflow: auto;
      padding: 22px;
      border-radius: 18px;
      background: rgba(255,255,255,0.98);
      box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.06);
    }

    .markdown-preview {
      color: #17212f;
      line-height: 1.75;
    }

    .markdown-preview > *:first-child {
      margin-top: 0;
    }

    .markdown-preview > *:last-child {
      margin-bottom: 0;
    }

    .markdown-preview h1,
    .markdown-preview h2,
    .markdown-preview h3,
    .markdown-preview h4,
    .markdown-preview h5,
    .markdown-preview h6 {
      margin: 1.2em 0 0.6em;
      line-height: 1.3;
    }

    .markdown-preview p,
    .markdown-preview ul,
    .markdown-preview ol,
    .markdown-preview blockquote,
    .markdown-preview pre,
    .markdown-preview table {
      margin: 0 0 1em;
    }

    .markdown-preview code {
      padding: 0.15em 0.4em;
      border-radius: 6px;
      background: rgba(15, 23, 42, 0.08);
      font-family: "SFMono-Regular", "Consolas", "Liberation Mono", monospace;
      font-size: 0.92em;
    }

    .markdown-preview pre {
      padding: 16px;
      overflow: auto;
      border-radius: 14px;
      background: linear-gradient(180deg, rgba(17, 24, 39, 0.98) 0%, rgba(15, 23, 42, 0.96) 100%);
      color: #d7e3f5;
    }

    .markdown-preview pre code {
      padding: 0;
      background: transparent;
      color: inherit;
    }

    .markdown-preview blockquote {
      padding: 12px 16px;
      border-left: 4px solid rgba(217, 119, 6, 0.7);
      background: rgba(217, 119, 6, 0.08);
      color: #6b4f1d;
    }

    .markdown-preview a {
      color: #b45309;
    }

    .markdown-preview table,
    .csv-preview-table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }

    .markdown-preview th,
    .markdown-preview td,
    .csv-preview-table th,
    .csv-preview-table td {
      padding: 10px 12px;
      border: 1px solid rgba(148, 163, 184, 0.32);
      vertical-align: top;
      text-align: left;
      word-break: break-word;
    }

    .markdown-preview thead th,
    .csv-preview-table thead th {
      position: sticky;
      top: 0;
      background: #fff8ef;
      z-index: 1;
    }

    .csv-preview-summary {
      margin-bottom: 14px;
      color: var(--sub);
      font-size: 12px;
    }

    .image-preview {
      display: block;
      max-width: 100%;
      max-height: 760px;
      margin: 0 auto;
      border-radius: 18px;
      background: #fff;
      box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.06);
    }

    .empty-state {
      padding: 22px;
      border-radius: 18px;
      background: rgba(255,255,255,0.7);
      border: 1px dashed var(--line-strong);
      color: var(--sub);
      text-align: center;
      line-height: 1.8;
    }

    .preview-overlay {
      position: fixed;
      inset: 0;
      z-index: 1200;
      display: none;
      padding: 22px;
      background: rgba(15, 23, 42, 0.72);
      backdrop-filter: blur(6px);
    }

    .preview-overlay.open {
      display: block;
    }

    .preview-overlay-panel {
      display: flex;
      flex-direction: column;
      width: min(1500px, 100%);
      height: 100%;
      margin: 0 auto;
      border-radius: 24px;
      overflow: hidden;
      background: #f8fafc;
      box-shadow: 0 28px 80px rgba(15, 23, 42, 0.35);
    }

    .preview-overlay-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      padding: 18px 22px;
      border-bottom: 1px solid rgba(148, 163, 184, 0.22);
      background: rgba(255,255,255,0.95);
    }

    .preview-overlay-title {
      font-size: 16px;
      font-weight: 800;
      word-break: break-word;
    }

    .preview-overlay-copy {
      margin-top: 4px;
      color: var(--sub);
      font-size: 12px;
    }

    .preview-overlay-body {
      flex: 1;
      min-height: 0;
      padding: 22px;
      overflow: auto;
      background:
        radial-gradient(circle at top right, rgba(245, 158, 11, 0.08), transparent 30%),
        linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.98));
    }

    .preview-overlay-body .text-preview,
    .preview-overlay-body .markdown-preview,
    .preview-overlay-body .csv-preview-shell,
    .preview-overlay-body .image-preview,
    .preview-overlay-body .file-preview-frame {
      max-height: none;
      min-height: calc(100vh - 220px);
    }

    @media (max-width: 1200px) {
      .layout,
      .hero,
      .results-layout {
        grid-template-columns: 1fr;
      }

      .report-meta,
      .field-grid,
      .stats-grid,
      .hardware-grid,
      .device-metrics,
      .browser-shell,
      .preview-meta {
        grid-template-columns: 1fr;
      }
    }

    @media (max-width: 720px) {
      .shell { padding: 16px; }
      .panel-head { padding: 18px 18px 0; }
      .panel-sub,
      .section-block,
      .task-switch,
      .stats-grid,
      .tool-row,
      .results-layout { padding-left: 18px; padding-right: 18px; }
      .log-viewer { margin: 0 18px 18px; min-height: 420px; }
      .hero { padding: 22px; }
      .hero-title { font-size: 32px; }
      .preview-overlay { padding: 12px; }
      .preview-overlay-head,
      .preview-overlay-body { padding: 16px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div>
        <div class="hero-kicker">LMSV CONTROL CENTER</div>
        <h1 class="hero-title">配置、启动、追踪、回看结果</h1>
        <div class="hero-copy" id="heroTagline">
          直接在浏览器里完成配置管理、任务启动和报告查看。参数字段与 CLI 保持一致，任务日志实时滚动，历史产物可从 output 目录快速回看。
        </div>
      </div>
      <div class="hero-side">
        <div class="hero-stat">
          <div class="hero-stat-label">当前任务</div>
          <strong id="heroTaskLabel">-</strong>
          <code id="heroTaskHint">-</code>
        </div>
        <div class="hero-stat">
          <div class="hero-stat-label">运行状态</div>
          <strong id="heroRunState">空闲</strong>
          <code id="heroOutputDir">尚未创建 output 目录</code>
        </div>
      </div>
    </section>

    <div class="layout">
      <section class="panel">
        <div class="panel-head">
          <h2>配置面板</h2>
          <div class="actions">
            <button class="btn-secondary" id="saveConfigBtn">保存配置</button>
            <button class="btn-primary" id="startRunBtn">保存并启动</button>
          </div>
        </div>
        <div class="panel-sub">
          表单字段与 `genconf.py` 一致。任务 2 中 `MODELS` 与 `SUBMODULES` 必须一一对应；任务 4/5 支持 `MULTI_NODE` 多机配置；任务 5 还支持 `MODULE_TYPE` 与 `MUTATE_STEPS` 配置。WebUI 和运行时都会做参数校验，高级配置默认折叠。
        </div>
        <div class="status-banner" id="messageBar"></div>
        <div class="task-switch" id="taskSwitch"></div>

        <div class="section-block">
          <div class="section-title">
            <h3>全局配置</h3>
            <span>应用于所有任务</span>
          </div>
          <div class="field-grid" id="globalFields"></div>
        </div>

        <div class="section-block">
          <div class="section-title">
            <h3>任务参数</h3>
            <span>按当前任务类型显示</span>
          </div>
          <div id="taskPanels"></div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <h2>任务运行</h2>
          <div class="actions">
            <button class="btn-secondary" id="refreshStatusBtn">刷新状态</button>
            <button class="btn-danger" id="stopRunBtn">停止任务</button>
          </div>
        </div>
        <div class="panel-sub">
          WebUI 直接启动 `do.py`，日志来自实时 stdout 采集，并自动关联本次产生的 `output/<时间戳>` 目录。
        </div>
        <div class="stats-grid">
          <div class="stat-card">
            <div class="stat-card-label">任务状态</div>
            <div class="stat-card-value" id="runStateCard">空闲</div>
            <div class="stat-card-foot" id="runStateFoot">等待启动</div>
          </div>
          <div class="stat-card">
            <div class="stat-card-label">进程信息</div>
            <div class="stat-card-value" id="runPidCard">-</div>
            <div class="stat-card-foot" id="runTimeCard">未运行</div>
          </div>
          <div class="stat-card">
            <div class="stat-card-label">输出目录</div>
            <div class="stat-card-value" id="runOutputCard">-</div>
            <div class="stat-card-foot" id="runReturnCard">尚未生成</div>
          </div>
          <div class="stat-card">
            <div class="stat-card-label">日志游标</div>
            <div class="stat-card-value" id="runCursorCard">0</div>
            <div class="stat-card-foot" id="runCursorFoot">首次拉取会完整刷新</div>
          </div>
        </div>
        <div class="tool-row">
          <label><input type="checkbox" id="autoScrollToggle" checked /> 日志自动滚动</label>
          <span id="runHintText">日志轮询中</span>
        </div>
        <div class="hardware-panel">
          <div class="hardware-head">
            <strong>NPU 硬件监视</strong>
            <span id="hardwareUpdatedAt">等待首次采样</span>
          </div>
          <div class="hardware-grid">
            <div class="hardware-card">
              <label>CPU</label>
              <strong id="hardwareCpuValue">-</strong>
              <span id="hardwareCpuFoot">等待采样</span>
            </div>
            <div class="hardware-card">
              <label>内存</label>
              <strong id="hardwareMemoryValue">-</strong>
              <span id="hardwareMemoryFoot">等待采样</span>
            </div>
            <div class="hardware-card">
              <label>NPU 概况</label>
              <strong id="hardwareAccelValue">-</strong>
              <span id="hardwareAccelFoot">等待采样</span>
            </div>
          </div>
          <button class="collapse-toggle" id="hardwareToggleBtn" type="button">展开 NPU 详情</button>
          <div class="device-list" id="hardwareDeviceList" hidden>
            <div class="empty-state">尚未获取到 NPU 设备信息。</div>
          </div>
        </div>
        <pre class="log-viewer" id="logViewer">等待任务启动…</pre>
      </section>
    </div>

    <section class="panel results-panel">
      <div class="panel-head">
        <h2>历史结果</h2>
        <div class="actions">
          <button class="btn-secondary" id="deleteResultBtn" disabled>删除结果</button>
          <button class="btn-secondary" id="regenerateAnalysisBtn" disabled>重新生成 analysis</button>
          <button class="btn-secondary" id="refreshResultsBtn">刷新结果</button>
        </div>
      </div>
      <div class="panel-sub">
        选择 output 目录中的任务后，默认优先展示 `analysis/report.html`，也可以直接在线浏览同目录下的其它文件。
      </div>
      <div class="results-layout">
        <div id="resultList" class="result-list"></div>
        <div class="report-shell">
          <div class="report-meta" id="reportMeta"></div>
          <div class="viewer-toolbar" id="viewerToolbar">
            <div class="viewer-tabs">
              <button class="viewer-tab active" id="reportModeBtn" type="button">报告视图</button>
              <button class="viewer-tab" id="browserModeBtn" type="button">文件浏览</button>
            </div>
            <div class="viewer-actions">
              <button class="btn-secondary" id="openRawBtn" type="button" disabled>打开原文件</button>
            </div>
          </div>
          <div class="report-empty" id="reportEmpty">
            从左侧选择一个任务，即可在这里加载 `analysis/report.html`。
          </div>
          <iframe class="report-frame" id="reportFrame" title="LMSV Report Viewer"></iframe>
          <div class="browser-shell" id="browserShell">
            <div class="browser-pane">
              <div class="browser-head">
                <div>
                  <div class="browser-title">文件目录</div>
                  <div class="browser-copy" id="browserDirHint">浏览 output 目录中的历史产物</div>
                </div>
                <div class="browser-head-actions">
                  <button class="btn-secondary" id="downloadCurrentIterBtn" type="button" disabled>下载当前 iter ZIP</button>
                </div>
              </div>
              <div class="breadcrumb-row" id="browserBreadcrumbs"></div>
              <div class="browser-list" id="browserList"></div>
            </div>
            <div class="preview-pane">
              <div class="preview-head">
                <div>
                  <div class="preview-title" id="previewTitle">文件预览</div>
                  <div class="preview-copy" id="previewCopy">选择一个文件即可在线预览</div>
                </div>
                <div class="preview-head-actions">
                  <button class="btn-secondary" id="fullscreenPreviewBtn" type="button" disabled>全屏查看</button>
                </div>
              </div>
              <div class="preview-meta" id="previewMeta"></div>
              <div class="report-empty" id="filePreviewEmpty">选择左侧文件后，会在这里显示内容预览。</div>
              <pre class="text-preview" id="textPreview" hidden></pre>
              <div class="markdown-preview" id="markdownPreview" hidden></div>
              <div class="csv-preview-shell" id="csvPreview" hidden></div>
              <img class="image-preview" id="imagePreview" alt="" hidden />
              <iframe class="file-preview-frame" id="filePreviewFrame" title="LMSV File Preview"></iframe>
            </div>
          </div>
        </div>
      </div>
    </section>
  </div>

  <div class="preview-overlay" id="previewOverlay">
    <div class="preview-overlay-panel">
      <div class="preview-overlay-head">
        <div>
          <div class="preview-overlay-title" id="previewOverlayTitle">全屏预览</div>
          <div class="preview-overlay-copy" id="previewOverlayCopy"></div>
        </div>
        <button class="btn-secondary" id="closePreviewOverlayBtn" type="button">关闭</button>
      </div>
      <div class="preview-overlay-body" id="previewOverlayBody"></div>
    </div>
  </div>

  <script>
    const FORM_SCHEMA = __FORM_SCHEMA__;
    const TASK_META = __TASK_META__;

    const state = {
      config: null,
      selectedTask: "1",
      results: [],
      selectedResultId: null,
      resultViewMode: "report",
      deletingResultId: null,
      regeneratingResultId: null,
      runCursor: 0,
      pollTimer: null,
      lastAutoSelectedOutput: null,
      lastTerminalRefreshKey: null,
      runStatus: null,
      hardwareExpanded: false,
      advancedExpanded: false,
      browserPath: "",
      browserEntries: [],
      browserSelectedFile: null,
      previewRequestToken: 0,
      currentPreview: null,
    };

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function renderInlineMarkdown(text) {
      let html = escapeHtml(text ?? "");
      html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
      html = html.replace(/\\*\\*([^*]+)\\*\\*/g, "<strong>$1</strong>");
      html = html.replace(/\\*([^*]+)\\*/g, "<em>$1</em>");
      html = html.replace(/\\[([^\\]]+)\\]\\((https?:\\/\\/[^\\s)]+)\\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
      return html;
    }

    function consumeMarkdownTable(lines, startIndex) {
      if (startIndex + 1 >= lines.length) {
        return null;
      }
      const headerLine = lines[startIndex];
      const separatorLine = lines[startIndex + 1];
      if (!headerLine.includes("|")) {
        return null;
      }
      const separatorCells = separatorLine.trim().split("|").map((cell) => cell.trim()).filter(Boolean);
      if (!separatorCells.length || !separatorCells.every((cell) => /^:?-{3,}:?$/.test(cell))) {
        return null;
      }
      const rows = [];
      let index = startIndex + 2;
      while (index < lines.length && lines[index].includes("|") && lines[index].trim()) {
        rows.push(lines[index]);
        index += 1;
      }
      const splitCells = (line) => {
        const normalized = line.trim().replace(/^\\|/, "").replace(/\\|$/, "");
        return normalized.split("|").map((cell) => cell.trim());
      };
      return {
        nextIndex: index,
        header: splitCells(headerLine),
        rows: rows.map(splitCells),
      };
    }

    function renderMarkdown(text) {
      const lines = String(text ?? "").replace(/\\r\\n/g, "\\n").split("\\n");
      const html = [];
      let index = 0;
      let paragraph = [];
      let listKind = null;
      let listItems = [];
      let codeFence = null;
      let codeLines = [];

      const flushParagraph = () => {
        if (!paragraph.length) {
          return;
        }
        html.push(`<p>${renderInlineMarkdown(paragraph.join("<br />"))}</p>`);
        paragraph = [];
      };

      const flushList = () => {
        if (!listItems.length || !listKind) {
          return;
        }
        html.push(`<${listKind}>${listItems.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</${listKind}>`);
        listItems = [];
        listKind = null;
      };

      const flushCodeFence = () => {
        if (codeFence === null) {
          return;
        }
        html.push(`<pre><code>${escapeHtml(codeLines.join("\\n"))}</code></pre>`);
        codeFence = null;
        codeLines = [];
      };

      while (index < lines.length) {
        const line = lines[index];
        const trimmed = line.trim();

        if (codeFence !== null) {
          if (trimmed.startsWith("```")) {
            flushCodeFence();
          } else {
            codeLines.push(line);
          }
          index += 1;
          continue;
        }

        if (trimmed.startsWith("```")) {
          flushParagraph();
          flushList();
          codeFence = trimmed.slice(3).trim() || "";
          codeLines = [];
          index += 1;
          continue;
        }

        const table = consumeMarkdownTable(lines, index);
        if (table) {
          flushParagraph();
          flushList();
          html.push(`
            <table>
              <thead><tr>${table.header.map((cell) => `<th>${renderInlineMarkdown(cell)}</th>`).join("")}</tr></thead>
              <tbody>${table.rows.map((row) => `<tr>${row.map((cell) => `<td>${renderInlineMarkdown(cell)}</td>`).join("")}</tr>`).join("")}</tbody>
            </table>
          `);
          index = table.nextIndex;
          continue;
        }

        if (!trimmed) {
          flushParagraph();
          flushList();
          index += 1;
          continue;
        }

        const headingMatch = trimmed.match(/^(#{1,6})\\s+(.*)$/);
        if (headingMatch) {
          flushParagraph();
          flushList();
          const level = headingMatch[1].length;
          html.push(`<h${level}>${renderInlineMarkdown(headingMatch[2])}</h${level}>`);
          index += 1;
          continue;
        }

        if (/^(-{3,}|\\*{3,}|_{3,})$/.test(trimmed)) {
          flushParagraph();
          flushList();
          html.push("<hr />");
          index += 1;
          continue;
        }

        const quoteMatch = line.match(/^>\\s?(.*)$/);
        if (quoteMatch) {
          flushParagraph();
          flushList();
          const quoteLines = [];
          let quoteIndex = index;
          while (quoteIndex < lines.length) {
            const match = lines[quoteIndex].match(/^>\\s?(.*)$/);
            if (!match) {
              break;
            }
            quoteLines.push(match[1]);
            quoteIndex += 1;
          }
          html.push(`<blockquote>${quoteLines.map((item) => `<p>${renderInlineMarkdown(item || "")}</p>`).join("")}</blockquote>`);
          index = quoteIndex;
          continue;
        }

        const ulMatch = line.match(/^\\s*[-*+]\\s+(.*)$/);
        if (ulMatch) {
          flushParagraph();
          if (listKind && listKind !== "ul") {
            flushList();
          }
          listKind = "ul";
          listItems.push(ulMatch[1]);
          index += 1;
          continue;
        }

        const olMatch = line.match(/^\\s*\\d+\\.\\s+(.*)$/);
        if (olMatch) {
          flushParagraph();
          if (listKind && listKind !== "ol") {
            flushList();
          }
          listKind = "ol";
          listItems.push(olMatch[1]);
          index += 1;
          continue;
        }

        if (listKind) {
          flushList();
        }
        paragraph.push(line);
        index += 1;
      }

      flushParagraph();
      flushList();
      flushCodeFence();
      return html.join("\\n");
    }

    function buildCsvTable(payload) {
      const headers = Array.isArray(payload.headers) ? payload.headers : [];
      const rows = Array.isArray(payload.rows) ? payload.rows : [];
      const totalRows = Number(payload.total_rows || rows.length || 0);
      const shownRows = rows.length;
      const headerCells = headers.length
        ? headers.map((header, idx) => `<th>${escapeHtml(header || `col_${idx + 1}`)}</th>`).join("")
        : (rows[0] || []).map((_, idx) => `<th>col_${idx + 1}</th>`).join("");
      const bodyRows = rows.map((row) => `
        <tr>${row.map((cell) => `<td>${escapeHtml(cell ?? "")}</td>`).join("")}</tr>
      `).join("");
      const summary = payload.truncated
        ? `已渲染前 ${shownRows} 行，共 ${totalRows} 行。`
        : `共 ${shownRows} 行。`;
      return `
        <div class="csv-preview-summary">${escapeHtml(summary)}</div>
        <table class="csv-preview-table">
          <thead><tr>${headerCells}</tr></thead>
          <tbody>${bodyRows}</tbody>
        </table>
      `;
    }

    function setCurrentPreview(preview) {
      state.currentPreview = preview;
      document.getElementById("fullscreenPreviewBtn").disabled = !preview;
    }

    function closePreviewOverlay() {
      document.getElementById("previewOverlay").classList.remove("open");
      document.getElementById("previewOverlayBody").innerHTML = "";
    }

    function openPreviewOverlay() {
      if (!state.currentPreview) {
        return;
      }
      document.getElementById("previewOverlayTitle").textContent = state.currentPreview.title || "全屏预览";
      document.getElementById("previewOverlayCopy").textContent = state.currentPreview.subtitle || "";
      const body = document.getElementById("previewOverlayBody");
      if (state.currentPreview.mode === "frame") {
        body.innerHTML = `<iframe class="file-preview-frame" title="Fullscreen Preview" src="${escapeHtml(state.currentPreview.src)}" style="display:block;"></iframe>`;
      } else if (state.currentPreview.mode === "image") {
        body.innerHTML = `<img class="image-preview" alt="" src="${escapeHtml(state.currentPreview.src)}" />`;
      } else if (state.currentPreview.mode === "html") {
        body.innerHTML = state.currentPreview.html || "";
      } else if (state.currentPreview.mode === "text") {
        body.innerHTML = `<pre class="text-preview" style="display:block;">${escapeHtml(state.currentPreview.text || "")}</pre>`;
      } else {
        body.innerHTML = '<div class="empty-state">当前预览暂不支持全屏。</div>';
      }
      document.getElementById("previewOverlay").classList.add("open");
    }

    function showMessage(text, tone = "info") {
      const node = document.getElementById("messageBar");
      node.textContent = text;
      node.className = `status-banner ${tone}`;
    }

    function clearMessage() {
      const node = document.getElementById("messageBar");
      node.textContent = "";
      node.className = "status-banner";
    }

    function cloneJson(value) {
      return JSON.parse(JSON.stringify(value));
    }

    function normalizeClusterClient(value, options = {}) {
      const keepEmptySlaves = Boolean(options.keepEmptySlaves);
      const raw = value && typeof value === "object" ? value : {};
      const rawSlaves = Array.isArray(raw.SLAVES) ? raw.SLAVES : ["192.168.0.203:19001"];
      const slaves = rawSlaves.map((item) => {
        if (item && typeof item === "object" && !Array.isArray(item)) {
          return {
            ENDPOINT: String(item.ENDPOINT || "").trim(),
            LABEL: String(item.LABEL || "").trim(),
            NPUS_PER_NODE: Number.parseInt(item.NPUS_PER_NODE || 0, 10) || 0,
          };
        }
        return {
          ENDPOINT: String(item || "").trim(),
          LABEL: "",
          NPUS_PER_NODE: 0,
        };
      }).filter((item) => keepEmptySlaves || item.ENDPOINT);
      return {
        ENABLED: Boolean(raw.ENABLED),
        MASTER_ADDR: String(raw.MASTER_ADDR || "192.168.0.170").trim() || "192.168.0.170",
        MASTER_PORT: Number.parseInt(raw.MASTER_PORT || 8118, 10) || 8118,
        LISTEN_HOST: String(raw.LISTEN_HOST || "0.0.0.0").trim() || "0.0.0.0",
        LISTEN_PORT: Number.parseInt(raw.LISTEN_PORT || 19001, 10) || 19001,
        REQUEST_TIMEOUT: Number.parseInt(raw.REQUEST_TIMEOUT || 30, 10) || 30,
        SESSION_TIMEOUT: Number.parseInt(raw.SESSION_TIMEOUT || 7200, 10) || 7200,
        LOCAL_NPUS_PER_NODE: Number.parseInt(raw.LOCAL_NPUS_PER_NODE || 0, 10) || 0,
        SLAVES: slaves,
      };
    }

    function normalizeTask45MultiNodeClient(value, options = {}) {
      const keepEmptyNodes = Boolean(options.keepEmptyNodes);
      const raw = value && typeof value === "object" ? value : {};
      const rawNodes = Array.isArray(raw.OTHER_NODES) ? raw.OTHER_NODES : [];
      const nodes = rawNodes.map((item) => {
        const node = item && typeof item === "object" ? item : {};
        return {
          HOST: String(node.HOST || "").trim(),
          SSH_PORT: Number.parseInt(node.SSH_PORT || 22, 10) || 22,
          LMSV_PATH: String(node.LMSV_PATH || "").trim(),
          PTA_NAME: String(node.PTA_NAME || "").trim(),
          MSA_NAME: String(node.MSA_NAME || "").trim(),
          PTA_PATH: String(node.PTA_PATH || "").trim(),
          MSA_PATH: String(node.MSA_PATH || "").trim(),
          HAS_CONTAINER: Boolean(node.HAS_CONTAINER),
          CONTAINER_NAME: String(node.CONTAINER_NAME || "").trim(),
        };
      }).filter((item) => keepEmptyNodes || item.HOST || item.LMSV_PATH || item.PTA_NAME || item.MSA_NAME || item.PTA_PATH || item.MSA_PATH);
      const expectedNnodes = Math.max(2, nodes.length + 1);
      const parsedNnodes = Number.parseInt(raw.NNODES || expectedNnodes, 10);
      return {
        ENABLED: Boolean(raw.ENABLED),
        MASTER_ADDR: String(raw.MASTER_ADDR || "127.0.0.1").trim() || "127.0.0.1",
        NNODES: Number.isFinite(parsedNnodes) && parsedNnodes > 0 ? parsedNnodes : expectedNnodes,
        OTHER_NODES: nodes,
      };
    }

    const APP_BASE_URL = (() => {
      const url = new URL(window.location.href);
      url.search = "";
      url.hash = "";
      if (!url.pathname.endsWith("/")) {
        url.pathname += "/";
      }
      return url;
    })();

    function appUrl(path) {
      const cleanPath = String(path || "").replace(/^\\/+/, "");
      return new URL(cleanPath, APP_BASE_URL).toString();
    }

    async function apiFetch(url, options = {}) {
      const response = await fetch(url, {
        headers: {
          "Content-Type": "application/json",
          ...(options.headers || {}),
        },
        ...options,
      });
      let payload = {};
      try {
        payload = await response.json();
      } catch (error) {
        if (!response.ok) {
          throw new Error(`请求失败：${response.status}`);
        }
        return {};
      }
      if (!response.ok) {
        throw new Error(payload.diagnosis || payload.error || `请求失败：${response.status}`);
      }
      return payload;
    }

    function fieldValueToString(field, value) {
      if (field.type === "checkbox") {
        return Boolean(value);
      }
      if (field.type === "cluster") {
        return normalizeClusterClient(value);
      }
      if (field.type === "task45_multinode") {
        return normalizeTask45MultiNodeClient(value);
      }
      if (field.type === "json") {
        if (value === null || value === undefined || value === "") {
          return "";
        }
        try {
          return JSON.stringify(value, null, 2);
        } catch (error) {
          return String(value);
        }
      }
      if (field.type === "list_text" || field.type === "list_int") {
        return Array.isArray(value) ? value.join(",") : "";
      }
      return value ?? "";
    }

    function renderField(scope, taskId, field, value) {
      const inputId = `${scope}-${taskId || "global"}-${field.key}`;
      const wideClass =
        field.type === "checkbox" ||
        field.type === "json" ||
        field.type === "cluster" ||
        field.type === "task45_multinode"
          ? "field-card wide"
          : "field-card";
      if (field.type === "checkbox") {
        return `
          <div class="${wideClass}">
            <label for="${inputId}">${escapeHtml(field.label)}</label>
            <div class="switch-row">
              <div class="switch-label">切换后会写入配置中的布尔参数。</div>
              <input
                class="toggle"
                id="${inputId}"
                data-scope="${scope}"
                data-task="${taskId || ""}"
                data-key="${field.key}"
                type="checkbox"
                ${value ? "checked" : ""}
              />
            </div>
          </div>
        `;
      }
      if (field.type === "cluster") {
        const cluster = normalizeClusterClient(value, { keepEmptySlaves: true });
        const slaveCards = cluster.SLAVES.map((item, index) => `
          <div class="field-card" style="border-style:dashed;" data-cluster-slave-row="${index}">
            <label>子节点 ${index + 1}</label>
            <input type="text" data-cluster-slave-field="ENDPOINT" data-cluster-index="${index}" value="${escapeHtml(item.ENDPOINT || "")}" placeholder="192.168.0.203:19001" />
            <input type="text" data-cluster-slave-field="LABEL" data-cluster-index="${index}" value="${escapeHtml(item.LABEL || "")}" placeholder="可选标签，如 node-b" style="margin-top:10px;" />
            <input type="number" min="0" data-cluster-slave-field="NPUS_PER_NODE" data-cluster-index="${index}" value="${Number(item.NPUS_PER_NODE || 0)}" placeholder="0=自动探测" style="margin-top:10px;" />
            <div class="field-help">NODE_RANK 会按列表顺序自动分配，子节点 1 对应 rank1。</div>
            <button type="button" class="secondary-btn" data-cluster-remove="${index}" style="margin-top:10px;">删除子节点</button>
          </div>
        `).join("");
        return `
          <div class="${wideClass}" data-cluster-editor="1">
            <label>${escapeHtml(field.label)}</label>
            <div class="switch-row">
              <div class="switch-label">启用后，核心节点会按阶段向子节点分发脚本、权重和运行参数。</div>
              <input class="toggle" data-cluster-field="ENABLED" type="checkbox" ${cluster.ENABLED ? "checked" : ""} />
            </div>
            <div class="field-grid" style="margin-top:16px;">
              <div class="field-card">
                <label>MASTER_ADDR</label>
                <input type="text" data-cluster-field="MASTER_ADDR" value="${escapeHtml(cluster.MASTER_ADDR)}" placeholder="192.168.0.170" />
                <div class="field-help">这里填子节点可访问到的核心节点地址；核心节点本地会自动回落到回环地址启动。</div>
              </div>
              <div class="field-card">
                <label>MASTER_PORT</label>
                <input type="number" min="1" data-cluster-field="MASTER_PORT" value="${cluster.MASTER_PORT}" />
              </div>
              <div class="field-card">
                <label>LOCAL_NPUS_PER_NODE</label>
                <input type="number" min="0" data-cluster-field="LOCAL_NPUS_PER_NODE" value="${cluster.LOCAL_NPUS_PER_NODE}" />
                <div class="field-help">0 表示自动探测当前机器卡数。</div>
              </div>
            </div>
            <div class="section-block" style="margin-top: 18px;">
              <div class="section-title">
                <h3>子节点列表</h3>
                <span>每个子节点填写监听地址，顺序就是 rank 分配顺序</span>
              </div>
              <div class="field-grid" id="clusterSlaveList">
                ${slaveCards || '<div class="empty-state">还没有子节点，点击下方按钮添加。</div>'}
              </div>
              <button type="button" class="secondary-btn" data-cluster-add="1" style="margin-top:12px;">添加子节点</button>
            </div>
            <div class="section-block" style="margin-top: 18px;">
              <div class="section-title">
                <h3>高级监听配置</h3>
                <span>通常保持默认即可，主要给 <code>./lmsv slave</code> 服务使用</span>
              </div>
              <div class="field-grid">
                <div class="field-card">
                  <label>LISTEN_HOST</label>
                  <input type="text" data-cluster-field="LISTEN_HOST" value="${escapeHtml(cluster.LISTEN_HOST)}" />
                </div>
                <div class="field-card">
                  <label>LISTEN_PORT</label>
                  <input type="number" min="1" data-cluster-field="LISTEN_PORT" value="${cluster.LISTEN_PORT}" />
                </div>
                <div class="field-card">
                  <label>REQUEST_TIMEOUT</label>
                  <input type="number" min="1" data-cluster-field="REQUEST_TIMEOUT" value="${cluster.REQUEST_TIMEOUT}" />
                </div>
                <div class="field-card">
                  <label>SESSION_TIMEOUT</label>
                  <input type="number" min="1" data-cluster-field="SESSION_TIMEOUT" value="${cluster.SESSION_TIMEOUT}" />
                </div>
              </div>
            </div>
            ${field.help ? `<div class="field-help">${escapeHtml(field.help)}</div>` : ""}
          </div>
        `;
      }
      if (field.type === "task45_multinode") {
        const multiNode = normalizeTask45MultiNodeClient(value, { keepEmptyNodes: true });
        const nodeCards = multiNode.OTHER_NODES.map((node, index) => `
          <div class="field-card" style="border-style:dashed;" data-mn-node-row="${index}" data-mn-task="${taskId}">
            <label>从节点 ${index + 1}（NODE_RANK ${index + 1}）</label>
            <input type="text" data-mn-node-field="HOST" data-mn-index="${index}" data-mn-task="${taskId}" value="${escapeHtml(node.HOST || "")}" placeholder="root@192.168.0.203" />
            <input type="number" min="1" data-mn-node-field="SSH_PORT" data-mn-index="${index}" data-mn-task="${taskId}" value="${Number(node.SSH_PORT || 22)}" placeholder="22" style="margin-top:10px;" />
            <input type="text" data-mn-node-field="LMSV_PATH" data-mn-index="${index}" data-mn-task="${taskId}" value="${escapeHtml(node.LMSV_PATH || "")}" placeholder="/data/yd/lm-sv" style="margin-top:10px;" />
            <input type="text" data-mn-node-field="PTA_NAME" data-mn-index="${index}" data-mn-task="${taskId}" value="${escapeHtml(node.PTA_NAME || "")}" placeholder="mindspeed" style="margin-top:10px;" />
            <input type="text" data-mn-node-field="MSA_NAME" data-mn-index="${index}" data-mn-task="${taskId}" value="${escapeHtml(node.MSA_NAME || "")}" placeholder="msadapter" style="margin-top:10px;" />
            <input type="text" data-mn-node-field="PTA_PATH" data-mn-index="${index}" data-mn-task="${taskId}" value="${escapeHtml(node.PTA_PATH || "")}" placeholder="/data/pta" style="margin-top:10px;" />
            <input type="text" data-mn-node-field="MSA_PATH" data-mn-index="${index}" data-mn-task="${taskId}" value="${escapeHtml(node.MSA_PATH || "")}" placeholder="/data/msa" style="margin-top:10px;" />
            <div class="switch-row" style="margin-top:10px;">
              <div class="switch-label">该节点是否在容器内执行</div>
              <input class="toggle" type="checkbox" data-mn-node-field="HAS_CONTAINER" data-mn-index="${index}" data-mn-task="${taskId}" ${node.HAS_CONTAINER ? "checked" : ""} />
            </div>
            <input type="text" data-mn-node-field="CONTAINER_NAME" data-mn-index="${index}" data-mn-task="${taskId}" value="${escapeHtml(node.CONTAINER_NAME || "")}" placeholder="容器名（HAS_CONTAINER=true 时必填）" style="margin-top:10px;" />
            <button type="button" class="secondary-btn" data-mn-remove="${index}" data-mn-task="${taskId}" style="margin-top:10px;">删除从节点</button>
          </div>
        `).join("");
        return `
          <div class="${wideClass}" data-mn-editor="${taskId}">
            <label>${escapeHtml(field.label)}</label>
            <div class="switch-row">
              <div class="switch-label">启用后，Task${taskId} 会在 pta/msa load 阶段按主从并行执行。</div>
              <input class="toggle" data-mn-field="ENABLED" data-mn-task="${taskId}" type="checkbox" ${multiNode.ENABLED ? "checked" : ""} />
            </div>
            <div class="field-grid" style="margin-top:16px;">
              <div class="field-card">
                <label>MASTER_ADDR</label>
                <input type="text" data-mn-field="MASTER_ADDR" data-mn-task="${taskId}" value="${escapeHtml(multiNode.MASTER_ADDR)}" placeholder="127.0.0.1" />
              </div>
              <div class="field-card">
                <label>NNODES（总节点数）</label>
                <input type="number" min="2" data-mn-field="NNODES" data-mn-task="${taskId}" value="${multiNode.NNODES}" />
                <div class="field-help">保存时会按从节点数量自动校正为 OTHER_NODES + 1。</div>
              </div>
            </div>
            <div class="section-block" style="margin-top: 18px;">
              <div class="section-title">
                <h3>OTHER_NODES（从节点）</h3>
                <span>每个从节点依次对应 NODE_RANK 1..N-1</span>
              </div>
              <div class="field-grid">
                ${nodeCards || '<div class="empty-state">还没有从节点，点击下方按钮添加。</div>'}
              </div>
              <button type="button" class="secondary-btn" data-mn-add="1" data-mn-task="${taskId}" style="margin-top:12px;">添加从节点</button>
            </div>
            ${field.help ? `<div class="field-help">${escapeHtml(field.help)}</div>` : ""}
          </div>
        `;
      }
      if (field.type === "json") {
        return `
          <div class="${wideClass}">
            <label for="${inputId}">${escapeHtml(field.label)}</label>
            <textarea
              id="${inputId}"
              data-scope="${scope}"
              data-task="${taskId || ""}"
              data-key="${field.key}"
              placeholder="${escapeHtml(field.placeholder || "")}"
            >${escapeHtml(fieldValueToString(field, value))}</textarea>
            ${field.help ? `<div class="field-help">${escapeHtml(field.help)}</div>` : ""}
          </div>
        `;
      }
      if (field.type === "select") {
        const options = Array.isArray(field.options) ? field.options : [];
        const selectedValue = fieldValueToString(field, value) || field.placeholder || "";
        return `
          <div class="${wideClass}">
            <label for="${inputId}">${escapeHtml(field.label)}</label>
            <select
              id="${inputId}"
              data-scope="${scope}"
              data-task="${taskId || ""}"
              data-key="${field.key}"
            >
              ${options.map((option) => `
                <option value="${escapeHtml(option)}" ${selectedValue === option ? "selected" : ""}>${escapeHtml(option)}</option>
              `).join("")}
            </select>
            ${field.help ? `<div class="field-help">${escapeHtml(field.help)}</div>` : ""}
          </div>
        `;
      }
      return `
        <div class="${wideClass}">
          <label for="${inputId}">${escapeHtml(field.label)}</label>
          <input
            id="${inputId}"
            data-scope="${scope}"
            data-task="${taskId || ""}"
            data-key="${field.key}"
            type="${field.type === "number" ? "number" : "text"}"
            ${field.min !== undefined ? `min="${field.min}"` : ""}
            value="${escapeHtml(fieldValueToString(field, value))}"
            placeholder="${escapeHtml(field.placeholder || "")}"
          />
          ${field.help ? `<div class="field-help">${escapeHtml(field.help)}</div>` : ""}
        </div>
      `;
    }

    function renderConfig(config) {
      state.config = config;
      const nextTask = String(config.task_type || 1);
      if (!state.selectedTask || !FORM_SCHEMA.tasks[state.selectedTask]) {
        state.selectedTask = nextTask;
      }

      const switchNode = document.getElementById("taskSwitch");
      switchNode.innerHTML = Object.entries(TASK_META).map(([taskId, meta]) => `
        <div
          class="task-pill ${state.selectedTask === taskId ? "active" : ""}"
          data-task-switch="${taskId}"
          style="${state.selectedTask === taskId ? `box-shadow: 0 18px 34px ${meta.accent}22; border-color: ${meta.accent}55;` : ""}"
        >
          <div class="task-pill-title">${escapeHtml(meta.label)}</div>
          <div class="task-pill-copy">${escapeHtml(meta.tagline)}</div>
        </div>
      `).join("");

      const visibleGlobalFields = FORM_SCHEMA.global.filter((field) => {
        if (field.key === "CLUSTER" && (state.selectedTask === "4" || state.selectedTask === "5")) {
          return false;
        }
        return true;
      });
      document.getElementById("globalFields").innerHTML = visibleGlobalFields
        .map((field) => renderField("global", null, field, config[field.key]))
        .join("");

      const taskPanels = [];
      for (const [taskId, fields] of Object.entries(FORM_SCHEMA.tasks)) {
        const taskConfig = (config.tasks || {})[taskId] || {};
        const basicFields = fields.filter((field) => !field.advanced);
        const advancedFields = fields.filter((field) => field.advanced);
        taskPanels.push(`
          <div class="task-panel ${state.selectedTask === taskId ? "active" : ""}" data-task-panel="${taskId}">
            <div class="field-grid">
              ${basicFields.map((field) => renderField("task", taskId, field, taskConfig[field.key])).join("")}
            </div>
            ${advancedFields.length ? `
              <div class="section-block" style="margin-top: 18px;">
                <div class="section-title">
                  <h3>高级配置</h3>
                  <span>默认折叠，通常无需改动</span>
                </div>
                <button class="collapse-toggle" data-advanced-toggle="${taskId}" type="button">
                  ${state.advancedExpanded && state.selectedTask === taskId ? "收起高级配置" : "展开高级配置"}
                </button>
                <div class="field-grid" data-advanced-panel="${taskId}" style="display:${state.advancedExpanded && state.selectedTask === taskId ? "grid" : "none"};">
                  ${advancedFields.map((field) => renderField("task", taskId, field, taskConfig[field.key])).join("")}
                </div>
              </div>
            ` : ""}
          </div>
        `);
      }
      document.getElementById("taskPanels").innerHTML = taskPanels.join("");

      bindFieldEvents();
      bindTaskSwitchEvents();
      bindAdvancedToggleEvents();
      updateHero();
    }

    function updateHero() {
      const meta = TASK_META[state.selectedTask];
      document.getElementById("heroTaskLabel").textContent = meta.label;
      document.getElementById("heroTaskHint").textContent = meta.tagline;
      document.getElementById("heroTagline").textContent =
        "直接在浏览器里完成配置管理、任务启动和报告查看。参数字段与 CLI 保持一致，任务日志实时滚动，历史产物可从 output 目录快速回看。";
    }

    function bindTaskSwitchEvents() {
      document.querySelectorAll("[data-task-switch]").forEach((node) => {
        node.onclick = () => {
          state.selectedTask = node.dataset.taskSwitch;
          // 任务切换时重渲染，确保全局字段按任务动态显隐（如 Task4/5 隐藏 CLUSTER）。
          renderConfig(state.config);
        };
      });
    }

    function bindAdvancedToggleEvents() {
      document.querySelectorAll("[data-advanced-toggle]").forEach((node) => {
        node.onclick = () => {
          const targetTask = node.dataset.advancedToggle;
          if (state.selectedTask !== targetTask) {
            state.selectedTask = targetTask;
          }
          state.advancedExpanded = !(state.advancedExpanded && state.selectedTask === targetTask);
          renderConfig(state.config);
        };
      });
    }

    function bindFieldEvents() {
      document.querySelectorAll("[data-key]").forEach((node) => {
        node.oninput = () => clearMessage();
        node.onchange = () => clearMessage();
      });
      document.querySelectorAll("[data-cluster-field], [data-cluster-slave-field]").forEach((node) => {
        node.oninput = () => clearMessage();
        node.onchange = () => clearMessage();
      });
      document.querySelectorAll("[data-mn-field], [data-mn-node-field]").forEach((node) => {
        node.oninput = () => clearMessage();
        node.onchange = () => clearMessage();
      });
      bindClusterEditorEvents();
      bindTask45MultiNodeEditorEvents();
    }

    function readClusterFromDom(options = {}) {
      const keepEmptySlaves = Boolean(options.keepEmptySlaves);
      const editor = document.querySelector("[data-cluster-editor]");
      if (!editor) {
        return normalizeClusterClient((state.config || {}).CLUSTER, { keepEmptySlaves });
      }
      const readText = (key, fallback = "") => {
        const node = editor.querySelector(`[data-cluster-field="${key}"]`);
        return node ? String(node.value || "").trim() || fallback : fallback;
      };
      const readInt = (key, fallback) => {
        const node = editor.querySelector(`[data-cluster-field="${key}"]`);
        const value = Number.parseInt(node ? node.value : "", 10);
        return Number.isFinite(value) ? value : fallback;
      };
      const enabledNode = editor.querySelector('[data-cluster-field="ENABLED"]');
      const slaves = Array.from(editor.querySelectorAll("[data-cluster-slave-row]")).map((row) => {
        const index = row.getAttribute("data-cluster-slave-row");
        const endpoint = row.querySelector(`[data-cluster-slave-field="ENDPOINT"][data-cluster-index="${index}"]`);
        const label = row.querySelector(`[data-cluster-slave-field="LABEL"][data-cluster-index="${index}"]`);
        const npus = row.querySelector(`[data-cluster-slave-field="NPUS_PER_NODE"][data-cluster-index="${index}"]`);
        return {
          ENDPOINT: String(endpoint?.value || "").trim(),
          LABEL: String(label?.value || "").trim(),
          NPUS_PER_NODE: Number.parseInt(npus?.value || "0", 10) || 0,
        };
      });
      return normalizeClusterClient({
        ENABLED: Boolean(enabledNode?.checked),
        MASTER_ADDR: readText("MASTER_ADDR", "192.168.0.170"),
        MASTER_PORT: readInt("MASTER_PORT", 8118),
        LISTEN_HOST: readText("LISTEN_HOST", "0.0.0.0"),
        LISTEN_PORT: readInt("LISTEN_PORT", 19001),
        REQUEST_TIMEOUT: readInt("REQUEST_TIMEOUT", 30),
        SESSION_TIMEOUT: readInt("SESSION_TIMEOUT", 7200),
        LOCAL_NPUS_PER_NODE: readInt("LOCAL_NPUS_PER_NODE", 0),
        SLAVES: slaves,
      }, { keepEmptySlaves });
    }

    function bindClusterEditorEvents() {
      document.querySelectorAll("[data-cluster-add]").forEach((node) => {
        node.onclick = () => {
          const cluster = readClusterFromDom({ keepEmptySlaves: true });
          cluster.SLAVES.push({ ENDPOINT: "", LABEL: "", NPUS_PER_NODE: 0 });
          state.config.CLUSTER = cluster;
          renderConfig(state.config);
        };
      });
      document.querySelectorAll("[data-cluster-remove]").forEach((node) => {
        node.onclick = () => {
          const cluster = readClusterFromDom({ keepEmptySlaves: true });
          const index = Number.parseInt(node.dataset.clusterRemove || "-1", 10);
          cluster.SLAVES = cluster.SLAVES.filter((_, itemIndex) => itemIndex !== index);
          state.config.CLUSTER = cluster;
          renderConfig(state.config);
        };
      });
    }

    function readTask45MultiNodeFromDom(taskId, options = {}) {
      const keepEmptyNodes = Boolean(options.keepEmptyNodes);
      const editor = document.querySelector(`[data-mn-editor="${taskId}"]`);
      if (!editor) {
        return normalizeTask45MultiNodeClient((((state.config || {}).tasks || {})[taskId] || {}).MULTI_NODE, { keepEmptyNodes });
      }
      const readText = (key, fallback = "") => {
        const node = editor.querySelector(`[data-mn-field="${key}"][data-mn-task="${taskId}"]`);
        return node ? String(node.value || "").trim() || fallback : fallback;
      };
      const readInt = (key, fallback) => {
        const node = editor.querySelector(`[data-mn-field="${key}"][data-mn-task="${taskId}"]`);
        const value = Number.parseInt(node ? node.value : "", 10);
        return Number.isFinite(value) ? value : fallback;
      };
      const enabledNode = editor.querySelector(`[data-mn-field="ENABLED"][data-mn-task="${taskId}"]`);
      const nodes = Array.from(editor.querySelectorAll(`[data-mn-node-row][data-mn-task="${taskId}"]`)).map((row) => {
        const index = row.getAttribute("data-mn-node-row");
        const host = row.querySelector(`[data-mn-node-field="HOST"][data-mn-index="${index}"][data-mn-task="${taskId}"]`);
        const sshPort = row.querySelector(`[data-mn-node-field="SSH_PORT"][data-mn-index="${index}"][data-mn-task="${taskId}"]`);
        const lmsvPath = row.querySelector(`[data-mn-node-field="LMSV_PATH"][data-mn-index="${index}"][data-mn-task="${taskId}"]`);
        const ptaName = row.querySelector(`[data-mn-node-field="PTA_NAME"][data-mn-index="${index}"][data-mn-task="${taskId}"]`);
        const msaName = row.querySelector(`[data-mn-node-field="MSA_NAME"][data-mn-index="${index}"][data-mn-task="${taskId}"]`);
        const ptaPath = row.querySelector(`[data-mn-node-field="PTA_PATH"][data-mn-index="${index}"][data-mn-task="${taskId}"]`);
        const msaPath = row.querySelector(`[data-mn-node-field="MSA_PATH"][data-mn-index="${index}"][data-mn-task="${taskId}"]`);
        const hasContainer = row.querySelector(`[data-mn-node-field="HAS_CONTAINER"][data-mn-index="${index}"][data-mn-task="${taskId}"]`);
        const containerName = row.querySelector(`[data-mn-node-field="CONTAINER_NAME"][data-mn-index="${index}"][data-mn-task="${taskId}"]`);
        return {
          HOST: String(host?.value || "").trim(),
          SSH_PORT: Number.parseInt(sshPort?.value || "22", 10) || 22,
          LMSV_PATH: String(lmsvPath?.value || "").trim(),
          PTA_NAME: String(ptaName?.value || "").trim(),
          MSA_NAME: String(msaName?.value || "").trim(),
          PTA_PATH: String(ptaPath?.value || "").trim(),
          MSA_PATH: String(msaPath?.value || "").trim(),
          HAS_CONTAINER: Boolean(hasContainer?.checked),
          CONTAINER_NAME: String(containerName?.value || "").trim(),
        };
      });
      return normalizeTask45MultiNodeClient({
        ENABLED: Boolean(enabledNode?.checked),
        MASTER_ADDR: readText("MASTER_ADDR", "127.0.0.1"),
        NNODES: readInt("NNODES", Math.max(2, nodes.length + 1)),
        OTHER_NODES: nodes,
      }, { keepEmptyNodes });
    }

    function bindTask45MultiNodeEditorEvents() {
      document.querySelectorAll("[data-mn-add]").forEach((node) => {
        node.onclick = () => {
          const taskId = String(node.dataset.mnTask || "");
          if (!taskId) {
            return;
          }
          const multiNode = readTask45MultiNodeFromDom(taskId, { keepEmptyNodes: true });
          multiNode.OTHER_NODES.push({
            HOST: "",
            SSH_PORT: 22,
            LMSV_PATH: "",
            PTA_NAME: "",
            MSA_NAME: "",
            PTA_PATH: "",
            MSA_PATH: "",
            HAS_CONTAINER: false,
            CONTAINER_NAME: "",
          });
          state.config.tasks = state.config.tasks || {};
          state.config.tasks[taskId] = state.config.tasks[taskId] || {};
          state.config.tasks[taskId].MULTI_NODE = multiNode;
          renderConfig(state.config);
        };
      });
      document.querySelectorAll("[data-mn-remove]").forEach((node) => {
        node.onclick = () => {
          const taskId = String(node.dataset.mnTask || "");
          if (!taskId) {
            return;
          }
          const multiNode = readTask45MultiNodeFromDom(taskId, { keepEmptyNodes: true });
          const index = Number.parseInt(node.dataset.mnRemove || "-1", 10);
          multiNode.OTHER_NODES = multiNode.OTHER_NODES.filter((_, itemIndex) => itemIndex !== index);
          state.config.tasks = state.config.tasks || {};
          state.config.tasks[taskId] = state.config.tasks[taskId] || {};
          state.config.tasks[taskId].MULTI_NODE = multiNode;
          renderConfig(state.config);
        };
      });
    }

    function parseFieldValue(field, input, taskId = null) {
      if (field.type === "cluster") {
        return readClusterFromDom();
      }
      if (field.type === "task45_multinode") {
        if (!taskId) {
          throw new Error(`${field.label} 缺少任务上下文`);
        }
        return readTask45MultiNodeFromDom(taskId);
      }
      if (!input) {
        throw new Error(`${field.label} 对应的输入框不存在，请刷新页面后重试`);
      }
      if (field.type === "checkbox") {
        return Boolean(input.checked);
      }
      if (field.type === "json") {
        const raw = input.value.trim();
        if (!raw) {
          return {};
        }
        let parsed;
        try {
          parsed = JSON.parse(raw);
        } catch (error) {
          throw new Error(`${field.label} 需要是合法 JSON：${error.message}`);
        }
        if (parsed === null) {
          return {};
        }
        if (typeof parsed !== "object" || Array.isArray(parsed)) {
          throw new Error(`${field.label} 需要是 JSON 对象`);
        }
        return parsed;
      }
      if (field.type === "number") {
        const raw = input.value.trim();
        if (!raw) {
          throw new Error(`${field.label} 不能为空`);
        }
        const value = Number.parseInt(raw, 10);
        if (!Number.isFinite(value)) {
          throw new Error(`${field.label} 需要是整数`);
        }
        if (field.min !== undefined && value < field.min) {
          throw new Error(`${field.label} 不能小于 ${field.min}`);
        }
        return value;
      }
      if (field.type === "list_text") {
        const items = input.value
          .split(",")
          .map((item) => item.trim())
          .filter(Boolean);
        if (!items.length) {
          throw new Error(`${field.label} 至少需要一个值`);
        }
        return items;
      }
      if (field.type === "list_int") {
        const items = input.value
          .split(",")
          .map((item) => item.trim())
          .filter(Boolean);
        if (!items.length) {
          throw new Error(`${field.label} 至少需要一个值`);
        }
        const parsed = items.map((item) => {
          const value = Number.parseInt(item, 10);
          if (!Number.isFinite(value)) {
            throw new Error(`${field.label} 中包含非法整数：${item}`);
          }
          return value;
        });
        const invalid = parsed.find((value) => value < 0 || value > 10);
        if (invalid !== undefined) {
          throw new Error(`${field.label} 取值范围必须在 0~10 之间`);
        }
        return parsed;
      }
      const value = input.value.trim();
      if (!value) {
        throw new Error(`${field.label} 不能为空`);
      }
      return value;
    }

    function collectConfigFromForm() {
      const nextConfig = cloneJson(state.config || { tasks: {} });
      nextConfig.task_type = Number.parseInt(state.selectedTask, 10);
      nextConfig.tasks = nextConfig.tasks || {};

      FORM_SCHEMA.global.forEach((field) => {
        const input = document.querySelector(`[data-scope="global"][data-key="${field.key}"]`);
        nextConfig[field.key] = parseFieldValue(field, input, null);
      });

      Object.entries(FORM_SCHEMA.tasks).forEach(([taskId, fields]) => {
        nextConfig.tasks[taskId] = nextConfig.tasks[taskId] || {};
      });

      const activeFields = FORM_SCHEMA.tasks[state.selectedTask] || [];
      activeFields.forEach((field) => {
        const input = document.querySelector(`[data-scope="task"][data-task="${state.selectedTask}"][data-key="${field.key}"]`);
        nextConfig.tasks[state.selectedTask][field.key] = parseFieldValue(field, input, state.selectedTask);
      });

      if ((nextConfig.tasks["2"]?.MODELS || []).length !== (nextConfig.tasks["2"]?.SUBMODULES || []).length) {
        throw new Error("任务 2 中 MODELS 与 SUBMODULES 必须一一对应");
      }

      return nextConfig;
    }

    async function loadConfig() {
      const payload = await apiFetch(appUrl("api/config"));
      renderConfig(payload.config);
    }

    function renderResults(entries) {
      state.results = entries;
      const listNode = document.getElementById("resultList");
      if (!entries.length) {
        listNode.innerHTML = '<div class="empty-state">`output/` 目录里还没有任务结果。</div>';
        renderReportMeta(null);
        setReportFrame(null);
        updateResultActions();
        return;
      }

      if (!state.selectedResultId || !entries.some((entry) => entry.id === state.selectedResultId)) {
        state.selectedResultId = entries[0].id;
      }

      listNode.innerHTML = entries.map((entry) => `
        <div class="result-card ${entry.id === state.selectedResultId ? "active" : ""}" data-result-id="${entry.id}">
          <div class="result-head">
            <div>
              <div class="result-title">${escapeHtml(entry.id)}</div>
              <div class="badge task">${escapeHtml(entry.task_label)}</div>
            </div>
            <div class="badge ${entry.has_report ? "report" : entry.has_log ? "pending" : "warn"}">
              ${escapeHtml(entry.status_text)}
            </div>
          </div>
          <div class="result-copy">${escapeHtml(entry.summary)}</div>
          <div class="result-foot">${escapeHtml(entry.updated_at)}</div>
        </div>
      `).join("");

      document.querySelectorAll("[data-result-id]").forEach((node) => {
        node.onclick = () => selectResult(node.dataset.resultId);
      });

      selectResult(state.selectedResultId, true);
      updateResultActions();
    }

    function renderReportMeta(entry) {
      const metaNode = document.getElementById("reportMeta");
      if (!entry) {
        metaNode.innerHTML = "";
        return;
      }
      metaNode.innerHTML = `
        <div class="report-meta-card">
          <label>任务目录</label>
          <strong>${escapeHtml(entry.id)}</strong>
        </div>
        <div class="report-meta-card">
          <label>任务类型</label>
          <strong>${escapeHtml(entry.task_label)}</strong>
        </div>
        <div class="report-meta-card">
          <label>配置摘要</label>
          <strong>${escapeHtml(entry.summary)}</strong>
        </div>
        <div class="report-meta-card">
          <label>快捷入口</label>
          <a href="${appUrl(entry.config_url)}" target="_blank" rel="noreferrer">config.json</a><br />
          <a href="${appUrl(entry.log_url)}" target="_blank" rel="noreferrer">log.txt</a>
        </div>
      `;
    }

    function preferredViewMode(entry) {
      if (!entry) {
        return "report";
      }
      if (state.resultViewMode === "browser") {
        return "browser";
      }
      return entry.report_url ? "report" : "browser";
    }

    function updateViewerToolbar(entry) {
      const reportBtn = document.getElementById("reportModeBtn");
      const browserBtn = document.getElementById("browserModeBtn");
      const openRawBtn = document.getElementById("openRawBtn");
      const fullscreenBtn = document.getElementById("fullscreenPreviewBtn");
      const canReport = Boolean(entry && entry.report_url);
      const rawUrl = browserSelectedRawUrl(entry);

      reportBtn.disabled = !canReport;
      browserBtn.disabled = !entry;
      reportBtn.classList.toggle("active", state.resultViewMode === "report");
      browserBtn.classList.toggle("active", state.resultViewMode === "browser");
      openRawBtn.disabled = !rawUrl;
      openRawBtn.dataset.rawUrl = rawUrl || "";
      fullscreenBtn.disabled = !(state.resultViewMode === "browser" && state.currentPreview);
    }

    function browserSelectedRawUrl(entry) {
      if (state.resultViewMode === "browser" && state.browserSelectedFile?.raw_url) {
        return appUrl(state.browserSelectedFile.raw_url);
      }
      if (state.resultViewMode === "report" && entry?.report_url) {
        return appUrl(entry.report_url);
      }
      return null;
    }

    function setReportFrame(entry, forceReload = false) {
      const frame = document.getElementById("reportFrame");
      const empty = document.getElementById("reportEmpty");
      const browserShell = document.getElementById("browserShell");
      const mode = preferredViewMode(entry);
      state.resultViewMode = mode;
      updateViewerToolbar(entry);

      if (mode === "browser") {
        frame.style.display = "none";
        frame.removeAttribute("src");
        delete frame.dataset.baseSrc;
        empty.style.display = "none";
        browserShell.style.display = "grid";
        if (entry) {
          loadFileBrowser(state.browserPath || "", state.browserSelectedFile?.path || null).catch((error) => {
            showMessage(`文件目录加载失败：${error.message}`, "error");
          });
        }
        return;
      }

      browserShell.style.display = "none";
      if (entry && entry.report_url) {
        const nextSrc = appUrl(entry.report_url);
        frame.style.display = "block";
        empty.style.display = "none";
        const currentBase = frame.dataset.baseSrc || "";
        if (forceReload || currentBase !== nextSrc) {
          frame.dataset.baseSrc = nextSrc;
          frame.src = `${nextSrc}?t=${Date.now()}`;
        }
      } else {
        frame.style.display = "none";
        frame.removeAttribute("src");
        delete frame.dataset.baseSrc;
        browserShell.style.display = "none";
        empty.style.display = "grid";
        empty.textContent = entry
          ? (entry.can_regenerate
              ? "该任务还没有生成 analysis/report.html，可先查看 config.json 或 log.txt，也可以手动重新生成 analysis。"
              : "该任务还没有生成 analysis/report.html，可以先查看 config.json 或 log.txt。")
          : "从左侧选择一个任务，即可在这里加载 analysis/report.html。";
      }
    }

    function selectedResultEntry() {
      return state.results.find((item) => item.id === state.selectedResultId) || null;
    }

    function updateResultActions() {
      const regenerateButton = document.getElementById("regenerateAnalysisBtn");
      const deleteButton = document.getElementById("deleteResultBtn");
      if (!regenerateButton || !deleteButton) {
        return;
      }

      const entry = selectedResultEntry();
      const activeOutputId = state.runStatus?.active ? state.runStatus.output_dir : null;
      const isCurrentRun = Boolean(entry && activeOutputId && entry.id === activeOutputId);
      const isBusy = state.regeneratingResultId !== null || state.deletingResultId !== null;

      regenerateButton.disabled = !entry || !entry.can_regenerate || isBusy || isCurrentRun;
      if (!entry) {
        regenerateButton.textContent = "重新生成 analysis";
      } else if (isBusy) {
        regenerateButton.textContent = state.regeneratingResultId ? "正在生成..." : "重新生成 analysis";
      } else if (isCurrentRun) {
        regenerateButton.textContent = "运行中不可生成";
      } else if (!entry.can_regenerate) {
        regenerateButton.textContent = "无旧数据可生成";
      } else {
        regenerateButton.textContent = "重新生成 analysis";
      }

      deleteButton.disabled = !entry || isBusy || isCurrentRun;
      if (!entry) {
        deleteButton.textContent = "删除结果";
      } else if (state.deletingResultId) {
        deleteButton.textContent = "正在删除...";
      } else if (isCurrentRun) {
        deleteButton.textContent = "运行中不可删除";
      } else {
        deleteButton.textContent = "删除结果";
      }
    }

    function describeEntryKind(entry) {
      if (entry.is_dir) {
        return "目录";
      }
      const mapping = {
        html: "HTML",
        text: "文本",
        image: "图片",
        pdf: "PDF",
        download: "文件",
      };
      return mapping[entry.preview_kind] || "文件";
    }

    function isIterDirectoryPath(path) {
      return new RegExp("^iters/iter_[^/]+$").test(String(path || ""));
    }

    function buildIterDownloadUrl(outputId, relativePath) {
      return appUrl(`api/results/iter-archive?output_id=${encodeURIComponent(outputId)}&path=${encodeURIComponent(relativePath)}`);
    }

    function updateBrowserDownloadAction(entryOverride = null, outputIdOverride = null) {
      const button = document.getElementById("downloadCurrentIterBtn");
      if (!button) {
        return;
      }
      const entry = entryOverride || selectedResultEntry();
      const outputId = outputIdOverride || entry?.id;
      const canDownload = Boolean(outputId && isIterDirectoryPath(state.browserPath));
      button.disabled = !canDownload;
      button.dataset.downloadUrl = canDownload ? buildIterDownloadUrl(outputId, state.browserPath) : "";
      button.textContent = canDownload ? "下载当前 iter ZIP" : "下载当前 iter ZIP";
    }

    function renderBrowser(payload) {
      state.browserPath = payload.path || "";
      state.browserEntries = Array.isArray(payload.entries) ? payload.entries : [];

      const crumbsNode = document.getElementById("browserBreadcrumbs");
      const listNode = document.getElementById("browserList");
      document.getElementById("browserDirHint").textContent = state.browserPath
        ? `当前目录 output/${payload.output_id}/${state.browserPath}`
        : `当前目录 output/${payload.output_id}`;
      updateBrowserDownloadAction(selectedResultEntry(), payload.output_id);

      crumbsNode.innerHTML = (payload.breadcrumbs || []).map((crumb) => `
        <a class="crumb" data-browse-path="${escapeHtml(crumb.path)}">${escapeHtml(crumb.name)}</a>
      `).join("");

      if (payload.parent_path !== null && payload.parent_path !== undefined) {
        crumbsNode.innerHTML += `<a class="crumb" data-browse-path="${escapeHtml(payload.parent_path)}">..</a>`;
      }

      document.querySelectorAll("[data-browse-path]").forEach((node) => {
        node.onclick = () => {
          loadFileBrowser(node.dataset.browsePath || "", null).catch((error) => {
            showMessage(`目录加载失败：${error.message}`, "error");
          });
        };
      });

      if (!state.browserEntries.length) {
        listNode.innerHTML = '<div class="empty-state">当前目录下暂无文件。</div>';
        renderFilePreview(null);
        return;
      }

      const hasSelected = state.browserSelectedFile
        && state.browserEntries.some((item) => !item.is_dir && item.path === state.browserSelectedFile.path);
      if (!hasSelected) {
        const preferred = state.browserEntries.find((item) => !item.is_dir && item.name === "report.html")
          || state.browserEntries.find((item) => !item.is_dir)
          || null;
        state.browserSelectedFile = preferred;
      }

      listNode.innerHTML = state.browserEntries.map((entry) => `
        <div class="browser-entry-row">
          <button
            class="browser-entry ${!entry.is_dir && state.browserSelectedFile?.path === entry.path ? "active" : ""}"
            data-browser-entry="${escapeHtml(entry.path)}"
            type="button"
          >
            <div>
              <div class="browser-entry-name">${escapeHtml(entry.is_dir ? `/${entry.name}` : entry.name)}</div>
              <div class="browser-entry-meta">${escapeHtml(entry.updated_at)}${entry.is_dir ? "" : ` · ${entry.size_text}`}</div>
            </div>
            <div class="browser-entry-kind">${escapeHtml(describeEntryKind(entry))}</div>
          </button>
          ${entry.is_dir && isIterDirectoryPath(entry.path) ? `<button class="btn-secondary" data-iter-download="${escapeHtml(entry.path)}" type="button">下载 ZIP</button>` : ""}
        </div>
      `).join("");

      document.querySelectorAll("[data-browser-entry]").forEach((node) => {
        node.onclick = async () => {
          const target = state.browserEntries.find((item) => item.path === node.dataset.browserEntry);
          if (!target) {
            return;
          }
          if (target.is_dir) {
            state.browserSelectedFile = null;
            await loadFileBrowser(target.path, null);
            return;
          }
          state.browserSelectedFile = target;
          renderBrowser(payload);
          await renderFilePreview(target);
        };
      });

      document.querySelectorAll("[data-iter-download]").forEach((node) => {
        node.onclick = (event) => {
          event.stopPropagation();
          const outputId = selectedResultEntry()?.id;
          const relativePath = node.dataset.iterDownload;
          if (!outputId || !relativePath) {
            showMessage("当前 iter 下载信息不完整。", "error");
            return;
          }
          window.open(buildIterDownloadUrl(outputId, relativePath), "_blank", "noopener,noreferrer");
        };
      });

      renderFilePreview(state.browserSelectedFile);
      updateViewerToolbar(selectedResultEntry());
    }

    function resetFilePreview() {
      closePreviewOverlay();
      document.getElementById("previewTitle").textContent = "文件预览";
      document.getElementById("previewCopy").textContent = "选择一个文件即可在线预览";
      document.getElementById("previewMeta").innerHTML = "";
      document.getElementById("filePreviewEmpty").style.display = "grid";
      document.getElementById("textPreview").hidden = true;
      document.getElementById("textPreview").textContent = "";
      document.getElementById("markdownPreview").hidden = true;
      document.getElementById("markdownPreview").innerHTML = "";
      document.getElementById("csvPreview").hidden = true;
      document.getElementById("csvPreview").innerHTML = "";
      const image = document.getElementById("imagePreview");
      image.hidden = true;
      image.onerror = null;
      image.onload = null;
      image.removeAttribute("src");
      const frame = document.getElementById("filePreviewFrame");
      frame.style.display = "none";
      frame.removeAttribute("src");
      delete frame.dataset.baseSrc;
      setCurrentPreview(null);
    }

    async function renderFilePreview(entry) {
      const requestToken = ++state.previewRequestToken;
      resetFilePreview();
      if (!entry) {
        updateViewerToolbar(selectedResultEntry());
        return;
      }

      document.getElementById("previewTitle").textContent = entry.name;
      document.getElementById("previewCopy").textContent = entry.path;
      document.getElementById("previewMeta").innerHTML = `
        <div class="preview-meta-card">
          <label>类型</label>
          <strong>${escapeHtml(describeEntryKind(entry))}</strong>
        </div>
        <div class="preview-meta-card">
          <label>大小</label>
          <strong>${escapeHtml(entry.size_text || "-")}</strong>
        </div>
        <div class="preview-meta-card">
          <label>原文件</label>
          <a href="${appUrl(entry.raw_url)}" target="_blank" rel="noreferrer">打开新窗口</a>
        </div>
      `;

      const empty = document.getElementById("filePreviewEmpty");
      if (entry.preview_kind === "html" || entry.preview_kind === "pdf") {
        const frame = document.getElementById("filePreviewFrame");
        frame.style.display = "block";
        frame.dataset.baseSrc = appUrl(entry.raw_url);
        frame.src = `${appUrl(entry.raw_url)}?t=${Date.now()}`;
        empty.style.display = "none";
        setCurrentPreview({
          mode: "frame",
          title: entry.name,
          subtitle: entry.path,
          src: frame.src,
        });
        updateViewerToolbar(selectedResultEntry());
        return;
      }
      if (entry.preview_kind === "image") {
        const image = document.getElementById("imagePreview");
        image.onload = () => {
          if (requestToken !== state.previewRequestToken) {
            return;
          }
          empty.style.display = "none";
          setCurrentPreview({
            mode: "image",
            title: entry.name,
            subtitle: entry.path,
            src: image.src,
          });
        };
        image.onerror = () => {
          if (requestToken !== state.previewRequestToken) {
            return;
          }
          image.hidden = true;
          image.removeAttribute("src");
          empty.style.display = "grid";
          empty.textContent = "图片预览加载失败，请使用“打开原文件”。";
        };
        image.hidden = false;
        image.src = `${appUrl(entry.raw_url)}?t=${Date.now()}`;
        updateViewerToolbar(selectedResultEntry());
        return;
      }
      if (["text", "markdown", "csv"].includes(entry.preview_kind)) {
        const payload = await apiFetch(appUrl(`api/results/file-preview?output_id=${encodeURIComponent(selectedResultEntry().id)}&path=${encodeURIComponent(entry.path)}`));
        if (requestToken !== state.previewRequestToken) {
          return;
        }
        if (payload.preview_kind === "markdown") {
          const markdownNode = document.getElementById("markdownPreview");
          markdownNode.hidden = false;
          markdownNode.innerHTML = renderMarkdown(payload.content || "");
          if (payload.truncated) {
            markdownNode.innerHTML += '<p><em>[webui] 文件较大，仅渲染前 500000 字节。</em></p>';
          }
          setCurrentPreview({
            mode: "html",
            title: entry.name,
            subtitle: entry.path,
            html: `<div class="markdown-preview">${markdownNode.innerHTML}</div>`,
          });
        } else if (payload.preview_kind === "csv") {
          const csvNode = document.getElementById("csvPreview");
          csvNode.hidden = false;
          csvNode.innerHTML = buildCsvTable(payload);
          setCurrentPreview({
            mode: "html",
            title: entry.name,
            subtitle: entry.path,
            html: `<div class="csv-preview-shell">${csvNode.innerHTML}</div>`,
          });
        } else {
          const textNode = document.getElementById("textPreview");
          textNode.hidden = false;
          textNode.textContent = payload.truncated
            ? `${payload.content}\n\n[webui] 文件较大，仅显示前 500000 字节。`
            : (payload.content || "");
          setCurrentPreview({
            mode: "text",
            title: entry.name,
            subtitle: entry.path,
            text: textNode.textContent,
          });
        }
        empty.style.display = "none";
        updateViewerToolbar(selectedResultEntry());
        return;
      }

      empty.style.display = "grid";
      empty.textContent = "该文件类型暂不支持内联预览，请使用“打开原文件”。";
      updateViewerToolbar(selectedResultEntry());
    }

    async function loadFileBrowser(path = "", preferredFilePath = null) {
      const entry = selectedResultEntry();
      if (!entry) {
        resetFilePreview();
        return;
      }
      const payload = await apiFetch(appUrl(`api/results/files?output_id=${encodeURIComponent(entry.id)}&path=${encodeURIComponent(path || "")}`));
      if (preferredFilePath) {
        const preferred = (payload.entries || []).find((item) => !item.is_dir && item.path === preferredFilePath);
        state.browserSelectedFile = preferred || null;
      } else {
        state.browserSelectedFile = null;
      }
      renderBrowser(payload);
    }

    function selectResult(resultId, silent = false, forceReload = false) {
      state.selectedResultId = resultId;
      state.browserPath = "";
      state.browserEntries = [];
      state.browserSelectedFile = null;
      const entry = selectedResultEntry();
      document.querySelectorAll("[data-result-id]").forEach((node) => {
        node.classList.toggle("active", node.dataset.resultId === resultId);
      });
      renderReportMeta(entry);
      setReportFrame(entry, forceReload);
      updateResultActions();
      if (!silent && entry) {
        showMessage(`已切换到结果目录 ${entry.id}`, "info");
      }
    }

    async function refreshResults(preferredId = null) {
      const payload = await apiFetch(appUrl("api/results"));
      if (preferredId) {
        state.selectedResultId = preferredId;
      }
      renderResults(payload.results || []);
    }

    function humanRunState(payload) {
      const mapping = {
        idle: "空闲",
        running: "运行中",
        starting: "启动中",
        stopping: "停止中",
        stopped: "已停止",
        finished: "已完成",
        failed: "执行失败",
      };
      return mapping[payload.state] || payload.state || "未知";
    }

    function formatTimestamp(value) {
      if (!value) {
        return "-";
      }
      return value;
    }

    function updateHardwareVisibility() {
      const listNode = document.getElementById("hardwareDeviceList");
      const toggleBtn = document.getElementById("hardwareToggleBtn");
      listNode.hidden = !state.hardwareExpanded;
      toggleBtn.textContent = state.hardwareExpanded ? "收起 NPU 详情" : "展开 NPU 详情";
    }

    function renderHardware(payload) {
      const hardware = payload.hardware || {};
      const cpu = hardware.cpu || {};
      const memory = hardware.memory || {};
      const devices = Array.isArray(hardware.accelerators) ? hardware.accelerators : [];
      const sources = Array.isArray(hardware.sources) ? hardware.sources : [];

      document.getElementById("hardwareUpdatedAt").textContent = hardware.updated_at
        ? `最近采样 ${hardware.updated_at}`
        : "等待首次采样";

      document.getElementById("hardwareCpuValue").textContent =
        cpu.usage_percent === null || cpu.usage_percent === undefined
          ? "采样中"
          : `${cpu.usage_percent}%`;
      document.getElementById("hardwareCpuFoot").textContent =
        `负载 ${Array.isArray(cpu.loadavg) ? cpu.loadavg.join(" / ") : "-"} · ${cpu.cores || 0} 核`;

      document.getElementById("hardwareMemoryValue").textContent =
        memory.usage_percent === null || memory.usage_percent === undefined
          ? (memory.display || "-")
          : `${memory.usage_percent}%`;
      document.getElementById("hardwareMemoryFoot").textContent = memory.display || "-";

      document.getElementById("hardwareAccelValue").textContent = hardware.accelerator_summary || "未检测到 NPU";
      document.getElementById("hardwareAccelFoot").textContent =
        sources.length ? `来源：${sources.join(" / ")}` : "未检测到 npu-smi";

      const deviceList = document.getElementById("hardwareDeviceList");
      if (!devices.length) {
        deviceList.innerHTML = '<div class="empty-state">未检测到 NPU 设备，或 `npu-smi info` 当前不可用。</div>';
      } else {
        deviceList.innerHTML = devices.map((device) => `
          <div class="device-card">
            <div class="device-title">
              <span>${escapeHtml(device.name || `NPU ${device.index || "-"}`)} · NPU ${escapeHtml(device.index || "-")} / Chip ${escapeHtml(device.chip_id || "-")}</span>
              <span>健康状态 ${escapeHtml(device.health || "-")}</span>
            </div>
            <div class="device-metrics">
              <div class="device-metric">
                <label>Bus-Id</label>
                <strong>${escapeHtml(device.bus_id || "-")}</strong>
              </div>
              <div class="device-metric">
                <label>AICore</label>
                <strong>${escapeHtml(device.utilization || "-")}</strong>
              </div>
              <div class="device-metric">
                <label>Memory / HBM</label>
                <strong>${escapeHtml(`${device.memory_usage || "-"} / ${device.hbm_usage || "-"}`)}</strong>
              </div>
              <div class="device-metric">
                <label>温度 / 功耗</label>
                <strong>${escapeHtml(`${device.temperature || "-"} / ${device.power || "-"}`)}</strong>
              </div>
            </div>
            <div class="stat-card-foot" style="margin-top: 10px;">
              Hugepages ${escapeHtml(device.hugepages || "-")}
              ${Array.isArray(device.processes) && device.processes.length
                ? ` · 进程 ${device.processes.map((proc) => `${proc.name}(${proc.pid}, ${proc.memory}MB)`).join(", ")}`
                : " · 当前无进程信息"}
            </div>
          </div>
        `).join("");
      }
      updateHardwareVisibility();
    }

    function updateRunCards(payload) {
      state.runStatus = payload;
      const stateText = humanRunState(payload);
      document.getElementById("heroRunState").textContent = stateText;
      document.getElementById("heroOutputDir").textContent = payload.output_dir
        ? `output/${payload.output_dir}`
        : "尚未创建 output 目录";

      document.getElementById("runStateCard").textContent = stateText;
      document.getElementById("runStateFoot").textContent = payload.active
        ? `启动时间 ${formatTimestamp(payload.started_at)}`
        : `结束时间 ${formatTimestamp(payload.ended_at)}`;
      document.getElementById("runPidCard").textContent = payload.pid ? String(payload.pid) : "-";
      document.getElementById("runTimeCard").textContent = payload.active
        ? `已运行 ${payload.elapsed || "0s"}`
        : "当前无活动进程";
      document.getElementById("runOutputCard").textContent = payload.output_dir || "-";
      document.getElementById("runReturnCard").textContent = payload.returncode === null
        ? "等待结束"
        : `returncode = ${payload.returncode}`;
      document.getElementById("runCursorCard").textContent = String(payload.next_cursor || 0);
      document.getElementById("runCursorFoot").textContent = payload.reset
        ? "日志缓存已重置，本次为全量同步"
        : "增量拉取正常";
      document.getElementById("runHintText").textContent = payload.active
        ? "日志轮询中"
        : "日志轮询空闲";

      document.getElementById("startRunBtn").disabled = Boolean(payload.active);
      document.getElementById("stopRunBtn").disabled = !payload.active;
      renderHardware(payload);
      updateResultActions();
    }

    function appendLogs(payload) {
      const viewer = document.getElementById("logViewer");
      if (payload.reset) {
        viewer.textContent = payload.logs || "";
      } else if (payload.logs) {
        viewer.textContent += payload.logs;
      }
      if (!viewer.textContent) {
        viewer.textContent = "等待任务启动…";
      }
      if (document.getElementById("autoScrollToggle").checked) {
        viewer.scrollTop = viewer.scrollHeight;
      }
    }

    async function refreshRunStatus() {
      try {
        const payload = await apiFetch(appUrl(`api/run/status?cursor=${state.runCursor}`));
        appendLogs(payload);
        updateRunCards(payload);
        state.runCursor = payload.next_cursor || 0;

        if (payload.active) {
          state.lastTerminalRefreshKey = null;
        }

        if (payload.output_dir && payload.output_dir !== state.lastAutoSelectedOutput) {
          state.lastAutoSelectedOutput = payload.output_dir;
          await refreshResults(payload.output_dir);
        }

        const terminalRefreshKey =
          !payload.active && (payload.state === "finished" || payload.state === "failed" || payload.state === "stopped")
            ? `${payload.state}:${payload.output_dir || "-"}:${payload.returncode ?? "null"}`
            : null;

        if (terminalRefreshKey && terminalRefreshKey !== state.lastTerminalRefreshKey) {
          state.lastTerminalRefreshKey = terminalRefreshKey;
          await refreshResults(payload.output_dir || state.selectedResultId);
        }
      } catch (error) {
        document.getElementById("runHintText").textContent = `状态拉取失败：${error.message}`;
      }
    }

    async function saveConfig(runAfterSave = false) {
      try {
        const config = collectConfigFromForm();
        const endpoint = runAfterSave ? appUrl("api/run/start") : appUrl("api/config");
        const payload = await apiFetch(endpoint, {
          method: "POST",
          body: JSON.stringify(config),
        });
        renderConfig(payload.config);
        if (runAfterSave) {
          state.runCursor = 0;
          state.lastTerminalRefreshKey = null;
          document.getElementById("logViewer").textContent = "";
          showMessage("配置已保存，任务已启动。", "success");
          await refreshRunStatus();
        } else {
          showMessage("配置已保存。", "success");
        }
      } catch (error) {
        showMessage(error.message, "error");
      }
    }

    async function stopRun() {
      try {
        const payload = await apiFetch(appUrl("api/run/stop"), { method: "POST", body: "{}" });
        showMessage(payload.message || "已发送停止请求。", "warn");
        await refreshRunStatus();
      } catch (error) {
        showMessage(error.message, "error");
      }
    }

    async function regenerateSelectedAnalysis() {
      const entry = selectedResultEntry();
      if (!entry) {
        showMessage("请先从左侧选择一个 output 目录。", "warn");
        return;
      }
      if (!entry.can_regenerate) {
        showMessage(`output/${entry.id} 缺少可分析的迭代数据，无法重新生成 analysis。`, "error");
        return;
      }

      state.regeneratingResultId = entry.id;
      updateResultActions();
      showMessage(`正在重新生成 output/${entry.id} 的 analysis...`, "info");

      try {
        const payload = await apiFetch(appUrl("api/results/analysis/regenerate"), {
          method: "POST",
          body: JSON.stringify({ output_id: entry.id }),
        });
        await refreshResults(entry.id);
        selectResult(entry.id, true, true);
        showMessage(`output/${entry.id} 的 analysis 已重新生成。`, "success");
      } catch (error) {
        showMessage(error.message, "error");
      } finally {
        state.regeneratingResultId = null;
        updateResultActions();
      }
    }

    async function deleteSelectedResult() {
      const entry = selectedResultEntry();
      if (!entry) {
        showMessage("请先从左侧选择一个 output 目录。", "warn");
        return;
      }

      if (!window.confirm(`确认删除 output/${entry.id} 吗？该操作会永久删除该次任务的所有历史产物。`)) {
        return;
      }

      state.deletingResultId = entry.id;
      updateResultActions();
      showMessage(`正在删除 output/${entry.id}...`, "warn");

      try {
        await apiFetch(appUrl("api/results/delete"), {
          method: "POST",
          body: JSON.stringify({ output_id: entry.id }),
        });
        await refreshResults();
        showMessage(`output/${entry.id} 已删除。`, "success");
      } catch (error) {
        showMessage(error.message, "error");
      } finally {
        state.deletingResultId = null;
        updateResultActions();
      }
    }

    async function init() {
      document.getElementById("saveConfigBtn").onclick = () => saveConfig(false);
      document.getElementById("startRunBtn").onclick = () => saveConfig(true);
      document.getElementById("deleteResultBtn").onclick = () => deleteSelectedResult();
      document.getElementById("regenerateAnalysisBtn").onclick = () => regenerateSelectedAnalysis();
      document.getElementById("refreshResultsBtn").onclick = () => refreshResults(state.selectedResultId);
      document.getElementById("refreshStatusBtn").onclick = () => refreshRunStatus();
      document.getElementById("stopRunBtn").onclick = () => stopRun();
      document.getElementById("reportModeBtn").onclick = () => {
        const entry = selectedResultEntry();
        if (!entry?.report_url) {
          return;
        }
        state.resultViewMode = "report";
        setReportFrame(entry, true);
      };
      document.getElementById("browserModeBtn").onclick = () => {
        const entry = selectedResultEntry();
        if (!entry) {
          return;
        }
        state.resultViewMode = "browser";
        setReportFrame(entry, true);
      };
      document.getElementById("openRawBtn").onclick = () => {
        const rawUrl = document.getElementById("openRawBtn").dataset.rawUrl;
        if (rawUrl) {
          window.open(rawUrl, "_blank", "noopener,noreferrer");
        }
      };
      document.getElementById("downloadCurrentIterBtn").onclick = () => {
        const downloadUrl = document.getElementById("downloadCurrentIterBtn").dataset.downloadUrl;
        if (downloadUrl) {
          window.open(downloadUrl, "_blank", "noopener,noreferrer");
        }
      };
      document.getElementById("fullscreenPreviewBtn").onclick = () => openPreviewOverlay();
      document.getElementById("closePreviewOverlayBtn").onclick = () => closePreviewOverlay();
      document.getElementById("previewOverlay").onclick = (event) => {
        if (event.target.id === "previewOverlay") {
          closePreviewOverlay();
        }
      };
      document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
          closePreviewOverlay();
        }
      });
      document.getElementById("hardwareToggleBtn").onclick = () => {
        state.hardwareExpanded = !state.hardwareExpanded;
        updateHardwareVisibility();
      };
      updateHardwareVisibility();

      try {
        await loadConfig();
      } catch (error) {
        renderConfig(cloneJson({
          task_type: 1,
          PTA_NAME: "mindspeed",
          MSA_NAME: "msadapter",
          MF_NAME: "mindf_py311",
          PTA_PATH: "",
          MSA_PATH: "",
          SAVE_ABNORMAL_WEIGHTS: true,
          tasks: {
            "1": {
              MODEL_NAME: "qwen2",
              TOTAL_ITER: 10,
              PTA_MAX_RUNTIME: 3000,
              MSA_MAX_RUNTIME: 3000,
              LOG_INIT_WAIT: 240,
              LOG_STABLE_THRESHOLD: 150,
              COMPARE_MODE: "pta_msa",
              BASE_SEED: 43,
              MUTNM: 2,
              SAVE_STEPS: 1,
              LOAD_STEPS: 30,
              MULTI_NODE: {
                ENABLED: false
              }
            },
            "2": {
              MODELS: ["qwen2", "qwen2", "qwen2"],
              TOTAL_ITER: 100,
              PTA_MAX_RUNTIME: 3000,
              MSA_MAX_RUNTIME: 3000,
              LOG_INIT_WAIT: 240,
              LOG_STABLE_THRESHOLD: 150,
              BASE_SEED: 43,
              SUBMODULES: [3, 4, 5],
              MUTNM: 2,
              SAVE_STEPS: 1,
              LOAD_STEPS: 15,
              COMPARE_MODE: "pta_msa",
              MULTI_NODE: {
                ENABLED: false
              }
            },
            "3": {
              MODELS: ["qwen2", "glm4"],
              TOTAL_ITER: 100,
              PTA_MAX_RUNTIME: 3000,
              MSA_MAX_RUNTIME: 3000,
              LOG_INIT_WAIT: 240,
              LOG_STABLE_THRESHOLD: 150,
              MAX_MUTATION_WAIT: 600,
              BASE_SEED: 43,
              MUTNM: 2,
              SAVE_STEPS: 1,
              LOAD_STEPS: 15,
              COMPARE_MODE: "pta_msa",
              MULTI_NODE: {
                ENABLED: false
              }
            },
            "4": {
              TOTAL_ITER: 10,
              COMPARE_MODE: "pta_msa",
              PTA_MAX_RUNTIME: 3000,
              MSA_MAX_RUNTIME: 3000,
              LOG_INIT_WAIT: 240,
              LOG_STABLE_THRESHOLD: 150,
              SAVE_STEPS: 1,
              RUN_STEPS: 20,
              MULTI_NODE: {
                ENABLED: false
              },
              ENABLE_MF_WEIGHT_LOAD: false
            },
            "5": {
              TOTAL_ITER: 10,
              COMPARE_MODE: "pta_msa",
              PTA_MAX_RUNTIME: 3000,
              MSA_MAX_RUNTIME: 3000,
              LOG_INIT_WAIT: 240,
              LOG_STABLE_THRESHOLD: 150,
              SAVE_STEPS: 1,
              RUN_STEPS: 20,
              MUTATE_STEPS: 10,
              MODULE_TYPE: "all",
              MULTI_NODE: {
                ENABLED: false
              },
              ENABLE_MF_WEIGHT_LOAD: false
            },
            "6": {
              MODEL_NAME: "internvl3",
              TOTAL_ITER: 10,
              MUTNM: 2,
              COMPARE_MODE: "pta_msa",
              TRAIN_ITERS: 5,
              PTA_MAX_RUNTIME: 900,
              MSA_MAX_RUNTIME: 900
            }
          }
        }));
        showMessage(`配置加载失败，已回退到默认值：${error.message}`, "warn");
      }

      try {
        await refreshResults();
      } catch (error) {
        showMessage(`结果列表加载失败：${error.message}`, "error");
      }

      try {
        await refreshRunStatus();
      } catch (error) {
        showMessage(`运行状态加载失败：${error.message}`, "error");
      }

      state.pollTimer = window.setInterval(refreshRunStatus, 1500);
    }

    window.addEventListener("error", (event) => {
      if (event?.message) {
        showMessage(`前端错误：${event.message}`, "error");
      }
    });

    window.addEventListener("DOMContentLoaded", init);
  </script>
</body>
</html>
"""


def deep_merge(base, override):
    if isinstance(base, dict) and isinstance(override, dict):
        merged = deepcopy(base)
        for key, value in override.items():
            if key in merged:
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = deepcopy(value)
        return merged
    return deepcopy(override)


def load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _ensure_string(value, field_name: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"{field_name} 不能为空")
    return text


def _ensure_int(value, field_name: str, *, min_value: int | None = None, max_value: int | None = None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} 需要是整数")
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} 需要是整数") from None
    if min_value is not None and number < min_value:
        raise ValueError(f"{field_name} 不能小于 {min_value}")
    if max_value is not None and number > max_value:
        raise ValueError(f"{field_name} 不能大于 {max_value}")
    return number


def _ensure_bool(value, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"{field_name} 需要是布尔值")


def _ensure_choice(value, field_name: str, options: tuple[str, ...]) -> str:
    text = str(value).strip().lower() if value is not None else ""
    if text in options:
        return text
    opts = "/".join(options)
    raise ValueError(f"{field_name} 只能是 {opts}")


def _ensure_string_list(value, field_name: str) -> list[str]:
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple)):
        items = [str(item).strip() for item in value if str(item).strip()]
    else:
        items = []
    if not items:
        raise ValueError(f"{field_name} 至少需要一个值")
    return items


def _ensure_int_list(value, field_name: str, *, min_value: int | None = None, max_value: int | None = None) -> list[int]:
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = []
    if not items:
        raise ValueError(f"{field_name} 至少需要一个值")
    numbers = []
    for item in items:
        numbers.append(_ensure_int(item, field_name, min_value=min_value, max_value=max_value))
    return numbers


def _normalize_cluster_config(value) -> dict:
    cluster_raw = value if isinstance(value, dict) else {}
    raw_slaves = cluster_raw.get("SLAVES", ["192.168.0.203:19001"])
    normalized_slaves = []
    if isinstance(raw_slaves, str):
        normalized_slaves = [item.strip() for item in raw_slaves.split(",") if item.strip()]
    elif isinstance(raw_slaves, (list, tuple)):
        for index, item in enumerate(raw_slaves):
            if isinstance(item, dict):
                endpoint = str(item.get("ENDPOINT", "")).strip()
                if not endpoint:
                    raise ValueError(f"CLUSTER.SLAVES[{index}] 缺少 ENDPOINT")
                normalized_item = {"ENDPOINT": endpoint}
                if item.get("NPUS_PER_NODE") is not None:
                    normalized_item["NPUS_PER_NODE"] = _ensure_int(item.get("NPUS_PER_NODE"), f"CLUSTER.SLAVES[{index}].NPUS_PER_NODE", min_value=0)
                label = str(item.get("LABEL", "")).strip()
                if label:
                    normalized_item["LABEL"] = label
                normalized_slaves.append(normalized_item)
            else:
                endpoint = str(item).strip()
                if endpoint:
                    normalized_slaves.append(endpoint)
    return {
        "ENABLED": _ensure_bool(cluster_raw.get("ENABLED", False), "CLUSTER.ENABLED"),
        "MASTER_ADDR": _ensure_string(cluster_raw.get("MASTER_ADDR", "192.168.0.170"), "CLUSTER.MASTER_ADDR"),
        "MASTER_PORT": _ensure_int(cluster_raw.get("MASTER_PORT", 8118), "CLUSTER.MASTER_PORT", min_value=1),
        "NODE_RANK": 0,
        "LISTEN_HOST": _ensure_string(cluster_raw.get("LISTEN_HOST", "0.0.0.0"), "CLUSTER.LISTEN_HOST"),
        "LISTEN_PORT": _ensure_int(cluster_raw.get("LISTEN_PORT", 19001), "CLUSTER.LISTEN_PORT", min_value=1),
        "REQUEST_TIMEOUT": _ensure_int(cluster_raw.get("REQUEST_TIMEOUT", 30), "CLUSTER.REQUEST_TIMEOUT", min_value=1),
        "SESSION_TIMEOUT": _ensure_int(cluster_raw.get("SESSION_TIMEOUT", 7200), "CLUSTER.SESSION_TIMEOUT", min_value=1),
        "LOCAL_NPUS_PER_NODE": _ensure_int(cluster_raw.get("LOCAL_NPUS_PER_NODE", 0), "CLUSTER.LOCAL_NPUS_PER_NODE", min_value=0),
        "SLAVES": normalized_slaves,
    }


def _normalize_multi_node_config(value, field_name: str = "MULTI_NODE") -> dict:
    raw = value if isinstance(value, dict) else {}
    enabled = _ensure_bool(raw.get("ENABLED", False), f"{field_name}.ENABLED")
    if not enabled:
        return {"ENABLED": False}

    raw_nodes = raw.get("OTHER_NODES")
    if not isinstance(raw_nodes, list):
        raise ValueError(f"{field_name}.OTHER_NODES 需要是列表")

    normalized_nodes = []
    for index, item in enumerate(raw_nodes):
        if not isinstance(item, dict):
            raise ValueError(f"{field_name}.OTHER_NODES[{index}] 需要是对象")
        node_prefix = f"{field_name}.OTHER_NODES[{index}]"
        has_container = _ensure_bool(item.get("HAS_CONTAINER", False), f"{node_prefix}.HAS_CONTAINER")
        normalized_item = {
            "HOST": _ensure_string(item.get("HOST"), f"{node_prefix}.HOST"),
            "SSH_PORT": _ensure_int(item.get("SSH_PORT", 22), f"{node_prefix}.SSH_PORT", min_value=1),
            "LMSV_PATH": _ensure_string(item.get("LMSV_PATH"), f"{node_prefix}.LMSV_PATH"),
            "PTA_NAME": _ensure_string(item.get("PTA_NAME"), f"{node_prefix}.PTA_NAME"),
            "MSA_NAME": _ensure_string(item.get("MSA_NAME"), f"{node_prefix}.MSA_NAME"),
            "PTA_PATH": _ensure_string(item.get("PTA_PATH"), f"{node_prefix}.PTA_PATH"),
            "MSA_PATH": _ensure_string(item.get("MSA_PATH"), f"{node_prefix}.MSA_PATH"),
            "HAS_CONTAINER": has_container,
        }
        mf_name = str(item.get("MF_NAME", "")).strip()
        if mf_name:
            normalized_item["MF_NAME"] = mf_name
        if has_container:
            normalized_item["CONTAINER_NAME"] = _ensure_string(item.get("CONTAINER_NAME"), f"{node_prefix}.CONTAINER_NAME")
        elif str(item.get("CONTAINER_NAME", "")).strip():
            normalized_item["CONTAINER_NAME"] = str(item.get("CONTAINER_NAME", "")).strip()
        if item.get("NPUS_PER_NODE") is not None:
            normalized_item["NPUS_PER_NODE"] = _ensure_int(item.get("NPUS_PER_NODE"), f"{node_prefix}.NPUS_PER_NODE", min_value=0)
        normalized_nodes.append(normalized_item)

    if not normalized_nodes:
        raise ValueError(f"{field_name}.OTHER_NODES 至少需要一个从节点")

    nnodes = _ensure_int(raw.get("NNODES", len(normalized_nodes) + 1), f"{field_name}.NNODES", min_value=2)
    corrected_nnodes = len(normalized_nodes) + 1
    if nnodes != corrected_nnodes:
        nnodes = corrected_nnodes

    normalized = {
        "ENABLED": True,
        "MASTER_ADDR": _ensure_string(raw.get("MASTER_ADDR", "127.0.0.1"), f"{field_name}.MASTER_ADDR"),
        "NNODES": nnodes,
        "OTHER_NODES": normalized_nodes,
    }
    if raw.get("MASTER_PORT") is not None:
        normalized["MASTER_PORT"] = _ensure_int(raw.get("MASTER_PORT"), f"{field_name}.MASTER_PORT", min_value=1)
    if raw.get("LOCAL_NPUS_PER_NODE") is not None:
        normalized["LOCAL_NPUS_PER_NODE"] = _ensure_int(raw.get("LOCAL_NPUS_PER_NODE"), f"{field_name}.LOCAL_NPUS_PER_NODE", min_value=0)
    return normalized


def _normalize_task45_multinode(value, field_name: str) -> dict:
    return _normalize_multi_node_config(value, field_name)


def normalize_config(raw_config: dict) -> dict:
    merged = deep_merge(BASE_CONFIG, raw_config or {})
    tasks_raw = merged.get("tasks") if isinstance(merged.get("tasks"), dict) else {}

    task_type = _ensure_int(merged.get("task_type"), "task_type", min_value=1, max_value=6)
    task1_raw = tasks_raw.get("1") if isinstance(tasks_raw.get("1"), dict) else {}
    task2_raw = tasks_raw.get("2") if isinstance(tasks_raw.get("2"), dict) else {}
    task3_raw = tasks_raw.get("3") if isinstance(tasks_raw.get("3"), dict) else {}
    task4_raw = tasks_raw.get("4") if isinstance(tasks_raw.get("4"), dict) else {}
    task5_raw = tasks_raw.get("5") if isinstance(tasks_raw.get("5"), dict) else {}
    task6_raw = tasks_raw.get("6") if isinstance(tasks_raw.get("6"), dict) else {}

    task2_models = _ensure_string_list(task2_raw.get("MODELS"), "任务2 MODELS")
    task2_submodules = _ensure_int_list(task2_raw.get("SUBMODULES"), "任务2 SUBMODULES", min_value=0, max_value=10)
    if len(task2_models) != len(task2_submodules):
        raise ValueError("任务2中 MODELS 与 SUBMODULES 数量必须一一对应")

    return {
        "task_type": task_type,
        "PTA_NAME": _ensure_string(merged.get("PTA_NAME"), "PTA_NAME"),
        "MSA_NAME": _ensure_string(merged.get("MSA_NAME"), "MSA_NAME"),
        "MF_NAME": _ensure_string(merged.get("MF_NAME"), "MF_NAME"),
        "PTA_PATH": _ensure_string(merged.get("PTA_PATH"), "PTA_PATH"),
        "MSA_PATH": _ensure_string(merged.get("MSA_PATH"), "MSA_PATH"),
        "SAVE_ABNORMAL_WEIGHTS": _ensure_bool(merged.get("SAVE_ABNORMAL_WEIGHTS", True), "SAVE_ABNORMAL_WEIGHTS"),
        "CLUSTER": _normalize_cluster_config(merged.get("CLUSTER")),
        "tasks": {
            "1": {
                "MODEL_NAME": _ensure_string(task1_raw.get("MODEL_NAME"), "任务1 MODEL_NAME"),
                "TOTAL_ITER": _ensure_int(task1_raw.get("TOTAL_ITER"), "任务1 TOTAL_ITER", min_value=1),
                "PTA_MAX_RUNTIME": _ensure_int(task1_raw.get("PTA_MAX_RUNTIME", 3000), "任务1 PTA_MAX_RUNTIME", min_value=1),
                "MSA_MAX_RUNTIME": _ensure_int(task1_raw.get("MSA_MAX_RUNTIME", task1_raw.get("MAX_VALIDATE_TIME", 3000)), "任务1 MSA_MAX_RUNTIME", min_value=1),
                "LOG_INIT_WAIT": _ensure_int(task1_raw.get("LOG_INIT_WAIT", 240), "任务1 LOG_INIT_WAIT", min_value=1),
                "LOG_STABLE_THRESHOLD": _ensure_int(task1_raw.get("LOG_STABLE_THRESHOLD", 150), "任务1 LOG_STABLE_THRESHOLD", min_value=1),
                "COMPARE_MODE": _ensure_choice(task1_raw.get("COMPARE_MODE", "pta_msa"), "任务1 COMPARE_MODE", ("pta_msa", "pta_mf")),
              "ENABLE_MF_WEIGHT_LOAD": _ensure_bool(task1_raw.get("ENABLE_MF_WEIGHT_LOAD", True), "任务1 ENABLE_MF_WEIGHT_LOAD"),
                "BASE_SEED": _ensure_int(task1_raw.get("BASE_SEED"), "任务1 BASE_SEED", min_value=0),
                "MUTNM": _ensure_int(task1_raw.get("MUTNM"), "任务1 MUTNM", min_value=1),
                "SAVE_STEPS": _ensure_int(task1_raw.get("SAVE_STEPS"), "任务1 SAVE_STEPS", min_value=1),
                "LOAD_STEPS": _ensure_int(task1_raw.get("LOAD_STEPS"), "任务1 LOAD_STEPS", min_value=1),
                "MULTI_NODE": _normalize_multi_node_config(task1_raw.get("MULTI_NODE"), "任务1 MULTI_NODE"),
            },
            "2": {
                "MODELS": task2_models,
                "TOTAL_ITER": _ensure_int(task2_raw.get("TOTAL_ITER"), "任务2 TOTAL_ITER", min_value=1),
                "PTA_MAX_RUNTIME": _ensure_int(task2_raw.get("PTA_MAX_RUNTIME", 3000), "任务2 PTA_MAX_RUNTIME", min_value=1),
                "MSA_MAX_RUNTIME": _ensure_int(task2_raw.get("MSA_MAX_RUNTIME", task2_raw.get("MAX_VALIDATE_TIME", 3000)), "任务2 MSA_MAX_RUNTIME", min_value=1),
                "LOG_INIT_WAIT": _ensure_int(task2_raw.get("LOG_INIT_WAIT", 240), "任务2 LOG_INIT_WAIT", min_value=1),
                "LOG_STABLE_THRESHOLD": _ensure_int(task2_raw.get("LOG_STABLE_THRESHOLD", 150), "任务2 LOG_STABLE_THRESHOLD", min_value=1),
                "BASE_SEED": _ensure_int(task2_raw.get("BASE_SEED"), "任务2 BASE_SEED", min_value=0),
                "SUBMODULES": task2_submodules,
                "MUTNM": _ensure_int(task2_raw.get("MUTNM"), "任务2 MUTNM", min_value=1),
                "SAVE_STEPS": _ensure_int(task2_raw.get("SAVE_STEPS", 1), "任务2 SAVE_STEPS", min_value=1),
                "LOAD_STEPS": _ensure_int(task2_raw.get("LOAD_STEPS", 15), "任务2 LOAD_STEPS", min_value=1),
                "COMPARE_MODE": _ensure_choice(task2_raw.get("COMPARE_MODE", "pta_msa"), "任务2 COMPARE_MODE", ("pta_msa", "pta_mf")),
                "MF_ARGS_PATH": _ensure_string(task2_raw.get("MF_ARGS_PATH", "assets/runtime/mf_templates/basic.yaml"), "任务2 MF_ARGS_PATH"),
                "ENABLE_MF_WEIGHT_LOAD": _ensure_bool(task2_raw.get("ENABLE_MF_WEIGHT_LOAD", False), "任务2 ENABLE_MF_WEIGHT_LOAD"),
                "MULTI_NODE": _normalize_multi_node_config(task2_raw.get("MULTI_NODE"), "任务2 MULTI_NODE"),
            },
            "3": {
                "MODELS": _ensure_string_list(task3_raw.get("MODELS"), "任务3 MODELS"),
                "TOTAL_ITER": _ensure_int(task3_raw.get("TOTAL_ITER"), "任务3 TOTAL_ITER", min_value=1),
                "PTA_MAX_RUNTIME": _ensure_int(task3_raw.get("PTA_MAX_RUNTIME", 3000), "任务3 PTA_MAX_RUNTIME", min_value=1),
                "MSA_MAX_RUNTIME": _ensure_int(task3_raw.get("MSA_MAX_RUNTIME", task3_raw.get("MAX_VALIDATE_TIME", 3000)), "任务3 MSA_MAX_RUNTIME", min_value=1),
                "LOG_INIT_WAIT": _ensure_int(task3_raw.get("LOG_INIT_WAIT", 240), "任务3 LOG_INIT_WAIT", min_value=1),
                "LOG_STABLE_THRESHOLD": _ensure_int(task3_raw.get("LOG_STABLE_THRESHOLD", 150), "任务3 LOG_STABLE_THRESHOLD", min_value=1),
                "MAX_MUTATION_WAIT": _ensure_int(task3_raw.get("MAX_MUTATION_WAIT", 600), "任务3 MAX_MUTATION_WAIT", min_value=1),
                "BASE_SEED": _ensure_int(task3_raw.get("BASE_SEED"), "任务3 BASE_SEED", min_value=0),
                "MUTNM": _ensure_int(task3_raw.get("MUTNM"), "任务3 MUTNM", min_value=1),
                "SAVE_STEPS": _ensure_int(task3_raw.get("SAVE_STEPS", 1), "任务3 SAVE_STEPS", min_value=1),
                "LOAD_STEPS": _ensure_int(task3_raw.get("LOAD_STEPS", 15), "任务3 LOAD_STEPS", min_value=1),
                "COMPARE_MODE": _ensure_choice(task3_raw.get("COMPARE_MODE", "pta_msa"), "任务3 COMPARE_MODE", ("pta_msa", "pta_mf")),
                "MULTI_NODE": _normalize_multi_node_config(task3_raw.get("MULTI_NODE"), "任务3 MULTI_NODE"),
            },
            "4": {
                "TOTAL_ITER": _ensure_int(task4_raw.get("TOTAL_ITER", 5), "任务4 TOTAL_ITER", min_value=1),
                "COMPARE_MODE": _ensure_choice(task4_raw.get("COMPARE_MODE", "pta_msa"), "任务4 COMPARE_MODE", ("pta_msa",)),
                "PTA_MAX_RUNTIME": _ensure_int(task4_raw.get("PTA_MAX_RUNTIME", 3000), "任务4 PTA_MAX_RUNTIME", min_value=1),
                "MSA_MAX_RUNTIME": _ensure_int(task4_raw.get("MSA_MAX_RUNTIME", task4_raw.get("MAX_VALIDATE_TIME", 3000)), "任务4 MSA_MAX_RUNTIME", min_value=1),
                "LOG_INIT_WAIT": _ensure_int(task4_raw.get("LOG_INIT_WAIT", 240), "任务4 LOG_INIT_WAIT", min_value=1),
                "LOG_STABLE_THRESHOLD": _ensure_int(task4_raw.get("LOG_STABLE_THRESHOLD", 150), "任务4 LOG_STABLE_THRESHOLD", min_value=1),
                "SAVE_STEPS": _ensure_int(task4_raw.get("SAVE_STEPS", 1), "任务4 SAVE_STEPS", min_value=1),
                "RUN_STEPS": _ensure_int(task4_raw.get("RUN_STEPS", 20), "任务4 RUN_STEPS", min_value=1),
                "ENABLE_MF_WEIGHT_LOAD": _ensure_bool(task4_raw.get("ENABLE_MF_WEIGHT_LOAD", False), "任务4 ENABLE_MF_WEIGHT_LOAD"),
                "MULTI_NODE": _normalize_task45_multinode(task4_raw.get("MULTI_NODE"), "任务4 MULTI_NODE"),
            },
            "5": {
                "TOTAL_ITER": _ensure_int(task5_raw.get("TOTAL_ITER", 5), "任务5 TOTAL_ITER", min_value=1),
                "COMPARE_MODE": _ensure_choice(task5_raw.get("COMPARE_MODE", "pta_msa"), "任务5 COMPARE_MODE", ("pta_msa",)),
                "PTA_MAX_RUNTIME": _ensure_int(task5_raw.get("PTA_MAX_RUNTIME", 3000), "任务5 PTA_MAX_RUNTIME", min_value=1),
                "MSA_MAX_RUNTIME": _ensure_int(task5_raw.get("MSA_MAX_RUNTIME", task5_raw.get("MAX_VALIDATE_TIME", 3000)), "任务5 MSA_MAX_RUNTIME", min_value=1),
                "LOG_INIT_WAIT": _ensure_int(task5_raw.get("LOG_INIT_WAIT", 240), "任务5 LOG_INIT_WAIT", min_value=1),
                "LOG_STABLE_THRESHOLD": _ensure_int(task5_raw.get("LOG_STABLE_THRESHOLD", 150), "任务5 LOG_STABLE_THRESHOLD", min_value=1),
                "SAVE_STEPS": _ensure_int(task5_raw.get("SAVE_STEPS", 1), "任务5 SAVE_STEPS", min_value=1),
                "RUN_STEPS": _ensure_int(task5_raw.get("RUN_STEPS", 20), "任务5 RUN_STEPS", min_value=1),
                "MUTATE_STEPS": _ensure_int(task5_raw.get("MUTATE_STEPS", 10), "任务5 MUTATE_STEPS", min_value=1),
                "MODULE_TYPE": _ensure_choice(task5_raw.get("MODULE_TYPE", "all"), "任务5 MODULE_TYPE", ("all", "text_decoder", "image_encoder")),
                "ENABLE_MF_WEIGHT_LOAD": _ensure_bool(task5_raw.get("ENABLE_MF_WEIGHT_LOAD", False), "任务5 ENABLE_MF_WEIGHT_LOAD"),
                "MULTI_NODE": _normalize_task45_multinode(task5_raw.get("MULTI_NODE"), "任务5 MULTI_NODE"),
            },
            "6": {
                "MODEL_NAME": _ensure_choice(task6_raw.get("MODEL_NAME", "internvl3"), "任务6 MODEL_NAME", ("internvl3", "qwenvl", "opensora", "cogvideox")),
                "TOTAL_ITER": _ensure_int(task6_raw.get("TOTAL_ITER", 10), "任务6 TOTAL_ITER", min_value=1),
                "MUTNM": _ensure_int(task6_raw.get("MUTNM", 2), "任务6 MUTNM", min_value=1),
                "COMPARE_MODE": _ensure_choice(task6_raw.get("COMPARE_MODE", "pta_msa"), "任务6 COMPARE_MODE", ("pta_msa",)),
                "TRAIN_ITERS": _ensure_int(task6_raw.get("TRAIN_ITERS", 5), "任务6 TRAIN_ITERS", min_value=1),
                "PTA_MAX_RUNTIME": _ensure_int(task6_raw.get("PTA_MAX_RUNTIME", 900), "任务6 PTA_MAX_RUNTIME", min_value=1),
                "MSA_MAX_RUNTIME": _ensure_int(task6_raw.get("MSA_MAX_RUNTIME", 900), "任务6 MSA_MAX_RUNTIME", min_value=1),
            },
        },
    }


def compact_config(config: dict) -> dict:
    task_type = int(config.get("task_type", 1))
    task_key = str(task_type)
    task_config = deepcopy((config.get("tasks") or {}).get(task_key, {}))
    return {
        "task_type": task_type,
        "PTA_NAME": config["PTA_NAME"],
        "MSA_NAME": config["MSA_NAME"],
        "MF_NAME": config["MF_NAME"],
        "PTA_PATH": config["PTA_PATH"],
        "MSA_PATH": config["MSA_PATH"],
        "SAVE_ABNORMAL_WEIGHTS": config["SAVE_ABNORMAL_WEIGHTS"],
        "CLUSTER": deepcopy(config.get("CLUSTER") or {}),
        "tasks": {
            task_key: task_config,
        },
    }


def load_effective_config() -> dict:
    config = deepcopy(BASE_CONFIG)
    config = deep_merge(config, load_json(EXAMPLE_CONFIG_PATH))
    user_config = load_json(CONFIG_PATH)
    if user_config:
        try:
            return normalize_config(deep_merge(config, user_config))
        except ValueError:
            return normalize_config(config)
    return normalize_config(config)


def save_config(config: dict) -> dict:
    normalized = normalize_config(config)
    compacted = compact_config(normalized)
    CONFIG_PATH.write_text(json.dumps(compacted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return normalized


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def format_clock(timestamp: float | None) -> str | None:
    if not timestamp:
        return None
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def format_elapsed(seconds: float | None) -> str:
    if seconds is None:
        return "0s"
    total = max(0, int(seconds))
    hour, rem = divmod(total, 3600)
    minute, sec = divmod(rem, 60)
    if hour:
        return f"{hour}h {minute}m {sec}s"
    if minute:
        return f"{minute}m {sec}s"
    return f"{sec}s"


def read_text_if_exists(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _parse_cpu_totals() -> tuple[int, int] | None:
    match = CPU_TOTAL_RE.search(read_text_if_exists(Path("/proc/stat")))
    if not match:
        return None
    parts = [int(item) for item in match.group(1).split() if item.isdigit()]
    if len(parts) < 4:
        return None
    idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
    total = sum(parts)
    return total, idle


def _parse_meminfo() -> dict[str, int]:
    info: dict[str, int] = {}
    for line in read_text_if_exists(Path("/proc/meminfo")).splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        value = raw_value.strip().split()[0]
        if value.isdigit():
            info[key] = int(value) * 1024
    return info


def _format_bytes(num_bytes: int | None) -> str:
    if not num_bytes or num_bytes < 0:
        return "-"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{num_bytes} B"


def _run_command(command: list[str], *, timeout: float = 2.0) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return False, ""
    output = (completed.stdout or completed.stderr or "").strip()
    return completed.returncode == 0, output


def _split_table_cells(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return []
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    return cells


def _split_columns(cell: str) -> list[str]:
    return [item for item in re.split(r"\s{2,}", cell.strip()) if item]


def _normalize_mem_pair(value: str) -> str:
    cleaned = re.sub(r"\s*/\s*", " / ", value.strip())
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned or "-"


def _parse_npu_summary_cell(cell: str) -> tuple[str, str, str] | None:
    match = re.match(r"^\s*([0-9.]+)\s+([0-9.]+)\s+([0-9]+\s*/\s*[0-9]+)\s*$", cell)
    if not match:
        return None
    power, temperature, hugepages = match.groups()
    return power, temperature, _normalize_mem_pair(hugepages)


def _parse_npu_metric_cell(cell: str) -> tuple[str, str, str] | None:
    match = re.match(
        r"^\s*([0-9.]+)\s+([0-9]+\s*/\s*[0-9]+)\s+([0-9]+\s*/\s*[0-9]+)\s*$",
        cell,
    )
    if not match:
        return None
    aicore, memory_usage, hbm_usage = match.groups()
    return aicore, _normalize_mem_pair(memory_usage), _normalize_mem_pair(hbm_usage)


def _parse_npu_smi_processes(lines: list[str]) -> dict[str, list[dict]]:
    processes: dict[str, list[dict]] = {}
    in_process_table = False

    for line in lines:
        if "Process id" in line and "Process name" in line:
            in_process_table = True
            continue
        if not in_process_table or not line.strip().startswith("|"):
            continue

        cells = _split_table_cells(line)
        if len(cells) < 4:
            continue

        left_columns = _split_columns(cells[0])
        if len(left_columns) < 2 or not left_columns[0].isdigit():
            continue

        npu_id, chip_id = left_columns[:2]
        process_id = cells[1].strip()
        process_name = cells[2].strip()
        process_memory = cells[3].strip()
        if not process_id.isdigit():
            continue

        processes.setdefault(npu_id, []).append(
            {
                "chip_id": chip_id,
                "pid": process_id,
                "name": process_name or "-",
                "memory": process_memory or "-",
            }
        )

    return processes


def _parse_npu_smi_output(output: str) -> list[dict]:
    lines = output.splitlines()
    process_map = _parse_npu_smi_processes(lines)
    devices: list[dict] = []

    i = 0
    while i < len(lines):
        first_line = lines[i]
        if "Process id" in first_line and "Process name" in first_line:
            break
        if not first_line.strip().startswith("|"):
            i += 1
            continue

        first_cells = _split_table_cells(first_line)
        if len(first_cells) != 3:
            i += 1
            continue

        left_columns = _split_columns(first_cells[0])
        if len(left_columns) < 2 or not left_columns[0].isdigit():
            i += 1
            continue
        summary_metrics = _parse_npu_summary_cell(first_cells[2])
        if not summary_metrics:
            i += 1
            continue

        if i + 1 >= len(lines):
            break
        second_line = lines[i + 1]
        second_cells = _split_table_cells(second_line)
        if len(second_cells) != 3:
            i += 1
            continue

        second_left_columns = _split_columns(second_cells[0])
        if len(second_left_columns) < 1:
            i += 1
            continue
        metric_values = _parse_npu_metric_cell(second_cells[2])
        if not metric_values:
            i += 1
            continue

        npu_id = left_columns[0]
        chip_id = second_left_columns[0]
        power, temperature, hugepages = summary_metrics
        aicore, memory_usage, hbm_usage = metric_values
        device = {
            "kind": "npu",
            "index": npu_id,
            "chip_id": chip_id,
            "name": left_columns[1],
            "health": first_cells[1] or "-",
            "power": f"{power} W",
            "temperature": f"{temperature} C",
            "hugepages": hugepages,
            "bus_id": second_cells[1] or "-",
            "utilization": f"{aicore}%",
            "memory": f"{hbm_usage} MB",
            "memory_usage": f"{memory_usage} MB",
            "hbm_usage": f"{hbm_usage} MB",
            "processes": process_map.get(npu_id, []),
        }

        devices.append(device)
        i += 2

    return devices


def collect_npu_info() -> tuple[list[dict], str | None]:
    binary = shutil.which("npu-smi")
    if not binary:
        return [], None

    ok, output = _run_command([binary, "info"])
    if not ok or not output:
        return [], "npu-smi info"
    devices = _parse_npu_smi_output(output)
    if not devices:
        # `npu-smi info` 已有返回，但当前正则未完全适配时，仍然视为已检测到 NPU。
        devices = [
            {
                "kind": "npu",
                "index": "-",
                "name": "NPU 已检测到",
                "health": "已获取输出",
                "utilization": "解析中",
                "memory": "解析中",
                "temperature": "解析中",
                "power": "解析中",
            }
        ]
    return devices, "npu-smi info"


class HardwareMonitor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_snapshot: dict | None = None
        self._last_updated = 0.0
        self._previous_cpu_totals: tuple[int, int] | None = None

    def snapshot(self) -> dict:
        now = time.time()
        with self._lock:
            if self._last_snapshot and now - self._last_updated < 1.0:
                return deepcopy(self._last_snapshot)

            snapshot = self._collect_snapshot()
            self._last_snapshot = snapshot
            self._last_updated = now
            return deepcopy(snapshot)

    def _collect_snapshot(self) -> dict:
        cpu_totals = _parse_cpu_totals()
        cpu_percent = None
        if cpu_totals and self._previous_cpu_totals:
            total_delta = cpu_totals[0] - self._previous_cpu_totals[0]
            idle_delta = cpu_totals[1] - self._previous_cpu_totals[1]
            if total_delta > 0:
                cpu_percent = round(max(0.0, min(100.0, (1 - idle_delta / total_delta) * 100)), 1)
        self._previous_cpu_totals = cpu_totals

        loadavg = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
        meminfo = _parse_meminfo()
        mem_total = meminfo.get("MemTotal", 0)
        mem_available = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
        mem_used = max(0, mem_total - mem_available)
        mem_percent = round(mem_used * 100 / mem_total, 1) if mem_total else None

        npu_devices, npu_source = collect_npu_info()
        devices = npu_devices
        accelerator_labels = [f"NPU {len(npu_devices)}"] if npu_devices else []
        if npu_source and not accelerator_labels:
            accelerator_labels = ["已获取 NPU 信息"]

        sources = [item for item in [npu_source] if item]
        return {
            "updated_at": format_clock(time.time()),
            "cpu": {
                "usage_percent": cpu_percent,
                "loadavg": [round(item, 2) for item in loadavg],
                "cores": os.cpu_count() or 0,
            },
            "memory": {
                "used_bytes": mem_used,
                "total_bytes": mem_total,
                "usage_percent": mem_percent,
                "display": f"{_format_bytes(mem_used)} / {_format_bytes(mem_total)}" if mem_total else "-",
            },
            "accelerators": devices,
            "accelerator_summary": ", ".join(accelerator_labels) if accelerator_labels else "未检测到 NPU",
            "sources": sources,
        }


def summarize_task(config: dict) -> str:
    task_type = int(config.get("task_type", 1))
    task = (config.get("tasks") or {}).get(str(task_type), {})
    if task_type == 1:
        return f"模型 {task.get('MODEL_NAME', '-')}, TOTAL_ITER={task.get('TOTAL_ITER', '-')}, MUTNM={task.get('MUTNM', '-')}"
    if task_type == 2:
        model_count = len(task.get("MODELS") or [])
        sub_count = len(task.get("SUBMODULES") or [])
        return f"模型/子模块 {model_count}/{sub_count} 组, TOTAL_ITER={task.get('TOTAL_ITER', '-')}, MUTNM={task.get('MUTNM', '-')}"
    if task_type == 4:
        multi = task.get("MULTI_NODE") if isinstance(task.get("MULTI_NODE"), dict) else {}
        multi_text = "off"
        if multi.get("ENABLED"):
            multi_text = f"on(nnodes={multi.get('NNODES', '-')})"
        return (
            f"TOTAL_ITER={task.get('TOTAL_ITER', '-')}, RUN_STEPS={task.get('RUN_STEPS', '-')}, "
            f"COMPARE_MODE={task.get('COMPARE_MODE', '-')}, MULTI_NODE={multi_text}"
        )
    if task_type == 5:
        multi = task.get("MULTI_NODE") if isinstance(task.get("MULTI_NODE"), dict) else {}
        multi_text = "off"
        if multi.get("ENABLED"):
            multi_text = f"on(nnodes={multi.get('NNODES', '-')})"
        return (
            f"TOTAL_ITER={task.get('TOTAL_ITER', '-')}, RUN_STEPS={task.get('RUN_STEPS', '-')}, "
            f"MUTATE_STEPS={task.get('MUTATE_STEPS', '-')}, MODULE_TYPE={task.get('MODULE_TYPE', '-')}, "
            f"COMPARE_MODE={task.get('COMPARE_MODE', '-')}, MULTI_NODE={multi_text}"
        )
    model_count = len(task.get("MODELS") or [])
    return f"模型 {model_count} 个, TOTAL_ITER={task.get('TOTAL_ITER', '-')}, MUTNM={task.get('MUTNM', '-')}"


def resolve_output_dir(name: str) -> Path:
    if not OUTPUT_NAME_RE.fullmatch(name):
        raise ValueError("非法 output 目录名")
    target = (OUTPUT_ROOT / name).resolve()
    output_root = OUTPUT_ROOT.resolve()
    if target.parent != output_root or not target.is_dir():
        raise FileNotFoundError("output 目录不存在")
    return target


def resolve_output_file(name: str, relative_path: str) -> Path:
    base = resolve_output_dir(name)
    target = (base / relative_path).resolve()
    if base != target and base not in target.parents:
        raise ValueError("非法文件路径")
    if not target.exists() or not target.is_file():
        raise FileNotFoundError("文件不存在")
    return target


def resolve_output_path(name: str, relative_path: str = "") -> Path:
    base = resolve_output_dir(name)
    cleaned = str(relative_path or "").strip().strip("/")
    target = (base / cleaned).resolve() if cleaned else base
    if base != target and base not in target.parents:
        raise ValueError("非法文件路径")
    if not target.exists():
        raise FileNotFoundError("路径不存在")
    return target


def _relative_output_path(base: Path, target: Path) -> str:
    if target == base:
        return ""
    return target.relative_to(base).as_posix()


def is_iter_directory(relative_path: str) -> bool:
    cleaned = str(relative_path or "").strip().strip("/")
    return bool(re.fullmatch(r"iters/iter_[^/]+", cleaned))


def _format_file_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    units = ["KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        value /= 1024
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
    return f"{num_bytes} B"


def detect_preview_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    mime_type, _ = mimetypes.guess_type(path.name)
    text_suffixes = {
        ".txt", ".log", ".md", ".json", ".jsonl", ".yaml", ".yml", ".py", ".sh",
        ".csv", ".tsv", ".ini", ".cfg", ".conf", ".xml", ".html", ".htm",
        ".js", ".ts", ".tsx", ".jsx", ".css", ".sql",
    }
    if mime_type:
        if mime_type.startswith("image/"):
            return "image"
        if mime_type == "application/pdf":
            return "pdf"
        if mime_type.startswith("text/"):
            if suffix in {".html", ".htm"}:
                return "html"
            if suffix == ".md":
                return "markdown"
            if suffix in {".csv", ".tsv"}:
                return "csv"
            return "text"
        if mime_type == "application/json":
            return "text"
    if suffix in {".html", ".htm"}:
        return "html"
    if suffix == ".md":
        return "markdown"
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp"}:
        return "image"
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".csv", ".tsv"}:
        return "csv"
    if suffix in text_suffixes:
        return "text"
    return "download"


def list_output_files(output_id: str, relative_path: str = "") -> dict:
    base = resolve_output_dir(output_id)
    current = resolve_output_path(output_id, relative_path)
    if not current.is_dir():
        raise ValueError("当前路径不是目录")

    entries = []
    for child in sorted(current.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
        stat = child.stat()
        rel_path = _relative_output_path(base, child)
        entries.append(
            {
                "name": child.name,
                "path": rel_path,
                "is_dir": child.is_dir(),
                "size": None if child.is_dir() else stat.st_size,
                "size_text": "-" if child.is_dir() else _format_file_size(stat.st_size),
                "updated_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "mime_type": mimetypes.guess_type(child.name)[0] or "application/octet-stream",
                "preview_kind": "directory" if child.is_dir() else detect_preview_kind(child),
                "raw_url": f"results/{quote(output_id)}/{quote(rel_path)}" if rel_path else f"results/{quote(output_id)}",
            }
        )

    parts = []
    cursor = base
    parts.append({"name": output_id, "path": ""})
    current_relative = _relative_output_path(base, current)
    if current_relative:
        for segment in current_relative.split("/"):
            cursor = cursor / segment
            parts.append({"name": segment, "path": _relative_output_path(base, cursor)})

    parent_path = _relative_output_path(base, current.parent) if current != base else None
    return {
        "output_id": output_id,
        "path": current_relative,
        "parent_path": parent_path,
        "breadcrumbs": parts,
        "entries": entries,
    }


def get_output_file_preview(output_id: str, relative_path: str) -> dict:
    file_path = resolve_output_file(output_id, relative_path)
    preview_kind = detect_preview_kind(file_path)
    stat = file_path.stat()
    payload = {
        "output_id": output_id,
        "path": _relative_output_path(resolve_output_dir(output_id), file_path),
        "name": file_path.name,
        "size": stat.st_size,
        "size_text": _format_file_size(stat.st_size),
        "updated_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "mime_type": mimetypes.guess_type(file_path.name)[0] or "application/octet-stream",
        "preview_kind": preview_kind,
        "raw_url": f"results/{quote(output_id)}/{quote(_relative_output_path(resolve_output_dir(output_id), file_path))}",
    }
    if preview_kind not in {"text", "markdown", "csv"}:
        return payload

    max_bytes = 500_000
    raw = file_path.read_bytes()
    truncated = len(raw) > max_bytes
    sample = raw[:max_bytes]
    try:
        content = sample.decode("utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        content = sample.decode("utf-8", errors="replace")
        encoding = "utf-8(replace)"
    payload["truncated"] = truncated
    payload["encoding"] = encoding
    if preview_kind == "csv":
        delimiter = "\t" if file_path.suffix.lower() == ".tsv" else ","
        rows = list(csv.reader(content.splitlines(), delimiter=delimiter))
        payload["headers"] = rows[0] if rows else []
        payload["rows"] = rows[1:201] if len(rows) > 1 else []
        payload["total_rows"] = max(len(rows) - 1, 0)
        payload["truncated"] = truncated or len(payload["rows"]) < max(len(rows) - 1, 0)
        payload["content"] = content
        return payload
    payload["content"] = content
    return payload


def create_iter_archive(output_id: str, relative_path: str) -> tuple[Path, str]:
    cleaned = str(relative_path or "").strip().strip("/")
    if not is_iter_directory(cleaned):
        raise ValueError("仅支持下载 output/<run>/iters/iter_x 目录")

    iter_dir = resolve_output_path(output_id, cleaned)
    if not iter_dir.is_dir():
        raise FileNotFoundError("目标 iter 目录不存在")

    temp_root = Path(tempfile.mkdtemp(prefix="lmsv_iter_zip_"))
    archive_base = temp_root / f"{output_id}_{iter_dir.name}"
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=str(iter_dir.parent), base_dir=iter_dir.name))
    download_name = f"{output_id}_{iter_dir.name}.zip"
    return archive_path, download_name


def list_output_runs() -> list[dict]:
    OUTPUT_ROOT.mkdir(exist_ok=True)
    entries = []
    for path in sorted((item for item in OUTPUT_ROOT.iterdir() if item.is_dir()), key=lambda item: item.stat().st_mtime, reverse=True):
        config = load_json(path / "config.json")
        try:
            config = normalize_config(config) if config else None
        except ValueError:
            config = None

        task_type = int(config.get("task_type", 1)) if config else 1
        report_path = path / "analysis" / "report.html"
        log_path = path / "log.txt"
        can_regenerate = output_has_replayable_analysis_data(path)
        status_text = "已生成报告" if report_path.exists() else ("仅有日志" if log_path.exists() else "待生成")
        entries.append(
            {
                "id": path.name,
                "task_type": task_type,
                "task_label": TASK_META.get(task_type, TASK_META[1])["label"],
                "summary": summarize_task(config) if config else "配置缺失或格式非法",
                "updated_at": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "has_report": report_path.exists(),
                "has_log": log_path.exists(),
                "can_regenerate": can_regenerate,
                "report_url": f"results/{quote(path.name)}/analysis/report.html" if report_path.exists() else None,
                "config_url": f"results/{quote(path.name)}/config.json",
                "log_url": f"results/{quote(path.name)}/log.txt",
                "status_text": status_text,
            }
        )
    return entries


def delete_output_run(output_id: str) -> dict:
    target = resolve_output_dir(output_id)
    shutil.rmtree(target)
    return {"output_id": output_id, "message": f"output/{output_id} 已删除"}


class RunManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._status = "idle"
        self._returncode: int | None = None
        self._started_at: float | None = None
        self._ended_at: float | None = None
        self._output_dir: str | None = None
        self._pid: int | None = None
        self._before_dirs: set[str] = set()
        self._log_lines: deque[str] = deque()
        self._first_cursor = 0
        self._next_cursor = 0
        self._stop_requested = False

    def _append_log_locked(self, line: str) -> None:
        cleaned = strip_ansi(line)
        if len(self._log_lines) >= LOG_BUFFER_LIMIT:
            self._log_lines.popleft()
            self._first_cursor += 1
        self._log_lines.append(cleaned)
        self._next_cursor += 1

    def _detect_output_dir_locked(self) -> None:
        if self._output_dir:
            return
        if not OUTPUT_ROOT.exists():
            return
        new_dirs = [item for item in OUTPUT_ROOT.iterdir() if item.is_dir() and item.name not in self._before_dirs]
        if not new_dirs:
            return
        latest = max(new_dirs, key=lambda item: item.stat().st_mtime)
        self._output_dir = latest.name

    def start(self, config: dict) -> dict:
        normalized = save_config(config)
        with self._lock:
            if self._process and self._process.poll() is None:
                raise RuntimeError("已有任务正在运行")

            OUTPUT_ROOT.mkdir(exist_ok=True)
            before_dirs = {item.name for item in OUTPUT_ROOT.iterdir() if item.is_dir()}
            self._process = subprocess.Popen(
                [sys.executable, "-u", str(DO_SCRIPT)],
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            self._status = "running"
            self._returncode = None
            self._started_at = time.time()
            self._ended_at = None
            self._output_dir = None
            self._pid = self._process.pid
            self._stop_requested = False
            self._before_dirs = before_dirs
            self._log_lines.clear()
            self._first_cursor = 0
            self._next_cursor = 0
            self._append_log_locked(f"[webui] 已启动 do.py，PID={self._pid}，PGID={self._pid}\n")

            thread = threading.Thread(target=self._consume_process_output, daemon=True)
            thread.start()

        return normalized

    def _consume_process_output(self) -> None:
        process = self._process
        if process is None:
            return

        try:
            if process.stdout is not None:
                for line in process.stdout:
                    with self._lock:
                        self._detect_output_dir_locked()
                        self._append_log_locked(line if line.endswith("\n") else line + "\n")
        finally:
            returncode = process.wait()
            with self._lock:
                self._detect_output_dir_locked()
                self._returncode = returncode
                self._ended_at = time.time()
                if self._stop_requested:
                    self._status = "stopped"
                else:
                    self._status = "finished" if returncode == 0 else "failed"
                self._append_log_locked(f"[webui] 进程结束，returncode={returncode}\n")
                self._process = None
                self._pid = None

    def _signal_process_group(self, pgid: int, sig: int, label: str) -> bool:
        try:
            os.killpg(pgid, sig)
            with self._lock:
                self._append_log_locked(f"[webui] 已向进程组 {pgid} 发送 {label}\n")
            return True
        except ProcessLookupError:
            return False
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._append_log_locked(f"[webui] 向进程组 {pgid} 发送 {label} 失败: {exc}\n")
            return False

    def _wait_for_process_exit(self, process: subprocess.Popen[str], timeout: float) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if process.poll() is not None:
                return True
            time.sleep(0.2)
        return process.poll() is not None

    def _stop_process_tree(self, process: subprocess.Popen[str], pgid: int) -> None:
        if process.poll() is not None:
            return

        sent_term = self._signal_process_group(pgid, signal.SIGTERM, "SIGTERM")
        if not sent_term:
            try:
                process.terminate()
                with self._lock:
                    self._append_log_locked("[webui] 进程组不存在，已回退为向主进程发送 terminate\n")
            except ProcessLookupError:
                return
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._append_log_locked(f"[webui] 回退 terminate 失败: {exc}\n")

        if self._wait_for_process_exit(process, STOP_TERM_TIMEOUT):
            return

        self._signal_process_group(pgid, signal.SIGKILL, "SIGKILL")
        if not self._wait_for_process_exit(process, STOP_KILL_TIMEOUT):
            with self._lock:
                self._append_log_locked("[webui] 警告：发送 SIGKILL 后进程仍未退出，请检查残留子进程\n")

    def stop(self) -> str:
        with self._lock:
            if not self._process or self._process.poll() is not None:
                raise RuntimeError("当前没有正在运行的任务")
            if self._status == "stopping":
                return "停止请求已发送，正在等待任务退出"
            self._status = "stopping"
            self._stop_requested = True
            process = self._process
            pgid = self._pid or process.pid
            self._append_log_locked(f"[webui] 正在停止任务，目标进程组={pgid}\n")

        thread = threading.Thread(target=self._stop_process_tree, args=(process, pgid), daemon=True)
        thread.start()
        return "已发送停止请求，正在终止任务进程组"

    def assert_output_not_active(self, output_name: str) -> None:
        with self._lock:
            active = bool(self._process and self._process.poll() is None)
            if not active:
                return
            self._detect_output_dir_locked()
            if self._output_dir == output_name:
                raise RuntimeError(f"output/{output_name} 仍在运行中，暂不支持当前操作")

    def snapshot(self, cursor: int | None) -> dict:
        with self._lock:
            active = bool(self._process and self._process.poll() is None)
            if active:
                self._detect_output_dir_locked()

            current_cursor = self._first_cursor if cursor is None else int(cursor)
            reset = current_cursor < self._first_cursor or current_cursor > self._next_cursor
            if reset:
                current_cursor = self._first_cursor

            offset = current_cursor - self._first_cursor
            logs = "".join(list(self._log_lines)[offset:])
            now = time.time()
            elapsed_seconds = None
            if self._started_at:
                if active:
                    elapsed_seconds = now - self._started_at
                elif self._ended_at:
                    elapsed_seconds = self._ended_at - self._started_at

            return {
                "state": self._status,
                "active": active,
                "pid": self._pid,
                "output_dir": self._output_dir,
                "returncode": self._returncode,
                "started_at": format_clock(self._started_at),
                "ended_at": format_clock(self._ended_at),
                "elapsed": format_elapsed(elapsed_seconds),
                "reset": reset,
                "logs": logs,
                "next_cursor": self._next_cursor,
            }


class LMSVRequestHandler(BaseHTTPRequestHandler):
    server_version = "LMSVWebUI/1.0"

    @property
    def run_manager(self) -> RunManager:
        return self.server.run_manager

    @property
    def hardware_monitor(self) -> HardwareMonitor:
        return self.server.hardware_monitor

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def _send_bytes(self, payload: bytes, *, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_file(self, file_path: Path, *, content_type: str, download_name: str | None = None, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(file_path.stat().st_size))
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{quote(download_name)}"')
        self.end_headers()
        with file_path.open("rb") as handle:
            shutil.copyfileobj(handle, self.wfile)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(data, content_type="application/json; charset=utf-8", status=status)

    def _send_text(self, text: str, *, content_type: str, status: int = 200) -> None:
        self._send_bytes(text.encode("utf-8"), content_type=f"{content_type}; charset=utf-8", status=status)

    def _read_json_body(self) -> dict:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("请求体不是合法 JSON") from None
        if not isinstance(data, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return data

    def _handle_exception(self, exc: Exception) -> None:
        status = HTTPStatus.BAD_REQUEST
        if isinstance(exc, FileNotFoundError):
            status = HTTPStatus.NOT_FOUND
        elif isinstance(exc, RuntimeError):
            status = HTTPStatus.CONFLICT
        diagnostic = utils.log.write.exception("[WebUI] 接口请求失败", exc, default_component="WebUI 接口层")
        self._send_json(
            {
                "error": str(exc),
                "diagnosis": diagnostic["summary"],
                "component": diagnostic["component"],
                "error_type": diagnostic["error_type"],
                "location": diagnostic["location"],
                "advice": diagnostic["advice"],
            },
            status=int(status),
        )

    def _normalize_route_path(self, path: str) -> str:
        normalized_path = path.rstrip("/") or "/"
        if normalized_path.startswith("/api/") or normalized_path == "/api":
            return normalized_path
        if normalized_path.startswith("/results/"):
            return normalized_path
        if "/api/" in normalized_path:
            return normalized_path[normalized_path.index("/api/") :]
        if "/results/" in normalized_path:
            return normalized_path[normalized_path.index("/results/") :]
        return normalized_path

    def _is_page_route(self, normalized_path: str) -> bool:
        if normalized_path.startswith("/api") or normalized_path.startswith("/results/"):
            return False
        if normalized_path == "/favicon.ico":
            return False
        return "." not in Path(normalized_path).name

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        normalized_path = self._normalize_route_path(path)

        try:
            if normalized_path == "/" or self._is_page_route(normalized_path):
                return self._send_text(render_index_page(), content_type="text/html")
            if normalized_path == "/api/config":
                return self._send_json({"config": load_effective_config()})
            if normalized_path == "/api/run/status":
                raw_cursor = parse_qs(parsed.query).get("cursor", [None])[0]
                cursor = int(raw_cursor) if raw_cursor not in (None, "") else None
                payload = self.run_manager.snapshot(cursor)
                payload["hardware"] = self.hardware_monitor.snapshot()
                return self._send_json(payload)
            if normalized_path == "/api/results":
                return self._send_json({"results": list_output_runs()})
            if normalized_path == "/api/results/files":
                query = parse_qs(parsed.query)
                output_id = _ensure_string(query.get("output_id", [""])[0], "output_id")
                relative_path = str(query.get("path", [""])[0] or "")
                return self._send_json(list_output_files(output_id, relative_path))
            if normalized_path == "/api/results/file-preview":
                query = parse_qs(parsed.query)
                output_id = _ensure_string(query.get("output_id", [""])[0], "output_id")
                relative_path = _ensure_string(query.get("path", [""])[0], "path")
                return self._send_json(get_output_file_preview(output_id, relative_path))
            if normalized_path == "/api/results/iter-archive":
                query = parse_qs(parsed.query)
                output_id = _ensure_string(query.get("output_id", [""])[0], "output_id")
                relative_path = _ensure_string(query.get("path", [""])[0], "path")
                return self._serve_iter_archive(output_id, relative_path)
            if normalized_path.startswith("/results/"):
                return self._serve_output_file(normalized_path)
            if normalized_path == "/favicon.ico":
                return self._send_bytes(b"", content_type="image/x-icon", status=204)
            raise FileNotFoundError("接口不存在")
        except Exception as exc:  # noqa: BLE001
            self._handle_exception(exc)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        normalized_path = self._normalize_route_path(path)

        try:
            if normalized_path == "/api/config":
                payload = self._read_json_body()
                return self._send_json({"config": save_config(payload)})
            if normalized_path == "/api/run/start":
                payload = self._read_json_body()
                config = self.run_manager.start(payload)
                return self._send_json({"config": config, "message": "任务已启动"})
            if normalized_path == "/api/run/stop":
                self._read_json_body()
                return self._send_json({"message": self.run_manager.stop()})
            if normalized_path == "/api/results/analysis/regenerate":
                payload = self._read_json_body()
                output_id = _ensure_string(payload.get("output_id"), "output_id")
                self.run_manager.assert_output_not_active(output_id)
                return self._send_json(regenerate_output_analysis(output_id))
            if normalized_path == "/api/results/delete":
                payload = self._read_json_body()
                output_id = _ensure_string(payload.get("output_id"), "output_id")
                self.run_manager.assert_output_not_active(output_id)
                return self._send_json(delete_output_run(output_id))
            raise FileNotFoundError("接口不存在")
        except Exception as exc:  # noqa: BLE001
            self._handle_exception(exc)

    def _serve_output_file(self, request_path: str) -> None:
        parts = request_path.split("/", 3)
        if len(parts) < 4:
            raise FileNotFoundError("文件路径不完整")
        output_name = parts[2]
        relative_path = parts[3]
        file_path = resolve_output_file(output_name, relative_path)
        mime_type, _ = mimetypes.guess_type(file_path.name)
        content_type = mime_type or "application/octet-stream"
        self._send_file(file_path, content_type=content_type)

    def _serve_iter_archive(self, output_id: str, relative_path: str) -> None:
        self.run_manager.assert_output_not_active(output_id)
        archive_path, download_name = create_iter_archive(output_id, relative_path)
        try:
            self._send_file(archive_path, content_type="application/zip", download_name=download_name)
        finally:
            try:
                archive_path.unlink(missing_ok=True)
                archive_path.parent.rmdir()
            except OSError:
                pass


def render_index_page() -> str:
    page = HTML_TEMPLATE.replace("__FORM_SCHEMA__", json.dumps(FORM_SCHEMA, ensure_ascii=False))
    page = page.replace("__TASK_META__", json.dumps({str(key): value for key, value in TASK_META.items()}, ensure_ascii=False))
    return page


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LMSV 本地 WebUI")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认 127.0.0.1")
    parser.add_argument("--port", type=int, default=8777, help="监听端口，默认 8765")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mimetypes.add_type("image/svg+xml", ".svg")
    OUTPUT_ROOT.mkdir(exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), LMSVRequestHandler)
    server.run_manager = RunManager()
    server.hardware_monitor = HardwareMonitor()

    print(f"LMSV WebUI listening on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping WebUI...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
