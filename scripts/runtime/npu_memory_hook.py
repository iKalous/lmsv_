#!/usr/bin/env python3
"""
NPU显存捕获钩子 - 在Python进程退出时自动捕获显存
通过PYTHONPATH注入
"""

import atexit
import sys
import os

def print_npu_memory():
    """在进程退出时打印NPU显存使用"""
    try:
        import torch_npu
        memory_mb = torch_npu.npu.max_memory_allocated() / 1024**2
        print(f"\n[NPU memory] max allocated: {memory_mb:.2f} MB", file=sys.stderr)
        print(f"\n[NPU memory] max allocated: {memory_mb:.2f} MB", file=sys.stdout)
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass

# 注册退出钩子
atexit.register(print_npu_memory)
