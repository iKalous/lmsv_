#!/usr/bin/env python3
"""
OpenSora推理包装器 - 捕获显存信息
在lmsv_rec内，不修改MindSpeed-MM代码
"""

import sys
import os
import runpy

# 获取MindSpeed-MM路径
mm_path = os.environ.get('MINDSPEED_MM_PATH')
if not mm_path:
    raise RuntimeError("MINDSPEED_MM_PATH environment variable is not set")
is_pta = 'pta' in sys.argv[0].lower() or os.environ.get('PTA_NAME') == 'mindspeed'
mindspeed_mm_path = mm_path

# 切换到MindSpeed-MM目录
os.chdir(mindspeed_mm_path)

# 添加路径
if mindspeed_mm_path not in sys.path:
    sys.path.insert(0, mindspeed_mm_path)

# 保存原始argv，替换为wrapper的argv
original_argv = sys.argv[:]
sys.argv = [os.path.join(mindspeed_mm_path, 'inference_sora.py')] + sys.argv[1:]

try:
    # 使用runpy执行inference_sora，这会设置__name__='__main__'
    runpy.run_path(os.path.join(mindspeed_mm_path, 'inference_sora.py'), run_name='__main__')
except SystemExit as e:
    if e.code != 0 and e.code is not None:
        sys.exit(e.code)

# 捕获显存
try:
    import torch_npu
    memory_mb = torch_npu.npu.max_memory_allocated() / 1024**2
    print(f"\n[NPU memory] max allocated: {memory_mb:.2f} MB", flush=True)
except Exception as e:
    print(f"\n[NPU memory] Failed to get memory: {e}", flush=True)
