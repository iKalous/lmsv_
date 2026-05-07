#!/usr/bin/env python3
"""
VLM推理包装器 - 捕获显存信息
在lmsv_rec内，不修改MindSpeed-MM代码
"""

import sys
import os

# 添加MindSpeed-MM路径
mm_path = os.environ.get('MINDSPEED_MM_PATH')
if not mm_path:
    raise RuntimeError("MINDSPEED_MM_PATH environment variable is not set")
mindspeed_mm_path = mm_path

if mindspeed_mm_path not in sys.path:
    sys.path.insert(0, mindspeed_mm_path)

# 导入原始inference_vlm
exec(open(os.path.join(mindspeed_mm_path, 'inference_vlm.py')).read())

# 在main()执行后捕获显存
if __name__ == '__main__':
    import torch_npu
    main()
    # 捕获显存并输出到日志
    memory_mb = torch_npu.npu.max_memory_allocated() / 1024**2
    print(f"\n[NPU memory] max allocated: {memory_mb:.2f} MB")
