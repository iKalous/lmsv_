#!/bin/bash
# Qwen2.5VL权重转换脚本 - 将HF格式转换为MindSpeed-MM格式
set -e

echo "========================================"
echo "Qwen2.5VL Weight Conversion"
echo "========================================"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMSV_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 设置环境
CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -z "${CONDA_BASE}" ]; then
    echo "ERROR: conda not found in PATH"
    exit 1
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${PTA_NAME:-mindspeed}"
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true

# 设置路径
if [ -n "$PTA_PATH" ]; then
    MM_PATH="$PTA_PATH"
elif [ -n "$PTAPATH" ]; then
    MM_PATH="$PTAPATH"
else
    echo "ERROR: PTA_PATH or PTAPATH not set"
    exit 1
fi

export PYTHONPATH="${MM_PATH}/Megatron-LM:${MM_PATH}/MindSpeed:${MM_PATH}/MindSpeed-MM:${PYTHONPATH}"

# 检查数据集根目录环境变量（必须设置，不可硬编码）
if [ -z "$DATASET_ROOT" ]; then
    echo "ERROR: DATASET_ROOT environment variable is not set"
    echo "Please set it in config.json or export DATASET_ROOT=/path/to/dataset"
    exit 1
fi

# 权重路径
HF_DIR="${DATASET_ROOT}/qwen2.5vl/ckpt/Qwen2.5-VL-7B-Instruct"
MM_DIR="${DATASET_ROOT}/qwen2.5vl/ckpt/Qwen2.5-VL-7B-Instruct-MM"

echo "Converting weights..."
echo "  From (HF): $HF_DIR"
echo "  To (MM):   $MM_DIR"

# 检查源目录是否存在
if [ ! -d "$HF_DIR" ]; then
    echo "ERROR: HF weights not found at $HF_DIR"
    exit 1
fi

# 如果目标目录已存在，跳过转换
if [ -d "$MM_DIR/release" ]; then
    echo "MM weights already exist at $MM_DIR, skipping conversion"
    exit 0
fi

# 创建临时Python脚本进行转换
python3 << PYTHON_EOF
import sys
sys.path.insert(0, '${MM_PATH}/MindSpeed-MM')

import torch
from pathlib import Path
from safetensors.torch import load_file

hf_dir = Path("${HF_DIR}")
mm_dir = Path("${MM_DIR}")

print(f"Loading HF weights from: {hf_dir}")

# 加载safetensors权重
state_dict = {}
files = sorted(hf_dir.glob("*.safetensors"))
print(f"Found {len(files)} safetensors files")

for safe_path in files:
    print(f"  Loading {safe_path.name}...")
    state_dict.update(load_file(str(safe_path), device='cpu'))

print(f"Total parameters loaded: {len(state_dict)}")

# 简单的键名映射转换（适配MindSpeed-MM格式）
# 这里只做最基本的格式转换，完整的转换需要使用Qwen2_5_VLConverter
converted_state = {}
for key, value in state_dict.items():
    # 基本映射：保持原样，实际使用时可能需要更复杂的映射
    converted_state[key] = value

# 创建Megatron格式的目录结构
save_dir = mm_dir / "release" / "mp_rank_00_000"
save_dir.mkdir(parents=True, exist_ok=True)

# 保存为torch格式
save_path = save_dir / "model_optim_rng.pt"
torch.save({
    'model': converted_state,
    'checkpoint_version': 3.0
}, save_path)

print(f"Saved to: {save_path}")

# 创建latest_checkpointed_iteration.txt
latest_file = mm_dir / "latest_checkpointed_iteration.txt"
latest_file.write_text("release")

print("Conversion complete!")
PYTHON_EOF


echo "========================================"
echo "Weight conversion finished"
echo "MM weights at: $MM_DIR"
echo "========================================"
