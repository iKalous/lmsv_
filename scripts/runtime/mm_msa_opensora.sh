#!/bin/bash
# Task6 MSA真实执行脚本 - OpenSora (推理模式，不需要data_config)
set -e

# 设置输出路径（未设置时使用默认值）
export LMSV_OUTPATH="${LMSV_OUTPATH:-output}"

# 设置环境名称（未设置时使用默认值）
export PTA_NAME="${PTA_NAME:-mindspeed}"
export MSA_NAME="${MSA_NAME:-msadapter}"

echo "========================================"
echo "Task6 MSA Real Execution - OpenSora (Inference)"
echo "MM_MODEL: $MM_MODEL"
echo "========================================"

# 清理端口占用（防止上一次执行未正常结束）
echo "Cleaning up ports..."
fuser -k 6000/tcp 2>/dev/null || true
fuser -k 6001/tcp 2>/dev/null || true
fuser -k 6002/tcp 2>/dev/null || true
pkill -f "msrun" 2>/dev/null || true
sleep 2

if [ -z "$MM_MODEL" ]; then
    echo "ERROR: MM_MODEL not set"
    exit 1
fi

if [ ! -f "$MM_MODEL" ]; then
    echo "ERROR: Model config not found: $MM_MODEL"
    exit 1
fi

echo "Configuration check PASSED"
echo "Using mutated config: $MM_MODEL"
echo "Note: OpenSora is inference mode, no data_config needed"

# 检查数据集根目录环境变量（必须设置，不可硬编码）
if [ -z "$DATASET_ROOT" ]; then
    echo "ERROR: DATASET_ROOT environment variable is not set"
    echo "Please set it in config.json or export DATASET_ROOT=/path/to/dataset"
    exit 1
fi
# 注：OpenSora是推理模型，不需要LOAD_PATH

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMSV_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# MSA执行时不清空日志（Task6主入口已清空）
LOG_DIR="${LMSV_ROOT}/msrun_log"
mkdir -p ${LOG_DIR}

# 处理配置文件 - 使用配置预处理脚本（替换{{LOAD_PATH}}等占位符，tmp/task6由Task6主入口清空）
mkdir -p "${LMSV_ROOT}/tmp/task6"
TMP_MODEL_CONFIG="${LMSV_ROOT}/tmp/task6/model_config_opensora_msa_$(date +%s).json"

# 调用配置预处理脚本处理模型配置（变异后的配置可能包含占位符）
bash "${SCRIPT_DIR}/prepare_mm_config.sh" "${MM_MODEL}" "${TMP_MODEL_CONFIG}" "opensora"

# 更新为临时文件
export MM_MODEL="${TMP_MODEL_CONFIG}"
echo "Processed model config: ${TMP_MODEL_CONFIG}"
echo "Note: No data_config needed for inference model"

# Modify device for MSA: ensure device is "npu" (not "npu:0" or "npu:Ascend")
# MSA MindSpeed-MM get_device only supports "npu" or "npu:0/1/2..." format
echo "Checking device field in config before fix..."
grep -o '"device"[[:space:]]*:[[:space:]]*"[^"]*"' "${MM_MODEL}" | head -5

# Force device to simple "npu" format using Python for reliable JSON handling
python3 - "${MM_MODEL}" << 'PYEOF'
import json
import sys

config_path = sys.argv[1]
with open(config_path, 'r') as f:
    config = json.load(f)

# Recursively fix all device fields
def fix_device(obj, path=""):
    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            if key == 'device' and isinstance(value, str):
                if value != 'npu':
                    print(f"Fixing {path}.{key}: '{value}' -> 'npu'")
                    obj[key] = 'npu'
            elif isinstance(value, (dict, list)):
                fix_device(value, f"{path}.{key}" if path else key)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            fix_device(item, f"{path}[{i}]")

fix_device(config, "")

# Write back
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print("Device field fix completed")
PYEOF

echo "Device field after fix:"
grep -o '"device"[[:space:]]*:[[:space:]]*"[^"]*"' "${MM_MODEL}" | head -5

# 设置MindSpeed-MM路径（MSA环境）
if [ -n "$MINDSPEED_MM_PATH" ]; then
    # 将MINDSPEED_MM_PATH转换为绝对路径（支持相对路径）
    if [[ "$MINDSPEED_MM_PATH" = /* ]]; then
        MM_PATH_ABS="$MINDSPEED_MM_PATH"
    else
        MM_PATH_ABS="$(cd "$MINDSPEED_MM_PATH" && pwd)"
    fi
    # Auto-derive MindSpeed-MM subdirectory if path is workspace root
    if [ ! -f "${MM_PATH_ABS}/pretrain_vlm.py" ] && [ ! -f "${MM_PATH_ABS}/pretrain_sora.py" ]; then
        DERIVED="${MM_PATH_ABS}/MindSpeed-MM"
        if [ -d "$DERIVED" ] && ([ -f "${DERIVED}/pretrain_vlm.py" ] || [ -f "${DERIVED}/pretrain_sora.py" ]); then
            MM_PATH_ABS="$DERIVED"
            echo "Auto-derived MindSpeed-MM path: ${MM_PATH_ABS}"
        fi
    fi
    export MINDSPEED_MM_PATH="$MM_PATH_ABS"
else
    echo "ERROR: MINDSPEED_MM_PATH environment variable not set"
    exit 1
fi

# 按照文档设置正确的PYTHONPATH
export PYTHONPATH="${MM_PATH_ABS}/msadapter:${MM_PATH_ABS}/msadapter/msa_thirdparty:${MM_PATH_ABS}/Megatron-LM:${MM_PATH_ABS}/MindSpeed:${MM_PATH_ABS}/MindSpeed-MM:${PYTHONPATH}"
echo "MindSpeed-MM path (MSA): ${MINDSPEED_MM_PATH}"
echo "PYTHONPATH: ${PYTHONPATH}"

# MSA环境: 创建 UntypedStorage mock 以支持 safetensors 加载
# 这是解决 msadapter 缺少 UntypedStorage 的临时方案
# 使用 sitecustomize.py 在 Python 启动时自动注入
MSA_PATCH_DIR="${LMSV_ROOT}/tmp/task6/msa_patch"
mkdir -p "${MSA_PATCH_DIR}"

# 创建 sitecustomize.py 会在 Python 启动时自动执行
cat > "${MSA_PATCH_DIR}/sitecustomize.py" << 'PATCH_EOF'
"""
Patch for msadapter to support UntypedStorage needed by safetensors
Auto-executed by Python at startup (via sitecustomize mechanism)
"""
import sys
import builtins

# Create a more complete UntypedStorage mock that works with safetensors
class FakeUntypedStorageMeta(type):
    """Metaclass to make FakeUntypedStorage callable like a real storage class"""
    def __call__(cls, *args, **kwargs):
        instance = cls.__new__(cls)
        instance.__init__(*args, **kwargs)
        return instance

class FakeUntypedStorage(metaclass=FakeUntypedStorageMeta):
    """Mock UntypedStorage for safetensors compatibility"""

    def __init__(self, size=0, *args, **kwargs):
        self._data = bytearray(size)
        self._size = size

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        return self._data[idx]

    def __setitem__(self, idx, value):
        self._data[idx] = value

    @classmethod
    def _new_with_file(cls, filename, size, offset=0):
        """Create storage from file (used by safetensors)."""
        instance = cls.__new__(cls)
        with open(filename, 'rb') as f:
            if offset > 0:
                f.seek(offset)
            data = f.read(size) if size > 0 else f.read()
            instance._data = bytearray(data)
            instance._size = len(instance._data)
        return instance

    @classmethod
    def _from_buffer(cls, buffer, byte_order=None):
        instance = cls.__new__(cls)
        if isinstance(buffer, (bytes, bytearray)):
            instance._data = bytearray(buffer)
        else:
            instance._data = bytearray()
        instance._size = len(instance._data)
        return instance

    @classmethod
    def from_file(cls, filename, shared, nbytes):
        """Create storage from file (for torch.load compatibility)."""
        return cls._new_with_file(filename, nbytes)

    @classmethod
    def from_buffer(cls, buffer, byte_order='native'):
        """Create storage from buffer."""
        return cls._from_buffer(buffer, byte_order)

    def resize_(self, size):
        if size > len(self._data):
            self._data.extend(b'\x00' * (size - len(self._data)))
        elif size < len(self._data):
            self._data = self._data[:size]
        self._size = size
        return self

    def clone(self):
        """Clone the storage."""
        new_storage = FakeUntypedStorage.__new__(FakeUntypedStorage)
        new_storage._data = bytearray(self._data)
        new_storage._size = self._size
        return new_storage

    def copy_(self, other):
        """Copy from another storage."""
        if isinstance(other, FakeUntypedStorage):
            self._data = bytearray(other._data)
            self._size = other._size
        return self

    def nbytes(self):
        """Return number of bytes."""
        return len(self._data)

    def data_ptr(self):
        """Return data pointer (mock)."""
        return 0

    def element_size(self):
        """Return element size (1 byte for untyped)."""
        return 1

    def fill_(self, value):
        """Fill storage with value."""
        for i in range(len(self._data)):
            self._data[i] = value
        return self

# Inject into msadapter before safetensors tries to use it
try:
    import msadapter
    if not hasattr(msadapter, 'UntypedStorage'):
        msadapter.UntypedStorage = FakeUntypedStorage
        sys.modules['msadapter.UntypedStorage'] = FakeUntypedStorage
except ImportError:
    pass

# Also inject into torch module proxy
try:
    import torch
    if not hasattr(torch, 'UntypedStorage'):
        torch.UntypedStorage = FakeUntypedStorage
except ImportError:
    pass

# Inject into torch.storage module if it exists
try:
    import torch.storage
    if not hasattr(torch.storage, 'UntypedStorage'):
        torch.storage.UntypedStorage = FakeUntypedStorage
except ImportError:
    pass

# Also inject at builtins level for maximum compatibility
try:
    if not hasattr(builtins, 'UntypedStorage'):
        builtins.UntypedStorage = FakeUntypedStorage
except:
    pass
PATCH_EOF

# 将 patch 目录加入 PYTHONPATH 最前面，确保最先加载
export PYTHONPATH="${MSA_PATCH_DIR}:${PYTHONPATH}"
echo "Applied UntypedStorage patch for MSA compatibility"
echo "Patch location: ${MSA_PATCH_DIR}/sitecustomize.py"

bash "${SCRIPT_DIR}/msa_opensora_real.sh"
