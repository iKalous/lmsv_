#!/bin/bash
# Task6 MSA环境变量设置脚本
# 在使用msadapter conda环境前source此脚本 - 与Task1保持一致

# 动态定位 conda 并激活环境
CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -z "${CONDA_BASE}" ]; then
    echo "错误: 未找到 conda 安装路径，请确认 conda 已安装并在 PATH 中"
    return 1
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${MSA_NAME:-msadapter}"

# 保存可能已预设的变量（多机模式）
_PRESERVED_WORKSPACE_ROOT="${WORKSPACE_ROOT}"
_PRESERVED_MINDSPEED_PATH="${MINDSPEED_PATH}"

# 设置CANN环境
source /usr/local/Ascend/cann/set_env.sh 2>/dev/null || true

# 设置基本环境变量（与PTA相同）
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=1200
# HCCL 通信需占用一段连续端口，参考 Task4-5 预留端口避免冲突 (error code 7)
if [ -z "${HCCL_IF_BASE_PORT}" ]; then
    export HCCL_IF_BASE_PORT=61000
fi
RESERVED_PORTS="${HCCL_IF_BASE_PORT}-$((${HCCL_IF_BASE_PORT} + 15))"
if sysctl -w net.ipv4.ip_local_reserved_ports="${RESERVED_PORTS}" 2>/dev/null; then
    echo "[HCCL] 已预留端口 ${RESERVED_PORTS}"
elif sudo -n sysctl -w net.ipv4.ip_local_reserved_ports="${RESERVED_PORTS}" 2>/dev/null; then
    echo "[HCCL] 已预留端口 ${RESERVED_PORTS} (sudo)"
fi
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACLNN_CACHE_LIMIT=100000

# 设置Python路径（必须从环境变量获取）
# MINDSPEED_MM_PATH 可为 workspace root 或 MindSpeed-MM 代码目录
if [ -n "$MINDSPEED_MM_PATH" ]; then
    # 自动判断：如果包含 pretrain_*.py 则为代码目录，否则为 workspace root
    if [ -f "${MINDSPEED_MM_PATH}/pretrain_vlm.py" ] || [ -f "${MINDSPEED_MM_PATH}/pretrain_sora.py" ]; then
        MM_PATH="$MINDSPEED_MM_PATH"
        WORKSPACE_ROOT=$(dirname "$MINDSPEED_MM_PATH")
    else
        WORKSPACE_ROOT="$MINDSPEED_MM_PATH"
        MM_PATH="${MINDSPEED_MM_PATH}/MindSpeed-MM"
    fi
elif [ -n "$MSAPATH" ]; then
    MM_PATH="$MSAPATH"
    WORKSPACE_ROOT="$MSAPATH"
elif [ -n "$MSA_PATH" ]; then
    MM_PATH="$MSA_PATH"
    WORKSPACE_ROOT="$MSA_PATH"
else
    echo "错误: 未设置 MINDSPEED_MM_PATH 或 MSA_PATH 环境变量"
    return 1
fi

# 多机模式下复用预设的 WORKSPACE_ROOT 和 MINDSPEED_PATH
if [ -n "${_PRESERVED_WORKSPACE_ROOT}" ] && [ -d "${_PRESERVED_WORKSPACE_ROOT}/Megatron-LM" ]; then
    WORKSPACE_ROOT="${_PRESERVED_WORKSPACE_ROOT}"
fi
if [ -n "${_PRESERVED_MINDSPEED_PATH}" ] && [ -d "${_PRESERVED_MINDSPEED_PATH}" ]; then
    MINDSPEED_PATH="${_PRESERVED_MINDSPEED_PATH}"
fi

# 推断 lm-sv 根目录（与 workspace root 同级的 lm-sv 目录）
PARENT=$(dirname "${WORKSPACE_ROOT}")
LM_SV_ROOT="${PARENT}/lm-sv"

# 检查 msadapter 路径：优先从 lm-sv 下找，否则 fallback 到 workspace root 或 MSA_PATH
if [ -d "${LM_SV_ROOT}/mm-new/msadapter" ]; then
    MSADAPTER_PATH="${LM_SV_ROOT}/mm-new/msadapter"
elif [ -d "${WORKSPACE_ROOT}/msadapter" ]; then
    MSADAPTER_PATH="${WORKSPACE_ROOT}/msadapter"
elif [ -n "${MSA_PATH}" ] && [ -d "${MSA_PATH}/msadapter" ]; then
    MSADAPTER_PATH="${MSA_PATH}/msadapter"
elif [ -n "${MSAPATH}" ] && [ -d "${MSAPATH}/msadapter" ]; then
    MSADAPTER_PATH="${MSAPATH}/msadapter"
else
    echo "错误: 未找到 msadapter 目录（已尝试 ${LM_SV_ROOT}/mm-new/msadapter、${WORKSPACE_ROOT}/msadapter 等位置）"
    return 1
fi

# 检查 Megatron-LM（必须在 workspace root 下）
if [ ! -d "${WORKSPACE_ROOT}/Megatron-LM" ]; then
    echo "错误: WORKSPACE_ROOT (${WORKSPACE_ROOT}) 下缺少 Megatron-LM 目录"
    return 1
fi

# MindSpeed 路径：优先从 lm-sv 下找，否则 fallback 到 workspace root
if [ -z "${MINDSPEED_PATH}" ]; then
    if [ -d "${LM_SV_ROOT}/mm-new/MindSpeed" ]; then
        MINDSPEED_PATH="${LM_SV_ROOT}/mm-new/MindSpeed"
    elif [ -d "${WORKSPACE_ROOT}/MindSpeed" ]; then
        MINDSPEED_PATH="${WORKSPACE_ROOT}/MindSpeed"
    fi
fi

# 修复 numpy/C++ 库版本不匹配（优先使用当前 conda 环境的 libstdc++）
if [ -n "${CONDA_PREFIX}" ] && [ -f "${CONDA_PREFIX}/lib/libstdc++.so.6" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
fi

# 设置 PYTHONPATH
PYTHONPATH_ENTRIES="${MSADAPTER_PATH}:${MSADAPTER_PATH}/msa_thirdparty"
PYTHONPATH_ENTRIES="${PYTHONPATH_ENTRIES}:${WORKSPACE_ROOT}/Megatron-LM"
if [ -n "$MINDSPEED_PATH" ]; then
    PYTHONPATH_ENTRIES="${PYTHONPATH_ENTRIES}:${MINDSPEED_PATH}"
fi
PYTHONPATH_ENTRIES="${PYTHONPATH_ENTRIES}:${MM_PATH}"
export PYTHONPATH=${PYTHONPATH_ENTRIES}:$PYTHONPATH

echo "MSA环境变量设置完成"
echo "MM_PATH: $MM_PATH"
echo "WORKSPACE_ROOT: $WORKSPACE_ROOT"
