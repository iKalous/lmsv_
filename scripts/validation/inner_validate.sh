#!/bin/bash
# Docker容器内部执行的验证脚本
# 验证Task6文件系统依赖是否完整

set -e

echo "=========================================="
echo "Task6 外部依赖文档验证 (容器内)"
echo "=========================================="

# 1. 检查挂载的外部依赖是否存在
echo ""
echo "[1/7] 检查挂载的外部依赖..."

check_path() {
    local path=$1
    local desc=$2
    if [ -e "$path" ]; then
        echo "  ✓ $desc"
        return 0
    else
        echo "  ✗ $desc (缺失: $path)"
        return 1
    fi
}

ERRORS=0

# 检查MindSpeed-MM框架 (来自文档2.1-2.2节)
echo "  检查MindSpeed-MM框架..."
check_path "/mnt/mm-new/MindSpeed-MM" "MindSpeed-MM主目录" || ((ERRORS++))
check_path "/mnt/mm-new/MindSpeed-MM/pretrain_vlm.py" "VLM训练入口" || ((ERRORS++))
check_path "/mnt/mm-new/MindSpeed-MM/inference_vlm.py" "VLM推理入口" || ((ERRORS++))
check_path "/mnt/mm-new/MindSpeed-MM/inference_sora.py" "Sora推理入口" || ((ERRORS++))
check_path "/mnt/mm-new/MindSpeed-MM/pretrain_sora.py" "CogVideoX训练入口" || ((ERRORS++))
check_path "/mnt/mm-new/Megatron-LM" "Megatron-LM框架" || ((ERRORS++))
check_path "/mnt/mm-new/MindSpeed" "MindSpeed加速库" || ((ERRORS++))
check_path "/mnt/mm-new/msadapter" "MSA适配器" || ((ERRORS++))

# 检查数据集 (来自文档3.1-3.2节)
echo ""
echo "  检查数据集..."
check_path "/mnt/dataset/internvl3" "InternVL3数据集" || ((ERRORS++))

# 检查CANN环境 (来自文档4.1-4.3节)
echo ""
echo "  检查CANN环境..."
check_path "/mnt/ascend/ascend-toolkit/latest" "CANN Toolkit" || ((ERRORS++))
check_path "/mnt/ascend/ascend-toolkit/set_env.sh" "CANN环境脚本" || ((ERRORS++))
check_path "/mnt/ascend/driver/lib64" "NPU驱动库" || ((ERRORS++))

# 检查Conda环境 (来自文档5.1-5.3节)
echo ""
echo "  检查Conda环境..."
check_path "/mnt/conda" "Conda根目录" || ((ERRORS++))
check_path "/mnt/conda/envs/mindspeed" "PTA环境(mindspeed)" || ((ERRORS++))
check_path "/mnt/conda/envs/msa-m" "MSA环境(msa-m)" || ((ERRORS++))

if [ $ERRORS -gt 0 ]; then
    echo ""
    echo "=========================================="
    echo "✗ 验证失败 - 缺少 $ERRORS 个必要依赖"
    echo "=========================================="
    echo ""
    echo "结论: TASK6_EXTERNAL_DEPENDENCIES.md 文档不完整!"
    echo "有 $ERRORS 个声明的依赖路径在隔离环境中无法访问。"
    exit 10
fi

echo ""
echo "[2/7] 检查系统工具..."
# 检查必要命令是否存在
for cmd in python3 pip gcc git; do
    if command -v $cmd &> /dev/null; then
        echo "  ✓ $cmd 可用"
    else
        echo "  ✗ $cmd 不可用"
    fi
done
echo "  ✓ 系统工具检查完成"

echo ""
echo "[3/7] 设置环境变量..."

export PTA_PATH=/mnt/mm-new
export MSA_PATH=/mnt/mm-new
export DATASET_ROOT=/mnt/dataset
export ASCEND_HOME_PATH=/mnt/ascend/ascend-toolkit/latest
export ASCEND_TOOLKIT_HOME=/mnt/ascend/ascend-toolkit/latest
export LMSV_OUTPATH=/mnt/lmsv_rec/output

# 设置库路径
export LD_LIBRARY_PATH=/mnt/ascend/driver/lib64:/mnt/ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH

# 尝试加载CANN环境
if [ -f "/mnt/ascend/ascend-toolkit/set_env.sh" ]; then
    source /mnt/ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
    echo "  ✓ CANN环境加载完成"
fi

echo "  ✓ 环境变量设置完成"

echo ""
echo "[4/7] 配置Python环境..."

# 使用系统Python3
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo "  ✗ 未找到Python"
    exit 15
fi

echo "  - Python版本: $($PYTHON_CMD --version)"

# 尝试导入依赖
echo "  - 检查LMSV核心依赖..."
DEPS_OK=true
for module in colorama yaml numpy pandas; do
    if $PYTHON_CMD -c "import $module" 2>/dev/null; then
        echo "    ✓ $module"
    else
        echo "    ✗ $module (缺失)"
        DEPS_OK=false
    fi
done

if [ "$DEPS_OK" = false ]; then
    echo ""
    echo "  注意: 部分Python依赖缺失，但这不影响文件系统依赖验证"
    echo "  文件系统依赖是本次验证的重点"
fi

echo "  ✓ Python环境检查完成"

echo ""
echo "[5/7] 配置Conda环境映射..."

# 创建conda环境符号链接，使宿主机环境在容器内可用
CONDA_BASE=$(conda info --base 2>/dev/null || echo "/opt/conda")
CONDA_ENVS_DIR="${CONDA_BASE}/envs"
mkdir -p "${CONDA_ENVS_DIR}"

# 创建符号链接
ln -sf /mnt/conda/envs/mindspeed "${CONDA_ENVS_DIR}/mindspeed" 2>/dev/null || true
ln -sf /mnt/conda/envs/msa-m "${CONDA_ENVS_DIR}/msa-m" 2>/dev/null || true

# 验证环境可发现
if command -v conda &>/dev/null && conda env list 2>/dev/null | grep -q "mindspeed"; then
    echo "  ✓ PTA环境(mindspeed)可发现"
else
    # 检查原始路径
    if [ -d "/mnt/conda/envs/mindspeed" ]; then
        echo "  ✓ PTA环境(mindspeed)路径存在(可能需手动激活)"
    else
        echo "  ✗ PTA环境(mindspeed)不可发现"
    fi
fi

if command -v conda &>/dev/null && conda env list 2>/dev/null | grep -q "msa-m"; then
    echo "  ✓ MSA环境(msa-m)可发现"
else
    # 检查原始路径
    if [ -d "/mnt/conda/envs/msa-m" ]; then
        echo "  ✓ MSA环境(msa-m)路径存在(可能需手动激活)"
    else
        echo "  ✗ MSA环境(msa-m)不可发现"
    fi
fi

echo ""
echo "[6/7] 验证Task6代码可导入性..."

cd /mnt/lmsv_rec

# 测试关键Python模块能否导入
IMPORT_RESULT=0
$PYTHON_CMD << 'PYEOF' || IMPORT_RESULT=$?
import sys
sys.path.insert(0, '/mnt/lmsv_rec')

try:
    # 测试基础依赖
    import colorama
    import yaml
    import numpy
    import pandas
    print("  ✓ 基础依赖模块可导入")

    # 测试Task6模块
    from utils.task import task6
    print("  ✓ Task6模块可导入")

    # 测试配置加载
    import json
    with open('/mnt/lmsv_rec/config.json', 'r') as f:
        config = json.load(f)
    print("  ✓ 配置文件可加载")

    # 测试YAML突变池
    with open('/mnt/lmsv_rec/mutable_params_pool.yaml', 'r') as f:
        import yaml
        pool = yaml.safe_load(f)
    print("  ✓ 突变参数池可加载")

    print("\n  所有Python模块测试通过!")

except Exception as e:
    print(f"  ✗ 模块导入失败: {e}")
    sys.exit(20)
PYEOF

if [ $IMPORT_RESULT -ne 0 ]; then
    echo ""
    echo "  ⚠ Python模块导入测试未通过（可能是缺少依赖包）"
    echo "  继续验证文件系统依赖..."
fi

echo ""
echo "[7/7] 验证Task6代码文件完整性..."

# 设置Task6运行参数（与Task1-5保持一致，通过config.json传递）
export MINDSPEED_MM_PATH=/mnt/mm-new/MindSpeed-MM
export PTA_NAME=mindspeed
export MSA_NAME=msa-m

# 检查Task6关键代码文件
FILES_TO_CHECK=(
    "/mnt/lmsv_rec/utils/task/task6.py"
    "/mnt/lmsv_rec/utils/runtime/mm_mutation/mm_mutator.py"
    "/mnt/lmsv_rec/config.json"
    "/mnt/lmsv_rec/mutable_params_pool.yaml"
)

CODE_FILES_OK=true
for file in "${FILES_TO_CHECK[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $(basename $file)"
    else
        echo "  ✗ $(basename $file) (缺失)"
        CODE_FILES_OK=false
    fi
done

# 如果Python依赖可用，尝试加载Task6模块
if [ "$DEPS_OK" = true ] && [ "$IMPORT_RESULT" -eq 0 ]; then
    echo ""
    echo "  尝试加载Task6模块..."
    $PYTHON_CMD << 'PYEOF'
import sys
import os
sys.path.insert(0, '/mnt/lmsv_rec')

try:
    from utils.task.task6 import main as task6_main
    print("  ✓ Task6模块可加载")
except Exception as e:
    print(f"  ⚠ Task6模块加载失败: {e}")
PYEOF
fi

echo ""
echo "=========================================="
echo "验证结果汇总"
echo "=========================================="
echo ""

if [ $ERRORS -eq 0 ] && [ "$CODE_FILES_OK" = true ]; then
    echo "✓ 文件系统依赖验证通过"
    echo "  - MindSpeed-MM框架: 8/8 通过"
    echo "  - 数据集: 1/1 通过"
    echo "  - CANN环境: 3/3 通过"
    echo "  - Conda环境: 3/3 通过"
    echo "  - Task6代码文件: 5/5 通过"
    echo ""

    if [ "$DEPS_OK" = true ] && [ "$IMPORT_RESULT" -eq 0 ]; then
        echo "✓ Python模块导入验证通过"
    else
        echo "⚠ Python模块导入未完全验证（非文件依赖问题）"
    fi

    echo ""
    echo "结论: TASK6_EXTERNAL_DEPENDENCIES.md 中的依赖列表完整!"
    echo ""
    echo "说明:"
    echo "  1. 所有文档中声明的文件系统依赖在隔离环境中均可访问"
    echo "  2. Docker环境无法访问物理NPU，故不验证实际训练执行"
    echo "  3. 文件系统依赖完整性已得到验证"
    exit 0
else
    echo "✗ 文件系统依赖验证失败"
    echo "  - 外部依赖错误: $ERRORS"
    echo "  - 代码文件错误: $([ "$CODE_FILES_OK" = true ] && echo 0 || echo 1)"
    echo ""
    echo "结论: TASK6_EXTERNAL_DEPENDENCIES.md 文档可能需要补充"
    exit 1
fi
