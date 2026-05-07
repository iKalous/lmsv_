#!/bin/bash
# Task6 外部依赖验证脚本
# 验证原理：在隔离的Docker环境中只挂载文档中声明的外部依赖
# 如果能成功运行Task6 InternVL3，则证明文档完整；否则文档有遗漏

set -e

echo "=========================================="
echo "Task6 外部依赖文档验证"
echo "=========================================="
echo ""
echo "验证策略:"
echo "1. 创建隔离Docker环境"
echo "2. 只挂载TASK6_EXTERNAL_DEPENDENCIES.md中声明的依赖目录"
echo "3. 运行Task6 InternVL3 (1 iter, 2 steps)"
echo "4. 成功=文档完整，失败=文档有遗漏"
echo ""

# 基础镜像（使用本地镜像，因网络受限无法拉取）
BASE_IMAGE="lm-sv:0.1.0"

# 检查本地镜像
if ! docker images | grep -q "lm-sv.*0.1.0"; then
    echo "[1/5] 错误: 本地镜像 ${BASE_IMAGE} 不存在"
    echo "      可用镜像:"
    docker images | grep -v REPOSITORY | head -5
    exit 1
else
    echo "[1/5] 使用本地镜像: ${BASE_IMAGE}"
fi

echo ""
echo "[2/5] 准备挂载目录..."

# 获取脚本所在目录，计算相对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMSV_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 定义挂载点（支持通过环境变量覆盖，默认使用相对路径）
# mm-new 默认与 lmsv_rec 同级
MM_NEW_HOST_PATH="${MM_NEW_HOST_PATH:-$(cd "$LMSV_ROOT/.." && pwd)/mm-new}"
LMSV_HOST_PATH="${LMSV_HOST_PATH:-$LMSV_ROOT}"

# 检查必需的环境变量
if [ -z "$DATASET_HOST_PATH" ]; then
    echo "ERROR: DATASET_HOST_PATH environment variable is not set"
    exit 1
fi

MOUNT_MM="${MM_NEW_HOST_PATH}:/mnt/mm-new:ro"
MOUNT_DATASET="${DATASET_HOST_PATH}:/mnt/dataset:ro"
MOUNT_ASCEND="${ASCEND_HOST_PATH:-/usr/local/Ascend}:/mnt/ascend:ro"
MOUNT_CONDA="${CONDA_HOST_PATH:-/root/anaconda3}:/mnt/conda:ro"
MOUNT_LMSV="${LMSV_HOST_PATH}:/mnt/lmsv_rec"

echo "  - MindSpeed-MM框架: ${MM_NEW_HOST_PATH}"
echo "  - 数据集: ${DATASET_HOST_PATH}"
echo "  - CANN环境: ${ASCEND_HOST_PATH:-/usr/local/Ascend}"
echo "  - Conda环境: ${CONDA_HOST_PATH:-/root/anaconda3}"
echo "  - LMSV项目: ${LMSV_HOST_PATH}"

echo ""
echo "[3/5] 创建验证容器..."

# 创建并运行验证容器（非交互模式）
docker run --rm \
    --name task6_validation \
    --hostname task6-validator \
    -v "${MOUNT_MM}" \
    -v "${MOUNT_DATASET}" \
    -v "${MOUNT_ASCEND}" \
    -v "${MOUNT_CONDA}" \
    -v "${MOUNT_LMSV}" \
    -e PTA_PATH=/mnt/mm-new \
    -e MSA_PATH=/mnt/mm-new \
    -e DATASET_ROOT=/mnt/dataset \
    -e ASCEND_HOME_PATH=/mnt/ascend/ascend-toolkit/latest \
    -e ASCEND_TOOLKIT_HOME=/mnt/ascend/ascend-toolkit/latest \
    -e LMSV_OUTPATH=/mnt/lmsv_rec/output \
    -e MINDSPEED_MM_PATH=/mnt/mm-new/MindSpeed-MM \
    ${BASE_IMAGE} \
    /bin/bash /mnt/lmsv_rec/scripts/validation/inner_validate.sh

EXIT_CODE=$?

echo ""
echo "[5/5] 验证结果:"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ 验证通过 - 文档中的依赖列表完整"
else
    echo "✗ 验证失败 - 文档中缺少必要的依赖"
    echo "  退出码: $EXIT_CODE"
fi

exit $EXIT_CODE
