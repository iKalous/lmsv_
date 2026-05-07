# Task6 外部依赖说明文档

> **作者**: 邹英龙
> **更新日期**: 2026-04-13
> **适用范围**: Task6 多模态整网变异和验证

本文档详细说明 Task6 对外部文件的依赖关系，包括 MindSpeed-MM 框架、数据集、CANN 环境、Conda 环境等。

---

## 1. 依赖概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Task6 外部依赖架构图                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐ │
│   │   PTA 环境       │      │   MSA 环境       │      │   数据集         │ │
│   │                  │      │                  │      │                  │ │
│   │  MindSpeed-MM    │◄────►│  MindSpeed-MM    │      │  /data2/dataset  │ │
│   │  Megatron-LM     │      │  msadapter       │      │                  │ │
│   │  MindSpeed       │      │  MindSpeed       │      │  InternVL3       │ │
│   │                  │      │                  │      │  QwenVL          │ │
│   └────────┬─────────┘      └────────┬─────────┘      │  OpenSora        │ │
│            │                         │                │  CogVideoX       │ │
│            │    ┌─────────────────┐  │                └────────┬─────────┘ │
│            └───►│   lmsv_rec      │◄─┘                         │           │
│                 │   (Task6)       │◄────────────────────────────┘           │
│                 └────────┬────────┘                                         │
│                          │                                                  │
│                 ┌────────▼────────┐      ┌──────────────────┐               │
│                 │   CANN 环境     │      │   Conda 环境     │               │
│                 │                 │      │                  │               │
│                 │  ascend-toolkit │      │  mindspeed (PTA) │               │
│                 │  driver         │      │  msa-m (MSA)     │               │
│                 └─────────────────┘      └──────────────────┘               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. MindSpeed-MM 框架依赖

### 2.1 MindSpeed-MM 路径 (`${MINDSPEED_MM_PATH}`)

Task6 统一使用 `MINDSPEED_MM_PATH` 指向 MindSpeed-MM 工作区根目录（兼容旧版 `PTA_PATH`/`MSA_PATH`）。
框架会自动检测：如果路径下存在 `MindSpeed-MM` 子目录，则自动推导到该子目录；否则直接使用当前路径。

| 路径 | 必需 | 说明 |
|------|------|------|
| `${MINDSPEED_MM_PATH}` | ✅ | MindSpeed-MM 多模态模型框架主目录 |
| `${MINDSPEED_MM_PATH}/pretrain_vlm.py` | ✅ | VLM 训练入口脚本 |
| `${MINDSPEED_MM_PATH}/inference_vlm.py` | ✅ | VLM 推理入口脚本 |
| `${MINDSPEED_MM_PATH}/inference_sora.py` | ✅ | OpenSora 推理入口脚本 |
| `${MINDSPEED_MM_PATH}/pretrain_sora.py` | ✅ | CogVideoX 训练入口脚本 |
| `${MINDSPEED_MM_PATH}/examples/internvl3/finetune_internvl3_8B.sh` | ✅ | InternVL3 PTA 训练脚本 |
| `${MINDSPEED_MM_PATH}/examples/qwen2.5vl/finetune_qwen2_5_vl_7b.sh` | ✅ | QwenVL PTA 训练脚本 |
| `${MINDSPEED_MM_PATH}/examples/opensora1.2/inference_opensora1_2.sh` | ✅ | OpenSora PTA 推理脚本 |
| `${MINDSPEED_MM_PATH}/examples/cogvideox/i2v_1.5/pretrain_cogvideox_i2v_1.5.sh` | ✅ | CogVideoX PTA 训练脚本 |
| `${MINDSPEED_MM_PATH}/scripts-ms/finetune_internvl3_8B.sh` | ✅ | InternVL3 MSA 训练脚本 |
| `${MINDSPEED_MM_PATH}/scripts-ms/finetune_qwen2_5_vl_7b.sh` | ✅ | QwenVL MSA 训练脚本 |
| `${MINDSPEED_MM_PATH}/scripts-ms/inference_opensora1_2.sh` | ✅ | OpenSora MSA 推理脚本 |
| `${MINDSPEED_MM_PATH}/scripts-ms/pretrain_cogvideox_i2v_1.5.sh` | ✅ | CogVideoX MSA 训练脚本 |
| `$(dirname ${MINDSPEED_MM_PATH})/Megatron-LM` | ✅ | Megatron-LM 并行训练框架 |
| `$(dirname ${MINDSPEED_MM_PATH})/MindSpeed` | ✅ | MindSpeed 加速库 |
| `$(dirname ${MINDSPEED_MM_PATH})/msadapter` | ✅ | MindSpore Adapter 适配层 |
| `$(dirname ${MINDSPEED_MM_PATH})/msadapter/msa_thirdparty` | ✅ | MSA 第三方依赖库 |
| `$(dirname ${MINDSPEED_MM_PATH})/MindSpeed-LLM` | ⚠️ | MindSpeed-LLM 大模型训练框架（部分脚本需要） |

**设置方式**（相对路径或绝对路径）：
```bash
# 绝对路径示例（推荐）：
export MINDSPEED_MM_PATH=/shared/mindspeed-mm  # 指向 workspace root，自动推导 MindSpeed-MM 子目录

# 或相对路径：
export MINDSPEED_MM_PATH=../mm-new
```

**兼容旧版**：如未设置 `MINDSPEED_MM_PATH`，Task6 会自动从 `PTA_PATH` / `MSA_PATH` 推导。

---

## 3. 数据集依赖

### 3.1 数据集根目录

| 路径 | 必需 | 说明 | 环境变量 |
|------|------|------|----------|
| `/data2/dataset` | ✅ | 默认数据集根目录 | `DATASET_ROOT` |

**设置方式**:
```bash
export DATASET_ROOT=/data2/dataset  # 可选，默认为 /data2/dataset
```

### 3.2 各模型数据路径

| 模型 | 默认路径 | 说明 |
|------|----------|------|
| InternVL3 | `${DATASET_ROOT}/internvl3/raw_ckpt/InternVL3-8B` | 模型检查点 |
| QwenVL2.5 | `${DATASET_ROOT}/qwen2.5vl/ckpt/Qwen2.5-VL-7B-Instruct` | 模型检查点 |
| OpenSora1.2 | `${DATASET_ROOT}/opensora1.2` | 模型和数据 |
| CogVideoX | `${DATASET_ROOT}/cogvideox/CogVideoX-5B` | 模型检查点 |

**自定义路径**:
```bash
export LOAD_PATH=/custom/path/to/model  # 覆盖默认模型路径
```

---

## 4. CANN 环境依赖

### 4.1 CANN Toolkit

| 路径 | 必需 | 说明 |
|------|------|------|
| `/usr/local/Ascend/ascend-toolkit/latest` | ✅ | CANN Toolkit 安装目录 |
| `/usr/local/Ascend/ascend-toolkit/set_env.sh` | ✅ | CANN 环境设置脚本 |

**环境变量**:
```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
```

### 4.2 NPU 驱动

| 路径 | 必需 | 说明 |
|------|------|------|
| `/usr/local/Ascend/driver/lib64` | ✅ | NPU 驱动库目录 |
| `/usr/local/Ascend/driver/lib64/common` | ✅ | NPU 驱动通用库 |
| `/usr/local/Ascend/driver/lib64/driver` | ✅ | NPU 驱动核心库 |

### 4.3 CANN 备用路径

| 路径 | 必需 | 说明 |
|------|------|------|
| `/usr/local/Ascend/cann/set_env.sh` | ⚠️ | CANN 环境设置脚本（备用） |

---

## 5. Conda 环境依赖

### 5.1 Conda 根目录

| 路径 | 必需 | 说明 |
|------|------|------|
| `/root/anaconda3` | ✅ | Conda 安装根目录 |
| `/root/anaconda3/etc/profile.d/conda.sh` | ✅ | Conda 初始化脚本 |

### 5.2 PTA Conda 环境

| 环境名 | 必需 | 说明 | 激活命令 |
|--------|------|------|----------|
| `mindspeed` | ✅ | PTA 环境（PyTorch Ascend） | `conda activate mindspeed` |

**PTA 环境关键包**:
```
torch >= 2.1.0
torch-npu (华为NPU适配)
transformers >= 4.39.0
accelerate >= 0.28.0
diffusers >= 0.27.0
safetensors
mindspeed-mm
```

### 5.3 MSA Conda 环境

| 环境名 | 必需 | 说明 | 激活命令 |
|--------|------|------|----------|
| `msa-m` | ✅ | MSA 环境（MindSpore Adapter） | `conda activate msa-m` |

**MSA 环境关键包（版本必须严格匹配）**:
```
python == 3.10
mindspore == 2.7.1
msadapter == 0.0.5
numpy == 1.26.0  (必须 <= 1.26.0)
ml_dtypes == 0.3.0
scipy == 1.11.4
transformers >= 4.39.0
accelerate >= 0.28.0
diffusers >= 0.27.0
```

**注意**: MSA 环境中 numpy 版本必须严格为 1.26.0，否则会出现兼容性问题。

### 5.4 FFmpeg 库（可选）

| 路径 | 必需 | 说明 |
|------|------|------|
| `/root/anaconda3/pkgs/ffmpeg-4.4.2-gpl_h773c8b4_113/lib` | ⚠️ | FFmpeg 库（decord 依赖） |

### 5.5 临时文件路径

| 路径 | 必需 | 说明 |
|------|------|------|
| `/tmp/decord_patch` | ⚠️ | decord 兼容性补丁目录（运行时自动创建） |
| `/tmp/task6` | ⚠️ | Task6 临时配置目录（运行时自动创建） |

---

## 6. 外部命令依赖

### 6.1 训练框架命令

| 命令 | 必需 | 说明 | 来源 |
|------|------|------|------|
| `torchrun` | ✅ | PyTorch 分布式启动 | PTA conda 环境 |
| `msrun` | ✅ | MindSpore 分布式启动 | MSA conda 环境 |

### 6.2 系统命令

| 命令 | 必需 | 说明 |
|------|------|------|
| `conda` | ✅ | Conda 包管理 |
| `fuser` | ✅ | 端口占用清理 |
| `pkill` | ✅ | 进程清理 |
| `source` | ✅ | 脚本执行 |
| `bash` | ✅ | Shell 执行 |

---

## 7. 环境设置脚本

Task6 使用以下内部脚本来设置外部环境：

| 脚本路径 | 说明 | 引用的外部路径 |
|----------|------|----------------|
| `scripts/envset/cann_set_env.sh` | 设置 CANN 环境变量 | `/usr/local/Ascend/ascend-toolkit/latest` |
| `scripts/envset/mm-pta-task6` | 设置 PTA 环境变量和 PYTHONPATH | `$(dirname ${MINDSPEED_MM_PATH})/Megatron-LM`, `$(dirname ${MINDSPEED_MM_PATH})/MindSpeed`, `${MINDSPEED_MM_PATH}`, `$(dirname ${MINDSPEED_MM_PATH})/MindSpeed-LLM` |
| `scripts/envset/mm-msa-task6` | 设置 MSA 环境变量和 PYTHONPATH | `$(dirname ${MINDSPEED_MM_PATH})/msadapter`, `$(dirname ${MINDSPEED_MM_PATH})/Megatron-LM`, `$(dirname ${MINDSPEED_MM_PATH})/MindSpeed`, `${MINDSPEED_MM_PATH}`, `$(dirname ${MINDSPEED_MM_PATH})/MindSpeed-LLM` |
| `scripts/envset/pta.sh` | PTA 额外环境变量 | `${ASCEND_TOOLKIT_HOME}/python/site-packages`, `${PTAPATH}/MindSpeed-LLM` |
| `scripts/envset/msa.sh` | MSA 额外环境变量 | `${ASCEND_TOOLKIT_HOME}/python/site-packages`, `${MSAPATH}/MindSpeed-LLM`, `${MSAPATH}/MSAdapter` |
| `scripts/envset/ptamm_set.sh` | PTA+MSA 混合环境 | `$(pwd)/MSAdapter`, `$(pwd)/Megatron-LM`, `$(pwd)/MindSpeed`, `$(pwd)/MindSpeed-MM` |

---

## 8. 环境准备检查清单

### 8.1 首次运行前检查

```bash
# 1. 检查 CANN 环境
ls /usr/local/Ascend/ascend-toolkit/latest/bin/
ls /usr/local/Ascend/driver/lib64/

# 2. 检查 Conda 环境
conda env list | grep mindspeed
conda env list | grep msa-m

# 3. 检查 MindSpeed-MM 主文件
ls ${MINDSPEED_MM_PATH}/pretrain_vlm.py
ls ${MINDSPEED_MM_PATH}/pretrain_vlm.py

# 4. 检查 PTA 示例脚本
ls ${MINDSPEED_MM_PATH}/examples/internvl3/finetune_internvl3_8B.sh
ls ${MINDSPEED_MM_PATH}/examples/qwen2.5vl/finetune_qwen2_5_vl_7b.sh
ls ${MINDSPEED_MM_PATH}/examples/opensora1.2/inference_opensora1_2.sh
ls ${MINDSPEED_MM_PATH}/examples/cogvideox/i2v_1.5/pretrain_cogvideox_i2v_1.5.sh

# 5. 检查 MSA 脚本 (scripts-ms)
ls ${MINDSPEED_MM_PATH}/scripts-ms/finetune_internvl3_8B.sh
ls ${MINDSPEED_MM_PATH}/scripts-ms/finetune_qwen2_5_vl_7b.sh
ls ${MINDSPEED_MM_PATH}/scripts-ms/inference_opensora1_2.sh
ls ${MINDSPEED_MM_PATH}/scripts-ms/pretrain_cogvideox_i2v_1.5.sh

# 6. 检查数据集
ls /data2/dataset/internvl3/
ls /data2/dataset/qwen2.5vl/
ls /data2/dataset/opensora1.2/
ls /data2/dataset/cogvideox/

# 7. 检查外部框架库
ls $(dirname ${MINDSPEED_MM_PATH})/Megatron-LM/
ls $(dirname ${MINDSPEED_MM_PATH})/MindSpeed/
ls $(dirname ${MINDSPEED_MM_PATH})/msadapter/
```

### 8.2 完整环境变量配置

```bash
# ========== 基础路径配置（使用相对路径或绝对路径） ==========
# 绝对路径示例（推荐）：
export LMSV_OUTPATH=./output
export MINDSPEED_MM_PATH=/shared/mindspeed-mm  # 指向 workspace root，自动推导 MindSpeed-MM 子目录

# 或相对路径示例：
# export LMSV_OUTPATH=./output
# export MINDSPEED_MM_PATH=../mm-new  # 指向 workspace root，自动推导 MindSpeed-MM 子目录

# ========== 可选环境变量（有默认值） ==========
# 基础配置
export DATASET_ROOT=/data2/dataset
export PTA_NAME=mindspeed
export MSA_NAME=msa-m

# Task6 配置
# 推荐使用 config.json 的 tasks.6 字段配置（与Task1-5一致）：
#   MODEL_NAME=internvl3
#   TOTAL_ITER=10
#   MUTNM=2
#   TRAIN_ITER=5
#   COMPARE_MODE=pta_msa

# 变异参数池配置（可选）
export MUTABLE_PARAMS_POOL_PATH=/custom/path/to/mutable_params_pool.yaml

# CANN 环境（通常自动设置）
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
```

---

## 9. 常见问题

### Q1: 可以修改 MindSpeed-MM 路径吗？

可以，通过设置 `MINDSPEED_MM_PATH` 环境变量：

```bash
# 推荐：指向 workspace root，自动推导 MindSpeed-MM 子目录
export MINDSPEED_MM_PATH=/shared/mindspeed-mm

# 兼容：直接指向 MindSpeed-MM 代码目录
export MINDSPEED_MM_PATH=/shared/mindspeed-mm/MindSpeed-MM
```

**兼容旧版**：仍可通过 `PTA_PATH` / `MSA_PATH` 分别指定（会自动拼接 `/MindSpeed-MM`）。

### Q2: 数据集可以放在其他位置吗？

可以，通过以下方式：

```bash
# 方式1: 修改数据集根目录
export DATASET_ROOT=/custom/data/path

# 方式2: 为特定模型指定路径
export LOAD_PATH=/custom/path/to/internvl3
```

### Q3: 没有 `/data2` 目录怎么办？

创建软链接：

```bash
sudo mkdir -p /data2
sudo ln -s /your/actual/data/path /data2/dataset
```

### Q4: CANN 路径不同怎么办？

修改 `scripts/envset/cann_set_env.sh` 中的路径，或设置：

```bash
export ASCEND_HOME_PATH=/your/cann/path
export ASCEND_TOOLKIT_HOME=/your/cann/path
```

### Q5: 如何检查 PTA/MSA 环境中的 Python 包是否安装正确？

```bash
# 检查 PTA 环境
conda activate mindspeed
python -c "import torch; import transformers; import accelerate; import diffusers; print('PTA 环境 OK')"

# 检查 MSA 环境
conda activate msa-m
python -c "import mindspore; import msadapter; import numpy; print(f'numpy version: {numpy.__version__}')"
```

### Q6: MindSpeed-LLM 是什么？必须安装吗？

MindSpeed-LLM 是部分旧版脚本引用的大模型训练框架。如果您的环境不需要运行旧版脚本，可以忽略。如需安装，确保目录存在：

```bash
ls $(dirname ${MINDSPEED_MM_PATH})/MindSpeed-LLM
ls $(dirname ${MINDSPEED_MM_PATH})/MindSpeed-LLM
```

### Q7: 如何验证所有外部依赖都已正确配置？

```bash
# 运行 Task6 前检查脚本
cd /path/to/lmsv_rec

# 1. 检查环境变量
echo "MINDSPEED_MM_PATH: ${MINDSPEED_MM_PATH}"
echo "MINDSPEED_MM_PATH: ${MINDSPEED_MM_PATH}"
echo "DATASET_ROOT: ${DATASET_ROOT}"

# 2. 检查关键文件
for path in \
  "${MINDSPEED_MM_PATH}/pretrain_vlm.py" \
  "${MINDSPEED_MM_PATH}/pretrain_vlm.py" \
  "/data2/dataset/internvl3" \
  "/usr/local/Ascend/ascend-toolkit/latest"
do
  if [ -e "$path" ]; then
    echo "✓ $path"
  else
    echo "✗ $path (缺失)"
  fi
done
```

---

## 10. 相关文档

- `docs/task6.md` - Task6 完整使用文档
- `docs/task6_skill.md` - 开发经验与避坑指南
- `docs/task6_model_handling.md` - 四模型处理逻辑
- `docs/task6_statistics.md` - 统计规则说明

---

**文档版本**: 1.1
**更新日期**: 2026-04-13
