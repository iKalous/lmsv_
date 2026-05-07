# Task6 多模态整网变异和验证 - 完整文档

> **作者**: 邹英龙
> **更新日期**: 2026-04-14

## 1. 项目概述

### 1.1 功能说明

Task6 是 LMSV 重构版中的多模态整网变异和验证任务，专门针对多模态大模型（视觉-语言模型、视频生成模型）进行自动化变异测试和跨框架对比验证。

### 1.2 核心流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    Task6 核心执行流程                            │
├─────────────────────────────────────────────────────────────────┤
│  1. 生成变异配置（基于前一轮结果，第一轮基于基础配置）              │
│  2. PTA环境执行训练/推理，记录loss、显存、执行时间                 │
│  3. 如果PTA异常，撤销本次变异并回滚                               │
│  4. MSA环境执行训练/推理，记录指标                                │
│  5. 对比PTA和MSA结果，检测差异                                    │
│  6. 重复至最大迭代次数                                           │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 对比模式

- **pta_msa**: PTA与MSA环境对比（默认）
- **pta_mf**: PTA与MF（Model Frontend，静态图）对比

---

## 2. 支持的模型

| 模型 | 名称 | 类型 | 基础配置 | PTA状态 | MSA状态 |
|------|------|------|---------|---------|---------|
| InternVL3 | internvl3 | 训练 | model_8B.json | ✅ 正常 | ✅ 正常 |
| QwenVL2.5 | qwenvl | 推理 | inference_qwen2_5_vl_7b.json | ✅ 正常 | ❌ InnerInplaceIndexPut shape mismatch |
| OpenSora1.2 | opensora | 推理 | inference_model_102x720x1280.json | ✅ 正常 | ❌ UntypedStorage 错误 |
| CogVideoX | cogvideox | 训练 | model_cogvideox.json | ✅ 正常 | ✅ 正常 |

**重要区分**：
- **训练模型**（InternVL3、CogVideoX）：有loss输出，需检查loss、显存、时间
- **推理模型**（QwenVL、OpenSora）：无loss输出，判断成功标准是返回码是否为0

---

## 3. 配置说明

Task6 的配置方式已与 Task1-5 完全统一，所有任务参数均通过 `config.json` 中的 `tasks["6"]` 字典传入，不再依赖 `TASK6_*` 环境变量。

### 3.1 全局配置字段

| 字段 | 含义 | 默认值 |
|------|------|--------|
| `task_type` | 固定为 `6` | — |
| `PTA_NAME` | PTA conda 环境名称 | `mindspeed` |
| `MSA_NAME` | MSA conda 环境名称（`pta_msa` 模式下） | `msadapter` |
| `MF_NAME` | MF conda 环境名称（`pta_mf` 模式下） | `mindf_py311` |
| `MINDSPEED_MM_PATH` | MindSpeed-MM 工作区根目录（支持相对/绝对路径，自动推导 MindSpeed-MM 子目录） | `<lm-sv-root>/../mindspeed-mm` |
| `SAVE_ABNORMAL_WEIGHTS` | 是否启用权重保存（每轮 PTA/MSA 均在 `iters/iter_N/weights/` 下保存模型权重） | `true` |

### 3.2 Task6 专属参数（`tasks["6"]`）

| 字段 | 含义 | 默认值 |
|------|------|--------|
| `MODEL_NAME` | 模型名称：`internvl3` / `qwenvl` / `opensora` / `cogvideox` | `internvl3` |
| `TOTAL_ITER` | 最大有效突变轮次 | `10` |
| `MUTNM` | 每轮变异参数个数 | `2` |
| `TRAIN_ITER` | 每轮训练/推理步数 | `5` |
| `COMPARE_MODE` | 对比模式：`pta_msa` 或 `pta_mf` | `pta_msa` |
| `BASE_SEED` | 基础随机种子，控制变异确定性 | `43` |
| `PTA_MAX_RUNTIME` | PTA 单次最长执行时间（秒） | `3000` |
| `MSA_MAX_RUNTIME` | MSA 单次最长执行时间（秒） | `3000` |

### 3.3 路径与环境变量

数据集和权重路径默认按模型自动推断，也可通过环境变量覆盖：

| 环境变量 | 说明 |
|----------|------|
| `DATASET_ROOT` | 数据集根目录，必须在 config.json 中配置 |
| `LOAD_PATH` | 见下方 | 模型检查点路径 |

**默认 LOAD_PATH 按模型自动设置**：
- InternVL3: `${DATASET_ROOT}/internvl3/raw_ckpt/InternVL3-8B`
- QwenVL: `${DATASET_ROOT}/qwen2.5vl/ckpt/Qwen2.5-VL-7B-Instruct`
- OpenSora: `${DATASET_ROOT}/opensora1.2`
- CogVideoX: `${DATASET_ROOT}/cogvideox/CogVideoX-5B`

**兼容说明**：
- `MINDSPEED_MM_PATH` 未设置时，会回退读取 `PTA_PATH` / `PTAPATH`（兼容旧配置）
- `TRAIN_ITER` 未设置时，会回退读取 `SAVE_STEPS` / `TRAIN_ITERS`（兼容旧配置）

### 3.4 环境安装（一键安装）

Task6 环境搭建采用"裸环境 + 一键补丁"流程：

```bash
# 1. 还原裸环境
cd /path/to/lm-sv/task6_conda_envs_export/standard_env
conda env create -f mindspeed_bare.yml -n mindspeed
conda env create -f msadapter_bare.yml -n msadapter

# 2. 运行一键定制化脚本
cd ../automated_setup
bash setup_task6_envs.sh
```

安装完成后可直接使用 `conda activate mindspeed` / `conda activate msadapter`。

详细文档参见：`docs/environment-modifications.md` 和 `docs/TASK6_CUSTOMIZATION_MODIFICATIONS.md`

---

### 3.5 从零开始完整部署（新机器/环境重建）

如果你是首次部署，或需要**完全重建环境**（删除旧环境后重新安装），按以下步骤执行。

#### 前置条件检查

```bash
# 1. 确认 CANN 已安装
ls /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2. 确认数据集存在（根据你的 DATASET_ROOT 配置）
#    例如：ls <DATASET_ROOT>
ls <YOUR_DATASET_ROOT>/dataset

# 3. 确认 MindSpeed-MM 代码存在（根据你的 MINDSPEED_MM_PATH 配置）
#    例如：ls /mindspeed-mm  （指向 workspace root，自动推导 MindSpeed-MM 子目录）
ls <YOUR_MINDSPEED_MM_PATH>

# 4. 确认 conda 可用
conda --version

# 5. 确认 NPU 可用
npu-smi info
```

#### 删除旧环境（如需重建）

```bash
# 删除已有的 Task6 相关 conda 环境
conda remove --name mindspeed --all -y 2>/dev/null || true
conda remove --name msadapter --all -y 2>/dev/null || true
conda remove --name msa-m --all -y 2>/dev/null || true
conda remove --name ptaa --all -y 2>/dev/null || true
conda remove --name mindf_py311 --all -y 2>/dev/null || true
```

#### 安装环境并配置

```bash
# 1. 安装 Task6 conda 环境
# 先还原裸环境，再运行定制化脚本
cd <YOUR_LMSV_ROOT>/task6_conda_envs_export/standard_env
conda env create -f mindspeed_bare.yml -n mindspeed
conda env create -f msadapter_bare.yml -n msadapter
cd ../automated_setup
bash setup_task6_envs.sh

# 2. 配置 config.json（所有路径通过此文件配置，代码中无硬编码）
# 请根据实际情况修改以下路径：
#   - MINDSPEED_MM_PATH: MindSpeed-MM 工作区根目录（自动推导 MindSpeed-MM 子目录）
#   - DATASET_ROOT: 数据集根目录
cat > <YOUR_LMSV_ROOT>/lmsv_rec/config.json << 'EOF'
{
  "task_type": 6,
  "PTA_NAME": "mindspeed",
  "MSA_NAME": "msadapter",
  "MINDSPEED_MM_PATH": "<YOUR_MINDSPEED_MM_PATH>",
  "DATASET_ROOT": "<YOUR_DATASET_ROOT>",
  "SAVE_ABNORMAL_WEIGHTS": true,
  "tasks": {
    "6": {
      "MODEL_NAME": "internvl3",
      "TOTAL_ITER": 1,
      "MUTNM": 2,
      "COMPARE_MODE": "pta_msa",
      "TRAIN_ITER": 2,
      "BASE_SEED": 43,
      "PTA_MAX_RUNTIME": 3000,
      "MSA_MAX_RUNTIME": 3000
    }
  }
}
EOF
```

#### 执行 Task6

```bash
# 加载 CANN 环境并执行
cd <YOUR_LMSV_ROOT>/lmsv_rec
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python do.py
```

#### 紧急停止

如果执行过程中需要强制停止，使用 `Ctrl+C` 中断当前终端命令。若存在分布式残留进程，可通过 `ps aux | grep msrun` / `ps aux | grep torchrun` 查找后手动 `kill`。

---

## 4. 执行方法

### 4.1 统一入口（与 Task1-5 完全一致）

```bash
cd /path/to/lmsv_rec

# 方式1：交互式生成配置后执行
python do.py conf   # 选择 Task6，按提示配置
python do.py

# 方式2：直接修改配置文件执行
# 按需修改 config.json 中的模型、迭代次数等参数
python do.py
```

四个模型的执行命令完全相同，只需在 `python do.py conf` 交互中或通过修改 `config.json` 的 `MODEL_NAME` 切换模型：

```bash
# InternVL3、QwenVL、OpenSora、CogVideoX 均使用同一入口
python do.py conf
python do.py
```

### 4.2 配置文件模板（`config.json`）

Task6 所有参数直接通过 `config.json` 配置，示例模板如下：

```json
{
  "task_type": 6,
  "PTA_NAME": "mindspeed",
  "MSA_NAME": "msadapter",
  "MINDSPEED_MM_PATH": "<YOUR_MINDSPEED_MM_PATH>",
  "DATASET_ROOT": "<YOUR_DATASET_ROOT>",
  "SAVE_ABNORMAL_WEIGHTS": true,
  "tasks": {
    "6": {
      "MODEL_NAME": "internvl3",
      "TOTAL_ITER": 10,
      "MUTNM": 2,
      "COMPARE_MODE": "pta_msa",
      "TRAIN_ITER": 5,
      "BASE_SEED": 43,
      "PTA_MAX_RUNTIME": 3000,
      "MSA_MAX_RUNTIME": 3000
    }
  }
}
```

### 4.3 四模型配置说明

Task6 的四个模型共用同一套 `config.json` 和 `mutable_params_pool.yaml` 文件，只需修改 `MODEL_NAME` 即可切换模型。

#### `config.json` 模型切换方式

| 模型 | `MODEL_NAME` | 建议 `TRAIN_ITER` | 说明 |
|------|-------------|-------------------|------|
| InternVL3 | `internvl3` | `2~5` | 训练模型，有 loss 输出 |
| QwenVL2.5 | `qwenvl` | `1~2` | 推理模型，无 loss，以返回码判断 |
| OpenSora1.2 | `opensora` | `1~2` | 推理模型，MSA 环境 safetensors 加载失败（框架问题） |
| CogVideoX | `cogvideox` | `2~5` | 训练模型，MSA 环境已可正常运行 |

**四模型最小可运行配置示例**：

```json
{
  "task_type": 6,
  "PTA_NAME": "mindspeed",
  "MSA_NAME": "msadapter",
  "MINDSPEED_MM_PATH": "<YOUR_MINDSPEED_MM_PATH>",
  "DATASET_ROOT": "<YOUR_DATASET_ROOT>",
  "SAVE_ABNORMAL_WEIGHTS": true,
  "tasks": {
    "6": {
      "MODEL_NAME": "internvl3",
      "TOTAL_ITER": 1,
      "MUTNM": 2,
      "COMPARE_MODE": "pta_msa",
      "TRAIN_ITER": 2,
      "BASE_SEED": 43,
      "PTA_MAX_RUNTIME": 3000,
      "MSA_MAX_RUNTIME": 3000
    }
  }
}
```

切换模型时，只需将 `"MODEL_NAME"` 的值替换为 `qwenvl`、`opensora` 或 `cogvideox`，并根据模型类型调整 `SAVE_STEPS`。

#### `mutable_params_pool.yaml` 配置说明

`mutable_params_pool.yaml` 位于项目根目录，定义了 Task6 的变异参数池，**四个模型共用同一文件**：

```yaml
# 数值型参数（范围变异）
numeric_params:
  mlp_ratio:
    min_val: 2.0
    max_val: 8.0
    min_factor: 0.7
    max_factor: 1.5

# 枚举型参数（离散值变异）
enum_params:
  activation_func:
    - gelu
    - silu
```

- 程序启动时会自动在当前目录（项目根目录）查找 `mutable_params_pool.yaml`
- 如需自定义路径，可通过环境变量覆盖：
  ```bash
  export MUTABLE_PARAMS_POOL_PATH=/custom/path/to/mutable_params_pool.yaml
  ```
- 修改 YAML 后无需重启或重新编译，直接再次执行 `python do.py` 即可生效

---

## 5. 项目结构

### 5.1 代码文件结构

```
lmsv_rec/
├── utils/task/task6.py                    # Task6 主流程实现
├── utils/analyze/task6_result.py          # 结果分析模块
├── utils/runtime/mm_mutation/
│   ├── mm_mutator.py                      # 多模态配置变异器
│   └── mutate_graph.py                    # 变异执行入口
├── scripts/runtime/
│   ├── mm_pta_internvl3.sh                # InternVL3 PTA入口
│   ├── mm_pta_qwenvl.sh                   # QwenVL PTA入口
│   ├── mm_pta_opensora.sh                 # OpenSora PTA入口
│   ├── mm_pta_cogvideox.sh                # CogVideoX PTA入口
│   ├── mm_msa_internvl3.sh                # InternVL3 MSA入口
│   ├── mm_msa_qwenvl.sh                   # QwenVL MSA入口
│   ├── mm_msa_opensora.sh                 # OpenSora MSA入口
│   ├── mm_msa_cogvideox.sh                # CogVideoX MSA入口
│   ├── pta_internvl3_8B_real.sh           # InternVL3 PTA真实执行
│   ├── pta_qwenvl_7b_real.sh              # QwenVL PTA真实执行
│   ├── pta_opensora_real.sh               # OpenSora PTA真实执行
│   ├── pta_cogvideox_real.sh              # CogVideoX PTA真实执行
│   ├── msa_internvl3_8B_real.sh           # InternVL3 MSA真实执行
│   ├── msa_qwenvl_7b_real.sh              # QwenVL MSA真实执行
│   ├── msa_opensora_real.sh               # OpenSora MSA真实执行
│   ├── msa_cogvideox_real.sh              # CogVideoX MSA真实执行
│   └── prepare_mm_config.sh               # 配置预处理脚本
├── assets/mm_configs/                     # 模型配置文件（自包含）
│   ├── model_8B.json                      # InternVL3配置
│   ├── data_8B.json                       # InternVL3数据配置
│   ├── inference_qwen2_5_vl_7b.json       # QwenVL配置
│   ├── inference_model_102x720x1280.json  # OpenSora配置
│   ├── model_cogvideox.json               # CogVideoX配置
│   └── data_cogvideox.json                # CogVideoX数据配置
├── mutable_params_pool.yaml               # 变异参数池（YAML配置）
├── docs/
│   ├── task6.md                           # 本文档（交付文档）
│   ├── task6_skill.md                     # 开发经验与避坑指南
│   ├── task6_statistics.md                # 统计规则说明
│   └── task6_model_handling.md            # 四模型处理逻辑
└── detected_bugs/                         # 检测到的Bug记录
    ├── opensora_msa/
    ├── cogvideox_msa_success/
    ├── qwenvl_msa_inner_inplace_indexput/
    └── internvl3_pta_msa_precision_diff/
```

### 5.2 输出目录结构

```
output/                                    # LMSV_OUTPATH 目录
├── analysis/                              # 分析报告目录（与Task1-5一致）
│   ├── summary.md                         # Markdown报告
│   ├── report.html                        # HTML报告
│   └── assets/                            # 图表资源目录
├── data/                                  # 数据目录（与Task1-5一致）
│   ├── summary.json                       # JSON汇总
│   └── iteration_metrics.csv              # 迭代指标CSV
├── final_report.json                      # 最终结果汇总
└── iters/                                 # 迭代归档目录（与Task1-5一致）
    ├── iter_1/
    │   ├── status.json                    # 状态记录
    │   ├── FAILED_FLAG                    # 失败标记（如失败）
    │   ├── failure_info.txt               # 失败详情（如失败）
    │   ├── metrics.json                   # 指标数据
    │   ├── runtime_logs/                  # 运行时日志
    │   │   ├── pta_verify_iter1.log
    │   │   └── msa_verify_iter1.log
    │   ├── msrun_log/                     # MSA worker日志
    │   ├── weights/                       # 每轮权重导出目录
    │   │   ├── pta/                       # PTA 环境保存的权重
    │   │   │   └── iter_0000002/
    │   │   │       └── mp_rank_00_000/
    │   │   │           └── model_optim_rng.pt
    │   │   └── msa/                       # MSA 环境保存的权重
    │   │       └── iter_0000002/
    │   │           └── mp_rank_00_000/
    │   │               └── model_optim_rng.pt
    │   ├── artifacts/                     # 产物目录
    │   │   └── mutation_inputs/
    │   │       └── mutation_config.json   # 变异配置
    │   ├── scripts/                       # 脚本备份（预留）
    │   └── res/                           # 结果产物（预留）
    ├── iter_2/
    └── ...

tmp/task6/                                 # 临时目录
├── mutation_results/                      # 变异配置文件
│   ├── internvl3/mutation_gen*.json
│   ├── qwenvl/mutation_gen*.json
│   ├── opensora/mutation_gen*.json
│   └── cogvideox/mutation_gen*.json
└── *_verify_iter*.log                     # 运行时日志

pta_logs/                                  # PTA日志目录
└── train_*.log

msrun_log/                                 # MSA日志目录
└── train_*.log
```

---

## 6. 外部依赖文件

### 6.1 数据集路径

**重要说明**：数据集根目录通过 `config.json` 中的 `DATASET_ROOT` 字段配置，代码中没有任何硬编码路径。

```bash
# 示例：假设 DATASET_ROOT 配置为 <DATASET_ROOT>
ls <DATASET_ROOT>
```

如需自定义数据集路径，修改 `config.json` 中的 `DATASET_ROOT` 字段即可：
```json
{
  "DATASET_ROOT": "/your/custom/data/path"
}
```

**各模型数据集子路径**（相对于 `DATASET_ROOT`）：

| 模型 | 相对路径 | 说明 |
|------|---------|------|
| InternVL3 | `internvl3` | 包含图像-文本对数据 |
| QwenVL2.5 | `qwen2.5vl` | 包含图像-文本对数据 |
| OpenSora1.2 | `opensora1.2` | 包含视频生成prompts |
| CogVideoX | `cogvideox` | 包含视频-文本对数据 |

### 6.2 MindSpeed-MM 路径（外部依赖）

**注意**：`MINDSPEED_MM_PATH` 支持两种写法，框架自动兼容：
- **推荐**：指向 workspace root（如 `<lm-sv-root>/../mindspeed-mm`），自动推导 `MindSpeed-MM` 子目录
- **兼容**：直接指向 `MindSpeed-MM` 代码目录（如 `<lm-sv-root>/../mindspeed-mm/MindSpeed-MM`）

**支持分离布局**：`MindSpeed` 框架和 `msadapter` 源码可以与 `Megatron-LM` / `MindSpeed-MM` 分开存放。环境脚本会自动从 `$(dirname ${MINDSPEED_MM_PATH})/lm-sv/mm-new` 推断 `MindSpeed` 和 `msadapter` 的位置。

| 路径 | 说明 |
|------|------|
| `${MINDSPEED_MM_PATH}` | MindSpeed-MM 工作区根目录（自动推导代码子目录） |
| `${MINDSPEED_MM_PATH}/MindSpeed-MM` 或 `${MINDSPEED_MM_PATH}` | MindSpeed-MM 代码目录（根据实际指向自动适配） |
| `${MINDSPEED_MM_PATH}/examples` 或 `${MINDSPEED_MM_PATH}/MindSpeed-MM/examples` | PTA 示例脚本 |
| `${MINDSPEED_MM_PATH}/scripts-ms` 或 `${MINDSPEED_MM_PATH}/MindSpeed-MM/scripts-ms` | MSA 示例脚本 |
| `${MINDSPEED_MM_PATH}/pretrain_vlm.py` 或 `${MINDSPEED_MM_PATH}/MindSpeed-MM/pretrain_vlm.py` | VLM 训练入口 |
| `${MINDSPEED_MM_PATH}/inference_vlm.py` 或 `${MINDSPEED_MM_PATH}/MindSpeed-MM/inference_vlm.py` | VLM 推理入口 |
| `${MINDSPEED_MM_PATH}/inference_sora.py` 或 `${MINDSPEED_MM_PATH}/MindSpeed-MM/inference_sora.py` | OpenSora 推理入口 |
| `${MINDSPEED_MM_PATH}/pretrain_sora.py` 或 `${MINDSPEED_MM_PATH}/MindSpeed-MM/pretrain_sora.py` | CogVideoX 训练入口 |

### 6.3 CANN 环境（外部依赖）

| 路径 | 说明 |
|------|------|
| `/usr/local/Ascend/ascend-toolkit/set_env.sh` | CANN 环境设置脚本 |
| `/usr/local/Ascend/driver` | NPU 驱动 |
| `/usr/local/Ascend/ascend-toolkit/latest` | CANN Toolkit |

### 6.4 Conda 环境（外部依赖）

| 环境名称 | 用途 |
|---------|------|
| `mindspeed` | PTA 环境（PyTorch Ascend） |
| `msadapter` | MSA 环境（MindSpore Adapter） |

---

## 7. 检测到的 Bug

### 7.1 Bug 汇总

| Bug/案例 | 模型 | 描述 | 状态 | 记录位置 |
|----------|------|------|------|----------|
| OpenSora MSA UntypedStorage | OpenSora | safetensors 加载时 `TypeError: 'UntypedStorage' object is not callable` | 记录 | `detected_bugs/opensora_msa/` |
| CogVideoX MSA 成功执行 | CogVideoX | 经环境层修复后，PTA/MSA 双端均可正常运行 | 成功案例 | `detected_bugs/cogvideox_msa_success/` |
| QwenVL MSA InnerInplaceIndexPut | QwenVL | `input_embeds[indices] = vit_embeds` 时 shape mismatch | 记录 | `detected_bugs/qwenvl_msa_inner_inplace_indexput/` |
| InternVL3 PTA/MSA精度差异 | InternVL3 | 成功测试案例，检测到 ~20% 的 loss 精度差异 | 成功案例 | `detected_bugs/internvl3_pta_msa_precision_diff/` |

### 7.2 Bug 详情

#### OpenSora MSA - UntypedStorage 错误

**问题描述**: 在 MSA 环境中运行 OpenSora 时，模型权重加载阶段出现 safetensors 加载失败。

**错误信息**:
```
TypeError: 'UntypedStorage' object is not callable
```

**根因分析**: msadapter 环境中的 safetensors 库与 PyTorch 兼容性存在问题，`safe_open` 函数返回了 `UntypedStorage` 对象而非预期的文件句柄。

**详细分析**: 见 `detected_bugs/opensora_msa/analysis.md`

#### CogVideoX MSA - 当前状态：成功执行

**状态**: 经 msadapter / transformers 环境层修复后，CogVideoX 在 MSA 环境下已可正常执行训练。

**关键修复**:
1. `msadapter` bfloat16 fallback (`_utils.py`, `serialization.py`)
2. `LD_LIBRARY_PATH` 优先加载 conda 的 `libstdc++.so.6`
3. `transformers` 调用签名关键字参数修复
4. `mindspore.Tensor.__getitem__` 对 `numpy.ndarray` 索引的自动转换
5. `msa_cogvideox_real.sh` 超时等待时间延长至 360 次检查

**验证结果**:
- PTA: 成功（loss≈1.03）
- MSA: 成功（loss≈1.10，差异约 6.49%）

**详细分析**: 见 `detected_bugs/cogvideox_msa_success/analysis.md`

#### QwenVL MSA - InnerInplaceIndexPut shape mismatch

**问题描述**: 在 MSA 环境下运行 QwenVL2.5 推理时，`vlm_model.py:548` 的 image token 替换操作因 shape mismatch 崩溃。

**错误信息**:
```
ValueError: For 'InnerInplaceIndexPut', shape mismatch: value tensor of shape [14308, 1280] cannot be broadcast to indexing result of shape [3577, 3584].
```

**根因分析**: MindSpore `InnerInplaceIndexPut` 的 shape 广播规则比 PyTorch `index_put_` 更严格；MSA 环境下 TP=4 导致 `vit_embeds` 的 batch 维度被放大 4 倍，与索引结果 shape 不一致。

**详细分析**: 见 `detected_bugs/qwenvl_msa_inner_inplace_indexput/analysis.md`

---

## 8. 核心机制

### 8.1 增量变异机制

```python
# 第1轮：基于基础配置
gen1 = mutate(base_config)

# 第2轮：基于gen1
gen2 = mutate(gen1)

# 第3轮：基于gen2
gen3 = mutate(gen2)
```

### 8.2 异常回滚机制

- **PTA失败**: 视为无效突变，撤销本轮变异，删除 `mutation_genX.json`，基于上一轮配置重新变异
- **MSA失败**: 视为有效突变（框架问题），记录问题，继续下一轮

### 8.3 结果对比算法

**Loss对比**:
```python
# 相对误差 < 1% 或绝对误差 < 0.01 视为匹配
if pta_loss > 0:
    relative_diff = abs(pta_loss - msa_loss) / pta_loss
    if relative_diff > 0.01 and abs_diff > 0.01:
        loss_match = False
```

**显存对比**:
```python
memory_diff = abs(pta_memory - msa_memory)
```

### 8.4 权重导出机制

每轮 PTA / MSA 执行完成后，模型权重会自动保存到输出目录的 `weights/` 子目录下：

```
iters/iter_N/
└── weights/
    ├── pta/
    │   └── iter_0000002/               # 对应训练步数
    │       └── mp_rank_00_000/
    │           └── model_optim_rng.pt  # 实际权重文件
    └── msa/
        └── iter_0000002/
            └── mp_rank_00_000/
                └── model_optim_rng.pt
```

**实现要点**：

1. **SAVE_PATH 环境变量**：`task6.py` 在每轮执行前通过 `_get_iter_weights_dir()` 生成权重目录，并通过 `SAVE_PATH` 环境变量传入脚本
2. **save-interval 设置**：所有模型脚本中的 `--save-interval` 设为 `${TRAIN_ITERS:-1}`，确保在短训练运行中也能保存权重
3. **空间占用**：训练模型权重约 11~15GB/侧/轮，10轮约 220~300GB，需确保输出目录所在磁盘有足够空间

**如需关闭权重保存**（节省磁盘空间）：

```json
{
  "SAVE_ABNORMAL_WEIGHTS": false
}
```

注意：关闭后 `weights/` 目录仍会创建，但内部为空。

---

### 8.5 参数突变合法性校验与成功率优化

为提高突变成功率并避免生成框架不兼容的配置，Task6 在参数突变流程中引入了多层合法性校验机制：

#### 1) TP（Tensor Parallel）兼容性约束 —— CogVideoX 专用

**问题背景**：CogVideoX 在 8 卡 TP 环境下运行，要求输入序列长度（sequence length）必须能被 TP size（8）整除。当 `concat_text_embed` 从 `true` 突变为 `false` 时，text token 不再拼接到 video patch token 序列中，仅剩下 patch token 数量（由 `input_size` 和 `patch_size` 决定）。若 patch token 总数不能被 8 整除，PTA 执行时会触发 `AssertionError: First dimension of the tensor should be divisible by tensor parallel size`。

**修复策略**（`utils/runtime/mm_mutation/mm_mutator.py`）：
- 在每次突变完成后执行 **post-validation**
- 若检测到 `concat_text_embed=false` 导致 patch tokens 不能被 TP size 整除，则 **自动回退** `concat_text_embed` 为原始值
- 为保持突变数量不变，从剩余可用参数中 **自动补选** 一个合法参数进行补偿突变
- 若所有突变均被回退导致配置与上一轮完全相同，则触发 **强制突变** 机制，随机修改一个参数以确保多样性

**关键代码逻辑**：
```python
def _fix_invalid_mutations(self, working_config, mutations_applied, ...):
    # 计算 patch tokens = input_size各维度 / patch_size各维度的乘积
    patch_tokens = self._compute_cogvideox_patch_tokens(working_config)
    if patch_tokens % tp_size != 0:
        # 回退 concat_text_embed 并补偿突变另一个参数
        ...
```

#### 2) hidden_size 可被 head 数整除

在整数参数突变后，若涉及 `hidden_size`，自动调整其值以确保能被 `num_heads` 与 `num_attention_heads` 的最小公倍数整除，避免注意力计算时维度不匹配。

#### 3) 参数池过滤 —— 只突变配置中存在的参数

早期版本会从全部 60+ 个参数中随机采样，导致大量采样到的参数在当前模型配置中不存在。现已修复为 **仅筛选 `working_config` 中真实存在的参数** 进入采样池，大幅提升有效突变率。

#### 4) 浮点数参数类型修复

`ucg_rate` 为浮点型参数，早期版本未将其加入 `float_params` 列表，导致突变时按整数处理产生 `TypeError`。现已将其纳入浮点参数列表，并在嵌套配置突变逻辑中同步修复。

#### 5) 突变多样性保障

- 每次重试使用不同的随机种子：`seed = BASE_SEED + attempt_count`，确保retry时探索不同参数组合
- `mutate_graph.py` 在保存配置前检查是否至少有一个参数发生变化，若完全相同则强制修改一个参数
- `mm_mutator.py` 的 `_fix_invalid_mutations` 最终兜底：若所有突变均被回退，强制突变一个合法参数

#### 6) 校验效果

引入上述约束前，CogVideoX 的 PTA 突变失败率约为 **60%**（主要因 `concat_text_embed=false` 导致 TP 不兼容）。引入约束后，该失败模式被完全消除，PTA 突变成功率提升至接近 **100%**（排除环境/资源等外部因素）。

---

## 9. 常见问题

| 问题 | 可能原因 | 解决方案 |
|------|----------|---------|
| PTA执行失败 | 环境变量未设置 | 检查 `LMSV_OUTPATH`、`MINDSPEED_MM_PATH`（兼容旧版 `PTA_PATH`/`MSA_PATH`） |
| MSA执行失败 | msrun未找到 | 检查 conda 环境激活 |
| 变异失败 | 配置文件格式错误 | 检查基础 JSON 文件 |
| Loss不匹配 | 精度问题或实现差异 | 查看详细日志分析 |
| 数据集路径错误 | 数据集根目录不存在 | 检查 `DATASET_ROOT` 环境变量或创建数据集目录 |

---

## 10. 参考文档

- `docs/task6_skill.md` - 开发经验、踩坑记录、核心原则
- `docs/task6_statistics.md` - 统计规则说明
- `docs/task6_model_handling.md` - 四模型处理逻辑详解
- `docs/PTA_MF_PRECISION_ALIGNMENT.md` - PTA/MF精度对齐

---

**文档版本**: 2.1
**更新日期**: 2026-04-13
**适用范围**: LMSV Task6 多模态整网变异和验证
