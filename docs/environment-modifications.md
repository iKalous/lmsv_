# Task6 从零开始环境搭建指南

> 本文档面向需要在全新机器上部署并运行 Task6（多模态整网泛化变异测试）的用户。所有步骤均提供可直接复制执行的命令。
>
> 本仓库已包含：
> - `lmsv_rec/`：主项目代码
> - `task6_conda_envs_export/`：Task6 环境搭建工具（裸环境定义 + 一键定制化脚本）

---

## 0. 前置条件

| 项目 | 要求 |
|------|------|
| 操作系统 | Linux (aarch64 或 x86_64) |
| NPU | Ascend 910B（8 卡） |
| CANN | 已安装并配置 `Ascend/ascend-toolkit/set_env.sh` |
| Conda | Anaconda 或 Miniconda 已安装 |
| Git | 已安装 |
| 磁盘空间 | 至少 20 GB 空闲空间（环境包 + 数据集） |

**验证 CANN：**
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info  # 应该能看到 NPU 信息
```

**验证 Conda：**
```bash
conda --version
```

---

## 1. 拉取代码

```bash
# 替换为实际的 git 仓库地址
GIT_URL="<你的仓库地址>"
CLONE_DIR="${HOME}/lm-sv"

git clone "${GIT_URL}" "${CLONE_DIR}"
cd "${CLONE_DIR}"
```

拉取后目录结构如下：
```
lm-sv/
├── lmsv_rec/                  # 主项目代码
├── task6_conda_envs_export/   # Task6 环境搭建工具
│   ├── automated_setup/       # 一键定制化脚本和补丁
│   └── standard_env/          # 裸 conda 环境定义
└── ...
```

---

## 2. 安装 Conda 环境（裸环境 + 定制化修改）

Task6 需要两个 conda 环境：
- `mindspeed`：PTA 侧运行环境
- `msadapter`：MSA 侧运行环境

环境搭建采用"**裸环境 → 补充安装 → 源码补丁**"的两阶段流程：

### 2.1 阶段一：还原裸环境

裸环境定义位于 `task6_conda_envs_export/standard_env/`，仅包含 `requirements.txt` 中的标准库，**不含任何定制化修改**。

```bash
cd "${CLONE_DIR}/task6_conda_envs_export/standard_env"

# 方式一：通过 yml 文件创建（推荐）
conda env create -f mindspeed_bare.yml -n mindspeed
conda env create -f msadapter_bare.yml -n msadapter

# 方式二：通过 requirements.txt 安装
conda create -n mindspeed python=3.10 -y
conda activate mindspeed
pip install -r ../automated_setup/requirements.txt

conda create -n msadapter python=3.10 -y
conda activate msadapter
pip install -r ../automated_setup/requirements.txt
```

### 2.2 阶段二：运行 setup_task6_envs.sh（一键定制化）

在裸环境准备就绪后，执行自动化脚本完成所有定制化修改：

```bash
cd "${CLONE_DIR}/task6_conda_envs_export/automated_setup"
bash setup_task6_envs.sh
```

脚本会自动完成：
1. **检查前置条件**：确认两个 conda 环境存在且关键包已安装
2. **推断工作区路径**：自动查找 MindSpeed-MM 工作区
3. **安装缺失包**：
   - mindspeed 环境：补充安装 ~23 个 PyPI 包 + 复制 apex（本地编译的昇腾版本）
   - msadapter 环境：补充安装 ~18 个 PyPI 包 + 从华为云安装 torch_npu 2.7.1.post2
4. **应用 transformers 兼容性补丁**（两个环境均需）
5. **应用 msadapter bfloat16 fallback 补丁**
6. **部署环境脚本**：将 `mm-pta-task6.sh` 和 `mm-msa-task6.sh` 复制到 `lmsv_rec/scripts/envset/`
7. **验证所有修改**

安装过程约 5-10 分钟。

**环境路径总结：**

| 环境名 | 默认安装路径 | 激活命令 |
|--------|-------------|----------|
| mindspeed | conda 默认 envs 目录 | `conda activate mindspeed` |
| msadapter | conda 默认 envs 目录 | `conda activate msadapter` |

**注意：** 脚本内部所有代码都已去除硬编码路径，改为通过 `conda info --base` 动态获取 conda 安装根目录。只要你的系统 `conda` 命令可用，Task6 就能自动找到环境。

### 2.3 验证环境安装

```bash
# 验证 mindspeed 环境
conda activate mindspeed
python -c "import torch; import mindspeed; print('PTA env OK')"

# 验证 msadapter 环境
conda activate msadapter
python -c "import mindspore; import ml_dtypes; print('MSA env OK')"
```

---

## 3. 配置项目

### 3.1 修改 `config.json`

复制示例配置并修改关键路径：

```bash
cd "${CLONE_DIR}/lmsv_rec"
cp config.json.example config.json
```

编辑 `config.json`。如果 MindSpeed-MM 工作区在标准位置 `/shared/mindspeed-mm/`，建议直接指向 workspace root：

```json
{
  "task_type": 6,
  "PTA_NAME": "mindspeed",
  "MSA_NAME": "msadapter",
  "MINDSPEED_MM_PATH": "/shared/mindspeed-mm",
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

**说明：**
- `MINDSPEED_MM_PATH` 支持指向 **workspace root**（如 `/shared/mindspeed-mm`），框架自动推导 `MindSpeed-MM` 子目录
- 旧写法（直接指向 `MindSpeed-MM` 代码目录，如 `/shared/mindspeed-mm/MindSpeed-MM`）仍然兼容
- 推荐指向 workspace root，路径更简洁且不易出错

### 3.2 环境脚本说明（已自动适配）

`scripts/envset/` 下的 `mm-pta-task6.sh` 和 `mm-msa-task6.sh` 已经去掉了所有硬编码路径，改为：
- 通过 `conda info --base` 动态获取 conda 安装根目录，再执行 `conda activate`
- 通过 `CONDA_PREFIX` 自动定位当前激活的 conda 环境库路径
- 通过 `MINDSPEED_MM_PATH` 环境变量定位工作区（支持 workspace root 和代码目录两种模式，自动检测）
- **支持分离布局**：`Megatron-LM` / `MindSpeed-MM` 与 `MindSpeed` / `msadapter` 可以不在同一个目录下

**目录布局说明：**

环境脚本支持两种目录布局方式：

**方式一：统一布局（所有组件在同一 workspace root 下）**
```
/workspace/
├── Megatron-LM/
├── MindSpeed-MM/
├── MindSpeed/      # PTA 脚本需要
└── msadapter/      # MSA 脚本需要
```

**方式二：分离布局（推荐，MindSpeed 和 msadapter 在 lm-sv 下）**
```
/shared/mindspeed-mm/          # workspace root（Megatron-LM + MindSpeed-MM）
├── Megatron-LM/
└── MindSpeed-MM/
<lm-sv-root>/mm-new/          # MindSpeed 框架和 msadapter 源码
├── MindSpeed/
└── msadapter/
```

在分离布局下，环境脚本会自动推断：
- `WORKSPACE_ROOT` = `MINDSPEED_MM_PATH`（或从代码目录推导的父目录）
- `MindSpeed` / `msadapter` 路径 = `$(dirname ${WORKSPACE_ROOT})/lm-sv/mm-new`

**你不需要手动修改任何脚本**，只要：
1. `setup_task6_envs.sh` 执行成功（所有补丁已应用）
2. `config.json` 中的 `MINDSPEED_MM_PATH` 指向正确的 workspace root 或 MindSpeed-MM 代码目录

---

## 4. 准备数据集与模型权重

Task6 默认从 `/data2/dataset` 读取数据。在新机器上，你需要：
1. 准备数据集目录
2. 准备模型权重（checkpoint）

### 4.1 设置数据路径

推荐通过**环境变量**覆盖，避免修改代码：

```bash
export DATASET_ROOT="/your/data/path"
export LOAD_PATH="/your/data/path/internvl3/raw_ckpt/InternVL3-8B"
```

以 `internvl3` 为例，目录结构应为：
```
/your/data/path/
└── internvl3/
    ├── raw_ckpt/
    │   └── InternVL3-8B/
    └── ...（数据集文件）
```

### 4.2 其他模型的数据路径

| 模型 | 默认 LOAD_PATH |
|------|----------------|
| internvl3 | `$DATASET_ROOT/internvl3/raw_ckpt/InternVL3-8B` |
| qwenvl | 使用 HuggingFace 路径 `init_from_hf_path`，无需 LOAD_PATH |
| opensora | `$DATASET_ROOT/opensora/ckpt/OpenSora1.2` |
| cogvideox | `$DATASET_ROOT/cogvideox/ckpt/CogVideoX` |

如果数据不在上述默认位置，可在运行前设置对应的环境变量：
```bash
export DATASET_ROOT="/your/data/path"
export LOAD_PATH="/your/custom/ckpt/path"
```

---

## 5. 运行 Task6

确保你在 `lmsv_rec` 目录下执行：

```bash
cd "${CLONE_DIR}/lmsv_rec"

# 设置路径（按需修改）
export MINDSPEED_MM_PATH="/shared/mindspeed-mm"  # 指向 workspace root，自动推导 MindSpeed-MM 子目录
export DATASET_ROOT="/your/data/path"
export LOAD_PATH="/your/data/path/internvl3/raw_ckpt/InternVL3-8B"

# 运行
python do.py
```

正常执行后会看到如下输出：
```
[主控][任务] 开始执行 【多模态模型】整网泛化变异测试
[Task6][阶段] 开始Task6多模态整网变异和验证任务
[Task6] 模型: InternVL3, 迭代: 1, 变异数: 2, 模式: pta_msa
...
[Task6] 第1轮完成（有效突变），耗时: xxxs
```

结果会保存在 `<lm-sv-root>/output/<时间戳>/` 目录下，报告在 `lmsv_rec/results/` 下。

---

## 6. 包含的手动修改说明

以下修改已内嵌在当前交付物中，**不需要你手动再打补丁**：

### 6.1 msadapter bfloat16 fallback 补丁
**位置：** `msadapter/msadapter/_utils.py` 和 `serialization.py`（位于 MindSpeed-MM 工作区的 `msadapter/` 目录下）

当 MindSpore 内置的 `np_dtype.bfloat16` 不可用时，自动 fallback 到 `ml_dtypes.bfloat16`，避免 MSA 执行时崩溃。

### 6.2 transformers 兼容性补丁
**位置：** 由 `setup_task6_envs.sh` 自动应用

`msadapter` 和 `mindspeed` 两个环境的 `site-packages/transformers/modeling_utils.py` 均针对 MindSpeed-MM 的 MSA 侧加载逻辑做了签名适配。虽然 transformers 4.55.2 的调用处已使用关键字参数，但 msadapter 的装饰器会将其重新展开为位置参数传递，导致运行时仍出现 `TypeError`。因此**所有版本均需应用此补丁**，由 `setup_task6_envs.sh` 自动完成。

### 6.3 decord 运行时补丁
**位置：** `scripts/envset/mm-pta-task6.sh`（运行时自动生成）

PTA 环境执行前，脚本会在 `/tmp/decord_patch/decord_fix.py` 中写入 monkey-patch，修复某些 decord 版本缺少 `cpu` 属性的问题。每次运行自动生效，无需手动干预。

### 6.4 libstdc++ 兼容性修复
**位置：** `scripts/envset/mm-msa-task6.sh`

MSA 执行前自动优先加载当前 conda 环境（`$CONDA_PREFIX/lib`）中的 `libstdc++.so.6`，解决 `GLIBCXX_3.4.29` 缺失问题。已去除旧版的 `/root/anaconda3/envs/msadapter/lib` 硬编码。

---

## 7. 常见问题排查

### Q1: `ERROR: PTA env script not found`
**原因：** 旧版 runtime 脚本引用 `mm-pta-task6` 时漏写了 `.sh` 后缀。
**解决：** 本仓库已修复此问题，如出现请确认你拉取的是最新代码。

### Q2: `ModuleNotFoundError: No module named 'mindspeed'`
**原因：** `MINDSPEED_MM_PATH` 指向错误，或 MindSpeed-MM 工作区目录结构不正确。
**解决：**
```bash
ls "${MINDSPEED_MM_PATH}"                        # 应存在
ls "${MINDSPEED_MM_PATH}/MindSpeed"              # 应存在（若 MINDSPEED_MM_PATH 为 workspace root）
ls "${MINDSPEED_MM_PATH}/Megatron-LM"            # 应存在（若 MINDSPEED_MM_PATH 为 workspace root）
```

### Q3: `GLIBCXX_3.4.29 not found`
**原因：** 系统 libstdc++ 版本过旧。
**解决：** 确认 `mm-msa-task6.sh` 中使用了 `$CONDA_PREFIX/lib/libstdc++.so.6`，并确认 `msadapter` 环境已正确激活。

### Q4: `FileNotFoundError: /data2/dataset/...`
**原因：** 数据集路径未覆盖。
**解决：** 运行前设置 `export DATASET_ROOT=/your/data/path` 和 `export LOAD_PATH=/your/ckpt/path`。

### Q5: `msadapter` 环境里 `import msadapter` 报错 `No module named 'msadapter'`
**原因：** `msadapter` conda 环境本身**没有安装 `msadapter` pip 包**（这是正常的）。
**解决：** `msadapter` 模块是通过 `PYTHONPATH` 从 MindSpeed-MM 工作区的 `msadapter/` 源码目录加载的。只要 `MINDSPEED_MM_PATH` 正确，且工作区下存在 `msadapter/` 目录，就会正常加载。

### Q6: 如何在新机器上快速部署环境？
**解决：** 使用 `standard_env/` 中的裸环境定义 + `setup_task6_envs.sh` 一键补丁：
```bash
cd task6_conda_envs_export/standard_env
conda env create -f mindspeed_bare.yml -n mindspeed
conda env create -f msadapter_bare.yml -n msadapter
cd ../automated_setup
bash setup_task6_envs.sh
```
此流程会自动完成所有定制化修改，无需手动干预。

---

## 8. 环境相关文件清单

| 文件/目录 | 说明 | 大小（参考） |
|-----------|------|-------------|
| `task6_conda_envs_export/standard_env/mindspeed_bare.yml` | PTA 裸环境定义 | ~7 KB |
| `task6_conda_envs_export/standard_env/msadapter_bare.yml` | MSA 裸环境定义 | ~7 KB |
| `task6_conda_envs_export/automated_setup/requirements.txt` | 标准库版本清单（157 个包） | ~5 KB |
| `task6_conda_envs_export/automated_setup/setup_task6_envs.sh` | 一键定制化脚本 | ~6 KB |
| `task6_conda_envs_export/automated_setup/patches/` | 补丁文件目录 | ~1 MB |
| `task6_conda_envs_export/automated_setup/patches/apex/` | apex 本地编译包 | ~500 KB |

---

*文档更新时间：2026-04-14*
*对应代码分支：lm-sv/lmsv_rec (Task6)*
