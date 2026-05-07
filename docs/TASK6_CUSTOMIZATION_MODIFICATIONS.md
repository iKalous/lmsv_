# Task6 环境定制化修改记录

> 本文档详细记录基于用户指定版本清单进行的最小必要定制化修改。
> 所有修改均服务于 CogVideoX Task6 在昇腾 NPU 上的正常运行。
> 两个 conda 环境（`mindspeed` 和 `msadapter`）基于**同一套版本清单**构建。
>
> 自动化脚本位置：`<lm-sv-root>/task6_conda_envs_export/automated_setup/setup_task6_envs.sh`
>
> **最新更新（2026-04-21）**：环境搭建采用"裸环境 → 补充安装 → 源码补丁"三阶段流程：
> 1. 从 `standard_env/` 加载裸 conda 环境（仅含 requirements.txt 标准库）
> 2. 运行 `setup_task6_envs.sh`：安装缺失包 + 应用所有定制化修改
> 3. 得到完整可运行的 Task6 环境

---

## 1. 基础版本清单

用户指定的统一版本清单共 **157 个包**，完整列表见：
`/zyl/lm-sv/task6_conda_envs_export/automated_setup/requirements.txt`

### 1.1 两个环境的共同关键包版本

| 包名 | 版本 | 备注 |
|------|------|------|
| transformers | 4.55.2 | 见下方修改 #2 |
| torch | 2.7.1 | mindspeed 环境实际为 `2.7.1+cpu`（见偏差说明） |
| torch-npu | 2.7.1.post2 | |
| torchvision | 0.22.1 | |
| mindspore | 2.8.0 | |
| numpy | 1.26.0 | |
| scipy | 1.15.3 | |
| accelerate | 1.9.0 | |
| tokenizers | 0.21.4 | |
| safetensors | 0.5.1 | |
| diffusers | 0.30.3 | |
| ml-dtypes | 0.5.4 | 见下方修改 #1 |
| mindspeed | 0.12.1 | editable，从本地路径安装 |

---

## 2. 环境搭建完整流程

Task6 的环境搭建遵循"**裸环境 → 补充安装 → 源码补丁**"的三阶段流程。所有操作由自动化脚本 `setup_task6_envs.sh` 一键完成。

### 2.1 阶段一：准备裸环境（Standard Environment）

**起点**：两个仅安装了 `requirements.txt` 中标准库的 conda 环境。

#### 2.1.1 裸环境的来源

裸环境定义位于 `task6_conda_envs_export/standard_env/`：

| 文件 | 说明 | 对应环境 |
|------|------|----------|
| `mindspeed_bare.yml` | PTA 侧裸环境（conda env export） | `mindspeed` |
| `msadapter_bare.yml` | MSA 侧裸环境（conda env export） | `msadapter` |

#### 2.1.2 还原裸环境的三种方式

**方式一：通过 yml 文件创建（推荐）**
```bash
cd task6_conda_envs_export/standard_env
conda env create -f mindspeed_bare.yml -n mindspeed
conda env create -f msadapter_bare.yml -n msadapter
```

**方式二：通过 requirements.txt 安装**
```bash
# 创建基础环境并安装依赖
conda create -n mindspeed python=3.10 -y
conda activate mindspeed
pip install -r ../automated_setup/requirements.txt

conda create -n msadapter python=3.10 -y
conda activate msadapter
pip install -r ../automated_setup/requirements.txt
```

**方式三：通过 conda-pack 跨机器部署**
```bash
# 在新机器上解压已打包的环境
mkdir -p ~/conda_envs/mindspeed ~/conda_envs/msadapter
tar -xzf mindspeed_bare.tar.gz -C ~/conda_envs/mindspeed
tar -xzf msadapter_bare.tar.gz -C ~/conda_envs/msadapter

# 注册到 conda
ln -s ~/conda_envs/mindspeed $(conda info --base)/envs/mindspeed
ln -s ~/conda_envs/msadapter $(conda info --base)/envs/msadapter
```

#### 2.1.3 裸环境的特征

裸环境**不含**任何 Task6 定制化修改：
- `transformers` 为原始版本（4.55.2）
- `msadapter` 源码为原始版本（无 bfloat16 fallback）
- 无 decord 运行时补丁
- 无 libstdc++ 路径修复

### 2.2 阶段二：运行 setup_task6_envs.sh（一键定制化）

在裸环境准备就绪后，执行自动化脚本完成所有定制化修改。

#### 2.2.1 脚本定位与用法

```bash
cd task6_conda_envs_export/automated_setup
bash setup_task6_envs.sh
```

**可选环境变量**：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `PTA_NAME` | PTA conda 环境名 | `mindspeed` |
| `MSA_NAME` | MSA conda 环境名 | `msadapter` |
| `MM_WORKSPACE` | MindSpeed-MM 工作区绝对路径（用于 msadapter bfloat16 补丁定位） | 自动推断（从脚本位置或 MINDSPEED_MM_PATH） |
| `LMSV_REC` | lmsv_rec 项目绝对路径 | 自动推断 |

#### 2.2.2 脚本执行流程（6 个阶段）

```
┌─────────────────────────────────────────────────────────────────────┐
│                    setup_task6_envs.sh 执行流程                      │
├─────────────────────────────────────────────────────────────────────┤
│ 阶段 1: 检查 conda 环境和前置条件                                     │
│    └─ 确认 mindspeed、msadapter 环境存在                             │
│    └─ 确认 transformers、torch 等关键包已安装                        │
├─────────────────────────────────────────────────────────────────────┤
│ 阶段 2: 推断 MindSpeed-MM 工作区路径                                 │
│    └─ 优先从 MM_WORKSPACE 环境变量获取                               │
│    └─ 自动推断：从脚本位置向上查找，或从 MINDSPEED_MM_PATH 推断      │
├─────────────────────────────────────────────────────────────────────┤
│ 阶段 2.5: 安装缺失包（ mindspeed 环境）                              │
│    └─ 从预定义列表安装 ~23 个 PyPI 包                                │
│    └─ 从 patches/apex/ 复制 apex 到 site-packages                    │
├─────────────────────────────────────────────────────────────────────┤
│ 阶段 3: 安装缺失包（ msadapter 环境）                                │
│    └─ 从预定义列表安装 ~18 个 PyPI 包                                │
│    └─ 从华为云 URL 安装 torch_npu 2.7.1.post2                       │
├─────────────────────────────────────────────────────────────────────┤
│ 阶段 4: 应用 transformers 兼容性补丁（两个环境）                      │
│    └─ 在 site-packages/transformers/modeling_utils.py 中修改调用签名 │
├─────────────────────────────────────────────────────────────────────┤
│ 阶段 5: 应用 msadapter bfloat16 fallback 补丁                        │
│    └─ 覆盖 msadapter/msadapter/_utils.py                           │
│    └─ 覆盖 msadapter/msadapter/serialization.py                    │
├─────────────────────────────────────────────────────────────────────┤
│ 阶段 6: 部署环境脚本到 lmsv_rec/scripts/envset/                      │
│    └─ mm-pta-task6.sh（含 decord 运行时补丁）                        │
│    └─ mm-msa-task6.sh（含 libstdc++ 兼容性修复）                     │
├─────────────────────────────────────────────────────────────────────┤
│ 阶段 7: 验证所有修改                                                 │
│    └─ 验证 transformers 补丁                                         │
│    └─ 验证 msadapter bfloat16 fallback                              │
│    └─ 验证环境脚本存在                                               │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3 阶段三：补充安装的完整包清单

以下包在裸环境中缺失，由脚本自动补充安装。

#### 2.3.1 mindspeed (PTA) 环境补充包

| # | 包名 | 版本 | 安装方式 | 说明 |
|---|------|------|----------|------|
| 1 | `annotated_types` | 0.7.0 | pip | pydantic 依赖 |
| 2 | `bitsandbytes_npu_beta` | 0.45.3 | pip | NPU 量化支持 |
| 3 | `click` | 8.3.1 | pip | CLI 框架 |
| 4 | `imageio` | 2.37.2 | pip | 图像 I/O |
| 5 | `imageio_ffmpeg` | 0.6.0 | pip | imageio 的 ffmpeg 后端 |
| 6 | `jaxtyping` | 0.3.5 | pip | JAX 类型注解 |
| 7 | `jsonschema` | 4.26.0 | pip | JSON Schema 验证 |
| 8 | `jsonschema_specifications` | 2025.9.1 | pip | jsonschema 依赖 |
| 9 | `nvidia_ml_py` | 13.590.48 | pip | NVIDIA ML 库 Python 绑定 |
| 10 | `prettytable` | 3.17.0 | pip | 表格格式化输出 |
| 11 | `pydantic` | 2.12.5 | pip | 数据验证 |
| 12 | `pydantic_core` | 2.41.5 | pip | pydantic 核心 |
| 13 | `pyecharts` | 2.0.9 | pip | 图表生成 |
| 14 | `pyvers` | 0.2.2 | pip | 版本管理工具 |
| 15 | `qwen_vl_utils` | 0.0.14 | pip | QwenVL 工具库 |
| 16 | `ray` | 2.10.0 | pip | 分布式计算框架 |
| 17 | `referencing` | 0.37.0 | pip | JSON 引用解析 |
| 18 | `rpds_py` | 0.30.0 | pip | Rust 持久化数据结构 |
| 19 | `simplejson` | 3.20.2 | pip | JSON 处理库 |
| 20 | `swanlab` | 0.7.8 | pip | 实验跟踪工具 |
| 21 | `typing_inspection` | 0.4.2 | pip | 类型检查工具 |
| 22 | `wadler_lindig` | 0.1.7 | pip | 格式化工具 |
| 23 | `wrapt` | 2.1.1 | pip | 装饰器工具 |
| 24 | `apex` | 0.1+ascend | 复制 | 从 patches/apex/ 复制，本地编译的昇腾版本 |

**安装说明**：
- 23 个 PyPI 包分批安装（每批 10 个，避免命令行过长）
- `apex` 为本地编译的昇腾适配版本，无 PyPI 发布，直接从 `patches/apex/` 复制到 site-packages
- `argparse` 为 Python 3.10 内置模块，跳过

#### 2.3.2 msadapter (MSA) 环境补充包

| # | 包名 | 版本 | 安装方式 | 说明 |
|---|------|------|----------|------|
| 1 | `annotated_doc` | 0.0.4 | pip | 文档注解 |
| 2 | `annotated_types` | 0.7.0 | pip | pydantic 依赖 |
| 3 | `click` | 8.3.1 | pip | CLI 框架 |
| 4 | `imageio` | 2.37.3 | pip | 图像 I/O |
| 5 | `jaxtyping` | 0.3.5 | pip | JAX 类型注解 |
| 6 | `nvidia_ml_py` | 13.590.48 | pip | NVIDIA ML 库 Python 绑定 |
| 7 | `prettytable` | 3.17.0 | pip | 表格格式化输出 |
| 8 | `pydantic` | 2.12.5 | pip | 数据验证 |
| 9 | `pydantic_core` | 2.41.5 | pip | pydantic 核心 |
| 10 | `pyecharts` | 2.0.9 | pip | 图表生成 |
| 11 | `qwen_vl_utils` | 0.0.14 | pip | QwenVL 工具库 |
| 12 | `shellingham` | 1.5.4 | pip | Shell 检测库 |
| 13 | `simplejson` | 3.20.2 | pip | JSON 处理库 |
| 14 | `swanlab` | 0.7.8 | pip | 实验跟踪工具 |
| 15 | `torch_npu` | 2.7.1.post2 | URL | 华为云 wheel 安装，昇腾 NPU 后端 |
| 16 | `tornado` | 6.5.4 | pip | Web 框架 |
| 17 | `typer` | 0.24.1 | pip | CLI 框架 |
| 18 | `typing_inspection` | 0.4.2 | pip | 类型检查工具 |
| 19 | `wadler_lindig` | 0.1.7 | pip | 格式化工具 |
| 20 | `wrapt` | 2.1.1 | pip | 装饰器工具 |

**特殊安装说明**：
- `torch_npu` 从华为云 URL 直接安装 wheel：`https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.7.1/torch_npu-2.7.1.post2-cp310-cp310-manylinux_2_28_aarch64.whl`
- `argparse` 为 Python 3.10 内置模块，跳过
- `packaging` 标准环境已有 26.0（高于原环境中的 25.0），以标准环境为准，跳过
- `hccl`/`te` 等 CANN 包已由 CANN toolkit 安装，跳过

### 2.4 最终环境状态

| 指标 | mindspeed | msadapter |
|------|-----------|-----------|
| 标准库包数（requirements.txt） | ~160 | ~160 |
| 补充安装的包（PyPI + patches） | ~24 | ~20 |
| transformers 补丁 | 已应用（所有版本均需） | 已应用（所有版本均需） |
| msadapter bfloat16 fallback | 不涉及（PTA 侧不加载 msadapter） | 已应用 |
| 环境脚本部署 | 已部署（mm-pta-task6.sh） | 已部署（mm-msa-task6.sh） |
| torch_npu | 已由 mindspeed 自带 | 从华为云 URL 安装 2.7.1.post2 |
| apex | 从 patches 复制 | 不涉及 |

---

## 3. 标准库源码修改详情

### 修改 1：msadapter bfloat16 fallback 补丁

**修改原因：** MindSpore 内置的 `np_dtype.bfloat16` 在某些版本/设备上不可用，导致 MSA 侧加载 checkpoint 时崩溃。

**修改位置（项目源码，非 conda site-packages）：**

#### 1) `msadapter/msadapter/_utils.py`

| 方法名 | 修改内容 |
|--------|----------|
| `_bf16()` | 新增 fallback 逻辑：当 `support_bf16()` 返回 False 或 `np_dtype` 没有 `bfloat16` 属性时，自动导入 `ml_dtypes` 并使用 `ml_dtypes.bfloat16` 替代 |
| `_rebuild_tensor_v2()` | 在重建 tensor 时，若 array 的 dtype 为 `_bf16()` 且设备不支持 bfloat16，则自动转换为 `np.float16` |
| `dtype_to_nptype()` | 当 dtype 为 `mindspore.bfloat16` 时，优先返回 `np_dtype.bfloat16`，不存在则返回 `ml_dtypes.bfloat16` |

**关键代码片段：**

```python
def _bf16():
    if not hasattr(_bf16, 'bf16'):
        if support_bf16() and hasattr(np_dtype, 'bfloat16'):
            _bf16.bf16 = np_dtype.bfloat16
        else:
            import ml_dtypes
            _bf16.bf16 = ml_dtypes.bfloat16
    return _bf16.bf16
```

#### 2) `msadapter/msadapter/serialization.py`

| 方法名 | 修改内容 |
|--------|----------|
| `_bf16()` | 同 `_utils.py`，提供 `ml_dtypes.bfloat16` fallback |
| `legacy_safe_load_file()` | 加载 safetensors 时，对 bfloat16 数据做兼容性处理：不支持则转换为 float16 |
| `safe_load_file()` | 同上，在 `convert()` 内部函数中对 bfloat16 做 fallback 转换 |

---

### 修改 2：transformers `modeling_utils.py` 签名适配补丁

**修改原因：** MindSpeed-MM 的 MSA 侧模型加载逻辑调用 `transformers` 的 `_load_state_dict_into_meta_model` 时，传参方式与该函数签名不兼容，导致 `TypeError`。

**修改位置（两个 conda 环境的 site-packages）：**

- **mindspeed 环境：** `$(conda env list | grep mindspeed | awk '{print $2}')/lib/python3.10/site-packages/transformers/modeling_utils.py`
- **msadapter 环境：** `$(conda env list | grep msadapter | awk '{print $2}')/lib/python3.10/site-packages/transformers/modeling_utils.py`

> 实际路径取决于 conda 安装位置，可通过 `python -c "import transformers, inspect; print(inspect.getsourcefile(transformers.modeling_utils))"` 获取。

**受影响方法：** `load_state_dict()` 方法内部对 `_load_state_dict_into_meta_model()` 的调用

**修改内容：** 将两个**位置参数**改为**关键字参数**：

| 参数 | 修改前（位置参数） | 修改后（关键字参数） |
|------|-------------------|---------------------|
| 第 4 个参数 | `expected_keys` | `expected_keys=expected_keys` |
| 第 5 个参数 | `reverse_key_renaming_mapping` | `reverse_renaming_mapping=reverse_key_renaming_mapping` |

**修改前代码（transformers 4.55.2，约第 975 行，`load_shard_file` 内）：**

```python
disk_offload_index, cpu_offload_index = _load_state_dict_into_meta_model(
    model_to_load,
    state_dict,
    shard_file,
    expected_keys,
    reverse_key_renaming_mapping,
    device_map=device_map,
    ...
)
```

**修改后代码：**

```python
disk_offload_index, cpu_offload_index = _load_state_dict_into_meta_model(
    model_to_load,
    state_dict,
    shard_file,
    expected_keys=expected_keys,
    reverse_renaming_mapping=reverse_key_renaming_mapping,
    device_map=device_map,
    ...
)
```

**注意：** 此修改在两个环境**均**需应用。虽然 mindspeed 为 PTA 侧环境，但 Task6 的代码路径在特定场景下也会加载该模块，保持两边一致可避免潜在差异。

---

## 4. 运行时脚本适配（非源码修改，仅作备忘）

以下修改不涉及标准库源码，但属于环境定制化的一部分，供重建环境时参考：

### 4.1 decord 运行时补丁
- **位置：** `scripts/envset/mm-pta-task6.sh`
- **作用：** 在 `/tmp/decord_patch/decord_fix.py` 中写入 monkey-patch，当 `decord` 模块缺少 `cpu` 属性时注入空实现，避免 `AttributeError`

### 4.2 libstdc++ 兼容性修复
- **位置：** `scripts/envset/mm-msa-task6.sh`
- **作用：** 在激活 msadapter 环境后，将 `$CONDA_PREFIX/lib` 加入 `LD_LIBRARY_PATH`，优先使用 conda 环境自带的 `libstdc++.so.6`，解决系统库版本过旧导致的 `GLIBCXX_3.4.29 not found`

---

## 5. 版本偏差说明

在实际搭建过程中，以下包版本与用户指定清单存在偏差，均为**必要的最小化调整**：

| 环境 | 包名 | 用户指定版本 | 实际版本 | 偏差原因 |
|------|------|-------------|----------|----------|
| mindspeed | torch | 2.7.1 | 2.7.1+cpu | PyTorch 在 aarch64/Linux 平台上仅提供 `+cpu` 构建标签的 wheel，`2.7.1+cpu` 与 `2.7.1` 功能完全一致，仅 build tag 不同 |
| both | grpcio | 1.78.1 | 1.78.0 | PyPI 上不存在 `grpcio==1.78.1`，最近可用版本为 `1.78.0` |

**偏差处理原则：**
- `torch==2.7.1+cpu`：aarch64 平台无纯 `2.7.1` wheel，此为平台限制，功能等价
- `grpcio==1.78.0`：次版本号内的小版本差异（1.78.0 vs 1.78.1），API 兼容

---

## 6. 修改验证方法

### 6.1 验证 transformers 补丁

```bash
# 动态获取 transformers 路径并验证
conda activate mindspeed
python -c "
import transformers, inspect
src = inspect.getsourcefile(transformers.modeling_utils)
import subprocess; subprocess.run(['grep', '-n', 'expected_keys=expected_keys', src])
"

conda activate msadapter
python -c "
import transformers, inspect
src = inspect.getsourcefile(transformers.modeling_utils)
import subprocess; subprocess.run(['grep', '-n', 'expected_keys=expected_keys', src])
"
```

两行均应有输出，且显示 `reverse_renaming_mapping=reverse_key_renaming_mapping` 在同一调用中。

### 6.2 验证 msadapter bfloat16 fallback

```bash
python -c "
import sys, os
mm_path = os.environ.get('MINDSPEED_MM_PATH', '/shared/mindspeed-mm')
sys.path.insert(0, os.path.join(mm_path, 'msadapter'))
from msadapter._utils import _bf16
print('bf16 dtype:', _bf16())
"
```

应正常输出 bfloat16 dtype 对象，不抛 `AttributeError`。

### 6.3 验证环境包数量

```bash
conda activate mindspeed && python -m pip list | wc -l  # 应 ≈ 157
conda activate msadapter && python -m pip list | wc -l    # 应 ≈ 157
```

---

## 7. 自动化重建

如需在其他机器上重建相同环境：

```bash
cd /zyl/lm-sv/task6_conda_envs_export/automated_setup
python setup_envs.py
```

脚本会自动完成：
1. 两个 conda 环境的包版本对齐
2. transformers 补丁自动应用
3. 偏差报告输出到 `logs/deviations_*.txt`

---

---

## 8. 最新源码级别修改文档

本文档侧重于环境搭建过程的记录。如需查看**完整的源码级别修改说明**（含每一条改动的精确代码、自动化补丁文件、一键应用脚本），请参阅：

> **`docs/source-level-customizations.md`**

该文档包含：
- 5 处定制化修改的完整源码级别说明
- `automated_setup/patches/` 补丁文件清单
- `setup_task6_envs.sh` 一键应用脚本说明
- 手动验证方法

---

*文档更新时间：2026-04-20*
*对应任务：Task6 CogVideoX 多模态整网泛化变异测试*
