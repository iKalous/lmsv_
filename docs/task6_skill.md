# Task6 开发经验与避坑指南

> **作者**: 邹英龙
> **更新日期**: 2026-04-13
>
> 本文档记录Task6开发过程中踩过的所有坑、经验、原则和上下文。仅供开发参考，不属于最终交付内容。

---

## 核心原则（不可违背）

### 原则0：InternVL3跑通后作为基准

**InternVL3已成功跑通，证明Task6框架、环境配置、日志捕获逻辑均正确。**

**在调试其他模型时，以下部分禁止修改：**

1. **Task6核心框架**（`utils/task/task6.py`）：
   - 环境变量读取逻辑（`_init_config`）
   - PTA/MSA执行流程（`_run_pta_verify`、`_run_msa_verify`）
   - 日志解析正则表达式
   - 成功/失败判定逻辑

2. **InternVL3相关脚本**：
   - `scripts/runtime/mm_pta_internvl3.sh`
   - `scripts/runtime/mm_msa_internvl3.sh`
   - `scripts/runtime/msa_internvl3_8B_real.sh`

3. **环境配置**：
   - MindSpeed-MM 路径（`MINDSPEED_MM_PATH`，兼容旧版 `PTA_PATH`/`MSA_PATH`）
   - Conda环境名称（`PTA_NAME`、`MSA_NAME`）
   - CANN环境设置方式

**调试其他模型的正确方式**：
- 如果其他模型执行失败，应修改该模型**专属的脚本**（如`mm_pta_opensora.sh`）
- 不应为了修复某个模型而回退Task6框架的修复
- 各模型脚本需独立适配，但框架层逻辑保持一致

---

### 原则1：PTA和日志分析代码不能随意修改

**四个模型（InternVL3、QwenVL2.5、OpenSora1.2、CogVideoX）的PTA执行和日志分析部分已经验证是正确的。**

| 模型 | 模式 | PTA状态 | MSA状态 |
|------|------|---------|---------|
| InternVL3 | 训练 | ✅ | ✅ |
| QwenVL2.5 | 推理 | ✅ | ❌ (InnerInplaceIndexPut shape mismatch) |
| OpenSora1.2 | 推理 | ✅ | ❌ (UntypedStorage错误) |
| CogVideoX | 训练 | ✅ | ✅ |

**风险**：修改这些已验证的代码可能导致回归问题，使原本正常的模型执行失败。

---

### 原则2：环境必须与Task1和docs保持一致

| 项目 | 要求 |
|------|------|
| PTA环境 | `mindspeed` conda环境 |
| MSA环境 | `msadapter` conda环境 |
| 路径配置 | 参考 `docs/environment-modifications.md` 一键安装指南 |
| 环境变量 | MINDSPEED_MM_PATH、LMSV_OUTPATH 必须正确设置（兼容 PTA_PATH/MSA_PATH） |

**风险**：环境不一致会导致各种难以排查的错误。

---

### 原则3：配置统一走 `config.json`，禁止硬编码（基本原则，不可违反）

**Task6 的配置方式已与 Task1-5 完全一致：所有参数通过 `config.json` 的 `tasks["6"]` 字典传入，不再使用 `TASK6_*` 环境变量。**

**实现方式**：
1. **`task6.py` 配置读取**（`_init_config` 函数）：
   - 与 Task1-5 一致，所有任务参数从 `params` 字典读取
   - `params` 由 `do.py` 从 `config.json` 的 `tasks["6"]` 中提取并传入
   - 不再读取 `TASK6_*` 环境变量
   - 支持的参数字段：
     - `MODE` - 运行模式
     - `TOTAL_ITER` - 最大迭代次数
     - `MUTNM` - 每轮变异参数个数
     - `TRAIN_ITER` - 每轮训练/推理步数（兼容旧版 `SAVE_STEPS` / `TRAIN_ITERS`）
     - `COMPARE_MODE` - 对比模式
     - `MODEL_NAME` - 模型名称
     - `BASE_SEED` - 基础随机种子

2. **全局路径/环境名**：
   - `PTA_NAME`、`MSA_NAME`、`MINDSPEED_MM_PATH` 等仍在 `config.json` 根级别配置
   - `do.py` 会将这些字段导出为环境变量供子脚本使用
   - Task6 统一使用 `MINDSPEED_MM_PATH`，不再拆分 `PTA_PATH` / `MSA_PATH`

3. **shell 脚本配置读取**：
   - 使用 `${VAR:-default}` 语法支持环境变量覆盖
   - 示例：`export LOAD_PATH="${LOAD_PATH:-${DATASET_ROOT}/model/ckpt}"`

---

### 原则4：无效突变处理

**无效突变定义**：PTA执行时日志中出现真正的error（Traceback、RuntimeError等，Warning不算）的突变。

**处理原则**：
1. **不计入轮次**：无效突变不计入总迭代次数，该轮次需要重新执行
2. **清理策略**：
   - ✅ **保留**：`tmp/task6/pta_verify_iterX.log` —— 用于调试分析
   - ❌ **删除**：`tmp/task6/mutation_results/`中的`mutation_genX.json`
   - ❌ **删除**：`tmp/task6/`中的临时配置文件
   - ❌ **删除**：`results/iters/iter_X/`归档目录（无效突变不归档）
3. **有效突变**：即使MSA失败也是有效突变，所有日志和配置都要保留

**关键区分**：
- **PTA失败** = 无效突变（崩溃/出错，无法验证）→ 清理 → 不计入轮次
- **MSA失败** = 有效突变（PTA正常执行，发现框架问题）→ 全部保留 → 计入轮次

---

### 原则5：成功状态判断标准

**绝对禁止**：通过"有没有loss"、"有没有metrics"、"返回码是否为0"来判断PTA/MSA是否成功。

**唯一正确方式**：
1. **执行时判断**：检查日志中是否有真正的error（Traceback、RuntimeError、OSError等，Warning不算）
2. **报告生成时**：从`status.json`的`PTA_VERIFY`和`MSA_VERIFY`字段读取状态

---

## 关键避坑点

### 1. 路径配置（重要）

**原则**: Task6 默认使用相对于项目根目录 `lmsv_rec/` 的相对路径，便于在不同环境部署。

**默认路径**:
- `MINDSPEED_MM_PATH` 默认: `<lm-sv-root>/../mindspeed-mm`（指向 workspace root，自动推导 MindSpeed-MM 子目录）
- `LMSV_OUTPATH` 默认: `output`

**解决方案**:
```bash
# 使用绝对路径（推荐，以实际部署路径为准）
export MINDSPEED_MM_PATH=<YOUR_MINDSPEED_MM_PATH>
export LMSV_OUTPATH=./output

# 或使用相对路径
export MINDSPEED_MM_PATH=../mm-new
```

**注意**: 所有 PTA/MSA 脚本内部会在执行前将相对路径转换为绝对路径，确保 `torchrun`/`msrun` 等分布式命令能正确找到入口文件。

---

### 2. 推理模型特殊处理

#### 2.1 异常判断不能通过loss

**关键避坑点**：对于QwenVL等推理模型，**不能**通过检查是否有loss输出来判断是否异常。

**原因**：
- 推理模型（inference）没有训练过程，自然不会有loss输出
- 判断推理模型成功的标准是：**返回码是否为0**

**正确做法**：
```bash
# 错误：检查loss（仅适用于训练模型）
if ! grep -q "loss:" ${LOG_FILE}; then
    echo "ERROR: No loss found"
fi

# 正确：检查返回码
if [ "$RETURN_CODE" -ne 0 ]; then
    echo "ERROR: Execution failed"
fi
```

#### 2.2 日志指标提取

推理模型没有loss、memory、time等指标，这些字段会是None，这是**正常行为**。

---

### 3. Shell脚本避坑指南

#### 3.1 set -e 与 grep 的陷阱

**问题**：使用`set -e`时，grep找不到匹配会返回1，导致脚本退出。

**解决方案**：
```bash
# 错误：会导致脚本退出
if grep -q "pattern" file; then
    ...
fi

# 正确：添加错误处理
if grep -q "pattern" file 2>/dev/null; then
    ...
fi

# 或：使用 || true
grep -q "pattern" file 2>/dev/null || true
```

#### 3.2 变量提取时的错误处理

```bash
# 错误：grep失败会导致脚本退出
VAR=$(grep "pattern" file | grep -oP '...')

# 正确：添加错误处理
VAR=$(grep "pattern" file 2>/dev/null | grep -oP '...' 2>/dev/null || echo "")
```

---

### 4. MSA Pipeline Parallel 日志捕获修复

**问题**：Pipeline Parallel (PP>1)配置下，loss只在最后一个pipeline stage的worker日志中输出（如worker_7），而脚本默认读取worker_0.log

**影响**：训练模型（InternVL3、CogVideoX）的MSA执行无法正确捕获loss

**修复详情**：
1. **查找逻辑**：修改等待循环，持续查找所有worker日志直到发现包含`loss:`的日志文件
2. **科学计数法支持**：修改正则表达式`[\d.]+` → `[\d.E+-]+`，支持`1.269427E+01`格式的loss值
3. **分离指标提取**：loss从包含loss的worker提取，memory从所有worker中查找

**涉及脚本**：`msa_internvl3_8B_real.sh`、`msa_cogvideox_real.sh`

---

### 5. 训练模型MSA失败判定

**原则**：训练模型（InternVL3、CogVideoX）的MSA执行必须有loss输出，无loss视为失败

**错误信息提取**：优先提取Python Error（Traceback/OSError/RuntimeError等），无Error时取最后一个WARNING作为错误信息

**结果展示**：`status.json`中`reason`字段包含提取的错误信息

---

### 6. OpenSora推理优化

**问题1**：`num_inference_steps=30`导致显存不足
- **解决**：改为与Task6的`TRAIN_ITER`一致（默认5）

**问题2**：`--train-iters 5010`导致执行5010轮推理
- **解决**：改为`--train-iters ${TRAIN_ITERS:-1}`（`TRAIN_ITER`映射而来）只执行1次完整推理

**问题3**：prompts文件包含10个prompts，导致执行10组推理
- **解决**：使用单prompt文件（只取第一个prompt）

**效果**：现在只执行一组0/5 → 5/5的推理，耗时约140秒

---

### 7. MSA环境UntypedStorage兼容性

**问题**：`ModuleNotFoundError: No module named 'msadapter.UntypedStorage'`

**原因**：safetensors库需要`torch.UntypedStorage`，但msadapter缺少该属性

**解决方案**：在`mm_msa_opensora.sh`中创建`sitecustomize.py`注入FakeUntypedStorage

```python
# sitecustomize.py
class FakeUntypedStorage:
    def __init__(self, *args, **kwargs):
        self._data = bytearray()
        self._size = 0

    @classmethod
    def from_file(cls, filename, shared, nbytes):
        instance = cls.__new__(cls)
        with open(filename, 'rb') as f:
            data = f.read(nbytes) if nbytes > 0 else f.read()
            instance._data = bytearray(data)
            instance._size = len(instance._data)
        return instance

    def resize_(self, size):
        ...

# 注入到msadapter和torch模块
import msadapter
msadapter.UntypedStorage = FakeUntypedStorage
import torch
torch.UntypedStorage = FakeUntypedStorage
```

**使用方式**：将patch目录加入PYTHONPATH最前面
```bash
export PYTHONPATH="${MSA_PATCH_DIR}:${PYTHONPATH}"
```

**注意**：这是用户侧解决方案，不属于框架bug

---

### 8. MSA环境依赖版本

**正确的依赖版本组合**：

| 包 | 版本 | 说明 |
|---|---|---|
| python | 3.10 | - |
| mindspore | 2.7.1 | 固定版本 |
| msadapter | 0.0.5 | msadapter环境已安装 |
| numpy | 1.26.0 | **关键**：必须<=1.26.0 |
| ml_dtypes | 0.3.0 | **关键**：必须与numpy兼容 |
| scipy | 1.11.4 | 与numpy 1.26兼容 |

**环境修复命令**：
```bash
conda activate msadapter
pip install numpy==1.26.0 --force-reinstall
pip install ml_dtypes==0.3.0 --force-reinstall
pip install scipy==1.11.4 --force-reinstall
```

---

### 9. CogVideoX模型文件路径

**问题**：CogVideoX配置中`vae.safetensors`和`transformer`路径与实际文件不匹配

**实际文件结构**：
```
${DATASET_ROOT}/cogvideox/CogVideoX-5B/
├── vae/3d-vae.pt              # 不是vae.safetensors
├── text_encoder/              # 不是transformer
└── ...
```

**修复方案**：修改`assets/mm_configs/model_cogvideox.json`

---

### 10. dtype字符串解析问题（框架Bug）

**问题**：transformers库的`dict_dtype_to_str`函数无法处理"bf16"简写格式

**根因代码**：
```python
def dict_dtype_to_str(self, d):
    if d.get("dtype", None) is not None and not isinstance(d["dtype"], str):
        d["dtype"] = str(d["dtype"]).split(".")[1]  # <-- 问题所在
```

**分析**：
- 代码假设格式为"torch.float32"或"msadapter.float32"（包含点号）
- 分割后得到`["torch", "float32"]`，取索引1得到"float32"
- MSA环境配置中使用短格式"bf16"（不包含点号）
- 分割后得到`["bf16"]`，取索引1导致IndexError

**建议修复方案**：
1. 短期：在配置预处理中将简写格式转换为标准格式
2. 长期：向transformers库提交PR增强健壮性

---

### 11. OpenSora MSA设备初始化失败（框架Bug）

**问题**：MSA环境下OpenSora分布式初始化时`aclrtSetDevice`失败

**错误信息**：
```
RuntimeError: Call aclrtSetDevice failed, ret[507033]. Got device count[1] and device id[0], please check if device id is valid.
```

**根因分析**：
- 错误代码`507033` (ACL_ERROR_RT_DEVICE_ID_INVALID)
- MindSpore HCCL初始化时设备ID验证失败
- 可能是PTA执行后NPU设备未正确释放，或MSA与PTA环境之间的设备状态冲突

**对比状态**：
- PTA: ✅ 正常执行 (time=122092ms)
- MSA: ❌ 初始化失败

**修复建议**：
1. 在PTA和MSA之间添加设备释放检查
2. 显式设置`ASCEND_DEVICE_ID`环境变量
3. 向msadapter/MindSpore提交issue

---

### 12. CogVideoX MSA _load_state_dict_into_meta_model 签名不匹配（框架Bug）

**问题**：MindSpeed-MM 的 `_load_state_dict_into_meta_model` 函数签名与 transformers 4.57.6 不匹配

**错误信息**：
```
TypeError: _load_state_dict_into_meta_model() missing 1 required positional argument: 'reverse_renaming_mapping'
```

**根因分析**：
1. MindSpeed-MM 的 `_load_state_dict_into_meta_model` 需要 `expected_keys`, `is_safetensors` 等参数
2. transformers 4.57.6 调用时只传递了 `reverse_renaming_mapping` 等必需参数
3. 签名不匹配导致调用失败

**修复方案**：

修改文件：
1. `${MINDSPEED_MM_PATH}/MindSpeed/mindspeed/mindspore/third_party/transformers/modeling_utils.py`
2. `$(dirname ${MINDSPEED_MM_PATH})/MindSpeed/mindspeed/mindspore/third_party/transformers/modeling_utils.py`

修改内容：
```python
# 原签名（不兼容）
def _load_state_dict_into_meta_model(
    model: "PreTrainedModel",
    state_dict: Dict,
    shard_file: str,
    expected_keys: List[str],  # transformers 4.57.6 不传递此参数
    reverse_renaming_mapping: Dict[str, str],
    ...
)

# 修复后签名（兼容）
def _load_state_dict_into_meta_model(
    model: "PreTrainedModel",
    state_dict: Dict,
    shard_file: str,
    reverse_renaming_mapping: Dict[str, str],
    device_map: Optional[Dict] = None,
    disk_offload_folder: Optional[str] = None,
    disk_offload_index: Optional[Dict] = None,
    hf_quantizer: Optional[HfQuantizer] = None,
    keep_in_fp32_regex: Optional[re.Pattern] = None,
    device_mesh: Optional["torch.distributed.device_mesh.DeviceMesh"] = None,
    # 可选参数，用于兼容不同 transformers 版本
    expected_keys: Optional[List[str]] = None,
    cpu_offload_folder: Optional[str] = None,
    cpu_offload_index: Optional[Dict] = None,
    is_safetensors: bool = False,
    unexpected_keys: Optional[List[str]] = None,
) -> Tuple[Optional[Dict], Optional[Dict]]:
```

附加修复：
1. 处理 `expected_keys is None` 的情况
2. 处理 `unexpected_keys is None` 的情况（创建空列表作为 fallback）
3. 当 `disk_offload_folder is None` 时，确保返回的 `disk_offload_index` 也为 None

---

### 13. CogVideoX MSA disk_offload_folder 为 None 问题（框架Bug）

**问题**：transformers `_load_pretrained_model` 调用 `_load_state_dict_into_meta_model` 后，得到的 `disk_offload_index` 不为 None，但 `disk_offload_folder` 为 None，导致 `save_offload_index` 崩溃

**错误信息**：
```
File "/root/anaconda3/envs/msa-m/lib/python3.10/site-packages/transformers/modeling_utils.py", line 5473, in _load_pretrained_model
    save_offload_index(disk_offload_index, disk_offload_folder)
File "/root/anaconda3/envs/msa-m/lib/python3.10/site-packages/accelerate/utils/offload.py", line 73, in save_offload_index
    offload_index_file = os.path.join(offload_folder, "index.json")
TypeError: expected str, bytes or os.PathLike object, not NoneType
```

**修复方案**：

在函数返回前添加检查：
```python
# Only return offload indices if the corresponding folder is set
# (transformers library checks this and fails if disk_offload_index is set but folder is None)
if disk_offload_folder is None:
    disk_offload_index = None
return disk_offload_index, cpu_offload_index
```

---

### 14. pretrain_sora.py MSA 补丁

**文件**：`${MINDSPEED_MM_PATH}/pretrain_sora.py`

**修改内容**：

1. **msa_patch 导入**：在导入 torch 前导入 msa_patch，修复 `torch.serialization.safe_load_file` 缺失问题

2. **_load_state_dict_into_meta_model 包装器**：添加补丁确保当 `disk_offload_folder` 为 None 时返回的 index 也为 None

```python
# MSA fix: patch _load_state_dict_into_meta_model to handle None disk_offload_folder
try:
    import transformers.modeling_utils
    _original_func = transformers.modeling_utils._load_state_dict_into_meta_model

    def _patched_load_state_dict_into_meta_model(...):
        result = _original_func(...)
        if disk_offload_folder is None and result is not None:
            return None, result[1] if len(result) > 1 else None
        return result

    transformers.modeling_utils._load_state_dict_into_meta_model = _patched_load_state_dict_into_meta_model
except Exception as e:
    print(f"[MSA_PATCH] Warning: {e}", flush=True)
```

**注意**：此修改会影响所有使用 pretrain_sora.py 的模型（CogVideoX 等）

---

## 调试技巧

### 快速调试：使用单轮短步长配置

在调试 MSA 问题时，不需要修改代码注释 PTA，只需修改 `config.json` 缩短流程：

```json
{
  "task_type": 6,
  "tasks": {
    "6": {
      "MODEL_NAME": "cogvideox",
      "TOTAL_ITER": 1,
      "TRAIN_ITER": 1,
      "COMPARE_MODE": "pta_msa"
    }
  }
}
```

通过 `TOTAL_ITER=1` 和 `TRAIN_ITER=1` 可大幅缩短单轮调试时间，同时保持完整 PTA/MSA 流程不变。

---

### 1. 手动执行脚本测试

```bash
export MINDSPEED_MM_PATH=<YOUR_MINDSPEED_MM_PATH>  # 指向 workspace root，自动推导 MindSpeed-MM 子目录
export MM_MODEL=assets/mm_configs/inference_qwen2_5_vl_7b.json
bash scripts/runtime/mm_pta_qwenvl.sh
echo "EXIT CODE: $?"
```

### 2. 查看详细日志

```bash
# PTA日志
tail -50 pta_logs/inference_*.log

# MSA日志
tail -50 msa_logs/train_*.log
```

### 3. 检查返回码

```bash
bash scripts/runtime/mm_pta_qwenvl.sh
echo "Exit code: $?"
```

### 4. 环境检查

```bash
# 检查conda环境
conda env list | grep mm-

# 检查CANN环境
echo $ASCEND_HOME_PATH
ls /usr/local/Ascend/cann/

# 检查模型配置文件
ls $MINDSPEED_MM_PATH/examples/internvl3/
ls $MINDSPEED_MM_PATH/scripts-ms/
```

---

## 历史修改记录

### 2026-04-09 CogVideoX dtype 格式转换修复

**问题**：CogVideoX 配置中使用 dtype 简写格式（"bf16"），导致 transformers 库的 `dict_dtype_to_str` 函数报错

**解决方案**：在 `prepare_mm_config.sh` 中添加 dtype 格式转换

**修改文件**：`scripts/runtime/prepare_mm_config.sh`

**转换逻辑**：
```bash
# 转换 dtype 简写格式为标准格式
sed -i 's/"dtype":[[:space:]]*"bf16"/"dtype": "bfloat16"/g' "$OUTPUT_CONFIG"
sed -i 's/"dtype":[[:space:]]*"fp16"/"dtype": "float16"/g' "$OUTPUT_CONFIG"
sed -i 's/"dtype":[[:space:]]*"fp32"/"dtype": "float32"/g' "$OUTPUT_CONFIG"
```

**效果**：所有模型配置在预处理阶段自动转换 dtype 格式，无需手动修改 JSON 文件

---

### 2026-04-09 变异参数池配置外置

**修改**：将`mm_mutation_system.py`中的`mutable_params_pool`字典提取到YAML文件

**文件**：
- 新增：`$(dirname ${MINDSPEED_MM_PATH})/net_mutation/mutable_params_pool.yaml`
- 修改：`$(dirname ${MINDSPEED_MM_PATH})/net_mutation/mm_mutation_system.py`

**目的**：Task不需要直接修改Python文件，只需修改YAML即可配置变异参数

**加载逻辑**：
```python
# 从YAML加载变异参数池配置
def _load_mutable_params_pool(self) -> Dict[str, Any]:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pool_config_path = os.path.join(current_dir, 'mutable_params_pool.yaml')
    # ... 加载并转换为内部格式
```

---

### 2026-04-11 Task6 变异参数池改为YAML配置

**修改**：将`mm_mutator.py`中的内置字典改为从YAML文件加载

**文件**：
- 新增：`mutable_params_pool.yaml`
- 修改：`utils/runtime/mm_mutation/mm_mutator.py`

**目的**：Task6不需要修改Python代码，直接修改YAML即可配置变异参数

**配置文件位置**：
```
mutable_params_pool.yaml
```

**文件格式**：
```yaml
# 数值型参数 (范围变异)
numeric_params:
  mlp_ratio:
    min_val: 2.0
    max_val: 8.0
    min_factor: 0.7
    max_factor: 1.5
  # ...

# 枚举型参数 (离散值变异)
enum_params:
  learnable_pos_embed:
    - true
    - false
  # ...
```

**自定义配置路径**：
可通过环境变量指定自定义配置文件：
```bash
export MUTABLE_PARAMS_POOL_PATH=/path/to/your/mutable_params_pool.yaml
```

**加载优先级**：
1. 环境变量 `MUTABLE_PARAMS_POOL_PATH`
2. 默认路径 `lmsv_rec/mutable_params_pool.yaml`
3. 内置默认配置（后备）

---

### 2026-04-14 InternVL3 成功测试案例

**状态**：✅ PTA成功，✅ MSA成功，⚠️ 检测到20.18%精度差异

**结果**：
- PTA Loss: 10.14411, Memory: — MB, Time: 192 ms
- MSA Loss: 12.19148, Memory: 32805.31 MB, Time: 128 ms
- Loss差异: 20.18% (超过1%阈值，属于框架级实现差异)

**记录**：完整结果保存到`detected_bugs/internvl3_pta_msa_precision_diff/`

---

### 2026-04-10 CogVideoX dtype 解析修复（环境修复）

**问题**：transformers `dict_dtype_to_str` 无法处理 "bf16" 简写格式

**根因**：`str(d["dtype"]).split(".")[1]` 假设格式为 "torch.float32"，但 MindSpeed-MM 使用 "bf16"

**修复位置**：`/root/anaconda3/envs/msa-m/lib/python3.10/site-packages/transformers/configuration_utils.py` (第1063-1064行)

**修复内容**：
```python
# 修复前
elif not isinstance(d["dtype"], (str, int)):
    d["dtype"] = str(d["dtype"]).split(".")[1]

# 修复后
elif not isinstance(d["dtype"], (str, int)):
    dtype_str = str(d["dtype"])
    if "." in dtype_str:
        d["dtype"] = dtype_str.split(".")[1]
    else:
        d["dtype"] = dtype_str
```

**文档**：详细修复文档保存在 `detected_bugs/cogvideox_dtype_parsing/fix_documentation.md`

---


### 2026-04-09 OpenSora MSA修复

**问题**：`msadapter.UntypedStorage`缺失导致safetensors加载失败

**解决**：通过`sitecustomize.py`机制注入FakeUntypedStorage

**文件**：`scripts/runtime/mm_msa_opensora.sh`

---

### 2026-04-09 QwenVL Task6修复

**问题1**：PTA脚本因grep返回码1导致退出
- 文件：`scripts/runtime/pta_qwenvl_7b_real.sh`
- 修复：为所有grep命令添加`2>/dev/null || true`保护

**问题2**：权重格式转换
- 解决方案：使用`init_from_hf_path`直接加载HF格式权重
- 无需手动转换safetensors到MindSpeed-MM格式

---

### 2026-04-13 相对路径与 MINDSPEED_MM_PATH 修复

**问题1**：Task6 绝对路径硬编码导致在其他环境中无法运行
- 修复：所有默认路径改为相对路径
  - `MINDSPEED_MM_PATH` 默认：`<lm-sv-root>/../mindspeed-mm`（指向 workspace root，自动推导 MindSpeed-MM 子目录）
  - `LMSV_OUTPATH` 默认：`output`（相对于项目根目录）
- 说明：Shell 脚本内部会在执行前通过 `$(cd "$PATH" && pwd)` 将相对路径转换为绝对路径

**问题2**：PTA 真实执行脚本未 `export MINDSPEED_MM_PATH`，导致 wrapper 脚本找不到入口文件
- 影响文件：
  - `scripts/runtime/pta_qwenvl_7b_real.sh`
  - `scripts/runtime/pta_cogvideox_real.sh`
  - `scripts/runtime/pta_internvl3_8B_real.sh`
  - `scripts/runtime/pta_opensora_real.sh`
- 修复：在设置 `MINDSPEED_MM_PATH` 后添加 `export MINDSPEED_MM_PATH`

### 2026-04-13 Task6 配置架构重构（与 Task1-5 统一）

**变更**：Task6 的调用方式、参数来源、路径配置全面与 Task1-5 对齐。

**具体修改**：
- **移除 `TASK6_*` 环境变量**：`TASK6_TOTAL_ITER`、`TASK6_TRAIN_ITERS`、`TASK6_MODEL_NAME`、`TASK6_COMPARE_MODE` 等全部废弃
- **统一配置入口**：所有任务参数通过 `config.json` 的 `tasks["6"]` 字典传入，由 `do.py` → `protect.task(6, params)` → `task6.main(params)` 传递
- **`SAVE_STEPS` 更名为 `TRAIN_ITER`**：Task6 统一使用 `TRAIN_ITER` 指定每轮训练/推理步数，保留对旧版 `SAVE_STEPS` / `TRAIN_ITERS` 的兼容回退
- **统一路径变量**：Task6 统一使用 `MINDSPEED_MM_PATH`，不再拆分 `PTA_PATH` / `MSA_PATH`；未设置时兼容回退到 `PTA_PATH`
- **合并配置文件**：`config_task6_qwenvl.json` 和 `config_task6_test.json` 合并为统一的 `config.json`
- **`BASE_SEED` 生效**：`mm_mutator.py` 支持按轮次接收 `seed` 参数，每轮使用 `BASE_SEED + iter_num - 1` 作为随机种子
- **移除 `if __name__ == "__main__"`**：`task6.py` 只能通过 `protect.task(6, params)` 调用，不能独立执行

**影响文件**：
- `utils/task/task6.py`
- `do.py`
- `genconf.py`
- `utils/runtime/mm_mutation/mutate_graph.py`
- `config.json`

---

### 2026-04-09 MSA Pipeline Parallel修复

**问题**：PP>1配置下loss只在最后一个worker日志中输出

**修复**：
1. 修改等待循环，持续查找所有worker日志
2. 支持科学计数法格式的loss值
3. 分离loss和memory的提取逻辑

**涉及文件**：`msa_internvl3_8B_real.sh`、`msa_cogvideox_real.sh`

---

### 2026-04-08 路径迁移

**变更**：Task6 默认路径改为相对路径 `../mm-new`

**影响文件**：
- `config.json`中的`MINDSPEED_MM_PATH`（兼容旧版`PTA_PATH`/`MSA_PATH`）
- 所有脚本中的路径引用

**说明**：
- 默认 `MINDSPEED_MM_PATH` 为 `<lm-sv-root>/../mindspeed-mm`（指向 workspace root，自动推导 MindSpeed-MM 子目录）
- `LMSV_OUTPATH` 默认相对于项目根目录为 `output`
- 脚本内部会在执行前自动将相对路径转换为绝对路径

---

## 开发上下文

### Task6与Task1的关系

Task6基于Task1架构，专门针对多模态模型进行适配：

| 特性 | Task1 | Task6 |
|------|-------|-------|
| 目标 | 单模态模型 | 多模态模型 |
| 变异 | 参数变异 | 配置变异 |
| 对比 | PTA vs MF/MSA | PTA vs MSA |
| 报告 | task1_report.* | task6_report.* |

### 模型配置要点

#### QwenVL2.5推理配置

关键参数：
- `pipeline_class`: "Qwen2VlPipeline"
- `init_from_hf_path`: 指向HuggingFace格式权重路径
- 使用此参数时不需要`--load`参数

#### 推理模式vs训练模式

| 特性 | 训练模式 | 推理模式 |
|------|----------|----------|
| data_config | 需要 | 不需要 |
| loss输出 | 有 | 无 |
| 成功判断 | returncode==0且有loss | returncode==0 |
| 指标提取 | loss, memory, time | 无或只有time |

### 真实执行要求

**Task6必须真实执行模型训练/推理，不允许mock或模拟**。

**真实执行要求**：
1. 必须调用真实的训练脚本（`*_real.sh`）
2. 必须执行`torchrun pretrain_vlm.py`或`msrun pretrain_vlm.py`
3. 必须从训练日志中提取真实指标
4. 不允许输出模拟数据

**指标来源**：
- Loss: 从训练日志`grep "loss:"`提取
- 显存: 从训练日志`grep "NPU memory"`提取
- 时间: 从训练日志`grep "elapsed time per iteration"`提取

**禁止**：
- 输出模拟数据
- 随机生成指标
- 固定值输出
- 训练失败时使用默认值

---

## 参考文档

- `docs/task6.md` - 最终交付文档
- `docs/PTA_MF_PRECISION_ALIGNMENT.md` - 环境配置参考
- `docs/how-to-add-a-new-task.md` - 任务扩展指南
- `docs/TASK1_MODEL_EXTENSION.md` - Task1扩展参考

---

## 当前进度与待办事项（2026-04-10）

### 已完成

| 任务 | 状态 | 说明 |
|------|------|------|
| InternVL3 PTA/MSA | ✅ | 完整流程已跑通，作为基准 |
| QwenVL PTA | ✅ | 推理模式正常 |
| QwenVL MSA | ❌ | InnerInplaceIndexPut shape mismatch |
| CogVideoX PTA | ✅ | 训练模式正常，loss≈1.0 |
| CogVideoX MSA | ✅ | 经环境修复后已可正常运行 |
| OpenSora PTA | ✅ | 推理模式正常 |
| OpenSora MSA | ❌ | UntypedStorage错误 |
| CogVideoX dtype 修复 | ✅ | 修复 transformers 的 dtype 解析问题 |
| CogVideoX 签名修复 | ✅ | 修复 `_load_state_dict_into_meta_model` 签名不匹配 |
| InternVL3 回归测试 | ✅ | 验证了修改对其他模型无负面影响 |

### 进行中

| 任务 | 状态 | 说明 |
|------|------|------|
| QwenVL MSA | ❌ | InnerInplaceIndexPut shape mismatch - 已记录为框架Bug |
| OpenSora MSA | ❌ | UntypedStorage 错误 - 已记录为框架Bug |

### 待修复问题

1. **QwenVL MSA 执行**
   - 问题：`ValueError: For 'InnerInplaceIndexPut', shape mismatch...`
   - 状态：已确认为 MSA 框架 `index_put_` 语义与 MindSpore 算子不兼容
   - 记录：`detected_bugs/qwenvl_msa_inner_inplace_indexput/`

2. **OpenSora MSA 执行**
   - 问题：`TypeError: 'UntypedStorage' object is not callable`
   - 状态：已确认为 MSA 框架 safetensors 兼容性问题
   - 记录：`detected_bugs/opensora_msa/`

### 修改文件汇总

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `msadapter` | 修改 | bfloat16 fallback、numpy 索引自动转换 |
| `transformers` | 修改 | `_load_state_dict_into_meta_model` 关键字参数修复 |
| `lmsv_rec/scripts/runtime/msa_cogvideox_real.sh` | 修改 | 超时等待次数延长至 360 |
| `lmsv_rec/utils/task/task6.py` | 恢复 | PTA 验证部分已恢复（之前注释用于调试） |

---

---

## InternVL3 回归验证检查点清单（2026-04-21）

在路径迁移到 `/data2/lm-sv` 后，对 InternVL3 执行了完整回归验证。以下检查点应在每次重大变更后重复执行。

### 检查点1：前后轮次配置参数必须不同

**验证方法**：对比相邻轮次的 `mutation_gen{N}.json`

**通过标准**：至少有一个参数值不同（不允许完全相同）

**实际结果**：
- iter1 vs iter2 差异字段：
  - `text_decoder.attention_softmax_in_fp32`: `False` → `True`
  - `text_decoder.parallel_output`: `True` → `False`
- ✅ 通过

**如果失败**：检查 `mm_mutator.py` 的种子生成逻辑，确保每轮使用不同种子（`BASE_SEED + iter_num - 1`）

---

### 检查点2：日志必须包含有意义的值

**验证方法**：检查 `runtime_logs/pta_verify_iter{N}.log` 和 `msa_verify_iter{N}.log`

**通过标准**：
- 训练模型必须有 `loss` 值（且为合理数值，非 NaN/Inf）
- 必须有 `memory` 值（非 0 或 None）
- 必须有 `time` 值（非 0 或 None）

**实际结果**：
- PTA: loss=10.14385, memory=25660.95MB, time=192ms
- MSA: loss=12.19092, memory=32805.21MB, time=128ms
- ✅ 通过

**如果失败**：
- 检查日志正则表达式是否匹配当前模型输出格式
- 检查 MSA Pipeline Parallel 日志是否从正确 worker 读取（见第4节）

---

### 检查点3：报错信息必须完整记录

**验证方法**：检查 `status.json` 和 `metrics.json`

**通过标准**：
- `status.json` 必须包含 `components` 字段（MUTATE, PTA_VERIFY, MSA_VERIFY, ANALYSIS）
- `status.json` 必须包含 `reason` 字段（即使通过也应记录差异信息）
- `metrics.json` 必须包含 `analysis.issues` 列表
- 日志中不得有未捕获的 Traceback/RuntimeError/OSError

**实际结果**：
- `status.json`: 包含 `overall_status: PASS`, `reason: "Analysis issues: [...]"`, 完整的 `components`
- `metrics.json`: 包含 `pta`, `msa`, `analysis` 三部分
- 日志中 ERROR/Traceback/RuntimeError 计数为 0
- ✅ 通过

**如果失败**：
- 检查 `task6.py` 的异常捕获是否完整
- 检查 `runtime_helpers.run_shell_to_file` 是否正确捕获 stderr

---

### 检查点4：权重必须正确保存

**验证方法**：检查 `iters/iter_{N}/weights/{pta,msa}/`

**通过标准**：
- 必须存在 `iter_0000002/` 子目录（或对应训练步数的目录）
- 必须包含 `mp_rank_*` 子目录
- 必须包含 `model_optim_rng.pt` 文件（大小 > 1GB）
- PTA 和 MSA 都必须有权重

**实际结果**：
- iter_1: pta=15GB, msa=15GB
- iter_2: pta=15GB, msa=15GB
- 每轮都有完整的 `iter_0000002/mp_rank_00_00{0-3}/model_optim_rng.pt`
- ✅ 通过

**如果失败**：
- 检查 `--save-interval` 是否设为 `${TRAIN_ITERS:-1}`
- 检查 `SAVE_PATH` 环境变量是否正确传入
- 检查 Megatron 的 `--no-save-optim --no-save-rng` 是否误删了权重文件本身

---

### 检查点5：msrun_log 必须归档

**验证方法**：检查 `iters/iter_{N}/msrun_log/`

**通过标准**：
- 必须包含 `scheduler.log`
- 必须包含 `worker_*.log`（数量与并行度匹配）
- 文件大小 > 0（非空日志）

**实际结果**：
- iter_1: scheduler.log + worker_0~7.log，全部非空
- iter_2: scheduler.log + worker_0~7.log，全部非空
- ✅ 通过

---

### 检查点6：有效突变率统计

**验证方法**：查看最终统计输出

**通过标准**：
- `PTA成功率` ≥ 80%（允许少数参数组合导致崩溃）
- `MSA成功率` ≥ 80%
- 如果低于 80%，需要分析失败模式并考虑在突变池中加入约束

**实际结果（2轮）**：
- 总迭代=2，成功=2
- PTA成功率=100.0%，MSA成功率=100.0%
- ✅ 通过

**提高有效突变率的策略**：
1. **分析失败模式**：收集所有失败轮次的 `mutation_genX.json`，找出共同特征
2. **加入参数约束**：在 `mm_mutator.py` 的约束检查中，排除已知会导致崩溃的参数组合
3. **调整变异范围**：缩小某些参数的 `min_factor`/`max_factor`，避免极端值
4. **添加黑名单**：将反复导致失败的参数从 `mutable_params_pool.yaml` 中移除或限制枚举范围

---

### 2026-04-26 Task6 多机推理模型Bug修复

**问题1**: `cannot access local variable 're'`
- **根因**: `main()` 函数内部 `if not pta_success:` 分支中有 `import re`，使 `re` 变为局部变量。当PTA成功时此import不执行，后续MSA失败处理使用 `re.findall` 触发 `UnboundLocalError`
- **修复**: 删除函数内部的 `import re`（模块级import已足够）
- **原则**: 绝不在函数内部的条件分支中import模块

**问题2**: MSA bug信息被"读取日志异常"掩盖
- **根因**: MSA失败后 `main()` 重复解析日志，日志格式复杂时提取失败
- **修复**: 优先使用 `msa_metrics.get("error_info")`（`run_msa_verify` 已提取的真实错误），仅当为空时才降级解析日志

**问题3**: 推理模型错误显示loss=0.0
- **根因**: 脚本fallback输出 `loss: 0.0`，被 `_parse_pta_log` 提取为有效loss
- **修复**: `run_pta_verify` / `run_msa_verify` 返回前，若 `type == "inference"` 则将 `metrics["loss"]` 重置为 `None`

**问题4**: qwen MSA失败后时间/显存N/A
- **根因**: `msa_qwenvl_7b_real.sh` 中 `set -e` 导致msrun失败时脚本立即退出，未执行到指标输出
- **修复**: msrun前后添加 `set +e` / `set -e`，将指标输出移到 `exit 1` 之前

**问题5**: 多机推理模型PTA被误判为失败
- **根因**: `run_pta_verify_multinode` 要求 `loss is not None` 才算成功，但推理模型无loss
- **修复**: 推理模型只需 `memory is not None` 或 `time is not None` 即返回本地成功，不受远程节点失败影响

---

**文档版本**: 1.4
**更新日期**: 2026-04-26
**适用范围**: Task6开发维护
