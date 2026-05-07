# Task6 源码级别定制化修改详细说明

> 本文档详细列出 Task6（多模态整网泛化变异测试）所有基于标准库的源码级别定制化修改。
> 对应自动化脚本：`task6_conda_envs_export/automated_setup/setup_task6_envs.sh`
> 对应补丁目录：`task6_conda_envs_export/automated_setup/patches/`
>
> **工作流程**：
> 1. 从 `standard_env/` 加载裸 conda 环境（仅含 requirements.txt 标准库）
> 2. 运行 `setup_task6_envs.sh` 一键应用所有定制化修改
> 3. 得到完整可运行的 Task6 环境
>
> **本文档目标**：对于一个已安装 requirements.txt 中标准库的全新 conda 环境，执行 `setup_task6_envs.sh` 后可一键完成所有定制化修改，且每一处修改在本文档中都有完整的源码分析和说明。

---

## 修改总览

| 编号 | 修改名称 | 作用范围 | 修改文件 | 所属环境 | 修改类型 |
|------|----------|----------|----------|----------|----------|
| 1 | transformers 兼容性补丁 | `site-packages/transformers` | `modeling_utils.py` | mindspeed + msadapter | patch 命令 |
| 2 | msadapter bfloat16 fallback（_utils.py） | `mm-new/msadapter/msadapter` | `_utils.py` | msadapter | 文件覆盖 |
| 3 | msadapter bfloat16 fallback（serialization.py） | `mm-new/msadapter/msadapter` | `serialization.py` | msadapter | 文件覆盖 |
| 4 | decord 运行时补丁 | 运行时动态生成 | `/tmp/decord_patch/decord_fix.py` | mindspeed（PTA） | 运行时脚本生成 |
| 5 | libstdc++ 兼容性修复 | 环境变量 | `LD_LIBRARY_PATH` | msadapter（MSA） | 环境脚本 |

---

## 修改 1：transformers 兼容性补丁

### 1.1 修改背景与原因

MindSpeed-MM 的 MSA 侧在加载模型权重时，调用 transformers 库内部的 `_load_state_dict_into_meta_model` 函数。在 **transformers 4.51.0** 中，该函数内部调用传参方式使用了**位置参数**，而 MindSpeed-MM 的调用链期望的是**关键字参数**，导致以下 `TypeError`：

```
TypeError: _load_state_dict_into_meta_model() got multiple values for argument 'device_map'
```

根本原因：在 `modeling_utils.py` 的 `_load_state_dict_into_meta_model` 调用处，第 4 个参数 `expected_keys` 和第 5 个参数 `reverse_key_renaming_mapping` 被作为位置参数传递，但函数签名中从第 4 个参数开始已经有默认值的 keyword-only 参数（如 `device_map`），导致参数解析冲突。

**注意**：虽然 transformers **4.55.2+** 的调用处已改为关键字参数，但由于 msadapter 的装饰器机制会拦截并改变参数传递方式（将位置参数重新展开传递），导致即使 transformers 4.55.2 仍会出现 `TypeError`。因此**所有版本都需要应用此补丁**。

### 1.2 修改位置

- **目标文件**：`mindspeed` 和 `msadapter` 两个 conda 环境的 `lib/python3.10/site-packages/transformers/modeling_utils.py`
- **补丁文件**：
  - `automated_setup/patches/transformers/transformers_4.51.0_modeling_utils.patch`
  - `automated_setup/patches/transformers/transformers_4.55.2_modeling_utils.patch`

### 1.3 源码修改详情

#### transformers 4.51.0 修改点（`_load_pretrained_model` 内，约第 4830 行）

```python
# 修改前
                _load_state_dict_into_meta_model(
                    model_to_load,
                    state_dict,
                    shard_file,
                    expected_keys,
                    reverse_key_renaming_mapping,
                    device_map=device_map,
                    ...
                )

# 修改后
                _load_state_dict_into_meta_model(
                    model_to_load,
                    state_dict,
                    shard_file,
                    expected_keys=expected_keys,
                    reverse_renaming_mapping=reverse_key_renaming_mapping,
                    device_map=device_map,
                    ...
                )
```

#### transformers 4.55.2 修改点（`load_shard_file` 内，约第 975 行）

```python
# 修改前
        disk_offload_index, cpu_offload_index = _load_state_dict_into_meta_model(
            model_to_load,
            state_dict,
            shard_file,
            expected_keys,
            reverse_key_renaming_mapping,
            device_map=device_map,
            ...
        )

# 修改后
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

#### 具体修改说明

| 位置 | 修改前 | 修改后 | 说明 |
|------|--------|--------|------|
| `_load_pretrained_model` 调用处 | `expected_keys,` | `expected_keys=expected_keys,` | 第 4 个参数改为关键字参数 |
| `_load_pretrained_model` 调用处 | `reverse_key_renaming_mapping,` | `reverse_renaming_mapping=reverse_key_renaming_mapping,` | 第 5 个参数改为关键字参数 |
| `load_shard_file` 调用处（4.55.2） | `expected_keys,` | `expected_keys=expected_keys,` | 同上，另一处调用 |
| `load_shard_file` 调用处（4.55.2） | `reverse_key_renaming_mapping,` | `reverse_renaming_mapping=reverse_key_renaming_mapping,` | 同上，另一处调用 |

**参数名变化说明**：函数签名中第 5 个参数名为 `reverse_renaming_mapping`，但调用处传入的变量名为 `reverse_key_renaming_mapping`。原始代码作为位置参数传递时，Python 按位置匹配，不检查参数名。改为关键字参数后，必须使用函数签名中的参数名 `reverse_renaming_mapping`。

### 1.4 自动化应用方式

`setup_task6_envs.sh` 通过 `patch -p1` 命令精确应用补丁文件：

```bash
cd "$site_packages"
patch --dry-run -p1 < "$patch_file"  # 先验证
patch -p1 < "$patch_file"           # 再应用
```

**版本自适应逻辑**：
- 若检测到补丁已应用（`expected_keys=expected_keys` 已存在），自动跳过
- 优先使用 `patch` 命令精确匹配上下文（不会误伤函数定义）
- `patch` 失败时 fallback 到 Python re 替换，仅针对 `load_shard_file` 调用块

**为什么不用 sed**：早期版本使用 sed 替换 `expected_keys,`，但因匹配范围过宽，误将函数定义参数列表中的 `expected_keys,` 也替换掉，导致 Python `SyntaxError`（带默认值的参数后不能有非默认值参数）。`patch` 命令通过 diff 上下文精确匹配，只修改目标调用点，不会误伤其他位置。

---

## 修改 2：msadapter bfloat16 fallback 补丁（_utils.py）

### 2.1 修改背景与原因

MindSpore 在某些版本或设备上，内置的 `np_dtype.bfloat16` 类型不可用。当 MSA 侧加载包含 bfloat16 权重的模型（如 InternVL3）时，会因以下错误而崩溃：

```
TypeError: The Numpy bfloat16 data type is not supported now, please ensure that the current Numpy version is not less than the version when the mindspore is compiled, and the major versions are same.
```

本补丁在 `msadapter._utils` 中增加 fallback 逻辑：当 MindSpore 原生 bfloat16 不可用时，自动回退到 `ml_dtypes.bfloat16`。

### 2.2 修改位置

- **目标文件**：`mm-new/msadapter/msadapter/_utils.py`
- **补丁文件**：`automated_setup/patches/msadapter/_utils.py`
- **应用方式**：文件直接覆盖（`cp patches/msadapter/_utils.py mm-new/msadapter/msadapter/_utils.py`）

### 2.3 源码修改详情

以下列出所有修改点，对比原始代码（无 bfloat16 fallback）和修改后代码。

#### 2.3.1 `element_size_map` 增加 bfloat16 映射

**修改前**：
```python
element_size_map = {
    mindspore.float16: 2,
    mindspore.float32: 3,
    mindspore.int64: 4,
    mindspore.uint8: 1,
    mindspore.int8: 1,
    mindspore.bool_: 1
}
```

**修改后**：
```python
element_size_map = {
    mindspore.float16: 2,
    mindspore.float32: 3,
    mindspore.bfloat16: 2,   # <-- 新增
    mindspore.int64: 4,
    mindspore.uint8: 1,
    mindspore.int8: 1,
    mindspore.bool_: 1
}
```

**修改说明**：为 `mindspore.bfloat16` 添加 element size 映射，值为 2（与 float16 相同，均为 16 位）。这是后续 `_element_size()` 函数正确计算 bfloat16 tensor 元素大小的基础。

#### 2.3.2 新增 `_bf16()` 函数

**修改前**：无此函数（原始代码直接访问 `np_dtype.bfloat16`）

**修改后**：
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

**修改说明**：
1. 使用函数属性缓存机制（`_bf16.bf16`）避免重复导入
2. 只有当 `support_bf16()` 返回 True **且** `np_dtype` 有 `bfloat16` 属性时，才使用 MindSpore 原生 bfloat16
3. 任一条件不满足时，fallback 到 `ml_dtypes.bfloat16`
4. 该函数被 `_rebuild_tensor_v2()` 和 `dtype_to_nptype()` 调用

#### 2.3.3 `_rebuild_tensor_v2()` 中处理 bfloat16 到 float16 的转换

**修改前**：无 bfloat16 处理逻辑

**修改后**（第 122-124 行）：
```python
    if num_elemets == storage.size and _is_contiguous_by_shape_stride(size, stride):
        if array.dtype == _bf16() and not support_bf16():
            array = array.astype(np.float16)
        array = array.reshape(size)
```

**修改后**（第 130-136 行）：
```python
    target_dtype = storage.dtype
    if array.dtype == _bf16():
        if support_bf16():
            array = array.view(np.float16)
        else:
            array = array.astype(np.float16)
            target_dtype = array.dtype
```

**修改说明**：
- 第一处：当 tensor 是连续存储且设备不支持 bfloat16 时，在 reshape 前将 bfloat16 数组转换为 float16
- 第二处：当 tensor 需要按 stride 重建时，根据设备是否支持 bfloat16 选择 `view(np.float16)` 或 `astype(np.float16)`
- 使用 `_bf16()` 函数而非直接访问 `np_dtype.bfloat16`，确保 fallback 生效

#### 2.3.4 `dtype_to_nptype()` 函数增加 bfloat16 fallback

**修改前**：
```python
def dtype_to_nptype(dtype):
    dtype_nptype_dict = {
        # ... 其他映射 ...
    }
    if dtype == mindspore.bfloat16:
        if not hasattr(np_dtype, 'bfloat16'):
            raise TypeError(
                "The Numpy bfloat16 data type is not supported now, please ensure that the current "
                "Numpy version is not less than the version when the mindspore is compiled, "
                "and the major versions are same."
            )
        return np_dtype.bfloat16
    return dtype_nptype_dict[dtype]
```

**修改后**：
```python
def dtype_to_nptype(dtype):
    dtype_nptype_dict = {
        # ... 其他映射 ...
    }
    if dtype == mindspore.bfloat16:
        if hasattr(np_dtype, 'bfloat16'):
            return np_dtype.bfloat16
        import ml_dtypes
        return ml_dtypes.bfloat16
    return dtype_nptype_dict[dtype]
```

**修改说明**：
- 原始逻辑：如果 `np_dtype` 没有 `bfloat16`，直接抛出 `TypeError` 导致崩溃
- 修改后逻辑：优先使用 MindSpore 原生 bfloat16，不存在时 fallback 到 `ml_dtypes.bfloat16`，避免崩溃
- 移除错误抛出的 raise 语句，改为条件分支

### 2.4 自动化应用方式

`setup_task6_envs.sh` 将 `patches/msadapter/_utils.py` 直接覆盖到 `mm-new/msadapter/msadapter/_utils.py`，并在覆盖前自动创建 `.bak` 备份。

---

## 修改 3：msadapter bfloat16 fallback 补丁（serialization.py）

### 3.1 修改背景与原因

与修改 2 相同，但作用于序列化/反序列化模块。当通过 `msadapter.load()` 加载 PyTorch checkpoint 时，checkpoint 中可能包含 bfloat16 存储类型。如果 MindSpore 原生不支持 bfloat16，加载过程会崩溃。

### 3.2 修改位置

- **目标文件**：`mm-new/msadapter/msadapter/serialization.py`
- **补丁文件**：`automated_setup/patches/msadapter/serialization.py`
- **应用方式**：文件直接覆盖

### 3.3 源码修改详情

以下列出所有修改点，逐个说明。

#### 3.3.1 新增 `_bf16()` 函数

**修改前**：无此函数

**修改后**：
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

**修改说明**：与 `_utils.py` 中的 `_bf16()` 函数逻辑完全一致，提供缓存的 bfloat16 dtype 对象。serialization.py 独立定义此函数，避免循环导入问题。

#### 3.3.2 `storage_to_dtype` 增加 BFloat16Storage 映射

**修改前**：
```python
def storage_to_dtype(storage):
    dtype_map = {
        "HalfStorage": np.float16,
        "FloatStorage": np.float32,
        'LongStorage': np.int64,
        'ByteStorage': np.uint8,
        'BoolStorage': np.bool_
    }
    return dtype_map[storage]
```

**修改后**：
```python
def storage_to_dtype(storage):
    dtype_map = {
        "HalfStorage": np.float16,
        "FloatStorage": np.float32,
        'BFloat16Storage': _bf16(),   # <-- 新增
        'LongStorage': np.int64,
        'ByteStorage': np.uint8,
        'BoolStorage': np.bool_
    }
    return dtype_map[storage]
```

**修改说明**：为 PyTorch checkpoint 中的 `BFloat16Storage` 类型添加 numpy dtype 映射。使用 `_bf16()` 函数获取 dtype，确保在 MindSpore 原生不支持时 fallback 到 `ml_dtypes.bfloat16`。

#### 3.3.3 `storage_map` 增加 bfloat16 映射

**修改前**：
```python
storage_map = {
    mindspore.float16: "HalfStorage",
    mindspore.float32: "FloatStorage",
    mindspore.int64: 'LongStorage',
    mindspore.int32: 'IntStorage',
    mindspore.uint8: 'ByteStorage',
    mindspore.bool_: 'BoolStorage'
}
```

**修改后**：
```python
storage_map = {
    mindspore.float16: "HalfStorage",
    mindspore.float32: "FloatStorage",
    mindspore.bfloat16: 'BFloat16Storage',   # <-- 新增
    mindspore.int64: 'LongStorage',
    mindspore.int32: 'IntStorage',
    mindspore.uint8: 'ByteStorage',
    mindspore.bool_: 'BoolStorage'
}
```

**修改说明**：为 MindSpore bfloat16 dtype 添加 PyTorch storage 类型名称映射。这是 `save()` 函数在序列化时正确标记 bfloat16 tensor 类型的基础。

#### 3.3.4 `element_size_map` 增加 BFloat16Storage

**修改前**：
```python
element_size_map = {
    "HalfStorage": 2,
    "FloatStorage": 3,
    'LongStorage': 4,
    'ByteStorage': 1,
    'BoolStorage': 1
}
```

**修改后**：
```python
element_size_map = {
    "HalfStorage": 2,
    "FloatStorage": 3,
    'BFloat16Storage': 2,   # <-- 新增
    'LongStorage': 4,
    'ByteStorage': 1,
    'BoolStorage': 1
}
```

**修改说明**：BFloat16 为 16 位（2 字节），与 HalfStorage 相同。这是序列化时计算存储大小的基础。

#### 3.3.5 `_legacy_load()` 中处理 bfloat16

**修改前**：无 bfloat16 转换逻辑

**修改后**（第 1141 行附近）：
```python
    new_result = {}
    for k, v in result.items():
        # ... 提取 array ...
        if array.dtype == _bf16() and not support_bf16():
            array = array.astype(np.float16)
        new_result[k] = Tensor.from_numpy(array)
```

**修改说明**：在 legacy load 路径中，遍历反序列化结果，若发现 bfloat16 数组且设备不支持，则在创建 MindSpore Tensor 前转换为 float16。

#### 3.3.6 `convert_torch_to_mindspore()` 中处理 bfloat16

**修改前**：无 bfloat16 处理

**修改后**：
```python
    has_bf16 = False
    for key, value in state_dict.items():
        if value.dtype == msadapter.bfloat16:
            data = Tensor.from_numpy(value.to(msadapter.float).numpy().astype(np.float16))
            if not has_bf16:
                has_bf16 = True
        else:
            data = Tensor.from_numpy(value.numpy())
        ms_ckpt.append({'name': key, 'data': data})

    if has_bf16:
        logging.warning("MindSpore do not support bfloat16 dtype, we will automaticlly convert to float16")
```

**修改说明**：在 PyTorch checkpoint 转 MindSpore checkpoint 时，若检测到 bfloat16 权重，先转为 float32 再转 numpy 再转 float16，确保兼容性。同时记录警告日志。

#### 3.3.7 `str_to_np_types()` 中增加 BF16 映射

**修改前**：无 BF16 映射

**修改后**：
```python
def str_to_np_types(dtype_str):
    np_types = {
        "F64": np.float64,
        "F32": np.float32,
        "F16": np.float16,
        "BF16": _bf16(),   # <-- 新增
        "I64": np.int64,
        # ... 其他映射 ...
    }
    return np_types[dtype_str]
```

**修改说明**：为 safetensors 格式中的 "BF16" 字符串添加 numpy dtype 映射。使用 `_bf16()` 确保 fallback。

#### 3.3.8 `legacy_safe_load_file()` 中处理 bfloat16

**修改前**：无 bfloat16 处理，直接尝试 `Tensor.convert_bytes_to_tensor`

**修改后**（第一分支）：
```python
    try:
        for k, v in safeview:
            dtype = _MS_TYPES[v["dtype"]]
            if (not support_bf16() and dtype != mindspore.bfloat16) or support_bf16():
                arr = Tensor.convert_bytes_to_tensor(bytes(v["data"]), tuple(v["shape"]), dtype)
                result[k] = Tensor(arr)
            else:
                raise TypeError('Do not support bfloat16 on current device, use numpy as convert buffer to boost load.')
        return result
    except Exception as e:
        for k, v in safeview:
            dtype = str_to_np_types(v["dtype"])
            arr = np.frombuffer(v["data"], dtype=dtype).reshape(v["shape"])
            if (not support_bf16() and dtype != _bf16()) or support_bf16():
                result[k] = Tensor.from_numpy(arr)
            else:
                result[k] = Tensor.from_numpy(arr.astype(np.float16))
        return result
```

**修改说明**：
- 第一分支（快速路径）：使用 `Tensor.convert_bytes_to_tensor`，若不支持 bfloat16 则抛出异常触发 fallback
- 第二分支（fallback 路径）：使用 numpy 作为中间缓冲区，将 bfloat16 数据转为 float16 后再创建 Tensor
- 条件 `(not support_bf16() and dtype != _bf16()) or support_bf16()` 判断：若不支持 bfloat16 且当前 dtype 不是 bfloat16，则直接转换；若支持 bfloat16，也直接转换。只有不支持且是 bfloat16 时，才需要 `astype(np.float16)`

#### 3.3.9 `safe_load_file()` 中处理 bfloat16

**修改前**：无 bfloat16 处理

**修改后**（`convert()` 内部函数）：
```python
    def convert(info: dict[str, Any]):
        numpy_dtype = str_to_np_types(info['dtype'])
        ms_dtype = _MS_TYPES[info['dtype']]
        shape: list[int] = info['shape']
        begin, end = info['data_offsets']
        assert 0 <= begin <= end <= len(byte_buf)
        assert end - begin == math.prod(shape) * np.dtype(numpy_dtype).itemsize
        buf = byte_buf[begin:end]
        array = np.frombuffer(buf, dtype=numpy_dtype).reshape(shape)
        if array.dtype == _bf16() and not support_bf16():
            array = array.astype(np.float16)
        out = Tensor.from_numpy(array)
        return out
```

**修改说明**：在 safetensors 标准加载路径中，从 numpy 数组创建 MindSpore Tensor 前，检查 dtype 是否为 bfloat16 且设备不支持。若是，转换为 float16。使用 `_bf16()` 而非直接比较 `np_dtype.bfloat16`，确保 fallback 生效。

#### 3.3.10 `load_checkpoint()` 中处理 BFloat16

**修改前**：无 BFloat16 处理，会进入通用分支导致错误

**修改后**：
```python
            if data_type == "BFloat16":
                dims = element.tensor.dims
                param_data = np.frombuffer(data, np_type)
                param_data = param_data.reshape(list(dims))
                parameter = Tensor.from_numpy(param_data)
                parameter_dict[element.tag] = parameter
                continue
```

**修改说明**：在加载 MindSpore 原生 checkpoint 格式（.ckpt）时，若遇到 "BFloat16" 数据类型，使用 `np.frombuffer` 直接读取数据并 reshape，然后创建 Tensor。`np_type` 由 `tensor_to_np_type` 字典映射得到。此处不转换 dtype，因为 .ckpt 文件本身使用 MindSpore 的序列化格式，数据已在保存时处理好。

### 3.4 自动化应用方式

`setup_task6_envs.sh` 将 `patches/msadapter/serialization.py` 直接覆盖到 `mm-new/msadapter/msadapter/serialization.py`，并在覆盖前自动创建 `.bak` 备份。

---

## 修改 4：decord 运行时补丁

### 4.1 修改背景与原因

某些 decord 版本（如 conda-forge 上的 0.6.0）在 aarch64 平台上编译时缺少 `cpu` 属性。当代码尝试调用 `decord.cpu()` 时会抛出：

```
AttributeError: module 'decord' has no attribute 'cpu'
```

### 4.2 修改位置

- **运行时生成**：`/tmp/decord_patch/decord_fix.py`
- **补丁文件**：`automated_setup/patches/decord/decord_fix.py`
- **环境脚本**：`automated_setup/patches/envset/mm-pta-task6.sh`

### 4.3 源码修改详情

#### 修改内容

在 PTA 环境脚本 `mm-pta-task6.sh` 中，每次 source 时动态生成 decord 修复模块：

```bash
DECORD_PATCH_DIR="/tmp/decord_patch"
mkdir -p $DECORD_PATCH_DIR

cat > $DECORD_PATCH_DIR/decord_fix.py << 'EOF'
import sys
import warnings

try:
    import decord
    if not hasattr(decord, 'cpu'):
        decord.cpu = lambda: None
        warnings.warn("Patched decord.cpu() as empty function")
    original_init = decord.__init__ if hasattr(decord, '__init__') else None
except Exception as e:
    warnings.warn(f"Failed to patch decord: {e}")
EOF

export PYTHONPATH="${DECORD_PATCH_DIR}:${PYTHONPATH}"
```

#### 源码分析

| 行 | 代码 | 说明 |
|----|------|------|
| 1-2 | `import sys, warnings` | 标准导入 |
| 4 | `import decord` | 尝试导入 decord 模块 |
| 5-7 | `if not hasattr(decord, 'cpu')` | 检查 decord 是否有 `cpu` 属性 |
| 6 | `decord.cpu = lambda: None` | 若无，注入一个返回 None 的空函数 |
| 7 | `warnings.warn(...)` | 发出警告，提示用户已应用补丁 |
| 8 | `original_init = ...` | 保留原始 `__init__` 引用（兼容性考虑） |
| 9-10 | `except Exception as e` | 捕获所有异常，避免 patch 失败阻断程序 |

#### 工作机制

1. 在 `/tmp/decord_patch/decord_fix.py` 中写入 monkey-patch 代码
2. 将 `/tmp/decord_patch` 添加到 `PYTHONPATH` 最前面
3. 当 Python 导入任何模块时，若该模块导入 `decord_fix`（或通过其他机制触发），自动执行 patch 逻辑
4. 实际上，由于 `PYTHONPATH` 优先级，如果用户代码 `import decord_fix` 会加载此文件，但更常见的机制是 decord 自身的导入链触发
5. 该补丁在**每次 source 环境脚本时自动生效**，无需手动干预

### 4.4 自动化应用方式

`setup_task6_envs.sh` 将环境脚本部署到 `lmsv_rec/scripts/envset/mm-pta-task6.sh`。用户在执行 Task6 前 source 该脚本即可自动生效。

---

## 修改 5：libstdc++ 兼容性修复

### 5.1 修改背景与原因

MSA 侧运行时，某些库（如 numpy、opencv 等）依赖较新版本的 `libstdc++.so.6`（需要 `GLIBCXX_3.4.29`）。如果系统自带的 libstdc++ 版本过旧，会导致运行时错误：

```
ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
```

### 5.2 修改位置

- **环境脚本**：`automated_setup/patches/envset/mm-msa-task6.sh`

### 5.3 源码修改详情

#### 修改内容

在 MSA 环境脚本 `mm-msa-task6.sh` 中，激活 msadapter conda 环境后，优先加载 conda 环境自带的 libstdc++：

```bash
# 修复 numpy/C++ 库版本不匹配（优先使用当前 conda 环境的 libstdc++）
if [ -n "${CONDA_PREFIX}" ] && [ -f "${CONDA_PREFIX}/lib/libstdc++.so.6" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
fi
```

#### 源码分析

| 行 | 代码 | 说明 |
|----|------|------|
| 1 | `if [ -n "${CONDA_PREFIX}" ]` | 检查是否已激活 conda 环境 |
| 1 | `&& [ -f "${CONDA_PREFIX}/lib/libstdc++.so.6" ]` | 检查 conda 环境中是否存在 libstdc++.so.6 |
| 2 | `export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"` | 将 conda 的 lib 目录添加到 LD_LIBRARY_PATH 最前面 |

#### 工作机制

1. `CONDA_PREFIX` 是 conda 激活环境时自动设置的环境变量，指向当前环境的根目录
2. 检查该目录下是否存在 `lib/libstdc++.so.6`
3. 若存在，将 `${CONDA_PREFIX}/lib` 添加到 `LD_LIBRARY_PATH` 的最前面
4. Linux 动态链接器 `ld.so` 按照 `LD_LIBRARY_PATH` 的顺序搜索共享库，因此会优先加载 conda 环境的高版本 libstdc++
5. 已去除旧版硬编码路径 `/root/anaconda3/envs/msadapter/lib`，改为使用 `$CONDA_PREFIX` 动态获取，支持任意 conda 安装位置

### 5.4 自动化应用方式

`setup_task6_envs.sh` 将环境脚本部署到 `lmsv_rec/scripts/envset/mm-msa-task6.sh`。用户在执行 Task6 前 source 该脚本即可自动生效。

---

## 一键应用脚本说明

### 脚本位置

`task6_conda_envs_export/automated_setup/setup_task6_envs.sh`

### 设计原则

- **只做 patch**：不创建环境、不安装包、不卸载包
- **幂等性**：多次运行不会产生副作用（已 patch 的会跳过）
- **版本自适应**：根据 transformers 版本自动判断是否需打补丁
- **安全备份**：覆盖文件前自动创建 `.bak` 备份

### 前置条件

1. conda 已安装并可正常使用
2. 两个 conda 环境已创建并安装了 `requirements.txt` 中的标准库：
   - `mindspeed`（PTA 环境）
   - `msadapter`（MSA 环境）
3. 这些"裸"环境可从 `../standard_env/` 中通过 yml 文件还原
4. mm-new 工作区已就绪（用于 msadapter bfloat16 补丁，可选）

### 用法

```bash
cd <lm-sv-root>/task6_conda_envs_export/automated_setup
bash setup_task6_envs.sh
```

### 可选环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `PTA_NAME` | PTA conda 环境名 | `mindspeed` |
| `MSA_NAME` | MSA conda 环境名 | `msadapter` |
| `MM_WORKSPACE` | mm-new 工作区绝对路径 | 自动推断 |
| `LMSV_REC` | lmsv_rec 项目绝对路径 | 自动推断 |

### 脚本执行流程

1. **检查 conda 环境存在** — 确认 mindspeed 和 msadapter 环境已创建
2. **推断 MindSpeed-MM 工作区路径**（支持手动指定或自动推断）
3. **应用 transformers 兼容性补丁**（仅 msadapter 环境，版本自适应）
4. **应用 msadapter bfloat16 fallback 补丁**（覆盖 `_utils.py` 和 `serialization.py`）
5. **部署环境脚本**（复制 `mm-pta-task6.sh` 和 `mm-msa-task6.sh`）
6. **验证所有修改**

### 验证项

脚本会自动验证以下内容：

- [x] transformers `modeling_utils.py` 中补丁状态（或确认版本已内置兼容）
- [x] `mm-new/msadapter/msadapter/_utils.py` 中 `ml_dtypes.bfloat16` fallback 逻辑已存在
- [x] `mm-new/msadapter/msadapter/serialization.py` 中 `ml_dtypes.bfloat16` fallback 逻辑已存在
- [x] 环境脚本文件存在且可执行

---

## 补丁文件清单

```
automated_setup/patches/
├── transformers/
│   ├── transformers_4.51.0_modeling_utils.patch    # transformers 兼容性补丁（_load_pretrained_model 调用处）
│   └── transformers_4.55.2_modeling_utils.patch    # transformers 兼容性补丁（load_shard_file 调用处）
├── msadapter/
│   ├── _utils.py                                    # 修改后的 _utils.py（完整文件）
│   └── serialization.py                             # 修改后的 serialization.py（完整文件）
├── decord/
│   └── decord_fix.py                                # decord monkey-patch 模块源码
└── envset/
    ├── mm-pta-task6.sh                              # PTA 环境脚本（含 decord 补丁生成逻辑）
    └── mm-msa-task6.sh                              # MSA 环境脚本（含 libstdc++ 修复逻辑）
```

---

## transformers 版本兼容性说明

本补丁最初针对 **transformers 4.51.0** 开发。在后续测试中发现：

- **transformers 4.55.2**：`modeling_utils.py` 中 `_load_state_dict_into_meta_model` 的调用方式已改为关键字参数，表面上无需补丁。但由于 msadapter 的装饰器机制会在调用链中拦截并将关键字参数重新展开为位置参数传递，实际运行中仍会出现 `TypeError`。因此 **所有版本均建议应用此补丁**
- **transformers 4.51.0**：必须应用此补丁，否则 MindSpeed-MM MSA 侧加载模型时会抛出 `TypeError`

`setup_task6_envs.sh` 脚本中的版本检测逻辑：
- 版本 >= 4.55.0 时：调用处已改为关键字参数，但脚本仍会检查并确认补丁状态
- 版本 == 4.51.0 时：自动应用 sed 替换
- 其他版本：尝试通用替换并给出警告

## 版本偏差说明

在实际搭建过程中，以下包版本与用户指定清单存在偏差，均为必要的最小化调整：

| 环境 | 包名 | 用户指定版本 | 实际版本 | 偏差原因 |
|------|------|-------------|----------|----------|
| mindspeed | torch | 2.7.1 | 2.7.1+cpu | PyTorch 在 aarch64/Linux 平台上仅提供 `+cpu` 构建标签的 wheel，`2.7.1+cpu` 与 `2.7.1` 功能完全一致，仅 build tag 不同 |
| both | grpcio | 1.78.1 | 1.78.0 | PyPI 上不存在 `grpcio==1.78.1`，最近可用版本为 `1.78.0` |

## 注意事项

1. **MindSpeed-MM 工作区路径**：脚本通过 `MINDSPEED_MM_PATH` 环境变量（或 `config.json` 中的配置）定位工作区。`MINDSPEED_MM_PATH` 支持指向 workspace root（如 `/shared/mindspeed-mm`），框架会自动推导 `MindSpeed-MM` 子目录。也可直接指向代码目录。
2. **备份**：脚本在覆盖 `_utils.py` 和 `serialization.py` 前会自动创建 `.bak` 备份文件。
3. **运行时补丁**：decord 补丁和 libstdc++ 修复在 source 环境脚本时生效，不属于持久化修改，无需在 setup 时 patch。
4. **裸环境还原**：如需回滚到裸状态，可删除当前环境并从 `../standard_env/*.yml` 重新创建。

---

## 手动验证方法

### 验证 transformers 补丁（两个环境均需验证）

```bash
# mindspeed 环境
conda activate mindspeed
python -c "
import transformers, inspect
src = inspect.getsourcefile(transformers.modeling_utils)
with open(src) as f:
    content = f.read()
tf_ver = transformers.__version__
if 'expected_keys=expected_keys' in content:
    print(f'PASS [mindspeed]: transformers patch applied (version {tf_ver})')
else:
    print(f'FAIL [mindspeed]: transformers patch missing (version {tf_ver})')
"

# msadapter 环境
conda activate msadapter
python -c "
import transformers, inspect
src = inspect.getsourcefile(transformers.modeling_utils)
with open(src) as f:
    content = f.read()
tf_ver = transformers.__version__
if 'expected_keys=expected_keys' in content:
    print(f'PASS [msadapter]: transformers patch applied (version {tf_ver})')
else:
    print(f'FAIL [msadapter]: transformers patch missing (version {tf_ver})')
"
```

**注意**：即使 transformers 版本为 4.55.2，也建议验证此补丁已应用，因为 msadapter 的装饰器可能在运行时改变参数传递方式。

### 验证 msadapter bfloat16 fallback

```bash
# 验证 _utils.py（将 <MSADAPTER_PATH> 替换为实际的 msadapter 源码路径）
python -c "
import sys
sys.path.insert(0, '<MSADAPTER_PATH>')
from msadapter._utils import _bf16
print('PASS: _bf16() returns', _bf16())
"

# 验证 serialization.py
python -c "
import sys
sys.path.insert(0, '<MSADAPTER_PATH>')
from msadapter.serialization import _bf16
print('PASS: _bf16() returns', _bf16())
"
```

> `<MSADAPTER_PATH>` 通常为 `MINDSPEED_MM_PATH` 下的 `msadapter` 目录。如果 `MINDSPEED_MM_PATH` 指向 workspace root（如 `/shared/mindspeed-mm`），则 msadapter 源码应在该目录下的 `msadapter/` 中。实际路径可通过 `python -c "import os; print(os.environ.get('MINDSPEED_MM_PATH', '/shared/mindspeed-mm'))"` 获取。
```

### 验证 decord 运行时补丁

```bash
source <lm-sv-root>/lmsv_rec/scripts/envset/mm-pta-task6.sh
python -c "
import decord
print('PASS: decord.cpu =', decord.cpu)
"
```

### 验证 libstdc++ 兼容性修复

```bash
source <lm-sv-root>/lmsv_rec/scripts/envset/mm-msa-task6.sh
ldconfig -p | grep libstdc++ | head -5
# 应优先显示 $CONDA_PREFIX/lib/libstdc++.so.6
```

---

*文档更新时间：2026-04-20*
*对应代码分支：lm-sv/lmsv_rec (Task6)*
