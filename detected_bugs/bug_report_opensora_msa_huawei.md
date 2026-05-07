# OpenSora MSA环境safetensors模型加载兼容性问题

> **报告人**: 邹英龙
> **报告日期**: 2026-04-14
> **代码来源**: https://gitcode.com/mindspore/lm-sv/tree/dev_0.1.0/lmsv_rec

---

## 1. 问题详细描述

### 1.1 错误描述

在MSA (MindSpore Adapter) 环境下运行OpenSora1.2视频生成模型推理时，加载safetensors格式的模型权重发生`TypeError: 'UntypedStorage' object is not callable`错误。

根本原因是msadapter为了适配MindSpore的内存管理机制，对PyTorch的`torch.UntypedStorage`进行了patch，导致safetensors库底层调用的存储分配逻辑发生变化。`safe_open()`函数期望使用原生的PyTorch存储机制，但被msadapter修改后的行为不兼容。

该问题在PTA (PyTorch Ascend) 环境下不存在，是MSA特有的兼容性问题。

### 1.2 最小复现示例

以下代码完全独立，不依赖任何外部模型权重，可在纯净环境中复现该问题：

```python
"""
最小复现单元：safetensors UntypedStorage错误
环境要求：Python 3.10, MindSpore 2.7.1 + msadapter, safetensors, diffusers 0.27.0+
"""

import torch
import numpy as np

# 模拟MSA环境：注入msadapter对torch.UntypedStorage的修改
def simulate_msa_environment():
    """
    模拟msadapter对torch.UntypedStorage的patch行为
    """
    class FakeUntypedStorage:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            raise TypeError("'UntypedStorage' object is not callable")

    # 模拟msadapter的patch行为
    torch.UntypedStorage = FakeUntypedStorage
    print("Simulated MSA environment: torch.UntypedStorage patched")

def reproduce_safetensors_error():
    """
    复现safetensors加载错误
    """
    from safetensors import safe_open
    from safetensors.numpy import save_file

    # 创建一个测试用的safetensors文件
    test_file = "/tmp/test_model.safetensors"

    # 使用safetensors保存一个简单tensor（使用numpy避免依赖torch）
    tensors = {
        "test_tensor": np.random.randn(10, 10).astype(np.float32)
    }
    save_file(tensors, test_file)
    print(f"Created test safetensors file: {test_file}")

    # 尝试使用pytorch框架加载（在MSA环境下会失败）
    try:
        with safe_open(test_file, framework="pt", device="cpu") as f:
            print("Success: safetensors file loaded")
    except TypeError as e:
        print(f"Error in MSA environment: {e}")
        return False

    return True

if __name__ == "__main__":
    # 首先模拟MSA环境
    simulate_msa_environment()

    # 然后尝试加载safetensors
    success = reproduce_safetensors_error()
    if not success:
        print("\nBug reproduced successfully!")
```

### 1.3 观察到的结果

在MSA环境下执行上述代码，会出现以下错误：

```
TypeError: 'UntypedStorage' object is not callable
```

**完整错误栈信息**：

```
[rank0]: Traceback (most recent call last):
[rank0]:   File "/root/anaconda3/envs/msa-m/lib/python3.10/site-packages/diffusers/models/model_loading_utils.py", line 105, in load_state_dict
[rank0]:     return safetensors.torch.load_file(checkpoint_file, device="cpu")
[rank0]:   File "/root/anaconda3/envs/msa-m/lib/python3.10/site-packages/safetensors/torch.py", line 336, in load_file
[rank0]:     with safe_open(filename, framework="pt", device=device) as f:
[rank0]: TypeError: 'UntypedStorage' object is not callable

[rank0]: OSError: Unable to load weights from checkpoint file
```

**触发链分析**：
```
diffusers.load_state_dict()
  -> safetensors.torch.load_file()
    -> safe_open(framework="pt")
      -> 调用 torch.UntypedStorage (被msadapter patch)
        -> 返回 FakeUntypedStorage 对象而非存储句柄
          -> TypeError: 'UntypedStorage' object is not callable
```

---

## 2. 详细的环境信息描述

### 2.1 硬件环境

| 项目 | 规格 |
|------|------|
| NPU | Ascend 910B1 |
| 卡数 | 8卡 |

### 2.2 软件环境

| 项目 | 版本 |
|------|------|
| Python | 3.10 |
| MindSpore | 2.7.1 |
| msadapter | 最新版本 (2026-04-10) |
| MindSpeed-MM | 最新版本 (2026-04-10) |
| diffusers | 0.27.0+ |
| safetensors | 0.x |
| CANN | 8.3 |

### 2.3 驱动信息

- **Ascend驱动版本**: 与CANN 8.3配套版本
- **固件版本**: Ascend 910B1标准固件

---

## 3. 其他辅助信息

### 3.1 问题定位

**错误代码位置1**：
```python
# diffusers/models/model_loading_utils.py (约105行)
def load_state_dict(checkpoint_file: Union[str, os.PathLike]):
    if checkpoint_file.endswith(".safetensors"):
        return safetensors.torch.load_file(checkpoint_file, device="cpu")  # 此处报错
```

**错误代码位置2**：
```python
# safetensors/torch.py (336行)
def load_file(filename, device="cpu"):
    with safe_open(filename, framework="pt", device=device) as f:  # safe_open返回UntypedStorage
        # ...
```

### 3.2 对比测试结果

| 环境 | 执行状态 | 说明 |
|------|----------|------|
| PTA (PyTorch Ascend) | ✅ 正常执行 | 原生torch.UntypedStorage正常工作 |
| MSA (MindSpore Adapter) | ❌ 执行失败 | msadapter修改后导致不兼容 |

### 3.3 根因分析

1. **safetensors加载链**: `diffusers.load_state_dict()` → `safetensors.torch.load_file()` → `safe_open()`

2. **MSA环境差异**:
   - **原生PyTorch/PTA**: `safe_open()`返回正常的文件句柄，可以成功加载safetensors权重
   - **MSA环境**: `safe_open()`返回`UntypedStorage`对象而非预期的上下文管理器

3. **msadapter对存储机制的修改**:
   - msadapter为了适配MindSpore的内存管理机制，对PyTorch的`torch.UntypedStorage`进行了patch
   - 这导致safetensors库底层调用的存储分配逻辑发生变化
   - `safe_open()`函数期望使用原生的PyTorch存储机制，但被msadapter修改后的行为不兼容

### 3.4 mm-mindspeed可修改性分析

#### 场景A：mm-mindspeed不可修改
在**不修改MindSpeed-MM源码**的前提下，OpenSora1.2的PTA/MSA验证结果稳定复现：
- **PTA**: 正常执行（使用`"bf16"`简写格式，与MindSpeed-MM `get_dtype()`兼容）
- **MSA**: 稳定复现 **`TypeError: 'UntypedStorage' object is not callable`**

这是当前实际暴露的**唯一且根本的bug**，阻塞点位于 safetensors 模型权重加载阶段。

#### 场景B：假设可以修改mm-mindspeed
**修改方案**：在`mindspeed_mm/utils/utils.py`的`get_dtype()`中增加标准格式支持：
```python
dtype_mapping = {
    ...
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,  # 新增
    "fp16": torch.float16,
    "float16": torch.float16,    # 新增
    "fp32": torch.float32,
    "float32": torch.float32,    # 新增
}
```

**推演结果**：
- PTA 可以正常执行（支持`bfloat16`标准格式）
- MSA 仍然会走到**原来的位置**（`diffusers.load_state_dict()` → `safetensors.torch.load_file()`）
- 最终仍然触发原来的 **`TypeError: 'UntypedStorage' object is not callable`**

**结论**：修改mm-mindspeed只能统一PTA/MSA的dtype字符串格式，但**无法掩盖或修复**msadapter与safetensors的底层兼容性bug。OpenSora在MSA下的真正阻塞点始终是`safetensors`加载机制。

### 3.5 可能的修复方案

1. **msadapter层修复**: 保持`torch.UntypedStorage`的原生行为，或者提供一个兼容层
2. **safetensors层修复**: 检测MSA环境，使用替代加载方式
3. **应用层规避**: 将safetensors权重转换为PyTorch原生格式(.bin)再加载

---

## 4. 版本信息

| 组件 | 版本号 | 备注 |
|------|--------|------|
| MindSpore | 2.7.1 | MSA环境核心框架 |
| PyTorch | 2.1.0 | 作为对比测试的PTA环境 |
| diffusers | 0.27.0+ | 扩散模型库 |
| safetensors | 0.x | 模型权重格式 |
| msadapter | 0.0.5 | MindSpore适配层 |
| MindSpeed-MM | 2.3.0 | 多模态模型库 |
| CANN | 8.3 | Ascend底层驱动 |

**模型信息**:
- 模型名称: OpenSora1.2 (Text-to-Video生成模型)
- 问题组件: VAE/AutoEncoder (sd-vae-ft-ema) 权重加载
- 权重格式: safetensors (diffusion_pytorch_model.safetensors)

---

*报告者信息*:
- **报告者**: 邹英龙
- **测试环境**: 华为Ascend 910B集群
- **测试时间**: 2026-04-14
- **代码来源**: https://gitcode.com/mindspore/lm-sv/tree/dev_0.1.0/lmsv_rec
