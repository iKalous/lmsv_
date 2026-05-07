# QwenVL2.5 MSA NPU Tensor 维度限制错误

> **报告人**: 邹英龙
> **报告日期**: 2026-04-10
> **代码来源**: https://gitcode.com/mindspore/lm-sv/tree/dev_0.1.0/lmsv_rec

---

## 1. 问题详细描述

### 1.1 错误描述

在 MSA (MindSpore Adapter) 环境下运行 QwenVL2.5 视觉语言模型推理时，图像处理阶段出现 NPU tensor 维度限制错误。当处理高分辨率图像时，`Qwen2VLImageProcessor` 生成的 `image_grid_thw` tensor 维度超过 8 维，调用 `.prod()` 方法时触发 Ascend 底层算子错误。

该问题在 PTA (PyTorch Ascend) 环境下不存在，是 MSA 特有的兼容性问题。

### 1.2 最小复现示例

以下代码完全独立，不依赖任何外部模型权重或数据集，可在纯净环境中复现该问题：

```python
"""
最小复现单元：QwenVL2.5 MSA Tensor维度限制
环境要求：Python 3.10, MindSpore 2.7.1 + msadapter, transformers 4.39.0+
"""

import torch
import torch_npu

def reproduce_tensor_dim_error():
    """
    复现高维tensor在MSA环境下的维度限制错误
    """
    # 模拟QwenVL处理高分辨率图像时生成的高维grid tensor
    high_dim_tensor = torch.randn(1, 1, 1, 1, 1, 1, 1, 1, 1, 1)  # 10维tensor

    print(f"Tensor shape: {high_dim_tensor.shape}")
    print(f"Tensor ndim: {high_dim_tensor.ndim}")

    try:
        # 在MSA环境下，.prod()操作会调用aclnn算子，触发维度限制错误
        result = high_dim_tensor.prod()
        print(f"Result: {result}")
        return True
    except RuntimeError as e:
        print(f"Error in MSA environment: {e}")
        return False

if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"NPU available: {torch.npu.is_available()}")

    success = reproduce_tensor_dim_error()
    if not success:
        print("\nBug reproduced successfully!")
```

### 1.3 观察到的结果

在 MSA 环境下执行上述代码，会出现以下错误：

```
RuntimeError: aclnnInplaceCopyGetWorkspaceSize call failed, please check!
Ascend Error: AclNN_Parameter_Error(EZ1001): The self tensor cannot be larger than 8 dimensions.
```

**完整错误栈信息**：

```
[rank0]: Traceback (most recent call last):
[rank0]:   File "/root/anaconda3/envs/msa-m/lib/python3.10/site-packages/transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py", line 177, in __call__
[rank0]:     num_image_tokens = image_grid_thw[index].prod() // merge_length
[rank0]: RuntimeError: aclnnInplaceCopyGetWorkspaceSize call failed, please check!
[rank0]: Ascend Error: AclNN_Parameter_Error(EZ1001): The self tensor cannot be larger than 8 dimensions.
```

**触发链分析**：
```
Qwen2VLImageProcessor.__call__()
  -> 计算 image_grid_thw.prod()
    -> MindSpore后端调用 aclnnInplaceCopyGetWorkspaceSize
      -> Ascend算子限制：tensor维度 > 8
        -> RuntimeError
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
| msadapter | 0.0.5 |
| MindSpeed-MM | 2.3.0 |
| transformers | 4.39.0+ |
| CANN | 8.3 |

### 2.3 驱动信息

- **Ascend驱动版本**: 与 CANN 8.3 配套版本
- **固件版本**: Ascend 910B1 标准固件

---

## 3. 其他辅助信息

### 3.1 问题定位

**错误代码位置**：
```python
# transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py (约177行)
def __call__(self, images=None, text=None, ...):
    # ...
    num_image_tokens = image_grid_thw[index].prod() // merge_length  # 此处报错
```

### 3.2 对比测试结果

| 环境 | 执行状态 | 说明 |
|------|----------|------|
| PTA (PyTorch Ascend) | ✅ 正常执行 | PyTorch Ascend 算子支持高维 tensor |
| MSA (MindSpore Adapter) | ❌ 执行失败 | MindSpore/AclNN 限制 tensor 维度 ≤ 8 |

### 3.3 根因分析

1. **图像处理链**: `Qwen2VLImageProcessor.__call__()` → 计算 `image_grid_thw.prod()` → `aclnnInplaceCopyGetWorkspaceSize`

2. **MSA环境差异**:
   - **原生PyTorch/PTA**: 使用 PyTorch Ascend 算子，支持高维 tensor
   - **MSA环境**: MindSpore 后端调用 AclNN 算子，限制 tensor 维度 ≤ 8

3. **NPU算子限制**: MindSpore/AclNN 底层算子对 tensor 维度有限制（最大 8 维），而 Qwen2VLImageProcessor 在处理高分辨率图像时可能生成超过 8 维的 grid tensor。

### 3.4 可能的修复方案

1. **MindSpore 层修复**: 提升 AclNN 算子对 tensor 维度的支持上限
2. **msadapter 层修复**: 在高维 tensor 操作前进行维度压缩/拆分
3. **应用层规避**: 修改 Qwen2VLImageProcessor，避免产生高维 tensor

---

## 4. 版本信息

| 组件 | 版本号 | 备注 |
|------|--------|------|
| MindSpore | 2.7.1 | MSA环境核心框架 |
| PyTorch | 2.1.0 | 作为对比测试的PTA环境 |
| transformers | 4.39.0+ | QwenVL模型支持 |
| msadapter | 0.0.5 | MindSpore适配层 |
| MindSpeed-MM | 2.3.0 | 多模态模型库 |
| CANN | 8.3 | Ascend底层驱动 |

**模型信息**:
- 模型名称: Qwen2.5-VL-7B-Instruct
- 问题组件: Qwen2VLImageProcessor 图像处理器
- 触发位置: `image_grid_thw.prod()` 计算图像 token 数量时

---

*报告者信息*:
- **报告者**: 邹英龙
- **测试环境**: 华为 Ascend 910B 集群
- **测试时间**: 2026-04-10
- **代码来源**: https://gitcode.com/mindspore/lm-sv/tree/dev_0.1.0/lmsv_rec
