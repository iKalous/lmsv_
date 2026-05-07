# QwenVL2.5 MSA `InnerInplaceIndexPut` Shape Mismatch Bug

> **报告人**: 邹英龙
> **报告日期**: 2026-04-14
> **代码来源**: https://gitcode.com/mindspore/lm-sv/tree/dev_0.1.0/lmsv_rec

---

## 1. 问题详细描述

### 1.1 错误描述

在 MSA (MindSpore Adapter) 环境下运行 QwenVL2.5 推理时，`process_multimodal_embeddings` 中的 image token 替换操作 (`input_embeds[indices_tuple] = vit_embeds`) 因 shape mismatch 崩溃。PTA 环境下正常执行。

### 1.2 最小复现示例

以下代码展示了核心崩溃点：

```python
import torch

# 模拟 MSA 环境下的 shape 不匹配
value = torch.randn(14308, 1280)   # vit_embeds 形状
target = torch.randn(3577, 3584)   # input_embeds[indices] 期望形状

# MSA 底层使用 InnerInplaceIndexPut，广播规则严格
target[:] = value  # ValueError: shape mismatch...
```

### 1.3 观察到的结果

```
ValueError: For 'InnerInplaceIndexPut', shape mismatch: value tensor of shape [14308, 1280] cannot be broadcast to indexing result of shape [3577, 3584].
```

**完整错误栈信息**：

```
[rank0]:   File "/shared/mindspeed-mm/MindSpeed-MM/mindspeed_mm/models/vlm_model.py", line 548, in process_multimodal_embeddings
[rank0]:     input_embeds[indices_tuple] = vit_embeds
[rank0]: ValueError: For 'InnerInplaceIndexPut', shape mismatch: value tensor of shape [14308, 1280] cannot be broadcast to indexing result of shape [3577, 3584].
```

---

## 2. 根因分析

### 2.1 底层算子语义差异

| 环境 | 底层算子 | shape 校验策略 |
|------|----------|----------------|
| PTA | PyTorch `index_put_` | **宽松**，支持隐式广播、view、expand |
| MSA | MindSpore `InnerInplaceIndexPut` | **严格**，要求 value shape 必须能直接广播到索引结果 shape |

### 2.2 Shape 不匹配的本质

崩溃时 shapes 对比：

| Tensor | Shape | 说明 |
|--------|-------|------|
| `vit_embeds` (value) | `[14308, 1280]` | vision encoder 输出，取第 0 帧后 |
| 索引结果 (target) | `[3577, 3584]` | `input_embeds[indices_tuple]` 的期望结果 shape |

广播校验失败：
- 第一维：`14308 = 3577 × 4`，既不相等也不为 1，无法广播
- 第二维：`1280 ≠ 3584`，无法广播

### 2.3 TP=4 导致的维度放大

`14308 ≈ 3577 × 4` 恰好等于 tensor parallelism size (`TP=4`)。这说明在 MSA 环境下，`vit_embeds` 的 batch/sequence 维度被放大了约 4 倍，可能的原因：

- `gather_from_sequence_parallel_region(input_embeds)` 只恢复了 `input_embeds`，而 `vit_embeds` 来自 vision encoder，未经过同样的 gather/投影，导致两者在 TP 维度上的切分状态不一致。
- 或者 MSA 下 `torch.nonzero(..., as_tuple=True)` 返回的索引 Tuple 在 TP 上下文中的行为与 PTA 不同。

---

## 3. 对比测试结果

| 环境 | 执行状态 | 说明 |
|------|----------|------|
| PTA (PyTorch Ascend) | ✅ 正常执行 | memory=18320.92MB, time=72679.0ms |
| MSA (MindSpore Adapter) | ❌ 执行失败 | `ValueError: InnerInplaceIndexPut shape mismatch` |

---

## 4. 可能的修复方案

1. **msadapter 层修复**：在 `__setitem__` 实现中，检测到 shape mismatch 时先对 `value` 做 view/expand 预处理。
2. **msadapter 层修复 TP/SP gather 行为**：确保 `vit_embeds` 在多模态 embedding 融合前已经过与 `input_embeds` 一致的 gather/切分恢复。

---

## 5. 版本信息

| 组件 | 版本号 | 备注 |
|------|--------|------|
| MindSpore | 2.7.1 | MSA环境核心框架 |
| PyTorch | 2.1.0 | 作为对比测试的PTA环境 |
| transformers | 4.51.0 | QwenVL模型支持 |
| msadapter | 0.0.5 | MindSpore适配层 |
| MindSpeed-MM | 2.3.0 | 多模态模型库 |
| CANN | 8.3 | Ascend底层驱动 |

**模型信息**:
- 模型名称: Qwen2.5-VL-7B-Instruct
- 问题组件: `vlm_model.py:548` (multimodal embedding 替换)
- 并行配置: TP=4

---

*报告者信息*:
- **报告者**: 邹英龙
- **测试环境**: 华为Ascend 910B集群
- **测试时间**: 2026-04-14
- **代码来源**: https://gitcode.com/mindspore/lm-sv/tree/dev_0.1.0/lmsv_rec
