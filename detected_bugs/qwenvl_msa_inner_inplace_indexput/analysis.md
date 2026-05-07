# QwenVL2.5 MSA `InnerInplaceIndexPut` Shape Mismatch Bug

> **日期**: 2026-04-14
> **状态**: 稳定复现，新发现 bug（与历史记录不一致）

## 问题描述

在 MSA (MindSpore Adapter) 环境下运行 QwenVL2.5 推理时，`process_multimodal_embeddings` 中的 image token 替换操作 (`input_embeds[indices_tuple] = vit_embeds`) 因 shape mismatch 崩溃。PTA 环境下正常执行。

## 错误信息

```
ValueError: For 'InnerInplaceIndexPut', shape mismatch: value tensor of shape [14308, 1280] cannot be broadcast to indexing result of shape [3577, 3584].
```

## 完整堆栈

```
[rank0]:   File "/shared/mindspeed-mm/MindSpeed-MM/mindspeed_mm/models/vlm_model.py", line 548, in process_multimodal_embeddings
[rank0]:     input_embeds[indices_tuple] = vit_embeds
[rank0]: ValueError: For 'InnerInplaceIndexPut', shape mismatch: value tensor of shape [14308, 1280] cannot be broadcast to indexing result of shape [3577, 3584].
```

## 触发代码

```python
# /shared/mindspeed-mm/MindSpeed-MM/mindspeed_mm/models/vlm_model.py:539-548
def process_multimodal_embeddings(self, input_embeds, input_ids, vit_embeds, **kwargs):
    if vit_embeds is not None:
        if self.config.sequence_parallel:
            input_embeds = gather_from_sequence_parallel_region(input_embeds)
        input_embeds = input_embeds.transpose(0, 1)  # bsh -> sbh

        image_mask = torch.eq(input_ids, self.img_context_token_id)
        vit_embeds = vit_embeds[:, 0, :]
        indices_tuple = torch.nonzero(image_mask, as_tuple=True)
        input_embeds[indices_tuple] = vit_embeds   # <-- 崩溃点

        ...
```

## 根因分析

### 1. 底层算子语义差异

| 环境 | 底层算子 | shape 校验策略 |
|------|----------|----------------|
| PTA | PyTorch `index_put_` | **宽松**，支持隐式广播、view、expand |
| MSA | MindSpore `InnerInplaceIndexPut` | **严格**，要求 value shape 必须能直接广播到索引结果 shape |

MindSpore 的 `InnerInplaceIndexPut::InferShape` 在 C++ 层直接拒绝任何无法广播的维度组合。

### 2. Shape 不匹配的本质

崩溃时 shapes 对比：

| Tensor | Shape | 说明 |
|--------|-------|------|
| `vit_embeds` (value) | `[14308, 1280]` | vision encoder 输出，取第 0 帧后 |
| 索引结果 (target) | `[3577, 3584]` | `input_embeds[indices_tuple]` 的期望结果 shape |

广播校验失败：
- 第一维：`14308 = 3577 × 4`，既不相等也不为 1，无法广播
- 第二维：`1280 ≠ 3584`，无法广播

### 3. 为什么第一维是 4 倍？

`14308 ≈ 3577 × 4` 恰好等于 tensor parallelism size (`TP=4`)。这说明在 MSA 环境下，**`vit_embeds` 的 batch/sequence 维度被放大了约 4 倍**，可能的原因：

- `gather_from_sequence_parallel_region(input_embeds)` 只恢复了 `input_embeds`，而 `vit_embeds` 来自 vision encoder，**未经过同样的 gather/投影**，导致两者在 TP 维度上的切分状态不一致。
- 或者 MSA 下 `torch.nonzero(..., as_tuple=True)` 返回的索引Tuple在 TP 上下文中的行为与 PTA 不同，使得目标索引结果 shape 被错误计算为 `3577`（本地 rank 视角），而 `vit_embeds` 却持有全局/多 rank 合并后的 `14308`。

## 与历史 bug 记录的对比

| 维度 | detected_bugs 历史记录 (2026-04-10) | 本次实际结果 (2026-04-14) |
|------|--------------------------------------|---------------------------|
| 预期错误 | `RuntimeError` (>8 tensor dims) | **未复现** |
| 实际错误 | — | `ValueError: InnerInplaceIndexPut shape mismatch` |
| 错误位置 | AclNN 后端维度限制 | `vlm_model.py:548` (multimodal embedding 替换) |
| 触发阶段 | 推测为 reshape/view | 推理中的 image token 注入阶段 |
| 一致性 | — | **不一致** — 新 bug 提前拦截了历史 bug 的触发路径 |

## 环境信息

- **模型**: QwenVL2.5 (推理模式)
- **PTA环境**: mindspeed (正常)
- **MSA环境**: msa-m (崩溃)
- **NPU**: 8 卡 Ascend 910B1
- **Tensor Parallelism**: TP=4
- **MindSpeed-MM 状态**: 已确认与 upstream `2.3.0` 完全一致，**无任何本地修改**

## 复现步骤

```bash
cd /shared/lm-sv/lmsv_rec
# config.json: tasks["6"].MODEL_NAME = "qwenvl"
python do.py --task 6
```

## 验证结果

| 环境 | 状态 | 指标 |
|------|------|------|
| PTA | 成功 | memory=18320.92MB, time=72679.0ms |
| MSA | 失败 | `ValueError: For 'InnerInplaceIndexPut', shape mismatch...` |

## 影响评估

- **严重程度**: 高 — 阻止 MSA 环境下 QwenVL2.5 的完整推理
- **影响范围**: QwenVL2.5 (确认)；其他使用 `input_embeds[indices] = vit_embeds` 模式的多模态模型可能同样受影响

## 修复方向建议

### 方案 1: 在 msadapter 层放宽 `InnerInplaceIndexPut` 的 shape 校验
在 msadapter 的 `__setitem__` 实现中，检测到 shape mismatch 时，先对 `value` 做 `view`/`expand` 预处理，使其满足 MindSpore 算子的广播要求。

### 方案 2: 在 MindSpeed-MM 中显式对齐 shape（不可行，因约束为不修改 mm-mindspeed）
在赋值前增加：
```python
vit_embeds = vit_embeds.view(-1, input_embeds.shape[-1])
```
但当前任务约束为**不能修改 MindSpeed-MM**。

### 方案 3: msadapter 层修复 TP/SP gather 行为
确保 `vit_embeds` 在多模态 embedding 融合前已经过与 `input_embeds` 一致的 gather/切分恢复，消除 4 倍维度差异。

## 结论

QwenVL2.5 在 MSA 下的实际崩溃点是 **MindSpore `InnerInplaceIndexPut` 的严格 shape 广播规则** 与 PyTorch 语义不兼容，叠加 MSA 环境下 TP=4 导致的 `vit_embeds` 维度放大问题。该 bug 稳定复现，且**提前掩盖了**历史上记录的 `>8 tensor dims` AclNN 错误。
